//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::gfx10::{self, Encoding};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, sources};

const ALL_STAGES: [HwStage; 8] = [
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Cs,
    HwStage::Ms,
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
];

#[derive(Default, Clone)]
struct StageStats {
    total: usize,
    ir_ok: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    unsupported: usize,
    malformed: usize,
    isa: usize,
    stage_unemitted: usize,
    validator_failed: usize,
    pipeline_failed: usize,
}

impl StageStats {
    fn merge(&mut self, other: &StageStats) {
        self.total += other.total;
        self.ir_ok += other.ir_ok;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.unsupported += other.unsupported;
        self.malformed += other.malformed;
        self.isa += other.isa;
        self.stage_unemitted += other.stage_unemitted;
        self.validator_failed += other.validator_failed;
        self.pipeline_failed += other.pipeline_failed;
    }
}

#[derive(Default, Clone)]
struct CohortStats {
    total: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    failed: usize,
    stage_unemitted: usize,
    by_stage: BTreeMap<HwStage, usize>,
    reasons: BTreeMap<String, usize>,
    examples: Vec<String>,
}

impl CohortStats {
    fn note_example(&mut self, example: String) {
        if self.examples.len() < 8 && !self.examples.contains(&example) {
            self.examples.push(example);
        }
    }

    fn merge(&mut self, other: CohortStats) {
        self.total += other.total;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.failed += other.failed;
        self.stage_unemitted += other.stage_unemitted;
        for (k, n) in other.by_stage {
            *self.by_stage.entry(k).or_default() += n;
        }
        for (k, n) in other.reasons {
            *self.reasons.entry(k).or_default() += n;
        }
        self.examples.extend(other.examples);
        self.examples.sort();
        self.examples.dedup();
        self.examples.truncate(8);
    }
}

#[derive(Default, Clone)]
struct FailEntry {
    count: usize,
    kind: &'static str,
    stages: BTreeMap<String, usize>,
    examples: Vec<String>,
}

impl FailEntry {
    fn merge(&mut self, other: FailEntry) {
        self.count += other.count;
        if self.kind.is_empty() {
            self.kind = other.kind;
        }
        for (s, n) in other.stages {
            *self.stages.entry(s).or_default() += n;
        }
        self.examples.extend(other.examples);
        self.examples.sort();
        self.examples.dedup();
        self.examples.truncate(5);
    }
}

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    translated: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Native mesh layout counts, before resource tracking or lowering.
    mesh_layouts: BTreeMap<(u32, u32, u32, u8), usize>,
    /// Detailed per-stage breakdown across all 8 GFX10 hardware stages.
    by_stage: BTreeMap<HwStage, StageStats>,
    /// Feature cohorts: Wave64, LDS, image_bvh*_intersect_ray, merged stages.
    wave64: CohortStats,
    lds: CohortStats,
    bvh_ray: CohortStats,
    merged_stages: CohortStats,
    merged_containers: usize,
    /// Reason → programs stopped by it, error kind, per-stage counts, examples.
    failed: BTreeMap<String, FailEntry>,
    /// With `--list`: one line per program and outcome.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail_with(
        &mut self,
        kind: &'static str,
        stage: Option<HwStage>,
        reason: String,
        example: impl FnOnce() -> String,
    ) {
        let e = example();
        if self.list {
            self.lines.push(format!("{e}  {reason}"));
        }
        let entry = self.failed.entry(reason).or_insert_with(|| FailEntry {
            count: 0,
            kind,
            stages: BTreeMap::new(),
            examples: Vec::new(),
        });
        entry.count += 1;
        if let Some(st) = stage {
            *entry.stages.entry(format!("{st:?}")).or_default() += 1;
        }
        if entry.examples.len() < 5 && !entry.examples.contains(&e) {
            entry.examples.push(e);
        }
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.translated += other.translated;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        for (layout, n) in other.mesh_layouts {
            *self.mesh_layouts.entry(layout).or_default() += n;
        }
        for (st, stats) in other.by_stage {
            self.by_stage.entry(st).or_default().merge(&stats);
        }
        self.wave64.merge(other.wave64);
        self.lds.merge(other.lds);
        self.bvh_ray.merge(other.bvh_ray);
        self.merged_stages.merge(other.merged_stages);
        self.merged_containers += other.merged_containers;
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, entry) in other.failed {
            self.failed.entry(reason).or_default().merge(entry);
        }
    }
}

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    if pipelines {
        Host::new(false).map_err(anyhow::Error::msg)?;
    }
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        let mut t = Tally {
                            list,
                            ..Tally::default()
                        };
                        let mut checks = Checks {
                            keep: (!pairs.is_empty()).then(|| kept.clone()),
                            validator: validate
                                .then(|| Validator::load(dll.as_deref()).ok())
                                .flatten(),
                            host: pipelines
                                .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                .flatten(),
                        };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t, &mut checks);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "programs     {} translated of {} ({} IR instructions)",
        t.translated, t.programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.translated,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.translated, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    println!("\nstage breakdown:");
    println!(
        "  {:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>11} {:>9} {:>5} {:>10} {:>7} {:>7}",
        "stg", "total", "ir_ok", "emit", "validated", "pipelines", "unsupported", "malformed", "isa", "unemitted", "val_err", "pso_err"
    );
    for st in ALL_STAGES {
        let s = t.by_stage.get(&st).cloned().unwrap_or_default();
        println!(
            "  {:<4} {:>7} {:>7} {:>7} {:>9} {:>9} {:>11} {:>9} {:>5} {:>10} {:>7} {:>7}",
            format!("{st:?}"),
            s.total,
            s.ir_ok,
            s.emitted,
            s.validated,
            s.pipelines,
            s.unsupported,
            s.malformed,
            s.isa,
            s.stage_unemitted,
            s.validator_failed,
            s.pipeline_failed
        );
    }
    if !t.mesh_layouts.is_empty() {
        println!("\nnative mesh layouts (before lowering):");
        for ((threads, vertices, primitives, topology), n) in t.mesh_layouts {
            println!(
                "  [{threads}, 1, 1] vertices={vertices} primitives={primitives} topology={topology}: {n}"
            );
        }
    }
    println!("\nfeature cohorts:");
    for (name, c) in [
        ("wave64", &t.wave64),
        ("lds", &t.lds),
        ("image_bvh*_intersect_ray", &t.bvh_ray),
        ("merged_stages", &t.merged_stages),
    ] {
        let stages_str: Vec<String> = c
            .by_stage
            .iter()
            .map(|(st, n)| format!("{st:?}={n}"))
            .collect();
        println!(
            "  {name:<26} total={} emitted={} validated={} pipelines={} failed={} stage_unemitted={} stages=[{}]",
            c.total,
            c.emitted,
            c.validated,
            c.pipelines,
            c.failed,
            c.stage_unemitted,
            stages_str.join(", ")
        );
        if name == "merged_stages" {
            println!("    multi-program containers: {}", t.merged_containers);
        }
        let mut reasons: Vec<_> = c.reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (r, n) in reasons.iter().take(12) {
            println!("    fail {:>5}  {r}", n);
        }
        if !c.examples.is_empty() {
            println!("    examples: {}", c.examples.join(" ; "));
        }
    }
    println!("\nfailures by reason:");
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(&b.0)));
    for (reason, entry) in failed {
        let st: Vec<String> = entry
            .stages
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        let ex = entry.examples.join(" ; ");
        println!(
            "  {:>7}  [{:<11}]  {reason}  stages=[{}]  (e.g. {ex})",
            entry.count,
            entry.kind,
            st.join(",")
        );
    }
    Ok(())
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail_with("Malformed", None, "corrupt record".into(), example);
    };
    // The hash a run's log names the shader by (`hash=` in "first bind",
    // "shader does not translate", "host pipeline refused").
    let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
    let example = || format!("{}{} #{hash}", path.display(), source.label());
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail_with("Container", None, "no DXIL part (SM5 or older)".into(), example);
    };
    let Some(dxil) = container.part(FourCC::DXIL).map(|p| p.data) else {
        return t.fail_with("Container", None, "no DXIL part (SM5 or older)".into(), example);
    };
    // `~`: the hash of the code alone; containers that differ only around
    // it are translated once, the others listed as duplicates of it.
    let code = blake3::hash(dxil);
    let code8 = code.to_hex()[..8].to_owned();
    let example = || format!("{} ~{code8}", example());
    // `VICTORY_DUMP_MATCH=<dir>;<text>|<text>...`: the containers whose
    // line holds one of the texts (a name, `#hash`), as the title has them.
    if let Ok(spec) = std::env::var("VICTORY_DUMP_MATCH")
        && let Some((dir, texts)) = spec.split_once(';')
    {
        let line = example();
        if texts.split('|').any(|text| !text.is_empty() && line.contains(text)) {
            let name: String = source
                .label()
                .chars()
                .skip_while(|c| *c != ' ')
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .take(70)
                .collect();
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(std::path::Path::new(dir).join(format!("{name}.{hash}.cdxil")), &bytes[..]);
        }
    }
    if !lock(seen).insert(code) {
        t.duplicates += 1;
        if t.list {
            t.lines.push(format!("{}  duplicate", example()));
        }
        return;
    }
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => {
            let kind = error_kind(&e);
            return t.fail_with(kind, None, e.to_string(), example);
        }
    };
    // `VICTORY_FAIL_DUMP=<dir>`: the first 3 containers of each failure reason, as the title
    // has them (`.cdxil`), to work on a translator gap offline
    let dump = std::env::var_os("VICTORY_FAIL_DUMP");
    let before: Option<BTreeMap<String, usize>> = dump
        .as_ref()
        .map(|_| t.failed.iter().map(|(k, v)| (k.clone(), v.count)).collect());
    translate_programs(&container, programs, &example, t, checks);
    if let (Some(dir), Some(before)) = (dump, before) {
        for (reason, entry) in &t.failed {
            let was = before.get(reason).copied().unwrap_or(0);
            if entry.count > was && was < 3 {
                let stem: String = reason
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                    .take(60)
                    .collect();
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(
                    std::path::Path::new(&dir).join(format!("{stem}_{was}.cdxil")),
                    &bytes[..],
                );
            }
        }
    }
}

fn error_kind(e: &ShaderError) -> &'static str {
    match e {
        ShaderError::Unsupported { .. } => "Unsupported",
        ShaderError::Malformed(_) | ShaderError::NotDxbc => "Malformed",
        ShaderError::Isa { .. } => "Isa",
    }
}

fn uses_lds(p: &NativeProgram, decoded: Option<&[gfx10::Instruction]>) -> bool {
    p.lds_bytes() > 0
        || decoded.is_some_and(|insts| {
            insts.iter().any(|i| {
                i.encoding == Encoding::Ds
                    && !i
                        .name()
                        .is_some_and(|n| n.starts_with("ds_swizzle") || n.contains("permute"))
            })
        })
}

fn uses_bvh_ray(decoded: Option<&[gfx10::Instruction]>) -> bool {
    decoded.is_some_and(|insts| {
        insts
            .iter()
            .any(|i| i.name().is_some_and(|n| n.starts_with("image_bvh")))
    })
}

fn translate_programs(
    container: &Container,
    programs: Vec<NativeProgram>,
    example: &dyn Fn() -> String,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let multi_program = programs.len() > 1;
    if multi_program {
        t.merged_containers += 1;
    }
    for p in programs {
        if let victory_shader::native::StageIo::Ms(ms) = &p.io {
            let layout = (ms.threads, ms.max_vertices, ms.max_primitives, ms.topology);
            *t.mesh_layouts.entry(layout).or_default() += 1;
        }
        t.programs += 1;
        t.by_stage.entry(p.stage).or_default().total += 1;
        let decoded = gfx10::decode(&p.code).ok();
        let is_wave64 = p.wave64;
        let is_lds = uses_lds(&p, decoded.as_deref());
        let is_bvh = uses_bvh_ray(decoded.as_deref());
        let is_merged =
            multi_program || matches!(p.stage, HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs);
        let mut primary_err: Option<(&'static str, String)> = None;
        let mut emitted = false;
        let mut validated = false;
        let mut pipelined = false;
        let mut stage_unemitted = false;
        match victory_shader::spill::build(&p) {
            Ok(program) => {
                t.translated += 1;
                t.by_stage.entry(p.stage).or_default().ir_ok += 1;
                t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                // Per-lane descriptors are traced on the per-lane form; its
                // failure is listed after the resources' and I/O's.
                let structured = structure::structure(&program);
                let lanes = structured.as_ref().ok();
                let res = match resource::track(&program, &p.layout, p.stage, lanes) {
                    Ok(r) => {
                        tally_resources(&p, &r, t);
                        Some(r)
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        let reason = format!("resources: {what}");
                        primary_err.get_or_insert(("Unsupported", reason.clone()));
                        t.fail_with("Unsupported", Some(p.stage), reason, || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        });
                        None
                    }
                    Err(e) => {
                        let kind = error_kind(&e);
                        let reason = e.to_string();
                        primary_err.get_or_insert((kind, reason.clone()));
                        t.fail_with(kind, Some(p.stage), reason, || {
                            format!("{} {:?}", example(), p.stage)
                        });
                        None
                    }
                };
                let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
                let compute = p.stage == HwStage::Cs;
                let mut raised = None;
                if graphics || compute || p.stage == HwStage::Ms {
                    t.io_programs += usize::from(graphics);
                    match io::raise(&program, &p) {
                        Ok(r) => {
                            if graphics {
                                tally_io(&r, t);
                            }
                            raised = Some(r);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => {
                            let reason = format!("I/O: {what}");
                            primary_err.get_or_insert(("Unsupported", reason.clone()));
                            t.fail_with("Unsupported", Some(p.stage), reason, || {
                                format!("{} {:?} at {offset:#x}", example(), p.stage)
                            });
                        }
                        Err(e) => {
                            let kind = error_kind(&e);
                            let reason = e.to_string();
                            primary_err.get_or_insert((kind, reason.clone()));
                            t.fail_with(kind, Some(p.stage), reason, || {
                                format!("{} {:?}", example(), p.stage)
                            });
                        }
                    }
                }
                let structured = match structured {
                    Ok(s) => {
                        tally_structured(&program, &s, t);
                        Some(s)
                    }
                    Err(e) => {
                        let kind = error_kind(&e);
                        let reason = match e {
                            ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                            ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                            e => e.to_string(),
                        };
                        primary_err.get_or_insert((kind, reason.clone()));
                        t.fail_with(kind, Some(p.stage), reason, || {
                            format!("{} {:?}", example(), p.stage)
                        });
                        None
                    }
                };
                if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                    let label = || format!("{} {:?}", example(), p.stage);
                    *counter(t, p.stage, 0) += 1;
                    match emit::emit(container, &p, &program, &s, &i, &r) {
                        Ok(shader) => {
                            *counter(t, p.stage, 1) += 1;
                            t.by_stage.entry(p.stage).or_default().emitted += 1;
                            emitted = true;
                            let (val_ok, pso_ok) = check(shader, t, checks, label);
                            validated = val_ok;
                            pipelined = pso_ok;
                        }
                        Err(ShaderError::Unsupported { offset, what }) => {
                            let reason = format!("emit: {what}");
                            primary_err.get_or_insert(("Unsupported", reason.clone()));
                            t.fail_with("Unsupported", Some(p.stage), reason, || {
                                format!("{} at {offset:#x}", label())
                            });
                        }
                        Err(e) => {
                            let kind = error_kind(&e);
                            let reason = format!("emit: {e}");
                            primary_err.get_or_insert((kind, reason.clone()));
                            t.fail_with(kind, Some(p.stage), reason, label);
                        }
                    }
                } else if !matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)
                    && primary_err.is_none()
                {
                    stage_unemitted = true;
                    t.by_stage.entry(p.stage).or_default().stage_unemitted += 1;
                }
            }
            Err(e) => {
                let kind = error_kind(&e);
                let reason = match e {
                    ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                    e => e.to_string(),
                };
                primary_err = Some((kind, reason.clone()));
                t.fail_with(kind, Some(p.stage), reason, || {
                    format!("{} {:?}", example(), p.stage)
                });
            }
        }
        if let Some((kind, _)) = &primary_err {
            let st = t.by_stage.entry(p.stage).or_default();
            match *kind {
                "Unsupported" => st.unsupported += 1,
                "Malformed" => st.malformed += 1,
                "Isa" => st.isa += 1,
                _ => st.unsupported += 1,
            }
        }
        let note = |c: &mut CohortStats| {
            c.total += 1;
            *c.by_stage.entry(p.stage).or_default() += 1;
            if emitted {
                c.emitted += 1;
                if validated {
                    c.validated += 1;
                }
                if pipelined {
                    c.pipelines += 1;
                }
            } else if stage_unemitted {
                c.stage_unemitted += 1;
                c.note_example(format!("{} {:?}", example(), p.stage));
            } else if let Some((_, reason)) = &primary_err {
                c.failed += 1;
                *c.reasons.entry(reason.clone()).or_default() += 1;
                c.note_example(format!("{} {:?} ({reason})", example(), p.stage));
            }
        };
        if is_wave64 {
            note(&mut t.wave64);
        }
        if is_lds {
            note(&mut t.lds);
        }
        if is_bvh {
            note(&mut t.bvh_ray);
        }
        if is_merged {
            note(&mut t.merged_stages);
        }
    }
}

/// Validates an emitted shader if asked to; returns `(validated, pipelined)`.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) -> (bool, bool) {
    let mut val_ok = false;
    let mut pso_ok = false;
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                t.by_stage.entry(shader.stage).or_default().pipelines += 1;
                pso_ok = true;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => {
                t.by_stage.entry(shader.stage).or_default().pipeline_failed += 1;
                t.fail_with("Pipeline", Some(shader.stage), format!("pipeline: {e}"), &label);
            }
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return (val_ok, pso_ok);
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counter(t, shader.stage, 2) += 1;
            t.by_stage.entry(shader.stage).or_default().validated += 1;
            val_ok = true;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.count);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.by_stage.entry(shader.stage).or_default().validator_failed += 1;
            t.fail_with("Validator", Some(shader.stage), format!("validator: {first}"), label);
        }
    }
    (val_ok, pso_ok)
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
            io::Interp::Weighted { .. } => t.interpolated += 1,
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
