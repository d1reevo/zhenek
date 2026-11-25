//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    AddF16,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    MinF16,
    MaxF16,
    Or,
    And,
    Xor,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // the low dword of a 64-bit sum (`v_add_co_ci_u32` carries into
            // the high one) is the 32-bit sum of the low dwords
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_add_f16" => Op::AddF16,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_min_f16" => Op::MinF16,
            "v_max_f16" => Op::MaxF16,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::AddF16 => (k & 0xFFFF) == 0 || (k & 0xFFFF) == 0x8000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
            Op::MinF16 => (k & 0xFFFF) == 0x7C00,
            Op::MaxF16 => (k & 0xFFFF) == 0xFC00,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::AddF16 | Op::Xor)
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
    /// One 32-bit half (`hi`: upper 32 bits) of a 64-bit `f64` sum over
    /// `lanes` of `mask ? (src_lo, src_hi) : 0.0`.
    Red64 {
        src_lo: Value,
        src_hi: Value,
        mask: Value,
        lanes: u64,
        hi: bool,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// 64-bit `f64` sum over every lane (`hi`: upper 32 bits when true).
    Active64(Value, Value, Value, bool),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        });
        let Some(end) = end else {
            break;
        };
        let shapes = simulate(&insts[i + 1..end], wave, saved);
        out.insert(i, Section { end, shapes });
        i = end + 1;
    }
    out
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(x), Sym::K(y)) => match op {
            Op::AddU => Sym::K(x.wrapping_add(y)),
            Op::MinU => Sym::K(x.min(y)),
            Op::MaxU => Sym::K(x.max(y)),
            Op::MinI => Sym::K((x as i32).min(y as i32) as u32),
            Op::MaxI => Sym::K((x as i32).max(y as i32) as u32),
            Op::Or => Sym::K(x | y),
            Op::And => Sym::K(x & y),
            Op::Xor => Sym::K(x ^ y),
            _ if op.identity(x) => Sym::K(y),
            _ if op.identity(y) => Sym::K(x),
            _ => Sym::Unknown,
        },
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

fn is_zero_f64(lo: Sym, hi: Sym) -> bool {
    matches!((lo, hi), (Sym::K(0), Sym::K(0 | 0x8000_0000)))
}

fn as_term_f64(lo: Sym, hi: Sym) -> Option<(Value, Value, Value, u64)> {
    match (lo, hi) {
        (
            Sym::Red {
                op: None,
                src: src_lo,
                mask: m0,
                ident: 0,
                lanes: l0,
            },
            Sym::Red {
                op: None,
                src: src_hi,
                mask: m1,
                ident: 0,
                lanes: l1,
            },
        ) if m0 == m1 && l0 == l1 => Some((src_lo, src_hi, m0, l0)),
        (
            Sym::Red64 {
                src_lo,
                src_hi,
                mask,
                lanes,
                hi: false,
            },
            Sym::Red64 {
                src_lo: s2,
                src_hi: s3,
                mask: m2,
                lanes: l2,
                hi: true,
            },
        ) if (src_lo, src_hi, mask, lanes) == (s2, s3, m2, l2) => Some((src_lo, src_hi, mask, lanes)),
        _ => None,
    }
}

fn combine_f64(a: (Sym, Sym), b: (Sym, Sym)) -> (Sym, Sym) {
    if is_zero_f64(a.0, a.1) {
        return b;
    }
    if is_zero_f64(b.0, b.1) {
        return a;
    }
    match (as_term_f64(a.0, a.1), as_term_f64(b.0, b.1)) {
        (Some((slo, shi, m, la)), Some((slo2, shi2, m2, lb)))
            if (slo, shi, m) == (slo2, shi2, m2) && la & lb == 0 =>
        {
            let lanes = la | lb;
            (
                Sym::Red64 {
                    src_lo: slo,
                    src_hi: shi,
                    mask: m,
                    lanes,
                    hi: false,
                },
                Sym::Red64 {
                    src_lo: slo,
                    src_hi: shi,
                    mask: m,
                    lanes,
                    hi: true,
                },
            )
        }
        _ => (Sym::Unknown, Sym::Unknown),
    }
}

fn simulate(insts: &[Inst], wave: u32, saved: Value) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten();
            let Some(k) = lo else {
                return shapes(vec, uni, wave);
            };
            exec = (if name == "s_mov_b64" {
                k
            } else {
                (exec & !0xFFFF_FFFF) | (k & 0xFFFF_FFFF)
            }) & all;
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten()
            else {
                return shapes(vec, uni, wave);
            };
            exec = ((exec & 0xFFFF_FFFF) | ((k & 0xFFFF_FFFF) << 32)) & all;
            continue;
        }
        if let Fields::Sop { dst, src, .. } = &inst.fields {
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            }
            // Propagate scalar copies (`s_mov_b32` / `s_mov_b64`) so reduced
            // values copied to another SGPR before restoring EXEC keep their shape.
            if matches!(name, "s_mov_b32" | "s_mov_b64")
                && let Some(Operand::Scalar { code: dcode, .. }) = *dst
                && dcode != NULL
                && let Some(op0) = src.first()
            {
                let wide = name == "s_mov_b64";
                match *op0 {
                    Operand::Scalar { code: scode, .. } if scode != NULL => {
                        if let (Some(w), Some(r)) = (inst.write(Reg::S(dcode)), inst.read(Reg::S(scode))) {
                            let s = uni.get(&r).copied().unwrap_or(Sym::Uni(r));
                            uni.insert(w, s);
                        }
                        if wide
                            && let (Some(w), Some(r)) =
                                (inst.write(Reg::S(dcode + 1)), inst.read(Reg::S(scode + 1)))
                        {
                            let s = uni.get(&r).copied().unwrap_or(Sym::Uni(r));
                            uni.insert(w, s);
                        }
                    }
                    Operand::Const(_) | Operand::Literal(_) | Operand::Scalar { code: NULL, .. } => {
                        if let Some(k) = constant(inst) {
                            if let Some(w) = inst.write(Reg::S(dcode)) {
                                uni.insert(w, Sym::K(k as u32));
                            }
                            if wide && let Some(w) = inst.write(Reg::S(dcode + 1)) {
                                uni.insert(w, Sym::K((k >> 32) as u32));
                            }
                        }
                    }
                    _ => {}
                }
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        if let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            ..
        } = &inst.fields
        {
            if name == "ds_bpermute_b32"
                && let (
                    Some(Operand::Vector { reg: dreg, .. }),
                    Some(addr_op),
                    Some(data_op),
                ) = (dst, addr, data.first())
                && let Some(dst_v) = inst.write(Reg::V(*dreg))
            {
                let old = lanes(
                    &Operand::Vector {
                        reg: *dreg,
                        dwords: 1,
                    },
                    &vec,
                    &uni,
                );
                let a = lanes(addr_op, &vec, &uni);
                let d = lanes(data_op, &vec, &uni);
                let off = u32::from(u16::from(*offset0) | (u16::from(*offset1) << 8));
                let merged = (0..wave as usize)
                    .map(|l| {
                        if exec >> l & 1 == 0 {
                            return old[l];
                        }
                        match a[l] {
                            Sym::K(byte_addr) => {
                                let target = ((byte_addr.wrapping_add(off)) >> 2) & (wave - 1);
                                d[target as usize]
                            }
                            _ => Sym::Unknown,
                        }
                    })
                    .collect();
                vec.insert(dst_v, merged);
            } else {
                for &(_, w) in &inst.writes {
                    vec.insert(w, vec![Sym::Unknown; wave as usize]);
                }
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        // 64-bit floating-point reduction step (`v_add_f64 v[d:d+1], a, b`).
        if name == "v_add_f64" {
            let (Some(dst_lo), Some(dst_hi)) = (inst.write(Reg::V(dreg)), inst.write(Reg::V(dreg + 1)))
            else {
                for &(_, w) in &inst.writes {
                    vec.insert(w, vec![Sym::Unknown; wave as usize]);
                }
                continue;
            };
            if modifiers || v.dpp.is_some() {
                vec.insert(dst_lo, vec![Sym::Unknown; wave as usize]);
                vec.insert(dst_hi, vec![Sym::Unknown; wave as usize]);
                continue;
            }
            let pair_lanes = |op: &Operand| -> (Vec<Sym>, Vec<Sym>) {
                let one = |s: Sym| vec![s; wave as usize];
                match *op {
                    Operand::Vector { reg, .. } => (
                        lanes(&Operand::Vector { reg, dwords: 1 }, &vec, &uni),
                        lanes(&Operand::Vector { reg: reg + 1, dwords: 1 }, &vec, &uni),
                    ),
                    Operand::Scalar { code, .. } if code != NULL => (
                        lanes(&Operand::Scalar { code, dwords: 1 }, &vec, &uni),
                        lanes(&Operand::Scalar { code: code + 1, dwords: 1 }, &vec, &uni),
                    ),
                    Operand::Const(128) => (one(Sym::K(0)), one(Sym::K(0))),
                    _ => (one(Sym::Unknown), one(Sym::Unknown)),
                }
            };
            let (Some(s0), Some(s1)) = (v.src.first(), v.src.get(1)) else {
                vec.insert(dst_lo, vec![Sym::Unknown; wave as usize]);
                vec.insert(dst_hi, vec![Sym::Unknown; wave as usize]);
                continue;
            };
            let (a_lo, a_hi) = pair_lanes(s0);
            let (b_lo, b_hi) = pair_lanes(s1);
            let old_lo = lanes(&Operand::Vector { reg: dreg, dwords: 1 }, &vec, &uni);
            let old_hi = lanes(&Operand::Vector { reg: dreg + 1, dwords: 1 }, &vec, &uni);
            let mut out_lo = Vec::with_capacity(wave as usize);
            let mut out_hi = Vec::with_capacity(wave as usize);
            for l in 0..wave as usize {
                if exec >> l & 1 != 0 {
                    let (rl, rh) = combine_f64((a_lo[l], a_hi[l]), (b_lo[l], b_hi[l]));
                    out_lo.push(rl);
                    out_hi.push(rh);
                } else {
                    out_lo.push(old_lo[l]);
                    out_hi.push(old_hi[l]);
                }
            }
            vec.insert(dst_lo, out_lo);
            vec.insert(dst_hi, out_hi);
            continue;
        }
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlane16_b32" | "v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read their own row (`permlane16`) or the
                // other row of their half (`permlanex16`); lane i's pick is
                // nibble i of {src2, src1}.
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        let swap = name == "v_permlanex16_b32";
                        (0..wave)
                            .map(|l| {
                                let base = if swap {
                                    (l & !31) | ((l & 16) ^ 16)
                                } else {
                                    l & !15
                                };
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(base | pick) as usize]
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_mbcnt_lo_u32_b32", Some(a)) => {
                let b = src(1).unwrap_or_default();
                (0..wave)
                    .map(|l| match (a[l as usize], b.get(l as usize).copied()) {
                        (Sym::K(mask), Some(Sym::K(acc))) => {
                            let bits = if l >= 32 { mask } else { mask & ((1u32 << l) - 1) };
                            Sym::K(bits.count_ones().wrapping_add(acc))
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_mbcnt_hi_u32_b32", Some(a)) => {
                let b = src(1).unwrap_or_default();
                (0..wave)
                    .map(|l| match (a[l as usize], b.get(l as usize).copied()) {
                        (Sym::K(mask), Some(Sym::K(acc))) => {
                            let add = if l < 32 {
                                0
                            } else {
                                (mask & ((1u32 << (l - 32)) - 1)).count_ones()
                            };
                            Sym::K(add.wrapping_add(acc))
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_lshlrev_b32", Some(a)) => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (Sym::K(sh), Some(Sym::K(val))) => Sym::K(val.wrapping_shl(sh & 31)),
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_lshrrev_b32", Some(a)) => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (Sym::K(sh), Some(Sym::K(val))) => Sym::K(val.wrapping_shr(sh & 31)),
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (Sym::K(x), Some(Sym::K(y))) => Sym::K(x.wrapping_sub(y)),
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), _)
                    if matches!(op, Op::AddF16 | Op::MinF16 | Op::MaxF16)
                        && (v.opsel != 0 || v.opsel_hi != 0) =>
                {
                    vec![Sym::Unknown; wave as usize]
                }
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
    }
    shapes(vec, uni, wave)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    let wide = inst.name() == "s_mov_b64";
    match *src.first()? {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Const(c @ 128..=208) => {
            let x = crate::resource::inline(c)?;
            Some(if wide {
                i64::from(x as i32) as u64
            } else {
                u64::from(x)
            })
        }
        Operand::Const(c) if !wide => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::Red64 {
            src_lo,
            src_hi,
            mask,
            lanes,
            hi,
        } if lanes == all => Some(Shape::Active64(src_lo, src_hi, mask, hi)),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        }
    }
    out
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            Sym::Red64 {
                src_lo,
                src_hi,
                mask,
                lanes,
                hi,
            } if lanes == all => Some(Shape::Active64(src_lo, src_hi, mask, hi)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut meta: Option<(Value, Value, u32, bool)> = None;
    let mut op_found: Option<Op> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | (1u64 << l);
        if !inclusive && set != below {
            return None;
        }
        if l == 0 && !inclusive {
            return None;
        }
        if op.is_none() && set.count_ones() != 1 {
            return None;
        }
        let lane_meta = (src, mask, ident, inclusive);
        match meta {
            None => meta = Some(lane_meta),
            Some(prev) if prev == lane_meta => {}
            _ => return None,
        }
        if let Some(o) = op {
            match op_found {
                None => op_found = Some(o),
                Some(prev) if prev == o => {}
                _ => return None,
            }
        }
    }
    let (src, mask, ident, inclusive) = meta?;
    let op = op_found?;
    if !op.identity(ident) {
        return None;
    }
    match lanes[0] {
        Sym::K(k) if !inclusive && k == ident => {}
        Sym::Red { .. } if inclusive => {}
        _ => return None,
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
    }

    const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
        (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
        (row_mask as u32) << 28
            | (bank_mask as u32) << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32) << 8
            | vsrc0 as u32
    }

    const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn constant_decodes_width_and_inline_kinds() {
        let b64 = 0x04;
        let b32 = 0x03;
        let check = |words: &[u32], wave64: bool| -> Option<u64> {
            let p = build(words, wave64);
            constant(&p.blocks[0].insts[0])
        };

        assert_eq!(
            check(&[sop1(EXEC, b64, 194)], true),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 196)], true),
            Some(0xFFFF_FFFF_FFFF_FFFC)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 208)], true),
            Some(0xFFFF_FFFF_FFFF_FFF0)
        );
        assert_eq!(check(&[sop1(EXEC, b64, 193)], true), Some(u64::MAX));
        assert_eq!(check(&[sop1(EXEC, b64, NULL)], true), Some(0));
        assert_eq!(check(&[sop1(EXEC, b64, 243)], true), None);
        assert_eq!(
            check(&[sop1(EXEC, b64, 255), 0x1234_5678], true),
            Some(0x1234_5678)
        );

        assert_eq!(check(&[sop1(EXEC, b32, 194)], false), Some(0xFFFF_FFFE));
        assert_eq!(check(&[sop1(EXEC, b32, 243)], false), Some(0xBF80_0000));
        assert_eq!(check(&[sop1(EXEC, b32, NULL)], false), Some(0));
        assert_eq!(
            check(&[sop1(EXEC, b32, 255), 0x1234_5678], false),
            Some(0x1234_5678)
        );
        assert_eq!(check(&[sop1(EXEC, b32, 0)], false), None);
    }

    #[test]
    fn exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
        for wave64 in [false, true] {
            let mut w = Vec::new();
            let save_op = if wave64 { 0x25 } else { 0x3D };
            let mov_exec = if wave64 { 0x04 } else { 0x03 };
            let push_exec = |w: &mut Vec<u32>, k: u8| w.push(sop1(EXEC, mov_exec, k));
            let push_exec_lit = |w: &mut Vec<u32>, lit: u32| {
                w.push(sop1(EXEC, mov_exec, 255));
                w.push(lit);
            };

            w.push(sop1(0, save_op, 193));
            w.extend(vop3(1, 0x101, 128, 256, 0));
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, 0x25, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0xFFFF_0000);
            w.push(vop2(1, 0x25, 2, 1));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0000_FFFF);
                w.push(vop2(1, 0x25, 2, 1));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0xFFFF_0000);
                w.push(vop2(1, 0x25, 2, 1));
            }

            push_exec(&mut w, 193);
            w.push(vop1(3, 0x01, 257));
            w.push(vop1(3, 0x01, 0xFA));
            w.push(dpp16(1, 0x111, 0xF, 0xF, false));

            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0x0001_0000);
            w.push(vop1(3, 0x01, 2));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 129));
                w.push(vop1(3, 0x01, 2));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0001_0000);
                w.push(vop1(3, 0x01, 2));
            }

            push_exec(&mut w, 193);
            w.push(vop1(2, 0x01, 128));
            w.push(vop1(5, 0x01, 135));
            push_exec(&mut w, 194);
            w.push(vop1(2, 0x01, 259));
            w.push(vop1(4, 0x01, 0xFA));
            w.push(dpp16(5, 0xE4, 0xF, 0xF, false));
            w.push(sop1(EXEC, mov_exec, 0));

            let p = build(&w, wave64);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = p
                .inputs
                .iter()
                .find_map(|&(r, v)| (r == Reg::V(0)).then_some(v))
                .unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v4_val = insts[insts.len() - 3].write(Reg::V(4)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&v2_val),
                Some(&Shape::Prefix(Op::AddU, v0_in, saved, 0, false))
            );
            assert_eq!(sec.shapes.get(&v4_val), None);
        }
    }

    #[test]
    fn wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
        let w = [
            sop1(0, 0x3D, 193),
            vop1(1, 0x01, 135),
            sop1(EXEC, 0x04, 193),
            vop1(2, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC + 1, 0x03, 193),
            vop1(3, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC, 0x03, 0),
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let v3_val = insts[5].write(Reg::V(3)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
        assert_eq!(sec.shapes.get(&v3_val), Some(&Shape::K(7)));
    }

    #[test]
    fn per_lane_prefix_soundness() {
        let (v0, v_other, m0, m_other) = (Value(1), Value(2), Value(3), Value(4));
        let red = |op: Option<Op>, src: Value, mask: Value, ident: u32, lanes: u64| Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
        };

        for (wave, all) in [(32usize, u64::from(u32::MAX)), (64usize, u64::MAX)] {
            let below_mask = |l: usize| if l == 0 { 0 } else { (1u64 << l) - 1 };

            let mut valid_inc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l) | (1u64 << l)))
                .collect();
            valid_inc[0] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_inc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, true))
            );

            let mut valid_exc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l)))
                .collect();
            valid_exc[0] = Sym::K(0);
            valid_exc[1] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_exc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, false))
            );

            let mut mixed_other_src = valid_exc.clone();
            mixed_other_src[0] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&mixed_other_src, all), None);

            let mut mixed_same_src = valid_exc.clone();
            mixed_same_src[0] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&mixed_same_src, all), None);

            let mut exc_bad_lane1_src = valid_exc.clone();
            exc_bad_lane1_src[1] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&exc_bad_lane1_src, all), None);

            let mut exc_bad_lane1_set = valid_exc.clone();
            exc_bad_lane1_set[1] = red(None, v0, m0, 0, 0b11);
            assert_eq!(per_lane(&exc_bad_lane1_set, all), None);

            for bad0 in [
                red(None, v0, m0, 1, 1),
                red(None, v_other, m0, 0, 1),
                red(None, v0, m_other, 0, 1),
                red(None, v0, m0, 0, 0),
                Sym::K(0),
            ] {
                let mut bad_inc = valid_inc.clone();
                bad_inc[0] = bad0;
                assert_eq!(per_lane(&bad_inc, all), None);
            }

            let mut bad_ident: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 1, below_mask(l) | (1u64 << l)))
                .collect();
            bad_ident[0] = red(None, v0, m0, 1, 1);
            assert_eq!(per_lane(&bad_ident, all), None);
        }
    }

    #[test]
    fn f16_wave64_reduction_is_recognised() {
        // Exact CS_TransposeVecMatMulOperation whole-wave v_add_f16 reduction sequence.
        let cnd = vop3(1, 0x101, 128, 256 + 7, 0); // v_cndmask_b32 v1, 0, v7, s[0:1]
        let dpp8 = |sel: u32| [vop2(1, 0x32, 233, 1), sel << 8 | 1]; // v_add_f16 v1, v1, v1 dpp8:...
        let mirror = [vop2(1, 0x32, 250, 1), dpp16(1, 0x140, 0xF, 0xF, true)];
        let px16 = vop3(2, 0x378, 256 + 1, 128, 128); // v_permlanex16_b32 v2, v1, 0, 0
        let add_rows = vop2(1, 0x32, 256 + 2, 1); // v_add_f16 v1, v2, v1
        let rl32 = vop3(2, 0x360, 256 + 1, 160, 0); // v_readlane_b32 s2, v1, 32
        let add_halves = vop2(1, 0x32, 2, 1); // v_add_f16 v1, s2, v1
        let rl0 = vop3(2, 0x360, 256 + 1, 128, 0); // v_readlane_b32 s2, v1, 0
        let s1 = 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21;
        let s2 = 2 | 2 << 3 | 6 << 12 | 6 << 15 | 4 << 18 | 4 << 21;
        let s4 = 4 | 4 << 3 | 4 << 6 | 4 << 9;
        let p = build(
            &[
                sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
                cnd[0],
                cnd[1],
                dpp8(s1)[0],
                dpp8(s1)[1],
                dpp8(s2)[0],
                dpp8(s2)[1],
                dpp8(s4)[0],
                dpp8(s4)[1],
                mirror[0],
                mirror[1],
                px16[0],
                px16[1],
                add_rows,
                rl32[0],
                rl32[1],
                add_halves,
                rl0[0],
                rl0[1],
                sop1(126, 0x04, 0), // s_mov_b64 exec, s[0:1]
            ],
            true,
        );
        let secs = sections(&p, 0);
        let sec = secs.get(&0).expect("section");
        let out_inst = &p.blocks[0].insts[sec.end - 1];
        let s2_out = out_inst.write(Reg::S(2)).unwrap();
        assert!(
            matches!(sec.shapes.get(&s2_out), Some(Shape::Active(Op::AddF16, _, _, 0))),
            "got {:?}",
            sec.shapes.get(&s2_out)
        );
    }

    #[test]
    fn f64_wave64_reduction_is_recognised() {
        // Exact CS_TransmissionVolumetrics* whole-wave v_add_f64 reduction sequence,
        // including scalar pair copies s_mov_b32 s20, s18; s_mov_b32 s21, s19.
        let cnd_lo = vop3(2, 0x101, 256 + 2, 256 + 29, 16);
        let cnd_hi = vop3(3, 0x101, 256 + 3, 256 + 30, 16);
        let mov_dpp8 = |vd: u8, vs: u8, sel: u32| [vop1(vd, 1, 233), sel << 8 | u32::from(vs)];
        let mov_mirror = |vd: u8, vs: u8| [vop1(vd, 1, 250), dpp16(vs, 0x140, 0xF, 0xF, true)];
        let add64_vv = vop3(2, 0x164, 256 + 4, 256 + 2, 0);
        let add64_sv = vop3(2, 0x164, 18, 256 + 2, 0);
        let px16_lo = vop3(4, 0x378, 256 + 2, 128, 128);
        let px16_hi = vop3(5, 0x378, 256 + 3, 128, 128);
        let rl32_lo = vop3(18, 0x360, 256 + 2, 160, 0);
        let rl32_hi = vop3(19, 0x360, 256 + 3, 160, 0);
        let rl0_lo = vop3(18, 0x360, 256 + 2, 128, 0);
        let rl0_hi = vop3(19, 0x360, 256 + 3, 128, 0);
        let s1 = 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21;
        let s2 = 2 | 2 << 3 | 6 << 12 | 6 << 15 | 4 << 18 | 4 << 21;
        let s4 = 4 | 4 << 3 | 4 << 6 | 4 << 9;
        let mut words = vec![
            sop1(16, 0x25, 193), // s_or_saveexec_b64 s[16:17], -1
            vop1(2, 1, 128),     // v_mov_b32 v2, 0
            vop1(3, 1, 128),     // v_mov_b32 v3, 0
            cnd_lo[0],
            cnd_lo[1],
            cnd_hi[0],
            cnd_hi[1],
        ];
        for sel in [s1, s2, s4] {
            words.extend_from_slice(&mov_dpp8(4, 2, sel));
            words.extend_from_slice(&mov_dpp8(5, 3, sel));
            words.extend_from_slice(&add64_vv);
        }
        words.extend_from_slice(&mov_mirror(4, 2));
        words.extend_from_slice(&mov_mirror(5, 3));
        words.extend_from_slice(&add64_vv);
        words.extend_from_slice(&px16_lo);
        words.extend_from_slice(&px16_hi);
        words.extend_from_slice(&add64_vv);
        words.extend_from_slice(&rl32_lo);
        words.extend_from_slice(&rl32_hi);
        words.extend_from_slice(&add64_sv);
        words.extend_from_slice(&rl0_lo);
        words.extend_from_slice(&rl0_hi);
        words.push(sop1(20, 0x03, 18)); // s_mov_b32 s20, s18
        words.push(sop1(21, 0x03, 19)); // s_mov_b32 s21, s19
        words.push(sop1(126, 0x04, 16)); // s_mov_b64 exec, s[16:17]

        let p = build(&words, true);
        let secs = sections(&p, 0);
        let sec = secs.get(&0).expect("section");
        let s20 = p.blocks[0].insts[sec.end - 2].write(Reg::S(20)).unwrap();
        let s21 = p.blocks[0].insts[sec.end - 1].write(Reg::S(21)).unwrap();
        assert!(
            matches!(sec.shapes.get(&s20), Some(Shape::Active64(_, _, _, false))),
            "s20 = {:?}",
            sec.shapes.get(&s20)
        );
        assert!(
            matches!(sec.shapes.get(&s21), Some(Shape::Active64(_, _, _, true))),
            "s21 = {:?}",
            sec.shapes.get(&s21)
        );
    }

    #[test]
    fn permlane16_and_bpermute_reductions_are_recognised() {
        // Wave32 reduction using v_permlane16_b32 within rows + ds_bpermute_b32 across rows.
        let cnd = vop3(1, 0x101, 128, 256 + 7, 0); // v_cndmask_b32 v1, 0, v7, s0
        let dpp8 = |sel: u32| [vop2(1, 0x25, 233, 1), sel << 8 | 1]; // v_add_nc_u32 dpp8
        let mirror = [vop2(1, 0x25, 250, 1), dpp16(1, 0x140, 0xF, 0xF, true)];
        let mbcnt = vop3(3, 0x365, 193, 128, 0); // v_mbcnt_lo_u32_b32 v3, -1, 0
        let xor16 = vop2(3, 0x1D, 128 + 16, 3); // v_xor_b32 v3, 16, v3
        let shl2 = vop2(3, 0x1A, 128 + 2, 3); // v_lshlrev_b32 v3, 2, v3
        // ds_bpermute_b32 v2, v3, v1 (DS op 0xB3)
        let bperm = [0xD800_0000 | (0xB3 << 18), (2 << 24) | (1 << 8) | 3];
        let add_halves = vop2(1, 0x25, 256 + 2, 1); // v_add_nc_u32 v1, v2, v1
        let rl0 = vop3(2, 0x360, 256 + 1, 128, 0);
        let s1 = 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21;
        let s2 = 2 | 2 << 3 | 6 << 12 | 6 << 15 | 4 << 18 | 4 << 21;
        let s4 = 4 | 4 << 3 | 4 << 6 | 4 << 9;
        let p = build(
            &[
                sop1(0, 0x3D, 193), // s_or_saveexec_b32 s0, -1
                cnd[0],
                cnd[1],
                dpp8(s1)[0],
                dpp8(s1)[1],
                dpp8(s2)[0],
                dpp8(s2)[1],
                dpp8(s4)[0],
                dpp8(s4)[1],
                mirror[0],
                mirror[1],
                mbcnt[0],
                mbcnt[1],
                xor16,
                shl2,
                bperm[0],
                bperm[1],
                add_halves,
                rl0[0],
                rl0[1],
                sop1(126, 0x03, 0), // s_mov_b32 exec_lo, s0
            ],
            false,
        );
        let secs = sections(&p, 0);
        let sec = secs.get(&0).expect("section");
        let s2_out = p.blocks[0].insts[sec.end - 1].write(Reg::S(2)).unwrap();
        assert!(
            matches!(sec.shapes.get(&s2_out), Some(Shape::Active(Op::AddU, _, _, 0))),
            "got {:?}",
            sec.shapes.get(&s2_out)
        );
    }
}
