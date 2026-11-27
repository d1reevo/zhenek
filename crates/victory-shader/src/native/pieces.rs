//! Programs a title passes as loose pieces instead of a container.
//!
//! Two device methods of the console (table offsets 0x218 and 0x1c8) make a
//! pipeline from pieces that lie apart in title memory: a packet of register
//! values, a piece of pipeline state, and per stage the GFX10 code and its
//! metadata. The metadata is the record stream an `ACCC` part's kind 7 entry
//! holds ([`crate::native::fspx_program`]); only its header differs. This
//! module reads the packets, and wraps each stage in a container of our own
//! so that everything built on containers (content hashes, the translation
//! cache, [`crate::emit::translate`]) takes it unchanged.
//!
//! Layouts as observed in 48 calls of each method of one title (offsets in
//! bytes, little endian; "seen" is all that the captures show, a guess is
//! labelled):
//!
//! ```text
//! compute description, 0x48 bytes (9 u64)
//! +0x00 root signature (an object of the caller's)
//! +0x08 address, length: the packet, 0x1c bytes
//! +0x18 address, length: 0x14 bytes of pipeline metadata (not read)
//! +0x28 address, length: the code
//! +0x38 address, length: the metadata
//!
//! compute packet, 0x1c bytes
//! +0x0c u16 x3 thread-group size
//! +0x12 u16 flags: bit 4 wave32 (never set in the calls seen)
//! +0x14 COMPUTE_PGM_RSRC1: bits 0-5 VGPR granules - 1
//! +0x18 COMPUTE_PGM_RSRC2
//!
//! graphics description, 0x68 bytes read (13 u64); the 20 words after
//! them were zero in every call (room for more stages: a guess)
//! +0x00 root signature
//! +0x08 address, length: the packet
//! +0x18 address, length: the state piece
//! +0x28 address, length: pixel code; +0x38 pixel metadata (all four 0
//!       without a pixel stage)
//! +0x48 address, length: vertex code; +0x58 vertex metadata
//!
//! graphics packet: register values at fixed places, no register numbers
//! +0x28 u16 0x11c (size of the block from +0x1c), +0x2a u16 size of the
//!       block from +0x138; length = 0x138 + that
//! +0x38 SPI_SHADER_PGM_RSRC1_VS, +0x3c _RSRC2_VS (bits 1-5 user SGPRs)
//! +0x4c SPI_SHADER_PGM_RSRC1_PS (0 without a pixel stage), +0x50 _RSRC2_PS
//! +0x58 SPI_PS_INPUT_ADDR, +0x5c SPI_PS_IN_CONTROL (bits 0-5 interpolants)
//! +0x60 SPI_SHADER_Z_FORMAT, +0x68 DB_SHADER_CONTROL
//! +0x70 SPI_SHADER_COL_FORMAT, +0x7c CB_COLOR_CONTROL
//! +0x84 SPI_VS_OUT_CONFIG, +0x90 SPI_SHADER_POS_FORMAT
//! +0x98 PA_CL_VS_OUT_CNTL, +0x9c PA_CL_CLIP_CNTL, +0xa0 PA_SU_SC_MODE_CNTL
//! +0xac three dwords, always 0 (depth bias: a guess)
//! +0xb8 DB_DEPTH_CONTROL, +0xbc DB_STENCIL_CONTROL
//! +0xc0 DB_STENCILREFMASK, +0xc4 DB_STENCILREFMASK_BF
//! +0xc8 always 0 (not identified), +0xcc CB_TARGET_MASK
//! +0xd0 CB_BLEND0_CONTROL .. CB_BLEND7_CONTROL
//! +0x170 SPI_PS_INPUT_CNTL_0.., as many as fit before the last 12 bytes
//!        (render target information, which repeats the state piece)
//!
//! state piece
//! +0x10 u8 x8 render target formats (DXGI numbers), 0 past the count
//! +0x1c u32 0xffffffff (sample mask: a guess)
//! +0x24 u8 depth format, +0x25 u8 render target count
//! +0x26 u8 primitive topology type (3 triangles, 19 the console's
//!       rectangles: by the two values seen)
//! +0x28 u32 1 (sample count: a guess)
//! +0x38 u32 size of the input layout block at +0x4c: u32 element count,
//!       then per element u32 name length with its NUL, the name, and six
//!       u32 of which the second is the DXGI format and the fourth the byte
//!       offset in the vertex. The other four were 0 in every element
//!       (semantic index, slot, class and step rate, in an order not known)
//!
//! metadata piece
//! +0x00 u32 0x403 (compute) or 2 (graphics)
//! +0x08 u32 code dwords: the code piece's length / 4 (compute), or that
//!       less 10 (graphics: 40 bytes at the end are not counted)
//! records of 32 bytes from +0x18 (compute) or +0x10 (graphics)
//! ```
//!
//! A vertex program has no fetch shader: it loads the 16-byte descriptor of
//! a vertex buffer from the table whose address arrives in s0 and s1, adds
//! s2 to the vertex index in v0 and reads its attributes with buffer loads
//! at the elements' byte offsets. Pixel exports are final (two 16-bit floats
//! per dword for the `FP16_ABGR` format): nothing patches them.
//!
//! The container made for a stage has one part of ours ([`FourCC::PIECES`])
//! and, for graphics stages, `ISG1` and `OSG1` parts made from the packet
//! (the pieces name no semantics: parameter n of the vertex program is
//! `TEXCOORD<n>` in register n on both sides). The part:
//!
//! ```text
//! +0x00 "VPP1"
//! +0x04 u8 stage (0 compute, 1 vertex, 2 pixel), u8 bit 0 wave32, u16 VGPRs
//! +0x08 u32 user SGPRs
//! +0x0c u32 x5 by stage
//!       compute: COMPUTE_PGM_RSRC2, thread-group size x, y, z
//!       vertex:  parameter exports, bit per parameter read flat, elements
//!       pixel:   SPI_PS_INPUT_ADDR, DB_SHADER_CONTROL, SPI_SHADER_COL_FORMAT,
//!                the vertex program's parameter exports, interpolants
//! +0x20 u32 size of what follows the header, u32 code size, u32 metadata
//!       size, u32 0
//! +0x30 vertex: elements of 48 bytes (32 of name, u32 semantic index, DXGI
//!       format, slot, byte offset); pixel: SPI_PS_INPUT_CNTL per interpolant
//! then  the code piece and the metadata piece, as the title has them
//! ```
//!
//! Only what a stage's translation depends on goes into its part, so two
//! pipelines that differ in blend state alone share their programs, and two
//! that differ in anything a program reads never do.

use std::ops::Range;

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{FourCC, u16_at, u32_at};
use crate::native::{
    self, ComputeIo, HwStage, NativeProgram, PsInput, RecordRules, RootParam, RootType, StageIo,
    Usage, VERTEX_BUFFERS, VsFetch, VsOutput,
};

/// What starts the part of ours.
pub const MAGIC: &[u8; 4] = b"VPP1";
/// The lengths of a compute description's first two pieces.
pub const COMPUTE_PACKET: usize = 0x1c;
pub const COMPUTE_PIPELINE_PIECE: usize = 0x14;
/// The longest pieces taken (the longest seen: 123 KiB of code, 6 KiB of
/// metadata, 0x18c bytes of packet, 0xbc of state).
pub const MAX_CODE: usize = 4 << 20;
pub const MAX_METADATA: usize = 64 << 10;
pub const MAX_PACKET: usize = 0x400;
pub const MAX_STATE: usize = 0x400;

/// Why a pipeline is not made from its pieces: something malformed, or
/// something understood too little to translate without guessing.
pub type Refusal = &'static str;

const HEAD: usize = 0x30;
const RECORD: usize = 32;
const COMPUTE: u8 = 0;
const VERTEX: u8 = 1;
const PIXEL: u8 = 2;
const WAVE32: u8 = 1;
/// (metadata tag, where its records start, code bytes it does not count)
const COMPUTE_METADATA: (u32, usize, usize) = (0x403, 0x18, 0);
const GRAPHICS_METADATA: (u32, usize, usize) = (2, 0x10, 0x28);
const ELEMENT: usize = 48;
const NAME: usize = 32;
const MAX_ELEMENTS: usize = 16;

// The compute packet.
const C_THREADS: usize = 0x0c;
const C_FLAGS: usize = 0x12;
const C_WAVE32: u16 = 1 << 4;
const C_RSRC1: usize = 0x14;
const C_RSRC2: usize = 0x18;

// The graphics packet.
const G_FIXED_SIZE: usize = 0x28;
const G_SECOND_SIZE: usize = 0x2a;
const G_FIXED: u16 = 0x11c;
const G_SECOND: usize = 0x138;
const G_RSRC1_VS: usize = 0x38;
const G_RSRC2_VS: usize = 0x3c;
const G_RSRC1_PS: usize = 0x4c;
const G_RSRC2_PS: usize = 0x50;
const G_PS_INPUT_ADDR: usize = 0x58;
const G_PS_IN_CONTROL: usize = 0x5c;
const G_Z_FORMAT: usize = 0x60;
const G_DB_SHADER_CONTROL: usize = 0x68;
const G_COL_FORMAT: usize = 0x70;
const G_CB_COLOR_CONTROL: usize = 0x7c;
const G_VS_OUT_CONFIG: usize = 0x84;
const G_POS_FORMAT: usize = 0x90;
const G_VS_OUT_CNTL: usize = 0x98;
const G_CLIP_CNTL: usize = 0x9c;
const G_SU_SC_MODE_CNTL: usize = 0xa0;
const G_DEPTH_BIAS: usize = 0xac;
const G_DEPTH_CONTROL: usize = 0xb8;
const G_STENCIL_CONTROL: usize = 0xbc;
const G_STENCILREFMASK: usize = 0xc0;
const G_UNKNOWN: usize = 0xc8;
const G_TARGET_MASK: usize = 0xcc;
const G_BLEND: usize = 0xd0;
const G_INPUT_CNTL: usize = 0x170;
/// The second block before the input controls, and the render target
/// information after them.
const G_SECOND_HEAD: usize = 0x38;
const G_TARGET_INFO: usize = 12;

// The state piece.
const S_TARGETS: usize = 0x10;
const S_SAMPLE_MASK: usize = 0x1c;
const S_DEPTH: usize = 0x24;
const S_COUNT: usize = 0x25;
const S_TOPOLOGY: usize = 0x26;
const S_SAMPLES: usize = 0x28;
const S_BLOCK_SIZE: usize = 0x38;
const S_BLOCK: usize = 0x4c;

/// DXGI formats of the input elements whose fetch is known, as (bytes,
/// channels): the 32-bit float formats, which the code loads as dwords, and
/// `R8G8B8A8_UNORM`, which it loads through a typed format.
pub(crate) fn element_shape(format: u32) -> Option<(u32, u8)> {
    match format {
        2 => Some((16, 4)),
        6 => Some((12, 3)),
        16 => Some((8, 2)),
        41 => Some((4, 1)),
        R8G8B8A8_UNORM => Some((4, 4)),
        _ => None,
    }
}

/// The one byte-wise element format among [`element_shape`]'s.
pub(crate) const R8G8B8A8_UNORM: u32 = 28;

/// A container for one stage, whole but for the two pieces the caller
/// reads into it from title memory: `bytes[code]` and `bytes[metadata]`.
/// [`check`] then tells whether the pieces agree with each other.
#[derive(Debug)]
pub struct Draft {
    pub bytes: Vec<u8>,
    pub code: Range<usize>,
    pub metadata: Range<usize>,
}

/// The header of the part of ours.
struct Head {
    stage: u8,
    flags: u8,
    vgprs: u16,
    user_sgprs: u32,
    words: [u32; 5],
}

/// One row of a signature part.
struct Row<'a> {
    name: &'a str,
    index: u32,
    /// `D3D_NAME`: 0 none, 1 position, 9 front face, 64 target, 65 depth.
    system: u32,
    /// `D3D_REGISTER_COMPONENT_TYPE`: 1 uint32, 3 float32, 6 float16.
    component: u32,
    register: u32,
    mask: u8,
}

/// A signature part's data (`ISG1`, `OSG1`): count, offset of the rows,
/// rows of 32 bytes, then the names.
fn signature(rows: &[Row]) -> Vec<u8> {
    let mut names = 8 + 32 * rows.len();
    let mut out = Vec::with_capacity(names + rows.iter().map(|r| r.name.len() + 1).sum::<usize>());
    out.extend((rows.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    for r in rows {
        // stream, name, semantic index, system value, type, register
        for v in [0, names as u32, r.index, r.system, r.component, r.register] {
            out.extend(v.to_le_bytes());
        }
        // mask, then the mask of what is always read or never written,
        // which nothing here reads; minimum precision none
        out.extend([r.mask, 0, 0, 0]);
        out.extend(0u32.to_le_bytes());
        names += r.name.len() + 1;
    }
    for r in rows {
        out.extend(r.name.as_bytes());
        out.push(0);
    }
    out
}

/// Whether the two pieces' lengths can be a code piece and a metadata piece
/// whose records start at `records`.
fn sizes(code: usize, metadata: usize, records: usize) -> Result<(), Refusal> {
    if code == 0 || !code.is_multiple_of(4) || code > MAX_CODE {
        return Err("a code piece that is empty, not whole dwords or implausibly long");
    }
    if metadata < records + RECORD
        || metadata > MAX_METADATA
        || !(metadata - records).is_multiple_of(RECORD)
    {
        return Err("a metadata piece that is not a header and whole records");
    }
    Ok(())
}

/// The container for a stage: the part of ours with `head` and `extra`
/// written and room for the code and the metadata, then the signatures.
fn draft(
    head: &Head,
    extra: &[u8],
    code: usize,
    metadata: usize,
    signatures: Option<[&[Row]; 2]>,
) -> Draft {
    let sigs = signatures.map(|[i, o]| [signature(i), signature(o)]);
    let part = HEAD + extra.len() + code + metadata;
    let mut parts = vec![(FourCC::PIECES, part)];
    if let Some([i, o]) = &sigs {
        parts.push((FourCC::ISG1, i.len()));
        parts.push((FourCC::OSG1, o.len()));
    }
    // The container: header, part offsets, then each part as its tag, its
    // size and its data.
    let table = 0x20 + 4 * parts.len();
    let total = table + parts.iter().map(|p| 8 + p.1).sum::<usize>();
    let mut bytes = vec![0u8; total];
    bytes[..4].copy_from_slice(b"DXBC");
    // (the digest stays zero: it names no shader anyone listed)
    bytes[0x14..0x16].copy_from_slice(&1u16.to_le_bytes());
    bytes[0x18..0x1c].copy_from_slice(&(total as u32).to_le_bytes());
    bytes[0x1c..0x20].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut at = table;
    let mut data = Vec::with_capacity(parts.len());
    for (i, (fourcc, size)) in parts.iter().enumerate() {
        bytes[0x20 + 4 * i..0x24 + 4 * i].copy_from_slice(&(at as u32).to_le_bytes());
        bytes[at..at + 4].copy_from_slice(&fourcc.0);
        bytes[at + 4..at + 8].copy_from_slice(&(*size as u32).to_le_bytes());
        data.push(at + 8);
        at += 8 + size;
    }
    // The part of ours.
    let p = data[0];
    bytes[p..p + 4].copy_from_slice(MAGIC);
    bytes[p + 4] = head.stage;
    bytes[p + 5] = head.flags;
    bytes[p + 6..p + 8].copy_from_slice(&head.vgprs.to_le_bytes());
    bytes[p + 8..p + 12].copy_from_slice(&head.user_sgprs.to_le_bytes());
    for (i, w) in head.words.iter().enumerate() {
        bytes[p + 12 + 4 * i..p + 16 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
    for (i, n) in [extra.len(), code, metadata].into_iter().enumerate() {
        bytes[p + 0x20 + 4 * i..p + 0x24 + 4 * i].copy_from_slice(&(n as u32).to_le_bytes());
    }
    let code_at = p + HEAD + extra.len();
    bytes[p + HEAD..code_at].copy_from_slice(extra);
    if let Some(sigs) = &sigs {
        for (sig, &at) in sigs.iter().zip(&data[1..]) {
            bytes[at..at + sig.len()].copy_from_slice(sig);
        }
    }
    Draft {
        bytes,
        code: code_at..code_at + code,
        metadata: code_at + code..code_at + code + metadata,
    }
}

/// VGPRs from a program resource register: granules of 4 (wave64) or 8.
fn vgprs(rsrc1: u32, wave32: bool) -> u16 {
    (((rsrc1 & 0x3f) + 1) * if wave32 { 8 } else { 4 }).min(256) as u16
}

/// The container for a compute program whose packet is `packet`, with room
/// for `code` and `metadata` bytes of the two other pieces.
pub fn compute(packet: &[u8], code: usize, metadata: usize) -> Result<Draft, Refusal> {
    if packet.len() != COMPUTE_PACKET {
        return Err("a compute packet that is not 0x1c bytes long");
    }
    sizes(code, metadata, COMPUTE_METADATA.1)?;
    let word = |at: usize| u32_at(packet, at).unwrap_or(0);
    let threads: [u32; 3] =
        std::array::from_fn(|i| u16_at(packet, C_THREADS + 2 * i).map_or(0, u32::from));
    // What a thread group can be on the host: no side over 1024, 1024
    // threads in all.
    if threads.contains(&0) || threads.iter().product::<u32>() > 1024 {
        return Err("a compute packet with an implausible thread-group size");
    }
    let wave32 = u16_at(packet, C_FLAGS).unwrap_or(0) & C_WAVE32 != 0;
    let rsrc2 = word(C_RSRC2);
    let head = Head {
        stage: COMPUTE,
        flags: if wave32 { WAVE32 } else { 0 },
        vgprs: vgprs(word(C_RSRC1), wave32),
        user_sgprs: rsrc2 >> 1 & 0x1f,
        words: [rsrc2, threads[0], threads[1], threads[2], 0],
    };
    Ok(draft(&head, &[], code, metadata, None))
}

/// One element of the input layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    /// Its semantic name: letters, digits and underscores.
    pub name: String,
    /// DXGI format.
    pub format: u32,
    /// Byte offset in the vertex.
    pub offset: u32,
}

/// The fixed-function registers of a graphics packet, as stored. What each
/// means on the host is the caller's to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    pub cb_color_control: u32,
    pub pa_cl_clip_cntl: u32,
    pub pa_su_sc_mode_cntl: u32,
    pub db_depth_control: u32,
    pub db_stencil_control: u32,
    pub db_stencilrefmask: u32,
    pub db_stencilrefmask_bf: u32,
    pub cb_target_mask: u32,
    pub cb_blend_control: [u32; 8],
}

/// A pixel program's part of the packet.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pixel {
    vgprs: u16,
    user_sgprs: u32,
    input_addr: u32,
    db_shader_control: u32,
    col_format: u32,
    /// `SPI_PS_INPUT_CNTL` of each interpolant.
    controls: Vec<u32>,
}

/// A graphics pipeline's packet and state piece, read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graphics {
    pub registers: Registers,
    /// DXGI formats of the render targets, `targets` of them.
    pub target_formats: [u8; 8],
    pub targets: u8,
    /// DXGI format of the depth buffer, 0 for none.
    pub depth_format: u8,
    /// 3 triangles, 19 the console's rectangles.
    pub topology: u8,
    /// Every element is of slot 0, per vertex, semantic index 0.
    pub elements: Vec<Element>,
    vertex_vgprs: u16,
    vertex_user_sgprs: u32,
    /// `exp param<n>` of the vertex program, n below this.
    params: u32,
    pixel: Option<Pixel>,
}

impl Graphics {
    /// Reads a graphics packet and its state piece; `pixel` says whether
    /// the description has a pixel stage. Refuses what the layouts in the
    /// module notes do not cover, and every value whose meaning for a
    /// translated program is not known.
    pub fn parse(packet: &[u8], state: &[u8], pixel: bool) -> Result<Graphics, Refusal> {
        let word = |at: usize| u32_at(packet, at).unwrap_or(0);
        // The packet's two sizes must add up to it.
        let second = usize::from(u16_at(packet, G_SECOND_SIZE).unwrap_or(0));
        if packet.len() > MAX_PACKET
            || packet.len() < G_INPUT_CNTL + G_TARGET_INFO
            || u16_at(packet, G_FIXED_SIZE) != Ok(G_FIXED)
            || packet.len() != G_SECOND + second
        {
            return Err("a graphics packet of an unknown shape");
        }
        // Input controls fill what lies between the second block's head
        // and the render target information.
        let controls = second - G_SECOND_HEAD - G_TARGET_INFO;
        if !controls.is_multiple_of(4) {
            return Err("a graphics packet of an unknown shape");
        }
        // The vertex program's parameter exports.
        let out = word(G_VS_OUT_CONFIG);
        let params = match out {
            0x80 => 0,
            _ if out & !0x3e == 0 => (out >> 1 & 0x1f) + 1,
            _ => return Err("SPI_VS_OUT_CONFIG bits that are not handled"),
        };
        // One position export and nothing else of the clipper's: a second
        // one carries clip and cull distances, whose rows are not known.
        if word(G_POS_FORMAT) != 4 || word(G_VS_OUT_CNTL) != 0 {
            return Err("a vertex program that exports clip or cull distances");
        }
        if (0..3).any(|i| word(G_DEPTH_BIAS + 4 * i) != 0) || word(G_UNKNOWN) != 0 {
            return Err("a graphics packet with values where only zeros were seen");
        }
        let pixel = if pixel {
            if word(G_RSRC1_PS) == 0 {
                return Err("a pixel stage whose packet has none");
            }
            let control = word(G_PS_IN_CONTROL);
            let interpolants = (control & 0x3f) as usize;
            // (bit 15 would be a wave32 pixel program: never seen)
            if control & !0x3f != 0 || interpolants > controls / 4 {
                return Err("SPI_PS_IN_CONTROL bits that are not handled");
            }
            let controls: Vec<u32> = (0..interpolants)
                .map(|k| word(G_INPUT_CNTL + 4 * k))
                .collect();
            for &c in &controls {
                // The parameter read (bits 0-4) and flat shading (bit 10):
                // a default value, 16-bit interpolation and point sprites
                // are other bits.
                if c & !0x41f != 0 || c & 0x1f >= params {
                    return Err("a pixel input control that is not handled");
                }
            }
            // One parameter read both flat and interpolated has no one type.
            for a in &controls {
                if controls.iter().any(|b| a & 0x1f == b & 0x1f && a != b) {
                    return Err("a vertex parameter read both flat and interpolated");
                }
            }
            let db = word(G_DB_SHADER_CONTROL);
            // Depth export, the depth test's order, kill, depth before the
            // shader. Stencil and coverage exports, conservative depth and
            // alpha to coverage are other bits.
            if db & !0x1071 != 0 || word(G_Z_FORMAT) != db & 1 {
                return Err("DB_SHADER_CONTROL bits that are not handled");
            }
            let col_format = word(G_COL_FORMAT);
            // Per target: nothing, 32-bit floats (32_R, 32_GR, 32_AR,
            // 32_ABGR) or 16-bit floats (FP16_ABGR). The 16-bit integer
            // formats were never seen.
            if (0..8).any(|n| !matches!(col_format >> (4 * n) & 0xf, 0..=4 | 9)) {
                return Err("an export format that is not handled");
            }
            Some(Pixel {
                vgprs: vgprs(word(G_RSRC1_PS), false),
                user_sgprs: word(G_RSRC2_PS) >> 1 & 0x1f,
                input_addr: word(G_PS_INPUT_ADDR),
                db_shader_control: db,
                col_format,
                controls,
            })
        } else {
            if word(G_RSRC1_PS) != 0 {
                return Err("a packet for a pixel stage the description lacks");
            }
            None
        };

        // The state piece.
        let byte = |at: usize| state.get(at).copied().unwrap_or(0);
        let block = u32_at(state, S_BLOCK_SIZE).unwrap_or(u32::MAX) as usize;
        if state.len() > MAX_STATE
            || state.len() < S_BLOCK + 4
            || block < 4
            || state.len() != (S_BLOCK + block).next_multiple_of(4)
        {
            return Err("a state piece of an unknown shape");
        }
        let targets = byte(S_COUNT);
        let mut target_formats = [0u8; 8];
        target_formats.copy_from_slice(&state[S_TARGETS..S_TARGETS + 8]);
        if targets > 8
            || target_formats
                .iter()
                .enumerate()
                .any(|(i, &f)| (f != 0) != (i < usize::from(targets)))
        {
            return Err("a state piece whose render targets disagree with their count");
        }
        if u32_at(state, S_SAMPLES) != Ok(1) || u32_at(state, S_SAMPLE_MASK) != Ok(u32::MAX) {
            return Err("a multisampled pipeline, or one with a sample mask");
        }
        let topology = byte(S_TOPOLOGY);
        if !matches!(topology, 3 | 19) {
            return Err("a primitive topology type other than triangles and rectangles");
        }
        // The input layout: a count, then elements back to back.
        let layout = &state[S_BLOCK..S_BLOCK + block];
        let count = u32_at(layout, 0).unwrap_or(u32::MAX) as usize;
        if count > MAX_ELEMENTS {
            return Err("an input layout with an implausible element count");
        }
        let mut elements = Vec::with_capacity(count);
        let mut at = 4;
        for _ in 0..count {
            const BAD: Refusal = "an input layout that does not parse";
            let len = u32_at(layout, at).map_err(|_| BAD)? as usize;
            // The name with its NUL, then six words.
            let name = layout
                .get(at + 4..(at + 4).saturating_add(len))
                .ok_or(BAD)?;
            let Some((&0, name)) = name.split_last() else {
                return Err(BAD);
            };
            // (a name that ends in a digit would carry its semantic index,
            // which the host wants apart from the name: none was seen)
            if name.is_empty()
                || name.len() >= NAME
                || !name.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
                || name.last().is_some_and(u8::is_ascii_digit)
                || name[..name.len().min(3)].eq_ignore_ascii_case(b"SV_")
            {
                return Err("an input element with a name that is not a plain semantic");
            }
            at += 4 + len;
            let words: [u32; 6] = match layout.get(at..at + 24) {
                Some(w) => std::array::from_fn(|i| u32_at(w, 4 * i).unwrap_or(0)),
                None => return Err(BAD),
            };
            at += 24;
            // Which of these four is the index, the slot, the class and the
            // step rate is not known: all were zero wherever seen.
            if [words[0], words[2], words[4], words[5]] != [0; 4] {
                return Err("an input element with an index, slot, class or step rate");
            }
            if words[3] > 0x800 {
                return Err("an input element at an implausible byte offset");
            }
            elements.push(Element {
                // (checked ASCII above)
                name: String::from_utf8_lossy(name).into_owned(),
                format: words[1],
                offset: words[3],
            });
        }
        if at != block {
            return Err("an input layout that does not fill its block");
        }
        Ok(Graphics {
            registers: Registers {
                cb_color_control: word(G_CB_COLOR_CONTROL),
                pa_cl_clip_cntl: word(G_CLIP_CNTL),
                pa_su_sc_mode_cntl: word(G_SU_SC_MODE_CNTL),
                db_depth_control: word(G_DEPTH_CONTROL),
                db_stencil_control: word(G_STENCIL_CONTROL),
                db_stencilrefmask: word(G_STENCILREFMASK),
                db_stencilrefmask_bf: word(G_STENCILREFMASK + 4),
                cb_target_mask: word(G_TARGET_MASK),
                cb_blend_control: std::array::from_fn(|i| word(G_BLEND + 4 * i)),
            },
            target_formats,
            targets,
            depth_format: byte(S_DEPTH),
            topology,
            elements,
            vertex_vgprs: vgprs(word(G_RSRC1_VS), false),
            vertex_user_sgprs: word(G_RSRC2_VS) >> 1 & 0x1f,
            params,
            pixel,
        })
    }

    /// Whether the packet has a pixel stage.
    pub fn has_pixel(&self) -> bool {
        self.pixel.is_some()
    }

    /// Bit per vertex parameter the pixel program reads flat. Such a
    /// parameter is declared a 32-bit integer on both sides, so that
    /// whatever the vertex program exports arrives bit for bit.
    fn flat(&self) -> u32 {
        let controls = self.pixel.iter().flat_map(|p| &p.controls);
        controls
            .filter(|&&c| c & 0x400 != 0)
            .fold(0, |m, &c| m | 1 << (c & 0x1f))
    }

    /// The row of vertex parameter `p` in both stages' signatures.
    fn param(&self, p: u32) -> Row<'static> {
        Row {
            name: "TEXCOORD",
            index: p,
            system: 0,
            component: if self.flat() >> p & 1 != 0 { 1 } else { 3 },
            register: p,
            mask: 0xf,
        }
    }

    /// The position's row: after the parameters, on both sides.
    fn position(&self) -> Row<'static> {
        Row {
            name: "SV_Position",
            index: 0,
            system: 1,
            component: 3,
            register: self.params,
            mask: 0xf,
        }
    }

    /// The container for the vertex program, with room for its two pieces.
    pub fn vertex(&self, code: usize, metadata: usize) -> Result<Draft, Refusal> {
        sizes(code, metadata, GRAPHICS_METADATA.1)?;
        let mut extra = vec![0u8; ELEMENT * self.elements.len()];
        for (e, to) in self.elements.iter().zip(extra.chunks_exact_mut(ELEMENT)) {
            to[..e.name.len()].copy_from_slice(e.name.as_bytes());
            // semantic index, format, slot, byte offset
            for (i, v) in [0, e.format, 0, e.offset].into_iter().enumerate() {
                to[NAME + 4 * i..NAME + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        // The elements whose fetch is known are the input signature, each
        // in the register of its place in the layout. The host's input
        // layout may hold more than the program declares.
        let inputs: Vec<Row> = self
            .elements
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let (_, channels) = element_shape(e.format)?;
                Some(Row {
                    name: &e.name,
                    index: 0,
                    system: 0,
                    component: 3,
                    register: i as u32,
                    mask: (1 << channels) - 1,
                })
            })
            .collect();
        let mut outputs: Vec<Row> = (0..self.params).map(|p| self.param(p)).collect();
        outputs.push(self.position());
        let head = Head {
            stage: VERTEX,
            flags: 0,
            vgprs: self.vertex_vgprs,
            user_sgprs: self.vertex_user_sgprs,
            words: [self.params, self.flat(), self.elements.len() as u32, 0, 0],
        };
        Ok(draft(
            &head,
            &extra,
            code,
            metadata,
            Some([&inputs[..], &outputs[..]]),
        ))
    }

    /// The container for the pixel program, with room for its two pieces.
    pub fn pixel(&self, code: usize, metadata: usize) -> Result<Draft, Refusal> {
        let Some(pixel) = &self.pixel else {
            return Err("a pixel stage whose packet has none");
        };
        sizes(code, metadata, GRAPHICS_METADATA.1)?;
        // Each parameter read, once, then the system values the packet
        // enables: the position (bits 8-11) where the vertex program has
        // it, the front face (bit 12) after it.
        let mut read: Vec<u32> = pixel.controls.iter().map(|c| c & 0x1f).collect();
        read.sort_unstable();
        read.dedup();
        let mut inputs: Vec<Row> = read.into_iter().map(|p| self.param(p)).collect();
        if pixel.input_addr & 0xf00 != 0 {
            inputs.push(self.position());
        }
        if pixel.input_addr & 0x1000 != 0 {
            inputs.push(Row {
                name: "SV_IsFrontFace",
                index: 0,
                system: 9,
                component: 1,
                register: self.params + 1,
                mask: 1,
            });
        }
        // A render target per export format: 16-bit floats travel two to a
        // dword and are declared so, which is what lets the lowering take
        // a compressed export apart.
        let mut outputs = Vec::new();
        for n in 0..8 {
            let component = match pixel.col_format >> (4 * n) & 0xf {
                0 => continue,
                4 => 6,
                _ => 3,
            };
            outputs.push(Row {
                name: "SV_Target",
                index: n,
                system: 64,
                component,
                register: n,
                mask: 0xf,
            });
        }
        if pixel.db_shader_control & 1 != 0 {
            outputs.push(Row {
                name: "SV_Depth",
                index: 0,
                system: 65,
                component: 3,
                register: u32::MAX,
                mask: 1,
            });
        }
        let extra: Vec<u8> = pixel
            .controls
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        let head = Head {
            stage: PIXEL,
            flags: 0,
            vgprs: pixel.vgprs,
            user_sgprs: pixel.user_sgprs,
            words: [
                pixel.input_addr,
                pixel.db_shader_control,
                pixel.col_format,
                self.params,
                pixel.controls.len() as u32,
            ],
        };
        Ok(draft(
            &head,
            &extra,
            code,
            metadata,
            Some([&inputs[..], &outputs[..]]),
        ))
    }
}

/// Whether a filled [`Draft`] holds a program: what is wrong with its
/// pieces if not. Copies nothing (a pipeline's creation runs it on the
/// title's thread).
pub fn check(container: &[u8]) -> Result<(), ShaderError> {
    let c = crate::Container::parse(container)?;
    let part = c
        .part(FourCC::PIECES)
        .ok_or(Malformed("no part made from pieces"))?;
    read(part.data).map(|_| ())
}

/// The program in the data of a part of ours.
pub(crate) fn program(part: &[u8]) -> Result<NativeProgram, ShaderError> {
    let (mut program, code) = read(part)?;
    program.code = code.to_vec();
    Ok(program)
}

/// The program in the data of a part of ours, without its code, and the
/// code where it lies in the part.
fn read(part: &[u8]) -> Result<(NativeProgram, &[u8]), ShaderError> {
    const SHORT: ShaderError = Malformed("a part made from pieces is cut short");
    if part.get(..4) != Some(&MAGIC[..]) {
        return Err(Malformed("not a part made from pieces"));
    }
    let head = part.get(..HEAD).ok_or(SHORT)?;
    let (stage, wave64) = (head[4], head[5] & WAVE32 == 0);
    let vgprs = u16_at(head, 6)?;
    let user_sgprs = u32_at(head, 8)?;
    let words: [u32; 5] = std::array::from_fn(|i| u32_at(head, 12 + 4 * i).unwrap_or(0));
    // The three stretches after the header, by their sizes.
    let mut at = HEAD;
    let mut take = |size_at: usize| -> Result<&[u8], ShaderError> {
        let size = u32_at(head, size_at)? as usize;
        let piece = part
            .get(at..at.checked_add(size).ok_or(SHORT)?)
            .ok_or(SHORT)?;
        at += size;
        Ok(piece)
    };
    let (extra, code, meta) = (take(0x20)?, take(0x24)?, take(0x28)?);
    let (tag, records_at, uncounted) = match stage {
        COMPUTE => COMPUTE_METADATA,
        VERTEX | PIXEL => GRAPHICS_METADATA,
        _ => return Err(Malformed("a part made from pieces of an unknown stage")),
    };
    if vgprs > 256 || user_sgprs > 32 {
        return Err(Malformed("implausible register counts"));
    }
    // The metadata must be of the stage's kind and count the code.
    if u32_at(meta, 0)? != tag {
        return Err(Malformed("a metadata piece of an unknown kind"));
    }
    if (u32_at(meta, 8)? as usize)
        .checked_mul(4)
        .and_then(|n| n.checked_add(uncounted))
        != Some(code.len())
    {
        return Err(Malformed(
            "the metadata's code size disagrees with the code piece",
        ));
    }
    // Whole records, the shader hash last.
    let recs = meta.get(records_at..).unwrap_or(&[]);
    let (chunks, rest) = recs.as_chunks::<RECORD>();
    if !rest.is_empty() || chunks.last().is_none_or(|e| e[0] != 0x43) {
        return Err(Malformed(
            "metadata records that do not end in the shader hash",
        ));
    }
    let rules = RecordRules {
        constants: true,
        strides: true,
    };
    let mut r = native::records(recs, rules)?;
    if r.heap_sampler {
        // A sampler from the sampler heap, whose base no record places: it
        // was the last user SGPR in the eleven pixel programs seen with
        // one, all of one root signature.
        if stage == COMPUTE {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a sampler from the heap in a compute program made from pieces",
            });
        }
        let last = user_sgprs
            .checked_sub(1)
            .ok_or(Malformed("a heap sampler without user SGPRs"))?;
        if r.layout.iter().any(|p| u32::from(p.user_data) >= last) {
            return Err(Malformed("no user SGPR is left for the sampler heap"));
        }
        r.layout.push(RootParam {
            root: RootType::SamplerHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: last as u8,
            slot: u8::MAX,
        });
    }
    native::size_constants(&mut r.layout, user_sgprs)?;
    let io = match stage {
        COMPUTE => StageIo::Cs(ComputeIo {
            rsrc2: words[0],
            threads: [words[1], words[2], words[3]],
        }),
        VERTEX => {
            // s0 and s1 are the vertex buffer table's address, s2 the base
            // vertex: no root parameter may claim them.
            if r.layout.iter().any(|p| p.user_data < 3) || user_sgprs < 3 {
                return Err(Malformed(
                    "a root parameter in the vertex program's system SGPRs",
                ));
            }
            r.layout.push(RootParam {
                root: RootType::Other(VERTEX_BUFFERS),
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: u8::MAX,
            });
            let (elements, rest) = extra.as_chunks::<ELEMENT>();
            if !rest.is_empty() || elements.len() != words[2] as usize || words[0] > 32 {
                return Err(Malformed(
                    "a vertex part whose elements disagree with their count",
                ));
            }
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: (0..words[0] as u8)
                    .map(|n| VsOutput {
                        usage: Usage::Generic,
                        row: n,
                        param: n,
                        fp16: false,
                        pack2: false,
                    })
                    .collect(),
                fetch: elements
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let word = |k: usize| u32_at(e, NAME + 4 * k).unwrap_or(0);
                        let format = word(1);
                        VsFetch {
                            row: i as u8,
                            slot: word(2),
                            offset: word(3),
                            // (0 for a format whose fetch is not known: no
                            // load then lies inside it)
                            bytes: element_shape(format).map_or(0, |s| s.0),
                            format,
                        }
                    })
                    .collect(),
            }
        }
        _ => {
            let (controls, rest) = extra.as_chunks::<4>();
            if !rest.is_empty() || controls.len() != words[4] as usize || controls.len() > 32 {
                return Err(Malformed(
                    "a pixel part whose inputs disagree with their count",
                ));
            }
            StageIo::Ps {
                inputs: controls
                    .iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let c = u32::from_le_bytes(*c);
                        PsInput {
                            usage: Usage::Generic,
                            row: (c & 0x1f) as u8,
                            mask: 0xf,
                            attr: k as u8,
                            flat: c & 0x400 != 0,
                            fp16: false,
                            pack2: false,
                            row2: 0,
                        }
                    })
                    .collect(),
                input_addr: words[0],
                db_shader_control: words[1],
                // The exports are final: nothing to patch.
                patches: Vec::new(),
            }
        }
    };
    let program = NativeProgram {
        stage: match stage {
            COMPUTE => HwStage::Cs,
            VERTEX => HwStage::Vs,
            _ => HwStage::Ps,
        },
        wave64,
        vgprs,
        layout: r.layout,
        used: r.used,
        io,
        code: Vec::new(),
    };
    Ok((program, code))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{ResourceType, UsedResource};

    /// `s_endpgm`.
    const END: u32 = 0xBF81_0000;

    fn put(to: &mut [u8], at: usize, v: u32) {
        to[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// A 0x41 record: resource type, root type, table offset, register,
    /// stride, descriptor dwords, user-data dword, root slot.
    fn resource(
        kind: u8,
        root: u8,
        offset: u32,
        register: u32,
        stride: u32,
        dwords: u8,
        user: u8,
        slot: u8,
    ) -> [u8; 32] {
        let mut e = [0u8; 32];
        e[0] = 0x41;
        (e[4], e[5]) = (kind, root);
        put(&mut e, 8, offset);
        put(&mut e, 12, register);
        put(&mut e, 16, stride);
        (e[20], e[21], e[22]) = (dwords, user, slot);
        put(&mut e, 24, u32::MAX);
        e
    }

    /// A 0x42 record of `class` for `register`.
    fn declaration(class: u8, kind: u8, register: u32) -> [u8; 32] {
        let mut e = [0u8; 32];
        (e[0], e[8], e[9], e[10]) = (0x42, 1, class, kind);
        put(&mut e, 12, register);
        e
    }

    /// The 0x43 record, with a hash of our own.
    fn hash() -> [u8; 32] {
        let mut e = [0u8; 32];
        e[0] = 0x43;
        for (i, b) in e[4..20].iter_mut().enumerate() {
            *b = i as u8 + 1;
        }
        e
    }

    /// A metadata piece for a stage with `code` bytes of code.
    fn metadata(kind: (u32, usize, usize), code: usize, records: &[[u8; 32]]) -> Vec<u8> {
        let mut m = vec![0u8; kind.1];
        put(&mut m, 0, kind.0);
        put(&mut m, 8, ((code - kind.2) / 4) as u32);
        m.extend(records.iter().flatten());
        let after = (m.len() - 0x0c) as u32;
        put(&mut m, 4, after);
        m
    }

    /// A code piece of `dwords` dwords that ends the program at once.
    fn code(dwords: usize) -> Vec<u8> {
        let mut c = END.to_le_bytes().to_vec();
        c.resize(4 * dwords, 0);
        c
    }

    /// A compute packet: threads 8x8x1, 2 VGPR granules, `user` user SGPRs
    /// and the group id's x after them.
    fn compute_packet(user: u32) -> [u8; COMPUTE_PACKET] {
        let mut p = [0u8; COMPUTE_PACKET];
        p[C_THREADS..C_THREADS + 6].copy_from_slice(&[8, 0, 8, 0, 1, 0]);
        put(&mut p, C_RSRC1, 1);
        put(&mut p, C_RSRC2, user << 1 | 1 << 7);
        p
    }

    /// The program of a filled container, once [`check`] has passed it.
    fn checked(bytes: &[u8]) -> Result<NativeProgram, ShaderError> {
        check(bytes)?;
        let c = crate::Container::parse(bytes)?;
        program(c.part(FourCC::PIECES).unwrap().data)
    }

    /// Fills a draft's two pieces.
    fn fill(mut d: Draft, code: &[u8], meta: &[u8]) -> Vec<u8> {
        d.bytes[d.code].copy_from_slice(code);
        d.bytes[d.metadata].copy_from_slice(meta);
        d.bytes
    }

    #[test]
    fn a_compute_program_from_its_pieces() {
        let code = code(4);
        // A table in s4 with a texture at its start, a constant in s5.
        let meta = metadata(
            COMPUTE_METADATA,
            code.len(),
            &[
                resource(0, 3, 0, 7, 4, 8, 4, 0),
                resource(2, 0, 0, 9, 0, 4, 5, 1),
                declaration(0, 2, 7),
                hash(),
            ],
        );
        let draft = compute(&compute_packet(6), code.len(), meta.len()).unwrap();
        let bytes = fill(draft, &code, &meta);
        let p = checked(&bytes).unwrap();
        assert_eq!((p.stage, p.wave64, p.vgprs), (HwStage::Cs, true, 8));
        assert_eq!(p.code, code);
        let param = |root, user_data, slot, base_register| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register,
            user_data,
            slot,
        };
        assert_eq!(
            p.layout,
            [
                param(RootType::Table, 4, 0, 7),
                param(RootType::Constants, 5, 1, 9)
            ]
        );
        assert_eq!(
            p.used[0],
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 7,
                stride: 4,
                dwords: 8,
                user_data: 4,
                slot: 0
            }
        );
        let StageIo::Cs(cs) = p.io else {
            panic!("{:?}", p.io)
        };
        assert_eq!((cs.threads, cs.user_sgprs()), ([8, 8, 1], 6));
        // The same pieces give the same bytes, other pieces other bytes.
        let again = fill(
            compute(&compute_packet(6), code.len(), meta.len()).unwrap(),
            &code,
            &meta,
        );
        assert_eq!(bytes, again);
        let other = fill(
            compute(&compute_packet(7), code.len(), meta.len()).unwrap(),
            &code,
            &meta,
        );
        assert_ne!(bytes, other);
        // And the program is found the way a container's is.
        let c = crate::Container::parse(&bytes).unwrap();
        let found = crate::native_programs(c.native_part().unwrap().data).unwrap();
        assert_eq!(found, [p]);
    }

    #[test]
    fn root_constants_run_to_the_next_parameter() {
        let code = code(1);
        // Constants in s4, a table in s6: the constants are s4 and s5.
        let records = [
            resource(2, 0, 0, 9, 0, 4, 4, 0),
            resource(3, 3, 0, 2, 4, 8, 6, 1),
            hash(),
        ];
        let meta = metadata(COMPUTE_METADATA, code.len(), &records);
        let p = checked(&fill(
            compute(&compute_packet(7), 4, meta.len()).unwrap(),
            &code,
            &meta,
        ))
        .unwrap();
        assert_eq!(
            (p.layout[0].root, p.layout[0].sgprs),
            (RootType::Constants, 2)
        );
        // A parameter past the user SGPRs is refused.
        let short = fill(
            compute(&compute_packet(6), 4, meta.len()).unwrap(),
            &code,
            &meta,
        );
        assert!(checked(&short).is_err());
    }

    #[test]
    fn compute_pieces_of_wrong_sizes_are_refused() {
        let packet = compute_packet(6);
        assert!(compute(&packet[..0x18], 4, 0x38).is_err(), "short packet");
        assert!(compute(&packet, 0, 0x38).is_err(), "no code");
        assert!(compute(&packet, 6, 0x38).is_err(), "half a dword");
        assert!(compute(&packet, MAX_CODE + 4, 0x38).is_err(), "absurd code");
        assert!(compute(&packet, 4, 0x18).is_err(), "no record");
        assert!(compute(&packet, 4, 0x39).is_err(), "part of a record");
        assert!(
            compute(&packet, 4, MAX_METADATA + 0x18).is_err(),
            "absurd metadata"
        );
        let mut threads = packet;
        threads[C_THREADS + 1] = 0x40;
        assert!(compute(&threads, 4, 0x38).is_err(), "absurd thread group");
        threads[C_THREADS..C_THREADS + 2].copy_from_slice(&[0, 0]);
        assert!(compute(&threads, 4, 0x38).is_err(), "no threads");
    }

    #[test]
    fn pieces_that_disagree_are_refused() {
        let code = code(2);
        let good = metadata(COMPUTE_METADATA, 8, &[hash()]);
        let make = |meta: &[u8]| {
            fill(
                compute(&compute_packet(6), 8, meta.len()).unwrap(),
                &code,
                meta,
            )
        };
        assert!(checked(&make(&good)).is_ok());
        // A code size that is not the code piece's.
        let mut wrong = good.clone();
        put(&mut wrong, 8, 3);
        assert!(checked(&make(&wrong)).is_err());
        // A graphics stage's tag on a compute stage's metadata.
        let mut tag = good.clone();
        put(&mut tag, 0, GRAPHICS_METADATA.0);
        assert!(checked(&make(&tag)).is_err());
        // Records of an unknown tag, and records without the last one.
        let mut unknown = hash();
        unknown[0] = 0x44;
        assert!(checked(&make(&metadata(COMPUTE_METADATA, 8, &[unknown, hash()]))).is_err());
        assert!(
            checked(&make(&metadata(
                COMPUTE_METADATA,
                8,
                &[declaration(0, 2, 1)]
            )))
            .is_err()
        );
        // A root type nothing is known of.
        let odd = resource(0, 7, 0, 1, 0, 4, 4, 0);
        assert!(checked(&make(&metadata(COMPUTE_METADATA, 8, &[odd, hash()]))).is_err());
        // A container cut anywhere does not parse.
        let whole = make(&good);
        for len in [0, 0x1f, 0x40, whole.len() - 1] {
            assert!(checked(&whole[..len]).is_err(), "{len:#x}");
        }
        // Nor does a part whose sizes reach past it.
        let c = crate::Container::parse(&whole).unwrap();
        let mut part = c.part(FourCC::PIECES).unwrap().data.to_vec();
        put(&mut part, 0x24, u32::MAX);
        assert!(program(&part).is_err());
    }

    /// A graphics packet: `interpolants` as (vertex parameter, flat), with
    /// a pixel stage when `pixel`, and `params` parameter exports.
    fn graphics_packet(pixel: bool, params: u32, interpolants: &[(u32, bool)]) -> Vec<u8> {
        let slots = interpolants.len().next_multiple_of(2);
        let second = G_SECOND_HEAD + 4 * slots + G_TARGET_INFO;
        let mut p = vec![0u8; G_SECOND + second];
        p[G_FIXED_SIZE..G_FIXED_SIZE + 2].copy_from_slice(&G_FIXED.to_le_bytes());
        p[G_SECOND_SIZE..G_SECOND_SIZE + 2].copy_from_slice(&(second as u16).to_le_bytes());
        // 2 VGPR granules and 10 user SGPRs for the vertex stage.
        put(&mut p, G_RSRC1_VS, 1);
        put(&mut p, G_RSRC2_VS, 10 << 1);
        put(
            &mut p,
            G_VS_OUT_CONFIG,
            if params == 0 { 0x80 } else { (params - 1) << 1 },
        );
        put(&mut p, G_POS_FORMAT, 4);
        if pixel {
            // 4 VGPR granules, 10 user SGPRs, perspective centre
            // barycentrics, the first target as 16-bit floats.
            put(&mut p, G_RSRC1_PS, 3);
            put(&mut p, G_RSRC2_PS, 10 << 1);
            put(&mut p, G_PS_INPUT_ADDR, 2);
            put(&mut p, G_PS_IN_CONTROL, interpolants.len() as u32);
            put(&mut p, G_DB_SHADER_CONTROL, 0x10);
            put(&mut p, G_COL_FORMAT, 4);
            for (k, &(param, flat)) in interpolants.iter().enumerate() {
                put(&mut p, G_INPUT_CNTL + 4 * k, param | u32::from(flat) << 10);
            }
        }
        p
    }

    /// A state piece: one target of format 28, triangles, and `elements`
    /// as (name, format, byte offset).
    fn state_piece(elements: &[(&str, u32, u32)]) -> Vec<u8> {
        let mut s = vec![0u8; S_BLOCK];
        s[S_TARGETS] = 28;
        (s[S_COUNT], s[S_TOPOLOGY]) = (1, 3);
        put(&mut s, S_SAMPLE_MASK, u32::MAX);
        put(&mut s, S_SAMPLES, 1);
        s.extend((elements.len() as u32).to_le_bytes());
        for &(name, format, offset) in elements {
            s.extend((name.len() as u32 + 1).to_le_bytes());
            s.extend(name.as_bytes());
            s.push(0);
            for v in [0, format, 0, offset, 0, 0] {
                s.extend(v.to_le_bytes());
            }
        }
        let block = (s.len() - S_BLOCK) as u32;
        put(&mut s, S_BLOCK_SIZE, block);
        s.resize(s.len().next_multiple_of(4), 0);
        s
    }

    const ELEMENTS: [(&str, u32, u32); 3] =
        [("POSITION", 6, 0), ("COLOR", 28, 12), ("TEXCOORD", 16, 16)];

    #[test]
    fn a_graphics_pipeline_from_its_pieces() {
        let packet = graphics_packet(true, 3, &[(0, false), (2, true), (1, false)]);
        let g = Graphics::parse(&packet, &state_piece(&ELEMENTS), true).unwrap();
        assert!(g.has_pixel());
        assert_eq!(
            (g.targets, g.target_formats[0], g.topology, g.depth_format),
            (1, 28, 3, 0)
        );
        assert_eq!(
            g.elements[1],
            Element {
                name: "COLOR".into(),
                format: 28,
                offset: 12
            }
        );
        let code = code(11);
        // A table in s8 and a sampler from the heap, which gets s9.
        let meta = metadata(
            GRAPHICS_METADATA,
            code.len(),
            &[
                resource(0, 3, 0, 1, 4, 8, 8, 0),
                declaration(0, 2, 1),
                declaration(3, 14, 0),
                hash(),
            ],
        );
        let vs = checked(&fill(
            g.vertex(code.len(), meta.len()).unwrap(),
            &code,
            &meta,
        ))
        .unwrap();
        assert_eq!((vs.stage, vs.vgprs), (HwStage::Vs, 8));
        let roots: Vec<_> = vs
            .layout
            .iter()
            .map(|p| (p.root, p.user_data, p.sgprs))
            .collect();
        assert_eq!(
            roots,
            [
                (RootType::Table, 8, 1),
                (RootType::SamplerHeap, 9, 1),
                (RootType::Other(VERTEX_BUFFERS), 0, 2)
            ]
        );
        let StageIo::Vs { outputs, fetch, .. } = &vs.io else {
            panic!("{:?}", vs.io)
        };
        assert_eq!(
            outputs.iter().map(|o| (o.row, o.param)).collect::<Vec<_>>(),
            [(0, 0), (1, 1), (2, 2)]
        );
        assert_eq!(
            fetch[2],
            VsFetch {
                row: 2,
                slot: 0,
                offset: 16,
                bytes: 8,
                format: 16
            }
        );
        let ps = checked(&fill(
            g.pixel(code.len(), meta.len()).unwrap(),
            &code,
            &meta,
        ))
        .unwrap();
        assert_eq!((ps.stage, ps.vgprs), (HwStage::Ps, 16));
        let StageIo::Ps {
            inputs,
            input_addr,
            db_shader_control,
            patches,
        } = &ps.io
        else {
            panic!("{:?}", ps.io)
        };
        assert_eq!(
            (*input_addr, *db_shader_control, patches.len()),
            (2, 0x10, 0)
        );
        assert_eq!(
            inputs
                .iter()
                .map(|i| (i.attr, i.row, i.flat))
                .collect::<Vec<_>>(),
            [(0, 0, false), (1, 2, true), (2, 1, false)]
        );
        // The two stages agree on every parameter: name, register, type.
        let rows = |bytes: &[u8], fourcc| {
            let c = crate::Container::parse(bytes).unwrap();
            let part = c.part(fourcc).unwrap().data;
            let count = u32_at(part, 0).unwrap() as usize;
            (0..count)
                .map(|i| {
                    let at = 8 + 32 * i;
                    let word = |k: usize| u32_at(part, at + 4 * k).unwrap();
                    (word(2), word(3), word(4), word(5))
                })
                .collect::<Vec<_>>()
        };
        let vs_bytes = fill(g.vertex(code.len(), meta.len()).unwrap(), &code, &meta);
        let ps_bytes = fill(g.pixel(code.len(), meta.len()).unwrap(), &code, &meta);
        let out = rows(&vs_bytes, FourCC::OSG1);
        // (index, system value, type, register): parameter 2 is flat.
        assert_eq!(
            out,
            [(0, 0, 3, 0), (1, 0, 3, 1), (2, 0, 1, 2), (0, 1, 3, 3)]
        );
        assert_eq!(rows(&ps_bytes, FourCC::ISG1), out[..3]);
        assert_eq!(rows(&ps_bytes, FourCC::OSG1), [(0, 64, 6, 0)]);
        assert_eq!(
            rows(&vs_bytes, FourCC::ISG1),
            [(0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2)]
        );
    }

    #[test]
    fn a_pipeline_without_a_pixel_stage() {
        let packet = graphics_packet(false, 0, &[]);
        let g = Graphics::parse(&packet, &state_piece(&[]), false).unwrap();
        assert!(!g.has_pixel() && g.elements.is_empty());
        assert!(g.pixel(44, 0x30).is_err());
        // Its vertex program exports the position alone.
        let code = code(11);
        let meta = metadata(GRAPHICS_METADATA, code.len(), &[hash()]);
        let vs = checked(&fill(
            g.vertex(code.len(), meta.len()).unwrap(),
            &code,
            &meta,
        ))
        .unwrap();
        let StageIo::Vs { outputs, fetch, .. } = &vs.io else {
            panic!("{:?}", vs.io)
        };
        assert!(outputs.is_empty() && fetch.is_empty());
        // The description and the packet must agree on the pixel stage.
        assert!(Graphics::parse(&packet, &state_piece(&[]), true).is_err());
        let with = graphics_packet(true, 1, &[(0, false)]);
        assert!(Graphics::parse(&with, &state_piece(&[]), false).is_err());
    }

    #[test]
    fn graphics_packets_not_understood_are_refused() {
        let state = state_piece(&ELEMENTS);
        let good = graphics_packet(true, 2, &[(0, false), (1, false)]);
        assert!(Graphics::parse(&good, &state, true).is_ok());
        let refused = |change: &dyn Fn(&mut Vec<u8>)| {
            let mut p = good.clone();
            change(&mut p);
            Graphics::parse(&p, &state, true).is_err()
        };
        assert!(refused(&|p| p.truncate(0x100)), "truncated");
        assert!(
            refused(&|p| p.truncate(p.len() - 4)),
            "sizes that do not add up"
        );
        assert!(refused(&|p| p.resize(MAX_PACKET + 4, 0)), "absurd length");
        assert!(refused(&|p| p[G_FIXED_SIZE] = 0x20), "another fixed block");
        assert!(
            refused(&|p| put(p, G_PS_IN_CONTROL, 9)),
            "more interpolants than controls"
        );
        assert!(refused(&|p| put(p, G_PS_IN_CONTROL, 2 | 1 << 15)), "wave32");
        assert!(
            refused(&|p| put(p, G_INPUT_CNTL, 5)),
            "a parameter not exported"
        );
        assert!(
            refused(&|p| put(p, G_INPUT_CNTL, 1 << 19)),
            "16-bit interpolation"
        );
        assert!(
            refused(&|p| put(p, G_INPUT_CNTL + 4, 1 << 10)),
            "one parameter, two types"
        );
        assert!(
            refused(&|p| put(p, G_COL_FORMAT, 5)),
            "a 16-bit integer export"
        );
        assert!(
            refused(&|p| put(p, G_DB_SHADER_CONTROL, 0x110)),
            "a coverage export"
        );
        assert!(
            refused(&|p| put(p, G_DB_SHADER_CONTROL, 0x11)),
            "depth export without its format"
        );
        assert!(refused(&|p| put(p, G_POS_FORMAT, 0x44)), "clip distances");
        assert!(
            refused(&|p| put(p, G_VS_OUT_CONFIG, 0x100)),
            "unknown output bits"
        );
        assert!(refused(&|p| put(p, G_DEPTH_BIAS + 4, 1)), "depth bias");
    }

    #[test]
    fn state_pieces_not_understood_are_refused() {
        let packet = graphics_packet(false, 0, &[]);
        let good = state_piece(&ELEMENTS);
        let refused = |change: &dyn Fn(&mut Vec<u8>)| {
            let mut s = good.clone();
            change(&mut s);
            Graphics::parse(&packet, &s, false).is_err()
        };
        assert!(!refused(&|_| ()));
        assert!(refused(&|s| s.truncate(S_BLOCK)), "no layout block");
        assert!(refused(&|s| s.truncate(s.len() - 4)), "a cut layout");
        assert!(refused(&|s| s.resize(MAX_STATE + 4, 0)), "absurd length");
        assert!(refused(&|s| put(s, S_BLOCK, 3000)), "absurd element count");
        assert!(refused(&|s| put(s, S_BLOCK, 4)), "more elements than bytes");
        assert!(
            refused(&|s| put(s, S_BLOCK + 4, 4000)),
            "a name past the piece"
        );
        assert!(
            refused(&|s| s[S_BLOCK + 8] = b' '),
            "a name that is no semantic"
        );
        assert!(
            refused(&|s| s[S_BLOCK + 8 + 7] = b'1'),
            "a name with its index in it"
        );
        assert!(refused(&|s| s[S_COUNT] = 9), "nine targets");
        assert!(
            refused(&|s| s[S_TARGETS + 3] = 28),
            "a target past the count"
        );
        assert!(refused(&|s| s[S_TOPOLOGY] = 2), "lines");
        assert!(refused(&|s| put(s, S_SAMPLES, 4)), "multisampling");
        // The element's first word after its name (index, slot, class or
        // step rate: all must be zero).
        assert!(refused(&|s| put(s, S_BLOCK + 8 + 9, 1)));
        // An empty layout is one.
        assert!(Graphics::parse(&packet, &state_piece(&[]), false).is_ok());
    }

    /// Every description captured by `VICTORY_PIPELINE_DUMP` in the folder
    /// that variable names, through the parsers above and the translator.
    /// Skipped when it is not set: the files are a title's, and stay
    /// outside the repository. Prints counts and each distinct failure.
    /// Where `dxil.dll` loads, each translation is validated; where the
    /// software adapter opens, each whole pipeline is made on it (which
    /// links the two stages' signatures); the card is not touched. All of
    /// it twice: as on a host of 64-lane waves and as on one of 32.
    #[test]
    fn captured_descriptions_parse_and_translate() {
        use std::collections::BTreeMap;
        let Some(folder) = std::env::var_os("VICTORY_PIPELINE_DUMP").map(std::path::PathBuf::from)
        else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&folder) else {
            panic!(
                "VICTORY_PIPELINE_DUMP names no folder: {}",
                folder.display()
            );
        };
        // The files of each call, by "<slot>-<call>".
        let mut calls: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in entries
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
        {
            if let Some(call) = name.get(..7).filter(|c| c.as_bytes()[3] == b'-') {
                calls.entry(call.to_owned()).or_default().push(name);
            }
        }
        // A description rebuilt from its files: the words, and each pair's
        // bytes where the dump has them (a blob whole, or the 256 bytes
        // taken at a word that had no length beside it).
        struct Call {
            words: Vec<u64>,
            pieces: BTreeMap<usize, Vec<u8>>,
        }
        let load = |call: &str, names: &[String]| -> Option<Call> {
            let read = |name: &str| std::fs::read(folder.join(name)).ok();
            let desc = read(&format!("{call}-desc.bin"))?;
            let words: Vec<u64> = desc
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            let mut pieces = BTreeMap::new();
            for name in names {
                let rest = &name[8..];
                if let Some(n) = rest
                    .strip_prefix("blob")
                    .and_then(|r| r.get(..2)?.parse::<usize>().ok())
                {
                    // Pair n is description words 1 + 2n and 2 + 2n.
                    pieces.insert(1 + 2 * n, read(name)?);
                } else if let Some(n) = rest
                    .strip_prefix("word")
                    .and_then(|r| r.get(..2)?.parse::<usize>().ok())
                {
                    pieces.entry(n).or_insert(read(name)?);
                }
            }
            Some(Call { words, pieces })
        };
        #[derive(Default)]
        struct Tally {
            calls: u32,
            parsed: u32,
            programs: u32,
            translated: u32,
            valid: u32,
            pipelines: u32,
            made: u32,
            why: BTreeMap<String, u32>,
        }
        #[cfg(windows)]
        let mut validator = {
            use crate::emit::validator::Validator;
            Validator::load(Validator::default_path().as_deref()).ok()
        };
        #[cfg(windows)]
        let host = crate::emit::host::Host::new(true).ok();
        #[cfg(windows)]
        println!(
            "validator: {}; software adapter: {}",
            if validator.is_some() {
                "loaded"
            } else {
                "not found"
            },
            if host.is_some() {
                "opened"
            } else {
                "not opened"
            }
        );
        for lanes in [64u32, 32] {
            crate::emit::with_test_host_wave(lanes, || {
                println!("as on a host of {lanes}-lane waves:");
                let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
                for (call, names) in &calls {
                    let method = if call.starts_with("218") {
                        "0x218"
                    } else {
                        "0x1c8"
                    };
                    let t = tallies.entry(method).or_default();
                    t.calls += 1;
                    let note = |t: &mut Tally, why: String| *t.why.entry(why).or_default() += 1;
                    let Some(c) = load(call, names) else {
                        note(t, "the dump has no description".into());
                        continue;
                    };
                    // The piece at description word `at`, cut to the length the
                    // description gives it; none if the dump has fewer bytes.
                    let piece = |at: usize| -> Option<&[u8]> {
                        let len = *c.words.get(at + 1)? as usize;
                        c.pieces.get(&at)?.get(..len)
                    };
                    // One stage: its draft filled from the dump, then translated.
                    let mut stage = |t: &mut Tally,
                                     what: &str,
                                     draft: Result<Draft, Refusal>,
                                     at: usize| {
                        let draft = match draft {
                            Ok(d) => d,
                            Err(why) => {
                                note(t, format!("{what} refused: {why}"));
                                return None;
                            }
                        };
                        let (Some(code), Some(meta)) = (piece(at), piece(at + 2)) else {
                            note(t, format!("{what}: the dump does not hold the whole code"));
                            return None;
                        };
                        let bytes = fill(draft, code, meta);
                        if let Err(e) = checked(&bytes) {
                            note(t, format!("{what} does not parse: {e}"));
                            return None;
                        }
                        t.programs += 1;
                        match crate::emit::translate(&bytes) {
                            Ok(shader) => {
                                t.translated += 1;
                                #[cfg(windows)]
                                if let Some(v) = &mut validator {
                                    match v.validate(&mut shader.dxil.clone()) {
                                        Ok(()) => t.valid += 1,
                                        Err(e) => {
                                            let first = e.lines().next().unwrap_or("").to_owned();
                                            note(t, format!("{what} does not validate: {first}"));
                                        }
                                    }
                                }
                                Some(shader)
                            }
                            Err(e) => {
                                // (without the offset, so that one cause is one line)
                                let why = match e {
                                    ShaderError::Unsupported { what, .. } => what.to_owned(),
                                    e => e.to_string(),
                                };
                                note(t, format!("{what} does not translate: {why}"));
                                None
                            }
                        }
                    };
                    let len = |at: usize| c.words.get(at + 1).map_or(0, |&n| n as usize);
                    if method == "0x218" {
                        let Some(packet) = piece(1) else {
                            note(t, "the dump has no packet".into());
                            continue;
                        };
                        t.parsed += 1;
                        if let Some(cs) = stage(t, "compute", compute(packet, len(5), len(7)), 5) {
                            t.pipelines += 1;
                            #[cfg(windows)]
                            if let Some(host) = &host {
                                match host.compute_pipeline(&cs) {
                                    Ok(_) => t.made += 1,
                                    Err(e) => note(
                                        t,
                                        format!(
                                            "compute pipeline refused by the software adapter: {e}"
                                        ),
                                    ),
                                }
                            }
                            let _ = cs;
                        }
                        continue;
                    }
                    let (Some(packet), Some(state)) = (piece(1), piece(3)) else {
                        note(t, "the dump has no packet or state piece".into());
                        continue;
                    };
                    let pixel = c.words.get(5).is_some_and(|&w| w != 0);
                    let g = match Graphics::parse(packet, state, pixel) {
                        Ok(g) => g,
                        Err(why) => {
                            note(t, format!("pipeline refused: {why}"));
                            continue;
                        }
                    };
                    t.parsed += 1;
                    let vs = stage(t, "vertex", g.vertex(len(9), len(11)), 9);
                    let ps = if pixel {
                        stage(t, "pixel", g.pixel(len(5), len(7)), 5)
                    } else {
                        None
                    };
                    let (Some(vs), true) = (vs, ps.is_some() || !pixel) else {
                        continue;
                    };
                    t.pipelines += 1;
                    // With both stages: a pipeline on the software adapter, with
                    // the vertex program's own inputs as the input layout.
                    #[cfg(windows)]
                    if let (Some(host), Some(ps)) = (&host, &ps) {
                        match host.pipeline(&vs, ps) {
                            Ok(_) => t.made += 1,
                            Err(e) => note(
                                t,
                                format!("graphics pipeline refused by the software adapter: {e}"),
                            ),
                        }
                    }
                    let _ = vs;
                }
                for (method, t) in &tallies {
                    println!(
                        "{method}: {} calls, {} descriptions parsed, {} programs built, {} translated ({} valid), {} pipelines whole ({} made on the software adapter)",
                        t.calls, t.parsed, t.programs, t.translated, t.valid, t.pipelines, t.made
                    );
                    for (why, n) in &t.why {
                        println!("    {n:3} x {why}");
                    }
                }
                assert!(
                    !tallies.is_empty(),
                    "no captured description in {}",
                    folder.display()
                );
            });
        }
    }
}
