//! Native GFX10 programs that the console compiler stores in a `DXIL` part.
//!
//! An `XDAT` index after the program header lists blobs stored around the
//! bitcode; one or more of them hold native programs, zlib-packed (a vertex
//! shader can carry two: compiled for use with and without a geometry
//! shader). Layout as observed in shipped titles (index offsets are relative
//! to the part data):
//!
//! ```text
//! +0x18 "XDAT"  u32 entry count, u32 entries offset (0x24), then per entry:
//!               u32 kind, u32 offset, u32 size
//! kind 0x0c     uncompressed "FSSM" blob
//! kind 0x0d     "XBCH" u32 version (3), u64 packed size, u64 unpacked size,
//!               zlib stream -> "FSSM" blob
//! unpacked      "FSSM" u32 version (0x101), u64 payload size, payload
//! payload +0x00 u64 graphics slot count (6)
//!         +0x08 u64 record offsets: LS HS ES GS VS PS (0x08..0x30),
//!               mesh 0x48, compute 0x60; unused ones are 0xDEADBEEF_DEADBEEF
//! record  +0x00 u64 code offset, +0x08 u64 code size (payload-relative),
//!         +0x14 u32 flags: bit 5 wave32 (else wave64)
//!         +0x28 u64 root layout table, +0x30 u64 used-resource table
//!               (payload-relative), +0x38 u32 and +0x3c u32 their counts
//!         +0x4c u32 LDS bytes (`lds_byte_size`)
//!         +0x50 u32 bits 0-9 VGPRs used, bits 20-27 SGPRs used
//! layout  20 B: u32 root type, u32 table offset (dwords), u32 user SGPRs,
//!               u32 base register, u8 first user-data dword, u8 root slot,
//!               u16 0
//! used    20 B: u8 resource type, u8 root type, u16 0, u32 table offset
//!               (dwords), u32 lower bound, u32 stride, u8 descriptor
//!               dwords, u8 first user-data dword, u8 root slot, u8 0
//! ```
//!
//! Stage data follows the record; offsets from its end (E = record + 0x54):
//!
//! ```text
//! PS  E+0x24  32 inputs of 8 B: u8 usage, u8 signature row, u8 channel
//!             mask, u8 attribute (v_interp `attr`), u8 default, u8 second
//!             row (of a packed pair), u8 parameter slot, u8 flags (bit 0
//!             flat, 1 linear, 3 fp16, 4 fp16 packed); E+0x124 u32 count
//!     E+0x134 u32 SPI_PS_INPUT_ADDR, E+0x138 u32 DB_SHADER_CONTROL
//!     E+0x148 u32 export patch size (dwords), E+0x14c 8 patches, one per
//!             MRT: u32 code offset (dwords, -1 none), then the export code
//!             for each color format in 10 slots of 10 dwords (ZERO, 32_R,
//!             32_GR, 32_AR, FP16, UNORM16, SNORM16, UINT16, SINT16, 32_ABGR)
//! VS  E+0x24  32 inputs of 5 B: u8 usage, u8 row, u8 channel mask, u8 VGPR
//!             of channel 0 (channel k in VGPR + k), u8 highest channel;
//!             E+0xc4 u32 count
//!     E+0xc8  32 outputs of 6 B: u8 usage, u8 row, u8 parameter (`exp
//!             param<n>`), u8 row again, u8 slot, u8 flags (bit 0 fp16,
//!             bit 1 two fp16 values per dword); E+0x188 u32 count
//! ES  same 0x1f4-byte vertex layout as VS (inputs at E+0x24/E+0xc4, outputs
//!     written to the ES->GS LDS ring at E+0xc8/E+0x188 including SV_Position
//!     as usage 0); E+0x1ac u32 ESGS ring itemsize in dwords (`outputs * 4`)
//! LS  same vertex-stage layout as ES/VS (inputs at E+0x24/E+0xc4, outputs
//!     at E+0xc8/E+0x188, itemsize at E+0x1ac, LDS bytes at record+0x4c)
//! GS  0x2a4-byte stage data:
//!     E+0x28  u8 output topology (2 = triangle strip), u8 input vertices (3)
//!     E+0x34  u64 GS copy shader code offset (payload-relative),
//!     E+0x3c  u64 GS copy shader code size (0 when absent)
//!     E+0x150 32 outputs of 6 B (as VS_OUTPUTS); E+0x210 u32 count
//!     E+0x220 u32 SPI_VS_OUT_CONFIG, E+0x224 u32 SPI_SHADER_POS_FORMAT,
//!             E+0x228 u32 PA_CL_VS_OUT_CNTL, E+0x234 u32 param dwords
//!     E+0x27c u32 VGT_GSVS_RING_ITEMSIZE (dwords = max_vertices * vertex_size_dwords)
//!     E+0x280 u32 instance count, E+0x284 u32 max vertices,
//!     E+0x28c u32 vertex size in dwords
//! HS  common record metadata (code, root layout, used resources, record+0x4c LDS bytes)
//! CS  E+0x24  u32 COMPUTE_PGM_RSRC2, E+0x28 u32 x3 thread-group size
//! MS  E+0xc8  outputs as VS, E+0x188 count; the param byte's bit 7 marks a
//!             per-primitive output (`PSG1` row); the 3 bytes more are the
//!             second row of a packed pair, a slot, flags (bit 0 fp16, bit 1
//!             two fp16 elements packed)
//!     E+0x18c u32 SPI_VS_OUT_CONFIG, SPI_SHADER_POS_FORMAT,
//!             PA_CL_VS_OUT_CNTL, VGT_STRMOUT_CONFIG, SPI_SHADER_PGM_RSRC2_GS,
//!             VGT_GS_OUT_PRIM_TYPE, VGT_GS_INSTANCE_CNT, SPI_SHADER_IDX_FORMAT
//!     E+0x448 u16 thread-group size in lanes, u16 maximum vertices,
//!             u16 maximum primitives, u8 user SGPRs, u8 output topology
//!             (2 triangles); code follows at E+0x45c
//! record +0x4c u32 LDS bytes (`lds_byte_size`; mesh and merged stages read it here)
//! ```
//!
//! Rows are registers of the container's signature parts (`ISG1`, `OSG1`).
//!
//! Other XDAT kinds seen: 0x07 program header + bitcode (or a 28 B `NOBC`
//! stub when stripped), 0x0a `xbop` (compiler options), 0x0c raw uncompressed
//! `FSSM` blob, 0x0d `XBCH`-compressed `FSSM` blob, 0x0e optional 4-byte
//! aligned metadata table (absent in GTA VI's 26,570 DXIL containers), 0x10
//! and 0x12 16-byte hashes.
//! Unknown versions are rejected so a format change shows up as an error.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::{u32_at, u64_at};

const XDAT_OFFSET: usize = 0x18;
const XDAT_ENTRIES_OFFSET: u32 = 0x24;
const MAX_XDAT_ENTRIES: usize = 64;
const KIND_BITCODE: u32 = 0x07;
const KIND_OPTIONS: u32 = 0x0a;
const KIND_NATIVE_UNPACKED: u32 = 0x0c;
const KIND_NATIVE: u32 = 0x0d;
const KIND_TABLE_0E: u32 = 0x0e;
const KIND_HASH_10: u32 = 0x10;
const KIND_HASH_12: u32 = 0x12;
const XBCH_VERSION: u32 = 3;
const XBCH_HEADER: usize = 0x18;
const FSSM_VERSION: u32 = 0x101;
const FSSM_HEADER: usize = 0x10;
const GRAPHICS_SLOT_COUNT: u64 = 6;
/// Where each stage's record offset sits in the FSSM payload.
const SLOTS: [(HwStage, usize); 8] = [
    (HwStage::Ls, 0x08),
    (HwStage::Hs, 0x10),
    (HwStage::Es, 0x18),
    (HwStage::Gs, 0x20),
    (HwStage::Vs, 0x28),
    (HwStage::Ps, 0x30),
    (HwStage::Ms, 0x48),
    (HwStage::Cs, 0x60),
];
const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
const RECORD_SIZE: u64 = 0x54;
const RECORD_FLAGS: usize = 0x14;
const FLAG_WAVE32: u32 = 1 << 5;
const RECORD_REGS: usize = 0x50;
const RECORD_LAYOUT: usize = 0x28;
const RECORD_USED: usize = 0x30;
const RECORD_COUNTS: usize = 0x38;
const ENTRY_SIZE: u64 = 20;
const MAX_SEMANTICS: u32 = 32;
const PS_INPUTS: usize = 0x24;
const PS_INPUT_COUNT: usize = 0x124;
const PS_INPUT_ADDR: usize = 0x134;
const PS_DB_SHADER_CONTROL: usize = 0x138;
const PS_PATCH_SIZE: usize = 0x148;
const PS_PATCHES: usize = 0x14c;
const PATCH_SLOT_DWORDS: usize = 10;
/// The variant that exports every channel as a raw 32-bit value.
const PATCH_FULL: usize = 9;
const PATCH_BYTES: usize = 4 + 10 * PATCH_SLOT_DWORDS * 4;
const MRTS: usize = 8;
const VS_INPUTS: usize = 0x24;
const VS_INPUT_COUNT: usize = 0xc4;
const VS_OUTPUTS: usize = 0xc8;
const VS_OUTPUT_COUNT: usize = 0x188;
const ES_ITEMSIZE_DWORDS: usize = 0x1ac;
const GS_TOPOLOGY: usize = 0x28;
const GS_INPUT_VERTICES: usize = 0x29;
const GS_COPY_CODE_OFFSET: usize = 0x34;
const GS_COPY_CODE_SIZE: usize = 0x3c;
const GS_OUTPUTS: usize = 0x150;
const GS_OUTPUT_COUNT: usize = 0x210;
const GS_VS_OUT_CONFIG: usize = 0x220;
const GS_POS_FORMAT: usize = 0x224;
const GS_CLIP_CNTL: usize = 0x228;
const GS_PARAM_DWORDS: usize = 0x234;
const GS_RING_ITEMSIZE_DWORDS: usize = 0x27c;
const GS_INSTANCE_COUNT: usize = 0x280;
const GS_MAX_VERTICES: usize = 0x284;
const GS_VERTEX_SIZE_DWORDS: usize = 0x28c;
const CS_RSRC2: usize = 0x24;
const CS_THREADS: usize = 0x28;
const RECORD_LDS: usize = 0x4c;
const MS_CLIP_CNTL: usize = 0x194;
const MS_INFO: usize = 0x448;
/// Unpacked blobs are a few KiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: u64 = 16 << 20;

/// Hardware shader stage a native program runs as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HwStage {
    Ls,
    Hs,
    Es,
    Gs,
    Vs,
    Ps,
    /// Mesh shader.
    Ms,
    Cs,
}

impl HwStage {
    /// The SGPR holding user-data dword `index`: merged stages (and mesh
    /// shaders, which run as one) have 8 system SGPRs first.
    pub fn user_sgpr(self, index: u8) -> u8 {
        match self {
            HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs | HwStage::Ms => index + 8,
            HwStage::Vs | HwStage::Ps | HwStage::Cs => index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProgram {
    pub stage: HwStage,
    /// Runs as 64 lanes per wave (else 32); sizes lane masks.
    pub wave64: bool,
    /// VGPRs the program uses (v0 up to this), the bound of indexed access.
    pub vgprs: u16,
    /// Where the root signature's parameters arrive (user SGPRs).
    pub layout: Vec<RootParam>,
    /// The descriptors the program reads, by root parameter.
    pub used: Vec<UsedResource>,
    pub io: StageIo,
    /// GFX10 machine code, as stored.
    pub code: Vec<u8>,
}

impl NativeProgram {
    /// LDS allocation in bytes declared by this stage (`COMPUTE_PGM_RSRC2` for
    /// compute, `record + 0x4c` for mesh and merged stages).
    pub fn lds_bytes(&self) -> u32 {
        match &self.io {
            StageIo::Cs(cs) => cs.lds_bytes(),
            StageIo::Ms(ms) => ms.lds_bytes,
            StageIo::Es(es) => es.lds_bytes,
            StageIo::Gs(gs) => gs.lds_bytes,
            StageIo::Ls(ls) => ls.lds_bytes,
            StageIo::Hs(hs) => hs.lds_bytes,
            StageIo::None | StageIo::Vs { .. } | StageIo::Ps { .. } => 0,
        }
    }

    /// The code with each render target's export patched for its full
    /// 32-bit format, so exports carry the shader's values unconverted (as
    /// the console does at pipeline creation for that format).
    pub fn patched_code(&self) -> Vec<u8> {
        let mut code = self.code.clone();
        if let StageIo::Ps { patches, .. } = &self.io {
            for p in patches {
                let at = p.offset as usize * 4;
                for (i, word) in p.code.iter().enumerate() {
                    code[at + 4 * i..at + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        code
    }
}

/// What a semantic is, as the stage data names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Usage {
    Generic,
    ClipDistance,
    Other(u8),
}

impl Usage {
    fn from_raw(raw: u8) -> Self {
        match raw {
            6 => Usage::Generic,
            7 => Usage::ClipDistance,
            n => Usage::Other(n),
        }
    }
}

/// A pixel shader input: what `v_interp` of attribute `attr` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsInput {
    pub usage: Usage,
    /// Register in the input signature.
    pub row: u8,
    pub mask: u8,
    pub attr: u8,
    pub flat: bool,
    pub fp16: bool,
    /// Two 16-bit values per dword ([`VsOutput::pack2`]); `row2` is the
    /// second element's register.
    pub pack2: bool,
    pub row2: u8,
}

/// A vertex attribute the fetch shader loads: channel k into VGPR `vgpr + k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsInput {
    pub usage: Usage,
    pub row: u8,
    pub mask: u8,
    pub vgpr: u8,
}

/// What `exp param<param>` of a vertex shader writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsOutput {
    pub usage: Usage,
    /// Register in the output signature.
    pub row: u8,
    pub param: u8,
    /// 16-bit values, one per dword (low half) unless `pack2`: then two per
    /// dword, channel c of this row in dword c / 2 (odd channels in the
    /// high half), then a second 16-bit row from the next whole dword (the
    /// next 16-bit row of the signature no entry names).
    pub fp16: bool,
    pub pack2: bool,
}

/// The export code of one render target for its full 32-bit format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPatch {
    pub mrt: u8,
    /// Where it goes in the code, in dwords.
    pub offset: u32,
    pub code: Vec<u32>,
}

/// Inputs and outputs, for the stages whose stage data is read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum StageIo {
    #[default]
    None,
    Vs {
        inputs: Vec<VsInput>,
        outputs: Vec<VsOutput>,
    },
    Ps {
        inputs: Vec<PsInput>,
        /// `SPI_PS_INPUT_ADDR`: which system values arrive in VGPRs.
        input_addr: u32,
        /// `DB_SHADER_CONTROL`: depth export, kill.
        db_shader_control: u32,
        patches: Vec<ExportPatch>,
    },
    Cs(ComputeIo),
    Ms(MeshIo),
    /// Export shader half of a merged ES+GS pipeline.
    Es(EsIo),
    /// Geometry shader half of a merged ES+GS pipeline.
    Gs(GsIo),
    /// Local shader half of a merged LS+HS pipeline.
    Ls(LsIo),
    /// Hull shader half of a merged LS+HS pipeline.
    Hs(HsIo),
}

/// Export shader (vertex stage feeding a geometry shader via LDS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EsIo {
    /// Vertex fetch inputs (VGPR indices start after ES/GS system VGPRs, e.g. `v9+`).
    pub inputs: Vec<VsInput>,
    /// All outputs written to the ES->GS LDS ring, including `SV_Position` (usage 0).
    pub outputs: Vec<VsOutput>,
    /// Dwords written per vertex to the ES->GS LDS ring (`E+0x1ac`).
    pub esgs_itemsize_dwords: u32,
    pub lds_bytes: u32,
}

/// Geometry shader (merged ES+GS) stage data, including its embedded copy shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GsIo {
    /// Non-position `exp param<n>` outputs exported by the GS copy shader (`E+0x150`).
    pub outputs: Vec<VsOutput>,
    /// `VGT_GS_OUT_PRIM_TYPE` (`E+0x28`: 2 = triangle strip).
    pub output_topology: u8,
    /// Vertices per input primitive (`E+0x29`: 3 = triangle).
    pub input_vertices: u8,
    /// `SPI_VS_OUT_CONFIG` (`E+0x220`).
    pub vs_out_config: u32,
    /// `SPI_SHADER_POS_FORMAT` (`E+0x224`).
    pub pos_format: u32,
    /// `PA_CL_VS_OUT_CNTL` (`E+0x228`).
    pub clip_cntl: u32,
    /// Non-position parameter dwords per output vertex (`E+0x234`).
    pub param_dwords: u32,
    /// `VGT_GSVS_RING_ITEMSIZE`: dwords per invocation (`max_vertices * vertex_size_dwords`, `E+0x27c`).
    pub ring_itemsize_dwords: u32,
    /// GS instance count (`E+0x280`).
    pub instance_count: u32,
    /// Maximum output vertices per invocation (`E+0x284`).
    pub max_vertices: u32,
    /// Total output dwords per vertex including position (`E+0x28c`).
    pub vertex_size_dwords: u32,
    /// `record + 0x4c` LDS bytes.
    pub lds_bytes: u32,
    /// Embedded GFX10 GS copy shader (hardware VS reading the GSVS ring buffer), empty if absent.
    pub copy_code: Vec<u8>,
}

/// Local shader (vertex stage feeding a hull shader in merged LS+HS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LsIo {
    pub inputs: Vec<VsInput>,
    pub outputs: Vec<VsOutput>,
    /// Dwords written per control point to the LS->HS LDS ring (`E+0x1ac`).
    pub lshs_itemsize_dwords: u32,
    pub lds_bytes: u32,
}

/// Hull shader (merged LS+HS) common metadata extracted from the FSSM record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HsIo {
    pub lds_bytes: u32,
    /// Byte length of the raw HS stage-data region preceding the machine code.
    pub stage_data_len: usize,
}

/// A mesh program's outputs and limits. It runs as one hardware NGG stage:
/// lane n of the group (`threads` lanes) exports vertex n (`exp pos`,
/// `exp param`) and primitive n (`exp prim`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshIo {
    /// `exp param<n>` targets; `param` bit 7 set: a per-primitive output,
    /// `row` then a row of the primitive signature (`PSG1`).
    pub outputs: Vec<VsOutput>,
    /// `PA_CL_VS_OUT_CNTL`: which clip and cull distances `pos1+` carry.
    pub clip_cntl: u32,
    /// Thread-group size in lanes (the HLSL group rounded up to waves).
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
    pub lds_bytes: u32,
}

/// A compute program's dispatch settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeIo {
    /// `COMPUTE_PGM_RSRC2`: user SGPRs, which system SGPRs and VGPRs
    /// follow them, LDS size.
    pub rsrc2: u32,
    /// Thread-group size (`numthreads`).
    pub threads: [u32; 3],
}

impl ComputeIo {
    /// User SGPRs before the system ones.
    pub fn user_sgprs(&self) -> u8 {
        (self.rsrc2 >> 1 & 0x1F) as u8
    }

    /// The SGPR of each enabled thread-group id component, in order after
    /// the user SGPRs, then the thread-group info SGPR if enabled.
    pub fn system_sgprs(&self) -> ([Option<u8>; 3], Option<u8>) {
        let mut next = self.user_sgprs();
        let mut take = |bit: u32| {
            (self.rsrc2 >> bit & 1 != 0).then(|| {
                next += 1;
                next - 1
            })
        };
        let ids = [take(7), take(8), take(9)];
        (ids, take(10))
    }

    /// SGPRs the hardware initializes: the user SGPRs, the enabled system
    /// ones, then the scratch wave offset when scratch is enabled (bit 0).
    /// The ones above hold whatever the previous wave left.
    pub fn initialized_sgprs(&self) -> u8 {
        let (ids, info) = self.system_sgprs();
        let end = ids.into_iter().chain([info]).flatten().map(|s| s + 1).max();
        end.unwrap_or(0).max(self.user_sgprs()) + (self.rsrc2 & 1) as u8
    }

    /// Thread id components in v0, v1, v2 (1 to 3).
    pub fn thread_id_vgprs(&self) -> u8 {
        (self.rsrc2 >> 11 & 3) as u8 + 1
    }

    /// LDS bytes per thread group (512-byte granules).
    pub fn lds_bytes(&self) -> u32 {
        (self.rsrc2 >> 15 & 0x1FF) * 512
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootType {
    Constants,
    /// A 64-bit buffer address.
    Descriptor,
    /// The 32-bit heap address of a descriptor table range.
    Table,
    /// A sampler the root signature fixes.
    ImmediateSampler,
    /// The 32-bit address of the CBV/SRV/UAV heap (SM 6.6 dynamic resources).
    ResourceHeap,
    /// The 32-bit address of the sampler heap.
    SamplerHeap,
    Other(u32),
}

impl RootType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            3 => RootType::Table,
            4 => RootType::ImmediateSampler,
            5 => RootType::ResourceHeap,
            6 => RootType::SamplerHeap,
            n => RootType::Other(n),
        }
    }
}

/// One root parameter (or table range) and the user SGPRs holding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootParam {
    pub root: RootType,
    /// Of the range within its table.
    pub table_offset_dwords: u32,
    /// User SGPRs from `user_data` on.
    pub sgprs: u32,
    /// Shader register of the first descriptor (`b`/`t`/`u`/`s` number).
    pub base_register: u32,
    /// First user-data dword ([`HwStage::user_sgpr`]).
    pub user_data: u8,
    /// Index of the parameter in the root signature.
    pub slot: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Srv,
    Sampler,
    Cbv,
    Uav,
    Other(u8),
}

/// A descriptor range the program reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedResource {
    pub kind: ResourceType,
    pub root: RootType,
    pub table_offset_dwords: u32,
    /// Shader register of the first descriptor read.
    pub lower_bound: u32,
    pub stride: u32,
    /// Size of the descriptor (4 or 8).
    pub dwords: u8,
    pub user_data: u8,
    pub slot: u8,
}

/// Known entry kinds in an Xbox `XDAT` directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XdatKind {
    /// `0x07`: DXIL program header + LLVM bitcode (or 28-byte `NOBC` stub).
    Bitcode,
    /// `0x0a`: `xbop` compiler options.
    Options,
    /// `0x0c`: Raw uncompressed `FSSM` native program blob.
    NativeUnpacked,
    /// `0x0d`: `XBCH`-compressed `FSSM` native program blob.
    Native,
    /// `0x0e`: Optional 4-byte aligned metadata table.
    Table0e,
    /// `0x10`: 16-byte shader hash.
    Hash10,
    /// `0x12`: 16-byte container/blob hash.
    Hash12,
    Other(u32),
}

impl XdatKind {
    fn from_raw(raw: u32) -> Self {
        match raw {
            KIND_BITCODE => XdatKind::Bitcode,
            KIND_OPTIONS => XdatKind::Options,
            KIND_NATIVE_UNPACKED => XdatKind::NativeUnpacked,
            KIND_NATIVE => XdatKind::Native,
            KIND_TABLE_0E => XdatKind::Table0e,
            KIND_HASH_10 => XdatKind::Hash10,
            KIND_HASH_12 => XdatKind::Hash12,
            other => XdatKind::Other(other),
        }
    }
}

/// One validated entry in an `XDAT` directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdatEntry<'a> {
    pub kind: XdatKind,
    pub offset: u32,
    pub data: &'a [u8],
}

/// Decoded `XDAT` `0x0e` table payload (4-byte aligned dword table).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdatTable0e {
    pub dwords: Vec<u32>,
}

impl XdatTable0e {
    /// Parses a non-empty, 4-byte aligned `0x0e` metadata table.
    pub fn parse(blob: &[u8]) -> Result<Self, ShaderError> {
        if blob.is_empty() || blob.len() % 4 != 0 {
            return Err(Malformed("XDAT 0x0e table has invalid size"));
        }
        let dwords = blob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        Ok(XdatTable0e { dwords })
    }
}

/// Parsed `XDAT` index embedded in an Xbox `DXIL` part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xdat<'a> {
    pub entries: Vec<XdatEntry<'a>>,
    pub xbop: Option<&'a [u8]>,
    pub table_0e: Vec<XdatTable0e>,
    pub hash_10: Option<[u8; 16]>,
    pub hash_12: Option<[u8; 16]>,
}

/// Parses the `XDAT` directory inside a `DXIL` part; `Ok(None)` for standard PC parts.
pub fn parse_xdat(part: &[u8]) -> Result<Option<Xdat<'_>>, ShaderError> {
    if part.get(XDAT_OFFSET..XDAT_OFFSET + 4) != Some(b"XDAT") {
        return Ok(None);
    }
    let count = u32_at(part, XDAT_OFFSET + 4)? as usize;
    if count > MAX_XDAT_ENTRIES {
        return Err(Malformed("implausible XDAT entry count"));
    }
    let table_offset = u32_at(part, XDAT_OFFSET + 8)?;
    if table_offset != XDAT_ENTRIES_OFFSET {
        return Err(Malformed("unexpected XDAT entries offset"));
    }
    let mut entries = Vec::with_capacity(count);
    let mut xbop = None;
    let mut table_0e = Vec::new();
    let mut hash_10 = None;
    let mut hash_12 = None;
    for i in 0..count {
        let at = table_offset as usize + 12 * i;
        let kind = XdatKind::from_raw(u32_at(part, at)?);
        let offset = u32_at(part, at + 4)?;
        let size = u32_at(part, at + 8)?;
        let data = bytes(part, offset.into(), size.into())
            .ok_or(Malformed("XDAT entry out of bounds"))?;
        match kind {
            XdatKind::Options => {
                if data.get(0..4) != Some(b"xbop") {
                    return Err(Malformed("XDAT options entry lacks xbop magic"));
                }
                xbop = Some(data);
            }
            XdatKind::Table0e => {
                table_0e.push(XdatTable0e::parse(data)?);
            }
            XdatKind::Hash10 => {
                hash_10 = Some(
                    data.try_into()
                        .map_err(|_| Malformed("unexpected XDAT 0x10 hash size"))?,
                );
            }
            XdatKind::Hash12 => {
                hash_12 = Some(
                    data.try_into()
                        .map_err(|_| Malformed("unexpected XDAT 0x12 hash size"))?,
                );
            }
            _ => {}
        }
        entries.push(XdatEntry { kind, offset, data });
    }
    Ok(Some(Xdat {
        entries,
        xbop,
        table_0e,
        hash_10,
        hash_12,
    }))
}

/// The native programs in a `DXIL` part's data; empty for the PC layout.
pub fn native_programs(part: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    let Some(xdat) = parse_xdat(part)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for entry in xdat.entries {
        match entry.kind {
            XdatKind::Native => out.extend(programs(&unpack(entry.data)?)?),
            XdatKind::NativeUnpacked => out.extend(programs(entry.data)?),
            _ => {}
        }
    }
    Ok(out)
}

fn unpack(blob: &[u8]) -> Result<Vec<u8>, ShaderError> {
    if blob.get(0..4) != Some(b"XBCH") {
        return Err(Malformed("native blob lacks XBCH magic"));
    }
    if u32_at(blob, 4)? != XBCH_VERSION {
        return Err(Malformed("unknown XBCH version"));
    }
    let packed = u64_at(blob, 8)?;
    let unpacked = u64_at(blob, 16)?;
    if unpacked > MAX_UNPACKED {
        return Err(Malformed("implausible XBCH unpacked size"));
    }
    let stream =
        bytes(blob, XBCH_HEADER as u64, packed).ok_or(Malformed("XBCH stream out of bounds"))?;
    let out = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(stream, unpacked as usize)
        .map_err(|_| Malformed("XBCH stream does not inflate to its stated size"))?;
    if out.len() as u64 != unpacked {
        return Err(Malformed("XBCH stream does not inflate to its stated size"));
    }
    Ok(out)
}

fn programs(fssm: &[u8]) -> Result<Vec<NativeProgram>, ShaderError> {
    if fssm.get(0..4) != Some(b"FSSM") {
        return Err(Malformed("native blob lacks FSSM magic"));
    }
    if u32_at(fssm, 4)? != FSSM_VERSION {
        return Err(Malformed("unknown FSSM version"));
    }
    let payload = &fssm[FSSM_HEADER.min(fssm.len())..];
    if u64_at(fssm, 8)? != payload.len() as u64 {
        return Err(Malformed("FSSM payload size disagrees"));
    }
    if u64_at(payload, 0)? != GRAPHICS_SLOT_COUNT {
        return Err(Malformed("unexpected FSSM slot count"));
    }
    let mut out = Vec::new();
    for (stage, at) in SLOTS {
        let start = u64_at(payload, at)?;
        if start == ABSENT {
            continue;
        }
        let record =
            bytes(payload, start, RECORD_SIZE).ok_or(Malformed("FSSM record out of bounds"))?;
        let code = bytes(payload, u64_at(record, 0)?, u64_at(record, 8)?)
            .ok_or(Malformed("native code out of bounds"))?;
        let vgprs = (u32_at(record, RECORD_REGS)? & 0x3FF) as u16;
        if vgprs > 256 {
            return Err(Malformed("implausible VGPR count"));
        }
        let table = |at: usize, count_at: usize| -> Result<&[u8], ShaderError> {
            let len = ENTRY_SIZE * u64::from(u32_at(record, count_at)?);
            if len == 0 {
                return Ok(&[]);
            }
            bytes(payload, u64_at(record, at)?, len)
                .ok_or(Malformed("FSSM root table out of bounds"))
        };
        let layout = table(RECORD_LAYOUT, RECORD_COUNTS)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| RootParam {
                root: RootType::from_raw(le32(e, 0)),
                table_offset_dwords: le32(e, 4),
                sgprs: le32(e, 8),
                base_register: le32(e, 12),
                user_data: e[16],
                slot: e[17],
            })
            .collect();
        let used = table(RECORD_USED, RECORD_COUNTS + 4)?
            .as_chunks::<{ ENTRY_SIZE as usize }>()
            .0
            .iter()
            .map(|e| UsedResource {
                kind: match e[0] {
                    0 => ResourceType::Srv,
                    1 => ResourceType::Sampler,
                    2 => ResourceType::Cbv,
                    3 => ResourceType::Uav,
                    n => ResourceType::Other(n),
                },
                root: RootType::from_raw(e[1].into()),
                table_offset_dwords: le32(e, 4),
                lower_bound: le32(e, 8),
                stride: le32(e, 12),
                dwords: e[16],
                user_data: e[17],
                slot: e[18],
            })
            .collect();
        // The record is in bounds, so its end is too.
        let code_offset = u64_at(record, 0)?;
        let stage_data_start = start + RECORD_SIZE;
        let stage_data = &payload[stage_data_start as usize..];
        let lds_bytes = u32_at(record, RECORD_LDS)?;
        let io = match stage {
            HwStage::Ps => ps_io(stage_data, code.len())?,
            HwStage::Vs => vs_io(stage_data)?,
            HwStage::Cs => StageIo::Cs(ComputeIo {
                rsrc2: u32_at(stage_data, CS_RSRC2)?,
                threads: [
                    u32_at(stage_data, CS_THREADS)?,
                    u32_at(stage_data, CS_THREADS + 4)?,
                    u32_at(stage_data, CS_THREADS + 8)?,
                ],
            }),
            HwStage::Ms => ms_io(stage_data, lds_bytes)?,
            HwStage::Es => es_io(stage_data, lds_bytes)?,
            HwStage::Gs => gs_io(stage_data, payload, lds_bytes)?,
            HwStage::Ls => ls_io(stage_data, lds_bytes)?,
            HwStage::Hs => {
                let stage_data_len = code_offset
                    .checked_sub(stage_data_start)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or(Malformed("HS stage data overlaps record"))?;
                StageIo::Hs(HsIo {
                    lds_bytes,
                    stage_data_len,
                })
            }
        };
        out.push(NativeProgram {
            stage,
            wave64: u32_at(record, RECORD_FLAGS)? & FLAG_WAVE32 == 0,
            vgprs,
            layout,
            used,
            io,
            code: code.to_vec(),
        });
    }
    Ok(out)
}

/// The entries of `size` bytes from `at`, with their count at `count_at`.
fn semantics(data: &[u8], at: usize, size: usize, count_at: usize) -> Result<&[u8], ShaderError> {
    let count = u32_at(data, count_at)?;
    if count > MAX_SEMANTICS {
        return Err(Malformed("implausible semantic count"));
    }
    data.get(at..at + size * count as usize)
        .ok_or(Malformed("stage data out of bounds"))
}

fn ps_io(data: &[u8], code_len: usize) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, PS_INPUTS, 8, PS_INPUT_COUNT)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|e| PsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            attr: e[3],
            flat: e[7] & 1 != 0,
            fp16: e[7] & 8 != 0,
            pack2: e[7] & 0x10 != 0,
            row2: e[5],
        })
        .collect();
    let size = u32_at(data, PS_PATCH_SIZE)? as usize;
    if size > PATCH_SLOT_DWORDS {
        return Err(Malformed("implausible export patch size"));
    }
    let mut patches = Vec::new();
    for mrt in 0..MRTS {
        let at = PS_PATCHES + mrt * PATCH_BYTES;
        let offset = u32_at(data, at)?;
        if offset == u32::MAX {
            continue;
        }
        if (offset as usize + size) * 4 > code_len {
            return Err(Malformed("export patch outside the code"));
        }
        let full = at + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        let code = (0..size)
            .map(|i| u32_at(data, full + 4 * i))
            .collect::<Result<_, _>>()?;
        patches.push(ExportPatch {
            mrt: mrt as u8,
            offset,
            code,
        });
    }
    Ok(StageIo::Ps {
        inputs,
        input_addr: u32_at(data, PS_INPUT_ADDR)?,
        db_shader_control: u32_at(data, PS_DB_SHADER_CONTROL)?,
        patches,
    })
}

fn vs_io(data: &[u8]) -> Result<StageIo, ShaderError> {
    let inputs = semantics(data, VS_INPUTS, 5, VS_INPUT_COUNT)?
        .as_chunks::<5>()
        .0
        .iter()
        .map(|e| VsInput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            mask: e[2] & 0xf,
            vgpr: e[3],
        })
        .collect();
    let outputs = semantics(data, VS_OUTPUTS, 6, VS_OUTPUT_COUNT)?
        .as_chunks::<6>()
        .0
        .iter()
        .map(|e| VsOutput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            param: e[2],
            fp16: e[5] & 1 != 0,
            pack2: e[5] & 2 != 0,
        })
        .collect();
    Ok(StageIo::Vs { inputs, outputs })
}

fn ms_io(data: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let StageIo::Vs { outputs, .. } = vs_io(data)? else {
        unreachable!()
    };
    let info = data
        .get(MS_INFO..MS_INFO + 8)
        .ok_or(Malformed("stage data out of bounds"))?;
    let half = |at: usize| u32::from(u16::from_le_bytes([info[at], info[at + 1]]));
    Ok(StageIo::Ms(MeshIo {
        outputs,
        clip_cntl: u32_at(data, MS_CLIP_CNTL)?,
        threads: half(0),
        max_vertices: half(2),
        max_primitives: half(4),
        topology: info[7],
        lds_bytes,
    }))
}

fn es_io(data: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let StageIo::Vs { inputs, outputs } = vs_io(data)? else {
        unreachable!()
    };
    Ok(StageIo::Es(EsIo {
        inputs,
        outputs,
        esgs_itemsize_dwords: u32_at(data, ES_ITEMSIZE_DWORDS)?,
        lds_bytes,
    }))
}

fn ls_io(data: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let StageIo::Vs { inputs, outputs } = vs_io(data)? else {
        unreachable!()
    };
    Ok(StageIo::Ls(LsIo {
        inputs,
        outputs,
        lshs_itemsize_dwords: u32_at(data, ES_ITEMSIZE_DWORDS)?,
        lds_bytes,
    }))
}

fn gs_io(data: &[u8], payload: &[u8], lds_bytes: u32) -> Result<StageIo, ShaderError> {
    let outputs = semantics(data, GS_OUTPUTS, 6, GS_OUTPUT_COUNT)?
        .as_chunks::<6>()
        .0
        .iter()
        .map(|e| VsOutput {
            usage: Usage::from_raw(e[0]),
            row: e[1],
            param: e[2],
            fp16: e[5] & 1 != 0,
            pack2: e[5] & 2 != 0,
        })
        .collect();
    let copy_offset = u64_at(data, GS_COPY_CODE_OFFSET)?;
    let copy_size = u64_at(data, GS_COPY_CODE_SIZE)?;
    let copy_code = if copy_size == 0 || copy_offset == ABSENT {
        Vec::new()
    } else {
        bytes(payload, copy_offset, copy_size)
            .ok_or(Malformed("GS copy shader out of bounds"))?
            .to_vec()
    };
    let output_topology = *data
        .get(GS_TOPOLOGY)
        .ok_or(Malformed("stage data out of bounds"))?;
    let input_vertices = *data
        .get(GS_INPUT_VERTICES)
        .ok_or(Malformed("stage data out of bounds"))?;
    Ok(StageIo::Gs(GsIo {
        outputs,
        output_topology,
        input_vertices,
        vs_out_config: u32_at(data, GS_VS_OUT_CONFIG)?,
        pos_format: u32_at(data, GS_POS_FORMAT)?,
        clip_cntl: u32_at(data, GS_CLIP_CNTL)?,
        param_dwords: u32_at(data, GS_PARAM_DWORDS)?,
        ring_itemsize_dwords: u32_at(data, GS_RING_ITEMSIZE_DWORDS)?,
        instance_count: u32_at(data, GS_INSTANCE_COUNT)?,
        max_vertices: u32_at(data, GS_MAX_VERTICES)?,
        vertex_size_dwords: u32_at(data, GS_VERTEX_SIZE_DWORDS)?,
        lds_bytes,
        copy_code,
    }))
}

/// For entries already sized by `as_chunks`.
fn le32(entry: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(entry[at..at + 4].try_into().unwrap())
}

/// `data[offset..offset + len]` with untrusted 64-bit bounds.
fn bytes(data: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    data.get(start..start.checked_add(usize::try_from(len).ok()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS_CODE: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const PS_CODE: [u8; 4] = [9, 10, 11, 12];
    const VGPRS: u32 = 28;

    fn slot(stage: HwStage) -> usize {
        SLOTS.iter().find(|s| s.0 == stage).unwrap().1
    }

    /// Stage data: no semantics, and every export patch absent.
    const STAGE_DATA: usize = PS_PATCHES + MRTS * PATCH_BYTES;

    fn empty_stage_data() -> Vec<u8> {
        let mut data = vec![0; STAGE_DATA];
        for mrt in 0..MRTS {
            let at = PS_PATCHES + mrt * PATCH_BYTES;
            data[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        data
    }

    /// Where a one-program blob's stage data starts.
    fn stage_data_at(blob: &[u8], stage: HwStage) -> usize {
        FSSM_HEADER
            + u64_at(blob, FSSM_HEADER + slot(stage)).unwrap() as usize
            + RECORD_SIZE as usize
    }

    /// FSSM blob with `programs` as (stage, code) pairs.
    fn fssm(programs: &[(HwStage, &[u8])]) -> Vec<u8> {
        let mut payload = GRAPHICS_SLOT_COUNT.to_le_bytes().to_vec();
        payload.resize(slot(HwStage::Cs) + 8, 0);
        for (_, at) in SLOTS {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        for &(stage, code) in programs {
            let slot = slot(stage);
            let record = payload.len() as u64;
            payload[slot..slot + 8].copy_from_slice(&record.to_le_bytes());
            let code_at = record + RECORD_SIZE + STAGE_DATA as u64;
            payload.extend_from_slice(&code_at.to_le_bytes());
            payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
            let flags = if stage == HwStage::Ps { FLAG_WAVE32 } else { 0 };
            payload.extend_from_slice(&[0; 4]);
            payload.extend_from_slice(&flags.to_le_bytes());
            payload.resize(record as usize + RECORD_REGS, 0);
            payload.extend_from_slice(&(VGPRS | 31 << 20).to_le_bytes());
            payload.extend_from_slice(&empty_stage_data());
            payload.extend_from_slice(code);
        }
        let mut out = b"FSSM".to_vec();
        out.extend_from_slice(&FSSM_VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn xbch(unpacked: &[u8]) -> Vec<u8> {
        let stream = miniz_oxide::deflate::compress_to_vec_zlib(unpacked, 6);
        let mut out = b"XBCH".to_vec();
        out.extend_from_slice(&XBCH_VERSION.to_le_bytes());
        out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        out.extend_from_slice(&(unpacked.len() as u64).to_le_bytes());
        out.extend_from_slice(&stream);
        out
    }

    /// `DXIL` part data: program header, XDAT index, then the blobs.
    fn part(blobs: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = vec![0u8; XDAT_OFFSET];
        out.extend_from_slice(b"XDAT");
        out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x24u32.to_le_bytes());
        let mut offset = out.len() + 12 * blobs.len();
        for (kind, blob) in blobs {
            for v in [*kind, offset as u32, blob.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            offset += blob.len();
        }
        for (_, blob) in blobs {
            out.extend_from_slice(blob);
        }
        out
    }

    #[test]
    fn finds_graphics_programs() {
        let native = xbch(&fssm(&[(HwStage::Vs, &VS_CODE), (HwStage::Ps, &PS_CODE)]));
        let data = part(&[(0x0a, b"xbop"), (KIND_NATIVE, &native)]);
        let found = native_programs(&data).unwrap();
        assert_eq!(
            found,
            [
                NativeProgram {
                    stage: HwStage::Vs,
                    wave64: true,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Vs {
                        inputs: vec![],
                        outputs: vec![],
                    },
                    code: VS_CODE.to_vec()
                },
                NativeProgram {
                    stage: HwStage::Ps,
                    wave64: false,
                    vgprs: 28,
                    layout: vec![],
                    used: vec![],
                    io: StageIo::Ps {
                        inputs: vec![],
                        input_addr: 0,
                        db_shader_control: 0,
                        patches: vec![],
                    },
                    code: PS_CODE.to_vec()
                },
            ]
        );
    }

    #[test]
    fn reads_root_tables() {
        let mut blob = fssm(&[(HwStage::Ps, &PS_CODE)]);
        let record = FSSM_HEADER + u64_at(&blob, FSSM_HEADER + slot(HwStage::Ps)).unwrap() as usize;
        let tables = (blob.len() - FSSM_HEADER) as u64;
        // A table range in s9 (root slot 1), and an SRV read from it.
        for v in [3u32, 2, 1, 5] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[9, 1, 0, 0, 0, 3, 0, 0]);
        for v in [2u32, 5, 4] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&[8, 9, 1, 0]);
        let payload = (blob.len() - FSSM_HEADER) as u64;
        blob[8..16].copy_from_slice(&payload.to_le_bytes());
        blob[record + RECORD_LAYOUT..][..8].copy_from_slice(&tables.to_le_bytes());
        blob[record + RECORD_USED..][..8].copy_from_slice(&(tables + ENTRY_SIZE).to_le_bytes());
        blob[record + RECORD_COUNTS..][..8].copy_from_slice(&[1, 0, 0, 0, 1, 0, 0, 0]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        assert_eq!(
            p.layout,
            [RootParam {
                root: RootType::Table,
                table_offset_dwords: 2,
                sgprs: 1,
                base_register: 5,
                user_data: 9,
                slot: 1
            }]
        );
        assert_eq!(
            p.used,
            [UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 2,
                lower_bound: 5,
                stride: 4,
                dwords: 8,
                user_data: 9,
                slot: 1
            }]
        );

        blob[record + RECORD_COUNTS] = 9;
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn reads_pixel_io_and_applies_the_full_export_patch() {
        let code = [0xBF80_0000u32; 4]; // s_nop
        let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut blob = fssm(&[(HwStage::Ps, &bytes)]);
        let e = stage_data_at(&blob, HwStage::Ps);
        // TEXCOORD in row 1 (xy), read as attr0, flat.
        blob[e + PS_INPUTS..][..8].copy_from_slice(&[6, 1, 3, 0, 0, 0, 1, 1]);
        blob[e + PS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[e + PS_INPUT_ADDR..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + PS_PATCH_SIZE..][..4].copy_from_slice(&2u32.to_le_bytes());
        // MRT1 at dword 1: the 32_ABGR variant is `exp mrt1 v0-v3 done`.
        let patch = e + PS_PATCHES + PATCH_BYTES;
        blob[patch..][..4].copy_from_slice(&1u32.to_le_bytes());
        let full = patch + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        blob[full..][..8].copy_from_slice(&[0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);

        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let StageIo::Ps {
            inputs,
            input_addr,
            patches,
            ..
        } = &p.io
        else {
            panic!("{:?}", p.io)
        };
        let input = PsInput {
            usage: Usage::Generic,
            row: 1,
            mask: 3,
            attr: 0,
            flat: true,
            fp16: false,
            pack2: false,
            row2: 0,
        };
        assert_eq!((&inputs[..], *input_addr), (&[input][..], 2));
        assert_eq!(patches.len(), 1);
        assert_eq!((patches[0].mrt, patches[0].offset), (1, 1));
        let patched = p.patched_code();
        assert_eq!(patched[..4], p.code[..4]);
        assert_eq!(patched[4..12], [0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3]);
        assert_eq!(patched[12..], p.code[12..]);

        blob[patch..][..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).is_err());
    }

    #[test]
    fn finds_mesh_and_compute_programs() {
        for stage in [HwStage::Ms, HwStage::Cs] {
            let data = part(&[(KIND_NATIVE, &xbch(&fssm(&[(stage, &VS_CODE)])))]);
            let found = native_programs(&data).unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].stage, stage);
        }
    }

    #[test]
    fn reads_mesh_outputs_and_limits() {
        let mut blob = fssm(&[(HwStage::Ms, &VS_CODE)]);
        let e = stage_data_at(&blob, HwStage::Ms);
        // param0 is row 1 per vertex, param2 row 0 of the primitive signature.
        blob[e + VS_OUTPUTS..][..12].copy_from_slice(&[6, 1, 0, 0, 0, 0, 6, 0, 0x82, 0, 0, 0]);
        blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        blob[e + MS_CLIP_CNTL..][..4].copy_from_slice(&0x0040_000Fu32.to_le_bytes());
        // 128 lanes, 128 vertices, 64 primitives, 13 user SGPRs, triangles.
        blob[e + MS_INFO..][..8].copy_from_slice(&[0x80, 0, 0x80, 0, 0x40, 0, 13, 2]);
        let record = e - RECORD_SIZE as usize;
        blob[record + RECORD_LDS..][..4].copy_from_slice(&16u32.to_le_bytes());
        let p = &native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap()[0];
        let out = |row, param| VsOutput {
            usage: Usage::Generic,
            row,
            param,
            fp16: false,
            pack2: false,
        };
        assert_eq!(
            p.io,
            StageIo::Ms(MeshIo {
                outputs: vec![out(1, 0), out(0, 0x82)],
                clip_cntl: 0x0040_000F,
                threads: 128,
                max_vertices: 128,
                max_primitives: 64,
                topology: 2,
                lds_bytes: 16,
            })
        );
    }

    #[test]
    fn compute_dispatch_settings() {
        // 5 user SGPRs, group id x and y, thread ids in v0-v1, 1 KiB LDS.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.user_sgprs(), 5);
        assert_eq!(cs.system_sgprs(), ([Some(5), Some(6), None], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (2, 1024));
        // 6 user SGPRs, all three group ids, v0-v2, no LDS.
        let cs = ComputeIo {
            rsrc2: 0x138C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.system_sgprs(), ([Some(6), Some(7), Some(8)], None));
        assert_eq!((cs.thread_id_vgprs(), cs.lds_bytes()), (3, 0));
    }

    #[test]
    fn initialized_sgprs_covers_user_system_and_scratch() {
        // 5 user SGPRs, group ids x and y (s5, s6), no scratch: s0..=s6.
        let cs = ComputeIo {
            rsrc2: 0x1098A,
            threads: [16, 16, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 7);
        // All three group ids after 6 user SGPRs, plus the info SGPR: s0..=s9.
        let cs = ComputeIo {
            rsrc2: 0x78C,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 10);
        // The same with scratch enabled adds the wave offset: s0..=s10.
        let cs = ComputeIo {
            rsrc2: 0x78D,
            threads: [8, 8, 1],
        };
        assert_eq!(cs.initialized_sgprs(), 11);
    }

    #[test]
    fn reads_every_native_blob() {
        let es = xbch(&fssm(&[(HwStage::Es, &VS_CODE)]));
        let vs = xbch(&fssm(&[(HwStage::Vs, &VS_CODE)]));
        let data = part(&[(0x12, &[0; 16]), (KIND_NATIVE, &es), (KIND_NATIVE, &vs)]);
        let stages: Vec<_> = native_programs(&data)
            .unwrap()
            .iter()
            .map(|p| p.stage)
            .collect();
        assert_eq!(stages, [HwStage::Es, HwStage::Vs]);
    }

    #[test]
    fn pc_layout_has_none() {
        assert!(native_programs(&[0u8; 64]).unwrap().is_empty());
        assert!(
            native_programs(&part(&[(0x0a, b"xbop")]))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_corrupt_blobs() {
        let good = xbch(&fssm(&[(HwStage::Ls, &VS_CODE)]));
        let corrupt = |at: usize, byte: u8| {
            let mut blob = good.clone();
            blob[at] = byte;
            native_programs(&part(&[(KIND_NATIVE, &blob)]))
        };
        assert!(corrupt(4, 9).is_err(), "unknown version");
        assert!(corrupt(16, 0xFF).is_err(), "stated size");
        assert!(corrupt(XBCH_HEADER + 2, 0xFF).is_err(), "zlib stream");
        assert!(native_programs(&part(&[(KIND_NATIVE, &good[..20])])).is_err());

        let mut bad_record = fssm(&[(HwStage::Ls, &VS_CODE)]);
        let at = FSSM_HEADER + slot(HwStage::Ls);
        bad_record[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(native_programs(&part(&[(KIND_NATIVE, &xbch(&bad_record))])).is_err());
    }

    #[test]
    fn parses_xdat_metadata_0e_and_raw_0c_fssm() {
        let raw_fssm = fssm(&[(HwStage::Vs, &VS_CODE)]);
        let table_0e: Vec<u8> = [1u32, 2, 0x10, 0x20]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let h10 = [0x11u8; 16];
        let h12 = [0x22u8; 16];
        let data = part(&[
            (KIND_OPTIONS, b"xbop\x01\x00"),
            (KIND_TABLE_0E, &table_0e),
            (KIND_HASH_10, &h10),
            (KIND_HASH_12, &h12),
            (KIND_NATIVE_UNPACKED, &raw_fssm),
        ]);

        let xdat = parse_xdat(&data).unwrap().unwrap();
        assert_eq!(xdat.entries.len(), 5);
        assert_eq!(xdat.xbop, Some(&b"xbop\x01\x00"[..]));
        assert_eq!(xdat.table_0e, [XdatTable0e { dwords: vec![1, 2, 0x10, 0x20] }]);
        assert_eq!(xdat.hash_10, Some(h10));
        assert_eq!(xdat.hash_12, Some(h12));

        let found = native_programs(&data).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].stage, HwStage::Vs);
        assert_eq!(found[0].code, VS_CODE);
    }

    #[test]
    fn rejects_invalid_xdat_xbch_and_fssm_versions() {
        // Unexpected XDAT entries offset (must be 0x24).
        let mut bad_xdat = part(&[(KIND_HASH_12, &[0u8; 16])]);
        bad_xdat[XDAT_OFFSET + 8..XDAT_OFFSET + 12].copy_from_slice(&0x28u32.to_le_bytes());
        assert!(matches!(parse_xdat(&bad_xdat), Err(Malformed(_))));

        // Misaligned / empty 0x0e table.
        assert!(matches!(
            parse_xdat(&part(&[(KIND_TABLE_0E, &[1, 2, 3])])),
            Err(Malformed(_))
        ));
        assert!(matches!(
            parse_xdat(&part(&[(KIND_TABLE_0E, &[])])),
            Err(Malformed(_))
        ));

        // Wrong hash size and bad xbop magic.
        assert!(matches!(
            parse_xdat(&part(&[(KIND_HASH_10, &[0u8; 12])])),
            Err(Malformed(_))
        ));
        assert!(matches!(
            parse_xdat(&part(&[(KIND_OPTIONS, b"nope")])),
            Err(Malformed(_))
        ));

        // Unknown FSSM version and unexpected FSSM slot count.
        let mut bad_fssm_ver = fssm(&[(HwStage::Vs, &VS_CODE)]);
        bad_fssm_ver[4..8].copy_from_slice(&0x102u32.to_le_bytes());
        assert!(matches!(
            native_programs(&part(&[(KIND_NATIVE_UNPACKED, &bad_fssm_ver)])),
            Err(Malformed(_))
        ));
        assert!(matches!(
            native_programs(&part(&[(KIND_NATIVE, &xbch(&bad_fssm_ver))])),
            Err(Malformed(_))
        ));

        let mut bad_slots = fssm(&[(HwStage::Vs, &VS_CODE)]);
        bad_slots[FSSM_HEADER..FSSM_HEADER + 8].copy_from_slice(&7u64.to_le_bytes());
        assert!(matches!(
            native_programs(&part(&[(KIND_NATIVE_UNPACKED, &bad_slots)])),
            Err(Malformed(_))
        ));
    }

    #[test]
    fn reads_es_and_gs_merged_stage_data() {
        // ES with 1 input (v9) and 2 outputs (SV_Position + TEXCOORD0, 8 dwords/vert).
        let mut es_blob = fssm(&[(HwStage::Es, &VS_CODE)]);
        let e = stage_data_at(&es_blob, HwStage::Es);
        es_blob[e + VS_INPUTS..][..5].copy_from_slice(&[6, 0, 7, 9, 2]);
        es_blob[e + VS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        es_blob[e + VS_OUTPUTS..][..12].copy_from_slice(&[0, 0, 0, 0, 0, 0, 6, 1, 1, 0, 0, 0]);
        es_blob[e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        es_blob[e + ES_ITEMSIZE_DWORDS..][..4].copy_from_slice(&8u32.to_le_bytes());
        let rec = e - RECORD_SIZE as usize;
        es_blob[rec + RECORD_LDS..][..4].copy_from_slice(&4096u32.to_le_bytes());

        let es = &native_programs(&part(&[(KIND_NATIVE, &xbch(&es_blob))])).unwrap()[0];
        assert_eq!(es.lds_bytes(), 4096);
        assert_eq!(
            es.io,
            StageIo::Es(EsIo {
                inputs: vec![VsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 7,
                    vgpr: 9,
                }],
                outputs: vec![
                    VsOutput {
                        usage: Usage::Other(0),
                        row: 0,
                        param: 0,
                        fp16: false,
                        pack2: false,
                    },
                    VsOutput {
                        usage: Usage::Generic,
                        row: 1,
                        param: 1,
                        fp16: false,
                        pack2: false,
                    },
                ],
                esgs_itemsize_dwords: 8,
                lds_bytes: 4096,
            })
        );

        // GS with 2 param outputs, max_vertices=3, vertex_size=12 dwords, embedded copy shader.
        let mut gs_blob = fssm(&[(HwStage::Gs, &VS_CODE)]);
        let copy_code = [0x00u8, 0x00, 0x81, 0xBF, 0x00, 0x00, 0x80, 0xBF];
        let copy_offset = (gs_blob.len() - FSSM_HEADER) as u64;
        gs_blob.extend_from_slice(&copy_code);
        let new_payload_len = (gs_blob.len() - FSSM_HEADER) as u64;
        gs_blob[8..16].copy_from_slice(&new_payload_len.to_le_bytes());

        let e = stage_data_at(&gs_blob, HwStage::Gs);
        gs_blob[e + GS_TOPOLOGY] = 2;
        gs_blob[e + GS_INPUT_VERTICES] = 3;
        gs_blob[e + GS_COPY_CODE_OFFSET..][..8].copy_from_slice(&copy_offset.to_le_bytes());
        gs_blob[e + GS_COPY_CODE_SIZE..][..8].copy_from_slice(&(copy_code.len() as u64).to_le_bytes());
        gs_blob[e + GS_OUTPUTS..][..12].copy_from_slice(&[6, 1, 0, 0, 0, 0, 6, 2, 1, 0, 0, 1]);
        gs_blob[e + GS_OUTPUT_COUNT..][..4].copy_from_slice(&2u32.to_le_bytes());
        gs_blob[e + GS_VS_OUT_CONFIG..][..4].copy_from_slice(&2u32.to_le_bytes());
        gs_blob[e + GS_POS_FORMAT..][..4].copy_from_slice(&4u32.to_le_bytes());
        gs_blob[e + GS_CLIP_CNTL..][..4].copy_from_slice(&0x0Fu32.to_le_bytes());
        gs_blob[e + GS_PARAM_DWORDS..][..4].copy_from_slice(&8u32.to_le_bytes());
        gs_blob[e + GS_RING_ITEMSIZE_DWORDS..][..4].copy_from_slice(&36u32.to_le_bytes());
        gs_blob[e + GS_INSTANCE_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        gs_blob[e + GS_MAX_VERTICES..][..4].copy_from_slice(&3u32.to_le_bytes());
        gs_blob[e + GS_VERTEX_SIZE_DWORDS..][..4].copy_from_slice(&12u32.to_le_bytes());
        let rec = e - RECORD_SIZE as usize;
        gs_blob[rec + RECORD_LDS..][..4].copy_from_slice(&8192u32.to_le_bytes());

        let gs = &native_programs(&part(&[(KIND_NATIVE, &xbch(&gs_blob))])).unwrap()[0];
        assert_eq!(gs.lds_bytes(), 8192);
        assert_eq!(
            gs.io,
            StageIo::Gs(GsIo {
                outputs: vec![
                    VsOutput {
                        usage: Usage::Generic,
                        row: 1,
                        param: 0,
                        fp16: false,
                        pack2: false,
                    },
                    VsOutput {
                        usage: Usage::Generic,
                        row: 2,
                        param: 1,
                        fp16: true,
                        pack2: false,
                    },
                ],
                output_topology: 2,
                input_vertices: 3,
                vs_out_config: 2,
                pos_format: 4,
                clip_cntl: 0x0F,
                param_dwords: 8,
                ring_itemsize_dwords: 36,
                instance_count: 1,
                max_vertices: 3,
                vertex_size_dwords: 12,
                lds_bytes: 8192,
                copy_code: copy_code.to_vec(),
            })
        );

        // Out-of-bounds GS copy shader must be rejected.
        gs_blob[e + GS_COPY_CODE_SIZE..][..8].copy_from_slice(&9999u64.to_le_bytes());
        assert!(matches!(
            native_programs(&part(&[(KIND_NATIVE, &xbch(&gs_blob))])),
            Err(Malformed(_))
        ));
    }

    #[test]
    fn reads_ls_and_hs_merged_stage_data() {
        let mut blob = fssm(&[(HwStage::Ls, &VS_CODE), (HwStage::Hs, &PS_CODE)]);
        let ls_e = stage_data_at(&blob, HwStage::Ls);
        blob[ls_e + VS_INPUTS..][..5].copy_from_slice(&[6, 0, 7, 4, 2]);
        blob[ls_e + VS_INPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[ls_e + VS_OUTPUTS..][..6].copy_from_slice(&[6, 0, 0, 0, 0, 0]);
        blob[ls_e + VS_OUTPUT_COUNT..][..4].copy_from_slice(&1u32.to_le_bytes());
        blob[ls_e + ES_ITEMSIZE_DWORDS..][..4].copy_from_slice(&4u32.to_le_bytes());
        let ls_rec = ls_e - RECORD_SIZE as usize;
        blob[ls_rec + RECORD_LDS..][..4].copy_from_slice(&2048u32.to_le_bytes());

        let hs_e = stage_data_at(&blob, HwStage::Hs);
        let hs_rec = hs_e - RECORD_SIZE as usize;
        blob[hs_rec + RECORD_LDS..][..4].copy_from_slice(&16384u32.to_le_bytes());

        let found = native_programs(&part(&[(KIND_NATIVE, &xbch(&blob))])).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].stage, HwStage::Ls);
        assert_eq!(found[0].lds_bytes(), 2048);
        assert_eq!(
            found[0].io,
            StageIo::Ls(LsIo {
                inputs: vec![VsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 7,
                    vgpr: 4,
                }],
                outputs: vec![VsOutput {
                    usage: Usage::Generic,
                    row: 0,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
                lshs_itemsize_dwords: 4,
                lds_bytes: 2048,
            })
        );
        assert_eq!(found[1].stage, HwStage::Hs);
        assert_eq!(found[1].lds_bytes(), 16384);
        assert_eq!(
            found[1].io,
            StageIo::Hs(HsIo {
                lds_bytes: 16384,
                stage_data_len: STAGE_DATA,
            })
        );
    }
}
