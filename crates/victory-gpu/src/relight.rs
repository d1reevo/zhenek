//! Selective shadow color replacement on validated material and depth targets.

use crate::object::This;
use crate::resource::HostTexture;
use std::sync::{Arc, Mutex, OnceLock};

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

pub(crate) struct Material {
    pub(crate) color: Arc<HostTexture>,
    pub(crate) depth: Arc<HostTexture>,
    width: u64,
    height: u32,
}

struct Candidate {
    key: (usize, usize),
    material: Option<Material>,
    draws: u32,
}

static MATERIALS: Mutex<Vec<Candidate>> = Mutex::new(Vec::new());

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VICTORY_SHOW_TARGET").is_ok_and(|mode| {
            ["relight", "albedo", "busiest"]
                .iter()
                .any(|name| mode.trim().eq_ignore_ascii_case(name))
        })
    })
}

pub(crate) fn note_material(color: usize, depth: usize) {
    if color == 0 || depth == 0 || !enabled() {
        return;
    }
    let key = (color, depth);
    let mut tracked = MATERIALS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = tracked.iter_mut().find(|entry| entry.key == key) {
        entry.draws = entry.draws.saturating_add(1);
        return;
    }
    if tracked.len() >= 64 {
        return;
    }
    drop(tracked);
    let material = material_pair(color, depth);
    let mut tracked = MATERIALS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = tracked.iter_mut().find(|entry| entry.key == key) {
        entry.draws = entry.draws.saturating_add(1);
    } else if tracked.len() < 64 {
        tracked.push(Candidate {
            key,
            material,
            draws: 1,
        });
    }
}

fn material_pair(color: usize, depth: usize) -> Option<Material> {
    let color = crate::resource::host_copy(color as This)?;
    let depth = crate::resource::host_copy(depth as This)?;
    // SAFETY: both resources are alive through their host copies.
    let (c, d) = unsafe { (color.texture.GetDesc(), depth.texture.GetDesc()) };
    (c.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS
        && supports(&c)
        && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Format == DXGI_FORMAT_R32G8X24_TYPELESS
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && c.Width == d.Width
        && c.Height == d.Height
        && d.Width >= 960
        && d.Height >= 540)
        .then_some(Material {
            color,
            depth,
            width: d.Width,
            height: d.Height,
        })
}

pub(crate) fn take_material(
    frame: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
) -> Option<Material> {
    use windows::core::Interface;
    if !enabled() {
        return None;
    }
    let material = MATERIALS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .drain(..)
        .filter(|entry| entry.draws >= 96)
        .filter_map(|entry| entry.material.map(|material| (entry.draws, material)))
        .filter(|(_, material)| {
            material.width * u64::from(desc.Height) == desc.Width * u64::from(material.height)
                && material.color.texture.as_raw() != frame.as_raw()
        })
        .max_by_key(|(draws, _)| *draws)
        .map(|(_, material)| material)?;
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        tracing::info!(
            width = material.width,
            height = material.height,
            "matched material and depth targets for shadow replacement"
        )
    });
    Some(material)
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), DescriptorTable(SRV(t0, numDescriptors=3), visibility=SHADER_VISIBILITY_PIXEL)"

cbuffer Parameters : register(b0) {
    uint hasMaterial;
};
Texture2D<float4> frameTex : register(t0);
Texture2D<float4> albedoTex : register(t1);
Texture2D<float> depthTex : register(t2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

[RootSignature(RS)]
float4 vs_main(uint vid : SV_VertexID) : SV_Position {
    return float4(vid == 2 ? 3.0 : -1.0, vid == 1 ? 3.0 : -1.0, 0.0, 1.0);
}

float peak(float3 color) {
    return max(color.r, max(color.g, color.b));
}

float3 atLuma(float3 color, float y, float target, float strength) {
    float3 chroma = color - y;
    float upper = (1.0 - target) / max(peak(chroma), 1e-5);
    float lower = target / max(peak(-chroma), 1e-5);
    return saturate(target + chroma * min(strength, min(upper, lower)));
}

[RootSignature(RS)]
float4 ps_main(float4 position : SV_Position) : SV_Target0 {
    int2 p = int2(position.xy);
    float4 pixel = frameTex.Load(int3(p, 0));
    float3 lit = pixel.rgb;
    float y = dot(lit, LUMA);
    float maxLit = peak(lit);

    // Restrain hot orange highlights without dimming neutral lamps and white skies.
    float glare = smoothstep(0.20, 0.51, lit.r - lit.b)
        * smoothstep(0.58, 0.96, maxLit) * smoothstep(0.23, 0.63, y);
    float3 finalColor = atLuma(lit, y, y * (1.0 - 0.09 * glare), 1.0 - 0.13 * glare);
    float shadow = (1.0 - smoothstep(0.065, 0.14, y)) * (1.0 - smoothstep(0.14, 0.28, maxLit));
    [branch]
    if (hasMaterial == 0u || y >= 0.225 || maxLit >= 0.36) {
        return float4(finalColor, pixel.a);
    }

    uint fw, fh, dw, dh;
    frameTex.GetDimensions(fw, fh);
    depthTex.GetDimensions(dw, dh);
    int2 last = int2(dw - 1, dh - 1);
    int2 q = min(int2(position.xy * (float2(dw, dh) / float2(fw, fh))), last);
    float z = depthTex.Load(int3(q, 0));
    if (z <= 0.0012 || z >= 0.99999) {
        return float4(finalColor, pixel.a);
    }

    // Extend the material treatment to mid-distance surfaces with recoverable color.
    // Near faces retain the original weighting; far geometry and sky fade out.
    float distanceBlend = smoothstep(0.0018, 0.004, z)
        * (1.0 - smoothstep(0.010, 0.027, z));
    float environment = distanceBlend * (1.0 - smoothstep(0.11, 0.225, y))
        * (1.0 - smoothstep(0.22, 0.36, maxLit));
    if (max(shadow, environment) <= 0.0) {
        return float4(finalColor, pixel.a);
    }

    // Accept samples only on a continuous surface, away from silhouette edges.
    int2 qx = q.x < last.x ? q + int2(1, 0) : max(q - int2(1, 0), int2(0, 0));
    int2 qy = q.y < last.y ? q + int2(0, 1) : max(q - int2(0, 1), int2(0, 0));
    float zx = depthTex.Load(int3(qx, 0));
    float zy = depthTex.Load(int3(qy, 0));
    float edge = max(abs(zx - z), abs(zy - z)) / z;
    float surface = (1.0 - smoothstep(0.012, 0.055, edge)) * smoothstep(0.0012, 0.0045, z);
    if (surface <= 0.0) {
        return float4(finalColor, pixel.a);
    }

    float3 albedo = albedoTex.Load(int3(q, 0)).rgb;
    float materialY = dot(albedo, LUMA);
    float available = smoothstep(0.012, 0.055, materialY);
    float3 material = sqrt(albedo);
    // Close, severely underlit surfaces can use more of the source
    // material's color, without lifting the whole game frame.
    float closeFill = smoothstep(0.016, 0.050, z)
        * (1.0 - smoothstep(0.055, 0.095, y)) * smoothstep(0.06, 0.22, materialY);
    float fillY = min(y + lerp(0.114, 0.205, closeFill),
        min(lerp(0.205, 0.30, closeFill), 0.042 + lerp(0.20, 0.35, closeFill) * sqrt(materialY)));
    float materialLuma = dot(material, LUMA);
    // Recover local albedo contrast in dark walls, clothing and props while
    // keeping the local brightness increase small in the broader range.
    float environmentY = min(0.255, y + 0.023 + 0.068 * sqrt(materialY));
    float environmentWeight = 0.46 * environment * step(y, environmentY);
    float shadowWeight = lerp(0.76, 0.88, closeFill) * shadow * step(y, fillY);
    float extent = environmentWeight / max(environmentWeight + shadowWeight, 1e-5);
    float targetY = lerp(fillY, environmentY, extent);
    float chroma = min(lerp(0.68, 0.84, extent), targetY / max(materialLuma, 0.02));
    float3 replacement = atLuma(material, materialLuma, targetY, chroma);
    float coverage = max(shadowWeight, environmentWeight) * surface * available;
    return float4(lerp(finalColor, replacement, coverage), pixel.a);
}
"#;

/// Apply the grade in frames with sustained rendering work. The normal present
/// path handles simple menus and loading screens.
pub(crate) fn world_frame() -> bool {
    static LAST_DRAWN: AtomicU64 = AtomicU64::new(0);
    if !enabled() {
        return false;
    }
    let drawn = crate::draw::drawn();
    drawn.saturating_sub(LAST_DRAWN.swap(drawn, Ordering::Relaxed)) >= 80
}

pub(crate) fn supports(d: &D3D12_RESOURCE_DESC) -> bool {
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1
        && d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && matches!(
            d.Format,
            DXGI_FORMAT_R8G8B8A8_TYPELESS | DXGI_FORMAT_R8G8B8A8_UNORM
        )
}

pub(crate) struct Relight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    srv_heap: ID3D12DescriptorHeap,
    srv_size: u32,
}

// The presenter serializes descriptor writes and waits for the recorded pass
// before the heap can be reused.
unsafe impl Send for Relight {}
unsafe impl Sync for Relight {}

impl Relight {
    /// The source must be in PIXEL_SHADER_RESOURCE and the target in RENDER_TARGET.
    pub(crate) unsafe fn record(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
        material: Option<&Material>,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
    ) {
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let viewport = D3D12_VIEWPORT {
            Width: width as f32,
            Height: height as f32,
            MaxDepth: 1.0,
            ..Default::default()
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        // SAFETY: live texture and render target with the documented states.
        unsafe {
            device.CreateShaderResourceView(
                texture,
                Some(&srv),
                self.srv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            let base = self.srv_heap.GetCPUDescriptorHandleForHeapStart();
            let color_slot = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + self.srv_size as usize,
            };
            let depth_slot = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + 2 * self.srv_size as usize,
            };
            device.CreateShaderResourceView(
                material.map(|m| &m.color.texture),
                Some(&srv),
                color_slot,
            );
            let depth_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: if material.is_some() {
                    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                } else {
                    DXGI_FORMAT_R32_FLOAT
                },
                ..srv
            };
            device.CreateShaderResourceView(
                material.map(|m| &m.depth.texture),
                Some(&depth_srv),
                depth_slot,
            );

            list.RSSetViewports(&[viewport]);
            list.RSSetScissorRects(&[scissor]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetGraphicsRoot32BitConstant(0, u32::from(material.is_some()), 0);
            list.SetGraphicsRootDescriptorTable(
                1,
                self.srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
}

pub(crate) fn pipeline(device: &ID3D12Device) -> Option<&'static Relight> {
    static RELIGHT: OnceLock<Option<Relight>> = OnceLock::new();
    RELIGHT
        .get_or_init(|| match make(device) {
            Ok(r) => {
                tracing::info!("selective material replacement shader ready");
                Some(r)
            }
            Err(e) => {
                tracing::warn!(error = %e, "relight shader unavailable; falling back to plain present");
                None
            }
        })
        .as_ref()
}

fn compile(
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: static source & NUL-terminated names live through the call.
    let res = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("relight"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = res {
        if let Some(err) = errors {
            // SAFETY: compiler-owned error message bytes.
            let text = unsafe {
                std::slice::from_raw_parts(err.GetBufferPointer().cast::<u8>(), err.GetBufferSize())
            };
            tracing::warn!(error = %String::from_utf8_lossy(text), "relight HLSL compile error");
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn make(device: &ID3D12Device) -> windows::core::Result<Relight> {
    let vs = compile(s!("vs_main"), s!("vs_5_1"))?;
    let ps = compile(s!("ps_main"), s!("ps_5_1"))?;
    // SAFETY: compiled shader blobs contain the embedded root signature.
    unsafe {
        let root_slice =
            std::slice::from_raw_parts(ps.GetBufferPointer().cast::<u8>(), ps.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, root_slice)?;
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let raster = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: false.into(),
            ..Default::default()
        };
        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.GetBufferPointer(),
                BytecodeLength: vs.GetBufferSize(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.GetBufferPointer(),
                BytecodeLength: ps.GetBufferSize(),
            },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: raster,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: rtv_formats,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
        let srv_heap: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        Ok(Relight {
            root,
            pso,
            srv_heap,
            srv_size: device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
        })
    }
}
