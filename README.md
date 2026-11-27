# VIctory on NVIDIA (RTX 5060 8 GB Clean Fork)

A clean, verified **8 GB VRAM** branch (`rtx5060-8gb-clean`) of [trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA) (fork of [Marvisak/VIctory](https://github.com/Marvisak/VIctory), rebased on [readmet3xt-lab/foxhound](https://github.com/readmet3xt-lab/foxhound)), tuned for 8 GB NVIDIA RTX GPUs with native DLSS Super Resolution + DLSS Frame Generation (`2x`–`6x`).

> **This repository contains no game files, code or assets.** You need your own legally obtained Xbox Series v1.0.2.1 build (`game_scarlett_master.exe`). It is not affiliated with or endorsed by the title's developer or publisher, nor by NVIDIA. See [docs/asset-policy.md](docs/asset-policy.md).

---

## Tested Hardware & Verified Results

**Test system:**
- **GPU:** NVIDIA GeForce RTX 5060 8 GB (`driver 610.88`)
- **CPU:** AMD Ryzen 7 5700X (8C / 16T)
- **RAM:** 32 GB
- **OS:** Windows 11 x64

**Measured performance (warm shader/pipeline cache, no `VICTORY_CAPTURE` overhead):**

| Profile | Resolution & Upscaling | Base FPS (cutscenes / open world) | With DLSS FG 4x (effective) | CPU `preparing` | GPU time | DXGI VRAM (`used / budget`) | PSO `being made` / Crashes / Faces |
|---|---|---|---|---|---|---|---|
| **Series S** *(recommended for 8 GB VRAM)* | 720p internal + DLSS Quality (`ideal-20261128-004703`) | **12.3–13.9 FPS** *(up to 14.7–18.0 FPS in cutscenes)* | **~50–55+ FPS** *(cutscenes ~59–72 FPS)* | **38.6–57.0 ms** | **3.4–25.2 ms** | **5882–5981 / 6203–6409 MB** *(4188 MB textures uploaded)* | `0 being made`, `0` crashes, **clean faces & clothes** |
| **Series X** *(option `--series-x`)* | 2560×1440 + DLSS Quality (`ideal-20261128-005131`) | **6.4–9.1 FPS** | **~25–36 FPS** | **76.9–104.1 ms** | **0.0–40.2 ms** | **6658–6719 / 6561–6637 MB** *(1510 MB textures uploaded)* | `270 being made` (1440p warm-up), `0` crashes |

**Visual verification:**
- Opening police station / patrol car cutscenes, prison interrogation scene, seaplane scene, and open-world boat scenes were visually verified via captured PNG frames: no black screen, no chrome/colored skin spots, and no black character faces.

---

## What Was Fixed on `rtx5060-8gb-clean`

### 1. Pipeline warm-up & stall elimination (`P0`)
- **`crates/victory-gpu/src/pipeline/library.rs`:** Ported atomic pipeline library save-retry handling (`e62e8a0`) so failed rename/publish attempts never drop compiled PSOs or reset `added`.
- **`crates/victory-gpu/src/pipeline/make.rs`:** Raised `slow_limit()` to `workers / 2` (6 of 12 workers) so slow shaders do not starve the `early` / `screen` warm-up queues.
- **Launcher defaults:** `VICTORY_PIPELINE_WAIT_MS=100` (prevents dropping first cutscene/menu frames into a black screen), `VICTORY_SCREEN_WAIT_MS=500`, `VICTORY_PIPELINE_WORKERS=12`, `VICTORY_PIPELINE_LOWEST=0`.

### 2. Black screen, descriptor spill & character-face fixes (`P1`)
- **`crates/victory-gpu/src/descriptor/host.rs`:** Fixed `ViewRemap::new()` overlap (`direct = count.min(spill_end)` so `direct` never exceeds `spill_end` and exhausts dynamic spill slots) and extended `slots_showing()` recent table-read protection to `240` frames. Disabled `VICTORY_DESCRIPTOR_REMAP` and `VICTORY_PIPELINE_PIECES` in the launcher (`=0`), returning `dropped without a host pipeline` from `1073–1443` down to `~120–160`.
- **Clean character skin/faces & 8 GB WDDM swap protection (`crates/victory-gpu/src/resource.rs`, `pool.rs`, `host.rs`):**
  - Disabled `VICTORY_SLOT_EVICT` (`=0`) and kept `VICTORY_LEVELS_ON_DEMAND=0` (preserving `VICTORY_HELD_LEVELS=1` mip trimming and avoiding unmapped-level black faces / `0x887A0006` faults on 8 GB cards).
  - Added `pool::texture_capped(..., allow_grow)` so speculative pool heap growth stops strictly at **84% raw DXGI VRAM**, while existing free pool holes (`>= 64 MiB`) can still be reused up to **93%** (and **89%** hard stop when pool holes `< 64 MiB`), paired with `VICTORY_CARD_ROOM_FOR_TARGETS_MB=2176` so pre-world texture streaming leaves room for render targets + DLSS FG buffers.
  - Synchronized the `the texture queue's room:` log line in `host.rs` with the live `crate::resource::room(memory)` check and `used / budget` percentage, and pruned unused `ROOM_REFUSED_MS` state in `resource.rs`.

### 3. CPU / SIMD & shadow-memory throughput (`P2`)
- Ported key CPU & SIMD optimizations from upstream `Vice-Tay` (`TaynanGT/Vice-Tay`) and contributors:
  - `e62e8a0` — Retry failed host pipeline cache saves
  - `369f940` — Collect due texture slots and counters in one traversal (`descriptor/host.rs`)
  - `447dbb1` — AVX2/SSE2 micro-tile texture detiler/tiler and persistent `DetilePool` (`victory-formats/src/detile.rs`)
  - `db28bb8` — O(dirty) `CopiedPages` bitmask, lock-free state mirrors, and AVX2 `Shadow::take` (`shadow.rs`)
  - `1f69d67` — Independent loose/exact shadow allocations outside buffer table locks (`shadow.rs`)
  - `bb529f1` — One-pass submission send layout (`host/send.rs`) and `Shadow::joined()` fast path
  - `dfcddbb` — Batched small host-copy refresh jobs (`wide_ranges`)
  - `62b8829` — Indexed/thread-local scan planning for GPU constant buffer fills (`host/card_steps.rs`)
  - `6141601` — Visit flagged pieces directly during relay landing (`host.rs`)
  - `0c9600f` — Early reuse support for completed relay command lists (`list.rs`)
  - `5c26ec8` — `ReadbackPool` reuse after list retirement (`host.rs`, `texback.rs`)
- Additional cleanups: thread-local tiled staging buffer reuse in `texback::land()` (no per-call `vec![0u8; size]`), early drop of `_mapped` before kernel-safe unmapped refresh paths in `texture.rs`, and reduced `!room` refuse scan limits under consecutive full passes in `draw.rs`.

---

## Build & Run

### Prerequisites
- **OS:** Windows 10 or 11 (x64), AVX2 CPU, 32 GB RAM, SSD, NVIDIA GeForce RTX GPU (Hardware-Accelerated GPU Scheduling enabled for Frame Generation).
- **Toolchain:** [Rust](https://rustup.rs) (pinned `1.88` toolchain; MSVC Build Tools with C++ **or** MinGW-w64 `x86_64-pc-windows-gnu`).
- **Title files:** Your own legal dump containing `game_scarlett_master.exe`.

### 1. Build `victory.exe`
```powershell
git clone https://github.com/d1reevo/zhenek.git victory-nvidia
cd victory-nvidia
git checkout rtx5060-8gb-clean
.\Get-NVIDIA-DLSS.cmd   # fetches NVIDIA Streamline DLSS / FG DLLs into nvidia-sdk\ (git-ignored)

# With MSVC Build Tools:
cargo build --release -p victory

# Or with MSYS2 MinGW-w64:
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
cargo +stable-x86_64-pc-windows-gnu build --release -p victory
```

### 2. Run directly or with the RTX 5060 Ideal Launcher

**Option A — Portable C# Launcher (`GTA6-RTX5060-Ideal.exe`):**
Compile once with .NET Framework (`csc.exe` built into Windows):
```powershell
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /nologo /optimize+ /platform:x64 /out:GTA6-RTX5060-Ideal.exe GTA6-RTX5060-Ideal.cs
```
Run (defaults to `series-s`, DLSS Quality, DLSS FG, saves & cache in `%LOCALAPPDATA%\victory`):
```powershell
.\GTA6-RTX5060-Ideal.exe --game-dir "D:\Games\MyDump"
```
If `--game-dir` is omitted on first launch, the launcher asks interactively for the folder containing `game_scarlett_master.exe` and remembers it under `%LOCALAPPDATA%\victory\game-dir.txt`.

Common launcher flags:
- `--series-s` *(default, recommended for 8 GB VRAM)* / `--series-x` *(1440p profile)*
- `--dlss <off|dlaa|quality|balanced|performance>`
- `--fg <0|2|3|4|5|6>` / `--smooth-motion`
- `--check` *(verify paths and NVIDIA DRS profile without starting the game)*

**Option B — Direct CLI / PowerShell:**
```powershell
.\target\release\victory.exe run "D:\Games\MyDump\game_scarlett_master.exe" --execute --console series-s
```

In-game controls: **F1** opens the live settings menu (`DLSS`, `Frame Generation 2x–6x`, `FPS cap`), **F3** toggles the performance/VRAM overlay.

---

## Credits & Upstream

Built on top of:
- **[readmet3xt-lab/foxhound](https://github.com/readmet3xt-lab/foxhound)** & **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)** (`origin`)
- **[TaynanGT/Vice-Tay](https://github.com/TaynanGT/Vice-Tay)** (`vicetay`) and contributors (`Vladimir-Human`, `Pototoooo`, `KGAT1W`, `im-a-cat`, `regiumlepidi`, `wike1701`, `Ahyildiz`, `fxceless`, `Kemerd`, `Belobug`, `avatarmanz`, `codec-d`, `DexNom`)
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**
- NVIDIA Streamline / DLSS / Reflex SDK (downloaded at runtime by `Get-NVIDIA-DLSS.cmd`, never included in the repo).
