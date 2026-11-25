# AGENTS.md — Cadence Architecture & Engineering Rules

This document defines the architectural invariants, rendering pipeline rules, aesthetic constraints, and verification workflow for **Cadence**, a cross-platform *Ligne Claire* stylized indoor cycling training application written in **Rust** (`wgpu 30.0` + WGSL, `winit 0.30`, `egui 0.37`, `btleplug 0.13`) targeting **Windows, Linux, Android, and Web (WASM)**.

All AI coding agents and contributors must follow these rules whenever modifying or extending the codebase.

---

## 1. Mandatory Build, Test & Verification Rules

1. **Always Pass `--offline` to Cargo:**
   - Every Cargo invocation must include `--offline`:
     - `cargo test --offline`
     - `cargo build --offline`
     - `cargo run --offline -- ...`
     - `cargo check --offline --target wasm32-unknown-unknown --no-default-features`
     - `powershell -ExecutionPolicy Bypass -File .\build_android_apk.ps1` (builds `aarch64-linux-android` release `libcadence_lib.so` and packages/signs `cadence.apk`)
2. **Keep Library and Binary Target Names Distinct (`Cargo.toml`):**
   - In `Cargo.toml`, `[lib]` must remain `name = "cadence_lib"` (`crate-type = ["cdylib", "rlib"]`) and `[[bin]]` must remain `name = "cadence"`. Never give `[lib]` and `[[bin]]` the same name, as Windows MSVC builds will emit `cadence.pdb` output filename collision warnings.
3. **ALWAYS Use the Blender MCP for 3D Modeling (NEVER Procedural Rust Meshes for Props/Models):**
   - All 3D models (vegetation, trees, palms, bushes, rocks, animals, buildings, vehicles, avatars, props) **MUST** be created and exported as `.glb` files into `assets/models/` using the **Blender MCP** (`execute_blender_code` / `blender` MCP tools) and loaded via `load_glb` in `src/assets.rs`.
   - **NEVER** write procedural mesh-building code in Rust to generate 3D prop, vegetation, rock, or animal models.
   - When adding species/prop variations (e.g., 10 variants of a tree, bush, palm, or rock species, or animals like horses, cows, sheep, mulitas), author and export the `.glb` models in Blender via the Blender MCP.
   - **NEVER call `bpy.ops.wm.read_factory_settings()` inside `execute_blender_code`**: `read_factory_settings` unloads all active Blender addons (including the `Blender MCP` addon server itself) and drops the MCP connection. Always clear the scene by removing objects/meshes via `for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)`.
4. **Mandatory 4-Step Verification Before Every Commit:**
   - **Step 1 — Unit Tests:** Run `cargo test --offline` (all physics, spline, octree frustum culling, TAA Halton jitter, and BLE FTMS/CPS/CSC/HRS packet parser tests must pass with zero warnings).
   - **Step 2 — WebAssembly Target Check:** Run `cargo check --offline --target wasm32-unknown-unknown --no-default-features` to ensure platform-agnostic WebGPU/WASM compatibility.
   - **Step 3 — Visual Screenshot Verification:** Render and visually inspect screenshots via the built-in headless renderer (which warms up 8 frames for full 8-phase Halton TAA convergence):
     ```powershell
     cargo run --offline -- --screenshot screenshots/mountain.png --biome mountain --distance 250
     cargo run --offline -- --screenshot screenshots/coastal.png --biome coastal --distance 180
     cargo run --offline -- --screenshot screenshots/rural.png --biome rural --distance 150
     cargo run --offline -- --screenshot screenshots/main_menu.png --screen main_menu --biome mountain
     cargo run --offline -- --screenshot screenshots/settings.png --screen settings --biome coastal
     ```
   - **Step 4 — Clean Git Commit:** Commit changes cleanly to git once tests, WASM check, and screenshots are verified.

---

## 2. Visual & Aesthetic Pillars (*Ligne Claire*)

- **Franco-Belgian *Ligne Claire* Comic Illustration Style** (inspired by Hergé, Moebius, and classic European cycling posters):
  - **1-Pixel Razor-Sharp Ink Outlines:** Computed in screen space (`src/shaders/post.wgsl`) using 1-pixel forward differences (`right` and `down` neighbors only) across linearized hardware depth, per-object IDs, and architectural-only normal creases.
  - **No Terrain or Avatar Facet Lines:** `crease_sensitivity` must be `0.0` on terrain, roads, rocks, foliage, and cyclists so low-poly triangles never show internal wireframe facet lines. Only architectural buildings use `crease_sensitivity > 0.0`.
  - **Stepped Cel-Shading & Warm Atmospheric Palette:** Clean 2-band/3-band stepped sun lighting with warm golden sunlight, cool slate-blue shadow tinting, and distance exponential haze. No glossy PBR specular noise or neon sci-fi UI clutter.

---

## 3. Codebase Module Map

| Path | Responsibility |
| :--- | :--- |
| `src/main.rs` | Desktop CLI entry point supporting `--level <cerro\|rambla\|campo>`, `--screenshot <path>`, `--biome <name>`, `--screen <main_menu\|world_select\|settings\|gear\|ingame>`, `--camera <mode>`, `--width <px>`, `--height <px>`. |
| `src/lib.rs` | Application state loop (`CadenceSim`, `WindowGpuState`), desktop `run_windowed`, WASM `run_web`, Android `android_main`, and 8-frame TAA-converged `render_headless_screenshot_screen`. |
| `src/assets.rs` | Compressed 24-byte `SceneVertex` format, `SceneVertex::new_packed_prop` per-vertex metadata packing, and `AssetBank` (procedural + embedded glTF models). |
| `src/world/mod.rs` | `BiomeId` (`MountainClimb`, `CoastalPromenade`, `RollingRuralGravel`), closed-loop 3D Catmull-Rom `RouteSpline`, and world unit tests. |
| `src/world/terrain.rs` | 3D `WorldOctree` (`OCTREE_MAX_DEPTH = 4`, `MAX_OCTREE_SECTORS = 126`), `WorldSector` leaf chunks, Gribb-Hartmann 6-plane `Frustum`, procedural road/terrain/wire/water mesh generation, and biome prop placement. |
| `src/world/riders.rs` | Articulated cyclist + bicycle hierarchy (`build_rider_draw_parts`), `RiderPoseMode::Pedaling` (in-game) and `RiderPoseMode::MenuStopped` (main menu foot-on-ground pose), and AI peloton riders. |
| `src/physics.rs` | Real-time cycling physics (`step_physics`): gravity, rolling resistance, aerodynamic drag, peloton drafting (`compute_draft_factor`), virtual shifting, cornering lean, and `estimate_cadence_from_power` heuristic when a trainer outputs `0` RPM. |
| `src/hardware/mod.rs` | Cross-platform BLE Smart Trainer & sensor manager (`FTMS 0x1826`, `CPS 0x1818`, `CSC 0x1816`, `HRS 0x180D`) + `SimulatedHardware` fallback supporting `ERG` and `SIM` grade resistance modes. |
| `src/ui/mod.rs` | Screen navigation (`AppScreen::MainMenu`, `WorldSelect`, `Settings`, `AvatarEditor`, `InGame`) and stylized `egui` HUD overlay. |
| `src/renderer/mod.rs` | `LigneClaireRenderer`: WebGPU buffers, pipelines, pre-baked per-sector CSM-size-tiered prop meshes, CPU octree frustum traversal, compute dispatches, CSM shadow passes, G-Buffer pass, outline pass, and TAA pass. |

---

## 4. Rendering Pipeline & GPU Architecture Rules

### 4.1 Compressed 24-Byte Vertex Format (`SceneVertex` in `src/assets.rs`)
Every 3D mesh vertex is strictly **24 bytes** (`#[repr(C)]`):
- `@location(0) pos`: `Float32x3` (12 bytes, offset 0) — local or pre-baked world position.
- `@location(1) normal_mat`: `Snorm8x4` (4 bytes, offset 12) — `xyz` = unit normal, `w` = `material_type` (`0..=6`) or packed `(material_type & 7) | (outline_q3 << 3) | (crease_bit << 6)` for baked static sector props.
- `@location(2) uv`: `Float16x2` (4 bytes, offset 16) — IEEE 754 half-precision UVs.
- `@location(3) color`: `Unorm8x4` (4 bytes, offset 20) — `rgb` = vertex albedo/tint, `a` = `1.0` or packed `object_id` (`0..255`) for baked static sector props.

### 4.2 Pre-Baked Per-Sector CSM-Tiered Static Props (`upload_world`)
- **Never iterate over individual static props or invert static prop transform matrices on the CPU per frame.**
- At `upload_world` time, static props within each 3D octree leaf sector (`WorldSector`) are pre-transformed into world space using `SceneVertex::new_packed_prop` (preserving each individual prop's `object_id`, `outline_weight`, and `crease_sensitivity` in the 24-byte vertex) and merged into at most **4 CSM size-tier meshes per sector**:
  - **Tier 0:** Non-shadow props (`cast_shadow = false`)
  - **Tier 1:** Small shadow casters (`world_radius < 0.90 m`, rendered in CSM Cascade 0 only)
  - **Tier 2:** Medium shadow casters (`0.90 m <= world_radius < 2.50 m`, rendered in CSM Cascades 0 & 1)
  - **Tier 3:** Large shadow casters (`world_radius >= 2.50 m`, rendered in CSM Cascades 0, 1 & 2)
- Each sector therefore emits at most **7 static `GpuDrawDesc` entries** (`road`, `terrain`, `wire`, `prop_tiers[0..4]`), uploaded once per world load and sorted by uniform slot (`uniform_idx`) so the entire static world requires at most **4 `set_bind_group` calls** in the G-Buffer pass and **1 `set_bind_group` call** per shadow cascade.

### 4.3 Two-Level Frustum Culling (CPU 3D Octree + GPU 6-Plane Indirect Compute)
1. **CPU Top-Down 3D Octree Frustum Culling (`WorldOctree::query_frustum`):**
   - Each frame, the CPU extracts 6 Gribb-Hartmann frustum planes (`Frustum::from_view_proj`) for the unjittered camera view-projection matrix and the 3 CSM orthographic light matrices.
   - The CPU traverses `geo.octree.query_frustum` top-down, setting visible leaf sector bits in 128-bit bitmasks (`cam_sector_mask`, `shadow_sector_mask`, and per-cascade `csm_sector_masks`). Bit `127` is reserved for dynamic cyclists.
2. **Multi-Dispatch GPU Per-Draw-Call 6-Plane Frustum & CSM Size Culling (`src/shaders/cull_indirect.wgsl`, Pass 0b):**
   - Culling runs in **4 separate compute dispatches** using **4 separate indirect draw argument buffers** (`cam_indirect_buf` for the color G-Buffer view and `csm_indirect_bufs[0..3]` for each CSM shadow cascade) and dedicated `SinglePassCullUniforms` buffers:
     - **Dispatch 0 (Color View):** Checks the sector bit in `cam_sector_mask`, tests the draw call's world bounding sphere (`pos_radius`) against the 6 camera frustum planes and `max_prop_draw_distance_m`, and writes `DrawIndexedIndirectCmd` into `cam_indirect_buf`.
     - **Dispatches 1..=3 (CSM Cascades C0, C1, C2):** Each cascade dispatch checks its per-cascade `csm_sector_masks[ci]`, filters the draw call's caster radius (`caster_r`) against `[csm_min_caster_r[ci], csm_max_caster_r[ci]]`, tests the 6 orthographic cascade frustum planes, and writes `DrawIndexedIndirectCmd` into `csm_indirect_bufs[ci]`.

### 4.4 Pre-Render Compute Skinning (`src/shaders/skinning.wgsl`, Pass 0a)
- **No inline skinning in vertex shaders.**
- Articulated cyclist and quadruped animal meshes are deformed in `skinning.wgsl` prior to rendering, reading unskinned template vertices from `skin_src_vb` and writing world-ready skinned vertices into the dynamic tail of `mega_vb` (`skinned_base_vertex..`).
- `shadow.wgsl` and `gbuf_skinned.wgsl` consume the pre-skinned vertices directly as rigid geometry.

### 4.5 Shared Shader Common Library (`src/shaders/common.wgsl`) & Domain-Permuted Unlit G-Buffer Pass (`Pass 4`)
- **No monolithic G-Buffer uber-shader or material `if/else` chains.**
- Shared structs (`FrameUniforms`, `ObjectUniforms`, `VertexInput`, `GBufferVertOut`, `GBufferOut`), octahedral normal encoding/decoding (`oct_encode`, `oct_decode`), procedural noise (`hash21`, `value_noise`), and non-periodic vegetation wind sway (`apply_vegetation_sway`) live in `src/shaders/common.wgsl` and are prepended at shader module creation time.
- **No lighting or shadow sampling in the G-Buffer pass.** Each material domain has its own dedicated G-Buffer shader and `wgpu::RenderPipeline` outputting only **unlit surface albedo + `outline_weight`** to `@location(0) albedo_tex` (`Rgba8Unorm`), **octahedral view normal + `crease_sensitivity` + `object_id / 255.0`** to `@location(1) normal_tex` (`Rgba8Unorm`), and hardware depth to `scene_depth_tex` (`Depth32Float`):
  - `src/shaders/gbuf_road.wgsl` (`gbuf_road_tarmac_pipeline` & `gbuf_road_gravel_pipeline`)
  - `src/shaders/gbuf_terrain.wgsl` (`gbuf_terrain_natural_pipeline` & `gbuf_terrain_coastal_pipeline`)
  - `src/shaders/gbuf_prop.wgsl` (`gbuf_prop_pipeline`: static props, rocks, architecture, and wind-swaying foliage)
  - `src/shaders/gbuf_skinned.wgsl` (`gbuf_skinned_pipeline`: dynamic cyclists and compute-skinned animals with zero procedural noise calls)
  - `src/shaders/gbuf_water.wgsl` (`gbuf_water_pipeline`: opaque water rendered directly into the G-Buffer with `object_id = 4`)

### 4.6 3-Cascade Directional Shadow Maps (`src/shaders/shadow.wgsl`, Passes 1..3) & Deferred Lighting + Outline Pass (`src/shaders/post.wgsl`, Pass 5)
- **Shadow Map Format:** `2048×2048×3` `Depth32Float` `D2Array` (`NUM_SHADOW_CASCADES = 3`), orthographic radii `[12.0 m, 42.0 m, 128.0 m]`, view-depth splits `[18.0 m, 68.0 m, 215.0 m]`, caster radius filters `min = [0.0 m, 0.90 m, 2.50 m]`, `max = [28.0 m, 45.0 m, 90.0 m]`.
- **Pass 5 — Fullscreen Deferred Lighting & *Ligne Claire* Ink Outlines (`src/shaders/post.wgsl`):**
  - Reconstructs 3D `world_pos` from `scene_depth_tex` (`frame.inv_view_proj`) and transforms the decoded view normal to world space (`view_to_world_normal`) so lighting and shadow sampling execute **at most once per visible screen pixel** with zero overdraw waste.
  - Uses a single unified 4-tap hardware-bilinear comparison PCF function (`sample_shadow_cascade`) across all 3 CSM cascades.
  - Evaluates stepped *Ligne Claire* cel-shading, Fresnel rim lighting, opaque water specular sun-path shimmer (`object_id == 4`), exponential distance fog, and 1-pixel forward-difference ink outlines (`ink_mask`), writing perceptual sRGB color + `ink_mask` (`.a`) into `post_color_tex` for Pass 5b (`src/shaders/taa.wgsl`).
- **Pass 5b (`taa.wgsl`):**
  - Applies 8-sample Halton(2, 3) sub-pixel projection jitter (`halton_2_3_jitter_px`), 3×3 closest-depth velocity dilation, unjittered world-space reprojection (`c_uv - jitter_uv`), 5-tap Filmic Catmull-Rom history sampling, and YCoCg variance clipping ($\mu \pm \gamma \sigma$, tightened on dynamic rider object IDs `>= 180` and fast motion).
  - Applies Karis luminance-weighted temporal blending and contrast-adaptive sharpening (boosted along `ink_mask` edges), writing the clean resolved sRGB color to `@location(1)` (`taa_history_tex[curr_idx]`) and the paper-grained display color to `@location(0)` (`target_view`).
  - Automatically resets TAA history on camera mode cuts, biome switches, or window resizes.

---

## 5. UI, Screen Flow & Physics Rules

1. **Screen Navigation Flow (`src/ui/mod.rs`):**
   - **Main Menu (`AppScreen::MainMenu`):** Features a random 3D world background (`BiomeId::random()`), the player avatar stationary on the right (`RiderPoseMode::MenuStopped`, right foot resting on the ground), and `RIDE`, `SETTINGS`, and `GEAR` buttons on the left.
   - **World Select (`AppScreen::WorldSelect`):** Opened by pressing `RIDE`; displays the 3 worlds (**Cerro** / `MountainClimb`, **Rambla** / `CoastalPromenade`, **Campo** / `RollingRuralGravel`) with elevation profile sparklines. Clicking a card launches that world. Players cannot switch worlds on the fly in-game.
   - **Settings (`AppScreen::Settings`):** Two-column Game & Graphics settings screen (outline weight, draw distance, FOV offset, directional shadows toggle, TAA toggle, and BLE/Simulator trainer controls).
   - **Gear / Avatar Editor (`AppScreen::AvatarEditor`):** Intentionally blank screen with a top-left `BACK` pill returning to the Main Menu.
   - **Direct Level Debugging CLI:** `--level <cerro|rambla|campo>` bypasses the Main Menu and launches directly into `AppScreen::InGame`.
2. **In-Game HUD Invariants (`draw_hud` in `src/ui/mod.rs`):**
   - **Top-Left Power Card:** Displays only Wattage (`W`), 6-zone power bar, `W/KG`, `AVG` watts, and `RPM` cadence (no circular leaf dial next to the card, no heart rate `BPM`, and no `GEAR` footer).
   - **Top-Center Route Card:** Ride timer (`elapsed_time_s`) starts at `0:00:00` upon entering a map; no fake XP/drops counter.
   - **Top-Right Minimap:** Clean zoomed route corridor minimap with grade badge (no circular floating buttons above the minimap).
   - **Bottom Action Pill:** Buttons are `Take Screenshot`, `Settings`, and `Camera View`.
3. **Zero-Cadence Power Heuristic (`src/physics.rs`):**
   - Some indoor smart trainers transmit power (`power_w > 0`) but always report `cadence_rpm == 0`. `step_physics` uses `estimate_cadence_from_power(power_w, speed_mps)` whenever `cadence_rpm <= 0.5 && power_w > 5.0` so the cyclist's legs and cranks always animate realistically.
