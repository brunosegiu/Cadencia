// Shared common structs, math helpers, procedural noise, and wind sway functions
// prepended to G-Buffer domain shaders, shadow shader, and deferred lighting shader.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view_mat: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    screen_params: vec4<f32>,
    outline_params: vec4<f32>,
    light_view_proj_c1: mat4x4<f32>,
    light_view_proj_c2: mat4x4<f32>,
    cascade_splits: vec4<f32>,
};

struct ObjectData {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>, // x = object_id/255, y = outline_weight, z = crease_sens, w = use_vertex_meta
};

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal_mat: vec4<f32>, // Snorm8x4: xyz = normal, w = packed_mat / 127.0
    @location(2) uv: vec2<f32>,         // Float16x2
    @location(3) color: vec4<f32>,      // Unorm8x4
    @location(4) object_idx: u32,       // VertexStepMode::Instance -> shared ObjectData[] index
};

struct GBufferVertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) view_depth: f32,
    @location(6) @interpolate(flat) vertex_obj_info: vec3<f32>, // (obj_id, outline_w, crease_s)
    @location(7) @interpolate(flat) mat_id: u32,
};

struct GBufferOut {
    @location(0) albedo_weight: vec4<f32>, // rgb = unlit surface albedo, a = outline_weight
    @location(1) normal_id: vec4<f32>,     // rg = octahedral view normal, b = crease_sensitivity, a = object_id
};

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-5));
    let res = select(
        (1.0 - abs(p.yx)) * vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0)),
        p,
        n.z >= 0.0
    );
    return res * 0.5 + 0.5;
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let f = e * 2.0 - 1.0;
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = clamp(-n.z, 0.0, 1.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(127.1, 311.7));
    return fract(dot(q, q + 34.56));
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn apply_vegetation_sway(
    world_pos: vec3<f32>,
    sway_uv: vec2<f32>,
    obj_id_norm: f32,
    time_s: f32,
) -> vec3<f32> {
    let stem_flex = sway_uv.x;
    let leaf_turb = sway_uv.y;
    if (stem_flex <= 0.0001 && leaf_turb <= 0.0001) {
        return world_pos;
    }

    // Per-plant constant seed from packed object_id (identical on every vertex of the same plant!)
    let seed = round(obj_id_norm * 255.0);
    let h1 = fract(sin(seed * 12.9898) * 43758.5453);
    let h2 = fract(sin(seed * 78.2330) * 28461.7193);
    let h3 = fract(sin(seed * 41.6351) * 61539.1847);

    // 1. Coherent Whole-Plant Cantilever Sway in the General Prevailing Wind Direction
    //    (Uses ONLY per-plant seed and time_s — zero per-vertex phase distortion so stiff stems/stalks/plumes never zig-zag!)
    let freq1 = 1.10 + h1 * 0.65;
    let freq2 = freq1 * 1.618034 + h2 * 0.35; // Golden-ratio harmonic avoids periodic loops
    let ph1 = h1 * 6.2831853;
    let ph2 = h2 * 6.2831853;

    // General prevailing wind direction (~32 deg) with small +/- 0.18 rad per-plant spread
    let wind_ang = 0.56 + (h3 - 0.5) * 0.36;
    let wind_dir = vec2<f32>(cos(wind_ang), sin(wind_ang));
    let cross_dir = vec2<f32>(-wind_dir.y, wind_dir.x);

    // Whole plant leans into the general wind direction and relaxes smoothly as gusts pass
    let main_lean = 0.32 + 0.52 * sin(time_s * freq1 + ph1) + 0.24 * sin(time_s * freq2 + ph2);
    let cross_sway = 0.18 * cos(time_s * (freq1 * 0.87) + ph2);
    let whole_xz = (wind_dir * main_lean + cross_dir * cross_sway) * stem_flex;

    // 2. High-Frequency Leaf / Blade Turbulence (only where leaf_turb > 0 on leaves, fronds, and grass blades)
    var turb_xyz = vec3<f32>(0.0);
    if (leaf_turb > 0.0001) {
        let leaf_freq = 2.85 + h2 * 1.35;
        let leaf_phase = dot(world_pos, vec3<f32>(3.1, 2.7, 3.3)) + ph1;
        let flutter_a = sin(time_s * leaf_freq + leaf_phase)
                      + 0.45 * cos(time_s * (leaf_freq * 1.618) + leaf_phase * 1.3);
        let flutter_b = cos(time_s * (leaf_freq * 1.13) + leaf_phase * 0.9);
        let flutter_y = 0.35 * sin(time_s * (leaf_freq * 1.29) + leaf_phase * 1.1);
        turb_xyz = vec3<f32>(
            (wind_dir.x * flutter_a + cross_dir.x * flutter_b * 0.5) * leaf_turb,
            flutter_y * leaf_turb,
            (wind_dir.y * flutter_a + cross_dir.y * flutter_b * 0.5) * leaf_turb
        );
    }

    let dx = whole_xz.x + turb_xyz.x;
    let dz = whole_xz.y + turb_xyz.z;
    let dy = turb_xyz.y - 0.10 * (abs(whole_xz.x) + abs(whole_xz.y));
    return world_pos + vec3<f32>(dx, dy, dz);
}

fn pack_gbuffer(
    albedo: vec3<f32>,
    view_normal: vec3<f32>,
    view_depth: f32,
    outline_w: f32,
    crease_s: f32,
    obj_id: f32,
) -> GBufferOut {
    let dist_outline_fade = 1.0 - smoothstep(140.0, 420.0, view_depth) * 0.65;
    let eff_outline = clamp(outline_w * dist_outline_fade, 0.0, 1.0);
    let oct_n = oct_encode(normalize(view_normal));
    var out: GBufferOut;
    out.albedo_weight = vec4<f32>(clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0)), eff_outline);
    out.normal_id = vec4<f32>(oct_n.x, oct_n.y, clamp(crease_s, 0.0, 1.0), obj_id);
    return out;
}
