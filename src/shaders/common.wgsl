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

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>, // x = object_id/255, y = outline_weight, z = crease_sens, w = use_vertex_meta
    skin_pose_r: vec4<f32>,
    skin_pose_l: vec4<f32>,
};

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal_mat: vec4<f32>, // Snorm8x4: xyz = normal, w = packed_mat / 127.0
    @location(2) uv: vec2<f32>,         // Float16x2
    @location(3) color: vec4<f32>,      // Unorm8x4
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

fn apply_vegetation_sway(world_pos: vec3<f32>, sway_uv: vec2<f32>, time_s: f32) -> vec3<f32> {
    let flex = sway_uv.x;
    if (flex <= 0.0001) {
        return world_pos;
    }
    let seed = sway_uv.y;
    let h1 = fract(sin(seed * 12.9898) * 43758.5453);
    let h2 = fract(sin(seed * 78.2330) * 28461.7193);
    let h3 = fract(sin(seed * 41.6351) * 61539.1847);

    let size_rate = mix(1.55, 0.88, clamp(flex * 3.5, 0.0, 1.0));
    let freq1 = (1.55 + h1 * 1.15) * size_rate;
    let freq2 = freq1 * 1.618034 + h2 * 0.65;
    let freq3 = 0.52 + h3 * 0.58;

    let ph1 = h1 * 6.2831853;
    let ph2 = h2 * 6.2831853;
    let ph3 = h3 * 6.2831853;

    let wind_wave = dot(world_pos.xz, vec2<f32>(0.085, 0.065));
    let branch_phase = dot(world_pos, vec3<f32>(1.55, 1.95, 1.65));
    let gust = 0.82 + 0.18 * sin(time_s * freq3 + ph3 + wind_wave * 0.45);

    let sway_a = (0.68 * sin(time_s * freq1 + ph1 + wind_wave)
                + 0.28 * sin(time_s * freq2 + ph2)
                + 0.11 * sin(time_s * (freq2 * 1.42) + branch_phase)) * gust;
    let sway_b = (0.60 * cos(time_s * (freq1 * 0.91) + ph2 + wind_wave)
                + 0.32 * sin(time_s * (freq2 * 1.17) + ph3 + branch_phase * 0.5)) * gust;
    let flutter_y = 0.12 * sin(time_s * (freq2 * 1.35) + branch_phase + ph1) * flex;

    let dir_ang = 0.55 + (h3 - 0.5) * 1.25;
    let dir_x = cos(dir_ang);
    let dir_z = sin(dir_ang);

    let dx = (dir_x * sway_a - dir_z * sway_b * 0.52) * flex;
    let dz = (dir_z * sway_a + dir_x * sway_b * 0.52) * flex;
    let dy = flutter_y - 0.14 * (abs(dx) + abs(dz));
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
