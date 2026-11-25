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
    obj_info: vec4<f32>, // x = object_id/255, y = outline_weight, z = crease_sens, w = mat_override
    skin_pose_r: vec4<f32>,
    skin_pose_l: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal_mat: vec4<f32>, // Snorm8x4: xyz = normal, w = material_type / 127.0
    @location(2) uv: vec2<f32>,         // Float16x2
    @location(3) color: vec4<f32>,      // Unorm8x4
};

struct SceneOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) material_type: u32,
    @location(6) view_depth: f32,
    @location(7) @interpolate(flat) vertex_obj_info: vec3<f32>,
};

struct GBufferOut {
    @location(0) lit_color: vec4<f32>, // rgb = lit color, a = outline_weight
    @location(1) normal_id: vec4<f32>, // rg = octahedral view normal, b = crease_sensitivity, a = object_id
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
    // Per-instance randomized parameters from seed (incommensurate frequencies prevent periodicity)
    let h1 = fract(sin(seed * 12.9898) * 43758.5453);
    let h2 = fract(sin(seed * 78.2330) * 28461.7193);
    let h3 = fract(sin(seed * 41.6351) * 61539.1847);

    // Faster, livelier frequency for small plants/blades (small flex), slower majestic sway for tall trees
    let size_rate = mix(1.55, 0.88, clamp(flex * 3.5, 0.0, 1.0));
    let freq1 = (1.55 + h1 * 1.15) * size_rate;
    let freq2 = freq1 * 1.618034 + h2 * 0.65; // Golden-ratio harmonic avoids periodic loops
    let freq3 = 0.52 + h3 * 0.58;             // Slow non-periodic breeze swell envelope

    let ph1 = h1 * 6.2831853;
    let ph2 = h2 * 6.2831853;
    let ph3 = h3 * 6.2831853;

    // Spatial wind wave + local branch/blade turbulence so individual stems/fronds ripple organically
    let wind_wave = dot(world_pos.xz, vec2<f32>(0.095, 0.075));
    let branch_phase = dot(world_pos, vec3<f32>(2.15, 2.65, 2.25));
    let gust = 0.85 + 0.35 * sin(time_s * freq3 + ph3 + wind_wave * 0.45);

    let sway_a = (0.72 * sin(time_s * freq1 + ph1 + wind_wave)
                + 0.34 * sin(time_s * freq2 + ph2)
                + 0.22 * sin(time_s * (freq2 * 1.42) + branch_phase)) * gust;
    let sway_b = (0.64 * cos(time_s * (freq1 * 0.91) + ph2 + wind_wave)
                + 0.44 * sin(time_s * (freq2 * 1.17) + ph3 + branch_phase * 0.6)) * gust;
    let flutter_y = 0.24 * sin(time_s * (freq2 * 1.35) + branch_phase + ph1) * flex;

    let dir_ang = 0.55 + (h3 - 0.5) * 1.25;
    let dir_x = cos(dir_ang);
    let dir_z = sin(dir_ang);

    let dx = (dir_x * sway_a - dir_z * sway_b * 0.52) * flex;
    let dz = (dir_z * sway_a + dir_x * sway_b * 0.52) * flex;
    let dy = flutter_y - 0.14 * (abs(dx) + abs(dz));
    return world_pos + vec3<f32>(dx, dy, dz);
}

// Skinning is pre-evaluated in skinning.wgsl compute shader prior to rendering.
@vertex
fn vs_scene(in: VertexInput) -> SceneOut {
    let use_vtx_meta = obj.obj_info.w > 0.5;
    var wp = obj.model * vec4<f32>(in.pos, 1.0);
    if (use_vtx_meta && in.uv.x > 0.0001) {
        wp = vec4<f32>(apply_vegetation_sway(wp.xyz, in.uv, frame.camera_pos.w), 1.0);
    }
    let local_nrm = normalize(in.normal_mat.xyz);
    let raw_mat = u32(round(max(in.normal_mat.w * 127.0, 0.0)));
    let mat_id = raw_mat & 7u;
    let wn = normalize((obj.normal_mat * vec4<f32>(local_nrm, 0.0)).xyz);
    let vn = normalize((frame.view_mat * vec4<f32>(wn, 0.0)).xyz);
    let vp = frame.view_mat * wp;

    let v_obj_id = select(obj.obj_info.x, in.color.a, use_vtx_meta);
    let v_outline_w = select(obj.obj_info.y, f32((raw_mat >> 3u) & 7u) / 7.0, use_vtx_meta);
    let v_crease_s = select(obj.obj_info.z, select(0.0, 0.45, ((raw_mat >> 6u) & 1u) != 0u), use_vtx_meta);

    var out: SceneOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = wn;
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = vec4<f32>(in.color.rgb * obj.tint.rgb, 1.0);
    out.material_type = mat_id;
    out.view_depth = -vp.z;
    out.vertex_obj_info = vec3<f32>(v_obj_id, v_outline_w, v_crease_s);
    return out;
}

fn sample_cascade_layer_pcf(shadow_pos: vec4<f32>, layer: i32, bias: f32, filter_radius: f32) -> f32 {
    let ndc = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z;
    let texel = 1.0 / 2048.0;
    let cmp_depth = depth - bias;

    var sum = 0.0;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel * filter_radius;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, layer, cmp_depth);
        }
    }
    let in_bounds = f32(
        uv.x >= 0.005 && uv.x <= 0.995 &&
        uv.y >= 0.005 && uv.y <= 0.995 &&
        depth >= 0.0 && depth <= 1.0
    );
    return mix(1.0, sum / 9.0, in_bounds);
}

fn sample_cascade_layer_pcf4(shadow_pos: vec4<f32>, layer: i32, bias: f32, filter_radius: f32) -> f32 {
    let ndc = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z;
    let step_uv = (1.0 / 2048.0) * 0.65 * filter_radius;
    let cmp_depth = depth - bias;

    var sum = 0.0;
    sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-step_uv, -step_uv), layer, cmp_depth);
    sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( step_uv, -step_uv), layer, cmp_depth);
    sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-step_uv,  step_uv), layer, cmp_depth);
    sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( step_uv,  step_uv), layer, cmp_depth);

    let in_bounds = f32(
        uv.x >= 0.005 && uv.x <= 0.995 &&
        uv.y >= 0.005 && uv.y <= 0.995 &&
        depth >= 0.0 && depth <= 1.0
    );
    return mix(1.0, sum * 0.25, in_bounds);
}

fn sample_cascade_layer_fast1(shadow_pos: vec4<f32>, layer: i32, bias: f32) -> f32 {
    let ndc = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z;
    let s = textureSampleCompareLevel(shadow_map, shadow_sampler, uv, layer, depth - bias);
    let in_bounds = f32(
        uv.x >= 0.005 && uv.x <= 0.995 &&
        uv.y >= 0.005 && uv.y <= 0.995 &&
        depth >= 0.0 && depth <= 1.0
    );
    return mix(1.0, s, in_bounds);
}

fn sample_csm_shadow(world_pos: vec3<f32>, world_normal: vec3<f32>, view_depth: f32, ndotl: f32) -> f32 {
    if (frame.cascade_splits.w < 0.5 || view_depth >= frame.cascade_splits.z) {
        return 1.0;
    }
    let slope = clamp(1.0 - ndotl, 0.0, 1.0);

    if (view_depth < frame.cascade_splits.x) {
        let wp0 = vec4<f32>(world_pos + world_normal * (0.004 * slope), 1.0);
        let sp0 = frame.light_view_proj * wp0;
        let ndc0 = sp0.xyz / sp0.w;
        let box0 = max(abs(ndc0.x), abs(ndc0.y));
        if (box0 < 0.94 && ndc0.z >= 0.0 && ndc0.z <= 1.0) {
            let bias0 = max(0.00018 * slope, 0.00005);
            let s0 = sample_cascade_layer_pcf(sp0, 0, bias0, 1.10);
            let blend1 = max(
                smoothstep(0.80, 0.94, box0),
                smoothstep(frame.cascade_splits.x * 0.82, frame.cascade_splits.x, view_depth)
            );
            if (blend1 > 0.01) {
                let wp1 = vec4<f32>(world_pos + world_normal * (0.012 * slope), 1.0);
                let sp1 = frame.light_view_proj_c1 * wp1;
                let bias1 = max(0.00030 * slope, 0.00010);
                let s1 = sample_cascade_layer_pcf4(sp1, 1, bias1, 1.0);
                return mix(s0, s1, blend1);
            }
            return s0;
        }
    }

    if (view_depth < frame.cascade_splits.y) {
        let wp1 = vec4<f32>(world_pos + world_normal * (0.012 * slope), 1.0);
        let sp1 = frame.light_view_proj_c1 * wp1;
        let ndc1 = sp1.xyz / sp1.w;
        let box1 = max(abs(ndc1.x), abs(ndc1.y));
        if (box1 < 0.95 && ndc1.z >= 0.0 && ndc1.z <= 1.0) {
            let bias1 = max(0.00030 * slope, 0.00010);
            let s1 = sample_cascade_layer_pcf4(sp1, 1, bias1, 1.0);
            let blend2 = max(
                smoothstep(0.82, 0.95, box1),
                smoothstep(frame.cascade_splits.y * 0.84, frame.cascade_splits.y, view_depth)
            );
            if (blend2 > 0.01) {
                let wp2 = vec4<f32>(world_pos + world_normal * (0.032 * slope), 1.0);
                let sp2 = frame.light_view_proj_c2 * wp2;
                let bias2 = max(0.00042 * slope, 0.00015);
                let s2 = sample_cascade_layer_fast1(sp2, 2, bias2);
                return mix(s1, s2, blend2);
            }
            return s1;
        }
    }

    let wp2 = vec4<f32>(world_pos + world_normal * (0.032 * slope), 1.0);
    let sp2 = frame.light_view_proj_c2 * wp2;
    let ndc2 = sp2.xyz / sp2.w;
    let box2 = max(abs(ndc2.x), abs(ndc2.y));
    if (box2 < 0.97 && ndc2.z >= 0.0 && ndc2.z <= 1.0) {
        let bias2 = max(0.00042 * slope, 0.00015);
        let s2 = sample_cascade_layer_fast1(sp2, 2, bias2);
        let fade_far = max(
            smoothstep(0.84, 0.97, box2),
            smoothstep(frame.cascade_splits.z * 0.80, frame.cascade_splits.z, view_depth)
        );
        return mix(s2, 1.0, fade_far);
    }
    return 1.0;
}

@fragment
fn fs_scene(in: SceneOut) -> GBufferOut {
    let N = normalize(in.world_normal);
    let L = normalize(frame.sun_dir.xyz);
    let V = normalize(frame.camera_pos.xyz - in.world_pos);
    let ndotl = dot(N, L);

    var albedo = in.color.rgb;
    var mat_type = in.material_type;
    var outline_w = in.vertex_obj_info.y;
    var crease_s = in.vertex_obj_info.z;
    let obj_id = in.vertex_obj_info.x;

    // Procedural Ligne Claire surface texture & markings
    if (mat_type == 1u) {
        let n_big = value_noise(in.world_pos.xz * 0.35);
        let n_med = value_noise(in.world_pos.xz * 1.9);
        let n_fine = value_noise(in.world_pos.xz * 6.5);
        albedo *= 0.92 + 0.14 * n_big + 0.06 * n_med + 0.04 * n_fine;
        let rut_jit = (n_big - 0.5) * 0.04;
        let rut_l = exp(-pow((in.uv.x - 0.34 + rut_jit) * 9.5, 2.0));
        let rut_r = exp(-pow((in.uv.x - 0.66 - rut_jit) * 9.5, 2.0));
        albedo *= 1.0 - 0.08 * (rut_l + rut_r);
        let verge = smoothstep(0.41, 0.50, abs(in.uv.x - 0.5));
        albedo = mix(albedo, vec3<f32>(0.58, 0.52, 0.36), verge * 0.60);
        let p_uv = in.world_pos.xz * 2.8 + vec2<f32>(n_big * 1.4, n_med * 1.4);
        let p_cell = floor(p_uv);
        let p_f = fract(p_uv) - 0.5;
        let p_seed = hash21(p_cell);
        if (p_seed > 0.72) {
            let off = vec2<f32>(hash21(p_cell + 2.3) - 0.5, hash21(p_cell + 9.7) - 0.5) * 0.68;
            let rad = 0.035 + hash21(p_cell + 4.1) * 0.045;
            let d_p = length(p_f - off);
            let dot_mask = smoothstep(rad, rad * 0.45, d_p) * smoothstep(55.0, 6.0, in.view_depth);
            albedo = mix(albedo, vec3<f32>(0.36, 0.29, 0.22), dot_mask * 0.58);
        }
        outline_w = 0.28;
        crease_s = 0.0;
    } else if (mat_type == 3u) {
        let n_asp = value_noise(in.world_pos.xz * 1.2);
        albedo *= 0.95 + 0.09 * n_asp;
        let u = in.uv.x;
        let d_dist = in.uv.y;
        let gold_l = smoothstep(0.011, 0.005, abs(u - 0.487));
        let gold_r = smoothstep(0.011, 0.005, abs(u - 0.513));
        let gold_mask = max(gold_l, gold_r);
        albedo = mix(albedo, vec3<f32>(0.84, 0.70, 0.41), gold_mask);
        let dash = step(0.46, fract(d_dist * 0.09));
        let lane_l = smoothstep(0.011, 0.005, abs(u - 0.24)) * dash;
        let lane_r = smoothstep(0.011, 0.005, abs(u - 0.76)) * dash;
        let edge_line = smoothstep(0.012, 0.006, abs(u - 0.04)) + smoothstep(0.012, 0.006, abs(u - 0.96));
        albedo = mix(albedo, vec3<f32>(0.87, 0.84, 0.78), clamp(max(max(lane_l, lane_r), edge_line), 0.0, 1.0));
        outline_w = 0.25;
        crease_s = 0.0;
    } else if (mat_type == 6u) {
        let tile_uv = vec2<f32>(in.uv.x * 0.82, in.uv.y * 0.55);
        let cell = floor(tile_uv);
        let f_uv = fract(tile_uv);
        let tone = 0.94 + 0.11 * hash21(cell);
        albedo *= tone;
        let seam_x = smoothstep(0.055, 0.02, min(f_uv.x, 1.0 - f_uv.x));
        let seam_y = smoothstep(0.045, 0.016, min(f_uv.y, 1.0 - f_uv.y));
        let seam = max(seam_x, seam_y) * smoothstep(130.0, 16.0, in.view_depth);
        albedo = mix(albedo, vec3<f32>(0.42, 0.35, 0.29), seam * 0.55);
        outline_w = 0.35;
        crease_s = 0.15;
    } else if (mat_type == 0u) {
        let n_mottle = value_noise(in.world_pos.xz * 0.12) * 0.65 + value_noise(in.world_pos.xz * 0.55) * 0.35;
        albedo *= 0.89 + 0.22 * n_mottle;
        let steep = smoothstep(0.76, 0.44, N.y);
        let strata = sin(in.world_pos.y * 2.2 + value_noise(in.world_pos.xz * 0.25) * 3.5);
        let strata_col = vec3<f32>(0.64, 0.57, 0.46) * (0.88 + 0.16 * strata + 0.12 * n_mottle);
        albedo = mix(albedo, strata_col, steep * 0.80);
    } else if (mat_type == 2u) {
        let n_leaf = value_noise(in.world_pos.xz * 1.8 + in.world_pos.y * 1.1);
        albedo *= 0.90 + 0.18 * n_leaf;
    }

    let shadow_vis = sample_csm_shadow(in.world_pos, N, in.view_depth, max(ndotl, 0.0));
    let diffuse_raw = ndotl * 0.5 + 0.5;
    let cel_lit = smoothstep(0.45, 0.52, diffuse_raw);
    let cel_hi = smoothstep(0.76, 0.86, diffuse_raw);
    let effective_sun = cel_lit * mix(0.18, 1.0, shadow_vis);

    let shadow_tint = albedo * vec3<f32>(0.64, 0.67, 0.76);
    let lit_base = albedo * mix(vec3<f32>(0.99, 0.97, 0.95), frame.sun_color.rgb * 1.06, 0.28 + 0.16 * cel_hi);

    var lit_col = mix(shadow_tint, lit_base, smoothstep(0.28, 0.68, effective_sun));

    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.6);
    let rim = fresnel * max(ndotl * 0.5 + 0.5, 0.20) * shadow_vis;
    lit_col += frame.sun_color.rgb * rim * 0.09;

    let fog_dist = max(in.view_depth - 35.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w);
    lit_col = mix(lit_col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.82));

    let dist_outline_fade = 1.0 - smoothstep(140.0, 420.0, in.view_depth) * 0.65;
    outline_w *= dist_outline_fade;

    var out: GBufferOut;
    out.lit_color = vec4<f32>(clamp(lit_col, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(outline_w, 0.0, 1.0));
    let oct_n = oct_encode(normalize(in.view_normal));
    out.normal_id = vec4<f32>(oct_n.x, oct_n.y, clamp(crease_s, 0.0, 1.0), obj_id);
    return out;
}
