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
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>, // x = object_id/255, y = outline_weight, z = crease_sens, w = mat_override
    skin_pose_r: vec4<f32>, // (thigh_ang_r, calf_ang_r, foot_ang_r, is_skinned)
    skin_pose_l: vec4<f32>, // (thigh_ang_l, calf_ang_l, foot_ang_l, 0.0)
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_type: u32,
};

struct SceneOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) material_type: u32,
    @location(6) shadow_pos: vec4<f32>,
    @location(7) view_depth: f32,
};

struct GBufferOut {
    @location(0) lit_color: vec4<f32>,
    @location(1) normal_id: vec4<f32>,
    @location(2) depth_meta: vec4<f32>,
};

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

fn rot_x_vec(v: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

@vertex
fn vs_scene(in: VertexInput) -> SceneOut {
    var local_pos = in.pos;
    var local_nrm = in.normal;

    // 4-bone Linear Blend Skinning for unified seamless cyclist character (Pelvis -> Thigh -> Calf -> Foot)
    if (obj.skin_pose_r.w > 0.5) {
        let x = local_pos.x;
        let y_bl = -local_pos.z;
        let z_bl = local_pos.y;
        if (!(z_bl > 0.035 || (z_bl > -0.14 && y_bl > 0.15))) {
            let angles = select(obj.skin_pose_l.xyz, obj.skin_pose_r.xyz, x >= 0.0);
            let side = select(-1.0, 1.0, x >= 0.0);
            let th_ang = angles.x;
            let ca_ang = angles.y;
            let fo_ang = angles.z;

            let w_hip_thigh = (1.0 - smoothstep(-0.095, 0.025, z_bl)) * smoothstep(0.010, 0.055, abs(x));
            let w_pelvis = 1.0 - w_hip_thigh;
            let w_knee_calf = 1.0 - smoothstep(-0.485, -0.375, z_bl);
            let w_ankle_foot = 1.0 - smoothstep(-0.865, -0.815, z_bl);

            let wt_pelvis = w_pelvis;
            let wt_thigh  = w_hip_thigh * (1.0 - w_knee_calf);
            let wt_calf   = w_hip_thigh * w_knee_calf * (1.0 - w_ankle_foot);
            let wt_foot   = w_hip_thigh * w_knee_calf * w_ankle_foot;

            let rest_hip   = vec3<f32>(side * 0.088,  0.000, 0.0);
            let rest_knee  = vec3<f32>(side * 0.088, -0.430, 0.0);
            let rest_ankle = vec3<f32>(side * 0.088, -0.845, 0.0);

            let posed_hip   = rest_hip;
            let posed_knee  = posed_hip  + vec3<f32>(0.0, -0.430 * cos(th_ang), -0.430 * sin(th_ang));
            let posed_ankle = posed_knee + vec3<f32>(0.0, -0.415 * cos(ca_ang), -0.415 * sin(ca_ang));

            let p_pelvis = local_pos;
            let p_thigh  = posed_hip   + rot_x_vec(local_pos - rest_hip,   th_ang);
            let p_calf   = posed_knee  + rot_x_vec(local_pos - rest_knee,  ca_ang);
            let p_foot   = posed_ankle + rot_x_vec(local_pos - rest_ankle, fo_ang);

            let n_pelvis = local_nrm;
            let n_thigh  = rot_x_vec(local_nrm, th_ang);
            let n_calf   = rot_x_vec(local_nrm, ca_ang);
            let n_foot   = rot_x_vec(local_nrm, fo_ang);

            local_pos = wt_pelvis * p_pelvis + wt_thigh * p_thigh + wt_calf * p_calf + wt_foot * p_foot;
            local_nrm = normalize(wt_pelvis * n_pelvis + wt_thigh * n_thigh + wt_calf * n_calf + wt_foot * n_foot);
        }
    }

    let wp = obj.model * vec4<f32>(local_pos, 1.0);
    let wn = normalize((obj.normal_mat * vec4<f32>(local_nrm, 0.0)).xyz);
    let vn = normalize((frame.view_mat * vec4<f32>(wn, 0.0)).xyz);
    let vp = frame.view_mat * wp;

    var out: SceneOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = wn;
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = in.color * obj.tint;
    out.material_type = in.material_type;
    out.shadow_pos = frame.light_view_proj * wp;
    out.view_depth = -vp.z;
    return out;
}

fn sample_shadow_pcf(shadow_pos: vec4<f32>, ndotl: f32) -> f32 {
    let ndc = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z;
    let texel = 1.0 / 2048.0;
    let bias = max(0.0028 * (1.0 - ndotl), 0.00075);
    let cmp_depth = depth - bias;

    var sum = 0.0;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel * 1.15;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, cmp_depth);
        }
    }
    let in_bounds = f32(
        uv.x >= 0.01 && uv.x <= 0.99 &&
        uv.y >= 0.01 && uv.y <= 0.99 &&
        depth >= 0.0 && depth <= 1.0
    );
    return mix(1.0, sum / 9.0, in_bounds);
}

@fragment
fn fs_scene(in: SceneOut) -> GBufferOut {
    let N = normalize(in.world_normal);
    let L = normalize(frame.sun_dir.xyz);
    let V = normalize(frame.camera_pos.xyz - in.world_pos);
    let ndotl = dot(N, L);

    var albedo = in.color.rgb;
    var mat_type = in.material_type;
    var outline_w = obj.obj_info.y;
    var crease_s = obj.obj_info.z;

    // Procedural Ligne Claire surface texture & markings
    if (mat_type == 1u) {
        // Gravel dirt road: warm ochre, organic wheel ruts, dusty edges, scattered irregular pebbles
        let n_big = value_noise(in.world_pos.xz * 0.35);
        let n_med = value_noise(in.world_pos.xz * 1.9);
        let n_fine = value_noise(in.world_pos.xz * 6.5);
        albedo *= 0.92 + 0.14 * n_big + 0.06 * n_med + 0.04 * n_fine;
        // Organic twin wheel-track darkening around u = 0.34 and u = 0.66
        let rut_jit = (n_big - 0.5) * 0.04;
        let rut_l = exp(-pow((in.uv.x - 0.34 + rut_jit) * 9.5, 2.0));
        let rut_r = exp(-pow((in.uv.x - 0.66 - rut_jit) * 9.5, 2.0));
        albedo *= 1.0 - 0.08 * (rut_l + rut_r);
        // Dusty grassy verge darkening at road margins
        let verge = smoothstep(0.41, 0.50, abs(in.uv.x - 0.5));
        albedo = mix(albedo, vec3<f32>(0.58, 0.52, 0.36), verge * 0.60);
        // Organic scattered small stones & faint comic ink specks (non-grid)
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
        // Keep interior road free of false crease ink lines
        outline_w = 0.28;
        crease_s = 0.0;
    } else if (mat_type == 3u) {
        // Coastal asphalt: warm slate-grey with crisp double gold center lines & dashed white lane dividers
        let n_asp = value_noise(in.world_pos.xz * 1.2);
        albedo *= 0.95 + 0.09 * n_asp;
        let u = in.uv.x;
        let d_dist = in.uv.y;
        // Double golden-ochre center lines
        let gold_l = smoothstep(0.011, 0.005, abs(u - 0.487));
        let gold_r = smoothstep(0.011, 0.005, abs(u - 0.513));
        let gold_mask = max(gold_l, gold_r);
        albedo = mix(albedo, vec3<f32>(0.84, 0.70, 0.41), gold_mask);
        // Dashed white lane lines at u = 0.24 and u = 0.76
        let dash = step(0.46, fract(d_dist * 0.09));
        let lane_l = smoothstep(0.011, 0.005, abs(u - 0.24)) * dash;
        let lane_r = smoothstep(0.011, 0.005, abs(u - 0.76)) * dash;
        let edge_line = smoothstep(0.012, 0.006, abs(u - 0.04)) + smoothstep(0.012, 0.006, abs(u - 0.96));
        albedo = mix(albedo, vec3<f32>(0.87, 0.84, 0.78), clamp(max(max(lane_l, lane_r), edge_line), 0.0, 1.0));
        outline_w = 0.25;
        crease_s = 0.0;
    } else if (mat_type == 6u) {
        // Coastal boardwalk sandstone paving tile grid
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
        // Terrain & stratified rocks: painterly mottled grass + horizontal sandstone cliff strata
        let n_mottle = value_noise(in.world_pos.xz * 0.12) * 0.65 + value_noise(in.world_pos.xz * 0.55) * 0.35;
        albedo *= 0.89 + 0.22 * n_mottle;
        let steep = smoothstep(0.76, 0.44, N.y);
        let strata = sin(in.world_pos.y * 2.2 + value_noise(in.world_pos.xz * 0.25) * 3.5);
        let strata_col = vec3<f32>(0.64, 0.57, 0.46) * (0.88 + 0.16 * strata + 0.12 * n_mottle);
        albedo = mix(albedo, strata_col, steep * 0.80);
    } else if (mat_type == 2u) {
        // Foliage subtle warm dappled leaf variation
        let n_leaf = value_noise(in.world_pos.xz * 1.8 + in.world_pos.y * 1.1);
        albedo *= 0.90 + 0.18 * n_leaf;
    }

    // Clean crisp 2-tone stylized cel lighting (bright warm sunlit plane + airy cool slate shadow)
    let shadow_vis = sample_shadow_pcf(in.shadow_pos, max(ndotl, 0.0));
    let diffuse_raw = ndotl * 0.5 + 0.5;
    // Crisp cel terminator with subtle soft anti-aliased transition
    let cel_lit = smoothstep(0.45, 0.52, diffuse_raw);
    let cel_hi = smoothstep(0.76, 0.86, diffuse_raw);
    let effective_sun = cel_lit * mix(0.18, 1.0, shadow_vis);

    let shadow_tint = albedo * vec3<f32>(0.64, 0.67, 0.76);
    let lit_base = albedo * mix(vec3<f32>(0.99, 0.97, 0.95), frame.sun_color.rgb * 1.06, 0.28 + 0.16 * cel_hi);

    var lit_col = mix(shadow_tint, lit_base, smoothstep(0.28, 0.68, effective_sun));

    // Subtle clean sunlit rim accent on silhouettes
    let fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.6);
    let rim = fresnel * max(ndotl * 0.5 + 0.5, 0.20) * shadow_vis;
    lit_col += frame.sun_color.rgb * rim * 0.09;

    // Aerial perspective atmospheric fog
    let fog_dist = max(in.view_depth - 35.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w);
    lit_col = mix(lit_col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.82));

    // Fade outline weight subtly on very distant hills so horizon stays painterly
    let dist_outline_fade = 1.0 - smoothstep(140.0, 420.0, in.view_depth) * 0.65;
    outline_w *= dist_outline_fade;

    // Encode 16-bit linear depth into R & G (0..far_plane)
    let norm_d = clamp(in.view_depth / frame.sky_zenith.w, 0.0, 0.9999);
    let d_scaled = norm_d * 255.0;
    let d_hi = floor(d_scaled) / 255.0;
    let d_lo = fract(d_scaled);

    var out: GBufferOut;
    out.lit_color = vec4<f32>(clamp(lit_col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    out.normal_id = vec4<f32>(normalize(in.view_normal) * 0.5 + 0.5, obj.obj_info.x);
    out.depth_meta = vec4<f32>(d_hi, d_lo, clamp(outline_w, 0.0, 1.0), clamp(crease_s, 0.0, 1.0));
    return out;
}
