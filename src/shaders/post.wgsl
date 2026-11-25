// Fullscreen Deferred Lighting & Ligne Claire Ink Outline Pass (prepended with common.wgsl)
// Evaluates 3-cascade CSM shadows (via a single unified PCF function), stepped cel-shading,
// Fresnel rim lighting, water specular shimmer, exponential distance fog, and 1px ink outlines.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var albedo_tex: texture_2d<f32>;     // rgb = unlit G-Buffer albedo, a = outline_weight
@group(0) @binding(2) var normal_tex: texture_2d<f32>;     // rg = octahedral view normal, b = crease_sensitivity, a = object_id
@group(0) @binding(3) var gbuf_sampler: sampler;
@group(0) @binding(4) var hw_depth_tex: texture_depth_2d;  // Single hardware Depth32Float buffer
@group(0) @binding(5) var shadow_map: texture_depth_2d_array;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;

struct PostOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_post(@builtin(vertex_index) idx: u32) -> PostOut {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[idx];
    var out: PostOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn load_linear_depth(coord: vec2<i32>, max_c: vec2<i32>) -> f32 {
    let c = clamp(coord, vec2<i32>(0, 0), max_c);
    let d_ndc = textureLoad(hw_depth_tex, c, 0);
    let near = frame.sun_color.w;
    let far = frame.sky_zenith.w;
    return (near * far) / max(far - d_ndc * (far - near), 0.0001);
}

fn view_to_world_normal(vn: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(
        dot(frame.view_mat[0].xyz, vn),
        dot(frame.view_mat[1].xyz, vn),
        dot(frame.view_mat[2].xyz, vn)
    ));
}

// Unified hardware-bilinear 4-tap PCF shadow cascade sampler (each hardware comparison tap filters 2x2 texels)
fn sample_shadow_cascade(shadow_pos: vec4<f32>, layer: i32, bias: f32, filter_radius: f32) -> f32 {
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
            let s0 = sample_shadow_cascade(sp0, 0, bias0, 1.15);
            let blend1 = max(
                smoothstep(0.80, 0.94, box0),
                smoothstep(frame.cascade_splits.x * 0.82, frame.cascade_splits.x, view_depth)
            );
            if (blend1 > 0.01) {
                let wp1 = vec4<f32>(world_pos + world_normal * (0.012 * slope), 1.0);
                let sp1 = frame.light_view_proj_c1 * wp1;
                let bias1 = max(0.00030 * slope, 0.00010);
                let s1 = sample_shadow_cascade(sp1, 1, bias1, 1.0);
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
            let s1 = sample_shadow_cascade(sp1, 1, bias1, 1.0);
            let blend2 = max(
                smoothstep(0.82, 0.95, box1),
                smoothstep(frame.cascade_splits.y * 0.84, frame.cascade_splits.y, view_depth)
            );
            if (blend2 > 0.01) {
                let wp2 = vec4<f32>(world_pos + world_normal * (0.032 * slope), 1.0);
                let sp2 = frame.light_view_proj_c2 * wp2;
                let bias2 = max(0.00042 * slope, 0.00015);
                let s2 = sample_shadow_cascade(sp2, 2, bias2, 0.85);
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
        let s2 = sample_shadow_cascade(sp2, 2, bias2, 0.85);
        let fade_far = max(
            smoothstep(0.84, 0.97, box2),
            smoothstep(frame.cascade_splits.z * 0.80, frame.cascade_splits.z, view_depth)
        );
        return mix(s2, 1.0, fade_far);
    }
    return 1.0;
}

fn evaluate_water_lighting(base_albedo: vec3<f32>, world_pos: vec3<f32>, view_depth: f32) -> vec3<f32> {
    let V = normalize(frame.camera_pos.xyz - world_pos);
    let L = normalize(frame.sun_dir.xyz);
    let t = frame.camera_pos.w;

    var col = base_albedo;
    if (frame.sun_dir.w > 0.5) {
        let view_dir_xz = normalize(vec2<f32>(-V.x, -V.z));
        let sun_dir_xz = normalize(vec2<f32>(L.x, L.z));
        let align = max(dot(view_dir_xz, sun_dir_xz), 0.0);

        let ripple1 = sin(world_pos.z * 0.85 + world_pos.x * 0.25 + t * 1.1);
        let ripple2 = cos(world_pos.x * 0.55 - world_pos.z * 0.4 + t * 0.7);
        let wave = ripple1 * 0.5 + ripple2 * 0.5;

        let far_t = smoothstep(50.0, 620.0, view_depth);
        col = mix(vec3<f32>(0.37, 0.50, 0.60), vec3<f32>(0.76, 0.68, 0.58), far_t * 0.65);

        let sun_path = pow(align, 11.0);
        let glitter = smoothstep(0.10, 0.75, sun_path + wave * 0.32 * sun_path);
        col = mix(col, vec3<f32>(1.00, 0.93, 0.70), clamp(glitter * 0.96, 0.0, 1.0));

        let streak = smoothstep(0.88, 0.98, sin(world_pos.z * 1.4 + sin(world_pos.x * 0.2) * 2.0 + t * 0.5));
        col = mix(col, vec3<f32>(0.92, 0.86, 0.74), streak * 0.28 * (1.0 - far_t * 0.55));
    } else {
        let ripple = sin(world_pos.z * 0.7 + world_pos.x * 0.4 + t * 0.8) * 0.5 + 0.5;
        col = mix(vec3<f32>(0.84, 0.88, 0.86), vec3<f32>(0.97, 0.96, 0.92), ripple * 0.65);
    }

    let fog_dist = max(view_depth - 45.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w * 0.75);
    return mix(col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.72));
}

fn evaluate_deferred_lighting(
    albedo: vec3<f32>,
    view_normal: vec3<f32>,
    world_pos: vec3<f32>,
    view_depth: f32,
) -> vec3<f32> {
    let N = view_to_world_normal(view_normal);
    let L = normalize(frame.sun_dir.xyz);
    let V = normalize(frame.camera_pos.xyz - world_pos);
    let ndotl = dot(N, L);

    let shadow_vis = sample_csm_shadow(world_pos, N, view_depth, max(ndotl, 0.0));
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

    let fog_dist = max(view_depth - 35.0, 0.0);
    let fog_factor = 1.0 - exp(-fog_dist * frame.fog_color.w);
    return mix(lit_col, frame.fog_color.rgb, clamp(fog_factor, 0.0, 0.82));
}

@fragment
fn fs_post(in: PostOut) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(frame.screen_params.xy) - vec2<i32>(1, 1);
    let p = clamp(vec2<i32>(in.clip_pos.xy), vec2<i32>(0, 0), max_c);
    let p_r = min(p + vec2<i32>(1, 0), max_c);
    let p_b = min(p + vec2<i32>(0, 1), max_c);

    let alb_c = textureSampleLevel(albedo_tex, gbuf_sampler, in.uv, 0.0);
    let alb_r = textureLoad(albedo_tex, p_r, 0);
    let alb_b = textureLoad(albedo_tex, p_b, 0);

    let nid_c = textureLoad(normal_tex, p, 0);
    let nid_r = textureLoad(normal_tex, p_r, 0);
    let nid_b = textureLoad(normal_tex, p_b, 0);

    let n_c = oct_decode(nid_c.xy);
    let n_r = oct_decode(nid_r.xy);
    let n_b = oct_decode(nid_b.xy);

    let d_ndc = textureLoad(hw_depth_tex, p, 0);
    let d_c = load_linear_depth(p, max_c);
    let d_r = load_linear_depth(p_r, max_c);
    let d_b = load_linear_depth(p_b, max_c);

    // 1. Deferred Lighting Evaluation (once per visible pixel; sky at obj_id == 0 is already self-lit)
    let obj_id_raw = round(nid_c.a * 255.0);
    var center_col = alb_c.rgb;
    if (d_ndc < 0.99999 && obj_id_raw >= 0.5) {
        let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0, d_ndc, 1.0);
        let world_h = frame.inv_view_proj * ndc;
        let world_pos = world_h.xyz / max(abs(world_h.w), 1e-6);

        if (abs(obj_id_raw - 4.0) < 0.5) {
            center_col = evaluate_water_lighting(alb_c.rgb, world_pos, d_c);
        } else {
            center_col = evaluate_deferred_lighting(alb_c.rgb, n_c, world_pos, d_c);
        }
    }

    // 2. 1-Pixel Forward-Difference Ligne Claire Ink Outlines
    let min_d_r = min(d_c, d_r);
    let n_close_r = select(n_r, n_c, d_c < d_r);
    let graz_r = 1.0 / max(abs(n_close_r.z), 0.40);
    let thresh_r = max(min_d_r * frame.outline_params.x, 0.065) * graz_r;
    let ed_r = smoothstep(thresh_r * 0.80, thresh_r * 1.45, abs(d_r - d_c));

    let min_d_b = min(d_c, d_b);
    let n_close_b = select(n_b, n_c, d_c < d_b);
    let graz_b = 1.0 / max(abs(n_close_b.z), 0.40);
    let thresh_b = max(min_d_b * frame.outline_params.x, 0.065) * graz_b;
    let ed_b = smoothstep(thresh_b * 0.80, thresh_b * 1.45, abs(d_b - d_c));

    let edge_d = max(ed_r, ed_b);

    let id_r = step(0.002, abs(nid_r.a - nid_c.a));
    let id_b = step(0.002, abs(nid_b.a - nid_c.a));
    let edge_id = max(id_r, id_b);

    let crease_r = min(nid_c.b, nid_r.b);
    let crease_b = min(nid_c.b, nid_b.b);
    let en_r = smoothstep(frame.outline_params.y * 0.85, frame.outline_params.y * 1.25, length(n_r - n_c)) * crease_r;
    let en_b = smoothstep(frame.outline_params.y * 0.85, frame.outline_params.y * 1.25, length(n_b - n_c)) * crease_b;
    let edge_n = max(en_r, en_b);

    let w_r = select(alb_r.a, alb_c.a, d_c < d_r);
    let w_b = select(alb_b.a, alb_c.a, d_c < d_b);
    let edge_w = max(alb_c.a, max(w_r * max(ed_r, id_r), w_b * max(ed_b, id_b)));

    let ink_mask = clamp(max(edge_d, max(edge_id * 0.90, edge_n * 0.70)) * edge_w * frame.outline_params.z, 0.0, 1.0);

    let edge_depth = min(d_c, min(d_r, d_b));
    let near_ink = vec3<f32>(0.13, 0.14, 0.16);
    let far_ink = mix(frame.fog_color.rgb * 0.65, vec3<f32>(0.36, 0.42, 0.46), 0.45);
    let ink_col = mix(near_ink, far_ink, smoothstep(55.0, 280.0, edge_depth));

    let final_col = mix(center_col, ink_col, ink_mask);
    return vec4<f32>(clamp(final_col, vec3<f32>(0.0), vec3<f32>(1.0)), ink_mask);
}
