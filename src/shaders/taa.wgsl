struct TaaUniforms {
    inv_view_proj: mat4x4<f32>,  // Unjittered current frame inverse view-projection
    prev_view_proj: mat4x4<f32>, // Unjittered previous frame view-projection
    screen_params: vec4<f32>,    // x = width, y = height, z = 1/width, w = 1/height
    jitter_params: vec4<f32>,    // xy = curr_jitter_uv, zw = prev_jitter_uv
    taa_params: vec4<f32>,       // x = taa_enabled, y = reset_history, z = output_is_srgb, w = grain_strength
    camera_params: vec4<f32>,    // x = near, y = far, z = base_alpha, w = sharpen_strength
};

@group(0) @binding(0) var<uniform> taa: TaaUniforms;
@group(0) @binding(1) var curr_color_tex: texture_2d<f32>; // Post-outline sRGB color (a = ink_mask)
@group(0) @binding(2) var history_tex: texture_2d<f32>;    // Previous frame TAA history (sRGB)
@group(0) @binding(3) var hw_depth_tex: texture_depth_2d;  // Hardware Depth32Float
@group(0) @binding(4) var normal_tex: texture_2d<f32>;     // G-Buffer normal + object_id in .a
@group(0) @binding(5) var linear_sampler: sampler;

struct TaaOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct TaaFragOut {
    @location(0) target_color: vec4<f32>,
    @location(1) history_color: vec4<f32>,
};

@vertex
fn vs_taa(@builtin(vertex_index) idx: u32) -> TaaOut {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[idx];
    var out: TaaOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let co = rgb.r - rgb.b;
    let t = rgb.b + co * 0.5;
    let cg = rgb.g - t;
    let y = t + cg * 0.5;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let t = ycocg.x - ycocg.z * 0.5;
    let g = ycocg.z + t;
    let b = t - ycocg.y * 0.5;
    let r = b + ycocg.y;
    return vec3<f32>(r, g, b);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn hash_grain(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(234.34, 435.345));
    return fract(dot(q, q + 34.23));
}

// Clip history color in YCoCg space toward the current neighborhood center
fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, center: vec3<f32>, history: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + vec3<f32>(1e-5);
    let v_clip = history - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    if (ma_unit > 1.0) {
        return p_clip + v_clip / ma_unit;
    }
    return history;
}

// 5-tap optimized Filmic Catmull-Rom bicubic history filter to prevent history blur during motion
fn sample_history_catmull_rom(uv: vec2<f32>) -> vec3<f32> {
    let tex_size = taa.screen_params.xy;
    let inv_size = taa.screen_params.zw;
    let position = uv * tex_size;
    let center_pos = floor(position - 0.5) + 0.5;
    let f = position - center_pos;
    let f2 = f * f;
    let f3 = f2 * f;

    let w0 = -0.5 * f3 + f2 - 0.5 * f;
    let w1 =  1.5 * f3 - 2.5 * f2 + 1.0;
    let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    let w3 =  0.5 * f3 - 0.5 * f2;

    let w12 = w1 + w2;
    let tc12 = (center_pos + w2 / max(w12, vec2<f32>(1e-5))) * inv_size;
    let tc0 = (center_pos - 1.0) * inv_size;
    let tc3 = (center_pos + 2.0) * inv_size;

    let s0 = textureSampleLevel(history_tex, linear_sampler, vec2<f32>(tc12.x, tc0.y), 0.0).rgb;
    let s1 = textureSampleLevel(history_tex, linear_sampler, vec2<f32>(tc0.x,  tc12.y), 0.0).rgb;
    let s2 = textureSampleLevel(history_tex, linear_sampler, vec2<f32>(tc12.x, tc12.y), 0.0).rgb;
    let s3 = textureSampleLevel(history_tex, linear_sampler, vec2<f32>(tc3.x,  tc12.y), 0.0).rgb;
    let s4 = textureSampleLevel(history_tex, linear_sampler, vec2<f32>(tc12.x, tc3.y), 0.0).rgb;

    let cw0 = w12.x * w0.y;
    let cw1 = w0.x  * w12.y;
    let cw2 = w12.x * w12.y;
    let cw3 = w3.x  * w12.y;
    let cw4 = w12.x * w3.y;

    let total_w = cw0 + cw1 + cw2 + cw3 + cw4;
    let res = (s0 * cw0 + s1 * cw1 + s2 * cw2 + s3 * cw3 + s4 * cw4) / max(total_w, 1e-5);
    return clamp(res, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_taa(in: TaaOut) -> TaaFragOut {
    let max_c = vec2<i32>(taa.screen_params.xy) - vec2<i32>(1, 1);
    let p = clamp(vec2<i32>(in.clip_pos.xy), vec2<i32>(0, 0), max_c);

    let c11 = textureLoad(curr_color_tex, p, 0);
    var resolved_rgb = c11.rgb;

    if (taa.taa_params.x > 0.5) {
        // 1. 3x3 Neighborhood & Closest-Depth Velocity Dilation
        var min_d_ndc = 1.0;
        var closest_coord = p;
        var m1 = vec3<f32>(0.0);
        var m2 = vec3<f32>(0.0);
        var n_min = vec3<f32>( 1e5);
        var n_max = vec3<f32>(-1e5);

        var c_up = c11.rgb;
        var c_dn = c11.rgb;
        var c_lt = c11.rgb;
        var c_rt = c11.rgb;

        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let sc = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0, 0), max_c);
                let s_rgb = textureLoad(curr_color_tex, sc, 0).rgb;
                if (dx == 0 && dy == -1) { c_up = s_rgb; }
                if (dx == 0 && dy ==  1) { c_dn = s_rgb; }
                if (dx == -1 && dy == 0) { c_lt = s_rgb; }
                if (dx ==  1 && dy == 0) { c_rt = s_rgb; }

                let ycc = rgb_to_ycocg(s_rgb);
                m1 += ycc;
                m2 += ycc * ycc;
                n_min = min(n_min, ycc);
                n_max = max(n_max, ycc);

                let d_sample = textureLoad(hw_depth_tex, sc, 0);
                if (d_sample < min_d_ndc) {
                    min_d_ndc = d_sample;
                    closest_coord = sc;
                }
            }
        }

        // 2. Reconstruct 3D world position at closest depth & project to previous unjittered frame
        // Remove this frame's sub-pixel jitter before reconstructing with the unjittered inverse VP
        let c_uv = (vec2<f32>(closest_coord) + vec2<f32>(0.5, 0.5)) * taa.screen_params.zw - taa.jitter_params.xy;
        let c_ndc = vec4<f32>(c_uv.x * 2.0 - 1.0, (1.0 - c_uv.y) * 2.0 - 1.0, min_d_ndc, 1.0);
        let world_h = taa.inv_view_proj * c_ndc;
        let world_pos = world_h.xyz / max(abs(world_h.w), 1e-6);
        let prev_clip = taa.prev_view_proj * vec4<f32>(world_pos, 1.0);
        let prev_ndc = prev_clip.xy / max(abs(prev_clip.w), 1e-6);
        let c_prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

        let velocity_uv = c_prev_uv - c_uv;
        let prev_uv = in.uv + velocity_uv;

        let in_bounds = prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
                        prev_uv.y >= 0.0 && prev_uv.y <= 1.0 &&
                        prev_clip.w > 0.0;
        let valid_history = (taa.taa_params.y < 0.5) && in_bounds;

        if (valid_history) {
            let hist_sample_rgb = sample_history_catmull_rom(prev_uv);
            let hist_ycc = rgb_to_ycocg(hist_sample_rgb);
            let curr_ycc = rgb_to_ycocg(c11.rgb);

            // 3. YCoCg Variance Clipping Box
            let mu = m1 * (1.0 / 9.0);
            let sigma = sqrt(max(m2 * (1.0 / 9.0) - mu * mu, vec3<f32>(0.0)));

            let obj_id_raw = textureLoad(normal_tex, p, 0).a * 255.0;
            let is_dynamic_rider = obj_id_raw >= 179.5;
            let vel_px = length(velocity_uv * taa.screen_params.xy);

            var gamma = mix(1.30, 0.85, clamp(vel_px * 0.15, 0.0, 1.0));
            if (is_dynamic_rider) {
                gamma = min(gamma, 0.88);
            }

            let box_min = max(mu - gamma * sigma, n_min);
            let box_max = min(mu + gamma * sigma, n_max);
            let clipped_hist_ycc = clip_aabb(box_min, box_max, curr_ycc, hist_ycc);
            let clipped_hist_rgb = clamp(ycocg_to_rgb(clipped_hist_ycc), vec3<f32>(0.0), vec3<f32>(1.0));

            // 4. Luminance-weighted temporal accumulation
            var alpha = taa.camera_params.z + clamp(vel_px * 0.015, 0.0, 0.14);
            if (is_dynamic_rider) {
                alpha = max(alpha, 0.17);
            }
            let w_curr = alpha / (1.0 + max(curr_ycc.x, 0.0));
            let w_hist = (1.0 - alpha) / (1.0 + max(clipped_hist_ycc.x, 0.0));
            let accum_rgb = (clipped_hist_rgb * w_hist + c11.rgb * w_curr) / max(w_hist + w_curr, 1e-5);

            // 5. Contrast-Adaptive Unsharp Filter to keep 1px Ligne Claire ink outlines razor-sharp
            let card_avg = (c_up + c_dn + c_lt + c_rt) * 0.25;
            let high_freq = c11.rgb - card_avg;
            let sharpen_str = taa.camera_params.w * mix(1.0, 1.25, c11.a);
            resolved_rgb = clamp(accum_rgb + high_freq * sharpen_str, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }

    // Apply subtle paper grain only to the final display output, not to the persistent TAA history
    let grain = (hash_grain(in.uv * taa.screen_params.xy) - 0.5) * taa.taa_params.w;
    var display_rgb = clamp(resolved_rgb + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(1.0));
    if (taa.taa_params.z > 0.5) {
        display_rgb = srgb_to_linear(display_rgb);
    }

    var out: TaaFragOut;
    out.target_color = vec4<f32>(display_rgb, 1.0);
    out.history_color = vec4<f32>(resolved_rgb, 1.0);
    return out;
}
