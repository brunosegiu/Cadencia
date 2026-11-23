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
    outline_params: vec4<f32>, // x: rel_depth_thresh, y: normal_thresh, z: strength, w: grain
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var lit_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var gbuf_sampler: sampler;
@group(0) @binding(5) var hw_depth_tex: texture_depth_2d;

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

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn hash_grain(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(234.34, 435.345));
    return fract(dot(q, q + 34.23));
}

@fragment
fn fs_post(in: PostOut) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(frame.screen_params.xy) - vec2<i32>(1, 1);
    let p = clamp(vec2<i32>(in.clip_pos.xy), vec2<i32>(0, 0), max_c);
    let p_r = min(p + vec2<i32>(1, 0), max_c);
    let p_b = min(p + vec2<i32>(0, 1), max_c);

    // Keep sampler referenced while reading exact 1:1 texels for razor-thin 1px contours
    let center_col = textureSampleLevel(lit_tex, gbuf_sampler, in.uv, 0.0).rgb;

    let nid_c = textureLoad(normal_tex, p, 0);
    let nid_r = textureLoad(normal_tex, p_r, 0);
    let nid_b = textureLoad(normal_tex, p_b, 0);

    let dm_c = textureLoad(depth_tex, p, 0);
    let dm_r = textureLoad(depth_tex, p_r, 0);
    let dm_b = textureLoad(depth_tex, p_b, 0);

    let n_c = nid_c.xyz * 2.0 - 1.0;
    let n_r = nid_r.xyz * 2.0 - 1.0;
    let n_b = nid_b.xyz * 2.0 - 1.0;

    // Exact 32-bit hardware float depth (eliminates all 8-bit contour rings on close curved limbs!)
    let d_c = load_linear_depth(p, max_c);
    let d_r = load_linear_depth(p_r, max_c);
    let d_b = load_linear_depth(p_b, max_c);

    // 1-pixel forward-difference depth silhouettes (right & down only -> always crisp 1px thin!)
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

    // 1-pixel forward-difference Object-ID boundary (separates bike, limbs, helmet, pack, posts, rocks cleanly)
    let id_r = step(0.002, abs(nid_r.a - nid_c.a));
    let id_b = step(0.002, abs(nid_b.a - nid_c.a));
    let edge_id = max(id_r, id_b);

    // Strict architectural-only corner crease (crease_s is 0.0 on riders, rocks, terrain, foliage!)
    let crease_r = min(dm_c.a, dm_r.a);
    let crease_b = min(dm_c.a, dm_b.a);
    let en_r = smoothstep(frame.outline_params.y * 0.85, frame.outline_params.y * 1.25, length(n_r - n_c)) * crease_r;
    let en_b = smoothstep(frame.outline_params.y * 0.85, frame.outline_params.y * 1.25, length(n_b - n_c)) * crease_b;
    let edge_n = max(en_r, en_b);

    // Pick outline weight from the foreground/edge-bearing pixel so silhouettes against sky stay crisp & even
    let w_r = select(dm_r.b, dm_c.b, d_c < d_r);
    let w_b = select(dm_b.b, dm_c.b, d_c < d_b);
    let edge_w = max(dm_c.b, max(w_r * max(ed_r, id_r), w_b * max(ed_b, id_b)));

    let ink_mask = clamp(max(edge_d, max(edge_id * 0.90, edge_n * 0.70)) * edge_w * frame.outline_params.z, 0.0, 1.0);

    // Use nearest participating pixel depth for smooth aerial-perspective ink fading
    let edge_depth = min(d_c, min(d_r, d_b));
    let near_ink = vec3<f32>(0.13, 0.14, 0.16);
    let far_ink = mix(frame.fog_color.rgb * 0.65, vec3<f32>(0.36, 0.42, 0.46), 0.45);
    let ink_col = mix(near_ink, far_ink, smoothstep(55.0, 280.0, edge_depth));

    var final_col = mix(center_col, ink_col, ink_mask);

    // Whisper-light paper texture
    let grain = (hash_grain(in.uv * frame.screen_params.xy) - 0.5) * frame.outline_params.w;
    final_col = clamp(final_col + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(1.0));

    // Convert perceptual display color to linear only when target format is sRGB
    if (frame.sky_horizon.w > 0.5) {
        return vec4<f32>(srgb_to_linear(final_col), 1.0);
    }
    return vec4<f32>(final_col, 1.0);
}
