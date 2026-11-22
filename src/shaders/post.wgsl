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
    outline_params: vec4<f32>, // x: depth_thresh, y: normal_thresh, z: strength, w: grain
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var lit_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var gbuf_sampler: sampler;

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

fn decode_depth(c: vec4<f32>) -> f32 {
    return (c.r * 255.0 + c.g) / 255.0 * frame.sky_zenith.w;
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
    let px = frame.screen_params.zw;
    let center_col = textureSampleLevel(lit_tex, gbuf_sampler, in.uv, 0.0).rgb;
    let center_nid = textureSampleLevel(normal_tex, gbuf_sampler, in.uv, 0.0);
    let center_dm = textureSampleLevel(depth_tex, gbuf_sampler, in.uv, 0.0);

    let c_norm = center_nid.xyz * 2.0 - 1.0;
    let c_id = center_nid.a;
    let c_depth = decode_depth(center_dm);
    let outline_w = center_dm.b;
    let crease_s = center_dm.a;

    // 3x3 Sobel edge detection across Depth, View-Normal, and Object-ID
    let o_tl = vec2<f32>(-px.x, -px.y);
    let o_tc = vec2<f32>( 0.0,  -px.y);
    let o_tr = vec2<f32>( px.x, -px.y);
    let o_ml = vec2<f32>(-px.x,  0.0);
    let o_mr = vec2<f32>( px.x,  0.0);
    let o_bl = vec2<f32>(-px.x,  px.y);
    let o_bc = vec2<f32>( 0.0,   px.y);
    let o_br = vec2<f32>( px.x,  px.y);

    let dm_tl = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_tl, 0.0);
    let dm_tc = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_tc, 0.0);
    let dm_tr = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_tr, 0.0);
    let dm_ml = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_ml, 0.0);
    let dm_mr = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_mr, 0.0);
    let dm_bl = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_bl, 0.0);
    let dm_bc = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_bc, 0.0);
    let dm_br = textureSampleLevel(depth_tex, gbuf_sampler, in.uv + o_br, 0.0);

    let d_tl = decode_depth(dm_tl);
    let d_tc = decode_depth(dm_tc);
    let d_tr = decode_depth(dm_tr);
    let d_ml = decode_depth(dm_ml);
    let d_mr = decode_depth(dm_mr);
    let d_bl = decode_depth(dm_bl);
    let d_bc = decode_depth(dm_bc);
    let d_br = decode_depth(dm_br);

    let gx_d = (d_tr + 2.0 * d_mr + d_br) - (d_tl + 2.0 * d_ml + d_bl);
    let gy_d = (d_bl + 2.0 * d_bc + d_br) - (d_tl + 2.0 * d_tc + d_tr);
    let edge_d_raw = sqrt(gx_d * gx_d + gy_d * gy_d);

    // Grazing angle protection so road & hillsides never get false depth bands
    let grazing_scale = 1.0 / max(abs(c_norm.z), 0.16);
    let depth_thresh = frame.outline_params.x * max(c_depth * 0.14, 0.6) * grazing_scale;
    let edge_d = smoothstep(depth_thresh * 0.55, depth_thresh * 1.4, edge_d_raw);

    let n_tl = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_tl, 0.0);
    let n_tc = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_tc, 0.0);
    let n_tr = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_tr, 0.0);
    let n_ml = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_ml, 0.0);
    let n_mr = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_mr, 0.0);
    let n_bl = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_bl, 0.0);
    let n_bc = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_bc, 0.0);
    let n_br = textureSampleLevel(normal_tex, gbuf_sampler, in.uv + o_br, 0.0);

    let gx_n = (n_tr.xyz + 2.0 * n_mr.xyz + n_br.xyz) - (n_tl.xyz + 2.0 * n_ml.xyz + n_bl.xyz);
    let gy_n = (n_bl.xyz + 2.0 * n_bc.xyz + n_br.xyz) - (n_tl.xyz + 2.0 * n_tc.xyz + n_tr.xyz);
    let edge_n_raw = length(gx_n) + length(gy_n);
    let edge_n = smoothstep(frame.outline_params.y * 0.65, frame.outline_params.y * 1.35, edge_n_raw) * crease_s;

    // Object-ID discontinuity separates bike frame, limbs, fence posts, rocks, buildings cleanly
    let id_diff = max(
        max(abs(n_tc.a - c_id), abs(n_bc.a - c_id)),
        max(abs(n_ml.a - c_id), abs(n_mr.a - c_id))
    );
    let edge_id = step(0.002, id_diff);

    // Also inherit max neighbor outline_weight on silhouettes against sky
    let max_w = max(outline_w, max(max(dm_tc.b, dm_bc.b), max(dm_ml.b, dm_mr.b)));
    let ink_mask = clamp(max(edge_d, max(edge_n * 0.85, edge_id * 0.88)) * max_w * frame.outline_params.z, 0.0, 1.0);

    // Distance-faded Ligne Claire charcoal-sepia ink -> soft misty slate on far hills
    let near_ink = vec3<f32>(0.11, 0.10, 0.09);
    let far_ink = mix(frame.fog_color.rgb * 0.58, vec3<f32>(0.32, 0.38, 0.42), 0.5);
    let ink_col = mix(near_ink, far_ink, smoothstep(45.0, 260.0, c_depth));

    var final_col = mix(center_col, ink_col, ink_mask);

    // Ultra-subtle warm paper grain
    let grain = (hash_grain(in.uv * frame.screen_params.xy) - 0.5) * frame.outline_params.w;
    final_col = clamp(final_col + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(1.0));

    // Convert perceptual display color to linear only when target format is sRGB
    if (frame.sky_horizon.w > 0.5) {
        return vec4<f32>(srgb_to_linear(final_col), 1.0);
    }
    return vec4<f32>(final_col, 1.0);
}
