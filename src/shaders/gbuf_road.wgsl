// Dedicated G-Buffer Domain Shader: Road Surfaces (Tarmac & Compact Rural Gravel)
// Zero material if/else chains; outputs unlit albedo + normal/ID to the G-Buffer.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

@vertex
fn vs_gbuf_road(in: VertexInput) -> GBufferVertOut {
    let wp = obj.model * vec4<f32>(in.pos, 1.0);
    let wn = normalize((obj.normal_mat * vec4<f32>(normalize(in.normal_mat.xyz), 0.0)).xyz);
    let vn = normalize((frame.view_mat * vec4<f32>(wn, 0.0)).xyz);
    let vp = frame.view_mat * wp;

    var out: GBufferVertOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = wn;
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = vec4<f32>(in.color.rgb * obj.tint.rgb, 1.0);
    out.view_depth = -vp.z;
    out.vertex_obj_info = vec3<f32>(obj.obj_info.x, 0.26, 0.0);
    out.mat_id = 0u;
    return out;
}

@fragment
fn fs_gbuf_road_tarmac(in: GBufferVertOut) -> GBufferOut {
    var albedo = in.color.rgb;
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

    return pack_gbuffer(albedo, in.view_normal, in.view_depth, 0.25, 0.0, in.vertex_obj_info.x);
}

@fragment
fn fs_gbuf_road_gravel(in: GBufferVertOut) -> GBufferOut {
    var albedo = in.color.rgb;
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
    let off = vec2<f32>(hash21(p_cell + 2.3) - 0.5, hash21(p_cell + 9.7) - 0.5) * 0.68;
    let rad = 0.035 + hash21(p_cell + 4.1) * 0.045;
    let d_p = length(p_f - off);
    let dot_mask = step(0.72, p_seed) * smoothstep(rad, rad * 0.45, d_p) * smoothstep(55.0, 6.0, in.view_depth);
    albedo = mix(albedo, vec3<f32>(0.36, 0.29, 0.22), dot_mask * 0.58);

    return pack_gbuffer(albedo, in.view_normal, in.view_depth, 0.28, 0.0, in.vertex_obj_info.x);
}
