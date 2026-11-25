// Dedicated G-Buffer Domain Shader: Terrain & Coastal Promenade Surfaces
// Outputs unlit albedo + normal/ID to the G-Buffer.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

@vertex
fn vs_gbuf_terrain(in: VertexInput) -> GBufferVertOut {
    let wp = obj.model * vec4<f32>(in.pos, 1.0);
    let wn = normalize((obj.normal_mat * vec4<f32>(normalize(in.normal_mat.xyz), 0.0)).xyz);
    let vn = normalize((frame.view_mat * vec4<f32>(wn, 0.0)).xyz);
    let vp = frame.view_mat * wp;
    let raw_mat = u32(round(max(in.normal_mat.w * 127.0, 0.0))) & 7u;

    var out: GBufferVertOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = wn;
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = vec4<f32>(in.color.rgb * obj.tint.rgb, 1.0);
    out.view_depth = -vp.z;
    out.vertex_obj_info = vec3<f32>(obj.obj_info.x, obj.obj_info.y, obj.obj_info.z);
    out.mat_id = raw_mat;
    return out;
}

@fragment
fn fs_gbuf_terrain_natural(in: GBufferVertOut) -> GBufferOut {
    let N = normalize(in.world_normal);
    var albedo = in.color.rgb;
    let n_mottle = value_noise(in.world_pos.xz * 0.12) * 0.65 + value_noise(in.world_pos.xz * 0.55) * 0.35;
    albedo *= 0.89 + 0.22 * n_mottle;
    let steep = smoothstep(0.76, 0.44, N.y);
    let strata = sin(in.world_pos.y * 2.2 + value_noise(in.world_pos.xz * 0.25) * 3.5);
    let strata_col = vec3<f32>(0.64, 0.57, 0.46) * (0.88 + 0.16 * strata + 0.12 * n_mottle);
    albedo = mix(albedo, strata_col, steep * 0.80);

    return pack_gbuffer(albedo, in.view_normal, in.view_depth, in.vertex_obj_info.y, 0.0, in.vertex_obj_info.x);
}

@fragment
fn fs_gbuf_terrain_coastal(in: GBufferVertOut) -> GBufferOut {
    let tile_uv = vec2<f32>(in.uv.x * 0.82, in.uv.y * 0.55);
    let cell = floor(tile_uv);
    let f_uv = fract(tile_uv);
    let tone = 0.94 + 0.11 * hash21(cell);
    var tile_albedo = in.color.rgb * tone;
    let seam_x = smoothstep(0.055, 0.02, min(f_uv.x, 1.0 - f_uv.x));
    let seam_y = smoothstep(0.045, 0.016, min(f_uv.y, 1.0 - f_uv.y));
    let seam = max(seam_x, seam_y) * smoothstep(130.0, 16.0, in.view_depth);
    tile_albedo = mix(tile_albedo, vec3<f32>(0.42, 0.35, 0.29), seam * 0.55);

    let n_mottle = value_noise(in.world_pos.xz * 0.12) * 0.65 + value_noise(in.world_pos.xz * 0.55) * 0.35;
    let natural_albedo = in.color.rgb * (0.89 + 0.22 * n_mottle);

    let is_tile = in.mat_id == 6u;
    let albedo = select(natural_albedo, tile_albedo, is_tile);
    let outline_w = select(in.vertex_obj_info.y, 0.35, is_tile);
    let crease_s = select(0.0, 0.15, is_tile);

    return pack_gbuffer(albedo, in.view_normal, in.view_depth, outline_w, crease_s, in.vertex_obj_info.x);
}
