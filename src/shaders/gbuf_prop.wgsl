// Dedicated G-Buffer Domain Shader: Static Props, Architecture, Rocks & Swaying Vegetation
// Evaluates non-periodic vertex wind sway and outputs unlit prop albedo + packed per-vertex metadata.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<storage, read> objects: array<ObjectData>;

@vertex
fn vs_gbuf_prop(in: VertexInput) -> GBufferVertOut {
    let obj = objects[in.object_idx];
    let use_vtx_meta = obj.obj_info.w > 0.5;
    var wp = obj.model * vec4<f32>(in.pos, 1.0);
    if (use_vtx_meta && (in.uv.x > 0.0001 || in.uv.y > 0.0001)) {
        wp = vec4<f32>(apply_vegetation_sway(wp.xyz, in.uv, in.color.a, frame.camera_pos.w), 1.0);
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

    var out: GBufferVertOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = wn;
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = vec4<f32>(in.color.rgb * obj.tint.rgb, 1.0);
    out.view_depth = -vp.z;
    out.vertex_obj_info = vec3<f32>(v_obj_id, v_outline_w, v_crease_s);
    out.mat_id = mat_id;
    return out;
}

@fragment
fn fs_gbuf_prop(in: GBufferVertOut) -> GBufferOut {
    let n_tex = value_noise(in.world_pos.xz * 1.4 + in.world_pos.y * 0.9);
    // Subtle organic variation on foliage (mat_id == 2) and rock (mat_id == 0), clean paint on architecture/props
    let mottle_amp = select(0.0, 0.16, in.mat_id == 2u || in.mat_id == 0u);
    let albedo = in.color.rgb * (1.0 + (n_tex - 0.5) * mottle_amp);

    return pack_gbuffer(
        albedo,
        in.view_normal,
        in.view_depth,
        in.vertex_obj_info.y,
        in.vertex_obj_info.z,
        in.vertex_obj_info.x
    );
}
