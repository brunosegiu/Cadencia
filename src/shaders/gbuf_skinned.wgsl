// Dedicated G-Buffer Domain Shader: Dynamic Cyclists, Bicycles & Compute-Skinned Quadruped Animals
// Minimal register footprint: no procedural noise textures, no branches, unlit albedo output.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

@vertex
fn vs_gbuf_skinned(in: VertexInput) -> GBufferVertOut {
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
    out.vertex_obj_info = vec3<f32>(obj.obj_info.x, obj.obj_info.y, obj.obj_info.z);
    out.mat_id = 4u;
    return out;
}

@fragment
fn fs_gbuf_skinned(in: GBufferVertOut) -> GBufferOut {
    return pack_gbuffer(
        in.color.rgb,
        in.view_normal,
        in.view_depth,
        in.vertex_obj_info.y,
        in.vertex_obj_info.z,
        in.vertex_obj_info.x
    );
}
