// Dedicated G-Buffer Domain Shader: Opaque Water Surfaces (Canyon River & Coastal Sunset Ocean)
// Writes opaque water base albedo, upward world normal, and object_id = 4 into the G-Buffer;
// sun reflection shimmer and atmospheric fog are evaluated in the deferred lighting pass.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<storage, read> objects: array<ObjectData>;

@vertex
fn vs_gbuf_water(in: VertexInput) -> GBufferVertOut {
    let obj = objects[in.object_idx];
    let wp = obj.model * vec4<f32>(in.pos, 1.0);
    let vn = normalize((frame.view_mat * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    let vp = frame.view_mat * wp;

    var out: GBufferVertOut;
    out.clip_pos = frame.view_proj * wp;
    out.world_pos = wp.xyz;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.view_normal = vn;
    out.uv = in.uv;
    out.color = in.color;
    out.view_depth = -vp.z;
    out.vertex_obj_info = vec3<f32>(obj.obj_info.x, 0.0, 0.0);
    out.mat_id = 7u;
    return out;
}

@fragment
fn fs_gbuf_water(in: GBufferVertOut) -> GBufferOut {
    return pack_gbuffer(
        in.color.rgb,
        in.view_normal,
        in.view_depth,
        0.0,
        0.0,
        in.vertex_obj_info.x
    );
}
