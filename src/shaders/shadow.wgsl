// Shadow Cascade Depth Vertex Shader (prepended with common.wgsl)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    var world_pos = obj.model * vec4<f32>(in.pos, 1.0);
    if (obj.obj_info.w > 0.5 && in.uv.x > 0.0001) {
        world_pos = vec4<f32>(apply_vegetation_sway(world_pos.xyz, in.uv, frame.camera_pos.w), 1.0);
    }
    return frame.light_view_proj * world_pos;
}
