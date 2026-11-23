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
    outline_params: vec4<f32>,
    light_view_proj_c1: mat4x4<f32>,
    light_view_proj_c2: mat4x4<f32>,
    cascade_splits: vec4<f32>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    tint: vec4<f32>,
    obj_info: vec4<f32>,
    skin_pose_r: vec4<f32>,
    skin_pose_l: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal_mat: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

// Skinning is pre-evaluated in skinning.wgsl compute shader prior to rendering.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    let world_pos = obj.model * vec4<f32>(in.pos, 1.0);
    return frame.light_view_proj * world_pos;
}
