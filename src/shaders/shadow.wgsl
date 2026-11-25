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

fn apply_vegetation_sway(world_pos: vec3<f32>, sway_uv: vec2<f32>, time_s: f32) -> vec3<f32> {
    let flex = sway_uv.x;
    if (flex <= 0.0001) {
        return world_pos;
    }
    let seed = sway_uv.y;
    let h1 = fract(sin(seed * 12.9898) * 43758.5453);
    let h2 = fract(sin(seed * 78.2330) * 28461.7193);
    let h3 = fract(sin(seed * 41.6351) * 61539.1847);

    let freq1 = 1.15 + h1 * 0.95;
    let freq2 = freq1 * 1.618034 + h2 * 0.55;
    let freq3 = 0.41 + h3 * 0.49;

    let ph1 = h1 * 6.2831853;
    let ph2 = h2 * 6.2831853;
    let ph3 = h3 * 6.2831853;

    let wind_wave = dot(world_pos.xz, vec2<f32>(0.085, 0.065));
    let branch_phase = dot(world_pos, vec3<f32>(1.35, 1.85, 1.45));
    let gust = 0.72 + 0.28 * sin(time_s * freq3 + ph3 + wind_wave * 0.45);

    let sway_a = (0.64 * sin(time_s * freq1 + ph1 + wind_wave)
                + 0.26 * sin(time_s * freq2 + ph2)
                + 0.10 * sin(time_s * (freq2 * 1.45) + branch_phase)) * gust;
    let sway_b = (0.58 * cos(time_s * (freq1 * 0.89) + ph2 + wind_wave)
                + 0.42 * sin(time_s * (freq2 * 1.13) + ph3 + branch_phase * 0.5)) * gust;

    let dir_ang = 0.55 + (h3 - 0.5) * 1.15;
    let dir_x = cos(dir_ang);
    let dir_z = sin(dir_ang);

    let dx = (dir_x * sway_a - dir_z * sway_b * 0.45) * flex;
    let dz = (dir_z * sway_a + dir_x * sway_b * 0.45) * flex;
    let dy = -0.16 * (abs(dx) + abs(dz));
    return world_pos + vec3<f32>(dx, dy, dz);
}

// Skinning is pre-evaluated in skinning.wgsl compute shader prior to rendering.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    var world_pos = obj.model * vec4<f32>(in.pos, 1.0);
    if (obj.obj_info.w > 0.5 && in.uv.x > 0.0001) {
        world_pos = vec4<f32>(apply_vegetation_sway(world_pos.xyz, in.uv, frame.camera_pos.w), 1.0);
    }
    return frame.light_view_proj * world_pos;
}
