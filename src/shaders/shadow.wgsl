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
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material_type: u32,
};

fn rot_x_vec(v: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    var local_pos = in.pos;
    if (obj.skin_pose_r.w > 0.5) {
        let x = local_pos.x;
        let y_bl = -local_pos.z;
        let z_bl = local_pos.y;
        if (!(z_bl > 0.035 || (z_bl > -0.14 && y_bl > 0.15))) {
            let angles = select(obj.skin_pose_l.xyz, obj.skin_pose_r.xyz, x >= 0.0);
            let side = select(-1.0, 1.0, x >= 0.0);
            let th_ang = angles.x;
            let ca_ang = angles.y;
            let fo_ang = angles.z;

            let w_hip_thigh = (1.0 - smoothstep(-0.095, 0.025, z_bl)) * smoothstep(0.010, 0.055, abs(x));
            let w_pelvis = 1.0 - w_hip_thigh;
            let w_knee_calf = 1.0 - smoothstep(-0.485, -0.375, z_bl);
            let w_ankle_foot = 1.0 - smoothstep(-0.865, -0.815, z_bl);

            let wt_pelvis = w_pelvis;
            let wt_thigh  = w_hip_thigh * (1.0 - w_knee_calf);
            let wt_calf   = w_hip_thigh * w_knee_calf * (1.0 - w_ankle_foot);
            let wt_foot   = w_hip_thigh * w_knee_calf * w_ankle_foot;

            let rest_hip   = vec3<f32>(side * 0.088,  0.000, 0.0);
            let rest_knee  = vec3<f32>(side * 0.088, -0.430, 0.0);
            let rest_ankle = vec3<f32>(side * 0.088, -0.845, 0.0);

            let posed_hip   = rest_hip;
            let posed_knee  = posed_hip  + vec3<f32>(0.0, -0.430 * cos(th_ang), -0.430 * sin(th_ang));
            let posed_ankle = posed_knee + vec3<f32>(0.0, -0.415 * cos(ca_ang), -0.415 * sin(ca_ang));

            let p_pelvis = local_pos;
            let p_thigh  = posed_hip   + rot_x_vec(local_pos - rest_hip,   th_ang);
            let p_calf   = posed_knee  + rot_x_vec(local_pos - rest_knee,  ca_ang);
            let p_foot   = posed_ankle + rot_x_vec(local_pos - rest_ankle, fo_ang);

            local_pos = wt_pelvis * p_pelvis + wt_thigh * p_thigh + wt_calf * p_calf + wt_foot * p_foot;
        }
    }
    let world_pos = obj.model * vec4<f32>(local_pos, 1.0);
    return frame.light_view_proj * world_pos;
}
