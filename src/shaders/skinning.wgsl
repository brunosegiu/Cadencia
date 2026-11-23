struct CompressedVertex {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    normal_mat: u32, // Snorm8x4: xyz = normal, w = material_type
    uv: u32,         // Float16x2
    color: u32,      // Unorm8x4
};

struct SkinJob {
    skin_pose_r: vec4<f32>, // (thigh_ang_r, calf_ang_r, foot_ang_r, mode: 1.0 = full leg+torso LBS, 2.0 = backpack follow)
    skin_pose_l: vec4<f32>, // (thigh_ang_l, calf_ang_l, foot_ang_l, anim_phase)
    src_offset: u32,
    dst_offset: u32,
    vertex_count: u32,
    _pad: u32,
};

struct SkinDispatchParams {
    job_count: u32,
    max_verts_per_job: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: SkinDispatchParams;
@group(0) @binding(1) var<storage, read> jobs: array<SkinJob>;
@group(0) @binding(2) var<storage, read> src_verts: array<CompressedVertex>;
@group(0) @binding(3) var<storage, read_write> dst_verts: array<CompressedVertex>;

fn rot_x_vec(v: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

fn animate_upper_body(local_pos: vec3<f32>, anim_phase: f32) -> vec3<f32> {
    let x = local_pos.x;
    let y_bl = -local_pos.z;
    let z_bl = local_pos.y;
    let side = select(-1.0, 1.0, x >= 0.0);

    // Hands at y_bl >= 0.61 stay locked onto the brake hoods while torso, shoulders & elbows flex!
    let hand_lock = 1.0 - smoothstep(0.30, 0.61, y_bl);
    let torso_w = smoothstep(0.04, 0.38, z_bl) * hand_lock;

    let sway_x = 0.011 * sin(anim_phase) * torso_w;
    let bob_z  = (0.0076 * sin(anim_phase * 2.0) + 0.0042 * cos(anim_phase * 0.85)) * torso_w;
    let breath_y = 0.006 * sin(anim_phase * 2.0) * torso_w;

    // Dynamic elbow flex & pump (peaks around elbow y_bl ~ 0.37, fades to 0 at shoulder & wrist)
    let elbow_zone = exp(-pow((y_bl - 0.37) / 0.13, 2.0)) * smoothstep(0.11, 0.18, abs(x));
    let side_phase = anim_phase + select(3.14159265, 0.0, x >= 0.0);
    let elbow_dx = side * 0.012 * (0.5 + 0.5 * sin(side_phase)) * elbow_zone;
    let elbow_dz = -0.0136 * (0.5 + 0.5 * cos(side_phase)) * elbow_zone;

    return vec3<f32>(
        local_pos.x + sway_x + elbow_dx,
        local_pos.y + bob_z + elbow_dz,
        local_pos.z - breath_y
    );
}

@compute @workgroup_size(64, 1, 1)
fn cs_skin_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = gid.x;
    let job_id = gid.y;
    if (job_id >= params.job_count) {
        return;
    }
    let job = jobs[job_id];
    if (vid >= job.vertex_count) {
        return;
    }

    var v = src_verts[job.src_offset + vid];
    var local_pos = vec3<f32>(v.pos_x, v.pos_y, v.pos_z);
    let unpacked_nrm = unpack4x8snorm(v.normal_mat);
    var local_nrm = normalize(unpacked_nrm.xyz);

    if (job.skin_pose_r.w > 0.5) {
        let anim_phase = job.skin_pose_l.w;
        local_pos = animate_upper_body(local_pos, anim_phase);

        if (job.skin_pose_r.w < 1.5) {
            let x = local_pos.x;
            let y_bl = -local_pos.z;
            let z_bl = local_pos.y;
            if (!(z_bl > 0.045 || (z_bl > -0.14 && y_bl > 0.15))) {
                let angles = select(job.skin_pose_l.xyz, job.skin_pose_r.xyz, x >= 0.0);
                let side = select(-1.0, 1.0, x >= 0.0);
                let th_ang = angles.x;
                let ca_ang = angles.y;
                let fo_ang = angles.z;

                // Anatomical hip crease: keeps full round buttcheeks (y_bl < -0.03) anchored on the saddle!
                let hip_fold_z   = -0.015 + clamp(y_bl * 0.45, -0.045, 0.030);
                let w_hip_thigh  = 1.0 - smoothstep(hip_fold_z - 0.045, hip_fold_z + 0.030, z_bl);
                let w_pelvis     = 1.0 - w_hip_thigh;
                let w_knee_calf  = 1.0 - smoothstep(-0.485, -0.375, z_bl);
                let w_ankle_foot = 1.0 - smoothstep(-0.865, -0.815, z_bl);

                let wt_pelvis = w_pelvis;
                let wt_thigh  = w_hip_thigh * (1.0 - w_knee_calf);
                let wt_calf   = w_hip_thigh * w_knee_calf * (1.0 - w_ankle_foot);
                let wt_foot   = w_hip_thigh * w_knee_calf * w_ankle_foot;

                let rest_hip   = vec3<f32>(side * 0.105,  0.000, 0.0);
                let rest_knee  = vec3<f32>(side * 0.105, -0.430, 0.0);
                let rest_ankle = vec3<f32>(side * 0.105, -0.845, 0.0);

                let posed_hip   = rest_hip;
                let posed_knee  = posed_hip  + vec3<f32>(0.0, -0.430 * cos(th_ang), -0.430 * sin(th_ang));
                let posed_ankle = posed_knee + vec3<f32>(0.0, -0.415 * cos(ca_ang), -0.415 * sin(ca_ang));

                let p_pelvis = local_pos;
                let p_thigh  = posed_hip   + rot_x_vec(local_pos - rest_hip,   th_ang);
                let p_calf   = posed_knee  + rot_x_vec(local_pos - rest_knee,  ca_ang);
                let p_foot   = posed_ankle + rot_x_vec(local_pos - rest_ankle, fo_ang);

                let n_pelvis = local_nrm;
                let n_thigh  = rot_x_vec(local_nrm, th_ang);
                let n_calf   = rot_x_vec(local_nrm, ca_ang);
                let n_foot   = rot_x_vec(local_nrm, fo_ang);

                local_pos = wt_pelvis * p_pelvis + wt_thigh * p_thigh + wt_calf * p_calf + wt_foot * p_foot;
                local_nrm = normalize(wt_pelvis * n_pelvis + wt_thigh * n_thigh + wt_calf * n_calf + wt_foot * n_foot);
            }
        }
    }

    v.pos_x = local_pos.x;
    v.pos_y = local_pos.y;
    v.pos_z = local_pos.z;
    v.normal_mat = pack4x8snorm(vec4<f32>(local_nrm, unpacked_nrm.w));
    dst_verts[job.dst_offset + vid] = v;
}
