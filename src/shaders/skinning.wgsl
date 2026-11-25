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

fn rot_y_vec(v: vec3<f32>, ang: f32) -> vec3<f32> {
    let c = cos(ang);
    let s = sin(ang);
    return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

struct PosedVertex {
    pos: vec3<f32>,
    nrm: vec3<f32>,
};

fn animate_quadruped(
    pos_in: vec3<f32>,
    nrm_in: vec3<f32>,
    pose_r: vec4<f32>,
    pose_l: vec4<f32>,
) -> PosedVertex {
    let walk_phase = pose_r.x;
    let walk_weight = clamp(pose_r.y, 0.0, 1.0);
    let eat_weight = clamp(pose_r.z, 0.0, 1.0);
    let mode = pose_r.w; // 3.0 = Horse, 4.0 = Cow, 5.0 = Sheep, 6.0 = Mulita

    let anim_time = pose_l.x;
    let idle_head_yaw = pose_l.y;
    let eat_pitch_max = pose_l.z;
    let leg_swing_amp = pose_l.w;

    var leg_top_y = 0.96;
    var knee_y = 0.48;
    var front_z = 0.44;
    var back_z = -0.44;
    var neck_pivot = vec3<f32>(0.0, 1.04, 0.42);
    var tail_pivot = vec3<f32>(0.0, 1.16, -0.64);
    var eat_dip_y = 0.16;
    var leg_z_radius = 0.22;

    if (mode > 3.5 && mode < 4.5) {
        // Cow
        leg_top_y = 0.78;
        knee_y = 0.40;
        front_z = 0.45;
        back_z = -0.46;
        neck_pivot = vec3<f32>(0.0, 0.90, 0.48);
        tail_pivot = vec3<f32>(0.0, 1.08, -0.60);
        eat_dip_y = 0.13;
        leg_z_radius = 0.24;
    } else if (mode >= 4.5 && mode < 5.5) {
        // Sheep
        leg_top_y = 0.42;
        knee_y = 0.21;
        front_z = 0.21;
        back_z = -0.22;
        neck_pivot = vec3<f32>(0.0, 0.50, 0.24);
        tail_pivot = vec3<f32>(0.0, 0.56, -0.34);
        eat_dip_y = 0.07;
        leg_z_radius = 0.14;
    } else if (mode >= 5.5) {
        // Mulita (seven-banded armadillo)
        leg_top_y = 0.12;
        knee_y = 0.06;
        front_z = 0.13;
        back_z = -0.13;
        neck_pivot = vec3<f32>(0.0, 0.13, 0.20);
        tail_pivot = vec3<f32>(0.0, 0.12, -0.21);
        eat_dip_y = 0.015;
        leg_z_radius = 0.09;
    }

    var pos = pos_in;
    var nrm = nrm_in;

    // 1. Torso idle breathing expansion, walking stride bounce, and grazing chest dip
    let above_ground_w = smoothstep(knee_y * 0.6, leg_top_y, pos.y);
    let rib_w = above_ground_w * (1.0 - smoothstep(abs(front_z) * 0.6, abs(front_z) * 1.4, abs(pos.z)));
    let breath = sin(anim_time * 2.15);
    pos.x += sign(pos.x) * 0.010 * breath * rib_w;
    pos.y += 0.007 * breath * rib_w;

    // 2-beat walking vertical bounce and slight body sway
    let stride_bob = 0.022 * abs(sin(walk_phase)) * walk_weight * (leg_top_y + 0.15) * above_ground_w;
    let stride_sway = 0.012 * cos(walk_phase) * walk_weight * above_ground_w;
    pos.y += stride_bob;
    pos.x += stride_sway;

    // Slight forward shoulder dip when lowering neck to eat grass (hooves stay planted)
    let front_chest_w = above_ground_w * smoothstep(-0.05, front_z, pos.z);
    pos.y -= eat_dip_y * eat_weight * front_chest_w;

    // 2. 4-Beat Quadruped Walking Leg Swing, Knee Flexion & Subtle Idle Weight Shift
    let is_front = pos.z >= 0.0;
    let is_right = pos.x >= 0.0;
    let leg_z = select(back_z, front_z, is_front);
    let z_dist_from_leg = abs(pos.z - leg_z);
    let w_leg_y = 1.0 - smoothstep(leg_top_y - 0.05, leg_top_y + 0.04, pos_in.y);
    let w_leg_z = 1.0 - smoothstep(leg_z_radius * 0.65, leg_z_radius, z_dist_from_leg);
    // Exclude low hanging tail tips (z < back_z - 0.12) from rear leg influence
    let not_tail = smoothstep(tail_pivot.z - 0.04, back_z + 0.06, pos_in.z);
    let w_leg = w_leg_y * w_leg_z * select(not_tail, 1.0, is_front);

    if (w_leg > 0.001) {
        // Natural 4-beat lateral walk sequence: FR (0), BL (pi/2), FL (pi), BR (3pi/2)
        var phase_off = 0.0;
        if (!is_right && !is_front) {
            phase_off = 1.5707963;
        } else if (!is_right && is_front) {
            phase_off = 3.1415926;
        } else if (is_right && !is_front) {
            phase_off = 4.7123890;
        }
        let p_leg = walk_phase + phase_off;

        // Idle subtle leg shift when standing or grazing
        let idle_shift = sin(anim_time * 0.75 + phase_off) * 0.035 * (1.0 - walk_weight);
        let hip_ang = sin(p_leg) * leg_swing_amp * walk_weight + idle_shift;

        // Forward swing phase has cos(p_leg) < 0: flex knee backward (+ang) so hoof lifts clear of the ground
        let swing_lift = max(0.0, -cos(p_leg)) * walk_weight;
        let knee_ang = swing_lift * leg_swing_amp * 0.82;

        let hip_pivot = vec3<f32>(pos.x, leg_top_y, leg_z);
        let rest_knee = vec3<f32>(pos.x, knee_y, leg_z);
        let posed_knee = hip_pivot + rot_x_vec(rest_knee - hip_pivot, hip_ang);

        let p_upper = hip_pivot + rot_x_vec(pos - hip_pivot, hip_ang);
        let p_lower = posed_knee + rot_x_vec(pos - rest_knee, hip_ang + knee_ang);
        let n_upper = rot_x_vec(nrm, hip_ang);
        let n_lower = rot_x_vec(nrm, hip_ang + knee_ang);

        let w_lower = 1.0 - smoothstep(knee_y - 0.045, knee_y + 0.045, pos_in.y);
        let p_leg_posed = mix(p_upper, p_lower, w_lower);
        let n_leg_posed = normalize(mix(n_upper, n_lower, w_lower));

        pos = mix(pos, p_leg_posed, w_leg);
        nrm = normalize(mix(nrm, n_leg_posed, w_leg));
    }

    // 3. Neck & Head Animation: Eating Grass (Grazing Downward Pitch + Chewing/Tugging Bob + Sweep) & Idle Look-Around
    let w_neck_z = smoothstep(neck_pivot.z - 0.06, neck_pivot.z + 0.14, pos_in.z);
    let w_neck_y = smoothstep(leg_top_y * 0.58, neck_pivot.y - 0.08, pos_in.y);
    let w_neck = select(w_neck_z * w_neck_y, w_neck_z, mode >= 5.5);

    if (w_neck > 0.001) {
        // Grazing rhythmic grass-pulling nod & chewing motion
        let chew_nod = (0.075 * sin(anim_time * 5.8) + 0.040 * cos(anim_time * 9.4)) * eat_weight;
        // Walking counter-balance head nod
        let walk_nod = 0.068 * sin(walk_phase * 2.0) * walk_weight * (1.0 - eat_weight * 0.7);
        // Idle gentle breathing head nod
        let idle_nod = 0.042 * sin(anim_time * 1.45) * (1.0 - eat_weight) * (1.0 - walk_weight);

        // Positive pitch in rot_x_vec rotates +Z forward snout downward toward -Y ground!
        let pitch_down = eat_pitch_max * eat_weight + chew_nod + walk_nod + idle_nod;

        // Grazing side-to-side grass sweep + idle look-around yaw
        let graze_sweep_yaw = (0.16 * sin(anim_time * 1.25) + 0.05 * cos(anim_time * 3.1)) * eat_weight;
        let head_yaw = idle_head_yaw * (1.0 - eat_weight * 0.65) + graze_sweep_yaw;

        let rel_neck = pos - neck_pivot;
        let pitched = rot_x_vec(rel_neck, pitch_down);
        let posed_neck = neck_pivot + rot_y_vec(pitched, head_yaw);

        let n_pitched = rot_x_vec(nrm, pitch_down);
        let n_posed = rot_y_vec(n_pitched, head_yaw);

        pos = mix(pos, posed_neck, w_neck);
        nrm = normalize(mix(nrm, n_posed, w_neck));
    }

    // 4. Tail Swish (fly-swatting lateral swish + walking tail lift)
    let w_tail = 1.0 - smoothstep(tail_pivot.z - 0.08, tail_pivot.z + 0.03, pos_in.z);
    if (w_tail > 0.001) {
        let tail_yaw = (0.28 * sin(anim_time * 3.4) + 0.14 * cos(anim_time * 5.6)) * (0.65 + 0.35 * eat_weight);
        // Negative rot_x_vec lifts -Z tail upward
        let tail_lift = -(0.16 * walk_weight + 0.06 * (0.5 + 0.5 * sin(anim_time * 2.1)));
        let rel_tail = pos - tail_pivot;
        let lifted = rot_x_vec(rel_tail, tail_lift);
        let posed_tail = tail_pivot + rot_y_vec(lifted, tail_yaw);

        let n_tail = rot_y_vec(rot_x_vec(nrm, tail_lift), tail_yaw);
        pos = mix(pos, posed_tail, w_tail);
        nrm = normalize(mix(nrm, n_tail, w_tail));
    }

    return PosedVertex(pos, nrm);
}

fn animate_biped_pedestrian(
    pos_in: vec3<f32>,
    nrm_in: vec3<f32>,
    pose_r: vec4<f32>,
    pose_l: vec4<f32>,
) -> PosedVertex {
    let walk_phase = pose_r.x;
    let walk_weight = clamp(pose_r.y, 0.0, 1.0);
    let sip_weight = clamp(pose_r.z, 0.0, 1.0);

    let anim_time = pose_l.x;
    let idle_head_yaw = pose_l.y;
    let leg_swing_amp = pose_l.w;

    var pos = pos_in;
    var nrm = nrm_in;

    // 1. Upper-body 2-beat walking bounce, gentle lateral hip/shoulder sway, and breathing
    let above_hip = smoothstep(0.48, 0.94, pos_in.y);
    let stride_bob = 0.018 * abs(sin(walk_phase)) * walk_weight * above_hip;
    let stride_sway = 0.010 * cos(walk_phase) * walk_weight * above_hip;
    let breath = 0.004 * sin(anim_time * 2.1) * above_hip;
    pos.y += stride_bob + breath;
    pos.x += stride_sway;

    // 2. Bipedal Anti-Phase Leg Swing & Knee Flexion (hip at y=0.90, knee at y=0.46)
    let w_leg = 1.0 - smoothstep(0.85, 0.93, pos_in.y);
    if (w_leg > 0.001) {
        let is_right = pos_in.x >= 0.0;
        let phase_off = select(3.14159265, 0.0, is_right);
        let p_leg = walk_phase + phase_off;

        let hip_ang = sin(p_leg) * leg_swing_amp * walk_weight;
        // During forward leg swing (cos(p_leg) < 0), bend knee backward (+ang in rot_x_vec moves -Y toward -Z)
        let swing_lift = max(0.0, -cos(p_leg)) * walk_weight;
        let knee_ang = swing_lift * leg_swing_amp * 0.90;

        let hip_pivot = vec3<f32>(pos.x, 0.90, 0.0);
        let rest_knee = vec3<f32>(pos.x, 0.46, 0.0);
        let posed_knee = hip_pivot + rot_x_vec(rest_knee - hip_pivot, hip_ang);

        let p_upper = hip_pivot + rot_x_vec(pos - hip_pivot, hip_ang);
        let p_lower = posed_knee + rot_x_vec(pos - rest_knee, hip_ang + knee_ang);
        let n_upper = rot_x_vec(nrm, hip_ang);
        let n_lower = rot_x_vec(nrm, hip_ang + knee_ang);

        let w_lower = 1.0 - smoothstep(0.41, 0.51, pos_in.y);
        let p_leg_posed = mix(p_upper, p_lower, w_lower);
        let n_leg_posed = normalize(mix(n_upper, n_lower, w_lower));

        pos = mix(pos, p_leg_posed, w_leg);
        nrm = normalize(mix(nrm, n_leg_posed, w_leg));
    }

    // 3. Counter-Phase Arm Swing for free walking arms (mate-holding arms at z > 0.11 stay steady!)
    let w_arm_side = smoothstep(0.15, 0.19, abs(pos_in.x))
        * smoothstep(0.88, 1.34, pos_in.y)
        * (1.0 - smoothstep(1.36, 1.44, pos_in.y))
        * (1.0 - smoothstep(0.08, 0.15, pos_in.z));
    if (w_arm_side > 0.001) {
        let is_right = pos_in.x >= 0.0;
        let arm_phase = walk_phase + select(0.0, 3.14159265, is_right);
        let arm_ang = sin(arm_phase) * 0.24 * walk_weight;
        let sh_pivot = vec3<f32>(pos.x, 1.33, 0.0);
        let p_arm = sh_pivot + rot_x_vec(pos - sh_pivot, arm_ang);
        let n_arm = rot_x_vec(nrm, arm_ang);
        pos = mix(pos, p_arm, w_arm_side);
        nrm = normalize(mix(nrm, n_arm, w_arm_side));
    }

    // 4. Head Look-Around Yaw & Mate-Sipping / Sea-Gazing Nod (neck pivot at y=1.40)
    let w_head = smoothstep(1.38, 1.45, pos_in.y);
    if (w_head > 0.001) {
        let neck_pivot = vec3<f32>(0.0, 1.40, 0.0);
        let sip_nod = 0.10 * sip_weight * (0.5 + 0.5 * sin(anim_time * 1.9));
        let rel_h = pos - neck_pivot;
        let pitched = rot_x_vec(rel_h, sip_nod);
        let posed_h = neck_pivot + rot_y_vec(pitched, idle_head_yaw * 0.65);
        let n_h = rot_y_vec(rot_x_vec(nrm, sip_nod), idle_head_yaw * 0.65);
        pos = mix(pos, posed_h, w_head);
        nrm = normalize(mix(nrm, n_h, w_head));
    }

    return PosedVertex(pos, nrm);
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

    if (job.skin_pose_r.w >= 6.5) {
        let posed = animate_biped_pedestrian(local_pos, local_nrm, job.skin_pose_r, job.skin_pose_l);
        local_pos = posed.pos;
        local_nrm = posed.nrm;
    } else if (job.skin_pose_r.w >= 2.5) {
        let posed = animate_quadruped(local_pos, local_nrm, job.skin_pose_r, job.skin_pose_l);
        local_pos = posed.pos;
        local_nrm = posed.nrm;
    } else if (job.skin_pose_r.w > 0.5) {
        let anim_phase = job.skin_pose_l.w;
        local_pos = animate_upper_body(local_pos, anim_phase);

        if (job.skin_pose_r.w < 1.5) {
            // Exact Blender-authored leg-chain weight baked in vertex UV.x by Catmull-Clark Subsurf:
            //   0.0        = upper body / torso / pelvis / arms / wrists / hands (untouched by leg IK)
            //   0.0 .. 1.0 = smooth hip-to-thigh transition around seated rest pose (REST_TH_ANG = 0.90 rad)
            //   1.0 .. 2.0 = smooth thigh-to-calf knee transition (REST_CA_ANG = -0.35 rad)
            //   2.0        = rigid calf & white cycling sock down to rest_ankle
            //   3.0        = rigid low-cut cycling shoe around rest_ankle
            let bw_x = unpack2x16float(v.uv).x;
            if (bw_x > 0.001) {
                let is_right = local_pos.x >= 0.0;
                let side = select(-1.0, 1.0, is_right);
                let angles = select(job.skin_pose_l.xyz, job.skin_pose_r.xyz, is_right);

                let d_th = angles.x - 0.90;
                let d_ca = angles.y - (-0.35);
                let d_fo = angles.z;

                // Exact seated Blender rest-pose joint positions (in wgpu Y-up, -Z forward):
                let rest_hip   = vec3<f32>(side * 0.102,  0.000000,  0.000000);
                let rest_knee  = vec3<f32>(side * 0.102, -0.267292, -0.336829);
                let rest_ankle = vec3<f32>(side * 0.124, -0.647738, -0.197956);

                let posed_hip   = rest_hip;
                let posed_knee  = posed_hip  + vec3<f32>(0.0, -0.430 * cos(angles.x), -0.430 * sin(angles.x));
                let posed_ankle = posed_knee + vec3<f32>(side * 0.022, -0.405 * cos(angles.y), -0.405 * sin(angles.y));

                if (bw_x <= 1.0) {
                    let eff_ang = d_th * bw_x;
                    local_pos = posed_hip + rot_x_vec(local_pos - rest_hip, eff_ang);
                    local_nrm = rot_x_vec(local_nrm, eff_ang);
                } else if (bw_x <= 2.0) {
                    let w_knee = bw_x - 1.0;
                    let eff_ang = mix(d_th, d_ca, w_knee);
                    local_pos = posed_knee + rot_x_vec(local_pos - rest_knee, eff_ang);
                    local_nrm = rot_x_vec(local_nrm, eff_ang);
                } else {
                    let w_ankle = clamp(bw_x - 2.0, 0.0, 1.0);
                    let eff_ang = mix(d_ca, d_fo, w_ankle);
                    local_pos = posed_ankle + rot_x_vec(local_pos - rest_ankle, eff_ang);
                    local_nrm = rot_x_vec(local_nrm, eff_ang);
                }
            }
        }
    }

    v.pos_x = local_pos.x;
    v.pos_y = local_pos.y;
    v.pos_z = local_pos.z;
    v.normal_mat = pack4x8snorm(vec4<f32>(local_nrm, unpacked_nrm.w));
    dst_verts[job.dst_offset + vid] = v;
}
