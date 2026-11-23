use glam::{Mat4, Quat, Vec3};

use crate::assets::ModelId;
use crate::world::RouteSpline;

#[derive(Debug, Clone)]
pub struct RiderProfile {
    pub name: &'static str,
    pub level: u32,
    pub w_per_kg: f32,
    pub jersey_primary: [f32; 4],
    pub jersey_secondary: [f32; 4],
    pub jersey_accent: [f32; 4],
    pub helmet_color: [f32; 4],
    pub helmet_accent: [f32; 4],
    pub frame_color: [f32; 4],
    pub has_backpack: bool,
    pub avatar_skin: [u8; 3],
    pub avatar_hair: [u8; 3],
    pub avatar_shirt: [u8; 3],
    pub is_player: bool,
}

#[derive(Debug, Clone)]
pub struct AiRider {
    pub profile: RiderProfile,
    pub distance_m: f32,
    pub lateral_m: f32,
    pub speed_mps: f32,
    pub crank_angle_rad: f32,
    pub wheel_angle_rad: f32,
    pub lean_angle_rad: f32,
}

#[derive(Debug, Clone)]
pub struct RiderPartDraw {
    pub model: ModelId,
    pub transform: Mat4,
    pub jersey_primary: [f32; 4],
    pub jersey_secondary: [f32; 4],
    pub jersey_accent: [f32; 4],
    pub helmet_color: [f32; 4],
    pub helmet_accent: [f32; 4],
    pub frame_color: [f32; 4],
    pub object_id: u32,
    pub skin_pose_r: [f32; 4],
    pub skin_pose_l: [f32; 4],
}

pub fn default_player_profile() -> RiderProfile {
    RiderProfile {
        name: "S. Miller",
        level: 12,
        w_per_kg: 2.8,
        jersey_primary: [0.282, 0.573, 0.808, 1.0],   // cobalt sky blue (turnaround sheet)
        jersey_secondary: [0.839, 0.306, 0.290, 1.0], // coral red panels & yoke
        jersey_accent: [0.949, 0.800, 0.298, 1.0],    // golden yellow sleeve patches
        helmet_color: [0.424, 0.455, 0.494, 1.0],     // matte slate grey vented shell
        helmet_accent: [0.839, 0.322, 0.298, 1.0],    // coral-red aero ribs
        frame_color: [0.463, 0.525, 0.588, 1.0],      // classic steel blue-grey road frame
        has_backpack: true,
        avatar_skin: [226, 178, 140],
        avatar_hair: [156, 102, 62],
        avatar_shirt: [72, 146, 206],
        is_player: true,
    }
}

pub fn spawn_ai_peloton(start_dist_m: f32, route_len_m: f32) -> Vec<AiRider> {
    let specs: &[(&str, f32, f32, f32, [f32; 4], [f32; 4], [f32; 4], [f32; 4], [u8; 3], [u8; 3], [u8; 3])] = &[
        // Left-ahead AI rider (matches O. Hansen in Reference Images: dark teal patterned kit, dark helmet)
        ("O. Hansen", 7.4, -1.55, 2.5, [0.10, 0.24, 0.22, 1.0], [0.24, 0.58, 0.48, 1.0], [0.18, 0.20, 0.23, 1.0], [0.26, 0.32, 0.38, 1.0], [218, 174, 146], [198, 118, 72], [58, 118, 108]),
        // Right-ahead AI rider (matches H. Dupont in Reference Images: seafoam turquoise jersey, crisp white helmet)
        ("H. Dupont", 5.6, 1.46, 2.5, [0.34, 0.68, 0.65, 1.0], [0.26, 0.54, 0.54, 1.0], [0.95, 0.95, 0.93, 1.0], [0.38, 0.56, 0.62, 1.0], [224, 178, 152], [110, 74, 56], [92, 164, 155]),
        ("A. Lsmith", -16.0, -0.8, 2.5, [0.22, 0.28, 0.54, 1.0], [0.76, 0.24, 0.28, 1.0], [0.22, 0.24, 0.28, 1.0], [0.45, 0.35, 0.32, 1.0], [164, 122, 98], [48, 40, 38], [56, 72, 138]),
        ("O. Hansen ", -20.0, 0.9, 2.5, [0.36, 0.52, 0.32, 1.0], [0.62, 0.72, 0.44, 1.0], [0.85, 0.83, 0.78, 1.0], [0.32, 0.40, 0.36, 1.0], [212, 168, 140], [148, 116, 84], [116, 134, 92]),
        ("M. Dupont", -24.0, -1.1, 2.5, [0.34, 0.44, 0.62, 1.0], [0.54, 0.66, 0.80, 1.0], [0.26, 0.28, 0.32, 1.0], [0.28, 0.36, 0.48, 1.0], [222, 178, 154], [78, 54, 46], [88, 112, 158]),
        ("L. Moreau", -32.0, 0.6, 2.4, [0.16, 0.46, 0.34, 1.0], [0.88, 0.72, 0.26, 1.0], [0.84, 0.84, 0.82, 1.0], [0.52, 0.36, 0.28, 1.0], [208, 160, 132], [58, 44, 40], [42, 118, 86]),
    ];
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| AiRider {
            profile: RiderProfile {
                name: s.0,
                level: 10 + (i as u32 % 7),
                w_per_kg: s.3,
                jersey_primary: s.4,
                jersey_secondary: s.5,
                jersey_accent: s.5,
                helmet_color: s.6,
                helmet_accent: s.5,
                frame_color: s.7,
                has_backpack: false,
                avatar_skin: s.8,
                avatar_hair: s.9,
                avatar_shirt: s.10,
                is_player: false,
            },
            distance_m: (start_dist_m + s.1).rem_euclid(route_len_m),
            lateral_m: s.2,
            speed_mps: 24.5 / 3.6,
            crank_angle_rad: -1.1 + (i as f32) * 1.45,
            wheel_angle_rad: (i as f32) * 0.7,
            lean_angle_rad: 0.0,
        })
        .collect()
}

pub fn step_ai_peloton(dt: f32, spline: &RouteSpline, riders: &mut [AiRider]) {
    let dt = dt.clamp(0.0, 0.1);
    for (idx, r) in riders.iter_mut().enumerate() {
        let s = spline.sample(r.distance_m);
        // Gentle realistic speed modulation based on grade & target W/kg
        let grade_factor = (1.0 - s.grade_pct * 0.045).clamp(0.45, 1.55);
        let target_v = (24.5 / 3.6) * (r.profile.w_per_kg / 2.55) * grade_factor;
        r.speed_mps += (target_v - r.speed_mps) * (dt * 1.8).min(1.0);
        let ds = r.speed_mps * dt;
        r.distance_m = (r.distance_m + ds).rem_euclid(spline.total_length_m);
        r.wheel_angle_rad = (r.wheel_angle_rad + ds / 0.34).rem_euclid(std::f32::consts::TAU);
        let cad_rpm = 82.0 + (idx as f32 * 3.0) - s.grade_pct * 1.2;
        r.crank_angle_rad =
            (r.crank_angle_rad + (cad_rpm / 60.0) * std::f32::consts::TAU * dt).rem_euclid(std::f32::consts::TAU);
        let target_lean = (r.speed_mps * r.speed_mps * s.curvature / 9.81).atan().clamp(-0.30, 0.30) * 0.65;
        r.lean_angle_rad += (target_lean - r.lean_angle_rad) * (dt * 5.0).min(1.0);
    }
}

/// Builds posed 3D draw parts for one cyclist (frame, 2 spinning wheels, spinning crank, torso, head, optional backpack, 2-bone IK thighs & calves).
/// Note on glTF Y-up coordinate mapping from Blender (+Y forward, +Z up, +X right):
/// In glTF / wgpu space: forward = -Z, up = +Y, right = +X!
/// So a Blender point `(x, y, z)` becomes `(x, z, -y)` in local rider space.
pub fn build_rider_draw_parts(
    spline: &RouteSpline,
    distance_m: f32,
    lateral_m: f32,
    wheel_angle_rad: f32,
    crank_angle_rad: f32,
    lean_angle_rad: f32,
    time_s: f32,
    profile: &RiderProfile,
    base_object_id: u32,
    out: &mut Vec<RiderPartDraw>,
) {
    let s = spline.sample(distance_m);
    let world_pos = s.pos + s.right * lateral_m + Vec3::Y * 0.04;

    // Rider basis: local +X = right, local +Y = up, local -Z = forward tangent
    let z_back = (-s.tangent).normalize_or_zero();
    let x_right = s.up.cross(z_back).normalize_or_zero();
    let y_up = z_back.cross(x_right).normalize_or_zero();
    let basis = Mat4::from_cols(
        x_right.extend(0.0),
        y_up.extend(0.0),
        z_back.extend(0.0),
        world_pos.extend(1.0),
    ) * Mat4::from_rotation_z(-lean_angle_rad);

    let push_skinned = |out: &mut Vec<RiderPartDraw>,
                        model: ModelId,
                        local: Mat4,
                        id_off: u32,
                        skin_pose_r: [f32; 4],
                        skin_pose_l: [f32; 4]| {
        out.push(RiderPartDraw {
            model,
            transform: basis * local,
            jersey_primary: profile.jersey_primary,
            jersey_secondary: profile.jersey_secondary,
            jersey_accent: profile.jersey_accent,
            helmet_color: profile.helmet_color,
            helmet_accent: profile.helmet_accent,
            frame_color: profile.frame_color,
            object_id: base_object_id + id_off,
            skin_pose_r,
            skin_pose_l,
        });
    };
    let push = |out: &mut Vec<RiderPartDraw>, model: ModelId, local: Mat4, id_off: u32| {
        push_skinned(out, model, local, id_off, [0.0; 4], [0.0; 4]);
    };

    // Frame & Crank share object_id base
    push(out, ModelId::BikeFrame, Mat4::IDENTITY, 0);

    // Wheels share distinct id +1 (Blender rear (0, -0.475, 0.340), front (0, +0.465, 0.340) -> wgpu (0, 0.340, +0.475 / -0.465))
    let wheel_rot = Mat4::from_rotation_x(-wheel_angle_rad);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.340, 0.475)) * wheel_rot, 1);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.340, -0.465)) * wheel_rot, 1);

    // Bottom bracket: Blender (0, -0.070, 0.275) -> wgpu (0, 0.275, 0.070)
    let crank_rot = Mat4::from_rotation_x(-crank_angle_rad);
    push(out, ModelId::BikeCrank, Mat4::from_translation(Vec3::new(0.0, 0.275, 0.070)) * crank_rot, 0);

    // Solve 2-bone analytical leg IK for RiderArmature skinning (l1 = 0.430, l2 = 0.415)
    let hip_center = Vec3::new(0.0, 0.965, 0.235);
    let l1 = 0.430_f32;
    let l2 = 0.415_f32;
    let solve_leg_angles = |side: f32, theta_off: f32| -> [f32; 3] {
        let th = crank_angle_rad + theta_off;
        let hip = Vec3::new(side * 0.105, 0.965, 0.235);
        let pedal = Vec3::new(
            side * 0.118,
            0.275 - 0.165 * th.sin(),
            0.070 - 0.165 * th.cos(),
        );
        let ankle = pedal + Vec3::new(0.0, 0.056, 0.035);
        let d_vec = ankle - hip;
        let d = d_vec.length().clamp(0.12, l1 + l2 - 0.004);
        let base_ang = (-d_vec.z).atan2(-d_vec.y);
        let cos_a = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
        let thigh_ang = base_ang + cos_a.acos();
        let knee = hip + Vec3::new(0.0, -l1 * thigh_ang.cos(), -l1 * thigh_ang.sin());
        let calf_vec = ankle - knee;
        let calf_ang = (-calf_vec.z).atan2(-calf_vec.y);
        let foot_ang = 0.10 * th.cos();
        [thigh_ang, calf_ang, foot_ang]
    };
    let ang_r = solve_leg_angles(1.0, 0.0);
    let ang_l = solve_leg_angles(-1.0, std::f32::consts::PI);

    // Continuous pedaling + idle breathing/sway animation phase for torso, shoulders, elbows, and neck/head!
    let rider_seed = (base_object_id as f32) * 0.37;
    let anim_phase = crank_angle_rad + time_s * 1.8 + rider_seed;

    // Unified seamless skinned cyclist body + shorts + thighs + knees + calves + shoes (id +2)
    push_skinned(
        out,
        ModelId::RiderTorso,
        Mat4::from_translation(hip_center),
        2,
        [ang_r[0], ang_r[1], ang_r[2], 1.0],
        [ang_l[0], ang_l[1], ang_l[2], anim_phase],
    );
    if profile.has_backpack {
        // Backpack follows upper-body sway & breathing heave via skin_pose_r.w = 2.0
        push_skinned(
            out,
            ModelId::RiderBackpack,
            Mat4::from_translation(hip_center),
            3,
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, anim_phase],
        );
    }

    // Neck & Head subtle idle + pedaling animation (toned down ~15%: shoulder-follow sway, breathing bob, look-around yaw, nod pitch, counter-roll)
    let head_sway_x = 0.011 * anim_phase.sin();
    let head_bob_y  = 0.0076 * (anim_phase * 2.0).sin() + 0.0042 * (anim_phase * 0.85).cos();
    let head_fwd_z  = -0.006 * (anim_phase * 2.0).sin();
    let head_yaw    = 0.055 * (time_s * 0.85 + rider_seed).sin() + 0.024 * anim_phase.cos() - lean_angle_rad * 0.30;
    let head_pitch  = 0.032 * (anim_phase * 2.0).cos() + 0.019 * (time_s * 1.5 + rider_seed).sin();
    let head_roll   = -0.036 * anim_phase.sin() + 0.015 * (time_s * 1.2 + rider_seed).cos();

    let head_xform = Mat4::from_translation(Vec3::new(
        head_sway_x,
        1.455 + head_bob_y,
        -0.015 + head_fwd_z,
    )) * Mat4::from_euler(glam::EulerRot::YXZ, head_yaw, head_pitch, head_roll);

    push(
        out,
        ModelId::RiderHeadHelmet,
        head_xform,
        4,
    );
}

#[allow(dead_code)]
fn _keep_quat(_: Quat) {}
