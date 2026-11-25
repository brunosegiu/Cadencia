use glam::{Mat4, Quat, Vec3};

use crate::assets::{ModelId, SkeletonInfo};
use crate::world::terrain::terrain_offset_y;
use crate::world::{BiomeId, RouteSpline};

pub const MAX_CYCLIST_BONES: usize = 16;

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
    pub base_lateral_m: f32,
    pub speed_mps: f32,
    /// `+1.0` = riding in the same direction as the player (right side of road);
    /// `-1.0` = riding in the opposite direction / oncoming (left side of road).
    pub direction: f32,
    /// `0` for solo cyclists; `1..=N` for members of a cohesive cycling group.
    pub group_id: u32,
    /// Longitudinal offset within a cycling group (`0.0` for solo riders or group leader).
    pub group_offset_m: f32,
    pub crank_angle_rad: f32,
    pub wheel_angle_rad: f32,
    pub lean_angle_rad: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiderPoseMode {
    Pedaling,
    MenuStopped,
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
    pub bone_matrices: Option<[Mat4; MAX_CYCLIST_BONES]>,
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
    // Mix of solo cyclists and cohesive cycling groups (some riding the player's direction on the right side
    // of the road, some riding the opposite direction on the left side of the road), spaced around the loop
    // so the player meets ~2 solo cyclists or cycling groups per minute!
    // Tuple: (name, dist_offset_m, base_lateral_m, w_per_kg, direction, group_id, group_offset_m,
    //         jersey_primary, jersey_secondary, helmet_color, frame_color, skin, hair, shirt)
    #[allow(clippy::type_complexity)]
    let specs: &[(
        &str,
        f32,
        f32,
        f32,
        f32,
        u32,
        f32,
        [f32; 4],
        [f32; 4],
        [f32; 4],
        [f32; 4],
        [u8; 3],
        [u8; 3],
        [u8; 3],
    )] = &[
        // 1. Opening Solo Cyclist ahead in the right lane (same direction, +14m ahead)
        (
            "H. Dupont",
            14.5,
            1.42,
            2.45,
            1.0,
            0,
            0.0,
            [0.34, 0.68, 0.65, 1.0],
            [0.26, 0.54, 0.54, 1.0],
            [0.95, 0.95, 0.93, 1.0],
            [0.38, 0.56, 0.62, 1.0],
            [224, 178, 152],
            [110, 74, 56],
            [92, 164, 155],
        ),
        // 2. Opening Oncoming Solo Cyclist in the left lane (opposite direction, +29m ahead riding toward player!)
        (
            "O. Hansen",
            29.0,
            -1.25,
            2.65,
            -1.0,
            0,
            0.0,
            [0.10, 0.24, 0.22, 1.0],
            [0.24, 0.58, 0.48, 1.0],
            [0.18, 0.20, 0.23, 1.0],
            [0.26, 0.32, 0.38, 1.0],
            [218, 174, 146],
            [198, 118, 72],
            [58, 118, 108],
        ),
        // 3. Oncoming Cycling Group #1 (3-rider paceline/echelon in the left lane, ~360m ahead -> met after ~26s)
        (
            "A. Lsmith",
            360.0,
            -1.08,
            2.70,
            -1.0,
            1,
            0.0,
            [0.22, 0.28, 0.54, 1.0],
            [0.76, 0.24, 0.28, 1.0],
            [0.22, 0.24, 0.28, 1.0],
            [0.45, 0.35, 0.32, 1.0],
            [164, 122, 98],
            [48, 40, 38],
            [56, 72, 138],
        ),
        (
            "M.Silva",
            362.5,
            -1.72,
            2.70,
            -1.0,
            1,
            2.5,
            [0.22, 0.28, 0.54, 1.0],
            [0.86, 0.74, 0.24, 1.0],
            [0.92, 0.90, 0.86, 1.0],
            [0.32, 0.38, 0.46, 1.0],
            [212, 168, 140],
            [148, 116, 84],
            [56, 72, 138],
        ),
        (
            "C. Rossi",
            365.0,
            -1.22,
            2.70,
            -1.0,
            1,
            5.0,
            [0.25, 0.32, 0.58, 1.0],
            [0.76, 0.24, 0.28, 1.0],
            [0.22, 0.24, 0.28, 1.0],
            [0.40, 0.42, 0.45, 1.0],
            [220, 174, 148],
            [64, 48, 40],
            [62, 80, 148],
        ),
        // 4. Same-Direction Solo Climber/Cruiser in the right lane (+75m ahead, slower 2.05 W/kg so player catches them in ~50s)
        (
            "M. Dupont",
            75.0,
            1.52,
            2.05,
            1.0,
            0,
            0.0,
            [0.34, 0.44, 0.62, 1.0],
            [0.54, 0.66, 0.80, 1.0],
            [0.26, 0.28, 0.32, 1.0],
            [0.28, 0.36, 0.48, 1.0],
            [222, 178, 154],
            [78, 54, 46],
            [88, 112, 158],
        ),
        // 5. Oncoming Solo Sprinter in the left lane (~760m ahead -> met after ~55s)
        (
            "L. Moreau",
            760.0,
            -1.28,
            2.85,
            -1.0,
            0,
            0.0,
            [0.16, 0.46, 0.34, 1.0],
            [0.88, 0.72, 0.26, 1.0],
            [0.84, 0.84, 0.82, 1.0],
            [0.52, 0.36, 0.28, 1.0],
            [208, 160, 132],
            [58, 44, 40],
            [42, 118, 86],
        ),
        // 6. Same-Direction Cycling Group #2 (3-rider group on the right side, +160m ahead at 2.15 W/kg)
        (
            "E. Vance",
            160.0,
            0.95,
            2.15,
            1.0,
            2,
            0.0,
            [0.78, 0.32, 0.24, 1.0],
            [0.94, 0.82, 0.32, 1.0],
            [0.92, 0.92, 0.90, 1.0],
            [0.30, 0.32, 0.35, 1.0],
            [226, 180, 150],
            [92, 64, 48],
            [198, 82, 62],
        ),
        (
            "K. Lind",
            157.6,
            1.58,
            2.15,
            1.0,
            2,
            -2.4,
            [0.78, 0.32, 0.24, 1.0],
            [0.94, 0.82, 0.32, 1.0],
            [0.24, 0.26, 0.30, 1.0],
            [0.36, 0.38, 0.42, 1.0],
            [230, 188, 160],
            [210, 168, 96],
            [198, 82, 62],
        ),
        (
            "P. Gomez",
            155.2,
            1.18,
            2.15,
            1.0,
            2,
            -4.8,
            [0.78, 0.32, 0.24, 1.0],
            [0.22, 0.26, 0.34, 1.0],
            [0.92, 0.92, 0.90, 1.0],
            [0.28, 0.30, 0.34, 1.0],
            [196, 152, 122],
            [44, 36, 32],
            [198, 82, 62],
        ),
        // 7. Oncoming Cycling Group #3 (4-rider 2x2 peloton in the left lane, ~1180m ahead -> met after ~85s)
        (
            "R. Weber",
            1180.0,
            -1.05,
            2.60,
            -1.0,
            3,
            0.0,
            [0.36, 0.52, 0.32, 1.0],
            [0.62, 0.72, 0.44, 1.0],
            [0.85, 0.83, 0.78, 1.0],
            [0.32, 0.40, 0.36, 1.0],
            [212, 168, 140],
            [148, 116, 84],
            [116, 134, 92],
        ),
        (
            "T. Novak",
            1180.4,
            -1.72,
            2.60,
            -1.0,
            3,
            0.4,
            [0.36, 0.52, 0.32, 1.0],
            [0.62, 0.72, 0.44, 1.0],
            [0.24, 0.28, 0.26, 1.0],
            [0.34, 0.42, 0.38, 1.0],
            [218, 174, 148],
            [72, 56, 44],
            [116, 134, 92],
        ),
        (
            "D. Costa",
            1183.0,
            -1.10,
            2.60,
            -1.0,
            3,
            3.0,
            [0.36, 0.52, 0.32, 1.0],
            [0.88, 0.78, 0.32, 1.0],
            [0.85, 0.83, 0.78, 1.0],
            [0.30, 0.38, 0.34, 1.0],
            [188, 144, 116],
            [38, 32, 28],
            [116, 134, 92],
        ),
        (
            "J. Borel",
            1183.4,
            -1.76,
            2.60,
            -1.0,
            3,
            3.4,
            [0.36, 0.52, 0.32, 1.0],
            [0.62, 0.72, 0.44, 1.0],
            [0.24, 0.28, 0.26, 1.0],
            [0.32, 0.40, 0.36, 1.0],
            [224, 180, 152],
            [120, 86, 60],
            [116, 134, 92],
        ),
    ];
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| AiRider {
            profile: RiderProfile {
                name: s.0,
                level: 10 + (i as u32 % 9),
                w_per_kg: s.3,
                jersey_primary: s.7,
                jersey_secondary: s.8,
                jersey_accent: s.8,
                helmet_color: s.9,
                helmet_accent: s.8,
                frame_color: s.10,
                has_backpack: false,
                avatar_skin: s.11,
                avatar_hair: s.12,
                avatar_shirt: s.13,
                is_player: false,
            },
            distance_m: (start_dist_m + s.1).rem_euclid(route_len_m),
            lateral_m: s.2,
            base_lateral_m: s.2,
            speed_mps: (24.0 / 3.6) * (s.3 / 2.55),
            direction: s.4,
            group_id: s.5,
            group_offset_m: s.6,
            crank_angle_rad: -1.1 + (i as f32) * 1.45,
            wheel_angle_rad: (i as f32) * 0.7,
            lean_angle_rad: 0.0,
        })
        .collect()
}

pub fn step_ai_peloton(
    dt: f32,
    spline: &RouteSpline,
    player_dist_m: f32,
    player_lat_m: f32,
    riders: &mut [AiRider],
) {
    let dt = dt.clamp(0.0, 0.1);
    let total = spline.total_length_m;

    // First compute leader distance & speed for each cycling group (group_id > 0) so groups stay tightly together
    let mut group_leaders: [(f32, f32, bool); 8] = [(0.0, 0.0, false); 8];

    for (idx, r) in riders.iter_mut().enumerate() {
        let dir = if r.direction >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        let s = spline.sample(r.distance_m);
        // Signed grade along the rider's actual travel direction (uphill for oncoming riders is -s.grade_pct!)
        let signed_grade = s.grade_pct * dir;
        let grade_factor = (1.0 - signed_grade * 0.045).clamp(0.45, 1.55);
        let target_v = (24.2 / 3.6) * (r.profile.w_per_kg / 2.55) * grade_factor;
        r.speed_mps += (target_v - r.speed_mps) * (dt * 1.8).min(1.0);
        let ds = r.speed_mps * dt;

        if r.group_id > 0 && (r.group_id as usize) < group_leaders.len() {
            let gid = r.group_id as usize;
            if !group_leaders[gid].2 {
                r.distance_m = (r.distance_m + dir * ds).rem_euclid(total);
                group_leaders[gid] = (r.distance_m, r.speed_mps, true);
            } else {
                // Group follower stays locked in formation relative to the group leader
                let (lead_d, lead_v, _) = group_leaders[gid];
                r.speed_mps = lead_v;
                r.distance_m = (lead_d + r.group_offset_m).rem_euclid(total);
            }
        } else {
            r.distance_m = (r.distance_m + dir * ds).rem_euclid(total);
        }

        // Smooth lateral passing avoidance when a same-direction rider passes near the player on the right side
        let mut target_lat = r.base_lateral_m;
        if dir > 0.0 {
            let mut d_rel = (r.distance_m - player_dist_m).rem_euclid(total);
            if d_rel > 0.5 * total {
                d_rel -= total;
            }
            if d_rel.abs() < 7.5 && (r.base_lateral_m - player_lat_m).abs() < 0.55 {
                // Shift slightly left (toward inner right lane, still > 0.35m on the right side) while overtaking/being overtaken
                target_lat = (player_lat_m - 0.62).max(0.35);
            }
        }
        r.lateral_m += (target_lat - r.lateral_m) * (dt * 3.5).min(1.0);

        r.wheel_angle_rad = (r.wheel_angle_rad + ds / 0.34).rem_euclid(std::f32::consts::TAU);
        let cad_rpm = (82.0 + ((idx % 5) as f32 * 2.5) - signed_grade * 1.2).clamp(55.0, 112.0);
        r.crank_angle_rad =
            (r.crank_angle_rad + (cad_rpm / 60.0) * std::f32::consts::TAU * dt).rem_euclid(std::f32::consts::TAU);
        let signed_curv = s.curvature * dir;
        let target_lean = (r.speed_mps * r.speed_mps * signed_curv / 9.81)
            .atan()
            .clamp(-0.30, 0.30)
            * 0.65;
        r.lean_angle_rad += (target_lean - r.lean_angle_rad) * (dt * 5.0).min(1.0);
    }
}

/// Builds posed 3D draw parts for one cyclist (frame, 2 spinning wheels, spinning crank, torso, head, optional backpack, 2-bone IK thighs & calves).
/// Note on glTF Y-up coordinate mapping from Blender (+Y forward, +Z up, +X right):
/// In glTF / wgpu space: forward = -Z, up = +Y, right = +X!
/// So a Blender point `(x, y, z)` becomes `(x, z, -y)` in local rider space.
#[allow(clippy::too_many_arguments)]
pub fn build_rider_draw_parts(
    spline: &RouteSpline,
    distance_m: f32,
    lateral_m: f32,
    direction: f32,
    wheel_angle_rad: f32,
    crank_angle_rad: f32,
    lean_angle_rad: f32,
    time_s: f32,
    pose_mode: RiderPoseMode,
    profile: &RiderProfile,
    base_object_id: u32,
    cyclist_skeleton: Option<&SkeletonInfo>,
    out: &mut Vec<RiderPartDraw>,
) {
    let s = spline.sample(distance_m);
    let world_pos = s.pos + s.right * lateral_m + Vec3::Y * 0.004;

    // Rider basis: local +X = right, local +Y = up, local -Z = forward tangent
    let z_back = (-s.tangent).normalize_or_zero();
    let x_right = s.up.cross(z_back).normalize_or_zero();
    let y_up = z_back.cross(x_right).normalize_or_zero();

    let dir_yaw = if direction < 0.0 {
        std::f32::consts::PI
    } else {
        0.0_f32
    };
    let (yaw_rot, eff_lean_rad, eff_wheel_rad, eff_crank_rad) = match pose_mode {
        RiderPoseMode::Pedaling => (dir_yaw, lean_angle_rad, wheel_angle_rad, crank_angle_rad),
        RiderPoseMode::MenuStopped => (
            // Rotate rider ~138 deg so front wheel points toward bottom-right and right drivetrain side faces the camera
            -2.40_f32,
            -0.085_f32,
            0.0_f32,
            // Negative crank angle puts the right pedal at the top-forward position (knee bent high)
            // while the left leg extends straight down to touch the pavement!
            -1.08_f32,
        ),
    };

    let basis = Mat4::from_cols(
        x_right.extend(0.0),
        y_up.extend(0.0),
        z_back.extend(0.0),
        world_pos.extend(1.0),
    ) * Mat4::from_rotation_y(yaw_rot)
        * Mat4::from_rotation_z(-eff_lean_rad);

    let push_rigged = |out: &mut Vec<RiderPartDraw>,
                       model: ModelId,
                       local: Mat4,
                       id_off: u32,
                       skin_pose_r: [f32; 4],
                       skin_pose_l: [f32; 4],
                       bone_matrices: Option<[Mat4; MAX_CYCLIST_BONES]>| {
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
            bone_matrices,
        });
    };
    let push_skinned = |out: &mut Vec<RiderPartDraw>,
                        model: ModelId,
                        local: Mat4,
                        id_off: u32,
                        skin_pose_r: [f32; 4],
                        skin_pose_l: [f32; 4]| {
        push_rigged(out, model, local, id_off, skin_pose_r, skin_pose_l, None);
    };
    let push = |out: &mut Vec<RiderPartDraw>, model: ModelId, local: Mat4, id_off: u32| {
        push_skinned(out, model, local, id_off, [0.0; 4], [0.0; 4]);
    };

    // Frame & Crank share object_id base
    push(out, ModelId::BikeFrame, Mat4::IDENTITY, 0);

    // Wheels share distinct id +1 (slight pneumatic tire contact-patch compression at y=0.333 anchors tire shadow flush!)
    let wheel_rot = Mat4::from_rotation_x(-eff_wheel_rad);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.333, 0.475)) * wheel_rot, 1);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.333, -0.465)) * wheel_rot, 1);

    // Bottom bracket: Blender (0, -0.070, 0.275) -> wgpu (0, 0.275, 0.070)
    let crank_rot = Mat4::from_rotation_x(-eff_crank_rad);
    push(out, ModelId::BikeCrank, Mat4::from_translation(Vec3::new(0.0, 0.275, 0.070)) * crank_rot, 0);

    // Solve 2-bone analytical leg IK for RiderArmature skinning (l1 = 0.430 thigh, l2 = 0.405 knee-to-low-cut-ankle)
    let hip_y = match pose_mode {
        RiderPoseMode::Pedaling => 0.965_f32,
        RiderPoseMode::MenuStopped => 0.885_f32,
    };
    let hip_center = Vec3::new(0.0, hip_y, 0.235);
    let l1 = 0.430_f32;
    let l2 = 0.405_f32;
    let solve_ik_target = |hip: Vec3, ankle: Vec3, foot_ang: f32| -> [f32; 3] {
        let d_vec = ankle - hip;
        let d = d_vec.length().clamp(0.12, l1 + l2 - 0.002);
        let base_ang = (-d_vec.z).atan2(-d_vec.y);
        let cos_a = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
        let thigh_ang = base_ang + cos_a.acos();
        let knee = hip + Vec3::new(0.0, -l1 * thigh_ang.cos(), -l1 * thigh_ang.sin());
        let calf_vec = ankle - knee;
        let calf_ang = (-calf_vec.z).atan2(-calf_vec.y);
        [thigh_ang, calf_ang, foot_ang]
    };
    let solve_leg_angles = |side: f32, theta_off: f32| -> [f32; 3] {
        let th = eff_crank_rad + theta_off;
        let hip = Vec3::new(side * 0.105, hip_y, 0.235);
        let pedal = Vec3::new(
            side * 0.118,
            0.275 - 0.165 * th.sin(),
            0.070 - 0.165 * th.cos(),
        );
        // Biomechanical cycling ankling:
        // Remember in skinning.wgsl rot_x_vec, positive foot_ang pitches the TOE UP (dorsiflexion) and
        // negative foot_ang pitches the TOE DOWN (plantarflexion). The interior shin-to-foot angle is
        // (90 deg + calf_ang - foot_ang). When the leg is UP (top of stroke), calf_ang is strongly negative
        // (~-45..-55 deg), so foot_ang MUST stay negative (toes dipped down) and strictly bounded by
        // `foot_ang <= calf_ang + 0.38` so the interior ankle angle never collapses narrower than ~68 degrees!
        let base_foot_ang = -0.16 + 0.10 * th.sin() - 0.06 * th.cos();
        let ankle_for_pitch = |f_ang: f32| -> Vec3 {
            let (sf, cf) = f_ang.sin_cos();
            let ball_from_ankle = Vec3::new(
                0.0,
                -0.073 * cf - (-0.085) * sf,
                -0.073 * sf + (-0.085) * cf,
            );
            pedal + Vec3::new(0.0, 0.018, 0.0) - ball_from_ankle
        };
        // 2-pass fixed-point solve so foot_ang respects anatomical shin-to-foot dorsiflexion limit
        let pass1 = solve_ik_target(hip, ankle_for_pitch(base_foot_ang), base_foot_ang);
        let calf_ang_est = pass1[1];
        let clamped_foot_ang = base_foot_ang
            .min(calf_ang_est + 0.36) // keeps shin-to-foot interior angle >= 69 deg even at top-dead-center!
            .max(calf_ang_est - 0.65)
            .clamp(-0.56, 0.05);
        solve_ik_target(hip, ankle_for_pitch(clamped_foot_ang), clamped_foot_ang)
    };
    let ang_r = solve_leg_angles(1.0, 0.0);
    let ang_l = match pose_mode {
        RiderPoseMode::Pedaling => solve_leg_angles(-1.0, std::f32::consts::PI),
        RiderPoseMode::MenuStopped => {
            // Left leg extended straight down past the bottom bracket with shoe planted flat on the ground!
            let hip_l = Vec3::new(-0.105, hip_y, 0.235);
            let ground_ankle_l = Vec3::new(-0.105, 0.073, 0.205);
            solve_ik_target(hip_l, ground_ankle_l, 0.0)
        }
    };

    // Continuous pedaling + idle breathing/sway animation phase for torso, shoulders, elbows, and neck/head!
    let rider_seed = (base_object_id as f32) * 0.37;
    let anim_phase = match pose_mode {
        RiderPoseMode::Pedaling => eff_crank_rad + time_s * 1.8 + rider_seed,
        RiderPoseMode::MenuStopped => time_s * 1.15 + rider_seed,
    };
    let motion_scale = match pose_mode {
        RiderPoseMode::Pedaling => 1.0_f32,
        RiderPoseMode::MenuStopped => 0.45_f32,
    };

    // Compute 16-bone hierarchical skeletal skinning matrices for the SINGLE unified rigged cyclist mesh
    // (head, chin, hair, helmet, neck, torso, bridged arms, sculpted wrists, hands, pelvis, legs, socks & shoes!)
    let head_yaw   = (0.055 * (time_s * 0.85 + rider_seed).sin() + 0.024 * anim_phase.cos() - eff_lean_rad * 0.30) * motion_scale;
    let head_pitch = (0.030 * (anim_phase * 2.0).cos() + 0.018 * (time_s * 1.5 + rider_seed).sin()) * motion_scale;
    let head_roll  = (-0.034 * anim_phase.sin() + 0.014 * (time_s * 1.2 + rider_seed).cos()) * motion_scale;
    let spine_sway = 0.022 * anim_phase.sin() * motion_scale;
    let spine_breath = 0.012 * (anim_phase * 2.0).sin() * motion_scale;

    let pivot_rot = |pivot: Vec3, rot: Mat4| -> Mat4 {
        Mat4::from_translation(pivot) * rot * Mat4::from_translation(-pivot)
    };

    let mut bone_mats = [Mat4::IDENTITY; MAX_CYCLIST_BONES];
    let get_bind = |name: &str, fallback: Vec3| -> (Option<usize>, Vec3) {
        if let Some(skel) = cyclist_skeleton {
            if let Some(i) = skel.bone_index(name) {
                let p = skel.bind_positions.get(i).copied().unwrap_or(fallback);
                return (Some(i), p);
            }
        }
        (None, fallback)
    };
    let set_bone = |mats: &mut [Mat4; MAX_CYCLIST_BONES], idx: Option<usize>, m: Mat4| {
        if let Some(i) = idx {
            if i < MAX_CYCLIST_BONES {
                mats[i] = m;
            }
        }
    };

    let (i_pelvis, _)      = get_bind("Pelvis",     Vec3::ZERO);
    let (i_spine, p_spine) = get_bind("Spine",      Vec3::new(0.0, 0.105, -0.030));
    let (i_neck,  p_neck)  = get_bind("Neck",       Vec3::new(0.0, 0.428, -0.228));
    let (i_head,  p_head)  = get_bind("Head",       Vec3::new(0.0, 0.514, -0.252));
    let (i_th_r,  p_th_r)  = get_bind("Thigh_R",    Vec3::new( 0.102,  0.000,  0.000));
    let (i_ca_r,  p_ca_r)  = get_bind("Calf_R",     Vec3::new( 0.102, -0.2673, -0.3368));
    let (i_fo_r,  p_fo_r)  = get_bind("Foot_R",     Vec3::new( 0.124, -0.6477, -0.1979));
    let (i_th_l,  p_th_l)  = get_bind("Thigh_L",    Vec3::new(-0.102,  0.000,  0.000));
    let (i_ca_l,  p_ca_l)  = get_bind("Calf_L",     Vec3::new(-0.102, -0.2673, -0.3368));
    let (i_fo_l,  p_fo_l)  = get_bind("Foot_L",     Vec3::new(-0.124, -0.6477, -0.1979));
    let (i_ua_r,  p_ua_r)  = get_bind("UpperArm_R", Vec3::new( 0.172,  0.335, -0.195));
    let (i_fa_r,  _p_fa_r) = get_bind("Forearm_R",  Vec3::new( 0.218,  0.082, -0.405));
    let (i_ha_r,  _)       = get_bind("Hand_R",     Vec3::new( 0.198, -0.062, -0.574));
    let (i_ua_l,  p_ua_l)  = get_bind("UpperArm_L", Vec3::new(-0.172,  0.335, -0.195));
    let (i_fa_l,  _p_fa_l) = get_bind("Forearm_L",  Vec3::new(-0.218,  0.082, -0.405));
    let (i_ha_l,  _)       = get_bind("Hand_L",     Vec3::new(-0.198, -0.062, -0.574));

    let m_pelvis = Mat4::IDENTITY;
    let m_spine = pivot_rot(
        p_spine,
        Mat4::from_euler(glam::EulerRot::YXZ, 0.0, spine_breath, -spine_sway),
    );
    let m_neck = m_spine
        * pivot_rot(
            p_neck,
            Mat4::from_euler(
                glam::EulerRot::YXZ,
                head_yaw * 0.42,
                head_pitch * 0.45,
                head_roll * 0.40,
            ),
        );
    let m_head = m_neck
        * pivot_rot(
            p_head,
            Mat4::from_euler(
                glam::EulerRot::YXZ,
                head_yaw * 0.58,
                head_pitch * 0.55,
                head_roll * 0.60,
            ),
        );

    // Legs: exact delta rotations from the seated Blender rest pose (REST_TH = 0.90, REST_CA = -0.35, REST_FO = 0.0)
    let d_th_r = ang_r[0] - 0.90;
    let d_ca_r = ang_r[1] - (-0.35);
    let d_fo_r = ang_r[2];
    let m_th_r = pivot_rot(p_th_r, Mat4::from_rotation_x(d_th_r));
    let m_ca_r = m_th_r * pivot_rot(p_ca_r, Mat4::from_rotation_x(d_ca_r - d_th_r));
    let m_fo_r = m_ca_r * pivot_rot(p_fo_r, Mat4::from_rotation_x(d_fo_r - d_ca_r));

    let d_th_l = ang_l[0] - 0.90;
    let d_ca_l = ang_l[1] - (-0.35);
    let d_fo_l = ang_l[2];
    let m_th_l = pivot_rot(p_th_l, Mat4::from_rotation_x(d_th_l));
    let m_ca_l = m_th_l * pivot_rot(p_ca_l, Mat4::from_rotation_x(d_ca_l - d_th_l));
    let m_fo_l = m_ca_l * pivot_rot(p_fo_l, Mat4::from_rotation_x(d_fo_l - d_ca_l));

    // Arms & Hands: shoulders follow half-spine sway, elbows pump subtly, hands stay 100% locked onto brake hoods!
    let m_ua_r = pivot_rot(
        p_ua_r,
        Mat4::from_euler(glam::EulerRot::YXZ, 0.0, spine_breath * 0.45, -spine_sway * 0.45),
    );
    let m_ua_l = pivot_rot(
        p_ua_l,
        Mat4::from_euler(glam::EulerRot::YXZ, 0.0, spine_breath * 0.45, -spine_sway * 0.45),
    );
    let m_fa_r = Mat4::from_translation(Vec3::new(
        0.005 * anim_phase.sin() * motion_scale,
        -0.004 * anim_phase.cos() * motion_scale,
        0.0,
    ));
    let m_fa_l = Mat4::from_translation(Vec3::new(
        0.005 * anim_phase.sin() * motion_scale,
        0.004 * anim_phase.cos() * motion_scale,
        0.0,
    ));

    set_bone(&mut bone_mats, i_pelvis, m_pelvis);
    set_bone(&mut bone_mats, i_spine, m_spine);
    set_bone(&mut bone_mats, i_neck, m_neck);
    set_bone(&mut bone_mats, i_head, m_head);
    set_bone(&mut bone_mats, i_th_r, m_th_r);
    set_bone(&mut bone_mats, i_ca_r, m_ca_r);
    set_bone(&mut bone_mats, i_fo_r, m_fo_r);
    set_bone(&mut bone_mats, i_th_l, m_th_l);
    set_bone(&mut bone_mats, i_ca_l, m_ca_l);
    set_bone(&mut bone_mats, i_fo_l, m_fo_l);
    set_bone(&mut bone_mats, i_ua_r, m_ua_r);
    set_bone(&mut bone_mats, i_fa_r, m_fa_r);
    set_bone(&mut bone_mats, i_ha_r, Mat4::IDENTITY);
    set_bone(&mut bone_mats, i_ua_l, m_ua_l);
    set_bone(&mut bone_mats, i_fa_l, m_fa_l);
    set_bone(&mut bone_mats, i_ha_l, Mat4::IDENTITY);

    // SINGLE unified rigged cyclist mesh (mode = 8.0 = true 4-bone glTF skeletal skinning!)
    push_rigged(
        out,
        ModelId::RiderTorso,
        Mat4::from_translation(hip_center),
        2,
        [ang_r[0], ang_r[1], ang_r[2], 8.0],
        [ang_l[0], ang_l[1], ang_l[2], anim_phase],
        Some(bone_mats),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimalSpecies {
    Horse,
    Cow,
    Sheep,
    Mulita,
    RamblaWalker,
}

impl AnimalSpecies {
    pub fn model_id(self, variant: u8) -> ModelId {
        let v = variant % 4;
        match self {
            AnimalSpecies::Horse => ModelId::Horse(v),
            AnimalSpecies::Cow => ModelId::Cow(v),
            AnimalSpecies::Sheep => ModelId::Sheep(v),
            AnimalSpecies::Mulita => ModelId::Mulita(v),
            AnimalSpecies::RamblaWalker => ModelId::RamblaPedestrian(v),
        }
    }

    pub fn skin_mode(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 3.0,
            AnimalSpecies::Cow => 4.0,
            AnimalSpecies::Sheep => 5.0,
            AnimalSpecies::Mulita => 6.0,
            AnimalSpecies::RamblaWalker => 7.0,
        }
    }

    pub fn walk_speed_mps(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.42,
            AnimalSpecies::Cow => 0.84,
            AnimalSpecies::Sheep => 0.78,
            AnimalSpecies::Mulita => 0.56,
            AnimalSpecies::RamblaWalker => 1.26,
        }
    }

    pub fn stride_length_m(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.55,
            AnimalSpecies::Cow => 1.22,
            AnimalSpecies::Sheep => 0.78,
            AnimalSpecies::Mulita => 0.32,
            AnimalSpecies::RamblaWalker => 1.32,
        }
    }

    pub fn eat_pitch_max_rad(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.24,
            AnimalSpecies::Cow => 1.04,
            AnimalSpecies::Sheep => 1.04,
            AnimalSpecies::Mulita => 0.36,
            AnimalSpecies::RamblaWalker => 0.12,
        }
    }

    pub fn leg_swing_amp_rad(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 0.48,
            AnimalSpecies::Cow => 0.38,
            AnimalSpecies::Sheep => 0.45,
            AnimalSpecies::Mulita => 0.52,
            AnimalSpecies::RamblaWalker => 0.38,
        }
    }

    pub fn wander_radius_m(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 9.5,
            AnimalSpecies::Cow => 7.5,
            AnimalSpecies::Sheep => 6.5,
            AnimalSpecies::Mulita => 4.2,
            AnimalSpecies::RamblaWalker => 240.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimalBehavior {
    Walking,
    EatingGrass,
    Idle,
}

#[derive(Debug, Clone)]
pub struct DynamicAnimal {
    pub species: AnimalSpecies,
    pub variant: u8,
    pub home_dist_m: f32,
    pub home_lat_m: f32,
    pub dist_m: f32,
    pub lat_m: f32,
    pub rel_yaw: f32,
    pub target_rel_yaw: f32,
    pub scale: f32,
    pub behavior: AnimalBehavior,
    pub state_timer_s: f32,
    pub walk_phase: f32,
    pub walk_weight: f32,
    pub eat_weight: f32,
    pub anim_time: f32,
    pub phase_seed: f32,
}

fn animal_hash(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn wrap_angle_pi(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

pub fn spawn_biome_animals(biome: BiomeId, spline: &RouteSpline) -> Vec<DynamicAnimal> {
    if biome != BiomeId::RollingRuralGravel && biome != BiomeId::CoastalPromenade {
        return Vec::new();
    }

    let total = spline.total_length_m;
    let hero_d = spline.hero_start_dist_m;
    let road_hw = biome.road_half_width();
    let mut animals = Vec::with_capacity(140);

    let mut push_animal = |species: AnimalSpecies,
                           variant: u8,
                           dist_m: f32,
                           lat_m: f32,
                           rel_yaw: f32,
                           scale: f32,
                           behavior: AnimalBehavior,
                           seed_idx: u32| {
        let d = dist_m.rem_euclid(total);
        let h0 = animal_hash(seed_idx * 97 + 13);
        let h1 = animal_hash(seed_idx * 151 + 29);
        let (walk_weight, eat_weight) = match behavior {
            AnimalBehavior::Walking => (1.0, 0.0),
            AnimalBehavior::EatingGrass => (0.0, 1.0),
            AnimalBehavior::Idle => (0.0, 0.0),
        };
        animals.push(DynamicAnimal {
            species,
            variant: variant % 4,
            home_dist_m: d,
            home_lat_m: lat_m,
            dist_m: d,
            lat_m,
            rel_yaw: wrap_angle_pi(rel_yaw),
            target_rel_yaw: wrap_angle_pi(rel_yaw),
            scale,
            behavior,
            state_timer_s: 2.8 + h0 * 5.2,
            walk_phase: h1 * std::f32::consts::TAU,
            walk_weight,
            eat_weight,
            anim_time: h0 * 12.0,
            phase_seed: h1 * 19.0,
        });
    };

    if biome == BiomeId::CoastalPromenade {
        // 1. Curated foreground/midground Rambla pedestrians (some walking & drinking mate with termo + bombilla!)
        let pi = std::f32::consts::PI;
        let hero_walkers: &[(f32, f32, u8, f32, f32, AnimalBehavior)] = &[
            // Foreground mate-drinker strolling along the right tiled promenade
            (11.5, road_hw + 2.85, 1, 1.00, 0.0, AnimalBehavior::Walking),
            // Pair of walkers strolling together (one drinking mate, one in casual coastal attire) coming toward camera
            (22.0, road_hw + 3.65, 3, 0.99, pi, AnimalBehavior::Walking),
            (22.6, road_hw + 4.35, 0, 1.01, pi, AnimalBehavior::Walking),
            // Walker near the seawall admiring the ocean & sipping mate
            (31.0, road_hw + 5.20, 1, 1.00, 0.0, AnimalBehavior::Walking),
            // Jogger/walker in sun cap further along the promenade
            (44.5, road_hw + 2.65, 2, 1.00, 0.0, AnimalBehavior::Walking),
            (58.0, road_hw + 4.10, 3, 0.98, pi, AnimalBehavior::Walking),
            // Pedestrians walking on the left building sidewalk under the trees
            (19.0, -(road_hw + 3.40), 0, 1.00, pi, AnimalBehavior::Walking),
            (37.5, -(road_hw + 3.15), 1, 1.00, 0.0, AnimalBehavior::Walking),
        ];
        for (i, &(d_off, lat_m, var, sc, yaw, beh)) in hero_walkers.iter().enumerate() {
            push_animal(
                AnimalSpecies::RamblaWalker,
                var,
                hero_d + d_off,
                lat_m,
                yaw,
                sc,
                beh,
                5000 + i as u32,
            );
        }

        // 2. Continuous Rambla promenade pedestrians around the 1.6 km coastal loop (solo walkers & pairs, ~50% drinking mate)
        let mut ped_d = 75.0_f32;
        let mut p_idx = 0_u32;
        while ped_d < total - 35.0 {
            let rel_hero = (ped_d - hero_d).rem_euclid(total);
            if !(0.0..68.0).contains(&rel_hero) {
                let ps = p_idx.wrapping_mul(137).wrapping_add(31);
                let r0 = animal_hash(ps);
                let r1 = animal_hash(ps + 17);
                let dir_yaw = if r0 > 0.48 { 0.0_f32 } else { pi };
                let on_right_promenade = p_idx % 5 != 4;
                let base_lat = if on_right_promenade {
                    road_hw + 2.25 + r1 * 3.15
                } else {
                    -(road_hw + 2.60 + r1 * 1.45)
                };
                let var = (p_idx % 4) as u8;
                push_animal(
                    AnimalSpecies::RamblaWalker,
                    var,
                    ped_d,
                    base_lat,
                    dir_yaw,
                    0.97 + animal_hash(ps + 29) * 0.07,
                    AnimalBehavior::Walking,
                    6000 + p_idx * 2,
                );
                // Every 3rd encounter is a pair of friends walking side-by-side (one with mate!)
                if on_right_promenade && p_idx % 3 == 0 {
                    let pair_lat = (base_lat + 0.72).min(road_hw + 5.55);
                    push_animal(
                        AnimalSpecies::RamblaWalker,
                        ((p_idx + 1) % 4) as u8,
                        ped_d + 0.45,
                        pair_lat,
                        dir_yaw,
                        0.98 + animal_hash(ps + 43) * 0.06,
                        AnimalBehavior::Walking,
                        6001 + p_idx * 2,
                    );
                }
            }
            ped_d += 24.0 + animal_hash(p_idx * 89 + 11) * 18.0;
            p_idx += 1;
        }

        return animals;
    }

    // 1. Curated Hero Vista Animals at hero_start_dist_m:
    // Small, realistic Campo groups (a pair of 2 horses in the mid-meadow, a small mixed group of 2 cows + 1 horse
    // on the distant pasture slope, 1 horse in the estancia paddock, and 1 mulita near the foreground grass).
    let hero_specs: &[(f32, f32, AnimalSpecies, u8, f32, f32, AnimalBehavior)] = &[
        // 1 foreground Mulita foraging by the left grass patch
        (8.4, -(road_hw + 1.05), AnimalSpecies::Mulita, 0, 1.22, 0.45, AnimalBehavior::Walking),
        // Pair of 2 horses in the open left meadow
        (34.5, -(road_hw + 5.6), AnimalSpecies::Horse, 0, 1.08, 1.25, AnimalBehavior::EatingGrass),
        (38.0, -(road_hw + 4.2), AnimalSpecies::Horse, 1, 1.06, 0.45, AnimalBehavior::Walking),
        // 1 horse grazing in the right estancia eucalyptus paddock
        (27.5, road_hw + 8.2, AnimalSpecies::Horse, 3, 1.08, -1.20, AnimalBehavior::EatingGrass),
        // Small mixed group of 2 cows and 1 horse on the distant left pasture hillside
        (56.0, -(road_hw + 11.8), AnimalSpecies::Cow, 0, 1.10, 1.10, AnimalBehavior::EatingGrass),
        (59.5, -(road_hw + 14.2), AnimalSpecies::Cow, 1, 1.08, 1.55, AnimalBehavior::Idle),
        (62.0, -(road_hw + 10.6), AnimalSpecies::Horse, 2, 1.06, 0.55, AnimalBehavior::EatingGrass),
    ];

    for (idx, &(d_off, lat_m, species, variant, sc, rel_yaw, behavior)) in
        hero_specs.iter().enumerate()
    {
        push_animal(
            species,
            variant,
            hero_d + d_off,
            lat_m,
            rel_yaw,
            sc,
            behavior,
            idx as u32 + 1,
        );
    }

    // 2. Rare, small natural groups across the rest of the 1.6 km Campo loop (every 155..245 m):
    // - Group 0: Pair of 2 horses
    // - Group 1: Mixed group of cows and horses (2 cows + 1 horse)
    // - Group 2: Small flock of 3 sheep
    // - Group 3: Pair of 2 cows + 1 nearby foraging mulita
    let mut herd_d = 175.0_f32;
    let mut h_idx = 0_u32;
    while herd_d < total - 95.0 {
        let rel_hero = (herd_d - hero_d).rem_euclid(total);
        if !(0.0..115.0).contains(&rel_hero) && rel_hero < total - 75.0 {
            let hs = h_idx.wrapping_mul(173).wrapping_add(19);
            let r0 = animal_hash(hs);
            let side = if h_idx % 2 == 0 { -1.0_f32 } else { 1.0_f32 };
            let base_lat = side * (road_hw + 7.0 + r0 * 22.0);
            let group_kind = h_idx % 4;

            let group_members: &[(AnimalSpecies, f32, f32)] = match group_kind {
                0 => &[
                    // Pair of 2 horses
                    (AnimalSpecies::Horse, -1.8, -1.4),
                    (AnimalSpecies::Horse, 2.1, 1.6),
                ],
                1 => &[
                    // Mixed group of 2 cows and 1 horse
                    (AnimalSpecies::Cow, -2.5, -1.8),
                    (AnimalSpecies::Cow, 1.4, 2.2),
                    (AnimalSpecies::Horse, 3.6, -1.2),
                ],
                2 => &[
                    // Small group of 3 sheep
                    (AnimalSpecies::Sheep, -1.6, -1.2),
                    (AnimalSpecies::Sheep, 0.8, 1.5),
                    (AnimalSpecies::Sheep, 2.2, -0.6),
                ],
                _ => &[
                    // Pair of 2 cows + 1 mulita near the roadside grass
                    (AnimalSpecies::Cow, -1.9, 0.8),
                    (AnimalSpecies::Cow, 2.4, -1.5),
                    (AnimalSpecies::Mulita, 0.0, -5.0 * side),
                ],
            };

            for (m_i, &(species, off_d, off_lat)) in group_members.iter().enumerate() {
                let ms = hs + (m_i as u32) * 29;
                let md = (herd_d + off_d + (animal_hash(ms + 1) - 0.5) * 2.5).rem_euclid(total);
                let raw_lat = base_lat + off_lat + (animal_hash(ms + 2) - 0.5) * 2.0;
                let min_l = if species == AnimalSpecies::Mulita {
                    road_hw + 1.15
                } else {
                    road_hw + 3.2
                };
                let safe_lat = if side < 0.0 {
                    raw_lat.min(-min_l)
                } else {
                    raw_lat.max(min_l)
                };
                let v = ((m_i as u32 + h_idx) % 4) as u8;
                let sc = 0.98 + animal_hash(ms + 3) * 0.14;
                let yaw = (animal_hash(ms + 4) - 0.5) * std::f32::consts::TAU;
                let beh = match (h_idx + m_i as u32) % 3 {
                    0 => AnimalBehavior::EatingGrass,
                    1 => AnimalBehavior::Walking,
                    _ => AnimalBehavior::Idle,
                };
                push_animal(species, v, md, safe_lat, yaw, sc, beh, 2000 + ms);
            }
        }
        let r_step = animal_hash(h_idx * 97 + 41);
        herd_d += 155.0 + r_step * 85.0;
        h_idx += 1;
    }

    animals
}

pub fn step_animals(
    dt: f32,
    spline: &RouteSpline,
    biome: BiomeId,
    animals: &mut [DynamicAnimal],
) {
    if animals.is_empty() {
        return;
    }
    let dt = dt.clamp(0.0, 0.12);
    let total = spline.total_length_m;
    let road_hw = biome.road_half_width();
    let min_safe_lat = road_hw + 1.05;

    for (idx, a) in animals.iter_mut().enumerate() {
        a.anim_time += dt;
        a.state_timer_s -= dt;

        if a.state_timer_s <= 0.0 {
            let seed = ((a.anim_time * 37.0) as u32)
                .wrapping_add((idx as u32).wrapping_mul(2654435761));
            let r_state = animal_hash(seed);
            let r_dur = animal_hash(seed.wrapping_add(17));
            let r_turn = animal_hash(seed.wrapping_add(53));

            if a.species == AnimalSpecies::RamblaWalker {
                // Rambla pedestrians walk along the promenade (85% of the time) and occasionally pause to sip mate / look at the sea
                a.behavior = if r_state < 0.85 {
                    AnimalBehavior::Walking
                } else {
                    AnimalBehavior::EatingGrass
                };
                a.state_timer_s = 5.0 + r_dur * 7.5;
                // Keep heading parallel to the promenade lane (either ~0.0 forward or ~PI oncoming)
                let base_dir = if a.rel_yaw.cos() >= 0.0 {
                    0.0_f32
                } else {
                    std::f32::consts::PI
                };
                a.target_rel_yaw = wrap_angle_pi(base_dir + (r_turn - 0.5) * 0.08);
            } else {
                a.behavior = match a.behavior {
                    AnimalBehavior::Walking => {
                        if r_state < 0.62 {
                            AnimalBehavior::EatingGrass
                        } else {
                            AnimalBehavior::Idle
                        }
                    }
                    AnimalBehavior::EatingGrass => {
                        if r_state < 0.52 {
                            AnimalBehavior::Walking
                        } else {
                            AnimalBehavior::Idle
                        }
                    }
                    AnimalBehavior::Idle => {
                        if r_state < 0.55 {
                            AnimalBehavior::EatingGrass
                        } else {
                            AnimalBehavior::Walking
                        }
                    }
                };
                a.state_timer_s = match a.behavior {
                    AnimalBehavior::Walking => 3.6 + r_dur * 4.8,
                    AnimalBehavior::EatingGrass => 4.2 + r_dur * 5.8,
                    AnimalBehavior::Idle => 2.6 + r_dur * 3.8,
                };

                // Compute displacement from home anchor so animals stay within their pasture/paddock
                let mut d_diff = (a.home_dist_m - a.dist_m).rem_euclid(total);
                if d_diff > 0.5 * total {
                    d_diff -= total;
                }
                let lat_diff = a.home_lat_m - a.lat_m;
                let wander_dist = (d_diff * d_diff + lat_diff * lat_diff).sqrt();
                if wander_dist > a.species.wander_radius_m() * 0.65 {
                    let to_home_yaw = lat_diff.atan2(d_diff);
                    a.target_rel_yaw = wrap_angle_pi(to_home_yaw + (r_turn - 0.5) * 0.85);
                } else if a.behavior == AnimalBehavior::Walking {
                    a.target_rel_yaw = wrap_angle_pi(a.rel_yaw + (r_turn - 0.5) * 1.9);
                } else {
                    a.target_rel_yaw = wrap_angle_pi(a.rel_yaw + (r_turn - 0.5) * 0.55);
                }
            }
        }

        // Smoothly blend animation weights between Walking, EatingGrass, and Idle
        let target_walk = if a.behavior == AnimalBehavior::Walking {
            1.0_f32
        } else {
            0.0_f32
        };
        let target_eat = if a.behavior == AnimalBehavior::EatingGrass {
            1.0_f32
        } else {
            0.0_f32
        };
        a.walk_weight += (target_walk - a.walk_weight) * (dt * 3.2).min(1.0);
        a.eat_weight += (target_eat - a.eat_weight) * (dt * 2.8).min(1.0);

        // Smoothly steer heading toward target_rel_yaw
        let yaw_err = wrap_angle_pi(a.target_rel_yaw - a.rel_yaw);
        let turn_rate = if a.walk_weight > 0.2 { 2.2 } else { 1.1 };
        a.rel_yaw = wrap_angle_pi(a.rel_yaw + yaw_err * (dt * turn_rate).min(1.0));

        // Advance world position & leg walk cycle phase when walking
        if a.walk_weight > 0.015 {
            let speed = a.species.walk_speed_mps() * a.walk_weight;
            let ds = speed * dt;
            let v_dist = ds * a.rel_yaw.cos();
            let v_lat = ds * a.rel_yaw.sin();

            a.dist_m = (a.dist_m + v_dist).rem_euclid(total);
            let next_lat = a.lat_m + v_lat;
            let home_side = if a.home_lat_m >= 0.0 { 1.0_f32 } else { -1.0_f32 };

            // Keep animal/walker on its side of the road and within safe lateral lane bounds
            if a.species == AnimalSpecies::RamblaWalker {
                if home_side > 0.0 {
                    // Right Rambla promenade walkway: stay clear of street-edge benches (road_hw + 0.98) and seawall (road_hw + 6.5)
                    a.lat_m = next_lat.clamp(road_hw + 2.05, road_hw + 5.65);
                } else {
                    // Left sidewalk walkway: stay between street curb and building facades
                    a.lat_m = next_lat.clamp(-(road_hw + 4.45), -(road_hw + 2.25));
                }
            } else if home_side > 0.0 {
                if next_lat < min_safe_lat {
                    a.lat_m = min_safe_lat;
                    a.target_rel_yaw = wrap_angle_pi(std::f32::consts::FRAC_PI_4);
                } else if next_lat > road_hw + 76.0 {
                    a.lat_m = road_hw + 76.0;
                    a.target_rel_yaw = wrap_angle_pi(-std::f32::consts::FRAC_PI_4);
                } else {
                    a.lat_m = next_lat;
                }
            } else if next_lat > -min_safe_lat {
                a.lat_m = -min_safe_lat;
                a.target_rel_yaw = wrap_angle_pi(-std::f32::consts::FRAC_PI_4);
            } else if next_lat < -(road_hw + 76.0) {
                a.lat_m = -(road_hw + 76.0);
                a.target_rel_yaw = wrap_angle_pi(std::f32::consts::FRAC_PI_4);
            } else {
                a.lat_m = next_lat;
            }

            let stride = a.species.stride_length_m().max(0.2);
            a.walk_phase =
                (a.walk_phase + (ds / stride) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
        }
    }
}

pub fn build_animal_draw_parts(
    biome: BiomeId,
    spline: &RouteSpline,
    animals: &[DynamicAnimal],
    cam_pos: Vec3,
    out: &mut Vec<RiderPartDraw>,
) {
    if animals.is_empty() {
        return;
    }
    let road_hw = biome.road_half_width();
    let total = spline.total_length_m;
    let max_dist_sq = 185.0_f32 * 185.0_f32;

    for (idx, a) in animals.iter().enumerate() {
        let s_c = spline.sample(a.dist_m);
        let y_c = terrain_offset_y(biome, a.lat_m, a.dist_m, road_hw);
        let pos_c = s_c.pos + s_c.right * a.lat_m + Vec3::Y * (y_c - 0.025);
        if pos_c.distance_squared(cam_pos) > max_dist_sq {
            continue;
        }

        // Sample terrain 0.55m ahead along the animal's heading so hooves align with hillside slope
        let step_ahead = 0.55_f32;
        let d_ahead = (a.dist_m + step_ahead * a.rel_yaw.cos()).rem_euclid(total);
        let lat_ahead = a.lat_m + step_ahead * a.rel_yaw.sin();
        let s_f = spline.sample(d_ahead);
        let y_f = terrain_offset_y(biome, lat_ahead, d_ahead, road_hw);
        let pos_f = s_f.pos + s_f.right * lat_ahead + Vec3::Y * (y_f - 0.025);

        let fwd_3d = (pos_f - pos_c).normalize_or_zero();
        let right_3d = Vec3::Y.cross(fwd_3d).normalize_or_zero();
        let up_3d = fwd_3d.cross(right_3d).normalize_or_zero();

        let transform = Mat4::from_cols(
            right_3d.extend(0.0),
            up_3d.extend(0.0),
            fwd_3d.extend(0.0),
            pos_c.extend(1.0),
        ) * Mat4::from_scale(Vec3::splat(a.scale));

        let t = a.anim_time + a.phase_seed;
        let idle_head_yaw = 0.24 * (t * 0.85).sin() + 0.11 * (t * 1.9).cos();

        out.push(RiderPartDraw {
            model: a.species.model_id(a.variant),
            transform,
            jersey_primary: [1.0; 4],
            jersey_secondary: [1.0; 4],
            jersey_accent: [1.0; 4],
            helmet_color: [1.0; 4],
            helmet_accent: [1.0; 4],
            frame_color: [1.0; 4],
            object_id: 80 + ((idx as u32) % 96),
            skin_pose_r: [
                a.walk_phase,
                a.walk_weight,
                a.eat_weight,
                a.species.skin_mode(),
            ],
            skin_pose_l: [
                t,
                idle_head_yaw,
                a.species.eat_pitch_max_rad(),
                a.species.leg_swing_amp_rad(),
            ],
            bone_matrices: None,
        });
    }
}

#[allow(dead_code)]
fn _keep_quat(_: Quat) {}
