use glam::{Mat4, Quat, Vec3};

use crate::assets::ModelId;
use crate::world::terrain::terrain_offset_y;
use crate::world::{BiomeId, RouteSpline};

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
    pose_mode: RiderPoseMode,
    profile: &RiderProfile,
    base_object_id: u32,
    out: &mut Vec<RiderPartDraw>,
) {
    let s = spline.sample(distance_m);
    let world_pos = s.pos + s.right * lateral_m + Vec3::Y * 0.004;

    // Rider basis: local +X = right, local +Y = up, local -Z = forward tangent
    let z_back = (-s.tangent).normalize_or_zero();
    let x_right = s.up.cross(z_back).normalize_or_zero();
    let y_up = z_back.cross(x_right).normalize_or_zero();

    let (yaw_rot, eff_lean_rad, eff_wheel_rad, eff_crank_rad) = match pose_mode {
        RiderPoseMode::Pedaling => (0.0_f32, lean_angle_rad, wheel_angle_rad, crank_angle_rad),
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

    // Wheels share distinct id +1 (slight pneumatic tire contact-patch compression at y=0.333 anchors tire shadow flush!)
    let wheel_rot = Mat4::from_rotation_x(-eff_wheel_rad);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.333, 0.475)) * wheel_rot, 1);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.333, -0.465)) * wheel_rot, 1);

    // Bottom bracket: Blender (0, -0.070, 0.275) -> wgpu (0, 0.275, 0.070)
    let crank_rot = Mat4::from_rotation_x(-eff_crank_rad);
    push(out, ModelId::BikeCrank, Mat4::from_translation(Vec3::new(0.0, 0.275, 0.070)) * crank_rot, 0);

    // Solve 2-bone analytical leg IK for RiderArmature skinning (l1 = 0.430, l2 = 0.415)
    let hip_y = match pose_mode {
        RiderPoseMode::Pedaling => 0.965_f32,
        RiderPoseMode::MenuStopped => 0.885_f32,
    };
    let hip_center = Vec3::new(0.0, hip_y, 0.235);
    let l1 = 0.430_f32;
    let l2 = 0.415_f32;
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
        let ankle = pedal + Vec3::new(0.0, 0.056, 0.035);
        let foot_ang = 0.10 * th.cos();
        solve_ik_target(hip, ankle, foot_ang)
    };
    let ang_r = solve_leg_angles(1.0, 0.0);
    let ang_l = match pose_mode {
        RiderPoseMode::Pedaling => solve_leg_angles(-1.0, std::f32::consts::PI),
        RiderPoseMode::MenuStopped => {
            // Left leg extended straight down past the bottom bracket with shoe planted flat on the ground!
            let hip_l = Vec3::new(-0.105, hip_y, 0.235);
            let ground_ankle_l = Vec3::new(-0.105, 0.054, 0.205);
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
    let head_yaw    = (0.055 * (time_s * 0.85 + rider_seed).sin() + 0.024 * anim_phase.cos() - eff_lean_rad * 0.30) * motion_scale;
    let head_pitch  = (0.032 * (anim_phase * 2.0).cos() + 0.019 * (time_s * 1.5 + rider_seed).sin()) * motion_scale;
    let head_roll   = (-0.036 * anim_phase.sin() + 0.015 * (time_s * 1.2 + rider_seed).cos()) * motion_scale;

    let head_xform = Mat4::from_translation(Vec3::new(
        head_sway_x,
        (hip_y + 0.490) + head_bob_y,
        -0.015 + head_fwd_z,
    )) * Mat4::from_euler(glam::EulerRot::YXZ, head_yaw, head_pitch, head_roll);

    push(
        out,
        ModelId::RiderHeadHelmet,
        head_xform,
        4,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimalSpecies {
    Horse,
    Cow,
    Sheep,
    Mulita,
}

impl AnimalSpecies {
    pub fn model_id(self, variant: u8) -> ModelId {
        let v = variant % 4;
        match self {
            AnimalSpecies::Horse => ModelId::Horse(v),
            AnimalSpecies::Cow => ModelId::Cow(v),
            AnimalSpecies::Sheep => ModelId::Sheep(v),
            AnimalSpecies::Mulita => ModelId::Mulita(v),
        }
    }

    pub fn skin_mode(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 3.0,
            AnimalSpecies::Cow => 4.0,
            AnimalSpecies::Sheep => 5.0,
            AnimalSpecies::Mulita => 6.0,
        }
    }

    pub fn walk_speed_mps(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.42,
            AnimalSpecies::Cow => 0.84,
            AnimalSpecies::Sheep => 0.78,
            AnimalSpecies::Mulita => 0.56,
        }
    }

    pub fn stride_length_m(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.55,
            AnimalSpecies::Cow => 1.22,
            AnimalSpecies::Sheep => 0.78,
            AnimalSpecies::Mulita => 0.32,
        }
    }

    pub fn eat_pitch_max_rad(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 1.24,
            AnimalSpecies::Cow => 1.04,
            AnimalSpecies::Sheep => 1.04,
            AnimalSpecies::Mulita => 0.36,
        }
    }

    pub fn leg_swing_amp_rad(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 0.48,
            AnimalSpecies::Cow => 0.38,
            AnimalSpecies::Sheep => 0.45,
            AnimalSpecies::Mulita => 0.52,
        }
    }

    pub fn wander_radius_m(self) -> f32 {
        match self {
            AnimalSpecies::Horse => 9.5,
            AnimalSpecies::Cow => 7.5,
            AnimalSpecies::Sheep => 6.5,
            AnimalSpecies::Mulita => 4.2,
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
    if biome != BiomeId::RollingRuralGravel {
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

            // Keep animal on its side of the road/fence and within outer terrain bounds
            if home_side > 0.0 {
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
        });
    }
}

#[allow(dead_code)]
fn _keep_quat(_: Quat) {}
