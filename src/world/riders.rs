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
    pub helmet_color: [f32; 4],
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
    pub helmet_color: [f32; 4],
    pub frame_color: [f32; 4],
    pub object_id: u32,
}

pub fn default_player_profile() -> RiderProfile {
    RiderProfile {
        name: "S. Miller",
        level: 12,
        w_per_kg: 2.8,
        jersey_primary: [0.545, 0.416, 0.525, 1.0],
        jersey_secondary: [0.357, 0.635, 0.608, 1.0],
        helmet_color: [0.20, 0.22, 0.26, 1.0],
        frame_color: [0.298, 0.459, 0.510, 1.0],
        has_backpack: true,
        avatar_skin: [216, 166, 138],
        avatar_hair: [64, 52, 46],
        avatar_shirt: [92, 162, 155],
        is_player: true,
    }
}

pub fn spawn_ai_peloton(start_dist_m: f32, route_len_m: f32) -> Vec<AiRider> {
    let specs: &[(&str, f32, f32, f32, [f32; 4], [f32; 4], [f32; 4], [f32; 4], [u8; 3], [u8; 3], [u8; 3])] = &[
        // Left-ahead AI rider (matches O. Hansen in Reference Images: dark teal patterned kit, dark helmet)
        ("O. Hansen", 6.8, -1.58, 2.5, [0.10, 0.22, 0.21, 1.0], [0.24, 0.58, 0.48, 1.0], [0.18, 0.20, 0.23, 1.0], [0.26, 0.32, 0.38, 1.0], [218, 174, 146], [198, 118, 72], [58, 118, 108]),
        // Right-ahead AI rider (matches H. Dupont in Reference Images: seafoam turquoise jersey, crisp white helmet)
        ("H. Dupont", 5.2, 1.48, 2.5, [0.34, 0.68, 0.65, 1.0], [0.46, 0.76, 0.72, 1.0], [0.95, 0.95, 0.93, 1.0], [0.38, 0.56, 0.62, 1.0], [224, 178, 152], [110, 74, 56], [92, 164, 155]),
        ("A. Lsmith", -16.0, -0.8, 2.5, [0.68, 0.42, 0.36, 1.0], [0.82, 0.62, 0.44, 1.0], [0.22, 0.24, 0.28, 1.0], [0.45, 0.35, 0.32, 1.0], [164, 122, 98], [48, 40, 38], [175, 108, 92]),
        ("O. Hansen ", -20.0, 0.9, 2.5, [0.45, 0.52, 0.36, 1.0], [0.62, 0.68, 0.48, 1.0], [0.85, 0.83, 0.78, 1.0], [0.32, 0.40, 0.36, 1.0], [212, 168, 140], [148, 116, 84], [116, 134, 92]),
        ("M. Dupont", -24.0, -1.1, 2.5, [0.34, 0.44, 0.62, 1.0], [0.54, 0.66, 0.80, 1.0], [0.26, 0.28, 0.32, 1.0], [0.28, 0.36, 0.48, 1.0], [222, 178, 154], [78, 54, 46], [88, 112, 158]),
        ("L. Moreau", -32.0, 0.6, 2.4, [0.74, 0.52, 0.32, 1.0], [0.88, 0.70, 0.46, 1.0], [0.84, 0.84, 0.82, 1.0], [0.52, 0.36, 0.28, 1.0], [208, 160, 132], [58, 44, 40], [188, 132, 82]),
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
                helmet_color: s.6,
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
            crank_angle_rad: (i as f32) * 1.15,
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

    let push = |out: &mut Vec<RiderPartDraw>, model: ModelId, local: Mat4, id_off: u32| {
        out.push(RiderPartDraw {
            model,
            transform: basis * local,
            jersey_primary: profile.jersey_primary,
            jersey_secondary: profile.jersey_secondary,
            helmet_color: profile.helmet_color,
            frame_color: profile.frame_color,
            object_id: base_object_id + id_off,
        });
    };

    // Frame & Crank share object_id base
    push(out, ModelId::BikeFrame, Mat4::IDENTITY, 0);

    // Wheels share distinct id +1
    let wheel_rot = Mat4::from_rotation_x(-wheel_angle_rad);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.34, 0.51)) * wheel_rot, 1);
    push(out, ModelId::BikeWheel, Mat4::from_translation(Vec3::new(0.0, 0.34, -0.51)) * wheel_rot, 1);

    let crank_rot = Mat4::from_rotation_x(-crank_angle_rad);
    push(out, ModelId::BikeCrank, Mat4::from_translation(Vec3::new(0.0, 0.28, 0.09)) * crank_rot, 0);

    // Rider body parts (torso, head, thighs, calves) share id +2 so hip joints stay seamless (no ring seam!)
    let hip_center = Vec3::new(0.0, 0.99, 0.25);
    push(out, ModelId::RiderTorso, Mat4::from_translation(hip_center), 2);
    if profile.has_backpack {
        push(out, ModelId::RiderBackpack, Mat4::from_translation(hip_center), 3);
    }

    push(
        out,
        ModelId::RiderHeadHelmet,
        Mat4::from_translation(Vec3::new(0.0, 1.39, -0.15)),
        2,
    );

    // 2-Bone Analytical Leg IK in wgpu local coords (thigh points -Y from hip, calf points -Y from knee)
    // Remember Blender crank at crank_angle_rad = 0 had right pedal at +Y = +0.172 (wgpu -Z = -0.172).
    // Rotating around +X by -crank_angle_rad maps (0, 0, -0.172) -> (0, -0.172*sin(a), -0.172*cos(a))?
    // Wait: in crank_rot = from_rotation_x(-theta), a vector (0, 0, -R) transforms to:
    // y' = y*cos(-th) - z*sin(-th) = +z*(-sin(-th)) = -R * sin(th)? Let's check glam rotation_x(alpha):
    // y' = y*cos(a) - z*sin(a), z' = y*sin(a) + z*cos(a).
    // With a = -crank_angle_rad and (y=0, z=-R):
    //   y' = -(-R)*sin(-theta) = -R*sin(theta) -> wait: pedal goes DOWN when theta > 0! Let's check!
    //   If pedal rotates forward-then-down (clockwise seen from right +X): top is +Y (+Z_bl), forward is -Z (+Y_bl).
    //   Rotating (0, 0, -R) clockwise from right (+X) is NEGATIVE angle around +X (right-hand thumb along +X curls +Y->+Z->-Y->-Z, so clockwise is -X? Wait: +X thumb points right, fingers curl +Y to +Z (back) to -Y (down) to -Z (front) -> +X rotation moves forward (-Z) UP (+Y), and -X rotation moves forward (-Z) DOWN (-Y)!).
    //   Let's verify: with rotation_x(alpha), for (0, 0, -R):
    //   y' = -(-R)*sin(alpha) = R*sin(alpha). If alpha = -theta (theta increasing 0..pi/2): y' = -R*sin(theta) (goes DOWN!), z' = -R*cos(theta) (stays forward-to-bottom!).
    //   YES! Forward-then-down pedaling IS rotation_x(-theta), giving exact pedal position:
    //   pedal_y = 0.28 - 0.172 * sin(theta_side)
    //   pedal_z = 0.09 - 0.172 * cos(theta_side)
    let l1 = 0.44_f32;
    let l2 = 0.43_f32;
    for (side, theta_off) in [(1.0_f32, 0.0_f32), (-1.0_f32, std::f32::consts::PI)].into_iter() {
        let th = crank_angle_rad + theta_off;
        let hip = Vec3::new(side * 0.094, 0.99, 0.25);
        let pedal = Vec3::new(
            side * 0.140,
            0.28 - 0.172 * th.sin(),
            0.09 - 0.172 * th.cos(),
        );
        // Ankle sits slightly behind (+Z) and above (+Y) pedal spindle
        let ankle = pedal + Vec3::new(0.0, 0.055, 0.03);
        let d_vec = ankle - hip;
        let d = d_vec.length().clamp(0.12, l1 + l2 - 0.008);

        // Solve in Y-Z plane (base_ang from -Y toward -Z forward)
        let base_ang = (-d_vec.z).atan2(-d_vec.y);
        let cos_a = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
        let a = cos_a.acos();
        let thigh_ang = base_ang + a; // knee bends forward (-Z)

        // With rotation_x(+ang), vector (0, -L, 0) goes to (0, -L*cos(ang), -L*sin(ang))
        let knee = hip + Vec3::new(0.0, -l1 * thigh_ang.cos(), -l1 * thigh_ang.sin());
        let calf_vec = ankle - knee;
        let calf_ang = (-calf_vec.z).atan2(-calf_vec.y);

        push(
            out,
            ModelId::RiderThigh,
            Mat4::from_translation(hip) * Mat4::from_rotation_x(thigh_ang),
            2,
        );
        push(
            out,
            ModelId::RiderCalfShoe,
            Mat4::from_translation(knee) * Mat4::from_rotation_x(calf_ang),
            2,
        );
    }
}

#[allow(dead_code)]
fn _keep_quat(_: Quat) {}
