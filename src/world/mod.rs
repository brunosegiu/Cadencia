pub mod riders;
pub mod terrain;

use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiomeId {
    MountainClimb,
    CoastalPromenade,
    RollingRuralGravel,
}

impl BiomeId {
    pub const ALL: [BiomeId; 3] = [
        BiomeId::MountainClimb,
        BiomeId::CoastalPromenade,
        BiomeId::RollingRuralGravel,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BiomeId::MountainClimb => "Cerro",
            BiomeId::CoastalPromenade => "Rambla",
            BiomeId::RollingRuralGravel => "Campo",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            BiomeId::MountainClimb => "High Alpine Ridge & Canyon Pass",
            BiomeId::CoastalPromenade => "Sunset Bay Seawall & Colonial Coast",
            BiomeId::RollingRuralGravel => "Olive Hills & Countryside Gravel Loop",
        }
    }

    pub fn stats_summary(self) -> (&'static str, &'static str, &'static str) {
        match self {
            BiomeId::MountainClimb => ("1.8 KM LOOP", "+6.8% MAX GRADE", "ALPINE TARMAC"),
            BiomeId::CoastalPromenade => ("2.2 KM LOOP", "+2.4% MAX GRADE", "COASTAL PAVÉ"),
            BiomeId::RollingRuralGravel => ("1.6 KM LOOP", "+4.5% MAX GRADE", "COMPACT GRAVEL"),
        }
    }

    pub fn random() -> Self {
        let nanos = web_time::SystemTime::now()
            .duration_since(web_time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        #[cfg(not(target_arch = "wasm32"))]
        let seed = nanos ^ ((std::process::id() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        #[cfg(target_arch = "wasm32")]
        let seed = nanos;

        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        Self::ALL[(z as usize) % Self::ALL.len()]
    }

    pub fn next(self) -> Self {
        match self {
            BiomeId::MountainClimb => BiomeId::CoastalPromenade,
            BiomeId::CoastalPromenade => BiomeId::RollingRuralGravel,
            BiomeId::RollingRuralGravel => BiomeId::MountainClimb,
        }
    }

    pub fn from_str_arg(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cerro" | "mountain" | "mountain_climb" | "mountainclimb" | "1" => {
                Some(BiomeId::MountainClimb)
            }
            "rambla" | "coastal" | "coastal_promenade" | "coastalpromenade" | "2" => {
                Some(BiomeId::CoastalPromenade)
            }
            "campo" | "rural" | "rolling_rural_gravel" | "rollingruralgravel" | "gravel" | "3" => {
                Some(BiomeId::RollingRuralGravel)
            }
            _ => None,
        }
    }

    pub fn road_half_width(self) -> f32 {
        match self {
            BiomeId::MountainClimb => 2.55,
            BiomeId::CoastalPromenade => 5.60,
            BiomeId::RollingRuralGravel => 2.65,
        }
    }

    /// Lateral offset (in meters, positive = right side of the road) for the main player cyclist.
    pub fn player_lane_lat_m(self) -> f32 {
        match self {
            BiomeId::MountainClimb => 1.15,
            BiomeId::CoastalPromenade => 1.65,
            BiomeId::RollingRuralGravel => 1.05,
        }
    }

    pub fn crr(self) -> f32 {
        match self {
            BiomeId::MountainClimb => 0.0055,
            BiomeId::CoastalPromenade => 0.0038,
            BiomeId::RollingRuralGravel => 0.0055,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RouteSample {
    pub pos: Vec3,
    pub tangent: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub grade_pct: f32,
    pub curvature: f32,
    pub distance_m: f32,
}

#[derive(Debug, Clone)]
pub struct RouteSpline {
    pub samples: Vec<RouteSample>,
    pub total_length_m: f32,
    pub elevation_profile: Vec<(f32, f32)>,
    pub minimap_points: Vec<[f32; 2]>,
    pub hero_start_dist_m: f32,
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

impl RouteSpline {
    pub fn for_biome(biome: BiomeId) -> Self {
        match biome {
            BiomeId::MountainClimb => Self::build_mountain_climb(),
            BiomeId::CoastalPromenade => Self::build_coastal_promenade(),
            BiomeId::RollingRuralGravel => Self::build_rolling_rural(),
        }
    }

    fn from_control_points(pts: &[Vec3], hero_start_dist_m: f32) -> Self {
        let n = pts.len();
        let sub_steps = 44usize;
        let mut raw_pts = Vec::with_capacity(n * sub_steps);
        for i in 0..n {
            let p0 = pts[(i + n - 1) % n];
            let p1 = pts[i];
            let p2 = pts[(i + 1) % n];
            let p3 = pts[(i + 2) % n];
            for s in 0..sub_steps {
                let t = s as f32 / sub_steps as f32;
                raw_pts.push(catmull_rom(p0, p1, p2, p3, t));
            }
        }

        let total_raw = raw_pts.len();
        let mut arc = Vec::with_capacity(total_raw + 1);
        arc.push(0.0_f32);
        for i in 0..total_raw {
            let d = raw_pts[i].distance(raw_pts[(i + 1) % total_raw]);
            arc.push(arc[i] + d);
        }
        let total_length_m = *arc.last().unwrap();

        let num_samples = (total_length_m / 2.0).round().max(400.0) as usize;
        let step_m = total_length_m / num_samples as f32;
        let mut uniform_pos = Vec::with_capacity(num_samples);
        let mut raw_idx = 0usize;
        for k in 0..num_samples {
            let target = k as f32 * step_m;
            while raw_idx + 1 < total_raw && arc[raw_idx + 1] < target {
                raw_idx += 1;
            }
            let seg_len = (arc[raw_idx + 1] - arc[raw_idx]).max(1e-5);
            let alpha = ((target - arc[raw_idx]) / seg_len).clamp(0.0, 1.0);
            let a = raw_pts[raw_idx];
            let b = raw_pts[(raw_idx + 1) % total_raw];
            uniform_pos.push(a.lerp(b, alpha));
        }

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let prev = uniform_pos[(i + num_samples - 1) % num_samples];
            let curr = uniform_pos[i];
            let next = uniform_pos[(i + 1) % num_samples];
            let tangent = (next - prev).normalize_or_zero();
            let horiz_len = (tangent.x * tangent.x + tangent.z * tangent.z).sqrt().max(1e-4);
            let grade_pct = (tangent.y / horiz_len) * 100.0;
            let right = tangent.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(tangent).normalize_or_zero();

            let t_prev = (curr - prev).normalize_or_zero();
            let t_next = (next - curr).normalize_or_zero();
            let d_tan = t_next - t_prev;
            let signed_turn = d_tan.dot(right);
            let curvature = signed_turn / step_m;

            samples.push(RouteSample {
                pos: curr,
                tangent,
                right,
                up,
                grade_pct,
                curvature,
                distance_m: i as f32 * step_m,
            });
        }

        let mut elevation_profile = Vec::with_capacity(200);
        for i in 0..200 {
            let s = &samples[(i * num_samples) / 200];
            elevation_profile.push((s.distance_m, s.pos.y));
        }

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for s in &samples {
            min_x = min_x.min(s.pos.x);
            max_x = max_x.max(s.pos.x);
            min_z = min_z.min(s.pos.z);
            max_z = max_z.max(s.pos.z);
        }
        let span = (max_x - min_x).max(max_z - min_z).max(1.0);
        let cx = 0.5 * (min_x + max_x);
        let cz = 0.5 * (min_z + max_z);
        let mut minimap_points = Vec::with_capacity(220);
        for i in 0..220 {
            let s = &samples[(i * num_samples) / 220];
            let nx = 0.5 + (s.pos.x - cx) / (span * 1.18);
            let nz = 0.5 + (s.pos.z - cz) / (span * 1.18);
            minimap_points.push([nx, nz]);
        }

        Self {
            samples,
            total_length_m,
            elevation_profile,
            minimap_points,
            hero_start_dist_m,
        }
    }

    /// Hero composition for Mountain Climb (Reference Images 1 & 3):
    /// Sweeps gracefully left and climbs upward onto the sunlit left ridge, opening the entire right half
    /// of the frame to the steep canyon drop-off, valley river, and layered mountain ridges!
    fn build_mountain_climb() -> Self {
        let pts = vec![
            Vec3::new(4.0, 86.2, 65.0),       // approach
            Vec3::new(0.0, 88.0, 0.0),        // HERO START (+2.8% grade)
            Vec3::new(-14.0, 89.2, -42.0),    // foreground curves left gently...
            Vec3::new(-20.0, 91.9, -95.0),    // ...sweeps cleanly along cliff rim...
            Vec3::new(-6.0, 96.5, -165.0),    // ...swings right & climbs visibly up center-left ridge!
            Vec3::new(-28.0, 97.8, -245.0),   // gentle saddle crest below rocky boulder hilltop
            Vec3::new(-160.0, 76.0, -320.0),  // drops behind left mountain saddle
            Vec3::new(-370.0, 62.0, -295.0),
            Vec3::new(-480.0, 52.0, -180.0),
            Vec3::new(-520.0, 48.0, -40.0),
            Vec3::new(-475.0, 52.0, 115.0),
            Vec3::new(-360.0, 58.0, 240.0),
            Vec3::new(-215.0, 68.0, 285.0),
            Vec3::new(-80.0, 78.0, 225.0),
            Vec3::new(-6.0, 84.4, 145.0),
        ];
        Self::from_control_points(&pts, 74.0)
    }

    /// Hero composition for Coastal Promenade (Reference Image 2):
    /// Curves gently left-then-right around a glowing sunset ocean bay so colonial townhouses & stepped
    /// coastal towers line the left 60% while the right 40% opens wide onto the sunset sea!
    fn build_coastal_promenade() -> Self {
        let pts = vec![
            Vec3::new(-6.0, 3.6, 90.0),
            Vec3::new(0.0, 4.2, 0.0),         // HERO START
            Vec3::new(-8.0, 5.1, -92.0),      // slight left sweep exposes entire right ocean vista
            Vec3::new(-2.0, 6.4, -190.0),     // curves smoothly right around bay promenade
            Vec3::new(36.0, 7.6, -295.0),
            Vec3::new(112.0, 8.2, -405.0),
            Vec3::new(195.0, 7.6, -505.0),
            Vec3::new(140.0, 6.4, -620.0),
            Vec3::new(-30.0, 7.5, -640.0),
            Vec3::new(-220.0, 11.5, -555.0),
            Vec3::new(-355.0, 15.0, -390.0),
            Vec3::new(-395.0, 12.0, -195.0),
            Vec3::new(-325.0, 8.0, 10.0),
            Vec3::new(-205.0, 4.8, 145.0),
            Vec3::new(-75.0, 3.4, 160.0),
        ];
        Self::from_control_points(&pts, 92.0)
    }

    /// Rolling Rural Gravel loop: gentle undulating country gravel road through olive hills & ranches.
    fn build_rolling_rural() -> Self {
        let pts = vec![
            Vec3::new(-12.0, 38.0, 80.0),
            Vec3::new(0.0, 40.0, 0.0),       // HERO START
            Vec3::new(16.0, 42.3, -85.0),
            Vec3::new(-8.0, 45.8, -175.0),
            Vec3::new(-55.0, 49.0, -260.0),
            Vec3::new(-145.0, 44.5, -330.0),
            Vec3::new(-265.0, 52.0, -315.0),
            Vec3::new(-360.0, 60.0, -210.0),
            Vec3::new(-380.0, 54.0, -70.0),
            Vec3::new(-315.0, 43.0, 65.0),
            Vec3::new(-200.0, 38.5, 160.0),
            Vec3::new(-80.0, 36.5, 150.0),
        ];
        Self::from_control_points(&pts, 84.0)
    }

    pub fn sample(&self, distance_m: f32) -> RouteSample {
        let len = self.total_length_m;
        let d = distance_m.rem_euclid(len);
        let n = self.samples.len();
        let f_idx = (d / len) * n as f32;
        let i0 = (f_idx.floor() as usize) % n;
        let i1 = (i0 + 1) % n;
        let frac = f_idx - f_idx.floor();
        let a = self.samples[i0];
        let b = self.samples[i1];
        let pos = a.pos.lerp(b.pos, frac);
        let tangent = a.tangent.lerp(b.tangent, frac).normalize_or_zero();
        let right = tangent.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(tangent).normalize_or_zero();
        RouteSample {
            pos,
            tangent,
            right,
            up,
            grade_pct: a.grade_pct + (b.grade_pct - a.grade_pct) * frac,
            curvature: a.curvature + (b.curvature - a.curvature) * frac,
            distance_m: d,
        }
    }

    pub fn minimap_point_at(&self, distance_m: f32) -> ([f32; 2], [f32; 2]) {
        let s = self.sample(distance_m);
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for pt in &self.samples {
            min_x = min_x.min(pt.pos.x);
            max_x = max_x.max(pt.pos.x);
            min_z = min_z.min(pt.pos.z);
            max_z = max_z.max(pt.pos.z);
        }
        let span = (max_x - min_x).max(max_z - min_z).max(1.0);
        let cx = 0.5 * (min_x + max_x);
        let cz = 0.5 * (min_z + max_z);
        let pos_2d = [
            0.5 + (s.pos.x - cx) / (span * 1.18),
            0.5 + (s.pos.z - cz) / (span * 1.18),
        ];
        let dir_2d = [s.tangent.x, s.tangent.z];
        (pos_2d, dir_2d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splines_are_closed_and_smooth_for_all_biomes() {
        for biome in [
            BiomeId::MountainClimb,
            BiomeId::CoastalPromenade,
            BiomeId::RollingRuralGravel,
        ] {
            let spline = RouteSpline::for_biome(biome);
            assert!(spline.total_length_m > 900.0);
            let s0 = spline.sample(0.0);
            let s_end = spline.sample(spline.total_length_m - 0.05);
            assert!(
                s0.pos.distance(s_end.pos) < 0.25,
                "{:?} loop gap too large",
                biome
            );
            let hero = spline.sample(spline.hero_start_dist_m);
            assert!(hero.tangent.length() > 0.99);
            assert!(hero.grade_pct.is_finite());
        }
    }

    #[test]
    fn level_name_parsing_and_random_biome_are_valid() {
        assert_eq!(std::mem::size_of::<crate::assets::SceneVertex>(), 24);
        assert_eq!(BiomeId::MountainClimb.name(), "Cerro");
        assert_eq!(BiomeId::CoastalPromenade.name(), "Rambla");
        assert_eq!(BiomeId::RollingRuralGravel.name(), "Campo");
        assert_eq!(BiomeId::from_str_arg("cerro"), Some(BiomeId::MountainClimb));
        assert_eq!(BiomeId::from_str_arg("rambla"), Some(BiomeId::CoastalPromenade));
        assert_eq!(BiomeId::from_str_arg("campo"), Some(BiomeId::RollingRuralGravel));
        assert_eq!(BiomeId::from_str_arg("mountain"), Some(BiomeId::MountainClimb));
        assert_eq!(BiomeId::from_str_arg("coastal"), Some(BiomeId::CoastalPromenade));
        assert_eq!(BiomeId::from_str_arg("rural"), Some(BiomeId::RollingRuralGravel));
        assert_eq!(BiomeId::from_str_arg("unknown_world"), None);
        let r = BiomeId::random();
        assert!(BiomeId::ALL.contains(&r));

        for &biome in &BiomeId::ALL {
            let sp = RouteSpline::for_biome(biome);
            let geo = crate::world::terrain::build_biome_world(biome, &sp);
            assert!(
                geo.sectors.len() >= 16
                    && geo.sectors.len() <= crate::world::terrain::MAX_OCTREE_SECTORS,
                "biome {:?} produced {} sectors",
                biome,
                geo.sectors.len()
            );
            assert!(geo.octree.nodes.len() > geo.sectors.len());

            // Verify CPU octree frustum culling selects only the visible subset of leaf sectors
            let hero_s = sp.sample(sp.hero_start_dist_m);
            let cam_pos = hero_s.pos - hero_s.tangent * 4.5 + Vec3::Y * 2.0;
            let cam_target = hero_s.pos + hero_s.tangent * 15.0 + Vec3::Y * 1.0;
            let view = glam::Mat4::look_at_rh(cam_pos, cam_target, Vec3::Y);
            let proj = glam::Mat4::perspective_rh(50.0_f32.to_radians(), 4.0 / 3.0, 0.25, 1900.0);
            let frustum = crate::world::terrain::Frustum::from_view_proj(proj * view);
            let mut vis = Vec::new();
            geo.octree.query_frustum(&frustum, &mut vis);
            assert!(!vis.is_empty() && vis.len() < geo.sectors.len());
        }

        // Verify 8-phase Halton(2, 3) TAA sub-pixel jitter distribution
        let mut sum_x = 0.0_f32;
        let mut sum_y = 0.0_f32;
        for i in 0..8 {
            let j = crate::renderer::halton_2_3_jitter_px(i);
            assert!(j[0].abs() <= 0.5 && j[1].abs() <= 0.5);
            sum_x += j[0];
            sum_y += j[1];
        }
        assert!((sum_x / 8.0).abs() < 0.1 && (sum_y / 8.0).abs() < 0.1);

        // Verify dynamic moving & compute-skinned animals in Campo (Walking, EatingGrass, Idle)
        let campo_sp = RouteSpline::for_biome(BiomeId::RollingRuralGravel);
        let mut animals =
            crate::world::riders::spawn_biome_animals(BiomeId::RollingRuralGravel, &campo_sp);
        assert!(
            animals.len() >= 18 && animals.len() <= 42,
            "expected sparse natural Campo animal groups (got {})",
            animals.len()
        );
        assert!(animals
            .iter()
            .any(|a| a.behavior == crate::world::riders::AnimalBehavior::Walking));
        assert!(animals
            .iter()
            .any(|a| a.behavior == crate::world::riders::AnimalBehavior::EatingGrass));
        assert!(animals
            .iter()
            .any(|a| a.behavior == crate::world::riders::AnimalBehavior::Idle));

        let init_pos: Vec<(f32, f32, f32)> = animals
            .iter()
            .map(|a| (a.dist_m, a.lat_m, a.walk_phase))
            .collect();
        for _ in 0..50 {
            crate::world::riders::step_animals(
                0.1,
                &campo_sp,
                BiomeId::RollingRuralGravel,
                &mut animals,
            );
        }
        let moved_count = animals
            .iter()
            .zip(init_pos.iter())
            .filter(|(a, (d0, l0, p0))| {
                (a.dist_m - *d0).abs() > 0.05
                    || (a.lat_m - *l0).abs() > 0.05
                    || (a.walk_phase - *p0).abs() > 0.05
            })
            .count();
        assert!(moved_count >= 8, "expected walking animals to advance position and walk_phase");

        let hero_s = campo_sp.sample(campo_sp.hero_start_dist_m);
        let cam_pos = hero_s.pos - hero_s.tangent * 4.5 + Vec3::Y * 2.0;
        let mut draw_parts = Vec::new();
        crate::world::riders::build_animal_draw_parts(
            BiomeId::RollingRuralGravel,
            &campo_sp,
            &animals,
            cam_pos,
            &mut draw_parts,
        );
        assert!(
            draw_parts.len() >= 5 && draw_parts.len() <= 14,
            "expected sparse visible dynamic animals in Campo hero vista (got {})",
            draw_parts.len()
        );
        assert!(draw_parts.iter().all(|p| p.skin_pose_r[3] >= 3.0));

        // Verify main player rider is always on the right side of the road and AI cyclists include
        // both same-direction and opposite-direction solo riders and cycling groups (~2 encounters/min)
        for &b in &BiomeId::ALL {
            assert!(
                b.player_lane_lat_m() > 0.8 && b.player_lane_lat_m() < b.road_half_width(),
                "expected right-lane player offset for {:?}",
                b
            );
        }
        let mut ai = crate::world::riders::spawn_ai_peloton(
            campo_sp.hero_start_dist_m,
            campo_sp.total_length_m,
        );
        assert!(ai.iter().any(|r| r.direction > 0.0 && r.group_id == 0));
        assert!(ai.iter().any(|r| r.direction < 0.0 && r.group_id == 0));
        assert!(ai.iter().any(|r| r.direction > 0.0 && r.group_id > 0));
        assert!(ai.iter().any(|r| r.direction < 0.0 && r.group_id > 0));
        assert!(ai.iter().filter(|r| r.direction > 0.0).all(|r| r.lateral_m > 0.0));
        assert!(ai.iter().filter(|r| r.direction < 0.0).all(|r| r.lateral_m < 0.0));

        let d0_fwd = ai.iter().find(|r| r.direction > 0.0).unwrap().distance_m;
        let d0_rev = ai.iter().find(|r| r.direction < 0.0).unwrap().distance_m;
        crate::world::riders::step_ai_peloton(
            0.1,
            &campo_sp,
            campo_sp.hero_start_dist_m,
            BiomeId::RollingRuralGravel.player_lane_lat_m(),
            &mut ai,
        );
        let d1_fwd = ai.iter().find(|r| r.direction > 0.0).unwrap().distance_m;
        let d1_rev = ai.iter().find(|r| r.direction < 0.0).unwrap().distance_m;
        assert!(d1_fwd > d0_fwd);
        assert!(d1_rev < d0_rev);

        // Verify Rambla (CoastalPromenade) has 20+ building types, reuses Campo trees,
        // places street-edge benches facing the sea with seated pedestrians, and spawns dynamic Rambla walkers
        let rambla_sp = RouteSpline::for_biome(BiomeId::CoastalPromenade);
        let rambla_geo = crate::world::terrain::build_biome_world(BiomeId::CoastalPromenade, &rambla_sp);
        let mut rambla_building_types = std::collections::HashSet::new();
        for p in &rambla_geo.props {
            if matches!(
                p.model,
                crate::assets::ModelId::RamblaBuilding(_)
                    | crate::assets::ModelId::ColonialTownhouse
                    | crate::assets::ModelId::CoastalBuildingMid
                    | crate::assets::ModelId::CoastalBuildingTall
            ) {
                rambla_building_types.insert(p.model);
            }
        }
        assert!(
            rambla_building_types.len() >= 20,
            "expected at least 20 distinct building types on the Rambla (got {})",
            rambla_building_types.len()
        );
        assert!(rambla_geo
            .props
            .iter()
            .any(|p| matches!(p.model, crate::assets::ModelId::TreeOmbu(_))));
        assert!(rambla_geo
            .props
            .iter()
            .any(|p| matches!(p.model, crate::assets::ModelId::TreeCeibo(_))));
        assert!(rambla_geo
            .props
            .iter()
            .any(|p| matches!(p.model, crate::assets::ModelId::TreeEucalyptus(_))));
        assert!(rambla_geo
            .props
            .iter()
            .any(|p| matches!(p.model, crate::assets::ModelId::SmallPalm(_))));
        assert!(rambla_geo
            .props
            .iter()
            .any(|p| matches!(p.model, crate::assets::ModelId::RamblaSeated(_))));
        assert!(!rambla_geo.props.iter().any(|p| matches!(
            p.model,
            crate::assets::ModelId::CeiboBush(_) | crate::assets::ModelId::ChircaBush(_)
        )));

        let mut rambla_walkers =
            crate::world::riders::spawn_biome_animals(BiomeId::CoastalPromenade, &rambla_sp);
        assert!(
            rambla_walkers.len() >= 25,
            "expected dynamic walking pedestrians on the Rambla (got {})",
            rambla_walkers.len()
        );
        assert!(rambla_walkers
            .iter()
            .all(|w| w.species == crate::world::riders::AnimalSpecies::RamblaWalker));
        let w0_d = rambla_walkers[0].dist_m;
        crate::world::riders::step_animals(
            0.2,
            &rambla_sp,
            BiomeId::CoastalPromenade,
            &mut rambla_walkers,
        );
        assert!((rambla_walkers[0].dist_m - w0_d).abs() > 0.05);
    }

    #[test]
    fn rambla_tall_buildings_never_clipped_or_culled_across_csm_cascades() {
        use crate::assets::{AssetBank, ModelId};
        use crate::renderer::{compute_csm_cascade_vp, CSM_ORTHO_RADII, NUM_SHADOW_CASCADES};
        use crate::world::terrain::{Frustum, OctreeNode};

        fn collect_ancestors(
            nodes: &[OctreeNode],
            node_idx: usize,
            target_sid: u32,
            path: &mut Vec<usize>,
        ) -> bool {
            path.push(node_idx);
            let node = &nodes[node_idx];
            if node.leaf_sector_id == Some(target_sid) {
                return true;
            }
            for c_idx in node.children.into_iter().flatten() {
                if collect_ancestors(nodes, c_idx, target_sid, path) {
                    return true;
                }
            }
            path.pop();
            false
        }

        let sp = RouteSpline::for_biome(BiomeId::CoastalPromenade);
        let geo = crate::world::terrain::build_biome_world(BiomeId::CoastalPromenade, &sp);
        let assets = AssetBank::load_embedded();

        struct BuildingBounds {
            sector_id: usize,
            sphere_center: Vec3,
            sphere_radius: f32,
            world_verts: Vec<Vec3>,
        }

        let mut buildings = Vec::new();
        for p in &geo.props {
            if !matches!(
                p.model,
                ModelId::RamblaBuilding(_)
                    | ModelId::ColonialTownhouse
                    | ModelId::CoastalBuildingMid
                    | ModelId::CoastalBuildingTall
            ) {
                continue;
            }
            let model = assets.get(p.model);
            let mut b_min = Vec3::splat(f32::MAX);
            let mut b_max = Vec3::splat(f32::MIN);
            let mut world_verts = Vec::new();
            for sec in &model.sections {
                for v in &sec.vertices {
                    let wp = p.transform.transform_point3(Vec3::from(v.pos));
                    b_min = b_min.min(wp);
                    b_max = b_max.max(wp);
                    world_verts.push(wp);
                }
            }

            let sid = p.sector_id as usize;
            let sector = &geo.sectors[sid];
            assert!(
                b_min.x >= sector.aabb_min.x
                    && b_min.y >= sector.aabb_min.y
                    && b_min.z >= sector.aabb_min.z
                    && b_max.x <= sector.aabb_max.x
                    && b_max.y <= sector.aabb_max.y
                    && b_max.z <= sector.aabb_max.z,
                "building {:?} bounds ({:?}..{:?}) exceed sector {} AABB ({:?}..{:?})",
                p.model,
                b_min,
                b_max,
                sid,
                sector.aabb_min,
                sector.aabb_max
            );

            let mut ancestors = Vec::new();
            assert!(collect_ancestors(&geo.octree.nodes, geo.octree.root, p.sector_id, &mut ancestors));
            for anc_idx in ancestors {
                let anc = &geo.octree.nodes[anc_idx];
                assert!(
                    b_min.x >= anc.aabb_min.x
                        && b_min.y >= anc.aabb_min.y
                        && b_min.z >= anc.aabb_min.z
                        && b_max.x <= anc.aabb_max.x
                        && b_max.y <= anc.aabb_max.y
                        && b_max.z <= anc.aabb_max.z,
                    "building {:?} bounds ({:?}..{:?}) exceed octree ancestor {} AABB ({:?}..{:?})",
                    p.model,
                    b_min,
                    b_max,
                    anc_idx,
                    anc.aabb_min,
                    anc.aabb_max
                );
            }

            let sx = p.transform.x_axis.truncate().length();
            let sy = p.transform.y_axis.truncate().length();
            let sz = p.transform.z_axis.truncate().length();
            let max_scale = sx.max(sy).max(sz).max(0.01);
            let sphere_center = p.transform.transform_point3(model.bounds_center);
            let sphere_radius = model.bounding_radius * max_scale;

            buildings.push(BuildingBounds {
                sector_id: sid,
                sphere_center,
                sphere_radius,
                world_verts,
            });
        }
        assert!(!buildings.is_empty());

        let sun_dir = geo.hero_sun_dir;
        let shadow_sun_dir = Vec3::new(sun_dir.x, sun_dir.y.max(0.46), sun_dir.z).normalize();
        let mut test_dists = vec![sp.hero_start_dist_m, sp.hero_start_dist_m + 400.0];
        let mut d = 0.0_f32;
        while d < sp.total_length_m {
            test_dists.push(d);
            d += 120.0;
        }

        for dist_m in test_dists {
            let s = sp.sample(dist_m);
            let player_pos = s.pos + s.right * BiomeId::CoastalPromenade.player_lane_lat_m();
            let cam_pos = player_pos - s.tangent * 4.45 + s.up * 1.94 + s.right * 0.02;
            let cam_target = player_pos + s.tangent * 15.5 + s.up * 0.85;
            let view_fwd = (cam_target - cam_pos).normalize_or_zero();
            let csm_centers = [
                player_pos + s.tangent * 3.0,
                cam_pos + view_fwd * 28.0,
                cam_pos + view_fwd * 86.0,
            ];

            for ci in 0..NUM_SHADOW_CASCADES {
                let vp = compute_csm_cascade_vp(csm_centers[ci], shadow_sun_dir, CSM_ORTHO_RADII[ci]);
                let frustum = Frustum::from_view_proj(vp);
                let mut vis_sectors = Vec::new();
                geo.octree.query_frustum(&frustum, &mut vis_sectors);

                for b in &buildings {
                    let mut any_in_ortho_xy = false;
                    for &v in &b.world_verts {
                        let clip = vp * v.extend(1.0);
                        let ndc = clip.truncate() / clip.w;
                        if ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 {
                            any_in_ortho_xy = true;
                            assert!(
                                ndc.z >= 0.0 && ndc.z <= 1.0,
                                "building vertex {:?} clipped in cascade {} at dist {} (ndc.z = {})",
                                v,
                                ci,
                                dist_m,
                                ndc.z
                            );
                        }
                    }
                    if any_in_ortho_xy {
                        assert!(
                            vis_sectors.contains(&b.sector_id),
                            "sector {} missing in cascade {} octree query at dist {}",
                            b.sector_id,
                            ci,
                            dist_m
                        );
                        assert!(
                            frustum.intersects_sphere(b.sphere_center, b.sphere_radius),
                            "building sphere ({:?}, r={}) culled in cascade {} at dist {}",
                            b.sphere_center,
                            b.sphere_radius,
                            ci,
                            dist_m
                        );
                    }
                }
            }
        }
    }
}
