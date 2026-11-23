#[derive(Debug, Clone)]
pub struct PhysicsParams {
    pub rider_mass_kg: f32,
    pub bike_mass_kg: f32,
    pub crr: f32,
    pub cda: f32,
    pub air_density: f32,
    pub drivetrain_eff: f32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            rider_mass_kg: 66.0,
            bike_mass_kg: 8.5,
            crr: 0.0055,
            cda: 0.34,
            air_density: 1.225,
            drivetrain_eff: 0.97,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiderPhysicsState {
    pub distance_m: f32,
    pub display_distance_m: f32,
    pub speed_mps: f32,
    pub elevation_m: f32,
    pub elevation_gain_m: f32,
    pub grade_pct: f32,
    pub crank_angle_rad: f32,
    pub wheel_angle_rad: f32,
    pub lean_angle_rad: f32,
    pub draft_factor: f32,
    pub avg_power_w: f32,
    pub elapsed_s: f32,
    pub drops_points: u64,
    pub last_forces: PhysicsForces,
    power_samples_sum: f64,
    power_samples_count: u64,
    drops_accum: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicsForces {
    pub f_drive_n: f32,
    pub f_gravity_n: f32,
    pub f_rolling_n: f32,
    pub f_aero_n: f32,
}

impl RiderPhysicsState {
    /// Initializes rider state matching the reference HUD screenshot (4.2 km, 88 m gain, 14:32, 24.5 km/h).
    pub fn new_hero_default(start_distance_m: f32, elevation_m: f32, grade_pct: f32) -> Self {
        Self {
            distance_m: start_distance_m,
            display_distance_m: 4200.0,
            speed_mps: 24.5 / 3.6,
            elevation_m,
            elevation_gain_m: 88.0,
            grade_pct,
            crank_angle_rad: -1.35,
            wheel_angle_rad: 0.0,
            lean_angle_rad: 0.0,
            draft_factor: 0.0,
            avg_power_w: 92.0,
            elapsed_s: 14.0 * 60.0 + 32.0,
            drops_points: 252_870,
            last_forces: PhysicsForces::default(),
            power_samples_sum: 92.0 * 872.0,
            power_samples_count: 872,
            drops_accum: 0.0,
        }
    }
}

/// Steps realistic cycling physics over `dt` seconds.
pub fn step_physics(
    dt: f32,
    power_w: f32,
    cadence_rpm: f32,
    grade_pct: f32,
    curvature: f32,
    draft_factor: f32,
    params: &PhysicsParams,
    state: &mut RiderPhysicsState,
) {
    let dt = dt.clamp(0.0, 0.1);
    if dt <= 0.0 {
        return;
    }

    let g = 9.80665_f32;
    let total_mass = (params.rider_mass_kg + params.bike_mass_kg).max(40.0);
    let slope = (grade_pct * 0.01).atan();
    let sin_s = slope.sin();
    let cos_s = slope.cos();

    let v = state.speed_mps.max(0.0);
    let f_gravity = total_mass * g * sin_s;
    let f_rolling = params.crr * total_mass * g * cos_s;
    let eff_cda = params.cda * (1.0 - draft_factor.clamp(0.0, 0.38));
    let f_aero = 0.5 * params.air_density * eff_cda * v * v;
    let f_drive = if power_w > 0.5 {
        (power_w * params.drivetrain_eff) / v.max(1.4)
    } else {
        0.0
    };

    let f_net = f_drive - (f_gravity + f_rolling + f_aero);
    let accel = (f_net / total_mass).clamp(-4.5, 4.5);
    let next_v = (v + accel * dt).clamp(0.0, 32.0);
    let avg_v = 0.5 * (v + next_v);
    let ds = avg_v * dt;

    let prev_elev = state.elevation_m;
    state.speed_mps = next_v;
    state.distance_m += ds;
    state.display_distance_m += ds;
    state.grade_pct = grade_pct;
    let dh = ds * (grade_pct * 0.01);
    state.elevation_m = prev_elev + dh;
    if dh > 0.0 {
        state.elevation_gain_m += dh;
    }

    // Wheel rotation (radius = 0.34 m) and crank rotation from cadence
    state.wheel_angle_rad = (state.wheel_angle_rad + ds / 0.34).rem_euclid(std::f32::consts::TAU);
    let omega_crank = (cadence_rpm / 60.0) * std::f32::consts::TAU;
    state.crank_angle_rad = (state.crank_angle_rad + omega_crank * dt).rem_euclid(std::f32::consts::TAU);

    // Banking lean into curves
    let target_lean = (next_v * next_v * curvature / g).atan().clamp(-0.34, 0.34) * 0.65;
    state.lean_angle_rad += (target_lean - state.lean_angle_rad) * (dt * 6.0).min(1.0);

    state.draft_factor = draft_factor;
    state.elapsed_s += dt;
    state.power_samples_sum += power_w as f64 * dt as f64;
    state.power_samples_count += 1;
    // Running average power (for display)
    state.avg_power_w = (state.power_samples_sum / state.elapsed_s.max(1.0) as f64) as f32;

    state.drops_accum += (power_w * 0.14 + ds * 0.8) * dt;
    if state.drops_accum >= 1.0 {
        let add = state.drops_accum.floor() as u64;
        state.drops_points += add;
        state.drops_accum -= add as f32;
    }

    state.last_forces = PhysicsForces {
        f_drive_n: f_drive,
        f_gravity_n: f_gravity,
        f_rolling_n: f_rolling,
        f_aero_n: f_aero,
    };
}

/// Computes aerodynamic drafting reduction (0.0 .. 0.30) behind nearby riders on the closed loop.
pub fn compute_draft_factor(
    player_dist: f32,
    player_lat: f32,
    others: &[(f32, f32)],
    route_len_m: f32,
) -> f32 {
    let mut best = 0.0_f32;
    for &(other_dist, other_lat) in others {
        let mut ahead = (other_dist - player_dist).rem_euclid(route_len_m);
        if ahead > route_len_m * 0.5 {
            ahead -= route_len_m;
        }
        if (0.8..=9.0).contains(&ahead) {
            let lat_diff = (other_lat - player_lat).abs();
            if lat_diff <= 1.25 {
                let dist_score = if ahead <= 2.5 {
                    1.0
                } else {
                    1.0 - (ahead - 2.5) / 6.5
                };
                let lat_score = 1.0 - (lat_diff / 1.25) * 0.7;
                best = best.max(0.30 * dist_score * lat_score);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_steady_state_speed_is_realistic() {
        let params = PhysicsParams {
            rider_mass_kg: 66.0,
            bike_mass_kg: 8.5,
            crr: 0.0040,
            cda: 0.32,
            air_density: 1.225,
            drivetrain_eff: 0.97,
        };
        let mut st = RiderPhysicsState::new_hero_default(0.0, 10.0, 0.0);
        st.speed_mps = 5.0;
        for _ in 0..800 {
            step_physics(0.05, 220.0, 90.0, 0.0, 0.0, 0.0, &params, &mut st);
        }
        let kph = st.speed_mps * 3.6;
        assert!(kph > 32.0 && kph < 38.0, "expected realistic ~35 kph at 220W flat, got {kph}");
    }

    #[test]
    fn climbing_is_slower_than_flat_and_descending_is_faster() {
        let params = PhysicsParams::default();
        let mut climb = RiderPhysicsState::new_hero_default(0.0, 100.0, 6.0);
        let mut flat = RiderPhysicsState::new_hero_default(0.0, 100.0, 0.0);
        let mut desc = RiderPhysicsState::new_hero_default(0.0, 100.0, -4.0);
        for _ in 0..600 {
            step_physics(0.05, 200.0, 88.0, 6.0, 0.0, 0.0, &params, &mut climb);
            step_physics(0.05, 200.0, 88.0, 0.0, 0.0, 0.0, &params, &mut flat);
            step_physics(0.05, 200.0, 88.0, -4.0, 0.0, 0.0, &params, &mut desc);
        }
        assert!(climb.speed_mps < flat.speed_mps);
        assert!(flat.speed_mps < desc.speed_mps);
    }

    #[test]
    fn drafting_reduces_aero_drag_and_increases_speed() {
        let others = vec![(103.0, 0.0)];
        let d = compute_draft_factor(100.0, 0.1, &others, 5000.0);
        assert!(d > 0.20 && d <= 0.30, "unexpected draft factor {d}");

        let params = PhysicsParams::default();
        let mut solo = RiderPhysicsState::new_hero_default(0.0, 0.0, 0.0);
        let mut draft = RiderPhysicsState::new_hero_default(0.0, 0.0, 0.0);
        for _ in 0..400 {
            step_physics(0.05, 250.0, 92.0, 0.0, 0.0, 0.0, &params, &mut solo);
            step_physics(0.05, 250.0, 92.0, 0.0, 0.0, d, &params, &mut draft);
        }
        assert!(draft.speed_mps > solo.speed_mps);
    }
}
