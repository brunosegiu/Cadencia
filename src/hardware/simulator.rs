use crate::hardware::{HardwareProvider, TelemetrySample, TrainerMode};

/// Built-in cross-platform smart trainer & sensor simulator.
/// Defaults match the reference HUD metrics (185W, 92 RPM, 142 BPM, Gear 4, 66kg -> 2.8 W/kg).
pub struct SimulatedHardware {
    pub target_power_w: f32,
    pub target_cadence_rpm: f32,
    pub target_hr_bpm: f32,
    pub gear: u8,
    pub rider_weight_kg: f32,
    pub erg_mode: bool,
    pub erg_target_w: u16,
    pub current_grade_pct: f32,
    pub micro_variation: bool,
    phase: f32,
}

impl Default for SimulatedHardware {
    fn default() -> Self {
        Self {
            target_power_w: 185.0,
            target_cadence_rpm: 92.8,
            target_hr_bpm: 142.0,
            gear: 4,
            rider_weight_kg: 66.0,
            erg_mode: false,
            erg_target_w: 185,
            current_grade_pct: 2.5,
            micro_variation: false,
            phase: 0.0,
        }
    }
}

impl SimulatedHardware {
    pub fn shift_gear(&mut self, delta: i8) {
        let next = (self.gear as i16 + delta as i16).clamp(1, 12) as u8;
        if next != self.gear {
            self.gear = next;
            if !self.erg_mode {
                // Shifting up adds modest resistance / cadence adjustment
                self.target_power_w = (self.target_power_w + delta as f32 * 14.0).clamp(60.0, 650.0);
            }
        }
    }
}

impl HardwareProvider for SimulatedHardware {
    fn poll_telemetry(&mut self) -> TelemetrySample {
        self.phase += 0.035;
        let base_w = if self.erg_mode {
            self.erg_target_w as f32
        } else {
            self.target_power_w
        };
        let wobble_w = if self.micro_variation {
            (self.phase * 1.3).sin() * 2.5
        } else {
            0.0
        };
        let wobble_cad = if self.micro_variation {
            (self.phase * 0.9).cos() * 0.8
        } else {
            0.0
        };
        // Simulate subtle cadence/HR response to grade in SIM mode
        let grade_cad_adj = if self.erg_mode {
            0.0
        } else {
            -(self.current_grade_pct * 0.35).clamp(-6.0, 4.0)
        };
        TelemetrySample {
            power_w: (base_w + wobble_w).max(0.0),
            cadence_rpm: (self.target_cadence_rpm + grade_cad_adj + wobble_cad).clamp(0.0, 145.0),
            heart_rate_bpm: self.target_hr_bpm.clamp(45.0, 210.0),
            speed_mps: None, // Computed from physics engine
            connected: true,
            source_name: format!("Sim Trainer (Gear {})", self.gear),
        }
    }

    fn set_trainer_mode(&mut self, mode: TrainerMode) {
        match mode {
            TrainerMode::Sim { grade_pct, .. } => {
                self.current_grade_pct = grade_pct;
            }
            TrainerMode::Erg { target_watts } => {
                self.erg_mode = true;
                self.erg_target_w = target_watts;
            }
        }
    }

    fn start_scan(&mut self) {}

    fn disconnect(&mut self) {}

    fn status_text(&self) -> String {
        if self.erg_mode {
            format!("Simulator · ERG {}W", self.erg_target_w)
        } else {
            format!(
                "Simulator · SIM {:.1}% · Gear {}",
                self.current_grade_pct, self.gear
            )
        }
    }

    fn is_simulator(&self) -> bool {
        true
    }
}
