pub mod simulator;

#[cfg(all(not(target_arch = "wasm32"), feature = "ble"))]
pub mod ble_native;

#[cfg(target_arch = "wasm32")]
pub mod ble_web;

use uuid::Uuid;

// Standard Cycling Bluetooth Low Energy GATT Service & Characteristic UUIDs
pub const FTMS_SERVICE_UUID: Uuid = Uuid::from_u128(0x00001826_0000_1000_8000_00805f9b34fb);
pub const FTMS_INDOOR_BIKE_DATA_UUID: Uuid = Uuid::from_u128(0x00002ad2_0000_1000_8000_00805f9b34fb);
pub const FTMS_CONTROL_POINT_UUID: Uuid = Uuid::from_u128(0x00002ad9_0000_1000_8000_00805f9b34fb);

pub const CPS_SERVICE_UUID: Uuid = Uuid::from_u128(0x00001818_0000_1000_8000_00805f9b34fb);
pub const CPS_MEASUREMENT_UUID: Uuid = Uuid::from_u128(0x00002a63_0000_1000_8000_00805f9b34fb);

pub const CSC_SERVICE_UUID: Uuid = Uuid::from_u128(0x00001816_0000_1000_8000_00805f9b34fb);
pub const CSC_MEASUREMENT_UUID: Uuid = Uuid::from_u128(0x00002a5b_0000_1000_8000_00805f9b34fb);

pub const HRS_SERVICE_UUID: Uuid = Uuid::from_u128(0x0000180d_0000_1000_8000_00805f9b34fb);
pub const HRS_MEASUREMENT_UUID: Uuid = Uuid::from_u128(0x00002a37_0000_1000_8000_00805f9b34fb);

#[derive(Debug, Clone)]
pub struct TelemetrySample {
    pub power_w: f32,
    pub cadence_rpm: f32,
    pub heart_rate_bpm: f32,
    pub speed_mps: Option<f32>,
    pub connected: bool,
    pub source_name: String,
}

impl Default for TelemetrySample {
    fn default() -> Self {
        Self {
            power_w: 185.0,
            cadence_rpm: 92.0,
            heart_rate_bpm: 142.0,
            speed_mps: None,
            connected: true,
            source_name: "Built-in Simulator".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainerMode {
    Sim {
        grade_pct: f32,
        crr: f32,
        cw: f32,
    },
    Erg {
        target_watts: u16,
    },
}

pub trait HardwareProvider {
    fn poll_telemetry(&mut self) -> TelemetrySample;
    fn set_trainer_mode(&mut self, mode: TrainerMode);
    fn start_scan(&mut self);
    fn disconnect(&mut self);
    fn status_text(&self) -> String;
    fn is_simulator(&self) -> bool;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct FtmsIndoorBikeData {
    pub speed_kph: Option<f32>,
    pub cadence_rpm: Option<f32>,
    pub power_w: Option<i16>,
    pub heart_rate_bpm: Option<u8>,
}

/// Parses GATT FTMS Indoor Bike Data characteristic (`0x2AD2`).
pub fn parse_ftms_indoor_bike_data(data: &[u8]) -> Option<FtmsIndoorBikeData> {
    if data.len() < 2 {
        return None;
    }
    let flags = u16::from_le_bytes([data[0], data[1]]);
    let mut offset = 2usize;
    let mut out = FtmsIndoorBikeData::default();

    // Bit 0 == 0 means Instantaneous Speed IS present (inverted flag per FTMS spec)
    if (flags & 0x0001) == 0 {
        if offset + 2 > data.len() {
            return None;
        }
        let raw_speed = u16::from_le_bytes([data[offset], data[offset + 1]]);
        out.speed_kph = Some(raw_speed as f32 * 0.01);
        offset += 2;
    }
    // Bit 1: Average Speed present
    if (flags & 0x0002) != 0 {
        offset = offset.saturating_add(2);
    }
    // Bit 2: Instantaneous Cadence present (0.5 rpm units)
    if (flags & 0x0004) != 0 {
        if offset + 2 > data.len() {
            return Some(out);
        }
        let raw_cad = u16::from_le_bytes([data[offset], data[offset + 1]]);
        out.cadence_rpm = Some(raw_cad as f32 * 0.5);
        offset += 2;
    }
    // Bit 3: Average Cadence present
    if (flags & 0x0008) != 0 {
        offset = offset.saturating_add(2);
    }
    // Bit 4: Total Distance present (uint24)
    if (flags & 0x0010) != 0 {
        offset = offset.saturating_add(3);
    }
    // Bit 5: Resistance Level present (sint16)
    if (flags & 0x0020) != 0 {
        offset = offset.saturating_add(2);
    }
    // Bit 6: Instantaneous Power present (sint16, 1W units)
    if (flags & 0x0040) != 0 {
        if offset + 2 > data.len() {
            return Some(out);
        }
        out.power_w = Some(i16::from_le_bytes([data[offset], data[offset + 1]]));
        offset += 2;
    }
    // Bit 7: Average Power present
    if (flags & 0x0080) != 0 {
        offset = offset.saturating_add(2);
    }
    // Bit 8: Expended Energy present (total 2 + per hour 2 + per min 1 = 5 bytes)
    if (flags & 0x0100) != 0 {
        offset = offset.saturating_add(5);
    }
    // Bit 9: Heart Rate present (uint8)
    if (flags & 0x0200) != 0 && offset < data.len() {
        out.heart_rate_bpm = Some(data[offset]);
    }
    Some(out)
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CpsMeasurement {
    pub power_w: i16,
    pub crank_revs: Option<u16>,
    pub crank_event_time_1024: Option<u16>,
}

/// Parses GATT Cycling Power Measurement (`0x2A63`).
pub fn parse_cps_measurement(data: &[u8]) -> Option<CpsMeasurement> {
    if data.len() < 4 {
        return None;
    }
    let flags = u16::from_le_bytes([data[0], data[1]]);
    let power_w = i16::from_le_bytes([data[2], data[3]]);
    let mut offset = 4usize;

    if (flags & 0x0001) != 0 {
        offset += 1; // pedal power balance
    }
    if (flags & 0x0004) != 0 {
        offset += 2; // accumulated torque
    }
    if (flags & 0x0010) != 0 {
        offset += 6; // wheel revolution data
    }
    let (crank_revs, crank_event_time_1024) = if (flags & 0x0020) != 0 && offset + 4 <= data.len() {
        let revs = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let time = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);
        (Some(revs), Some(time))
    } else {
        (None, None)
    };
    Some(CpsMeasurement {
        power_w,
        crank_revs,
        crank_event_time_1024,
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CscMeasurement {
    pub wheel_revs: Option<u32>,
    pub wheel_event_time_1024: Option<u16>,
    pub crank_revs: Option<u16>,
    pub crank_event_time_1024: Option<u16>,
}

/// Parses GATT Cycling Speed & Cadence Measurement (`0x2A5B`).
pub fn parse_csc_measurement(data: &[u8]) -> Option<CscMeasurement> {
    if data.is_empty() {
        return None;
    }
    let flags = data[0];
    let mut offset = 1usize;
    let mut out = CscMeasurement::default();
    if (flags & 0x01) != 0 {
        if offset + 6 > data.len() {
            return None;
        }
        out.wheel_revs = Some(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]));
        out.wheel_event_time_1024 = Some(u16::from_le_bytes([data[offset + 4], data[offset + 5]]));
        offset += 6;
    }
    if (flags & 0x02) != 0 && offset + 4 <= data.len() {
        out.crank_revs = Some(u16::from_le_bytes([data[offset], data[offset + 1]]));
        out.crank_event_time_1024 = Some(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
    }
    Some(out)
}

/// Parses GATT Heart Rate Measurement (`0x2A37`).
pub fn parse_hrs_measurement(data: &[u8]) -> Option<u16> {
    if data.len() < 2 {
        return None;
    }
    let flags = data[0];
    if (flags & 0x01) != 0 {
        if data.len() < 3 {
            return None;
        }
        Some(u16::from_le_bytes([data[1], data[2]]))
    } else {
        Some(data[1] as u16)
    }
}

/// Encodes FTMS Control Point (`0x2AD9`) OpCode `0x05`: Set Target Power (ERG watts).
pub fn encode_ftms_set_erg_power(target_watts: u16) -> [u8; 3] {
    let w = (target_watts.min(i16::MAX as u16) as i16).to_le_bytes();
    [0x05, w[0], w[1]]
}

/// Encodes FTMS Control Point (`0x2AD9`) OpCode `0x11`: Set Indoor Bike Simulation Parameters.
pub fn encode_ftms_set_sim_params(grade_pct: f32, crr: f32, cw: f32) -> [u8; 7] {
    let wind_speed_mps: i16 = 0; // 0.001 m/s
    let grade_raw = (grade_pct.clamp(-40.0, 40.0) * 100.0).round() as i16; // 0.01% units
    let crr_raw = (crr.clamp(0.0, 0.025) * 10_000.0).round() as u8; // 0.0001 units
    let cw_raw = (cw.clamp(0.0, 2.5) * 100.0).round() as u8; // 0.01 kg/m units
    let ws = wind_speed_mps.to_le_bytes();
    let gr = grade_raw.to_le_bytes();
    [0x11, ws[0], ws[1], gr[0], gr[1], crr_raw, cw_raw]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ftms_indoor_bike_data_correctly() {
        // Flags: speed present (bit0=0), inst cadence (bit2=1), inst power (bit6=1), HR (bit9=1)
        // flags = 0x0004 | 0x0040 | 0x0200 = 0x0244
        let pkt = [
            0x44, 0x02, // flags LE
            0x92, 0x09, // speed = 2450 -> 24.50 kph
            0xB8, 0x00, // cadence = 184 -> 92.0 rpm
            0xB9, 0x00, // power = 185 W
            0x8E,       // HR = 142 bpm
        ];
        let parsed = parse_ftms_indoor_bike_data(&pkt).unwrap();
        assert!((parsed.speed_kph.unwrap() - 24.5).abs() < 0.01);
        assert!((parsed.cadence_rpm.unwrap() - 92.0).abs() < 0.01);
        assert_eq!(parsed.power_w, Some(185));
        assert_eq!(parsed.heart_rate_bpm, Some(142));
    }

    #[test]
    fn parses_cps_csc_hrs_and_encodes_ftms_control() {
        let cps = parse_cps_measurement(&[0x20, 0x00, 0xFA, 0x00, 0x0A, 0x00, 0x00, 0x04]).unwrap();
        assert_eq!(cps.power_w, 250);
        assert_eq!(cps.crank_revs, Some(10));
        assert_eq!(cps.crank_event_time_1024, Some(1024));

        let csc = parse_csc_measurement(&[0x03, 0x10, 0, 0, 0, 0x00, 0x02, 0x05, 0x00, 0x20, 0x01]).unwrap();
        assert_eq!(csc.wheel_revs, Some(16));
        assert_eq!(csc.crank_revs, Some(5));

        assert_eq!(parse_hrs_measurement(&[0x00, 142]), Some(142));
        assert_eq!(parse_hrs_measurement(&[0x01, 0xA4, 0x00]), Some(164));

        assert_eq!(encode_ftms_set_erg_power(220), [0x05, 0xDC, 0x00]);
        let sim = encode_ftms_set_sim_params(2.5, 0.004, 0.51);
        assert_eq!(sim[0], 0x11);
        assert_eq!(i16::from_le_bytes([sim[3], sim[4]]), 250);
        assert_eq!(sim[5], 40);
        assert_eq!(sim[6], 51);
    }
}
