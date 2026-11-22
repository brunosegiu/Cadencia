use std::sync::{Arc, Mutex};
use std::time::Duration;

use btleplug::api::{Central, Manager as _, Peripheral as _, ScanFilter, WriteType};
use btleplug::platform::Manager;
use futures_util::StreamExt;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

use crate::hardware::{
    encode_ftms_set_erg_power, encode_ftms_set_sim_params, parse_cps_measurement,
    parse_csc_measurement, parse_ftms_indoor_bike_data, parse_hrs_measurement,
    CPS_MEASUREMENT_UUID, CPS_SERVICE_UUID, CSC_MEASUREMENT_UUID, CSC_SERVICE_UUID,
    FTMS_CONTROL_POINT_UUID, FTMS_INDOOR_BIKE_DATA_UUID, FTMS_SERVICE_UUID, HRS_MEASUREMENT_UUID,
    HRS_SERVICE_UUID, HardwareProvider, TelemetrySample, TrainerMode,
};

struct SharedBleState {
    sample: TelemetrySample,
    status: String,
    scanning: bool,
    last_crank_revs: Option<u16>,
    last_crank_time: Option<u16>,
}

enum BleCommand {
    ScanAndConnect,
    SetMode(TrainerMode),
    Disconnect,
}

/// Native BLE cycling trainer/sensor provider powered by `btleplug` + `tokio`.
/// Supports Windows, Linux, macOS, and Android.
pub struct NativeBleProvider {
    runtime: Runtime,
    state: Arc<Mutex<SharedBleState>>,
    cmd_tx: mpsc::UnboundedSender<BleCommand>,
}

impl NativeBleProvider {
    pub fn new() -> Result<Self, String> {
        let runtime = Runtime::new().map_err(|e| format!("tokio runtime: {e}"))?;
        let state = Arc::new(Mutex::new(SharedBleState {
            sample: TelemetrySample {
                power_w: 0.0,
                cadence_rpm: 0.0,
                heart_rate_bpm: 0.0,
                speed_mps: None,
                connected: false,
                source_name: "BLE Idle".to_owned(),
            },
            status: "BLE Ready (Not Connected)".to_owned(),
            scanning: false,
            last_crank_revs: None,
            last_crank_time: None,
        }));
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<BleCommand>();
        let state_bg = Arc::clone(&state);

        runtime.spawn(async move {
            let mut active_peripheral: Option<btleplug::platform::Peripheral> = None;
            let mut control_point_char = None;
            let mut has_control = false;

            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    BleCommand::Disconnect => {
                        if let Some(p) = active_peripheral.take() {
                            let _ = p.disconnect().await;
                        }
                        control_point_char = None;
                        has_control = false;
                        if let Ok(mut st) = state_bg.lock() {
                            st.sample.connected = false;
                            st.scanning = false;
                            st.status = "BLE Disconnected".to_owned();
                        }
                    }
                    BleCommand::SetMode(mode) => {
                        if let (Some(p), Some(cp)) = (&active_peripheral, &control_point_char) {
                            if !has_control {
                                let _ = p.write(cp, &[0x00], WriteType::WithResponse).await;
                                has_control = true;
                            }
                            let payload: Vec<u8> = match mode {
                                TrainerMode::Erg { target_watts } => {
                                    encode_ftms_set_erg_power(target_watts).to_vec()
                                }
                                TrainerMode::Sim { grade_pct, crr, cw } => {
                                    encode_ftms_set_sim_params(grade_pct, crr, cw).to_vec()
                                }
                            };
                            let _ = p.write(cp, &payload, WriteType::WithResponse).await;
                        }
                    }
                    BleCommand::ScanAndConnect => {
                        if let Ok(mut st) = state_bg.lock() {
                            st.scanning = true;
                            st.status = "Scanning BLE trainers/sensors...".to_owned();
                        }
                        let Ok(manager) = Manager::new().await else {
                            if let Ok(mut st) = state_bg.lock() {
                                st.scanning = false;
                                st.status = "BLE adapter unavailable".to_owned();
                            }
                            continue;
                        };
                        let Ok(adapters) = manager.adapters().await else {
                            continue;
                        };
                        let Some(adapter) = adapters.into_iter().next() else {
                            if let Ok(mut st) = state_bg.lock() {
                                st.scanning = false;
                                st.status = "No Bluetooth adapter found".to_owned();
                            }
                            continue;
                        };
                        let filter = ScanFilter {
                            services: vec![
                                FTMS_SERVICE_UUID,
                                CPS_SERVICE_UUID,
                                CSC_SERVICE_UUID,
                                HRS_SERVICE_UUID,
                            ],
                        };
                        if adapter.start_scan(filter).await.is_err() {
                            let _ = adapter.start_scan(ScanFilter::default()).await;
                        }
                        tokio::time::sleep(Duration::from_secs(4)).await;
                        let _ = adapter.stop_scan().await;

                        let Ok(peripherals) = adapter.peripherals().await else {
                            continue;
                        };

                        let mut chosen = None;
                        for p in peripherals {
                            if let Ok(Some(props)) = p.properties().await {
                                let matches_service = props.services.iter().any(|s| {
                                    *s == FTMS_SERVICE_UUID
                                        || *s == CPS_SERVICE_UUID
                                        || *s == CSC_SERVICE_UUID
                                        || *s == HRS_SERVICE_UUID
                                });
                                if matches_service {
                                    chosen = Some((p, props.local_name.unwrap_or_else(|| "BLE Trainer".into())));
                                    break;
                                }
                            }
                        }

                        let Some((peripheral, name)) = chosen else {
                            if let Ok(mut st) = state_bg.lock() {
                                st.scanning = false;
                                st.status = "No FTMS/CPS/HRS BLE devices found".to_owned();
                            }
                            continue;
                        };

                        if peripheral.connect().await.is_err() || peripheral.discover_services().await.is_err() {
                            if let Ok(mut st) = state_bg.lock() {
                                st.scanning = false;
                                st.status = format!("Failed to connect to {name}");
                            }
                            continue;
                        }

                        let chars = peripheral.characteristics();
                        let mut cp_char = None;
                        for ch in &chars {
                            if ch.uuid == FTMS_INDOOR_BIKE_DATA_UUID
                                || ch.uuid == CPS_MEASUREMENT_UUID
                                || ch.uuid == CSC_MEASUREMENT_UUID
                                || ch.uuid == HRS_MEASUREMENT_UUID
                            {
                                let _ = peripheral.subscribe(ch).await;
                            } else if ch.uuid == FTMS_CONTROL_POINT_UUID {
                                cp_char = Some(ch.clone());
                            }
                        }
                        control_point_char = cp_char;
                        active_peripheral = Some(peripheral.clone());

                        if let Ok(mut st) = state_bg.lock() {
                            st.scanning = false;
                            st.sample.connected = true;
                            st.sample.source_name = name.clone();
                            st.status = format!("Connected: {name}");
                        }

                        if let Ok(mut notifs) = peripheral.notifications().await {
                            let st_notifs = Arc::clone(&state_bg);
                            tokio::spawn(async move {
                                while let Some(n) = notifs.next().await {
                                    if let Ok(mut st) = st_notifs.lock() {
                                        if n.uuid == FTMS_INDOOR_BIKE_DATA_UUID {
                                            if let Some(d) = parse_ftms_indoor_bike_data(&n.value) {
                                                if let Some(w) = d.power_w {
                                                    st.sample.power_w = w.max(0) as f32;
                                                }
                                                if let Some(c) = d.cadence_rpm {
                                                    st.sample.cadence_rpm = c;
                                                }
                                                if let Some(hr) = d.heart_rate_bpm {
                                                    st.sample.heart_rate_bpm = hr as f32;
                                                }
                                                if let Some(kph) = d.speed_kph {
                                                    st.sample.speed_mps = Some(kph / 3.6);
                                                }
                                            }
                                        } else if n.uuid == CPS_MEASUREMENT_UUID {
                                            if let Some(cps) = parse_cps_measurement(&n.value) {
                                                st.sample.power_w = cps.power_w.max(0) as f32;
                                                if let (Some(revs), Some(t)) = (cps.crank_revs, cps.crank_event_time_1024) {
                                                    if let (Some(pr), Some(pt)) = (st.last_crank_revs, st.last_crank_time) {
                                                        let dr = revs.wrapping_sub(pr);
                                                        let dt = t.wrapping_sub(pt);
                                                        if dt > 0 && dr > 0 {
                                                            st.sample.cadence_rpm = (dr as f32) * 60.0 / (dt as f32 / 1024.0);
                                                        }
                                                    }
                                                    st.last_crank_revs = Some(revs);
                                                    st.last_crank_time = Some(t);
                                                }
                                            }
                                        } else if n.uuid == CSC_MEASUREMENT_UUID {
                                            if let Some(csc) = parse_csc_measurement(&n.value) {
                                                if let (Some(revs), Some(t)) = (csc.crank_revs, csc.crank_event_time_1024) {
                                                    if let (Some(pr), Some(pt)) = (st.last_crank_revs, st.last_crank_time) {
                                                        let dr = revs.wrapping_sub(pr);
                                                        let dt = t.wrapping_sub(pt);
                                                        if dt > 0 && dr > 0 {
                                                            st.sample.cadence_rpm = (dr as f32) * 60.0 / (dt as f32 / 1024.0);
                                                        }
                                                    }
                                                    st.last_crank_revs = Some(revs);
                                                    st.last_crank_time = Some(t);
                                                }
                                            }
                                        } else if n.uuid == HRS_MEASUREMENT_UUID {
                                            if let Some(hr) = parse_hrs_measurement(&n.value) {
                                                st.sample.heart_rate_bpm = hr as f32;
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    }
                }
            }
        });

        Ok(Self {
            runtime,
            state,
            cmd_tx,
        })
    }
}

impl HardwareProvider for NativeBleProvider {
    fn poll_telemetry(&mut self) -> TelemetrySample {
        let _ = &self.runtime;
        self.state.lock().map(|s| s.sample.clone()).unwrap_or_default()
    }

    fn set_trainer_mode(&mut self, mode: TrainerMode) {
        let _ = self.cmd_tx.send(BleCommand::SetMode(mode));
    }

    fn start_scan(&mut self) {
        let _ = self.cmd_tx.send(BleCommand::ScanAndConnect);
    }

    fn disconnect(&mut self) {
        let _ = self.cmd_tx.send(BleCommand::Disconnect);
    }

    fn status_text(&self) -> String {
        self.state.lock().map(|s| s.status.clone()).unwrap_or_default()
    }

    fn is_simulator(&self) -> bool {
        false
    }
}
