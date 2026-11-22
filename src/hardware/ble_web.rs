use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};

use crate::hardware::{
    encode_ftms_set_erg_power, encode_ftms_set_sim_params, parse_ftms_indoor_bike_data,
    parse_hrs_measurement, HardwareProvider, TelemetrySample, TrainerMode,
};

struct WebBleState {
    sample: TelemetrySample,
    status: String,
    control_point: Option<JsValue>,
    _closures: Vec<Closure<dyn FnMut(JsValue)>>,
}

/// Web Bluetooth API provider for browser WASM builds.
/// `start_scan()` must be invoked from a UI button click (browser user gesture requirement).
pub struct WebBleProvider {
    state: Rc<RefCell<WebBleState>>,
}

impl Default for WebBleProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl WebBleProvider {
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(WebBleState {
                sample: TelemetrySample {
                    power_w: 0.0,
                    cadence_rpm: 0.0,
                    heart_rate_bpm: 0.0,
                    speed_mps: None,
                    connected: false,
                    source_name: "Web Bluetooth Idle".to_owned(),
                },
                status: "Web Bluetooth Ready".to_owned(),
                control_point: None,
                _closures: Vec::new(),
            })),
        }
    }
}

async fn subscribe_characteristic(
    server: &JsValue,
    service_uuid: u32,
    char_uuid: u32,
    state: &Rc<RefCell<WebBleState>>,
    is_hr: bool,
) -> Option<JsValue> {
    let get_svc = Reflect::get(server, &JsValue::from_str("getPrimaryService")).ok()?;
    let svc_fn: Function = get_svc.dyn_into().ok()?;
    let svc_promise: Promise = svc_fn
        .call1(server, &JsValue::from_f64(service_uuid as f64))
        .ok()?
        .dyn_into()
        .ok()?;
    let svc = JsFuture::from(svc_promise).await.ok()?;

    let get_chr = Reflect::get(&svc, &JsValue::from_str("getCharacteristic")).ok()?;
    let chr_fn: Function = get_chr.dyn_into().ok()?;
    let chr_promise: Promise = chr_fn
        .call1(&svc, &JsValue::from_f64(char_uuid as f64))
        .ok()?
        .dyn_into()
        .ok()?;
    let chr = JsFuture::from(chr_promise).await.ok()?;

    if let Ok(start_notifs) = Reflect::get(&chr, &JsValue::from_str("startNotifications")) {
        if let Ok(start_fn) = start_notifs.dyn_into::<Function>() {
            if let Ok(p) = start_fn.call0(&chr) {
                if let Ok(prom) = p.dyn_into::<Promise>() {
                    let _ = JsFuture::from(prom).await;
                }
            }
        }
    }

    let st_cb = Rc::clone(state);
    let cb = Closure::wrap(Box::new(move |evt: JsValue| {
        let Ok(target) = Reflect::get(&evt, &JsValue::from_str("target")) else {
            return;
        };
        let Ok(value) = Reflect::get(&target, &JsValue::from_str("value")) else {
            return;
        };
        let Ok(buf) = Reflect::get(&value, &JsValue::from_str("buffer")) else {
            return;
        };
        let byte_len = Reflect::get(&value, &JsValue::from_str("byteLength"))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u32;
        let byte_off = Reflect::get(&value, &JsValue::from_str("byteOffset"))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u32;
        let arr = Uint8Array::new_with_byte_offset_and_length(&buf, byte_off, byte_len);
        let bytes = arr.to_vec();
        let mut st = st_cb.borrow_mut();
        if is_hr {
            if let Some(hr) = parse_hrs_measurement(&bytes) {
                st.sample.heart_rate_bpm = hr as f32;
            }
        } else if let Some(ftms) = parse_ftms_indoor_bike_data(&bytes) {
            if let Some(w) = ftms.power_w {
                st.sample.power_w = w.max(0) as f32;
            }
            if let Some(c) = ftms.cadence_rpm {
                st.sample.cadence_rpm = c;
            }
            if let Some(hr) = ftms.heart_rate_bpm {
                st.sample.heart_rate_bpm = hr as f32;
            }
            if let Some(kph) = ftms.speed_kph {
                st.sample.speed_mps = Some(kph / 3.6);
            }
        }
    }) as Box<dyn FnMut(JsValue)>);

    if let Ok(add_el) = Reflect::get(&chr, &JsValue::from_str("addEventListener")) {
        if let Ok(add_fn) = add_el.dyn_into::<Function>() {
            let _ = add_fn.call2(
                &chr,
                &JsValue::from_str("characteristicvaluechanged"),
                cb.as_ref().unchecked_ref(),
            );
        }
    }
    state.borrow_mut()._closures.push(cb);
    Some(chr)
}

impl HardwareProvider for WebBleProvider {
    fn poll_telemetry(&mut self) -> TelemetrySample {
        self.state.borrow().sample.clone()
    }

    fn set_trainer_mode(&mut self, mode: TrainerMode) {
        let Some(cp) = self.state.borrow().control_point.clone() else {
            return;
        };
        let payload: Vec<u8> = match mode {
            TrainerMode::Erg { target_watts } => encode_ftms_set_erg_power(target_watts).to_vec(),
            TrainerMode::Sim { grade_pct, crr, cw } => {
                encode_ftms_set_sim_params(grade_pct, crr, cw).to_vec()
            }
        };
        spawn_local(async move {
            let arr = Uint8Array::from(payload.as_slice());
            if let Ok(write_val) = Reflect::get(&cp, &JsValue::from_str("writeValue")) {
                if let Ok(write_fn) = write_val.dyn_into::<Function>() {
                    if let Ok(p) = write_fn.call1(&cp, &arr) {
                        if let Ok(prom) = p.dyn_into::<Promise>() {
                            let _ = JsFuture::from(prom).await;
                        }
                    }
                }
            }
        });
    }

    fn start_scan(&mut self) {
        let state = Rc::clone(&self.state);
        state.borrow_mut().status = "Requesting Web Bluetooth device...".to_owned();

        spawn_local(async move {
            let Some(window) = web_sys::window() else {
                return;
            };
            let nav: JsValue = window.navigator().into();
            let Ok(bt) = Reflect::get(&nav, &JsValue::from_str("bluetooth")) else {
                state.borrow_mut().status = "Web Bluetooth unsupported in this browser".to_owned();
                return;
            };
            if bt.is_undefined() || bt.is_null() {
                state.borrow_mut().status = "Web Bluetooth unsupported in this browser".to_owned();
                return;
            }

            let opts = Object::new();
            let filters = Array::new();
            for uuid in [0x1826_u32, 0x1818_u32, 0x180D_u32] {
                let f = Object::new();
                let svcs = Array::new();
                svcs.push(&JsValue::from_f64(uuid as f64));
                let _ = Reflect::set(&f, &JsValue::from_str("services"), &svcs);
                filters.push(&f);
            }
            let _ = Reflect::set(&opts, &JsValue::from_str("filters"), &filters);
            let opt_svcs = Array::new();
            opt_svcs.push(&JsValue::from_f64(0x1826 as f64));
            opt_svcs.push(&JsValue::from_f64(0x180D as f64));
            let _ = Reflect::set(&opts, &JsValue::from_str("optionalServices"), &opt_svcs);

            let Ok(req_dev) = Reflect::get(&bt, &JsValue::from_str("requestDevice")) else {
                return;
            };
            let Ok(req_fn) = req_dev.dyn_into::<Function>() else {
                return;
            };
            let Ok(dev_p) = req_fn.call1(&bt, &opts) else {
                return;
            };
            let Ok(dev_prom) = dev_p.dyn_into::<Promise>() else {
                return;
            };
            let Ok(device) = JsFuture::from(dev_prom).await else {
                state.borrow_mut().status = "Web Bluetooth selection cancelled".to_owned();
                return;
            };

            let name = Reflect::get(&device, &JsValue::from_str("name"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "BLE Trainer".to_owned());

            let Ok(gatt) = Reflect::get(&device, &JsValue::from_str("gatt")) else {
                return;
            };
            let Ok(conn) = Reflect::get(&gatt, &JsValue::from_str("connect")) else {
                return;
            };
            let Ok(conn_fn) = conn.dyn_into::<Function>() else {
                return;
            };
            let Ok(srv_p) = conn_fn.call0(&gatt) else {
                return;
            };
            let Ok(srv_prom) = srv_p.dyn_into::<Promise>() else {
                return;
            };
            let Ok(server) = JsFuture::from(srv_prom).await else {
                state.borrow_mut().status = format!("Failed connecting to {name}");
                return;
            };

            let _ = subscribe_characteristic(&server, 0x1826, 0x2AD2, &state, false).await;
            let _ = subscribe_characteristic(&server, 0x180D, 0x2A37, &state, true).await;
            if let Some(cp) = subscribe_characteristic(&server, 0x1826, 0x2AD9, &state, false).await {
                state.borrow_mut().control_point = Some(cp);
            }

            let mut st = state.borrow_mut();
            st.sample.connected = true;
            st.sample.source_name = name.clone();
            st.status = format!("Connected: {name}");
        });
    }

    fn disconnect(&mut self) {
        let mut st = self.state.borrow_mut();
        st.sample.connected = false;
        st.control_point = None;
        st.status = "Web Bluetooth Disconnected".to_owned();
    }

    fn status_text(&self) -> String {
        self.state.borrow().status.clone()
    }

    fn is_simulator(&self) -> bool {
        false
    }
}
