pub mod assets;
pub mod hardware;
pub mod physics;
pub mod renderer;
pub mod ui;
pub mod world;

use std::sync::Arc;

use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::assets::AssetBank;
use crate::hardware::simulator::SimulatedHardware;
use crate::hardware::{HardwareProvider, TelemetrySample, TrainerMode};
use crate::physics::{compute_draft_factor, step_physics, PhysicsParams, RiderPhysicsState};
use crate::renderer::{CameraMode, GraphicsSettings, LigneClaireRenderer};
use crate::ui::{draw_ui, AppScreen, HudState};
use crate::world::riders::{default_player_profile, spawn_ai_peloton, step_ai_peloton, AiRider, RiderProfile};
use crate::world::terrain::{build_biome_world, StaticWorldGeometry};
use crate::world::{BiomeId, RouteSpline};

pub struct CadenceSim {
    pub screen: AppScreen,
    pub biome: BiomeId,
    pub spline: RouteSpline,
    pub world_geo: StaticWorldGeometry,
    pub physics_params: PhysicsParams,
    pub physics: RiderPhysicsState,
    pub player_profile: RiderProfile,
    pub ai_riders: Vec<AiRider>,
    pub camera_mode: CameraMode,
    pub graphics_settings: GraphicsSettings,
    pub sim_hw: SimulatedHardware,
    pub ble_hw: Option<Box<dyn HardwareProvider>>,
    pub use_simulator: bool,
    pub last_telemetry: TelemetrySample,
    pub hud_state: HudState,
    pub time_s: f32,
}

impl CadenceSim {
    pub fn new(biome: BiomeId, camera_mode: CameraMode) -> Self {
        Self::new_with_screen(biome, AppScreen::InGame, camera_mode)
    }

    pub fn new_with_screen(biome: BiomeId, screen: AppScreen, camera_mode: CameraMode) -> Self {
        let spline = RouteSpline::for_biome(biome);
        let world_geo = build_biome_world(biome, &spline);
        let start_s = spline.sample(spline.hero_start_dist_m);
        let physics_params = PhysicsParams {
            crr: biome.crr(),
            ..Default::default()
        };
        let physics = RiderPhysicsState::new_hero_default(
            spline.hero_start_dist_m,
            start_s.pos.y,
            start_s.grade_pct,
        );
        let player_profile = default_player_profile();
        let ai_riders = spawn_ai_peloton(spline.hero_start_dist_m, spline.total_length_m);
        let eff_camera = if screen == AppScreen::InGame {
            camera_mode
        } else {
            CameraMode::MainMenu
        };

        #[cfg(all(not(target_arch = "wasm32"), feature = "ble"))]
        let ble_hw: Option<Box<dyn HardwareProvider>> = hardware::ble_native::NativeBleProvider::new()
            .ok()
            .map(|b| Box::new(b) as Box<dyn HardwareProvider>);
        #[cfg(target_arch = "wasm32")]
        let ble_hw: Option<Box<dyn HardwareProvider>> =
            Some(Box::new(hardware::ble_web::WebBleProvider::new()));
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "ble")))]
        let ble_hw: Option<Box<dyn HardwareProvider>> = None;

        Self {
            screen,
            biome,
            spline,
            world_geo,
            physics_params,
            physics,
            player_profile,
            ai_riders,
            camera_mode: eff_camera,
            graphics_settings: GraphicsSettings::default(),
            sim_hw: SimulatedHardware::default(),
            ble_hw,
            use_simulator: true,
            last_telemetry: TelemetrySample::default(),
            hud_state: HudState::default(),
            time_s: 0.0,
        }
    }

    pub fn switch_biome(&mut self, next: BiomeId, renderer: Option<(&wgpu::Device, &mut LigneClaireRenderer)>) {
        let changed = self.biome != next;
        if changed {
            self.biome = next;
            self.spline = RouteSpline::for_biome(next);
            self.world_geo = build_biome_world(next, &self.spline);
            self.physics_params.crr = next.crr();
            if let Some((dev, ren)) = renderer {
                ren.upload_world(dev, &self.world_geo);
            }
        }
        let start_s = self.spline.sample(self.spline.hero_start_dist_m);
        self.physics = RiderPhysicsState::new_hero_default(
            self.spline.hero_start_dist_m,
            start_s.pos.y,
            start_s.grade_pct,
        );
        self.ai_riders = spawn_ai_peloton(self.spline.hero_start_dist_m, self.spline.total_length_m);
    }

    pub fn launch_world(&mut self, world: BiomeId, renderer: Option<(&wgpu::Device, &mut LigneClaireRenderer)>) {
        self.switch_biome(world, renderer);
        self.screen = AppScreen::InGame;
        self.camera_mode = CameraMode::Chase;
        self.hud_state.menu_open = false;
    }

    pub fn navigate_to(&mut self, next_screen: AppScreen, renderer: Option<(&wgpu::Device, &mut LigneClaireRenderer)>) {
        let was_in_game = self.screen == AppScreen::InGame;
        self.screen = next_screen;
        self.hud_state.menu_open = false;
        if next_screen == AppScreen::InGame {
            if self.camera_mode == CameraMode::MainMenu {
                self.camera_mode = CameraMode::Chase;
            }
        } else {
            self.camera_mode = CameraMode::MainMenu;
            if was_in_game && next_screen == AppScreen::MainMenu {
                let random_bg = BiomeId::random();
                self.switch_biome(random_bg, renderer);
            }
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.time_s += dt;
        if self.screen != AppScreen::InGame {
            return;
        }
        let sample = if self.use_simulator {
            self.sim_hw.poll_telemetry()
        } else if let Some(ble) = self.ble_hw.as_mut() {
            let s = ble.poll_telemetry();
            if s.connected {
                s
            } else {
                self.sim_hw.poll_telemetry()
            }
        } else {
            self.sim_hw.poll_telemetry()
        };
        self.last_telemetry = sample.clone();

        let route_s = self.spline.sample(self.physics.distance_m);
        let others: Vec<(f32, f32)> = self
            .ai_riders
            .iter()
            .map(|r| (r.distance_m, r.lateral_m))
            .collect();
        let draft = compute_draft_factor(
            self.physics.distance_m,
            0.0,
            &others,
            self.spline.total_length_m,
        );
        self.physics_params.rider_mass_kg = self.sim_hw.rider_weight_kg;

        step_physics(
            dt,
            sample.power_w,
            sample.cadence_rpm,
            route_s.grade_pct,
            route_s.curvature,
            draft,
            &self.physics_params,
            &mut self.physics,
        );
        step_ai_peloton(dt, &self.spline, &mut self.ai_riders);

        let mode = if self.sim_hw.erg_mode {
            TrainerMode::Erg {
                target_watts: self.sim_hw.erg_target_w,
            }
        } else {
            TrainerMode::Sim {
                grade_pct: route_s.grade_pct,
                crr: self.physics_params.crr,
                cw: 0.51,
            }
        };
        self.sim_hw.set_trainer_mode(mode);
        if !self.use_simulator {
            if let Some(ble) = self.ble_hw.as_mut() {
                ble.set_trainer_mode(mode);
            }
        }
    }

    pub fn ble_status(&self) -> String {
        if self.use_simulator {
            self.sim_hw.status_text()
        } else if let Some(ble) = &self.ble_hw {
            ble.status_text()
        } else {
            "BLE compiled without `ble` feature".to_owned()
        }
    }
}

struct WindowGpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: LigneClaireRenderer,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    sim: CadenceSim,
    last_frame: Instant,
    save_screenshot_flag: bool,
}

impl WindowGpuState {
    async fn init(window: Arc<Window>, initial_level: Option<BiomeId>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cadence_device"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .expect("GPU device");

        let config = surface
            .get_default_config(&adapter, width, height)
            .expect("surface config");
        surface.configure(&device, &config);

        let assets = AssetBank::load_embedded();
        let mut renderer = LigneClaireRenderer::new(&device, config.format, width, height, &assets);
        let (biome, screen, cam) = match initial_level {
            Some(b) => (b, AppScreen::InGame, CameraMode::Chase),
            None => (BiomeId::random(), AppScreen::MainMenu, CameraMode::MainMenu),
        };
        let sim = CadenceSim::new_with_screen(biome, screen, cam);
        renderer.upload_world(&device, &sim.world_geo);

        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        Self {
            window,
            surface,
            device,
            queue,
            config,
            renderer,
            egui_ctx,
            egui_winit,
            sim,
            last_frame: Instant::now(),
            save_screenshot_flag: false,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.renderer.resize(&self.device, width, height);
        }
    }

    fn render_and_present(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().clamp(0.001, 0.05);
        self.last_frame = now;
        self.sim.tick(dt);

        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let mut pending_actions = None;
        let mut full_output = self.egui_ctx.run_pass(raw_input, |ctx| {
            let ble_txt = self.sim.ble_status();
            let act = draw_ui(
                ctx,
                self.sim.screen,
                &mut self.sim.hud_state,
                &mut self.sim.graphics_settings,
                self.sim.biome,
                &self.sim.spline,
                &self.sim.last_telemetry,
                &self.sim.physics,
                &self.sim.player_profile,
                &self.sim.ai_riders,
                &mut self.sim.sim_hw,
                self.sim.use_simulator,
                &ble_txt,
                self.sim.camera_mode,
            );
            pending_actions = Some(act);
        });

        self.egui_winit
            .handle_platform_output(&self.window, full_output.platform_output);

        if let Some(act) = pending_actions {
            if let Some(world) = act.launch_biome {
                self.sim
                    .launch_world(world, Some((&self.device, &mut self.renderer)));
            }
            if let Some(next_screen) = act.navigate_to {
                self.sim
                    .navigate_to(next_screen, Some((&self.device, &mut self.renderer)));
            }
            if let Some(c) = act.cycle_camera {
                if self.sim.screen == AppScreen::InGame {
                    self.sim.camera_mode = c;
                }
            }
            if act.start_ble_scan {
                self.sim.use_simulator = false;
                if let Some(ble) = self.sim.ble_hw.as_mut() {
                    ble.start_scan();
                }
            }
            if act.disconnect_ble {
                if let Some(ble) = self.sim.ble_hw.as_mut() {
                    ble.disconnect();
                }
                self.sim.use_simulator = true;
            }
            if act.use_simulator {
                self.sim.use_simulator = true;
            }
            if act.request_screenshot {
                self.save_screenshot_flag = true;
            }
        }

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex) => tex,
            wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render_frame(
            &self.device,
            &self.queue,
            &view,
            self.config.width,
            self.config.height,
            self.sim.biome,
            &self.sim.spline,
            &self.sim.world_geo,
            self.sim.physics.distance_m,
            0.0,
            self.sim.physics.wheel_angle_rad,
            self.sim.physics.crank_angle_rad,
            self.sim.physics.lean_angle_rad,
            &self.sim.player_profile,
            &self.sim.ai_riders,
            self.sim.camera_mode,
            &self.sim.graphics_settings,
            self.sim.time_s,
            Some((&paint_jobs, &screen_desc)),
        );

        for id in &full_output.textures_delta.free {
            self.renderer.egui_renderer.free_texture(id);
        }
        full_output.textures_delta.clear();

        self.queue.present(frame);

        #[cfg(not(target_arch = "wasm32"))]
        if self.save_screenshot_flag {
            self.save_screenshot_flag = false;
            let _ = render_headless_screenshot_screen(
                "screenshots/capture.png",
                self.sim.biome,
                self.sim.screen,
                self.config.width,
                self.config.height,
                self.sim.camera_mode,
            );
        }
    }
}

struct CadenceApp {
    #[allow(dead_code)]
    initial_level: Option<BiomeId>,
    state: Option<WindowGpuState>,
}

impl ApplicationHandler for CadenceApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        #[allow(unused_mut)]
        let mut attrs = Window::default_attributes()
            .with_title("Cadence — Ligne Claire Indoor Cycling")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 960.0));

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("cadence-canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            if let Some(c) = canvas {
                attrs = attrs.with_canvas(Some(c));
            }
        }

        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(WindowGpuState::init(
                window,
                self.initial_level,
            )));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = window;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else {
            return;
        };
        let resp = st.egui_winit.on_window_event(&st.window, &event);
        if resp.consumed {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => st.resize(sz.width, sz.height),
            WindowEvent::KeyboardInput { event: kev, .. } if kev.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = kev.physical_key {
                    match code {
                        KeyCode::Escape => match st.sim.screen {
                            AppScreen::WorldSelect
                            | AppScreen::Settings
                            | AppScreen::AvatarEditor => {
                                st.sim.navigate_to(
                                    AppScreen::MainMenu,
                                    Some((&st.device, &mut st.renderer)),
                                );
                            }
                            AppScreen::InGame => {
                                st.sim.hud_state.menu_open = !st.sim.hud_state.menu_open;
                            }
                            AppScreen::MainMenu => {}
                        },
                        KeyCode::KeyC if st.sim.screen == AppScreen::InGame => {
                            st.sim.camera_mode = st.sim.camera_mode.next();
                        }
                        KeyCode::KeyM if st.sim.screen == AppScreen::InGame => {
                            st.sim.hud_state.menu_open = !st.sim.hud_state.menu_open;
                        }
                        KeyCode::ArrowUp if st.sim.screen == AppScreen::InGame => {
                            st.sim.sim_hw.target_power_w = (st.sim.sim_hw.target_power_w + 15.0).min(600.0);
                        }
                        KeyCode::ArrowDown if st.sim.screen == AppScreen::InGame => {
                            st.sim.sim_hw.target_power_w = (st.sim.sim_hw.target_power_w - 15.0).max(40.0);
                        }
                        KeyCode::ArrowRight if st.sim.screen == AppScreen::InGame => {
                            st.sim.sim_hw.shift_gear(1);
                        }
                        KeyCode::ArrowLeft if st.sim.screen == AppScreen::InGame => {
                            st.sim.sim_hw.shift_gear(-1);
                        }
                        KeyCode::F12 => st.save_screenshot_flag = true,
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                st.render_and_present();
                st.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(st) = &self.state {
            st.window.request_redraw();
        }
    }
}

pub fn run_windowed(initial_level: Option<BiomeId>) {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = CadenceApp {
        initial_level,
        state: None,
    };
    let _ = event_loop.run_app(&mut app);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn run_web() {
    console_error_panic_hook::set_once();
    use wasm_bindgen::JsCast;
    use winit::platform::web::WindowAttributesExtWebSys;

    let event_loop = EventLoop::new().expect("wasm event loop");
    #[allow(deprecated)]
    let window = {
        let mut attrs = Window::default_attributes().with_title("Cadence");
        if let Some(c) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("cadence-canvas"))
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        {
            attrs = attrs.with_canvas(Some(c));
        }
        Arc::new(event_loop.create_window(attrs).expect("wasm canvas window"))
    };

    let state_cell: std::rc::Rc<std::cell::RefCell<Option<WindowGpuState>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    let cell_clone = state_cell.clone();
    let win_clone = window.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let st = WindowGpuState::init(win_clone, None).await;
        *cell_clone.borrow_mut() = Some(st);
    });

    struct WasmApp {
        cell: std::rc::Rc<std::cell::RefCell<Option<WindowGpuState>>>,
        window: Arc<Window>,
    }
    impl ApplicationHandler for WasmApp {
        fn resumed(&mut self, _el: &ActiveEventLoop) {}
        fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
            let mut guard = self.cell.borrow_mut();
            let Some(st) = guard.as_mut() else {
                return;
            };
            let resp = st.egui_winit.on_window_event(&st.window, &ev);
            if resp.consumed {
                return;
            }
            match ev {
                WindowEvent::CloseRequested => el.exit(),
                WindowEvent::Resized(sz) => st.resize(sz.width, sz.height),
                WindowEvent::RedrawRequested => {
                    st.render_and_present();
                    st.window.request_redraw();
                }
                _ => {}
            }
        }
        fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
            self.window.request_redraw();
        }
    }
    let mut app = WasmApp {
        cell: state_cell,
        window,
    };
    let _ = event_loop.run_app(&mut app);
}

#[cfg(target_os = "android")]
#[no_mangle]
pub fn android_main(app: android_activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;
    let event_loop = EventLoop::with_user_event()
        .with_android_app(app)
        .build()
        .expect("android event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = CadenceApp {
        initial_level: None,
        state: None,
    };
    let _ = event_loop.run_app(&mut app);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render_headless_screenshot(
    out_path: &str,
    biome: BiomeId,
    width: u32,
    height: u32,
    camera_mode: CameraMode,
) -> Result<(), String> {
    render_headless_screenshot_screen(out_path, biome, AppScreen::InGame, width, height, camera_mode)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render_headless_screenshot_screen(
    out_path: &str,
    biome: BiomeId,
    screen: AppScreen,
    width: u32,
    height: u32,
    camera_mode: CameraMode,
) -> Result<(), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .map_err(|e| format!("adapter error: {e:?}"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cadence_headless_dev"),
        ..Default::default()
    }))
    .map_err(|e| format!("device error: {e:?}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let assets = AssetBank::load_embedded();
    let mut renderer = LigneClaireRenderer::new(&device, format, width, height, &assets);
    let mut sim = CadenceSim::new_with_screen(biome, screen, camera_mode);
    renderer.upload_world(&device, &sim.world_geo);

    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_out"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&Default::default());

    let egui_ctx = egui::Context::default();
    let pixels_per_point = (width as f32 / 1280.0).clamp(0.75, 2.0);
    // Warm up 2 frames so egui fonts, textures, and layout settle cleanly
    for frame_i in 0..2 {
        sim.last_telemetry = sim.sim_hw.poll_telemetry();
        let raw_input = egui::RawInput {
            time: Some(frame_i as f64 * 0.5),
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32 / pixels_per_point, height as f32 / pixels_per_point),
            )),
            ..Default::default()
        };
        let mut full_out = egui_ctx.run_pass(raw_input, |ctx| {
            let ble_txt = sim.ble_status();
            let _ = draw_ui(
                ctx,
                sim.screen,
                &mut sim.hud_state,
                &mut sim.graphics_settings,
                sim.biome,
                &sim.spline,
                &sim.last_telemetry,
                &sim.physics,
                &sim.player_profile,
                &sim.ai_riders,
                &mut sim.sim_hw,
                sim.use_simulator,
                &ble_txt,
                sim.camera_mode,
            );
        });
        let paint_jobs = egui_ctx.tessellate(full_out.shapes, pixels_per_point);
        for (id, deltas) in &full_out.textures_delta.set {
            for delta in deltas {
                renderer
                    .egui_renderer
                    .update_texture(&device, &queue, *id, delta);
            }
        }
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };
        renderer.render_frame(
            &device,
            &queue,
            &out_view,
            width,
            height,
            sim.biome,
            &sim.spline,
            &sim.world_geo,
            sim.physics.distance_m,
            0.0,
            sim.physics.wheel_angle_rad,
            sim.physics.crank_angle_rad,
            sim.physics.lean_angle_rad,
            &sim.player_profile,
            &sim.ai_riders,
            sim.camera_mode,
            &sim.graphics_settings,
            sim.time_s,
            Some((&paint_jobs, &screen_desc)),
        );
        for id in &full_out.textures_delta.free {
            renderer.egui_renderer.free_texture(id);
        }
        full_out.textures_delta.clear();
    }

    let bytes_per_row_unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = bytes_per_row_unpadded.div_ceil(align) * align;
    let buf_size = (padded_bytes_per_row * height) as u64;

    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot_readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("screenshot_copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|e| format!("map recv: {e}"))?
        .map_err(|e| format!("map failed: {e:?}"))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| format!("mapped_range: {e:?}"))?;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + bytes_per_row_unpadded as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    read_buf.unmap();

    if let Some(parent) = std::path::Path::new(out_path).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    image::save_buffer(out_path, &pixels, width, height, image::ColorType::Rgba8)
        .map_err(|e| format!("save_buffer: {e}"))?;
    Ok(())
}
