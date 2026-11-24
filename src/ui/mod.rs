use std::collections::HashMap;

use egui::{
    epaint::{Mesh, Vertex},
    Align2, Color32, ColorImage, CornerRadius, FontFamily, FontId, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::hardware::simulator::SimulatedHardware;
use crate::hardware::TelemetrySample;
use crate::physics::RiderPhysicsState;
use crate::renderer::{CameraMode, GraphicsSettings};
use crate::world::riders::{AiRider, RiderProfile};
use crate::world::{BiomeId, RouteSpline};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppScreen {
    MainMenu,
    WorldSelect,
    Settings,
    AvatarEditor,
    InGame,
}

impl AppScreen {
    pub fn from_str_arg(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "menu" | "main_menu" | "mainmenu" => Some(AppScreen::MainMenu),
            "world_select" | "worldselect" | "worlds" | "maps" => Some(AppScreen::WorldSelect),
            "settings" => Some(AppScreen::Settings),
            "gear" | "avatar" | "avatar_editor" => Some(AppScreen::AvatarEditor),
            "game" | "in_game" | "ingame" | "ride" => Some(AppScreen::InGame),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct HudActions {
    pub navigate_to: Option<AppScreen>,
    pub launch_biome: Option<BiomeId>,
    pub cycle_camera: Option<CameraMode>,
    pub request_screenshot: bool,
    pub start_ble_scan: bool,
    pub disconnect_ble: bool,
    pub use_simulator: bool,
}

pub struct HudState {
    pub menu_open: bool,
    pub help_toast_s: f32,
    avatars: HashMap<&'static str, TextureHandle>,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            menu_open: false,
            help_toast_s: 0.0,
            avatars: HashMap::new(),
        }
    }
}

fn generate_avatar_image(skin: [u8; 3], hair: [u8; 3], shirt: [u8; 3], hat: bool) -> ColorImage {
    let size = 48usize;
    let mut pixels = vec![Color32::TRANSPARENT; size * size];
    let c = (size as f32 - 1.0) * 0.5;
    let r_outer = 22.5_f32;
    let ink = Color32::from_rgb(38, 33, 28);
    let bg = Color32::from_rgb(225, 212, 186);

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            if d > r_outer {
                continue;
            }
            let mut col = bg;
            // Shoulders / jersey
            let s_dx = dx / 15.5;
            let s_dy = (dy - 17.5) / 10.5;
            let d_sh = s_dx * s_dx + s_dy * s_dy;
            if d_sh <= 1.0 {
                col = if d_sh > 0.78 {
                    ink
                } else {
                    Color32::from_rgb(shirt[0], shirt[1], shirt[2])
                };
            }
            // Neck
            if dx.abs() < 3.5 && (4.0..=11.0).contains(&dy) {
                col = if dx.abs() > 2.4 {
                    ink
                } else {
                    Color32::from_rgb(skin[0], skin[1], skin[2])
                };
            }
            // Face circle
            let f_dx = dx / 9.2;
            let f_dy = (dy + 2.2) / 10.6;
            let d_face = f_dx * f_dx + f_dy * f_dy;
            if d_face <= 1.0 {
                col = if d_face > 0.78 {
                    ink
                } else {
                    Color32::from_rgb(skin[0], skin[1], skin[2])
                };
            }
            // Hair / cap
            if hat {
                let h_dx = dx / 9.6;
                let h_dy = (dy + 8.5) / 6.2;
                let d_hat = h_dx * h_dx + h_dy * h_dy;
                if d_hat <= 1.0 && dy < -3.5 {
                    col = if d_hat > 0.72 {
                        ink
                    } else {
                        Color32::from_rgb(138, 120, 96)
                    };
                }
            } else {
                let h_dx = dx / 9.6;
                let h_dy = (dy + 7.5) / 7.2;
                let d_hair = h_dx * h_dx + h_dy * h_dy;
                if d_hair <= 1.0 && dy < -3.0 {
                    col = if d_hair > 0.74 {
                        ink
                    } else {
                        Color32::from_rgb(hair[0], hair[1], hair[2])
                    };
                }
            }
            // Eyes & smile
            if ((dx - 3.0).powi(2) + (dy + 2.0).powi(2)) < 1.8
                || ((dx + 3.0).powi(2) + (dy + 2.0).powi(2)) < 1.8
            {
                col = ink;
            }
            if dy > 2.2 && dy < 3.6 && dx.abs() < 2.8 {
                col = ink;
            }
            // Crisp outer ink ring
            if d > r_outer - 2.0 {
                col = ink;
            }
            pixels[y * size + x] = col;
        }
    }
    ColorImage {
        size: [size, size],
        source_size: egui::vec2(size as f32, size as f32),
        pixels,
    }
}

fn grade_to_color(grade_pct: f32) -> Color32 {
    if grade_pct < 0.8 {
        Color32::from_rgba_unmultiplied(64, 152, 198, 225)
    } else if grade_pct < 2.2 {
        Color32::from_rgba_unmultiplied(112, 186, 88, 225)
    } else if grade_pct < 4.5 {
        Color32::from_rgba_unmultiplied(226, 178, 58, 225)
    } else {
        Color32::from_rgba_unmultiplied(212, 74, 52, 225)
    }
}

#[allow(dead_code)]
fn draw_ink_button(
    ui: &mut Ui,
    center: Pos2,
    radius: f32,
    kind: u8,
    tooltip: &str,
) -> bool {
    let rect = Rect::from_center_size(center, Vec2::splat(radius * 2.0));
    let resp = ui.interact(rect, ui.scope_id().with(("ink_btn", kind, center.x as i32)), Sense::click());
    let hover = resp.hovered();
    let p = ui.painter();
    let ink = Color32::from_rgb(38, 32, 26);
    let fill = if hover {
        Color32::from_rgb(220, 196, 154)
    } else {
        Color32::from_rgb(203, 177, 136)
    };
    p.circle(center + Vec2::new(0.0, 2.0), radius, Color32::from_black_alpha(70), Stroke::NONE);
    p.circle(center, radius, fill, Stroke::new(2.0, ink));
    p.circle(center, radius - 2.5, Color32::TRANSPARENT, Stroke::new(1.0, Color32::from_rgb(232, 214, 180)));

    match kind {
        0 | 1 => {
            // Question mark
            p.text(
                center + Vec2::new(0.0, 1.0),
                Align2::CENTER_CENTER,
                "?",
                FontId::new(radius * 1.15, FontFamily::Proportional),
                Color32::from_rgb(104, 86, 62),
            );
        }
        2 => {
            // Map pin icon
            p.circle(center + Vec2::new(0.0, -3.0), 5.5, Color32::from_rgb(162, 118, 76), Stroke::new(1.5, ink));
            p.circle_filled(center + Vec2::new(0.0, -3.0), 2.2, Color32::from_rgb(235, 220, 192));
            p.line_segment(
                [center + Vec2::new(-3.5, 1.0), center + Vec2::new(0.0, 8.0)],
                Stroke::new(2.0, ink),
            );
            p.line_segment(
                [center + Vec2::new(3.5, 1.0), center + Vec2::new(0.0, 8.0)],
                Stroke::new(2.0, ink),
            );
        }
        3 => {
            // Folded biome map icon
            let r = Rect::from_center_size(center, Vec2::new(17.0, 14.0));
            p.rect(r, CornerRadius::same(2), Color32::from_rgb(122, 166, 108), Stroke::new(1.6, ink), StrokeKind::Middle);
            p.line_segment([Pos2::new(r.min.x + 5.5, r.min.y), Pos2::new(r.min.x + 5.5, r.max.y)], Stroke::new(1.2, ink));
            p.line_segment([Pos2::new(r.min.x + 11.5, r.min.y), Pos2::new(r.min.x + 11.5, r.max.y)], Stroke::new(1.2, ink));
        }
        _ => {
            // Settings gear
            p.circle(center, 6.5, Color32::TRANSPARENT, Stroke::new(3.0, Color32::from_rgb(102, 86, 66)));
            p.circle(center, 3.0, fill, Stroke::new(1.4, ink));
            for i in 0..8 {
                let a = (i as f32) * std::f32::consts::FRAC_PI_4;
                let p0 = center + Vec2::new(a.cos(), a.sin()) * 6.0;
                let p1 = center + Vec2::new(a.cos(), a.sin()) * 9.5;
                p.line_segment([p0, p1], Stroke::new(2.5, Color32::from_rgb(92, 76, 56)));
            }
        }
    }
    resp.on_hover_text(tooltip).clicked()
}

pub fn draw_hud(
    ctx: &egui::Context,
    hud_state: &mut HudState,
    biome: BiomeId,
    spline: &RouteSpline,
    telemetry: &TelemetrySample,
    physics: &RiderPhysicsState,
    player_profile: &RiderProfile,
    ai_riders: &[AiRider],
    sim: &mut SimulatedHardware,
    using_sim: bool,
    ble_status: &str,
    camera_mode: CameraMode,
) -> HudActions {
    let mut actions = HudActions::default();
    let screen = ctx.input(|i| i.content_rect());
    let w = screen.width().max(640.0);
    let h = screen.height().max(480.0);

    // Ensure inked portrait textures exist
    if !hud_state.avatars.contains_key(player_profile.name) {
        let img = generate_avatar_image(
            player_profile.avatar_skin,
            player_profile.avatar_hair,
            player_profile.avatar_shirt,
            true,
        );
        hud_state.avatars.insert(
            player_profile.name,
            ctx.load_texture(format!("avatar_{}", player_profile.name), img, TextureOptions::LINEAR),
        );
    }
    for ai in ai_riders {
        if !hud_state.avatars.contains_key(ai.profile.name) {
            let img = generate_avatar_image(
                ai.profile.avatar_skin,
                ai.profile.avatar_hair,
                ai.profile.avatar_shirt,
                false,
            );
            hud_state.avatars.insert(
                ai.profile.name,
                ctx.load_texture(format!("avatar_{}", ai.profile.name), img, TextureOptions::LINEAR),
            );
        }
    }

    let ink = Color32::from_rgb(36, 30, 24);
    let slate_bg = Color32::from_rgba_unmultiplied(36, 48, 58, 224);
    let slate_border = Color32::from_rgba_unmultiplied(192, 214, 224, 165);
    let brown_bg = Color32::from_rgba_unmultiplied(56, 42, 32, 218);

    // 1. TOP-LEFT POWER & BIOMETRICS CARD (no leaf dial, no heart rate, no gear footer)
    egui::Area::new(egui::Id::unique("hud_top_left"))
        .fade_in(false)
        .fixed_pos(Pos2::new(16.0, 14.0))
        .interactable(false)
        .show(ctx, |ui| {
            let card_rect = Rect::from_min_size(Pos2::new(16.0, 14.0), Vec2::new(178.0, 122.0));
            let p = ui.painter();
            p.rect(
                card_rect,
                CornerRadius::same(10),
                slate_bg,
                Stroke::new(1.5, slate_border),
                StrokeKind::Inside,
            );

            // Lightning icon
            let lx = card_rect.min.x + 28.0;
            let ly = card_rect.min.y + 26.0;
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(lx + 2.0, ly - 10.0),
                    Pos2::new(lx - 6.0, ly + 1.0),
                    Pos2::new(lx - 0.5, ly + 1.0),
                    Pos2::new(lx - 3.0, ly + 11.0),
                    Pos2::new(lx + 6.0, ly - 1.0),
                    Pos2::new(lx + 0.5, ly - 1.0),
                ],
                Color32::from_rgb(236, 198, 72),
                Stroke::NONE,
            ));

            // Wattage readout
            let watts = telemetry.power_w.round() as i32;
            p.text(
                Pos2::new(card_rect.min.x + 95.0, card_rect.min.y + 27.0),
                Align2::CENTER_CENTER,
                format!("{watts}"),
                FontId::new(29.0, FontFamily::Proportional),
                Color32::WHITE,
            );
            p.text(
                Pos2::new(card_rect.min.x + 132.0, card_rect.min.y + 30.0),
                Align2::LEFT_CENTER,
                "W",
                FontId::new(17.0, FontFamily::Proportional),
                Color32::from_rgb(218, 226, 232),
            );

            // 6-segment zone color bar
            let zone_cols = [
                Color32::from_rgb(138, 146, 152),
                Color32::from_rgb(66, 148, 204),
                Color32::from_rgb(92, 178, 96),
                Color32::from_rgb(230, 186, 56),
                Color32::from_rgb(228, 122, 48),
                Color32::from_rgb(206, 62, 54),
            ];
            let bar_x = card_rect.min.x + 10.0;
            let bar_y = card_rect.min.y + 49.0;
            let seg_w = (card_rect.width() - 20.0) / 6.0;
            for (i, &c) in zone_cols.iter().enumerate() {
                let r = Rect::from_min_size(Pos2::new(bar_x + i as f32 * seg_w, bar_y), Vec2::new(seg_w - 1.5, 6.0));
                p.rect_filled(r, CornerRadius::same(2), c);
            }

            // Metrics: W/KG, AVG, RPM (heart rate removed)
            let wkg = telemetry.power_w / sim.rider_weight_kg.max(35.0);
            let avg_w = physics.avg_power_w.round() as i32;
            let cad = telemetry.cadence_rpm.round() as i32;
            let cells = [
                (0, 0, format!("{wkg:.1}"), "W/KG"),
                (1, 0, format!("{avg_w}"), "AVG"),
                (0, 1, format!("{cad}"), "RPM"),
            ];
            for (cx_i, cy_i, val, lbl) in cells {
                let bx = card_rect.min.x + 18.0 + cx_i as f32 * 82.0;
                let by = card_rect.min.y + 72.0 + cy_i as f32 * 30.0;
                let ic = Pos2::new(bx + 6.0, by + 3.0);
                let w_col = Color32::from_rgb(232, 236, 240);
                match (cx_i, cy_i) {
                    (0, 0) => {
                        // Kettlebell / weight icon
                        p.circle(ic + Vec2::new(0.0, -3.5), 3.0, Color32::TRANSPARENT, Stroke::new(1.5, w_col));
                        p.rect_filled(Rect::from_center_size(ic + Vec2::new(0.0, 2.5), Vec2::new(9.0, 7.5)), CornerRadius::same(2), w_col);
                    }
                    (1, 0) => {
                        // Lightning icon
                        p.add(Shape::convex_polygon(
                            vec![
                                ic + Vec2::new(1.5, -6.0),
                                ic + Vec2::new(-4.0, 0.5),
                                ic + Vec2::new(0.0, 0.5),
                                ic + Vec2::new(-2.0, 6.5),
                                ic + Vec2::new(4.0, -0.5),
                                ic + Vec2::new(0.0, -0.5),
                            ],
                            w_col,
                            Stroke::NONE,
                        ));
                    }
                    _ => {
                        // Crank & pedal icon
                        p.circle(ic, 5.2, Color32::TRANSPARENT, Stroke::new(1.5, w_col));
                        p.line_segment([ic, ic + Vec2::new(5.5, -3.5)], Stroke::new(1.8, w_col));
                        p.circle_filled(ic, 2.0, w_col);
                    }
                };
                p.text(
                    Pos2::new(bx + 36.0, by - 1.0),
                    Align2::CENTER_CENTER,
                    val,
                    FontId::new(14.5, FontFamily::Proportional),
                    Color32::WHITE,
                );
                p.text(
                    Pos2::new(bx + 36.0, by + 11.5),
                    Align2::CENTER_CENTER,
                    lbl,
                    FontId::new(8.5, FontFamily::Proportional),
                    Color32::from_rgb(196, 206, 214),
                );
            }
        });

    // 2. TOP-CENTER TELEMETRY & ELEVATION PROFILE CARD
    egui::Area::new(egui::Id::unique("hud_top_center"))
        .fade_in(false)
        .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 14.0))
        .interactable(false)
        .show(ctx, |ui| {
            let cw = 376.0_f32;
            let ch = 138.0_f32;
            let rect = Rect::from_min_size(Pos2::new((w - cw) * 0.5, 14.0), Vec2::new(cw, ch));
            let p = ui.painter();
            p.rect(
                rect,
                CornerRadius::same(10),
                Color32::from_rgba_unmultiplied(54, 62, 69, 212),
                Stroke::new(1.4, Color32::from_rgba_unmultiplied(166, 178, 188, 150)),
                StrokeKind::Inside,
            );

            let kph = physics.speed_mps * 3.6;
            let km = physics.display_distance_m / 1000.0;
            let elev_gain = physics.elevation_gain_m.round() as i32;
            let total_s = physics.elapsed_s.round() as u32;
            let hrs = total_s / 3600;
            let mins = (total_s % 3600) / 60;
            let secs = total_s % 60;

            let y_top = rect.min.y + 20.0;
            let metrics = [
                (rect.min.x + 52.0, format!("{kph:.1}"), "KPH"),
                (rect.min.x + 138.0, format!("{km:.1}"), "KM"),
                (rect.min.x + 212.0, format!("{elev_gain}"), "M"),
                (rect.min.x + 310.0, format!("{hrs}:{mins:02}:{secs:02}"), ""),
            ];
            for (mx, val, unit) in metrics {
                p.text(
                    Pos2::new(mx, y_top),
                    Align2::RIGHT_CENTER,
                    val,
                    FontId::new(19.5, FontFamily::Proportional),
                    Color32::WHITE,
                );
                if !unit.is_empty() {
                    p.text(
                        Pos2::new(mx + 3.0, y_top + 2.0),
                        Align2::LEFT_CENTER,
                        unit,
                        FontId::new(10.0, FontFamily::Proportional),
                        Color32::from_rgb(206, 214, 220),
                    );
                }
            }

            // Thin orange-red progress bar
            let prog_rect = Rect::from_min_size(Pos2::new(rect.min.x + 12.0, rect.min.y + 36.0), Vec2::new(cw - 24.0, 3.5));
            p.rect_filled(prog_rect, CornerRadius::same(2), Color32::from_rgb(88, 96, 102));
            let progress_frac = (physics.distance_m / spline.total_length_m).fract().clamp(0.08, 0.96);
            p.rect_filled(
                Rect::from_min_size(prog_rect.min, Vec2::new(prog_rect.width() * progress_frac, 3.5)),
                CornerRadius::same(2),
                Color32::from_rgb(218, 96, 62),
            );

            // Gradient-colored Elevation Area Graph (windowed so rider sits ~70% across with uphill ahead)
            let graph_rect = Rect::from_min_max(
                Pos2::new(rect.min.x + 10.0, rect.min.y + 52.0),
                Pos2::new(rect.max.x - 10.0, rect.max.y - 8.0),
            );
            let prof = &spline.elevation_profile;
            if prof.len() > 2 {
                let win_len = spline.total_length_m * 0.42;
                let win_start = physics.distance_m - win_len * 0.70;
                let n_pts = 56usize;
                let mut elevs = Vec::with_capacity(n_pts);
                let mut grades = Vec::with_capacity(n_pts);
                let mut min_e = f32::MAX;
                let mut max_e = f32::MIN;
                for i in 0..n_pts {
                    let t = i as f32 / (n_pts - 1) as f32;
                    let s = spline.sample(win_start + t * win_len);
                    min_e = min_e.min(s.pos.y);
                    max_e = max_e.max(s.pos.y);
                    elevs.push(s.pos.y);
                    grades.push(s.grade_pct);
                }
                let span_e = (max_e - min_e).max(4.0);
                let mut mesh = Mesh::default();
                let mut top_pts = Vec::with_capacity(n_pts);
                for i in 0..n_pts {
                    let t = i as f32 / (n_pts - 1) as f32;
                    let norm_h = ((elevs[i] - min_e) / span_e).clamp(0.14, 0.94);
                    let gx = graph_rect.min.x + t * graph_rect.width();
                    let gy = graph_rect.max.y - norm_h * (graph_rect.height() - 10.0);
                    let col = grade_to_color(grades[i]);
                    top_pts.push((Pos2::new(gx, gy), col));
                }
                for i in 0..(n_pts - 1) {
                    let (p0, c0) = top_pts[i];
                    let (p1, c1) = top_pts[i + 1];
                    let b0 = Pos2::new(p0.x, graph_rect.max.y);
                    let b1 = Pos2::new(p1.x, graph_rect.max.y);
                    let base = mesh.vertices.len() as u32;
                    mesh.vertices.push(Vertex { pos: p0, uv: Pos2::ZERO, color: c0 });
                    mesh.vertices.push(Vertex { pos: p1, uv: Pos2::ZERO, color: c1 });
                    mesh.vertices.push(Vertex { pos: b1, uv: Pos2::ZERO, color: c1 });
                    mesh.vertices.push(Vertex { pos: b0, uv: Pos2::ZERO, color: c0 });
                    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
                p.add(Shape::mesh(mesh));

                // Live rider position needle at 70% & white grade callout pill
                let rider_t = 0.70_f32;
                let nx = graph_rect.min.x + rider_t * graph_rect.width();
                let cur_s = spline.sample(physics.distance_m);
                let cur_h = ((cur_s.pos.y - min_e) / span_e).clamp(0.14, 0.94);
                let ny = graph_rect.max.y - cur_h * (graph_rect.height() - 10.0);
                p.line_segment([Pos2::new(nx, ny), Pos2::new(nx, graph_rect.max.y)], Stroke::new(1.8, Color32::WHITE));
                let pill = Rect::from_center_size(Pos2::new(nx.clamp(graph_rect.min.x + 28.0, graph_rect.max.x - 28.0), ny - 10.0), Vec2::new(40.0, 17.0));
                p.rect(pill, CornerRadius::same(4), Color32::WHITE, Stroke::new(1.0, ink), StrokeKind::Inside);
                p.text(
                    pill.center(),
                    Align2::CENTER_CENTER,
                    format!("{:.1}%", physics.grade_pct.abs()),
                    FontId::new(10.5, FontFamily::Proportional),
                    ink,
                );
            }
        });

    // 3. TOP-RIGHT MINIMAP (top circular icons removed)
    egui::Area::new(egui::Id::unique("hud_top_right"))
        .fade_in(false)
        .fixed_pos(Pos2::new(w - 284.0, 14.0))
        .interactable(false)
        .show(ctx, |ui| {
            let map_rect = Rect::from_min_size(Pos2::new(w - 284.0, 14.0), Vec2::new(268.0, 114.0));
            let p = ui.painter();
            let land_col = Color32::from_rgb(142, 166, 98);
            let sea_col = Color32::from_rgb(64, 146, 198);
            p.rect(
                map_rect,
                CornerRadius::same(10),
                land_col,
                Stroke::new(1.5, Color32::from_rgba_unmultiplied(240, 232, 210, 190)),
                StrokeKind::Inside,
            );
            if biome == BiomeId::CoastalPromenade {
                let ocean_r = Rect::from_min_max(Pos2::new(map_rect.min.x + 30.0, map_rect.min.y + 38.0), map_rect.max);
                p.rect_filled(ocean_r, CornerRadius::same(8), sea_col);
            }

            // Zoomed scenic route corridor in Top-Right Minimap
            let s_here = spline.sample(physics.distance_m);
            let mut corr_pts = Vec::new();
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_z = f32::MAX;
            let mut max_z = f32::MIN;
            for step_i in -18..=56 {
                let s = spline.sample(physics.distance_m + step_i as f32 * 5.0);
                min_x = min_x.min(s.pos.x);
                max_x = max_x.max(s.pos.x);
                min_z = min_z.min(s.pos.z);
                max_z = max_z.max(s.pos.z);
                corr_pts.push(s.pos);
            }
            let span_x = (max_x - min_x).max(60.0);
            let span_z = (max_z - min_z).max(60.0);
            let to_map_world = |pos: glam::Vec3| -> Pos2 {
                Pos2::new(
                    map_rect.min.x + 24.0 + ((pos.x - min_x) / span_x) * (map_rect.width() - 48.0),
                    map_rect.min.y + 18.0 + ((pos.z - min_z) / span_z) * (map_rect.height() - 36.0),
                )
            };
            let pts_screen: Vec<Pos2> = corr_pts.iter().map(|&pt| to_map_world(pt)).collect();
            if pts_screen.len() > 2 {
                p.line(pts_screen.clone(), Stroke::new(7.5, Color32::from_rgb(232, 212, 178)));
                p.line(pts_screen.clone(), Stroke::new(4.2, Color32::from_rgb(192, 108, 72)));
            }
            let rp = to_map_world(s_here.pos);
            let dir_v = Vec2::new(s_here.tangent.x / span_x, s_here.tangent.z / span_z).normalized();
            let perp = Vec2::new(-dir_v.y, dir_v.x);
            p.add(Shape::convex_polygon(
                vec![rp + dir_v * 8.0, rp - dir_v * 5.0 + perp * 5.5, rp - dir_v * 5.0 - perp * 5.5],
                Color32::from_rgb(66, 148, 214),
                Stroke::new(1.2, Color32::WHITE),
            ));

            // Concentric orange start/target ring at corner of minimap
            let target_pt = Pos2::new(map_rect.min.x + 14.0, map_rect.max.y - 14.0);
            p.circle(target_pt, 5.5, Color32::from_rgb(224, 118, 62), Stroke::new(1.5, Color32::WHITE));
            p.circle_filled(target_pt, 2.2, Color32::WHITE);

            // Dark-red grade pill in top-right corner of minimap (with vector slope triangle)
            let gp = Rect::from_min_size(Pos2::new(map_rect.max.x - 74.0, map_rect.min.y + 6.0), Vec2::new(66.0, 20.0));
            p.rect_filled(gp, CornerRadius::same(4), Color32::from_rgb(142, 48, 44));
            let tri_x = gp.min.x + 8.0;
            let tri_y = gp.center().y + 4.0;
            p.add(Shape::convex_polygon(
                vec![Pos2::new(tri_x, tri_y), Pos2::new(tri_x + 12.0, tri_y), Pos2::new(tri_x + 12.0, tri_y - 8.0)],
                Color32::WHITE,
                Stroke::NONE,
            ));
            p.text(
                gp.center() + Vec2::new(9.0, 0.0),
                Align2::CENTER_CENTER,
                format!("{:.1}%", physics.grade_pct.abs()),
                FontId::new(11.5, FontFamily::Proportional),
                Color32::WHITE,
            );
        });

    // 4. BOTTOM-LEFT PROFILE + ORNATE PARCHMENT CLOSE-UP MINIMAP
    egui::Area::new(egui::Id::unique("hud_bottom_left"))
        .fade_in(false)
        .fixed_pos(Pos2::new(14.0, h - 252.0))
        .interactable(false)
        .show(ctx, |ui| {
            let p = ui.painter();
            let prof_rect = Rect::from_min_size(Pos2::new(16.0, h - 246.0), Vec2::new(172.0, 56.0));
            p.rect(
                prof_rect,
                CornerRadius::same(8),
                brown_bg,
                Stroke::new(1.2, Color32::from_rgba_unmultiplied(150, 128, 102, 160)),
                StrokeKind::Inside,
            );
            if let Some(tex) = hud_state.avatars.get(player_profile.name) {
                let av_rect = Rect::from_min_size(Pos2::new(prof_rect.min.x + 6.0, prof_rect.min.y + 6.0), Vec2::splat(42.0));
                p.image(tex.id(), av_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
            p.text(
                Pos2::new(prof_rect.min.x + 56.0, prof_rect.min.y + 15.0),
                Align2::LEFT_CENTER,
                "Nansen",
                FontId::new(14.5, FontFamily::Proportional),
                Color32::WHITE,
            );
            p.text(
                Pos2::new(prof_rect.min.x + 56.0, prof_rect.min.y + 31.0),
                Align2::LEFT_CENTER,
                format!("LVL {}", player_profile.level),
                FontId::new(10.5, FontFamily::Proportional),
                Color32::from_rgb(212, 198, 178),
            );
            let xp_bg = Rect::from_min_size(Pos2::new(prof_rect.min.x + 56.0, prof_rect.min.y + 42.0), Vec2::new(104.0, 5.5));
            p.rect_filled(xp_bg, CornerRadius::same(3), Color32::from_rgb(38, 30, 24));
            p.rect_filled(Rect::from_min_size(xp_bg.min, Vec2::new(76.0, 5.5)), CornerRadius::same(3), Color32::from_rgb(208, 152, 82));
            // Small gold turn/share arrow on right of profile card
            let ar_c = Pos2::new(prof_rect.max.x - 14.0, prof_rect.min.y + 22.0);
            p.add(Shape::convex_polygon(
                vec![ar_c + Vec2::new(-5.0, 3.0), ar_c + Vec2::new(1.0, -3.0), ar_c + Vec2::new(5.0, 0.0)],
                Color32::from_rgb(218, 168, 86),
                Stroke::NONE,
            ));

            // Ornate parchment close-up minimap
            let outer_map = Rect::from_min_size(Pos2::new(16.0, h - 180.0), Vec2::new(164.0, 164.0));
            let inner_map = outer_map.shrink(11.0);
            p.rect(outer_map, CornerRadius::same(6), Color32::from_rgb(223, 203, 164), Stroke::new(2.0, ink), StrokeKind::Inside);
            // Ornate corner scroll flourishes
            for corner in [
                outer_map.left_top(),
                outer_map.right_top(),
                outer_map.left_bottom(),
                outer_map.right_bottom(),
            ] {
                p.circle(corner, 8.5, Color32::from_rgb(212, 186, 140), Stroke::new(1.8, ink));
                p.circle_filled(corner, 2.5, Color32::from_rgb(132, 108, 76));
            }
            let bg_inner = if biome == BiomeId::CoastalPromenade {
                Color32::from_rgb(66, 148, 200)
            } else {
                Color32::from_rgb(154, 176, 108)
            };
            p.rect(inner_map, CornerRadius::same(3), bg_inner, Stroke::new(1.5, ink), StrokeKind::Inside);

            // Draw zoomed local curve around rider
            let s_cur = spline.sample(physics.distance_m);
            let mut local_pts = Vec::new();
            let scale_px = 1.65_f32;
            for step in -24..=24 {
                let s = spline.sample(physics.distance_m + step as f32 * 2.5);
                let rel = s.pos - s_cur.pos;
                let px = inner_map.center().x + rel.x * scale_px;
                let py = inner_map.center().y + rel.z * scale_px;
                if inner_map.contains(Pos2::new(px, py)) {
                    local_pts.push(Pos2::new(px, py));
                }
            }
            if local_pts.len() >= 2 {
                p.line(local_pts.clone(), Stroke::new(10.0, Color32::from_rgb(236, 218, 176)));
                p.line(local_pts, Stroke::new(6.0, Color32::from_rgb(196, 134, 88)));
            }
            let d2 = Vec2::new(s_cur.tangent.x, s_cur.tangent.z).normalized();
            let p2 = Vec2::new(-d2.y, d2.x);
            let mc = inner_map.center();
            p.add(Shape::convex_polygon(
                vec![mc + d2 * 9.0, mc - d2 * 6.0 + p2 * 6.0, mc - d2 * 6.0 - p2 * 6.0],
                Color32::from_rgb(216, 122, 54),
                Stroke::new(1.4, ink),
            ));
        });

    // 5. BOTTOM-RIGHT "RIDERS NEARBY" LEADERBOARD
    egui::Area::new(egui::Id::unique("hud_riders_nearby"))
        .fade_in(false)
        .fixed_pos(Pos2::new(w - 208.0, h - 294.0))
        .interactable(false)
        .show(ctx, |ui| {
            let lb_rect = Rect::from_min_size(Pos2::new(w - 208.0, h - 294.0), Vec2::new(192.0, 276.0));
            let p = ui.painter();
            p.rect(
                lb_rect,
                CornerRadius::same(8),
                Color32::from_rgba_unmultiplied(64, 48, 36, 216),
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(150, 128, 104, 120)),
                StrokeKind::Inside,
            );
            let hdr = Rect::from_min_size(lb_rect.min, Vec2::new(lb_rect.width(), 28.0));
            p.rect_filled(hdr, CornerRadius { nw: 8, ne: 8, sw: 0, se: 0 }, Color32::from_rgb(248, 245, 240));
            p.text(
                hdr.center(),
                Align2::CENTER_CENTER,
                "Riders Nearby",
                FontId::new(13.0, FontFamily::Proportional),
                ink,
            );

            // Build sorted nearby list relative to player
            let mut entries: Vec<(&'static str, f32, f32, bool)> = Vec::new();
            let p_wkg = telemetry.power_w / sim.rider_weight_kg.max(35.0);
            entries.push((player_profile.name, p_wkg, 0.0, true));
            for ai in ai_riders {
                let mut diff_m = ai.distance_m - physics.distance_m;
                if diff_m > spline.total_length_m * 0.5 {
                    diff_m -= spline.total_length_m;
                } else if diff_m < -spline.total_length_m * 0.5 {
                    diff_m += spline.total_length_m;
                }
                let gap_s = diff_m / physics.speed_mps.max(4.0);
                entries.push((ai.profile.name, ai.profile.w_per_kg, gap_s, false));
            }
            entries.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, &(name, wkg_val, gap_s, is_player)) in entries.iter().take(6).enumerate() {
                let ry = hdr.max.y + 4.0 + idx as f32 * 38.0;
                let row_rect = Rect::from_min_size(Pos2::new(lb_rect.min.x + 2.0, ry), Vec2::new(lb_rect.width() - 4.0, 36.0));
                if is_player {
                    p.rect_filled(row_rect, CornerRadius::same(4), Color32::from_rgb(47, 132, 168));
                }
                if let Some(tex) = hud_state.avatars.get(name) {
                    let av = Rect::from_min_size(Pos2::new(row_rect.min.x + 6.0, row_rect.min.y + 2.0), Vec2::splat(32.0));
                    p.image(tex.id(), av, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                }
                p.text(
                    Pos2::new(row_rect.min.x + 44.0, row_rect.min.y + 11.0),
                    Align2::LEFT_CENTER,
                    name.trim(),
                    FontId::new(12.0, FontFamily::Proportional),
                    Color32::WHITE,
                );
                p.text(
                    Pos2::new(row_rect.min.x + 44.0, row_rect.min.y + 26.0),
                    Align2::LEFT_CENTER,
                    format!("{wkg_val:.1} w/kg"),
                    FontId::new(10.0, FontFamily::Proportional),
                    Color32::from_rgb(218, 210, 200),
                );
                let gap_txt = if is_player {
                    "-0.03".to_owned()
                } else if gap_s.abs() < 0.05 {
                    "0.0*".to_owned()
                } else {
                    format!("{:+.2}", gap_s * 0.02)
                };
                p.text(
                    Pos2::new(row_rect.max.x - 8.0, row_rect.min.y + 24.0),
                    Align2::RIGHT_CENTER,
                    gap_txt,
                    FontId::new(10.5, FontFamily::Proportional),
                    Color32::from_rgb(228, 222, 214),
                );
            }
            let cy_v = lb_rect.max.y - 9.0;
            let cx_v = lb_rect.center().x;
            p.line(
                vec![Pos2::new(cx_v - 5.0, cy_v - 2.0), Pos2::new(cx_v, cy_v + 3.0), Pos2::new(cx_v + 5.0, cy_v - 2.0)],
                Stroke::new(1.8, Color32::from_rgb(224, 218, 210)),
            );
        });

    // 6. BOTTOM-CENTER ACTION BAR (custom vector icons, zero missing Unicode glyphs!)
    egui::Area::new(egui::Id::unique("hud_bottom_bar"))
        .fade_in(false)
        .anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -10.0))
        .show(ctx, |ui| {
            egui::Frame::NONE
                .fill(Color32::from_rgba_unmultiplied(64, 48, 36, 228))
                .corner_radius(CornerRadius::same(8))
                .stroke(Stroke::new(1.2, Color32::from_rgba_unmultiplied(156, 134, 108, 160)))
                .inner_margin(egui::Margin::symmetric(14, 6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 18.0;
                        let draw_action = |ui: &mut Ui, kind: u8, label: &str| -> bool {
                            let (rect, resp) = ui.allocate_exact_size(Vec2::new(124.0, 20.0), Sense::click());
                            let p = ui.painter();
                            let ic = Pos2::new(rect.min.x + 9.0, rect.center().y);
                            let col = if resp.hovered() { Color32::WHITE } else { Color32::from_rgb(235, 228, 218) };
                            match kind {
                                0 => {
                                    // Camera icon
                                    p.rect(Rect::from_center_size(ic, Vec2::new(13.0, 9.5)), CornerRadius::same(2), Color32::TRANSPARENT, Stroke::new(1.4, col), StrokeKind::Inside);
                                    p.circle(ic, 2.5, Color32::TRANSPARENT, Stroke::new(1.3, col));
                                }
                                1 => {
                                    // Crossed wrench/tool icon
                                    p.line_segment([ic + Vec2::new(-5.0, 5.0), ic + Vec2::new(5.0, -5.0)], Stroke::new(1.8, col));
                                    p.line_segment([ic + Vec2::new(-5.0, -5.0), ic + Vec2::new(5.0, 5.0)], Stroke::new(1.5, col));
                                }
                                _ => {
                                    // Camcorder icon
                                    p.rect(Rect::from_center_size(ic + Vec2::new(-1.5, 0.0), Vec2::new(9.5, 8.5)), CornerRadius::same(2), Color32::TRANSPARENT, Stroke::new(1.4, col), StrokeKind::Inside);
                                    p.add(Shape::convex_polygon(
                                        vec![ic + Vec2::new(3.5, 0.0), ic + Vec2::new(7.0, -3.5), ic + Vec2::new(7.0, 3.5)],
                                        col,
                                        Stroke::NONE,
                                    ));
                                }
                            }
                            p.text(
                                Pos2::new(rect.min.x + 22.0, rect.center().y),
                                Align2::LEFT_CENTER,
                                label,
                                FontId::new(12.2, FontFamily::Proportional),
                                col,
                            );
                            resp.clicked()
                        };
                        if draw_action(ui, 0, "Take Screenshot") {
                            actions.request_screenshot = true;
                        }
                        if draw_action(ui, 1, "Settings") {
                            hud_state.menu_open = !hud_state.menu_open;
                        }
                        if draw_action(ui, 2, "Camera View") {
                            actions.cycle_camera = Some(camera_mode.next());
                        }
                    });
                });
        });

    // 7. INTERACTIVE ACTION MENU / SIMULATOR / BLE DRAWER (no on-the-fly world switching)
    if hud_state.menu_open {
        egui::Window::new("Cadence Control & Trainer Hub")
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .default_width(390.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("Active World: {}", biome.name())).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("End Ride / Main Menu").clicked() {
                            hud_state.menu_open = false;
                            actions.navigate_to = Some(AppScreen::MainMenu);
                        }
                    });
                });

                ui.separator();
                ui.label(egui::RichText::new("Camera Angle").strong());
                ui.horizontal_wrapped(|ui| {
                    for cam in [
                        CameraMode::Chase,
                        CameraMode::Close,
                        CameraMode::Wide,
                        CameraMode::Side,
                        CameraMode::FirstPerson,
                    ] {
                        if ui.selectable_label(camera_mode == cam, cam.name()).clicked() {
                            actions.cycle_camera = Some(cam);
                        }
                    }
                });

                ui.separator();
                ui.label(egui::RichText::new("Hardware & BLE Smart Trainer").strong());
                ui.horizontal(|ui| {
                    if ui.selectable_label(using_sim, "Built-in Simulator").clicked() {
                        actions.use_simulator = true;
                    }
                    if ui.button("Scan & Connect BLE").clicked() {
                        actions.start_ble_scan = true;
                    }
                    if !using_sim && ui.button("Disconnect BLE").clicked() {
                        actions.disconnect_ble = true;
                    }
                });
                ui.small(format!("Status: {ble_status}"));

                ui.separator();
                ui.label(egui::RichText::new("Trainer Simulator Controls").strong());
                ui.add(egui::Slider::new(&mut sim.target_power_w, 0.0..=600.0).text("Power (W)"));
                ui.add(egui::Slider::new(&mut sim.target_cadence_rpm, 0.0..=130.0).text("Cadence (RPM)"));
                ui.add(egui::Slider::new(&mut sim.target_hr_bpm, 75.0..=195.0).text("Heart Rate (BPM)"));
                ui.add(egui::Slider::new(&mut sim.rider_weight_kg, 45.0..=115.0).text("Rider Weight (kg)"));
                ui.horizontal(|ui| {
                    ui.label(format!("Virtual Gear: {}", sim.gear));
                    if ui.button(" - ").clicked() {
                        sim.shift_gear(-1);
                    }
                    if ui.button(" + ").clicked() {
                        sim.shift_gear(1);
                    }
                    ui.checkbox(&mut sim.erg_mode, "ERG Mode");
                    ui.checkbox(&mut sim.micro_variation, "Live Noise");
                });
                if sim.erg_mode {
                    let mut erg_w = sim.erg_target_w as f32;
                    if ui.add(egui::Slider::new(&mut erg_w, 60.0..=500.0).text("ERG Target (W)")).changed() {
                        sim.erg_target_w = erg_w as u16;
                    }
                }

                ui.separator();
                let f = &physics.last_forces;
                ui.small(format!(
                    "Forces: Drive {:.0}N | Grav {:.0}N | Roll {:.1}N | Aero {:.0}N | Draft {:.0}%",
                    f.f_drive_n,
                    f.f_gravity_n,
                    f.f_rolling_n,
                    f.f_aero_n,
                    physics.draft_factor * 100.0
                ));
                ui.small("Hotkeys: C Camera · M/Esc Menu · ↑/↓ Watts · ←/→ Gears · F12 Screenshot");
                if ui.button("Close Menu").clicked() {
                    hud_state.menu_open = false;
                }
            });
    }

    actions
}

fn draw_outlined_text(
    p: &egui::Painter,
    pos: Pos2,
    align: Align2,
    text: &str,
    font: FontId,
    fill: Color32,
    outline: Color32,
    thickness: f32,
) {
    let dirs = [
        Vec2::new(-1.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, -1.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(-0.707, -0.707),
        Vec2::new(0.707, -0.707),
        Vec2::new(-0.707, 0.707),
        Vec2::new(0.707, 0.707),
    ];
    for d in dirs {
        p.text(pos + d * thickness, align, text, font.clone(), outline);
    }
    p.text(pos, align, text, font, fill);
}

fn draw_double_outlined_title(
    p: &egui::Painter,
    pos: Pos2,
    text: &str,
    font_size: f32,
    inner_fill: Color32,
    mid_outline: Color32,
    outer_ink: Color32,
) {
    let font = FontId::new(font_size, FontFamily::Proportional);
    let r_outer = (font_size * 0.055).max(3.5);
    let r_mid = (font_size * 0.032).max(2.0);
    for i in 0..16 {
        let a = (i as f32 / 16.0) * std::f32::consts::TAU;
        let d = Vec2::new(a.cos(), a.sin());
        p.text(pos + d * r_outer, Align2::CENTER_CENTER, text, font.clone(), outer_ink);
    }
    for i in 0..16 {
        let a = (i as f32 / 16.0) * std::f32::consts::TAU;
        let d = Vec2::new(a.cos(), a.sin());
        p.text(pos + d * r_mid, Align2::CENTER_CENTER, text, font.clone(), mid_outline);
    }
    p.text(pos, Align2::CENTER_CENTER, text, font, inner_fill);
}

fn draw_menu_card_frame(p: &egui::Painter, rect: Rect, hovered: bool, corner: u8) {
    let shadow_rect = rect.translate(Vec2::new(0.0, 6.0)).expand(2.0);
    p.rect_filled(
        shadow_rect,
        CornerRadius::same(corner + 2),
        Color32::from_black_alpha(115),
    );
    let fill = if hovered {
        Color32::from_rgba_unmultiplied(48, 46, 38, 228)
    } else {
        Color32::from_rgba_unmultiplied(34, 34, 28, 214)
    };
    let outer_ink = Color32::from_rgba_unmultiplied(18, 16, 14, 235);
    let inner_rim = if hovered {
        Color32::from_rgba_unmultiplied(235, 224, 196, 215)
    } else {
        Color32::from_rgba_unmultiplied(168, 164, 148, 155)
    };
    p.rect(
        rect,
        CornerRadius::same(corner),
        fill,
        Stroke::new(2.2, outer_ink),
        StrokeKind::Outside,
    );
    p.rect(
        rect.shrink(1.5),
        CornerRadius::same(corner.saturating_sub(1)),
        Color32::TRANSPARENT,
        Stroke::new(1.6, inner_rim),
        StrokeKind::Inside,
    );
}

fn draw_settings_cog_icon(p: &egui::Painter, center: Pos2, scale: f32) {
    let ink = Color32::from_rgb(28, 24, 20);
    let cream = Color32::from_rgb(236, 232, 222);
    let inner_ring_col = Color32::from_rgb(184, 178, 166);
    let hole_fill = Color32::from_rgb(44, 42, 36);

    // Drop shadow
    p.circle_filled(center + Vec2::new(0.0, 3.0 * scale), 26.0 * scale, Color32::from_black_alpha(95));

    // 8 chunky rounded gear teeth
    let teeth = 8;
    let r_in = 16.5 * scale;
    let r_out = 26.5 * scale;
    let half_ang = 0.20_f32;
    for i in 0..teeth {
        let a = (i as f32 / teeth as f32) * std::f32::consts::TAU;
        let a0 = a - half_ang;
        let a1 = a + half_ang;
        let a0_in = a - half_ang * 1.35;
        let a1_in = a + half_ang * 1.35;
        let pts = vec![
            center + Vec2::new(a0_in.cos(), a0_in.sin()) * r_in,
            center + Vec2::new(a0.cos(), a0.sin()) * r_out,
            center + Vec2::new(a1.cos(), a1.sin()) * r_out,
            center + Vec2::new(a1_in.cos(), a1_in.sin()) * r_in,
        ];
        p.add(Shape::convex_polygon(pts, cream, Stroke::new(2.0 * scale, ink)));
    }

    // Main gear wheel disc
    p.circle(center, 19.5 * scale, cream, Stroke::new(2.2 * scale, ink));
    // Inner recessed bevel ring
    p.circle(center, 12.2 * scale, inner_ring_col, Stroke::new(1.5 * scale, ink));
    // Center axle hole
    p.circle(center, 7.6 * scale, hole_fill, Stroke::new(2.0 * scale, ink));
}

fn draw_jersey_gear_icon(p: &egui::Painter, center: Pos2, scale: f32) {
    let ink = Color32::from_rgb(26, 22, 18);
    let blue = Color32::from_rgb(72, 146, 206);
    let red = Color32::from_rgb(214, 74, 68);
    let yellow = Color32::from_rgb(242, 202, 74);
    let collar_dark = Color32::from_rgb(54, 48, 44);

    // Subtle drop shadow
    p.circle_filled(center + Vec2::new(0.0, 4.0 * scale), 26.0 * scale, Color32::from_black_alpha(95));

    // Left & right blue sleeves
    let left_sleeve = vec![
        center + Vec2::new(-14.0, -19.0) * scale,
        center + Vec2::new(-27.0, -9.0) * scale,
        center + Vec2::new(-23.0, 4.0) * scale,
        center + Vec2::new(-15.0, 0.0) * scale,
    ];
    let right_sleeve = vec![
        center + Vec2::new(14.0, -19.0) * scale,
        center + Vec2::new(27.0, -9.0) * scale,
        center + Vec2::new(23.0, 4.0) * scale,
        center + Vec2::new(15.0, 0.0) * scale,
    ];
    p.add(Shape::convex_polygon(left_sleeve, blue, Stroke::new(2.0 * scale, ink)));
    p.add(Shape::convex_polygon(right_sleeve, blue, Stroke::new(2.0 * scale, ink)));

    // Golden-yellow sleeve cuff bands
    let left_cuff = vec![
        center + Vec2::new(-25.2, -4.0) * scale,
        center + Vec2::new(-27.0, -9.0) * scale,
        center + Vec2::new(-21.5, -13.0) * scale,
        center + Vec2::new(-19.5, -7.5) * scale,
    ];
    let right_cuff = vec![
        center + Vec2::new(25.2, -4.0) * scale,
        center + Vec2::new(27.0, -9.0) * scale,
        center + Vec2::new(21.5, -13.0) * scale,
        center + Vec2::new(19.5, -7.5) * scale,
    ];
    p.add(Shape::convex_polygon(left_cuff, yellow, Stroke::new(1.6 * scale, ink)));
    p.add(Shape::convex_polygon(right_cuff, yellow, Stroke::new(1.6 * scale, ink)));

    // Main jersey torso (cobalt blue sides)
    let torso_rect = Rect::from_center_size(
        center + Vec2::new(0.0, 4.0 * scale),
        Vec2::new(31.0 * scale, 44.0 * scale),
    );
    p.rect(
        torso_rect,
        CornerRadius::same((7.0 * scale) as u8),
        blue,
        Stroke::new(2.2 * scale, ink),
        StrokeKind::Middle,
    );

    // Center coral-red vertical panel
    let red_panel = Rect::from_center_size(
        center + Vec2::new(0.0, 4.0 * scale),
        Vec2::new(15.0 * scale, 43.0 * scale),
    );
    p.rect(
        red_panel,
        CornerRadius::same((4.0 * scale) as u8),
        red,
        Stroke::new(1.8 * scale, ink),
        StrokeKind::Middle,
    );

    // Red shoulder yoke top
    let yoke = Rect::from_center_size(
        center + Vec2::new(0.0, -12.5 * scale),
        Vec2::new(28.0 * scale, 10.0 * scale),
    );
    p.rect_filled(yoke, CornerRadius::same((4.0 * scale) as u8), red);
    p.rect_stroke(
        torso_rect,
        CornerRadius::same((7.0 * scale) as u8),
        Stroke::new(2.2 * scale, ink),
        StrokeKind::Middle,
    );

    // Collar opening + full front zipper line
    p.circle(
        center + Vec2::new(0.0, -18.0 * scale),
        5.5 * scale,
        collar_dark,
        Stroke::new(2.0 * scale, ink),
    );
    p.line_segment(
        [
            center + Vec2::new(0.0, -12.5 * scale),
            center + Vec2::new(0.0, 25.5 * scale),
        ],
        Stroke::new(2.0 * scale, ink),
    );
    // Small dark zipper pull tab near chest
    p.circle_filled(center + Vec2::new(0.0, -5.0 * scale), 2.3 * scale, ink);
}

fn draw_main_menu(ctx: &egui::Context) -> HudActions {
    let mut actions = HudActions::default();
    let screen = ctx.input(|i| i.content_rect());
    let w = screen.width().max(640.0);
    let h = screen.height().max(480.0);
    let scale_x = (w / 1280.0).clamp(0.65, 1.6);
    let scale_y = (h / 960.0).clamp(0.65, 1.6);
    let scale = scale_x.min(scale_y);

    egui::Area::new(egui::Id::unique("cadence_main_menu_cluster"))
        .fade_in(false)
        .fixed_pos(Pos2::ZERO)
        .show(ctx, |ui| {
            let _ = ui.allocate_rect(screen, Sense::hover());

            let x_left = 74.0 * scale_x;
            let y_top = 246.0 * scale_y;
            let ride_w = 456.0 * scale_x;
            let ride_h = 200.0 * scale_y;
            let ride_rect = Rect::from_min_size(Pos2::new(x_left, y_top), Vec2::new(ride_w, ride_h));

            let sub_gap_x = 26.0 * scale_x;
            let sub_gap_y = 44.0 * scale_y;
            let sub_w = (ride_w - sub_gap_x) * 0.5;
            let sub_h = 114.0 * scale_y;
            let sub_y = ride_rect.max.y + sub_gap_y;

            let settings_rect = Rect::from_min_size(Pos2::new(x_left, sub_y), Vec2::new(sub_w, sub_h));
            let gear_rect = Rect::from_min_size(
                Pos2::new(x_left + sub_w + sub_gap_x, sub_y),
                Vec2::new(sub_w, sub_h),
            );

            let ride_resp = ui.interact(ride_rect, ui.scope_id().with("btn_main_ride"), Sense::click());
            let settings_resp = ui.interact(
                settings_rect.expand2(Vec2::new(0.0, 22.0 * scale_y)),
                ui.scope_id().with("btn_main_settings"),
                Sense::click(),
            );
            let gear_resp = ui.interact(
                gear_rect.expand2(Vec2::new(0.0, 22.0 * scale_y)),
                ui.scope_id().with("btn_main_gear"),
                Sense::click(),
            );

            let p = ui.painter();

            // 1. Large top "RIDE" card
            draw_menu_card_frame(&p, ride_rect, ride_resp.hovered(), (18.0 * scale) as u8);
            draw_double_outlined_title(
                &p,
                ride_rect.center() + Vec2::new(0.0, 2.0 * scale_y),
                "RIDE",
                108.0 * scale,
                Color32::from_rgb(36, 36, 30),
                Color32::from_rgb(245, 242, 234),
                Color32::from_rgb(20, 18, 16),
            );

            // 2. Bottom-left "SETTINGS" card + straddling cog icon
            draw_menu_card_frame(&p, settings_rect, settings_resp.hovered(), (16.0 * scale) as u8);
            draw_settings_cog_icon(
                &p,
                Pos2::new(settings_rect.center().x, settings_rect.min.y + 3.0 * scale_y),
                scale,
            );
            draw_outlined_text(
                &p,
                Pos2::new(settings_rect.center().x, settings_rect.min.y + 68.0 * scale_y),
                Align2::CENTER_CENTER,
                "SETTINGS",
                FontId::new(31.0 * scale, FontFamily::Proportional),
                Color32::from_rgb(244, 240, 232),
                Color32::from_rgb(22, 20, 16),
                2.0 * scale,
            );

            // 3. Bottom-right "GEAR" card + straddling color-blocked jersey icon
            draw_menu_card_frame(&p, gear_rect, gear_resp.hovered(), (16.0 * scale) as u8);
            draw_jersey_gear_icon(
                &p,
                Pos2::new(gear_rect.center().x, gear_rect.min.y + 3.0 * scale_y),
                scale,
            );
            draw_outlined_text(
                &p,
                Pos2::new(gear_rect.center().x, gear_rect.min.y + 68.0 * scale_y),
                Align2::CENTER_CENTER,
                "GEAR",
                FontId::new(31.0 * scale, FontFamily::Proportional),
                Color32::from_rgb(244, 240, 232),
                Color32::from_rgb(22, 20, 16),
                2.0 * scale,
            );

            if ride_resp.clicked() {
                actions.navigate_to = Some(AppScreen::WorldSelect);
            }
            if settings_resp.clicked() {
                actions.navigate_to = Some(AppScreen::Settings);
            }
            if gear_resp.clicked() {
                actions.navigate_to = Some(AppScreen::AvatarEditor);
            }
        });

    actions
}

fn draw_back_pill(ui: &mut Ui, pos: Pos2, scale: f32) -> bool {
    let rect = Rect::from_min_size(pos, Vec2::new(138.0 * scale, 46.0 * scale));
    let resp = ui.interact(rect, ui.scope_id().with("menu_back_pill"), Sense::click());
    let p = ui.painter();
    draw_menu_card_frame(&p, rect, resp.hovered(), (12.0 * scale) as u8);
    draw_outlined_text(
        &p,
        rect.center(),
        Align2::CENTER_CENTER,
        "BACK",
        FontId::new(21.0 * scale, FontFamily::Proportional),
        Color32::from_rgb(244, 240, 232),
        Color32::from_rgb(22, 20, 16),
        1.8 * scale,
    );
    resp.clicked()
}

fn draw_world_preview_banner(p: &egui::Painter, rect: Rect, biome: BiomeId) {
    let ink = Color32::from_rgb(32, 28, 22);
    match biome {
        BiomeId::MountainClimb => {
            p.rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(174, 198, 212));
            // Distant mountain ridge
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(rect.min.x, rect.max.y - 25.0),
                    Pos2::new(rect.min.x + rect.width() * 0.28, rect.min.y + 22.0),
                    Pos2::new(rect.min.x + rect.width() * 0.58, rect.min.y + 44.0),
                    Pos2::new(rect.min.x + rect.width() * 0.84, rect.min.y + 16.0),
                    Pos2::new(rect.max.x, rect.min.y + 38.0),
                    Pos2::new(rect.max.x, rect.max.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ],
                Color32::from_rgb(138, 126, 108),
                Stroke::new(1.8, ink),
            ));
            // Green alpine slope & winding road
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(rect.min.x, rect.max.y - 34.0),
                    Pos2::new(rect.min.x + rect.width() * 0.52, rect.max.y - 54.0),
                    Pos2::new(rect.max.x, rect.max.y - 22.0),
                    Pos2::new(rect.max.x, rect.max.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ],
                Color32::from_rgb(134, 158, 94),
                Stroke::new(1.8, ink),
            ));
        }
        BiomeId::CoastalPromenade => {
            p.rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(244, 168, 82));
            // Glowing setting sun disc
            p.circle_filled(
                Pos2::new(rect.min.x + rect.width() * 0.76, rect.min.y + rect.height() * 0.52),
                24.0,
                Color32::from_rgb(254, 232, 142),
            );
            // Sunset ocean water
            let sea = Rect::from_min_max(
                Pos2::new(rect.min.x, rect.min.y + rect.height() * 0.56),
                rect.max,
            );
            p.rect_filled(sea, CornerRadius::same(8), Color32::from_rgb(92, 138, 156));
            // Promenade seawall & road on left
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(rect.min.x, rect.min.y + rect.height() * 0.48),
                    Pos2::new(rect.min.x + rect.width() * 0.56, rect.min.y + rect.height() * 0.58),
                    Pos2::new(rect.min.x + rect.width() * 0.44, rect.max.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ],
                Color32::from_rgb(198, 166, 124),
                Stroke::new(1.8, ink),
            ));
        }
        BiomeId::RollingRuralGravel => {
            p.rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(194, 212, 220));
            // Rolling golden-green hills
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(rect.min.x, rect.min.y + 48.0),
                    Pos2::new(rect.min.x + rect.width() * 0.35, rect.min.y + 28.0),
                    Pos2::new(rect.min.x + rect.width() * 0.72, rect.min.y + 38.0),
                    Pos2::new(rect.max.x, rect.min.y + 24.0),
                    Pos2::new(rect.max.x, rect.max.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ],
                Color32::from_rgb(156, 174, 102),
                Stroke::new(1.8, ink),
            ));
            p.add(Shape::convex_polygon(
                vec![
                    Pos2::new(rect.min.x, rect.max.y - 26.0),
                    Pos2::new(rect.min.x + rect.width() * 0.48, rect.max.y - 44.0),
                    Pos2::new(rect.max.x, rect.max.y - 24.0),
                    Pos2::new(rect.max.x, rect.max.y),
                    Pos2::new(rect.min.x, rect.max.y),
                ],
                Color32::from_rgb(206, 182, 134),
                Stroke::new(1.8, ink),
            ));
        }
    }
    p.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(2.0, ink),
        StrokeKind::Inside,
    );
}

fn draw_world_select(ctx: &egui::Context) -> HudActions {
    let mut actions = HudActions::default();
    let screen = ctx.input(|i| i.content_rect());
    let w = screen.width().max(640.0);
    let h = screen.height().max(480.0);
    let scale_x = (w / 1280.0).clamp(0.65, 1.6);
    let scale_y = (h / 960.0).clamp(0.65, 1.6);
    let scale = scale_x.min(scale_y);

    egui::Area::new(egui::Id::unique("cadence_world_select"))
        .fade_in(false)
        .fixed_pos(Pos2::ZERO)
        .show(ctx, |ui| {
            let _ = ui.allocate_rect(screen, Sense::hover());
            let p = ui.painter().clone();

            // Subtle backdrop tint so cards pop cleanly over the live 3D world
            p.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(75));

            if draw_back_pill(ui, Pos2::new(42.0 * scale_x, 36.0 * scale_y), scale) {
                actions.navigate_to = Some(AppScreen::MainMenu);
            }

            draw_double_outlined_title(
                &p,
                Pos2::new(w * 0.5, 62.0 * scale_y),
                "SELECT WORLD",
                52.0 * scale,
                Color32::from_rgb(36, 36, 30),
                Color32::from_rgb(245, 242, 234),
                Color32::from_rgb(20, 18, 16),
            );

            let card_w = 356.0 * scale_x;
            let card_h = 540.0 * scale_y;
            let gap_x = 34.0 * scale_x;
            let total_w = card_w * 3.0 + gap_x * 2.0;
            let start_x = (w - total_w) * 0.5;
            let start_y = (h - card_h) * 0.5 + 24.0 * scale_y;

            for (idx, biome) in BiomeId::ALL.iter().copied().enumerate() {
                let cx = start_x + idx as f32 * (card_w + gap_x);
                let card_rect = Rect::from_min_size(Pos2::new(cx, start_y), Vec2::new(card_w, card_h));
                let resp = ui.interact(
                    card_rect,
                    ui.scope_id().with(("world_select_card", idx)),
                    Sense::click(),
                );
                let p = ui.painter();
                draw_menu_card_frame(&p, card_rect, resp.hovered(), (18.0 * scale) as u8);

                // Top landscape banner
                let banner_rect = Rect::from_min_size(
                    Pos2::new(card_rect.min.x + 16.0 * scale_x, card_rect.min.y + 16.0 * scale_y),
                    Vec2::new(card_w - 32.0 * scale_x, 148.0 * scale_y),
                );
                draw_world_preview_banner(&p, banner_rect, biome);

                // World Name & Subtitle
                draw_outlined_text(
                    &p,
                    Pos2::new(card_rect.center().x, banner_rect.max.y + 32.0 * scale_y),
                    Align2::CENTER_CENTER,
                    biome.name(),
                    FontId::new(27.0 * scale, FontFamily::Proportional),
                    Color32::from_rgb(246, 242, 232),
                    Color32::from_rgb(20, 18, 14),
                    2.0 * scale,
                );
                p.text(
                    Pos2::new(card_rect.center().x, banner_rect.max.y + 60.0 * scale_y),
                    Align2::CENTER_CENTER,
                    biome.subtitle(),
                    FontId::new(14.0 * scale, FontFamily::Proportional),
                    Color32::from_rgb(206, 200, 186),
                );

                // 3 Stat badges
                let (len_lbl, grade_lbl, surf_lbl) = biome.stats_summary();
                let badges = [len_lbl, grade_lbl, surf_lbl];
                let bw = (card_w - 44.0 * scale_x) / 3.0;
                for (bi, &btxt) in badges.iter().enumerate() {
                    let br = Rect::from_min_size(
                        Pos2::new(
                            card_rect.min.x + 16.0 * scale_x + bi as f32 * (bw + 6.0 * scale_x),
                            banner_rect.max.y + 82.0 * scale_y,
                        ),
                        Vec2::new(bw, 26.0 * scale_y),
                    );
                    p.rect(
                        br,
                        CornerRadius::same(6),
                        Color32::from_rgba_unmultiplied(56, 54, 46, 230),
                        Stroke::new(1.2, Color32::from_rgba_unmultiplied(178, 172, 156, 140)),
                        StrokeKind::Inside,
                    );
                    p.text(
                        br.center(),
                        Align2::CENTER_CENTER,
                        btxt,
                        FontId::new(10.5 * scale, FontFamily::Proportional),
                        Color32::from_rgb(238, 232, 218),
                    );
                }

                // Route Elevation Sparkline Box
                let graph_box = Rect::from_min_size(
                    Pos2::new(card_rect.min.x + 18.0 * scale_x, banner_rect.max.y + 124.0 * scale_y),
                    Vec2::new(card_w - 36.0 * scale_x, 128.0 * scale_y),
                );
                p.rect(
                    graph_box,
                    CornerRadius::same(10),
                    Color32::from_rgba_unmultiplied(24, 26, 24, 195),
                    Stroke::new(1.2, Color32::from_rgba_unmultiplied(142, 138, 124, 120)),
                    StrokeKind::Inside,
                );
                let sp = RouteSpline::for_biome(biome);
                let n_pts = 48usize;
                let mut min_e = f32::MAX;
                let mut max_e = f32::MIN;
                let mut samples = Vec::with_capacity(n_pts);
                for k in 0..n_pts {
                    let dist = (k as f32 / (n_pts - 1) as f32) * sp.total_length_m;
                    let s = sp.sample(dist);
                    min_e = min_e.min(s.pos.y);
                    max_e = max_e.max(s.pos.y);
                    samples.push((s.pos.y, s.grade_pct));
                }
                let span_e = (max_e - min_e).max(6.0);
                let mut line_pts = Vec::with_capacity(n_pts);
                let mut mesh = Mesh::default();
                let inner_g = graph_box.shrink2(Vec2::new(10.0 * scale_x, 14.0 * scale_y));
                for (k, &(ey, gr)) in samples.iter().enumerate() {
                    let t = k as f32 / (n_pts - 1) as f32;
                    let nh = ((ey - min_e) / span_e).clamp(0.12, 0.92);
                    let gx = inner_g.min.x + t * inner_g.width();
                    let gy = inner_g.max.y - nh * inner_g.height();
                    line_pts.push((Pos2::new(gx, gy), grade_to_color(gr)));
                }
                for k in 0..(n_pts - 1) {
                    let (p0, c0) = line_pts[k];
                    let (p1, c1) = line_pts[k + 1];
                    let b0 = Pos2::new(p0.x, inner_g.max.y);
                    let b1 = Pos2::new(p1.x, inner_g.max.y);
                    let base = mesh.vertices.len() as u32;
                    mesh.vertices.push(Vertex { pos: p0, uv: Pos2::ZERO, color: c0 });
                    mesh.vertices.push(Vertex { pos: p1, uv: Pos2::ZERO, color: c1 });
                    mesh.vertices.push(Vertex { pos: b1, uv: Pos2::ZERO, color: c1 });
                    mesh.vertices.push(Vertex { pos: b0, uv: Pos2::ZERO, color: c0 });
                    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
                p.add(Shape::mesh(mesh));

                // Launch CTA pill at bottom of card
                let btn_rect = Rect::from_min_size(
                    Pos2::new(card_rect.min.x + 24.0 * scale_x, card_rect.max.y - 68.0 * scale_y),
                    Vec2::new(card_w - 48.0 * scale_x, 48.0 * scale_y),
                );
                let btn_fill = if resp.hovered() {
                    Color32::from_rgb(218, 96, 62)
                } else {
                    Color32::from_rgb(66, 138, 178)
                };
                p.rect(
                    btn_rect,
                    CornerRadius::same(10),
                    btn_fill,
                    Stroke::new(2.0, Color32::from_rgb(24, 20, 16)),
                    StrokeKind::Inside,
                );
                draw_outlined_text(
                    &p,
                    btn_rect.center(),
                    Align2::CENTER_CENTER,
                    "RIDE WORLD",
                    FontId::new(21.0 * scale, FontFamily::Proportional),
                    Color32::WHITE,
                    Color32::from_rgb(22, 18, 14),
                    1.6 * scale,
                );

                if resp.clicked() {
                    actions.launch_biome = Some(biome);
                }
            }
        });

    actions
}

fn draw_settings_screen(
    ctx: &egui::Context,
    graphics: &mut GraphicsSettings,
    sim: &mut SimulatedHardware,
    using_sim: bool,
    ble_status: &str,
) -> HudActions {
    let mut actions = HudActions::default();
    let screen = ctx.input(|i| i.content_rect());
    let w = screen.width().max(640.0);
    let h = screen.height().max(480.0);
    let scale_x = (w / 1280.0).clamp(0.65, 1.6);
    let scale_y = (h / 960.0).clamp(0.65, 1.6);
    let scale = scale_x.min(scale_y);

    egui::Area::new(egui::Id::unique("cadence_settings_backdrop"))
        .fade_in(false)
        .fixed_pos(Pos2::ZERO)
        .show(ctx, |ui| {
            let _ = ui.allocate_rect(screen, Sense::hover());
            let p = ui.painter().clone();
            p.rect_filled(screen, CornerRadius::ZERO, Color32::from_black_alpha(85));
            if draw_back_pill(ui, Pos2::new(42.0 * scale_x, 36.0 * scale_y), scale) {
                actions.navigate_to = Some(AppScreen::MainMenu);
            }
            draw_double_outlined_title(
                &p,
                Pos2::new(w * 0.5, 62.0 * scale_y),
                "SETTINGS",
                52.0 * scale,
                Color32::from_rgb(36, 36, 30),
                Color32::from_rgb(245, 242, 234),
                Color32::from_rgb(20, 18, 16),
            );
        });

    egui::Window::new("Game & Graphics Settings")
        .anchor(Align2::CENTER_CENTER, Vec2::new(0.0, 20.0 * scale_y))
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .fixed_size(Vec2::new(720.0 * scale_x, 460.0 * scale_y))
        .frame(
            egui::Frame::NONE
                .fill(Color32::from_rgba_unmultiplied(34, 34, 28, 235))
                .corner_radius(CornerRadius::same(16))
                .stroke(Stroke::new(2.0, Color32::from_rgba_unmultiplied(178, 172, 156, 175)))
                .inner_margin(egui::Margin::symmetric(28, 24)),
        )
        .show(ctx, |ui| {
            ui.columns(2, |cols| {
                // Column 0: Graphics Settings
                cols[0].heading(egui::RichText::new("Graphics & Rendering").color(Color32::WHITE).strong());
                cols[0].add_space(10.0);
                cols[0].label(egui::RichText::new("Ligne Claire Ink Outline Weight").color(Color32::from_rgb(224, 218, 206)));
                cols[0].add(egui::Slider::new(&mut graphics.outline_scale, 0.25..=1.75).text("Scale"));
                cols[0].add_space(8.0);
                cols[0].label(egui::RichText::new("Scenery & Prop Draw Distance").color(Color32::from_rgb(224, 218, 206)));
                cols[0].add(egui::Slider::new(&mut graphics.prop_draw_distance_m, 250.0..=900.0).text("Meters"));
                cols[0].add_space(8.0);
                cols[0].label(egui::RichText::new("Camera Field of View Offset").color(Color32::from_rgb(224, 218, 206)));
                cols[0].add(egui::Slider::new(&mut graphics.fov_offset_deg, -10.0..=10.0).text("Degrees"));
                cols[0].add_space(10.0);
                cols[0].checkbox(&mut graphics.shadows_enabled, "Enable Directional Sun Shadows (2048x2048)");
                cols[0].add_space(6.0);
                cols[0].checkbox(&mut graphics.taa_enabled, "Enable Temporal Anti-Aliasing (TAA)");
                cols[0].add_space(14.0);
                if cols[0].button("Reset Graphics Defaults").clicked() {
                    *graphics = GraphicsSettings::default();
                }

                // Column 1: Game & Smart Trainer Settings
                cols[1].heading(egui::RichText::new("Game & Trainer Hardware").color(Color32::WHITE).strong());
                cols[1].add_space(10.0);
                cols[1].horizontal(|ui| {
                    if ui.selectable_label(using_sim, "Built-in Simulator").clicked() {
                        actions.use_simulator = true;
                    }
                    if ui.button("Scan & Connect BLE").clicked() {
                        actions.start_ble_scan = true;
                    }
                    if !using_sim && ui.button("Disconnect").clicked() {
                        actions.disconnect_ble = true;
                    }
                });
                cols[1].small(egui::RichText::new(format!("Status: {ble_status}")).color(Color32::from_rgb(200, 196, 184)));
                cols[1].add_space(10.0);
                cols[1].add(egui::Slider::new(&mut sim.rider_weight_kg, 45.0..=115.0).text("Rider Weight (kg)"));
                cols[1].add(egui::Slider::new(&mut sim.target_power_w, 0.0..=600.0).text("Sim Power (W)"));
                cols[1].add(egui::Slider::new(&mut sim.target_cadence_rpm, 0.0..=130.0).text("Sim Cadence (RPM)"));
                cols[1].add(egui::Slider::new(&mut sim.target_hr_bpm, 75.0..=195.0).text("Sim Heart Rate (BPM)"));
                cols[1].add_space(6.0);
                cols[1].horizontal(|ui| {
                    ui.label(format!("Virtual Gear: {}", sim.gear));
                    if ui.button(" - ").clicked() {
                        sim.shift_gear(-1);
                    }
                    if ui.button(" + ").clicked() {
                        sim.shift_gear(1);
                    }
                    ui.checkbox(&mut sim.erg_mode, "ERG Mode");
                    ui.checkbox(&mut sim.micro_variation, "Live Noise");
                });
                if sim.erg_mode {
                    let mut erg_w = sim.erg_target_w as f32;
                    if cols[1].add(egui::Slider::new(&mut erg_w, 60.0..=500.0).text("ERG Target (W)")).changed() {
                        sim.erg_target_w = erg_w as u16;
                    }
                }
            });

            ui.add_space(18.0);
            ui.separator();
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Back to Main Menu").clicked() {
                    actions.navigate_to = Some(AppScreen::MainMenu);
                }
            });
        });

    actions
}

fn draw_avatar_editor_screen(ctx: &egui::Context) -> HudActions {
    let mut actions = HudActions::default();
    let screen = ctx.input(|i| i.content_rect());
    let w = screen.width().max(640.0);
    let h = screen.height().max(480.0);
    let scale_x = (w / 1280.0).clamp(0.65, 1.6);
    let scale_y = (h / 960.0).clamp(0.65, 1.6);
    let scale = scale_x.min(scale_y);

    // Left blank as requested ("Gear opens the avatar editor screen. For now leave the avatar editor screen blank.")
    // with only a discrete Back button in the top-left corner so the player can return to the Main Menu.
    egui::Area::new(egui::Id::unique("cadence_avatar_editor_blank"))
        .fade_in(false)
        .fixed_pos(Pos2::ZERO)
        .show(ctx, |ui| {
            let _ = ui.allocate_rect(screen, Sense::hover());
            if draw_back_pill(ui, Pos2::new(42.0 * scale_x, 36.0 * scale_y), scale) {
                actions.navigate_to = Some(AppScreen::MainMenu);
            }
        });

    actions
}

#[allow(clippy::too_many_arguments)]
pub fn draw_ui(
    ctx: &egui::Context,
    screen_mode: AppScreen,
    hud_state: &mut HudState,
    graphics: &mut GraphicsSettings,
    biome: BiomeId,
    spline: &RouteSpline,
    telemetry: &TelemetrySample,
    physics: &RiderPhysicsState,
    player_profile: &RiderProfile,
    ai_riders: &[AiRider],
    sim: &mut SimulatedHardware,
    using_sim: bool,
    ble_status: &str,
    camera_mode: CameraMode,
) -> HudActions {
    match screen_mode {
        AppScreen::MainMenu => draw_main_menu(ctx),
        AppScreen::WorldSelect => draw_world_select(ctx),
        AppScreen::Settings => draw_settings_screen(ctx, graphics, sim, using_sim, ble_status),
        AppScreen::AvatarEditor => draw_avatar_editor_screen(ctx),
        AppScreen::InGame => draw_hud(
            ctx,
            hud_state,
            biome,
            spline,
            telemetry,
            physics,
            player_profile,
            ai_riders,
            sim,
            using_sim,
            ble_status,
            camera_mode,
        ),
    }
}
