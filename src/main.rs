#[cfg(not(target_arch = "wasm32"))]
use cadence::renderer::CameraMode;
#[cfg(not(target_arch = "wasm32"))]
use cadence::ui::AppScreen;
#[cfg(not(target_arch = "wasm32"))]
use cadence::world::BiomeId;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let args: Vec<String> = std::env::args().collect();
        let mut screenshot_path: Option<String> = None;
        let mut explicit_biome: Option<BiomeId> = None;
        let mut direct_level: Option<BiomeId> = None;
        let mut explicit_screen: Option<AppScreen> = None;
        let mut width = 1280_u32;
        let mut height = 960_u32;
        let mut camera = CameraMode::Chase;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--screenshot" if i + 1 < args.len() => {
                    screenshot_path = Some(args[i + 1].clone());
                    i += 2;
                }
                "--level" if i + 1 < args.len() => {
                    match BiomeId::from_str_arg(&args[i + 1]) {
                        Some(b) => {
                            direct_level = Some(b);
                            explicit_biome = Some(b);
                        }
                        None => {
                            eprintln!(
                                "Unknown level '{}'. Valid levels: mountain, coastal, rural",
                                args[i + 1]
                            );
                            std::process::exit(1);
                        }
                    }
                    i += 2;
                }
                "--biome" if i + 1 < args.len() => {
                    if let Some(b) = BiomeId::from_str_arg(&args[i + 1]) {
                        explicit_biome = Some(b);
                    }
                    i += 2;
                }
                "--screen" if i + 1 < args.len() => {
                    if let Some(s) = AppScreen::from_str_arg(&args[i + 1]) {
                        explicit_screen = Some(s);
                    }
                    i += 2;
                }
                "--width" if i + 1 < args.len() => {
                    if let Ok(w) = args[i + 1].parse() {
                        width = w;
                    }
                    i += 2;
                }
                "--height" if i + 1 < args.len() => {
                    if let Ok(h) = args[i + 1].parse() {
                        height = h;
                    }
                    i += 2;
                }
                "--camera" if i + 1 < args.len() => {
                    camera = match args[i + 1].to_ascii_lowercase().as_str() {
                        "menu" | "main_menu" => CameraMode::MainMenu,
                        "close" => CameraMode::Close,
                        "wide" => CameraMode::Wide,
                        "side" => CameraMode::Side,
                        "back" => CameraMode::TurnaroundBack,
                        "34_back" | "back34" => CameraMode::TurnaroundBack34,
                        "34_front" | "front34" => CameraMode::TurnaroundFront34,
                        "fp" | "first_person" => CameraMode::FirstPerson,
                        _ => CameraMode::Chase,
                    };
                    i += 2;
                }
                _ => i += 1,
            }
        }

        if let Some(out) = screenshot_path {
            let screen = explicit_screen.unwrap_or(if direct_level.is_some() || explicit_biome.is_some() {
                AppScreen::InGame
            } else {
                AppScreen::MainMenu
            });
            let biome = explicit_biome.unwrap_or_else(BiomeId::random);
            if let Err(err) =
                cadence::render_headless_screenshot_screen(&out, biome, screen, width, height, camera)
            {
                eprintln!("Screenshot failed: {err}");
                std::process::exit(1);
            }
            println!("Saved screenshot to {out}");
        } else {
            cadence::run_windowed(direct_level);
        }
    }
}
