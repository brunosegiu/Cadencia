#[cfg(not(target_arch = "wasm32"))]
use cadence::renderer::CameraMode;
#[cfg(not(target_arch = "wasm32"))]
use cadence::world::BiomeId;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let args: Vec<String> = std::env::args().collect();
        let mut screenshot_path: Option<String> = None;
        let mut biome = BiomeId::MountainClimb;
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
                "--biome" if i + 1 < args.len() => {
                    if let Some(b) = BiomeId::from_str_arg(&args[i + 1]) {
                        biome = b;
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
            if let Err(err) = cadence::render_headless_screenshot(&out, biome, width, height, camera) {
                eprintln!("Screenshot failed: {err}");
                std::process::exit(1);
            }
            println!("Saved screenshot to {out}");
        } else {
            cadence::run_windowed(biome);
        }
    }
}
