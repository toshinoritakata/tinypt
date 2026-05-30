//! tinypt レンダラーの CLI エントリーポイント。
//!
//! コマンドライン引数を解析し、シーン構築 → レンダリング → 後処理 → 画像出力を実行する。
//! 対応フォーマット: PPM / HDR (Radiance) / EXR (ACEScg)

use tinypt::{build_scene, ckpt_path, denoise, load_scene, render, resolve_pixels, OutputFormat, OutputSettings, RenderConfig, Tonemap};

/// コマンドライン引数を解析して `RenderConfig` に反映する。
fn parse_args(config: &mut RenderConfig) {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spp" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        config.spp = n.max(1);
                    }
                }
            }
            "--out" | "-o" => {
                if let Some(v) = args.next() {
                    config.output_path = v;
                }
            }
            "--env" => {
                if let Some(v) = args.next() {
                    config.env_map_path = Some(v);
                }
            }
            "--scene" => {
                if let Some(v) = args.next() {
                    config.scene_path = Some(v);
                }
            }
            "--no-env" => {
                config.env_map_path = None;
            }
            "--denoise" => {
                config.denoise_enabled = true;
            }
            "--no-denoise" => {
                config.denoise_enabled = false;
            }
            "--adaptive" => {
                config.adaptive_enabled = true;
            }
            "--no-adaptive" => {
                config.adaptive_enabled = false;
            }
            "--adaptive-min-spp" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        config.adaptive_min_spp = n.max(1);
                    }
                }
            }
            "--adaptive-threshold" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<f64>() {
                        config.adaptive_threshold = n.max(0.0);
                    }
                }
            }
            "--seed" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<u64>() {
                        config.seed = n;
                    }
                }
            }
            "--tonemap" => {
                if let Some(v) = args.next() {
                    if let Some(tm) = Tonemap::from_str(&v) {
                        config.tonemap = tm;
                    }
                }
            }
            "--exposure" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<f64>() {
                        config.exposure = n;
                    }
                }
            }
            "--morton" => {
                config.morton_enabled = true;
            }
            "--no-morton" => {
                config.morton_enabled = false;
            }
            _ => {}
        }
    }
}

fn main() -> std::io::Result<()> {
    // 1. 設定の初期化と引数解析
    let mut config = RenderConfig::default();
    parse_args(&mut config);
    let ckpt_file = ckpt_path(config.scene_hash);

    // 2. シーン構築（カメラ・ジオメトリ・マテリアル・環境マップ）
    //    --scene 指定時は Mitsuba XML サブセットから、なければハードコードのデフォルトシーン。
    let scene = match &config.scene_path {
        Some(path) => load_scene(path, &config)?,
        None => build_scene(&config),
    };

    // 3. レンダリング実行（マルチスレッド・タイルベース）
    let output = render(&scene, &config, &ckpt_file)?;

    // 4. 後処理パイプライン
    //    蓄積バッファを一度だけリニア RGB に解決 → （オプションで）デノイズ →
    //    フォーマットが色空間・トーンマップ・ガンマを所有して出力。
    let mut pixels = resolve_pixels(config.width, config.height, &output.acc, &output.acc_w);
    if config.denoise_enabled {
        eprintln!("Denoising with Intel OIDN...");
        pixels = denoise::denoise_oidn(&pixels, config.width, config.height);
    }
    let settings = OutputSettings { exposure: config.exposure, tonemap: config.tonemap };
    OutputFormat::from_path(&config.output_path)
        .write(&config.output_path, config.width, config.height, &pixels, settings)?;

    // 5. レンダリング完了後、チェックポイントファイルを削除
    let _ = std::fs::remove_file(&ckpt_file);
    Ok(())
}
