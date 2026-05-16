//! tinypt レンダラーの CLI エントリーポイント。
//!
//! コマンドライン引数を解析し、シーン構築 → レンダリング → 後処理 → 画像出力を実行する。
//! 対応フォーマット: PPM / HDR (Radiance) / EXR (ACEScg)

use tinypt::{apply_exposure_tonemap, build_scene, ckpt_path, denoise, render, resolve_pixels, write_image, write_image_pixels, RenderConfig, Tonemap};

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
    let scene = build_scene(&config);

    // 3. レンダリング実行（マルチスレッド・タイルベース）
    let output = render(&scene, &config, &ckpt_file)?;

    // 4. 後処理パイプライン
    //    - EXR/HDR はリニアのまま出力するためトーンマップ不要
    //    - PPM 出力時のみ露出補正・トーンマップを適用
    let lower = config.output_path.to_ascii_lowercase();
    let is_exr = lower.ends_with(".exr");
    let is_hdr = lower.ends_with(".hdr");
    let apply_tonemap = !is_exr && !is_hdr && (config.tonemap != Tonemap::None || config.exposure != 0.0);

    if config.denoise_enabled {
        // デノイズ有効: 蓄積バッファ → ピクセル解決 → OIDN デノイズ → トーンマップ → 出力
        let pixels = resolve_pixels(config.width, config.height, &output.acc, &output.acc_w);
        eprintln!("Denoising with Intel OIDN...");
        let mut denoised = denoise::denoise_oidn(&pixels, config.width, config.height);
        if apply_tonemap {
            denoised = apply_exposure_tonemap(&denoised, config.exposure, config.tonemap);
        }
        write_image_pixels(&config.output_path, config.width, config.height, &denoised)?;
    } else {
        // デノイズ無効: 蓄積バッファから直接出力
        if apply_tonemap {
            let pixels = resolve_pixels(config.width, config.height, &output.acc, &output.acc_w);
            let tm = apply_exposure_tonemap(&pixels, config.exposure, config.tonemap);
            write_image_pixels(&config.output_path, config.width, config.height, &tm)?;
        } else {
            write_image(&config.output_path, config.width, config.height, &output.acc, &output.acc_w)?;
        }
    }

    // 5. レンダリング完了後、チェックポイントファイルを削除
    let _ = std::fs::remove_file(&ckpt_file);
    Ok(())
}
