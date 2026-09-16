//! tinypt レンダラーの CLI エントリーポイント。
//!
//! コマンドライン引数を解析し、シーン構築 → レンダリング → 後処理 → 画像出力を実行する。
//! 対応フォーマット: PPM / HDR (Radiance) / EXR (ACEScg)

use tinypt::{build_default_scene, ckpt_path, denoise, load_scene, remove_stale_tmp, render, resolve_pixels, scene_hash, OutputFormat, OutputSettings, RenderConfig, Tonemap};

/// CLI で明示的に指定された値（シーンファイルの設定より優先させる）。
#[derive(Default)]
struct CliOverrides {
    /// `--spp` が指定された場合のサンプル数
    spp: Option<usize>,
}

/// フラグの値（次の引数）を取り出す。無ければ警告して `None`。
fn next_value(args: &mut impl Iterator<Item = String>, flag: &str, warnings: &mut Vec<String>) -> Option<String> {
    let v = args.next();
    if v.is_none() {
        warnings.push(format!("{} requires a value; ignored", flag));
    }
    v
}

/// フラグの値を数値として取り出す。値が無い・解釈できない場合は警告して `None`
/// （値の引数は消費する。設定は変更しない）。
fn next_number<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
    warnings: &mut Vec<String>,
) -> Option<T> {
    let v = next_value(args, flag, warnings)?;
    match v.parse::<T>() {
        Ok(n) => Some(n),
        Err(_) => {
            warnings.push(format!("invalid value '{}' for {}; ignored", v, flag));
            None
        }
    }
}

/// コマンドライン引数を解析して `RenderConfig` に反映する。
/// シーンファイルの設定より優先すべき CLI 明示値と、警告メッセージの一覧を返す。
///
/// 不正な値や未知のフラグはエラーにせず警告して無視する（終了コードは変えない）。
fn parse_args(args: impl IntoIterator<Item = String>, config: &mut RenderConfig) -> (CliOverrides, Vec<String>) {
    let mut overrides = CliOverrides::default();
    let mut warnings = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let w = &mut warnings;
        match arg.as_str() {
            "--spp" => {
                if let Some(n) = next_number::<usize>(&mut args, &arg, w) {
                    config.spp = n.max(1);
                    overrides.spp = Some(n.max(1));
                }
            }
            "--out" | "-o" => {
                if let Some(v) = next_value(&mut args, &arg, w) {
                    config.output_path = v;
                }
            }
            "--env" => {
                if let Some(v) = next_value(&mut args, &arg, w) {
                    config.env_map_path = Some(v);
                }
            }
            "--scene" => {
                if let Some(v) = next_value(&mut args, &arg, w) {
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
                if let Some(n) = next_number::<usize>(&mut args, &arg, w) {
                    config.adaptive_min_spp = n.max(1);
                }
            }
            "--adaptive-threshold" => {
                if let Some(n) = next_number::<f64>(&mut args, &arg, w) {
                    config.adaptive_threshold = n.max(0.0);
                }
            }
            "--seed" => {
                if let Some(n) = next_number::<u64>(&mut args, &arg, w) {
                    config.seed = n;
                }
            }
            "--tonemap" => {
                if let Some(v) = next_value(&mut args, &arg, w) {
                    match Tonemap::from_str(&v) {
                        Some(tm) => config.tonemap = tm,
                        None => w.push(format!("invalid value '{}' for --tonemap (expected none|aces); ignored", v)),
                    }
                }
            }
            "--exposure" => {
                if let Some(n) = next_number::<f64>(&mut args, &arg, w) {
                    config.exposure = n;
                }
            }
            "--checkpoint" => {
                config.checkpoint_enabled = true;
            }
            "--no-checkpoint" => {
                config.checkpoint_enabled = false;
            }
            "--checkpoint-every" => {
                if let Some(n) = next_number::<usize>(&mut args, &arg, w) {
                    config.checkpoint_every_tasks = n.max(1);
                    config.checkpoint_enabled = true;
                }
            }
            "--morton" => {
                config.morton_enabled = true;
            }
            "--no-morton" => {
                config.morton_enabled = false;
            }
            other => {
                w.push(format!("unknown argument '{}'; ignored", other));
            }
        }
    }
    (overrides, warnings)
}

fn main() -> std::io::Result<()> {
    // 1. 設定の初期化と引数解析
    let mut config = RenderConfig::default();
    let (overrides, warnings) = parse_args(std::env::args().skip(1), &mut config);
    for w in &warnings {
        eprintln!("Warning: {}", w);
    }

    // 2. シーン構築（カメラ・ジオメトリ・マテリアル・環境マップ）
    //    --scene 指定時は Mitsuba XML サブセットから解像度・spp・integrator 設定も読み込む。
    let scene = if let Some(path) = config.scene_path.clone() {
        load_scene(&path, &mut config)?
    } else {
        build_default_scene(&config)
    };
    // シーンファイルの設定より CLI 明示値を優先する（唯一の優先順位解決ポイント）。
    if let Some(spp) = overrides.spp {
        config.spp = spp;
    }
    // チェックポイントのキーは最終 config（シーン設定 + CLI 上書き後）から導出する。
    // 無効時はファイルに触れないので、参照ファイルの再読込コストも払わない。
    let ckpt_file = if config.checkpoint_enabled {
        config.scene_hash = scene_hash(&config)?;
        let path = ckpt_path(config.scene_hash);
        // 前回の強制終了で残った書き込み途中の一時ファイルを掃除する
        if remove_stale_tmp(&path) {
            eprintln!("Removed stale checkpoint temp file: {}.tmp", path);
        }
        Some(path)
    } else {
        None
    };

    // 3. レンダリング実行（マルチスレッド・タイルベース）
    let output = render(&scene, &config, ckpt_file.as_deref().unwrap_or(""))?;

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
    if let Some(f) = &ckpt_file {
        let _ = std::fs::remove_file(f);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> (RenderConfig, CliOverrides, Vec<String>) {
        let mut config = RenderConfig::default();
        let (o, w) = parse_args(args.iter().map(|s| s.to_string()), &mut config);
        (config, o, w)
    }

    /// 正しい引数では警告が出ず、値が反映される。
    #[test]
    fn valid_args_apply_without_warnings() {
        let (c, o, w) = parse(&["--spp", "16", "--seed", "3", "--exposure", "-1.5", "--tonemap", "none", "--no-denoise"]);
        assert!(w.is_empty(), "{:?}", w);
        assert_eq!(c.spp, 16);
        assert_eq!(o.spp, Some(16));
        assert_eq!(c.seed, 3);
        assert_eq!(c.exposure, -1.5);
        assert_eq!(c.tonemap, Tonemap::None);
        assert!(!c.denoise_enabled);
    }

    /// 数値として解釈できない値は警告し、設定は既定値のまま。値の引数は消費される。
    #[test]
    fn invalid_numbers_warn_and_keep_defaults() {
        let def = RenderConfig::default();
        let (c, o, w) = parse(&["--spp", "abc", "--adaptive-threshold", "x", "--seed", "-1", "--no-denoise"]);
        assert_eq!(w.len(), 3, "{:?}", w);
        assert!(w[0].contains("--spp") && w[0].contains("abc"));
        assert_eq!(c.spp, def.spp);
        assert_eq!(o.spp, None);
        assert_eq!(c.adaptive_threshold, def.adaptive_threshold);
        assert_eq!(c.seed, def.seed);
        // 値の直後のフラグは通常どおり解釈される
        assert!(!c.denoise_enabled);
    }

    /// 未知のフラグ・不正な tonemap・値の欠落は警告する。
    #[test]
    fn unknown_flags_bad_enum_and_missing_values_warn() {
        let (c, _, w) = parse(&["--sppp", "--tonemap", "filmic", "--out"]);
        assert_eq!(w.len(), 3, "{:?}", w);
        assert!(w[0].contains("--sppp"));
        assert!(w[1].contains("filmic"));
        assert!(w[2].contains("--out"));
        assert_eq!(c.tonemap, RenderConfig::default().tonemap);
        assert_eq!(c.output_path, RenderConfig::default().output_path);
    }
}
