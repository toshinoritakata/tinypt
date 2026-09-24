//! tinypt レンダラーの CLI エントリーポイント。
//!
//! コマンドライン引数を解析し、シーン構築 → レンダリング → 後処理 → 画像出力を実行する。
//! 対応フォーマット: PPM / HDR (Radiance) / EXR (ACEScg)

use tinypt::{build_default_scene, ckpt_path, denoise, load_scene, remove_stale_tmp, render, resolve_pixels, scene_hash_with_medium, OutputFormat, OutputSettings, RenderConfig, Tonemap};

/// CLI で明示的に指定された値（シーンファイルの設定より優先させる）。
#[derive(Default)]
struct CliOverrides {
    /// `--spp` が指定された場合のサンプル数
    spp: Option<usize>,
    /// `--width` / `--res` が指定された場合の幅
    width: Option<usize>,
    /// `--height` / `--res` が指定された場合の高さ
    height: Option<usize>,
    /// `-h` / `--help` が指定された（使い方を表示してレンダーせずに終了する）
    help: bool,
}

impl CliOverrides {
    /// CLI で解像度が明示されたなら、シーンファイルの `<film>` より優先する最終解像度。
    /// 片方だけの指定でも、もう片方は `config`（= 既定値かシーンファイルの値）を使えるよう
    /// 呼び出し側が埋める。
    fn resolution(&self) -> (Option<usize>, Option<usize>) {
        (self.width, self.height)
    }
}

/// 解像度の 1 辺の上限。これを超える値は打ち間違い（`--width 19201080` など）とみなして警告し無視する。
/// 実在のフィルムサイズを十分に超える値を選んである。
const MAX_DIMENSION: usize = 65536;

/// この画素数を超えたら「重い」と警告する（拒否はしない。RAM があるなら通す）。
/// 画素あたり約 60 B（蓄積 Color 24 B + 重み 8 B + 解決後の Color 24 B + 8bit 出力 3 B）。
const LARGE_PIXEL_COUNT: usize = 64 << 20; // 64M 画素 ≒ 3.8 GB

/// `--width` / `--height` の値を検証する。0・上限超えは警告して `None`（無視）。
fn valid_dimension(n: usize, flag: &str, warnings: &mut Vec<String>) -> Option<usize> {
    if n == 0 {
        warnings.push(format!("{} 0 is not valid; ignored", flag));
        None
    } else if n > MAX_DIMENSION {
        warnings.push(format!("{} {} exceeds the maximum of {}; ignored", flag, n, MAX_DIMENSION));
        None
    } else {
        Some(n)
    }
}

/// `--res WxH`（`1920x1080`）を解析する。`x` は小文字・大文字のどちらでもよい。
fn parse_resolution(v: &str, warnings: &mut Vec<String>) -> Option<(usize, usize)> {
    let mut it = v.split(['x', 'X']);
    let (a, b, rest) = (it.next(), it.next(), it.next());
    let bad = |warnings: &mut Vec<String>| {
        warnings.push(format!("invalid value '{}' for --res (expected WxH, e.g. 1920x1080); ignored", v));
        None
    };
    let (Some(a), Some(b), None) = (a, b, rest) else { return bad(warnings) };
    let (Ok(w), Ok(h)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) else {
        return bad(warnings);
    };
    // 幅・高さのどちらかが不正なら --res 全体を無視する（片側だけ効くと分かりにくいため）
    let (Some(w), Some(h)) = (valid_dimension(w, "--res width", warnings), valid_dimension(h, "--res height", warnings)) else {
        return None;
    };
    Some((w, h))
}

/// `-h` / `--help` で表示する使い方。
const USAGE: &str = "\
Usage: tinypt [OPTIONS]

Scene / output:
  --scene PATH               Mitsuba XML scene file (default: built-in scene, 1920x1080)
  --width N / --height N     Image size in pixels, 1..=65536 (overrides the scene file's <film>)
  --res WxH                  Both at once, e.g. --res 1920x1080 (same precedence; last flag wins)
  -o, --out PATH             Output file; format from extension: .ppm .hdr .exr (default: out.ppm)
  --env PATH                 HDR/EXR environment map (built-in scene only)
  --no-env                   Cancel an earlier --env

Sampling:
  --spp N                    Samples per pixel, N >= 1 (default: 512; overrides the scene file)
  --adaptive / --no-adaptive Adaptive sampling (default: off)
  --adaptive-min-spp N       Minimum samples per pixel for adaptive sampling, N >= 1 (default: 8)
  --adaptive-threshold X     Convergence threshold, relative std. dev., finite (default: 0.02)
  --seed N                   Random seed (default: 0)
  --morton / --no-morton     Morton-order tiles (default: on)

Post-processing (PPM only for tonemap/exposure):
  --denoise / --no-denoise   Intel OIDN denoising (default: on)
  --tonemap none|aces        Tone mapping (default: aces)
  --exposure EV              Exposure compensation, finite (default: 0.0)

Checkpoints:
  --checkpoint / --no-checkpoint  Save and resume checkpoints (default: off)
  --checkpoint-every N       Save every N tiles, N >= 1; also enables checkpoints (default: 128)

  -h, --help                 Show this help and exit

Invalid values and unknown arguments are reported as warnings and ignored.
";

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

/// フラグの値を有限の `f64` として取り出す。NaN / 無限大は警告して `None`。
fn next_finite(args: &mut impl Iterator<Item = String>, flag: &str, warnings: &mut Vec<String>) -> Option<f64> {
    let n = next_number::<f64>(args, flag, warnings)?;
    if n.is_finite() {
        Some(n)
    } else {
        warnings.push(format!("non-finite value '{}' for {}; ignored", n, flag));
        None
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
                    if n == 0 {
                        w.push("--spp 0 is not valid; using 1".to_string());
                    }
                    config.spp = n.max(1);
                    overrides.spp = Some(n.max(1));
                }
            }
            "--width" => {
                if let Some(n) = next_number::<usize>(&mut args, &arg, w) {
                    if let Some(n) = valid_dimension(n, &arg, w) {
                        config.width = n;
                        overrides.width = Some(n);
                    }
                }
            }
            "--height" => {
                if let Some(n) = next_number::<usize>(&mut args, &arg, w) {
                    if let Some(n) = valid_dimension(n, &arg, w) {
                        config.height = n;
                        overrides.height = Some(n);
                    }
                }
            }
            "--res" => {
                if let Some(v) = next_value(&mut args, &arg, w) {
                    if let Some((rw, rh)) = parse_resolution(&v, w) {
                        config.width = rw;
                        config.height = rh;
                        overrides.width = Some(rw);
                        overrides.height = Some(rh);
                    }
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
                    if n == 0 {
                        w.push("--adaptive-min-spp 0 is not valid; using 1".to_string());
                    }
                    config.adaptive_min_spp = n.max(1);
                }
            }
            "--adaptive-threshold" => {
                if let Some(n) = next_finite(&mut args, &arg, w) {
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
                if let Some(n) = next_finite(&mut args, &arg, w) {
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
                    // 0 を 1 に丸めると 1 タイルごとの保存（1080p で数百 GB の書き込み）になるため、
                    // 0 は無効値として警告し、間隔は変えない（チェックポイントは有効化する）。
                    if n == 0 {
                        w.push(format!(
                            "--checkpoint-every 0 is not valid; keeping {}",
                            config.checkpoint_every_tasks
                        ));
                    } else {
                        config.checkpoint_every_tasks = n;
                    }
                    config.checkpoint_enabled = true;
                }
            }
            "--morton" => {
                config.morton_enabled = true;
            }
            "--no-morton" => {
                config.morton_enabled = false;
            }
            "-h" | "--help" => {
                overrides.help = true;
            }
            other => {
                w.push(format!("unknown argument '{}'; ignored", other));
            }
        }
    }
    (overrides, warnings)
}

/// ビルド情報（バージョンと有効な feature）を 1 行ずつ返す。
///
/// feature は `cfg!` で判定するので、**このバイナリが実際にどうビルドされたか**を示す
/// （`Cargo.toml` の既定ではなく実体。`--no-default-features` ビルドでは denoise が無効と出る）。
fn build_info() -> String {
    let oidn = if cfg!(feature = "oidn") { "oidn (denoise)" } else { "no oidn (--denoise is a no-op)" };
    // render revision はバージョンから導出される数値なので、バージョンと並べて 1 行にまとめる
    // （同じバージョンなら同じ絵、という規約がそのまま読み取れるように）
    format!(
        "tinypt {} (render revision {}) — Monte Carlo path tracer\nFeatures: {}\n",
        env!("CARGO_PKG_VERSION"),
        tinypt::constants::RENDER_REVISION,
        oidn,
    )
}

fn main() -> std::io::Result<()> {
    // 引数なしで起動したら、レンダーせずにビルド情報と使い方を出す。
    // 既定シーンは 1920x1080 / 512spp で数分かかるので、「試しに叩いた」人を待たせない。
    // 既定シーンを描きたいときは `--scene` 無しで何かフラグを 1 つ付ける（例: `-o out.ppm`）。
    if std::env::args().nth(1).is_none() {
        print!("{}", build_info());
        println!();
        print!("{}", USAGE);
        println!("Rendering the built-in scene: tinypt -o out.ppm");
        return Ok(());
    }

    // 1. 設定の初期化と引数解析
    let mut config = RenderConfig::default();
    let (overrides, warnings) = parse_args(std::env::args().skip(1), &mut config);
    for w in &warnings {
        eprintln!("Warning: {}", w);
    }
    if overrides.help {
        print!("{}", build_info());
        println!();
        print!("{}", USAGE);
        return Ok(());
    }

    // 2. シーン構築（カメラ・ジオメトリ・マテリアル・環境マップ）。時間の内訳も出す
    //    --scene 指定時は Mitsuba XML サブセットから解像度・spp・integrator 設定も読み込む。
    //    CLI の解像度はローダーに渡す: センサーのアスペクト比が解像度から決まるので、
    //    読み込んだ後に width/height だけ差し替えると画角がずれる。
    let forced_resolution = overrides.resolution();
    let load_start = std::time::Instant::now();
    let scene = if let Some(path) = config.scene_path.clone() {
        load_scene(&path, &mut config, forced_resolution)?
    } else {
        // 組み込みシーンには `<film>` が無いので、parse_args が設定した config の解像度がそのまま最終値。
        build_default_scene(&config)
    };
    // シーンファイルの設定より CLI 明示値を優先する（唯一の優先順位解決ポイント）。
    // 解像度は load_scene の中で同じ規則で解決済み（組み込みシーンは上書き不要）。
    if let Some(spp) = overrides.spp {
        config.spp = spp;
    }
    // 読み込みの内訳。大きなシーンでは BVH 構築が支配的なので、描画時間と分けて見えるようにする
    {
        let total = load_start.elapsed();
        let st = &scene.load_stats;
        eprintln!(
            "Scene: {} ({} tris, {} instances, {} spheres, {} lights, {} textures)",
            config.scene_path.as_deref().unwrap_or("built-in"),
            scene.world.triangle_count(),
            scene.world.instances().len(),
            scene.world.spheres().len(),
            scene.world.lights().len(),
            scene.textures.len(),
        );
        eprintln!(
            "Loaded in {:.2}s (obj parse {:.2}s, mesh + BVH build {:.2}s, textures {:.2}s)",
            total.as_secs_f64(),
            st.obj_parse.as_secs_f64(),
            st.mesh_build.as_secs_f64(),
            st.texture_load.as_secs_f64(),
        );
    }

    if config.width.saturating_mul(config.height) > LARGE_PIXEL_COUNT {
        eprintln!(
            "Warning: {}x{} is {:.1}M pixels; the accumulation buffers alone need about {:.1} GB",
            config.width,
            config.height,
            (config.width * config.height) as f64 / (1 << 20) as f64,
            (config.width * config.height) as f64 * 60.0 / (1u64 << 30) as f64,
        );
    }
    // チェックポイントのキーは最終 config（シーン設定 + CLI 上書き後）から導出する。
    // 無効時はファイルに触れないので、参照ファイルの再読込コストも払わない。
    let ckpt_file = if config.checkpoint_enabled {
        config.scene_hash = scene_hash_with_medium(&config, scene.medium.as_ref())?;
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

    /// NaN / 無限大の露出・閾値は警告して無視する（NaN 露出で画像が真っ黒になるのを防ぐ）。
    #[test]
    fn non_finite_floats_warn_and_keep_defaults() {
        let def = RenderConfig::default();
        let (c, _, w) = parse(&["--exposure", "nan", "--adaptive-threshold", "inf", "--exposure", "-inf"]);
        assert_eq!(w.len(), 3, "{:?}", w);
        assert!(w.iter().all(|m| m.contains("non-finite")), "{:?}", w);
        assert_eq!(c.exposure, def.exposure);
        assert_eq!(c.adaptive_threshold, def.adaptive_threshold);
    }

    /// 0 の spp / min spp は 1 に丸めて警告、checkpoint-every 0 は間隔を変えずに警告する。
    #[test]
    fn zero_counts_warn() {
        let def = RenderConfig::default();
        let (c, o, w) = parse(&["--spp", "0", "--adaptive-min-spp", "0", "--checkpoint-every", "0"]);
        assert_eq!(w.len(), 3, "{:?}", w);
        assert_eq!(c.spp, 1);
        assert_eq!(o.spp, Some(1));
        assert_eq!(c.adaptive_min_spp, 1);
        assert_eq!(c.checkpoint_every_tasks, def.checkpoint_every_tasks);
        assert!(c.checkpoint_enabled);

        let (c, _, w) = parse(&["--checkpoint-every", "7"]);
        assert!(w.is_empty());
        assert_eq!(c.checkpoint_every_tasks, 7);
    }

    /// -h / --help は警告を出さずヘルプ要求として記録される。
    #[test]
    fn help_flag_is_recognized() {
        for flag in ["-h", "--help"] {
            let (_, o, w) = parse(&[flag]);
            assert!(o.help);
            assert!(w.is_empty(), "{:?}", w);
        }
        let (_, o, _) = parse(&["--spp", "4"]);
        assert!(!o.help);
    }
}
