//! tinypt レンダラーの CLI エントリーポイント。
//!
//! コマンドライン引数を解析し、シーン構築 → レンダリング → 後処理 → 画像出力を実行する。
use tinypt::cli::{load_with_overrides, parse_args, USAGE};
use tinypt::{ckpt_path, denoise, remove_stale_tmp, render, resolve_pixels, scene_hash_with_medium, OutputFormat, OutputSettings, RenderConfig};

/// この画素数を超えたら「重い」と警告する（拒否はしない。RAM があるなら通す）。
/// 画素あたり約 60 B（蓄積 Color 24 B + 重み 8 B + 解決後の Color 24 B + 8bit 出力 3 B）。
const LARGE_PIXEL_COUNT: usize = 64 << 20; // 64M 画素 ≒ 3.8 GB

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
    let load_start = std::time::Instant::now();
    let scene = load_with_overrides(&mut config, &overrides)?;
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
            scene.shaders.textures.len(),
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

