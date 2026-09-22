//! マルチスレッド・タイルベース・レンダリングエンジン。
//!
//! 画像をタイルに分割し、ワーカースレッドで並列処理する。
//! 各タイルのサンプリング結果はメインスレッドで決定論的順序（タスクID順）にマージされ、
//! チェックポイント保存・進捗表示を行う。
//!
//! ## タイル処理の流れ
//! 1. 画像をタイル（デフォルト 16×16）に分割
//! 2. Morton 順序でソート（空間局所性によるキャッシュ効率向上）
//! 3. crossbeam チャネルでワーカースレッドにタスクを配布
//! 4. 各ワーカーがタイル内の全ピクセルをサンプリング
//! 5. 結果をタスクID順にマージ（アウトオブオーダー完了に対応）

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use crossbeam::scope;
use crossbeam_channel as chan;

use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::config::RenderConfig;
use crate::constants::ui::PROGRESS_INTERVAL_MS;
use crate::env::EnvMap;
use crate::integrator::{radiance, PathLimits, Surfaces};
use crate::material::Material;
use crate::math::Color;
use crate::ray::Camera;
use crate::rng::{seed_for, Rng};
use crate::scene::Scene;
use crate::task::{idx, Task, TileResult};
use crate::world::World;

/// レンダラーが出力する蓄積バッファ。
///
/// 各ピクセルの放射輝度の加算値 (`acc`) と重み合計 (`acc_w`) を保持する。
/// 最終ピクセル色 = acc[i] / acc_w[i] で求まる。
pub struct RenderOutput {
    /// 各ピクセルの放射輝度の加算値（リニア RGB）
    pub acc: Vec<Color>,
    /// 各ピクセルのサンプル重み合計
    pub acc_w: Vec<f64>,
}

impl RenderOutput {
    /// タイル結果を（アウトオブオーダー完了後、決定論的な順序で）アキュムレータへマージする。
    fn merge_tile(&mut self, r: &TileResult, width: usize) {
        let tile_w = r.x1 - r.x0;
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let global_idx = idx(x, y, width);
                let local_idx = (y - r.y0) * tile_w + (x - r.x0);
                self.acc[global_idx] = self.acc[global_idx] + r.sum[local_idx];
                self.acc_w[global_idx] += r.w[local_idx];
            }
        }
    }
}

/// タイル分割: 画像を `config.tile` サイズのブロックに分割しタスクリストを生成する。
/// `config.morton_enabled` なら空間局所性のために Morton 順序で並べ替える。
/// ID は（並べ替え後に）0 から昇順で振られ、`render` のレジューム機構が
/// 「次に処理すべきタスク ID」として扱う前提になっている。
fn build_tasks(config: &RenderConfig) -> Vec<Task> {
    let w = config.width;
    let h = config.height;
    let mut tasks: Vec<Task> = Vec::new();
    for y in (0..h).step_by(config.tile) {
        for x in (0..w).step_by(config.tile) {
            let x1 = (x + config.tile).min(w);
            let y1 = (y + config.tile).min(h);
            tasks.push(Task {
                id: 0,
                x0: x,
                y0: y,
                x1,
                y1,
                sample_start: 0,
                sample_end: config.spp,
            });
        }
    }
    if config.morton_enabled {
        let tiles_x = (w + config.tile - 1) / config.tile;
        let tiles_y = (h + config.tile - 1) / config.tile;
        tasks.sort_by_key(|t| morton2(t.x0 / config.tile, t.y0 / config.tile, tiles_x, tiles_y));
    }
    for (i, t) in tasks.iter_mut().enumerate() {
        t.id = i;
    }
    tasks
}

/// 1 ピクセル分のサンプリングを行い `(放射輝度合計, 使用サンプル数)` を返す。
///
/// `config.adaptive_enabled` なら Welford のオンライン分散で相対標準偏差が
/// 閾値を下回った時点で早期終了する（`min_spp` 到達後）。ジッター・スクリーン
/// 座標変換・カメラレイ生成・`radiance` 呼び出しは適応/固定の両方で共有される。
#[allow(clippy::too_many_arguments)]
fn sample_pixel(
    x: usize,
    y: usize,
    inv_w: f64,
    inv_h: f64,
    t: &Task,
    world: &World,
    mats: &[Material],
    surfaces: &Surfaces,
    env: Option<&EnvMap>,
    cam: &Camera,
    limits: PathLimits,
    config: &RenderConfig,
) -> (Color, f64) {
    let mut rng = Rng::new(seed_for(x as u32, y as u32, t.sample_start as u32, config.seed));
    let mut c = Color::new(0.0, 0.0, 0.0);

    let sample_once = |rng: &mut Rng| -> Color {
        let jx = rng.next_f64();
        let jy = rng.next_f64();
        let sx = (x as f64 + jx) * inv_w * 2.0 - 1.0;
        let sy = 1.0 - (y as f64 + jy) * inv_h * 2.0;
        let ray = cam.ray(sx, sy, rng);
        radiance(world, mats, surfaces, env, ray, rng, limits)
    };

    if config.adaptive_enabled {
        // 適応的サンプリング: 分散が閾値以下になったら早期終了
        let max_spp = (t.sample_end - t.sample_start).max(1);
        let min_spp = config.adaptive_min_spp.max(1).min(max_spp);
        // Welford のオンライン分散計算アルゴリズム
        let mut n: usize = 0;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for _s in t.sample_start..t.sample_end {
            let sample = sample_once(&mut rng);
            c = c + sample;
            n += 1;
            // Welford: 輝度ベースのオンライン分散更新
            let lum = sample.luminance();
            let delta = lum - mean;
            mean += delta / (n as f64);
            let delta2 = lum - mean;
            m2 += delta * delta2;
            // 最小サンプル数到達後、相対標準偏差で収束判定
            if n >= min_spp {
                let var = if n > 1 { m2 / ((n - 1) as f64) } else { f64::INFINITY };
                let denom = mean.abs().max(1e-4);
                let rel_std = var.sqrt() / denom;
                if rel_std < config.adaptive_threshold {
                    break;
                }
            }
        }
        (c, n as f64)
    } else {
        for _s in t.sample_start..t.sample_end {
            c = c + sample_once(&mut rng);
        }
        (c, (t.sample_end - t.sample_start) as f64)
    }
}

/// 読み込んだチェックポイントがこのレンダーで再開に使えるかを検証する。
///
/// バッファ長が画素数と一致し、`next_id` がタスク数以内のときだけ `Some` を返す。
/// `None` の場合、呼び出し側はレジューム位置を 0 のまま（最初から）レンダーする。
fn validate_resume(
    loaded: Option<(usize, Vec<Color>, Vec<f64>)>,
    n_pixels: usize,
    n_tasks: usize,
) -> Option<(usize, Vec<Color>, Vec<f64>)> {
    let (next_id, acc, acc_w) = loaded?;
    if acc.len() != n_pixels || acc_w.len() != n_pixels || next_id > n_tasks {
        return None;
    }
    Some((next_id, acc, acc_w))
}

/// Renders the scene and returns accumulation buffers.
///
/// ワーカースレッド数は `available_parallelism` に従う。結果はスレッド数に依存しない
/// （ピクセル単位のシードとタスク ID 順のマージによる）。
pub fn render(scene: &Scene, config: &RenderConfig, ckpt_file: &str) -> std::io::Result<RenderOutput> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    render_with_threads(scene, config, ckpt_file, threads)
}

/// ワーカースレッド数を指定してレンダーする（[`render`] の本体。テストでスレッド数非依存性を確かめるため分離）。
fn render_with_threads(
    scene: &Scene,
    config: &RenderConfig,
    ckpt_file: &str,
    threads: usize,
) -> std::io::Result<RenderOutput> {
    let threads = threads.max(1);
    let w = config.width;
    let h = config.height;
    let inv_w = 1.0 / (w as f64);
    let inv_h = 1.0 / (h as f64);

    // Accumulation buffers (may be restored from checkpoint)
    let mut out = RenderOutput {
        acc: vec![Color::new(0.0, 0.0, 0.0); w * h],
        acc_w: vec![0.0; w * h],
    };

    let ckpt_enabled = config.checkpoint_enabled && config.checkpoint_every_tasks > 0;

    let tasks = build_tasks(config);
    let tid = tasks.len();

    // Resume state
    let mut resume_next_id: usize = 0;
    if ckpt_enabled {
        let loaded = load_checkpoint(ckpt_file, config.scene_hash, w, h).ok().flatten();
        if let Some((next_id, acc0, acc_w0)) = validate_resume(loaded, w * h, tid) {
            resume_next_id = next_id;
            out.acc = acc0;
            out.acc_w = acc_w0;
            eprintln!(
                "Resumed from checkpoint: {} (next task id: {})",
                ckpt_file, resume_next_id
            );
        }
    }

    // タスク配布用チャネル (tx→rx) と結果回収用チャネル (rtx→rrx)
    let (tx, rx) = chan::unbounded::<Task>();
    let (rtx, rrx) = chan::unbounded::<TileResult>();

    // Send only tasks not yet merged (resume_next_id is the next expected merge id).
    for t in tasks.iter().skip(resume_next_id) {
        tx.send(*t).unwrap();
    }
    drop(tx);

    eprintln!(
        "Render: {}x{}, spp={}, tile={}, threads={}, morton={}, seed={}",
        w, h, config.spp, config.tile, threads, config.morton_enabled, config.seed
    );

    let limits = PathLimits { max_depth: config.max_depth, rr_depth: config.rr_depth };

    let surfaces_val = Surfaces {
        textures: &scene.textures,
        normal_maps: &scene.normal_maps,
        mat_maps: &scene.mat_maps,
    };
    let surfaces_ref = &surfaces_val;

    scope(|sp| {
        let world_ref = &scene.world;
        let mats_ref = &scene.mats;
        let cam_ref = &scene.cam;
        let env_ref = scene.env.as_ref();

        for _ in 0..threads {
            let rx = rx.clone();
            let rtx = rtx.clone();
            let world = world_ref;
            let mats = mats_ref;
            let surfaces = surfaces_ref;
            let cam = cam_ref;
            let env = env_ref;
            sp.spawn(move |_| {
                // ワーカーループ: チャネルからタスクを受信し処理
                while let Ok(t) = rx.recv() {
                    let tile_w = t.x1 - t.x0;
                    let tile_h = t.y1 - t.y0;
                    let mut sum = vec![Color::new(0.0, 0.0, 0.0); tile_w * tile_h];
                    let mut wsum = vec![0.0; tile_w * tile_h];

                    for y in t.y0..t.y1 {
                        for x in t.x0..t.x1 {
                            let local_idx = (y - t.y0) * tile_w + (x - t.x0);
                            let (c, n) = sample_pixel(x, y, inv_w, inv_h, &t, world, mats, surfaces, env, cam, limits, config);
                            sum[local_idx] = c;
                            wsum[local_idx] = n;
                        }
                    }
                    rtx.send(TileResult {
                        id: t.id,
                        x0: t.x0,
                        y0: t.y0,
                        x1: t.x1,
                        y1: t.y1,
                        sum,
                        w: wsum,
                    })
                    .unwrap();
                }
            });
        }

        drop(rtx);

        // Stream results and merge in deterministic task-id order to reduce memory.
        // We only buffer out-of-order completions.
        let mut pending: BTreeMap<usize, TileResult> = BTreeMap::new();
        let mut next_id: usize = resume_next_id;

        let remaining = tid.saturating_sub(resume_next_id);
        let start_time = Instant::now();
        let mut last_print = Instant::now();

        // 初回表示（resume時にも有効）
        if tid > 0 {
            let pct = (next_id as f64) * 100.0 / (tid as f64);
            eprint!("\rProgress: {}/{} tiles ({:.1}%)", next_id, tid, pct);
            std::io::stderr().flush().ok();
        }

        for _ in 0..remaining {
            let r = rrx.recv().unwrap();
            pending.insert(r.id, r);

            // Merge any consecutive ready tiles.
            while let Some(r) = pending.remove(&next_id) {
                out.merge_tile(&r, w);
                next_id += 1;

                // 進捗表示（PROGRESS_INTERVAL_MSごと、または完了時）
                let now = Instant::now();
                if now.duration_since(last_print).as_millis() >= PROGRESS_INTERVAL_MS || next_id == tid {
                    let pct = (next_id as f64) * 100.0 / (tid as f64);
                    eprint!("\rProgress: {}/{} tiles ({:.1}%)", next_id, tid, pct);
                    std::io::stderr().flush().ok();
                    last_print = now;
                }

                if ckpt_enabled && (next_id % config.checkpoint_every_tasks == 0) {
                    if let Err(e) = save_checkpoint(ckpt_file, config.scene_hash, w, h, next_id, &out.acc, &out.acc_w)
                    {
                        eprintln!("Checkpoint save failed: {}", e);
                    } else {
                        // eprintln!("Checkpoint saved: {} (next task id: {})", ckpt_file, next_id);
                    }
                }
            }
        }
        eprintln!(); // progress 行を確定（改行）
        eprintln!("Done in {:.2}s", start_time.elapsed().as_secs_f64());
    })
    .unwrap();

    // Final checkpoint
    if ckpt_enabled {
        if let Err(e) = save_checkpoint(ckpt_file, config.scene_hash, w, h, tid, &out.acc, &out.acc_w) {
            eprintln!("Final checkpoint save failed: {}", e);
        }
    }

    Ok(out)
}

/// Morton（Z オーダー）曲線によるタイル座標のキーを返す。
///
/// Morton 曲線は 2D 座標を 1D に変換する空間充填曲線で、
/// 空間的に近いタイルが近い ID を持つため、キャッシュ効率が向上する。
fn morton2(tx: usize, ty: usize, tiles_x: usize, tiles_y: usize) -> u64 {
    let x = tx.min(tiles_x.saturating_sub(1)) as u64;
    let y = ty.min(tiles_y.saturating_sub(1)) as u64;
    interleave_bits(x) | (interleave_bits(y) << 1)
}

/// 下位 32 ビットにゼロビットを挿入（ビット分離）して Morton キーの半分を生成する。
fn interleave_bits(v: u64) -> u64 {
    let mut x = v & 0x0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::ray::Camera;
    use crate::scene::Scene;
    use crate::world::World;

    fn empty_scene(w: usize, h: usize) -> Scene {
        let cam = Camera::look_at(
            Vec3::new(0.0, 0.0, 1.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            45.0,
            w as f64 / h as f64,
        );
        Scene {
            cam,
            world: World::new(),
            mats: Vec::new(),
            textures: Vec::new(),
            normal_maps: Vec::new(),
            mat_maps: Vec::new(),
            env: None,
        }
    }

    #[test]
    fn render_accumulates_expected_weights() {
        let w = 2;
        let h = 2;
        let scene = empty_scene(w, h);
        let config = RenderConfig {
            width: w,
            height: h,
            spp: 1,
            max_depth: 8,
            rr_depth: 3,
            tile: 1,
            checkpoint_enabled: false,
            checkpoint_every_tasks: 1,
            scene_hash: 0,
            output_path: String::new(),
            env_map_path: None,
            scene_path: None,
            denoise_enabled: false,
            adaptive_enabled: false,
            adaptive_min_spp: 1,
            adaptive_threshold: 0.01,
            morton_enabled: true,
            seed: 0,
            tonemap: crate::config::Tonemap::None,
            exposure: 0.0,
        };

        let out = render(&scene, &config, "ignored").expect("render should succeed");

        assert_eq!(out.acc.len(), w * h);
        assert_eq!(out.acc_w.len(), w * h);
        for wsum in out.acc_w {
            assert_eq!(wsum, config.spp as f64);
        }
        for c in out.acc {
            let v: Vec3 = c.into();
            assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
        }
    }

    fn base_config(w: usize, h: usize, tile: usize) -> RenderConfig {
        RenderConfig {
            width: w,
            height: h,
            spp: 4,
            max_depth: 8,
            rr_depth: 3,
            tile,
            checkpoint_enabled: false,
            checkpoint_every_tasks: 1,
            scene_hash: 0,
            output_path: String::new(),
            env_map_path: None,
            scene_path: None,
            denoise_enabled: false,
            adaptive_enabled: false,
            adaptive_min_spp: 1,
            adaptive_threshold: 0.01,
            morton_enabled: false,
            seed: 0,
            tonemap: crate::config::Tonemap::None,
            exposure: 0.0,
        }
    }

    /// タイル分割は画像全体を過不足なく覆い、ID は 0 から連番になる。
    #[test]
    fn build_tasks_covers_image_with_sequential_ids() {
        let config = base_config(5, 3, 2);
        let tasks = build_tasks(&config);
        // 5x3 を 2x2 タイルで割ると ceil(5/2)*ceil(3/2) = 3*2 = 6 タイル
        assert_eq!(tasks.len(), 6);
        for (i, t) in tasks.iter().enumerate() {
            assert_eq!(t.id, i);
            assert!(t.x1 <= config.width && t.y1 <= config.height);
            assert!(t.x0 < t.x1 && t.y0 < t.y1);
        }
        // 全ピクセルがちょうど 1 タイルに属する
        let mut covered = vec![0u32; config.width * config.height];
        for t in &tasks {
            for y in t.y0..t.y1 {
                for x in t.x0..t.x1 {
                    covered[y * config.width + x] += 1;
                }
            }
        }
        assert!(covered.iter().all(|&c| c == 1), "every pixel covered exactly once");
    }

    /// Morton 順序を有効にしてもタスク ID は並べ替え後に 0 から連番のまま。
    #[test]
    fn build_tasks_morton_order_keeps_sequential_ids() {
        let mut config = base_config(8, 8, 2);
        config.morton_enabled = true;
        let tasks = build_tasks(&config);
        assert_eq!(tasks.len(), 16);
        for (i, t) in tasks.iter().enumerate() {
            assert_eq!(t.id, i);
        }
    }

    /// merge_tile はタイル内のローカルインデックスをグローバル座標に正しくマップする。
    #[test]
    fn merge_tile_accumulates_at_correct_global_offset() {
        let width = 4;
        let height = 4;
        let mut out = RenderOutput {
            acc: vec![Color::new(0.0, 0.0, 0.0); width * height],
            acc_w: vec![0.0; width * height],
        };
        // (1,1)-(3,3) の 2x2 タイル
        let tile = TileResult {
            id: 0,
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
            sum: vec![
                Color::new(1.0, 0.0, 0.0),
                Color::new(2.0, 0.0, 0.0),
                Color::new(3.0, 0.0, 0.0),
                Color::new(4.0, 0.0, 0.0),
            ],
            w: vec![1.0, 1.0, 1.0, 1.0],
        };
        out.merge_tile(&tile, width);

        fn r(out: &RenderOutput, x: usize, y: usize, width: usize) -> f64 {
            Vec3::from(out.acc[idx(x, y, width)]).x
        }
        assert_eq!(r(&out, 1, 1, width), 1.0);
        assert_eq!(r(&out, 2, 1, width), 2.0);
        assert_eq!(r(&out, 1, 2, width), 3.0);
        assert_eq!(r(&out, 2, 2, width), 4.0);
        // タイル外は変化しない
        assert_eq!(r(&out, 0, 0, width), 0.0);
        assert_eq!(out.acc_w[idx(1, 1, width)], 1.0);

        // 二回目のマージは加算される
        out.merge_tile(&tile, width);
        assert_eq!(r(&out, 1, 1, width), 2.0);
        assert_eq!(out.acc_w[idx(1, 1, width)], 2.0);
    }

    /// 不整合なチェックポイント（バッファ長・next_id 超過）はレジュームに使わない。
    /// 以前はバッファ不一致でも resume_next_id だけが残り、タイルが欠落していた。
    #[test]
    fn validate_resume_rejects_inconsistent_checkpoints() {
        let px = |n| vec![Color::new(0.0, 0.0, 0.0); n];
        assert!(validate_resume(None, 4, 4).is_none());
        assert!(validate_resume(Some((2, px(3), vec![0.0; 4])), 4, 4).is_none());
        assert!(validate_resume(Some((2, px(4), vec![0.0; 3])), 4, 4).is_none());
        assert!(validate_resume(Some((5, px(4), vec![0.0; 4])), 4, 4).is_none());
        let ok = validate_resume(Some((4, px(4), vec![0.0; 4])), 4, 4).unwrap();
        assert_eq!(ok.0, 4);
    }

    /// ハッシュが一致するチェックポイントだけから再開し、不一致なら最初からレンダーする。
    #[test]
    fn render_resumes_only_from_matching_scene_hash() {
        let (w, h) = (2, 2);
        let scene = empty_scene(w, h);
        let mut config = base_config(w, h, 1);
        config.checkpoint_enabled = true;
        config.scene_hash = 0xABCD;
        let dir = std::env::temp_dir().join(format!("tinypt_render_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ckpt.bin").to_string_lossy().into_owned();

        // 全タスク完了済み（next_id = タスク数）で目印の値を持つチェックポイント
        let n_tasks = build_tasks(&config).len();
        let marker = vec![Color::new(7.0, 7.0, 7.0); w * h];
        save_checkpoint(&path, 0xABCD, w, h, n_tasks, &marker, &vec![99.0; w * h]).unwrap();

        let out = render(&scene, &config, &path).unwrap();
        assert!(out.acc_w.iter().all(|&v| v == 99.0), "matching hash should resume");

        save_checkpoint(&path, 0x1234, w, h, n_tasks, &marker, &vec![99.0; w * h]).unwrap();
        let out = render(&scene, &config, &path).unwrap();
        assert!(out.acc_w.iter().all(|&v| v == config.spp as f64), "mismatched hash must not resume");

        std::fs::remove_file(&path).ok();
    }

    /// 64-bit FNV-1a（ゴールデン値の計算用）。
    fn fnv64(data: impl IntoIterator<Item = u8>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// 蓄積バッファのハッシュ。`quantize` なら有限値を 1e-9 単位に丸めてからハッシュする。
    ///
    /// ゴールデン比較には丸めた値を使う。macOS の最適化ビルドは同じ引数の sin/cos を
    /// `__sincos_stret` にまとめるため、debug と release で数画素の値が最終 ulp だけ
    /// 異なる（出力の 8bit PPM は同一）。スレッド数比較は同一ビルド内なのでビット単位で行う。
    ///
    /// 丸めは有限かつ |x| < 1e9 の値だけに適用し、それ以外（NaN・±inf・巨大値）は
    /// 生ビットを別タグで混ぜる。`NaN as i64` が 0 になり NaN と 0 が区別できなくなるのを防ぐ。
    fn buffer_hash(out: &RenderOutput, quantize: bool) -> u64 {
        let mut bytes = Vec::with_capacity(out.acc.len() * 36);
        for (c, w) in out.acc.iter().zip(&out.acc_w) {
            let v: Vec3 = (*c).into();
            for x in [v.x, v.y, v.z, *w] {
                if quantize && x.is_finite() && x.abs() < 1e9 {
                    bytes.push(0);
                    bytes.extend_from_slice(&((x * 1e9).round() as i64).to_le_bytes());
                } else {
                    bytes.push(1);
                    bytes.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
        }
        fnv64(bytes)
    }

    /// 丸めハッシュは NaN / ±inf / 0 を区別し、ulp 程度の差は同一視する。
    #[test]
    fn buffer_hash_distinguishes_non_finite() {
        let make = |x: f64| RenderOutput { acc: vec![Color::new(x, 0.5, 0.25)], acc_w: vec![1.0] };
        let h = |x: f64| buffer_hash(&make(x), true);
        let zero = h(0.0);
        assert_ne!(h(f64::NAN), zero);
        assert_ne!(h(f64::INFINITY), zero);
        assert_ne!(h(f64::NEG_INFINITY), zero);
        assert_ne!(h(f64::INFINITY), h(f64::NEG_INFINITY));
        assert_ne!(h(f64::NAN), h(f64::INFINITY));
        let x = 0.3945;
        assert_eq!(h(x), h(f64::from_bits(x.to_bits() + 1)), "ulp difference must not change the quantized hash");
        assert_ne!(h(x), h(x + 1e-6));
    }

    /// ゴールデン値の組（`RENDER_REVISION` と対で更新する。片方だけ変えるとテストが失敗する）。
    const GOLDEN_REVISION: u32 = 15;

    /// sample/cornell.xml を 48x48・2spp（seed 0、tile 16、Morton）で描画した蓄積バッファの
    /// 丸めハッシュと、それを `--tonemap none` 相当で書いた PPM（P6）ファイルのハッシュ。
    /// PPM を P3 から P6 に変えたとき、PPM のハッシュだけを更新した（蓄積バッファのハッシュと画素値は不変。
    /// P6 の画素値を P3 に書き直すと旧ハッシュ 0xdd80_3964_fecc_5c3c / 0xf4e9_ab88_c297_cdc3 に一致した）。
    /// スムーズシェーディング導入（RENDER_REVISION 11）でも**値は変わっていない**: このシーンは
    /// rectangle / cube だけで頂点法線を持たず、シェーディング法線 = 幾何法線のままだから。
    /// 値が変わっていないこと自体が「頂点法線の無いシーンの出力は不変」の回帰テストになっている。
    /// テクスチャ導入（RENDER_REVISION 12）でも同じく不変: rectangle / cube に UV は付いたが、
    /// テクスチャを参照しないマテリアルでは UV を一度も読まないため。
    /// 法線マップ／バンプマップ（RENDER_REVISION 14）でも不変: マップを持つ材質が無いシーンは摂動経路に入らない。
    /// `map_Kd` を GGX 分岐より優先する不具合修正（RENDER_REVISION 15）でも不変: golden シーンは map_Kd と
    /// 明るい Ks/Ns の両方を持つ材質を含まない。
    /// アルファマスク導入（RENDER_REVISION 13）でも不変: マスクを持たないメッシュは従来の交差経路のまま。
    const GOLDEN_CORNELL: (u64, u64) = (0x6779_3a10_a998_bdb0, 0x7c6a_81d0_9dd4_74a2);

    /// [`GOLDEN_SPHERES_XML`] を 64x36・2spp で描画したもののハッシュ。
    const GOLDEN_SPHERES: (u64, u64) = (0x9d2f_abcf_f7df_7882, 0xdd5e_c0af_a8da_343c);

    /// sample/default.xml 相当（Lambert・金属・GGX・吸収付きガラス・球光源・地面の大球）に、
    /// constant 環境 emitter と被写界深度（aperture_radius > 0）を加えたシーン。
    const GOLDEN_SPHERES_XML: &str = r#"<scene version="3.0.0">
      <integrator type="path"><integer name="max_depth" value="9"/><integer name="rr_depth" value="4"/></integrator>
      <sensor type="perspective">
        <float name="fov" value="40"/>
        <float name="aperture_radius" value="0.05"/>
        <float name="focus_distance" value="3.5"/>
        <transform name="to_world"><lookat origin="0, 1.2, 4" target="0, 0.5, 0" up="0, 1, 0"/></transform>
      </sensor>
      <shape type="sphere"><point name="center" x="0" y="-1000" z="0"/><float name="radius" value="1000"/>
        <bsdf type="diffuse"><srgb name="reflectance" value="0.5, 0.5, 0.5"/></bsdf></shape>
      <shape type="sphere"><point name="center" x="-1.8" y="0.5" z="0"/><float name="radius" value="0.5"/>
        <bsdf type="diffuse"><srgb name="reflectance" value="0.8, 0.3, 0.3"/></bsdf></shape>
      <shape type="sphere"><point name="center" x="-0.6" y="0.5" z="0"/><float name="radius" value="0.5"/>
        <bsdf type="conductor"><srgb name="specular_reflectance" value="0.8, 0.8, 0.8"/></bsdf></shape>
      <shape type="sphere"><point name="center" x="0.6" y="0.5" z="0"/><float name="radius" value="0.5"/>
        <bsdf type="roughconductor"><string name="distribution" value="ggx"/><float name="alpha" value="0.25"/>
          <srgb name="specular_reflectance" value="0.95, 0.78, 0.35"/></bsdf></shape>
      <shape type="sphere"><point name="center" x="1.8" y="0.5" z="0"/><float name="radius" value="0.5"/>
        <bsdf type="dielectric"><float name="int_ior" value="1.5"/><rgb name="absorption" value="0.02, 0.05, 0.02"/></bsdf></shape>
      <shape type="sphere"><point name="center" x="0" y="3" z="-1"/><float name="radius" value="0.8"/>
        <emitter type="area"><rgb name="radiance" value="8, 7, 5"/></emitter></shape>
      <emitter type="constant"><rgb name="radiance" value="0.4, 0.5, 0.7"/></emitter>
    </scene>"#;

    /// 小さく描画し、スレッド数 1 / 3 / 利用可能数で出力がビット単位で同じことを確かめてから、
    /// ゴールデン値と比較する。浮動小数点演算の決定性（Rust は FMA 縮約や fast-math をしない）に
    /// 依拠する。別プラットフォームで libm（sin/cos/pow 等）の丸めが違うと値が変わりうる。
    fn check_golden(name: &str, scene: &Scene, config: &RenderConfig, golden: (u64, u64)) {
        let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(2);
        let mut hashes = Vec::new();
        for threads in [1, 3, max_threads] {
            let out = render_with_threads(scene, config, "", threads).unwrap();
            let pixels = crate::output::resolve_pixels(config.width, config.height, &out.acc, &out.acc_w);
            let path = std::env::temp_dir().join(format!("tinypt_golden_{}_{}_{}.ppm", name, std::process::id(), threads));
            let path = path.to_string_lossy().into_owned();
            let settings = crate::output::OutputSettings { exposure: 0.0, tonemap: crate::config::Tonemap::None };
            crate::output::OutputFormat::Ppm.write(&path, config.width, config.height, &pixels, settings).unwrap();
            let ppm = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).ok();
            hashes.push((threads, buffer_hash(&out, false), buffer_hash(&out, true), fnv64(ppm)));
        }
        for &(threads, exact, _, p) in &hashes[1..] {
            assert_eq!((exact, p), (hashes[0].1, hashes[0].3), "{}: output depends on thread count ({} vs 1)", name, threads);
        }
        let (_, _, buffer, ppm) = hashes[0];
        assert_eq!(
            crate::constants::RENDER_REVISION, GOLDEN_REVISION,
            "RENDER_REVISION was bumped: re-record the golden hashes for the new output \
             ({}: 0x{:016x} / 0x{:016x}) and set GOLDEN_REVISION to match",
            name, buffer, ppm
        );
        assert!(
            (buffer, ppm) == golden,
            "{}: rendered output changed: buffer hash 0x{:016x} (golden 0x{:016x}), PPM hash 0x{:016x} (golden 0x{:016x}).\n\
             If this change is meant to alter the output, bump constants::RENDER_REVISION (so old checkpoints \
             are not resumed) and update the GOLDEN_* hashes in src/render.rs to the new values.\n\
             If it is not meant to alter the output, this is a regression.",
            name, buffer, golden.0, ppm, golden.1
        );
    }

    fn golden_config(config: &mut RenderConfig, width: usize, height: usize) {
        config.width = width;
        config.height = height;
        config.spp = 2;
        config.seed = 0;
        config.tile = 16;
        config.morton_enabled = true;
        config.adaptive_enabled = false;
        config.checkpoint_enabled = false;
    }

    /// sample/cornell.xml 全体（形状の変換とカメラ）を `k` 倍に拡大縮小したシーンを読み込む。
    /// 放射輝度はそのままなので、理想的には k によらず同じ画像になる。
    fn scaled_cornell(k: f64, width: usize, height: usize) -> (Scene, RenderConfig) {
        let xml = std::fs::read_to_string("sample/cornell.xml").expect("read sample/cornell.xml");
        let n_shapes = xml.matches("<shape").count();
        // 各形状の to_world の最も外側に一様スケールを挿入する（センサーの transform では無視される）
        let scaled = xml.replace(r#"<transform name="to_world">"#, &format!(r#"<transform name="to_world"><scale value="{}"/>"#, k));
        assert_eq!(scaled.matches("<scale value=").count() - xml.matches("<scale value=").count(), n_shapes + 1);
        let lookat = r#"origin="0, 1, 3.9" target="0, 1, 0""#;
        assert_eq!(scaled.matches(lookat).count(), 1, "cornell.xml camera changed; update this test");
        let scaled = scaled.replace(lookat, &format!(r#"origin="0, {}, {}" target="0, {}, 0""#, k, 3.9 * k, k));
        let mut config = RenderConfig::default();
        let (scene, settings) = crate::mitsuba::load_scene_from_str(&scaled, std::path::Path::new("sample"), &config, (None, None)).unwrap();
        settings.apply(&mut config);
        config.width = width;
        config.height = height;
        config.adaptive_enabled = false;
        config.checkpoint_enabled = false;
        (scene, config)
    }

    /// スケール不変性: Cornell box を 1e-3 倍・1e3 倍にしても、画像（全体と 4×4 ブロックの平均輝度）は
    /// 等倍と統計的に一致する。自己交差回避（交差点の誤差上界に基づく原点のずらし、`offset_ray_origin`）が
    /// シーンのスケールに依存しないことの回帰テスト。以前の絶対オフセット 1e-4 では、1e-3 倍（箱の辺が 0.0006）で
    /// 接地部の光漏れや角の暗さが出た。
    #[test]
    fn cornell_is_scale_invariant() {
        let (w, h, spp, seeds) = (24usize, 24usize, 32usize, 8u64);
        let block_means = |k: f64| -> Vec<[f64; 17]> {
            let (scene, mut config) = scaled_cornell(k, w, h);
            config.spp = spp;
            (1..=seeds)
                .map(|seed| {
                    config.seed = seed;
                    let out = render_with_threads(&scene, &config, "", 4).unwrap();
                    let px = crate::output::resolve_pixels(w, h, &out.acc, &out.acc_w);
                    let mut m = [0.0; 17];
                    for y in 0..h {
                        for x in 0..w {
                            let l = px[y * w + x].luminance();
                            m[16] += l / (w * h) as f64;
                            m[(y * 4 / h) * 4 + x * 4 / w] += l / ((w / 4) * (h / 4)) as f64;
                        }
                    }
                    m
                })
                .collect()
        };
        let stats = |v: &[[f64; 17]], i: usize| {
            let n = v.len() as f64;
            let mean = v.iter().map(|m| m[i]).sum::<f64>() / n;
            let var = v.iter().map(|m| (m[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var / n)
        };
        let base = block_means(1.0);
        for k in [1e-3, 1e3] {
            let other = block_means(k);
            for i in 0..17 {
                let (m1, v1) = stats(&base, i);
                let (mk, vk) = stats(&other, i);
                let tol = 5.0 * (v1 + vk).sqrt() + 1e-4 * m1;
                assert!((mk - m1).abs() < tol, "scale {}: {} mean {} vs scale 1 {} (|Δ| {:.3e} ≥ tol {:.3e})",
                    k, if i == 16 { "image".to_string() } else { format!("block {}", i) }, mk, m1, (mk - m1).abs(), tol);
            }
        }
    }

    /// Cornell box（面光源・長方形/立方体インスタンス・黒背景）の出力を固定する。
    #[test]
    fn golden_cornell_output_is_unchanged() {
        let mut config = RenderConfig::default();
        let scene = crate::mitsuba::load_scene("sample/cornell.xml", &mut config, (None, None)).expect("load sample/cornell.xml");
        golden_config(&mut config, 48, 48);
        check_golden("cornell", &scene, &config, GOLDEN_CORNELL);
    }

    /// 球シーン（GGX・ガラス・金属・球光源・constant 環境光・DOF）の出力を固定する。
    #[test]
    fn golden_spheres_output_is_unchanged() {
        let mut config = RenderConfig::default();
        golden_config(&mut config, 64, 36);
        let (scene, settings) = crate::mitsuba::load_scene_from_str(GOLDEN_SPHERES_XML, std::path::Path::new("."), &config, (None, None))
            .expect("parse golden spheres scene");
        settings.apply(&mut config);
        golden_config(&mut config, 64, 36);
        check_golden("spheres", &scene, &config, GOLDEN_SPHERES);
    }
}
