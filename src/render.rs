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
use crate::integrator::{radiance, PathLimits};
use crate::math::Color;
use crate::rng::{seed_for, Rng};
use crate::scene::Scene;
use crate::task::{idx, Task, TileResult};

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

/// Renders the scene and returns accumulation buffers.
pub fn render(scene: &Scene, config: &RenderConfig, ckpt_file: &str) -> std::io::Result<RenderOutput> {
    let w = config.width;
    let h = config.height;
    let inv_w = 1.0 / (w as f64);
    let inv_h = 1.0 / (h as f64);

    // Accumulation buffers (may be restored from checkpoint)
    let mut acc: Vec<Color> = vec![Color::new(0.0, 0.0, 0.0); w * h];
    let mut acc_w: Vec<f64> = vec![0.0; w * h];

    let ckpt_enabled = config.checkpoint_enabled && config.checkpoint_every_tasks > 0;

    // Resume state
    let mut resume_next_id: usize = 0;
    if ckpt_enabled {
        if let Ok(Some((next_id, acc0, acc_w0))) = load_checkpoint(ckpt_file, config.scene_hash, w, h) {
            resume_next_id = next_id.min(usize::MAX);
            if acc0.len() == w * h && acc_w0.len() == w * h {
                acc = acc0;
                acc_w = acc_w0;
                eprintln!(
                    "Resumed from checkpoint: {} (next task id: {})",
                    ckpt_file, resume_next_id
                );
            }
        }
    }

    // タスク配布用チャネル (tx→rx) と結果回収用チャネル (rtx→rrx)
    let (tx, rx) = chan::unbounded::<Task>();
    let (rtx, rrx) = chan::unbounded::<TileResult>();

    // タイル分割: 画像を tile×tile のブロックに分割しタスクリストを生成
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
    // Morton-order tiles for better spatial locality.
    if config.morton_enabled {
        let tiles_x = (w + config.tile - 1) / config.tile;
        let tiles_y = (h + config.tile - 1) / config.tile;
        tasks.sort_by_key(|t| morton2(t.x0 / config.tile, t.y0 / config.tile, tiles_x, tiles_y));
    }
    for (i, t) in tasks.iter_mut().enumerate() {
        t.id = i;
    }
    let tid = tasks.len();

    // Send only tasks not yet merged (resume_next_id is the next expected merge id).
    for t in tasks.iter().skip(resume_next_id) {
        tx.send(*t).unwrap();
    }
    drop(tx);

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    eprintln!(
        "Render: {}x{}, spp={}, tile={}, threads={}, morton={}, seed={}",
        w, h, config.spp, config.tile, threads, config.morton_enabled, config.seed
    );

    let limits = PathLimits { max_bounces: config.max_bounces, rr_start: config.rr_start };

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
                            let mut c = Color::new(0.0, 0.0, 0.0);
                            // ピクセル座標とサンプル番号から決定論的シードを生成
                            let mut rng = Rng::new(seed_for(x as u32, y as u32, t.sample_start as u32) ^ config.seed);
                            if config.adaptive_enabled {
                                // 適応的サンプリング: 分散が閾値以下になったら早期終了
                                let max_spp = (t.sample_end - t.sample_start).max(1);
                                let min_spp = config.adaptive_min_spp.max(1).min(max_spp);
                                // Welford のオンライン分散計算アルゴリズム
                                let mut n: usize = 0;
                                let mut mean = 0.0;
                                let mut m2 = 0.0;
                                for _s in t.sample_start..t.sample_end {
                                    // ピクセル内のサブピクセルジッター（ストラティファイドではなくランダム）
                                    let jx = rng.next_f64();
                                    let jy = rng.next_f64();
                                    // スクリーン座標を [-1, 1] に正規化
                                    let sx = (x as f64 + jx) * inv_w * 2.0 - 1.0;
                                    let sy = 1.0 - (y as f64 + jy) * inv_h * 2.0;
                                    let ray = cam.ray(sx, sy, &mut rng);
                                    let sample = radiance(world, mats, env, ray, &mut rng, limits);
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
                                sum[local_idx] = c;
                                wsum[local_idx] = n as f64;
                            } else {
                                for _s in t.sample_start..t.sample_end {
                                    let jx = rng.next_f64();
                                    let jy = rng.next_f64();
                                    let sx = (x as f64 + jx) * inv_w * 2.0 - 1.0;
                                    let sy = 1.0 - (y as f64 + jy) * inv_h * 2.0;
                                    let ray = cam.ray(sx, sy, &mut rng);
                                    c = c + radiance(world, mats, env, ray, &mut rng, limits);
                                }
                                sum[local_idx] = c;
                                wsum[local_idx] = (t.sample_end - t.sample_start) as f64;
                            }
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
                let tile_w = r.x1 - r.x0;
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        let global_idx = idx(x, y, w);
                        let local_idx = (y - r.y0) * tile_w + (x - r.x0);
                        acc[global_idx] = acc[global_idx] + r.sum[local_idx];
                        acc_w[global_idx] += r.w[local_idx];
                    }
                }
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
                    if let Err(e) = save_checkpoint(ckpt_file, config.scene_hash, w, h, next_id, &acc, &acc_w)
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
        if let Err(e) = save_checkpoint(ckpt_file, config.scene_hash, w, h, tid, &acc, &acc_w) {
            eprintln!("Final checkpoint save failed: {}", e);
        }
    }

    Ok(RenderOutput {
        acc,
        acc_w,
    })
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
            max_bounces: 8,
            rr_start: 3,
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
}
