//! MIS（Multiple Importance Sampling）付きパストレーシング積分器。
//!
//! 単方向パストレーサーとして以下の機能を実装:
//! - **NEE（Next Event Estimation）**: 各バウンスで光源を直接サンプリングし直接照明を推定
//! - **MIS**: BSDF サンプリングと光源サンプリングを Power Heuristic (β=2) で統合
//! - **Russian Roulette**: スループットに基づく確率的なパス打ち切り（不偏性を維持）
//! - **Firefly クランプ**: 異常に明るいサンプルを輝度ベースでクランプ
//!
//! ## レンダリング方程式
//! L_o(x, ω_o) = L_e(x, ω_o) + ∫ f(x, ω_i, ω_o) L_i(x, ω_i) cos(θ_i) dω_i
//!
//! パストレーシングではこの積分をモンテカルロ推定で近似する。

use crate::constants::path::FIREFLY_CLAMP;
use crate::constants::{RAY_EPSILON, RAY_T_MAX};
use crate::env::EnvMap;
use crate::material::{BsdfSample, Material};
use crate::math::{Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::world::World;

/// デフォルトの空色を返す（環境マップ未使用時のフォールバック）。
/// 方向の Y 成分で白〜青のグラデーションを線形補間する。
fn sky(d: Vec3) -> Color {
    let t = 0.5 * (d.y + 1.0);
    Color::from((1.0 - t) * Vec3::new(0.9, 0.95, 1.0) + t * Vec3::new(0.6, 0.75, 1.0))
}

/// 背景色を取得する。環境マップがあればそれを参照、なければ手続き的な空色を返す。
fn background(d: Vec3, env: Option<&EnvMap>) -> Color {
    if let Some(m) = env {
        m.sample(d)
    } else {
        sky(d)
    }
}

/// パストレーシングのパラメータ（シーン/設定由来でランタイムに与える）。
#[derive(Clone, Copy)]
pub struct PathLimits {
    /// 最大バウンス数
    pub max_bounces: usize,
    /// Russian Roulette を開始するバウンス数
    pub rr_start: usize,
}

/// パスを追跡し推定放射輝度を返す。
///
/// カメラレイから出発し、最大 `limits.max_bounces` 回の散乱を追跡する。
/// 各バウンスで NEE（直接照明推定）と BSDF サンプリングを行い、
/// MIS で重みを統合して蓄積する。
pub fn radiance(
    world: &World,
    mats: &[Material],
    env: Option<&EnvMap>,
    ray: Ray,
    rng: &mut Rng,
    limits: PathLimits,
) -> Color {
    let mut accumulated_radiance = Color::new(0.0, 0.0, 0.0); // パス全体の蓄積放射輝度
    let mut path_throughput = Color::new(1.0, 1.0, 1.0);       // パスのスループット（減衰係数）
    let mut ray = ray;
    let mut last_bsdf_pdf = 0.0;     // 前バウンスの BSDF PDF（MIS 用）
    let mut last_non_delta = false;   // 前バウンスが非デルタ散乱か（MIS 適用判定）

    for bounce in 0..limits.max_bounces {
        // レイとシーンの交差判定
        let hit = match world.hit(ray, RAY_EPSILON, RAY_T_MAX) {
            Some(v) => v,
            None => {
                // ミス: 背景（環境マップまたは空）からの寄与を加算
                let mut contrib = path_throughput.hadamard(background(ray.d, env));
                // 非デルタ散乱後なら環境マップ PDF との MIS 重みを適用
                if env.is_some() && last_non_delta {
                    let pdf_env = env.unwrap().pdf(ray.d);
                    let w = mis_weight(last_bsdf_pdf, pdf_env);
                    contrib = contrib * w;
                }
                let contrib = contrib.clamp_luminance(FIREFLY_CLAMP);
                accumulated_radiance = accumulated_radiance + contrib;
                break;
            }
        };

        // Russian Roulette: probabilistic path termination for unbiased rendering.
        // The survival probability is based on the maximum throughput component,
        // clamped to [0.05, 0.95] to avoid extreme variance.
        if bounce >= limits.rr_start {
            let p = path_throughput.r().max(path_throughput.g()).max(path_throughput.b()).min(0.95).max(0.05);
            if rng.next_f64() > p {
                break;
            }
            path_throughput = path_throughput / p;
        }

        let mat = mats[hit.mat_id];

        // 発光体に命中: 放射輝度を蓄積しパス終了
        if let Some(emit) = mat.emitted() {
            let contrib = path_throughput.hadamard(emit).clamp_luminance(FIREFLY_CLAMP);
            accumulated_radiance = accumulated_radiance + contrib;
            break;
        }

        let n = oriented_normal(hit.n, ray.d);

        // NEE（Next Event Estimation）はデルタ散乱マテリアルでは行わない
        if !mat.is_delta() {
            // NEE: Environment map
            if let Some(env_map) = env {
                let contrib = nee_environment(world, env_map, &mat, path_throughput, hit.p, n, ray, rng);
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: Area lights
            if let Some(ls) = world.sample_light(rng, ray.time, hit.p) {
                let contrib = nee_area_light(world, &mat, path_throughput, hit.p, n, ray, &ls);
                accumulated_radiance = accumulated_radiance + contrib;
            }
        }

        // BSDF サンプリング: 散乱レイ・スループット重み・PDF を BSDF から取得
        match mat.sample(&ray, &hit, rng) {
            Some(BsdfSample { scattered, weight, pdf, is_delta }) => {
                last_bsdf_pdf = pdf;
                last_non_delta = !is_delta;
                path_throughput = path_throughput.hadamard(weight);
                ray = scattered;
            }
            None => break,
        }
    }

    accumulated_radiance
}

/// レイの進行方向に対して正しい向きの法線を返す（裏面判定）。
fn oriented_normal(n: Vec3, ray_d: Vec3) -> Vec3 {
    if n.dot(ray_d) < 0.0 { n } else { -n }
}

use crate::world::LightSample;

/// 環境マップに対する NEE（Next Event Estimation / 直接照明推定）。
///
/// 環境マップから重点的にサンプリングした方向に対し、
/// シャドウレイで遮蔽判定を行い、MIS 重みを適用して寄与を返す。
fn nee_environment(
    world: &World,
    env_map: &EnvMap,
    mat: &Material,
    path_throughput: Color,
    hit_p: Vec3,
    n: Vec3,
    ray: Ray,
    rng: &mut Rng,
) -> Color {
    let (wi, li, pdf_env) = env_map.sample_dir(rng);
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 || pdf_env <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let shadow = Ray { o: hit_p + RAY_EPSILON * wi, d: wi, time: ray.time };
    if world.hit(shadow, RAY_EPSILON, RAY_T_MAX).is_some() {
        return Color::new(0.0, 0.0, 0.0);
    }

    let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), wi, n);
    let w = mis_weight(pdf_env, pdf_bsdf);
    path_throughput.hadamard(f).hadamard(li) * (cos * w / pdf_env)
}

/// 面光源に対する NEE（Next Event Estimation / 直接照明推定）。
///
/// CDF で選択されたライトの表面上をサンプリングし、
/// シャドウレイで遮蔽判定後、MIS 重みを適用して寄与を返す。
fn nee_area_light(
    world: &World,
    mat: &Material,
    path_throughput: Color,
    hit_p: Vec3,
    n: Vec3,
    ray: Ray,
    ls: &LightSample,
) -> Color {
    let to_light = ls.position - hit_p;
    let dist2 = to_light.dot(to_light);
    let dist = dist2.sqrt();
    if dist <= 1e-6 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let wi = to_light / dist;
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let shadow = Ray { o: hit_p + RAY_EPSILON * wi, d: wi, time: ray.time };
    let tmax = (dist - RAY_EPSILON).max(RAY_EPSILON);
    if world.hit(shadow, RAY_EPSILON, tmax).is_some() {
        return Color::new(0.0, 0.0, 0.0);
    }

    let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), wi, n);
    if ls.pdf <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let w = mis_weight(ls.pdf, pdf_bsdf);
    path_throughput.hadamard(f).hadamard(ls.emit) * (cos * w / ls.pdf)
}

/// Power Heuristic (β=2) による MIS 重みを計算する。
///
/// w_a = pdf_a² / (pdf_a² + pdf_b²)
///
/// Balance Heuristic (β=1) より分散削減効果が高く、
/// 多くのレンダラーで標準的に使用される。
fn mis_weight(pdf_a: f64, pdf_b: f64) -> f64 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    if a2 + b2 > 0.0 { a2 / (a2 + b2) } else { 0.0 }
}
