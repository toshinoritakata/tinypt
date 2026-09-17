//! MIS（Multiple Importance Sampling）付きパストレーシング積分器。
//!
//! 単方向パストレーサーとして以下の機能を実装:
//! - **NEE（Next Event Estimation）**: 各バウンスで光源を直接サンプリングし直接照明を推定
//! - **MIS**: BSDF サンプリングと光源サンプリングを Power Heuristic (β=2) で統合
//! - **Russian Roulette**: スループットに基づく確率的なパス打ち切り（不偏性を維持）。
//!   発光・NEE の寄与を積んだ後、BSDF サンプリングの直前で判定する
//! - **Firefly クランプ**: 異常に明るい寄与を輝度ベースでクランプ。MIS の両側（BSDF サンプリングで
//!   光源/背景に当たった寄与と、NEE の寄与）に寄与単位で同じ閾値を掛け、対称に保つ
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
    /// 最大パス長（Mitsuba の `max_depth`）。長さ k のパスはカメラから数えて k 個目の頂点で
    /// 光源（発光体・背景）に到達するパス。1 = 直接見える発光体のみ、2 = 直接照明まで。
    /// `usize::MAX` は無制限
    pub max_depth: usize,
    /// Russian Roulette を開始するパス長（Mitsuba の `rr_depth`）。長さ `rr_depth` 以上のパスを
    /// さらに延長するかどうかを確率的に決める
    pub rr_depth: usize,
}

/// パスを追跡し推定放射輝度を返す。
///
/// カメラレイから出発し、長さ `limits.max_depth` までのパスの寄与を推定する。
///
/// `bounce` 番目（0 始まり）の交差点はパス長 `bounce + 1` の頂点。発光体・背景への到達は
/// 長さ `bounce + 1` の寄与、この点からの NEE と BSDF サンプリングは長さ `bounce + 2` の寄与になるので、
/// 後者は `bounce + 2 <= max_depth` のときだけ行う。こうすると最後の長さでも NEE と BSDF 側の
/// 発光ヒットが必ず対で揃い、MIS の重みの和が 1 になる（以前は最後の長さの NEE だけが加算され、
/// 対になる BSDF 側のヒットが打ち切られていた）。
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
    let mut last_p = ray.o;           // 前バウンスのシェーディング点（面光源 MIS の light_pdf 計算用）

    for bounce in 0..limits.max_depth {
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

        let mat = mats[hit.mat_id];

        // 発光体に命中: 放射輝度を蓄積しパス終了
        if let Some(emit) = mat.emitted() {
            let mut contrib = path_throughput.hadamard(emit);
            // 非デルタ散乱後なら、NEE で同じ光源をサンプリング済みなので MIS 重みを適用
            if last_non_delta {
                let pdf_light = world.light_pdf(last_p, ray.time, &hit);
                let w = mis_weight(last_bsdf_pdf, pdf_light);
                contrib = contrib * w;
            }
            let contrib = contrib.clamp_luminance(FIREFLY_CLAMP);
            accumulated_radiance = accumulated_radiance + contrib;
            break;
        }

        // この点から先（NEE・BSDF サンプリング）はパス長 bounce + 2 の寄与。上限を超えるなら終了
        if bounce + 2 > limits.max_depth {
            break;
        }

        let n = oriented_normal(hit.n, ray.d);

        // NEE（Next Event Estimation）はデルタ散乱マテリアルでは行わない
        if !mat.is_delta() {
            // NEE: Environment map
            if let Some(env_map) = env {
                let occluded = |shadow: Ray| world.hit(shadow, RAY_EPSILON, RAY_T_MAX).is_some();
                let contrib = nee_environment(occluded, env_map, &mat, path_throughput, hit.p, n, ray, rng);
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: Area lights
            if let Some(ls) = world.sample_light(rng, ray.time, hit.p) {
                let contrib = nee_area_light(
                    |shadow: Ray, tmax: f64| world.hit(shadow, RAY_EPSILON, tmax).is_some(),
                    &mat, path_throughput, hit.p, n, ray, &ls,
                );
                accumulated_radiance = accumulated_radiance + contrib;
            }
        }

        // Russian Roulette: 確率的にパスを打ち切る（生存時は 1/p で補償するので不偏）。
        // この頂点での発光・NEE の寄与を積んだ後、続きのパス（BSDF サンプリング）を
        // 延ばすかどうかだけを判定する。打ち切っても既に得た直接光は失われない。
        // 生存確率はスループットの最大成分を [0.05, 0.95] にクランプしたもの。
        // 延長するのはパス長 bounce + 1 のパスなので、それが rr_depth 以上なら判定する。
        if bounce + 1 >= limits.rr_depth {
            let p = path_throughput.r().max(path_throughput.g()).max(path_throughput.b()).min(0.95).max(0.05);
            if rng.next_f64() > p {
                break;
            }
            path_throughput = path_throughput / p;
        }

        // BSDF サンプリング: 散乱レイ・スループット重み・PDF を BSDF から取得
        match mat.sample(&ray, &hit, rng) {
            Some(BsdfSample { scattered, weight, pdf, is_delta }) => {
                last_bsdf_pdf = pdf;
                last_non_delta = !is_delta;
                last_p = hit.p;
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
///
/// `occluded` はシャドウレイの遮蔽判定（通常は `world.hit`）。テストで呼び出し回数を
/// 観測できるよう注入する。
///
/// サンプルした放射輝度が厳密にゼロ（黒背景の constant emitter など）なら、寄与は
/// どうせ 0 なのでシャドウレイを撃たずに返す。`sample_dir` は必ず先に呼ぶので
/// RNG 消費列は変わらず、出力はビット単位で同一のまま。
fn nee_environment(
    occluded: impl Fn(Ray) -> bool,
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
    if cos <= 0.0 || pdf_env <= 0.0 || is_black(li) {
        return Color::new(0.0, 0.0, 0.0);
    }

    let shadow = Ray { o: hit_p + RAY_EPSILON * wi, d: wi, time: ray.time };
    if occluded(shadow) {
        return Color::new(0.0, 0.0, 0.0);
    }

    let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), wi, n);
    let w = mis_weight(pdf_env, pdf_bsdf);
    // BSDF 側（背景ヒット）と同じ閾値でクランプし、MIS の両側を対称にする
    (path_throughput.hadamard(f).hadamard(li) * (cos * w / pdf_env)).clamp_luminance(FIREFLY_CLAMP)
}

/// 全チャネルが厳密に 0 か。負値チャネルを含む色を誤って捨てないよう、
/// 輝度ではなく成分ごとに判定する（スキップ前後で寄与が完全に同じになる条件）。
fn is_black(c: Color) -> bool {
    c.r() == 0.0 && c.g() == 0.0 && c.b() == 0.0
}

/// 面光源に対する NEE（Next Event Estimation / 直接照明推定）。
///
/// CDF で選択されたライトの表面上をサンプリングし、
/// シャドウレイで遮蔽判定後、MIS 重みを適用して寄与を返す。
///
/// `occluded(shadow, tmax)` はシャドウレイの遮蔽判定（通常は `world.hit`）。
fn nee_area_light(
    occluded: impl Fn(Ray, f64) -> bool,
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
    // 原点を ε 前進させているため、ライト面は新原点から dist−ε に位置する。
    // tmax を dist−2ε にしないと丸め次第でライト自身に遮蔽判定される
    let tmax = (dist - 2.0 * RAY_EPSILON).max(RAY_EPSILON);
    if occluded(shadow, tmax) {
        return Color::new(0.0, 0.0, 0.0);
    }

    let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), wi, n);
    if ls.pdf <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let w = mis_weight(ls.pdf, pdf_bsdf);
    // BSDF 側（発光体ヒット）と同じ閾値でクランプし、MIS の両側を対称にする
    (path_throughput.hadamard(f).hadamard(ls.emit) * (cos * w / ls.pdf)).clamp_luminance(FIREFLY_CLAMP)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn floor_setup() -> (Material, Vec3, Vec3, Ray) {
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) };
        let p = Vec3::new(0.0, 0.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let ray = Ray { o: Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 };
        (mat, p, n, ray)
    }

    /// 黒い env ではシャドウレイ（遮蔽判定）を一度も撃たず、寄与は 0。
    /// RNG は sample_dir 単体と同じだけ消費される（出力のバイト一致の前提）。
    #[test]
    fn black_env_skips_shadow_ray_but_consumes_same_rng() {
        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
        let (mat, p, n, ray) = floor_setup();
        let calls = Cell::new(0usize);
        let mut rng = Rng::new(42);
        let mut reference = Rng::new(42);
        for _ in 0..1000 {
            let c = nee_environment(
                |_| { calls.set(calls.get() + 1); false },
                &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, &mut rng,
            );
            let _ = env.sample_dir(&mut reference);
            assert!(is_black(c));
        }
        assert_eq!(calls.get(), 0, "shadow rays cast against a black env");
        assert_eq!(rng.next_u32(), reference.next_u32(), "RNG consumption must match sample_dir");
    }

    /// 非ゼロの env では従来どおりシャドウレイを撃ち、非遮蔽なら正の寄与を返す。
    #[test]
    fn nonblack_env_still_casts_shadow_rays() {
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let (mat, p, n, ray) = floor_setup();
        let calls = Cell::new(0usize);
        let mut rng = Rng::new(42);
        let mut total = 0.0;
        for _ in 0..1000 {
            let c = nee_environment(
                |_| { calls.set(calls.get() + 1); false },
                &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, &mut rng,
            );
            total += c.luminance();
        }
        assert!(calls.get() > 0);
        assert!(total > 0.0);
    }

    /// 環境 NEE の高輝度寄与は FIREFLY_CLAMP でクランプされる（BSDF 側と対称）。
    #[test]
    fn env_nee_contribution_is_clamped() {
        let env = EnvMap::constant(Color::new(1e6, 1e6, 1e6));
        let (mat, p, n, ray) = floor_setup();
        let mut rng = Rng::new(7);
        let mut max_l: f64 = 0.0;
        for _ in 0..200 {
            let c = nee_environment(|_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, &mut rng);
            max_l = max_l.max(c.luminance());
        }
        assert!(max_l > 0.0);
        assert!(max_l <= FIREFLY_CLAMP * (1.0 + 1e-12), "max luminance = {}", max_l);
    }

    /// 面光源 NEE の高輝度寄与は FIREFLY_CLAMP でクランプされ、色相（比率）は保たれる。
    #[test]
    fn area_light_nee_contribution_is_clamped() {
        let (mat, p, n, ray) = floor_setup();
        let ls = LightSample {
            position: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            emit: Color::new(2e5, 1e5, 5e4),
            pdf: 1.0,
        };
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, &ls);
        assert!((c.luminance() - FIREFLY_CLAMP).abs() < 1e-9, "luminance = {}", c.luminance());
        assert!((c.r() / c.g() - 2.0).abs() < 1e-9);
    }

    /// 閾値以下の NEE 寄与はクランプの影響を受けない。
    #[test]
    fn dim_area_light_nee_contribution_is_unchanged() {
        let (mat, p, n, ray) = floor_setup();
        let ls = LightSample {
            position: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            emit: Color::new(1.0, 1.0, 1.0),
            pdf: 1.0,
        };
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, &ls);
        let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), Vec3::new(0.0, 1.0, 0.0), n);
        let expected = f.r() * mis_weight(1.0, pdf_bsdf);
        assert!((c.r() - expected).abs() < 1e-12, "{} vs {}", c.r(), expected);
    }

    /// Lambert の床（半径 1000 の球の上面、アルベド ρ）の真上に球光源（中心高さ 3、半径 1、放射輝度 L）、
    /// 背景は黒。床の原点で観測される放射輝度は直接照明だけで ρ·L·sin²θmax = ρ·L/9
    /// （床は凸なので床から床への相互反射はなく、発光体は反射しない）。
    fn floor_under_sphere_light() -> (World, Vec<Material>, EnvMap) {
        use crate::geometry::Sphere;
        let mats = vec![
            Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) },
            Material::DiffuseLight { emit: Color::new(4.0, 4.0, 4.0) },
        ];
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, -1000.0, 0.0), r: 1000.0, mat_id: 0 });
        world.add_sphere(Sphere { c: Vec3::new(0.0, 3.0, 0.0), r: 1.0, mat_id: 1 });
        world.build_lights(&mats);
        (world, mats, EnvMap::constant(Color::new(0.0, 0.0, 0.0)))
    }

    /// 平均と標準誤差。
    fn estimate(world: &World, mats: &[Material], env: &EnvMap, ray: Ray, limits: PathLimits, n: usize, seed: u64) -> (f64, f64) {
        let mut rng = Rng::new(seed);
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = radiance(world, mats, Some(env), ray, &mut rng, limits).r();
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        (mean, ((s2 / n as f64 - mean * mean).max(0.0) / n as f64).sqrt())
    }

    /// max_depth は Mitsuba と同じパス長: 0 は何も寄与せず、1 は直接見える発光体だけ、
    /// 2 以上では直接照明が理論値に一致する（最後の長さでも NEE と BSDF 側の発光ヒットが対で揃う）。
    /// 旧実装は「最大バウンス数」で、最後の長さの NEE だけが加算され MIS の相方が欠けて暗くなっていた。
    #[test]
    fn max_depth_follows_mitsuba_path_length() {
        let (world, mats, env) = floor_under_sphere_light();
        let to_light = Ray { o: Vec3::new(0.0, 6.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 };
        let to_floor = Ray { o: Vec3::new(2.0, 1.0, 0.0), d: Vec3::new(-2.0, -1.0, 0.0).norm(), time: 0.0 };
        let limits = |max_depth| PathLimits { max_depth, rr_depth: 1000 };

        let mut rng = Rng::new(1);
        assert_eq!(radiance(&world, &mats, Some(&env), to_light, &mut rng, limits(0)).r(), 0.0);
        assert_eq!(radiance(&world, &mats, Some(&env), to_light, &mut rng, limits(1)).r(), 4.0);
        assert_eq!(radiance(&world, &mats, Some(&env), to_floor, &mut rng, limits(1)).r(), 0.0);

        let exact = 0.5 * 4.0 / 9.0;
        for max_depth in [2usize, 3, 8, usize::MAX] {
            let (mean, se) = estimate(&world, &mats, &env, to_floor, limits(max_depth), 200_000, 7);
            assert!((mean - exact).abs() < 5.0 * se + 1e-3 * exact, "max_depth={}: {} ± {} vs exact {}", max_depth, mean, se, exact);
        }
    }

    /// Russian Roulette をどの長さから始めても（rr_depth = 1 でも）推定は不偏。
    #[test]
    fn russian_roulette_start_does_not_bias_direct_light() {
        let (world, mats, env) = floor_under_sphere_light();
        let to_floor = Ray { o: Vec3::new(2.0, 1.0, 0.0), d: Vec3::new(-2.0, -1.0, 0.0).norm(), time: 0.0 };
        let exact = 0.5 * 4.0 / 9.0;
        for rr_depth in [1usize, 2, 5] {
            let (mean, se) = estimate(&world, &mats, &env, to_floor, PathLimits { max_depth: usize::MAX, rr_depth }, 200_000, 11);
            assert!((mean - exact).abs() < 5.0 * se + 1e-3 * exact, "rr_depth={}: {} ± {} vs exact {}", rr_depth, mean, se, exact);
        }
    }

    /// 負値チャネルを含む色は黒扱いしない（寄与を変えないため輝度判定を使わない）。
    #[test]
    fn is_black_is_exact_per_channel() {
        assert!(is_black(Color::new(0.0, 0.0, 0.0)));
        assert!(!is_black(Color::new(0.0, 1e-300, 0.0)));
        assert!(!is_black(Color::new(-0.1, 0.0, 0.0)));
    }
}
