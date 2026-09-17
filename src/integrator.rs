//! MIS（Multiple Importance Sampling）付きパストレーシング積分器。
//!
//! 単方向パストレーサーとして以下の機能を実装:
//! - **NEE（Next Event Estimation）**: 各バウンスで光源を直接サンプリングし直接照明を推定
//! - **MIS**: BSDF サンプリングと光源サンプリングを Power Heuristic (β=2) で統合
//! - **Russian Roulette**: スループットに基づく確率的なパス打ち切り（不偏性を維持）。
//!   生存確率に Mitsuba 3 と同じ透過の η² 補償を入れる。判定位置は Mitsuba と異なり、
//!   BSDF 重みを掛ける前（BSDF サンプリングの直前）: rr_depth が浅いときに拡散領域の効率が落ちないため
//! - **Firefly クランプ**: 異常に明るい寄与を輝度ベースでクランプ。MIS の両側（BSDF サンプリングで
//!   光源/背景に当たった寄与と、NEE の寄与）に寄与単位で同じ閾値を掛け、対称に保つ
//!
//! ## レンダリング方程式
//! L_o(x, ω_o) = L_e(x, ω_o) + ∫ f(x, ω_i, ω_o) L_i(x, ω_i) cos(θ_i) dω_i
//!
//! パストレーシングではこの積分をモンテカルロ推定で近似する。

use crate::constants::path::FIREFLY_CLAMP;
use crate::constants::RAY_T_MAX;
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
    let mut eta_scale = 1.0;
    // 自己交差回避オフセット（シーンの大きさに比例、`World::ray_epsilon`）。カメラレイ・散乱レイの
    // 交差判定の tmin と、シャドウレイの原点のずらし量・tmin に使う（散乱レイの原点は BSDF が
    // `Hit::ray_eps` でずらす）
    let ray_eps = world.ray_epsilon();          // パス上の透過で掛かった相対屈折率 η_t/η_i の積（RR 用）

    for bounce in 0..limits.max_depth {
        // レイとシーンの交差判定
        let hit = match world.hit(ray, ray_eps, RAY_T_MAX) {
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
                let occluded = |shadow: Ray| world.hit(shadow, ray_eps, RAY_T_MAX).is_some();
                let contrib = nee_environment(occluded, env_map, &mat, path_throughput, hit.p, n, ray, ray_eps, rng);
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: Area lights
            if let Some(ls) = world.sample_light(rng, ray.time, hit.p) {
                let contrib = nee_area_light(
                    |shadow: Ray, tmax: f64| world.hit(shadow, ray_eps, tmax).is_some(),
                    &mat, path_throughput, hit.p, n, ray, ray_eps, &ls,
                );
                accumulated_radiance = accumulated_radiance + contrib;
            }
        }

        // Russian Roulette: この頂点の発光・NEE の寄与を積んだ後、BSDF サンプリングの前に、
        // パスを延長するかどうかを確率的に決める（生存時は 1/p で補償するので不偏）。
        // 延長するパス長 bounce + 1 が rr_depth 以上なら判定する。
        //
        // 生存確率は max(throughput)·η² を [0.05, 0.95] にクランプしたもの（η² 補償は Mitsuba 3 と同じ）。
        // η² はパス上の透過で掛かった放射輝度の 1/η² 倍を打ち消す。これが無いとガラスに入っただけで
        // 生存確率が 1/η²（ior 1.5 で 0.44 倍）に下がり、ガラス内部を通る経路（コースティクス・全反射）
        // が強く打ち切られ、生き残った経路に大きな 1/p が掛かってノイズになる。
        //
        // 判定位置は Mitsuba 3（BSDF 重みを掛けた**後**の throughput で判定）と意図的に異なり、
        // この頂点の重みを掛ける**前**の throughput を使う。重み適用後で判定すると、拡散面では
        // 最初の散乱直後から生存確率が ≈ アルベドになり、rr_depth が浅い設定で拡散主体の領域の
        // ノイズが大きく増える（計測: rr_depth 1 で default.xml の拡散領域の分散×時間が 2.7 倍、
        // cornell で 1.26 倍）。重み適用前ならこの悪化がなく、η² 補償によるガラス領域の分散低下は同じ
        // （rr_depth 1: default 全体の分散×時間 0.62、cornell 0.99）。通常の設定（rr_depth 4〜5）では
        // 両者に差はない。
        let throughput_max = path_throughput.r().max(path_throughput.g()).max(path_throughput.b());
        if throughput_max <= 0.0 {
            break;
        }
        if bounce + 1 >= limits.rr_depth {
            let p = rr_survival_probability(throughput_max, eta_scale);
            if rng.next_f64() >= p {
                break;
            }
            path_throughput = path_throughput / p;
        }

        // BSDF サンプリング: 散乱レイ・スループット重み・PDF を BSDF から取得
        match mat.sample(&ray, &hit, rng) {
            Some(BsdfSample { scattered, weight, pdf, is_delta, eta }) => {
                last_bsdf_pdf = pdf;
                last_non_delta = !is_delta;
                last_p = hit.p;
                path_throughput = path_throughput.hadamard(weight);
                eta_scale *= eta;
                ray = scattered;
            }
            None => break,
        }

    }

    accumulated_radiance
}

/// Russian Roulette の生存確率: `max(throughput)·η²` を [0.05, 0.95] にクランプする。
/// `eta_scale` はこの頂点までのパス上の透過の相対屈折率 η_t/η_i の積（Mitsuba 3 の path 積分器と同じ η² 補償。
/// 下限 0.05 は tinypt 独自で、極端に小さい確率で生き残った経路の重みの爆発を抑える）。
fn rr_survival_probability(throughput_max: f64, eta_scale: f64) -> f64 {
    (throughput_max * eta_scale * eta_scale).min(0.95).max(0.05)
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
    ray_eps: f64,
    rng: &mut Rng,
) -> Color {
    let (wi, li, pdf_env) = env_map.sample_dir(rng);
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 || pdf_env <= 0.0 || is_black(li) {
        return Color::new(0.0, 0.0, 0.0);
    }

    let shadow = Ray { o: hit_p + ray_eps * wi, d: wi, time: ray.time };
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
    ray_eps: f64,
    ls: &LightSample,
) -> Color {
    let to_light = ls.position - hit_p;
    let dist2 = to_light.dot(to_light);
    let dist = dist2.sqrt();
    // 方向が定義できない距離 0 だけを除く（以前の絶対しきい値 1e-6 はシーンのスケールに依存していた）
    if !(dist > 0.0) {
        return Color::new(0.0, 0.0, 0.0);
    }

    let wi = to_light / dist;
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    // シャドウレイ: 原点を ray_eps だけ光源側へ進め、光源上の点の手前で止める。光源は新しい原点から
    // dist − ray_eps にあり、光源側にもシェーディング点側と対称に ray_eps の余裕を取って
    // tmax = dist·(1 − 1e-9) − 2·ray_eps とする（光源自身を遮蔽物と判定しない。相対マージン 1e-9 は
    // 座標が大きく光源が近いときの丸め対策）。区間が空（光源が ~3·ray_eps より近い）なら、間に遮蔽物は
    // 置けないので遮蔽なしとする。
    let shadow = Ray { o: hit_p + ray_eps * wi, d: wi, time: ray.time };
    let tmax = dist * (1.0 - 1e-9) - 2.0 * ray_eps;
    if tmax > ray_eps && occluded(shadow, tmax) {
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
                &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, 1e-4, &mut rng,
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
                &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, 1e-4, &mut rng,
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
            let c = nee_environment(|_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, 1e-4, &mut rng);
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
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, 1e-4, &ls);
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
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), p, n, ray, 1e-4, &ls);
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

    /// 生存確率は max(throughput)·η² を [0.05, 0.95] にクランプしたもの。
    #[test]
    fn rr_survival_probability_compensates_eta_squared() {
        // ガラス（ior 1.5）に入った直後: 放射輝度の重み 1/1.5² と η_t/η_i = 1.5 が打ち消し合う
        let after_entering = 1.0 / (1.5 * 1.5);
        assert!((rr_survival_probability(after_entering, 1.5) - 0.95).abs() < 1e-12);
        // 補償なし（η = 1）なら 0.444 に下がる
        assert!((rr_survival_probability(after_entering, 1.0) - after_entering).abs() < 1e-12);
        // 出た後は η の積が 1 に戻る
        assert!((rr_survival_probability(0.3, 1.5 * (1.0 / 1.5)) - 0.3).abs() < 1e-12);
        // クランプ
        assert_eq!(rr_survival_probability(1e-6, 1.0), 0.05);
        assert_eq!(rr_survival_probability(10.0, 1.0), 0.95);
    }

    /// 誘電体の BsdfSample.eta: 入る透過で ior、出る透過で 1/ior、反射で 1。
    /// 吸収なしの透過では weight·eta² = 1（放射輝度の η² 倍率が eta で打ち消される）。
    #[test]
    fn dielectric_sample_reports_relative_ior() {
        let ior = 1.5;
        let mat = Material::Dielectric { ior, absorption: Color::new(0.0, 0.0, 0.0) };
        let hit = crate::geometry::Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), n: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, ray_eps: 1e-4 };
        let enter = Ray { o: Vec3::new(0.3, 1.0, 0.0), d: Vec3::new(-0.3, -1.0, 0.0).norm(), time: 0.0 };
        let exit = Ray { o: Vec3::new(0.1, -1.0, 0.0), d: Vec3::new(-0.1, 1.0, 0.0).norm(), time: 0.0 };
        let mut rng = Rng::new(2);
        let (mut seen_enter_t, mut seen_exit_t, mut seen_refl) = (false, false, false);
        for _ in 0..2000 {
            for (ray, entering) in [(enter, true), (exit, false)] {
                let s = mat.sample(&ray, &hit, &mut rng).unwrap();
                let transmitted = s.scattered.d.dot(ray.d) > 0.0 && s.scattered.d.dot(hit.n).signum() == ray.d.dot(hit.n).signum();
                if transmitted {
                    let expect = if entering { ior } else { 1.0 / ior };
                    assert!((s.eta - expect).abs() < 1e-12, "transmission eta {} vs {}", s.eta, expect);
                    assert!((s.weight.r() * s.eta * s.eta - 1.0).abs() < 1e-9, "weight·eta² = {}", s.weight.r() * s.eta * s.eta);
                    if entering { seen_enter_t = true } else { seen_exit_t = true }
                } else {
                    assert_eq!(s.eta, 1.0);
                    seen_refl = true;
                }
            }
        }
        assert!(seen_enter_t && seen_exit_t && seen_refl);
    }

    /// 白炉テスト（ガラス）: 一様な環境光 L の中に置いた吸収のない誘電体球は、どこから見ても L に見える
    /// （反射と透過の和が 1、入射と射出の η² 倍率が打ち消し合う）。深さ無制限・RR を最初の頂点から
    /// 効かせても（rr_depth = 1）平均が L に一致し、RR の η² 補償と判定位置が不偏であることを確かめる。
    #[test]
    fn glass_sphere_white_furnace_is_unbiased_with_russian_roulette() {
        use crate::geometry::Sphere;
        let mats = vec![Material::Dielectric { ior: 1.5, absorption: Color::new(0.0, 0.0, 0.0) }];
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        world.build_lights(&mats);
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        for (rr_depth, target_y) in [(1usize, 0.0), (1, 0.6), (1, 0.95), (3, 0.6)] {
            let o = Vec3::new(0.0, 0.0, 5.0);
            let ray = Ray { o, d: (Vec3::new(0.0, target_y, 0.0) - o).norm(), time: 0.0 };
            let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth }, 200_000, 13);
            assert!((mean - 1.0).abs() < 5.0 * se + 1e-3, "rr_depth={} y={}: {} ± {} vs 1", rr_depth, target_y, mean, se);
        }
    }

    /// 白炉テスト（拡散）: 一様な環境光 L = 1 の中の凸な Lambert 球（アルベド 0.8）は、どこから見ても 0.8
    /// （反射光は球に再び当たらない）。rr_depth = 1 で RR を最初の頂点から効かせても平均が 0.8 に一致する。
    /// 環境光の BSDF 側の MIS の取り分が大きいので、生存時の 1/p の補償を忘れると大きくずれる
    /// （ガラスの白炉テストとは独立に、拡散経路で RR の補償を検証する）。
    #[test]
    fn lambert_sphere_white_furnace_is_unbiased_with_russian_roulette() {
        use crate::geometry::Sphere;
        let mats = vec![Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) }];
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        world.build_lights(&mats);
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        for (rr_depth, target_y) in [(1usize, 0.0), (1, 0.7), (2, 0.3)] {
            let o = Vec3::new(0.0, 0.0, 5.0);
            let ray = Ray { o, d: (Vec3::new(0.0, target_y, 0.0) - o).norm(), time: 0.0 };
            let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth }, 200_000, 17);
            assert!((mean - 0.8).abs() < 5.0 * se + 1e-3, "rr_depth={} y={}: {} ± {} vs 0.8", rr_depth, target_y, mean, se);
        }
    }

    /// 光源のすぐ近くの NEE: 三角形の床（4×4、アルベド 0.5）の真上、隙間 1e-5 に球光源（半径 1、L = 4）。
    /// 床の原点での放射輝度は ρ·L·sin²θmax = ρ·L/(1 + h)² で、シーン全体を 1e-3 倍しても同じ。
    /// 以前の絶対オフセット 1e-4（> 隙間）では、シャドウレイと散乱レイの原点が光源の内部に入り、
    /// 結果が理論値からずれた。
    #[test]
    fn nee_right_next_to_a_light_matches_analytic_and_is_scale_invariant() {
        use crate::geometry::{Sphere, Triangle};
        use crate::transform::Transform;
        let (albedo, emit, gap) = (0.5, 4.0, 1e-5);
        let exact = albedo * emit / (1.0 + gap) / (1.0 + gap);
        for k in [1.0, 1e-3] {
            let mats = vec![
                Material::Lambert { albedo: Color::new(albedo, albedo, albedo) },
                Material::DiffuseLight { emit: Color::new(emit, emit, emit) },
            ];
            let mut world = World::new();
            let v = |x: f64, z: f64| Vec3::new(x * k, 0.0, z * k);
            let floor = vec![
                Triangle::new_static(v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), 0),
                Triangle::new_static(v(-2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0), 0),
            ];
            world.add_mesh_instance(floor, Transform::identity(), None);
            world.add_sphere(Sphere { c: Vec3::new(0.0, (1.0 + gap) * k, 0.0), r: k, mat_id: 1 });
            world.build_lights(&mats);
            assert!(world.ray_epsilon() < 0.1 * gap * k, "test setup: offset {} must be below the gap", world.ray_epsilon());
            let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
            // 光源の下をくぐる浅い角度で床の原点を見る
            let o = Vec3::new(2.0 * k, 0.001 * 2.0 * k, 0.0);
            let ray = Ray { o, d: (Vec3::new(0.0, 0.0, 0.0) - o).norm(), time: 0.0 };
            let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth: 1000 }, 100_000, 23);
            assert!((mean - exact).abs() < 5.0 * se + 1e-3 * exact, "scale {}: {} ± {} vs exact {}", k, mean, se, exact);
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
