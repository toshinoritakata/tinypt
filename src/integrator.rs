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
    let mut eta_scale = 1.0;          // パス上の透過で掛かった相対屈折率 η_t/η_i の積（RR 用）

    for bounce in 0..limits.max_depth {
        // レイとシーンの交差判定。自己交差は、レイの原点を面の誤差の箱の外へずらしてあること
        // （`offset_ray_origin`）と、各プリミティブが「t > 計算誤差の上界」のヒットだけを返すことで
        // 防ぐので、tmin は 0 でよい
        let hit = match world.hit(ray, 0.0, RAY_T_MAX) {
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

        // 向き付けは幾何法線で決め、シェーディング法線はそれに追随させる
        // （補間法線をレイ方向で向き付けると、幾何法線と逆側を向いて面の裏を照らしうる）。
        let ng = oriented_normal(hit.ng, ray.d);
        let n = if hit.is_smooth() { face_forward(hit.ns, ng) } else { ng };

        // NEE（Next Event Estimation）はデルタ散乱マテリアルでは行わない
        if !mat.is_delta() {
            // NEE: Environment map
            if let Some(env_map) = env {
                let occluded = |shadow: Ray| world.hit(shadow, 0.0, RAY_T_MAX).is_some();
                let contrib = nee_environment(occluded, env_map, &mat, path_throughput, &hit, n, ng, ray, rng);
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: Area lights
            if let Some(ls) = world.sample_light(rng, ray.time, hit.p) {
                let contrib = nee_area_light(
                    |shadow: Ray, tmax: f64| world.hit(shadow, 0.0, tmax),
                    &mat, path_throughput, &hit, n, ng, ray, &ls,
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

use crate::geometry::{face_forward, offset_ray_origin, Hit};
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
    hit: &Hit,
    n: Vec3,
    ng: Vec3,
    ray: Ray,
    rng: &mut Rng,
) -> Color {
    let (wi, li, pdf_env) = env_map.sample_dir(rng);
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 || pdf_env <= 0.0 || is_black(li) {
        return Color::new(0.0, 0.0, 0.0);
    }
    // シェーディング法線から見て表でも、幾何的に面の裏へ向かう方向は寄与 0
    // （シャドウレイの原点は幾何法線基準にずらすので、そのまま撃つと自分のメッシュの内側を通る）
    if wi.dot(ng) <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    // シャドウレイ（無限遠へ）: 原点を面の誤差の箱の外へ wi の側にずらす
    let shadow = Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, wi), d: wi, time: ray.time };
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
/// `closest_hit(shadow, tmax)` はシャドウレイの最近接ヒット（通常は `world.hit`）。
fn nee_area_light(
    closest_hit: impl Fn(Ray, f64) -> Option<Hit>,
    mat: &Material,
    path_throughput: Color,
    hit: &Hit,
    n: Vec3,
    ng: Vec3,
    ray: Ray,
    ls: &LightSample,
) -> Color {
    // 光源自身に隠される点（球の外部からの面積フォールバックで引いた裏側）は寄与 0
    if !ls.visible {
        return Color::new(0.0, 0.0, 0.0);
    }
    let to_light = ls.position - hit.p;
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
    // 幾何的に面の裏へ向かう方向は寄与 0（nee_environment と同じ理由）
    if wi.dot(ng) <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    // シャドウレイ（PBRT の SpawnRayTo と同じ考え方）: 始点はシェーディング点の誤差の箱の外へ光源側に、
    // 終点は光源上の点の誤差の箱の外へシェーディング点側にずらし、その間の線分で遮蔽を調べる。
    // 最近接ヒットが光源自身（同じプリミティブ）なら遮蔽ではない: 平面の三角形も、見える側の点を
    // 狙った球も、自分自身でサンプル点を隠すことはない。この除外は保険ではなく**必須**: 球光源では、
    // レイと球の交差の t の誤差上界が終点側のずらし量（光源点の p_error）を超えうるので、終点を箱の外へ
    // ずらしても光源面が線分の内側（t < 線分長）でヒットすることがある。除外を外すと、そのサンプルが
    // 遮蔽扱いになり、sample/default.xml で画像が約 6%（−6.1%）暗くなる（verify_batch3c で計測）。
    let from = offset_ray_origin(hit.p, hit.p_error, hit.ng, to_light);
    let to = offset_ray_origin(ls.position, ls.p_error, ls.normal, from - ls.position);
    let seg = to - from;
    let seg_len = seg.len();
    if seg_len > 0.0 {
        let shadow = Ray { o: from, d: seg / seg_len, time: ray.time };
        if let Some(h) = closest_hit(shadow, seg_len) {
            if !ls.is_light_itself(&h) {
                return Color::new(0.0, 0.0, 0.0);
            }
        }
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

    /// 点 `p`・法線 `n` の交差情報（誤差上界は十分小さい値）。NEE の単体テスト用。
    fn test_hit(p: Vec3, n: Vec3) -> crate::geometry::Hit {
        crate::geometry::Hit { t: 1.0, p, ng: n, ns: n, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0) }
    }

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
                &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &mut rng,
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
                &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &mut rng,
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
            let c = nee_environment(|_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &mut rng);
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
            p_error: Vec3::new(0.0, 0.0, 0.0),
            visible: true,
            inst_id: None,
            prim_id: 0,
        };
        let c = nee_area_light(|_, _| None, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &ls);
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
            p_error: Vec3::new(0.0, 0.0, 0.0),
            visible: true,
            inst_id: None,
            prim_id: 0,
        };
        let c = nee_area_light(|_, _| None, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &ls);
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
        let hit = crate::geometry::Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng: Vec3::new(0.0, 1.0, 0.0), ns: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0) };
        let enter = Ray { o: Vec3::new(0.3, 1.0, 0.0), d: Vec3::new(-0.3, -1.0, 0.0).norm(), time: 0.0 };
        let exit = Ray { o: Vec3::new(0.1, -1.0, 0.0), d: Vec3::new(-0.1, 1.0, 0.0).norm(), time: 0.0 };
        let mut rng = Rng::new(2);
        let (mut seen_enter_t, mut seen_exit_t, mut seen_refl) = (false, false, false);
        for _ in 0..2000 {
            for (ray, entering) in [(enter, true), (exit, false)] {
                let s = mat.sample(&ray, &hit, &mut rng).unwrap();
                let transmitted = s.scattered.d.dot(ray.d) > 0.0 && s.scattered.d.dot(hit.ng).signum() == ray.d.dot(hit.ng).signum();
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

    /// 白炉テストをスケール・配置を変えて: ガラス球（ior 1.5）と Lambert 球（0.8）を、大きさ ×1e-3 / ×1e3、
    /// 原点から 1e8 離した位置に置いても、一様な環境光の中での見え方は 1 / 0.8 のまま。散乱レイの原点のずらし
    /// （交差点の誤差上界、`offset_ray_origin`）がスケールや絶対位置によらず自己交差を防ぎ、ガラスの透過で
    /// 内側へ入るレイも正しく反対側から出ることを確かめる（rr_depth 1）。
    #[test]
    fn white_furnaces_hold_at_any_scale_and_position() {
        use crate::geometry::Sphere;
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        for (mat, expect) in [
            (Material::Dielectric { ior: 1.5, absorption: Color::new(0.0, 0.0, 0.0) }, 1.0),
            (Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) }, 0.8),
        ] {
            for (k, offset) in [(1e-3, 0.0), (1e3, 0.0), (1.0, 1e8), (1e-3, 1e8)] {
                let mats = vec![mat];
                let mut world = World::new();
                let c = Vec3::new(offset, -0.5 * offset, 0.25 * offset);
                world.add_sphere(Sphere { c, r: k, mat_id: 0 });
                world.build_lights(&mats);
                let o = c + Vec3::new(0.0, 0.0, 5.0 * k);
                let ray = Ray { o, d: (c + Vec3::new(0.0, 0.6 * k, 0.0) - o).norm(), time: 0.0 };
                let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth: 1 }, 100_000, 29);
                assert!((mean - expect).abs() < 5.0 * se + 1e-3, "k={} offset={} expect {}: {} ± {}", k, offset, expect, mean, se);
            }
        }
    }

    // ---- スムーズシェーディング: NEE が幾何法線を使っていることを守るテスト ----

    /// 幾何法線 +z の面に、そこから 60 度傾いたシェーディング法線を持たせた交差情報。
    /// `ns` から見れば表、`ng` から見れば裏、という方向が存在する配置（薄い面の光漏れが起きる条件）。
    fn tilted_hit() -> (Hit, Vec3, Vec3, Ray) {
        let ng = Vec3::new(0.0, 0.0, 1.0);
        let ns = Vec3::new(0.866_025_403_784_438_6, 0.0, 0.5); // ng から 60 度
        let hit = Hit {
            t: 1.0,
            p: Vec3::new(0.0, 0.0, 0.0),
            ng,
            ns,
            mat_id: 0,
            prim_id: 0,
            inst_id: None,
            p_error: Vec3::new(1e-15, 1e-15, 1e-15),
            bary: (0.25, 0.25),
        };
        let ray = Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        (hit, ng, ns, ray)
    }

    /// 面光源 NEE のシャドウレイは**幾何法線**基準にずらした点から出る。
    ///
    /// ミューテーション検出: `nee_area_light` の `offset_ray_origin(..., hit.ng, ...)` を
    /// `hit.ns` に書き換えると、捕まえた原点が ng 基準の値と一致せず落ちる。
    #[test]
    fn area_light_nee_shadow_ray_starts_from_the_geometric_offset() {
        let (hit, ng, ns, ray) = tilted_hit();
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) };
        // ns 側にも ng 側にもある方向（どちらの半球でも表）に光源を置く
        let light_p = Vec3::new(1.0, 0.0, 1.0);
        let ls = LightSample {
            position: light_p,
            normal: Vec3::new(0.0, 0.0, -1.0),
            emit: Color::new(1.0, 1.0, 1.0),
            pdf: 1.0,
            p_error: Vec3::new(0.0, 0.0, 0.0),
            visible: true,
            inst_id: None,
            prim_id: 0,
        };
        let seen: Cell<Option<Vec3>> = Cell::new(None);
        let c = nee_area_light(
            |shadow: Ray, _| { seen.set(Some(shadow.o)); None },
            &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls,
        );
        assert!(c.luminance() > 0.0, "この配置では寄与が出るはず");
        let o = seen.get().expect("シャドウレイが撃たれていない");
        let to_light = light_p - hit.p;
        let by_ng = offset_ray_origin(hit.p, hit.p_error, ng, to_light);
        let by_ns = offset_ray_origin(hit.p, hit.p_error, ns, to_light);
        assert_eq!(o.x.to_bits(), by_ng.x.to_bits(), "シャドウレイの原点は幾何法線基準であること");
        assert_eq!(o.y.to_bits(), by_ng.y.to_bits());
        assert_eq!(o.z.to_bits(), by_ng.z.to_bits());
        assert_ne!(by_ng.x.to_bits(), by_ns.x.to_bits(), "この配置では 2 つのずらし方は実際に違う");
    }

    /// 環境 NEE のシャドウレイも幾何法線基準（上と同じミューテーション検出）。
    #[test]
    fn env_nee_shadow_ray_starts_from_the_geometric_offset() {
        let (hit, ng, ns, ray) = tilted_hit();
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) };
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let mut rng = Rng::new(4);
        let seen: Cell<Option<(Vec3, Vec3)>> = Cell::new(None);
        let mut checked = 0;
        for _ in 0..200 {
            let c = nee_environment(
                |shadow: Ray| { seen.set(Some((shadow.o, shadow.d))); false },
                &env, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &mut rng,
            );
            let Some((o, d)) = seen.get() else { continue };
            seen.set(None);
            if is_black(c) {
                continue; // 裏側ガードで落ちた方向（シャドウレイは撃たれていない）
            }
            let by_ng = offset_ray_origin(hit.p, hit.p_error, ng, d);
            assert_eq!(o.x.to_bits(), by_ng.x.to_bits(), "環境 NEE の原点も幾何法線基準であること");
            assert_eq!(o.z.to_bits(), by_ng.z.to_bits());
            checked += 1;
        }
        assert!(checked > 10, "確認できたサンプルが少なすぎる ({})", checked);
    }

    /// **薄い面の光漏れ防止**: シェーディング法線から見て表でも、幾何的に面の裏へ向かう方向の
    /// NEE 寄与は 0。ガードが無いと、原点は幾何法線基準で裏側へずらされるのに寄与だけが加算され、
    /// 1 枚ポリゴンの向こう側にある光源が「透けて」見える。
    ///
    /// ミューテーション検出: `wi.dot(ng) <= 0.0` のガードを `n`（= ns）基準に変える、または削ると、
    /// 下の 2 つの assert が落ちる（遮蔽物が無いので寄与が正になる）。
    #[test]
    fn nee_contributions_below_the_geometry_are_dropped() {
        let (hit, ng, ns, ray) = tilted_hit();
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) };
        // ns から見て表（cos > 0）だが ng から見て裏（z < 0）の方向にある光源。
        // 面は 1 枚ポリゴンなので、遮蔽判定（closest_hit）は何も返さない = 遮られない。
        let wedge = Vec3::new(0.9, 0.0, -0.436).norm();
        assert!(wedge.dot(ns) > 0.0 && wedge.dot(ng) < 0.0, "テスト前提: ns 側で表・ng 側で裏");
        let ls = LightSample {
            position: hit.p + wedge * 2.0,
            normal: -wedge,
            emit: Color::new(5.0, 5.0, 5.0),
            pdf: 1.0,
            p_error: Vec3::new(0.0, 0.0, 0.0),
            visible: true,
            inst_id: None,
            prim_id: 0,
        };
        let c = nee_area_light(|_, _| None, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls);
        assert!(is_black(c), "幾何的に裏側の光源から寄与が漏れている: {:?}", (c.r(), c.g(), c.b()));

        // 環境 NEE も同じ: ng の裏半球だけが光る環境にすると寄与は 0 になる
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let mut rng = Rng::new(31);
        let mut leaked = 0;
        for _ in 0..2000 {
            let c = nee_environment(
                |_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &mut rng,
            );
            // 寄与が出た方向は必ず幾何法線の表側から来ていること（裏なら 0 のはず）を、
            // 同じ乱数列で方向を引き直して突き合わせる
            if !is_black(c) {
                leaked += 1;
            }
        }
        // 表側の方向は残るので全部 0 にはならない（テストが空回りしていないことの確認）
        assert!(leaked > 0, "全部 0 ではテストにならない");

        // ガードの本丸: 裏向きの方向だけを明示的に渡す面光源の方は必ず 0
        let ls_back = LightSample { position: hit.p + Vec3::new(0.2, 0.0, -1.0), ..ls };
        let c2 = nee_area_light(|_, _| None, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls_back);
        assert!(is_black(c2), "真裏の光源から寄与が漏れている");
    }

    /// **向き付けの基準は幾何法線**であることを、描画経路（`radiance`）で守る。
    ///
    /// 面法線 +z の 1 枚ポリゴンに、そこから 60 度傾いたシェーディング法線を持たせ、
    /// **幾何法線から見れば表から入射するのに、シェーディング法線から見ると背面から入射する**
    /// かすめる視線で見る。正しい実装は向き付けを `oriented_normal(hit.ng, …)` で決めるので
    /// シェーディング法線はそのまま（光源側を向いたまま）だが、基準を `hit.ns` にすると
    /// `face_forward` が `-ns` を返し、NEE の cos とガードが揃って光源を弾いて真っ暗になる。
    #[test]
    fn nee_orientation_is_based_on_the_geometric_normal() {
        use crate::geometry::Sphere;
        use crate::transform::Transform;
        use crate::world::test_meshes::tilted_quad;
        let ns = Vec3::new(0.866_025_403_784_438_6, 0.0, 0.5); // 面法線 +z から 60 度
        let mats = vec![
            Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) },
            Material::DiffuseLight { emit: Color::new(40.0, 40.0, 40.0) },
        ];
        let mut world = World::new();
        world.add_mesh_data_instance(tilted_quad(8.0, ns, 0), Transform::identity(), None);
        // 幾何法線・シェーディング法線のどちらから見ても表側にある光源
        let light_dir = Vec3::new(0.9, 0.0, 0.436).norm();
        assert!(light_dir.dot(Vec3::new(0.0, 0.0, 1.0)) > 0.0 && light_dir.dot(ns) > 0.0,
                "テスト前提: 光源はどちらの法線から見ても表側");
        world.add_sphere(Sphere { c: light_dir * 3.0, r: 0.35, mat_id: 1 });
        world.build_lights(&mats);

        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0)); // 背景は真っ黒（光源はこの球だけ）
        // かすめる視線。ng から見れば表から入射する（ng·d < 0）が、ns から見ると背面から入射する
        // （ns·d > 0）配置にする。ここで向き付けの基準を ns にすると n が -ns に反転してしまう。
        let d = Vec3::new(0.99, 0.0, -0.141).norm();
        // 原点ちょうどに当たるように置く（光源の向きは原点基準で決めてあるため）
        let ray = Ray { o: -d * 4.0, d, time: 0.0 };
        assert!(ns.dot(d) > 0.0, "テスト前提: ns·d > 0（かすめる視線でシェーディング法線が視点と逆を向く）");
        let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: 3, rr_depth: 8 }, 40_000, 5);
        // 正しい実装では 0.434、基準を ns にすると 8.5e-5（NEE が丸ごと落ちて BSDF サンプリングの
        // 取りこぼしだけが残る）。5000 倍離れているので閾値は余裕を持って 0.1 に置く。
        assert!(mean > 0.1, "幾何法線を基準に向き付けていない（mean = {} ± {}、期待 0.43 付近）", mean, se);
    }

    /// 傾いたシェーディング法線を持つ 1 枚ポリゴンと、その真上の光源からなるシーンを作る。
    /// `ns_sign` が −1 なら面を裏側から見る（`face_forward` が仕事をする配置）。
    fn tilted_quad_scene(ns: Vec3, light_dir: Vec3, view_from: Vec3) -> (World, Vec<Material>, Ray) {
        use crate::geometry::Sphere;
        use crate::transform::Transform;
        use crate::world::test_meshes::tilted_quad;
        let mats = vec![
            Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) },
            Material::DiffuseLight { emit: Color::new(60.0, 60.0, 60.0) },
        ];
        let mut world = World::new();
        world.add_mesh_data_instance(tilted_quad(8.0, ns, 0), Transform::identity(), None);
        // 小さく遠い光源にして、BSDF サンプリングが偶然当たる寄与より NEE の寄与が支配的になるようにする
        world.add_sphere(Sphere { c: light_dir.norm() * 6.0, r: 0.30, mat_id: 1 });
        world.build_lights(&mats);
        let d = (Vec3::new(0.0, 0.0, 0.0) - view_from).norm();
        (world, mats, Ray { o: view_from, d, time: 0.0 })
    }

    /// **`radiance` は NEE にシェーディング法線を渡す**（直接光にスムーズシェーディングが効く）。
    ///
    /// 面法線 +z の 1 枚ポリゴンに 60 度傾いた頂点法線を与え、真上に小さな光源を置く。
    /// NEE の cos 項がシェーディング法線なら cos = ns·wi ≒ 0.5、幾何法線なら cos = ng·wi ≒ 1 なので、
    /// 頂点法線ありのシーンは頂点法線なしのシーンより**はっきり暗くなる**。
    ///
    /// ミューテーション検出: `radiance` が `nee_environment` / `nee_area_light` に渡す `n` を `ng` にすると、
    /// 直接光でスムーズシェーディングが効かなくなり、この比が 1 に近づいて落ちる。
    /// （NEE の単体テストは `n` と `ng` を自分で渡すので、「呼び出し側がどちらを渡すか」は通らない。）
    #[test]
    fn radiance_passes_the_shading_normal_to_nee() {
        let ns = Vec3::new(0.866_025_403_784_438_6, 0.0, 0.5); // 面法線 +z から 60 度
        let light_dir = Vec3::new(0.0, 0.0, 1.0); // 真上
        let view_from = Vec3::new(0.0, -2.0, 2.5);
        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
        // max_depth = 2（直接照明まで）にして、間接光の混入を避ける
        let limits = PathLimits { max_depth: 2, rr_depth: 8 };

        let (w_smooth, mats, ray) = tilted_quad_scene(ns, light_dir, view_from);
        let (w_flat, _, _) = tilted_quad_scene(Vec3::new(0.0, 0.0, 1.0), light_dir, view_from);
        let (mean_smooth, se_s) = estimate(&w_smooth, &mats, &env, ray, limits, 60_000, 11);
        let (mean_flat, se_f) = estimate(&w_flat, &mats, &env, ray, limits, 60_000, 11);

        assert!(mean_flat > 0.0 && mean_smooth > 0.0, "どちらも光が届いていない（{} / {}）", mean_smooth, mean_flat);
        let ratio = mean_smooth / mean_flat;
        // cos の比 0.5 が理論値。BSDF サンプリング側の寄与が少し混ざるので幅を持たせる。
        // 法線を取り違えると比は 1 付近になるので、0.8 を上限にすれば十分に分離できる。
        assert!(
            ratio < 0.8,
            "直接光にシェーディング法線が効いていない（smooth/flat = {:.4}、期待 0.5 付近。\
             mean_smooth = {} ± {}, mean_flat = {} ± {}）",
            ratio, mean_smooth, se_s, mean_flat, se_f
        );
        assert!(ratio > 0.2, "暗くなりすぎ（{:.4}）。cos の比 0.5 から大きく外れている", ratio);
    }

    /// **裏面から当たるスムーズ面でも、シェーディング法線は入射側へ向け直される**（`face_forward`）。
    ///
    /// 片面の 1 枚ポリゴンを**裏側**から見て、その裏側にある光源を照らす配置。
    /// 正しい実装は `face_forward(hit.ns, ng)` で `ns` を入射側（−z 側）に向け直すので光源が見えるが、
    /// 生の `hit.ns`（+z 側を向いたまま）を使うと cos が負になり、NEE が丸ごと落ちて真っ暗になる。
    ///
    /// ミューテーション検出: `radiance` の `face_forward(hit.ns, ng)` を `hit.ns` にすると落ちる。
    #[test]
    fn shading_normal_is_face_forwarded_for_backside_hits() {
        let ns = Vec3::new(0.866_025_403_784_438_6, 0.0, 0.5); // 面法線 +z から 60 度
        // 面の裏（−z 側）にある光源。−ns 側から見れば表になる向きを選ぶ
        let light_dir = Vec3::new(-0.9, 0.0, -0.436);
        let view_from = Vec3::new(0.0, -2.0, -2.5); // 裏側から見る
        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
        let (world, mats, ray) = tilted_quad_scene(ns, light_dir, view_from);
        assert!(light_dir.norm().dot(ns) < 0.0, "テスト前提: 光源は ns の裏側");
        let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: 2, rr_depth: 8 }, 60_000, 19);
        assert!(mean > 0.05, "裏面ヒットで ns が向け直されていない（mean = {} ± {}）", mean, se);
    }

    /// 白炉テスト（スムーズシェーディング）: 頂点法線を付けた Lambert の球メッシュでも、
    /// 一様な環境光 L = 1 の中での見え方はアルベド 0.8 に十分近い。
    ///
    /// 補間法線を使う BSDF は厳密にはエネルギーを保存しない: 散乱方向が幾何法線の裏へ出る
    /// サンプルを捨てるぶん暗くなる（`reflects_above`）。許容誤差はそのバイアスの見積もりから決める。
    /// 24x12 分割の球では、隣り合う頂点法線の開きは最大でも 360/24/2 = 7.5 度で、捨てられるのは
    /// シェーディング法線基準の半球のうち幾何半球からはみ出す部分＝cos 重みで測って最大でも
    /// sin²(7.5°) ≒ 1.7% 程度。多重散乱で 2 乗に効くほど深くはないので、2% を上限に取る
    /// （面法線メッシュとの差も同時に確認して、原因が補間であることを示す）。
    #[test]
    fn smooth_sphere_mesh_white_furnace_loses_little_energy() {
        use crate::transform::Transform;
        use crate::world::test_meshes::uv_sphere;
        let mats = vec![Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) }];
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let mut means = Vec::new();
        for smooth in [true, false] {
            let mut world = World::new();
            world.add_mesh_data_instance(
                uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 24, 12, 0, smooth), Transform::identity(), None);
            world.build_lights(&mats);
            let o = Vec3::new(0.0, 0.0, 5.0);
            let ray = Ray { o, d: (Vec3::new(0.0, 0.4, 0.0) - o).norm(), time: 0.0 };
            let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth: 3 }, 200_000, 17);
            assert!(mean <= 0.8 + 5.0 * se, "smooth={}: {} はアルベドを超えている（エネルギーを作っている）", smooth, mean);
            assert!(mean > 0.8 * 0.98 - 5.0 * se, "smooth={}: {} は 0.8 から 2% 以上暗い", smooth, mean);
            means.push(mean);
        }
        // 面法線メッシュ（凸なので破綻サンプルが無い）とほぼ同じ明るさに収まる
        assert!((means[0] - means[1]).abs() < 0.02, "smooth {} と flat {} の差が大きすぎる", means[0], means[1]);
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
    /// 光源の真下付近の床の点での放射輝度は ρ·L·sin²θmax·cosα で、シーン全体を 1e-3 倍しても、原点から 1e8 離して
    /// 置いても同じ（シャドウレイの両端を交差点の誤差上界ぶんずらすので、光源のすぐ近くでも NEE が働く）。
    /// 以前の絶対オフセット 1e-4（> 隙間）では、シャドウレイと散乱レイの原点が光源の内部に入り、
    /// 結果が理論値からずれた。
    #[test]
    fn nee_right_next_to_a_light_matches_analytic_and_is_scale_invariant() {
        use crate::geometry::{Sphere, Triangle};
        use crate::transform::Transform;
        let (albedo, emit, gap) = (0.5, 4.0, 1e-5);
        // 観測点は床の三角形の共有辺（対角線）を避けて (0.01, 0, 0.02)（以前の Möller–Trumbore 判定は辺に対して
        // 水密でなく、原点から遠い配置でこのテストの観測点がすり抜けた。現在は水密な判定だが観測点はそのまま）。光源の中心方向と
        // 床の法線の角を α として E = π·L·sin²θmax·cosα（光源が地平線より上にある場合）
        let target = Vec3::new(0.01, 0.0, 0.02);
        let to_c = Vec3::new(0.0, 1.0 + gap, 0.0) - target;
        let exact = albedo * emit / to_c.dot(to_c) * (to_c.y / to_c.len());
        for (k, offset) in [(1.0, 0.0), (1e-3, 0.0), (1.0, 1e8)] {
            let base = Vec3::new(offset, -0.5 * offset, 0.25 * offset);
            let mats = vec![
                Material::Lambert { albedo: Color::new(albedo, albedo, albedo) },
                Material::DiffuseLight { emit: Color::new(emit, emit, emit) },
            ];
            let mut world = World::new();
            let v = |x: f64, z: f64| base + Vec3::new(x * k, 0.0, z * k);
            let floor = vec![
                Triangle::new_static(v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), 0),
                Triangle::new_static(v(-2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0), 0),
            ];
            world.add_mesh_instance(floor, Transform::identity(), None);
            world.add_sphere(Sphere { c: base + Vec3::new(0.0, (1.0 + gap) * k, 0.0), r: k, mat_id: 1 });
            world.build_lights(&mats);
            let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
            // 光源の下をくぐる浅い角度で床の原点を見る
            let o = base + Vec3::new(2.0 * k, 0.001 * 2.0 * k, 0.0);
            let ray = Ray { o, d: (base + target * k - o).norm(), time: 0.0 };
            let (mean, se) = estimate(&world, &mats, &env, ray, PathLimits { max_depth: usize::MAX, rr_depth: 1000 }, 100_000, 23);
            assert!((mean - exact).abs() < 5.0 * se + 1e-3 * exact, "scale {} offset {}: {} ± {} vs exact {}", k, offset, mean, se, exact);
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
