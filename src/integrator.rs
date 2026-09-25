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

use crate::constants::normal_map::NS_NG_MIN;
use crate::constants::path::FIREFLY_CLAMP;
use crate::constants::RAY_T_MAX;
use crate::env::EnvMap;
use crate::material::{BsdfSample, Material};
use crate::math::{Color, Vec3};
use crate::medium::{hg_eval, hg_sample, Medium, MediumEvent};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::normal_map::{orthonormalize, MapId, NormalMap};
use crate::texture::Texture;
use crate::world::{DeltaLight, World};

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

/// PERF-3: 層化サンプリングの文脈。1 ピクセルの `spp` 本のサンプルのうち、いま何番目かを
/// 「層番号」として伝える。**最初のバウンス（`bounce == 0`）の NEE 面光源サンプリングだけ**が
/// これを使う（Sponza のように光源が小さく NEE 主体のシーンで効くのはここで、深いバウンスまで
/// 層化しても複雑さに見合わないため。README／レポート参照）。
#[derive(Clone, Copy)]
pub struct Strata {
    /// このサンプルの層番号（0 起点。ピクセルごとに乱択した回転を含む — 適応的サンプリングが
    /// 途中で打ち切っても、常に同じ部分格子だけが選ばれて偏らないようにするため）。
    pub stratum: usize,
    /// 格子の横方向の層数
    pub nx: usize,
    /// 格子の縦方向の層数（`nx * ny >= spp` で、`spp` が `nx` の倍数でなければ余りが出る）
    pub ny: usize,
}

impl Strata {
    /// この層番号に対応する 2 次元乱数 [0,1)²（`rng` で層内をジッターする）。
    fn uv(&self, rng: &mut Rng) -> (f64, f64) {
        let cx = self.stratum % self.nx;
        let cy = (self.stratum / self.nx) % self.ny;
        let u = (cx as f64 + rng.next_f64()) / self.nx as f64;
        let v = (cy as f64 + rng.next_f64()) / self.ny as f64;
        (u, v)
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
    surfaces: &Surfaces,
    env: Option<&EnvMap>,
    medium: Option<&Medium>,
    ray: Ray,
    rng: &mut Rng,
    limits: PathLimits,
    strata: Option<Strata>,
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
        let hit = world.hit(ray, 0.0, RAY_T_MAX);

        // 参加媒質: 表面（または無限遠）までの区間で自由行程をサンプリングする。
        // `medium` が `None` なら丸ごと飛ばし、乱数も引かない（媒質の無いシーンの出力はビット単位で不変）。
        if let Some(med) = medium {
            let t_surface = hit.as_ref().map(|h| h.t).unwrap_or(RAY_T_MAX);
            match med.sample_distance(ray, t_surface, rng) {
                MediumEvent::Pass { weight } => {
                    path_throughput = path_throughput.hadamard(weight);
                }
                MediumEvent::Scatter { t, weight } => {
                    path_throughput = path_throughput.hadamard(weight);
                    // 深さの数え方は表面頂点と同じ（散乱を 1 頂点と数える。Mitsuba の depth と同じ）
                    if bounce + 2 > limits.max_depth {
                        break;
                    }
                    let p = ray.o + ray.d * t;
                    let wo = (-ray.d).norm();

                    // NEE（環境光・面光源）。表面版との違い: cos 項なし・原点ずらしなし・裏面棄却なし、
                    // f = 位相関数。別関数にしてあるのは表面版のバイト一致を守るため
                    if let Some(env_map) = env {
                        let occluded = |shadow: Ray| world.occluded(shadow, 0.0, RAY_T_MAX, None);
                        let contrib = nee_environment_phase(occluded, env_map, med, path_throughput, p, wo, ray.time, rng);
                        accumulated_radiance = accumulated_radiance + contrib;
                    }
                    let ls = match (bounce, strata) {
                        (0, Some(s)) => {
                            let uv = s.uv(rng);
                            world.sample_light_with_uv(rng, ray.time, p, uv)
                        }
                        _ => world.sample_light(rng, ray.time, p),
                    };
                    if let Some(ls) = ls {
                        let contrib = nee_area_light_phase(
                            |shadow: Ray, tmax: f64| world.occluded(shadow, 0.0, tmax, Some((ls.inst_id, ls.prim_id))),
                            med, path_throughput, p, wo, ray.time, &ls,
                        );
                        accumulated_radiance = accumulated_radiance + contrib;
                    }
                    // デルタ光源（乱数を引かない。0 個なら空回り）
                    for dl in world.delta_lights() {
                        let occluded = |shadow: Ray, tmax: f64| world.occluded(shadow, 0.0, tmax, None);
                        let contrib = nee_delta_light_phase(occluded, dl, med, path_throughput, p, wo, ray.time);
                        accumulated_radiance = accumulated_radiance + contrib;
                    }

                    // Russian Roulette: 表面と完全に同じ規則（省くと光学的に厚い媒質で経路が終わらない）。
                    // `eta_scale` は変えない（η² 補償は透過専用）
                    let throughput_max = path_throughput.r().max(path_throughput.g()).max(path_throughput.b());
                    if throughput_max <= 0.0 {
                        break;
                    }
                    if bounce + 1 >= limits.rr_depth {
                        let p_rr = rr_survival_probability(throughput_max, eta_scale);
                        if rng.next_f64() >= p_rr {
                            break;
                        }
                        path_throughput = path_throughput / p_rr;
                    }

                    // 位相関数サンプリング。HG は位相関数そのものに比例してサンプルする（完全重点サンプリング）
                    // ので weight = f / pdf = 1 で、`path_throughput` には何も掛けない
                    let (wi, pdf) = hg_sample(wo, med.g, rng);
                    ray = Ray { o: p, d: wi, time: ray.time };
                    last_bsdf_pdf = pdf;
                    last_non_delta = true;
                    last_p = p;
                    continue;
                }
            }
        }

        let hit = match hit {
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

        // テクスチャはここで 1 度だけ交差点の UV で評価し、以降の BSDF はテクスチャを知らない
        let mat = mats[hit.mat_id].resolve_textures(surfaces.textures, hit.uv);

        // 法線マップ／バンプマップ: シェーディング法線 `ns` だけを摂動する（1 か所。NEE も `Material::sample` も
        // この後の `hit.ns` を見るので、両方が同じ摂動後の法線になる）。`ng` / `p` / `p_error` には触れない
        // （原点ずらし・表裏判定・光源の面積と pdf は幾何法線基準のまま）。
        let mut hit = hit;
        if let Some(map_id) = surfaces.map_for(hit.mat_id) {
            perturb_shading_normal(world, surfaces, map_id, &mut hit, ray.time);
        }
        let hit = hit;

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
                // any-hit: 遮蔽の有無だけが要るので、最初に見つかった交差で打ち切る（最近接は不要）。
                // 環境光には「除外すべき光源自身」が無いので skip は None
                let occluded = |shadow: Ray| world.occluded(shadow, 0.0, RAY_T_MAX, None);
                let contrib = nee_environment(occluded, env_map, &mat, path_throughput, &hit, n, ng, ray, medium, rng);
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: Area lights
            // 最初のバウンスだけ、層化した 2 次元乱数で光源面上の点を選ぶ（PERF-3）。
            // 深いバウンスは従来どおり rng から直接引く（層化しない）
            let ls = match (bounce, strata) {
                (0, Some(s)) => {
                    let uv = s.uv(rng);
                    world.sample_light_with_uv(rng, ray.time, hit.p, uv)
                }
                _ => world.sample_light(rng, ray.time, hit.p),
            };
            if let Some(ls) = ls {
                let contrib = nee_area_light(
                    // any-hit + 光源自身の除外（`(ls.inst_id, ls.prim_id)` に一致する交差は遮蔽と数えない）
                    |shadow: Ray, tmax: f64| world.occluded(shadow, 0.0, tmax, Some((ls.inst_id, ls.prim_id))),
                    &mat, path_throughput, &hit, n, ng, ray, &ls, medium,
                );
                accumulated_radiance = accumulated_radiance + contrib;
            }
            // NEE: デルタ光源（点・平行・スポット）。乱数を引かず、MIS もしない（BSDF サンプリングでは当たらない）。
            // 0 個ならループが空回りするだけで、乱数の消費列は変わらない
            for dl in world.delta_lights() {
                let occluded = |shadow: Ray, tmax: f64| world.occluded(shadow, 0.0, tmax, None);
                let contrib = nee_delta_light(occluded, dl, &mat, path_throughput, &hit, n, ng, ray, medium);
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

/// 材質ごとのサーフェス属性（色テクスチャと法線マップ）への参照の束。
pub struct Surfaces<'a> {
    pub textures: &'a [Texture],
    pub normal_maps: &'a [NormalMap],
    /// `mat_id` → `normal_maps` の添字。空ならマップ無し（[`Scene::mat_maps`](crate::scene::Scene::mat_maps) の不変条件）
    pub mat_maps: &'a [Option<MapId>],
}

impl<'a> Surfaces<'a> {
    /// マップ無し・テクスチャ無し（テスト用）。
    pub const fn none() -> Surfaces<'static> {
        Surfaces { textures: &[], normal_maps: &[], mat_maps: &[] }
    }

    /// 色テクスチャだけ（法線マップ無し）。
    pub const fn textures_only(textures: &'a [Texture]) -> Surfaces<'a> {
        Surfaces { textures, normal_maps: &[], mat_maps: &[] }
    }

    /// 材質 `mat_id` の法線マップ。テーブルが空なら即 `None`（マップを使わないシーンのコストは分岐 1 つ）。
    #[inline]
    fn map_for(&self, mat_id: usize) -> Option<MapId> {
        if self.mat_maps.is_empty() {
            return None;
        }
        self.mat_maps.get(mat_id).copied().flatten()
    }
}

/// `hit.ns` を材質のマップで摂動する（`ng` などは不変）。接空間を作れない（球・UV 無し・UV 縮退・退化）、
/// または摂動後が幾何法線の地平線を割る場合は何もしない（元の `ns` のまま）。
fn perturb_shading_normal(world: &World, surfaces: &Surfaces, map_id: MapId, hit: &mut Hit, time: f64) {
    let Some(map) = surfaces.normal_maps.get(map_id as usize) else { return };
    let Some((dpdu, dpdv)) = world.surface_tangents(hit, time) else { return };
    let Some((t, b)) = orthonormalize(dpdu, dpdv, hit.ns) else { return };
    let n_pert = map.perturb(hit.uv, t, b, hit.ns, dpdu.len(), dpdv.len());
    // 摂動結果を幾何法線と同じ側へ揃える。それでも地平線すれすれなら採用しない
    // （裏返すと `ns·ng > 0` が壊れ、原点ずらしが反対側へ出て自己交差する）
    let n_pert = face_forward(n_pert, hit.ng);
    if n_pert.dot(hit.ng) > NS_NG_MIN {
        hit.ns = n_pert;
    }
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
    medium: Option<&Medium>,
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
    let mut c = path_throughput.hadamard(f).hadamard(li) * (cos * w / pdf_env);
    // 参加媒質内のシャドウレイは減衰する（None なら何もしない）。クランプは透過率を掛けた後
    if let Some(med) = medium {
        c = c.hadamard(med.transmittance(shadow, 0.0, RAY_T_MAX));
    }
    c.clamp_luminance(FIREFLY_CLAMP)
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
/// `occluded(shadow, tmax)` はシャドウレイの遮蔽判定（any-hit。通常は `world.occluded`）。
/// 光源自身（同じプリミティブ）への交差は、呼び出し側があらかじめ除外して渡す前提
/// （`World::occluded` の `skip` 引数。理由は下のコメント参照）。
fn nee_area_light(
    occluded: impl Fn(Ray, f64) -> bool,
    mat: &Material,
    path_throughput: Color,
    hit: &Hit,
    n: Vec3,
    ng: Vec3,
    ray: Ray,
    ls: &LightSample,
    medium: Option<&Medium>,
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
    // 光源自身（同じプリミティブ）への交差は遮蔽ではない: 平面の三角形も、見える側の点を狙った球も、
    // 自分自身でサンプル点を隠すことはない。この除外は保険ではなく**必須**: 球光源では、
    // レイと球の交差の t の誤差上界が終点側のずらし量（光源点の p_error）を超えうるので、終点を箱の外へ
    // ずらしても光源面が線分の内側（t < 線分長）でヒットすることがある。除外を外すと、そのサンプルが
    // 遮蔽扱いになり、sample/default.xml で画像が約 6%（−6.1%）暗くなる（verify_batch3c で計測）。
    // any-hit 化した現在は `occluded` の呼び出し側（`World::occluded` の `skip` 引数）がこの除外を担う。
    let from = offset_ray_origin(hit.p, hit.p_error, hit.ng, to_light);
    let to = offset_ray_origin(ls.position, ls.p_error, ls.normal, from - ls.position);
    let seg = to - from;
    let seg_len = seg.len();
    let mut transmittance = None;
    if seg_len > 0.0 {
        let shadow = Ray { o: from, d: seg / seg_len, time: ray.time };
        if occluded(shadow, seg_len) {
            return Color::new(0.0, 0.0, 0.0);
        }
        if let Some(med) = medium {
            transmittance = Some(med.transmittance(shadow, 0.0, seg_len));
        }
    }

    let (f, pdf_bsdf) = mat.eval((-ray.d).norm(), wi, n);
    if ls.pdf <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    let w = mis_weight(ls.pdf, pdf_bsdf);
    // BSDF 側（発光体ヒット）と同じ閾値でクランプし、MIS の両側を対称にする
    let mut c = path_throughput.hadamard(f).hadamard(ls.emit) * (cos * w / ls.pdf);
    // 参加媒質内のシャドウレイは減衰する（None なら何もしない）。クランプは透過率を掛けた後
    if let Some(tr) = transmittance {
        c = c.hadamard(tr);
    }
    c.clamp_luminance(FIREFLY_CLAMP)
}

/// 媒質散乱点 `p` からの環境光 NEE。[`nee_environment`] との違い: cos 項なし（位相関数は立体角あたりの
/// 密度）、原点ずらしなし（面の上ではない）、裏面棄却なし、`f = hg_eval(wo·wi, g)`（`pdf_bsdf` も同じ値）。
/// 乱数の引き方（`sample_dir` を先に必ず 1 回）は表面版と同じ。
fn nee_environment_phase(
    occluded: impl Fn(Ray) -> bool,
    env_map: &EnvMap,
    med: &Medium,
    path_throughput: Color,
    p: Vec3,
    wo: Vec3,
    time: f64,
    rng: &mut Rng,
) -> Color {
    let (wi, li, pdf_env) = env_map.sample_dir(rng);
    if pdf_env <= 0.0 || is_black(li) {
        return Color::new(0.0, 0.0, 0.0);
    }
    let shadow = Ray { o: p, d: wi, time };
    if occluded(shadow) {
        return Color::new(0.0, 0.0, 0.0);
    }
    let f = hg_eval(wo.dot(wi), med.g);
    let w = mis_weight(pdf_env, f);
    let c = path_throughput.hadamard(li) * (f * w / pdf_env);
    c.hadamard(med.transmittance(shadow, 0.0, RAY_T_MAX)).clamp_luminance(FIREFLY_CLAMP)
}

/// 媒質散乱点 `p` からの面光源 NEE。[`nee_area_light`] との違いは [`nee_environment_phase`] と同じ。
fn nee_area_light_phase(
    occluded: impl Fn(Ray, f64) -> bool,
    med: &Medium,
    path_throughput: Color,
    p: Vec3,
    wo: Vec3,
    time: f64,
    ls: &LightSample,
) -> Color {
    if !ls.visible {
        return Color::new(0.0, 0.0, 0.0);
    }
    let to_light = ls.position - p;
    let dist = to_light.dot(to_light).sqrt();
    if !(dist > 0.0) {
        return Color::new(0.0, 0.0, 0.0);
    }
    let wi = to_light / dist;
    // 始点は散乱点そのもの。終点は光源側だけ表面版と同じく誤差の箱の外へずらす
    let to = offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
    let seg = to - p;
    let seg_len = seg.len();
    let mut transmittance = Color::new(1.0, 1.0, 1.0);
    if seg_len > 0.0 {
        let shadow = Ray { o: p, d: seg / seg_len, time };
        if occluded(shadow, seg_len) {
            return Color::new(0.0, 0.0, 0.0);
        }
        transmittance = med.transmittance(shadow, 0.0, seg_len);
    }
    if ls.pdf <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }
    let f = hg_eval(wo.dot(wi), med.g);
    let w = mis_weight(ls.pdf, f);
    let c = path_throughput.hadamard(ls.emit) * (f * w / ls.pdf);
    c.hadamard(transmittance).clamp_luminance(FIREFLY_CLAMP)
}

/// デルタ光源（点・平行・スポット）に対する表面の NEE。1 個ぶんの寄与を返す。
///
/// `contrib = throughput ⊙ f ⊙ value · cos`。**pdf で割らず、MIS 重みも掛けない**: デルタ光源は BSDF
/// サンプリングでは絶対に当たらないので、MIS の相方がおらず重みは 1（掛けると暗くなる）。乱数は引かない。
/// 表面 NEE と同じく、`ng` の裏向き・cos ≤ 0 は寄与 0、シャドウレイの始点は面の誤差の箱の外へずらす。
/// 光源には幾何が無いので、遮蔽判定に光源自身の除外（skip）は無い。
/// 終点は、**ずらした始点から**光源位置までの距離を測り直して決める（始点がずれるので `distance` を流用しない）。
fn nee_delta_light(
    occluded: impl Fn(Ray, f64) -> bool,
    light: &DeltaLight,
    mat: &Material,
    path_throughput: Color,
    hit: &Hit,
    n: Vec3,
    ng: Vec3,
    ray: Ray,
    medium: Option<&Medium>,
) -> Color {
    let Some(dh) = light.sample_at(hit.p) else { return Color::new(0.0, 0.0, 0.0) };
    let wi = dh.wi;
    let cos = n.dot(wi).max(0.0);
    if cos <= 0.0 || wi.dot(ng) <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }
    let from = offset_ray_origin(hit.p, hit.p_error, hit.ng, wi);
    let (dir, tmax) = match dh.position {
        Some(pos) => {
            let seg = pos - from;
            let len = seg.len();
            if !(len > 0.0) {
                return Color::new(0.0, 0.0, 0.0);
            }
            (seg / len, len)
        }
        None => (wi, RAY_T_MAX),
    };
    let shadow = Ray { o: from, d: dir, time: ray.time };
    if occluded(shadow, tmax) {
        return Color::new(0.0, 0.0, 0.0);
    }
    let (f, _pdf) = mat.eval((-ray.d).norm(), wi, n);
    let mut c = path_throughput.hadamard(f).hadamard(dh.value) * cos;
    if let Some(med) = medium {
        c = c.hadamard(med.transmittance(shadow, 0.0, tmax));
    }
    c.clamp_luminance(FIREFLY_CLAMP)
}

/// 媒質散乱点 `p` からのデルタ光源 NEE。[`nee_delta_light`] との違いは他の `*_phase` と同じ: cos 項なし、
/// 原点ずらしなし（`p` から直接撃つ）、裏面棄却なし、`f = hg_eval(wo·wi, g)`。MIS なし・乱数なし。
fn nee_delta_light_phase(
    occluded: impl Fn(Ray, f64) -> bool,
    light: &DeltaLight,
    med: &Medium,
    path_throughput: Color,
    p: Vec3,
    wo: Vec3,
    time: f64,
) -> Color {
    let Some(dh) = light.sample_at(p) else { return Color::new(0.0, 0.0, 0.0) };
    let tmax = match dh.position {
        Some(pos) => (pos - p).len(),
        None => RAY_T_MAX,
    };
    let shadow = Ray { o: p, d: dh.wi, time };
    if occluded(shadow, tmax) {
        return Color::new(0.0, 0.0, 0.0);
    }
    let f = hg_eval(wo.dot(dh.wi), med.g);
    let c = path_throughput.hadamard(dh.value) * f;
    c.hadamard(med.transmittance(shadow, 0.0, tmax)).clamp_luminance(FIREFLY_CLAMP)
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
        crate::geometry::Hit { t: 1.0, p, ng: n, ns: n, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0), uv: (0.0, 0.0) }
    }

    fn floor_setup() -> (Material, Vec3, Vec3, Ray) {
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None };
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
                &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, None, &mut rng,
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
                &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, None, &mut rng,
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
            let c = nee_environment(|_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, None, &mut rng);
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
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &ls, None);
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
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), &test_hit(p, n), n, n, ray, &ls, None);
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
            Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None },
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
            let x = radiance(world, mats, &Surfaces::none(), Some(env), None, ray, &mut rng, limits, None).r();
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
        assert_eq!(radiance(&world, &mats, &Surfaces::none(), Some(&env), None, to_light, &mut rng, limits(0), None).r(), 0.0);
        assert_eq!(radiance(&world, &mats, &Surfaces::none(), Some(&env), None, to_light, &mut rng, limits(1), None).r(), 4.0);
        assert_eq!(radiance(&world, &mats, &Surfaces::none(), Some(&env), None, to_floor, &mut rng, limits(1), None).r(), 0.0);

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

    /// PERF-3: 最初のバウンスの NEE を層化（`Strata` 付き）しても、直接照明の理論値
    /// （ρ·L/9）に不偏で収束する。256 層を順に一巡させながら呼ぶ（`render.rs` の
    /// `sample_pixel` が層番号を割り当てる使い方の最小再現）。
    #[test]
    fn stratified_nee_is_unbiased_for_the_analytic_direct_lighting_value() {
        let (world, mats, env) = floor_under_sphere_light();
        let to_floor = Ray { o: Vec3::new(2.0, 1.0, 0.0), d: Vec3::new(-2.0, -1.0, 0.0).norm(), time: 0.0 };
        let exact = 0.5 * 4.0 / 9.0;
        let limits = PathLimits { max_depth: usize::MAX, rr_depth: 1000 };
        let (nx, ny) = (16usize, 16usize);
        let n_cells = nx * ny;

        let mut rng = Rng::new(42);
        let n = 200_000usize;
        let (mut s, mut s2) = (0.0, 0.0);
        for i in 0..n {
            let strata = Strata { stratum: i % n_cells, nx, ny };
            let x = radiance(&world, &mats, &Surfaces::none(), Some(&env), None, to_floor, &mut rng, limits, Some(strata)).r();
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        let se = ((s2 / n as f64 - mean * mean).max(0.0) / n as f64).sqrt();
        assert!((mean - exact).abs() < 5.0 * se + 1e-3 * exact, "{} ± {} vs exact {}", mean, se, exact);
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
        let hit = crate::geometry::Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng: Vec3::new(0.0, 1.0, 0.0), ns: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0), uv: (0.0, 0.0) };
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
            (Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None }, 0.8),
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
            uv: (0.0, 0.0),
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
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None };
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
            |shadow: Ray, _| { seen.set(Some(shadow.o)); false },
            &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls, None,
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
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None };
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let mut rng = Rng::new(4);
        let seen: Cell<Option<(Vec3, Vec3)>> = Cell::new(None);
        let mut checked = 0;
        for _ in 0..200 {
            let c = nee_environment(
                |shadow: Ray| { seen.set(Some((shadow.o, shadow.d))); false },
                &env, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, None, &mut rng,
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
        let mat = Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None };
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
        let c = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls, None);
        assert!(is_black(c), "幾何的に裏側の光源から寄与が漏れている: {:?}", (c.r(), c.g(), c.b()));

        // 環境 NEE も同じ: ng の裏半球だけが光る環境にすると寄与は 0 になる
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let mut rng = Rng::new(31);
        let mut leaked = 0;
        for _ in 0..2000 {
            let c = nee_environment(
                |_| false, &env, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, None, &mut rng,
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
        let c2 = nee_area_light(|_, _| false, &mat, Color::new(1.0, 1.0, 1.0), &hit, ns, ng, ray, &ls_back, None);
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
            Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None },
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
            Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None },
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

    /// **テクスチャが描画経路で効く**: 同じシーンでテクスチャの値だけを変えると明るさが変わり、
    /// その比はテクスチャの反射率の比に一致する。
    ///
    /// `radiance` が `resolve_textures` を呼ばない（= テクスチャを無視する）ようにすると、
    /// どちらも定数の倍率（白）のままになって比が 1 になり落ちる。
    #[test]
    fn radiance_uses_the_bound_texture() {
        use crate::geometry::Sphere;
        use crate::texture::{Texture, Wrap};
        use crate::transform::Transform;
        use crate::world::test_meshes::tilted_quad;

        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
        let light_dir = Vec3::new(0.0, 0.0, 1.0);
        let d = Vec3::new(0.0, 0.6, -1.0).norm();
        let ray = Ray { o: -d * 4.0, d, time: 0.0 };

        // 反射率だけが違う 2 つの一様テクスチャ
        let mean_for = |reflectance: f64| {
            let mats = vec![
                Material::Lambert { albedo: Color::new(1.0, 1.0, 1.0), albedo_tex: Some(0) },
                Material::DiffuseLight { emit: Color::new(60.0, 60.0, 60.0) },
            ];
            let v = (reflectance * 255.0).round() as u8;
            let textures = vec![Texture::from_texels_u8(1, 1, vec![v, v, v], false, Wrap::Repeat)];
            let mut world = World::new();
            world.add_mesh_data_instance(
                tilted_quad(8.0, Vec3::new(0.0, 0.0, 1.0), 0), Transform::identity(), None);
            world.add_sphere(Sphere { c: light_dir * 6.0, r: 0.3, mat_id: 1 });
            world.build_lights(&mats);
            let mut rng = Rng::new(3);
            let (mut sum, n) = (0.0, 20_000);
            for _ in 0..n {
                sum += radiance(&world, &mats, &Surfaces::textures_only(&textures), Some(&env), None, ray, &mut rng,
                                PathLimits { max_depth: 2, rr_depth: 8 }, None).r();
            }
            sum / n as f64
        };

        let bright = mean_for(0.8);
        let dark = mean_for(0.2);
        assert!(bright > 0.0, "光が届いていない");
        // 直接照明だけ（max_depth = 2）なので、明るさは反射率に比例する
        let ratio = dark / bright;
        assert!((ratio - 0.25).abs() < 0.02, "テクスチャの反射率が効いていない（比 = {:.4}、期待 0.25）", ratio);
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
        let mats = vec![Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None }];
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
        let mats = vec![Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8), albedo_tex: None }];
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
                Material::Lambert { albedo: Color::new(albedo, albedo, albedo), albedo_tex: None },
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

    // ---- 参加媒質（M2）----

    use crate::geometry::Aabb;

    /// 媒質付きの推定。全チャンネルの (平均, 標準誤差)。
    fn estimate_medium(world: &World, mats: &[Material], env: &EnvMap, med: &Medium, ray: Ray, limits: PathLimits, n: usize, seed: u64) -> ([f64; 3], [f64; 3]) {
        let mut rng = Rng::new(seed);
        let (mut s, mut s2) = ([0.0; 3], [0.0; 3]);
        for _ in 0..n {
            let c = radiance(world, mats, &Surfaces::none(), Some(env), Some(med), ray, &mut rng, limits, None);
            for (i, x) in [c.r(), c.g(), c.b()].into_iter().enumerate() {
                s[i] += x;
                s2[i] += x * x;
            }
        }
        let nf = n as f64;
        let mean = [s[0] / nf, s[1] / nf, s[2] / nf];
        let se = [0, 1, 2].map(|i| ((s2[i] / nf - mean[i] * mean[i]).max(0.0) / nf).sqrt());
        (mean, se)
    }

    /// 原点中心・半辺長 `k` の箱に閉じた媒質と、それを +x 方向に貫くカメラレイ（箱の手前 `4k` から）。
    fn box_medium(k: f64, sigma_t: Color, albedo: Color, g: f64) -> (Medium, Ray) {
        let bounds = Aabb { min: Vec3::new(-k, -k, -k), max: Vec3::new(k, k, k) };
        (
            Medium { sigma_t, albedo, g, bounds: Some(bounds) },
            Ray { o: Vec3::new(-5.0 * k, 0.0, 0.0), d: Vec3::new(1.0, 0.0, 0.0), time: 0.0 },
        )
    }

    /// B + F: 白炉。幾何なし・一様環境光 1・吸収なしの媒質は、どの `g`・`rr_depth`・色付き `σt`・スケールでも 1。
    #[test]
    fn medium_white_furnace_is_one() {
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let world = World::new();
        let white = Color::new(1.0, 1.0, 1.0);
        let cases = [
            (1.0, Color::new(1.0, 1.0, 1.0), 0.0, 3),
            (1.0, Color::new(1.0, 1.0, 1.0), 0.7, 1),
            (1.0, Color::new(0.1, 0.5, 2.0), 0.0, 1),
            (1.0, Color::new(0.1, 0.5, 2.0), 0.7, 3),
            (1e-3, Color::new(100.0, 500.0, 2000.0), 0.7, 3),
            (1e3, Color::new(1e-4, 5e-4, 2e-3), 0.0, 1),
        ];
        for (k, sigma_t, g, rr_depth) in cases {
            let (med, ray) = box_medium(k, sigma_t, white, g);
            let limits = PathLimits { max_depth: usize::MAX, rr_depth };
            let (mean, se) = estimate_medium(&world, &[], &env, &med, ray, limits, 200_000, 5);
            for i in 0..3 {
                assert!((mean[i] - 1.0).abs() < 5.0 * se[i] + 1e-12, "k={k} σt={sigma_t:?} g={g} rr={rr_depth} ch{i}: {} ± {}", mean[i], se[i]);
            }
        }
    }

    /// C + F: 吸収のみ（albedo 0）の媒質は Beer-Lambert に一致する。
    #[test]
    fn absorbing_medium_matches_beer_lambert() {
        let env = EnvMap::constant(Color::new(1.0, 1.0, 1.0));
        let world = World::new();
        for k in [1.0, 1e-3, 1e3] {
            let sigma_t = Color::new(0.1, 0.5, 2.0) * (1.0 / k);
            let (med, ray) = box_medium(k, sigma_t, Color::new(0.0, 0.0, 0.0), 0.0);
            let limits = PathLimits { max_depth: usize::MAX, rr_depth: 3 };
            let (mean, se) = estimate_medium(&world, &[], &env, &med, ray, limits, 400_000, 9);
            let l = 2.0 * k;
            let want = [(-0.1 / k * l as f64).exp(), (-0.5 / k * l).exp(), (-2.0 / k * l).exp()];
            for i in 0..3 {
                assert!((mean[i] - want[i]).abs() < 4.0 * se[i] + 1e-12, "k={k} ch{i}: {} ± {} vs {}", mean[i], se[i], want[i]);
            }
        }
    }

    /// D: 吸収のみの媒質（全空間）を挟んだ直接照明。床の原点から見える球光源（中心高さ 3、半径 1、L=4）への
    /// 立体角積分を、光源への距離込みで数値積分した値 × カメラレイ側の減衰と一致する。表面 NEE と
    /// BSDF サンプリング側（Pass の weight）の両方が入っているので、片方でも透過率が抜けると外れる。
    #[test]
    fn surface_nee_attenuates_by_the_medium() {
        let (world, mats, env) = floor_under_sphere_light();
        let to_floor = Ray { o: Vec3::new(2.0, 1.0, 0.0), d: Vec3::new(-2.0, -1.0, 0.0).norm(), time: 0.0 };
        let cam_len = 5.0f64.sqrt();
        let limits = PathLimits { max_depth: usize::MAX, rr_depth: 1000 };
        let sigma = 0.3;
        // E = ∫ cosθ e^{-σ d(θ)} 2π sinθ dθ（軸まわり対称）。d = 3cosθ - sqrt(1 - 9 sin²θ)
        let theta_max = (1.0f64 / 3.0).asin();
        let m = 200_000;
        let mut integral = 0.0;
        for i in 0..m {
            let th = (i as f64 + 0.5) / m as f64 * theta_max;
            let d = 3.0 * th.cos() - (1.0 - 9.0 * th.sin().powi(2)).max(0.0).sqrt();
            integral += th.cos() * (-sigma * d).exp() * 2.0 * std::f64::consts::PI * th.sin() * theta_max / m as f64;
        }
        let exact = 0.5 / std::f64::consts::PI * 4.0 * integral * (-sigma * cam_len).exp();
        let med = Medium { sigma_t: Color::new(sigma, sigma, sigma), albedo: Color::new(0.0, 0.0, 0.0), g: 0.0, bounds: None };
        let (mean, se) = estimate_medium(&world, &mats, &env, &med, to_floor, limits, 400_000, 3);
        assert!((mean[0] - exact).abs() < 5.0 * se[0] + 1e-3 * exact, "{} ± {} vs {}", mean[0], se[0], exact);
        // 媒質なしの解析値 ρL/9 より確かに暗い（テストが自明に通っていないことの確認）
        assert!(exact < 0.5 * 4.0 / 9.0 * 0.6);
    }

    /// 媒質散乱点の NEE（面光源）と位相関数サンプリング側の MIS: `max_depth = 2` は単一散乱だけを数える。
    /// 球光源（中心 (0,0,2)、半径 0.3、L=4）のそばを通る +x 向きのカメラレイに対し、単一散乱の
    /// 積分 ∫ σ e^{-σ(t-a)} · L·Ω(t)/(4π) · e^{-σD(t)} dt（Ω は球の立体角）と一致する。
    #[test]
    fn single_scattering_from_a_sphere_light_matches_analytic() {
        use crate::geometry::Sphere;
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.0, 4.0, 4.0) }];
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 2.0), r: 0.3, mat_id: 0 });
        world.build_lights(&mats);
        let env = EnvMap::constant(Color::new(0.0, 0.0, 0.0));
        let sigma = 0.05;
        let (med, ray) = box_medium(3.0, Color::new(sigma, sigma, sigma), Color::new(1.0, 1.0, 1.0), 0.0);
        let limits = PathLimits { max_depth: 2, rr_depth: 1000 };
        // ボックス内は t ∈ [2, 8]（x = -3..3）
        let m = 100_000;
        let mut exact = 0.0;
        for i in 0..m {
            let t = 2.0 + 6.0 * (i as f64 + 0.5) / m as f64;
            let x = -5.0 + t;
            let dd = (x * x + 4.0f64).sqrt();
            let omega = 2.0 * std::f64::consts::PI * (1.0 - (1.0 - 0.09 / (dd * dd)).sqrt());
            exact += sigma * (-sigma * (t - 2.0)).exp() * 4.0 * omega / (4.0 * std::f64::consts::PI) * (-sigma * dd).exp() * 6.0 / m as f64;
        }
        let (mean, se) = estimate_medium(&world, &mats, &env, &med, ray, limits, 2_000_000, 17);
        // 光源側の減衰を中心距離で近似しているぶん（σ·r ≈ 1.5%）を許容する
        assert!((mean[0] - exact).abs() < 5.0 * se[0] + 0.02 * exact, "{} ± {} vs {}", mean[0], se[0], exact);
    }
    // ---- デルタ光源（点・平行・スポット）----

    use crate::world::DeltaLight;

    /// y = 0 の大きな平面（材質 0）だけのワールド。背景は黒（環境光 NEE は寄与 0）。
    fn plane_world(lights: &[DeltaLight]) -> World {
        plane_world_sized(lights, 200.0)
    }

    /// 半辺 `e` の平面。スケール不変のテストでは平面の大きさも一緒に変える（座標の大きさが誤差上界に効くため）。
    fn plane_world_sized(lights: &[DeltaLight], e: f64) -> World {
        use crate::geometry::Triangle;
        use crate::transform::Transform;
        let mut world = World::new();
        let v = |x: f64, z: f64| Vec3::new(x, 0.0, z);
        world.add_mesh_instance(
            vec![
                Triangle::new_static(v(-e, -e), v(e, e), v(e, -e), 0),
                Triangle::new_static(v(-e, -e), v(-e, e), v(e, e), 0),
            ],
            Transform::identity(),
            None,
        );
        for l in lights {
            world.add_delta_light(*l);
        }
        world
    }

    const RHO: f64 = 0.6;
    fn lambert() -> Vec<Material> {
        vec![Material::Lambert { albedo: Color::new(RHO, RHO, RHO), albedo_tex: None }]
    }
    fn black_env() -> EnvMap {
        EnvMap::constant(Color::new(0.0, 0.0, 0.0))
    }

    /// 平面上の点 `target` を斜め上から見たカメラレイの放射輝度（赤）。デルタ光源の NEE は乱数を引かないので決定的。
    fn plane_radiance(world: &World, mats: &[Material], medium: Option<&Medium>, target: Vec3, from: Vec3) -> f64 {
        let ray = Ray { o: from, d: (target - from).norm(), time: 0.0 };
        let mut rng = Rng::new(1);
        let env = black_env();
        radiance(world, mats, &Surfaces::none(), Some(&env), medium, ray, &mut rng, PathLimits { max_depth: 3, rr_depth: 1000 }, None).r()
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    /// 点光源: 直下 ρ/π · I/h²、斜めの点 ρ/π · I/d² · cosθ に厳密一致。
    #[test]
    fn point_light_matches_analytic() {
        let (h, i) = (2.0, 5.0);
        let light = Vec3::new(0.0, h, 0.0);
        let world = plane_world(&[DeltaLight::Point { position: light, intensity: Color::new(i, i, i) }]);
        let mats = lambert();
        let from = |t: Vec3| t + Vec3::new(-0.3, 1.0, -0.1);
        let t0 = Vec3::new(0.0, 0.0, 0.0);
        let got = plane_radiance(&world, &mats, None, t0, from(t0));
        assert!(close(got, RHO / std::f64::consts::PI * i / (h * h), 1e-12), "{got}");
        for t in [Vec3::new(2.0, 0.0, 1.0), Vec3::new(-3.0, 0.0, 0.5), Vec3::new(10.0, 0.0, -7.0)] {
            let d2 = (light - t).dot(light - t);
            let cos = h / d2.sqrt();
            let want = RHO / std::f64::consts::PI * i / d2 * cos;
            let got = plane_radiance(&world, &mats, None, t, from(t));
            assert!(close(got, want, 1e-12), "{:?}: {} vs {}", t, got, want);
        }
        // 逆二乗則: 高さ 2 倍で 1/4
        let world2 = plane_world(&[DeltaLight::Point { position: Vec3::new(0.0, 2.0 * h, 0.0), intensity: Color::new(i, i, i) }]);
        let got2 = plane_radiance(&world2, &mats, None, t0, from(t0));
        assert!(close(got2 / got, 0.25, 1e-12), "{}", got2 / got);
    }

    /// 平行光源: ρ/π · E · cosθ（距離に依らない）。
    #[test]
    fn directional_light_matches_analytic() {
        let e = 3.0;
        for deg in [0.0f64, 20.0, 45.0, 70.0] {
            let th = deg.to_radians();
            let dir = Vec3::new(-th.sin(), -th.cos(), 0.0); // 光の進む向き
            let world = plane_world(&[DeltaLight::Directional { direction: dir * 5.0, irradiance: Color::new(e, e, e) }]);
            let mats = lambert();
            for t in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 30.0)] {
                let got = plane_radiance(&world, &mats, None, t, t + Vec3::new(-0.3, 1.0, -0.1));
                let want = RHO / std::f64::consts::PI * e * th.cos();
                assert!(close(got, want, 1e-12), "θ={} {:?}: {} vs {}", deg, t, got, want);
            }
        }
    }

    /// スポット: 軸上は点光源と一致、cutoff の外はちょうど 0、beam〜cutoff の間で単調に落ちて境界に飛びが無い。
    #[test]
    fn spot_light_falloff() {
        let (h, i) = (2.0, 5.0);
        let (cut, beam) = (30f64.to_radians(), 20f64.to_radians());
        let pos = Vec3::new(0.0, h, 0.0);
        let spot = DeltaLight::Spot { position: pos, direction: Vec3::new(0.0, -1.0, 0.0), intensity: Color::new(i, i, i), cutoff_angle: cut, beam_width: beam };
        let point = DeltaLight::Point { position: pos, intensity: Color::new(i, i, i) };
        let mats = lambert();
        let (ws, wp) = (plane_world(&[spot]), plane_world(&[point]));
        let t0 = Vec3::new(0.0, 0.0, 0.0);
        let a = plane_radiance(&ws, &mats, None, t0, t0 + Vec3::new(-0.3, 1.0, -0.1));
        let b = plane_radiance(&wp, &mats, None, t0, t0 + Vec3::new(-0.3, 1.0, -0.1));
        assert_eq!(a.to_bits(), b.to_bits(), "軸上は点光源と同じ");
        // 角度（軸からの角）を掃引して係数 = 値 / 点光源の値
        let value_at = |deg: f64| {
            let p = Vec3::new(h * deg.to_radians().tan(), 0.0, 0.0);
            spot.sample_at(p).map_or(0.0, |x| x.value.r()) / point.sample_at(p).unwrap().value.r()
        };
        assert_eq!(value_at(35.0), 0.0);
        assert_eq!(value_at(30.0), 0.0, "cutoff ちょうどは 0");
        assert!(value_at(29.999) < 1e-3, "cutoff の直前はほぼ 0（飛びなし）: {}", value_at(29.999));
        assert!(value_at(20.001) > 1.0 - 1e-3 && value_at(20.001) <= 1.0, "beam の直後はほぼ 1（飛びなし）");
        assert_eq!(value_at(10.0), 1.0);
        let mut prev = 1.0;
        for k in 0..=100 {
            let deg = 20.0 + 10.0 * k as f64 / 100.0;
            let f = value_at(deg);
            assert!(f <= prev + 1e-15, "単調に落ちる: {deg}° {f} > {prev}");
            assert!((prev - f).abs() < 0.05, "刻みごとの飛びが小さい");
            prev = f;
        }
        // 中間の係数は角度に対して線形（Mitsuba 3 の spot: (cutoff − θ)/(cutoff − beam)）
        for deg in [21.0f64, 25.0, 29.0] {
            let want = (cut - deg.to_radians()) / (cut - beam);
            assert!(close(value_at(deg), want, 1e-12), "{deg}°: {} vs {want}", value_at(deg));
        }
    }

    /// 光源と点の間に板を挟むと寄与がちょうど 0。板が無ければ非 0。
    #[test]
    fn occluder_blocks_delta_lights() {
        use crate::geometry::Triangle;
        use crate::transform::Transform;
        let mats = lambert();
        let t0 = Vec3::new(0.0, 0.0, 0.0);
        let from = Vec3::new(20.0, 0.2, 0.0); // 板の下をくぐる低い角度
        let lights = [
            DeltaLight::Point { position: Vec3::new(0.0, 2.0, 0.0), intensity: Color::new(5.0, 5.0, 5.0) },
            DeltaLight::Directional { direction: Vec3::new(0.0, -1.0, 0.0), irradiance: Color::new(3.0, 3.0, 3.0) },
        ];
        for l in lights {
            let mut world = plane_world(&[l]);
            assert!(plane_radiance(&world, &mats, None, t0, from) > 0.1);
            let v = |x: f64, z: f64| Vec3::new(x, 1.0, z);
            world.add_mesh_instance(
                vec![Triangle::new_static(v(-3.0, -3.0), v(3.0, -3.0), v(3.0, 3.0), 0), Triangle::new_static(v(-3.0, -3.0), v(3.0, 3.0), v(-3.0, 3.0), 0)],
                Transform::identity(),
                None,
            );
            assert_eq!(plane_radiance(&world, &mats, None, t0, from), 0.0, "{:?}", l);
        }
    }

    /// スケール則。長さを k 倍すると点光源の d² が k² 倍になるので、同じ絵にするには強度 I を k² 倍にする
    /// （放射照度 = I/d²）。平行光源は距離減衰が無いので放射照度 E はそのまま。
    #[test]
    fn delta_lights_scale_invariance() {
        let mats = lambert();
        let (h, i, e) = (2.0, 5.0, 3.0);
        let base_t = Vec3::new(2.0, 0.0, 1.0);
        let mk = |k: f64| {
            plane_world_sized(&[
                DeltaLight::Point { position: Vec3::new(0.0, h * k, 0.0), intensity: Color::new(i * k * k, i * k * k, i * k * k) },
                DeltaLight::Directional { direction: Vec3::new(-0.3, -1.0, 0.2), irradiance: Color::new(e, e, e) },
            ], 1e3 * k)
        };
        let want = plane_radiance(&mk(1.0), &mats, None, base_t, base_t + Vec3::new(-0.3, 1.0, -0.1));
        for k in [1e-3, 1e3] {
            let t = base_t * k;
            let got = plane_radiance(&mk(k), &mats, None, t, t + Vec3::new(-0.3, 1.0, -0.1) * k);
            assert!(close(got, want, 1e-9), "k={k}: {got} vs {want}");
        }
    }

    /// 回帰: デルタ光源に MIS を掛けてはいけない。BSDF pdf が大きい粗い GGX でも、値は
    /// `f · I/d² · cos`（`mat.eval` の f）にちょうど一致する。MIS 重み（<1）を掛けると暗くなり落ちる。
    #[test]
    fn delta_light_is_not_mis_weighted_on_rough_ggx() {
        let mats = vec![Material::Ggx { albedo: Color::new(0.8, 0.8, 0.8), alpha: 0.9 }];
        let (h, i) = (2.0, 5.0);
        let light = Vec3::new(0.5, h, -0.3);
        let world = plane_world(&[DeltaLight::Point { position: light, intensity: Color::new(i, i, i) }]);
        let t = Vec3::new(0.0, 0.0, 0.0);
        let from = t + Vec3::new(-0.4, 1.0, -0.2);
        let ray = Ray { o: from, d: (t - from).norm(), time: 0.0 };
        let wi = (light - t).norm();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let (f, pdf) = mats[0].eval((-ray.d).norm(), wi, n);
        assert!(pdf > 0.05, "MIS 重みが目立つほど pdf が大きい材質のはず: {pdf}");
        let want = f.r() * i / (light - t).dot(light - t) * n.dot(wi);
        let got = plane_radiance(&world, &mats, None, t, from);
        assert!(close(got, want, 1e-12), "{got} vs {want}");
    }

    /// 参加媒質との併用: 霧（吸収のみ）の中の点光源は exp(-σt·d) ぶん減衰する。
    /// 媒質の箱は評価点の真上（y ∈ [0.1, 2.5]）だけで、カメラレイは箱の下を通る（カメラ側は減衰しない）。
    #[test]
    fn delta_light_is_attenuated_by_the_medium() {
        use crate::geometry::Aabb;
        let (h, i, sigma) = (2.0, 5.0, 0.7);
        let world = plane_world(&[DeltaLight::Point { position: Vec3::new(0.0, h, 0.0), intensity: Color::new(i, i, i) }]);
        let mats = lambert();
        let med = Medium {
            sigma_t: Color::new(sigma, sigma, sigma),
            albedo: Color::new(0.0, 0.0, 0.0),
            g: 0.0,
            bounds: Some(Aabb { min: Vec3::new(-0.5, 0.1, -0.5), max: Vec3::new(0.5, 2.5, 0.5) }),
        };
        let t = Vec3::new(0.0, 0.0, 0.0);
        let from = Vec3::new(20.0, 0.01, 0.0);
        let plain = plane_radiance(&world, &mats, None, t, from);
        let fog = plane_radiance(&world, &mats, Some(&med), t, from);
        // 影の光路は y = 0（の少し上）から光源 y = h まで。箱の中は y ∈ [0.1, h]
        let want = plain * (-sigma * (h - 0.1)).exp();
        assert!(close(fog, want, 1e-9), "{fog} vs {want}");
    }

    /// 媒質散乱点のデルタ NEE: `throughput · I/d² · hg · exp(-σt d)`（cos なし・MIS なし）、遮蔽で 0。
    #[test]
    fn delta_light_phase_nee_matches_analytic() {
        let (i, sigma, g) = (5.0, 0.4, 0.5);
        let pos = Vec3::new(0.0, 3.0, 0.0);
        let light = DeltaLight::Point { position: pos, intensity: Color::new(i, i, i) };
        let med = Medium { sigma_t: Color::new(sigma, sigma, sigma), albedo: Color::new(1.0, 1.0, 1.0), g, bounds: None };
        let p = Vec3::new(1.0, 0.0, 0.5);
        let wo = Vec3::new(0.2, -0.7, 0.4).norm();
        let tp = Color::new(0.8, 0.8, 0.8);
        let wi = (pos - p).norm();
        let d = (pos - p).len();
        let want = 0.8 * i / (d * d) * hg_eval(wo.dot(wi), g) * (-sigma * d).exp();
        let got = nee_delta_light_phase(|_, _| false, &light, &med, tp, p, wo, 0.0).r();
        assert!(close(got, want, 1e-12), "{got} vs {want}");
        assert_eq!(nee_delta_light_phase(|_, _| true, &light, &med, tp, p, wo, 0.0).r(), 0.0);
        // 平行光源: 距離減衰なし（無限遠までの透過率 = 0 になる無限媒質では 0）
        let sun = DeltaLight::Directional { direction: Vec3::new(0.0, -1.0, 0.0), irradiance: Color::new(2.0, 2.0, 2.0) };
        assert_eq!(nee_delta_light_phase(|_, _| false, &sun, &med, tp, p, wo, 0.0).r(), 0.0);
    }

}

/// 法線マップ／バンプマップの配線テスト（Mitsuba のラッパー構文で作ったシーンを積分器に通す）。
#[cfg(test)]
mod map_tests {
    use super::*;
    use crate::config::RenderConfig;
    use crate::mitsuba::load_scene_from_str;
    use crate::scene::Scene;
    use std::path::PathBuf;

    fn tmpdir() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!("tinypt_maps_{}_{}", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32, px: &[[u8; 3]]) {
        let mut img = image::RgbImage::new(w, h);
        for (i, p) in px.iter().enumerate() {
            img.put_pixel((i as u32) % w, (i as u32) / w, image::Rgb(*p));
        }
        img.save(dir.join(name)).unwrap();
    }

    fn scene(xml: &str, dir: &std::path::Path) -> Scene {
        load_scene_from_str(xml, dir, &RenderConfig::default(), (None, None)).unwrap().0
    }

    fn surfaces(s: &Scene) -> Surfaces<'_> {
        Surfaces { textures: &s.textures, normal_maps: &s.normal_maps, mat_maps: &s.mat_maps }
    }

    /// `tf`（`<transform>` の中身）を付けた 1 枚の板（法線 +z、UV = (x+1)/2, (y+1)/2）に、`bsdf` を貼る。
    fn plate_xml(tf: &str, bsdf: &str, extra: &str) -> String {
        format!(
            r#"<scene version="3.0.0"><shape type="rectangle"><transform name="to_world">{}</transform>{}</shape>{}</scene>"#,
            tf, bsdf, extra
        )
    }

    fn normalmap_bsdf(file: &str) -> String {
        format!(
            r#"<bsdf type="normalmap"><texture type="bitmap" name="normalmap"><string name="filename" value="{}"/></texture><bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf></bsdf>"#,
            file
        )
    }

    fn down_ray(x: f64, y: f64) -> Ray {
        Ray { o: Vec3::new(x, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }
    }

    /// 摂動は `ns` だけを変える: `ng` / `p` / `p_error` はビット単位で不変で、`ns · ng > 0`、`ns` は単位。
    ///
    /// ミューテーション検出: 摂動結果を `hit.ng` に書くと（`ng` が変わるので）落ちる。
    #[test]
    fn perturbation_changes_only_ns() {
        let dir = tmpdir();
        write_png(&dir, "n.png", 1, 1, &[[230, 128, 190]]);
        let s = scene(&plate_xml("", &normalmap_bsdf("n.png"), ""), &dir);
        let sf = surfaces(&s);
        let orig = s.world.hit(down_ray(0.3, 0.2), 0.0, 1e30).unwrap();
        let mut h = orig;
        let id = sf.map_for(h.mat_id).expect("材質にマップが付いていない");
        perturb_shading_normal(&s.world, &sf, id, &mut h, 0.0);
        let bits = |v: Vec3| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
        assert_eq!(bits(h.ng), bits(orig.ng), "ng が変わった");
        assert_eq!(bits(h.p), bits(orig.p), "p が変わった");
        assert_eq!(h.p_error.x.to_bits(), orig.p_error.x.to_bits());
        assert!((h.ns - orig.ns).len() > 0.1, "ns が摂動されていない");
        assert!(h.ns.dot(h.ng) > 0.0 && (h.ns.len() - 1.0).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 強い（z 成分が負の）マップと鏡像インスタンスでも、全ヒットで摂動が効き（`ns` が変わる）、`ns · ng > 0`。
    ///
    /// ミューテーション検出: `face_forward(n_pert, hit.ng)` を外すと、`n_pert · ng < 0` になった摂動が
    /// 地平線判定で捨てられて `ns` が変わらない（`ns` が元のまま）ので落ちる。判定まで外すと `ns · ng < 0` で落ちる。
    #[test]
    fn strong_map_on_mirrored_instance_keeps_ns_on_the_geometric_side() {
        let dir = tmpdir();
        write_png(&dir, "n.png", 1, 1, &[[255, 128, 60]]);
        for tf in [r#"<scale x="-5" y="5" z="1"/>"#, r#"<scale x="5" y="-5" z="1"/>"#, r#"<scale x="5" y="5" z="-1"/>"#] {
            let s = scene(&plate_xml(tf, &normalmap_bsdf("n.png"), ""), &dir);
            let sf = surfaces(&s);
            for &(x, y) in &[(0.3, 0.2), (-1.0, 0.7), (2.0, -1.5)] {
                for &zs in &[3.0, -3.0] {
                    let r = Ray { o: Vec3::new(x, y, zs), d: Vec3::new(0.0, 0.0, -zs.signum()), time: 0.0 };
                    let orig = s.world.hit(r, 0.0, 1e30).unwrap();
                    let mut h = orig;
                    perturb_shading_normal(&s.world, &sf, sf.map_for(h.mat_id).unwrap(), &mut h, 0.0);
                    assert!(h.ns.dot(h.ng) > 0.0, "{}: ns·ng = {}", tf, h.ns.dot(h.ng));
                    assert!((h.ns - orig.ns).len() > 0.1, "{}: 摂動が捨てられた", tf);
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 一様な環境光 L=1 の下で、摂動した法線での「幾何法線の地平線でクリップした」放射照度と一致する。
    /// NEE と BSDF サンプリングの**両方**が同じ摂動後の `ns` を見ている証拠: 片方だけを摂動すると、
    /// 2 つの戦略が別の積分を推定するので MIS の混合が参照値から数 % ずれる。
    ///
    /// ミューテーション検出: 摂動を NEE 側にだけ／BSDF 側にだけ適用すると落ちる。
    #[test]
    fn nee_and_bsdf_strategies_see_the_same_perturbed_normal() {
        let dir = tmpdir();
        let rgb = [200u8, 128, 200]; // 約 45° 傾く
        write_png(&dir, "n.png", 1, 1, &[rgb]);
        let env = r#"<emitter type="constant"><rgb name="radiance" value="1,1,1"/></emitter>"#;
        let s = scene(&plate_xml("", &normalmap_bsdf("n.png"), env), &dir);
        let sf = surfaces(&s);
        let limits = PathLimits { max_depth: 2, rr_depth: 8 };
        let mut rng = Rng::new(11);
        let n = 60_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += radiance(&s.world, &s.mats, &sf, s.env.as_ref(), None, down_ray(0.0, 0.0), &mut rng, limits, None).g();
        }
        let got = sum / n as f64;

        // 参照: L_o = ρ/π · ∫_{w·z>0} max(0, n'·w) dw（一様半球サンプリングの数値積分）
        let dec = |v: u8| 2.0 * v as f64 / 255.0 - 1.0;
        let np = Vec3::new(dec(rgb[0]), dec(rgb[1]), dec(rgb[2])).norm();
        let mut rr = Rng::new(5);
        let m = 2_000_000;
        let mut acc = 0.0;
        for _ in 0..m {
            let mut w = crate::rng::uniform_sphere_dir(&mut rr);
            if w.z < 0.0 {
                w = -w;
            }
            acc += np.dot(w).max(0.0);
        }
        let want = 0.5 / std::f64::consts::PI * (2.0 * std::f64::consts::PI * acc / m as f64);
        assert!((got - want).abs() / want < 0.02, "got {:.4}, want {:.4}", got, want);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// バンプが効く: 傾いたハイトマップは出力を有意に変え、`scale = 0` は**ビット単位で**マップ無しと同じ。
    #[test]
    fn bump_map_changes_output_and_zero_strength_is_bit_identical() {
        let dir = tmpdir();
        let ramp: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 36; 3]).collect();
        write_png(&dir, "h.png", 8, 1, &ramp);
        let env = r#"<emitter type="constant"><rgb name="radiance" value="1,1,1"/></emitter>"#;
        let bump = |scale: f64| {
            format!(
                r#"<bsdf type="bumpmap"><float name="scale" value="{}"/><texture type="bitmap" name="bumpmap"><string name="filename" value="h.png"/><string name="wrap_mode" value="clamp"/></texture><bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf></bsdf>"#,
                scale
            )
        };
        let plain = r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>"#;
        let limits = PathLimits { max_depth: 2, rr_depth: 8 };
        let run = |xml: &str| {
            let s = scene(xml, &dir);
            let sf = surfaces(&s);
            let mut rng = Rng::new(3);
            let mut sum = 0.0;
            for _ in 0..4000 {
                sum += radiance(&s.world, &s.mats, &sf, s.env.as_ref(), None, down_ray(0.0, 0.0), &mut rng, limits, None).g();
            }
            sum
        };
        let base = run(&plate_xml("", plain, env));
        let zero = run(&plate_xml("", &bump(0.0), env));
        let strong = run(&plate_xml("", &bump(5.0), env));
        assert_eq!(zero.to_bits(), base.to_bits(), "scale=0 がマップ無しとビット一致しない");
        assert!((strong - base).abs() / base > 0.03, "バンプが効いていない: {} vs {}", strong, base);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 球状の法線を焼いた 8x8 マップを板に貼ると、各テクセル中心で摂動後の `ns` が
    /// 焼いたバイト値から復号した解析解（t = +x, b = +y, n = +z の接空間）と 1e-6 以内で一致する。
    #[test]
    fn baked_sphere_normal_map_matches_the_analytic_solution() {
        let dir = tmpdir();
        let n = 8usize;
        let (mut px, mut want) = (vec![[0u8; 3]; n * n], vec![Vec3::new(0.0, 0.0, 0.0); n * n]);
        let enc = |c: f64| ((c * 0.5 + 0.5) * 255.0).round() as u8;
        for row in 0..n {
            for col in 0..n {
                let (u, v) = ((col as f64 + 0.5) / n as f64, 1.0 - (row as f64 + 0.5) / n as f64);
                let (a, b) = ((2.0 * u - 1.0) * 0.6, (2.0 * v - 1.0) * 0.6);
                let nz = (1.0f64 - a * a - b * b).sqrt();
                let bytes = [enc(a), enc(b), enc(nz)];
                px[row * n + col] = bytes;
                let dec = |x: u8| 2.0 * x as f64 / 255.0 - 1.0;
                want[row * n + col] = Vec3::new(dec(bytes[0]), dec(bytes[1]), dec(bytes[2])).norm();
            }
        }
        write_png(&dir, "n.png", n as u32, n as u32, &px);
        let s = scene(&plate_xml("", &normalmap_bsdf("n.png"), ""), &dir);
        let sf = surfaces(&s);
        for row in 0..n {
            for col in 0..n {
                let (u, v) = ((col as f64 + 0.5) / n as f64, 1.0 - (row as f64 + 0.5) / n as f64);
                let mut h = s.world.hit(down_ray(2.0 * u - 1.0, 2.0 * v - 1.0), 0.0, 1e30).unwrap();
                perturb_shading_normal(&s.world, &sf, sf.map_for(h.mat_id).unwrap(), &mut h, 0.0);
                let w = want[row * n + col];
                assert!((h.ns - w).len() < 1e-6, "({},{}): {:?} vs {:?}", row, col, (h.ns.x, h.ns.y, h.ns.z), (w.x, w.y, w.z));
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 法線マップを貼った閉じたメッシュ（立方体）の内側からは、外の光源が見えない（自己交差・光漏れなし）。
    ///
    /// ミューテーション検出: 摂動を `ng` に書くと、原点ずらしが摂動した向きになり内側のレイが外へ漏れうる。
    #[test]
    fn closed_mesh_with_a_normal_map_does_not_leak_light() {
        let dir = tmpdir();
        // 強く傾いた法線を市松に並べる
        write_png(&dir, "n.png", 2, 2, &[[240, 128, 150], [20, 200, 150], [128, 20, 150], [240, 240, 100]]);
        let xml = format!(
            r#"<scene version="3.0.0">
                 <shape type="cube">{}</shape>
                 <shape type="rectangle"><transform name="to_world"><translate x="0" y="0" z="3"/><rotate x="1" angle="180"/><scale x="2" y="2" z="1"/></transform>
                   <emitter type="area"><rgb name="radiance" value="30,30,30"/></emitter></shape>
               </scene>"#,
            normalmap_bsdf("n.png")
        );
        let s = scene(&xml, &dir);
        let sf = surfaces(&s);
        let mut rng = Rng::new(9);
        let limits = PathLimits { max_depth: 6, rr_depth: 8 };
        for _ in 0..20_000 {
            let d = crate::rng::uniform_sphere_dir(&mut rng);
            let c = radiance(&s.world, &s.mats, &sf, s.env.as_ref(), None, Ray { o: Vec3::new(0.1, -0.2, 0.05), d, time: 0.0 }, &mut rng, limits, None);
            assert_eq!(c.r(), 0.0, "立方体の内側に光が漏れた");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
