//! マテリアルモデルと BSDF サンプリング。
//!
//! 各 `Material` は BSDF として振る舞う:
//! - `sample()`  — 散乱レイ・スループット重み・PDF を返す（[`BsdfSample`]）
//! - `eval()`    — 方向ペアに対する BSDF 値 `f`（cosine 抜き）と PDF を返す
//! - `is_delta()` — デルタ（鏡面）散乱か。NEE 対象外の判定に使う
//! - `emitted()` — 発光体なら放射輝度を返す
//!
//! `sample()` が報告する PDF は、同じ方向ペアに対して `eval()` が返す PDF と一致する
//! （MIS の単一の真実）。法線の向き補正（entering 判定）は BSDF 内部に閉じている。
//!
//! ## 対応マテリアル
//! - **Lambert**: 完全拡散反射（コサイン重み付き半球サンプリング）
//! - **Metal**: 完全鏡面反射（デルタ BSDF）
//! - **Dielectric**: 屈折体（フレネル + Beer-Lambert 吸収、デルタ BSDF）
//! - **GGX**: マイクロファセットモデル（VNDF サンプリング + Smith 遮蔽関数）
//! - **Subsurface**: 簡易サブサーフェス（現状は Lambert と同一の拡散反射）
//! - **DiffuseLight**: 拡散発光体（散乱なし、放射輝度を返す）

use std::f64::consts::PI;

use crate::geometry::{face_forward, offset_ray_origin, Hit};
use crate::math::{reflect, refract, Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;

/// テクスチャ配列（[`crate::scene::Scene::textures`]）への添字。
/// `Material` を `Copy` のまま保つために、テクスチャ本体ではなく添字を持たせている
/// （マテリアルは交差ごとにコピーされるので、`Vec` を抱えさせたくない）。
pub type TexId = u32;

#[derive(Clone, Copy)]
/// 積分器が対応するマテリアルモデル。各 variant が一つの BSDF を表す。
pub enum Material {
    /// 完全拡散反射（Lambertian BRDF）。
    Lambert { albedo: Color },
    /// 完全鏡面反射（デルタ BRDF）
    Metal   { albedo: Color },
    /// 誘電体（屈折 + フレネル反射 + Beer-Lambert 吸収）
    Dielectric { ior: f64, absorption: Color },
    /// GGX マイクロファセット（粗さパラメータ alpha）
    Ggx     { albedo: Color, alpha: f64 },
    /// 簡易サブサーフェス（現状は Lambert と同一の拡散反射）
    Subsurface { albedo: Color },
    /// 拡散面光源
    DiffuseLight { emit: Color },
}

/// BSDF サンプリングの結果。
///
/// 散乱レイは原点込みで保持するため、透過（Dielectric）などの
/// 原点ずらしは BSDF 内部に閉じ、積分器はレイ構築の知識を持たない。
pub struct BsdfSample {
    /// 散乱レイ（原点ずらしを含む）
    pub scattered: Ray,
    /// スループット重み `f·cos/pdf`
    pub weight: Color,
    /// 立体角 PDF（`eval()` が返す PDF と同一。デルタ散乱では 0）
    pub pdf: f64,
    /// デルタ（鏡面）散乱か
    pub is_delta: bool,
    /// この散乱での相対屈折率 η_t/η_i（透過で媒質に入ると > 1）。反射・非透過の BSDF では 1。
    /// 透過の `weight` は放射輝度の η_i²/η_t² 倍を含むので、`throughput·eta²` はその倍率を
    /// 打ち消した量になる。積分器が Russian Roulette の生存確率に使う（Mitsuba 3 の `BSDFSample3f::eta`）。
    pub eta: f64,
}

/// 散乱方向 `d` が、レイの来た側を向いた幾何法線 `ng` の**表側**にあるか。
///
/// シェーディング法線（頂点法線の補間）は実際の面の向きと一致しないので、それを基準に
/// サンプルした方向が幾何的には面の裏へ潜ることがある。そのまま飛ばすと、原点ずらしは
/// 幾何法線基準なのでレイが自分のメッシュの内側に入り、閉じた物体では光が漏れたり
/// 黒い斑点が出たりする。tinypt はこの破綻したサンプルを**捨てる**（寄与 0）。
///
/// 捨てる＝そのぶんのエネルギーは失われる（増えることはない）。損失はシェーディング法線と
/// 幾何法線が大きく開くグレージング付近に限られ、分割の粗いメッシュの輪郭に薄い暗い縁として出る。
/// 代わりに「潜る方向を面に沿って倒す」補正も広く使われるが、pdf と weight の対応（`sample` の
/// weight == f·cos/pdf、pdf == eval の pdf）が崩れて MIS が壊れるので採らない。
#[inline]
fn reflects_above(ng: Vec3, d: Vec3) -> bool {
    d.dot(ng) > 0.0
}

impl Material {
    /// 反射率（アルベド）を持つ材質ならその色（`Lambert` / `Metal` / `Ggx` / `Subsurface`）。
    pub fn albedo(&self) -> Option<Color> {
        match *self {
            Material::Lambert { albedo } | Material::Metal { albedo } | Material::Ggx { albedo, .. } | Material::Subsurface { albedo } => Some(albedo),
            Material::Dielectric { .. } | Material::DiffuseLight { .. } => None,
        }
    }

    /// 反射率だけを `albedo` に置き換えた材質（反射率を持たない材質はそのまま）。シェーダーの式の評価結果を
    /// 具体的な `Material` にするのに使う（[`crate::shader::Shader`]）。
    pub fn with_albedo(self, albedo: Color) -> Self {
        match self {
            Material::Lambert { .. } => Material::Lambert { albedo },
            Material::Metal { .. } => Material::Metal { albedo },
            Material::Ggx { alpha, .. } => Material::Ggx { albedo, alpha },
            Material::Subsurface { .. } => Material::Subsurface { albedo },
            other => other,
        }
    }

    /// `Ggx` の粗さだけを置き換えた材質（他はそのまま）。範囲 `[1e-3, 1]` への保護は呼び出し側（シェーダー）が行う。
    pub fn with_alpha(self, alpha: f64) -> Self {
        match self {
            Material::Ggx { albedo, .. } => Material::Ggx { albedo, alpha },
            other => other,
        }
    }

    /// `Dielectric` の屈折率だけを置き換えた材質（他はそのまま）。
    pub fn with_ior(self, ior: f64) -> Self {
        match self {
            Material::Dielectric { absorption, .. } => Material::Dielectric { ior, absorption },
            other => other,
        }
    }

    /// `Dielectric` の吸収だけを置き換えた材質（他はそのまま）。
    pub fn with_absorption(self, absorption: Color) -> Self {
        match self {
            Material::Dielectric { ior, .. } => Material::Dielectric { ior, absorption },
            other => other,
        }
    }

    /// `DiffuseLight` の放射輝度だけを置き換えた材質（他はそのまま）。
    pub fn with_emit(self, emit: Color) -> Self {
        match self {
            Material::DiffuseLight { .. } => Material::DiffuseLight { emit },
            other => other,
        }
    }

    /// 発光体なら放射輝度を返す（`DiffuseLight` のみ `Some`）。
    pub fn emitted(&self) -> Option<Color> {
        match self {
            Material::DiffuseLight { emit } => Some(*emit),
            _ => None,
        }
    }

    /// デルタ（鏡面）散乱マテリアルか。`true` の場合 NEE の対象外。
    pub fn is_delta(&self) -> bool {
        matches!(self, Material::Metal { .. } | Material::Dielectric { .. })
    }

    /// 散乱レイ・スループット重み・PDF をサンプリングする。
    /// 発光体や無効サンプル（半球外など）の場合は `None`（パス終了）。
    pub fn sample(&self, ray_in: &Ray, hit: &Hit, rng: &mut Rng) -> Option<BsdfSample> {
        // 表裏の判定は**幾何法線**で行う（シェーディング法線は補間でどちらを向くか保証が無い）。
        let entering = hit.ng.dot(ray_in.d) < 0.0; // レイが表面に入射するか
        // レイの来た側を向いた幾何法線。散乱方向が幾何的に妥当かの判定に使う
        let ng = if entering { hit.ng } else { -hit.ng };
        // BSDF が使うのは**シェーディング法線**（頂点法線の補間。無ければ ng と同じ）。
        // 向きは ng と揃える（Hit 生成時に揃えてあるので、ここは entering の反転に追随するだけ）。
        let n = face_forward(hit.ns, ng);
        // レイの原点ずらしは常に幾何法線（hit.ng）で行う。シェーディング法線を使うと
        // 誤差の箱を抜けられず自己交差する。

        match self {
            Material::Lambert { albedo, .. } => {
                let d = sample_cosine_hemisphere(n, rng);
                // 補間法線のせいで幾何的な裏側へ飛ぶサンプルは捨てる（下の reflects_above を参照）
                if !reflects_above(ng, d) {
                    return None;
                }
                // f·cos/pdf = (albedo/π)·cos/(cos/π) = albedo
                Some(BsdfSample {
                    scattered: Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, d), d, time: ray_in.time },
                    weight: *albedo,
                    pdf: n.dot(d).max(0.0) / PI,
                    is_delta: false,
                    eta: 1.0,
                })
            }
            Material::Metal { albedo } => {
                let d = reflect(ray_in.d, n);
                if !reflects_above(ng, d) {
                    return None;
                }
                Some(BsdfSample {
                    scattered: Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, d), d, time: ray_in.time },
                    weight: *albedo,
                    pdf: 0.0,
                    is_delta: true,
                    eta: 1.0,
                })
            }
            Material::Dielectric { ior, absorption } => {
                // Snell の法則: η = η_i / η_t（入射側/透過側の屈折率比）
                let eta = if entering { 1.0 / ior } else { *ior };
                let cos_i = (-ray_in.d).dot(n).max(0.0);
                // Schlick 近似によるフレネル反射率（媒質から出る側は透過側の cos を使う）
                let fresnel = schlick_dielectric(cos_i, eta, *ior);

                let mut current_beta = Color::new(1.0, 1.0, 1.0);
                // Beer-Lambert for path inside the medium (only apply when exiting/inside)
                if !entering {
                    let dist = hit.t.max(0.0);
                    let att = Color::new(
                        (-absorption.r() * dist).exp(),
                        (-absorption.g() * dist).exp(),
                        (-absorption.b() * dist).exp(),
                    );
                    current_beta = current_beta.hadamard(att);
                }

                let refl_dir = reflect(ray_in.d, n);
                let refr_dir = refract(ray_in.d, n, eta);

                let choose_refl = refr_dir.is_none() || rng.next_f64() < fresnel;
                // 透過なら η_t/η_i（eta は η_i/η_t）、反射なら 1
                let eta_scatter = if choose_refl { 1.0 } else { 1.0 / eta };
                let d = if choose_refl {
                    current_beta = current_beta * (fresnel / fresnel.max(1e-6));
                    refl_dir
                } else {
                    let tdir = refr_dir.unwrap();
                    // Correct radiance scaling for transmission
                    let scale = (1.0 - fresnel) / (1.0 - fresnel).max(1e-6);
                    current_beta = current_beta * scale * (eta * eta);
                    tdir
                };
                // 反射なら幾何的に表側、透過なら裏側でなければならない（補間法線による破綻を捨てる）
                if choose_refl != reflects_above(ng, d) {
                    return None;
                }
                Some(BsdfSample {
                    scattered: Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, d), d, time: ray_in.time },
                    weight: current_beta,
                    pdf: 0.0,
                    is_delta: true,
                    eta: eta_scatter,
                })
            }
            Material::Ggx { albedo, alpha } => {
                let alpha_val = alpha.max(1e-3); // α → 0 で鏡面に近づく（数値安定性のためクランプ）
                // 法線から接線空間の基底ベクトルを構築
                let (t, b) = tangent_frame(n);

                // ワールド空間の出射方向を接線空間に変換
                let wo = (-ray_in.d).norm();
                let wo_local = Vec3::new(wo.dot(t), wo.dot(b), wo.dot(n));
                if wo_local.z <= 0.0 {
                    return None;
                }
                let m_local = sample_ggx_vndf(wo_local, alpha_val, rng.next_f64(), rng.next_f64());
                let m = t * m_local.x + b * m_local.y + n * m_local.z;

                let d = reflect(ray_in.d, m);
                let cos_o = d.dot(n);
                let cos_i = (-ray_in.d).dot(n);
                if cos_o <= 1e-6 || cos_i <= 1e-6 {
                    return None;
                }

                if !reflects_above(ng, d) {
                    return None;
                }
                let cos_h = n.dot(m).max(0.0);
                let d_ggx = ggx_distribution(alpha_val, cos_h);
                let g = ggx_smith(alpha_val, cos_i, cos_o);
                let f = fresnel_schlick(d.dot(m).abs(), *albedo);

                let denom = 4.0 * cos_i * cos_o + 1e-6;
                let spec = f * (d_ggx * g / denom); // = ggx_eval(albedo, alpha, n, wo, d)

                // VNDF サンプリングなので重みも VNDF PDF で割る（報告 PDF と一貫）。
                // weight = f·cos_o/pdf, pdf は eval() が返すものと同一。
                let pdf = ggx_pdf(alpha_val, n, wo, d);

                Some(BsdfSample {
                    scattered: Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, d), d, time: ray_in.time },
                    weight: spec * (cos_o / pdf.max(1e-6)),
                    pdf,
                    is_delta: false,
                    eta: 1.0,
                })
            }
            Material::Subsurface { albedo } => {
                // 向き補正済み n 周りのコサイン半球サンプリングで Lambert と同じ契約に揃える:
                // weight = f·cos/pdf = albedo、pdf = eval() の pdf、原点は hit.p（NEE と同一点）。
                // 以前は散乱距離ぶん原点を面の内側へずらしていたが、NEE のシャドウレイ
                // （hit.p 起点）と別の点を推定して MIS が不整合になるため撤去した。
                let d = sample_cosine_hemisphere(n, rng);
                if !reflects_above(ng, d) {
                    return None;
                }
                Some(BsdfSample {
                    scattered: Ray { o: offset_ray_origin(hit.p, hit.p_error, hit.ng, d), d, time: ray_in.time },
                    weight: *albedo,
                    pdf: n.dot(d).max(0.0) / PI,
                    is_delta: false,
                    eta: 1.0,
                })
            }
            Material::DiffuseLight { .. } => None,
        }
    }

    /// BSDF 値 `f`（cosine 抜き）と立体角 PDF を返す（NEE での MIS 計算用）。
    ///
    /// `n` はシェーディング点の向き付き法線、`wo`/`wi` は出射/入射方向。
    /// デルタ散乱マテリアルは有限の値を持たないため `(0, 0)` を返す。
    pub fn eval(&self, wo: Vec3, wi: Vec3, n: Vec3) -> (Color, f64) {
        match *self {
            Material::Lambert { albedo, .. } | Material::Subsurface { albedo } => {
                let cos = n.dot(wi).max(0.0);
                if cos <= 0.0 {
                    (Color::new(0.0, 0.0, 0.0), 0.0)
                } else {
                    (albedo * (1.0 / PI), cos / PI)
                }
            }
            Material::Ggx { albedo, alpha } => {
                let a = alpha.max(1e-3);
                (ggx_eval(albedo, a, n, wo, wi), ggx_pdf(a, n, wo, wi))
            }
            _ => (Color::new(0.0, 0.0, 0.0), 0.0),
        }
    }
}

/// GGX BRDF の評価: f(ω_i, ω_o) = F(θ_h) · D(θ_h) · G(θ_i, θ_o) / (4 cos θ_i cos θ_o)
fn ggx_eval(albedo: Color, alpha: f64, n: Vec3, wo: Vec3, wi: Vec3) -> Color {
    let cos_i = n.dot(wo);
    let cos_o = n.dot(wi);
    if cos_i <= 0.0 || cos_o <= 0.0 {
        return Color::new(0.0, 0.0, 0.0);
    }
    let m = (wo + wi).norm();
    let cos_h = n.dot(m).max(0.0);
    let d_ggx = ggx_distribution(alpha, cos_h);
    let g = ggx_smith(alpha, cos_i, cos_o);
    let f = fresnel_schlick(wi.dot(m).abs(), albedo);
    let denom = 4.0 * cos_i * cos_o + 1e-6;
    f * (d_ggx * g / denom)
}

/// GGX サンプリングの立体角 PDF（VNDF サンプリングに対応、反射方向 `wi` について）。
fn ggx_pdf(alpha: f64, n: Vec3, wo: Vec3, wi: Vec3) -> f64 {
    let cos_i = n.dot(wo);
    let cos_o = n.dot(wi);
    if cos_i <= 0.0 || cos_o <= 0.0 {
        return 0.0;
    }
    let m = (wo + wi).norm();
    let cos_h = n.dot(m).max(0.0);
    let d_ggx = ggx_distribution(alpha, cos_h);
    let g1 = ggx_smith_g1(alpha, cos_i);
    // 可視法線分布: D_wo(m) = G1(wo)·max(0, wo·m)·D(m) / (n·wo)（Heitz 2018 式 2）。
    // 反射のヤコビアン 1/(4|wi·m|) を掛けると pdf(wi) = G1(wo)·D(m) / (4 n·wo)。
    let pdf_m = d_ggx * g1 * wo.dot(m).max(0.0) / cos_i.max(1e-6);
    let denom = 4.0 * wi.dot(m).abs().max(1e-6);
    (pdf_m / denom).max(0.0)
}

/// 誘電体境界の Schlick 近似による反射率。`eta` = η_i/η_t、`ior` は媒質の屈折率。
///
/// Schlick 近似の角度は、常に屈折率の**低い側**（ここでは外側）の角度でなければならない。
/// 外から入射する場合（eta < 1）は入射角の cos、媒質から出る場合（eta > 1）は Snell の法則で
/// 求めた透過角の cos を使う。こうすると同じ境界を逆向きに通る光路で反射率が一致し
/// （Stokes の関係）、臨界角に近づくと反射率が 1 に連続的に近づく。
/// 全反射（透過角が存在しない）では 1 を返す。
fn schlick_dielectric(cos_i: f64, eta: f64, ior: f64) -> f64 {
    let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
    let cos = if eta > 1.0 {
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
        if sin2_t >= 1.0 {
            return 1.0;
        }
        (1.0 - sin2_t).sqrt()
    } else {
        cos_i
    };
    f0 + (1.0 - f0) * pow5(1.0 - cos)
}

/// x^5 を効率的に計算する（フレネルの Schlick 近似用）。
fn pow5(x: f64) -> f64 {
    let x2 = x * x;
    x2 * x2 * x
}

/// Schlick 近似によるフレネル反射率: F(θ) = F_0 + (1 - F_0)(1 - cosθ)^5
fn fresnel_schlick(cos_theta: f64, f0: Color) -> Color {
    let one = Color::new(1.0, 1.0, 1.0);
    let x = 1.0 - cos_theta.max(0.0);
    f0 + (one - f0) * pow5(x)
}

/// GGX（Trowbridge-Reitz）法線分布関数: D(θ_h) = α² / (π (cos²θ_h (α²-1) + 1)²)
fn ggx_distribution(alpha: f64, cos_theta_h: f64) -> f64 {
    if cos_theta_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0;
    a2 / (std::f64::consts::PI * denom * denom)
}

/// Smith の GGX 遮蔽関数（片方向）: G1(θ) = 2 / (1 + √(1 + α²tan²θ))
fn ggx_smith_g1(alpha: f64, cos_theta: f64) -> f64 {
    if cos_theta <= 0.0 {
        return 0.0;
    }
    let a = alpha * (1.0 - cos_theta * cos_theta).max(0.0).sqrt() / cos_theta.max(1e-6);
    2.0 / (1.0 + (1.0 + a * a).sqrt())
}

/// Smith の分離可能な遮蔽-シャドウイング関数: G(θ_i, θ_o) = G1(θ_i) · G1(θ_o)
fn ggx_smith(alpha: f64, cos_i: f64, cos_o: f64) -> f64 {
    ggx_smith_g1(alpha, cos_i) * ggx_smith_g1(alpha, cos_o)
}

/// コサイン重み付き半球サンプリング。
///
/// PDF = cos(θ) / π で、Lambert BRDF のサンプリングに最適。
/// Malley の方法: 単位円上の一様サンプルを半球に投影する。
fn sample_cosine_hemisphere(n: Vec3, rng: &mut Rng) -> Vec3 {
    let u = rng.next_f64();
    let v = rng.next_f64();
    let r = u.sqrt();
    let phi = 2.0 * std::f64::consts::PI * v;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u).sqrt();

    let (t, b) = tangent_frame(n);
    t * x + b * y + n * z
}

/// 法線 `n` から正規直交接線フレーム (tangent, bitangent) を構築する。
pub(crate) fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() > 0.9 { Vec3::new(0.0,1.0,0.0) } else { Vec3::new(1.0,0.0,0.0) };
    let t = n.cross(a).norm();
    let b = t.cross(n);
    (t, b)
}

/// GGX 可視法線分布（VNDF）のサンプリング。
///
/// 従来の NDF サンプリングより効率が高く、特にグレイジング角での
/// 無効サンプル（裏面を向くハーフベクトル）を大幅に削減する。
///
/// 参考文献: Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 2018
/// https://jcgt.org/published/0007/04/01/
fn sample_ggx_vndf(wo: Vec3, alpha: f64, u1: f64, u2: f64) -> Vec3 {
    let v = Vec3::new(alpha * wo.x, alpha * wo.y, wo.z).norm();

    let (t1, t2) = if v.z < 0.9999 {
        let t1 = Vec3::new(-v.y, v.x, 0.0).norm();
        let t2 = v.cross(t1);
        (t1, t2)
    } else {
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
    };

    let r = u1.sqrt();
    let phi = std::f64::consts::TAU * u2;
    let t1p = r * phi.cos();
    let t2p_init = r * phi.sin();
    let s = 0.5 * (1.0 + v.z);
    // Heitz 2018 Listing 1: t2 = (1 − s)·√(1 − t1²) + s·t2
    let t2p = (1.0 - s) * (1.0 - t1p * t1p).max(0.0).sqrt() + s * t2p_init;

    let nh = (t1 * t1p + t2 * t2p + v * (1.0 - t1p * t1p - t2p * t2p).max(0.0).sqrt()).norm();
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z).norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 下向きレイが床（法線 +Y）に当たる状況の Hit を作る。
    fn floor_hit() -> (Ray, Hit) {
        let ray = Ray { o: Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 };
        let hit = Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng: Vec3::new(0.0, 1.0, 0.0), ns: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0), uv: (0.0, 0.0) };
        (ray, hit)
    }

    /// sample() が報告する PDF は、同じ方向に対する eval() の PDF と一致する（Lambert）。
    #[test]
    fn lambert_sample_pdf_matches_eval() {
        let mat = Material::Lambert { albedo: Color::new(0.6, 0.4, 0.2) };
        let (ray, hit) = floor_hit();
        let mut rng = Rng::new(1);
        let n = hit.ng; // 入射なので向き付き法線 = 幾何法線
        let wo = (-ray.d).norm();
        for _ in 0..1000 {
            let s = mat.sample(&ray, &hit, &mut rng).unwrap();
            let (_, pdf_eval) = mat.eval(wo, s.scattered.d, n);
            assert!((s.pdf - pdf_eval).abs() < 1e-9, "{} vs {}", s.pdf, pdf_eval);
        }
    }

    /// 法線 +Y の点に、出射方向 `wo`（天頂角 `theta_o`）から入射するレイと Hit。
    fn oblique_hit(theta_o: f64) -> (Ray, Hit, Vec3) {
        let wo = Vec3::new(theta_o.sin(), theta_o.cos(), 0.0);
        let ray = Ray { o: wo * 2.0, d: -wo, time: 0.0 };
        let hit = Hit { t: 2.0, p: Vec3::new(0.0, 0.0, 0.0), ng: Vec3::new(0.0, 1.0, 0.0), ns: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0), uv: (0.0, 0.0) };
        (ray, hit, wo)
    }

    /// GGX テストの (α, θo) の組。垂直入射・斜め入射・グレイジングを含む。
    const GGX_CASES: [(f64, f64); 8] = [
        (0.1, 0.0), (0.25, 0.0), (0.25, 0.6), (0.25, 1.2), (0.5, 1.0), (0.1, 1.45), (1.0, 0.3), (1.0, 1.2),
    ];

    /// `check_ggx_sample_matches_eval` の本体。失敗理由を返す（シード掃引テストと共有）。
    ///
    /// sample() の PDF は同じ方向の eval() の PDF と一致し、weight は f·cos/pdf と一致する。
    /// どの (α, θo) でも有効サンプルが十分に出ることを要求する（以前は垂直入射で常に None に
    /// なり、アサーションが一度も実行されない空振りテストだった）。有効率の下限 2/5 は、
    /// 最も有効率の低い (1.0, 0.3)（真値 ≈ 0.512、2000 試行で SD ≈ 0.011）に対して約 10σ の余裕。
    fn check_ggx_sample_matches_eval(seed: u64) -> Result<(), String> {
        let mut rng = Rng::new(seed);
        for &(alpha, theta_o) in &GGX_CASES {
            let mat = Material::Ggx { albedo: Color::new(0.9, 0.8, 0.7), alpha };
            let (ray, hit, wo) = oblique_hit(theta_o);
            let n = hit.ng;
            let trials = 2000;
            let mut valid = 0;
            for _ in 0..trials {
                let Some(s) = mat.sample(&ray, &hit, &mut rng) else { continue };
                valid += 1;
                let wi = s.scattered.d;
                let (f, pdf) = mat.eval(wo, wi, n);
                if (s.pdf - pdf).abs() > 1e-9 * pdf.max(1.0) {
                    return Err(format!("α={} θo={}: pdf {} vs eval {}", alpha, theta_o, s.pdf, pdf));
                }
                let expected = f * (n.dot(wi).max(0.0) / pdf.max(1e-6));
                for (a, b) in [(s.weight.r(), expected.r()), (s.weight.g(), expected.g()), (s.weight.b(), expected.b())] {
                    if (a - b).abs() > 1e-9 * b.abs().max(1.0) {
                        return Err(format!("α={} θo={}: weight {} vs f·cos/pdf {}", alpha, theta_o, a, b));
                    }
                }
            }
            if valid * 5 < trials * 2 {
                return Err(format!("α={} θo={}: only {} / {} valid samples", alpha, theta_o, valid, trials));
            }
        }
        Ok(())
    }

    /// 標本の平均と、平均の標準誤差。
    fn mean_and_se(sum: f64, sum_sq: f64, n: usize) -> (f64, f64) {
        let n = n as f64;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0) * n / (n - 1.0);
        (mean, (var / n).sqrt())
    }

    /// `ggx_sampling_is_unbiased_and_pdf_normalized` の本体。失敗理由を返す（シード掃引テストと共有）。
    ///
    /// - E[weight] = ∫ f·cos dω（一様半球サンプリングによる参照積分と比較）
    /// - ∫ pdf dω = 有効サンプル率（地平線の下へ反射した分だけ 1 より小さい）
    ///
    /// 許容誤差は固定の相対値ではなく、両辺の標本分散から求めた標準誤差 SE に対して
    /// |Δ| < 5·SE + 0.003（相対の絶対下限）とする。鋭いローブ（α = 0.1）では一様半球の参照積分の
    /// 分散が大きく、固定 2% だとシード次第で約半数が偶然に失敗していた。
    fn check_ggx_sampling_is_unbiased(seed: u64) -> Result<(), String> {
        const K: f64 = 5.0;
        const FLOOR: f64 = 0.003;
        let tau = std::f64::consts::TAU;
        let mut rng = Rng::new(seed);
        for &(alpha, theta_o) in &GGX_CASES {
            let mat = Material::Ggx { albedo: Color::new(1.0, 1.0, 1.0), alpha };
            let (ray, hit, wo) = oblique_hit(theta_o);
            let n = hit.ng;

            let m = 200_000;
            let (mut sw, mut sw2, mut valid) = (0.0, 0.0, 0usize);
            for _ in 0..m {
                if let Some(s) = mat.sample(&ray, &hit, &mut rng) {
                    let w = s.weight.r();
                    sw += w;
                    sw2 += w * w;
                    valid += 1;
                }
            }
            let (mean_w, se_w) = mean_and_se(sw, sw2, m);
            let valid_rate = valid as f64 / m as f64;
            let se_rate = (valid_rate * (1.0 - valid_rate) / m as f64).sqrt();

            // 参照: 一様半球（pdf = 1/2π）での ∫ f·cos dω と ∫ pdf dω
            let (mut sf, mut sf2, mut sp, mut sp2) = (0.0, 0.0, 0.0, 0.0);
            for _ in 0..m {
                let z = rng.next_f64();
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = tau * rng.next_f64();
                let wi = Vec3::new(r * phi.cos(), z, r * phi.sin());
                let (f, pdf) = mat.eval(wo, wi, n);
                let (x, y) = (f.r() * z * tau, pdf * tau);
                sf += x;
                sf2 += x * x;
                sp += y;
                sp2 += y * y;
            }
            let (ref_fcos, se_fcos) = mean_and_se(sf, sf2, m);
            let (ref_pdf, se_pdf) = mean_and_se(sp, sp2, m);

            let case = format!("α={} θo={}", alpha, theta_o);
            if valid_rate <= 0.4 {
                return Err(format!("{}: valid rate {}", case, valid_rate));
            }
            let d_w = mean_w - ref_fcos;
            let tol_w = K * (se_w * se_w + se_fcos * se_fcos).sqrt() + FLOOR * ref_fcos;
            if d_w.abs() >= tol_w {
                return Err(format!("{}: E[weight] {} vs ∫f·cos {} (|Δ| {:.3e} ≥ tol {:.3e})", case, mean_w, ref_fcos, d_w.abs(), tol_w));
            }
            let d_p = ref_pdf - valid_rate;
            let tol_p = K * (se_pdf * se_pdf + se_rate * se_rate).sqrt() + FLOOR * valid_rate;
            if d_p.abs() >= tol_p {
                return Err(format!("{}: ∫pdf {} vs valid rate {} (|Δ| {:.3e} ≥ tol {:.3e})", case, ref_pdf, valid_rate, d_p.abs(), tol_p));
            }
            if ref_pdf >= 1.0 + K * se_pdf + FLOOR {
                return Err(format!("{}: ∫pdf {} exceeds 1 (SE {:.3e})", case, ref_pdf, se_pdf));
            }
        }
        Ok(())
    }

    #[test]
    fn ggx_sample_matches_eval_for_pdf_and_weight() {
        check_ggx_sample_matches_eval(7).unwrap();
    }

    /// VNDF サンプラの式の誤り・pdf の cos_h/wo·m の取り違え・weight や pdf の定数倍の誤りを検出する。
    #[test]
    fn ggx_sampling_is_unbiased_and_pdf_normalized() {
        check_ggx_sampling_is_unbiased(11).unwrap();
    }

    /// 上の 2 テストがシードに依存して偶然失敗しないことの確認（重いので通常は実行しない）。
    /// `cargo test --release --no-default-features ggx_tests_are_seed_robust -- --ignored`
    /// 環境変数 `TINYPT_GGX_SEEDS` で掃引するシード数を変えられる（既定 256）。
    #[test]
    #[ignore]
    fn ggx_tests_are_seed_robust() {
        let seeds: u64 = std::env::var("TINYPT_GGX_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let mut failures = Vec::new();
        for seed in 0..seeds {
            if let Err(e) = check_ggx_sample_matches_eval(seed) {
                failures.push(format!("match seed {}: {}", seed, e));
            }
            if let Err(e) = check_ggx_sampling_is_unbiased(seed) {
                failures.push(format!("unbiased seed {}: {}", seed, e));
            }
        }
        assert!(failures.is_empty(), "{} failures over {} seeds:\n{}", failures.len(), seeds, failures.join("\n"));
    }

    /// Lambert の重みは albedo に一致する（f·cos/pdf が打ち消し合う／拡散の白炉テスト）。
    #[test]
    fn lambert_weight_is_albedo() {
        let albedo = Color::new(0.5, 0.7, 0.3);
        let mat = Material::Lambert { albedo };
        let (ray, hit) = floor_hit();
        let mut rng = Rng::new(3);
        let s = mat.sample(&ray, &hit, &mut rng).unwrap();
        assert!((s.weight.r() - albedo.r()).abs() < 1e-12);
        assert!((s.weight.g() - albedo.g()).abs() < 1e-12);
        assert!((s.weight.b() - albedo.b()).abs() < 1e-12);
    }

    /// GGX BRDF はヘルムホルツ相反性を満たす: f(ω_o, ω_i) = f(ω_i, ω_o)。
    #[test]
    fn ggx_eval_is_reciprocal() {
        let mat = Material::Ggx { albedo: Color::new(0.8, 0.5, 0.2), alpha: 0.25 };
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.4, 1.0, 0.1).norm();
        let wi = Vec3::new(-0.3, 1.0, 0.2).norm();
        let (f1, _) = mat.eval(wo, wi, n);
        let (f2, _) = mat.eval(wi, wo, n);
        assert!((f1.r() - f2.r()).abs() < 1e-12);
        assert!((f1.g() - f2.g()).abs() < 1e-12);
        assert!((f1.b() - f2.b()).abs() < 1e-12);
    }

    /// Lambert の PDF は半球上で 1 に積分される（∫ pdf dω = 1）。
    /// 一様半球サンプリングによるモンテカルロ推定（pdf_uniform = 1/2π）。
    #[test]
    fn lambert_pdf_integrates_to_one() {
        let mat = Material::Lambert { albedo: Color::new(1.0, 1.0, 1.0) };
        let n = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(99);
        let n_samples = 200_000;
        let mut sum = 0.0;
        for _ in 0..n_samples {
            // 一様半球サンプリング: z ∈ [0,1] 一様
            let z = rng.next_f64();
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = std::f64::consts::TAU * rng.next_f64();
            let wi = Vec3::new(r * phi.cos(), z, r * phi.sin());
            let (_, pdf) = mat.eval(wo, wi, n);
            sum += pdf;
        }
        // ∫ pdf dω ≈ (1/N) Σ pdf / (1/2π) = (2π/N) Σ pdf
        let integral = 2.0 * PI * sum / n_samples as f64;
        assert!((integral - 1.0).abs() < 0.02, "integral = {}", integral);
    }

    /// 下から上向きのレイが床（幾何法線 +Y）の裏面に当たる状況の Hit を作る。
    fn floor_backface_hit() -> (Ray, Hit) {
        let ray = Ray { o: Vec3::new(0.0, -1.0, 0.0), d: Vec3::new(0.0, 1.0, 0.0), time: 0.0 };
        let hit = Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng: Vec3::new(0.0, 1.0, 0.0), ns: Vec3::new(0.0, 1.0, 0.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.0, 0.0), uv: (0.0, 0.0) };
        (ray, hit)
    }

    fn subsurface() -> Material {
        Material::Subsurface { albedo: Color::new(0.8, 0.5, 0.3) }
    }

    /// Subsurface: sample() の PDF は eval() の PDF と一致する（表面・裏面の両方）。
    #[test]
    fn subsurface_sample_pdf_matches_eval() {
        let mat = subsurface();
        for (ray, hit) in [floor_hit(), floor_backface_hit()] {
            let n = if hit.ng.dot(ray.d) < 0.0 { hit.ng } else { -hit.ng };
            let wo = (-ray.d).norm();
            let mut rng = Rng::new(5);
            for _ in 0..1000 {
                let s = mat.sample(&ray, &hit, &mut rng).unwrap();
                let (_, pdf_eval) = mat.eval(wo, s.scattered.d, n);
                assert!((s.pdf - pdf_eval).abs() < 1e-9, "{} vs {}", s.pdf, pdf_eval);
            }
        }
    }

    /// Subsurface: weight == f·cos/pdf（期待値が albedo/2 に落ちていた不整合への回帰テスト）。
    #[test]
    fn subsurface_weight_matches_f_cos_over_pdf() {
        let mat = subsurface();
        let (ray, hit) = floor_hit();
        let n = hit.ng;
        let wo = (-ray.d).norm();
        let mut rng = Rng::new(13);
        for _ in 0..1000 {
            let s = mat.sample(&ray, &hit, &mut rng).unwrap();
            let wi = s.scattered.d;
            let (f, pdf) = mat.eval(wo, wi, n);
            if pdf <= 1e-9 {
                continue;
            }
            let expected = f * (n.dot(wi).max(0.0) / pdf);
            assert!((s.weight.r() - expected.r()).abs() < 1e-9, "{} vs {}", s.weight.r(), expected.r());
            assert!((s.weight.g() - expected.g()).abs() < 1e-9);
            assert!((s.weight.b() - expected.b()).abs() < 1e-9);
        }
    }

    /// Subsurface の裏面ヒット: 散乱方向は向き補正済み法線側、PDF は正、
    /// 原点は NEE と同じ hit.p（+ε）に置かれる。pdf=0 で MIS 重みが消える不具合への回帰テスト。
    #[test]
    fn subsurface_backface_hit_has_positive_pdf() {
        let mat = subsurface();
        let (ray, hit) = floor_backface_hit();
        let n = -hit.ng;
        let mut rng = Rng::new(21);
        for _ in 0..1000 {
            let s = mat.sample(&ray, &hit, &mut rng).unwrap();
            let d = s.scattered.d;
            assert!(n.dot(d) >= 0.0, "direction not in oriented hemisphere");
            assert!(s.pdf > 0.0 || n.dot(d) < 1e-9, "pdf = {}", s.pdf);
            let expected_o = offset_ray_origin(hit.p, hit.p_error, hit.ng, d);
            assert!((s.scattered.o - expected_o).len() < 1e-12);
        }
    }

    /// 誘電体の厳密な Fresnel 反射率（非偏光、s/p 偏光の平均）。`eta` = η_i/η_t。全反射なら 1。
    fn fresnel_dielectric_exact(cos_i: f64, eta: f64) -> f64 {
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
        if sin2_t >= 1.0 {
            return 1.0;
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
        let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
        0.5 * (rs * rs + rp * rp)
    }

    /// 旧実装（出る側でも入射角の cos を使っていた）。比較用。
    fn schlick_dielectric_old(cos_i: f64, ior: f64) -> f64 {
        let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
        f0 + (1.0 - f0) * pow5(1.0 - cos_i)
    }

    /// 媒質から出る側の Schlick 反射率は、同じ光路を外から入る側の反射率と一致する（Stokes の関係。
    /// 厳密な Fresnel はこれを満たす）。旧実装は内側の角度を使っていたため一致しなかった。
    #[test]
    fn dielectric_schlick_is_symmetric_across_the_interface() {
        for &ior in &[1.33f64, 1.5, 2.4] {
            for k in 0..=100 {
                let cos_out = k as f64 / 100.0; // 外側（低屈折率側）の角度
                let sin_out = (1.0 - cos_out * cos_out).sqrt();
                let cos_in = (1.0 - (sin_out / ior).powi(2)).sqrt(); // Snell で内側の角度
                let enter = schlick_dielectric(cos_out, 1.0 / ior, ior);
                let exit = schlick_dielectric(cos_in, ior, ior);
                assert!((enter - exit).abs() < 1e-12, "ior={} cos_out={}: enter {} vs exit {}", ior, cos_out, enter, exit);
                let exact_enter = fresnel_dielectric_exact(cos_out, 1.0 / ior);
                let exact_exit = fresnel_dielectric_exact(cos_in, ior);
                assert!((exact_enter - exact_exit).abs() < 1e-12, "exact Fresnel must be symmetric");
            }
        }
    }

    /// 出る側の反射率は厳密な Fresnel に近い（旧実装より大幅に誤差が小さい）。臨界角を超えると 1。
    /// 入る側は旧実装と同一。
    #[test]
    fn dielectric_schlick_exit_side_tracks_exact_fresnel() {
        for &ior in &[1.33f64, 1.5, 2.4] {
            let critical_cos = (1.0 - 1.0 / (ior * ior)).sqrt();
            let (mut max_new, mut max_old) = (0.0f64, 0.0f64);
            for k in 0..=2000 {
                let cos_i = k as f64 / 2000.0;
                let exact = fresnel_dielectric_exact(cos_i, ior);
                let new = schlick_dielectric(cos_i, ior, ior);
                let old = schlick_dielectric_old(cos_i, ior);
                max_new = max_new.max((new - exact).abs());
                max_old = max_old.max((old - exact).abs());
                if cos_i < critical_cos {
                    assert_eq!(new, 1.0, "ior={} cos_i={}: beyond the critical angle must reflect totally", ior, cos_i);
                }
                // 入る側は変更なし
                assert_eq!(schlick_dielectric(cos_i, 1.0 / ior, ior), schlick_dielectric_old(cos_i, ior));
            }
            // Schlick 近似そのものの誤差（入る側の最大誤差）と同程度に収まる
            let max_enter = (0..=2000)
                .map(|k| k as f64 / 2000.0)
                .map(|c| (schlick_dielectric(c, 1.0 / ior, ior) - fresnel_dielectric_exact(c, 1.0 / ior)).abs())
                .fold(0.0, f64::max);
            assert!(max_new <= max_enter + 1e-12, "ior={}: exit-side error {} exceeds enter-side error {}", ior, max_new, max_enter);
            assert!(max_old > 0.5, "ior={}: old exit-side error {} (expected large near the critical angle)", ior, max_old);
            println!("ior={}: max |Schlick−exact| exit side new {:.4} old {:.4}, enter side {:.4}", ior, max_new, max_old, max_enter);
        }
    }

    /// is_delta / emitted の分類が正しい。
    #[test]
    fn classification_is_correct() {
        assert!(Material::Metal { albedo: Color::new(1.0, 1.0, 1.0) }.is_delta());
        assert!(Material::Dielectric { ior: 1.5, absorption: Color::new(0.0, 0.0, 0.0) }.is_delta());
        assert!(!Material::Lambert { albedo: Color::new(1.0, 1.0, 1.0) }.is_delta());
        assert!(!Material::Ggx { albedo: Color::new(1.0, 1.0, 1.0), alpha: 0.2 }.is_delta());

        let emit = Color::new(3.0, 3.0, 3.0);
        assert!(Material::DiffuseLight { emit }.emitted().is_some());
        assert!(Material::Lambert { albedo: Color::new(1.0, 1.0, 1.0) }.emitted().is_none());
    }

    // ---- スムーズシェーディング: 幾何法線とシェーディング法線の分離 ----

    /// シェーディング法線が幾何法線と違っても、**レイの原点ずらしは幾何法線で行う**。
    ///
    /// これは仕様であると同時にミューテーション検出でもある: `sample` の
    /// `offset_ray_origin(..., hit.ng, d)` を `hit.ns` に書き換えると、
    /// 下の 2 つの assert のうち後者（ns 基準とは一致しない）が落ちる。
    #[test]
    fn ray_origin_offset_uses_the_geometric_normal_not_the_shading_normal() {
        let ng = Vec3::new(0.0, 1.0, 0.0);
        let ns = Vec3::new(0.6, 0.8, 0.0).norm(); // 幾何法線から約 37 度
        let hit = Hit {
            t: 1.0,
            p: Vec3::new(3.0, 5.0, -7.0), // 原点から離して p_error を成分ごとに非ゼロにする
            ng,
            ns,
            mat_id: 0,
            prim_id: 0,
            inst_id: None,
            p_error: Vec3::new(4e-16, 7e-16, 9e-16),
            bary: (0.25, 0.25),
            uv: (0.0, 0.0),
        };
        let ray = Ray { o: hit.p + Vec3::new(0.3, 1.0, 0.2), d: Vec3::new(-0.3, -1.0, -0.2).norm(), time: 0.0 };
        let mat = Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) };
        let mut rng = Rng::new(5);
        let mut checked = 0;
        for _ in 0..200 {
            let Some(s) = mat.sample(&ray, &hit, &mut rng) else { continue };
            let d = s.scattered.d;
            let by_ng = offset_ray_origin(hit.p, hit.p_error, ng, d);
            let by_ns = offset_ray_origin(hit.p, hit.p_error, ns, d);
            assert_eq!(s.scattered.o.x.to_bits(), by_ng.x.to_bits(), "原点ずらしは幾何法線基準であること");
            assert_eq!(s.scattered.o.y.to_bits(), by_ng.y.to_bits());
            assert_eq!(s.scattered.o.z.to_bits(), by_ng.z.to_bits());
            assert_ne!(by_ng.x.to_bits(), by_ns.x.to_bits(), "この配置では 2 つのずらし方は実際に違う");
            checked += 1;
        }
        assert!(checked > 100, "サンプルが少なすぎる ({})", checked);
    }

    /// **破綻判定は全マテリアルで幾何法線が基準**: 反射サンプルは必ず幾何的に表側へ、
    /// 透過サンプルは必ず幾何的に裏側へ出る。
    ///
    /// シェーディング法線が傾いていると、`ns` 基準では「面を突き抜けていない」方向を透過と判定したり、
    /// 幾何的には面の裏へ潜る方向を反射として残したりしうる。そのまま飛ばすと、原点ずらしは
    /// 幾何法線基準なのでレイが自分のメッシュの内側を進む。
    ///
    /// ミューテーション検出: どのマテリアルでも `reflects_above(ng, d)` の `ng` を `n`（ns）にすると落ちる。
    /// **マテリアルごとに同じ判定が書かれている**ので、1 つだけ直して安心しないようループで回す
    /// （`Ggx` だけ ns 基準にすると画像が 13% 変わるのに、誘電体だけのテストでは素通りした）。
    #[test]
    fn breakdown_check_is_geometric_for_every_material() {
        let ng = Vec3::new(0.0, 1.0, 0.0);
        let ns = Vec3::new(0.6, 0.8, 0.0).norm(); // ng から約 37 度
        // 浅い角度から深い角度まで掃く（全反射の境界付近も含める）
        let dirs = [
            Vec3::new(0.10, -1.0, 0.0).norm(),
            Vec3::new(0.60, -1.0, 0.0).norm(),
            Vec3::new(1.00, -0.6, 0.0).norm(),
            Vec3::new(1.00, -0.2, 0.0).norm(),
            Vec3::new(-1.0, -0.3, 0.0).norm(),
            Vec3::new(-0.4, -1.0, 0.3).norm(),
        ];
        let cases: [(&str, Material, bool); 5] = [
            ("Lambert", Material::Lambert { albedo: Color::new(0.8, 0.8, 0.8) }, false),
            ("Metal", Material::Metal { albedo: Color::new(0.9, 0.9, 0.9) }, false),
            ("Ggx", Material::Ggx { albedo: Color::new(0.9, 0.9, 0.9), alpha: 0.35 }, false),
            ("Subsurface", Material::Subsurface { albedo: Color::new(0.7, 0.7, 0.7) }, false),
            ("Dielectric", Material::Dielectric { ior: 1.5, absorption: Color::new(0.0, 0.0, 0.0) }, true),
        ];
        for (name, mat, transmits) in cases {
            let mut rng = Rng::new(77);
            let (mut accepted, mut transmitted, mut rejected) = (0usize, 0usize, 0usize);
            for d in dirs {
                let hit = Hit {
                    t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng, ns, mat_id: 0, prim_id: 0, inst_id: None,
                    p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.25, 0.25), uv: (0.0, 0.0),
                };
                let ray = Ray { o: -d * 2.0, d, time: 0.0 };
                for _ in 0..4000 {
                    match mat.sample(&ray, &hit, &mut rng) {
                        Some(s) => {
                            let side = s.scattered.d.dot(ng);
                            // eta == 1 が反射、それ以外が透過（BsdfSample の規約）
                            if s.eta == 1.0 {
                                assert!(side > 0.0, "{}: 反射なのに幾何的に裏側へ出た（d·ng = {}）", name, side);
                            } else {
                                assert!(side < 0.0, "{}: 透過なのに幾何的に表側へ出た（d·ng = {}）", name, side);
                                transmitted += 1;
                            }
                            accepted += 1;
                        }
                        None => rejected += 1,
                    }
                }
            }
            assert!(accepted > 1000, "{}: 採用されたサンプルが少なすぎる ({})", name, accepted);
            // 傾いた ns のせいで捨てられるサンプルが実際に出ている（テストが空回りしていないこと）
            assert!(rejected > 0, "{}: 破綻サンプルが 1 つも出ない配置ではテストにならない", name);
            if transmits {
                assert!(transmitted > 1000, "{}: 透過サンプルが少なすぎる ({})", name, transmitted);
                assert!(accepted - transmitted > 200, "{}: 反射サンプルが少なすぎる ({})", name, accepted - transmitted);
            } else {
                assert_eq!(transmitted, 0, "{}: 透過しないはずのマテリアルで透過サンプルが出た", name);
            }
        }
    }

    /// 散乱方向は**シェーディング法線**の周りに分布する（cos 重み付き半球が ns 側に寄る）。
    /// 同時に、幾何法線の裏側へ出るサンプルは 1 つも返らない（破綻サンプルは捨てる方針）。
    #[test]
    fn scattering_follows_the_shading_normal_and_never_goes_below_the_geometry() {
        let ng = Vec3::new(0.0, 1.0, 0.0);
        let ns = Vec3::new(0.6, 0.8, 0.0).norm();
        let hit = Hit {
            t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), ng, ns, mat_id: 0, prim_id: 0, inst_id: None,
            p_error: Vec3::new(1e-15, 1e-15, 1e-15), bary: (0.25, 0.25), uv: (0.0, 0.0),
        };
        let ray = Ray { o: Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 };
        let mat = Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) };
        let mut rng = Rng::new(9);
        let (mut kept, mut rejected) = (0, 0);
        let mut mean = Vec3::new(0.0, 0.0, 0.0);
        for _ in 0..20_000 {
            match mat.sample(&ray, &hit, &mut rng) {
                Some(s) => {
                    assert!(s.scattered.d.dot(ng) > 0.0, "幾何法線の裏へ出るサンプルが残っている");
                    mean = mean + s.scattered.d;
                    kept += 1;
                }
                None => rejected += 1,
            }
        }
        assert!(rejected > 0, "ns が ng から 37 度傾いていれば、捨てられるサンプルが出るはず");
        // 平均方向は ns 側に寄る（面法線周りなら x 成分は 0 になる）
        let mean = mean / (kept as f64);
        assert!(mean.x > 0.05, "散乱の平均方向が ns 側に寄っていない: {:?}", mean);
        // 捨てた割合は「ns 基準の半球のうち ng の裏側」の面積比なので、極端に大きくはならない
        let reject_frac = rejected as f64 / (kept + rejected) as f64;
        assert!(reject_frac < 0.2, "捨てすぎ: {}", reject_frac);
    }
}
