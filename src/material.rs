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
//! - **Subsurface**: 簡易サブサーフェス散乱（指数分布による散乱距離）
//! - **DiffuseLight**: 拡散発光体（散乱なし、放射輝度を返す）

use std::f64::consts::PI;

use crate::constants::RAY_EPSILON;
use crate::geometry::Hit;
use crate::math::{reflect, refract, Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;

#[derive(Clone, Copy)]
/// 積分器が対応するマテリアルモデル。各 variant が一つの BSDF を表す。
pub enum Material {
    /// 完全拡散反射（Lambertian BRDF）
    Lambert { albedo: Color },
    /// 完全鏡面反射（デルタ BRDF）
    Metal   { albedo: Color },
    /// 誘電体（屈折 + フレネル反射 + Beer-Lambert 吸収）
    Dielectric { ior: f64, absorption: Color },
    /// GGX マイクロファセット（粗さパラメータ alpha）
    Ggx     { albedo: Color, alpha: f64 },
    /// 簡易サブサーフェス散乱
    Subsurface { albedo: Color, scatter_dist: f64 },
    /// 拡散面光源
    DiffuseLight { emit: Color },
}

/// BSDF サンプリングの結果。
///
/// 散乱レイは原点込みで保持するため、透過（Dielectric）やサブサーフェスの
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
}

impl Material {
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
        // 法線をレイの進行方向に対して正しい向きに修正
        let entering = hit.n.dot(ray_in.d) < 0.0; // レイが表面に入射するか
        let n = if entering { hit.n } else { -hit.n };

        match self {
            Material::Lambert { albedo } => {
                let d = sample_cosine_hemisphere(n, rng);
                // f·cos/pdf = (albedo/π)·cos/(cos/π) = albedo
                Some(BsdfSample {
                    scattered: Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time },
                    weight: *albedo,
                    pdf: n.dot(d).max(0.0) / PI,
                    is_delta: false,
                })
            }
            Material::Metal { albedo } => {
                let d = reflect(ray_in.d, n);
                Some(BsdfSample {
                    scattered: Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time },
                    weight: *albedo,
                    pdf: 0.0,
                    is_delta: true,
                })
            }
            Material::Dielectric { ior, absorption } => {
                // Snell の法則: η = η_i / η_t（入射側/透過側の屈折率比）
                let eta = if entering { 1.0 / ior } else { *ior };
                let cos_i = (-ray_in.d).dot(n).max(0.0);
                // Schlick 近似によるフレネル反射率
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let fresnel = f0 + (1.0 - f0) * pow5(1.0 - cos_i);

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
                Some(BsdfSample {
                    scattered: Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time },
                    weight: current_beta,
                    pdf: 0.0,
                    is_delta: true,
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
                    scattered: Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time },
                    weight: spec * (cos_o / pdf.max(1e-6)),
                    pdf,
                    is_delta: false,
                })
            }
            Material::Subsurface { albedo, scatter_dist } => {
                let scale = scatter_dist.max(1e-4);
                let u = rng.next_f64().max(1e-12);
                let dist = -u.ln() * scale;

                let normal_out = hit.n;
                let d = sample_cosine_hemisphere(normal_out, rng);

                let att = *albedo * (-dist / scale).exp();
                let origin = hit.p - normal_out * dist + RAY_EPSILON * d;
                Some(BsdfSample {
                    scattered: Ray { o: origin, d, time: ray_in.time },
                    weight: att,
                    pdf: n.dot(d).max(0.0) / PI,
                    is_delta: false,
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
            Material::Lambert { albedo } | Material::Subsurface { albedo, .. } => {
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

/// GGX サンプリングの PDF を計算する（VNDF サンプリングに対応）。
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
    let pdf_m = d_ggx * g1 * cos_h / cos_i.max(1e-6);
    let denom = 4.0 * wi.dot(m).abs().max(1e-6);
    (pdf_m / denom).max(0.0)
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
fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
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
    let t2p = (1.0 - s).sqrt() * t2p_init + s * (1.0 - t1p * t1p).max(0.0).sqrt();

    let nh = (t1 * t1p + t2 * t2p + v * (1.0 - t1p * t1p - t2p * t2p).max(0.0).sqrt()).norm();
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z).norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 下向きレイが床（法線 +Y）に当たる状況の Hit を作る。
    fn floor_hit() -> (Ray, Hit) {
        let ray = Ray { o: Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 };
        let hit = Hit { t: 1.0, p: Vec3::new(0.0, 0.0, 0.0), n: Vec3::new(0.0, 1.0, 0.0), mat_id: 0 };
        (ray, hit)
    }

    /// sample() が報告する PDF は、同じ方向に対する eval() の PDF と一致する（Lambert）。
    #[test]
    fn lambert_sample_pdf_matches_eval() {
        let mat = Material::Lambert { albedo: Color::new(0.6, 0.4, 0.2) };
        let (ray, hit) = floor_hit();
        let mut rng = Rng::new(1);
        let n = hit.n; // 入射なので向き付き法線 = 幾何法線
        let wo = (-ray.d).norm();
        for _ in 0..1000 {
            let s = mat.sample(&ray, &hit, &mut rng).unwrap();
            let (_, pdf_eval) = mat.eval(wo, s.scattered.d, n);
            assert!((s.pdf - pdf_eval).abs() < 1e-9, "{} vs {}", s.pdf, pdf_eval);
        }
    }

    /// sample() が報告する PDF は eval() の PDF と一致する（GGX / VNDF）。
    #[test]
    fn ggx_sample_pdf_matches_eval() {
        let mat = Material::Ggx { albedo: Color::new(0.9, 0.9, 0.9), alpha: 0.3 };
        let (ray, hit) = floor_hit();
        let mut rng = Rng::new(7);
        let n = hit.n;
        let wo = (-ray.d).norm();
        for _ in 0..1000 {
            if let Some(s) = mat.sample(&ray, &hit, &mut rng) {
                let (_, pdf_eval) = mat.eval(wo, s.scattered.d, n);
                assert!((s.pdf - pdf_eval).abs() < 1e-9, "{} vs {}", s.pdf, pdf_eval);
            }
        }
    }

    /// GGX の重みは f·cos/pdf と一致する（sample の weight と eval の f・pdf が一貫）。
    /// VNDF サンプリングなのに NDF PDF で割っていた不整合への回帰テスト。
    #[test]
    fn ggx_weight_matches_f_cos_over_pdf() {
        let mat = Material::Ggx { albedo: Color::new(0.9, 0.8, 0.7), alpha: 0.35 };
        let (ray, hit) = floor_hit();
        let mut rng = Rng::new(11);
        let n = hit.n;
        let wo = (-ray.d).norm();
        for _ in 0..2000 {
            if let Some(s) = mat.sample(&ray, &hit, &mut rng) {
                let wi = s.scattered.d;
                let (f, pdf) = mat.eval(wo, wi, n);
                let cos_o = n.dot(wi).max(0.0);
                let expected = f * (cos_o / pdf.max(1e-6));
                assert!((s.weight.r() - expected.r()).abs() < 1e-9, "{} vs {}", s.weight.r(), expected.r());
                assert!((s.weight.g() - expected.g()).abs() < 1e-9);
                assert!((s.weight.b() - expected.b()).abs() < 1e-9);
            }
        }
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
}
