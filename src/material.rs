//! マテリアルモデルと BSDF サンプリング。
//!
//! 各マテリアルは `scatter()` メソッドでレイの散乱方向と減衰色を返す。
//!
//! ## 対応マテリアル
//! - **Lambert**: 完全拡散反射（コサイン重み付き半球サンプリング）
//! - **Metal**: 完全鏡面反射（デルタ BSDF）
//! - **Dielectric**: 屈折体（フレネル + Beer-Lambert 吸収）
//! - **GGX**: マイクロファセットモデル（VNDF サンプリング + Smith 遮蔽関数）
//! - **Subsurface**: 簡易サブサーフェス散乱（指数分布による散乱距離）
//! - **DiffuseLight**: 拡散発光体（散乱なし、放射輝度を返す）

use crate::constants::RAY_EPSILON;
use crate::geometry::Hit;
use crate::math::{reflect, refract, Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;

#[derive(Clone, Copy)]
/// 積分器が対応するマテリアルモデル。
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

impl Material {
    /// 散乱レイと減衰色をサンプリングする。
    /// 発光体 (`DiffuseLight`) の場合は `None` を返す（パス終了）。
    pub fn scatter(&self, ray_in: &Ray, hit: &Hit, rng: &mut Rng) -> Option<(Ray, Color)> {
        // 法線をレイの進行方向に対して正しい向きに修正
        let entering = hit.n.dot(ray_in.d) < 0.0; // レイが表面に入射するか
        let n = if entering { hit.n } else { -hit.n };

        match self {
            Material::Lambert { albedo } => {
                let d = sample_cosine_hemisphere(n, rng);
                Some((Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time }, *albedo))
            }
            Material::Metal { albedo } => {
                let d = reflect(ray_in.d, n);
                Some((Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time }, *albedo))
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
                if choose_refl {
                    current_beta = current_beta * (fresnel / fresnel.max(1e-6));
                    Some((Ray { o: hit.p + RAY_EPSILON * refl_dir, d: refl_dir, time: ray_in.time }, current_beta))
                } else {
                    let tdir = refr_dir.unwrap();
                    // Correct radiance scaling for transmission
                    let scale = (1.0 - fresnel) / (1.0 - fresnel).max(1e-6);
                    current_beta = current_beta * scale * (eta * eta);
                    Some((Ray { o: hit.p + RAY_EPSILON * tdir, d: tdir, time: ray_in.time }, current_beta))
                }
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
                let spec = f * (d_ggx * g / denom);

                let pdf_h = d_ggx * cos_h;
                let pdf = (pdf_h / (4.0 * d.dot(m).abs().max(1e-6))).max(1e-6);

                let beta = spec * (cos_o / pdf);
                Some((Ray { o: hit.p + RAY_EPSILON * d, d, time: ray_in.time }, beta))
            }
            Material::Subsurface { albedo, scatter_dist } => {
                let scale = scatter_dist.max(1e-4);
                let u = rng.next_f64().max(1e-12);
                let dist = -u.ln() * scale;

                let normal_out = hit.n;
                let d = sample_cosine_hemisphere(normal_out, rng);

                let att = *albedo * (-dist / scale).exp();
                let origin = hit.p - normal_out * dist + RAY_EPSILON * d;
                Some((Ray { o: origin, d, time: ray_in.time }, att))
            }
            Material::DiffuseLight { emit: _ } => None,
        }
    }
}

/// GGX BRDF の評価: f(ω_i, ω_o) = F(θ_h) · D(θ_h) · G(θ_i, θ_o) / (4 cos θ_i cos θ_o)
pub fn ggx_eval(albedo: Color, alpha: f64, n: Vec3, wo: Vec3, wi: Vec3) -> Color {
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
pub fn ggx_pdf(alpha: f64, n: Vec3, wo: Vec3, wi: Vec3) -> f64 {
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
