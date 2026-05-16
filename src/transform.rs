//! 平行移動・Y 軸回転・均一スケールの簡易トランスフォーム。
//!
//! フル 4x4 行列の代わりに TRS（Translate-Rotate-Scale）分解で保持し、
//! 逆変換を明示的に計算する。Y 軸回転のみのため sin/cos を事前計算。

use crate::math::Vec3;

/// Y 軸回りの回転を sin/cos で適用する。
fn rotate_y_sincos(p: Vec3, s: f64, c: f64) -> Vec3 {
    Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)
}

#[derive(Clone, Copy, Debug)]
/// 平行移動 + Y 軸回転 + 均一スケールのトランスフォーム。
pub struct Transform {
    /// 平行移動ベクトル
    pub t: Vec3,
    /// Y 軸回転角（ラジアン）
    pub rot_y: f64,
    /// 均一スケール係数
    pub s: f64,
    /// 事前計算した sin(rot_y)
    sin_y: f64,
    /// 事前計算した cos(rot_y)
    cos_y: f64,
    /// スケールの逆数 (1/s)
    inv_s: f64,
}

impl Transform {
    /// 平行移動・Y 軸回転（度）・スケールからトランスフォームを生成する。
    pub fn new(t: Vec3, rot_y_deg: f64, s: f64) -> Self {
        let rot_y = rot_y_deg.to_radians();
        let (sin_y, cos_y) = rot_y.sin_cos();
        let inv_s = if s.abs() < 1e-30 { 1.0 } else { 1.0 / s };
        Self { t, rot_y, s, sin_y, cos_y, inv_s }
    }

    /// オブジェクト空間の点をワールド空間に変換する: p' = R(p × s) + t
    pub fn apply_point(self, p: Vec3) -> Vec3 {
        let p = p * self.s;
        let p = rotate_y_sincos(p, self.sin_y, self.cos_y);
        p + self.t
    }


    /// ワールド空間の点をオブジェクト空間に逆変換する: p_obj = R⁻¹(p_world - t) / s
    pub fn apply_point_inv(self, p_world: Vec3) -> Vec3 {
        let p = p_world - self.t;
        let p = rotate_y_sincos(p, -self.sin_y, self.cos_y);
        p * self.inv_s
    }

    /// ワールド空間のベクトルをオブジェクト空間に逆変換する: v_obj = R⁻¹(v_world) / s
    pub fn apply_vec_inv(self, v_world: Vec3) -> Vec3 {
        let v = rotate_y_sincos(v_world, -self.sin_y, self.cos_y);
        v * self.inv_s
    }

    /// オブジェクト空間の法線をワールド空間に変換する。
    /// 均一スケールのため回転のみで正しい（正規化して返す）。
    pub fn apply_normal(self, n_obj: Vec3) -> Vec3 {
        rotate_y_sincos(n_obj, self.sin_y, self.cos_y).norm()
    }
}
