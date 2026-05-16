//! レイとカメラモデル。
//!
//! カメラは eye → target の look-at 方式で構築される。
//! 被写界深度（DOF）はレンズ上のランダムサンプリングで実現し、
//! モーションブラーはシャッター開閉時間内のランダムな time を割り当てる。

use crate::math::Vec3;
use crate::rng::Rng;

#[derive(Clone, Copy)]
/// レイ: 原点 (o) + 方向 (d) + シャッター時間 (time)。
pub struct Ray { pub o: Vec3, pub d: Vec3, pub time: f64 }
impl Ray {
    /// レイ上のパラメータ `t` での点を返す: P(t) = o + d * t
    pub fn at(self, t: f64) -> Vec3 { self.o + self.d * t }
}

/// ピンホール / 薄レンズ（DOF）カメラ。
///
/// 座標系: u=右, v=上, w=後ろ（カメラの視線方向は -w）。
/// `lens_radius > 0` のとき薄レンズモデルで被写界深度を表現する。
pub struct Camera {
    o: Vec3,          // カメラ位置（eye）
    u: Vec3,          // 右方向の基底ベクトル
    v: Vec3,          // 上方向の基底ベクトル
    w: Vec3,          // 後方向の基底ベクトル（視線は -w）
    half_h: f64,      // 垂直方向の半画角の tan 値
    half_w: f64,      // 水平方向の半画角の tan 値
    lens_radius: f64, // レンズ半径（0 でピンホール）
    focus_dist: f64,  // 焦点距離
    shutter_open: f64,  // シャッター開時間
    shutter_close: f64, // シャッター閉時間
}
impl Camera {
    /// `target` を注視するピンホールカメラを構築する。
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, fov_deg: f64, aspect: f64) -> Self {
        let theta = fov_deg.to_radians();
        let half_h = (theta * 0.5).tan();
        let half_w = aspect * half_h;
        let w = (eye - target).norm();
        let u = up.cross(w).norm();
        let v = w.cross(u);
        let focus_dist = (eye - target).len();
        Self {
            o: eye,
            u,
            v,
            w,
            half_h,
            half_w,
            lens_radius: 0.0,
            focus_dist,
            shutter_open: 0.0,
            shutter_close: 1.0,
        }
    }
    /// 被写界深度（DOF）付きカメラを構築する。
    /// `aperture` が大きいほどボケが強くなり、`focus_dist` の位置にピントが合う。
    pub fn look_at_dof(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_deg: f64,
        aspect: f64,
        focus_dist: f64,
        aperture: f64,
    ) -> Self {
        let theta = fov_deg.to_radians();
        let half_h = (theta * 0.5).tan();
        let half_w = aspect * half_h;
        let w = (eye - target).norm();
        let u = up.cross(w).norm();
        let v = w.cross(u);
        Self {
            o: eye,
            u,
            v,
            w,
            half_h,
            half_w,
            lens_radius: 0.5 * aperture.max(0.0),
            focus_dist: focus_dist.max(1e-6),
            shutter_open: 0.0,
            shutter_close: 1.0,
        }
    }
    /// 正規化スクリーン座標 [-1, 1] からカメラレイを生成する。
    ///
    /// DOF 有効時: レンズ上のランダムな点から焦点面上の点へレイを飛ばす。
    /// モーションブラー: シャッター開閉の間のランダムな time を割り当てる。
    pub fn ray(&self, sx: f64, sy: f64, rng: &mut Rng) -> Ray {
        // ピンホール方向（レンズ中心から見たスクリーン上の方向）
        let dir_base = (-self.w + sx*self.half_w*self.u + sy*self.half_h*self.v).norm();
        // 薄レンズモデル: レンズ上のランダムな点をサンプリング
        let lens_offset = if self.lens_radius > 0.0 {
            let (dx, dy) = sample_unit_disk(rng);
            self.u * (dx * self.lens_radius) + self.v * (dy * self.lens_radius)
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        };
        let origin = self.o + lens_offset;
        // 焦点面上の点に向けてレイを飛ばす（ピント面で全レイが収束）
        let focus_point = self.o + dir_base * self.focus_dist;
        let dir = (focus_point - origin).norm();
        // モーションブラー: シャッター間のランダムな時間を割り当て
        let time = self.shutter_open + (self.shutter_close - self.shutter_open) * rng.next_f64();
        Ray { o: origin, d: dir, time }
    }
}

/// 単位円内の一様ランダム点を棄却法でサンプリング（レンズ面用）。
fn sample_unit_disk(rng: &mut Rng) -> (f64, f64) {
    loop {
        let x = 2.0 * rng.next_f64() - 1.0;
        let y = 2.0 * rng.next_f64() - 1.0;
        if x * x + y * y < 1.0 {
            return (x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dof_camera_varies_origin() {
        let eye = Vec3::new(0.0, 0.0, 1.0);
        let target = Vec3::new(0.0, 0.0, 0.0);
        let cam = Camera::look_at_dof(eye, target, Vec3::new(0.0, 1.0, 0.0), 45.0, 1.0, 1.0, 0.5);

        let mut rng = Rng::new(123);
        let r0 = cam.ray(0.0, 0.0, &mut rng);
        let r1 = cam.ray(0.0, 0.0, &mut rng);

        let delta = (r0.o - r1.o).len();
        assert!(delta > 0.0);
    }
}
