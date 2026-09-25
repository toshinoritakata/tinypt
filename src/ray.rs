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
    /// シャッター区間（レイの `time` の範囲）を設定する。既定は [0, 1]。`open == close` は時刻固定（ブラー無し）。
    pub fn set_shutter(&mut self, open: f64, close: f64) {
        self.shutter_open = open;
        self.shutter_close = close;
    }
    /// シャッター区間 `(open, close)`。
    pub fn shutter(&self) -> (f64, f64) {
        (self.shutter_open, self.shutter_close)
    }
    /// 正規化スクリーン座標 [-1, 1] からカメラレイを生成する。
    ///
    /// DOF 有効時: レンズ上のランダムな点から焦点面上の点へレイを飛ばす。焦点面は視線（−w）に
    /// 垂直で、レンズ中心から距離 `focus_dist` にある**平面**（薄レンズモデル。Mitsuba / PBRT と同じ）。
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
        let dir = if self.lens_radius > 0.0 {
            // ピンホール方向が焦点平面 (p − o)·(−w) = focus_dist と交わる点に向けてレイを飛ばす
            // （画面の端ほど焦点までの距離が 1/cos 倍になる。以前は距離 focus_dist の球面だった）
            let focus_point = self.o + dir_base * (self.focus_dist / dir_base.dot(-self.w));
            (focus_point - origin).norm()
        } else {
            dir_base
        };
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

    /// レイが焦点平面（視線に垂直、レンズ中心から focus_dist）と交わる点。
    fn hit_focus_plane(r: Ray, eye: Vec3, forward: Vec3, focus_dist: f64) -> Vec3 {
        let t = (focus_dist - (r.o - eye).dot(forward)) / r.d.dot(forward);
        r.at(t)
    }

    /// 同じ画素のレイは、レンズ上のどこから出ても焦点**平面**上の同じ点に収束する（画面の端でも）。
    /// その点はピンホールレイと焦点平面の交点。旧実装は距離 focus_dist の球面に収束していたため、
    /// 画面の端では平面上の交点がレンズ位置によってばらついた。
    #[test]
    fn dof_rays_converge_on_the_focus_plane() {
        let eye = Vec3::new(0.3, 1.2, 4.0);
        let target = Vec3::new(0.0, 0.5, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let focus = 3.0;
        let cam = Camera::look_at_dof(eye, target, up, 60.0, 16.0 / 9.0, focus, 0.4);
        let pinhole = Camera::look_at(eye, target, up, 60.0, 16.0 / 9.0);
        let forward = (target - eye).norm();
        let mut rng = Rng::new(9);
        for &(sx, sy) in &[(0.0, 0.0), (0.9, 0.0), (-0.95, 0.8), (0.7, -0.99)] {
            let reference = hit_focus_plane(pinhole.ray(sx, sy, &mut rng), eye, forward, focus);
            for _ in 0..64 {
                let r = cam.ray(sx, sy, &mut rng);
                let p = hit_focus_plane(r, eye, forward, focus);
                assert!((p - reference).len() < 1e-9, "screen ({}, {}): focus point {:?} vs {:?}", sx, sy, (p.x, p.y, p.z), (reference.x, reference.y, reference.z));
            }
        }
    }

    /// レンズ上の原点はレンズ円板（u, v 平面、半径 aperture/2）内に一様に分布し、ピンホール
    /// （aperture 0）のレイは従来どおり eye から画素方向へ出る。
    #[test]
    fn lens_origins_lie_on_the_lens_disk_and_pinhole_is_unchanged() {
        let eye = Vec3::new(0.0, 0.0, 1.0);
        let target = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let cam = Camera::look_at_dof(eye, target, up, 45.0, 1.0, 2.0, 0.5);
        let mut rng = Rng::new(4);
        for _ in 0..1000 {
            let r = cam.ray(0.3, -0.2, &mut rng);
            let off = r.o - eye;
            assert!(off.z.abs() < 1e-12 && off.len() <= 0.25 + 1e-12, "lens origin off the disk: {:?}", (off.x, off.y, off.z));
        }
        let pin = Camera::look_at(eye, target, up, 45.0, 1.0);
        let r = pin.ray(0.3, -0.2, &mut rng);
        assert_eq!((r.o.x, r.o.y, r.o.z), (eye.x, eye.y, eye.z));
        let half = (22.5f64).to_radians().tan();
        let expect = Vec3::new(0.3 * half, -0.2 * half, -1.0).norm();
        assert!((r.d - expect).len() < 1e-12);
    }
}
