//! レイとカメラモデル。
//!
//! カメラは eye → target の look-at 方式で構築される。
//! 被写界深度（DOF）はレンズ上のランダムサンプリングで実現し、
//! モーションブラーはシャッター開閉時間内のランダムな time を割り当てる。

use crate::math::{Mat3, Vec3};
use crate::rng::Rng;
use crate::transform::{AnimatedTransform, Transform};

/// カメラ姿勢（位置と正規直交基底）。
type Pose = (Vec3, Vec3, Vec3, Vec3);

/// 動くカメラ: 姿勢の補間と、両端の姿勢（端点はそのまま返して厳密に一致させる）。
#[derive(Clone, Copy)]
struct CameraMotion {
    anim: AnimatedTransform,
    end: Pose,
}

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
    /// シャッター閉じ時点の姿勢への補間（`to_world_end`）。`None` は静止カメラ（従来と完全に同じ経路）
    motion: Option<CameraMotion>,
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
            motion: None,
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
            motion: None,
        }
    }
    /// シャッター区間（レイの `time` の範囲）を設定する。既定は [0, 1]。`open == close` は時刻固定（ブラー無し）。
    pub fn set_shutter(&mut self, open: f64, close: f64) {
        self.shutter_open = open;
        self.shutter_close = close;
    }
    /// シャッター閉じ時点の姿勢（`lookat` 3 点）を与え、`time` で姿勢を補間するカメラにする（独自拡張 `to_world_end`）。
    ///
    /// 姿勢は**ワールド → カメラ**のアフィン変換 `p_cam = A·p + t`（`A` の行が `u, v, w`、`t = −A·o`）で表す。
    /// `u, v, w` は正規直交なので極分解の回転部がそのまま姿勢で、伸縮部は単位行列に収束する。
    /// 補間は [`AnimatedTransform`]（`A` は四元数 slerp、`t` は線形）。ワールド → カメラで持つので、
    /// 対象のまわりを旋回するカット（カメラ座標での `t` が一定）ではカメラ位置 `o = −Aᵀ·t` が**弧**を描く
    /// （カメラ → ワールドで持つと位置が弦を線形に横切る）。
    /// `fov` / `focus_dist` / `lens_radius` は固定。基底が作れない（`up` と視線が平行など）・補間できないときは
    /// 何も変えず `false`（呼び出し側が警告して静止のままにする）。
    pub fn set_end_pose(&mut self, eye: Vec3, target: Vec3, up: Vec3) -> bool {
        let w = (eye - target).norm();
        let u = up.cross(w).norm();
        let v = w.cross(u);
        let start = pose_transform((self.o, self.u, self.v, self.w));
        let end = (eye, u, v, w);
        match AnimatedTransform::new(start, pose_transform(end)) {
            Some(anim) => {
                self.motion = Some(CameraMotion { anim, end });
                true
            }
            None => false,
        }
    }
    /// 時刻 `time` の姿勢。静止カメラは常に同じ。両端（`≤ 0` / `≥ 1`）は与えた姿勢そのもの。
    #[inline]
    fn pose_at(&self, time: f64) -> Pose {
        match &self.motion {
            None => (self.o, self.u, self.v, self.w),
            Some(_) if time <= 0.0 => (self.o, self.u, self.v, self.w),
            Some(m) if time >= 1.0 => m.end,
            Some(m) => {
                let x = m.anim.at(time);
                let a = x.linear().m;
                let t = x.translation();
                // 行が u, v, w。o = −Aᵀ·t
                let row = |i: usize| Vec3::new(a[i][0], a[i][1], a[i][2]);
                let (u, v, w) = (row(0), row(1), row(2));
                (-(u * t.x + v * t.y + w * t.z), u, v, w)
            }
        }
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
        // 薄レンズモデル: レンズ上のランダムな点をサンプリング（乱数の消費順は「レンズ → time」で従来通り。
        // 姿勢が time で決まるので、レンズ点は (dx, dy) だけ先に引き、基底に掛けるのは time を引いた後）
        rng.set_dim(crate::sampler::first::LENS);
        let lens_xy = if self.lens_radius > 0.0 { Some(sample_unit_disk(rng)) } else { None };
        rng.set_dim(crate::sampler::first::TIME);
        // モーションブラー: シャッター間のランダムな時間を割り当て
        let time = self.shutter_open + (self.shutter_close - self.shutter_open) * rng.next_f64();
        // 動くカメラは time の姿勢の基底を使う。**レンズ点も補間後の u, v に掛ける**（固定基底だとボケが動きとずれる）
        let (o, u, v, w) = self.pose_at(time);
        // ピンホール方向（レンズ中心から見たスクリーン上の方向）
        let dir_base = (-w + sx*self.half_w*u + sy*self.half_h*v).norm();
        let lens_offset = match lens_xy {
            Some((dx, dy)) => u * (dx * self.lens_radius) + v * (dy * self.lens_radius),
            None => Vec3::new(0.0, 0.0, 0.0),
        };
        let origin = o + lens_offset;
        let dir = if self.lens_radius > 0.0 {
            // ピンホール方向が焦点平面 (p − o)·(−w) = focus_dist と交わる点に向けてレイを飛ばす
            // （画面の端ほど焦点までの距離が 1/cos 倍になる。以前は距離 focus_dist の球面だった）
            let focus_point = o + dir_base * (self.focus_dist / dir_base.dot(-w));
            (focus_point - origin).norm()
        } else {
            dir_base
        };
        Ray { o: origin, d: dir, time }
    }
}

/// カメラ姿勢を、ワールド → カメラのアフィン変換にする: 行が `(u, v, w)`、平行移動が `−A·o`。
fn pose_transform((o, u, v, w): Pose) -> Transform {
    let a = Mat3::from_rows([[u.x, u.y, u.z], [v.x, v.y, v.z], [w.x, w.y, w.z]]);
    Transform::from_affine(a, -a.mul_vec(o))
}

/// 単位円内の一様ランダム点を Shirley–Chiu の同心円写像でサンプリングする（レンズ面用）。
/// 乱数をちょうど 2 個使い（棄却法と違って消費量が一定）、[0,1)² の層化がそのまま円盤の層化になる。
fn sample_unit_disk(rng: &mut Rng) -> (f64, f64) {
    let a = 2.0 * rng.next_f64() - 1.0;
    let b = 2.0 * rng.next_f64() - 1.0;
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, phi) = if a * a > b * b {
        (a, std::f64::consts::FRAC_PI_4 * (b / a))
    } else {
        (b, std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (a / b))
    };
    (r * phi.cos(), r * phi.sin())
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

#[cfg(test)]
mod camera_motion_tests {
    use super::*;

    const UP: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    fn v(x: f64, y: f64, z: f64) -> Vec3 { Vec3::new(x, y, z) }
    fn cam(eye: Vec3, lens: f64) -> Camera {
        Camera::look_at_dof(eye, v(0.0, 0.0, 0.0), UP, 40.0, 16.0 / 9.0, 4.0, 2.0 * lens)
    }
    /// 開 `eye0` → 閉 `eye1`（どちらも原点を注視）の動くカメラ。
    fn moving(eye0: Vec3, eye1: Vec3, lens: f64) -> Camera {
        let mut c = cam(eye0, lens);
        assert!(c.set_end_pose(eye1, v(0.0, 0.0, 0.0), UP));
        c
    }
    fn same_ray(a: Ray, b: Ray) -> bool {
        (a.o.x, a.o.y, a.o.z, a.d.x, a.d.y, a.d.z, a.time) == (b.o.x, b.o.y, b.o.z, b.d.x, b.d.y, b.d.z, b.time)
    }
    fn rays(c: &Camera, seed: u64) -> Vec<Ray> {
        let mut rng = Rng::new(seed);
        (0..50).map(|i| c.ray(-0.9 + 0.036 * i as f64, 0.5 - 0.02 * i as f64, &mut rng)).collect()
    }

    /// time = 0 / 1 のレイは、開・閉それぞれの静止カメラのレイとビット一致（レンズ有り・同じシード）。
    #[test]
    fn endpoints_match_static_cameras_exactly() {
        let (e0, e1) = (v(4.0, 1.0, 0.0), v(0.5, 2.0, -3.5));
        for (t, eye) in [(0.0, e0), (1.0, e1)] {
            let mut m = moving(e0, e1, 0.2);
            m.set_shutter(t, t);
            let mut s = cam(eye, 0.2);
            s.set_shutter(t, t);
            for (a, b) in rays(&m, 7).into_iter().zip(rays(&s, 7)) {
                assert!(same_ray(a, b), "time {t}");
            }
        }
    }

    /// `to_world_end` の無いカメラは、シャッター区間を持つ従来のカメラとビット一致（同じ乱数の消費）。
    #[test]
    fn static_camera_unchanged_by_the_motion_path() {
        // 開 = 閉のアニメ付きでも、静止と同じレイ（time は同じ乱数から）になる
        let e = v(4.0, 1.0, 0.0);
        let a = cam(e, 0.2);
        let mut b = cam(e, 0.2);
        assert!(b.set_end_pose(e, v(0.0, 0.0, 0.0), UP));
        for (x, y) in rays(&a, 3).into_iter().zip(rays(&b, 3)) {
            assert_eq!(x.time.to_bits(), y.time.to_bits());
            assert!((x.o - y.o).len() < 1e-12 && (x.d - y.d).len() < 1e-12);
        }
        // 乱数の消費: 静止でも動くでもレイ 1 本の後の乱数列は同じ
        let (mut r1, mut r2) = (Rng::new(1), Rng::new(1));
        let _ = a.ray(0.1, 0.1, &mut r1);
        let _ = moving(e, v(0.0, 3.0, 4.0), 0.2).ray(0.1, 0.1, &mut r2);
        assert_eq!(r1.next_f64().to_bits(), r2.next_f64().to_bits());
    }

    /// 90° 旋回の time = 0.5 で、カメラ位置は弧の中点（半径が縮まない）。伸縮部は単位行列で基底は正規直交。
    #[test]
    fn orbit_midpoint_is_on_the_arc() {
        let r = 4.0;
        let (e0, e1) = (v(r, 1.0, 0.0), v(0.0, 1.0, -r));
        let mut c = moving(e0, e1, 0.0);
        c.set_shutter(0.5, 0.5);
        let ray = c.ray(0.0, 0.0, &mut Rng::new(0));
        let a = std::f64::consts::FRAC_PI_4;
        let arc_mid = v(r * a.cos(), 1.0, -r * a.sin());
        assert!((ray.o - arc_mid).len() < 1e-9, "arc {:?} vs {:?}", ray.o, arc_mid);
        let chord_mid = (e0 + e1) * 0.5;
        assert!((ray.o - chord_mid).len() > 0.5, "弦の中点ではない");
        let (_, u, vv, w) = c.pose_at(0.5);
        for (a, b, want) in [(u, u, 1.0), (vv, vv, 1.0), (w, w, 1.0), (u, vv, 0.0), (u, w, 0.0), (vv, w, 0.0)] {
            assert!((a.dot(b) - want).abs() < 1e-12, "正規直交");
        }
        // 姿勢は原点を向いたまま（中間でも注視点がずれない）
        assert!((ray.d - (-ray.o).norm()).len() < 1e-9);
    }

    /// レンズ点は補間後の基底の面にある: time = 1 のレイ原点 − 閉の位置 は閉の (u, v) が張る面上で、半径以内。
    #[test]
    fn lens_lies_in_the_interpolated_plane() {
        let (e0, e1) = (v(4.0, 1.0, 0.0), v(0.0, 1.0, -4.0));
        let mut c = moving(e0, e1, 0.3);
        c.set_shutter(1.0, 1.0);
        let end = cam(e1, 0.3);
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let r = c.ray(0.2, -0.1, &mut rng);
            let off = r.o - e1;
            assert!(off.dot(end.w).abs() < 1e-12, "閉じ姿勢のレンズ面上にない");
            assert!(off.len() <= 0.3 + 1e-12);
        }
        // 中間時刻でも補間後の w に垂直
        c.set_shutter(0.5, 0.5);
        let (o, _, _, w) = c.pose_at(0.5);
        for _ in 0..50 {
            let r = c.ray(0.0, 0.0, &mut rng);
            assert!((r.o - o).dot(w).abs() < 1e-12);
        }
    }

    /// 退化: 開 = 閉、180° 反対側、`up` と視線が平行。
    #[test]
    fn degenerate_poses() {
        let e = v(4.0, 1.0, 0.0);
        let mut same = moving(e, e, 0.0);
        same.set_shutter(0.3, 0.3);
        let s = cam(e, 0.0).ray(0.1, 0.1, &mut Rng::new(0));
        let m = same.ray(0.1, 0.1, &mut Rng::new(0));
        assert!((s.o - m.o).len() < 1e-12 && (s.d - m.d).len() < 1e-12);
        // 180°: 位置は原点まわりの半周（半径 = 4 の xz）の途中、姿勢は有限で正規直交
        let mut half = moving(v(4.0, 0.0, 0.0), v(-4.0, 0.0, 0.0), 0.0);
        half.set_shutter(0.5, 0.5);
        let (o, u, _, w) = half.pose_at(0.5);
        assert!(o.x.is_finite() && u.len() > 0.999 && w.len() > 0.999);
        assert!((o.len() - 4.0).abs() < 1e-9, "半径が縮まない: {}", o.len());
        // up ∥ 視線: 基底が作れず false（静止のまま）
        let mut c = cam(e, 0.0);
        assert!(!c.set_end_pose(v(0.0, 4.0, 0.0), v(0.0, 0.0, 0.0), UP));
        assert!(c.motion.is_none());
        // 上下にほぼ平行（ほぼ真上から）でも有限に補間できる
        let mut steep = moving(e, v(0.01, 4.0, 0.0), 0.0);
        steep.set_shutter(0.5, 0.5);
        let r = steep.ray(0.0, 0.0, &mut Rng::new(0));
        assert!(r.o.x.is_finite() && r.d.x.is_finite());
    }
}
