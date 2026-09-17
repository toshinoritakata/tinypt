//! 幾何プリミティブと軸平行バウンディングボックス（AABB）。
//!
//! レンダラーの基本的な幾何形状を定義する:
//! - `Hit`: レイとサーフェスの交差情報
//! - `Sphere`: 球プリミティブ（解析的なレイ交差判定）
//! - `Triangle`: モーションブラー対応三角形（Möller–Trumbore 法）
//! - `Aabb`: 軸平行バウンディングボックス（スラブ法によるレイ交差判定）

use crate::math::{gamma, Vec3};
use crate::ray::Ray;

#[derive(Clone, Copy)]
/// レイとサーフェスの交差情報。
pub struct Hit {
    /// レイのパラメータ t（交差距離）
    pub t: f64,
    /// 交差点のワールド座標
    pub p: Vec3,
    /// 交差点の法線ベクトル
    pub n: Vec3,
    /// 交差したマテリアルのインデックス
    pub mat_id: usize,
    /// ヒットしたプリミティブのインデックス（球: World::spheres 上の位置、
    /// 三角形: メッシュ内の三角形インデックス）。呼び出し側（World::hit）が設定する。
    pub prim_id: usize,
    /// メッシュインスタンス経由のヒットなら Some(inst_id)、球なら None。
    /// 呼び出し側（World::hit）が設定する。
    pub inst_id: Option<usize>,
    /// 交差点 `p` の成分ごとの浮動小数点誤差の上界（ワールド空間）。真の交差点は
    /// `p ± p_error` の箱の中にある。新しいレイを出すときは [`offset_ray_origin`] でこの箱の外へ
    /// 幾何法線方向にずらす（シーンの大きさや絶対位置に依存しない自己交差回避）。
    pub p_error: Vec3,
}

/// PBRT v4 の `OffsetRayOrigin`: 交差点 `p`（誤差上界 `p_err`、幾何法線 `n`）から方向 `w` へ出すレイの原点。
///
/// 誤差の箱 `p ± p_err` を法線方向に抜けるだけの距離 `d = |n|·p_err` だけ、`w` の側（法線と同じ側なら +n、
/// 反対側なら −n）にずらし、さらに各成分をずらす向きに 1 ulp 進める（ずらした点自体の丸めで箱に
/// 戻らないため）。こうして出したレイは、元の面との交点の t が真に負になり、交差判定の誤差を考慮した
/// 「t > 誤差上界」の条件で必ず棄却される。
pub fn offset_ray_origin(p: Vec3, p_err: Vec3, n: Vec3, w: Vec3) -> Vec3 {
    let d = n.abs().dot(p_err);
    let mut offset = n * d;
    if w.dot(n) < 0.0 {
        offset = -offset;
    }
    let po = p + offset;
    // ずらす向きに少なくとも 1 ulp 進める（|v|·ε は v の ulp 以上、MIN_POSITIVE は v = 0 のとき用）。
    // next_up / next_down より安価で、どの成分も箱の外へ確実に出る
    let bump = |v: f64, o: f64| v + o.signum() * (o != 0.0) as u8 as f64 * (v.abs() * f64::EPSILON + f64::MIN_POSITIVE);
    Vec3::new(bump(po.x, offset.x), bump(po.y, offset.y), bump(po.z, offset.z))
}

#[derive(Clone, Copy)]
/// 球プリミティブ。
pub struct Sphere {
    /// 中心座標
    pub c: Vec3,
    /// 半径
    pub r: f64,
    /// マテリアルインデックス
    pub mat_id: usize,
}

#[derive(Clone, Copy)]
/// モーションブラー対応の三角形。
///
/// シャッター開（_0）と閉（_1）の頂点を持ち、レイの time で線形補間する。
/// 静的メッシュでは open = close の同一頂点を設定する。
pub struct Triangle {
    pub v0_0: Vec3, pub v1_0: Vec3, pub v2_0: Vec3, // シャッター開の頂点
    pub v0_1: Vec3, pub v1_1: Vec3, pub v2_1: Vec3, // シャッター閉の頂点
    pub e1_0: Vec3, pub e2_0: Vec3, // シャッター開の事前計算エッジ
    pub e1_1: Vec3, pub e2_1: Vec3, // シャッター閉の事前計算エッジ
    pub mat_id: usize,
}

#[derive(Clone, Copy, Debug)]
/// 軸平行バウンディングボックス（AABB）。
///
/// BVH ノードの境界として使用される。スラブ法（Slab Test）で
/// レイとの交差を高速に判定する。
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

/// スラブ法の遠い側の t に掛ける係数 1 + 2γ(3)（PBRT と同じ）。丸めで区間が縮んでレイが箱をすり抜けないようにする。
/// 区間が空かどうかは `tmax < tmin`（等号を含まない）で判定し、厚さ 0 の平面の箱も通す。
const SLAB_FAR_SCALE: f64 = 1.0 + 2.0 * f64::EPSILON * 0.5 * 3.0 / (1.0 - f64::EPSILON * 0.5 * 3.0);

impl Aabb {
    /// 空の AABB を返す（min=+∞, max=-∞ で初期化）。
    /// `grow()` や `union()` で拡張して使用する。
    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(-f64::INFINITY, -f64::INFINITY, -f64::INFINITY),
        }
    }

    /// 3 点を囲む AABB を生成する（数値安定性のため微小量だけ拡張）。
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let mut bb = Self::empty();
        bb = bb.grow(a);
        bb = bb.grow(b);
        bb = bb.grow(c);
        // 数値的な頑健性のため、座標の大きさに比例してわずかに広げる（絶対値の余白は、原点から遠い
        // 配置では f64 の刻みより小さくなり、厚さ 0 の平面の箱が広がらない）
        let pad = |lo: f64, hi: f64| gamma(3) * lo.abs().max(hi.abs());
        let (px, py, pz) = (pad(bb.min.x, bb.max.x), pad(bb.min.y, bb.max.y), pad(bb.min.z, bb.max.z));
        bb.min = Vec3::new(bb.min.x - px, bb.min.y - py, bb.min.z - pz);
        bb.max = Vec3::new(bb.max.x + px, bb.max.y + py, bb.max.z + pz);
        bb
    }

    /// 点 `p` を含むように AABB を拡張する。
    pub fn grow(mut self, p: Vec3) -> Self {
        self.min = Vec3::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z));
        self.max = Vec3::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z));
        self
    }

    /// 2 つの AABB の和を返す。
    pub fn union(self, b: Aabb) -> Aabb {
        Aabb {
            min: Vec3::new(self.min.x.min(b.min.x), self.min.y.min(b.min.y), self.min.z.min(b.min.z)),
            max: Vec3::new(self.max.x.max(b.max.x), self.max.y.max(b.max.y), self.max.z.max(b.max.z)),
        }
    }

    /// AABB の中心座標を返す。
    pub fn centroid(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// AABB の対角線ベクトル（各軸の幅）を返す。
    pub fn extent(self) -> Vec3 {
        self.max - self.min
    }

    /// レイと AABB の交差判定（逆方向ベクトルを内部で計算）。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> bool {
        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        self.hit_inv(r, inv, tmin, tmax)
    }

    /// 事前計算した逆方向ベクトルを使ったレイ-AABB 交差判定（スラブ法）。
    ///
    /// 各軸のスラブ（2 つの平行面の間）とレイの交差区間を求め、
    /// 全軸の交差区間が重なるかを判定する。min/max で符号分岐を回避。
    pub fn hit_inv(&self, r: Ray, inv: Vec3, mut tmin: f64, mut tmax: f64) -> bool {
        let t0x = (self.min.x - r.o.x) * inv.x;
        let t1x = (self.max.x - r.o.x) * inv.x;
        tmin = tmin.max(t0x.min(t1x));
        tmax = tmax.min(t0x.max(t1x) * SLAB_FAR_SCALE);
        if tmax < tmin { return false; }

        let t0y = (self.min.y - r.o.y) * inv.y;
        let t1y = (self.max.y - r.o.y) * inv.y;
        tmin = tmin.max(t0y.min(t1y));
        tmax = tmax.min(t0y.max(t1y) * SLAB_FAR_SCALE);
        if tmax < tmin { return false; }

        let t0z = (self.min.z - r.o.z) * inv.z;
        let t1z = (self.max.z - r.o.z) * inv.z;
        tmin = tmin.max(t0z.min(t1z));
        tmax = tmax.min(t0z.max(t1z) * SLAB_FAR_SCALE);
        if tmax < tmin { return false; }
        true
    }

    /// レイと AABB の交差区間 (tmin, tmax) を返す。
    pub fn hit_range(&self, r: Ray, tmin: f64, tmax: f64) -> Option<(f64, f64)> {
        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        self.hit_range_inv(r, inv, tmin, tmax)
    }

    /// 事前計算した逆方向ベクトルを使って交差区間を返す。
    /// BVH トラバーサルで近い子ノードを先に処理するために使用。
    pub fn hit_range_inv(&self, r: Ray, inv: Vec3, mut tmin: f64, mut tmax: f64) -> Option<(f64, f64)> {
        let t0x = (self.min.x - r.o.x) * inv.x;
        let t1x = (self.max.x - r.o.x) * inv.x;
        tmin = tmin.max(t0x.min(t1x));
        tmax = tmax.min(t0x.max(t1x) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }

        let t0y = (self.min.y - r.o.y) * inv.y;
        let t1y = (self.max.y - r.o.y) * inv.y;
        tmin = tmin.max(t0y.min(t1y));
        tmax = tmax.min(t0y.max(t1y) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }

        let t0z = (self.min.z - r.o.z) * inv.z;
        let t1z = (self.max.z - r.o.z) * inv.z;
        tmin = tmin.max(t0z.min(t1z));
        tmax = tmax.min(t0z.max(t1z) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }
        Some((tmin, tmax))
    }
}

impl Sphere {
    /// レイと球の交差判定（二次方程式の解法、誤差上界付き）。
    ///
    /// |P − C|² = r² にレイ P(t) = O + tD を代入した at² + 2bt + c = 0 を、桁落ちしない形
    /// q = −(b + sign(b)·√disc)、t = q/a と c/q で解く。各根について計算誤差の上界を見積もり、
    /// `t > 誤差上界`（真に正と言える）かつ [tmin, tmax] の近い方を採用する。交差点は球面上に
    /// 射影し直し（PBRT と同じ）、誤差上界 `p_error` を付ける。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let oc = r.o - self.c;
        let oc_err = (r.o.abs() + self.c.abs()).max_abs() * gamma(1);
        let a = r.d.dot(r.d);
        let b = oc.dot(r.d);
        let rr = self.r * self.r;
        let c = oc.dot(oc) - rr;
        let disc = b * b - a * c;
        if !(disc > 0.0) {
            return None;
        }
        // 誤差の見積もりは交差しうるときだけ行う
        let dl1 = r.d.l1();
        let ocl1 = oc.l1();
        let a_err = gamma(3) * dl1 * dl1;
        let b_err = gamma(3) * ocl1 * dl1 + dl1 * oc_err;
        let c_err = gamma(3) * (ocl1 * ocl1 + rr) + 2.0 * ocl1 * oc_err;
        let disc_err = gamma(3) * (b * b + (a * c).abs()) + 2.0 * b.abs() * b_err + a * c_err + c.abs() * a_err;
        let sq = disc.sqrt();
        let sq_err = disc_err / (2.0 * sq);
        let q = -(b + b.signum() * sq);
        if q == 0.0 {
            return None;
        }
        let q_err = b_err + sq_err + gamma(1) * (b.abs() + sq);
        // t0 = q/a, t1 = c/q
        let t0 = q / a;
        let t0_err = gamma(1) * t0.abs() + (q_err + t0.abs() * a_err) / a;
        let t1 = c / q;
        let t1_err = gamma(1) * t1.abs() + (c_err + t1.abs() * q_err) / q.abs();
        let (near, far) = if t0 <= t1 { ((t0, t0_err), (t1, t1_err)) } else { ((t1, t1_err), (t0, t0_err)) };
        let accept = |(t, e): (f64, f64)| t > e && t >= tmin && t <= tmax;
        let t = if accept(near) {
            near.0
        } else if accept(far) {
            far.0
        } else {
            return None;
        };
        // 球面上へ射影し直す
        let mut p_obj = oc + r.d * t;
        let len = p_obj.len();
        if len > 0.0 {
            p_obj = p_obj * (self.r / len);
        }
        let p = self.c + p_obj;
        let p_error = p_obj.abs() * gamma(5) + (self.c.abs() + p_obj.abs()) * gamma(2);
        let n = p_obj / self.r;
        Some(Hit { t, p, n, mat_id: self.mat_id, prim_id: 0, inst_id: None, p_error })
    }
}

impl Triangle {
    /// 静止三角形（モーションブラーなし）を 3 頂点から構築する。
    /// シャッター開/閉に同一頂点を設定し、エッジを事前計算する。
    pub fn new_static(v0: Vec3, v1: Vec3, v2: Vec3, mat_id: usize) -> Self {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        Self {
            v0_0: v0, v1_0: v1, v2_0: v2,
            v0_1: v0, v1_1: v1, v2_1: v2,
            e1_0: e1, e2_0: e2, e1_1: e1, e2_1: e2,
            mat_id,
        }
    }

    /// シャッター時間 `time` での補間頂点を返す（モーションブラー用）。
    pub fn vertices_at(&self, time: f64) -> (Vec3, Vec3, Vec3) {
        let t = time;
        let v0 = self.v0_0 * (1.0 - t) + self.v0_1 * t;
        let v1 = self.v1_0 * (1.0 - t) + self.v1_1 * t;
        let v2 = self.v2_0 * (1.0 - t) + self.v2_1 * t;
        (v0, v1, v2)
    }

    /// モーションブラー全体を囲む保守的な AABB を返す（開+閉の和）。
    pub fn bounds(&self) -> Aabb {
        let b0 = Aabb::from_points(self.v0_0, self.v1_0, self.v2_0);
        let b1 = Aabb::from_points(self.v0_1, self.v1_1, self.v2_1);
        b0.union(b1)
    }

    /// BVH 分割用の重心を返す（モーションブラー全体の AABB の中心）。
    pub fn centroid(&self) -> Vec3 {
        self.bounds().centroid()
    }

    /// Möller–Trumbore 法によるレイ-三角形交差判定（誤差上界付き）。
    ///
    /// レイの time で頂点を線形補間してからテストする（モーションブラー対応）。両面判定。
    /// t の計算誤差の上界（頂点・原点の丸めと、行列式・分子の積の丸め）を見積もり、`t > 誤差上界` の
    /// ヒットだけを採用する（原点が面の誤差の箱の外にずらしてあれば、元の面は必ず棄却される）。
    /// 以前の行列式の絶対しきい値（|det| < 1e-10、面積の 2 乗に比例するためスケール依存）は使わない。
    /// 交差点は重心座標から求め、成分ごとの誤差上界 `p_error` を付ける。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let (t, u, v) = self.intersect(r, tmin, tmax)?;
        Some(self.hit_at(r, t, u, v))
    }

    /// 交差の判定だけを行い、`(t, u, v)`（u, v は v1, v2 の重心座標）を返す。交差点と誤差上界の計算は
    /// [`hit_at`](Self::hit_at) に分けてあり、BVH の走査では最近接の 1 つだけを確定させる（高速化）。
    pub fn intersect(&self, r: Ray, tmin: f64, tmax: f64) -> Option<(f64, f64, f64)> {
        // レイの time で頂点を補間
        let time = r.time;
        let (v0, v1, v2) = self.vertices_at(time);
        let e1 = v1 - v0;
        let e2 = v2 - v0;

        let pvec = r.d.cross(e2);
        let det = e1.dot(pvec);
        if det == 0.0 || !det.is_finite() {
            return None;
        }

        let inv_det = 1.0 / det;
        let tvec = r.o - v0;
        let u = tvec.dot(pvec) * inv_det;
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let qvec = tvec.cross(e1);
        let v = r.d.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let thit = e2.dot(qvec) * inv_det;
        if thit < tmin || thit > tmax {
            return None;
        }

        // t の誤差上界（保守的）: 頂点の補間・辺の引き算の丸め（err_v）、原点の引き算の丸め（err_o）、
        // 3 重積（分子 e2·(tvec×e1)、行列式 e1·(d×e2)）の丸めと入力誤差の伝播
        let vmax = v0.max_abs().max(v1.max_abs()).max(v2.max_abs());
        let err_v = gamma(4) * vmax;
        let err_o = gamma(2) * (r.o.max_abs() + v0.max_abs());
        let (e1l, e2l, tl, dl) = (e1.l1(), e2.l1(), tvec.l1(), r.d.l1());
        let num_err = gamma(7) * e1l * e2l * tl + 3.0 * (err_o * e1l * e2l + tl * err_v * (e1l + e2l));
        let det_err = gamma(7) * e1l * e2l * dl + 3.0 * dl * err_v * (e1l + e2l);
        let t_err = (num_err + thit.abs() * det_err) * inv_det.abs() * 2.0;
        if thit <= t_err {
            return None;
        }
        Some((thit, u, v))
    }

    /// [`intersect`](Self::intersect) の結果から交差情報（重心座標で求めた交差点・誤差上界・法線）を作る。
    pub fn hit_at(&self, r: Ray, thit: f64, u: f64, v: f64) -> Hit {
        let (v0, v1, v2) = self.vertices_at(r.time);
        let (e1, e2) = (v1 - v0, v2 - v0);
        let b0 = 1.0 - u - v;
        let p = v0 * b0 + v1 * u + v2 * v;
        let p_error = ((v0 * b0).abs() + (v1 * u).abs() + (v2 * v).abs()) * gamma(9);
        let n = e1.cross(e2).norm();
        Hit { t: thit, p, n, mat_id: self.mat_id, prim_id: 0, inst_id: None, p_error }
    }
}
