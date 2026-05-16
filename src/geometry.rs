//! 幾何プリミティブと軸平行バウンディングボックス（AABB）。
//!
//! レンダラーの基本的な幾何形状を定義する:
//! - `Hit`: レイとサーフェスの交差情報
//! - `Sphere`: 球プリミティブ（解析的なレイ交差判定）
//! - `Triangle`: モーションブラー対応三角形（Möller–Trumbore 法）
//! - `Aabb`: 軸平行バウンディングボックス（スラブ法によるレイ交差判定）

use crate::math::Vec3;
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
        // Expand slightly for numerical robustness
        let eps = 1e-9;
        bb.min = Vec3::new(bb.min.x - eps, bb.min.y - eps, bb.min.z - eps);
        bb.max = Vec3::new(bb.max.x + eps, bb.max.y + eps, bb.max.z + eps);
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
        tmax = tmax.min(t0x.max(t1x));
        if tmax <= tmin { return false; }

        let t0y = (self.min.y - r.o.y) * inv.y;
        let t1y = (self.max.y - r.o.y) * inv.y;
        tmin = tmin.max(t0y.min(t1y));
        tmax = tmax.min(t0y.max(t1y));
        if tmax <= tmin { return false; }

        let t0z = (self.min.z - r.o.z) * inv.z;
        let t1z = (self.max.z - r.o.z) * inv.z;
        tmin = tmin.max(t0z.min(t1z));
        tmax = tmax.min(t0z.max(t1z));
        if tmax <= tmin { return false; }
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
        tmax = tmax.min(t0x.max(t1x));
        if tmax <= tmin { return None; }

        let t0y = (self.min.y - r.o.y) * inv.y;
        let t1y = (self.max.y - r.o.y) * inv.y;
        tmin = tmin.max(t0y.min(t1y));
        tmax = tmax.min(t0y.max(t1y));
        if tmax <= tmin { return None; }

        let t0z = (self.min.z - r.o.z) * inv.z;
        let t1z = (self.max.z - r.o.z) * inv.z;
        tmin = tmin.max(t0z.min(t1z));
        tmax = tmax.min(t0z.max(t1z));
        if tmax <= tmin { return None; }
        Some((tmin, tmax))
    }
}

impl Sphere {
    /// レイと球の交差判定（二次方程式の解法）。
    ///
    /// |P - C|² = r² にレイ P(t) = O + tD を代入し、
    /// at² + 2bt + c = 0 の判別式で交差を判定する。
    /// 近い方の解が [tmin, tmax] 外なら遠い方を試す。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let oc = r.o - self.c;
        let a = r.d.dot(r.d);
        let b = oc.dot(r.d);
        let c = oc.dot(oc) - self.r * self.r;
        let d = b * b - a * c;
        if d <= 0.0 {
            return None;
        }
        let sd = d.sqrt();
        let mut t = (-b - sd) / a;
        if t < tmin || t > tmax {
            t = (-b + sd) / a;
            if t < tmin || t > tmax {
                return None;
            }
        }
        let p = r.at(t);
        let n = (p - self.c) / self.r;
        Some(Hit { t, p, n, mat_id: self.mat_id })
    }
}

impl Triangle {
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

    /// Möller–Trumbore 法によるレイ-三角形交差判定。
    ///
    /// レイの time で頂点を線形補間してからテストする（モーションブラー対応）。
    /// 両面判定（det の符号で表裏を区別しない）。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        // レイの time で頂点を補間
        let t = r.time;
        let (v0, v1, v2) = self.vertices_at(t);
        let e1 = v1 - v0;
        let e2 = v2 - v0;

        let pvec = r.d.cross(e2);
        let det = e1.dot(pvec);

        // Double-sided: reject rays nearly parallel to the triangle
        const EPS: f64 = 1e-10;
        if det.abs() < EPS {
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

        let p = r.at(thit);
        let n = e1.cross(e2).norm();
        Some(Hit { t: thit, p, n, mat_id: self.mat_id })
    }
}
