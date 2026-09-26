//! ワールド表現、インスタンシング、ライトサンプリング、交差クエリ。
//!
//! `World` はシーン内の全ジオメトリ（球・メッシュインスタンス）と
//! ライトサンプリング用の CDF を保持する。
//!
//! ## ライトサンプリング
//! 発光マテリアルを持つプリミティブから CDF を構築し、
//! 面積 × 輝度に比例した確率でライトを選択する。
//! 選んだライト上の点は `Light::sample` で求める。球光源は参照点から見える円錐を立体角一様に
//! サンプリングし（参照点が球の内部か表面から丸め誤差の距離以内なら表面積一様）、三角形光源は表面積一様。
//! PDF は `Light::pdf_omega` に一本化され、`sample_light` と `light_pdf` が共有する。

use std::cell::Cell;
use std::sync::{Arc, OnceLock};

use crate::bvh::Bvh;
use crate::geometry::{face_forward, Aabb, Hit, Sphere, Triangle, TriangleSource};
use crate::material::Material;
use crate::obj_loader::{MeshData, NO_NORMAL, NO_UV};
use crate::math::{cdf_search, gamma, Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::texture::AlphaMask;
use crate::transform::{AnimatedTransform, Transform};


/// 三角形メッシュ（メッシュ単位の BVH 付き）。
///
/// 三角形配列と、それを対象に構築した BVH を対で保持する。BVH は自身の構築元と
/// 異なる三角形配列を渡されると壊れるため、その対応関係はこの型の外に出さない
/// （`bvh` が非公開なのはそのため）。交差判定は [`Mesh::hit`] を通して行う。
pub struct Mesh {
    /// メッシュの三角形リスト（シャッター開頂点のみ。閉頂点は `motion`、PERF-4 P4b）
    pub tris: Vec<Triangle>,
    /// モーションブラーのシャッター閉頂点。空なら全三角形が静止（閉 = 開、追加メモリゼロ）。
    /// 非空なら `tris` と同じ長さで添字がそのまま対応する。
    motion: Vec<[Vec3; 3]>,
    /// 頂点法線（OBJ の `vn`、正規化済み）。スムーズシェーディングしないメッシュでは空
    vn: Vec<Vec3>,
    /// 三角形ごとの `vn` の添字。空ならメッシュ全体が面法線
    tri_vn: Vec<[u32; 3]>,
    /// テクスチャ座標（OBJ の `vt`）。UV を持たないメッシュでは空
    uv: Vec<[f64; 2]>,
    /// 三角形ごとの `vt` の添字。空ならメッシュ全体が UV 無し
    tri_uv: Vec<[u32; 3]>,
    /// アルファマスク（`map_d`）の表。三角形が持つのは `tri_alpha` 経由の添字だけ。
    /// マスクを持たないメッシュでは空（交差判定は従来の経路で、コストはゼロ）
    masks: Vec<Arc<AlphaMask>>,
    /// 三角形ごとのマスク添字 + 1（0 = マスク無し）と、その材質の不透明度 `d`。
    /// `masks` が空なら空
    tri_alpha: Vec<(u16, f32)>,
    /// メッシュ内の BVH（高速交差判定用）
    bvh: Bvh,
}

impl Mesh {
    /// 三角形リストからメッシュと BVH を構築する（面法線のみ、モーションブラー無し）。
    pub fn new(tris: Vec<Triangle>) -> Self {
        let bvh = Bvh::build(TriangleSource::from(&tris));
        Self { tris, motion: Vec::new(), vn: Vec::new(), tri_vn: Vec::new(), uv: Vec::new(), tri_uv: Vec::new(), masks: Vec::new(), tri_alpha: Vec::new(), bvh }
    }

    /// 頂点法線付きでメッシュを構築する。`tri_vn` の長さが三角形数と合わない場合は
    /// 面法線だけのメッシュとして扱う（壊れた入力で添字がずれるより安全側）。
    pub fn with_normals(tris: Vec<Triangle>, vn: Vec<Vec3>, tri_vn: Vec<[u32; 3]>) -> Self {
        Self::build(tris, Vec::new(), vn, tri_vn, Vec::new(), Vec::new())
    }

    /// 頂点法線と UV（どちらも省略可）を付けてメッシュを構築する。
    /// 添字配列の長さが三角形数と合わない場合は、その属性だけ無かったことにする
    /// （壊れた入力で添字がずれるより安全側）。`motion` は空か `tris` と同じ長さであること
    /// （合わなければモーションブラー無しとして扱う。壊れた入力で添字がずれるより安全側）。
    pub fn build(
        tris: Vec<Triangle>,
        motion: Vec<[Vec3; 3]>,
        vn: Vec<Vec3>,
        tri_vn: Vec<[u32; 3]>,
        uv: Vec<[f64; 2]>,
        tri_uv: Vec<[u32; 3]>,
    ) -> Self {
        let (vn, tri_vn) = if vn.is_empty() || tri_vn.len() != tris.len() {
            (Vec::new(), Vec::new())
        } else {
            (vn, tri_vn)
        };
        let (uv, tri_uv) = if uv.is_empty() || tri_uv.len() != tris.len() {
            (Vec::new(), Vec::new())
        } else {
            (uv, tri_uv)
        };
        let motion = if motion.len() == tris.len() { motion } else { Vec::new() };
        let bvh = Bvh::build(TriangleSource::new(&tris, &motion));
        Self { tris, motion, vn, tri_vn, uv, tri_uv, masks: Vec::new(), tri_alpha: Vec::new(), bvh }
    }

    /// [`MeshData`] からメッシュを構築する。
    pub fn with_normals_from(data: MeshData) -> Self {
        Self::build(data.tris, data.motion, data.vn, data.tri_vn, data.uv, data.tri_uv)
    }

    /// 三角形 `ti` のシャッター閉頂点。`motion` が空なら開頂点と同じ（静止三角形）。
    #[inline(always)]
    fn close(&self, ti: usize) -> (Vec3, Vec3, Vec3) {
        if self.motion.is_empty() {
            let t = &self.tris[ti];
            (t.v0_0, t.v1_0, t.v2_0)
        } else {
            let m = self.motion[ti];
            (m[0], m[1], m[2])
        }
    }

    /// 三角形 `tri_id` のシャッター時刻 `time` での補間頂点（モーションブラー対応）。
    pub fn vertices_at(&self, tri_id: usize, time: f64) -> Option<(Vec3, Vec3, Vec3)> {
        let tri = self.tris.get(tri_id)?;
        Some(tri.vertices_at_with(self.close(tri_id), time))
    }

    /// このメッシュ用の [`TriangleSource`]（`Bvh` に渡す束）。
    #[inline]
    fn source(&self) -> TriangleSource<'_> {
        TriangleSource::new(&self.tris, &self.motion)
    }

    /// アルファマスクを付ける。`masks[i]` を三角形が `tri_alpha[t] = (i + 1, d)` で参照する
    /// （`(0, _)` はマスク無し）。UV を持たないメッシュや三角形数の合わない入力では何もしない。
    /// 交差判定の形は変わるが BVH は作り直さない（採否だけを変える）。
    pub fn with_alpha(mut self, masks: Vec<Arc<AlphaMask>>, tri_alpha: Vec<(u16, f32)>) -> Self {
        if !masks.is_empty() && tri_alpha.len() == self.tris.len() && self.has_uv() {
            self.masks = masks;
            self.tri_alpha = tri_alpha;
        }
        self
    }

    /// アルファマスクを持つか。
    pub fn has_alpha(&self) -> bool {
        !self.masks.is_empty()
    }

    /// 三角形 `ti` の重心座標 `(u, v)` の位置が不透明か（マスクの無い三角形・UV の無い三角形は常に不透明）。
    #[inline]
    fn alpha_opaque(&self, ti: usize, u: f64, v: f64) -> bool {
        let (k, d) = self.tri_alpha[ti];
        if k == 0 {
            return true;
        }
        match self.texture_coords(ti, (u, v)) {
            Some(uv) => self.masks[k as usize - 1].opaque(uv, d as f64),
            None => true,
        }
    }

    /// このメッシュがスムーズシェーディング（頂点法線の補間）を行うか。
    pub fn is_smooth(&self) -> bool {
        !self.tri_vn.is_empty()
    }

    /// 頂点法線の本数（メモリ量の報告用）。
    pub fn normal_count(&self) -> usize {
        self.vn.len()
    }

    /// このメッシュが頂点 UV を持つか。
    pub fn has_uv(&self) -> bool {
        !self.tri_uv.is_empty()
    }

    /// UV の個数（メモリ量の報告用）。
    pub fn uv_count(&self) -> usize {
        self.uv.len()
    }

    /// メッシュ内三角形に対するレイ交差判定（オブジェクト空間）。
    /// 頂点法線を持つ三角形なら、重心座標で補間したシェーディング法線を `Hit::ns` に入れる。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        // アルファマスクを持つメッシュだけ、採否判定つきの探索にする（他は従来の経路のまま）。
        // 透明な交差は無かったことにして探索を続けるので、シャドウレイでも穴を光が抜ける
        let mut h = if self.masks.is_empty() {
            self.bvh.hit(self.source(), r, tmin, tmax)?
        } else {
            self.bvh.hit_filtered(self.source(), r, tmin, tmax, |ti, u, v| self.alpha_opaque(ti, u, v))?
        };
        if !self.tri_vn.is_empty() {
            if let Some(ns) = self.shading_normal(h.prim_id, h.bary, h.ng) {
                h.ns = ns;
            }
        }
        if !self.tri_uv.is_empty() {
            if let Some(uv) = self.texture_coords(h.prim_id, h.bary) {
                h.uv = uv;
            }
        }
        Some(h)
    }

    /// メッシュ内三角形に対する遮蔽判定（オブジェクト空間、any-hit）。`skip_prim` はこのメッシュ内の
    /// 三角形番号で、サンプルした光源自身がこのメッシュにある場合に渡す（一致する交差は遮蔽と数えず
    /// 探索を続ける。アルファ透明と同じ「棄却して継続」の仕組みに乗せる）。シェーディング法線・UV は
    /// 解決しない（遮蔽の有無にしか使わないので無駄な計算を省く）。
    pub fn occluded(&self, r: Ray, tmin: f64, tmax: f64, skip_prim: Option<usize>) -> Option<Hit> {
        if self.masks.is_empty() {
            self.bvh.any_hit_filtered(self.source(), r, tmin, tmax, |ti, _, _| Some(ti) != skip_prim)
        } else {
            self.bvh.any_hit_filtered(self.source(), r, tmin, tmax, |ti, u, v| {
                Some(ti) != skip_prim && self.alpha_opaque(ti, u, v)
            })
        }
    }

    /// 三角形 `tri_id` の ∂p/∂u, ∂p/∂v（オブジェクト空間、正規化しない）。`time` は交差判定と同じく
    /// シャッター時刻で、辺 `e1`/`e2` を `time` で補間して使う（交差点と接空間が別時刻の幾何にならない）。
    ///
    /// UV が無い三角形、および UV 三角形が縮退している（行列式が丸めの範囲でゼロ）三角形は `None`。
    /// 任意基底へのフォールバックはしない（隣接三角形で接空間が飛んで縞になるより、摂動しない方が安全）。
    /// 縮退の判定は相対: `|det| <= (|a| + |b|)·γ(2)`（`det = a − b`、桁落ちの上界）。
    pub(crate) fn uv_derivatives(&self, tri_id: usize, time: f64) -> Option<(Vec3, Vec3)> {
        let idx = *self.tri_uv.get(tri_id)?;
        if idx[0] == NO_UV {
            return None;
        }
        let (a, b, c) = (self.uv[idx[0] as usize], self.uv[idx[1] as usize], self.uv[idx[2] as usize]);
        // bary = (v1 の重み, v2 の重み) に合わせて uv0 からの差を取る
        let (du1, dv1) = (b[0] - a[0], b[1] - a[1]);
        let (du2, dv2) = (c[0] - a[0], c[1] - a[1]);
        let (p, q) = (du1 * dv2, du2 * dv1);
        let det = p - q;
        if !det.is_finite() || det.abs() <= (p.abs() + q.abs()) * gamma(2) {
            return None;
        }
        let tri = self.tris.get(tri_id)?;
        // 事前計算エッジは持たない（PERF-4 P4a）。以前 `obj_loader`/`Triangle::new_static` が
        // 構築時にやっていたのと全く同じ式・同じ順序（先に開/閉それぞれの辺を作り、それを time で
        // 補間する）でその場で計算する。頂点を先に time 補間してから引き算する順序だと
        // 浮動小数点演算が非結合的なため丸めが変わりうるので、あえてこの順序を踏襲している。
        // シャッター閉頂点は三角形自身ではなくメッシュ側（`self.motion`）に持つ（PERF-4 P4b）。
        let (c0, c1, c2) = self.close(tri_id);
        let e1_0 = tri.v1_0 - tri.v0_0;
        let e2_0 = tri.v2_0 - tri.v0_0;
        let e1_1 = c1 - c0;
        let e2_1 = c2 - c0;
        let e1 = e1_0 * (1.0 - time) + e1_1 * time;
        let e2 = e2_0 * (1.0 - time) + e2_1 * time;
        let dpdu = (e1 * dv2 - e2 * dv1) / det;
        let dpdv = (e2 * du1 - e1 * du2) / det;
        Some((dpdu, dpdv))
    }

    /// UV を持つのに接空間を作れない（UV 三角形が縮退した）三角形の数。
    /// 呼び出し側（マップを持つ材質を使うとき）がシーンごとに 1 回だけ警告するために使う。
    /// 全三角形を走査するので、マップを使わないシーンでは呼ばない（読み込み時間を増やさない）。
    pub fn degenerate_uv_count(&self) -> usize {
        (0..self.tri_uv.len())
            .filter(|&i| self.tri_uv[i][0] != NO_UV && self.uv_derivatives(i, 0.0).is_none())
            .count()
    }

    /// 三角形 `tri_id` の重心座標 `(b1, b2)` での補間 UV。UV を持たない三角形は `None`。
    fn texture_coords(&self, tri_id: usize, bary: (f64, f64)) -> Option<(f64, f64)> {
        let idx = *self.tri_uv.get(tri_id)?;
        if idx[0] == NO_UV {
            return None;
        }
        let (b1, b2) = bary;
        let b0 = 1.0 - b1 - b2;
        let (a, b, c) = (
            self.uv[idx[0] as usize],
            self.uv[idx[1] as usize],
            self.uv[idx[2] as usize],
        );
        Some((a[0] * b0 + b[0] * b1 + c[0] * b2, a[1] * b0 + b[1] * b1 + c[1] * b2))
    }

    /// 三角形 `tri_id` の重心座標 `(b1, b2)` での補間法線。頂点法線が無い三角形や、
    /// 補間結果が退化した（長さ 0 の）場合は `None`（= 面法線のまま）。
    fn shading_normal(&self, tri_id: usize, bary: (f64, f64), ng: Vec3) -> Option<Vec3> {
        let idx = *self.tri_vn.get(tri_id)?;
        if idx[0] == NO_NORMAL {
            return None;
        }
        let (b1, b2) = bary;
        let b0 = 1.0 - b1 - b2;
        let n = self.vn[idx[0] as usize] * b0 + self.vn[idx[1] as usize] * b1 + self.vn[idx[2] as usize] * b2;
        let len = n.len();
        if !(len > 0.0) {
            return None;
        }
        // 幾何法線と同じ側に揃える（向きの取り違えを Hit の不変条件として吸収する）
        Some(face_forward(n / len, ng))
    }
}

#[derive(Clone, Copy, Debug)]
/// メッシュのインスタンス（トランスフォーム + マテリアルオーバーライド）。
///
/// 同一メッシュを異なる位置・回転・スケール・マテリアルで配置できる。
pub struct Instance {
    /// 参照するメッシュの ID
    pub mesh_id: usize,
    /// オブジェクト → ワールド変換
    pub xform: Transform,
    /// マテリアルオーバーライド（None なら三角形のマテリアルを使用）
    pub mat_override: Option<usize>,
    /// ワールド空間の保守的な境界ボックス（メッシュの AABB の 8 頂点を変換し、変換の誤差上界ぶん広げたもの）。
    /// `World::hit` で、レイを物体空間へ変換する前の安価な棄却に使う。
    /// **アニメーション変換のあるインスタンスでは、シャッター区間の全時刻の位置を含む掃過ボリューム**
    /// （[`AnimatedTransform::swept_bounds`]。TLAS の葉の箱も同じ値を使うので、TLAS も掃過ボリュームで判定する）。
    pub world_bounds: Aabb,
    /// 静止時（`xform` = シャッター開の変換）の `world_bounds`
    pub static_bounds: Aabb,
    /// アニメーション変換（シャッター開 `xform` → 閉）。`None` なら静止インスタンスで、従来と完全に同じ経路を通る
    pub anim: Option<AnimatedTransform>,
}

/// 球のワールド空間の境界ボックス（中心 ± 半径）。球の交差判定の丸め誤差ぶんを見込んで少し広げる。
fn sphere_world_bounds(s: &Sphere) -> Aabb {
    let pad = s.r * 1e-9 + gamma(8) * (s.c.x.abs().max(s.c.y.abs()).max(s.c.z.abs()) + s.r);
    let e = Vec3::new(s.r + pad, s.r + pad, s.r + pad);
    Aabb::empty().grow(s.c - e).grow(s.c + e)
}

/// 動く球の時刻 `t` の中心 `c0·(1−t) + c1·t` と、その丸め誤差の上界（成分ごとの L∞、余裕を見て γ(4)）。
#[inline(always)]
fn lerp_center(c0: Vec3, c1: Vec3, t: f64) -> (Vec3, f64) {
    (c0 * (1.0 - t) + c1 * t, gamma(4) * c0.max_abs().max(c1.max_abs()))
}

/// 動く球の掃過ボリューム: シャッター区間の両端の中心に置いた球の箱の和。中心は時刻の**線形**関数なので、
/// 区間の途中の球は両端の球の凸包に入り、箱は凸なのでこの和は**厳密に保守的**（回転のあるインスタンスと違って
/// 途中で膨らまない）。補間の丸めぶん（`lerp_center` の誤差）を足す。
fn moving_sphere_bounds(s: &Sphere, end: Vec3, shutter: (f64, f64)) -> Aabb {
    let at = |t: f64| {
        let (c, err) = lerp_center(s.c, end, t);
        let b = sphere_world_bounds(&Sphere { c, ..*s });
        let e = Vec3::new(err, err, err);
        Aabb::empty().grow(b.min - e).grow(b.max + e)
    };
    at(shutter.0).union(at(shutter.1))
}

/// メッシュを `xform` で配置したインスタンスの、ワールド空間の保守的な境界ボックス。
fn instance_world_bounds(mesh: &Mesh, xform: &Transform) -> Aabb {
    let mut b = Aabb::empty();
    let Some(root) = mesh.bvh.root_bounds() else { return b };
    let (lo, hi) = (root.min, root.max);
    let zero = Vec3::new(0.0, 0.0, 0.0);
    for k in 0..8 {
        let corner = Vec3::new(
            if k & 1 == 0 { lo.x } else { hi.x },
            if k & 2 == 0 { lo.y } else { hi.y },
            if k & 4 == 0 { lo.z } else { hi.z },
        );
        let (w, err) = xform.apply_point_with_error(corner, zero);
        b = b.grow(w - err).grow(w + err);
    }
    // 箱の角の座標自体の丸めのぶん、座標に比例してさらに広げる
    let pad = |lo: f64, hi: f64| gamma(3) * lo.abs().max(hi.abs());
    let (px, py, pz) = (pad(b.min.x, b.max.x), pad(b.min.y, b.max.y), pad(b.min.z, b.max.z));
    b.min = Vec3::new(b.min.x - px, b.min.y - py, b.min.z - pz);
    b.max = Vec3::new(b.max.x + px, b.max.y + py, b.max.z + pz);
    b
}

/// インスタンスの交差判定で、ワールド空間の tmin 判定に却下された面の先を探し直す最大回数。
const INSTANCE_RETRY_LIMIT: usize = 4;

/// ジオメトリ・インスタンス・ライトの集合体。
///
/// ジオメトリの追加は [`add_sphere`](World::add_sphere) /
/// [`add_mesh_instance`](World::add_mesh_instance) を通して行い、全て追加し終えたら
/// 必ず [`build_lights`](World::build_lights) を呼ぶこと。フィールドが非公開なのは、
/// 「追加してから CDF を構築し忘れる」という呼び出し側の不変条件違反を防ぐため。
pub struct World {
    /// シーン内の球プリミティブ
    spheres: Vec<Sphere>,
    /// メッシュ（三角形群 + BVH）
    meshes: Vec<Mesh>,
    /// メッシュのインスタンス（トランスフォーム付き）
    instances: Vec<Instance>,
    /// 発光プリミティブのリスト
    lights: Vec<LightInfo>,
    /// ライト選択用の累積分布関数（CDF）
    light_cdf: Vec<f64>,
    /// CDF の総重み
    light_total: f64,
    /// 光源のまとまり（発光球・発光インスタンス）が 2 つ以上のとき true: 光源選択を参照点からの重み（[`World::selection_weight`]）で行う。
    /// false（まとまりが 1 つ以下、または多すぎる）のときは従来の出力パワーだけの CDF（選択確率が変わりようがない場合や、O(N) が引き合わない場合）
    light_select: LightSelect,
    /// 光源 BVH（`LightSelect::Bvh` のときの選択に使う。光源が 0 個なら空）
    light_bvh: LightBvh,
    /// 球インデックス → lights 上の ID（発光体でなければ None）。
    /// `light_pdf` が BSDF サンプリングで命中した発光体を逆引きするために使う。
    sphere_light_id: Vec<Option<usize>>,
    /// メッシュ構築（BVH 構築を含む）に費やした累計時間。起動時の内訳表示に使う。
    mesh_build_time: std::time::Duration,
    /// (インスタンス ID, メッシュ内三角形 ID) → lights 上の ID。
    tri_light_id: std::collections::HashMap<(usize, usize), usize>,
    /// トップレベル BVH（インスタンス + 球）。最初の交差判定で遅延構築し、ジオメトリを足すたびに捨てる。
    /// プリミティブが `TLAS_MIN_PRIMS` 未満なら `None`（従来どおり線形に総当たりする）
    tlas: OnceLock<Option<Tlas>>,
    /// デルタ光源（点・平行・スポット）。`light_cdf` などの面光源の仕組みには**入れない**（[`DeltaLight`] 参照）
    delta_lights: Vec<DeltaLight>,
    /// シャッター区間（アニメーション変換の掃過ボリュームを作るのに使う）
    shutter: (f64, f64),
    /// 球ごとのシャッター閉じ時点の中心（`None` は静止）。**動く球が 1 つも無ければ空**で、静止球の経路は従来のまま
    sphere_end: Vec<Option<Vec3>>,
}

/// トップレベル BVH。葉の添字 `k < n_inst` はインスタンス `k`、それ以外は球 `k - n_inst`
/// （線形総当たりの走査順「インスタンス → 球」と同じ通し番号で、同値の t のタイブレークにも使う）。
struct Tlas {
    bvh: Bvh,
    n_inst: usize,
    n_sph: usize,
}

/// この数以上のプリミティブ（インスタンス + 球）があるときだけ TLAS を使う。少数では総当たりのほうが速い
/// （決め方は tlas_report.md 参照）。
const TLAS_MIN_PRIMS: usize = 20;

/// 光源のまとまり（発光球・発光インスタンス）の数 `n_groups` に応じた既定の選び方（実測: light_bvh_report.md）:
/// 1 つ以下は出力パワーの CDF（選択確率が変わりようがない）。2〜32 は線形の重み付け（N = 10 で等価時間 0.90）。
/// 33〜79 はどれも得にならない（spiral の 45 で線形 1.06 / BVH 1.07 の損、人工 50 個で ±0）ので CDF のまま。
/// 80 以上は光源 BVH（N = 100 で 0.84、200 で 0.70、500 で 0.63、1000 で 0.65、2000 で 0.82。線形は N = 1000 から損）。
fn default_light_select(n_groups: usize) -> LightSelect {
    const LINEAR_MAX_GROUPS: usize = 32;
    const BVH_MIN_GROUPS: usize = 80;
    if n_groups < 2 {
        LightSelect::Power
    } else if n_groups <= LINEAR_MAX_GROUPS {
        LightSelect::Linear
    } else if n_groups >= BVH_MIN_GROUPS {
        LightSelect::Bvh
    } else {
        LightSelect::Power
    }
}

/// 光源の選び方（NEE がどの発光体を狙うか）。`build_lights` が既定を決め、`set_light_select` で切り替えられる（測定用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightSelect {
    /// 出力パワーだけの CDF（従来。光源のまとまりが 1 つ以下のときの既定）
    Power,
    /// 全光源に参照点からの重み `Φ / max(d², r²)` を付けて線形に選ぶ（O(N)）
    Linear,
    /// 光源 BVH（Conty & Kulla 2018）でルートから確率的に降りて選ぶ（O(log N)）
    Bvh,
}

/// 光源 BVH のノード。
#[derive(Clone, Copy, Debug)]
struct LightNode {
    /// 光源のまとまりの AABB
    bbox: Aabb,
    /// 出力パワーの総和
    power: f64,
    /// 向きの円錐: 軸 `w`（発光の向き）、放射の広がり `θ_o`（`cos_o` = −1 は全方向 = 球）、発光の広がり `θ_e`（`cos_e`）
    w: Vec3,
    cos_o: f64,
    sin_o: f64,
    cos_e: f64,
    left: i32,
    right: i32,
    parent: i32,
    /// 葉のとき `World::lights` の添字、内部ノードは -1
    light: i32,
}

/// 光源 BVH（1 光源 = 1 葉）。木の形は SAH（パワー × 表面積）の 2 分割。
#[derive(Clone, Debug, Default)]
struct LightBvh {
    nodes: Vec<LightNode>,
    /// 光源 → 葉のノード番号（`light_pdf` が葉から根へ経路を引くのに使う）
    leaf_of: Vec<u32>,
}

/// 向きの円錐 `(軸, cos θ_o, cos θ_e)` どうしの和（両方を含む最小の円錐。PBRT v4 の `Union(DirectionCone)` と同じ手続き）。
fn cone_union(a: (Vec3, f64), b: (Vec3, f64)) -> (Vec3, f64) {
    // (軸, cos θ_o) のみ（θ_e は呼び出し側が min(cos_e) で合わせる）
    let (wa, ca) = a;
    let (wb, cb) = b;
    if ca <= -1.0 || cb <= -1.0 {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    let (ta, tb) = (ca.clamp(-1.0, 1.0).acos(), cb.clamp(-1.0, 1.0).acos());
    let td = wa.dot(wb).clamp(-1.0, 1.0).acos();
    if (td + tb).min(std::f64::consts::PI) <= ta {
        return a;
    }
    if (td + ta).min(std::f64::consts::PI) <= tb {
        return b;
    }
    let to = (ta + td + tb) * 0.5;
    if to >= std::f64::consts::PI {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    let tr = to - ta;
    let wr = wa.cross(wb);
    if wr.len() <= 1e-12 {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    // wa を wr まわりに tr だけ回す（Rodrigues。wr ⟂ wa）
    let k = wr.norm();
    let w = wa * tr.cos() + k.cross(wa) * tr.sin() + k * (k.dot(wa) * (1.0 - tr.cos()));
    (w.norm(), to.cos())
}

/// `cos(a − b)` と `sin(a − b)`（`a ≤ b` なら 0 に丸める = `θ' = max(0, a − b)`）。
#[inline(always)]
fn sub_angle_clamped(sa: f64, ca: f64, sb: f64, cb: f64) -> (f64, f64) {
    if ca > cb {
        (0.0, 1.0)
    } else {
        (sa * cb - ca * sb, ca * cb + sa * sb)
    }
}

impl LightBvh {
    /// 光源のリストから作る。`world` は三角形の頂点（境界・向き）を引くのに使う。
    fn build(lights: &[LightInfo], world: &World) -> Self {
        let n = lights.len();
        if n == 0 {
            return Self::default();
        }
        // 葉のデータ: (AABB, パワー, 軸, cos θ_o, cos θ_e)
        let leaf: Vec<(Aabb, f64, Vec3, f64, f64)> = lights
            .iter()
            .map(|info| match info.light {
                Light::Sphere { idx } => {
                    let s = &world.spheres[idx];
                    let r = Vec3::new(s.r, s.r, s.r);
                    (Aabb { min: s.c - r, max: s.c + r }, info.weight, Vec3::new(0.0, 1.0, 0.0), -1.0, 0.0)
                }
                Light::Triangle { mesh_id, tri_id, inst_id } => {
                    let (a, b, c) = tri_world_verts(world, mesh_id, tri_id, inst_id, 0.5).unwrap_or((Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)));
                    (Aabb::empty().grow(a).grow(b).grow(c), info.weight, info.normal, 1.0, 0.0)
                }
            })
            .collect();
        let mut bvh = Self { nodes: Vec::with_capacity(2 * n), leaf_of: vec![0; n] };
        let mut idx: Vec<usize> = (0..n).collect();
        bvh.build_range(&leaf, &mut idx, -1);
        bvh
    }

    fn build_range(&mut self, leaf: &[(Aabb, f64, Vec3, f64, f64)], idx: &mut [usize], parent: i32) -> i32 {
        let id = self.nodes.len() as i32;
        if idx.len() == 1 {
            let (bbox, power, w, cos_o, cos_e) = leaf[idx[0]];
            self.nodes.push(LightNode { bbox, power, w, cos_o, sin_o: (1.0 - cos_o * cos_o).max(0.0).sqrt(), cos_e, left: -1, right: -1, parent, light: idx[0] as i32 });
            self.leaf_of[idx[0]] = id as u32;
            return id;
        }
        // 分割: 重心の広がりが最大の軸で、ビン分割の SAH（パワー × 表面積）。有効な分割が無ければ中央値
        let mut cb = Aabb::empty();
        for &i in idx.iter() {
            cb = cb.grow(leaf[i].0.centroid());
        }
        let e = cb.extent();
        let axis = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
        let key = |i: usize| {
            let c = leaf[i].0.centroid();
            match axis { 0 => c.x, 1 => c.y, _ => c.z }
        };
        idx.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
        let area = |b: Aabb| {
            let e = b.extent();
            2.0 * (e.x * e.y + e.y * e.z + e.z * e.x) + 1e-12
        };
        // 並べ替えた列での全分割位置を評価（N が小さいので全数。累積で O(N)）
        let m = idx.len();
        let mut left_cost = vec![0.0; m];
        let (mut bb, mut pw) = (Aabb::empty(), 0.0);
        for k in 0..m {
            bb = bb.union(leaf[idx[k]].0);
            pw += leaf[idx[k]].1;
            left_cost[k] = pw * area(bb);
        }
        let (mut bb, mut pw) = (Aabb::empty(), 0.0);
        let mut best = (f64::INFINITY, m / 2);
        for k in (1..m).rev() {
            bb = bb.union(leaf[idx[k]].0);
            pw += leaf[idx[k]].1;
            let cost = left_cost[k - 1] + pw * area(bb);
            if cost < best.0 {
                best = (cost, k);
            }
        }
        let mid = best.1.clamp(1, m - 1);
        self.nodes.push(LightNode { bbox: Aabb::empty(), power: 0.0, w: Vec3::new(0.0, 1.0, 0.0), cos_o: 1.0, sin_o: 0.0, cos_e: 0.0, left: -1, right: -1, parent, light: -1 });
        let (li, ri) = idx.split_at_mut(mid);
        let l = self.build_range(leaf, li, id);
        let r = self.build_range(leaf, ri, id);
        let (a, b) = (self.nodes[l as usize], self.nodes[r as usize]);
        let (w, cos_o) = cone_union((a.w, a.cos_o), (b.w, b.cos_o));
        let node = &mut self.nodes[id as usize];
        node.bbox = a.bbox.union(b.bbox);
        node.power = a.power + b.power;
        node.w = w;
        node.cos_o = cos_o;
        node.sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
        node.cos_e = a.cos_e.min(b.cos_e);
        node.left = l;
        node.right = r;
        id
    }
}

impl World {
    /// 光源の選び方を切り替える（既定は `build_lights` が決める。測定・テスト用）。
    pub fn set_light_select(&mut self, mode: LightSelect) {
        self.light_select = mode;
    }

    /// 現在の光源の選び方。
    pub fn light_select(&self) -> LightSelect {
        self.light_select
    }

    /// 光源 `id` を選ぶ確率（現在の選び方 `light_select` に従う。`sample_light` の `pdf_select` と同じ値）。テスト・診断用。
    pub fn light_selection_prob(&self, id: usize, from: Vec3) -> f64 {
        match self.light_select {
            LightSelect::Bvh => self.bvh_prob(id, from),
            _ => {
                let total = self.selection_total(from);
                if total > 0.0 { self.selection_prob(id, from, total) } else { 0.0 }
            }
        }
    }

    /// 光源 BVH の内部ノードの、参照点 `p` から見た重要度: `Φ / max(d², r²)`（`d` = ノードの境界ボックスの中心までの距離、
    /// `r` = その半径。前回の線形の重みと同じ形）を、向きの円錐が「`p` は寄与しえない側にある」と保守的に判定できるときは 0 にする
    /// （軸と `p` への方向の角 θ_w、放射の広がり θ_o、箱の見込み角 θ_b から `θ' = max(0, θ_w − θ_o − θ_b)` を作り、`θ' ≥ θ_e` なら 0）。
    /// 箱の中に `p` があれば向きでは落とさない。**葉（光源 1 個）は前回の `selection_weight` そのもの**（片面発光の平面テストが厳密）。
    #[inline]
    fn node_importance(&self, n: &LightNode, p: Vec3) -> f64 {
        if n.light >= 0 {
            return Self::selection_weight(&self.lights[n.light as usize], p);
        }
        let c = n.bbox.centroid();
        let h = n.bbox.max - c;
        let r2 = h.dot(h);
        let dv = p - c;
        let d2 = dv.dot(dv);
        if n.cos_o > -1.0 && d2 > r2 {
            let d = d2.sqrt();
            let cos_w = (n.w.dot(dv) / d).clamp(-1.0, 1.0);
            let sin_w = (1.0 - cos_w * cos_w).max(0.0).sqrt();
            let sin_b = (r2 / d2).sqrt();
            let cos_b = (1.0 - sin_b * sin_b).max(0.0).sqrt();
            let (s1, c1) = sub_angle_clamped(sin_w, cos_w, n.sin_o, n.cos_o);
            let (_, c2) = sub_angle_clamped(s1, c1, sin_b, cos_b);
            if c2 <= n.cos_e {
                return 0.0;
            }
        }
        n.power / d2.max(r2).max(f64::MIN_POSITIVE)
    }

    /// 内部ノードの 2 つの子を選ぶ確率 `(左, 右)`（子の重要度の比）。両方 0 なら `(0, 0)`。
    /// **`sample_light`（`bvh_select`）と `light_pdf`（`bvh_prob`）の両方がこの関数だけで確率を作る**。
    #[inline]
    fn bvh_child_probs(&self, node: &LightNode, p: Vec3) -> (f64, f64) {
        let il = self.node_importance(&self.light_bvh.nodes[node.left as usize], p);
        let ir = self.node_importance(&self.light_bvh.nodes[node.right as usize], p);
        let sum = il + ir;
        if !(sum > 0.0) {
            return (0.0, 0.0);
        }
        (il / sum, ir / sum)
    }

    /// 光源 BVH で光源を選ぶ。`u` ∈ [0,1) 1 個を各段で使い回す（子を選んだら `u` を選んだ区間に写して一様に戻す: 確率的な分割）。
    /// 返り値は `(光源の添字, 選択確率)`。選択確率は根から葉までの各段の確率の**根側から順の積**。
    fn bvh_select(&self, mut u: f64, p: Vec3) -> Option<(usize, f64)> {
        let nodes = &self.light_bvh.nodes;
        if nodes.is_empty() {
            return None;
        }
        let mut cur = 0usize;
        let mut prob = 1.0;
        loop {
            let n = &nodes[cur];
            if n.light >= 0 {
                // 根が葉（光源 1 個）のときは重要度が 0（裏側）でないことだけ確かめる
                if cur == 0 && !(self.node_importance(n, p) > 0.0) {
                    return None;
                }
                return Some((n.light as usize, prob));
            }
            let (pl, pr) = self.bvh_child_probs(n, p);
            if !(pl + pr > 0.0) {
                return None;
            }
            if u < pl {
                prob *= pl;
                u = (u / pl).min(1.0 - f64::EPSILON / 2.0);
                cur = n.left as usize;
            } else {
                prob *= pr;
                u = ((u - pl) / pr).clamp(0.0, 1.0 - f64::EPSILON / 2.0);
                cur = n.right as usize;
            }
        }
    }

    /// 光源 `id` を光源 BVH で選ぶ確率。**`bvh_select` が降りるのと同じ経路**（葉から親をたどって根側から並べる）で、
    /// 同じ `bvh_child_probs` の値を同じ順序で掛ける（ビット一致）。
    fn bvh_prob(&self, id: usize, p: Vec3) -> f64 {
        let nodes = &self.light_bvh.nodes;
        if nodes.is_empty() {
            return 0.0;
        }
        let leaf = self.light_bvh.leaf_of[id] as usize;
        // 葉 → 根の経路（各要素 = (親, 葉側が左か)）
        let mut path = [(0usize, false); 128];
        let mut depth = 0usize;
        let mut cur = leaf;
        while nodes[cur].parent >= 0 {
            let par = nodes[cur].parent as usize;
            path[depth] = (par, nodes[par].left as usize == cur);
            depth += 1;
            cur = par;
        }
        if depth == 0 {
            // 根が葉
            return if self.node_importance(&nodes[leaf], p) > 0.0 { 1.0 } else { 0.0 };
        }
        let mut prob = 1.0;
        for k in (0..depth).rev() {
            let (par, is_left) = path[k];
            let (pl, pr) = self.bvh_child_probs(&nodes[par], p);
            prob *= if is_left { pl } else { pr };
        }
        prob
    }
}

impl World {
    /// 空のワールドを生成する。
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            meshes: Vec::new(),
            instances: Vec::new(),
            lights: Vec::new(),
            light_cdf: Vec::new(),
            light_total: 0.0,
            light_select: LightSelect::Power,
            light_bvh: LightBvh::default(),
            sphere_light_id: Vec::new(),
            mesh_build_time: std::time::Duration::ZERO,
            tri_light_id: std::collections::HashMap::new(),
            tlas: OnceLock::new(),
            delta_lights: Vec::new(),
            shutter: (0.0, 1.0),
            sphere_end: Vec::new(),
        }
    }

    /// デルタ光源を足す。面光源用の `light_cdf` / `sample_light` / `light_pdf` には影響しない。
    pub fn add_delta_light(&mut self, light: DeltaLight) {
        self.delta_lights.push(light);
    }

    /// デルタ光源の一覧。積分器の NEE が毎回すべてを順に評価する。
    pub fn delta_lights(&self) -> &[DeltaLight] {
        &self.delta_lights
    }

    /// 全ジオメトリ（球と、変換後のメッシュインスタンス）のワールド空間の境界ボックス。
    /// インスタンスはメッシュの BVH ルートの AABB の 8 頂点を変換して包む。
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::empty();
        for (i, s) in self.spheres.iter().enumerate() {
            let r = Vec3::new(s.r, s.r, s.r);
            b = b.grow(s.c - r).grow(s.c + r);
            if let Some(Some(end)) = self.sphere_end.get(i) {
                b = b.grow(*end - r).grow(*end + r);
            }
        }
        for inst in &self.instances {
            let Some(mesh) = self.meshes.get(inst.mesh_id) else { continue };
            let Some(root) = mesh.bvh.root_bounds() else { continue };
            let (lo, hi) = (root.min, root.max);
            for k in 0..8 {
                let corner = Vec3::new(
                    if k & 1 == 0 { lo.x } else { hi.x },
                    if k & 2 == 0 { lo.y } else { hi.y },
                    if k & 4 == 0 { lo.z } else { hi.z },
                );
                b = b.grow(inst.xform.apply_point(corner));
            }
        }
        b
    }

    /// 球プリミティブを追加し、その `World::spheres` 上のインデックスを返す。
    pub fn add_sphere(&mut self, sphere: Sphere) -> usize {
        self.tlas = OnceLock::new();
        let idx = self.spheres.len();
        self.spheres.push(sphere);
        idx
    }

    /// 球 `idx` にシャッター閉じ時点の中心 `end` を与え、レイの `time`（0 = 開 = `Sphere::c`、1 = 閉）で**線形補間**する
    /// 動く球にする。球は回転しても見た目が変わらず、スケールは半径で表せるので、必要なのは平行移動だけ
    /// （インスタンスの `AnimatedTransform` は使わない）。`end` が非有限なら `false`（静止のまま）。
    /// **発光する球には使わないこと**（光源サンプリングは時刻を見ないので、呼び出し側が警告して静止にする）。
    pub fn set_sphere_end(&mut self, idx: usize, end: Vec3) -> bool {
        if !(end.x.is_finite() && end.y.is_finite() && end.z.is_finite()) {
            return false;
        }
        self.tlas = OnceLock::new();
        self.sphere_end.resize(self.spheres.len(), None);
        self.sphere_end[idx] = Some(end);
        true
    }

    /// 球 `idx` の交差判定。動く球はレイの時刻で中心を補間してから解く（静止は従来と同じ `Sphere::hit`）。
    #[inline(always)]
    fn sphere_hit(&self, idx: usize, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let s = &self.spheres[idx];
        if let Some(Some(end)) = self.sphere_end.get(idx) {
            let (c, err) = lerp_center(s.c, *end, r.time);
            return s.hit_at(c, err, r, tmin, tmax);
        }
        s.hit(r, tmin, tmax)
    }

    /// 三角形群からメッシュを構築し、`xform` で配置したインスタンスを追加する。
    /// 追加したインスタンスの ID を返す。
    pub fn add_mesh_instance(&mut self, tris: Vec<Triangle>, xform: Transform, mat_override: Option<usize>) -> usize {
        let (mesh, dt) = timed(|| Mesh::new(tris));
        self.mesh_build_time += dt;
        self.add_mesh(mesh, xform, mat_override)
    }

    /// OBJ から読んだメッシュ（頂点法線付きでありうる）を `xform` で配置したインスタンスを追加する。
    pub fn add_mesh_data_instance(&mut self, data: MeshData, xform: Transform, mat_override: Option<usize>) -> usize {
        let (mesh, dt) = timed(|| Mesh::with_normals_from(data));
        self.mesh_build_time += dt;
        self.add_mesh(mesh, xform, mat_override)
    }

    /// 構築済みのメッシュを登録してインスタンスを追加する（上の 2 つの共通部分）。
    /// アルファマスク付きでメッシュ（`MeshData`）をインスタンス配置する（[`Mesh::with_alpha`] 参照）。
    pub fn add_mesh_data_instance_with_alpha(
        &mut self,
        data: MeshData,
        masks: Vec<Arc<AlphaMask>>,
        tri_alpha: Vec<(u16, f32)>,
        xform: Transform,
    ) -> usize {
        let (mesh, dt) = timed(|| Mesh::with_normals_from(data).with_alpha(masks, tri_alpha));
        self.mesh_build_time += dt;
        self.add_mesh(mesh, xform, None)
    }

    fn add_mesh(&mut self, mesh: Mesh, xform: Transform, mat_override: Option<usize>) -> usize {
        let mesh_id = self.meshes.len();
        self.meshes.push(mesh);
        self.add_instance_of(mesh_id, xform, mat_override)
    }

    /// 既存のメッシュ `mesh_id`（三角形配列と BVH）の新しいインスタンスを足し、インスタンス ID を返す。
    /// 同じ OBJ を何度も配置するとき、メッシュを作り直さずに共有するための入口。材質を変えたいときは
    /// `mat_override` を使う（インスタンス側の属性なので、メッシュを共有したまま材質だけ違えられる）。
    pub fn add_instance_of(&mut self, mesh_id: usize, xform: Transform, mat_override: Option<usize>) -> usize {
        self.tlas = OnceLock::new();
        let inst_id = self.instances.len();
        let world_bounds = instance_world_bounds(&self.meshes[mesh_id], &xform);
        self.instances.push(Instance { mesh_id, xform, mat_override, world_bounds, static_bounds: world_bounds, anim: None });
        inst_id
    }

    /// インスタンスにシャッター閉の変換 `end` を与え、レイの `time`（0 = 開 = `xform`、1 = 閉）で補間する
    /// アニメーション変換にする。**開・閉のどちらかが特異・鏡像（det ≤ 0）・非有限、または極分解が収束しない
    /// ときは `false` を返し、静止のまま**（呼び出し側が警告する）。`world_bounds` は現在のシャッター区間
    /// （[`World::set_shutter`]、既定 [0, 1]）の掃過ボリュームになる。
    pub fn set_instance_end_transform(&mut self, inst_id: usize, end: Transform) -> bool {
        let inst = &self.instances[inst_id];
        let Some(anim) = AnimatedTransform::new(inst.xform, end) else { return false };
        self.tlas = OnceLock::new();
        self.instances[inst_id].anim = Some(anim);
        self.refresh_swept_bounds(inst_id);
        true
    }

    /// カメラのシャッター区間 `[open, close]`（レイの `time` の範囲）。アニメーション変換のあるインスタンスの
    /// 掃過ボリュームをこの区間で作り直す。頂点モーション（OBJ 2 枚）の鍵は time = 0 と 1 なので、区間は [0, 1] 内であること。
    pub fn set_shutter(&mut self, open: f64, close: f64) {
        self.tlas = OnceLock::new();
        self.shutter = (open, close);
        for id in 0..self.instances.len() {
            if self.instances[id].anim.is_some() {
                self.refresh_swept_bounds(id);
            }
        }
    }

    fn refresh_swept_bounds(&mut self, inst_id: usize) {
        let (open, close) = self.shutter;
        let inst = &self.instances[inst_id];
        let Some(anim) = inst.anim else { return };
        // メッシュ（頂点モーションを含む）の物体空間の境界球: ルート AABB の中心と半対角線
        let bbox = self.meshes[inst.mesh_id].bvh.root_bounds();
        let bounds = match bbox {
            Some(b) => {
                let c = (b.min + b.max) * 0.5;
                anim.swept_bounds(c, (b.max - c).len(), open, close)
            }
            None => inst.static_bounds,
        };
        self.instances[inst_id].world_bounds = bounds;
    }

    /// メッシュ構築（BVH 構築を含む）に費やした累計時間。
    pub fn mesh_build_time(&self) -> std::time::Duration {
        self.mesh_build_time
    }

    /// 全インスタンス展開後の三角形数（表示用。インスタンスごとにメッシュの三角形数を足す）。
    /// インスタンス `inst_id` が参照するメッシュの ID（`add_mesh_*` が返すのはインスタンス ID なので、
    /// 共有用にメッシュ ID が要る呼び出し側はこれで引く。既存 API のシグネチャを変えないための手段）。
    pub fn instance_mesh_id(&self, inst_id: usize) -> usize {
        self.instances[inst_id].mesh_id
    }
    /// メッシュ（三角形配列 + BVH）の数。同じ OBJ を共有していれば、インスタンス数より少ない。
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
    /// インスタンスの数。
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }
    pub fn triangle_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.meshes.get(i.mesh_id).map_or(0, |m| m.tris.len()))
            .sum()
    }

    /// 球プリミティブの一覧を返す。
    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    /// メッシュの一覧を返す。
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    /// メッシュインスタンスの一覧を返す。
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// 登録済みライトの一覧を返す（`build_lights` 実行後に有効）。
    pub fn lights(&self) -> &[LightInfo] {
        &self.lights
    }

    /// トップレベル BVH（あれば）。最初の呼び出しで構築する。プリミティブが少ない、または構築後に
    /// ジオメトリが増えて古くなっているときは `None`（呼び出し側は線形に総当たりする）。
    fn tlas(&self) -> Option<&Tlas> {
        let n_prims = self.instances.len() + self.spheres.len();
        if n_prims < TLAS_MIN_PRIMS {
            return None;
        }
        let t = self
            .tlas
            .get_or_init(|| {
                let mut bounds: Vec<Aabb> = Vec::with_capacity(n_prims);
                bounds.extend(self.instances.iter().map(|i| i.world_bounds));
                bounds.extend(self.spheres.iter().enumerate().map(|(i, s)| match self.sphere_end.get(i) {
                    Some(Some(end)) => moving_sphere_bounds(s, *end, self.shutter),
                    _ => sphere_world_bounds(s),
                }));
                Some(Tlas { bvh: Bvh::build_from_bounds(&bounds, 1), n_inst: self.instances.len(), n_sph: self.spheres.len() })
            })
            .as_ref()?;
        (t.n_inst == self.instances.len() && t.n_sph == self.spheres.len()).then_some(t)
    }

    /// TLAS を深さ優先で辿り、葉のプリミティブ番号ごとに `visit(k)` を呼ぶ（`k` は [`Tlas`] の通し番号）。
    /// ノードの箱は「インスタンスの境界」と同じ判定（区間 `(min(tmin, 0), tmax·(1+1e-9))`）で棄却する。
    /// `tmax` は呼び出し側が `visit` の中で縮められる（最近接探索）。`visit` が true を返したら打ち切る（any-hit）。
    fn tlas_traverse(tlas: &Tlas, r: Ray, tmin: f64, tmax: &Cell<f64>, visit: impl FnMut(usize) -> bool) {
        // 広い BVH（[`crate::constants::bvh::WIDE_WIDTH`] 分木）で走査する。箱の判定区間は従来と同じ
        // （`tmin.min(0.0)` から `tmax·(1 + 1e-9)`）。訪問順は 2 分木と違うが、同値 t は呼び出し側が添字で解決する
        tlas.bvh.traverse_wide(r, tmin.min(0.0), tmax, visit);
    }

    /// ワールド内の全ジオメトリに対するレイ交差判定。
    ///
    /// インスタンスのレイはオブジェクト空間に変換してからメッシュ BVH でテストし、
    /// ヒット結果をワールド空間に戻す。球はワールド空間で直接テストする。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        if let Some(tlas) = self.tlas() {
            let inv_d = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
            // トップレベル BVH 経由。線形総当たり（下）と同じ結果を返す: 最近接を採り、t が完全に同値のときは
            // 総当たりの走査順（インスタンス → 球、添字の小さい方）が勝つように通し番号 `k` で決める
            let closest = Cell::new(tmax);
            let mut best: Option<Hit> = None;
            let mut best_k = usize::MAX;
            Self::tlas_traverse(tlas, r, tmin, &closest, |k| {
                let c = closest.get();
                if k < tlas.n_inst {
                    if let Some(h) = self.hit_instance(k, r, inv_d, tmin, c, k < best_k) {
                        closest.set(h.t);
                        best = Some(h);
                        best_k = k;
                    }
                } else {
                    let idx = k - tlas.n_inst;
                    // 同値タイで勝てる（添字が小さい）ときだけ、区間の上端をわずかに広げて t == closest の交差を拾う
                    let hi = if k < best_k && best.is_some() { c.next_up() } else { c };
                    if let Some(mut h) = self.sphere_hit(idx, r, tmin, hi) {
                        if h.t < c || (h.t == c && k < best_k) {
                            h.prim_id = idx;
                            closest.set(h.t);
                            best = Some(h);
                            best_k = k;
                        }
                    }
                }
                false
            });
            return best;
        }
        self.hit_linear(r, tmin, tmax)
    }

    /// 線形総当たり版の [`World::hit`]（インスタンス → 球の順）。プリミティブが少ないときの本体で、
    /// TLAS 経由の結果と一致するべき基準でもある（テストが比較する）。
    fn hit_linear(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let inv_d = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        let mut closest = tmax;
        let mut best: Option<Hit> = None;

        // Instances: ray -> object space
        for inst_id in 0..self.instances.len() {
            if let Some(h) = self.hit_instance(inst_id, r, inv_d, tmin, closest, false) {
                closest = h.t;
                best = Some(h);
            }
        }

        // Spheres
        for idx in 0..self.spheres.len() {
            if let Some(mut h) = self.sphere_hit(idx, r, tmin, closest) {
                h.prim_id = idx;
                closest = h.t;
                best = Some(h);
            }
        }

        best
    }

    /// インスタンス `inst_id` に対する最近接交差の候補。ワールド空間の `t` が `closest` より小さいもの
    /// （`tie_ok` なら `closest` と同値でもよい）だけを返す。物体空間への変換・tmin 写像・自己交差時の再探索・
    /// 誤差上界は従来の [`World::hit`] のループ本体そのまま（TLAS の葉と線形総当たりの両方から呼ばれる）。
    #[inline(always)]
    fn hit_instance(&self, inst_id: usize, r: Ray, inv_d: Vec3, tmin: f64, closest: f64, tie_ok: bool) -> Option<Hit> {
        let inst = &self.instances[inst_id];
        let mesh = match self.meshes.get(inst.mesh_id) {
            Some(m) => m,
            None => return None,
        };
        // ワールド空間の境界ボックスで先に棄却する（物体空間への変換と誤差計算を省く）。箱は保守的で、
        // スラブ判定も遠い側を広げてあるので、ここで棄却されるインスタンスに当たるレイは無い
        if !inst.world_bounds.hit_inv(r, inv_d, tmin.min(0.0), closest * (1.0 + 1e-9)) {
            return None;
        }

        // アニメーション変換のインスタンスだけ、レイの時刻で補間した変換を使う（静止は従来の `inst.xform`
        // をそのまま参照するので、経路も演算も従来と同一 = バイト一致の根拠）
        let anim_xf;
        let xf: &Transform = match &inst.anim {
            Some(a) => {
                anim_xf = a.at(r.time);
                &anim_xf
            }
            None => &inst.xform,
        };
        let (o_obj, o_err) = xf.apply_point_inv_with_error_linf(r.o);
        let d_obj_raw = xf.apply_vec_inv(r.d);
        let d_len = d_obj_raw.len().max(1e-30); // Vec3::norm と同じ式（ビット一致）
        let d_obj = d_obj_raw / d_len; // stabilize
        // 物体空間へ写した原点には変換の丸め誤差 o_err が乗る。PBRT と同じく、原点をその誤差ぶん
        // レイ方向に進めておく（真の原点がどこにあっても、進めた原点より手前にある）。ワールド空間で
        // 面の誤差の箱の外へずらしてある原点は、物体空間でも元の面より先に出るので自己交差しない。
        // 進めた距離 dt のぶん物体空間の t は小さくなるが、採否はワールド空間の t で決めるので影響しない。
        let dt = d_obj.l1() * o_err;
        let r_obj = Ray { o: o_obj + d_obj * dt, d: d_obj, time: r.time };

        // ワールド空間の区間 (tmin, closest) を物体空間に写す。|r.d| = 1 なので t_obj = t_world·|A⁻¹d|。
        // 丸めで境界上の候補を落とさないよう、両端とも相対 1e-9 だけ外側に広げる（採否は下の
        // ワールド空間 t 判定が決めるので、広げても結果は変わらない）。tmin も同じく写す: 写さないと
        // 拡大インスタンスでは近い面を取りこぼし（t_obj < tmin）、縮小インスタンスでは自己交差回避の
        // 帯の中の面を BVH が返してしまう。
        let tmin_obj = tmin * d_len * (1.0 - 1e-9);
        let tmax_obj = closest * d_len * (1.0 + 1e-9);

        // Object-space BVH。ワールド空間の判定で「近すぎる」（t_world <= tmin）と却下された場合は、
        // 丸めで帯の内側に入った面なので、その面より先から同じインスタンスを探し直す（奥の面を
        // 取りこぼさないため）。同一平面に重なった面が帯に多数あっても止まるよう回数を制限する。
        let mut search_from = tmin_obj;
        for _ in 0..=INSTANCE_RETRY_LIMIT {
            let Some(h_obj) = mesh.hit(r_obj, search_from, tmax_obj) else { break };
            let (p_world, p_error) = xf.apply_point_with_error(h_obj.p, h_obj.p_error);

            // r.d is normalized in Camera::ray()
            let t_world = (p_world - r.o).dot(r.d);
            if t_world <= tmin {
                search_from = h_obj.t.next_up();
                continue;
            }
            if t_world < closest || (tie_ok && t_world == closest) {
                let mat_id = inst.mat_override.unwrap_or(h_obj.mat_id);
                // 法線は逆転置行列で変換する（非一様スケールでも面に垂直なまま）。
                // 鏡像（負のスケール）を含む変換では逆転置が向きを反転させうるので、
                // シェーディング法線は変換後の幾何法線と同じ側に揃え直す。
                let ng = xf.apply_normal(h_obj.ng);
                let ns = if h_obj.is_smooth() {
                    face_forward(xf.apply_normal(h_obj.ns), ng)
                } else {
                    ng
                };
                return Some(Hit {
                    t: t_world,
                    p: p_world,
                    ng,
                    ns,
                    mat_id,
                    prim_id: h_obj.prim_id,
                    inst_id: Some(inst_id),
                    p_error,
                    bary: h_obj.bary,
                    uv: h_obj.uv,
                });
            }
            break;
        }
        None
    }

    /// シャドウレイの遮蔽判定（any-hit）。`hit` と違い「最近接」ではなく「(tmin, tmax) に採用できる
    /// 交差が 1 つでもあるか」だけを返す。見つかり次第、残りのインスタンス・球は調べずに打ち切る。
    ///
    /// `skip` はサンプルした光源自身（`(inst_id, prim_id)`、`Hit`/`LightSample` と同じ規約）。
    /// これに一致する交差は遮蔽と数えず探索を続ける（球光源では交差の t の誤差上界が終点側の
    /// ずらし量を超えうるので、光源自身への交差が `(tmin, tmax)` 内に来ることがある。この除外は
    /// 保険ではなく必須 — 外すと `sample/default.xml` で光が約 6% 失われる。§`nee_area_light` 参照）。
    /// 環境光 NEE のように除外すべき光源が無い場合は `None` を渡す。
    ///
    /// インスタンスの物体空間変換・tmin 写像・自己交差時の再探索・誤差上界は [`World::hit`] と同じ
    /// 規律に従う（水密交差・スケール不変を壊さない）。アルファマスクの透明判定は [`Mesh::occluded`] に
    /// 委譲する。
    pub fn occluded(&self, r: Ray, tmin: f64, tmax: f64, skip: Option<(Option<usize>, usize)>) -> bool {
        if let Some(tlas) = self.tlas() {
            let inv_d = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
            // any-hit なので探索順は結果に影響しない（`tmax` は縮めない）
            let tmax_cell = Cell::new(tmax);
            let mut found = false;
            Self::tlas_traverse(tlas, r, tmin, &tmax_cell, |k| {
                found = if k < tlas.n_inst {
                    self.occluded_instance(k, r, inv_d, tmin, tmax, skip)
                } else {
                    let idx = k - tlas.n_inst;
                    skip != Some((None, idx)) && self.sphere_hit(idx, r, tmin, tmax).is_some()
                };
                found
            });
            return found;
        }
        self.occluded_linear(r, tmin, tmax, skip)
    }

    /// 線形総当たり版の [`World::occluded`]。
    fn occluded_linear(&self, r: Ray, tmin: f64, tmax: f64, skip: Option<(Option<usize>, usize)>) -> bool {
        let inv_d = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        for inst_id in 0..self.instances.len() {
            if self.occluded_instance(inst_id, r, inv_d, tmin, tmax, skip) {
                return true;
            }
        }

        // Spheres
        for idx in 0..self.spheres.len() {
            if skip == Some((None, idx)) {
                continue;
            }
            if self.sphere_hit(idx, r, tmin, tmax).is_some() {
                return true;
            }
        }

        false
    }

    /// インスタンス `inst_id` が `(tmin, tmax)` の遮蔽になるか（[`World::occluded`] のループ本体そのまま）。
    #[inline(always)]
    fn occluded_instance(&self, inst_id: usize, r: Ray, inv_d: Vec3, tmin: f64, tmax: f64, skip: Option<(Option<usize>, usize)>) -> bool {
        let inst = &self.instances[inst_id];
        let mesh = match self.meshes.get(inst.mesh_id) {
            Some(m) => m,
            None => return false,
        };
        if !inst.world_bounds.hit_inv(r, inv_d, tmin.min(0.0), tmax * (1.0 + 1e-9)) {
            return false;
        }
        // このインスタンスが光源自身を含むなら、そのメッシュ内三角形番号を渡して除外する
        let skip_prim = match skip {
            Some((Some(light_inst), prim)) if light_inst == inst_id => Some(prim),
            _ => None,
        };

        // アニメーション変換のインスタンスだけ、レイの時刻で補間した変換を使う（静止は従来の `inst.xform`
        // をそのまま参照するので、経路も演算も従来と同一 = バイト一致の根拠）
        let anim_xf;
        let xf: &Transform = match &inst.anim {
            Some(a) => {
                anim_xf = a.at(r.time);
                &anim_xf
            }
            None => &inst.xform,
        };
        let (o_obj, o_err) = xf.apply_point_inv_with_error_linf(r.o);
        let d_obj_raw = xf.apply_vec_inv(r.d);
        let d_len = d_obj_raw.len().max(1e-30);
        let d_obj = d_obj_raw / d_len;
        let dt = d_obj.l1() * o_err;
        let r_obj = Ray { o: o_obj + d_obj * dt, d: d_obj, time: r.time };

        // `World::hit` と同じ理由でどちらの端も相対 1e-9 だけ外側に広げる（採否はワールド空間 t が決める）。
        // `tmax` は `closest` のように縮めない（any-hit なので他インスタンスの結果と比べる必要が無い）。
        let tmin_obj = tmin * d_len * (1.0 - 1e-9);
        let tmax_obj = tmax * d_len * (1.0 + 1e-9);

        let mut search_from = tmin_obj;
        for _ in 0..=INSTANCE_RETRY_LIMIT {
            let Some(h_obj) = mesh.occluded(r_obj, search_from, tmax_obj, skip_prim) else { break };
            let (p_world, _) = xf.apply_point_with_error(h_obj.p, h_obj.p_error);
            let t_world = (p_world - r.o).dot(r.d);
            if t_world <= tmin {
                search_from = h_obj.t.next_up();
                continue;
            }
            if t_world < tmax {
                return true;
            }
            break;
        }
        false
    }

    /// ヒット点のワールド空間の接ベクトル対 (∂p/∂u, ∂p/∂v)（長さを保つ = 正規化しない）。
    /// 法線マップ／バンプマップを持つ材質のときだけ呼ぶ想定（`Hit` は接ベクトルを持たず、
    /// `inst_id` / `prim_id` から必要な点でだけ導出する）。球・UV 無し・UV 縮退は `None`。
    ///
    /// 接ベクトルは法線ではないので、インスタンス変換は逆転置ではなく順方向の線形部 `A` で行う。
    pub fn surface_tangents(&self, hit: &Hit, time: f64) -> Option<(Vec3, Vec3)> {
        let inst = self.instances.get(hit.inst_id?)?;
        let mesh = self.meshes.get(inst.mesh_id)?;
        let (dpdu, dpdv) = mesh.uv_derivatives(hit.prim_id, time)?;
        let xf = match &inst.anim {
            Some(a) => a.at(time),
            None => inst.xform,
        };
        Some((xf.apply_vec(dpdu), xf.apply_vec(dpdv)))
    }

    /// 交差点を、その物体の空間へ戻した位置。手続き的テクスチャをローカル座標で評価するために使う。
    ///
    /// - メッシュのインスタンス（`inst_id = Some(i)`）: インスタンス変換の逆。**アニメーションしていれば `time` で
    ///   補間した変換の逆**（開き姿勢の逆だと、動いている間に模様がずれる）。
    /// - 球（`inst_id = None`、`prim_id` = 球の添字）: `p − その時刻の中心`。動く球は補間後の中心。半径では割らない
    ///   （`scale` の意味が球ごとに変わるため）。
    pub fn object_space_point(&self, hit: &Hit, time: f64) -> Vec3 {
        match hit.inst_id {
            Some(i) => match self.instances.get(i) {
                Some(inst) => match &inst.anim {
                    Some(a) => a.at(time).apply_point_inv(hit.p),
                    None => inst.xform.apply_point_inv(hit.p),
                },
                None => hit.p,
            },
            None => match self.spheres.get(hit.prim_id) {
                Some(s) => match self.sphere_end.get(hit.prim_id) {
                    Some(Some(end)) => hit.p - lerp_center(s.c, *end, time).0,
                    _ => hit.p - s.c,
                },
                None => hit.p,
            },
        }
    }

    /// 発光マテリアルからライトサンプリング構造（CDF）を構築する。
    /// 各ライトの重み = 表面積 × 放射輝度の輝度値。
    ///
    /// シェープ別の面積計算は [`Light::area`] に委譲する（[`Self::sample_light`] と共有）。
    pub fn build_lights(&mut self, mats: &[Material]) {
        let mut lights: Vec<LightInfo> = Vec::new();
        // cdf_search は先頭に 0.0 を持つ配列（[0, w0, w0+w1, …]）を前提とする
        // （env.rs と同じ規約）。
        let mut cdf: Vec<f64> = vec![0.0];
        let mut total = 0.0;
        let mut sphere_light_id: Vec<Option<usize>> = vec![None; self.spheres.len()];
        let mut tri_light_id: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();

        let mut n_groups = 0usize;
        let mut add = |light: Light, emit: Color, area: f64, p_error: Vec3, geo: (Vec3, f64, Vec3, Vec3, bool)| -> Option<usize> {
            let weight = area * emit.luminance();
            if weight > 0.0 {
                total += weight;
                let id = lights.len();
                lights.push(LightInfo { light, emit, weight, p_error, center: geo.0, r2: geo.1, plane_p: geo.2, normal: geo.3, one_sided: geo.4 });
                cdf.push(total);
                Some(id)
            } else {
                None
            }
        };

        // Spheres
        for (idx, s) in self.spheres.iter().enumerate() {
            if let Some(emit) = mats.get(s.mat_id).and_then(|m| m.emitted()) {
                let light = Light::Sphere { idx };
                let area = light.area(self, 0.5);
                // 球面上の点 c + n·r の誤差上界（どの点でも |c| + r で抑えられる）
                let m = s.c.abs() + Vec3::new(s.r, s.r, s.r);
                let p_error = m * (gamma(4) + gamma(2));
                let geo = (s.c, s.r * s.r, s.c, Vec3::new(0.0, 0.0, 0.0), false);
                if let Some(id) = add(light, emit, area, p_error, geo) {
                    sphere_light_id[idx] = Some(id);
                    n_groups += 1;
                }
            }
        }

        // Triangles (per instance, using effective material)
        for (inst_id, inst) in self.instances.iter().enumerate() {
            let mesh = match self.meshes.get(inst.mesh_id) {
                Some(m) => m,
                None => continue,
            };
            // このインスタンスの発光三角形の境界球（光源選択の距離の項。インスタンス全体で共有）
            let mut bb = Aabb::empty();
            let mut any = false;
            for (tri_id, tri) in mesh.tris.iter().enumerate() {
                let mat_id = inst.mat_override.unwrap_or(tri.mat_id);
                if mats.get(mat_id).and_then(|m| m.emitted()).is_some() {
                    if let Some((a, b, c)) = tri_world_verts(self, inst.mesh_id, tri_id, inst_id, 0.5) {
                        bb = bb.grow(a).grow(b).grow(c);
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            let center = bb.centroid();
            let r2 = {
                let h = bb.max - center;
                h.dot(h)
            };
            let mut group_has_light = false;
            for (tri_id, tri) in mesh.tris.iter().enumerate() {
                let mat_id = inst.mat_override.unwrap_or(tri.mat_id);
                if let Some(emit) = mats.get(mat_id).and_then(|m| m.emitted()) {
                    let light = Light::Triangle { mesh_id: inst.mesh_id, tri_id, inst_id };
                    let area = light.area(self, 0.5);
                    // シャッター開・閉の両方の頂点で見積もった大きい方（補間はその凸結合）
                    let e0 = tri_point_error(self, inst.mesh_id, tri_id, inst_id, 0.0);
                    let e1 = tri_point_error(self, inst.mesh_id, tri_id, inst_id, 1.0);
                    let p_error = Vec3::new(e0.x.max(e1.x), e0.y.max(e1.y), e0.z.max(e1.z));
                    let (plane_p, normal) = match tri_world_verts(self, inst.mesh_id, tri_id, inst_id, 0.5) {
                        Some((a, b, c)) => {
                            let n = (b - a).cross(c - a);
                            let len = n.len();
                            // Light::sample と同じ向き（cross(v1 − v0, v2 − v0)）と退化時の既定
                            ((a + b + c) / 3.0, if len > 0.0 { n / len } else { Vec3::new(0.0, 1.0, 0.0) })
                        }
                        None => (center, Vec3::new(0.0, 1.0, 0.0)),
                    };
                    if let Some(id) = add(light, emit, area, p_error, (center, r2, plane_p, normal, true)) {
                        tri_light_id.insert((inst_id, tri_id), id);
                        group_has_light = true;
                    }
                }
            }
            if group_has_light {
                n_groups += 1;
            }
        }

        let lights_snapshot = lights.clone();
        self.lights = lights;
        self.light_cdf = cdf;
        self.light_total = total;
        self.light_bvh = LightBvh::build(&lights_snapshot, self);
        self.light_select = default_light_select(n_groups);
        self.sphere_light_id = sphere_light_id;
        self.tri_light_id = tri_light_id;
    }

    /// BSDF サンプリングで発光体に命中した際の、光源選択の立体角 PDF を計算する（MIS 用）。
    ///
    /// `sample_light` が返す `LightSample.pdf` と同一の値を、逆方向（命中結果 `hit` から）
    /// 再構成する。`from` は前バウンスのシェーディング点、`time` はレイの time。
    /// `hit` が発光体でない、または `build_lights` 未実行なら 0 を返す。
    pub fn light_pdf(&self, from: Vec3, time: f64, hit: &Hit) -> f64 {
        if self.light_total <= 0.0 {
            return 0.0;
        }
        let light_id = match hit.inst_id {
            None => self.sphere_light_id.get(hit.prim_id).copied().flatten(),
            Some(inst_id) => self.tri_light_id.get(&(inst_id, hit.prim_id)).copied(),
        };
        let light_id = match light_id {
            Some(id) => id,
            None => return 0.0,
        };
        let info = &self.lights[light_id];
        if self.light_select == LightSelect::Bvh {
            let pdf_select = self.bvh_prob(light_id, from);
            if !(pdf_select > 0.0) {
                return 0.0;
            }
            return pdf_select * info.light.pdf_omega(self, time, from, hit.p, hit.ng);
        }
        // 選択確率は sample_light と同じ関数（`selection_total` と `selection_prob`）で求める
        let total = self.selection_total(from);
        if !(total > 0.0) {
            return 0.0;
        }
        let pdf_select = self.selection_prob(light_id, from, total);
        pdf_select * info.light.pdf_omega(self, time, from, hit.p, hit.ng)
    }

    /// テスト用: 光源選択の重み付けを強制的に切り替える（重み付けの有無で平均が変わらないこと＝不偏性の検定に使う）。
    #[cfg(test)]
    pub(crate) fn force_light_weighting(&mut self, on: bool) {
        self.light_select = if on { LightSelect::Linear } else { LightSelect::Power };
    }

    /// 光源 `info` の、参照点 `from` から見た選択の重み: `Φ / max(d², r²)`（`Φ` = 出力パワー、`d` = 光源のまとまりの
    /// 境界球の中心までの距離、`r` = その半径。近すぎて `d → 0` でも発散しない）。片面発光の三角形は、
    /// 参照点が裏側（`normal · (from − 重心) ≤ 0`）なら 0（裏側からは `pdf_omega` も 0 でサンプルが無駄になるだけ）。
    ///
    /// **`sample_light` と `light_pdf` の両方がこの関数だけを使う**（式を 2 か所に書かない）。MIS は `light_pdf` が
    /// `sample_light` の使った選択確率とビット単位で一致することに依存する。
    #[inline]
    fn selection_weight(info: &LightInfo, from: Vec3) -> f64 {
        if info.one_sided && info.normal.dot(from - info.plane_p) <= 0.0 {
            return 0.0;
        }
        let d = from - info.center;
        info.weight / d.dot(d).max(info.r2).max(f64::MIN_POSITIVE)
    }

    /// 選択確率の分母（全光源の重みの和）。重み付けなしなら従来の `light_total`。**加算は光源の番号順**
    /// （`sample_light` の累積和と同じ順序・同じ丸め）。
    #[inline]
    fn selection_total(&self, from: Vec3) -> f64 {
        if self.light_select != LightSelect::Linear {
            return self.light_total;
        }
        let mut total = 0.0;
        for info in &self.lights {
            total += Self::selection_weight(info, from);
        }
        total
    }

    /// 光源 `id` を選ぶ確率（`total` は [`Self::selection_total`]）。
    #[inline]
    fn selection_prob(&self, id: usize, from: Vec3, total: f64) -> f64 {
        let info = &self.lights[id];
        if self.light_select != LightSelect::Linear {
            return info.weight / total;
        }
        Self::prob_of_weight(Self::selection_weight(info, from), total)
    }

    /// 重み `w` の光源を選ぶ確率 `w / total`（`selection_prob` と `sample_light` の両方がこの 1 か所を通る）。
    #[inline(always)]
    fn prob_of_weight(w: f64, total: f64) -> f64 {
        w / total
    }

    /// CDF を使ってライトを重点的にサンプリングし、位置・法線・放射輝度・PDF を返す。
    /// PDF は立体角ベース（面積 PDF をジオメトリ変換で立体角に変換）。
    ///
    /// 光源面上の点を選ぶ 2 次元乱数は `rng` から引く（層化なし）。乱数消費順は
    /// 「光源の選択 → 面上の点」で、[`Self::sample_light_with_uv`]（`uv_override: None`）と
    /// ビット単位で同じ結果になる。
    pub fn sample_light(&self, rng: &mut Rng, time: f64, p: Vec3) -> Option<LightSample> {
        self.sample_light_impl(rng, time, p, None)
    }

    /// [`Self::sample_light`] の、光源面上の点を選ぶ 2 次元乱数 `uv` を外から指定できる版
    /// （PERF-3: 層化サンプリング用）。**光源の選択**（複数の発光体があるときにどれを狙うか）は
    /// 引き続き `rng` から引く — 層化するのは「選んだ光源の面上のどこを狙うか」だけ
    /// （Sponza のように光源が 1 つで小さいシーンで効くのはこちらのため）。
    pub fn sample_light_with_uv(&self, rng: &mut Rng, time: f64, p: Vec3, uv: (f64, f64)) -> Option<LightSample> {
        self.sample_light_impl(rng, time, p, Some(uv))
    }

    fn sample_light_impl(&self, rng: &mut Rng, time: f64, p: Vec3, uv_override: Option<(f64, f64)>) -> Option<LightSample> {
        if self.light_total <= 0.0 || self.lights.is_empty() {
            return None;
        }
        // 選択: 重み付けありなら参照点からの重みの累積和（番号順）で、なしなら従来の CDF
        let (idx, total, pdf_select) = if self.light_select == LightSelect::Bvh {
            // 光源 BVH: ルートから確率的に降りる（乱数は 1 個を各段で再利用。`bvh_select`）
            let u = rng.next_f64();
            rng.align_pair();
            match self.bvh_select(u, p) {
                Some((i, pr)) => (i, 0.0, pr),
                None => return None,
            }
        } else if self.light_select == LightSelect::Linear {
            // 重みは光源ごとに 1 回だけ計算して持つ（`selection_weight` の値は決定的なので、`light_pdf` が
            // 別に計算した値とビット単位で同じ）。光源が多すぎて配列に入らないときは 2 回に分けて計算する
            const CACHE: usize = 64;
            let n = self.lights.len();
            let mut ws = [0.0f64; CACHE];
            let mut total = 0.0;
            if n <= CACHE {
                for (j, info) in self.lights.iter().enumerate() {
                    let w = Self::selection_weight(info, p);
                    ws[j] = w;
                    total += w;
                }
            } else {
                total = self.selection_total(p);
            }
            if !(total > 0.0) {
                // 全光源が裏側など（どの光源も寄与しない）。乱数の消費は他と揃える
                let _ = rng.next_f64();
                rng.align_pair();
                return None;
            }
            let r = rng.next_f64() * total;
            rng.align_pair();
            let mut acc = 0.0;
            let mut chosen = None;
            let mut last_positive = 0usize;
            let mut w_chosen = 0.0;
            for (j, info) in self.lights.iter().enumerate() {
                let w = if n <= CACHE { ws[j] } else { Self::selection_weight(info, p) };
                acc += w;
                if w > 0.0 {
                    last_positive = j;
                    w_chosen = w;
                    if r < acc {
                        chosen = Some(j);
                        break;
                    }
                }
            }
            // r が丸めで total に達した場合は、重みが正の最後の光源（`w_chosen` はその重み）
            (chosen.unwrap_or(last_positive), total, Self::prob_of_weight(w_chosen, total))
        } else {
            let r = rng.next_f64() * self.light_total;
            // 光源の選択（1 次元）の後、面上の点（2 次元）は次の組の先頭から引く（Sobol の次元の表: `sampler`）
            rng.align_pair();
            let idx = cdf_search(&self.light_cdf, r).min(self.lights.len().saturating_sub(1));
            (idx, self.light_total, self.selection_prob(idx, p, self.light_total))
        };
        let info = self.lights[idx];
        let _ = total;
        // 層化なし（`uv_override` が None）のときは、元の実装と同じ順で rng から引く
        // （「選択 → 面上の点」）ので、`sample_light` はビット単位で従来どおりの挙動になる。
        let uv = uv_override.unwrap_or_else(|| (rng.next_f64(), rng.next_f64()));

        // シェープ別のサンプリングと PDF は Light に委譲する。PDF は light_pdf と同じ関数で
        // 求めるので、BSDF サンプリング側の MIS 重みと常に一致する。
        let (pos, normal) = info.light.sample(self, time, p, uv)?;
        let pdf = pdf_select * info.light.pdf_omega(self, time, p, pos, normal);
        if !(pdf > 0.0 && pdf.is_finite()) {
            return None;
        }
        let p_error = info.p_error;
        let (visible, inst_id, prim_id) = match info.light {
            Light::Sphere { idx } => {
                let s = &self.spheres[idx];
                // 球の外部からは、参照点側を向いた点だけが光源自身に隠されない（円錐サンプリングの点は常に見える）
                let to_c = s.c - p;
                let inside = to_c.dot(to_c) <= s.r * s.r;
                let visible = inside || normal.dot(p - pos) > 0.0;
                (visible, None, idx)
            }
            Light::Triangle { tri_id, inst_id, .. } => (true, Some(inst_id), tri_id),
        };
        Some(LightSample {
            position: pos,
            normal,
            emit: info.emit,
            pdf,
            p_error,
            visible,
            inst_id,
            prim_id,
        })
    }
}

#[derive(Clone, Copy)]
/// ライトが参照するジオメトリの種類（World 内のインデックス参照）。
///
/// シェープ別の発光面の幾何（面積・表面サンプリング）はこの型のメソッドに集約され、
/// CDF 構築（[`World::build_lights`]）とライトサンプリング（[`World::sample_light`]）の
/// 両方から共有される。新しい発光シェープの追加はここに 1 アームを足すだけで済む。
pub enum Light {
    Sphere { idx: usize },
    Triangle { mesh_id: usize, tri_id: usize, inst_id: usize },
}

impl Light {
    /// 発光面の表面積を返す（モーションブラー対応のため `time` に依存）。
    /// ジオメトリが見つからない場合は 0。
    fn area(&self, world: &World, time: f64) -> f64 {
        match *self {
            Light::Sphere { idx } => {
                world.spheres.get(idx).map_or(0.0, |s| 4.0 * std::f64::consts::PI * s.r * s.r)
            }
            Light::Triangle { mesh_id, tri_id, inst_id } => {
                match tri_world_verts(world, mesh_id, tri_id, inst_id, time) {
                    Some((v0, v1, v2)) => 0.5 * (v1 - v0).cross(v2 - v0).len(),
                    None => 0.0,
                }
            }
        }
    }

    /// 参照点 `from` から発光面上の点をサンプリングし、`(位置, 外向き法線)` を返す。
    /// 対応する立体角 PDF（選択確率を除く）は [`Light::pdf_omega`] が与える。
    ///
    /// - 球: `from` が球の外なら、`from` から見える円錐（立体角）を一様サンプリングする。
    ///   球の内部（境界を含む）・表面すれすれの外部なら表面積一様サンプリングにフォールバックする。
    /// - 三角形: 表面積一様サンプリング。
    /// `uv` は面上の点を選ぶ 2 次元乱数（各成分 [0,1)）。層化サンプリング（PERF-3）で外から
    /// 指定できるよう、乱数生成器そのものではなく既に引いた値を受け取る形にしてある。
    fn sample(&self, world: &World, time: f64, from: Vec3, uv: (f64, f64)) -> Option<(Vec3, Vec3)> {
        let (u, v) = uv;
        match *self {
            Light::Sphere { idx } => {
                let s = world.spheres.get(idx)?;
                match sphere_cone(s, from) {
                    Some(cone) => {
                        // 球の円錐サンプリング（PBRT v4 と同じ幾何）。cosθ を [cosθmax, 1] で一様に取る。
                        // 1 − cosθ を直接持ち、sin²θ = (1 − cosθ)(1 + cosθ) とすることで、
                        // 小さな円錐でも桁落ちせず近似も使わない（一次近似の偏りと分岐の不連続がない）。
                        let one_minus_cos = u * cone.one_minus_cos_max;
                        let cos_theta = 1.0 - one_minus_cos;
                        let sin2_theta = (one_minus_cos * (2.0 - one_minus_cos)).max(0.0);
                        // 円錐内の方向 θ に対応する球面上の点の、球中心から見た角 α
                        let cos_alpha = sin2_theta / cone.sin2_max.sqrt()
                            + cos_theta * (1.0 - sin2_theta / cone.sin2_max).max(0.0).sqrt();
                        let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();
                        let phi = std::f64::consts::TAU * v;
                        let (t, b) = orthonormal_basis(cone.axis);
                        let n = -(t * (sin_alpha * phi.cos()) + b * (sin_alpha * phi.sin()) + cone.axis * cos_alpha);
                        Some((s.c + n * s.r, n))
                    }
                    None => {
                        let z = 1.0 - 2.0 * u;
                        let r = (1.0 - z * z).max(0.0).sqrt();
                        let phi = std::f64::consts::TAU * v;
                        let n = Vec3::new(r * phi.cos(), z, r * phi.sin());
                        Some((s.c + n * s.r, n))
                    }
                }
            }
            Light::Triangle { mesh_id, tri_id, inst_id } => {
                let (v0w, v1w, v2w) = tri_world_verts(world, mesh_id, tri_id, inst_id, time)?;
                let su = u.sqrt();
                let b0 = 1.0 - su;
                let b1 = v * su;
                let b2 = 1.0 - b0 - b1;
                let pos = v0w * b0 + v1w * b1 + v2w * b2;

                let n = (v1w - v0w).cross(v2w - v0w);
                let len = n.len();
                let normal = if len > 0.0 { n / len } else { Vec3::new(0.0, 1.0, 0.0) };
                Some((pos, normal))
            }
        }
    }

    /// 参照点 `from` から発光面上の点 `pos`（法線 `normal`）への方向の立体角 PDF
    /// （ライト選択確率を除く）。[`Light::sample`] のサンプル分布と一致する。
    /// その方向がサンプルされえない（裏向き・退化）場合は 0。
    fn pdf_omega(&self, world: &World, time: f64, from: Vec3, pos: Vec3, normal: Vec3) -> f64 {
        let to_light = pos - from;
        let dist2 = to_light.dot(to_light);
        // 距離の絶対しきい値（以前は 1e-12）は使わない。小さな球の表面すれすれの参照点では
        // 正当なサンプルの多くが 1e-6 以内に落ち、シーンのスケール次第で棄却されてしまう。
        // 円錐の pdf は距離に依らず、面積由来の pdf は d² → 0 で自然に 0 になるので、
        // 方向が定義できない距離 0 だけを除く。
        if !(dist2 > 0.0) {
            return 0.0;
        }
        let wi = to_light / dist2.sqrt();
        let cos_light = normal.dot(-wi);
        match *self {
            Light::Sphere { idx } => {
                let Some(s) = world.spheres.get(idx) else { return 0.0 };
                match sphere_cone(s, from) {
                    // 外部: 見える側（cos_light > 0）の点だけがサンプルされ、円錐内で一様
                    Some(cone) => {
                        if cos_light <= 0.0 {
                            0.0
                        } else {
                            1.0 / (std::f64::consts::TAU * cone.one_minus_cos_max)
                        }
                    }
                    // 内部・表面すれすれの外部: 表面積一様。内側からは外向き法線と逆向きに見えるので |cos| を使う。
                    // 外部から裏側の点がサンプルされた場合は、NEE のシャドウレイが同じ球の手前側に
                    // 遮られて寄与 0 になるだけで、見える側の点の密度は変わらない（不偏）
                    None => {
                        let area = 4.0 * std::f64::consts::PI * s.r * s.r;
                        let c = cos_light.abs();
                        if c <= 0.0 || area <= 0.0 { 0.0 } else { dist2 / (area * c) }
                    }
                }
            }
            Light::Triangle { .. } => {
                let area = self.area(world, time);
                if cos_light <= 0.0 || area <= 0.0 { 0.0 } else { dist2 / (area * cos_light) }
            }
        }
    }
}

/// 円錐サンプリングで sin²θmax の一次近似に切り替える閾値（PBRT v4 と同じ。約 1.5°）。
/// 表面すれすれとみなす sin²θmax の下限。これより大きい（参照点が球面から相対 ~5e-13 以内、
/// 座標の数 ulp で位置関係が分解できない）と、円錐サンプリングをやめて表面積サンプリングに
/// フォールバックする。
///
/// 以前の「表面すれすれでほぼ全サンプルが棄却される」問題（相対距離 1e-9 で 99.9%）の原因は
/// 円錐そのものではなく `pdf_omega` の距離の絶対しきい値（dist² ≤ 1e-12）だった: 接点付近への
/// サンプル距離は h/cosθ 程度まで小さくなる。これを外した後は、相対距離 1e-12 まで円錐サンプリングで
/// 棄却 0・E[1/pdf] = Ω を確認している。面積サンプリングは外部から見える小さなキャップを
/// ほとんど引けず（NEE が実質働かない）ので、フォールバックは本当に分解できない距離だけに限る。
const NEAR_SURFACE_SIN2: f64 = 1.0 - 1e-12;

/// 球の外部の点から見た円錐。
struct SphereCone {
    /// 参照点から球中心への単位ベクトル
    axis: Vec3,
    sin2_max: f64,
    /// 1 − cosθmax = sin²θmax / (1 + √(1 − sin²θmax))（桁落ちのない厳密な形）
    one_minus_cos_max: f64,
}

/// `from` が球の十分に外部なら、`from` から球を見込む円錐を返す。
/// 内部（境界を含む）または表面すれすれ（sin²θmax > [`NEAR_SURFACE_SIN2`]）なら `None`
/// （呼び出し側は表面積サンプリングにフォールバックする）。
fn sphere_cone(s: &Sphere, from: Vec3) -> Option<SphereCone> {
    let to_c = s.c - from;
    let dc2 = to_c.dot(to_c);
    let r2 = s.r * s.r;
    if dc2 <= r2 {
        return None;
    }
    let sin2_max = r2 / dc2;
    if !(sin2_max <= NEAR_SURFACE_SIN2) {
        return None;
    }
    let one_minus_cos_max = sin2_max / (1.0 + (1.0 - sin2_max).sqrt());
    if one_minus_cos_max <= 0.0 {
        return None;
    }
    Some(SphereCone { axis: to_c / dc2.sqrt(), sin2_max, one_minus_cos_max })
}

/// 単位ベクトル `w` に直交する正規直交基底 (t, b)。
fn orthonormal_basis(w: Vec3) -> (Vec3, Vec3) {
    let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let t = w.cross(a).norm();
    let b = t.cross(w);
    (t, b)
}

/// 三角形のワールド空間頂点を `time` における（インスタンス変換適用後の）位置で返す。
/// 発光三角形上の点（ワールド空間）の成分ごとの誤差上界（保守的）: 頂点の変換誤差の最大と、
/// 重心座標による補間の丸め γ(9)·max|v|。
fn tri_point_error(world: &World, mesh_id: usize, tri_id: usize, inst_id: usize, time: f64) -> Vec3 {
    let (Some(mesh), Some(inst)) = (world.meshes.get(mesh_id), world.instances.get(inst_id)) else {
        return Vec3::new(0.0, 0.0, 0.0);
    };
    let Some((v0, v1, v2)) = mesh.vertices_at(tri_id, time) else { return Vec3::new(0.0, 0.0, 0.0) };
    let zero = Vec3::new(0.0, 0.0, 0.0);
    let (w0, e0) = inst.xform.apply_point_with_error(v0, zero);
    let (w1, e1) = inst.xform.apply_point_with_error(v1, zero);
    let (w2, e2) = inst.xform.apply_point_with_error(v2, zero);
    let vmax = |a: Vec3, b: Vec3, c: Vec3| Vec3::new(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z));
    vmax(e0, e1, e2) + vmax(w0.abs(), w1.abs(), w2.abs()) * gamma(9)
}

fn tri_world_verts(
    world: &World,
    mesh_id: usize,
    tri_id: usize,
    inst_id: usize,
    time: f64,
) -> Option<(Vec3, Vec3, Vec3)> {
    let mesh = world.meshes.get(mesh_id)?;
    let inst = world.instances.get(inst_id)?;
    let (v0, v1, v2) = mesh.vertices_at(tri_id, time)?;
    Some((
        inst.xform.apply_point(v0),
        inst.xform.apply_point(v1),
        inst.xform.apply_point(v2),
    ))
}

#[derive(Clone, Copy)]
/// ライト情報（放射輝度と CDF 選択重み）。
pub struct LightInfo {
    pub light: Light,
    pub emit: Color,
    pub weight: f64,
    /// この光源上の任意の点（シャッター区間全体）の成分ごとの誤差上界（`build_lights` で事前計算）
    pub p_error: Vec3,
    /// 光源選択の重み（[`World::selection_weight`]）に使う量。距離は**光源のまとまり**（発光球 1 個、または発光インスタンス 1 個の
    /// 三角形全部）の境界球 `(center, r2)` で測る。同じ矩形の 2 つの三角形は同じ距離の項を共有し、出力パワー `weight` の比が保たれる
    pub center: Vec3,
    /// 境界球の半径の二乗（距離の下限。`d²` がこれ未満なら `r2`）
    pub r2: f64,
    /// 三角形の重心と外向き法線（片面発光。参照点が裏側なら選択重み 0）。球は `one_sided = false`
    pub plane_p: Vec3,
    pub normal: Vec3,
    pub one_sided: bool,
}

/// デルタ光源: 発光が面積ゼロに集中した光源（点・平行・スポット）。
///
/// 方向の確率密度が Dirac のデルタなので有限の立体角 pdf として書けず、次の性質を持つ:
/// - BSDF サンプリングでは絶対に当たらない → **MIS を適用しない**（NEE の寄与に重み 1 で足す。pdf でも割らない）
/// - 幾何を持たないので `light_pdf` の対象ではない（[`Light`] とは別の型にして、`inst_id: None` = 球という
///   既存の暗黙の約束に紛れ込ませない）
///
/// **面光源の CDF に入れず、NEE のたびに全部を順に評価する**: 面積ゼロなので「サンプリング」が要らず
/// （乱数を引かない）、CDF に入れても選択の分散が増えるだけ。デルタ光源が 0 個ならループが空回りするだけで
/// 乱数の消費列が変わらない。光源が数個の想定で、多数（数百〜）あるシーンでは NEE が光源数に比例して重くなる
/// （その規模は想定外）。
#[derive(Clone, Copy, Debug)]
pub enum DeltaLight {
    /// 点光源。`intensity` は放射強度 I [W/sr]。距離 d の点での放射照度は I/d²
    Point { position: Vec3, intensity: Color },
    /// 平行光源（太陽）。`direction` は光の進む向き（光源 → シーン）。`irradiance` は光に垂直な面での
    /// 放射照度 E [W/m²]。距離減衰なし
    Directional { direction: Vec3, irradiance: Color },
    /// スポットライト。`direction` は光軸（光の進む向き）。光軸からの角度 θ が `cutoff_angle`（ラジアン）以上で 0、
    /// `beam_width` 以内は減衰なし、その間は **θ に対して線形**に落ちる: 係数 = `(cutoff − θ)/(cutoff − beam)`。
    /// Mitsuba 3 の spot（`src/emitters/spot.cpp` の `falloff_curve`）と同じ規約で、ソースで確認済み。
    /// （PBRT v4 の spot は余弦上の smoothstep で、曲線が異なる。）`beam_width <= cutoff_angle`
    Spot { position: Vec3, direction: Vec3, intensity: Color, cutoff_angle: f64, beam_width: f64 },
}

/// [`DeltaLight::sample_at`] の結果。
#[derive(Clone, Copy, Debug)]
pub struct DeltaLightHit {
    /// 評価点から光源へ向かう単位ベクトル
    pub wi: Vec3,
    /// 光源の位置（平行光源は `None`）。シャドウレイの終点は、原点をずらした後にここまでの距離を測り直して決める
    pub position: Option<Vec3>,
    /// 評価点に届く放射照度に相当する量（点・スポット: `I·falloff/d²`、平行: `E`）。BSDF と cos を掛ければ寄与になる
    pub value: Color,
}

impl DeltaLight {
    /// 評価点 `p` から見たこの光源。寄与 0（スポットの外・距離 0）なら `None`。乱数は引かない。
    pub fn sample_at(&self, p: Vec3) -> Option<DeltaLightHit> {
        match *self {
            DeltaLight::Point { position, intensity } => Self::point_like(p, position, intensity),
            DeltaLight::Directional { direction, irradiance } => {
                Some(DeltaLightHit { wi: -direction.norm(), position: None, value: irradiance })
            }
            DeltaLight::Spot { position, direction, intensity, cutoff_angle, beam_width } => {
                let to = position - p;
                let d2 = to.dot(to);
                if !(d2 > 0.0) {
                    return None;
                }
                let wi = to / d2.sqrt();
                // 光軸と「光源から評価点への向き（-wi）」のなす角の余弦
                let cos_theta = (-wi).dot(direction.norm());
                let (cos_cut, cos_beam) = (cutoff_angle.cos(), beam_width.cos());
                let falloff = if cos_theta <= cos_cut {
                    return None;
                } else if cos_theta >= cos_beam {
                    1.0
                } else {
                    // Mitsuba 3: 角度に対する線形の傾斜（cos ではなく acos で測る）
                    (cutoff_angle - cos_theta.min(1.0).acos()) / (cutoff_angle - beam_width)
                };
                Self::point_like(p, position, intensity * falloff)
            }
        }
    }

    fn point_like(p: Vec3, position: Vec3, intensity: Color) -> Option<DeltaLightHit> {
        let to = position - p;
        let d2 = to.dot(to);
        if !(d2 > 0.0) {
            return None;
        }
        Some(DeltaLightHit { wi: to / d2.sqrt(), position: Some(position), value: intensity / d2 })
    }
}

#[derive(Clone, Copy)]
/// ライトサンプル結果（位置・法線・放射輝度・PDF）。
pub struct LightSample {
    /// ライト表面上のサンプル位置
    pub position: Vec3,
    /// サンプル位置の法線
    pub normal: Vec3,
    /// 放射輝度
    pub emit: Color,
    /// 参照点での立体角 PDF
    pub pdf: f64,
    /// サンプル位置の成分ごとの誤差上界（シャドウレイの終点を光源面の誤差の箱の外へずらすのに使う）
    pub p_error: Vec3,
    /// 参照点からこのサンプル点が光源自身に隠されずに見えるか。球の外部からの面積フォールバック
    /// （表面すれすれ）では裏側の点も引くので false になりうる（寄与 0）。
    pub visible: bool,
    /// サンプルした発光プリミティブ（`Hit` と同じ規約: 球なら `inst_id = None`・`prim_id` は球の番号、
    /// 三角形なら `inst_id = Some`・`prim_id` はメッシュ内の三角形番号）。シャドウレイの判定で、
    /// 光源自身へのヒットを遮蔽物と数えないために使う（必須: 球光源では交差の t の誤差上界が終点の
    /// p_error を超えうるので、ずらした終点より手前で光源面にヒットすることがある）。
    pub inst_id: Option<usize>,
    pub prim_id: usize,
}

impl LightSample {
    /// `hit` がこのサンプルの発光プリミティブ自身へのヒットか。
    pub fn is_light_itself(&self, hit: &Hit) -> bool {
        hit.inst_id == self.inst_id && hit.prim_id == self.prim_id
    }
}

/// テスト用のメッシュ生成（頂点法線つき）。スムーズシェーディングの検証で world / integrator の
/// 両方から使う。
#[cfg(test)]
pub(crate) mod test_meshes {
    use super::*;
    use crate::obj_loader::MeshData;

    /// z=0 平面の四角形（2 三角形、面法線 +z）。頂点法線を `ns` で指定できる。
    /// 幾何法線とシェーディング法線を大きく食い違わせた「1 枚ポリゴン」を作るのに使う。
    pub fn tilted_quad(half: f64, ns: Vec3, mat_id: usize) -> MeshData {
        let v = |x: f64, y: f64| Vec3::new(x, y, 0.0);
        let tris = vec![
            Triangle::new_static(v(-half, -half), v(half, -half), v(half, half), mat_id),
            Triangle::new_static(v(-half, -half), v(half, half), v(-half, half), mat_id),
        ];
        MeshData { tris, vn: vec![ns.norm()], tri_vn: vec![[0, 0, 0], [0, 0, 0]], uv: Vec::new(), tri_uv: Vec::new(), motion: Vec::new() }
    }

    /// 経度 `nu` × 緯度 `nv` の UV 球。頂点法線は解析的な法線（中心からの単位ベクトル）。
    /// `smooth = false` なら頂点法線を付けない（面法線だけの同一形状）。
    pub fn uv_sphere(center: Vec3, radius: f64, nu: usize, nv: usize, mat_id: usize, smooth: bool) -> MeshData {
        let mut pos: Vec<Vec3> = Vec::new();
        let at = |iu: usize, iv: usize| -> Vec3 {
            let theta = std::f64::consts::PI * (iv as f64) / (nv as f64);
            let phi = 2.0 * std::f64::consts::PI * (iu as f64) / (nu as f64);
            Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
        };
        for iv in 0..=nv {
            for iu in 0..nu {
                pos.push(at(iu, iv));
            }
        }
        let idx = |iu: usize, iv: usize| iv * nu + (iu % nu);
        let mut tris = Vec::new();
        let mut tri_vn = Vec::new();
        let push = |a: usize, b: usize, c: usize, tris: &mut Vec<Triangle>, tri_vn: &mut Vec<[u32; 3]>, pos: &Vec<Vec3>| {
            let p = |i: usize| center + pos[i] * radius;
            tris.push(Triangle::new_static(p(a), p(b), p(c), mat_id));
            tri_vn.push([a as u32, b as u32, c as u32]);
        };
        for iv in 0..nv {
            for iu in 0..nu {
                let (a, b, c, d) = (idx(iu, iv), idx(iu + 1, iv), idx(iu + 1, iv + 1), idx(iu, iv + 1));
                push(a, b, c, &mut tris, &mut tri_vn, &pos);
                push(a, c, d, &mut tris, &mut tri_vn, &pos);
            }
        }
        // 頂点法線 = 単位球面上の位置（解析的な法線）
        let vn = pos.clone();
        if smooth { MeshData { tris, vn, tri_vn, uv: Vec::new(), tri_uv: Vec::new(), motion: Vec::new() } } else { MeshData::flat(tris) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::uniform_sphere_dir;
    use crate::geometry::Sphere;

    /// z=0 の [0,1]² の板（UV = xy）に、左半分が不透明・右半分が透明の 2x1 マスクを貼ったワールド。
    /// 板の奥 z=-1 に不透明の床（マスク無し）を置く。
    fn masked_plate_world(mask_values: [u8; 2]) -> World {
        let mut world = World::new();
        let (a, b, c, d) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let tris = vec![Triangle::new_static(a, b, c, 0), Triangle::new_static(a, c, d, 0)];
        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri_uv = vec![[0, 1, 2], [0, 2, 3]];
        let mask = Arc::new(AlphaMask::from_u8(2, 1, mask_values.to_vec(), crate::texture::Wrap::Clamp));
        let data = MeshData::with_uv(tris, uv, tri_uv);
        world.add_mesh_data_instance_with_alpha(data, vec![mask], vec![(1, 1.0); 2], Transform::identity());
        // 奥の床（マスク無し）
        let f = |x: f64, y: f64| Vec3::new(x, y, -1.0);
        world.add_mesh_instance(
            vec![
                Triangle::new_static(f(-5.0, -5.0), f(5.0, -5.0), f(5.0, 5.0), 1),
                Triangle::new_static(f(-5.0, -5.0), f(5.0, 5.0), f(-5.0, 5.0), 1),
            ],
            Transform::identity(),
            None,
        );
        world
    }

    /// 不透明部分ではレイが止まり、透明部分は素通りして奥の面に当たる（シャドウレイも同じ経路）。
    #[test]
    fn alpha_mask_transparent_texels_are_passed_through() {
        // マスク: 左 255（不透明）、右 0（透明）。クランプなので 2 テクセルの境界は u=0.5 付近で補間される
        let world = masked_plate_world([255, 0]);
        let down = |x: f64, y: f64| Ray { o: Vec3::new(x, y, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let opaque = world.hit(down(0.1, 0.5), 0.0, 1e30).unwrap();
        assert_eq!(opaque.mat_id, 0, "不透明部分で板に当たるはず");
        assert!((opaque.t - 2.0).abs() < 1e-9);
        let through = world.hit(down(0.9, 0.5), 0.0, 1e30).unwrap();
        assert_eq!(through.mat_id, 1, "透明部分は素通りして床に当たるはず");
        assert!((through.t - 3.0).abs() < 1e-9);
        // シャドウレイ: 板の手前から奥の点へ。不透明側は遮られ、穴側は抜ける
        let occluded = |x: f64| {
            let r = Ray { o: Vec3::new(x, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            world.hit(r, 0.0, 2.9).is_some() // 床（t=3）の手前まで
        };
        assert!(occluded(0.1));
        assert!(!occluded(0.9), "穴を通る光が遮られている");
    }

    /// 透明部分を捨てた後の探索で、同じ板（および裏面側から）を再び拾わない = 自己交差しない。
    /// 板の透明部分の上に原点を置き、板を貫いて上下どちらへ撃っても板自身には当たらない。
    #[test]
    fn alpha_mask_rejected_surface_is_not_hit_again() {
        let world = masked_plate_world([255, 0]);
        for &dz in &[-1.0, 1.0] {
            // 原点は板の面上（透明部分）
            let r = Ray { o: Vec3::new(0.9, 0.5, 0.0), d: Vec3::new(0.0, 0.0, dz), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30);
            match (dz < 0.0, h) {
                (true, Some(h)) => assert_eq!(h.mat_id, 1),
                (false, None) => {}
                (_, h) => panic!("板に自己交差した: {:?}", h.map(|h| (h.mat_id, h.t))),
            }
        }
    }

    /// しきい値ちょうどの扱いは決定的（`alpha >= 0.5` が不透明）。8bit で 127 は透明、128 は不透明。
    #[test]
    fn alpha_mask_threshold_is_inclusive_and_deterministic() {
        for (v, expect_plate) in [(127u8, false), (128u8, true)] {
            let world = masked_plate_world([v, v]);
            let r = Ray { o: Vec3::new(0.5, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30).unwrap();
            assert_eq!(h.mat_id == 0, expect_plate, "mask={}", v);
        }
    }

    // ---- PERF-1: シャドウレイの any-hit 化（`World::occluded`） ----

    /// `World::occluded` は `tmax` を守る（`tmax` を超えて存在する遮蔽物を遮蔽と数えない）。
    /// 1 つのメッシュに、真の tmax 内にある遮蔽物（近い三角形）と、tmax の外にある別の遮蔽物
    /// （遠い三角形、配列の先頭に置いて any-hit の探索順で先に見つかりやすくしてある）を同居させる。
    /// `tmax` を無視して探すと、遠い三角形を先に見つけて「範囲外だから」と探索を打ち切ってしまい、
    /// 本来見つかるはずの近い三角形を取りこぼして「遮蔽なし」と誤判定しうる。
    #[test]
    fn occluded_respects_tmax_even_when_a_farther_triangle_is_probed_first() {
        let far = Triangle::new_static(Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, -1.0, 10.0), Vec3::new(0.0, 1.0, 10.0), 0);
        let near = Triangle::new_static(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(0.0, 1.0, 1.0), 0);
        let mut world = World::new();
        // far を先に入れる（1 リーフに収まる三角形数なので、探索順に影響しうる）
        world.add_mesh_instance(vec![far, near], Transform::identity(), None);
        let ray = Ray { o: Vec3::new(0.0, 0.0, 0.0), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // tmax=2: near（t=1）は範囲内、far（t=10）は範囲外
        assert!(world.occluded(ray, 0.0, 2.0, None), "範囲内の近い三角形を遮蔽として見つけられるはず");
        // tmax=0.5: どちらも範囲外
        assert!(!world.occluded(ray, 0.0, 0.5, None), "どちらも範囲外なら遮蔽なしのはず");
        // 対照: tmax=20 なら両方範囲内（当然遮蔽）
        assert!(world.occluded(ray, 0.0, 20.0, None));
    }

    /// `World::occluded` の `tmin` 境界（`t_world <= tmin` で自己交差とみなして再探索する規律）と、
    /// 再探索ループ（インスタンスの近い面が帯の中で棄却されたとき、奥の面を諦めずに探す）が、
    /// `World::hit` と同じく効いていることを確認する（PERF-1 追補: check がこの領域の穴を発見）。
    ///
    /// `s=1`（恒等に近い変換）にして、手前の板のワールド距離がちょうど `tmin` になるようレイ原点を
    /// 置く。正しい実装なら「ちょうど `tmin`」は自己交差とみなして棄却・再探索し、奥の板（t≈1+tmin）
    /// を見つける。`tmax` を奥の板より手前に絞れば、手前の板を遮蔽物と誤認しない限り「遮蔽なし」になる
    /// はず — これが `<=`→`<` の境界緩和（手前の板を誤って採用してしまう）を検出する。
    /// `tmax` を奥の板まで届く値にすれば「遮蔽あり」になるはずで、これが再探索ループの欠落
    /// （奥の板を探しにいかない）を検出する。
    #[test]
    fn occluded_treats_exact_tmin_as_self_intersection_and_retries_for_the_far_plate() {
        let tmin = 1e-4;
        let world = two_plates_world(1.0, 1.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -tmin), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 前提: 手前の板のワールド距離がちょうど tmin であること（境界値のすり替えを検出するための土台）
        let h = world.hit(r, 0.0, 1e30).expect("near plate must be hittable without a tmin floor");
        assert_eq!(h.t.to_bits(), tmin.to_bits(), "test setup: near-plate t must equal tmin exactly");

        // 手前の板だけが範囲内（tmax=0.5 < 奥の板の t≈1）でも、手前の板はちょうど tmin なので
        // 自己交差とみなし、遮蔽物として数えてはいけない
        assert!(
            !world.occluded(r, tmin, 0.5, None),
            "ちょうど tmin の手前の板を遮蔽物として採用してしまった（tmin 境界の緩和を検出）"
        );
        // tmax を奥の板まで伸ばせば、再探索で見つかって遮蔽ありになるはず
        assert!(
            world.occluded(r, tmin, 2.0, None),
            "再探索で奥の板を見つけられなかった（再探索ループの欠落を検出）"
        );
    }

    /// 再探索の上限回数（`INSTANCE_RETRY_LIMIT`）に達しても無限ループせず、諦めて「遮蔽なし」を返す
    /// （`World::hit` の `retry_limit_terminates_on_many_faces_inside_tmin_band` と同じ状況を
    /// `occluded` でも確認する）。
    #[test]
    fn occluded_terminates_when_the_retry_limit_is_reached() {
        let tmin = 1e-4;
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        let mut tris = Vec::new();
        for k in 0..20 {
            tris.extend(quad(k as f64 * 1e-12));
        }
        tris.extend(quad(0.5));
        world.add_mesh_instance(tris, Transform::identity(), None);
        let r = Ray { o: Vec3::new(0.1, 0.1, -tmin * (1.0 - 1e-6)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 終了すること自体が要件（タイムアウトしないこと）。tmax を極端に大きくしても壊れない
        let _ = world.occluded(r, tmin, 1e30, None);
    }

    /// `World::occluded` は `tmax` の境界でも `World::hit` と同じ規約（`t_world < closest` 相当。
    /// ちょうど `tmax` の面は範囲外）に従う。
    #[test]
    fn occluded_excludes_a_surface_exactly_at_tmax() {
        let world = two_plates_world(1.0, 1.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -0.5), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 手前の板までの距離はちょうど 0.5
        let h = world.hit(r, 0.0, 1e30).expect("hit");
        assert_eq!(h.t, 0.5);
        assert!(!world.occluded(r, 0.0, 0.5, None), "ちょうど tmax の面は範囲外のはず（World::hit と同じ規約）");
        assert!(world.occluded(r, 0.0, 0.5 + 1e-9, None), "tmax をわずかに超えれば範囲内のはず");
    }

    /// `World::occluded` は「解決済みの `skip` に一致する三角形だけ」を除外し、アルファ不透明な三角形は
    /// `skip` の有無に関わらず遮蔽として扱う（除外条件の取り違えが無いことの確認）。
    #[test]
    fn occluded_skip_and_alpha_combine_correctly() {
        let world = masked_plate_world([255, 0]);
        let opaque_ray = Ray { o: Vec3::new(0.1, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let h = world.hit(opaque_ray, 0.0, 1e30).expect("hit");
        // skip は「指定した三角形番号だけ」を除外する。板は 2 枚の三角形でできているので、
        // 当たっていない方（prim_id が違う方）を指定しても、当たった三角形はまだ遮蔽扱いのはず
        let other_prim = 1 - h.prim_id;
        assert!(
            world.occluded(opaque_ray, 0.0, 2.9, Some((h.inst_id, other_prim))),
            "当たっていない三角形を skip しても不透明側はまだ遮蔽するはず"
        );
        // 当たった三角形そのものを skip すれば、遮蔽されない
        assert!(!world.occluded(opaque_ray, 0.0, 2.9, Some((h.inst_id, h.prim_id))), "skip した三角形自身は遮蔽しないはず");
    }

    /// `World::occluded` は `World::hit(...).is_some()` と常に一致する（遮蔽の有無について。
    /// アルファマスク付きの板で、不透明部分は遮蔽・透明部分は素通しになることを any-hit 経路でも確認する）。
    #[test]
    fn occluded_agrees_with_hit_through_alpha_mask() {
        let world = masked_plate_world([255, 0]);
        let down = |x: f64, y: f64| Ray { o: Vec3::new(x, y, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        // 不透明側: 板で遮蔽される（奥の床 t=3 より手前）
        assert!(world.occluded(down(0.1, 0.5), 0.0, 2.9, None), "不透明部分は遮蔽するはず");
        assert_eq!(world.occluded(down(0.1, 0.5), 0.0, 2.9, None), world.hit(down(0.1, 0.5), 0.0, 2.9).is_some());
        // 透明側: 板は素通しで、奥の床（t=3）より手前の tmax=2.9 では何にも当たらない
        assert!(!world.occluded(down(0.9, 0.5), 0.0, 2.9, None), "透明部分は遮蔽しないはず");
        assert_eq!(world.occluded(down(0.9, 0.5), 0.0, 2.9, None), world.hit(down(0.9, 0.5), 0.0, 2.9).is_some());
        // 透明側でも、奥の床まで届く tmax なら床で遮蔽される
        assert!(world.occluded(down(0.9, 0.5), 0.0, 3.5, None), "奥の床までは届くはず");
    }

    /// 光源自身（三角形光源）への交差は `World::occluded` の `skip` で遮蔽と数えない。
    /// ランダムサンプリングでの再現（元の `shadow_ray_does_not_self_hit_sampled_light` と同じ配置）に加え、
    /// 光源面をわざと突き抜ける tmax を渡して「必ず光源自身に当たる」状況を作り、`skip` の有無で
    /// 結果が変わることを直接確認する（除外を外すミューテーションで確実に落ちる）。
    #[test]
    fn occluded_excludes_the_sampled_light_itself() {
        use crate::transform::Transform;
        let mut world = World::new();
        let m = |x: f64, z: f64| Vec3::new(0.35 * x, 1.98, 0.35 * z);
        let tris = vec![
            Triangle::new_static(m(-1.0, -1.0), m(1.0, -1.0), m(1.0, 1.0), 0),
            Triangle::new_static(m(-1.0, -1.0), m(1.0, 1.0), m(-1.0, 1.0), 0),
        ];
        let inst_id = world.add_mesh_instance(tris, Transform::identity(), None);
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.6, 3.9, 2.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 1.4, -1.0);
        let mut sampled = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                sampled += 1;
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                let ray = Ray { o: p, d: seg / seg.len(), time: 0.0 };
                assert!(
                    !world.occluded(ray, 0.0, seg.len(), Some((ls.inst_id, ls.prim_id))),
                    "光源自身を遮蔽と数えてしまった"
                );
                assert_eq!(ls.inst_id, Some(inst_id));
            }
        }
        assert!(sampled > 0, "no light samples drawn");

        // 光源面を確実に突き抜ける tmax（真の距離 + 大きめの余白）で、必ず光源自身に当たる状況を作る
        let ls = world.sample_light(&mut rng, 0.0, p).expect("light sample");
        let to_light = ls.position - p;
        let dist = to_light.len();
        let ray = Ray { o: p, d: to_light / dist, time: 0.0 };
        assert!(!world.occluded(ray, 0.0, dist + 0.5, Some((ls.inst_id, ls.prim_id))), "skip 付きなら光源自身は遮蔽ではない");
        assert!(world.occluded(ray, 0.0, dist + 0.5, None), "skip 無しなら光源自身への交差も遮蔽扱いのはず（比較用）");
    }

    /// 光源自身への交差の除外は球光源でも効く（`inst_id = None`）。`default.xml` 実測（S3）で、
    /// この除外を外すと画像が約 6% 暗くなったのと同じ配置（床の真上の球光源）で確認する。
    /// ランダムサンプリングに加え、球面を確実に突き抜ける tmax で「必ず光源自身に当たる」状況も直接確認する
    /// （除外を外すミューテーションで確実に落ちる）。
    #[test]
    fn occluded_excludes_the_sampled_sphere_light_itself() {
        let mut world = World::new();
        let light_idx = 0usize;
        world.spheres.push(Sphere { c: Vec3::new(0.0, 3.0, 0.0), r: 1.0, mat_id: 0 });
        // 床（大きな四角形、光源の真下）
        let f = |x: f64, z: f64| Vec3::new(x, 0.0, z);
        world.add_mesh_instance(
            vec![
                Triangle::new_static(f(-5.0, -5.0), f(5.0, -5.0), f(5.0, 5.0), 1),
                Triangle::new_static(f(-5.0, -5.0), f(5.0, 5.0), f(-5.0, 5.0), 1),
            ],
            Transform::identity(),
            None,
        );
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.0, 4.0, 4.0) }, Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        world.build_lights(&mats);

        let mut rng = Rng::new(9);
        let p = Vec3::new(0.0, 0.0, 0.0);
        let mut sampled = 0;
        let mut last_ls = None;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                if !ls.visible {
                    continue;
                }
                sampled += 1;
                assert_eq!(ls.inst_id, None);
                assert_eq!(ls.prim_id, light_idx);
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                let ray = Ray { o: p, d: seg / seg.len(), time: 0.0 };
                assert!(
                    !world.occluded(ray, 0.0, seg.len(), Some((ls.inst_id, ls.prim_id))),
                    "球光源自身を遮蔽と数えてしまった"
                );
                last_ls = Some(ls);
            }
        }
        assert!(sampled > 0, "no light samples drawn");

        // 球面を確実に突き抜ける tmax で、必ず光源自身に当たる状況を作る
        let ls = last_ls.expect("at least one visible sample");
        let to_light = ls.position - p;
        let dist = to_light.len();
        let ray = Ray { o: p, d: to_light / dist, time: 0.0 };
        assert!(!world.occluded(ray, 0.0, dist + 0.5, Some((ls.inst_id, ls.prim_id))), "skip 付きなら球光源自身は遮蔽ではない");
        assert!(world.occluded(ray, 0.0, dist + 0.5, None), "skip 無しなら球光源自身への交差も遮蔽扱いのはず（比較用）");
    }

    fn emissive_sphere_world(c: Vec3, r: f64) -> World {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        world.spheres.push(Sphere { c, r, mat_id: 0 });
        world.build_lights(&mats);
        world
    }

    /// 光源が多数・混在（球 5 個 + 向きの違う矩形 2 枚）のワールド。矩形の法線は +z（cross(v1−v0, v2−v0)）。
    fn many_lights_world() -> World {
        use crate::world::test_meshes::tilted_quad;
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        for (c, r) in [
            (Vec3::new(-6.0, 1.0, 2.0), 0.4),
            (Vec3::new(4.0, 0.5, -3.0), 1.5),
            (Vec3::new(0.0, 5.0, 0.0), 0.05),
            (Vec3::new(20.0, 2.0, 20.0), 3.0),
            (Vec3::new(0.0, 0.0, 0.0), 0.7),
        ] {
            world.add_sphere(Sphere { c, r, mat_id: 0 });
        }
        // +z を向く矩形と、y 軸まわりに 180° 回して −z を向く矩形
        world.add_mesh_data_instance(tilted_quad(1.0, Vec3::new(0.0, 0.0, 1.0), 0), Transform::translate(Vec3::new(2.0, 1.0, 6.0)), None);
        world.add_mesh_data_instance(
            tilted_quad(2.0, Vec3::new(0.0, 0.0, 1.0), 0),
            Transform::translate(Vec3::new(-3.0, 1.0, -6.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 180.0)),
            None,
        );
        world.build_lights(&mats);
        assert!(world.light_select != LightSelect::Power, "テスト設定: 光源のまとまりが複数なので重み付けになるはず");
        // 既定は線形（7 個）。光源 BVH の経路もこのワールドで通す
        world.set_light_select(LightSelect::Bvh);
        world
    }

    /// **MIS の前提**: `sample_light` が返した `pdf` は、同じ光源・同じ点に対する `light_pdf` とビット単位で一致する
    /// （選択確率が参照点に依存しても）。光源 1 個 / 多数 / 裏側 / 距離 0 近傍 / 重み 0 の光源を含む。
    #[test]
    fn light_pdf_is_bit_identical_to_the_sampled_pdf_with_position_dependent_selection() {
        let mut world = many_lights_world();
        let mut rng = Rng::new(77);
        // 既定（線形）と光源 BVH の両方でビット一致を確かめる
        for mode in [LightSelect::Linear, LightSelect::Bvh] {
        world.set_light_select(mode);
        let mut points = vec![
            Vec3::new(0.0, 0.0, 0.0),           // 球 [4] の中心（距離 0）
            Vec3::new(0.0, 5.0, 1e-9),          // 小さな球のすぐそば
            Vec3::new(2.0, 1.0, 6.0 + 1e-9),    // +z 矩形の面上すれすれ（表側）
            Vec3::new(2.0, 1.0, 6.0 - 1e-3),    // 同・裏側（この矩形の重みは 0）
            Vec3::new(-3.0, 1.0, -6.0 - 0.5),   // −z を向く矩形の裏側（−z 側 = 表側なので実は表）
            Vec3::new(-3.0, 1.0, -5.0),         // 同・裏側
            Vec3::new(300.0, 100.0, -200.0),    // 遠方
        ];
        for _ in 0..40 {
            points.push(Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 12.0 - 3.0, rng.next_f64() * 40.0 - 20.0));
        }
        let mut checked = 0;
        for from in points {
            for _ in 0..200 {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                let hit = Hit {
                    t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0,
                    prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0),
                };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert_eq!(pdf.to_bits(), ls.pdf.to_bits(), "from {:?}: light_pdf {} != sample pdf {}", from, pdf, ls.pdf);
                checked += 1;
            }
        }
        assert!(checked > 3000, "{mode:?}: too few samples: {checked}");
        }
    }

    /// 光源が 64 個を超える（重みを配列に持てず 2 回に分けて計算する経路）ワールドでも、`light_pdf` とビット一致する。
    #[test]
    fn light_pdf_is_bit_identical_when_there_are_more_lights_than_the_weight_cache() {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(2.0, 2.0, 2.0) }];
        let mut rng = Rng::new(3);
        for _ in 0..150 {
            let c = Vec3::new(rng.next_f64() * 30.0 - 15.0, rng.next_f64() * 5.0, rng.next_f64() * 30.0 - 15.0);
            world.add_sphere(Sphere { c, r: 0.1 + rng.next_f64() * 0.4, mat_id: 0 });
        }
        world.build_lights(&mats);
        // 線形の重み付け（64 個超で重みを配列に持てない経路）を強制して通す
        world.force_light_weighting(true);
        assert!(world.lights.len() > 64);
        let mut checked = 0;
        for _ in 0..300 {
            let from = Vec3::new(rng.next_f64() * 30.0 - 15.0, rng.next_f64() * 6.0 - 1.0, rng.next_f64() * 30.0 - 15.0);
            let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
            let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
            assert_eq!(world.light_pdf(from, 0.0, &hit).to_bits(), ls.pdf.to_bits());
            checked += 1;
        }
        assert!(checked > 200);
    }

    /// 選択確率は参照点ごとに全光源で足して 1（線形の重み付けも光源 BVH も。木が正しい直接の証拠）、
    /// 裏側の片面発光は選ばれず（確率 0）、近い光源ほど選ばれやすい。
    #[test]
    fn selection_probabilities_sum_to_one_and_respect_orientation_and_distance() {
        let mut world = many_lights_world();
        for mode in [LightSelect::Linear, LightSelect::Bvh] {
            world.set_light_select(mode);
            let mut rng = Rng::new(5);
            let mut full = 0;
            for _ in 0..300 {
                let from = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 12.0 - 3.0, rng.next_f64() * 40.0 - 20.0);
                let sum: f64 = (0..world.lights.len()).map(|i| world.light_selection_prob(i, from)).sum();
                assert!(sum <= 1.0 + 1e-12, "{mode:?}: sum {sum}");
                if (sum - 1.0).abs() < 1e-12 {
                    full += 1;
                } else {
                    assert!(sum == 0.0 || mode == LightSelect::Bvh, "{mode:?}: sum {sum}");
                }
            }
            // 光源 BVH は向きの円錐が保守的なので、裏側の矩形だけを含むノードを引いて「行き止まり」になる分だけ総和が 1 に満たない点がある
            // （その確率は無駄になるだけで、選ばれる光源の確率は変わらない = 不偏）。球だけの構成は下のテストで厳密に 1
            assert!(full > 240, "{mode:?}: only {full} of 300 points sum to 1");
            // +z を向く矩形（インスタンス 0）の裏側からは、その 2 枚の三角形の選択確率が 0
            let behind = Vec3::new(2.0, 1.0, 3.0);
            for (i, info) in world.lights.iter().enumerate() {
                if let Light::Triangle { inst_id: 0, .. } = info.light {
                    assert_eq!(world.light_selection_prob(i, behind), 0.0, "{mode:?}");
                }
            }
            // 小さな球 [2]（0,5,0）は、すぐそばのほうが遠くからより選ばれやすい（出力パワーは同じ）
            let i_small = world.sphere_light_id[2].unwrap();
            assert!(world.light_selection_prob(i_small, Vec3::new(0.0, 5.3, 0.0)) > 5.0 * world.light_selection_prob(i_small, Vec3::new(0.0, 5.0, 12.0)), "{mode:?}");
        }
    }

    /// 光源が球だけ（向きの円錐が全方向で行き止まりが無い）の 2 / 33 / 150 / 500 個で、光源 BVH の選択確率の総和が
    /// **すべての参照点で 1**（木が正しい直接の証拠）、かつ `sample_light` の pdf と `light_pdf` がビット一致する。
    #[test]
    fn light_bvh_probabilities_sum_to_exactly_one_for_sphere_lights() {
        for n in [2usize, 33, 150, 500] {
            let mut world = World::new();
            let mats = vec![Material::DiffuseLight { emit: Color::new(2.0, 1.0, 3.0) }];
            let mut rng = Rng::new(n as u64);
            for _ in 0..n {
                let c = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 6.0, rng.next_f64() * 40.0 - 20.0);
                world.add_sphere(Sphere { c, r: 0.05 + rng.next_f64() * 0.6, mat_id: 0 });
            }
            world.build_lights(&mats);
            world.set_light_select(LightSelect::Bvh);
            for _ in 0..200 {
                // 光源の中・ごく近く・遠方を含む
                let from = match (rng.next_f64() * 4.0) as usize {
                    0 => world.spheres[(rng.next_f64() * n as f64) as usize % n].c + Vec3::new(1e-9, 0.0, 0.0),
                    1 => Vec3::new(rng.next_f64() * 1e4, rng.next_f64() * 1e4, rng.next_f64() * 1e4),
                    _ => Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 8.0 - 1.0, rng.next_f64() * 40.0 - 20.0),
                };
                let sum: f64 = (0..n).map(|i| world.light_selection_prob(i, from)).sum();
                assert!((sum - 1.0).abs() < 1e-12, "n={n} from {from:?}: sum {sum}");
                if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                    let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                    assert_eq!(world.light_pdf(from, 0.0, &hit).to_bits(), ls.pdf.to_bits(), "n={n}");
                }
            }
        }
    }

    /// 枝刈りなし（インスタンス BVH に tmax = 1e30）の参照実装。tmin の写像と再探索の規則は World::hit と同じ。
    /// 枝刈り距離の誤りをビット単位で検出するために使う。
    fn hit_without_instance_pruning(world: &World, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut closest = tmax;
        let mut best: Option<Hit> = None;
        for (inst_id, inst) in world.instances.iter().enumerate() {
            let mesh = &world.meshes[inst.mesh_id];
            let (o_obj, o_err) = inst.xform.apply_point_inv_with_error_linf(r.o);
            let d_raw = inst.xform.apply_vec_inv(r.d);
            let d_len = d_raw.len().max(1e-30);
            let d_obj = d_raw / d_len;
            let r_obj = Ray { o: o_obj + d_obj * (d_obj.l1() * o_err), d: d_obj, time: r.time };
            // World::hit と同じ tmin の写像・再探索の規則で、tmax だけ無制限（枝刈りなし）
            let mut search_from = tmin * d_len * (1.0 - 1e-9);
            for _ in 0..=INSTANCE_RETRY_LIMIT {
                let Some(h_obj) = mesh.hit(r_obj, search_from, 1e30) else { break };
                let (p_world, p_error) = inst.xform.apply_point_with_error(h_obj.p, h_obj.p_error);
                let t_world = (p_world - r.o).dot(r.d);
                if t_world <= tmin {
                    search_from = h_obj.t.next_up();
                    continue;
                }
                if t_world < closest {
                    closest = t_world;
                    let ng = inst.xform.apply_normal(h_obj.ng);
                    let ns = if h_obj.is_smooth() {
                        face_forward(inst.xform.apply_normal(h_obj.ns), ng)
                    } else {
                        ng
                    };
                    best = Some(Hit {
                        t: t_world,
                        p: p_world,
                        ng,
                        ns,
                        mat_id: inst.mat_override.unwrap_or(h_obj.mat_id),
                        prim_id: h_obj.prim_id,
                        inst_id: Some(inst_id),
                        p_error,
                        bary: h_obj.bary,
                        uv: h_obj.uv,
                    });
                }
                break;
            }
        }
        for (idx, s) in world.spheres.iter().enumerate() {
            if let Some(mut h) = s.hit(r, tmin, closest) {
                h.prim_id = idx;
                closest = h.t;
                best = Some(h);
            }
        }
        best
    }

    /// 2 つの Hit がビット単位で等しいか（参照実装との比較用）。
    fn assert_same_hit(a: Option<Hit>, b: Option<Hit>, what: &str) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let bits = |v: Vec3| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "{}: t {} vs {}", what, a.t, b.t);
                assert_eq!(bits(a.p), bits(b.p), "{}: p", what);
                assert_eq!(bits(a.ng), bits(b.ng), "{}: ng", what);
                assert_eq!(bits(a.ns), bits(b.ns), "{}: ns", what);
                assert_eq!((a.mat_id, a.prim_id, a.inst_id), (b.mat_id, b.prim_id, b.inst_id), "{}: ids", what);
            }
            (a, b) => panic!("{}: hit mismatch {:?} vs {:?}", what, a.map(|h| (h.t, h.inst_id)), b.map(|h| (h.t, h.inst_id))),
        }
    }

    /// ワールド空間の総当たり参照: 全インスタンスの全三角形をワールド座標に変換して直接交差判定する
    /// （インスタンス変換・物体空間の tmin/tmax の写像に依存しない独立な実装）。球も含む。
    fn hit_world_brute_force(world: &World, r: Ray, tmin: f64, tmax: f64) -> Option<(f64, Option<usize>, usize)> {
        let mut closest = tmax;
        let mut best = None;
        for (inst_id, inst) in world.instances.iter().enumerate() {
            for (tri_id, t) in world.meshes[inst.mesh_id].tris.iter().enumerate() {
                let w = Triangle::new_static(inst.xform.apply_point(t.v0_0), inst.xform.apply_point(t.v1_0), inst.xform.apply_point(t.v2_0), t.mat_id);
                if let Some(h) = w.hit(r, tmin, closest) {
                    closest = h.t;
                    best = Some((h.t, Some(inst_id), tri_id));
                }
            }
        }
        for (idx, s) in world.spheres.iter().enumerate() {
            if let Some(h) = s.hit(r, tmin, closest) {
                closest = h.t;
                best = Some((h.t, None, idx));
            }
        }
        best
    }

    /// World::hit とワールド空間総当たりが一致するか（計算経路が違うので t は相対 1e-7 で比較）。
    /// 片方だけがヒットする場合は、そのヒットが tmin / tmax の境界（相対 1e-6 以内）にあるときだけ許す。
    /// ID が違う場合は、t が一致する重なり面（同一平面・一致インスタンス）なら許す。
    fn check_against_brute_force(world: &World, r: Ray, tmin: f64, tmax: f64, what: &str) {
        let a = world.hit(r, tmin, tmax);
        let b = hit_world_brute_force(world, r, tmin, tmax);
        let near_bound = |t: f64| (t - tmin).abs() <= 1e-6 * tmin.max(t) || (t - tmax).abs() <= 1e-6 * tmax.max(t);
        match (a, b) {
            (None, None) => {}
            (Some(h), None) => assert!(near_bound(h.t), "{}: World::hit found t={} (inst {:?}) but brute force found nothing", what, h.t, h.inst_id),
            (None, Some((t, inst, _))) => assert!(near_bound(t), "{}: brute force found t={} (inst {:?}) but World::hit found nothing", what, t, inst),
            (Some(h), Some((t, inst, prim))) => {
                assert!((h.t - t).abs() <= 1e-7 * t.max(1e-3), "{}: t {} vs brute force {} (inst {:?} vs {:?})", what, h.t, t, h.inst_id, inst);
                let _ = prim;
            }
        }
    }

    /// インスタンス BVH を最近接距離で枝刈りしても、結果（t・点・法線・ID）はビット単位で不変で、
    /// かつワールド空間の総当たりと一致する（tmin の写像・再探索の正しさ）。
    ///
    /// 枝刈り距離の誤り（相対マージンの撤去・`|A⁻¹d|` の掛け忘れ・`|A⁻¹d|` で割る）を
    /// 検出できるよう、次を含める:
    /// - 先に走査される拡大インスタンス（×100）の手前に、後から走査される縮小インスタンス（×0.01）
    /// - 完全一致・ほぼ一致（相対 1e-8 / 1e-12 のスケール差）のインスタンス対
    /// - 一次レイに加え、ヒット点からの二次レイと有限 tmax のシャドウレイ
    #[test]
    fn instance_pruning_preserves_hits() {
        // 板 20 枚（z = 0, 0.1, …, 1.9）を重ねたメッシュ。1 インスタンス内でも奥の板が枝刈り対象になる
        let plates = || -> Vec<Triangle> {
            (0..20)
                .flat_map(|k| {
                    let z = k as f64 * 0.1;
                    vec![
                        Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                        Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
                    ]
                })
                .collect()
        };
        let mut rng = Rng::new(77);
        let rnd = |rng: &mut Rng, s: f64| Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * s;
        let mut world = World::new();
        let mut xforms: Vec<Transform> = Vec::new();
        let add = |world: &mut World, xforms: &mut Vec<Transform>, xf: Transform| {
            let id = world.add_mesh_instance(plates(), xf, Some(xforms.len()));
            xforms.push(xf);
            id
        };

        // 1. 拡大（×100、非一様）を先に。原点付近を奥行き方向に大きく覆う
        for k in 0..3 {
            let xf = Transform::translate(Vec3::new(0.0, 0.0, -150.0 + k as f64 * 7.0))
                .compose(Transform::rotate(Vec3::new(0.2, 1.0, 0.1), 11.0 * k as f64))
                .compose(Transform::scale(Vec3::new(100.0, 80.0, 100.0)));
            add(&mut world, &mut xforms, xf);
        }
        // 2. 縮小（×0.01）を後に、拡大インスタンスの手前に多数
        for _ in 0..40 {
            let xf = Transform::translate(rnd(&mut rng, 6.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(0.0, 1.0, 0.0), rng.next_f64() * 360.0))
                .compose(Transform::scale(Vec3::new(0.01, 0.013, 0.008) * (1.0 + 30.0 * rng.next_f64())));
            add(&mut world, &mut xforms, xf);
        }
        // 3. 完全一致・ほぼ一致の対（同じメッシュ・ほぼ同じ変換）
        for &(s, eps) in &[(0.8, 0.0), (0.8, 1e-8), (1.0, 1e-12), (0.01, 1e-8), (100.0, 1e-12), (3.0, 1e-8)] {
            let base = Transform::translate(rnd(&mut rng, 4.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(1.0, 0.0, 0.0), rng.next_f64() * 360.0));
            add(&mut world, &mut xforms, base.compose(Transform::scale(Vec3::new(s, s, s))));
            let s2 = s * (1.0 + eps);
            add(&mut world, &mut xforms, base.compose(Transform::scale(Vec3::new(s2, s2, s2))));
        }
        // 4. 一般的なスケールと回転の混在
        for i in 0..20 {
            let s = [0.01, 0.3, 1.0, 3.0, 100.0][i % 5];
            let xf = Transform::translate(rnd(&mut rng, 8.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(0.0, 0.0, 1.0), 37.0 * i as f64))
                .compose(Transform::scale(Vec3::new(s, s * 0.7, s * 1.3)));
            add(&mut world, &mut xforms, xf);
        }
        world.spheres.push(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 999 });

        let n_inst = xforms.len();
        let (mut primary_hits, mut queries) = (0usize, 0usize);
        for _ in 0..20_000 {
            // インスタンス内の点を狙った一次レイ（ヒットが多くなるように）
            let target_inst = (rng.next_f64() * n_inst as f64) as usize % n_inst;
            let local = Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 1.9);
            let target = xforms[target_inst].apply_point(local);
            let o = target + uniform_sphere_dir(&mut rng) * (0.05 + 30.0 * rng.next_f64());
            let r = Ray { o, d: (target - o).norm(), time: 0.0 };
            let a = world.hit(r, 1e-4, 1e30);
            assert_same_hit(a, hit_without_instance_pruning(&world, r, 1e-4, 1e30), "primary");
            // 総当たりは重いので最初の 4000 本だけ（二次レイ・シャドウレイも同様）
            let brute = queries < 12_000;
            if brute {
                check_against_brute_force(&world, r, 1e-4, 1e30, "primary/brute");
            }
            queries += 1;
            let Some(h) = a else { continue };
            primary_hits += 1;

            // 二次レイ（ヒット点から任意方向）
            let d2 = uniform_sphere_dir(&mut rng);
            let r2 = Ray { o: h.p + 1e-4 * d2, d: d2, time: 0.0 };
            assert_same_hit(world.hit(r2, 1e-4, 1e30), hit_without_instance_pruning(&world, r2, 1e-4, 1e30), "secondary");
            if brute {
                check_against_brute_force(&world, r2, 1e-4, 1e30, "secondary/brute");
            }

            // シャドウレイ（有限 tmax。別インスタンス内の点へ）
            let other = (rng.next_f64() * n_inst as f64) as usize % n_inst;
            let lp = xforms[other].apply_point(Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, 1.9 * rng.next_f64()));
            let to = lp - h.p;
            let dist = to.len();
            if dist > 1e-3 {
                let d3 = to / dist;
                let r3 = Ray { o: h.p + 1e-4 * d3, d: d3, time: 0.0 };
                let tmax = (dist - 2e-4).max(1e-4);
                assert_same_hit(world.hit(r3, 1e-4, tmax), hit_without_instance_pruning(&world, r3, 1e-4, tmax), "shadow");
                if brute {
                    check_against_brute_force(&world, r3, 1e-4, tmax, "shadow/brute");
                }
            }
            queries += 2;
        }
        assert!(primary_hits > 10_000, "too few hits to be meaningful: {} of {}", primary_hits, queries);
    }

    /// z = 0 と z = `gap` の 2 枚の板（xy は [-1, 1]）を `s` 倍に拡大縮小したインスタンスだけのワールド。
    fn two_plates_world(gap: f64, s: f64) -> World {
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        let tris: Vec<Triangle> = quad(0.0).into_iter().chain(quad(gap)).collect();
        world.add_mesh_instance(tris, Transform::scale(Vec3::new(s, s, s)), None);
        world
    }

    /// ケース A（verify_batch1 の tminprobe）: ×100 の拡大インスタンスで、ワールド距離 0.005（> tmin）にある
    /// 手前の板に当たる。旧実装は物体空間の t = 5e-5 < tmin で手前の板を取りこぼし、奥の板（t ≈ 100）に当たっていた。
    #[test]
    fn scaled_up_instance_keeps_near_surface_beyond_tmin() {
        let world = two_plates_world(1.0, 100.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -0.005), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        let h = world.hit(r, 1e-4, 1e30).expect("hit");
        assert!((h.t - 0.005).abs() < 1e-9, "t = {}", h.t);
    }

    /// ケース B（verify_batch1 の tminprobe）: ×0.01 の縮小インスタンス（板はワールド z = 0 と z = 1）で、
    /// 始点が手前の板の 5e-6（< tmin）手前。手前の板は自己交差回避の帯の中なので無視し、奥の板（t ≈ 1）に
    /// 当たる。旧実装は物体空間で手前の板を返し、ワールド判定で却下してインスタンスごと None になっていた。
    #[test]
    fn scaled_down_instance_skips_surface_inside_tmin_and_finds_far_one() {
        let world = two_plates_world(100.0, 0.01);
        let r = Ray { o: Vec3::new(0.001, 0.002, -5e-6), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        let h = world.hit(r, 1e-4, 1e30).expect("the far plate must be found");
        assert!((h.t - (1.0 + 5e-6)).abs() < 1e-9, "t = {}", h.t);
    }

    /// ケース C: 丸めで帯の境界に落ちる面。手前の板がワールド距離 tmin·(1 − 5e-10) にあると、物体空間では
    /// 広げた下限 tmin_obj を超えるので BVH が返すが、ワールド判定では t <= tmin で却下される。この場合も
    /// 再探索で同じインスタンスの奥の板が見つかる（再探索がないと None になる）。
    #[test]
    fn rejected_near_surface_retries_same_instance() {
        let tmin = 1e-4;
        let world = two_plates_world(100.0, 0.01);
        let r = Ray { o: Vec3::new(0.001, 0.002, -tmin * (1.0 - 5e-10)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 前提の確認: 物体空間では手前の板が下限を超える
        let inst = &world.instances[0];
        let d_len = inst.xform.apply_vec_inv(r.d).len();
        let t_obj_near = (0.0 - inst.xform.apply_point_inv(r.o).z) / (inst.xform.apply_vec_inv(r.d).z / d_len);
        assert!(t_obj_near > tmin * d_len * (1.0 - 1e-9), "test setup: the near plate must pass the object-space lower bound");
        let h = world.hit(r, tmin, 1e30).expect("the far plate must be found after retrying");
        assert!((h.t - (1.0 + tmin * (1.0 - 5e-10))).abs() < 1e-9, "t = {}", h.t);
    }

    /// 同一平面の面が自己交差回避の帯の中に多数重なっていても、再探索は上限回数で止まる（無限ループしない）。
    /// その場合、帯の先にある面は諦めて None を返しうる（退化した入力に対する割り切り）。
    #[test]
    fn retry_limit_terminates_on_many_faces_inside_tmin_band() {
        let tmin = 1e-4;
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        // 帯の中（ワールド距離 tmin の直前）に、わずかに z がずれた 20 枚。奥に 1 枚。
        let mut tris = Vec::new();
        for k in 0..20 {
            tris.extend(quad(k as f64 * 1e-12));
        }
        tris.extend(quad(0.5));
        world.add_mesh_instance(tris, Transform::identity(), None);
        let r = Ray { o: Vec3::new(0.1, 0.1, -tmin * (1.0 - 1e-6)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 終了すること自体が要件。結果は奥の板か None のどちらか
        match world.hit(r, tmin, 1e30) {
            None => {}
            Some(h) => assert!((h.t - (0.5 + tmin * (1.0 - 1e-6))).abs() < 1e-9, "t = {}", h.t),
        }
    }

    /// 12 枚の三角形で作る [-1, 1]³ の立方体。
    fn unit_box() -> Vec<Triangle> {
        let c = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let faces = [
            [c(-1., -1., -1.), c(1., -1., -1.), c(1., 1., -1.), c(-1., 1., -1.)],
            [c(-1., -1., 1.), c(1., -1., 1.), c(1., 1., 1.), c(-1., 1., 1.)],
            [c(-1., -1., -1.), c(1., -1., -1.), c(1., -1., 1.), c(-1., -1., 1.)],
            [c(-1., 1., -1.), c(1., 1., -1.), c(1., 1., 1.), c(-1., 1., 1.)],
            [c(-1., -1., -1.), c(-1., 1., -1.), c(-1., 1., 1.), c(-1., -1., 1.)],
            [c(1., -1., -1.), c(1., 1., -1.), c(1., 1., 1.), c(1., -1., 1.)],
        ];
        faces.iter().flat_map(|q| [Triangle::new_static(q[0], q[1], q[2], 0), Triangle::new_static(q[0], q[2], q[3], 0)]).collect()
    }

    /// 二次レイの自己再ヒットなし: 球と、回転・非一様スケールした立方体インスタンスを、大きさ 1e-3〜1e3、
    /// 原点からの距離 0・1e3 倍・**絶対 1e8** に置き、表面の点から `offset_ray_origin`（誤差上界ぶん法線方向に
    /// ずらす）で二次レイを出す（tmin = 0）。外向き（幾何法線側）のレイは凸な自分自身に当たらず、内向きのレイ
    /// （透過）は入射した面に再ヒットせず（t が交差点の誤差上界より十分大きい）物体の反対側へ抜ける。
    #[test]
    fn secondary_rays_do_not_rehit_their_own_surface_at_any_scale() {
        use crate::geometry::offset_ray_origin;
        let mut rng = Rng::new(31);
        for &k in &[1e-3, 1.0, 1e3] {
            for &dist in &[0.0, 1e3 * k, 1e8] {
                let center = Vec3::new(dist, 0.5 * dist, -0.3 * dist);
                let mut world = World::new();
                world.add_sphere(Sphere { c: center, r: k, mat_id: 0 });
                let box_center = center + Vec3::new(4.0 * k, 0.0, 0.0);
                let xf = Transform::translate(box_center)
                    .compose(Transform::rotate(Vec3::new(0.3, 1.0, 0.2), 33.0))
                    .compose(Transform::scale(Vec3::new(k, 0.7 * k, 1.3 * k)));
                world.add_mesh_instance(unit_box(), xf, None);
                let mut checked = 0;
                for i in 0..2000 {
                    let target = if i % 2 == 0 { center } else { box_center };
                    let o = target + uniform_sphere_dir(&mut rng) * (5.0 * k);
                    let Some(h) = world.hit(Ray { o, d: (target - o).norm(), time: 0.0 }, 0.0, 1e30) else { continue };
                    // 三角形の法線の向き（巻き順）は保証されないので、実際に当たった物体の中心から外向きにそろえる
                    let hit_center = if h.inst_id.is_some() { box_center } else { center };
                    let n = if h.ng.dot(h.p - hit_center) < 0.0 { -h.ng.norm() } else { h.ng.norm() };
                    for _ in 0..4 {
                        let mut d = uniform_sphere_dir(&mut rng);
                        if d.dot(n) < 0.0 { d = -d; }
                        // 外向き
                        if let Some(h2) = world.hit(Ray { o: offset_ray_origin(h.p, h.p_error, h.ng, d), d, time: 0.0 }, 0.0, 1e30) {
                            assert!(!(h2.inst_id == h.inst_id && (h.inst_id.is_some() || h2.prim_id == h.prim_id)),
                                "k={} dist={}: outward ray re-hit its own object at t={} (p_error {:?})", k, dist, h2.t, h.p_error.max_abs());
                        }
                        // 内向き（浅すぎる角度は除く）
                        let di = -d;
                        if di.dot(-n) > 0.1 {
                            let h2 = world.hit(Ray { o: offset_ray_origin(h.p, h.p_error, h.ng, di), d: di, time: 0.0 }, 0.0, 1e30)
                                .unwrap_or_else(|| panic!("k={} dist={}: inward ray escaped its object", k, dist));
                            // 入射面への再ヒットかどうか: 立方体なら新しい交点の面が入射面と平行で、入射面の上（法線方向の
                            // 変位が誤差上界程度）に残る。球なら新しい交点が入射点とほぼ一致する。立方体の辺の近くでは
                            // 隣の面から抜ける正当な短い弦（原点から遠い配置では誤差上界の数倍程度）もあるので、
                            // 弦の長さでは判定しない
                            let err = h.p_error.max_abs().max(h2.p_error.max_abs());
                            let same_object = h2.inst_id == h.inst_id;
                            let rehit = same_object
                                && if h.inst_id.is_some() {
                                    h2.ng.norm().dot(h.ng.norm()).abs() > 0.999 && (h2.p - h.p).dot(n).abs() <= 10.0 * err
                                } else {
                                    (h2.p - h.p).len() <= 100.0 * err
                                };
                            assert!(!rehit, "k={} dist={}: inward ray re-hit the entry surface at t={} (p_error {})", k, dist, h2.t, err);
                        }
                        checked += 1;
                    }
                }
                assert!(checked > 4000, "k={} dist={}: too few checks ({})", k, dist, checked);
            }
        }
    }

    /// 大きな床と薄い遮蔽物の混在（バッチ 3b では「期待される失敗」だったもの、3c の合格条件）:
    /// 1 万単位四方の床と、その上の非常に薄い壁（厚さ 5e-5、高さ 1e-2）。床上で壁から 5e-4 離れた点から
    /// 壁越しに出すシャドウレイは遮蔽される。シーンの大きさに比例したオフセット（≈ 1.4e-3）では原点が壁を
    /// 飛び越えていたが、交差点ごとの誤差上界に基づくオフセット（ここでは ~1e-12）なら壁を検出できる。
    #[test]
    fn large_floor_and_thin_occluder() {
        use crate::geometry::offset_ray_origin;
        let mut world = World::new();
        let q = |x: f64, z: f64| Vec3::new(x, 0.0, z);
        let floor = vec![
            Triangle::new_static(q(-5e3, -5e3), q(5e3, -5e3), q(5e3, 5e3), 0),
            Triangle::new_static(q(-5e3, -5e3), q(5e3, 5e3), q(-5e3, 5e3), 0),
        ];
        world.add_mesh_instance(floor, Transform::identity(), None);
        // 薄い壁: x ∈ [0, 5e-5]、y ∈ [0, 1e-2]、z ∈ [-5e-3, 5e-3]
        let wall = Transform::translate(Vec3::new(2.5e-5, 5e-3, 0.0)).compose(Transform::scale(Vec3::new(2.5e-5, 5e-3, 5e-3)));
        world.add_mesh_instance(unit_box(), wall, None);
        // 床の点（壁の手前 5e-4）を上から見て交差情報を得る
        let p = Vec3::new(-5e-4, 0.0, 0.0);
        let h = world.hit(Ray { o: p + Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 }, 0.0, 1e30).expect("floor hit");
        assert!((h.p - p).len() < 1e-9 && h.inst_id == Some(0));
        let d = Vec3::new(1.0, 0.002, 0.0).norm();
        let o = offset_ray_origin(h.p, h.p_error, h.ng, d);
        assert!((o - h.p).len() < 1e-9, "offset {} must be far below the distance to the wall", (o - h.p).len());
        let occluded = world.hit(Ray { o, d, time: 0.0 }, 0.0, 10.0);
        assert!(matches!(occluded, Some(w) if w.inst_id == Some(1)), "thin occluder not detected");
    }

    /// `World::bounds` は回転したインスタンスでも、変換後の頂点を包む（8 頂点の変換）。
    #[test]
    fn bounds_contain_rotated_instance() {
        let mut world = World::new();
        let xf = Transform::translate(Vec3::new(10.0, -2.0, 3.0)).compose(Transform::rotate(Vec3::new(0.0, 0.0, 1.0), 45.0));
        world.add_mesh_instance(unit_box(), xf, None);
        let b = world.bounds();
        let s = 2f64.sqrt();
        let expect_min = Vec3::new(10.0 - s, -2.0 - s, 2.0);
        let expect_max = Vec3::new(10.0 + s, -2.0 + s, 4.0);
        // BVH の AABB はわずかに（~1e-9）広げてある
        assert!((b.min - expect_min).len() < 1e-8 && (b.max - expect_max).len() < 1e-8, "{:?} {:?}", (b.min.x, b.min.y, b.min.z), (b.max.x, b.max.y, b.max.z));
    }

    /// インスタンスのワールド空間の境界ボックス（`Instance::world_bounds`）による事前棄却は、ヒットを落とさない。
    /// 事前棄却を使わない参照実装（`hit_without_instance_pruning`）と、ビット単位で同じ結果になることを確かめる。
    ///
    /// 箱の境界ちょうどに当たるレイを多く含める。立方体の頂点は箱の角・辺・面の上にあり、軸平行の配置では
    /// 立方体の面が箱の面と重なる。配置は大きさ 1e-3〜1e3、原点からの距離 0 と**絶対 1e8**、軸平行と
    /// 回転＋非一様スケール。レイは頂点・辺・面を狙い、すれすれの方向（面にほぼ平行）も含める。
    /// 箱のパディング（角の誤差上界と γ(3)·|座標|）を両方外すと、1e8 の配置でヒットを落として失敗する。
    #[test]
    fn instance_world_bounds_never_drop_hits() {
        let mut rng = Rng::new(97);
        let (mut hits, mut checked) = (0usize, 0usize);
        for case in 0..160 {
            let k = [1e-3, 1.0, 1e3][case % 3];
            let dist = if case % 2 == 0 { 0.0 } else { 1e8 };
            let rotated = case % 4 >= 2;
            let offset = Vec3::new(dist, -0.6 * dist, 0.8 * dist) + uniform_sphere_dir(&mut rng) * k;
            let mut xf = Transform::translate(offset);
            if rotated {
                xf = xf.compose(Transform::rotate(uniform_sphere_dir(&mut rng), 360.0 * rng.next_f64()));
            }
            let xf = xf.compose(Transform::scale(Vec3::new(k, (0.5 + rng.next_f64()) * k, (0.5 + rng.next_f64()) * k)));
            let mut world = World::new();
            world.add_mesh_instance(unit_box(), xf, None);
            for i in 0..150 {
                // 狙う点（物体空間）: 頂点・辺上・面上（座標の成分を ±1 に固定する数で切り替え）
                let mut q = [rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0];
                let fixed = 3 - i % 3; // 3 = 頂点, 2 = 辺, 1 = 面
                let first = rng.next_u32() as usize % 3;
                for j in 0..fixed {
                    q[(first + j) % 3] = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
                }
                let target = xf.apply_point(Vec3::new(q[0], q[1], q[2]));
                // 方向: 一様な向き、または軸方向にほぼ平行（すれすれ）
                let mut d = uniform_sphere_dir(&mut rng);
                if i % 2 == 1 {
                    let axis = rng.next_u32() as usize % 3;
                    let mut a = [d.x * 1e-3, d.y * 1e-3, d.z * 1e-3];
                    a[axis] = if rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
                    d = Vec3::new(a[0], a[1], a[2]).norm();
                }
                let o = target - d * (4.0 * k);
                let r = Ray { o, d, time: 0.0 };
                for &(tmin, tmax) in &[(0.0, 1e30), (0.0, 4.0 * k * (1.0 + 1e-12))] {
                    let expect = hit_without_instance_pruning(&world, r, tmin, tmax);
                    hits += expect.is_some() as usize;
                    checked += 1;
                    assert_same_hit(world.hit(r, tmin, tmax), expect, &format!("case {} (k={} dist={} rotated={}) ray {}", case, k, dist, rotated, i));
                }
            }
        }
        assert!(checked == 48_000 && hits > 10_000, "too few hits checked ({} of {})", hits, checked);
    }

    /// 中心頂点 `c` の周りに 6 枚の三角形を並べた扇（共有辺 6 本と共有頂点 1 つ）。法線 `n` の平面上、半径 `rad`。
    fn triangle_fan(c: Vec3, n: Vec3, rad: f64) -> (Vec<Triangle>, Vec<Vec3>) {
        let a = if n.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let t = n.cross(a).norm();
        let b = n.cross(t);
        let rim: Vec<Vec3> = (0..6)
            .map(|i| {
                let ang = std::f64::consts::TAU * i as f64 / 6.0 + 0.3;
                c + (t * ang.cos() + b * ang.sin()) * rad
            })
            .collect();
        let tris = (0..6).map(|i| Triangle::new_static(c, rim[i], rim[(i + 1) % 6], 0)).collect();
        (tris, rim)
    }

    /// 水密性の回帰テスト: 三角形の扇の**共有頂点ちょうど**と**共有辺上の点**を狙うレイは、隣り合う三角形の
    /// どちらかに必ず当たる（すり抜け 0 件）。大きさ ×1e-3 / 等倍 / ×1e3 と、原点から 1e8 離した配置、
    /// ランダムな向きの平面、平面の両側からのレイで確認する。三角形単体（ワールド座標）と、回転・非一様
    /// スケールしたインスタンス経由（`World::hit`）の両方を調べる。
    /// Möller–Trumbore では共有辺ちょうどを狙うレイの約 2% がすり抜けていた。
    #[test]
    fn shared_edges_and_vertices_are_watertight() {
        let mut rng = Rng::new(41);
        let mut total = 0usize;
        for &(k, dist) in &[(1.0, 0.0), (1e-3, 0.0), (1e3, 0.0), (1.0, 1e8), (1e-3, 1e8)] {
            for _ in 0..20 {
                let c = Vec3::new(dist, -0.7 * dist, 0.4 * dist) + uniform_sphere_dir(&mut rng) * (0.3 * k);
                let n = uniform_sphere_dir(&mut rng);
                let (tris, rim) = triangle_fan(c, n, k);
                // 同じ扇をインスタンスとしても置く（物体空間では原点中心・等倍、ワールドへ回転・非一様スケール・平行移動）
                let (obj_tris, obj_rim) = triangle_fan(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0);
                let xf = Transform::translate(c)
                    .compose(Transform::rotate(uniform_sphere_dir(&mut rng), 360.0 * rng.next_f64()))
                    .compose(Transform::scale(Vec3::new(k, 0.8 * k, 1.3 * k)));
                let mut world = World::new();
                world.add_mesh_instance(obj_tris, xf, None);
                for i in 0..120 {
                    // 狙う点: 共有頂点（中心）、または共有辺（中心 → 外周の頂点）上の点。外周の頂点・辺はメッシュの
                    // 境界なので、丸めで外側に出た点を狙うレイが外れるのは正当（水密性の対象外）
                    let edge = i % 6;
                    let s = if i % 4 == 0 { 0.0 } else { rng.next_f64() };
                    let target = c + (rim[edge] - c) * s;
                    let obj_target = obj_rim[edge] * s;
                    let world_target = xf.apply_point(obj_target);
                    for (tgt, is_instance) in [(target, false), (world_target, true)] {
                        let o = tgt + uniform_sphere_dir(&mut rng) * (3.0 * k);
                        let d = (tgt - o).norm();
                        let r = Ray { o, d, time: 0.0 };
                        let hit = if is_instance {
                            world.hit(r, 0.0, 1e30).is_some()
                        } else {
                            tris.iter().any(|t| t.hit(r, 0.0, 1e30).is_some())
                        };
                        // 平面にほぼ平行なレイは除く（平面に届く前に扇の外を通りうる）
                        let plane_n = if is_instance { xf.apply_normal(Vec3::new(0.0, 0.0, 1.0)) } else { n };
                        if d.dot(plane_n).abs() < 0.05 {
                            continue;
                        }
                        total += 1;
                        assert!(hit, "k={} dist={} instance={} s={}: ray aimed at a shared {} slipped through", k, dist, is_instance, s, if s == 0.0 { "vertex" } else { "edge" });
                    }
                }
            }
        }
        assert!(total > 20_000, "too few rays checked ({})", total);
    }

    /// build_lights の重み = 面積 × 輝度（Light::area と共有された面積計算）。
    #[test]
    fn sphere_light_weight_is_area_times_luminance() {
        let r = 2.0;
        let world = emissive_sphere_world(Vec3::new(0.0, 0.0, 0.0), r);
        assert_eq!(world.lights.len(), 1);
        let area = 4.0 * std::f64::consts::PI * r * r;
        let lum = Color::new(3.0, 4.0, 5.0).luminance();
        assert!((world.light_total - area * lum).abs() < 1e-9);
    }

    /// sample_light のサンプルは球面上にあり、法線は外向き、PDF は有限正値。
    #[test]
    fn sphere_light_samples_lie_on_surface() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(5);
        let p = Vec3::new(5.0, 0.0, 0.0);
        let mut got = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                got += 1;
                assert!(((ls.position - c).len() - r).abs() < 1e-9, "off surface");
                assert!(ls.normal.dot(ls.position - c) > 0.0, "normal not outward");
                assert!(ls.pdf > 0.0 && ls.pdf.is_finite(), "bad pdf");
            }
        }
        assert!(got > 0, "no valid light samples");
    }

    /// `light_pdf` は `sample_light` の逆演算: サンプルされた点への Hit を作って
    /// `light_pdf` に渡すと、`sample_light` が返した pdf と一致しなければならない。
    /// これは BSDF サンプリングが発光体に命中した際の MIS 重み付けが正しく機能する
    /// ための前提条件で、この一致が壊れると面光源の寄与が二重計上/過小評価される。
    #[test]
    fn light_pdf_matches_sample_light_pdf() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(7);
        let from = Vec3::new(5.0, 0.0, 0.0);
        let mut checked = 0;
        for _ in 0..500 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                let hit = Hit {
                    t: 0.0,
                    p: ls.position,
                    ng: ls.normal,
                    ns: ls.normal,
                    mat_id: 0,
                    prim_id: 0,
                    inst_id: None,
                    p_error: Vec3::new(0.0, 0.0, 0.0),
                    bary: (0.0, 0.0),
                    uv: (0.0, 0.0),
                };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() < 1e-9 * ls.pdf.max(1.0), "light_pdf {} != sample_light pdf {}", pdf, ls.pdf);
                checked += 1;
            }
        }
        assert!(checked > 0, "no valid light samples");
    }

    /// 参照点のバリエーション: 近い外部・遠い外部（sin²θmax が小円錐近似の閾値未満）・
    /// 表面すれすれの外部・内部（中心付近と表面付近）。
    fn sphere_light_reference_points() -> Vec<(&'static str, Vec3)> {
        vec![
            ("near outside", Vec3::new(2.5, 0.7, -0.4)),
            ("far outside (small cone)", Vec3::new(300.0, -50.0, 120.0)),
            ("just outside", Vec3::new(1.5 + 1e-3, 0.0, 0.0)),
            ("inside center", Vec3::new(0.1, -0.2, 0.05)),
            // 表面から 0.2。表面ごく近傍（例 0.01）だと面積サンプリングの 1/pdf の分散が
            // 対数発散し、有限サンプルの平均推定が安定しないため
            ("inside near surface", Vec3::new(0.0, 1.3, 0.0)),
        ]
    }

    /// 円錐サンプリング／内部フォールバックの両方で、sample_light の pdf は同じ点への light_pdf と一致し、
    /// サンプル点は球面上にある。外部からのサンプルは参照点から見える側（cos_light > 0）にある。
    #[test]
    fn sphere_light_pdf_matches_for_cone_and_inside_fallback() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(21);
        for (name, from) in sphere_light_reference_points() {
            let inside = (from - c).len() <= r;
            let mut got = 0;
            for _ in 0..4000 {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                got += 1;
                assert!(((ls.position - c).len() - r).abs() < 1e-9, "{}: off surface", name);
                if !inside {
                    let wi = (ls.position - from).norm();
                    assert!(ls.normal.dot(-wi) > 0.0, "{}: sampled a point not visible from outside", name);
                }
                let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() <= 1e-9 * ls.pdf, "{}: light_pdf {} != sample pdf {}", name, pdf, ls.pdf);
            }
            assert!(got > 3900, "{}: too many rejected samples ({} / 4000)", name, got);
        }
    }

    /// light_pdf を立体角で積分すると 1（BSDF 側から見た光源の方向分布が正規化されている）。
    /// 全球一様な方向にレイを飛ばし、光源に当たった点の light_pdf の平均 × 4π で推定する。
    #[test]
    fn sphere_light_pdf_integrates_to_one_over_solid_angle() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(99);
        for (name, from) in sphere_light_reference_points() {
            // 外部では円錐を少し広げたキャップ（立体角は円錐の 1.5 倍）、内部では全球に
            // 一様な方向で推定する（遠い小円錐は全球一様だとほとんど当たらないため）
            let axis = (c - from).norm();
            let dc = (c - from).len();
            let cap_cos = if dc > r {
                let sin2 = (r / dc).powi(2);
                let one_minus_cos = if sin2 < 1e-3 { 0.5 * sin2 } else { 1.0 - (1.0 - sin2).sqrt() };
                1.0 - 1.5 * one_minus_cos
            } else {
                -1.0
            };
            let omega_cap = std::f64::consts::TAU * (1.0 - cap_cos);
            let (t, b) = orthonormal_basis(axis);
            let n = 400_000;
            let mut sum = 0.0;
            for _ in 0..n {
                let z = 1.0 - rng.next_f64() * (1.0 - cap_cos);
                let rr = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f64::consts::TAU * rng.next_f64();
                let d = t * (rr * phi.cos()) + b * (rr * phi.sin()) + axis * z;
                if let Some(h) = world.hit(Ray { o: from, d, time: 0.0 }, 1e-9, 1e30) {
                    sum += world.light_pdf(from, 0.0, &h);
                }
            }
            let integral = sum / n as f64 * omega_cap;
            assert!((integral - 1.0).abs() < 0.01, "{}: ∫pdf dω = {}", name, integral);
        }
    }

    /// 円錐サンプリングの推定は不偏: E[cosθ / pdf] = ∫_cone cosθ dω = π·sin²θmax
    /// （θ は球中心方向からの角）。内部フォールバックでは E[1/pdf] = 4π。
    #[test]
    fn sphere_light_estimates_are_unbiased() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(3);
        let n = 200_000;
        for (name, from) in sphere_light_reference_points() {
            let dc = (c - from).len();
            let axis = (c - from) / dc;
            let mut sum = 0.0;
            let inside = dc <= r;
            for _ in 0..n {
                if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                    let f = if inside { 1.0 } else { (ls.position - from).norm().dot(axis) };
                    sum += f / ls.pdf;
                }
            }
            let est = sum / n as f64;
            let exact = if inside { 4.0 * std::f64::consts::PI } else { std::f64::consts::PI * (r / dc).powi(2) };
            assert!((est / exact - 1.0).abs() < 0.01, "{}: estimate {} vs exact {}", name, est, exact);
        }
    }

    /// 旧実装の小円錐近似の閾値（sin²θmax）。この前後で推定値が不連続にならないことを確かめる。
    const OLD_SMALL_CONE_SIN2: f64 = 0.00068523;

    /// 中心 c・半径 r の球に対し、sin²θmax が `sin2` になる外部の参照点。
    fn from_for_sin2(c: Vec3, r: f64, sin2: f64) -> Vec3 {
        c + Vec3::new(0.3, 0.8, -0.52).norm() * (r / sin2.sqrt())
    }

    /// 1 − cosθmax は小さな円錐でも厳密（桁落ち・一次近似の誤差がない）。
    #[test]
    fn cone_one_minus_cos_max_is_exact() {
        let s = Sphere { c: Vec3::new(0.3, -0.2, 0.1), r: 1.3, mat_id: 0 };
        let t = OLD_SMALL_CONE_SIN2;
        for sin2 in [0.5, 1e-2, t * (1.0 + 1e-3), t * (1.0 + 1e-9), t * (1.0 - 1e-9), t * (1.0 - 1e-3), 1e-5, 1e-8, 1e-12] {
            let cone = sphere_cone(&s, from_for_sin2(s.c, s.r, sin2)).expect("outside");
            let sin2 = cone.sin2_max; // 参照点の丸め後の実際の値で比較する
            let reference = if sin2 >= 1e-4 {
                1.0 - (1.0 - sin2).sqrt()
            } else {
                // 1 − √(1 − x) = x/2 + x²/8 + x³/16 + 5x⁴/128 + …
                sin2 / 2.0 + sin2 * sin2 / 8.0 + sin2.powi(3) / 16.0 + 5.0 * sin2.powi(4) / 128.0
            };
            let rel = (cone.one_minus_cos_max / reference - 1.0).abs();
            assert!(rel < 1e-11, "sin²θmax={:e}: 1−cosθmax {:e} vs reference {:e} (rel {:e})", sin2, cone.one_minus_cos_max, reference, rel);
        }
    }

    /// 小さな円錐でも推定が偏らず、旧近似の閾値の前後で連続:
    /// E[cosθ/pdf] = π·sin²θmax を相対 2e-5 で満たす（旧実装は閾値未満で −1.7e-4 の偏り）。
    /// cosθ/pdf = (1 − u·(1−cosθmax))·2π(1−cosθmax) は u に線形なので、統計誤差は (1−cosθmax) 倍に縮み
    /// 小さな円錐では 1e-6 未満になる。
    #[test]
    fn small_cone_estimate_is_unbiased_and_continuous() {
        let c = Vec3::new(0.3, -0.2, 0.1);
        let r = 1.3;
        let world = emissive_sphere_world(c, r);
        let t = OLD_SMALL_CONE_SIN2;
        let mut pdfs = Vec::new();
        for sin2 in [t * (1.0 - 1e-3), t * (1.0 - 1e-9), t * (1.0 + 1e-9), t * (1.0 + 1e-3), 1e-6, 1e-9] {
            let from = from_for_sin2(c, r, sin2);
            let dc = (c - from).len();
            let axis = (c - from) / dc;
            let sin2_actual = (r / dc).powi(2);
            let mut rng = Rng::new(17);
            let n = 200_000;
            let mut sum = 0.0;
            let mut pdf0 = 0.0;
            for _ in 0..n {
                let ls = world.sample_light(&mut rng, 0.0, from).expect("cone sample must not be rejected");
                sum += (ls.position - from).norm().dot(axis) / ls.pdf;
                pdf0 = ls.pdf;
            }
            let est = sum / n as f64;
            let exact = std::f64::consts::PI * sin2_actual;
            assert!((est / exact - 1.0).abs() < 2e-5, "sin²θmax={:e}: E[cosθ/pdf] {:e} vs π·sin² {:e} (rel {:e})", sin2, est, exact, est / exact - 1.0);
            pdfs.push(pdf0 * sin2_actual); // pdf ∝ 1/sin²（小円錐）なので正規化して連続性を見る
        }
        // 閾値のすぐ下とすぐ上（sin² の相対差 2e-9）で正規化 pdf が連続
        assert!((pdfs[1] / pdfs[2] - 1.0).abs() < 1e-6, "discontinuity at the old threshold: {} vs {}", pdfs[1], pdfs[2]);
    }

    /// 表面すれすれの外部の参照点でもサンプルがほぼ棄却されず（旧実装は r(1+1e-9) で 99.9% 棄却）、
    /// pdf は light_pdf と一致する。円錐サンプリングの範囲では、全サンプルが参照点から見える側にあり
    /// E[1/pdf] = Ω（円錐の立体角）が厳密に成り立つ（面積フォールバックに落ちると見える側をほぼ引けない）。
    #[test]
    fn near_surface_reference_points_keep_light_samples() {
      // 標準的な球・原点から離れた小さな球・大きな球（丸めの効き方が座標の大きさで変わるため）
      for (c, r) in [(Vec3::new(0.3, -0.2, 0.1), 1.3), (Vec3::new(1000.0, 500.0, -300.0), 0.05), (Vec3::new(-20.0, 3.0, 7.0), 1000.0)] {
        let world = emissive_sphere_world(c, r);
        let dir = Vec3::new(0.3, 0.8, -0.52).norm();
        for eps in [1e-3, 1e-5, 1e-6, 6e-7, 1e-7, 1e-9, 1e-11, 1e-12, 1e-14, 0.0, -1e-9] {
            let from = c + dir * (r * (1.0 + eps));
            let cone = sphere_cone(&world.spheres[0], from);
            let mut rng = Rng::new(5);
            let n = 20_000;
            let (mut got, mut sum_inv) = (0usize, 0.0);
            for _ in 0..n {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                got += 1;
                sum_inv += 1.0 / ls.pdf;
                if cone.is_some() {
                    let wi = (ls.position - from).norm();
                    assert!(ls.normal.dot(-wi) > 0.0, "r={} eps={:e}: cone sample on the hidden side", r, eps);
                }
                let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() <= 1e-9 * ls.pdf, "r={} eps={:e}: light_pdf {} != sample pdf {}", r, eps, pdf, ls.pdf);
            }
            assert!(got as f64 >= 0.99 * n as f64, "r={} eps={:e}: {} / {} samples rejected (cone: {})", r, eps, n - got, n, cone.is_some());
            if let Some(cone) = cone {
                let omega = std::f64::consts::TAU * cone.one_minus_cos_max;
                assert!((sum_inv / got as f64 / omega - 1.0).abs() < 1e-9, "eps={:e}: E[1/pdf] {} vs Ω {}", eps, sum_inv / got as f64, omega);
            }
        }
      }
    }

    /// 発光体でないヒット（`inst_id`/`prim_id` が既知の発光体と一致しない）に対しては 0 を返す。
    #[test]
    fn light_pdf_is_zero_for_non_emitting_hit() {
        let world = emissive_sphere_world(Vec3::new(0.0, 0.0, 0.0), 1.5);
        let hit = Hit {
            t: 0.0,
            p: Vec3::new(10.0, 0.0, 0.0),
            ng: Vec3::new(1.0, 0.0, 0.0),
            ns: Vec3::new(1.0, 0.0, 0.0),
            mat_id: 0,
            prim_id: 3, // no sphere at this index
            inst_id: None,
            p_error: Vec3::new(0.0, 0.0, 0.0),
            bary: (0.0, 0.0),
            uv: (0.0, 0.0),
        };
        assert_eq!(world.light_pdf(Vec3::new(5.0, 0.0, 0.0), 0.0, &hit), 0.0);
    }

    /// `Mesh::hit`（BVH 経由）は全三角形を線形探索するブルートフォースと同じ最近接ヒットを返す。
    /// `Mesh` が三角形配列と BVH の対応関係を自分で保証しているからこそ書ける回帰テスト。
    #[test]
    fn mesh_hit_matches_brute_force() {
        let mut rng = Rng::new(42);
        let mut rand_range = |lo: f64, hi: f64| lo + rng.next_f64() * (hi - lo);

        let mut tris = Vec::new();
        for _ in 0..200 {
            let center = Vec3::new(rand_range(-5.0, 5.0), rand_range(-5.0, 5.0), rand_range(-5.0, 5.0));
            let v0 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            let v1 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            let v2 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            tris.push(Triangle::new_static(v0, v1, v2, 0));
        }
        let mesh = Mesh::new(tris.clone());

        let mut checked_hits = 0;
        for _ in 0..500 {
            let o = Vec3::new(rand_range(-8.0, 8.0), rand_range(-8.0, 8.0), rand_range(-8.0, 8.0));
            let d = Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0)).norm();
            let r = Ray { o, d, time: 0.0 };

            let via_bvh = mesh.hit(r, 1e-6, 1e30);

            let mut brute: Option<Hit> = None;
            let mut closest = 1e30;
            for tri in &tris {
                if let Some(h) = tri.hit(r, 1e-6, closest) {
                    closest = h.t;
                    brute = Some(h);
                }
            }

            match (via_bvh, brute) {
                (Some(a), Some(b)) => {
                    checked_hits += 1;
                    assert!((a.t - b.t).abs() < 1e-9, "t mismatch: bvh={} brute={}", a.t, b.t);
                    assert!((a.p - b.p).len() < 1e-9, "p mismatch");
                }
                (None, None) => {}
                (a, b) => panic!("hit disagreement: bvh={:?}, brute={:?}", a.map(|h| h.t), b.map(|h| h.t)),
            }
        }
        assert!(checked_hits > 0, "no rays hit any triangle; test is vacuous");
    }

    /// 回帰テスト: NEE のシャドウレイは、原点を ε だけライト方向へ前進させても
    /// tmax を dist−2ε に取っておけば、サンプルした光源自身を誤って遮蔽物として
    /// 検出しない（tmax が dist−ε のままだと丸め次第で約半数が自己遮蔽してしまい、
    /// Cornell box が暗くなる/バンディングが出るバグがあった）。
    #[test]
    fn shadow_ray_does_not_self_hit_sampled_light() {
        use crate::transform::Transform;

        let mut world = World::new();
        // y=1.98 に、下向き法線（-Y）の矩形光源を三角形2枚で構成する。
        let m = |x: f64, z: f64| Vec3::new(0.35 * x, 1.98, 0.35 * z);
        let tris = vec![
            Triangle::new_static(m(-1.0, -1.0), m(1.0, -1.0), m(1.0, 1.0), 0),
            Triangle::new_static(m(-1.0, -1.0), m(1.0, 1.0), m(-1.0, 1.0), 0),
        ];
        world.add_mesh_instance(tris, Transform::identity(), None);
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.6, 3.9, 2.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 1.4, -1.0);
        let mut sampled = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                sampled += 1;
                let to = ls.position - p;
                let dist = to.dot(to).sqrt();
                let wi = to / dist;
                // 終点を光源面の誤差の箱の外へずらした線分。途中で当たるのは光源自身だけ（それ以外の遮蔽物はない）
                let _ = wi;
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                if let Some(h) = world.hit(Ray { o: p, d: seg / seg.len(), time: 0.0 }, 0.0, seg.len()) {
                    assert!(ls.is_light_itself(&h), "shadow ray hit something other than the light");
                }
            }
        }
        assert!(sampled > 0, "no light samples drawn");
    }

    /// 回帰テスト: `light_cdf` は先頭に 0.0 を持つ規約（cdf_search が前提とする
    /// [0, w0, w0+w1, …]）で構築されなければならない。先頭の 0.0 を欠くと
    /// cdf_search が常にインデックス 0 を返し、2 光源の場合は 2 番目の光源が
    /// 一切選ばれなくなる（面積比によらない偏ったサンプリングになる）。
    #[test]
    fn sample_light_selects_lights_area_proportionally() {
        let mut world = World::new();
        let c1 = Vec3::new(-5.0, 0.0, 0.0);
        let r1 = 1.0;
        let c2 = Vec3::new(5.0, 0.0, 0.0);
        let r2 = 2.0;
        world.spheres.push(Sphere { c: c1, r: r1, mat_id: 0 });
        world.spheres.push(Sphere { c: c2, r: r2, mat_id: 0 });
        let mats = vec![Material::DiffuseLight { emit: Color::new(1.0, 1.0, 1.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 0.0, 10.0);
        let n = 10_000;
        let mut count1 = 0;
        let mut count2 = 0;
        for _ in 0..n {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                if (ls.position - c1).len() < (ls.position - c2).len() {
                    count1 += 1;
                } else {
                    count2 += 1;
                }
            }
        }
        // sample_light は法線が p を向いていないサンプルを内部で棄却する（可視半球の
        // みを受理）ため、分母は総試行回数 n ではなく採択されたサンプル数にする。
        // 棄却率は両光源でほぼ等しいため、採択後の内訳は面積比をそのまま反映する。
        let accepted = count1 + count2;
        assert!(accepted > n / 4, "too few accepted samples ({}) to be meaningful", accepted);
        let frac1 = count1 as f64 / accepted as f64;
        let frac2 = count2 as f64 / accepted as f64;
        // 面積比: 4π·1² : 4π·2² = 1 : 4 -> 選択確率 0.2 : 0.8
        assert!((frac1 - 0.2).abs() < 0.05, "sphere1 fraction {} not near 0.2", frac1);
        assert!((frac2 - 0.8).abs() < 0.05, "sphere2 fraction {} not near 0.8", frac2);
    }

    // ---- スムーズシェーディング（頂点法線の補間） ----

    use test_meshes::uv_sphere;

    /// 全頂点の法線が同じなら、補間したシェーディング法線は面法線と（向きも含めて）一致する。
    /// 頂点法線を持たないメッシュとの差が出ないことの最小確認。
    #[test]
    fn uniform_vertex_normals_reproduce_the_face_normal() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::with_normals(tris, vec![n, n, n], vec![[0, 1, 2]]);
        for (x, y) in [(0.0, 0.0), (0.4, -0.3), (-0.3, -0.5), (0.0, 0.8)] {
            let o = Vec3::new(x, y, 3.0);
            let h = mesh.hit(Ray { o, d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
            assert!((h.ns - h.ng).len() < 1e-15, "ns {:?} != ng {:?}", h.ns, h.ng);
            assert!(!h.is_smooth(), "一様な頂点法線は面法線と同一なので is_smooth は false");
        }
    }

    /// 球メッシュの補間法線は解析的な法線と一致し、面法線の誤差は分割を上げるまで大きい。
    ///
    /// 中心が原点の球で頂点法線 = 頂点位置（単位ベクトル）にすると、交点 p は三角形上の
    /// 重心座標の線形結合なので `normalize(Σ bᵢ·vᵢ) == normalize(p)`、つまり補間法線は
    /// **分割によらず解析解と一致する**（丸め誤差のみ）。面法線の方は三角形の大きさぶんずれ、
    /// 分割を上げると 1/nu のオーダーで減る。この 2 つの差が「補間が効いている」ことの証拠。
    #[test]
    fn interpolated_normals_match_the_analytic_sphere_normal() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let mut prev_flat = f64::INFINITY;
        for &nu in &[8usize, 16, 32, 64] {
            let smooth = Mesh::with_normals_from(uv_sphere(center, 1.0, nu, nu / 2, 0, true));
            let flat = Mesh::with_normals_from(uv_sphere(center, 1.0, nu, nu / 2, 0, false));
            let (mut worst_smooth, mut worst_flat) = (0.0f64, 0.0f64);
            let mut rng = Rng::new(7);
            for _ in 0..300 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = center + dir * 4.0;
                let r = Ray { o, d: -dir, time: 0.0 };
                let (Some(hs), Some(hf)) = (smooth.hit(r, 0.0, 1e30), flat.hit(r, 0.0, 1e30)) else { continue };
                let ang = |n: Vec3, p: Vec3| n.dot((p - center).norm()).clamp(-1.0, 1.0).acos();
                worst_smooth = worst_smooth.max(ang(hs.ns, hs.p));
                worst_flat = worst_flat.max(ang(hf.ns, hf.p));
            }
            assert!(worst_smooth < 1e-6, "nu={}: 補間法線は解析解と一致するはず（最大 {} rad）", nu, worst_smooth);
            assert!(worst_flat > 20.0 * worst_smooth.max(1e-9), "nu={}: 面法線 {} は補間 {} より明確に大きいはず", nu, worst_flat, worst_smooth);
            assert!(worst_flat < prev_flat, "nu={}: 面法線の誤差は分割を上げると減るはず（{} -> {}）", nu, prev_flat, worst_flat);
            prev_flat = worst_flat;
        }
        // 面法線は 64 分割でもまだ 1 度以上ずれている（補間の 1e-6 rad とは桁が違う）
        assert!(prev_flat > 0.02, "面法線の最大誤差 {} rad", prev_flat);
    }

    /// 補間しても `Hit::ng` は面法線のまま（＝自己交差回避の基準が動かない）。
    /// さらに、原点ずらしを**シェーディング法線で行うミューテーション**では自己交差が起きることを
    /// 同じテストの中で示す（分離が効いていることの証拠）。
    #[test]
    fn geometric_normal_is_kept_for_ray_offsets() {
        // 大きく傾けた頂点法線を持つ 1 枚の三角形（z=0 平面、面法線は +z）
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let tilted = Vec3::new(0.9, 0.0, 0.436).norm(); // 面法線から約 64 度
        let mesh = Mesh::with_normals(tris, vec![tilted; 3], vec![[0, 1, 2]]);
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!((h.ng - Vec3::new(0.0, 0.0, 1.0)).len() < 1e-15, "ng は面法線のまま: {:?}", h.ng);
        assert!(h.is_smooth() && (h.ns - tilted).len() < 1e-12);

        // 面に沿って浅く出ていく方向。幾何法線基準なら誤差の箱の外に出るので自己交差しない
        let d = Vec3::new(1.0, 0.0, 1e-9).norm();
        let o_geom = crate::geometry::offset_ray_origin(h.p, h.p_error, h.ng, d);
        assert!(mesh.hit(Ray { o: o_geom, d, time: 0.0 }, 0.0, 1e30).is_none(), "幾何法線でずらせば自分に当たらない");
    }

    /// 非一様スケールと鏡像（負のスケール）を含むインスタンスでも、
    /// シェーディング法線は幾何法線と同じ側を向き、単位長のままになる。
    #[test]
    fn instance_transform_keeps_shading_normal_on_the_geometric_side() {
        for scale in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 0.5, 1.0),   // 非一様
            Vec3::new(-1.0, 1.0, 1.0),  // 鏡像
            Vec3::new(-2.0, 0.5, 3.0),  // 鏡像 + 非一様
        ] {
            let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
            let mut world = World::new();
            let xform = Transform::scale(scale);
            world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 24, 12, 0, true), xform, None);
            world.build_lights(&mats);

            let mut rng = Rng::new(11);
            let mut checked = 0;
            for _ in 0..200 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 20.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                assert!((h.ns.len() - 1.0).abs() < 1e-12, "scale {:?}: ns が単位長でない ({})", scale, h.ns.len());
                assert!((h.ng.len() - 1.0).abs() < 1e-12, "scale {:?}: ng が単位長でない ({})", scale, h.ng.len());
                assert!(h.ns.dot(h.ng) > 0.0, "scale {:?}: ns が ng の反対を向いた（{:?} vs {:?}）", scale, h.ns, h.ng);
                checked += 1;
            }
            assert!(checked > 100, "scale {:?}: ヒットが少なすぎる ({})", scale, checked);
        }
    }

    /// `MeshData` の頂点法線を捨てた（face_normals=true 相当）メッシュは、
    /// 頂点法線を最初から持たないメッシュと完全に同じヒットを返す。
    #[test]
    fn face_normals_flag_matches_a_mesh_without_normals() {
        let smooth_dropped = Mesh::with_normals_from(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 16, 8, 0, true).into_flat());
        let never_had = Mesh::with_normals_from(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 16, 8, 0, false));
        assert!(!smooth_dropped.is_smooth() && !never_had.is_smooth());
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let dir = uniform_sphere_dir(&mut rng);
            let r = Ray { o: dir * 5.0, d: -dir, time: 0.0 };
            match (smooth_dropped.hit(r, 0.0, 1e30), never_had.hit(r, 0.0, 1e30)) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                    assert_eq!(a.ng.x.to_bits(), b.ng.x.to_bits());
                    assert_eq!(a.ns.x.to_bits(), b.ns.x.to_bits());
                }
                (None, None) => {}
                _ => panic!("ヒットの有無が食い違う"),
            }
        }
    }

    /// 頂点法線の配列長が三角形数と合わない壊れた入力は、面法線メッシュとして扱う（添字ずれで落ちない）。
    #[test]
    fn mismatched_normal_index_array_is_ignored() {
        let tris = vec![
            Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
            Triangle::new_static(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::with_normals(tris, vec![Vec3::new(0.0, 0.0, 1.0)], vec![[0, 0, 0]]); // 1 個しかない
        assert!(!mesh.is_smooth());
    }

    // ---- スムーズシェーディング: ng / ns の取り違えを検出するテスト ----

    /// UV = (x, y) を張った直角三角形（2 枚で正方形）: ∂p/∂u = +x、∂p/∂v = +y。
    fn uv_plane_world(xform: Transform) -> World {
        let mut world = World::new();
        let (a, b, c, d) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let tris = vec![Triangle::new_static(a, b, c, 0), Triangle::new_static(a, c, d, 0)];
        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        world.add_mesh_data_instance(MeshData::with_uv(tris, uv, vec![[0, 1, 2], [0, 2, 3]]), xform, None);
        world
    }

    fn hit_plane_at(world: &World, x: f64, y: f64, from: Vec3) -> Hit {
        let target = Vec3::new(x, y, 0.0);
        let world_target = {
            // 変換後の板の位置へ撃つため、インスタンスの変換を通した点へ向ける
            let inst = &world.instances[0];
            inst.xform.apply_point_with_error(target, Vec3::new(0.0, 0.0, 0.0)).0
        };
        let d = (world_target - from).norm();
        world.hit(Ray { o: from, d, time: 0.0 }, 0.0, 1e30).expect("plane not hit")
    }

    #[test]
    fn uv_derivatives_of_an_axis_aligned_uv_plane() {
        let world = uv_plane_world(Transform::identity());
        let h = hit_plane_at(&world, 0.7, 0.2, Vec3::new(0.7, 0.2, 3.0));
        let (dpdu, dpdv) = world.surface_tangents(&h, 0.0).unwrap();
        assert!((dpdu - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-12, "{:?}", (dpdu.x, dpdu.y, dpdu.z));
        assert!((dpdv - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-12, "{:?}", (dpdv.x, dpdv.y, dpdv.z));
    }

    /// 接空間は共有辺をまたいでも連続（同じ平面の 2 三角形で向きが一致）。
    #[test]
    fn tangents_are_continuous_across_a_shared_edge() {
        let world = uv_plane_world(Transform::identity());
        // 対角線 (0,0)-(1,1) をまたぐ 2 点
        let h1 = hit_plane_at(&world, 0.6, 0.4, Vec3::new(0.6, 0.4, 3.0));
        let h2 = hit_plane_at(&world, 0.4, 0.6, Vec3::new(0.4, 0.6, 3.0));
        assert_ne!(h1.prim_id, h2.prim_id);
        let (a, _) = world.surface_tangents(&h1, 0.0).unwrap();
        let (b, _) = world.surface_tangents(&h2, 0.0).unwrap();
        assert!(a.norm().dot(b.norm()) > 1.0 - 1e-9);
    }

    /// インスタンス変換: 接ベクトルは順方向の `A·t`。せん断・鏡像・非一様スケールでも、
    /// 変換後の 2 点の差（板の上の実際の方向）と一致する。
    ///
    /// ミューテーション検出: `surface_tangents` の `apply_vec` を `apply_normal`（逆転置 + 正規化）に
    /// 差し替えると、せん断変換で dpdu が実際の面上の方向からずれてこのテストが落ちる。
    #[test]
    fn tangents_follow_the_instance_transform_including_shear() {
        for (name, xform) in tricky_transforms() {
            let world = uv_plane_world(xform);
            let (u0, v0) = (0.3, 0.2);
            let du = 1e-3;
            let (pu0, pu1) = (Vec3::new(u0, v0, 0.0), Vec3::new(u0 + du, v0, 0.0));
            let (pv1, w) = (Vec3::new(u0, v0 + du, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let map = |p: Vec3| xform.apply_point_with_error(p, w).0;
            let expect_u = (map(pu1) - map(pu0)) / du;
            let expect_v = (map(pv1) - map(pu0)) / du;
            // 板を撃つ点はワールド側の任意の位置から（変換後の板に向けて）
            let from = map(Vec3::new(u0, v0, 0.0)) + xform.apply_normal(Vec3::new(0.0, 0.0, 1.0)) * 3.0;
            let h = hit_plane_at(&world, u0, v0, from);
            let (dpdu, dpdv) = world.surface_tangents(&h, 0.0).unwrap();
            assert!((dpdu - expect_u).len() < 1e-9 * (1.0 + expect_u.len()), "{}: dpdu {:?} != {:?}", name, (dpdu.x, dpdu.y, dpdu.z), (expect_u.x, expect_u.y, expect_u.z));
            assert!((dpdv - expect_v).len() < 1e-9 * (1.0 + expect_v.len()), "{}: dpdv", name);
        }
    }

    /// UV 三角形が縮退（3 頂点とも同じ UV、または一直線）なら `None`。UV の無い三角形・球も `None`。
    #[test]
    fn degenerate_uv_and_uvless_hits_have_no_tangents() {
        for uv in [vec![[0.5, 0.5]; 3], vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]] {
            let mut world = World::new();
            let tris = vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
            let data = MeshData::with_uv(tris, uv, vec![[0, 1, 2]]);
            world.add_mesh_data_instance(data, Transform::identity(), None);
            let h = world.hit(Ray { o: Vec3::new(0.2, 0.2, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
            assert!(world.surface_tangents(&h, 0.0).is_none());
            assert_eq!(world.meshes()[0].degenerate_uv_count(), 1);
        }
        // UV 無しのメッシュ
        let mut world = World::new();
        world.add_mesh_instance(
            vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)],
            Transform::identity(),
            None,
        );
        let h = world.hit(Ray { o: Vec3::new(0.2, 0.2, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(world.surface_tangents(&h, 0.0).is_none());
        // 球
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        let h = world.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(world.surface_tangents(&h, 0.0).is_none());
    }

    /// モーションブラー: `time` で辺を補間した接ベクトルは、その時刻の頂点から直接求めたものと一致する。
    #[test]
    fn tangents_interpolate_with_shutter_time() {
        let (a0, b0, c0) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        // 閉時刻では x 方向に 3 倍に伸ばす
        let s = |v: Vec3| Vec3::new(v.x * 3.0, v.y, v.z);
        let tri = Triangle { v0_0: a0, v1_0: b0, v2_0: c0, mat_id: 0 };
        let mut data = MeshData::with_uv(vec![tri], vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]]);
        data.motion = vec![[s(a0), s(b0), s(c0)]];
        let mesh = Mesh::with_normals_from(data);
        for &t in &[0.0, 0.25, 1.0] {
            let (dpdu, dpdv) = mesh.uv_derivatives(0, t).unwrap();
            let (v0, v1, v2) = mesh.vertices_at(0, t).unwrap();
            assert!((dpdu - (v1 - v0)).len() < 1e-12, "t={}", t);
            assert!((dpdv - (v2 - v0)).len() < 1e-12, "t={}", t);
        }
    }

    /// せん断・鏡像を含む「意地悪な」変換の一覧（法線の逆転置がもっとも効く形）。
    fn tricky_transforms() -> Vec<(&'static str, Transform)> {
        let m = |a: [[f64; 4]; 4]| Transform::from_matrix4(a);
        vec![
            ("shear x+=2z", m([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear+mirror x", m([[-1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear+mirror y", m([[1.0, 3.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear xyz + mirror", m([[-1.0, 1.5, 2.5, 0.0], [0.5, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("anisotropic + shear + mirror", m([[-4.0, 2.0, 0.0, 0.0], [0.0, 0.25, 3.0, 0.0], [1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("mirror xyz", Transform::scale(Vec3::new(-1.0, -1.0, -1.0))),
            ("anisotropic", Transform::scale(Vec3::new(-5.0, 0.2, 2.0))),
        ]
    }

    /// インスタンス変換の後も `ns · ng > 0` が保たれる（せん断 + 鏡像を含む）。
    ///
    /// ミューテーション検出: `World::hit` の変換後の `face_forward` を外すと、
    /// せん断と鏡像を組み合わせた変換で `ns` が `ng` の反対側へ回り、このテストが落ちる。
    /// 粗い球（8x4 = 48 三角形）を使うのは、オブジェクト空間での `ns` と `ng` の開きが
    /// 大きいほど逆転置による向きの入れ替わりが起きやすいから。
    #[test]
    fn shading_normal_stays_on_the_geometric_side_through_shearing_transforms() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        for (name, xform) in tricky_transforms() {
            let mut world = World::new();
            world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 8, 4, 0, true), xform, None);
            world.build_lights(&mats);
            let mut rng = Rng::new(17);
            let (mut checked, mut worst) = (0usize, f64::INFINITY);
            for _ in 0..4000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 40.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                let dot = h.ns.dot(h.ng);
                worst = worst.min(dot);
                assert!(dot > 0.0, "{}: ns が ng の裏へ回った（ns·ng = {}）", name, dot);
                assert!((h.ns.len() - 1.0).abs() < 1e-12, "{}: ns が単位長でない", name);
                checked += 1;
            }
            assert!(checked > 500, "{}: ヒットが少なすぎる ({})", name, checked);
            let _ = worst;
        }
    }

    /// **シェーディング法線はインスタンス変換で実際に変換される**（オブジェクト空間のまま使われない）。
    ///
    /// 頂点法線が一様なメッシュ（面ごとに補間値が定数）なら、変換後の `ns` は
    /// `face_forward(apply_normal(ns_obj), ng)` と一致するはず。正しい実装を再実装せずに書ける形にしてある。
    /// 重心座標での補間（`Σ bᵢ·vᵢ` を計算してから正規化）が入るぶん完全なビット一致にはならないので、
    /// 許容差 1e-12 で比べる（取り違えたときの差は 0.1 rad 以上なので、これで十分に鋭い）。
    ///
    /// ミューテーション検出: `World::hit` で `ns` を変換せず `h_obj.ns` のまま使うと落ちる。
    /// `ns·ng > 0` と単位長だけを見るテストでは、この取り違えは通り抜ける
    /// （オブジェクト空間の法線でも両方の条件を満たしてしまうため）。
    #[test]
    fn instance_transform_actually_transforms_the_shading_normal() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        // 面法線 +z から 50 度傾けた一様な頂点法線。回転・非一様・せん断で「変換しない」と
        // 明確に違う向きになる。
        let ns_obj = Vec3::new(0.766, 0.0, 0.643).norm();
        let mut meaningful_transforms = 0usize;
        for (name, xform) in tricky_transforms() {
            let mut world = World::new();
            world.add_mesh_data_instance(test_meshes::tilted_quad(4.0, ns_obj, 0), xform, None);
            world.build_lights(&mats);

            let expected_raw = xform.apply_normal(ns_obj);
            // 点対称（-I）のように、法線の向きが符号だけしか変わらない変換では
            // 「変換しない」ミューテーションと区別できない。その変換では下の空回り判定を外す。
            let direction_changes = expected_raw.dot(ns_obj).abs() < 1.0 - 1e-12;
            if direction_changes {
                meaningful_transforms += 1;
            }
            let mut rng = Rng::new(37);
            let mut checked = 0usize;
            let mut differs_from_object_space = 0usize;
            for _ in 0..2000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 30.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                let expected = face_forward(expected_raw, h.ng);
                assert!(
                    (h.ns - expected).len() < 1e-12,
                    "{}: ns が変換後の値と違う（ns = {:?}, 期待 {:?}）", name, h.ns, expected
                );
                // オブジェクト空間の法線とは実際に違う（テストが空回りしていないこと）
                if (h.ns - face_forward(ns_obj, h.ng)).len() > 1e-9 {
                    differs_from_object_space += 1;
                }
                checked += 1;
            }
            assert!(checked > 200, "{}: ヒットが少なすぎる ({})", name, checked);
            if direction_changes {
                assert!(
                    differs_from_object_space > checked / 2,
                    "{}: 変換前後で ns が変わらない配置ばかり（テストが空回り）", name
                );
            }
        }
        assert!(meaningful_transforms >= 4, "向きを変える変換が少なすぎる（{}）", meaningful_transforms);
    }

    /// **幾何法線はスムーズ化の影響を受けない**: 同じ形状・同じ変換で、頂点法線の有無だけが違う
    /// 2 つのインスタンスは、同じレイに対して**ビット単位で同じ `ng`** を返す（`ns` だけが違う）。
    ///
    /// ミューテーション検出: `World::hit` で `ng` と `ns` を取り違える（入れ替える）と落ちる。
    /// 幾何法線は自己交差回避・表裏判定・光源 pdf の基準なので、ここが補間値に化けると静かに壊れる。
    #[test]
    fn instance_geometric_normal_is_unaffected_by_vertex_normals() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        for (name, xform) in tricky_transforms() {
            let mut smooth_world = World::new();
            smooth_world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, true), xform, None);
            smooth_world.build_lights(&mats);
            let mut flat_world = World::new();
            flat_world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, false), xform, None);
            flat_world.build_lights(&mats);

            let mut rng = Rng::new(23);
            let (mut checked, mut smooth_seen) = (0usize, 0usize);
            for _ in 0..2000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 40.0;
                let r = Ray { o, d: -dir, time: 0.0 };
                match (smooth_world.hit(r, 0.0, 1e30), flat_world.hit(r, 0.0, 1e30)) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.t.to_bits(), b.t.to_bits(), "{}: 交差距離が違う", name);
                        assert_eq!(a.ng.x.to_bits(), b.ng.x.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(a.ng.y.to_bits(), b.ng.y.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(a.ng.z.to_bits(), b.ng.z.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(b.ns.x.to_bits(), b.ng.x.to_bits(), "{}: 面法線メッシュは ns == ng", name);
                        if a.is_smooth() {
                            smooth_seen += 1;
                        }
                        checked += 1;
                    }
                    (None, None) => {}
                    _ => panic!("{}: ヒットの有無が食い違う", name),
                }
            }
            assert!(checked > 300, "{}: ヒットが少なすぎる ({})", name, checked);
            assert!(smooth_seen > checked / 2, "{}: 補間が効いているヒットが少なすぎる（テストが空回り）", name);
        }
    }

    // ---- テクスチャ座標（UV）の配管 ----

    /// 頂点 UV は重心座標で補間される（三角形の各頂点で厳密に頂点 UV になる）。
    #[test]
    fn vertex_uv_is_interpolated_by_barycentric_coordinates() {
        // z=0 平面の直角三角形。v0=(0,0) v1=(1,0) v2=(0,1) に UV [0,0] [1,0] [0,1] を割り当てる
        let tris = vec![Triangle::new_static(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::build(
            tris, Vec::new(), Vec::new(), Vec::new(),
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]],
        );
        assert!(mesh.has_uv());
        // この割り当てでは UV は (x, y) と一致するので、当てた位置から期待値が直に決まる
        for (x, y) in [(0.05, 0.05), (0.5, 0.25), (0.25, 0.5), (0.3, 0.3), (0.8, 0.1)] {
            let h = mesh
                .hit(Ray { o: Vec3::new(x, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
                .unwrap_or_else(|| panic!("({}, {}) に当たらない", x, y));
            assert!((h.uv.0 - x).abs() < 1e-12 && (h.uv.1 - y).abs() < 1e-12,
                    "uv = {:?}, 期待 ({}, {})", h.uv, x, y);
        }
    }

    /// UV を持たないメッシュのヒットは uv = (0, 0)（テクスチャを引かない既定値）。
    #[test]
    fn mesh_without_uv_reports_zero_uv() {
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::new(tris);
        assert!(!mesh.has_uv());
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert_eq!(h.uv, (0.0, 0.0));
    }

    /// UV 添字の配列長が三角形数と合わない壊れた入力は、UV 無しとして扱う（添字ずれで落ちない）。
    #[test]
    fn mismatched_uv_index_array_is_ignored() {
        let tris = vec![
            Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
            Triangle::new_static(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::build(tris, Vec::new(), Vec::new(), Vec::new(), vec![[0.0, 0.0]], vec![[0, 0, 0]]); // 1 個しかない
        assert!(!mesh.has_uv());
    }

    /// UV あり／なしが混在するメッシュでも、UV なしの三角形は (0, 0) になる（`NO_UV` の分岐）。
    #[test]
    fn mesh_with_mixed_uv_falls_back_per_triangle() {
        let tris = vec![
            Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, -0.05, 0.0), 0),
            Triangle::new_static(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(0.0, 0.05, 0.0), Vec3::new(1.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::build(
            tris, Vec::new(), Vec::new(), Vec::new(),
            vec![[0.3, 0.4]], vec![[0, 0, 0], [NO_UV; 3]],
        );
        let shoot = |y: f64| mesh.hit(Ray { o: Vec3::new(0.0, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30);
        let a = shoot(-0.5).expect("UV つきの三角形に当たらない");
        assert!((a.uv.0 - 0.3).abs() < 1e-12 && (a.uv.1 - 0.4).abs() < 1e-12, "uv = {:?}", a.uv);
        let b = shoot(0.5).expect("UV なしの三角形に当たらない");
        assert_eq!(b.uv, (0.0, 0.0));
    }

    /// インスタンス変換は UV を変えない（UV はオブジェクト空間の属性）。
    #[test]
    fn instance_transform_does_not_change_uv() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        let tris = vec![Triangle::new_static(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let data = MeshData::with_uv(
            tris, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]]);
        let mut world = World::new();
        world.add_mesh_data_instance(data, Transform::scale(Vec3::new(4.0, 0.5, 2.0)), None);
        world.build_lights(&mats);
        // 変換後の (x, y) = (4·u, 0.5·v) に当たるレイを撃つ
        for (u, v) in [(0.2, 0.3), (0.5, 0.25), (0.1, 0.8)] {
            let r = Ray { o: Vec3::new(4.0 * u, 0.5 * v, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30).unwrap_or_else(|| panic!("({}, {}) に当たらない", u, v));
            assert!((h.uv.0 - u).abs() < 1e-12 && (h.uv.1 - v).abs() < 1e-12,
                    "uv = {:?}, 期待 ({}, {})", h.uv, u, v);
        }
    }

    /// 球の UV は Mitsuba の球面座標（u = φ/2π、v = θ/π、極は ±z）。
    #[test]
    fn sphere_uv_follows_the_mitsuba_spherical_parameterisation() {
        let s = Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 };
        let hit_from = |dir: Vec3| {
            s.hit(Ray { o: dir * 5.0, d: -dir, time: 0.0 }, 0.0, 1e30).expect("球に当たらない")
        };
        // +x 方向（φ = 0, θ = π/2）
        let h = hit_from(Vec3::new(1.0, 0.0, 0.0));
        assert!(h.uv.0.abs() < 1e-12 && (h.uv.1 - 0.5).abs() < 1e-12, "+x: {:?}", h.uv);
        // +y 方向（φ = π/2）
        let h = hit_from(Vec3::new(0.0, 1.0, 0.0));
        assert!((h.uv.0 - 0.25).abs() < 1e-12 && (h.uv.1 - 0.5).abs() < 1e-12, "+y: {:?}", h.uv);
        // −x 方向（φ = π）
        let h = hit_from(Vec3::new(-1.0, 0.0, 0.0));
        assert!((h.uv.0 - 0.5).abs() < 1e-12, "−x: {:?}", h.uv);
        // +z（北極、θ = 0）と −z（南極、θ = π）
        let h = hit_from(Vec3::new(0.0, 0.0, 1.0));
        assert!(h.uv.1.abs() < 1e-9, "+z 極: {:?}", h.uv);
        let h = hit_from(Vec3::new(0.0, 0.0, -1.0));
        assert!((h.uv.1 - 1.0).abs() < 1e-9, "−z 極: {:?}", h.uv);
        // u は常に [0, 1]
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let d = uniform_sphere_dir(&mut rng);
            let h = hit_from(d);
            assert!((0.0..=1.0).contains(&h.uv.0) && (0.0..=1.0).contains(&h.uv.1), "uv = {:?}", h.uv);
        }
    }

    /// **法線あり／なしが混在するメッシュ**を交差判定（描画が通る経路）で扱える。
    ///
    /// `f 1//1 2//2 3//3` と `f 1 2 3` が混ざった OBJ は実在する。ローダーは法線を持たない
    /// 三角形に番兵 `NO_NORMAL` を入れ、`Mesh::shading_normal` がそれを見て面法線に落とす。
    ///
    /// ミューテーション検出: 番兵チェック（`if idx[0] == NO_NORMAL { return None }`）を外すと、
    /// 法線を持たない三角形に当たった瞬間に `vn[u32::MAX]` で**添字外アクセスのパニック**になる。
    /// `obj_loader` 側には混在のパーステストがあるが、そこはヒットを通らないので気づけない。
    #[test]
    fn mesh_with_mixed_vertex_normals_is_hit_without_panicking() {
        // 三角形 0（z=0 平面、y<0 側）は傾いた頂点法線つき、三角形 1（y>0 側）は法線なし
        let tilted = Vec3::new(0.6, 0.0, 0.8).norm();
        let tris = vec![
            Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, -0.05, 0.0), 0),
            Triangle::new_static(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(0.0, 0.05, 0.0), Vec3::new(1.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::with_normals(tris, vec![tilted], vec![[0, 0, 0], [NO_NORMAL; 3]]);
        assert!(mesh.is_smooth(), "混在メッシュはスムーズ扱い（三角形ごとに分岐する）");

        let shoot = |y: f64| {
            mesh.hit(Ray { o: Vec3::new(0.0, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
        };
        // 頂点法線を持つ側: 補間法線が使われる
        let h_smooth = shoot(-0.5).expect("法線つきの三角形に当たらない");
        assert!(h_smooth.is_smooth(), "頂点法線を持つ三角形で補間されていない");
        assert!((h_smooth.ns - tilted).len() < 1e-12, "補間法線が頂点法線と違う: {:?}", h_smooth.ns);
        // 頂点法線を持たない側: 面法線のまま（ここで番兵の分岐を踏む）
        let h_flat = shoot(0.5).expect("法線なしの三角形に当たらない");
        assert!(!h_flat.is_smooth(), "法線なしの三角形が補間されている");
        assert_eq!(h_flat.ns.z.to_bits(), h_flat.ng.z.to_bits(), "法線なしの三角形は ns == ng");

        // インスタンス経由（World::hit）でも同じ。変換つきでも番兵の分岐を踏む
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }];
        let mut world = World::new();
        world.add_mesh_data_instance(
            MeshData {
                tris: mesh.tris.clone(),
                vn: vec![tilted],
                tri_vn: vec![[0, 0, 0], [NO_NORMAL; 3]],
                uv: Vec::new(),
                tri_uv: Vec::new(),
                motion: Vec::new(),
            },
            Transform::scale(Vec3::new(2.0, 0.5, 1.5)),
            None,
        );
        world.build_lights(&mats);
        let mut smooth_hits = 0;
        let mut flat_hits = 0;
        for i in 0..40 {
            let y = -1.2 + 2.4 * (i as f64) / 39.0;
            let r = Ray { o: Vec3::new(0.0, y, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            if let Some(h) = world.hit(r, 0.0, 1e30) {
                if h.is_smooth() { smooth_hits += 1 } else { flat_hits += 1 }
            }
        }
        assert!(smooth_hits > 0 && flat_hits > 0,
                "両方の三角形を通っていない（smooth {} / flat {}）", smooth_hits, flat_hits);
    }

    /// 頂点法線が面法線と逆を向いている（壊れた、あるいは巻き順の違う）メッシュでも、
    /// `Hit::ns` は `ng` と同じ側に揃う。
    ///
    /// ミューテーション検出: `Mesh::shading_normal` の `face_forward` を外すと落ちる。
    #[test]
    fn shading_normal_is_flipped_to_the_geometric_side_at_the_mesh() {
        // 面法線は +z（反時計回り）だが、頂点法線は全て -z を向いている
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let back = Vec3::new(0.0, 0.0, -1.0);
        let mesh = Mesh::with_normals(tris, vec![back; 3], vec![[0, 1, 2]]);
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(h.ns.dot(h.ng) > 0.0, "ns が ng の裏を向いたまま: ns={:?} ng={:?}", h.ns, h.ng);
    }

    /// **スムーズな発光メッシュの `light_pdf` は幾何法線で計算される**。
    ///
    /// 光源上の点の pdf は「その点の面が参照点をどれだけ斜めに見るか」で決まるので、
    /// シェーディング法線を使うと MIS の重みが狂う（BSDF サンプリングで光源に当たった経路の重み）。
    /// 頂点法線の有無だけが違う 2 つの発光メッシュで、同じレイのヒットに対する `light_pdf` が
    /// ビット単位で一致することを確かめる。
    ///
    /// ミューテーション検出: `World::light_pdf` の `hit.ng` を `hit.ns` にすると落ちる。
    /// （既存の `light_pdf` のテストは球・矩形の光源しか使っておらず、`ns == ng` なので素通りする。）
    #[test]
    fn light_pdf_on_a_smooth_emissive_mesh_uses_the_geometric_normal() {
        let mats = vec![Material::DiffuseLight { emit: Color::new(5.0, 5.0, 5.0) }];
        let build = |smooth: bool| {
            let mut w = World::new();
            w.add_mesh_data_instance(
                uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, smooth), Transform::identity(), None);
            w.build_lights(&mats);
            w
        };
        let (smooth_world, flat_world) = (build(true), build(false));
        let mut rng = Rng::new(29);
        let (mut checked, mut smooth_hits) = (0usize, 0usize);
        for _ in 0..1500 {
            let dir = uniform_sphere_dir(&mut rng);
            let from = dir * 6.0;
            let r = Ray { o: from, d: -dir, time: 0.0 };
            let (Some(hs), Some(hf)) = (smooth_world.hit(r, 0.0, 1e30), flat_world.hit(r, 0.0, 1e30)) else { continue };
            let ps = smooth_world.light_pdf(from, 0.0, &hs);
            let pf = flat_world.light_pdf(from, 0.0, &hf);
            assert!(ps > 0.0, "発光メッシュへのヒットなのに pdf が 0");
            assert_eq!(ps.to_bits(), pf.to_bits(), "light_pdf が頂点法線に影響されている（{} vs {}）", ps, pf);
            if hs.is_smooth() {
                smooth_hits += 1;
            }
            checked += 1;
        }
        assert!(checked > 300, "ヒットが少なすぎる ({})", checked);
        assert!(smooth_hits > checked / 2, "補間が効いているヒットが少なすぎる（テストが空回り）");
    }
    // ---- トップレベル BVH（TLAS）----

    /// 乱数のインスタンス（大小・回転・非一様スケール・同じ位置の重なりを含む）と球からなるワールド。
    fn random_tlas_world(n_inst: usize, n_sph: usize, seed: u64, same_place: bool) -> World {
        let mut rng = Rng::new(seed);
        let mut u = |a: f64, b: f64| a + (b - a) * rng.next_f64();
        let mut world = World::new();
        let tri = |k: f64| vec![
            Triangle::new_static(Vec3::new(-k, -k, 0.0), Vec3::new(k, -k, 0.0), Vec3::new(0.0, k, 0.5), 0),
            Triangle::new_static(Vec3::new(-k, -k, 0.5), Vec3::new(k, -k, 0.5), Vec3::new(0.0, k, 0.0), 0),
        ];
        world.add_mesh_instance(tri(1.0), Transform::identity(), None);
        let cube_like = world.instance_mesh_id(0);
        for i in 0..n_inst {
            let (p, sc) = if same_place {
                (Vec3::new(0.0, 0.0, 0.0), 1.0)
            } else if i % 5 == 0 {
                (Vec3::new(u(-30.0, 30.0), u(-30.0, 30.0), u(-30.0, 30.0)), u(20.0, 60.0)) // 巨大
            } else if i % 5 == 1 {
                (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(1e-4, 1e-3)) // 極小
            } else {
                (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(0.3, 2.0))
            };
            let x = Transform::translate(p).compose(
                Transform::rotate(Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.1, 1.0)), u(0.0, 360.0))
                    .compose(Transform::scale(Vec3::new(sc, sc * u(0.5, 1.5), sc))),
            );
            world.add_instance_of(cube_like, x, Some(i % 3));
        }
        for j in 0..n_sph {
            let (c, r) = if same_place { (Vec3::new(0.0, 0.0, 0.0), 1.0) } else { (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(0.05, 2.0)) };
            world.add_sphere(Sphere { c, r, mat_id: j % 4 });
        }
        world
    }

    fn assert_tlas_hit(a: &Option<Hit>, b: &Option<Hit>, what: &str) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "{what}: t");
                assert_eq!((a.inst_id, a.prim_id, a.mat_id), (b.inst_id, b.prim_id, b.mat_id), "{what}: ids");
                assert_eq!((a.p.x.to_bits(), a.ng.z.to_bits(), a.uv.0.to_bits()), (b.p.x.to_bits(), b.ng.z.to_bits(), b.uv.0.to_bits()), "{what}: geometry");
            }
            _ => panic!("{what}: {:?} vs {:?}", a.map(|h| h.t), b.map(|h| h.t)),
        }
    }

    /// TLAS 経由の `hit` / `occluded` は線形総当たりと完全に一致する（t のビットまで、`inst_id`・`prim_id`・`mat_id` も）。
    #[test]
    fn tlas_matches_linear_bruteforce() {
        let mut rng = Rng::new(99);
        for (n_inst, n_sph, same) in [(30, 30, false), (60, 0, false), (0, 60, false), (25, 5, false), (40, 40, true), (1, 30, false)] {
            let world = random_tlas_world(n_inst, n_sph, 7 + n_inst as u64, same);
            assert!(world.tlas().is_some(), "{} prims で TLAS が使われるはず", n_inst + n_sph + 1);
            for k in 0..3000 {
                let o = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 40.0 - 20.0);
                let d = uniform_sphere_dir(&mut rng);
                let r = Ray { o, d, time: 0.0 };
                let tmin = if k % 7 == 0 { 1e-3 } else { 0.0 };
                let tmax = if k % 5 == 0 { 15.0 } else { 1e30 };
                let what = format!("case ({n_inst},{n_sph},{same}) ray {k}");
                assert_tlas_hit(&world.hit(r, tmin, tmax), &world.hit_linear(r, tmin, tmax), &what);
                for skip in [None, Some((None, 0usize)), Some((Some(1usize), 0usize))] {
                    assert_eq!(world.occluded(r, tmin, tmax, skip), world.occluded_linear(r, tmin, tmax, skip), "{what}: occluded {:?}", skip);
                }
            }
        }
    }

    /// 退化ケース: 空、インスタンスだけ／球だけ、1 個だけ、しきい値の前後。どれも落ちず線形と一致する。
    #[test]
    fn tlas_degenerate_worlds() {
        let r = Ray { o: Vec3::new(0.3, 0.2, 9.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let empty = World::new();
        assert!(empty.hit(r, 0.0, 1e30).is_none() && !empty.occluded(r, 0.0, 1e30, None));
        for (ni, ns) in [(0, 1), (1, 0), (1, 1), (0, TLAS_MIN_PRIMS - 1), (0, TLAS_MIN_PRIMS), (0, TLAS_MIN_PRIMS + 1), (TLAS_MIN_PRIMS, 0)] {
            let world = random_tlas_world(ni, ns, 3, false);
            let used = world.tlas().is_some();
            assert_eq!(used, ni + ns + 1 >= TLAS_MIN_PRIMS, "({ni},{ns})");
            assert_tlas_hit(&world.hit(r, 0.0, 1e30), &world.hit_linear(r, 0.0, 1e30), "degenerate");
            assert_eq!(world.occluded(r, 0.0, 1e30, None), world.occluded_linear(r, 0.0, 1e30, None));
        }
        // 構築後にジオメトリを足しても古い TLAS を使わない（足した球にも当たる）
        let mut world = random_tlas_world(TLAS_MIN_PRIMS, 0, 5, false);
        let _ = world.hit(r, 0.0, 1e30);
        world.add_sphere(Sphere { c: Vec3::new(0.3, 0.2, 50.0), r: 0.5, mat_id: 0 });
        assert!(world.hit(r, 0.0, 1e30).is_some_and(|h| h.t < 9.0 + 1e-9 || h.mat_id == 0));
        assert_tlas_hit(&world.hit(r, 0.0, 1e30), &world.hit_linear(r, 0.0, 1e30), "after add");
    }

    // ---- アニメーション変換 ----

    fn tri_mesh(k: f64) -> Vec<Triangle> {
        vec![
            Triangle::new_static(Vec3::new(-k, -k, -0.2), Vec3::new(k, -k, 0.2), Vec3::new(0.0, k, 0.0), 0),
            Triangle::new_static(Vec3::new(-k, -k, 0.3), Vec3::new(k, -k, -0.3), Vec3::new(0.0, k, 0.4), 0),
        ]
    }

    fn down_ray(x: f64, y: f64, time: f64) -> Ray {
        Ray { o: Vec3::new(x, y, 20.0), d: Vec3::new(0.0, 0.0, -1.0), time }
    }

    /// `time = 0` で開の位置、`time = 1` で閉の位置、`time = 0.5` でちょうど中間に交差する（平行移動）。
    #[test]
    fn animated_instance_is_at_open_middle_and_close_positions() {
        let mut world = World::new();
        let id = world.add_mesh_instance(tri_mesh(0.5), Transform::translate(Vec3::new(-2.0, 0.0, 0.0)), None);
        assert!(world.set_instance_end_transform(id, Transform::translate(Vec3::new(2.0, 1.0, 0.0))));
        for (time, cx, cy) in [(0.0, -2.0, 0.0), (0.5, 0.0, 0.5), (1.0, 2.0, 1.0)] {
            let h = world.hit(down_ray(cx, cy - 0.2, time), 0.0, 1e30).unwrap_or_else(|| panic!("time {time}: 当たるはず"));
            assert!((h.p.x - cx).abs() < 1e-12 && (h.p.y - (cy - 0.2)).abs() < 1e-12, "time {time}: {:?}", h.p);
            assert!(world.hit(down_ray(cx + 3.0, cy, time), 0.0, 1e30).is_none());
        }
        // 開の位置のままで時刻 1 を撃つと外れる（動いている）
        assert!(world.hit(down_ray(-2.0, -0.2, 1.0), 0.0, 1e30).is_none());
        // 遮蔽判定も同じ時刻の位置で判定する
        let occ = |x: f64, y: f64, time: f64| world.occluded(down_ray(x, y, time), 0.0, 1e30, None);
        assert!(occ(0.0, 0.3, 0.5) && !occ(0.0, 0.3, 0.0));
    }

    /// 掃過ボリュームの保守性: 回転・平行移動・非一様スケールのアニメーション変換に、ランダムなレイ × 時刻を
    /// 2 万本投げ、箱で棄却する通常の経路と、箱の棄却を外した総当たり（箱を巨大にした同じワールド）が
    /// ビット単位で一致する。取りこぼしがあれば保守的でない（TLAS の有無も両方: 24 個 = 閾値超）。
    #[test]
    fn swept_bounds_are_conservative_against_no_culling() {
        for n_inst in [3usize, 24] {
            let rng = std::cell::RefCell::new(Rng::new(31 + n_inst as u64));
            let u = |a: f64, b: f64| a + (b - a) * rng.borrow_mut().next_f64();
            let mut world = World::new();
            for i in 0..n_inst {
                let p0 = Vec3::new(u(-6.0, 6.0), u(-6.0, 6.0), u(-2.0, 2.0));
                let p1 = Vec3::new(u(-6.0, 6.0), u(-6.0, 6.0), u(-2.0, 2.0));
                let ax = Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.2, 1.0));
                let s0 = u(0.3, 1.5);
                let start = Transform::translate(p0).compose(Transform::rotate(ax, u(0.0, 360.0))).compose(Transform::scale(Vec3::new(s0, s0 * u(0.5, 2.0), s0)));
                let end = Transform::translate(p1)
                    .compose(Transform::rotate(Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.2, 1.0)), u(0.0, 360.0)))
                    .compose(Transform::scale(Vec3::new(u(0.3, 2.5), u(0.3, 2.5), u(0.3, 2.5))));
                let id = world.add_mesh_instance(tri_mesh(1.5), start, Some(i % 3));
                assert!(world.set_instance_end_transform(id, end));
            }
            let mut reference = World::new();
            reference.meshes = world.meshes.iter().map(|m| Mesh::new(m.tris.clone())).collect();
            reference.instances = world.instances.clone();
            let huge = Aabb { min: Vec3::new(-1e9, -1e9, -1e9), max: Vec3::new(1e9, 1e9, 1e9) };
            for inst in &mut reference.instances {
                inst.world_bounds = huge;
            }
            let mut hits = 0;
            for _ in 0..20_000 {
                let o = Vec3::new(u(-14.0, 14.0), u(-14.0, 14.0), u(-14.0, 14.0));
                let d = uniform_sphere_dir(&mut rng.borrow_mut());
                let r = Ray { o, d, time: u(0.0, 1.0) };
                let (a, b) = (world.hit(r, 0.0, 1e30), reference.hit(r, 0.0, 1e30));
                assert_eq!(a.is_some(), b.is_some(), "取りこぼし: t={:?} time={}", b.map(|h| h.t), r.time);
                if let (Some(a), Some(b)) = (a, b) {
                    hits += 1;
                    assert_eq!((a.t.to_bits(), a.inst_id, a.prim_id), (b.t.to_bits(), b.inst_id, b.prim_id));
                }
                assert_eq!(world.occluded(r, 0.0, 30.0, None), reference.occluded(r, 0.0, 30.0, None));
            }
            assert!(hits > 200, "テストが自明でない程度に当たる: {hits}");
        }
    }

    /// 頂点モーションとアニメーション変換の併用: どちらも効く。
    #[test]
    fn vertex_motion_and_animated_transform_combine() {
        let tri = Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0);
        let data = MeshData {
            tris: vec![tri],
            vn: Vec::new(),
            tri_vn: Vec::new(),
            uv: Vec::new(),
            tri_uv: Vec::new(),
            // 閉: 頂点が x に +2
            motion: vec![[Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 0.0)]],
        };
        let mut world = World::new();
        let id = world.add_mesh_data_instance(data, Transform::identity(), None);
        // 変換は y に +5
        assert!(world.set_instance_end_transform(id, Transform::translate(Vec3::new(0.0, 5.0, 0.0))));
        for time in [0.0, 0.25, 0.5, 1.0] {
            let (x, y) = (0.2 + 2.0 * time, 0.2 + 5.0 * time);
            assert!(world.hit(down_ray(x, y, time), 0.0, 1e30).is_some(), "time {time}: 両方の動きを足した位置に当たる");
            assert!(world.hit(down_ray(0.2, y, time), 0.0, 1e30).is_none() == (time > 0.0), "頂点モーションだけ無視した位置は外れる");
            assert!(world.hit(down_ray(x, 0.2, time), 0.0, 1e30).is_none() == (time > 0.0), "変換のモーションだけ無視した位置は外れる");
        }
    }

    /// アニメーション変換を持たないインスタンスは、他にアニメーションするインスタンスがあっても、単独のときとビット一致する
    /// （静止は従来の経路: `anim` が `None`）。
    #[test]
    fn static_instance_is_bit_identical_next_to_animated_ones() {
        let xf = Transform::translate(Vec3::new(0.3, 0.1, 0.0)).compose(Transform::rotate(Vec3::new(0.2, 1.0, 0.4), 33.0));
        let mut alone = World::new();
        alone.add_mesh_instance(tri_mesh(1.0), xf, None);
        let mut mixed = World::new();
        mixed.add_mesh_instance(tri_mesh(1.0), xf, None);
        let other = mixed.add_mesh_instance(tri_mesh(1.0), Transform::translate(Vec3::new(5.0, 0.0, 0.0)), None);
        assert!(mixed.set_instance_end_transform(other, Transform::translate(Vec3::new(9.0, 0.0, 0.0))));
        assert!(mixed.instances()[0].anim.is_none());
        let mut rng = Rng::new(4);
        for _ in 0..3000 {
            let r = Ray { o: Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, 9.0), d: Vec3::new(0.0, 0.0, -1.0), time: rng.next_f64() };
            let (a, b) = (alone.hit(r, 0.0, 1e30), mixed.hit(r, 0.0, 1e30));
            match (a, b) {
                (Some(a), Some(b)) if b.inst_id == Some(0) => {
                    assert_eq!((a.t.to_bits(), a.p.x.to_bits(), a.p_error.x.to_bits(), a.ng.z.to_bits()), (b.t.to_bits(), b.p.x.to_bits(), b.p_error.x.to_bits(), b.ng.z.to_bits()));
                }
                (Some(_), _) => panic!("静止インスタンスに当たらない"),
                _ => {}
            }
        }
    }


    // ---- ローカル座標（object_space_point）----

    fn local_noise() -> crate::noise::NoiseTexture {
        crate::noise::NoiseTexture {
            pattern: crate::noise::Pattern::Marble, scale: 3.0, octaves: 4, lacunarity: 2.0, gain: 0.5, strength: 4.0,
            color0: Color::new(0.0, 0.0, 0.0), color1: Color::new(1.0, 1.0, 1.0), local: true, offset: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    /// 動く球: 時刻 0 / 0.5 / 1 で、球上の同じ点（中心からの相対位置）のローカル座標は同じで、ノイズの値も同じ
    /// （模様が球に追従する）。ワールド座標で評価すると値が違う。
    #[test]
    fn local_point_follows_a_moving_sphere() {
        let mut world = World::new();
        let idx = world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 0 });
        world.set_sphere_end(idx, Vec3::new(4.0, 1.0, 0.0));
        let n = local_noise();
        let mut local = Vec::new();
        let mut worldv = Vec::new();
        for time in [0.0, 0.5, 1.0] {
            let c = Vec3::new(4.0, 1.0, 0.0) * time;
            let h = world.hit(ray_at(c + Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), time), 1e-9, 1e30).expect("hit");
            let p = world.object_space_point(&h, time);
            assert!((p - Vec3::new(0.0, 0.0, 0.5)).len() < 1e-9, "time {time}: {p:?}");
            local.push(n.factor(p));
            worldv.push(n.factor(h.p));
        }
        assert!(local.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9), "{local:?}");
        assert!((worldv[0] - worldv[1]).abs() > 1e-3, "ワールド評価は泳ぐ: {worldv:?}");
    }

    /// アニメーションするインスタンス: 補間した変換の逆を使う（開き姿勢の逆だと time 0.5 / 1 でずれる）。
    #[test]
    fn local_point_follows_an_animated_instance() {
        let tri = Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0);
        let mut world = World::new();
        let start = Transform::translate(Vec3::new(0.0, 0.0, 0.0));
        let end = Transform::translate(Vec3::new(4.0, 1.0, 0.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 70.0));
        let inst = world.add_mesh_instance(vec![tri], start, None);
        assert!(world.set_instance_end_transform(inst, end));
        let anim = AnimatedTransform::new(start, end).unwrap();
        let local = Vec3::new(0.1, 0.0, 0.0);
        for time in [0.0, 0.3, 0.5, 1.0] {
            let xf = anim.at(time);
            let (pw, nw) = (xf.apply_point(local), xf.apply_normal(Vec3::new(0.0, 0.0, 1.0)));
            let h = world.hit(ray_at(pw + nw * 2.0, -nw, time), 1e-9, 1e30).expect("hit");
            let p = world.object_space_point(&h, time);
            assert!((p - local).len() < 1e-9, "time {time}: {p:?}");
        }
    }

    // ---- 動く球（center_end）----

    fn ray_at(o: Vec3, d: Vec3, time: f64) -> Ray {
        Ray { o, d, time }
    }

    /// time = 0 / 0.5 / 1 で開・中間・閉の位置に当たり、他の位置では外れる。
    #[test]
    fn moving_sphere_hits_at_interpolated_center() {
        let mut world = World::new();
        let idx = world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 0 });
        assert!(world.set_sphere_end(idx, Vec3::new(4.0, 0.0, 0.0)));
        let down = Vec3::new(0.0, 0.0, -1.0);
        let shoot = |x: f64, time: f64| world.hit(ray_at(Vec3::new(x, 0.0, 5.0), down, time), 1e-9, 1e30).is_some();
        for (time, cx) in [(0.0, 0.0), (0.5, 2.0), (1.0, 4.0)] {
            assert!(shoot(cx, time) && shoot(cx + 0.4, time) && shoot(cx - 0.4, time), "time {time}");
            assert!(!shoot(cx + 0.6, time) && !shoot(cx - 0.6, time), "time {time}");
        }
        // 別の時刻の位置には当たらない
        assert!(!shoot(4.0, 0.0) && !shoot(0.0, 1.0) && !shoot(2.0, 0.0));
        // 非有限の終点は拒否して静止のまま
        assert!(!world.set_sphere_end(idx, Vec3::new(f64::NAN, 0.0, 0.0)));
    }

    /// 掃過ボリュームの保守性: 動く球を含むシーンで、箱（TLAS）で棄却する通常経路と、箱の無い総当たり
    /// （`hit_linear` / `occluded_linear`）が、ランダムなレイ × 時刻でビット単位で一致する。
    /// プリミティブ数は TLAS の閾値（20）の前後、移動距離は半径より大きいものを含む。
    #[test]
    fn moving_sphere_sweep_bounds_are_conservative() {
        let mut rng = Rng::new(2024);
        let mut checked = 0;
        for n in [3usize, 19, 20, 21, 60] {
            let mut world = World::new();
            for i in 0..n {
                let c = Vec3::new(rng.next_f64() * 12.0 - 6.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 12.0 - 6.0);
                let r = 0.2 + rng.next_f64() * 0.6;
                let idx = world.add_sphere(Sphere { c, r, mat_id: 0 });
                if i % 2 == 0 {
                    let d = Vec3::new(rng.next_f64() * 8.0 - 4.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 8.0 - 4.0);
                    world.set_sphere_end(idx, c + d);
                }
            }
            world.set_shutter(0.1, 0.9);
            for _ in 0..2500 {
                let o = Vec3::new(rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 8.0 - 4.0, rng.next_f64() * 20.0 - 10.0);
                let target = Vec3::new(rng.next_f64() * 12.0 - 6.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 12.0 - 6.0);
                let time = 0.1 + 0.8 * rng.next_f64();
                let r = ray_at(o, (target - o).norm(), time);
                assert_same_hit(world.hit(r, 1e-9, 1e30), world.hit_linear(r, 1e-9, 1e30), "sphere sweep");
                let tmax = (target - o).len() * 1.2;
                assert_eq!(world.occluded(r, 1e-9, tmax, None), world.occluded_linear(r, 1e-9, tmax, None), "occluded");
                checked += 1;
            }
        }
        assert!(checked >= 10_000);
    }

    /// `center_end` の無い球は、動く球が同じワールドにあってもビット一致（従来と同じ `Sphere::hit`）。
    /// 開と閉が同じ中心なら、どの時刻でも静止とほぼ一致（補間の丸めだけ）。
    #[test]
    fn static_sphere_is_bit_identical_and_degenerate_move_is_static() {
        let s = Sphere { c: Vec3::new(0.3, -0.2, 0.1), r: 0.9, mat_id: 0 };
        let mut world = World::new();
        let a = world.add_sphere(s);
        let b = world.add_sphere(Sphere { c: Vec3::new(5.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        world.set_sphere_end(b, Vec3::new(6.0, 1.0, 0.0));
        let c = world.add_sphere(s);
        world.set_sphere_end(c, s.c); // 開 = 閉
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let o = Vec3::new(rng.next_f64() * 6.0 - 3.0, rng.next_f64() * 6.0 - 3.0, 4.0);
            let r = ray_at(o, (Vec3::new(0.3, -0.2, 0.1) - o + Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, 0.0)).norm(), rng.next_f64());
            assert_same_hit(world.sphere_hit(a, r, 1e-9, 1e30), s.hit(r, 1e-9, 1e30), "static");
            match (world.sphere_hit(c, r, 1e-9, 1e30), s.hit(r, 1e-9, 1e30)) {
                (None, None) => {}
                (Some(x), Some(y)) => assert!((x.t - y.t).abs() < 1e-12 && (x.p - y.p).len() < 1e-12),
                (x, y) => panic!("degenerate move differs: {:?} {:?}", x.map(|h| h.t), y.map(|h| h.t)),
            }
        }
    }
}

/// クロージャの所要時間を測って `(結果, 経過)` を返す（構築時間の内訳表示用）。
fn timed<T>(f: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let v = f();
    (v, t0.elapsed())
}
