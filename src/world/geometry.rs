//! ジオメトリの保管と追加: メッシュ・インスタンス・球・SDF の格納、ワールド空間の境界。
//! 交差判定（`World::hit`/`World::occluded`）は `hit.rs`、ライトサンプリングは `lights.rs` を参照。

use std::sync::{Arc, OnceLock};

use crate::bvh::Bvh;
use crate::geometry::{face_forward, Aabb, Hit, Sphere, Triangle, TriangleSource};
use crate::obj_loader::{MeshData, NO_NORMAL, NO_UV};
use crate::math::{gamma, Vec3};
use crate::ray::Ray;
use crate::sdf::SdfShape;
use crate::texture::AlphaMask;
use crate::transform::{AnimatedTransform, Transform};

use super::World;

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
    pub(super) bvh: Bvh,
}

impl Mesh {
    /// BVH の葉が持つ三角形参照の総数（SBVH の重複を含む）。三角形数との比が重複率。
    pub fn bvh_ref_count(&self) -> usize {
        self.bvh.ref_count()
    }
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
        if !self.tri_vn.is_empty()
            && let Some(ns) = self.shading_normal(h.prim_id, h.bary, h.ng)
        {
            h.ns = ns;
        }
        if !self.tri_uv.is_empty()
            && let Some(uv) = self.texture_coords(h.prim_id, h.bary)
        {
            h.uv = uv;
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
    pub(super) fn uv_derivatives(&self, tri_id: usize, time: f64) -> Option<(Vec3, Vec3)> {
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
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(len > 0.0)` also catches a NaN length (degenerate interpolation); `len <= 0.0` would not
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
pub(super) fn sphere_world_bounds(s: &Sphere) -> Aabb {
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
pub(super) fn moving_sphere_bounds(s: &Sphere, end: Vec3, shutter: (f64, f64)) -> Aabb {
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
    let Some(root) = mesh.bvh.root_bounds() else { return Aabb::empty() };
    box_world_bounds(root, xform)
}

/// 物体空間の箱 `root` を `xform` でワールドへ写した、保守的な境界ボックス（8 頂点を変換し、変換の誤差上界ぶん広げる）。
pub(crate) fn box_world_bounds(root: Aabb, xform: &Transform) -> Aabb {
    let mut b = Aabb::empty();
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

impl World {
    /// 球プリミティブを追加し、その `World::spheres` 上のインデックスを返す。
    pub fn add_sphere(&mut self, sphere: Sphere) -> usize {
        self.tlas = OnceLock::new();
        let idx = self.spheres.len();
        self.spheres.push(sphere);
        idx
    }


    /// SDF を追加し、その `World::sdfs` 上のインデックスを返す。
    pub fn add_sdf(&mut self, mut sdf: SdfShape) -> usize {
        self.tlas = OnceLock::new();
        sdf.refresh_bounds(self.shutter); // 動く SDF は現在のシャッター区間の掃過ボリューム（set_shutter との順序を問わない）
        let idx = self.sdfs.len();
        self.sdfs.push(sdf);
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
    pub(super) fn sphere_hit(&self, idx: usize, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
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
        for sdf in &mut self.sdfs {
            if sdf.is_animated() {
                sdf.refresh_bounds((open, close));
            }
        }
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
                // SDF（`prim_id` が球の数以上）: ヒットの時刻の SDF の変換の逆
                None => match hit.prim_id.checked_sub(self.spheres.len()).and_then(|i| self.sdfs.get(i)) {
                    Some(sdf) => sdf.transform_at(time).apply_point_inv(hit.p),
                    None => hit.p,
                },
            },
        }
    }

}

/// クロージャの所要時間を測って `(結果, 経過)` を返す（構築時間の内訳表示用）。
fn timed<T>(f: impl FnOnce() -> T) -> (T, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let v = f();
    (v, t0.elapsed())
}

