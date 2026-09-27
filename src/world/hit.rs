//! 交差判定: TLAS 経由と線形総当たりの両方の `hit`/`occluded`、インスタンス・球・SDF の交差判定の本体。

use std::cell::Cell;

use crate::bvh::Bvh;
use crate::geometry::{face_forward, Aabb, Hit};
use crate::math::Vec3;
use crate::ray::Ray;
use crate::transform::Transform;

use super::geometry::{moving_sphere_bounds, sphere_world_bounds};
use super::World;

/// インスタンスの交差判定で、ワールド空間の tmin 判定に却下された面の先を探し直す最大回数。
pub(super) const INSTANCE_RETRY_LIMIT: usize = 4;


/// トップレベル BVH。葉の添字 `k < n_inst` はインスタンス `k`、`n_inst <= k < n_inst + n_sph` は球 `k - n_inst`、
/// それ以降は SDF `k - n_inst - n_sph`
/// （線形総当たりの走査順「インスタンス → 球 → SDF」と同じ通し番号で、同値の t のタイブレークにも使う。
/// SDF の `Hit::prim_id` は `n_sph + SDF の添字` で、この通し番号から `n_inst` を引いたものに等しい）。
pub(super) struct Tlas {
    bvh: Bvh,
    n_inst: usize,
    n_sph: usize,
    n_sdf: usize,
}

/// TLAS の葉の通し番号 `k` が指す先（`Tlas::category` が判定する）。`hit` と `occluded` の TLAS 経由の
/// 経路が共有する分類で、値そのものは `Tlas` の 3 つの数から即座に決まる添字なので、この分岐自体に
/// 実行コストは無い（各腕はそのまま従来と同じ 1 回の添字計算）。
enum TlasLeaf {
    Instance(usize),
    Sphere(usize),
    Sdf(usize),
}

impl Tlas {
    /// 通し番号 `k` がどのプリミティブ配列の何番目かを返す（`k < n_inst` はインスタンス、
    /// 次の `n_sph` 個は球、残りは SDF。[`Tlas`] のドキュメント参照）。
    #[inline(always)]
    fn category(&self, k: usize) -> TlasLeaf {
        if k < self.n_inst {
            TlasLeaf::Instance(k)
        } else if k >= self.n_inst + self.n_sph {
            TlasLeaf::Sdf(k - self.n_inst - self.n_sph)
        } else {
            TlasLeaf::Sphere(k - self.n_inst)
        }
    }
}

/// この数以上のプリミティブ（インスタンス + 球）があるときだけ TLAS を使う。少数では総当たりのほうが速い
/// （決め方は tlas_report.md 参照）。
pub(super) const TLAS_MIN_PRIMS: usize = 20;


impl World {
    /// トップレベル BVH（あれば）。最初の呼び出しで構築する。プリミティブが少ない、または構築後に
    /// ジオメトリが増えて古くなっているときは `None`（呼び出し側は線形に総当たりする）。
    pub(super) fn tlas(&self) -> Option<&Tlas> {
        let n_prims = self.instances.len() + self.spheres.len() + self.sdfs.len();
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
                bounds.extend(self.sdfs.iter().map(|s| s.world_bounds()));
                Some(Tlas { bvh: Bvh::build_from_bounds(&bounds, 1), n_inst: self.instances.len(), n_sph: self.spheres.len(), n_sdf: self.sdfs.len() })
            })
            .as_ref()?;
        (t.n_inst == self.instances.len() && t.n_sph == self.spheres.len() && t.n_sdf == self.sdfs.len()).then_some(t)
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
                match tlas.category(k) {
                    TlasLeaf::Instance(inst_id) => {
                        if let Some(h) = self.hit_instance(inst_id, r, inv_d, tmin, c, k < best_k) {
                            closest.set(h.t);
                            best = Some(h);
                            best_k = k;
                        }
                    }
                    TlasLeaf::Sdf(sdf_idx) => {
                        let hi = if k < best_k && best.is_some() { c.next_up() } else { c };
                        let cand = self.sdfs[sdf_idx].hit(r, tmin, hi).filter(|h| h.t < c || (h.t == c && k < best_k));
                        if let Some(mut h) = cand {
                            h.prim_id = tlas.n_sph + sdf_idx;
                            closest.set(h.t);
                            best = Some(h);
                            best_k = k;
                        }
                    }
                    TlasLeaf::Sphere(idx) => {
                        // 同値タイで勝てる（添字が小さい）ときだけ、区間の上端をわずかに広げて t == closest の交差を拾う
                        let hi = if k < best_k && best.is_some() { c.next_up() } else { c };
                        if let Some(mut h) = self.sphere_hit(idx, r, tmin, hi)
                            && (h.t < c || (h.t == c && k < best_k))
                        {
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
    pub(super) fn hit_linear(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
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

        // SDF（球の後。`prim_id` は球の数からの通し番号）
        for (i, sdf) in self.sdfs.iter().enumerate() {
            if let Some(mut h) = sdf.hit(r, tmin, closest) {
                h.prim_id = self.spheres.len() + i;
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
        let mesh = self.meshes.get(inst.mesh_id)?;
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
                found = match tlas.category(k) {
                    TlasLeaf::Instance(inst_id) => self.occluded_instance(inst_id, r, inv_d, tmin, tmax, skip),
                    TlasLeaf::Sdf(sdf_idx) => self.sdfs[sdf_idx].hit(r, tmin, tmax).is_some(),
                    TlasLeaf::Sphere(idx) => skip != Some((None, idx)) && self.sphere_hit(idx, r, tmin, tmax).is_some(),
                };
                found
            });
            return found;
        }
        self.occluded_linear(r, tmin, tmax, skip)
    }


    /// 線形総当たり版の [`World::occluded`]。
    pub(super) fn occluded_linear(&self, r: Ray, tmin: f64, tmax: f64, skip: Option<(Option<usize>, usize)>) -> bool {
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

        // SDF は光源にならないので skip の対象外
        self.sdfs.iter().any(|s| s.hit(r, tmin, tmax).is_some())
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

}
