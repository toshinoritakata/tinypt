//! 発光プリミティブ（`Light`）とライトサンプリング: CDF 構築（`build_lights`）、面積・立体角サンプリング
//! （`Light::sample`/`Light::pdf_omega`）、デルタ光源（`DeltaLight`）、`LightSample`。
//! 光源 BVH のノード構築・選択は `light_bvh.rs` を参照。

use crate::geometry::{Aabb, Hit, Sphere};
use crate::material::Material;
use crate::math::{cdf_search, gamma, Color, Vec3};
use crate::rng::Rng;

use super::light_bvh::{default_light_select, LightBvh, LightSelect};
use super::World;

impl World {
    /// デルタ光源を足す。面光源用の `light_cdf` / `sample_light` / `light_pdf` には影響しない。
    pub fn add_delta_light(&mut self, light: DeltaLight) {
        self.delta_lights.push(light);
    }


    /// デルタ光源の一覧。積分器の NEE が毎回すべてを順に評価する。
    pub fn delta_lights(&self) -> &[DeltaLight] {
        &self.delta_lights
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
                if mats.get(mat_id).and_then(|m| m.emitted()).is_some()
                    && let Some((a, b, c)) = tri_world_verts(self, inst.mesh_id, tri_id, inst_id, 0.5)
                {
                    bb = bb.grow(a).grow(b).grow(c);
                    any = true;
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
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` also catches a NaN pdf/weight; `x <= 0.0` would not
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
    pub(crate) fn selection_weight(info: &LightInfo, from: Vec3) -> f64 {
        if info.one_sided && info.normal.dot(from - info.plane_p) <= 0.0 {
            return 0.0;
        }
        let d = from - info.center;
        info.weight / d.dot(d).max(info.r2).max(f64::MIN_POSITIVE)
    }


    /// 選択確率の分母（全光源の重みの和）。重み付けなしなら従来の `light_total`。**加算は光源の番号順**
    /// （`sample_light` の累積和と同じ順序・同じ丸め）。
    #[inline]
    pub(crate) fn selection_total(&self, from: Vec3) -> f64 {
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
    pub(crate) fn selection_prob(&self, id: usize, from: Vec3, total: f64) -> f64 {
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


    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(total > 0.0)` also catches a NaN weight total; `total <= 0.0` would not
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
    ///
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
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(dist2 > 0.0)` also catches a NaN squared distance; `dist2 <= 0.0` would not
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
pub(crate) struct SphereCone {
    /// 参照点から球中心への単位ベクトル
    axis: Vec3,
    pub(crate) sin2_max: f64,
    /// 1 − cosθmax = sin²θmax / (1 + √(1 − sin²θmax))（桁落ちのない厳密な形）
    pub(crate) one_minus_cos_max: f64,
}

/// `from` が球の十分に外部なら、`from` から球を見込む円錐を返す。
/// 内部（境界を含む）または表面すれすれ（sin²θmax > [`NEAR_SURFACE_SIN2`]）なら `None`
/// （呼び出し側は表面積サンプリングにフォールバックする）。
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(sin2_max <= NEAR_SURFACE_SIN2)` also catches a NaN ratio (falls back to the near-surface path)
pub(crate) fn sphere_cone(s: &Sphere, from: Vec3) -> Option<SphereCone> {
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
pub(crate) fn orthonormal_basis(w: Vec3) -> (Vec3, Vec3) {
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

pub(crate) fn tri_world_verts(
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
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(d2 > 0.0)` also catches a NaN squared distance (light at the point); `d2 <= 0.0` would not
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

    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(d2 > 0.0)` also catches a NaN squared distance; `d2 <= 0.0` would not
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
