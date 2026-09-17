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

use crate::bvh::Bvh;
use crate::geometry::{Aabb, Hit, Sphere, Triangle};
use crate::material::Material;
use crate::math::{cdf_search, Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::transform::Transform;


/// 三角形メッシュ（メッシュ単位の BVH 付き）。
///
/// 三角形配列と、それを対象に構築した BVH を対で保持する。BVH は自身の構築元と
/// 異なる三角形配列を渡されると壊れるため、その対応関係はこの型の外に出さない
/// （`bvh` が非公開なのはそのため）。交差判定は [`Mesh::hit`] を通して行う。
pub struct Mesh {
    /// メッシュの三角形リスト
    pub tris: Vec<Triangle>,
    /// メッシュ内の BVH（高速交差判定用）
    bvh: Bvh,
}

impl Mesh {
    /// 三角形リストからメッシュと BVH を構築する。
    pub fn new(tris: Vec<Triangle>) -> Self {
        let bvh = Bvh::build(&tris);
        Self { tris, bvh }
    }

    /// メッシュ内三角形に対するレイ交差判定（オブジェクト空間）。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        self.bvh.hit(&self.tris, r, tmin, tmax)
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
    /// 球インデックス → lights 上の ID（発光体でなければ None）。
    /// `light_pdf` が BSDF サンプリングで命中した発光体を逆引きするために使う。
    sphere_light_id: Vec<Option<usize>>,
    /// (インスタンス ID, メッシュ内三角形 ID) → lights 上の ID。
    tri_light_id: std::collections::HashMap<(usize, usize), usize>,
    /// 自己交差回避オフセットのキャッシュ（シーンの境界ボックスから初回参照時に求める）。
    /// 形状を追加するとリセットされる。
    ray_eps: std::sync::OnceLock<f64>,
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
            sphere_light_id: Vec::new(),
            tri_light_id: std::collections::HashMap::new(),
            ray_eps: std::sync::OnceLock::new(),
        }
    }

    /// 全ジオメトリ（球と、変換後のメッシュインスタンス）のワールド空間の境界ボックス。
    /// インスタンスはメッシュの BVH ルートの AABB の 8 頂点を変換して包む。
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::empty();
        for s in &self.spheres {
            let r = Vec3::new(s.r, s.r, s.r);
            b = b.grow(s.c - r).grow(s.c + r);
        }
        for inst in &self.instances {
            let Some(mesh) = self.meshes.get(inst.mesh_id) else { continue };
            let Some(root) = mesh.bvh.nodes.first() else { continue };
            let (lo, hi) = (root.bbox.min, root.bbox.max);
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

    /// 自己交差回避オフセット（ワールド空間の距離）: `RAY_EPSILON_REL × 境界ボックスの対角線長`。
    ///
    /// 散乱レイ・シャドウレイの原点のずらし量と、それらの交差判定の tmin に使う。シーン全体を
    /// 拡大縮小すると同じ比率で変わるので、結果がスケールに依存しない。形状が無い場合は
    /// `RAY_EPSILON_EMPTY_SCENE`。
    ///
    /// 限界: 1 つのシーンに非常に大きな形状（広い床）と非常に小さな形状が混在すると、オフセットが
    /// 小さな形状に対して大きすぎる（小物体の接地影・細部の遮蔽が失われる）。交差点ごとの誤差上界に
    /// 基づくオフセット（段階 2）で解消する予定。
    pub fn ray_epsilon(&self) -> f64 {
        *self.ray_eps.get_or_init(|| {
            let b = self.bounds();
            let diag = (b.max - b.min).len();
            if diag.is_finite() && diag > 0.0 {
                crate::constants::RAY_EPSILON_REL * diag
            } else {
                crate::constants::RAY_EPSILON_EMPTY_SCENE
            }
        })
    }

    /// 球プリミティブを追加し、その `World::spheres` 上のインデックスを返す。
    pub fn add_sphere(&mut self, sphere: Sphere) -> usize {
        self.ray_eps = std::sync::OnceLock::new();
        let idx = self.spheres.len();
        self.spheres.push(sphere);
        idx
    }

    /// 三角形群からメッシュを構築し、`xform` で配置したインスタンスを追加する。
    /// 追加したインスタンスの ID を返す。
    pub fn add_mesh_instance(&mut self, tris: Vec<Triangle>, xform: Transform, mat_override: Option<usize>) -> usize {
        self.ray_eps = std::sync::OnceLock::new();
        let mesh_id = self.meshes.len();
        self.meshes.push(Mesh::new(tris));
        let inst_id = self.instances.len();
        self.instances.push(Instance { mesh_id, xform, mat_override });
        inst_id
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

    /// ワールド内の全ジオメトリに対するレイ交差判定。
    ///
    /// インスタンスのレイはオブジェクト空間に変換してからメッシュ BVH でテストし、
    /// ヒット結果をワールド空間に戻す。球はワールド空間で直接テストする。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut closest = tmax;
        let mut best: Option<Hit> = None;

        // Instances: ray -> object space
        for (inst_id, inst) in self.instances.iter().enumerate() {
            let mesh = match self.meshes.get(inst.mesh_id) {
                Some(m) => m,
                None => continue,
            };

            let o_obj = inst.xform.apply_point_inv(r.o);
            let d_obj_raw = inst.xform.apply_vec_inv(r.d);
            let d_len = d_obj_raw.len().max(1e-30); // Vec3::norm と同じ式（ビット一致）
            let d_obj = d_obj_raw / d_len; // stabilize
            let r_obj = Ray { o: o_obj, d: d_obj, time: r.time };

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
                let p_world = inst.xform.apply_point(h_obj.p);

                // r.d is normalized in Camera::ray()
                let t_world = (p_world - r.o).dot(r.d);
                if t_world <= tmin {
                    search_from = h_obj.t.next_up();
                    continue;
                }
                if t_world < closest {
                    closest = t_world;
                    let mat_id = inst.mat_override.unwrap_or(h_obj.mat_id);
                    best = Some(Hit {
                        t: t_world,
                        p: p_world,
                        n: inst.xform.apply_normal(h_obj.n),
                        mat_id,
                        prim_id: h_obj.prim_id,
                        inst_id: Some(inst_id),
                        ray_eps: 0.0,
                    });
                }
                break;
            }
        }

        // Spheres
        for (idx, s) in self.spheres.iter().enumerate() {
            if let Some(mut h) = s.hit(r, tmin, closest) {
                h.prim_id = idx;
                closest = h.t;
                best = Some(h);
            }
        }

        if let Some(h) = best.as_mut() {
            h.ray_eps = self.ray_epsilon();
        }
        best
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

        let mut add = |light: Light, emit: Color, area: f64| -> Option<usize> {
            let weight = area * emit.luminance();
            if weight > 0.0 {
                total += weight;
                let id = lights.len();
                lights.push(LightInfo { light, emit, weight });
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
                if let Some(id) = add(light, emit, area) {
                    sphere_light_id[idx] = Some(id);
                }
            }
        }

        // Triangles (per instance, using effective material)
        for (inst_id, inst) in self.instances.iter().enumerate() {
            let mesh = match self.meshes.get(inst.mesh_id) {
                Some(m) => m,
                None => continue,
            };
            for (tri_id, tri) in mesh.tris.iter().enumerate() {
                let mat_id = inst.mat_override.unwrap_or(tri.mat_id);
                if let Some(emit) = mats.get(mat_id).and_then(|m| m.emitted()) {
                    let light = Light::Triangle { mesh_id: inst.mesh_id, tri_id, inst_id };
                    let area = light.area(self, 0.5);
                    if let Some(id) = add(light, emit, area) {
                        tri_light_id.insert((inst_id, tri_id), id);
                    }
                }
            }
        }

        self.lights = lights;
        self.light_cdf = cdf;
        self.light_total = total;
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
        let pdf_select = info.weight / self.light_total;
        pdf_select * info.light.pdf_omega(self, time, from, hit.p, hit.n)
    }

    /// CDF を使ってライトを重点的にサンプリングし、位置・法線・放射輝度・PDF を返す。
    /// PDF は立体角ベース（面積 PDF をジオメトリ変換で立体角に変換）。
    pub fn sample_light(&self, rng: &mut Rng, time: f64, p: Vec3) -> Option<LightSample> {
        if self.light_total <= 0.0 || self.lights.is_empty() {
            return None;
        }
        let r = rng.next_f64() * self.light_total;
        let idx = cdf_search(&self.light_cdf, r).min(self.lights.len().saturating_sub(1));
        let info = self.lights[idx];
        let pdf_select = info.weight / self.light_total;

        // シェープ別のサンプリングと PDF は Light に委譲する。PDF は light_pdf と同じ関数で
        // 求めるので、BSDF サンプリング側の MIS 重みと常に一致する。
        let (pos, normal) = info.light.sample(self, time, p, rng)?;
        let pdf = pdf_select * info.light.pdf_omega(self, time, p, pos, normal);
        if !(pdf > 0.0 && pdf.is_finite()) {
            return None;
        }
        Some(LightSample {
            position: pos,
            normal,
            emit: info.emit,
            pdf,
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
    fn sample(&self, world: &World, time: f64, from: Vec3, rng: &mut Rng) -> Option<(Vec3, Vec3)> {
        match *self {
            Light::Sphere { idx } => {
                let s = world.spheres.get(idx)?;
                let u = rng.next_f64();
                let v = rng.next_f64();
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
                let u = rng.next_f64();
                let v = rng.next_f64();
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
fn tri_world_verts(
    world: &World,
    mesh_id: usize,
    tri_id: usize,
    inst_id: usize,
    time: f64,
) -> Option<(Vec3, Vec3, Vec3)> {
    let mesh = world.meshes.get(mesh_id)?;
    let tri = mesh.tris.get(tri_id)?;
    let inst = world.instances.get(inst_id)?;
    let (v0, v1, v2) = tri.vertices_at(time);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::uniform_sphere_dir;
    use crate::geometry::Sphere;

    fn emissive_sphere_world(c: Vec3, r: f64) -> World {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        world.spheres.push(Sphere { c, r, mat_id: 0 });
        world.build_lights(&mats);
        world
    }

    /// 枝刈りなし（インスタンス BVH に tmax = 1e30）の参照実装。tmin の写像と再探索の規則は World::hit と同じ。
    /// 枝刈り距離の誤りをビット単位で検出するために使う。
    fn hit_without_instance_pruning(world: &World, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut closest = tmax;
        let mut best: Option<Hit> = None;
        for (inst_id, inst) in world.instances.iter().enumerate() {
            let mesh = &world.meshes[inst.mesh_id];
            let o_obj = inst.xform.apply_point_inv(r.o);
            let d_raw = inst.xform.apply_vec_inv(r.d);
            let d_len = d_raw.len().max(1e-30);
            let r_obj = Ray { o: o_obj, d: d_raw / d_len, time: r.time };
            // World::hit と同じ tmin の写像・再探索の規則で、tmax だけ無制限（枝刈りなし）
            let mut search_from = tmin * d_len * (1.0 - 1e-9);
            for _ in 0..=INSTANCE_RETRY_LIMIT {
                let Some(h_obj) = mesh.hit(r_obj, search_from, 1e30) else { break };
                let p_world = inst.xform.apply_point(h_obj.p);
                let t_world = (p_world - r.o).dot(r.d);
                if t_world <= tmin {
                    search_from = h_obj.t.next_up();
                    continue;
                }
                if t_world < closest {
                    closest = t_world;
                    best = Some(Hit {
                        t: t_world,
                        p: p_world,
                        n: inst.xform.apply_normal(h_obj.n),
                        mat_id: inst.mat_override.unwrap_or(h_obj.mat_id),
                        prim_id: h_obj.prim_id,
                        inst_id: Some(inst_id),
                        ray_eps: 0.0,
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
                assert_eq!(bits(a.n), bits(b.n), "{}: n", what);
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

    /// 二次レイの自己再ヒットなし: 球と回転した立方体インスタンスを、大きさ 1e-3〜1e3、原点からの距離 0〜1e3 倍で
    /// 置き、表面の点から `Hit::ray_eps` だけずらした二次レイを出す。外向き（幾何法線側）のレイは凸な自分自身に
    /// 当たらず、内向きのレイ（透過）は入射した面に再ヒットせず（t > 10·ε）物体の反対側へ抜ける。
    #[test]
    fn secondary_rays_do_not_rehit_their_own_surface_at_any_scale() {
        let mut rng = Rng::new(31);
        for &k in &[1e-3, 1.0, 1e3] {
            for &offset in &[0.0, 1e3] {
                let center = Vec3::new(offset * k, 0.5 * offset * k, -0.3 * offset * k);
                let mut world = World::new();
                world.add_sphere(Sphere { c: center, r: k, mat_id: 0 });
                let xf = Transform::translate(center + Vec3::new(4.0 * k, 0.0, 0.0))
                    .compose(Transform::rotate(Vec3::new(0.3, 1.0, 0.2), 33.0))
                    .compose(Transform::scale(Vec3::new(k, 0.7 * k, 1.3 * k)));
                world.add_mesh_instance(unit_box(), xf, None);
                let mut checked = 0;
                for i in 0..2000 {
                    let target = if i % 2 == 0 { center } else { center + Vec3::new(4.0 * k, 0.0, 0.0) };
                    let o = target + uniform_sphere_dir(&mut rng) * (5.0 * k);
                    let Some(h) = world.hit(Ray { o, d: (target - o).norm(), time: 0.0 }, world.ray_epsilon(), 1e30) else { continue };
                    let eps = h.ray_eps;
                    assert_eq!(eps, world.ray_epsilon());
                    // 三角形の法線の向き（巻き順）は保証されないので、物体の中心から外向きにそろえる
                    // （狙った物体の手前に別の物体があることもあるので、実際に当たった物体の中心を使う）
                    let hit_center = if h.inst_id.is_some() { center + Vec3::new(4.0 * k, 0.0, 0.0) } else { center };
                    let n = if h.n.dot(h.p - hit_center) < 0.0 { -h.n.norm() } else { h.n.norm() };
                    for _ in 0..4 {
                        let mut d = uniform_sphere_dir(&mut rng);
                        // 外向き
                        if d.dot(n) < 0.0 { d = -d; }
                        if let Some(h2) = world.hit(Ray { o: h.p + eps * d, d, time: 0.0 }, eps, 1e30) {
                            assert!(!(h2.inst_id == h.inst_id && (h.inst_id.is_some() || h2.prim_id == h.prim_id)),
                                "k={} offset={}: outward ray re-hit its own object at t={} (eps {})", k, offset, h2.t, eps);
                        }
                        // 内向き（浅すぎる角度は除く）
                        let di = -d;
                        if di.dot(-n) > 0.1 {
                            let h2 = world.hit(Ray { o: h.p + eps * di, d: di, time: 0.0 }, eps, 1e30)
                                .unwrap_or_else(|| panic!("k={} offset={}: inward ray escaped its object", k, offset));
                            // 自分の入射面への再ヒットなら t は ε 程度になる（立方体の辺の近くでは隣の面から
                            // 抜ける正当な短い弦もあるので、弦の長さではなく ε との比で判定する）
                            assert!(h2.t > 10.0 * eps, "k={} offset={}: inward ray re-hit the entry surface at t={} (eps {})", k, offset, h2.t, eps);
                        }
                        checked += 1;
                    }
                }
                assert!(checked > 4000, "k={} offset={}: too few checks ({})", k, offset, checked);
            }
        }
    }

    /// **期待される失敗（段階 2 = バッチ 3c の合格条件）**: 大きな床（1 万単位四方）と、その上の非常に薄い
    /// 壁（厚さ 5e-5、高さ 1e-2）が同じシーンにあると、シーンの大きさに比例するオフセット（≈ 1.4e-3）が
    /// 壁の厚さや壁までの距離（5e-4）より大きくなる。床の点から壁越しに光源へ向かうシャドウレイは、原点を
    /// ずらした時点で壁を飛び越えてしまい、遮蔽を検出できない（小物体の接地影・細部の遮蔽の欠落）。
    ///
    /// ここでは正しい振る舞い（遮蔽される）を assert しており、現状は失敗するので `should_panic` にしてある。
    /// 交差点ごとの誤差上界に基づくオフセット（バッチ 3c）で通るようになったら、`should_panic` を外すこと
    /// （外さないとこのテストが失敗し、変更に気付ける）。
    #[test]
    #[should_panic(expected = "thin occluder not detected")]
    fn expected_failure_large_floor_and_thin_occluder() {
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
        let eps = world.ray_epsilon();
        let p = Vec3::new(-5e-4, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.002, 0.0).norm();
        let occluded = world.hit(Ray { o: p + eps * d, d, time: 0.0 }, eps, 10.0).is_some();
        assert!(occluded, "thin occluder not detected (offset {} vs distance to the wall 5e-4)", eps);
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
                    n: ls.normal,
                    mat_id: 0,
                    prim_id: 0,
                    inst_id: None,
                    ray_eps: 0.0,
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
                let hit = Hit { t: 0.0, p: ls.position, n: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, ray_eps: 0.0 };
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
                let hit = Hit { t: 0.0, p: ls.position, n: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, ray_eps: 0.0 };
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
            n: Vec3::new(1.0, 0.0, 0.0),
            mat_id: 0,
            prim_id: 3, // no sphere at this index
            inst_id: None,
            ray_eps: 0.0,
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
                let eps = world.ray_epsilon();
                let shadow = Ray { o: p + eps * wi, d: wi, time: 0.0 };
                let tmax = dist * (1.0 - 1e-9) - 2.0 * eps;
                assert!(world.hit(shadow, eps, tmax).is_none(), "shadow ray must not self-hit the sampled light");
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
}
