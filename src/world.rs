//! ワールド表現、インスタンシング、ライトサンプリング、交差クエリ。
//!
//! `World` はシーン内の全ジオメトリ（球・メッシュインスタンス）と
//! ライトサンプリング用の CDF を保持する。
//!
//! ## ライトサンプリング
//! 発光マテリアルを持つプリミティブから CDF を構築し、
//! 面積 × 輝度に比例した確率でライトを選択する。

use crate::bvh::Bvh;
use crate::geometry::{Hit, Sphere, Triangle};
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::transform::Transform;


/// 三角形メッシュ（メッシュ単位の BVH 付き）。
pub struct Mesh {
    /// メッシュの三角形リスト
    pub tris: Vec<Triangle>,
    /// メッシュ内の BVH（高速交差判定用）
    pub bvh: Bvh,
}

impl Mesh {
    /// 三角形リストからメッシュと BVH を構築する。
    pub fn new(tris: Vec<Triangle>) -> Self {
        let bvh = Bvh::build(&tris);
        Self { tris, bvh }
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

/// ジオメトリ・インスタンス・ライトの集合体。
pub struct World {
    /// シーン内の球プリミティブ
    pub spheres: Vec<Sphere>,
    /// メッシュ（三角形群 + BVH）
    pub meshes: Vec<Mesh>,
    /// メッシュのインスタンス（トランスフォーム付き）
    pub instances: Vec<Instance>,
    /// 発光プリミティブのリスト
    pub lights: Vec<LightInfo>,
    /// ライト選択用の累積分布関数（CDF）
    pub light_cdf: Vec<f64>,
    /// CDF の総重み
    pub light_total: f64,
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
        }
    }

    /// ワールド内の全ジオメトリに対するレイ交差判定。
    ///
    /// インスタンスのレイはオブジェクト空間に変換してからメッシュ BVH でテストし、
    /// ヒット結果をワールド空間に戻す。球はワールド空間で直接テストする。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut closest = tmax;
        let mut best: Option<Hit> = None;

        // Instances: ray -> object space
        for inst in &self.instances {
            let mesh = match self.meshes.get(inst.mesh_id) {
                Some(m) => m,
                None => continue,
            };

            let o_obj = inst.xform.apply_point_inv(r.o);
            let mut d_obj = inst.xform.apply_vec_inv(r.d);
            d_obj = d_obj.norm(); // stabilize
            let r_obj = Ray { o: o_obj, d: d_obj, time: r.time };

            // Object-space BVH
            if let Some(h_obj) = mesh.bvh.hit(&mesh.tris, r_obj, tmin, 1e30) {
                let p_world = inst.xform.apply_point(h_obj.p);
                let n_world = inst.xform.apply_normal(h_obj.n);

                // r.d is normalized in Camera::ray()
                let t_world = (p_world - r.o).dot(r.d);
                if t_world > tmin && t_world < closest {
                    closest = t_world;
                    let mat_id = inst.mat_override.unwrap_or(h_obj.mat_id);
                    best = Some(Hit { t: t_world, p: p_world, n: n_world, mat_id });
                }
            }
        }

        // Spheres
        for s in &self.spheres {
            if let Some(h) = s.hit(r, tmin, closest) {
                closest = h.t;
                best = Some(h);
            }
        }

        best
    }

    /// 発光マテリアルからライトサンプリング構造（CDF）を構築する。
    /// 各ライトの重み = 表面積 × 放射輝度の輝度値。
    ///
    /// シェープ別の面積計算は [`Light::area`] に委譲する（[`Self::sample_light`] と共有）。
    pub fn build_lights(&mut self, mats: &[Material]) {
        let mut lights: Vec<LightInfo> = Vec::new();
        let mut cdf: Vec<f64> = Vec::new();
        let mut total = 0.0;

        let mut add = |light: Light, emit: Color, area: f64| {
            let weight = area * emit.luminance();
            if weight > 0.0 {
                total += weight;
                lights.push(LightInfo { light, emit, weight });
                cdf.push(total);
            }
        };

        // Spheres
        for (idx, s) in self.spheres.iter().enumerate() {
            if let Some(emit) = material_emit(mats.get(s.mat_id)) {
                let light = Light::Sphere { idx };
                add(light, emit, light.area(self, 0.5));
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
                if let Some(emit) = material_emit(mats.get(mat_id)) {
                    let light = Light::Triangle { mesh_id: inst.mesh_id, tri_id, inst_id };
                    add(light, emit, light.area(self, 0.5));
                }
            }
        }

        self.lights = lights;
        self.light_cdf = cdf;
        self.light_total = total;
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

        // シェープ別の表面サンプリングは Light に委譲（build_lights と共有）
        let (pos, normal, area) = info.light.sample_surface(self, time, rng)?;

        if area <= 0.0 {
            return None;
        }
        let to_light = pos - p;
        let dist2 = to_light.dot(to_light);
        if dist2 <= 1e-12 {
            return None;
        }
        let dist = dist2.sqrt();
        let wi = to_light / dist;
        let cos_light = normal.dot(-wi).max(0.0);
        if cos_light <= 0.0 {
            return None;
        }
        let pdf_area = 1.0 / area;
        let pdf_omega = pdf_area * dist2 / cos_light;
        let pdf = pdf_select * pdf_omega;
        if pdf <= 0.0 {
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

    /// 発光面を一様サンプリングし、`(位置, 法線, 面積)` を返す。
    /// ジオメトリが見つからない場合は `None`。
    fn sample_surface(&self, world: &World, time: f64, rng: &mut Rng) -> Option<(Vec3, Vec3, f64)> {
        match *self {
            Light::Sphere { idx } => {
                let s = world.spheres.get(idx)?;
                let u = rng.next_f64();
                let v = rng.next_f64();
                let z = 1.0 - 2.0 * u;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f64::consts::TAU * v;
                let dir = Vec3::new(r * phi.cos(), z, r * phi.sin());
                let pos = s.c + dir * s.r;
                let area = 4.0 * std::f64::consts::PI * s.r * s.r;
                Some((pos, dir, area))
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
                let area = 0.5 * n.len();
                let normal = if area > 0.0 { n / (2.0 * area) } else { Vec3::new(0.0, 1.0, 0.0) };
                Some((pos, normal, area))
            }
        }
    }
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

/// マテリアルが発光体なら放射輝度を返す。
fn material_emit(mat: Option<&Material>) -> Option<Color> {
    match mat {
        Some(Material::DiffuseLight { emit }) => Some(*emit),
        _ => None,
    }
}

/// CDF 内で値 `x` 以下の最大インデックスを二分探索で返す。
fn cdf_search(cdf: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = cdf.len().saturating_sub(1);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    fn emissive_sphere_world(c: Vec3, r: f64) -> World {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        world.spheres.push(Sphere { c, r, mat_id: 0 });
        world.build_lights(&mats);
        world
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
}
