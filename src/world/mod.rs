//! ワールド表現、インスタンシング、ライトサンプリング、交差クエリ。
//!
//! - `geometry`: メッシュ・インスタンス・球・SDF の格納、ワールド空間の境界
//! - `hit`: TLAS/線形の `hit`/`occluded`
//! - `lights`: 発光プリミティブとライトサンプリング
//! - `light_bvh`: 光源 BVH のノード構築・選択

mod geometry;
mod hit;
mod light_bvh;
mod lights;
#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use crate::geometry::{Aabb, Sphere};
use crate::math::Vec3;
use crate::sdf::SdfShape;

pub use geometry::{Instance, Mesh};
pub(crate) use geometry::box_world_bounds;
pub use light_bvh::LightSelect;
use light_bvh::LightBvh;
pub use lights::{DeltaLight, DeltaLightHit, Light, LightInfo, LightSample};
use hit::Tlas;
#[cfg(test)]
pub(crate) use tests::test_meshes;

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
    /// SDF（陰関数曲面）。ヒットは `inst_id = None`・`prim_id = spheres.len() + SDF の添字`（TLAS の通し番号と同じ）。光源にはならない
    sdfs: Vec<SdfShape>,
}


impl Default for World {
    fn default() -> Self {
        Self::new()
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
            sdfs: Vec::new(),
        }
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
        for sdf in &self.sdfs {
            b = b.union(sdf.world_bounds());
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


    /// SDF 一覧。ヒットの `prim_id` は `spheres().len() + ここでの添字`。
    pub fn sdfs(&self) -> &[SdfShape] {
        &self.sdfs
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

}
