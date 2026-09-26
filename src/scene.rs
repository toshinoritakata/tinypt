//! シーン構築。
//!
//! デフォルトシーン: 地面球 + Lambert / Metal / GGX / Glass 各1球 + 発光球

use crate::config::RenderConfig;
use crate::env::EnvMap;
use crate::geometry::Sphere;
use crate::material::Material;
use crate::noise::NoiseTexture;
use crate::normal_map::{MapId, NormalMap};
use crate::texture::Texture;
use crate::math::{Color, Vec3};
use crate::ray::Camera;
use crate::world::World;

/// シーン読み込みの内訳（起動時の表示用）。描画そのものには影響しない。
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    /// OBJ の解析（ファイル読み込み + 三角形化）
    pub obj_parse: std::time::Duration,
    /// メッシュ構築（BVH 構築を含む）
    pub mesh_build: std::time::Duration,
    /// テクスチャ・マップ・環境マップの読み込み
    pub texture_load: std::time::Duration,
}

/// シーンコンテナ（カメラ・ワールド・マテリアル・環境マップ）。
pub struct Scene {
    /// カメラ（レイ生成に使用）
    pub cam: Camera,
    /// ワールド（全ジオメトリとライト）
    pub world: World,
    /// マテリアルリスト（インデックスで参照）
    pub mats: Vec<Material>,
    /// テクスチャ置き場（`Material` が `TexId` で参照する）。
    /// マテリアルを `Copy` のまま保つため、本体はここに集約して添字で引く。
    pub textures: Vec<Texture>,
    /// 法線を摂動するマップ（ハイトマップ／タンジェント空間ノーマルマップ）。色テクスチャとは別に持つ
    /// （sRGB の誤適用を型で防ぐ）。`Material` を `Copy` のまま保つため、材質側ではなく `mat_maps` から引く。
    pub normal_maps: Vec<NormalMap>,
    /// `mat_id` → `normal_maps` の添字。**空か、さもなくば `mats` と同じ長さ**（不変条件。空ならマップ無しで、
    /// 積分器は何も引かない）。
    pub mat_maps: Vec<Option<MapId>>,
    /// 手続き的な 3D ノイズ（`Material` が `NOISE_TEX_FLAG` 付きの `TexId` で参照する。画像の `textures` とは別）
    pub noises: Vec<NoiseTexture>,
    /// 環境マップ（None でデフォルトの空色を使用）
    pub env: Option<EnvMap>,
    /// 一様な参加媒質（None で真空）。M3 までシーンファイルからは読めず、テストとコードから直接与える
    pub medium: Option<crate::medium::Medium>,
    /// 読み込みの内訳（表示用。`World::mesh_build_time` などから埋める）
    pub load_stats: LoadStats,
}

/// レンダリング設定からデフォルトシーンを構築する。
pub fn build_default_scene(config: &RenderConfig) -> Scene {
    let eye    = Vec3::new(0.0, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.5, 0.0);
    let aspect = config.width as f64 / config.height as f64;
    let cam = Camera::look_at_dof(
        eye,
        target,
        Vec3::new(0.0, 1.0, 0.0),
        40.0,
        aspect,
        (eye - target).len(),
        0.0,
    );

    let mut mats: Vec<Material> = Vec::new();
    let mut world = World::new();

    // 地面（大球）
    let id_ground = push(&mut mats, Material::Lambert {
        albedo: Color::from_srgb(0.5, 0.5, 0.5),
        albedo_tex: None,
    });
    world.add_sphere(Sphere { c: Vec3::new(0.0, -1000.0, 0.0), r: 1000.0, mat_id: id_ground });

    // Lambert（左端）
    let id_lambert = push(&mut mats, Material::Lambert {
        albedo: Color::from_srgb(0.8, 0.3, 0.3),
        albedo_tex: None,
    });
    world.add_sphere(Sphere { c: Vec3::new(-1.8, 0.5, 0.0), r: 0.5, mat_id: id_lambert });

    // Metal（左中）
    let id_metal = push(&mut mats, Material::Metal {
        albedo: Color::from_srgb(0.8, 0.8, 0.8),
    });
    world.add_sphere(Sphere { c: Vec3::new(-0.6, 0.5, 0.0), r: 0.5, mat_id: id_metal });

    // GGX（右中）— 粗い金属（ゴールド、α=0.25）
    let id_ggx = push(&mut mats, Material::Ggx {
        albedo: Color::from_srgb(0.95, 0.78, 0.35),
        alpha: 0.25,
    });
    world.add_sphere(Sphere { c: Vec3::new(0.6, 0.5, 0.0), r: 0.5, mat_id: id_ggx });

    // Glass（右端）
    let id_glass = push(&mut mats, Material::Dielectric {
        ior: 1.5,
        absorption: Color::new(0.02, 0.05, 0.02),
    });
    world.add_sphere(Sphere { c: Vec3::new(1.8, 0.5, 0.0), r: 0.5, mat_id: id_glass });

    // 発光球（上方）
    let id_light = push(&mut mats, Material::DiffuseLight {
        emit: Color::new(8.0, 7.0, 5.0),
    });
    world.add_sphere(Sphere { c: Vec3::new(0.0, 3.0, -1.0), r: 0.8, mat_id: id_light });

    world.build_lights(&mats);

    let env = match config.env_map_path.as_ref() {
        Some(path) => match EnvMap::from_hdr(path) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Env map load failed ({}): {}", path, e);
                None
            }
        },
        None => None,
    };

    let load_stats = LoadStats { mesh_build: world.mesh_build_time(), ..Default::default() };
    Scene { cam, world, mats, textures: Vec::new(), normal_maps: Vec::new(), mat_maps: Vec::new(), noises: Vec::new(), env, medium: None, load_stats }
}

fn push(mats: &mut Vec<Material>, m: Material) -> usize {
    let id = mats.len();
    mats.push(m);
    id
}
