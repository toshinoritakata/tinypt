//! シーン構築。
//!
//! デフォルトシーン: 地面球 + Lambert / Metal / GGX / Glass 各1球 + 発光球

use crate::config::RenderConfig;
use crate::env::EnvMap;
use crate::geometry::Sphere;
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::ray::Camera;
use crate::world::World;

/// シーンコンテナ（カメラ・ワールド・マテリアル・環境マップ）。
pub struct Scene {
    /// カメラ（レイ生成に使用）
    pub cam: Camera,
    /// ワールド（全ジオメトリとライト）
    pub world: World,
    /// マテリアルリスト（インデックスで参照）
    pub mats: Vec<Material>,
    /// 環境マップ（None でデフォルトの空色を使用）
    pub env: Option<EnvMap>,
}

/// レンダリング設定からデフォルトシーンを構築する。
pub fn build_scene(config: &RenderConfig) -> Scene {
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
    });
    world.spheres.push(Sphere { c: Vec3::new(0.0, -1000.0, 0.0), r: 1000.0, mat_id: id_ground });

    // Lambert（左）
    let id_lambert = push(&mut mats, Material::Lambert {
        albedo: Color::from_srgb(0.8, 0.3, 0.3),
    });
    world.spheres.push(Sphere { c: Vec3::new(-1.2, 0.5, 0.0), r: 0.5, mat_id: id_lambert });

    // Metal（中央）
    let id_metal = push(&mut mats, Material::Metal {
        albedo: Color::from_srgb(0.8, 0.8, 0.8),
    });
    world.spheres.push(Sphere { c: Vec3::new(0.0, 0.5, 0.0), r: 0.5, mat_id: id_metal });

    // Glass（右）
    let id_glass = push(&mut mats, Material::Dielectric {
        ior: 1.5,
        absorption: Color::new(0.02, 0.05, 0.02),
    });
    world.spheres.push(Sphere { c: Vec3::new(1.2, 0.5, 0.0), r: 0.5, mat_id: id_glass });

    // 発光球（上方）
    let id_light = push(&mut mats, Material::DiffuseLight {
        emit: Color::new(8.0, 7.0, 5.0),
    });
    world.spheres.push(Sphere { c: Vec3::new(0.0, 3.0, -1.0), r: 0.8, mat_id: id_light });

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

    Scene { cam, world, mats, env }
}

fn push(mats: &mut Vec<Material>, m: Material) -> usize {
    let id = mats.len();
    mats.push(m);
    id
}
