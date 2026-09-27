//! OBJ + MTL の読み込み: マテリアル/テクスチャ/マスクの解決。

use std::path::{Path, PathBuf};
use std::sync::Arc;


use crate::constants::normal_map::MTL_BUMP_K;
use crate::material::{Material, TexId};
use crate::math::Color;
use crate::mtl::{parse_mtl, MtlFile, MtlMaterial};
use crate::normal_map::{HeightMap, MapId, NormalMap};
use crate::obj_loader::{load_obj_groups, ObjGroups};
use crate::shader::{Exprs, ValueNode};
use crate::texture::{AlphaMask, Texture, Wrap};
use crate::world::World;

use super::xml::Element;
use super::{timed_obj, timed_texture, warn};
use super::bsdf::{add_value, mul_const};
use super::shape::{apply_end_transform, shape_to_world};

pub(super) fn push_material(mats: &mut Vec<Material>, mat_maps: &mut Vec<Extra>, mat: Material, map: Extra) -> usize {
    debug_assert_eq!(mats.len(), mat_maps.len(), "mats and mat_maps out of sync");
    mats.push(mat);
    mat_maps.push(map);
    mats.len() - 1
}

#[derive(Clone, Default)]
pub(super) struct Extra {
    pub(super) exprs: Exprs,
    pub(super) normals: Vec<MapId>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MapKind {
    Height,
    Tangent,
}

#[derive(Default)]
pub(super) struct MtlState {
    /// 解決済み絶対パス → テクスチャ添字。同じ画像を 2 度読まない（失敗も覚えて再試行しない）
    tex_cache: std::collections::HashMap<PathBuf, Option<TexId>>,
    /// 解決済み絶対パス → アルファマスク（`map_d`）。テクスチャと同じく同じ画像を 2 度読まない
    mask_cache: std::collections::HashMap<PathBuf, Option<Arc<AlphaMask>>>,
    /// 解決済み絶対パス + 種別 + 強度（`f64` のビット）→ 登録済みマップ。同じ画像を 2 度読まない。
    /// 種別をキーに含めるのは、同じ画像をハイト用／ノーマル用の両方に読む場合があるため。強度も含めるのは、
    /// `HeightMap` が強度を持つので `-bm` の違う材質が同じ登録を共有すると強度が入れ替わるため
    map_cache: std::collections::HashMap<(PathBuf, MapKind, u64), Option<MapId>>,
    /// 定数の `d < 1`（マスク無し）を無視する警告を出したか（シーンで 1 回だけ）
    warned_alpha: bool,
    /// 読み込み済みの OBJ メッシュ（`<bsdf>` 指定の経路だけ）。キー = (解決後の絶対パス, `face_normals`)、値 = メッシュ ID。
    /// 同じ OBJ を何度も配置するとき、解析と BVH 構築を 1 回にしてメッシュを共有する。
    /// `xform` と材質はインスタンス側（`mat_override`）なのでキーに入れない。MTL 経路（`parse_obj_with_mtl`）は
    /// 三角形に焼き込む `mat_id` が形状ごとに積む材質で決まるので共有しない。`load_scene_from_str` 1 回ぶんのローカル状態
    pub(super) obj_cache: std::collections::HashMap<(PathBuf, Option<PathBuf>, bool), usize>,
}

pub(super) fn parse_obj_with_mtl(
    el: &Element,
    base_dir: &Path,
    world: &mut World,
    mats: &mut Vec<Material>,
    mat_maps: &mut Vec<Extra>,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
    state: &mut MtlState,
) {
    let filename = match el.string("filename") {
        Some(f) => f,
        None => {
            warn("obj shape without filename; skipped");
            return;
        }
    };
    let resolved = resolve_path(base_dir, filename);
    let ObjGroups { mut mesh, mat_names, mtllibs } = match timed_obj(|| load_obj_groups(resolved.to_string_lossy().as_ref())) {
        Ok(g) => g,
        Err(e) => {
            warn(&format!("failed to load obj '{}': {}; skipped", resolved.display(), e));
            return;
        }
    };
    if el.boolean_or("face_normals", false) {
        mesh = mesh.into_flat();
    }

    // MTL は OBJ からの相対。MTL 内のテクスチャパスは MTL からの相対
    let obj_dir = resolved.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut libs: Vec<(PathBuf, MtlFile)> = Vec::new();
    for lib in &mtllibs {
        let path = obj_dir.join(lib.replace('\\', "/"));
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let f = parse_mtl(&text);
                for d in &f.dup_names {
                    warn(&format!("{}: duplicate newmtl '{}'; keeping the first definition", path.display(), d));
                }
                libs.push((path.parent().map(Path::to_path_buf).unwrap_or_default(), f));
            }
            Err(e) => warn(&format!("failed to read mtl '{}': {}; using default materials", path.display(), e)),
        }
    }
    if mtllibs.is_empty() {
        warn(&format!("obj '{}' has no mtllib and the shape has no bsdf; defaulting to diffuse", resolved.display()));
    }

    let base = mats.len();
    // 材質ごとのアルファ（マスク, d）。`mat_names` と同じ添字
    let mut alphas: Vec<Option<(Arc<AlphaMask>, f32)>> = Vec::with_capacity(mat_names.len());
    let mut maps: Vec<Extra> = Vec::with_capacity(mat_names.len());
    for (mi, name) in mat_names.iter().enumerate() {
        let found = libs.iter().find_map(|(dir, f)| f.get(name).map(|m| (dir, m)));
        let mat = match found {
            Some((dir, m)) => {
                let (mat, alpha, map) = mtl_to_material(m, dir, textures, normal_maps, state);
                alphas.push(alpha);
                maps.push(map);
                mat
            }
            None => {
                alphas.push(None);
                maps.push(Extra::default());
                if !name.is_empty() && !mtllibs.is_empty() {
                    warn(&format!("material '{}' not found in mtl; defaulting to diffuse", name));
                }
                Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }
            }
        };
        push_material(mats, mat_maps, mat, maps[mi].clone());
    }
    for t in mesh.tris.iter_mut() {
        t.mat_id += base;
    }

    let xform = shape_to_world(el);
    // アルファを持つ材質があれば、三角形ごとにマスク添字を振る（メッシュ内で同じ Arc は 1 つに畳む）
    if alphas.iter().any(|a| a.is_some()) {
        let mut masks: Vec<Arc<AlphaMask>> = Vec::new();
        let mut mat_slot: Vec<(u16, f32)> = Vec::with_capacity(alphas.len());
        for a in &alphas {
            mat_slot.push(match a {
                None => (0, 1.0),
                Some((m, d)) => {
                    let i = masks.iter().position(|x| Arc::ptr_eq(x, m)).unwrap_or_else(|| {
                        masks.push(m.clone());
                        masks.len() - 1
                    });
                    (i as u16 + 1, *d)
                }
            });
        }
        let tri_alpha: Vec<(u16, f32)> = mesh.tris.iter().map(|t| mat_slot[t.mat_id - base]).collect();
        let id = world.add_mesh_data_instance_with_alpha(mesh, masks, tri_alpha, xform);
        apply_end_transform(el, world, id, false);
    } else {
        let id = world.add_mesh_data_instance(mesh, xform, None);
        apply_end_transform(el, world, id, false);
    }
}

fn load_mtl_texture(dir: &Path, rel: &str, textures: &mut Vec<Texture>, state: &mut MtlState) -> Option<TexId> {
    let joined = dir.join(rel);
    let key = std::fs::canonicalize(&joined).unwrap_or(joined);
    if let Some(&cached) = state.tex_cache.get(&key) {
        return cached;
    }
    let id = match timed_texture(|| Texture::load(key.to_string_lossy().as_ref(), true, Wrap::Repeat)) {
        Ok(t) => {
            textures.push(t);
            Some((textures.len() - 1) as TexId)
        }
        Err(e) => {
            warn(&format!("failed to load texture '{}': {}; ignored", key.display(), e));
            None
        }
    };
    state.tex_cache.insert(key, id);
    id
}

fn load_mtl_map(
    dir: &Path,
    rel: &str,
    kind: MapKind,
    strength: f64,
    normal_maps: &mut Vec<NormalMap>,
    state: &mut MtlState,
) -> Option<MapId> {
    let joined = dir.join(rel);
    let key = std::fs::canonicalize(&joined).unwrap_or(joined);
    let cache_key = (key.clone(), kind, strength.to_bits());
    if let Some(&cached) = state.map_cache.get(&cache_key) {
        return cached;
    }
    let path = key.to_string_lossy();
    let map = match kind {
        MapKind::Tangent => timed_texture(|| Texture::load(path.as_ref(), false, Wrap::Repeat))
            .map(|tex| NormalMap::Tangent { tex, scale: 1.0 }),
        MapKind::Height => {
            timed_texture(|| HeightMap::load(path.as_ref(), Wrap::Repeat)).map(|map| NormalMap::Height { map, strength })
        }
    };
    let id = match map {
        Ok(m) => {
            normal_maps.push(m);
            Some((normal_maps.len() - 1) as MapId)
        }
        Err(e) => {
            warn(&format!("failed to load normal/bump map '{}': {}; ignored", key.display(), e));
            None
        }
    };
    state.map_cache.insert(cache_key, id);
    id
}

fn load_mtl_mask(dir: &Path, rel: &str, state: &mut MtlState) -> Option<Arc<AlphaMask>> {
    let joined = dir.join(rel);
    let key = std::fs::canonicalize(&joined).unwrap_or(joined);
    if let Some(cached) = state.mask_cache.get(&key) {
        return cached.clone();
    }
    let m = match timed_texture(|| AlphaMask::load(key.to_string_lossy().as_ref(), Wrap::Repeat)) {
        Ok(m) => Some(Arc::new(m)),
        Err(e) => {
            warn(&format!("failed to load alpha mask '{}': {}; ignored", key.display(), e));
            None
        }
    };
    state.mask_cache.insert(key, m.clone());
    m
}

fn mtl_to_material(
    m: &MtlMaterial,
    dir: &Path,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
    state: &mut MtlState,
) -> (Material, Option<(Arc<AlphaMask>, f32)>, Extra) {
    let mut unsupported: Vec<&str> = Vec::new();
    if m.map_ka.is_some() { unsupported.push("map_Ka"); }
    if m.ke.iter().any(|&c| c != 0.0) { unsupported.push("Ke (emission)"); }
    if !unsupported.is_empty() {
        warn(&format!("mtl material '{}': unsupported {} ignored", m.name, unsupported.join(", ")));
    }
    // アルファ: `map_d` のマスク × `d`。マスクの無い定数 `d < 1` は半透明（確率的な透過）が要るので未対応
    let alpha = match m.map_d.as_deref() {
        Some(p) => load_mtl_mask(dir, p, state).map(|mask| (mask, m.d.clamp(0.0, 1.0) as f32)),
        None => {
            if m.d < 1.0 && !state.warned_alpha {
                state.warned_alpha = true;
                warn(&format!(
                    "mtl material '{}': constant d < 1 without map_d ignored (translucency is not supported; further materials are not reported)",
                    m.name
                ));
            }
            None
        }
    };

    // 法線の摂動: `norm`（タンジェント空間ノーマルマップ）があればそれ、無ければ `map_bump`（ハイトマップ）。
    // GGX の分岐でも同じく付くので、材質の種類を決める前に取得する
    if m.norm.is_some() && m.map_bump.is_some() {
        warn(&format!("mtl material '{}': both norm and map_bump given; using norm", m.name));
    }
    let map = if let Some(p) = m.norm.as_deref() {
        load_mtl_map(dir, p, MapKind::Tangent, 1.0, normal_maps, state)
    } else if let Some(p) = m.map_bump.as_deref() {
        load_mtl_map(dir, p, MapKind::Height, m.bm * MTL_BUMP_K, normal_maps, state)
    } else {
        None
    };

    let kd = Color::new(m.kd[0], m.kd[1], m.kd[2]);
    let ks = Color::new(m.ks[0], m.ks[1], m.ks[2]);
    // `map_Kd` があれば拡散テクスチャを優先する（Ks/Ns は使わない）。これが無い経路だと、
    // 明るい Ks + Ns>1 を持つ材質（Sponza の floor/arch/chain/vase_hanging）が map_Kd を読む前に
    // Ggx へ早期 return し、拡散テクスチャがまるごと捨てられて単色に見えてしまう（不具合修正）。
    // 拡散 + 光沢の合成 BSDF（正しい GGX 表現）は Material enum に variant を足す話になるので、
    // ここでは扱わない（優先順位で解く）。
    if let Some(p) = m.map_kd.as_deref() {
        let tex = load_mtl_texture(dir, p, textures, state);
        let albedo = tex.map(|t| {
            let root = add_value(ValueNode::Texture(t));
            mul_const(kd, root)
        });
        return (Material::Lambert { albedo: kd }, alpha, Extra { exprs: Exprs { albedo, ..Exprs::default() }, normals: map.into_iter().collect() });
    }
    if ks.luminance() > 0.05 && m.ns > 1.0 {
        // Blinn-Phong 指数 → GGX の粗さ: alpha = sqrt(2 / (Ns + 2))
        let rough = (2.0 / (m.ns + 2.0)).sqrt().clamp(1e-3, 1.0);
        return (Material::Ggx { albedo: ks, alpha: rough }, alpha, Extra { exprs: Exprs::default(), normals: map.into_iter().collect() });
    }
    (Material::Lambert { albedo: kd }, alpha, Extra { exprs: Exprs::default(), normals: map.into_iter().collect() })
}

pub(super) fn resolve_path(base_dir: &Path, filename: &str) -> PathBuf {
    let p = Path::new(filename);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}
