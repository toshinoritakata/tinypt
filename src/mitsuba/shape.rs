//! `shape` 要素のパース: メッシュ/球/SDF 共通の分岐と SDF 木の構築。

use std::path::{Path, PathBuf};


use crate::geometry::Sphere;
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::normal_map::NormalMap;
use crate::obj_loader::{load_obj_mesh, load_obj_mesh_mb, MeshData};
use crate::sdf::{SdfId, SdfNode, SdfNoise, SdfNoisePattern, SdfOp, SdfPrim, SdfPrimEnd, SdfShape, SdfTree};
use crate::texture::Texture;
use crate::transform::Transform;
use crate::world::World;

use super::xml::Element;
use super::{shape_emitter, timed_obj, warn};
use super::bsdf::{parse_bsdf, parse_emitter};
use super::obj_mtl::{parse_obj_with_mtl, push_material, resolve_path, Extra, MtlState};
use super::parametric::{unit_cube_tris, unit_disk_tris, unit_rectangle_tris};
use super::xform::parse_transform;

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_shape(
    el: &Element,
    base_dir: &Path,
    world: &mut World,
    mats: &mut Vec<Material>,
    mat_maps: &mut Vec<Extra>,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
    mtl_state: &mut MtlState,
) {
    if el.children.iter().any(|c| c.tag == "emitter" && matches!(c.typ(), "point" | "directional" | "spot")) {
        warn("point / directional / spot <emitter> inside a <shape> is unsupported (they have no geometry); skipped. Put it directly under <scene>");
    }
    // 形状の内側に閉じ込める媒質（interior media）は未対応。誤解を生まないよう明示して読み飛ばす
    if el.child_tag("medium").is_some() {
        warn("<medium> inside a <shape> is unsupported (interior media); skipped. Put it directly under <scene>");
    }
    // OBJ で `<bsdf>` も `<emitter>` も無く、`use_mtl` が false でなければ MTL から材質を作る。
    // `<bsdf>` 指定があれば従来どおり全体を上書きする（既存シーンの見た目・出力を保つ）。
    if el.typ() == "obj"
        && el.child_tag("bsdf").is_none()
        && shape_emitter(el).is_none()
        && el.string("filename_end").is_none()
        && el.boolean_or("use_mtl", true)
    {
        parse_obj_with_mtl(el, base_dir, world, mats, mat_maps, textures, normal_maps, mtl_state);
        return;
    }
    if el.typ() == "sdf" {
        parse_sdf_shape(el, base_dir, world, mats, mat_maps, textures, normal_maps);
        return;
    }
    // area emitter があれば面光源、なければ bsdf、どちらも無ければ拡散にフォールバック。
    let (mat, map) = if let Some(em) = shape_emitter(el) {
        (parse_emitter(em), Extra::default())
    } else if let Some(b) = el.child_tag("bsdf") {
        parse_bsdf(b, base_dir, textures, normal_maps)
    } else {
        warn(&format!("shape type '{}' without bsdf or emitter; defaulting to diffuse", el.typ()));
        (Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }, Extra::default())
    };
    let mat_id = mats.len();

    // メッシュ系シェープの三角形（正準形オブジェクト空間）。
    // Mitsuba の `face_normals`: true なら頂点法線を使わず面法線だけで陰影を付ける。
    // 既定は false（= OBJ に頂点法線があれば補間する）。パラメトリック形状は元から
    // 頂点法線を持たないので、この指定があっても結果は変わらない。
    let face_normals = el.boolean_or("face_normals", false);
    // OBJ をメッシュキャッシュに登録するときのキー（キャッシュミスで読んだときだけ Some）
    let mut cache_key: Option<(PathBuf, Option<PathBuf>, bool)> = None;
    let is_emitter = shape_emitter(el).is_some();
    let mesh: MeshData = match el.typ() {
        "sphere" => {
            let center = el.point("center").unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            let radius = el.float("radius").unwrap_or(1.0);
            push_material(mats, mat_maps, mat, map.clone());
            if el.children.iter().any(|c| c.tag == "transform" && c.attr("name") == Some("to_world_end")) {
                warn("to_world_end on a sphere is unsupported (spheres move by <point name=\"center_end\"> only); ignored");
            }
            let idx = world.add_sphere(Sphere { c: center, r: radius, mat_id });
            // 独自拡張: シャッター閉じ時点の中心（線形補間）。発光する球は光源サンプリングが時刻を見ないので動かせない
            if el.prop("point", "center_end").is_some() {
                match el.point("center_end") {
                    _ if is_emitter => warn("center_end on an area-light sphere is unsupported (the light would not move); ignored"),
                    Some(end) if world.set_sphere_end(idx, end) => {}
                    _ => warn("sphere center_end is invalid (needs finite x / y / z); the sphere stays static"),
                }
            }
            return;
        }
        // Mitsuba 正準形: 中心原点・法線 +Z・[-1,1]² の正方形
        "rectangle" => unit_rectangle_tris(mat_id),
        // Mitsuba 正準形: [-1,1]³ の立方体
        "cube" => unit_cube_tris(mat_id),
        // Mitsuba 正準形: z=0 平面の半径 1 の円盤
        "disk" => unit_disk_tris(mat_id),
        "obj" => {
            let filename = match el.string("filename") {
                Some(f) => f,
                None => {
                    warn("obj shape without filename; skipped");
                    return;
                }
            };
            let resolved = resolve_path(base_dir, filename);
            // 頂点モーション（独自拡張）: `filename_end` はシャッター閉の OBJ（トポロジー一致が必要）
            let resolved_end = el.string("filename_end").map(|f| resolve_path(base_dir, f));
            // キーは「同じ Mesh になる条件」: 開側のパス・閉側のパス・face_normals。閉側を含め忘れると、
            // 静止版と動く版（または閉側が違うもの）が共有されて静かに壊れる
            let canon = |p: &PathBuf| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
            let key = (canon(&resolved), resolved_end.as_ref().map(canon), face_normals);
            // 同じ OBJ（同じ face_normals）が既に読まれていれば、メッシュを共有してインスタンスだけ足す。
            // 最初のメッシュには最初の形状の mat_id が焼き込まれているので、材質は必ず mat_override で与える
            // （`Hit.mat_id` と面光源の判定は mat_override を見る）
            if let Some(&mesh_id) = mtl_state.obj_cache.get(&key) {
                let xform = shape_to_world(el);
                push_material(mats, mat_maps, mat, map.clone());
                let inst_id = world.add_instance_of(mesh_id, xform, Some(mat_id));
                apply_end_transform(el, world, inst_id, is_emitter);
                return;
            }
            cache_key = Some(key);
            let loaded = match &resolved_end {
                Some(end) => timed_obj(|| load_obj_mesh_mb(resolved.to_string_lossy().as_ref(), end.to_string_lossy().as_ref(), mat_id))
                    .or_else(|e| {
                        // トポロジー不一致・読み込み失敗: 警告して、モーション無しで開側だけを読む
                        warn(&format!("vertex motion '{}' -> '{}' unusable: {}; loading without vertex motion", resolved.display(), end.display(), e));
                        timed_obj(|| load_obj_mesh(resolved.to_string_lossy().as_ref(), mat_id))
                    }),
                None => timed_obj(|| load_obj_mesh(resolved.to_string_lossy().as_ref(), mat_id)),
            };
            match loaded {
                Ok(m) => if face_normals { m.into_flat() } else { m },
                Err(e) => {
                    warn(&format!("failed to load obj '{}': {}; skipped", resolved.display(), e));
                    return;
                }
            }
        }
        other => {
            warn(&format!("unsupported shape type '{}', skipped", other));
            return;
        }
    };

    // to_world 変換（なければ恒等）でメッシュをインスタンス配置する。
    let xform = shape_to_world(el);
    push_material(mats, mat_maps, mat, map.clone());
    let inst_id = world.add_mesh_data_instance(mesh, xform, None);
    if let Some(key) = cache_key {
        mtl_state.obj_cache.insert(key, world.instance_mesh_id(inst_id));
    }
    apply_end_transform(el, world, inst_id, is_emitter);
}

fn parse_sdf_shape(
    el: &Element,
    base_dir: &Path,
    world: &mut World,
    mats: &mut Vec<Material>,
    mat_maps: &mut Vec<Extra>,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
) {
    let roots: Vec<&Element> = el.children.iter().filter(|c| c.tag == "sdf").collect();
    if roots.len() != 1 {
        warn(&format!("sdf shape needs exactly one root <sdf> (found {}); skipped", roots.len()));
        return;
    }
    let mut tree = SdfTree::new();
    if let Err(e) = parse_sdf_node(roots[0], &mut tree) {
        warn(&format!("sdf shape: {}; skipped", e));
        return;
    }
    let (mat, map) = match (shape_emitter(el), el.child_tag("bsdf")) {
        (em, Some(b)) => {
            if em.is_some() {
                warn("area <emitter> on an sdf shape is unsupported (an sdf is never a light source); emission ignored");
            }
            parse_bsdf(b, base_dir, textures, normal_maps)
        }
        (em, None) => {
            if em.is_some() {
                warn("area <emitter> on an sdf shape is unsupported (an sdf is never a light source); emission ignored");
            } else {
                warn("shape type 'sdf' without bsdf; defaulting to diffuse");
            }
            (Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }, Extra::default())
        }
    };
    let mat_id = mats.len();
    let Some(mut shape) = SdfShape::new(tree, shape_to_world(el), mat_id) else {
        warn("sdf shape has empty or non-finite bounds; skipped");
        return;
    };
    // シャッター閉の変換（独自拡張。インスタンスと同じ `AnimatedTransform`）。不正なら警告して静止のまま
    let end_el = el.children.iter().find(|c| c.tag == "transform" && c.attr("name") == Some("to_world_end"));
    if end_el.is_some_and(|e| !shape.set_end_transform(parse_transform(e))) {
        warn("to_world / to_world_end is singular, mirrored (negative determinant) or not decomposable; the shape stays static");
    }
    push_material(mats, mat_maps, mat, map);
    world.add_sdf(shape);
}

fn parse_sdf_node(el: &Element, tree: &mut SdfTree) -> Result<SdfId, String> {
    let typ = el.typ();
    let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let float = |name: &str, default: f64| el.float(name).unwrap_or(default);
    let center = el.point("center").unwrap_or(Vec3::new(0.0, 0.0, 0.0));
    if !finite(center) {
        return Err(format!("<sdf type=\"{}\"> center is not finite", typ));
    }
    let pos = |name: &str, v: f64| -> Result<f64, String> {
        if v.is_finite() && v > 0.0 { Ok(v) } else { Err(format!("<sdf type=\"{}\"> {} must be positive and finite (got {})", typ, name, v)) }
    };
    let nonneg = |name: &str, v: f64| -> Result<f64, String> {
        if v.is_finite() && v >= 0.0 { Ok(v) } else { Err(format!("<sdf type=\"{}\"> {} must be >= 0 and finite (got {})", typ, name, v)) }
    };
    let prim = match typ {
        "sphere" => SdfPrim::Sphere { center, radius: pos("radius", float("radius", 1.0))? },
        "box" => {
            let half = el.vector("half").or_else(|| el.point("half")).unwrap_or(Vec3::new(1.0, 1.0, 1.0));
            if !finite(half) || half.x <= 0.0 || half.y <= 0.0 || half.z <= 0.0 {
                return Err("<sdf type=\"box\"> half must be positive and finite".to_string());
            }
            SdfPrim::Box { center, half, round: nonneg("round", float("round", 0.0))? }
        }
        "torus" => SdfPrim::Torus { center, major: pos("major", float("major", 1.0))?, minor: pos("minor", float("minor", 0.25))? },
        "cylinder" => {
            let radius = pos("radius", float("radius", 1.0))?;
            let half_height = pos("half_height", float("half_height", 1.0))?;
            let round = nonneg("round", float("round", 0.0))?;
            if round > radius.min(half_height) {
                return Err("<sdf type=\"cylinder\"> round must not exceed radius or half_height".to_string());
            }
            SdfPrim::Cylinder { center, radius, half_height, round }
        }
        "capsule" => {
            let a = el.point("a").unwrap_or(Vec3::new(0.0, -0.5, 0.0));
            let b = el.point("b").unwrap_or(Vec3::new(0.0, 0.5, 0.0));
            if !finite(a) || !finite(b) {
                return Err("<sdf type=\"capsule\"> a / b must be finite".to_string());
            }
            SdfPrim::Capsule { a, b, radius: pos("radius", float("radius", 0.5))? }
        }
        "union" | "intersection" | "difference" | "smooth_union" | "smooth_intersection" | "smooth_difference" => {
            let kids: Vec<&Element> = el.children.iter().filter(|c| c.tag == "sdf").collect();
            if kids.len() < 2 {
                return Err(format!("<sdf type=\"{}\"> needs at least 2 child <sdf> (found {})", typ, kids.len()));
            }
            for name in ["center_end", "a_end", "b_end"] {
                if el.prop("point", name).is_some() {
                    warn(&format!("<sdf type=\"{typ}\"> is an operator and does not take {name} (put it on a primitive); ignored"));
                }
            }
            let smooth = typ.starts_with("smooth_");
            let k = if smooth { nonneg("k", float("k", 0.2))? } else { 0.0 };
            let mut ids = Vec::with_capacity(kids.len());
            for kid in kids {
                ids.push(parse_sdf_node(kid, tree)?);
            }
            let mut acc = ids[0];
            for &b in &ids[1..] {
                let op = match typ {
                    "union" => SdfOp::Union(acc, b),
                    "intersection" => SdfOp::Intersect(acc, b),
                    "difference" => SdfOp::Subtract(acc, b),
                    "smooth_union" => SdfOp::SmoothUnion(acc, b, k),
                    "smooth_intersection" => SdfOp::SmoothIntersect(acc, b, k),
                    _ => SdfOp::SmoothSubtract(acc, b, k),
                };
                acc = tree.push(SdfNode::Op(op));
            }
            return Ok(acc);
        }
        "displace" => {
            // 子の面をノイズ場でずらす（独自拡張）。`f = f_child + amplitude · n((p + offset) · scale)`。名前と既定値は
            // `<texture type="noise">` に揃える（pattern / scale / octaves / lacunarity / gain）
            let kids: Vec<&Element> = el.children.iter().filter(|c| c.tag == "sdf").collect();
            if kids.len() != 1 {
                return Err(format!("<sdf type=\"displace\"> needs exactly 1 child <sdf> (found {})", kids.len()));
            }
            let pattern = match el.string("pattern") {
                None | Some("fbm") => SdfNoisePattern::Fbm,
                Some("perlin") => SdfNoisePattern::Perlin,
                Some("turbulence") => SdfNoisePattern::Turbulence,
                Some(s) => {
                    warn(&format!("unknown displace pattern '{s}' (perlin | fbm | turbulence); using fbm"));
                    SdfNoisePattern::Fbm
                }
            };
            let amplitude = float("amplitude", 0.05);
            if !amplitude.is_finite() {
                return Err("<sdf type=\"displace\"> amplitude must be finite".to_string());
            }
            let offset = el.vector("offset").or_else(|| el.point("offset")).unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            if !finite(offset) {
                return Err("<sdf type=\"displace\"> offset must be finite".to_string());
            }
            let raw = (float("scale", 1.0), el.int("octaves").map_or(4, |o| o.clamp(0, 1000) as u32), float("lacunarity", 2.0), float("gain", 0.5));
            let fixed = crate::noise::sanitize_octave_params(raw.0, raw.1, raw.2, raw.3);
            if fixed != raw {
                warn("displace parameter out of range (scale must be in (0, 1e6], octaves 1..10, lacunarity 1..8, gain 0..1); clamped");
            }
            let (scale, octaves, lacunarity, gain) = fixed;
            let child = parse_sdf_node(kids[0], tree)?;
            let noise = SdfNoise { pattern, amplitude, scale, octaves, lacunarity, gain, offset };
            return Ok(tree.push(SdfNode::Op(SdfOp::Displace(child, noise))));
        }
        other => return Err(format!("unsupported <sdf type=\"{}\">", other)),
    };
    let id = tree.push(SdfNode::Prim(prim));
    // 個別のシャッター閉の位置（独自拡張）: 球・箱・トーラス・円柱は `center_end`、カプセルは `a_end` / `b_end`。
    // 非有限なら警告して静止のまま。時刻 0 = 開の位置、1 = 閉の位置で線形補間する
    let end_point = |name: &str| -> Option<Option<Vec3>> {
        el.prop("point", name)?;
        match el.point(name) {
            Some(v) if finite(v) => Some(Some(v)),
            _ => {
                warn(&format!("<sdf type=\"{typ}\"> {name} is invalid (needs finite x / y / z); the primitive stays static in that respect"));
                Some(None)
            }
        }
    };
    let is_capsule = typ == "capsule";
    for (name, applies) in [("center_end", !is_capsule), ("a_end", is_capsule), ("b_end", is_capsule)] {
        if !applies && el.prop("point", name).is_some() {
            warn(&format!("<sdf type=\"{typ}\"> does not take {name}; ignored"));
        }
    }
    let end = if is_capsule {
        let (a, b) = (end_point("a_end").flatten(), end_point("b_end").flatten());
        (a.is_some() || b.is_some()).then_some(SdfPrimEnd::Capsule { a, b })
    } else {
        end_point("center_end").flatten().map(SdfPrimEnd::Center)
    };
    if let Some(end) = end {
        tree.set_prim_end(id, end);
    }
    Ok(id)
}

pub(super) fn shape_to_world(el: &Element) -> Transform {
    el.children
        .iter()
        .find(|c| c.tag == "transform" && matches!(c.attr("name"), None | Some("to_world")))
        .map(parse_transform)
        .unwrap_or_else(Transform::identity)
}

pub(super) fn apply_end_transform(el: &Element, world: &mut World, inst_id: usize, is_emitter: bool) {
    let Some(end_el) = el.children.iter().find(|c| c.tag == "transform" && c.attr("name") == Some("to_world_end")) else { return };
    if is_emitter {
        warn("to_world_end on an area-light shape is unsupported (the light would not move); ignored");
        return;
    }
    if !world.set_instance_end_transform(inst_id, parse_transform(end_el)) {
        warn("to_world / to_world_end is singular, mirrored (negative determinant) or not decomposable; the shape stays static");
    }
}
