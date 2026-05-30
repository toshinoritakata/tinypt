//! Mitsuba XML（サブセット）シーンローダー。
//!
//! [Mitsuba レンダラー](https://www.mitsuba-renderer.org/) の XML シーン記述の
//! サブセットを読み込み、[`Scene`] を構築する。採用理由とマッピング方針は
//! `docs/adr/0002-mitsuba-xml-scene-format.md` を参照。
//!
//! ## 対応要素（M1）
//! - `sensor type="perspective"`: `fov` / `to_world`(`lookat`) / `aperture_radius` / `focus_distance`
//! - `shape type="sphere"`: `center` / `radius` + 子 `bsdf` または `emitter`
//! - `bsdf`: `diffuse` / `conductor` / `roughconductor`(ggx) / `dielectric` / `twosided`(unwrap)
//! - `emitter type="area"`: `radiance`
//!
//! ## 方針
//! - 色: `<rgb>` はリニア、`<srgb>` は sRGB（ガンマ展開）。
//! - 未対応の要素・型は警告してスキップ／フォールバック（寛容）。必須フィールド欠落のみ既定値。
//! - スペクトルは扱わず RGB トリプルとして読む。

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use crate::config::RenderConfig;
use crate::geometry::Sphere;
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::obj_loader::load_obj_triangles;
use crate::ray::Camera;
use crate::scene::Scene;
use crate::transform::Transform;
use crate::world::{Instance, Mesh, World};

/// パース済み XML 要素（タグ名・属性・子要素）。
struct Element {
    tag: String,
    attrs: HashMap<String, String>,
    children: Vec<Element>,
}

impl Element {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).map(|s| s.as_str())
    }

    /// `type` 属性（Mitsuba のプラグイン種別）。
    fn typ(&self) -> &str {
        self.attr("type").unwrap_or("")
    }

    /// `name` 属性が一致する子プロパティ要素を返す。
    fn prop(&self, tag: &str, name: &str) -> Option<&Element> {
        self.children
            .iter()
            .find(|c| c.tag == tag && c.attr("name") == Some(name))
    }

    /// 最初の指定タグ子要素を返す。
    fn child_tag(&self, tag: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.tag == tag)
    }

    fn float(&self, name: &str) -> Option<f64> {
        self.prop("float", name)?.attr("value")?.trim().parse().ok()
    }

    fn string(&self, name: &str) -> Option<&str> {
        self.prop("string", name)?.attr("value")
    }

    /// `point` プロパティ（`x`/`y`/`z` 属性または `value="x,y,z"`）。
    fn point(&self, name: &str) -> Option<Vec3> {
        let e = self.prop("point", name)?;
        if let (Some(x), Some(y), Some(z)) = (e.attr("x"), e.attr("y"), e.attr("z")) {
            Some(Vec3::new(parse_f64(x)?, parse_f64(y)?, parse_f64(z)?))
        } else {
            parse_vec3(e.attr("value")?)
        }
    }

    /// `rgb`（リニア）または `srgb`（ガンマ展開）プロパティを `Color` として読む。
    fn color(&self, name: &str) -> Option<Color> {
        if let Some(e) = self.prop("rgb", name) {
            let v = parse_vec3(e.attr("value")?)?;
            Some(Color::new(v.x, v.y, v.z))
        } else if let Some(e) = self.prop("srgb", name) {
            let v = parse_vec3(e.attr("value")?)?;
            Some(Color::from_srgb(v.x, v.y, v.z))
        } else {
            None
        }
    }
}

fn parse_f64(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}

/// "x,y,z" / "x y z"（または単一スカラ）を `Vec3` に解析する。
fn parse_vec3(s: &str) -> Option<Vec3> {
    let parts: Vec<f64> = s
        .split([',', ' '])
        .filter(|t| !t.trim().is_empty())
        .map(parse_f64)
        .collect::<Option<Vec<f64>>>()?;
    match parts.as_slice() {
        [x, y, z] => Some(Vec3::new(*x, *y, *z)),
        [v] => Some(Vec3::new(*v, *v, *v)), // スカラはブロードキャスト
        _ => None,
    }
}

fn warn(msg: &str) {
    eprintln!("[mitsuba] warning: {}", msg);
}

fn err(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Mitsuba XML サブセットを読み込み、[`Scene`] を構築する。
pub fn load_scene(path: &str, config: &RenderConfig) -> io::Result<Scene> {
    let xml = std::fs::read_to_string(path)?;
    let root = parse_tree(&xml)?;
    if root.tag != "scene" {
        return Err(err("root element is not <scene>"));
    }

    // OBJ パスは XML ファイルのあるディレクトリからの相対で解決する。
    let base_dir = Path::new(path).parent().map(Path::to_path_buf).unwrap_or_default();

    let aspect = config.width as f64 / config.height as f64;
    let mut world = World::new();
    let mut mats: Vec<Material> = Vec::new();
    let mut cam: Option<Camera> = None;

    for child in &root.children {
        match child.tag.as_str() {
            "sensor" => {
                if child.typ() == "perspective" {
                    cam = Some(parse_sensor(child, aspect));
                } else {
                    warn(&format!("unsupported sensor type '{}', ignored", child.typ()));
                }
            }
            "shape" => parse_shape(child, &base_dir, &mut world, &mut mats),
            // レンダリング設定ブロックは無視（このレンダラーは CLI で制御する）
            "integrator" | "sampler" | "film" | "default" | "rfilter" => {}
            other => warn(&format!("unsupported element <{}>, skipped", other)),
        }
    }

    let cam = cam.unwrap_or_else(|| {
        warn("no perspective sensor found; using a default camera");
        Camera::look_at(
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            40.0,
            aspect,
        )
    });

    world.build_lights(&mats);
    Ok(Scene { cam, world, mats, env: None })
}

/// XML を要素ツリーに解析する。
fn parse_tree(xml: &str) -> io::Result<Element> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => stack.push(make_element(&e)?),
            Ok(Event::Empty(e)) => {
                let el = make_element(&e)?;
                attach(&mut stack, &mut root, el);
            }
            Ok(Event::End(_)) => {
                let el = stack.pop().ok_or_else(|| err("unbalanced XML"))?;
                attach(&mut stack, &mut root, el);
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
        }
    }

    root.ok_or_else(|| err("empty XML"))
}

/// 完成した要素を親（あれば）へ、無ければルートとして格納する。
fn attach(stack: &mut [Element], root: &mut Option<Element>, el: Element) {
    if let Some(parent) = stack.last_mut() {
        parent.children.push(el);
    } else {
        *root = Some(el);
    }
}

fn make_element(e: &BytesStart) -> io::Result<Element> {
    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
    let mut attrs = HashMap::new();
    for a in e.attributes() {
        let a = a.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let key = String::from_utf8_lossy(a.key.as_ref()).into_owned();
        let val = a
            .unescape_value()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .into_owned();
        attrs.insert(key, val);
    }
    Ok(Element { tag, attrs, children: Vec::new() })
}

/// `sensor type="perspective"` を `Camera` にマップする。
fn parse_sensor(el: &Element, aspect: f64) -> Camera {
    let fov = el.float("fov").unwrap_or(40.0);
    let fov_axis = el.string("fov_axis").unwrap_or("x");
    // 内部カメラは垂直 fov を取るため、水平指定は変換する。
    let vfov = match fov_axis {
        "y" => fov,
        _ => {
            // 水平 fov → 垂直 fov
            let h = fov.to_radians();
            (2.0 * ((h * 0.5).tan() / aspect).atan()).to_degrees()
        }
    };

    let (eye, target, up) = el
        .child_tag("transform")
        .and_then(|t| t.child_tag("lookat"))
        .and_then(parse_lookat)
        .unwrap_or((
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ));

    let aperture_radius = el.float("aperture_radius").unwrap_or(0.0);
    let focus = el.float("focus_distance").unwrap_or((eye - target).len());

    // look_at_dof は lens_radius = 0.5 * aperture とするため、aperture_radius を
    // レンズ半径として渡すには 2 倍する。
    Camera::look_at_dof(eye, target, up, vfov, aspect, focus, 2.0 * aperture_radius)
}

/// `<lookat origin=".." target=".." up=".."/>` を解析する。
fn parse_lookat(el: &Element) -> Option<(Vec3, Vec3, Vec3)> {
    let origin = parse_vec3(el.attr("origin")?)?;
    let target = parse_vec3(el.attr("target")?)?;
    let up = el
        .attr("up")
        .and_then(parse_vec3)
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    Some((origin, target, up))
}

/// `shape` を World へ追加する（sphere / obj メッシュ）。
fn parse_shape(el: &Element, base_dir: &Path, world: &mut World, mats: &mut Vec<Material>) {
    // area emitter があれば面光源、なければ bsdf、どちらも無ければ拡散にフォールバック。
    let mat = if let Some(em) = el.child_tag("emitter") {
        parse_emitter(em)
    } else if let Some(b) = el.child_tag("bsdf") {
        parse_bsdf(b)
    } else {
        warn(&format!("shape type '{}' without bsdf or emitter; defaulting to diffuse", el.typ()));
        Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }
    };
    let mat_id = mats.len();

    match el.typ() {
        "sphere" => {
            let center = el.point("center").unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            let radius = el.float("radius").unwrap_or(1.0);
            mats.push(mat);
            world.spheres.push(Sphere { c: center, r: radius, mat_id });
        }
        "obj" => {
            let filename = match el.string("filename") {
                Some(f) => f,
                None => {
                    warn("obj shape without filename; skipped");
                    return;
                }
            };
            let resolved = resolve_path(base_dir, filename);
            let tris = match load_obj_triangles(resolved.to_string_lossy().as_ref(), mat_id) {
                Ok(t) => t,
                Err(e) => {
                    warn(&format!("failed to load obj '{}': {}; skipped", resolved.display(), e));
                    return;
                }
            };
            // to_world 変換（なければ恒等）でインスタンス配置する。
            let xform = el
                .child_tag("transform")
                .map(parse_transform)
                .unwrap_or_else(Transform::identity);

            mats.push(mat);
            let mesh_id = world.meshes.len();
            world.meshes.push(Mesh::new(tris));
            world.instances.push(Instance { mesh_id, xform, mat_override: None });
        }
        other => warn(&format!("unsupported shape type '{}', skipped", other)),
    }
}

/// ファイル名を XML のあるディレクトリ基準で解決する（絶対パスはそのまま）。
fn resolve_path(base_dir: &Path, filename: &str) -> PathBuf {
    let p = Path::new(filename);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

/// `<transform>` の子（translate / rotate / scale / matrix）を文書順に合成する。
/// Mitsuba 規約に従い、`trafo = op1 · op2 · … · opN`（最後の子が最も内側）。
fn parse_transform(el: &Element) -> Transform {
    let mut acc = Transform::identity();
    for child in &el.children {
        if let Some(op) = parse_transform_op(child) {
            acc = acc.compose(op);
        }
    }
    acc
}

fn parse_transform_op(el: &Element) -> Option<Transform> {
    match el.tag.as_str() {
        "translate" => Some(Transform::translate(xyz(el, 0.0))),
        "scale" => {
            // value（均一）または x/y/z（成分ごと）
            if let Some(v) = el.attr("value").and_then(parse_f64) {
                Some(Transform::scale(Vec3::new(v, v, v)))
            } else {
                Some(Transform::scale(xyz(el, 1.0)))
            }
        }
        "rotate" => {
            let axis = xyz(el, 0.0);
            let angle = el.attr("angle").and_then(parse_f64).unwrap_or(0.0);
            if axis.len() < 1e-12 {
                warn("rotate with zero axis; ignored");
                None
            } else {
                Some(Transform::rotate(axis, angle))
            }
        }
        "matrix" => {
            let vals: Vec<f64> = el
                .attr("value")?
                .split([',', ' '])
                .filter(|t| !t.trim().is_empty())
                .filter_map(parse_f64)
                .collect();
            if vals.len() == 16 {
                let mut m = [[0.0; 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        m[r][c] = vals[r * 4 + c];
                    }
                }
                Some(Transform::from_matrix4(m))
            } else {
                warn(&format!("matrix expects 16 values, got {}; ignored", vals.len()));
                None
            }
        }
        "lookat" => None, // sensor 以外の lookat は未対応
        other => {
            warn(&format!("unsupported transform op <{}>, ignored", other));
            None
        }
    }
}

/// 要素の x/y/z 属性を `Vec3` に読む（欠落は `default`）。
fn xyz(el: &Element, default: f64) -> Vec3 {
    let g = |k: &str| el.attr(k).and_then(parse_f64).unwrap_or(default);
    Vec3::new(g("x"), g("y"), g("z"))
}

/// `emitter` を発光マテリアルにマップする。
fn parse_emitter(el: &Element) -> Material {
    let emit = el.color("radiance").unwrap_or(Color::new(1.0, 1.0, 1.0));
    if el.typ() != "area" {
        warn(&format!("emitter type '{}' treated as area light", el.typ()));
    }
    Material::DiffuseLight { emit }
}

/// `bsdf` をマテリアルにマップする。
fn parse_bsdf(el: &Element) -> Material {
    match el.typ() {
        // 両面 BSDF はラッパーなので内側を展開
        "twosided" => el
            .child_tag("bsdf")
            .map(parse_bsdf)
            .unwrap_or(Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }),
        "diffuse" => Material::Lambert {
            albedo: el.color("reflectance").unwrap_or(Color::new(0.5, 0.5, 0.5)),
        },
        "conductor" => Material::Metal {
            albedo: el.color("specular_reflectance").unwrap_or(Color::new(1.0, 1.0, 1.0)),
        },
        "roughconductor" => {
            let dist = el.string("distribution").unwrap_or("ggx");
            if dist != "ggx" {
                warn(&format!("roughconductor distribution '{}' unsupported; using ggx", dist));
            }
            Material::Ggx {
                albedo: el.color("specular_reflectance").unwrap_or(Color::new(1.0, 1.0, 1.0)),
                alpha: el.float("alpha").unwrap_or(0.1),
            }
        }
        "dielectric" | "thindielectric" | "roughdielectric" => {
            let int_ior = el.float("int_ior").unwrap_or(1.5);
            let ext_ior = el.float("ext_ior").unwrap_or(1.0);
            // absorption は独自拡張（標準 Mitsuba は medium で表現）
            let absorption = el.color("absorption").unwrap_or(Color::new(0.0, 0.0, 0.0));
            Material::Dielectric { ior: int_ior / ext_ior, absorption }
        }
        other => {
            warn(&format!("unsupported bsdf type '{}'; defaulting to diffuse", other));
            Material::Lambert {
                albedo: el.color("reflectance").unwrap_or(Color::new(0.5, 0.5, 0.5)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RenderConfig {
        let mut c = RenderConfig::default();
        c.width = 16;
        c.height = 9;
        c
    }

    fn load(xml: &str) -> Scene {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        // 並列テストでファイル名が衝突しないよう、プロセス ID + 連番で一意化する
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tinypt_mitsuba_{}_{}.xml", std::process::id(), n));
        let path = path.to_str().unwrap().to_string();
        std::fs::write(&path, xml).unwrap();
        let scene = load_scene(&path, &cfg()).unwrap();
        std::fs::remove_file(&path).ok();
        scene
    }

    #[test]
    fn parses_shapes_and_materials() {
        let scene = load(
            r#"<scene version="3.0.0">
              <sensor type="perspective">
                <float name="fov" value="40"/>
                <string name="fov_axis" value="y"/>
                <transform name="to_world">
                  <lookat origin="0,1.2,4" target="0,0.5,0" up="0,1,0"/>
                </transform>
              </sensor>
              <shape type="sphere">
                <point name="center" x="-1" y="0.5" z="0"/>
                <float name="radius" value="0.5"/>
                <bsdf type="diffuse"><srgb name="reflectance" value="0.8,0.3,0.3"/></bsdf>
              </shape>
              <shape type="sphere">
                <point name="center" x="0" y="0.5" z="0"/>
                <float name="radius" value="0.5"/>
                <bsdf type="roughconductor">
                  <string name="distribution" value="ggx"/>
                  <float name="alpha" value="0.25"/>
                  <rgb name="specular_reflectance" value="0.9,0.7,0.3"/>
                </bsdf>
              </shape>
              <shape type="sphere">
                <point name="center" x="0" y="3" z="-1"/>
                <float name="radius" value="0.8"/>
                <emitter type="area"><rgb name="radiance" value="8,7,5"/></emitter>
              </shape>
            </scene>"#,
        );

        assert_eq!(scene.world.spheres.len(), 3);
        assert_eq!(scene.mats.len(), 3);
        assert!(matches!(scene.mats[0], Material::Lambert { .. }));
        assert!(matches!(scene.mats[1], Material::Ggx { alpha, .. } if (alpha - 0.25).abs() < 1e-12));
        assert!(matches!(scene.mats[2], Material::DiffuseLight { .. }));
        // area emitter は build_lights でライトとして登録される
        assert_eq!(scene.world.lights.len(), 1);
    }

    #[test]
    fn srgb_tag_gamma_decodes_but_rgb_is_linear() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="sphere">
                <bsdf type="diffuse"><srgb name="reflectance" value="0.8,0.8,0.8"/></bsdf>
              </shape>
              <shape type="sphere">
                <bsdf type="diffuse"><rgb name="reflectance" value="0.8,0.8,0.8"/></bsdf>
              </shape>
            </scene>"#,
        );
        let srgb = match scene.mats[0] { Material::Lambert { albedo } => albedo, _ => panic!() };
        let lin = match scene.mats[1] { Material::Lambert { albedo } => albedo, _ => panic!() };
        // srgb 0.8 はガンマ展開で約 0.603、rgb 0.8 はそのまま 0.8
        assert!((srgb.r() - Color::from_srgb(0.8, 0.8, 0.8).r()).abs() < 1e-12);
        assert!((lin.r() - 0.8).abs() < 1e-12);
        assert!(srgb.r() < lin.r());
    }

    #[test]
    fn parses_obj_mesh_with_transform() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_m2_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        let xml = format!(
            r#"<scene version="3.0.0">
              <shape type="obj">
                <string name="filename" value="{}"/>
                <transform name="to_world"><translate x="10" y="0" z="0"/></transform>
                <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
              </shape>
            </scene>"#,
            objname
        );
        let xmlpath = dir.join(format!("tinypt_m2_{}_{}.xml", std::process::id(), n));
        std::fs::write(&xmlpath, &xml).unwrap();
        let scene = load_scene(xmlpath.to_str().unwrap(), &cfg()).unwrap();
        std::fs::remove_file(&obj).ok();
        std::fs::remove_file(&xmlpath).ok();

        assert_eq!(scene.world.meshes.len(), 1);
        assert_eq!(scene.world.instances.len(), 1);
        assert_eq!(scene.world.meshes[0].tris.len(), 1);
        assert_eq!(scene.mats.len(), 1);
        // 頂点 v0=(0,0,0) は translate(10,0,0) でワールド (10,0,0) になる
        let inst = scene.world.instances[0];
        let p = inst.xform.apply_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((p - Vec3::new(10.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn unknown_bsdf_falls_back_to_diffuse() {
        let scene = load(
            r#"<scene version="3.0.0">
              <integrator type="path"><integer name="max_depth" value="8"/></integrator>
              <shape type="sphere">
                <bsdf type="plastic"><rgb name="reflectance" value="0.2,0.4,0.6"/></bsdf>
              </shape>
            </scene>"#,
        );
        assert_eq!(scene.mats.len(), 1);
        assert!(matches!(scene.mats[0], Material::Lambert { .. }));
    }
}
