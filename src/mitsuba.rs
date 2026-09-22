//! Mitsuba XML（サブセット）シーンローダー。
//!
//! [Mitsuba レンダラー](https://www.mitsuba-renderer.org/) の XML シーン記述の
//! サブセットを読み込み、[`Scene`] を構築する。採用理由とマッピング方針は
//! `docs/adr/0002-mitsuba-xml-scene-format.md` を参照。
//!
//! ## 対応要素
//! - `sensor type="perspective"`: `fov` / `fov_axis` / `to_world`(`lookat`) / `aperture_radius` / `focus_distance`
//! - `shape type="sphere"`: `center` / `radius`
//! - `shape type="obj"`: `filename`（XML 相対）+ `to_world`（translate/rotate/scale/matrix）
//! - `bsdf`: `diffuse` / `conductor` / `roughconductor`(ggx) / `dielectric` / `thindielectric`・`roughdielectric`(dielectric 扱い) / `twosided`(unwrap)。未知の型は警告して diffuse
//! - `emitter type="area"`: `radiance`（shape に付随）
//! - `emitter type="envmap"`(filename) / `constant`(radiance): 環境マップ。`scale` 対応
//! - `film`(width/height) / `sampler`(sample_count) / `integrator`(max_depth/rr_depth、Mitsuba と同じ意味、max_depth=-1 は無制限): RenderConfig へ反映
//!
//! ## 方針
//! - 色: `<rgb>` はリニア、`<srgb>` は sRGB（ガンマ展開）。
//! - 未対応の要素・型は警告してスキップ／フォールバック（寛容）。必須フィールド欠落のみ既定値。
//! - スペクトルは扱わず RGB トリプルとして読む。
//! - 環境 emitter が無ければ背景は黒（Mitsuba 準拠）。組み込みデフォルトシーンの
//!   手続き的な `sky()` フォールバックは適用しない。

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use crate::config::RenderConfig;
use crate::env::EnvMap;
use crate::geometry::{Sphere, Triangle};
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::material::TexId;
use crate::mtl::{parse_mtl, MtlFile, MtlMaterial};
use crate::constants::normal_map::MTL_BUMP_K;
use crate::normal_map::{HeightMap, MapId, NormalMap};
use crate::obj_loader::{load_obj_groups, load_obj_mesh, MeshData, ObjGroups};
use crate::texture::{AlphaMask, Texture, Wrap};
use std::sync::Arc;
use crate::ray::Camera;
use crate::scene::Scene;
use crate::transform::Transform;
use crate::world::World;

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

    fn int(&self, name: &str) -> Option<usize> {
        self.prop("integer", name)?.attr("value")?.trim().parse().ok()
    }

    /// 負の値も読める整数プロパティ（`max_depth = -1` など）。
    fn int_signed(&self, name: &str) -> Option<i64> {
        self.prop("integer", name)?.attr("value")?.trim().parse().ok()
    }

    fn string(&self, name: &str) -> Option<&str> {
        self.prop("string", name)?.attr("value")
    }

    /// `<boolean name="..." value="true|false"/>` を既定値つきで読む。
    ///
    /// Mitsuba の boolean は `true` / `false` のみ（`1` / `yes` / `TRUE` は不正）。
    /// 要素はあるのに値が解釈できない場合は**警告して既定値にフォールバック**する。
    /// 黙って既定値にすると `value="1"` と書いた人が逆の挙動を静かに得てしまうため
    /// （README の「未対応の要素・型・属性は警告してスキップ」に合わせる）。
    fn boolean_or(&self, name: &str, default: bool) -> bool {
        let Some(e) = self.prop("boolean", name) else { return default };
        match e.attr("value").map(str::trim) {
            Some("true") => true,
            Some("false") => false,
            Some(other) => {
                warn(&format!(
                    "boolean '{}' has invalid value '{}' (expected true or false); using {}",
                    name, other, default
                ));
                default
            }
            None => {
                warn(&format!("boolean '{}' has no value attribute; using {}", name, default));
                default
            }
        }
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

#[cfg(test)]
thread_local! {
    /// テスト中に `warn` が出したメッセージを記録するバッファ（[`capture_warnings`] が有効化する）。
    /// 警告は stderr に出るだけなので、そのままでは「警告を出すこと」をテストできない。
    static CAPTURED_WARNINGS: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

fn warn(msg: &str) {
    #[cfg(test)]
    CAPTURED_WARNINGS.with(|w| {
        if let Some(v) = w.borrow_mut().as_mut() {
            v.push(msg.to_string());
        }
    });
    eprintln!("[mitsuba] warning: {}", msg);
}

/// `f` の実行中に出た警告を集めて返す（同じスレッド内のみ。テスト専用）。
#[cfg(test)]
fn capture_warnings<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    CAPTURED_WARNINGS.with(|w| *w.borrow_mut() = Some(Vec::new()));
    let out = f();
    let msgs = CAPTURED_WARNINGS.with(|w| w.borrow_mut().take()).unwrap_or_default();
    (out, msgs)
}

fn err(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// シーンファイル（`<film>`/`<sampler>`/`<integrator>`）が要求する設定値。
/// `None` の項目は XML に記述が無かったことを示し、呼び出し側の既定値を維持する。
#[derive(Default, Clone, Copy, Debug)]
pub struct SceneSettings {
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub spp: Option<usize>,
    pub max_depth: Option<usize>,
    pub rr_depth: Option<usize>,
}

impl SceneSettings {
    /// `config` へ反映する。`None` の項目は変更しない。
    pub fn apply(&self, config: &mut RenderConfig) {
        if let Some(w) = self.width {
            config.width = w;
        }
        if let Some(h) = self.height {
            config.height = h;
        }
        if let Some(spp) = self.spp {
            config.spp = spp;
        }
        if let Some(m) = self.max_depth {
            config.max_depth = m;
        }
        if let Some(r) = self.rr_depth {
            config.rr_depth = r;
        }
    }
}

/// Mitsuba XML サブセット文字列を解析し、[`Scene`] と [`SceneSettings`] を返す。
///
/// ファイル I/O を含まない純粋な関数（`parse_tree` の上に構築）なので、テストは
/// 一時ファイルを介さず XML 文字列を直接渡せる。`base_dir` は `obj`/環境マップの
/// 相対パス解決に使う基準ディレクトリ。`base_config` は `<film>` 未指定時の
/// フォールバック解像度（アスペクト比計算用）——書き換えない。
pub fn load_scene_from_str(
    xml: &str,
    base_dir: &Path,
    base_config: &RenderConfig,
    forced_resolution: (Option<usize>, Option<usize>),
) -> io::Result<(Scene, SceneSettings)> {
    let root = parse_tree(xml)?;
    if root.tag != "scene" {
        return Err(err("root element is not <scene>"));
    }

    // 1st pass: レンダリング設定（film / sampler / integrator）を settings に集める。
    let mut settings = SceneSettings::default();
    for child in &root.children {
        match child.tag.as_str() {
            "film" => {
                if let Some(w) = child.int("width") {
                    settings.width = Some(w.max(1));
                }
                if let Some(h) = child.int("height") {
                    settings.height = Some(h.max(1));
                }
            }
            "sampler" => {
                if let Some(n) = child.int("sample_count") {
                    settings.spp = Some(n.max(1));
                }
            }
            "integrator" => {
                // Mitsuba の max_depth / rr_depth をそのままの意味で使う（パス長。1 = 直接見える
                // 発光体のみ、2 = 直接照明まで）。max_depth = -1 は無制限（Russian Roulette で打ち切る）。
                if let Some(d) = child.int_signed("max_depth") {
                    match d {
                        -1 => settings.max_depth = Some(usize::MAX),
                        d if d >= 0 => settings.max_depth = Some(d as usize),
                        d => warn(&format!("integrator max_depth {} is invalid (expected -1 or >= 0); ignored", d)),
                    }
                }
                if let Some(r) = child.int_signed("rr_depth") {
                    if r >= 1 {
                        settings.rr_depth = Some(r as usize);
                    } else {
                        warn(&format!("integrator rr_depth {} is invalid (expected >= 1); ignored", r));
                    }
                }
            }
            _ => {}
        }
    }

    // アスペクト比は最終的な解像度から決める。CLI で解像度が明示されていればそれが最優先で、
    // 次に settings（film 指定）、無ければ base_config。センサーはこの aspect で構築されるので、
    // ここで最終値を使わないと CLI 指定時に画角がずれる。
    // 片方だけの CLI 指定（`--width` のみ等）では、もう片方はシーンファイルの `<film>`、
    // それも無ければ base_config を使う。解決した値を settings に書き戻すので、
    // 呼び出し側は `settings.apply` だけで最終解像度を得られる（優先順位の分岐は 1 か所）。
    let width = forced_resolution.0.or(settings.width).unwrap_or(base_config.width);
    let height = forced_resolution.1.or(settings.height).unwrap_or(base_config.height);
    settings.width = Some(width);
    settings.height = Some(height);
    let aspect = width as f64 / height as f64;
    let mut world = World::new();
    let mut mats: Vec<Material> = Vec::new();
    let mut textures: Vec<Texture> = Vec::new();
    let mut mtl_state = MtlState::default();
    let mut normal_maps: Vec<NormalMap> = Vec::new();
    let mut mat_maps: Vec<Option<MapId>> = Vec::new();
    let mut cam: Option<Camera> = None;
    let mut env: Option<EnvMap> = None;

    for child in &root.children {
        match child.tag.as_str() {
            "sensor" => {
                if child.typ() == "perspective" {
                    cam = Some(parse_sensor(child, aspect));
                } else {
                    warn(&format!("unsupported sensor type '{}', ignored", child.typ()));
                }
            }
            "shape" => parse_shape(
                child, base_dir, &mut world, &mut mats, &mut mat_maps, &mut textures, &mut normal_maps, &mut mtl_state,
            ),
            // シーン直下の emitter は環境マップ（envmap / constant）
            "emitter" => {
                if let Some(e) = parse_scene_emitter(child, base_dir) {
                    env = Some(e);
                }
            }
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

    // Mitsuba 準拠: 環境 emitter が無ければ背景は黒。
    // （未指定だと integrator が手続き的な sky() を返し、開いたシーンに環境光が漏れ込むため）
    let env = Some(env.unwrap_or_else(|| EnvMap::constant(Color::new(0.0, 0.0, 0.0))));

    world.build_lights(&mats);
    // マップを持つ材質が 1 つも無ければテーブルを空にする（積分器は空テーブルなら何も引かない）
    if mat_maps.iter().all(|m| m.is_none()) {
        mat_maps.clear();
    }
    debug_assert!(mat_maps.is_empty() || mat_maps.len() == mats.len());
    Ok((Scene { cam, world, mats, textures, normal_maps, mat_maps, env }, settings))
}

/// ファイルパスから Mitsuba シーンを読み込む（[`load_scene_from_str`] の薄いファイル I/O
/// アダプタ）。読み取った設定値は `config` に反映する（`None` の項目は現状維持）。
/// CLI 明示値で上書きしたい場合は呼び出し側で行う（[`crate::load_scene`] の利用側参照）。
pub fn load_scene(
    path: &str,
    config: &mut RenderConfig,
    forced_resolution: (Option<usize>, Option<usize>),
) -> io::Result<Scene> {
    let xml = std::fs::read_to_string(path)?;
    // OBJ パスは XML ファイルのあるディレクトリからの相対で解決する。
    let base_dir = Path::new(path).parent().map(Path::to_path_buf).unwrap_or_default();
    let (scene, settings) = load_scene_from_str(&xml, &base_dir, config, forced_resolution)?;
    // settings には CLI 指定を織り込んだ最終解像度が入っている（load_scene_from_str 参照）。
    settings.apply(config);
    Ok(scene)
}

/// XML が参照する外部ファイル（`<string name="filename">`）を文書順に列挙する。
///
/// パスは [`load_scene_from_str`] と同じく `base_dir` 基準で解決する。
/// チェックポイントのシーンハッシュが OBJ/環境マップの内容まで含むために使う。
pub fn referenced_files(xml: &str, base_dir: &Path) -> io::Result<Vec<PathBuf>> {
    fn walk(el: &Element, base_dir: &Path, out: &mut Vec<PathBuf>) {
        if el.tag == "string" && el.attr("name") == Some("filename") {
            if let Some(v) = el.attr("value") {
                out.push(resolve_path(base_dir, v));
            }
        }
        for c in &el.children {
            walk(c, base_dir, out);
        }
    }
    let root = parse_tree(xml)?;
    let mut out = Vec::new();
    walk(&root, base_dir, &mut out);
    Ok(out)
}

/// シーン直下の `<emitter>`（環境マップ）を `EnvMap` にマップする。
/// `envmap`（ファイル）と `constant`（定数色）に対応。`scale` を放射輝度に乗算する。
fn parse_scene_emitter(el: &Element, base_dir: &Path) -> Option<EnvMap> {
    if el.child_tag("transform").is_some() {
        warn("envmap to_world rotation is unsupported; ignored");
    }
    let scale = el.float("scale").unwrap_or(1.0);
    match el.typ() {
        "envmap" => {
            let filename = el.string("filename")?;
            let resolved = resolve_path(base_dir, filename);
            match EnvMap::from_hdr(resolved.to_string_lossy().as_ref()) {
                Ok(m) => Some(m.scaled(scale)),
                Err(e) => {
                    warn(&format!("failed to load envmap '{}': {}; ignored", resolved.display(), e));
                    None
                }
            }
        }
        "constant" => {
            let radiance = el.color("radiance").unwrap_or(Color::new(1.0, 1.0, 1.0));
            Some(EnvMap::constant(radiance).scaled(scale))
        }
        other => {
            warn(&format!("unsupported scene emitter type '{}', ignored", other));
            None
        }
    }
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

/// `shape` を World へ追加する。
/// `sphere` は解析的プリミティブ、`obj` / `rectangle` / `cube` / `disk` は
/// 三角形メッシュ + インスタンス（`to_world` 変換）として配置する。
#[allow(clippy::too_many_arguments)]
fn parse_shape(
    el: &Element,
    base_dir: &Path,
    world: &mut World,
    mats: &mut Vec<Material>,
    mat_maps: &mut Vec<Option<MapId>>,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
    mtl_state: &mut MtlState,
) {
    // OBJ で `<bsdf>` も `<emitter>` も無く、`use_mtl` が false でなければ MTL から材質を作る。
    // `<bsdf>` 指定があれば従来どおり全体を上書きする（既存シーンの見た目・出力を保つ）。
    if el.typ() == "obj"
        && el.child_tag("bsdf").is_none()
        && el.child_tag("emitter").is_none()
        && el.boolean_or("use_mtl", true)
    {
        parse_obj_with_mtl(el, base_dir, world, mats, mat_maps, textures, normal_maps, mtl_state);
        return;
    }
    // area emitter があれば面光源、なければ bsdf、どちらも無ければ拡散にフォールバック。
    let (mat, map) = if let Some(em) = el.child_tag("emitter") {
        (parse_emitter(em), None)
    } else if let Some(b) = el.child_tag("bsdf") {
        parse_bsdf(b, base_dir, textures, normal_maps)
    } else {
        warn(&format!("shape type '{}' without bsdf or emitter; defaulting to diffuse", el.typ()));
        (Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }, None)
    };
    let mat_id = mats.len();

    // メッシュ系シェープの三角形（正準形オブジェクト空間）。
    // Mitsuba の `face_normals`: true なら頂点法線を使わず面法線だけで陰影を付ける。
    // 既定は false（= OBJ に頂点法線があれば補間する）。パラメトリック形状は元から
    // 頂点法線を持たないので、この指定があっても結果は変わらない。
    let face_normals = el.boolean_or("face_normals", false);
    let mesh: MeshData = match el.typ() {
        "sphere" => {
            let center = el.point("center").unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            let radius = el.float("radius").unwrap_or(1.0);
            push_material(mats, mat_maps, mat, map);
            world.add_sphere(Sphere { c: center, r: radius, mat_id });
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
            match load_obj_mesh(resolved.to_string_lossy().as_ref(), mat_id) {
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
    let xform = el
        .child_tag("transform")
        .map(parse_transform)
        .unwrap_or_else(Transform::identity);
    push_material(mats, mat_maps, mat, map);
    world.add_mesh_data_instance(mesh, xform, None);
}

/// 材質を `mats` に積む**唯一の入口**。`mat_maps` は常に `mats` と同じ長さに保つ
/// （ずれると別の材質にマップが掛かる）。返り値は `mat_id`。
fn push_material(mats: &mut Vec<Material>, mat_maps: &mut Vec<Option<MapId>>, mat: Material, map: Option<MapId>) -> usize {
    debug_assert_eq!(mats.len(), mat_maps.len(), "mats and mat_maps out of sync");
    mats.push(mat);
    mat_maps.push(map);
    mats.len() - 1
}

/// 法線マップの種別（キャッシュのキー）。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MapKind {
    Height,
    Tangent,
}

/// MTL 読み込みの状態（シーン読み込み 1 回ぶん）。
#[derive(Default)]
struct MtlState {
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
}

/// `usemtl` の材質を `mat_id` に振り直しつつ、OBJ を 1 メッシュのままインスタンス配置する。
///
/// 材質は「面に実際に使われた名前」だけを `mats` に積む（三角形ごとの `mat_id` で引く）。
/// `Instance.mat_override` は使わない（None のまま = 三角形の `mat_id` が効く）。
fn parse_obj_with_mtl(
    el: &Element,
    base_dir: &Path,
    world: &mut World,
    mats: &mut Vec<Material>,
    mat_maps: &mut Vec<Option<MapId>>,
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
    let ObjGroups { mut mesh, mat_names, mtllibs } = match load_obj_groups(resolved.to_string_lossy().as_ref()) {
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
    let mut maps: Vec<Option<MapId>> = Vec::with_capacity(mat_names.len());
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
                maps.push(None);
                if !name.is_empty() && !mtllibs.is_empty() {
                    warn(&format!("material '{}' not found in mtl; defaulting to diffuse", name));
                }
                Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None }
            }
        };
        push_material(mats, mat_maps, mat, maps[mi]);
    }
    for t in mesh.tris.iter_mut() {
        t.mat_id += base;
    }

    let xform = el
        .child_tag("transform")
        .map(parse_transform)
        .unwrap_or_else(Transform::identity);
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
        world.add_mesh_data_instance_with_alpha(mesh, masks, tri_alpha, xform);
    } else {
        world.add_mesh_data_instance(mesh, xform, None);
    }
}

/// MTL のテクスチャを（キャッシュ経由で）読む。色テクスチャなので sRGB デコード。
fn load_mtl_texture(dir: &Path, rel: &str, textures: &mut Vec<Texture>, state: &mut MtlState) -> Option<TexId> {
    let joined = dir.join(rel);
    let key = std::fs::canonicalize(&joined).unwrap_or(joined);
    if let Some(&cached) = state.tex_cache.get(&key) {
        return cached;
    }
    let id = match Texture::load(key.to_string_lossy().as_ref(), true, Wrap::Repeat) {
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

/// MTL の `norm` / `map_bump` を読んで `normal_maps` に登録し、その添字を返す（キャッシュ経由）。
/// どちらもリニア（データ）として読む。読み込みに失敗したら警告して `None`（摂動なし）。
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
        MapKind::Tangent => Texture::load(path.as_ref(), false, Wrap::Repeat)
            .map(|tex| NormalMap::Tangent { tex, scale: 1.0 }),
        MapKind::Height => {
            HeightMap::load(path.as_ref(), Wrap::Repeat).map(|map| NormalMap::Height { map, strength })
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

/// MTL の `map_d` をアルファマスクとして（キャッシュ経由で）読む。リニア（データ）扱い。
fn load_mtl_mask(dir: &Path, rel: &str, state: &mut MtlState) -> Option<Arc<AlphaMask>> {
    let joined = dir.join(rel);
    let key = std::fs::canonicalize(&joined).unwrap_or(joined);
    if let Some(cached) = state.mask_cache.get(&key) {
        return cached.clone();
    }
    let m = match AlphaMask::load(key.to_string_lossy().as_ref(), Wrap::Repeat) {
        Ok(m) => Some(Arc::new(m)),
        Err(e) => {
            warn(&format!("failed to load alpha mask '{}': {}; ignored", key.display(), e));
            None
        }
    };
    state.mask_cache.insert(key, m.clone());
    m
}

/// MTL 材質 → BSDF（README「MTL → BSDF のマッピング」）。
/// 未対応キーの警告は**材質ごとに 1 回**（面ごとには出さない）。
fn mtl_to_material(
    m: &MtlMaterial,
    dir: &Path,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
    state: &mut MtlState,
) -> (Material, Option<(Arc<AlphaMask>, f32)>, Option<MapId>) {
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
        return (Material::Lambert { albedo: kd, albedo_tex: tex }, alpha, map);
    }
    if ks.luminance() > 0.05 && m.ns > 1.0 {
        // Blinn-Phong 指数 → GGX の粗さ: alpha = sqrt(2 / (Ns + 2))
        let rough = (2.0 / (m.ns + 2.0)).sqrt().clamp(1e-3, 1.0);
        return (Material::Ggx { albedo: ks, alpha: rough }, alpha, map);
    }
    (Material::Lambert { albedo: kd, albedo_tex: None }, alpha, map)
}

/// Mitsuba `rectangle`: 中心原点・法線 +Z・頂点 [-1,1]² の正方形（2 三角形）。
fn unit_rectangle_tris(mat_id: usize) -> MeshData {
    let a = Vec3::new(-1.0, -1.0, 0.0);
    let b = Vec3::new(1.0, -1.0, 0.0);
    let c = Vec3::new(1.0, 1.0, 0.0);
    let d = Vec3::new(-1.0, 1.0, 0.0);
    let tris = vec![
        Triangle::new_static(a, b, c, mat_id),
        Triangle::new_static(a, c, d, mat_id),
    ];
    // Mitsuba の rectangle と同じ割り当て: uv = ((x+1)/2, (y+1)/2)
    // 頂点 a, b, c, d の順に [0,0] [1,0] [1,1] [0,1]
    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    MeshData::with_uv(tris, uv, vec![[0, 1, 2], [0, 2, 3]])
}

/// Mitsuba `cube`: [-1,1]³ の立方体（12 三角形）。
fn unit_cube_tris(mat_id: usize) -> MeshData {
    let v = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    let quads = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ];
    let mut tris = Vec::with_capacity(12);
    let mut tri_uv = Vec::with_capacity(12);
    for q in quads {
        tris.push(Triangle::new_static(v[q[0]], v[q[1]], v[q[2]], mat_id));
        tris.push(Triangle::new_static(v[q[0]], v[q[2]], v[q[3]], mat_id));
        // Mitsuba の cube は面ごとに [0,1]² の UV を張る。四角形の 4 頂点を
        // [0,0] [1,0] [1,1] [0,1] の順に対応させる（quads の並びがその順）
        tri_uv.push([0, 1, 2]);
        tri_uv.push([0, 2, 3]);
    }
    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    MeshData::with_uv(tris, uv, tri_uv)
}

/// Mitsuba `disk`: z=0 平面の半径 1 の円盤（ファン三角形化）。
fn unit_disk_tris(mat_id: usize) -> MeshData {
    let n = 64;
    let center = Vec3::new(0.0, 0.0, 0.0);
    let mut tris = Vec::with_capacity(n);
    let mut uv: Vec<[f64; 2]> = Vec::with_capacity(2 * n + 1);
    let mut tri_uv = Vec::with_capacity(n);
    // Mitsuba の disk は極座標を UV に割り当てる: u = 半径 r、v = 角度 φ/2π。
    // 中心は r = 0 なので u = 0（v は縮退するので扇の始端の角度に合わせる）。
    uv.push([0.0, 0.0]); // 中心（扇ごとに v を変えたいので下で差し替える）
    for i in 0..n {
        let a0 = std::f64::consts::TAU * (i as f64) / (n as f64);
        let a1 = std::f64::consts::TAU * ((i + 1) as f64) / (n as f64);
        tris.push(Triangle::new_static(
            center,
            Vec3::new(a0.cos(), a0.sin(), 0.0),
            Vec3::new(a1.cos(), a1.sin(), 0.0),
            mat_id,
        ));
        let c = uv.len() as u32;
        uv.push([0.0, (i as f64) / (n as f64)]); // この扇の中心（u=0）
        let e0 = uv.len() as u32;
        uv.push([1.0, (i as f64) / (n as f64)]);
        let e1 = uv.len() as u32;
        uv.push([1.0, ((i + 1) as f64) / (n as f64)]);
        tri_uv.push([c, e0, e1]);
    }
    MeshData::with_uv(tris, uv, tri_uv)
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

/// `<texture type="bitmap">` を読み込んで `textures` に積み、その添字を返す。
///
/// 対応するのは `filename`（XML からの相対パス）と `wrap_mode`（`repeat` / `clamp`）、
/// および Mitsuba の `raw`（true でリニア、既定の false なら sRGB デコード）。
/// 読み込みに失敗したら警告して `None`（呼び出し側が定数色にフォールバックする）。
fn parse_texture(el: &Element, base_dir: &Path, textures: &mut Vec<Texture>) -> Option<TexId> {
    let (resolved, wrap) = bitmap_source(el, base_dir)?;
    // 色テクスチャは sRGB。`raw=true` はデータテクスチャ（リニア）
    let srgb = !el.boolean_or("raw", false);
    match Texture::load(resolved.to_string_lossy().as_ref(), srgb, wrap) {
        Ok(t) => {
            textures.push(t);
            Some((textures.len() - 1) as TexId)
        }
        Err(e) => {
            warn(&format!("failed to load texture '{}': {}; ignored", resolved.display(), e));
            None
        }
    }
}

/// `<texture type="bitmap">` の `filename`（XML からの相対）と `wrap_mode` を読む。
/// 型が違う・`filename` が無いときは警告して `None`。
fn bitmap_source(el: &Element, base_dir: &Path) -> Option<(PathBuf, Wrap)> {
    if el.typ() != "bitmap" {
        warn(&format!("unsupported texture type '{}', ignored", el.typ()));
        return None;
    }
    let filename = match el.string("filename") {
        Some(f) => f,
        None => {
            warn("bitmap texture without filename; ignored");
            return None;
        }
    };
    let resolved = resolve_path(base_dir, filename);
    let wrap = match el.string("wrap_mode") {
        Some(w) => match Wrap::from_str(w) {
            Some(w) => w,
            None => {
                warn(&format!("unsupported wrap_mode '{}' (expected repeat or clamp); using repeat", w));
                Wrap::Repeat
            }
        },
        None => Wrap::Repeat,
    };
    Some((resolved, wrap))
}

/// `normalmap` / `bumpmap` ラッパーの子テクスチャ（`name="normalmap"` / `name="bumpmap"`、無名でも可）。
fn wrapper_texture<'a>(el: &'a Element, name: &str) -> Option<&'a Element> {
    el.prop("texture", name)
        .or_else(|| el.children.iter().find(|c| c.tag == "texture" && c.attr("name").is_none()))
}

/// `bsdf` をマテリアルにマップする。法線マップ／バンプマップのラッパーなら、内側の材質と一緒に
/// `normal_maps` に登録したマップの添字も返す（材質は `Copy` のままなので、マップは側テーブルで持つ）。
fn parse_bsdf(
    el: &Element,
    base_dir: &Path,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
) -> (Material, Option<MapId>) {
    let default_mat = || Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5), albedo_tex: None };
    match el.typ() {
        // 両面 BSDF はラッパーなので内側を展開（内側にマップのラッパーがあればそのまま素通し）
        "twosided" => el
            .child_tag("bsdf")
            .map(|c| parse_bsdf(c, base_dir, textures, normal_maps))
            .unwrap_or((default_mat(), None)),
        // Mitsuba 準拠: `<bsdf type="normalmap">` は子のテクスチャ（タンジェント空間ノーマルマップ）を
        // 内側の `<bsdf>` に適用する。テクスチャは**リニア（raw）固定**で読む
        "normalmap" | "bumpmap" => {
            let is_normal = el.typ() == "normalmap";
            let (mat, inner_map) = match el.child_tag("bsdf") {
                Some(inner) => parse_bsdf(inner, base_dir, textures, normal_maps),
                None => {
                    warn(&format!("{} without an inner <bsdf>; defaulting to diffuse", el.typ()));
                    (default_mat(), None)
                }
            };
            if inner_map.is_some() {
                // 1 材質スロットに 1 マップ: 外側を採用し、内側のマップは捨てる（登録済みの分は参照されないまま残る）
                warn(&format!("nested {} inside {}; the outer map is used", el.typ(), el.typ()));
            }
            let tex_el = wrapper_texture(el, if is_normal { "normalmap" } else { "bumpmap" });
            let map = tex_el.and_then(|t| {
                let (path, wrap) = bitmap_source(t, base_dir)?;
                let path_s = path.to_string_lossy();
                if is_normal {
                    if !t.boolean_or("raw", true) {
                        warn("normalmap texture with raw=false: ignoring it and reading the map as linear (raw)");
                    }
                    match Texture::load(path_s.as_ref(), false, wrap) {
                        Ok(tex) => Some(NormalMap::Tangent { tex, scale: 1.0 }),
                        Err(e) => {
                            warn(&format!("failed to load normal map '{}': {}; ignored", path.display(), e));
                            None
                        }
                    }
                } else {
                    match HeightMap::load(path_s.as_ref(), wrap) {
                        Ok(map) => Some(NormalMap::Height { map, strength: el.float("scale").unwrap_or(1.0) }),
                        Err(e) => {
                            warn(&format!("failed to load height map '{}': {}; ignored", path.display(), e));
                            None
                        }
                    }
                }
            });
            if tex_el.is_none() {
                warn(&format!("{} without a <texture>; map ignored", el.typ()));
            }
            match map {
                Some(m) => {
                    normal_maps.push(m);
                    (mat, Some((normal_maps.len() - 1) as MapId))
                }
                None => (mat, inner_map),
            }
        }
        _ => (parse_leaf_bsdf(el, base_dir, textures), None),
    }
}

/// マップのラッパーではない通常の `bsdf` をマテリアルにマップする。
fn parse_leaf_bsdf(el: &Element, base_dir: &Path, textures: &mut Vec<Texture>) -> Material {
    match el.typ() {
        "diffuse" => {
            // `reflectance` はテクスチャか定数色。テクスチャがある場合、定数色は色の倍率になる
            // （両方あれば掛け合わせる。片方だけなら他方は白 = 1 倍）。
            let tex = el
                .prop("texture", "reflectance")
                .and_then(|t| parse_texture(t, base_dir, textures));
            let default = if tex.is_some() { Color::new(1.0, 1.0, 1.0) } else { Color::new(0.5, 0.5, 0.5) };
            Material::Lambert {
                albedo: el.color("reflectance").unwrap_or(default),
                albedo_tex: tex,
            }
        }
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
                albedo_tex: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::Ray;
    use crate::rng::Rng;

    fn cfg() -> RenderConfig {
        let mut c = RenderConfig::default();
        c.width = 16;
        c.height = 9;
        c
    }

    /// referenced_files は入れ子の filename を文書順・base_dir 基準で列挙する。
    #[test]
    fn referenced_files_lists_filenames_in_document_order() {
        let xml = r#"<scene version="3.0.0">
              <shape type="obj"><string name="filename" value="a.obj"/></shape>
              <shape type="sphere"><float name="radius" value="1"/></shape>
              <emitter type="envmap"><string name="filename" value="/abs/env.exr"/></emitter>
            </scene>"#;
        let files = referenced_files(xml, Path::new("dir")).unwrap();
        assert_eq!(files, vec![PathBuf::from("dir/a.obj"), PathBuf::from("/abs/env.exr")]);
    }

    fn load(xml: &str) -> Scene {
        load_scene_from_str(xml, Path::new("."), &cfg(), (None, None)).unwrap().0
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

        assert_eq!(scene.world.spheres().len(), 3);
        assert_eq!(scene.mats.len(), 3);
        assert!(matches!(scene.mats[0], Material::Lambert { .. }));
        assert!(matches!(scene.mats[1], Material::Ggx { alpha, .. } if (alpha - 0.25).abs() < 1e-12));
        assert!(matches!(scene.mats[2], Material::DiffuseLight { .. }));
        // area emitter は build_lights でライトとして登録される
        assert_eq!(scene.world.lights().len(), 1);
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
        let srgb = match scene.mats[0] { Material::Lambert { albedo, .. } => albedo, _ => panic!() };
        let lin = match scene.mats[1] { Material::Lambert { albedo, .. } => albedo, _ => panic!() };
        // srgb 0.8 はガンマ展開で約 0.603、rgb 0.8 はそのまま 0.8
        assert!((srgb.r() - Color::from_srgb(0.8, 0.8, 0.8).r()).abs() < 1e-12);
        assert!((lin.r() - 0.8).abs() < 1e-12);
        assert!(srgb.r() < lin.r());
    }

    #[test]
    fn reads_film_sampler_integrator_into_config() {
        let (_scene, settings) = load_scene_from_str(
            r#"<scene version="3.0.0">
              <integrator type="path">
                <integer name="max_depth" value="12"/>
                <integer name="rr_depth" value="5"/>
              </integrator>
              <sampler type="independent"><integer name="sample_count" value="256"/></sampler>
              <film type="hdrfilm">
                <integer name="width" value="800"/>
                <integer name="height" value="600"/>
              </film>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
            Path::new("."),
            &cfg(),
            (None, None),
        )
        .unwrap();
        let mut config = cfg();
        settings.apply(&mut config);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.spp, 256);
        assert_eq!(config.max_depth, 12);
        assert_eq!(config.rr_depth, 5);
    }

    /// max_depth = -1 は無制限、0 はそのまま（何も描かない）、それ以外の負値と rr_depth < 1 は無視。
    #[test]
    fn integrator_depth_values_follow_mitsuba() {
        let settings_for = |max_depth: &str, rr_depth: &str| {
            let xml = format!(
                r#"<scene version="3.0.0"><integrator type="path"><integer name="max_depth" value="{}"/><integer name="rr_depth" value="{}"/></integrator>
                   <shape type="sphere"><bsdf type="diffuse"/></shape></scene>"#,
                max_depth, rr_depth
            );
            load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().1
        };
        let s = settings_for("-1", "5");
        assert_eq!(s.max_depth, Some(usize::MAX));
        assert_eq!(s.rr_depth, Some(5));
        assert_eq!(settings_for("0", "1").max_depth, Some(0));
        assert_eq!(settings_for("1", "1").max_depth, Some(1));
        let s = settings_for("-2", "0");
        assert_eq!(s.max_depth, None);
        assert_eq!(s.rr_depth, None);
    }

    #[test]
    fn parses_parametric_shapes() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="rectangle"><bsdf type="diffuse"/></shape>
              <shape type="cube"><bsdf type="diffuse"/></shape>
              <shape type="disk"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        assert_eq!(scene.world.meshes().len(), 3);
        assert_eq!(scene.world.instances().len(), 3);
        assert_eq!(scene.world.meshes()[0].tris.len(), 2); // rectangle
        assert_eq!(scene.world.meshes()[1].tris.len(), 12); // cube
        assert_eq!(scene.world.meshes()[2].tris.len(), 64); // disk
    }


    /// `face_normals` の値が `true` / `false` 以外なら警告して既定（補間）にフォールバックする。
    /// 黙って既定値にすると `value="1"` が逆の意味に解釈される。
    #[test]
    fn invalid_face_normals_value_warns_and_falls_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_fnbad_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        let load = |flag: &str| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="obj">
                    <string name="filename" value="{}"/>{}
                    <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
                  </shape>
                </scene>"#,
                objname, flag
            );
            let (r, warnings) = capture_warnings(|| load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0);
            (r, warnings)
        };
        for bad in [r#"<boolean name="face_normals" value="1"/>"#,
                    r#"<boolean name="face_normals" value="yes"/>"#,
                    r#"<boolean name="face_normals" value="TRUE"/>"#,
                    r#"<boolean name="face_normals"/>"#] {
            let (scene, warnings) = load(bad);
            assert!(scene.world.meshes()[0].is_smooth(), "{}: 既定（補間）にフォールバックする", bad);
            assert!(
                warnings.iter().any(|w| w.contains("face_normals")),
                "{}: 警告が出ていない（warnings = {:?}）", bad, warnings
            );
        }
        // 正しい値では警告を出さない
        for good in ["", r#"<boolean name="face_normals" value="true"/>"#, r#"<boolean name="face_normals" value="false"/>"#] {
            let (_, warnings) = load(good);
            assert!(!warnings.iter().any(|w| w.contains("face_normals")), "{}: 余計な警告", good);
        }
        std::fs::remove_file(&obj).ok();
    }

    // ---- テクスチャ（T1） ----

    /// テスト用の PNG を書いて渡す。
    fn with_png<T>(w: u32, h: u32, px: &[[u8; 3]], f: impl FnOnce(&std::path::Path, &str) -> T) -> T {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let name = format!("tinypt_mtex_{}_{}.png", std::process::id(), n);
        let path = dir.join(&name);
        let mut img = image::RgbImage::new(w, h);
        for (i, p) in px.iter().enumerate() {
            img.put_pixel((i as u32) % w, (i as u32) / w, image::Rgb(*p));
        }
        img.save(&path).unwrap();
        let out = f(&dir, &name);
        std::fs::remove_file(&path).ok();
        out
    }

    /// `<texture type="bitmap">` を `diffuse` の `reflectance` に指定でき、
    /// シーンのテクスチャ置き場に積まれてマテリアルが添字で参照する。
    #[test]
    fn parses_bitmap_texture_on_diffuse_reflectance() {
        let scene = with_png(1, 1, &[[255, 0, 0]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        assert_eq!(scene.textures.len(), 1, "テクスチャが積まれていない");
        match scene.mats[0] {
            Material::Lambert { albedo, albedo_tex: Some(id) } => {
                assert_eq!(id, 0);
                // テクスチャがある場合、定数側は倍率なので白（1 倍）
                assert!((albedo.r() - 1.0).abs() < 1e-12, "既定の倍率は白のはず: {}", albedo.r());
                // 赤 255 は sRGB デコードでリニア 1.0
                let c = scene.textures[0].sample((0.5, 0.5));
                assert!((c.r() - 1.0).abs() < 1e-9 && c.g().abs() < 1e-12, "{:?}", (c.r(), c.g(), c.b()));
            }
            _ => panic!("diffuse がテクスチャ付き Lambert になっていない"),
        }
    }

    /// テクスチャと定数色を両方書くと、定数色は**倍率**として掛かる。
    #[test]
    fn constant_reflectance_scales_the_texture() {
        let scene = with_png(1, 1, &[[255, 255, 255]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <rgb name="reflectance" value="0.25, 0.5, 1.0"/>
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        let mat = scene.mats[0];
        let resolved = mat.resolve_textures(&scene.textures, (0.5, 0.5));
        match resolved {
            Material::Lambert { albedo, albedo_tex: None } => {
                assert!((albedo.r() - 0.25).abs() < 1e-9 && (albedo.b() - 1.0).abs() < 1e-9,
                        "倍率が掛かっていない: {:?}", (albedo.r(), albedo.g(), albedo.b()));
            }
            _ => panic!("resolve_textures がテクスチャを畳み込んでいない"),
        }
    }

    /// 読み込めないテクスチャは警告して定数色にフォールバックする（描画は続く）。
    #[test]
    fn missing_texture_warns_and_falls_back_to_a_constant() {
        let dir = std::env::temp_dir();
        let xml = r#"<scene version="3.0.0">
              <shape type="rectangle">
                <bsdf type="diffuse">
                  <texture type="bitmap" name="reflectance">
                    <string name="filename" value="definitely_not_here_12345.png"/>
                  </texture>
                </bsdf>
              </shape>
            </scene>"#;
        let (scene, warnings) = capture_warnings(|| load_scene_from_str(xml, &dir, &cfg(), (None, None)).unwrap().0);
        assert!(scene.textures.is_empty());
        assert!(matches!(scene.mats[0], Material::Lambert { albedo_tex: None, .. }));
        assert!(warnings.iter().any(|w| w.contains("failed to load texture")), "警告が出ていない: {:?}", warnings);
    }

    /// 未知のテクスチャ型・`wrap_mode` は警告する。
    #[test]
    fn unsupported_texture_type_and_wrap_mode_warn() {
        let (_, warnings) = with_png(1, 1, &[[10, 20, 30]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="checkerboard" name="reflectance"/>
                    </bsdf>
                  </shape>
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                        <string name="wrap_mode" value="mirror"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0)
        });
        assert!(warnings.iter().any(|w| w.contains("unsupported texture type")), "{:?}", warnings);
        assert!(warnings.iter().any(|w| w.contains("wrap_mode")), "{:?}", warnings);
    }

    /// `raw="true"` は sRGB デコードを掛けない（データテクスチャ）。
    #[test]
    fn raw_texture_is_linear() {
        let scene = with_png(1, 1, &[[128, 128, 128]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                        <boolean name="raw" value="true"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        let c = scene.textures[0].sample((0.5, 0.5));
        assert!((c.r() - 128.0 / 255.0).abs() < 1e-12, "raw なのに sRGB デコードされている: {}", c.r());
    }

    /// パラメトリック形状には Mitsuba 準拠の UV が付く。
    #[test]
    fn parametric_shapes_get_uv() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="rectangle"><bsdf type="diffuse"/></shape>
              <shape type="cube"><bsdf type="diffuse"/></shape>
              <shape type="disk"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        for (i, name) in ["rectangle", "cube", "disk"].iter().enumerate() {
            assert!(scene.world.meshes()[i].has_uv(), "{} に UV が無い", name);
        }
        // 値の確認は形状ごとに単独のシーンで行う（3 つを重ねると手前の立方体に当たってしまう）
        let only = |shape: &str| {
            load(&format!(r#"<scene version="3.0.0"><shape type="{}"><bsdf type="diffuse"/></shape></scene>"#, shape))
        };
        let shoot = |sc: &crate::scene::Scene, x: f64, y: f64| {
            sc.world
                .hit(Ray { o: Vec3::new(x, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
                .expect("当たらない")
                .uv
        };

        // rectangle: uv = ((x+1)/2, (y+1)/2)
        let rect = only("rectangle");
        let uv = shoot(&rect, 0.0, 0.0);
        assert!((uv.0 - 0.5).abs() < 1e-12 && (uv.1 - 0.5).abs() < 1e-12, "rectangle 中心: {:?}", uv);
        let uv = shoot(&rect, 0.5, -0.5);
        assert!((uv.0 - 0.75).abs() < 1e-12 && (uv.1 - 0.25).abs() < 1e-12, "rectangle (0.5,-0.5): {:?}", uv);

        // cube: 面ごとに [0,1]²。+z 面の中心は (0.5, 0.5)
        let cube = only("cube");
        let uv = shoot(&cube, 0.0, 0.0);
        assert!((uv.0 - 0.5).abs() < 1e-12 && (uv.1 - 0.5).abs() < 1e-12, "cube +z 面の中心: {:?}", uv);
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let (x, y) = (rng.next_f64() * 1.8 - 0.9, rng.next_f64() * 1.8 - 0.9);
            let uv = shoot(&cube, x, y);
            assert!((0.0..=1.0).contains(&uv.0) && (0.0..=1.0).contains(&uv.1), "cube の UV が範囲外: {:?}", uv);
        }

        // disk: u = 半径 r、v = 角度 φ/2π
        let disk = only("disk");
        let uv = shoot(&disk, 0.0, 0.0);
        assert!(uv.0.abs() < 1e-9, "disk の中心は u = 0（r = 0）: {:?}", uv);
        for r in [0.25, 0.5, 0.9] {
            let uv = shoot(&disk, r, 0.0);
            assert!((uv.0 - r).abs() < 0.02, "disk の u は半径: r = {}, uv = {:?}", r, uv);
        }
    }

    /// OBJ の頂点法線は既定で使われ、`<boolean name="face_normals" value="true"/>` で捨てられる。
    /// パラメトリック形状（rectangle など）は元から頂点法線を持たないので、この指定で何も変わらない。
    #[test]
    fn face_normals_flag_controls_vertex_normal_use() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_fn_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        let scene_of = |flag: &str| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="obj">
                    <string name="filename" value="{}"/>{}
                    <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
                  </shape>
                  <shape type="rectangle">{}<bsdf type="diffuse"/></shape>
                </scene>"#,
                objname, flag, flag
            );
            load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0
        };
        let smooth = scene_of("");
        let flat = scene_of(r#"<boolean name="face_normals" value="true"/>"#);
        let explicit_false = scene_of(r#"<boolean name="face_normals" value="false"/>"#);
        std::fs::remove_file(&obj).ok();

        assert!(smooth.world.meshes()[0].is_smooth(), "既定では頂点法線を使う");
        assert!(explicit_false.world.meshes()[0].is_smooth(), "false は既定と同じ");
        assert!(!flat.world.meshes()[0].is_smooth(), "face_normals=true で頂点法線を捨てる");
        // rectangle は元から頂点法線を持たないので、どちらでも面法線
        assert!(!smooth.world.meshes()[1].is_smooth());
        assert!(!flat.world.meshes()[1].is_smooth());
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
        // XML 自体はファイル不要（load_scene_from_str）。obj だけは obj_loader が
        // ファイルパスを要求するため実ファイルとして書き出す。
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
        let scene = load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0;
        std::fs::remove_file(&obj).ok();

        assert_eq!(scene.world.meshes().len(), 1);
        assert_eq!(scene.world.instances().len(), 1);
        assert_eq!(scene.world.meshes()[0].tris.len(), 1);
        assert_eq!(scene.mats.len(), 1);
        // 頂点 v0=(0,0,0) は translate(10,0,0) でワールド (10,0,0) になる
        let inst = scene.world.instances()[0];
        let p = inst.xform.apply_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((p - Vec3::new(10.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn constant_emitter_sets_uniform_env() {
        let scene = load(
            r#"<scene version="3.0.0">
              <emitter type="constant"><rgb name="radiance" value="0.1,0.2,0.4"/></emitter>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        let env = scene.env.expect("constant emitter should set env");
        let a = env.sample(Vec3::new(0.0, 1.0, 0.0));
        let b = env.sample(Vec3::new(1.0, 0.0, 0.0));
        // 定数なので方向によらず同じ放射輝度
        assert!((a.r() - 0.1).abs() < 1e-9 && (a.g() - 0.2).abs() < 1e-9 && (a.b() - 0.4).abs() < 1e-9);
        assert!((b.r() - 0.1).abs() < 1e-9 && (b.b() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn no_env_emitter_defaults_to_black_background() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        // Mitsuba 準拠で背景は黒（sky() フォールバックは使わない）
        let env = scene.env.expect("env should default to a black constant");
        let c = env.sample(Vec3::new(0.3, 0.8, 0.1));
        assert!(c.r() == 0.0 && c.g() == 0.0 && c.b() == 0.0);
    }

    #[test]
    fn emitter_scale_multiplies_radiance() {
        let scene = load(
            r#"<scene version="3.0.0">
              <emitter type="constant">
                <rgb name="radiance" value="1,1,1"/>
                <float name="scale" value="3"/>
              </emitter>
            </scene>"#,
        );
        let c = scene.env.unwrap().sample(Vec3::new(0.0, 1.0, 0.0));
        assert!((c.r() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn envmap_file_loads() {
        let exr = concat!(env!("CARGO_MANIFEST_DIR"), "/sample/env.exr");
        let xml = format!(
            r#"<scene version="3.0.0">
              <emitter type="envmap"><string name="filename" value="{}"/></emitter>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
            exr
        );
        let scene = load(&xml);
        let env = scene.env.expect("envmap file should load");
        assert!(env.width > 1 && env.height > 1);
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

    // ---- MTL / usemtl（T2） ----

    /// 一時ディレクトリに OBJ + MTL + テクスチャ 1 枚を作る（MTL はタブ字下げ・`\` パス・実データ風）。
    /// 面は 8 枚（A×4, B×3, 未定義 C×1、先頭 1 枚は `usemtl` 前）で、警告が面ごとに出ないことも見られる。
    fn with_mtl_dir<T>(f: impl FnOnce(&Path) -> T) -> T {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("tinypt_mtl_{}_{}", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(dir.join("textures")).unwrap();
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.save(dir.join("textures/a.png")).unwrap();
        let mut obj = String::from("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        for (name, n) in [("A", 4), ("B", 3), ("C", 1)] {
            obj.push_str(&format!("usemtl {}\n", name));
            for _ in 0..n {
                obj.push_str("f 1 2 3\n");
            }
        }
        std::fs::write(dir.join("m.obj"), obj).unwrap();
        std::fs::write(
            dir.join("m.mtl"),
            "newmtl A\n\tKd 1 1 1\n\tKs 0 0 0\n\tmap_Kd textures\\a.png\n\tmap_Ka textures\\a.png\n\tmap_d textures\\a.png\n\
             newmtl B\n\tKd 0.5 0.5 0.5\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_Kd textures\\a.png\n\td 0.5\n",
        )
        .unwrap();
        let out = f(&dir);
        std::fs::remove_dir_all(&dir).ok();
        out
    }

    fn obj_scene_xml(extra: &str) -> String {
        format!(
            r#"<scene version="3.0.0"><shape type="obj"><string name="filename" value="m.obj"/>{}</shape></scene>"#,
            extra
        )
    }

    /// `usemtl` ごとに材質が作られ、1 メッシュのまま三角形ごとの `mat_id` で引く。
    /// `usemtl` 前の面は既定、MTL に無い名前も既定。同じ画像は 1 度しか読まない。
    #[test]
    fn mtl_materials_resolve_per_triangle_and_textures_are_cached() {
        with_mtl_dir(|dir| {
            let ((scene, _), warnings) = capture_warnings(|| {
                load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap()
            });
            // 面に使われた名前だけ: 既定("")・A・B・C
            assert_eq!(scene.mats.len(), 4);
            assert_eq!(scene.world.meshes().len(), 1, "usemtl でメッシュを割ってはいけない");
            let ids: Vec<usize> = scene.world.meshes()[0].tris.iter().map(|t| t.mat_id).collect();
            assert_eq!(ids, vec![0, 1, 1, 1, 1, 2, 2, 2, 3]);
            assert!(scene.world.instances()[0].mat_override.is_none());
            // A: テクスチャ付き Lambert。B は明るい Ks かつ Ns>1 だが map_Kd を持つので、
            // テクスチャ付き Lambert が優先される（map_Kd があれば Ks/Ns は使わない。不具合修正）。
            // C・既定: 灰色 Lambert
            assert!(matches!(scene.mats[1], Material::Lambert { albedo_tex: Some(_), .. }));
            assert!(matches!(scene.mats[2], Material::Lambert { albedo_tex: Some(_), .. }), "map_Kd がある B は Lambert のはず");
            assert!(matches!(scene.mats[3], Material::Lambert { albedo_tex: None, .. }));
            // 同じ a.png を A の map_Kd / map_Ka / map_d と B の map_Kd が指しているが、読むのは 1 回（キャッシュ）
            assert_eq!(scene.textures.len(), 1);
            // 警告: map_Ka はマテリアル A に 1 回だけ。定数 d<1 はシーンで 1 回だけ。面ごとには出ない
            let n = |pat: &str| warnings.iter().filter(|w| w.contains(pat)).count();
            assert_eq!(n("map_Ka"), 1, "{:?}", warnings);
            assert_eq!(n("constant d < 1"), 1, "{:?}", warnings);
            assert_eq!(n("failed to load alpha mask"), 0, "{:?}", warnings);
            assert_eq!(n("'C' not found"), 1, "{:?}", warnings);
            // 発光マテリアルを含まない → ライト CDF は空
            assert!(scene.world.lights().is_empty());
        });
    }

    /// 2 つの材質が同じ画像を指しても 1 回しか読まない（キャッシュ）。
    #[test]
    fn same_texture_path_in_two_materials_is_loaded_once() {
        with_mtl_dir(|dir| {
            std::fs::write(
                dir.join("m.mtl"),
                "newmtl A\n\tKd 1 1 1\n\tmap_Kd textures\\a.png\nnewmtl B\n\tKd 1 1 1\n\tmap_Kd textures/a.png\n",
            )
            .unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.textures.len(), 1);
            match (&scene.mats[1], &scene.mats[2]) {
                (
                    Material::Lambert { albedo_tex: Some(a), .. },
                    Material::Lambert { albedo_tex: Some(b), .. },
                ) => assert_eq!(a, b),
                _ => panic!("A / B ともテクスチャ付き Lambert のはず"),
            }
        });
    }

    /// `<bsdf>` 指定があれば MTL は読まず、従来どおり 1 材質で全体を覆う。
    #[test]
    fn bsdf_child_overrides_mtl() {
        with_mtl_dir(|dir| {
            let xml = obj_scene_xml(r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.2,0.3,0.4"/></bsdf>"#);
            let scene = load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.mats.len(), 1);
            assert!(scene.textures.is_empty(), "MTL のテクスチャを読んではいけない");
            assert!(scene.world.meshes()[0].tris.iter().all(|t| t.mat_id == 0));
        });
    }

    /// `use_mtl=false` で MTL を無視し（`<bsdf>` 無しなら従来の既定拡散 + 警告）。
    #[test]
    fn use_mtl_false_ignores_mtl() {
        with_mtl_dir(|dir| {
            let xml = obj_scene_xml(r#"<boolean name="use_mtl" value="false"/>"#);
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0);
            assert_eq!(scene.mats.len(), 1);
            assert!(scene.textures.is_empty());
            assert!(warnings.iter().any(|w| w.contains("without bsdf")), "{:?}", warnings);
        });
    }

    /// MTL ファイルが無い / 空でも落ちず、既定の灰色拡散になる。
    #[test]
    fn missing_or_empty_mtl_falls_back_to_default() {
        with_mtl_dir(|dir| {
            std::fs::remove_file(dir.join("m.mtl")).unwrap();
            let (scene, warnings) = capture_warnings(|| {
                load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0
            });
            assert_eq!(scene.mats.len(), 4);
            assert!(scene.mats.iter().all(|m| matches!(m, Material::Lambert { albedo_tex: None, .. })));
            assert!(warnings.iter().any(|w| w.contains("failed to read mtl")), "{:?}", warnings);

            std::fs::write(dir.join("m.mtl"), "").unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.mats.len(), 4);
        });
    }

    /// `map_d` を持つ材質の三角形だけがアルファ付きになり、`<bsdf>` 上書きや map_d 無しでは付かない。
    /// 同じマスク画像は 1 度しか読まない（Arc を共有）。
    #[test]
    fn map_d_attaches_alpha_only_to_masked_material_triangles() {
        with_mtl_dir(|dir| {
            // UV を持つ OBJ にする
            let mut obj = String::from("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n");
            obj.push_str("usemtl A\nf 1/1 2/2 3/3\nusemtl B\nf 1/1 2/2 3/3\n");
            std::fs::write(dir.join("m.obj"), obj).unwrap();
            std::fs::write(dir.join("m.mtl"), "newmtl A\n\tKd 1 1 1\n\tmap_d textures\\a.png\nnewmtl B\n\tKd 1 1 1\n").unwrap();
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert!(scene.world.meshes()[0].has_alpha());
            assert!(!warnings.iter().any(|w| w.contains("alpha")), "{:?}", warnings);
            let over = obj_scene_xml(r#"<bsdf type="diffuse"/>"#);
            let scene = load_scene_from_str(&over, dir, &cfg(), (None, None)).unwrap().0;
            assert!(!scene.world.meshes()[0].has_alpha());
            // map_d 無しの材質だけなら付かない
            std::fs::write(dir.join("m.mtl"), "newmtl A\n\tKd 1 1 1\nnewmtl B\n\tKd 1 1 1\n").unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(!scene.world.meshes()[0].has_alpha());
        });
    }

    // ---- 法線マップ / バンプマップのラッパー（NM S4） ----

    fn map_scene<T>(body: &str, f: impl FnOnce(Scene, Vec<String>) -> T) -> T {
        with_png(1, 1, &[[128, 128, 255]], |dir, name| {
            let xml = format!(r#"<scene version="3.0.0">{}</scene>"#, body.replace("MAP.png", name));
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0);
            f(scene, warnings)
        })
    }

    const NMAP: &str = r#"<texture type="bitmap" name="normalmap"><string name="filename" value="MAP.png"/></texture>"#;
    const DIFF: &str = r#"<bsdf type="diffuse"/>"#;

    /// 材質表とマップ表が全 shape 経路（球・メッシュ・MTL 混在）で同じ長さに保たれ、マップの添字が正しい材質に付く。
    #[test]
    fn mat_maps_stay_in_sync_with_mats_across_shapes() {
        let body = format!(
            r#"<shape type="sphere"><bsdf type="diffuse"/></shape>
               <shape type="rectangle"><bsdf type="normalmap">{n}{d}</bsdf></shape>
               <shape type="cube"><bsdf type="twosided"><bsdf type="normalmap">{n}{d}</bsdf></bsdf></shape>
               <shape type="rectangle"><bsdf type="bumpmap"><float name="scale" value="2"/><texture type="bitmap" name="bumpmap"><string name="filename" value="MAP.png"/></texture>{d}</bsdf></shape>
               <shape type="disk"><bsdf type="diffuse"/></shape>"#,
            n = NMAP, d = DIFF
        );
        map_scene(&body, |s, w| {
            assert_eq!(s.mats.len(), 5);
            assert_eq!(s.mat_maps.len(), s.mats.len());
            let some: Vec<bool> = s.mat_maps.iter().map(|m| m.is_some()).collect();
            assert_eq!(some, vec![false, true, true, true, false]);
            assert_eq!(s.normal_maps.len(), 3);
            assert!(matches!(s.normal_maps[0], NormalMap::Tangent { .. }));
            assert!(matches!(s.normal_maps[2], NormalMap::Height { strength, .. } if strength == 2.0));
            assert!(w.iter().all(|m| m.contains("sensor")), "{:?}", w);
        });
    }

    /// マップが 1 つも無ければ `mat_maps` は空（積分器は何も引かない）。
    #[test]
    fn scenes_without_maps_have_empty_tables() {
        let s = load(r#"<scene version="3.0.0"><shape type="sphere"><bsdf type="diffuse"/></shape></scene>"#);
        assert!(s.mat_maps.is_empty() && s.normal_maps.is_empty());
    }

    /// MTL 経路（マップ無し）と混在しても表がずれない。
    #[test]
    fn mat_maps_stay_in_sync_with_mtl_materials() {
        with_mtl_dir(|dir| {
            let xml = r#"<scene version="3.0.0">
                <shape type="obj"><string name="filename" value="m.obj"/></shape>
                <shape type="rectangle"><bsdf type="normalmap"><texture type="bitmap"><string name="filename" value="textures/a.png"/></texture><bsdf type="diffuse"/></bsdf></shape>
              </scene>"#;
            let s = load_scene_from_str(xml, dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.mat_maps.len(), s.mats.len());
            assert_eq!(s.mat_maps.iter().filter(|m| m.is_some()).count(), 1);
            assert!(s.mat_maps.last().unwrap().is_some());
        });
    }

    /// ノーマルマップのテクスチャは raw（リニア）固定。`raw="false"` は警告して無視。
    /// 内側の `<bsdf>` が無ければ diffuse + 警告。二重ラップは外側を採用して警告。
    #[test]
    fn wrapper_warnings_and_raw_reading() {
        let raw_false = format!(
            r#"<shape type="rectangle"><bsdf type="normalmap"><texture type="bitmap" name="normalmap"><string name="filename" value="MAP.png"/><boolean name="raw" value="false"/></texture>{}</bsdf></shape>"#,
            DIFF
        );
        map_scene(&raw_false, |s, w| {
            assert!(w.iter().any(|m| m.contains("raw=false")), "{:?}", w);
            // (128,128,255) をリニアで読んだ値（sRGB デコードされていない）
            match &s.normal_maps[0] {
                NormalMap::Tangent { tex, .. } => {
                    let c = tex.sample((0.5, 0.5));
                    assert!((c.r() - 128.0 / 255.0).abs() < 1e-12, "sRGB で読まれている: {}", c.r());
                }
                _ => panic!(),
            }
        });
        let no_inner = format!(r#"<shape type="rectangle"><bsdf type="normalmap">{}</bsdf></shape>"#, NMAP);
        map_scene(&no_inner, |s, w| {
            assert!(w.iter().any(|m| m.contains("without an inner")), "{:?}", w);
            assert!(matches!(s.mats[0], Material::Lambert { .. }));
        });
        let nested = format!(
            r#"<shape type="rectangle"><bsdf type="normalmap">{n}<bsdf type="normalmap">{n}{d}</bsdf></bsdf></shape>"#,
            n = NMAP, d = DIFF
        );
        map_scene(&nested, |s, w| {
            assert!(w.iter().any(|m| m.contains("nested")), "{:?}", w);
            assert_eq!(s.mat_maps.len(), 1);
            // 外側のマップ（2 番目に登録されたもの）が使われる
            assert_eq!(s.mat_maps[0], Some(1));
        });
    }

    // ---- MTL の map_bump / norm（NM S5） ----

    /// UV 付きの OBJ（2 三角形: 材質 A, B）と、指定の MTL を書く。テクスチャは textures/a.png（1x1）。
    fn write_uv_obj_and_mtl(dir: &Path, mtl: &str) {
        let obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n\
                   usemtl A\nf 1/1 2/2 3/3\nusemtl B\nf 1/1 2/2 3/3\n";
        std::fs::write(dir.join("m.obj"), obj).unwrap();
        std::fs::write(dir.join("m.mtl"), mtl).unwrap();
    }

    /// `map_bump` を持つ材質だけがバンプ付きになり、強度は `bm · MTL_BUMP_K`。`unsupported` の警告は出ない。
    #[test]
    fn map_bump_attaches_only_to_its_material() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tKd 1 1 1\n\tmap_bump -bm 2 textures\\a.png\nnewmtl B\n\tKd 1 1 1\n");
            let (s, w) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert_eq!(s.mats.len(), 2);
            assert_eq!(s.mat_maps.len(), s.mats.len());
            assert_eq!((s.mat_maps[0].is_some(), s.mat_maps[1].is_some()), (true, false));
            match &s.normal_maps[s.mat_maps[0].unwrap() as usize] {
                NormalMap::Height { strength, .. } => assert!((strength - 2.0 * MTL_BUMP_K).abs() < 1e-12),
                _ => panic!("ハイトマップのはず"),
            }
            assert!(!w.iter().any(|m| m.contains("map_bump")), "{:?}", w);
        });
    }

    /// `<bsdf>` 上書きと `use_mtl=false` ではマップは付かない。
    #[test]
    fn bump_maps_are_not_attached_when_mtl_is_bypassed() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump textures\\a.png\n");
            let over = obj_scene_xml(r#"<bsdf type="diffuse"/>"#);
            let s = load_scene_from_str(&over, dir, &cfg(), (None, None)).unwrap().0;
            assert!(s.mat_maps.is_empty() && s.normal_maps.is_empty());
            let off = obj_scene_xml(r#"<boolean name="use_mtl" value="false"/>"#);
            let s = load_scene_from_str(&off, dir, &cfg(), (None, None)).unwrap().0;
            assert!(s.mat_maps.is_empty() && s.normal_maps.is_empty());
        });
    }

    /// GGX の分岐（明るい Ks かつ Ns > 1）でもマップは付く。
    #[test]
    fn ggx_materials_get_the_bump_map_too() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_bump textures\\a.png\nnewmtl B\n\tKd 1 1 1\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(matches!(s.mats[0], Material::Ggx { .. }));
            assert!(s.mat_maps[0].is_some());
        });
    }

    /// 不具合修正の回帰テスト: `map_Kd` と明るい `Ks`/`Ns` を両方持つ材質は、テクスチャ付き Lambert になる
    /// （Ggx へ早期 return して拡散テクスチャを捨ててはいけない。Sponza の floor/arch/chain/vase_hanging）。
    /// `map_Kd` を持たない明るい `Ks`/`Ns` は従来どおり Ggx。どちらの経路でもバンプマップは付く。
    #[test]
    fn map_kd_takes_priority_over_the_ggx_branch() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(
                dir,
                "newmtl A\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_Kd textures\\a.png\n\tmap_bump textures\\a.png\n\
                 newmtl B\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_bump textures\\a.png\n",
            );
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(matches!(s.mats[0], Material::Lambert { albedo_tex: Some(_), .. }), "map_Kd 付き材質は Lambert のはず");
            assert!(matches!(s.mats[1], Material::Ggx { .. }), "map_Kd の無い材質は従来どおり Ggx のはず");
            assert!(s.mat_maps[0].is_some() && s.mat_maps[1].is_some(), "どちらの経路でもバンプマップが付くはず");
        });
    }

    /// `norm` があれば `norm`（タンジェント）を採用し、`map_bump` 併存の警告は材質ごとに 1 回。
    #[test]
    fn norm_wins_over_map_bump_with_one_warning_per_material() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(
                dir,
                "newmtl A\n\tnorm textures\\a.png\n\tmap_bump textures\\a.png\nnewmtl B\n\tnorm textures\\a.png\n\tmap_bump textures\\a.png\n",
            );
            let (s, w) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert!(s.normal_maps.iter().all(|m| matches!(m, NormalMap::Tangent { .. })));
            assert_eq!(w.iter().filter(|m| m.contains("both norm and map_bump")).count(), 2, "{:?}", w);
        });
    }

    /// 同じ画像・同じ種別・同じ強度は 1 度だけ読む。強度が違えば別登録、種別が違えば別登録。
    #[test]
    fn map_cache_is_keyed_by_path_kind_and_strength() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.normal_maps.len(), 1);
            assert_eq!(s.mat_maps[0], s.mat_maps[1]);
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump -bm 3 textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.normal_maps.len(), 2, "強度違いが同じ登録を共有した");
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tnorm textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.normal_maps.len(), 2, "種別違いが同じ登録を共有した");
        });
    }
}

