//! Mitsuba XML（サブセット）シーンローダー。
//!
//! [Mitsuba レンダラー](https://www.mitsuba-renderer.org/) の XML シーン記述の
//! サブセットを読み込み、[`Scene`] を構築する。採用理由とマッピング方針は
//! `docs/adr/0002-mitsuba-xml-scene-format.md` を参照。
//!
//! ## 対応要素
//! - `sensor type="perspective"`: `fov` / `fov_axis` / `to_world`(`lookat`) / `aperture_radius` / `focus_distance` / `shutter_open` / `shutter_close` / `shutter_angle`（独自拡張、度。`close = open + angle/360`）
//! - `shape type="sphere"`: `center` / `radius`
//! - `shape type="sdf"`: 直下の `<sdf>` 木（`sphere` / `box` / `torus` / `cylinder` / `capsule` と `union` / `intersection` / `difference` / `smooth_*`）+ `to_world`。`to_world_end`（モーションブラー）対応。`displace`（子の面を `perlin` / `fbm` / `turbulence` のノイズでずらす。リプシッツ定数でマーチの歩幅を割る）。各プリミティブは `center_end`（カプセルは `a_end` / `b_end`）で個別に動かせる（時刻で線形補間）。スフィアトレーシング、光源にはならない
//! - `shape type="obj"`: `filename`（XML 相対）+ `to_world`（translate/rotate/scale/matrix）
//! - `bsdf`: `diffuse` / `conductor` / `roughconductor`(ggx) / `dielectric` / `thindielectric`・`roughdielectric`(dielectric 扱い) / `twosided`(unwrap)。未知の型は警告して diffuse
//! - `emitter type="area"`: `radiance`（shape に付随）
//! - `emitter type="envmap"`(filename) / `constant`(radiance): 環境マップ。`scale` 対応
//! - `film`(width/height) / `sampler`(sample_count) / `integrator`(max_depth/rr_depth、Mitsuba と同じ意味、max_depth=-1 は無制限): RenderConfig へ反映
//! - `film` の `<rfilter type="box|tent|gaussian|mitchell">`: 画素の再構成フィルタ（既定 gaussian、Mitsuba と同じ。フィルタ重点サンプリングで実装、`filter.rs` 参照）
//!
//! ## 方針
//! - 色: `<rgb>` はリニア、`<srgb>` は sRGB（ガンマ展開）。
//! - 未対応の要素・型は警告してスキップ／フォールバック（寛容）。必須フィールド欠落のみ既定値。
//! - スペクトルは扱わず RGB トリプルとして読む。
//! - 環境 emitter が無ければ背景は黒（Mitsuba 準拠）。組み込みデフォルトシーンの
//!   手続き的な `sky()` フォールバックは適用しない。
//!
//! サブモジュール: `xml`（要素木）/ `sensor` / `shape`（SDF 含む）/ `obj_mtl` / `parametric`
//! （組み込みメッシュ）/ `xform`（`transform` 要素）/ `bsdf`（bsdf・texture・式）。

mod bsdf;
mod obj_mtl;
mod parametric;
mod sensor;
mod shape;
#[cfg(test)]
mod tests;
mod xform;
mod xml;

use std::io;
use std::path::{Path, PathBuf};


use crate::config::RenderConfig;
use crate::env::EnvMap;
use crate::filter::PixelFilter;
use crate::geometry::Aabb;
use crate::material::Material;
use crate::math::{Color, Vec3};
use crate::medium::Medium;
use crate::noise::NoiseTexture;
use crate::normal_map::NormalMap;
use crate::ray::Camera;
use crate::scene::Scene;
use crate::shader::{ShaderSet, ValueNode};
use crate::texture::Texture;
use crate::world::{DeltaLight, World};


use obj_mtl::{resolve_path, Extra, MtlState};
use sensor::parse_sensor;
use shape::parse_shape;
use xml::{parse_tree, Element};

#[cfg(test)]
thread_local! {
    /// テスト中に `warn` が出したメッセージを記録するバッファ（[`capture_warnings`] が有効化する）。
    /// 警告は stderr に出るだけなので、そのままでは「警告を出すこと」をテストできない。
    static CAPTURED_WARNINGS: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

thread_local! {
    static OBJ_PARSE_TIME: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
    /// 読み込み中の `<texture type="noise">`（`parse_leaf_bsdf` が積み、`Scene::noises` になる。引数を通さないための thread_local）
    static NOISES: std::cell::RefCell<Vec<NoiseTexture>> = const { std::cell::RefCell::new(Vec::new()) };
    /// 読み込み中の式のアリーナ（`ShaderSet::values` になる。ノイズと同じく引数を通さないための thread_local）
    static VALUES: std::cell::RefCell<Vec<ValueNode>> = const { std::cell::RefCell::new(Vec::new()) };
    static TEXTURE_LOAD_TIME: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

fn timed_obj<T>(f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let v = f();
    OBJ_PARSE_TIME.with(|c| c.set(c.get() + t0.elapsed()));
    v
}

fn timed_texture<T>(f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let v = f();
    TEXTURE_LOAD_TIME.with(|c| c.set(c.get() + t0.elapsed()));
    v
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

#[derive(Default, Clone, Copy, Debug)]
pub struct SceneSettings {
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub spp: Option<usize>,
    pub max_depth: Option<usize>,
    pub rr_depth: Option<usize>,
    pub filter: Option<PixelFilter>,
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
        if let Some(f) = self.filter {
            config.filter = f;
        }
    }
}

fn parse_rfilter(el: &Element) -> Option<PixelFilter> {
    let typ = el.typ();
    match typ {
        "box" => Some(PixelFilter::Box),
        "tent" => Some(PixelFilter::Tent),
        "gaussian" => {
            let stddev = el.float("stddev").unwrap_or(0.5);
            PixelFilter::gaussian(stddev).or_else(|| {
                warn(&format!("rfilter gaussian stddev must be positive and finite (got {stddev}); ignored"));
                None
            })
        }
        "mitchell" => {
            let b = el.float("B").unwrap_or(1.0 / 3.0);
            let c = el.float("C").unwrap_or(1.0 / 3.0);
            PixelFilter::mitchell(b, c).or_else(|| {
                warn(&format!("rfilter mitchell B/C must be finite (got B={b}, C={c}); ignored"));
                None
            })
        }
        other => {
            warn(&format!("unsupported rfilter type '{other}' (expected box | tent | gaussian | mitchell); ignored"));
            None
        }
    }
}

pub fn load_scene_from_str(
    xml: &str,
    base_dir: &Path,
    base_config: &RenderConfig,
    forced_resolution: (Option<usize>, Option<usize>),
) -> io::Result<(Scene, SceneSettings)> {
    OBJ_PARSE_TIME.with(|c| c.set(std::time::Duration::ZERO));
    TEXTURE_LOAD_TIME.with(|c| c.set(std::time::Duration::ZERO));
    NOISES.with(|n| n.borrow_mut().clear());
    VALUES.with(|v| v.borrow_mut().clear());
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
                if let Some(rf) = child.child_tag("rfilter") {
                    settings.filter = parse_rfilter(rf);
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
    let mut mat_maps: Vec<Extra> = Vec::new();
    let mut cam: Option<Camera> = None;
    let mut env: Option<EnvMap> = None;
    let mut medium: Option<Medium> = None;
    let mut delta_lights: Vec<DeltaLight> = Vec::new();
    let mut seen_medium = false;

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
            // シーン直下の emitter は `type` で振り分ける: point / directional / spot はデルタ光源（複数置ける）、
            // それ以外（envmap / constant / 未知の型）は従来どおり環境マップ（1 個。未知の型は警告）
            "emitter" => match child.typ() {
                "point" | "directional" | "spot" => {
                    if let Some(l) = parse_delta_emitter(child) {
                        delta_lights.push(l);
                    }
                }
                _ => {
                    if let Some(e) = parse_scene_emitter(child, base_dir) {
                        env = Some(e);
                    }
                }
            },
            // シーン直下の medium は空間全体（または bounds）に広がる一様媒質。最初の 1 つだけ使う
            "medium" => {
                if seen_medium {
                    warn("more than one <medium>; only the first is used");
                } else {
                    seen_medium = true;
                    medium = parse_medium(child);
                }
            }
            // レンダリング設定ブロックは無視（このレンダラーは CLI で制御する）
            "integrator" | "sampler" | "film" | "default" => {}
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

    for l in delta_lights {
        world.add_delta_light(l);
    }
    // 動くインスタンスの掃過ボリュームは、シャッター区間（センサーの shutter_open / shutter_close）で作る
    let (shutter_open, shutter_close) = cam.shutter();
    world.set_shutter(shutter_open, shutter_close);
    world.build_lights(&mats);
    let load_stats = crate::scene::LoadStats {
        obj_parse: OBJ_PARSE_TIME.with(|c| c.get()),
        mesh_build: world.mesh_build_time(),
        texture_load: TEXTURE_LOAD_TIME.with(|c| c.get()),
    };
    // 材質・式・テクスチャ・ノイズ・法線マップを 1 つの `ShaderSet` にまとめる
    debug_assert_eq!(mats.len(), mat_maps.len());
    let mut shaders = ShaderSet::default();
    shaders.textures = textures;
    shaders.noises = NOISES.with(|n| std::mem::take(&mut *n.borrow_mut()));
    shaders.normal_maps = normal_maps;
    shaders.set_values(VALUES.with(|v| std::mem::take(&mut *v.borrow_mut())));
    for (m, e) in mats.iter().zip(&mat_maps) {
        shaders.push_shader(*m, e.exprs, &e.normals);
    }
    Ok((Scene { cam, world, shaders, env, medium, load_stats }, settings))
}

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

pub fn referenced_files(xml: &str, base_dir: &Path) -> io::Result<Vec<PathBuf>> {
    fn walk(el: &Element, base_dir: &Path, out: &mut Vec<PathBuf>) {
        if el.tag == "string"
            && matches!(el.attr("name"), Some("filename") | Some("filename_end"))
            && let Some(v) = el.attr("value")
        {
            out.push(resolve_path(base_dir, v));
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

fn color_or_float(el: &Element, name: &str) -> Option<Color> {
    el.color(name).or_else(|| el.float(name).map(|v| Color::new(v, v, v)))
}

fn parse_medium(el: &Element) -> Option<Medium> {
    if el.typ() != "homogeneous" {
        warn(&format!("unsupported medium type '{}', ignored", el.typ()));
        return None;
    }
    let mut sigma_t = color_or_float(el, "sigma_t").unwrap_or(Color::new(1.0, 1.0, 1.0));
    if sigma_t.r() < 0.0 || sigma_t.g() < 0.0 || sigma_t.b() < 0.0 {
        warn("medium sigma_t must be >= 0; negative components set to 0");
        sigma_t = Color::new(sigma_t.r().max(0.0), sigma_t.g().max(0.0), sigma_t.b().max(0.0));
    }
    let mut albedo = color_or_float(el, "albedo").unwrap_or(Color::new(1.0, 1.0, 1.0));
    if [albedo.r(), albedo.g(), albedo.b()].iter().any(|&a| !(0.0..=1.0).contains(&a)) {
        warn("medium albedo must be within [0, 1]; clamped");
        albedo = albedo.clamp01();
    }
    let mut g = 0.0;
    if let Some(ph) = el.child_tag("phase") {
        match ph.typ() {
            "hg" => {
                g = ph.float("g").unwrap_or(0.0);
                if !(g > -1.0 && g < 1.0) {
                    warn(&format!("phase g = {} must be within (-1, 1); clamped to +-0.99", g));
                    g = if g.is_nan() { 0.0 } else { g.clamp(-0.99, 0.99) };
                }
            }
            "isotropic" => {}
            other => warn(&format!("unsupported phase type '{}'; using isotropic", other)),
        }
    }
    let bounds = match (el.point("bounds_min"), el.point("bounds_max")) {
        (Some(lo), Some(hi)) => {
            if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                warn("medium bounds_min exceeds bounds_max; treating the medium as unbounded");
                None
            } else {
                Some(Aabb { min: lo, max: hi })
            }
        }
        (None, None) => None,
        _ => {
            warn("medium needs both bounds_min and bounds_max; treating the medium as unbounded");
            None
        }
    };
    Some(Medium { sigma_t, albedo, g, bounds })
}

fn shape_emitter(el: &Element) -> Option<&Element> {
    el.children.iter().find(|c| c.tag == "emitter" && !matches!(c.typ(), "point" | "directional" | "spot"))
}

#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(d.len() > 0.0)` also catches a NaN length; `d.len() <= 0.0` would not
fn parse_delta_emitter(el: &Element) -> Option<DeltaLight> {
    let kind = el.typ();
    let radiance_name = if kind == "directional" { "irradiance" } else { "intensity" };
    let mut value = color_or_float(el, radiance_name).unwrap_or(Color::new(1.0, 1.0, 1.0));
    if value.r() < 0.0 || value.g() < 0.0 || value.b() < 0.0 {
        warn(&format!("{} emitter {} must be >= 0; negative components set to 0", kind, radiance_name));
        value = Color::new(value.r().max(0.0), value.g().max(0.0), value.b().max(0.0));
    }
    let direction = || -> Option<Vec3> {
        let d = el.vector("direction").unwrap_or(Vec3::new(0.0, -1.0, 0.0));
        if !(d.len() > 0.0) || !d.len().is_finite() {
            warn(&format!("{} emitter direction is zero or not finite; light ignored", kind));
            None
        } else {
            Some(d.norm())
        }
    };
    let position = || el.point("position").unwrap_or(Vec3::new(0.0, 0.0, 0.0));
    match kind {
        "point" => Some(DeltaLight::Point { position: position(), intensity: value }),
        "directional" => Some(DeltaLight::Directional { direction: direction()?, irradiance: value }),
        _ => {
            let direction = direction()?;
            let mut cutoff = el.float("cutoff_angle").unwrap_or(20.0);
            if !(cutoff > 0.0 && cutoff <= 90.0) {
                warn(&format!("spot cutoff_angle {} is outside (0, 90]; using 20", cutoff));
                cutoff = 20.0;
            }
            // Mitsuba の既定: beam_width = cutoff_angle × 3/4
            let mut beam = el.float("beam_width").unwrap_or(cutoff * 0.75);
            if !(beam > 0.0 && beam <= cutoff) {
                warn(&format!("spot beam_width {} must be within (0, cutoff_angle]; using cutoff_angle", beam));
                beam = cutoff;
            }
            Some(DeltaLight::Spot {
                position: position(),
                direction,
                intensity: value,
                cutoff_angle: cutoff.to_radians(),
                beam_width: beam.to_radians(),
            })
        }
    }
}

fn parse_scene_emitter(el: &Element, base_dir: &Path) -> Option<EnvMap> {
    if el.child_tag("transform").is_some() {
        warn("envmap to_world rotation is unsupported; ignored");
    }
    let scale = el.float("scale").unwrap_or(1.0);
    match el.typ() {
        "envmap" => {
            let filename = el.string("filename")?;
            let resolved = resolve_path(base_dir, filename);
            match timed_texture(|| EnvMap::from_hdr(resolved.to_string_lossy().as_ref())) {
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
