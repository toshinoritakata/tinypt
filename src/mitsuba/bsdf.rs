//! `bsdf`/`texture`/ノイズ/式のパース。

use std::path::{Path, PathBuf};


use crate::material::{Material, TexId};
use crate::math::{Color, Vec3};
use crate::noise::{NoiseTexture, Pattern};
use crate::normal_map::{HeightMap, MapId, NormalMap};
use crate::shader::{Exprs, TexRef, ValueId, ValueNode};
use crate::texture::{Texture, Wrap};

use super::xml::{parse_vec3, Element};
use super::{timed_texture, warn, NOISES, VALUES};
use super::obj_mtl::{resolve_path, Extra};

pub(super) fn parse_emitter(el: &Element) -> Material {
    let emit = el.color("radiance").unwrap_or(Color::new(1.0, 1.0, 1.0));
    if el.typ() != "area" {
        warn(&format!("emitter type '{}' treated as area light", el.typ()));
    }
    Material::DiffuseLight { emit }
}

fn parse_texture(el: &Element, base_dir: &Path, textures: &mut Vec<Texture>) -> Option<TexId> {
    let (resolved, wrap) = bitmap_source(el, base_dir)?;
    // 色テクスチャは sRGB。`raw=true` はデータテクスチャ（リニア）
    let srgb = !el.boolean_or("raw", false);
    match timed_texture(|| Texture::load(resolved.to_string_lossy().as_ref(), srgb, wrap)) {
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

fn parse_noise_texture(el: &Element) -> Option<TexRef> {
    let pattern = match el.string("pattern") {
        None => Pattern::Fbm,
        Some(s) => Pattern::parse(s).unwrap_or_else(|| {
            warn(&format!("unknown noise pattern '{s}'; using fbm"));
            Pattern::Fbm
        }),
    };
    let raw = NoiseTexture {
        pattern,
        scale: el.float("scale").unwrap_or(1.0),
        octaves: el.int("octaves").map_or(4, |o| o.clamp(0, 1000) as u32),
        lacunarity: el.float("lacunarity").unwrap_or(2.0),
        gain: el.float("gain").unwrap_or(0.5),
        strength: el.float("strength").unwrap_or(1.0),
        color0: el.color("color0").unwrap_or(Color::new(0.0, 0.0, 0.0)),
        color1: el.color("color1").unwrap_or(Color::new(1.0, 1.0, 1.0)),
        local: match el.string("space") {
            None | Some("local") => true,
            Some("world") => false,
            Some(s) => {
                warn(&format!("unknown noise space '{s}'; using local"));
                true
            }
        },
        offset: el.point("offset").unwrap_or(Vec3::new(0.0, 0.0, 0.0)),
    };
    let fixed = raw.sanitized();
    if (fixed.scale, fixed.octaves, fixed.lacunarity, fixed.gain, fixed.strength) != (raw.scale, raw.octaves, raw.lacunarity, raw.gain, raw.strength) {
        warn("noise texture parameter out of range (scale must be in (0, 1e6], octaves 1..10, lacunarity 1..8, gain 0..1); clamped");
    }
    NOISES.with(|n| {
        let mut n = n.borrow_mut();
        n.push(fixed);
        Some(TexRef::Noise((n.len() - 1) as u32))
    })
}

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

fn wrapper_texture<'a>(el: &'a Element, name: &str) -> Option<&'a Element> {
    el.prop("texture", name)
        .or_else(|| el.children.iter().find(|c| c.tag == "texture" && c.attr("name").is_none()))
}

pub(super) fn parse_bsdf(
    el: &Element,
    base_dir: &Path,
    textures: &mut Vec<Texture>,
    normal_maps: &mut Vec<NormalMap>,
) -> (Material, Extra) {
    let default_mat = || Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) };
    match el.typ() {
        // 両面 BSDF はラッパーなので内側を展開（内側にマップのラッパーがあればそのまま素通し）
        "twosided" => el
            .child_tag("bsdf")
            .map(|c| parse_bsdf(c, base_dir, textures, normal_maps))
            .unwrap_or((default_mat(), Extra::default())),
        // Mitsuba 準拠: `<bsdf type="normalmap">` は子のテクスチャ（タンジェント空間ノーマルマップ）を
        // 内側の `<bsdf>` に適用する。テクスチャは**リニア（raw）固定**で読む
        "normalmap" | "bumpmap" => {
            let is_normal = el.typ() == "normalmap";
            let (mat, inner) = match el.child_tag("bsdf") {
                Some(inner) => parse_bsdf(inner, base_dir, textures, normal_maps),
                None => {
                    warn(&format!("{} without an inner <bsdf>; defaulting to diffuse", el.typ()));
                    (default_mat(), Extra::default())
                }
            };
            let tex_el = wrapper_texture(el, if is_normal { "normalmap" } else { "bumpmap" });
            let map = tex_el.and_then(|t| {
                let (path, wrap) = bitmap_source(t, base_dir)?;
                let path_s = path.to_string_lossy();
                if is_normal {
                    if !t.boolean_or("raw", true) {
                        warn("normalmap texture with raw=false: ignoring it and reading the map as linear (raw)");
                    }
                    match timed_texture(|| Texture::load(path_s.as_ref(), false, wrap)) {
                        Ok(tex) => Some(NormalMap::Tangent { tex, scale: 1.0 }),
                        Err(e) => {
                            warn(&format!("failed to load normal map '{}': {}; ignored", path.display(), e));
                            None
                        }
                    }
                } else {
                    match timed_texture(|| HeightMap::load(path_s.as_ref(), wrap)) {
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
                    // 外側（先に書いた方）を先に適用し、内側のマップがその後に続く
                    let mut normals = vec![(normal_maps.len() - 1) as MapId];
                    normals.extend(inner.normals.iter().copied());
                    (mat, Extra { exprs: inner.exprs, normals })
                }
                None => (mat, inner),
            }
        }
        _ => {
            let (mat, exprs) = parse_leaf_bsdf(el, base_dir, textures);
            (mat, Extra { exprs, normals: Vec::new() })
        }
    }
}

fn parse_leaf_bsdf(el: &Element, base_dir: &Path, textures: &mut Vec<Texture>) -> (Material, Exprs) {
    let mut ex = Exprs::default();
    let white = Color::new(1.0, 1.0, 1.0);
    let mat = match el.typ() {
        "diffuse" => {
            // `reflectance` はテクスチャ（式）か定数色。テクスチャがある場合、定数色は色の倍率になる
            // （両方あれば掛け合わせる。片方だけなら他方は白 = 1 倍）。
            let root = param_expr(el, "reflectance", base_dir, textures);
            let default = if root.is_some() { white } else { Color::new(0.5, 0.5, 0.5) };
            let albedo = el.color("reflectance").unwrap_or(default);
            ex.albedo = root.map(|r| mul_const(albedo, r));
            Material::Lambert { albedo }
        }
        "conductor" => {
            let root = param_expr(el, "specular_reflectance", base_dir, textures);
            let albedo = el.color("specular_reflectance").unwrap_or(white);
            ex.albedo = root.map(|r| mul_const(albedo, r));
            Material::Metal { albedo }
        }
        "roughconductor" => {
            let dist = el.string("distribution").unwrap_or("ggx");
            if dist != "ggx" {
                warn(&format!("roughconductor distribution '{}' unsupported; using ggx", dist));
            }
            let root = param_expr(el, "specular_reflectance", base_dir, textures);
            let albedo = el.color("specular_reflectance").unwrap_or(white);
            ex.albedo = root.map(|r| mul_const(albedo, r));
            // 粗さもテクスチャ（式）にできる（Mitsuba の roughconductor と同じ。式の第 1 成分が alpha）
            ex.alpha = param_expr(el, "alpha", base_dir, textures);
            Material::Ggx { albedo, alpha: el.float("alpha").unwrap_or(0.1) }
        }
        "dielectric" | "thindielectric" | "roughdielectric" => {
            let int_ior = el.float("int_ior").unwrap_or(1.5);
            let ext_ior = el.float("ext_ior").unwrap_or(1.0);
            // absorption は独自拡張（標準 Mitsuba は medium で表現）
            let absorption = el.color("absorption").unwrap_or(Color::new(0.0, 0.0, 0.0));
            // 屈折率と吸収もテクスチャ（式）にできる。屈折率は int_ior / ext_ior なので、式の第 1 成分を ext_ior で割る
            ex.ior = param_expr(el, "int_ior", base_dir, textures).map(|r| {
                let k = add_value(ValueNode::Const(Color::new(1.0 / ext_ior, 1.0 / ext_ior, 1.0 / ext_ior)));
                add_value(ValueNode::Mul(r, k))
            });
            let abs_root = param_expr(el, "absorption", base_dir, textures);
            ex.absorption = abs_root.map(|r| mul_const(el.color("absorption").unwrap_or(white), r));
            Material::Dielectric { ior: int_ior / ext_ior, absorption }
        }
        other => {
            warn(&format!("unsupported bsdf type '{}'; defaulting to diffuse", other));
            Material::Lambert {
                albedo: el.color("reflectance").unwrap_or(Color::new(0.5, 0.5, 0.5)),
            }
        }
    };
    (mat, ex)
}

pub(super) fn add_value(node: ValueNode) -> ValueId {
    VALUES.with(|v| {
        let mut v = v.borrow_mut();
        v.push(node);
        (v.len() - 1) as ValueId
    })
}

pub(super) fn mul_const(factor: Color, root: ValueId) -> ValueId {
    let c = add_value(ValueNode::Const(factor));
    add_value(ValueNode::Mul(c, root))
}

fn param_expr(el: &Element, name: &str, base_dir: &Path, textures: &mut Vec<Texture>) -> Option<ValueId> {
    el.prop("texture", name).and_then(|t| parse_expr(t, base_dir, textures, 0))
}

const MAX_XML_EXPR_DEPTH: u32 = 16;

fn parse_expr(el: &Element, base_dir: &Path, textures: &mut Vec<Texture>, depth: u32) -> Option<ValueId> {
    if depth > MAX_XML_EXPR_DEPTH {
        warn("texture expression nested too deeply; ignored");
        return None;
    }
    match el.typ() {
        "bitmap" => parse_texture(el, base_dir, textures).map(|id| add_value(ValueNode::Texture(id))),
        "noise" => parse_noise_texture(el).map(|r| match r {
            TexRef::Noise(id) => add_value(ValueNode::Noise(id)),
            TexRef::Image(id) => add_value(ValueNode::Texture(id)),
        }),
        op @ ("mul" | "add" | "mix") => {
            // 子は `<texture>`（入れ子の式）か、`<rgb>` / `<srgb>`（定数の色）。現れた順に並べる
            let kids: Vec<ValueId> = el
                .children
                .iter()
                .filter_map(|c| match c.tag.as_str() {
                    "texture" => parse_expr(c, base_dir, textures, depth + 1),
                    "rgb" | "srgb" => {
                        let v = parse_vec3(c.attr("value")?)?;
                        let col = if c.tag == "srgb" { Color::from_srgb(v.x, v.y, v.z) } else { Color::new(v.x, v.y, v.z) };
                        Some(add_value(ValueNode::Const(col)))
                    }
                    _ => None,
                })
                .collect();
            let need = 2;
            if kids.len() < need {
                warn(&format!("texture '{op}' needs at least {need} child textures (got {}); {}", kids.len(), if kids.is_empty() { "ignored" } else { "using the one it has" }));
                return kids.first().copied();
            }
            match op {
                "mul" => Some(kids[1..].iter().fold(kids[0], |acc, &k| add_value(ValueNode::Mul(acc, k)))),
                "add" => Some(kids[1..].iter().fold(kids[0], |acc, &k| add_value(ValueNode::Add(acc, k)))),
                _ => {
                    if kids.len() > 3 {
                        warn("texture 'mix' takes 2 textures and an optional third for the weight; the extra ones are ignored");
                    }
                    let t = match kids.get(2) {
                        Some(&t) => t,
                        None => {
                            let w = el.float("weight").unwrap_or(0.5);
                            add_value(ValueNode::Const(Color::new(w, w, w)))
                        }
                    };
                    Some(add_value(ValueNode::Mix(kids[0], kids[1], t)))
                }
            }
        }
        other => {
            warn(&format!("unsupported texture type '{other}', ignored"));
            None
        }
    }
}
