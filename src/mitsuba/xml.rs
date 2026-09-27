//! XML パース: 要素木（`Element`）と生の `quick_xml` イベントからの構築。

use std::collections::HashMap;
use std::io;

use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;

use crate::math::{Color, Vec3};

use super::{err, warn};

pub(super) struct Element {
    pub(super) tag: String,
    attrs: HashMap<String, String>,
    pub(super) children: Vec<Element>,
}

impl Element {
    pub(super) fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).map(|s| s.as_str())
    }

    /// `type` 属性（Mitsuba のプラグイン種別）。
    pub(super) fn typ(&self) -> &str {
        self.attr("type").unwrap_or("")
    }

    /// `name` 属性が一致する子プロパティ要素を返す。
    pub(super) fn prop(&self, tag: &str, name: &str) -> Option<&Element> {
        self.children
            .iter()
            .find(|c| c.tag == tag && c.attr("name") == Some(name))
    }

    /// 最初の指定タグ子要素を返す。
    pub(super) fn child_tag(&self, tag: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.tag == tag)
    }

    pub(super) fn float(&self, name: &str) -> Option<f64> {
        self.prop("float", name)?.attr("value")?.trim().parse().ok()
    }

    pub(super) fn int(&self, name: &str) -> Option<usize> {
        self.prop("integer", name)?.attr("value")?.trim().parse().ok()
    }

    /// 負の値も読める整数プロパティ（`max_depth = -1` など）。
    pub(super) fn int_signed(&self, name: &str) -> Option<i64> {
        self.prop("integer", name)?.attr("value")?.trim().parse().ok()
    }

    pub(super) fn string(&self, name: &str) -> Option<&str> {
        self.prop("string", name)?.attr("value")
    }

    /// `<boolean name="..." value="true|false"/>` を既定値つきで読む。
    ///
    /// Mitsuba の boolean は `true` / `false` のみ（`1` / `yes` / `TRUE` は不正）。
    /// 要素はあるのに値が解釈できない場合は**警告して既定値にフォールバック**する。
    /// 黙って既定値にすると `value="1"` と書いた人が逆の挙動を静かに得てしまうため
    /// （README の「未対応の要素・型・属性は警告してスキップ」に合わせる）。
    pub(super) fn boolean_or(&self, name: &str, default: bool) -> bool {
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
    pub(super) fn point(&self, name: &str) -> Option<Vec3> {
        let e = self.prop("point", name)?;
        if let (Some(x), Some(y), Some(z)) = (e.attr("x"), e.attr("y"), e.attr("z")) {
            Some(Vec3::new(parse_f64(x)?, parse_f64(y)?, parse_f64(z)?))
        } else {
            parse_vec3(e.attr("value")?)
        }
    }

    /// `vector` プロパティ（`x`/`y`/`z` 属性または `value="x,y,z"`）。
    pub(super) fn vector(&self, name: &str) -> Option<Vec3> {
        let e = self.prop("vector", name)?;
        if let (Some(x), Some(y), Some(z)) = (e.attr("x"), e.attr("y"), e.attr("z")) {
            Some(Vec3::new(parse_f64(x)?, parse_f64(y)?, parse_f64(z)?))
        } else {
            parse_vec3(e.attr("value")?)
        }
    }

    /// `rgb`（リニア）または `srgb`（ガンマ展開）プロパティを `Color` として読む。
    pub(super) fn color(&self, name: &str) -> Option<Color> {
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

pub(super) fn parse_f64(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}

pub(super) fn parse_vec3(s: &str) -> Option<Vec3> {
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

pub(super) fn parse_tree(xml: &str) -> io::Result<Element> {
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

pub(super) fn xyz(el: &Element, default: f64) -> Vec3 {
    let g = |k: &str| el.attr(k).and_then(parse_f64).unwrap_or(default);
    Vec3::new(g("x"), g("y"), g("z"))
}
