//! `transform` 要素のパース。



use crate::math::Vec3;
use crate::transform::Transform;

use super::xml::{parse_f64, xyz, Element};
use super::warn;

pub(super) fn parse_transform(el: &Element) -> Transform {
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
