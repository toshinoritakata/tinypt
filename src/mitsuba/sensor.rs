//! `sensor` 要素のパース: `Camera` の構築。



use crate::math::Vec3;
use crate::ray::Camera;

use super::xml::{parse_vec3, Element};
use super::warn;

pub(super) fn parse_sensor(el: &Element, aspect: f64) -> Camera {
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

    // to_world は名前で選ぶ（`to_world_end` を先に書いても取り違えない）
    let sensor_transform = |name_ok: fn(Option<&str>) -> bool| {
        el.children.iter().find(|c| c.tag == "transform" && name_ok(c.attr("name")))
    };
    let (eye, target, up) = sensor_transform(|n| matches!(n, None | Some("to_world")))
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
    let mut cam = Camera::look_at_dof(eye, target, up, vfov, aspect, focus, 2.0 * aperture_radius);
    // シャッター時刻（Mitsuba 準拠、既定 0 と 1）。頂点モーションの鍵は time = 0 と 1 なので [0, 1] に収める。
    // open > close は入れ替える。open == close はブラー無し（時刻固定）で有効な指定
    let mut open = el.float("shutter_open").unwrap_or(0.0);
    let mut close = el.float("shutter_close").unwrap_or(1.0);
    // 独自拡張: `shutter_angle`（度）。時刻 0..1 が 1 フレームなので、フィルムのシャッター角がそのまま
    // `close = open + angle / 360`（180° = 標準の半分、360° = 既定）になる。`shutter_close` より優先する。
    // 有限で [0, 360] でないものは警告して従来どおり（`shutter_close` か既定）にする
    if el.prop("float", "shutter_angle").is_some() {
        match el.float("shutter_angle") {
            Some(a) if a.is_finite() && (0.0..=360.0).contains(&a) => {
                if el.prop("float", "shutter_close").is_some() {
                    warn("shutter_close is ignored because shutter_angle is given");
                }
                close = open + a / 360.0;
            }
            _ => warn("shutter_angle must be a finite number of degrees in [0, 360]; ignored"),
        }
    }
    if !(open.is_finite() && close.is_finite()) {
        warn("shutter_open / shutter_close must be finite; using 0 and 1");
        (open, close) = (0.0, 1.0);
    }
    if open > close {
        warn(&format!("shutter_open {} > shutter_close {}; swapped", open, close));
        std::mem::swap(&mut open, &mut close);
    }
    if open < 0.0 || close > 1.0 {
        warn(&format!("shutter [{}, {}] is outside [0, 1]; clamped (motion keyframes are at time 0 and 1)", open, close));
        open = open.clamp(0.0, 1.0);
        close = close.clamp(0.0, 1.0);
    }
    cam.set_shutter(open, close);
    // 独自拡張: シャッター閉じ時点のカメラ姿勢（`<lookat>` のみ）。補間できなければ警告して静止のまま
    if let Some(end_el) = sensor_transform(|n| n == Some("to_world_end")) {
        match end_el.child_tag("lookat").and_then(parse_lookat) {
            Some((e, t, u)) => {
                if !cam.set_end_pose(e, t, u) {
                    warn("sensor to_world_end is degenerate (e.g. `up` parallel to the view direction); the camera stays static");
                }
            }
            None => warn("sensor to_world_end needs a <lookat>; the camera stays static"),
        }
    }
    cam
}

fn parse_lookat(el: &Element) -> Option<(Vec3, Vec3, Vec3)> {
    let origin = parse_vec3(el.attr("origin")?)?;
    let target = parse_vec3(el.attr("target")?)?;
    let up = el
        .attr("up")
        .and_then(parse_vec3)
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    Some((origin, target, up))
}
