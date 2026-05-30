//! 色空間変換ヘルパー（リニア sRGB/Rec.709 ↔ ACEScg）。
//!
//! EXR 出力時にリニア sRGB（レンダラー内部色空間）を ACEScg に変換する。
//! 変換行列は以下の手順で導出:
//! 1. sRGB/Rec.709 原色 → XYZ 変換行列を構築
//! 2. ACEScg 原色 → XYZ 変換行列を構築
//! 3. Bradford 色順応で D65 → D60 白色点を変換
//! 4. sRGB→XYZ→色順応→ACEScg の合成行列を `OnceLock` でキャッシュ

use std::sync::OnceLock;

use crate::math::{Color, Mat3, Vec3};

/// 対角行列を生成する。
fn diag(v: Vec3) -> Mat3 {
    Mat3 {
        m: [[v.x, 0.0, 0.0], [0.0, v.y, 0.0], [0.0, 0.0, v.z]],
    }
}

/// CIE xy 色度座標を XYZ に変換する（Y=1 に正規化）。
fn xy_to_xyz(x: f64, y: f64) -> Vec3 {
    let x_val = x / y;
    let y_val = 1.0;
    let z_val = (1.0 - x - y) / y;
    Vec3::new(x_val, y_val, z_val)
}

/// RGB 原色と白色点から RGB→XYZ 変換行列を導出する。
fn rgb_to_xyz_matrix(primaries: [(f64, f64); 3], white: (f64, f64)) -> Mat3 {
    let r = xy_to_xyz(primaries[0].0, primaries[0].1);
    let g = xy_to_xyz(primaries[1].0, primaries[1].1);
    let b = xy_to_xyz(primaries[2].0, primaries[2].1);

    let m = Mat3 {
        m: [[r.x, g.x, b.x], [r.y, g.y, b.y], [r.z, g.z, b.z]],
    };

    let w = xy_to_xyz(white.0, white.1);
    let s = m.invert().mul_vec(w);
    m.mul(diag(s))
}

/// Bradford 色順応変換行列を計算する（白色点の変換）。
/// 人間の視覚の色順応をシミュレートし、異なる照明条件間の色を対応づける。
fn chromatic_adaptation_bradford(src_white: Vec3, dst_white: Vec3) -> Mat3 {
    let m = Mat3 {
        m: [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
    };
    let m_inv = Mat3 {
        m: [
            [0.9869929, -0.1470543, 0.1599627],
            [0.4323053, 0.5183603, 0.0492912],
            [-0.0085287, 0.0400428, 0.9684867],
        ],
    };

    let src = m.mul_vec(src_white);
    let dst = m.mul_vec(dst_white);
    let scale = Vec3::new(dst.x / src.x, dst.y / src.y, dst.z / src.z);
    m_inv.mul(diag(scale)).mul(m)
}

/// sRGB → ACEScg の 3×3 変換行列を計算してキャッシュする。
fn srgb_to_acescg_matrix() -> Mat3 {
    static MAT: OnceLock<Mat3> = OnceLock::new();
    *MAT.get_or_init(|| {
        let rec709 = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)];
        let d65 = (0.3127, 0.3290);

        let acescg = [(0.713, 0.293), (0.165, 0.830), (0.128, 0.044)];
        let d60 = (0.32168, 0.33767);

        let m_src = rgb_to_xyz_matrix(rec709, d65);
        let m_dst = rgb_to_xyz_matrix(acescg, d60);
        let adapt = chromatic_adaptation_bradford(xy_to_xyz(d65.0, d65.1), xy_to_xyz(d60.0, d60.1));
        m_dst.invert().mul(adapt).mul(m_src)
    })
}

/// リニア sRGB/Rec.709 の色をリニア ACEScg に変換する。
pub fn srgb_to_acescg(color: Color) -> Color {
    let v: Vec3 = color.into();
    Color(srgb_to_acescg_matrix().mul_vec(v))
}

/// ピクセルバッファ全体をリニア sRGB/Rec.709 → リニア ACEScg に一括変換する。
pub fn srgb_to_acescg_pixels(pixels: &[Color]) -> Vec<Color> {
    pixels.iter().map(|&c| srgb_to_acescg(c)).collect()
}
