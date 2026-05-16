//! 画像出力ヘルパー（PPM / HDR / EXR フォーマット対応）。
//!
//! 蓄積バッファからピクセルを解決し、拡張子に応じて適切なフォーマットで出力する。
//! - PPM: sRGB ガンマ補正 (γ=2.2) を適用して 8bit 出力
//! - HDR: リニア RGB を RGBE エンコーディングで出力
//! - EXR: リニア sRGB → ACEScg 変換後に float32 で出力

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::aces::srgb_to_acescg_pixels;
use crate::exr::write_exr;
use crate::hdr::write_hdr;
use crate::config::Tonemap;
use crate::math::{clamp, Color};
use crate::task::idx;

/// 蓄積バッファを最終リニア RGB ピクセルに変換する（acc[i] / acc_w[i]）。
pub fn resolve_pixels(w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> Vec<Color> {
    let mut pixels = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let c = acc[i] / acc_w[i].max(1.0);
        pixels.push(c);
    }
    pixels
}

/// 蓄積バッファを PPM 画像として出力する（sRGB ガンマ補正付き）。
pub fn write_ppm(path: &str, w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P3\n{} {}\n255", w, h)?;
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            let c = (acc[i] / acc_w[i].max(1.0)).clamp01();
            write!(
                out,
                "{} {} {} ",
                to_u8(c.r()),
                to_u8(c.g()),
                to_u8(c.b())
            )?;
        }
        writeln!(out)?;
    }
    Ok(())
}

/// 蓄積バッファを Radiance HDR ファイルとして出力する。
pub fn write_hdr_image(path: &str, w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> std::io::Result<()> {
    let pixels = resolve_pixels(w, h, acc, acc_w);
    write_hdr(path, w, h, &pixels)
}

/// 蓄積バッファを EXR ファイルとして出力する（ACEScg 色空間に変換）。
pub fn write_exr_image(path: &str, w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> std::io::Result<()> {
    let pixels = resolve_pixels(w, h, acc, acc_w);
    let pixels = srgb_to_acescg_pixels(&pixels);
    write_exr(path, w, h, &pixels)
}

/// ピクセルをファイルに出力する（拡張子から形式を推定）。
pub fn write_image_pixels(path: &str, w: usize, h: usize, pixels: &[Color]) -> std::io::Result<()> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".exr") {
        let pixels = srgb_to_acescg_pixels(pixels);
        write_exr(path, w, h, &pixels)
    } else if lower.ends_with(".hdr") {
        write_hdr(path, w, h, pixels)
    } else {
        write_ppm_pixels(path, w, h, pixels)
    }
}

/// 蓄積バッファをファイルに出力する（拡張子から形式を推定）。
pub fn write_image(path: &str, w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> std::io::Result<()> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".exr") {
        write_exr_image(path, w, h, acc, acc_w)
    } else if lower.ends_with(".hdr") {
        write_hdr_image(path, w, h, acc, acc_w)
    } else {
        write_ppm(path, w, h, acc, acc_w)
    }
}

/// リニア空間で露出補正とトーンマッピングを適用する。
/// 露出: 2^exposure 倍のスケーリング。EV=1 で 2 倍の明るさ。
pub fn apply_exposure_tonemap(pixels: &[Color], exposure: f64, tonemap: Tonemap) -> Vec<Color> {
    let scale = 2.0_f64.powf(exposure);
    pixels
        .iter()
        .map(|&c| {
            let c = c * scale;
            match tonemap {
                Tonemap::None => c,
                Tonemap::Aces => Color::new(
                    tonemap_aces_fitted(c.r()),
                    tonemap_aces_fitted(c.g()),
                    tonemap_aces_fitted(c.b()),
                ),
            }
        })
        .collect()
}

fn write_ppm_pixels(path: &str, w: usize, h: usize, pixels: &[Color]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P3\n{} {}\n255", w, h)?;
    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            let c = pixels[i].clamp01();
            write!(
                out,
                "{} {} {} ",
                to_u8(c.r()),
                to_u8(c.g()),
                to_u8(c.b())
            )?;
        }
        writeln!(out)?;
    }
    Ok(())
}

/// リニア値を sRGB ガンマ補正（γ=2.2 近似）して 8bit に量子化する。
fn to_u8(x: f64) -> u8 {
    let v = clamp(x, 0.0, 1.0).powf(1.0 / 2.2);
    (v * 255.0 + 0.5) as u8
}

/// ACES フィルミック・トーンマッピング（Narkowicz 近似）。
/// HDR → SDR のS字カーブで、暗部のコントラストと明部のロールオフを提供する。
fn tonemap_aces_fitted(x: f64) -> f64 {
    let x = x.max(0.0);
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
}
