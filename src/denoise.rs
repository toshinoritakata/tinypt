//! Intel Open Image Denoise (OIDN) によるデノイズ。
//!
//! 機械学習ベースのフィルターでモンテカルロノイズを除去する。
//! `oidn` feature が有効な場合のみ OIDN ライブラリを使用し、
//! 無効な場合はスタブ（パススルー）が使用される。

use crate::math::Color;

/// OIDN デノイザー（oidn feature 有効時）。
///
/// Color (f64) → f32 バッファに変換し、OIDN の RayTracing フィルターを適用後、
/// f64 に戻して返す。HDR モードで動作し、sRGB ガンマは適用しない。
#[cfg(feature = "oidn")]
pub fn denoise_oidn(pixels: &[Color], w: usize, h: usize) -> Vec<Color> {
    use oidn::{Device, RayTracing};

    if pixels.is_empty() {
        return Vec::new();
    }

    // Convert Color to f32 RGB buffer (OIDN expects f32)
    let mut buffer: Vec<f32> = Vec::with_capacity(w * h * 3);
    for c in pixels {
        buffer.push(c.r() as f32);
        buffer.push(c.g() as f32);
        buffer.push(c.b() as f32);
    }

    // Create OIDN device and filter (filter_in_place modifies buffer directly)
    let device = Device::new();
    RayTracing::new(&device)
        .srgb(false)
        .hdr(true)
        .image_dimensions(w, h)
        .filter_in_place(&mut buffer)
        .expect("OIDN denoising failed");

    // Convert back to Color
    let mut result = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        let r = buffer[i * 3] as f64;
        let g = buffer[i * 3 + 1] as f64;
        let b = buffer[i * 3 + 2] as f64;
        result.push(Color::new(r, g, b));
    }

    result
}

/// OIDN feature 無効時のスタブ（入力をそのまま返す）。
#[cfg(not(feature = "oidn"))]
pub fn denoise_oidn(pixels: &[Color], _w: usize, _h: usize) -> Vec<Color> {
    eprintln!("Warning: OIDN denoiser requested but 'oidn' feature is not enabled.");
    eprintln!("         Falling back to no denoising. Build with --features oidn to enable.");
    pixels.to_vec()
}
