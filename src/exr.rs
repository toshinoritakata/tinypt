//! OpenEXR 画像の読み書きヘルパー。
//!
//! 読み込みは `image` crate を使用し、書き出しは `exr` crate で直接 EXR を生成する。
//! 書き出し時には ACEScg 色空間のクロマティシティを EXR メタデータに埋め込む。

use std::io;

use image::ImageReader;
use exr::prelude::*;
use exr::meta::attribute::Chromaticities;

use crate::hdr::HdrImage;
use crate::math::Color;

fn map_image_err(err: image::ImageError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

/// EXR ファイルを `HdrImage` として読み込む（image crate で RGB32F にデコード）。
pub fn read_exr(path: &str) -> io::Result<HdrImage> {
    let img = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .map_err(map_image_err)?
        .to_rgb32f();

    let (w, h) = img.dimensions();
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y).0;
            data.push(Color::new(px[0] as f64, px[1] as f64, px[2] as f64));
        }
    }

    Ok(HdrImage {
        width: w as usize,
        height: h as usize,
        data,
    })
}

/// リニア RGB ピクセルを EXR ファイルに書き出す（ACEScg クロマティシティ付き）。
pub fn write_exr(path: &str, w: usize, h: usize, pixels: &[Color]) -> io::Result<()> {
    let mut attrs = ImageAttributes::with_size((w, h));
    attrs.chromaticities = Some(Chromaticities {
        red: Vec2(0.713, 0.293),
        green: Vec2(0.165, 0.830),
        blue: Vec2(0.128, 0.044),
        white: Vec2(0.32168, 0.33767),
    });

    let channels = SpecificChannels::build()
        .with_channel::<f32>("R")
        .with_channel::<f32>("G")
        .with_channel::<f32>("B")
        .with_pixel_fn(|pos: Vec2<usize>| {
            let i = pos.y() * w + pos.x();
            let c = pixels[i];
            (c.r() as f32, c.g() as f32, c.b() as f32)
        });

    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("ACEScg"),
        Encoding::FAST_LOSSLESS,
        channels,
    );

    let image = Image::empty(attrs).with_layer(layer);
    image
        .write()
        .to_file(path)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))
}
