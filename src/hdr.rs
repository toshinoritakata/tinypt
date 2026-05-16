//! Radiance RGBE (.hdr) 画像の読み書き。
//!
//! RGBE は RGB 各 8bit + 共有指数 8bit = 32bit/pixel の HDR フォーマット。
//! 環境マップの読み込みやレンダリング結果の出力に使用する。
//! RLE（Run-Length Encoding）圧縮に対応。

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

use crate::math::{clamp, Color, Vec3};

/// メモリ上の HDR 画像（リニア RGB ピクセル）。
pub struct HdrImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Color>,
}

/// Radiance RGBE (.hdr) ファイルをリニア RGB として読み込む。
pub fn read_hdr(path: &str) -> io::Result<HdrImage> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    if !line.starts_with("#?RADIANCE") && !line.starts_with("#?RGBE") {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid HDR header"));
    }

    // Read header until empty line.
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing resolution"));
        }
        if line.trim().is_empty() {
            break;
        }
    }

    // Resolution line.
    line.clear();
    reader.read_line(&mut line)?;
    let (x_sign, y_sign, w, h) = parse_resolution(&line)?;

    let mut data = vec![Color::new(0.0, 0.0, 0.0); w * h];
    let mut scanline = vec![0u8; w * 4];

    for y in 0..h {
        read_scanline(&mut reader, &mut scanline, w)?;
        for x in 0..w {
            let src_x = if x_sign == '-' { w - 1 - x } else { x };
            let dst_y = if y_sign == '-' { y } else { h - 1 - y };
            let sidx = src_x * 4;
            let rgbe = &scanline[sidx..sidx + 4];
            let c = rgbe_to_color(rgbe);
            data[dst_y * w + x] = c;
        }
    }

    Ok(HdrImage { width: w, height: h, data })
}

/// リニア RGB ピクセルを Radiance RGBE (.hdr) ファイルに書き出す。
pub fn write_hdr(path: &str, w: usize, h: usize, pixels: &[Color]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    writeln!(out, "#?RADIANCE")?;
    writeln!(out, "FORMAT=32-bit_rle_rgbe")?;
    writeln!(out)?;
    writeln!(out, "-Y {} +X {}", h, w)?;

    let mut scanline = vec![0u8; w * 4];
    for y in 0..h {
        for x in 0..w {
            let c = pixels[y * w + x];
            let rgbe = color_to_rgbe(c);
            let i = x * 4;
            scanline[i] = rgbe[0];
            scanline[i + 1] = rgbe[1];
            scanline[i + 2] = rgbe[2];
            scanline[i + 3] = rgbe[3];
        }
        write_scanline(&mut out, &scanline, w)?;
    }
    Ok(())
}

/// 解像度行を解析する。例: "-Y 1080 +X 1920"
fn parse_resolution(line: &str) -> io::Result<(char, char, usize, usize)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid resolution line"));
    }
    let y_sign = parts[0].chars().next().unwrap_or('-');
    let h: usize = parts[1].parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid height"))?;
    let x_sign = parts[2].chars().next().unwrap_or('+');
    let w: usize = parts[3].parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid width"))?;
    Ok((x_sign, y_sign, w, h))
}

/// 1 スキャンラインを読み込む。新形式（チャネル別 RLE）と旧形式に対応。
fn read_scanline<R: Read>(r: &mut R, out: &mut [u8], w: usize) -> io::Result<()> {
    if w < 8 || w > 0x7fff {
        r.read_exact(out)?;
        return Ok(());
    }

    let mut header = [0u8; 4];
    r.read_exact(&mut header)?;
    if header[0] != 2 || header[1] != 2 || ((header[2] as usize) << 8 | header[3] as usize) != w {
        // Old format: header is first pixel.
        out[0] = header[0];
        out[1] = header[1];
        out[2] = header[2];
        out[3] = header[3];
        r.read_exact(&mut out[4..])?;
        return Ok(());
    }

    for channel in 0..4 {
        let mut x = 0;
        while x < w {
            let mut code = [0u8; 1];
            r.read_exact(&mut code)?;
            if code[0] > 128 {
                let count = (code[0] - 128) as usize;
                let mut val = [0u8; 1];
                r.read_exact(&mut val)?;
                for _ in 0..count {
                    out[x * 4 + channel] = val[0];
                    x += 1;
                }
            } else {
                let count = code[0] as usize;
                for _ in 0..count {
                    let mut val = [0u8; 1];
                    r.read_exact(&mut val)?;
                    out[x * 4 + channel] = val[0];
                    x += 1;
                }
            }
        }
    }
    Ok(())
}

/// 1 スキャンラインを RLE 圧縮して書き出す。
fn write_scanline<W: Write>(wtr: &mut W, scanline: &[u8], w: usize) -> io::Result<()> {
    if w < 8 || w > 0x7fff {
        wtr.write_all(scanline)?;
        return Ok(());
    }

    let header = [2u8, 2u8, ((w >> 8) & 0xff) as u8, (w & 0xff) as u8];
    wtr.write_all(&header)?;

    for channel in 0..4 {
        let mut x = 0;
        while x < w {
            let mut run_len = 1usize;
            let val = scanline[x * 4 + channel];
            while x + run_len < w && run_len < 127 && scanline[(x + run_len) * 4 + channel] == val {
                run_len += 1;
            }
            if run_len >= 4 {
                wtr.write_all(&[(128 + run_len) as u8, val])?;
                x += run_len;
            } else {
                let start = x;
                let mut count = 0usize;
                while x < w && count < 128 {
                    let v = scanline[x * 4 + channel];
                    let mut look = 1usize;
                    while x + look < w && look < 127 && scanline[(x + look) * 4 + channel] == v {
                        look += 1;
                    }
                    if look >= 4 {
                        break;
                    }
                    x += 1;
                    count += 1;
                }
                wtr.write_all(&[count as u8])?;
                for i in 0..count {
                    let v = scanline[(start + i) * 4 + channel];
                    wtr.write_all(&[v])?;
                }
            }
        }
    }
    Ok(())
}

/// RGBE (4 bytes) をリニア RGB に変換する。
/// color = (R, G, B) × 2^(E - 128 - 8)
fn rgbe_to_color(rgbe: &[u8]) -> Color {
    let e = rgbe[3];
    if e == 0 {
        return Color::new(0.0, 0.0, 0.0);
    }
    let f = 2.0f64.powi(e as i32 - 128 - 8);
    Color::from(Vec3::new(
        (rgbe[0] as f64) * f,
        (rgbe[1] as f64) * f,
        (rgbe[2] as f64) * f,
    ))
}

/// リニア RGB を RGBE (4 bytes) に変換する。
fn color_to_rgbe(c: Color) -> [u8; 4] {
    let v: Vec3 = c.into();
    let r = v.x.max(0.0);
    let g = v.y.max(0.0);
    let b = v.z.max(0.0);
    let max = r.max(g).max(b);
    if max < 1e-32 {
        return [0, 0, 0, 0];
    }
    let exp = (max.log2().floor() as i32) + 1;
    let f = 2.0f64.powi(exp - 8);
    let rf = clamp(r / f, 0.0, 255.0) as u8;
    let gf = clamp(g / f, 0.0, 255.0) as u8;
    let bf = clamp(b / f, 0.0, 255.0) as u8;
    [rf, gf, bf, (exp + 128) as u8]
}
