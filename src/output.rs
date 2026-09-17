//! 画像出力。リニア RGB ピクセルを各フォーマットのバイト列まで変換する。
//!
//! [`OutputFormat`] が拡張子ごとの色パイプライン全体を所有する。呼び出し側は
//! 蓄積バッファを一度だけ [`resolve_pixels`] でリニア RGB に解決し、フォーマットに
//! 渡すだけでよい（露出・トーンマップ・色空間・ガンマの判断はフォーマット内部）。
//!
//! - **PPM**: 露出補正 → トーンマップ → sRGB エンコード（入力デコードと対称） → 8bit、バイナリ（P6）
//! - **HDR**: リニア RGB を RGBE エンコーディングで出力（シーン参照値を保存）
//! - **EXR**: リニア sRGB → ACEScg 変換後に float32 で出力（シーン参照値を保存）
//!
//! 露出・トーンマップは LDR の PPM のみに適用し、HDR/EXR はシーン参照リニア値を
//! そのまま保存する。

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::aces::srgb_to_acescg_pixels;
use crate::config::Tonemap;
use crate::exr::write_exr;
use crate::hdr::write_hdr;
use crate::math::{clamp, linear_to_srgb, Color};
use crate::task::idx;

/// 出力時の色調整設定（LDR フォーマットにのみ適用される）。
#[derive(Clone, Copy)]
pub struct OutputSettings {
    /// 露出補正（EV 単位、2^exposure 倍のスケーリング）
    pub exposure: f64,
    /// トーンマッピング方式
    pub tonemap: Tonemap,
}

/// 出力フォーマット。各 variant が自身の色パイプラインを所有する。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    /// PPM（P6 バイナリ, 8bit, sRGB ガンマ）
    Ppm,
    /// Radiance HDR（RGBE, リニア）
    Hdr,
    /// OpenEXR（float32, ACEScg）
    Exr,
}

impl OutputFormat {
    /// 出力パスの拡張子からフォーマットを判定する（唯一の拡張子マッチ）。
    pub fn from_path(path: &str) -> Self {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".exr") {
            OutputFormat::Exr
        } else if lower.ends_with(".hdr") {
            OutputFormat::Hdr
        } else {
            OutputFormat::Ppm
        }
    }

    /// リニア RGB ピクセルをこのフォーマットでファイルに書き出す。
    /// 色空間変換・トーンマップ・ガンマはフォーマットが内部で適用する。
    pub fn write(
        self,
        path: &str,
        w: usize,
        h: usize,
        pixels: &[Color],
        settings: OutputSettings,
    ) -> std::io::Result<()> {
        match self {
            OutputFormat::Ppm => {
                // LDR: 露出補正 → トーンマップ → sRGB ガンマ → 8bit。バイナリ PPM（P6）で書く
                let mut out = BufWriter::new(File::create(path)?);
                write_ppm_p6(&mut out, w, h, &ppm_bytes(w, h, pixels, settings))?;
                out.flush()
            }
            OutputFormat::Hdr => {
                // シーン参照リニア値をそのまま RGBE 出力
                write_hdr(path, w, h, pixels)
            }
            OutputFormat::Exr => {
                // シーン参照リニア値を ACEScg に変換して float32 出力
                let pixels = srgb_to_acescg_pixels(pixels);
                write_exr(path, w, h, &pixels)
            }
        }
    }
}

/// PPM の画素値（行優先、1 画素 R, G, B の 8bit）。露出補正 → トーンマップ → sRGB エンコード → 量子化。
fn ppm_bytes(w: usize, h: usize, pixels: &[Color], settings: OutputSettings) -> Vec<u8> {
    let scale = 2.0_f64.powf(settings.exposure);
    let mut bytes = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let c = tonemap(pixels[idx(x, y, w)] * scale, settings.tonemap).clamp01();
            bytes.extend_from_slice(&[to_u8(c.r()), to_u8(c.g()), to_u8(c.b())]);
        }
    }
    bytes
}

/// バイナリ PPM（P6、maxval 255）を書く。ヘッダの後に画素値のバイト列がそのまま続く。
fn write_ppm_p6(out: &mut impl Write, w: usize, h: usize, rgb: &[u8]) -> std::io::Result<()> {
    write!(out, "P6\n{} {}\n255\n", w, h)?;
    out.write_all(rgb)
}

/// 蓄積バッファを最終リニア RGB ピクセルに変換する（acc[i] / acc_w[i]）。
pub fn resolve_pixels(w: usize, h: usize, acc: &[Color], acc_w: &[f64]) -> Vec<Color> {
    let mut pixels = Vec::with_capacity(w * h);
    for i in 0..w * h {
        pixels.push(acc[i] / acc_w[i].max(1.0));
    }
    pixels
}

/// 露出補正後の色にトーンマッピングを適用する。
fn tonemap(c: Color, tonemap: Tonemap) -> Color {
    match tonemap {
        Tonemap::None => c,
        Tonemap::Aces => Color::new(
            tonemap_aces_fitted(c.r()),
            tonemap_aces_fitted(c.g()),
            tonemap_aces_fitted(c.b()),
        ),
    }
}

/// リニア値を sRGB エンコード（入力の `srgb_to_linear` と対称）して 8bit に量子化する。
fn to_u8(x: f64) -> u8 {
    let v = linear_to_srgb(clamp(x, 0.0, 1.0));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 拡張子マッチは1か所（from_path）に集約され、大文字小文字を無視する。
    #[test]
    fn from_path_picks_format() {
        assert_eq!(OutputFormat::from_path("out.exr"), OutputFormat::Exr);
        assert_eq!(OutputFormat::from_path("OUT.EXR"), OutputFormat::Exr);
        assert_eq!(OutputFormat::from_path("out.hdr"), OutputFormat::Hdr);
        assert_eq!(OutputFormat::from_path("out.HDR"), OutputFormat::Hdr);
        assert_eq!(OutputFormat::from_path("out.ppm"), OutputFormat::Ppm);
        // 未知の拡張子は PPM にフォールバック
        assert_eq!(OutputFormat::from_path("out.txt"), OutputFormat::Ppm);
        assert_eq!(OutputFormat::from_path("out"), OutputFormat::Ppm);
    }

    /// resolve_pixels は acc/重みを割り、重み 0 でも発散しない（max(1.0) クランプ）。
    #[test]
    fn resolve_divides_by_weight() {
        let acc = vec![Color::new(4.0, 2.0, 1.0), Color::new(0.0, 0.0, 0.0)];
        let acc_w = vec![2.0, 0.0];
        let px = resolve_pixels(2, 1, &acc, &acc_w);
        assert!((px[0].r() - 2.0).abs() < 1e-12);
        assert!((px[0].g() - 1.0).abs() < 1e-12);
        assert!((px[1].r() - 0.0).abs() < 1e-12); // 0/max(1.0) = 0、NaN にならない
    }

    /// PPM は露出 0・トーンマップ None でも sRGB エンコードを適用する。
    /// 入力の srgb_to_linear と対称な正確な sRGB カーブを使う。
    #[test]
    fn ppm_applies_srgb_encode() {
        let expected = (linear_to_srgb(0.5) * 255.0 + 0.5) as u8;
        assert_eq!(to_u8(0.5), expected);
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert!(to_u8(2.0) == 255); // クランプ
        // sRGB は srgb_to_linear の逆: round-trip が一致する
        let lin = crate::math::srgb_to_linear(0.6);
        assert!((linear_to_srgb(lin) - 0.6).abs() < 1e-9);
    }

    /// HDR 書き出し → 読み戻しでリニア値が概ね保存される（RGBE 往復）。
    #[test]
    fn hdr_roundtrip_preserves_linear() {
        use crate::hdr::{read_hdr, write_hdr};
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tinypt_test_{}.hdr", std::process::id()));
        let path = path.to_str().unwrap();
        let pixels = vec![
            Color::new(0.25, 0.5, 1.0),
            Color::new(2.0, 0.1, 0.0),
            Color::new(0.0, 0.0, 0.0),
            Color::new(8.0, 4.0, 2.0),
        ];
        write_hdr(path, 2, 2, &pixels).unwrap();
        let img = read_hdr(path).unwrap();
        std::fs::remove_file(path).ok();
        assert_eq!((img.width, img.height), (2, 2));
        for (a, b) in pixels.iter().zip(img.data.iter()) {
            // RGBE は共有指数のため、量子化誤差はピクセル最大チャンネルに比例する
            let pmax = a.r().max(a.g()).max(a.b());
            let close = |x: f64, y: f64| (x - y).abs() <= pmax / 128.0 + 1e-6;
            assert!(close(a.r(), b.r()) && close(a.g(), b.g()) && close(a.b(), b.b()), "{:?} vs {:?}", a, b);
        }
    }

    /// 旧形式の ASCII PPM（P3）を書く（P6 への移行で画素値が変わらないことの比較用。旧 writer と同じ書式）。
    fn write_ppm_p3(out: &mut impl Write, w: usize, h: usize, rgb: &[u8]) -> std::io::Result<()> {
        writeln!(out, "P3\n{} {}\n255", w, h)?;
        for row in rgb.chunks(w * 3) {
            for px in row.chunks(3) {
                write!(out, "{} {} {} ", px[0], px[1], px[2])?;
            }
            writeln!(out)?;
        }
        Ok(())
    }

    /// P3 / P6 の PPM（maxval 255）を読み、(幅, 高さ, 画素値) を返す。ヘッダのコメント（`#`）にも対応する。
    fn read_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
        let mut pos = 0;
        // ヘッダのトークンを 1 つ読む（空白とコメントを飛ばす）
        let mut token = || -> String {
            loop {
                while data[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if data[pos] != b'#' {
                    break;
                }
                while data[pos] != b'\n' {
                    pos += 1;
                }
            }
            let start = pos;
            while pos < data.len() && !data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            String::from_utf8(data[start..pos].to_vec()).unwrap()
        };
        let magic = token();
        let w: usize = token().parse().unwrap();
        let h: usize = token().parse().unwrap();
        assert_eq!(token(), "255");
        let rgb = match magic.as_str() {
            // P6: maxval の後の空白 1 バイトの直後から画素値
            "P6" => {
                let body = &data[pos + 1..];
                assert_eq!(body.len(), w * h * 3, "P6 body length");
                body.to_vec()
            }
            "P3" => {
                let v: Vec<u8> = std::str::from_utf8(&data[pos..]).unwrap().split_ascii_whitespace().map(|t| t.parse().unwrap()).collect();
                assert_eq!(v.len(), w * h * 3, "P3 value count");
                v
            }
            m => panic!("unknown PPM magic {}", m),
        };
        (w, h, rgb)
    }

    /// PPM 出力は P6（バイナリ）: ヘッダ + 3·w·h バイト。書いたファイルを読み戻すと、同じバッファから
    /// 旧形式（P3）で書いたものと画素値が一致し、to_u8 で直接計算した値とも一致する（出力画素値は不変）。
    /// 非正方形（行と列の取り違えを検出）、0 と 255 に飽和する値、露出・ACES の両方で確かめる。
    #[test]
    fn ppm_p6_roundtrip_matches_p3_values() {
        let (w, h) = (7, 3);
        let pixels: Vec<Color> = (0..w * h)
            .map(|i| {
                let f = i as f64 / (w * h) as f64;
                Color::new(f * 1.5, (1.0 - f) * 0.3, if i % 5 == 0 { 20.0 } else { f * f })
            })
            .collect();
        let dir = std::env::temp_dir();
        for (k, settings) in [
            OutputSettings { exposure: 0.0, tonemap: Tonemap::None },
            OutputSettings { exposure: 1.5, tonemap: Tonemap::Aces },
        ]
        .into_iter()
        .enumerate()
        {
            let path = dir.join(format!("tinypt_test_{}_{}.ppm", std::process::id(), k));
            let path = path.to_str().unwrap();
            OutputFormat::Ppm.write(path, w, h, &pixels, settings).unwrap();
            let file = std::fs::read(path).unwrap();
            std::fs::remove_file(path).ok();
            let header = format!("P6\n{} {}\n255\n", w, h);
            assert!(file.starts_with(header.as_bytes()), "P6 header");
            assert_eq!(file.len(), header.len() + 3 * w * h, "P6 file size");

            let (w6, h6, rgb6) = read_ppm(&file);
            let mut p3 = Vec::new();
            write_ppm_p3(&mut p3, w, h, &ppm_bytes(w, h, &pixels, settings)).unwrap();
            let (w3, h3, rgb3) = read_ppm(&p3);
            assert_eq!((w6, h6), (w, h));
            assert_eq!((w3, h3), (w, h));
            assert_eq!(rgb6, rgb3, "P6 and P3 decode to different pixel values");

            let scale = 2.0_f64.powf(settings.exposure);
            for y in 0..h {
                for x in 0..w {
                    let c = tonemap(pixels[idx(x, y, w)] * scale, settings.tonemap).clamp01();
                    let i = 3 * (y * w + x);
                    assert_eq!(&rgb6[i..i + 3], &[to_u8(c.r()), to_u8(c.g()), to_u8(c.b())], "pixel ({}, {})", x, y);
                }
            }
            assert!(rgb6.contains(&255) && rgb6.contains(&0) || k == 1, "test values should saturate");
        }
    }

    /// 読み取りの自己確認: 手書きの P3（コメント付き）と、同じ値の P6 のバイト列が同じ画素値に読める。
    #[test]
    fn read_ppm_parses_both_encodings() {
        let p3 = b"P3\n# comment\n2 1\n255\n0 128 255  10 20 30\n";
        let p6 = [b"P6\n2 1\n255\n".as_slice(), &[0, 128, 255, 10, 20, 30]].concat();
        assert_eq!(read_ppm(p3), (2, 1, vec![0, 128, 255, 10, 20, 30]));
        assert_eq!(read_ppm(&p6), (2, 1, vec![0, 128, 255, 10, 20, 30]));
        // 画素値に空白・改行と同じバイト（10, 32）が含まれても、P6 は長さで読むので壊れない
        let p6_ws = [b"P6\n2 1\n255\n".as_slice(), &[10, 32, 9, 13, 35, 10]].concat();
        assert_eq!(read_ppm(&p6_ws), (2, 1, vec![10, 32, 9, 13, 35, 10]));
    }
}
