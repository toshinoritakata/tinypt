//! 画像出力。リニア RGB ピクセルを各フォーマットのバイト列まで変換する。
//!
//! [`OutputFormat`] が拡張子ごとの色パイプライン全体を所有する。呼び出し側は
//! 蓄積バッファを一度だけ [`resolve_pixels`] でリニア RGB に解決し、フォーマットに
//! 渡すだけでよい（露出・トーンマップ・色空間・ガンマの判断はフォーマット内部）。
//!
//! - **PPM**: 露出補正 → トーンマップ → sRGB ガンマ補正 (γ=2.2) → 8bit
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
use crate::math::{clamp, Color};
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
    /// PPM（8bit, sRGB ガンマ）
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
                // LDR: 露出補正 → トーンマップ → sRGB ガンマ → 8bit
                let mut out = BufWriter::new(File::create(path)?);
                writeln!(out, "P3\n{} {}\n255", w, h)?;
                let scale = 2.0_f64.powf(settings.exposure);
                for y in 0..h {
                    for x in 0..w {
                        let c = tonemap(pixels[idx(x, y, w)] * scale, settings.tonemap).clamp01();
                        write!(out, "{} {} {} ", to_u8(c.r()), to_u8(c.g()), to_u8(c.b()))?;
                    }
                    writeln!(out)?;
                }
                Ok(())
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

    /// PPM は露出 0・トーンマップ None でも sRGB ガンマを適用する。
    /// リニア 0.5 → 0.5^(1/2.2)·255 ≈ 186。
    #[test]
    fn ppm_applies_srgb_gamma() {
        let expected = (clamp(0.5_f64, 0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        assert_eq!(to_u8(0.5), expected);
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert!(to_u8(2.0) == 255); // クランプ
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
}
