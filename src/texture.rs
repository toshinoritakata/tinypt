//! ビットマップテクスチャ（T1: 色テクスチャのみ）。
//!
//! ## 色空間
//! 色テクスチャ（`reflectance` など）は **sRGB** としてデコードする。入力の
//! [`srgb_to_linear`](crate::math::srgb_to_linear) と対称で、PPM 出力の `linear_to_srgb` の逆にあたる
//! （README の「色は入出力で対称」の規約をテクスチャにも通す）。データ系（roughness 等）は
//! リニアのまま読む（Mitsuba の `raw=true` 相当）。8bit 画像は 256 要素の表で 1 回だけ変換する。
//!
//! ## UV の向き
//! Wavefront OBJ と Mitsuba の `vt` は**左下が原点**（v が上向き）。画像は上の行から並ぶので、
//! テクセルの行は `(1 − v)` 側から数える。この上下の向きはテストで固定してある。
//!
//! ## フィルタとラップ
//! バイリニア補間（テクセル**中心**を基準）。範囲外の UV は [`Wrap`] で決める
//! （既定は Mitsuba と同じ `repeat`）。

use crate::math::{srgb_to_linear, Color};

/// テクスチャ座標が [0, 1] の外に出たときの扱い。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wrap {
    /// 繰り返す（Mitsuba の既定）
    Repeat,
    /// 端の値で止める
    Clamp,
}

impl Wrap {
    /// シーンファイルの文字列から。未知の値は `None`（呼び出し側が警告する）。
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "repeat" => Some(Wrap::Repeat),
            "clamp" => Some(Wrap::Clamp),
            _ => None,
        }
    }
}

/// ビットマップテクスチャ（リニア色で保持）。
pub struct Texture {
    /// 幅（テクセル）
    width: usize,
    /// 高さ（テクセル）
    height: usize,
    /// リニア色のテクセル（行優先、1 行目が画像の上端）
    texels: Vec<Color>,
    /// 範囲外 UV の扱い
    wrap: Wrap,
}

impl Texture {
    /// リニア色のテクセル列から作る（テスト・合成テクスチャ用）。
    pub fn from_linear(width: usize, height: usize, texels: Vec<Color>, wrap: Wrap) -> Self {
        assert_eq!(texels.len(), width * height, "texel count must be width * height");
        Self { width, height, texels, wrap }
    }

    /// 画像ファイル（PNG / JPEG など `image` crate が読める形式）から読み込む。
    ///
    /// `srgb` が true なら sRGB としてデコードしてリニアに直す（色テクスチャ）。
    /// false なら 0..1 の値をそのまま使う（データテクスチャ）。
    pub fn load(path: &str, srgb: bool, wrap: Wrap) -> Result<Self, String> {
        let img = image::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        if w == 0 || h == 0 {
            return Err(format!("{}: empty image", path));
        }
        // 8bit は取りうる値が 256 通りしかないので、変換表を 1 度だけ作る
        let lut: Vec<f64> = (0..256)
            .map(|i| {
                let x = i as f64 / 255.0;
                if srgb { srgb_to_linear(x) } else { x }
            })
            .collect();
        let texels = rgb
            .pixels()
            .map(|p| Color::new(lut[p.0[0] as usize], lut[p.0[1] as usize], lut[p.0[2] as usize]))
            .collect();
        Ok(Self { width: w, height: h, texels, wrap })
    }

    /// 幅・高さ（テクセル）。
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// 範囲外の扱い。
    pub fn wrap(&self) -> Wrap {
        self.wrap
    }

    /// テクセルを直接引く（行優先、範囲外は [`Wrap`] に従う）。
    fn texel(&self, x: i64, y: i64) -> Color {
        let (w, h) = (self.width as i64, self.height as i64);
        let (x, y) = match self.wrap {
            Wrap::Repeat => (x.rem_euclid(w), y.rem_euclid(h)),
            Wrap::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
        };
        self.texels[(y as usize) * self.width + (x as usize)]
    }

    /// UV でバイリニア補間した色を返す。
    ///
    /// テクセル中心を基準にする（`u·width − 0.5` が中心どうしの座標）。`v` は下から上なので
    /// 行は `(1 − v)` 側から数える。
    pub fn sample(&self, uv: (f64, f64)) -> Color {
        let (u, v) = uv;
        if !u.is_finite() || !v.is_finite() {
            return Color::new(0.0, 0.0, 0.0);
        }
        let x = u * self.width as f64 - 0.5;
        let y = (1.0 - v) * self.height as f64 - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let c00 = self.texel(x0, y0);
        let c10 = self.texel(x0 + 1, y0);
        let c01 = self.texel(x0, y0 + 1);
        let c11 = self.texel(x0 + 1, y0 + 1);
        let top = c00 * (1.0 - fx) + c10 * fx;
        let bottom = c01 * (1.0 - fx) + c11 * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2 のテクスチャ。左下が赤・右下が緑・左上が青・右上が白。
    fn checker() -> Texture {
        // 行優先で上の行から: 左上(青) 右上(白) / 左下(赤) 右下(緑)
        Texture::from_linear(
            2,
            2,
            vec![
                Color::new(0.0, 0.0, 1.0), Color::new(1.0, 1.0, 1.0),
                Color::new(1.0, 0.0, 0.0), Color::new(0.0, 1.0, 0.0),
            ],
            Wrap::Clamp,
        )
    }

    /// テクセル中心（(0.25, 0.25) など）ではそのテクセルの値がそのまま返る。
    /// **v は下から上**なので、v = 0.25 は画像の下の行を指す。
    #[test]
    fn samples_texel_centers_exactly_with_v_up() {
        let t = checker();
        let approx = |a: Color, b: Color| (a.0 - b.0).len() < 1e-12;
        assert!(approx(t.sample((0.25, 0.25)), Color::new(1.0, 0.0, 0.0)), "左下は赤");
        assert!(approx(t.sample((0.75, 0.25)), Color::new(0.0, 1.0, 0.0)), "右下は緑");
        assert!(approx(t.sample((0.25, 0.75)), Color::new(0.0, 0.0, 1.0)), "左上は青");
        assert!(approx(t.sample((0.75, 0.75)), Color::new(1.0, 1.0, 1.0)), "右上は白");
    }

    /// テクセル中心の中間ではバイリニアに混ざる。
    #[test]
    fn interpolates_bilinearly_between_centers() {
        let t = checker();
        // 左下と右下の中間（v は下の行の中心）
        let c = t.sample((0.5, 0.25));
        assert!((c.r() - 0.5).abs() < 1e-12 && (c.g() - 0.5).abs() < 1e-12 && c.b().abs() < 1e-12, "{:?}", (c.r(), c.g(), c.b()));
        // 4 テクセルの中心（全部の平均）
        let c = t.sample((0.5, 0.5));
        assert!((c.r() - 0.5).abs() < 1e-12 && (c.g() - 0.5).abs() < 1e-12 && (c.b() - 0.5).abs() < 1e-12);
    }

    /// `repeat` は 1 を超える UV を巻き戻す。`clamp` は端で止める。
    #[test]
    fn wrap_modes_behave_differently_outside_the_unit_square() {
        let texels = vec![Color::new(1.0, 0.0, 0.0), Color::new(0.0, 0.0, 1.0)]; // 左:赤 右:青
        let rep = Texture::from_linear(2, 1, texels.iter().copied().collect(), Wrap::Repeat);
        let cla = Texture::from_linear(2, 1, texels, Wrap::Clamp);
        // u = 1.25 は repeat では u = 0.25（左のテクセル中心 = 赤）に戻る
        let c = rep.sample((1.25, 0.5));
        assert!((c.r() - 1.0).abs() < 1e-12 && c.b().abs() < 1e-12, "repeat: {:?}", (c.r(), c.b()));
        // clamp では右端（青）のまま
        let c = cla.sample((1.25, 0.5));
        assert!(c.r().abs() < 1e-12 && (c.b() - 1.0).abs() < 1e-12, "clamp: {:?}", (c.r(), c.b()));
        // 負の側も同じ
        let c = rep.sample((-0.25, 0.5));
        assert!((c.b() - 1.0).abs() < 1e-12, "repeat 負側: {:?}", (c.r(), c.b()));
        let c = cla.sample((-0.25, 0.5));
        assert!((c.r() - 1.0).abs() < 1e-12, "clamp 負側: {:?}", (c.r(), c.b()));
    }

    /// NaN / 無限大の UV でもパニックせず黒を返す。
    #[test]
    fn non_finite_uv_is_black() {
        let t = checker();
        for uv in [(f64::NAN, 0.5), (0.5, f64::INFINITY), (f64::NEG_INFINITY, f64::NAN)] {
            let c = t.sample(uv);
            assert_eq!((c.r(), c.g(), c.b()), (0.0, 0.0, 0.0), "uv = {:?}", uv);
        }
    }

    /// PNG を読み込むと sRGB デコードが掛かり、`raw` 指定ではリニアのまま読む。
    #[test]
    fn png_is_decoded_as_srgb_unless_raw() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tinypt_tex_{}.png", std::process::id()));
        // 1x1 の中間グレー (128, 128, 128)
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([128, 128, 128]));
        img.save(&path).unwrap();
        let p = path.to_string_lossy().into_owned();

        let srgb = Texture::load(&p, true, Wrap::Repeat).unwrap();
        let raw = Texture::load(&p, false, Wrap::Repeat).unwrap();
        std::fs::remove_file(&path).ok();

        let expected_srgb = srgb_to_linear(128.0 / 255.0);
        assert!((srgb.sample((0.5, 0.5)).r() - expected_srgb).abs() < 1e-12, "sRGB デコードが掛かっていない");
        assert!((raw.sample((0.5, 0.5)).r() - 128.0 / 255.0).abs() < 1e-12, "raw はリニアのまま");
        assert!(expected_srgb < 128.0 / 255.0, "sRGB デコードは中間グレーを暗くする");
        assert_eq!(srgb.size(), (1, 1));
    }

    /// 読めないファイルはエラーを返す（パニックしない）。
    #[test]
    fn missing_file_is_an_error() {
        assert!(Texture::load("/definitely/not/here.png", true, Wrap::Repeat).is_err());
    }
}
