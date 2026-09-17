//! 環境マップの読み込みと重点的サンプリング。
//!
//! 等距離円筒図法（Equirectangular）の HDR/EXR 画像を環境マップとして使用する。
//! 輝度ベースの 2D CDF（行方向 + 列方向）を構築し、
//! 明るい領域を優先的にサンプリングすることで分散を低減する。
//!
//! ## サンプリングの流れ
//! 1. 行 CDF から行 y をサンプリング（sin(θ) による面積補正付き）
//! 2. その行の列 CDF から列 x をサンプリング
//! 3. (x, y) → (θ, φ) → 方向ベクトルに変換

use std::io;

use crate::exr::read_exr;
use crate::hdr::read_hdr;
use crate::math::{cdf_search, clamp, Color, Vec3};
use crate::rng::{uniform_sphere_dir, Rng};

/// 環境マップ（サンプリング用 CDF 付き）。
pub struct EnvMap {
    /// 画像の幅（ピクセル）
    pub width: usize,
    /// 画像の高さ（ピクセル）
    pub height: usize,
    /// リニア RGB ピクセルデータ
    pub data: Vec<Color>,
    /// 行方向の累積分布関数（サイズ: height + 1）
    row_cdf: Vec<f64>,
    /// 列方向の累積分布関数（各行ごと、サイズ: height × (width + 1)）
    col_cdf: Vec<f64>,
    /// CDF の総重み
    total_weight: f64,
}

impl EnvMap {
    /// HDR/EXR ファイルから環境マップを読み込み、サンプリング用 CDF を構築する。
    pub fn from_hdr(path: &str) -> io::Result<Self> {
        let img = if path.to_ascii_lowercase().ends_with(".exr") {
            read_exr(path)?
        } else {
            read_hdr(path)?
        };
        Ok(Self::from_pixels(img.width, img.height, img.data))
    }

    /// 一様な定数色の環境マップ（1×1）。`constant` emitter 用。
    pub fn constant(color: Color) -> Self {
        Self::from_pixels(1, 1, vec![color])
    }

    /// 放射輝度を `factor` 倍した環境マップを返す（`scale` 属性用）。
    pub fn scaled(self, factor: f64) -> Self {
        if factor == 1.0 {
            return self;
        }
        let data = self.data.iter().map(|c| *c * factor).collect();
        Self::from_pixels(self.width, self.height, data)
    }

    /// リニア RGB ピクセル列から環境マップとサンプリング用 CDF を構築する。
    pub fn from_pixels(w: usize, h: usize, data: Vec<Color>) -> Self {
        let mut row_cdf = vec![0.0; h + 1];
        let mut col_cdf = vec![0.0; h * (w + 1)];
        let mut total = 0.0;
        for y in 0..h {
            let theta0 = std::f64::consts::PI * (y as f64) / (h as f64);
            let theta1 = std::f64::consts::PI * ((y + 1) as f64) / (h as f64);
            let row_weight = (theta0.cos() - theta1.cos()).max(0.0);
            let row_offset = y * (w + 1);
            let mut row_sum = 0.0;
            col_cdf[row_offset] = 0.0;
            for x in 0..w {
                let c = data[y * w + x];
                let lum = c.luminance().max(0.0);
                row_sum += lum * row_weight;
                col_cdf[row_offset + x + 1] = row_sum;
            }
            total += row_sum;
            row_cdf[y + 1] = total;
        }
        Self {
            width: w,
            height: h,
            data,
            row_cdf,
            col_cdf,
            total_weight: total,
        }
    }

    /// 方向 `dir` から環境マップの放射輝度をバイリニア補間でサンプリングする。
    ///
    /// テクセル (x, y) は (u, v) ∈ [x/W, (x+1)/W) × [y/H, (y+1)/H) を覆い、その中心で値が
    /// テクセル値に一致する（`pdf` / CDF と同じ `u·W`, `v·H` 基準）。補間はテクセル中心間で行い、
    /// U（経度）方向は周期的に折り返し、V（緯度）方向は端の行でクランプする。
    pub fn sample(&self, dir: Vec3) -> Color {
        let (u, v, _theta) = dir_to_uv(dir);

        let x = u * (self.width as f64) - 0.5;
        let y = v * (self.height as f64) - 0.5;

        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let w = self.width as i64;
        let h = self.height as i64;
        let x0 = (xf as i64).rem_euclid(w) as usize;
        let x1 = (xf as i64 + 1).rem_euclid(w) as usize;
        let y0 = (yf as i64).clamp(0, h - 1) as usize;
        let y1 = (yf as i64 + 1).clamp(0, h - 1) as usize;
        // 端の行の外側（最初の行の中心より上・最後の行の中心より下）では y0 == y1 となり、
        // 補間係数に関係なくその行の値になる
        let fy = clamp(y - yf, 0.0, 1.0);

        let c00 = self.data[y0 * self.width + x0];
        let c10 = self.data[y0 * self.width + x1];
        let c01 = self.data[y1 * self.width + x0];
        let c11 = self.data[y1 * self.width + x1];

        let c0 = c00 * (1.0 - fx) + c10 * fx;
        let c1 = c01 * (1.0 - fx) + c11 * fx;
        c0 * (1.0 - fy) + c1 * fy
    }

    /// CDF ベースで方向を重点的サンプリングし、(方向, 放射輝度, PDF) を返す。
    pub fn sample_dir(&self, rng: &mut Rng) -> (Vec3, Color, f64) {
        if self.total_weight <= 0.0 {
            return sample_uniform_env(self, rng);
        }

        let r0 = rng.next_f64() * self.total_weight;
        let y = cdf_search(&self.row_cdf, r0).min(self.height - 1);
        let row_start = self.row_cdf[y];
        let row_end = self.row_cdf[y + 1];
        let row_sum = (row_end - row_start).max(0.0);
        if row_sum <= 0.0 {
            return sample_uniform_env(self, rng);
        }

        let r1 = rng.next_f64() * row_sum;
        let row_offset = y * (self.width + 1);
        let x = cdf_search(&self.col_cdf[row_offset..row_offset + self.width + 1], r1)
            .min(self.width - 1);

        let u = (x as f64 + rng.next_f64()) / (self.width as f64);
        let v = (y as f64 + rng.next_f64()) / (self.height as f64);
        let theta = std::f64::consts::PI * v;
        let phi = std::f64::consts::TAU * u;
        let sin_theta = theta.sin();
        let dir = Vec3::new(phi.cos() * sin_theta, theta.cos(), phi.sin() * sin_theta);
        let radiance = self.sample(dir);
        let pdf = self.pdf(dir);
        (dir, radiance, pdf)
    }

    /// 環境マップ分布における方向の PDF を返す。
    /// PDF(ω) = (lum × row_weight / total_weight × W × H) / (2π² sinθ)
    pub fn pdf(&self, dir: Vec3) -> f64 {
        if self.total_weight <= 0.0 {
            return 1.0 / (4.0 * std::f64::consts::PI);
        }
        let (u, v, theta) = dir_to_uv(dir);
        let x = (u * (self.width as f64)).floor().clamp(0.0, (self.width - 1) as f64) as usize;
        let y = (v * (self.height as f64)).floor().clamp(0.0, (self.height - 1) as f64) as usize;

        let theta0 = std::f64::consts::PI * (y as f64) / (self.height as f64);
        let theta1 = std::f64::consts::PI * ((y + 1) as f64) / (self.height as f64);
        let sin_theta = theta.sin().max(1e-6);
        let row_weight = (theta0.cos() - theta1.cos()).max(0.0);
        let lum = self.data[y * self.width + x].luminance().max(0.0);
        let weight = lum * row_weight;
        if weight <= 0.0 {
            return 0.0;
        }
        let pdf_uv = (weight / self.total_weight) * (self.width as f64) * (self.height as f64);
        pdf_uv / (2.0 * std::f64::consts::PI * std::f64::consts::PI * sin_theta)
    }
}

/// 方向を等距離円筒図法の (u, v) ＋ θ（天頂角）に変換する。`sample` と `pdf` で共有される、
/// `sample_dir` の (u,v)→方向マッピング（下記）の逆写像。
fn dir_to_uv(dir: Vec3) -> (f64, f64, f64) {
    let d = dir.norm();
    let theta = clamp(d.y, -1.0, 1.0).acos();
    let mut phi = d.z.atan2(d.x);
    if phi < 0.0 {
        phi += std::f64::consts::TAU;
    }
    let u = phi / std::f64::consts::TAU;
    let v = theta / std::f64::consts::PI;
    (u, v, theta)
}

/// 球面上の一様サンプリング（CDF が無効な場合のフォールバック）。
fn sample_uniform_env(env: &EnvMap, rng: &mut Rng) -> (Vec3, Color, f64) {
    let dir = uniform_sphere_dir(rng);
    let radiance = env.sample(dir);
    let pdf = 1.0 / (4.0 * std::f64::consts::PI);
    (dir, radiance, pdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sample_dir` が (u,v) から方向を作るのに使うマッピング（本体はインライン、
    /// ここではテスト用に複製）が `dir_to_uv` の逆写像であることを確認する。
    /// 両者が独立に発散すると MIS の重みが静かに壊れるため、明示的な往復テストを置く。
    fn uv_to_dir(u: f64, v: f64) -> Vec3 {
        let theta = std::f64::consts::PI * v;
        let phi = std::f64::consts::TAU * u;
        let sin_theta = theta.sin();
        Vec3::new(phi.cos() * sin_theta, theta.cos(), phi.sin() * sin_theta)
    }

    /// 各テクセル中心の方向では、`sample` はそのテクセル値をそのまま返す（`pdf` / CDF と同じ
    /// v·H 基準の写像）。旧実装は v·(H−1) 基準で、行が中心からずれていた。
    #[test]
    fn sample_returns_texel_values_at_texel_centers() {
        let (w, h) = (8usize, 5usize);
        let data: Vec<Color> = (0..w * h).map(|i| Color::new(i as f64, (i * 7 % 11) as f64, 1.0 + (i % 3) as f64)).collect();
        let env = EnvMap::from_pixels(w, h, data.clone());
        for y in 0..h {
            for x in 0..w {
                let dir = uv_to_dir((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                let c = env.sample(dir);
                let e = data[y * w + x];
                assert!((c.r() - e.r()).abs() < 1e-6 && (c.g() - e.g()).abs() < 1e-6 && (c.b() - e.b()).abs() < 1e-6,
                    "texel ({}, {}): got ({}, {}, {}) expected ({}, {}, {})", x, y, c.r(), c.g(), c.b(), e.r(), e.g(), e.b());
            }
        }
    }

    /// U 方向は周期的（u=0 の経線をまたいでも連続）、V 方向は極付近で端の行にクランプされる。
    /// テクセル中心の間は線形補間。
    #[test]
    fn sample_wraps_in_u_clamps_in_v_and_interpolates_linearly() {
        let (w, h) = (4usize, 3usize);
        let data: Vec<Color> = (0..w * h).map(|i| Color::new(i as f64, 0.0, 0.0)).collect();
        let env = EnvMap::from_pixels(w, h, data.clone());
        let r = |u: f64, v: f64| env.sample(uv_to_dir(u, v)).r();
        let row = 1usize;
        let v = (row as f64 + 0.5) / h as f64;
        // u = 0 の経線（テクセル W−1 と 0 の中心の中間）: 両者の平均。0 の直前と直後で連続
        let mid = 0.5 * (data[row * w + w - 1].r() + data[row * w].r());
        assert!((r(0.0, v) - mid).abs() < 1e-6, "seam value {} vs {}", r(0.0, v), mid);
        assert!((r(1e-9, v) - r(1.0 - 1e-9, v)).abs() < 1e-6, "discontinuity across the u seam");
        // テクセル中心 0 と 1 の 1/4 の位置は 3:1 の線形補間
        let u = (0.5 + 0.25) / w as f64;
        let expect = 0.75 * data[row * w].r() + 0.25 * data[row * w + 1].r();
        assert!((r(u, v) - expect).abs() < 1e-6, "interpolation {} vs {}", r(u, v), expect);
        // 極の近く（最初の行の中心より上、最後の行の中心より下）は端の行の値
        let u0 = 0.5 / w as f64;
        assert!((r(u0, 1e-6) - data[0].r()).abs() < 1e-6, "north pole clamp");
        assert!((r(u0, 1.0 - 1e-6) - data[(h - 1) * w].r()).abs() < 1e-6, "south pole clamp");
    }

    /// 方向について一様に平均した `sample` の放射輝度は、CDF と同じテクセル立体角重みで求めた
    /// 画像の平均に一致する（行のずれがあると明るい行の重みがずれて一致しない）。
    #[test]
    fn sample_integral_matches_texel_solid_angle_weights() {
        let (w, h) = (16usize, 8usize);
        // 緯度で大きく変わる画像（行ごとに 1, 2, 4, …）
        let data: Vec<Color> = (0..w * h).map(|i| { let y = i / w; Color::new((1u32 << y) as f64, 1.0, 1.0) }).collect();
        let env = EnvMap::from_pixels(w, h, data.clone());
        let mut rng = Rng::new(3);
        let n = 400_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += env.sample(uniform_sphere_dir(&mut rng)).r();
        }
        let mc = sum / n as f64 * 4.0 * std::f64::consts::PI;
        let mut exact = 0.0;
        for y in 0..h {
            let t0 = std::f64::consts::PI * y as f64 / h as f64;
            let t1 = std::f64::consts::PI * (y + 1) as f64 / h as f64;
            let omega = std::f64::consts::TAU * (t0.cos() - t1.cos());
            exact += (1u32 << y) as f64 * omega;
        }
        // 双線形補間は行の中間で隣の行と混ざるので完全一致はしないが、1〜2% の範囲に入る。
        // 旧実装（v·(H−1) 基準）は行の対応が系統的にずれ、この画像では約 −12% になる。
        let rel = mc / exact - 1.0;
        assert!(rel.abs() < 0.03, "∫sample dω {} vs texel-weighted {} (rel {:+.4})", mc, exact, rel);
        println!("integral rel error {:+.4}", rel);
    }

    #[test]
    fn dir_to_uv_round_trips_with_sample_dir_mapping() {
        for i in 0..7 {
            for j in 0..5 {
                let u = (i as f64 + 0.5) / 7.0;
                let v = (j as f64 + 0.5) / 5.0;
                let dir = uv_to_dir(u, v);
                let (u2, v2, _theta) = dir_to_uv(dir);
                assert!((u - u2).abs() < 1e-9, "u mismatch at ({u}, {v}): got {u2}");
                assert!((v - v2).abs() < 1e-9, "v mismatch at ({u}, {v}): got {v2}");
            }
        }
    }
}
