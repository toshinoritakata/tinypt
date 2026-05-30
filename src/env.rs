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
use crate::math::{clamp, Color, Vec3};
use crate::rng::Rng;

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
    pub fn sample(&self, dir: Vec3) -> Color {
        let d = dir.norm();
        let theta = clamp(d.y, -1.0, 1.0).acos();
        let mut phi = d.z.atan2(d.x);
        if phi < 0.0 {
            phi += std::f64::consts::TAU;
        }
        let u = phi / std::f64::consts::TAU;
        let v = theta / std::f64::consts::PI;

        let x = u * (self.width as f64);
        let y = v * (self.height as f64 - 1.0);

        let x0 = (x.floor() as usize) % self.width;
        let x1 = (x0 + 1) % self.width;
        let y0 = clamp(y, 0.0, (self.height - 1) as f64).floor() as usize;
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x.floor();
        let fy = y - y.floor();

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
        let d = dir.norm();
        let theta = clamp(d.y, -1.0, 1.0).acos();
        let mut phi = d.z.atan2(d.x);
        if phi < 0.0 {
            phi += std::f64::consts::TAU;
        }
        let u = phi / std::f64::consts::TAU;
        let v = theta / std::f64::consts::PI;
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

fn cdf_search(cdf: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = cdf.len().saturating_sub(1);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// 球面上の一様サンプリング（CDF が無効な場合のフォールバック）。
fn sample_uniform_env(env: &EnvMap, rng: &mut Rng) -> (Vec3, Color, f64) {
    let u = rng.next_f64();
    let v = rng.next_f64();
    let z = 1.0 - 2.0 * u;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * v;
    let dir = Vec3::new(r * phi.cos(), z, r * phi.sin());
    let radiance = env.sample(dir);
    let pdf = 1.0 / (4.0 * std::f64::consts::PI);
    (dir, radiance, pdf)
}
