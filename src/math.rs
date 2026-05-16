//! ベクトル・色・数学ユーティリティ。
//!
//! - `Vec3`: 3D ベクトル（位置・方向・法線に使用）
//! - `Color`: リニア RGB 色空間のラッパー（`Vec3` を内包し意味的な区別を提供）
//! - 反射・屈折・sRGB→リニア変換などのユーティリティ関数

#[derive(Clone, Copy, Debug, Default)]
/// 3 次元ベクトル。位置・方向・法線に汎用的に使用する。
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// リニア空間の RGB 色。`Vec3` のラッパーで型安全性を提供する。
///
/// レンダラー内部では常にリニア空間で計算し、
/// 出力時に sRGB ガンマ補正や ACEScg 変換を適用する。
#[derive(Clone, Copy, Debug, Default)]
pub struct Color(pub Vec3);

impl Color {
    /// リニア RGB 成分から色を生成する。
    pub fn new(r: f64, g: f64, b: f64) -> Self { Self(Vec3::new(r, g, b)) }

    /// sRGB 成分からリニア空間に変換して色を生成する。
    pub fn from_srgb(r: f64, g: f64, b: f64) -> Self {
        Self(Vec3::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)))
    }

    /// 成分ごとの乗算（色の掛け合わせ、アルベドの適用など）。
    pub fn hadamard(self, o: Color) -> Color { Color(self.0.hadamard(o.0)) }
    /// R チャネルを返す。
    pub fn r(self) -> f64 { self.0.x }
    /// G チャネルを返す。
    pub fn g(self) -> f64 { self.0.y }
    /// B チャネルを返す。
    pub fn b(self) -> f64 { self.0.z }
    /// Rec.709 係数による輝度（Y）を計算する。
    pub fn luminance(self) -> f64 { 0.2126 * self.r() + 0.7152 * self.g() + 0.0722 * self.b() }
    /// 輝度ベースでクランプし、ファイアフライ（異常に明るいサンプル）を抑制する。
    pub fn clamp_luminance(self, max_luma: f64) -> Color {
        let l = self.luminance();
        if l > max_luma && l > 0.0 {
            self * (max_luma / l)
        } else {
            self
        }
    }
    /// 各チャネルを [0, 1] にクランプする。
    pub fn clamp01(self) -> Color { Color(self.0.clamp01()) }
}
impl Vec3 {
    /// 成分を指定してベクトルを生成する。
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    /// 内積（ドット積）。
    pub fn dot(self, b: Vec3) -> f64 { self.x*b.x + self.y*b.y + self.z*b.z }
    /// 外積（クロス積）。
    pub fn cross(self, b: Vec3) -> Vec3 {
        Vec3::new(
            self.y*b.z - self.z*b.y,
            self.z*b.x - self.x*b.z,
            self.x*b.y - self.y*b.x,
        )
    }
    /// ユークリッド長（ノルム）。
    pub fn len(self) -> f64 { self.dot(self).sqrt() }
    /// 正規化ベクトルを返す（ゼロ長付近でも安全）。
    pub fn norm(self) -> Vec3 {
        let l = self.len().max(1e-30);
        self / l
    }
    /// Component-wise max with scalar.
    pub fn max(self, s: f64) -> Vec3 { Vec3::new(self.x.max(s), self.y.max(s), self.z.max(s)) }
    /// Component-wise min with scalar.
    pub fn min(self, s: f64) -> Vec3 { Vec3::new(self.x.min(s), self.y.min(s), self.z.min(s)) }
    /// Clamps each component into [0, 1].
    pub fn clamp01(self) -> Vec3 { self.max(0.0).min(1.0) }
    /// Component-wise multiplication.
    pub fn hadamard(self, b: Vec3) -> Vec3 { Vec3::new(self.x*b.x, self.y*b.y, self.z*b.z) }
}
use std::ops::{Add, Sub, Mul, Div, Neg};
impl Add for Vec3 { type Output = Vec3; fn add(self, b: Vec3) -> Vec3 { Vec3::new(self.x+b.x, self.y+b.y, self.z+b.z) } }
impl Sub for Vec3 { type Output = Vec3; fn sub(self, b: Vec3) -> Vec3 { Vec3::new(self.x-b.x, self.y-b.y, self.z-b.z) } }
impl Mul<f64> for Vec3 { type Output = Vec3; fn mul(self, s: f64) -> Vec3 { Vec3::new(self.x*s, self.y*s, self.z*s) } }
impl Div<f64> for Vec3 { type Output = Vec3; fn div(self, s: f64) -> Vec3 { Vec3::new(self.x/s, self.y/s, self.z/s) } }
impl Mul<Vec3> for f64 { type Output = Vec3; fn mul(self, v: Vec3) -> Vec3 { v * self } }
impl Neg for Vec3 { type Output = Vec3; fn neg(self) -> Vec3 { Vec3::new(-self.x, -self.y, -self.z) } }

impl Add for Color { type Output = Color; fn add(self, b: Color) -> Color { Color(self.0 + b.0) } }
impl Sub for Color { type Output = Color; fn sub(self, b: Color) -> Color { Color(self.0 - b.0) } }
impl Mul<f64> for Color { type Output = Color; fn mul(self, s: f64) -> Color { Color(self.0 * s) } }
impl Div<f64> for Color { type Output = Color; fn div(self, s: f64) -> Color { Color(self.0 / s) } }
impl Mul<Color> for f64 { type Output = Color; fn mul(self, c: Color) -> Color { Color(c.0 * self) } }
impl From<Vec3> for Color { fn from(v: Vec3) -> Color { Color(v) } }
impl From<Color> for Vec3 { fn from(c: Color) -> Vec3 { c.0 } }

/// スカラーを [a, b] にクランプする。
pub fn clamp(x: f64, a: f64, b: f64) -> f64 { x.max(a).min(b) }

/// sRGB の 1 成分をリニアに変換する（IEC 61966-2-1 規格に準拠）。
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
/// ベクトル `v` を法線 `n` に対して反射する: R = v - 2(v·n)n
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 { v - 2.0 * v.dot(n) * n }
/// Snell の法則に基づく屈折。全反射の場合は `None` を返す。
///
/// - `v`: 入射ベクトル（表面に向かう方向）
/// - `n`: 外向き法線（単位ベクトル）
/// - `eta`: 屈折率比 η_i / η_t
pub fn refract(v: Vec3, n: Vec3, eta: f64) -> Option<Vec3> {
    let cosi = (-v).dot(n).max(-1.0).min(1.0);
    let sin2t = eta * eta * (1.0 - cosi * cosi);
    if sin2t >= 1.0 {
        return None; // total internal reflection
    }
    let cost = (1.0 - sin2t).sqrt();
    Some(eta * v + (eta * cosi - cost) * n)
}
