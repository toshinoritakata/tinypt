//! 手続き的な 3D ソリッドノイズ（Perlin の勾配ノイズと、そこから作るパターン）。
//!
//! **3D で交差点の座標を直接評価する**のが要点。2D のノイズを UV に貼ると、球の極（UV が潰れる）で模様が歪み、
//! UV の継ぎ目で不連続が出る。3D なら原理的にどちらも起きず、物体を「削り出した」見え方になる。
//!
//! **模様は座標に固定される**: シーンを k 倍すれば模様も k 倍になる（`scale` は空間周波数）。これは正しい挙動
//! （模様が物体に付いてくる）で、材質の見た目のスケール不変性とは別の話。座標は既定で**物体（ローカル）座標**
//! （`World::object_space_point`）なので、動く物体でも複数配置でも模様は物体に追従する。`space="world"` ならワールド座標。
//!
//! 置換表は `splitmix64` から **コンパイル時**に作る固定表（実行ごとに変わらない = レンダリングが再現する）。
//! これは暫定の一歩で、将来はマテリアルグラフのような統一的な仕組みに載せ替える想定（CONTEXT 参照）。

use crate::math::{Color, Vec3};
use crate::rng::splitmix64;

/// 置換表（0..255 の並べ替えを 2 周分）。Fisher–Yates を固定シードの splitmix64 で回して `const` で作る。
const PERM: [u8; 512] = {
    let mut p = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        p[i] = i as u8;
        i += 1;
    }
    let mut state = 0x5EED_0FA1_1u64;
    let mut i = 255;
    while i > 0 {
        state = splitmix64(state);
        let j = (state % (i as u64 + 1)) as usize;
        let t = p[i];
        p[i] = p[j];
        p[j] = t;
        i -= 1;
    }
    let mut out = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        out[i] = p[i & 255];
        i += 1;
    }
    out
};

#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn grad(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    // Perlin の改良版: 立方体の 12 辺方向の勾配（下位 4 ビットで選ぶ）
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// 3D Perlin ノイズ。値域は [-1, 1]、整数格子点で 0、連続（勾配ノイズ）。
pub fn perlin(p: Vec3) -> f64 {
    let (fx, fy, fz) = (p.x.floor(), p.y.floor(), p.z.floor());
    let (xi, yi, zi) = ((fx as i64 & 255) as usize, (fy as i64 & 255) as usize, (fz as i64 & 255) as usize);
    let (x, y, z) = (p.x - fx, p.y - fy, p.z - fz);
    let (u, v, w) = (fade(x), fade(y), fade(z));
    let a = PERM[xi] as usize + yi;
    let (aa, ab) = (PERM[a] as usize + zi, PERM[a + 1] as usize + zi);
    let b = PERM[xi + 1] as usize + yi;
    let (ba, bb) = (PERM[b] as usize + zi, PERM[b + 1] as usize + zi);
    let lerp = |t: f64, a: f64, b: f64| a + t * (b - a);
    lerp(
        w,
        lerp(
            v,
            lerp(u, grad(PERM[aa], x, y, z), grad(PERM[ba], x - 1.0, y, z)),
            lerp(u, grad(PERM[ab], x, y - 1.0, z), grad(PERM[bb], x - 1.0, y - 1.0, z)),
        ),
        lerp(
            v,
            lerp(u, grad(PERM[aa + 1], x, y, z - 1.0), grad(PERM[ba + 1], x - 1.0, y, z - 1.0)),
            lerp(u, grad(PERM[ab + 1], x, y - 1.0, z - 1.0), grad(PERM[bb + 1], x - 1.0, y - 1.0, z - 1.0)),
        ),
    )
    .clamp(-1.0, 1.0)
}

/// fBm: オクターブごとに周波数を `lacunarity` 倍・振幅を `gain` 倍して足す。**振幅の総和で割って**正規化するので
/// 値域は [-1, 1] のまま（オクターブを増やしても発散せず、`gain < 1` なら振幅の総和が収束する）。
pub fn fbm(p: Vec3, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let (mut sum, mut norm, mut amp, mut freq) = (0.0, 0.0, 1.0, 1.0);
    for _ in 0..octaves.max(1) {
        sum += amp * perlin(p * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / norm
}

/// 乱流: 各オクターブの `|perlin|` の（振幅で正規化した）和。値域 [0, 1]。
pub fn turbulence(p: Vec3, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let (mut sum, mut norm, mut amp, mut freq) = (0.0, 0.0, 1.0, 1.0);
    for _ in 0..octaves.max(1) {
        sum += amp * perlin(p * freq).abs();
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    sum / norm
}

/// パターン（`color0` と `color1` を混ぜる係数 t ∈ [0, 1] の作り方）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Fbm,
    Turbulence,
    Marble,
    Wood,
    Granite,
}

impl Pattern {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "fbm" => Pattern::Fbm,
            "turbulence" => Pattern::Turbulence,
            "marble" => Pattern::Marble,
            "wood" => Pattern::Wood,
            "granite" => Pattern::Granite,
            _ => return None,
        })
    }
}

/// 3D ソリッドノイズのテクスチャ（画像の [`crate::texture::Texture`] とは別物。`Scene::noises` に置く）。
#[derive(Clone, Copy, Debug)]
pub struct NoiseTexture {
    pub pattern: Pattern,
    /// 空間周波数（座標に掛ける倍率）。正の有限値
    pub scale: f64,
    pub octaves: u32,
    pub lacunarity: f64,
    pub gain: f64,
    /// marble / wood の歪み（乱流の掛かり方）
    pub strength: f64,
    pub color0: Color,
    pub color1: Color,
    /// `true` なら物体（ローカル）座標、`false` ならワールド座標で評価する（既定 `true`）。
    /// ローカルだと模様が物体に追従する（動く物体・複数配置でも「その物体から削り出した」見え方）。
    pub local: bool,
    /// 評価前に座標へ足す（ローカル座標では、同じ形の物体ごとに切り口を変える手段）
    pub offset: Vec3,
}

impl NoiseTexture {
    /// 不正な値は安全な範囲に丸める（呼び出し側が警告する）: `scale` は (0, 1e6] の有限値（他は 1）、`octaves` は 1..=10、
    /// `lacunarity` は [1, 8]（他は 2）、`gain` は [0, 1]（他は 0.5）、`strength` は有限値（他は 1）。
    pub fn sanitized(mut self) -> Self {
        if !(self.scale.is_finite() && self.scale > 0.0) {
            self.scale = 1.0;
        }
        self.scale = self.scale.min(1e6);
        self.octaves = self.octaves.clamp(1, 10);
        self.lacunarity = if self.lacunarity.is_finite() { self.lacunarity.clamp(1.0, 8.0) } else { 2.0 };
        self.gain = if self.gain.is_finite() { self.gain.clamp(0.0, 1.0) } else { 0.5 };
        if !self.strength.is_finite() {
            self.strength = 1.0;
        }
        self
    }

    /// 混ぜる係数 t ∈ [0, 1]。`p` はワールド空間の点。
    pub fn factor(&self, p: Vec3) -> f64 {
        if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
            return 0.0;
        }
        let q = (p + self.offset) * self.scale;
        let (o, l, g) = (self.octaves, self.lacunarity, self.gain);
        let t = match self.pattern {
            Pattern::Fbm => 0.5 + 0.5 * fbm(q, o, l, g),
            Pattern::Turbulence => turbulence(q, o, l, g),
            // 縞（x 方向、周期 2 単位）を乱流で歪ませた大理石の脈
            Pattern::Marble => 0.5 + 0.5 * (q.x * std::f64::consts::PI + turbulence(q, o, l, g) * self.strength).sin(),
            // y 軸まわりの年輪（のこぎり波）を乱流で歪ませた木目
            Pattern::Wood => {
                let v = (q.x * q.x + q.z * q.z).sqrt() + turbulence(q, o, l, g) * self.strength;
                v - v.floor()
            }
            // 高周波の fBm のコントラストを引き上げた粒状の模様
            Pattern::Granite => 0.5 + (fbm(q * 3.0, o, l, g) * 0.5) * 4.0,
        };
        if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) }
    }

    /// 色: `color0` と `color1` の線形補間。
    pub fn eval(&self, p: Vec3) -> Color {
        let t = self.factor(p);
        self.color0 * (1.0 - t) + self.color1 * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z)
    }
    fn lcg(s: &mut u64) -> f64 {
        *s = splitmix64(*s);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }
    fn tex(pattern: Pattern) -> NoiseTexture {
        NoiseTexture { pattern, scale: 1.0, octaves: 4, lacunarity: 2.0, gain: 0.5, strength: 3.0, color0: Color::new(0.0, 0.0, 0.0), color1: Color::new(1.0, 1.0, 1.0), local: true, offset: Vec3::new(0.0, 0.0, 0.0) }
    }

    #[test]
    fn perlin_is_bounded_and_zero_at_lattice_points() {
        let mut s = 1;
        for _ in 0..20_000 {
            let p = v(lcg(&mut s) * 200.0 - 100.0, lcg(&mut s) * 200.0 - 100.0, lcg(&mut s) * 200.0 - 100.0);
            assert!(perlin(p).abs() <= 1.0);
        }
        for (x, y, z) in [(0.0, 0.0, 0.0), (3.0, -2.0, 7.0), (-40.0, 5.0, 300.0)] {
            assert_eq!(perlin(v(x, y, z)), 0.0);
        }
        // 置換表は 0..255 の並べ替え
        let mut seen = [false; 256];
        for i in 0..256 {
            seen[PERM[i] as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    /// 固定表なので、同じ座標は常に同じ値（プロセスをまたいでも。下の値は実装の固定値の記録）。
    #[test]
    fn deterministic_across_calls_and_processes() {
        let p = v(0.31, 1.7, -2.4);
        assert_eq!(perlin(p).to_bits(), perlin(p).to_bits());
        assert_eq!(PERM[0..4], PERM[256..260]);
        assert_eq!(PERM[0], EXPECTED_PERM0, "置換表が変わった（レンダリングの再現性が壊れる）");
        assert_eq!(perlin(p).to_bits(), EXPECTED_PERLIN_BITS, "ノイズの値が変わった: {:#x}", perlin(p).to_bits());
    }
    const EXPECTED_PERM0: u8 = 28;
    const EXPECTED_PERLIN_BITS: u64 = 0xbfc6a5e938913a30;

    #[test]
    fn fbm_converges_and_is_continuous() {
        let p = v(0.37, 0.61, 0.13);
        let vals: Vec<f64> = (1..=10).map(|o| fbm(p, o, 2.0, 0.5)).collect();
        assert!(vals.iter().all(|x| x.abs() <= 1.0));
        // gain = 0.5: 追加オクターブの寄与は 2^-o で減衰する（収束）
        assert!((vals[9] - vals[8]).abs() < (vals[2] - vals[1]).abs() + 1e-12);
        assert!((vals[9] - vals[8]).abs() < 0.01);
        // 連続性: 近い 2 点の差は小さい（勾配ノイズ）
        let mut s = 7;
        for _ in 0..5000 {
            let a = v(lcg(&mut s) * 20.0, lcg(&mut s) * 20.0, lcg(&mut s) * 20.0);
            let b = a + v(1e-4, -1e-4, 1e-4);
            assert!((perlin(a) - perlin(b)).abs() < 2e-3);
        }
    }

    #[test]
    fn patterns_stay_in_unit_range_and_bad_params_do_not_panic() {
        let mut s = 3;
        for pat in [Pattern::Fbm, Pattern::Turbulence, Pattern::Marble, Pattern::Wood, Pattern::Granite] {
            let t = tex(pat);
            for _ in 0..5000 {
                let p = v(lcg(&mut s) * 60.0 - 30.0, lcg(&mut s) * 60.0 - 30.0, lcg(&mut s) * 60.0 - 30.0);
                let f = t.factor(p);
                assert!((0.0..=1.0).contains(&f), "{pat:?} {f}");
            }
            for (scale, octaves, lac, gain, strength) in [(0.0, 0, 2.0, 0.5, 1.0), (-3.0, 99, f64::NAN, 5.0, f64::INFINITY), (f64::NAN, 10, 0.0, -1.0, 0.0), (1e300, 1, 1e9, 0.5, 1e300)] {
                let bad = NoiseTexture { scale, octaves, lacunarity: lac, gain, strength, ..t }.sanitized();
                assert!((1..=10).contains(&bad.octaves) && bad.scale > 0.0 && bad.scale.is_finite());
                let c = bad.eval(v(0.3, 0.4, 0.5));
                assert!(c.0.x.is_finite(), "{pat:?}");
            }
            assert_eq!(t.factor(v(f64::NAN, 0.0, 1.0)), 0.0);
        }
        assert_eq!(Pattern::parse("marble"), Some(Pattern::Marble));
        assert_eq!(Pattern::parse("nope"), None);
    }
}
