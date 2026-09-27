//! 画素の再構成フィルタ（アンチエイリアスの質）。
//!
//! 画素 (x, y) の値は、画素の中心からのずれ `o` に掛かるフィルタ `f(o)` で重み付けした放射輝度の平均
//! `∫ f(o)·L(x + 0.5 + o) do / ∫ f(o) do`。**フィルタ重点サンプリング（FIS）**で推定する: 各軸を `|f|` に比例して
//! サンプルし（可分なので軸ごとに独立）、負のローブを持つフィルタだけ符号と正規化の重みを掛ける。
//! サンプルは画素の外に落ちてよいが、蓄積は常に**そのサンプルを撃った画素**の分（タイルの処理は変わらない）。
//!
//! - **box**: 半径 0.5 の一様。従来と同じ（`x + jx`）。表を使わないので出力はビット単位で従来のまま。
//! - **tent**: 半径 1 の三角形 `1 − |x|`。
//! - **gaussian**: Mitsuba と同じ定義。`stddev`（既定 0.5）、半径 `4·stddev`、裾を引いて半径で 0 になる
//!   `max(0, exp(−x²/2σ²) − exp(−r²/2σ²))`。
//! - **mitchell**: Mitchell–Netravali（`B`, `C`、既定 1/3・1/3）、半径 2。`1 < |x| < 2` に負のローブがある。
//!
//! ## 重み（推定量は不偏）
//! 1 軸の推定量は `|f|` からサンプルした `o` に対し `sign(f(o))·(∫|f| / ∫f)·L`。2 軸の積では
//! `sign(f(ox))·sign(f(oy))·(∫|f| / ∫f)²`。box・tent・gaussian は `f ≥ 0` なので重みはちょうど 1。
//! Mitchell の重みは符号つきで、平均はほぼ 1（テスト参照）。蓄積は従来どおりサンプル数で割る（重みの和では割らない）ので、
//! 適応的サンプリングの Welford 分散と `acc_w`（サンプル数）は符号つきの重みでも正しい（分散は重みつきサンプル `w·L` の分散）。
//! 負のローブの影響で境界付近の画素値が負になりうる。**負の値は出力の直前（`resolve_pixels`）で 0 に切る**（Mitsuba と同じ。
//! 蓄積・チェックポイントは符号つきのまま）。
//!
//! ## サンプリング
//! `|f|` の 1 次元の逆 CDF を、レンダーごとに 1 度だけ作った表（[`TABLE_N`] 区間 + 線形補間）で引く。入力は従来のジッターと同じ
//! Sobol 次元 0, 1 なので層化は保たれる。

/// 画素フィルタの種類と（あれば）パラメータ。
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum PixelFilter {
    /// 半径 0.5 の一様（従来の挙動。既定）
    #[default]
    Box,
    /// 半径 1 の三角形
    Tent,
    /// Mitsuba のガウス（半径 `4·stddev`、裾を引く）
    Gaussian { stddev: f64 },
    /// Mitchell–Netravali（半径 2）
    Mitchell { b: f64, c: f64 },
}

/// 表の区間数。
pub const TABLE_N: usize = 1024;
/// 表の 1 区間あたりの数値積分の細分数。
const SUBSTEPS: usize = 8;

impl PixelFilter {
    /// 名前（`--filter` と XML の `type`）から、既定のパラメータで作る。
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "box" => PixelFilter::Box,
            "tent" => PixelFilter::Tent,
            "gaussian" => PixelFilter::Gaussian { stddev: 0.5 },
            "mitchell" => PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 },
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            PixelFilter::Box => "box",
            PixelFilter::Tent => "tent",
            PixelFilter::Gaussian { .. } => "gaussian",
            PixelFilter::Mitchell { .. } => "mitchell",
        }
    }

    /// フィルタの半径（画素単位。1 軸あたり `[-r, r]`）。
    pub fn radius(&self) -> f64 {
        match *self {
            PixelFilter::Box => 0.5,
            PixelFilter::Tent => 1.0,
            PixelFilter::Gaussian { stddev } => 4.0 * stddev,
            PixelFilter::Mitchell { .. } => 2.0,
        }
    }

    /// 1 軸のフィルタ値 `f(x)`（半径の外は 0。box は 1、値は正規化しない）。
    pub fn eval(&self, x: f64) -> f64 {
        let ax = x.abs();
        if ax > self.radius() {
            return 0.0;
        }
        match *self {
            PixelFilter::Box => 1.0,
            PixelFilter::Tent => (1.0 - ax).max(0.0),
            PixelFilter::Gaussian { stddev } => {
                let alpha = -1.0 / (2.0 * stddev * stddev);
                let r = self.radius();
                ((alpha * ax * ax).exp() - (alpha * r * r).exp()).max(0.0)
            }
            PixelFilter::Mitchell { b, c } => {
                let x2 = ax * ax;
                let x3 = x2 * ax;
                if ax < 1.0 {
                    ((12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b)) / 6.0
                } else {
                    ((-b - 6.0 * c) * x3 + (6.0 * b + 30.0 * c) * x2 + (-12.0 * b - 48.0 * c) * ax + (8.0 * b + 24.0 * c)) / 6.0
                }
            }
        }
    }
}

/// 1 軸の `|f|` の逆 CDF の表と、負のローブの重み。レンダーごとに [`FilterSampler::new`] で 1 度だけ作る。
pub struct FilterSampler {
    filter: PixelFilter,
    radius: f64,
    /// `cdf[i]` = 区間 `[−r, −r + 2r·i/N]` の `∫|f|`（全体で 1 に正規化）。長さ `TABLE_N + 1`
    cdf: Vec<f64>,
    /// `(∫|f| / ∫f)²`（負のローブが無ければちょうど 1）
    weight: f64,
}

impl FilterSampler {
    /// box なら `None`（従来の一様ジッターを使う）。それ以外は表を作る。
    pub fn new(filter: PixelFilter) -> Option<Self> {
        if filter == PixelFilter::Box {
            return None;
        }
        let r = filter.radius();
        let cell = 2.0 * r / TABLE_N as f64;
        let h = cell / SUBSTEPS as f64;
        let mut cdf = Vec::with_capacity(TABLE_N + 1);
        let (mut abs_sum, mut signed_sum) = (0.0f64, 0.0f64);
        cdf.push(0.0);
        for i in 0..TABLE_N {
            for k in 0..SUBSTEPS {
                let x = -r + cell * i as f64 + h * (k as f64 + 0.5);
                let f = filter.eval(x);
                abs_sum += f.abs() * h;
                signed_sum += f * h;
            }
            cdf.push(abs_sum);
        }
        for v in cdf.iter_mut() {
            *v /= abs_sum;
        }
        let ratio = if abs_sum == signed_sum { 1.0 } else { abs_sum / signed_sum };
        Some(Self { filter, radius: r, cdf, weight: ratio * ratio })
    }

    /// 1 軸: 一様乱数 `u ∈ [0, 1)` から `|f|` に比例した位置 `o ∈ [−r, r]` を引く。
    pub fn sample_axis(&self, u: f64) -> f64 {
        // cdf[i] <= u < cdf[i + 1] の i を二分探索
        let i = self.cdf.partition_point(|&c| c <= u).saturating_sub(1).min(TABLE_N - 1);
        let (lo, hi) = (self.cdf[i], self.cdf[i + 1]);
        let t = if hi > lo { ((u - lo) / (hi - lo)).clamp(0.0, 1.0) } else { 0.5 };
        -self.radius + 2.0 * self.radius * (i as f64 + t) / TABLE_N as f64
    }

    /// 2 軸: 次元 0, 1 の乱数から、画素中心からのずれ `(ox, oy)` と、そのサンプルの重み
    /// （`sign(f(ox))·sign(f(oy))·(∫|f| / ∫f)²`。負のローブが無ければちょうど 1）。
    pub fn sample(&self, ux: f64, uy: f64) -> (f64, f64, f64) {
        let (ox, oy) = (self.sample_axis(ux), self.sample_axis(uy));
        let sign = |x: f64| if self.filter.eval(x) < 0.0 { -1.0 } else { 1.0 };
        (ox, oy, sign(ox) * sign(oy) * self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> [PixelFilter; 4] {
        [PixelFilter::Box, PixelFilter::Tent, PixelFilter::Gaussian { stddev: 0.5 }, PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 }]
    }

    #[test]
    fn filter_definitions_match_mitsuba() {
        let g = PixelFilter::Gaussian { stddev: 0.5 };
        assert_eq!(g.radius(), 2.0);
        assert!(g.eval(2.0).abs() < 1e-15, "gaussian reaches 0 at the radius");
        assert!(g.eval(0.0) > 0.98 && g.eval(0.0) < 1.0, "the tail (exp(-8)) is subtracted: {}", g.eval(0.0));
        assert!((PixelFilter::Gaussian { stddev: 0.3 }.radius() - 1.2).abs() < 1e-12);
        assert_eq!(PixelFilter::Tent.eval(0.25), 0.75);
        assert_eq!(PixelFilter::Tent.eval(1.5), 0.0);
        let m = PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 };
        assert!((m.eval(0.0) - (6.0 - 2.0 / 3.0) / 6.0).abs() < 1e-12);
        assert!(m.eval(1.5) < 0.0, "negative lobe");
        assert!(m.eval(2.0).abs() < 1e-12, "continuous at the radius");
        assert!((m.eval(1.0 - 1e-9) - m.eval(1.0 + 1e-9)).abs() < 1e-6, "continuous at |x| = 1");
        // 積分は 1（正規化されたフィルタ）
        let n = 40000;
        let integral: f64 = (0..n).map(|i| m.eval(-2.0 + 4.0 * (i as f64 + 0.5) / n as f64) * 4.0 / n as f64).sum();
        assert!((integral - 1.0).abs() < 1e-6, "{integral}");
        assert_eq!(PixelFilter::from_name("Gaussian"), Some(g));
        assert_eq!(PixelFilter::from_name("lanczos"), None);
    }

    /// サンプルしたずれのヒストグラムが `|f|` を正規化した確率に一致する（表の逆引きが正しい）。
    #[test]
    fn sampled_offsets_follow_the_filter_magnitude() {
        for filter in all().into_iter().skip(1) {
            let s = FilterSampler::new(filter).unwrap();
            let r = filter.radius();
            let bins = 32usize;
            let mut hist = vec![0.0f64; bins];
            let n = 400_000;
            for i in 0..n {
                let u = (i as f64 + 0.5) / n as f64; // 一様な格子（層化）
                let o = s.sample_axis(u);
                assert!(o >= -r && o <= r, "{filter:?}: offset {o} outside the radius");
                let b = (((o + r) / (2.0 * r)) * bins as f64) as usize;
                hist[b.min(bins - 1)] += 1.0 / n as f64;
            }
            // 期待: 各ビンの ∫|f| / 全体
            let sub = 200;
            let expect: Vec<f64> = (0..bins)
                .map(|b| (0..sub).map(|k| filter.eval(-r + 2.0 * r * (b as f64 + (k as f64 + 0.5) / sub as f64) / bins as f64).abs()).sum::<f64>())
                .collect();
            let total: f64 = expect.iter().sum();
            for b in 0..bins {
                assert!((hist[b] - expect[b] / total).abs() < 1.5e-3, "{filter:?} bin {b}: {} vs {}", hist[b], expect[b] / total);
            }
        }
    }

    /// 重み: box・tent・gaussian は 1、Mitchell は符号つきで平均がほぼ 1（不偏）。
    #[test]
    fn weights_are_one_except_mitchell_whose_mean_is_one() {
        for filter in [PixelFilter::Tent, PixelFilter::Gaussian { stddev: 0.5 }] {
            let s = FilterSampler::new(filter).unwrap();
            for i in 0..100 {
                let u = (i as f64 + 0.5) / 100.0;
                assert_eq!(s.sample(u, 1.0 - u).2, 1.0);
            }
        }
        assert!(FilterSampler::new(PixelFilter::Box).is_none());
        let s = FilterSampler::new(PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 }).unwrap();
        let (mut sum, mut neg, n) = (0.0, 0usize, 600usize);
        for i in 0..n {
            for j in 0..n {
                let (_, _, w) = s.sample((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
                sum += w;
                neg += (w < 0.0) as usize;
            }
        }
        let mean = sum / (n * n) as f64;
        assert!((mean - 1.0).abs() < 5e-3, "mean weight {mean}");
        assert!(neg > 0, "the negative lobe must produce some negative weights");
    }
}
