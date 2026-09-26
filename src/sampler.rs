//! Owen スクランブル付き Sobol 列のサンプラー（Burley 2020, "Practical Hash-based Owen Scrambling"）。
//!
//! **仕組み**: 1 画素の `N` 本のサンプル（サンプル番号 `index = 0..N`）が、パスの各「次元」を同じ点集合で層化する。
//! 次元 `d` は 2 次元ずつ組（ペア `d / 2`）にして、ペアごとに
//! 1. サンプル番号を**入れ子の一様スクランブル**でシャッフルし（ペア間の相関を消す。先頭 2^k 個は 2^k 個のままの集合）、
//! 2. (0,2) 列（第 0 次元 = ビット反転の van der Corput、第 1 次元 = Sobol の第 2 次元）の点を作り、
//! 3. 各成分に**ハッシュベースの Owen スクランブル**（Laine–Karras の入れ子の一様置換）を掛ける。
//!
//! 方向数ベクトルは表を持たず、(0,2) 列の 2 つの生成行列（ビット反転と `v ^= v >> 1`）だけを使う（実行ごとに変わらない
//! 固定のアルゴリズム。Joe–Kuo の表は要らない）。ペアごとにシャッフルとスクランブルの種が独立なので、次元数は
//! 上限（[`MAX_DIMS`]）まで事実上いくらでも取れる。スクランブルの種は画素座標とユーザーシード（`--seed`）から決まる。
//!
//! **次元の割り当て**（[`Rng::set_dim`] で「今から引く乱数の次元」を明示する。役割ごとに次元が固定されるので、
//! 分岐で途中の乱数消費量が変わっても、同じ役割は常に同じ次元から引かれる）:
//!
//! | 次元 | 役割 |
//! |---|---|
//! | 0, 1 | 画素内のジッター |
//! | 2, 3 | レンズ（被写界深度。同心円写像で 2 次元ちょうど） |
//! | 4 | 時刻（モーションブラー） |
//! | 5 | （予備） |
//! | `6 + 16b + 0, 1` | バウンス `b`: 参加媒質の距離（チャンネル選択, 距離） |
//! | `+ 2, 3` | 媒質の位相関数の方向 |
//! | `+ 4, 5` | 環境マップの NEE（方向） |
//! | `+ 6` | 面光源の選択（`+ 7` は予備） |
//! | `+ 8, 9` | 選んだ光源面上の点 |
//! | `+ 10` | Russian roulette（`+ 11` は予備） |
//! | `+ 12, 13, 14` | BSDF（`12, 13` = 方向 / 屈折の選択 / 法線、`14` = 追加の 1 個） |
//! | `+ 15` | （予備） |
//!
//! **上限**: バウンス [`MAX_BOUNCES`] = 8 まで（`6 + 16 × 8 = 134` 次元 = 67 ペア）。それより深いバウンス（と、
//! 各役割の枠を超えた乱数）は、既存の PCG（[`Rng`]）にフォールバックする。深いバウンスは Russian roulette で
//! 生き残る確率が低く、寄与も小さいので層化の効果がほとんど無いため。1 ペアの計算は 1 次元あたり数十 ns で、
//! 上限を大きくしても遅くなるのは深いパスだけだが、深いパスは稀なので 8 で十分と判断した。
//!
//! **不偏性**: 各サンプルの各次元は（Owen スクランブルにより）[0,1) の一様分布に従う。分岐で使う次元が変わっても、
//! 1 サンプルの点は一様なので、期待値は変わらない（層化は分散を下げるだけ）。


/// Sobol で層化するバウンスの数。これ以降は PCG。
pub const MAX_BOUNCES: u32 = 8;
/// パスの最初（画素・レンズ・時刻）が使う次元数。
pub const FIRST_BOUNCE_DIM: u32 = 6;
/// 1 バウンスが予約する次元数。
pub const BOUNCE_STRIDE: u32 = 16;
/// Sobol で扱う次元の総数。これ以上は PCG。
pub const MAX_DIMS: u32 = FIRST_BOUNCE_DIM + BOUNCE_STRIDE * MAX_BOUNCES;

/// バウンス内の役割ごとのオフセット（[`bounce_dim`] に足す）。
pub mod role {
    pub const MEDIUM_DISTANCE: u32 = 0;
    pub const MEDIUM_PHASE: u32 = 2;
    pub const NEE_ENV: u32 = 4;
    pub const NEE_LIGHT_SELECT: u32 = 6;
    pub const NEE_LIGHT_POINT: u32 = 8;
    pub const ROULETTE: u32 = 10;
    pub const BSDF: u32 = 12;
}
/// 画素ジッター・レンズ・時刻の次元。
pub mod first {
    pub const LENS: u32 = 2;
    pub const TIME: u32 = 4;
}

/// バウンス `b` の役割 `offset` の次元。上限を超えるバウンスは `u32::MAX`（= PCG にフォールバック）。
#[inline]
pub fn bounce_dim(bounce: usize, offset: u32) -> u32 {
    if bounce >= MAX_BOUNCES as usize {
        u32::MAX
    } else {
        FIRST_BOUNCE_DIM + BOUNCE_STRIDE * bounce as u32 + offset
    }
}

/// 32 ビットの整数ハッシュ（MurmurHash3 の finalizer。全単射でアバランシェが良い）。
#[inline]
pub fn mix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// ハッシュ結合。`seed` は画素ごとに十分混ざった値（呼び出し側が `mix32` / splitmix で作る）、`v` は連番の添字
/// （黄金比の定数を掛けて散らしてから `mix32` する）。
#[inline]
fn hash(seed: u32, v: u32) -> u32 {
    mix32(seed.wrapping_add(v.wrapping_mul(0x9E37_79B9)))
}

/// Sobol の第 2 次元の生成行列（GF(2) 上の 32x32）を、入力の 8 ビットごとの表 4 枚にしたもの。
/// `SOBOL1[j][b]` = 入力のバイト `j`（下位から）が値 `b` のときの寄与。行列は `v_k = v_{k-1} ^ (v_{k-1} >> 1)`、`v_0 = 2^31`。
const SOBOL1: [[u32; 256]; 4] = {
    let mut cols = [0u32; 32];
    let mut v = 1u32 << 31;
    let mut k = 0;
    while k < 32 {
        cols[k] = v;
        v ^= v >> 1;
        k += 1;
    }
    let mut t = [[0u32; 256]; 4];
    let mut j = 0;
    while j < 4 {
        let mut b = 0;
        while b < 256 {
            let mut r = 0u32;
            let mut bit = 0;
            while bit < 8 {
                if (b >> bit) & 1 != 0 {
                    r ^= cols[8 * j + bit];
                }
                bit += 1;
            }
            t[j][b] = r;
            b += 1;
        }
        j += 1;
    }
    t
};

/// Laine–Karras の入れ子の一様置換（ビット反転した値に対する Owen スクランブル）。
#[inline]
fn laine_karras(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// 入れ子の一様スクランブル（= Owen スクランブル）。上位ビットほど下位ビットの置換を決めるので、
/// ビット反転して Laine–Karras を掛け、戻す。
#[inline]
pub fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    laine_karras(x.reverse_bits(), seed).reverse_bits()
}

/// Sobol の第 2 次元（生成行列の列が `v_k = v_{k-1} ^ (v_{k-1} >> 1)`、`v_0 = 2^31`）。
#[inline]
fn sobol_dim1(i: u32) -> u32 {
    SOBOL1[0][(i & 0xFF) as usize] ^ SOBOL1[1][((i >> 8) & 0xFF) as usize] ^ SOBOL1[2][((i >> 16) & 0xFF) as usize] ^ SOBOL1[3][(i >> 24) as usize]
}

/// 参照実装（ループ）。表の実装との一致をテストで確認する。
#[cfg(test)]
fn sobol_dim1_loop(mut i: u32) -> u32 {
    let mut v = 1u32 << 31;
    let mut r = 0u32;
    while i != 0 {
        if i & 1 != 0 {
            r ^= v;
        }
        i >>= 1;
        v ^= v >> 1;
    }
    r
}

/// 1 サンプルぶんの Sobol ストリーム（画素の種 + サンプル番号 + 次に引く次元）。
#[derive(Clone, Copy, Debug)]
pub struct SobolStream {
    seed: u32,
    index: u32,
    dim: u32,
    /// 直前に使ったペアと、そのシャッフル済みサンプル番号（同じペアの 2 成分でシャッフルを 1 回で済ませる）
    cached_pair: u32,
    cached_idx: u32,
    /// フォールバック用の PCG のシード。最初に必要になるまで初期化しない（`Rng` が `take` する。
    /// 深いバウンスに届かないサンプルが大半なので、サンプルごとの初期化を省く）
    pub(crate) pcg_seed: Option<u64>,
}

impl SobolStream {
    /// `pixel_seed`: 画素座標とユーザーシードから決めたスクランブルの種。`index`: 画素内のサンプル番号。
    pub fn new(pixel_seed: u32, index: u32) -> Self {
        Self { seed: pixel_seed, index, dim: 0, cached_pair: u32::MAX, cached_idx: 0, pcg_seed: None }
    }

    /// 次元 `dim` の値を、`idx`（そのペアのシャッフル済みサンプル番号）から作る。
    #[inline]
    fn from_shuffled(&self, dim: u32, idx: u32) -> f64 {
        let pair = dim >> 1;
        let raw = if dim & 1 == 0 { idx.reverse_bits() } else { sobol_dim1(idx) };
        let x = nested_uniform_scramble(raw, hash(self.seed, pair.wrapping_mul(3).wrapping_add(1 + (dim & 1))));
        x as f64 * (1.0 / 4_294_967_296.0)
    }

    /// 次元 `dim` の値（[0, 1)）。
    #[inline]
    pub fn value(&self, dim: u32) -> f64 {
        let idx = nested_uniform_scramble(self.index, hash(self.seed, (dim >> 1).wrapping_mul(3)));
        self.from_shuffled(dim, idx)
    }

    /// 次の次元を引く。上限を超えたら `None`（呼び出し側が PCG にフォールバックする）。
    #[inline]
    pub fn next(&mut self) -> Option<f64> {
        if self.dim >= MAX_DIMS {
            return None;
        }
        let pair = self.dim >> 1;
        if pair != self.cached_pair {
            self.cached_pair = pair;
            self.cached_idx = nested_uniform_scramble(self.index, hash(self.seed, pair.wrapping_mul(3)));
        }
        let v = self.from_shuffled(self.dim, self.cached_idx);
        self.dim += 1;
        Some(v)
    }

    /// 次に引く次元を `dim` にする。
    #[inline]
    pub fn set_dim(&mut self, dim: u32) {
        self.dim = dim;
    }

    /// 次に引く次元を偶数（ペアの先頭）に切り上げる。
    #[inline]
    pub fn align_pair(&mut self) {
        self.dim = (self.dim.saturating_add(1)) & !1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_values(seed: u32, index: u32, dims: u32) -> Vec<f64> {
        let s = SobolStream::new(seed, index);
        (0..dims).map(|d| s.value(d)).collect()
    }

    /// (0,2) の性質: 2^k 個の点を、面積 2^-k の任意の長方形格子 2^a × 2^b（a + b = k）に分けると、
    /// どのマスにもちょうど 1 点。すべてのペア（シャッフル込み）と、複数の種で成り立つ。
    #[test]
    fn every_pair_is_a_02_sequence() {
        for seed in [0u32, 1, 12345, 0xDEAD_BEEF] {
            for pair in [0u32, 1, 2, 7, 30, 66] {
                for k in [1u32, 3, 6] {
                    let n = 1u32 << k;
                    let pts: Vec<(f64, f64)> = (0..n)
                        .map(|i| {
                            let s = SobolStream::new(seed, i);
                            (s.value(2 * pair), s.value(2 * pair + 1))
                        })
                        .collect();
                    for a in 0..=k {
                        let b = k - a;
                        let mut cells = vec![0u32; 1 << k];
                        for &(x, y) in &pts {
                            let cx = (x * (1u64 << a) as f64) as usize;
                            let cy = (y * (1u64 << b) as f64) as usize;
                            cells[cy * (1usize << a) + cx] += 1;
                        }
                        assert!(cells.iter().all(|&c| c == 1), "seed {seed} pair {pair} k {k} a {a}: {cells:?}");
                    }
                }
            }
        }
    }

    /// 1 次元ごとにも各 1/2^k 区間に 1 点（1D の層化）。
    #[test]
    fn every_dimension_is_stratified_in_1d() {
        for dim in [0u32, 1, 4, 5, 9, 40, 133] {
            let n = 64u32;
            let mut cells = [0u32; 64];
            for i in 0..n {
                cells[(SobolStream::new(7, i).value(dim) * 64.0) as usize] += 1;
            }
            assert!(cells.iter().all(|&c| c == 1), "dim {dim}: {cells:?}");
        }
    }

    /// 決定論的（固定のアルゴリズム）で、種・サンプル番号・次元に敏感。値は [0, 1)。
    #[test]
    fn deterministic_and_sensitive_to_seed_index_dim() {
        assert_eq!(stream_values(3, 5, 20), stream_values(3, 5, 20));
        assert_ne!(stream_values(3, 5, 20), stream_values(4, 5, 20));
        assert_ne!(stream_values(3, 5, 20), stream_values(3, 6, 20));
        // 値を固定（実装が変わったら気付く。実行ごとに変わらないことの記録）
        assert_eq!(SobolStream::new(3, 5).value(0).to_bits(), EXPECT_V0);
        for i in 0..1000 {
            for d in [0u32, 1, 2, 3, 100, 133] {
                let v = SobolStream::new(99, i).value(d);
                assert!((0.0..1.0).contains(&v));
            }
        }
    }
    const EXPECT_V0: u64 = 4605271294179540992;

    /// 次元間の相関が小さい（異なるペア・同じペアの 2 成分）。平均 0.5、相関の二乗平均が独立な一様乱数
    /// （RMS = 1/√n）と同程度以下（32 個の種の平均。1 つの種だけを見ると偶然の揺らぎで外れうる）。
    #[test]
    fn dimensions_are_uncorrelated_and_uniform() {
        let n = 1024u32;
        for (d0, d1) in [(0u32, 1u32), (0, 2), (1, 3), (4, 5), (6, 18), (12, 13), (12, 28)] {
            let (mut sum_c2, mut sum_m) = (0.0, 0.0);
            for seed in 0..32u32 {
                let (mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for i in 0..n {
                    let s = SobolStream::new(seed * 7919 + 1, i);
                    let (x, y) = (s.value(d0), s.value(d1));
                    sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y;
                }
                let nf = n as f64;
                let (mx, my) = (sx / nf, sy / nf);
                let corr = (sxy / nf - mx * my) / ((sxx / nf - mx * mx).sqrt() * (syy / nf - my * my).sqrt());
                sum_c2 += corr * corr;
                sum_m += (mx - 0.5).abs() + (my - 0.5).abs();
            }
            let rms = (sum_c2 / 32.0).sqrt();
            assert!(rms < 1.5 / (n as f64).sqrt(), "{d0},{d1}: rms corr {rms}");
            assert!(sum_m / 64.0 < 0.01, "{d0},{d1}: mean drift {}", sum_m / 64.0);
        }
    }

    /// 表引きの Sobol 第 2 次元は、ループの参照実装と全域で一致する。
    #[test]
    fn sobol_dim1_table_matches_loop() {
        let mut s = 1u32;
        for i in (0..70_000u32).chain([u32::MAX, 0x8000_0000, 0x0101_0101]) {
            assert_eq!(sobol_dim1(i), sobol_dim1_loop(i));
        }
        for _ in 0..100_000 {
            s = mix32(s.wrapping_add(0x9E37_79B9));
            assert_eq!(sobol_dim1(s), sobol_dim1_loop(s));
        }
    }

    /// `next` のキャッシュ経路が、純粋な `value` と全次元・全サンプルで同じ値を返す（`set_dim` の後も）。
    #[test]
    fn cached_next_matches_pure_value() {
        for index in [0u32, 1, 5, 1000] {
            let mut s = SobolStream::new(77, index);
            for d in 0..MAX_DIMS {
                assert_eq!(s.next().unwrap().to_bits(), SobolStream::new(77, index).value(d).to_bits());
            }
            s.set_dim(13);
            assert_eq!(s.next().unwrap().to_bits(), SobolStream::new(77, index).value(13).to_bits());
            s.set_dim(4);
            assert_eq!(s.next().unwrap().to_bits(), SobolStream::new(77, index).value(4).to_bits());
        }
    }

    /// 上限を超えたら `None`、`set_dim` / `align_pair` / `bounce_dim` の挙動。
    #[test]
    fn stream_limits_and_dimension_control() {
        let mut s = SobolStream::new(1, 0);
        s.set_dim(MAX_DIMS - 1);
        assert!(s.next().is_some());
        assert!(s.next().is_none());
        s.set_dim(5);
        s.align_pair();
        assert_eq!(s.dim, 6);
        s.align_pair();
        assert_eq!(s.dim, 6, "偶数ならそのまま");
        s.set_dim(u32::MAX);
        s.align_pair();
        assert!(s.next().is_none());
        assert_eq!(bounce_dim(0, role::BSDF), 18);
        assert_eq!(bounce_dim(7, 15), MAX_DIMS - 1);
        assert_eq!(bounce_dim(8, 0), u32::MAX);
        assert_eq!(bounce_dim(usize::MAX, 0), u32::MAX);
        // すべてのバウンスの基準次元は偶数（2 次元の役割がペアに揃う）
        assert!((0..8).all(|b| bounce_dim(b, 0) % 2 == 0));
    }
}
