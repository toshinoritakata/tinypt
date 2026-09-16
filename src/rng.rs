//! 擬似乱数生成器（PCG）とシード生成ヘルパー。
//!
//! PCG (Permuted Congruential Generator) は統計的品質が高く高速な乱数生成器。
//! 各ピクセル・サンプルに対して決定論的なシードを生成し、
//! 再現可能なレンダリング結果を保証する。

use crate::math::Vec3;

#[derive(Clone, Copy)]
/// PCG32 ベースの乱数生成器。
pub struct Rng { state: u64 }
impl Rng {
    /// シードから RNG を初期化する（任意の `u64`、0 も可）。
    ///
    /// シードを SplitMix64 で攪拌してから PCG の推奨初期化（state=0 → 1 ステップ →
    /// シード加算 → 1 ステップ）を行う。近いシード（隣接ピクセルなど）同士でも
    /// 初期状態が十分に離れ、出力の先頭が相関しない。
    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: 0 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(splitmix64(seed));
        rng.next_u32();
        rng
    }

    /// 次の 32 ビット値を生成する（PCG32 出力関数）。
    pub fn next_u32(&mut self) -> u32 {
        // PCG32: LCG + XSH-RR（XorShift + Random Rotation）出力関数
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// [0, 1) の一様分布 `f64` を生成する。
    pub fn next_f64(&mut self) -> f64 {
        // 2 回の 32 ビット出力から 53 ビットの精度を得る（IEEE 754 倍精度の仮数部）
        let hi = (self.next_u32() as u64) << 21;
        let lo = (self.next_u32() as u64) & ((1u64 << 21) - 1);
        let u = hi | lo;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// 単位球面上の一様サンプリング（`env::EnvMap` の一様フォールバックなどで使う）。
pub fn uniform_sphere_dir(rng: &mut Rng) -> Vec3 {
    let u = rng.next_f64();
    let v = rng.next_f64();
    let z = 1.0 - 2.0 * u;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * v;
    Vec3::new(r * phi.cos(), z, r * phi.sin())
}

/// SplitMix64 のミキサ（Steele et al., "Fast Splittable Pseudorandom Number Generators"）。
/// 入力の 1 ビットの違いが出力全体に拡散する全単射。
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// ピクセル座標 (px, py)・サンプル開始インデックス s・ユーザーシード `seed` から
/// 決定論的な RNG シードを導出する。
///
/// 各要素を SplitMix64 で順に畳み込むので、XOR 合成と違い要素同士が打ち消し合わない
/// （例: 以前は `(px, py)` の組み替えや `seed` の XOR で別ピクセルと同じシードになりえた）。
pub fn seed_for(px: u32, py: u32, s: u32, seed: u64) -> u64 {
    let mut h = splitmix64(seed);
    h = splitmix64(h ^ px as u64);
    h = splitmix64(h ^ py as u64);
    splitmix64(h ^ s as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 同じシードからは同じ列、異なるシードからは異なる列（0 も有効なシード）。
    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let seq = |seed: u64| {
            let mut r = Rng::new(seed);
            (0..8).map(|_| r.next_u32()).collect::<Vec<_>>()
        };
        assert_eq!(seq(0), seq(0));
        assert_ne!(seq(0), seq(1));
        assert_ne!(seq(1), seq(2));
    }

    /// seed_for は全要素に依存し、隣接ピクセル・座標の入れ替え・ユーザーシードで衝突しない。
    #[test]
    fn seed_for_has_no_collisions_on_a_grid() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..3u64 {
            for s in [0u32, 1, 64] {
                for y in 0..64u32 {
                    for x in 0..64u32 {
                        assert!(seen.insert(seed_for(x, y, s, seed)), "collision at {:?}", (x, y, s, seed));
                    }
                }
            }
        }
        assert_ne!(seed_for(1, 2, 0, 0), seed_for(2, 1, 0, 0));
    }

    /// 隣接シードの RNG の先頭出力は相関しない（ビット単位の一致率がほぼ 1/2）。
    /// 旧実装はシードを state に直接入れていたため、隣接シードの初期出力が強く相関した。
    #[test]
    fn adjacent_seeds_give_uncorrelated_first_outputs() {
        let n = 4096u64;
        let mut same_bits = 0u64;
        for i in 0..n {
            let a = Rng::new(i).next_u32();
            let b = Rng::new(i + 1).next_u32();
            same_bits += (!(a ^ b)).count_ones() as u64;
        }
        let frac = same_bits as f64 / (n * 32) as f64;
        assert!((frac - 0.5).abs() < 0.01, "bit agreement {}", frac);
    }

    /// next_f64 の平均・分散が一様分布 [0,1) と整合する（1/2, 1/12）。
    #[test]
    fn next_f64_moments_match_uniform() {
        let mut r = Rng::new(12345);
        let n = 200_000;
        let (mut m, mut m2) = (0.0, 0.0);
        for _ in 0..n {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            m += x;
            m2 += x * x;
        }
        let mean = m / n as f64;
        let var = m2 / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 0.005, "mean {}", mean);
        assert!((var - 1.0 / 12.0).abs() < 0.002, "var {}", var);
    }
}
