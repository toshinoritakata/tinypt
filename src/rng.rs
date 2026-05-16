//! 擬似乱数生成器（PCG）とシード生成ヘルパー。
//!
//! PCG (Permuted Congruential Generator) は統計的品質が高く高速な乱数生成器。
//! 各ピクセル・サンプルに対して決定論的なシードを生成し、
//! 再現可能なレンダリング結果を保証する。

#[derive(Clone, Copy)]
/// PCG32 ベースの乱数生成器。
pub struct Rng { state: u64 }
impl Rng {
    /// ゼロでないシードで RNG を初期化する。
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
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

/// ピクセル座標 (px, py) とサンプルインデックス s から決定論的シードを導出する。
/// 各ピクセル・サンプルに一意のシードを割り当て、再現可能な結果を保証する。
pub fn seed_for(px: u32, py: u32, s: u32) -> u64 {
    let mut h = 0xD1B54A32D192ED03u64;
    h ^= (px as u64).wrapping_mul(0x9E3779B185EBCA87);
    h ^= (py as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    h ^= (s  as u64).wrapping_mul(0x165667B19E3779F9);
    h ^ (h >> 32)
}
