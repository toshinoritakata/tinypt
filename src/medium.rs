//! 一様な参加媒質（フォグ・煙）の数学。
//!
//! - `Medium`: 消衰係数 `σt`・単一散乱アルベド・HG 異方性 `g`・存在範囲（AABB）を持つ一様媒質
//! - `Medium::transmittance`: Beer-Lambert の透過率 `exp(-σt·L)`
//! - `Medium::sample_distance`: チャンネル MIS 付きの自由行程サンプリング
//! - `hg_eval` / `hg_sample`: Henyey-Greenstein 位相関数の評価とサンプリング
//!
//! このモジュールは積分器には繋がっていない（M1 段階）。数学だけを単体でテストできる形にしてある。
//!
//! # 距離サンプリングの設計
//! 媒質区間 `[a, b]` で、3 チャンネルのうち 1 つ `c` を一様に選び、`σt[c]` の指数分布で距離 `t` を引く。
//! 全チャンネルの pdf の平均（**balance heuristic**）で割るので、`σt` が色付きでも
//! weight が特定チャンネルで爆発しない。1 チャンネル固定でも期待値は合う（不偏）が、
//! `Tr` が桁違いに小さいチャンネルで分散が跳ね上がる。これは分散のための選択でバイアスのためではない。
//!
//! 期待値は成分ごとに `E[weight] = albedo·(1 - Tr(L)) + Tr(L)`（`albedo = 1` なら厳密に 1）。

use crate::geometry::Aabb;
use crate::material::tangent_frame;
use crate::math::{Color, Vec3};
use crate::ray::Ray;
use crate::rng::{uniform_sphere_dir, Rng};
use std::f64::consts::PI;

/// 一様な参加媒質。
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    /// 消衰係数（吸収 + 散乱）。成分ごと。単位は 1/長さ。
    pub sigma_t: Color,
    /// 単一散乱アルベド `σs/σt`（1 = 吸収なし、0 = 吸収のみ）。
    pub albedo: Color,
    /// Henyey-Greenstein の異方性 [-1, 1]。0 で等方、正で前方散乱。
    pub g: f64,
    /// 媒質の存在範囲。`None` で空間全体。
    pub bounds: Option<Aabb>,
}

/// 距離サンプリングの結果。
#[derive(Clone, Copy, Debug)]
pub enum MediumEvent {
    /// 媒質と相互作用せずに `t_max` まで到達した。`weight` は throughput に掛ける係数。
    Pass { weight: Color },
    /// レイ上の距離 `t` で散乱した。`weight` は throughput に掛ける係数（位相関数は含まない）。
    Scatter { t: f64, weight: Color },
}

const WHITE: Color = Color(Vec3 { x: 1.0, y: 1.0, z: 1.0 });

fn exp_neg(sigma: Color, len: f64) -> Color {
    Color::new((-sigma.r() * len).exp(), (-sigma.g() * len).exp(), (-sigma.b() * len).exp())
}

fn ch(c: Color, i: usize) -> f64 {
    match i { 0 => c.r(), 1 => c.g(), _ => c.b() }
}

impl Medium {
    /// 散乱係数 `σs = albedo ⊙ σt`。
    pub fn sigma_s(&self) -> Color { self.albedo.hadamard(self.sigma_t) }

    /// レイ `r` の `[t0, t1]` のうち、媒質が実際に存在する区間 `[a, b]` を返す。
    /// `bounds` が `None` なら `(t0, t1)`、あれば AABB との交差区間と `[t0, t1]` の共通部分。
    /// 重なりが無ければ `None`。
    pub fn interval(&self, r: Ray, t0: f64, t1: f64) -> Option<(f64, f64)> {
        if !(t1 > t0) { return None; }
        match self.bounds {
            None => Some((t0, t1)),
            Some(bb) => {
                let (a, b) = bb.hit_range(r, t0, t1)?;
                let (a, b) = (a.max(t0), b.min(t1));
                if b > a { Some((a, b)) } else { None }
            }
        }
    }

    /// 区間 `[t0, t1]` の透過率 `exp(-σt·L)`（`L` は媒質内の長さ）。区間が空なら白。乱数は引かない。
    pub fn transmittance(&self, r: Ray, t0: f64, t1: f64) -> Color {
        match self.interval(r, t0, t1) {
            None => WHITE,
            Some((a, b)) => exp_neg(self.sigma_t, b - a),
        }
    }

    /// `[0, t_max]` の自由行程をサンプリングする。
    ///
    /// 区間が空なら乱数を引かずに `Pass { weight: 白 }`。透過率が全チャンネル 0 に落ちるほど厚い場合は
    /// `Pass { weight: 黒 }`（経路を殺す）。そうでなければ乱数を 2 個引く
    /// （チャンネル選択 1 個 + 距離 1 個）。pdf は 3 チャンネルの balance heuristic
    /// （散乱: `Σ σt[i]·Tr[i] / 3`、通過: `Σ Tr[i] / 3`）。
    pub fn sample_distance(&self, r: Ray, t_max: f64, rng: &mut Rng) -> MediumEvent {
        // 区間が空 = 媒質に一切触れない。throughput は素通し（白）
        let untouched = MediumEvent::Pass { weight: WHITE };
        // 透過率が全チャンネル 0 に落ちた（光学的に極端に厚い）場合。経路はエネルギーを運ばないので
        // throughput を 0 にする。白を返すと背景がそのまま抜けてファイアフライになる
        let extinguished = MediumEvent::Pass { weight: Color::new(0.0, 0.0, 0.0) };
        let (a, b) = match self.interval(r, 0.0, t_max) {
            None => return untouched,
            Some(ab) => ab,
        };
        let c = ((rng.next_f64() * 3.0) as usize).min(2);
        let u = rng.next_f64();
        let sc = ch(self.sigma_t, c);
        let t = if sc <= 0.0 { f64::INFINITY } else { -(1.0 - u).ln() / sc };

        if a + t < b {
            let tr = exp_neg(self.sigma_t, t);
            let s = self.sigma_t;
            let pdf = (s.r() * tr.r() + s.g() * tr.g() + s.b() * tr.b()) / 3.0;
            if pdf <= 0.0 { return extinguished; }
            MediumEvent::Scatter { t: a + t, weight: self.sigma_s().hadamard(tr) / pdf }
        } else {
            let tr = exp_neg(self.sigma_t, b - a);
            let pdf = (tr.r() + tr.g() + tr.b()) / 3.0;
            if pdf <= 0.0 { return extinguished; }
            MediumEvent::Pass { weight: tr / pdf }
        }
    }
}

/// これ未満の `|g|` は等方として扱う。逆関数法が `1/(2g)` で桁落ちする領域を避けるためで、
/// `hg_eval` と `hg_sample` が**同じ**しきい値を使うことで pdf == eval の契約を保つ。
const ISOTROPIC_G: f64 = 1e-3;

/// Henyey-Greenstein 位相関数 `(1 - g²) / (4π (1 + g² + 2g·cosθ)^{3/2})`。
///
/// **符号規約**: `cos_theta = wo · wi`。`wo` は散乱点から見てレイが来た方向（`-ray.d`）、
/// `wi` は散乱後にレイが向かう方向として扱う。分母の `+2g·cosθ` により、`g > 0` では
/// `wi = -wo`（= `ray.d` 方向に進み続ける、つまり前方散乱）で最大になる。
/// 平均コサイン `E[ray.d · wi] = E[-cosθ] = +g` になる（テストで確認）。
pub fn hg_eval(cos_theta: f64, g: f64) -> f64 {
    // `hg_sample` と同じしきい値で等方に丸める。こうしないと |g| が微小なとき
    // 「sample が返す pdf」と「eval が返す pdf」がずれ、MIS が依存する契約
    // （pdf == eval の値）が壊れる
    if g.abs() < ISOTROPIC_G { return 1.0 / (4.0 * PI); }
    let denom = (1.0 + g * g + 2.0 * g * cos_theta).max(1e-300);
    (1.0 - g * g) / (4.0 * PI * denom * denom.sqrt())
}

/// HG 位相関数のサンプリング。`wo`（単位ベクトル、`-ray.d`）に対する散乱方向 `wi` と pdf を返す。
/// 乱数 2 個。`|g| < ISOTROPIC_G` は等方（pdf = 1/4π、一様球方向）。
/// 返す pdf は `hg_eval(wo·wi, g)` と厳密に同じ式・同じ値。
pub fn hg_sample(wo: Vec3, g: f64, rng: &mut Rng) -> (Vec3, f64) {
    if g.abs() < ISOTROPIC_G {
        let d = uniform_sphere_dir(rng);
        return (d, 1.0 / (4.0 * PI));
    }
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();
    // cosθ = wo·wi の逆関数法（PBRT v3 の規約）
    let sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
    let cos_theta = (-(1.0 / (2.0 * g)) * (1.0 + g * g - sq * sq)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = tangent_frame(wo);
    let wi = (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + wo * cos_theta).norm();
    (wi, hg_eval(wo.dot(wi), g))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray_x(ox: f64, oy: f64) -> Ray {
        Ray { o: Vec3::new(ox, oy, 0.0), d: Vec3::new(1.0, 0.0, 0.0), time: 0.0 }
    }
    fn unit_box() -> Aabb {
        Aabb { min: Vec3::new(-1.0, -1.0, -1.0), max: Vec3::new(1.0, 1.0, 1.0) }
    }
    fn med(sigma_t: Color, albedo: Color) -> Medium {
        Medium { sigma_t, albedo, g: 0.0, bounds: None }
    }
    fn close(a: Color, b: Color, tol: f64) -> bool {
        (a.r() - b.r()).abs() <= tol && (a.g() - b.g()).abs() <= tol && (a.b() - b.b()).abs() <= tol
    }
    fn w_of(e: MediumEvent) -> Color {
        match e { MediumEvent::Pass { weight } | MediumEvent::Scatter { weight, .. } => weight }
    }

    // 1. 球面積分 = 1
    #[test]
    fn hg_integrates_to_one() {
        for &g in &[-0.8, 0.0, 0.8] {
            let mut rng = Rng::new(11);
            let n = 400_000;
            let mut sum = 0.0;
            for _ in 0..n {
                let d = uniform_sphere_dir(&mut rng);
                sum += hg_eval(d.dot(Vec3::new(0.0, 1.0, 0.0)), g) * 4.0 * PI;
            }
            let mean = sum / n as f64;
            assert!((mean - 1.0).abs() < 0.01, "g={g}: {mean}");
        }
    }

    // 2. pdf の厳密一致
    #[test]
    fn hg_sample_pdf_matches_eval() {
        let mut rng = Rng::new(3);
        for &g in &[-0.9, -0.3, 0.0, 0.0005, 0.4, 0.95] {
            for _ in 0..2000 {
                let wo = uniform_sphere_dir(&mut rng);
                let (wi, pdf) = hg_sample(wo, g, &mut rng);
                assert!((wi.len() - 1.0).abs() < 1e-9);
                let e = if g.abs() < 1e-3 { 1.0 / (4.0 * PI) } else { hg_eval(wo.dot(wi), g) };
                assert!((pdf - e).abs() <= 1e-12 * e.max(1.0), "g={g}: {pdf} vs {e}");
            }
        }
    }

    // 3. 平均コサイン = +g（前方 = ray.d 方向）、標準誤差の 4 倍以内
    #[test]
    fn hg_sample_mean_cosine_is_g() {
        for &g in &[-0.8, 0.0, 0.5, 0.8] {
            let mut rng = Rng::new(21);
            let ray_d = Vec3::new(0.3, -0.5, 0.8).norm();
            let wo = -ray_d;
            let n = 400_000;
            let (mut s, mut s2) = (0.0, 0.0);
            for _ in 0..n {
                let (wi, _) = hg_sample(wo, g, &mut rng);
                let c = ray_d.dot(wi);
                s += c; s2 += c * c;
            }
            let mean = s / n as f64;
            let se = ((s2 / n as f64 - mean * mean) / n as f64).sqrt();
            assert!((mean - g).abs() < 4.0 * se, "g={g}: mean {mean}, se {se}");
        }
    }

    // 3'. eval 側も同じ規約（重み付き平均コサインが +g）
    #[test]
    fn hg_eval_convention_forward() {
        let g = 0.7;
        assert!(hg_eval(-1.0, g) > hg_eval(1.0, g)); // wi = ray.d 方向が最大
        let mut rng = Rng::new(5);
        let ray_d = Vec3::new(0.0, 0.0, 1.0);
        let (mut s, n) = (0.0, 400_000);
        for _ in 0..n {
            let d = uniform_sphere_dir(&mut rng);
            s += hg_eval((-ray_d).dot(d), g) * 4.0 * PI * ray_d.dot(d);
        }
        assert!((s / n as f64 - g).abs() < 0.02);
    }

    // 4. g = 0 は厳密に 1/4π
    #[test]
    fn hg_isotropic_exact() {
        for &c in &[-1.0, -0.3, 0.0, 0.7, 1.0] {
            assert_eq!(hg_eval(c, 0.0), 1.0 / (4.0 * PI));
        }
        let mut rng = Rng::new(1);
        let (_, pdf) = hg_sample(Vec3::new(0.0, 0.0, 1.0), 0.0, &mut rng);
        assert_eq!(pdf, 1.0 / (4.0 * PI));
    }

    // 5. Beer-Lambert
    #[test]
    fn beer_lambert_per_channel() {
        let m = med(Color::new(0.1, 0.5, 2.0), Color::new(1.0, 1.0, 1.0));
        let tr = m.transmittance(ray_x(0.0, 0.0), 0.0, 3.0);
        assert!(close(tr, Color::new((-0.3f64).exp(), (-1.5f64).exp(), (-6.0f64).exp()), 1e-14));
        // 区間 [t0, t1] の長さだけが効く
        let tr = m.transmittance(ray_x(0.0, 0.0), 1.0, 2.0);
        assert!(close(tr, Color::new((-0.1f64).exp(), (-0.5f64).exp(), (-2.0f64).exp()), 1e-14));
    }

    // 6. bounds の場合分け（ボックス [-1,1]^3、レイは +x 向き）
    #[test]
    fn bounds_cases() {
        let m = Medium { bounds: Some(unit_box()), ..med(Color::new(1.0, 1.0, 1.0), WHITE) };
        let near = |x: Option<(f64, f64)>, a: f64, b: f64| {
            let (p, q) = x.expect("interval");
            assert!((p - a).abs() < 1e-12 && (q - b).abs() < 1e-12, "{p} {q} vs {a} {b}");
        };
        // 外から入って内側で終わる: o=-5, t1=5 → box 内 t∈[4,6] ∩ [0,5]
        near(m.interval(ray_x(-5.0, 0.0), 0.0, 5.0), 4.0, 5.0);
        // 貫通
        near(m.interval(ray_x(-5.0, 0.0), 0.0, 100.0), 4.0, 6.0);
        // 内側で始まり内側で終わる
        near(m.interval(ray_x(-0.5, 0.0), 0.0, 1.0), 0.0, 1.0);
        // 内側から出る
        near(m.interval(ray_x(0.0, 0.0), 0.0, 100.0), 0.0, 1.0);
        // 手前で終わる／通り過ぎた後
        assert!(m.interval(ray_x(-5.0, 0.0), 0.0, 3.0).is_none());
        assert!(m.interval(ray_x(-5.0, 0.0), 7.0, 9.0).is_none());
        // 外れる
        assert!(m.interval(ray_x(-5.0, 2.0), 0.0, 100.0).is_none());
        // 後ろ向き
        let back = Ray { o: Vec3::new(-5.0, 0.0, 0.0), d: Vec3::new(-1.0, 0.0, 0.0), time: 0.0 };
        assert!(m.interval(back, 0.0, 100.0).is_none());
        // かすめる: 面のわずかに内側／外側
        near(m.interval(ray_x(-5.0, 1.0 - 1e-9), 0.0, 100.0), 4.0, 6.0);
        assert!(m.interval(ray_x(-5.0, 1.0 + 1e-9), 0.0, 100.0).is_none());
        // 面上ちょうど: 落ちずに有限の透過率を返す
        let tr = m.transmittance(ray_x(-5.0, 1.0), 0.0, 100.0).r();
        assert!(tr.is_finite() && (0.0..=1.0).contains(&tr));
        // 透過率も区間に従う
        assert!((m.transmittance(ray_x(-5.0, 0.0), 0.0, 100.0).g() - (-2.0f64).exp()).abs() < 1e-12);
        assert!(close(m.transmittance(ray_x(-5.0, 2.0), 0.0, 100.0), WHITE, 0.0));
    }

    // 7. エネルギー保存（解析値と一致）
    fn mean_weight(m: &Medium, l: f64, n: usize, seed: u64) -> ([f64; 3], [f64; 3]) {
        let mut rng = Rng::new(seed);
        let r = ray_x(0.0, 0.0);
        let (mut s, mut s2) = ([0.0; 3], [0.0; 3]);
        for _ in 0..n {
            let w = w_of(m.sample_distance(r, l, &mut rng));
            for i in 0..3 { let x = ch(w, i); s[i] += x; s2[i] += x * x; }
        }
        let nf = n as f64;
        let mean = [s[0] / nf, s[1] / nf, s[2] / nf];
        let se = [0, 1, 2].map(|i| ((s2[i] / nf - mean[i] * mean[i]).max(0.0) / nf).sqrt());
        (mean, se)
    }
    fn analytic(m: &Medium, l: f64) -> [f64; 3] {
        [0, 1, 2].map(|i| {
            let tr = (-ch(m.sigma_t, i) * l).exp();
            ch(m.albedo, i) * (1.0 - tr) + tr
        })
    }

    #[test]
    fn distance_sampling_energy() {
        let st = Color::new(0.1, 0.5, 2.0);
        for &alb in &[Color::new(1.0, 1.0, 1.0), Color::new(0.9, 0.5, 0.2)] {
            let m = med(st, alb);
            for &l in &[0.5, 2.0, 5.0] {
                let (mean, se) = mean_weight(&m, l, 1_000_000, 42);
                let want = analytic(&m, l);
                for i in 0..3 {
                    assert!((mean[i] - want[i]).abs() < 4.0 * se[i] + 1e-12,
                        "alb={alb:?} L={l} ch{i}: {} vs {} (se {})", mean[i], want[i], se[i]);
                }
            }
        }
    }

    // 8. チャンネル MIS の分散は、単一チャンネル固定より小さい。
    //    （σt=(0.1,0.5,2.0) では、最も薄いチャンネル 0 固定は albedo が色付きのとき MIS より
    //     良いことがある。MIS の利点は「どのチャンネルを固定しても大外れになりうる」ことを避ける点。
    //     そこで albedo=1 では 3 固定すべてに、色付き albedo では平均・最悪の固定に勝つことを見る。）
    fn fixed_channel_var(m: &Medium, c: usize, l: f64, n: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        let (sc, ss) = (ch(m.sigma_t, c), m.sigma_s());
        let (mut s, mut s2) = ([0.0; 3], [0.0; 3]);
        for _ in 0..n {
            let t = -(1.0 - rng.next_f64()).ln() / sc;
            let w = if t < l {
                ss.hadamard(exp_neg(m.sigma_t, t)) / (sc * (-sc * t).exp())
            } else {
                exp_neg(m.sigma_t, l) / (-sc * l).exp()
            };
            for i in 0..3 { let x = ch(w, i); s[i] += x; s2[i] += x * x; }
        }
        let nf = n as f64;
        let want = analytic(m, l);
        (0..3).map(|i| {
            let mean = s[i] / nf;
            let v = s2[i] / nf - mean * mean;
            assert!((mean - want[i]).abs() < 4.0 * (v / nf).sqrt() + 1e-12, "固定チャンネルも不偏のはず");
            v
        }).sum()
    }

    #[test]
    fn channel_mis_reduces_variance() {
        let (l, n) = (5.0, 300_000);
        for &alb in &[Color::new(1.0, 1.0, 1.0), Color::new(0.9, 0.5, 0.2)] {
            let m = med(Color::new(0.1, 0.5, 2.0), alb);
            let (_, se) = mean_weight(&m, l, n, 7);
            let var_mis: f64 = se.iter().map(|x| x * x * n as f64).sum();
            let one: Vec<f64> = (0..3).map(|c| fixed_channel_var(&m, c, l, n, 7)).collect();
            assert!(var_mis < one[1] && var_mis < one[2], "alb={alb:?}: MIS {var_mis} vs {one:?}");
            assert!(var_mis < one.iter().sum::<f64>() / 3.0);
            if alb.r() == 1.0 {
                assert!(var_mis < one[0], "MIS {var_mis} vs ch0 {}", one[0]);
            }
        }
    }

    // 9. スケール不変
    #[test]
    fn scale_invariance() {
        let st = Color::new(0.1, 0.5, 2.0);
        let base = med(st, Color::new(0.9, 0.5, 0.2));
        let (bm, _) = mean_weight(&base, 2.0, 200_000, 9);
        let tr0 = base.transmittance(ray_x(0.0, 0.0), 0.0, 2.0);
        for &k in &[1e-3, 1e3] {
            let m = med(st / k, base.albedo);
            let tr = m.transmittance(ray_x(0.0, 0.0), 0.0, 2.0 * k);
            assert!(close(tr, tr0, 1e-12), "k={k}");
            // 同じシード → 同じ乱数列 → 実質同じ weight 列
            let (mean, _) = mean_weight(&m, 2.0 * k, 200_000, 9);
            for i in 0..3 { assert!((mean[i] - bm[i]).abs() < 1e-8, "k={k} ch{i}"); }
            // 解析値とも一致
            let (mean, se) = mean_weight(&m, 2.0 * k, 500_000, 10);
            let want = analytic(&m, 2.0 * k);
            for i in 0..3 { assert!((mean[i] - want[i]).abs() < 4.0 * se[i]); }
        }
    }

    // 10. 退化ケース
    #[test]
    fn degenerate_cases() {
        let r = ray_x(0.0, 0.0);
        // σt = 0: 必ず Pass、weight 白（albedo に依らない）
        let m = med(Color::new(0.0, 0.0, 0.0), Color::new(0.5, 0.5, 0.5));
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            match m.sample_distance(r, 10.0, &mut rng) {
                MediumEvent::Pass { weight } => assert!(close(weight, WHITE, 1e-15)),
                e => panic!("{e:?}"),
            }
        }
        // 一部チャンネルだけ σt = 0 でも有限
        let m = med(Color::new(0.0, 1.0, 1.0), WHITE);
        for _ in 0..10_000 {
            let w = w_of(m.sample_distance(r, 3.0, &mut rng));
            assert!(w.r().is_finite() && w.g().is_finite() && w.b().is_finite());
        }
        // t_max = 0: 乱数を引かず Pass 白
        let m = med(Color::new(1.0, 1.0, 1.0), WHITE);
        let mut a = Rng::new(5);
        let mut b = Rng::new(5);
        match m.sample_distance(r, 0.0, &mut a) {
            MediumEvent::Pass { weight } => assert!(close(weight, WHITE, 0.0)),
            e => panic!("{e:?}"),
        }
        assert_eq!(a.next_f64(), b.next_f64());
        // 潰れた AABB（体積 0）と点 AABB は乱数を引かず、透過率は白
        let flat = Aabb { min: Vec3::new(0.0, -1.0, -1.0), max: Vec3::new(0.0, 1.0, 1.0) };
        let mf = Medium { bounds: Some(flat), ..m };
        let mut a = Rng::new(5);
        // 厚み 0 の板は区間長 ~1e-15 にしかならない（乱数は引くが weight は白）
        match mf.sample_distance(ray_x(-5.0, 0.0), 100.0, &mut a) {
            MediumEvent::Pass { weight } => assert!(close(weight, WHITE, 1e-12)),
            e => panic!("{e:?}"),
        }
        assert!(close(mf.transmittance(ray_x(-5.0, 0.0), 0.0, 100.0), WHITE, 1e-12));
        let point = Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(0.0, 0.0, 0.0) };
        let mp = Medium { bounds: Some(point), ..m };
        assert!(close(mp.transmittance(ray_x(-5.0, 0.0), 0.0, 100.0), WHITE, 1e-12));
        // 散乱位置は区間内
        let mb = Medium { bounds: Some(unit_box()), ..med(Color::new(5.0, 5.0, 5.0), WHITE) };
        let mut rng = Rng::new(2);
        for _ in 0..1000 {
            if let MediumEvent::Scatter { t, .. } = mb.sample_distance(ray_x(-5.0, 0.0), 100.0, &mut rng) {
                assert!((4.0..6.0).contains(&t), "{t}");
            }
        }
    }
}
