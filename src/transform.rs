//! アフィン変換（任意の線形部 + 平行移動）。
//!
//! 線形部を `Mat3`、平行移動を `Vec3` で保持し、逆変換用に線形部の逆行列を、
//! 法線変換用に逆転置行列を事前計算する。任意軸回転・非均一スケール・任意の
//! 4×4 行列を表現できる。

use crate::geometry::Aabb;
use crate::math::{gamma, Mat3, Vec3};

#[derive(Clone, Copy, Debug)]
/// オブジェクト → ワールドのアフィン変換 `p' = A·p + t`。
pub struct Transform {
    /// 線形部（回転・スケール・せん断）
    a: Mat3,
    /// 平行移動
    t: Vec3,
    /// 線形部の逆行列（ワールド → オブジェクト）
    a_inv: Mat3,
    /// 法線変換行列（線形部の逆転置）
    normal_mat: Mat3,
    /// 数値的な逆行列の残差 `max |(A·A⁻¹ − I)_ij|`（ワールド → オブジェクト変換の誤差上界に使う。
    /// 条件数の大きい変換ほど大きい）
    inv_residual: f64,
    /// `|A|`（成分の絶対値、誤差上界の伝播用に事前計算）
    abs_a: Mat3,
    /// `|A⁻¹|`
    abs_a_inv: Mat3,
    /// `‖A⁻¹‖∞`（行の絶対値和の最大）
    inv_linf: f64,
}

impl Transform {
    /// 線形部 `a` と平行移動 `t` からアフィン変換を構築する。
    pub fn from_affine(a: Mat3, t: Vec3) -> Self {
        let a_inv = a.invert();
        let prod = a.mul(a_inv);
        let mut inv_residual: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let ideal = if i == j { 1.0 } else { 0.0 };
                inv_residual = inv_residual.max((prod.m[i][j] - ideal).abs());
            }
        }
        let abs = |m: Mat3| Mat3 { m: m.m.map(|row| row.map(f64::abs)) };
        let inv_linf = a_inv.m.iter().map(|row| row.iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max);
        Self {
            a, t, a_inv, normal_mat: a_inv.transpose(), inv_residual,
            abs_a: abs(a), abs_a_inv: abs(a_inv), inv_linf,
        }
    }

    /// 恒等変換。
    pub fn identity() -> Self {
        Self::from_affine(Mat3::identity(), Vec3::new(0.0, 0.0, 0.0))
    }

    /// 平行移動 + Y 軸回転（度）+ 均一スケールから構築する（後方互換）。
    /// `p' = R_y(scale·p) + t`。
    pub fn new(t: Vec3, rot_y_deg: f64, s: f64) -> Self {
        Self::from_affine(rotation(Vec3::new(0.0, 1.0, 0.0), rot_y_deg).mul(scale_uniform(s)), t)
    }

    /// 平行移動のみの変換。
    pub fn translate(t: Vec3) -> Self {
        Self::from_affine(Mat3::identity(), t)
    }

    /// 任意軸 `axis` 周りの回転（角度は度）。
    pub fn rotate(axis: Vec3, deg: f64) -> Self {
        Self::from_affine(rotation(axis, deg), Vec3::new(0.0, 0.0, 0.0))
    }

    /// 成分ごとのスケール。
    pub fn scale(s: Vec3) -> Self {
        Self::from_affine(
            Mat3::from_rows([[s.x, 0.0, 0.0], [0.0, s.y, 0.0], [0.0, 0.0, s.z]]),
            Vec3::new(0.0, 0.0, 0.0),
        )
    }

    /// 行優先の 4×4 行列（最終行は `0 0 0 1` を仮定）から構築する。
    pub fn from_matrix4(m: [[f64; 4]; 4]) -> Self {
        let a = Mat3::from_rows([
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ]);
        Self::from_affine(a, Vec3::new(m[0][3], m[1][3], m[2][3]))
    }

    /// `inner` を先に適用し、その後に `self` を適用する合成変換を返す。
    /// `self ∘ inner`（点には inner が内側）。
    pub fn compose(self, inner: Transform) -> Transform {
        Self::from_affine(self.a.mul(inner.a), self.a.mul_vec(inner.t) + self.t)
    }

    /// オブジェクト空間の点をワールド空間へ: `p' = A·p + t`。
    pub fn apply_point(self, p: Vec3) -> Vec3 {
        self.a.mul_vec(p) + self.t
    }

    /// オブジェクト空間の点 `p`（成分ごとの誤差上界 `p_err`）をワールド空間へ写し、
    /// ワールド空間の点と、その成分ごとの誤差上界を返す（PBRT の `Transform::operator()(Point3fi)`）。
    /// 変換自体の丸め `γ(3)·(|A|·|p| + |t|)` と、入力の誤差の伝播 `(1 + γ(3))·|A|·p_err` の和。
    pub fn apply_point_with_error(self, p: Vec3, p_err: Vec3) -> (Vec3, Vec3) {
        let pw = self.a.mul_vec(p) + self.t;
        let err = (self.abs_a.mul_vec(p.abs()) + self.t.abs()) * gamma(3) + self.abs_a.mul_vec(p_err) * (1.0 + gamma(3));
        (pw, err)
    }

    /// ワールド空間の点（誤差なし）をオブジェクト空間へ写し、オブジェクト空間の点と成分ごとの誤差上界を返す。
    /// 平行移動の引き算・行列積の丸めに加え、数値的な逆行列が厳密な逆でないこと（残差 × |A⁻¹|）も含める。
    pub fn apply_point_inv_with_error(self, p_world: Vec3) -> (Vec3, Vec3) {
        let diff = p_world - self.t;
        let diff_err = (p_world.abs() + self.t.abs()) * gamma(1);
        let p = self.a_inv.mul_vec(diff);
        let residual = self.inv_residual * diff.l1();
        // |A⁻¹|·(γ(3)·|diff| + (1 + γ(3))·(diff_err + residual)) を 1 回の行列積で
        let g = 1.0 + gamma(3);
        let err = self.abs_a_inv.mul_vec(diff.abs() * gamma(3) + (diff_err + Vec3::new(residual, residual, residual)) * g);
        (p, err)
    }

    /// ワールド空間の点をオブジェクト空間へ写し、全成分共通の誤差上界（L∞）と一緒に返す。
    /// [`apply_point_inv_with_error`](Self::apply_point_inv_with_error) の成分ごとの上界を安価に上から抑えたもの
    /// （インスタンスごと・レイごとに呼ぶため）:
    /// `‖A⁻¹‖∞ · (γ(3)·‖d‖∞ + (1 + γ(3))·(γ(1)·(‖p‖∞ + ‖t‖∞) + 3·残差·‖d‖∞))`、d = p − t。
    /// 引き算の丸めは座標の大きさに、行列積と逆行列の残差は平行移動からの距離 d に比例する。
    #[inline]
    pub fn apply_point_inv_with_error_linf(&self, p_world: Vec3) -> (Vec3, f64) {
        let diff = p_world - self.t;
        let p = self.a_inv.mul_vec(diff);
        let dl = diff.max_abs();
        let g3 = gamma(3);
        let err = self.inv_linf * (g3 * dl + (1.0 + g3) * (gamma(1) * (p_world.max_abs() + self.t.max_abs()) + 3.0 * self.inv_residual * dl));
        (p, err * (1.0 + 2.0 * g3))
    }

    /// ワールド空間の点をオブジェクト空間へ: `p = A⁻¹·(p' − t)`。
    pub fn apply_point_inv(self, p_world: Vec3) -> Vec3 {
        self.a_inv.mul_vec(p_world - self.t)
    }

    /// ワールド空間のベクトル（方向）をオブジェクト空間へ: `v = A⁻¹·v'`。
    pub fn apply_vec_inv(self, v_world: Vec3) -> Vec3 {
        self.a_inv.mul_vec(v_world)
    }

    /// オブジェクト空間の方向ベクトル（接ベクトルなど）をワールドへ: `v' = A·v`（正規化しない）。
    /// 法線ではないので逆転置は**使わない**（せん断・非一様スケールで向きがずれる）。
    pub fn apply_vec(self, v_obj: Vec3) -> Vec3 {
        self.a.mul_vec(v_obj)
    }

    /// オブジェクト空間の法線をワールド空間へ（逆転置行列で変換し正規化）。
    pub fn apply_normal(self, n_obj: Vec3) -> Vec3 {
        self.normal_mat.mul_vec(n_obj).norm()
    }
}

/// 均一スケール行列。
fn scale_uniform(s: f64) -> Mat3 {
    Mat3::from_rows([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]])
}

/// Rodrigues の公式による軸 `axis` 周り `deg` 度の回転行列。
fn rotation(axis: Vec3, deg: f64) -> Mat3 {
    let k = axis.norm();
    let (s, c) = deg.to_radians().sin_cos();
    let one_c = 1.0 - c;
    let (x, y, z) = (k.x, k.y, k.z);
    Mat3::from_rows([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])
}

// ---------------------------------------------------------------------------
// アニメーション変換（シャッター開 → 閉の 2 キーフレーム）
// ---------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

fn m3_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn m3_t(a: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[j][i];
        }
    }
    r
}

fn m3_det(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// 余因子から逆行列（`det` が 0 でないこと）。
fn m3_inv(a: &M3, det: f64) -> M3 {
    let c = |r0: usize, r1: usize, c0: usize, c1: usize| a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0];
    let inv = 1.0 / det;
    [
        [c(1, 2, 1, 2) * inv, -c(0, 2, 1, 2) * inv, c(0, 1, 1, 2) * inv],
        [-c(1, 2, 0, 2) * inv, c(0, 2, 0, 2) * inv, -c(0, 1, 0, 2) * inv],
        [c(1, 2, 0, 1) * inv, -c(0, 2, 0, 1) * inv, c(0, 1, 0, 1) * inv],
    ]
}

fn m3_frob(a: &M3) -> f64 {
    a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

/// 極分解の反復の上限と収束判定（`‖R_next − R‖_F`）。回転行列に近い入力なら数回で収束する。
const POLAR_MAX_ITERS: usize = 100;
const POLAR_TOL: f64 = 1e-14;

/// 線形部 `A = R·S` の極分解（`R` は回転行列、`S` は対称な伸縮）。`R ← (R + R⁻ᵀ)/2` を収束まで繰り返す。
/// `A` が特異・非有限・鏡像（`det ≤ 0`）、または収束しなければ `None`（呼び出し側は静止扱いに落とす）。
fn polar_decompose(a: &M3) -> Option<(M3, M3)> {
    if a.iter().flatten().any(|x| !x.is_finite()) {
        return None;
    }
    let det_a = m3_det(a);
    // 小さすぎる行列式（ほぼ特異）と負（鏡像）は補間できない
    if !(det_a > 1e-12 * m3_frob(a).powi(3)) {
        return None;
    }
    let mut r = *a;
    for _ in 0..POLAR_MAX_ITERS {
        let d = m3_det(&r);
        if !(d.abs() > 0.0) || !d.is_finite() {
            return None;
        }
        let it = m3_t(&m3_inv(&r, d));
        let mut next = [[0.0; 3]; 3];
        let mut diff = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + it[i][j]);
                diff += (next[i][j] - r[i][j]).powi(2);
            }
        }
        r = next;
        if diff.sqrt() < POLAR_TOL {
            let s = m3_mul(&m3_t(&r), a);
            return Some((r, s));
        }
    }
    None
}

/// 回転行列 → 単位四元数 (w, x, y, z)。
fn quat_from_rot(r: &M3) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [(r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [(r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [(r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s]
    };
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn rot_from_quat(q: [f64; 4]) -> M3 {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// 四元数の球面線形補間。`q1` を反対側に取り直して常に短い経路を通る（`dot < 0` なら `-q1`。
/// ちょうど 180° 離れた回転は `dot = 0` で経路が曖昧になりうるが、決定的に片方が選ばれ、両端は正確）。
/// ほぼ同じ向き（`dot > 0.9995`）は正規化した線形補間（0 除算を避ける）。
fn quat_slerp(t: f64, q0: [f64; 4], q1: [f64; 4]) -> [f64; 4] {
    let mut dot: f64 = q0.iter().zip(&q1).map(|(a, b)| a * b).sum();
    let mut q1 = q1;
    if dot < 0.0 {
        q1 = q1.map(|x| -x);
        dot = -dot;
    }
    let (a, b) = if dot > 0.9995 {
        (1.0 - t, t)
    } else {
        let theta = dot.min(1.0).acos();
        let sin = theta.sin();
        (((1.0 - t) * theta).sin() / sin, (t * theta).sin() / sin)
    };
    let q = [0, 1, 2, 3].map(|i| a * q0[i] + b * q1[i]);
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}

/// シャッター開（time = 0）と閉（time = 1）の 2 つの変換を、`time` で補間するアニメーション変換
/// （PBRT の `AnimatedTransform` と同じ方式）。
///
/// 行列を直接線形補間すると回転が縮む（中間で角度が中間にならず、大きな回転では潰れる）ので、
/// 線形部を `A = R·S`（極分解）に**読み込み時に 1 回だけ**分解しておき、`at(time)` では
/// 平行移動を線形補間・回転を四元数の slerp・伸縮 `S` を成分ごとに線形補間して再合成する。
/// 端点（`time ≤ 0` / `≥ 1`）は元の変換をそのまま返すので、両端は厳密に一致する。
#[derive(Clone, Copy, Debug)]
pub struct AnimatedTransform {
    start: Transform,
    end: Transform,
    t0: Vec3,
    t1: Vec3,
    q0: [f64; 4],
    q1: [f64; 4],
    s0: M3,
    s1: M3,
    /// 掃過ボリューム用の量（読み込み時に計算）: 補間した伸縮の作用素ノルムの上界、
    /// 回転の総角度、`‖S1 − S0‖_F`、`|T1 − T0|`
    stretch_bound: f64,
    rot_angle: f64,
    ds_frob: f64,
    dt_len: f64,
}

impl AnimatedTransform {
    /// 2 つの変換から作る。**どちらかの線形部が特異・鏡像（det ≤ 0）・非有限、または極分解が収束しない**
    /// ときは `None`（鏡像は slerp で扱えない。呼び出し側は警告して静止扱いに落とす）。
    pub fn new(start: Transform, end: Transform) -> Option<Self> {
        let (r0, s0) = polar_decompose(&start.a.m)?;
        let (r1, s1) = polar_decompose(&end.a.m)?;
        let q0 = quat_from_rot(&r0);
        let q1 = quat_from_rot(&r1);
        let dot: f64 = q0.iter().zip(&q1).map(|(a, b)| a * b).sum::<f64>().abs().min(1.0);
        let mut ds = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                ds[i][j] = s1[i][j] - s0[i][j];
            }
        }
        Some(Self {
            start,
            end,
            t0: start.t,
            t1: end.t,
            q0,
            q1,
            s0,
            s1,
            // ‖S_t‖₂ ≤ ‖S_t‖_F ≤ (1−t)‖S0‖_F + t‖S1‖_F ≤ max(‖S0‖_F, ‖S1‖_F)（ノルムの凸性）
            stretch_bound: m3_frob(&s0).max(m3_frob(&s1)),
            rot_angle: 2.0 * dot.acos(),
            ds_frob: m3_frob(&ds),
            dt_len: (end.t - start.t).len(),
        })
    }

    /// 時刻 `time` の変換。`time ≤ 0` は開、`time ≥ 1` は閉の変換そのもの。それ以外は補間して再合成する。
    #[inline]
    pub fn at(&self, time: f64) -> Transform {
        if time <= 0.0 {
            return self.start;
        }
        if time >= 1.0 {
            return self.end;
        }
        let t = self.t0 * (1.0 - time) + self.t1 * time;
        let r = rot_from_quat(quat_slerp(time, self.q0, self.q1));
        let mut s = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = (1.0 - time) * self.s0[i][j] + time * self.s1[i][j];
            }
        }
        Transform::from_affine(Mat3::from_rows(m3_mul(&r, &s)), t)
    }

    /// 物体空間の境界球（中心 `c`、半径 `r`）が、時刻区間 `[open, close]`（`0 ≤ open ≤ close ≤ 1`）の全時刻で
    /// ワールド空間のどこにあるかを覆う保守的な AABB。
    ///
    /// **なぜ保守的か**: (1) 球は回転で膨らまず、時刻 `t` のワールド空間の像は中心 `c_t = A_t·c + T_t`、
    /// 半径 `≤ ‖A_t‖₂·r = ‖S_t‖₂·r ≤ stretch_bound·r`（回転は作用素ノルムを変えない。`S_t` の上界は
    /// ノルムの凸性から両端の大きい方）。(2) 中心の軌跡は `N` 個の時刻でサンプルし、サンプル間は
    /// リプシッツ定数 `L = Θ·σ·|c| + ‖S1 − S0‖_F·|c| + |T1 − T0|`（`Θ` は回転の総角度、`σ = stretch_bound`）で
    /// 押さえる: `|d c_t/dt| ≤ |ω|·|S_t c| + |(S1−S0) c| + |T1−T0|`。サンプル間隔 `h` の中の任意の時刻は
    /// 最寄りのサンプルから `L·h/2` 以内。よって各サンプル中心を `stretch_bound·r + L·h/2` だけ広げた箱の和は
    /// 全時刻の球を含む。(3) 丸めぶんに座標に比例した余白を足す。
    pub fn swept_bounds(&self, center: Vec3, radius: f64, open: f64, close: f64) -> Aabb {
        const N: usize = 32;
        let span = (close - open).max(0.0);
        let c_len = center.len();
        let lipschitz = self.rot_angle * self.stretch_bound * c_len + self.ds_frob * c_len + self.dt_len;
        let h = span / N as f64;
        let rad = self.stretch_bound * radius + 0.5 * lipschitz * h;
        let mut out = Aabb::empty();
        for k in 0..=N {
            let time = open + span * k as f64 / N as f64;
            let c = self.at(time).apply_point(center);
            out = out.grow(c - Vec3::new(rad, rad, rad)).grow(c + Vec3::new(rad, rad, rad));
        }
        // 丸め（中心の計算と箱の端）と、大きさに比例した安全余白
        let pad_of = |lo: f64, hi: f64| gamma(8) * lo.abs().max(hi.abs()) + 1e-9 * rad;
        let (px, py, pz) = (pad_of(out.min.x, out.max.x), pad_of(out.min.y, out.max.y), pad_of(out.min.z, out.max.z));
        out.min = Vec3::new(out.min.x - px, out.min.y - py, out.min.z - pz);
        out.max = Vec3::new(out.max.x + px, out.max.y + py, out.max.z + pz);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).len() < 1e-9
    }

    /// 後方互換: new(t, rot_y, s) は p' = R_y(s·p) + t。
    #[test]
    fn legacy_new_matches_trs() {
        let xf = Transform::new(Vec3::new(1.0, 2.0, 3.0), 90.0, 2.0);
        // (1,0,0) を 2 倍 → (2,0,0)、Y 90° 回転 → (0,0,-2)、+t → (1,2,1)
        assert!(close(xf.apply_point(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(1.0, 2.0, 1.0)));
    }

    /// 任意軸（X 軸）回転が表現できる（Y 軸限定だった旧実装では不可）。
    #[test]
    fn rotate_about_x_axis() {
        let xf = Transform::rotate(Vec3::new(1.0, 0.0, 0.0), 90.0);
        // X 軸 90°: (0,1,0) → (0,0,1)
        assert!(close(xf.apply_point(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(0.0, 0.0, 1.0)));
    }

    /// 非均一スケール下で法線が逆転置で正しく変換される。
    #[test]
    fn nonuniform_scale_normal_uses_inverse_transpose() {
        // x を 2 倍。平面 x=const の法線 (1,0,0) はスケール後も (1,0,0) のまま（正規化後）。
        // 一方、点をそのまま掛けると (2,0,0)。逆転置なら法線は (0.5,0,0)→正規化(1,0,0)。
        let xf = Transform::scale(Vec3::new(2.0, 1.0, 1.0));
        let n = xf.apply_normal(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(n, Vec3::new(1.0, 0.0, 0.0)));
        // 45°方向の法線は非均一スケールで向きが変わる
        let n2 = xf.apply_normal(Vec3::new(1.0, 1.0, 0.0).norm());
        // 逆転置 diag(0.5,1,1) を (1,1,0)/√2 に適用 → (0.5,1,0) 正規化
        let expected = Vec3::new(0.5, 1.0, 0.0).norm();
        assert!(close(n2, expected));
    }

    /// 逆変換が順変換の逆になっている。
    #[test]
    fn inverse_roundtrips() {
        let xf = Transform::rotate(Vec3::new(0.3, 1.0, 0.5), 37.0)
            .compose(Transform::scale(Vec3::new(2.0, 0.5, 1.5)));
        let p = Vec3::new(1.0, -2.0, 3.0);
        assert!(close(xf.apply_point_inv(xf.apply_point(p)), p));
    }

    /// 合成順序: <translate y=1> の後に <scale 2>（最後の子が内側）。
    /// trafo = translate * scale なので、点はスケール → 平行移動の順。
    #[test]
    fn compose_applies_inner_first() {
        // acc = identity.compose(translate).compose(scale)
        let xf = Transform::identity()
            .compose(Transform::translate(Vec3::new(0.0, 1.0, 0.0)))
            .compose(Transform::scale(Vec3::new(2.0, 2.0, 2.0)));
        // (1,0,0): スケール → (2,0,0)、平行移動 → (2,1,0)
        assert!(close(xf.apply_point(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(2.0, 1.0, 0.0)));
    }

    // ---- アニメーション変換 ----

    fn mat_close(a: &Mat3, b: &Mat3, tol: f64) -> bool {
        (0..3).all(|i| (0..3).all(|j| (a.m[i][j] - b.m[i][j]).abs() <= tol))
    }

    /// 端点は元の変換とビット一致、`time = 0.5` の回転は角度の中間（行列の線形補間なら中間にならない）。
    #[test]
    fn animated_endpoints_are_exact_and_midpoint_is_the_middle_angle() {
        let start = Transform::translate(Vec3::new(-1.0, 0.5, 0.0));
        let end = Transform::translate(Vec3::new(1.0, 0.5, 0.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 90.0));
        let anim = AnimatedTransform::new(start, end).expect("decomposable");
        for (t, want) in [(0.0, start), (-0.5, start), (1.0, end), (2.0, end)] {
            let got = anim.at(t);
            assert!(mat_close(&got.a, &want.a, 0.0) && got.t.x == want.t.x && got.t.y == want.t.y, "t={t}");
        }
        // 中間: 45° の回転、平行移動は中点
        let mid = anim.at(0.5);
        let want = Transform::translate(Vec3::new(0.0, 0.5, 0.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 45.0));
        assert!(mat_close(&mid.a, &want.a, 1e-12), "{:?} vs {:?}", mid.a, want.a);
        assert!((mid.t.x).abs() < 1e-15 && (mid.t.y - 0.5).abs() < 1e-15);
        // 行列を直接線形補間すると、中間の行列は回転ではなく縮む（det = 0.5 < 1）: 極分解 + slerp では det = 1
        let lin = Mat3::from_rows([0, 1, 2].map(|i| [0, 1, 2].map(|j| 0.5 * (start.a.m[i][j] + end.a.m[i][j]))));
        let det = |a: &Mat3| {
            let m = a.m;
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        assert!((det(&lin) - 0.5).abs() < 1e-12);
        assert!((det(&mid.a) - 1.0).abs() < 1e-12);
    }

    /// 極分解は `A = R·S` を再構成し、`R` は回転（直交・det = +1）。非一様スケール + せん断 + 回転でも。
    #[test]
    fn polar_decomposition_reconstructs_general_affine() {
        let a = Transform::rotate(Vec3::new(0.3, 1.0, 0.5), 37.0)
            .compose(Transform::from_matrix4([[1.5, 0.4, 0.0, 0.0], [0.0, 0.8, 0.3, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]));
        let (r, s) = polar_decompose(&a.a.m).expect("decomposable");
        assert!(mat_close(&Mat3::from_rows(m3_mul(&r, &s)), &a.a, 1e-12));
        assert!(mat_close(&Mat3::from_rows(m3_mul(&r, &m3_t(&r))), &Mat3::identity(), 1e-12));
        assert!((m3_det(&r) - 1.0).abs() < 1e-12);
        assert!((0..3).all(|i| (0..3).all(|j| (s[i][j] - s[j][i]).abs() < 1e-12)), "S は対称");
    }

    /// 退化: 特異・鏡像・非有限は `None`（静止扱いに落とす）。開 = 閉、180° の回転は正しく補間できる。
    #[test]
    fn animated_degenerate_cases() {
        let id = Transform::identity();
        assert!(AnimatedTransform::new(id, Transform::scale(Vec3::new(1.0, 0.0, 1.0))).is_none(), "特異");
        assert!(AnimatedTransform::new(Transform::scale(Vec3::new(0.0, 0.0, 0.0)), id).is_none());
        assert!(AnimatedTransform::new(id, Transform::scale(Vec3::new(-1.0, 1.0, 1.0))).is_none(), "鏡像");
        assert!(AnimatedTransform::new(Transform::scale(Vec3::new(1.0, 1.0, -2.0)), id).is_none(), "鏡像（開側）");
        // 開と閉が同一: どの時刻でも同じ変換（丸めの範囲）
        let x = Transform::translate(Vec3::new(1.0, 2.0, 3.0)).compose(Transform::rotate(Vec3::new(1.0, 1.0, 0.0), 50.0));
        let same = AnimatedTransform::new(x, x).unwrap();
        for t in [0.0, 0.3, 0.77, 1.0] {
            assert!(mat_close(&same.at(t).a, &x.a, 1e-12) && (same.at(t).t.z - 3.0).abs() < 1e-12);
        }
        // 180° の回転（y 軸）: 中間は 90°、端点は正確
        let flip = AnimatedTransform::new(id, Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 180.0)).unwrap();
        assert!(mat_close(&flip.at(0.5).a, &Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 90.0).a, 1e-12));
        assert!(mat_close(&flip.at(0.25).a, &Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 45.0).a, 1e-12));
        // 360° 相当（開と閉が同じ向き）は静止に等しい（四元数の符号違いで逆回りしない）
        let full = AnimatedTransform::new(id, Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 360.0)).unwrap();
        assert!(mat_close(&full.at(0.5).a, &Mat3::identity(), 1e-9));
    }
}
