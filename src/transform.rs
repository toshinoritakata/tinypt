//! アフィン変換（任意の線形部 + 平行移動）。
//!
//! 線形部を `Mat3`、平行移動を `Vec3` で保持し、逆変換用に線形部の逆行列を、
//! 法線変換用に逆転置行列を事前計算する。任意軸回転・非均一スケール・任意の
//! 4×4 行列を表現できる。

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
}
