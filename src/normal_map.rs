//! ハイトマップ（`map_bump`）とノーマルマップ（`norm` など）による法線の摂動（S3: まだ積分器には繋がない）。
//!
//! 摂動後の法線は**シェーディング法線 `ns` 側**に入れるためのもので、幾何法線 `ng` は変えない
//! （原点のずらし・表裏判定・光源の面積/pdf は `ng`、BSDF と NEE の cos は `ns`）。
//! 摂動結果を `ng` と同じ側へ揃える `face_forward` は呼び出し側で行う。
//!
//! ## バンプ強度の意味（スケール不変・テクセル正規化）
//! 傾きは `su = strength·hu·g / |∂p/∂u|`, `sv = strength·hv·g / |∂p/∂v|`
//! （`hu, hv` = **1 テクセルあたり**の高さ勾配 = UV 単位の勾配 ÷ マップの幅/高さ、`g = √(|∂p/∂u|·|∂p/∂v|)`）とする。
//! - **テクセル正規化**: UV 単位の勾配のまま使うと、512px のマップでテクセルあたり Δh=0.05 の縁が `hu ≈ 25` になり、
//!   `strength = 1` で `SLOPE_MAX` に張り付く（`-bm 1` が MTL の標準的な値なのに使えない）。1 テクセルあたりに直すと
//!   `strength = 1` は「隣のテクセルと高さが Δh 違えば、傾きは Δh（テクセル 1 個ぶんの水平距離あたり）」という、
//!   ちょうど 45° に達する（全レンジの段差で）扱いやすい基準になる。
//! - `1/|∂p/∂u|` は UV の引き伸ばし（異方性）の補正。
//! - `g` を掛けるので、同じモデルを一様に k 倍して置くと `|∂p/∂u|`, `|∂p/∂v|`, `g` が全部 k 倍になり、
//!   `su, sv` は**変わらない**。PBRT 流（`pu = dpdu + bm·hu·ns` の外積）は `bm` がワールド長の単位を持つので、
//!   cm 単位の Sponza を `to_world` で 0.01 倍した途端に起伏の見え方が変わってしまう。このレンダラーは
//!   「同じモデルを別スケールで置いても結果が変わらない」ことを保証してきたので、`strength` は
//!   無次元（UV 空間での傾きの倍率）として扱う。
//! - 傾きは `±SLOPE_MAX`（tan 85°）でクランプする。

use crate::constants::normal_map::SLOPE_MAX;
use crate::math::Vec3;
use crate::texture::{Texture, Wrap};

/// ハイトマップ。1 テクセル 1 バイト（`AlphaMask` と同じ形）。値は 0..1 の高さ。
pub struct HeightMap {
    width: usize,
    height: usize,
    data: Vec<u8>,
    wrap: Wrap,
}

impl HeightMap {
    /// 0..255 の値列（行優先、1 行目が上端）から作る（テスト・合成用）。
    pub fn from_u8(width: usize, height: usize, data: Vec<u8>, wrap: Wrap) -> Self {
        assert_eq!(data.len(), width * height, "texel count must be width * height");
        Self { width, height, data, wrap }
    }

    /// 画像ファイルから読む。値はデータ（リニア）として、RGB の Rec.709 輝度（MTL 既定の `-imfchan l`）を使う。
    /// Sponza の `map_bump` は全てグレースケール（RGB 3ch のファイルも中身はグレー）。
    pub fn load(path: &str, wrap: Wrap) -> Result<Self, String> {
        let img = image::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        if w == 0 || h == 0 {
            return Err(format!("{}: empty image", path));
        }
        let data = rgb
            .pixels()
            .map(|p| (0.2126 * p.0[0] as f64 + 0.7152 * p.0[1] as f64 + 0.0722 * p.0[2] as f64).round() as u8)
            .collect();
        Ok(Self { width: w, height: h, data, wrap })
    }

    fn texel(&self, x: i64, y: i64) -> f64 {
        let (w, h) = (self.width as i64, self.height as i64);
        let (x, y) = match self.wrap {
            Wrap::Repeat => (x.rem_euclid(w), y.rem_euclid(h)),
            Wrap::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
        };
        self.data[(y as usize) * self.width + (x as usize)] as f64 / 255.0
    }

    /// UV でバイリニア補間した高さ（0..1）。座標の規約は [`Texture::sample`] と同じ（v は下から上）。
    /// 非有限の UV は 0。
    pub fn sample(&self, uv: (f64, f64)) -> f64 {
        let (u, v) = uv;
        if !u.is_finite() || !v.is_finite() {
            return 0.0;
        }
        let x = u * self.width as f64 - 0.5;
        let y = (1.0 - v) * self.height as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = self.texel(x0, y0) * (1.0 - fx) + self.texel(x0 + 1, y0) * fx;
        let bottom = self.texel(x0, y0 + 1) * (1.0 - fx) + self.texel(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// マップの大きさ（テクセル）。
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// UV 単位あたりの高さ勾配 `(∂h/∂u, ∂h/∂v)`。中心差分で、差分の幅は**ちょうど 1 テクセル**
    /// （`±0.5` テクセル）。半テクセル未満だと双一次補間の折れ目で勾配が跳ね、テクセル格子のモアレが出る。
    pub fn gradient(&self, uv: (f64, f64)) -> (f64, f64) {
        let (u, v) = uv;
        let (du, dv) = (1.0 / self.width as f64, 1.0 / self.height as f64);
        let hu = (self.sample((u + 0.5 * du, v)) - self.sample((u - 0.5 * du, v))) / du;
        let hv = (self.sample((u, v + 0.5 * dv)) - self.sample((u, v - 0.5 * dv))) / dv;
        (hu, hv)
    }
}

/// 法線を摂動するマップ。`Scene.textures`（色）とは別に持つ（sRGB の誤適用を型で防ぐ）。
pub enum NormalMap {
    /// タンジェント空間ノーマルマップ（RGB。**必ずリニア（raw）で読んだ** `Texture`）。
    /// `scale` は x, y 成分の倍率（1 で等倍、0 で平ら）。
    Tangent { tex: Texture, scale: f64 },
    /// ハイトマップによるバンプ。`strength` は無次元の傾き倍率（上記「スケール不変」参照）。
    Height { map: HeightMap, strength: f64 },
}

/// 材質側テーブル（`Scene::mat_maps`）から `Scene::normal_maps` を引く添字。
pub type MapId = u32;

impl NormalMap {
    /// 接空間 `(t, b, ns)`（正規直交、`ns` は単位・向き合わせ済み）での摂動後の法線（単位ベクトル）。
    /// `dpdu_len` / `dpdv_len` は接ベクトルの長さ（バンプの無次元化に使う）。
    /// 退化入力（非有限、長さ 0）では `ns` をそのまま返す。
    pub fn perturb(&self, uv: (f64, f64), t: Vec3, b: Vec3, ns: Vec3, dpdu_len: f64, dpdv_len: f64) -> Vec3 {
        let n = match self {
            NormalMap::Tangent { tex, scale } => {
                let c = tex.sample(uv);
                let (x, y, z) = (2.0 * c.r() - 1.0, 2.0 * c.g() - 1.0, 2.0 * c.b() - 1.0);
                if !((x * x + y * y + z * z) > 0.0) {
                    return ns;
                }
                t * (x * scale) + b * (y * scale) + ns * z
            }
            NormalMap::Height { map, strength } => {
                if !(dpdu_len > 0.0 && dpdv_len > 0.0 && dpdu_len.is_finite() && dpdv_len.is_finite()) {
                    return ns;
                }
                // UV 単位の勾配を 1 テクセルあたりに直す（解像度に依らず strength = 1 が扱える範囲になる）
                let (hu, hv) = map.gradient(uv);
                let (w, h) = map.size();
                let (hu, hv) = (hu / w as f64, hv / h as f64);
                let g = (dpdu_len * dpdv_len).sqrt();
                let su = (strength * hu * g / dpdu_len).clamp(-SLOPE_MAX, SLOPE_MAX);
                let sv = (strength * hv * g / dpdv_len).clamp(-SLOPE_MAX, SLOPE_MAX);
                if su == 0.0 && sv == 0.0 {
                    return ns; // 平らなら bit 単位で ns のまま（再正規化で末尾がずれない）
                }
                ns - t * su - b * sv
            }
        };
        let len = n.len();
        if len > 0.0 && len.is_finite() {
            n / len
        } else {
            ns
        }
    }
}

/// `∂p/∂u`, `∂p/∂v`（ワールド空間、長さを保つ）とシェーディング法線 `ns` から、正規直交な接空間 `(t, b)` を作る。
///
/// `t` は `dpdu` を `ns` に直交させて正規化したもの、`b = ns × t` に**利き手の符号**を掛けたもの。
/// 利き手は `dpdu × dpdv` と `ns` の向きで決める（UV が鏡像に貼られていれば b が反転する）。
/// 符号は**変換後（ワールド空間）**の接ベクトルで計算するので、鏡像インスタンスでも正しく反転する。
/// `dpdu` が `ns` と平行、または非有限なら `None`（摂動しない）。
pub fn orthonormalize(dpdu: Vec3, dpdv: Vec3, ns: Vec3) -> Option<(Vec3, Vec3)> {
    let t = dpdu - ns * ns.dot(dpdu);
    let len = t.len();
    if !(len > 0.0 && len.is_finite()) {
        return None;
    }
    let t = t / len;
    let sign = if dpdu.cross(dpdv).dot(ns) < 0.0 { -1.0 } else { 1.0 };
    Some((t, ns.cross(t) * sign))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> (Vec3, Vec3, Vec3) {
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0))
    }

    /// 幅 255 のランプ（u 方向）: テクセル i の値 = i。
    fn ramp_u() -> HeightMap {
        let data: Vec<u8> = (0..255u32 * 4).map(|k| (k % 255) as u8).collect();
        HeightMap::from_u8(255, 4, data, Wrap::Clamp)
    }

    /// 高さが v とともに増える縦ランプ（画像の下の行ほど大きい = v が大きいほど高い）。
    fn ramp_v() -> HeightMap {
        // 高さ 255 行、上の行から: 254, 253, ..., 0 ではなく「v 上向きで増加」なので上の行ほど大きい
        let h = 255usize;
        let mut data = Vec::with_capacity(h * 4);
        for row in 0..h {
            let val = (h - 1 - row) as u8; // 上の行（row 小）ほど大きい
            data.extend([val; 4]);
        }
        HeightMap::from_u8(4, h, data, Wrap::Clamp)
    }

    #[test]
    fn gradient_of_linear_ramps_matches_slope_and_v_convention() {
        let m = ramp_u();
        let (hu, hv) = m.gradient((0.5, 0.5));
        assert!((hu - 1.0).abs() < 1e-9 && hv.abs() < 1e-12, "{:?}", (hu, hv));
        let m = ramp_v();
        let (hu, hv) = m.gradient((0.5, 0.5));
        assert!(hu.abs() < 1e-12 && (hv - 1.0).abs() < 1e-9, "v は上向き: {:?}", (hu, hv));
    }

    #[test]
    fn flat_height_map_returns_ns_bit_exactly() {
        let (t, b, _) = frame();
        let ns = Vec3::new(0.3, -0.5, 0.8).norm();
        let m = NormalMap::Height { map: HeightMap::from_u8(2, 2, vec![90; 4], Wrap::Repeat), strength: 5.0 };
        let n = m.perturb((0.3, 0.7), t, b, ns, 1.0, 1.0);
        assert_eq!((n.x.to_bits(), n.y.to_bits(), n.z.to_bits()), (ns.x.to_bits(), ns.y.to_bits(), ns.z.to_bits()));
    }

    #[test]
    fn height_ramp_tilts_against_the_gradient() {
        let (t, b, ns) = frame();
        let m = NormalMap::Height { map: ramp_u(), strength: 0.5 };
        let n = m.perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        // ∂h/∂u = 1（UV 単位）= 1/255 テクセルあたり → 法線は -t 側へ、傾き 0.5/255
        assert!(n.x < 0.0 && n.y.abs() < 1e-12);
        assert!((n.x / n.z + 0.5 / 255.0).abs() < 1e-9, "{}", n.x / n.z);
        assert!((n.len() - 1.0).abs() < 1e-12);
    }

    /// 接ベクトルの長さを 100 倍しても（= モデルを 100 倍のスケールで置いても）同じ摂動になる。
    /// これが `strength` を無次元にした意味。異方性（|dpdu| ≠ |dpdv|）でも成り立つ。
    #[test]
    fn height_perturbation_is_invariant_to_uniform_scaling() {
        let (t, b, ns) = frame();
        let h = HeightMap::from_u8(3, 3, vec![0, 40, 90, 10, 60, 120, 30, 80, 200], Wrap::Repeat);
        let m = NormalMap::Height { map: h, strength: 1.0 };
        let (a, c) = (0.7, 2.3);
        let n1 = m.perturb((0.4, 0.6), t, b, ns, a, c);
        for k in [0.01, 100.0] {
            let n2 = m.perturb((0.4, 0.6), t, b, ns, a * k, c * k);
            assert!((n1 - n2).len() < 1e-12, "k={}: {:?} vs {:?}", k, (n1.x, n1.y, n1.z), (n2.x, n2.y, n2.z));
        }
        // 傾きが 0 でも飽和でもないことを確認（自明な一致を避ける）
        assert!(n1.z < 1.0 - 1e-6 && n1.z > 0.5);
    }

    /// 極端なハイトマップでも、傾きは tan 85° で頭打ち（法線は地平線から 5° 以上離れる）。
    #[test]
    fn slope_is_clamped_to_slope_max() {
        let (t, b, ns) = frame();
        let m = NormalMap::Height { map: HeightMap::from_u8(2, 1, vec![0, 255], Wrap::Clamp), strength: 1.0e6 };
        let n = m.perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        let slope = (n.x * n.x + n.y * n.y).sqrt() / n.z;
        assert!((slope - SLOPE_MAX).abs() < 1e-9, "{}", slope);
        assert!(n.z > 0.08); // cos 85° ≈ 0.087
    }

    fn tangent_map(rgb: [u8; 3], scale: f64) -> NormalMap {
        NormalMap::Tangent { tex: Texture::from_texels_u8(1, 1, rgb.to_vec(), false, Wrap::Repeat), scale }
    }

    #[test]
    fn tangent_map_flat_tilts_to_plus_t_and_is_right_handed() {
        let (t, b, ns) = frame();
        // (128,128,255) は 8bit では厳密な平坦が表せない（2·128/255 − 1 ≈ 0.004）ので、約 0.2° 以内なら平坦
        let n = tangent_map([128, 128, 255], 1.0).perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        assert!((n - ns).len() < 0.01, "{:?}", (n.x, n.y, n.z));
        // (255,128,128) は +t 方向へ倒れる（x 成分 +1、z は ≈ 0）
        let n = tangent_map([255, 128, 128], 1.0).perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        assert!(n.x > 0.99, "{:?}", (n.x, n.y, n.z));
        // 利き手: b を反転すると結果の b 成分（y）の符号が反転する
        let up = tangent_map([128, 255, 200], 1.0);
        let n1 = up.perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        let n2 = up.perturb((0.5, 0.5), t, -b, ns, 1.0, 1.0);
        assert!(n1.y > 0.5 && n2.y < -0.5 && (n1.y + n2.y).abs() < 1e-12);
        // scale = 0 は平ら
        let n = tangent_map([255, 255, 128], 0.0).perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
        assert!((n - ns).len() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_do_not_panic_and_fall_back_to_ns() {
        let (t, b, ns) = frame();
        let h = NormalMap::Height { map: ramp_u(), strength: 1.0 };
        for (u, v) in [(f64::NAN, 0.5), (0.5, f64::INFINITY)] {
            let n = h.perturb((u, v), t, b, ns, 1.0, 1.0);
            assert!(n.x.is_finite() && (n.len() - 1.0).abs() < 1e-12);
        }
        for (a, c) in [(0.0, 1.0), (1.0, 0.0), (f64::NAN, 1.0), (f64::INFINITY, 1.0)] {
            assert_eq!(h.perturb((0.5, 0.5), t, b, ns, a, c).z, 1.0);
        }
        let zero = Vec3::new(0.0, 0.0, 0.0);
        // 接ベクトルが 0 でも（長さ 0 の入力）パニックしない
        let _ = h.perturb((0.5, 0.5), zero, zero, ns, 1.0, 1.0);
        let tm = tangent_map([0, 0, 0], 1.0);
        let _ = tm.perturb((0.5, 0.5), t, b, ns, 1.0, 1.0);
    }
}
