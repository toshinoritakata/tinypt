//! シェーダー: 交差点で材質のパラメータを計算する層（`Material` の手前）。
//!
//! ```text
//! Shader（基底の Material + アルベドの式 + 法線マップ）
//!     ↓ 交差点ごとに 1 回 evaluate
//! Material（具体的な値。既存の sample / eval はこれだけを知る）
//! ```
//!
//! 以前は、テクスチャが `Material` の variant の中（`Lambert::albedo_tex`）、法線のマップが `mat_id` 並びの側テーブル、
//! 手続き的ノイズが `TexId` の最上位ビットへの相乗り、と置き場所がばらばらだった。ここに 1 つに畳んである
//! （設計の経緯と却下した案は `docs/adr/0004-shader-layer-before-material.md`）。
//!
//! **式（[`ValueNode`]）** は小さな平坦なアリーナ（`Vec<ValueNode>`、子は添字）で持つ: `Const`、`Texture`（UV で引く画像。
//! sRGB / raw はロード時に `Texture` が決めている）、`Noise`（位置で引く手続きノイズ）、`Mul`、`Add`、`Mix`。
//! 式はアルベド・粗さ・屈折率・吸収（と API だけの放射輝度）のどれにも付けられる。法線の摂動は複数枚を順に適用する。
//!
//! **早道**: 式を持たない材質（`has_expr == false`）は、基底の `Material` をそのまま返す（コピー 1 回。積分器から見た
//! コストは以前の `resolve_textures` が `other => other` を返すのと同じ）。式が要るのは、テクスチャまたはノイズを
//! 持つ材質だけ。

use crate::geometry::Hit;
use crate::material::{Material, TexId};
use crate::math::Color;
use crate::noise::NoiseTexture;
use crate::normal_map::{MapId, NormalMap};
use crate::texture::Texture;
use crate::world::World;

/// `ShaderSet::values` への添字。
pub type ValueId = u32;
/// `ShaderSet::noises` への添字。
pub type NoiseId = u32;

/// 式のノード（子は `ValueId`）。値はいつも色（3 成分）。スカラのパラメータ（`alpha`、`ior`、`Mix` の `t`）は
/// **第 1 成分（`r`）を取り出して使う**（`Const(Color::splat(x))` がスカラ `x`。グレースケールの画像やノイズはそのまま使える）。
/// 輝度ではなく `r` にしたのは、成分がそろった値では `r` が値そのもの（輝度は `0.2126 + 0.7152 + 0.0722` の丸めで
/// 1 ビットずれる）で、材質の定数と式が同じ値を返すため。`Scalar` ノードを別に作らないのも同じ理由（`Const` で足りる）。
#[derive(Clone, Copy, Debug)]
pub enum ValueNode {
    /// 定数の色
    Const(Color),
    /// 画像テクスチャ（`ShaderSet::textures` の添字）を交差点の UV で引く
    Texture(TexId),
    /// 手続き的ノイズ（`ShaderSet::noises` の添字）を交差点の位置（ローカル / ワールドはノイズが持つ）で引く
    Noise(NoiseId),
    /// 成分ごとの積（左 × 右）
    Mul(ValueId, ValueId),
    /// 成分ごとの和（左 + 右）
    Add(ValueId, ValueId),
    /// `a·(1 − t) + b·t`。`t` は式の第 1 成分を [0, 1] に収めたもの（範囲外でも値が発散しないように）
    Mix(ValueId, ValueId, ValueId),
}

/// 式の評価が入れ子をたどる最大の深さ。循環参照・不正な木で無限再帰しないための保護（超えたら白）。
const MAX_EXPR_DEPTH: u32 = 32;

/// ローダーがテクスチャ / ノイズを指すときの参照（式を組み立てるのに使う）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexRef {
    Image(TexId),
    Noise(NoiseId),
}

/// 1 つの材質のシェーダー。
#[derive(Clone, Copy)]
pub struct Shader {
    /// 基底の材質（具体的な値。式が無ければこれがそのまま評価結果）。発光の判定などにも使う
    pub base: Material,
    /// アルベド（Lambert / Metal / Ggx / Subsurface の反射率）を置き換える式。`None` なら `base` のまま
    pub albedo: Option<ValueId>,
    /// `Ggx` の粗さ `alpha`（スカラ。`[1e-3, 1]` に収める）
    pub alpha: Option<ValueId>,
    /// `Dielectric` の屈折率（スカラ。正に保つ）
    pub ior: Option<ValueId>,
    /// `Dielectric` の吸収（色。負にしない）
    pub absorption: Option<ValueId>,
    /// `DiffuseLight` の放射輝度（色。負にしない）。**光源のサンプリング（NEE）は基底の放射輝度を前提にしている**ので、
    /// 場所で変わる式を付けると NEE と BSDF 側で放射輝度が食い違う（XML からは付けられない。API だけ）
    pub emit: Option<ValueId>,
    /// 式を 1 つでも持つか（早道の判定。false なら `base` をそのまま返す）
    has_expr: bool,
    /// 法線の摂動の列（`ShaderSet::normal_chain` の `[start, start + len)`。先頭から順に適用する）
    normal_start: u32,
    normal_len: u32,
}

/// 材質が持ちうる式の集まり（`ShaderSet::push_shader` に渡す）。
#[derive(Clone, Copy, Default)]
pub struct Exprs {
    pub albedo: Option<ValueId>,
    pub alpha: Option<ValueId>,
    pub ior: Option<ValueId>,
    pub absorption: Option<ValueId>,
    pub emit: Option<ValueId>,
}

/// 負の成分を 0 にする（吸収・放射輝度）。NaN も 0。
fn non_negative(c: Color) -> Color {
    let f = |x: f64| if x > 0.0 { x } else { 0.0 };
    Color::new(f(c.r()), f(c.g()), f(c.b()))
}

/// 評価の文脈: 交差点（UV・位置・ローカル座標の計算に要る `World`）と時刻。
pub struct ShadeCtx<'a> {
    pub world: &'a World,
    pub hit: &'a Hit,
    pub time: f64,
}

/// シーンの材質・式・テクスチャ・ノイズ・法線マップを 1 か所に持つ。`mat_id` が `shaders` の添字。
#[derive(Default)]
pub struct ShaderSet {
    pub shaders: Vec<Shader>,
    values: Vec<ValueNode>,
    /// 画像テクスチャ
    pub textures: Vec<Texture>,
    /// 手続き的ノイズ
    pub noises: Vec<NoiseTexture>,
    /// 法線を摂動するマップ
    pub normal_maps: Vec<NormalMap>,
    /// 材質ごとの法線の摂動の列（`Shader::normal_start` / `normal_len` が指す。**順に適用**する）
    normal_chain: Vec<MapId>,
}

impl ShaderSet {
    /// 定数だけの材質の列から作る（式もマップも無し）。
    pub fn from_materials(mats: &[Material]) -> Self {
        let mut set = Self::default();
        for &m in mats {
            set.push_shader(m, Exprs::default(), &[]);
        }
        set
    }

    /// 材質を 1 つ足して `mat_id` を返す。`tex` があれば `Mul(Const(基底のアルベド), テクスチャ)` の式を作る
    /// （定数の色はテクスチャの倍率）。
    pub fn push(&mut self, base: Material, tex: Option<TexRef>, normal: Option<MapId>) -> usize {
        let albedo = match (tex, base.albedo()) {
            (Some(t), Some(factor)) => {
                let c = self.add_value(ValueNode::Const(factor));
                let v = self.add_value(match t {
                    TexRef::Image(id) => ValueNode::Texture(id),
                    TexRef::Noise(id) => ValueNode::Noise(id),
                });
                Some(self.add_value(ValueNode::Mul(c, v)))
            }
            _ => None,
        };
        let normals: Vec<MapId> = normal.into_iter().collect();
        self.push_shader(base, Exprs { albedo, ..Exprs::default() }, &normals)
    }

    /// 材質を 1 つ足して `mat_id` を返す（式とマップの列を直接与える版）。
    pub fn push_shader(&mut self, base: Material, e: Exprs, normals: &[MapId]) -> usize {
        let normal_start = self.normal_chain.len() as u32;
        self.normal_chain.extend_from_slice(normals);
        let has_expr = e.albedo.is_some() || e.alpha.is_some() || e.ior.is_some() || e.absorption.is_some() || e.emit.is_some();
        self.shaders.push(Shader { base, albedo: e.albedo, alpha: e.alpha, ior: e.ior, absorption: e.absorption, emit: e.emit, has_expr, normal_start, normal_len: normals.len() as u32 });
        self.shaders.len() - 1
    }

    /// 式のアリーナを差し替える（ローダーが組み立て済みのノードを渡す）。
    pub fn set_values(&mut self, values: Vec<ValueNode>) {
        self.values = values;
    }

    /// 式のノードを足して添字を返す。**子は既に足した（添字が小さい）ノードだけを指せる**: そうでない添字を持つノードは
    /// 白の定数に置き換える。これで式は必ず非循環（足した順が評価の順）になり、循環参照で評価が終わらなくなることがない。
    pub fn add_value(&mut self, node: ValueNode) -> ValueId {
        let n = self.values.len() as ValueId;
        let ok = match node {
            ValueNode::Mul(a, b) | ValueNode::Add(a, b) => a < n && b < n,
            ValueNode::Mix(a, b, t) => a < n && b < n && t < n,
            _ => true,
        };
        self.values.push(if ok { node } else { ValueNode::Const(Color::new(1.0, 1.0, 1.0)) });
        (self.values.len() - 1) as ValueId
    }

    /// 基底の材質の列（`World::build_lights` が発光の判定に使う）。
    pub fn base_materials(&self) -> Vec<Material> {
        self.shaders.iter().map(|s| s.base).collect()
    }

    /// 材質 `mat_id` を交差点で評価して、具体的な `Material` にする。式が無ければ基底をそのまま返す（早道）。
    #[inline(always)]
    pub fn evaluate(&self, mat_id: usize, ctx: &ShadeCtx) -> Material {
        let sh = &self.shaders[mat_id];
        if !sh.has_expr {
            return sh.base;
        }
        self.evaluate_exprs(sh, ctx)
    }

    /// 式を持つ材質の評価（早道の外）。各パラメータは既存の値域に収める: `alpha` は `[1e-3, 1]`、`ior` は正
    /// （`>= 1e-3`）、吸収と放射輝度は負にしない。NaN は下限に倒す。
    #[inline(never)]
    fn evaluate_exprs(&self, sh: &Shader, ctx: &ShadeCtx) -> Material {
        let mut m = sh.base;
        if let Some(id) = sh.albedo {
            m = m.with_albedo(self.eval_value(id, ctx, 0));
        }
        if let Some(id) = sh.alpha {
            let v = self.eval_value(id, ctx, 0).r();
            m = m.with_alpha(if v.is_nan() { 1e-3 } else { v.clamp(1e-3, 1.0) });
        }
        if let Some(id) = sh.ior {
            let v = self.eval_value(id, ctx, 0).r();
            m = m.with_ior(if v.is_nan() { 1e-3 } else { v.max(1e-3) });
        }
        if let Some(id) = sh.absorption {
            m = m.with_absorption(non_negative(self.eval_value(id, ctx, 0)));
        }
        if let Some(id) = sh.emit {
            m = m.with_emit(non_negative(self.eval_value(id, ctx, 0)));
        }
        m
    }

    /// 式のノード（テスト・診断用）。
    pub fn value(&self, id: ValueId) -> ValueNode {
        self.values[id as usize]
    }

    /// 材質 `mat_id` の法線の摂動の列（先頭から順に適用する。空なら摂動なし）。
    #[inline(always)]
    pub fn normal_chain(&self, mat_id: usize) -> &[MapId] {
        let sh = &self.shaders[mat_id];
        &self.normal_chain[sh.normal_start as usize..(sh.normal_start + sh.normal_len) as usize]
    }

    /// 材質 `mat_id` の最初の法線マップ（テスト・診断用）。
    pub fn normal_map(&self, mat_id: usize) -> Option<MapId> {
        self.normal_chain(mat_id).first().copied()
    }

    /// 式を評価する。範囲外の添字（読み込みに失敗したテクスチャなど）と、入れ子が深すぎる式（循環参照）は
    /// 白（1 倍）として扱う（以前の `resolve_textures` が範囲外の添字を「テクスチャ無し」にしたのと同じ）。
    fn eval_value(&self, id: ValueId, ctx: &ShadeCtx, depth: u32) -> Color {
        let white = Color::new(1.0, 1.0, 1.0);
        if depth > MAX_EXPR_DEPTH {
            return white;
        }
        match self.values.get(id as usize) {
            None => white,
            Some(&ValueNode::Const(c)) => c,
            Some(&ValueNode::Texture(t)) => match self.textures.get(t as usize) {
                Some(tex) => tex.sample(ctx.hit.uv),
                None => white,
            },
            Some(&ValueNode::Noise(n)) => match self.noises.get(n as usize) {
                Some(noise) => {
                    let p = if noise.local { ctx.world.object_space_point(ctx.hit, ctx.time) } else { ctx.hit.p };
                    noise.eval(p)
                }
                None => white,
            },
            Some(&ValueNode::Mul(a, b)) => self.eval_value(a, ctx, depth + 1).hadamard(self.eval_value(b, ctx, depth + 1)),
            Some(&ValueNode::Add(a, b)) => self.eval_value(a, ctx, depth + 1) + self.eval_value(b, ctx, depth + 1),
            Some(&ValueNode::Mix(a, b, t)) => {
                let tv = self.eval_value(t, ctx, depth + 1).r();
                let tv = if tv.is_nan() { 0.0 } else { tv.clamp(0.0, 1.0) };
                self.eval_value(a, ctx, depth + 1) * (1.0 - tv) + self.eval_value(b, ctx, depth + 1) * tv
            }
        }
    }

    /// テスト用: 交差点の文脈なしで式を評価する（位置 `p`・UV `uv` を直接与える）。
    #[cfg(test)]
    pub fn eval_at(&self, mat_id: usize, world: &World, p: crate::math::Vec3, uv: (f64, f64)) -> Material {
        let hit = Hit { t: 0.0, p, ng: crate::math::Vec3::new(0.0, 0.0, 1.0), ns: crate::math::Vec3::new(0.0, 0.0, 1.0), mat_id, prim_id: 0, inst_id: None, p_error: crate::math::Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv };
        self.evaluate(mat_id, &ShadeCtx { world, hit: &hit, time: 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::texture::Wrap;

    fn white_world() -> World {
        World::new()
    }

    /// 式の無い材質は基底の `Material` をそのまま返す（早道）。
    #[test]
    fn constant_material_is_returned_unchanged() {
        let m = Material::Ggx { albedo: Color::new(0.1, 0.2, 0.3), alpha: 0.25 };
        let set = ShaderSet::from_materials(&[m]);
        assert!(set.shaders[0].albedo.is_none());
        match set.eval_at(0, &white_world(), Vec3::new(1.0, 2.0, 3.0), (0.3, 0.7)) {
            Material::Ggx { albedo, alpha } => assert_eq!((albedo.r(), albedo.g(), albedo.b(), alpha), (0.1, 0.2, 0.3, 0.25)),
            _ => panic!(),
        }
    }

    /// `push` は `Mul(Const(倍率), Texture)` を作り、評価は `倍率.hadamard(テクスチャの値)` とビット単位で同じ
    /// （以前の `resolve_textures` と同じ演算）。
    #[test]
    fn texture_is_a_product_of_the_factor_and_the_image() {
        let mut set = ShaderSet::default();
        set.textures.push(Texture::from_texels_u8(2, 1, vec![255, 128, 10, 0, 64, 200], true, Wrap::Repeat));
        let factor = Color::new(0.25, 0.5, 1.0);
        let id = set.push(Material::Lambert { albedo: factor }, Some(TexRef::Image(0)), None);
        let uv = (0.3, 0.6);
        let want = factor.hadamard(set.textures[0].sample(uv));
        match set.eval_at(id, &white_world(), Vec3::new(0.0, 0.0, 0.0), uv) {
            Material::Lambert { albedo } => {
                assert_eq!((albedo.r().to_bits(), albedo.g().to_bits(), albedo.b().to_bits()), (want.r().to_bits(), want.g().to_bits(), want.b().to_bits()));
            }
            _ => panic!(),
        }
        assert!(matches!(set.value(set.shaders[id].albedo.unwrap()), ValueNode::Mul(..)));
    }

    /// ノイズは位置で引き、範囲外の添字は白（1 倍）= 基底の色のまま。反射率を持たない材質には式を付けない。
    #[test]
    fn noise_is_read_by_position_and_bad_indices_are_white() {
        use crate::noise::{NoiseTexture, Pattern};
        let mut set = ShaderSet::default();
        set.noises.push(NoiseTexture { pattern: Pattern::Fbm, scale: 3.0, octaves: 3, lacunarity: 2.0, gain: 0.5, strength: 1.0, color0: Color::new(0.0, 0.0, 0.0), color1: Color::new(1.0, 1.0, 1.0), local: false, offset: Vec3::new(0.0, 0.0, 0.0) });
        let id = set.push(Material::Lambert { albedo: Color::new(1.0, 1.0, 1.0) }, Some(TexRef::Noise(0)), None);
        let w = white_world();
        let (a, b) = (Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.7, -0.4, 2.2));
        let val = |p: Vec3| match set.eval_at(id, &w, p, (0.0, 0.0)) { Material::Lambert { albedo } => albedo.r(), _ => panic!() };
        assert_ne!(val(a).to_bits(), val(b).to_bits());
        assert_eq!(val(a).to_bits(), set.noises[0].eval(a).r().to_bits());
        // 範囲外のノイズ / テクスチャ: 白（倍率だけが残る）
        let bad = set.push(Material::Lambert { albedo: Color::new(0.4, 0.5, 0.6) }, Some(TexRef::Noise(9)), None);
        match set.eval_at(bad, &w, a, (0.0, 0.0)) { Material::Lambert { albedo } => assert_eq!((albedo.r(), albedo.g()), (0.4, 0.5)), _ => panic!() }
        let bad = set.push(Material::Lambert { albedo: Color::new(0.4, 0.5, 0.6) }, Some(TexRef::Image(9)), None);
        match set.eval_at(bad, &w, a, (0.0, 0.0)) { Material::Lambert { albedo } => assert_eq!((albedo.r(), albedo.g()), (0.4, 0.5)), _ => panic!() }
        // ガラス・発光は反射率を持たないので式は付かない
        let g = set.push(Material::Dielectric { ior: 1.5, absorption: Color::new(0.0, 0.0, 0.0) }, Some(TexRef::Noise(0)), Some(3));
        assert!(set.shaders[g].albedo.is_none() && set.normal_map(g) == Some(3));
        assert_eq!(set.base_materials().len(), 4);
    }

    fn hit_at(p: Vec3, uv: (f64, f64)) -> Hit {
        Hit { t: 0.0, p, ng: Vec3::new(0.0, 0.0, 1.0), ns: Vec3::new(0.0, 0.0, 1.0), mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv }
    }
    fn eval_root(set: &ShaderSet, id: ValueId) -> Color {
        let w = World::new();
        let hit = hit_at(Vec3::new(0.3, 0.2, 0.1), (0.4, 0.6));
        set.eval_value(id, &ShadeCtx { world: &w, hit: &hit, time: 0.0 }, 0)
    }
    fn c(x: f64, y: f64, z: f64) -> ValueNode {
        ValueNode::Const(Color::new(x, y, z))
    }
    fn bits(col: Color) -> (u64, u64, u64) {
        (col.r().to_bits(), col.g().to_bits(), col.b().to_bits())
    }

    /// `Add` / `Mix` / 入れ子の値: `Mix` は `a·(1−t) + b·t`（`t` は第 1 成分を [0,1] に収めたもの）で、演算は素直な式とビット一致。
    #[test]
    fn add_mix_and_nesting_evaluate_as_written() {
        let mut set = ShaderSet::default();
        let a = set.add_value(c(0.2, 0.4, 0.8));
        let b = set.add_value(c(1.0, 0.0, 0.5));
        let sum = set.add_value(ValueNode::Add(a, b));
        assert_eq!(bits(eval_root(&set, sum)), bits(Color::new(0.2, 0.4, 0.8) + Color::new(1.0, 0.0, 0.5)));
        let t = set.add_value(c(0.25, 9.0, 9.0)); // 第 1 成分だけが使われる
        let mix = set.add_value(ValueNode::Mix(a, b, t));
        assert_eq!(bits(eval_root(&set, mix)), bits(Color::new(0.2, 0.4, 0.8) * 0.75 + Color::new(1.0, 0.0, 0.5) * 0.25));
        // t は [0,1] に収める: 2 → b、−1 → a、NaN → a
        for (tv, want) in [(2.0, Color::new(1.0, 0.0, 0.5)), (-1.0, Color::new(0.2, 0.4, 0.8)), (f64::NAN, Color::new(0.2, 0.4, 0.8))] {
            let t = set.add_value(c(tv, 0.0, 0.0));
            let m = set.add_value(ValueNode::Mix(a, b, t));
            let got = eval_root(&set, m);
            assert!((got.r() - want.r()).abs() < 1e-12 && (got.b() - want.b()).abs() < 1e-12);
        }
        // 入れ子: Mul(Add(a, b), Mix(a, b, t))
        let prod = set.add_value(ValueNode::Mul(sum, mix));
        let want = (Color::new(0.2, 0.4, 0.8) + Color::new(1.0, 0.0, 0.5)).hadamard(Color::new(0.2, 0.4, 0.8) * 0.75 + Color::new(1.0, 0.0, 0.5) * 0.25);
        assert_eq!(bits(eval_root(&set, prod)), bits(want));
    }

    fn scalar_shader(base: Material, expr: impl Fn(&mut ShaderSet) -> Exprs) -> (ShaderSet, usize) {
        let mut set = ShaderSet::default();
        let e = expr(&mut set);
        let id = set.push_shader(base, e, &[]);
        (set, id)
    }

    /// スカラのパラメータは式の第 1 成分から取り、既存の値域に収める: `alpha` は [1e-3, 1]、`ior` は正、吸収と放射輝度は負にしない。
    /// NaN は下限に倒す。
    #[test]
    fn parameter_expressions_are_clamped_to_their_ranges() {
        let w = World::new();
        let ggx = Material::Ggx { albedo: Color::new(0.5, 0.5, 0.5), alpha: 0.3 };
        for (v, want) in [(0.4, 0.4), (5.0, 1.0), (-2.0, 1e-3), (0.0, 1e-3), (f64::NAN, 1e-3), (f64::INFINITY, 1.0)] {
            let (set, id) = scalar_shader(ggx, |s| Exprs { alpha: Some(s.add_value(c(v, 9.0, 9.0))), ..Exprs::default() });
            match set.eval_at(id, &w, Vec3::new(0.0, 0.0, 0.0), (0.0, 0.0)) {
                Material::Ggx { alpha, albedo } => assert_eq!((alpha, albedo.r()), (want, 0.5), "alpha expr {v}"),
                _ => panic!(),
            }
        }
        let glass = Material::Dielectric { ior: 1.5, absorption: Color::new(0.1, 0.1, 0.1) };
        for (v, want) in [(1.33, 1.33), (-1.0, 1e-3), (f64::NAN, 1e-3)] {
            let (set, id) = scalar_shader(glass, |s| Exprs { ior: Some(s.add_value(c(v, 0.0, 0.0))), ..Exprs::default() });
            match set.eval_at(id, &w, Vec3::new(0.0, 0.0, 0.0), (0.0, 0.0)) { Material::Dielectric { ior, .. } => assert_eq!(ior, want), _ => panic!() }
        }
        let (set, id) = scalar_shader(glass, |s| Exprs { absorption: Some(s.add_value(c(-1.0, 0.5, f64::NAN))), ..Exprs::default() });
        match set.eval_at(id, &w, Vec3::new(0.0, 0.0, 0.0), (0.0, 0.0)) {
            Material::Dielectric { absorption, ior } => assert_eq!((absorption.r(), absorption.g(), absorption.b(), ior), (0.0, 0.5, 0.0, 1.5)),
            _ => panic!(),
        }
        let light = Material::DiffuseLight { emit: Color::new(1.0, 1.0, 1.0) };
        let (set, id) = scalar_shader(light, |s| Exprs { emit: Some(s.add_value(c(-3.0, 2.0, 1.0))), ..Exprs::default() });
        assert!(matches!(set.eval_at(id, &w, Vec3::new(0.0, 0.0, 0.0), (0.0, 0.0)), Material::DiffuseLight { emit } if (emit.r(), emit.g()) == (0.0, 2.0)));
        // 対応しない材質（ガラスに alpha、ランバートに ior）は式を付けても変わらない
        let (set, id) = scalar_shader(glass, |s| Exprs { alpha: Some(s.add_value(c(0.7, 0.0, 0.0))), ..Exprs::default() });
        assert!(matches!(set.eval_at(id, &w, Vec3::new(0.0, 0.0, 0.0), (0.0, 0.0)), Material::Dielectric { ior, .. } if ior == 1.5));
    }

    /// 不正な式（範囲外の添字・自分自身を指す循環・深すぎる木）で落ちず、白（1 倍）に倒れる。
    #[test]
    fn broken_expressions_do_not_crash() {
        let mut set = ShaderSet::default();
        // 未来の（まだ無い）ノードを指す = 循環しうる子は、足すときに白の定数に置き換わる
        let cyc = set.add_value(ValueNode::Mul(1, 1));
        let cyc2 = set.add_value(ValueNode::Add(0, 5));
        assert!(matches!(set.value(cyc), ValueNode::Const(_)) && matches!(set.value(cyc2), ValueNode::Add(0, 5)) == false);
        assert_eq!(bits(eval_root(&set, cyc)), bits(Color::new(1.0, 1.0, 1.0)));
        let dangling = set.add_value(ValueNode::Mul(999, 998));
        assert_eq!(bits(eval_root(&set, dangling)), bits(Color::new(1.0, 1.0, 1.0)));
        assert_eq!(bits(eval_root(&set, 12345)), bits(Color::new(1.0, 1.0, 1.0)));
        // 深さ 200 の鎖
        let mut id = set.add_value(c(0.5, 0.5, 0.5));
        for _ in 0..200 {
            let one = set.add_value(c(1.0, 1.0, 1.0));
            id = set.add_value(ValueNode::Mul(id, one));
        }
        assert!(eval_root(&set, id).r().is_finite());
    }

    /// 法線の摂動の列: 与えた順に保たれ、材質ごとに独立で、空の材質は空の列。
    #[test]
    fn normal_chain_keeps_the_given_order_per_material() {
        let mut set = ShaderSet::default();
        let m = Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) };
        let a = set.push_shader(m, Exprs::default(), &[4, 2, 9]);
        let b = set.push_shader(m, Exprs::default(), &[]);
        let c2 = set.push_shader(m, Exprs::default(), &[7]);
        assert_eq!(set.normal_chain(a), &[4, 2, 9]);
        assert!(set.normal_chain(b).is_empty());
        assert_eq!(set.normal_chain(c2), &[7]);
        assert_eq!(set.normal_map(a), Some(4));
        assert_eq!(set.normal_map(b), None);
    }
}
