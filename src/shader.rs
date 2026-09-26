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
//! **式（[`ValueNode`]）** は小さな平坦なアリーナ（`Vec<ValueNode>`、子は添字）で持つ。今ある機能に必要な 4 つだけ:
//! `Const`（定数）、`Texture`（UV で引く画像。sRGB / raw はロード時に `Texture` が決めている）、`Noise`（位置で引く
//! 手続きノイズ）、`Mul`（成分ごとの積。「定数の倍率 × テクスチャ」がこれ）。
//!
//! **早道**: 式を持たない材質（`albedo == None`）は、基底の `Material` をそのまま返す（コピー 1 回。積分器から見た
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

/// 式のノード（子は `ValueId`）。
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
}

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
    /// 法線の摂動（`ShaderSet::normal_maps` の添字。**この段階では 1 材質 1 枚**）
    pub normal: Option<MapId>,
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
}

impl ShaderSet {
    /// 定数だけの材質の列から作る（式もマップも無し）。
    pub fn from_materials(mats: &[Material]) -> Self {
        Self { shaders: mats.iter().map(|&m| Shader { base: m, albedo: None, normal: None }).collect(), ..Self::default() }
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
        self.shaders.push(Shader { base, albedo, normal });
        self.shaders.len() - 1
    }

    /// 式のノードを足して添字を返す。
    pub fn add_value(&mut self, node: ValueNode) -> ValueId {
        self.values.push(node);
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
        match sh.albedo {
            None => sh.base,
            Some(id) => sh.base.with_albedo(self.eval_value(id, ctx)),
        }
    }

    /// 式のノード（テスト・診断用）。
    pub fn value(&self, id: ValueId) -> ValueNode {
        self.values[id as usize]
    }

    /// 材質 `mat_id` の法線マップ。
    #[inline(always)]
    pub fn normal_map(&self, mat_id: usize) -> Option<MapId> {
        self.shaders[mat_id].normal
    }

    /// 式を評価する。範囲外の添字（読み込みに失敗したテクスチャなど）は白（1 倍）として扱う
    /// （以前の `resolve_textures` が範囲外の添字を「テクスチャ無し」にしたのと同じ）。
    fn eval_value(&self, id: ValueId, ctx: &ShadeCtx) -> Color {
        let white = Color::new(1.0, 1.0, 1.0);
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
            Some(&ValueNode::Mul(a, b)) => self.eval_value(a, ctx).hadamard(self.eval_value(b, ctx)),
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
}
