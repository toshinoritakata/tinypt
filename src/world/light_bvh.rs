//! 光源 BVH（`LightSelect::Bvh`）: ノード構築とコーン重要度による選択。面光源・球光源の選択確率を
//! 参照点から重み付けする（`World::light_selection_prob` / `World::sample_light` が使う）。

use crate::geometry::Aabb;
use crate::math::Vec3;

use super::lights::{tri_world_verts, Light, LightInfo};
use super::World;

/// 光源のまとまり（発光球・発光インスタンス）の数 `n_groups` に応じた既定の選び方（実測: light_bvh_report.md）:
/// 1 つ以下は出力パワーの CDF（選択確率が変わりようがない）。2〜32 は線形の重み付け（N = 10 で等価時間 0.90）。
/// 33〜79 はどれも得にならない（spiral の 45 で線形 1.06 / BVH 1.07 の損、人工 50 個で ±0）ので CDF のまま。
/// 80 以上は光源 BVH（N = 100 で 0.84、200 で 0.70、500 で 0.63、1000 で 0.65、2000 で 0.82。線形は N = 1000 から損）。
pub(crate) fn default_light_select(n_groups: usize) -> LightSelect {
    const LINEAR_MAX_GROUPS: usize = 32;
    const BVH_MIN_GROUPS: usize = 80;
    if n_groups < 2 {
        LightSelect::Power
    } else if n_groups <= LINEAR_MAX_GROUPS {
        LightSelect::Linear
    } else if n_groups >= BVH_MIN_GROUPS {
        LightSelect::Bvh
    } else {
        LightSelect::Power
    }
}

/// 光源の選び方（NEE がどの発光体を狙うか）。`build_lights` が既定を決め、`set_light_select` で切り替えられる（測定用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightSelect {
    /// 出力パワーだけの CDF（従来。光源のまとまりが 1 つ以下のときの既定）
    Power,
    /// 全光源に参照点からの重み `Φ / max(d², r²)` を付けて線形に選ぶ（O(N)）
    Linear,
    /// 光源 BVH（Conty & Kulla 2018）でルートから確率的に降りて選ぶ（O(log N)）
    Bvh,
}

/// 光源 BVH のノード。
#[derive(Clone, Copy, Debug)]
struct LightNode {
    /// 光源のまとまりの AABB
    bbox: Aabb,
    /// 出力パワーの総和
    power: f64,
    /// 向きの円錐: 軸 `w`（発光の向き）、放射の広がり `θ_o`（`cos_o` = −1 は全方向 = 球）、発光の広がり `θ_e`（`cos_e`）
    w: Vec3,
    cos_o: f64,
    sin_o: f64,
    cos_e: f64,
    left: i32,
    right: i32,
    parent: i32,
    /// 葉のとき `World::lights` の添字、内部ノードは -1
    light: i32,
}

/// 光源 BVH（1 光源 = 1 葉）。木の形は SAH（パワー × 表面積）の 2 分割。
#[derive(Clone, Debug, Default)]
pub(crate) struct LightBvh {
    nodes: Vec<LightNode>,
    /// 光源 → 葉のノード番号（`light_pdf` が葉から根へ経路を引くのに使う）
    leaf_of: Vec<u32>,
}

/// 向きの円錐 `(軸, cos θ_o, cos θ_e)` どうしの和（両方を含む最小の円錐。PBRT v4 の `Union(DirectionCone)` と同じ手続き）。
fn cone_union(a: (Vec3, f64), b: (Vec3, f64)) -> (Vec3, f64) {
    // (軸, cos θ_o) のみ（θ_e は呼び出し側が min(cos_e) で合わせる）
    let (wa, ca) = a;
    let (wb, cb) = b;
    if ca <= -1.0 || cb <= -1.0 {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    let (ta, tb) = (ca.clamp(-1.0, 1.0).acos(), cb.clamp(-1.0, 1.0).acos());
    let td = wa.dot(wb).clamp(-1.0, 1.0).acos();
    if (td + tb).min(std::f64::consts::PI) <= ta {
        return a;
    }
    if (td + ta).min(std::f64::consts::PI) <= tb {
        return b;
    }
    let to = (ta + td + tb) * 0.5;
    if to >= std::f64::consts::PI {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    let tr = to - ta;
    let wr = wa.cross(wb);
    if wr.len() <= 1e-12 {
        return (Vec3::new(0.0, 1.0, 0.0), -1.0);
    }
    // wa を wr まわりに tr だけ回す（Rodrigues。wr ⟂ wa）
    let k = wr.norm();
    let w = wa * tr.cos() + k.cross(wa) * tr.sin() + k * (k.dot(wa) * (1.0 - tr.cos()));
    (w.norm(), to.cos())
}

/// `cos(a − b)` と `sin(a − b)`（`a ≤ b` なら 0 に丸める = `θ' = max(0, a − b)`）。
#[inline(always)]
fn sub_angle_clamped(sa: f64, ca: f64, sb: f64, cb: f64) -> (f64, f64) {
    if ca > cb {
        (0.0, 1.0)
    } else {
        (sa * cb - ca * sb, ca * cb + sa * sb)
    }
}

impl LightBvh {
    /// 光源のリストから作る。`world` は三角形の頂点（境界・向き）を引くのに使う。
    pub(crate) fn build(lights: &[LightInfo], world: &World) -> Self {
        let n = lights.len();
        if n == 0 {
            return Self::default();
        }
        // 葉のデータ: (AABB, パワー, 軸, cos θ_o, cos θ_e)
        let leaf: Vec<(Aabb, f64, Vec3, f64, f64)> = lights
            .iter()
            .map(|info| match info.light {
                Light::Sphere { idx } => {
                    let s = &world.spheres[idx];
                    let r = Vec3::new(s.r, s.r, s.r);
                    (Aabb { min: s.c - r, max: s.c + r }, info.weight, Vec3::new(0.0, 1.0, 0.0), -1.0, 0.0)
                }
                Light::Triangle { mesh_id, tri_id, inst_id } => {
                    let (a, b, c) = tri_world_verts(world, mesh_id, tri_id, inst_id, 0.5).unwrap_or((Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)));
                    (Aabb::empty().grow(a).grow(b).grow(c), info.weight, info.normal, 1.0, 0.0)
                }
            })
            .collect();
        let mut bvh = Self { nodes: Vec::with_capacity(2 * n), leaf_of: vec![0; n] };
        let mut idx: Vec<usize> = (0..n).collect();
        bvh.build_range(&leaf, &mut idx, -1);
        bvh
    }

    fn build_range(&mut self, leaf: &[(Aabb, f64, Vec3, f64, f64)], idx: &mut [usize], parent: i32) -> i32 {
        let id = self.nodes.len() as i32;
        if idx.len() == 1 {
            let (bbox, power, w, cos_o, cos_e) = leaf[idx[0]];
            self.nodes.push(LightNode { bbox, power, w, cos_o, sin_o: (1.0 - cos_o * cos_o).max(0.0).sqrt(), cos_e, left: -1, right: -1, parent, light: idx[0] as i32 });
            self.leaf_of[idx[0]] = id as u32;
            return id;
        }
        // 分割: 重心の広がりが最大の軸で、ビン分割の SAH（パワー × 表面積）。有効な分割が無ければ中央値
        let mut cb = Aabb::empty();
        for &i in idx.iter() {
            cb = cb.grow(leaf[i].0.centroid());
        }
        let e = cb.extent();
        let axis = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
        let key = |i: usize| {
            let c = leaf[i].0.centroid();
            match axis { 0 => c.x, 1 => c.y, _ => c.z }
        };
        idx.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
        let area = |b: Aabb| {
            let e = b.extent();
            2.0 * (e.x * e.y + e.y * e.z + e.z * e.x) + 1e-12
        };
        // 並べ替えた列での全分割位置を評価（N が小さいので全数。累積で O(N)）
        let m = idx.len();
        let mut left_cost = vec![0.0; m];
        let (mut bb, mut pw) = (Aabb::empty(), 0.0);
        for k in 0..m {
            bb = bb.union(leaf[idx[k]].0);
            pw += leaf[idx[k]].1;
            left_cost[k] = pw * area(bb);
        }
        let (mut bb, mut pw) = (Aabb::empty(), 0.0);
        let mut best = (f64::INFINITY, m / 2);
        for k in (1..m).rev() {
            bb = bb.union(leaf[idx[k]].0);
            pw += leaf[idx[k]].1;
            let cost = left_cost[k - 1] + pw * area(bb);
            if cost < best.0 {
                best = (cost, k);
            }
        }
        let mid = best.1.clamp(1, m - 1);
        self.nodes.push(LightNode { bbox: Aabb::empty(), power: 0.0, w: Vec3::new(0.0, 1.0, 0.0), cos_o: 1.0, sin_o: 0.0, cos_e: 0.0, left: -1, right: -1, parent, light: -1 });
        let (li, ri) = idx.split_at_mut(mid);
        let l = self.build_range(leaf, li, id);
        let r = self.build_range(leaf, ri, id);
        let (a, b) = (self.nodes[l as usize], self.nodes[r as usize]);
        let (w, cos_o) = cone_union((a.w, a.cos_o), (b.w, b.cos_o));
        let node = &mut self.nodes[id as usize];
        node.bbox = a.bbox.union(b.bbox);
        node.power = a.power + b.power;
        node.w = w;
        node.cos_o = cos_o;
        node.sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
        node.cos_e = a.cos_e.min(b.cos_e);
        node.left = l;
        node.right = r;
        id
    }
}


impl World {
    /// 光源の選び方を切り替える（既定は `build_lights` が決める。測定・テスト用）。
    pub fn set_light_select(&mut self, mode: LightSelect) {
        self.light_select = mode;
    }

    /// 現在の光源の選び方。
    pub fn light_select(&self) -> LightSelect {
        self.light_select
    }

    /// 光源 `id` を選ぶ確率（現在の選び方 `light_select` に従う。`sample_light` の `pdf_select` と同じ値）。テスト・診断用。
    pub fn light_selection_prob(&self, id: usize, from: Vec3) -> f64 {
        match self.light_select {
            LightSelect::Bvh => self.bvh_prob(id, from),
            _ => {
                let total = self.selection_total(from);
                if total > 0.0 { self.selection_prob(id, from, total) } else { 0.0 }
            }
        }
    }

    /// 光源 BVH の内部ノードの、参照点 `p` から見た重要度: `Φ / max(d², r²)`（`d` = ノードの境界ボックスの中心までの距離、
    /// `r` = その半径。前回の線形の重みと同じ形）を、向きの円錐が「`p` は寄与しえない側にある」と保守的に判定できるときは 0 にする
    /// （軸と `p` への方向の角 θ_w、放射の広がり θ_o、箱の見込み角 θ_b から `θ' = max(0, θ_w − θ_o − θ_b)` を作り、`θ' ≥ θ_e` なら 0）。
    /// 箱の中に `p` があれば向きでは落とさない。**葉（光源 1 個）は前回の `selection_weight` そのもの**（片面発光の平面テストが厳密）。
    #[inline]
    fn node_importance(&self, n: &LightNode, p: Vec3) -> f64 {
        if n.light >= 0 {
            return Self::selection_weight(&self.lights[n.light as usize], p);
        }
        let c = n.bbox.centroid();
        let h = n.bbox.max - c;
        let r2 = h.dot(h);
        let dv = p - c;
        let d2 = dv.dot(dv);
        if n.cos_o > -1.0 && d2 > r2 {
            let d = d2.sqrt();
            let cos_w = (n.w.dot(dv) / d).clamp(-1.0, 1.0);
            let sin_w = (1.0 - cos_w * cos_w).max(0.0).sqrt();
            let sin_b = (r2 / d2).sqrt();
            let cos_b = (1.0 - sin_b * sin_b).max(0.0).sqrt();
            let (s1, c1) = sub_angle_clamped(sin_w, cos_w, n.sin_o, n.cos_o);
            let (_, c2) = sub_angle_clamped(s1, c1, sin_b, cos_b);
            if c2 <= n.cos_e {
                return 0.0;
            }
        }
        n.power / d2.max(r2).max(f64::MIN_POSITIVE)
    }

    /// 内部ノードの 2 つの子を選ぶ確率 `(左, 右)`（子の重要度の比）。両方 0 なら `(0, 0)`。
    /// **`sample_light`（`bvh_select`）と `light_pdf`（`bvh_prob`）の両方がこの関数だけで確率を作る**。
    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(sum > 0.0)` also catches a NaN importance sum; `sum <= 0.0` would not
    fn bvh_child_probs(&self, node: &LightNode, p: Vec3) -> (f64, f64) {
        let il = self.node_importance(&self.light_bvh.nodes[node.left as usize], p);
        let ir = self.node_importance(&self.light_bvh.nodes[node.right as usize], p);
        let sum = il + ir;
        if !(sum > 0.0) {
            return (0.0, 0.0);
        }
        (il / sum, ir / sum)
    }

    /// 光源 BVH で光源を選ぶ。`u` ∈ [0,1) 1 個を各段で使い回す（子を選んだら `u` を選んだ区間に写して一様に戻す: 確率的な分割）。
    /// 返り値は `(光源の添字, 選択確率)`。選択確率は根から葉までの各段の確率の**根側から順の積**。
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` also catches NaN importance/probability; `x <= 0.0` would not
    pub(crate) fn bvh_select(&self, mut u: f64, p: Vec3) -> Option<(usize, f64)> {
        let nodes = &self.light_bvh.nodes;
        if nodes.is_empty() {
            return None;
        }
        let mut cur = 0usize;
        let mut prob = 1.0;
        loop {
            let n = &nodes[cur];
            if n.light >= 0 {
                // 根が葉（光源 1 個）のときは重要度が 0（裏側）でないことだけ確かめる
                if cur == 0 && !(self.node_importance(n, p) > 0.0) {
                    return None;
                }
                return Some((n.light as usize, prob));
            }
            let (pl, pr) = self.bvh_child_probs(n, p);
            if !(pl + pr > 0.0) {
                return None;
            }
            if u < pl {
                prob *= pl;
                u = (u / pl).min(1.0 - f64::EPSILON / 2.0);
                cur = n.left as usize;
            } else {
                prob *= pr;
                u = ((u - pl) / pr).clamp(0.0, 1.0 - f64::EPSILON / 2.0);
                cur = n.right as usize;
            }
        }
    }

    /// 光源 `id` を光源 BVH で選ぶ確率。**`bvh_select` が降りるのと同じ経路**（葉から親をたどって根側から並べる）で、
    /// 同じ `bvh_child_probs` の値を同じ順序で掛ける（ビット一致）。
    pub(crate) fn bvh_prob(&self, id: usize, p: Vec3) -> f64 {
        let nodes = &self.light_bvh.nodes;
        if nodes.is_empty() {
            return 0.0;
        }
        let leaf = self.light_bvh.leaf_of[id] as usize;
        // 葉 → 根の経路（各要素 = (親, 葉側が左か)）
        let mut path = [(0usize, false); 128];
        let mut depth = 0usize;
        let mut cur = leaf;
        while nodes[cur].parent >= 0 {
            let par = nodes[cur].parent as usize;
            path[depth] = (par, nodes[par].left as usize == cur);
            depth += 1;
            cur = par;
        }
        if depth == 0 {
            // 根が葉
            return if self.node_importance(&nodes[leaf], p) > 0.0 { 1.0 } else { 0.0 };
        }
        let mut prob = 1.0;
        for k in (0..depth).rev() {
            let (par, is_left) = path[k];
            let (pl, pr) = self.bvh_child_probs(&nodes[par], p);
            prob *= if is_left { pl } else { pr };
        }
        prob
    }
}
