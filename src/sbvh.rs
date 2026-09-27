//! 空間分割つきの BVH ビルダー（SBVH: Stich, Friedrich, Dietrich, "Spatial Splits in Bounding Volume
//! Hierarchies", 2009）。メッシュの BVH（BLAS）専用で、TLAS・光源 BVH は従来のオブジェクト分割のまま。
//!
//! 細長い斜めの三角形は AABB が実体よりずっと大きく、オブジェクト分割（三角形を左右どちらかへ振る）では
//! 兄弟の箱が大きく重なる。SBVH は、重なりが大きいノードで**三角形を分割面で切って両側の子に入れる**（参照の重複）ことで
//! 箱を小さくする。葉は三角形番号の並びを持ち、**同じ三角形が複数の葉に現れてよい**（交差判定は三角形全体に対して
//! 行い、最近接は `t` の比較なので重複しても結果は変わらない）。
//!
//! ## 各ノードの手順
//! 1. 従来と同じビン化 SAH で最良のオブジェクト分割を求める。
//! 2. その左右の子の箱の重なりの表面積 / 根の表面積 > α（[`SBVH_ALPHA`]）なら、同じ軸で空間分割を評価する:
//!    軸を [`SBVH_BINS`] 個の一様なビンに分け、各参照の**三角形のポリゴンを**ビンの境界面で切って（Sutherland–Hodgman。
//!    AABB でなく実体を切るので斜めの三角形でも箱が締まる）、ビンごとの箱と、入る / 出る参照数を集計する。
//! 3. 空間分割のコストがオブジェクト分割より小さく、重複の予算に収まるときだけ空間分割を採る。
//!
//! ## 重複の予算（決定的）
//! 参照の総数は `三角形数 × SBVH_DUP_CAP` 以下。ノードごとに「その部分木が増やしてよい参照数」を持ち、分割後は残りを
//! 左右の参照数の比で配る。スレッドの実行順に依存しないので、並列構築でも逐次構築とビット単位で同じ木になる。
//!
//! ## 省略したもの
//! 論文の **参照の unsplitting**（切った三角形を片側に戻して重複を減らす後処理）は入れていない
//! （箱を縮められないので近似になる割に、重複予算で総量を抑えているため）。

use crate::bvh::{parallel_depth_budget, BvhNode};
use crate::constants::bvh::{LEAF_SIZE, PARALLEL_MIN_TRIS, SAH_BINS};
use crate::geometry::{Aabb, TriangleSource};
use crate::math::{gamma, Vec3};

/// 空間分割の調整値（既定は [`crate::constants::bvh`]）。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    /// 重なり / 根の表面積 がこれを超えたら空間分割を試す
    pub alpha: f64,
    /// 空間分割のビン数
    pub bins: usize,
    /// 参照の総数の上限（三角形数の何倍まで）。1.0 で空間分割なし
    pub dup_cap: f64,
}

impl Default for Params {
    fn default() -> Self {
        use crate::constants::bvh::{SBVH_ALPHA, SBVH_BINS, SBVH_DUP_CAP};
        Self { alpha: SBVH_ALPHA, bins: SBVH_BINS, dup_cap: SBVH_DUP_CAP }
    }
}

/// 三角形への参照 1 つ。`bbox` は「その三角形のうち、このノードの領域に入る部分」の箱（切られていれば元の箱より小さい）。
#[derive(Clone, Copy)]
struct Ref {
    tri: u32,
    bbox: Aabb,
}

struct Ctx<'a> {
    src: TriangleSource<'a>,
    p: Params,
    /// 根の箱の表面積（重なりの相対評価に使う）
    root_area: f64,
}

/// 部分木（ノードはローカルの 0 起点、葉の `start` はこの部分木の `leaves` への添字）。
struct Sub {
    nodes: Vec<BvhNode>,
    leaves: Vec<u32>,
}

#[inline]
fn axis_val(v: Vec3, a: usize) -> f64 {
    match a {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// 箱の表面積（空・反転した箱は 0）。
fn area(b: &Aabb) -> f64 {
    let e = b.extent();
    if e.x < 0.0 || e.y < 0.0 || e.z < 0.0 {
        return 0.0;
    }
    2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
}

/// 2 つの箱の共通部分の表面積（重ならなければ 0）。
fn overlap_area(a: &Aabb, b: &Aabb) -> f64 {
    let min = Vec3::new(a.min.x.max(b.min.x), a.min.y.max(b.min.y), a.min.z.max(b.min.z));
    let max = Vec3::new(a.max.x.min(b.max.x), a.max.y.min(b.max.y), a.max.z.min(b.max.z));
    area(&Aabb { min, max })
}

/// 参照の箱の和と、重心の箱。
fn bounds_of(refs: &[Ref]) -> (Aabb, Aabb) {
    let (mut bbox, mut cbox) = (Aabb::empty(), Aabb::empty());
    for r in refs {
        bbox = bbox.union(r.bbox);
        cbox = cbox.grow(r.bbox.centroid());
    }
    (bbox, cbox)
}

// ------------------------------------------------------------------ polygon clipping

/// 凸ポリゴン（最大 8 頂点。三角形を面で 2 回切っても 5 頂点を超えない）。
#[derive(Clone, Copy)]
struct Poly {
    v: [Vec3; 8],
    n: usize,
}

impl Poly {
    fn triangle(t: (Vec3, Vec3, Vec3)) -> Self {
        let z = Vec3::new(0.0, 0.0, 0.0);
        Poly { v: [t.0, t.1, t.2, z, z, z, z, z], n: 3 }
    }

    fn push(&mut self, p: Vec3) {
        if self.n < self.v.len() {
            self.v[self.n] = p;
            self.n += 1;
        }
    }

    /// 軸 `axis` の値 `plane` でポリゴンを `(下側, 上側)` に切る（Sutherland–Hodgman）。面上の頂点は両側に入る。
    fn split(&self, axis: usize, plane: f64) -> (Poly, Poly) {
        let z = Vec3::new(0.0, 0.0, 0.0);
        let mut lo = Poly { v: [z; 8], n: 0 };
        let mut hi = Poly { v: [z; 8], n: 0 };
        for i in 0..self.n {
            let (a, b) = (self.v[i], self.v[(i + 1) % self.n]);
            let (sa, sb) = (axis_val(a, axis) - plane, axis_val(b, axis) - plane);
            if sa <= 0.0 {
                lo.push(a);
            }
            if sa >= 0.0 {
                hi.push(a);
            }
            if (sa < 0.0 && sb > 0.0) || (sa > 0.0 && sb < 0.0) {
                let t = sa / (sa - sb);
                let p = a + (b - a) * t;
                lo.push(p);
                hi.push(p);
            }
        }
        (lo, hi)
    }

    /// ポリゴンの箱（丸めの分だけ座標に比例して広げる）。頂点が無ければ `None`。
    fn bbox(&self) -> Option<Aabb> {
        if self.n == 0 {
            return None;
        }
        let mut b = Aabb::empty();
        for i in 0..self.n {
            b = b.grow(self.v[i]);
        }
        let pad = |lo: f64, hi: f64| gamma(8) * lo.abs().max(hi.abs());
        let (px, py, pz) = (pad(b.min.x, b.max.x), pad(b.min.y, b.max.y), pad(b.min.z, b.max.z));
        b.min = Vec3::new(b.min.x - px, b.min.y - py, b.min.z - pz);
        b.max = Vec3::new(b.max.x + px, b.max.y + py, b.max.z + pz);
        Some(b)
    }
}

/// `piece`（ポリゴンの一部）の箱を、参照の箱 `within` に切り詰める。空（反転）なら `None`。
fn clipped_box(piece: &Poly, within: &Aabb) -> Option<Aabb> {
    let b = piece.bbox()?;
    let min = Vec3::new(b.min.x.max(within.min.x), b.min.y.max(within.min.y), b.min.z.max(within.min.z));
    let max = Vec3::new(b.max.x.min(within.max.x), b.max.y.min(within.max.y), b.max.z.min(within.max.z));
    if min.x > max.x || min.y > max.y || min.z > max.z {
        return None;
    }
    Some(Aabb { min, max })
}

// ------------------------------------------------------------------ splits

/// 最良のオブジェクト分割（従来の binned SAH と同じ: 重心の広がりが最大の軸、[`SAH_BINS`] ビン）。
struct ObjectSplit {
    cost: f64,
    axis: usize,
    minc: f64,
    inv_extent: f64,
    /// 左に入るビンの数（ビン `< split_bin` が左）
    split_bin: usize,
    left_box: Aabb,
    right_box: Aabb,
}

fn object_split(refs: &[Ref], cbox: Aabb) -> Option<ObjectSplit> {
    let n = refs.len();
    if n < SAH_BINS * 2 {
        return None;
    }
    let ext = cbox.extent();
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    let minc = axis_val(cbox.min, axis);
    let extent = axis_val(cbox.max, axis) - minc;
    if extent <= 1e-12 {
        return None;
    }
    let inv_extent = 1.0 / extent;
    let bin_of = |r: &Ref| (((axis_val(r.bbox.centroid(), axis) - minc) * inv_extent * (SAH_BINS as f64)) as usize).min(SAH_BINS - 1);
    let mut bins = [(Aabb::empty(), 0usize); SAH_BINS];
    for r in refs {
        let b = bin_of(r);
        bins[b].0 = bins[b].0.union(r.bbox);
        bins[b].1 += 1;
    }
    let mut left_box = [Aabb::empty(); SAH_BINS];
    let mut right_box = [Aabb::empty(); SAH_BINS];
    let mut left_count = [0usize; SAH_BINS];
    let mut right_count = [0usize; SAH_BINS];
    let (mut acc, mut cnt) = (Aabb::empty(), 0usize);
    for i in 0..SAH_BINS {
        if bins[i].1 > 0 {
            acc = acc.union(bins[i].0);
            cnt += bins[i].1;
        }
        left_box[i] = acc;
        left_count[i] = cnt;
    }
    let (mut acc, mut cnt) = (Aabb::empty(), 0usize);
    for i in (0..SAH_BINS).rev() {
        if bins[i].1 > 0 {
            acc = acc.union(bins[i].0);
            cnt += bins[i].1;
        }
        right_box[i] = acc;
        right_count[i] = cnt;
    }
    let mut best: Option<ObjectSplit> = None;
    for i in 0..(SAH_BINS - 1) {
        let (lc, rc) = (left_count[i], right_count[i + 1]);
        if lc == 0 || rc == 0 {
            continue;
        }
        let cost = area(&left_box[i]) * lc as f64 + area(&right_box[i + 1]) * rc as f64;
        if best.as_ref().is_none_or(|b| cost < b.cost) {
            best = Some(ObjectSplit { cost, axis, minc, inv_extent, split_bin: i + 1, left_box: left_box[i], right_box: right_box[i + 1] });
        }
    }
    best
}

/// 軸方向の空間分割（ビン `i` までが左、`i + 1` からが右）。
struct SpatialSplit {
    cost: f64,
    /// 分割面はビン `i` と `i + 1` の境目
    bin: usize,
    bmin: f64,
    width: f64,
}

fn spatial_bin(x: f64, bmin: f64, inv_w: f64, nb: usize) -> usize {
    let b = ((x - bmin) * inv_w).floor();
    if b <= 0.0 {
        0
    } else {
        (b as usize).min(nb - 1)
    }
}

/// 空間分割の最良位置（`budget` の範囲に収まる候補だけ）。`axis` は呼び出し側が決める。
fn spatial_split(ctx: &Ctx, refs: &[Ref], bbox: &Aabb, axis: usize, budget: usize) -> Option<SpatialSplit> {
    let nb = ctx.p.bins.max(2);
    let bmin = axis_val(bbox.min, axis);
    let ext = axis_val(bbox.max, axis) - bmin;
    if ext.is_nan() || ext <= 1e-12 {
        return None;
    }
    let width = ext / nb as f64;
    let inv_w = 1.0 / width;
    let mut bin_box = vec![Aabb::empty(); nb];
    let mut entries = vec![0usize; nb];
    let mut exits = vec![0usize; nb];
    for r in refs {
        let b0 = spatial_bin(axis_val(r.bbox.min, axis), bmin, inv_w, nb);
        let b1 = spatial_bin(axis_val(r.bbox.max, axis), bmin, inv_w, nb);
        entries[b0] += 1;
        exits[b1] += 1;
        if b0 == b1 {
            bin_box[b0] = bin_box[b0].union(r.bbox);
            continue;
        }
        // 三角形を境界面で順に切っていく（残りを次のビンへ持ち越す）
        let mut poly = Poly::triangle(ctx.src.open_vertices(r.tri as usize));
        for (b, bb) in bin_box.iter_mut().enumerate().take(b1).skip(b0) {
            let (lo, hi) = poly.split(axis, bmin + (b + 1) as f64 * width);
            if let Some(cb) = clipped_box(&lo, &r.bbox) {
                *bb = bb.union(cb);
            }
            poly = hi;
        }
        if let Some(cb) = clipped_box(&poly, &r.bbox) {
            bin_box[b1] = bin_box[b1].union(cb);
        }
    }
    // 右からの累積
    let mut right_box = vec![Aabb::empty(); nb];
    let mut right_count = vec![0usize; nb];
    let (mut acc, mut cnt) = (Aabb::empty(), 0usize);
    for i in (0..nb).rev() {
        acc = acc.union(bin_box[i]);
        cnt += exits[i];
        right_box[i] = acc;
        right_count[i] = cnt;
    }
    let n = refs.len();
    let (mut lbox, mut lcount) = (Aabb::empty(), 0usize);
    let mut best: Option<SpatialSplit> = None;
    for i in 0..(nb - 1) {
        lbox = lbox.union(bin_box[i]);
        lcount += entries[i];
        let rc = right_count[i + 1];
        if lcount == 0 || rc == 0 || lcount >= n || rc >= n {
            continue;
        }
        if lcount + rc > n + budget {
            continue;
        }
        let cost = area(&lbox) * lcount as f64 + area(&right_box[i + 1]) * rc as f64;
        if best.as_ref().is_none_or(|b| cost < b.cost) {
            best = Some(SpatialSplit { cost, bin: i, bmin, width });
        }
    }
    best
}

/// 空間分割で実際に参照を左右へ振り分ける（またぐ三角形は面で切って両側へ入れる）。進まない結果（片側が空、
/// または片側が全参照を含む）は `None`。
fn partition_spatial(ctx: &Ctx, refs: &[Ref], axis: usize, s: &SpatialSplit) -> Option<(Vec<Ref>, Vec<Ref>)> {
    let nb = ctx.p.bins.max(2);
    let inv_w = 1.0 / s.width;
    let plane = s.bmin + (s.bin + 1) as f64 * s.width;
    let (mut left, mut right) = (Vec::with_capacity(refs.len() / 2 + 8), Vec::with_capacity(refs.len() / 2 + 8));
    for r in refs {
        let b0 = spatial_bin(axis_val(r.bbox.min, axis), s.bmin, inv_w, nb);
        let b1 = spatial_bin(axis_val(r.bbox.max, axis), s.bmin, inv_w, nb);
        if b1 <= s.bin {
            left.push(*r);
        } else if b0 > s.bin {
            right.push(*r);
        } else {
            let (lo, hi) = Poly::triangle(ctx.src.open_vertices(r.tri as usize)).split(axis, plane);
            if let Some(bb) = clipped_box(&lo, &r.bbox) {
                left.push(Ref { tri: r.tri, bbox: bb });
            }
            if let Some(bb) = clipped_box(&hi, &r.bbox) {
                right.push(Ref { tri: r.tri, bbox: bb });
            }
        }
    }
    let n = refs.len();
    if left.is_empty() || right.is_empty() || left.len() >= n || right.len() >= n {
        return None;
    }
    Some((left, right))
}

enum NodeSplit {
    Leaf(Vec<Ref>),
    Two { left: Vec<Ref>, right: Vec<Ref>, budget_left: usize, budget_right: usize },
}

/// 1 ノードの処理: 葉にするか、左右に分ける。`budget` はこの部分木が増やしてよい参照数。
fn split_node(ctx: &Ctx, mut refs: Vec<Ref>, bbox: &Aabb, cbox: Aabb, budget: usize) -> NodeSplit {
    let n = refs.len();
    if n <= LEAF_SIZE {
        return NodeSplit::Leaf(refs);
    }
    let two = |left: Vec<Ref>, right: Vec<Ref>, budget: usize| {
        let extra = (left.len() + right.len()).saturating_sub(n);
        let rest = budget.saturating_sub(extra);
        let total = (left.len() + right.len()).max(1);
        let budget_left = (rest as u128 * left.len() as u128 / total as u128) as usize;
        NodeSplit::Two { budget_left, budget_right: rest - budget_left, left, right }
    };

    let obj = object_split(&refs, cbox);
    if let Some(o) = &obj {
        // 兄弟の箱の重なりが大きいときだけ空間分割を試す
        // 空間分割はオブジェクト分割と同じ軸だけで評価する（全 3 軸にしても sponza で誤差程度の改善しか
        // 無く、ヘアーでは構築が 2〜3 倍になった: sbvh_report 参照）
        let spatial = if budget > 0 && overlap_area(&o.left_box, &o.right_box) / ctx.root_area > ctx.p.alpha {
            spatial_split(ctx, &refs, bbox, o.axis, budget).filter(|s| s.cost < o.cost)
        } else {
            None
        };
        if let Some((l, r)) = spatial.and_then(|s| partition_spatial(ctx, &refs, o.axis, &s)) {
            return two(l, r, budget);
        }
        // オブジェクト分割: ビン < split_bin が左
        let bin_of = |r: &Ref| (((axis_val(r.bbox.centroid(), o.axis) - o.minc) * o.inv_extent * (SAH_BINS as f64)) as usize).min(SAH_BINS - 1);
        let (mut left, mut right) = (Vec::with_capacity(n / 2 + 8), Vec::with_capacity(n / 2 + 8));
        for r in refs.drain(..) {
            if bin_of(&r) < o.split_bin { left.push(r) } else { right.push(r) }
        }
        return two(left, right, budget);
    }
    // SAH が使えない（候補が少ない・重心が一点）: 最大広がり軸の重心で中央値分割
    let ext = cbox.extent();
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    let mid = n / 2;
    refs.select_nth_unstable_by(mid, |a, b| {
        axis_val(a.bbox.centroid(), axis).partial_cmp(&axis_val(b.bbox.centroid(), axis)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let right = refs.split_off(mid);
    two(refs, right, budget)
}

// ------------------------------------------------------------------ recursion

fn build_seq(ctx: &Ctx, refs: Vec<Ref>, budget: usize, nodes: &mut Vec<BvhNode>, leaves: &mut Vec<u32>) -> i32 {
    let (bbox, cbox) = bounds_of(&refs);
    let idx = nodes.len();
    nodes.push(BvhNode { bbox, left: -1, right: -1, start: 0, count: 0 });
    match split_node(ctx, refs, &bbox, cbox, budget) {
        NodeSplit::Leaf(refs) => {
            nodes[idx].start = leaves.len() as u32;
            nodes[idx].count = refs.len() as u32;
            leaves.extend(refs.iter().map(|r| r.tri));
        }
        NodeSplit::Two { left, right, budget_left, budget_right } => {
            let l = build_seq(ctx, left, budget_left, nodes, leaves);
            let r = build_seq(ctx, right, budget_right, nodes, leaves);
            nodes[idx].left = l;
            nodes[idx].right = r;
        }
    }
    idx as i32
}

fn build_par(ctx: &Ctx, refs: Vec<Ref>, budget: usize, depth_budget: usize) -> Sub {
    if depth_budget == 0 || refs.len() < PARALLEL_MIN_TRIS {
        let (mut nodes, mut leaves) = (Vec::new(), Vec::new());
        build_seq(ctx, refs, budget, &mut nodes, &mut leaves);
        return Sub { nodes, leaves };
    }
    let (bbox, cbox) = bounds_of(&refs);
    match split_node(ctx, refs, &bbox, cbox, budget) {
        NodeSplit::Leaf(refs) => Sub {
            nodes: vec![BvhNode { bbox, left: -1, right: -1, start: 0, count: refs.len() as u32 }],
            leaves: refs.iter().map(|r| r.tri).collect(),
        },
        NodeSplit::Two { left, right, budget_left, budget_right } => {
            let (l, r) = crossbeam::scope(|s| {
                let h = s.spawn(|_| build_par(ctx, left, budget_left, depth_budget - 1));
                let r = build_par(ctx, right, budget_right, depth_budget - 1);
                (h.join().expect("SBVH left-subtree build thread panicked"), r)
            })
            .expect("crossbeam::scope failed");
            // 連結: 自分 + 左（子の添字 +1）+ 右（子の添字 + 1 + 左の数、葉の start + 左の葉の数）
            let mut nodes = Vec::with_capacity(1 + l.nodes.len() + r.nodes.len());
            nodes.push(BvhNode { bbox, left: 1, right: (1 + l.nodes.len()) as i32, start: 0, count: 0 });
            nodes.extend(l.nodes.into_iter().map(|mut nd| {
                if nd.left != -1 {
                    nd.left += 1;
                    nd.right += 1;
                }
                nd
            }));
            let off = nodes.len() as i32;
            let leaf_off = l.leaves.len() as u32;
            nodes.extend(r.nodes.into_iter().map(|mut nd| {
                if nd.left != -1 {
                    nd.left += off;
                    nd.right += off;
                } else {
                    nd.start += leaf_off;
                }
                nd
            }));
            let mut leaves = l.leaves;
            leaves.extend(r.leaves);
            Sub { nodes, leaves }
        }
    }
}

/// 静止メッシュの SBVH を作る。戻り値は 2 分木のノードと、葉が参照する三角形番号の並び（重複あり）。
/// 三角形が 0 個なら空。`threads` は並列構築のスレッド数（結果は `threads` に依らずビット単位で同じ）。
pub(crate) fn build(src: TriangleSource<'_>, p: Params, threads: usize) -> (Vec<BvhNode>, Vec<usize>) {
    let n = src.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    debug_assert!(src.is_static(), "SBVH is for static meshes only");
    let refs: Vec<Ref> = (0..n).map(|ti| Ref { tri: ti as u32, bbox: src.bounds(ti) }).collect();
    let (root, _) = bounds_of(&refs);
    let ctx = Ctx { src, p, root_area: area(&root).max(f64::MIN_POSITIVE) };
    let budget = ((p.dup_cap - 1.0).max(0.0) * n as f64) as usize;
    let sub = build_par(&ctx, refs, budget, parallel_depth_budget(threads, n));
    (sub.nodes, sub.leaves.into_iter().map(|t| t as usize).collect())
}
