//! BVH（Bounding Volume Hierarchy）によるレイ-三角形交差の高速化。
//!
//! ## SAH（Surface Area Heuristic）
//! BVH の分割位置を決定するヒューリスティック。
//! レイがノードを横切るコストを以下で近似し、最小コストの分割を選択:
//!   コスト = 左子の表面積 × 左の三角形数 + 右子の表面積 × 右の三角形数
//!
//! ## 構造
//! - 線形配列に格納されたノードツリー（ポインタ不要）
//! - リーフノードは最大 `LEAF_SIZE` 個の三角形を保持
//! - トラバーサルはスタックベース（固定長 64 + ヒープフォールバック）

use crate::constants::bvh::{LEAF_SIZE, PARALLEL_MIN_TRIS, SAH_BINS, WIDE_WIDTH};
use crate::geometry::SLAB_FAR_SCALE;

/// 最近接候補が見つかった後の区間の上端の広げ幅（相対）。同値 `t` の候補が、`t_scaled` と `tmax·det` の丸めで
/// 採用判定に落ちないようにする（勝敗は `(t, 番号)` の比較だけで決まる）。丸め（数 ulp）より十分に大きく、
/// 別の三角形の `t` とは区別できる大きさ。
const TIE_SLACK: f64 = 1.0 + 1e-12;
use crate::geometry::{Aabb, Hit, TriangleSource};
use crate::math::Vec3;
use crate::ray::Ray;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
/// 線形 BVH ツリーのノード。
///
/// - 内部ノード: `left`/`right` が子ノードのインデックス、`count == 0`
/// - リーフノード: `left == -1`, `right == -1`, `start`/`count` で三角形範囲を指定
pub struct BvhNode {
    /// ノードの AABB（バウンディングボックス）
    pub bbox: Aabb,
    /// 左子ノードのインデックス（-1 でリーフ）
    pub left: i32,
    /// 右子ノードのインデックス（-1 でリーフ）
    pub right: i32,
    /// リーフ: indices 配列内の開始位置
    pub start: u32,
    /// リーフ: 三角形の数
    pub count: u32,
}

/// 走査スタックの要素: 親ノード `w` の子スロット `slot` と、積んだときの入口距離 `t0`。
#[derive(Clone, Copy)]
struct Entry {
    t0: f64,
    w: u32,
    slot: u32,
}

impl WideNode {
    /// 子スロット `i` の箱に対するスラブ判定。当たれば入口距離 `t0`。式は [`Aabb::hit_range_inv`] と同じ
    /// （丸めまで同じにして、従来の 2 分木の走査と同じ棄却をする）。
    #[inline(always)]
    fn slot_entry(&self, i: usize, r: &Ray, inv: Vec3, mut tmin: f64, mut tmax: f64) -> Option<f64> {
        let t0x = (self.min_x[i] - r.o.x) * inv.x;
        let t1x = (self.max_x[i] - r.o.x) * inv.x;
        tmin = tmin.max(t0x.min(t1x));
        tmax = tmax.min(t0x.max(t1x) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }
        let t0y = (self.min_y[i] - r.o.y) * inv.y;
        let t1y = (self.max_y[i] - r.o.y) * inv.y;
        tmin = tmin.max(t0y.min(t1y));
        tmax = tmax.min(t0y.max(t1y) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }
        let t0z = (self.min_z[i] - r.o.z) * inv.z;
        let t1z = (self.max_z[i] - r.o.z) * inv.z;
        tmin = tmin.max(t0z.min(t1z));
        tmax = tmax.min(t0z.max(t1z) * SLAB_FAR_SCALE);
        if tmax < tmin { return None; }
        Some(tmin)
    }
}

/// 三角形群に対する BVH（Bounding Volume Hierarchy）。
pub struct Bvh {
    /// リーフが参照する三角形インデックスの並び順
    pub indices: Vec<usize>,
    /// 走査用の広い BVH（SAH で作った 2 分木を畳んだもの。2 分木そのものは畳んだ後に捨てる）
    wide: Vec<WideNode>,
    /// 全体の AABB（2 分木のルートの箱。`World` がインスタンスの境界を作るのに使う）
    bounds: Option<Aabb>,
}

/// 広い BVH のノード（[`WIDE_WIDTH`] 個の子スロット）。**AABB は成分ごとにまとめた SoA**: 子の箱の判定は
/// 「子 i の min.x, max.x, min.y, …」を i について並べて読むので、1 ノードぶんの箱が連続領域に収まり
/// （W=4 で 6×32 B。ノード全体は 224 B）、SIMD に載せるときもそのままレーンに並ぶ。AoS（子ごとに 6 個の f64）
/// だと同じ成分が飛び飛びになる。
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
struct WideNode {
    /// 使っているスロット数（`1..=WIDE_WIDTH`）。残りのスロットは見ない（空の箱を無限大で表すと 0 × ∞ が NaN になるので、個数で管理する）
    n: u8,
    min_x: [f64; WIDE_WIDTH],
    min_y: [f64; WIDE_WIDTH],
    min_z: [f64; WIDE_WIDTH],
    max_x: [f64; WIDE_WIDTH],
    max_y: [f64; WIDE_WIDTH],
    max_z: [f64; WIDE_WIDTH],
    /// `count[i] > 0`: リーフで、`child[i]` は `indices` 内の開始位置。`count[i] == 0`: 内部で、`child[i]` は子ノードの添字
    child: [u32; WIDE_WIDTH],
    count: [u8; WIDE_WIDTH],
}

/// 三角形範囲の AABB（`bbox`）と重心の AABB（`cbox`）。分割軸の選択・SAH のビン化の両方で使う。
fn bounds_for_range(bounds: &[Aabb], centroids: &[Vec3], indices: &[usize]) -> (Aabb, Aabb) {
    let mut bbox = Aabb::empty();
    let mut cbox = Aabb::empty();
    for &i in indices {
        let b = bounds[i];
        bbox = bbox.union(b);
        cbox = cbox.grow(centroids[i]);
    }
    (bbox, cbox)
}

/// この範囲の分割位置を SAH（ビン分割）で決め、ダメなら最大広がり軸の重心の中央値分割にフォールバックし、
/// `indices` をその位置で並べ替える。戻り値は `indices` に対する**ローカルな**分割位置
/// （`indices` の 0 起点、`0 < mid < indices.len()` を満たす）。`indices.len() > LEAF_SIZE`
/// （呼び出し側が保証）が前提。逐次版（[`build_node_sequential`]）・並列版（[`build_range`]）の
/// 両方から呼ぶ共通ロジック（**ここを 2 箇所に重複させないことが、逐次版とのビット一致を保つ鍵**）。
fn choose_split(indices: &mut [usize], bounds: &[Aabb], centroids: &[Vec3], cbox: Aabb) -> usize {
    let n = indices.len();

    // 分割軸: 重心の広がりが最大の軸
    let ext = cbox.extent();
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };

    // SAH（ビン分割）で分割位置を探す。候補が少ない・重心が一点に潰れている・
    // 有効な分割が無い場合は None を返し、下の中央値分割に任せる。
    let sah_mid = if n >= SAH_BINS * 2 {
        let minc = match axis { 0 => cbox.min.x, 1 => cbox.min.y, _ => cbox.min.z };
        let maxc = match axis { 0 => cbox.max.x, 1 => cbox.max.y, _ => cbox.max.z };
        let extent = maxc - minc;

        if extent > 1e-12 {
            let inv_extent = 1.0 / extent;
            let bin_of = |idx: usize| -> usize {
                let c = centroids[idx];
                let cv = match axis { 0 => c.x, 1 => c.y, _ => c.z };
                (((cv - minc) * inv_extent * (SAH_BINS as f64)) as usize).min(SAH_BINS - 1)
            };
            let mut bins = [(Aabb::empty(), 0usize); SAH_BINS];

            for &idx in indices.iter() {
                let bi = bin_of(idx);
                bins[bi].0 = bins[bi].0.union(bounds[idx]);
                bins[bi].1 += 1;
            }

            let mut left_bbox = [Aabb::empty(); SAH_BINS];
            let mut right_bbox = [Aabb::empty(); SAH_BINS];
            let mut left_count = [0usize; SAH_BINS];
            let mut right_count = [0usize; SAH_BINS];

            let mut acc_bbox = Aabb::empty();
            let mut acc_count = 0usize;
            for i in 0..SAH_BINS {
                if bins[i].1 > 0 {
                    acc_bbox = acc_bbox.union(bins[i].0);
                    acc_count += bins[i].1;
                }
                left_bbox[i] = acc_bbox;
                left_count[i] = acc_count;
            }

            acc_bbox = Aabb::empty();
            acc_count = 0usize;
            for i in (0..SAH_BINS).rev() {
                if bins[i].1 > 0 {
                    acc_bbox = acc_bbox.union(bins[i].0);
                    acc_count += bins[i].1;
                }
                right_bbox[i] = acc_bbox;
                right_count[i] = acc_count;
            }

            let mut best_cost = f64::INFINITY;
            let mut best_split = 0usize;
            for i in 0..(SAH_BINS - 1) {
                let lc = left_count[i];
                let rc = right_count[i + 1];
                if lc == 0 || rc == 0 { continue; }
                let la = left_bbox[i].extent();
                let ra = right_bbox[i + 1].extent();
                let left_area = 2.0 * (la.x * la.y + la.y * la.z + la.z * la.x);
                let right_area = 2.0 * (ra.x * ra.y + ra.y * ra.z + ra.z * ra.x);
                let cost = left_area * (lc as f64) + right_area * (rc as f64);
                if cost < best_cost {
                    best_cost = cost;
                    best_split = i + 1;
                }
            }

            if best_cost.is_finite() {
                // lc > 0 && rc > 0 の分割だけが候補なので 0 < left_total < n が保証される
                let left_total = left_count[best_split - 1];
                indices.sort_by_key(|&idx| bin_of(idx));
                Some(left_total)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    match sah_mid {
        Some(mid) => mid,
        None => {
            // 中央値分割: 最大広がり軸の重心で select_nth し、左右を空間的に分ける
            let mid = n / 2;
            indices.select_nth_unstable_by(mid, |&a, &b| {
                let ca = centroids[a];
                let cb = centroids[b];
                let va = match axis { 0 => ca.x, 1 => ca.y, _ => ca.z };
                let vb = match axis { 0 => cb.x, 1 => cb.y, _ => cb.z };
                va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
            });
            mid
        }
    }
}

/// 元の（並列化前の）逐次構築。`nodes` に直接 push していくので、部分木を丸ごと 1 つの `Vec` に
/// 書き込むときのコピーが要らない（並列版の「これ以上分割しない」部分木で使う。§設計参照）。
/// `base_start` は `indices` の 0 起点のローカル位置に足す絶対オフセット
/// （並列版が `split_at_mut` で切り出した部分列を渡すときに、リーフの `start` を全体配列での
/// 絶対位置に直すために使う。逐次単体では 0）。
fn build_node_sequential(
    nodes: &mut Vec<BvhNode>,
    base_start: usize,
    indices: &mut [usize],
    bounds: &[Aabb],
    centroids: &[Vec3],
    start: usize,
    end: usize,
) -> i32 {
    let n = end - start;
    debug_assert!(n > 0);

    let (bbox, cbox) = bounds_for_range(bounds, centroids, &indices[start..end]);

    // Create node now; fill fields after recursion.
    let node_index = nodes.len() as i32;
    nodes.push(BvhNode {
        bbox,
        left: -1,
        right: -1,
        start: 0,
        count: 0,
    });

    // Leaf
    if n <= LEAF_SIZE {
        let idx = node_index as usize;
        nodes[idx].start = (start + base_start) as u32;
        nodes[idx].count = n as u32;
        return node_index;
    }

    let mid_local = choose_split(&mut indices[start..end], bounds, centroids, cbox);
    let mid = start + mid_local;
    debug_assert!(start < mid && mid < end);

    let left = build_node_sequential(nodes, base_start, indices, bounds, centroids, start, mid);
    let right = build_node_sequential(nodes, base_start, indices, bounds, centroids, mid, end);

    let idx = node_index as usize;
    nodes[idx].left = left;
    nodes[idx].right = right;
    nodes[idx].start = 0;
    nodes[idx].count = 0;
    node_index
}

/// 並列版の構築本体。`indices`（`base_start` を起点とする絶対配列の一部を指す、実体を共有する
/// `&mut` スライス）を対象に、**自己完結した部分木**（ローカル 0 起点の `Vec<BvhNode>`、
/// ルートは常にインデックス 0）を返す。呼び出し側（親の呼び出し、または `build_with_threads`）が
/// この部分木をそのまま採用するか、兄弟の部分木と 1 回だけ連結する（インデックスを一括してずらす）。
///
/// `depth_budget` が尽きる（0）か、範囲が小さすぎる（`PARALLEL_MIN_TRIS` 未満、スレッド起動コストに
/// 見合わない）と、そこから先は丸ごと [`build_node_sequential`] に切り替える（1 回の呼び出しで
/// 部分木全体を 1 つの `Vec` に直接 push するので、その部分木の内部では連結によるコピーが発生しない）。
/// これにより、連結のコピーコストは「並列化した上位の深さぶんの合計サイズ」に留まる
/// （木全体のサイズには比例しない）。
fn build_range(
    indices: &mut [usize],
    base_start: usize,
    bounds: &[Aabb],
    centroids: &[Vec3],
    depth_budget: usize,
) -> Vec<BvhNode> {
    let n = indices.len();
    debug_assert!(n > 0);

    let (bbox, cbox) = bounds_for_range(bounds, centroids, indices);

    if n <= LEAF_SIZE {
        return vec![BvhNode { bbox, left: -1, right: -1, start: base_start as u32, count: n as u32 }];
    }

    if depth_budget == 0 || n < PARALLEL_MIN_TRIS {
        let mut local: Vec<BvhNode> = Vec::new();
        build_node_sequential(&mut local, base_start, indices, bounds, centroids, 0, n);
        return local;
    }

    let mid = choose_split(indices, bounds, centroids, cbox);
    debug_assert!(0 < mid && mid < n);
    let (left_idx, right_idx) = indices.split_at_mut(mid);

    // 左を別スレッドで、右を現在のスレッドで並行に構築する（`crossbeam::scope` は render.rs と同じ
    // 依存。左右は `split_at_mut` で得た非重複の可変スライスなので、データ競合なく本当に並行に書ける）。
    let (left, right) = crossbeam::scope(|s| {
        let handle = s.spawn(|_| build_range(left_idx, base_start, bounds, centroids, depth_budget - 1));
        let right = build_range(right_idx, base_start + mid, bounds, centroids, depth_budget - 1);
        (handle.join().expect("BVH left-subtree build thread panicked"), right)
    })
    .expect("crossbeam::scope failed");

    // 連結: 自分がローカル 0、続けて左部分木（各ノードの left/right を +1）、
    // 続けて右部分木（+= 1 + left.len()）。これは逐次版の「自分を push → 左を再帰 → 右を再帰」と
    // 同じ前順序（pre-order）になる（連結を 1 回にまとめているだけで、番号の割り当ては同一）。
    let mut out = Vec::with_capacity(1 + left.len() + right.len());
    out.push(BvhNode { bbox, left: 1, right: (1 + left.len()) as i32, start: 0, count: 0 });
    out.extend(left.into_iter().map(|mut nd| {
        if nd.left != -1 { nd.left += 1; }
        if nd.right != -1 { nd.right += 1; }
        nd
    }));
    let right_offset = out.len() as i32; // = 1 + left.len()
    out.extend(right.into_iter().map(|mut nd| {
        if nd.left != -1 { nd.left += right_offset; }
        if nd.right != -1 { nd.right += right_offset; }
        nd
    }));
    out
}

/// 並列化する再帰の深さ（`2^depth >= threads` を満たす最小の `depth`）。トップから `depth` 段だけ
/// 左右を別スレッドに振る（合計で高々 `2^depth - 1` 回スレッドを起こす）。`threads <= 1` か、
/// 三角形数が `PARALLEL_MIN_TRIS` 未満なら 0（＝並列化しない。`build_range` が 1 段目で
/// `build_node_sequential` に切り替わり、旧実装と同じ 1 パスの構築になる）。
fn parallel_depth_budget(threads: usize, n_tris: usize) -> usize {
    if threads <= 1 || n_tris < PARALLEL_MIN_TRIS {
        return 0;
    }
    let mut d = 0usize;
    while (1usize << d) < threads {
        d += 1;
    }
    d
}

/// 2 分木 `nodes` を [`WIDE_WIDTH`] 分木に畳む。**分割ロジックには触れず**、できあがった 2 分木を上から畳むだけ:
/// ノードの子を並べ、子の数が `WIDE_WIDTH` になるまで「コスト（表面積 × 三角形数）が最大の内部の子」を
/// その 2 つの子で置き換える（貪欲。子が `WIDE_WIDTH` に満たないノードは、残りが全部リーフのとき）。
/// 展開は元の左右の順を保つ。
fn build_wide(nodes: &[BvhNode]) -> Vec<WideNode> {
    if nodes.is_empty() {
        return Vec::new();
    }
    // 各 2 分木ノードの三角形数（コストの見積もり）
    let mut cnt = vec![0u32; nodes.len()];
    fn fill(nodes: &[BvhNode], cnt: &mut [u32], i: usize) -> u32 {
        let n = &nodes[i];
        let c = if n.left == -1 { n.count } else { fill(nodes, cnt, n.left as usize) + fill(nodes, cnt, n.right as usize) };
        cnt[i] = c;
        c
    }
    fill(nodes, &mut cnt, 0);
    let area = |i: usize| {
        let e = nodes[i].bbox.extent();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    };

    fn emit(nodes: &[BvhNode], cnt: &[u32], area: &dyn Fn(usize) -> f64, out: &mut Vec<WideNode>, root: usize) -> usize {
        // 子の並び（2 分木のノード番号）。根がリーフなら 1 つだけ
        let mut slots: Vec<usize> = if nodes[root].left == -1 { vec![root] } else { vec![nodes[root].left as usize, nodes[root].right as usize] };
        while slots.len() < WIDE_WIDTH {
            let mut best: Option<(usize, f64)> = None;
            for (k, &s) in slots.iter().enumerate() {
                if nodes[s].left != -1 {
                    let cost = area(s) * cnt[s] as f64;
                    if best.map_or(true, |(_, c)| cost > c) {
                        best = Some((k, cost));
                    }
                }
            }
            let Some((k, _)) = best else { break };
            let (l, r) = (nodes[slots[k]].left as usize, nodes[slots[k]].right as usize);
            slots[k] = l;
            slots.insert(k + 1, r);
        }
        let w = out.len();
        let inf = f64::INFINITY;
        out.push(WideNode {
            n: slots.len() as u8,
            min_x: [inf; WIDE_WIDTH], min_y: [inf; WIDE_WIDTH], min_z: [inf; WIDE_WIDTH],
            max_x: [-inf; WIDE_WIDTH], max_y: [-inf; WIDE_WIDTH], max_z: [-inf; WIDE_WIDTH],
            child: [0; WIDE_WIDTH], count: [0; WIDE_WIDTH],
        });
        for (k, &s) in slots.iter().enumerate() {
            let b = nodes[s].bbox;
            let node = &mut out[w];
            node.min_x[k] = b.min.x; node.min_y[k] = b.min.y; node.min_z[k] = b.min.z;
            node.max_x[k] = b.max.x; node.max_y[k] = b.max.y; node.max_z[k] = b.max.z;
            if nodes[s].left == -1 {
                node.child[k] = nodes[s].start;
                debug_assert!(nodes[s].count as usize <= LEAF_SIZE && LEAF_SIZE < 256);
                node.count[k] = nodes[s].count as u8;
            } else {
                let c = emit(nodes, cnt, area, out, s);
                out[w].child[k] = c as u32;
            }
        }
        w
    }

    let mut out = Vec::new();
    emit(nodes, &cnt, &area, &mut out, 0);
    out
}

impl Bvh {
    /// SAH を用いて三角形群から BVH を構築する。ワーカースレッド数は
    /// `std::thread::available_parallelism`（`render.rs` と同じ）に合わせる。
    ///
    /// 1. 全三角形の AABB と重心を事前計算
    /// 2. 再帰的に最適分割位置を SAH で決定
    /// 3. SAH が使えない場合（要素数 < 2·SAH_BINS、重心が縮退、有効分割なし）は
    ///    最大広がり軸の重心による中央値分割（select_nth）にフォールバック
    ///
    /// **並列化してもビット単位で逐次版と同じ BVH を作る**（分割の判断そのもの・分割順序は
    /// 一切変えていない。上位の再帰を複数スレッドに分けて、木の同じ場所を並行に組み立てているだけ。
    /// 詳細は [`build_range`] のドキュメント）。
    pub fn build(src: TriangleSource<'_>) -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::build_with_threads(src, threads)
    }

    /// [`Bvh::build`] のスレッド数を明示できる版。`build` は `available_parallelism` を渡すだけの
    /// 薄いラッパー。`threads <= 1` で呼べば常に逐次版と同じコード経路（`build_node_sequential` 1 回）
    /// を通るので、テストで「逐次 = `build_with_threads(tris, 1)`」「並列 = `build_with_threads(tris, N)`」
    /// を比較できる。
    fn build_with_threads(src: TriangleSource<'_>, threads: usize) -> Self {
        let tri_bounds: Vec<Aabb> = (0..src.len()).map(|ti| src.bounds(ti)).collect();
        Self::build_from_bounds(&tri_bounds, threads)
    }

    /// プリミティブごとの AABB だけから BVH を作る共通ビルダー（binned SAH・並列部分木構築）。
    /// プリミティブの種類には依存しない: 三角形用（[`Bvh::build`]）と、`World` のトップレベル BVH
    /// （インスタンスと球の境界）が同じ実装を共有する。リーフの `indices` は `bounds` の添字。
    pub(crate) fn build_from_bounds(bounds: &[Aabb], threads: usize) -> Self {
        let (nodes, indices) = Self::build_binary(bounds, threads);
        let wide = build_wide(&nodes);
        Self { wide, indices, bounds: nodes.first().map(|n| n.bbox) }
    }

    /// SAH の 2 分木（ノード配列と、リーフが参照するプリミティブ番号の並び）。走査用の広い BVH は
    /// これを畳んで作る（[`Bvh::build_from_bounds`]）。並列構築 = 逐次構築のビット一致をテストするために分けてある。
    pub(crate) fn build_binary(bounds: &[Aabb], threads: usize) -> (Vec<BvhNode>, Vec<usize>) {
        let mut indices: Vec<usize> = (0..bounds.len()).collect();
        let centroids: Vec<Vec3> = bounds.iter().map(|b| b.centroid()).collect();
        let nodes = if indices.is_empty() {
            Vec::new()
        } else {
            let depth_budget = parallel_depth_budget(threads, indices.len());
            build_range(&mut indices, 0, bounds, &centroids, depth_budget)
        };
        (nodes, indices)
    }

    /// 全体の AABB（プリミティブが 0 個なら `None`）。
    pub fn root_bounds(&self) -> Option<Aabb> {
        self.bounds
    }

    /// BVH をトラバースしてレイとの最近接交差を返す。
    ///
    /// スタックベースの反復トラバーサルを使用。
    /// 子ノードの AABB 交差距離を比較し、近い方を先に処理して早期枝刈りを最大化する。
    pub fn hit(&self, src: TriangleSource<'_>, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        self.hit_filtered(src, r, tmin, tmax, |_, _, _| true)
    }

    /// 広い BVH（[`WIDE_WIDTH`] 分木）での最近接探索。子のスラブ判定は 1 ノードぶんをまとめて行い、
    /// 当たった子を入口 `t0` の昇順に辿る。棄却した候補（アルファ透明）では `tmax` を縮めない。
    ///
    /// **同値 `t` のタイブレーク: プリミティブ番号（`indices` に格納された値 = 三角形の添字）が小さい方が勝つ。**
    /// 走査順に依存しない決定的な規則で、広い BVH に変えても（走査順が変わっても）結果が変わらない。
    /// `World` の TLAS（葉の通し番号が小さい方が勝つ）と同じ規則。
    #[inline(always)]
    pub fn hit_filtered<F: Fn(usize, f64, f64) -> bool>(
        &self,
        src: TriangleSource<'_>,
        r: Ray,
        tmin: f64,
        mut tmax: f64,
        accept: F,
    ) -> Option<Hit> {
        if self.wide.is_empty() {
            return None;
        }
        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        // 未初期化のスタック（積んだ分 `[..sp]` だけを読む。0 埋めは走査 1 回ごとに 2KB の書き込みになるので避ける）
        let mut stack_buf: [std::mem::MaybeUninit<Entry>; 96] = [const { std::mem::MaybeUninit::uninit() }; 96];
        let mut sp = 0usize;
        let mut heap: Vec<Entry> = Vec::new();
        // 最近接候補は (三角形, t, u, v) だけを保持し、交差点と誤差上界は最後に 1 回だけ計算する
        let mut best: Option<(usize, f64, f64, f64)> = None;
        macro_rules! push {
            ($e:expr) => {{
                let e = $e;
                if heap.is_empty() && sp < stack_buf.len() {
                    stack_buf[sp] = std::mem::MaybeUninit::new(e);
                    sp += 1;
                } else {
                    if heap.is_empty() {
                        // SAFETY: `[..sp]` は積んで初期化した要素だけ
                        heap.extend(stack_buf[..sp].iter().map(|x| unsafe { x.assume_init() }));
                        sp = 0;
                    }
                    heap.push(e);
                }
            }};
        }
        macro_rules! push_children {
            ($w:expr, $tmax:expr) => {{
                let w = $w;
                let nd = &self.wide[w as usize];
                let mut hits = [(0.0f64, 0u32); WIDE_WIDTH];
                let mut k = 0usize;
                for i in 0..nd.n as usize {
                    if let Some(t0) = nd.slot_entry(i, &r, inv, tmin, $tmax) {
                        hits[k] = (t0, i as u32);
                        k += 1;
                    }
                }
                // 入口 t0 の降順に並べて積む（後入れ先出しなので、最も近い子が先に出る）
                for a in 1..k {
                    let mut b = a;
                    while b > 0 && hits[b - 1].0 < hits[b].0 {
                        hits.swap(b - 1, b);
                        b -= 1;
                    }
                }
                for j in 0..k {
                    push!(Entry { t0: hits[j].0, w, slot: hits[j].1 });
                }
            }};
        }

        push_children!(0u32, tmax);
        loop {
            let e = if !heap.is_empty() {
                heap.pop().unwrap()
            } else if sp > 0 {
                sp -= 1;
                // SAFETY: `stack_buf[sp]` は積んで初期化済み
                unsafe { stack_buf[sp].assume_init() }
            } else {
                break;
            };
            // 積んだ後に最近接が縮んで、もう届かない子は捨てる（従来の「取り出したときの箱判定」と同じ）
            if e.t0 > tmax {
                continue;
            }
            let nd = &self.wide[e.w as usize];
            let i = e.slot as usize;
            if nd.count[i] > 0 {
                let start = nd.child[i] as usize;
                for pos in start..start + nd.count[i] as usize {
                    let ti = self.indices[pos];
                    if let Some((t, u, v)) = src.intersect(ti, r, tmin, tmax) {
                        if !accept(ti, u, v) {
                            continue;
                        }
                        // 同値 t は番号の小さい方が勝つ（走査順に依存しない規則）。t が小さければ無条件に勝つ
                        if best.map_or(true, |(bi, bt, _, _)| t < bt || (t == bt && ti < bi)) {
                            // 区間の上端は、同値 t の候補（丸めで t_scaled が tmax·det をわずかに超える）が
                            // 採用判定で落ちないよう少し広げる。勝敗は上の (t, 番号) の比較だけで決まる
                            tmax = t * TIE_SLACK;
                            best = Some((ti, t, u, v));
                        }
                    }
                }
            } else {
                push_children!(nd.child[i], tmax);
            }
        }

        best.map(|(ti, t, u, v)| {
            let mut h = src.hit_at(ti, r, t, u, v);
            h.prim_id = ti;
            h
        })
    }

    /// 葉のプリミティブ番号ごとに `visit(k)` を呼ぶ汎用の走査（TLAS 用。広い BVH）。箱の判定は
    /// 区間 `(tmin, tmax·(1 + 1e-9))`（`tmax` は `visit` の中で縮められる）で、当たった子を入口 `t0` の昇順に辿る。
    /// `visit` が true を返したら打ち切る（any-hit）。訪問順は 2 分木の走査と違うが、呼び出し側（`World`）が
    /// 同値 `t` を添字で解決するので結果は同じ。
    pub(crate) fn traverse_wide(&self, r: Ray, tmin: f64, tmax: &std::cell::Cell<f64>, mut visit: impl FnMut(usize) -> bool) {
        if self.wide.is_empty() {
            return;
        }
        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        let mut stack_buf: [std::mem::MaybeUninit<Entry>; 96] = [const { std::mem::MaybeUninit::uninit() }; 96];
        let mut sp = 0usize;
        let mut heap: Vec<Entry> = Vec::new();
        macro_rules! push {
            ($e:expr) => {{
                let e = $e;
                if heap.is_empty() && sp < stack_buf.len() {
                    stack_buf[sp] = std::mem::MaybeUninit::new(e);
                    sp += 1;
                } else {
                    if heap.is_empty() {
                        // SAFETY: `[..sp]` は積んで初期化した要素だけ
                        heap.extend(stack_buf[..sp].iter().map(|x| unsafe { x.assume_init() }));
                        sp = 0;
                    }
                    heap.push(e);
                }
            }};
        }
        macro_rules! push_children {
            ($w:expr) => {{
                let w = $w;
                let nd = &self.wide[w as usize];
                let tmax_box = tmax.get() * (1.0 + 1e-9);
                let mut hits = [(0.0f64, 0u32); WIDE_WIDTH];
                let mut k = 0usize;
                for i in 0..nd.n as usize {
                    if let Some(t0) = nd.slot_entry(i, &r, inv, tmin, tmax_box) {
                        hits[k] = (t0, i as u32);
                        k += 1;
                    }
                }
                for a in 1..k {
                    let mut b = a;
                    while b > 0 && hits[b - 1].0 < hits[b].0 {
                        hits.swap(b - 1, b);
                        b -= 1;
                    }
                }
                for j in 0..k {
                    push!(Entry { t0: hits[j].0, w, slot: hits[j].1 });
                }
            }};
        }
        push_children!(0u32);
        loop {
            let e = if !heap.is_empty() {
                heap.pop().unwrap()
            } else if sp > 0 {
                sp -= 1;
                // SAFETY: `stack_buf[sp]` は積んで初期化済み
                unsafe { stack_buf[sp].assume_init() }
            } else {
                return;
            };
            if e.t0 > tmax.get() * (1.0 + 1e-9) {
                continue;
            }
            let nd = &self.wide[e.w as usize];
            let i = e.slot as usize;
            if nd.count[i] > 0 {
                let start = nd.child[i] as usize;
                for pos in start..start + nd.count[i] as usize {
                    if visit(self.indices[pos]) {
                        return;
                    }
                }
            } else {
                push_children!(nd.child[i]);
            }
        }
    }

    /// [`Bvh::hit_filtered`] の any-hit 版（広い BVH）。採用できる交差が 1 つ見つかった時点で返す
    /// （最近接の保証はない。遮蔽の有無だけが要る呼び出し側でのみ使う）。
    #[inline(always)]
    pub fn any_hit_filtered<F: Fn(usize, f64, f64) -> bool>(
        &self,
        src: TriangleSource<'_>,
        r: Ray,
        tmin: f64,
        tmax: f64,
        accept: F,
    ) -> Option<Hit> {
        if self.wide.is_empty() {
            return None;
        }
        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);
        // 未初期化のスタック（積んだ分 `[..sp]` だけを読む。0 埋めは走査 1 回ごとに 2KB の書き込みになるので避ける）
        let mut stack_buf: [std::mem::MaybeUninit<Entry>; 96] = [const { std::mem::MaybeUninit::uninit() }; 96];
        let mut sp = 0usize;
        let mut heap: Vec<Entry> = Vec::new();

        macro_rules! push {
            ($e:expr) => {{
                let e = $e;
                if heap.is_empty() && sp < stack_buf.len() {
                    stack_buf[sp] = std::mem::MaybeUninit::new(e);
                    sp += 1;
                } else {
                    if heap.is_empty() {
                        // SAFETY: `[..sp]` は積んで初期化した要素だけ
                        heap.extend(stack_buf[..sp].iter().map(|x| unsafe { x.assume_init() }));
                        sp = 0;
                    }
                    heap.push(e);
                }
            }};
        }
        macro_rules! push_children {
            ($w:expr) => {{
                let w = $w;
                let nd = &self.wide[w as usize];
                let mut hits = [(0.0f64, 0u32); WIDE_WIDTH];
                let mut k = 0usize;
                for i in 0..nd.n as usize {
                    if let Some(t0) = nd.slot_entry(i, &r, inv, tmin, tmax) {
                        hits[k] = (t0, i as u32);
                        k += 1;
                    }
                }
                for a in 1..k {
                    let mut b = a;
                    while b > 0 && hits[b - 1].0 < hits[b].0 {
                        hits.swap(b - 1, b);
                        b -= 1;
                    }
                }
                for j in 0..k {
                    push!(Entry { t0: hits[j].0, w, slot: hits[j].1 });
                }
            }};
        }

        push_children!(0u32);
        loop {
            let e = if !heap.is_empty() {
                heap.pop().unwrap()
            } else if sp > 0 {
                sp -= 1;
                // SAFETY: `stack_buf[sp]` は積んで初期化済み
                unsafe { stack_buf[sp].assume_init() }
            } else {
                return None;
            };
            let nd = &self.wide[e.w as usize];
            let i = e.slot as usize;
            if nd.count[i] > 0 {
                let start = nd.child[i] as usize;
                for pos in start..start + nd.count[i] as usize {
                    let ti = self.indices[pos];
                    if let Some((t, u, v)) = src.intersect(ti, r, tmin, tmax) {
                        if !accept(ti, u, v) {
                            continue;
                        }
                        let mut h = src.hit_at(ti, r, t, u, v);
                        h.prim_id = ti;
                        return Some(h);
                    }
                }
            } else {
                push_children!(nd.child[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Triangle;
    use crate::rng::Rng;

    fn random_tris(n: usize, rng: &mut Rng) -> Vec<Triangle> {
        let mut rv = |s: f64| Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * s;
        (0..n)
            .map(|i| {
                let c = rv(10.0);
                Triangle::new_static(c + rv(1.0), c + rv(1.0), c + rv(1.0), i)
            })
            .collect()
    }

    /// SAH の 2 分木（構築の直後。走査用の広い BVH は畳んだ後に 2 分木を捨てるので、構造のテストはこちらを見る）。
    struct Bin {
        nodes: Vec<BvhNode>,
        indices: Vec<usize>,
    }

    fn binary(tris: &[Triangle], threads: usize) -> Bin {
        let bounds: Vec<Aabb> = tris.iter().map(|t| t.bounds()).collect();
        let (nodes, indices) = Bvh::build_binary(&bounds, threads);
        Bin { nodes, indices }
    }

    // ---- PERF-2: BVH 構築の並列化（逐次版とのビット一致） ----


    /// ノード配列・インデックス配列が完全に一致するか（`bbox` はビット単位）。
    fn bvh_bit_identical(a: &Bin, b: &Bin) -> bool {
        if a.indices != b.indices || a.nodes.len() != b.nodes.len() {
            return false;
        }
        a.nodes.iter().zip(b.nodes.iter()).all(|(x, y)| {
            x.bbox.min.x.to_bits() == y.bbox.min.x.to_bits()
                && x.bbox.min.y.to_bits() == y.bbox.min.y.to_bits()
                && x.bbox.min.z.to_bits() == y.bbox.min.z.to_bits()
                && x.bbox.max.x.to_bits() == y.bbox.max.x.to_bits()
                && x.bbox.max.y.to_bits() == y.bbox.max.y.to_bits()
                && x.bbox.max.z.to_bits() == y.bbox.max.z.to_bits()
                && x.left == y.left
                && x.right == y.right
                && x.start == y.start
                && x.count == y.count
        })
    }

    /// 並列版（`threads` 複数）は逐次版（`threads=1`）とノード配列・インデックス配列がビット単位で
    /// 完全に一致する。`PARALLEL_MIN_TRIS` を超える三角形数（並列経路が実際に有効になるサイズ）で、
    /// スレッド数を複数通り振って確認する。
    #[test]
    fn parallel_build_matches_sequential_bit_for_bit_for_a_large_mesh() {
        let mut rng = Rng::new(4242);
        let tris = random_tris(PARALLEL_MIN_TRIS + 20_000, &mut rng);
        let seq = binary(&tris, 1);
        assert!(seq.nodes.len() > 1, "test setup: mesh should actually split");
        for &threads in &[2usize, 3, 4, 8, 16] {
            let par = binary(&tris, threads);
            assert!(bvh_bit_identical(&seq, &par), "threads={}: BVH differs from the sequential build", threads);
        }
    }

    /// 三角形数がちょうど `PARALLEL_MIN_TRIS` をまたぐあたりでも一致すること（しきい値の境界）。
    #[test]
    fn parallel_build_matches_sequential_around_the_min_tris_threshold() {
        let mut rng = Rng::new(99);
        for &n in &[PARALLEL_MIN_TRIS - 1, PARALLEL_MIN_TRIS, PARALLEL_MIN_TRIS + 1] {
            let tris = random_tris(n, &mut rng);
            let seq = binary(&tris, 1);
            let par = binary(&tris, 8);
            assert!(bvh_bit_identical(&seq, &par), "n={}: BVH differs from the sequential build", n);
        }
    }

    /// 退化ケース（空・三角形 1 個・全部同一位置）でも並列版と逐次版が一致し、パニックしない。
    /// 同一位置の三角形は重心が一点に潰れる＝ SAH が使えず中央値分割にフォールバックする経路。
    #[test]
    fn parallel_build_matches_sequential_for_degenerate_inputs() {
        // 空メッシュ
        let empty: Vec<Triangle> = Vec::new();
        let seq = binary(&empty, 1);
        let par = binary(&empty, 8);
        assert!(bvh_bit_identical(&seq, &par));
        assert!(seq.nodes.is_empty());

        // 三角形 1 個
        let one = vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let seq = binary(&one, 1);
        let par = binary(&one, 8);
        assert!(bvh_bit_identical(&seq, &par));

        // 全部同一位置（重心が一点に潰れる）。並列経路に乗る数まで増やす
        let degenerate: Vec<Triangle> = (0..PARALLEL_MIN_TRIS + 1000)
            .map(|i| Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), i))
            .collect();
        let seq = binary(&degenerate, 1);
        let par = binary(&degenerate, 8);
        assert!(bvh_bit_identical(&seq, &par), "degenerate（同一位置）でビット一致しない");
    }

    /// 並列構築されたツリーも、通常の交差判定テスト（総当たり比較・水密性・リーフの分割）を満たす
    /// （逐次と同一なら自明だが、独立した確認として）。
    #[test]
    fn parallel_build_is_a_valid_bvh() {
        let mut rng = Rng::new(55);
        let tris = random_tris(PARALLEL_MIN_TRIS + 5000, &mut rng);
        let bin = binary(&tris, 8);
        let bvh = Bvh::build_with_threads((&tris).into(), 8);
        // 全三角形がちょうど 1 つのリーフに属す
        let mut seen = vec![0u32; tris.len()];
        for node in &bin.nodes {
            if node.left == -1 {
                assert!(node.count as usize <= LEAF_SIZE);
                for &i in &bin.indices[node.start as usize..(node.start + node.count) as usize] {
                    seen[i] += 1;
                }
            } else {
                assert!(node.right != -1 && node.count == 0);
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        // 総当たりとの一致（サンプル）
        for _ in 0..200 {
            let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
            let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
            let r = Ray { o, d: (target - o).norm(), time: 0.0 };
            let a = bvh.hit((&tris).into(), r, 1e-4, 1e30).map(|h| h.t);
            let b = brute_force(&tris, r, 1e-4, 1e30);
            assert_eq!(a, b);
        }
    }

    fn brute_force(tris: &[Triangle], r: Ray, tmin: f64, tmax: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for t in tris {
            if let Some(h) = t.hit(r, tmin, best.unwrap_or(tmax)) {
                best = Some(h.t);
            }
        }
        best
    }

    /// BVH の最近接交差距離は総当たりと一致する（SAH 経路・中央値分割経路の両方を含むサイズ）。
    #[test]
    fn bvh_hit_matches_brute_force() {
        let mut rng = Rng::new(123);
        for &n in &[1usize, 5, 12, 15, 16, 40, 500] {
            let tris = random_tris(n, &mut rng);
            let bvh = Bvh::build((&tris).into());
            for _ in 0..500 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                let a = bvh.hit((&tris).into(), r, 1e-4, 1e30).map(|h| h.t);
                let b = brute_force(&tris, r, 1e-4, 1e30);
                assert_eq!(a, b, "n={}", n);
            }
        }
    }

    /// `any_hit_filtered` の「交差の有無」は `hit_filtered`（最近接探索）と常に一致する
    /// （総当たりのランダムシーン・レイで比較。`accept` は毎回同じ判定関数を渡すので、
    /// 採否そのものではなく「どこかに採用できる交差があるか」の一致だけを見る）。
    #[test]
    fn any_hit_existence_matches_nearest_hit_existence() {
        let mut rng = Rng::new(777);
        for &n in &[1usize, 5, 12, 16, 40, 500] {
            let tris = random_tris(n, &mut rng);
            let bvh = Bvh::build((&tris).into());
            for _ in 0..500 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                // 三角形番号が偶数のものだけ採用する（アルファ透明の棄却を模した accept）
                let accept = |ti: usize, _u: f64, _v: f64| ti % 2 == 0;
                let nearest = bvh.hit_filtered((&tris).into(), r, 1e-4, 1e30, accept).is_some();
                let any = bvh.any_hit_filtered((&tris).into(), r, 1e-4, 1e30, accept).is_some();
                assert_eq!(any, nearest, "n={}: any_hit_filtered と hit_filtered の存在判定が食い違った", n);
            }
        }
    }

    /// 全三角形がちょうど 1 つのリーフに属し、リーフ以外は子を 2 つ持つ。
    #[test]
    fn bvh_leaves_partition_all_triangles() {
        let mut rng = Rng::new(5);
        for &n in &[3usize, 12, 15, 100] {
            let tris = random_tris(n, &mut rng);
            let bvh = binary(&tris, 1);
            let mut seen = vec![0u32; n];
            for node in &bvh.nodes {
                if node.left == -1 {
                    assert!(node.count as usize <= LEAF_SIZE);
                    for &i in &bvh.indices[node.start as usize..(node.start + node.count) as usize] {
                        seen[i] += 1;
                    }
                } else {
                    assert!(node.right != -1 && node.count == 0);
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "n={}", n);
        }
    }

    /// SAH を使わない小さなノード（n < 2·SAH_BINS）でも、重心で空間的に分割される。
    /// 以前は並び順のまま半分に割っていたため、左右の子の重心範囲が重なっていた。
    #[test]
    fn small_node_split_is_spatial() {
        // x 方向に並んだ 12 枚を逆順・交互に並べて入力する
        let order = [11usize, 0, 9, 2, 7, 4, 5, 6, 3, 8, 1, 10];
        let tris: Vec<Triangle> = order
            .iter()
            .map(|&i| {
                let x = i as f64 * 2.0;
                Triangle::new_static(Vec3::new(x, 0.0, 0.0), Vec3::new(x + 0.5, 0.0, 0.0), Vec3::new(x, 0.5, 0.0), i)
            })
            .collect();
        let bvh = binary(&tris, 1);
        let root = bvh.nodes[0];
        let l = bvh.nodes[root.left as usize].bbox;
        let r = bvh.nodes[root.right as usize].bbox;
        assert!(l.max.x < r.min.x || r.max.x < l.min.x, "children overlap on the split axis: {:?} {:?}", l, r);
    }

    /// 採否判定つき探索は「棄却した三角形を最初から無いものとした総当たり」と一致する。
    /// 常に true の判定は従来の `hit` と完全に同じ結果。
    #[test]
    fn hit_filtered_matches_brute_force_over_accepted_triangles() {
        let mut rng = Rng::new(77);
        for &n in &[1usize, 12, 40, 500] {
            let tris = random_tris(n, &mut rng);
            let bvh = Bvh::build((&tris).into());
            let kept: Vec<Triangle> = tris.iter().enumerate().filter(|(i, _)| i % 3 != 0).map(|(_, t)| *t).collect();
            for _ in 0..400 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                let a = bvh.hit_filtered((&tris).into(), r, 1e-4, 1e30, |ti, _, _| ti % 3 != 0).map(|h| h.t);
                assert_eq!(a, brute_force(&kept, r, 1e-4, 1e30), "n={}", n);
                let all = bvh.hit_filtered((&tris).into(), r, 1e-4, 1e30, |_, _, _| true).map(|h| h.t);
                assert_eq!(all, bvh.hit((&tris).into(), r, 1e-4, 1e30).map(|h| h.t));
            }
        }
    }

    // ---- 広い BVH（2 分木を畳んだ N 分木の走査）----

    /// 2 つの結果がビット単位で同じか（`None` 同士も一致）。
    fn same_hit(a: &Option<Hit>, b: &Option<Hit>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                let bits = |v: Vec3| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
                a.t.to_bits() == b.t.to_bits() && bits(a.p) == bits(b.p) && bits(a.ng) == bits(b.ng) && bits(a.ns) == bits(b.ns)
                    && a.prim_id == b.prim_id && a.mat_id == b.mat_id && a.bary.0.to_bits() == b.bary.0.to_bits() && a.bary.1.to_bits() == b.bary.1.to_bits()
            }
            _ => false,
        }
    }

    /// 総当たりで決める勝者: 全三角形を独立に判定し、`(t, 番号)` が最小のもの（同値 t は番号の小さい方）。
    fn brute_winner(tris: &[Triangle], r: Ray, tmin: f64, tmax: f64) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for (ti, t) in tris.iter().enumerate() {
            if let Some((tt, _, _)) = t.intersect(r, tmin, tmax) {
                if best.map_or(true, |(bt, _)| tt < bt) {
                    best = Some((tt, ti));
                }
            }
        }
        best
    }

    /// 広い BVH の `hit` は、総当たりと `t`・`prim_id`・`mat_id` まで一致する。**同一の三角形を重ねた
    /// （同値 `t` が必ず出る）メッシュ**を含め、同値 t は番号の小さい方が勝つ規則が走査順によらず守られていることを見る。
    /// any-hit は存在が総当たりと一致する。
    #[test]
    fn wide_hit_matches_brute_force_including_ties() {
        let mut rng = Rng::new(2024);
        for (n, dup) in [(1usize, 0usize), (3, 1), (40, 0), (500, 0), (400, 3), (3000, 2)] {
            let mut tris = random_tris(n, &mut rng);
            // 重ね置き: 元と同じ頂点の三角形を、別の材質番号で（番号が大きい側に）追加する
            let base = tris.len();
            for _ in 0..dup {
                for k in 0..base {
                    let t = tris[k];
                    tris.push(Triangle::new_static(t.v0_0, t.v1_0, t.v2_0, 1000 + tris.len()));
                }
            }
            let bvh = Bvh::build(TriangleSource::from(&tris));
            let mut ties = 0;
            for _ in 0..4000 {
                // 三角形の重心付近を狙うと、同値の重なりに当たりやすい
                let target = { let t = tris[(rng.next_f64() * tris.len() as f64) as usize % tris.len()]; (t.v0_0 + t.v1_0 + t.v2_0) / 3.0 };
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                let src = TriangleSource::from(&tris);
                let got = bvh.hit(src, r, 0.0, 1e30);
                let want = brute_winner(&tris, r, 0.0, 1e30);
                match (&got, want) {
                    (None, None) => {}
                    (Some(h), Some((t, ti))) => {
                        assert_eq!(h.t.to_bits(), t.to_bits(), "n={n} dup={dup}: t");
                        assert_eq!(h.prim_id, ti, "n={n} dup={dup}: prim_id");
                        assert_eq!(h.mat_id, tris[ti].mat_id, "n={n} dup={dup}: mat_id");
                        if dup > 0 { ties += 1; }
                    }
                    _ => panic!("n={n} dup={dup}: hit {:?} vs brute force {:?}", got.map(|h| (h.t, h.prim_id)), want),
                }
                let any = bvh.any_hit_filtered(src, r, 0.0, 1e30, |_, _, _| true).is_some();
                assert_eq!(any, want.is_some());
            }
            if dup > 0 { assert!(ties > 100, "重ね置きの同値ケースが少なすぎる: {ties}"); }
        }
    }

    /// 走査順を変える（ノードの畳み方が違う）ことがあっても結果が変わらないこと: 別のスレッド数で作った BVH
    /// （構造は同一だが、念のため）と、レイの向きを変えた多数のレイで `hit` の結果が完全に一致する。
    /// 加えて、同じメッシュを反転した並びで作った BVH（番号 → 位置の対応が違う = 走査順が違う）でも、
    /// 番号で見た勝者は同じ。
    #[test]
    fn winner_does_not_depend_on_the_traversal_order() {
        let mut rng = Rng::new(31);
        let mut tris = random_tris(600, &mut rng);
        let base = tris.len();
        for k in 0..base {
            let t = tris[k];
            tris.push(Triangle::new_static(t.v0_0, t.v1_0, t.v2_0, 5000 + k));
        }
        let a = Bvh::build_with_threads((&tris).into(), 1);
        let b = Bvh::build_with_threads((&tris).into(), 8);
        // 並びを逆にしたメッシュ: 番号 i は base*2 - 1 - i に対応
        let rev: Vec<Triangle> = tris.iter().rev().copied().collect();
        let c = Bvh::build((&rev).into());
        let n = tris.len();
        for _ in 0..3000 {
            let target = { let t = tris[(rng.next_f64() * n as f64) as usize % n]; (t.v0_0 + t.v1_0 + t.v2_0) / 3.0 };
            let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
            let r = Ray { o, d: (target - o).norm(), time: 0.0 };
            let ha = a.hit((&tris).into(), r, 0.0, 1e30);
            let hb = b.hit((&tris).into(), r, 0.0, 1e30);
            assert!(same_hit(&ha, &hb));
            // 逆並びでは「番号が小さい方が勝つ」は元の番号で大きい方 = 重ね置きの複製側が勝つので、t だけ一致すればよい
            let hc = c.hit((&rev).into(), r, 0.0, 1e30);
            assert_eq!(ha.map(|h| h.t.to_bits()), hc.map(|h| h.t.to_bits()));
        }
    }

    /// 広い BVH の構造: 全三角形がちょうど 1 回リーフに現れ、子スロット数は 1..=WIDE_WIDTH、リーフ以外は子ノードを指す。
    #[test]
    fn wide_nodes_partition_all_triangles() {
        let mut rng = Rng::new(9);
        for n in [1usize, 2, 4, 5, 17, 300, 5000] {
            let tris = random_tris(n, &mut rng);
            let bvh = Bvh::build(TriangleSource::from(&tris));
            let mut seen = vec![0u32; n];
            for nd in &bvh.wide {
                assert!((1..=WIDE_WIDTH as u8).contains(&nd.n));
                for i in 0..nd.n as usize {
                    if nd.count[i] > 0 {
                        for pos in nd.child[i] as usize..nd.child[i] as usize + nd.count[i] as usize {
                            seen[bvh.indices[pos]] += 1;
                        }
                    } else {
                        assert!((nd.child[i] as usize) < bvh.wide.len());
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "n={n}");
        }
    }
}
