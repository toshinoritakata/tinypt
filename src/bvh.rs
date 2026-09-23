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

use crate::constants::bvh::{LEAF_SIZE, PARALLEL_MIN_TRIS, SAH_BINS};
use crate::geometry::{Aabb, Hit, Triangle};
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

/// 三角形群に対する BVH（Bounding Volume Hierarchy）。
pub struct Bvh {
    /// 線形配列に格納された BVH ノード群
    pub nodes: Vec<BvhNode>,
    /// リーフが参照する三角形インデックスの並び順
    pub indices: Vec<usize>,
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
    pub fn build(tris: &[Triangle]) -> Self {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::build_with_threads(tris, threads)
    }

    /// [`Bvh::build`] のスレッド数を明示できる版。`build` は `available_parallelism` を渡すだけの
    /// 薄いラッパー。`threads <= 1` で呼べば常に逐次版と同じコード経路（`build_node_sequential` 1 回）
    /// を通るので、テストで「逐次 = `build_with_threads(tris, 1)`」「並列 = `build_with_threads(tris, N)`」
    /// を比較できる。
    fn build_with_threads(tris: &[Triangle], threads: usize) -> Self {
        let mut indices: Vec<usize> = (0..tris.len()).collect();

        let tri_bounds: Vec<Aabb> = tris.iter().map(|t| t.bounds()).collect();
        let tri_centroids: Vec<Vec3> = tri_bounds.iter().map(|b| b.centroid()).collect();

        let nodes = if indices.is_empty() {
            Vec::new()
        } else {
            let depth_budget = parallel_depth_budget(threads, indices.len());
            build_range(&mut indices, 0, &tri_bounds, &tri_centroids, depth_budget)
        };
        Self { nodes, indices }
    }

    /// BVH をトラバースしてレイとの最近接交差を返す。
    ///
    /// スタックベースの反復トラバーサルを使用。
    /// 子ノードの AABB 交差距離を比較し、近い方を先に処理して早期枝刈りを最大化する。
    pub fn hit(&self, tris: &[Triangle], r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        self.hit_filtered(tris, r, tmin, tmax, |_, _, _| true)
    }

    /// [`Bvh::hit`] に候補の採否判定を足したもの。`accept(三角形番号, u, v)` が false の交差は
    /// **無かったことにして**探索を続ける（アルファマスクの透明部分）。
    ///
    /// 棄却した候補では `tmax` を縮めず、区間 `(tmin, tmax)` もそのまま。だから棄却した面の
    /// 先も同じ区間で探し続けるだけで、再開位置の取り方による自己交差は起きない（水密交差と
    /// 誤差上界はそのまま）。`hit` は常に true を返す判定で呼ぶ（単相化されて従来と同じコード）。
    #[inline(always)]
    pub fn hit_filtered<F: Fn(usize, f64, f64) -> bool>(
        &self,
        tris: &[Triangle],
        r: Ray,
        tmin: f64,
        mut tmax: f64,
        accept: F,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);

        let mut stack_buf = [0i32; 64];
        let mut sp = 0usize;
        stack_buf[sp] = 0;
        sp += 1;
        let mut heap_stack: Vec<i32> = Vec::new();
        // 最近接候補は (三角形, t, u, v) だけを保持し、交差点と誤差上界は最後に 1 回だけ計算する
        let mut best: Option<(usize, f64, f64, f64)> = None;

        macro_rules! push_id {
            ($id:expr) => {{
                let id = $id;
                if heap_stack.is_empty() {
                    if sp < stack_buf.len() {
                        stack_buf[sp] = id;
                        sp += 1;
                    } else {
                        heap_stack = stack_buf[..sp].to_vec();
                        heap_stack.push(id);
                    }
                } else {
                    heap_stack.push(id);
                }
            }};
        }

        loop {
            let nid = if heap_stack.is_empty() {
                if sp == 0 {
                    break;
                }
                sp -= 1;
                stack_buf[sp]
            } else {
                match heap_stack.pop() {
                    Some(v) => v,
                    None => break,
                }
            };
            let n = &self.nodes[nid as usize];
            if !n.bbox.hit_inv(r, inv, tmin, tmax) {
                continue;
            }

            if n.left == -1 && n.right == -1 {
                let start = n.start as usize;
                let end = start + n.count as usize;
                for &ti in &self.indices[start..end] {
                    if let Some((t, u, v)) = tris[ti].intersect(r, tmin, tmax) {
                        if !accept(ti, u, v) {
                            continue;
                        }
                        tmax = t;
                        best = Some((ti, t, u, v));
                    }
                }
            } else {
                // Push farther child first so nearer is processed first (LIFO stack).
                let a_id = n.left;
                let b_id = n.right;

                // If either is missing, fall back.
                if a_id == -1 {
                    if b_id != -1 { push_id!(b_id); }
                    continue;
                }
                if b_id == -1 {
                    push_id!(a_id);
                    continue;
                }

                let a = &self.nodes[a_id as usize];
                let b = &self.nodes[b_id as usize];

                let a_hit = a.bbox.hit_range_inv(r, inv, tmin, tmax);
                let b_hit = b.bbox.hit_range_inv(r, inv, tmin, tmax);

                match (a_hit, b_hit) {
                    (Some((a_t0, _)), Some((b_t0, _))) => {
                        // Smaller entry t0 is nearer.
                        if a_t0 <= b_t0 {
                            // push far then near
                            push_id!(b_id);
                            push_id!(a_id);
                        } else {
                            push_id!(a_id);
                            push_id!(b_id);
                        }
                    }
                    (Some(_), None) => {
                        push_id!(a_id);
                    }
                    (None, Some(_)) => {
                        push_id!(b_id);
                    }
                    (None, None) => {}
                }
            }
        }

        best.map(|(ti, t, u, v)| {
            let mut h = tris[ti].hit_at(r, t, u, v);
            h.prim_id = ti;
            h
        })
    }

    /// `hit_filtered` の any-hit 版（シャドウレイ専用）: 採用できる交差が 1 つ見つかった時点で
    /// 探索を打ち切り、その交差を返す（**最近接である保証はない**。遮蔽の有無だけが要る呼び出し側でのみ使うこと）。
    ///
    /// 区間 `(tmin, tmax)` は最後まで縮めない（採用しない候補があっても同じ区間で探し続けるのは
    /// `hit_filtered` と同じで、アルファ透明の扱いと自己交差回避はそのまま）。子ノードは近い方から
    /// 押す（`hit_filtered` と同じ順）が、any-hit では正しさに影響しない（見つかり次第即座に返すため）。
    /// 平均的には近い方から見つかりやすく、無駄なノード訪問を減らせる。
    #[inline(always)]
    pub fn any_hit_filtered<F: Fn(usize, f64, f64) -> bool>(
        &self,
        tris: &[Triangle],
        r: Ray,
        tmin: f64,
        tmax: f64,
        accept: F,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);

        let mut stack_buf = [0i32; 64];
        let mut sp = 0usize;
        stack_buf[sp] = 0;
        sp += 1;
        let mut heap_stack: Vec<i32> = Vec::new();

        macro_rules! push_id {
            ($id:expr) => {{
                let id = $id;
                if heap_stack.is_empty() {
                    if sp < stack_buf.len() {
                        stack_buf[sp] = id;
                        sp += 1;
                    } else {
                        heap_stack = stack_buf[..sp].to_vec();
                        heap_stack.push(id);
                    }
                } else {
                    heap_stack.push(id);
                }
            }};
        }

        loop {
            let nid = if heap_stack.is_empty() {
                if sp == 0 {
                    return None;
                }
                sp -= 1;
                stack_buf[sp]
            } else {
                match heap_stack.pop() {
                    Some(v) => v,
                    None => return None,
                }
            };
            let n = &self.nodes[nid as usize];
            if !n.bbox.hit_inv(r, inv, tmin, tmax) {
                continue;
            }

            if n.left == -1 && n.right == -1 {
                let start = n.start as usize;
                let end = start + n.count as usize;
                for &ti in &self.indices[start..end] {
                    if let Some((t, u, v)) = tris[ti].intersect(r, tmin, tmax) {
                        if !accept(ti, u, v) {
                            continue;
                        }
                        let mut h = tris[ti].hit_at(r, t, u, v);
                        h.prim_id = ti;
                        return Some(h);
                    }
                }
            } else {
                let a_id = n.left;
                let b_id = n.right;

                if a_id == -1 {
                    if b_id != -1 { push_id!(b_id); }
                    continue;
                }
                if b_id == -1 {
                    push_id!(a_id);
                    continue;
                }

                let a = &self.nodes[a_id as usize];
                let b = &self.nodes[b_id as usize];

                let a_hit = a.bbox.hit_range_inv(r, inv, tmin, tmax);
                let b_hit = b.bbox.hit_range_inv(r, inv, tmin, tmax);

                match (a_hit, b_hit) {
                    (Some((a_t0, _)), Some((b_t0, _))) => {
                        if a_t0 <= b_t0 {
                            push_id!(b_id);
                            push_id!(a_id);
                        } else {
                            push_id!(a_id);
                            push_id!(b_id);
                        }
                    }
                    (Some(_), None) => {
                        push_id!(a_id);
                    }
                    (None, Some(_)) => {
                        push_id!(b_id);
                    }
                    (None, None) => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    // ---- PERF-2: BVH 構築の並列化（逐次版とのビット一致） ----


    /// ノード配列・インデックス配列が完全に一致するか（`bbox` はビット単位）。
    fn bvh_bit_identical(a: &Bvh, b: &Bvh) -> bool {
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
        let seq = Bvh::build_with_threads(&tris, 1);
        assert!(seq.nodes.len() > 1, "test setup: mesh should actually split");
        for &threads in &[2usize, 3, 4, 8, 16] {
            let par = Bvh::build_with_threads(&tris, threads);
            assert!(bvh_bit_identical(&seq, &par), "threads={}: BVH differs from the sequential build", threads);
        }
    }

    /// 三角形数がちょうど `PARALLEL_MIN_TRIS` をまたぐあたりでも一致すること（しきい値の境界）。
    #[test]
    fn parallel_build_matches_sequential_around_the_min_tris_threshold() {
        let mut rng = Rng::new(99);
        for &n in &[PARALLEL_MIN_TRIS - 1, PARALLEL_MIN_TRIS, PARALLEL_MIN_TRIS + 1] {
            let tris = random_tris(n, &mut rng);
            let seq = Bvh::build_with_threads(&tris, 1);
            let par = Bvh::build_with_threads(&tris, 8);
            assert!(bvh_bit_identical(&seq, &par), "n={}: BVH differs from the sequential build", n);
        }
    }

    /// 退化ケース（空・三角形 1 個・全部同一位置）でも並列版と逐次版が一致し、パニックしない。
    /// 同一位置の三角形は重心が一点に潰れる＝ SAH が使えず中央値分割にフォールバックする経路。
    #[test]
    fn parallel_build_matches_sequential_for_degenerate_inputs() {
        // 空メッシュ
        let empty: Vec<Triangle> = Vec::new();
        let seq = Bvh::build_with_threads(&empty, 1);
        let par = Bvh::build_with_threads(&empty, 8);
        assert!(bvh_bit_identical(&seq, &par));
        assert!(seq.nodes.is_empty());

        // 三角形 1 個
        let one = vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let seq = Bvh::build_with_threads(&one, 1);
        let par = Bvh::build_with_threads(&one, 8);
        assert!(bvh_bit_identical(&seq, &par));

        // 全部同一位置（重心が一点に潰れる）。並列経路に乗る数まで増やす
        let degenerate: Vec<Triangle> = (0..PARALLEL_MIN_TRIS + 1000)
            .map(|i| Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), i))
            .collect();
        let seq = Bvh::build_with_threads(&degenerate, 1);
        let par = Bvh::build_with_threads(&degenerate, 8);
        assert!(bvh_bit_identical(&seq, &par), "degenerate（同一位置）でビット一致しない");
    }

    /// 並列構築されたツリーも、通常の交差判定テスト（総当たり比較・水密性・リーフの分割）を満たす
    /// （逐次と同一なら自明だが、独立した確認として）。
    #[test]
    fn parallel_build_is_a_valid_bvh() {
        let mut rng = Rng::new(55);
        let tris = random_tris(PARALLEL_MIN_TRIS + 5000, &mut rng);
        let bvh = Bvh::build_with_threads(&tris, 8);
        // 全三角形がちょうど 1 つのリーフに属す
        let mut seen = vec![0u32; tris.len()];
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
        assert!(seen.iter().all(|&c| c == 1));
        // 総当たりとの一致（サンプル）
        for _ in 0..200 {
            let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
            let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
            let r = Ray { o, d: (target - o).norm(), time: 0.0 };
            let a = bvh.hit(&tris, r, 1e-4, 1e30).map(|h| h.t);
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
            let bvh = Bvh::build(&tris);
            for _ in 0..500 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                let a = bvh.hit(&tris, r, 1e-4, 1e30).map(|h| h.t);
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
            let bvh = Bvh::build(&tris);
            for _ in 0..500 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                // 三角形番号が偶数のものだけ採用する（アルファ透明の棄却を模した accept）
                let accept = |ti: usize, _u: f64, _v: f64| ti % 2 == 0;
                let nearest = bvh.hit_filtered(&tris, r, 1e-4, 1e30, accept).is_some();
                let any = bvh.any_hit_filtered(&tris, r, 1e-4, 1e30, accept).is_some();
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
            let bvh = Bvh::build(&tris);
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
        let bvh = Bvh::build(&tris);
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
            let bvh = Bvh::build(&tris);
            let kept: Vec<Triangle> = tris.iter().enumerate().filter(|(i, _)| i % 3 != 0).map(|(_, t)| *t).collect();
            for _ in 0..400 {
                let o = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 30.0;
                let target = Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * 8.0;
                let r = Ray { o, d: (target - o).norm(), time: 0.0 };
                let a = bvh.hit_filtered(&tris, r, 1e-4, 1e30, |ti, _, _| ti % 3 != 0).map(|h| h.t);
                assert_eq!(a, brute_force(&kept, r, 1e-4, 1e30), "n={}", n);
                let all = bvh.hit_filtered(&tris, r, 1e-4, 1e30, |_, _, _| true).map(|h| h.t);
                assert_eq!(all, bvh.hit(&tris, r, 1e-4, 1e30).map(|h| h.t));
            }
        }
    }
}
