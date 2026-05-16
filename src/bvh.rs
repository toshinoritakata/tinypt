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

use crate::constants::bvh::{LEAF_SIZE, SAH_BINS};
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

impl Bvh {
    /// SAH を用いて三角形群から BVH を構築する。
    ///
    /// 1. 全三角形の AABB と重心を事前計算
    /// 2. 再帰的に最適分割位置を SAH で決定
    /// 3. 分割不可能な場合は中央値分割にフォールバック
    pub fn build(tris: &[Triangle]) -> Self {
        let mut indices: Vec<usize> = (0..tris.len()).collect();
        let mut nodes: Vec<BvhNode> = Vec::new();

        let tri_bounds: Vec<Aabb> = tris.iter().map(|t| t.bounds()).collect();
        let tri_centroids: Vec<Vec3> = tri_bounds.iter().map(|b| b.centroid()).collect();

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

        fn build_node(
            nodes: &mut Vec<BvhNode>,
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
                nodes[idx].start = start as u32;
                nodes[idx].count = n as u32;
                return node_index;
            }

            // Split by largest centroid extent (median split)
            let ext = cbox.extent();
            let axis = if ext.x >= ext.y && ext.x >= ext.z {
                0
            } else if ext.y >= ext.z {
                1
            } else {
                2
            };

            let mid = if n >= SAH_BINS * 2 {
                let minc = match axis { 0 => cbox.min.x, 1 => cbox.min.y, _ => cbox.min.z };
                let maxc = match axis { 0 => cbox.max.x, 1 => cbox.max.y, _ => cbox.max.z };
                let extent = maxc - minc;

                if extent > 1e-12 {
                    let inv_extent = 1.0 / extent;
                    let mut bins = [(Aabb { min: Vec3::new(0.0, 0.0, 0.0), max: Vec3::new(0.0, 0.0, 0.0) }, 0usize); SAH_BINS];
                    for b in &mut bins {
                        b.0 = Aabb::empty();
                        b.1 = 0;
                    }

                    for &idx in &indices[start..end] {
                        let c = centroids[idx];
                        let cv = match axis { 0 => c.x, 1 => c.y, _ => c.z };
                        let mut bi = ((cv - minc) * inv_extent * (SAH_BINS as f64)) as usize;
                        if bi >= SAH_BINS { bi = SAH_BINS - 1; }
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
                        indices[start..end].sort_by_key(|&idx| {
                            let c = centroids[idx];
                            let cv = match axis { 0 => c.x, 1 => c.y, _ => c.z };
                            let mut bi = ((cv - minc) * inv_extent * (SAH_BINS as f64)) as usize;
                            if bi >= SAH_BINS { bi = SAH_BINS - 1; }
                            bi
                        });
                        let mut left_total = 0usize;
                        for i in 0..best_split {
                            left_total += bins[i].1;
                        }
                        if left_total > 0 && left_total < n {
                            start + left_total
                        } else {
                            start + n / 2
                        }
                    } else {
                        start + n / 2
                    }
                } else {
                    start + n / 2
                }
            } else {
                start + n / 2
            };

            if mid == start || mid == end {
                // Fallback to median split if SAH produced a degenerate partition.
                let mid = start + n / 2;
                indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
                    let ca = centroids[a];
                    let cb = centroids[b];
                    let va = match axis { 0 => ca.x, 1 => ca.y, _ => ca.z };
                    let vb = match axis { 0 => cb.x, 1 => cb.y, _ => cb.z };
                    va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
                });
                let left = build_node(nodes, indices, bounds, centroids, start, mid);
                let right = build_node(nodes, indices, bounds, centroids, mid, end);

                let idx = node_index as usize;
                nodes[idx].left = left;
                nodes[idx].right = right;
                nodes[idx].start = 0;
                nodes[idx].count = 0;
                return node_index;
            }

            let left = build_node(nodes, indices, bounds, centroids, start, mid);
            let right = build_node(nodes, indices, bounds, centroids, mid, end);

            let idx = node_index as usize;
            nodes[idx].left = left;
            nodes[idx].right = right;
            nodes[idx].start = 0;
            nodes[idx].count = 0;
            node_index
        }

        if !indices.is_empty() {
            let end = indices.len();
            let root = build_node(&mut nodes, &mut indices, &tri_bounds, &tri_centroids, 0, end);
            debug_assert!(root == 0);
        }

        Self { nodes, indices }
    }

    /// BVH をトラバースしてレイとの最近接交差を返す。
    ///
    /// スタックベースの反復トラバーサルを使用。
    /// 子ノードの AABB 交差距離を比較し、近い方を先に処理して早期枝刈りを最大化する。
    pub fn hit(&self, tris: &[Triangle], r: Ray, tmin: f64, mut tmax: f64) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv = Vec3::new(1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z);

        let mut stack_buf = [0i32; 64];
        let mut sp = 0usize;
        stack_buf[sp] = 0;
        sp += 1;
        let mut heap_stack: Vec<i32> = Vec::new();
        let mut best: Option<Hit> = None;

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
                    if let Some(h) = tris[ti].hit(r, tmin, tmax) {
                        tmax = h.t;
                        best = Some(h);
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

        best
    }
}
