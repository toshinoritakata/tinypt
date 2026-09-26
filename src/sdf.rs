//! 陰関数曲面（SDF: signed-distance function）のノード木。
//!
//! `SdfTree` はプリミティブと CSG（合成）演算をフラットな配列（`Vec<SdfNode>`）で持つ。
//! **根は最後のノード**で、各ノードは自分より小さい添字のノードしか参照できない（`shader.rs` の
//! `ValueNode` アリーナと同じ非循環の規約）。これにより `eval` / `bounds` は前から順に 1 パスで
//! 評価でき、再帰も循環の心配も要らない。
//!
//! オブジェクト空間で定義する（`SdfShape` が `Transform` でワールドへ配置し、レイマーチする）。

use crate::geometry::{Aabb, Hit};
use crate::math::Vec3;
use crate::ray::Ray;
use crate::transform::{AnimatedTransform, Transform};

/// `SdfTree::nodes` への添字。
pub type SdfId = u32;

/// プリミティブ（オブジェクト空間、`center` は原点からの平行移動。カプセルのみ `center` を持たない）。
#[derive(Clone, Copy, Debug)]
pub enum SdfPrim {
    /// 球（半径 `radius`）
    Sphere { center: Vec3, radius: f64 },
    /// 角丸の直方体。`half` は角丸を含まない半辺長、`round` は丸めの半径（0 でシャープ）
    Box { center: Vec3, half: Vec3, round: f64 },
    /// トーラス（XZ 平面上、軸は Y）。`major` は中心から管の中心までの半径、`minor` は管の半径
    Torus { center: Vec3, major: f64, minor: f64 },
    /// 両端を丸めた円柱（軸は Y）。`half_height` は丸めを含まない半分の高さ、`round` は丸めの半径
    Cylinder { center: Vec3, radius: f64, half_height: f64, round: f64 },
    /// カプセル（線分 `a`-`b` からの距離が `radius` 以下）。他のプリミティブと違い `center` を持たない
    /// （線分の両端で位置を指定するので、平行移動を別に持つ意味が薄い）
    Capsule { a: Vec3, b: Vec3, radius: f64 },
}

/// CSG 演算。`SmoothUnion` / `SmoothIntersect` / `SmoothSubtract` は Inigo Quilez の多項式 smin を使う
/// （`k` は滑らかさの半径。`k = 0` は対応する非平滑演算と同じ）。
#[derive(Clone, Copy, Debug)]
pub enum SdfOp {
    Union(SdfId, SdfId),
    Intersect(SdfId, SdfId),
    /// a マイナス b（a の内側かつ b の外側）
    Subtract(SdfId, SdfId),
    SmoothUnion(SdfId, SdfId, f64),
    SmoothIntersect(SdfId, SdfId, f64),
    SmoothSubtract(SdfId, SdfId, f64),
}

/// 木のノード: プリミティブか演算のどちらか。
#[derive(Clone, Copy, Debug)]
pub enum SdfNode {
    Prim(SdfPrim),
    Op(SdfOp),
}

/// `eval` のスタック評価がヒープ確保を避けられる最大ノード数（超えたら `Vec` にフォールバック）。
const EVAL_STACK_N: usize = 32;

/// SDF ノード木。根は `nodes` の最後の要素。
#[derive(Clone, Debug, Default)]
pub struct SdfTree {
    nodes: Vec<SdfNode>,
}

/// ノードが参照する子の添字（境界チェック・非循環の検証に使う）。演算以外は空。
fn children_of(node: &SdfNode) -> [Option<SdfId>; 2] {
    match node {
        SdfNode::Prim(_) => [None, None],
        SdfNode::Op(op) => match *op {
            SdfOp::Union(a, b) | SdfOp::Intersect(a, b) | SdfOp::Subtract(a, b) => [Some(a), Some(b)],
            SdfOp::SmoothUnion(a, b, _) | SdfOp::SmoothIntersect(a, b, _) | SdfOp::SmoothSubtract(a, b, _) => [Some(a), Some(b)],
        },
    }
}

impl SdfTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// ノードを 1 つ足して添字を返す。**子は既に足した（添字が小さい）ノードだけを指せる**
    /// （`shader.rs::ShaderSet::add_value` と同じ規約）。違反したら panic — ローダー側のバグを早期に検出する
    /// （XML から不正な木は作れない設計なので、ここは「呼び出し側の契約」であり、ユーザー入力の検証ではない）。
    pub fn push(&mut self, node: SdfNode) -> SdfId {
        let n = self.nodes.len() as SdfId;
        for child in children_of(&node).into_iter().flatten() {
            assert!(child < n, "SdfTree: child {child} >= {n} (nodes must reference earlier nodes only)");
        }
        self.nodes.push(node);
        n
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 根（最後に足したノード）の添字。空の木では `None`。
    pub fn root(&self) -> Option<SdfId> {
        if self.nodes.is_empty() { None } else { Some(self.nodes.len() as SdfId - 1) }
    }

    /// オブジェクト空間の点 `p` での符号付き距離（内側が負）。ノードを添字の昇順に 1 回だけ評価する
    /// （非循環なので、各ノードを評価する時点で子はすでに埋まっている）。
    pub fn eval(&self, p: Vec3) -> f64 {
        let n = self.nodes.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let mut stack = [0.0f64; EVAL_STACK_N];
        let mut heap: Vec<f64>;
        let buf: &mut [f64] = if n <= EVAL_STACK_N {
            &mut stack[..n]
        } else {
            heap = vec![0.0; n];
            &mut heap
        };
        for i in 0..n {
            buf[i] = self.eval_node(&self.nodes[i], p, buf);
        }
        buf[n - 1]
    }

    fn eval_node(&self, node: &SdfNode, p: Vec3, buf: &[f64]) -> f64 {
        match node {
            SdfNode::Prim(prim) => eval_prim(prim, p),
            SdfNode::Op(op) => match *op {
                SdfOp::Union(a, b) => buf[a as usize].min(buf[b as usize]),
                SdfOp::Intersect(a, b) => buf[a as usize].max(buf[b as usize]),
                SdfOp::Subtract(a, b) => buf[a as usize].max(-buf[b as usize]),
                SdfOp::SmoothUnion(a, b, k) => smin(buf[a as usize], buf[b as usize], k),
                SdfOp::SmoothIntersect(a, b, k) => -smin(-buf[a as usize], -buf[b as usize], k),
                SdfOp::SmoothSubtract(a, b, k) => -smin(-buf[a as usize], buf[b as usize], k),
            },
        }
    }

    /// オブジェクト空間の保守的な境界（サーフェスは必ずこの中）。
    /// - プリミティブ: 厳密な境界
    /// - Union / SmoothUnion: 子の和集合を `k` だけ広げる（多項式 smin は最大 `k/4` 膨らむので、`k` は十分すぎる余白）
    /// - Intersect / SmoothIntersect: 子の交差を `k` だけ広げる（同じ理由。`k = 0` の通常演算は広げなくて厳密だが、
    ///   式を共通化するため常に広げる分だけ保守的にしておく）
    /// - Subtract / SmoothSubtract: 左（a）を `k` だけ広げる（結果は a の内側なので a の境界で十分だが、平滑版は
    ///   境界付近で少しだけ a の外にはみ出しうるので `k` を足す）
    pub fn bounds(&self) -> Aabb {
        let Some(root) = self.root() else { return Aabb::empty() };
        let mut memo: Vec<Aabb> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let bb = match node {
                SdfNode::Prim(prim) => prim_bounds(prim),
                SdfNode::Op(op) => match *op {
                    SdfOp::Union(a, b) | SdfOp::Intersect(a, b) => {
                        let (ba, bb) = (memo[a as usize], memo[b as usize]);
                        if matches!(op, SdfOp::Union(..)) { ba.union(bb) } else { intersect_aabb(ba, bb) }
                    }
                    SdfOp::Subtract(a, _) => memo[a as usize],
                    SdfOp::SmoothUnion(a, b, k) => grow(memo[a as usize].union(memo[b as usize]), k),
                    SdfOp::SmoothIntersect(a, b, k) => grow(intersect_aabb(memo[a as usize], memo[b as usize]), k),
                    SdfOp::SmoothSubtract(a, _, k) => grow(memo[a as usize], k),
                },
            };
            memo.push(bb);
        }
        memo[root as usize]
    }

    /// 勾配（外向き法線、正規化前）を四面体中心差分で求める（4 回評価。Inigo Quilez の手法）。
    /// `h` はステップ幅（呼び出し側がシーンスケールに合わせて決める）。
    pub fn gradient(&self, p: Vec3, h: f64) -> Vec3 {
        // 四面体の頂点方向（正 4 面体、各成分 ±1）
        let k1 = Vec3::new(1.0, -1.0, -1.0);
        let k2 = Vec3::new(-1.0, -1.0, 1.0);
        let k3 = Vec3::new(-1.0, 1.0, -1.0);
        let k4 = Vec3::new(1.0, 1.0, 1.0);
        k1 * self.eval(p + k1 * h) + k2 * self.eval(p + k2 * h) + k3 * self.eval(p + k3 * h) + k4 * self.eval(p + k4 * h)
    }

    /// 単位法線（`gradient` を正規化。退化した勾配なら +Y を返す）。
    pub fn normal(&self, p: Vec3, h: f64) -> Vec3 {
        let g = self.gradient(p, h);
        if g.len() > 0.0 { g.norm() } else { Vec3::new(0.0, 1.0, 0.0) }
    }
}

/// Union の AABB の交差（円柱・トーラスなど非直方体の保守境界どうしの交差なので、実際のサーフェスより
/// 広くなることがあるが、常にサーフェスを含む）。
fn intersect_aabb(a: Aabb, b: Aabb) -> Aabb {
    let min = Vec3::new(a.min.x.max(b.min.x), a.min.y.max(b.min.y), a.min.z.max(b.min.z));
    let max = Vec3::new(a.max.x.min(b.max.x), a.max.y.min(b.max.y), a.max.z.min(b.max.z));
    // 交差が空（min > max）でも、そのまま返す（Aabb::hit 系は min > max を「当たらない」として扱う）
    Aabb { min, max }
}

fn grow(b: Aabb, k: f64) -> Aabb {
    let k = k.abs();
    Aabb { min: b.min - Vec3::new(k, k, k), max: b.max + Vec3::new(k, k, k) }
}

/// Inigo Quilez の多項式 smooth min（k = 0 で通常の min と一致）。
fn smin(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

fn eval_prim(prim: &SdfPrim, p: Vec3) -> f64 {
    match *prim {
        SdfPrim::Sphere { center, radius } => (p - center).len() - radius,
        SdfPrim::Box { center, half, round } => {
            let q = (p - center).abs() - half;
            let outside = q.max(0.0);
            outside.len() + q.x.max(q.y.max(q.z)).min(0.0) - round
        }
        SdfPrim::Torus { center, major, minor } => {
            let d = p - center;
            let qx = Vec3::new(d.x, 0.0, d.z).len() - major;
            (qx * qx + d.y * d.y).sqrt() - minor
        }
        SdfPrim::Cylinder { center, radius, half_height, round } => {
            let d = p - center;
            let r_xz = Vec3::new(d.x, 0.0, d.z).len();
            let qx = r_xz - radius + round;
            let qy = d.y.abs() - half_height + round;
            let outside = qx.max(0.0).hypot(qy.max(0.0));
            outside + qx.max(qy).min(0.0) - round
        }
        SdfPrim::Capsule { a, b, radius } => {
            let pa = p - a;
            let ba = b - a;
            let denom = ba.dot(ba).max(1e-30);
            let t = (pa.dot(ba) / denom).clamp(0.0, 1.0);
            (pa - ba * t).len() - radius
        }
    }
}

fn prim_bounds(prim: &SdfPrim) -> Aabb {
    match *prim {
        SdfPrim::Sphere { center, radius } => {
            let r = Vec3::new(radius, radius, radius);
            Aabb { min: center - r, max: center + r }
        }
        SdfPrim::Box { center, half, round } => {
            let h = half + Vec3::new(round, round, round);
            Aabb { min: center - h, max: center + h }
        }
        SdfPrim::Torus { center, major, minor } => {
            let xz = major + minor;
            let h = Vec3::new(xz, minor, xz);
            Aabb { min: center - h, max: center + h }
        }
        SdfPrim::Cylinder { center, radius, half_height, round: _ } => {
            // round は丸めた分だけ形状を内側に切り詰めるので、丸め無し（round=0）の境界は既に保守的
            let h = Vec3::new(radius, half_height, radius);
            Aabb { min: center - h, max: center + h }
        }
        SdfPrim::Capsule { a, b, radius } => {
            let r = Vec3::new(radius, radius, radius);
            Aabb::empty().grow(a - r).grow(a + r).grow(b - r).grow(b + r)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// SdfShape: 木 + 変換 + 材質。物体空間でスフィアトレーシングする
// ---------------------------------------------------------------------------------------------

/// スフィアトレーシングの最大歩数（超えたらミス扱い）。
const MAX_STEPS: usize = 512;
/// 収束判定 `|f| < eps_obj` の、バウンディングボックス対角に対する比。
const EPS_REL: f64 = 1e-6;
/// 勾配（法線）の差分幅の、バウンディングボックス対角に対する比。
const NORMAL_H_REL: f64 = 1e-7;
/// 近似ヒットの誤差箱の大きさ（`eps_obj` の倍数）。`offset_ray_origin` が面から `eps_obj` より遠くへ
/// 押し出せるよう、収束幅より大きく取る（下の `hit` のコメント参照）。
const ERR_BOX_EPS: f64 = 4.0;
/// 収束後の t の仕上げ（割線法）の最大反復数。
const REFINE_STEPS: usize = 5;
/// 仕上げを打ち切る `|f|` の、収束幅 `eps_obj` に対する比。
const REFINE_DONE_REL: f64 = 1e-6;

/// SDF で表す 1 つの形状。ワールド空間のレイを物体空間へ写し、`|f|` の大きさだけ進める。
#[derive(Clone, Debug)]
pub struct SdfShape {
    tree: SdfTree,
    /// 物体 → ワールド変換
    xform: Transform,
    pub mat_id: usize,
    /// 物体空間の保守的な境界（収束幅ぶん広げ済み）。マーチの開始・終了区間の切り出しに使う
    obj_bounds: Aabb,
    /// ワールド空間の境界（TLAS の葉・`World::bounds` 用）。アニメーションがあればシャッター区間の掃過ボリューム
    world_bounds: Aabb,
    /// シャッター閉の変換への補間（`None` なら静止で、従来と同じ経路・同じ演算を通る）
    anim: Option<AnimatedTransform>,
    /// 収束幅（物体空間）
    eps_obj: f64,
    /// 法線の差分幅（物体空間）
    normal_h: f64,
}

impl SdfShape {
    /// 木が空、または境界が有限でないときは `None`。
    pub fn new(tree: SdfTree, xform: Transform, mat_id: usize) -> Option<Self> {
        let b = tree.bounds();
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        if tree.is_empty() || !finite(b.min) || !finite(b.max) || b.min.x > b.max.x || b.min.y > b.max.y || b.min.z > b.max.z {
            return None;
        }
        let diag = (b.max - b.min).len().max(1e-9);
        let eps_obj = EPS_REL * diag;
        let obj_bounds = grow(b, ERR_BOX_EPS * eps_obj * 2.0);
        let world_bounds = crate::world::box_world_bounds(obj_bounds, &xform);
        Some(Self { tree, xform, mat_id, obj_bounds, world_bounds, anim: None, eps_obj, normal_h: (NORMAL_H_REL * diag).max(1e-9) })
    }

    pub fn tree(&self) -> &SdfTree {
        &self.tree
    }
    pub fn transform(&self) -> &Transform {
        &self.xform
    }
    pub fn world_bounds(&self) -> Aabb {
        self.world_bounds
    }

    /// シャッター閉の変換 `end` を与え、レイの `time`（0 = 開 = `to_world`、1 = 閉）で補間するアニメーションにする
    /// （インスタンスの [`crate::world::World::set_instance_end_transform`] と同じ仕組み）。開・閉のどちらかが特異・鏡像・
    /// 非有限、または極分解が収束しないときは `false` を返し、静止のまま。`world_bounds` は
    /// [`refresh_bounds`](Self::refresh_bounds) を呼ぶまで更新されない（`World::add_sdf` が呼ぶ）。
    pub fn set_end_transform(&mut self, end: Transform) -> bool {
        match AnimatedTransform::new(self.xform, end) {
            Some(a) => {
                self.anim = Some(a);
                true
            }
            None => false,
        }
    }

    pub fn is_animated(&self) -> bool {
        self.anim.is_some()
    }

    /// 時刻 `time` の物体 → ワールド変換（静止なら `to_world` そのもの）。
    pub fn transform_at(&self, time: f64) -> Transform {
        match &self.anim {
            Some(a) => a.at(time),
            None => self.xform,
        }
    }

    /// ワールド境界を作り直す。静止なら `to_world` で写した箱、アニメーションがあればシャッター区間 `shutter` の
    /// 掃過ボリューム（物体空間の箱の中心と半対角線の球を `AnimatedTransform::swept_bounds` に渡す。
    /// `World::refresh_swept_bounds` と同じやり方）。
    pub fn refresh_bounds(&mut self, shutter: (f64, f64)) {
        self.world_bounds = match &self.anim {
            Some(a) => {
                let b = self.obj_bounds;
                let c = (b.min + b.max) * 0.5;
                a.swept_bounds(c, (b.max - c).len(), shutter.0, shutter.1)
            }
            None => crate::world::box_world_bounds(self.obj_bounds, &self.xform),
        };
    }
    pub fn eps_obj(&self) -> f64 {
        self.eps_obj
    }

    /// 最近接交差（`tmin < t < tmax`）。`prim_id` は呼び出し側（`World`）が設定する。
    ///
    /// 物体空間の `d_obj = M⁻¹d` は**正規化しない**ので、物体空間の `t` はワールドの `t` と一致する
    /// （非一様スケールでもそのまま正しい）。1 歩は `|f| / |d_obj|`（`|∇f| ≤ 1` なので、その距離だけ進んでも
    /// 面を越えない）。`|f|` で進めるので、内側から始まるレイ（誘電体の透過）も同じ式で反対側の面に届く。
    pub fn hit(&self, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        // アニメーションのある SDF だけ、レイの時刻で補間した変換を使う（静止は `self.xform` をそのまま参照する）
        let anim_xf;
        let xf: &Transform = match &self.anim {
            Some(a) => {
                anim_xf = a.at(r.time);
                &anim_xf
            }
            None => &self.xform,
        };
        let o_obj = xf.apply_point_inv(r.o);
        let d_obj = xf.apply_vec_inv(r.d);
        let d_len = d_obj.len();
        if d_len <= 0.0 || !d_len.is_finite() {
            return None;
        }
        let ray_obj = Ray { o: o_obj, d: d_obj, time: r.time };
        let (t_in, t_out) = self.obj_bounds.hit_range(ray_obj, tmin, tmax)?;
        let mut t = t_in.max(tmin);
        let t_end = t_out.min(tmax);
        let eps = self.eps_obj;
        // 直前のマーチ位置（t, f）。収束点の手前で、f は収束点と同じ符号（外側から始まれば正、内側から始まれば負）
        let mut prev: Option<(f64, f64)> = None;
        for _ in 0..MAX_STEPS {
            if t >= t_end {
                return None;
            }
            let p = o_obj + d_obj * t;
            let f = self.tree.eval(p);
            let af = f.abs();
            if af < eps {
                if t > tmin {
                    let (t, p) = match prev {
                        Some(pv) => self.refine(o_obj, d_obj, (t, f), pv, tmin, t_end),
                        None => (t, p),
                    };
                    return Some(self.make_hit(xf, t, p));
                }
                // tmin の帯の中（自己交差回避の内側）にある面は飛ばして先へ進む
                t += eps / d_len;
                prev = None;
                continue;
            }
            prev = Some((t, f));
            t += af / d_len;
        }
        None
    }

    /// 収束点 `cur = (t, f)` を、直前のマーチ位置 `prev` との割線法で面の根へ寄せる（最大 `REFINE_STEPS` 回）。
    /// 符号付きの `f` をそのまま使うので、外側から始まったレイも内側から始まったレイも同じ式でよい（どちらも
    /// `prev` と `cur` は同じ符号で、割線は面のほうへ外挿する。面を少し越えて符号が変わっても割線は続けてよい）。
    /// 採用するのは `(tmin, t_end)` の中で `|f|` が最小になった点だけ（増えたら元の点のまま）。返り値は `(t, p_obj)`。
    fn refine(&self, o_obj: Vec3, d_obj: Vec3, cur: (f64, f64), prev: (f64, f64), tmin: f64, t_end: f64) -> (f64, Vec3) {
        let (mut ta, mut fa) = prev;
        let (mut tb, mut fb) = cur;
        let mut best = cur;
        let done = self.eps_obj * REFINE_DONE_REL;
        for _ in 0..REFINE_STEPS {
            let denom = fb - fa;
            if denom == 0.0 || !denom.is_finite() {
                break;
            }
            let tn = tb - fb * (tb - ta) / denom;
            if !(tn > tmin && tn < t_end) {
                break;
            }
            let fnew = self.tree.eval(o_obj + d_obj * tn);
            if !fnew.is_finite() {
                break;
            }
            if fnew.abs() <= best.1.abs() {
                best = (tn, fnew);
            }
            if fnew.abs() < done {
                break;
            }
            (ta, fa) = (tb, fb);
            (tb, fb) = (tn, fnew);
        }
        (best.0, o_obj + d_obj * best.0)
    }

    fn make_hit(&self, xf: &Transform, t: f64, p_obj: Vec3) -> Hit {
        // 収束は面から eps_obj 以内なので、真の面は `p_obj ± eps_obj`（成分ごと）の箱に入る。余裕を見て
        // ERR_BOX_EPS·eps_obj の箱をワールドへ写す（線形部の |A| × 箱 + 通常の浮動小数点誤差）。
        // `offset_ray_origin` は `|n|·p_error` だけ法線方向へ押し出す。これは物体空間で少なくとも
        // ERR_BOX_EPS·eps_obj（= 4·eps_obj > eps_obj）の深さに相当し、反射・透過の再ヒットで
        // `|f| < eps_obj` に再び当たらない。
        let b = ERR_BOX_EPS * self.eps_obj;
        let (p, p_error) = xf.apply_point_with_error(p_obj, Vec3::new(b, b, b));
        let n = xf.apply_normal(self.tree.normal(p_obj, self.normal_h));
        Hit {
            t,
            p,
            ng: n,
            ns: n,
            mat_id: self.mat_id,
            prim_id: 0,
            inst_id: None,
            p_error,
            bary: (0.0, 0.0),
            uv: (0.0, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn sphere_eval_at_known_points() {
        let mut t = SdfTree::new();
        let s = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 2.0 }));
        assert_eq!(s, 0);
        assert!(close(t.eval(Vec3::new(0.0, 0.0, 0.0)), -2.0, 1e-12));
        assert!(close(t.eval(Vec3::new(2.0, 0.0, 0.0)), 0.0, 1e-12));
        assert!(close(t.eval(Vec3::new(5.0, 0.0, 0.0)), 3.0, 1e-12));
        // 中心をずらしても同じ形
        let mut t2 = SdfTree::new();
        t2.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(1.0, 2.0, 3.0), radius: 2.0 }));
        assert!(close(t2.eval(Vec3::new(3.0, 2.0, 3.0)), 0.0, 1e-12));
    }

    #[test]
    fn box_eval_sharp_and_rounded() {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Box { center: Vec3::new(0.0, 0.0, 0.0), half: Vec3::new(1.0, 1.0, 1.0), round: 0.0 }));
        // 面の中心
        assert!(close(t.eval(Vec3::new(1.0, 0.0, 0.0)), 0.0, 1e-12));
        assert!(close(t.eval(Vec3::new(0.0, 0.0, 0.0)), -1.0, 1e-12));
        // 角 (2,2,2) から (1,1,1) の角までの距離は sqrt(3)
        assert!(close(t.eval(Vec3::new(2.0, 2.0, 2.0)), 3f64.sqrt(), 1e-9));

        let mut tr = SdfTree::new();
        tr.push(SdfNode::Prim(SdfPrim::Box { center: Vec3::new(0.0, 0.0, 0.0), half: Vec3::new(1.0, 1.0, 1.0), round: 0.2 }));
        // 面までの距離は round 分だけ縮む
        assert!(close(tr.eval(Vec3::new(1.2, 0.0, 0.0)), 0.0, 1e-9));
    }

    #[test]
    fn torus_eval() {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Torus { center: Vec3::new(0.0, 0.0, 0.0), major: 2.0, minor: 0.5 }));
        // 管の中心円上（XZ 平面、半径 2）は表面から minor 分だけ内側
        assert!(close(t.eval(Vec3::new(2.0, 0.0, 0.0)), -0.5, 1e-12));
        // 表面ちょうど
        assert!(close(t.eval(Vec3::new(2.5, 0.0, 0.0)), 0.0, 1e-12));
        // 管の断面の「上端」: XZ 距離が major、y が minor
        assert!(close(t.eval(Vec3::new(2.0, 0.5, 0.0)), 0.0, 1e-9));
    }

    #[test]
    fn cylinder_eval_capped() {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Cylinder { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0, half_height: 2.0, round: 0.0 }));
        assert!(close(t.eval(Vec3::new(0.0, 0.0, 0.0)), -1.0, 1e-12));
        assert!(close(t.eval(Vec3::new(1.0, 0.0, 0.0)), 0.0, 1e-12));
        assert!(close(t.eval(Vec3::new(0.0, 2.0, 0.0)), 0.0, 1e-12));
        assert!(close(t.eval(Vec3::new(0.0, 3.0, 0.0)), 1.0, 1e-12));
    }

    #[test]
    fn capsule_eval() {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Capsule { a: Vec3::new(0.0, 0.0, 0.0), b: Vec3::new(0.0, 2.0, 0.0), radius: 0.5 }));
        assert!(close(t.eval(Vec3::new(0.0, 1.0, 0.0)), -0.5, 1e-12)); // 中間軸上
        assert!(close(t.eval(Vec3::new(0.5, 1.0, 0.0)), 0.0, 1e-12)); // 側面
        assert!(close(t.eval(Vec3::new(0.0, -0.5, 0.0)), 0.0, 1e-12)); // 端の球キャップ
        assert!(close(t.eval(Vec3::new(0.0, 2.5, 0.0)), 0.0, 1e-12));
    }

    #[test]
    fn csg_union_intersect_subtract() {
        let p = Vec3::new(0.0, 0.0, 0.0); // 両方の球の内側（重なりの中心）
        // 各演算を根にした木をそれぞれ作って確認する
        let mk = |root_op: fn(SdfId, SdfId) -> SdfOp| {
            let mut t = SdfTree::new();
            let a = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(-0.5, 0.0, 0.0), radius: 1.0 }));
            let b = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(0.5, 0.0, 0.0), radius: 1.0 }));
            t.push(SdfNode::Op(root_op(a, b)));
            t
        };
        let tu = mk(SdfOp::Union);
        let ti = mk(SdfOp::Intersect);
        let ts = mk(SdfOp::Subtract);
        assert!(tu.eval(p) < 0.0, "union: center is inside both");
        assert!(ti.eval(p) < 0.0, "intersect: center is inside both");
        assert!(ts.eval(p) > 0.0, "a - b: center is inside b, so outside a-b");
        assert!(ts.eval(Vec3::new(-1.0, 0.0, 0.0)) < 0.0, "a - b: far side of a, outside b");
    }

    #[test]
    fn smooth_ops_match_sharp_at_k_zero_and_blend_with_k() {
        let mk = |k: f64| {
            let mut t = SdfTree::new();
            let a = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(-1.0, 0.0, 0.0), radius: 1.0 }));
            let b = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 1.0 }));
            t.push(SdfNode::Op(SdfOp::SmoothUnion(a, b, k)));
            t
        };
        let sharp = mk(0.0);
        let smooth = mk(0.6);
        // 遠く離れた点では k=0 とほぼ同じ形（smin の影響が減衰する場所）
        let far = Vec3::new(-1.0, 0.0, 0.0);
        assert!(close(sharp.eval(far), smooth.eval(far), 1e-6));
        // 2 球の中間（両方から等距離）では、平滑化のほうがより内側（負に大きい）
        let mid = Vec3::new(0.0, 0.0, 0.0);
        assert!(smooth.eval(mid) < sharp.eval(mid) - 1e-6);
    }

    #[test]
    fn bounds_contain_sampled_surface_points() {
        let mut t = SdfTree::new();
        let a = t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }));
        let b = t.push(SdfNode::Prim(SdfPrim::Torus { center: Vec3::new(0.5, 0.0, 0.0), major: 1.0, minor: 0.2 }));
        t.push(SdfNode::Op(SdfOp::SmoothUnion(a, b, 0.3)));
        let bb = t.bounds();
        // 球面上・球の内部・原点近傍の粗いサンプルで、境界の中にあること、かつ表面付近では |f| が小さいこと
        let mut rng = 12345u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for _ in 0..2000 {
            let p = Vec3::new(next(), next(), next()) * 3.0;
            let f = t.eval(p);
            if f.abs() < 0.05 {
                let inside_bb = p.x >= bb.min.x && p.x <= bb.max.x && p.y >= bb.min.y && p.y <= bb.max.y && p.z >= bb.min.z && p.z <= bb.max.z;
                assert!(inside_bb, "surface point {:?} (f={f}) outside bounds {:?}", p, bb);
            }
        }
    }

    #[test]
    fn normal_points_outward_on_sphere() {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }));
        for p in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)] {
            let n = t.normal(p, 1e-4);
            let want = p.norm();
            assert!((n - want).len() < 1e-3, "{:?} vs {:?}", n, want);
        }
    }

    #[test]
    fn push_panics_on_forward_reference() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut t2 = SdfTree::new();
            t2.push(SdfNode::Prim(SdfPrim::Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }));
            t2.push(SdfNode::Op(SdfOp::Union(0, 5))); // 5 はまだ存在しない添字
        }));
        assert!(result.is_err(), "forward reference must panic");
    }

    // ---- SdfShape（スフィアトレーシング） ----

    use crate::geometry::{offset_ray_origin, Sphere};
    use crate::transform::Transform;
    use crate::world::World;

    fn sphere_tree(center: Vec3, radius: f64) -> SdfTree {
        let mut t = SdfTree::new();
        t.push(SdfNode::Prim(SdfPrim::Sphere { center, radius }));
        t
    }

    fn ray(o: Vec3, d: Vec3) -> Ray {
        Ray { o, d: d.norm(), time: 0.0 }
    }

    #[test]
    fn sdf_sphere_matches_analytic_sphere_hit() {
        let c = Vec3::new(0.3, -0.2, 0.1);
        let shape = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.3), Transform::translate(c), 5).unwrap();
        let sph = Sphere { c, r: 1.3, mat_id: 5 };
        let mut n = 0;
        for i in 0..200 {
            let a = i as f64 * 0.37;
            let o = Vec3::new(4.0 * a.cos(), 1.5 * (a * 1.3).sin(), 4.0 * a.sin()) + c;
            let target = c + Vec3::new((a * 2.1).sin(), (a * 0.7).cos(), (a * 1.9).sin()) * 0.9;
            let r = ray(o, target - o);
            let (Some(h), Some(e)) = (shape.hit(r, 0.0, 1e9), sph.hit(r, 0.0, 1e9)) else { continue };
            n += 1;
            // 収束後に割線法で仕上げるので、収束幅 eps_obj（1e-6 × bbox 対角）よりずっと精密に一致する
            assert!((h.t - e.t).abs() <= 1e-9 * e.t, "t {} vs {}", h.t, e.t);
            assert!((h.ng - e.ng).len() < 2e-6, "normal {:?} vs {:?}", h.ng, e.ng);
            assert_eq!(h.mat_id, 5);
            assert_eq!(h.inst_id, None);
        }
        assert!(n > 100, "too few hits: {n}");
    }

    #[test]
    fn non_uniform_scale_hit_is_the_first_root_of_the_scaled_surface() {
        let xf = Transform::scale(Vec3::new(2.0, 0.5, 1.0)).compose(Transform::translate(Vec3::new(0.0, 0.0, 0.0)));
        let shape = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), xf, 0).unwrap();
        for i in 0..100 {
            let a = i as f64 * 0.61;
            let o = Vec3::new(6.0 * a.cos(), 3.0 * (a * 0.8).sin(), 6.0 * a.sin());
            let r = ray(o, Vec3::new(0.1 * a.sin(), 0.05, 0.0) - o);
            let Some(h) = shape.hit(r, 0.0, 1e9) else { continue };
            // 面上（楕円体 (x/2)² + (y/0.5)² + z² = 1）にあり、それより手前に面が無い
            let q = |p: Vec3| (p.x / 2.0).powi(2) + (p.y / 0.5).powi(2) + p.z * p.z - 1.0;
            assert!(q(h.p).abs() < 1e-8, "not on surface: {}", q(h.p));
            // 物体空間の |f| は収束幅 eps_obj よりずっと小さい
            let f_obj = shape.tree().eval(xf.apply_point_inv(h.p));
            assert!(f_obj.abs() < 1e-3 * shape.eps_obj(), "|f| = {} vs eps_obj {}", f_obj.abs(), shape.eps_obj());
            let mut t = 0.0;
            while t < h.t - 1e-3 {
                assert!(q(r.at(t)) > 0.0, "surface crossed before the reported hit at t={t}");
                t += 0.01;
            }
            // 法線は楕円体の勾配（逆転置）と同じ向き
            let g = Vec3::new(h.p.x / 4.0, h.p.y / 0.25, h.p.z).norm();
            assert!((h.ng - g).len() < 2e-6, "{:?} vs {:?}", h.ng, g);
        }
    }

    #[test]
    fn ray_starting_inside_exits_on_the_far_side_with_outward_normal() {
        let shape = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), Transform::identity(), 0).unwrap();
        let r = ray(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let h = shape.hit(r, 0.0, 1e9).expect("exit hit");
        assert!((h.t - 1.0).abs() < 1e-5, "t = {}", h.t);
        assert!((h.ng - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-4, "normal must be outward: {:?}", h.ng);
    }

    /// 反射レイは自分の面に当たり直さない。SDF 球の内側へ透過したレイは t≈0 でなく反対側で出る。
    #[test]
    fn offset_origin_avoids_self_hit_for_reflection_and_transmission() {
        let xfs = [Transform::identity(), Transform::scale(Vec3::new(3.0, 0.4, 1.0)), Transform::new(Vec3::new(5.0, 1.0, -2.0), 40.0, 0.01)];
        for xf in xfs {
            let shape = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), xf, 0).unwrap();
            for i in 0..100 {
                let a = i as f64 * 0.53;
                let center = xf.apply_point(Vec3::new(0.0, 0.0, 0.0));
                let far = xf.apply_vec(Vec3::new(1.0, 1.0, 1.0)).len() * 4.0;
                let o = center + Vec3::new(a.cos() * far, (a * 0.7).sin() * far * 0.5, a.sin() * far);
                let r = ray(o, center - o);
                let Some(h) = shape.hit(r, 0.0, 1e9) else { continue };
                // 反射: 面の外側へ出したレイ
                let refl = h.ng * 2.0 * (-r.d).dot(h.ng) + r.d;
                let ro = offset_ray_origin(h.p, h.p_error, h.ng, refl);
                assert!(shape.hit(Ray { o: ro, d: refl.norm(), time: 0.0 }, 0.0, 1e9).is_none(), "reflected ray re-hit its own surface (xf case, i={i})");
                // 透過: 面の内側へ出したレイは反対側で出る（t が 0 に近い自己ヒットではない）
                let ro = offset_ray_origin(h.p, h.p_error, h.ng, r.d);
                let h2 = shape.hit(Ray { o: ro, d: r.d, time: 0.0 }, 0.0, 1e9).expect("transmitted ray must exit");
                let chord = (h2.p - h.p).len();
                assert!(chord > 1e-3 * far * 0.0 + 1e-6 && (h2.p - h.p).dot(r.d) > 0.0, "transmitted ray self-hit: chord {chord}");
                assert!(h2.ng.dot(r.d) > 0.0, "exit normal must point along the ray");
            }
        }
    }

    #[test]
    fn world_assigns_prim_id_after_spheres_and_agrees_between_linear_and_tlas() {
        let mut w = World::new();
        w.add_sphere(Sphere { c: Vec3::new(-4.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        let i = w.add_sdf(SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), Transform::translate(Vec3::new(4.0, 0.0, 0.0)), 3).unwrap());
        assert_eq!(i, 0);
        assert_eq!(w.sdfs().len(), 1);
        let r = ray(Vec3::new(4.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
        let h = w.hit(r, 0.0, 1e9).expect("sdf hit");
        assert_eq!((h.inst_id, h.prim_id, h.mat_id), (None, 1, 3)); // 球 1 個の後
        assert!((h.t - 9.0).abs() < 1e-5);
        assert!(w.occluded(r, 0.0, 1e9, None));
        assert!(!w.occluded(r, 0.0, 5.0, None));
        // 球のヒットは従来どおり
        let rs = ray(Vec3::new(-4.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(w.hit(rs, 0.0, 1e9).map(|h| (h.inst_id, h.prim_id)), Some((None, 0)));
        // TLAS を使う数まで球を足しても同じ結果
        for k in 0..25 {
            w.add_sphere(Sphere { c: Vec3::new(20.0 + 3.0 * k as f64, 0.0, 0.0), r: 1.0, mat_id: 0 });
        }
        let h2 = w.hit(r, 0.0, 1e9).expect("sdf hit with tlas");
        assert_eq!((h2.inst_id, h2.prim_id, h2.mat_id), (None, 26, 3));
        assert_eq!(h2.t, h.t);
        assert!(w.occluded(r, 0.0, 1e9, None));
        assert!(!w.occluded(r, 0.0, 5.0, None));
    }

    #[test]
    fn empty_tree_or_non_finite_bounds_make_no_shape() {
        assert!(SdfShape::new(SdfTree::new(), Transform::identity(), 0).is_none());
        assert!(SdfShape::new(sphere_tree(Vec3::new(f64::NAN, 0.0, 0.0), 1.0), Transform::identity(), 0).is_none());
    }

    // ---- モーションブラー ----

    fn moving_sphere(end_x: f64) -> SdfShape {
        let mut s = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), Transform::identity(), 2).unwrap();
        assert!(s.set_end_transform(Transform::translate(Vec3::new(end_x, 0.0, 0.0))));
        s
    }

    #[test]
    fn moving_sdf_sphere_hits_the_interpolated_position() {
        let shape = moving_sphere(4.0);
        assert!(shape.is_animated());
        for time in [0.0, 0.25, 0.5, 1.0] {
            let cx = 4.0 * time;
            let r = Ray { o: Vec3::new(cx, 0.0, 10.0), d: Vec3::new(0.0, 0.0, -1.0), time };
            let h = shape.hit(r, 0.0, 1e9).unwrap_or_else(|| panic!("no hit at time {time}"));
            assert!((h.t - 9.0).abs() < 1e-5, "time {time}: t = {}", h.t);
            // 別の時刻の位置には当たらない（球の外）
            let far = Ray { o: Vec3::new(cx + 3.0, 0.0, 10.0), d: Vec3::new(0.0, 0.0, -1.0), time };
            assert!(shape.hit(far, 0.0, 1e9).is_none(), "time {time}");
        }
        // 静止形は time を見ない
        let still = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), Transform::identity(), 0).unwrap();
        assert!(!still.is_animated());
        let r = Ray { o: Vec3::new(0.0, 0.0, 10.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.9 };
        assert!(still.hit(r, 0.0, 1e9).is_some());
    }

    #[test]
    fn swept_bounds_contain_the_shape_at_every_time_for_either_call_order() {
        let contains = |b: Aabb, p: Vec3| (0..3).all(|k| [p.x, p.y, p.z][k] >= [b.min.x, b.min.y, b.min.z][k] && [p.x, p.y, p.z][k] <= [b.max.x, b.max.y, b.max.z][k]);
        for shutter_first in [true, false] {
            let mut w = World::new();
            if shutter_first {
                w.set_shutter(0.0, 1.0);
            }
            let shape = {
                let mut s = SdfShape::new(sphere_tree(Vec3::new(0.0, 0.0, 0.0), 1.0), Transform::identity(), 0).unwrap();
                // 動きは平行移動 + 回転
                assert!(s.set_end_transform(Transform::translate(Vec3::new(5.0, 1.0, 0.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 90.0))));
                s
            };
            w.add_sdf(shape);
            if !shutter_first {
                w.set_shutter(0.0, 1.0);
            }
            let b = w.sdfs()[0].world_bounds();
            for k in 0..=20 {
                let time = k as f64 / 20.0;
                let xf = w.sdfs()[0].transform_at(time);
                for d in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)] {
                    assert!(contains(b, xf.apply_point(d)), "shutter_first={shutter_first} time={time}: {:?} outside {:?}", xf.apply_point(d), b);
                }
            }
            assert!(contains(w.bounds(), Vec3::new(5.0, 1.0, 0.0)), "World::bounds covers the end position");
        }
        // シャッターを狭めると境界も縮む（作り直されている）
        let mut w = World::new();
        w.add_sdf(moving_sphere(10.0));
        let wide = w.sdfs()[0].world_bounds().max.x;
        w.set_shutter(0.0, 0.2);
        assert!(w.sdfs()[0].world_bounds().max.x < wide - 5.0);
    }

    #[test]
    fn moving_sdf_agrees_between_linear_and_tlas() {
        let mut w = World::new();
        w.add_sdf(moving_sphere(6.0));
        let rays: Vec<Ray> = (0..40)
            .map(|i| {
                let time = (i % 5) as f64 / 4.0;
                Ray { o: Vec3::new(6.0 * time + 0.3 * (i as f64 * 0.7).sin(), 0.4 * (i as f64).cos(), 8.0), d: Vec3::new(0.0, 0.0, -1.0), time }
            })
            .collect();
        let lin: Vec<_> = rays.iter().map(|&r| (w.hit(r, 0.0, 1e9).map(|h| h.t), w.occluded(r, 0.0, 1e9, None))).collect();
        for k in 0..25 {
            w.add_sphere(Sphere { c: Vec3::new(30.0 + 3.0 * k as f64, 0.0, 0.0), r: 1.0, mat_id: 0 });
        }
        let tlas: Vec<_> = rays.iter().map(|&r| (w.hit(r, 0.0, 1e9).map(|h| h.t), w.occluded(r, 0.0, 1e9, None))).collect();
        assert_eq!(lin, tlas);
        assert!(lin.iter().any(|(h, _)| h.is_some()));
    }

    #[test]
    fn object_space_point_follows_the_hit_time() {
        let mut w = World::new();
        w.add_sdf(moving_sphere(4.0));
        let r = Ray { o: Vec3::new(2.0, 0.0, 10.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.5 };
        let h = w.hit(r, 0.0, 1e9).unwrap();
        let p = w.object_space_point(&h, 0.5);
        assert!((p - Vec3::new(0.0, 0.0, 1.0)).len() < 1e-4, "{p:?}"); // 中心 x=2 の球の上端 → 物体空間 (0, 0, 1)
    }
}
