//! Hyperion レンダラーの集中定数定義。

/// レイ生成時の微小オフセット（自己交差回避用）。
pub const RAY_EPSILON: f64 = 1e-4;

/// レイの最大距離（実質的に無限遠）。
pub const RAY_T_MAX: f64 = 1e30;

/// パストレーシングの定数。
pub mod path {
    /// パスあたりの最大バウンス回数。
    pub const MAX_BOUNCES: usize = 8;
    /// Russian Roulette を開始するバウンス数。
    pub const RR_START_BOUNCE: usize = 3;
    /// ファイアフライ抑制のための輝度クランプ閾値。
    pub const FIREFLY_CLAMP: f64 = 50.0;
}

/// BVH 構築の定数。
pub mod bvh {
    /// SAH（Surface Area Heuristic）評価のビン数。
    pub const SAH_BINS: usize = 8;
    /// リーフノードの最大プリミティブ数。
    pub const LEAF_SIZE: usize = 4;
}

/// UI / 進捗表示の定数。
pub mod ui {
    /// 進捗更新の間隔（ミリ秒）。
    pub const PROGRESS_INTERVAL_MS: u128 = 250;
}
