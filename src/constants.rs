//! tinypt レンダラーの集中定数定義。

/// レイの最大距離（実質的に無限遠）。
pub const RAY_T_MAX: f64 = 1e30;

/// レンダラーの出力挙動のリビジョン。
///
/// 同じシーン・設定でも蓄積バッファの中身が変わる変更（積分器・BSDF・サンプリング・
/// 乱数列など）を入れたら 1 つ上げる。チェックポイントのシーンハッシュに混ぜ、
/// 挙動の異なるビルドが書いたチェックポイントから再開しないようにする。
pub const RENDER_REVISION: u32 = 10;

/// パストレーシングの定数。
pub mod path {
    /// 既定の最大パス長（Mitsuba の `max_depth` と同じ意味。1 = 直接見える発光体のみ、
    /// 2 = 直接照明まで、9 = 最大 8 回の散乱）。
    pub const MAX_DEPTH: usize = 9;
    /// 既定の Russian Roulette 開始パス長（Mitsuba の `rr_depth` と同じ意味。長さ `rr_depth` の
    /// パスを延長するかどうかから判定する＝散乱点のインデックス rr_depth − 1 以降）。
    pub const RR_DEPTH: usize = 4;
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
