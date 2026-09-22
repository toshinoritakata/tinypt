//! tinypt レンダラーの集中定数定義。

/// レイの最大距離（実質的に無限遠）。
pub const RAY_T_MAX: f64 = 1e30;

/// レンダラーの出力挙動のリビジョン。
///
/// 同じシーン・設定でも蓄積バッファの中身が変わる変更（積分器・BSDF・サンプリング・
/// 乱数列など）を入れたら 1 つ上げる。チェックポイントのシーンハッシュに混ぜ、
/// 挙動の異なるビルドが書いたチェックポイントから再開しないようにする。
pub const RENDER_REVISION: u32 = 15;

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

/// アルファマスク（`map_d`）の定数。
pub mod alpha {
    /// この値**未満**の alpha は透明（交差を無かったことにする）、**以上**は不透明。
    /// 0.5 はカットアウト（葉・鎖・柵）で標準的な値で、8bit マスクを 2 値化したとき
    /// 中間調の縁がどちらにも偏らない。境界は `alpha >= ALPHA_CUTOFF` で決定的（128/255 は不透明、127/255 は透明）。
    pub const ALPHA_CUTOFF: f64 = 0.5;
}

/// 法線摂動（ハイトマップ／ノーマルマップ）の定数。
pub mod normal_map {
    /// バンプの傾き（∂h/∂長さ）の上限 = tan 85°。ハイトマップの急なエッジで法線が地平線に
    /// 張り付く（cos → 0 で BSDF の項が暴れる）のを防ぐ。85° は「ほぼ水平」の手前で止める値。
    pub const SLOPE_MAX: f64 = 11.430_052_302_761_343;

    /// 摂動後の法線と幾何法線 `ng` の内積の下限。これ以下（ほぼ水平・裏返り）になる摂動は採用せず、
    /// 元のシェーディング法線に戻す（裏返すと `ns·ng > 0` が壊れ、原点ずらしが反対側へ出て自己交差する）。
    pub const NS_NG_MIN: f64 = 1e-4;

    /// MTL の `-bm`（既定 1.0）からバンプ強度への換算係数: `strength = bm · MTL_BUMP_K`。
    /// バンプはテクセル正規化（1 テクセルあたりの傾き）なので、`bm = 1` そのままでは穏やかすぎる。
    /// 値は Sponza の煉瓦・柱・アーチを K = 1 / 4 / 8 / 16 で描いて選んだ: 8 で目地と石の凹凸が明瞭で、
    /// 16 は陰影が強すぎる。Sponza の 9 枚のハイトマップは、テクセルあたりの傾きが p99 で 0.05〜0.09、
    /// 最大でも 0.17 なので、K = 16 でも傾きの上限（`SLOPE_MAX`）に達する画素は 0%。
    pub const MTL_BUMP_K: f64 = 8.0;
}
