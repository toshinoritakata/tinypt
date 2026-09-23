//! tinypt レンダラーの集中定数定義。

/// レイの最大距離（実質的に無限遠）。
pub const RAY_T_MAX: f64 = 1e30;

/// レンダラーの出力挙動のリビジョン。**クレートのバージョン（`Cargo.toml` の `version`、
/// `major.minor.patch`）から一意に導出する**（`major*1_000_000 + minor*1_000 + patch`）。
/// バージョンが唯一の情報源で、この定数はそれを数値にしただけ — 別に手で管理する値ではない。
///
/// 同じシーン・設定でも蓄積バッファの中身が変わる変更（積分器・BSDF・サンプリング・
/// 乱数列など）を入れたら、**`Cargo.toml` のパッチバージョンを上げる**こと。上げ忘れは
/// `render::tests::golden_*` が（`RENDER_REVISION != GOLDEN_REVISION` として）検出する。
/// チェックポイントのシーンハッシュに混ぜ、挙動の異なるビルドが書いたチェックポイントから
/// 再開しないようにする（[`crate::checkpoint::scene_hash`]）。
pub const RENDER_REVISION: u32 = revision_from_version(env!("CARGO_PKG_VERSION"));

/// `"major.minor.patch"` 形式のバージョン文字列を `major*1_000_000 + minor*1_000 + patch` に変換する。
/// `const fn` なのでコンパイル時に評価される（`RENDER_REVISION` の定義に使う）。
///
/// 数字と `.` 以外の文字（pre-release/build サフィックス `-beta.1` や `+build` など）、
/// 3 つ組でない形式、各成分の桁あふれ（minor/patch は 1000 未満、全体は `u32` に収まる範囲）は
/// **コンパイルエラー**にする（実行時に落ちるより早く気づけるように。`trybuild` 等は使わず、
/// 呼び出し元の doc コメントで「不正な形式は const 評価でコンパイルが失敗する」ことを示すに留める）。
const fn revision_from_version(v: &str) -> u32 {
    let bytes = v.as_bytes();
    let mut parts: [u64; 3] = [0, 0, 0];
    let mut part_idx: usize = 0;
    let mut has_digit = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' {
            if !has_digit {
                panic!("invalid CARGO_PKG_VERSION: empty version component");
            }
            part_idx += 1;
            if part_idx > 2 {
                panic!("invalid CARGO_PKG_VERSION: expected exactly 3 dot-separated components (major.minor.patch)");
            }
            has_digit = false;
        } else if b.is_ascii_digit() {
            has_digit = true;
            let digit = (b - b'0') as u64;
            let Some(mul) = parts[part_idx].checked_mul(10) else {
                panic!("invalid CARGO_PKG_VERSION: version component overflow");
            };
            let Some(sum) = mul.checked_add(digit) else {
                panic!("invalid CARGO_PKG_VERSION: version component overflow");
            };
            parts[part_idx] = sum;
        } else {
            panic!("invalid CARGO_PKG_VERSION: expected only digits and '.' (pre-release/build suffixes are not supported)");
        }
        i += 1;
    }
    if !has_digit || part_idx != 2 {
        panic!("invalid CARGO_PKG_VERSION: expected exactly 3 dot-separated components (major.minor.patch)");
    }
    if parts[1] >= 1000 {
        panic!("invalid CARGO_PKG_VERSION: minor version must be < 1000 (RENDER_REVISION encodes it as 3 digits)");
    }
    if parts[2] >= 1000 {
        panic!("invalid CARGO_PKG_VERSION: patch version must be < 1000 (RENDER_REVISION encodes it as 3 digits)");
    }
    let combined = parts[0] * 1_000_000 + parts[1] * 1_000 + parts[2];
    if combined > u32::MAX as u64 {
        panic!("invalid CARGO_PKG_VERSION: major version too large to fit RENDER_REVISION (u32)");
    }
    combined as u32
}

#[cfg(test)]
mod revision_tests {
    use super::revision_from_version;

    #[test]
    fn known_versions_map_to_the_documented_encoding() {
        assert_eq!(revision_from_version("0.2.0"), 2000);
        assert_eq!(revision_from_version("0.2.1"), 2001);
        assert_eq!(revision_from_version("1.0.0"), 1_000_000);
        assert_eq!(revision_from_version("0.10.3"), 10_003);
        assert_eq!(revision_from_version("12.345.678"), 12_345_678);
        assert_eq!(revision_from_version("0.0.0"), 0);
    }

    /// 現在の `Cargo.toml` バージョンが `RENDER_REVISION` の実値と一致する
    /// （`env!` の値が変わってもこのテストが追随して確かめてくれる）。
    #[test]
    fn render_revision_matches_the_crate_version() {
        assert_eq!(super::RENDER_REVISION, revision_from_version(env!("CARGO_PKG_VERSION")));
    }
}

// 不正な形式・桁あふれがコンパイルエラーになることの確認（`trybuild` は使わない）:
// これらは `const fn` の中で `panic!` に到達し、const 評価はコンパイル時に走るので、
// 以下はどれも「一時的にコメントを外すとビルドが失敗する」ことを手元で確認済み。
// - `revision_from_version("1.2")`            // 3 つ組でない
// - `revision_from_version("1.2.3.4")`        // 3 つ組でない（多すぎる）
// - `revision_from_version("1.2.x")`          // 数字以外
// - `revision_from_version("1.2.3-beta.1")`   // pre-release サフィックス
// - `revision_from_version("1.1000.0")`       // minor が桁あふれ（1000 未満でない）
// - `revision_from_version("1.0.1000")`       // patch が桁あふれ（1000 未満でない）

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
    /// BVH 構築を並列化する最小の三角形数（部分木の三角形数がこれ未満ならスレッドを起こさず
    /// 逐次構築に切り替える）。値の根拠は PERF-2 のレポート参照（Sponza 相当の数万三角形の
    /// シーンで並列化のオーバーヘッドが計測できなくなる値を実測で選んだ）。
    pub const PARALLEL_MIN_TRIS: usize = 50_000;
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
