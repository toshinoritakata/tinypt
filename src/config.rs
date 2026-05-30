//! レンダリング設定とデフォルト値。

#[derive(Clone, Debug)]
/// レンダリングの各種パラメータ。
pub struct RenderConfig {
    /// 画像の幅（ピクセル）
    pub width: usize,
    /// 画像の高さ（ピクセル）
    pub height: usize,
    /// ピクセルあたりのサンプル数（Samples Per Pixel）
    pub spp: usize,
    /// パスの最大バウンス数（Mitsuba の integrator max_depth に対応）
    pub max_bounces: usize,
    /// Russian Roulette を開始するバウンス数（Mitsuba の rr_depth に対応）
    pub rr_start: usize,
    /// タイルサイズ（ピクセル、正方形）
    pub tile: usize,
    /// チェックポイント保存の有効/無効
    pub checkpoint_enabled: bool,
    /// N タスクごとにチェックポイントを保存
    pub checkpoint_every_tasks: usize,
    /// シーンのハッシュ値（チェックポイントの一致判定に使用）
    pub scene_hash: u64,
    /// 出力ファイルパス（拡張子で形式を自動判別）
    pub output_path: String,
    /// 環境マップのファイルパス（HDR/EXR）
    pub env_map_path: Option<String>,
    /// シーンファイルのパス（Mitsuba XML サブセット。None ならハードコードのデフォルトシーン）
    pub scene_path: Option<String>,
    /// Intel OIDN デノイズの有効/無効
    pub denoise_enabled: bool,
    /// 適応的サンプリングの有効/無効
    pub adaptive_enabled: bool,
    /// 適応的サンプリングの最小サンプル数
    pub adaptive_min_spp: usize,
    /// 適応的サンプリングの収束閾値（相対標準偏差）
    pub adaptive_threshold: f64,
    /// Morton 順序タイルソートの有効/無効
    pub morton_enabled: bool,
    /// 乱数シード（0 で追加ミキシングなし）
    pub seed: u64,
    /// トーンマッピングの種類
    pub tonemap: Tonemap,
    /// 露出補正（EV 単位、0.0 で補正なし）
    pub exposure: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// トーンマッピングの種類。
pub enum Tonemap {
    /// トーンマッピングなし（リニア出力）
    None,
    /// ACES フィルミック・トーンマッピング
    Aces,
}

impl Tonemap {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(Tonemap::None),
            "aces" => Some(Tonemap::Aces),
            _ => None,
        }
    }
}

impl RenderConfig {
    /// デフォルトのレンダリング設定を返す。
    pub fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            spp: 512,
            max_bounces: crate::constants::path::MAX_BOUNCES,
            rr_start: crate::constants::path::RR_START_BOUNCE,
            tile: 16,
            checkpoint_enabled: false,
            checkpoint_every_tasks: 128,
            scene_hash: 0x4859503000000001u64, // 'HYP0' prefix
            output_path: "hyperion_min.ppm".to_string(),
            env_map_path: None,
            scene_path: None,
            denoise_enabled: true,
            adaptive_enabled: false,
            adaptive_min_spp: 8,
            adaptive_threshold: 0.02,
            morton_enabled: true,
            seed: 0,
            tonemap: Tonemap::Aces,
            exposure: 0.0,
        }
    }
}
