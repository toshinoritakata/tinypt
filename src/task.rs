//! タイル分割レンダリングのタスク定義。
//!
//! 画像をタイルに分割し、各タイルをワーカースレッドで処理する。
//! `Task` はタイルの座標範囲とサンプル範囲を持ち、
//! `TileResult` はタイル内の蓄積結果を返す。

use crate::math::Color;

#[derive(Clone, Copy)]
/// レンダリングタスク（タイルの座標範囲 + サンプル範囲）。
pub struct Task {
    /// タスク ID（決定論的マージ順序に使用）
    pub id: usize,
    /// タイルの左端 X 座標
    pub x0: usize,
    /// タイルの上端 Y 座標
    pub y0: usize,
    /// タイルの右端 X 座標（排他的）
    pub x1: usize,
    /// タイルの下端 Y 座標（排他的）
    pub y1: usize,
    /// サンプル開始インデックス（含む）
    pub sample_start: usize,
    /// サンプル終了インデックス（含まない）
    pub sample_end: usize,
}

/// タイルのレンダリング結果。
pub struct TileResult {
    /// 対応するタスク ID
    pub id: usize,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    /// 各ピクセルの放射輝度合計
    pub sum: Vec<Color>,
    /// 各ピクセルのサンプル重み合計
    pub w: Vec<f64>,
}

/// (x, y) とイメージ幅 w から 1 次元インデックスを返す。
pub fn idx(x: usize, y: usize, w: usize) -> usize {
    y * w + x
}
