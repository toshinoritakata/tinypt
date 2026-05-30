//! Hyperion — Rust で書かれたモンテカルロ・パストレーシング・レンダラー。
//!
//! # アーキテクチャ概要
//! - **math / ray / geometry**: ベクトル演算、レイ、プリミティブ（球・三角形・AABB）
//! - **bvh**: SAH ベースの BVH（Bounding Volume Hierarchy）による高速レイ交差判定
//! - **material**: Lambert / Metal / GGX / Dielectric / Subsurface / DiffuseLight の BSDF モデル
//! - **integrator**: MIS（Multiple Importance Sampling）付きパストレーサー + NEE（Next Event Estimation）
//! - **world / scene / transform**: シーングラフ、インスタンシング、ライトサンプリング
//! - **render / task**: タイルベース・マルチスレッド・レンダリングエンジン
//! - **output / hdr / exr / aces**: PPM / HDR / EXR 出力、ACES カラースペース変換
//! - **denoise**: Intel OIDN によるモンテカルロノイズ除去
//! - **checkpoint / config / rng / constants**: チェックポイント永続化、設定、乱数生成

pub mod constants;   // レンダラー全体の定数定義
pub mod math;        // ベクトル・色・数学ユーティリティ
pub mod aces;        // sRGB ↔ ACEScg 色空間変換
pub mod rng;         // PCG 擬似乱数生成器
pub mod ray;         // レイとカメラモデル
pub mod geometry;    // 幾何プリミティブ（球・三角形・AABB）
pub mod bvh;         // BVH（Bounding Volume Hierarchy）加速構造
pub mod world;       // ワールド（全ジオメトリ・インスタンス・ライト）
pub mod transform;   // 平行移動・Y軸回転・均一スケールの変換
pub mod material;    // マテリアルと BSDF サンプリング
pub mod integrator;  // パストレーシング積分器（MIS + NEE）
pub mod task;        // タイル分割タスク定義
pub mod obj_loader;  // Wavefront OBJ ローダー
pub mod checkpoint;  // レンダリング中間状態の永続化
pub mod config;      // レンダリング設定
pub mod scene;       // シーン構築
pub mod render;      // マルチスレッド・レンダリング実行
pub mod output;      // 画像出力（PPM / HDR / EXR）
pub mod denoise;     // Intel OIDN デノイザー
pub mod hdr;         // Radiance HDR (.hdr) 読み書き
pub mod exr;         // OpenEXR (.exr) 読み書き
pub mod env;         // 環境マップの読み込みと重点的サンプリング

pub use checkpoint::ckpt_path;
pub use config::{RenderConfig, Tonemap};
pub use output::{resolve_pixels, OutputFormat, OutputSettings};
pub use render::render;
pub use scene::build_scene;
