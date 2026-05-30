# tinypt

Rust 製のモンテカルロパストレーサー。

## 特徴

- **BVH 加速構造** (SAH) による高速レイ-ジオメトリ交差判定
- **マテリアル**: ランバート拡散・完全鏡面金属・GGX マイクロファセット・誘電体 (ガラス)・サブサーフェス散乱・面光源 ([詳細](#マテリアル))
- **Multiple Importance Sampling (MIS)** + **Next Event Estimation (NEE)** による分散低減
- **アダプティブサンプリング**: 収束判定による効率的なサンプル配分
- **Intel OIDN** を使った AI デノイズ (デフォルト有効)
- **タイル並列レンダリング** (Morton オーダー対応)
- **チェックポイント**: レンダリング途中状態の保存・再開
- **出力フォーマット**: PPM / HDR / EXR (ACEScg カラースペース)

## マテリアル

各マテリアルは BSDF として `sample`（散乱方向・スループット重み・PDF）と `eval`（NEE 用の値・PDF）を提供する。

| 種類 | パラメータ | 概要 |
|---|---|---|
| `Lambert` | `albedo` | 完全拡散反射。コサイン重み付き半球サンプリング |
| `Metal` | `albedo` | 完全鏡面反射（デルタ BSDF） |
| `Dielectric` | `ior`, `absorption` | 屈折体。フレネル + Beer-Lambert 吸収（デルタ BSDF） |
| `Ggx` | `albedo`, `alpha` | GGX マイクロファセット反射（下記参照） |
| `Subsurface` | `albedo`, `scatter_dist` | 簡易サブサーフェス散乱（指数分布の散乱距離） |
| `DiffuseLight` | `emit` | 拡散面光源 |

### GGX マイクロファセット

物理ベースの光沢反射マテリアル。`Metal` の完全鏡面と異なり、表面の微細な凹凸（マイクロファセット）による粗さを表現する。

- **法線分布関数 (NDF)**: Trowbridge-Reitz (GGX) — `D(θ_h) = α² / (π (cos²θ_h (α²−1) + 1)²)`
- **遮蔽・シャドウイング**: Smith の分離可能 G 項（`G = G₁(ω_i)·G₁(ω_o)`）
- **フレネル**: Schlick 近似（`albedo` を F₀ 反射率として使用）
- **サンプリング**: 可視法線分布 (VNDF) サンプリング（Heitz, JCGT 2018）。グレイジング角での無効サンプルを大幅に削減
- **重要度サンプリングの一貫性**: `sample` が返すスループット重み `f·cos/pdf` と MIS に使う PDF は同一の VNDF PDF に基づく

#### パラメータ

| パラメータ | 型 | 説明 |
|---|---|---|
| `albedo` | `Color` | 鏡面反射率（フレネル F₀）。金属の色味を決める |
| `alpha` | `f64` | 粗さ。`0` に近いほど鏡面、大きいほど拡散的な光沢。内部で `max(1e-3)` にクランプ |

`alpha = 0.25` 程度で、デフォルトシーンのゴールド球のような柔らかいハイライトを持つ光沢金属になる。

## ビルド

```bash
cargo build --release
```

デノイズなしでビルドする場合:

```bash
cargo build --release --no-default-features
```

## 使い方

```bash
tinypt [オプション]
```

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--spp N` | 512 | サンプル数 (samples per pixel) |
| `-o, --out PATH` | `tinypt_min.ppm` | 出力ファイル (拡張子で形式を自動判定) |
| `--env PATH` | — | HDR/EXR 環境マップ |
| `--no-env` | — | 環境マップを無効化 |
| `--denoise` / `--no-denoise` | 有効 | Intel OIDN デノイズ |
| `--adaptive` / `--no-adaptive` | 無効 | アダプティブサンプリング |
| `--adaptive-threshold N` | — | 収束閾値 (相対標準偏差) |
| `--seed N` | — | 再現性のある乱数シード |
| `--tonemap none\|aces` | `aces` | トーンマッピング |
| `--exposure N` | 0.0 | 露出補正 (EV 単位) |

### 出力例

```bash
# PPM に 1024 spp でレンダリング
tinypt --spp 1024 -o output.ppm

# EXR 出力 + 環境マップ使用
tinypt --spp 2048 -o output.exr --env sky.hdr

# デノイズなし + アダプティブサンプリング
tinypt --no-denoise --adaptive --spp 4096 -o output.hdr
```

## プロファイリング

```bash
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling
```

macOS では Instruments の Time Profiler で `target/profiling/tinypt` を指定してサンプリング。
