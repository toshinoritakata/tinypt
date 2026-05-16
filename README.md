# tinypt

Rust 製のモンテカルロパストレーサー。

## 特徴

- **BVH 加速構造** (SAH) による高速レイ-ジオメトリ交差判定
- **マテリアル**: ランバーシアン拡散・金属 (GGX マイクロファセット)・誘電体 (ガラス)・サブサーフェス散乱・面光源
- **Multiple Importance Sampling (MIS)** + **Next Event Estimation (NEE)** による分散低減
- **アダプティブサンプリング**: 収束判定による効率的なサンプル配分
- **Intel OIDN** を使った AI デノイズ (デフォルト有効)
- **タイル並列レンダリング** (Morton オーダー対応)
- **チェックポイント**: レンダリング途中状態の保存・再開
- **出力フォーマット**: PPM / HDR / EXR (ACEScg カラースペース)

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
