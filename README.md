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
- **シーンファイル**: Mitsuba XML サブセットの読み込み (`--scene`) ([詳細](#シーンファイル-mitsuba-xml))

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
| `--scene PATH` | — | Mitsuba XML シーンファイル (未指定で組み込みシーン) |
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

# Mitsuba XML シーンを読み込んでレンダリング
tinypt --scene sample/default.xml -o output.ppm
```

## シーンファイル (Mitsuba XML)

`--scene` で [Mitsuba レンダラー](https://www.mitsuba-renderer.org/) の XML シーン記述のサブセットを読み込める（未指定時は組み込みのデフォルトシーン）。採用理由は [`docs/adr/0002`](docs/adr/0002-mitsuba-xml-scene-format.md) を参照。

```bash
tinypt --scene sample/default.xml -o output.ppm
```

### 対応要素

| 要素 | 対応内容 |
|---|---|
| `<sensor type="perspective">` | `fov` / `fov_axis` / `to_world`(`lookat`) / `aperture_radius` / `focus_distance` (DOF) |
| `<shape type="sphere">` | `center` / `radius` |
| `<shape type="obj">` | `filename` (XML からの相対パス) + `to_world` |
| `<shape type="rectangle"\|"cube"\|"disk">` | Mitsuba 正準形メッシュ + `to_world` |
| `<transform>` | `translate` / `rotate` (任意軸) / `scale` (均一・非均一) / `matrix` (4×4) |
| `<bsdf>` | `diffuse` / `conductor` / `roughconductor`(ggx) / `dielectric` / `twosided` |
| `<emitter type="area">` | `radiance` (shape に付随する面光源) |
| `<emitter type="envmap"\|"constant">` | 環境マップ (`filename` / `radiance`、`scale` 対応) |
| `<film>` / `<sampler>` / `<integrator>` | 解像度 / `sample_count` / `max_depth`・`rr_depth` |

- **色**: `<rgb>` はリニア、`<srgb>` は sRGB (ガンマ展開)。
- **CLI 優先**: `--spp` 等の明示値はシーンファイルの設定を上書きする。
- **背景**: 環境 emitter が無ければ黒 (Mitsuba 準拠)。組み込みシーンの手続き的な空は使わない。
- 未対応の要素・型・属性は警告してスキップ／フォールバックする (寛容なパース)。
- スペクトルや `<default>`/`$param` 置換、環境マップの `to_world` 回転は未対応。

### サンプル

| ファイル | 内容 |
|---|---|
| `sample/default.xml` | 組み込みデフォルトシーン相当 (球5個 + 面光源、背景は黒) |
| `sample/mesh.xml` | OBJ メッシュ (立方体) + transform/instance |
| `sample/env_scene.xml` | 環境マップ (`env.exr`) によるライティング |
| `sample/cornell.xml` | Cornell box (rectangle/cube + 面光源)。`--tonemap none` 推奨 |

```bash
tinypt --scene sample/mesh.xml -o mesh.ppm
tinypt --scene sample/env_scene.xml -o env.ppm
tinypt --scene sample/cornell.xml --tonemap none -o cornell.ppm
```

### 記述例

```xml
<scene version="3.0.0">
  <sensor type="perspective">
    <float name="fov" value="40"/>
    <string name="fov_axis" value="y"/>
    <transform name="to_world">
      <lookat origin="0, 1.2, 4" target="0, 0.5, 0" up="0, 1, 0"/>
    </transform>
  </sensor>

  <shape type="sphere">
    <point name="center" x="0" y="0.5" z="0"/>
    <float name="radius" value="0.5"/>
    <bsdf type="roughconductor">
      <string name="distribution" value="ggx"/>
      <float name="alpha" value="0.25"/>
      <srgb name="specular_reflectance" value="0.95, 0.78, 0.35"/>
    </bsdf>
  </shape>

  <emitter type="constant"><rgb name="radiance" value="1, 1, 1"/></emitter>
</scene>
```

## プロファイリング

```bash
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling
```

macOS では Instruments の Time Profiler で `target/profiling/tinypt` を指定してサンプリング。
