# tinypt

Rust 製のモンテカルロパストレーサー。

## 特徴

- **BVH 加速構造** (SAH) による高速レイ-ジオメトリ交差判定
- **マテリアル**: ランバート拡散・完全鏡面金属・GGX マイクロファセット・誘電体 (ガラス)・面光源 ([詳細](#マテリアル))
- **Multiple Importance Sampling (MIS)** + **Next Event Estimation (NEE)** による分散低減
- **Firefly クランプ**: 寄与単位・輝度ベース (閾値 50)。発光体/背景ヒットと NEE のすべての寄与に適用し、MIS の両側で同じ上限になる (バイアスあり)
- **アダプティブサンプリング**: 収束判定による効率的なサンプル配分
- **Intel OIDN** を使った AI デノイズ (デフォルト有効)
- **タイル並列レンダリング** (Morton オーダー対応)
- **チェックポイント**: レンダリング途中状態の保存・再開 ([詳細](#チェックポイント))
- **出力フォーマット**: PPM (トーンマップ後 8bit sRGB) / HDR (RGBE、リニア sRGB) / EXR (float32、リニア ACEScg)
- **シーンファイル**: Mitsuba XML サブセットの読み込み (`--scene`) ([詳細](#シーンファイル-mitsuba-xml))

## マテリアル

各マテリアルは BSDF として `sample`（散乱方向・スループット重み・PDF）と `eval`（NEE 用の値・PDF）を提供する。

| 種類 | パラメータ | 概要 |
|---|---|---|
| `Lambert` | `albedo` | 完全拡散反射。コサイン重み付き半球サンプリング |
| `Metal` | `albedo` | 完全鏡面反射（デルタ BSDF） |
| `Dielectric` | `ior`, `absorption` | 屈折体。フレネル + Beer-Lambert 吸収（デルタ BSDF） |
| `Ggx` | `albedo`, `alpha` | GGX マイクロファセット反射（下記参照） |
| `Subsurface` | `albedo` | 簡易サブサーフェス（現状は Lambert と同一の拡散反射）。シーンファイル・組み込みシーンからは指定不可 |
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

この場合もデノイズはデフォルト有効のままなので、実行ごとに OIDN 未対応の警告 (2 行) が出てデノイズはスキップされる。`--no-denoise` を付けると警告は出ない。

## 使い方

```bash
./target/release/tinypt [オプション]
# または
cargo run --release -- [オプション]
```

以下の例はリポジトリのルートで実行する前提。

未知のオプション、数値として解釈できない値、値の欠落は stderr に `Warning:` を出して無視する (終了コードは変わらない)。

組み込みシーンの解像度は常に 1920x1080 (CLI から変更不可)。解像度を変えるにはシーンファイルの `<film>` を使う。

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--scene PATH` | — | Mitsuba XML シーンファイル (未指定で組み込みシーン) |
| `--spp N` | 512 | サンプル数 (samples per pixel) |
| `-o, --out PATH` | `out.ppm` | 出力ファイル (拡張子で形式を自動判定) |
| `--env PATH` | — | HDR/EXR 環境マップ。**組み込みシーンのみ有効** (`--scene` 時は無視され XML の `<emitter>` が使われる)。読み込み失敗時は警告して空のグラデーションにフォールバック |
| `--no-env` | — | それ以前に指定した `--env` を取り消す (デフォルトも未指定＝手続き的な空) |
| `--denoise` / `--no-denoise` | 有効 | Intel OIDN デノイズ |
| `--adaptive` / `--no-adaptive` | 無効 | アダプティブサンプリング |
| `--adaptive-min-spp N` | 8 | アダプティブサンプリングの最小サンプル数 (1 未満は 1 に丸め) |
| `--adaptive-threshold N` | 0.02 | 収束閾値 (相対標準偏差、負値は 0 に丸め) |
| `--seed N` | 0 | 乱数シード。同じ設定なら出力はスレッド数に依らず決定論的 |
| `--tonemap none\|aces` | `aces` | トーンマッピング (**PPM 出力のみ**。HDR/EXR はシーン参照リニア値をそのまま保存) |
| `--exposure N` | 0.0 | 露出補正 (EV 単位、**PPM 出力のみ**) |
| `--morton` / `--no-morton` | 有効 | タイルを Morton (Z) オーダーで処理 |
| `--checkpoint` / `--no-checkpoint` | 無効 | チェックポイント保存・再開 (`checkpoint_<hash>.bin`、完了時に削除) |
| `--checkpoint-every N` | 128 | N タイルごとに保存 (指定でチェックポイントも有効化、0 は 1 に丸め) |

### チェックポイント

- ファイルは**カレントディレクトリ**に `checkpoint_<hash>.bin` として書かれ、レンダリング完了時に削除される。
- `<hash>` はシーン内容 (XML 本体と参照する OBJ / 環境マップの内容、組み込みシーンでは `--env` の内容)、解像度、spp、`max_depth`・`rr_depth`、seed、アダプティブ設定 (有効/min spp/閾値)、タイルサイズ、Morton 順序、レンダラーのバージョン (`Cargo.toml` の version と出力挙動のリビジョン定数 `RENDER_REVISION`) から導出する。これらを変えると古いチェックポイントは使われず最初からレンダーする。`--tonemap` / `--exposure` / `-o` / デノイズ設定は変えても再開できる。
- 保存は一時ファイル (`.tmp`) に書いてからリネームする。保存中に強制終了して残った同名の `.tmp` は、次回チェックポイント有効で起動したときに削除される。
- ファイル末尾にチェックサム (FNV-1a 64bit) を持ち、一致しない・フォーマットのバージョンが違う (旧形式を含む) 場合は読み込まずに最初からレンダーする。
- 保存 1 回あたり画素数 × 32 バイトを書き込む (1920x1080 で約 66 MB)。

### 出力例

```bash
# PPM に 1024 spp でレンダリング
./target/release/tinypt --spp 1024 -o output.ppm

# EXR 出力 + 環境マップ使用 (組み込みシーン)
./target/release/tinypt --spp 2048 -o output.exr --env sample/env.exr

# デノイズなし + アダプティブサンプリング
./target/release/tinypt --no-denoise --adaptive --spp 4096 -o output.hdr

# Mitsuba XML シーンを読み込んでレンダリング
./target/release/tinypt --scene sample/default.xml -o output.ppm
```

## シーンファイル (Mitsuba XML)

`--scene` で [Mitsuba レンダラー](https://www.mitsuba-renderer.org/) の XML シーン記述のサブセットを読み込める（未指定時は組み込みのデフォルトシーン）。採用理由は [`docs/adr/0002`](docs/adr/0002-mitsuba-xml-scene-format.md) を参照。

```bash
./target/release/tinypt --scene sample/default.xml -o output.ppm
```

### 対応要素

| 要素 | 対応内容 |
|---|---|
| `<sensor type="perspective">` | `fov` / `fov_axis` / `to_world`(`lookat`) / `aperture_radius` / `focus_distance` (DOF) |
| `<shape type="sphere">` | `center` / `radius` |
| `<shape type="obj">` | `filename` (XML からの相対パス) + `to_world` |
| `<shape type="rectangle"\|"cube"\|"disk">` | Mitsuba 正準形メッシュ + `to_world` |
| `<transform>` | `translate` / `rotate` (任意軸) / `scale` (均一・非均一) / `matrix` (4×4) |
| `<bsdf>` | `diffuse` / `conductor` / `roughconductor`(ggx) / `dielectric`・`thindielectric`・`roughdielectric` (いずれも `Dielectric`、独自拡張の `absorption` 対応) / `twosided`。未知の型は警告して `diffuse` にフォールバック |
| `<emitter type="area">` | `radiance` (shape に付随する面光源) |
| `<emitter type="envmap"\|"constant">` | 環境マップ (`filename` / `radiance`、`scale` 対応) |
| `<film>` / `<sampler>` / `<integrator>` | 解像度 / `sample_count` / `max_depth`・`rr_depth` |

- **色**: `<rgb>` はリニア、`<srgb>` は sRGB (ガンマ展開)。
- **CLI 優先**: `--spp` はシーンファイルの `sample_count` を上書きする (解像度・`max_depth`・`rr_depth` は CLI から変更不可)。
- **背景**: 環境 emitter が無ければ黒 (Mitsuba 準拠)。組み込みシーンの手続き的な空は使わない。
- 未対応の要素・型・属性は警告してスキップ／フォールバックする (寛容なパース)。ただし `<default>` と `<rfilter>` は**警告なし**で無視される。
- スペクトルや `<default>`/`$param` 置換、環境マップの `to_world` 回転は未対応。`$param` に依存するシーンでも警告は出ない。

### サンプル

| ファイル | 内容 |
|---|---|
| `sample/default.xml` | 組み込みデフォルトシーン相当 (地面 + 球4個 + 球光源、背景は黒) |
| `sample/mesh.xml` | OBJ メッシュ (立方体) + transform/instance |
| `sample/env_scene.xml` | 環境マップ (`env.exr`) によるライティング |
| `sample/cornell.xml` | Cornell box (rectangle/cube + 面光源)。`--tonemap none` 推奨 |

```bash
./target/release/tinypt --scene sample/mesh.xml -o mesh.ppm
./target/release/tinypt --scene sample/env_scene.xml -o env.ppm
./target/release/tinypt --scene sample/cornell.xml --tonemap none -o cornell.ppm
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
