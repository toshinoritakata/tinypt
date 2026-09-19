#!/bin/sh
# 外部ベンチマークモデル（Crytek Sponza / Rungholt）を取得して assets/models/ に展開する。
#
# 出典: Morgan McGuire, Computer Graphics Archive, July 2017 (https://casual-effects.com/data)
#   Sponza   CC BY 3.0  (c) 2010 Frank Meinl, Crytek
#   Rungholt CC BY 3.0  (c) kescha
#
# モデルデータはリポジトリに入れない（zip だけで 128MB、展開して 361MB）。
# 展開先の assets/ は .gitignore 済み。
#
# 使い方:
#   tools/fetch_models.sh            # 両方（既に展開済みならスキップ）
#   tools/fetch_models.sh sponza     # 片方だけ
#   FORCE=1 tools/fetch_models.sh    # 既存を無視して取り直す
#
# 引数はモデル名（sponza / rungholt）か all。未知の名前は使い方を出して非ゼロ終了する。
#
# 挙動のメモ:
# - 展開済みの判定はトップレベルの OBJ の有無だけを見る。unzip を途中で中断すると
#   「OBJ はあるがテクスチャや copyright.txt が欠けている」状態をスキップしうるので、
#   中断したと分かっているときは FORCE=1 で取り直すこと。
# - 1 つ目の取得が失敗したらそこで止まる（fail-fast）。もう片方だけ欲しいときは
#   モデル名を指定して実行する。
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dest="$root/assets/models"

# name|url|zip 内のトップレベル OBJ|zip の SHA-256
# チェックサムは配布元が公開していないため、2026-09-19 に取得したファイルの実測値。
# 上流が差し替えられた場合はここが不一致になるので、その時点で中身を確認して更新すること。
models="\
sponza|https://casual-effects.com/g3d/data10/common/model/crytek_sponza/sponza.zip|sponza.obj|da005cbee0be2df2abc8513f3ceb61bcb6f69aac112babcd9c00169a27c2770c
rungholt|https://casual-effects.com/g3d/data10/research/model/rungholt/rungholt.zip|rungholt.obj|dd927901828e19e042e02448f3814e18592c9c076b39bff226bb8dc7a5ad6f4a"

want=${1:-all}

usage() {
    echo "usage: ${0##*/} [all|sponza|rungholt]" >&2
    echo "  env: FORCE=1 で既存を無視して取り直す" >&2
}

case "$want" in
    all|sponza|rungholt) ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "ERROR: unknown model '$want'" >&2
        usage
        exit 2
        ;;
esac

sha256() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else echo ""; fi
}

mkdir -p "$dest"
# パイプで while に流すとサブシェルになり、中の exit がスクリプトを止められないので
# 改行区切りの for で回す。
oldifs=$IFS
IFS='
'
for entry in $models; do
    IFS=$oldifs
    name=${entry%%|*}; rest=${entry#*|}
    url=${rest%%|*};  rest=${rest#*|}
    obj=${rest%%|*};  want_sum=${rest#*|}
    [ "$want" = all ] || [ "$want" = "$name" ] || { IFS='
'; continue; }

    if [ -f "$dest/$name/$obj" ] && [ -z "${FORCE:-}" ]; then
        echo "$name: already extracted ($dest/$name/$obj) -- skipped"
        continue
    fi

    zip="$dest/$name.zip"
    if [ ! -f "$zip" ] || [ -n "${FORCE:-}" ]; then
        echo "$name: downloading $url"
        curl -fSL --retry 3 -o "$zip.part" "$url"
        mv "$zip.part" "$zip"
    fi

    got=$(sha256 "$zip")
    if [ -z "$got" ]; then
        echo "$name: WARNING no sha256 tool found; checksum not verified"
    elif [ "$got" != "$want_sum" ]; then
        echo "$name: ERROR sha256 mismatch" >&2
        echo "  expected $want_sum" >&2
        echo "  got      $got" >&2
        echo "  zip はそのまま残してある: $zip" >&2
        echo "  取り直すなら FORCE=1 ${0##*/} ${name}（または上の zip を削除してから再実行）" >&2
        echo "  配布元が差し替えたのなら、中身を確認して tools/fetch_models.sh の値を更新すること" >&2
        exit 1
    else
        echo "$name: sha256 ok"
    fi

    echo "$name: extracting to $dest/$name"
    mkdir -p "$dest/$name"
    unzip -q -o "$zip" -d "$dest/$name"
    echo "$name: ready -- $dest/$name/$obj"
    IFS='
'
done
IFS=$oldifs

cat <<EOS

展開先: $dest
描画例:
  ./target/release/tinypt --scene sample/sponza.xml   --spp 256 --no-denoise -o renders/sponza_256spp_nodenoise.ppm
  ./target/release/tinypt --scene sample/rungholt.xml --spp 512 --no-denoise -o renders/rungholt_512spp_nodenoise.ppm
EOS
