#!/usr/bin/env python3
"""``sample/shader.xml`` 用の小さなテクスチャを手続き的に生成する（PIL 不要、numpy のみ）。

sample/textures/shader_color.png   128x128 RGB  市松（暖色 / 寒色）。画像 × ノイズの合成の「画像」側
sample/textures/shader_height.png  128x128 L    煉瓦のハイトマップ（バンプ用。目地が低い）
sample/textures/shader_normal.png  128x128 RGB  タンジェント空間ノーマルマップ（細かいドーム）。バンプの上に重ねる

使い方:  python3 tools/gen_shader_textures.py
"""

import argparse
import os
import struct
import zlib

import numpy as np


def write_png(path, arr):
    h, w = arr.shape[:2]
    ctype = 0 if arr.ndim == 2 else 2
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def color(n=128):
    y, x = np.mgrid[0:n, 0:n]
    chk = ((x // 16 + y // 16) % 2).astype(bool)
    return np.where(chk[..., None], np.array([235, 205, 160]), np.array([70, 95, 140])).astype(np.uint8)


def height(n=128):
    y, x = np.mgrid[0:n, 0:n]
    rh = n // 8
    row = y // rh
    xo = (x + (row % 2) * (n // 8)) % (n // 4)
    mortar = ((y % rh) < 2) | (xo < 2)
    return np.where(mortar, 60, 200).astype(np.uint8)


def normal(n=128, k=8):
    y, x = np.mgrid[0:n, 0:n]
    fx = ((x + 0.5) / n * k) % 1.0 - 0.5
    fy = ((y + 0.5) / n * k) % 1.0 - 0.5
    r2 = np.minimum((fx * fx + fy * fy) / 0.16, 1.0)
    h = np.sqrt(1.0 - r2) * (n / k) * 0.35
    dhdx = np.gradient(h, axis=1)
    dhdy = -np.gradient(h, axis=0)
    nx, ny, nz = -dhdx, -dhdy, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    v = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return np.clip((v * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "sample", "textures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_png(os.path.join(args.outdir, "shader_color.png"), color())
    write_png(os.path.join(args.outdir, "shader_height.png"), height())
    write_png(os.path.join(args.outdir, "shader_normal.png"), normal())
    print("wrote 3 textures")


if __name__ == "__main__":
    main()
