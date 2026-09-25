#!/usr/bin/env python3
"""頂点モーション（OBJ 2 枚）のサンプル用に、小さな UV 球の「開」「閉」を生成する。

``sample/motion.xml`` の ``filename`` / ``filename_end`` が指す 2 枚で、**トポロジー（頂点数・面の添字）は同一**
（頂点モーションの必須条件）。開: 半径 0.5 の球。閉: 同じ球が横に潰れて縦に伸び、少し持ち上がる（変形するボール）。
約 400 三角形なのでリポジトリに入れてよい大きさ。

使い方:

    python3 tools/gen_motion_ball.py            # sample/motion_ball_open.obj / motion_ball_close.obj
    python3 tools/gen_motion_ball.py --outdir /tmp
"""

import argparse
import math
import os

NU, NV = 20, 10  # 経度・緯度の分割数


def sphere(radius, sx, sy, sz, dy):
    """UV 球の頂点（極を 1 頂点ずつ）を (sx, sy, sz) で伸縮して dy だけ持ち上げる。"""
    verts = [(0.0, radius * sy + dy, 0.0)]
    for j in range(1, NV):
        th = math.pi * j / NV
        for i in range(NU):
            ph = 2.0 * math.pi * i / NU
            verts.append((radius * sx * math.sin(th) * math.cos(ph),
                          radius * sy * math.cos(th) + dy,
                          radius * sz * math.sin(th) * math.sin(ph)))
    verts.append((0.0, -radius * sy + dy, 0.0))
    return verts


def faces():
    f = []
    top, bot = 1, 1 + (NV - 1) * NU + 1
    ring = lambda j, i: 2 + (j - 1) * NU + (i % NU) - 1 + 0  # 1-origin の頂点番号
    ring = lambda j, i: 1 + (j - 1) * NU + (i % NU) + 1
    for i in range(NU):
        f.append((top, ring(1, i + 1), ring(1, i)))
    for j in range(1, NV - 1):
        for i in range(NU):
            a, b, c, d = ring(j, i), ring(j, i + 1), ring(j + 1, i + 1), ring(j + 1, i)
            f.append((a, b, c))
            f.append((a, c, d))
    for i in range(NU):
        f.append((bot, ring(NV - 1, i), ring(NV - 1, i + 1)))
    return f


def write(path, verts, fs):
    with open(path, "w") as fh:
        fh.write("# tools/gen_motion_ball.py が生成（頂点モーション用の UV 球）\n")
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % v)
        for a, b, c in fs:
            fh.write("f %d %d %d\n" % (a, b, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "sample"))
    args = ap.parse_args()
    fs = faces()
    write(os.path.join(args.outdir, "motion_ball_open.obj"), sphere(0.5, 1.0, 1.0, 1.0, 0.0), fs)
    write(os.path.join(args.outdir, "motion_ball_close.obj"), sphere(0.5, 1.5, 0.6, 1.5, 0.9), fs)
    print("wrote %d vertices, %d triangles" % (len(sphere(0.5, 1, 1, 1, 0)), len(fs)))


if __name__ == "__main__":
    main()
