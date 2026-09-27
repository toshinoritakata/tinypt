#!/usr/bin/env python3
"""髪の毛のストレステスト用 OBJ を生成する（極細の多角形チューブを大量に並べる）。

SAH の BVH（空間分割なし）にとって最悪に近い入力を作るのが目的:

- 1 本の毛は細くて長いチューブ（半径 0.002 に対して長さ約 1）。三角形は「長辺だけが長い」
  極端な細長さになり、AABB は斜めの毛ほど三角形の実体に比べて巨大になる。
- 毛は頭皮の近くで密に重なるので、根本付近に重なった AABB が集中する（BVH の走査コスト）。
- 半径はミリ以下のスケールで、自己交差回避（``p_error``）の頑健性も試される。
- ピクセルより細い毛が大量に並ぶので、エイリアシング / ノイズの挙動も見える。

生成するもの（既定の出力先 ``assets/hair/`` は ``.gitignore`` の ``*.obj`` で無視される）:

- ``hair.obj``  全ストランドを 1 メッシュにまとめた OBJ（頂点法線つき: 丸いチューブとして陰影が付く）。

頭皮（球）は OBJ にしない。``sample/hair.xml`` が ``<shape type="sphere">`` で置く
（半径 1・中心 (0, 1, 0)。このスクリプトの ``SCALP_*`` と一致させること）。

各ストランド:

- 根は頭皮の上半球（中心 (0,1,0)、半径 1、``y >= 中心`` の半球より少し上まで）に一様に散らす。
- 最初は表面の法線方向へ伸び、重力（``-y``）で曲がって垂れ下がる。ストランドごとの
  巻き（カール）とゆらぎを、ストランド固有の位相を持つ低周波の正弦で足す。
  頭の中に潜った点は球の外へ押し戻す。
- チューブは ``--sides`` 角形（既定 3）、``--segments`` 分割（既定 16）。半径は根で ``--radius``
  （既定 0.002）、先端で 30%。フレームは平行移動（回転しない）で継ぎ目のねじれを避ける。
- **先端は開いたまま**（蓋なし）: 半径が 30% あり、先端を塞ぐ三角形は細長いチューブの
  ストレスとは別の話なので足さない。根本も開いている（頭皮の球に埋まっていて見えない）。
- 決定的: ``--seed`` を固定すれば同じ OBJ になる。

使い方:

    python3 tools/gen_hair.py                          # 既定 20000 本 ≈ 1.9M 三角形
    python3 tools/gen_hair.py --preset small           # 5000 本 ≈ 480K 三角形
    python3 tools/gen_hair.py --strands 80000          # 80000 本 ≈ 7.7M 三角形
    python3 tools/gen_hair.py --preset tiny --out /tmp/hair.obj   # 動作確認用の小さいもの

三角形数 = 本数 × 分割数 × 辺数 × 2。プリセット: tiny=500 / small=5000 / default=20000 / large=80000 本。
"""

import argparse
import math
import os
import random

# 頭皮の球（sample/hair.xml の <shape type="sphere"> と一致させる）
SCALP_CENTER = (0.0, 1.0, 0.0)
SCALP_RADIUS = 1.0

PRESETS = {"tiny": 500, "small": 5000, "default": 20000, "large": 80000}


# ---------------------------------------------------------------- small vector helpers

def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _perp(t):
    """``t`` に垂直な単位ベクトル（軸のうち ``t`` から最も遠いものを使う）。"""
    ax = (1.0, 0.0, 0.0) if abs(t[0]) < 0.6 else (0.0, 1.0, 0.0)
    return _norm(_cross(t, ax))


# ---------------------------------------------------------------- strand curve

def strand_points(rng, segments, length, gravity, curl):
    """1 本のストランドの中心線（``segments + 1`` 点）。頭皮の上半球から伸びて垂れ下がる。"""
    # 上半球に一様（y は中心より少し下の 0.05 まで許す = 生え際が赤道より少し下まで届く）
    while True:
        n = _norm((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        if n[1] >= -0.05:
            break
    root = _add(SCALP_CENTER, _mul(n, SCALP_RADIUS))
    seg_len = length * (0.85 + 0.3 * rng.random()) / segments
    d = n
    p = root
    # ストランド固有のカールとゆらぎ（周波数と位相）
    fa, fb = 2.0 + 4.0 * rng.random(), 2.0 + 4.0 * rng.random()
    pa, pb = 6.283 * rng.random(), 6.283 * rng.random()
    amp = curl * (0.5 + rng.random())
    pts = [p]
    for i in range(segments):
        s = (i + 1) / segments
        u = _perp(d)
        v = _cross(d, u)
        wob = _add(_mul(u, math.sin(fa * s * 3.14159 + pa)), _mul(v, math.sin(fb * s * 3.14159 + pb)))
        # 重力は根から離れるほど効く（根本は法線方向に立ち上がる）
        d = _norm(_add(_add(d, _mul(wob, amp)), (0.0, -gravity * (0.15 + s), 0.0)))
        p = _add(p, _mul(d, seg_len))
        # 頭の中へ潜ったら球の外へ押し戻す
        off = _sub(p, SCALP_CENTER)
        r = math.sqrt(_dot(off, off))
        if r < SCALP_RADIUS + 0.004:
            p = _add(SCALP_CENTER, _mul(off, (SCALP_RADIUS + 0.004) / (r or 1.0)))
        pts.append(p)
    return pts


# ---------------------------------------------------------------- tube mesh

def add_tube(verts, norms, faces, pts, sides, r_root, r_tip_frac=0.3):
    """中心線 ``pts`` に沿った ``sides`` 角形のチューブを追加する。開いた両端（蓋なし）。"""
    segs = len(pts) - 1
    # 各点の接線（前後差分）
    tang = []
    for i in range(segs + 1):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, segs)]
        tang.append(_norm(_sub(b, a)))
    nrm = _perp(tang[0])
    base = len(verts)
    for i in range(segs + 1):
        t = tang[i]
        # 平行移動フレーム: 前の法線から接線成分を落として作る（ねじれない）
        nrm = _sub(nrm, _mul(t, _dot(nrm, t)))
        nrm = _norm(nrm) if _dot(nrm, nrm) > 1e-18 else _perp(t)
        bn = _cross(t, nrm)
        rad = r_root * (1.0 - (1.0 - r_tip_frac) * i / segs)
        for k in range(sides):
            a = 6.283185307179586 * k / sides
            ca, sa = math.cos(a), math.sin(a)
            n = (ca * nrm[0] + sa * bn[0], ca * nrm[1] + sa * bn[1], ca * nrm[2] + sa * bn[2])
            verts.append((pts[i][0] + rad * n[0], pts[i][1] + rad * n[1], pts[i][2] + rad * n[2]))
            norms.append(n)
    for i in range(segs):
        for k in range(sides):
            k1 = (k + 1) % sides
            a = base + i * sides + k
            b = base + i * sides + k1
            c = base + (i + 1) * sides + k1
            d = base + (i + 1) * sides + k
            faces.append((a, b, c))
            faces.append((a, c, d))


# ---------------------------------------------------------------- OBJ writing

def write_obj(path, verts, faces, normals):
    """頂点・頂点法線・面（``f v//vn``）を OBJ で書き出す。"""
    with open(path, "w", buffering=1 << 20) as f:
        f.write("# generated by tools/gen_hair.py -- do not commit\n")
        out = []

        def flush(force=False):
            if out and (force or len(out) >= 65536):
                f.write("\n".join(out) + "\n")
                del out[:]

        for x, y, z in verts:
            out.append("v %.6f %.6f %.6f" % (x, y, z))
            flush()
        flush(True)
        for x, y, z in normals:
            out.append("vn %.5f %.5f %.5f" % (x, y, z))
            flush()
        flush(True)
        for a, b, c in faces:
            out.append("f %d//%d %d//%d %d//%d" % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1))
            flush()
        flush(True)


def main():
    ap = argparse.ArgumentParser(description="髪の毛（極細チューブ）のストレステスト用 OBJ を生成する")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default", help="本数のプリセット（--strands が優先）")
    ap.add_argument("--strands", type=int, default=None, help="ストランド数")
    ap.add_argument("--segments", type=int, default=16, help="1 本あたりの分割数（既定 16）")
    ap.add_argument("--sides", type=int, default=3, help="チューブの辺数（既定 3）")
    ap.add_argument("--radius", type=float, default=0.002, help="根の半径（既定 0.002、先端は 30%%）")
    ap.add_argument("--length", type=float, default=0.95, help="ストランドの長さ（既定 0.95）")
    ap.add_argument("--gravity", type=float, default=0.22, help="垂れ下がりの強さ（既定 0.22）")
    ap.add_argument("--curl", type=float, default=0.10, help="カール・ゆらぎの強さ（既定 0.10）")
    ap.add_argument("--seed", type=int, default=1, help="乱数シード（既定 1、決定的）")
    ap.add_argument("--out", default=os.path.join("assets", "hair", "hair.obj"), help="出力 OBJ")
    args = ap.parse_args()
    if args.sides < 3 or args.segments < 1:
        ap.error("--sides は 3 以上、--segments は 1 以上")
    strands = args.strands if args.strands is not None else PRESETS[args.preset]

    rng = random.Random(args.seed)
    verts, norms, faces = [], [], []
    for _ in range(strands):
        pts = strand_points(rng, args.segments, args.length, args.gravity, args.curl)
        add_tube(verts, norms, faces, pts, args.sides, args.radius)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_obj(args.out, verts, faces, norms)
    print("%s: %d strands, %d vertices, %d triangles (%.1f MB)" % (
        args.out, strands, len(verts), len(faces), os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
