#!/usr/bin/env python3
"""カメラを被写体のまわりで旋回させた連番を描き、動画にまとめる。

レンダラー自体はカメラアニメーションも連番出力も持たない（モーションブラーが動かすのは
インスタンスの変換と頂点で、カメラは対象外）。そこでこのスクリプトが、フレームごとに
`<lookat>` の `origin` だけを書き換えたシーンを作って 1 枚ずつ描き、ffmpeg でつなぐ。

シーンは毎フレーム読み直されるので、読み込みの重いシーン（Sponza / Rungholt）では
その時間がフレーム数ぶん積み上がる。軽いシーン向けの道具と割り切っている。

例:
    python3 tools/render_orbit.py --scene sample/materials.xml --seconds 3 --fps 24 \
        --turns 0.3 --radius 6.4 --height 3.15 --spp 256 --res 960x540

旋回の速さは `--turns`（何周するか）と `--seconds` の比で決まる。速すぎるとフレーム間の
移動量が大きくなり、コマ送りに見えるうえカメラのモーションブラーも過剰になる。
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys

TO_WORLD = re.compile(
    r'<transform\s+name="to_world"\s*>\s*(<lookat\s+origin="[^"]*"[^>]*/>)\s*</transform>')
ORIGIN = re.compile(r'(origin=")([^"]*)(")')


def frame_scene(xml, origin_open, origin_close):
    """1 フレームぶんのシーン。`to_world` を開き時の姿勢にし、`to_world_end` を足す。

    `to_world_end` があるとレンダラーがレイごとに姿勢を補間するので、**そのフレームが掃く
    弧のぶんだけカメラにモーションブラーが乗る**。これが無いと、各フレームが完全に静止した
    絵になってコマ送りに見える。
    """
    m = TO_WORLD.search(xml)
    if m is None:
        sys.exit('シーンに <transform name="to_world"><lookat origin="..."/></transform> が見つからない')
    look = m.group(1)

    def with_origin(o):
        return ORIGIN.sub(lambda g: g.group(1) + ("%.4f, %.4f, %.4f" % o) + g.group(3), look, count=1)

    block = ('<transform name="to_world">%s</transform>\n'
             '    <transform name="to_world_end">%s</transform>'
             % (with_origin(origin_open), with_origin(origin_close)))
    return xml[:m.start()] + block + xml[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--radius", type=float, default=6.4, help="旋回半径（水平距離）")
    ap.add_argument("--height", type=float, default=3.15)
    ap.add_argument("--center", default="0,0,0", help="旋回の中心 x,y,z")
    ap.add_argument("--start-deg", type=float, default=90.0)
    ap.add_argument("--turns", type=float, default=0.3,
                    help="何周するか。1.0 = 3 秒で 360°（120°/秒）は速すぎて目が追えないので、"
                         "既定は 0.3 周（36°/秒）にしてある")
    ap.add_argument("--shutter-angle", type=float, default=0.5,
                    help="1 フレームの時間のうち露光する割合。0.5 = 180°シャッター（実写の標準）。"
                         "0 にするとカメラのモーションブラー無し")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--denoise", action="store_true", default=True)
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--bin", default="target/release/tinypt")
    ap.add_argument("--outdir", default="renders/orbit")
    ap.add_argument("--name", default=None, help="出力名（既定: シーン名）")
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.scene))[0]
    frames = max(1, int(round(args.seconds * args.fps)))
    cx, cy, cz = (float(v) for v in args.center.split(","))
    xml = open(args.scene).read()

    outdir = os.path.join(args.outdir, name)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    for i in range(frames):
        step = 2.0 * math.pi * args.turns / frames
        a0 = math.radians(args.start_deg) + step * i
        a1 = a0 + step * args.shutter_angle
        def pos(a):
            return (cx + args.radius * math.cos(a), cy + args.height, cz + args.radius * math.sin(a))
        scene_path = os.path.join(outdir, "frame_%04d.xml" % i)
        open(scene_path, "w").write(frame_scene(xml, pos(a0), pos(a1)))
        ppm = os.path.join(outdir, "frame_%04d.ppm" % i)
        cmd = [args.bin, "--scene", scene_path, "--spp", str(args.spp),
               "--res", args.res, "-o", ppm]
        if not args.denoise:
            cmd.append("--no-denoise")
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not os.path.exists(ppm):
            sys.exit("フレーム %d の描画に失敗した: %s" % (i, " ".join(cmd)))
        print("frame %d/%d" % (i + 1, frames), flush=True)

    mp4 = os.path.join(args.outdir, "%s_orbit.mp4" % name)
    gif = os.path.join(args.outdir, "%s_orbit.gif" % name)
    pat = os.path.join(outdir, "frame_%04d.ppm")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", pat, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", mp4], check=True)
    # GIF はパレットを作ってから。作らないと色が潰れる
    pal = os.path.join(outdir, "palette.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pat,
                    "-vf", "palettegen=max_colors=192", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", pat, "-i", pal,
                    "-lavfi", "scale=640:-1:flags=lanczos[s];[s][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", gif], check=True)
    print("wrote %s and %s (%d frames)" % (mp4, gif, frames))


if __name__ == "__main__":
    main()
