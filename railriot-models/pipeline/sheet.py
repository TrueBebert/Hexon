"""Contact sheet of the preview renders: python -m pipeline.sheet output/<name>/<version>"""
import os
import sys

from PIL import Image, ImageDraw


def sheet(outdir):
    name = os.path.basename(os.path.dirname(os.path.abspath(outdir)))
    r = os.path.join(outdir, "renders")
    big = ["34", "profile", "back34"]
    small = ["front", "head_34", "head_front", "head_profile", "head_low"]
    W = 1600
    rows = []
    for v in big:
        p = os.path.join(r, f"{name}_{v}.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            rows.append(im.resize((W, int(im.height * W / im.width)), Image.LANCZOS))
    tiles = [Image.open(os.path.join(r, f"{name}_{v}.png")).convert("RGB") for v in small
             if os.path.exists(os.path.join(r, f"{name}_{v}.png"))]
    if tiles:
        tw = W // len(tiles)
        th = max(int(t.height * tw / t.width) for t in tiles)
        strip = Image.new("RGB", (W, th), (200, 199, 206))
        for i, t in enumerate(tiles):
            t = t.resize((tw, int(t.height * tw / t.width)), Image.LANCZOS)
            strip.paste(t, (i * tw, (th - t.height) // 2))
        rows.append(strip)
    H = sum(x.height for x in rows)
    out = Image.new("RGB", (W, H), (200, 199, 206))
    y = 0
    for x in rows:
        out.paste(x, (0, y))
        y += x.height
    path = os.path.join(outdir, f"{name}_planche.jpg")
    out.save(path, quality=90)
    return path


if __name__ == "__main__":
    print(sheet(sys.argv[1]))
