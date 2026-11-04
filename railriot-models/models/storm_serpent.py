"""PRIMAL - Storm serpent wagon (ref: blue lightning sea serpent on a wooden wagon).

Size target (class Mosa / Parasaur): ~44 x 13 x 29 studs.  Small cube 0.16, big cube 0.48.
X = front of the train (head side), Y = width, Z = up, wheels on Z = 0.
"""
import math

import numpy as np

from pipeline.vox import (VoxelModel, Box, BoxC, RBox, Ellipsoid, Sphere, Cylinder, RoundCone,
                          Sweep, bezier, Plate, Scale, sym, rot)
from pipeline.texture import hash01
from pipeline.fx import FX as _FX

NAME = "storm_serpent"
B, S = 0.48, 0.16

WX = 44 * B   # 21.12 half length of the wagon box
WY = 11 * B   # 5.28  half width
Z0 = 5 * B    # 2.40  bottom of the box
ZD = 14 * B   # 6.72  deck (inside the rim)
ZR = 15 * B   # 7.20  top of the rim beam
RIM = 2 * B

FX = _FX
PALETTE = {
    # wagon
    "wood": dict(rgb=(96, 58, 42), fx="wood"),
    "wood_dark": dict(rgb=(64, 39, 31), fx="wood"),
    "rope": dict(rgb=(126, 36, 34)),
    "moss": dict(rgb=(96, 112, 38), jitter=0.10),
    "ivory": dict(rgb=(214, 200, 164)),
    "ivory_dark": dict(rgb=(176, 156, 122)),
    "stone": dict(rgb=(62, 62, 88)),
    "stone_dark": dict(rgb=(40, 40, 60)),
    "wheel": dict(rgb=(44, 44, 64)),
    "wheel_rim": dict(rgb=(78, 78, 104)),
    "iron": dict(rgb=(42, 38, 46)),
    "banner": dict(rgb=(40, 56, 150)),
    "banner_trim": dict(rgb=(146, 166, 230)),
    "emblem": dict(rgb=(228, 228, 238)),
    "rune": dict(rgb=(108, 214, 255), glow=True, studs=False),
    # serpent
    "navy": dict(rgb=(58, 64, 150), fx="scales2"),
    "navy_light": dict(rgb=(82, 94, 186), fx="scales2"),
    "ice_dark": dict(rgb=(64, 124, 222), glow=True, studs=False),
    "ice": dict(rgb=(104, 180, 252), glow=True, studs=False),
    "ice_light": dict(rgb=(176, 224, 255), glow=True, studs=False),
    "belly": dict(rgb=(72, 112, 206)),
    "cloud": dict(rgb=(26, 26, 54)),
    "cloud_light": dict(rgb=(46, 48, 98)),
    "cloud_line": dict(rgb=(86, 94, 172)),
    "teal": dict(rgb=(52, 198, 212)),
    "teal_dark": dict(rgb=(16, 92, 124)),
    "mouth": dict(rgb=(22, 14, 30), studs=False, jitter=0.0),
    "tooth": dict(rgb=(232, 226, 206), studs=False, jitter=0.0),
    "socket": dict(rgb=(14, 12, 24), studs=False, jitter=0.0),
}

PARTS = [
    dict(name="Wagon", parent=None, pivot=(0, 0, 0), group="static", outline=0.16, mirror_x=True),
    dict(name="Body", parent="Wagon", pivot=(0, 0, ZD), group="static", outline=0.2),
    dict(name="Crystals", parent="Body", pivot=(-2.8, 0, 14.0), group="static", outline=0.0, emissive=True, flat=True),
    dict(name="Tail1", parent="Body", pivot=(-14.4, 0, 7.9), outline=0.2),
    dict(name="Tail2", parent="Tail1", pivot=(-17.76, 0, 7.7), outline=0.2),
    dict(name="Neck1", parent="Body", pivot=(10.0, 0, 10.08), outline=0.2),
    dict(name="Neck2", parent="Neck1", pivot=(10.5, 0, 17.28), outline=0.2),
    dict(name="Head", parent="Neck2", pivot=(14.4, 0, 25.6), outline=0.2),
    dict(name="Jaw", parent="Head", pivot=(16.32, 0, 25.92), outline=0.2),
]

NEON = []
MIN_KEEP = 30
LIGHTS = [  # a few PointLights suggested for the game (not geometry)
    dict(name="CrystalGlow", part="Body", position=(-2.8, 0, 16.0), color=(120, 205, 255), range=16, brightness=2),
    dict(name="AlcoveGlowL", part="Wagon", position=(0, 4.9, 4.6), color=(120, 205, 255), range=7, brightness=1),
    dict(name="AlcoveGlowR", part="Wagon", position=(0, -4.9, 4.6), color=(120, 205, 255), range=7, brightness=1),
]
DROP_DOWN = dict(parts=["Wagon"], below=2.5)
# no outline hull inside the eye sockets and in front of them (it would hide the eyes)
NO_OUTLINE = [sym(Box((17.1, 1.12, 26.88), (18.25, 2.3, 27.52))), sym(Box((18.24, 1.12, 26.88), (22.6, 2.3, 27.52)))]
ATTACHMENTS = []
RENDER = dict(head=dict(center=(17.8, 0, 26.9), radius=4.4), rail_gauge=8.64)

C_CRYSTAL = (118, 204, 255)
C_CRYSTAL_TIP = (196, 238, 255)
C_EYE = (150, 246, 255)
C_STRIPE = (78, 206, 255)
C_WHEEL = (58, 138, 255)
C_LANTERN = (255, 166, 58)


def neon(name, part, center, size, R=None, color=C_CRYSTAL, shape="Block"):
    NEON.append(dict(name=name, part=part, center=tuple(float(c) for c in center), size=tuple(size),
                     R=(np.eye(3) if R is None else np.asarray(R)).tolist(), color=color, shape=shape))


def spiral_where(cx, cz, p, w, rmin, rmax, plane="xz", cy=0.0):
    """Archimedean spiral groove (pitch p, width w) in the XZ (or YZ) plane."""
    def fn(X, Y, Z):
        A = X - cx if plane == "xz" else Y - cy
        r = np.sqrt(A * A + (Z - cz) ** 2)
        th = np.arctan2(Z - cz, A)
        s = np.mod(r / p - th / (2 * np.pi), 1.0)
        return (s < w / p) & (r >= rmin) & (r <= rmax)
    return fn


# --------------------------------------------------------------------------------- wagon
def tusk(vm, base, ctrl, tip, r0, r1, part="Wagon", n=28, rope=True):
    """Square-section tusk along a planar curve, in big cubes (blocky, few triangles)."""
    pts = bezier(base, ctrl, tip, n=n)
    t = np.linspace(0, 1, n)
    rr = r0 * (1 - t) ** 0.85 + r1 * t
    for p, r in zip(pts, rr):
        vm.add(Box(p - r, p + r), "ivory", part, big=True)
    for f in (0.34, 0.6):
        i = int(round(f * (n - 1)))
        c, r = pts[i], rr[i] + 0.1
        vm.paint(Box(c - r, c + r), "ivory_dark", only=["ivory"], parts=[part])
    if rope:
        c, r = pts[3], rr[3] + 0.12
        vm.paint(Box(c - r, c + r), "rope", only=["ivory"], parts=[part])
    return pts


def lantern_side(vm, x, side, name):
    """Lantern hanging from the rim beam on a long side: cap, base and a neon core."""
    W = "Wagon"
    y0 = side * WY
    o = lambda a, b: sorted((y0 + side * a, y0 + side * b))
    ya, yb = o(0, 0.64)
    vm.add(Box((x - 0.08, ya, 6.08), (x + 0.08, yb, 6.56)), "iron", W, big=False)       # hook
    ya, yb = o(0.0, 1.12)
    vm.add(Box((x - 0.56, ya, 5.92), (x + 0.56, yb, 6.08)), "iron", W, big=False)       # cap
    vm.add(Box((x - 0.56, ya, 4.32), (x + 0.56, yb, 4.48)), "iron", W, big=False)       # base
    neon(name, W, (x, y0 + side * 0.56, 5.20), (0.88, 0.88, 1.40), color=C_LANTERN)


def lantern_end(vm, y, end, name):
    """Lantern on an end face (end = +1 front / -1 back)."""
    W = "Wagon"
    x0 = end * WX  # hangs on the end face beside the plaque (keeps the length within the class)
    o = lambda a, b: sorted((x0 + end * a, x0 + end * b))
    xa, xb = o(0, 0.64)
    vm.add(Box((xa, y - 0.08, 6.08), (xb, y + 0.08, 6.56)), "iron", W, big=False)
    xa, xb = o(0.0, 1.12)
    vm.add(Box((xa, y - 0.56, 5.92), (xb, y + 0.56, 6.08)), "iron", W, big=False)
    vm.add(Box((xa, y - 0.56, 4.32), (xb, y + 0.56, 4.48)), "iron", W, big=False)
    neon(name, W, (x0 + end * 0.56, y, 5.20), (0.88, 0.88, 1.40), color=C_LANTERN)


EMBLEM = [  # triple spiral over a chevron (one char = one small cube), top row first
    "......###......",
    ".....#...#.....",
    ".....#.#.#.....",
    ".....#..##.....",
    "..###.###..###.",
    ".#...#...##...#",
    ".#.#.#....#.#.#",
    ".#..##....#..##",
    "..##........##.",
    "...............",
    "..#.........#..",
    "...#.......#...",
    "....#.....#....",
    ".....#...#.....",
    "......#.#......",
    ".......#.......",
]


def banner(vm, cx, side):
    W = "Wagon"
    y0 = side * WY
    ya, yb = sorted((y0, y0 + side * S))
    vm.add(Box((cx - 1.76, ya, 6.24), (cx + 1.76, yb, 6.40)), "wood_dark", W, big=False)   # rod
    poly = [(-1.44, 0), (1.44, 0), (1.44, -3.2), (0, -4.0), (-1.44, -3.2)]
    inner = [(-1.28, -0.32), (1.28, -0.32), (1.28, -3.12), (0, -3.84), (-1.28, -3.12)]
    org = (cx, y0 + side * S / 2, 6.24)
    vm.add(Plate(org, (1, 0, 0), (0, 0, 1), poly, S), "banner_trim", W, big=False)
    vm.paint(Plate(org, (1, 0, 0), (0, 0, 1), inner, S * 1.2), "banner", parts=[W])
    w = len(EMBLEM[0]) * S
    u = (-1, 0, 0) if side > 0 else (1, 0, 0)
    ox = cx + w / 2 if side > 0 else cx - w / 2
    vm.paint_bitmap((ox, y0 + side * S, 3.2), u, (0, 0, 1), EMBLEM, {"#": "emblem"},
                    depth=1, normal=(0, side, 0), parts=[W])


def wagon(vm):
    W = "Wagon"
    vm.add(Box((-WX, -WY, Z0), (WX, WY, ZD)), "wood", W)
    rim = Box((-WX, -WY, ZD - B), (WX, WY, ZR)) - Box((-WX + RIM, -WY + RIM, ZD - B - 1), (WX - RIM, WY - RIM, ZR + 1))
    vm.add(rim, "wood_dark", W)
    vm.paint(Box((-WX, -WY, Z0), (WX, WY, Z0 + 2 * B)), "wood_dark", parts=[W])            # bottom beam
    posts = [(-WX, -19.68), (-11.04, -10.08), (-3.84, -2.88), (2.88, 3.84), (10.08, 11.04), (19.68, WX)]
    for a, b in posts:
        vm.paint(sym(Box((a, WY - B, Z0), (b, WY, ZR))), "wood_dark", parts=[W])
    # recessed plank panels (relief by rows set back one small cube)
    for a, b in [(-19.68, -11.04), (-10.08, -3.84), (3.84, 10.08), (11.04, 19.68)]:
        vm.cut(sym(Box((a, WY - S, Z0 + 2 * B), (b, WY + 0.01, ZD - B))))
    # end faces: recessed panel + stone plaque with a glowing rune
    for e in (-1, 1):
        xa, xb = sorted((e * WX, e * (WX - S)))
        vm.cut(Box((xa, -3.84, Z0 + 2 * B), (xb, 3.84, ZD - B)))
        xa, xb = sorted((e * (WX - S), e * (WX + B)))
        vm.add(Box((xa, -1.92, 3.36), (xb, 1.92, 6.24)), "stone", W, big=False)
        vm.paint(Box((xa, -1.92, 3.36), (xb, 1.92, 6.24)), "rune", parts=[W],
                 where=lambda X, Y, Z, e=e: spiral_where(0, 4.8, 0.5, 0.17, 0.2, 1.3, plane="yz")(X, Y, Z)
                 & (np.abs(X) > WX + B - 0.17))
        for y in (-3.84, 3.84):
            lantern_end(vm, y, e, f"Lantern_{'F' if e > 0 else 'B'}{'L' if y > 0 else 'R'}")
    # central alcove with a crystal
    vm.cut(sym(Box((-2.88, WY - 2 * B, Z0 + 2 * B), (2.88, WY + 0.01, ZD - B))))
    vm.paint(sym(Box((-2.88, WY - 2 * B - S, Z0 + 2 * B - S), (2.88, WY, ZD - B + S))), "stone_dark", parts=[W],
             only=["wood", "wood_dark"])
    for side in (-1, 1):
        y = side * (WY - 1.2)
        R = rot(rz=45)
        neon(f"AlcoveCrystal{'L' if side > 0 else 'R'}", W, (0, y, 4.5), (0.9, 0.9, 1.9), R=R)
        neon(f"AlcoveCrystalTip{'L' if side > 0 else 'R'}", W, (0, y, 5.75), (0.5, 0.5, 0.7), R=R, color=C_CRYSTAL_TIP)
        for sx in (-1, 1):
            R2 = rot(ry=sx * 28) @ rot(rz=45)
            neon(f"AlcoveShard{'L' if side > 0 else 'R'}{'F' if sx > 0 else 'B'}", W,
                 (sx * 0.95, y, 4.0), (0.5, 0.5, 1.0), R=R2)
    # wheels, bogies, under-frame
    for x in (-17.04, -11.76, 11.76, 17.04):
        vm.add(sym(Cylinder((x, 3.84, 1.68), (x, 4.80, 1.68), 1.68)), "wheel", W)
        ring = Cylinder((x, 4.64, 1.68), (x, 4.81, 1.68), 1.70) - Cylinder((x, 4.5, 1.68), (x, 4.9, 1.68), 1.22)
        vm.paint(sym(ring), "wheel_rim", parts=[W])
        for side in (-1, 1):
            neon(f"Wheel{'F' if x > 0 else 'B'}{abs(int(x))}{'L' if side > 0 else 'R'}", W,
                 (x, side * 4.88, 1.68), (2.56, 0.16, 0.32), color=C_WHEEL)
    for e in (-1, 1):
        xa, xb = sorted((e * 11.76, e * 17.04))
        vm.add(sym(Box((xa, 2.88, 0.96), (xb, 3.84, 2.40))), "stone_dark", W)
    vm.add(sym(Box((-2.88, 2.88, 0.96), (2.88, 4.32, 2.40))), "stone_dark", W)
    for side in (-1, 1):
        neon(f"UnderBar{'L' if side > 0 else 'R'}", W, (0, side * 4.40, 1.68), (3.84, 0.16, 0.32), color=C_WHEEL)
    # tusks: corner pairs and the alcove frame
    for e in (-1, 1):
        for side in (-1, 1):
            tusk(vm, (e * 20.6, side * 5.04, 5.9), (e * 22.6, side * 5.04, 10.6), (e * 18.6, side * 5.04, 13.6), 0.72, 0.24)
            tusk(vm, (e * 18.2, side * 5.52, 5.6), (e * 20.0, side * 5.52, 9.6), (e * 16.4, side * 5.52, 12.0), 0.60, 0.24)
            tusk(vm, (e * 4.3, side * 5.52, 2.9), (e * 5.2, side * 5.52, 6.8), (e * 1.8, side * 5.52, 9.2), 0.66, 0.24)
    # banners and lanterns on the long sides
    for side in (-1, 1):
        for cx in (-7.04, 7.04):      # edges on cube boundaries: exact front / back symmetry
            banner(vm, cx, side)
        for x in (-10.56, 10.56):
            lantern_side(vm, x, side, f"Lantern_{'F' if x > 0 else 'B'}{'L' if side > 0 else 'R'}")
    # moss on the rim and hanging strands
    def moss_noise(X, Y, Z):
        ix = np.floor(X / 0.32).astype(np.int64)
        iy = np.floor(np.abs(Y) / 0.32).astype(np.int64)
        iz = np.floor(Z / 0.32).astype(np.int64)
        return (0.6 * hash01(ix // 2, iy // 2, iz // 2, seed=5) + 0.4 * hash01(ix, iy, iz, seed=6)) > 0.58
    vm.paint(Box((-WX - 0.5, -WY - 0.5, ZD - B), (WX + 0.5, WY + 0.5, ZR + 0.01)), "moss",
             only=["wood_dark"], parts=[W], where=moss_noise)
    # moss drips painted down the beams and panels (no geometry)
    def drips(X, Y, Z):
        col = np.floor(np.abs(X) / 0.32).astype(np.int64)   # symmetric front / back
        L = 0.3 + 1.6 * hash01(col, 0, 0, seed=9) ** 2
        on = hash01(col // 2, 1, 0, seed=8) > 0.45
        return on & (Z > ZD - B - L)
    vm.paint(sym(Box((-WX, WY - 2 * S, Z0 + B), (WX, WY + 0.01, ZD))), "moss", parts=[W],
             only=["wood", "wood_dark"], where=drips)
    vm.symmetrize_x(W)   # exact front / back symmetry: the back reuses the front's texels


# ------------------------------------------------------------------------------- serpent
def cloud(vm, puffs, part, spiral=True, only_empty=False, light=True):
    for (x, y, z, r) in puffs:
        vm.add(Ellipsoid((x, y, z), (r, r * 0.92, r * 0.86)), "cloud", part, only_empty=only_empty)
    if light:
        for (x, y, z, r) in puffs:
            vm.paint(Ellipsoid((x, y, z), (r + 0.2, r * 0.92 + 0.2, r * 0.86 + 0.2)), "cloud_light", only=["cloud"],
                     parts=[part], where=lambda X, Y, Z, z=z, r=r: Z > z + 0.42 * r)
    if spiral:
        for (x, y, z, r) in puffs:
            if r < 0.9:
                continue
            band = Box((x - r, y - r - 0.1, z - r), (x + r, y + r + 0.1, z + r)) - Box((x - r, y - r + 0.42, z - r), (x + r, y + r - 0.42, z + r))
            vm.paint(band, "cloud_line", only=["cloud"], parts=[part],
                     where=spiral_where(x, z, 0.42, 0.15, 0.2, 0.78 * r))


SWIRLS = [  # x, z, radius, half thickness: rolled scrolls in 3 stepped discs, sitting ON the back
    (-11.6, 12.4, 1.9, 1.20),
    (-7.4, 13.8, 2.5, 1.68),
    (-2.6, 14.6, 3.0, 2.16),
    (2.4, 14.1, 2.6, 1.68),
    (6.2, 13.0, 1.9, 1.20),
]

CRYSTALS = [  # x, y, base z, radius, body height, tip height: on the spine, rising behind the scrolls
    (-2.6, 0.0, 15.0, 1.60, 4.8, 3.0),
    (-7.2, 0.0, 14.4, 1.25, 3.6, 2.2),
    (2.2, 0.0, 14.6, 1.30, 3.8, 2.3),
    (-11.2, 0.0, 13.4, 0.95, 2.6, 1.6),
    (5.8, 0.0, 13.6, 0.95, 2.6, 1.6),
]


def crystal_shape(x, y, zb, r, hb, ht):
    """Hexagonal prism (flat faces toward +-Y) with a pyramid tip."""
    from pipeline.vox import Func
    c = np.array([x, y])

    def fn(P):
        dx, dy, z = P[:, 0] - c[0], P[:, 1] - c[1], P[:, 2]
        hexd = np.maximum(np.abs(dy), np.maximum(np.abs(dy * 0.5 + dx * 0.866), np.abs(dy * 0.5 - dx * 0.866)))
        reff = np.where(z <= zb + hb, r, r * (1 - (z - zb - hb) / ht))
        return (z >= zb) & (z <= zb + hb + ht) & (hexd <= reff)
    return Func(fn, (x - r - 0.2, y - r - 0.2, zb), (x + r + 0.2, y + r + 0.2, zb + hb + ht))


def body(vm):
    P = "Body"
    spine = [(-19.6, 0, 7.6), (-16.0, 0, 8.3), (-11.0, 0, 8.8), (-4.0, 0, 9.0), (3.0, 0, 8.9), (8.4, 0, 8.8)]
    radii = [1.4, 2.8, 3.9, 4.4, 4.4, 4.2]
    sh = Scale(Sweep(spine, radii), (0, 0, ZD), (1, 1, 0.95))
    inside = Box((-WX + RIM, -10, ZD), (WX - RIM, 10, 40)) | Box((-30, -10, ZR), (30, 10, 40))   # never eats the rim
    vm.add(sh & inside, "navy", P)
    vm.add(Ellipsoid((8.0, 0, 9.4), (4.4, 4.3, 3.4)) & inside, "navy", P)   # shoulder
    vm.paint(Box((-17, -1.44, 11.0), (8, 1.44, 14)), "navy_light", parts=[P], only=["navy"])           # back ridge
    # tail = rear of the body, fin at the tip
    vm.setpart(Box((-30, -10, 0), (-14.4, 10, 40)), "Tail1", parts=[P])
    vm.setpart(Box((-30, -10, 0), (-17.76, 10, 40)), "Tail2", parts=["Tail1"])
    fin = [(0, 0), (-0.9, 1.6), (-2.2, 2.7), (-2.0, 1.2), (-1.2, 0.0)]
    vm.add(Plate((-18.9, 0, 7.9), (1, 0, 0), (0, 0, 1), fin, 0.48), "navy_light", "Tail2", big=False)
    vm.paint(Plate((-18.9, 0, 7.9), (1, 0, 0), (0, 0, 1), [(-0.9, 1.6), (-2.2, 2.7), (-2.0, 1.2), (-1.5, 1.4)], 0.6),
             "teal", parts=["Tail2"])
    # storm: a big cloud mass over the back and the tail, clouds on the flanks
    cloud(vm, [(-16.6, 0, 11.6, 1.8), (-14.4, 0, 13.4, 2.1), (-12.6, 0, 13.6, 1.6), (-15.4, 2.0, 11.6, 1.4),
               (-15.4, -2.0, 11.6, 1.4), (-18.4, 0, 10.6, 1.3), (-13.6, 0, 15.4, 1.2)], P)
    for (x, y, z, r) in [(-7.6, 3.2, 10.6, 1.45), (1.6, 3.3, 10.8, 1.4)]:
        cloud(vm, [(x, y, z, r), (x, -y, z, r)], P)
    # turquoise wave scrolls: 3 stepped discs each (relief), spiral painted on the sides
    for (x, z, r, t) in SWIRLS:
        for k, (fr, dt) in enumerate(((1.0, 0.0), (0.68, 0.48), (0.38, 0.96))):
            rr_, tt = r * fr, t + dt
            vm.add(Cylinder((x, -tt, z), (x, tt, z), rr_), "teal", P)
            band = Box((x - rr_, -tt - 0.1, z - rr_), (x + rr_, tt + 0.1, z + rr_)) - \
                Box((x - rr_, -tt + 0.33, z - rr_), (x + rr_, tt - 0.33, z + rr_))
            vm.paint(band, "teal_dark", only=["teal"], parts=[P], where=spiral_where(x, z, 0.66, 0.26, 0.25, r - 0.2))
    # crystals: pointed hexagonal prisms, a glowing mesh of their own (set it to Neon in Studio)
    C = "Crystals"
    for i, (x, y, zb, r, hb, ht) in enumerate(CRYSTALS):
        vm.add(crystal_shape(x, y, zb, r, hb, ht), "ice", C, big=False)
        vm.paint(Box((x - r - 0.3, y - r - 0.3, zb - 0.1), (x + r + 0.3, y + r + 0.3, zb + 0.32 * hb)), "ice_dark", parts=[C])
        vm.paint(Box((x - r - 0.3, y - r - 0.3, zb + hb - 0.1), (x + r + 0.3, y + r + 0.3, zb + hb + ht + 0.2)),
                 "ice_light", parts=[C])
        if i == 0:
            ATTACHMENTS.append(dict(name="CrystalTop", part=C, position=(x, y, zb + hb + ht), direction=(0, 0, 1)))
        if i < 3:
            ATTACHMENTS.append(dict(name=f"Lightning{i + 1}", part=C, position=(x, y, zb + hb + ht), direction=(0, 0, 1)))
    ATTACHMENTS.append(dict(name="Lightning4", part=P, position=(-14.6, 0, 14.0), direction=(0, 0, 1)))


NECK = [(8.0, 0, 9.0), (9.4, 0, 12.6), (10.0, 0, 16.4), (10.4, 0, 19.8), (11.5, 0, 22.9), (13.4, 0, 25.0), (15.4, 0, 26.0)]
NECK_R = [3.8, 3.0, 2.45, 2.15, 1.9, 1.75, 1.6]


def neck(vm):
    sh = Scale(Sweep(NECK, NECK_R), (10, 0, 0), (1, 0.9, 1))
    vm.add(sh & Box((-30, -10, ZD), (30, 10, 40)), "navy", "Neck1")     # never digs into the wagon
    vm.setpart(Box((-30, -10, 0), (30, 10, 10.08)), "Body", parts=["Neck1"])
    pts = np.array(NECK)
    dense, rad = [], []
    for i in range(len(pts) - 1):
        for t in np.linspace(0, 1, 12, endpoint=False):
            dense.append(pts[i] * (1 - t) + pts[i + 1] * t)
            rad.append(NECK_R[i] * (1 - t) + NECK_R[i + 1] * t)
    dense, rad = np.array(dense), np.array(rad)
    tang = np.gradient(dense, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    front = np.stack([tang[:, 2], np.zeros(len(tang)), -tang[:, 0]], 1)

    def belly(X, Y, Z):
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        out = np.zeros(len(P), bool)
        for s in range(0, len(P), 8000):
            Q = P[s:s + 8000]
            d2 = ((Q[:, None, :] - dense[None, :, :]) ** 2).sum(2)
            k = np.argmin(d2, 1)
            off = Q - dense[k]
            out[s:s + 8000] = (off * front[k]).sum(1) > 0.42 * rad[k]
        return out.reshape(X.shape)
    vm.paint(Box((4, -4, 8), (17, 4, 28)), "belly", only=["navy"], parts=["Body", "Neck1"], where=belly)
    vm.setpart(Box((-30, -10, 17.28), (30, 10, 40)), "Neck2", parts=["Neck1"])
    # wide neon stripe on the front of the neck, up to the throat
    idx = [6, 15, 24, 33, 41, 49, 56, 62, 67]
    for j in range(len(idx) - 1):
        a, b = idx[j], idx[j + 1]
        c = (dense[a] + dense[b]) / 2
        t = dense[b] - dense[a]
        L = np.linalg.norm(t)
        t /= L
        f = np.array([t[2], 0, -t[0]])
        r = (rad[a] + rad[b]) / 2
        R = np.column_stack([t, [0, 1, 0], -f])
        center = c + f * (r - 0.02)
        part = "Neck1" if center[2] < 17.28 else "Neck2"
        if center[0] >= 14.4 and center[2] > 23.0:
            part = "Head"
        neon(f"NeckStripe{j + 1}", part, center, (L + 0.14, 0.8, 0.24), R=R, color=C_STRIPE)
    # continuous storm mane down the back of the neck, collar of clouds at its base
    cloud(vm, [(6.6, 0, 11.2, 1.7), (7.4, 2.7, 10.6, 1.4), (7.4, -2.7, 10.6, 1.4)], "Body", only_empty=True)
    for k, rr in ((10, 1.55), (17, 1.5), (24, 1.45), (31, 1.4), (38, 1.35), (45, 1.3), (52, 1.25), (58, 1.2)):
        c = dense[k] - front[k] * (rad[k] * 0.72)
        part = "Neck1" if c[2] < 17.28 else "Neck2"
        cloud(vm, [(c[0], 0, c[2], rr)], part, only_empty=True)
        if k in (17, 38):
            cloud(vm, [(c[0] + 0.5, 1.5, c[2] - 0.4, rr * 0.75), (c[0] + 0.5, -1.5, c[2] - 0.4, rr * 0.75)], part,
                  only_empty=True, spiral=False)
    ATTACHMENTS.append(dict(name="Lightning5", part="Neck2", position=tuple(dense[45] - front[45] * 2.6), direction=(0, 0, 1)))


def snout_hw(x):
    """Half width of the upper snout (tapers toward the nose)."""
    return np.where(x < 19.2, 1.28, np.where(x < 20.8, 1.12, 0.96))


def head(vm):
    H, J = "Head", "Jaw"
    vm.setpart(Box((14.4, -10, 23.0), (30, 10, 40)), H, parts=["Neck2"])
    # skull: wide rounded block (big cubes)
    skull = Box((14.4, -1.92, 25.44), (18.24, 1.92, 28.32)) & Ellipsoid((16.5, 0, 26.6), (3.0, 2.45, 2.55))
    vm.add(skull, "navy", H)
    # upper snout sculpted in small cubes: side profiles extruded with a tapering width
    o, u, v = (0, 0, 0), (1, 0, 0), (0, 0, 1)
    vm.add(Plate(o, u, v, [(18.08, 25.92), (19.36, 25.92), (19.36, 27.68), (18.08, 27.84)], 2.56), "navy", H, big=False)
    vm.add(Plate(o, u, v, [(19.2, 25.92), (20.96, 25.92), (20.96, 27.36), (19.2, 27.68)], 2.24), "navy", H, big=False)
    vm.add(Plate(o, u, v, [(20.8, 25.92), (22.08, 25.92), (22.08, 26.88), (21.76, 27.2), (20.8, 27.36)], 1.92),
           "navy", H, big=False)
    vm.paint(Box((18.0, -0.5, 27.3), (22.2, 0.5, 28.0)), "navy_light", parts=[H])        # nose ridge
    # crest along the skull
    vm.add(Box((14.88, -0.32, 28.32), (17.76, 0.32, 28.64)), "navy_light", H, big=False)
    # cheek plates
    vm.paint(sym(Box((15.4, 1.6, 25.44), (17.4, 2.0, 26.4))), "navy_light", parts=[H])
    # eyes: notch at the front-side corner (2 layers deep from the front AND the side),
    # neon at the bottom, heavy brow overhanging both ways -> readable from the front and in profile
    vm.cut(sym(Box((17.28, 1.28, 26.88), (18.25, 1.93, 27.52))))
    vm.paint(sym(Box((17.08, 1.08, 26.70), (17.30, 1.95, 27.54))), "socket", parts=[H])
    vm.paint(sym(Box((17.08, 1.08, 26.70), (18.30, 1.30, 27.54))), "socket", parts=[H])
    vm.paint(sym(Box((17.08, 1.08, 26.70), (18.30, 1.95, 26.90))), "socket", parts=[H])
    for side in (-1, 1):
        neon(f"Eye{'L' if side > 0 else 'R'}", H, (17.60, side * 1.44, 27.12), (0.64, 0.32, 0.48), color=C_EYE)
    vm.add(sym(Box((16.80, 1.12, 27.52), (18.40, 2.08, 27.84))), "navy_light", H, big=False)
    # nostrils
    vm.cut(sym(Box((21.44, 0.32, 26.96), (21.76, 0.64, 27.30))))
    # mouth line along the sides of the upper jaw
    vm.paint(Box((17.6, -1.5, 25.92), (22.1, 1.5, 26.08)), "mouth", parts=[H],
             where=lambda X, Y, Z: (np.abs(Y) > snout_hw(X) - 0.17) | (X > 21.9))
    # fangs hanging outside the jaw
    for x, hw in ((19.52, 1.12), (20.48, 1.12), (21.28, 0.96)):
        for side in (-1, 1):
            ya, yb = sorted((side * (hw - 0.16), side * hw))
            vm.add(Box((x, ya, 25.60), (x + 0.32, yb, 25.92)), "tooth", H, big=False)
            vm.add(Box((x, ya, 25.44), (x + 0.16, yb, 25.60)), "tooth", H, big=False)
    # lower jaw: narrower than the snout, chin steps up
    vm.add(Box((15.36, -1.12, 24.80), (19.20, 1.12, 25.92)), "navy", J, big=False)
    vm.add(Box((19.20, -0.96, 24.96), (20.80, 0.96, 25.92)), "navy", J, big=False)
    vm.add(Box((20.80, -0.80, 25.12), (21.44, 0.80, 25.92)), "navy", J, big=False)
    vm.add(Box((21.44, -0.64, 25.28), (21.92, 0.64, 25.92)), "navy", J, big=False)
    vm.paint(Box((15.3, -1.3, 24.7), (22.0, 1.3, 25.12)), "belly", parts=[J])
    for x in (19.04, 20.0, 20.96):
        vm.paint(sym(Box((x, 0.5, 25.76), (x + 0.16, 1.13, 25.92))), "tooth", parts=[J])
    # cloud mane trailing from the back of the head
    cloud(vm, [(14.2, 0, 28.0, 1.35), (12.4, 0, 27.6, 1.75), (10.3, 0, 26.9, 1.7), (8.3, 0, 26.0, 1.5),
               (6.5, 0, 25.0, 1.2), (5.0, 0, 24.2, 0.9), (12.2, 1.5, 26.6, 1.2), (12.2, -1.5, 26.6, 1.2),
               (9.8, 1.4, 25.6, 1.1), (9.8, -1.4, 25.6, 1.1)], H)
    ATTACHMENTS.append(dict(name="Lightning6", part=H, position=(10.3, 0, 28.3), direction=(0, 0, 1)))
    # swept-back fins on the skull and frills on the cheeks
    for side in (-1, 1):
        pts = bezier((15.6, side * 1.1, 28.0), (14.6, side * 1.3, 29.0), (13.0, side * 1.6, 29.3), n=18)
        for p_, r_ in zip(pts, np.linspace(0.34, 0.1, 18)):
            vm.add(Box(p_ - r_, p_ + r_), "navy_light", H, big=False)
        pts = bezier((15.3, side * 1.8, 26.1), (14.5, side * 2.2, 26.3), (13.5, side * 2.55, 27.0), n=14)
        for i_, (p_, r_) in enumerate(zip(pts, np.linspace(0.3, 0.1, 14))):
            vm.add(Box(p_ - r_, p_ + r_), "teal" if i_ > 9 else "navy_light", H, big=False)
    ATTACHMENTS.append(dict(name="BreathFX", part=H, position=(22.1, 0, 26.0), direction=(1, 0, 0)))


def make():
    NEON.clear()
    ATTACHMENTS.clear()
    vm = VoxelModel((-23.04, -7.2, -0.48), (23.04, 7.2, 30.24))   # centred in X and Y (mirrored texels)
    wagon(vm)
    body(vm)
    neck(vm)
    head(vm)
    return vm
