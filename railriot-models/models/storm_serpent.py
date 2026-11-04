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
    "navy": dict(rgb=(50, 54, 132), fx="scales"),
    "navy_light": dict(rgb=(72, 82, 170), fx="scales"),
    "belly": dict(rgb=(72, 112, 206)),
    "cloud": dict(rgb=(26, 26, 54)),
    "cloud_line": dict(rgb=(86, 94, 172)),
    "teal": dict(rgb=(42, 182, 198)),
    "teal_dark": dict(rgb=(22, 108, 140)),
    "mouth": dict(rgb=(22, 14, 30), studs=False, jitter=0.0),
    "tooth": dict(rgb=(232, 226, 206), studs=False, jitter=0.0),
    "socket": dict(rgb=(14, 12, 24), studs=False, jitter=0.0),
}

PARTS = [
    dict(name="Wagon", parent=None, pivot=(0, 0, 0), group="static", outline=0.16),
    dict(name="Body", parent="Wagon", pivot=(0, 0, ZD), group="static", outline=0.2),
    dict(name="Tail1", parent="Body", pivot=(-14.4, 0, 7.9), outline=0.2),
    dict(name="Tail2", parent="Tail1", pivot=(-17.76, 0, 7.7), outline=0.2),
    dict(name="Neck1", parent="Body", pivot=(10.0, 0, 10.08), outline=0.2),
    dict(name="Neck2", parent="Neck1", pivot=(10.5, 0, 17.28), outline=0.2),
    dict(name="Head", parent="Neck2", pivot=(14.4, 0, 25.6), outline=0.2),
    dict(name="Jaw", parent="Head", pivot=(16.32, 0, 25.92), outline=0.2),
]

NEON = []
DROP_DOWN = dict(parts=["Wagon"], below=2.5)
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
def tusk(vm, base, ctrl, tip, r0, r1, part="Wagon", n=40, rope=True):
    """Square-section tusk along a planar curve (flat sides = few triangles)."""
    pts = bezier(base, ctrl, tip, n=n)
    t = np.linspace(0, 1, n)
    rr = r0 * (1 - t) ** 0.85 + r1 * t
    for p, r in zip(pts, rr):
        vm.add(Box(p - r, p + r), "ivory", part, big=False)
    for f in (0.34, 0.6):
        i = int(round(f * (n - 1)))
        c, r = pts[i], rr[i] + 0.1
        vm.paint(Box(c - r, c + r), "ivory_dark", only=["ivory"], parts=[part])
    if rope:
        c, r = pts[3], rr[3] + 0.12
        vm.paint(Box(c - r, c + r), "rope", only=["ivory"], parts=[part])
    return pts


def lantern_side(vm, x, side, name):
    """Lantern hanging from the rim beam on a long side (side = +1 / -1)."""
    W = "Wagon"
    y0 = side * WY
    o = lambda a, b: (y0 + side * a, y0 + side * b)
    ya, yb = o(0, 0.48)
    vm.add(Box((x - 0.08, ya, 6.40), (x + 0.08, yb, 6.56)), "iron", W, big=False)       # arm
    ya, yb = o(0.32, 0.48)
    vm.add(Box((x - 0.08, ya, 5.92), (x + 0.08, yb, 6.40)), "iron", W, big=False)       # chain
    ya, yb = o(0.0, 0.80)
    vm.add(Box((x - 0.40, ya, 5.76), (x + 0.40, yb, 5.92)), "iron", W, big=False)       # cap
    vm.add(Box((x - 0.40, ya, 4.64), (x + 0.40, yb, 4.80)), "iron", W, big=False)       # base
    for dx in (-0.40, 0.24):
        for dy in (0.0, 0.64):
            ya, yb = o(dy, dy + 0.16)
            vm.add(Box((x + dx, ya, 4.80), (x + dx + 0.16, yb, 5.76)), "iron", W, big=False)
    neon(name, W, (x, y0 + side * 0.40, 5.28), (0.48, 0.48, 0.96), color=C_LANTERN)


def lantern_end(vm, y, end, name):
    """Lantern on an end face (end = +1 front / -1 back)."""
    W = "Wagon"
    x0 = end * (WX + 0.48)  # end plaques stick out by one big cube
    o = lambda a, b: (x0 + end * a, x0 + end * b)
    xa, xb = o(-0.48, 0.48)
    vm.add(Box((xa, y - 0.08, 6.40), (xb, y + 0.08, 6.56)), "iron", W, big=False)
    xa, xb = o(0.32, 0.48)
    vm.add(Box((xa, y - 0.08, 5.92), (xb, y + 0.08, 6.40)), "iron", W, big=False)
    xa, xb = o(0.0, 0.80)
    vm.add(Box((xa, y - 0.40, 5.76), (xb, y + 0.40, 5.92)), "iron", W, big=False)
    vm.add(Box((xa, y - 0.40, 4.64), (xb, y + 0.40, 4.80)), "iron", W, big=False)
    for dy in (-0.40, 0.24):
        for dx in (0.0, 0.64):
            xa, xb = o(dx, dx + 0.16)
            vm.add(Box((xa, y + dy, 4.80), (xb, y + dy + 0.16, 5.76)), "iron", W, big=False)
    neon(name, W, (x0 + end * 0.40, y, 5.28), (0.48, 0.48, 0.96), color=C_LANTERN)


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
    # crossed bones on the back plaque
    for sgn in (-1, 1):
        vm.add(RoundCone((-WX - 0.72, -2.5 * sgn, 2.9), (-WX - 0.72, 2.5 * sgn, 6.7), 0.30, 0.30), "ivory", W, big=False)
        for p in ((-WX - 0.72, -2.6 * sgn, 2.8), (-WX - 0.72, 2.6 * sgn, 6.8)):
            vm.add(Sphere(p, 0.42), "ivory", W, big=False)
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
            tusk(vm, (e * 20.9, side * 5.2, 5.9), (e * 22.9, side * 5.2, 10.4), (e * 18.8, side * 5.2, 13.3), 0.60, 0.13)
            tusk(vm, (e * 18.9, side * 5.44, 5.7), (e * 20.7, side * 5.44, 9.6), (e * 16.9, side * 5.44, 12.1), 0.52, 0.11)
            tusk(vm, (e * 4.25, side * 5.6, 2.9), (e * 5.1, side * 5.6, 6.7), (e * 1.7, side * 5.6, 9.1), 0.56, 0.12)
    # banners and lanterns on the long sides
    for side in (-1, 1):
        for cx in (-6.96, 6.96):
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
    rng = np.random.default_rng(3)
    strands = [(x, float(rng.choice([0.48, 0.64, 0.96, 1.28])))
               for x in (-20.4, -18.2, -15.0, -12.8, -8.9, -4.8, -2.4, 4.6, 8.9, 13.2, 15.4, 18.6)]
    for side in (-1, 1):
        for x, L in strands:
            xs = math.floor(x / S + 0.5) * S
            ya, yb = sorted((side * WY, side * (WY + S)))
            vm.add(Box((xs, ya, 6.40 - L), (xs + 0.32, yb, 6.40)), "moss", W, big=False, only_empty=True)


# ------------------------------------------------------------------------------- serpent
def cloud(vm, puffs, part, spiral=True):
    for (x, y, z, r) in puffs:
        vm.add(Ellipsoid((x, y, z), (r, r * 0.92, r * 0.86)), "cloud", part)
    if spiral:
        for (x, y, z, r) in puffs:
            if r < 0.9:
                continue
            band = Box((x - r, y - r - 0.1, z - r), (x + r, y + r + 0.1, z + r)) - Box((x - r, y - r + 0.42, z - r), (x + r, y + r - 0.42, z + r))
            vm.paint(band, "cloud_line", only=["cloud"], parts=[part],
                     where=spiral_where(x, z, 0.42, 0.15, 0.2, 0.78 * r))


SWIRLS = [  # x, z, radius, half thickness
    (-11.2, 10.5, 1.70, 0.96),
    (-7.4, 11.4, 2.30, 1.44),
    (-2.8, 12.0, 2.85, 1.92),
    (2.1, 11.6, 2.40, 1.44),
    (6.0, 10.8, 1.75, 0.96),
]

CRYSTALS = [  # x, y, z base, height, width, tilt about X, tilt about Y
    (-2.8, 0.0, 12.6, 8.0, 2.40, 0, -3),
    (-7.0, 0.6, 12.0, 6.0, 1.90, 9, 15),
    (1.8, -0.6, 12.2, 6.4, 2.00, -9, -15),
    (-10.6, -0.4, 11.0, 4.4, 1.50, -8, 25),
    (5.6, 0.4, 11.4, 4.6, 1.50, 8, -26),
    (-5.0, 2.5, 10.6, 3.2, 1.20, 28, 6),
    (-5.0, -2.5, 10.6, 3.2, 1.20, -28, 6),
    (-0.4, 2.6, 10.8, 3.0, 1.10, 30, -8),
    (-0.4, -2.6, 10.8, 3.0, 1.10, -30, -8),
]


def crystal(name, part, base, h, w, tx, ty):
    """Stepped ice crystal: 3 neon prisms with a diamond section, tilted."""
    R = rot(ry=ty) @ rot(rx=tx) @ rot(rz=45)
    ax = R @ np.array([0, 0, 1.0])
    base = np.asarray(base, float)
    neon(f"{name}", part, base + ax * 0.30 * h, (w, w * 0.85, 0.60 * h), R=R)
    neon(f"{name}Mid", part, base + ax * 0.725 * h, (w * 0.66, w * 0.56, 0.25 * h), R=R)
    neon(f"{name}Tip", part, base + ax * 0.925 * h, (w * 0.36, w * 0.30, 0.15 * h), R=R, color=C_CRYSTAL_TIP)
    return base + ax * h, ax


def body(vm):
    P = "Body"
    spine = [(-19.2, 0, 7.6), (-15.8, 0, 8.1), (-10.8, 0, 8.4), (-4.0, 0, 8.6), (3.0, 0, 8.6), (9.4, 0, 8.6)]
    radii = [1.3, 2.5, 3.6, 4.3, 4.4, 3.9]
    sh = Scale(Sweep(spine, radii), (0, 0, ZD), (1, 1, 0.80))
    vm.add(sh & Box((-30, -10, ZD), (30, 10, 40)), "navy", P)
    vm.paint(Box((-17, -1.44, 9.6), (10, 1.44, 13)), "navy_light", parts=[P], only=["navy"])   # back ridge
    # tail = rear of the body, fin at the tip
    vm.setpart(Box((-30, -10, 0), (-14.4, 10, 40)), "Tail1", parts=[P])
    vm.setpart(Box((-30, -10, 0), (-17.76, 10, 40)), "Tail2", parts=["Tail1"])
    fin = [(0, 0), (-0.9, 1.6), (-2.2, 2.7), (-2.0, 1.2), (-1.2, 0.0)]
    vm.add(Plate((-18.6, 0, 7.8), (1, 0, 0), (0, 0, 1), fin, 0.48), "navy_light", "Tail2", big=False)
    vm.paint(Plate((-18.6, 0, 7.8), (1, 0, 0), (0, 0, 1), [(-0.9, 1.6), (-2.2, 2.7), (-2.0, 1.2), (-1.5, 1.4)], 0.6),
             "teal", parts=["Tail2"])
    # storm clouds: behind the swirls and along the flanks
    cloud(vm, [(-16.0, 0, 9.8, 1.40), (-14.2, 0, 11.0, 1.70), (-12.4, 0, 10.9, 1.35)], P)
    for (x, y, z, r) in [(-8.8, 3.0, 9.8, 1.15), (-4.0, 3.25, 10.1, 1.2), (1.0, 3.2, 10.0, 1.1), (5.6, 2.9, 9.8, 1.0)]:
        cloud(vm, [(x, y, z, r), (x, -y, z, r)], P)
    # turquoise wave swirls along the spine
    for (x, z, r, t) in SWIRLS:
        vm.add(Cylinder((x, -t, z), (x, t, z), r), "teal", P)
        band = Box((x - r, -t - 0.1, z - r), (x + r, t + 0.1, z + r)) - Box((x - r, -t + 0.33, z - r), (x + r, t - 0.33, z + r))
        vm.paint(band, "teal_dark", only=["teal"], parts=[P], where=spiral_where(x, z, 0.66, 0.22, 0.25, r - 0.22))
    for i, (x, y, z, h, w, tx, ty) in enumerate(CRYSTALS):
        top, ax = crystal(f"Crystal{i + 1}", P, (x, y, z), h, w, tx, ty)
        if i == 0:
            ATTACHMENTS.append(dict(name="CrystalTop", part=P, position=top.tolist(), direction=ax.tolist()))


NECK = [(9.4, 0, 8.4), (10.3, 0, 12.4), (10.4, 0, 16.4), (10.6, 0, 19.8), (11.6, 0, 22.9), (13.4, 0, 25.0), (15.4, 0, 26.0)]
NECK_R = [3.0, 2.5, 2.15, 1.95, 1.8, 1.7, 1.6]


def neck(vm):
    sh = Scale(Sweep(NECK, NECK_R), (10, 0, 0), (1, 0.9, 1))
    vm.add(sh, "navy", "Neck1")
    vm.setpart(Box((-30, -10, 0), (30, 10, 10.08)), "Body", parts=["Neck1"])
    # belly (front/under side of the curve) + neon stripe
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
    vm.paint(Box((5, -3.5, 8), (17, 3.5, 28)), "belly", only=["navy"], parts=["Body", "Neck1"], where=belly)
    vm.setpart(Box((-30, -10, 17.28), (30, 10, 40)), "Neck2", parts=["Neck1"])
    idx = [8, 17, 26, 34, 42, 50, 57, 63]
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
        neon(f"NeckStripe{j + 1}", part, center, (L + 0.14, 0.56, 0.24), R=R, color=C_STRIPE)
    cloud(vm, [(8.4, 0, 14.2, 1.35), (8.9, 1.9, 13.4, 1.1), (8.9, -1.9, 13.4, 1.1)], "Neck1")
    cloud(vm, [(9.0, 0, 21.2, 1.25), (9.5, 1.7, 20.4, 1.05), (9.5, -1.7, 20.4, 1.05)], "Neck2")


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
    cloud(vm, [(14.2, 0, 28.0, 1.30), (12.4, 0, 27.5, 1.60), (10.4, 0, 26.7, 1.50), (8.6, 0, 25.9, 1.30),
               (7.0, 0, 25.2, 1.05), (12.6, 1.4, 26.6, 1.0), (12.6, -1.4, 26.6, 1.0)], H)
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
    vm = VoxelModel((-23.04, -7.2, -0.48), (23.52, 7.2, 30.24))
    wagon(vm)
    body(vm)
    neck(vm)
    head(vm)
    return vm
