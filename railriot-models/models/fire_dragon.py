"""PRIMAL - Fire dragon wagon (ref: red dragon with a lava tank on an industrial wagon).

Size: ~42 x 11 x 21 studs (between the two PRIMAL classes: the reference is lower
than Mosa/Parasaur and taller than Saber/Tricera).  Small cube 0.16, big cube 0.48.
X = front of the train (head side), Y = width, Z = up, wheels on Z = 0.
"""
import math

import numpy as np

from pipeline.vox import (VoxelModel, Box, BoxC, Ellipsoid, Sphere, Cylinder, RoundCone,
                          Sweep, bezier, Plate, Scale, sym, rot)
from pipeline.fx import FX as _FX

NAME = "fire_dragon"
B, S = 0.48, 0.16

WX0, WX1 = -19.68, 16.80   # chassis box along X
WY = 10 * B                 # 4.80 half width
Z0 = 5 * B                  # 2.40 bottom of the chassis box
ZD = 12 * B                 # 5.76 deck

FX = _FX
PALETTE = {
    "metal": dict(rgb=(54, 44, 48), fx="metal"),
    "metal_dark": dict(rgb=(36, 30, 34), fx="metal"),
    "hazard": dict(rgb=(200, 200, 200), fx="hazard"),
    "rail": dict(rgb=(198, 118, 34)),
    "wood": dict(rgb=(96, 58, 42), fx="wood"),
    "wood_dark": dict(rgb=(64, 40, 30)),
    "banner": dict(rgb=(136, 30, 30)),
    "banner_dark": dict(rgb=(92, 20, 22)),
    "bone": dict(rgb=(226, 194, 150)),
    "bone_dark": dict(rgb=(184, 150, 110)),
    "wheel": dict(rgb=(40, 34, 38)),
    "wheel_rim": dict(rgb=(84, 72, 74)),
    "hub": dict(rgb=(122, 40, 32)),
    "claw": dict(rgb=(255, 136, 40), glow=True, studs=False),
    # dragon
    "red": dict(rgb=(158, 34, 30), fx="scales"),
    "red_dark": dict(rgb=(100, 24, 26), fx="scales"),
    "char": dict(rgb=(48, 32, 34)),
    "flame": dict(rgb=(244, 122, 30), glow=True),
    "flame_hot": dict(rgb=(255, 198, 66), glow=True),
    "lava": dict(rgb=(244, 118, 24), fx="lava", glow=True, studs=False),
    "mouth": dict(rgb=(104, 20, 22), studs=False, jitter=0.0),
    "tooth": dict(rgb=(236, 216, 180), studs=False, jitter=0.0),
    "socket": dict(rgb=(22, 12, 14), studs=False, jitter=0.0),
}

PARTS = [
    dict(name="Wagon", parent=None, pivot=(0, 0, 0), group="static", outline=0.16),
    dict(name="Body", parent="Wagon", pivot=(0, 0, ZD), group="static", outline=0.2),
    dict(name="Lava", parent="Body", pivot=(-5.5, 0, 12.0), group="static", outline=0.2),
    dict(name="Tail1", parent="Body", pivot=(-12.96, 0, 13.6), outline=0.2),
    dict(name="Tail2", parent="Tail1", pivot=(-16.32, 0, 14.3), outline=0.2),
    dict(name="Tail3", parent="Tail2", pivot=(-19.2, 0, 12.8), outline=0.2),
    dict(name="Neck", parent="Body", pivot=(7.2, 0, 12.5), outline=0.2),
    dict(name="Head", parent="Neck", pivot=(10.56, 0, 14.6), outline=0.2),
    dict(name="Jaw", parent="Head", pivot=(12.0, 0, 13.6), outline=0.2),
]

NEON = []
ATTACHMENTS = []
MIN_KEEP = 64
# no outline hull inside the eye sockets and in front of them (it would hide the eyes)
NO_OUTLINE = [sym(Box((14.2, 1.44, 15.52), (15.85, 2.80, 16.48))), sym(Box((15.84, 1.44, 15.52), (22.6, 2.80, 16.48)))]
LIGHTS = []
DROP_DOWN = dict(parts=["Wagon"], below=2.5)
RENDER = dict(head=dict(center=(15.6, 0, 15.0), radius=5.4), rail_gauge=7.68)

C_LIGHT = (255, 196, 72)
C_EYE = (255, 150, 36)
C_PIPE = (255, 140, 36)
C_VENT = (255, 118, 30)


def neon(name, part, center, size, R=None, color=C_LIGHT, shape="Block"):
    NEON.append(dict(name=name, part=part, center=tuple(float(c) for c in center), size=tuple(size),
                     R=(np.eye(3) if R is None else np.asarray(R)).tolist(), color=color, shape=shape))


def horn(vm, base, ctrl, tip, r0, r1, mat, part, n=24, tip_mat=None, tip_from=0.7):
    """Square-section horn / tusk along a curve (flat sides = few triangles)."""
    pts = bezier(base, ctrl, tip, n=n)
    t = np.linspace(0, 1, n)
    rr = r0 * (1 - t) ** 0.85 + r1 * t
    for i, (p, r) in enumerate(zip(pts, rr)):
        m = tip_mat if (tip_mat and t[i] >= tip_from) else mat
        vm.add(Box(p - r, p + r), m, part, big=False)
    return pts


def arc_band(R_in, R_out, a0, a1, n=28):
    """Annular sector polygon in a plane (angles in degrees from +v, towards +u)."""
    angs = np.radians(np.linspace(a0, a1, n))
    outer = [(R_out * math.sin(a), R_out * math.cos(a)) for a in angs]
    inner = [(R_in * math.sin(a), R_in * math.cos(a)) for a in angs[::-1]]
    return outer + inner


def scale_poly(poly, f, cx=0.0, cy=0.0):
    return [(cx + (x - cx) * f, cy + (y - cy) * f) for x, y in poly]


# ------------------------------------------------------------------------------- wagon
SKULL = [  # T-rex skull, one char = one small cube, top row first (faces +u)
    "......######....",
    "....##########..",
    "...####..#######",
    "..#####..##..###",
    ".###############",
    "################",
    "#####.#.#.#.#.#.",
    "####............",
    "####.#.#.#.#....",
    ".###########....",
    "..#########.....",
]


def banner(vm, cx, side):
    W = "Wagon"
    y0 = side * WY
    ya, yb = sorted((y0, y0 + side * S))
    vm.add(Box((cx - 1.6, ya, ZD - 0.32), (cx + 1.6, yb, ZD)), "metal_dark", W, big=False)    # rod
    torn = [(-1.44, 0), (1.44, 0), (1.44, -3.4), (1.0, -3.9), (0.7, -3.3), (0.3, -4.2), (-0.1, -3.5),
            (-0.5, -4.0), (-0.9, -3.4), (-1.2, -3.8), (-1.44, -3.3)]
    org = (cx, y0 + side * S / 2, ZD - 0.32)
    vm.add(Plate(org, (1, 0, 0), (0, 0, 1), torn, S), "banner", W, big=False)
    vm.paint(Plate(org, (1, 0, 0), (0, 0, 1), [(-1.44, -0.32), (1.44, -0.32), (1.44, 0), (-1.44, 0)], S * 1.2),
             "banner_dark", parts=[W])
    w = len(SKULL[0]) * S
    u = (1, 0, 0)   # the skull looks toward the front of the train on both sides
    ox = cx - w / 2
    vm.paint_bitmap((ox, y0 + side * S, 3.2), u, (0, 0, 1), SKULL, {"#": "bone"}, depth=1,
                    normal=(0, side, 0), parts=[W])


def railing(vm, x0, x1, y, h=1.6):
    W = "Wagon"
    n = max(2, int(round((x1 - x0) / 1.44)) + 1)
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        xs = math.floor(x / S + 0.5) * S
        vm.add(Box((xs - 0.16, y - 0.16, ZD), (xs + 0.16, y + 0.16, ZD + h)), "rail", W, big=False)
    vm.add(Box((x0 - 0.16, y - 0.16, ZD + h - 0.32), (x1 + 0.16, y + 0.16, ZD + h)), "rail", W, big=False)
    vm.add(Box((x0, y - 0.16, ZD + 0.64), (x1, y + 0.16, ZD + 0.80)), "rail", W, big=False)


def crate(vm, lo, hi):
    W = "Wagon"
    vm.add(Box(lo, hi), "wood", W)
    lo, hi = np.array(lo, float), np.array(hi, float)
    e = 0.16
    # dark frame on the edges + X brace painted on the faces
    # edge = within one small cube of the boundary along at least two axes
    frame = Box(lo, hi) - (Box(lo + [e, e, -1], hi - [e, e, -1]) | Box(lo + [-1, e, e], hi - [-1, e, e])
                           | Box(lo + [e, -1, e], hi - [e, -1, e]))
    vm.paint(frame, "wood_dark", parts=[W])
    c = (lo + hi) / 2
    sx, sz = hi[0] - lo[0], hi[2] - lo[2]

    def brace(X, Y, Z):
        a = (X - c[0]) / sx
        b = (Z - c[2]) / sz
        return (np.abs(np.abs(a) - np.abs(b)) < 0.07)
    vm.paint(Box(lo, hi), "wood_dark", parts=[W], where=brace)


def wagon(vm):
    W = "Wagon"
    vm.add(Box((WX0, -WY, Z0), (WX1, WY, ZD)), "metal", W)
    vm.paint(Box((WX0, -WY, ZD - 2 * B), (WX1, WY, ZD)), "hazard", parts=[W])
    vm.paint(Box((WX0, -WY + 0.5, ZD - 0.2), (WX1, WY - 0.5, ZD + 0.1)), "metal", parts=[W])   # deck top stays metal
    vm.paint(Box((WX0, -WY, Z0), (WX1, WY, Z0 + B)), "metal_dark", parts=[W])                 # skirt
    # deck edge lip
    lip = Box((WX0, -WY, ZD), (WX1, WY, ZD + S)) - Box((WX0 + 0.32, -WY + 0.32, ZD - 1), (WX1 - 0.32, WY - 0.32, ZD + 1))
    vm.add(lip, "metal_dark", W, big=False)
    # side panels set back one small cube under the hazard band
    for a, b in ((-18.72, -12.48), (-11.52, -6.24), (-5.28, 0.96), (1.92, 8.16), (9.12, 15.84)):
        vm.cut(sym(Box((a, WY - S, Z0 + B), (b, WY + 0.01, ZD - 2 * B))))
    # front cab with lights
    vm.add(Box((10.56, -3.84, ZD), (16.32, 3.84, 8.16)), "metal", W)
    vm.add(Box((10.08, -4.32, 8.16), (16.80, 4.32, 8.64)), "metal_dark", W)
    for i, y in enumerate((-1.6, 0.0, 1.6)):
        neon(f"CabLight{i + 1}", W, (16.40, y, 7.2), (0.16, 1.12, 0.64))
    vm.cut(Box((16.16, -2.32, 6.80), (16.33, 2.32, 7.60)))
    for side in (-1, 1):
        for i, x in enumerate((11.68, 13.12, 14.56)):
            neon(f"CabSide{'L' if side > 0 else 'R'}{i + 1}", W, (x, side * 3.76, 7.2), (0.96, 0.16, 0.64))
        vm.cut(Box((11.2, side * 3.68 - 0.17, 6.88), (15.04, side * 3.68 + 0.17, 7.52)) if False else
               Box((11.2, min(side * 3.68, side * 3.85), 6.88), (15.04, max(side * 3.68, side * 3.85), 7.52)))
    # front bumper with bone tusks (cow-catcher) and horns
    vm.add(Box((16.80, -4.32, Z0), (19.20, 4.32, 4.80)), "metal_dark", W)
    vm.add(Box((16.80, -3.36, 4.80), (18.72, 3.36, ZD)), "metal", W)
    for i, y in enumerate((-1.2, 1.2)):
        neon(f"BumperLight{i + 1}", W, (18.80, y, 5.28), (0.16, 1.28, 0.48))
    vm.cut(Box((18.56, -2.0, 5.04), (18.73, 2.0, 5.52)))
    for y in (-3.0, -1.1, 1.1, 3.0):
        horn(vm, (18.9, y, 3.4), (20.6, y, 3.5), (21.9, y, 2.3), 0.40, 0.10, "bone", W, n=20)
    for side in (-1, 1):
        horn(vm, (18.0, side * 3.6, 5.5), (18.8, side * 3.8, 6.8), (19.6, side * 4.0, 8.1), 0.38, 0.10, "bone", W, n=18)
    # wheels, bogies
    for x in (-15.36, -10.56, 3.84, 8.64):
        vm.add(sym(Cylinder((x, 3.36, 1.92), (x, 4.32, 1.92), 1.92)), "wheel", W)
        ring = Cylinder((x, 4.16, 1.92), (x, 4.33, 1.92), 1.93) - Cylinder((x, 4.0, 1.92), (x, 4.5, 1.92), 1.45)
        vm.paint(sym(ring), "wheel_rim", parts=[W])
        vm.add(sym(Cylinder((x, 4.32, 1.92), (x, 4.48, 1.92), 0.48)), "hub", W, big=False)
    for a, b in ((-15.36, -10.56), (3.84, 8.64)):
        vm.add(sym(Box((a, 2.40, 1.44), (b, 3.36, 2.40))), "metal_dark", W)
    # side lights, claw scratches
    for side in (-1, 1):
        for i, x in enumerate((-9.6, -8.16)):
            neon(f"SideLight{'L' if side > 0 else 'R'}{i + 1}", W, (x, side * (WY - 0.08), 3.84), (0.96, 0.16, 0.48))
        for i, x in enumerate((5.28, 6.72)):
            neon(f"SideLightF{'L' if side > 0 else 'R'}{i + 1}", W, (x, side * (WY - 0.08), 2.88), (0.96, 0.16, 0.48))
    vm.cut(sym(Box((-10.24, WY - 0.33, 3.60), (-7.52, WY + 0.01, 4.08))))
    vm.cut(sym(Box((4.64, WY - 0.33, 2.64), (7.36, WY + 0.01, 3.12))))

    def claws(X, Y, Z):
        out = np.zeros(X.shape, bool)
        for k in range(4):
            xo = -3.4 + k * 0.72
            d = (X - xo) - (Z - 2.9) * 0.62
            out |= (np.abs(d) < 0.14) & (Z > 2.9 + k * 0.08) & (Z < 4.64 - k * 0.1)
        return out
    # claw marks gouged one small cube deep into the panel, glowing at the bottom
    from pipeline.vox import Func
    for side in (-1, 1):
        ya, yb = sorted((side * (WY - 2 * S), side * (WY - S)))
        gouge = Func(lambda P: claws(P[:, 0], P[:, 1], P[:, 2]), (-4.6, ya, 2.8), (0.2, yb, 4.7))
        vm.cut(gouge)
        ya2, yb2 = sorted((side * (WY - 3 * S), side * (WY - 2 * S)))
        vm.paint(Box((-4.6, ya2, 2.8), (0.2, yb2, 4.7)), "claw", parts=[W], where=claws)
    # ladder on the chassis side, by the rear railing
    for side in (-1, 1):
        ya, yb = sorted((side * WY, side * (WY + S)))
        for x in (-12.48, -11.68):
            vm.add(Box((x, ya, Z0 + 0.32), (x + 0.16, yb, ZD + 0.16)), "rail", W, big=False)
        for z in np.arange(Z0 + 0.64, ZD, 0.48):
            vm.add(Box((-12.32, ya, z), (-11.68, yb, z + 0.16)), "rail", W, big=False)
    # railings, crates, banners
    for side in (-1, 1):
        railing(vm, -19.2, -13.0, side * (WY - 0.24))
        railing(vm, 1.6, 9.6, side * (WY - 0.24))
        banner(vm, -15.6, side)
        banner(vm, 3.4, side)
    crate(vm, (-19.20, -4.32, ZD), (-17.28, -2.40, ZD + 1.92))
    crate(vm, (-19.20, 2.40, ZD), (-17.28, 4.32, ZD + 1.92))
    for side in (-1, 1):
        ya, yb = sorted((side * (WY - S), side * (WY + 0.48)))
        vm.add(Box((-1.92, ya, Z0 + 0.48), (0.48, yb, 4.80)), "wood", W, big=False)
        crate_frame = Box((-1.92, ya, Z0 + 0.48), (0.48, yb, 4.80))
        vm.paint(crate_frame - Box((-1.76, ya - 1, Z0 + 0.64), (0.32, yb + 1, 4.64)), "wood_dark", parts=[W])
    LIGHTS.append(dict(name="CabGlow", part=W, position=(16.8, 0, 7.2), color=C_LIGHT, range=10, brightness=1.5))


# ------------------------------------------------------------------------------ dragon
SPIKES = [  # x, height, half width of the base
    (-11.8, 3.6, 1.45), (-9.4, 4.6, 1.6), (-6.9, 5.3, 1.7), (-4.4, 5.6, 1.75),
    (-1.9, 5.4, 1.7), (0.6, 4.8, 1.6), (3.0, 4.0, 1.45), (5.2, 3.0, 1.2),
]


def spike(vm, x, zb, h, hw, part, thick=0.96):
    """Flame spike: dark red blade curving back, glowing core (geometry in big cubes, paint fine)."""
    poly = [(-hw, 0), (hw, 0), (hw * 0.75, 0.3 * h), (hw * 0.25, 0.62 * h), (-hw * 0.35, 0.9 * h),
            (-hw * 0.95, h), (-hw * 0.7, 0.66 * h), (-hw * 0.95, 0.3 * h)]
    o = (x, 0, zb)
    vm.add(Plate(o, (1, 0, 0), (0, 0, 1), poly, thick), "red_dark", part, big=True)
    vm.add(Plate(o, (1, 0, 0), (0, 0, 1), scale_poly(poly, 0.6, 0, 0), thick * 2), "red_dark", part, big=True)
    flame = scale_poly(poly, 0.72, -hw * 0.05, -0.1)
    core = scale_poly(poly, 0.42, -hw * 0.05, -0.2)
    vm.paint(Plate(o, (1, 0, 0), (0, 0, 1), flame, 2.0), "flame", parts=[part])
    vm.paint(Plate(o, (1, 0, 0), (0, 0, 1), core, 2.0), "flame_hot", parts=[part])
    tipz = zb + 0.82 * h
    vm.paint(Box((x - hw, -1, tipz), (x + hw, 1, zb + h + 0.5)), "char", parts=[part], only=["red_dark"])


def body(vm):
    P, L = "Body", "Lava"
    cx, cz, r = -5.5, 12.0, 3.12
    vm.add(Cylinder((-12.0, 0, cz), (1.0, 0, cz), r), "lava", L)
    for xa, xb in ((-12.96, -12.0), (0.96, 1.92)):
        vm.add(Cylinder((xa, 0, cz), (xb, 0, cz), 3.6), "metal_dark", P)
        vm.add(Cylinder((xa - 0.16, 0, cz), (xb + 0.16, 0, cz), 2.4), "metal", P, big=False)
    # tank cradle on the deck
    vm.add(Box((-12.48, -2.88, ZD), (1.44, 2.88, 9.12)), "metal_dark", P)
    for xa in (-11.04, -5.76, -0.48):
        vm.add(Box((xa, -3.36, ZD), (xa + 0.96, 3.36, 9.6)), "metal", P)
    # bone ribs over the tank
    for x in (-10.08, -7.44, -4.80, -2.16):
        band = arc_band(r - 0.2, r + 0.9, -152, 152)
        vm.add(Plate((x, 0, cz), (0, 1, 0), (0, 0, 1), band, 0.96), "bone", P, big=True)
        vm.paint(Plate((x, 0, cz), (0, 1, 0), (0, 0, 1), arc_band(r + 0.62, r + 1.0, -152, 152), 1.2), "bone_dark", parts=[P])
    # armoured back over the tank (the ribs come out from under it), spikes on it
    saddle = Ellipsoid((-3.6, 0, 14.6), (10.2, 3.0, 2.3)) & Box((-30, -10, 13.92), (30, 10, 40))
    vm.add(saddle, "red_dark", P)
    vm.paint(Box((-14, -1.0, 15.8), (7, 1.0, 18)), "char", parts=[P], only=["red_dark"])
    vm.paint(Box((-14, -3.2, 13.9), (7, 3.2, 14.9)), "char", parts=[P], only=["red_dark"])
    for (x, h, hw) in SPIKES:
        spike(vm, x, 15.84, h, hw, P)
    # chest / shoulders (front of the body), armour plate and a glowing vent
    ch = Ellipsoid((4.6, 0, 11.8), (3.7, 3.3, 3.7)) & Box((-30, -10, ZD), (30, 10, 40))
    vm.add(ch, "red_dark", P)
    vm.paint(Box((1.0, -3.6, 5.5), (9.0, 3.6, 9.6)), "char", parts=[P], only=["red_dark"])
    vm.add(sym(Box((1.44, 2.88, 10.08), (4.32, 3.36, 13.44))), "char", P)
    vm.cut(sym(Box((2.24, 3.20, 11.20), (3.52, 3.37, 12.32))))
    for side in (-1, 1):
        neon(f"Vent{'L' if side > 0 else 'R'}", P, (2.88, side * 3.24, 11.76), (1.28, 0.16, 1.12), color=C_VENT)
    # glowing pipes from the tank collar down to the cab, held by dark brackets
    pipe = [(1.6, 3.55, 13.4), (3.2, 3.65, 13.4), (4.6, 3.75, 11.2), (6.4, 3.75, 10.8), (7.6, 3.75, 8.6), (10.2, 3.75, 8.4)]
    for side in (-1, 1):
        for j in range(len(pipe) - 1):
            a, b = np.array(pipe[j]), np.array(pipe[j + 1])
            a[1] *= side
            b[1] *= side
            t = b - a
            L_ = np.linalg.norm(t)
            t /= L_
            up = np.array([0, 0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0, 0])
            yv = np.cross(up, t)
            yv /= np.linalg.norm(yv)
            zv = np.cross(t, yv)
            R = np.column_stack([t, yv, zv])
            neon(f"Pipe{'L' if side > 0 else 'R'}{j + 1}", P, (a + b) / 2, (L_ + 0.48, 0.52, 0.52), R=R, color=C_PIPE)
        for (x, z) in ((3.2, 13.4), (4.6, 11.2), (10.0, 8.4)):
            ya, yb = sorted((side * 2.4, side * 4.16))
            vm.add(Box((x - 0.32, ya, z - 0.32), (x + 0.32, yb, z + 0.32)), "metal_dark", P, big=False)
    LIGHTS.append(dict(name="LavaGlow", part=L, position=(cx, 0, cz), color=(255, 130, 40), range=16, brightness=2))
    # tail: rises above the crates then sweeps down, spikes on top (dark red, then bone near the tip)
    tail = [(-12.4, 0, 13.4), (-15.0, 0, 14.4), (-17.6, 0, 14.0), (-19.6, 0, 12.4), (-21.0, 0, 10.6), (-21.9, 0, 9.4)]
    tr = [2.0, 1.7, 1.35, 1.0, 0.65, 0.38]
    vm.add(Scale(Sweep(tail[:4], tr[:4]), (0, 0, 0), (1, 0.85, 1)), "red", "Tail1", big=True)
    vm.add(Scale(Sweep(tail[3:], tr[3:]), (0, 0, 0), (1, 0.85, 1)), "red", "Tail1", big=False)
    tx_, tz_ = [p_[0] for p_ in tail][::-1], [p_[2] for p_ in tail][::-1]
    trr = tr[::-1]
    vm.paint(Box((-23, -2.2, 6), (-11.5, 2.2, 17)), "red_dark", parts=["Tail1"],
             where=lambda X, Y, Z: Z < np.interp(X, tx_, tz_) - 0.35 * np.interp(X, tx_, trr))
    vm.setpart(Box((-30, -10, 0), (-16.32, 10, 40)), "Tail2", parts=["Tail1"])
    vm.setpart(Box((-30, -10, 0), (-19.2, 10, 40)), "Tail3", parts=["Tail2"])
    for (x, h, hw, part, m) in ((-13.4, 2.3, 0.66, "Tail1", "red_dark"), (-14.9, 2.1, 0.6, "Tail1", "red_dark"),
                                (-16.6, 1.9, 0.55, "Tail2", "red_dark"), (-18.0, 1.6, 0.5, "Tail2", "bone"),
                                (-19.4, 1.35, 0.44, "Tail3", "bone"), (-20.5, 1.1, 0.38, "Tail3", "bone"),
                                (-21.4, 0.8, 0.3, "Tail3", "bone")):
        ztop = np.interp(x, tx_, tz_) + 0.8 * np.interp(x, tx_, trr) - 0.25
        poly = [(-hw, 0), (hw, 0), (0.0, 0.6 * h), (-hw * 1.2, h)]
        vm.add(Plate((x, 0, ztop), (1, 0, 0), (0, 0, 1), poly, 0.48), m, part, big=False)


def tooth(vm, x0, ya, yb, zedge, rows, part, down=True):
    """Triangular tooth: one row of small cubes per step, widths in cells (odd), narrowing to the tip."""
    x0 = math.floor(x0 / S + 0.5) * S
    xc = x0 + 1.5 * S
    for i, w in enumerate(rows):
        z = zedge - (i + 1) * S if down else zedge + i * S
        vm.add(Box((xc - w * S / 2, ya, z), (xc + w * S / 2, yb, z + S)), "tooth", part, big=False)


JAW_TOP_X = [11.6, 14.0, 16.4, 18.6, 19.6]
JAW_TOP_Z = [13.76, 13.36, 12.72, 12.0, 11.68]


def neck_head(vm):
    N, H, J = "Neck", "Head", "Jaw"
    # neck from the chest up to the back of the head, pale throat underneath
    neck = [(5.6, 0, 12.4), (7.6, 0, 13.6), (9.6, 0, 14.6), (11.4, 0, 15.4)]
    vm.add(Scale(Sweep(neck, [2.7, 2.5, 2.35, 2.25]), (0, 0, 0), (1, 0.88, 1)), "red", N)
    vm.setpart(Box((-30, -10, 0), (7.2, 10, 40)), "Body", parts=[N])
    vm.paint(Box((5, -3, 9), (12.5, 3, 15)), "bone", parts=[N], only=["red"],
             where=lambda X, Y, Z: Z < 12.4 + 0.517 * (X - 5.6) - 1.25)
    for (x, z, h, hw) in ((7.8, 15.6, 2.1, 0.62), (9.6, 16.4, 1.9, 0.58)):
        poly = [(-hw, 0), (hw, 0), (0.0, 0.6 * h), (-hw * 1.15, h)]
        vm.add(Plate((x, 0, z), (1, 0, 0), (0, 0, 1), poly, 0.48), "char", N, big=False)
    # ---- head: big skull (big cubes), short tall snout sculpted in small cubes, open jaw
    vm.setpart(Box((10.56, -10, 0), (30, 10, 40)), H, parts=[N])
    skull = Box((10.56, -2.40, 13.92), (15.84, 2.40, 17.76)) & Ellipsoid((13.2, 0, 15.8), (3.7, 2.9, 2.95))
    vm.add(skull, "red", H)
    o, u, v = (0, 0, 0), (1, 0, 0), (0, 0, 1)
    # snout slopes down from the brow to the nose
    vm.add(Plate(o, u, v, [(15.68, 14.08), (17.76, 14.08), (17.76, 16.96), (15.68, 17.44)], 3.52), "red", H, big=False)
    vm.add(Plate(o, u, v, [(17.6, 14.08), (19.52, 14.08), (19.52, 16.32), (17.6, 16.96)], 3.20), "red", H, big=False)
    vm.add(Plate(o, u, v, [(19.36, 14.08), (20.80, 14.08), (20.80, 15.52), (20.48, 15.84), (19.36, 16.32)], 2.88),
           "red", H, big=False)
    vm.paint(Box((15.5, -0.66, 15.9), (21.0, 0.66, 18.0)), "red_dark", parts=[H],
             where=lambda X, Y, Z: Z > 17.44 - 0.33 * (X - 15.68) - 0.4)                       # nose ridge
    vm.cut(sym(Box((20.16, 0.32, 15.52), (20.48, 0.64, 15.85))))                               # nostrils
    vm.paint(sym(Box((20.0, 0.16, 15.3), (20.7, 0.8, 16.0))), "char", parts=[H], only=["red", "red_dark"],
             where=lambda X, Y, Z: (np.abs(Y) > 0.3) & (np.abs(Y) < 0.66))
    # palate (dark) under the upper jaw
    vm.paint(Box((13.4, -1.6, 13.9), (21.0, 1.6, 14.24)), "mouth", parts=[H])
    # upper teeth: two triangular fangs and smaller teeth, outside the lower jaw
    hw_up = lambda x: 1.76 if x < 17.6 else (1.6 if x < 19.36 else 1.44)
    for x0, rows in ((19.84, [3, 3, 1, 1, 1]), (19.04, [1, 1]), (18.40, [3, 1, 1]), (17.76, [1, 1]),
                     (16.96, [3, 3, 1, 1, 1]), (16.32, [1, 1]), (15.68, [3, 1])):
        hw = hw_up(x0 + 0.24)
        for side in (-1, 1):
            ya, yb = sorted((side * (hw - 0.16), side * hw))
            tooth(vm, x0, ya, yb, 14.08, rows, H)
    # eyes: notch at the front-side corner of the skull, neon at the bottom, heavy brow
    vm.cut(sym(Box((14.40, 1.60, 15.52), (15.85, 2.41, 16.48))))
    vm.paint(sym(Box((14.2, 1.40, 15.3), (14.42, 2.45, 16.56))), "socket", parts=[H])
    vm.paint(sym(Box((14.2, 1.40, 15.3), (15.9, 1.62, 16.56))), "socket", parts=[H])
    vm.paint(sym(Box((14.2, 1.40, 15.3), (15.9, 2.45, 15.54))), "socket", parts=[H])
    for side in (-1, 1):
        neon(f"Eye{'L' if side > 0 else 'R'}", H, (15.12, side * 1.92, 16.0), (0.80, 0.32, 0.64), color=C_EYE)
    vm.add(sym(Box((13.44, 1.28, 16.48), (16.00, 2.56, 17.12))), "red_dark", H, big=False)
    vm.add(sym(Box((14.40, 1.60, 17.12), (15.84, 2.40, 17.44))), "char", H, big=False)
    # crown of swept-back horns, dorsal spikes down the head and neck, cheek spikes
    for side in (-1, 1):
        horn(vm, (13.0, side * 1.4, 17.6), (11.8, side * 1.8, 19.1), (9.8, side * 2.2, 19.7), 0.50, 0.12, "char", H, n=24)
        horn(vm, (11.8, side * 2.1, 17.2), (10.4, side * 2.6, 18.2), (8.8, side * 3.0, 18.4), 0.42, 0.11, "char", H, n=20)
        horn(vm, (11.6, side * 2.2, 14.8), (10.2, side * 2.8, 15.0), (8.9, side * 3.2, 15.8), 0.36, 0.10, "red_dark", H, n=18)
    for (x, h) in ((14.4, 1.0), (13.2, 1.4), (12.0, 1.7), (10.9, 1.6)):
        poly = [(-0.5, 0), (0.5, 0), (0.0, 0.6 * h), (-0.65, h)]
        vm.add(Plate((x, 0, 17.5 if x > 12 else 17.4), (1, 0, 0), (0, 0, 1), poly, 0.48), "char", H, big=False)
    # lower jaw, hanging open: deep, shorter, stepped profile (axis-aligned, keeps its studs)
    top = list(zip(JAW_TOP_X, JAW_TOP_Z))
    bottom = [(19.6, 10.88), (18.4, 10.4), (16.0, 10.56), (13.6, 11.36), (11.4, 12.32)]
    vm.add(Plate(o, u, v, top + bottom, 3.20), "red", J, big=False)
    vm.add(Plate(o, u, v, [(11.4, 12.32), (13.6, 11.36), (16.0, 10.56), (18.4, 10.4), (19.6, 10.88),
                           (19.6, 11.2), (11.4, 12.9)], 3.52), "red", J, big=False)
    vm.paint(Box((11.0, -2.0, 9.9), (20.0, 2.0, 13.0)), "bone", parts=[J],
             where=lambda X, Y, Z: Z < 12.9 - 0.207 * (X - 11.4))                      # pale underside
    vm.paint(Box((11.4, -1.3, 10.6), (19.7, 1.3, 14.0)), "mouth", parts=[J],
             where=lambda X, Y, Z: Z > np.interp(X, JAW_TOP_X, JAW_TOP_Z) - 0.36)      # tongue
    for x0, rows in ((18.72, [3, 3, 1, 1]), (17.76, [1, 1]), (16.80, [3, 1, 1]), (15.84, [1, 1]), (14.88, [3, 1])):
        zt = math.floor(np.interp(x0 + 0.24, JAW_TOP_X, JAW_TOP_Z) / S) * S
        for side in (-1, 1):
            ya, yb = sorted((side * 1.28, side * 1.44))
            vm.add(Box((x0, ya, zt - 0.32), (x0 + 0.48, yb, zt)), "tooth", J, big=False)   # root in the gum
            tooth(vm, x0, ya, yb, zt, rows, J, down=False)
    ATTACHMENTS.append(dict(name="FireBreath", part=H, position=(20.6, 0, 13.0), direction=(1, 0, -0.15)))
    LIGHTS.append(dict(name="MouthGlow", part=H, position=(18.6, 0, 13.0), color=(255, 150, 50), range=8, brightness=1.5))


def make():
    NEON.clear()
    ATTACHMENTS.clear()
    LIGHTS.clear()
    vm = VoxelModel((-22.56, -6.24, -0.48), (23.04, 6.24, 21.6))
    wagon(vm)
    body(vm)
    neck_head(vm)
    return vm
