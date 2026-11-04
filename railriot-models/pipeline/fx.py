"""Procedural paint (texel level).  Signature: fx(col, vox, d, pu, pv) -> col
col: (N,3) float RGB, vox: (N,3) voxel indices of the texel's cell, d: face
direction index (0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z), pu/pv: texel position
inside the small-cube face (0..5).  Detail is painted, never built."""
import numpy as np

from .texture import hash01, PX


def wood(col, vox, d, pu, pv):
    col = col.copy()
    side = d < 4
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    along = np.where(d < 2, y, x)  # in-plane horizontal index
    # horizontal planks: one per big cube row, dark seam at the bottom of each plank
    row = z // 3
    seam = side & (z % 3 == 0) & (pv == 0)
    joint = side & (((along + row * 7) % 15) == 0) & (pu == 0)
    grain = 1.0 + 0.045 * np.sin((along * PX + pu) * 0.55 + row * 2.3) * np.sin(row * 1.7 + 0.5)
    # top faces: planks along X
    top = d == 4
    trow = y // 3
    tseam = top & (y % 3 == 0) & (pv == 0)
    tjoint = top & (((x + trow * 5) % 15) == 0) & (pu == 0)
    tgrain = 1.0 + 0.04 * np.sin((x * PX + pu) * 0.5 + trow * 1.9)
    g = np.where(side, grain, np.where(top, tgrain, 1.0))
    col *= g[:, None]
    dark = seam | joint | tseam | tjoint
    col[dark] *= 0.62
    return col


def lava(col, vox, d, pu, pv):
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    n = 0.55 * hash01(x // 3, y // 3, z // 3, seed=11) + 0.45 * hash01(x, y, z, seed=12)
    out = np.empty_like(col)
    dark = np.array([196, 52, 18.0])
    mid = np.array([244, 118, 24.0])
    hot = np.array([255, 196, 60.0])
    out[:] = mid
    out[n < 0.34] = dark
    out[n > 0.70] = hot
    return out


def hazard(col, vox, d, pu, pv):
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    along = np.where(d < 2, -y, np.where(d == 2, -x, x))
    t = ((along + z) // 3) % 2
    out = np.where(t[:, None] == 0, np.array([[232, 168, 34.0]]), np.array([[34, 30, 34.0]]))
    return out * (col / np.maximum(col.mean(1, keepdims=True), 1)).clip(0.9, 1.1)


def scales(col, vox, d, pu, pv):
    """Overlapping scale rows: a darker lower rim on every small cube, staggered."""
    col = col.copy()
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    side = d < 4
    stagger = ((z + (x + y) // 2) % 2) == 0
    rim = side & (pv == 0) & stagger
    hi = side & (pv == PX - 1) & stagger
    col[rim] *= 0.84
    col[hi] *= 1.08
    return col


def metal(col, vox, d, pu, pv):
    """Plates of 3x3 big cubes with a faint seam."""
    col = col.copy()
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    along = np.where(d < 2, y, x)
    vert = np.where(d < 4, z, y)
    seam = ((along % 9 == 0) & (pu == 0)) | ((vert % 9 == 0) & (pv == 0))
    col[seam] *= 0.78
    return col


def scales2(col, vox, d, pu, pv):
    """Overlapping scales, 2 x 2 small cubes each, staggered rows: light upper edge, dark lower rim."""
    col = col.copy()
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    side = d < 4
    along = np.where(d < 2, y, x)
    # side faces: rows go up the body; top faces: rows run along X
    row = np.where(side, z // 2, x // 2)
    a = np.where(side, along, y) + (row % 2)
    vcell = np.where(side, z, x) % 2
    u = (a % 2) * PX + pu
    v = vcell * PX + pv
    f = np.ones(len(col))
    f += 0.07 * (v / (2 * PX - 1) - 0.5) * 2               # brighter toward the overlap
    f[(v < 2)] *= 0.82                                      # shadow under the scale
    f[(v < 4) & ((u < 2) | (u > 2 * PX - 3))] *= 0.90       # rounded lower corners
    f[(v > 2 * PX - 3) & (u > 2) & (u < 2 * PX - 3)] *= 1.06
    f[d == 5] = 1.0
    return col * f[:, None]


FX = dict(wood=wood, lava=lava, hazard=hazard, scales=scales, scales2=scales2, metal=metal)
