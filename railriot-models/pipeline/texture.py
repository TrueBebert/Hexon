"""Texture atlas for voxel meshes.

Every exposed small-cube face gets PX x PX texels of its own (PX = 6 -> 37.5 px
per stud with 0.16 small cubes).  Per texel:
  material colour (+ per-big-cube jitter) -> procedural paint (wood, lava, ...)
  -> orientation shade (top lighter, bottom darker) -> ambient occlusion
  (computed by ray marching the voxel grid, smoothed between cells, crevices
  tinted blue-violet) -> native Roblox studs on complete flat big-cube squares
  -> light lines on convex edges.
"""
import numpy as np
from PIL import Image

from .mesh import DIRS, INPLANE, unit

PX = 6
OUTLINE_RGB = (16, 14, 20)
JUNCTION_RGB = (34, 30, 40)
SHADE = np.array([0.93, 0.93, 0.97, 0.97, 1.07, 0.72])  # +x -x +y -y +z -z
CREVICE_TINT = np.array([0.50, 0.46, 0.78])


def hash01(a, b, c, seed=0):
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    c = np.asarray(c, np.int64)
    with np.errstate(over="ignore"):
        h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791) ^ np.int64(seed * 2654435761 % (2 ** 31))
        h = (h ^ (h >> 13)) * np.int64(1274126177)
        h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 65535.0


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------
class Cells:
    """One record per exposed small-cube face of every quad."""

    def __init__(self, quads):
        # quads: (nq, 6) int array of (d, plane, u0, v0, du, dv)
        Q = np.asarray(quads, np.int64).reshape(-1, 6)
        self.Q = Q
        d, plane, u0, v0, du, dv = Q.T
        counts = du * dv
        self.qbase = np.concatenate([[0], np.cumsum(counts)[:-1]])
        N = int(counts.sum())
        qi = np.repeat(np.arange(len(Q)), counts)
        local = np.arange(N) - self.qbase[qi]
        cu = local // dv[qi]
        cv = local % dv[qi]
        self.qi, self.cu, self.cv = qi, cu, cv
        self.d = d[qi]
        axis = np.array([DIRS[i][0] for i in range(6)])[self.d]
        sign = np.array([DIRS[i][1] for i in range(6)])[self.d]
        k = plane[qi] - (sign > 0)
        uu = u0[qi] + cu
        vv = v0[qi] + cv
        vox = np.zeros((N, 3), np.int64)
        for a in range(3):
            U, V = INPLANE[a]
            sel = axis == a
            vox[sel, a] = k[sel]
            vox[sel, U] = uu[sel]
            vox[sel, V] = vv[sel]
        self.vox = vox
        self.u, self.v = uu, vv
        self.N = N


def hemisphere(n=36, seed=7):
    rng = np.random.default_rng(seed)
    k = int(round(np.sqrt(n)))
    out = []
    for i in range(k):
        for j in range(k):
            u1 = (i + rng.random()) / k
            u2 = (j + rng.random()) / k
            r = np.sqrt(u1)
            ph = 2 * np.pi * u2
            out.append((r * np.cos(ph), r * np.sin(ph), np.sqrt(max(0.0, 1 - u1))))
    return np.array(out)


def compute_ao(occ, cells, steps=(0.55, 1.0, 1.5, 2.1, 2.8, 3.6, 4.6, 5.8, 7.2), strength=0.80, chunk=3000):
    dims = np.array(occ.shape)
    tmax = steps[-1] * 1.15
    H = hemisphere()
    ao = np.ones(cells.N)
    for d in range(6):
        sel = np.nonzero(cells.d == d)[0]
        if len(sel) == 0:
            continue
        axis, sign = DIRS[d]
        U, V = INPLANE[axis]
        n, t1, t2 = unit(axis, sign), unit(U), unit(V)
        dirs = H[:, 0:1] * t1 + H[:, 1:2] * t2 + H[:, 2:3] * n
        for s in range(0, len(sel), chunk):
            idx = sel[s:s + chunk]
            p0 = cells.vox[idx] + 0.5 + n * 0.52
            hit = np.full((len(idx), len(dirs)), np.inf)
            for t in steps:
                q = np.floor(p0[:, None, :] + dirs[None, :, :] * t).astype(np.int64)
                inb = np.all((q >= 0) & (q < dims), axis=2)
                qc = np.clip(q, 0, dims - 1)
                o = occ[qc[..., 0], qc[..., 1], qc[..., 2]] & inb
                new = o & np.isinf(hit)
                hit[new] = t
            w = np.where(np.isinf(hit), 0.0, 1.0 - (hit / tmax) ** 2)
            ao[idx] = 1.0 - strength * w.mean(1)
    return ao


def corner_ao(dims, cells, ao):
    """AO at the 4 corners of each cell (avg of the coplanar cells sharing the corner)."""
    C = np.zeros((cells.N, 2, 2))
    for d in range(6):
        sel = np.nonzero(cells.d == d)[0]
        if len(sel) == 0:
            continue
        axis, _ = DIRS[d]
        U, V = INPLANE[axis]
        A = np.full(dims, np.nan, np.float32)
        vx = cells.vox[sel]
        A[vx[:, 0], vx[:, 1], vx[:, 2]] = ao[sel]
        nb = np.full((len(sel), 3, 3), np.nan)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                w = vx + a * unit(U) + b * unit(V)
                ok = np.all((w >= 0) & (w < np.array(dims)), axis=1)
                val = np.full(len(sel), np.nan)
                ww = w[ok]
                val[ok] = A[ww[:, 0], ww[:, 1], ww[:, 2]]
                nb[:, a + 1, b + 1] = val
        for i in (0, 1):
            for j in (0, 1):
                blk = nb[:, i:i + 2, j:j + 2].reshape(len(sel), 4)
                C[sel, i, j] = np.nanmean(blk, axis=1)
    return C


def edge_flags(occ, cells):
    """Convex edge flags for the 4 in-plane sides: order (-U, +U, -V, +V)."""
    dims = np.array(occ.shape)
    F = np.zeros((cells.N, 4), bool)
    for d in range(6):
        sel = np.nonzero(cells.d == d)[0]
        if len(sel) == 0:
            continue
        axis, _ = DIRS[d]
        U, V = INPLANE[axis]
        vx = cells.vox[sel]
        for i, e in enumerate((-unit(U), unit(U), -unit(V), unit(V))):
            w = vx + e
            ok = np.all((w >= 0) & (w < dims), axis=1)
            sol = np.zeros(len(sel), bool)
            ww = w[ok]
            sol[ok] = occ[ww[:, 0], ww[:, 1], ww[:, 2]]
            F[sel, i] = ~sol
    return F


def stud_flags(expo, vm, studded, cells):
    """True when the cell belongs to a complete flat 3x3 big-cube square (not facing down)."""
    out = np.zeros(cells.N, bool)
    stud_mat = np.asarray(studded, bool)[vm.mat]
    for d in range(5):  # never on downward faces (d = 5)
        sel = np.nonzero(cells.d == d)[0]
        if len(sel) == 0:
            continue
        axis, _ = DIRS[d]
        U, V = INPLANE[axis]
        S = expo.E[d] & ~expo.J[d] & stud_mat
        St = np.moveaxis(S, axis, 0)
        k, nu, nv = St.shape
        blk = St.reshape(k, nu // 3, 3, nv // 3, 3).all(axis=(2, 4))
        vx = cells.vox[sel]
        out[sel] = blk[vx[:, axis], vx[:, U] // 3, vx[:, V] // 3]
    return out


# --------------------------------------------------------------------------
# Atlas packing
# --------------------------------------------------------------------------
def shelf_pack(sizes, W=1024, H=1024, reserve=(4, 4)):
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    x, y, shelf = reserve[0], 0, reserve[1]
    pos = [None] * len(sizes)
    for i in order:
        w, h = sizes[i]
        if w > W:
            return None
        if x + w > W:
            y += shelf
            x, shelf = 0, 0
        if y + h > H:
            return None
        pos[i] = (x, y)
        x += w
        shelf = max(shelf, h)
    return pos


def quad_px_size(q):
    return (int(q[4]) * PX + 2, int(q[5]) * PX + 2)


# --------------------------------------------------------------------------
# Texel painting
# --------------------------------------------------------------------------
def load_stud_tile(path, size=3 * PX):
    im = Image.open(path).convert("L")
    t = np.asarray(im, np.float64)
    border = np.concatenate([t[:6].ravel(), t[-6:].ravel(), t[:, :6].ravel(), t[:, -6:].ravel()])
    bg = np.median(border)
    small = np.asarray(im.resize((size, size), Image.LANCZOS), np.float64)
    return small / bg


class Painter:
    def __init__(self, vm, expo, cells, palette, stud_tile, ao, cao, edges, studs, fx=None):
        self.vm, self.expo, self.c = vm, expo, cells
        self.pal = palette
        names = vm.mat_names
        self.rgb = np.array([palette.get(n, {}).get("rgb", (255, 0, 255)) for n in names], np.float64)
        self.glow = np.array([palette.get(n, {}).get("glow", False) for n in names], bool)
        self.jit = np.array([palette.get(n, {}).get("jitter", 0.05) for n in names], np.float64)
        self.fxname = [palette.get(n, {}).get("fx") for n in names]
        self.fx = fx or {}
        self.tile = stud_tile
        self.ao, self.cao, self.edges, self.studs = ao, cao, edges, studs
        v = cells.vox
        self.cmat = vm.mat[v[:, 0], v[:, 1], v[:, 2]]
        jun = np.zeros(cells.N, bool)
        for d in range(6):
            sel = cells.d == d
            vv = v[sel]
            jun[sel] = expo.J[d][vv[:, 0], vv[:, 1], vv[:, 2]]
        self.junction = jun
        h = hash01(v[:, 0] // 3, v[:, 1] // 3, v[:, 2] // 3, seed=3) * 2 - 1
        self.cjit = 1.0 + h * self.jit[self.cmat]

    def paint_page(self, placed, W=1024):
        """placed: list of (quad_index, x, y).  Returns HxWx3 uint8 image."""
        c = self.c
        cell_map = np.full((W, W), -1, np.int64)
        pu_map = np.zeros((W, W), np.int64)
        pv_map = np.zeros((W, W), np.int64)
        for qi, x, y in placed:
            du, dv = int(c.Q[qi, 4]), int(c.Q[qi, 5])
            w, h = du * PX + 2, dv * PX + 2
            cols = np.arange(w) - 1
            colcell = np.clip(cols // PX, 0, du - 1)
            pu = np.clip(cols - PX * colcell, 0, PX - 1)
            rows = np.arange(h) - 1
            rowcell = np.clip(rows // PX, 0, dv - 1)
            pvtop = np.clip(rows - PX * rowcell, 0, PX - 1)
            cv = dv - 1 - rowcell
            pv = PX - 1 - pvtop
            ids = c.qbase[qi] + colcell[None, :] * dv + cv[:, None]
            cell_map[y:y + h, x:x + w] = ids
            pu_map[y:y + h, x:x + w] = pu[None, :]
            pv_map[y:y + h, x:x + w] = pv[:, None]
        img = np.zeros((W, W, 3), np.float64)
        img[:, :] = OUTLINE_RGB
        valid = cell_map >= 0
        cid = cell_map[valid]
        pu = pu_map[valid]
        pv = pv_map[valid]
        img[valid] = self.shade_texels(cid, pu, pv)
        img[0:4, 0:4] = OUTLINE_RGB
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)

    def shade_texels(self, cid, pu, pv):
        c = self.c
        m = self.cmat[cid]
        col = self.rgb[m] * self.cjit[cid][:, None]
        fu = (pu + 0.5) / PX
        fv = (pv + 0.5) / PX
        # procedural paint
        for mi in np.unique(m):
            name = self.fxname[mi]
            if name and name in self.fx:
                sel = m == mi
                col[sel] = self.fx[name](col[sel], c.vox[cid[sel]], c.d[cid[sel]], pu[sel], pv[sel])
        d = c.d[cid]
        glow = self.glow[m]
        sh = np.where(glow, 1.0, SHADE[d])
        col *= sh[:, None]
        # ambient occlusion (bilinear between cell corners)
        C = self.cao[cid]
        a = (C[:, 0, 0] * (1 - fu) * (1 - fv) + C[:, 1, 0] * fu * (1 - fv)
             + C[:, 0, 1] * (1 - fu) * fv + C[:, 1, 1] * fu * fv)
        a = np.where(glow, 1 - 0.35 * (1 - a), a)
        f = a[:, None] + (1 - a[:, None]) * CREVICE_TINT
        col *= f
        # studs
        st = self.studs[cid]
        if st.any():
            uu = c.u[cid[st]] % 3
            vv = c.v[cid[st]] % 3
            tx = uu * PX + pu[st]
            ty = 3 * PX - 1 - (vv * PX + pv[st])
            col[st] *= self.tile[ty, tx][:, None]
        # light lines on convex edges
        E = self.edges[cid]
        e = ((pu == 0) & E[:, 0]) | ((pu == PX - 1) & E[:, 1]) | ((pv == 0) & E[:, 2]) | ((pv == PX - 1) & E[:, 3])
        col[e] = col[e] * 1.13 + 9
        # junction faces
        j = self.junction[cid]
        col[j] = JUNCTION_RGB
        return col


def save_png(img, path, max_bytes=1_000_000):
    """Save RGB PNG; quantise with pngquant if it is bigger than the Roblox-safe size."""
    import os
    import subprocess
    Image.fromarray(img, "RGB").save(path, optimize=True)
    size = os.path.getsize(path)
    if size > max_bytes:
        subprocess.run(["pngquant", "--force", "--skip-if-larger", "--quality=80-100", "--speed", "1",
                        "--nofs", "--output", path, path], check=False)
        size = os.path.getsize(path)
    return size
