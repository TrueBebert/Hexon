"""Greedy meshing of voxel parts + inverted-hull outline offsets.

Faces between voxels of the same rigid group are dropped.  A face touching a
voxel of another (moving) group is kept on both sides and flagged as a
junction (painted dark), so nothing can be seen through when a limb rotates.
"""
import numpy as np

# direction index -> (axis, sign)
DIRS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
INPLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
HANDED = {0: 1, 1: -1, 2: 1}  # sign of (e_U x e_V) . e_axis


def unit(axis, sign=1):
    e = np.zeros(3, int)
    e[axis] = sign
    return e


def shift_get(a, axis, sign, fill=0):
    """out[v] = a[v + sign * e_axis]  (fill outside the grid)."""
    out = np.full_like(a, fill)
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    if sign > 0:
        src[axis], dst[axis] = slice(1, None), slice(0, -1)
    else:
        src[axis], dst[axis] = slice(0, -1), slice(1, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def greedy_rects(mask, maxlen=48):
    """Greedy rectangles over a 2D bool mask. Returns (u0, v0, du, dv) in mask index space."""
    m = mask.copy()
    H, W = m.shape
    rects = []
    us, vs = np.nonzero(m)
    for u, v in zip(us.tolist(), vs.tolist()):
        if not m[u, v]:
            continue
        row = m[u, v:min(W, v + maxlen)]
        w = len(row) if row.all() else int(np.argmin(row))
        h = 1
        while h < maxlen and u + h < H and m[u + h, v:v + w].all():
            h += 1
        m[u:u + h, v:v + w] = False
        rects.append((u, v, h, w))
    return rects


class Exposure:
    """Per-direction exposed-face masks for a voxel model with rigid groups."""

    def __init__(self, vm, part_group):
        # part_group: array part_id -> rigid group id (>0)
        self.vm = vm
        G = np.asarray(part_group, np.int32)[vm.part]
        G[vm.mat == 0] = 0
        self.G = G
        self.E, self.J = [], []
        for (axis, sign) in DIRS:
            nb = shift_get(G, axis, sign, 0)
            e = (G > 0) & (nb != G)
            self.E.append(e)
            self.J.append(e & (nb > 0))


def chunk_quads(expo, chunk_mask, maxlen=48):
    """Quads for one mesh chunk: list of (d, plane, u0, v0, du, dv)."""
    quads = []
    for d, (axis, sign) in enumerate(DIRS):
        M = expo.E[d] & chunk_mask
        if not M.any():
            continue
        Mt = np.moveaxis(M, axis, 0)  # (k, U, V) with U, V in INPLANE order
        ks = np.nonzero(Mt.reshape(Mt.shape[0], -1).any(1))[0]
        for k in ks.tolist():
            for (u0, v0, du, dv) in greedy_rects(Mt[k], maxlen):
                plane = k + (1 if sign > 0 else 0)
                quads.append((d, plane, u0, v0, du, dv))
    return quads


def mirror_quad(q, Ny):
    """Mirror a quad across the plane Y = 0 (grid centred: lattice y <-> Ny - y)."""
    d, plane, u0, v0, du, dv = q
    axis, _ = DIRS[d]
    if axis == 0:   # U = y
        return (d, plane, Ny - (u0 + du), v0, du, dv)
    if axis == 2:   # V = y
        return (d, plane, u0, Ny - (v0 + dv), du, dv)
    return (d ^ 1, Ny - plane, u0, v0, du, dv)  # y faces flip direction


def quad_voxels(q):
    """Voxel indices (n, 3) of the cells of a quad, ordered cu-major."""
    d, plane, u0, v0, du, dv = q
    axis, sign = DIRS[d]
    U, V = INPLANE[axis]
    k = plane - (1 if sign > 0 else 0)
    cu, cv = np.meshgrid(np.arange(du), np.arange(dv), indexing="ij")
    out = np.zeros((du * dv, 3), np.int64)
    out[:, axis] = k
    out[:, U] = u0 + cu.ravel()
    out[:, V] = v0 + cv.ravel()
    return out


def sym_quads(expo, mask, mat, maxlen=48):
    """Quads of the y >= 0 half, mirrored to the other half when the geometry is
    symmetric.  src[i] = index of the quad whose texels quad i reuses (mirrored), or -1."""
    Ny = mask.shape[1]
    h = Ny // 2
    half = mask.copy()
    half[:, :h, :] = False
    lower = mask.copy()
    lower[:, h:, :] = False
    sym_ok = Ny % 2 == 0
    if sym_ok:
        for d, (axis, _) in enumerate(DIRS):
            dm = d ^ 1 if axis == 1 else d
            A = (expo.E[d] & lower)[:, :h, :]
            Bm = np.flip((expo.E[dm] & half)[:, h:, :], axis=1)
            if not np.array_equal(A, Bm):
                sym_ok = False
                break
    qh = chunk_quads(expo, half, maxlen)
    if not sym_ok:
        ql = chunk_quads(expo, lower, maxlen)
        return qh + ql, [-1] * (len(qh) + len(ql))
    quads, src = list(qh), [-1] * len(qh)
    for i, q in enumerate(qh):
        m = mirror_quad(q, Ny)
        a = quad_voxels(q)
        b = a.copy()
        b[:, 1] = Ny - 1 - b[:, 1]
        same = np.array_equal(mat[a[:, 0], a[:, 1], a[:, 2]], mat[b[:, 0], b[:, 1], b[:, 2]])
        quads.append(m)
        src.append(i if same else -1)
    return quads, src


def shell_quads(cells, D, maxlen=128):
    """Outline shell: faces of the dilated set D owned by `cells` that face out of D."""
    quads = []
    for d, (axis, sign) in enumerate(DIRS):
        M = cells & ~shift_get(D, axis, sign, False)
        if not M.any():
            continue
        Mt = np.moveaxis(M, axis, 0)
        ks = np.nonzero(Mt.reshape(Mt.shape[0], -1).any(1))[0]
        for k in ks.tolist():
            for (u0, v0, du, dv) in greedy_rects(Mt[k], maxlen):
                plane = k + (1 if sign > 0 else 0)
                quads.append((d, plane, u0, v0, du, dv))
    return quads


def surface_touch(D):
    """Lattice touch arrays (per direction) for the outer surface of a bool set D."""
    dims = np.array(D.shape)
    L = []
    for d, (axis, sign) in enumerate(DIRS):
        E = D & ~shift_get(D, axis, sign, False)
        T = np.zeros(tuple(dims + 1), bool)
        U, V = INPLANE[axis]
        oa = 1 if sign > 0 else 0
        for du in (0, 1):
            for dv in (0, 1):
                o = [0, 0, 0]
                o[axis], o[U], o[V] = oa, du, dv
                sl = tuple(slice(o[i], o[i] + dims[i]) for i in range(3))
                T[sl] |= E
        L.append(T)
    return L


def quad_corners(q):
    """Lattice coordinates (4x3) of a quad's corners, CCW seen from outside."""
    d, plane, u0, v0, du, dv = q
    axis, sign = DIRS[d]
    U, V = INPLANE[axis]

    def P(u, v):
        p = [0, 0, 0]
        p[axis], p[U], p[V] = plane, u, v
        return p

    c00, c10, c11, c01 = P(u0, v0), P(u0 + du, v0), P(u0 + du, v0 + dv), P(u0, v0 + dv)
    if HANDED[axis] * sign > 0:
        return np.array([c00, c10, c11, c01])
    return np.array([c00, c01, c11, c10])


def lattice_touch(expo, group_mask):
    """For each direction, which lattice points touch an exposed face of this group."""
    dims = np.array(expo.vm.mat.shape)
    L = []
    for d, (axis, sign) in enumerate(DIRS):
        E = expo.E[d] & group_mask
        T = np.zeros(tuple(dims + 1), bool)
        U, V = INPLANE[axis]
        oa = 1 if sign > 0 else 0
        for du in (0, 1):
            for dv in (0, 1):
                o = [0, 0, 0]
                o[axis], o[U], o[V] = oa, du, dv
                sl = tuple(slice(o[i], o[i] + dims[i]) for i in range(3))
                T[sl] |= E
        L.append(T)
    return L


def outline_offsets(L, lattice_pts, T):
    """Offset vectors (in studs) for lattice points: T * sum of touching face normals."""
    off = np.zeros((len(lattice_pts), 3))
    lp = np.asarray(lattice_pts, int)
    for d, (axis, sign) in enumerate(DIRS):
        hit = L[d][lp[:, 0], lp[:, 1], lp[:, 2]]
        off[hit, axis] += sign * T
    return off
