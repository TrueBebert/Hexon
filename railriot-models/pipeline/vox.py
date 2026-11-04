"""Voxel construction for Rail Riot models.

Shapes are described analytically, then sampled on two aligned grids:
  * big grid   (gros cubes)   : cell = 3 x small cell
  * small grid (petits cubes) : cell = small
A cell is filled when its centre is inside the shape (with a small epsilon so a
centre lying exactly on a surface counts the same way everywhere).

Units: 1 unit = 1 Roblox stud.  Axes: X = length (front of the train = +X),
Y = width, Z = up.  Wheels touch the ground at Z = 0.
"""
import math
import numpy as np
from scipy import ndimage

EPS = 1e-4


def snap(x, cell):
    """Round half up onto a grid (never Python's round(): it rounds .5 to even)."""
    return math.floor(x / cell + 0.5) * cell


def rot(rx=0.0, ry=0.0, rz=0.0):
    """Rotation matrix (degrees). world = Rz @ Ry @ Rx @ local."""
    ax, ay, az = (math.radians(a) for a in (rx, ry, rz))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def v3(v):
    return np.asarray(v, dtype=np.float64).reshape(3)


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
class Shape:
    def contains(self, P):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersect(self, other)

    def __sub__(self, other):
        return Difference(self, other)


class Box(Shape):
    """Axis-aligned box given by its two corners (lo, hi)."""

    def __init__(self, lo, hi):
        self.lo = np.minimum(v3(lo), v3(hi))
        self.hi = np.maximum(v3(lo), v3(hi))

    def contains(self, P):
        return np.all((P >= self.lo - EPS) & (P <= self.hi + EPS), axis=1)

    def bounds(self):
        return self.lo, self.hi


def BoxC(center, size):
    """Axis-aligned box from centre and FULL size (converted to half sizes here)."""
    c, h = v3(center), v3(size) / 2.0
    return Box(c - h, c + h)


class RBox(Shape):
    """Rotated box: centre, FULL size, rotation matrix (local -> world)."""

    def __init__(self, center, size, R=None):
        self.c = v3(center)
        self.h = v3(size) / 2.0
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    def contains(self, P):
        L = (P - self.c) @ self.R  # world -> local
        return np.all(np.abs(L) <= self.h + EPS, axis=1)

    def bounds(self):
        ext = np.abs(self.R) @ self.h
        return self.c - ext, self.c + ext


class Ellipsoid(Shape):
    def __init__(self, center, radii, R=None):
        self.c = v3(center)
        self.r = v3(radii) if np.ndim(radii) else np.full(3, float(radii))
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    def contains(self, P):
        L = (P - self.c) @ self.R
        return np.sum((L / self.r) ** 2, axis=1) <= 1.0 + EPS

    def bounds(self):
        ext = np.abs(self.R) @ self.r
        return self.c - ext, self.c + ext


def Sphere(center, r):
    return Ellipsoid(center, (r, r, r))


class Cylinder(Shape):
    """Capped cylinder between points a and b."""

    def __init__(self, a, b, r):
        self.a, self.b, self.r = v3(a), v3(b), float(r)

    def contains(self, P):
        ab = self.b - self.a
        L2 = ab @ ab
        t = ((P - self.a) @ ab) / L2
        d = P - (self.a + np.outer(t, ab))
        return (t >= -EPS) & (t <= 1 + EPS) & (np.sum(d * d, axis=1) <= self.r ** 2 + EPS)

    def bounds(self):
        lo = np.minimum(self.a, self.b) - self.r
        hi = np.maximum(self.a, self.b) + self.r
        return lo, hi


class RoundCone(Shape):
    """Cone with rounded ends: sphere ra at a, sphere rb at b (exact sdf, iq)."""

    def __init__(self, a, b, ra, rb):
        self.a, self.b, self.ra, self.rb = v3(a), v3(b), float(ra), float(rb)

    def sdf(self, P):
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = ba @ ba
        if l2 < 1e-12:
            return np.linalg.norm(P - a, axis=1) - max(r1, r2)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pa = P - a
        y = pa @ ba
        z = y - l2
        q = pa * l2 - np.outer(y, ba)
        x2 = np.sum(q * q, axis=1)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d1 = np.sqrt(x2 + z2) * il2 - r2
        d2 = np.sqrt(x2 + y2) * il2 - r1
        d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))

    def contains(self, P):
        return self.sdf(P) <= EPS

    def bounds(self):
        lo = np.minimum(self.a - self.ra, self.b - self.rb)
        hi = np.maximum(self.a + self.ra, self.b + self.rb)
        return lo, hi


def Capsule(a, b, r):
    return RoundCone(a, b, r, r)


class Sweep(Shape):
    """Tube along a polyline, radius interpolated per point (chain of RoundCones)."""

    def __init__(self, pts, radii):
        pts = [v3(p) for p in pts]
        if np.ndim(radii) == 0:
            radii = [float(radii)] * len(pts)
        self.segs = [RoundCone(pts[i], pts[i + 1], radii[i], radii[i + 1]) for i in range(len(pts) - 1)]

    def contains(self, P):
        m = np.zeros(len(P), bool)
        for s in self.segs:
            lo, hi = s.bounds()
            sel = np.all((P >= lo - EPS) & (P <= hi + EPS), axis=1) & ~m
            if sel.any():
                m[sel] = s.contains(P[sel])
        return m

    def bounds(self):
        b = [s.bounds() for s in self.segs]
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)


def bezier(p0, p1, p2, p3=None, n=12):
    """Points along a quadratic (3 pts) or cubic (4 pts) Bezier curve."""
    p0, p1, p2 = v3(p0), v3(p1), v3(p2)
    t = np.linspace(0, 1, n)[:, None]
    if p3 is None:
        return list((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)
    p3 = v3(p3)
    return list((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _point_in_poly(x, y, poly):
    """Even-odd rule, vectorised over points (x, y)."""
    inside = np.zeros(len(x), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
        inside ^= cond & (x < xin)
    return inside


class Plate(Shape):
    """2D polygon (in the plane spanned by u, v through origin) extruded by thickness along u x v."""

    def __init__(self, origin, u, v, poly, thickness):
        self.o = v3(origin)
        self.u = v3(u) / np.linalg.norm(u)
        self.v = v3(v) / np.linalg.norm(v)
        self.w = np.cross(self.u, self.v)
        self.poly = [tuple(map(float, p)) for p in poly]
        self.t = float(thickness)

    def contains(self, P):
        d = P - self.o
        pu, pv, pw = d @ self.u, d @ self.v, d @ self.w
        m = np.abs(pw) <= self.t / 2 + EPS
        if m.any():
            idx = np.nonzero(m)[0]
            m[idx] = _point_in_poly(pu[idx], pv[idx], self.poly)
        return m

    def bounds(self):
        pts = []
        for (a, b) in self.poly:
            for s in (-0.5, 0.5):
                pts.append(self.o + a * self.u + b * self.v + s * self.t * self.w)
        pts = np.array(pts)
        return pts.min(0), pts.max(0)


class TorusArc(Shape):
    """Part of a torus: centre, axis, major radius R, minor radius r,
    angles (degrees) measured from ref direction e1 towards e2 = axis x e1."""

    def __init__(self, center, axis, ref, R, r, a0=-180.0, a1=180.0):
        self.c = v3(center)
        self.n = v3(axis) / np.linalg.norm(axis)
        e1 = v3(ref) - (v3(ref) @ self.n) * self.n
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.n, self.e1)
        self.R, self.r = float(R), float(r)
        self.a0, self.a1 = math.radians(a0), math.radians(a1)

    def contains(self, P):
        q = P - self.c
        h = q @ self.n
        x, y = q @ self.e1, q @ self.e2
        rho = np.sqrt(x * x + y * y)
        ang = np.arctan2(y, x)
        inside = (rho - self.R) ** 2 + h * h <= self.r ** 2 + EPS
        return inside & (ang >= self.a0 - 1e-6) & (ang <= self.a1 + 1e-6)

    def bounds(self):
        e = self.R + self.r
        return self.c - e, self.c + e


class Func(Shape):
    """Arbitrary predicate fn(P)->bool inside an explicit bounding box."""

    def __init__(self, fn, lo, hi):
        self.fn, self.lo, self.hi = fn, v3(lo), v3(hi)

    def contains(self, P):
        return self.fn(P)

    def bounds(self):
        return self.lo, self.hi


class Union(Shape):
    def __init__(self, *shapes):
        self.shapes = shapes

    def contains(self, P):
        m = np.zeros(len(P), bool)
        for s in self.shapes:
            lo, hi = s.bounds()
            sel = np.all((P >= lo - EPS) & (P <= hi + EPS), axis=1) & ~m
            if sel.any():
                m[sel] = s.contains(P[sel])
        return m

    def bounds(self):
        b = [s.bounds() for s in self.shapes]
        return np.min([x[0] for x in b], axis=0), np.max([x[1] for x in b], axis=0)


class Intersect(Shape):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def contains(self, P):
        m = self.a.contains(P)
        if m.any():
            idx = np.nonzero(m)[0]
            m[idx] = self.b.contains(P[idx])
        return m

    def bounds(self):
        (l1, h1), (l2, h2) = self.a.bounds(), self.b.bounds()
        return np.maximum(l1, l2), np.minimum(h1, h2)


class Difference(Shape):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def contains(self, P):
        m = self.a.contains(P)
        if m.any():
            idx = np.nonzero(m)[0]
            m[idx] = ~self.b.contains(P[idx])
        return m

    def bounds(self):
        return self.a.bounds()


class Scale(Shape):
    """Shape scaled by factors (fx, fy, fz) around a centre point."""

    def __init__(self, s, center, factors):
        self.s, self.c, self.f = s, v3(center), v3(factors)

    def contains(self, P):
        return self.s.contains(self.c + (P - self.c) / self.f)

    def bounds(self):
        lo, hi = self.s.bounds()
        a = self.c + (lo - self.c) * self.f
        b = self.c + (hi - self.c) * self.f
        return np.minimum(a, b), np.maximum(a, b)


class MirrorY(Shape):
    """The shape plus its mirror image across the plane Y = 0."""

    def __init__(self, s):
        self.s = s

    def contains(self, P):
        Q = P.copy()
        Q[:, 1] = -Q[:, 1]
        return self.s.contains(P) | self.s.contains(Q)

    def bounds(self):
        lo, hi = self.s.bounds()
        m = np.maximum(np.abs(lo[1]), np.abs(hi[1]))
        lo, hi = lo.copy(), hi.copy()
        lo[1], hi[1] = -m, m
        return lo, hi


def sym(s):
    return MirrorY(s)


# --------------------------------------------------------------------------
# Voxel model
# --------------------------------------------------------------------------
class VoxelModel:
    """Material + part ids on the small grid.  0 = empty."""

    def __init__(self, lo, hi, small=0.16, ratio=3):
        self.s = float(small)
        self.r = int(ratio)
        self.b = self.s * self.r
        lo = np.floor(v3(lo) / self.b + 1e-9) * self.b
        hi = np.ceil(v3(hi) / self.b - 1e-9) * self.b
        self.origin = lo
        self.dims = tuple(int(math.floor((hi[i] - lo[i]) / self.s + 0.5)) for i in range(3))
        assert all(d % self.r == 0 for d in self.dims)
        self.mat = np.zeros(self.dims, np.uint8)
        self.part = np.zeros(self.dims, np.uint8)
        self.mat_names = ["empty"]
        self.part_names = ["none"]
        self.log = []
        self.offset = np.zeros(3)   # translation applied to every shape (move a whole group of shapes)

    # ids --------------------------------------------------------------
    def mid(self, name):
        if name not in self.mat_names:
            self.mat_names.append(name)
        return self.mat_names.index(name)

    def pid(self, name):
        if name not in self.part_names:
            self.part_names.append(name)
        return self.part_names.index(name)

    # sampling -----------------------------------------------------------
    def centers(self, idx_slices, big=False):
        cell = self.b if big else self.s
        return [self.origin[d] + (np.arange(idx_slices[d].start, idx_slices[d].stop) + 0.5) * cell for d in range(3)]

    def _mask(self, shape, big):
        cell = self.b if big else self.s
        n = np.array(self.dims) // (self.r if big else 1)
        lo, hi = shape.bounds()
        lo, hi = lo + self.offset, hi + self.offset
        i0 = np.clip(np.floor((lo - self.origin) / cell).astype(int) - 1, 0, n)
        i1 = np.clip(np.ceil((hi - self.origin) / cell).astype(int) + 1, 0, n)
        if np.any(i1 <= i0):
            return None
        axes = [self.origin[d] + (np.arange(i0[d], i1[d]) + 0.5) * cell for d in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) - self.offset
        m = shape.contains(P).reshape(X.shape)
        if big:
            m = m.repeat(self.r, 0).repeat(self.r, 1).repeat(self.r, 2)
            i0, i1 = i0 * self.r, i1 * self.r
        sl = tuple(slice(int(i0[d]), int(i1[d])) for d in range(3))
        return sl, m

    # operations -----------------------------------------------------------
    def add(self, shape, mat, part, big=True, only_empty=False):
        r = self._mask(shape, big)
        if r is None:
            return 0
        sl, m = r
        if only_empty:
            m = m & (self.mat[sl] == 0)
        self.mat[sl][m] = self.mid(mat)
        self.part[sl][m] = self.pid(part)
        return int(m.sum())

    def cut(self, shape, big=False, only_parts=None):
        r = self._mask(shape, big)
        if r is None:
            return 0
        sl, m = r
        if only_parts is not None:
            m = m & np.isin(self.part[sl], [self.pid(p) for p in only_parts])
        self.mat[sl][m] = 0
        self.part[sl][m] = 0
        return int(m.sum())

    def paint(self, shape, mat, big=False, only=None, parts=None, where=None):
        """Recolour existing voxels inside shape.  only = list of material names allowed.
        where = optional fn(X, Y, Z) -> bool on voxel centres."""
        r = self._mask(shape, big)
        if r is None:
            return 0
        sl, m = r
        m = m & (self.mat[sl] > 0)
        if only is not None:
            m &= np.isin(self.mat[sl], [self.mid(o) for o in only])
        if parts is not None:
            m &= np.isin(self.part[sl], [self.pid(p) for p in parts])
        if where is not None:
            X, Y, Z = np.meshgrid(*self.centers(sl), indexing="ij")
            m &= where(X - self.offset[0], Y - self.offset[1], Z - self.offset[2])
        self.mat[sl][m] = self.mid(mat)
        return int(m.sum())

    def setpart(self, shape, part, big=False, parts=None):
        r = self._mask(shape, big)
        if r is None:
            return 0
        sl, m = r
        m = m & (self.mat[sl] > 0)
        if parts is not None:
            m &= np.isin(self.part[sl], [self.pid(p) for p in parts])
        self.part[sl][m] = self.pid(part)
        return int(m.sum())

    def paint_bitmap(self, origin, u, v, rows, colors, depth=1, normal=None, parts=None):
        """Paint a pixel-art bitmap onto the surface.
        origin: world point of the bitmap's bottom-left corner; u, v: unit axes
        (one small cube per character).  rows[0] is the TOP row.  colors maps
        characters to material names ('.' or ' ' = untouched).  Voxels are
        recoloured from the surface inward (along -normal) for `depth` cells."""
        o, u, v = v3(origin) + self.offset, v3(u), v3(v)
        n = v3(normal) if normal is not None else np.cross(u, v)
        H = len(rows)
        cnt = 0
        for ri, row in enumerate(rows):
            for ci, ch in enumerate(row):
                if ch not in colors:
                    continue
                p = o + (ci + 0.5) * self.s * u + (H - 1 - ri + 0.5) * self.s * v
                # walk inward from a bit outside the surface, find the first solid voxel
                found = 0
                for k in range(-4, 12):
                    q = p - n * (k + 0.5) * self.s
                    idx = np.floor((q - self.origin) / self.s).astype(int)
                    if np.any(idx < 0) or np.any(idx >= self.dims):
                        continue
                    if self.mat[tuple(idx)] > 0:
                        if parts is not None and self.part_names[self.part[tuple(idx)]] not in parts:
                            break
                        self.mat[tuple(idx)] = self.mid(colors[ch])
                        found += 1
                        cnt += 1
                        if found >= depth:
                            break
                    elif found:
                        break
        return cnt

    def symmetrize_x(self, part):
        """Make one part exactly symmetric front/back: its -X half becomes the mirror of its +X half
        (needs a grid centred on X = 0).  Never overwrites other parts."""
        pid = self.pid(part)
        nx = self.dims[0]
        assert nx % 2 == 0 and abs(self.origin[0] + nx * self.s / 2) < 1e-6, "grid not centred in X"
        h = nx // 2
        lo = self.part[:h] == pid
        self.mat[:h][lo] = 0
        self.part[:h][lo] = 0
        src_m = np.flip(self.mat[h:], axis=0)
        src_p = np.flip(self.part[h:], axis=0)
        put = (src_p == pid) & (self.mat[:h] == 0)
        self.mat[:h][put] = src_m[put]
        self.part[:h][put] = pid

    # analysis ------------------------------------------------------------
    def world(self, idx):
        return self.origin + (np.asarray(idx) + 0.5) * self.s

    def clean(self, min_keep=27, verbose=True):
        """Remove floating pieces smaller than min_keep voxels, report bigger ones."""
        solid = self.mat > 0
        lab, n = ndimage.label(solid)
        if n == 0:
            return []
        sizes = ndimage.sum(solid, lab, range(1, n + 1)).astype(int)
        big = int(np.argmax(sizes)) + 1
        removed, floating = 0, []
        for i, sz in enumerate(sizes, start=1):
            if i == big:
                continue
            if sz < min_keep:
                m = lab == i
                self.mat[m] = 0
                self.part[m] = 0
                removed += sz
            else:
                c = np.argwhere(lab == i).mean(0)
                floating.append((int(sz), np.round(self.world(c), 2).tolist()))
        if verbose:
            print(f"[clean] components={n} removed_voxels={removed} floating_kept={floating}")
        return floating

    def part_components(self):
        out = {}
        for pid, name in enumerate(self.part_names):
            if pid == 0:
                continue
            m = self.part == pid
            if not m.any():
                continue
            lab, n = ndimage.label(m)
            sizes = sorted(ndimage.sum(m, lab, range(1, n + 1)).astype(int).tolist(), reverse=True)
            out[name] = sizes
        return out

    def symmetry_report(self):
        flip = self.mat[:, ::-1, :]
        rep = {}
        for mid, name in enumerate(self.mat_names):
            if mid == 0:
                continue
            a = (self.mat == mid)
            if not a.any():
                continue
            b = (flip == mid)
            rep[name] = int((a ^ b).sum())
        return rep

    def bbox(self):
        idx = np.argwhere(self.mat > 0)
        lo = self.origin + idx.min(0) * self.s
        hi = self.origin + (idx.max(0) + 1) * self.s
        return lo, hi
