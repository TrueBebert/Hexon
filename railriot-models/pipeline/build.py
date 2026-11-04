"""Build a Rail Riot voxel model: voxels -> meshes -> atlases -> data for Blender.

usage: python -m pipeline.build models.storm_serpent v1
"""
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from .mesh import DIRS, INPLANE, Exposure, chunk_quads, sym_quads, quad_corners, lattice_touch, outline_offsets
from . import texture as tx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS = 1024
CAP = int(ATLAS * ATLAS * 0.93)
MAX_TRIS = 19000  # per mesh, outline included (Roblox limit 20 000)


def to_roblox(p):
    """Model axes (X fwd, Y width, Z up) -> Roblox/glTF axes (X, Y up, Z)."""
    x, y, z = p
    return [round(float(x), 4), round(float(z), 4), round(float(-y), 4)]


M_R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def rot_to_roblox(R):
    return M_R @ np.asarray(R, float) @ M_R.T


def quads_area(quads):
    return sum((q[4] * tx.PX + 2) * (q[5] * tx.PX + 2) for q in quads)


def split_part(expo, mask, k):
    """Split a part into k slabs along its longest axis, balanced by exposed faces."""
    idx = np.argwhere(mask)
    ext = idx.max(0) - idx.min(0)
    ext[1] = -1  # never split across the width: halves must stay mirror images
    axis = int(np.argmax(ext))
    w = np.zeros(mask.shape[axis])
    for d in range(6):
        e = expo.E[d] & mask
        other = tuple(i for i in range(3) if i != axis)
        w += e.sum(axis=other)
    cum = np.cumsum(w)
    cuts = [0]
    for i in range(1, k):
        c = int(np.searchsorted(cum, cum[-1] * i / k))
        c = int(math.floor(c / 3 + 0.5) * 3)
        cuts.append(max(c, cuts[-1] + 3))
    cuts.append(mask.shape[axis])
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sl = [slice(None)] * 3
        sl[axis] = slice(a, b)
        m = np.zeros_like(mask)
        m[tuple(sl)] = mask[tuple(sl)]
        if m.any():
            out.append(m)
    return out


def ffd(areas, cap):
    pages, loads = [], []
    for i in sorted(range(len(areas)), key=lambda i: -areas[i]):
        for p in range(len(pages)):
            if loads[p] + areas[i] <= cap:
                pages[p].append(i)
                loads[p] += areas[i]
                break
        else:
            pages.append([i])
            loads.append(areas[i])
    return pages


def build(module_name, version, render=True, views=None, quick=False):
    t0 = time.time()
    spec = importlib.import_module(module_name)
    name = spec.NAME
    outdir = os.path.join(ROOT, "output", name, version)
    os.makedirs(outdir, exist_ok=True)

    vm = spec.make()
    print(f"[{name}] voxels built in {time.time() - t0:.1f}s, grid {vm.dims}")
    floating = vm.clean(min_keep=spec.__dict__.get("MIN_KEEP", 27))
    comps = vm.part_components()
    sym = vm.symmetry_report()
    lo, hi = vm.bbox()
    size = hi - lo
    print(f"[{name}] bbox lo={np.round(lo, 2)} hi={np.round(hi, 2)} size={np.round(size, 2)}")
    print(f"[{name}] part components: { {k: v[:4] for k, v in comps.items()} }")
    print(f"[{name}] symmetry mismatches: {sym}")

    # parts & rigid groups ----------------------------------------------------------
    parts = spec.PARTS
    pnames = [p["name"] for p in parts]
    groups = {}
    part_group = np.zeros(len(vm.part_names), np.int32)
    for p in parts:
        g = p.get("group", p["name"])
        groups.setdefault(g, len(groups) + 1)
        if p["name"] in vm.part_names:
            part_group[vm.part_names.index(p["name"])] = groups[g]
    unknown = [n for n in vm.part_names[1:] if n not in pnames]
    assert not unknown, f"parts without definition: {unknown}"

    expo = Exposure(vm, part_group)
    drop = getattr(spec, "DROP_DOWN", None)
    if drop:  # downward faces nobody can see (under the wagon): no triangles, no texels
        zc = vm.origin[2] + (np.arange(vm.dims[2]) + 0.5) * vm.s
        pm = np.isin(vm.part, [vm.part_names.index(p) for p in drop["parts"] if p in vm.part_names])
        low = pm & (zc[None, None, :] < drop["below"])
        expo.E[5] &= ~low
        expo.J[5] &= ~low

    # chunks ----------------------------------------------------------------------
    # quads of the y >= 0 half are mirrored to the other half and share their texels
    def make_quads(m):
        return sym_quads(expo, m, vm.mat)

    def painted_area(q, src):
        return quads_area([x for x, s_ in zip(q, src) if s_ < 0])

    chunks = []
    for p in parts:
        if p["name"] not in vm.part_names:
            continue
        mask = vm.part == vm.part_names.index(p["name"])
        if not mask.any():
            continue
        q, src = make_quads(mask)
        a = painted_area(q, src)
        k = max(1, math.ceil(a / CAP), math.ceil(len(q) * 4 / MAX_TRIS))
        masks, qs = [mask], [(q, src)]
        while k > 1:
            masks = split_part(expo, mask, k)
            qs = [make_quads(m) for m in masks]
            if all(painted_area(*x) <= CAP and len(x[0]) * 4 <= MAX_TRIS for x in qs):
                break
            k += 1
        for i, (m, (qq, ss)) in enumerate(zip(masks, qs)):
            nm = p["name"] if i == 0 else f"{p['name']}_{i + 1}"
            chunks.append(dict(name=nm, part=p["name"], mask=m, quads=qq, src=ss, area=painted_area(qq, ss)))
    print(f"[{name}] chunks: {[(c['name'], len(c['quads']), c['area']) for c in chunks]}")

    # pages -----------------------------------------------------------------------
    cap = CAP
    while True:
        pages = ffd([c["area"] for c in chunks], cap)
        placements, ok = [], True
        for pg in pages:
            sizes, refs = [], []
            for ci in pg:
                for qi, q in enumerate(chunks[ci]["quads"]):
                    if chunks[ci]["src"][qi] >= 0:
                        continue
                    sizes.append(tx.quad_px_size(q))
                    refs.append((ci, qi))
            pos = tx.skyline_pack(sizes, ATLAS, ATLAS)
            if pos is None:
                ok = False
                break
            placements.append([(refs[i], pos[i]) for i in range(len(refs))])
        if ok:
            break
        cap = int(cap * 0.95)
    for pi, pg in enumerate(pages):
        for ci in pg:
            chunks[ci]["page"] = pi
    print(f"[{name}] atlas pages: {len(pages)} (cap {cap})")

    # cells (global over all chunk quads) ---------------------------------------------
    allq, gidx = [], {}
    for ci, c in enumerate(chunks):
        for qi, q in enumerate(c["quads"]):
            if c["src"][qi] < 0:
                gidx[(ci, qi)] = len(allq)
                allq.append(q)
    cells = tx.Cells(np.array(allq))
    occ = vm.mat > 0
    t1 = time.time()
    ao = tx.compute_ao(occ, cells)
    cao = tx.corner_ao(vm.mat.shape, cells, ao)
    edges = tx.edge_flags(occ, cells)
    studded = [spec.PALETTE.get(n, {}).get("studs", True) for n in vm.mat_names]
    studs = tx.stud_flags(expo, vm, studded, cells)
    print(f"[{name}] cells={cells.N} ao+flags {time.time() - t1:.1f}s")

    tile = tx.load_stud_tile(os.path.join(ROOT, "assets", "roblox_studs_tile.png"))
    painter = tx.Painter(vm, expo, cells, spec.PALETTE, tile, ao, cao, edges, studs, fx=getattr(spec, "FX", None))

    page_files = []
    for pi, plc in enumerate(placements):
        placed = [(gidx[(ci, qi)], x, y) for ((ci, qi), (x, y)) in plc]
        img = painter.paint_page(placed, ATLAS)
        fn = f"{name}_atlas{pi + 1}.png"
        path = os.path.join(outdir, fn)
        nbytes = save_png(img, path)
        page_files.append(dict(file=fn, bytes=nbytes))
        print(f"[{name}] {fn}: {nbytes / 1024:.0f} KiB")

    # meshes ----------------------------------------------------------------------
    s, origin = vm.s, vm.origin
    group_touch = {}
    mesh_data, mesh_meta = {}, []
    quad_pos = {}
    for pi, plc in enumerate(placements):
        for ((ci, qi), (x, y)) in plc:
            quad_pos[(ci, qi)] = (x, y)
    pdef = {p["name"]: p for p in parts}
    # outline shells: the rigid group dilated by one small cube (Minkowski sum with a
    # cube = faces pushed out, corners pushed along every face meeting there), each
    # dilated cell owned by one chunk so shells never overlap or double up.
    from scipy import ndimage
    from .mesh import shell_quads, surface_touch
    chunk_label = np.zeros(vm.mat.shape, np.int32)
    for ci, c in enumerate(chunks):
        chunk_label[c["mask"]] = ci + 1
    group_shell = {}
    for ci, c in enumerate(chunks):
        p = pdef[c["part"]]
        g = groups[p.get("group", p["name"])]
        if g not in group_shell:
            members = [i + 1 for i, cc in enumerate(chunks)
                       if groups[pdef[cc["part"]].get("group", cc["part"])] == g]
            gl = np.where(np.isin(chunk_label, members), chunk_label, 0)
            D = ndimage.binary_dilation(gl > 0, structure=np.ones((3, 3, 3), bool))
            # keep the hull out of eye sockets and their line of sight (it would cover them in black)
            for zone in getattr(spec, "NO_OUTLINE", []):
                r = vm._mask(zone, False)
                if r is not None:
                    sl, m = r
                    D[sl] &= ~(m & (gl[sl] == 0))
            owner = np.where(gl > 0, gl, ndimage.grey_dilation(gl, size=(3, 3, 3)))
            owner[~D] = 0
            group_shell[g] = (D, owner, surface_touch(D))
        D, owner, LD = group_shell[g]
        T = p.get("outline", 0.2)
        V, UV, F = [], [], []
        shellq = []
        Ny = vm.dims[1]
        for qi, q in enumerate(c["quads"]):
            cor = quad_corners(q)
            sq_ = c["src"][qi]
            tq = c["quads"][sq_] if sq_ >= 0 else q      # quad that owns the texels
            d, plane, u0, v0, du, dv = tq
            axis, sign = DIRS[d]
            U, Vax = INPLANE[axis]
            x0, y0 = quad_pos[(ci, sq_ if sq_ >= 0 else qi)]
            base = len(V)
            for cc in cor:
                V.append(origin + cc * s)
                tc = cc.copy()
                if sq_ >= 0:
                    tc[1] = Ny - tc[1]                      # mirrored corner on the source quad
                px = x0 + 1 + (tc[U] - u0) * tx.PX
                py = y0 + 1 + (v0 + dv - tc[Vax]) * tx.PX
                UV.append((px / ATLAS, 1.0 - py / ATLAS))
            F.append([base, base + 1, base + 2, base + 3])
        n_main = len(F)
        if T > 0:
            sq = shell_quads(D & (owner == ci + 1), D)
            shellq = [quad_corners(q) for q in sq]
            sc = np.array(shellq).reshape(-1, 3)
            uniq, inv = np.unique(sc, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            off = outline_offsets(LD, uniq, T - s)  # dilation gives s; push the rest
            sv = origin + uniq * s + off
            base = len(V)
            V.extend(list(sv))
            buv = (2.0 / ATLAS, 1.0 - 2.0 / ATLAS)
            UV.extend([buv] * len(sv))
            inv = inv.reshape(-1, 4)
            for f in inv:
                F.append([base + f[0], base + f[3], base + f[2], base + f[1]])  # reversed
        V = np.array(V, np.float32)
        UV = np.array(UV, np.float32)
        F = np.array(F, np.int32)
        key = f"m{ci}"
        mesh_data[key + "_V"] = V
        mesh_data[key + "_UV"] = UV
        mesh_data[key + "_F"] = F
        tris = 2 * len(F)
        mesh_meta.append(dict(key=key, name=c["name"], part=c["part"], page=c["page"], n_main=n_main,
                              n_faces=len(F), triangles=tris, verts=len(V)))
    np.savez_compressed(os.path.join(outdir, "mesh_data.npz"), **mesh_data)

    # neon ------------------------------------------------------------------------
    neon = []
    for n in getattr(spec, "NEON", []):
        R = np.asarray(n.get("R", np.eye(3)), float)
        nsize = np.asarray(n["size"], float) - 0.04  # no coplanar faces with voxels (z-fighting)
        neon.append(dict(name=n["name"], part=n["part"], shape=n.get("shape", "Block"),
                         center=[float(x) for x in n["center"]], size=nsize.tolist(), R=R.tolist(),
                         color=list(n["color"])))

    tri_total = sum(m["triangles"] for m in mesh_meta)
    tri_neon = sum(12 if n["shape"] == "Block" else 64 for n in neon)
    info = dict(name=name, version=version, small=vm.s, big=vm.b, bbox=[lo.tolist(), hi.tolist()],
                size=size.tolist(), parts=parts, meshes=mesh_meta, pages=page_files, neon=neon,
                attachments=getattr(spec, "ATTACHMENTS", []), lights=getattr(spec, "LIGHTS", []),
                triangles_mesh=tri_total,
                triangles_neon=tri_neon, floating=floating, symmetry=sym, components=comps,
                render=getattr(spec, "RENDER", {}), atlas=ATLAS)
    with open(os.path.join(outdir, "build_info.json"), "w") as f:
        json.dump(info, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_manifest(info, outdir)
    print(f"[{name}] triangles: meshes={tri_total} (+neon {tri_neon}) total={tri_total + tri_neon}")
    for m in mesh_meta:
        flag = "  <-- OVER LIMIT" if m["triangles"] > 20000 else ""
        print(f"    {m['name']:<12} tris={m['triangles']:>6} verts={m['verts']:>6} page={m['page'] + 1}{flag}")
    print(f"[{name}] python stage {time.time() - t0:.1f}s")

    if render is not False:
        run_blender(outdir, views=views, quick=quick)
    return outdir


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img, "RGB").save(path, optimize=True)
    size = os.path.getsize(path)
    if size > 1_000_000:
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "q.png")
            subprocess.run(["pngquant", "--force", "--quality=85-100", "--speed", "1", "--nofs",
                            "--output", tmp, path], check=False)
            if os.path.exists(tmp) and os.path.getsize(tmp) < size:
                shutil.copyfile(tmp, path)
        size = os.path.getsize(path)
    return size


def write_manifest(info, outdir):
    parts = []
    for p in info["parts"]:
        meshes = [m for m in info["meshes"] if m["part"] == p["name"]]
        if not meshes:
            continue
        parts.append(dict(name=p["name"], parent=p.get("parent"), pivot=to_roblox(p["pivot"]),
                          meshes=[m["name"] for m in meshes],
                          atlas=[info["pages"][m["page"]]["file"] for m in meshes],
                          triangles=sum(m["triangles"] for m in meshes)))
    neon = []
    for n in info["neon"]:
        R = rot_to_roblox(n["R"])
        neon.append(dict(name=n["name"], part=n["part"], shape=n["shape"], material="Neon",
                         color=n["color"], position=to_roblox(n["center"]),
                         size=[round(n["size"][0], 4), round(n["size"][2], 4), round(n["size"][1], 4)],
                         rotationMatrix=np.round(R, 5).tolist()))
    atts = [dict(name=a["name"], part=a["part"], position=to_roblox(a["position"]),
                 direction=to_roblox(a.get("direction", (1, 0, 0)))) for a in info["attachments"]]
    lights = [dict(name=l["name"], part=l["part"], type="PointLight", position=to_roblox(l["position"]),
                   color=list(l["color"]), range=l.get("range", 8), brightness=l.get("brightness", 1))
              for l in info.get("lights", [])]
    lo, hi = info["bbox"]
    info = dict(info, size=(np.asarray(hi) - np.asarray(lo)).tolist())
    man = dict(
        name=info["name"], version=info["version"], units="studs",
        axes="Roblox: X = avant du train, Y = haut, Z = -largeur (glTF Y-up)",
        cubeSize=dict(small=info["small"], big=info["big"]),
        groundHeight=round(float(lo[2]), 4),
        size=[round(info["size"][0], 3), round(info["size"][2], 3), round(info["size"][1], 3)],
        triangles=dict(meshes=info["triangles_mesh"], neon=info["triangles_neon"],
                       total=info["triangles_mesh"] + info["triangles_neon"]),
        textures=info["pages"], parts=parts, neon=neon, attachments=atts, lights=lights,
    )
    with open(os.path.join(outdir, f"{info['name']}_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)


def run_blender(outdir, views=None, quick=False):
    script = os.path.join(ROOT, "pipeline", "blender_export.py")
    args = ["blender", "-b", "--factory-startup", "--python", script, "--", outdir]
    if views:
        args += ["--views", ",".join(views)]
    if quick:
        args += ["--quick"]
    if os.environ.get("RR_SAMPLES"):
        args += ["--samples", os.environ["RR_SAMPLES"]]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("[bl]") or "Error" in l or "error" in l]
    print("\n".join(lines[-40:]))
    if r.returncode != 0:
        print(r.stderr[-3000:])
    print(f"[blender] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    mod = sys.argv[1]
    ver = sys.argv[2] if len(sys.argv) > 2 else "wip"
    views = None
    quick = "--quick" in sys.argv
    norender = "--norender" in sys.argv
    for a in sys.argv[3:]:
        if a.startswith("--views="):
            views = a.split("=", 1)[1].split(",")
    build(mod, ver, render=not norender, views=views, quick=quick)
