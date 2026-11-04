"""Run inside Blender:  blender -b --factory-startup --python blender_export.py -- <outdir> [--views a,b] [--quick]

Builds the mesh objects (1 Blender unit = 1 stud, pivot of each part at its
joint), exports <name>.glb and <name>.fbx, then renders preview views with the
outline shell (back-face culled like in Roblox) and neon parts.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = argv[0]
QUICK = "--quick" in argv
VIEWS = None
if "--views" in argv:
    VIEWS = argv[argv.index("--views") + 1].split(",")
EXPORT = "--noexport" not in argv

info = json.load(open(os.path.join(OUT, "build_info.json")))
data = np.load(os.path.join(OUT, "mesh_data.npz"))
NAME = info["name"]


def log(*a):
    print("[bl]", *a, flush=True)


# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "NONE"


def srgb_to_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ---------------------------------------------------------------- materials
atlas_mats = []
for i, pg in enumerate(info["pages"]):
    img = bpy.data.images.load(os.path.join(OUT, pg["file"]))
    img.name = pg["file"]
    m = bpy.data.materials.new(f"{NAME}_Atlas{i + 1}")
    m.use_nodes = True
    m.use_backface_culling = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    try:
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
    except KeyError:
        bsdf.inputs["Specular"].default_value = 0.15
    atlas_mats.append(m)

emissive_mats = {}


def emissive_mat(page):
    """Atlas material that also glows (parts meant to be set to Neon in Studio)."""
    if page in emissive_mats:
        return emissive_mats[page]
    m = atlas_mats[page].copy()
    m.name = f"{NAME}_Atlas{page + 1}_Glow"
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = [n for n in nt.nodes if n.type == "TEX_IMAGE"][0]
    try:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    except KeyError:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission"])
    bsdf.inputs["Emission Strength"].default_value = 1.1
    emissive_mats[page] = m
    return m


neon_mats = {}


def neon_mat(color):
    key = tuple(color)
    if key in neon_mats:
        return neon_mats[key]
    m = bpy.data.materials.new(f"Neon_{color[0]}_{color[1]}_{color[2]}")
    m.use_nodes = True
    m.use_backface_culling = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    lin = [srgb_to_lin(c) for c in color] + [1.0]
    bsdf.inputs["Base Color"].default_value = lin
    try:
        bsdf.inputs["Emission Color"].default_value = lin
    except KeyError:
        bsdf.inputs["Emission"].default_value = lin
    bsdf.inputs["Emission Strength"].default_value = 1.25
    neon_mats[key] = m
    return m


# ---------------------------------------------------------------- objects
parts = {p["name"]: p for p in info["parts"]}
main_obj = {}
objs = []
for mm in info["meshes"]:
    V = data[mm["key"] + "_V"].astype(np.float64)
    UV = data[mm["key"] + "_UV"]
    F = data[mm["key"] + "_F"]
    pivot = np.array(parts[mm["part"]]["pivot"], float)
    me = bpy.data.meshes.new(mm["name"])
    me.from_pydata((V - pivot).tolist(), [], F.tolist())
    uvl = me.uv_layers.new(name="UVMap")
    loop_v = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    uvl.data.foreach_set("uv", UV[loop_v].ravel())
    if parts[mm["part"]].get("emissive"):
        me.materials.append(emissive_mat(mm["page"]))
    else:
        me.materials.append(atlas_mats[mm["page"]])
    for poly in me.polygons:
        poly.use_smooth = False
    me.update()
    ob = bpy.data.objects.new(mm["name"], me)
    scene.collection.objects.link(ob)
    ob.location = Vector(pivot)
    ob["n_main"] = mm["n_main"]
    if mm["part"] not in main_obj:
        main_obj[mm["part"]] = ob
    objs.append((ob, mm))

# hierarchy: part -> parent part; extra chunks -> the part's main object.
# Nothing is rotated, so local location = own pivot - parent pivot.
obj_part = {ob.name: mm["part"] for ob, mm in objs}
for ob, mm in objs:
    p = parts[mm["part"]]
    if main_obj[mm["part"]] is not ob:
        par = main_obj[mm["part"]]
    else:
        par = main_obj.get(p.get("parent")) if p.get("parent") else None
    pivot = Vector(p["pivot"])
    if par is not None:
        ob.parent = par
        ob.location = pivot - Vector(parts[obj_part[par.name]]["pivot"])
    else:
        ob.location = pivot

neon_objs = []
for n in info["neon"]:
    R = Matrix(n["R"])
    c = Vector(n["center"])
    sx, sy, sz = n["size"]
    me = bpy.data.meshes.new(n["name"])
    if n["shape"] == "Cylinder":  # axis = local X (like a Roblox cylinder part)
        seg = 16
        verts, faces = [], []
        for side in (-1, 1):
            for i in range(seg):
                a = 2 * math.pi * i / seg
                verts.append((side * sx / 2, math.cos(a) * sy / 2, math.sin(a) * sz / 2))
        for i in range(seg):
            j = (i + 1) % seg
            faces.append((i, j, seg + j, seg + i))
        faces.append(tuple(reversed(range(seg))))
        faces.append(tuple(range(seg, 2 * seg)))
    else:
        verts = [(x * sx / 2, y * sy / 2, z * sz / 2) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me.from_pydata(verts, [], faces)
    me.materials.append(neon_mat(n["color"]))
    me.update()
    me.validate()
    ob = bpy.data.objects.new(n["name"], me)
    scene.collection.objects.link(ob)
    par = main_obj.get(n["part"])
    world = Matrix.Translation(c) @ R.to_4x4()
    if par is not None:
        ob.parent = par
        pp = Vector(parts[n["part"]]["pivot"])
        ob.matrix_basis = Matrix.Translation(-pp) @ world
    else:
        ob.matrix_basis = world
    neon_objs.append(ob)

# make normals consistent for neon boxes
for ob in neon_objs:
    me = ob.data
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()

bpy.context.view_layer.update()
model_objects = [o for o, _ in objs] + neon_objs

# ---------------------------------------------------------------- export
if EXPORT:
    bpy.ops.object.select_all(action="DESELECT")
    for o in model_objects:
        o.select_set(True)
    glb = os.path.join(OUT, f"{NAME}.glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=False, export_texcoords=True, export_normals=True,
                              export_materials="EXPORT", export_cameras=False, export_lights=False)
    log("glb", os.path.getsize(glb) // 1024, "KiB")
    fbx = os.path.join(OUT, f"{NAME}.fbx")
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={"MESH", "EMPTY"},
                             axis_forward="-Z", axis_up="Y", apply_unit_scale=True, global_scale=1.0,
                             apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
                             path_mode="COPY", embed_textures=True, bake_anim=False)
    log("fbx", os.path.getsize(fbx) // 1024, "KiB")

# ---------------------------------------------------------------- render setup
outline_mat = bpy.data.materials.new("OutlineRender")
outline_mat.use_nodes = True
nt = outline_mat.node_tree
for nd in list(nt.nodes):
    nt.nodes.remove(nd)
out = nt.nodes.new("ShaderNodeOutputMaterial")
mix = nt.nodes.new("ShaderNodeMixShader")
geo = nt.nodes.new("ShaderNodeNewGeometry")
black = nt.nodes.new("ShaderNodeEmission")
black.inputs["Color"].default_value = (0.004, 0.0035, 0.006, 1)
black.inputs["Strength"].default_value = 1.0
transp = nt.nodes.new("ShaderNodeBsdfTransparent")
# visible (black) only for camera rays that see the shell's front side, like a
# back-face culled hull in Roblox; shadow / bounce rays always pass through.
lp = nt.nodes.new("ShaderNodeLightPath")
m_sub = nt.nodes.new("ShaderNodeMath")
m_sub.operation = "SUBTRACT"
m_sub.inputs[0].default_value = 1.0
nt.links.new(geo.outputs["Backfacing"], m_sub.inputs[1])
m_mul = nt.nodes.new("ShaderNodeMath")
m_mul.operation = "MULTIPLY"
nt.links.new(lp.outputs["Is Camera Ray"], m_mul.inputs[0])
nt.links.new(m_sub.outputs[0], m_mul.inputs[1])
m_inv = nt.nodes.new("ShaderNodeMath")
m_inv.operation = "SUBTRACT"
m_inv.inputs[0].default_value = 1.0
nt.links.new(m_mul.outputs[0], m_inv.inputs[1])
nt.links.new(m_inv.outputs[0], mix.inputs["Fac"])
nt.links.new(black.outputs["Emission"], mix.inputs[1])
nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
outline_mat.blend_method = "BLEND"

for ob, mm in objs:
    me = ob.data
    me.materials.append(outline_mat)
    n_main = mm["n_main"]
    mi = np.zeros(len(me.polygons), np.int32)
    mi[n_main:] = 1
    me.polygons.foreach_set("material_index", mi)
    me.update()

lo = Vector(info["bbox"][0])
hi = Vector(info["bbox"][1])
center = (lo + hi) / 2
size = hi - lo
R = info.get("render", {})


def add_box(name, c, s, color, rough=0.9):
    bpy.ops.mesh.primitive_cube_add(size=1, location=c)
    o = bpy.context.active_object
    o.name = name
    o.scale = s
    m = bpy.data.materials.new(name + "_m")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = [srgb_to_lin(x) for x in color] + [1]
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = rough
    o.data.materials.append(m)
    return o


# ground, rails, sleepers
add_box("Ground", (center.x, 0, -0.55), (400, 400, 1), (182, 180, 192))
gauge = R.get("rail_gauge", size.y * 0.62)
for sy in (-1, 1):
    add_box(f"Rail{sy}", (center.x, sy * gauge / 2, -0.12), (size.x + 16, 0.3, 0.26), (70, 62, 72), 0.6)
nsl = int((size.x + 16) / 2.4)
for i in range(nsl):
    x = center.x - (size.x + 16) / 2 + 1.2 + i * 2.4
    add_box(f"Sleeper{i}", (x, 0, -0.3), (0.9, gauge + 1.6, 0.25), (110, 66, 44))

# 5-stud blocky player for scale
px, py = hi.x + 3.0, hi.y + 2.5
add_box("Fig_legs", (px, py, 1.0), (1.0, 0.9, 2.0), (60, 70, 120))
add_box("Fig_torso", (px, py, 3.0), (1.0, 2.0, 2.0), (90, 140, 90))
add_box("Fig_head", (px, py, 4.5), (1.0, 1.0, 1.0), (235, 200, 120))

# lights
sun = bpy.data.lights.new("Sun", "SUN")
sun.energy = 3.2
sun.angle = math.radians(8)
so = bpy.data.objects.new("Sun", sun)
scene.collection.objects.link(so)
so.rotation_euler = (math.radians(50), math.radians(0), math.radians(-35 + 180 + 30))
fill = bpy.data.lights.new("Fill", "SUN")
fill.energy = 0.6
fo = bpy.data.objects.new("Fill", fill)
scene.collection.objects.link(fo)
fo.rotation_euler = (math.radians(65), 0, math.radians(150))

world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.56, 0.55, 0.62, 1)
bg.inputs["Strength"].default_value = 0.85

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 20 if QUICK else 72
if "--samples" in argv:
    scene.cycles.samples = int(argv[argv.index("--samples") + 1])
scene.cycles.use_denoising = False
try:
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
except Exception:
    pass
scene.cycles.max_bounces = 4
scene.cycles.transparent_max_bounces = 12
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"

cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def frame(direction, target, radius, res, lens=50, elev=None):
    cam_data.lens = lens
    scene.render.resolution_x, scene.render.resolution_y = res
    aspect = res[0] / res[1]
    fov = 2 * math.atan(36 / 2 / lens)  # sensor fit AUTO: 36 mm on the larger side
    if aspect >= 1:
        hfov, vfov = fov, 2 * math.atan(math.tan(fov / 2) / aspect)
    else:
        vfov, hfov = fov, 2 * math.atan(math.tan(fov / 2) * aspect)
    dist = radius / math.sin(min(hfov, vfov) / 2) * 1.02
    dvec = Vector(direction).normalized()
    cam.location = Vector(target) + dvec * dist
    look_at(cam, target)
    cam_data.clip_end = dist * 4


views = {
    "34": dict(dir=(1.0, 1.05, 0.55), target=center, radius=size.length * 0.48, res=(1280, 800)),
    "front": dict(dir=(1, 0, 0.18), target=center, radius=max(size.y, size.z) * 0.62, res=(900, 1000)),
    "profile": dict(dir=(0, 1, 0.12), target=center, radius=max(size.x, size.z) * 0.53, res=(1400, 900)),
    "back34": dict(dir=(-1.0, -1.05, 0.5), target=center, radius=size.length * 0.48, res=(1280, 800)),
    "ref": dict(dir=(0.42, -1.0, 0.22), target=center, radius=size.length * 0.40, res=(1400, 760)),
}
if "head" in R:
    hc = Vector(R["head"]["center"])
    hr = R["head"]["radius"]
    views["head_34"] = dict(dir=(1.0, 0.95, 0.35), target=hc, radius=hr, res=(1000, 1000), lens=110)
    views["head_front"] = dict(dir=(1, 0, 0.12), target=hc, radius=hr, res=(1000, 1000), lens=110)
    views["head_profile"] = dict(dir=(0, 1, 0.05), target=hc, radius=hr, res=(1000, 1000), lens=110)
    views["head_low"] = dict(dir=(1.0, 0.8, -0.35), target=hc, radius=hr, res=(1000, 1000), lens=110)

if QUICK:
    for v in views.values():
        v["res"] = (int(v["res"][0] * 0.62), int(v["res"][1] * 0.62))

os.makedirs(os.path.join(OUT, "renders"), exist_ok=True)
for vname in (VIEWS or ["ref", "34", "front", "profile", "back34", "head_34", "head_front", "head_profile", "head_low"]):
    if vname not in views:
        continue
    v = views[vname]
    frame(v["dir"], v["target"], v["radius"], v["res"], lens=v.get("lens", 50))
    scene.render.filepath = os.path.join(OUT, "renders", f"{NAME}_{vname}.png")
    bpy.ops.render.render(write_still=True)
    log("render", vname)
log("done")
