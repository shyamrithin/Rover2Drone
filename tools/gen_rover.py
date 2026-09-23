#!/usr/bin/env python3
"""
gen_rover.py
============

Generates the Rover2Drone ground vehicle: a four-wheel skid-steer utility
platform that carries the drone on a top-mounted landing deck with an
inductive charging pad, and acts as its mobile base station.

Design intent
-------------
The rover is the energy reservoir of the system. It carries a large battery
bank and recharges the drone between inspection flights through wireless
power transfer on the deck. The geometry reflects a buildable machine
rather than a toy: a sloped-sided steel body shell, an aluminium top plate
that doubles as the landing deck, large treaded tyres on suspension
brackets with inboard gearmotors, tubular bumpers, and rear masts carrying
a GNSS receiver and a communications antenna. The masts sit on the rear
bumper, outside the deck footprint, so they cannot strike the drone's
propellers during landing.

Generated files
---------------
  models/r2d_rover/model.sdf          the model
  models/r2d_rover/model.config
  models/r2d_rover/meshes/body.obj    sloped, chamfered body shell
  models/r2d_rover/meshes/tyre.obj    treaded tyre, reused on all wheels
  config/rover.yaml                   dimensions other tools and nodes read

config/rover.yaml is the single source of truth for deck height, pad
position and wheel geometry. place_turbines.py reads it to compute spawn
heights; the precision-landing and charging nodes will read it for the pad
location. Change a dimension here, regenerate, and everything follows.

Frames
------
base_link origin sits at the geometric centre of the body shell, BASE_H
above flat ground. x forward, y left, z up. Wheel joints rotate about +y.

Meshes are written flat-shaded (per-face normals) because the body and
tread are hard-edged; smoothed normals would make them look melted.

Drone latch
-----------
A DetachableJoint system welds the drone's base_link to the deck with a
fixed joint, standing in for the electromagnetic latch a real platform would
use to hold the drone during transit over rough or sloped ground. Four
latch plates are drawn where the drone's landing skids rest.

The system latches automatically as soon as the drone model appears, and
the weld captures whatever relative pose exists at that moment. The
latch_manager node therefore releases, lets the drone settle onto the deck,
and re-latches, so the drone is always locked resting flat on the pad.

Interface (the drive topics are unchanged, so existing nodes still work)
  /rover/cmd_vel         geometry_msgs Twist in (via ros_gz_bridge)
  /rover/odometry        nav_msgs Odometry out, frame odom -> rover/base_link
  /rover/latch/attach    gz.msgs.Empty in: engage latch
  /rover/latch/detach    gz.msgs.Empty in: release latch
  /rover/latch/state     gz.msgs.StringMsg out: "attached" / "detached",
                         published on change only

Usage
  python3 tools/gen_rover.py                 # writes into the repo
  python3 tools/gen_rover.py --preview p.png # also renders a check image
"""

import argparse
import math
import os

# ----------------------------------------------------------------------------
# Dimensions, metres. Edit here and regenerate.
# ----------------------------------------------------------------------------
WHEEL_R = 0.20            # tyre outer radius
WHEEL_W = 0.12            # tyre width
TREAD_DEPTH = 0.009
TRACK = 1.04              # lateral wheel-centre spacing
WHEELBASE = 0.84          # longitudinal axle spacing
BASE_H = 0.46             # base_link above flat ground

BODY_Z0, BODY_Z1 = -0.14, 0.14                     # shell bottom/top
BODY_BOT = (1.10, 0.72, 0.12)                      # length, width, chamfer
BODY_TOP = (0.98, 0.60, 0.10)

LID = (0.96, 0.62, 0.02)                           # aluminium deck plate
DECK_TOP = BODY_Z1 + LID[2]                        # deck surface, base_link z
PAD_SIZE = 0.52
PAD_X = -0.04                                      # pad centre, base_link x

# Drone the latch welds to. PX4 names spawned models <model>_<instance>.
DRONE_MODEL = "x500_gimbal_0"
DRONE_LINK = "base_link"
# x500 landing skids run along the drone's x axis at y = +/-0.132 m.
SKID_Y = 0.132

M_BODY = 90.0             # shell, frame, battery bank, electronics
M_WHEEL = 4.0

WHEEL_Z = WHEEL_R - BASE_H

MATERIALS = {
    #            ambient              diffuse              specular        emissive
    "gunmetal": ("0.12 0.13 0.15 1", "0.20 0.21 0.24 1", "0.35 0.35 0.35 1", None),
    "panel":    ("0.40 0.41 0.42 1", "0.60 0.61 0.62 1", "0.30 0.30 0.30 1", None),
    "alu":      ("0.55 0.56 0.58 1", "0.80 0.81 0.83 1", "0.70 0.70 0.70 1", None),
    "black":    ("0.03 0.03 0.03 1", "0.07 0.07 0.07 1", "0.20 0.20 0.20 1", None),
    "rubber":   ("0.03 0.03 0.03 1", "0.05 0.05 0.05 1", "0.05 0.05 0.05 1", None),
    "hub":      ("0.25 0.26 0.28 1", "0.38 0.39 0.42 1", "0.40 0.40 0.40 1", None),
    "pad":      ("0.06 0.07 0.09 1", "0.10 0.11 0.14 1", "0.15 0.15 0.15 1", None),
    "white":    ("0.80 0.80 0.80 1", "0.95 0.95 0.95 1", "0.20 0.20 0.20 1", None),
    "yellow":   ("0.70 0.52 0.05 1", "0.95 0.72 0.08 1", "0.30 0.30 0.30 1", None),
    "red":      ("0.50 0.03 0.03 1", "0.85 0.06 0.06 1", "0.30 0.30 0.30 1", "0.60 0.02 0.02 1"),
    "coil":     ("0.00 0.35 0.40 1", "0.00 0.70 0.80 1", "0.30 0.30 0.30 1", "0.00 0.55 0.65 1"),
    "lamp":     ("0.90 0.90 0.85 1", "1.00 1.00 0.95 1", "0.50 0.50 0.50 1", "0.95 0.95 0.85 1"),
    "lens":     ("0.02 0.02 0.05 1", "0.05 0.05 0.12 1", "0.90 0.90 0.90 1", None),
    "latch":    ("0.45 0.22 0.02 1", "0.85 0.42 0.05 1", "0.40 0.40 0.40 1", None),
}


# ----------------------------------------------------------------------------
# Small linear-algebra helpers (standard library only).
# ----------------------------------------------------------------------------
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_norm(a):
    n = math.sqrt(v_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def rot_matrix(r, p, y):
    """SDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def rpy_from_axes(z_axis, x_hint):
    """RPY of a frame whose z points along z_axis and x as close to x_hint."""
    z = v_norm(z_axis)
    x = v_norm(v_sub(x_hint, v_mul(z, v_dot(x_hint, z))))
    y = v_cross(z, x)
    R = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    pitch = -math.asin(max(-1.0, min(1.0, R[2][0])))
    roll = math.atan2(R[2][1], R[2][2])
    yaw = math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def mat_apply(R, v):
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


# ----------------------------------------------------------------------------
# Mesh generation (flat-shaded Wavefront OBJ).
# ----------------------------------------------------------------------------
class FlatMesh:
    """Triangle soup with one normal per face, written without shared verts."""

    def __init__(self):
        self.tris = []

    def tri(self, a, b, c):
        n = v_cross(v_sub(b, a), v_sub(c, a))
        if v_dot(n, n) < 1e-16:
            return
        self.tris.append((a, b, c, v_norm(n)))

    def quad(self, a, b, c, d):
        self.tri(a, b, c)
        self.tri(a, c, d)

    def write(self, path, name):
        with open(path, "w") as f:
            f.write("# Generated by gen_rover.py, do not edit by hand\n")
            f.write(f"o {name}\n")
            for a, b, c, _ in self.tris:
                for p in (a, b, c):
                    f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for _, _, _, n in self.tris:
                f.write(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
            for i in range(len(self.tris)):
                a, b, c = 3 * i + 1, 3 * i + 2, 3 * i + 3
                f.write(f"f {a}//{i + 1} {b}//{i + 1} {c}//{i + 1}\n")
        return len(self.tris)


def chamfered_rect(length, width, chamfer):
    """Eight-point chamfered rectangle, counter-clockwise seen from +z."""
    L, W, c = length / 2.0, width / 2.0, chamfer
    return [(L, -W + c), (L, W - c), (L - c, W), (-L + c, W),
            (-L, W - c), (-L, -W + c), (-L + c, -W), (L - c, -W)]


def build_body():
    """Frustum between two chamfered outlines: sloped sides, flat caps."""
    m = FlatMesh()
    bot = [(x, y, BODY_Z0) for x, y in chamfered_rect(*BODY_BOT)]
    top = [(x, y, BODY_Z1) for x, y in chamfered_rect(*BODY_TOP)]
    n = len(bot)
    for i in range(n):
        j = (i + 1) % n
        m.quad(bot[i], bot[j], top[j], top[i])
    ct = (0.0, 0.0, BODY_Z1)
    cb = (0.0, 0.0, BODY_Z0)
    for i in range(n):
        j = (i + 1) % n
        m.tri(ct, top[i], top[j])
        m.tri(cb, bot[j], bot[i])
    return m


def build_tyre(segments=96, blocks=24, inner_r=0.13):
    """
    Treaded tyre around local +z. The outline alternates between full radius
    and groove radius, giving raised tread blocks with sloped walls.
    """
    m = FlatMesh()
    per = segments // blocks
    h = WHEEL_W / 2.0
    outer, inner = [], []
    for i in range(segments):
        phase = i % per
        r = WHEEL_R if phase < per // 2 else WHEEL_R - TREAD_DEPTH
        a = 2.0 * math.pi * i / segments
        outer.append((r * math.cos(a), r * math.sin(a)))
        inner.append((inner_r * math.cos(a), inner_r * math.sin(a)))
    for i in range(segments):
        j = (i + 1) % segments
        o0, o1, i0, i1 = outer[i], outer[j], inner[i], inner[j]
        # Tread band, facing outward.
        m.quad((o0[0], o0[1], -h), (o1[0], o1[1], -h),
               (o1[0], o1[1], h), (o0[0], o0[1], h))
        # Sidewalls.
        m.quad((i0[0], i0[1], h), (o0[0], o0[1], h),
               (o1[0], o1[1], h), (i1[0], i1[1], h))
        m.quad((i1[0], i1[1], -h), (o1[0], o1[1], -h),
               (o0[0], o0[1], -h), (i0[0], i0[1], -h))
        # Inner bore, facing the axle.
        m.quad((i1[0], i1[1], -h), (i0[0], i0[1], -h),
               (i0[0], i0[1], h), (i1[0], i1[1], h))
    return m


# ----------------------------------------------------------------------------
# SDF assembly.
# ----------------------------------------------------------------------------
def material_xml(name):
    a, d, s, e = MATERIALS[name]
    x = f"<material><ambient>{a}</ambient><diffuse>{d}</diffuse>"
    if s:
        x += f"<specular>{s}</specular>"
    if e:
        x += f"<emissive>{e}</emissive>"
    return x + "</material>"


def pose_xml(xyz, rpy=(0.0, 0.0, 0.0)):
    return ("<pose>" + " ".join(f"{v:.5f}" for v in (*xyz, *rpy)) + "</pose>")


class Link:
    def __init__(self, name, xyz=(0, 0, 0), rpy=(0, 0, 0)):
        self.name, self.xyz, self.rpy = name, xyz, rpy
        self.parts, self.cols, self.inertial = [], [], ""
        self.prims = []          # for the preview renderer

    def _add(self, name, geom_xml, xyz, rpy, mat, prim):
        self.parts.append(
            f'        <visual name="{name}">{pose_xml(xyz, rpy)}'
            f"<geometry>{geom_xml}</geometry>{material_xml(mat)}</visual>")
        self.prims.append((prim, xyz, rpy, MATERIALS[mat][1]))

    def box(self, name, size, xyz, rpy=(0, 0, 0), mat="gunmetal"):
        g = f"<box><size>{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}</size></box>"
        self._add(name, g, xyz, rpy, mat, ("box", size))

    def cyl(self, name, r, length, xyz, rpy=(0, 0, 0), mat="black"):
        g = (f"<cylinder><radius>{r:.5f}</radius>"
             f"<length>{length:.5f}</length></cylinder>")
        self._add(name, g, xyz, rpy, mat, ("cyl", r, length))

    def mesh(self, name, uri, xyz, rpy=(0, 0, 0), mat="gunmetal", local=None):
        g = f"<mesh><uri>{uri}</uri></mesh>"
        self._add(name, g, xyz, rpy, mat, ("mesh", local))

    def collide(self, name, geom_xml, xyz, rpy=(0, 0, 0), surface=""):
        self.cols.append(
            f'        <collision name="{name}">{pose_xml(xyz, rpy)}'
            f"<geometry>{geom_xml}</geometry>{surface}</collision>")

    def xml(self):
        body = "\n".join([self.inertial] + self.parts + self.cols)
        return (f'      <link name="{self.name}">\n'
                f"        {pose_xml(self.xyz, self.rpy)}\n{body}\n      </link>")


def box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12.0, m * (x * x + z * z) / 12.0,
            m * (x * x + y * y) / 12.0)


def inertial_xml(m, ixx, iyy, izz, xyz=(0, 0, 0)):
    return (f"        <inertial>{pose_xml(xyz)}<mass>{m:.3f}</mass>"
            f"<inertia><ixx>{ixx:.5f}</ixx><iyy>{iyy:.5f}</iyy>"
            f"<izz>{izz:.5f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>"
            f"</inertia></inertial>")


def face_frame(bottom_edge_x, top_edge_x, axis):
    """
    Centre, outward normal and up-slope vector of one sloped body face.

    axis 'x' for front (+) / rear (-) faces, 'y' for left (+) / right (-).
    Returns (centre, normal, up, tilt) where tilt is the lean from vertical.
    """
    run = abs(bottom_edge_x) - abs(top_edge_x)
    tilt = math.atan2(run, BODY_Z1 - BODY_Z0)
    s = 1.0 if bottom_edge_x > 0 else -1.0
    mid = (bottom_edge_x + top_edge_x) / 2.0
    if axis == "x":
        c = (mid, 0.0, 0.0)
        n = (s * math.cos(tilt), 0.0, math.sin(tilt))
        u = (-s * math.sin(tilt), 0.0, math.cos(tilt))
    else:
        c = (0.0, mid, 0.0)
        n = (0.0, s * math.cos(tilt), math.sin(tilt))
        u = (0.0, -s * math.sin(tilt), math.cos(tilt))
    return c, n, u, tilt


def on_face(c, n, u, lateral, along, lift, lat_axis):
    """Point on a face: `along` up the slope, `lateral` across, `lift` out."""
    p = v_add(c, v_mul(u, along))
    p = v_add(p, v_mul(n, lift))
    return v_add(p, v_mul(lat_axis, lateral))


def build_base_link():
    L = Link("base_link")
    ix, iy, iz = box_inertia(M_BODY, BODY_BOT[0], BODY_BOT[1],
                             BODY_Z1 - BODY_Z0)
    # Battery bank sits low, so the centre of mass is below the shell centre.
    L.inertial = inertial_xml(M_BODY, ix, iy, iz, xyz=(0.0, 0.0, -0.06))

    L.mesh("shell", "model://r2d_rover/meshes/body.obj", (0, 0, 0),
           mat="gunmetal", local="body")

    # Deck plate, black nose trim, charging pad.
    lid_z = BODY_Z1 + LID[2] / 2.0
    L.box("deck_plate", LID, (0, 0, lid_z), mat="alu")
    L.box("deck_nose_trim", (0.13, LID[1], LID[2] + 0.001),
          (LID[0] / 2.0 - 0.065, 0, lid_z + 0.0005), mat="black")

    pz = DECK_TOP
    L.box("pad_base", (PAD_SIZE, PAD_SIZE, 0.004), (PAD_X, 0, pz + 0.002),
          mat="pad")
    for k, (ro, ri) in enumerate(((0.22, 0.205), (0.13, 0.115))):
        L.cyl(f"coil_ring_{k}", ro, 0.002, (PAD_X, 0, pz + 0.0045 + k * 0.001),
              mat="coil")
        L.cyl(f"coil_gap_{k}", ri, 0.002, (PAD_X, 0, pz + 0.0050 + k * 0.001),
              mat="pad")
    hz = pz + 0.0068
    L.box("pad_H_left", (0.16, 0.03, 0.001), (PAD_X, 0.05, hz), mat="white")
    L.box("pad_H_right", (0.16, 0.03, 0.001), (PAD_X, -0.05, hz), mat="white")
    L.box("pad_H_bar", (0.03, 0.10, 0.001), (PAD_X, 0, hz), mat="white")
    half = PAD_SIZE / 2.0 - 0.02
    for sx in (1, -1):
        for sy in (1, -1):
            cx, cy = PAD_X + sx * half, sy * half
            L.box(f"pad_mark_{sx}{sy}_a", (0.06, 0.012, 0.001),
                  (cx - sx * 0.024, cy, hz), mat="white")
            L.box(f"pad_mark_{sx}{sy}_b", (0.012, 0.06, 0.001),
                  (cx, cy - sy * 0.024, hz), mat="white")

    # Electromagnetic latch plates where the drone's landing skids rest.
    for lx in (PAD_X + 0.08, PAD_X - 0.08):
        for ly in (SKID_Y, -SKID_Y):
            L.cyl(f"latch_plate_{'f' if lx > PAD_X else 'r'}"
                  f"{'l' if ly > 0 else 'r'}", 0.028, 0.003,
                  (lx, ly, pz + 0.0055), mat="latch")

    # Front face: control panel, lamps, camera. Rear face: tail lamps.
    for sign in (1, -1):
        c, n, u, tilt = face_frame(sign * BODY_BOT[0] / 2.0,
                                   sign * BODY_TOP[0] / 2.0, "x")
        # Boxes on this face: thin axis (local x) along the outward normal,
        # local z up the slope, local y across the face.
        face_rpy = rpy_from_axes(u, n)
        yax = (0.0, 1.0, 0.0)
        tag = "front" if sign > 0 else "rear"
        plate_t = 0.006
        L.box(f"{tag}_fascia", (plate_t, 0.40, 0.22),
              on_face(c, n, u, 0, 0, plate_t / 2 + 0.001, yax),
              face_rpy, mat="panel")
        lift = plate_t + 0.004
        lamp = "lamp" if sign > 0 else "red"
        for side in (1, -1):
            L.box(f"{tag}_lamp_{side}", (0.008, 0.07, 0.028),
                  on_face(c, n, u, side * 0.15, 0.075, lift, yax),
                  face_rpy, mat=lamp)
        if sign > 0:
            items = [("xt60", (0.012, 0.035, 0.018), 0.15, "yellow"),
                     ("sw_main", (0.012, 0.020, 0.030), 0.10, "red"),
                     ("sw_a", (0.012, 0.030, 0.045), 0.03, "black"),
                     ("sw_b", (0.012, 0.030, 0.045), -0.02, "black")]
            for nm, sz, lat, mt in items:
                L.box(f"front_{nm}", sz, on_face(c, n, u, lat, -0.045, lift, yax),
                      face_rpy, mat=mt)
            L.cyl("front_fan", 0.034, 0.010,
                  on_face(c, n, u, -0.11, -0.045, lift, yax),
                  rpy_from_axes(n, (0, 1, 0)), mat="black")
            L.box("front_cam_housing", (0.05, 0.12, 0.045),
                  on_face(c, n, u, 0, 0.075, 0.028, yax), face_rpy, mat="black")
            L.cyl("front_cam_lens", 0.016, 0.012,
                  on_face(c, n, u, 0, 0.075, 0.056, yax),
                  rpy_from_axes(n, (0, 1, 0)), mat="lens")

    # Side faces: ventilation slats and cable grommets.
    for sign in (1, -1):
        c, n, u, tilt = face_frame(sign * BODY_BOT[1] / 2.0,
                                   sign * BODY_TOP[1] / 2.0, "y")
        tag = "left" if sign > 0 else "right"
        xax = (1.0, 0.0, 0.0)
        # Thin axis (local y) along the outward normal, local z up the slope.
        side_rpy = rpy_from_axes(u, xax)
        for bank, bx in (("fwd", 0.26), ("aft", -0.26)):
            for k in range(5):
                L.box(f"{tag}_vent_{bank}_{k}", (0.15, 0.004, 0.009),
                      on_face(c, n, u, bx, 0.04 - k * 0.019, 0.003, xax),
                      side_rpy, mat="black")
        for gx in (0.07, -0.07):
            L.cyl(f"{tag}_grommet_{'p' if gx > 0 else 'n'}", 0.022, 0.006,
                  on_face(c, n, u, gx, 0.02, 0.003, xax),
                  rpy_from_axes(n, (1, 0, 0)), mat="black")

    # Tubular bumpers with standoffs.
    for sign in (1, -1):
        tag = "front" if sign > 0 else "rear"
        bx = sign * 0.60
        L.cyl(f"{tag}_bumper", 0.018, 0.58, (bx, 0, -0.10), (math.pi / 2, 0, 0))
        for sy in (1, -1):
            L.cyl(f"{tag}_standoff_{sy}", 0.014, 0.08,
                  (sign * 0.56, sy * 0.24, -0.10), (0, math.pi / 2, 0))

    # Rear masts: GNSS receiver (right), comms antenna (left).
    mast_top = DECK_TOP + 0.40
    mast_len = mast_top - (-0.10)
    for sy, tag in ((-1, "gnss"), (1, "comms")):
        L.cyl(f"{tag}_mast", 0.013, mast_len,
              (-0.60, sy * 0.22, -0.10 + mast_len / 2.0))
    L.cyl("gnss_puck", 0.055, 0.022, (-0.60, -0.22, mast_top + 0.011),
          mat="white")
    L.cyl("comms_base", 0.020, 0.025, (-0.60, 0.22, mast_top + 0.0125))
    L.cyl("comms_whip", 0.006, 0.30, (-0.60, 0.22, mast_top + 0.175))

    # Suspension brackets and gearmotors at each corner.
    for sx in (1, -1):
        for sy in (1, -1):
            tag = f"{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}"
            top = (sx * 0.40, sy * 0.28, BODY_Z0)
            bot = (sx * WHEELBASE / 2.0, sy * 0.40, WHEEL_Z + 0.04)
            d = v_sub(bot, top)
            mid = v_add(top, v_mul(d, 0.5))
            length = math.sqrt(v_dot(d, d))
            L.box(f"bracket_{tag}", (0.09, 0.012, length), mid,
                  rpy_from_axes(d, (1, 0, 0)), mat="black")
            L.box(f"gearbox_{tag}", (0.09, 0.09, 0.10),
                  (sx * WHEELBASE / 2.0, sy * 0.405, WHEEL_Z), mat="hub")
            L.cyl(f"motor_{tag}", 0.032, 0.10,
                  (sx * WHEELBASE / 2.0, sy * 0.31, WHEEL_Z),
                  (math.pi / 2, 0, 0), mat="black")

    # Collisions: body, and the deck as a high-friction landing surface.
    L.collide("body_collision",
              f"<box><size>{BODY_BOT[0]} {BODY_BOT[1]} "
              f"{BODY_Z1 - BODY_Z0}</size></box>", (0, 0, 0))
    grip = ("<surface><friction><ode><mu>1.6</mu><mu2>1.6</mu2></ode>"
            "</friction></surface>")
    L.collide("deck_collision",
              f"<box><size>{LID[0]} {LID[1]} 0.026</size></box>",
              (0, 0, BODY_Z1 + 0.013), surface=grip)
    return L


def build_wheel(sx, sy):
    tag = f"{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}"
    xyz = (sx * WHEELBASE / 2.0, sy * TRACK / 2.0, WHEEL_Z)
    W = Link(f"wheel_{tag}", xyz, (-math.pi / 2.0, 0.0, 0.0))
    ia = M_WHEEL * WHEEL_R ** 2 / 2.0
    it = M_WHEEL * (3 * WHEEL_R ** 2 + WHEEL_W ** 2) / 12.0
    W.inertial = inertial_xml(M_WHEEL, it, it, ia)

    # Local +z is world +y after the -pi/2 roll: outward for the left wheels,
    # inward for the right, so hub details go to the side that faces out.
    out = 1.0 if sy > 0 else -1.0
    W.mesh("tyre", "model://r2d_rover/meshes/tyre.obj", (0, 0, 0),
           mat="rubber", local="tyre")
    W.cyl("hub_disc", 0.135, 0.010, (0, 0, out * (WHEEL_W / 2 - 0.004)),
          mat="hub")
    for k in range(5):
        a = 2.0 * math.pi * k / 5.0
        W.box(f"spoke_{k}", (0.10, 0.026, 0.012),
              (0.07 * math.cos(a), 0.07 * math.sin(a),
               out * (WHEEL_W / 2 + 0.003)), (0, 0, a), mat="panel")
    W.cyl("hub_cap", 0.030, 0.020, (0, 0, out * (WHEEL_W / 2 + 0.007)),
          mat="alu")
    grip = ("<surface><friction><ode><mu>1.1</mu><mu2>0.6</mu2></ode>"
            "</friction></surface>")
    W.collide("collision", f"<cylinder><radius>{WHEEL_R}</radius>"
              f"<length>{WHEEL_W}</length></cylinder>", (0, 0, 0), surface=grip)
    return W, tag


def build_sdf():
    base = build_base_link()
    wheels, joints, left, right = [], [], [], []
    for sx in (1, -1):
        for sy in (1, -1):
            w, tag = build_wheel(sx, sy)
            wheels.append(w)
            jn = f"wheel_{tag}_joint"
            (left if sy > 0 else right).append(jn)
            joints.append(
                f'      <joint name="{jn}" type="revolute">\n'
                f"        <parent>base_link</parent><child>{w.name}</child>\n"
                f'        <axis><xyz expressed_in="__model__">0 1 0</xyz>\n'
                f"          <limit><lower>-1e16</lower><upper>1e16</upper>"
                f"</limit></axis>\n      </joint>")

    drive = "\n".join([f"        <left_joint>{j}</left_joint>" for j in left] +
                      [f"        <right_joint>{j}</right_joint>" for j in right])
    links = "\n".join([base.xml()] + [w.xml() for w in wheels])
    sdf = f"""<?xml version="1.0" ?>
<!--
  r2d_rover/model.sdf
  Generated by tools/gen_rover.py. Do not edit by hand; edit the generator.

  Four-wheel skid-steer base station for the Rover2Drone system. Carries the
  drone on an aluminium landing deck with an inductive charging pad (teal
  coil rings, white H and corner fiducials for precision landing).

  Deck surface: {DECK_TOP:.3f} m above base_link, {BASE_H + DECK_TOP:.3f} m above
  flat ground. Pad centre at base_link x = {PAD_X:+.3f}. Mass {M_BODY + 4 * M_WHEEL:.0f} kg.
  Topics: /rover/cmd_vel (in), /rover/odometry (out).
-->
<sdf version="1.9">
  <model name="rover">
{links}
{chr(10).join(joints)}
      <plugin filename="gz-sim-diff-drive-system"
              name="gz::sim::systems::DiffDrive">
{drive}
        <wheel_separation>{TRACK:.3f}</wheel_separation>
        <wheel_radius>{WHEEL_R:.3f}</wheel_radius>
        <max_linear_acceleration>1.0</max_linear_acceleration>
        <max_linear_velocity>1.5</max_linear_velocity>
        <max_angular_velocity>1.0</max_angular_velocity>
        <topic>/rover/cmd_vel</topic>
        <odom_topic>/rover/odometry</odom_topic>
        <tf_topic>/rover/tf</tf_topic>
        <frame_id>odom</frame_id>
        <child_frame_id>rover/base_link</child_frame_id>
        <odom_publish_frequency>30</odom_publish_frequency>
      </plugin>
      <plugin filename="gz-sim-joint-state-publisher-system"
              name="gz::sim::systems::JointStatePublisher"/>
      <!-- Drone latch. Attaches automatically when {DRONE_MODEL} appears
           (retrying silently until PX4 spawns it); latch_manager then
           re-seats it flat on the pad. -->
      <plugin filename="gz-sim-detachable-joint-system"
              name="gz::sim::systems::DetachableJoint">
        <parent_link>base_link</parent_link>
        <child_model>{DRONE_MODEL}</child_model>
        <child_link>{DRONE_LINK}</child_link>
        <detach_topic>/rover/latch/detach</detach_topic>
        <attach_topic>/rover/latch/attach</attach_topic>
        <output_topic>/rover/latch/state</output_topic>
        <suppress_child_warning>true</suppress_child_warning>
      </plugin>
  </model>
</sdf>
"""
    return sdf, [base] + wheels


# ----------------------------------------------------------------------------
# Optional preview render, for checking geometry without launching Gazebo.
# ----------------------------------------------------------------------------
def render_preview(links, meshes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def to_rgb(s):
        return tuple(float(v) for v in s.split()[:3])

    def box_faces(sx, sy, sz):
        x, y, z = sx / 2, sy / 2, sz / 2
        c = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
             (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
        idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
               (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
        return [[c[i] for i in f] for f in idx]

    def cyl_faces(r, l, n=24):
        top = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), l / 2) for i in range(n)]
        bot = [(p[0], p[1], -l / 2) for p in top]
        faces = [top, bot]
        for i in range(n):
            j = (i + 1) % n
            faces.append([bot[i], bot[j], top[j], top[i]])
        return faces

    polys, colors = [], []
    for link in links:
        LR = rot_matrix(*link.rpy)
        for prim, xyz, rpy, diffuse in link.prims:
            if prim[0] == "box":
                faces = box_faces(*prim[1])
            elif prim[0] == "cyl":
                faces = cyl_faces(prim[1], prim[2])
            else:
                faces = [[t[0], t[1], t[2]] for t in meshes[prim[1]].tris]
            R = rot_matrix(*rpy)
            col = to_rgb(diffuse)
            for f in faces:
                pts = []
                for p in f:
                    q = v_add(mat_apply(R, p), xyz)
                    q = v_add(mat_apply(LR, q), link.xyz)
                    pts.append((q[0], q[1], q[2] + BASE_H))
                polys.append(pts)
                colors.append(col)

    # Simple Lambert shading so shapes read clearly in the preview.
    light = v_norm((0.4, -0.5, 0.8))
    shaded = []
    for pts, col in zip(polys, colors):
        nrm = v_cross(v_sub(pts[1], pts[0]), v_sub(pts[2], pts[0]))
        k = 0.45
        if v_dot(nrm, nrm) > 1e-18:
            k = 0.35 + 0.65 * abs(v_dot(v_norm(nrm), light))
        shaded.append(tuple(min(1.0, c * k + 0.04) for c in col))

    fig = plt.figure(figsize=(14, 7), dpi=110)
    for k, (el, az, title) in enumerate(((22, -52, "front-left"),
                                         (30, 128, "rear-right"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        pc = Poly3DCollection(polys, facecolors=shaded, linewidths=0)
        ax.add_collection3d(pc)
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_zlim(0, 1.5)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=el, azim=az)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path, facecolor="white")
    print(f"  preview written to {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--preview", help="Render a PNG check image to this path")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    mdir = os.path.join(repo, "models", "r2d_rover")
    os.makedirs(os.path.join(mdir, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(repo, "config"), exist_ok=True)

    body, tyre = build_body(), build_tyre()
    nb = body.write(os.path.join(mdir, "meshes", "body.obj"), "rover_body")
    nt = tyre.write(os.path.join(mdir, "meshes", "tyre.obj"), "rover_tyre")
    print(f"  body.obj {nb} triangles, tyre.obj {nt} triangles")

    sdf, links = build_sdf()
    with open(os.path.join(mdir, "model.sdf"), "w") as f:
        f.write(sdf)
    with open(os.path.join(mdir, "model.config"), "w") as f:
        f.write('<?xml version="1.0"?>\n<model>\n  <name>r2d_rover</name>\n'
                '  <version>2.0</version>\n  <sdf version="1.9">model.sdf</sdf>\n'
                '  <description>Rover2Drone four-wheel base station with '
                'charging landing deck.</description>\n</model>\n')

    total_mass = M_BODY + 4 * M_WHEEL
    with open(os.path.join(repo, "config", "rover.yaml"), "w") as f:
        f.write(f"""# rover.yaml
# Generated by tools/gen_rover.py. Do not edit by hand; edit the generator.
# Rover geometry read by place_turbines.py (spawn heights) and by the
# landing and charging nodes (pad location). Frame: rover/base_link.
base_height_m: {BASE_H:.4f}           # base_link above flat ground
deck_height_m: {DECK_TOP:.4f}           # deck surface above base_link
deck_top_above_ground_m: {BASE_H + DECK_TOP:.4f}
deck_size_m: [{LID[0]:.3f}, {LID[1]:.3f}]
pad_center_x_m: {PAD_X:.4f}
pad_center_y_m: 0.0
pad_size_m: {PAD_SIZE:.3f}
wheel_radius_m: {WHEEL_R:.3f}
wheel_separation_m: {TRACK:.3f}
wheelbase_m: {WHEELBASE:.3f}
mass_kg: {total_mass:.1f}
latch_child_model: {DRONE_MODEL}
latch_skid_y_m: {SKID_Y:.3f}
""")
    print(f"  wrote {mdir}/model.sdf and config/rover.yaml")
    print(f"  deck surface {BASE_H + DECK_TOP:.3f} m above ground, "
          f"mass {total_mass:.0f} kg")

    if args.preview:
        render_preview(links, {"body": body, "tyre": tyre}, args.preview)


if __name__ == "__main__":
    main()
