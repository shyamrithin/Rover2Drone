#!/usr/bin/env python3
"""
gen_turbine_meshes.py
=====================

Generates smooth OBJ meshes for the Rover2Drone demonstration wind turbine.

Two meshes are produced:

  blade.obj  A lofted wind-turbine blade. Sections are NACA 4-digit airfoils
             whose thickness ratio, chord and twist all vary along the span.
             The inboard 22% blends from a circular root (where the blade
             bolts to the hub) into the first true airfoil section, which is
             how real blades are built. Vertex normals are averaged across
             adjacent faces so ogre2 shades the surface smoothly rather than
             faceted.

  tower.obj  A tapered tubular tower with a base flare, as a single smooth
             surface instead of stacked cylinders.

Output is Wavefront OBJ with vertex normals. Standard library only.

Coordinate conventions (blade local frame, matching turbine_site.sdf):
  +Z  spanwise, root at z=0, tip at z=BLADE_LENGTH
  +Y  chordwise (trailing edge toward +Y at zero twist)
  +X  thickness / flapwise

Usage:
  python3 gen_turbine_meshes.py [outdir]
"""

import math
import os
import sys

# ----------------------------------------------------------------------------
# Blade planform parameters. Tune these to reshape the blade.
# ----------------------------------------------------------------------------
BLADE_LENGTH = 6.0        # metres, root to tip
ROOT_DIAMETER = 0.42      # metres, circular root
MAX_CHORD = 0.78          # metres, at the widest station
TIP_CHORD = 0.14          # metres
MAX_CHORD_STATION = 0.18  # fraction of span where chord peaks
BLEND_END = 0.22          # fraction of span where root blend finishes
ROOT_TWIST_DEG = 18.0     # degrees, washout at the root
TIP_TWIST_DEG = 1.0       # degrees
ROOT_THICKNESS = 0.42     # t/c at the first airfoil station
TIP_THICKNESS = 0.14      # t/c at the tip
PITCH_AXIS = 0.30         # chord fraction the sections rotate about
CAMBER_M = 0.04           # NACA first digit / 100
CAMBER_P = 0.40           # NACA second digit / 10

N_SPAN = 60               # spanwise sections
N_HALF = 40               # chordwise points per surface (2*N_HALF per loop)

# ----------------------------------------------------------------------------
# Tower parameters.
# ----------------------------------------------------------------------------
TOWER_HEIGHT = 12.5
TOWER_BASE_R = 0.95
TOWER_TOP_R = 0.42
TOWER_FLARE_H = 0.9       # height over which the base flare blends in
N_TOWER_RADIAL = 56
N_TOWER_AXIAL = 28


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    """Hermite ease so blends have no visible crease at their endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def naca_half_thickness(x, t):
    """NACA 4-digit thickness distribution, closed trailing edge."""
    return 5.0 * t * (0.2969 * math.sqrt(max(x, 0.0))
                      - 0.1260 * x
                      - 0.3516 * x * x
                      + 0.2843 * x ** 3
                      - 0.1036 * x ** 4)


def naca_camber(x, m, p):
    """Mean camber line and its slope."""
    if m <= 1e-9:
        return 0.0, 0.0
    if x < p:
        yc = (m / (p * p)) * (2.0 * p * x - x * x)
        dy = (2.0 * m / (p * p)) * (p - x)
    else:
        q = 1.0 - p
        yc = (m / (q * q)) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x)
        dy = (2.0 * m / (q * q)) * (p - x)
    return yc, dy


def airfoil_loop(n_half, thickness, m, p):
    """
    Closed airfoil outline as a list of (chordwise, thickness) pairs.

    Ordering starts at the trailing edge, runs forward over the upper
    surface to the leading edge, then aft along the lower surface. Cosine
    spacing clusters points where curvature is highest.
    """
    xs = [0.5 * (1.0 - math.cos(math.pi * i / n_half)) for i in range(n_half + 1)]

    upper, lower = [], []
    for x in xs:
        yt = naca_half_thickness(x, thickness)
        yc, dy = naca_camber(x, m, p)
        theta = math.atan(dy)
        upper.append((x - yt * math.sin(theta), yc + yt * math.cos(theta)))
        lower.append((x + yt * math.sin(theta), yc - yt * math.cos(theta)))

    loop = list(reversed(upper))          # TE -> LE over the top
    loop += lower[1:-1]                   # LE -> TE underneath, no duplicates
    return loop


def circle_loop(n_points):
    """Unit-chord circle parametrised to match airfoil_loop's ordering."""
    pts = []
    for i in range(n_points):
        a = 2.0 * math.pi * i / n_points
        pts.append((0.5 + 0.5 * math.cos(a), 0.5 * math.sin(a)))
    return pts


def chord_at(r):
    """Chord length as a function of span position r in [0, 1]."""
    if r <= MAX_CHORD_STATION:
        return lerp(ROOT_DIAMETER, MAX_CHORD, smoothstep(r / MAX_CHORD_STATION))
    t = (r - MAX_CHORD_STATION) / (1.0 - MAX_CHORD_STATION)
    return lerp(MAX_CHORD, TIP_CHORD, t ** 0.85)


def thickness_at(r):
    if r <= BLEND_END:
        return ROOT_THICKNESS
    t = (r - BLEND_END) / (1.0 - BLEND_END)
    return lerp(ROOT_THICKNESS, TIP_THICKNESS, t ** 0.7)


def twist_at(r):
    """Radians. Nonlinear washout, steepest inboard, as on a real blade."""
    deg = lerp(TIP_TWIST_DEG, ROOT_TWIST_DEG, (1.0 - r) ** 1.6)
    return math.radians(deg)


def build_blade():
    """Returns (vertices, faces) with faces as tuples of 0-based indices."""
    n_loop = 2 * N_HALF
    circ = circle_loop(n_loop)

    verts = []
    rings = []

    for s in range(N_SPAN + 1):
        r = s / N_SPAN
        c = chord_at(r)
        tw = twist_at(r)
        blend = 1.0 - smoothstep(min(r / BLEND_END, 1.0))

        foil = airfoil_loop(N_HALF, thickness_at(r), CAMBER_M, CAMBER_P)

        ring = []
        for i in range(n_loop):
            fx, fy = foil[i]
            cx, cy = circ[i]
            px = lerp(fx, cx, blend)
            py = lerp(fy, cy, blend)

            # Offset to the pitch axis, scale to chord.
            chordwise = (px - PITCH_AXIS) * c
            thick = py * c

            # Twist about the span axis.
            ct, st = math.cos(tw), math.sin(tw)
            x = thick * ct - chordwise * st
            y = thick * st + chordwise * ct

            ring.append(len(verts))
            verts.append((x, y, r * BLADE_LENGTH))
        rings.append(ring)

    faces = []
    for s in range(N_SPAN):
        a, b = rings[s], rings[s + 1]
        for i in range(n_loop):
            j = (i + 1) % n_loop
            faces.append((a[i], a[j], b[j]))
            faces.append((a[i], b[j], b[i]))

    # Cap the root and the tip with triangle fans around a centroid.
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        cx = sum(verts[k][0] for k in ring) / len(ring)
        cy = sum(verts[k][1] for k in ring) / len(ring)
        cz = verts[ring[0]][2]
        centre = len(verts)
        verts.append((cx, cy, cz))
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            if flip:
                faces.append((centre, ring[j], ring[i]))
            else:
                faces.append((centre, ring[i], ring[j]))

    return verts, faces


def build_tower():
    verts = []
    rings = []
    for a in range(N_TOWER_AXIAL + 1):
        t = a / N_TOWER_AXIAL
        z = t * TOWER_HEIGHT
        radius = lerp(TOWER_BASE_R * 0.78, TOWER_TOP_R, t ** 0.85)
        if z < TOWER_FLARE_H:
            f = 1.0 - smoothstep(z / TOWER_FLARE_H)
            radius = lerp(radius, TOWER_BASE_R, f * 0.55)
        ring = []
        for i in range(N_TOWER_RADIAL):
            ang = 2.0 * math.pi * i / N_TOWER_RADIAL
            ring.append(len(verts))
            verts.append((radius * math.cos(ang), radius * math.sin(ang), z))
        rings.append(ring)

    faces = []
    for a in range(N_TOWER_AXIAL):
        lo, hi = rings[a], rings[a + 1]
        for i in range(N_TOWER_RADIAL):
            j = (i + 1) % N_TOWER_RADIAL
            faces.append((lo[i], lo[j], hi[j]))
            faces.append((lo[i], hi[j], hi[i]))

    base = len(verts)
    verts.append((0.0, 0.0, 0.0))
    for i in range(N_TOWER_RADIAL):
        j = (i + 1) % N_TOWER_RADIAL
        faces.append((base, rings[0][j], rings[0][i]))

    top = len(verts)
    verts.append((0.0, 0.0, TOWER_HEIGHT))
    for i in range(N_TOWER_RADIAL):
        j = (i + 1) % N_TOWER_RADIAL
        faces.append((top, rings[-1][i], rings[-1][j]))

    return verts, faces


def vertex_normals(verts, faces):
    """Area-weighted average of adjacent face normals, giving smooth shading."""
    acc = [[0.0, 0.0, 0.0] for _ in verts]
    for (i0, i1, i2) in faces:
        ax, ay, az = verts[i0]
        bx, by, bz = verts[i1]
        cx, cy, cz = verts[i2]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for k in (i0, i1, i2):
            acc[k][0] += nx
            acc[k][1] += ny
            acc[k][2] += nz

    out = []
    for n in acc:
        mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if mag < 1e-12:
            out.append((0.0, 0.0, 1.0))
        else:
            out.append((n[0] / mag, n[1] / mag, n[2] / mag))
    return out


def write_obj(path, verts, faces, name):
    norms = vertex_normals(verts, faces)
    with open(path, "w") as f:
        f.write("# Generated by gen_turbine_meshes.py, do not edit by hand\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for (a, b, c) in faces:
            f.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")
    return len(verts), len(faces)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    bv, bf = build_blade()
    nv, nf = write_obj(os.path.join(outdir, "blade.obj"), bv, bf, "turbine_blade")
    print(f"blade.obj  {nv:6d} vertices  {nf:6d} triangles")

    tv, tf = build_tower()
    nv, nf = write_obj(os.path.join(outdir, "tower.obj"), tv, tf, "turbine_tower")
    print(f"tower.obj  {nv:6d} vertices  {nf:6d} triangles")


if __name__ == "__main__":
    main()
