#!/usr/bin/env python3
"""
gen_terrain_collision.py
========================

Replaces the terrain's heightmap collision with a triangle-mesh collision
of the same surface.

Why
---
In Gazebo Harmonic's DART physics, heightmap collision is unreliable for
wheeled vehicles: with the default ODE collision detector the rover floats
with no traction, and with the Bullet detector the heightmap is not
collided at all. Triangle-mesh collision is robust with the default
detector, so the heightmap stays for rendering and a mesh carries physics.

Geometry
--------
The mesh is built from the (road-carved) collision heightmap, so it is
exactly the surface validated in V2:

  fine    every collision-grid vertex (2.93 m by default) inside the
          operating area: rover spawn, turbines and access roads there,
          plus a margin
  coarse  every 8th vertex (about 23 m) over the rest of the terrain, so
          nothing falls out of the world, with cells covered by the fine
          patch skipped

Written as binary STL (compact, fast to load; collision needs no normals
beyond per-face). The terrain model.sdf collision block is rewritten to
reference it with explicit friction.

Pipeline position: after make_terrain_model.py and gen_roads.py, because
it meshes the carved surface. make_terrain_model.py rewrites model.sdf with
a heightmap collision, so rerun this after every terrain rebuild.

Usage
  python3 tools/gen_terrain_collision.py --terrain-model attappadi
"""

import argparse
import json
import os
import re

import numpy as np

from terrain_io import Terrain, read_turbines

STL = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def grid_tris(Z, X, Y, rows, cols, keep=None):
    """Two upward-facing triangles per cell over the given row/col indices."""
    r = np.asarray(rows)[:-1, None]
    c = np.asarray(cols)[None, :-1]
    r1 = np.asarray(rows)[1:, None]
    c1 = np.asarray(cols)[None, 1:]
    rr, cc = np.broadcast_arrays(r, c)
    rr1, cc1 = np.broadcast_arrays(r1, c1)
    if keep is not None:
        m = keep(rr, cc, rr1, cc1)
        rr, cc, rr1, cc1 = rr[m], cc[m], rr1[m], cc1[m]
    rr, cc, rr1, cc1 = (a.ravel() for a in (rr, cc, rr1, cc1))

    def p(ri, ci):
        return np.stack([X[ci], Y[ri], Z[ri, ci]], axis=-1)

    A, B, C, D = p(rr, cc), p(rr1, cc), p(rr1, cc1), p(rr, cc1)
    return np.concatenate([np.stack([A, B, C], 1), np.stack([A, C, D], 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain-model", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--margin", type=float, default=200.0)
    ap.add_argument("--coarse-step", type=int, default=8)
    a = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(a.repo))
    ter = Terrain(os.path.join(repo, "models", a.terrain_model))
    m = ter.meta
    hm = ter.heightmap("collision")
    n = hm.shape[0]
    res = ter.extent / (n - 1)
    Z = hm * m["zrange_m"] + m["z_offset_m"]
    X = -ter.half + np.arange(n) * res
    Y = ter.half - np.arange(n) * res

    # Operating area: spawn, turbines, and roads within reach of them.
    turb = read_turbines(os.path.join(repo, "config", "turbines.yaml"))
    xs, ys = [0.0] + [t["x"] for t in turb], [0.0] + [t["y"] for t in turb]
    x0, x1 = min(xs) - a.margin, max(xs) + a.margin
    y0, y1 = min(ys) - a.margin, max(ys) + a.margin
    c0 = max(int((x0 + ter.half) / res), 0)
    c1 = min(int((x1 + ter.half) / res) + 1, n - 1)
    r0 = max(int((ter.half - y1) / res), 0)
    r1 = min(int((ter.half - y0) / res) + 1, n - 1)

    fine = grid_tris(Z, X, Y, range(r0, r1 + 1), range(c0, c1 + 1))

    s = a.coarse_step
    idx = list(range(0, n, s))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    def outside_fine(rr, cc, rr1, cc1):
        return ~((rr >= r0) & (rr1 <= r1) & (cc >= c0) & (cc1 <= c1))

    coarse = grid_tris(Z, X, Y, idx, idx, keep=outside_fine)
    tris = np.concatenate([fine, coarse]).astype(np.float32)

    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(tris), dtype=STL)
    rec["n"], rec["v"] = nrm, tris

    mesh_dir = os.path.join(ter.dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    path = os.path.join(mesh_dir, "collision.stl")
    with open(path, "wb") as f:
        f.write(b"rover2drone terrain collision".ljust(80, b" "))
        f.write(np.uint32(len(rec)).tobytes())
        f.write(rec.tobytes())

    sdf_path = os.path.join(ter.dir, "model.sdf")
    sdf = open(sdf_path).read()
    new_col = (f'<collision name="collision">\n'
               f'        <geometry><mesh><uri>model://{a.terrain_model}/meshes/'
               f'collision.stl</uri></mesh></geometry>\n'
               f'        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode>'
               f'</friction></surface>\n      </collision>')
    sdf, k = re.subn(r'<collision name="collision">.*?</collision>', new_col,
                     sdf, count=1, flags=re.S)
    if k != 1:
        raise SystemExit(f"Could not find the collision block in {sdf_path}")
    open(sdf_path, "w").write(sdf)

    fa = (x1 - x0) * (y1 - y0) / 1e6
    print(f"  fine {len(fine)} tris ({res:.2f} m, {fa:.2f} km2), coarse "
          f"{len(coarse)} tris ({res * s:.1f} m) -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB); model.sdf collision -> mesh")


if __name__ == "__main__":
    main()
