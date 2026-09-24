#!/usr/bin/env python3
# =============================================================================
# File:        tools/road_planner.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     python3, numpy, matplotlib (system); optional: terrain_io.py
#              + the attappadi terrain model for the aerial-image backdrop
# =============================================================================
"""
road_planner.py
===============

Builds a routable graph from config/roads.json and plans the rover's route
from its spawn to a chosen wind turbine with A*.

Description
-----------
config/roads.json (written by gen_roads.py) holds 15 separate OSM polylines,
resampled every 4 m, in world ENU metres with road-surface z. OSM's shared
nodes were lost in the resampling, so junctions are recovered geometrically:

  1. End-to-road snaps: every polyline endpoint within --snap-m of another
     polyline becomes a junction. A node is inserted into the other polyline
     at the projection point and joined to the endpoint by a short edge.
  2. Crossings: interior segment intersections between polylines become a
     shared node on both.
  3. Consecutive vertices along each polyline become bidirectional edges with
     horizontal length, 3D length and signed grade.

Routing (A*, admissible heuristics, per-segment hard grade cap):

  --cost distance   minimise 3D path length          (h = 2D straight line)
  --cost energy     minimise rover traction energy   (h = lower bound from
                    rolling resistance + net climb)

  Rover energy per segment (planning; no regen, traction cannot go negative):
      E = max(0, m g (Crr * dxy + dz)) / eta
  The reported energy also gives a with-regen estimate if --regen > 0.

The goal is the road node nearest (2D) to the turbine base, or, with
--stop-radius R, the cheapest reachable node within R metres of the base.
The rover's start is joined to the nearest road node by a straight access leg.

Outputs
-------
  config/routes/route_<turbine>.json   waypoints (x, y, z, yaw, s, grade) +
                                       summary stats, read by the waypoint
                                       follower
  docs/routes/route_<turbine>.png      map (roads coloured by grade, route,
                                       turbines) with the elevation and grade
                                       profile directly below
  stdout                               per-turbine summary table (--turbine all)

Usage
-----
  cd ~/Rover2Drone
  /usr/bin/python3 tools/road_planner.py --turbine turbine_03 --plot
  /usr/bin/python3 tools/road_planner.py --turbine all --cost energy --plot
  /usr/bin/python3 tools/road_planner.py --turbine turbine_04 --stop-radius 120

Re-run after any rebuild of the world (§5 of HANDOFF.md): the graph is built
from roads.json at run time, nothing is cached.
"""

import argparse
import heapq
import json
import math
import os
import sys

import numpy as np

G = 9.81
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------- config io
def read_turbines_yaml(path):
    """Parse the flow-style turbine lines of config/turbines.yaml (no PyYAML)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- {"):
                continue
            body = line[3:].rstrip("}")
            rec = {}
            for kv in body.split(","):
                k, v = kv.split(":", 1)
                k, v = k.strip(), v.strip()
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            out.append(rec)
    if not out:
        raise SystemExit(f"No turbines found in {path}")
    return out


def read_rover_mass(path, default=106.0):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("mass_kg:"):
                    return float(line.split(":")[1].split("#")[0])
    except OSError:
        pass
    return default


# ------------------------------------------------------------------- graph
class RoadGraph:
    """Undirected road graph; nodes are 3D points, edges carry geometry."""

    def __init__(self):
        self.xyz = []          # node -> (x, y, z)
        self.adj = []          # node -> list of (nbr, dxy, dz, road_idx)
        self.road_of_node = [] # node -> road index (first owner)

    def add_node(self, p, road):
        self.xyz.append(tuple(float(v) for v in p))
        self.adj.append([])
        self.road_of_node.append(road)
        return len(self.xyz) - 1

    def add_edge(self, a, b, road):
        if a == b:
            return
        pa, pb = self.xyz[a], self.xyz[b]
        dxy = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        dz = pb[2] - pa[2]
        self.adj[a].append((b, dxy, dz, road))
        self.adj[b].append((a, dxy, -dz, road))

    @property
    def n(self):
        return len(self.xyz)

    def nearest(self, x, y):
        P = np.asarray(self.xyz)
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        i = int(d.argmin())
        return i, float(d[i])

    def components(self):
        comp = [-1] * self.n
        c = 0
        for s in range(self.n):
            if comp[s] >= 0:
                continue
            stack = [s]
            comp[s] = c
            while stack:
                u = stack.pop()
                for v, *_ in self.adj[u]:
                    if comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
        return comp, c


def project_to_polyline(p, P):
    """Closest point on polyline P (N x 3) to 2D point p: (dist, seg, t, xyz)."""
    a, b = P[:-1], P[1:]
    ab = b[:, :2] - a[:, :2]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip(((p[:2] - a[:, :2]) * ab).sum(1) / L2, 0.0, 1.0)
    q = a + (b - a) * t[:, None]
    d = np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])
    k = int(d.argmin())
    return float(d[k]), k, float(t[k]), q[k]


def seg_intersections(P, Q):
    """Proper 2D intersections between segments of P and Q: [(i, s, j, u)]."""
    out = []
    pa, pb = P[:-1, :2], P[1:, :2]
    qa, qb = Q[:-1, :2], Q[1:, :2]
    # Bounding-box prefilter per P segment against all Q segments.
    for i in range(len(pa)):
        a, b = pa[i], pb[i]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        m = ((np.minimum(qa, qb) <= hi).all(1) & (np.maximum(qa, qb) >= lo).all(1))
        for j in np.nonzero(m)[0]:
            c, d = qa[j], qb[j]
            r, s = b - a, d - c
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            w = c - a
            tt = (w[0] * s[1] - w[1] * s[0]) / den
            uu = (w[0] * r[1] - w[1] * r[0]) / den
            if 1e-6 < tt < 1 - 1e-6 and 1e-6 < uu < 1 - 1e-6:
                out.append((i, tt, int(j), uu))
    return out


def build_graph(roads, snap_m=5.0, merge_m=1.0, log=print):
    """Turn roads.json polylines into a RoadGraph with recovered junctions."""
    polys = [np.asarray(r["points"], dtype=float) for r in roads]
    g = RoadGraph()
    vnode = []   # per road: node id of each original vertex
    for ri, P in enumerate(polys):
        vnode.append([g.add_node(p, ri) for p in P])

    # insertions[ri] -> list of (segment k, t, node) splitting segment k
    insertions = [[] for _ in polys]
    extra_edges = []   # (a, b, road)
    n_snap = n_merge = n_cross = 0

    # 1. endpoint snaps
    for ri, P in enumerate(polys):
        for end, vi in ((0, 0), (1, len(P) - 1)):
            p = P[vi]
            best = None
            for rj, Q in enumerate(polys):
                if rj == ri:
                    continue
                d, k, t, q = project_to_polyline(p, Q)
                if d <= snap_m and (best is None or d < best[0]):
                    best = (d, rj, k, t, q)
            if best is None:
                continue
            d, rj, k, t, q = best
            Q = polys[rj]
            a = vnode[ri][vi]
            # Merge onto an existing vertex of rj if the projection is on it.
            d0 = math.hypot(*(q[:2] - Q[k][:2]))
            d1 = math.hypot(*(q[:2] - Q[k + 1][:2]))
            if min(d0, d1) <= merge_m:
                b = vnode[rj][k] if d0 <= d1 else vnode[rj][k + 1]
                n_merge += 1
            else:
                b = g.add_node(q, rj)
                insertions[rj].append((k, t, b))
                n_snap += 1
            extra_edges.append((a, b, ri))

    # 2. interior crossings
    for ri in range(len(polys)):
        for rj in range(ri + 1, len(polys)):
            for (i, s, j, u) in seg_intersections(polys[ri], polys[rj]):
                P, Q = polys[ri], polys[rj]
                pz = P[i] + (P[i + 1] - P[i]) * s
                qz = Q[j] + (Q[j + 1] - Q[j]) * u
                node = g.add_node(((pz[0] + qz[0]) / 2, (pz[1] + qz[1]) / 2,
                                   (pz[2] + qz[2]) / 2), ri)
                insertions[ri].append((i, s, node))
                insertions[rj].append((j, u, node))
                n_cross += 1

    # 3. chain vertices (with insertions) into edges
    for ri, P in enumerate(polys):
        ins = sorted(insertions[ri])
        seq, ii = [], 0
        for k in range(len(P)):
            seq.append(vnode[ri][k])
            while ii < len(ins) and ins[ii][0] == k:
                seq.append(ins[ii][2])
                ii += 1
        for a, b in zip(seq[:-1], seq[1:]):
            g.add_edge(a, b, ri)
    for a, b, r in extra_edges:
        g.add_edge(a, b, r)

    steps = [abs(g.xyz[a][2] - g.xyz[b][2]) for a, b, _ in extra_edges]
    for ri, P in enumerate(polys):
        for k, t, node in insertions[ri]:
            zq = P[k][2] + (P[k + 1][2] - P[k][2]) * t
            steps.append(abs(g.xyz[node][2] - zq))
    comp, nc = g.components()
    sizes = np.bincount(comp)
    log(f"[graph] {len(polys)} roads -> {g.n} nodes, "
        f"{sum(len(a) for a in g.adj) // 2} edges; junctions: {n_merge} vertex "
        f"merges, {n_snap} snaps (<= {snap_m} m), {n_cross} crossings; "
        f"{nc} component(s), largest {sizes.max()} nodes")
    if steps:
        log(f"[graph] junction z mismatch between separately carved roads: "
            f"max {max(steps):.2f} m, mean {np.mean(steps):.2f} m "
            f"(check these junctions in Gazebo for steps)")
    g.junction_steps = steps
    return g, comp


# -------------------------------------------------------------------- A*
class Params:
    def __init__(self, a):
        self.cost = a.cost
        self.max_grade = a.max_grade / 100.0
        self.mass = a.mass + a.payload
        self.crr = a.crr
        self.eta = a.eta
        self.regen = a.regen
        self.min_run = a.min_grade_run


def edge_cost(dxy, dz, p):
    # Edges shorter than min_run are junction connectors or split stubs whose
    # dz is an artefact of each road being carved separately; the grade cap
    # applies to real road segments only.
    if dxy >= p.min_run and abs(dz) / dxy > p.max_grade:
        return None
    if p.cost == "distance":
        return math.hypot(dxy, dz)
    return max(0.0, p.mass * G * (p.crr * dxy + dz)) / p.eta


def heuristic(a, goal_xyz, p):
    d = math.hypot(goal_xyz[0] - a[0], goal_xyz[1] - a[1])
    if p.cost == "distance":
        return d
    return max(0.0, p.mass * G * (p.crr * d + goal_xyz[2] - a[2])) / p.eta


def astar(g, s, goal, p):
    gx = g.xyz[goal]
    best = {s: 0.0}
    prev = {s: None}
    pq = [(heuristic(g.xyz[s], gx, p), 0.0, s)]
    closed = set()
    while pq:
        f, c, u = heapq.heappop(pq)
        if u in closed:
            continue
        if u == goal:
            path = []
            while u is not None:
                path.append(u)
                u = prev[u]
            return path[::-1], c
        closed.add(u)
        for v, dxy, dz, _ in g.adj[u]:
            ec = edge_cost(dxy, dz, p)
            if ec is None or v in closed:
                continue
            nc = c + ec
            if nc < best.get(v, math.inf):
                best[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc + heuristic(g.xyz[v], gx, p), nc, v))
    return None, math.inf


def dijkstra_all(g, s, p):
    dist = {s: 0.0}
    pq = [(0.0, s)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist.get(u, math.inf):
            continue
        for v, dxy, dz, _ in g.adj[u]:
            ec = edge_cost(dxy, dz, p)
            if ec is None:
                continue
            if c + ec < dist.get(v, math.inf):
                dist[v] = c + ec
                heapq.heappush(pq, (c + ec, v))
    return dist


# ---------------------------------------------------------------- route
def route_stats(pts, p):
    """pts: (N,3) world points incl. access leg. Returns dict + per-point arrays."""
    d = np.diff(pts, axis=0)
    dxy = np.hypot(d[:, 0], d[:, 1])
    dz = d[:, 2]
    l3 = np.hypot(dxy, dz)
    grade = np.where(dxy > 1e-9, dz / np.maximum(dxy, 1e-9), 0.0)
    trac = p.mass * G * (p.crr * dxy + dz)
    e_plan = np.maximum(0.0, trac).sum() / p.eta
    e_regen = (np.maximum(0.0, trac).sum() / p.eta
               + np.minimum(0.0, trac).sum() * p.eta * p.regen)
    s = np.concatenate([[0.0], np.cumsum(l3)])
    yaw = np.arctan2(d[:, 1], d[:, 0])
    yaw = np.concatenate([yaw, yaw[-1:]])
    # For the per-waypoint grade, junction stubs inherit the previous segment's
    # grade instead of their artefact value (a 1 m step over 0.1 m = 1000%).
    gclean = grade.copy()
    for i in np.nonzero(dxy < p.min_run)[0]:
        gclean[i] = gclean[i - 1] if i > 0 else 0.0
    grade_pt = np.concatenate([gclean, gclean[-1:]])
    st = {
        "length_3d_m": round(float(l3.sum()), 1),
        "length_2d_m": round(float(dxy.sum()), 1),
        "climb_m": round(float(np.maximum(dz, 0).sum()), 1),
        "descent_m": round(float(-np.minimum(dz, 0).sum()), 1),
        "net_dz_m": round(float(pts[-1, 2] - pts[0, 2]), 1),
        # junction stubs (< min_run) excluded: their dz is a carving artefact
        "grade_max_pct": round(float(100 * np.abs(grade[dxy >= p.min_run]).max()), 1),
        "grade_mean_abs_pct": round(float(100 * np.average(np.abs(grade), weights=dxy)), 1),
        "energy_wh_no_regen": round(float(e_plan / 3600.0), 2),
        "energy_wh_with_regen": round(float(e_regen / 3600.0), 2),
    }
    return st, s, yaw, grade_pt


def plan_to_turbine(g, comp, turb, start_xy, a, p):
    s_node, s_gap = g.nearest(*start_xy)
    tx, ty = turb["x"], turb["y"]
    P = np.asarray(g.xyz)

    if a.stop_radius > 0:
        dist = dijkstra_all(g, s_node, p)
        dd = np.hypot(P[:, 0] - tx, P[:, 1] - ty)
        cand = [i for i in np.nonzero(dd <= a.stop_radius)[0] if i in dist]
        if not cand:
            return None
        goal = min(cand, key=lambda i: dist[i])
    else:
        # nearest node in the start's component
        dd = np.hypot(P[:, 0] - tx, P[:, 1] - ty)
        dd[np.asarray(comp) != comp[s_node]] = np.inf
        goal = int(dd.argmin())

    path, cost = astar(g, s_node, goal, p)
    if path is None:
        return None
    start_z = g.xyz[s_node][2] if a.start_z is None else a.start_z
    pts = np.array([(start_xy[0], start_xy[1], start_z)] + [g.xyz[i] for i in path])
    st, s, yaw, grade = route_stats(pts, p)
    gx, gy, gz = g.xyz[goal]
    st.update({
        "turbine": turb["id"],
        "start_access_m": round(s_gap, 1),
        "stop_xyz": [round(gx, 2), round(gy, 2), round(gz, 2)],
        "stop_to_base_2d_m": round(math.hypot(gx - tx, gy - ty), 1),
        "base_minus_stop_z_m": round(turb["ground_z"] - gz, 1),
        "hub_minus_stop_z_m": round(turb["hub_z"] - gz, 1),
        "planner_cost": round(cost, 1),
    })
    wps = [{"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3),
            "yaw": round(float(w), 4), "s": round(float(si), 2),
            "grade_pct": round(float(100 * gr), 2)}
           for (x, y, z), w, si, gr in zip(pts, yaw, s, grade)]
    return st, wps, pts, s, grade


# ------------------------------------------------------------------ plot
def load_backdrop(model_dir):
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from terrain_io import Terrain
        from PIL import Image
        t = Terrain(model_dir)
        img = Image.open(os.path.join(t.dir, t.meta["texture"]))
        img.thumbnail((2048, 2048))
        return np.asarray(img), t.half
    except Exception as e:   # LFS pointer, missing model, no rasterio, ...
        print(f"[plot] no aerial backdrop ({type(e).__name__}); roads only")
        return None, None


def plot_route(roads, turbines, res, start_xy, out_png, backdrop, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    st, _, pts, s, grade = res
    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_axes([0.08, 0.36, 0.84, 0.60])
    img, half = backdrop
    if img is not None:
        ax.imshow(img, extent=(-half, half, -half, half), origin="upper", alpha=0.8)

    segs, cols = [], []
    for r in roads:
        P = np.asarray(r["points"])
        d = np.diff(P, axis=0)
        gr = np.abs(d[:, 2]) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        segs += [P[i:i + 2, :2] for i in range(len(P) - 1)]
        cols += list(100 * gr)
    lc = LineCollection(segs, array=np.asarray(cols), cmap="viridis",
                        linewidths=2.0, clim=(0, 15))
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("road grade |%|")

    ax.plot(pts[:, 0], pts[:, 1], "-", color="#e8412c", lw=3.5, label="route")
    ax.plot(*start_xy, "o", ms=9, mfc="white", mec="k", label="rover start")
    ax.plot(pts[-1, 0], pts[-1, 1], "s", ms=9, mfc="#e8412c", mec="k", label="stop")
    for t in turbines:
        hit = t["id"] == st["turbine"]
        ax.plot(t["x"], t["y"], "^", ms=12 if hit else 9,
                mfc="gold" if hit else "white", mec="k")
        ax.annotate(t["id"].replace("turbine_", "T"), (t["x"], t["y"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9,
                    weight="bold" if hit else "normal")
    xs = np.r_[pts[:, 0], [t["x"] for t in turbines]]
    ys = np.r_[pts[:, 1], [t["y"] for t in turbines]]
    m = 120
    ax.set_xlim(xs.min() - m, xs.max() + m)
    ax.set_ylim(ys.min() - m, ys.max() + m)
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Road route to {st['turbine']}{title_extra}: "
                 f"{st['length_3d_m']:.0f} m, climb {st['climb_m']:.0f} m, "
                 f"{st['energy_wh_no_regen']:.1f} Wh", fontsize=10)

    ax2 = fig.add_axes([0.08, 0.06, 0.84, 0.24])
    ax2.plot(s, pts[:, 2], color="k", lw=1.8, label="road surface z")
    ax2.set_xlabel("distance along route (m)")
    ax2.set_ylabel("world z (m)")
    ax2.set_xlim(0, s[-1])
    ax3 = ax2.twinx()
    ax3.fill_between(s, 100 * grade, step="post", color="#e8412c", alpha=0.25)
    ax3.set_ylabel("grade (%)", color="#e8412c")
    ax3.axhline(0, color="#e8412c", lw=0.5)
    ax2.set_title(f"Elevation and grade (max {st['grade_max_pct']}%, "
                  f"stop {st['stop_to_base_2d_m']} m from base, base "
                  f"{st['base_minus_stop_z_m']:+.1f} m above stop)", fontsize=9)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roads", default=os.path.join(REPO, "config/roads.json"))
    ap.add_argument("--turbines", default=os.path.join(REPO, "config/turbines.yaml"))
    ap.add_argument("--rover-yaml", default=os.path.join(REPO, "config/rover.yaml"))
    ap.add_argument("--terrain-model", default="attappadi",
                    help="models/<name> for the plot backdrop")
    ap.add_argument("--turbine", default="all", help="turbine id or 'all'")
    ap.add_argument("--start", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("X", "Y"), help="rover start, world m (spawn = 0 0)")
    ap.add_argument("--start-z", type=float, default=None,
                    help="start z; default = nearest road node z")
    ap.add_argument("--cost", choices=["distance", "energy"], default="distance")
    ap.add_argument("--max-grade", type=float, default=20.0,
                    help="hard per-segment grade cap, percent")
    ap.add_argument("--stop-radius", type=float, default=0.0,
                    help=">0: choose cheapest node within R m of the base")
    ap.add_argument("--snap-m", type=float, default=5.0)
    ap.add_argument("--min-grade-run", type=float, default=3.0,
                    help="edges shorter than this (m) are exempt from the grade cap")
    ap.add_argument("--payload", type=float, default=2.0, help="drone mass, kg")
    ap.add_argument("--mass", type=float, default=None,
                    help="rover mass kg (default: rover.yaml)")
    ap.add_argument("--crr", type=float, default=0.03,
                    help="rolling resistance coefficient (gravel ~0.02-0.05)")
    ap.add_argument("--eta", type=float, default=0.75, help="drivetrain efficiency")
    ap.add_argument("--regen", type=float, default=0.0,
                    help="regen fraction for the reported with-regen energy")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "config/routes"))
    ap.add_argument("--fig-dir", default=os.path.join(REPO, "docs/routes"))
    a = ap.parse_args()
    if a.mass is None:
        a.mass = read_rover_mass(a.rover_yaml)
    p = Params(a)

    with open(a.roads) as f:
        roads = json.load(f)["roads"]
    turbines = read_turbines_yaml(a.turbines)
    g, comp = build_graph(roads, snap_m=a.snap_m)

    ids = [t["id"] for t in turbines] if a.turbine == "all" else [a.turbine]
    backdrop = load_backdrop(os.path.join(REPO, "models", a.terrain_model)) \
        if a.plot else (None, None)
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for tid in ids:
        turb = next((t for t in turbines if t["id"] == tid), None)
        if turb is None:
            raise SystemExit(f"unknown turbine {tid}")
        res = plan_to_turbine(g, comp, turb, a.start, a, p)
        if res is None:
            print(f"[plan] {tid}: NO ROUTE (grade cap {a.max_grade}%?)")
            continue
        st, wps, *_ = res
        out = {
            "generated_by": "tools/road_planner.py",
            "frame": "world ENU (Gazebo world frame), metres",
            "params": {"cost": a.cost, "max_grade_pct": a.max_grade,
                       "stop_radius_m": a.stop_radius, "mass_kg": p.mass,
                       "crr": a.crr, "eta": a.eta, "regen": a.regen,
                       "start": a.start},
            "summary": st,
            "waypoints": wps,
        }
        path = os.path.join(a.out_dir, f"route_{tid}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        if a.plot:
            plot_route(roads, turbines, res, a.start,
                       os.path.join(a.fig_dir, f"route_{tid}.png"), backdrop,
                       f" ({a.cost})")
        rows.append(st)

    if rows:
        hdr = (f"{'turbine':<11}{'len m':>8}{'climb m':>9}{'desc m':>8}"
               f"{'max %':>7}{'Wh':>7}{'stop->base m':>14}{'base-stop dz':>14}"
               f"{'hub-stop dz':>13}")
        print("\n" + hdr + "\n" + "-" * len(hdr))
        for r in rows:
            print(f"{r['turbine']:<11}{r['length_3d_m']:>8.0f}{r['climb_m']:>9.1f}"
                  f"{r['descent_m']:>8.1f}{r['grade_max_pct']:>7.1f}"
                  f"{r['energy_wh_no_regen']:>7.1f}{r['stop_to_base_2d_m']:>14.1f}"
                  f"{r['base_minus_stop_z_m']:>+14.1f}{r['hub_minus_stop_z_m']:>+13.1f}")
        print(f"\nroutes -> {os.path.relpath(a.out_dir, REPO)}/"
              + (f", figures -> {os.path.relpath(a.fig_dir, REPO)}/" if a.plot else ""))


if __name__ == "__main__":
    main()
