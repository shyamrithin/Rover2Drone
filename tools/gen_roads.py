#!/usr/bin/env python3
# =============================================================================
# File:        tools/gen_roads.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Updated:     2026-09-24  junction height matching (roads met with steps of
#              up to 2.2 m because each profile was smoothed separately)
# Depends:     numpy, Pillow, tools/terrain_io.py
# =============================================================================
"""
gen_roads.py
============

Carves the access roads into the terrain and records them as a navigation
graph for the rover.

Why this is needed
------------------
The 30 m elevation model has no road cuts: a 4 m track across a hillside is
invisible in it, so without carving the rover drives on raw slope. This
tool takes road centrelines from an OpenStreetMap GeoJSON export and, for
each road:

  1. Samples the terrain along the centreline and smooths the profile
     longitudinally (Gaussian, default 30 m), giving a plausible road grade.
  2. Sets every heightmap vertex within the road half-width to that profile,
     level across the width, and blends a shoulder back into the natural
     terrain. Applied identically to the visual and collision heightmaps.
  2b. Before carving, matches road heights at junctions and crossings so
     roads meet without a step (harmonise_junctions).
  3. Tints the road strip in the satellite texture with a dirt colour.
  4. Writes config/roads.json: each road's centreline in world metres with
     road-surface z, for the rover's waypoint follower.

Idempotent: the first run saves pristine copies (*_raw.png) of the
heightmaps and texture, and every run carves from those, so re-running
with different parameters does not compound. make_terrain_model.py
overwrites the raw copies when the terrain is rebuilt.

OSM tracks are traced by volunteers from imagery and can sit a few metres
off the road visible in the texture. Check alignment in Gazebo.

Usage
  python3 tools/gen_roads.py --terrain-model attappadi \
      --geojson ~/Downloads/attappadi_osm.geojson
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
from PIL import Image

from terrain_io import Terrain

WIDTH = {"track": 4.0, "service": 4.0, "unclassified": 5.0, "residential": 5.0,
         "tertiary": 6.0, "secondary": 7.0, "primary": 7.5, "road": 5.0}
DIRT = np.array([158.0, 128.0, 92.0])


def resample(pts, step):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def smooth(z, step, sigma_m):
    s = max(1, int(sigma_m / step))
    k = np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2)
    k /= k.sum()
    zp = np.pad(z, 3 * s, mode="edge")
    return np.convolve(zp, k, mode="valid")


def _priority(r):
    return (r["width"], len(r["xy"]))

def harmonise_junctions(roads, step, snap_m=6.0, blend_m=30.0, max_added_grade=0.04,
                        log=print):
    """
    Make roads agree in height where they meet.

    Each road's profile is smoothed on its own, so two roads meeting at a
    junction can disagree by metres; carving then leaves a step along the
    line where the nearest road switches. Roads are processed from the
    highest class (width) and longest down. Each road is matched to every
    already-processed road it touches (its endpoints near the other road,
    the other road's endpoints near it, or an interior crossing): the height
    difference at the contact is spread over a smoothstep ramp of at least
    blend_m, lengthened so the added grade stays near max_added_grade.
    Roads of the same class split the difference. Modifies roads in place.
    """
    order = sorted(range(len(roads)), key=lambda i: _priority(roads[i]), reverse=True)
    XY = [np.asarray(r["xy"], dtype=float) for r in roads]
    done, fixes = [], []
    for i in order:
        P = XY[i]
        n = len(P)
        s = np.arange(n) * step
        contacts = []                       # (index on i, target z)
        for j in done:
            Q = XY[j]
            # endpoints of i near any vertex of j (T-junctions, end-to-end)
            for k in (0, n - 1):
                d = np.hypot(Q[:, 0] - P[k, 0], Q[:, 1] - P[k, 1])
                m = int(d.argmin())
                if d[m] <= snap_m:
                    contacts.append((k, j, m))
            # endpoints of j near the interior of i (j ends on i)
            for m in (0, len(Q) - 1):
                d = np.hypot(P[:, 0] - Q[m, 0], P[:, 1] - Q[m, 1])
                k = int(d.argmin())
                if d[k] <= snap_m and 0 < k < n - 1:
                    contacts.append((k, j, m))
            # interior crossings: closest vertex pair well inside both roads
            D = np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
            k, m = np.unravel_index(D.argmin(), D.shape)
            if (D[k, m] <= step * 0.75 and 2 < k < n - 3 and 2 < m < len(Q) - 3):
                contacts.append((int(k), j, int(m)))
        # dedupe contacts at the same place
        uniq = {}
        for k, j, m in contacts:
            uniq.setdefault(k, (j, m))
        z = np.asarray(roads[i]["z"], dtype=float).copy()
        z0 = z.copy()
        def ramp(zarr, sarr, k, dz):
            L = max(blend_m, 1.5 * abs(dz) / max_added_grade)
            u = np.clip(np.abs(sarr - sarr[k]) / L, 0, 1)
            zarr += dz * (1 - u * u * (3 - 2 * u))
        for k, (j, m) in uniq.items():
            zj = roads[j]["z"]
            dz = float(zj[m]) - z0[k]
            if abs(dz) < 1e-3:
                continue
            # Same road class: meet in the middle, each road takes half the
            # difference. Lower class: the minor road takes all of it.
            share = 0.5 if roads[j]["width"] == roads[i]["width"] else 1.0
            ramp(z, s, k, dz * share)
            if share < 1.0:
                sj = np.arange(len(zj)) * step
                zj2 = np.asarray(zj, dtype=float).copy()
                ramp(zj2, sj, m, -dz * (1 - share))
                roads[j]["z"] = zj2
            fixes.append(abs(dz))
        roads[i]["z"] = z
        done.append(i)
    if fixes:
        log(f"  junctions: {len(fixes)} height mismatches fixed, max {max(fixes):.2f} m, "
            f"mean {np.mean(fixes):.2f} m")
    return roads


def carve(hm, meta, roads, shoulder):
    """Carve roads into one normalised heightmap array (in place)."""
    n = hm.shape[0]
    ext, half = meta["extent_m"], meta["extent_m"] / 2.0
    res = ext / (n - 1)
    zr, off = meta["zrange_m"], meta["z_offset_m"]
    best = np.full(hm.shape, np.inf)
    target = np.zeros(hm.shape)
    for rd in roads:
        P, Z, hw = rd["xy"], rd["z"], rd["width"] / 2.0
        reach = hw + shoulder
        for i in range(len(P) - 1):
            ax, ay = P[i]
            bx, by = P[i + 1]
            c0 = max(int((min(ax, bx) - reach + half) / res), 0)
            c1 = min(int((max(ax, bx) + reach + half) / res) + 1, n - 1)
            r0 = max(int((half - max(ay, by) - reach) / res), 0)
            r1 = min(int((half - min(ay, by) + reach) / res) + 1, n - 1)
            if c0 > c1 or r0 > r1:
                continue
            cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
            px, py = cc * res - half, half - rr * res
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy or 1e-9
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
            d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
            zt = Z[i] + t * (Z[i + 1] - Z[i])
            sub = d < best[r0:r1 + 1, c0:c1 + 1]
            sel = sub & (d < reach)
            best[r0:r1 + 1, c0:c1 + 1][sel] = d[sel]
            target[r0:r1 + 1, c0:c1 + 1][sel] = zt[sel]
    return np.isfinite(best), best, target


def apply(hm, meta, hit, best, target, hwgrid, shoulder):
    zr, off = meta["zrange_m"], meta["z_offset_m"]
    zt = (target - off) / zr                       # target in normalised units
    w = np.zeros(hm.shape)
    core = hit & (best <= hwgrid)
    edge = hit & ~core
    w[core] = 1.0
    u = np.clip((best[edge] - hwgrid[edge]) / shoulder, 0, 1)
    w[edge] = 1 - u * u * (3 - 2 * u)              # smoothstep falloff
    before = hm.copy()
    hm[:] = np.clip(hm * (1 - w) + zt * w, 0, 1)
    return (hm - before) * zr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain-model", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--smooth-m", type=float, default=30.0)
    ap.add_argument("--shoulder-m", type=float, default=4.0)
    ap.add_argument("--tint", type=float, default=0.45, help="Texture tint 0..1")
    ap.add_argument("--snap-m", type=float, default=6.0,
                    help="Junction detection distance between roads")
    ap.add_argument("--blend-m", type=float, default=30.0,
                    help="Minimum ramp length for junction height matching")
    ap.add_argument("--no-junction-blend", action="store_true",
                    help="Disable junction height matching (old behaviour)")
    ap.add_argument("--types", default=",".join(WIDTH),
                    help="Comma-separated OSM highway types to use")
    a = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(a.repo))
    ter = Terrain(os.path.join(repo, "models", a.terrain_model))
    m = ter.meta
    types = set(a.types.split(","))
    step = 2.0
    # Sample road profiles from the pristine terrain, never from a previously
    # carved one, so re-running gives identical results.
    raw_v = os.path.join(ter.dir, m["visual_heightmap"]).replace(".png", "_raw.png")
    if os.path.exists(raw_v):
        ter._hm["visual"] = np.array(Image.open(raw_v)).astype(np.float64) / 65535.0

    roads, skipped = [], {}
    for ft in json.load(open(os.path.expanduser(a.geojson))).get("features", []):
        g, pr = ft.get("geometry") or {}, ft.get("properties") or {}
        hw = pr.get("highway")
        if not hw:
            continue
        if hw not in types:
            skipped[hw] = skipped.get(hw, 0) + 1
            continue
        lines = [g["coordinates"]] if g.get("type") == "LineString" else \
            g.get("coordinates", []) if g.get("type") == "MultiLineString" else []
        for line in lines:
            xy = [ter.latlon_to_world(la, lo) for lo, la in line]
            xy = [p for p in resample(xy, step)
                  if ter.ground_z(*p) is not None]
            if len(xy) < 3:
                continue
            z = smooth(np.array([ter.ground_z(*p) for p in xy]), step, a.smooth_m)
            roads.append({"id": pr.get("@id", pr.get("id", f"road_{len(roads)}")),
                          "type": hw, "width": WIDTH.get(hw, 4.0),
                          "xy": xy, "z": z})

    if roads and not a.no_junction_blend:
        harmonise_junctions(roads, step, snap_m=a.snap_m, blend_m=a.blend_m)

    if not roads:
        raise SystemExit("No usable roads in the GeoJSON inside the terrain. "
                         f"Types seen but skipped: {skipped}")

    tex_dir = os.path.join(ter.dir, "materials", "textures")
    stats = {}
    for which in ("visual", "collision"):
        cur = os.path.join(ter.dir, m[f"{which}_heightmap"])
        raw = cur.replace(".png", "_raw.png")
        if not os.path.exists(raw):
            shutil.copy(cur, raw)
        hm = np.array(Image.open(raw)).astype(np.float64) / 65535.0
        hit, best, target = carve(hm, m, roads, a.shoulder_m)
        # Per-vertex half-width: nearest road's width. One pass per road type.
        hwgrid = np.full(hm.shape, 2.0)
        for w in sorted({r["width"] for r in roads}):
            sub = [r for r in roads if r["width"] == w]
            h2, b2, _ = carve(hm, m, sub, a.shoulder_m)
            closer = h2 & (b2 <= best + 1e-9)
            hwgrid[closer] = w / 2.0
        dz = apply(hm, m, hit, best, target, hwgrid, a.shoulder_m)
        out = (np.clip(hm, 0, 1) * 65535.0).astype("<u2")
        Image.frombytes("I;16", (hm.shape[1], hm.shape[0]), out.tobytes()).save(cur)
        moved = np.abs(dz) > 0.01
        stats[which] = (float(dz.min()), float(dz.max()), int(moved.sum()),
                        moved.sum() * (m["extent_m"] / (hm.shape[0] - 1)) ** 2)
        print(f"  carved {which}: cut {-dz.min():.2f} m, fill {dz.max():.2f} m, "
              f"{moved.sum()} vertices changed")

    # Texture tint along the roads.
    tcur = os.path.join(ter.dir, m["texture"])
    traw = tcur.replace(".png", "_raw.png")
    if not os.path.exists(traw):
        shutil.copy(tcur, traw)
    img = np.asarray(Image.open(traw).convert("RGB"), dtype=np.float64)
    ts = img.shape[0]
    fake = {"extent_m": m["extent_m"], "zrange_m": 1.0, "z_offset_m": 0.0}
    tgrid = np.zeros((ts + 1, ts + 1))
    hit, best, _ = carve(tgrid, fake, roads, 1.0)
    best, hit = best[:ts, :ts], hit[:ts, :ts]
    alpha = np.zeros((ts, ts))
    core = hit & (best <= 2.0)
    alpha[core] = a.tint
    edge = hit & ~core
    alpha[edge] = a.tint * np.clip(1 - (best[edge] - 2.0), 0, 1)
    img = img * (1 - alpha[..., None]) + DIRT * alpha[..., None]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(tcur)

    total = sum(len(r["xy"]) * step for r in roads)
    grades = []
    for r in roads:
        dz = np.abs(np.diff(r["z"])) / step * 100
        grades.append(float(np.percentile(dz, 95)))
    out = {"generated_by": "tools/gen_roads.py", "terrain": a.terrain_model,
           "source": os.path.abspath(os.path.expanduser(a.geojson)),
           "smooth_m": a.smooth_m, "point_spacing_m": step * 2,
           "carve": {k: {"max_cut_m": -v[0], "max_fill_m": v[1],
                         "vertices_changed": v[2], "area_m2": v[3]}
                     for k, v in stats.items()},
           "roads": [{"id": r["id"], "type": r["type"], "width_m": r["width"],
                      "length_m": round(len(r["xy"]) * step, 1),
                      "grade_p95_pct": round(g, 1),
                      "points": [[round(x, 2), round(y, 2), round(float(z), 3)]
                                 for (x, y), z in zip(r["xy"][::2], r["z"][::2])]}
                     for r, g in zip(roads, grades)]}
    with open(os.path.join(repo, "config", "roads.json"), "w") as f:
        json.dump(out, f, indent=1)
    by = {}
    for r in roads:
        by[r["type"]] = by.get(r["type"], 0) + len(r["xy"]) * step
    print(f"  {len(roads)} roads, {total / 1000:.1f} km: " +
          ", ".join(f"{k} {v / 1000:.1f} km" for k, v in sorted(by.items())))
    if skipped:
        print(f"  skipped types: {skipped}")
    print(f"  steepest road (95th pct grade): {max(grades):.0f}%  -> config/roads.json")


if __name__ == "__main__":
    main()
