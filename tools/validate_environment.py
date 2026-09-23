#!/usr/bin/env python3
"""
validate_environment.py
=======================

Offline validation of the Rover2Drone simulation environment against its
real-world sources. Produces docs/validation/report.md plus figures, with
every number computed from the files Gazebo actually loads.

Checks
  V1  Georeferencing. World coordinates (site-centred transverse Mercator)
      are compared with Gazebo's own conversion (ECEF tangent plane, used by
      the NavSat sensor that feeds PX4's GPS) across the whole terrain. The
      old UTM grid is shown for contrast.
  V2  Heightmap fidelity. The rendered visual and collision heightmaps are
      reconstructed from their PNGs and compared with the source DEM
      resampled at the same points, and with each other.
  V3  Elevation cross-check. The Copernicus DEM against an independent DEM
      (AWS Terrain Tiles, SRTM-derived) over the terrain and at each turbine.
      Two independent DEMs agreeing is evidence; neither is ground truth.
  V4  Turbine placement. Surveyed coordinates, world positions, ground and
      hub elevations, and round-trip error; overlay on the imagery.
  V5  Layout plausibility. Spacing in rotor diameters.
  V6  Terrain profiles. Elevation and grade from the rover spawn to each
      turbine, which is why the rover must follow roads.

Usage
  python3 tools/validate_environment.py --terrain-model attappadi \
      --ref-dem ~/Downloads/attappadi_dem.tif
"""

import argparse
import math
import os

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import Resampling
from rasterio.warp import transform as warp_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_terrain_model import extract_window, fill_voids  # noqa: E402
from terrain_io import (Terrain, gz_local_to_latlon, read_turbines,  # noqa: E402
                        read_yaml_flat)

COP = {"abs_v": 4.0, "rel_v_flat": 2.0, "rel_v_steep": 4.0, "abs_h": 6.0}


def carve_note(repo):
    """Summary of road carving, which deliberately departs from the DEM."""
    p = os.path.join(repo, "config", "roads.json")
    if not os.path.exists(p):
        return ""
    import json
    r = json.load(open(p))
    c = r["carve"]["visual"]
    km = sum(x["length_m"] for x in r["roads"]) / 1000
    return (f" These comparisons use the pre-carving heightmaps. Road carving then "
            f"deliberately modifies {c['area_m2'] / 1e4:.2f} ha along {km:.1f} km of "
            f"OSM roads (max cut {c['max_cut_m']:.1f} m, max fill "
            f"{c['max_fill_m']:.1f} m) to represent road cuts the 30 m DEM cannot resolve.")


def stats(d):
    d = d[np.isfinite(d)]
    return dict(bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                le90=float(np.percentile(np.abs(d), 90)), max=float(np.abs(d).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain-model", required=True)
    ap.add_argument("--ref-dem", help="Independent DEM GeoTIFF for V3")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="docs/validation")
    a = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(a.repo))
    out = os.path.join(repo, a.out)
    os.makedirs(out, exist_ok=True)
    ter = Terrain(os.path.join(repo, "models", a.terrain_model))
    m = ter.meta
    turb = read_turbines(os.path.join(repo, "config", "turbines.yaml"))
    tcfg = read_yaml_flat(os.path.join(repo, "config", "turbines.yaml"))
    rotor_d = float(tcfg.get("rotor_diameter_m", 52.0))
    dem_path = m["dem_source"]
    R = []

    # ---------------------------------------------------------------- V1
    g = np.linspace(-ter.half, ter.half, 13)
    tm_err, utm_err = [], []
    zone = int((ter.lon0 + 180) // 6) + 1
    utm = CRS.from_epsg(32600 + zone)
    ux0, uy0 = warp_transform(CRS.from_epsg(4326), utm, [ter.lon0], [ter.lat0])
    for x in g:
        for y in g:
            la_g, lo_g, _ = gz_local_to_latlon(x, y, 0.0, ter.lat0, ter.lon0, ter.elev0)
            la_t, lo_t = ter.world_to_latlon(x, y)
            ex = (lo_t - lo_g) * 111320 * math.cos(math.radians(ter.lat0))
            ey = (la_t - la_g) * 110574
            tm_err.append(math.hypot(ex, ey))
            uxs, uys = warp_transform(utm, CRS.from_epsg(4326), [ux0[0] + x], [uy0[0] + y])
            ex = (uxs[0] - lo_g) * 111320 * math.cos(math.radians(ter.lat0))
            ey = (uys[0] - la_g) * 110574
            utm_err.append(math.hypot(ex, ey))
    corner = math.hypot(ter.half, ter.half)
    R.append(("V1", "Georeferencing vs Gazebo's GPS model",
              f"Across the full {ter.extent:.0f} m terrain, world coordinates differ from "
              f"Gazebo's NavSat conversion by at most **{max(tm_err) * 100:.1f} cm** "
              f"(mean {np.mean(tm_err) * 100:.1f} cm). The earlier UTM grid would have "
              f"differed by up to {max(utm_err):.2f} m (grid convergence). Earth curvature "
              f"makes Gazebo report altitude {corner ** 2 / (2 * 6371000) * 100:.0f} cm "
              f"higher at the terrain corners than the flat heightmap; negligible.",
              max(tm_err) < 0.10))

    # ---------------------------------------------------------------- V2
    lines = []
    raw_hm = {}
    for which in ("visual", "collision"):
        rp = os.path.join(ter.dir, m[f"{which}_heightmap"]).replace(".png", "_raw.png")
        raw_hm[which] = (np.array(Image.open(rp)).astype(np.float64) / 65535.0
                         if os.path.exists(rp) else ter.heightmap(which))
    for which in ("visual", "collision"):
        hm = raw_hm[which]
        n = hm.shape[0]
        ref = fill_voids(extract_window(dem_path, ter.lat0, ter.lon0, ter.extent, n,
                                        ter.crs, Resampling.bilinear, 1)[0])
        rec = m["zmin_m"] + hm * m["zrange_m"]
        s = stats(rec - ref)
        lines.append(f"{which} ({n}x{n}): RMSE {s['rmse'] * 100:.2f} cm, max "
                     f"{s['max'] * 100:.2f} cm")
    hv, hc = raw_hm["visual"], raw_hm["collision"]
    step = (hv.shape[0] - 1) // (hc.shape[0] - 1)
    vc = stats((hv[::step, ::step] - hc) * m["zrange_m"])
    quant = m["zrange_m"] / 65535 * 100
    R.append(("V2", "Rendered heightmaps vs source DEM",
              "Reconstructed from the PNGs Gazebo loads: " + "; ".join(lines) +
              f". 16-bit quantisation step is {quant:.2f} cm. Visual vs collision "
              f"surfaces at shared samples: max {vc['max'] * 100:.2f} cm, so the rover "
              f"(physics) and what you see (rendering) agree." + carve_note(repo),
              vc["max"] < 0.05))

    # ---------------------------------------------------------------- V3
    if a.ref_dem:
        n = hc.shape[0]
        cop = fill_voids(extract_window(dem_path, ter.lat0, ter.lon0, ter.extent, n,
                                        ter.crs, Resampling.bilinear, 1)[0])
        refd = fill_voids(extract_window(os.path.expanduser(a.ref_dem), ter.lat0,
                                         ter.lon0, ter.extent, n, ter.crs,
                                         Resampling.bilinear, 1)[0])
        d = cop - refd
        s = stats(d)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(d, cmap="RdBu_r", vmin=-15, vmax=15,
                          extent=(-ter.half, ter.half, -ter.half, ter.half))
        ax[0].set_title("Copernicus minus reference DEM (m)")
        plt.colorbar(im, ax=ax[0])
        ax[0].plot([t["x"] for t in turb], [t["y"] for t in turb], "k^", ms=6)
        ax[1].hist(d.ravel(), bins=80, color="0.4")
        ax[1].set_xlabel("difference (m)")
        ax[1].set_title(f"bias {s['bias']:.2f} m, RMSE {s['rmse']:.2f} m, LE90 {s['le90']:.2f} m")
        plt.tight_layout(); plt.savefig(os.path.join(out, "v3_dem_crosscheck.png"), dpi=110)
        plt.close()
        rows = []
        for t in turb:
            c = (t["x"] + ter.half) / ter.extent * (n - 1)
            r = (ter.half - t["y"]) / ter.extent * (n - 1)
            rows.append(f"| {t['id']} | {cop[int(round(r)), int(round(c))]:.1f} | "
                        f"{refd[int(round(r)), int(round(c))]:.1f} | "
                        f"{d[int(round(r)), int(round(c))]:+.1f} |")
        R.append(("V3", "Elevation cross-check against an independent DEM",
                  f"Over the terrain: bias {s['bias']:+.2f} m, RMSE {s['rmse']:.2f} m, "
                  f"LE90 {s['le90']:.2f} m. For context, Copernicus GLO-30 is specified at "
                  f"< {COP['abs_v']:.0f} m absolute vertical (LE90), < {COP['rel_v_flat']:.0f} m "
                  f"relative on slopes up to 20%, < {COP['abs_h']:.0f} m horizontal (CE90); the "
                  f"reference is SRTM-derived and coarser, so disagreement bounds the combined "
                  f"error of both. Datums differ slightly (EGM2008 vs EGM96).\n\n"
                  f"| Turbine | Copernicus (m) | Reference (m) | Diff (m) |\n|---|---|---|---|\n"
                  + "\n".join(rows) + "\n\n![](v3_dem_crosscheck.png)", s["le90"] < 15))

    # ---------------------------------------------------------------- V4
    rows, rt = [], []
    for t in turb:
        x, y = ter.latlon_to_world(t["lat"], t["lon"])
        la, lo = ter.world_to_latlon(t["x"], t["y"])
        rt.append(math.hypot((lo - t["lon"]) * 111320 * math.cos(math.radians(ter.lat0)),
                             (la - t["lat"]) * 110574))
        gz = ter.ground_z(t["x"], t["y"])
        rows.append(f"| {t['id']} | {t['lat']:.6f} | {t['lon']:.6f} | {t['x']:.1f} | "
                    f"{t['y']:.1f} | {gz + ter.elev0:.1f} | {t['hub_z'] + ter.elev0:.1f} | "
                    f"{abs(gz - t['ground_z']) * 100:.1f} |")
    tex = np.asarray(Image.open(os.path.join(ter.dir, m["texture"])).convert("RGB"))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(tex, extent=(-ter.half, ter.half, -ter.half, ter.half))
    ax.plot([t["x"] for t in turb], [t["y"] for t in turb], "o", mfc="none",
            mec="red", ms=14, mew=2)
    for t in turb:
        ax.annotate(t["id"].replace("turbine_", "T"), (t["x"] + 18, t["y"] + 18),
                    color="yellow", fontsize=11, weight="bold")
    ax.plot(0, 0, "c*", ms=16)
    ax.annotate("rover spawn", (15, 15), color="cyan", fontsize=10)
    xs = [t["x"] for t in turb] + [0]; ys = [t["y"] for t in turb] + [0]
    ax.set_xlim(min(xs) - 150, max(xs) + 150); ax.set_ylim(min(ys) - 150, max(ys) + 150)
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.set_title("Placed turbines (red) over the simulation's satellite texture")
    plt.tight_layout(); plt.savefig(os.path.join(out, "v4_turbine_overlay.png"), dpi=110)
    plt.close()
    R.append(("V4", "Turbine placement",
              f"Positions from satellite-read coordinates; round-trip WGS84 -> world -> WGS84 "
              f"error at most {max(rt) * 1000:.2f} mm. Tower base vs rendered surface in the "
              f"last column. Each red ring should sit on a turbine foundation pad in the "
              f"imagery, an independent visual check of the georeferencing.\n\n"
              "| ID | Lat | Lon | x (m) | y (m) | Ground AMSL (m) | Hub AMSL (m) | Base gap (cm) |\n"
              "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) +
              "\n\n![](v4_turbine_overlay.png)", max(rt) < 0.01))

    # ---------------------------------------------------------------- V5
    sp = []
    for t in turb:
        dmin = min(math.hypot(t["x"] - u["x"], t["y"] - u["y"]) for u in turb if u is not t)
        sp.append(dmin / rotor_d)
    R.append(("V5", "Layout plausibility",
              f"Nearest-neighbour spacing {min(sp):.1f} to {max(sp):.1f} rotor diameters "
              f"({rotor_d:.0f} m rotor). Ridge-line rows are typically 2.5 to 5 D, consistent "
              f"with the inferred Suzlon S52 scale. The turbine model is inferred, not "
              f"confirmed from a published site list.", 2.0 <= min(sp) <= 6.0))

    # ---------------------------------------------------------------- V6
    fig, ax = plt.subplots(figsize=(10, 5))
    gtxt = []
    for t in turb:
        L = math.hypot(t["x"], t["y"])
        dd = np.linspace(0, L, 200)
        zz = np.array([ter.ground_z(t["x"] * s / L, t["y"] * s / L) for s in dd])
        gr = np.abs(np.diff(zz) / np.diff(dd)) * 100
        ax.plot(dd, zz + ter.elev0, label=t["id"].replace("turbine_", "T"))
        gtxt.append(f"| {t['id']} | {L:.0f} | {zz[-1]:+.1f} | {abs(zz[-1]) / L * 100:.1f} | "
                    f"{np.percentile(gr, 95):.0f} |")
    ax.set_xlabel("straight-line distance from rover spawn (m)")
    ax.set_ylabel("ground elevation AMSL (m)"); ax.legend(ncol=4)
    ax.set_title("Terrain profiles, spawn to each turbine")
    plt.tight_layout(); plt.savefig(os.path.join(out, "v6_profiles.png"), dpi=110)
    plt.close()
    R.append(("V6", "Terrain profiles",
              "Straight-line profiles are steeper than a wheeled rover should climb, which "
              "is why rover navigation must follow the access roads.\n\n"
              "| Turbine | Distance (m) | Rise (m) | Mean grade (%) | 95th pct grade (%) |\n"
              "|---|---|---|---|---|\n" + "\n".join(gtxt) + "\n\n![](v6_profiles.png)", True))

    # ------------------------------------------------------------- write
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(f"# Simulation environment validation: {a.terrain_model}\n\n"
                f"Generated by `tools/validate_environment.py`. Site centre "
                f"{ter.lat0:.6f} N, {ter.lon0:.6f} E, {ter.elev0:.1f} m (EGM2008). "
                f"Terrain {ter.extent:.0f} m square; visual grid {m['visual_grid']}, "
                f"collision grid {m['collision_grid']}, texture {m['texture_px']} px "
                f"({ter.extent / m['texture_px']:.2f} m/px).\n\n| Check | Result |\n|---|---|\n")
        for k, title, _, ok in R:
            f.write(f"| {k} {title} | {'PASS' if ok else 'REVIEW'} |\n")
        for k, title, body, ok in R:
            f.write(f"\n## {k}. {title}\n\n{body}\n")
        f.write("\n## Known limitations\n\n"
                "- Elevation source is 30 m; finer grids interpolate, they do not add detail. "
                "Road cuts narrower than ~30 m are not in the DEM.\n"
                "- Copernicus is a surface model: canopy and structures are included in the "
                "terrain height.\n"
                "- Vertical datum is EGM2008 (orthometric); a real GNSS receiver reports "
                "ellipsoidal height, which differs by the local geoid undulation.\n"
                "- Imagery (Esri World Imagery) has an unknown capture date.\n"
                "- Turbine model (Suzlon S52 class) is inferred from spacing and tower type.\n"
                "- Foliage is decorative, placed from an RGB vegetation index, visual only.\n")
    for k, title, _, ok in R:
        print(f"  {k} {'PASS  ' if ok else 'REVIEW'} {title}")
    print(f"  report: {out}/report.md")


if __name__ == "__main__":
    main()
