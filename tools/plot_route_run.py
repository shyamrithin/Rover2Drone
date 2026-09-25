#!/usr/bin/env python3
# =============================================================================
# File:        tools/plot_route_run.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Updated:     2026-09-24  turn rate from unwrapped yaw (ground-truth twist
#              spikes at the +/-180 deg wrap); roll/pitch stats skip the
#              first 3 s and add 95th percentiles
#              2026-09-25  rover-drone distance panel + stats when logged
#              2026-09-25  step 4: estimate-vs-truth mode (true CTE,
#              localisation error panel); analyse()/plot() reusable by
#              tools/run_batch.py
# Depends:     python3, numpy, matplotlib;
#              src/rover2drone_nav/rover2drone_nav/route_io.py
# =============================================================================
"""
plot_route_run.py
=================

Figure and summary for one route_follower run (logs/route_<t>[_tag]_<stamp>.csv).

Layout: the travelled path over the planned route (map), with the control
signals directly below it, sharing the time axis:
  speed (commanded vs measured), turn rate (commanded vs measured),
  cross-track error, [localisation error], [rover-drone distance],
  and elevation with road grade.

Runs driven on an ESTIMATE (EKF) log ground truth beside it. Then the map
shows the true path (and the estimated one), the cross-track panel shows
the TRUE error (ground truth vs route) with the error the controller saw,
and a localisation-error panel (estimate vs truth) is added. All stats
headed cte_* are the true ones in that case; cte_seen_* are the
controller's view.

Prints the statistics and writes them to a .json beside the figure.

Usage:
  /usr/bin/python3 tools/plot_route_run.py                 # newest log
  /usr/bin/python3 tools/plot_route_run.py logs/route_turbine_07_m8n_s1_20260925_150000.csv
Output: docs/runs/<log name>.png and .json
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src", "rover2drone_nav"))
from rover2drone_nav.route_io import load_route, route_path  # noqa: E402


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = {}
    for k in rows[0]:
        if k == "state":
            cols[k] = np.array([r[k] for r in rows])
        else:
            cols[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan
                                for r in rows])
    return cols


def pct(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def analyse(path):
    name = os.path.splitext(os.path.basename(path))[0]
    turbine = "_".join(name.split("_")[1:3])
    d = read_log(path)
    route = load_route(route_path(REPO, turbine))

    # Turn rate from the logged yaw, unwrapped (the ground-truth twist's
    # angular z spikes when yaw wraps through +/-180 deg).
    yaw_u = np.unwrap(d["yaw"])
    dt = np.gradient(d["t"])
    d["w_meas"] = np.convolve(np.gradient(yaw_u) / np.maximum(dt, 1e-3),
                              np.ones(5) / 5, mode="same")

    est_mode = "loc_err" in d and np.isfinite(d["loc_err"]).any() \
        and np.nanmax(d["loc_err"]) > 1e-3
    if est_mode:
        tx, ty, tz = d["gt_x"], d["gt_y"], d["gt_z"]
        true_cte = d["true_cte"]
    else:
        tx, ty, tz = d["x"], d["y"], d["z"]
        true_cte = d["cte"]
    d["tx"], d["ty"], d["tz"] = tx, ty, tz

    settle = d["t"] > d["t"][0] + 3.0
    trk = d["state"] == "track"
    sel = trk if trk.any() else np.ones_like(trk)
    cte = np.abs(true_cte[sel])
    ok = np.isfinite(tx)
    dist = float(np.sum(np.hypot(np.diff(tx[ok]), np.diff(ty[ok]))))
    gx, gy = route.p[-1, 0], route.p[-1, 1]
    stats = {
        "log": os.path.relpath(path, REPO), "turbine": turbine,
        "mode": "estimate" if est_mode else "ground_truth",
        "arrived": bool((d["state"] == "arrived").any()),
        "sim_time_s": round(float(d["t"][-1]), 1),
        "distance_m": round(dist, 1),
        "s_start_m": round(float(d["s"][0]), 1), "s_end_m": round(float(d["s"][-1]), 1),
        "mean_speed_mps": round(dist / max(float(d["t"][-1]), 1e-6), 3),
        "cte_mean_m": round(float(np.nanmean(cte)), 3),
        "cte_rms_m": round(float(np.sqrt(np.nanmean(cte ** 2))), 3),
        "cte_p95_m": round(pct(cte, 95), 3),
        "cte_max_m": round(float(np.nanmax(cte)), 3),
        "goal_error_m": round(float(np.hypot(tx[ok][-1] - gx, ty[ok][-1] - gy)), 2),
        "pitch_p95_deg": round(float(np.degrees(pct(np.abs(d["pitch"][settle]), 95))), 1),
        "pitch_max_deg": round(float(np.degrees(np.nanmax(np.abs(d["pitch"][settle])))), 1),
        "roll_p95_deg": round(float(np.degrees(pct(np.abs(d["roll"][settle]), 95))), 1),
        "roll_max_deg": round(float(np.degrees(np.nanmax(np.abs(d["roll"][settle])))), 1),
    }
    if est_mode:
        seen = np.abs(d["cte"][sel])
        le = d["loc_err"][sel]
        stats.update({
            "cte_seen_mean_m": round(float(np.nanmean(seen)), 3),
            "cte_seen_p95_m": round(pct(seen, 95), 3),
            "loc_err_mean_m": round(float(np.nanmean(le)), 3),
            "loc_err_p95_m": round(pct(le, 95), 3),
            "loc_err_max_m": round(float(np.nanmax(le)), 3),
        })
    sl = d.get("slant")
    if sl is not None and np.isfinite(sl).any():
        v = sl[np.isfinite(sl)]
        base = float(np.median(v))
        stats.update({
            "slant_median_m": round(base, 3),
            "slant_min_m": round(float(v.min()), 3),
            "slant_max_m": round(float(v.max()), 3),
            "slant_max_dev_m": round(float(np.abs(v - base).max()), 3),
            "slant_coverage": round(float(np.isfinite(sl).mean()), 3),
        })
    return d, stats, route, name


def plot(d, stats, route, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = stats["mode"] == "estimate"
    has_slant = "slant_median_m" in stats
    panels = ["speed", "turn", "cte"] + (["loc"] if est else []) + \
        (["slant"] if has_slant else []) + ["elev"]
    nrow = 1 + len(panels)
    fig = plt.figure(figsize=(9, 3.9 + 1.9 * len(panels)))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[3.2] + [1] * len(panels), hspace=0.35)
    ax = fig.add_subplot(gs[0])
    tx, ty = d["tx"], d["ty"]
    ax.plot(route.p[:, 0], route.p[:, 1], "-", color="0.6", lw=6, alpha=0.5,
            label="planned route")
    if est:
        ax.plot(d["x"], d["y"], "-", color="#e8412c", lw=0.8, alpha=0.8,
                label="estimated path (EKF)")
    sc = ax.scatter(tx, ty, c=d["v_meas"], s=4, cmap="viridis", vmin=0,
                    vmax=max(0.6, float(np.nanmax(d["v_meas"]))),
                    label="true path" if est else "travelled")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("speed (m/s)")
    ok = np.isfinite(tx)
    ax.plot(tx[ok][0], ty[ok][0], "o", mfc="white", mec="k", ms=8, label="start")
    ax.plot(route.p[-1, 0], route.p[-1, 1], "s", mfc="#e8412c", mec="k", ms=8, label="goal")
    pad = 15
    ax.set_xlim(min(np.nanmin(tx), route.p[-1, 0]) - pad, max(np.nanmax(tx), route.p[-1, 0]) + pad)
    ax.set_ylim(min(np.nanmin(ty), route.p[-1, 1]) - pad, max(np.nanmax(ty), route.p[-1, 1]) + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.legend(loc="best", fontsize=8)
    title = (f"{stats['turbine']}: {stats['distance_m']:.0f} m in {d['t'][-1] / 60:.1f} min, "
             f"{'TRUE ' if est else ''}|CTE| mean {stats['cte_mean_m']:.2f} m, "
             f"p95 {stats['cte_p95_m']:.2f} m")
    if est:
        title += f"\nlocalisation error mean {stats['loc_err_mean_m']:.2f} m, " \
                 f"p95 {stats['loc_err_p95_m']:.2f} m; goal error {stats['goal_error_m']:.1f} m"
    if not stats["arrived"]:
        title += "  (NOT ARRIVED)"
    ax.set_title(title, fontsize=10)

    t = d["t"] / 60.0
    axes = []
    for i, kind in enumerate(panels):
        a = fig.add_subplot(gs[i + 1], sharex=axes[0] if axes else None)
        axes.append(a)
        if kind == "speed":
            a.plot(t, d["v_cmd"], lw=1.2, label="commanded")
            a.plot(t, d["v_meas"], lw=1.0, alpha=0.8, label="measured")
            a.set_ylabel("speed (m/s)")
            a.legend(fontsize=8, loc="lower right")
        elif kind == "turn":
            a.plot(t, d["w_cmd"], lw=1.2, label="commanded")
            a.plot(t, d["w_meas"], lw=1.0, alpha=0.8, label="measured")
            a.set_ylabel("turn rate (rad/s)")
            a.set_ylim(-1.0, 1.0)
            a.legend(fontsize=8, loc="lower right")
        elif kind == "cte":
            if est:
                a.plot(t, d["cte"], lw=0.8, color="0.5", label="seen by controller")
                a.plot(t, d["true_cte"], lw=1.0, color="#e8412c", label="true")
                a.legend(fontsize=8, loc="lower right")
            else:
                a.plot(t, d["cte"], lw=1.0, color="#e8412c")
            a.axhline(0, color="k", lw=0.5)
            a.set_ylabel("cross-track (m)")
        elif kind == "loc":
            a.plot(t, d["loc_err"], lw=1.0, color="#1f78b4")
            a.set_ylabel("localisation\nerror (m)")
        elif kind == "slant":
            a.plot(t, d["slant"], lw=1.0, color="#6a3d9a")
            a.axhline(stats["slant_median_m"], color="k", lw=0.5, ls="--")
            a.set_ylabel("rover-drone\ndistance (m)")
        elif kind == "elev":
            a.plot(t, d["tz"], color="k", lw=1.2)
            a.set_ylabel("elevation z (m)")
            a.set_xlabel("sim time (min)")
            a2 = a.twinx()
            a2.fill_between(t, 100 * d["grade"], color="#e8412c", alpha=0.2, step="post")
            a2.set_ylabel("road grade (%)", color="#e8412c")
        if kind != "elev":
            plt.setp(a.get_xticklabels(), visible=False)

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(stats, f, indent=1)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "docs", "runs"))
    a = ap.parse_args()
    path = a.log or max(glob.glob(os.path.join(REPO, "logs", "route_*.csv")),
                        key=os.path.getmtime, default=None)
    if not path:
        raise SystemExit("no logs/route_*.csv found")
    d, stats, route, name = analyse(path)
    png = plot(d, stats, route, name, a.out_dir)
    for k, v in stats.items():
        print(f"  {k:<16} {v}")
    print(f"figure -> {os.path.relpath(png, REPO)}")


if __name__ == "__main__":
    main()
