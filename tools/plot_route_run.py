#!/usr/bin/env python3
# =============================================================================
# File:        tools/plot_route_run.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Updated:     2026-09-24  turn rate from unwrapped yaw (ground-truth twist
#              spikes at the +/-180 deg wrap); roll/pitch stats skip the
#              first 3 s and add 95th percentiles
# Depends:     python3, numpy, matplotlib;
#              src/rover2drone_nav/rover2drone_nav/route_io.py
# =============================================================================
"""
plot_route_run.py
=================

Figure and summary for one route_follower run (logs/route_<t>_<stamp>.csv).

Layout: the travelled path over the planned route (map), with the control
signals directly below it, sharing the time axis:
  speed (commanded vs measured), turn rate (commanded vs measured),
  cross-track error, and elevation with road grade.

Prints tracking statistics (mean / RMS / 95th pct / max |cross-track
error|, time, distance, mean speed) and writes them to a .json beside
the figure.

Usage:
  /usr/bin/python3 tools/plot_route_run.py                 # newest log
  /usr/bin/python3 tools/plot_route_run.py logs/route_turbine_07_20260924_150000.csv
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
            cols[k] = np.array([float(r[k]) for r in rows])
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "docs", "runs"))
    a = ap.parse_args()
    path = a.log or max(glob.glob(os.path.join(REPO, "logs", "route_*.csv")),
                        key=os.path.getmtime, default=None)
    if not path:
        raise SystemExit("no logs/route_*.csv found")
    name = os.path.splitext(os.path.basename(path))[0]
    turbine = "_".join(name.split("_")[1:3])
    d = read_log(path)
    route = load_route(route_path(REPO, turbine))

    # Measured turn rate from the logged yaw (unwrapped). The ground-truth
    # twist's angular z spikes to about +/-150 rad/s whenever yaw wraps
    # through +/-180 deg, so it is not used.
    yaw_u = np.unwrap(d["yaw"])
    dt = np.gradient(d["t"])
    w_raw = np.gradient(yaw_u) / np.maximum(dt, 1e-3)
    k = 5
    d["w_meas"] = np.convolve(w_raw, np.ones(k) / k, mode="same")
    settle = d["t"] > d["t"][0] + 3.0      # skip the teleport drop / start
    trk = d["state"] == "track"
    cte = np.abs(d["cte"][trk]) if trk.any() else np.abs(d["cte"])
    dist = float(np.sum(np.hypot(np.diff(d["x"]), np.diff(d["y"]))))
    stats = {
        "log": os.path.relpath(path, REPO), "turbine": turbine,
        "arrived": bool((d["state"] == "arrived").any()),
        "sim_time_s": round(float(d["t"][-1]), 1),
        "distance_m": round(dist, 1),
        "s_start_m": round(float(d["s"][0]), 1), "s_end_m": round(float(d["s"][-1]), 1),
        "mean_speed_mps": round(dist / max(float(d["t"][-1]), 1e-6), 3),
        "cte_mean_m": round(float(cte.mean()), 3),
        "cte_rms_m": round(float(np.sqrt((cte ** 2).mean())), 3),
        "cte_p95_m": round(float(np.percentile(cte, 95)), 3),
        "cte_max_m": round(float(cte.max()), 3),
        "pitch_p95_deg": round(float(np.degrees(np.percentile(np.abs(d["pitch"][settle]), 95))), 1),
        "pitch_max_deg": round(float(np.degrees(np.abs(d["pitch"][settle]).max())), 1),
        "roll_p95_deg": round(float(np.degrees(np.percentile(np.abs(d["roll"][settle]), 95))), 1),
        "roll_max_deg": round(float(np.degrees(np.abs(d["roll"][settle]).max())), 1),
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[3.2, 1, 1, 1, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax.plot(route.p[:, 0], route.p[:, 1], "-", color="0.6", lw=6, alpha=0.5,
            label="planned route")
    sc = ax.scatter(d["x"], d["y"], c=d["v_meas"], s=4, cmap="viridis",
                    vmin=0, vmax=max(0.6, float(d["v_meas"].max())), label="travelled")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("speed (m/s)")
    ax.plot(d["x"][0], d["y"][0], "o", mfc="white", mec="k", ms=8, label="start")
    ax.plot(route.p[-1, 0], route.p[-1, 1], "s", mfc="#e8412c", mec="k", ms=8, label="goal")
    pad = 15
    ax.set_xlim(min(d["x"].min(), route.p[-1, 0]) - pad, max(d["x"].max(), route.p[-1, 0]) + pad)
    ax.set_ylim(min(d["y"].min(), route.p[-1, 1]) - pad, max(d["y"].max(), route.p[-1, 1]) + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{turbine}: {dist:.0f} m in {d['t'][-1] / 60:.1f} min sim time, "
                 f"|CTE| mean {stats['cte_mean_m']:.2f} m, p95 {stats['cte_p95_m']:.2f} m"
                 + ("" if stats["arrived"] else "  (not arrived)"), fontsize=10)

    t = d["t"] / 60.0
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, d["v_cmd"], lw=1.2, label="commanded")
    ax1.plot(t, d["v_meas"], lw=1.0, alpha=0.8, label="measured")
    ax1.set_ylabel("speed (m/s)")
    ax1.legend(fontsize=8, loc="lower right")
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.plot(t, d["w_cmd"], lw=1.2, label="commanded")
    ax2.plot(t, d["w_meas"], lw=1.0, alpha=0.8, label="measured")
    ax2.set_ylabel("turn rate (rad/s)")
    ax2.set_ylim(-1.0, 1.0)
    ax2.legend(fontsize=8, loc="lower right")
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax3.plot(t, d["cte"], lw=1.0, color="#e8412c")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_ylabel("cross-track (m)")
    ax4 = fig.add_subplot(gs[4], sharex=ax1)
    ax4.plot(t, d["z"], color="k", lw=1.2)
    ax4.set_ylabel("elevation z (m)")
    ax4.set_xlabel("sim time (min)")
    ax5 = ax4.twinx()
    ax5.fill_between(t, 100 * d["grade"], color="#e8412c", alpha=0.2, step="post")
    ax5.set_ylabel("road grade (%)", color="#e8412c")
    for a_ in (ax1, ax2, ax3):
        plt.setp(a_.get_xticklabels(), visible=False)

    os.makedirs(a.out_dir, exist_ok=True)
    png = os.path.join(a.out_dir, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    with open(os.path.join(a.out_dir, name + ".json"), "w") as f:
        json.dump(stats, f, indent=1)
    for k, v in stats.items():
        print(f"  {k:<16} {v}")
    print(f"figure -> {os.path.relpath(png, REPO)}")


if __name__ == "__main__":
    main()
