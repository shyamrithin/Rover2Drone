#!/usr/bin/env python3
# =============================================================================
# File:        tools/run_batch.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Depends:     ROS 2 Humble (sourced install/), Gazebo running demo.launch.py
#              WITHOUT PX4; tools/teleport_rover.py, tools/plot_route_run.py,
#              rover2drone_nav (route_follower, sensor_sim, rover_ekf)
# =============================================================================
"""
run_batch.py
============

Runs a set of route-following trials back to back and summarises them.
Each trial: teleport the rover to the start of the segment, start the
localisation stack for that configuration, drive with route_follower until
ARRIVED or timeout, stop everything, analyse the log.

Configurations
  gt        follower drives on ground truth (baseline; run once)
  rtk, m8n, degraded
            sensor_sim (that GNSS/compass profile, per seed) + rover_ekf;
            follower drives on /rover/ekf/odom

Outputs (docs/runs/batch_<stamp>/):
  <run>.png / .json    per-run figure and stats (plot_route_run layout)
  summary.csv, summary.md   one row per run
  comparison.png       true CTE p95, localisation error p95 and goal error
                       per configuration (each seed a dot), plus the true
                       paths of all runs over the route

Start Gazebo first (no PX4 needed; the drone is not used and teleporting
with a latched drone would yank it):
  ros2 launch rover2drone_bringup demo.launch.py

Usage:
  /usr/bin/python3 tools/run_batch.py                       # gt + rtk + m8n, seeds 1-3, T07 last 400 m
  /usr/bin/python3 tools/run_batch.py --configs gt m8n degraded --seeds 1 2 --remaining 300
  /usr/bin/python3 tools/run_batch.py --summarise docs/runs/batch_20260925_150000
"""

import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src", "rover2drone_nav"))
import plot_route_run as prr  # noqa: E402
from rover2drone_nav.route_io import load_route, route_path  # noqa: E402

COLOURS = {"gt": "#333333", "rtk": "#1b9e77", "m8n": "#d95f02", "degraded": "#7570b3"}
SIM = ["--ros-args", "-p", "use_sim_time:=true"]


def start(cmd, log):
    f = open(log, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def stop(p, wait=5.0):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGINT)
        p.wait(timeout=wait)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def run_one(a, config, seed, out_dir):
    tag = f"{config}_s{seed}" if config != "gt" else "gt"
    print(f"\n=== {tag}: {a.turbine}, last {a.remaining:.0f} m ===", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "tools", "teleport_rover.py"),
                    "--turbine", a.turbine, "--remaining", str(a.remaining)], check=False)
    time.sleep(a.settle)
    aux = []
    pose_topic = "/rover/ground_truth"
    if config != "gt":
        aux.append(start(["ros2", "run", "rover2drone_nav", "sensor_sim"] + SIM +
                         ["-p", f"profile:={config}", "-p", f"seed:={seed}"],
                         os.path.join(out_dir, f"{tag}_sensor_sim.txt")))
        aux.append(start(["ros2", "run", "rover2drone_nav", "rover_ekf"] + SIM,
                         os.path.join(out_dir, f"{tag}_ekf.txt")))
        pose_topic = "/rover/ekf/odom"
        time.sleep(a.ekf_warmup)
    t0 = time.time()
    fol = start(["ros2", "run", "rover2drone_nav", "route_follower"] + SIM +
                ["-p", f"turbine:={a.turbine}", "-p", f"pose_topic:={pose_topic}",
                 "-p", f"run_tag:={tag}", "-p", "exit_on_arrival:=true",
                 "-p", f"timeout_s:={a.sim_timeout}", "-p", "pose_timeout:=1.0"],
                os.path.join(out_dir, f"{tag}_follower.txt"))
    try:
        fol.wait(timeout=a.wall_timeout_min * 60.0)
    except subprocess.TimeoutExpired:
        print(f"  wall-clock timeout after {a.wall_timeout_min} min", flush=True)
    stop(fol)
    for p in aux:
        stop(p)
    logs = [p for p in glob.glob(os.path.join(REPO, "logs", f"route_{a.turbine}_{tag}_*.csv"))
            if os.path.getmtime(p) >= t0 - 5]
    if not logs:
        print("  no log written (did the follower start? see "
              f"{tag}_follower.txt)", flush=True)
        return None
    path = max(logs, key=os.path.getmtime)
    d, st, route, name = prr.analyse(path)
    prr.plot(d, st, route, name, out_dir)
    st.update({"config": config, "seed": seed})
    print(f"  arrived={st['arrived']}  true |CTE| p95 {st['cte_p95_m']:.2f} m"
          + (f"  loc err p95 {st['loc_err_p95_m']:.2f} m" if "loc_err_p95_m" in st else "")
          + f"  goal err {st['goal_error_m']:.2f} m  ({time.time() - t0:.0f} s wall)", flush=True)
    return st


def summarise(out_dir, turbine=None):
    rows = []
    for j in sorted(glob.glob(os.path.join(out_dir, "route_*.json"))):
        st = json.load(open(j))
        if "config" not in st:
            base = os.path.basename(j)
            parts = base[len("route_"):].split("_")
            st["config"] = parts[2] if parts[2] in COLOURS else "gt"
            st["seed"] = int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else 0
        rows.append(st)
    if not rows:
        raise SystemExit(f"no run json in {out_dir}")
    keys = ["config", "seed", "arrived", "sim_time_s", "distance_m", "cte_mean_m", "cte_p95_m",
            "cte_max_m", "loc_err_mean_m", "loc_err_p95_m", "loc_err_max_m", "goal_error_m",
            "cte_seen_p95_m", "log"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    order = [c for c in COLOURS if any(r["config"] == c for r in rows)]
    lines = ["| config | runs | arrived | true CTE p95 (m) | loc err p95 (m) | goal error (m) |",
             "|---|---|---|---|---|---|"]
    for c in order:
        rr = [r for r in rows if r["config"] == c]

        def agg(k):
            v = [r[k] for r in rr if k in r]
            return f"{np.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})" if v else "-"
        lines.append(f"| {c} | {len(rr)} | {sum(r['arrived'] for r in rr)}/{len(rr)} | "
                     f"{agg('cte_p95_m')} | {agg('loc_err_p95_m')} | {agg('goal_error_m')} |")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n\nmean (min-max) over seeds\n")
    print("\n" + md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4], hspace=0.35, wspace=0.35)
    for i, (k, lab) in enumerate([("cte_p95_m", "true |CTE| p95 (m)"),
                                  ("loc_err_p95_m", "localisation error p95 (m)"),
                                  ("goal_error_m", "stop error at goal (m)")]):
        a = fig.add_subplot(gs[0, i])
        for j, c in enumerate(order):
            v = [r[k] for r in rows if r["config"] == c and k in r]
            if v:
                a.scatter([j] * len(v), v, color=COLOURS[c], s=30, zorder=3)
                a.hlines(np.mean(v), j - 0.3, j + 0.3, color=COLOURS[c], lw=2)
        a.set_xticks(range(len(order)))
        a.set_xticklabels(order)
        a.set_ylabel(lab)
        a.grid(axis="y", alpha=0.3)
    am = fig.add_subplot(gs[1, :])
    turbine = turbine or rows[0]["turbine"]
    route = load_route(route_path(REPO, turbine))
    s0 = min(r.get("s_start_m", 0) for r in rows)
    keep = route.s >= s0 - 10
    am.plot(route.p[keep, 0], route.p[keep, 1], "-", color="0.6", lw=8, alpha=0.4,
            label="planned route")
    seen = set()
    for r in rows:
        d = prr.read_log(os.path.join(REPO, r["log"]))
        tx = d.get("gt_x", d["x"]) if "gt_x" in d and np.isfinite(d["gt_x"]).any() else d["x"]
        ty = d.get("gt_y", d["y"]) if "gt_y" in d and np.isfinite(d["gt_y"]).any() else d["y"]
        c = r["config"]
        am.plot(tx, ty, "-", color=COLOURS[c], lw=1.0, alpha=0.8,
                label=c if c not in seen else None)
        seen.add(c)
    am.plot(route.p[-1, 0], route.p[-1, 1], "s", mfc="#e8412c", mec="k", ms=9, label="goal")
    am.set_aspect("equal", adjustable="datalim")
    am.set_xlabel("x east (m)")
    am.set_ylabel("y north (m)")
    am.set_title("true paths of all runs")
    am.legend(fontsize=8)
    png = os.path.join(out_dir, "comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\nsummary -> {os.path.relpath(out_dir, REPO)}/summary.md, comparison.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turbine", default="turbine_07")
    ap.add_argument("--remaining", type=float, default=400.0)
    ap.add_argument("--configs", nargs="+", default=["gt", "rtk", "m8n"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--sim-timeout", type=float, default=2400.0, help="sim s per run")
    ap.add_argument("--wall-timeout-min", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=4.0, help="wall s after teleport")
    ap.add_argument("--ekf-warmup", type=float, default=4.0, help="wall s before driving")
    ap.add_argument("--summarise", default=None, help="only summarise this batch dir")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise, a.turbine)
        return
    out_dir = os.path.join(REPO, "docs", "runs", "batch_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    plan = [(c, s) for c in a.configs for s in (a.seeds if c != "gt" else [0])]
    print(f"{len(plan)} runs -> {os.path.relpath(out_dir, REPO)}: "
          + ", ".join(f"{c}{'' if c == 'gt' else f'/s{s}'}" for c, s in plan))
    with open(os.path.join(out_dir, "batch.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    try:
        for c, s in plan:
            run_one(a, c, s, out_dir)
    except KeyboardInterrupt:
        print("\ninterrupted; summarising completed runs")
    summarise(out_dir, a.turbine)


if __name__ == "__main__":
    main()
