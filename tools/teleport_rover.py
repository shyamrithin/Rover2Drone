#!/usr/bin/env python3
# =============================================================================
# File:        tools/teleport_rover.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     python3, numpy, gz CLI (Gazebo Harmonic running);
#              src/rover2drone_nav/rover2drone_nav/route_io.py
# =============================================================================
"""
teleport_rover.py
=================

Moves the rover to a point on a planned route, facing along it, so route
tests don't have to drive the whole way (the sim runs well below real
time). Uses Gazebo's /world/<world>/set_pose service.

  --s S           arc length along the route (m)
  --remaining R   or: R metres before the goal
  (neither)       the route start

The rover is placed 0.10 m above its resting height and drops onto the
road. Start the route follower AFTER teleporting (it locks onto the
nearest route point on its first pose).

Do NOT teleport with the drone latched: set_pose moves only the rover and
the latch joint would yank the drone. Test without PX4, or release the
latch first.

Usage:
  /usr/bin/python3 tools/teleport_rover.py --turbine turbine_07 --remaining 150
  /usr/bin/python3 tools/teleport_rover.py --turbine turbine_03 --s 800
"""

import argparse
import math
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src", "rover2drone_nav"))
from rover2drone_nav.route_io import load_route, read_rover_yaml, route_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turbine", default="turbine_07")
    ap.add_argument("--route-file", default="")
    ap.add_argument("--s", type=float, default=None)
    ap.add_argument("--remaining", type=float, default=None)
    ap.add_argument("--world", default="attappadi_windfarm")
    ap.add_argument("--model", default="rover")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    r = load_route(a.route_file or route_path(REPO, a.turbine))
    s = a.s if a.s is not None else (r.length - a.remaining if a.remaining is not None else 0.0)
    s = min(max(s, 0.0), r.length - 1.0)
    x, y, z, yaw = r.point_at(s)
    base_h = read_rover_yaml(os.path.join(REPO, "config", "rover.yaml")).get("base_height_m", 0.20)
    z += base_h + 0.10
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    req = (f'name: "{a.model}" position: {{x: {x:.3f} y: {y:.3f} z: {z:.3f}}} '
           f'orientation: {{x: 0 y: 0 z: {qz:.6f} w: {qw:.6f}}}')
    cmd = ["gz", "service", "-s", f"/world/{a.world}/set_pose",
           "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
           "--timeout", "3000", "--req", req]
    print(f"{r.name}: s = {s:.0f} of {r.length:.0f} m ({r.length - s:.0f} m to go) -> "
          f"({x:.1f}, {y:.1f}, {z:.2f}), yaw {math.degrees(yaw):.0f} deg")
    if a.dry_run:
        print(" ".join(cmd[:-1]), f"'{req}'")
        return
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())


if __name__ == "__main__":
    main()
