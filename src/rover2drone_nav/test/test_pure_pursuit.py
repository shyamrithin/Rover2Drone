# =============================================================================
# File:        src/rover2drone_nav/test/test_pure_pursuit.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-24
# Depends:     pytest, numpy; config/routes/*.json in the repo
# =============================================================================
"""
Offline check of the pure-pursuit controller on every planned route with a
kinematic unicycle (first-order actuator lag, rover speed/turn limits),
starting at the spawn with the spawn heading. Asserts the rover reaches the
goal with bounded cross-track error. Run: pytest -q src/rover2drone_nav/test
"""
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from rover2drone_nav.pure_pursuit import PurePursuit  # noqa: E402
from rover2drone_nav.route_io import load_route  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SPAWN_YAW = -0.9857


def simulate(route, dt=0.05, tau=0.3, t_max=8000.0):
    pp = PurePursuit(route)
    x, y, yaw = route.p[0, 0], route.p[0, 1], SPAWN_YAW
    v = w = 0.0
    ctes, t = [], 0.0
    while t < t_max:
        c = pp.step(x, y, yaw, dt)
        if c.state == "arrived":
            return True, t, np.abs(ctes), (x, y)
        v += (c.v - v) * dt / tau
        w += (c.omega - w) * dt / tau
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
        if c.state == "track" and c.s > 20.0:
            ctes.append(c.cte)
        t += dt
    return False, t, np.abs(ctes), (x, y)


def test_all_routes():
    files = sorted(glob.glob(os.path.join(REPO, "config", "routes", "route_*.json")))
    assert files
    for f in files:
        r = load_route(f)
        ok, t, cte, end = simulate(r)
        gx, gy = r.p[-1, :2]
        print(f"{r.name}: arrived={ok} in {t / 60:.1f} min, |cte| mean "
              f"{cte.mean():.2f} p95 {np.percentile(cte, 95):.2f} max {cte.max():.2f} m, "
              f"end err {math.hypot(end[0] - gx, end[1] - gy):.2f} m")
        assert ok
        assert np.percentile(cte, 95) < 0.5
        assert math.hypot(end[0] - gx, end[1] - gy) < 1.5


if __name__ == "__main__":
    test_all_routes()
