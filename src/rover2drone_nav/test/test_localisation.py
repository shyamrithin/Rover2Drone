# =============================================================================
# File:        src/rover2drone_nav/test/test_localisation.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Updated:     2026-09-25  gate-lockout regression (latency, odometry scale,
#              slip bursts)
# Depends:     pytest, numpy; config/routes, config/turbines.yaml
# =============================================================================
"""
Offline closed-loop check of step 4: kinematic rover (first-order actuator
lag, wheel slip on speed), gyro with bias, GNSS and compass from
sensor_models, the planar EKF, and pure pursuit driving on the EKF
estimate. Reports localisation error and the TRUE cross-track error
(ground truth vs route) per GNSS profile and seed.
Run: pytest -q -s src/rover2drone_nav/test/test_localisation.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from rover2drone_nav.ekf import PlanarEkf  # noqa: E402
from rover2drone_nav.pure_pursuit import PurePursuit  # noqa: E402
from rover2drone_nav.route_io import load_route, load_turbines, route_path  # noqa: E402
from rover2drone_nav.sensor_models import CompassModel, GnssModel, get_profile  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def run(profile, seed, turbine="turbine_07", remaining=400.0, dt=0.05,
        latency=0.0, odo_scale=1.0, slip_bursts=False):
    route = load_route(route_path(REPO, turbine))
    towers = [(t["x"], t["y"]) for t in load_turbines(os.path.join(REPO, "config", "turbines.yaml"))]
    gp, cp = get_profile(profile)
    gnss, comp = GnssModel(gp, towers, seed), CompassModel(cp, towers, seed)
    rng = np.random.default_rng(seed + 100)
    ekf, pp = PlanarEkf(), PurePursuit(route)
    s0 = route.length - remaining
    x, y, _, yaw = route.point_at(s0)
    v = w = 0.0
    gyro_bias = rng.normal(0.0, 0.003)
    t = tg = tc = 0.0
    loc_err, true_cte, hist = [], [], []
    while t < 3000.0:
        hist.append((t, x, y))
        # sensors (GNSS optionally reports where the rover was `latency` ago)
        if t >= tg:
            ox, oy = x, y
            if latency > 0:
                old = [h for h in hist[-int(latency / dt) - 3:] if h[0] <= t - latency]
                if old:
                    ox, oy = old[-1][1], old[-1][2]
            m = gnss.step(1.0 / gp.rate_hz, ox, oy, 0.0)
            if m is not None:
                ekf.update_gnss(m[0], m[1], m[3])
            tg += 1.0 / gp.rate_hz
        if t >= tc:
            h, sh = comp.step(x, y, yaw)
            ekf.update_compass(h, sh)
            tc += 1.0 / cp.rate_hz
        burst = 0.6 if (slip_bursts and 100.0 < t % 240.0 < 106.0) else 0.0
        v_wheel = v * odo_scale * (1.0 + burst + rng.normal(0.0, 0.03)) + rng.normal(0.0, 0.01)
        w_gyro = w + gyro_bias + rng.normal(0.0, 0.005)
        ekf.predict(v_wheel, w_gyro, dt)
        if not ekf.ready:
            t += dt
            continue
        ex, ey, eyaw = ekf.pose()
        c = pp.step(ex, ey, eyaw, dt)
        if c.state == "arrived":
            break
        _, _, cte, _ = route.project(x, y, max(pp.seg - 30, 0), pp.seg + 60)
        loc_err.append(math.hypot(ex - x, ey - y))
        true_cte.append(abs(cte))
        v += (c.v - v) * dt / 0.3
        w += (c.omega - w) * dt / 0.3
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = math.atan2(math.sin(yaw + w * dt), math.cos(yaw + w * dt))
        t += dt
    le, tc_ = np.array(loc_err), np.array(true_cte)
    return {"arrived": c.state == "arrived", "t": t, "loc_mean": le.mean(),
            "loc_p95": np.percentile(le, 95), "cte_mean": tc_.mean(),
            "cte_p95": np.percentile(tc_, 95), "cte_max": tc_.max(),
            "gnss_rej": ekf.n_gnss_rej / max(ekf.n_gnss, 1)}


def test_profiles():
    for prof in ("rtk", "m8n", "degraded"):
        for seed in (1, 2):
            r = run(prof, seed)
            print(f"{prof:9s} seed {seed}: arrived={r['arrived']} {r['t'] / 60:4.1f} min | "
                  f"loc err mean {r['loc_mean']:.2f} p95 {r['loc_p95']:.2f} m | "
                  f"TRUE |cte| mean {r['cte_mean']:.2f} p95 {r['cte_p95']:.2f} "
                  f"max {r['cte_max']:.2f} m | gnss rejected {100 * r['gnss_rej']:.1f}%")
            assert r["arrived"]
            if prof == "rtk":
                assert r["cte_p95"] < 0.3


def test_no_gate_lockout():
    """
    Regression for the first Gazebo batch (2026-09-25): with RTK the EKF
    rejected ~86 % of fixes after a slip transient and drifted off the road.
    GNSS latency 0.15 s, 4 % odometry scale error and 6 s wheel-slip bursts.
    """
    for seed in (1, 2, 3):
        r = run("rtk", seed, latency=0.15, odo_scale=1.04, slip_bursts=True)
        print(f"rtk stress seed {seed}: loc p95 {r['loc_p95']:.2f} m, TRUE |cte| p95 "
              f"{r['cte_p95']:.2f} m, gnss rejected {100 * r['gnss_rej']:.1f}%")
        assert r["arrived"]
        assert r["gnss_rej"] < 0.05
        assert r["cte_p95"] < 0.5


if __name__ == "__main__":
    test_profiles()
    test_no_gate_lockout()
