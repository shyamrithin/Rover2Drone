# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/sensor_models.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Depends:     numpy (no ROS: shared by sensor_sim and the offline tests)
# =============================================================================
"""
sensor_models.py
================

Error models that turn the rover's ground-truth pose into the imperfect
measurements a real rover would get. Gazebo's own sensors only add white
noise; real GNSS error is dominated by slowly wandering bias, outages and
multipath, which is what makes localisation hard.

GNSS (per horizontal axis, and vertical)
  error = bias + white
  bias  : first-order Gauss-Markov, stationary sigma bias_sigma, correlation
          time bias_tau  (b <- b*exp(-dt/tau) + sigma*sqrt(1-exp(-2dt/tau))*n)
  white : zero-mean Gaussian, white_sigma
  near a turbine tower (within tower_radius of any base) both terms are
  scaled by tower_factor: multipath and partial sky blockage off the steel
  tower. Outages start as a Poisson process (mean interval outage_every_s)
  and last a uniform outage_min_s..outage_max_s; no fix is published then.
  The reported horizontal sigma is what a receiver would claim
  (sqrt(bias^2 + white^2) of the model's current sigmas), not the true error.

Compass (absolute heading, e.g. BNO085 with magnetometer)
  heading = true + constant per-run bias (bias_sigma draw) + white noise,
  plus a distortion of up to tower_bias_deg that grows linearly as the
  rover comes within tower_radius of a steel tower (it does not know).

Profiles
  rtk        RTK fix, cm-level; compass as m8n
  m8n        u-blox M8N-class single-frequency receiver, open sky (default)
  degraded   poor sky view / worse receiver, frequent outages
All parameters can be overridden individually.
"""

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class GnssParams:
    rate_hz: float = 5.0
    white_sigma: float = 0.4
    bias_sigma: float = 1.2
    bias_tau: float = 180.0
    vert_factor: float = 1.8
    tower_radius: float = 50.0
    tower_factor: float = 2.0
    outage_every_s: float = 600.0
    outage_min_s: float = 3.0
    outage_max_s: float = 10.0


@dataclass
class CompassParams:
    rate_hz: float = 10.0
    bias_sigma_deg: float = 3.0
    white_sigma_deg: float = 1.5
    tower_radius: float = 25.0
    tower_bias_deg: float = 8.0


PROFILES = {
    "rtk": (GnssParams(white_sigma=0.015, bias_sigma=0.01, bias_tau=60.0,
                       tower_factor=1.5, outage_every_s=1e9), CompassParams()),
    "m8n": (GnssParams(), CompassParams()),
    "degraded": (GnssParams(white_sigma=0.8, bias_sigma=2.5, bias_tau=120.0,
                            tower_factor=2.5, outage_every_s=180.0,
                            outage_min_s=5.0, outage_max_s=30.0),
                 CompassParams(bias_sigma_deg=5.0, white_sigma_deg=3.0)),
}


def get_profile(name, **overrides):
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name}; choose {sorted(PROFILES)}")
    g, c = PROFILES[name]
    g = replace(g, **{k: v for k, v in overrides.items() if hasattr(g, k)})
    c = replace(c, **{k[8:]: v for k, v in overrides.items()
                      if k.startswith("compass_") and hasattr(c, k[8:])})
    return g, c


class GnssModel:
    def __init__(self, params, towers_xy=(), seed=0):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.towers = np.asarray(towers_xy, dtype=float).reshape(-1, 2)
        self.bias = self.rng.normal(0.0, params.bias_sigma, 3) * \
            np.array([1.0, 1.0, params.vert_factor])
        self.outage_left = 0.0
        self.t_next_outage = self.rng.exponential(params.outage_every_s)
        self.t = 0.0

    def _near_tower(self, x, y):
        if not len(self.towers):
            return 1.0
        d = np.hypot(self.towers[:, 0] - x, self.towers[:, 1] - y).min()
        return self.p.tower_factor if d < self.p.tower_radius else 1.0

    def step(self, dt, x, y, z):
        """
        Advance the error state by dt and return a measurement
        (x_m, y_m, z_m, sigma_h, sigma_v), or None during an outage.
        """
        p = self.p
        self.t += dt
        a = math.exp(-dt / p.bias_tau)
        q = math.sqrt(max(1.0 - a * a, 0.0))
        scale = np.array([1.0, 1.0, p.vert_factor]) * p.bias_sigma
        self.bias = a * self.bias + q * scale * self.rng.normal(size=3)

        if self.outage_left > 0.0:
            self.outage_left -= dt
            return None
        if self.t >= self.t_next_outage:
            self.outage_left = self.rng.uniform(p.outage_min_s, p.outage_max_s)
            self.t_next_outage = self.t + self.outage_left + \
                self.rng.exponential(p.outage_every_s)
            return None

        f = self._near_tower(x, y)
        white = self.rng.normal(0.0, p.white_sigma, 3) * \
            np.array([1.0, 1.0, p.vert_factor])
        err = f * (self.bias + white)
        sig_h = f * math.hypot(p.bias_sigma, p.white_sigma)
        return (x + err[0], y + err[1], z + err[2], sig_h, sig_h * p.vert_factor)


class CompassModel:
    def __init__(self, params, towers_xy=(), seed=0):
        self.p = params
        self.rng = np.random.default_rng(seed + 7919)
        self.towers = np.asarray(towers_xy, dtype=float).reshape(-1, 2)
        self.bias = math.radians(self.rng.normal(0.0, params.bias_sigma_deg))
        self.tower_sign = self.rng.choice([-1.0, 1.0])

    def step(self, x, y, yaw):
        p = self.p
        dist = 0.0
        if len(self.towers):
            d = np.hypot(self.towers[:, 0] - x, self.towers[:, 1] - y).min()
            if d < p.tower_radius:
                dist = self.tower_sign * math.radians(p.tower_bias_deg) * \
                    (1.0 - d / p.tower_radius)
        h = yaw + self.bias + dist + math.radians(
            self.rng.normal(0.0, p.white_sigma_deg))
        h = math.atan2(math.sin(h), math.cos(h))
        # Claimed accuracy: bias + noise (the distortion is unmodelled).
        sig = math.radians(math.hypot(p.bias_sigma_deg, p.white_sigma_deg))
        return h, sig
