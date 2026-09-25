# =============================================================================
# File:        src/rover2drone_nav/rover2drone_nav/ekf.py
# Project:     Rover2Drone - marsupial UGV-UAV wind turbine inspection
# Author:      Shyam (with Claude)
# Created:     2026-09-25
# Updated:     2026-09-25  gate-lockout fix (GNSS sigma floor, re-acquisition),
#              slip-sized process noise, odometry scale state
# Depends:     numpy (no ROS: unit-testable offline)
# =============================================================================
"""
ekf.py
======

Planar extended Kalman filter for the rover, kept small and explicit so it
can be reported (and tested) exactly.

State      s = [x, y, yaw, b_g, k]  world ENU position, heading, gyro z bias,
                                    wheel-odometry scale factor
Predict    (wheel speed v from the drive odometry, gyro rate w from the IMU)
             x   += k v cos(yaw) dt
             y   += k v sin(yaw) dt
             yaw += (w - b_g) dt
             b_g, k : random walks (k absorbs wheel radius error / steady slip)
           Process noise: speed sigma_v0 + k_slip*|v| (wheel slip grows with
           speed and grade), gyro white noise, bias random walk.
Updates    GNSS position [x, y] with R = sigma_h^2 I as reported by the
           receiver; compass heading with its reported sigma.
           Both gated by the Mahalanobis distance (chi-square, 99.9 %) so a
           multipath jump or a distorted compass reading near a tower is
           rejected, not trusted.
Init       first GNSS fix for position, first compass reading for heading.

Robustness (found in the first Gazebo batch, 2026-09-25): with RTK the
reported sigma (~1.5 cm) made the filter so confident that a transient
(slip over a crest, GNSS latency, odometry scale error) moved the estimate
outside the gate, after which EVERY fix was rejected and the filter
dead-reckoned off the road ("gate lockout"). Now:
  * GNSS sigma is floored at gnss_min_sigma (covers latency and model error)
  * after gate_reset_n consecutive rejected fixes the position covariance
    is inflated by the residual and the fix is accepted (re-acquisition)
  * process noise is sized for skid-steer slip, and the odometry scale is
    estimated
The EKF treats GNSS errors as white although real ones are correlated
(bias); gnss_r_scale inflates R to stay consistent.
"""

import math
from dataclasses import dataclass

import numpy as np

CHI2_2DOF_999 = 13.82
CHI2_1DOF_999 = 10.83


@dataclass
class EkfParams:
    sigma_v0: float = 0.03         # m/s
    k_slip: float = 0.10           # fraction of |v| (skid-steer slip)
    sigma_w: float = 0.01          # rad/s gyro white noise
    sigma_bg_rw: float = 2e-4      # rad/s/sqrt(s) bias random walk
    gnss_r_scale: float = 1.5      # inflate reported GNSS sigma
    gnss_min_sigma: float = 0.10   # m, floor on the GNSS sigma used
    gate_reset_n: float = 10       # consecutive gated fixes -> re-acquire
    sigma_k_rw: float = 1e-4       # odometry scale random walk /sqrt(s)
    init_k_sigma: float = 0.05
    compass_r_scale: float = 1.0
    init_bias_sigma: float = 0.01
    gate_gnss: float = CHI2_2DOF_999
    gate_compass: float = CHI2_1DOF_999


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PlanarEkf:
    def __init__(self, params=None):
        self.p = params or EkfParams()
        self.s = None
        self.P = None
        self.pending_xy = None
        self.pending_sig = None
        self.pending_yaw = None
        self.n_gnss = self.n_gnss_rej = self.n_compass = self.n_compass_rej = 0
        self.n_reacquire = 0
        self.gnss_rej_run = 0

    @property
    def ready(self):
        return self.s is not None

    def _try_init(self):
        if self.pending_xy is None or self.pending_yaw is None:
            return
        x, y = self.pending_xy
        yaw, sy = self.pending_yaw
        self.s = np.array([x, y, yaw, 0.0, 1.0])
        sg = self.pending_sig
        self.P = np.diag([sg ** 2, sg ** 2, sy ** 2, self.p.init_bias_sigma ** 2,
                          self.p.init_k_sigma ** 2])

    def predict(self, v, w, dt):
        if not self.ready or dt <= 0.0:
            return
        p = self.p
        x, y, th, bg, k = self.s
        c, s = math.cos(th), math.sin(th)
        kv = k * v
        self.s = np.array([x + kv * c * dt, y + kv * s * dt,
                           wrap(th + (w - bg) * dt), bg, k])
        F = np.array([[1, 0, -kv * s * dt, 0, v * c * dt],
                      [0, 1, kv * c * dt, 0, v * s * dt],
                      [0, 0, 1, -dt, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]], dtype=float)
        G = np.array([[c * dt, 0], [s * dt, 0], [0, dt], [0, 0], [0, 0]], dtype=float)
        sv = p.sigma_v0 + p.k_slip * abs(v)
        Qu = np.diag([sv ** 2, p.sigma_w ** 2])
        Q = G @ Qu @ G.T
        Q[3, 3] += p.sigma_bg_rw ** 2 * dt
        Q[4, 4] += p.sigma_k_rw ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, x, y, sigma_h):
        sig = max(sigma_h * self.p.gnss_r_scale, self.p.gnss_min_sigma)
        if not self.ready:
            self.pending_xy, self.pending_sig = (x, y), sig
            self._try_init()
            return True
        H = np.zeros((2, 5))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * sig ** 2
        r = np.array([x, y]) - self.s[:2]
        S = H @ self.P @ H.T + R
        d2 = float(r @ np.linalg.solve(S, r))
        self.n_gnss += 1
        if d2 > self.p.gate_gnss:
            self.n_gnss_rej += 1
            self.gnss_rej_run += 1
            if self.gnss_rej_run < self.p.gate_reset_n:
                return False
            # Gate lockout: the receiver has disagreed consistently, so the
            # estimate is what is wrong. Inflate the position covariance by
            # the residual and accept this fix.
            self.P[0, 0] += r[0] ** 2 + sig ** 2
            self.P[1, 1] += r[1] ** 2 + sig ** 2
            self.n_reacquire += 1
            S = H @ self.P @ H.T + R
        self.gnss_rej_run = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ r
        self.s[2] = wrap(self.s[2])
        self.P = (np.eye(5) - K @ H) @ self.P
        return True

    def update_compass(self, yaw, sigma):
        sig = sigma * self.p.compass_r_scale
        if not self.ready:
            self.pending_yaw = (yaw, sig)
            self._try_init()
            return True
        r = wrap(yaw - self.s[2])
        S = float(self.P[2, 2]) + sig ** 2
        self.n_compass += 1
        if r * r / S > self.p.gate_compass:
            self.n_compass_rej += 1
            return False
        K = self.P[:, 2] / S
        self.s = self.s + K * r
        self.s[2] = wrap(self.s[2])
        self.P = self.P - np.outer(K, self.P[2, :])
        return True

    def pose(self):
        return float(self.s[0]), float(self.s[1]), float(self.s[2])

    def sigma_xy(self):
        return float(math.sqrt(max(self.P[0, 0], 0.0))), float(math.sqrt(max(self.P[1, 1], 0.0)))
