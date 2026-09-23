#!/usr/bin/env python3
"""
terrain_io.py
=============

Shared terrain access for the Rover2Drone tools. Reads the terrain.yaml that
make_terrain_model.py writes next to each terrain model, and answers the
questions every other tool needs, from the exact files Gazebo renders:

  latlon_to_world / world_to_latlon   WGS84 <-> world ENU metres, using the
                                      same site-centred transverse Mercator
                                      the terrain was built in
  ground_z(x, y, which)               world z of the rendered surface,
                                      bilinearly sampled from the visual or
                                      collision heightmap PNG
  gz_local_to_latlon                  Gazebo's own conversion (ECEF tangent
                                      plane), used by the validation tools to
                                      check our projection against Gazebo's

One file, one geometry: turbines, foliage and validation all sample the
same heightmap bytes Gazebo loads, so they cannot disagree with it.
"""

import math
import os

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def site_scale(lat, h):
    """
    Scale that makes projected metres equal true ground metres at height h.
    Gazebo's GPS model is a tangent plane at the site elevation, where a
    metre subtends less angle than at sea level by the factor R/(R+h), with
    R the Gaussian radius of curvature at this latitude.
    """
    sl = math.sin(math.radians(lat)) ** 2
    m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sl) ** 1.5
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sl)
    return 1.0 + h / math.sqrt(m * n)


def site_crs(lat, lon, k=1.0):
    """Transverse Mercator centred on the site: true north, scale k."""
    return CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat:.9f} +lon_0={lon:.9f} +k={k:.12f} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")


def read_yaml_flat(path):
    """Minimal parser for flat 'key: value' YAML (no dependency needed)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if ":" not in line or line.startswith(" "):
                continue
            k, v = (t.strip() for t in line.split(":", 1))
            v = v.strip('"').strip("'")
            try:
                out[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                out[k] = v
    return out


def read_turbines(path):
    """Turbine rows from config/turbines.yaml as dicts."""
    import re
    rows = []
    for line in open(path):
        if "{id:" not in line:
            continue
        d = {}
        for k, v in re.findall(r"(\w+):\s*([^,}]+)", line):
            v = v.strip()
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        rows.append(d)
    return rows


# --------------------------------------------------------- geodesy helpers
def geodetic_to_ecef(lat, lon, h):
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(la) ** 2)
    return ((n + h) * math.cos(la) * math.cos(lo),
            (n + h) * math.cos(la) * math.sin(lo),
            (n * (1 - WGS84_E2) + h) * math.sin(la))


def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(8):
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n


def gz_local_to_latlon(e, n, u, lat0, lon0, h0):
    """
    Gazebo's SphericalCoordinates conversion: a local ENU tangent plane at
    the world origin, via ECEF. This is what the NavSat sensor reports.
    """
    la, lo = math.radians(lat0), math.radians(lon0)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)
    dx = -math.sin(lo) * e - math.sin(la) * math.cos(lo) * n + math.cos(la) * math.cos(lo) * u
    dy = math.cos(lo) * e - math.sin(la) * math.sin(lo) * n + math.cos(la) * math.sin(lo) * u
    dz = math.cos(la) * n + math.sin(la) * u
    return ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)


# ------------------------------------------------------------------ terrain
class Terrain:
    def __init__(self, model_dir):
        self.dir = os.path.abspath(os.path.expanduser(model_dir))
        meta_path = os.path.join(self.dir, "terrain.yaml")
        if not os.path.exists(meta_path):
            raise SystemExit(f"{meta_path} missing. Regenerate the terrain with "
                             f"the current tools/make_terrain_model.py.")
        self.meta = m = read_yaml_flat(meta_path)
        self.lat0, self.lon0 = m["centre_lat"], m["centre_lon"]
        self.elev0 = m["centre_elevation_m"]
        self.extent = m["extent_m"]
        self.half = self.extent / 2.0
        self.k = float(m.get("crs_scale_k", 1.0))
        self.crs = site_crs(self.lat0, self.lon0, self.k)
        self._hm = {}

    def latlon_to_world(self, lat, lon):
        xs, ys = warp_transform(CRS.from_epsg(4326), self.crs, [lon], [lat])
        return xs[0], ys[0]

    def world_to_latlon(self, x, y):
        lons, lats = warp_transform(self.crs, CRS.from_epsg(4326), [x], [y])
        return lats[0], lons[0]

    def heightmap(self, which="visual"):
        if which not in self._hm:
            p = os.path.join(self.dir, self.meta[f"{which}_heightmap"])
            self._hm[which] = np.array(Image.open(p)).astype(np.float64) / 65535.0
        return self._hm[which]

    def ground_z(self, x, y, which="visual"):
        """World z of the rendered surface at (x, y), or None if outside."""
        hm = self.heightmap(which)
        n = hm.shape[0]
        res = self.extent / (n - 1)
        col, row = (x + self.half) / res, (self.half - y) / res
        if not (0 <= col <= n - 1 and 0 <= row <= n - 1):
            return None
        c0, r0 = min(int(col), n - 2), min(int(row), n - 2)
        fc, fr = col - c0, row - r0
        v = ((hm[r0, c0] * (1 - fc) + hm[r0, c0 + 1] * fc) * (1 - fr) +
             (hm[r0 + 1, c0] * (1 - fc) + hm[r0 + 1, c0 + 1] * fc) * fr)
        # Elevation is zmin + v*zrange; world z is elevation minus the centre
        # elevation, which is exactly v*zrange + z_offset (z_offset = zmin - centre).
        return float(v * self.meta["zrange_m"] + self.meta["z_offset_m"])
