#!/usr/bin/env python3
"""
make_terrain_model.py
=====================

Builds a georeferenced Gazebo Harmonic terrain model from a real-world DEM
and an optional satellite image, both supplied as GeoTIFFs.

The script clips a square, metric window centred on a given latitude and
longitude, reprojects it into the local UTM zone so that one pixel is one
consistent ground distance in both axes, resamples to a (2^n)+1 grid as
Gazebo's heightmap loader requires, and writes out a complete model
directory ready to <include> in a world file.

Vertical placement is handled explicitly. A Gazebo heightmap places its
lowest elevation at the model's own origin, so the model pose is offset
downwards by the difference between the centre elevation and the minimum
elevation in the window. The result is that world z = 0 sits at ground
level in the middle of the map, which is where you want vehicles to spawn.

Inputs
  --dem       GeoTIFF elevation model. Copernicus GLO-30 from
              opentopography.org is a good free source with no card needed.
  --texture   Optional GeoTIFF satellite image, e.g. Sentinel-2 true colour
              from browser.dataspace.copernicus.eu. Falls back to a plain
              colour if omitted.
  --lat/--lon Centre of the extracted window, WGS84 degrees.
  --extent    Side length of the square window in metres.
  --grid      Heightmap resolution. Must be (2^n)+1: 129, 257, 513, 1025.

Output (under <out>/<name>/)
  model.config
  model.sdf
  materials/textures/heightmap.png   16-bit grayscale
  materials/textures/aerial.png      RGB satellite texture
  materials/textures/normal.png      normal map derived from the heightmap
  world_snippet.txt                  spherical_coordinates and include tags

Dependencies
  pip install rasterio numpy pillow

Usage
  python3 make_terrain_model.py \
      --dem kanjikode_dem.tif --texture kanjikode_rgb.tif \
      --lat 10.79597 --lon 76.73253 --extent 1500 --grid 513 \
      --name kanjikode --out ~/Rover2Drone/models
"""

import argparse
import math
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.crs import CRS
from rasterio.transform import from_origin
from PIL import Image


def utm_epsg(lat, lon):
    """EPSG code of the UTM zone containing this coordinate."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def is_valid_grid(n):
    """Gazebo heightmaps need (2^k)+1 square dimensions."""
    m = n - 1
    return m > 0 and (m & (m - 1)) == 0


def latlon_to_utm(lat, lon, epsg):
    """Project a single WGS84 point into the given UTM CRS."""
    from rasterio.warp import transform as warp_transform
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(epsg),
                            [lon], [lat])
    return xs[0], ys[0]


def extract_window(src_path, lat, lon, extent_m, grid, epsg, resampling,
                   band_count=None):
    """
    Reproject and clip a source raster into a square UTM window.

    Returns an array shaped (bands, grid, grid).
    """
    cx, cy = latlon_to_utm(lat, lon, epsg)
    half = extent_m / 2.0
    res = extent_m / (grid - 1)

    # Origin is the top-left corner of the window.
    dst_transform = from_origin(cx - half, cy + half, res, res)
    dst_crs = CRS.from_epsg(epsg)

    with rasterio.open(src_path) as src:
        n_bands = band_count or src.count
        out = np.zeros((n_bands, grid, grid), dtype=np.float32)
        for b in range(n_bands):
            reproject(
                source=rasterio.band(src, b + 1),
                destination=out[b],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                dst_nodata=np.nan,
            )
    return out


def fill_voids(arr):
    """
    Replace NaN and absurd nodata values with the mean of what remains.

    DEMs frequently contain voids where the sensor could not measure. Left
    alone these become spikes or pits that will catch a vehicle.
    """
    bad = ~np.isfinite(arr) | (arr < -1000) | (arr > 9000)
    if bad.all():
        raise SystemExit("DEM window contains no valid elevation data. "
                         "Check that --lat/--lon fall inside the GeoTIFF.")
    if bad.any():
        arr = arr.copy()
        arr[bad] = np.nanmean(arr[~bad])
        print(f"  filled {int(bad.sum())} void pixels")
    return arr


def normal_map(height, res_m, z_range):
    """Tangent-space normal map derived from the height field."""
    h = height.astype(np.float32) * z_range
    dzdx = np.gradient(h, res_m, axis=1)
    dzdy = np.gradient(h, res_m, axis=0)
    nx, ny, nz = -dzdx, dzdy, np.ones_like(h)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([(nx / length * 0.5 + 0.5),
                    (ny / length * 0.5 + 0.5),
                    (nz / length * 0.5 + 0.5)], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Terrain generated from real elevation and imagery.
    Centre {lat:.6f}, {lon:.6f} (WGS84). Window {extent:.0f} m square,
    {grid}x{grid} heightmap. Elevation range {zmin:.1f} to {zmax:.1f} m.
  </description>
</model>
"""

MODEL_SDF = """<?xml version="1.0" ?>
<!--
  model.sdf
  Terrain heightmap generated by tools/make_terrain_model.py.

  Source centre: {lat:.6f}, {lon:.6f} WGS84
  Window:        {extent:.0f} m square, {grid}x{grid} samples ({res:.2f} m/pixel)
  Elevation:     {zmin:.1f} m to {zmax:.1f} m ({zrange:.1f} m range)

  The heightmap places its lowest sample at this model's origin. The world
  file should therefore include this model at z = {z_offset:.2f} so that
  ground level at the window centre coincides with world z = 0.

  Collision uses the same heightmap as the visual. Heightmap collision is
  expensive: if real-time factor suffers, replace the collision block with
  a flat plane and keep the heightmap visual only.
-->
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="terrain">
      <collision name="collision">
        <geometry>
          <heightmap>
            <uri>model://{name}/materials/textures/heightmap.png</uri>
            <size>{extent:.1f} {extent:.1f} {zrange:.3f}</size>
            <pos>0 0 0</pos>
          </heightmap>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <heightmap>
            <use_terrain_paging>false</use_terrain_paging>
            <texture>
              <diffuse>model://{name}/materials/textures/aerial.png</diffuse>
              <normal>model://{name}/materials/textures/normal.png</normal>
              <size>{extent:.1f}</size>
            </texture>
            <uri>model://{name}/materials/textures/heightmap.png</uri>
            <size>{extent:.1f} {extent:.1f} {zrange:.3f}</size>
            <pos>0 0 0</pos>
          </heightmap>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""

SNIPPET = """Paste into your world file.

1. Georeference the world so NavSat reports real coordinates:

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{lat:.6f}</latitude_deg>
      <longitude_deg>{lon:.6f}</longitude_deg>
      <elevation>{z_centre:.1f}</elevation>
    </spherical_coordinates>

2. Include the terrain, offset so ground level at the centre is world z = 0:

    <include>
      <uri>model://{name}</uri>
      <pose>0 0 {z_offset:.3f} 0 0 0</pose>
    </include>

3. Remove any existing <model name="ground_plane"> from the world, or the
   flat plane will intersect the terrain.

Elevation at window centre: {z_centre:.1f} m
Elevation range in window:  {zmin:.1f} m to {zmax:.1f} m
Ground resolution:          {res:.2f} m per heightmap pixel
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True, help="Elevation GeoTIFF")
    ap.add_argument("--texture", help="Satellite GeoTIFF (optional)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--extent", type=float, default=1500.0,
                    help="Side of the square window, metres")
    ap.add_argument("--grid", type=int, default=513,
                    help="Heightmap size, must be (2^n)+1")
    ap.add_argument("--name", required=True, help="Model directory name")
    ap.add_argument("--out", default=".", help="Parent models directory")
    args = ap.parse_args()

    if not is_valid_grid(args.grid):
        sys.exit(f"--grid must be (2^n)+1, e.g. 129, 257, 513, 1025. "
                 f"Got {args.grid}.")

    epsg = utm_epsg(args.lat, args.lon)
    res = args.extent / (args.grid - 1)
    print(f"UTM zone EPSG:{epsg}, {res:.2f} m per pixel")

    print("Reading DEM...")
    dem = extract_window(args.dem, args.lat, args.lon, args.extent,
                         args.grid, epsg, Resampling.bilinear, band_count=1)[0]
    dem = fill_voids(dem)

    zmin, zmax = float(dem.min()), float(dem.max())
    zrange = max(zmax - zmin, 1.0)
    z_centre = float(dem[args.grid // 2, args.grid // 2])
    z_offset = -(z_centre - zmin)
    print(f"  elevation {zmin:.1f} to {zmax:.1f} m, centre {z_centre:.1f} m")

    norm = (dem - zmin) / zrange

    root = os.path.join(os.path.expanduser(args.out), args.name)
    tex_dir = os.path.join(root, "materials", "textures")
    os.makedirs(tex_dir, exist_ok=True)

    # Row 0 of a PNG is the top. Gazebo reads heightmaps with +Y north, so
    # the array is flipped to keep north at the top of the image.
    hm = (np.clip(norm, 0.0, 1.0) * 65535.0).astype("<u2")
    Image.frombytes("I;16", (args.grid, args.grid), hm.tobytes()).save(
        os.path.join(tex_dir, "heightmap.png"))
    print(f"  wrote heightmap.png ({args.grid}x{args.grid}, 16-bit)")

    Image.fromarray(normal_map(norm, res, zrange)).save(
        os.path.join(tex_dir, "normal.png"))
    print("  wrote normal.png")

    if args.texture:
        print("Reading texture...")
        rgb = extract_window(args.texture, args.lat, args.lon, args.extent,
                             args.grid, epsg, Resampling.cubic, band_count=3)
        rgb = np.nan_to_num(rgb, nan=0.0)
        # Percentile stretch: satellite reflectance is rarely 0-255.
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(tex_dir, "aerial.png"))
        print("  wrote aerial.png")
    else:
        flat = np.zeros((args.grid, args.grid, 3), dtype=np.uint8)
        flat[:, :] = (110, 125, 85)
        Image.fromarray(flat).save(os.path.join(tex_dir, "aerial.png"))
        print("  no texture given, wrote flat colour aerial.png")

    fields = dict(name=args.name, lat=args.lat, lon=args.lon,
                  extent=args.extent, grid=args.grid, res=res,
                  zmin=zmin, zmax=zmax, zrange=zrange,
                  z_centre=z_centre, z_offset=z_offset)

    with open(os.path.join(root, "model.config"), "w") as f:
        f.write(MODEL_CONFIG.format(**fields))
    with open(os.path.join(root, "model.sdf"), "w") as f:
        f.write(MODEL_SDF.format(**fields))
    with open(os.path.join(root, "world_snippet.txt"), "w") as f:
        f.write(SNIPPET.format(**fields))

    print(f"\nModel written to {root}")
    print(SNIPPET.format(**fields))


if __name__ == "__main__":
    main()
