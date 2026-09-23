#!/usr/bin/env python3
"""
make_terrain_model.py
=====================

Builds a georeferenced Gazebo Harmonic terrain model from a real-world DEM
and an optional satellite image, both supplied as GeoTIFFs.

The script clips a square, metric window centred on a given latitude and
longitude, reprojects it into a transverse Mercator projection centred on
the site itself, resamples to a (2^n)+1 grid as Gazebo's heightmap loader
requires, and writes out a complete model directory ready to <include> in a
world file.

Why a site-centred projection rather than UTM
---------------------------------------------
Gazebo's NavSat sensor, and therefore PX4's GPS and EKF, work in a local
east-north-up frame aligned with TRUE north at the world origin. UTM grid
north is rotated from true north by the grid convergence angle, which grows
with distance from the zone's central meridian. At Attappadi (1.7 deg east
of UTM zone 43's meridian) it is 0.33 deg, enough to put simulated GPS
positions up to 5 m out at turbines 1 km from the origin. A transverse
Mercator with its central meridian and origin at the site centre has zero
convergence and unit scale there, and matches Gazebo's local frame to
millimetres over a few kilometres. The window centre maps to (0, 0).

Vertical placement is handled explicitly. A Gazebo heightmap places its
lowest elevation at z = 0 of its own geometry, so it is lowered by the
difference between the centre elevation and the minimum elevation in the
window. The result is that world z = 0 sits at ground level in the middle
of the map, which is where you want vehicles to spawn.

The offset is written into the heightmap's own <pos> element, NOT into a
model or <include> pose. Gazebo ignores model and include poses for
heightmap geometry: an offset placed there is silently dropped and the
terrain renders at its unshifted height.

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
  model.config, model.sdf
  terrain.yaml                               geometry metadata, read by every
                                             other tool via terrain_io.py
  materials/textures/heightmap_visual.png    16-bit, fine grid (rendering)
  materials/textures/heightmap_collision.png 16-bit, coarser grid (physics)
  materials/textures/aerial.png              satellite texture, own resolution
  materials/textures/normal.png              normal map from the visual grid
  world_snippet.txt

Resolution is set independently for three things, because they have very
different costs. The texture carries almost all the visible detail and is
cheap for the GPU, so it runs at the imagery's native resolution
(--texture-size, default 4096 px). The visual heightmap only needs to be fine
enough that relief looks smooth rather than faceted (--visual-grid, default
2049). Physics collision against a heightmap is expensive, so it uses a
coarser grid (--collision-grid, default 1025). Both heightmaps are
normalised with the same elevation range, so they agree wherever their
samples coincide. Note the source DEM is 30 m: finer grids interpolate it
smoothly, they do not add real elevation detail.

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


def site_crs(lat, lon, k=1.0):
    """
    Transverse Mercator centred on (lat, lon): true north, unit scale and
    zero grid convergence at the site, so it agrees with Gazebo's local ENU
    frame. See the module docstring for why this replaces UTM.
    """
    return CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat:.9f} +lon_0={lon:.9f} +k={k:.12f} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")


def is_valid_grid(n):
    """Gazebo heightmaps need (2^k)+1 square dimensions."""
    m = n - 1
    return m > 0 and (m & (m - 1)) == 0


def latlon_to_local(lat, lon, crs):
    """Project a single WGS84 point into the site CRS (metres, ENU)."""
    from rasterio.warp import transform as warp_transform
    xs, ys = warp_transform(CRS.from_epsg(4326), crs, [lon], [lat])
    return xs[0], ys[0]


def extract_window(src_path, lat, lon, extent_m, grid, crs, resampling,
                   band_count=None, vertex=True):
    """
    Reproject and clip a source raster into a square window in the site CRS.

    Returns an array shaped (bands, grid, grid).
    """
    cx, cy = latlon_to_local(lat, lon, crs)
    half = extent_m / 2.0
    if vertex:
        # Heightmaps: Gazebo places vertex i at -half + i*res, edges included.
        # Rasterio samples pixel CENTRES, so shift the grid origin by half a
        # pixel to make those centres land exactly on the vertices. Without
        # this the terrain renders half a pixel north-west of true position.
        res = extent_m / (grid - 1)
        dst_transform = from_origin(cx - half - res / 2, cy + half + res / 2,
                                    res, res)
    else:
        # Textures: pixels are areas spanning the window edge to edge.
        res = extent_m / grid
        dst_transform = from_origin(cx - half, cy + half, res, res)
    dst_crs = crs

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
        from rasterio.fill import fillnodata
        arr[bad] = 0.0
        arr = fillnodata(arr.astype("float32"), mask=(~bad).astype("uint8"),
                         max_search_distance=600, smoothing_iterations=0)
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

  The heightmap is lowered by {z_offset:.3f} m via its own <pos> element so
  that ground level at the window centre coincides with world z = 0.
  Gazebo ignores model and include poses for heightmaps, so the offset
  must live here; include this model at pose 0 0 0.

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
            <uri>model://{name}/materials/textures/heightmap_collision.png</uri>
            <size>{extent:.1f} {extent:.1f} {zrange:.4f}</size>
            <pos>0 0 {z_offset:.4f}</pos>
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
            <uri>model://{name}/materials/textures/heightmap_visual.png</uri>
            <size>{extent:.1f} {extent:.1f} {zrange:.4f}</size>
            <pos>0 0 {z_offset:.4f}</pos>
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

2. Include the terrain at the origin. The vertical offset is already baked
   into the heightmap's <pos>; Gazebo ignores include poses for heightmaps.

    <include>
      <uri>model://{name}</uri>
      <pose>0 0 0 0 0 0</pose>
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
    ap.add_argument("--visual-grid", type=int, default=2049,
                    help="Rendered heightmap size, (2^n)+1")
    ap.add_argument("--collision-grid", type=int, default=1025,
                    help="Physics heightmap size, (2^n)+1")
    ap.add_argument("--grid", type=int,
                    help="Deprecated: sets --collision-grid")
    ap.add_argument("--texture-size", type=int, default=4096,
                    help="Satellite texture size in pixels")
    ap.add_argument("--name", required=True, help="Model directory name")
    ap.add_argument("--out", default=".", help="Parent models directory")
    args = ap.parse_args()

    if args.grid:
        args.collision_grid = args.grid
    for g in (args.visual_grid, args.collision_grid):
        if not is_valid_grid(g):
            sys.exit(f"Grids must be (2^n)+1, e.g. 513, 1025, 2049. Got {g}.")

    # Scale the projection so world metres equal ground metres at the site
    # elevation, matching Gazebo's GPS model. Needs the centre elevation
    # first, so take it from a quick coarse pass.
    from terrain_io import site_scale
    probe = extract_window(args.dem, args.lat, args.lon, args.extent, 129,
                           site_crs(args.lat, args.lon), Resampling.bilinear, 1)[0]
    k = site_scale(args.lat, float(fill_voids(probe)[64, 64]))
    crs = site_crs(args.lat, args.lon, k)
    res = args.extent / (args.visual_grid - 1)
    print(f"Site-centred transverse Mercator. Visual grid {args.visual_grid} "
          f"({res:.2f} m), collision grid {args.collision_grid} "
          f"({args.extent / (args.collision_grid - 1):.2f} m), texture "
          f"{args.texture_size} px ({args.extent / args.texture_size:.2f} m)")

    print("Reading DEM...")
    dem_v = fill_voids(extract_window(args.dem, args.lat, args.lon, args.extent,
                       args.visual_grid, crs, Resampling.bilinear, 1)[0])
    dem_c = fill_voids(extract_window(args.dem, args.lat, args.lon, args.extent,
                       args.collision_grid, crs, Resampling.bilinear, 1)[0])
    # One normalisation for both grids so their surfaces coincide.
    zmin = float(min(dem_v.min(), dem_c.min()))
    zmax = float(max(dem_v.max(), dem_c.max()))
    zrange = max(zmax - zmin, 1.0)
    z_centre = float(dem_v[args.visual_grid // 2, args.visual_grid // 2])
    z_offset = -(z_centre - zmin)
    print(f"  elevation {zmin:.1f} to {zmax:.1f} m, centre {z_centre:.1f} m")

    root = os.path.join(os.path.expanduser(args.out), args.name)
    tex_dir = os.path.join(root, "materials", "textures")
    os.makedirs(tex_dir, exist_ok=True)
    # A rebuilt terrain invalidates gen_roads.py's pristine copies.
    for f in os.listdir(tex_dir):
        if f.endswith("_raw.png"):
            os.remove(os.path.join(tex_dir, f))
    for tag, grid, dem in (("visual", args.visual_grid, dem_v),
                           ("collision", args.collision_grid, dem_c)):
        hm = (np.clip((dem - zmin) / zrange, 0, 1) * 65535.0).astype("<u2")
        Image.frombytes("I;16", (grid, grid), hm.tobytes()).save(
            os.path.join(tex_dir, f"heightmap_{tag}.png"))
        print(f"  wrote heightmap_{tag}.png ({grid}x{grid})")
    Image.fromarray(normal_map((dem_v - zmin) / zrange, res, zrange)).save(
        os.path.join(tex_dir, "normal.png"))

    ts = args.texture_size
    if args.texture:
        print("Reading texture...")
        rgb = extract_window(args.texture, args.lat, args.lon, args.extent,
                             ts, crs, Resampling.cubic, band_count=3,
                             vertex=False)
        rgb = np.nan_to_num(rgb, nan=0.0)
        lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
                        ).save(os.path.join(tex_dir, "aerial.png"))
        print(f"  wrote aerial.png ({ts}x{ts})")
    else:
        flat = np.full((512, 512, 3), (110, 125, 85), dtype=np.uint8)
        Image.fromarray(flat).save(os.path.join(tex_dir, "aerial.png"))

    with open(os.path.join(root, "terrain.yaml"), "w") as f:
        f.write(f"""# terrain.yaml, written by make_terrain_model.py. Read via terrain_io.py.
name: {args.name}
centre_lat: {args.lat:.9f}
centre_lon: {args.lon:.9f}
centre_elevation_m: {z_centre:.4f}     # EGM2008 orthometric, = world z 0
extent_m: {args.extent:.3f}
zmin_m: {zmin:.4f}
zmax_m: {zmax:.4f}
zrange_m: {zrange:.4f}
z_offset_m: {z_offset:.4f}             # heightmap <pos> z
visual_grid: {args.visual_grid}
collision_grid: {args.collision_grid}
texture_px: {ts}
visual_heightmap: materials/textures/heightmap_visual.png
collision_heightmap: materials/textures/heightmap_collision.png
texture: materials/textures/aerial.png
dem_source: "{os.path.abspath(args.dem)}"
texture_source: "{os.path.abspath(args.texture) if args.texture else ''}"
projection: site-centred transverse Mercator (true north)
crs_scale_k: {k:.12f}                  # ground metres at site elevation
""")
    grid = args.visual_grid
    fields = dict(name=args.name, lat=args.lat, lon=args.lon,
                  extent=args.extent, grid=grid, res=res,
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
