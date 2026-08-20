#!/usr/bin/env python3
"""
fetch_terrain.py
================

Downloads elevation and satellite imagery for a square window centred on a
latitude/longitude, and writes them as GeoTIFFs ready for
make_terrain_model.py. No API key, no account, no payment method.

Elevation comes from the AWS Open Data Terrain Tiles bucket in Terrarium
format: a global, publicly readable S3 bucket of PNG tiles where elevation
in metres is packed into the colour channels as

    height = (red * 256 + green + blue / 256) - 32768

Attribution is required when using these tiles. The underlying data is a
blend of SRTM, NED, and other national datasets depending on region; for
India expect roughly 30 m native resolution regardless of the zoom level
requested, so zoom beyond 13 interpolates rather than adding detail.

Imagery comes from Esri World Imagery tiles by default. Check Esri's terms
before using the output in a publication. Pass --no-imagery to skip it and
supply your own texture (e.g. Sentinel-2 from the Copernicus Browser, which
is openly licensed and better suited to publication).

Outputs, in EPSG:3857 Web Mercator:
    <name>_dem.tif      single band float32 elevation in metres
    <name>_rgb.tif      three band uint8 imagery

Dependencies
    pip install rasterio numpy pillow requests

Usage
    python3 fetch_terrain.py --lat 11.0813198 --lon 76.713563 \
        --extent 3000 --name attappadi --out ~/Downloads

Zoom guidance
    dem-zoom   13 gives ~19 m/pixel at this latitude, matched to the source
    img-zoom   17 gives ~1.2 m/pixel, good for a ground texture
"""

import argparse
import io
import math
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from PIL import Image

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

R_EARTH = 6378137.0
TILE = 256

DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
IMG_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def lonlat_to_merc(lon, lat):
    """WGS84 degrees to Web Mercator metres."""
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def merc_to_tile(x, y, z):
    """Web Mercator metres to fractional tile indices at zoom z."""
    n = 2 ** z
    origin = math.pi * R_EARTH
    tx = (x + origin) / (2 * origin) * n
    ty = (origin - y) / (2 * origin) * n
    return tx, ty


def tile_to_merc(tx, ty, z):
    """Fractional tile indices back to Web Mercator metres."""
    n = 2 ** z
    origin = math.pi * R_EARTH
    x = tx / n * (2 * origin) - origin
    y = origin - ty / n * (2 * origin)
    return x, y


def window_bounds(lat, lon, extent_m):
    """
    Web Mercator bounds of a square window of `extent_m` GROUND metres.

    Mercator distances are inflated by 1/cos(latitude), so the window must
    be widened accordingly or the requested area comes out too small.
    """
    cx, cy = lonlat_to_merc(lon, lat)
    half = (extent_m / math.cos(math.radians(lat))) / 2.0
    return cx - half, cy - half, cx + half, cy + half


def fetch_tiles(url_tmpl, z, tx0, ty0, tx1, ty1, decode, bands, label):
    """
    Download and stitch a rectangular block of tiles.

    `decode` turns a PIL image into an array shaped (bands, TILE, TILE).
    """
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    total = nx * ny
    print(f"  {label}: {nx}x{ny} = {total} tiles at zoom {z}")
    if total > 400:
        sys.exit(f"{total} tiles is too many. Reduce --extent or the zoom.")

    mosaic = np.zeros((bands, ny * TILE, nx * TILE), dtype=np.float32)
    session = requests.Session()
    session.headers.update({"User-Agent": "rover2drone-terrain/1.0"})

    done = 0
    for iy in range(ny):
        for ix in range(nx):
            url = url_tmpl.format(z=z, x=tx0 + ix, y=ty0 + iy)
            for attempt in range(3):
                try:
                    r = session.get(url, timeout=30)
                    if r.status_code == 404:
                        # Missing tile: leave as zeros rather than aborting.
                        break
                    r.raise_for_status()
                    img = Image.open(io.BytesIO(r.content))
                    arr = decode(img)
                    mosaic[:, iy * TILE:(iy + 1) * TILE,
                           ix * TILE:(ix + 1) * TILE] = arr
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"    failed {url}: {exc}")
                    time.sleep(1.0 + attempt)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"    {done}/{total}")
    return mosaic


def decode_terrarium(img):
    """Terrarium PNG to elevation in metres."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h = (a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0) - 32768.0
    return h[np.newaxis, :, :]


def decode_rgb(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    return np.transpose(a, (2, 0, 1))


def write_geotiff(path, data, z, tx0, ty0, dtype):
    """Write a stitched mosaic with its Web Mercator georeferencing."""
    x0, y0 = tile_to_merc(tx0, ty0, z)
    n = 2 ** z
    res = (2 * math.pi * R_EARTH) / (n * TILE)
    transform = from_origin(x0, y0, res, res)
    bands = data.shape[0]
    with rasterio.open(path, "w", driver="GTiff",
                       height=data.shape[1], width=data.shape[2],
                       count=bands, dtype=dtype,
                       crs=CRS.from_epsg(3857), transform=transform,
                       compress="deflate") as dst:
        dst.write(data.astype(dtype))
    print(f"  wrote {path} ({data.shape[2]}x{data.shape[1]}, "
          f"{res:.2f} m/pixel in Mercator units)")


def tile_range(lat, lon, extent_m, z):
    x0, y0, x1, y1 = window_bounds(lat, lon, extent_m)
    tx0f, ty1f = merc_to_tile(x0, y0, z)
    tx1f, ty0f = merc_to_tile(x1, y1, z)
    return (int(math.floor(tx0f)), int(math.floor(ty0f)),
            int(math.floor(tx1f)), int(math.floor(ty1f)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--extent", type=float, default=3000.0,
                    help="Square window side in ground metres")
    ap.add_argument("--dem-zoom", type=int, default=13)
    ap.add_argument("--img-zoom", type=int, default=17)
    ap.add_argument("--no-imagery", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    ground_res = (156543.03392 * math.cos(math.radians(args.lat))
                  / (2 ** args.dem_zoom))
    print(f"Window {args.extent:.0f} m square at {args.lat}, {args.lon}")
    print(f"DEM ground resolution ~{ground_res:.1f} m/pixel")

    tx0, ty0, tx1, ty1 = tile_range(args.lat, args.lon, args.extent,
                                    args.dem_zoom)
    dem = fetch_tiles(DEM_URL, args.dem_zoom, tx0, ty0, tx1, ty1,
                      decode_terrarium, 1, "elevation")
    valid = dem[np.isfinite(dem) & (dem > -1000)]
    if valid.size:
        print(f"  elevation range {valid.min():.1f} to {valid.max():.1f} m")
    write_geotiff(os.path.join(out, f"{args.name}_dem.tif"),
                  dem, args.dem_zoom, tx0, ty0, "float32")

    if not args.no_imagery:
        tx0, ty0, tx1, ty1 = tile_range(args.lat, args.lon, args.extent,
                                        args.img_zoom)
        rgb = fetch_tiles(IMG_URL, args.img_zoom, tx0, ty0, tx1, ty1,
                          decode_rgb, 3, "imagery")
        write_geotiff(os.path.join(out, f"{args.name}_rgb.tif"),
                      rgb, args.img_zoom, tx0, ty0, "uint8")

    print(f"\nNow run:\n"
          f"  python3 tools/make_terrain_model.py \\\n"
          f"    --dem {out}/{args.name}_dem.tif \\\n"
          f"    --texture {out}/{args.name}_rgb.tif \\\n"
          f"    --lat {args.lat} --lon {args.lon} \\\n"
          f"    --extent {args.extent:.0f} --grid 513 \\\n"
          f"    --name {args.name} --out models")


if __name__ == "__main__":
    main()
