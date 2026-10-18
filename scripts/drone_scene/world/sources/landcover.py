#!/usr/bin/env python
"""landcover_imagery recipe: ESA WorldCover 10 m land cover + Sentinel-2 L2A true-colour RGB.

Everything is read straight from public AWS S3 buckets with HTTP range requests
(no credentials, no STAC API, nothing from blocked hosts):

  * ESA WorldCover  : https://esa-worldcover.s3.eu-central-1.amazonaws.com/
                      v200/2021/map/ESA_WorldCover_10m_2021_v200_<tile>_Map.tif   (CC-BY 4.0)
                      v100/2020/map/ESA_WorldCover_10m_2020_v100_<tile>_Map.tif
  * Sentinel-2 L2A  : https://sentinel-cogs.s3.us-west-2.amazonaws.com/
                      sentinel-s2-l2a-cogs/<utm zone>/<lat band>/<grid square>/<year>/<month>/<product id>/
                      (Element84 "Earth Search" COGs of Copernicus Sentinel-2 L2A; free, full and open)

Public API (all take a lon/lat bbox = (west, south, east, north), EPSG:4326):

  make_grid(bbox, res=10)                -> metric target grid (UTM of bbox centre, snapped to the S2 pixel grid)
  read_worldcover(bbox, year=2021)       -> (uint8 class array, affine, crs)   native EPSG:4326 1/12000 deg
  worldcover_on_grid(bbox, grid)         -> uint8 class array on the metric grid (nearest/mode)
  worldcover_histogram(arr, ...)         -> {class: {name, pixels, percent, km2}}
  s2_tiles_for_bbox(bbox)                -> MGRS tiles (zone/band/square) needed, incl. 'covers whole bbox'
  list_s2_scenes(tile, years, months)    -> product dicts incl. STAC metadata (cloud cover etc.)
  pick_s2_scene(bbox, grid, ...)         -> best (least cloudy, leaf-on) scene, with local SCL cloud stats
  read_s2_rgb(scene, grid, ...)          -> uint8 sRGB (H, W, 3) + reflectance + valid mask on the grid
  get_s2_rgb(bbox, res=10, ...)          -> one-call convenience (handles several tiles -> mosaic)
  save_geotiff / save_png                -> outputs

Run `python recipe.py` to execute everything on the Ames, Iowa test bbox.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date

# GDAL/HTTP tuning (must be in the environment before the first /vsicurl/ open)
for _k, _v in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import rasterio
import requests
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio.windows import Window

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
WC_HOST = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
S2_HOST = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S2_PREFIX = "sentinel-s2-l2a-cogs"

# WorldCover legend: code -> (name, legend RGB, hint for a simulation scene)
WC_CLASSES = {
    10: ("Tree cover", (0, 100, 0), "forest / tree instancing"),
    20: ("Shrubland", (255, 187, 34), "shrubs / low brush"),
    30: ("Grassland", (255, 255, 76), "grass ground"),
    40: ("Cropland", (240, 150, 255), "field texture (flat, regular)"),
    50: ("Built-up", (250, 0, 0), "buildings / roads (cover with OSM footprints)"),
    60: ("Bare / sparse vegetation", (180, 180, 180), "bare soil / rock / sand"),
    70: ("Snow and ice", (240, 240, 240), "snow"),
    80: ("Permanent water bodies", (0, 100, 200), "water surface"),
    90: ("Herbaceous wetland", (0, 150, 160), "marsh / reeds"),
    95: ("Mangroves", (0, 207, 117), "mangrove"),
    100: ("Moss and lichen", (250, 230, 160), "tundra"),
}
WC_NODATA = 0  # tile nodata; also what a missing (ocean) tile is filled with here
WC_VERSIONS = {2021: ("v200", "2021"), 2020: ("v100", "2020")}
WC_PX = 1.0 / 12000.0  # degrees per pixel, all tiles; 3 deg tiles = 36000 x 36000 px

# Sentinel-2 Scene Classification (SCL) codes
SCL_CLASSES = {
    0: "no_data", 1: "saturated_defective", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation",
    5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium_prob", 9: "cloud_high_prob",
    10: "thin_cirrus", 11: "snow_ice",
}
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)  # unusable for a clean true-colour picture
S2_BANDS_RGB = ("B04", "B03", "B02")  # red, green, blue (10 m, uint16, reflectance = DN / 10000)
S2_REFL_SCALE = 1e-4
S2_REFL_OFFSET = 0.0  # NOTE: STAC says raster:bands offset -0.1 for baseline>=04.00; the COG DNs are NOT offset
                      # (verified: water pixels DN~250 in every baseline). Do not subtract 0.1.
BANDS = "CDEFGHJKLMNPQRSTUVWX"

_SES = requests.Session()  # honours HTTPS_PROXY / REQUESTS_CA_BUNDLE from the environment


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx north, 327xx south."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _boundary_lonlat(bbox, n=25):
    w, s, e, nn = bbox
    xs = np.linspace(w, e, n)
    ys = np.linspace(s, nn, n)
    lon = np.concatenate([xs, xs, np.full(n, w), np.full(n, e)])
    lat = np.concatenate([np.full(n, s), np.full(n, nn), ys, ys])
    return lon, lat


def bbox_to_crs_bounds(bbox, epsg: int):
    """Axis-aligned bounds (minx, miny, maxx, maxy) of the lon/lat bbox in another CRS (densified edges)."""
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(*_boundary_lonlat(bbox))
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def make_grid(bbox, res: float = 10.0, epsg: int | None = None) -> dict:
    """Metric target grid covering the bbox.

    CRS defaults to the UTM zone of the bbox centre (e.g. EPSG:32615 for Ames, Iowa). The grid is snapped
    outward to multiples of `res` metres, which makes it pixel-aligned with the Sentinel-2 10/20/60 m grids
    of that UTM zone (tile origins are multiples of 10 m), so same-zone reads need no resampling at res=10.
    Returns dict(crs, epsg, transform, width, height, res, bounds=(x0, y0, x1, y1)); north-up, row 0 = north.
    """
    w, s, e, n = bbox
    if epsg is None:
        epsg = utm_epsg((w + e) / 2, (s + n) / 2)
    x0, y0, x1, y1 = bbox_to_crs_bounds(bbox, epsg)
    x0, y0 = math.floor(x0 / res) * res, math.floor(y0 / res) * res
    x1, y1 = math.ceil(x1 / res) * res, math.ceil(y1 / res) * res
    width, height = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return dict(crs=CRS.from_epsg(epsg), epsg=epsg, transform=Affine(res, 0, x0, 0, -res, y1),
                width=width, height=height, res=res, bounds=(x0, y0, x1, y1))


# --------------------------------------------------------------------------------------
# 1. ESA WorldCover
# --------------------------------------------------------------------------------------
def worldcover_tile_url(tile: str, year: int = 2021, vsicurl: bool = True) -> str:
    """URL of a WorldCover class-map tile. `tile` like 'N42W096' (lower-left corner, multiples of 3 deg)."""
    ver, yr = WC_VERSIONS[year]
    url = f"{WC_HOST}/{ver}/{yr}/map/ESA_WorldCover_10m_{yr}_{ver}_{tile}_Map.tif"
    return "/vsicurl/" + url if vsicurl else url


def worldcover_tile_name(lat_sw: int, lon_sw: int) -> str:
    """Tile id from the integer SW corner, e.g. (42, -96) -> 'N42W096'."""
    return f"{'N' if lat_sw >= 0 else 'S'}{abs(lat_sw):02d}{'E' if lon_sw >= 0 else 'W'}{abs(lon_sw):03d}"


def read_worldcover(bbox, year: int = 2021):
    """Read the WorldCover class raster for a lon/lat bbox at native resolution.

    Returns (arr uint8 [H, W], transform Affine, crs 'EPSG:4326'). Pixel size is exactly 1/12000 deg
    (~9.27 m N-S, ~6.9-9.3 m E-W depending on latitude). 0 = nodata (also used for missing ocean tiles).
    The window is snapped outward to whole pixels of the global 1/12000-degree grid; tiles are mosaicked
    on that grid, so bboxes spanning several 3x3-degree tiles work. Only the needed 1024x1024 COG blocks are
    downloaded (a 4x3 km bbox is ~0.3 MB).
    """
    w, s, e, n = bbox
    c0, c1 = int(math.floor((w + 180) * 12000)), int(math.ceil((e + 180) * 12000))
    r0, r1 = int(math.floor((90 - n) * 12000)), int(math.ceil((90 - s) * 12000))
    out = np.zeros((r1 - r0, c1 - c0), np.uint8)
    T = 36000  # pixels per tile edge
    for tr_ in range(r0 // T, (r1 - 1) // T + 1):
        for tc in range(c0 // T, (c1 - 1) // T + 1):
            lat_sw = 90 - (tr_ + 1) * 3
            lon_sw = tc * 3 - 180
            name = worldcover_tile_name(lat_sw, lon_sw)
            ra, rb = max(r0, tr_ * T), min(r1, (tr_ + 1) * T)
            ca, cb = max(c0, tc * T), min(c1, (tc + 1) * T)
            try:
                with rasterio.open(worldcover_tile_url(name, year)) as ds:
                    a = ds.read(1, window=Window(ca - tc * T, ra - tr_ * T, cb - ca, rb - ra))
                out[ra - r0:rb - r0, ca - c0:cb - c0] = a
            except rasterio.errors.RasterioIOError:
                pass  # ocean tiles do not exist in the bucket -> stay 0
    tf = Affine(WC_PX, 0, c0 * WC_PX - 180, 0, -WC_PX, 90 - r0 * WC_PX)
    return out, tf, CRS.from_epsg(4326)


def _lonlat_bbox_of_grid(grid, pad_deg):
    """Lon/lat bbox that contains the whole (rotated) grid rectangle, plus pad."""
    x0, y0, x1, y1 = grid["bounds"]
    tr = Transformer.from_crs(grid["epsg"], 4326, always_xy=True)
    xs, ys = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return (min(xs) - pad_deg, min(ys) - pad_deg, max(xs) + pad_deg, max(ys) + pad_deg)


def worldcover_on_grid(grid: dict, year: int = 2021) -> np.ndarray:
    """WorldCover classes resampled to the metric `grid` (nearest if grid res <= 10 m, else majority/mode)."""
    arr, tf, crs = read_worldcover(_lonlat_bbox_of_grid(grid, 0.002), year)
    dst = np.zeros((grid["height"], grid["width"]), np.uint8)
    reproject(arr, dst, src_transform=tf, src_crs=crs, dst_transform=grid["transform"], dst_crs=grid["crs"],
              resampling=Resampling.nearest if grid["res"] <= 10 else Resampling.mode,
              src_nodata=WC_NODATA, dst_nodata=WC_NODATA)
    return dst


def worldcover_histogram(arr: np.ndarray, transform: Affine | None = None, res_m: float | None = None) -> dict:
    """Class histogram of a WorldCover array.

    Area: if `res_m` is given (metric grid) every pixel is res_m^2. Else the array is in EPSG:4326
    (`transform` given) and pixel area is computed per row from latitude (spherical R=6371008.8 m).
    Returns {code: dict(name, pixels, percent, km2)} for valid (non-zero) pixels, sorted by code.
    """
    h = arr.shape[0]
    if res_m is not None:
        row_area = np.full(h, res_m * res_m)
    else:
        lat = transform.f + (np.arange(h) + 0.5) * transform.e
        R = 6371008.8
        row_area = (np.radians(abs(transform.a)) * R * np.cos(np.radians(lat))) * (np.radians(abs(transform.e)) * R)
    valid = arr != WC_NODATA
    total_px = int(valid.sum())
    res = {}
    for code in sorted(int(c) for c in np.unique(arr[valid])):
        m = arr == code
        res[code] = dict(name=WC_CLASSES.get(code, ("?",))[0], pixels=int(m.sum()),
                         percent=100.0 * m.sum() / max(total_px, 1),
                         km2=float((m.sum(axis=1) * row_area).sum() / 1e6))
    return res


def colorize_worldcover(arr: np.ndarray) -> np.ndarray:
    """uint8 class array -> RGB image using the official legend colours (nodata = black)."""
    lut = np.zeros((256, 3), np.uint8)
    for k, (_, rgb, _) in WC_CLASSES.items():
        lut[k] = rgb
    return lut[arr]


# --------------------------------------------------------------------------------------
# 2. Sentinel-2 L2A COGs on S3: discovery without a STAC API
# --------------------------------------------------------------------------------------
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def s3_list(host: str, prefix: str, delimiter: str | None = "/"):
    """ListObjectsV2 over plain HTTPS on a public bucket. Returns (common_prefixes, [(key, size)]); pages through >1000."""
    prefixes, keys, token = [], [], None
    while True:
        params = {"list-type": 2, "prefix": prefix, "max-keys": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = _SES.get(host + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        prefixes += [p.find("s3:Prefix", _NS).text for p in root.findall("s3:CommonPrefixes", _NS)]
        keys += [(c.find("s3:Key", _NS).text, int(c.find("s3:Size", _NS).text)) for c in root.findall("s3:Contents", _NS)]
        if root.findtext("s3:IsTruncated", namespaces=_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=_NS)
        else:
            return prefixes, keys


def lat_band(lat: float) -> str:
    """MGRS latitude band letter (C..X, 8 deg bands, X = 72..84)."""
    return BANDS[max(0, min(19, int((lat + 80) // 8)))]


def mgrs_square(zone: int, easting: float, northing: float) -> str:
    """Two-letter MGRS 100 km square id for UTM coordinates (northing includes the 10,000 km south false northing)."""
    col = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")[(zone - 1) % 3][int(easting // 100000) - 1]
    row = "ABCDEFGHJKLMNPQRSTUV"[(int(northing // 100000) + (5 if zone % 2 == 0 else 0)) % 20]
    return col + row


def s2_tiles_for_bbox(bbox, overlap_m: float = 10000.0) -> list[dict]:
    """MGRS (Sentinel-2) tiles that intersect the bbox.

    Each tile is the nominal 100 km MGRS square, extended by 9.76 km towards the east and south (S2 tiles are
    109.8 km wide: UL corner = square NW corner + (-40 m, +40 m)). Returns dicts
    {id:'15TVG', zone, band, square, epsg, extent:(x0,y0,x1,y1) in tile CRS, covers:bool (whole bbox inside the
    tile extent), primary:bool (bbox centre in nominal square)}, covering tiles first.
    """
    w, s, e, n = bbox
    lat0 = (s + n) / 2
    coslat = max(math.cos(math.radians(lat0)), 0.05)
    dlat = overlap_m / 111320.0
    dlon = overlap_m / (111320.0 * coslat)
    nx = int(min(100, max(2, math.ceil((e - w + 2 * dlon) * 111320 * coslat / 2000) + 1)))
    ny = int(min(100, max(2, math.ceil((n - s + 2 * dlat) * 111320 / 2000) + 1)))
    found = {}
    for lat in np.linspace(s - dlat, n + dlat, ny):
        for lon in np.linspace(w - dlon, e + dlon, nx):
            zone = int((lon + 180) // 6) % 60 + 1
            epsg = (32600 if lat >= 0 else 32700) + zone
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            key = (zone, lat_band(lat), mgrs_square(zone, x, y))
            found[key] = (epsg, math.floor(x / 1e5) * 1e5, math.floor(y / 1e5) * 1e5)
    cx, cy = (w + e) / 2, lat0
    tiles = []
    for (zone, band, sq), (epsg, e0, n0) in found.items():
        ext = (e0 - 40, n0 - 9760, e0 + 109760, n0 + 100040)
        bx0, by0, bx1, by1 = bbox_to_crs_bounds(bbox, epsg)
        covers = bx0 >= ext[0] and bx1 <= ext[2] and by0 >= ext[1] and by1 <= ext[3]
        inter = bx0 < ext[2] and bx1 > ext[0] and by0 < ext[3] and by1 > ext[1]
        if not inter:
            continue
        ccx, ccy = Transformer.from_crs(4326, epsg, always_xy=True).transform(cx, cy)
        primary = e0 <= ccx < e0 + 1e5 and n0 <= ccy < n0 + 1e5
        tiles.append(dict(id=f"{zone}{band}{sq}", zone=zone, band=band, square=sq, epsg=epsg, extent=ext,
                          covers=covers, primary=primary))
    tiles.sort(key=lambda t: (not t["covers"], not t["primary"], t["id"]))
    return tiles


def _scene_base(zone, band, square, year, month, pid) -> str:
    return f"{S2_HOST}/{S2_PREFIX}/{zone}/{band}/{square}/{year}/{month}/{pid}"  # month is NOT zero padded


def _band_candidates(band):
    i = BANDS.find(band)
    return [band] + [BANDS[i + d] for d in (-1, 1) if 0 <= i + d < 20]


def list_s2_products(tile: dict, year: int, month: int) -> list[tuple[str, str]]:
    """[(band_letter_used, product_id)] e.g. ('T', 'S2B_15TVG_20250709_0_L2A') for a tile and sensing month,
    via S3 listing. Month folders are unpadded ('7' not '07'). Falls back to neighbouring latitude band
    letters (tiles straddling a band boundary are filed under one of them)."""
    for b in _band_candidates(tile["band"]):
        pre, _ = s3_list(S2_HOST, f"{S2_PREFIX}/{tile['zone']}/{b}/{tile['square']}/{year}/{month}/")
        if pre:
            return [(b, p.rstrip("/").split("/")[-1]) for p in pre]
    return []


def fetch_s2_item(tile: dict, year: int, month: int, band: str, pid: str) -> dict | None:
    """Download the per-product STAC item json (~22 kB, `<pid>.json`) and return the fields we care about."""
    base = _scene_base(tile["zone"], band, tile["square"], year, month, pid)
    r = _SES.get(f"{base}/{pid}.json", timeout=60)
    if r.status_code != 200:
        return None
    j = r.json()
    p = j["properties"]
    return dict(
        id=pid, base=base, tile=tile["id"], epsg=p.get("proj:epsg"), datetime=p["datetime"],
        platform=p.get("platform"), baseline=p.get("s2:processing_baseline"),
        cloud=p.get("eo:cloud_cover"), cloud_shadow=p.get("s2:cloud_shadow_percentage"),
        nodata=p.get("s2:nodata_pixel_percentage"), snow=p.get("s2:snow_ice_percentage"),
        vegetation=p.get("s2:vegetation_percentage"), thin_cirrus=p.get("s2:thin_cirrus_percentage"),
        sun_elev=p.get("view:sun_elevation"), view_incidence=p.get("view:incidence_angle"),
        boa_offset_applied=p.get("earthsearch:boa_offset_applied"),
        assets={k: v["href"] for k, v in j["assets"].items() if v["href"].startswith("https://")},
    )


def list_s2_scenes(tile: dict, years, months=(6, 7, 8), workers: int = 12) -> list[dict]:
    """All scenes of a tile for the given years/months with their STAC metadata (see fetch_s2_item)."""
    jobs = [(y, m) for y in years for m in months]
    with ThreadPoolExecutor(workers) as ex:
        lists = list(ex.map(lambda ym: [(ym, bp) for bp in list_s2_products(tile, *ym)], jobs))
        flat = [x for lst in lists for x in lst]
        items = list(ex.map(lambda x: fetch_s2_item(tile, x[0][0], x[0][1], x[1][0], x[1][1]), flat))
    return [i for i in items if i]


# --------------------------------------------------------------------------------------
# 3. reading windows of Sentinel-2 COGs on the metric grid
# --------------------------------------------------------------------------------------
def _read_on_grid(url: str, grid: dict, resampling=Resampling.bilinear, count: int = 1) -> np.ndarray:
    """Read a COG into `grid` (reprojecting/resampling if CRS, resolution or pixel phase differ). Outside = 0.

    Fast path (same CRS, grid pixel-aligned, grid res = integer multiple k >= 1 of the file's res): plain
    windowed read with out_shape, which GDAL serves from the COG overviews (k=2,4,8,16 for S2) so coarse
    previews of big areas are cheap. Otherwise a WarpedVRT (reads full-res blocks: no overview use!).
    """
    with rasterio.open("/vsicurl/" + url if not url.startswith("/vsi") else url) as ds:
        t = ds.transform
        idx = list(range(1, count + 1))
        k = grid["res"] / t.a
        fast = ds.crs == grid["crs"] and abs(t.a + t.e) < 1e-9 and k >= 1 - 1e-9 and abs(k - round(k)) < 1e-6
        if fast:
            k = int(round(k))
            col0 = (grid["transform"].c - t.c) / t.a
            row0 = (grid["transform"].f - t.f) / t.e
            fast = abs(col0 - round(col0)) < 1e-6 and abs(row0 - round(row0)) < 1e-6
        if fast:
            col0, row0 = int(round(col0)), int(round(row0))
            W, H = grid["width"], grid["height"]
            # output pixel range whose k x k native footprint lies inside the file (rest stays 0 = outside)
            i0, i1 = max(0, -(-(0 - col0) // k)), min(W, (ds.width - col0) // k)
            j0, j1 = max(0, -(-(0 - row0) // k)), min(H, (ds.height - row0) // k)
            a = np.zeros((len(idx), H, W), ds.dtypes[0])
            if i1 > i0 and j1 > j0:
                win = Window(col0 + i0 * k, row0 + j0 * k, (i1 - i0) * k, (j1 - j0) * k)
                a[:, j0:j1, i0:i1] = ds.read(indexes=idx, window=win, out_shape=(len(idx), j1 - j0, i1 - i0),
                                             resampling=resampling if k > 1 else Resampling.nearest)
        else:
            with WarpedVRT(ds, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                           height=grid["height"], resampling=resampling, nodata=0) as vrt:
                a = vrt.read(indexes=idx)
    return a[0] if count == 1 else a


def coarse_grid(grid: dict, k: int = 4) -> dict:
    """Same area with k-times coarser pixels, origin snapped outward to multiples of the new res (so it stays
    pixel-aligned with S2 tiles for res in {10, 20, 40} m): cheap previews / stats served from COG overviews."""
    x0, y0, x1, y1 = grid["bounds"]
    res = grid["res"] * k
    x0, y0 = math.floor(x0 / res) * res, math.floor(y0 / res) * res
    x1, y1 = math.ceil(x1 / res) * res, math.ceil(y1 / res) * res
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return dict(crs=grid["crs"], epsg=grid["epsg"], transform=Affine(res, 0, x0, 0, -res, y1), width=w, height=h,
                res=res, bounds=(x0, y0, x1, y1))


def stats_grid(grid: dict, target_px: int = 600) -> dict:
    """Coarse version of `grid` (>= 20 m pixels, <= ~target_px along the long side) used to rank scenes cheaply."""
    ext = max(grid["width"], grid["height"]) * grid["res"]
    k = max(1, int(math.ceil(max(20.0, ext / target_px) / grid["res"])))
    while (grid["res"] * k) % 20 != 0:
        k += 1
    return coarse_grid(grid, k) if k > 1 else grid


def scene_scl(scene: dict, grid: dict) -> np.ndarray:
    """Scene Classification (SCL, 20 m native, uint8) nearest-resampled onto the grid. 0 = nodata/outside."""
    return _read_on_grid(f"{scene['base']}/SCL.tif", grid, Resampling.nearest)


def scl_stats(scl: np.ndarray) -> dict:
    """Fractions (0..1) of SCL classes plus 'bad' (= any of SCL_BAD)."""
    n = scl.size
    st = {SCL_CLASSES[c]: float((scl == c).sum()) / n for c in SCL_CLASSES}
    st["bad"] = float(np.isin(scl, SCL_BAD).sum()) / n
    return st


def scene_ndvi(scene: dict, grid: dict, k: int = 4) -> float:
    """Median NDVI (B08, B04) over SCL 'vegetation' pixels: a leaf-on indicator (summer Iowa crops/trees
    ~0.7-0.9, leaf-off / bare < 0.3). Evaluated on a k-times coarser grid (default 40 m) so it is served from
    the COG overviews (~0.1 MB instead of ~5.5 MB). NaN if no vegetation pixels."""
    g = coarse_grid(grid, k)
    scl = scene_scl(scene, g)
    red = _read_on_grid(f"{scene['base']}/B04.tif", g).astype(np.float32) * S2_REFL_SCALE
    nir = _read_on_grid(f"{scene['base']}/B08.tif", g).astype(np.float32) * S2_REFL_SCALE
    m = (scl == 4) & (red > 0) & (nir > 0)
    if not m.any():
        return float("nan")
    return float(np.median((nir[m] - red[m]) / (nir[m] + red[m])))


def pick_s2_scene(bbox, grid: dict, tiles: list[dict] | None = None, years=None, months=(6, 7, 8),
                  max_scenes: int = 60, verbose: bool = True, clear_tol: float = 0.01):
    """Choose the least cloudy leaf-on Sentinel-2 L2A scene for the bbox.

    1. tiles: the given tiles (default: covering tiles if any, else the primary tile).
    2. List products per tile/year/month on S3 and read each product's STAC json: tile-wide `eo:cloud_cover`
       (%, whole 110x110 km tile, not the bbox!) etc. Keep the `max_scenes` with the lowest tile-wide cloud.
    3. For every candidate read the SCL (scene classification, 20 m) window of the bbox on a coarse grid
       (`stats_grid`, ~0.08 MB per scene) and compute the LOCAL fraction of cloud / shadow / cirrus / snow /
       nodata -> local['bad']. `bad_adj` = bad minus the smallest nodata fraction among the candidates (removes the
       part of the bbox that lies outside the tile, so partial-coverage tiles in a mosaic are judged fairly).
    4. Among scenes within `clear_tol` (absolute fraction, default 1 %) of the best bad_adj, prefer the highest
       median NDVI over vegetation pixels (peak leaf-on). Ties -> lower tile-wide cloud.
    `years` default: this year if today is >= Sept 1, else last year.
    Returns (best_scene or None, all candidates ranked best-first; each has s['local'] stats).
    """
    if years is None:
        today = date.today()
        years = (today.year,) if today.month >= 9 else (today.year - 1,)
    if tiles is None:
        tiles = s2_tiles_for_bbox(bbox)
        cov = [t for t in tiles if t["covers"]]
        tiles = cov if cov else tiles[:1]
    scenes = []
    for t in tiles:
        scenes += list_s2_scenes(t, years, months)
    if verbose:
        print(f"  {len(scenes)} candidate scenes on tiles {[t['id'] for t in tiles]} years={list(years)} months={list(months)}")
    scenes.sort(key=lambda s: s["cloud"] if s["cloud"] is not None else 100)
    scenes = scenes[:max_scenes]
    sg = stats_grid(grid)

    def local(s):
        try:
            s["local"] = scl_stats(scene_scl(s, sg))
        except Exception as ex:  # noqa: BLE001 - one broken scene must not kill the search
            s["local"] = dict(bad=1.0, no_data=1.0, error=str(ex))
        return s

    with ThreadPoolExecutor(8) as ex:
        scenes = list(ex.map(local, scenes))
    scenes = [s for s in scenes if "error" not in s["local"]]
    if not scenes:
        return None, scenes
    base_nodata = min(s["local"]["no_data"] for s in scenes)
    for s in scenes:
        s["local"]["bad_adj"] = max(0.0, s["local"]["bad"] - base_nodata)
    scenes.sort(key=lambda s: (s["local"]["bad_adj"], s["cloud"] if s["cloud"] is not None else 100))
    near = [s for s in scenes if s["local"]["bad_adj"] <= scenes[0]["local"]["bad_adj"] + clear_tol][:5]
    for s in near:
        s["ndvi"] = scene_ndvi(s, grid)
    near.sort(key=lambda s: (-(s["ndvi"] if s["ndvi"] == s["ndvi"] else -1), s["cloud"]))
    return near[0], scenes


# --------------------------------------------------------------------------------------
# 4. reflectance -> display RGB
# --------------------------------------------------------------------------------------
def _srgb_oetf(lin):
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(np.maximum(lin, 1e-9), 1 / 2.4) - 0.055)


def reflectance_to_srgb(refl: np.ndarray, gain: float = 2.5) -> np.ndarray:
    """Surface reflectance [0..1] -> uint8 sRGB. lin = clip(gain*refl, 0, 1) then the sRGB transfer function.

    gain 2.5 is the usual 'true colour' setting (Sentinel Hub style): vegetation (refl 0.03-0.10) -> sRGB
    ~60-110, bright roofs/soil (0.25) -> ~200, clouds/snow saturate. Keep gain FIXED across all tiles of a
    scene so brightness matches between tiles.
    """
    lin = np.clip(refl * gain, 0.0, 1.0)
    return np.clip(_srgb_oetf(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def read_s2_rgb(scene: dict, grid: dict, mode: str = "srgb", gain: float = 2.5, resampling=Resampling.bilinear):
    """True-colour image of a scene on `grid`.

    mode 'srgb'  : B04/B03/B02 reflectance (DN/10000) -> reflectance_to_srgb(gain).  (recommended)
         'tci'   : ESA's 8-bit TCI.tif as is (linear stretch, reflectance 0.25 -> 255, no gamma: looks dark/contrasty)
         'stretch': per-image 2-98 percentile stretch + sRGB gamma (pretty, but brightness differs per tile)
    Returns dict(rgb uint8 [H,W,3], refl float32 [H,W,3] or None, valid bool [H,W]).
    """
    if mode == "tci":
        a = _read_on_grid(f"{scene['base']}/TCI.tif", grid, resampling, count=3)
        rgb = np.moveaxis(a, 0, -1).copy()
        return dict(rgb=rgb, refl=None, valid=(rgb.max(-1) > 0))
    dn = [_read_on_grid(f"{scene['base']}/{b}.tif", grid, resampling) for b in S2_BANDS_RGB]
    valid = np.all([d > 0 for d in dn], axis=0)
    refl = np.stack(dn, -1).astype(np.float32) * S2_REFL_SCALE + S2_REFL_OFFSET
    if mode == "stretch":
        v = refl[valid] if valid.any() else refl.reshape(-1, 3)
        lo, hi = np.percentile(v, [2, 98])
        lin = np.clip((refl - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = (np.clip(_srgb_oetf(lin), 0, 1) * 255 + 0.5).astype(np.uint8)
    else:
        rgb = reflectance_to_srgb(refl, gain)
    rgb[~valid] = 0
    return dict(rgb=rgb, refl=refl, valid=valid)


def get_s2_rgb(bbox, res: float = 10.0, years=None, months=(6, 7, 8), mode: str = "srgb", gain: float = 2.5,
               fill_extra: int = 2, fill_threshold: float = 0.005, verbose: bool = True) -> dict:
    """One call: grid -> tiles -> best scene(s) -> RGB on the metric grid.

    If a single tile (incl. its 9.8 km east/south overlap) covers the bbox, the best scene of all covering tiles
    is used. Otherwise the best scene of each intersecting tile is picked and the images are mosaicked
    (seams are visible if dates differ). If the best scene still has > `fill_threshold` (0.5 %) cloud/shadow in
    the bbox, up to `fill_extra` next-best scenes are used to fill only the cloudy pixels. `res` should be
    10, 20 or 40 m (aligned with the S2 grid; 20/40 m are served from COG overviews = much cheaper).
    Returns dict(rgb uint8, refl float32, valid bool, grid, scenes=[used scene dicts (best first)], tiles,
    candidates=[ranked scenes]).
    """
    grid = make_grid(bbox, res)
    tiles = s2_tiles_for_bbox(bbox)
    if verbose:
        print(f"  grid EPSG:{grid['epsg']} {grid['width']}x{grid['height']} px @ {res} m; tiles: " +
              ", ".join(f"{t['id']}{'*' if t['covers'] else ''}" for t in tiles))
    cov = [t for t in tiles if t["covers"]]
    groups = [cov] if cov else [[t] for t in tiles]
    shape = (grid["height"], grid["width"])
    rgb = np.zeros(shape + (3,), np.uint8)
    refl = np.zeros(shape + (3,), np.float32)
    valid = np.zeros(shape, bool)  # any scene has data here
    ok = np.zeros(shape, bool)     # a clear (non-cloud, non-shadow ...) pixel has been placed here
    used, cands = [], []
    for g in groups:
        best, ranked = pick_s2_scene(bbox, grid, g, years, months, verbose=verbose)
        cands += ranked
        if best is None:
            continue
        use = [best]
        if fill_extra and best["local"]["bad_adj"] > fill_threshold:
            use += [s for s in ranked if s is not best and s["local"]["bad_adj"] < 0.5][:fill_extra]
        for j, s in enumerate(use):
            r = read_s2_rgb(s, grid, mode, gain)
            clear = r["valid"] & ~np.isin(scene_scl(s, grid), SCL_BAD)
            take = (clear & ~ok) | (r["valid"] & ~valid)  # clear pixels replace unclear ones; else only fill holes
            rgb[take] = r["rgb"][take]
            if r["refl"] is not None:
                refl[take] = r["refl"][take]
            ok |= clear
            valid |= r["valid"]
            s["role"] = "best" if j == 0 else "fill"
            used.append(s)
    return dict(rgb=rgb, refl=refl, valid=valid, grid=grid, scenes=used, tiles=tiles, candidates=cands)


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------
def save_geotiff(path: str, arr: np.ndarray, grid: dict, nodata=None, palette: dict | None = None):
    """Write [H,W] or [H,W,C] uint8 array as a deflate-compressed, tiled GeoTIFF in the grid CRS."""
    a = arr[None] if arr.ndim == 2 else np.moveaxis(arr, -1, 0)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=a.dtype, crs=grid["crs"], transform=grid["transform"], compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256, nodata=nodata) as ds:
        ds.write(a)
        if palette is not None and a.shape[0] == 1:
            ds.write_colormap(1, palette)


def save_png(path: str, rgb: np.ndarray):
    from PIL import Image
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------------------
# demo / test: west Ames, Iowa
# --------------------------------------------------------------------------------------
TEST_BBOX = (-93.715, 42.012, -93.665, 42.038)  # west, south, east, north


class DownloadMeter:
    """Context manager that counts bytes fetched: GDAL /vsicurl/ range requests (parsed from CPL_DEBUG log lines)
    and plain `requests` calls (S3 listings, STAC json). Use: `with DownloadMeter() as m: ...; print(m)`."""

    _pat = re.compile(r"Downloading (\d+)-(\d+)")

    def __init__(self):
        self.gdal_bytes = self.gdal_requests = self.http_bytes = self.http_requests = 0

    def __enter__(self):
        meter = self

        class _H(logging.Handler):
            def emit(self, rec):
                m = meter._pat.search(rec.getMessage())
                if m:
                    meter.gdal_bytes += int(m.group(2)) - int(m.group(1)) + 1
                    meter.gdal_requests += 1

        def _hook(r, *a, **k):
            meter.http_bytes += len(r.content)
            meter.http_requests += 1

        self._h, self._hook = _H(), _hook
        self._lg = logging.getLogger()  # root: rasterio/GDAL messages come from several child loggers
        self._lvl = self._lg.level
        self._lg.addHandler(self._h)
        self._lg.setLevel(logging.DEBUG)
        _SES.hooks["response"].append(_hook)
        self._env = rasterio.Env(CPL_DEBUG=True)
        self._env.__enter__()
        return self

    def __exit__(self, *exc):
        self._env.__exit__(*exc)
        _SES.hooks["response"].remove(self._hook)
        self._lg.removeHandler(self._h)
        self._lg.setLevel(self._lvl)
        return False

    @property
    def megabytes(self):
        return (self.gdal_bytes + self.http_bytes) / 1e6

    def __str__(self):
        return (f"{self.megabytes:.1f} MB total = GDAL ranges {self.gdal_bytes / 1e6:.1f} MB ({self.gdal_requests} req) + "
                f"S3 list/json {self.http_bytes / 1e6:.2f} MB ({self.http_requests} req)")


def main(out_dir: str | None = None, bbox=TEST_BBOX, years=None):
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    with DownloadMeter() as meter:
        grid = make_grid(bbox, 10.0)
        print(f"bbox {bbox}  -> grid EPSG:{grid['epsg']} {grid['width']}x{grid['height']} px @10 m, bounds {grid['bounds']}")

        # ---- WorldCover ----
        t1 = time.time()
        arr, tf, crs = read_worldcover(bbox)
        print(f"\n[WorldCover 2021 v200] native EPSG:4326 window {arr.shape} px, {time.time() - t1:.1f} s, "
              f"{meter.megabytes:.2f} MB so far")
        hist = worldcover_histogram(arr, tf)
        for c, d in hist.items():
            print(f"  {c:3d} {d['name']:<24s} {d['pixels']:7d} px {d['percent']:6.2f} %  {d['km2']:7.3f} km2")
        wc10 = worldcover_on_grid(grid)
        h10 = worldcover_histogram(wc10, res_m=10)
        print("  on the 10 m UTM grid (bbox bounding rectangle):", {c: f"{d['percent']:.1f}%" for c, d in h10.items()})

        # ---- Sentinel-2 ----
        t1 = time.time()
        tiles = s2_tiles_for_bbox(bbox)
        print("\n[Sentinel-2 L2A COG] tiles:", [(t["id"], "covers" if t["covers"] else "partial", "primary" if t["primary"] else "")
                                               for t in tiles])
        res = get_s2_rgb(bbox, 10.0, years=years)
    save_geotiff(os.path.join(out_dir, "worldcover_utm10.tif"), wc10, grid, nodata=0,
                 palette={k: (*v[1], 255) for k, v in WC_CLASSES.items()})
    save_png(os.path.join(out_dir, "worldcover_utm10.png"), colorize_worldcover(wc10))
    sc = res["scenes"][0]
    print(f"  picked {sc['id']}  {sc['datetime']}  baseline {sc['baseline']}  sun elevation {sc['sun_elev']:.1f} deg")
    print(f"    tile-wide eo:cloud_cover {sc['cloud']:.2f} %  cloud_shadow {sc['cloud_shadow']:.2f} %  nodata {sc['nodata']:.2f} %")
    print(f"    LOCAL (SCL in bbox): bad {100 * sc['local']['bad']:.2f} %  " +
          ", ".join(f"{k} {100 * v:.1f}%" for k, v in sc["local"].items() if k not in ("bad", "bad_adj") and v > 0.0005))
    print(f"    median NDVI over vegetation px: {sc['ndvi']:.3f}   (fill scenes used: {len(res['scenes']) - 1})")
    print("  top candidates:")
    for s in res["candidates"][:6]:
        nd = f"ndvi {s['ndvi']:.3f}" if "ndvi" in s else ""
        print(f"    {s['id']}  local_bad {100 * s['local']['bad']:5.2f} %  tile_cloud {s['cloud']:6.2f} %  {nd}")
    rgb, valid = res["rgb"], res["valid"]
    print(f"  RGB {rgb.shape} valid {100 * valid.mean():.1f} %, per-channel mean {rgb[valid].mean(0).round(1)}, "
          f"p2/p50/p98 {np.percentile(rgb[valid], [2, 50, 98]).round(0)}")
    r = res["refl"][valid]
    print(f"  reflectance mean R/G/B {r.mean(0).round(4)}  min {r.min():.4f} max {r.max():.4f}")
    save_geotiff(os.path.join(out_dir, "s2_rgb_utm10.tif"), rgb, grid, nodata=0)
    save_png(os.path.join(out_dir, "s2_rgb_utm10.png"), rgb)
    print(f"  S2 part took {time.time() - t1:.1f} s")
    print(f"\ntotal {time.time() - t0:.1f} s; downloaded {meter}; outputs in {out_dir}")
    return res


if __name__ == "__main__":
    y = tuple(int(v) for v in sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    main(years=y)
