#!/usr/bin/env python
"""Meta / WRI global ~1 m canopy height maps (public S3 bucket ``dataforgood-fb-data``)
-> canopy-height window for a lon/lat bbox -> individual tree detection.

Two generations live in the same bucket (both CC BY 4.0, anonymous HTTPS, no sign-up):

  version  S3 prefix                                           tile level  tile px  layout
  -------  --------------------------------------------------  ----------  -------  ------------------------------
  "v1"     forests/v1/alsgedi_global_v6_float/chm/<qk>.tif     9           65536^2  uint8 m, 1-row strips, no overviews
  "v2"     forests/v2/global/dinov3_global_chm_v2_ml3/chm/     10          32768^2  uint8 m, 512px COG tiles + overviews
           <qk>.tif                                                              + internal mask (CHMv2, 2026)

Both are EPSG:3857, uint8 metres, 1.19433 m pixel (= web-mercator zoom-17 pixel, which is only
~0.89 m on the ground at 42 N: ground size = 1.19433 * cos(lat)).  All tiles sit on ONE global
zoom-17 pixel grid, so mosaicking tiles is pure integer arithmetic.

Public API
----------
    quadkeys_for_bbox(bbox, version)      -> list of tile quadkeys covering the bbox
    read_chm(bbox, version="v2")          -> Chm (uint8 metres, 255 = nodata, EPSG:3857 grid)
    acquisition_info(bbox, version)       -> imagery vintage (v1: global date raster; v2: tile geojson)
    detect_trees(chm, min_height=3.0)     -> numpy structured array of trees
    trees_to_csv(trees, path)

Everything is plain numpy/rasterio/scipy/pyproj/requests/shapely; no GDAL python bindings.

Usage:  python recipe.py [v1|v2] [west south east north]     (default: v2, W Ames test box)
Measured (this sandbox, Ames 4.1 x 2.9 km): v2 read 2.8 s / ~2 MB, detect 5.9 s; v1 read 5.4 s / ~14 MB.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from pyproj import Transformer, Proj
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)  # v1 .msk has no georef

BUCKET_URL = "https://dataforgood-fb-data.s3.amazonaws.com"

SOURCES = {
    "v1": dict(prefix="forests/v1/alsgedi_global_v6_float/chm/", zoom=9,
               mask_prefix="forests/v1/alsgedi_global_v6_float/msk/",
               meta_prefix="forests/v1/alsgedi_global_v6_float/metadata/",
               date_raster="forests/v1/alsgedi_global_v6_float/CHM_acquisition_date.tif"),
    "v2": dict(prefix="forests/v2/global/dinov3_global_chm_v2_ml3/chm/", zoom=10,
               mask_prefix=None,
               meta_prefix="forests/v2/global/dinov3_global_chm_v2_ml3/metadata/",
               date_raster=None),
}

# Web-mercator zoom-17 pixel grid (shared by every tile of both versions)
WM_HALF = 20037508.342789244
PIX_ZOOM = 17
RES = 2 * WM_HALF / (256 * 2 ** PIX_ZOOM)  # 1.1943285668550503 m (mercator metres)
NODATA_U8 = 255

_T_LL2M = Transformer.from_crs(4326, 3857, always_xy=True)
_T_M2LL = Transformer.from_crs(3857, 4326, always_xy=True)

# GDAL/HTTP settings that matter for speed (S3 COG / strip reads over /vsicurl)
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.msk",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="1",
    GDAL_HTTP_MULTIPLEX="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE=str(64 * 1024 * 1024),
)


# --------------------------------------------------------------------------------------
# quadkey / tile maths
# --------------------------------------------------------------------------------------
def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Slippy-map tile (x, y) at zoom ``z`` containing lon/lat."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return min(max(int(x), 0), n - 1), min(max(int(y), 0), n - 1)


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    """Bing quadkey string (z digits) for tile x, y, z.  Digit = (x bit) + 2*(y bit)."""
    out = []
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        out.append(str((1 if x & m else 0) + (2 if y & m else 0)))
    return "".join(out)


def quadkeys_for_bbox(bbox, version: str = "v2") -> list[str]:
    """Quadkeys of all tiles (level 9 for v1, level 10 for v2) touching ``bbox`` = (W, S, E, N) lon/lat.

    Tile names are simply the quadkey; no index download needed.  (The bucket also has
    ``tiles.geojson`` per product: 15 MB for v1, 59 MB for v2 -- the *absence* of a tile there/in
    the bucket means no data, e.g. open ocean.)"""
    z = SOURCES[version]["zoom"]
    w, s, e, n = bbox
    x0, y0 = lonlat_to_tile(w, n, z)   # NW corner -> smallest x,y
    x1, y1 = lonlat_to_tile(e, s, z)
    return [tile_to_quadkey(x, y, z) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def quadkey_to_tile(qk: str) -> tuple[int, int, int]:
    x = y = 0
    for ch in qk:
        x <<= 1
        y <<= 1
        d = int(ch)
        x |= d & 1
        y |= (d >> 1) & 1
    return x, y, len(qk)


def tile_url(qk: str, version: str = "v2") -> str:
    """HTTPS URL of the CHM COG for a quadkey."""
    return f"{BUCKET_URL}/{SOURCES[version]['prefix']}{qk}.tif"


_exists_cache: dict[str, bool] = {}


def tile_exists(url: str) -> bool:
    """HEAD request; 404/403 -> tile not in bucket (no data there, e.g. ocean)."""
    if url not in _exists_cache:
        r = requests.head(url, timeout=30)
        if r.status_code == 200:
            _exists_cache[url] = True
        elif r.status_code in (403, 404):
            _exists_cache[url] = False
        else:
            r.raise_for_status()
    return _exists_cache[url]


# --------------------------------------------------------------------------------------
# CHM container + window reader
# --------------------------------------------------------------------------------------
@dataclass
class Chm:
    """Canopy height window on the global zoom-17 web-mercator pixel grid.

    data       uint8 metres, 255 = nodata (missing tile / cloud mask / internal mask), 2-D (rows, cols)
    col0,row0  global zoom-17 pixel index of data[0, 0]  (x = col*RES - WM_HALF, y = WM_HALF - row*RES)
    bbox       requested lon/lat bbox (W, S, E, N); the array covers it plus < 1 px margin
    """
    data: np.ndarray
    col0: int
    row0: int
    bbox: tuple
    version: str
    tiles: list = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def transform(self) -> Affine:
        """Affine transform of the array in EPSG:3857."""
        return Affine(RES, 0, self.col0 * RES - WM_HALF, 0, -RES, WM_HALF - self.row0 * RES)

    crs = "EPSG:3857"

    @property
    def lonlat_center(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return 0.5 * (w + e), 0.5 * (s + n)

    def ground_pixel_m(self, row: float | None = None) -> float:
        """True ground size (m) of one pixel at the given (local) row; default: window centre."""
        if row is None:
            row = self.data.shape[0] / 2
        y = WM_HALF - (self.row0 + row) * RES
        lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
        return RES * math.cos(math.radians(lat))

    def height_m(self) -> np.ndarray:
        """float32 metres with NaN for nodata (copy)."""
        h = self.data.astype(np.float32)
        h[self.data == NODATA_U8] = np.nan
        return h

    def pix_to_lonlat(self, rows, cols):
        """(row, col) pixel *centres* (may be fractional, local to the array) -> lon, lat."""
        x = (self.col0 + np.asarray(cols) + 0.5) * RES - WM_HALF
        y = WM_HALF - (self.row0 + np.asarray(rows) + 0.5) * RES
        return _T_M2LL.transform(x, y)

    def save_geotiff(self, path: str):
        """Write the window as a small EPSG:3857 GeoTIFF (uint8, nodata=255, deflate)."""
        with rasterio.open(path, "w", driver="GTiff", width=self.data.shape[1], height=self.data.shape[0],
                           count=1, dtype="uint8", crs="EPSG:3857", transform=self.transform, nodata=NODATA_U8,
                           compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
            dst.write(self.data, 1)


def _pixel_window_for_bbox(bbox):
    """Global zoom-17 pixel range [c0, c1) x [r0, r1) covering the lon/lat bbox (outward rounding)."""
    w, s, e, n = bbox
    x0, y0 = _T_LL2M.transform(w, s)
    x1, y1 = _T_LL2M.transform(e, n)
    c0 = int(math.floor((x0 + WM_HALF) / RES))
    c1 = int(math.ceil((x1 + WM_HALF) / RES))
    r0 = int(math.floor((WM_HALF - y1) / RES))
    r1 = int(math.ceil((WM_HALF - y0) / RES))
    return c0, c1, r0, r1


def read_chm(bbox, version: str = "v2", use_cloud_mask: bool = True, verbose: bool = False) -> Chm:
    """Read canopy height (metres) for ``bbox`` = (W, S, E, N) lon/lat from the Meta/WRI COGs.

    Only the needed rows/tiles are fetched via HTTP range requests (/vsicurl).  The window may
    span several tiles; they are mosaicked on the common zoom-17 pixel grid.  Missing tiles
    (ocean etc.) stay 255.

    version        "v2" (CHMv2, 2026, DINOv3 depth model; recommended: taller/less biased canopy) or "v1" (2024).
                   Both are predicted from the same ~2016-2020 Maxar/Vantor imagery (Ames: 2017-04 .. 2018-08).
    use_cloud_mask v1 only: also read ``msk/<qk>.tif.msk`` and set flagged pixels to 255.  GOTCHA: the polarity
                   in the files is the OPPOSITE of the bucket readme: 255 = flagged (cloud/no data, CHM is 0 there),
                   0 = valid (verified: 99.99 % of Ames is 0 and an all-255 Arctic tile has CHM == 0).  Never put the
                   .msk next to the .tif (GDAL would auto-apply it with the inverse meaning).  v2 uses the COG's
                   internal mask instead.

    GOTCHAS: values are integer metres (1 m quantisation); adjacent tiles are predicted independently (different
    imagery dates) so there is a visible step along tile seams (mean |dh| between the two rows at the seam 1.5 m
    (v2) / 0.6 m (v1) vs ~0.15 m normally); v2 tiles are 29-39 km, v1 tiles 58-78 km apart.  Windows near the
    window edge lose neighbours (truncated crowns) so read with a small margin and crop trees afterwards.

    Cost (4.1 x 2.9 km window): v1 ~13 MB (each 1-pixel-high strip holds the whole 65536 px row, so
    cost ~ number of rows, not width); v2 ~0.7 MB (512 px tiles, only touched tiles).
    """
    src = SOURCES[version]
    z = src["zoom"]
    tp = 2 ** (PIX_ZOOM + 8 - z)  # tile size in zoom-17 pixels (65536 for z9, 32768 for z10)
    t0 = time.time()
    c0, c1, r0, r1 = _pixel_window_for_bbox(bbox)
    out = np.full((r1 - r0, c1 - c0), NODATA_U8, np.uint8)
    used = []
    with rasterio.Env(**GDAL_ENV):
        for ty in range(r0 // tp, (r1 - 1) // tp + 1):
            for tx in range(c0 // tp, (c1 - 1) // tp + 1):
                qk = tile_to_quadkey(tx, ty, z)
                url = tile_url(qk, version)
                if not tile_exists(url):
                    if verbose:
                        print(f"  tile {qk}: not in bucket (no data)")
                    continue
                # overlap of window with this tile, in global px and in tile-local px
                gc0, gc1 = max(c0, tx * tp), min(c1, (tx + 1) * tp)
                gr0, gr1 = max(r0, ty * tp), min(r1, (ty + 1) * tp)
                win = Window(gc0 - tx * tp, gr0 - ty * tp, gc1 - gc0, gr1 - gr0)
                with rasterio.open("/vsicurl/" + url) as ds:
                    # sanity: tile grid must coincide with the global zoom-17 grid
                    ex = tx * tp * RES - WM_HALF
                    assert abs(ds.transform.c - ex) < 0.01, (ds.transform.c, ex)
                    a = ds.read(1, window=win)
                    if version == "v2":
                        a[ds.read_masks(1, window=win) == 0] = NODATA_U8
                if version == "v1" and use_cloud_mask and src["mask_prefix"]:
                    murl = f"{BUCKET_URL}/{src['mask_prefix']}{qk}.tif.msk"
                    if tile_exists(murl):
                        with rasterio.open("/vsicurl/" + murl) as ms:   # no georef, same pixel grid
                            m = ms.read(1, window=win)
                        a[m != 0] = NODATA_U8      # 255 = flagged (see docstring)
                out[gr0 - r0:gr1 - r0, gc0 - c0:gc1 - c0] = a
                used.append(qk)
                if verbose:
                    print(f"  tile {qk}: read {a.shape}")
    return Chm(out, c0, r0, tuple(bbox), version, used, time.time() - t0)


# --------------------------------------------------------------------------------------
# imagery vintage
# --------------------------------------------------------------------------------------
def acquisition_info(bbox, version: str = "v1", cache_dir: str | None = None) -> dict:
    """Imagery acquisition dates within bbox.

    v1: cheap -- window read of the global ``CHM_acquisition_date.tif`` (EPSG:4326, 0.01 deg, float32,
        value = (year - 2000) as a decimal fraction, 0 = none).  Returns {"2018.67": npixels, ...}.
    v2: per-tile ``metadata/<qk>.geojson`` (polygons with property ``acq_date``, 14-45 MB per tile!).
        Downloaded once into ``cache_dir`` and intersected with the bbox using shapely.
        Returns {"2020-03-06": area_fraction_of_bbox, ...}.
    """
    src = SOURCES[version]
    if version == "v1":
        w, s, e, n = bbox
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open("/vsicurl/" + f"{BUCKET_URL}/{src['date_raster']}") as ds:
                fw = rasterio.windows.from_bounds(w, s, e, n, ds.transform)
                c0, r0 = int(math.floor(fw.col_off)), int(math.floor(fw.row_off))
                c1, r1 = int(math.ceil(fw.col_off + fw.width)), int(math.ceil(fw.row_off + fw.height))
                a = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        vals, cnt = np.unique(np.round(a[a > 0] + 2000.0, 2), return_counts=True)
        return {f"{v:.2f}": int(c) for v, c in zip(vals, cnt)}
    # v2
    from shapely.geometry import shape, box
    cache = Path(cache_dir or ".") / "meta_cache"
    cache.mkdir(parents=True, exist_ok=True)
    bb = box(*bbox)
    acc: dict[str, float] = {}
    for qk in quadkeys_for_bbox(bbox, "v2"):
        p = cache / f"v2_{qk}.geojson"
        if not p.exists():
            r = requests.get(f"{BUCKET_URL}/{src['meta_prefix']}{qk}.geojson", timeout=300)
            if r.status_code != 200:
                continue
            p.write_bytes(r.content)
        for f in json.loads(p.read_bytes())["features"]:
            g = shape(f["geometry"])
            if g.intersects(bb):
                a = g.intersection(bb).area / bb.area
                d = f["properties"].get("acq_date", "?")
                acc[d] = acc.get(d, 0.0) + a
    return dict(sorted(acc.items()))


# --------------------------------------------------------------------------------------
# tree detection
# --------------------------------------------------------------------------------------
TREE_DTYPE = np.dtype([("lon", "f8"), ("lat", "f8"), ("x_m", "f8"), ("y_m", "f8"),
                       ("height_m", "f4"), ("crown_diam_m", "f4")])


def crown_diameter_allometry(h):
    """Typical temperate broadleaf/conifer crown diameter (m) vs height (m) used to size the local-max window
    (rough: 3 m -> 2.4, 10 m -> 5.2, 20 m -> 9.2, 30 m -> 13)."""
    return 1.2 + 0.4 * np.asarray(h, dtype=np.float32)


def _detect_block(h, g, min_height, sigma_m, window_frac, crown_frac, max_crown_d):
    """Tree tops + crowns in one float32 block ``h`` (NaN already replaced by 0).  g = ground px size (m).
    Returns peaks (row, col float arrays), tree heights, crown diameters."""
    sig = max(sigma_m / g, 0.5)
    hs = ndi.gaussian_filter(h, sig, mode="nearest")
    thr = 0.8 * min_height
    # --- height-dependent window: bigger/taller tree -> bigger exclusion window --------------
    win_m = window_frac * crown_diameter_allometry(hs)
    win_px = np.maximum(3, 2 * np.round(win_m / g / 2).astype(np.int32) + 1)   # odd, >= 3
    win_px[hs < thr] = 0
    peak = np.zeros(hs.shape, bool)
    for w in np.unique(win_px[win_px > 0]):
        mf = ndi.maximum_filter(hs, size=int(w), mode="nearest")
        peak |= (win_px == w) & (hs >= mf)
    if not peak.any():
        return (np.empty(0), np.empty(0), np.empty(0, np.float32), np.empty(0, np.float32))
    # merge plateau ties (8-connected) into one peak
    lab, n = ndi.label(peak, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    com = np.array(ndi.center_of_mass(hs, lab, idx)).reshape(-1, 2)
    pr = np.clip(np.round(com[:, 0]).astype(int), 0, h.shape[0] - 1)
    pc = np.clip(np.round(com[:, 1]).astype(int), 0, h.shape[1] - 1)
    # tree height = max of raw CHM in 3x3 around the top (smoothing lowers narrow peaks)
    hmax3 = ndi.maximum_filter(h, size=3, mode="nearest")
    th = hmax3[pr, pc].astype(np.float32)
    keep = th >= min_height
    pr, pc, th, com = pr[keep], pc[keep], th[keep], com[keep]
    if len(th) == 0:
        return (np.empty(0), np.empty(0), th, th)
    # --- crowns: marker-controlled watershed on inverted smoothed CHM ---------------------------
    markers = np.zeros(hs.shape, np.int32)
    markers[pr, pc] = np.arange(1, len(th) + 1, dtype=np.int32)
    inv = (65535 - np.clip(hs * 256.0, 0, 65535)).astype(np.uint16)
    ws = ndi.watershed_ift(inv, markers)
    # crown = basin pixels higher than crown_frac * treetop height and within max radius of the top
    lbl = ws
    top_h = np.concatenate([[1e9], th])[lbl]
    rr, cc = np.nonzero((hs >= crown_frac * top_h) & (lbl > 0))
    l = lbl[rr, cc]
    d = np.hypot((rr - pr[l - 1]).astype(np.float32), (cc - pc[l - 1]).astype(np.float32)) * g
    rmax = 0.5 * np.minimum(max_crown_d, 2.0 * crown_diameter_allometry(th[l - 1]))
    ok = d <= rmax
    area = np.bincount(l[ok], minlength=len(th) + 1)[1:] * (g * g)
    cd = (2.0 * np.sqrt(area / np.pi)).astype(np.float32)
    cd = np.maximum(cd, np.float32(g))   # at least one pixel
    return com[:, 0], com[:, 1], th, cd


def detect_trees(chm: Chm, min_height: float = 3.0, sigma_m: float = 1.0, window_frac: float = 0.55,
                 crown_frac: float = 0.5, max_crown_d: float = 25.0, block: int = 2048,
                 origin_lonlat: tuple | None = None, verbose: bool = False) -> np.ndarray:
    """Individual tree tops from a canopy-height window.

    Algorithm (per ~2048 px block with 40 px halo, so any bbox size works in bounded memory):
      1. nodata -> 0, Gaussian smoothing (sigma ``sigma_m`` metres, converted to px with the true ground pixel size);
      2. local maxima of the smoothed CHM using a *height-dependent square window*
         (side = ``window_frac`` * crown_diameter_allometry(height), >= 3 px): tall trees suppress neighbours farther away;
      3. tree height = max raw CHM in 3x3 around the top; trees < ``min_height`` dropped;
      4. crowns via marker-controlled watershed (scipy.ndimage.watershed_ift); crown = basin pixels with
         height >= ``crown_frac`` * treetop height and within the allometric max radius; crown_diam = 2*sqrt(A/pi).

    Returns structured array TREE_DTYPE: lon, lat (WGS84), x_m/y_m (east/north metres in a local azimuthal-
    equidistant frame centred on ``origin_lonlat`` = bbox centre by default), height_m, crown_diam_m.
    Note a 1.19 m (mercator) pixel is only ~0.89 m on the ground at 42 N; this is handled via the
    true ground pixel size.  Trees closer than ~1.5 px are not resolvable; the CHM is a model output
    (not a lidar): tops of adjacent crowns are often merged, so treat counts as a lower bound.
    """
    data = chm.data
    R, C = data.shape
    halo = 40
    lons, lats, hts, cds = [], [], [], []
    for r0 in range(0, R, block):
        for c0 in range(0, C, block):
            r1, c1 = min(r0 + block, R), min(c0 + block, C)
            ra, ca = max(r0 - halo, 0), max(c0 - halo, 0)
            rb, cb = min(r1 + halo, R), min(c1 + halo, C)
            sub = data[ra:rb, ca:cb]
            if not (sub != NODATA_U8).any() or sub.max() < min_height * 0.8 or (sub == NODATA_U8).all():
                continue
            h = sub.astype(np.float32)
            h[sub == NODATA_U8] = 0.0
            g = chm.ground_pixel_m((ra + rb) / 2)
            pr, pc, th, cd = _detect_block(h, g, min_height, sigma_m, window_frac, crown_frac, max_crown_d)
            if len(th) == 0:
                continue
            gr, gc = pr + ra, pc + ca
            inside = (gr >= r0 - 0.5) & (gr < r1 - 0.5) & (gc >= c0 - 0.5) & (gc < c1 - 0.5)   # own the core only
            lon, lat = chm.pix_to_lonlat(gr[inside], gc[inside])
            lons.append(lon); lats.append(lat); hts.append(th[inside]); cds.append(cd[inside])
            if verbose:
                print(f"  block r{r0} c{c0}: {inside.sum()} trees")
    out = np.zeros(sum(len(x) for x in lons), TREE_DTYPE)
    if len(out) == 0:
        return out
    out["lon"] = np.concatenate(lons)
    out["lat"] = np.concatenate(lats)
    out["height_m"] = np.concatenate(hts)
    out["crown_diam_m"] = np.concatenate(cds)
    lon0, lat0 = origin_lonlat or chm.lonlat_center
    aeqd = Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84", units="m")
    out["x_m"], out["y_m"] = aeqd(out["lon"], out["lat"])
    return out


def trees_to_csv(trees: np.ndarray, path: str):
    """Write trees as CSV (lon,lat,x_m,y_m,height_m,crown_diam_m)."""
    hdr = ",".join(trees.dtype.names)
    np.savetxt(path, np.column_stack([trees[n].astype(np.float64) for n in trees.dtype.names]),
               delimiter=",", header=hdr, comments="", fmt=["%.7f", "%.7f", "%.2f", "%.2f", "%.1f", "%.1f"])


# --------------------------------------------------------------------------------------
# optional: ESA WorldCover cross-check (public S3 bucket, eu-central-1)
# --------------------------------------------------------------------------------------
def worldcover_class_areas(bbox, year: int = 2021) -> dict:
    """Class areas (km2, ground-corrected) inside bbox from ESA WorldCover 10 m (v200/2021 or v100/2020), read
    window-wise from the public COGs (3x3 deg tiles, s3://esa-worldcover, eu-central-1).
    Class codes: 10 tree cover, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare, 80 water, 90 wetland ..."""
    w, s, e, n = bbox
    ver = {2021: "v200", 2020: "v100"}[year]
    out: dict[int, float] = {}
    with rasterio.Env(**GDAL_ENV):
        for lat0 in range(int(math.floor(s / 3.0) * 3), int(math.floor((n - 1e-9) / 3.0) * 3) + 1, 3):
            for lon0 in range(int(math.floor(w / 3.0) * 3), int(math.floor((e - 1e-9) / 3.0) * 3) + 1, 3):
                name = (f"ESA_WorldCover_10m_{year}_{ver}_{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
                        f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}_Map.tif")
                url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{ver}/{year}/map/{name}"
                bw, bs, be, bn = max(w, lon0), max(s, lat0), min(e, lon0 + 3), min(n, lat0 + 3)
                try:
                    with rasterio.open("/vsicurl/" + url) as ds:
                        win = rasterio.windows.from_bounds(bw, bs, be, bn, ds.transform).round_offsets().round_lengths()
                        a = ds.read(1, window=win)
                except rasterio.errors.RasterioIOError:
                    continue   # no tile (ocean)
                # pixel is 1/12000 deg: ground area per pixel varies with latitude
                rows = np.arange(a.shape[0])
                lat_row = bn - (rows + 0.5) / 12000.0
                px_km2 = (111.32 * np.cos(np.radians(lat_row)) / 12000.0) * (110.57 / 12000.0)
                for v in np.unique(a):
                    out[int(v)] = out.get(int(v), 0.0) + float((((a == v).sum(axis=1)) * px_km2).sum())
    return out


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SOURCES else "v2"
    nums = [float(a) for a in sys.argv[1:] if a not in SOURCES]
    test_bbox = tuple(nums) if len(nums) == 4 else (-93.715, 42.012, -93.665, 42.038)  # W Ames, Iowa
    outdir = Path(__file__).resolve().parent / "out"
    outdir.mkdir(exist_ok=True)

    print(f"== Meta/WRI canopy height {ver}  bbox={test_bbox}")
    print("quadkeys:", quadkeys_for_bbox(test_bbox, ver))
    t0 = time.time()
    chm = read_chm(test_bbox, ver, verbose=True)
    valid = chm.data != NODATA_U8
    print(f"CHM window {chm.data.shape} px (rows x cols), pixel {RES:.4f} m mercator = {chm.ground_pixel_m():.3f} m ground,"
          f" read in {chm.elapsed_s:.1f} s, valid {valid.mean()*100:.2f}%")
    h = chm.data[valid]
    print(f"height: max {h.max()} m, canopy >=3 m: {(h >= 3).mean()*100:.1f}% of pixels, >=10 m: {(h >= 10).mean()*100:.1f}%")
    chm.save_geotiff(str(outdir / f"chm_{ver}.tif"))

    t1 = time.time()
    trees = detect_trees(chm, min_height=3.0)
    dt = time.time() - t1
    area_km2 = (chm.data.shape[0] * chm.data.shape[1]) * chm.ground_pixel_m() ** 2 / 1e6
    print(f"trees: {len(trees)} in {area_km2:.2f} km2 window -> {len(trees)/area_km2/100:.1f} trees/ha ({dt:.1f} s)")
    if len(trees):
        q = np.percentile(trees["height_m"], [5, 25, 50, 75, 95])
        print("tree height p5/25/50/75/95 [m]:", np.round(q, 1), " max", trees["height_m"].max())
        print("crown diam p5/50/95 [m]:", np.round(np.percentile(trees["crown_diam_m"], [5, 50, 95]), 1))
        print("height histogram (3-6,6-9,9-12,12-15,15-20,20+):",
              np.histogram(trees["height_m"], [3, 6, 9, 12, 15, 20, 99])[0].tolist())
        print("first trees (lon, lat, x_m, y_m, h, crown):")
        for t in trees[:3]:
            print("  ", tuple(np.round(list(t), 3)))
        trees_to_csv(trees, str(outdir / f"trees_{ver}.csv"))
    try:
        ai = acquisition_info(test_bbox, "v1") if ver == "v1" else acquisition_info(test_bbox, "v2", cache_dir=str(outdir))
        print("imagery vintage:", ai)
    except Exception as ex:  # pragma: no cover
        print("acquisition_info failed:", repr(ex))
    try:
        wc = worldcover_class_areas(test_bbox)
        tot = sum(wc.values())
        tree_km2 = wc.get(10, 0.0)
        chm_tree_km2 = (h >= 3).sum() * chm.ground_pixel_m() ** 2 / 1e6
        print(f"WorldCover 2021: total {tot:.2f} km2, tree {tree_km2:.2f} km2 ({tree_km2/tot*100:.1f}%), "
              f"built-up {wc.get(50,0):.2f}, crop {wc.get(40,0):.2f}, grass {wc.get(30,0):.2f}; "
              f"CHM>=3 m {chm_tree_km2:.2f} km2 ({chm_tree_km2/tot*100:.1f}%)")
    except Exception as ex:  # pragma: no cover
        print("worldcover failed:", repr(ex))
    print(f"total wall time {time.time()-t0:.1f} s")
