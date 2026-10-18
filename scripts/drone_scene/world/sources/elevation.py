"""Terrain elevation recipe for the large-scene Isaac Sim pre-processing step.

Returns a regular height grid (metres) in a local metric CRS for a lon/lat
bounding box, from open data on public S3 buckets (no API keys, no TNM API):

  priority  source id    data                                     res        kind
  --------  -----------  ---------------------------------------  ---------  ------------------
  1         "3dep_1m"    USGS 3DEP 1 m DEM (lidar, per-project)   1 m        bare earth (DTM)
  2         "3dep_13"    USGS 3DEP 1/3 arc-second seamless DEM    ~10 m      bare earth (DTM)
  3         "cop30"      Copernicus DEM GLO-30 (copernicus-dem-30m) 1 arcsec   surface (DSM)
  4         "terrarium"  AWS Terrain Tiles (elevation-tiles-prod)  z15 ~3.5 m* mixed, US = 3DEP
                         (* nominal pixel size at 42N, true detail ~10 m in the US)

Only plain Python + numpy + rasterio + pyproj + shapely + scipy + requests +
Pillow are needed; `apsw` (pip install apsw) is needed ONLY to locate 1 m tiles
(see find_3dep_1m_tiles); without it the recipe silently skips "3dep_1m".

Main entry point: get_height_grid(bbox, spacing).

    res = get_height_grid((-93.715, 42.012, -93.665, 42.038), spacing=2.0)
    res["height"]    # (ny, nx) float32 metres, row 0 = NORTH edge, x increases with column
    res["transform"] # rasterio Affine of the grid (pixel = cell, samples are cell centres)
    res["crs"]       # local transverse-mercator CRS, origin at bbox centre, units metres
    res["x"], res["y"]  # cell-centre coordinates in that CRS (east, north), metres

Run `python recipe.py` for a self-test on the Ames, Iowa test area.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import math
import os
import re
import tempfile
import time

import numpy as np
import rasterio
import requests
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

log = logging.getLogger("world_elevation")

# --------------------------------------------------------------------------- constants
TNM_BUCKET = "https://prd-tnm.s3.amazonaws.com"
FESM_URL = TNM_BUCKET + "/StagedProducts/Elevation/1m/FullExtentSpatialMetadata/FESM_1m.gpkg"
PROJECTS_PREFIX = "StagedProducts/Elevation/1m/Projects"
DEM13_PREFIX = "StagedProducts/Elevation/13/TIFF/current"
COP30_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
TERRAIN_BUCKET = "https://elevation-tiles-prod.s3.amazonaws.com"

USE_OVERVIEWS = True  # read the matching internal overview of tiles when the grid is coarser than the source

DEFAULT_SOURCES = ("3dep_1m", "3dep_13", "cop30", "terrarium")
SOURCE_INFO = {
    "3dep_1m": dict(vertical="NAVD88 (GEOID18/12B, project specific), metres", surface="bare earth DTM, hydro-flattened"),
    "3dep_13": dict(vertical="NAVD88 (CONUS), metres", surface="bare earth DTM"),
    "cop30": dict(vertical="EGM2008 orthometric, metres", surface="DSM (includes trees/buildings)"),
    "terrarium": dict(vertical="NAVD88 in US (3DEP), EGM96-ish elsewhere, metres", surface="mixed; US = 3DEP bare earth"),
}

GDAL_OPTS = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="1",
    VSI_CACHE="TRUE",
    GDAL_CACHEMAX=512,
    CPL_DEBUG="ON",  # only so that we can count bytes (see _ByteCounter); output is not printed
)


# --------------------------------------------------------------------------- byte counting
class _ByteCounter(logging.Handler):
    """Counts bytes GDAL downloads over /vsicurl (parses 'VSICURL: Downloading a-b')."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.gdal = 0
        self.http = 0  # bytes fetched by this module with requests / apsw

    def emit(self, record):
        m = re.search(r"Downloading (\d+)-(\d+) ", record.getMessage())
        if m:
            self.gdal += int(m.group(2)) - int(m.group(1)) + 1

    @property
    def total(self):
        return self.gdal + self.http


BYTES = _ByteCounter()
_err_logger = logging.getLogger("rasterio._err")
_err_logger.addHandler(BYTES)
_err_logger.setLevel(logging.DEBUG)
_err_logger.propagate = False  # keep GDAL debug chatter out of the user's logging output


def gdal_env():
    """rasterio.Env with the settings that make /vsicurl reads of COG-ish GeoTIFFs fast."""
    return rasterio.Env(**GDAL_OPTS)


_session = None


def _http() -> requests.Session:
    global _session
    if _session is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        _session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504), allowed_methods=None)
        _session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return _session


def _get(url, **kw) -> requests.Response:
    r = _http().get(url, timeout=kw.pop("timeout", 60), **kw)
    BYTES.http += len(r.content)
    return r


def _cache_dir(cache_dir=None) -> str:
    d = cache_dir or os.environ.get("WORLD_ELEVATION_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "world_elevation")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.join(tempfile.gettempdir(), "world_elevation_cache")
        os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- local CRS / grid
def local_crs(bbox, kind="local_tm") -> CRS:
    """Metric CRS for the scene.

    kind="local_tm": transverse Mercator, k=1, origin (0,0) at the bbox centre, GRS80
        (east/north metres from the scene centre; scale error < 1e-5 within 30 km).
    kind="utm": WGS84 UTM zone of the bbox centre (EPSG:326xx / 327xx).
    An explicit rasterio/pyproj CRS object or 'EPSG:xxxx' string is passed through.
    """
    w, s, e, n = bbox
    lon0, lat0 = (w + e) / 2, (s + n) / 2
    if kind == "local_tm":
        return CRS.from_proj4(f"+proj=tmerc +lat_0={lat0:.8f} +lon_0={lon0:.8f} +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
    if kind == "utm":
        zone = int((lon0 + 180) // 6) + 1
        return CRS.from_epsg((32600 if lat0 >= 0 else 32700) + zone)
    return CRS.from_user_input(kind)


def local_bounds(bbox, crs) -> tuple[float, float, float, float]:
    """Axis-aligned envelope (xmin, ymin, xmax, ymax) in `crs` of a lon/lat bbox (edges densified)."""
    w, s, e, n = bbox
    return transform_bounds("EPSG:4326", crs, w, s, e, n, densify_pts=41)


def make_grid(bounds, spacing):
    """(transform, width, height) of a north-up grid of `spacing`-metre cells covering `bounds`.

    xmin/ymax are snapped outwards to a multiple of the spacing; samples are cell centres."""
    xmin, ymin, xmax, ymax = bounds
    x0 = math.floor(xmin / spacing) * spacing
    y1 = math.ceil(ymax / spacing) * spacing
    nx = int(math.ceil((xmax - x0) / spacing))
    ny = int(math.ceil((y1 - ymin) / spacing))
    return from_origin(x0, y1, spacing, spacing), nx, ny


# --------------------------------------------------------------------------- 3DEP 1 m: locating tiles
_VFS = None


def _fesm_connect():
    """Open FESM_1m.gpkg (1.9 GB) over HTTP range requests through an apsw VFS (only ~2-3 MB are read)."""
    import apsw  # pip install apsw

    global _VFS
    chunk = 256 * 1024

    class HttpFile(apsw.VFSFile):
        def __init__(self, url):
            r = _http().head(url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            self.url, self.size, self.cache = url, int(r.headers["Content-Length"]), {}

        def _chunk(self, i):
            if i not in self.cache:
                a = i * chunk
                b = min(self.size, a + chunk) - 1
                r = _http().get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
                r.raise_for_status()
                BYTES.http += len(r.content)
                self.cache[i] = r.content
            return self.cache[i]

        def xRead(self, amount, offset):
            out, pos, end = bytearray(), offset, min(offset + amount, self.size)
            while pos < end:
                c = self._chunk(pos // chunk)
                o = pos % chunk
                take = min(len(c) - o, end - pos)
                out += c[o : o + take]
                pos += take
            out += b"\x00" * (amount - len(out))
            return bytes(out)

        def xFileSize(self):
            return self.size

        def xClose(self):
            pass

        def xLock(self, level):
            pass

        def xUnlock(self, level):
            pass

        def xCheckReservedLock(self):
            return False

        def xFileControl(self, op, ptr):
            return False

        def xSectorSize(self):
            return 4096

        def xDeviceCharacteristics(self):
            return apsw.SQLITE_IOCAP_IMMUTABLE

    class HttpVFS(apsw.VFS):
        def __init__(self):
            super().__init__("fesm_http", base="")

        def xOpen(self, name, flags):
            return HttpFile(name.filename())

        def xFullPathname(self, name):
            return name

        def xAccess(self, pathname, flags):
            # the bucket holds a stale 'FESM_1m.gpkg-journal': never let SQLite see it
            return not pathname.endswith(("-journal", "-wal", "-shm"))

    if _VFS is None:
        _VFS = HttpVFS()
    return apsw.Connection(FESM_URL, flags=apsw.SQLITE_OPEN_READONLY, vfs="fesm_http")


def fesm_projects(bbox):
    """3DEP 1 m lidar projects whose footprint really intersects `bbox`, newest first.

    Reads FESM_1m.gpkg (EPSG:4269 multipolygons, one row per project: fields project,
    project_id, pub_date, product_link, metadata_link) via its R-tree + shapely.
    Returns list of dicts {project, project_id, pub_date, product_link}."""
    import shapely
    from shapely.geometry import box

    w, s, e, n = bbox
    con = _fesm_connect()
    out = []
    q = (
        "select f.project, f.project_id, f.pub_date, f.product_link, f.geometry "
        "from fesm_1m f join rtree_fesm_1m_geometry r on f.fid = r.id "
        "where r.maxx >= ? and r.minx <= ? and r.maxy >= ? and r.miny <= ?"
    )
    bb = box(w, s, e, n)
    for project, pid, pub, link, blob in con.cursor().execute(q, (w, e, s, n)):
        blob = bytes(blob)
        env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]  # GeoPackage binary header
        geom = shapely.from_wkb(blob[8 + env_len :])
        if geom.intersects(bb):
            out.append(dict(project=project, project_id=pid, pub_date=pub, product_link=link, covered=geom.intersection(bb).area / bb.area))
    con.close()
    out.sort(key=lambda d: d["pub_date"] or "", reverse=True)
    return out


_TILE_RE = re.compile(r"USGS_(?:1M|1m|one_meter)_(?:(\d{1,2})_)?x(\d+)y(\d+)_", re.I)


def _project_tile_index(project, cache_dir=None, max_age_days=14):
    """{(utm_zone|None, x, y): url} for all 10 km GeoTIFF tiles of a 3DEP 1 m project.

    Uses '<project>/0_file_download_links.txt' (cached on disk); falls back to a paged S3
    listing of '<project>/TIFF/' when that file is missing (it is for a few old projects)."""
    path = os.path.join(_cache_dir(cache_dir), f"links_{project}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400:
        with open(path) as f:
            urls = json.load(f)
    else:
        r = _get(f"{TNM_BUCKET}/{PROJECTS_PREFIX}/{project}/0_file_download_links.txt")
        if r.status_code == 200:
            urls = [u for u in r.text.split() if u.lower().endswith(".tif")]
        else:
            urls, token = [], None
            while True:
                params = {"list-type": "2", "prefix": f"{PROJECTS_PREFIX}/{project}/TIFF/"}
                if token:
                    params["continuation-token"] = token
                x = _get(TNM_BUCKET + "/", params=params).text
                urls += [f"{TNM_BUCKET}/{k}" for k in re.findall(r"<Key>([^<]+\.tif)</Key>", x)]
                m = re.search(r"<NextContinuationToken>([^<]+)<", x)
                token = m.group(1) if m else None
                if not token:
                    break
        with open(path, "w") as f:
            json.dump(urls, f)
    idx = {}
    for u in urls:
        m = _TILE_RE.search(u.rsplit("/", 1)[-1])
        if m:
            idx[(int(m.group(1)) if m.group(1) else None, int(m.group(2)), int(m.group(3)))] = u
    return idx


def find_3dep_1m_tiles(bbox, cache_dir=None, pad_m=30.0):
    """URLs of the 3DEP 1 m GeoTIFF tiles covering `bbox`, grouped per project, newest project first.

    TNM API (tnmaccess.nationalmap.gov) is NOT needed. Steps:
      1. FESM_1m.gpkg in the bucket (project footprints, range-read via apsw) -> projects.
      2. Tile naming: USGS_1M_<utmzone>_x<E/10000>y<ceil(N/10000)>_<project>.tif, i.e.
         x = left easting/10 km and y = TOP northing/10 km in NAD83 UTM (older projects:
         USGS_one_meter_x..y.. / USGS_1m_x..y.. without zone). Tiles are 10012 x 10012 px
         (6 px buffer each side). Matching cells are looked up in the project's link list.
    Returns list of {project, pub_date, urls:[...]} (empty if apsw missing / no coverage)."""
    w, s, e, n = bbox
    zc = int(((w + e) / 2 + 180) // 6) + 1
    zones = sorted({int((lon + 180) // 6) + 1 for lon in (w, e)}, key=lambda z: abs(z - zc))  # centre zone first
    projs = fesm_projects(bbox)
    result = []
    for p in projs:
        idx = _project_tile_index(p["project"], cache_dir)
        urls = []
        for z in zones:
            t = Transformer.from_crs(4326, 32600 + z, always_xy=True)
            xs, ys = t.transform([w, e, w, e], [s, s, n, n])
            x0, x1 = min(xs) - pad_m, max(xs) + pad_m
            y0, y1 = min(ys) - pad_m, max(ys) + pad_m
            for ix in range(int(x0 // 10000), int(x1 // 10000) + 1):
                for iy in range(int(y0 // 10000) + 1, int(y1 // 10000) + 2):
                    for key in ((z, ix, iy), (None, ix, iy)):
                        if key in idx and idx[key] not in urls:
                            urls.append(idx[key])
        if urls:
            result.append(dict(project=p["project"], pub_date=p["pub_date"], urls=urls))
    return result


# --------------------------------------------------------------------------- 3DEP 1/3 arc-second / Copernicus: tile urls
def find_3dep_13_tiles(bbox, pad_deg=0.001):
    """URLs of 3DEP 1/3 arc-second 1x1 degree tiles (named by their NW corner) covering `bbox`.

    s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n43w094/USGS_13_n43w094.tif
    (tile n43w094 covers lat 42..43, lon -94..-93). Existence is checked with one tiny
    S3 listing per tile (the file name is taken from the listing)."""
    w, s, e, n = bbox
    urls = []
    for lat_n in range(math.floor(s - pad_deg) + 1, math.floor(n + pad_deg) + 2):
        for lon_w in range(math.floor(w - pad_deg), math.floor(e + pad_deg) + 1):
            name = f"n{lat_n:02d}{'w' if lon_w < 0 else 'e'}{abs(lon_w):03d}"
            x = _get(TNM_BUCKET + "/", params={"list-type": "2", "prefix": f"{DEM13_PREFIX}/{name}/"}).text
            tifs = [k for k in re.findall(r"<Key>([^<]+\.tif)</Key>", x)]
            if tifs:
                urls.append(f"{TNM_BUCKET}/{tifs[0]}")
    return urls


def find_cop30_tiles(bbox, pad_deg=0.001):
    """URLs of Copernicus GLO-30 COG tiles (1x1 degree, named by SW corner) covering `bbox`.

    https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N42_00_W094_00_DEM/Copernicus_DSM_COG_10_N42_00_W094_00_DEM.tif
    Missing ocean tiles are simply not returned (checked by a tiny range request)."""
    w, s, e, n = bbox
    urls = []
    for lat in range(math.floor(s - pad_deg), math.floor(n + pad_deg) + 1):
        for lon in range(math.floor(w - pad_deg), math.floor(e + pad_deg) + 1):
            nm = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
            u = f"{COP30_BUCKET}/{nm}/{nm}.tif"
            if _http().head(u, timeout=30).status_code == 200:
                urls.append(u)
    return urls


# --------------------------------------------------------------------------- generic raster reader
def _native_res_m(ds) -> float:
    """Approximate ground resolution of a dataset in metres."""
    if ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        return math.sqrt(ds.res[0] * 111320.0 * math.cos(math.radians(lat)) * ds.res[1] * 110574.0)
    return float(ds.res[0])


def read_geotiffs_onto_grid(urls, crs, transform, width, height, spacing, nodata_override=None, resampling=None):
    """Mosaic remote GeoTIFFs onto the destination grid (first valid tile wins).

    For every tile: the overview whose resolution is closest to (but not coarser than) the
    grid spacing is opened (so a 5 m grid from a 1 m DEM reads the 4 m overview only), the
    tile is warped (rasterio.vrt.WarpedVRT) to `crs`/`transform` and only the window it
    overlaps is read. Source nodata is excluded from the resampling. Returns float32
    (height, width) with NaN where no tile has data."""
    out = np.full((height, width), np.nan, dtype=np.float32)
    west, south, east, north = transform.c, transform.f + transform.e * height, transform.c + transform.a * width, transform.f
    for url in urls:
        path = url if url.startswith("/vsi") else "/vsicurl/" + url
        try:
            with rasterio.open(path) as probe:
                res_m = _native_res_m(probe)
                factors = probe.overviews(1)
                ovl = None
                for k, f in enumerate(factors if USE_OVERVIEWS else []):
                    if res_m * f <= spacing * 1.05:
                        ovl = k
                sb = transform_bounds(probe.crs, crs, *probe.bounds, densify_pts=21)
                nod = probe.nodata if nodata_override is None else nodata_override
        except rasterio.errors.RasterioIOError as exc:
            log.warning("cannot open %s: %s", url, exc)
            continue
        if sb[2] < west or sb[0] > east or sb[3] < south or sb[1] > north:
            continue  # tile does not overlap the grid
        # window of the destination grid that this tile can touch (+2 px margin)
        win = from_bounds(max(sb[0], west), max(sb[1], south), min(sb[2], east), min(sb[3], north), transform)
        c0, r0 = max(0, int(math.floor(win.col_off)) - 2), max(0, int(math.floor(win.row_off)) - 2)
        c1, r1 = min(width, int(math.ceil(win.col_off + win.width)) + 2), min(height, int(math.ceil(win.row_off + win.height)) + 2)
        if c1 <= c0 or r1 <= r0:
            continue
        if not np.isnan(out[r0:r1, c0:c1]).any():
            continue  # already fully covered (e.g. same tile delivered in two UTM zones): save the download
        window = Window(c0, r0, c1 - c0, r1 - r0)
        with rasterio.open(path, **({"overview_level": ovl} if ovl is not None else {})) as ds:
            rs = resampling or (Resampling.average if _native_res_m(ds) < spacing * 0.9 else Resampling.bilinear)
            with WarpedVRT(ds, crs=crs, transform=transform, width=width, height=height, resampling=rs, src_nodata=nod, nodata=np.nan, dtype="float32") as vrt:
                a = vrt.read(1, window=window)
        cur = out[r0:r1, c0:c1]
        take = np.isnan(cur) & np.isfinite(a)
        cur[take] = a[take]
    return out


# --------------------------------------------------------------------------- AWS terrain tiles (terrarium)
def read_terrarium_onto_grid(bbox, crs, transform, width, height, spacing, zoom=None, workers=8):
    """AWS Terrain Tiles 'terrarium' PNGs -> grid.

    https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png  (z 0..15, XYZ scheme)
    256x256 RGB, Web-Mercator, height_m = R*256 + G + B/256 - 32768. Zoom is the smallest z (<=15)
    whose pixel is <= `spacing` at this latitude. Tiles are mosaicked in EPSG:3857 and warped."""
    w, s, e, n = bbox
    lat0 = (s + n) / 2
    if zoom is None:
        zoom = int(min(15, max(6, math.ceil(math.log2(156543.03392 * math.cos(math.radians(lat0)) / spacing)))))
    from PIL import Image

    def tile_xy(lon, lat, z):
        k = 2**z
        x = (lon + 180.0) / 360.0 * k
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * k
        return int(math.floor(x)), int(math.floor(y))

    # bbox of the destination grid in lon/lat (the grid envelope is a bit larger than the lon/lat bbox)
    gw, gs, ge, gn = transform_bounds(crs, "EPSG:4326", transform.c, transform.f + transform.e * height, transform.c + transform.a * width, transform.f, densify_pts=41)
    x0, y1 = tile_xy(gw, gs, zoom)
    x1, y0 = tile_xy(ge, gn, zoom)
    x0, y0 = x0 - 1, y0 - 1  # 1-tile margin for the resampling kernel
    x1, y1 = x1 + 1, y1 + 1
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.full((ny * 256, nx * 256), np.nan, dtype=np.float32)

    def fetch(xy):
        x, y = xy
        r = _get(f"{TERRAIN_BUCKET}/terrarium/{zoom}/{x}/{y}.png")
        if r.status_code != 200:
            return xy, None
        rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
        return xy, rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0

    with cf.ThreadPoolExecutor(workers) as ex:
        for (x, y), h in ex.map(fetch, [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]):
            if h is not None:
                mosaic[(y - y0) * 256 : (y - y0 + 1) * 256, (x - x0) * 256 : (x - x0 + 1) * 256] = h
    R = 6378137.0
    px = 2 * math.pi * R / (256 * 2**zoom)
    src_tf = Affine(px, 0, -math.pi * R + x0 * 256 * px, 0, -px, math.pi * R - y0 * 256 * px)
    out = np.full((height, width), np.nan, dtype=np.float32)
    rs = Resampling.average if px * math.cos(math.radians(lat0)) < spacing * 0.9 else Resampling.bilinear
    reproject(mosaic, out, src_transform=src_tf, src_crs="EPSG:3857", src_nodata=np.nan, dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=rs)
    return out


# --------------------------------------------------------------------------- one source -> grid
def read_source(name, bbox, crs, transform, width, height, spacing, cache_dir=None):
    """Read one named source onto the grid. Returns (array float32 NaN=nodata, info dict)."""
    t0, b0 = time.time(), BYTES.total
    info = dict(source=name, **SOURCE_INFO[name])
    if name == "3dep_1m":
        try:
            groups = find_3dep_1m_tiles(bbox, cache_dir)
        except ImportError:
            log.warning("apsw not installed: cannot locate 3DEP 1 m tiles (pip install apsw)")
            groups = []
        arr = np.full((height, width), np.nan, dtype=np.float32)
        info["projects"] = []
        for g in groups:  # newest project first; older projects only fill remaining gaps
            a = read_geotiffs_onto_grid(g["urls"], crs, transform, width, height, spacing)
            fill = np.isnan(arr) & np.isfinite(a)
            arr[fill] = a[fill]
            info["projects"].append(dict(project=g["project"], pub_date=g["pub_date"], tiles=[u.rsplit("/", 1)[-1] for u in g["urls"]], cells=int(fill.sum())))
            if not np.isnan(arr).any():
                break
    elif name == "3dep_13":
        urls = find_3dep_13_tiles(bbox)
        info["tiles"] = [u.rsplit("/", 1)[-1] for u in urls]
        arr = read_geotiffs_onto_grid(urls, crs, transform, width, height, spacing)
    elif name == "cop30":
        urls = find_cop30_tiles(bbox)
        info["tiles"] = [u.rsplit("/", 1)[-1] for u in urls]
        arr = read_geotiffs_onto_grid(urls, crs, transform, width, height, spacing)
    elif name == "terrarium":
        arr = read_terrarium_onto_grid(bbox, crs, transform, width, height, spacing)
    else:
        raise ValueError(name)
    info.update(seconds=round(time.time() - t0, 2), bytes=BYTES.total - b0, valid_fraction=float(np.isfinite(arr).mean()))
    return arr, info


# --------------------------------------------------------------------------- main API
def get_height_grid(bbox, spacing=2.0, crs="local_tm", sources=DEFAULT_SOURCES, min_coverage=0.995, fill_gaps=True, cache_dir=None, verbose=False):
    """Height grid (metres) for a lon/lat bbox on a regular local grid.

    bbox      (west, south, east, north) degrees, WGS84.
    spacing   grid cell size in metres (1.0 = full 3DEP 1 m resolution).
    crs       "local_tm" (default; east/north metres from the bbox centre), "utm", or any CRS.
    sources   priority list. The first source that covers >= min_coverage of the cells is the
              base; if none does, lower-priority sources fill the holes after matching their
              vertical offset to the base (median difference over the overlap). Remaining
              holes (if fill_gaps) are filled by nearest-neighbour interpolation.

    Returns dict: height (ny,nx float32, row 0 = north), transform (Affine), crs (rasterio CRS),
    x (nx,) / y (ny,) cell-centre coordinates, bounds, spacing, origin_lonlat (centre of bbox =
    origin of local_tm), source_map (uint8 index into `source_names`, 255 = interpolated),
    source_names, info (per-source stats incl. bytes/seconds), seconds, bytes."""
    t0, b0 = time.time(), BYTES.total
    ccrs = local_crs(bbox, crs)
    bounds = local_bounds(bbox, ccrs)
    transform, nx, ny = make_grid(bounds, spacing)
    height = np.full((ny, nx), np.nan, dtype=np.float32)
    smap = np.full((ny, nx), 255, dtype=np.uint8)
    infos, names = [], []
    with gdal_env():
        for name in sources:
            try:
                arr, info = read_source(name, bbox, ccrs, transform, nx, ny, spacing, cache_dir)
            except Exception as exc:  # network/tile problems must not kill the fallback chain
                log.warning("source %s failed: %s: %s", name, type(exc).__name__, exc)
                infos.append(dict(source=name, error=f"{type(exc).__name__}: {exc}"))
                continue
            infos.append(info)
            if verbose:
                print(f"  [{name}] valid {info['valid_fraction']*100:.1f}%  {info['seconds']} s  {info['bytes']/1e6:.1f} MB")
            have = np.isfinite(height)
            new = np.isfinite(arr) & ~have
            if not new.any():
                continue
            if have.any():  # match vertical offset to what we already have
                ov = have & np.isfinite(arr)
                if ov.sum() > 100:
                    off = float(np.median(height[ov] - arr[ov]))
                    arr = arr + np.float32(off)
                    info["offset_to_base_m"] = off
            height[new] = arr[new]
            smap[new] = len(names)
            names.append(name)
            if np.isfinite(height).mean() >= min_coverage:
                break
    if fill_gaps and np.isnan(height).any() and np.isfinite(height).any():
        from scipy.ndimage import distance_transform_edt

        bad = np.isnan(height)
        _, (ii, jj) = distance_transform_edt(bad, return_indices=True)
        height[bad] = height[ii[bad], jj[bad]]
    xs = transform.c + (np.arange(nx) + 0.5) * spacing
    ys = transform.f - (np.arange(ny) + 0.5) * spacing
    return dict(
        height=height, transform=transform, crs=ccrs, x=xs, y=ys,
        bounds=(transform.c, transform.f - ny * spacing, transform.c + nx * spacing, transform.f),
        spacing=spacing, origin_lonlat=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        source_map=smap, source_names=names, info=infos, seconds=round(time.time() - t0, 2), bytes=BYTES.total - b0,
    )


def save_geotiff(res, path):
    """Write the height grid as a single-band float32 GeoTIFF (nodata NaN) for inspection in QGIS."""
    h = res["height"]
    with rasterio.open(path, "w", driver="GTiff", height=h.shape[0], width=h.shape[1], count=1, dtype="float32", crs=res["crs"], transform=res["transform"], nodata=np.nan, compress="deflate") as dst:
        dst.write(h, 1)


def hillshade(h, spacing, az=315.0, alt=45.0, zf=2.0):
    """Quick hillshade (uint8) for eyeballing a height grid."""
    gy, gx = np.gradient(h.astype(np.float64) * zf, spacing)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(360 - az + 90), np.radians(alt)
    s = np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect)
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- self-test
TEST_BBOX = (-93.715, 42.012, -93.665, 42.038)  # west edge of Ames, Iowa: ~4.2 km x 2.9 km
LARGE_BBOX = (-93.76, 41.98, -93.60, 42.08)  # ~13 km x 11 km, crosses 1 m tile borders


def _stats(h):
    return f"min {np.nanmin(h):.2f}  max {np.nanmax(h):.2f}  mean {np.nanmean(h):.2f}  relief {np.nanmax(h)-np.nanmin(h):.2f} m"


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), default=TEST_BBOX)
    ap.add_argument("--large", action="store_true", help="use the ~13x11 km LARGE_BBOX instead")
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
    ap.add_argument("--compare", action="store_true", help="also read every source separately and compare them")
    a = ap.parse_args(argv)
    bbox = LARGE_BBOX if a.large else tuple(a.bbox)
    os.makedirs(a.out, exist_ok=True)
    print(f"bbox {bbox} spacing {a.spacing} m")

    if a.compare:
        ccrs = local_crs(bbox)
        tf, nx, ny = make_grid(local_bounds(bbox, ccrs), a.spacing)
        print(f"grid {nx} x {ny} cells = {nx*a.spacing/1000:.2f} x {ny*a.spacing/1000:.2f} km")
        grids = {}
        with gdal_env():
            for name in DEFAULT_SOURCES:
                try:
                    arr, info = read_source(name, bbox, ccrs, tf, nx, ny, a.spacing, a.cache)
                except Exception as exc:
                    print(f"{name:10s} FAILED {type(exc).__name__}: {exc}")
                    continue
                grids[name] = arr
                print(f"{name:10s} valid {info['valid_fraction']*100:5.1f}%  {info['seconds']:6.2f} s  {info['bytes']/1e6:7.2f} MB   {_stats(arr)}")
        ref = grids.get("3dep_1m")
        if ref is not None:
            for name, arr in grids.items():
                if name == "3dep_1m":
                    continue
                m = np.isfinite(ref) & np.isfinite(arr)
                d = arr[m] - ref[m]
                print(f"  {name:10s} - 3dep_1m : bias {d.mean():+.2f} m  rmse {np.sqrt((d**2).mean()):.2f} m  p95|d| {np.percentile(np.abs(d),95):.2f} m  (n={m.sum()})")
        print(f"total downloaded in compare run: {BYTES.total/1e6:.1f} MB")

    res = get_height_grid(bbox, a.spacing, cache_dir=a.cache, verbose=True)
    h = res["height"]
    print(f"\nget_height_grid: grid {h.shape[1]} x {h.shape[0]} cells @ {a.spacing} m, crs local_tm origin lon/lat {res['origin_lonlat']}")
    print("  sources used:", res["source_names"], " | " + _stats(h))
    print(f"  {res['seconds']} s, {res['bytes']/1e6:.1f} MB downloaded, bounds {tuple(round(v,1) for v in res['bounds'])}")
    for i in res["info"]:
        if "projects" in i:
            print("  3dep_1m projects:", [(p['project'], p['pub_date'], p['tiles']) for p in i["projects"]])
    np.save(os.path.join(a.out, "height.npy"), h)
    save_geotiff(res, os.path.join(a.out, "height.tif"))
    from PIL import Image

    Image.fromarray(hillshade(h, a.spacing)).save(os.path.join(a.out, "hillshade.png"))
    print("  wrote", a.out)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
