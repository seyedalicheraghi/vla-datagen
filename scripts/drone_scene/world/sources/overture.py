"""Overture Maps -> bbox reader (GeoParquet on S3, read via HTTP range requests).

Reads ONLY the features inside a lon/lat bounding box from the public Overture
bucket ``overturemaps-us-west-2`` without downloading the (hundreds of MB) files:

1. ``latest_release()``      S3 ListObjectsV2 (delimiter=/) -> newest ``release/<YYYY-MM-DD.N>/``.
2. ``file_index()``          one 16 KB tail range-GET per parquet file -> the file-level
                             ``bbox`` stored in the GeoParquet ``geo`` metadata. Cached as JSON
                             (about 19 MB of requests for all types, 2-4 s with 32 threads).
3. ``read_bbox()``           for the (1-4) files whose bbox overlaps the query:
                               - fetch footer (0.6-1.5 MB),
                               - prune row groups with the statistics of ``bbox.xmin/xmax/ymin/ymax``,
                               - phase 1: fetch only the small bbox column (0.25 MB/row group) and
                                 keep rows whose own bbox intersects the query,
                               - phase 2: fetch the heavy columns (geometry, ...) only for row groups
                                 that still have rows (coalesced, parallel range GETs), decode with pyarrow.
   ``estimate_cost()`` is a dry run (footers only). Row groups are the smallest unit that can be read
   (no parquet page index in these files): a 4x3 km box costs ~90 MB for all 7 layers, 20x20 km ~115 MB.

Gotchas: land_cover is a zoom pyramid (coarse continent-size polygons overlap the fine ones; read_bbox
keeps cartography.max_zoom >= 15 only); transportation/segment lines can be 30 km long (use clip_to_bbox);
there is NO dedicated tree/vegetation type: trees are base/land with subtype='tree' (Point, from OSM);
attribute richness (height / num_floors / roof_shape) varies a lot by region (see README numbers in the report).

Needs only: requests, numpy, pyarrow, shapely (all in the venv). Proxy / CA bundle come from the
standard env vars (HTTPS_PROXY, REQUESTS_CA_BUNDLE) because ``requests`` honours them.

Example::

    from recipe import read_bbox, to_shapely
    tbl = read_bbox("building", (-93.715, 42.012, -93.665, 42.038))   # pyarrow.Table
    geoms = to_shapely(tbl)                                             # list/array of shapely geoms

Licences: see LICENSES below. Observed ``sources[].license`` values in the Ames test box: OpenStreetMap
ODbL-1.0, Microsoft ML Buildings ODbL-1.0, Esri Community Maps CC-BY-4.0 (with OSM waivers), TomTom
ODbL-1.0 (roads), ESA WorldCover CC-BY-4.0 (land_cover). Attribution to OpenStreetMap contributors /
Overture Maps Foundation is required and ODbL share-alike applies to derived *databases*.

Coordinates: EPSG:4326 lon/lat degrees, geometry = WKB (OGC), ``bbox`` = struct<xmin,xmax,ymin,ymax> (double).
Heights/widths are metres. Not handled: boxes that cross the antimeridian (split them).
"""
from __future__ import annotations

import io
import json
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import shapely

S3_HOST = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"

# friendly name -> (theme, type)
TYPES = {
    "building": ("buildings", "building"),
    "building_part": ("buildings", "building_part"),
    "segment": ("transportation", "segment"),        # roads / paths / rail / water-lines (linestrings)
    "connector": ("transportation", "connector"),    # junction points between segments
    "water": ("base", "water"),
    "land_use": ("base", "land_use"),
    "land_cover": ("base", "land_cover"),
    "land": ("base", "land"),                        # incl. trees/forest/etc. (OSM natural=*), points+polys
    "infrastructure": ("base", "infrastructure"),    # bridges, power lines, towers, poles, pipelines ...
    "bathymetry": ("base", "bathymetry"),
    "place": ("places", "place"),
}

# columns returned by default (top-level parquet column names; nested structs are read whole).
# "sources" (list<struct> with dataset/license/record_id) and "names" are heavy, opt-in.
DEFAULT_COLUMNS = {
    "building": ["id", "geometry", "bbox", "height", "min_height", "num_floors", "num_floors_underground",
                 "min_floor", "level", "is_underground", "has_parts", "subtype", "class", "roof_shape",
                 "roof_height", "roof_material", "roof_color", "roof_direction", "roof_orientation",
                 "facade_material", "facade_color"],
    "building_part": ["id", "geometry", "bbox", "building_id", "height", "min_height", "num_floors",
                      "min_floor", "is_underground", "roof_shape", "roof_height", "roof_material",
                      "roof_color", "roof_direction", "roof_orientation", "facade_material", "facade_color"],
    "segment": ["id", "geometry", "bbox", "subtype", "class", "subclass", "subclass_rules", "road_surface",
                "road_flags", "width_rules", "level_rules", "access_restrictions", "speed_limits",
                "names", "destinations", "prohibited_transitions", "rail_flags"],    # add "connectors" (0.9 MB/rowgroup) for topology
    "connector": ["id", "geometry", "bbox"],
    "water": ["id", "geometry", "bbox", "subtype", "class", "is_salt", "is_intermittent", "level", "names"],
    "land_use": ["id", "geometry", "bbox", "subtype", "class", "surface", "level", "names"],
    "land_cover": ["id", "geometry", "bbox", "subtype", "cartography"],
    "land": ["id", "geometry", "bbox", "subtype", "class", "surface", "elevation", "level", "names"],
    "infrastructure": ["id", "geometry", "bbox", "subtype", "class", "surface", "height", "level", "names"],
    "bathymetry": ["id", "geometry", "bbox", "depth", "cartography"],
    "place": ["id", "geometry", "bbox", "names", "categories", "confidence", "operating_status"],
}

LICENSES = {      # "observed" = seen in sources[].license of the Ames test box; others recalled from docs.overturemaps.org/attribution (verify)
    "buildings": "ODbL-1.0 (observed: OpenStreetMap ODbL-1.0, Microsoft ML Buildings ODbL-1.0, Esri Community Maps CC-BY-4.0 w/ OSM waivers); other regions may add Google Open Buildings etc.",
    "transportation": "ODbL-1.0 (observed: OpenStreetMap, TomTom -> ODbL-1.0)",
    "base": "water/land/land_use/infrastructure: ODbL-1.0 (observed, OSM); land_cover: CC-BY-4.0 (observed, ESA WorldCover); bathymetry: CC0 / public-domain sources (not checked)",
    "places": "CDLA-Permissive-2.0 (recalled, not checked)",
}

DEFAULT_CACHE = os.environ.get("OVERTURE_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=64, max_retries=3))
    return s


class ByteCounter:
    """Thread-safe counter of HTTP bytes and requests (pass the same instance to all calls)."""

    def __init__(self):
        self.bytes = 0
        self.requests = 0
        self.by_kind = {}           # kind -> [bytes, requests]; kinds: list, index, footer, data
        self._lock = threading.Lock()

    def add(self, n, kind="other"):
        with self._lock:
            self.bytes += n
            self.requests += 1
            k = self.by_kind.setdefault(kind, [0, 0])
            k[0] += n
            k[1] += 1

    def report(self):
        return ", ".join(f"{k}: {v[0] / 1e6:.1f} MB/{v[1]} req" for k, v in sorted(self.by_kind.items()))


def _get(url, headers=None, counter: ByteCounter | None = None, retries=4, kind="other"):
    for attempt in range(retries):
        try:
            r = _session().get(url, headers=headers, timeout=60)
            if r.status_code in (200, 206):
                if counter:
                    counter.add(len(r.content), kind)
                return r
            if r.status_code in (500, 502, 503, 504):
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1 + attempt)
    raise RuntimeError(f"GET failed: {url} {headers}")


# ----------------------------------------------------------------------------------------------
# release / file listing
# ----------------------------------------------------------------------------------------------
def _list(prefix, delimiter=None, counter=None):
    """S3 ListObjectsV2 on the public bucket. Returns (common_prefixes, [(key, size), ...])."""
    prefixes, keys, token = [], [], None
    while True:
        params = f"list-type=2&prefix={requests.utils.quote(prefix)}"
        if delimiter:
            params += f"&delimiter={delimiter}"
        if token:
            params += f"&continuation-token={requests.utils.quote(token, safe='')}"
        x = _get(f"{S3_HOST}/?{params}", counter=counter, kind="list").text
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", x)
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x, re.S)]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", x)
        if not m:
            return prefixes, keys
        token = m.group(1)


def latest_release(counter=None) -> str:
    """Newest stable release id, e.g. '2026-09-23.1' (release names are 'YYYY-MM-DD.N'; ignores '-alpha' etc.)."""
    prefixes, _ = _list("release/", "/", counter)
    rel = [p.split("/")[1] for p in prefixes]
    stable = [r for r in rel if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.\d+", r)]
    return sorted(stable, key=lambda r: (r[:10], int(r.split(".")[-1])))[-1]


def list_files(release, type_name, counter=None):
    """[(s3_key, size_bytes)] of all parquet files of a type in a release."""
    theme, typ = TYPES[type_name]
    _, keys = _list(f"release/{release}/theme={theme}/type={typ}/", counter=counter)
    return [(k, s) for k, s in keys if k.endswith(".parquet")]


# ----------------------------------------------------------------------------------------------
# file-level bbox index (from GeoParquet 'geo' metadata in the last 16 KB of every file)
# ----------------------------------------------------------------------------------------------
def file_index(release, type_name, cache_dir=DEFAULT_CACHE, counter=None, tail=16384, refresh=False):
    """List of {'key','size','bbox':[w,s,e,n]} for every file of the type; cached in ``cache_dir`` as JSON.

    The bbox is the extent of the whole file taken from the GeoParquet ``geo`` key/value metadata
    (found in the last ``tail`` bytes of the file), so no footer has to be downloaded.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"index_{release}_{type_name}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)
    files = list_files(release, type_name, counter)

    def one(item):
        key, size = item
        bbox = None
        for tl in (tail, tail * 4, tail * 16):            # 16 KB, 64 KB, 256 KB
            r = _get(f"{S3_HOST}/{key}", {"Range": f"bytes=-{tl}"}, counter, kind="index")
            m = re.search(rb'"geometry": \{[^}]*?"bbox": \[([^\]]*)\]', r.content)
            if m:
                bbox = [float(v) for v in m.group(1).split(b",")]
                break
            if tl >= size:
                break
        return {"key": key, "size": size, "bbox": bbox}      # bbox None -> treated as "global"

    with ThreadPoolExecutor(32) as ex:
        idx = list(ex.map(one, files))
    with open(path, "w") as f:
        json.dump(idx, f)
    return idx


def candidate_files(index, bbox):
    """Files whose file-level bbox intersects bbox=(west, south, east, north)."""
    w, s, e, n = bbox
    out = []
    for it in index:
        b = it["bbox"]
        if b is None or (b[0] <= e and b[2] >= w and b[1] <= n and b[3] >= s):
            out.append(it)
    return out


# ----------------------------------------------------------------------------------------------
# range-reading file object for pyarrow
# ----------------------------------------------------------------------------------------------
class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with explicit parallel prefetch.

    pyarrow's ParquetFile reads through it; ``prefetch(ranges)`` downloads the byte ranges in
    parallel first so that the later reads are served from memory (no per-page round trips).
    """

    def __init__(self, url, size, counter=None, merge_gap=64 * 1024):
        self.url, self.size, self.counter, self.merge_gap = url, size, counter, merge_gap
        self.kind = "data"
        self.pos = 0
        self.blocks = []            # list of (start, bytes)
        self._lock = threading.Lock()

    # -- io protocol
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _from_blocks(self, start, end):
        for s, b in self.blocks:
            if s <= start and end <= s + len(b):
                return b[start - s:end - s]
        return None

    def _fetch(self, start, end):
        r = _get(self.url, {"Range": f"bytes={start}-{end - 1}"}, self.counter, kind=self.kind)
        return r.content

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        start = self.pos
        if end <= start:
            return b""
        data = self._from_blocks(start, end)
        if data is None:
            data = self._fetch(start, end)
            with self._lock:
                self.blocks.append((start, data))
        self.pos = end
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

    # -- explicit prefetch
    def prefetch(self, ranges, threads=16):
        """ranges: iterable of (start, length). Adjacent ranges (gap < merge_gap) are merged."""
        rs = sorted((s, s + l) for s, l in ranges if l > 0)
        merged = []
        for s, e in rs:
            if merged and s - merged[-1][1] <= self.merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        def one(se):
            s, e = se
            return s, self._fetch(s, min(e, self.size))

        with ThreadPoolExecutor(threads) as ex:
            got = list(ex.map(one, merged))
        with self._lock:
            self.blocks.extend(got)
        return len(merged)

    def seed(self, start, data):
        with self._lock:
            self.blocks.append((start, data))


def open_parquet(key, size, cache_dir=DEFAULT_CACHE, counter=None):
    """pq.ParquetFile over HTTP; the footer is fetched with 1-2 range GETs and cached on disk."""
    url = f"{S3_HOST}/{key}"
    rf = RangeFile(url, size, counter)
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, "footer_" + re.sub(r"[^A-Za-z0-9._=-]", "_", key.split("release/")[-1]))
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            tail = f.read()
    else:
        tail = _get(url, {"Range": "bytes=-65536"}, counter, kind="footer").content
        flen = struct.unpack("<I", tail[-8:-4])[0]
        if flen + 8 > len(tail):
            tail = _get(url, {"Range": f"bytes=-{flen + 8}"}, counter, kind="footer").content
        with open(cpath, "wb") as f:
            f.write(tail)
    rf.seed(size - len(tail), tail)
    return pq.ParquetFile(rf), rf


# ----------------------------------------------------------------------------------------------
# main API
# ----------------------------------------------------------------------------------------------
def _rg_overlaps(md, bbox_cols, bbox):
    """Indices of row groups whose bbox statistics intersect bbox=(w,s,e,n)."""
    w, s, e, n = bbox
    ix = bbox_cols
    keep = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        st = {k: rg.column(ix[k]).statistics for k in ix}
        if any(v is None or not v.has_min_max for v in st.values()):
            keep.append(g)
            continue
        if st["xmin"].min <= e and st["xmax"].max >= w and st["ymin"].min <= n and st["ymax"].max >= s:
            keep.append(g)
    return keep


def _chunk_ranges(md, leaf, rgs, top_cols):
    """(start, length) of every leaf column chunk of the given top-level columns in the given row groups."""
    leaves = [j for p, j in leaf.items() if p.split(".")[0] in top_cols]
    ranges = []
    for g in rgs:
        rg = md.row_group(g)
        for j in leaves:
            c = rg.column(j)
            start = c.dictionary_page_offset if (c.has_dictionary_page and c.dictionary_page_offset) else c.data_page_offset
            ranges.append((start, c.total_compressed_size))
    return ranges


def _land_cover_finest(t):
    """land_cover is a multi-resolution pyramid: cartography.max_zoom 7 = continent-size generalised polygons
    (tens of thousands of vertices, overlapping the fine ones), max_zoom 15 = the real 10 m ESA WorldCover
    polygons. Keep only the finest level."""
    return pc.greater_equal(pc.struct_field(t.column("cartography").combine_chunks(), "max_zoom"), 15)


# type -> (extra small columns needed by the filter, filter(table)->bool mask); applied unless finest_only=False
ROW_FILTERS = {"land_cover": (["cartography"], _land_cover_finest)}


def read_bbox(type_name, bbox, columns=None, release=None, cache_dir=DEFAULT_CACHE, counter=None,
              exact=True, verbose=False, cache_result=False, finest_only=True):
    """Read all features of an Overture type whose bounding box intersects ``bbox``.

    Parameters
    ----------
    type_name : key of TYPES ('building', 'segment', 'water', 'land_use', 'land_cover', 'land',
                'infrastructure', 'building_part', 'connector', 'place', 'bathymetry')
    bbox      : (west, south, east, north) in lon/lat degrees (EPSG:4326)
    columns   : list of top-level column names, or None for DEFAULT_COLUMNS[type_name], or 'all'.
                Names that do not exist in the file's schema are silently dropped.
    exact     : drop rows whose own bbox (bbox.xmin..ymax columns) does not intersect ``bbox``
                (False: return the whole surviving row groups, single phase).
    counter   : optional ByteCounter to measure the network traffic.
    cache_result : also store/reuse the resulting table as parquet in ``cache_dir/tables`` (offline re-runs).
    finest_only  : land_cover only: drop the coarse zoom 0-7 pyramid levels (see _land_cover_finest).

    Two-phase read per surviving row group: (1) fetch only the small ``bbox`` (+ filter) columns (~0.25 MB),
    keep the rows that intersect; (2) fetch the heavy columns (geometry, ...) only for row groups that still
    have rows. Returns a pyarrow.Table (geometry = WKB, bbox = struct<xmin,xmax,ymin,ymax> double) or None.
    Features that merely intersect the bbox are returned whole (not clipped); use ``clip_to_bbox``.
    """
    release = release or latest_release(counter)
    rowfilter = ROW_FILTERS.get(type_name) if finest_only else None
    rpath = None
    if cache_result:
        import hashlib
        key = (release, type_name, tuple(bbox), columns if isinstance(columns, str) else tuple(columns or DEFAULT_COLUMNS[type_name]), exact)
        if rowfilter:
            key += ("finest",)
        h = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
        os.makedirs(os.path.join(cache_dir, "tables"), exist_ok=True)
        rpath = os.path.join(cache_dir, "tables", f"{type_name}_{h}.parquet")
        if os.path.exists(rpath):
            return pq.read_table(rpath)
    idx = file_index(release, type_name, cache_dir, counter)
    cands = candidate_files(idx, bbox)
    w, s, e, n = bbox
    tables = []
    for it in cands:
        pf, rf = open_parquet(it["key"], it["size"], cache_dir, counter)
        md = pf.metadata
        leaf = {md.row_group(0).column(j).path_in_schema: j for j in range(md.num_columns)}
        bcols = {k: leaf[f"bbox.{k}"] for k in ("xmin", "xmax", "ymin", "ymax")}
        rgs = _rg_overlaps(md, bcols, bbox)
        if verbose:
            print(f"  {it['key'].split('/')[-1][:14]}: {len(rgs)}/{md.num_row_groups} row groups overlap (stats)", end="")
        if not rgs:
            if verbose:
                print()
            continue
        top = pf.schema_arrow.names
        want = list(top) if columns == "all" else [c for c in (columns or DEFAULT_COLUMNS[type_name]) if c in top]
        for need in ("bbox", "geometry"):
            if need not in want:
                want.append(need)
        small = ["bbox"] + ([c for c in rowfilter[0] if c in top] if rowfilter else [])
        if exact or rowfilter:
            # ---- phase 1: small columns, exact row mask per row group
            rf.prefetch(_chunk_ranges(md, leaf, rgs, small))
            t1 = pf.read_row_groups(rgs, columns=small, use_threads=True)
            b = t1.column("bbox").combine_chunks()
            m = pc.and_(pc.and_(pc.less_equal(pc.struct_field(b, "xmin"), e), pc.greater_equal(pc.struct_field(b, "xmax"), w)),
                        pc.and_(pc.less_equal(pc.struct_field(b, "ymin"), n), pc.greater_equal(pc.struct_field(b, "ymax"), s)))
            if rowfilter:
                m = pc.and_(m, rowfilter[1](t1))
            mask = np.asarray(m.fill_null(False))
            starts = np.concatenate([[0], np.cumsum([md.row_group(g).num_rows for g in rgs])])
            hit = [i for i in range(len(rgs)) if mask[starts[i]:starts[i + 1]].any()]
            if verbose:
                print(f" -> {len(hit)} with matching rows, {int(mask.sum())} features")
            if not hit:
                continue
            hit_rgs = [rgs[i] for i in hit]
            sel = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in hit])
            # ---- phase 2: heavy columns of the hit row groups only
            rest = [c for c in want if c not in small]
            rf.prefetch(_chunk_ranges(md, leaf, hit_rgs, rest))
            t2 = pf.read_row_groups(hit_rgs, columns=rest, use_threads=True)
            t1s = t1.take(pa.array(sel))
            for c in small:
                if c in want:
                    t2 = t2.append_column(t1.schema.field(c), t1s.column(c))
            t2 = t2.select([c for c in want])
            t = t2.filter(pa.array(mask[sel]))
        else:
            rf.prefetch(_chunk_ranges(md, leaf, rgs, want))
            t = pf.read_row_groups(rgs, columns=want, use_threads=True)
            if verbose:
                print(f" -> {t.num_rows} rows (not filtered)")
        tables.append(t)
        rf.blocks.clear()
    if not tables:
        return None
    out = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    if rpath:
        pq.write_table(out, rpath)
    return out


def estimate_cost(type_name, bbox, columns=None, release=None, cache_dir=DEFAULT_CACHE, counter=None):
    """Dry run: which files / row groups would ``read_bbox`` touch and how many data bytes (compressed)?

    Downloads only the (cacheable, 0.15-2 MB) footers of the candidate files, no column data.
    Returns {'files': n, 'row_groups': n, 'rows_in_row_groups': n, 'data_bytes': n}.
    """
    release = release or latest_release(counter)
    idx = file_index(release, type_name, cache_dir, counter)
    out = {"files": 0, "row_groups": 0, "rows_in_row_groups": 0, "data_bytes": 0}
    for it in candidate_files(idx, bbox):
        pf, rf = open_parquet(it["key"], it["size"], cache_dir, counter)
        md = pf.metadata
        leaf = {md.row_group(0).column(j).path_in_schema: j for j in range(md.num_columns)}
        rgs = _rg_overlaps(md, {k: leaf[f"bbox.{k}"] for k in ("xmin", "xmax", "ymin", "ymax")}, bbox)
        want = columns or DEFAULT_COLUMNS[type_name]
        leaves = [j for p, j in leaf.items() if p.split(".")[0] in want or p.split(".")[0] in ("bbox", "geometry")]
        out["files"] += bool(rgs)
        out["row_groups"] += len(rgs)
        for g in rgs:
            rg = md.row_group(g)
            out["rows_in_row_groups"] += rg.num_rows
            out["data_bytes"] += sum(rg.column(j).total_compressed_size for j in leaves)
    return out


def bbox_around(lon, lat, width_m, height_m):
    """(west, south, east, north) of a width_m x height_m box centred on lon/lat (spherical approx.)."""
    dlat = height_m / 2 / 111_320.0
    dlon = width_m / 2 / (111_320.0 * np.cos(np.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def to_shapely(table):
    """numpy object array of shapely geometries decoded from the WKB 'geometry' column."""
    wkb = table.column("geometry").to_pylist()
    return shapely.from_wkb(np.array(wkb, dtype=object))


def clip_to_bbox(geoms, bbox):
    """Clip shapely geometries to the bbox rectangle (lon/lat). Empty results become None."""
    w, s, e, n = bbox
    out = shapely.clip_by_rect(geoms, w, s, e, n)
    return np.where(shapely.is_empty(out), None, out)


def to_local_metres(geoms, lon0, lat0):
    """Project shapely lon/lat geometries to a local metric frame (UTM zone of lon0/lat0, shifted so that
    (lon0, lat0) -> (0, 0)). Returns (geoms_m, EPSG code). Uses pyproj."""
    import pyproj
    zone = int((lon0 + 180) // 6) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    x0, y0 = tr.transform(lon0, lat0)
    out = shapely.transform(geoms, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])) - np.array([x0, y0]))
    return out, epsg


def read_all(bbox, types=("building", "segment", "water", "land_use", "land_cover", "land", "infrastructure"),
             **kw):
    """Convenience: {type_name: pyarrow.Table or None} for several types."""
    release = kw.pop("release", None) or latest_release(kw.get("counter"))
    return {t: read_bbox(t, bbox, release=release, **kw) for t in types}


def summarize(type_name, tbl, top=8):
    """Print feature count, per-column fill rate and top values (for categorical columns) of a table."""
    from collections import Counter
    print(f"--- {type_name}: {tbl.num_rows} features")
    for name in tbl.schema.names:
        if name in ("geometry", "bbox", "id"):
            continue
        col = tbl.column(name).combine_chunks()
        typ = col.type
        if pa.types.is_list(typ) or pa.types.is_large_list(typ):
            nn = int(pc.sum(pc.greater(pc.list_value_length(col).fill_null(0), 0)).as_py() or 0)
            flat = pc.list_flatten(col)
            if name == "sources":
                lic = Counter(zip(pc.struct_field(flat, "dataset").to_pylist(), pc.struct_field(flat, "license").to_pylist()))
                print(f"  sources: {nn} non-empty; (dataset, license) x records: {lic.most_common(top)}")
                continue
            ex = ""
            if pa.types.is_struct(flat.type):
                for f in ("value", "values"):
                    if f in [x.name for x in flat.type]:
                        vals = pc.struct_field(flat, f).to_pylist()
                        if f == "values":
                            vals = [v for vv in vals for v in (vv or [])]
                        ex = f" top {f}: {Counter(vals).most_common(top)}"
            print(f"  {name}: {nn} non-empty{ex}")
        elif pa.types.is_struct(typ):
            nn = int(pc.sum(pc.invert(pc.is_null(col))).as_py() or 0)
            sub = ""
            if name == "names" and "primary" in [x.name for x in typ]:
                pr = pc.struct_field(col, "primary")
                sub = f", primary name set: {int(pc.sum(pc.invert(pc.is_null(pr))).as_py() or 0)}"
            print(f"  {name}: {nn} non-null{sub}")
        elif pa.types.is_floating(typ) or pa.types.is_integer(typ):
            v = col.drop_null()
            if len(v):
                a = v.to_numpy(zero_copy_only=False)
                print(f"  {name}: {len(v)} non-null, min/median/max = {a.min():.2f}/{np.median(a):.2f}/{a.max():.2f}")
            else:
                print(f"  {name}: 0 non-null")
        elif pa.types.is_string(typ) or pa.types.is_boolean(typ):
            v = col.drop_null()
            cnt = Counter(v.to_pylist())
            print(f"  {name}: {len(v)} non-null; top: {cnt.most_common(top)}")
        else:
            print(f"  {name}: {typ}")
    geoms = to_shapely(tbl)
    print("  geometry types:", dict(Counter(shapely.get_type_id(geoms).tolist())), "(0 pt,1 line,2 ring,3 poly,4 mpt,5 mline,6 mpoly)")


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    BBOX = (-93.715, 42.012, -93.665, 42.038)      # west edge of Ames, Iowa (about 4 km x 3 km)
    # --fresh: use an empty cache dir so that index + footers are downloaded too
    # NOTE: results are cached as parquet in ./cache/tables (cache_result=True below) so a 2nd run is offline;
    # use --fresh to start from an empty cache (downloads index + footers + data again, about 120 MB).
    cache = DEFAULT_CACHE if "--fresh" not in sys.argv else os.path.join(DEFAULT_CACHE, "fresh_%d" % int(time.time()))
    counter = ByteCounter()
    t0 = time.time()
    rel = latest_release(counter)
    print("latest release:", rel, "bbox:", BBOX)
    tables = {}
    for name in ["building", "building_part", "segment", "water", "land_use", "land_cover", "land", "infrastructure"]:
        t1, b1 = time.time(), counter.bytes
        tbl = read_bbox(name, BBOX, columns=DEFAULT_COLUMNS[name] + ["sources"], release=rel, cache_dir=cache,
                        counter=counter, verbose=True, cache_result=True)
        n = 0 if tbl is None else tbl.num_rows
        tables[name] = tbl
        print(f"{name:15s} features={n:6d}  network={(counter.bytes - b1) / 1e6:7.2f} MB  {time.time() - t1:5.1f}s")
        if tbl is not None and "--summary" in sys.argv:
            summarize(name, tbl)
    # key facts (projected to local metres around the bbox centre)
    lon0, lat0 = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2
    b = tables["building"]
    gm, epsg = to_local_metres(to_shapely(b), lon0, lat0)
    fill = lambda t, c: t.num_rows - t.column(c).null_count
    print(f"buildings: {b.num_rows}; with height {fill(b, 'height')}, num_floors {fill(b, 'num_floors')}, "
          f"roof_shape {fill(b, 'roof_shape')}, class {fill(b, 'class')}; footprint area total {shapely.area(gm).sum() / 1e4:.1f} ha (EPSG:{epsg})")
    sg = tables["segment"]
    lm, _ = to_local_metres(to_shapely(sg), lon0, lat0)
    ln = shapely.length(lm)
    cls = np.array(sg.column("class").to_pylist())
    print("roads/paths: %d segments, %.1f km total; km by class: %s" % (
        sg.num_rows, ln.sum() / 1e3, {str(c): round(float(ln[cls == c].sum()) / 1e3, 1) for c in sorted(set(cls), key=lambda c: -ln[cls == c].sum())[:6]}))
    ld = tables["land"]
    gl = shapely.get_type_id(to_shapely(ld))
    sub = np.array(ld.column("subtype").to_pylist())
    print(f"land: {int(((sub == 'tree') & (gl == 0)).sum())} individual tree points (subtype=tree, Point), "
          f"{int((gl == 3).sum())} vegetation polygons")
    print(f"TOTAL network: {counter.bytes / 1e6:.1f} MB in {counter.requests} requests, {time.time() - t0:.1f}s  [{counter.report()}]")
