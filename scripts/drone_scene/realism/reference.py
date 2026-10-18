"""Real pictures of what the probe's nadir stills show: USDA NAIP orthophotos (public domain; 0.6 m or finer; flown
in summer, leaves on) cut to exactly the square each nadir still covers, north up, at the still's resolution.  Read
over HTTP from Microsoft Planetary Computer (no account needed; only the needed blocks are downloaded).

Needs the map tools (rasterio, pyproj, requests), so run it through with_world_libs.py:

    ./isaaclab.sh -p scripts/drone_scene/realism/with_world_libs.py scripts/drone_scene/realism/reference.py \\
        .realism/baseline/probe .realism/reference/naip
    -> .realism/reference/naip/nadir_<moment>.png (the same names as the stills) and naip.json (which NAIP image)
    then: metrics.py .realism/baseline/probe --pairs .realism/reference/naip

The square assumes flat ground at the take-off height; a few metres of relief change the scale by ~1 %.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
ATTRIBUTION = "NAIP imagery: USDA Farm Production and Conservation Business Center (public domain), via Microsoft Planetary Computer"


def footprint_lonlat(frame, x: float, y: float, side: float) -> tuple[float, float, float, float]:
    return frame.bbox_lonlat((x - side / 2, x + side / 2, y - side / 2, y + side / 2), pad_m=5.0)


def search(session, bbox) -> list[dict]:
    """NAIP items over the box, newest first."""
    r = session.post(STAC, json={"collections": ["naip"], "bbox": list(bbox), "limit": 100}, timeout=60)
    r.raise_for_status()
    items = r.json().get("features", [])
    return sorted(items, key=lambda it: it.get("properties", {}).get("datetime") or "", reverse=True)


def signed(session, href: str, tokens: dict) -> str:
    """The asset URL with a read token for its Azure storage container."""
    u = urllib.parse.urlparse(href)
    account, container = u.netloc.split(".")[0], u.path.lstrip("/").split("/")[0]
    if (account, container) not in tokens:
        r = session.get(f"{SAS}/{account}/{container}", timeout=60)
        r.raise_for_status()
        tokens[(account, container)] = r.json()["token"]
    return f"{href}?{tokens[(account, container)]}"


def covers(item: dict, bbox) -> bool:
    w, s, e, n = item.get("bbox", [0, 0, 0, 0])[:4]
    return w < bbox[2] and e > bbox[0] and s < bbox[3] and n > bbox[1]


def cut(session, items: list[dict], frame, x: float, y: float, side: float, px: int, tokens: dict) -> np.ndarray:
    """The square (side x side metres around local x, y; north up) as RGB8 (px, px, 3), mosaicked from the
    newest NAIP year that covers it."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    dst = np.zeros((3, px, px), np.uint8)
    transform = from_origin(x - side / 2, y + side / 2, side / px, side / px)
    crs = CRS.from_proj4(frame.proj4)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
        for it in items:
            with rasterio.open(signed(session, it["assets"]["image"]["href"], tokens)) as src:
                tmp = np.zeros_like(dst)
                reproject(source=rasterio.band(src, [1, 2, 3]), destination=tmp, src_transform=src.transform,
                          src_crs=src.crs, src_nodata=0, dst_transform=transform, dst_crs=crs, dst_nodata=0,
                          resampling=Resampling.bilinear)
            empty = dst.sum(0) == 0
            dst[:, empty] = tmp[:, empty]
            if not (dst.sum(0) == 0).any():
                break
    return np.transpose(dst, (1, 2, 0))


def main(argv=None) -> int:
    import requests
    from PIL import Image

    from drone_scene.world.geo import Frame

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="a --probe output folder (with probe.json and nadir_*.png)")
    ap.add_argument("out", help="where to put the NAIP squares")
    ap.add_argument("--force", action="store_true", help="fetch again even if a square is already there")
    a = ap.parse_args(argv)
    with open(os.path.join(a.probe, "probe.json"), encoding="utf-8") as f:
        probe = json.load(f)
    origin = (probe.get("map") or {}).get("origin") or (probe.get("map") or {}).get("takeoff")
    if not origin:
        print("probe.json has no map origin: this needs a --route straight probe", file=sys.stderr)
        return 2
    frame = Frame(float(origin["lat"]), float(origin["lon"]))
    stills = [s for s in probe.get("stills", []) if s.get("camera") == "nadir" and "nadir" in s]
    if not stills:
        print("no nadir stills in this probe: render with --cameras ... nadir", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "naip.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f).get("squares", {})
    session, tokens = requests.Session(), {}
    for s in stills:
        x, y = s["nadir"]["position_m"][:2]
        side = float(s["nadir"]["footprint_m"])
        with Image.open(os.path.join(a.probe, s["file"])) as im:
            px = im.width
        want = {"x": round(x, 2), "y": round(y, 2), "side_m": side, "px": px}
        path = os.path.join(a.out, s["file"])
        if not a.force and os.path.exists(path) and all(index.get(s["file"], {}).get(k) == v for k, v in want.items()):
            print(f"{s['file']}: already there")
            continue
        bbox = footprint_lonlat(frame, x, y, side)
        items = [it for it in search(session, bbox) if covers(it, bbox)]
        if not items:
            print(f"{s['file']}: no NAIP image covers it", file=sys.stderr)
            continue
        year = (items[0]["properties"].get("datetime") or "")[:4]
        items = [it for it in items if (it["properties"].get("datetime") or "")[:4] == year]
        rgb = cut(session, items, frame, x, y, side, px, tokens)
        Image.fromarray(rgb).save(path)
        index[s["file"]] = {**want, "year": year, "items": [it["id"] for it in items],
                            "gsd_m": items[0]["properties"].get("gsd"), "date": items[0]["properties"].get("datetime"),
                            "empty_fraction": round(float((rgb.sum(-1) == 0).mean()), 4)}
        print(f"{s['file']}: NAIP {year} ({', '.join(it['id'] for it in items)}), {side:.0f} m square")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"attribution": ATTRIBUTION, "squares": index}, f, indent=1)
    print(f"wrote {a.out} ({len(index)} squares). {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
