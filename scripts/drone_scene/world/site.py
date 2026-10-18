"""Choosing where the flight starts and ends, from the map itself (numpy + shapely).

Take-off: a backyard.  Candidate spots lie 9-14 m from a house, on the side away from the nearest street, on lawn,
with no building within 6 m, no road within 8 m of the road edge and no tree crown above (canopy < 1.5 m within
6 m).  Landing: open grass (meadow, pasture or a park lawn) at the requested distance (+-5 %) from the take-off,
at least 35 m from buildings, trees and roads and 30 m from water.  Among the pairs that work, the route that
crosses the most varied ground (roads, houses, trees, fields) wins, so the onboard cameras see a real mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import landcover as lc
from .geo import Grid

HOUSE_CLASSES = {"house", "detached", "semidetached_house", "residential", "bungalow", "farmhouse", None}


@dataclass
class Site:
    takeoff: tuple[float, float]           # local frame of the search
    landing: tuple[float, float]
    distance: float
    heading_deg: float                     # counter-clockwise from +x (east)
    score: float
    notes: dict


def _min_dist(tree, geoms, pts):
    import shapely
    if tree is None:
        return np.full(len(pts), np.inf)
    p = shapely.points(pts[:, 0], pts[:, 1])
    idx = tree.nearest(p)
    return shapely.distance(p, np.asarray(geoms, dtype=object)[idx])


def _canopy_max(canopy: Grid | None, pts, radius):
    if canopy is None:
        return np.zeros(len(pts))
    out = np.zeros(len(pts))
    offs = [(dx, dy) for dx in np.arange(-radius, radius + 0.1, 2.0) for dy in np.arange(-radius, radius + 0.1, 2.0)
            if dx * dx + dy * dy <= radius * radius]
    for dx, dy in offs:
        out = np.maximum(out, np.nan_to_num(canopy.sample(pts[:, 0] + dx, pts[:, 1] + dy, "nearest")))
    return out


def takeoff_candidates(houses, roads, buildings, classes: Grid, canopy: Grid | None, centre=(0.0, 0.0),
                       max_houses: int = 600) -> np.ndarray:
    """(N, 3): x, y, clearance score of backyard spots."""
    import shapely
    if not houses:
        return np.zeros((0, 3))
    bgeoms = list(buildings)
    btree = shapely.STRtree(bgeoms) if bgeoms else None
    rgeoms = [line.buffer(w / 2) for line, w in roads]
    rtree = shapely.STRtree(rgeoms) if rgeoms else None
    cx, cy = centre
    houses = sorted(houses, key=lambda p: (p.centroid.x - cx) ** 2 + (p.centroid.y - cy) ** 2)[:max_houses]
    cands = []
    for h in houses:
        c = np.array([h.centroid.x, h.centroid.y])
        # direction away from the nearest street
        d_road = _min_dist(rtree, rgeoms, c[None])[0]
        if not np.isfinite(d_road) or d_road > 80:
            continue
        p_road = np.asarray(shapely.shortest_line(shapely.points(*c), rgeoms[rtree.nearest(shapely.points(*c))]).coords[1])
        away = c - p_road
        away = away / max(np.linalg.norm(away), 1e-6)
        for r in (9.0, 11.5, 14.0):
            for a in (-0.6, -0.3, 0.0, 0.3, 0.6):
                ca, sa = math.cos(a), math.sin(a)
                d = np.array([ca * away[0] - sa * away[1], sa * away[0] + ca * away[1]])
                q = c + d * (r + math.sqrt(h.area) / 2)
                cands.append((q[0], q[1]))
    if not cands:
        return np.zeros((0, 3))
    P = np.asarray(cands)
    d_b = _min_dist(btree, bgeoms, P)
    d_r = _min_dist(rtree, rgeoms, P)
    cls = classes.sample(P[:, 0], P[:, 1], "nearest")
    can = _canopy_max(canopy, P, 6.0)
    ok = (d_b >= 6.0) & (d_r >= 8.0) & (cls == lc.LAWN) & (can < 1.5)
    score = np.minimum(d_b, 15) + 0.5 * np.minimum(d_r, 20)
    return np.column_stack([P[ok], score[ok]])


def landing_ok(P, buildings, roads, water, classes: Grid, canopy: Grid | None) -> np.ndarray:
    import shapely
    if len(P) == 0:
        return np.zeros(0, bool)
    bt = shapely.STRtree(list(buildings)) if buildings else None
    rg = [line.buffer(w / 2) for line, w in roads]
    rt = shapely.STRtree(rg) if rg else None
    wt = shapely.STRtree(list(water)) if water else None
    cls = classes.sample(P[:, 0], P[:, 1], "nearest")
    grassy = np.isin(cls, [lc.MEADOW, lc.LAWN])
    # the whole landing zone must be grass, not just its centre
    for dx, dy in ((15, 0), (-15, 0), (0, 15), (0, -15)):
        grassy &= np.isin(classes.sample(P[:, 0] + dx, P[:, 1] + dy, "nearest"), [lc.MEADOW, lc.LAWN])
    ok = grassy & (_min_dist(bt, list(buildings), P) >= 35) & (_min_dist(rt, rg, P) >= 35)
    ok &= (_min_dist(wt, list(water), P) >= 30) & (_canopy_max(canopy, P, 30.0) < 1.0)
    return ok


def route_score(a, b, classes: Grid, buildings, roads, canopy: Grid | None) -> tuple[float, dict]:
    """Variety along the straight line: distinct ground classes, roads crossed, houses and trees passed."""
    import shapely
    n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / 10) + 1
    t = np.linspace(0, 1, n)
    P = np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
    cls = classes.sample(P[:, 0], P[:, 1], "nearest")
    kinds = len(np.unique(cls))
    line = shapely.LineString([a, b])
    corridor = line.buffer(60)
    crossings = sum(1 for g, _ in roads if g.intersects(line))
    near_b = sum(1 for g in buildings if g.intersects(corridor))
    trees = float(np.mean(np.nan_to_num(canopy.sample(P[:, 0], P[:, 1], "nearest")) > 3)) if canopy is not None else 0
    water = float(np.mean(cls == lc.WATER))
    score = 3 * kinds + 1.5 * min(crossings, 8) + 0.2 * min(near_b, 60) + 10 * min(trees, 0.3) - 20 * water
    return score, {"classes": int(kinds), "road_crossings": int(crossings), "buildings_near": int(near_b),
                   "tree_fraction": round(trees, 3), "water_fraction": round(water, 3)}


def choose(houses, buildings, roads, water, classes: Grid, canopy: Grid | None, distance: float = 2000.0,
           centre=(0.0, 0.0), max_takeoffs: int = 40, bearing_step_deg: float = 3.0, log=None,
           takeoff_roads=None) -> Site:
    """The best (take-off, landing) pair; raises if the area has none.  ``takeoff_roads`` (default ``roads``) are
    the lines a backyard must keep clear of: pass footpaths and cycle paths too, the backyard set is ~12 m across."""
    T = takeoff_candidates(houses, takeoff_roads if takeoff_roads is not None else roads, buildings, classes, canopy,
                           centre)
    if len(T) == 0:
        raise RuntimeError("no backyard take-off spot found near the requested place (no lawn behind a house "
                           "clear of trees and roads); give --takeoff lat,lon")
    T = T[np.argsort(-T[:, 2])][:max_takeoffs]
    if log:
        log(f"[site] {len(T)} backyard candidates")
    best = None
    bearings = np.radians(np.arange(0, 360, bearing_step_deg))
    for tx, ty, tscore in T:
        for f in (1.0, 0.975, 1.025, 0.95, 1.05):
            P = np.column_stack([tx + distance * f * np.cos(bearings), ty + distance * f * np.sin(bearings)])
            inside = (P[:, 0] > classes.rect[0] + 40) & (P[:, 0] < classes.rect[1] - 40) & \
                     (P[:, 1] > classes.rect[2] + 40) & (P[:, 1] < classes.rect[3] - 40)
            P = P[inside]
            ok = landing_ok(P, buildings, roads, water, classes, canopy)
            for q in P[ok]:
                s, notes = route_score((tx, ty), tuple(q), classes, buildings, roads, canopy)
                s += 0.3 * tscore - 10 * abs(f - 1)
                if best is None or s > best.score:
                    head = math.degrees(math.atan2(q[1] - ty, q[0] - tx))
                    best = Site((float(tx), float(ty)), (float(q[0]), float(q[1])),
                                float(np.hypot(q[0] - tx, q[1] - ty)), head, float(s), notes)
            if best is not None and f == 1.0:
                break
    if best is None:
        raise RuntimeError(f"no open grass found {distance:.0f} m from any backyard candidate; give --landing lat,lon "
                           "or another --region")
    if log:
        log(f"[site] take-off {best.takeoff[0]:.0f},{best.takeoff[1]:.0f} -> landing {best.landing[0]:.0f},"
            f"{best.landing[1]:.0f} ({best.distance:.0f} m, heading {best.heading_deg:.0f} deg): {best.notes}")
    return best
