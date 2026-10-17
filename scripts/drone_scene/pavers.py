"""Real 3D herringbone clay pavers for the patio: micro-geometry instead of a flat texture.

Unreal Engine 5 scenes get their depth from real micro-geometry (Nanite) and scanned assets, not from
one big texture.  This module does the same for the patio: every paver of a 90-degree herringbone is a
box with chamfered top edges, a hair of height and tilt noise, and its own crop of the NVIDIA
``Brick_Pavers`` texture, over a flat joint-sand surface a few millimetres lower.

Pipeline (``author_patio`` runs all of it):

1. ``brick_atlas``: find the individual pavers in the texture (ORM occlusion is dark in the joints:
   threshold, connected components, size/aspect filter, shrink) and return each as a UV rectangle in USD
   UV space (v = 0 is the bottom row of the image).  Cached as JSON next to the textures.
2. ``herringbone``: lay 200 mm pavers over the rect; pavers crossing its edge are cut (sawn edge, no
   chamfer there).
3. ``build_bricks``: one polygon mesh per ~2 m chunk and colour class: top, four chamfers and four sides per
   paver, normals split at the hard edges, top UVs mapped into one random crop (rotated by 90 degrees where
   the paver's orientation differs from the crop's).  Every paver also draws one of a few colour tints.
4. A sand surface under the whole patio (darker in bands along the curb) so the joints show sand.

The crop transform is plain mesh UVs (``st`` primvar): no UsdTransform2d, so RTX and the Mitsuba
preview read it identically.  Only pxr is needed to author, and it is imported inside functions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from PIL import Image

from . import assets, scene

ATLAS_VERSION = 2             # bump when the segmentation changes in a way the constants below do not show
ATLAS_ASPECT = 2.0            # long side : short side of every crop (a 200 x 98.5 mm paver is 2.03)
ATLAS_MARGIN_PX = 8           # crops stay this far inside the flat part of the brick (mip filtering blurs)
ATLAS_ERODE_PX = 2            # shrink of the brick mask before the components are found (cuts thin bridges)
ATLAS_SIZE_TOL = 0.12         # a brick's sides must be within this fraction of the median brick's
ATLAS_LEVEL = 0.6             # mask threshold: this far from the joint class towards the brick class
ATLAS_BLUR_PX = 2             # box-blur radius of the occlusion before thresholding
ATLAS_SHAPE = (1.6, 2.6, 0.85)    # accepted long : short range of a component, and its minimum box fill

# Brick-to-brick colour: the texture's pavers differ by under 2 % in luminance, real ones by 10 % or more.
# Every paver draws one of these (linear RGB multipliers of the albedo, mean luminance ~1.0, a little
# warmer in the light ones); each class is its own material because UsdPreviewSurface has no vertex colour.
BRICK_TINTS = ((0.84, 0.88, 0.92), (0.95, 0.95, 0.95), (1.05, 1.03, 1.01), (1.16, 1.10, 1.04))

# distance from the curb (m) and the tint of the joint sand there: damp, dirty joints along the edge,
# in small steps so a joint running towards the curb darkens without a visible edge
JOINT_BANDS = ((0.04, 0.58), (0.09, 0.68), (0.15, 0.79), (0.24, 0.90))
# the bare Sand texture is a clean orange-beige; joint sand in a paver patio is a dustier grey-tan
SAND_TINT = (0.62, 0.74, 0.92)

MIN_SLIVER = 0.0015           # a cut paver narrower than this is left out (the joint sand shows instead)


# ---------------------------------------------------------------------------
# Brick atlas
# ---------------------------------------------------------------------------

class Crop(NamedTuple):
    """One paver of the texture as a UV rectangle (USD UV: v = 0 is the bottom of the image)."""
    u0: float
    v0: float
    u1: float
    v1: float
    horizontal: bool          # the long side runs along u


def _otsu(values: np.ndarray) -> int:
    """Otsu threshold of a uint8 image: pixels above it are the bright class."""
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m = np.cumsum(hist * np.arange(256))
    mu0 = m / np.maximum(w0, 1.0)
    mu1 = (m[-1] - m) / np.maximum(w1, 1.0)
    return int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)^2 box blur with replicated borders (the speckle in a brick must not break its mask)."""
    h, w = img.shape
    p = np.pad(img.astype(np.float32), r, mode="edge")
    c = np.cumsum(np.pad(p, ((0, 0), (1, 0))), axis=1)
    rows = (c[:, 2 * r + 1:2 * r + 1 + w] - c[:, :w]) / (2 * r + 1)
    c = np.cumsum(np.pad(rows, ((1, 0), (0, 0))), axis=0)
    return (c[2 * r + 1:2 * r + 1 + h] - c[:h]) / (2 * r + 1)


def _brick_mask(ao: np.ndarray, lum: np.ndarray | None, level: float = ATLAS_LEVEL,
                blur_px: int = ATLAS_BLUR_PX) -> np.ndarray:
    """Bright (brick) pixels of the ORM occlusion, or of the albedo luminance when occlusion is flat.

    The threshold sits ``level`` of the way from the joint class to the brick class, a little above Otsu's
    midpoint, so the rough rim of a brick (where the joint begins) is not part of the mask.
    """
    img = ao
    if lum is not None and int(np.percentile(ao, 98)) - int(np.percentile(ao, 2)) < 10:
        img = lum
    img = _box_blur(np.asarray(img, np.uint8), blur_px) if blur_px > 0 else np.asarray(img, np.float32)
    t = _otsu(np.round(img).astype(np.uint8))
    lo, hi = img[img <= t], img[img > t]
    if len(lo) == 0 or len(hi) == 0:
        return np.zeros(img.shape, bool)
    return img >= lo.mean() + level * (float(np.median(hi)) - float(lo.mean()))


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Square erosion by r pixels; the image border counts as set so cut bricks keep touching it."""
    if r <= 0:
        return mask
    h, w = mask.shape
    p = np.pad(mask, r, mode="edge")
    rows = np.logical_and.reduce([p[:, i:i + w] for i in range(2 * r + 1)])
    return np.logical_and.reduce([rows[i:i + h] for i in range(2 * r + 1)])


def component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected components of a boolean mask as (x0, y0, x1, y1, area), x1/y1 exclusive.

    Run-length union-find: runs of set pixels in a row join the overlapping runs of the row above, so a
    2048 x 2048 mask takes ~0.1 s without scipy.
    """
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs: list[tuple[int, int, int, int]] = []
    prev: list[tuple[int, int, int]] = []
    for y in range(mask.shape[0]):
        d = np.diff(mask[y].astype(np.int8), prepend=0, append=0)
        cur, j = [], 0
        for s, e in zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()):
            rid = len(parent)
            parent.append(rid)
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                a, b = find(rid), find(prev[k][2])
                if a != b:
                    parent[b] = a
                k += 1
            cur.append((s, e, rid))
            runs.append((y, s, e, rid))
        prev = cur
    boxes: dict[int, list[int]] = {}
    for y, s, e, rid in runs:
        b = boxes.setdefault(find(rid), [s, y, e, y + 1, 0])
        b[0], b[1], b[2], b[3], b[4] = min(b[0], s), min(b[1], y), max(b[2], e), y + 1, b[4] + e - s
    return [tuple(b) for b in boxes.values()]


def segment_bricks(ao: np.ndarray, lum: np.ndarray | None = None, *, margin_px: int = ATLAS_MARGIN_PX,
                   aspect: float = ATLAS_ASPECT, erode_px: int = ATLAS_ERODE_PX,
                   size_tol: float = ATLAS_SIZE_TOL) -> list[Crop]:
    """The whole bricks of a herringbone texture as UV crops.

    ``ao`` (H, W) uint8 is the brick/joint contrast image (the ORM red channel), ``lum`` an optional
    albedo luminance used when ``ao`` has no contrast.  Components touching the image border are cut
    bricks and are dropped; the rest must be rectangles (``ATLAS_SHAPE``: box fill and long : short range)
    whose size is within ``size_tol`` of the median brick.  Each crop is the largest ``aspect`` : 1 rectangle centred in the
    brick and ``margin_px`` inside it.
    """
    h, w = ao.shape[:2]
    mask = _brick_mask(ao, lum)
    lo_ratio, hi_ratio, min_fill = ATLAS_SHAPE
    boxes = []
    for x0, y0, x1, y1, area in component_boxes(_erode(mask, erode_px)):
        if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
            continue
        bw, bh = x1 - x0, y1 - y0
        if area < min_fill * bw * bh or not lo_ratio <= max(bw, bh) / max(min(bw, bh), 1) <= hi_ratio:
            continue
        boxes.append((x0 - erode_px, y0 - erode_px, x1 + erode_px, y1 + erode_px))
    if not boxes:
        return []
    arr = np.array(boxes, float)
    bw, bh = arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]
    long_, short = np.maximum(bw, bh), np.minimum(bw, bh)
    ok = (abs(long_ - np.median(long_)) <= size_tol * np.median(long_)) & \
         (abs(short - np.median(short)) <= size_tol * np.median(short))
    crops = []
    for (x0, y0, x1, y1), good in sorted(zip(boxes, ok), key=lambda t: (t[0][1], t[0][0])):
        if not good:
            continue
        horizontal = (x1 - x0) >= (y1 - y0)
        lg, sh = max(x1 - x0, y1 - y0) - 2 * margin_px, min(x1 - x0, y1 - y0) - 2 * margin_px
        if sh < 4:
            continue
        sh = min(sh, lg / aspect)
        lg = sh * aspect
        cw, ch = (lg, sh) if horizontal else (sh, lg)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        # image rows run top-down, USD v bottom-up
        crops.append(Crop((cx - cw / 2) / w, 1.0 - (cy + ch / 2) / h, (cx + cw / 2) / w, 1.0 - (cy - ch / 2) / h,
                          horizontal))
    return crops


def _luminance(path: str) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float64)
    return (rgb @ np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8)


def brick_atlas(textures: dict[str, str], *, use_cache: bool = True) -> list[Crop]:
    """The bricks of the Brick_Pavers texture set (``{'orm': path, 'albedo': path, ...}``) as UV crops.

    Cached as ``paver_atlas.json`` next to the ORM texture (not rewritten when the cache is read-only),
    keyed by the file sizes of the textures read and every ``ATLAS_*`` constant, so tuning one re-segments.
    """
    orm, albedo = textures["orm"], textures.get("albedo")
    cache = os.path.join(os.path.dirname(orm), "paver_atlas.json")
    key = {"version": ATLAS_VERSION, "orm": os.path.basename(orm), "bytes": os.path.getsize(orm),
           "albedo_bytes": os.path.getsize(albedo) if albedo and os.path.exists(albedo) else 0,
           "margin_px": ATLAS_MARGIN_PX, "aspect": ATLAS_ASPECT, "erode_px": ATLAS_ERODE_PX,
           "size_tol": ATLAS_SIZE_TOL, "level": ATLAS_LEVEL, "blur_px": ATLAS_BLUR_PX, "shape": list(ATLAS_SHAPE)}
    if use_cache and os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("key") == key:
                return [Crop(*c[:4], bool(c[4])) for c in data["crops"]]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    ao = np.asarray(Image.open(orm).convert("RGB"))[..., 0]
    lum = _luminance(albedo) if albedo else None
    crops = segment_bricks(ao, lum)
    if not crops:
        raise RuntimeError(f"no paver bricks found in {orm}")
    if use_cache:
        try:
            with open(cache + ".part", "w", encoding="utf-8") as f:
                json.dump({"key": key, "crops": [[*c[:4], c[4]] for c in crops]}, f)
            os.replace(cache + ".part", cache)
        except OSError:
            pass
    return crops


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaverSpec:
    """A paver and how it is laid.  The width follows from the joint: in a herringbone, length = 2 x width
    + joint, so 200 mm pavers with 3 mm joints are 98.5 mm wide."""
    length: float = 0.200
    joint: float = 0.003
    thickness: float = 0.060
    chamfer: float = 0.004
    height_jitter: float = 0.001          # +- m, each paver
    tilt_deg: float = 0.25                # +- degrees about each horizontal axis, each paver
    bevel_smooth: float = 0.5             # 0: flat chamfer shading, 1: normals continuous with top and side

    @property
    def width(self) -> float:
        return 0.5 * (self.length - self.joint)

    @property
    def pitch(self) -> float:
        return self.width + self.joint

    @property
    def top_rise(self) -> float:
        """The most a paver top can stand above the nominal top (m): height jitter plus the tilt over a
        half diagonal.  Put anything lying on the pavers (leaves) above this."""
        return self.height_jitter + math.tan(math.radians(self.tilt_deg)) * 0.5 * (self.length + self.width)


@dataclass
class Layout:
    """The pavers over a rect: full footprints, the parts inside the rect, orientation, sawn sides."""
    rect: tuple[float, float, float, float]
    full: np.ndarray             # (N, 4) x0, x1, y0, y1 of the whole paver
    box: np.ndarray              # (N, 4) the part of it inside the rect
    horizontal: np.ndarray       # (N,) bool: long side along x
    cut: np.ndarray              # (N, 4) bool: west, east, south, north side is a sawn edge

    def __len__(self) -> int:
        return len(self.full)


def herringbone(rect, spec: PaverSpec = PaverSpec()) -> Layout:
    """A 90-degree herringbone anchored at the rect's (x0, y0) corner, cut to the rect.

    On a grid of cells of one paver width (+ joint), a horizontal paver covers cells (g, h), (g+1, h) and a
    vertical one (g+2, h), (g+2, h+1) for g = 3k - s, h = k + s over all integers k, s: zig-zag lines that
    shift by (-1, 1) from one to the next tile the plane exactly.
    """
    x0, x1, y0, y1 = (float(v) for v in rect)
    p, j = spec.pitch, spec.joint
    ox, oy = x0 - j / 2, y0 - j / 2                     # a paver's edge, not its joint, starts at the corner
    nx, ny = int(math.ceil((x1 - ox) / p)) + 2, int(math.ceil((y1 - oy) / p)) + 2
    # cells with -3 <= g <= nx and -3 <= h <= ny: k = (g + h) / 4, s = (3h - g) / 4, plus a ring of slack
    k, s = np.meshgrid(np.arange(-3, (nx + ny) // 4 + 3), np.arange((-9 - nx) // 4 - 2, (3 * ny + 3) // 4 + 3))
    gx, gy = (3 * k - s).ravel().astype(float), (k + s).ravel().astype(float)
    horiz = np.concatenate([np.ones(len(gx), bool), np.zeros(len(gx), bool)])
    # grown cell of a horizontal paver: 2 x 1 pitches, of a vertical one: 1 x 2, from cell (g, h)
    fx0 = np.concatenate([gx, gx + 2]) * p + ox + j / 2
    fy0 = np.concatenate([gy, gy]) * p + oy + j / 2
    fx1 = np.concatenate([gx + 2, gx + 3]) * p + ox - j / 2
    fy1 = np.concatenate([gy + 1, gy + 2]) * p + oy - j / 2
    full = np.stack([fx0, fx1, fy0, fy1], axis=1)
    box = np.stack([np.maximum(fx0, x0), np.minimum(fx1, x1), np.maximum(fy0, y0), np.minimum(fy1, y1)], axis=1)
    keep = ((box[:, 1] - box[:, 0]) > MIN_SLIVER) & ((box[:, 3] - box[:, 2]) > MIN_SLIVER)
    full, box, horiz = full[keep], box[keep], horiz[keep]
    order = np.lexsort((box[:, 0], box[:, 2]))
    full, box, horiz = full[order], box[order], horiz[order]
    eps = 1e-9
    cut = np.stack([full[:, 0] < x0 - eps, full[:, 1] > x1 + eps, full[:, 2] < y0 - eps, full[:, 3] > y1 + eps], axis=1)
    return Layout((x0, x1, y0, y1), full, box, horiz, cut)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _crop_uv(a: np.ndarray, b: np.ndarray, crop: Crop, horizontal: bool, flip: bool):
    """UV of paver-local (a, b) in [0, 1]^2 (a along x, b along y) inside ``crop``.

    Always a rotation (no mirroring, so tangent frames stay right): by 0 or 180 degrees when the paver and
    the crop have the same orientation, by +-90 otherwise; ``flip`` adds 180 degrees.
    """
    if horizontal == crop.horizontal:
        p, q = (1 - a, 1 - b) if flip else (a, b)
    elif crop.horizontal:                                     # vertical paver, crop long along u
        p, q = (1 - b, a) if flip else (b, 1 - a)
    else:                                                     # horizontal paver, crop long along v
        p, q = (b, 1 - a) if flip else (1 - b, a)
    return crop.u0 + p * (crop.u1 - crop.u0), crop.v0 + q * (crop.v1 - crop.v0)


_SIDE_OUT = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])   # south, east, north, west
_SIDE_IN = -_SIDE_OUT


def build_paver(box, full, cut, horizontal: bool, crop: Crop, flip: bool, top_z: float, plane,
                spec: PaverSpec = PaverSpec()):
    """One paver as (points, counts, normals, st): top, chamfers and sides, every face with its own vertices.

    ``box`` is the part inside the rect, ``full`` the whole footprint (UVs follow the whole paver, so a cut
    one shows its part of the crop unstretched), ``cut`` = (west, east, south, north) sawn sides: those have no
    chamfer.  ``plane`` = (dz, sx, sy, cx, cy): the top plane z = top_z + dz + sx (x - cx) + sy (y - cy).
    """
    x0, x1, y0, y1 = (float(v) for v in box)
    dz, sx, sy, cx, cy = plane
    c = min(spec.chamfer, 0.3 * min(x1 - x0, y1 - y0))
    sawn = [bool(cut[2]), bool(cut[1]), bool(cut[3]), bool(cut[0])]               # south, east, north, west
    chamfered = [not s and c > 0 for s in sawn]
    cs = [c if f else 0.0 for f in chamfered]
    outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])                    # SW, SE, NE, NW
    inner = np.array([outer[i] + _SIDE_IN[i - 1] * cs[i - 1] + _SIDE_IN[i] * cs[i] for i in range(4)])
    drop = [c if (chamfered[i - 1] or chamfered[i]) else 0.0 for i in range(4)]   # outer corner i below the top

    def at(xy, d: float) -> list[float]:
        return [xy[0], xy[1], top_z + dz + sx * (xy[0] - cx) + sy * (xy[1] - cy) - d]

    polys: list[tuple[list[list[float]], str, int]] = []                           # vertices, kind, side
    polys.append(([at(inner[i], 0.0) for i in range(4)], "top", -1))
    for i in range(4):
        if chamfered[i]:
            n = (i + 1) % 4
            polys.append(([at(outer[i], c), at(outer[n], c), at(inner[n], 0.0), at(inner[i], 0.0)], "chamfer", i))
    for i in range(4):
        n = (i + 1) % 4
        if sawn[i]:
            profile = [(outer[i], drop[i])]
            if chamfered[i - 1]:
                profile.append((inner[i], 0.0))
            if chamfered[n]:
                profile.append((inner[n], 0.0))
            profile.append((outer[n], drop[n]))
            profile = [q for k, q in enumerate(profile)
                       if k == 0 or np.hypot(*(q[0] - profile[k - 1][0])) > 1e-9 or q[1] != profile[k - 1][1]]
        else:
            profile = [(outer[i], c), (outer[n], c)]
        poly = [at(outer[i], spec.thickness), at(outer[n], spec.thickness)] + [at(xy, d) for xy, d in profile[::-1]]
        polys.append((poly, "side", i))

    top_n = _unit(np.cross(np.subtract(polys[0][0][1], polys[0][0][0]), np.subtract(polys[0][0][2], polys[0][0][0])))
    pts, cnt, nrm = [], [], []
    for verts, kind, i in polys:
        v = np.array(verts)
        face_n = _unit(np.cross(v[1] - v[0], v[2] - v[0]))
        if kind == "chamfer":
            side_n = np.array([*_SIDE_OUT[i], 0.0])
            lower = _unit(face_n + spec.bevel_smooth * (side_n - face_n))
            upper = _unit(face_n + spec.bevel_smooth * (top_n - face_n))
            n = np.array([lower, lower, upper, upper])
        else:
            n = np.tile(face_n, (len(v), 1))
        pts.append(v)
        cnt.append(len(v))
        nrm.append(n)
    pts = np.concatenate(pts)
    fx0, fx1, fy0, fy1 = (float(v) for v in full)
    a = np.clip((pts[:, 0] - fx0) / (fx1 - fx0), 0.0, 1.0)
    b = np.clip((pts[:, 1] - fy0) / (fy1 - fy0), 0.0, 1.0)
    u, v = _crop_uv(a, b, crop, horizontal, flip)
    return pts, cnt, np.concatenate(nrm), np.stack([u, v], axis=1)


@dataclass
class BrickSet:
    """The built pavers: per-chunk mesh data plus what was randomised per paver."""
    layout: Layout
    crop_index: np.ndarray                   # (N,) index into the atlas
    flip: np.ndarray                         # (N,) bool: crop turned by 180 degrees
    tint: np.ndarray                         # (N,) int: colour class, an index into the brick tints
    dz: np.ndarray                           # (N,) height offset (m)
    slope: np.ndarray                        # (N, 2) tilt: dz/dx, dz/dy
    chunk: list[tuple[int, int, int]]        # (N,) mesh key of each paver: chunk x, chunk y, colour class
    spans: list[tuple[int, int]]             # (N,) vertex range [start, end) of each paver inside its mesh
    meshes: dict[tuple[int, int, int], tuple]    # mesh key -> (points, counts, indices, normals, st)


def build_bricks(layout: Layout, atlas: list[Crop], top_z: float, *, spec: PaverSpec = PaverSpec(),
                 seed: int = 0, chunk: float = 2.2, tints: int = len(BRICK_TINTS)) -> BrickSet:
    """Build every paver and merge them into one mesh per ``chunk`` x ``chunk`` m cell and colour class.

    Each paver draws one of ``tints`` colour classes at random (a material per class, see ``BRICK_TINTS``).
    """
    rng = np.random.default_rng(seed)
    n = len(layout)
    x0, _, y0, _ = layout.rect
    tilt = math.tan(math.radians(spec.tilt_deg))
    dz = rng.uniform(-spec.height_jitter, spec.height_jitter, n)
    slope = rng.uniform(-tilt, tilt, (n, 2))
    flip = rng.random(n) < 0.5
    tint = rng.integers(0, max(tints, 1), n)
    crop_index = np.zeros(n, int)
    near: dict[tuple[int, int], list[int]] = {}               # pavers laid so far, by cell: neighbours must differ
    cell = 0.25                                               # > the centre distance of touching pavers
    parts: dict[tuple[int, int, int], list] = {}
    keys, spans = [], []
    sizes: dict[tuple[int, int, int], int] = {}
    everything = list(range(len(atlas)))
    for i in range(n):
        bx0, bx1, by0, by1 = layout.box[i]
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        gx, gy = int(cx // cell), int(cy // cell)
        around = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m in near.get((gx + dx, gy + dy), ())]
        touching = set()
        for m in around:
            b = layout.box[m]
            gap = math.hypot(max(b[0] - bx1, bx0 - b[1], 0.0), max(b[2] - by1, by0 - b[3], 0.0))
            if gap < spec.joint + 1e-6:
                touching.add(int(crop_index[m]))
        # touching pavers must differ; the ones a little further out should too, if the atlas is big enough
        nearby = {int(crop_index[m]) for m in around}
        free = ([m for m in everything if m not in nearby] or [m for m in everything if m not in touching]
                or everything)
        crop_index[i] = free[int(rng.integers(len(free)))]
        near.setdefault((gx, gy), []).append(i)
        pts, cnt, nrm, st = build_paver(layout.box[i], layout.full[i], layout.cut[i], bool(layout.horizontal[i]),
                                        atlas[crop_index[i]], bool(flip[i]), top_z, (dz[i], *slope[i], cx, cy), spec)
        key = (int((cx - x0) // chunk), int((cy - y0) // chunk), int(tint[i]))
        start = sizes.get(key, 0)
        sizes[key] = start + len(pts)
        parts.setdefault(key, []).append((pts, cnt, nrm, st))
        keys.append(key)
        spans.append((start, start + len(pts)))
    meshes = {}
    for key, items in sorted(parts.items()):
        pts = np.concatenate([p for p, _, _, _ in items])
        cnt = np.array([k for _, c, _, _ in items for k in c])
        meshes[key] = (pts, cnt, np.arange(len(pts)), np.concatenate([nn for _, _, nn, _ in items]),
                       np.concatenate([s for _, _, _, s in items]))
    return BrickSet(layout, crop_index, flip, tint, dz, slope, keys, spans, meshes)


def _sand_bands(rect, top_z: float, tile: float, bands=JOINT_BANDS) -> list[tuple[str, float, tuple]]:
    """Sand surface at ``top_z`` as (name, tint, mesh data) rings: the rect minus the rect shrunk by each band."""
    x0, x1, y0, y1 = rect
    edges = [0.0] + [min(d, 0.5 * min(x1 - x0, y1 - y0)) for d, _ in bands]
    tints = [t for _, t in bands] + [1.0]
    out = []
    for k, tint in enumerate(tints):
        ax0, ax1, ay0, ay1 = x0 + edges[k], x1 - edges[k], y0 + edges[k], y1 - edges[k]
        if k == len(bands):                                    # the middle
            quads = [(ax0, ax1, ay0, ay1)]
        else:
            bx0, bx1, by0, by1 = x0 + edges[k + 1], x1 - edges[k + 1], y0 + edges[k + 1], y1 - edges[k + 1]
            quads = [(ax0, ax1, ay0, by0), (ax0, ax1, by1, ay1), (ax0, bx0, by0, by1), (bx1, ax1, by0, by1)]
        quads = [q for q in quads if q[1] - q[0] > 1e-6 and q[3] - q[2] > 1e-6]
        if quads:
            out.append(("PatioSand" if k == len(bands) else f"PatioSandEdge{k}", tint,
                        scene.merge(*[scene.box(*q, top_z - 0.01, top_z, tile, top_only=True) for q in quads])))
    return out


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------

def _clear(stage, root: str) -> None:
    """Remove what an earlier ``author_patio`` call wrote, so a new rect, chunk size or seed leaves no stale mesh."""
    for path in (f"{root}/Ground/PatioBricks", f"{root}/Ground/PatioSand"):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    looks = stage.GetPrimAtPath(f"{root}/Looks")
    for child in (looks.GetChildren() if looks else []):
        if child.GetName().startswith(("PatioBricks", "PatioSand")):
            stage.RemovePrim(child.GetPath())


def author_patio(stage, root: str, rect, top_z: float, cache_dir: str, *, seed: int = 0,
                 spec: PaverSpec = PaverSpec(), sand_depth: float = 0.008, sand_tint=SAND_TINT,
                 joint_bands=JOINT_BANDS, brick_tints=BRICK_TINTS, chunk: float = 2.2, collision: bool = False,
                 textures: dict | None = None, atlas: list[Crop] | None = None) -> dict:
    """Author the paver patio under ``{root}/Ground/PatioBricks`` and ``{root}/Ground/PatioSand``.

    rect = (x0, x1, y0, y1), ``top_z`` the paver top.  Materials (UsdPreviewSurface through
    ``scene.preview_surface``, the same textures and flip_green as ``scene.author_ground_materials``) go to
    ``{root}/Looks/PatioBricks{k}`` (one per entry of ``brick_tints``) and ``{root}/Looks/PatioSand*`` (one
    per sand tint band).  Calling it again replaces what an earlier call wrote.
    ``collision=True`` makes the brick meshes triangle-mesh colliders; by default they are visual only
    (keep a flat collider).  ``textures`` ({'pavers': {...}, 'sand': {...}}, e.g. ``manifest['textures']``)
    and ``atlas`` default to ``assets.material_textures`` and ``brick_atlas``; a material missing from
    ``textures`` is fetched into ``cache_dir``.  Returns a summary of counts.
    """
    from pxr import UsdGeom, UsdPhysics

    tex = dict(textures or {})
    for name in ("pavers", "sand"):
        if name not in tex:
            if not cache_dir:
                raise ValueError(f"no '{name}' textures: pass them in `textures` or give a `cache_dir` to fetch them")
            tex[name] = assets.material_textures(name, cache_dir)
    crops = atlas if atlas is not None else brick_atlas(tex["pavers"])
    brick_tints = tuple(brick_tints) or ((1.0, 1.0, 1.0),)
    layout = herringbone(rect, spec)
    bricks = build_bricks(layout, crops, top_z, spec=spec, seed=seed, chunk=chunk, tints=len(brick_tints))

    _clear(stage, root)
    UsdGeom.Xform.Define(stage, f"{root}/Ground")
    UsdGeom.Xform.Define(stage, f"{root}/Ground/PatioBricks")
    UsdGeom.Xform.Define(stage, f"{root}/Ground/PatioSand")

    def material(path: str, name: str, mult):
        m, t = assets.MATERIALS[name], tex[name]
        tint = tuple(float(v) for v in np.array(m.tint, float) * np.array(mult, float))
        return scene.preview_surface(stage, path, albedo=t.get("albedo"), orm=t.get("orm"), normal=t.get("normal"),
                                     roughness_tex=t.get("roughness"), flip_green=m.flip_green,
                                     tint=tint, roughness_scale=m.roughness_scale)

    brick_names = [f"PatioBricks{k}" for k in range(len(brick_tints))]
    brick_mats = [material(f"{root}/Looks/{n}", "pavers", t) for n, t in zip(brick_names, brick_tints)]
    triangles = 0
    for (ix, iy, k), data in bricks.meshes.items():
        prim = scene._mesh(stage, f"{root}/Ground/PatioBricks/Bricks_{ix}_{iy}_{k}", data, brick_mats[k])
        triangles += int(np.sum(data[1] - 2))
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)

    sand_names = []
    for name, tint, data in _sand_bands(rect, top_z - sand_depth, assets.MATERIALS["sand"].tile_m, joint_bands):
        mat = material(f"{root}/Looks/{name}", "sand", np.array(sand_tint, float) * tint)
        scene._mesh(stage, f"{root}/Ground/PatioSand/{name.removeprefix('PatioSand') or 'Joints'}", data, mat)
        sand_names.append(name)
        triangles += int(np.sum(data[1] - 2))

    x0, x1, y0, y1 = (float(v) for v in rect)
    area = float(((layout.box[:, 1] - layout.box[:, 0]) * (layout.box[:, 3] - layout.box[:, 2])).sum())
    return {
        "bricks": len(layout),
        "cut_bricks": int(layout.cut.any(axis=1).sum()),
        "crops": len(crops),
        "brick_meshes": len(bricks.meshes),
        "sand_meshes": len(sand_names),
        "triangles": triangles,
        "materials": brick_names + sand_names,
        "paver_area": area,
        "rect_area": (x1 - x0) * (y1 - y0),
        "collision": bool(collision),
        "seed": seed,
    }


def hide_flat_patio(stage, root: str) -> bool:
    """Make the flat textured ``{root}/Ground/Patio`` box invisible (it keeps working as a collider).

    Its top is at the paver top, so left visible it would z-fight the bricks.  False if it is not there.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(f"{root}/Ground/Patio")
    if not prim or not prim.IsA(UsdGeom.Mesh):
        return False
    UsdGeom.Imageable(prim).MakeInvisible()
    return True
