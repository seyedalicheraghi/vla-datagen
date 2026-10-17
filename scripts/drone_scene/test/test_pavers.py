"""Offline checks of the 3D herringbone pavers (numpy + usd-core; the real-texture checks need the asset cache).

    python scripts/drone_scene/test/test_pavers.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    HAVE_PXR = True
except ImportError:  # pragma: no cover
    HAVE_PXR = False

from drone_scene import assets, pavers, scene  # noqa: E402

CACHE = os.environ.get("DRONE_SCENE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "drone_scene"))
PAVERS_DIR = os.path.join(CACHE, "NVIDIA", "Materials", "Base", "Masonry", "Brick_Pavers")
REAL = {"albedo": os.path.join(PAVERS_DIR, "Brick_Pavers_BaseColor.png"), "orm": os.path.join(PAVERS_DIR, "Brick_Pavers_ORM.png"),
        "normal": os.path.join(PAVERS_DIR, "Brick_Pavers_N.png")}
HAVE_TEXTURES = os.path.exists(REAL["orm"]) and os.path.exists(REAL["albedo"])
RECT = scene.SceneCfg().patio
SPEC = pavers.PaverSpec()


def synthetic_atlas() -> list[pavers.Crop]:
    """Square-texture crops, 2:1 in UV space, half of them vertical; no texture needed."""
    out = []
    for i in range(12):
        u0, v0 = 0.05 + 0.07 * (i % 6), 0.1 + 0.4 * (i // 6)
        out.append(pavers.Crop(u0, v0, u0 + 0.06, v0 + 0.03, True) if i % 2 == 0 else
                   pavers.Crop(u0, v0, u0 + 0.03, v0 + 0.06, False))
    return out


def fake_textures(tmp: str) -> dict:
    return {n: {"albedo": f"{tmp}/{n}_c.png", "orm": f"{tmp}/{n}_orm.png", "normal": f"{tmp}/{n}_n.png"}
            for n in ("pavers", "sand")}


def split_polys(pts, cnt):
    out, k = [], 0
    for c in cnt:
        out.append(slice(k, k + c))
        k += c
    return out


def synthetic_ao() -> np.ndarray:
    """128 rows x 192 columns: joints 228, bricks 255; three whole bricks, one touching the border."""
    ao = np.full((128, 192), 228, np.uint8)
    ao[10:40, 20:80] = 255          # horizontal, near the TOP of the image
    ao[80:110, 20:80] = 255         # horizontal, lower
    ao[60:120, 100:130] = 255       # vertical
    ao[0:30, 150:192] = 255         # cut by the border: not a whole brick
    return ao


class TestComponents(unittest.TestCase):
    def test_u_shape_is_one_component(self):
        m = np.zeros((8, 10), bool)
        m[1:7, 1:3] = True
        m[1:7, 6:8] = True
        m[5:7, 1:8] = True              # the two bars only join on the bottom rows
        (box,) = pavers.component_boxes(m)
        self.assertEqual(box, (1, 1, 8, 7, int(m.sum())))

    def test_separate_blobs_and_diagonals_not_merged(self):
        m = np.zeros((6, 6), bool)
        m[0, 0] = m[1, 1] = True        # 4-connectivity: diagonal neighbours are separate
        m[3:5, 3:6] = True
        boxes = sorted(pavers.component_boxes(m))
        self.assertEqual(boxes, [(0, 0, 1, 1, 1), (1, 1, 2, 2, 1), (3, 3, 6, 5, 6)])


class TestAtlasSynthetic(unittest.TestCase):
    def test_v_is_up_and_sizes(self):
        crops = pavers.segment_bricks(synthetic_ao())
        self.assertEqual(len(crops), 3, "the brick cut by the border is dropped")
        w, h = 192, 128
        # brick centres (x, y in image pixels, y down), long side horizontal or not; v = 1 - row / h
        for cx, cy, horizontal in ((50, 25, True), (50, 95, True), (115, 90, False)):
            hits = [c for c in crops if c.horizontal == horizontal and
                    abs(0.5 * (c.u0 + c.u1) * w - cx) < 1.0 and abs((1 - 0.5 * (c.v0 + c.v1)) * h - cy) < 1.0]
            self.assertEqual(len(hits), 1, f"crop for the brick at ({cx}, {cy}): {crops}")
            c = hits[0]
            pw, ph = (c.u1 - c.u0) * w, (c.v1 - c.v0) * h
            self.assertAlmostEqual(max(pw, ph) / min(pw, ph), 2.0, places=6, msg="exactly 2:1")
            # the 60 x 30 brick minus the 8 px margin: 28 x 14 (a pixel or two less where the blur ate the edge)
            self.assertTrue(22 <= max(pw, ph) <= 28.01 and 11 <= min(pw, ph) <= 14.01, (pw, ph))
        top = max(crops, key=lambda c: c.v1)
        self.assertAlmostEqual(0.5 * (top.v0 + top.v1), 1 - 25 / h, delta=0.02, msg="the top brick has the HIGH v")

    def test_crops_land_on_brick_pixels_with_v_flipped(self):
        ao = synthetic_ao()
        h, w = ao.shape
        for c in pavers.segment_bricks(ao):
            rows = slice(int(round((1 - c.v1) * h)), int(round((1 - c.v0) * h)))
            cols = slice(int(round(c.u0 * w)), int(round(c.u1 * w)))
            self.assertTrue((ao[rows, cols] == 255).all(), c)

    def test_flat_occlusion_falls_back_to_luminance(self):
        ao = np.full((128, 192), 255, np.uint8)
        lum = np.where(synthetic_ao() == 255, 200, 90).astype(np.uint8)
        self.assertEqual(len(pavers.segment_bricks(ao, lum)), 3)

    def test_odd_sized_bricks_are_dropped(self):
        ao = synthetic_ao()
        ao[10:40, 150:170] = 255        # 20 x 30: wrong size
        ao[100:105, 140:180] = 255      # 40 x 5: wrong aspect
        self.assertEqual(len(pavers.segment_bricks(ao)), 3)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            orm = os.path.join(tmp, "Brick_Pavers_ORM.png")
            Image.fromarray(synthetic_ao()).save(orm)
            first = pavers.brick_atlas({"orm": orm})
            cache = os.path.join(tmp, "paver_atlas.json")
            self.assertTrue(os.path.exists(cache), "cached next to the textures")
            with open(cache, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)["crops"]), 3)
            real = pavers.segment_bricks
            pavers.segment_bricks = lambda *a, **k: self.fail("must come from the cache")
            try:
                self.assertEqual(pavers.brick_atlas({"orm": orm}), first)
            finally:
                pavers.segment_bricks = real
            Image.fromarray(np.full((16, 16), 255, np.uint8)).save(orm)           # other texture: cache invalid
            with self.assertRaises(RuntimeError):
                pavers.brick_atlas({"orm": orm})

    def test_cache_key_covers_every_tuning_constant(self):
        """Tuning any segmentation constant must re-segment instead of reusing a stale paver_atlas.json."""
        with tempfile.TemporaryDirectory() as tmp:
            orm = os.path.join(tmp, "Brick_Pavers_ORM.png")
            Image.fromarray(synthetic_ao()).save(orm)
            pavers.brick_atlas({"orm": orm})
            calls = []
            real = pavers.segment_bricks
            pavers.segment_bricks = lambda *a, **k: calls.append(1) or real(*a, **k)
            try:
                pavers.brick_atlas({"orm": orm})
                self.assertEqual(calls, [], "unchanged constants: cache hit")
                for name, value in (("ATLAS_VERSION", 99), ("ATLAS_ASPECT", 2.1), ("ATLAS_MARGIN_PX", 7),
                                    ("ATLAS_ERODE_PX", 1), ("ATLAS_SIZE_TOL", 0.2), ("ATLAS_LEVEL", 0.5),
                                    ("ATLAS_BLUR_PX", 1), ("ATLAS_SHAPE", (1.5, 2.6, 0.85))):
                    old = getattr(pavers, name)
                    setattr(pavers, name, value)
                    try:
                        calls.clear()
                        pavers.brick_atlas({"orm": orm}, use_cache=True)
                        self.assertEqual(calls, [1], f"{name} is not part of the cache key")
                    finally:
                        setattr(pavers, name, old)
                    pavers.brick_atlas({"orm": orm})                  # rewrite the cache for the next constant
            finally:
                pavers.segment_bricks = real


@unittest.skipUnless(HAVE_TEXTURES, "needs the mirrored Brick_Pavers textures (run prepare.py)")
class TestAtlasReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = pavers.brick_atlas(REAL)
        cls.ao = np.asarray(Image.open(REAL["orm"]).convert("RGB"))[..., 0]

    def test_count_rects_size_and_aspect(self):
        self.assertGreaterEqual(len(self.atlas), 40)          # 72 bricks per repeat, the ones on the border are cut
        h, w = self.ao.shape
        for c in self.atlas:
            self.assertTrue(0.0 <= c.u0 < c.u1 <= 1.0 and 0.0 <= c.v0 < c.v1 <= 1.0, c)
            pw, ph = (c.u1 - c.u0) * w, (c.v1 - c.v0) * h
            long_, short = max(pw, ph), min(pw, ph)
            self.assertAlmostEqual(long_ / short, 2.0, delta=0.04, msg=str(c))
            self.assertTrue(225 <= long_ <= 280 and 112 <= short <= 140, f"{c}: {pw:.0f} x {ph:.0f} px")
            self.assertEqual(c.horizontal, pw > ph)
        self.assertGreater(sum(c.horizontal for c in self.atlas), 10)
        self.assertGreater(sum(not c.horizontal for c in self.atlas), 10)

    def test_every_crop_is_inside_a_brick_not_in_a_joint(self):
        """v = 0 is the bottom of the image: the AO of the rows the crop's v range names is all brick."""
        h, w = self.ao.shape
        for c in self.atlas:
            rows = slice(int(np.floor((1 - c.v1) * h)), int(np.ceil((1 - c.v0) * h)))
            cols = slice(int(np.floor(c.u0 * w)), int(np.ceil(c.u1 * w)))
            region = self.ao[rows, cols]
            self.assertGreater(float(region.mean()), 252.0, c)
            self.assertLess(float((region < 247).mean()), 0.01, c)

    def test_flipping_v_would_be_wrong(self):
        h, w = self.ao.shape
        bad = 0
        for c in self.atlas:
            rows = slice(int(np.floor(c.v0 * h)), int(np.ceil(c.v1 * h)))       # the wrong convention
            cols = slice(int(np.floor(c.u0 * w)), int(np.ceil(c.u1 * w)))
            bad += float((self.ao[rows, cols] < 247).mean()) > 0.01
        self.assertGreater(bad, len(self.atlas) // 3, "the check above must be able to fail")

    def test_crops_are_distinct_bricks(self):
        a = np.array([c[:4] for c in self.atlas])
        ox = np.minimum(a[:, None, 2], a[None, :, 2]) - np.maximum(a[:, None, 0], a[None, :, 0])
        oy = np.minimum(a[:, None, 3], a[None, :, 3]) - np.maximum(a[:, None, 1], a[None, :, 1])
        overlap = (ox > 0) & (oy > 0)
        np.fill_diagonal(overlap, False)
        self.assertFalse(overlap.any())


class TestLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = pavers.herringbone(RECT, SPEC)
        cls.area = (RECT[1] - RECT[0]) * (RECT[3] - RECT[2])

    def test_spec(self):
        self.assertAlmostEqual(SPEC.length, 0.200)
        self.assertAlmostEqual(SPEC.joint, 0.003)
        self.assertAlmostEqual(2 * SPEC.width + SPEC.joint, SPEC.length, msg="herringbone: length = 2 width + joint")
        self.assertAlmostEqual(SPEC.width, 0.1, delta=0.002)

    def test_spec_defaults_are_pinned(self):
        """The look rests on these numbers (4 mm chamfers, 60 mm pavers, a hair of noise)."""
        self.assertEqual(SPEC.chamfer, 0.004)
        self.assertEqual(SPEC.thickness, 0.060)
        self.assertEqual(SPEC.height_jitter, 0.001)
        self.assertEqual(SPEC.tilt_deg, 0.25)
        self.assertEqual(SPEC.bevel_smooth, 0.5)

    def test_top_rise_bounds_the_noise(self):
        # 1 mm of height + 0.25 degrees over a half diagonal of 149 mm (length + width over two)
        self.assertAlmostEqual(SPEC.top_rise, 0.001 + math.tan(math.radians(0.25)) * 0.14925, places=9)
        self.assertAlmostEqual(SPEC.top_rise, 0.00165, delta=1e-5)
        self.assertEqual(pavers.PaverSpec(height_jitter=0.0, tilt_deg=0.0).top_rise, 0.0)

    def test_coverage_is_rect_minus_joints(self):
        top = ((self.L.box[:, 1] - self.L.box[:, 0]) * (self.L.box[:, 3] - self.L.box[:, 2])).sum()
        expected = self.area * SPEC.length * SPEC.width / (SPEC.pitch * (SPEC.length + SPEC.joint))
        self.assertAlmostEqual(top / expected, 1.0, delta=0.03)
        self.assertLess(top, self.area)
        self.assertAlmostEqual(1 - top / self.area, 0.044, delta=0.015, msg="~4 % of a 200 x 100 herringbone is joint")

    def test_all_bricks_inside_the_rect(self):
        x0, x1, y0, y1 = RECT
        b = self.L.box
        self.assertTrue((b[:, 0] >= x0 - 1e-9).all() and (b[:, 1] <= x1 + 1e-9).all())
        self.assertTrue((b[:, 2] >= y0 - 1e-9).all() and (b[:, 3] <= y1 + 1e-9).all())
        self.assertTrue(((b[:, 1] - b[:, 0]) > 0).all() and ((b[:, 3] - b[:, 2]) > 0).all())

    def test_no_two_bricks_overlap_or_touch(self):
        b = self.L.box
        gx = np.maximum(b[None, :, 0] - b[:, None, 1], b[:, None, 0] - b[None, :, 1])
        gy = np.maximum(b[None, :, 2] - b[:, None, 3], b[:, None, 2] - b[None, :, 3])
        dist = np.hypot(np.maximum(gx, 0), np.maximum(gy, 0))
        np.fill_diagonal(dist, 1.0)
        self.assertGreaterEqual(dist.min(), SPEC.joint - 1e-9, "closest pavers are one joint apart")
        self.assertAlmostEqual(dist.min(), SPEC.joint, places=9)
        rng = np.random.default_rng(1)
        pts = np.stack([rng.uniform(RECT[0], RECT[1], 20000), rng.uniform(RECT[2], RECT[3], 20000)], axis=1)
        inside = ((pts[:, None, 0] >= b[None, :, 0]) & (pts[:, None, 0] <= b[None, :, 1]) &
                  (pts[:, None, 1] >= b[None, :, 2]) & (pts[:, None, 1] <= b[None, :, 3])).sum(axis=1)
        self.assertLessEqual(int(inside.max()), 1)
        self.assertAlmostEqual(float((inside == 1).mean()), 0.956, delta=0.02, msg="sample points hit pavers, not joints")

    def test_herringbone_not_stack_bond(self):
        h = self.L.horizontal
        self.assertAlmostEqual(float(h.mean()), 0.5, delta=0.03)
        full = self.L.full
        size = np.stack([full[:, 1] - full[:, 0], full[:, 3] - full[:, 2]], axis=1)
        np.testing.assert_allclose(size[h], np.tile([SPEC.length, SPEC.width], (int(h.sum()), 1)), atol=1e-9)
        np.testing.assert_allclose(size[~h], np.tile([SPEC.width, SPEC.length], (int((~h).sum()), 1)), atol=1e-9)
        # the pattern repeats every 4 pitches along x and y
        inner = ~self.L.cut.any(axis=1)
        key = {(round(x / 1e-6), round(y / 1e-6), bool(o)) for x, y, o in zip(full[inner, 0], full[inner, 2], h[inner])}
        for dx, dy in ((4 * SPEC.pitch, 0.0), (0.0, 4 * SPEC.pitch), (2 * SPEC.pitch, 2 * SPEC.pitch)):
            moved = [(round((x + dx) / 1e-6), round((y + dy) / 1e-6), bool(o))
                     for x, y, o in zip(full[inner, 0], full[inner, 2], h[inner])]
            hit = sum(m in key for m in moved) / len(moved)
            self.assertGreater(hit, 0.85, f"shift ({dx:.4f}, {dy:.4f})")

    def test_cut_flags_match_the_geometry(self):
        x0, x1, y0, y1 = RECT
        for i in range(len(self.L)):
            f, b, c = self.L.full[i], self.L.box[i], self.L.cut[i]
            self.assertEqual(bool(c[0]), f[0] < x0 - 1e-9)
            self.assertEqual(bool(c[1]), f[1] > x1 + 1e-9)
            self.assertEqual(bool(c[2]), f[2] < y0 - 1e-9)
            self.assertEqual(bool(c[3]), f[3] > y1 + 1e-9)
            for side in range(4):
                if not c[side]:
                    self.assertAlmostEqual(b[side], f[side], places=12)
        self.assertGreater(int(self.L.cut.any(axis=1).sum()), 30)

    def test_no_gap_at_the_border(self):
        """Walking along each edge of the rect, the gaps between pavers are one joint (or a tiny sliver)."""
        x0, x1, y0, y1 = RECT
        b = self.L.box
        worst = 0.0
        for axis, edge, lo, hi in ((0, x0, y0, y1), (0, x1, y0, y1), (1, y0, x0, x1), (1, y1, x0, x1)):
            a0, a1 = (0, 1) if axis == 0 else (2, 3)           # coordinate across the edge
            c0, c1 = (2, 3) if axis == 0 else (0, 1)           # coordinate along the edge
            touching = b[(b[:, a0] <= edge + 1e-9) & (b[:, a1] >= edge - 1e-9)]
            spans = sorted(zip(touching[:, c0], touching[:, c1]))
            self.assertAlmostEqual(spans[0][0], lo, places=9)
            self.assertAlmostEqual(max(s[1] for s in spans), hi, delta=pavers.MIN_SLIVER + 1e-9)
            for (_, e), (s, _) in zip(spans, spans[1:]):
                worst = max(worst, s - e)
        self.assertLessEqual(worst, SPEC.joint + pavers.MIN_SLIVER + 1e-9)

    def test_other_rects(self):
        for rect in ((0.0, 0.5, 0.0, 0.3), (-0.0123, 0.7321, 0.4, 1.1), (1.0, 1.03, 2.0, 2.2)):
            L = pavers.herringbone(rect, SPEC)
            self.assertGreater(len(L), 0)
            area = ((L.box[:, 1] - L.box[:, 0]) * (L.box[:, 3] - L.box[:, 2])).sum()
            self.assertLess(area, (rect[1] - rect[0]) * (rect[3] - rect[2]) + 1e-9)
            self.assertGreater(area, 0.85 * (rect[1] - rect[0]) * (rect[3] - rect[2]))


def outward(i: int) -> np.ndarray:
    return np.array([*pavers._SIDE_OUT[i], 0.0])


def one_brick(cut=(False, False, False, False), horizontal=True, crop=None, flip=False, smooth=0.5, plane=None):
    spec = pavers.PaverSpec(bevel_smooth=smooth)
    full = (0.0, spec.length, 0.0, spec.width) if horizontal else (0.0, spec.width, 0.0, spec.length)
    box = list(full)
    for side, (idx, sign) in enumerate(((0, 1), (1, -1), (2, 1), (3, -1))):       # west, east, south, north
        if cut[side]:
            box[idx] += sign * 0.03
    crop = crop or pavers.Crop(0.2, 0.3, 0.2 + 0.06, 0.3 + 0.03, True)
    plane = plane or (0.0, 0.0, 0.0, 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3]))
    pts, cnt, nrm, st = pavers.build_paver(box, full, cut, horizontal, crop, flip, 0.0, plane, spec)
    return spec, box, pts, cnt, nrm, st


class TestPaverGeometry(unittest.TestCase):
    def test_faces_of_an_uncut_paver(self):
        spec, box, pts, cnt, nrm, st = one_brick()
        self.assertEqual(list(cnt), [4] * 9, "top, four chamfers, four sides")
        top = pts[:4]
        np.testing.assert_allclose(top[:, 2], 0.0, atol=1e-12)
        self.assertAlmostEqual(top[:, 0].min(), spec.chamfer)
        self.assertAlmostEqual(top[:, 0].max(), spec.length - spec.chamfer)
        self.assertAlmostEqual(pts[:, 2].min(), -spec.thickness)
        # the chamfers are 4 mm wide and 4 mm tall
        cham = pts[4:20]
        self.assertAlmostEqual(cham[:, 2].min(), -spec.chamfer)
        self.assertAlmostEqual(pts[:, 0].min(), 0.0)
        self.assertAlmostEqual(pts[:, 0].max(), spec.length)

    def test_chamfer_normals_are_between_top_and_side(self):
        for smooth in (0.0, 0.5, 1.0):
            spec, box, pts, cnt, nrm, st = one_brick(smooth=smooth)
            np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-9)
            np.testing.assert_allclose(nrm[:4], [[0, 0, 1]] * 4, atol=1e-9)
            for i in range(4):
                n = nrm[4 + 4 * i:8 + 4 * i]
                side = outward(i)
                span = np.cross([0, 0, 1], side)                       # normal of the plane of both
                for v in n:
                    self.assertGreaterEqual(v[2], -1e-12, "never facing down")
                    self.assertGreaterEqual(v @ side, -1e-12, "never facing inwards")
                    self.assertGreater(v[2] + v @ side, 0.7, "not degenerate")
                    self.assertAlmostEqual(float(v @ span), 0.0, places=9, msg="in the plane of top and side normals")
                if smooth == 0.0:
                    np.testing.assert_allclose(n, [(side + [0, 0, 1]) / math.sqrt(2)] * 4, atol=1e-9)
                else:
                    ang = np.degrees(np.arccos(np.clip(n[:, 2], -1, 1)))                # lower edge: towards the side
                    self.assertGreater(ang[0], ang[3], "normals turn from the top towards the side down the chamfer")
            sides = nrm[20:]
            for i in range(4):
                np.testing.assert_allclose(sides[4 * i:4 * i + 4], [outward(i)] * 4, atol=1e-9)

    def test_vertex_normals_match_winding(self):
        spec, box, pts, cnt, nrm, st = one_brick(plane=(0.0007, 0.003, -0.002, 0.1, 0.05))
        for sl in split_polys(pts, cnt):
            v = pts[sl]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            n /= np.linalg.norm(n)
            self.assertGreater(float((nrm[sl] @ n).min()), 0.9, "outward")

    def test_watertight_except_the_bottom(self):
        cases = [(False,) * 4, (True, False, False, False), (False, True, False, False), (False, False, True, False),
                 (False, False, False, True), (True, False, True, False), (False, True, False, True),
                 (True, True, True, True), (True, True, False, False)]
        for cut in cases:
            spec, box, pts, cnt, nrm, st = one_brick(cut=cut, plane=(0.0004, 0.002, -0.003, 0.1, 0.05))
            edges: dict = {}
            for sl in split_polys(pts, cnt):
                v = np.round(pts[sl], 9)
                for k in range(len(v)):
                    a, b = tuple(v[k]), tuple(v[(k + 1) % len(v)])
                    edges[(a, b)] = edges.get((a, b), 0) + 1
            bottom = lambda p: p[2] < -0.05  # noqa: E731
            for (a, b), n in edges.items():
                self.assertEqual(n, 1, f"cut={cut}: directed edge used twice")
                if bottom(a) and bottom(b):
                    continue
                self.assertEqual(edges.get((b, a)), 1, f"cut={cut}: open edge {a} -> {b}")

    def test_sawn_side_has_no_chamfer(self):
        spec, box, pts, cnt, nrm, st = one_brick(cut=(True, False, False, False))
        self.assertEqual(len(cnt), 1 + 3 + 4)
        self.assertAlmostEqual(pts[:4, 0].min(), box[0], msg="the top reaches the sawn edge")
        west = pts[pts[:, 0] < box[0] + 1e-9]
        self.assertTrue(len(west) > 0 and np.allclose(west[:, 0], box[0]))
        self.assertAlmostEqual(float(west[:, 2].max()), 0.0, msg="a sawn face runs up to the top")
        corner = (True, False, True, False)
        self.assertEqual(len(one_brick(cut=corner)[3]), 1 + 2 + 4)

    def test_sliver_is_still_valid(self):
        spec = pavers.PaverSpec()
        full = (0.0, 0.2, 0.0, 0.0985)
        box = (0.0, 0.003, 0.0, 0.0985)                     # sawn to 3 mm
        pts, cnt, nrm, st = pavers.build_paver(box, full, (False, True, False, False), True,
                                               pavers.Crop(0.1, 0.1, 0.16, 0.13, True), False, 0.0,
                                               (0, 0, 0, 0.0015, 0.05), spec)
        self.assertTrue(np.isfinite(pts).all() and np.isfinite(nrm).all() and np.isfinite(st).all())
        self.assertLessEqual(pts[:, 0].max(), 0.003 + 1e-12)

    def test_uvs_stay_in_the_crop_and_are_a_rotation(self):
        h_crop = pavers.Crop(0.2, 0.3, 0.26, 0.33, True)
        v_crop = pavers.Crop(0.5, 0.1, 0.53, 0.16, False)
        for horizontal in (True, False):
            for crop in (h_crop, v_crop):
                for flip in (False, True):
                    spec, box, pts, cnt, nrm, st = one_brick(horizontal=horizontal, crop=crop, flip=flip)
                    lo, hi = st.min(axis=0), st.max(axis=0)
                    # the outer edge of the chamfers/sides IS the crop's border: the whole crop is used, nothing else
                    np.testing.assert_allclose([*lo, *hi], [crop.u0, crop.v0, crop.u1, crop.v1], atol=1e-9)
                    top = pts[:4]
                    dx, dy = top[1] - top[0], top[3] - top[0]
                    ju = np.array([st[1] - st[0], st[3] - st[0]]).T / np.array([dx[0], dy[1]])   # d(u,v)/d(x,y)
                    self.assertGreater(np.linalg.det(ju), 0.0, "a rotation, not a mirror")
                    sx, sy = np.linalg.norm(ju[:, 0]), np.linalg.norm(ju[:, 1])
                    self.assertAlmostEqual(sx / sy, 1.0, delta=0.03, msg="no UV stretching")
                    # the long side of the paver runs along the long side of the crop
                    long_axis = 0 if horizontal else 1
                    uv_long = np.abs(ju[:, long_axis])
                    self.assertEqual(bool(uv_long[0] > uv_long[1]), crop.horizontal)

    def test_cut_paver_shows_its_part_of_the_crop_unstretched(self):
        crop = pavers.Crop(0.2, 0.3, 0.26, 0.33, True)
        _, box, pts, cnt, nrm, st = one_brick(cut=(False, True, False, False), crop=crop)   # east 30 mm sawn off
        lo, hi = st.min(axis=0), st.max(axis=0)
        self.assertAlmostEqual(lo[0], crop.u0, places=9)
        self.assertAlmostEqual(hi[0], crop.u0 + (crop.u1 - crop.u0) * (0.2 - 0.03) / 0.2, places=9)
        self.assertAlmostEqual(lo[1], crop.v0, places=9)
        self.assertAlmostEqual(hi[1], crop.v1, places=9)


class TestBricks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = synthetic_atlas()
        cls.L = pavers.herringbone(RECT, SPEC)
        cls.B = pavers.build_bricks(cls.L, cls.atlas, 0.0, seed=3)

    def test_chunks_and_counts(self):
        meshes = self.B.meshes
        chunks = {k[:2] for k in meshes}
        self.assertTrue(2 <= len(chunks) <= 6, "a few chunks")
        self.assertEqual({k[2] for k in meshes}, set(range(len(pavers.BRICK_TINTS))), "one mesh per colour class")
        self.assertLessEqual(len(meshes), len(chunks) * len(pavers.BRICK_TINTS))
        faces = sum(len(m[1]) for m in meshes.values())
        chamfers = (~self.L.cut).sum()                       # one chamfer per uncut side (small slivers aside)
        self.assertEqual(faces, len(self.L) * 5 + int(chamfers))
        for pts, cnt, idx, nrm, st in meshes.values():
            self.assertEqual(len(pts), len(nrm))
            self.assertEqual(len(pts), len(st))
            self.assertEqual(int(cnt.sum()), len(idx))
            self.assertEqual(int(idx.max()), len(pts) - 1)
        self.assertEqual(sum(len(m[0]) for m in meshes.values()), sum(e - s for s, e in self.B.spans))

    def test_everything_inside_the_rect(self):
        x0, x1, y0, y1 = RECT
        for pts, *_ in self.B.meshes.values():
            self.assertGreaterEqual(float(pts[:, 0].min()), x0 - 1e-9)
            self.assertLessEqual(float(pts[:, 0].max()), x1 + 1e-9)
            self.assertGreaterEqual(float(pts[:, 1].min()), y0 - 1e-9)
            self.assertLessEqual(float(pts[:, 1].max()), y1 + 1e-9)

    def test_uvs_of_each_brick_are_inside_its_crop(self):
        for i in range(len(self.L)):
            s, e = self.B.spans[i]
            st = self.B.meshes[self.B.chunk[i]][4][s:e]
            c = self.atlas[self.B.crop_index[i]]
            self.assertGreaterEqual(float(st[:, 0].min()), c.u0 - 1e-9)
            self.assertLessEqual(float(st[:, 0].max()), c.u1 + 1e-9)
            self.assertGreaterEqual(float(st[:, 1].min()), c.v0 - 1e-9)
            self.assertLessEqual(float(st[:, 1].max()), c.v1 + 1e-9)

    def test_neighbouring_bricks_use_different_crops(self):
        b = self.L.box
        gx = np.maximum(b[None, :, 0] - b[:, None, 1], b[:, None, 0] - b[None, :, 1])
        gy = np.maximum(b[None, :, 2] - b[:, None, 3], b[:, None, 2] - b[None, :, 3])
        dist = np.hypot(np.maximum(gx, 0), np.maximum(gy, 0))
        i, j = np.nonzero(np.triu(dist < SPEC.joint + 1e-6, 1))
        self.assertGreater(len(i), 1000)
        self.assertTrue((self.B.crop_index[i] != self.B.crop_index[j]).all())
        self.assertEqual(len(set(self.B.crop_index.tolist())), len(self.atlas), "every crop gets used")
        self.assertGreater(int(self.B.flip.sum()), len(self.L) // 3)

    def test_height_and_tilt_noise(self):
        self.assertLessEqual(float(np.abs(self.B.dz).max()), SPEC.height_jitter)
        self.assertGreater(float(np.abs(self.B.dz).max()), 0.8 * SPEC.height_jitter)
        tilt = math.tan(math.radians(SPEC.tilt_deg))
        self.assertLessEqual(float(np.abs(self.B.slope).max()), tilt)
        self.assertGreater(float(np.abs(self.B.slope).max()), 0.8 * tilt)
        top = np.concatenate([m[0][:, 2][m[3][:, 2] > 0.99] for m in self.B.meshes.values()])
        self.assertLessEqual(float(np.abs(top).max()), 0.0017, "tops stay within ~1.7 mm of the nominal height")
        self.assertLessEqual(float(top.max()), SPEC.top_rise + 1e-12)
        self.assertGreater(float(np.abs(top).max()), 0.001)

    def test_colour_classes(self):
        tint = self.B.tint
        counts = np.bincount(tint, minlength=len(pavers.BRICK_TINTS))
        self.assertEqual(len(tint), len(self.L))
        self.assertTrue((counts > 0.15 * len(self.L)).all(), f"every class is common: {counts}")
        for i in range(len(self.L)):                       # the mesh a paver lands in has its class in the key
            self.assertEqual(self.B.chunk[i][2], int(tint[i]))
        one = pavers.build_bricks(self.L, self.atlas, 0.0, seed=3, tints=1)
        self.assertEqual({k[2] for k in one.meshes}, {0})
        self.assertEqual(sum(len(m[0]) for m in one.meshes.values()), sum(len(m[0]) for m in self.B.meshes.values()))

    def test_tint_table(self):
        t = np.array(pavers.BRICK_TINTS)
        lum = t @ np.array([0.2126, 0.7152, 0.0722])
        self.assertAlmostEqual(float(lum.mean()), 1.0, delta=0.03, msg="the patio keeps the texture's overall brightness")
        self.assertGreater(float(lum.max() / lum.min()), 1.2, "a visible brick-to-brick spread")
        self.assertLess(float(t.max()), 1.25, "no clipped albedo")

    def test_top_area_matches_the_layout(self):
        """Projected area of everything facing up (top + chamfers) is the paver footprint."""
        total = 0.0
        for pts, cnt, idx, nrm, st in self.B.meshes.values():
            k = 0
            for c in cnt:
                v = pts[k:k + c]
                if nrm[k, 2] > 0.1:
                    total += 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
                k += c
        footprint = ((self.L.box[:, 1] - self.L.box[:, 0]) * (self.L.box[:, 3] - self.L.box[:, 2])).sum()
        self.assertAlmostEqual(total / footprint, 1.0, delta=0.002)
        expected = (RECT[1] - RECT[0]) * (RECT[3] - RECT[2]) * SPEC.length * SPEC.width / (SPEC.pitch * (SPEC.length + SPEC.joint))
        self.assertAlmostEqual(total / expected, 1.0, delta=0.03)

    def test_seed(self):
        again = pavers.build_bricks(self.L, self.atlas, 0.0, seed=3)
        other = pavers.build_bricks(self.L, self.atlas, 0.0, seed=4)
        for k, m in self.B.meshes.items():
            np.testing.assert_array_equal(m[0], again.meshes[k][0])
            np.testing.assert_array_equal(m[4], again.meshes[k][4])
        self.assertFalse(np.array_equal(self.B.crop_index, other.crop_index))

    def test_top_z_offsets_everything(self):
        up = pavers.build_bricks(self.L, self.atlas, 0.25, seed=3)
        for k, m in self.B.meshes.items():
            np.testing.assert_allclose(up.meshes[k][0][:, 2] - m[0][:, 2], 0.25, atol=1e-12)

    def test_sand_surface_under_the_whole_patio(self):
        bands = pavers._sand_bands(RECT, -0.008, 0.8)
        self.assertEqual([n for n, _, _ in bands], [f"PatioSandEdge{k}" for k in range(len(pavers.JOINT_BANDS))] + ["PatioSand"])
        total = 0.0
        for _, tint, (pts, cnt, idx, nrm, st) in bands:
            np.testing.assert_allclose(pts[:, 2], -0.008)
            np.testing.assert_allclose(nrm, [[0, 0, 1]] * len(nrm))
            self.assertEqual(set(cnt.tolist()), {4})
            for k in range(0, len(pts), 4):
                q = pts[k:k + 4]
                total += (q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min())
        self.assertAlmostEqual(total, (RECT[1] - RECT[0]) * (RECT[3] - RECT[2]), places=9, msg="rings tile the rect exactly")
        tints = [t for _, t, _ in bands]
        self.assertEqual(tints, [t for _, t in pavers.JOINT_BANDS] + [1.0])
        self.assertTrue(all(a < b for a, b in zip(tints, tints[1:])), "darker towards the curb")
        self.assertLess(max(b / a for a, b in zip(tints, tints[1:])), 1.2, "small steps, no visible edge in a joint")
        # one texture across the rings: UV = world xy / tile
        pts, _, _, _, st = bands[0][2]
        np.testing.assert_allclose(st, pts[:, :2] / 0.8)


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestUsd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.stage = Usd.Stage.CreateInMemory()
        cls.root = "/World/Scenery"
        cls.atlas = synthetic_atlas()
        cls.summary = pavers.author_patio(cls.stage, cls.root, RECT, 0.0, cls.tmp, seed=5,
                                          textures=fake_textures(cls.tmp), atlas=cls.atlas)

    def bricks(self):
        return [p for p in self.stage.GetPrimAtPath(f"{self.root}/Ground/PatioBricks").GetChildren()]

    def sand(self):
        return [p for p in self.stage.GetPrimAtPath(f"{self.root}/Ground/PatioSand").GetChildren()]

    def test_summary_counts(self):
        s = self.summary
        L = pavers.herringbone(RECT, SPEC)
        self.assertEqual(s["bricks"], len(L))
        self.assertEqual(s["cut_bricks"], int(L.cut.any(axis=1).sum()))
        self.assertGreater(s["cut_bricks"], 0)
        self.assertEqual(s["crops"], len(self.atlas))
        self.assertEqual(s["brick_meshes"], len(self.bricks()))
        self.assertEqual(s["sand_meshes"], len(self.sand()))
        self.assertEqual(s["sand_meshes"], len(pavers.JOINT_BANDS) + 1)
        self.assertEqual(s["materials"], [f"PatioBricks{k}" for k in range(len(pavers.BRICK_TINTS))] +
                         [f"PatioSandEdge{k}" for k in range(len(pavers.JOINT_BANDS))] + ["PatioSand"])
        self.assertFalse(s["collision"])
        self.assertEqual(s["seed"], 5)
        tris = 0
        for p in self.bricks() + self.sand():
            tris += sum(c - 2 for c in UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
        self.assertEqual(s["triangles"], tris)
        self.assertGreater(s["triangles"], 10000)
        self.assertAlmostEqual(s["paver_area"] / s["rect_area"], 0.9565, delta=0.01)

    def test_meshes_have_st_normals_and_materials(self):
        for prim in self.bricks() + self.sand():
            self.assertTrue(prim.IsA(UsdGeom.Mesh), prim.GetPath())
            mesh = UsdGeom.Mesh(prim)
            n = len(mesh.GetPointsAttr().Get())
            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            self.assertTrue(st and st.GetInterpolation() == UsdGeom.Tokens.vertex, prim.GetPath())
            self.assertEqual(str(st.GetTypeName()), "texCoord2f[]")
            self.assertEqual(len(st.Get()), n)
            self.assertEqual(len(mesh.GetNormalsAttr().Get()), n)
            self.assertEqual(mesh.GetNormalsInterpolation(), UsdGeom.Tokens.vertex)
            self.assertEqual(mesh.GetSubdivisionSchemeAttr().Get(), UsdGeom.Tokens.none)
            idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            self.assertEqual(int(idx.max()), n - 1)
            self.assertEqual(int(sum(mesh.GetFaceVertexCountsAttr().Get())), len(idx))
            ext = mesh.GetExtentAttr().Get()
            self.assertEqual(len(ext), 2)
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            self.assertTrue(bound, f"{prim.GetPath()} has no material")
            if "PatioBricks" in str(prim.GetPath()):
                k = int(prim.GetName().rsplit("_", 1)[1])
                self.assertEqual(str(bound.GetPath()), f"{self.root}/Looks/PatioBricks{k}", "the mesh's colour class")
            else:
                self.assertTrue(str(bound.GetPath()).startswith(f"{self.root}/Looks/PatioSand"))
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI), "visual only by default")

    def test_st_values_are_inside_the_atlas(self):
        lo = np.array([min(c.u0 for c in self.atlas), min(c.v0 for c in self.atlas)])
        hi = np.array([max(c.u1 for c in self.atlas), max(c.v1 for c in self.atlas)])
        for prim in self.bricks():
            st = np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get())
            self.assertTrue((st.min(axis=0) >= lo - 1e-6).all() and (st.max(axis=0) <= hi + 1e-6).all())

    def test_sand_is_8mm_below_the_pavers_and_covers_the_rect(self):
        zs, area = [], 0.0
        for prim in self.sand():
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get())
            zs.append(pts[:, 2])
            cnt = mesh.GetFaceVertexCountsAttr().Get()
            self.assertEqual(set(cnt), {4})
            for k in range(0, len(pts), 4):
                q = pts[k:k + 4]
                area += (q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min())
        np.testing.assert_allclose(np.concatenate(zs), -0.008, atol=1e-6)
        self.assertAlmostEqual(area, (RECT[1] - RECT[0]) * (RECT[3] - RECT[2]), places=4)

    def test_material_networks_use_the_ground_textures(self):
        for name, key in (("PatioBricks0", "pavers"), ("PatioBricks3", "pavers"), ("PatioSand", "sand")):
            mat = UsdShade.Material(self.stage.GetPrimAtPath(f"{self.root}/Looks/{name}"))
            pbr = mat.ComputeSurfaceSource()[0]
            self.assertEqual(pbr.GetIdAttr().Get(), "UsdPreviewSurface")
            tex = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.root}/Looks/{name}/albedoTex"))
            self.assertEqual(tex.GetInput("file").Get().path, f"{self.tmp}/{key}_c.png")
            rough = pbr.GetInput("roughness").GetConnectedSources()[0][0]
            self.assertEqual(rough.sourceName, "g", "roughness from the ORM green channel")
            n = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.root}/Looks/{name}/normalTex"))
            self.assertEqual(n.GetInput("sourceColorSpace").Get(), "raw")
            scale, bias = n.GetInput("scale").Get(), n.GetInput("bias").Get()
            self.assertEqual(assets.MATERIALS[key].flip_green, True)
            np.testing.assert_allclose(list(scale), [2, -2, 2, 1])             # DirectX normal map: green flipped
            np.testing.assert_allclose(list(bias), [-1, 1, -1, 0])
            reader = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.root}/Looks/{name}/stReader"))
            self.assertEqual(reader.GetInput("varname").Get(), "st")

    def test_brick_materials_differ_only_by_tint(self):
        def net(k):
            base = f"{self.root}/Looks/PatioBricks{k}"
            albedo = UsdShade.Shader(self.stage.GetPrimAtPath(f"{base}/albedoTex"))
            files = [str(UsdShade.Shader(self.stage.GetPrimAtPath(f"{base}/{n}")).GetInput("file").Get().path)
                     for n in ("albedoTex", "ormTex", "normalTex")]
            return np.array(list(albedo.GetInput("scale").Get())[:3]), files
        base_tint = np.array(assets.MATERIALS["pavers"].tint)
        for k, t in enumerate(pavers.BRICK_TINTS):
            tint, files = net(k)
            np.testing.assert_allclose(tint, base_tint * t, rtol=1e-6)
            self.assertEqual(files, net(0)[1], "one set of textures")

    def test_no_transform2d_anywhere(self):
        for prim in self.stage.Traverse():
            if prim.IsA(UsdShade.Shader):
                self.assertNotEqual(UsdShade.Shader(prim).GetIdAttr().Get(), "UsdTransform2d", prim.GetPath())

    def test_joint_sand_darkens_towards_the_curb(self):
        def tint(name):
            tex = UsdShade.Shader(self.stage.GetPrimAtPath(f"{self.root}/Looks/{name}/albedoTex"))
            return np.array(list(tex.GetInput("scale").Get())[:3])
        rings = [tint(f"PatioSandEdge{k}") for k in range(len(pavers.JOINT_BANDS))] + [tint("PatioSand")]
        for a, b in zip(rings, rings[1:]):
            self.assertTrue((a < b).all())
        np.testing.assert_allclose(rings[-1], pavers.SAND_TINT)

    def test_stage_is_valid_and_saves(self):
        path = os.path.join(self.tmp, "patio.usda")
        self.stage.Export(path)
        again = Usd.Stage.Open(path)
        self.assertEqual(len(list(again.Traverse())), len(list(self.stage.Traverse())))
        UsdGeom.Xform(again.GetPrimAtPath(f"{self.root}/Ground"))
        for prim in again.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                self.assertTrue(UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited)

    def test_collision_flag(self):
        stage = Usd.Stage.CreateInMemory()
        s = pavers.author_patio(stage, "/W", (0.0, 0.6, 0.0, 0.5), 0.0, self.tmp, textures=fake_textures(self.tmp),
                                atlas=self.atlas, collision=True)
        self.assertTrue(s["collision"])
        for prim in stage.GetPrimAtPath("/W/Ground/PatioBricks").GetChildren():
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        for prim in stage.GetPrimAtPath("/W/Ground/PatioSand").GetChildren():
            self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))

    def test_small_rect_and_top_z(self):
        stage = Usd.Stage.CreateInMemory()
        s = pavers.author_patio(stage, "/W", (0.0, 0.6, 0.0, 0.5), 0.12, self.tmp, textures=fake_textures(self.tmp),
                                atlas=self.atlas)
        kids = stage.GetPrimAtPath("/W/Ground/PatioBricks").GetChildren()
        self.assertEqual(s["brick_meshes"], len(kids))
        self.assertTrue(all(k.GetName().startswith("Bricks_0_0_") for k in kids), "one chunk, one mesh per colour class")
        top = max(float(np.array(UsdGeom.Mesh(k).GetPointsAttr().Get())[:, 2].max()) for k in kids)
        self.assertAlmostEqual(top, 0.12, delta=SPEC.top_rise)
        sand = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/W/Ground/PatioSand/Joints")).GetPointsAttr().Get())
        np.testing.assert_allclose(sand[:, 2], 0.12 - 0.008, atol=1e-6)

    def test_authoring_again_replaces_the_earlier_patio(self):
        stage = Usd.Stage.CreateInMemory()
        kw = dict(textures=fake_textures(self.tmp), atlas=self.atlas)
        pavers.author_patio(stage, "/W", RECT, 0.0, self.tmp, seed=1, **kw)
        big = len(stage.GetPrimAtPath("/W/Ground/PatioBricks").GetChildren())
        small_rect = (0.0, 0.6, 0.0, 0.5)
        small = pavers.author_patio(stage, "/W", small_rect, 0.0, self.tmp, seed=2, joint_bands=((0.05, 0.6),),
                                    brick_tints=((1.0, 1.0, 1.0), (0.9, 0.9, 0.9)), chunk=1.0, **kw)
        fresh = Usd.Stage.CreateInMemory()
        again = pavers.author_patio(fresh, "/W", small_rect, 0.0, self.tmp, seed=2, joint_bands=((0.05, 0.6),),
                                    brick_tints=((1.0, 1.0, 1.0), (0.9, 0.9, 0.9)), chunk=1.0, **kw)
        self.assertEqual(small, again)
        self.assertGreater(big, small["brick_meshes"])
        names = lambda st, path: sorted(c.GetName() for c in st.GetPrimAtPath(path).GetChildren())  # noqa: E731
        for path in ("/W/Ground/PatioBricks", "/W/Ground/PatioSand", "/W/Looks"):
            self.assertEqual(names(stage, path), names(fresh, path), path)
        self.assertEqual(sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)),
                         sum(1 for p in fresh.Traverse() if p.IsA(UsdGeom.Mesh)))

    def test_clearing_leaves_other_prims_alone(self):
        stage = Usd.Stage.CreateInMemory()
        scene._mesh(stage, "/W/Ground/Patio", scene.box(0, 1, 0, 1, -0.04, 0.0, 1.6))
        scene.preview_surface(stage, "/W/Looks/Lawn")
        pavers.author_patio(stage, "/W", (0.0, 0.6, 0.0, 0.5), 0.0, self.tmp, textures=fake_textures(self.tmp), atlas=self.atlas)
        pavers.author_patio(stage, "/W", (0.0, 0.6, 0.0, 0.5), 0.0, self.tmp, textures=fake_textures(self.tmp), atlas=self.atlas)
        self.assertTrue(stage.GetPrimAtPath("/W/Ground/Patio") and stage.GetPrimAtPath("/W/Looks/Lawn"))

    def test_missing_textures_need_a_cache_dir(self):
        stage = Usd.Stage.CreateInMemory()
        with self.assertRaises(ValueError) as ctx:
            pavers.author_patio(stage, "/W", (0.0, 0.6, 0.0, 0.5), 0.0, "", textures={"pavers": fake_textures(self.tmp)["pavers"]},
                                atlas=self.atlas)
        self.assertIn("sand", str(ctx.exception))

    def test_hide_flat_patio(self):
        stage = Usd.Stage.CreateInMemory()
        self.assertFalse(pavers.hide_flat_patio(stage, "/W"))
        scene._mesh(stage, "/W/Ground/Patio", scene.box(0, 1, 0, 1, -0.04, 0.0, 1.6))
        self.assertTrue(pavers.hide_flat_patio(stage, "/W"))
        self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath("/W/Ground/Patio")).ComputeVisibility(),
                         UsdGeom.Tokens.invisible)


@unittest.skipUnless(HAVE_PXR and HAVE_TEXTURES and os.path.exists(os.path.join(CACHE, "NVIDIA", "Materials", "Base", "Natural", "Sand.mdl")),
                     "needs the mirrored Brick_Pavers and Sand materials")
class TestUsdRealAssets(unittest.TestCase):
    def test_defaults_fetch_the_real_textures(self):
        stage = Usd.Stage.CreateInMemory()
        s = pavers.author_patio(stage, "/World/Scenery", RECT, 0.0, CACHE)
        self.assertGreaterEqual(s["crops"], 40)
        self.assertEqual(s["bricks"], len(pavers.herringbone(RECT, SPEC)))
        tex = UsdShade.Shader(stage.GetPrimAtPath("/World/Scenery/Looks/PatioBricks2/albedoTex"))
        self.assertTrue(tex.GetInput("file").Get().path.endswith("Brick_Pavers_BaseColor.png"))
        tex = UsdShade.Shader(stage.GetPrimAtPath("/World/Scenery/Looks/PatioSand/albedoTex"))
        self.assertTrue(tex.GetInput("file").Get().path.endswith("Sand_BaseColor.png"))
        # every brick's st maps to texels that are brick, not joint (the real texture, v = 0 at the bottom)
        ao = np.asarray(Image.open(REAL["orm"]).convert("RGB"))[..., 0]
        h, w = ao.shape
        for prim in stage.GetPrimAtPath("/World/Scenery/Ground/PatioBricks").GetChildren():
            st = np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get())
            cols = np.clip((st[:, 0] * w).astype(int), 0, w - 1)
            rows = np.clip(((1 - st[:, 1]) * h).astype(int), 0, h - 1)
            self.assertGreater(float((ao[rows, cols] >= 247).mean()), 0.99)

    def test_manifest_without_sand_is_completed_from_the_cache(self):
        """prepare.GROUND has no sand: author_patio fetches it into cache_dir instead of raising KeyError."""
        stage = Usd.Stage.CreateInMemory()
        pavers.author_patio(stage, "/World/Scenery", (0.0, 0.6, 0.0, 0.5), 0.0, CACHE, textures={"pavers": REAL})
        tex = UsdShade.Shader(stage.GetPrimAtPath("/World/Scenery/Looks/PatioSand/albedoTex"))
        self.assertTrue(tex.GetInput("file").Get().path.endswith("Sand_BaseColor.png"))
        tex = UsdShade.Shader(stage.GetPrimAtPath("/World/Scenery/Looks/PatioBricks0/albedoTex"))
        self.assertEqual(tex.GetInput("file").Get().path, REAL["albedo"])


if __name__ == "__main__":
    unittest.main()
