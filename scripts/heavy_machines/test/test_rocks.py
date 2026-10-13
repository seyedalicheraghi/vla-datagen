"""Offline checks of the blasted-rock meshes and the boulder / gravel scatter (numpy; no Isaac Sim).

    python scripts/heavy_machines/test/test_rocks.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def weld(mesh):
    """Triangles indexed by unique vertex *position* (crease / UV-seam duplicates merged)."""
    uniq, inv = np.unique(np.round(mesh.points.astype(np.float64), 6), axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)[mesh.triangles]


def topology(mesh):
    """(V, E, F, bad): bad counts directed edges not used exactly once or lacking their reverse."""
    uniq, tris = weld(mesh)
    d = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    keys, counts = np.unique(d[:, 0] * len(uniq) + d[:, 1], return_counts=True)
    und = np.unique(np.minimum(d[:, 0], d[:, 1]) * len(uniq) + np.maximum(d[:, 0], d[:, 1]))
    rev = d[:, 1] * len(uniq) + d[:, 0]
    bad = int((counts != 1).sum()) + int((~np.isin(rev, keys)).sum())
    return len(uniq), len(und), len(tris), bad


def tri_geometry(mesh):
    p = mesh.points.astype(np.float64)
    a, b, c = (p[mesh.triangles[:, k]] for k in range(3))
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    return a, b, c, cr, area


def signed_volume(mesh):
    a, b, c, _, _ = tri_geometry(mesh)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def build_flat_terrain():
    from heavy_machines.scenarios.terrain import QuarryTerrain

    class FlatTerrain(QuarryTerrain):
        """Zero height, no lane: isolates the scatter rules from the terrain relief."""

        def height(self, x, y):
            return np.zeros(np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape)

        def lane_mask(self, x, y):
            return np.zeros(np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape)

    return FlatTerrain()


def lane_geometry(terr, x, y):
    """Signed distance outside the lane (< 0 inside) using the terrain's own edge definitions."""
    left = y - terr.bench_toe(x)
    right = (terr.right_berm_center(x) + 1.3) - y
    return np.maximum(left, right)


def in_rect(x, y, r, margin=0.0):
    return (x > r[0] - margin) & (x < r[1] + margin) & (y > r[2] - margin) & (y < r[3] + margin)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestRockMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios import rocks
        cls.rocks = rocks
        cls.meshes = {f"seed{s}_d3_a{a}": rocks.rock_mesh(s, 3, a) for s in (0, 1, 2, 3, 4, 5) for a in (0.2, 0.6, 0.95)}
        cls.protos = rocks.rock_prototypes(8, 0)

    def all_meshes(self):
        yield from self.meshes.items()
        for i, m in enumerate(self.protos):
            yield f"proto{i}", m

    def test_valid_mesh(self):
        for name, m in self.all_meshes():
            m.validate()
            self.assertEqual(m.points.dtype, np.float32, name)
            self.assertEqual(m.triangles.dtype, np.int32, name)
            self.assertEqual(m.normals.dtype, np.float32, name)
            self.assertEqual(m.uvs.dtype, np.float32, name)
            self.assertEqual(m.normals.shape, m.points.shape, name)
            self.assertEqual(m.uvs.shape, (len(m.points), 2), name)
            self.assertTrue(np.isfinite(m.uvs).all(), name)
            ln = np.linalg.norm(m.normals.astype(np.float64), axis=1)
            self.assertLess(np.abs(ln - 1.0).max(), 1e-4, name)
            _, _, _, _, area = tri_geometry(m)
            self.assertGreater(area.min(), 0.0, f"{name}: no zero-area triangles")
            x0, x1, y0, y1 = m.rect
            self.assertAlmostEqual(x0, float(m.points[:, 0].min()), places=5)
            self.assertAlmostEqual(y1, float(m.points[:, 1].max()), places=5)

    def test_closed_manifold(self):
        """Welded by position: every directed edge once and its reverse once, sphere topology."""
        for name, m in self.all_meshes():
            v, e, f, bad = topology(m)
            self.assertEqual(bad, 0, f"{name}: open or inconsistently wound")
            self.assertEqual(v - e + f, 2, f"{name}: Euler characteristic")

    def test_outward_winding_and_normals(self):
        for name, m in self.all_meshes():
            self.assertGreater(signed_volume(m), 0.02, f"{name}: CCW outward means positive volume")
            a, b, c, cr, _ = tri_geometry(m)
            cen = (a + b + c) / 3.0
            ctr = 0.5 * (m.points.min(axis=0) + m.points.max(axis=0)).astype(np.float64)
            self.assertGreater((np.einsum("ij,ij->i", cr, cen - ctr) > 0).mean(), 0.97, f"{name}: faces point away from the centre")
            fn = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-18)
            vn = m.normals.astype(np.float64)[m.triangles]
            self.assertGreater((np.einsum("fkj,fj->fk", vn, fn) > 0).mean(), 0.98, f"{name}: vertex normals agree with facets")
            # outside the rock: a point just above each face centroid along its normal must not be inside the bbox hull test
            # (cheap proxy) -> the normals' z component dominates on the topmost vertices
            top = m.points[:, 2] > m.points[:, 2].max() - 0.02
            self.assertGreater(m.normals[top, 2].mean(), 0.2, name)
            bot = m.points[:, 2] < m.points[:, 2].min() + 0.01
            self.assertLess(m.normals[bot, 2].min(), -0.8, name)

    def test_size_and_centering(self):
        for name, m in self.all_meshes():
            lo, hi = m.points.min(axis=0).astype(np.float64), m.points.max(axis=0).astype(np.float64)
            self.assertLess(np.abs(0.5 * (lo + hi)).max(), 1e-4, f"{name}: bbox centred at the origin")
            horiz = max(hi[0] - lo[0], hi[1] - lo[1])
            self.assertTrue(0.9 <= horiz <= 1.1, f"{name}: about 1 m across ({horiz:.3f})")
            height = hi[2] - lo[2]
            self.assertTrue(0.15 <= height <= 1.0, f"{name}: height {height:.2f}")

    def test_flat_bottom(self):
        for name, m in self.all_meshes():
            a, b, c, cr, area = tri_geometry(m)
            fn = cr / np.maximum(2.0 * area, 1e-18)[:, None]
            down = fn[:, 2] < -0.9
            flat = float((area[down] * -fn[down, 2]).sum())
            p = m.points
            bbox = float((p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
            self.assertGreater(flat / bbox, 0.40, f"{name}: broad flat underside ({flat / bbox:.2f} of the footprint)")
            nmean = (fn[down] * area[down, None]).sum(axis=0)
            nmean /= np.linalg.norm(nmean)
            tilt = math.degrees(math.acos(float(np.clip(-nmean[2], -1, 1))))
            self.assertLess(tilt, 6.0, f"{name}: the underside is within a few degrees of level ({tilt:.1f})")

    @staticmethod
    def plane_share(m):
        """Share of the surface area within the 10 biggest 8-degree normal clusters (planarity measure)."""
        a, b, c, cr, area = tri_geometry(m)
        fn = cr / np.maximum(2.0 * area, 1e-18)[:, None]
        remaining = np.ones(len(fn), bool)
        covered = 0.0
        for _ in range(10):
            if not remaining.any():
                break
            seed = np.argmax(np.where(remaining, area, 0.0))
            near = remaining & (fn @ fn[seed] > math.cos(math.radians(8.0)))
            covered += area[near].sum()
            remaining &= ~near
        return float(covered / area.sum())

    def test_angular_not_a_blob(self):
        """Blasted rock is a few planar faces: a large share of the surface lies in a handful of planes
        and the silhouette is not spherical; a blob (sphere) fails both."""
        from heavy_machines.scenarios.terrain import Mesh
        P, T = self.rocks._icosphere(4)
        sphere = Mesh(P.astype(np.float32), T.astype(np.int32), P.astype(np.float32), np.zeros((len(P), 2), np.float32), (0, 0, 0, 0))
        control = self.plane_share(sphere)
        self.assertLess(control, 0.12, "the metric separates a sphere from rock")
        for i, m in enumerate(self.protos):
            share = self.plane_share(m)
            self.assertGreater(share, 0.2, f"proto {i}: top-10 face directions cover {share:.2f} of the surface")
            self.assertGreater(share, 2.0 * control)
            r = np.linalg.norm(m.points.astype(np.float64) - m.points.mean(axis=0), axis=1)
            self.assertGreater(r.std() / r.mean(), 0.12, f"proto {i}: not a sphere")
        shares = [self.plane_share(m) for m in self.protos]
        self.assertGreater(np.mean(shares), 0.4)

    def test_no_folds_or_needles(self):
        """No triangle is flipped back on its neighbours and no edge folds over (every stage is guarded)."""
        rocks = self.rocks
        for name, m in self.all_meshes():
            uniq, tris = weld(m)
            a, b, c = (uniq[tris[:, k]] for k in range(3))
            cr = np.cross(b - a, c - a)
            fn = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-18)
            ef = rocks._edge_faces(tris)
            ang = rocks._dihedrals(fn, ef)
            self.assertLess(float(ang.max()), 125.0, f"{name}: sharpest edge {ang.max():.0f} degrees")

    def test_angularity_controls_edges_and_shading(self):
        """angularity 0 = weathered (rounded edges, smooth shading), 1 = freshly blasted (sharp edges,
        normals that follow the fracture facets); the effect is monotone."""
        rocks = self.rocks

        def facet_deviation(m):
            _, _, _, cr, area = tri_geometry(m)
            fn = cr / np.maximum(2.0 * area, 1e-18)[:, None]
            vn = m.normals.astype(np.float64)[m.triangles]
            dev = np.degrees(np.arccos(np.clip(np.einsum("fkj,fj->fk", vn, fn), -1, 1))).mean(axis=1)
            return float(np.average(dev, weights=area))

        def sharp_edge_share(m):
            """Share of the mesh-edge length whose dihedral angle exceeds 45 degrees (fracture edges)."""
            uniq, tris = weld(m)
            a, b, c = (uniq[tris[:, k]] for k in range(3))
            cr = np.cross(b - a, c - a)
            fn = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-18)
            ef = rocks._edge_faces(tris)
            ang = rocks._dihedrals(fn, ef)
            ln = np.linalg.norm(uniq[ef[2]] - uniq[ef[3]], axis=1)
            return float(ln[ang > 45.0].sum() / ln.sum())

        for seed in (1, 2, 3, 4):
            ms = [rocks.rock_mesh(seed, 3, a) for a in (0.0, 0.5, 1.0)]
            self.assertEqual(len({len(m.triangles) for m in ms}), 1)
            dev = [facet_deviation(m) for m in ms]
            sharp = [sharp_edge_share(m) for m in ms]
            # the surface carries centimetre relief now, so the blend towards the (smooth) fracture-plane normal
            # no longer tracks the individual facets exactly: weathered stone is the smoothest-shaded
            # relative to its facets, fresh stone the closest to them; the in-between is not strictly ordered
            self.assertTrue(dev[0] > dev[1] and dev[0] > dev[2] + 0.8, f"seed {seed}: shading follows the facets more as angularity grows {dev}")
            self.assertTrue(sharp[0] < sharp[1] < sharp[2], f"seed {seed}: edges get sharper {sharp}")
            self.assertGreater(sharp[2], 2.0 * sharp[0], f"seed {seed}: {sharp}")
            self.assertGreater(sharp[2], 0.03, f"seed {seed}: fracture edges are sharp")

    def test_shading_angularity_on_fixed_geometry(self):
        """Same geometry, only the shading changes: angularity blends smooth (0) into flat (1) normals,
        i.e. towards the normal of the fracture plane each triangle belongs to, so a face shades uniformly
        however much its surface is bowed."""
        rocks = self.rocks
        P, T, group, Np, ang, rng = rocks._build_rock(2, 3, 0.6)
        devs = []
        for a in (0.0, 0.5, 1.0):
            pts, tris, nrm, uvs = rocks._shade_and_uv(P, T, group, Np, a, np.random.default_rng(1))
            self.assertTrue(np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6))
            plane_n = Np[group]                                       # (F, 3) fracture plane of each triangle
            cosd = np.einsum("fkj,fj->fk", nrm[tris], plane_n)
            devs.append(float(np.degrees(np.arccos(np.clip(cosd, -1, 1))).mean()))
        self.assertTrue(devs[0] > devs[1] > devs[2], devs)
        self.assertGreater(devs[0] - devs[2], 1.0, devs)

    def test_deterministic_and_distinct(self):
        rocks = self.rocks
        a, b = rocks.rock_mesh(7, 3, 0.6), rocks.rock_mesh(7, 3, 0.6)
        for k in ("points", "triangles", "normals", "uvs"):
            self.assertTrue(np.array_equal(getattr(a, k), getattr(b, k)), k)
        c = rocks.rock_mesh(8, 3, 0.6)
        self.assertFalse(len(a.points) == len(c.points) and np.allclose(a.points, c.points))
        p1, p2 = rocks.rock_prototypes(8, 0), rocks.rock_prototypes(8, 0)
        for m1, m2 in zip(p1, p2):
            self.assertTrue(np.array_equal(m1.points, m2.points))
        p3 = rocks.rock_prototypes(8, 1)
        self.assertFalse(all(len(m1.points) == len(m3.points) and np.allclose(m1.points, m3.points) for m1, m3 in zip(p1, p3)))
        # the cache must hand out independent copies
        p1[0].points[:] = 0.0
        self.assertGreater(float(np.abs(rocks.rock_prototypes(8, 0)[0].points).max()), 0.1)

    def test_detail_scales_triangle_count(self):
        rocks = self.rocks
        counts = [len(rocks.rock_mesh(2, d, 0.7).triangles) for d in (1, 2, 3, 4)]
        self.assertEqual(counts, [20 * 4 ** d for d in (1, 2, 3, 4)])

    def test_uv_in_metres_box_projection(self):
        """rock_mesh: one UV unit = one metre of the unit rock (box projection per fracture face).  The
        prototypes' UVs are the same, times ``proto_uv_scale`` (see test_texel_density_at_instances)."""
        uvs = self.rocks.proto_uv_scale(8)
        for name, m in self.all_meshes():
            if name.startswith("seed") and not name.endswith("a0.6"):
                continue
            k = float(uvs[int(name[5:])]) if name.startswith("proto") else 1.0
            a, b, c, cr, area = tri_geometry(m)
            uv = m.uvs.astype(np.float64)[m.triangles] / k
            ua = 0.5 * np.abs((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                              - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0]))
            ratio = ua / area
            self.assertLessEqual(float(ratio.max()), 1.0 + 1e-3, f"{name}: a projection never grows an area")
            self.assertGreater(float(np.median(ratio)), 0.8, f"{name}: texel density is close to metres on the surface")
            # one UV unit = one metre: the UV extent of the rock is of the order of its size
            self.assertGreater((np.ptp(m.uvs[:, 0]) + np.ptp(m.uvs[:, 1])) / k, 0.5)

    def test_detail_is_only_tessellation(self):
        """A seed is the same rock at every detail level (style, planes, noise, chips): bounding boxes agree
        within ~10% for detail 2..5 (they used to differ by up to 44%), so a LOD swap does not pop and
        ``scatter`` may size instances for any detail."""
        rocks = self.rocks
        for seed in range(8):
            boxes = np.array([np.ptp(rocks.rock_mesh(seed, d, 0.6).points.astype(np.float64), axis=0) for d in (2, 3, 4, 5)])
            ref = boxes[2]
            rel = np.abs(boxes - ref) / ref
            self.assertLess(float(rel.max()), 0.10, f"seed {seed}: bboxes by detail {boxes.round(3).tolist()}")
        # the prototypes too
        e = [rocks.proto_extents(8, 0, d) for d in (3, 4, 5)]
        self.assertLess(float((np.abs(e[0] / e[1] - 1)).max()), 0.10)
        self.assertLess(float((np.abs(e[2] / e[1] - 1)).max()), 0.10)
        # out-of-range detail is clipped, never an error
        self.assertEqual(len(rocks.rock_mesh(3, -2, 0.6).triangles), 20 * 4)
        self.assertEqual(len(rocks.rock_mesh(3, 9, 0.6).triangles), 20 * 4 ** 5)

    def test_surface_is_rough_not_billiard_flat(self):
        """The faces carry centimetre-scale relief and the edges are chipped, not ruler-straight: the
        facet normals of one fracture face spread over a few degrees, and some faces are rougher than others."""
        spreads = []
        for i, m in enumerate(self.protos[:5]):
            _, _, _, cr, area = tri_geometry(m)
            fn = cr / np.maximum(2.0 * area, 1e-18)[:, None]
            big = np.argmax(area)
            # the cluster of facets within 25 degrees of the biggest facet, minus its own planar part
            near = fn @ fn[big] > math.cos(math.radians(25.0))
            mean = (fn[near] * area[near, None]).sum(axis=0)
            mean /= np.linalg.norm(mean)
            dev = np.degrees(np.arccos(np.clip(fn[near] @ mean, -1, 1)))
            spreads.append(float(np.average(dev, weights=area[near])))
        self.assertGreater(float(np.mean(spreads)), 1.2, f"facets are not perfectly planar: {spreads}")
        self.assertLess(float(np.max(spreads)), 12.0, "... but they are still faces, not noise")

    def test_vertex_weights_and_tint(self):
        rocks = self.rocks
        for i, m in enumerate(self.protos):
            w = rocks.rock_vertex_weights(m)
            self.assertEqual(w.shape, (len(m.points), 2))
            self.assertEqual(w.dtype, np.float32)
            self.assertTrue(np.isfinite(w).all() and (w >= 0).all() and (w <= 1).all(), i)
            up = m.normals[:, 2] > 0.7
            down = m.normals[:, 2] < -0.7
            self.assertGreater(float(w[up, 0].mean()), 3.0 * float(w[down, 0].mean() + 0.01), f"proto {i}: dust sits on top")
            z = m.points[:, 2]
            lo, hi = z < z.min() + 0.15 * np.ptp(z), z > z.max() - 0.15 * np.ptp(z)
            self.assertGreater(float(w[lo, 1].mean()), 0.6, f"proto {i}: soil at the foot")
            self.assertLess(float(w[hi, 1].mean()), 0.05, f"proto {i}: none at the top")
            t = rocks.rock_vertex_tint(m)
            self.assertEqual(t.shape, (len(m.points), 3))
            self.assertTrue(np.isfinite(t).all() and (t > 0.2).all() and (t < 1.8).all())
            self.assertGreater(float(t[hi].mean(axis=0)[0]), float(t[lo].mean(axis=0)[0]) + 0.15,
                               f"proto {i}: dusty top is lighter than the soil-stained foot")
            self.assertLess(float(t[lo, 2].mean()), 0.95, "soil darkens and browns the base (blue channel most)")

    def test_prototypes_family(self):
        rocks = self.rocks
        protos = self.protos
        self.assertEqual(len(protos), 8)
        self.assertEqual(len(rocks.rock_prototypes(3, 0)), 3)
        self.assertEqual(len(rocks.rock_prototypes(13, 2)), 13)
        nb = rocks.proto_split(8)
        self.assertTrue(1 <= nb < 8)
        tri = [len(m.triangles) for m in protos]
        self.assertTrue(all(t > tri[-1] for t in tri[:nb]), "boulder-grade prototypes are finer than gravel-grade ones")
        # a variety of proportions: slabs and blocks
        heights = np.array([np.ptp(m.points[:, 2]) for m in protos])
        self.assertGreater(heights.max() / heights.min(), 1.8)
        lens = np.array([[np.ptp(m.points[:, 0]), np.ptp(m.points[:, 1])] for m in protos])
        aspect = lens.min(axis=1) / lens.max(axis=1)
        self.assertGreater(aspect.max() - aspect.min(), 0.15)
        self.assertEqual(len(rocks.proto_extents(8, 0)), 8)
        np.testing.assert_allclose(rocks.proto_extents(8, 0)[0], protos[0].points.max(axis=0) - protos[0].points.min(axis=0), atol=1e-5)
        self.assertEqual(rocks.rock_prototypes(0), [])

    def test_proto_uv_scale_matches_group(self):
        rocks = self.rocks
        k = rocks.proto_uv_scale(8)
        nb = rocks.proto_split(8)
        self.assertEqual(k.shape, (8,))
        self.assertTrue(k[:nb].min() > 5 * k[nb:].max(), "gravel UVs are scaled much harder than boulder UVs")
        self.assertTrue((k > 0).all() and (k <= 1.0).all())

    def test_numpy_only(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios", "rocks.py")
        with open(path) as fh:
            src = fh.read()
        for banned in ("scipy", "cv2", "PIL", "torch", "trimesh"):
            self.assertIsNone(re.search(rf"^\s*(import|from)\s+{banned}\b", src, re.M), banned)


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestScatter(unittest.TestCase):
    RECT = (-45.0, 45.0, -30.0, 35.0)
    KEEP = [(8.0, 22.0, 5.0, 13.0)]

    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios import rocks
        from heavy_machines.scenarios.terrain import QuarryTerrain
        cls.rocks = rocks
        cls.terr = QuarryTerrain(flat_pads=[(9.0, 17.0, 6.5, 11.5)])
        cls.sc = rocks.scatter(cls.terr, cls.RECT, seed=0, keep_clear=cls.KEEP, boulders=160, gravel_per_m2=1.2)
        cls.ext = rocks.proto_extents(cls.sc.n_protos, cls.sc.proto_seed, cls.sc.proto_detail)
        cls.protos = cls.sc.prototypes()
        cls.flat = build_flat_terrain()

    def widths(self, sc):
        e = self.ext[sc.proto]
        return np.maximum(e[:, 0] * sc.scales[:, 0], e[:, 1] * sc.scales[:, 1])

    def heights(self, sc):
        return self.ext[sc.proto][:, 2] * sc.scales[:, 2]

    def test_structure(self):
        sc = self.sc
        n = len(sc)
        self.assertEqual(sc.positions.shape, (n, 3))
        self.assertEqual(sc.scales.shape, (n, 3))
        self.assertEqual(sc.yaws.shape, (n,))
        self.assertEqual(sc.proto.shape, (n,))
        self.assertEqual(sc.positions.dtype, np.float32)
        self.assertEqual(sc.scales.dtype, np.float32)
        self.assertEqual(sc.yaws.dtype, np.float32)
        self.assertEqual(sc.proto.dtype, np.int32)
        self.assertTrue(np.isfinite(sc.positions).all() and np.isfinite(sc.scales).all())
        self.assertTrue((sc.scales > 0).all())
        self.assertTrue(((sc.yaws >= -math.pi - 1e-6) & (sc.yaws <= math.pi + 1e-6)).all())
        self.assertTrue(((sc.proto >= 0) & (sc.proto < sc.n_protos)).all())
        self.assertEqual(sc.n_boulders, 160, "the requested number of boulders is delivered")
        self.assertGreater(len(sc) - sc.n_boulders, 3000, "gravel everywhere")
        x, y = sc.positions[:, 0], sc.positions[:, 1]
        self.assertTrue(((x > self.RECT[0]) & (x < self.RECT[1]) & (y > self.RECT[2]) & (y < self.RECT[3])).all())
        self.assertEqual(sc.shade.shape, (n,))
        # boulders use the boulder-grade prototypes, gravel the low-poly ones
        nb = self.rocks.proto_split(sc.n_protos)
        self.assertTrue((sc.proto[:sc.n_boulders] < nb).all())
        self.assertTrue((sc.proto[sc.n_boulders:] >= nb).all())

    def test_size_ranges(self):
        sc, w = self.sc, self.widths(self.sc)
        bw, gw = w[:sc.n_boulders], w[sc.n_boulders:]
        self.assertGreaterEqual(bw.min(), 0.25 - 1e-4)
        self.assertLessEqual(bw.max(), 1.2 + 1e-4)
        self.assertGreaterEqual(gw.min(), 0.03 - 1e-4)
        self.assertLessEqual(gw.max(), 0.15 + 1e-4)
        # power law: many small, a few big
        self.assertLess(np.median(bw), 0.55)
        self.assertGreater(bw.max(), 0.8)
        self.assertLess(np.median(gw), 0.075)
        # rocks are flatter than they are wide (boulders: aspect held in [0.5, 1.0])
        self.assertTrue((self.heights(sc)[:sc.n_boulders] <= 1.0 * bw * 1.001).all())
        self.assertTrue((self.heights(sc)[:sc.n_boulders] >= 0.5 * bw * 0.999).all())
        self.assertTrue((self.heights(sc) < 1.3 * w).all())

    def test_boulders_never_on_the_lane_or_in_keep_clear(self):
        sc, terr = self.sc, self.terr
        b = slice(0, sc.n_boulders)
        x, y = sc.positions[b, 0].astype(float), sc.positions[b, 1].astype(float)
        s = self.widths(sc)[b]
        self.assertTrue(np.all(terr.lane_mask(x, y) < 0.05), "no boulder centre on the lane")
        for dx, dy in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)):
            self.assertTrue(np.all(terr.lane_mask(x + dx * s, y + dy * s) < 0.05), "no boulder overhangs the lane by much")
        for r in self.KEEP:
            self.assertFalse(in_rect(x, y, r, 0.3 * s).any())

    def test_keep_clear_is_respected_by_everything_and_not_vacuous(self):
        sc = self.sc
        x, y = sc.positions[:, 0].astype(float), sc.positions[:, 1].astype(float)
        for r in self.KEEP:
            self.assertFalse(in_rect(x, y, r).any(), "neither boulders nor gravel inside keep_clear")
        free = self.rocks.scatter(self.terr, self.RECT, seed=0, keep_clear=None, boulders=160, gravel_per_m2=1.2)
        fx, fy = free.positions[:, 0].astype(float), free.positions[:, 1].astype(float)
        inside = in_rect(fx, fy, self.KEEP[0])
        self.assertGreater(inside.sum(), 50, "without keep_clear the rect would hold rocks")
        self.assertGreater(len(sc), len(free) - inside.sum() - 400, "keep_clear only removes what is inside")

    def test_gravel_density_rules(self):
        sc, terr = self.sc, self.terr
        g = slice(sc.n_boulders, None)
        gx, gy = sc.positions[g, 0].astype(float), sc.positions[g, 1].astype(float)
        xs, ys = np.meshgrid(np.arange(-44.75, 45, 0.5), np.arange(-29.75, 35, 0.5))
        gridx, gridy = xs.ravel(), ys.ravel()
        cell = 0.25

        def region_density(pred_fn):
            m_grid = pred_fn(gridx, gridy) & ~in_rect(gridx, gridy, self.KEEP[0])
            m_g = pred_fn(gx, gy)
            area = m_grid.sum() * cell
            return m_g.sum() / area, area

        out = lambda x, y: lane_geometry(terr, x, y)  # noqa: E731
        slope = lambda x, y: terr.surface_class(x, y)[1]  # noqa: E731
        lane_core = lambda x, y: (terr.lane_mask(x, y) > 0.95) & (out(x, y) < -2.0) & (x > -20)  # noqa: E731
        open_far = lambda x, y: (out(x, y) > 4.0) & (slope(x, y) < 0.08) & (terr.lane_mask(x, y) < 0.01)  # noqa: E731
        near_edge = lambda x, y: (out(x, y) > 0.0) & (out(x, y) < 1.5) & (x > -20)  # noqa: E731
        d_lane, a_lane = region_density(lane_core)
        d_open, a_open = region_density(open_far)
        d_edge, a_edge = region_density(near_edge)
        self.assertGreater(a_lane, 300.0)
        self.assertGreater(a_open, 1000.0)
        self.assertGreater(d_open, 0.9, "open ground carries about gravel_per_m2")
        self.assertLess(d_open, 1.9)
        ratio = d_lane / d_open
        self.assertTrue(0.025 <= ratio <= 0.10, f"on the lane the gravel density is ~5% of the open ground's ({ratio:.3f})")
        self.assertGreater(d_lane, 0.0, "sparse spillage on the lane, not none")
        self.assertGreater(d_edge / d_open, 1.5, f"denser along the lane edges ({d_edge / d_open:.2f}x)")
        # lane spillage is small stuff only
        on_lane = terr.lane_mask(gx, gy) > 0.5
        gw = self.widths(sc)[sc.n_boulders:]
        self.assertLessEqual(gw[on_lane].max(), 0.07 + 1e-4)

    def test_boulders_pile_up_at_edges_and_toes(self):
        sc, terr = self.sc, self.terr
        bx, by = sc.positions[:sc.n_boulders, 0].astype(float), sc.positions[:sc.n_boulders, 1].astype(float)
        xs, ys = np.meshgrid(np.arange(-44.75, 45, 0.5), np.arange(-29.75, 35, 0.5))
        near = lambda x, y: np.abs(lane_geometry(terr, x, y) - 0.5) < 3.0  # noqa: E731
        area_frac = near(xs.ravel(), ys.ravel()).mean()
        frac = near(bx, by).mean()
        self.assertGreater(frac, 2.5 * area_frac, f"boulders cluster at the lane edges ({frac:.2f} vs {area_frac:.2f} of the area)")
        # ... and not on steep faces
        slope = terr.surface_class(bx, by)[1]
        self.assertLess(float((slope > 0.6).mean()), 0.08)

    def test_boulders_do_not_overlap(self):
        sc = self.sc
        p = sc.positions[:sc.n_boulders, :2].astype(float)
        s = self.widths(sc)[:sc.n_boulders]
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        lim = 0.5 * (s[:, None] + s[None, :])
        np.fill_diagonal(d, 1e9)
        self.assertGreater(float((d / lim).min()), 0.55, "boulders do not interpenetrate")

    def test_gravel_not_under_boulders(self):
        sc = self.sc
        p = sc.positions[:sc.n_boulders, :2].astype(float)
        s = self.widths(sc)[:sc.n_boulders]
        g = sc.positions[sc.n_boulders:, :2].astype(float)
        d = np.hypot(g[:, None, 0] - p[None, :, 0], g[:, None, 1] - p[None, :, 1])
        self.assertTrue((d >= 0.42 * s[None, :] - 1e-4).all())

    def underside_clearance(self, G, sc, idx):
        """(height of every underside vertex above the ground, instance height) per instance index."""
        out = []
        for i in idx:
            m = self.protos[int(sc.proto[i])]
            P = sc.transform(int(i), m.points)
            c = P[:, 2] - G.height(P[:, 0], P[:, 1])
            und = m.normals[:, 2] < -0.85
            out.append((c[und], c, float(self.heights(sc)[i])))
        return out

    def test_embedding_on_flat_ground(self):
        """25-40% of the height below the ground at the highest point of the underside (a tilted underside
        puts its lowest corner a few percent deeper); every underside vertex is below the ground."""
        rocks = self.rocks
        rect = (-30.0, 10.0, -80.0, -40.0)
        sc = rocks.scatter(self.flat, rect, seed=3, boulders=40, gravel_per_m2=1.0)
        G = rocks._Ground(self.flat, rect, 0.25)
        fr_hi, fr_lo = [], []
        for und, allv, h in self.underside_clearance(G, sc, range(len(sc))):
            fr_hi.append(-und.max() / h)
            fr_lo.append(-allv.min() / h)
        fr_hi, fr_lo = np.array(fr_hi), np.array(fr_lo)
        self.assertGreaterEqual(float(fr_hi.min()), 0.25 - 0.04)
        self.assertLessEqual(float(fr_hi.max()), 0.40 + 0.04)
        self.assertLessEqual(float(fr_lo.max()), 0.40 + 0.20, "never buried more than about half")
        self.assertGreater(float(fr_hi.max() - fr_hi.min()), 0.08, "embedding depth varies")
        for part in (fr_hi[:sc.n_boulders], fr_hi[sc.n_boulders:]):
            self.assertTrue(0.28 < part.mean() < 0.37, float(part.mean()))
        # the position is consistent with the documented transform: rock bottom = z - 0.5 * height
        bottom = sc.positions[:, 2].astype(np.float64) - 0.5 * self.heights(sc)
        self.assertTrue((bottom < 0.0).all())

    def test_nothing_floats_on_real_terrain(self):
        """On the real (lumpy, sloping) terrain the whole underside of every rock is below the rendered ground:
        no air gap under the downhill side (the old 5-point footprint left gaps up to 16 cm)."""
        sc, rocks = self.sc, self.rocks
        G = rocks._Ground(self.terr, self.RECT, 0.25)
        nb = sc.n_boulders
        res = self.underside_clearance(G, sc, range(nb))
        worst = np.array([und.max() for und, _, _ in res])
        self.assertLess(float(worst.max()), 0.0, f"a boulder underside hangs {worst.max():.3f} m above the ground")
        self.assertLess(float(np.percentile(worst, 50)), -0.02)
        g_idx = np.arange(nb, len(sc), 7)
        gworst = np.array([und.max() for und, _, _ in self.underside_clearance(G, sc, g_idx)])
        self.assertLess(float(np.percentile(gworst, 99)), 0.003, "gravel sits on the ground")
        # not buried either: the top of every boulder stands >= 20% of its height above the ground under it, the
        # typical one ~55% (rocks lodged in a slope are embedded on the uphill side but stand free downhill)
        show = np.array([allv.max() / h for _, allv, h in res])
        self.assertGreater(float(show.min()), 0.2)
        self.assertGreater(float(np.median(show)), 0.5)
        burial = np.array([-allv.min() / h for _, allv, h in res])
        self.assertGreater(float(burial.min()), 0.2)

    def test_embedding_on_real_terrain(self):
        sc, rocks = self.sc, self.rocks
        G = rocks._Ground(self.terr, self.RECT, 0.25)
        x, y, z = (sc.positions[:, k].astype(np.float64) for k in range(3))
        h = self.heights(sc)
        bottom = z - 0.5 * h
        low, relief = rocks._feet_ground(G, rocks._feet(sc.n_protos, sc.proto_seed, sc.proto_detail), sc.proto.astype(np.int64),
                                         sc.scales.astype(np.float64), sc.yaws.astype(np.float64), x, y)
        frac = (low - bottom) / h
        self.assertGreaterEqual(float(frac.min()), 0.25 - 5e-3)
        self.assertLessEqual(float(frac.max()), 0.40 + 5e-3)
        # the ground under a boulder's underside is nearly level against it
        nb = sc.n_boulders
        self.assertLess(float((relief[:nb] / h[:nb]).max()), 0.40 + 1e-3)
        ground_c = G.height(x, y)
        self.assertGreaterEqual(float(((ground_c - bottom) / h).min()), 0.25 - 0.02, "never sits above the ground under its centre")
        self.assertLess(float(np.median((ground_c - bottom) / h)), 0.55)

    def test_ground_matches_the_rendered_terrain_mesh(self):
        """Rocks are embedded relative to the surface that is actually rendered: the piecewise-linear
        grid mesh.  The scatter's ground must equal that mesh (this is what keeps rocks from floating)."""
        from heavy_machines.scenarios import terrain as tmod
        rocks, terr = self.rocks, self.terr
        rect = (-12.0, 20.0, -8.0, 14.0)
        G = rocks._Ground(terr, rect, 0.25)
        meshes = [("grid_mesh", tmod.grid_mesh(terr, rect, 0.25))]
        if hasattr(tmod, "split_terrain_meshes"):
            far = (-300.0, 300.0, -300.0, 300.0)
            meshes.append(("split_terrain_meshes", tmod.split_terrain_meshes(terr, rect, far)[0]))
        for name, mesh in meshes:
            p = mesh.points.astype(np.float64)
            tri = p[mesh.triangles]
            cx, cy, cz = tri[:, :, 0].mean(1), tri[:, :, 1].mean(1), tri[:, :, 2].mean(1)
            sel = np.random.default_rng(0).choice(len(tri), min(3000, len(tri)), replace=False)
            err = np.abs(G.height(cx[sel], cy[sel]) - cz[sel])
            self.assertLess(float(err.max()), 1e-4, f"{name}: the ground surrogate follows the rendered triangles")

    def test_counts_scale_with_area_and_density(self):
        rocks, flat = self.rocks, self.flat
        a = rocks.scatter(flat, (-30.0, -10.0, -80.0, -60.0), seed=1, boulders=0, gravel_per_m2=1.0)
        b = rocks.scatter(flat, (-30.0, 10.0, -80.0, -60.0), seed=1, boulders=0, gravel_per_m2=1.0)
        c = rocks.scatter(flat, (-30.0, -10.0, -80.0, -60.0), seed=1, boulders=0, gravel_per_m2=2.5)
        self.assertEqual(a.n_boulders, 0)
        self.assertTrue(1.7 < len(b) / len(a) < 2.3, f"double the area: {len(b) / len(a):.2f}x")
        self.assertTrue(2.1 < len(c) / len(a) < 2.9, f"2.5x the density: {len(c) / len(a):.2f}x")
        self.assertTrue(250 < len(a) < 700, f"~1 per m2 over 400 m2 on open ground ({len(a)})")
        for n in (5, 30):
            s = rocks.scatter(flat, (-30.0, 10.0, -80.0, -40.0), seed=2, boulders=n, gravel_per_m2=0.0)
            self.assertEqual(len(s), n)
            self.assertEqual(s.n_boulders, n)

    def test_degenerate_inputs(self):
        rocks = self.rocks
        none = rocks.scatter(self.flat, (-30.0, 10.0, -80.0, -40.0), seed=1, boulders=0, gravel_per_m2=0.0)
        self.assertEqual(len(none), 0)
        self.assertEqual(none.positions.shape, (0, 3))
        self.assertEqual(none.scales.shape, (0, 3))
        self.assertEqual(none.proto.dtype, np.int32)
        everything = [(-100.0, 100.0, -200.0, 200.0)]
        cleared = rocks.scatter(self.flat, (-30.0, 10.0, -80.0, -40.0), seed=1, keep_clear=everything, boulders=50)
        self.assertEqual(len(cleared), 0)

    def test_deterministic(self):
        rocks = self.rocks
        rect = (-20.0, 20.0, -10.0, 15.0)
        a = rocks.scatter(self.terr, rect, seed=5, boulders=40)
        b = rocks.scatter(self.terr, rect, seed=5, boulders=40)
        for k in ("positions", "scales", "yaws", "proto", "shade"):
            self.assertTrue(np.array_equal(getattr(a, k), getattr(b, k)), k)
        c = rocks.scatter(self.terr, rect, seed=6, boulders=40)
        self.assertFalse(len(a) == len(c) and np.array_equal(a.positions, c.positions))

    def test_texel_density_at_instances(self):
        """One UV unit covers about ``TEXTURE_METRES`` of rock on a typical instance of either group, and
        the density stays within ~3x across the size range (an unscaled gravel UV set is 7-30x too dense)."""
        sc, rocks = self.sc, self.rocks
        uvs = rocks.proto_uv_scale(sc.n_protos)
        want = 1.0 / rocks.TEXTURE_METRES
        area_world = []
        for k, m in enumerate(self.protos):
            a, b, c, cr, area = tri_geometry(m)          # area of the *unit* rock
            uv = m.uvs.astype(np.float64)[m.triangles]
            ua = 0.5 * np.abs((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                              - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0]))
            area_world.append(float(np.median(ua / np.maximum(area, 1e-12))))          # = uv_scale^2 * projection
            self.assertAlmostEqual(math.sqrt(area_world[-1]) / float(uvs[k]), 1.0, delta=0.2)
        w = self.widths(sc)
        proto_k = uvs[sc.proto]
        density = proto_k / w                             # UV units per metre of the instance (width ~ scale)
        for name, sel in (("boulders", slice(0, sc.n_boulders)), ("gravel", slice(sc.n_boulders, None))):
            d = density[sel]
            self.assertTrue(want / 2.0 < float(np.median(d)) < want * 2.0, f"{name}: median density {np.median(d):.2f} vs {want}")
            self.assertGreater(float(np.percentile(d, 2)), want / 4.0, name)
            self.assertLess(float(np.percentile(d, 98)), want * 4.0, name)
        # the failure mode being prevented: unscaled gravel UVs would be ~15x too dense
        self.assertGreater(float(np.median(1.0 / w[sc.n_boulders:])), 8.0)

    def test_scatter_detail_kwarg_and_prototypes(self):
        """Scatter remembers the detail its sizes were computed for; sc.prototypes() matches it; and since a rock
        is the same rock at every detail, a different-detail prototype set is still well placed."""
        rocks = self.rocks
        rect = (-20.0, 20.0, -10.0, 15.0)
        a = rocks.scatter(self.terr, rect, seed=2, boulders=40, detail=3)
        b = rocks.scatter(self.terr, rect, seed=2, boulders=40, detail=4)
        self.assertEqual((a.proto_detail, b.proto_detail), (3, 4))
        self.assertLess(abs(len(a) - len(b)), 0.05 * len(b))      # same rocks, only the contact details differ
        ma = a.prototypes()
        self.assertEqual(len(ma[0].triangles), len(rocks.rock_prototypes(a.n_protos, a.proto_seed, 3)[0].triangles))
        self.assertLess(len(ma[0].triangles), len(b.prototypes()[0].triangles))
        np.testing.assert_allclose(rocks.proto_extents(a.n_protos, a.proto_seed, 3), np.array([np.ptp(m.points, axis=0) for m in ma]), atol=1e-5)
        # LOD swap: a detail-4 scatter drawn with detail-3 meshes stays within a few cm of the ground
        G = rocks._Ground(self.terr, rect, 0.25)
        lo = rocks.rock_prototypes(b.n_protos, b.proto_seed, 3)
        worst = []
        for i in range(b.n_boulders):
            m = lo[int(b.proto[i])]
            P = b.transform(i, m.points)
            und = m.normals[:, 2] < -0.85
            worst.append(float((P[und, 2] - G.height(P[und, 0], P[und, 1])).max()))
        self.assertLess(float(np.percentile(worst, 90)), 0.03, "no visible air gap after a LOD swap")

    def test_presence_in_the_clip(self):
        """The rocks must be visible from the clip's camera: sizes, visible height and where they stand, on the real
        scenario terrain / rect, measured with the prototype meshes (the old 2.3 power law gave a 0.36 m median,
        12 cm of visible height and a bare lane edge)."""
        from heavy_machines.scenarios import adt_approach_hex
        rocks = self.rocks
        sc0 = adt_approach_hex.build()
        terr, rect = sc0.terrain, sc0.local_rect
        view = [(-22.0, 30.0, -16.0, 16.0)]                     # what the front camera sees (x ahead of the truck)
        sc = rocks.scatter(terr, rect, seed=0, keep_clear=sc0.keep_clear, boulders=480, focus=view)
        G = rocks._Ground(terr, rect, 0.25)
        protos = sc.prototypes()
        nb = sc.n_boulders
        self.assertEqual(nb, 480)
        w = np.zeros(nb)
        prom = np.zeros(nb)
        for i in range(nb):
            P = sc.transform(i, protos[int(sc.proto[i])].points)
            w[i] = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
            prom[i] = max(float((P[:, 2] - G.height(P[:, 0], P[:, 1])).max()), 0.0)
        self.assertGreater(float(np.median(w)), 0.38)
        self.assertGreater(float((w >= 0.5).mean()), 0.30, "a real share of readable blocks")
        self.assertGreater(int((w >= 0.9).sum()), 20)
        self.assertGreater(float(np.median(prom)), 0.14, "median visible height above the ground")
        self.assertLess(float((prom < 0.08).mean()), 0.10, "few boulders stick out by under 8 cm")
        x, y = sc.positions[:nb, 0].astype(float), sc.positions[:nb, 1].astype(float)
        self.assertLess(float((sc.positions[:nb, 2] > 2.0).mean()), 0.06, "not on the bench top")
        out = lane_geometry(terr, x, y)
        near = np.abs(out - 0.5) < 3.0
        self.assertGreater(float(near.mean()), 0.5, "most boulders stand along the lane edges / windrow")
        # in the first 40 m ahead of the camera (eye x = -20.45) inside its frustum
        dx = x - (-20.45)
        fr = (dx > 3.0) & (dx < 40.0) & (np.abs(y) < math.tan(math.radians(46.0)) * dx)
        big = (w >= 0.45) & (prom >= 0.2)
        self.assertGreaterEqual(int((fr & big).sum()), 20, "blocks >= 0.45 m wide and >= 0.2 m high within the first 40 m")
        self.assertGreater(int((fr & near).sum()), 40, "the lane edges in the frustum carry boulders")
        # tunable: a shallower / steeper power law moves the size distribution
        steep = rocks.scatter(terr, rect, seed=0, keep_clear=sc0.keep_clear, boulders=200, size_alpha=2.6)
        shallow = rocks.scatter(terr, rect, seed=0, keep_clear=sc0.keep_clear, boulders=200, size_alpha=1.0)
        ws = lambda s: np.maximum(self.ext_of(s)[:, 0] * s.scales[:, 0], self.ext_of(s)[:, 1] * s.scales[:, 1])[:s.n_boulders]  # noqa: E731
        self.assertGreater(float(np.median(ws(shallow))), float(np.median(ws(steep))) + 0.05)
        self.assertGreater(float(np.percentile(ws(shallow), 90)), float(np.percentile(ws(steep), 90)) + 0.1)
        # focus really concentrates: more boulders in the view zone with it than without
        plain = rocks.scatter(terr, rect, seed=0, keep_clear=sc0.keep_clear, boulders=480)
        inview = lambda s: int(in_rect(s.positions[:s.n_boulders, 0], s.positions[:s.n_boulders, 1], view[0]).sum())  # noqa: E731
        self.assertGreater(inview(sc), 1.4 * inview(plain))

    def ext_of(self, sc):
        return self.rocks.proto_extents(sc.n_protos, sc.proto_seed, sc.proto_detail)[sc.proto]

    def test_robust_inputs(self):
        rocks, terr = self.rocks, self.terr
        R = (-20.0, 20.0, -10.0, 15.0)
        ok = lambda s: self.assertTrue(np.isfinite(s.positions).all() and (s.scales > 0).all())  # noqa: E731
        ok(rocks.scatter(terr, R, keep_clear=np.array([[0.0, 5.0, 0.0, 5.0]]), boulders=20))
        ok(rocks.scatter(terr, R, keep_clear=np.array([0.0, 5.0, 0.0, 5.0]), boulders=20))   # a single rect
        ok(rocks.scatter(terr, R, keep_clear=((0, 5, 0, 5), [10, 12, -3, 3]), boulders=20))
        ok(rocks.scatter(terr, np.array(R), boulders=20))
        ok(rocks.scatter(terr, R, seed=2 ** 33 + 5, boulders=20))
        ok(rocks.scatter(terr, R, seed=-7, boulders=20))
        ok(rocks.scatter(terr, R, boulders=-3, gravel_per_m2=-1.0))
        ok(rocks.scatter(terr, R, boulders=7.0, n_protos=1))
        for tiny in ((0.0, 0.1, 0.0, 0.1), (0.0, 0.2, 0.0, 30.0)):                # tiny rects give (almost) nothing, no crash
            ok(rocks.scatter(terr, tiny, boulders=5))
        self.assertEqual(len(rocks.scatter(terr, (0.0, 0.1, 0.0, 0.1))), len(rocks.scatter(terr, (0.0, 0.1, 0.0, 0.1))))
        for bad in ((0.0, 0.0, 0.0, 5.0), (20.0, -20.0, 15.0, -10.0), (0.0, 1.0, 2.0), (0.0, float("nan"), 0.0, 1.0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                rocks.scatter(terr, bad)
        with self.assertRaises(ValueError):
            rocks.scatter(terr, R, n_protos=0)
        with self.assertRaises(ValueError):
            rocks.scatter(terr, R, size_range=(0.5, 0.2))
        with self.assertRaises(ValueError):
            rocks.scatter(terr, R, keep_clear=[(1.0, 2.0, 3.0)])
        # rock_mesh: huge seeds, negative / out-of-range detail
        for seed, d in ((2 ** 40, 3), (-5, 3), (1, -1), (1, 0), (1, 12)):
            rocks.rock_mesh(seed, d).validate()

    def test_contact_map(self):
        rocks, sc = self.rocks, self.sc
        rect = (-12.0, 12.0, -10.0, 12.0)
        cm = rocks.contact_map(sc, rect, px_per_m=8.0)
        self.assertEqual(cm.shape, (176, 192, 3))
        self.assertEqual(cm.dtype, np.float32)
        self.assertTrue(np.isfinite(cm).all() and (cm > 0.3).all() and (cm <= 1.0).all())
        self.assertGreater(float((cm[..., 0] < 0.99).mean()), 0.01, "boulders darken some ground")
        self.assertGreater(float((cm[..., 0] > 0.999).mean()), 0.5, "most of the ground is untouched")
        self.assertTrue((cm[..., 2] <= cm[..., 0] + 1e-6).all(), "shadow is warmer: blue drops more than red")
        # right at a boulder's edge the ground is clearly darker than 1 m further out
        i = int(np.argmax(self.widths(sc)[:sc.n_boulders] * in_rect(sc.positions[:sc.n_boulders, 0], sc.positions[:sc.n_boulders, 1], rect, -2.0)))
        x, y = float(sc.positions[i, 0]), float(sc.positions[i, 1])
        e = self.ext[int(sc.proto[i])]
        a = 0.5 * e[0] * float(sc.scales[i, 0])

        def at(px, py):
            c = int((px - rect[0]) * 8.0)
            r = int((rect[3] - py) * 8.0)
            return float(cm[r, c, 0])
        self.assertLess(at(x + a * 0.9 * math.cos(float(sc.yaws[i])), y + a * 0.9 * math.sin(float(sc.yaws[i]))), 0.7)
        self.assertTrue(np.array_equal(cm, rocks.contact_map(sc, rect, px_per_m=8.0)))
        empty = rocks.scatter(self.flat, (-30.0, 10.0, -80.0, -40.0), boulders=0, gravel_per_m2=0.0)
        self.assertTrue((rocks.contact_map(empty, (-30.0, 10.0, -80.0, -40.0), 2.0) == 1.0).all())

    def test_transform_helper(self):
        sc, rocks = self.sc, self.rocks
        m = rocks.rock_prototypes(sc.n_protos, sc.proto_seed)[int(sc.proto[0])]
        pts = sc.transform(0, m.points)
        lo = pts.min(axis=0)
        self.assertAlmostEqual(float(lo[2]), float(sc.positions[0, 2] - 0.5 * self.heights(sc)[0]), places=4)
        c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        self.assertLess(float(np.abs(c[:2] - sc.positions[0, :2]).max()), 0.5 * float(self.widths(sc)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
