"""Offline checks of the render-quality machine meshes (numpy; no Isaac Sim).

    python scripts/heavy_machines/test/test_visual_meshes.py
"""

from __future__ import annotations

import copy
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


# ---------------------------------------------------------------------------
# Mesh analysis helpers
# ---------------------------------------------------------------------------

def weld(mesh):
    """Triangles indexed by unique vertex *position* (seam duplicates merged)."""
    uniq, inv = np.unique(mesh.points, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)[mesh.triangles]


def edge_stats(mesh):
    """(#vertices, #edges, #faces, max |directed use - 1| over all directed edges)."""
    uniq, tris = weld(mesh)
    d = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    keys, counts = np.unique(d[:, 0] * len(uniq) + d[:, 1], return_counts=True)
    und = np.unique(np.minimum(d[:, 0], d[:, 1]) * len(uniq) + np.maximum(d[:, 0], d[:, 1]))
    rev_keys = d[:, 1] * len(uniq) + d[:, 0]
    # every directed edge exactly once and its reverse exactly once => closed, consistently oriented
    bad = int((counts != 1).sum()) + int((~np.isin(rev_keys, keys)).sum())
    return len(uniq), len(und), len(tris), bad


def tri_geometry(mesh):
    p = mesh.points.astype(np.float64)
    a, b, c = (p[mesh.triangles[:, k]] for k in range(3))
    cr = np.cross(b - a, c - a)
    return a, b, c, cr


def signed_volume(mesh):
    a, b, c, _ = tri_geometry(mesh)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


MAX_PUSH = 0.011        # largest total growth of one shape along one axis (m) from coplanar separation


def flat_vertices(mesh, normal, tol=1e-6):
    n = np.asarray(normal, dtype=np.float64)
    sel = (mesh.normals.astype(np.float64) @ n) > 1.0 - tol
    return np.nonzero(sel)[0]


# ---------------------------------------------------------------------------
# Independent detector of exposed coplanar faces (does not use the module's code)
# ---------------------------------------------------------------------------

def _link_faces(shape, frame=None):
    """(centre, normal, u, v, kind, dims) of the planar faces of a box / cylinder (link frame,
    or the frame ``frame`` maps the link frame into)."""
    from heavy_machines import kinematics as K
    from heavy_machines.spec import Box, Cylinder

    tf = K.Tf.from_xyz_rpy(shape.xyz, shape.rpy)
    if frame is not None:
        tf = frame @ tf
    R, t = np.array(tf.R), np.array(tf.t)
    g, out = shape.geom, []
    if isinstance(g, Box):
        h = np.array(g.size) / 2
        for ax in range(3):
            for sg in (1, -1):
                n = np.zeros(3)
                n[ax] = sg
                u = np.zeros(3)
                u[(ax + 1) % 3] = 1
                v = np.zeros(3)
                v[(ax + 2) % 3] = 1
                out.append((R @ (n * h[ax]) + t, R @ n, R @ u, R @ v, "rect", (h[(ax + 1) % 3], h[(ax + 2) % 3])))
    elif isinstance(g, Cylinder):
        for sg in (1, -1):
            n = np.array([0.0, 0.0, float(sg)])
            out.append((R @ (n * g.length / 2) + t, R @ n, R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0]),
                        "disc", (g.radius,)))
    return out


def _inside_shape(shape, pts, frame=None):
    from heavy_machines import kinematics as K
    from heavy_machines.spec import Box, Cylinder

    tf = K.Tf.from_xyz_rpy(shape.xyz, shape.rpy)
    if frame is not None:
        tf = frame @ tf
    q = (pts - np.array(tf.t)) @ np.array(tf.R)
    g = shape.geom
    if isinstance(g, Box):
        return np.all(np.abs(q) <= np.array(g.size) / 2, axis=1)
    if isinstance(g, Cylinder):
        return (np.hypot(q[:, 0], q[:, 1]) <= g.radius) & (np.abs(q[:, 2]) <= g.length / 2)
    return np.linalg.norm(q, axis=1) <= g.radius


def exposed_coplanar_pairs(shapes, tol=4.5e-4, inset=5e-3, min_area=1.5e-4, grid=25, frames=None, link_of=None):
    """Pairs ``(i, j)`` of visible shapes with overlapping, same-direction faces in one plane
    (within ``tol``) whose overlap is visible from outside (a point 1 cm above it lies in no shape).
    With ``link_of`` only pairs of shapes on different links are reported (``frames`` maps each
    shape's link frame into the common frame)."""
    from heavy_machines.spec import Sphere

    vis = [(i, s) for i, s in enumerate(shapes) if s.visual and not isinstance(s.geom, Sphere)]
    faces, owner = [], []
    for i, sh in vis:
        for f in _link_faces(sh, None if frames is None else frames[i]):
            faces.append(f)
            owner.append(i)
    if not faces:
        return []
    N = np.array([f[1] for f in faces])
    C = np.array([f[0] for f in faces])
    owner = np.array(owner)
    dist = (N * C).sum(1)[:, None] - N @ C.T
    ok = (N @ N.T > 0.999) & (np.abs(dist) <= tol) & (owner[:, None] < owner[None, :])
    found = set()
    for a, b in zip(*np.nonzero(ok)):
        if link_of is not None and link_of[owner[a]] == link_of[owner[b]]:
            continue
        for f1, f2 in ((faces[a], faces[b]), (faces[b], faces[a])):
            c, n, u, v, kind, d = f1
            hu = max(d[0] - inset, 0.5 * d[0])
            hv = hu if kind == "disc" else max(d[1] - inset, 0.5 * d[1])
            A, B = np.meshgrid(np.linspace(-1, 1, grid) * hu, np.linspace(-1, 1, grid) * hv)
            A, B = A.ravel(), B.ravel()
            if kind == "disc":
                keep = A * A + B * B <= hu * hu
                A, B = A[keep], B[keep]
            pts = c + A[:, None] * u + B[:, None] * v
            c2, _, u2, v2, kind2, d2 = f2
            rel = pts - c2
            x, y = rel @ u2, rel @ v2
            if kind2 == "disc":
                m = x * x + y * y <= max(d2[0] - inset, 0.5 * d2[0]) ** 2
            else:
                m = (np.abs(x) <= max(d2[0] - inset, 0.5 * d2[0])) & (np.abs(y) <= max(d2[1] - inset, 0.5 * d2[1]))
            area = m.mean() * (math.pi * hu * hu if kind == "disc" else 4 * hu * hv)
            if m.sum() < 3 or area < min_area:
                continue
            P = pts[m] + n * 0.01
            free = np.ones(len(P), bool)
            for k, sh in vis:
                free &= ~_inside_shape(sh, P, None if frames is None else frames[k])
            if free.sum() >= 3:
                found.add((int(owner[a]), int(owner[b])))
                break
    return sorted(found)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class MeshAssertions(unittest.TestCase):
    """Properties every closed, convex shape mesh must have."""

    def check_closed_convex(self, mesh, name="", convex=True):
        mesh.validate()
        self.assertEqual(mesh.rect, (0.0, 0.0, 0.0, 0.0), name)
        self.assertEqual(mesh.points.dtype, np.float32)
        self.assertEqual(mesh.triangles.dtype, np.int32)
        self.assertEqual(mesh.normals.dtype, np.float32)
        self.assertEqual(mesh.uvs.dtype, np.float32)
        self.assertEqual(mesh.uvs.shape, (len(mesh.points), 2))
        self.assertTrue(np.isfinite(mesh.uvs).all())
        # unit normals
        ln = np.linalg.norm(mesh.normals.astype(np.float64), axis=1)
        self.assertLess(np.abs(ln - 1.0).max(), 1e-5, name)
        # outward normals on a convex shape
        cen = mesh.points.astype(np.float64).mean(0)
        dots = np.einsum("ij,ij->i", mesh.normals.astype(np.float64), mesh.points - cen)
        self.assertGreater(dots.mean(), 0.0, name)
        # no degenerate triangles; winding agrees with the smooth normals
        a, b, c, cr = tri_geometry(mesh)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        zero_area = area < 1e-12
        # (cylinder / sphere poles can leave none, but tolerate none beyond float noise)
        self.assertEqual(int(zero_area.sum()), 0, f"{name}: degenerate triangles")
        nm = (mesh.normals[mesh.triangles].astype(np.float64)).sum(1)
        self.assertGreater(np.einsum("ij,ij->i", cr, nm).min(), 0.0, f"{name}: winding vs normal")
        # outward from the centroid for a convex shape (every face sees the centroid behind it)
        fc = (a + b + c) / 3.0
        if convex:
            self.assertGreater(np.einsum("ij,ij->i", cr, fc - cen).min(), 0.0, f"{name}: winding vs centroid")
        # closed, consistently oriented 2-manifold of genus 0
        v, e, f, bad = edge_stats(mesh)
        self.assertEqual(bad, 0, f"{name}: open or inconsistently oriented edges")
        self.assertEqual(v - e + f, 2, f"{name}: Euler characteristic")
        self.assertGreater(signed_volume(mesh), 0.0, name)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestBox(MeshAssertions):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios.visual_meshes import fillet_radius, shape_mesh
        from heavy_machines.spec import box

        cls.shape_mesh, cls.fillet_radius, cls.box = staticmethod(shape_mesh), staticmethod(fillet_radius), staticmethod(box)

    SIZES = [(1.0, 2.0, 3.0), (0.8, 0.8, 0.8), (6.0, 2.5, 0.4), (0.05, 0.3, 0.02),
             (2.0, 0.01, 1.0), (0.01, 0.01, 0.01)]

    def test_bounding_box_equals_size(self):
        for size in self.SIZES:
            for bevel in (0.025, 0.1, 0.0):
                m = self.shape_mesh(self.box(*size), bevel=bevel)
                half = np.array(size) / 2
                np.testing.assert_allclose(m.points.min(0), -half, atol=1e-6, err_msg=f"{size} {bevel}")
                np.testing.assert_allclose(m.points.max(0), half, atol=1e-6, err_msg=f"{size} {bevel}")

    def test_closed_outward_convex(self):
        for size in self.SIZES:
            for bevel in (0.025, 0.2, 0.0):
                self.check_closed_convex(self.shape_mesh(self.box(*size), bevel=bevel), f"box {size} bevel {bevel}")

    def test_fillet_radius_rule(self):
        # r = min(bevel, 0.3 * min(size)): the flat part of the +z face spans +-(h - r)
        for size, bevel in [((1.0, 2.0, 3.0), 0.025), ((4.0, 3.0, 0.04), 0.025), ((0.05, 0.3, 0.2), 0.1)]:
            r = min(bevel, 0.3 * min(size))
            self.assertAlmostEqual(self.fillet_radius(bevel, *size), r)
            m = self.shape_mesh(self.box(*size), bevel=bevel)
            idx = flat_vertices(m, (0, 0, 1))
            self.assertEqual(len(idx), 4)                      # corners of the flat region only
            pts = m.points[idx].astype(np.float64)
            np.testing.assert_allclose(np.abs(pts[:, 0]), size[0] / 2 - r, atol=1e-6)
            np.testing.assert_allclose(np.abs(pts[:, 1]), size[1] / 2 - r, atol=1e-6)
            np.testing.assert_allclose(pts[:, 2], size[2] / 2, atol=1e-6)

    def test_zero_bevel_is_a_sharp_box(self):
        m = self.shape_mesh(self.box(1.0, 2.0, 3.0), bevel=0.0)
        self.assertEqual((len(m.points), len(m.triangles)), (24, 12))
        self.assertTrue(np.all(np.abs(m.normals).max(1) == 1.0))

    def test_fillets_have_smooth_normals(self):
        m = self.shape_mesh(self.box(1.0, 1.0, 1.0), bevel=0.1)
        # normals must take intermediate directions, not just the six axes
        axis_aligned = (np.abs(m.normals).max(1) > 1 - 1e-6)
        self.assertGreater((~axis_aligned).sum(), 100)
        # a point on a fillet: its normal points from the fillet axis to the point
        n = m.normals.astype(np.float64)
        p = m.points.astype(np.float64)
        sel = (~axis_aligned) & (np.abs(n[:, 2]) < 1e-6)       # on the four vertical edge fillets
        self.assertTrue(sel.any())
        centre = np.sign(p[sel][:, :2]) * (0.5 - 0.1)
        radial = (p[sel][:, :2] - centre) / 0.1
        np.testing.assert_allclose(radial, n[sel][:, :2], atol=1e-5)

    def test_uv_scale_is_metres(self):
        size = (1.0, 2.0, 3.0)
        r = 0.025
        m = self.shape_mesh(self.box(*size), bevel=r)
        for normal in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
            idx = flat_vertices(m, normal)
            self.assertEqual(len(idx), 4, normal)
            p, uv = m.points[idx].astype(np.float64), m.uvs[idx].astype(np.float64)
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertAlmostEqual(float(np.linalg.norm(p[i] - p[j])),
                                           float(np.linalg.norm(uv[i] - uv[j])), places=5,
                                           msg=f"face {normal}")
            # u x v points along the outward normal (textures are never mirrored)
            e1 = (p[1] - p[0]); e2 = (p[2] - p[0])
            u1 = uv[1] - uv[0]; u2 = uv[2] - uv[0]
            det = u1[0] * u2[1] - u1[1] * u2[0]
            self.assertGreater(det * np.dot(np.cross(e1, e2), normal), 0.0, f"mirrored UVs on {normal}")
        # a whole face spans (size - 2r) + two half fillet arcs of UV
        uv_all = m.uvs.astype(np.float64)
        self.assertAlmostEqual(float(uv_all.min()), 0.0, places=6)

    def test_uv_stretch_is_small_everywhere(self):
        for size, bevel in [((1.0, 2.0, 3.0), 0.025), ((1.0, 1.0, 1.0), 0.1), ((3.0, 0.2, 0.6), 0.025)]:
            m = self.shape_mesh(self.box(*size), bevel=bevel)
            p = m.points.astype(np.float64)
            uv = m.uvs.astype(np.float64)
            ratios = []
            for k in range(3):
                i, j = m.triangles[:, k], m.triangles[:, (k + 1) % 3]
                d3 = np.linalg.norm(p[i] - p[j], axis=1)
                d2 = np.linalg.norm(uv[i] - uv[j], axis=1)
                ratios.append(d2[d3 > 1e-9] / d3[d3 > 1e-9])
            ratios = np.concatenate(ratios)
            self.assertGreater(ratios.min(), 0.55, f"{size}")
            self.assertLess(ratios.max(), 1.6, f"{size}")
            self.assertAlmostEqual(float(np.median(ratios)), 1.0, delta=0.05)

    def test_transform_applied(self):
        from heavy_machines import kinematics as K

        shape = self.box(1.0, 2.0, 3.0, at=(1.0, -2.0, 0.5), rpy=(0.3, -0.7, 1.1))
        local = self.shape_mesh(self.box(1.0, 2.0, 3.0))
        m = self.shape_mesh(shape)
        tf = K.Tf.from_xyz_rpy(shape.xyz, shape.rpy)
        R = np.array(tf.R)
        np.testing.assert_allclose(m.points, local.points @ R.T + np.array(tf.t), atol=2e-6)
        np.testing.assert_allclose(m.normals, local.normals @ R.T, atol=2e-6)
        np.testing.assert_allclose(m.uvs, local.uvs)
        np.testing.assert_array_equal(m.triangles, local.triangles)
        self.check_closed_convex(m, "rotated box")
        np.testing.assert_allclose(m.points.astype(np.float64).mean(0),
                                   np.array(tf.t) + R @ local.points.astype(np.float64).mean(0), atol=2e-6)

    def test_volume_close_to_rounded_box(self):
        size, r = (1.0, 2.0, 3.0), 0.1
        m = self.shape_mesh(self.box(*size), bevel=r)
        a, b, c = size
        # exact rounded box volume (Steiner): box shrunk by r + faces + edges + sphere
        inner = (a - 2 * r) * (b - 2 * r) * (c - 2 * r)
        faces = r * 2 * ((a - 2 * r) * (b - 2 * r) + (b - 2 * r) * (c - 2 * r) + (a - 2 * r) * (c - 2 * r))
        edges = math.pi * r * r / 4 * 4 * ((a - 2 * r) + (b - 2 * r) + (c - 2 * r))
        sph = 4 / 3 * math.pi * r ** 3
        exact = inner + faces + edges + sph
        self.assertAlmostEqual(signed_volume(m) / exact, 1.0, delta=0.002)


    def test_adaptive_fillet_rule(self):
        """bevel_scale > 0: r = max(bevel, min(max_bevel, bevel_scale * middle dim)) <= 0.3 * min dim."""
        cases = [
            ((3.0, 2.0, 0.5), 0.035, 0.07),            # grows with the part
            ((6.0, 4.0, 3.0), 0.035, 0.10),            # capped by max_bevel
            ((3.0, 2.0, 0.04), 0.035, 0.012),          # thin plate: capped by 0.3 * thickness
            ((0.3, 0.2, 0.1), 0.035, 0.025),           # small part keeps the base bevel
        ]
        for size, scale, want in cases:
            self.assertAlmostEqual(self.fillet_radius(0.025, *size, bevel_scale=scale, max_bevel=0.10), want, places=9)
            m = self.shape_mesh(self.box(*size), bevel=0.025, bevel_scale=scale, max_bevel=0.10)
            idx = flat_vertices(m, (0, 0, 1))
            pts = m.points[idx].astype(np.float64)
            np.testing.assert_allclose(np.abs(pts[:, 0]), size[0] / 2 - want, atol=1e-6)
            np.testing.assert_allclose(np.abs(pts[:, 1]), size[1] / 2 - want, atol=1e-6)
            np.testing.assert_allclose(m.points.max(0), np.array(size) / 2, atol=1e-6)
            self.check_closed_convex(m, f"adaptive box {size}")
        # bevel_scale = 0 (the default) is exactly the plain rule
        a = self.shape_mesh(self.box(3.0, 2.0, 0.5))
        b = self.shape_mesh(self.box(3.0, 2.0, 0.5), bevel_scale=0.0)
        np.testing.assert_array_equal(a.points, b.points)
        self.assertEqual(self.fillet_radius(0.025, 3.0, 2.0, 0.5), 0.025)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestCylinder(MeshAssertions):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios.visual_meshes import shape_mesh
        from heavy_machines.spec import cyl

        cls.shape_mesh, cls.cyl = staticmethod(shape_mesh), staticmethod(cyl)

    CASES = [(0.5, 1.2), (0.805, 0.6), (0.02, 0.5), (1.17, 0.66), (0.3, 0.02), (0.1, 3.0)]

    def test_bounding_box(self):
        for R, L in self.CASES:
            m = self.shape_mesh(self.cyl(R, L))
            np.testing.assert_allclose(m.points.min(0), [-R, -R, -L / 2], atol=1e-6, err_msg=f"{R} {L}")
            np.testing.assert_allclose(m.points.max(0), [R, R, L / 2], atol=1e-6, err_msg=f"{R} {L}")

    def test_closed_outward_convex(self):
        for R, L in self.CASES:
            for bevel in (0.025, 0.1, 0.0):
                self.check_closed_convex(self.shape_mesh(self.cyl(R, L), bevel=bevel), f"cyl {R} {L} {bevel}")

    def test_volume_and_roundness(self):
        for R, L in [(0.5, 1.2), (1.17, 0.66)]:
            m = self.shape_mesh(self.cyl(R, L), bevel=0.025)
            vol = math.pi * R * R * L
            self.assertLess(abs(signed_volume(m) / vol - 1.0), 0.01)
            # silhouette deviation of the wall polygon stays small even for big wheels
            p = m.points.astype(np.float64)
            n = m.normals.astype(np.float64)
            wall = p[np.abs(n[:, 2]) < 1e-6]                     # straight wall vertices
            rho = np.hypot(wall[:, 0], wall[:, 1])
            self.assertLess(np.abs(rho - R).max(), 1e-6)
            # chord sag between neighbouring wall vertices: R * (1 - cos(pi / N)) <= ~3 mm
            n_seg = len(np.unique(np.round(np.arctan2(wall[:, 1], wall[:, 0]), 6)))
            self.assertLess(R * (1.0 - math.cos(math.pi / n_seg)), 0.0031)

    def test_axis_orientation(self):
        # cyl(axis='y') runs along y
        m = self.shape_mesh(self.cyl(0.4, 1.0, axis="y"))
        np.testing.assert_allclose(m.points.max(0), [0.4, 0.5, 0.4], atol=1e-6)
        np.testing.assert_allclose(m.points.min(0), [-0.4, -0.5, -0.4], atol=1e-6)
        self.check_closed_convex(m, "y cylinder")
        m = self.shape_mesh(self.cyl(0.4, 1.0, axis="x", at=(1.0, 2.0, 3.0)))
        np.testing.assert_allclose(m.points.max(0), [1.5, 2.4, 3.4], atol=1e-6)

    def test_fillet_size(self):
        R, L = 0.8, 0.6
        r = min(0.025, 0.3 * min(R, L))
        m = self.shape_mesh(self.cyl(R, L), bevel=0.025)
        p = m.points.astype(np.float64)
        n = m.normals.astype(np.float64)
        # the cap is flat out to R - r
        cap = np.abs(n[:, 2]) > 1 - 1e-6
        self.assertAlmostEqual(float(np.hypot(p[cap][:, 0], p[cap][:, 1]).max()), R - r, places=6)
        # the straight wall spans z within +-(L/2 - r)
        wall = np.abs(n[:, 2]) < 1e-6
        self.assertAlmostEqual(float(np.abs(p[wall][:, 2]).max()), L / 2 - r, places=6)

    def test_uv_scale_is_metres(self):
        R, L, r = 0.8, 0.6, 0.025
        m = self.shape_mesh(self.cyl(R, L), bevel=r)
        p, uv, n = (m.points.astype(np.float64), m.uvs.astype(np.float64), m.normals.astype(np.float64))
        # caps: planar, distances preserved
        cap_vertices = set()
        for sign in (1, -1):
            # cap vertices = those of triangles whose three vertices all face along +-z
            ctri = m.triangles[np.all(n[m.triangles][:, :, 2] * sign > 1 - 1e-6, axis=1)]
            sel = np.unique(ctri)
            cap_vertices.update(sel.tolist())
            self.assertGreater(len(sel), 24)
            rng = np.random.default_rng(1)
            for _ in range(50):
                i, j = rng.choice(sel, 2)
                self.assertAlmostEqual(float(np.linalg.norm(p[i] - p[j])), float(np.linalg.norm(uv[i] - uv[j])), places=5)
            # u x v along the outward normal: UV triangles keep the CCW-from-outside orientation
            tri = m.triangles[np.all(n[m.triangles][:, :, 2] * sign > 1 - 1e-6, axis=1)]
            ua, ub, uc = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
            det = (ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1]) - (ub[:, 1] - ua[:, 1]) * (uc[:, 0] - ua[:, 0])
            self.assertTrue(np.all(det > 0), "mirrored cap UVs")
        # wall: u = arc length, v = height
        wall = np.nonzero(np.abs(n[:, 2]) < 1e-6)[0]
        ang = np.arctan2(p[wall][:, 1], p[wall][:, 0])
        zz = p[wall][:, 2]
        for i in range(len(wall)):
            for j in range(i + 1, len(wall)):
                if abs(ang[i] - ang[j]) < 1e-9:                    # same meridian: v distance = z distance
                    self.assertAlmostEqual(abs(uv[wall[i], 1] - uv[wall[j], 1]), abs(zz[i] - zz[j]), places=5)
        self.assertAlmostEqual(float(uv[:, 0].max()), 2 * math.pi * R, places=4)
        # height v covers the profile arc length L - 2r + pi*r
        side_v = uv[[i for i in range(len(uv)) if i not in cap_vertices]][:, 1]
        self.assertAlmostEqual(float(side_v.max() - side_v.min()), L - 2 * r + math.pi * r, places=5)
        # arc spacing between neighbouring wall vertices is the chord length (to a small sag)
        ring = np.nonzero((np.abs(n[:, 2]) < 1e-6) & (np.abs(p[:, 2] - zz.max()) < 1e-9))[0]
        order = ring[np.argsort(uv[ring, 0], kind="stable")]
        du = np.diff(uv[order, 0])
        dp = np.linalg.norm(np.diff(p[order], axis=0), axis=1)
        keep = dp > 1e-9
        self.assertLess(np.abs(du[keep] / dp[keep] - 1.0).max(), 0.01)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestSphere(MeshAssertions):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios.visual_meshes import shape_mesh
        from heavy_machines.spec import sphere

        cls.shape_mesh, cls.sphere = staticmethod(shape_mesh), staticmethod(sphere)

    def test_sphere(self):
        for R in (0.03, 0.4, 1.5):
            m = self.shape_mesh(self.sphere(R))
            self.check_closed_convex(m, f"sphere {R}")
            np.testing.assert_allclose(m.points.max(0), [R] * 3, atol=1e-6)
            np.testing.assert_allclose(m.points.min(0), [-R] * 3, atol=1e-6)
            self.assertLess(np.abs(np.linalg.norm(m.points.astype(np.float64), axis=1) - R).max(), 2e-6)
            # normals are radial
            np.testing.assert_allclose(m.normals, m.points / R, atol=2e-6)
            self.assertAlmostEqual(float(m.uvs[:, 0].max()), 2 * math.pi * R, delta=1e-4 * R + 1e-5)
            self.assertAlmostEqual(float(m.uvs[:, 1].max()), math.pi * R, delta=1e-5)
            self.assertAlmostEqual(signed_volume(m) / (4 / 3 * math.pi * R ** 3), 1.0, delta=0.03)

    def test_translated(self):
        m = self.shape_mesh(self.sphere(0.4, at=(1.0, 2.0, 3.0)))
        np.testing.assert_allclose(m.points.max(0), [1.4, 2.4, 3.4], atol=1e-6)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestTireMesh(MeshAssertions):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios.visual_meshes import shape_mesh
        from heavy_machines.spec import cyl

        cls.shape_mesh, cls.cyl = staticmethod(shape_mesh), staticmethod(cyl)

    CASES = [(0.805, 0.6), (1.17, 0.66), (0.42, 0.31), (0.52, 0.46)]

    def tire(self, R, L, shoulder=0.14):
        return self.shape_mesh(self.cyl(R, L), tire_shoulder=shoulder)

    def test_bounding_box_is_exactly_radius_by_width(self):
        for R, L in self.CASES:
            m = self.tire(R, L)
            np.testing.assert_allclose(m.points.max(0), [R, R, L / 2], atol=1e-6, err_msg=f"{R} {L}")
            np.testing.assert_allclose(m.points.min(0), [-R, -R, -L / 2], atol=1e-6, err_msg=f"{R} {L}")

    def test_closed_manifold_with_outward_normals(self):
        for R, L in self.CASES:
            # not convex: the sidewall dips towards the bead, so skip the centroid test
            self.check_closed_convex(self.tire(R, L), f"tyre {R} {L}", convex=False)
            m = self.tire(R, L)
            self.assertGreater(signed_volume(m), 0.0)
            self.assertLess(signed_volume(m), math.pi * R * R * L)
            self.assertGreater(signed_volume(m), 0.80 * math.pi * R * R * L)

    def test_profile_has_crown_shoulder_and_bulging_sidewall(self):
        R, L = 0.805, 0.6
        m = self.tire(R, L)
        p = m.points.astype(np.float64)
        rho = np.hypot(p[:, 0], p[:, 1])
        z = p[:, 2]
        # crown: radius R on the centre line, smaller towards the shoulders
        self.assertAlmostEqual(float(rho[np.abs(z) < 1e-9].max()), R, places=6)
        crown = (np.abs(z) > 0.2 * L) & (np.abs(z) < 0.37 * L)
        self.assertLess(float(rho[crown].max()), R - 0.005)
        # the widest ring is the shoulder / sidewall belly; the bead is narrower
        self.assertAlmostEqual(float(np.abs(z).max()), L / 2, places=6)
        bead = (rho > 0.4 * R) & (rho < 0.5 * R)
        self.assertLess(float(np.abs(z[bead]).max()), L / 2 - 0.01)
        self.assertGreater(float(np.abs(z[bead]).max()), 0.9 * L / 2)
        # shoulders are rounded: nothing sits in the sharp corner (R, L/2)
        self.assertGreater(float(np.hypot(R - rho, L / 2 - np.abs(z)).min()), 0.0)
        corner = np.hypot((R - rho) / (0.14 * L), (L / 2 - np.abs(z)) / (0.14 * L))
        self.assertGreater(float(corner[rho > R - 0.14 * L].min()), 0.5)         # a circular fillet of that size gives 0.41
        # normals vary smoothly over the sidewall (not one flat direction) and are radial on the crown
        n = m.normals.astype(np.float64)
        side = rho < R - 0.14 * L
        self.assertGreater(float(np.abs(n[side & (z > 0)][:, 2]).min()), 0.9)
        self.assertGreater(float(np.abs(n[side & (z > 0)][:, 2]).max()) - float(np.abs(n[side & (z > 0)][:, 2]).min()), 1e-4)
        centre_line = np.abs(z) < 1e-9
        np.testing.assert_allclose(n[centre_line][:, 2], 0.0, atol=1e-9)

    def test_uv_scale_is_metres(self):
        for R, L in [(0.805, 0.6), (1.17, 0.66)]:
            m = self.tire(R, L)
            p, uv = m.points.astype(np.float64), m.uvs.astype(np.float64)
            ratios = []
            for k in range(3):
                i, j = m.triangles[:, k], m.triangles[:, (k + 1) % 3]
                d3 = np.linalg.norm(p[i] - p[j], axis=1)
                d2 = np.linalg.norm(uv[i] - uv[j], axis=1)
                ratios.append(d2[d3 > 1e-9] / d3[d3 > 1e-9])
            ratios = np.concatenate(ratios)
            self.assertAlmostEqual(float(np.median(ratios)), 1.0, delta=0.03)
            self.assertGreater(float(ratios.min()), 0.5)
            self.assertLess(float(ratios.max()), 1.6)
            # the tread is unrolled: u spans the circumference
            self.assertAlmostEqual(float(uv[:, 0].max()), 2 * math.pi * R, delta=0.02 * R + 0.1)

    def test_disabled_without_a_shoulder(self):
        plain = self.shape_mesh(self.cyl(0.8, 0.6))
        for sh in (None, 0.0):
            m = self.shape_mesh(self.cyl(0.8, 0.6), tire_shoulder=sh)
            np.testing.assert_array_equal(m.points, plain.points)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestMachines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines import registry
        from heavy_machines.scenarios import visual_meshes as vm
        from heavy_machines.spec import Cylinder

        cls.registry, cls.vm, cls.Cylinder = registry, vm, Cylinder
        cls.specs = registry.all_specs()
        cls.meshes = {name: vm.machine_visual_meshes(spec) for name, spec in cls.specs.items()}
        cls.resolved = {name: vm.resolve_coplanar_machine(spec) for name, spec in cls.specs.items()}

    def test_registry_loaded(self):
        self.assertEqual(self.registry.load_errors(), {})
        self.assertGreaterEqual(len(self.specs), 10)

    def test_every_link_every_machine(self):
        for name, spec in self.specs.items():
            d = self.meshes[name]
            self.assertEqual(set(d), {lk.name for lk in spec.links}, name)
            for lk in spec.links:
                mats = []
                for mat, mesh in d[lk.name]:
                    mats.append(mat)
                    mesh.validate()
                    self.assertGreater(len(mesh.triangles), 0)
                    self.assertEqual(mesh.rect, (0.0, 0.0, 0.0, 0.0))
                    self.assertEqual(mesh.uvs.shape, (len(mesh.points), 2))
                    self.assertTrue(np.isfinite(mesh.uvs).all())
                    self.assertEqual(mesh.triangles.dtype, np.int32)
                want = []
                for sh in lk.shapes:
                    if sh.visual and sh.material not in want:
                        want.append(sh.material)
                self.assertEqual(mats, want, f"{name}.{lk.name}: one mesh per material, first-use order")

    def test_triangle_budget(self):
        for name, d in self.meshes.items():
            tris = sum(len(m.triangles) for lst in d.values() for _, m in lst)
            self.assertLess(tris, 400_000, name)
            self.assertGreater(tris, 1000, name)

    def test_meshes_stay_inside_the_sharp_shapes(self):
        """Rounding only removes material: every mesh vertex is inside the sharp
        primitive hull bounding box of its link, and the extent shrinks little."""
        from heavy_machines import kinematics as K

        ident = K.Tf()
        for name, spec in self.specs.items():
            for lk in spec.links:
                pts = [p for s in lk.shapes if s.visual for p in K.shape_points(ident, s, 48)]
                if not pts:
                    self.assertEqual(self.meshes[name][lk.name], [])
                    continue
                lo, hi = np.min(pts, 0), np.max(pts, 0)
                allp = np.concatenate([m.points for _, m in self.meshes[name][lk.name]]).astype(np.float64)
                # coplanar faces are pushed outwards by whole millimetres (<= MAX_PUSH in total)
                self.assertTrue(np.all(allp.min(0) >= lo - MAX_PUSH), f"{name}.{lk.name}")
                self.assertTrue(np.all(allp.max(0) <= hi + MAX_PUSH), f"{name}.{lk.name}")
                span, got = hi - lo, allp.max(0) - allp.min(0)
                self.assertTrue(np.all(span - got < 0.25 * span + 0.05), f"{name}.{lk.name} {span} {got}")

    def test_invisible_shapes_are_skipped(self):
        spec = copy.deepcopy(self.specs["hydraulic_excavator"])
        lk = next(l for l in spec.links if sum(s.visual for s in l.shapes) >= 3)
        full = sum(len(m.points) for _, m in self.vm.machine_visual_meshes(spec)[lk.name])
        for s in lk.shapes[:2]:
            s.visual = False
        less = sum(len(m.points) for _, m in self.vm.machine_visual_meshes(spec)[lk.name])
        self.assertLess(less, full)
        for s in lk.shapes:
            s.visual = False
        self.assertEqual(self.vm.machine_visual_meshes(spec)[lk.name], [])

    def test_merge_preserves_shapes(self):
        """A material mesh is exactly the concatenation of its shape meshes."""
        spec = self.specs["articulated_dump_truck"]
        lk = spec.links[0]
        plain = dict(uv_jitter=False, bevel_scale=0.0, tire_shoulder=0.0, resolve_coplanar_faces=False)
        d = self.vm.machine_visual_meshes(spec, **plain)[lk.name]
        for mat, mesh in d:
            parts = [self.vm.shape_mesh(s) for s in lk.shapes if s.visual and s.material == mat]
            self.assertEqual(len(mesh.points), sum(len(m.points) for m in parts))
            self.assertEqual(len(mesh.triangles), sum(len(m.triangles) for m in parts))
            np.testing.assert_array_equal(mesh.points, np.concatenate([m.points for m in parts]))
        # whole-mesh closedness: every connected shape is closed, so the welded edge
        # graph of each material mesh has every directed edge used once per orientation
        for mat, mesh in d:
            self.assertEqual(edge_stats(mesh)[3], 0)
        # with every extra on, link meshes are still closed per material
        for mat, mesh in self.meshes["articulated_dump_truck"][lk.name]:
            self.assertEqual(edge_stats(mesh)[3], 0)

    # ---- finding 1: explicit tyre rule ---------------------------------------------------------

    def rolling(self, spec):
        from heavy_machines.spec import ROLLING_ROLES

        return {j.child: j for j in spec.joints if j.role in ROLLING_ROLES and j.wheel_radius > 0.0}

    def test_tire_rule_is_explicit_fleet_wide(self):
        for name, spec in self.specs.items():
            got = self.vm.tire_shapes(spec)
            rolling = self.rolling(spec)
            want = set()
            for lk in spec.links:
                j = rolling.get(lk.name)
                for i, sh in enumerate(lk.shapes):
                    if (j is not None and sh.visual and sh.material == "rubber" and isinstance(sh.geom, self.Cylinder)
                            and 0.9 * j.wheel_radius <= sh.geom.radius <= 1.1 * j.wheel_radius
                            and sh.geom.length <= 1.2 * sh.geom.radius):
                        want.add((lk.name, i))
            self.assertEqual(got, want, name)
            for ln, i in got:
                sh = spec.link(ln).shapes[i]
                self.assertEqual(sh.material, "rubber")
                self.assertLessEqual(sh.geom.length / sh.geom.radius, 1.2)

    def test_skid_steer_tyres_are_found_on_roll_links(self):
        spec = self.specs["skid_steer_loader"]
        tires = self.vm.tire_shapes(spec)
        self.assertEqual({ln for ln, _ in tires},
                         {"front_left_wheel", "rear_left_wheel", "front_right_wheel", "rear_right_wheel"})
        self.assertEqual({j.role for j in spec.joints if j.child in {ln for ln, _ in tires}},
                         {"roll_left", "roll_right"})
        for ln, i in tires:
            sh = spec.link(ln).shapes[i]
            self.assertEqual((sh.geom.radius, sh.geom.length), (0.42, 0.31))
            m = dict(self.meshes["skid_steer_loader"][ln])["rubber"]
            p = m.points.astype(np.float64)       # axis is link y
            rho = np.hypot(p[:, 0], p[:, 2])
            np.testing.assert_allclose(p[:, 1].max(), 0.155, atol=1e-6)
            np.testing.assert_allclose(rho.max(), 0.42, atol=1e-6)
            # a tyre: the radius falls off towards the shoulders (crown), unlike a plain cylinder wall
            self.assertLess(float(rho[(np.abs(p[:, 1]) > 0.06) & (np.abs(p[:, 1]) < 0.11)].max()), 0.42 - 1e-3)
            # tyre shoulders are much bigger than a 2.5 cm panel bevel: nothing near the sharp corner
            corner = np.hypot((0.42 - rho) / (0.14 * 0.31), (0.155 - np.abs(p[:, 1])) / (0.14 * 0.31))
            self.assertGreater(float(corner[rho > 0.42 - 0.14 * 0.31].min()), 0.5)    # a 2.5 cm bevel would give 0.24

    def test_compactor_drum_is_not_a_tyre(self):
        spec = self.specs["soil_compactor"]
        tires = self.vm.tire_shapes(spec)
        self.assertEqual({ln for ln, _ in tires}, {"rear_left_wheel", "rear_right_wheel"})
        self.assertNotIn("drum", {ln for ln, _ in tires})
        drum = spec.link("drum")
        R, L = drum.shapes[0].geom.radius, drum.shapes[0].geom.length
        d = dict(self.meshes["soil_compactor"]["drum"])
        for mat, mesh in d.items():
            self.assertGreater(len(mesh.triangles), 0)
        steel = d["steel"]
        p, n = steel.points.astype(np.float64), steel.normals.astype(np.float64)      # roller axis is y
        rho = np.hypot(p[:, 0], p[:, 2])
        own = rho > 0.70            # the steel end discs (r = 0.677) share the material; the drum is r = 0.767
        # nothing outside the sharp drum hull
        self.assertLessEqual(float(rho.max()), R + 1e-6)
        self.assertLessEqual(float(np.abs(p[own][:, 1]).max()), L / 2 + 1e-6)
        # the straight wall runs out to L/2 - 2.5 cm (plain rim fillet), not 23 cm short of the end
        wall = own & (np.abs(n[:, 1]) < 1e-6)
        self.assertAlmostEqual(float(np.abs(p[wall][:, 1]).max()), L / 2 - 0.025, places=6)
        # the drum cap stays flat out to R - 2.5 cm, reaching under the 0.737 m dark end discs
        # (with a 23 cm shoulder it would stop at 0.54 m and the discs would overhang)
        cap = own & (np.abs(np.abs(p[:, 1]) - L / 2) < 1e-6)
        self.assertAlmostEqual(float(rho[cap].max()), R - 0.025, places=6)
        dark = d["dark"].points.astype(np.float64)
        discs = dark[np.abs(dark[:, 1]) > L / 2 - 1e-6]
        self.assertGreater(len(discs), 0)
        self.assertLessEqual(float(np.hypot(discs[:, 0], discs[:, 2]).max()), R)

    def test_cylinders_that_are_not_tyres_get_no_tyre_profile(self):
        base = self.specs["articulated_dump_truck"]
        wheel = next(j for j in base.joints if j.role == "wheel")

        def plain_vs_default(spec, link):
            d = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False)[link])
            return d

        # (a) same size, but steel: no tyre
        spec = copy.deepcopy(base)
        sh = next(s for s in spec.link(wheel.child).shapes if isinstance(s.geom, self.Cylinder) and s.material == "rubber")
        sh.material = "steel"
        self.assertNotIn(wheel.child, {ln for ln, _ in self.vm.tire_shapes(spec)})
        steel = plain_vs_default(spec, wheel.child)["steel"]
        p, n = steel.points.astype(np.float64), steel.normals.astype(np.float64)
        wall = np.abs(n[:, 1]) < 1e-6
        self.assertAlmostEqual(float(np.abs(p[wall][:, 1]).max()), sh.geom.length / 2 - 0.025, places=6)
        # (b) rubber but squat / wide like a roller: no tyre
        spec = copy.deepcopy(base)
        sh = next(s for s in spec.link(wheel.child).shapes if isinstance(s.geom, self.Cylinder) and s.material == "rubber")
        sh.geom = self.Cylinder(sh.geom.radius, 2.0 * sh.geom.radius)
        self.assertNotIn(wheel.child, {ln for ln, _ in self.vm.tire_shapes(spec)})
        # (c) rubber but far from the rolling radius (a small idler): no tyre
        spec = copy.deepcopy(base)
        sh = next(s for s in spec.link(wheel.child).shapes if isinstance(s.geom, self.Cylinder) and s.material == "rubber")
        sh.geom = self.Cylinder(0.5 * sh.geom.radius, sh.geom.length)
        self.assertNotIn(wheel.child, {ln for ln, _ in self.vm.tire_shapes(spec)})
        # (d) right size and material but the link is not the child of a rolling joint
        spec = copy.deepcopy(base)
        rolling_joint = spec.joint(wheel.name)
        rolling_joint.role = "fixed"
        self.assertNotIn(wheel.child, {ln for ln, _ in self.vm.tire_shapes(spec)})
        # (e) an invisible tyre is not meshed at all
        spec = copy.deepcopy(base)
        sh = next(s for s in spec.link(wheel.child).shapes if isinstance(s.geom, self.Cylinder) and s.material == "rubber")
        sh.visual = False
        self.assertNotIn(wheel.child, {ln for ln, _ in self.vm.tire_shapes(spec)})

    def test_tyre_profile_can_be_switched_off(self):
        spec = self.specs["articulated_dump_truck"]
        wheel = next(j for j in spec.joints if j.role == "wheel")
        on = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False)[wheel.child])["rubber"]
        off = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False, tire_shoulder=0.0)[wheel.child])["rubber"]
        self.assertNotEqual(len(on.points), len(off.points))
        p, n = off.points.astype(np.float64), off.normals.astype(np.float64)
        sh = next(s for s in spec.link(wheel.child).shapes if s.material == "rubber")
        wall = np.abs(n[:, 1]) < 1e-6
        self.assertAlmostEqual(float(np.abs(p[wall][:, 1]).max()), sh.geom.length / 2 - 0.025, places=6)

    # ---- finding 3: size-adaptive panel fillets -------------------------------------------------

    def test_panel_fillets_scale_with_the_part(self):
        from heavy_machines.spec import box

        spec = copy.deepcopy(self.specs["articulated_dump_truck"])
        lk = spec.links[0]
        lk.shapes = [box(2.0, 1.4, 0.6, at=(0, 0, 0), mat="body"),
                     box(2.0, 1.4, 0.6, at=(0, 5, 0), mat="steel"),
                     box(0.3, 0.2, 0.1, at=(0, 10, 0), mat="body")]
        d = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False)[lk.name])

        def top_flat_extent(mesh):
            idx = flat_vertices(mesh, (0, 0, 1))
            return float(np.abs(mesh.points[idx].astype(np.float64)[:, 0]).max())

        body = d["body"]
        top = body.points[:, 1] < 2.5                          # the big body box (y around 0)
        big = type(body)(body.points[top], np.zeros((0, 3), np.int32), body.normals[top], body.uvs[top], (0, 0, 0, 0))
        big_idx = flat_vertices(big, (0, 0, 1))
        r_big = 1.0 - float(np.abs(big.points[big_idx].astype(np.float64)[:, 0]).max())
        self.assertAlmostEqual(r_big, 0.035 * 1.4, places=5)           # bevel_scale * middle dimension
        self.assertAlmostEqual(1.0 - top_flat_extent(d["steel"]), 0.025, places=6)   # not a panel material
        small = body.points[:, 1] > 5
        sm = type(body)(body.points[small], np.zeros((0, 3), np.int32), body.normals[small], body.uvs[small], (0, 0, 0, 0))
        sm_idx = flat_vertices(sm, (0, 0, 1))
        self.assertAlmostEqual(0.15 - float(np.abs(sm.points[sm_idx].astype(np.float64)[:, 0]).max()), 0.025, places=6)
        # bevel_scale = 0 restores the plain 2.5 cm rule everywhere
        d0 = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False, bevel_scale=0.0)[lk.name])
        b0 = d0["body"]
        t0 = b0.points[:, 1] < 2.5
        big0 = type(b0)(b0.points[t0], np.zeros((0, 3), np.int32), b0.normals[t0], b0.uvs[t0], (0, 0, 0, 0))
        i0 = flat_vertices(big0, (0, 0, 1))
        self.assertAlmostEqual(1.0 - float(np.abs(big0.points[i0].astype(np.float64)[:, 0]).max()), 0.025, places=6)

    # ---- finding 2: coplanar faces --------------------------------------------------------------

    def test_no_exposed_coplanar_faces_remain(self):
        for name in ("articulated_dump_truck", "hydraulic_excavator"):
            spec = self.specs[name]
            before = after = 0
            for lk in spec.links:
                before += len(exposed_coplanar_pairs(lk.shapes))
                resolved = self.vm.resolve_coplanar(lk.shapes)
                left = exposed_coplanar_pairs(resolved)
                self.assertEqual(left, [], f"{name}.{lk.name}")
                after += len(left)
            self.assertGreater(before, 10, f"{name}: the detector should find the reviewer's cases")
            self.assertEqual(after, 0)

    def test_module_detectors_find_nothing_after_resolution_fleet_wide(self):
        for name, spec in self.specs.items():
            for lk in spec.links:
                res = self.resolved[name][lk.name]
                self.assertEqual(self.vm.find_coplanar_pairs(res), [], f"{name}.{lk.name}")
                self.assertEqual(self.vm.find_coaxial_pairs(res), [], f"{name}.{lk.name}")

    def test_per_link_resolve_coplanar_matches_on_a_few_machines(self):
        for name in ("articulated_dump_truck", "motor_grader", "wheel_loader"):
            for lk in self.specs[name].links:
                self.assertEqual(self.vm.find_coplanar_pairs(self.vm.resolve_coplanar(lk.shapes)), [], f"{name}.{lk.name}")

    def test_coplanar_pushes_are_small_fleet_wide(self):
        """Almost every shape moves 1-2 mm; nothing moves inwards or by more than MAX_PUSH."""
        from heavy_machines.spec import Box

        sizes = []
        for name, spec in self.specs.items():
            res = self.resolved[name]
            for lk in spec.links:
                for a, b in zip(lk.shapes, res[lk.name]):
                    if a is b:
                        continue
                    if isinstance(a.geom, Box):
                        grow = np.array(b.geom.size) - np.array(a.geom.size)
                    else:
                        grow = np.array([b.geom.length - a.geom.length])
                    self.assertGreaterEqual(float(grow.min()), -1e-12, f"{name}.{lk.name}")
                    self.assertLessEqual(float(grow.max()), MAX_PUSH, f"{name}.{lk.name}")
                    sizes.append(float(grow.max()))
        sizes = np.array(sizes)
        self.assertGreater(len(sizes), 300)
        self.assertGreater(float((sizes <= 0.0021).mean()), 0.85)

    def test_cross_link_coplanar_faces_are_separated(self):
        """Spinning sprockets / pivot bosses flush with their neighbour link (HEX: 16 pairs)."""
        from heavy_machines import kinematics as K

        spec = self.specs["hydraulic_excavator"]
        tfs = K.link_transforms(spec)
        flat, frames, link_of = [], [], []
        for lk in spec.links:
            for sh in lk.shapes:
                flat.append(sh)
                frames.append(tfs[lk.name])
                link_of.append(lk.name)
        before = exposed_coplanar_pairs(flat, frames=frames, link_of=link_of)
        self.assertGreater(len(before), 3)
        res = self.resolved["hydraulic_excavator"]
        flat2 = [sh for lk in spec.links for sh in res[lk.name]]
        after = exposed_coplanar_pairs(flat2, frames=frames, link_of=link_of)
        self.assertEqual(after, [])
        for lk in spec.links:
            self.assertEqual(len(res[lk.name]), len(lk.shapes))

    def test_cross_link_resolution_respects_joint_motion(self):
        """Faces on two links are separated only when the joint keeps them coplanar in every pose."""
        from heavy_machines.spec import box, cyl

        base = copy.deepcopy(self.specs["articulated_dump_truck"])
        joint = next(j for j in base.joints if j.role == "wheel")

        def run(jtype, axis):
            spec = copy.deepcopy(base)
            for lk in spec.links:
                lk.shapes = []
            spec.link(joint.parent).shapes = [box(1.0, 1.0, 1.0, mat="steel")]
            spec.link(joint.child).shapes = [cyl(0.3, 1.0, axis="y", mat="dark")]   # cap flush with the +y / -y faces
            jt = spec.joint(joint.name)
            jt.type, jt.axis, jt.xyz, jt.rpy, jt.init = jtype, axis, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0
            return spec.link(joint.child).shapes[0], self.vm.resolve_coplanar_machine(spec)[joint.child][0]

        for jtype, axis, pushed in [("continuous", (0.0, 1.0, 0.0), True),     # spins about the face normal
                                    ("revolute", (0.0, -1.0, 0.0), True),
                                    ("fixed", (0.0, 0.0, 1.0), True),
                                    ("prismatic", (1.0, 0.0, 0.0), True),      # slides inside the face plane
                                    ("continuous", (0.0, 0.0, 1.0), False),    # tips the face out of its plane
                                    ("revolute", (1.0, 0.0, 0.0), False),
                                    ("prismatic", (0.0, 1.0, 0.0), False)]:    # slides along the normal
            orig, out = run(jtype, axis)
            if pushed:
                self.assertAlmostEqual(out.geom.length, 1.002, places=9, msg=f"{jtype} {axis}")
            else:
                self.assertIs(out, orig, f"{jtype} {axis}")

    def test_sprocket_and_track_end_walls_no_longer_coincide(self):
        res = self.resolved["hydraulic_excavator"]
        for side in ("left", "right"):
            sprocket = res[f"{side}_sprocket"][0].geom
            track_ends = [sh.geom for sh in res["undercarriage"] if isinstance(sh.geom, self.Cylinder)
                          and abs(sh.geom.radius - 0.475) < 3e-3 and abs(sh.geom.length - 0.6) < 1e-9]
            self.assertTrue(track_ends)
            self.assertAlmostEqual(sprocket.radius, 0.475, places=9)                     # untouched
            self.assertTrue(any(abs(g.radius - 0.476) < 1e-9 for g in track_ends), side)   # belt end 1 mm proud

    def test_cross_link_coaxial_walls_respect_joint_motion(self):
        from heavy_machines.spec import cyl

        base = copy.deepcopy(self.specs["articulated_dump_truck"])
        joint = next(j for j in base.joints if j.role == "wheel")

        def run(jtype, axis):
            spec = copy.deepcopy(base)
            for lk in spec.links:
                lk.shapes = []
            spec.link(joint.parent).shapes = [cyl(0.4, 1.0, axis="y", mat="steel")]
            spec.link(joint.child).shapes = [cyl(0.4, 1.4, axis="y", mat="dark")]
            jt = spec.joint(joint.name)
            jt.type, jt.axis, jt.xyz, jt.rpy, jt.init = jtype, axis, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0
            return spec.link(joint.parent).shapes[0], self.vm.resolve_coplanar_machine(spec)[joint.parent][0]

        for jtype, axis, pushed in [("continuous", (0.0, 1.0, 0.0), True),      # spins about the common axis
                                    ("fixed", (0.0, 0.0, 1.0), True),
                                    ("prismatic", (0.0, 1.0, 0.0), True),       # slides along it
                                    ("continuous", (0.0, 0.0, 1.0), False),     # swings off the axis
                                    ("prismatic", (1.0, 0.0, 0.0), False)]:     # slides off the axis
            orig, out = run(jtype, axis)
            if pushed:
                self.assertAlmostEqual(out.geom.radius, 0.401, places=9, msg=f"{jtype} {axis}")
            else:
                self.assertIs(out, orig, f"{jtype} {axis}")

    def test_coplanar_resolution_is_per_machine_consistent_with_per_link(self):
        """Machine-level resolution is a superset of the per-link one (same-link pairs always separate)."""
        spec = self.specs["articulated_dump_truck"]
        for lk in spec.links:
            self.assertEqual(self.vm.find_coplanar_pairs(self.resolved["articulated_dump_truck"][lk.name]), [], lk.name)

    def test_ring_of_identical_segments_does_not_staircase(self):
        """motor_grader 'circle': 12 identical overlapping steel segments share their faces' plane."""
        lk = self.specs["motor_grader"].link("circle")
        resolved = self.vm.resolve_coplanar(lk.shapes)
        from heavy_machines.spec import Box
        grows = [b.geom.size[2] - a.geom.size[2] for a, b in zip(lk.shapes, resolved) if isinstance(a.geom, Box)]
        self.assertGreater(max(grows), 0.0)
        self.assertLessEqual(max(grows), 0.0031)
        self.assertEqual(self.vm.find_coplanar_pairs(resolved), [])

    def test_meshes_use_the_separated_shapes(self):
        """machine_visual_meshes raises the smaller of two coplanar faces by a millimetre."""
        from heavy_machines.spec import box

        spec = copy.deepcopy(self.specs["articulated_dump_truck"])
        lk = spec.links[0]
        lk.shapes = [box(1.0, 1.0, 1.0, mat="body"), box(0.4, 0.4, 0.5, at=(0.0, 0.0, 0.25), mat="accent")]
        d = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False)[lk.name])
        self.assertAlmostEqual(float(d["body"].points[:, 2].max()), 0.5, places=6)
        self.assertAlmostEqual(float(d["accent"].points[:, 2].max()), 0.5 + 1e-3, places=6)
        off = dict(self.vm.machine_visual_meshes(spec, uv_jitter=False, resolve_coplanar_faces=False)[lk.name])
        self.assertAlmostEqual(float(off["accent"].points[:, 2].max()), 0.5, places=6)


    def test_uv_jitter(self):
        spec = self.specs["wheel_loader"]
        a = self.vm.machine_visual_meshes(spec, uv_jitter=False)
        b = self.vm.machine_visual_meshes(spec, uv_jitter=True)
        c = self.vm.machine_visual_meshes(spec, uv_jitter=True)
        shifted = 0
        for ln in a:
            for (ma, xa), (mb, xb), (mc, xc) in zip(a[ln], b[ln], c[ln]):
                np.testing.assert_array_equal(xa.points, xb.points)
                np.testing.assert_array_equal(xb.uvs, xc.uvs)            # deterministic
                self.assertGreaterEqual(float(xa.uvs.min()), 0.0)
                if not np.allclose(xa.uvs, xb.uvs):
                    shifted += 1
        self.assertGreater(shifted, 0)

    def test_uv_density_is_one_per_metre(self):
        """Mean UV edge length / mesh edge length over every mesh of a machine is ~1."""
        d = self.meshes["hydraulic_excavator"]
        num = den = 0.0
        for lst in d.values():
            for _, m in lst:
                p, uv = m.points.astype(np.float64), m.uvs.astype(np.float64)
                for k in range(3):
                    i, j = m.triangles[:, k], m.triangles[:, (k + 1) % 3]
                    d3 = np.linalg.norm(p[i] - p[j], axis=1)
                    d2 = np.linalg.norm(uv[i] - uv[j], axis=1)
                    ok = d3 > 1e-9
                    num += d2[ok].sum()
                    den += d3[ok].sum()
        self.assertAlmostEqual(num / den, 1.0, delta=0.03)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestResolveCoplanar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios import visual_meshes as vm
        from heavy_machines.spec import box, cyl

        cls.vm, cls.box, cls.cyl = vm, staticmethod(box), staticmethod(cyl)

    def top(self, shape):
        from heavy_machines import kinematics as K

        pts = np.array(K.shape_points(K.Tf(), shape, 32))
        return float(pts[:, 2].max())

    def test_smaller_face_wins_regardless_of_order(self):
        big, small = self.box(1.0, 1.0, 1.0), self.box(0.4, 0.4, 0.5, at=(0, 0, 0.25))
        for order in ((big, small), (small, big)):
            out = self.vm.resolve_coplanar(list(order))
            tops = [self.top(o) for o in out]
            i_small = order.index(small)
            self.assertAlmostEqual(tops[i_small], 0.501, places=9)
            self.assertAlmostEqual(tops[1 - i_small], 0.5, places=9)
            self.assertIs(out[1 - i_small], order[1 - i_small])
            # the other faces of the pushed box did not move
            sh = out[i_small]
            self.assertAlmostEqual(sh.geom.size[2], 0.501, places=9)
            self.assertAlmostEqual(sh.xyz[2] - sh.geom.size[2] / 2, 0.0, places=9)

    def test_ties_go_to_the_later_shape_and_pair_is_separated(self):
        a, b = self.box(1.0, 1.0, 1.0), self.box(1.0, 1.0, 1.0)
        out = self.vm.resolve_coplanar([a, b])
        self.assertIs(out[0], a)
        np.testing.assert_allclose(out[1].geom.size, (1.002, 1.002, 1.002), atol=1e-9)
        np.testing.assert_allclose(out[1].xyz, (0, 0, 0), atol=1e-12)

    def test_nested_plates_stack_by_size(self):
        shapes = [self.box(1.0, 1.0, 1.0), self.box(0.5, 0.5, 0.5, at=(0, 0, 0.25)), self.box(0.2, 0.2, 0.4, at=(0, 0, 0.3))]
        out = self.vm.resolve_coplanar(shapes)
        tops = [self.top(o) for o in out]
        self.assertAlmostEqual(tops[0], 0.5, places=9)
        self.assertAlmostEqual(tops[1], 0.501, places=9)
        self.assertAlmostEqual(tops[2], 0.502, places=9)
        self.assertEqual(self.vm.find_coplanar_pairs(out), [])

    def test_ring_needs_only_two_levels(self):
        n = 12
        shapes = [self.box(0.16, 0.418, 0.09, at=(0.7 * math.cos(2 * math.pi * k / n), 0.7 * math.sin(2 * math.pi * k / n), 0.0),
                           rpy=(0, 0, 2 * math.pi * k / n)) for k in range(n)]
        self.assertGreater(len(self.vm.find_coplanar_pairs(shapes)), n)
        out = self.vm.resolve_coplanar(shapes)
        self.assertEqual(self.vm.find_coplanar_pairs(out), [])
        grow = [o.geom.size[2] - s.geom.size[2] for o, s in zip(out, shapes)]
        self.assertLessEqual(max(grow), 0.0021)

    def test_cylinder_cap_against_box_face_and_rotated_box(self):
        base = self.box(2.0, 2.0, 1.0)
        cap = self.cyl(0.3, 0.6, at=(0.2, 0.1, 0.2))                  # cap at z = 0.5, coplanar with the top face
        out = self.vm.resolve_coplanar([base, cap])
        self.assertAlmostEqual(out[1].geom.length, 0.601, places=9)
        self.assertAlmostEqual(out[1].xyz[2], 0.2005, places=9)
        # a box rotated about the face normal: still coplanar in the link frame
        rot = self.box(0.5, 0.3, 0.5, at=(-0.4, 0.2, 0.25), rpy=(0, 0, 0.7))
        out = self.vm.resolve_coplanar([base, rot])
        self.assertAlmostEqual(out[1].geom.size[2], 0.501, places=9)
        # a box tilted about x has a different normal: untouched
        tilt = self.box(0.5, 0.3, 0.5, at=(-0.4, 0.2, 0.25), rpy=(0.3, 0, 0))
        out = self.vm.resolve_coplanar([base, tilt])
        self.assertIs(out[1], tilt)

    def test_coaxial_cylinder_walls_are_separated(self):
        """Equal-radius coaxial cylinders (sprocket inside a track end) share a curved surface."""
        track = self.cyl(0.475, 0.6, axis="y", mat="rubber")
        sprocket = self.cyl(0.475, 0.62, axis="y", mat="dark")
        self.assertEqual(len(self.vm.find_coaxial_pairs([track, sprocket])), 1)
        for order in ((track, sprocket), (sprocket, track)):
            out = self.vm.resolve_coplanar(list(order))
            i_track = order.index(track)
            self.assertAlmostEqual(out[i_track].geom.radius, 0.476, places=9)      # narrower wall is the visible one
            self.assertIs(out[1 - i_track], order[1 - i_track])
            self.assertEqual(out[i_track].geom.length, 0.6)
            self.assertEqual(self.vm.find_coaxial_pairs(out), [])
        # a chain of three equal cylinders stacks by wall area
        a, b, c = self.cyl(0.3, 1.0), self.cyl(0.3, 0.6), self.cyl(0.3, 0.3)
        out = self.vm.resolve_coplanar([a, b, c])
        np.testing.assert_allclose([o.geom.radius for o in out], [0.3, 0.301, 0.302], atol=1e-9)

    def test_coaxial_untouched_cases(self):
        base = self.cyl(0.3, 1.0)
        for other in (self.cyl(0.3, 0.8, at=(0.01, 0, 0)),                 # axes 1 cm apart
                      self.cyl(0.31, 0.8),                                  # different radius
                      self.cyl(0.3, 1.0, at=(0, 0, 1.2)),                   # no axial overlap
                      self.cyl(0.3, 1.0, axis="x"),                         # not parallel
                      self.cyl(0.3, 1.0, visual=False)):
            out = self.vm.resolve_coplanar([base, other])
            self.assertIs(out[0], base)
            self.assertIs(out[1], other)
        # sub-millimetre radius differences are separated
        out = self.vm.resolve_coplanar([base, self.cyl(0.3004, 0.5)])
        self.assertGreaterEqual(out[1].geom.radius - out[0].geom.radius, 5e-4 - 1e-12)

    def test_untouched_cases(self):
        a = self.box(1.0, 1.0, 1.0)
        shapes = [
            a,
            self.box(1.0, 1.0, 1.0, at=(1.0, 0, 0)),                 # flush neighbour: coplanar but no overlap
            self.box(1.0, 1.0, 1.0, at=(0, 0, 1.0)),                 # stacked: faces touch with opposite normals
            self.box(0.5, 0.5, 0.5, at=(0, 0, 0.3)),                 # top 0.55 (not coplanar with 0.5)
            self.box(0.5, 0.5, 0.5, at=(5, 0, 0.25)),                # far away
            self.box(0.5, 0.5, 0.5, at=(0, 0, 0.25), visual=False),    # invisible shapes are ignored
        ]
        out = self.vm.resolve_coplanar(shapes)
        for i in (0, 1, 3, 4, 5):
            self.assertIs(out[i], shapes[i], i)
        self.assertIs(out[2], shapes[2])

    def test_sub_millimetre_offsets_are_separated(self):
        big = self.box(1.0, 1.0, 1.0)
        for dz in (-0.0004, -0.0001, 0.0001, 0.0004):
            small = self.box(0.4, 0.4, 0.5 + dz, at=(0, 0, 0.25 + dz / 2))
            out = self.vm.resolve_coplanar([big, small])
            self.assertGreaterEqual(self.top(out[1]) - 0.5, 5e-4 - 1e-12, dz)
        # a face already 1 mm proud is left alone
        small = self.box(0.4, 0.4, 0.501, at=(0, 0, 0.2505))
        self.assertIs(self.vm.resolve_coplanar([big, small])[1], small)

    def test_idempotent(self):
        shapes = [self.box(1.0, 1.0, 1.0), self.box(0.5, 0.5, 0.5, at=(0, 0, 0.25)), self.box(0.2, 0.2, 0.4, at=(0, 0, 0.3))]
        once = self.vm.resolve_coplanar(shapes)
        twice = self.vm.resolve_coplanar(once)
        for a, b in zip(once, twice):
            self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
