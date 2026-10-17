"""Checks of the HDR sky tools (numpy only; two 4k skies are downloaded once and cached).

    python scripts/drone_scene/test/test_hdr.py

The skies (~44 MB) are cached in ``$DRONE_SCENE_HDR_CACHE`` or ``<tmp>/drone_scene_hdr_cache``; the tests
that need them are skipped when the asset server is unreachable.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import importlib.util
import itertools
import math
import os
import re
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drone_scene import hdr as H  # noqa: E402

CACHE = os.environ.get("DRONE_SCENE_HDR_CACHE") or os.path.join(tempfile.gettempdir(), "drone_scene_hdr_cache")
NOON_GRASS = H.NUCLEUS_ROOT + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr"
KLOOFENDAL = H.NUCLEUS_ROOT + "/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"

_real: dict[str, tuple[str, np.ndarray]] = {}


def real_sky(url: str) -> tuple[str, np.ndarray]:
    """(local path, image) of a downloaded sky, read once for the whole run."""
    if url not in _real:
        try:
            path = H.fetch(url, CACHE)
        except (urllib.error.URLError, OSError) as e:
            raise unittest.SkipTest(f"asset server unreachable: {e}")
        _real[url] = (path, H.read_hdr(path))
    return _real[url]


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def disk_sky(h, w, az_deg, el_deg, radius_deg=1.5, rgb=(1.0e5, 0.9e5, 0.7e5), base=1.0):
    """Uniform sky of radiance ``base`` plus a disk (pixel centres within ``radius_deg``) of radiance ``rgb``."""
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    dirs = H.direction_from_uv((jj + 0.5) / w, (ii + 0.5) / h)
    d0 = H.direction_from_uv(0.5 + az_deg / 360.0, 0.5 - el_deg / 180.0)
    inside = dirs @ d0 >= math.cos(math.radians(radius_deg))
    img = np.full((h, w, 3), base, np.float32)
    img[inside] = np.asarray(rgb, np.float32)
    return img, d0, inside


def rgbe_body(rgbe, rle):
    """Independent (naive) RGBE scanline encoder for crafted test files; ``rgbe`` is (h, w, 4) uint8."""
    w = rgbe.shape[1]
    out = bytearray()
    for row in rgbe:
        if not rle:
            out += row.tobytes()
            continue
        out += bytes((2, 2, w >> 8, w & 255))
        for c in range(4):
            lit: list[int] = []

            def flush():
                while lit:
                    out.append(len(lit[:128]))
                    out.extend(lit[:128])
                    del lit[:128]
            for val, grp in itertools.groupby(row[:, c].tolist()):
                n = len(list(grp))
                if n >= 3:
                    flush()
                    while n:
                        k = min(n, 127)
                        out += bytes((128 + k, val))
                        n -= k
                else:
                    lit.extend([val] * n)
            flush()
    return bytes(out)


def rgbe_bytes(h, w, rgbe, rle, header=b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n", res=None):
    return header + (res or f"-Y {h} +X {w}\n".encode()) + rgbe_body(rgbe, rle)


def rgbe_values(rgbe):
    return (rgbe[..., :3] * np.ldexp(1.0, rgbe[..., 3].astype(int) - 136)[..., None]).astype(np.float32)


class RgbeFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name="a.hdr"):
        return os.path.join(self.tmp.name, name)

    def test_decodes_flat_and_rle_scanlines(self):
        rng = np.random.default_rng(1)
        rgbe = rng.integers(1, 255, (3, 20, 4), dtype=np.uint8)
        rgbe[..., 3] = rng.integers(100, 160, (3, 20))
        rgbe[0, 3:15, 0] = 77                      # runs in some channels, literals in the others
        rgbe[1, :, 1] = 5
        rgbe[2, 2:6, 2] = 9
        for rle in (False, True):
            with open(self.path(), "wb") as f:
                f.write(rgbe_bytes(3, 20, rgbe, rle))
            np.testing.assert_array_equal(H.read_hdr(self.path()), rgbe_values(rgbe), err_msg=f"rle={rle}")

    def test_long_runs_and_mixed_scanlines(self):
        w = 300
        rgbe = np.zeros((3, w, 4), np.uint8)
        rgbe[0, :, :3] = 100                       # one run of 300 > 127: several chunks
        rgbe[0, :, 3] = 130
        rgbe[1:, :, :3] = np.random.default_rng(2).integers(1, 255, (2, w, 3))
        rgbe[1:, :, 3] = 131
        with open(self.path(), "wb") as f:         # RLE, flat, RLE scanlines in one file
            f.write(b"#?RADIANCE\n\n-Y 3 +X 300\n" + rgbe_body(rgbe[:1], True) + rgbe_body(rgbe[1:2], False)
                    + rgbe_body(rgbe[2:], True))
        np.testing.assert_array_equal(H.read_hdr(self.path()), rgbe_values(rgbe))

    def test_header_variants(self):
        rgbe = np.zeros((2, 3, 4), np.uint8)
        rgbe[..., 3] = 129                       # value = mantissa / 128
        rgbe[0, :, 0] = (128, 64, 32)
        rgbe[1, :, 0] = (16, 8, 4)
        base = np.array([[1.0, 0.5, 0.25], [0.125, 0.0625, 0.03125]], np.float32)

        def read(header, res):
            with open(self.path(), "wb") as f:
                f.write(rgbe_bytes(2, 3, rgbe, False, header=header, res=res))
            return H.read_hdr(self.path())[..., 0]

        np.testing.assert_array_equal(read(b"#?RGBE\n\n", b"-Y 2 +X 3\n"), base)
        np.testing.assert_array_equal(read(b"#?RADIANCE\r\n# comment\r\nFORMAT=32-bit_rle_rgbe\r\n\r\n", b"-Y 2 +X 3\r\n"), base)
        np.testing.assert_array_equal(read(b"#?RADIANCE\n\n", b"+Y 2 +X 3\n"), base[::-1])
        np.testing.assert_array_equal(read(b"#?RADIANCE\n\n", b"-Y 2 -X 3\n"), base[:, ::-1])
        # EXPOSURE lines are cumulative and divided out
        np.testing.assert_array_equal(read(b"#?RADIANCE\nEXPOSURE=2\nEXPOSURE=4\n\n", b"-Y 2 +X 3\n"), base / 8)

    def test_rejects_bad_files(self):
        good = rgbe_bytes(2, 3, np.ones((2, 3, 4), np.uint8), False)
        for name, data in [("magic", b"P6\n1 1\n255\n\0\0\0"),
                           ("xyze", good.replace(b"rgbe", b"xyze")),
                           ("transposed", good.replace(b"-Y 2 +X 3", b"+X 3 -Y 2")),
                           ("truncated", good[:-5]),
                           ("header", b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n")]:
            with open(self.path(), "wb") as f:
                f.write(data)
            with self.assertRaises(ValueError, msg=name):
                H.read_hdr(self.path())

    def test_absurd_headers_fail_cleanly(self):
        # a header that promises far more pixels than the file holds must not allocate them (37 GiB here)
        for name, res in [("huge", b"-Y 100000 +X 100000"), ("huge flat", b"-Y 1000000 +X 7"), ("zero", b"-Y 0 +X 4"),
                          ("negative", b"-Y -3 +X 4")]:
            with open(self.path(), "wb") as f:
                f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + res + b"\n" + bytes(100))
            with self.assertRaises(ValueError, msg=name):
                H.read_hdr(self.path())

    def test_old_style_rle_is_named_in_the_error(self):
        # Radiance's pre-2.2 run-length code is a (1, 1, 1, count) pixel; the file is shorter than a flat one
        pixels = bytes((100, 100, 100, 130, 1, 1, 1, 5, 50, 50, 50, 130))
        with open(self.path(), "wb") as f:
            f.write(b"#?RADIANCE\n\n-Y 1 +X 10\n" + pixels)
        with self.assertRaisesRegex(ValueError, "old-style"):
            H.read_hdr(self.path())
        with open(self.path(), "wb") as f:                       # a plain truncation does not blame the format
            f.write(b"#?RADIANCE\n\n-Y 1 +X 10\n" + pixels.replace(b"\x01\x01\x01", b"\x07\x07\x07"))
        with self.assertRaises(ValueError) as cm:
            H.read_hdr(self.path())
        self.assertNotIn("old-style", str(cm.exception))

    def test_round_trip_within_rgbe_precision(self):
        rng = np.random.default_rng(3)
        img = np.exp(rng.uniform(math.log(1e-4), math.log(2e5), (24, 61, 3))).astype(np.float32)
        img[0, 0] = 0.0
        img[1, :20] = 3.0                          # runs
        img[2, 5] = (1e5, 1e-3, 5.0)               # colourful pixel: small channels quantise to the big one's step
        img[3, 3] = 255.5 / 256 * 2.0 ** 10        # mantissa rounds up to 256: exponent bump
        for rle in (True, False):
            p = self.path(f"rt{int(rle)}.hdr")
            H.write_hdr(p, img, rle=rle)
            back = H.read_hdr(p)
            self.assertEqual(back.shape, img.shape)
            err = np.abs(back - img)
            bound = img.max(axis=-1, keepdims=True) * 2.0 ** -8 * 1.001
            self.assertTrue((err <= bound).all(), f"rle={rle} worst {(err / np.maximum(bound, 1e-30)).max():.3f} of the bound")
            np.testing.assert_array_equal(back[0, 0], 0.0)

    def test_writer_edge_cases(self):
        img = np.array([[[-1.0, np.nan, 2.0], [np.inf, 0.0, 1e-45]]], np.float32)   # width 2: flat
        p = self.path()
        H.write_hdr(p, img)
        back = H.read_hdr(p)
        self.assertTrue(np.isfinite(back).all())
        self.assertEqual(tuple(back[0, 0]), (0.0, 0.0, 2.0))
        self.assertGreater(back[0, 1, 0], 1e38)
        with self.assertRaises(ValueError):
            H.write_hdr(p, np.zeros((4, 4)))

    def test_rle_output_is_standard_layout(self):
        img = np.full((4, 40, 3), 0.5, np.float32)
        p = self.path()
        H.write_hdr(p, img)
        with open(p, "rb") as f:
            data = f.read()
        self.assertTrue(data.startswith(b"#?RADIANCE\n"))
        self.assertIn(b"FORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 40\n", data)
        body = data[data.index(b"+X 40\n") + 6:]
        self.assertEqual(body[:4], bytes((2, 2, 0, 40)))
        self.assertLess(len(body), 4 * 40 * 4)           # a constant image must compress

    @unittest.skipUnless(importlib.util.find_spec("mitsuba"), "mitsuba not installed")
    def test_agrees_with_mitsuba_reader(self):
        import mitsuba as mi
        if mi.variant() is None:
            mi.set_variant("scalar_rgb")
        rng = np.random.default_rng(4)
        img = np.exp(rng.uniform(math.log(1e-3), math.log(1e5), (32, 70, 3))).astype(np.float32)
        for rle in (True, False):
            p = self.path(f"m{int(rle)}.hdr")
            H.write_hdr(p, img, rle=rle)
            theirs = np.array(mi.Bitmap(p))[..., :3]
            np.testing.assert_array_equal(theirs, H.read_hdr(p), err_msg=f"rle={rle}")

    def test_fetch_caches(self):
        src = self.path("src.hdr")
        H.write_hdr(src, np.ones((2, 8, 3), np.float32))
        cache = os.path.join(self.tmp.name, "cache")
        logs: list[str] = []
        url = "file://" + src
        p1 = H.fetch(url, cache, log=logs.append)
        os.remove(src)
        self.assertEqual(H.fetch(url, cache, log=logs.append), p1)      # served from the cache
        self.assertEqual(len(logs), 1)
        self.assertEqual(H.read_hdr(p1).shape, (2, 8, 3))

    def test_fetch_keeps_files_with_the_same_name_apart(self):
        # the asset server has e.g. Clear/sunflowers_4k.hdr and Cloudy/sunflowers_4k.hdr
        urls = []
        for sub, value in (("a", 1.0), ("b", 5.0)):
            os.makedirs(self.path(sub))
            H.write_hdr(self.path(f"{sub}/same.hdr"), np.full((2, 8, 3), value, np.float32))
            urls.append("file://" + self.path(f"{sub}/same.hdr"))
        cache = self.path("cache")
        paths = [H.fetch(u, cache) for u in urls]
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual([float(H.read_hdr(p).mean()) for p in paths], [1.0, 5.0])
        self.assertTrue(all(p.endswith("same.hdr") for p in paths))

    def test_fetch_does_not_cache_a_truncated_download(self):
        src = self.path("src.hdr")
        H.write_hdr(src, np.ones((4, 64, 3), np.float32))
        with open(src, "rb") as f:
            body = f.read()
        state = {"truncate": True}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[:len(body) // 2] if state["truncate"] else body)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/sky.hdr"
        cache = self.path("cache")
        with unittest.mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}):
            with self.assertRaises(OSError):
                H.fetch(url, cache)
            self.assertEqual(os.listdir(cache), [])                   # no half file, not even a .part
            state["truncate"] = False                                 # the next attempt is not served the corpse
            p = H.fetch(url, cache)
        self.assertEqual(H.read_hdr(p).shape, (4, 64, 3))


class Geometry(unittest.TestCase):
    def test_convention(self):
        d = H.direction_from_uv
        np.testing.assert_allclose(d(0.5, 0.5), (1, 0, 0), atol=1e-12)        # centre of the image: +X
        np.testing.assert_allclose(d(0.75, 0.5), (0, 1, 0), atol=1e-12)       # to the right: +X towards +Y (ccw from above)
        np.testing.assert_allclose(d(0.25, 0.5), (0, -1, 0), atol=1e-12)
        np.testing.assert_allclose(d(0.0, 0.5), (-1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(d(0.3, 0.0), (0, 0, 1), atol=1e-12)        # top row: zenith
        np.testing.assert_allclose(d(0.3, 1.0), (0, 0, -1), atol=1e-12)
        np.testing.assert_allclose(d(0.5, 0.25), (math.sqrt(0.5), 0, math.sqrt(0.5)), atol=1e-12)
        u, v = H.uv_from_direction((0.0, 1.0, 0.0))
        self.assertAlmostEqual(float(u), 0.75)
        self.assertAlmostEqual(float(v), 0.5)

    def test_is_the_mirror_of_a_usd_dome_light(self):
        # UsdLuxDomeLight (pxr .../usdLux/schema.usda, Z-up and poleAxis "scene"): the longitude falls as u
        # grows and latitude 0, longitude 0 (u = 0.5) points along -Y.  The module docstring states how the
        # two relate; this pins it so the docstring cannot drift from the code.
        def usd_dir(u, v):
            az = -2.0 * math.pi * (np.asarray(u, float) - 0.25)
            el = (0.5 - np.asarray(v, float)) * math.pi
            return np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) + 0 * az), axis=-1)

        np.testing.assert_allclose(usd_dir(0.25, 0.5), (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(usd_dir(0.5, 0.5), (0, -1, 0), atol=1e-12)
        np.testing.assert_allclose(usd_dir(0.75, 0.5), (-1, 0, 0), atol=1e-12)
        # so no rotation maps one onto the other (u = 0.75 is +Y here but a quarter turn clockwise there) ...
        self.assertFalse(np.allclose(usd_dir(0.75, 0.5), H.direction_from_uv(0.75, 0.5)))
        # ... but the image flipped left-right on a dome turned by +90 deg about Z does
        u, v = np.linspace(0.0, 1.0, 37)[:, None], np.linspace(0.0, 1.0, 5)[None, :]
        rot_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(usd_dir(1.0 - u, v) @ rot_z90.T, H.direction_from_uv(u, v), atol=1e-12)

    def test_uv_direction_are_inverse(self):
        rng = np.random.default_rng(5)
        u, v = rng.uniform(0.0, 1.0, 5000), rng.uniform(0.01, 0.99, 5000)
        d = H.direction_from_uv(u, v)
        self.assertEqual(d.shape, (5000, 3))
        np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
        u2, v2 = H.uv_from_direction(d)
        du = np.abs(u2 - u)
        np.testing.assert_allclose(np.minimum(du, 1 - du), 0.0, atol=1e-9)    # u = 0 and 1 are the same direction
        np.testing.assert_allclose(v2, v, atol=1e-9)
        self.assertTrue(((u2 >= 0) & (u2 < 1)).all())
        # and the other way round, with unnormalised input
        x = rng.normal(size=(5000, 3)) * 7.0
        np.testing.assert_allclose(H.direction_from_uv(*H.uv_from_direction(x)),
                                   x / np.linalg.norm(x, axis=-1, keepdims=True), atol=1e-9)
        # broadcasting: a column of u against a row of v
        self.assertEqual(H.direction_from_uv(np.linspace(0, 1, 5)[:, None], np.linspace(0, 1, 3)[None, :]).shape, (5, 3, 3))

    def test_solid_angles_sum_to_4pi(self):
        for h, w in [(2048, 4096), (64, 128), (17, 33), (1, 8)]:
            sa = H.solid_angle_map(h, w)
            self.assertEqual(sa.shape, (h, w))
            self.assertAlmostEqual(float(sa.sum()) / (4 * math.pi), 1.0, delta=1e-3)
        sa = H.solid_angle_map(512, 1024)
        self.assertTrue((sa > 0).all())
        self.assertLess(sa[0, 0], sa[256, 0] / 50)                            # tiny near the poles
        np.testing.assert_allclose(sa[:, 0], sa[:, 500])

    def test_luminance_is_rec709(self):
        self.assertAlmostEqual(float(H.luminance([1.0, 0.0, 0.0])), 0.2126)
        self.assertAlmostEqual(float(H.luminance([0.0, 1.0, 0.0])), 0.7152)
        self.assertAlmostEqual(float(H.luminance([0.0, 0.0, 1.0])), 0.0722)
        self.assertEqual(H.luminance(np.ones((4, 5, 3), np.float32)).shape, (4, 5))
        np.testing.assert_allclose(H.luminance(np.full((2, 2, 3), 3.0)), 3.0, rtol=1e-6)


class Irradiance(unittest.TestCase):
    def test_uniform_sky_is_pi(self):
        sky = np.ones((256, 512, 3), np.float32)
        for n in [(0, 0, 1), (1, 0, 0), (0, -1, 0), (1, 2, 3), (0, 0, -1)]:
            np.testing.assert_allclose(H.irradiance(sky, n), math.pi, rtol=1e-2, err_msg=str(n))
        np.testing.assert_allclose(H.irradiance(sky * np.float32([1, 2, 3])), (math.pi, 2 * math.pi, 3 * math.pi), rtol=1e-2)

    def test_patch_follows_cosine_law_with_right_handedness(self):
        img, d0, inside = disk_sky(512, 1024, 70.0, 25.0, radius_deg=2.0, rgb=(1, 1, 1), base=0.0)
        e_normal = float(H.irradiance(img, d0)[0])
        self.assertAlmostEqual(e_normal, float(H.solid_angle_map(512, 1024)[inside].sum()), delta=2e-3 * e_normal)
        for n in [(0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 1, 1), (0, 0, -1)]:
            nn = np.asarray(n, float) / np.linalg.norm(n)
            self.assertAlmostEqual(float(H.irradiance(img, n)[0]), e_normal * max(0.0, float(nn @ d0)), delta=2e-3 * e_normal, msg=str(n))

    def test_mask_restricts_the_integral(self):
        img, d0, inside = disk_sky(256, 512, 0.0, 40.0)
        whole = H.irradiance(img)
        np.testing.assert_allclose(H.irradiance(img, mask=inside) + H.irradiance(img, mask=~inside), whole, rtol=1e-9)
        np.testing.assert_allclose(H.irradiance(img, mask=np.zeros(inside.shape, bool)), 0.0)
        with self.assertRaises(ValueError):
            H.irradiance(img, mask=inside[:10])
        with self.assertRaises(ValueError):
            H.irradiance(img, (0, 0, 0))


class SyntheticSun(unittest.TestCase):
    H_, W_ = 1024, 2048

    def check(self, az, el, radius=1.5):
        img, d0, inside = disk_sky(self.H_, self.W_, az, el, radius)
        sun = H.find_sun(img)
        self.assertIsNotNone(sun, f"az {az} el {el}")
        self.assertLess(angle_deg(sun.direction, d0), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(sun.direction)), 1.0, places=9)
        self.assertAlmostEqual(sun.elevation_deg, el, delta=0.5)
        self.assertAlmostEqual((sun.azimuth_deg - az + 180) % 360 - 180, 0.0, delta=0.5 / max(math.cos(math.radians(el)), 0.05))
        # irradiance: the pixels of the disk (exact), and the analytic disk (pixelisation error)
        sa = H.solid_angle_map(self.H_, self.W_)
        exact = np.asarray((1.0e5, 0.9e5, 0.7e5)) * sa[inside].sum()
        np.testing.assert_allclose(sun.irradiance, exact, rtol=5e-3)
        analytic = np.asarray((1.0e5, 0.9e5, 0.7e5)) * 2 * math.pi * (1 - math.cos(math.radians(radius)))
        np.testing.assert_allclose(sun.irradiance, analytic, rtol=0.05)
        self.assertEqual(sun.mask.shape, (self.H_, self.W_))
        self.assertTrue(sun.mask[inside].all())
        self.assertLess(sun.solid_angle, 4 * sa[inside].sum())
        self.assertAlmostEqual(sun.solid_angle, float(sa[sun.mask].sum()), places=9)
        # remove_sun: back to ~1 everywhere, nothing else touched, no NaNs
        out = H.remove_sun(img, sun)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.dtype, img.dtype)
        np.testing.assert_allclose(out[sun.mask], 1.0, atol=0.02)
        np.testing.assert_array_equal(out[~sun.mask], img[~sun.mask])
        self.assertFalse(np.shares_memory(out, img))
        self.assertGreater(float(img.max()), 1e4)                              # the input was not modified
        return sun

    def test_recovers_direction_and_irradiance(self):
        sun = self.check(40.0, 35.0)
        print(f"\n  synthetic sun: direction {np.round(sun.direction, 4)}, elevation {sun.elevation_deg:.3f}, "
              f"azimuth {sun.azimuth_deg:.3f}, irradiance {np.round(sun.irradiance, 1)}, {sun.solid_angle:.3g} sr")

    def test_other_places(self):
        for az, el in [(180.0, 20.0), (-179.5, 20.0), (0.0, 3.0), (60.0, 8.0), (-120.0, 88.0), (10.0, 89.8)]:
            with self.subTest(az=az, el=el):
                self.check(az, el)

    def test_small_sun_and_downsampled_sky(self):
        img, d0, _ = disk_sky(self.H_, self.W_, 25.0, 50.0, 0.4)           # a few pixels wide
        sun = H.find_sun(img)
        self.assertLess(angle_deg(sun.direction, d0), 0.5)
        small = H.downsample(img, 8)
        sun_small = H.find_sun(small)
        self.assertIsNotNone(sun_small)
        self.assertLess(angle_deg(sun_small.direction, d0), 1.5)
        self.assertEqual(sun_small.mask.shape, small.shape[:2])
        self.assertTrue(np.isfinite(H.remove_sun(small, sun_small)).all())

    def test_no_clear_sun(self):
        self.assertIsNone(H.find_sun(np.ones((128, 256, 3), np.float32)))
        self.assertIsNone(H.find_sun(np.zeros((128, 256, 3), np.float32)))
        rng = np.random.default_rng(6)
        overcast = rng.uniform(0.8, 1.2, (128, 256, 3)).astype(np.float32)
        self.assertIsNone(H.find_sun(overcast))
        weak, _, _ = disk_sky(256, 512, 0.0, 40.0, 2.0, rgb=(30, 30, 30))      # 30x the sky: below the 50x default
        self.assertIsNone(H.find_sun(weak))
        self.assertIsNotNone(H.find_sun(weak, min_peak_ratio=10))
        ground = np.ones((256, 512, 3), np.float32)
        ground[128:] = 1e4                                                      # a bright ground is not a sun
        ground[200, 100] = 1e6
        self.assertIsNone(H.find_sun(ground))

    def test_sun_in_noisy_sky_with_hot_pixel(self):
        img, d0, inside = disk_sky(self.H_, self.W_, -60.0, 30.0)
        img *= np.random.default_rng(7).uniform(0.7, 1.3, img.shape[:2])[..., None].astype(np.float32)
        img[300, 500] = 3e4                                                    # a lone hot pixel, dimmer than the sun
        sun = H.find_sun(img)
        self.assertLess(angle_deg(sun.direction, d0), 0.5)
        self.assertTrue(np.isfinite(H.remove_sun(img, sun)).all())

    def test_a_firefly_brighter_than_the_sun_is_not_the_sun(self):
        # one pixel at 20x the sun's radiance: it holds less light than the sun, which spans many pixels
        for radius in (1.5, 0.6):
            with self.subTest(radius=radius):
                img, d0, inside = disk_sky(self.H_, self.W_, 20.0, 40.0, radius)
                img[200, 1500] = 2e6
                sun = H.find_sun(img)
                self.assertIsNotNone(sun)
                self.assertLess(angle_deg(sun.direction, d0), 0.5)
                self.assertTrue(sun.mask[inside].all())
                self.assertFalse(sun.mask[200, 1500])
                sa = H.solid_angle_map(self.H_, self.W_)
                np.testing.assert_allclose(sun.irradiance, np.asarray((1.0e5, 0.9e5, 0.7e5)) * sa[inside].sum(), rtol=5e-3)

    def test_smooth_deg_sets_the_peak_search_box(self):
        img, d0, _ = disk_sky(self.H_, self.W_, 20.0, 40.0, 1.5)
        img[200, 1500] = 2e6
        self.assertGreater(angle_deg(H.find_sun(img, smooth_deg=0.0).direction, d0), 10.0)    # 3x3 box: the firefly wins
        self.assertLess(angle_deg(H.find_sun(img, smooth_deg=0.5).direction, d0), 0.5)

    def test_remove_sun_keeps_extra_channels(self):
        img, _, _ = disk_sky(256, 512, 0.0, 40.0)
        rgba = np.concatenate((img, np.full(img.shape[:2] + (1,), 0.25, np.float32)), axis=2)
        sun = H.find_sun(rgba)
        out = H.remove_sun(rgba, sun)
        self.assertEqual(out.shape, rgba.shape)
        np.testing.assert_array_equal(out[..., 3], rgba[..., 3])
        np.testing.assert_array_equal(out[..., :3], H.remove_sun(img, sun))

    def test_remove_sun_checks_and_noop(self):
        img, _, _ = disk_sky(256, 512, 0.0, 40.0)
        sun = H.find_sun(img)
        with self.assertRaises(ValueError):
            H.remove_sun(img[:100], sun)
        empty = H.SunInfo(direction=np.array([0, 0, 1.0]), elevation_deg=90.0, azimuth_deg=0.0,
                          irradiance=np.zeros(3), solid_angle=0.0, mask=np.zeros(img.shape[:2], bool))
        np.testing.assert_array_equal(H.remove_sun(img, empty), img)

    def test_remove_sun_is_smooth_on_a_gradient(self):
        h, w = 512, 1024
        el = H._row_elevation(h)
        sky = np.repeat((0.5 + 0.4 * np.sin(el))[:, None, None], w, axis=1).repeat(3, axis=2).astype(np.float32)
        img, d0, inside = disk_sky(h, w, 120.0, 45.0, 2.0, rgb=(1e5, 1e5, 1e5))
        img = np.where(inside[..., None], img, sky)
        sun = H.find_sun(img)
        out = H.remove_sun(img, sun)
        self.assertTrue(np.isfinite(out).all())
        rows = np.flatnonzero(sun.mask.any(1))
        cols = np.flatnonzero(sun.mask.any(0))
        sl = (slice(rows[0] - 3, rows[-1] + 4), slice(cols[0] - 3, cols[-1] + 4))
        ratio = out[sl][..., 1] / sky[sl][..., 1]
        self.assertLess(float(np.abs(ratio - 1).max()), 0.05)                  # follows the gradient: no seam
        self.assertLess(float(np.abs(np.diff(out[sl][..., 1], axis=1)).max()), 0.02)


class Calibration(unittest.TestCase):
    def sky(self, az, el, base):
        img, _, _ = disk_sky(512, 1024, az, el, 2.0, rgb=(8e3, 7e3, 6e3), base=base)
        return img, H.find_sun(img)

    def test_both_equations_hold(self):
        sky, sun = self.sky(30.0, 55.0, 1.0)
        ref, _ = self.sky(-80.0, 40.0, 2.5)
        cal = H.calibrate(sky, sun, ref, 750.0)
        e_sky = float(H.luminance(H.irradiance(H.remove_sun(sky, sun))))
        e_ref = float(H.luminance(H.irradiance(ref)))
        cos_z = math.sin(math.radians(sun.elevation_deg))
        lhs = cal["dome_intensity"] * e_sky + cal["sun_intensity"] * cos_z
        self.assertAlmostEqual(lhs / (750.0 * e_ref), 1.0, delta=1e-6)                                    # (a)
        self.assertAlmostEqual(cal["sun_intensity"] / (cal["dome_intensity"] * float(H.luminance(sun.irradiance))), 1.0, delta=1e-6)   # (b)
        self.assertGreater(cal["dome_intensity"], 0)
        self.assertGreater(cal["sun_intensity"], 0)
        self.assertAlmostEqual(cal["sun_cos_zenith"], float(sun.direction[2]), places=12)
        self.assertAlmostEqual(float(H.luminance(cal["sun_color"])), 1.0, places=6)
        # a different reference intensity scales both intensities linearly
        cal2 = H.calibrate(sky, sun, ref, 1500.0)
        self.assertAlmostEqual(cal2["dome_intensity"] / cal["dome_intensity"], 2.0, places=9)
        self.assertAlmostEqual(cal2["sun_intensity"] / cal["sun_intensity"], 2.0, places=9)

    def test_default_reference_intensity_and_no_sun(self):
        ref, _ = self.sky(0.0, 40.0, 1.0)
        sky = np.full((64, 128, 3), 0.7, np.float32)
        cal = H.calibrate(sky, None, ref)
        e_ref = float(H.luminance(H.irradiance(ref)))
        self.assertAlmostEqual(cal["dome_intensity"] * float(H.luminance(H.irradiance(sky))), 750.0 * e_ref, delta=1e-6 * 750.0 * e_ref)
        self.assertEqual(cal["sun_intensity"], 0.0)

    def test_sun_below_horizon_carries_no_light(self):
        ref, _ = self.sky(0.0, 40.0, 1.0)
        # find_sun only looks above the horizon, so a sun at -20 deg is "no sun": the dome carries it all
        sky, sun = self.sky(30.0, -20.0, 1.0)
        self.assertIsNone(sun)
        cal = H.calibrate(sky, sun, ref)
        self.assertEqual(cal["sun_intensity"], 0.0)
        # the below-horizon branch itself, with a hand-built SunInfo
        sky, up = self.sky(30.0, 55.0, 1.0)
        low = H.SunInfo(direction=H.direction_from_uv(0.5 + 30.0 / 360.0, 0.5 + 20.0 / 180.0), elevation_deg=-20.0,
                        azimuth_deg=30.0, irradiance=up.irradiance, solid_angle=up.solid_angle, mask=up.mask)
        cal = H.calibrate(sky, low, ref)
        self.assertEqual((cal["sun_intensity"], cal["sun_cos_zenith"], cal["sun_illuminance"]), (0.0, 0.0, 0.0))
        self.assertEqual(cal["sun_color"], (1.0, 1.0, 1.0))
        self.assertAlmostEqual(cal["dome_intensity"] * cal["sky_illuminance"] / (750.0 * cal["reference_illuminance"]), 1.0, delta=1e-9)
        # ... and the same sun placed above the horizon carries light again
        self.assertGreater(H.calibrate(sky, low, ref, sun_cos_zenith=0.5)["sun_intensity"], 0.0)

    def test_relocated_sun(self):
        # The DistantLight is placed where the scene wants it (e.g. 55 deg), not where the HDR's sun was (85 deg);
        # the horizontal illuminance must still hit the target.
        sky, sun = self.sky(30.0, 85.0, 1.0)
        ref, _ = self.sky(-80.0, 40.0, 2.5)
        placed = math.sin(math.radians(55.0))
        cal = H.calibrate(sky, sun, ref, 750.0, sun_cos_zenith=placed)
        target = 750.0 * cal["reference_illuminance"]
        self.assertAlmostEqual((cal["dome_intensity"] * cal["sky_illuminance"] + cal["sun_intensity"] * placed) / target, 1.0, delta=1e-9)
        self.assertAlmostEqual(cal["sun_intensity"] / (cal["dome_intensity"] * float(H.luminance(sun.irradiance))), 1.0, delta=1e-9)   # (b)
        self.assertEqual(cal["sun_cos_zenith"], placed)
        # without it the sun is assumed to stay at 85 deg and the scene with the light at 55 deg is off target
        own = H.calibrate(sky, sun, ref, 750.0)
        self.assertAlmostEqual(own["sun_cos_zenith"], float(sun.direction[2]), places=12)
        off = (own["dome_intensity"] * own["sky_illuminance"] + own["sun_intensity"] * placed) / target
        self.assertGreater(abs(off - 1.0), 0.1)
        # giving the HDR sun's own cosine changes nothing
        same = H.calibrate(sky, sun, ref, 750.0, sun_cos_zenith=float(sun.direction[2]))
        self.assertEqual((same["dome_intensity"], same["sun_intensity"]), (own["dome_intensity"], own["sun_intensity"]))

    def test_sun_cos_zenith_edge_cases(self):
        sky, sun = self.sky(30.0, 55.0, 1.0)
        ref, _ = self.sky(0.0, 40.0, 1.0)
        self.assertEqual(H.calibrate(sky, sun, ref, sun_cos_zenith=1.7)["sun_cos_zenith"], 1.0)         # clamped to the zenith
        for c in (0.0, -0.3):                                                                           # at or below the horizon
            cal = H.calibrate(sky, sun, ref, sun_cos_zenith=c)
            self.assertEqual((cal["sun_intensity"], cal["sun_cos_zenith"]), (0.0, 0.0), msg=str(c))
        for c in (float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=str(c)):
                H.calibrate(sky, sun, ref, sun_cos_zenith=c)
        # no sun: nothing to place, the argument is ignored
        flat = np.full((64, 128, 3), 0.7, np.float32)
        self.assertEqual(H.calibrate(flat, None, ref, sun_cos_zenith=0.8), H.calibrate(flat, None, ref))

    def test_sun_with_ratio_preserved(self):
        sky_a, sun_a = self.sky(30.0, 55.0, 1.0)
        ref, _ = self.sky(0.0, 40.0, 1.0)
        base = H.calibrate(sky_a, sun_a, ref)
        # same HDR, twice as bright: calibration undoes it
        sky_b = sky_a * 2
        cal_b = H.calibrate(sky_b, H.find_sun(sky_b), ref)
        self.assertAlmostEqual(cal_b["dome_intensity"] * 2 / base["dome_intensity"], 1.0, places=6)
        self.assertAlmostEqual(cal_b["sun_intensity"] / base["sun_intensity"], 1.0, places=6)


class Downsample(unittest.TestCase):
    def test_box_filter(self):
        img = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        out = H.downsample(img, 2)
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[0, 0], img[:2, :2].reshape(-1, 3).mean(axis=0))
        self.assertEqual(H.downsample(img, 5 - 4).shape, img.shape)
        self.assertFalse(np.shares_memory(H.downsample(img, 1), img))
        self.assertEqual(H.downsample(np.zeros((9, 13, 3), np.float32), 4).shape, (2, 3, 3))      # ragged edge dropped
        self.assertEqual(H.downsample(np.zeros((8, 8), np.float32), 4).shape, (2, 2))
        with self.assertRaises(ValueError):
            H.downsample(img, 0)
        with self.assertRaises(ValueError):
            H.downsample(img, 8)

    def test_preserves_mean_and_illuminance(self):
        rng = np.random.default_rng(8)
        img = rng.uniform(0.5, 1.5, (128, 256, 3)).astype(np.float32)
        small = H.downsample(img, 4)
        self.assertAlmostEqual(float(small.mean()), float(img.mean()), places=4)
        np.testing.assert_allclose(H.irradiance(small), H.irradiance(img), rtol=2e-2)


class RealSkies(unittest.TestCase):
    def test_read_both_4k_files(self):
        for url in (NOON_GRASS, KLOOFENDAL):
            _, img = real_sky(url)
            self.assertEqual(img.shape, (2048, 4096, 3), url)
            self.assertEqual(img.dtype, np.float32)
            self.assertTrue(np.isfinite(img).all())
            self.assertGreaterEqual(float(img.min()), 0.0)
            self.assertTrue((img.max(axis=-1) > 0).all())          # no black pixel; a few single channels are 0
            self.assertGreater(float(img.max()), 1e4)               # the sun is HDR, not clipped to 1
            print(f"\n  {os.path.basename(url)}: min {img.min():.4g} median {np.median(img):.4g} max {img.max():.4g}")

    def test_round_trip_on_a_crop(self):
        _, img = real_sky(NOON_GRASS)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        crop = np.ascontiguousarray(img[:96, 1100:1500])          # holds the sun
        self.assertGreater(float(crop.max()), 1e4)
        for rle in (True, False):
            p = os.path.join(tmp.name, f"crop{int(rle)}.hdr")
            H.write_hdr(p, crop, rle=rle)
            back = H.read_hdr(p)
            err = np.abs(back - crop)
            self.assertTrue((err <= crop.max(axis=-1, keepdims=True) * 2.0 ** -8).all())
        # perturbed (non-RGBE) values
        noisy = (crop * np.random.default_rng(9).uniform(0.9, 1.1, crop.shape)).astype(np.float32)
        p = os.path.join(tmp.name, "noisy.hdr")
        H.write_hdr(p, noisy)
        self.assertTrue((np.abs(H.read_hdr(p) - noisy) <= noisy.max(axis=-1, keepdims=True) * 2.0 ** -8 * 1.001).all())

    def test_noon_grass_sun(self):
        _, img = real_sky(NOON_GRASS)
        sun = H.find_sun(img)
        self.assertIsNotNone(sun)
        sunless = H.remove_sun(img, sun)
        e_sky = float(H.luminance(H.irradiance(sunless)))
        e_sun = float(H.luminance(sun.irradiance))
        e_all = float(H.luminance(H.irradiance(img)))
        print(f"\n  noon_grass sun: direction {np.round(sun.direction, 4)}, elevation {sun.elevation_deg:.2f} deg, "
              f"azimuth {sun.azimuth_deg:.2f} deg, {int(sun.mask.sum())} px, {sun.solid_angle:.4g} sr, "
              f"peak {sun.peak_luminance:.4g} (sky median {sun.sky_luminance:.3g})\n"
              f"  sun irradiance rgb {np.round(sun.irradiance, 3)}, luminance {e_sun:.4g}; horizontal luminance irradiance: "
              f"sunless sky {e_sky:.4g}, whole HDR {e_all:.4g}; sun / sunless sky = {e_sun / e_sky:.2f}")
        self.assertGreater(sun.elevation_deg, 30.0)
        # The HDR's own numbers: sun 4.2 vs sunless sky 1.06 (ratio ~4), not the 5x one might expect of a noon.
        self.assertGreater(e_sun, 3.0 * e_sky)
        self.assertGreater(e_sun, 0.5 * e_all)
        # the direction is that of the brightest pixel
        h, w = img.shape[:2]
        i, j = np.unravel_index(int(np.argmax(H.luminance(img))), (h, w))
        self.assertLess(angle_deg(sun.direction, H.direction_from_uv((j + 0.5) / w, (i + 0.5) / h)), 1.0)
        u, v = H.uv_from_direction(sun.direction)
        self.assertTrue(sun.mask[min(int(v * h), h - 1), min(int(u * w), w - 1)])
        self.check_fill(img, sun, sunless)

    def test_kloofendal_sun(self):
        _, img = real_sky(KLOOFENDAL)
        sun = H.find_sun(img)
        self.assertIsNotNone(sun)
        sunless = H.remove_sun(img, sun)
        e_sky = float(H.luminance(H.irradiance(sunless)))
        e_sun = float(H.luminance(sun.irradiance))
        print(f"\n  kloofendal sun: elevation {sun.elevation_deg:.2f} deg (file name says 43), azimuth {sun.azimuth_deg:.2f} deg, "
              f"{int(sun.mask.sum())} px, sun luminance irradiance {e_sun:.4g}, sunless sky {e_sky:.4g}, ratio {e_sun / e_sky:.2f}")
        self.assertAlmostEqual(sun.elevation_deg, 43.0, delta=1.5)
        self.assertGreater(e_sun, 3.0 * e_sky)
        self.check_fill(img, sun, sunless)

    def check_fill(self, img, sun, sunless):
        """The filled patch is finite, leaves the rest alone and stays within the ring around it."""
        self.assertTrue(np.isfinite(sunless).all())
        np.testing.assert_array_equal(sunless[~sun.mask], img[~sun.mask])
        r0, r1, cols, wrap = H._mask_crop(sun.mask, 8)
        m = sun.mask[r0:r1][:, cols]
        ring = H._dilate_ellipse(m, 3, 3, wrap) & ~m
        before = H.luminance(img[r0:r1][:, cols])
        after = H.luminance(sunless[r0:r1][:, cols])
        # The fill is harmonic, so by the maximum principle it lies between the extremes of the ring (per
        # channel; luminance is a positive mix of channels).  It is not bounded by the ring median: where
        # the ring is lit unevenly, as around a partly cloudy sun, the fill reaches 3x the median.
        self.assertLessEqual(float(after[m].max()), 1.01 * float(before[ring].max()))
        self.assertGreaterEqual(float(after[m].min()), 0.9 * float(before[ring].min()))
        self.assertLess(float(after[m].max()), 1e-3 * float(img.max()))                    # the sun is gone

    def test_calibration_with_the_sun_placed_by_the_scene(self):
        _, noon = real_sky(NOON_GRASS)                       # sun at 85 deg
        _, kl = real_sky(KLOOFENDAL)
        sun = H.find_sun(noon)
        placed = math.sin(math.radians(55.0))                # where the scene puts its DistantLight
        cal = H.calibrate(noon, sun, kl, 750.0, sun_cos_zenith=placed)
        target = 750.0 * cal["reference_illuminance"]
        scene = cal["dome_intensity"] * cal["sky_illuminance"] + cal["sun_intensity"] * placed
        self.assertAlmostEqual(scene / target, 1.0, delta=1e-9)
        own = H.calibrate(noon, sun, kl, 750.0)
        off = (own["dome_intensity"] * own["sky_illuminance"] + own["sun_intensity"] * placed) / target
        print(f"\n  noon_grass, light at 55 deg: dome {cal['dome_intensity']:.1f} sun {cal['sun_intensity']:.1f} "
              f"(on-target); calibrated for its own 85 deg sun it would reach {off:.3f} of the target")
        self.assertLess(off, 0.95)

    def test_calibration_and_cli(self):
        noon_path, noon = real_sky(NOON_GRASS)
        kl_path, kl = real_sky(KLOOFENDAL)
        sun = H.find_sun(noon)
        cal = H.calibrate(noon, sun, kl, 750.0)
        e_sky = float(H.luminance(H.irradiance(H.remove_sun(noon, sun))))
        e_ref = float(H.luminance(H.irradiance(kl)))
        lhs = cal["dome_intensity"] * e_sky + cal["sun_intensity"] * float(sun.direction[2])
        self.assertAlmostEqual(lhs / (750.0 * e_ref), 1.0, delta=1e-6)
        self.assertAlmostEqual(cal["sun_intensity"] / (cal["dome_intensity"] * float(H.luminance(sun.irradiance))), 1.0, delta=1e-6)
        print(f"\n  noon_grass vs kloofendal@750: dome {cal['dome_intensity']:.1f}, sun {cal['sun_intensity']:.1f}")
        # a sky calibrated against itself keeps about the reference brightness
        own = H.calibrate(kl, H.find_sun(kl), kl, 750.0)
        print(f"  kloofendal vs itself @750: dome {own['dome_intensity']:.1f}, sun {own['sun_intensity']:.1f}")
        self.assertAlmostEqual(own["dome_intensity"], 750.0, delta=0.15 * 750.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(H.main(["info", noon_path, "--reference", kl_path]), 0)
        out = buf.getvalue()
        print("  CLI output:\n    " + out.replace("\n", "\n    "))
        self.assertIn("sun: direction", out)
        m = re.search(r"dome_intensity ([\d.e+-]+)\s+sun_intensity ([\d.e+-]+)", out)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), cal["dome_intensity"], delta=1e-4 * cal["dome_intensity"])


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_info(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(H.main(["info", *args, "--cache", os.path.join(self.tmp.name, "cache")]), 0)
        return buf.getvalue()

    def test_unreadable_files_exit_with_a_message_not_a_traceback(self):
        missing = os.path.join(self.tmp.name, "nope.hdr")
        bad = os.path.join(self.tmp.name, "bad.hdr")
        with open(bad, "wb") as f:
            f.write(b"not an hdr")
        for args in ([missing], [bad], [bad, "--reference", missing]):
            with self.assertRaises(SystemExit) as cm:
                with contextlib.redirect_stdout(io.StringIO()):
                    H.main(["info", *args])
            self.assertRegex(str(cm.exception.code), r"^error: cannot read .*(nope|bad)\.hdr", msg=str(args))
        # the file that is actually unreadable is named, also as the reference
        img, _, _ = disk_sky(64, 128, 0.0, 40.0)
        good = os.path.join(self.tmp.name, "good.hdr")
        H.write_hdr(good, img)
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                H.main(["info", good, "--reference", missing])
        self.assertIn("nope.hdr", str(cm.exception.code))

    def test_sun_elevation_places_the_light(self):
        img, _, _ = disk_sky(256, 512, 30.0, 80.0, 2.0, rgb=(8e3, 7e3, 6e3))
        path = os.path.join(self.tmp.name, "sky.hdr")
        H.write_hdr(path, img)
        own = self.run_info(path, "--reference", path)
        placed = self.run_info(path, "--reference", path, "--sun-elevation", "50")

        def number(pattern, out):
            return float(re.search(pattern, out).group(1))

        self.assertAlmostEqual(number(r"cos\(zenith\) ([\d.]+)", own), math.sin(math.radians(80.0)), delta=0.01)
        self.assertAlmostEqual(number(r"cos\(zenith\) ([\d.]+)", placed), math.sin(math.radians(50.0)), delta=1e-4)
        self.assertNotAlmostEqual(number(r"dome_intensity ([\d.e+-]+)", own), number(r"dome_intensity ([\d.e+-]+)", placed), delta=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
