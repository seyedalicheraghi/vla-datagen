"""Checks of the Nucleus asset fetcher: URLs, atomic cached downloads, listings and USD/MDL mirroring.

    python scripts/drone_scene/test/test_nucleus.py

Most tests serve a fake asset tree by patching ``urllib.request.urlopen``.  The ``Live*`` classes hit
NVIDIA's public asset server (about 30 MB, cached in a temp dir for the whole run) and skip themselves
when it is unreachable or ``NUCLEUS_OFFLINE=1`` is set.  The pxr-based checks skip without pxr.
"""

from __future__ import annotations

import contextlib
import errno
import gc
import http.server
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import warnings
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drone_scene import nucleus as N  # noqa: E402

try:
    from pxr import Sdf, Usd, UsdGeom
    HAVE_PXR = True
except ImportError:  # pragma: no cover
    HAVE_PXR = False

DOWNLOAD_BUDGET = 60e6
_LIVE_CACHE = ""


def setUpModule() -> None:
    global _LIVE_CACHE
    _LIVE_CACHE = tempfile.mkdtemp(prefix="nucleus_live_")


def tearDownModule() -> None:
    shutil.rmtree(_LIVE_CACHE, ignore_errors=True)


# ---------------------------------------------------------------------------
# A fake server
# ---------------------------------------------------------------------------

class _Response:
    """What ``urlopen`` returns; can fail after ``fail_after`` bytes or claim a wrong length."""

    def __init__(self, data: bytes, fail_after: int | None = None, claimed_length: int | None = None,
                 on_fail=None, error: BaseException | None = None):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data) if claimed_length is None else claimed_length)}
        self.read_sizes: list[int] = []
        self.closed = False
        self._fail_after, self._on_fail, self._served = fail_after, on_fail, 0
        self._error = error or ConnectionResetError("connection reset mid-stream")

    def read(self, n: int = -1) -> bytes:
        if self._fail_after is not None and self._served >= self._fail_after:
            if self._on_fail:
                self._on_fail()
            raise self._error
        self.read_sizes.append(n)
        chunk = self._buf.read(n)
        self._served += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeServer:
    """Callable standing in for ``urllib.request.urlopen``; serves ``files`` (rel path -> bytes)."""

    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = {N.url(k): v for k, v in (files or {}).items()}
        self.requests: list[str] = []
        self.timeouts: list[float | None] = []
        self.script: list = []         # optional per-request overrides, consumed first (exception or _Response)

    def __call__(self, req, timeout=None):
        target = req.full_url
        self.requests.append(target)
        self.timeouts.append(timeout)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        if target in self.files:
            return _Response(self.files[target])
        raise urllib.error.HTTPError(target, 404, "Not Found", {}, io.BytesIO())

    def patch(self):
        return mock.patch("urllib.request.urlopen", self)


def _files_under(root: str) -> list[str]:
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]


def _dir_bytes(root: str) -> int:
    return sum(os.path.getsize(p) for p in _files_under(root))


class _TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = tempfile.mkdtemp(prefix="nucleus_test_")
        self.addCleanup(shutil.rmtree, self.cache, ignore_errors=True)
        self.addCleanup(setattr, N, "_BACKOFF_S", N._BACKOFF_S)
        N._BACKOFF_S = 0.0           # no real sleeping between retries


# ---------------------------------------------------------------------------
# url()
# ---------------------------------------------------------------------------

class UrlTests(unittest.TestCase):
    def test_joins_under_the_root(self):
        self.assertEqual(N.url("NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl"),
                         N.NUCLEUS_ROOT + "/NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl")
        self.assertTrue(N.NUCLEUS_ROOT.startswith("https://"))

    def test_each_segment_is_percent_encoded(self):
        self.assertEqual(N.url("Props/My Asset.usd"), N.NUCLEUS_ROOT + "/Props/My%20Asset.usd")
        u = N.url("a b/c#d/e?f%g.usd")
        self.assertEqual(u, N.NUCLEUS_ROOT + "/a%20b/c%23d/e%3Ff%25g.usd")
        self.assertEqual(N.url("Mat\u00e9riel/x.usd"), N.NUCLEUS_ROOT + "/Mat%C3%A9riel/x.usd")

    def test_separators_are_not_encoded_and_input_is_normalised(self):
        want = N.NUCLEUS_ROOT + "/a/b/c.usd"
        for rel in ("/a/b/c.usd", "a//b/c.usd", "./a/b/c.usd", "a/x/../b/c.usd", "a\\b\\c.usd"):
            self.assertEqual(N.url(rel), want, rel)

    def test_cannot_leave_the_root(self):
        with self.assertRaises(ValueError):
            N.url("../etc/passwd")
        with self.assertRaises(ValueError):
            N.url("a/../../b")

    def test_local_paths_stay_inside_the_cache(self):
        self.assertEqual(N._local_path("a/b c/d.usd", "/cache"), os.path.join("/cache", "a", "b c", "d.usd"))
        with self.assertRaises(ValueError):
            N._local_path("../../x", "/cache")


# ---------------------------------------------------------------------------
# fetch()
# ---------------------------------------------------------------------------

class FetchTests(_TempDirCase):
    REL = "NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl"

    def test_downloads_into_the_same_layout(self):
        srv = FakeServer({self.REL: b"mdl 1.4;\n"})
        with srv.patch():
            path = N.fetch(self.REL, self.cache)
        self.assertEqual(path, os.path.join(self.cache, *self.REL.split("/")))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"mdl 1.4;\n")
        self.assertEqual(srv.requests, [N.url(self.REL)])

    def test_second_call_is_served_from_the_cache(self):
        srv = FakeServer({self.REL: b"x" * 100})
        with srv.patch():
            first = N.fetch(self.REL, self.cache)
            t = time.perf_counter()
            second = N.fetch(self.REL, self.cache)
            dt = time.perf_counter() - t
        self.assertEqual(first, second)
        self.assertEqual(len(srv.requests), 1)
        self.assertLess(dt, 0.05)

    def test_empty_file_in_the_cache_is_downloaded_again(self):
        local = os.path.join(self.cache, *self.REL.split("/"))
        os.makedirs(os.path.dirname(local))
        open(local, "wb").close()
        srv = FakeServer({self.REL: b"data"})
        with srv.patch():
            N.fetch(self.REL, self.cache)
        self.assertEqual(len(srv.requests), 1)
        self.assertEqual(os.path.getsize(local), 4)

    def test_large_files_are_streamed_in_chunks(self):
        data = os.urandom(300_000)
        responses: list[_Response] = []

        def urlopen(req, timeout=None):
            responses.append(_Response(data))
            return responses[-1]
        with mock.patch("urllib.request.urlopen", urlopen), mock.patch.object(N, "_CHUNK", 64 * 1024):
            path = N.fetch("big.bin", self.cache)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), data)
        self.assertGreaterEqual(len(responses[0].read_sizes), 5)
        self.assertTrue(all(n == 64 * 1024 for n in responses[0].read_sizes))
        self.assertTrue(responses[0].closed)

    def test_transient_errors_are_retried(self):
        srv = FakeServer({self.REL: b"ok"})
        srv.script = [urllib.error.URLError("temporary DNS failure"), TimeoutError("timed out"),
                      urllib.error.HTTPError(N.url(self.REL), 503, "Slow Down", {}, io.BytesIO())]
        lines: list[str] = []
        with srv.patch():
            path = N.fetch(self.REL, self.cache, retries=3, log=lines.append)
        self.assertEqual(len(srv.requests), 4)
        self.assertEqual(os.path.getsize(path), 2)
        self.assertEqual(sum(l.startswith("retry") for l in lines), 3)

    def test_final_failure_is_a_runtime_error_naming_the_url(self):
        srv = FakeServer()
        srv.script = [urllib.error.URLError("unreachable")] * 3
        with srv.patch(), self.assertRaises(RuntimeError) as cm:
            N.fetch(self.REL, self.cache, retries=2)
        self.assertIn(N.url(self.REL), str(cm.exception))
        self.assertEqual(len(srv.requests), 3)
        self.assertNotIsInstance(cm.exception, N.AssetNotFound)
        self.assertEqual(_files_under(self.cache), [])

    def test_missing_file_raises_asset_not_found_without_retrying(self):
        srv = FakeServer()
        with srv.patch(), self.assertRaises(N.AssetNotFound) as cm:
            N.fetch(self.REL, self.cache, retries=5)
        self.assertIsInstance(cm.exception, RuntimeError)
        self.assertIn(N.url(self.REL), str(cm.exception))
        self.assertEqual(len(srv.requests), 1)

    def test_access_denied_is_permanent(self):
        srv = FakeServer()
        srv.script = [urllib.error.HTTPError(N.url(self.REL), 403, "Forbidden", {}, io.BytesIO())]
        with srv.patch(), self.assertRaises(RuntimeError) as cm:
            N.fetch(self.REL, self.cache, retries=5)
        self.assertIn("403", str(cm.exception))
        self.assertEqual(len(srv.requests), 1)

    def test_failing_download_leaves_no_partial_file(self):
        data = os.urandom(300_000)
        seen_during_failure: list[list[str]] = []

        def snapshot():
            seen_during_failure.append([os.path.basename(p) for p in _files_under(self.cache)])
        srv = FakeServer()
        srv.script = [_Response(data, fail_after=128 * 1024, on_fail=snapshot) for _ in range(2)]
        with srv.patch(), mock.patch.object(N, "_CHUNK", 64 * 1024), self.assertRaises(RuntimeError) as cm:
            N.fetch(self.REL, self.cache, retries=1)
        self.assertIn(N.url(self.REL), str(cm.exception))
        # while the transfer was running the bytes were going to a .part file, never to the final name
        self.assertEqual(len(seen_during_failure), 2)
        for names in seen_during_failure:
            self.assertEqual(len(names), 1)
            self.assertTrue(names[0].endswith(".part"), names)
            self.assertNotEqual(names[0], os.path.basename(self.REL))
        self.assertEqual(_files_under(self.cache), [])

    def test_truncated_download_is_rejected_and_retried(self):
        srv = FakeServer({self.REL: b"complete"})
        srv.script = [_Response(b"comp", claimed_length=8)]
        with srv.patch():
            path = N.fetch(self.REL, self.cache, retries=1)
        self.assertEqual(len(srv.requests), 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"complete")
        self.assertEqual([os.path.basename(p) for p in _files_under(self.cache)], [os.path.basename(self.REL)])

    def test_a_truncated_file_is_never_published_even_after_all_retries(self):
        srv = FakeServer()
        srv.script = [_Response(b"half", claimed_length=100) for _ in range(3)]
        with srv.patch(), self.assertRaises(RuntimeError):
            N.fetch(self.REL, self.cache, retries=2)
        self.assertEqual(_files_under(self.cache), [])

    def test_backoff_doubles_and_is_capped(self):
        N._BACKOFF_S = 1.0
        srv = FakeServer()
        srv.script = [urllib.error.URLError("down")] * 8
        delays: list[float] = []
        with srv.patch(), mock.patch.object(N.time, "sleep", delays.append), self.assertRaises(RuntimeError):
            N.fetch(self.REL, self.cache, retries=7)
        self.assertEqual(len(srv.requests), 8)
        self.assertEqual(delays, [1, 2, 4, 8, 16, 30, 30])

    def test_timeout_reaches_urlopen(self):
        srv = FakeServer({self.REL: b"abc", "other.bin": b"def"})
        with srv.patch():
            N.fetch(self.REL, self.cache, timeout=7.5)
            N.fetch("other.bin", self.cache)
        self.assertEqual(srv.timeouts, [7.5, 120])

    def test_empty_response_is_rejected_and_retried(self):
        for with_length in (True, False):
            with self.subTest(content_length=with_length):
                empty = _Response(b"")
                if not with_length:
                    empty.headers = {}           # chunked reply: only the empty check can catch it
                srv = FakeServer({self.REL: b"data"})
                srv.script = [empty]
                with srv.patch():
                    path = N.fetch(self.REL, self.cache, retries=1)
                self.assertEqual(len(srv.requests), 2)
                self.assertEqual(os.path.getsize(path), 4)
                os.remove(path)

    def test_an_empty_response_is_never_published(self):
        srv = FakeServer()
        srv.script = [_Response(b"") for _ in range(2)]
        srv.script[1].headers = {}
        with srv.patch(), self.assertRaises(RuntimeError) as cm:
            N.fetch(self.REL, self.cache, retries=1)
        self.assertIn("empty response", str(cm.exception))
        self.assertEqual(_files_under(self.cache), [])

    def test_gone_is_not_found(self):
        srv = FakeServer()
        srv.script = [urllib.error.HTTPError(N.url(self.REL), 410, "Gone", {}, io.BytesIO())]
        with srv.patch(), self.assertRaises(N.AssetNotFound):
            N.fetch(self.REL, self.cache, retries=5)
        self.assertEqual(len(srv.requests), 1)

    def test_http_error_responses_are_always_closed(self):
        # an HTTPError owns the response socket and is kept as the failure cause, so it must be closed here
        for code, retries, outcome in ((404, 0, N.AssetNotFound), (410, 0, N.AssetNotFound),
                                       (403, 0, RuntimeError), (429, 0, RuntimeError), (503, 1, None)):
            with self.subTest(code=code):
                body = io.BytesIO(b"error page")
                srv = FakeServer({self.REL: b"ok"})
                srv.script = [urllib.error.HTTPError(N.url(self.REL), code, "x", {}, body)]
                with srv.patch():
                    if outcome is None:
                        N.fetch(self.REL, self.cache, retries=retries)
                    else:
                        with self.assertRaises(outcome):
                            N.fetch(self.REL, self.cache, retries=retries)
                self.assertTrue(body.closed)

    @contextlib.contextmanager
    def _cache_fails(self, where: str):
        """Make one local file operation of ``_download`` fail with ENOSPC: open, write, close or replace."""
        no_space = OSError(errno.ENOSPC, "No space left on device")
        real_open = open

        class FullDisk:
            def __init__(self, f):
                self._f = f

            def write(self, data):
                if where == "write":
                    raise no_space
                return self._f.write(data)

            def close(self):
                was_open = not self._f.closed
                self._f.close()
                if where == "close" and was_open:        # flushing is where a full disk often shows up
                    raise no_space

        def opener(path, mode="r", *args, **kwargs):
            if not str(path).endswith(".part"):
                return real_open(path, mode, *args, **kwargs)
            if where == "open":
                raise no_space
            return FullDisk(real_open(path, mode, *args, **kwargs))

        with contextlib.ExitStack() as stack:
            if where == "replace":
                stack.enter_context(mock.patch("os.replace", side_effect=no_space))
            else:
                stack.enter_context(mock.patch("builtins.open", opener))
            yield

    def test_cache_write_failures_are_not_retried(self):
        for where in ("open", "write", "close", "replace"):
            with self.subTest(failing=where):
                srv = FakeServer({self.REL: b"payload"})
                sleeps: list[float] = []
                with srv.patch(), mock.patch.object(N.time, "sleep", sleeps.append), \
                        self.assertRaises(RuntimeError) as cm:
                    with self._cache_fails(where):
                        N.fetch(self.REL, self.cache, retries=3)
                self.assertNotIsInstance(cm.exception, N.AssetNotFound)
                self.assertIn("cannot write " + os.path.join(self.cache, *self.REL.split("/")), str(cm.exception))
                self.assertIn("No space left", str(cm.exception))
                self.assertEqual(len(srv.requests), 1)          # no second download
                self.assertEqual(sleeps, [])                    # and no backoff
                self.assertEqual(_files_under(self.cache), [])  # the .part file is gone too

    def test_interrupt_removes_the_part_file_and_is_not_retried(self):
        data = os.urandom(300_000)
        seen: list[list[str]] = []
        srv = FakeServer()
        srv.script = [_Response(data, fail_after=64 * 1024, error=KeyboardInterrupt(),
                                on_fail=lambda: seen.append([os.path.basename(p) for p in _files_under(self.cache)]))]
        with srv.patch(), mock.patch.object(N, "_CHUNK", 64 * 1024), self.assertRaises(KeyboardInterrupt):
            N.fetch(self.REL, self.cache, retries=3)
        self.assertEqual(len(srv.requests), 1)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0][0].endswith(".part"), seen)     # it was there while the transfer ran
        self.assertEqual(_files_under(self.cache), [])

    def test_concurrent_downloads_of_one_file_use_separate_part_files(self):
        workers = 4
        mid_transfer: list[list[str]] = []
        barrier = threading.Barrier(workers, timeout=10, action=lambda: mid_transfer.append(
            sorted(os.path.basename(p) for p in _files_under(self.cache))))

        class Gated(_Response):
            gated = False

            def read(self, n=-1):
                if not self.gated:
                    self.gated = True
                    barrier.wait()              # every worker has opened its .part file by now
                return super().read(n)

        def urlopen(req, timeout=None):
            return Gated(b"shared payload")
        errors: list[BaseException] = []

        def work():
            try:
                N.fetch(self.REL, self.cache, retries=0)
            except BaseException as e:   # noqa: BLE001
                errors.append(e)
        with mock.patch("urllib.request.urlopen", urlopen):
            threads = [threading.Thread(target=work) for _ in range(workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(len(mid_transfer), 1)
        self.assertEqual(len(mid_transfer[0]), workers)
        self.assertEqual(len(set(mid_transfer[0])), workers)
        self.assertTrue(all(n.endswith(".part") for n in mid_transfer[0]), mid_transfer)
        with open(os.path.join(self.cache, *self.REL.split("/")), "rb") as f:
            self.assertEqual(f.read(), b"shared payload")
        self.assertEqual([os.path.basename(p) for p in _files_under(self.cache)], [os.path.basename(self.REL)])

    def test_log_reports_downloads_only(self):
        srv = FakeServer({self.REL: b"abc"})
        lines: list[str] = []
        with srv.patch():
            N.fetch(self.REL, self.cache, log=lines.append)
            N.fetch(self.REL, self.cache, log=lines.append)
        self.assertEqual(len(lines), 1)
        self.assertIn(self.REL, lines[0])

    def test_local_or_url(self):
        srv = FakeServer({self.REL: b"abc"})
        self.assertEqual(N.local_or_url(self.REL, self.cache), N.url(self.REL))
        self.assertEqual(N.local_or_url(self.REL, None), N.url(self.REL))
        with srv.patch():
            path = N.fetch(self.REL, self.cache)
        self.assertEqual(N.local_or_url(self.REL, self.cache), path)
        self.assertEqual(N.local_or_url("other/file.usd", self.cache), N.url("other/file.usd"))


# ---------------------------------------------------------------------------
# list_dir()
# ---------------------------------------------------------------------------

_S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def _listing(prefix: str, dirs=(), keys=(), token: str | None = None) -> bytes:
    body = "".join(f"<CommonPrefixes><Prefix>{prefix}{d}/</Prefix></CommonPrefixes>" for d in dirs)
    body += "".join(f"<Contents><Key>{prefix}{k}</Key><Size>1</Size></Contents>" for k in keys)
    tail = (f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>"
            if token else "<IsTruncated>false</IsTruncated>")
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{_S3_NS}">'
            f"<Prefix>{prefix}</Prefix>{body}{tail}</ListBucketResult>").encode()


class ListDirTests(_TempDirCase):
    PREFIX = "Assets/Isaac/5.1/NVIDIA/Materials/"

    def _server(self, pages: list[bytes]) -> FakeServer:
        srv = FakeServer()
        srv.script = [_Response(p) for p in pages]
        return srv

    def test_follows_continuation_tokens(self):
        token = "1ab+/c=="
        srv = self._server([
            _listing(self.PREFIX, dirs=["Base", ".thumbs"], keys=["", "b.mdl"], token=token),
            _listing(self.PREFIX, dirs=["Architecture"], keys=["a.mdl"]),
        ])
        with srv.patch():
            subdirs, files = N.list_dir("NVIDIA/Materials/")
        self.assertEqual(subdirs, ["Architecture", "Base"])          # sorted, .thumbs dropped
        self.assertEqual(files, ["a.mdl", "b.mdl"])                  # the directory placeholder key is dropped
        self.assertEqual(len(srv.requests), 2)
        first, second = (urllib.parse.parse_qs(urllib.parse.urlsplit(u).query) for u in srv.requests)
        self.assertEqual(first["prefix"], [self.PREFIX])
        self.assertEqual(first["delimiter"], ["/"])
        self.assertEqual(first["list-type"], ["2"])
        self.assertNotIn("continuation-token", first)
        self.assertEqual(second["continuation-token"], [token])
        self.assertIn("continuation-token=1ab%2B%2Fc%3D%3D", srv.requests[1])   # '+' would mean a space to S3

    def test_directory_spelling_does_not_matter(self):
        prefixes = []
        for spelling in ("NVIDIA/Materials", "NVIDIA/Materials/", "/NVIDIA/Materials/"):
            srv = self._server([_listing(self.PREFIX)])
            with srv.patch():
                self.assertEqual(N.list_dir(spelling), ([], []))
            prefixes.append(urllib.parse.parse_qs(urllib.parse.urlsplit(srv.requests[0]).query)["prefix"])
        self.assertEqual(prefixes, [[self.PREFIX]] * 3)

    def test_http_errors_are_runtime_errors(self):
        srv = FakeServer()
        srv.script = [urllib.error.URLError("down")] * 2
        with srv.patch(), self.assertRaises(RuntimeError):
            N.list_dir("NVIDIA/", retries=1)

    def test_a_repeating_continuation_token_ends_the_listing(self):
        calls: list[str] = []

        def urlopen(req, timeout=None):
            calls.append(req.full_url)
            if len(calls) > 10:
                raise AssertionError("list_dir keeps paging")
            return _Response(_listing(self.PREFIX, keys=["a.mdl"], token="same"))
        with mock.patch("urllib.request.urlopen", urlopen):
            _, files = N.list_dir("NVIDIA/Materials/", retries=0)
        self.assertEqual(len(calls), 2)                  # the second reply repeats the token: stop
        self.assertIn("a.mdl", files)

    def test_timeout_reaches_urlopen(self):
        srv = self._server([_listing(self.PREFIX, keys=["a.mdl"], token="t"), _listing(self.PREFIX)])
        with srv.patch():
            N.list_dir("NVIDIA/Materials/", timeout=9)
        self.assertEqual(srv.timeouts, [9, 9])


# ---------------------------------------------------------------------------
# MDL / reference parsing (no pxr needed)
# ---------------------------------------------------------------------------

class ParsingTests(unittest.TestCase):
    MDL = '''mdl 1.4;
import ::OmniPBR::OmniPBR;
import ::tex::gamma_mode;
import ..::Templates::OmniShared::*;
import GlassUtils::get_color, GlassUtils::apply_roughness_influence;
import .::Local::thing;
using ::OmniPBR import OmniPBR;
using .::Sibling import *;
// import ::fake::commented_out;
/* import ::fake::block_comment;
   texture_2d("./block_comment.png") */
export material M(*)
[[ ::anno::description("see https://example.com/a.png; import ::fake::in_string;") ]]
= ::OmniPBR::OmniPBR(
    diffuse_texture: texture_2d("./M/M_BaseColor.png" /* tag 1 */, ::tex::gamma_srgb),
    ORM_texture: texture_2d("M/M_ORM.png", ::tex::gamma_linear), // texture_2d("./line_comment.png")
    reflectionroughness_texture: texture_2d(),
    empty: texture_2d(""),
    cube: texture_cube("../env/sky.exr"),
    profile: light_profile ( "lamps/spot.ies" ),
    again: texture_2d("./M/M_BaseColor.png"));
'''

    def test_resources(self):
        self.assertEqual(N._mdl_resources(self.MDL),
                         ["./M/M_BaseColor.png", "M/M_ORM.png", "../env/sky.exr", "lamps/spot.ies"])

    def test_modules(self):
        self.assertEqual(N._mdl_modules(self.MDL),
                         ["::OmniPBR", "::tex", "..::Templates::OmniShared", "GlassUtils", ".::Local",
                          ".::Sibling"])

    def test_module_files(self):
        here = "NVIDIA/Materials/OmniSurface/Fabric/Velvet.mdl"
        rel = lambda m: N._module_rel(here, m)       # noqa: E731
        self.assertEqual(rel("..::Templates::OmniShared"), "NVIDIA/Materials/OmniSurface/Templates/OmniShared.mdl")
        self.assertEqual(rel(".::Local"), "NVIDIA/Materials/OmniSurface/Fabric/Local.mdl")
        self.assertEqual(rel("GlassUtils"), "NVIDIA/Materials/OmniSurface/Fabric/GlassUtils.mdl")
        self.assertEqual(rel("..::..::Shared::Deep"), "NVIDIA/Materials/Shared/Deep.mdl")
        for core in ("::OmniPBR", "::tex", "::base", "::anno", "::state", "::math", "::df", "::nvidia::core_definitions",
                     "tex", "::OmniSurface"):
            self.assertIsNone(rel(core), core)
        self.assertIsNone(N._module_rel("a.mdl", "..::up"))          # would leave the root
        self.assertEqual(N._module_rel("a.mdl", ".::up"), "up.mdl")

    def test_reference_resolution(self):
        here = "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
        ok = lambda ref: N._dep_rel(here, ref)[0]       # noqa: E731
        self.assertEqual(ok("configuration/schema.usd"), "Isaac/Robots/Bitcraze/Crazyflie/configuration/schema.usd")
        self.assertEqual(ok("./tex/a.png"), "Isaac/Robots/Bitcraze/Crazyflie/tex/a.png")
        self.assertEqual(ok("../Other/b.usd"), "Isaac/Robots/Bitcraze/Other/b.usd")
        self.assertEqual(ok("/NVIDIA/Materials/Base/x.mdl"), "NVIDIA/Materials/Base/x.mdl")
        self.assertEqual(ok(N.NUCLEUS_ROOT + "/Isaac/My%20Prop/p.usd"), "Isaac/My Prop/p.usd")
        self.assertEqual(ok("  ./padded.png "), "Isaac/Robots/Bitcraze/Crazyflie/padded.png")

    def test_unmirrorable_references_say_why(self):
        here = "A/b.usd"
        cases = {
            "OmniPBR.mdl": "Kit", "omniverse://localhost/NVIDIA/x.usd": "external", "https://example.com/x.png": "external",
            "anon:0x1:layer.usda": "anonymous", "C:\\textures\\a.png": "filesystem", "tiles/a.<UDIM>.png": "templated",
            "${HOME}/a.png": "templated", "../../up.png": "root", "": "empty",
        }
        for ref, word in cases.items():
            dep, reason = N._dep_rel(here, ref)
            self.assertIsNone(dep, ref)
            self.assertIn(word, reason, ref)
        self.assertEqual(N._dep_rel(here, "./OmniPBR.mdl")[0], "A/OmniPBR.mdl")     # a real sibling is kept

    KIT_ONLY = ("OmniPBR", "OmniPBR_ClearCoat", "OmniPBR_ClearCoat_Opacity", "OmniUe4Base", "OmniUe4Function",
                "OmniUe4Translucent", "OmniRL_PBR")       # none of these has a file anywhere on the server

    def test_kit_materials_are_skipped_as_files_and_as_modules(self):
        here = "Isaac/Environments/Simple_Room/Materials/MI_Table.mdl"
        for name in self.KIT_ONLY:
            dep, reason = N._dep_rel(here, f"{name}.mdl")
            self.assertIsNone(dep, name)
            self.assertIn("Kit", reason)
            for module in (name, f"::{name}"):
                self.assertIsNone(N._module_rel(here, module), module)
            self.assertEqual(N._dep_rel(here, f"./{name}.mdl")[0], f"Isaac/Environments/Simple_Room/Materials/{name}.mdl")
        # the file-name set is derived from the module set, so the two cannot drift apart again
        self.assertTrue(N._KIT_MATERIALS <= N._KIT_MODULES)
        self.assertEqual(N._KIT_MDL, {f"{m}.mdl" for m in N._KIT_MATERIALS})
        self.assertEqual(N._mdl_modules("using OmniUe4Function import *;\nusing OmniUe4Base import *;"),
                         ["OmniUe4Function", "OmniUe4Base"])


class MirrorMdlTests(_TempDirCase):
    """``mirror_usd`` on an MDL root needs no pxr."""

    def test_kit_materials_are_never_requested(self):
        mdl = "Isaac/Environments/Simple_Room/Materials/MI_Table.mdl"
        text = (b'mdl 1.4;\nimport math::*;\nimport tex::*;\nusing OmniUe4Function import *;\n'
                b'using OmniUe4Base import *;\nimport ::OmniPBR_ClearCoat::OmniPBR_ClearCoat;\n'
                b'export material T(*) = OmniUe4Base(base: texture_2d("T/table.png"));\n')
        srv = FakeServer({mdl: text, "Isaac/Environments/Simple_Room/Materials/T/table.png": b"png"})
        lines: list[str] = []
        with srv.patch():
            N.mirror_usd(mdl, self.cache, log=lines.append)
        self.assertEqual(sorted(srv.requests),
                         sorted([N.url(mdl), N.url("Isaac/Environments/Simple_Room/Materials/T/table.png")]))
        self.assertFalse(any("not on server" in l for l in lines), lines)


# ---------------------------------------------------------------------------
# mirror_usd(), against a fake tree
# ---------------------------------------------------------------------------

SCENE_USD = f'''#usda 1.0
(
    defaultPrim = "Scene"
    subLayers = [@./layers/base.usd@]
)
def Xform "Scene"
{{
    def Xform "Prop" (
        prepend references = @../Shared/prop.usd@</Prop>
    )
    {{
    }}
    def Xform "Heavy" (
        prepend payload = @./heavy.usd@</Heavy>
    )
    {{
    }}
    def Material "Mat"
    {{
        def Shader "Shader"
        {{
            uniform asset info:mdl:sourceAsset = @./mats/Paint.mdl@
            uniform token info:mdl:sourceAsset:subIdentifier = "Paint"
            asset inputs:albedo = @./tex/ok.png@
            asset inputs:missing = @./tex/missing.png@
            asset inputs:rootabs = @/Library/abs.png@
            asset inputs:url = @{N.NUCLEUS_ROOT}/Library/with%20space/u.png@
            asset inputs:kit = @OmniPBR.mdl@
            asset inputs:ext = @omniverse://localhost/NVIDIA/elsewhere.png@
            asset[] inputs:many = [@./tex/many_a.png@, @./tex/many_b.png@]
        }}
    }}
}}
'''
PAINT_MDL = b'''mdl 1.4;
import ::OmniPBR::OmniPBR;
import ::tex::*;
import ..::Templates::Shared::*;
// texture_2d("./paint/commented_out.png")
export material Paint(*) = ::OmniPBR::OmniPBR(
    diffuse_texture: texture_2d("./paint/albedo.png" /* tag */, ::tex::gamma_srgb),
    ORM_texture: texture_2d("./paint/orm.png", ::tex::gamma_linear));
'''
TREE = {
    "Pkg/scene.usd": SCENE_USD.encode(),
    "Pkg/layers/base.usd": b'#usda 1.0\n(\n    subLayers = [@../tex/layer_tex.usd@]\n)\n',
    "Pkg/tex/layer_tex.usd": b'#usda 1.0\ndef Shader "S"\n{\n    asset inputs:t = @./deep.png@\n}\n',
    "Pkg/tex/deep.png": b"png-deep",
    "Shared/prop.usd": b'#usda 1.0\ndef Xform "Prop"\n{\n    asset a:tex = @textures/prop.png@\n}\n',
    "Shared/textures/prop.png": b"png-prop",
    "Pkg/heavy.usd": b'#usda 1.0\n(\n    subLayers = [@./scene.usd@]\n)\ndef Xform "Heavy"\n{\n}\n',   # cycle
    "Pkg/mats/Paint.mdl": PAINT_MDL,
    "Pkg/mats/paint/albedo.png": b"png-albedo",
    "Pkg/mats/paint/orm.png": b"png-orm",
    "Pkg/Templates/Shared.mdl": b'mdl 1.4;\nexport material S(*) = ::OmniPBR::OmniPBR(n: texture_2d("noise.png"));\n',
    "Pkg/Templates/noise.png": b"png-noise",
    "Pkg/tex/ok.png": b"png-ok",
    "Pkg/tex/many_a.png": b"png-a",
    "Pkg/tex/many_b.png": b"png-b",
    "Library/abs.png": b"png-abs",
    "Library/with space/u.png": b"png-url",
    "Pkg/commented_out.png": b"must not be requested",
}


@unittest.skipUnless(HAVE_PXR, "pxr not available")
class MirrorOfflineTests(_TempDirCase):
    def mirror(self, srv: FakeServer, rel: str = "Pkg/scene.usd", **kw):
        lines: list[str] = []
        with srv.patch():
            root = N.mirror_usd(rel, self.cache, log=lines.append, **kw)
        return root, lines

    def exists(self, rel: str) -> bool:
        return os.path.isfile(os.path.join(self.cache, *rel.split("/")))

    def test_follows_every_kind_of_reference(self):
        srv = FakeServer(TREE)
        root, lines = self.mirror(srv)
        self.assertEqual(root, os.path.join(self.cache, "Pkg", "scene.usd"))
        for rel in ("Pkg/layers/base.usd",            # sublayer
                    "Pkg/tex/layer_tex.usd",          # sublayer of a sublayer, resolved against base.usd's directory
                    "Pkg/tex/deep.png",               # asset in that nested layer
                    "Shared/prop.usd",                # reference through ../
                    "Shared/textures/prop.png",       # asset inside the referenced layer
                    "Pkg/heavy.usd",                  # payload (and a reference cycle back to scene.usd)
                    "Pkg/mats/Paint.mdl",             # MDL source asset
                    "Pkg/mats/paint/albedo.png",      # texture named in the MDL
                    "Pkg/mats/paint/orm.png",
                    "Pkg/Templates/Shared.mdl",       # relative MDL import
                    "Pkg/Templates/noise.png",        # texture of the imported module
                    "Pkg/tex/ok.png", "Pkg/tex/many_a.png", "Pkg/tex/many_b.png",   # asset and asset[] attributes
                    "Library/abs.png",                # root-absolute path
                    "Library/with space/u.png"):      # full URL, percent-decoded into the layout
            self.assertTrue(self.exists(rel), rel)
        requested = [urllib.parse.unquote(u[len(N.NUCLEUS_ROOT) + 1:]) for u in srv.requests]
        self.assertNotIn("Pkg/commented_out.png", requested)
        self.assertNotIn("Pkg/mats/paint/commented_out.png", requested)      # inside an MDL comment
        self.assertNotIn("Pkg/OmniPBR.mdl", requested)                       # Kit library
        self.assertFalse(any("elsewhere" in r for r in requested))           # foreign scheme
        self.assertEqual(len(requested), len(set(requested)))                # nothing fetched twice

    def test_missing_optional_dependency_is_skipped_with_a_log_line(self):
        _, lines = self.mirror(FakeServer(TREE))
        self.assertFalse(self.exists("Pkg/tex/missing.png"))
        self.assertTrue(any("missing.png" in l and "skipped" in l for l in lines), lines)
        self.assertTrue(any("OmniPBR.mdl" in l and "Kit" in l for l in lines), lines)
        self.assertTrue(lines[-1].startswith("mirrored Pkg/scene.usd"), lines[-1])

    def test_local_copy_opens_with_pxr_and_resolves_locally(self):
        root, _ = self.mirror(FakeServer(TREE))
        stage = Usd.Stage.Open(root)
        names = {os.path.basename(l.realPath) for l in stage.GetLayerStack() if l.realPath}
        self.assertEqual(names, {"scene.usd", "base.usd", "layer_tex.usd"})
        self.assertTrue(stage.GetPrimAtPath("/Scene/Prop").GetAttribute("a:tex").Get().resolvedPath.endswith("prop.png"))
        shader = stage.GetPrimAtPath("/Scene/Mat/Shader")
        for attr in ("info:mdl:sourceAsset", "inputs:albedo"):
            resolved = shader.GetAttribute(attr).Get().resolvedPath
            self.assertTrue(resolved and os.path.isfile(resolved), attr)
            self.assertTrue(os.path.realpath(resolved).startswith(os.path.realpath(self.cache)))
        self.assertTrue(stage.GetPrimAtPath("/S").GetAttribute("inputs:t").Get().resolvedPath.endswith("deep.png"))

    def test_second_run_only_asks_again_for_what_the_server_lacked(self):
        srv = FakeServer(TREE)
        self.mirror(srv)
        srv.requests.clear()
        self.mirror(srv)
        self.assertEqual(sorted(srv.requests), sorted([N.url("Pkg/tex/missing.png")]))

    def test_root_must_exist(self):
        with self.assertRaises(N.AssetNotFound):
            self.mirror(FakeServer(TREE), "Pkg/nope.usd")
        self.assertEqual(_files_under(self.cache), [])

    def test_failure_other_than_not_found_is_not_swallowed(self):
        srv = FakeServer(TREE)
        broken = N.url("Pkg/tex/ok.png")

        def urlopen(req, timeout=None):
            if req.full_url == broken:
                raise urllib.error.URLError("network down")
            return srv(req, timeout)
        with mock.patch("urllib.request.urlopen", urlopen), self.assertRaises(RuntimeError) as cm:
            N.mirror_usd("Pkg/scene.usd", self.cache)
        self.assertIn(broken, str(cm.exception))
        self.assertNotIsInstance(cm.exception, N.AssetNotFound)

    def test_max_files_guard(self):
        with self.assertRaises(RuntimeError) as cm:
            self.mirror(FakeServer(TREE), max_files=5)
        self.assertIn("max_files", str(cm.exception))

    def test_workers_do_not_change_the_result(self):
        a = tempfile.mkdtemp(prefix="nucleus_w1_")
        self.addCleanup(shutil.rmtree, a, ignore_errors=True)
        with FakeServer(TREE).patch():
            N.mirror_usd("Pkg/scene.usd", a, workers=1)
            N.mirror_usd("Pkg/scene.usd", self.cache, workers=8)
        rel = lambda root: sorted(os.path.relpath(p, root) for p in _files_under(root))     # noqa: E731
        self.assertEqual(rel(a), rel(self.cache))

    def test_mirroring_a_texture_just_fetches_it(self):
        srv = FakeServer(TREE)
        root, _ = self.mirror(srv, "Pkg/tex/ok.png")
        with open(root, "rb") as f:
            self.assertEqual(f.read(), b"png-ok")
        self.assertEqual(len(srv.requests), 1)


# ---------------------------------------------------------------------------
# A real socket (loopback), for what mocks cannot show
# ---------------------------------------------------------------------------

class _FlakyHandler(http.server.BaseHTTPRequestHandler):
    """Answers 429 to the first request and an 11-byte body to every later one."""

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        self.server.hits += 1
        body = b"slow down" if self.server.hits == 1 else b"hello world"
        self.send_response(429 if self.server.hits == 1 else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LoopbackTests(_TempDirCase):
    def serve(self) -> http.server.HTTPServer:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
        server.hits = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for cleanup in (thread.join, server.server_close, server.shutdown):     # run in reverse order
            self.addCleanup(cleanup)
        return server

    def test_retry_after_an_http_error_leaks_no_socket(self):
        server = self.serve()
        root = f"http://127.0.0.1:{server.server_port}/root"
        with mock.patch.object(N, "NUCLEUS_ROOT", root), \
                mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = N.fetch("a.bin", self.cache, retries=2)
            gc.collect()                       # the unclosed HTTPError socket is reported when it is collected
        self.assertEqual(server.hits, 2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"hello world")
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, ResourceWarning)], [])


# ---------------------------------------------------------------------------
# Against NVIDIA's server
# ---------------------------------------------------------------------------

class _LiveCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("NUCLEUS_OFFLINE"):
            raise unittest.SkipTest("NUCLEUS_OFFLINE is set")
        try:
            N.list_dir("NVIDIA/Materials/Base/Masonry/", timeout=20, retries=1)
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"asset server unreachable: {e}")


class LiveFetchTests(_LiveCase):
    BRICK = "NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl"

    def test_list_dir(self):
        subdirs, files = N.list_dir("NVIDIA/Materials/Base/Masonry/")
        self.assertIn("Brick_Pavers.mdl", files)
        self.assertIn("Brick_Pavers", subdirs)               # the texture folder next to it
        self.assertNotIn(".thumbs", subdirs)
        self.assertTrue(all("/" not in n for n in subdirs + files))
        base_dirs, _ = N.list_dir("NVIDIA/Materials/Base")
        self.assertIn("Masonry", base_dirs)
        self.assertNotIn(".thumbs", base_dirs)

    def test_fetch_is_cached_on_the_second_call(self):
        path = N.fetch(self.BRICK, _LIVE_CACHE)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith("mdl "))
        self.assertIn("Brick_Pavers_BaseColor.png", text)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used for a cached file")):
            t = time.perf_counter()
            again = N.fetch(self.BRICK, _LIVE_CACHE)
            self.assertLess(time.perf_counter() - t, 0.05)
        self.assertEqual(again, path)
        self.assertEqual(N.local_or_url(self.BRICK, _LIVE_CACHE), path)
        self.assertEqual(os.listdir(os.path.dirname(path)).count(os.path.basename(path)), 1)
        self.assertFalse([p for p in _files_under(_LIVE_CACHE) if p.endswith(".part")])

    def test_missing_file(self):
        with self.assertRaises(N.AssetNotFound):
            N.fetch("NVIDIA/Materials/Base/Masonry/No_Such_Material.mdl", _LIVE_CACHE)


@unittest.skipUnless(HAVE_PXR, "pxr not available")
class LiveMirrorTests(_LiveCase):
    GRASS = "NVIDIA/Assets/Vegetation/Shrub/Grass_Short_A.usd"
    CF2X = "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"

    @classmethod
    def tearDownClass(cls) -> None:
        used = _dir_bytes(_LIVE_CACHE)
        print(f"\n  live downloads: {used / 1e6:.1f} MB", file=sys.stderr)
        assert used < DOWNLOAD_BUDGET, f"tests downloaded {used / 1e6:.1f} MB (budget {DOWNLOAD_BUDGET / 1e6:.0f} MB)"

    @staticmethod
    def asset_paths(stage) -> list:
        found = []
        for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot()):
            for attr in prim.GetAttributes():
                if attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                    value = attr.Get()
                    if value:
                        found += [value] if attr.GetTypeName() == Sdf.ValueTypeNames.Asset else list(value)
        return found

    def test_grass_resolves_entirely_from_the_cache(self):
        lines: list[str] = []
        root = N.mirror_usd(self.GRASS, _LIVE_CACHE, log=lines.append)
        self.assertEqual(root, os.path.join(_LIVE_CACHE, *self.GRASS.split("/")))

        paths = self.asset_paths(Usd.Stage.Open(root))
        names = sorted(os.path.basename(p.path) for p in paths)
        self.assertEqual(names, ["lawngrass_a_basecolor.png", "lawngrass_a_mat.mdl", "lawngrass_a_normal.png",
                                 "lawngrass_a_roughness.png"])
        for p in paths:
            self.assertTrue(p.resolvedPath, f"{p.path} does not resolve")
            self.assertTrue(os.path.isfile(p.resolvedPath), p.resolvedPath)
            self.assertGreater(os.path.getsize(p.resolvedPath), 0)
            self.assertTrue(os.path.realpath(p.resolvedPath).startswith(os.path.realpath(_LIVE_CACHE)))

        # the textures the MDL itself names are the same files
        mdl = next(p.resolvedPath for p in paths if p.path.endswith(".mdl"))
        with open(mdl, encoding="utf-8") as f:
            for tex in N._mdl_resources(f.read()):
                self.assertTrue(os.path.isfile(os.path.normpath(os.path.join(os.path.dirname(mdl), tex))), tex)

        # everything is local now: a second mirror must not touch the network at all
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(N.mirror_usd(self.GRASS, _LIVE_CACHE), root)

    def test_crazyflie_sublayer_resolves(self):
        lines: list[str] = []
        root = N.mirror_usd(self.CF2X, _LIVE_CACHE, log=lines.append)
        layer = Sdf.Layer.FindOrOpen(root)
        self.assertEqual(list(layer.subLayerPaths), ["configuration/cf2x_robot_schema.usd"])
        sub = os.path.join(os.path.dirname(root), "configuration", "cf2x_robot_schema.usd")
        self.assertTrue(os.path.isfile(sub))

        stage = Usd.Stage.Open(root)
        self.assertIn(os.path.realpath(sub), [os.path.realpath(l.realPath) for l in stage.GetLayerStack() if l.realPath])
        self.assertEqual(sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)), 10)
        # OmniPBR.mdl is Kit's own library: never requested, and said so
        self.assertTrue(any("OmniPBR.mdl" in l and "Kit" in l for l in lines), lines)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(root), "OmniPBR.mdl")))

    def test_mdl_relative_imports_are_followed(self):
        rel = "NVIDIA/Materials/OmniSurface/Fabric/OmniSurface_Velvet.mdl"
        root = N.mirror_usd(rel, _LIVE_CACHE)
        templates = os.path.join(_LIVE_CACHE, "NVIDIA", "Materials", "OmniSurface", "Templates")
        self.assertTrue(os.path.isfile(root))
        self.assertTrue(os.path.isfile(os.path.join(templates, "OmniShared.mdl")))
        self.assertTrue(os.path.isfile(os.path.join(templates, "OmniSurfaceUber.mdl")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
