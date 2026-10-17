"""Fetch and cache NVIDIA Nucleus assets over HTTPS (standard library only; pxr only inside ``mirror_usd``).

NVIDIA publishes the Isaac Sim asset tree on a public S3 bucket.  Paths below are *relative to*
``NUCLEUS_ROOT`` and are the same ones Isaac uses, e.g. ``Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd`` or
``NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl``::

    from drone_scene import nucleus
    path = nucleus.fetch("NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl", cache_dir)
    root = nucleus.mirror_usd("NVIDIA/Assets/Vegetation/Shrub/Grass_Short_A.usd", cache_dir, log=print)
    subdirs, files = nucleus.list_dir("NVIDIA/Materials/Base/Masonry/")

``mirror_usd`` keeps the server's directory layout under ``cache_dir`` so every relative reference
inside the USD and MDL files resolves to the local copy, offline, with plain ``Usd.Stage.Open``.
Downloads are written to a ``.part`` file and renamed, so an interrupted run never leaves a truncated
asset that a later run would trust.  Only numpy and Pillow are needed elsewhere in this package, so
this module runs on a bare Isaac Sim Python too.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import os
import posixpath
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from typing import IO, Any, Callable, Iterator

__all__ = ["NUCLEUS_ROOT", "AssetNotFound", "url", "fetch", "list_dir", "mirror_usd", "local_or_url"]

NUCLEUS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

_CHUNK = 1 << 20
_BACKOFF_S = 1.0                     # first retry delay; doubles each attempt
_USER_AGENT = "vla-datagen-drone-scene/1.0"
_USD_EXT = (".usd", ".usda", ".usdc")
_PERMANENT_HTTP = (400, 401, 403)    # retrying cannot fix these (404/410 raise AssetNotFound)

# Materials that Kit resolves from its own library: none of these is anywhere on the asset server.
_KIT_MATERIALS = frozenset({
    "OmniPBR", "OmniPBR_Opacity", "OmniPBR_ClearCoat", "OmniPBR_ClearCoat_Opacity", "OmniGlass",
    "OmniGlass_Opacity", "OmniSurface", "OmniSurfaceBase", "OmniSurfaceLite", "OmniSurfaceBlend", "OmniHair",
    "AperturePBR_Opacity", "AperturePBR_Translucent", "OmniUe4Base", "OmniUe4Function", "OmniUe4Translucent",
    "OmniRL_PBR",
})
# As bare file names (``info:mdl:sourceAsset = @OmniPBR.mdl@``) ...
_KIT_MDL = frozenset(f"{name}.mdl" for name in _KIT_MATERIALS)
# ... and as top-level MDL modules (``import ::tex::*``), which also include the MDL SDK's own packages.
_KIT_MODULES = _KIT_MATERIALS | frozenset({
    "base", "tex", "anno", "state", "math", "df", "std", "limits", "nvidia", "scene", "debug", "noise",
    "geometry", "builtin",
})


class AssetNotFound(RuntimeError):
    """The server has no such file (HTTP 404/410); ``mirror_usd`` treats this as a skippable dependency."""


class _LocalIOError(RuntimeError):
    """The cache could not be written (disk full, permissions): the network is fine, so never retried."""


def _say(log: Callable[[str], None] | None, msg: str) -> None:
    if log is not None:
        log(msg)


# ---------------------------------------------------------------------------
# Paths and URLs
# ---------------------------------------------------------------------------

def _segments(rel: str) -> list[str]:
    """Clean path segments of a root-relative path; ``.``/``..`` are resolved and may not leave the root."""
    parts: list[str] = []
    for seg in rel.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise ValueError(f"path leaves the Nucleus root: {rel!r}")
            parts.pop()
        else:
            parts.append(seg)
    return parts


def _local_path(rel: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, *_segments(rel))


def _is_cached(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def url(rel: str) -> str:
    """Absolute URL of ``rel`` (relative to ``NUCLEUS_ROOT``); every segment is percent-encoded.

    ``rel`` is the literal path, so ``"Props/My Asset.usd"`` becomes ``.../Props/My%20Asset.usd``.
    """
    return "/".join([NUCLEUS_ROOT.rstrip("/"), *(urllib.parse.quote(s, safe="") for s in _segments(rel))])


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------

def _content_length(resp: Any) -> int | None:
    headers = getattr(resp, "headers", None)
    value = headers.get("Content-Length") if hasattr(headers, "get") else None
    return int(value) if isinstance(value, (str, int)) and str(value).isdigit() else None


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@contextlib.contextmanager
def _local_io(path: str) -> Iterator[None]:
    """Re-raise a file error of the cache as ``_LocalIOError``; network errors (also ``OSError``) pass through."""
    try:
        yield
    except OSError as e:
        raise _LocalIOError(f"cannot write {path}: {e}") from e


def _download(target: str, local: str, timeout: float) -> int:
    """Stream ``target`` into ``local`` via a sibling ``.part`` file; returns the byte count."""
    part = f"{local}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"    # unique: threads/processes may race
    resp = urllib.request.urlopen(urllib.request.Request(target, headers={"User-Agent": _USER_AGENT}),
                                  timeout=timeout)
    out: IO[bytes] | None = None
    try:
        expected = _content_length(resp)
        size = 0
        try:
            with _local_io(local):
                out = open(part, "wb")
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                with _local_io(local):
                    out.write(chunk)
                size += len(chunk)
            with _local_io(local):
                out.close()                  # flushes, so a full disk can surface here
            # http.client does not raise when the peer closes early, so compare the length ourselves.
            if expected is not None and size != expected:
                raise OSError(f"truncated download: got {size} of {expected} bytes")
            if size == 0:
                raise OSError("empty response")
            with _local_io(local):
                os.replace(part, local)
        except BaseException:
            if out is not None:
                try:
                    out.close()
                except OSError:
                    pass
            _discard(part)
            raise
        return size
    finally:
        close = getattr(resp, "close", None)
        if close is not None:
            close()


def _retrying(target: str, attempt: Callable[[], Any], retries: int, log: Callable[[str], None] | None) -> Any:
    """Run ``attempt()`` up to ``retries + 1`` times with exponential backoff; map failures to RuntimeError."""
    last: BaseException | None = None
    for n in range(max(0, retries) + 1):
        if n:
            delay = min(_BACKOFF_S * 2 ** (n - 1), 30.0)
            _say(log, f"retry {n}/{retries} in {delay:g}s: {target} ({last!r})")
            time.sleep(delay)
        try:
            return attempt()
        except _LocalIOError:
            raise
        except urllib.error.HTTPError as e:
            try:
                e.close()      # the error owns the response socket and may outlive this handler (``last``)
            except Exception:  # noqa: BLE001 -- an error built without a body has nothing to close
                pass
            if e.code in (404, 410):
                raise AssetNotFound(f"not found (HTTP {e.code}): {target}") from None
            if e.code in _PERMANENT_HTTP:
                raise RuntimeError(f"HTTP {e.code} {e.reason} for {target}") from e
            last = e
        except Exception as e:   # noqa: BLE001 -- timeouts, resets, truncation: all worth another try
            last = e
    raise RuntimeError(f"failed to download {target} after {max(0, retries) + 1} attempt(s): {last!r}") from last


def fetch(rel: str, cache_dir: str, *, timeout: float = 120, retries: int = 3,
          log: Callable[[str], None] | None = None) -> str:
    """Download ``rel`` to ``cache_dir/rel`` unless it is already there (non-empty); returns the local path.

    ``timeout`` is per socket operation, not for the whole transfer.  ``retries`` are extra attempts after
    the first, with 1 s, 2 s, 4 s ... backoff.  Raises ``AssetNotFound`` (a ``RuntimeError``) on 404 and
    ``RuntimeError`` naming the URL on any other final failure; no partial file is left behind.  A cache
    that cannot be written (disk full, no permission) raises ``RuntimeError`` at once instead of retrying.
    """
    local = _local_path(rel, cache_dir)
    if _is_cached(local):
        return local
    target = url(rel)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    size = _retrying(target, lambda: _download(target, local, timeout), retries, log)
    _say(log, f"fetched {rel} ({size / 1e6:.2f} MB)")
    return local


def list_dir(rel_dir: str, *, timeout: float = 60, retries: int = 3) -> tuple[list[str], list[str]]:
    """``(subdirs, files)`` directly inside ``rel_dir``, as sorted bare names (``.thumbs`` skipped).

    Uses the S3 ListObjectsV2 API and follows continuation tokens, so directories of any size are listed.
    S3 has no real directories: a path that does not exist lists as ``([], [])`` rather than raising.
    """
    parts = urllib.parse.urlsplit(NUCLEUS_ROOT)
    endpoint = f"{parts.scheme}://{parts.netloc}/"
    prefix = "/".join([*(s for s in parts.path.split("/") if s), *_segments(rel_dir)])
    prefix = prefix + "/" if prefix else ""
    subdirs: list[str] = []
    files: list[str] = []
    token, seen_tokens = None, set()
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            query["continuation-token"] = token
        target = endpoint + "?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        root = ET.fromstring(_retrying(target, lambda: _read_all(target, timeout), retries, None))
        token = None
        for el in root:
            tag = el.tag.rpartition("}")[2]
            if tag == "CommonPrefixes":
                for p in el:
                    name = (p.text or "")[len(prefix):].strip("/")
                    if name and name != ".thumbs":
                        subdirs.append(name)
            elif tag == "Contents":
                key = next(((k.text or "") for k in el if k.tag.rpartition("}")[2] == "Key"), "")
                name = key[len(prefix):]
                if name and "/" not in name:
                    files.append(name)
            elif tag == "NextContinuationToken":
                token = el.text
        if not token or token in seen_tokens:
            break
        seen_tokens.add(token)
    return sorted(subdirs), sorted(files)


def _read_all(target: str, timeout: float) -> bytes:
    resp = urllib.request.urlopen(urllib.request.Request(target, headers={"User-Agent": _USER_AGENT}),
                                  timeout=timeout)
    try:
        return resp.read()
    finally:
        close = getattr(resp, "close", None)
        if close is not None:
            close()


def local_or_url(rel: str, cache_dir: str | None) -> str:
    """The cached local path of ``rel`` if it is there, else its URL (Isaac Sim can load URLs directly)."""
    if cache_dir:
        local = _local_path(rel, cache_dir)
        if _is_cached(local):
            return local
    return url(rel)


# ---------------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _dep_rel(base_rel: str, ref: str) -> tuple[str | None, str]:
    """Root-relative path a USD/MDL reference points at, or ``(None, reason)`` if it cannot be mirrored.

    ``ref`` is anchored at the directory of ``base_rel`` (the file that contains it) unless it is a
    Nucleus URL or starts with ``/`` (relative to the asset root, as Nucleus servers treat it).
    """
    ref = ref.strip()
    if not ref:
        return None, "empty path"
    if ref.startswith(NUCLEUS_ROOT.rstrip("/") + "/"):
        ref = "/" + urllib.parse.unquote(ref[len(NUCLEUS_ROOT.rstrip("/")) + 1:])
    elif _SCHEME_RE.match(ref) or ref.startswith("anon:"):
        return None, "external or anonymous location"
    elif _DRIVE_RE.match(ref):
        return None, "absolute filesystem path"
    if "<UDIM>" in ref.upper() or "${" in ref:
        return None, "templated path"
    if "/" not in ref and ref in _KIT_MDL:
        return None, "ships with Kit"
    anchored = ref if ref.startswith("/") else posixpath.join(posixpath.dirname(base_rel), ref)
    try:
        parts = _segments(anchored)
    except ValueError:
        return None, "leaves the asset root"
    return ("/".join(parts), "") if parts else (None, "empty path")


def _usd_references(path: str) -> list[str]:
    """Every file a USD layer names: sublayers, references, payloads and asset-valued attributes."""
    try:
        from pxr import UsdUtils
    except ImportError as e:
        raise ImportError("mirror_usd needs the pxr module (pip install usd-core, or run in Isaac Sim's Python)") from e
    try:
        groups = UsdUtils.ExtractExternalReferences(path)
    except Exception as e:   # noqa: BLE001 -- pxr raises assorted Tf errors for unreadable layers
        raise RuntimeError(f"cannot read USD layer {path}: {e}") from e
    return list(dict.fromkeys(p for group in groups for p in group))


_MDL_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|//[^\n]*|/\*.*?\*/', re.S)
_MDL_RESOURCE = re.compile(r'\b(?:texture_2d|texture_3d|texture_cube|texture_ptex|light_profile|'
                           r'bsdf_measurement)\s*\(\s*"((?:[^"\\\n]|\\.)*)"')
_MDL_STATEMENT = re.compile(r"\b(import|using)\b([^;]*);")
_MDL_PATH = re.compile(r"^[\w.:*]+$")


def _mdl_code(text: str, keep_strings: bool) -> str:
    """MDL source without comments; string bodies are blanked unless ``keep_strings`` (they may hold ``//``)."""
    def sub(m: re.Match) -> str:
        s = m.group(0)
        return (s if keep_strings else '""') if s[0] == '"' else " "
    return _MDL_TOKEN.sub(sub, text)


def _mdl_resources(text: str) -> list[str]:
    """Paths in ``texture_2d("...")``-style resource constructors (comments ignored)."""
    found = (m.group(1) for m in _MDL_RESOURCE.finditer(_mdl_code(text, keep_strings=True)))
    return list(dict.fromkeys(p for p in found if p))


def _mdl_modules(text: str) -> list[str]:
    """Module paths (``..::Templates::Shared``) from ``import``/``using`` statements, imported names dropped."""
    out: list[str] = []
    for m in _MDL_STATEMENT.finditer(_mdl_code(text, keep_strings=False)):
        body = m.group(2).strip()
        if m.group(1) == "using":
            head = re.split(r"\bimport\b", body, maxsplit=1)[0].strip()
            candidates = [head]
        else:
            candidates = [p.strip().rpartition("::")[0] for p in body.split(",")]
        out += [c for c in candidates if c and _MDL_PATH.match(c)]
    return list(dict.fromkeys(out))


def _module_rel(base_rel: str, module: str) -> str | None:
    """File an MDL module path would be, next to or relative to ``base_rel``; ``None`` for Kit libraries."""
    toks = [t for t in module.split("::") if t]
    ups, relative = 0, False
    while toks and toks[0] in (".", ".."):
        relative = True
        ups += toks[0] == ".."
        toks = toks[1:]
    if not toks or (not relative and toks[0] in _KIT_MODULES) or any(t == "*" for t in toks):
        return None
    # ``::Foo::x`` (absolute) and ``Foo::x`` (weak relative) are looked up beside the importing file.
    anchored = posixpath.join(posixpath.dirname(base_rel), *([".."] * ups), *toks) + ".mdl"
    try:
        return "/".join(_segments(anchored)) or None
    except ValueError:
        return None


def _dependencies(rel: str, local: str, log: Callable[[str], None] | None) -> list[str]:
    """Root-relative paths of the files that the USD/MDL file ``rel`` (cached at ``local``) needs."""
    ext = posixpath.splitext(rel)[1].lower()
    out: list[str] = []

    def add(raw: str, dep: str | None, reason: str) -> None:
        if dep is not None:
            out.append(dep)
        elif reason:
            _say(log, f"skipped {raw!r} in {rel}: {reason}")

    if ext in _USD_EXT:
        for raw in _usd_references(local):
            add(raw, *_dep_rel(rel, raw))
    elif ext == ".mdl":
        with open(local, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for raw in _mdl_resources(text):
            add(raw, *_dep_rel(rel, raw))
        for module in _mdl_modules(text):
            add(module, _module_rel(rel, module), "")
    return out


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------

def _fetch_many(rels: list[str], cache_dir: str, workers: int, log: Callable[[str], None] | None) -> list[bool]:
    """Fetch ``rels`` (in parallel); ``False`` marks files the server does not have."""
    def one(r: str) -> bool:
        try:
            fetch(r, cache_dir, log=log)
            return True
        except AssetNotFound:
            _say(log, f"skipped (not on server): {r}")
            return False
    if workers <= 1 or len(rels) <= 1:
        return [one(r) for r in rels]
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(rels))) as pool:
        return list(pool.map(one, rels))


def mirror_usd(rel: str, cache_dir: str, *, log: Callable[[str], None] | None = None, max_files: int = 500,
               workers: int = 8) -> str:
    """Fetch ``rel`` and everything it needs into ``cache_dir``, keeping the server layout; returns the local root.

    Dependencies come from pxr (sublayers, references, payloads, every asset-valued attribute such as
    textures and MDL sources) for USD files, and from the MDL source itself for ``.mdl`` files
    (``texture_2d("...")`` resources and ``import`` of sibling modules; Kit's own libraries such as
    ``::OmniPBR`` or ``::tex`` are not on the server and are ignored).  Relative (``./``, ``../``),
    root-absolute (``/NVIDIA/...``) and full Nucleus-URL references are followed.  A dependency the
    server answers 404 for is logged and skipped; the root itself must exist.  Any other failure (network
    errors after the retries) raises, and a re-run resumes since cached files are not fetched again.

    Raises ``RuntimeError`` if the closure exceeds ``max_files`` files (guard against runaway scenes).
    Root-absolute references are downloaded to the same layout but pxr cannot resolve them locally,
    since an OS-absolute path points outside ``cache_dir``.
    """
    root = "/".join(_segments(rel))
    local_root = fetch(root, cache_dir, log=log)
    seen = {root}
    fresh = [root]                     # downloaded but not yet scanned for their own dependencies
    missing = 0
    while fresh:
        wanted: list[str] = []
        for r in fresh:
            for dep in _dependencies(r, _local_path(r, cache_dir), log):
                if dep not in seen:
                    seen.add(dep)
                    wanted.append(dep)
        if len(seen) > max_files:
            raise RuntimeError(f"mirror_usd({rel!r}) needs more than max_files={max_files} files; raise max_files")
        present = _fetch_many(wanted, cache_dir, workers, log)
        fresh = [d for d, ok in zip(wanted, present) if ok]
        missing += present.count(False)
    _say(log, f"mirrored {root}: {len(seen) - missing} files in {cache_dir} ({missing} skipped)")
    return local_root
