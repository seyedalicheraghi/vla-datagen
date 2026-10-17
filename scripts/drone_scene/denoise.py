"""Intel Open Image Denoise (OIDN) through ctypes, for the offline previews and flight videos.

``denoise(color, albedo=None, normal=None)`` returns the denoised linear HDR image (the "RT" filter, as in
Blender/Cycles), or the input unchanged when OIDN is not installed (``pip install oidn``; it ships its own
library).  Albedo and normal guides keep texture and geometric detail.
"""

from __future__ import annotations

import ctypes
import glob
import os

import numpy as np

_LIB = None
_DEVICE = None
_FORMAT_FLOAT3 = 3                      # OIDN_FORMAT_FLOAT3


def _load():
    global _LIB, _DEVICE
    if _LIB is not None:
        return _LIB
    import importlib.util
    spec = importlib.util.find_spec("oidn")          # only for its bundled shared libraries; importing it
    if spec is None or not spec.submodule_search_locations:   # needs LD_LIBRARY_PATH, so it is not imported
        _LIB = False
        return _LIB
    libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib.linux.x64")
    # its TBB first, so the denoiser does not need LD_LIBRARY_PATH
    for dep in sorted(glob.glob(os.path.join(libdir, "libtbb.so*"))):
        try:
            ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    libs = sorted(glob.glob(os.path.join(libdir, "libOpenImageDenoise.so*")), key=len)
    try:
        lib = ctypes.CDLL(libs[0])
    except (OSError, IndexError):
        _LIB = False
        return _LIB
    lib.oidnNewDevice.restype = ctypes.c_void_p
    lib.oidnNewDevice.argtypes = [ctypes.c_int]
    lib.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    lib.oidnNewFilter.restype = ctypes.c_void_p
    lib.oidnNewFilter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.oidnSetSharedFilterImage.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                             ctypes.c_size_t]
    set_bool = getattr(lib, "oidnSetFilterBool", None) or getattr(lib, "oidnSetFilter1b")
    set_bool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
    lib.set_bool = set_bool
    lib.oidnCommitFilter.argtypes = [ctypes.c_void_p]
    lib.oidnExecuteFilter.argtypes = [ctypes.c_void_p]
    lib.oidnReleaseFilter.argtypes = [ctypes.c_void_p]
    lib.oidnGetDeviceError.restype = ctypes.c_int
    lib.oidnGetDeviceError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    _DEVICE = lib.oidnNewDevice(0)
    lib.oidnCommitDevice(_DEVICE)
    _LIB = lib
    return _LIB


def available() -> bool:
    return bool(_load())


def denoise(color: np.ndarray, albedo: np.ndarray | None = None, normal: np.ndarray | None = None) -> np.ndarray:
    lib = _load()
    if not lib:
        return color
    h, w, _ = color.shape
    bufs = {"color": np.ascontiguousarray(color, np.float32)}
    if albedo is not None:
        bufs["albedo"] = np.ascontiguousarray(np.clip(albedo, 0.0, 1.0), np.float32)
        if normal is not None:
            bufs["normal"] = np.ascontiguousarray(normal, np.float32)
    out = np.empty_like(bufs["color"])
    f = lib.oidnNewFilter(_DEVICE, b"RT")
    try:
        for name, a in list(bufs.items()) + [("output", out)]:
            lib.oidnSetSharedFilterImage(f, name.encode(), a.ctypes.data, _FORMAT_FLOAT3, w, h, 0, 0, 0)
        lib.set_bool(f, b"hdr", True)
        lib.oidnCommitFilter(f)
        lib.oidnExecuteFilter(f)
        msg = ctypes.c_char_p()
        if lib.oidnGetDeviceError(_DEVICE, ctypes.byref(msg)) != 0:
            raise RuntimeError(f"OIDN: {msg.value.decode(errors='replace') if msg.value else 'error'}")
    finally:
        lib.oidnReleaseFilter(f)
    return out
