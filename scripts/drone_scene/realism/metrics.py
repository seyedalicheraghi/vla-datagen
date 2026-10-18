"""Image statistics for judging how real a rendered frame looks, and whether a change made it better or worse.
numpy + Pillow only: runs with any Python on the PC, no GPU, no Isaac Sim.

    python scripts/drone_scene/realism/metrics.py drone_scene_videos/probe                  # one probe run
    python scripts/drone_scene/realism/metrics.py runs/base/probe runs/cand/probe           # A/B: per-still change
    python scripts/drone_scene/realism/metrics.py runs/cand/probe --reference refs/front    # against real photos
    python scripts/drone_scene/realism/metrics.py runs/cand/probe --pairs .realism/reference/naip
        # each still against the real picture of the same name (the nadir stills against NAIP orthophotos of the
        # same square, made by reference.py): colour difference and coarse structural similarity

It prints a table and writes ``metrics.json`` (``--out``).  What each number catches (rough values for real
daylight photos in brackets; they are guides for the agents, not pass/fail limits):

- ``mean_luma``: mean linear luminance (exposure) [0.10-0.25 for a ground view in sun];
- ``clip_hi`` / ``clip_lo``: fraction of pixels at 254-255 in any channel / at 0-2 in all channels [clip_hi under
  ~2 % unless the sky is in view; clip_lo under ~1 %];
- ``contrast``: RMS contrast of the sRGB grey image;
- ``colorfulness``: Hasler and Suesstrunk (2003) [about 15-50 for landscapes; CG is often too high or too grey];
- ``saturation``: mean HSV saturation;
- ``spectral_slope``: slope of the radially averaged power spectrum (log-log, 0.02-0.25 cycles/pixel) [about -2
  for natural scenes; too smooth (blurry textures, no fine clutter) is steeper, aliasing or noise is flatter];
- ``noise_sigma``: Immerkaer (1996) noise estimate in 8-bit levels; texture raises it, so compare A with B, not with
  a fixed number;
- ``sharpness``: variance of the Laplacian (8-bit levels squared);
- ``repetition``: strongest off-centre peak of the normalised autocorrelation of the lower half (the ground)
  [tiled textures seen from above give high values; best on the ``nadir`` view];
- ``haze_contrast`` / ``haze_blue``: RMS contrast of the top third over the bottom third, and how much bluer (B-R)
  the top third is [front/back views: distant land should have less contrast and be bluer, aerial perspective];
- ``ground_lab``: mean CIE L*a*b* of the lower half (compare with the real place, e.g. the NAIP reference).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

KEYS = ["mean_luma", "clip_hi", "clip_lo", "contrast", "colorfulness", "saturation", "spectral_slope", "noise_sigma",
        "sharpness", "repetition", "haze_contrast", "haze_blue"]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --- conversions ----------------------------------------------------------------

def load(path: str) -> np.ndarray:
    """An image as float32 sRGB in [0, 1], (H, W, 3)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def grey(rgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded grey (Rec. 709 weights on the encoded values), in [0, 1]."""
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB values in [0, 1]."""
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    t = (srgb_to_linear(rgb.astype(np.float64)) @ m.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window (reflected edges), via an integral image."""
    k = 2 * r + 1
    c = np.pad(np.pad(a.astype(np.float64), r, mode="reflect").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


# --- statistics -----------------------------------------------------------------

def colorfulness(rgb: np.ndarray) -> float:
    """Hasler and Suesstrunk (2003), on 8-bit sRGB."""
    r, g, b = (rgb[..., i].astype(np.float64) * 255.0 for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def saturation(rgb: np.ndarray) -> float:
    mx, mn = rgb.max(-1), rgb.min(-1)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean())


def spectral_slope(g: np.ndarray, lo: float = 0.02, hi: float = 0.25) -> float:
    """Slope of log power against log spatial frequency (cycles/pixel) over [lo, hi], on the central square crop
    with a Hann window."""
    h, w = g.shape
    n = min(h, w)
    sq = g[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n].astype(np.float64)
    sq = (sq - sq.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fft2(sq)) ** 2
    f = np.fft.fftfreq(n)
    freq = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    edges = np.logspace(np.log10(lo), np.log10(hi), 21)
    idx = np.digitize(freq.ravel(), edges)
    pw = power.ravel()
    xs, ys = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.any():
            xs.append(np.log(np.sqrt(edges[i - 1] * edges[i])))
            ys.append(np.log(max(float(pw[m].mean()), 1e-30)))
    if len(xs) < 3:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def noise_sigma(g: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate, in 8-bit levels (an upper bound: texture adds to it)."""
    a = g.astype(np.float64) * 255.0
    m = (a[:-2, :-2] - 2 * a[:-2, 1:-1] + a[:-2, 2:] - 2 * a[1:-1, :-2] + 4 * a[1:-1, 1:-1] - 2 * a[1:-1, 2:]
         + a[2:, :-2] - 2 * a[2:, 1:-1] + a[2:, 2:])
    h, w = a.shape
    return float(np.sqrt(np.pi / 2) * np.abs(m).sum() / (6.0 * (w - 2) * (h - 2)))


def sharpness(g: np.ndarray) -> float:
    a = g.astype(np.float64) * 255.0
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def repetition(g: np.ndarray, min_shift: int = 12, max_side: int = 768) -> float:
    """Strongest normalised autocorrelation at shifts of ``min_shift`` px up to half the image, on the high-passed
    lower half: near 0 for natural ground, high for textures that visibly tile."""
    a = g[g.shape[0] // 2:].astype(np.float64)
    step = max(1, int(np.ceil(max(a.shape) / max_side)))
    a = a[::step, ::step]
    a = a - box_blur(a, 15)
    a = a - a.mean()
    h, w = a.shape
    if h < 4 * min_shift or w < 4 * min_shift or not a.any():
        return float("nan")
    spec = np.fft.rfft2(a, s=(2 * h, 2 * w))                       # zero-padded: linear, not circular
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(spec) ** 2, s=(2 * h, 2 * w)))
    dy, dx = np.meshgrid(np.arange(2 * h) - h, np.arange(2 * w) - w, indexing="ij")
    overlap = (h - np.abs(dy)) * (w - np.abs(dx))
    r = ac / np.maximum(overlap, 1) / (ac[h, w] / (h * w))
    ok = (np.abs(dy) < h // 2) & (np.abs(dx) < w // 2) & (np.hypot(dy, dx) >= min_shift)
    return float(r[ok].max())


def haze(rgb: np.ndarray) -> tuple[float, float]:
    """(contrast of the top third / bottom third, mean B-R of the top third minus the bottom third)."""
    h = rgb.shape[0]
    top, bot = rgb[: h // 3], rgb[h - h // 3:]
    c = float(grey(top).std() / max(float(grey(bot).std()), 1e-6))
    blue = float((top[..., 2] - top[..., 0]).mean() - (bot[..., 2] - bot[..., 0]).mean())
    return c, blue


def image_stats(rgb: np.ndarray) -> dict:
    g = grey(rgb)
    lin = srgb_to_linear(rgb.astype(np.float64)) @ np.array([0.2126, 0.7152, 0.0722])
    hc, hb = haze(rgb)
    lab = srgb_to_lab(rgb[rgb.shape[0] // 2:]).reshape(-1, 3).mean(0)
    return {
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "mean_luma": float(lin.mean()),
        "clip_hi": float((rgb.max(-1) >= 254 / 255).mean()),
        "clip_lo": float((rgb.max(-1) <= 2 / 255).mean()),
        "contrast": float(g.std()),
        "colorfulness": colorfulness(rgb),
        "saturation": saturation(rgb),
        "spectral_slope": spectral_slope(g),
        "noise_sigma": noise_sigma(g),
        "sharpness": sharpness(g),
        "repetition": repetition(g),
        "haze_contrast": hc,
        "haze_blue": hb,
        "ground_lab": [round(float(v), 2) for v in lab],
    }


# --- runs -----------------------------------------------------------------------

def images_in(folder: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXT)
                  and not os.path.basename(p).startswith(("contact", "raw_")))


def probe_records(folder: str) -> dict:
    """{file name: its probe.json record} when the folder is a ``--probe`` output."""
    path = os.path.join(folder, "probe.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {s["file"]: s for s in json.load(f).get("stills", [])}


def measure(folder: str) -> dict:
    recs = probe_records(folder)
    out = {}
    for p in images_in(folder):
        name = os.path.basename(p)
        out[name] = image_stats(load(p))
        if name in recs:
            out[name]["probe"] = recs[name]
    return out


def compare(a: dict, b: dict) -> dict:
    """Per still in both runs: B minus A for every statistic, and plain-language warnings for big moves."""
    out = {}
    for name in sorted(set(a) & set(b)):
        d = {k: b[name][k] - a[name][k] for k in KEYS if np.isfinite(a[name][k]) and np.isfinite(b[name][k])}
        warn = []
        if d.get("clip_hi", 0) > 0.02:
            warn.append(f"more blown highlights (+{100 * d['clip_hi']:.1f} % of pixels)")
        if d.get("clip_lo", 0) > 0.01:
            warn.append(f"more crushed shadows (+{100 * d['clip_lo']:.1f} % of pixels)")
        if "spectral_slope" in d and abs(b[name]["spectral_slope"] + 2) > abs(a[name]["spectral_slope"] + 2) + 0.2:
            warn.append(f"spectrum further from natural (-2): {a[name]['spectral_slope']:.2f} -> {b[name]['spectral_slope']:.2f}")
        if d.get("repetition", 0) > 0.1:
            warn.append(f"more visible texture tiling ({a[name]['repetition']:.2f} -> {b[name]['repetition']:.2f})")
        if abs(d.get("mean_luma", 0)) > 0.5 * max(a[name]["mean_luma"], 1e-6):
            warn.append(f"exposure changed a lot ({a[name]['mean_luma']:.3f} -> {b[name]['mean_luma']:.3f})")
        dl = float(np.linalg.norm(np.subtract(b[name]["ground_lab"], a[name]["ground_lab"])))
        out[name] = {"delta": d, "ground_lab_shift": round(dl, 2), "warnings": warn}
    return out


def against(run: dict, ref: dict) -> dict:
    """Where each still sits among the real reference images: per statistic, the reference range (10th-90th
    percentile; min-max with fewer than 5 references) and whether the still is inside it."""
    if not ref:
        return {}
    rng = {}
    for k in KEYS:
        v = np.array([r[k] for r in ref.values() if np.isfinite(r[k])])
        if len(v):
            lo, hi = (np.percentile(v, 10), np.percentile(v, 90)) if len(v) >= 5 else (v.min(), v.max())
            rng[k] = (float(lo), float(hi))
    lab = np.array([r["ground_lab"] for r in ref.values()]).mean(0)
    out = {"reference_range": rng, "reference_ground_lab": [round(float(v), 2) for v in lab], "stills": {}}
    for name, s in run.items():
        outside = {k: round(s[k], 4) for k, (lo, hi) in rng.items() if np.isfinite(s[k]) and not lo <= s[k] <= hi}
        out["stills"][name] = {"outside_reference": outside,
                               "ground_lab_distance": round(float(np.linalg.norm(np.subtract(s["ground_lab"], lab))), 2)}
    return out


def coarse_ssim(a: np.ndarray, b: np.ndarray, size: int = 128) -> float:
    """SSIM of the two grey images shrunk to size x size (7 x 7 windows): does the layout match (roads, roofs,
    fields, tree clumps), ignoring fine texture and small misalignment."""
    def small(x):
        im = Image.fromarray(np.clip(grey(x) * 255.0 + 0.5, 0, 255).astype(np.uint8))
        return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.float64) / 255.0
    x, y = small(a), small(b)
    mx, my = box_blur(x, 3), box_blur(y, 3)
    vx, vy = box_blur(x * x, 3) - mx * mx, box_blur(y * y, 3) - my * my
    cxy = box_blur(x * y, 3) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())


def paired(run_dir: str, ref_dir: str) -> dict:
    """Each still against the real picture with the same file name: mean colour difference (CIE76 dE of the
    whole-image mean L*a*b*), coarse SSIM, and the real picture's own statistics for comparison."""
    out = {}
    for p in images_in(run_dir):
        name = os.path.basename(p)
        q = os.path.join(ref_dir, name)
        if not os.path.exists(q):
            continue
        sim, real = load(p), load(q)
        if real.shape != sim.shape:
            real = np.asarray(Image.fromarray((real * 255).astype(np.uint8)).resize((sim.shape[1], sim.shape[0]),
                                                                                    Image.LANCZOS)) / 255.0
        ls, lr = srgb_to_lab(sim).reshape(-1, 3).mean(0), srgb_to_lab(real).reshape(-1, 3).mean(0)
        rs = image_stats(real)
        out[name] = {"lab_sim": [round(float(v), 2) for v in ls], "lab_real": [round(float(v), 2) for v in lr],
                     "colour_dE": round(float(np.linalg.norm(ls - lr)), 2), "ssim_coarse": round(coarse_ssim(sim, real), 4),
                     "real": {k: rs[k] for k in ("colorfulness", "saturation", "spectral_slope", "contrast")}}
    return out


def table(run: dict) -> str:
    cols = ["mean_luma", "clip_hi", "colorfulness", "spectral_slope", "noise_sigma", "repetition", "haze_contrast", "haze_blue"]
    lines = ["| still | " + " | ".join(cols) + " | ground L*a*b* |", "|---" * (len(cols) + 2) + "|"]
    for name, s in run.items():
        lines.append(f"| {name} | " + " | ".join(f"{s[k]:.3g}" for k in cols) + " | "
                     + ", ".join(f"{v:.0f}" for v in s["ground_lab"]) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder of stills (a --probe output: <out>/probe)")
    ap.add_argument("other", nargs="?", help="a second folder: compare it (B) with the first (A)")
    ap.add_argument("--reference", help="folder of real photos to compare the (last) run with")
    ap.add_argument("--pairs", help="folder of real pictures named like the stills (e.g. reference.py's NAIP squares)")
    ap.add_argument("--out", help="where to write the JSON (default: metrics.json in the last run's folder)")
    a = ap.parse_args(argv)
    for d in [a.run, a.other, a.reference, a.pairs]:
        if d and not os.path.isdir(d):
            print(f"no such folder: {d}", file=sys.stderr)
            return 2
    runs = {"A": measure(a.run)}
    if not runs["A"]:
        print(f"no images in {a.run}", file=sys.stderr)
        return 2
    result = {"A": {"folder": os.path.abspath(a.run), "stills": runs["A"]}}
    last = runs["A"]
    if a.other:
        runs["B"] = measure(a.other)
        result["B"] = {"folder": os.path.abspath(a.other), "stills": runs["B"]}
        result["B_minus_A"] = compare(runs["A"], runs["B"])
        last = runs["B"]
    if a.reference:
        ref = {os.path.basename(p): image_stats(load(p)) for p in images_in(a.reference)}
        result["reference"] = {"folder": os.path.abspath(a.reference), **against(last, ref)}
    if a.pairs:
        result["pairs"] = {"folder": os.path.abspath(a.pairs), "A": paired(a.run, a.pairs)}
        if a.other:
            result["pairs"]["B"] = paired(a.other, a.pairs)
    out = a.out or os.path.join(a.other or a.run, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for key, run in runs.items():
        print(f"\n{key}: {result[key]['folder']}\n{table(run)}")
    if "B_minus_A" in result:
        print("\nB - A warnings:")
        for name, c in result["B_minus_A"].items():
            print(f"  {name}: " + ("; ".join(c["warnings"]) or "none") + f" (ground colour moved {c['ground_lab_shift']} dE)")
    if "reference" in result:
        print("\noutside the real photos' range:")
        for name, c in result["reference"]["stills"].items():
            print(f"  {name}: " + (", ".join(f"{k}={v}" for k, v in c["outside_reference"].items()) or "none")
                  + f" (ground colour {c['ground_lab_distance']} dE from the references)")
    if "pairs" in result:
        for key in ("A", "B"):
            if key not in result["pairs"]:
                continue
            print(f"\n{key} against the real picture of the same place:")
            for name, c in result["pairs"][key].items():
                print(f"  {name}: colour {c['colour_dE']} dE, coarse SSIM {c['ssim_coarse']}")
            if not result["pairs"][key]:
                print("  no still has a real picture with the same name")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
