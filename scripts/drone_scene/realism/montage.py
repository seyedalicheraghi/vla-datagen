"""Pictures for the agents (and you) to look at: a contact sheet of a probe run, and blind A/B pairs for the judges.
numpy + Pillow only.

    python scripts/drone_scene/realism/montage.py contact drone_scene_videos/probe
        -> drone_scene_videos/probe/contact.jpg: every still, labelled, in flight order
    python scripts/drone_scene/realism/montage.py ab runs/base/probe runs/cand/probe --out runs/cand/ab --seed 3
        -> runs/cand/ab/<still>.jpg (the two versions side by side, labelled only "1" and "2", in an order set by
           --seed), <still>_detail.jpg (the same crop of both at full resolution), the same two with the sides
           swapped (<still>_swap.jpg, <still>_detail_swap.jpg) and key.json (which side is A in each picture).
           Show the judges the pictures, never key.json, and let them pick "1", "2" or "tie".  Judge the normal
           and the swapped pictures in separate calls: a verdict that flips with the order is a tie (vision judges
           favour one screen position often enough for this to matter).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ORDER = ["ground", "liftoff", "climb", "cruise_early", "cruise_mid", "cruise_late", "descent", "final"]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _label(im: Image.Image, text: str, size: int = 28) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    font = _font(size)
    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.rectangle([0, 0, w + 16, h + 14], fill=(0, 0, 0))
    d.text((8 - box[0], 6 - box[1]), text, fill=(255, 255, 255), font=font)
    return im


def _fit(im: Image.Image, width: int) -> Image.Image:
    if im.width <= width:
        return im
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def _sort_key(name: str):
    """Camera, then flight order: named moments first, then times (t2 < t10), then anything else."""
    stem = os.path.splitext(name)[0]
    cam, _, moment = stem.partition("_")
    m = re.fullmatch(r"t(\d+(?:\.\d+)?)", moment)
    return (cam, ORDER.index(moment) if moment in ORDER else len(ORDER), float(m.group(1)) if m else math.inf, moment)


def stills(folder: str) -> list[str]:
    return sorted((f for f in os.listdir(folder) if f.lower().endswith(".png") and not f.startswith("raw_")),
                  key=_sort_key)


def contact(folder: str, out: str | None = None, tile: int = 640) -> str:
    """Every still of a probe run on one sheet: one row per camera, flight order left to right."""
    names = stills(folder)
    if not names:
        raise SystemExit(f"no stills in {folder}")
    rows: dict[str, list] = {}
    for n in names:
        rows.setdefault(n.partition("_")[0], []).append(n)
    tiles = {n: _label(_fit(Image.open(os.path.join(folder, n)).convert("RGB"), tile), os.path.splitext(n)[0], 22)
             for n in names}
    th = max(t.height for t in tiles.values())
    cols = max(len(r) for r in rows.values())
    sheet = Image.new("RGB", (cols * (tile + 8) + 8, len(rows) * (th + 8) + 8), (40, 40, 40))
    for i, row in enumerate(rows.values()):
        for j, n in enumerate(row):
            sheet.paste(tiles[n], (8 + j * (tile + 8), 8 + i * (th + 8)))
    out = out or os.path.join(folder, "contact.jpg")
    sheet.save(out, quality=90)
    return out


def a_is_left(name: str, seed: int) -> bool:
    """The side A goes on for this still and seed: fixed by a hash, so a run can be re-made, but unpredictable to
    the judge (the judges never see which side is which)."""
    return hashlib.sha1(f"{seed}:{name}".encode()).digest()[0] % 2 == 0


def pair(left: Image.Image, right: Image.Image, width: int) -> Image.Image:
    left, right = _label(_fit(left, width), "1", 34), _label(_fit(right, width), "2", 34)
    h = max(left.height, right.height)
    im = Image.new("RGB", (left.width + right.width + 16, h), (255, 255, 255))
    im.paste(left, (0, 0))
    im.paste(right, (left.width + 16, 0))
    return im


def detail_box(w: int, h: int, size: int = 768) -> tuple[int, int, int, int]:
    """A full-resolution crop below the image centre (the ground at a few tens of metres in the onboard views)."""
    s = min(size, w, h)
    cx, cy = w // 2, int(h * 0.68)
    x0 = min(max(0, cx - s // 2), w - s)
    y0 = min(max(0, cy - s // 2), h - s)
    return x0, y0, x0 + s, y0 + s


def ab(a_dir: str, b_dir: str, out: str, seed: int = 0, width: int = 960, detail: bool = True) -> dict:
    """Blind side-by-side pairs of every still in both folders, each also with the sides swapped, and key.json
    saying which side is A in every picture ({picture name: {"1": "A" or "B", "2": ...}})."""
    os.makedirs(out, exist_ok=True)
    common = [n for n in stills(a_dir) if os.path.exists(os.path.join(b_dir, n))]
    if not common:
        raise SystemExit(f"no stills in both {a_dir} and {b_dir}")
    key = {}
    for n in common:
        ia = Image.open(os.path.join(a_dir, n)).convert("RGB")
        ib = Image.open(os.path.join(b_dir, n)).convert("RGB")
        stem = os.path.splitext(n)[0]
        crops = [(stem, None)]
        if detail and ia.size == ib.size:
            crops.append((f"{stem}_detail", detail_box(*ia.size)))
        for name, box in crops:
            pa, pb = (ia, ib) if box is None else (ia.crop(box), ib.crop(box))
            w = width if box is None else box[2] - box[0]
            for suffix, left_is_a in (("", a_is_left(n, seed)), ("_swap", not a_is_left(n, seed))):
                left, right = (pa, pb) if left_is_a else (pb, pa)
                pair(left, right, w).save(os.path.join(out, f"{name}{suffix}.jpg"), quality=92)
                key[name + suffix] = {"1": "A" if left_is_a else "B", "2": "B" if left_is_a else "A"}
    with open(os.path.join(out, "key.json"), "w", encoding="utf-8") as f:
        json.dump({"A": os.path.abspath(a_dir), "B": os.path.abspath(b_dir), "seed": seed, "pairs": key}, f, indent=1)
    return key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contact", help="one sheet with every still of a probe run")
    c.add_argument("folder")
    c.add_argument("--out")
    c.add_argument("--tile", type=int, default=640)
    p = sub.add_parser("ab", help="blind A/B pairs for the judges")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--no-detail", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "contact":
        print(contact(a.folder, a.out, a.tile))
    else:
        key = ab(a.a, a.b, a.out, a.seed, a.width, not a.no_detail)
        print(f"{len(key)} pairs in {a.out} (key.json says which side is A; do not show it to the judges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
