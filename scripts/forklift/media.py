"""Image helpers shared by the dataset writers, the closed-loop evaluator and
the demo videos (numpy + Pillow + imageio only — no Isaac Sim imports).

`dataset_image` is the single definition of how a camera frame becomes a
training image. The evaluator applies the same function before calling a
policy, so at test time the policy sees exactly what it was trained on.
"""

from __future__ import annotations

import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_rgb_uint8(frame) -> np.ndarray:
    """Camera output (torch or numpy; HxWxC or 1xHxWxC; RGB or RGBA) → HxWx3 uint8."""
    arr = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    return np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8)


def dataset_image(rgb: np.ndarray, size: int) -> Image.Image:
    """Center-crop to a square, then resize to size×size.

    Cropping first keeps the aspect ratio — resizing a 4:3 frame straight to
    a square would squash the scene.
    """
    h, w = rgb.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    img = Image.fromarray(rgb[top:top + side, left:left + side])
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


def video_path(arg: str, episode: int, prefix: str) -> tuple[str, bool]:
    """`--video` value → (file path, one-clip-per-episode?).

    A name ending in .mp4 = one video for the whole run; anything else is a
    folder that gets one short clip per episode (<prefix>_ep000.mp4, ...).
    """
    import os
    arg = os.path.expanduser(arg)
    if arg.lower().endswith(".mp4"):
        os.makedirs(os.path.dirname(os.path.abspath(arg)), exist_ok=True)
        return arg, False
    os.makedirs(arg, exist_ok=True)
    return os.path.join(arg, f"{prefix}_ep{episode:03d}.mp4"), True


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 10.1
        return ImageFont.load_default()


class DemoVideo:
    """MP4 for demos: the chase view (1280×720) on the left, the three
    dataset cameras stacked on the right, captions and a badge on top.

    The canvas is 1520×720 — both sides multiples of 16, as H.264 wants.
    """

    CHASE_W, CHASE_H = 1280, 720
    TILE = 240                 # 3 tiles stacked = 720

    def __init__(self, path: str, fps: int = 30):
        import imageio.v2 as imageio
        # Web-ready: H.264 + yuv420p plays in every browser, CRF 23 keeps a
        # 15 s clip at a few MB, +faststart lets a page start playing at once.
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                          quality=None, pixelformat="yuv420p",
                                          macro_block_size=16,
                                          ffmpeg_params=["-crf", "23", "-preset", "medium",
                                                         "-movflags", "+faststart"])
        self._font = _font(24)
        self._small = _font(17)
        self.n_frames = 0

    def add(self, chase_rgb: np.ndarray, cams: dict, lines: list[str],
            prompt: str = "", badge: str = "", badge_color=(220, 40, 40)) -> None:
        canvas = Image.new("RGB", (self.CHASE_W + self.TILE, self.CHASE_H), (16, 16, 16))
        chase = Image.fromarray(chase_rgb)
        if chase.size != (self.CHASE_W, self.CHASE_H):
            chase = chase.resize((self.CHASE_W, self.CHASE_H), Image.BILINEAR)
        canvas.paste(chase, (0, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")

        # Right column: what the dataset stores / the policy sees.
        for i, (name, img) in enumerate(cams.items()):
            tile = img if isinstance(img, Image.Image) else Image.fromarray(img)
            canvas.paste(tile.resize((self.TILE, self.TILE), Image.BILINEAR),
                         (self.CHASE_W, i * self.TILE))
            draw.rectangle([self.CHASE_W, i * self.TILE,
                            self.CHASE_W + self.TILE - 1, i * self.TILE + 24],
                           fill=(0, 0, 0, 150))
            draw.text((self.CHASE_W + 8, i * self.TILE + 3), name,
                      font=self._small, fill=(255, 255, 255))

        # Caption block (top-left) and badge (top-right of the chase view).
        if lines:
            h = 14 + 32 * len(lines)
            draw.rectangle([0, 0, 760, h], fill=(0, 0, 0, 160))
            for i, line in enumerate(lines):
                draw.text((16, 10 + 32 * i), line, font=self._font, fill=(255, 255, 255))
        if badge:
            w = int(draw.textlength(badge, font=self._font)) + 36
            draw.rectangle([self.CHASE_W - w - 16, 14, self.CHASE_W - 16, 54],
                           fill=badge_color + (220,))
            draw.text((self.CHASE_W - w + 2, 20), badge, font=self._font,
                      fill=(255, 255, 255))

        # Prompt along the bottom of the chase view.
        if prompt:
            wrapped = textwrap.wrap(f'Prompt: "{prompt}"', width=96)
            top = self.CHASE_H - 16 - 26 * len(wrapped)
            draw.rectangle([0, top - 10, self.CHASE_W, self.CHASE_H], fill=(0, 0, 0, 160))
            for i, line in enumerate(wrapped):
                draw.text((16, top + 26 * i), line, font=self._small, fill=(255, 230, 140))

        self._last = np.asarray(canvas)
        self._writer.append_data(self._last)
        self.n_frames += 1

    def hold(self, n_frames: int) -> None:
        """Repeat the last frame (a short pause at the end of an episode)."""
        for _ in range(n_frames if self.n_frames else 0):
            self._writer.append_data(self._last)
            self.n_frames += 1

    def close(self) -> None:
        self._writer.close()
