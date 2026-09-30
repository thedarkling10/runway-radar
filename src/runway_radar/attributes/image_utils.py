"""Shared image-cleanup helpers used by both the CLIP and color pipelines.

Split out because the same fix (removing letterbox bars) matters for two
different downstream consumers -- color extraction and CLIP classification
-- and duplicating it would risk them drifting out of sync.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DARK_THRESHOLD = 15  # mean RGB below this counts as "solid black" for a bar


def crop_letterbox(image: Image.Image, threshold: int = DARK_THRESHOLD) -> Image.Image:
    """Crop away solid near-black bars from any edge (left/right/top/bottom).

    Some frame-grabbed seasons (see docs/problems_and_fixes.md) have real
    letterboxing baked into the screenshot -- a portrait crop of a
    different-aspect-ratio video source, left over as solid black canvas
    rather than garment content. Left uncropped, this both skews color
    extraction toward "black" and dilutes CLIP's view of the actual look
    after padding+resizing. Only trims uniformly dark columns/rows
    starting from each edge; stops at the first column/row that isn't
    dark, so it never eats into real (possibly dark-toned) content.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    row_means = arr.mean(axis=(1, 2))  # brightness per row
    col_means = arr.mean(axis=(0, 2))  # brightness per column

    def first_non_dark(means: np.ndarray) -> int:
        idx = np.argmax(means > threshold)
        return int(idx) if means[idx] > threshold else 0

    def last_non_dark(means: np.ndarray) -> int:
        reversed_idx = first_non_dark(means[::-1])
        return len(means) - reversed_idx

    top = first_non_dark(row_means)
    bottom = last_non_dark(row_means)
    left = first_non_dark(col_means)
    right = last_non_dark(col_means)

    # Guard against a pathological all-dark image cropping to nothing
    if bottom - top < 10 or right - left < 10:
        return image

    return image.crop((left, top, right, bottom))
