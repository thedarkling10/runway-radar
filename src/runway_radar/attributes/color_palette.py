"""Dominant color extraction via k-means pixel clustering.

CLIP is weak at precise color naming, so color is handled separately here
rather than folded into the CLIP attribute vocabulary (see vocab.py).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Downscale before clustering -- k-means cost scales with pixel count, and
# a runway photo's color palette doesn't need full resolution to extract
CLUSTER_IMAGE_SIZE = (100, 150)


def extract_dominant_colors(image: Image.Image, n_colors: int = 5) -> list[dict]:
    """Return the n_colors most prevalent colors in the image, sorted by
    prevalence descending. Each entry has hex, rgb, and proportion (0-1)."""
    small = image.convert("RGB").resize(CLUSTER_IMAGE_SIZE)
    pixels = np.array(small).reshape(-1, 3)

    n_colors = min(n_colors, len(np.unique(pixels, axis=0)))
    kmeans = KMeans(n_clusters=n_colors, n_init=4, random_state=0)
    labels = kmeans.fit_predict(pixels)

    counts = np.bincount(labels)
    order = np.argsort(-counts)

    results = []
    for cluster_idx in order:
        rgb = tuple(int(c) for c in kmeans.cluster_centers_[cluster_idx])
        proportion = float(counts[cluster_idx]) / len(pixels)
        results.append(
            {
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "rgb": rgb,
                "proportion": round(proportion, 4),
            }
        )
    return results
