"""Dominant color extraction via k-means pixel clustering.

CLIP is weak at precise color naming, so color is handled separately here
rather than folded into the CLIP attribute vocabulary (see vocab.py).

Caveat worth keeping in mind: "dominant color" here means dominant color
of the WHOLE PHOTO -- runway backdrop, floor, and lighting included, not
just the garment. For heavily-cropped or plainly-lit shots this tracks
the garment reasonably well; for others it can pick up more background
than outfit. Treated as a known limitation, not silently hidden.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pillow_avif  # noqa: F401 -- registers the AVIF codec with Pillow
from PIL import Image
from sklearn.cluster import KMeans

from runway_radar.acquisition.register import RAW_DIR, read_metadata, season_slug
from runway_radar.attributes.image_utils import crop_letterbox

# Downscale before clustering -- k-means cost scales with pixel count, and
# a runway photo's color palette doesn't need full resolution to extract
CLUSTER_IMAGE_SIZE = (100, 150)

OUTPUT_CSV = Path(__file__).resolve().parents[3] / "data" / "processed" / "colors.csv"

# A small, fixed palette of common color names -- raw hex codes from k-means
# vary continuously and would almost never repeat exactly across images, so
# they can't be aggregated into a trend on their own. Snapping every image's
# dominant color to the nearest of these turns color into a categorical
# "axis" with a fixed vocabulary, exactly like the CLIP attributes.
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "gray": (128, 128, 128),
    "beige_tan": (210, 180, 140),
    "brown": (101, 67, 33),
    "red": (200, 30, 30),
    "burgundy": (128, 0, 32),
    "orange": (230, 126, 34),
    "gold_yellow": (212, 175, 55),
    "olive_khaki": (128, 128, 0),
    "green": (34, 139, 34),
    "teal": (0, 128, 128),
    "blue": (30, 90, 200),
    "navy": (0, 0, 80),
    "purple": (128, 0, 128),
    "pink": (230, 130, 170),
}


def nearest_named_color(rgb: tuple[int, int, int]) -> str:
    """Snap an arbitrary RGB triple to the closest name in NAMED_COLORS by
    Euclidean distance in RGB space -- simple nearest-neighbor, no learned
    model involved."""
    r, g, b = rgb
    best_name, best_dist = None, float("inf")
    for name, (nr, ng, nb) in NAMED_COLORS.items():
        dist = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


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


def run_color_extraction():
    rows = read_metadata()
    print(f"Extracting dominant color for {len(rows)} looks...")

    output_rows = []
    for i, row in enumerate(rows, start=1):
        path = RAW_DIR / season_slug(row["season"], row["year"]) / row["line"] / row["image_filename"]
        image = crop_letterbox(Image.open(path))
        top_color = extract_dominant_colors(image, n_colors=5)[0]
        output_rows.append(
            {
                "look_id": row["look_id"],
                "dominant_hex": top_color["hex"],
                "dominant_color_name": nearest_named_color(top_color["rgb"]),
                "proportion": top_color["proportion"],
            }
        )
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["look_id", "dominant_hex", "dominant_color_name", "proportion"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_color_extraction()
