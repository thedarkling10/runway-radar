"""Turn per-image predictions into per-season, per-line prevalence tables.

Color is folded in as a sixth "axis" (see attributes/color_palette.py) so
everything downstream -- trend fitting, plotting -- treats all six axes
identically, with no special-casing for color.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from runway_radar.attributes.color_palette import NAMED_COLORS
from runway_radar.attributes.vocab import ATTRIBUTE_AXES

REPO_ROOT = Path(__file__).resolve().parents[3]
LOOKS_CSV = REPO_ROOT / "data" / "metadata" / "looks.csv"
ATTRIBUTES_CSV = REPO_ROOT / "data" / "processed" / "attributes.csv"
COLORS_CSV = REPO_ROOT / "data" / "processed" / "colors.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "seasonal_prevalence.csv"

# The full, fixed label set per axis -- needed to fill in explicit zeros for
# (season, line, axis, label) combinations where nothing was predicted, so a
# season a label simply never appeared in isn't silently dropped from its
# own trend line instead of counting as a real zero.
ALL_LABELS: dict[str, list[str]] = {axis: list(labels.keys()) for axis, labels in ATTRIBUTE_AXES.items()}
ALL_LABELS["color"] = list(NAMED_COLORS.keys())


def load_predictions_long() -> pd.DataFrame:
    """Combine CLIP attribute predictions and color into one long table:
    one row per (look_id, axis, label)."""
    attrs = pd.read_csv(ATTRIBUTES_CSV)[["look_id", "axis", "predicted_label"]]
    attrs = attrs.rename(columns={"predicted_label": "label"})

    colors = pd.read_csv(COLORS_CSV)[["look_id", "dominant_color_name"]]
    colors = colors.rename(columns={"dominant_color_name": "label"})
    colors["axis"] = "color"

    return pd.concat([attrs, colors[["look_id", "axis", "label"]]], ignore_index=True)


def build_prevalence_table() -> pd.DataFrame:
    looks = pd.read_csv(LOOKS_CSV)[["look_id", "season_index", "season", "year", "line"]]
    predictions = load_predictions_long()

    merged = predictions.merge(looks, on="look_id", how="left")

    # total looks per (season, line) group -- the denominator for prevalence
    group_cols = ["season_index", "season", "year", "line"]
    season_line_groups = looks[group_cols].drop_duplicates()
    n_images = looks.groupby(group_cols)["look_id"].nunique().rename("n_images")

    counts = (
        merged.groupby(group_cols + ["axis", "label"])["look_id"]
        .nunique()
        .rename("count")
        .reset_index()
    )

    # Build the full (season, line) x (axis, label) grid so a label that
    # never appeared in a given season is an explicit 0, not a missing row.
    axis_label_pairs = pd.DataFrame(
        [(axis, label) for axis, labels in ALL_LABELS.items() for label in labels],
        columns=["axis", "label"],
    )
    full_grid = season_line_groups.merge(axis_label_pairs, how="cross")

    result = full_grid.merge(counts, on=group_cols + ["axis", "label"], how="left")
    result["count"] = result["count"].fillna(0).astype(int)
    result = result.merge(n_images, on=group_cols)
    result["prevalence"] = result["count"] / result["n_images"]

    return result.sort_values(group_cols + ["axis", "label"]).reset_index(drop=True)


def run():
    table = build_prevalence_table()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT_CSV}")
    print(f"Covering {table['season_index'].nunique()} seasons x {table['line'].nunique()} lines x {table['axis'].nunique()} axes")


if __name__ == "__main__":
    run()
