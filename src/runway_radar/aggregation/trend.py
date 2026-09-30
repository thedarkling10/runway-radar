"""Fit a simple linear trend to each (line, axis, label) prevalence series.

Deliberately simple: 16 seasons is a short time series, and anything more
sophisticated than a linear fit would be reading precision into the data
that isn't really there. See docs/problems_and_fixes.md-style reasoning --
matching the method's complexity to what the data can actually support is
the point, not a limitation to apologize for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
PREVALENCE_CSV = REPO_ROOT / "data" / "processed" / "seasonal_prevalence.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "trend_summary.csv"

# A relaxed threshold, not the conventional 0.05 -- with only 16 points per
# series, requiring p<0.05 would label almost everything "stable" even when
# there's a real, visible trend. 0.1 is a deliberate judgment call to keep
# from being falsely conservative at this sample size; it should be read as
# "probably a real trend," not a strict significance claim.
SIGNIFICANCE_THRESHOLD = 0.1


def fit_trend(season_index: pd.Series, prevalence: pd.Series) -> dict:
    result = stats.linregress(season_index, prevalence)
    is_significant = result.pvalue < SIGNIFICANCE_THRESHOLD
    if not is_significant:
        direction = "stable"
    else:
        direction = "rising" if result.slope > 0 else "falling"

    next_season_index = season_index.max() + 1
    projected_next = result.slope * next_season_index + result.intercept
    projected_next = min(1.0, max(0.0, projected_next))  # prevalence can't leave [0, 1]

    return {
        "slope": result.slope,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "direction": direction,
        "projected_next_season": projected_next,
    }


def build_trend_summary() -> pd.DataFrame:
    prevalence = pd.read_csv(PREVALENCE_CSV)

    rows = []
    for (line, axis, label), group in prevalence.groupby(["line", "axis", "label"]):
        group = group.sort_values("season_index")
        trend = fit_trend(group["season_index"], group["prevalence"])
        rows.append(
            {
                "line": line,
                "axis": axis,
                "label": label,
                "mean_prevalence": group["prevalence"].mean(),
                **trend,
            }
        )

    return pd.DataFrame(rows).sort_values(["axis", "line", "slope"], ascending=[True, True, False])


def run():
    summary = build_trend_summary()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(summary)} trend lines to {OUTPUT_CSV}")

    n_rising = (summary["direction"] == "rising").sum()
    n_falling = (summary["direction"] == "falling").sum()
    n_stable = (summary["direction"] == "stable").sum()
    print(f"{n_rising} rising, {n_falling} falling, {n_stable} stable (at p<{SIGNIFICANCE_THRESHOLD})")


if __name__ == "__main__":
    run()
