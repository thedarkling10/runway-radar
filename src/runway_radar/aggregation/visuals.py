"""Static seaborn charts summarizing the trend analysis.

Distinct from the interactive Streamlit+Plotly dashboard (Phase 4) --
these are polished, standalone PNGs meant for a README, a writeup, or a
slide, not for clicking around in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
PREVALENCE_CSV = REPO_ROOT / "data" / "processed" / "seasonal_prevalence.csv"
TREND_CSV = REPO_ROOT / "data" / "processed" / "trend_summary.csv"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

TOP_N_PER_AXIS = 5
TOP_N_MOVERS = 10

sns.set_theme(style="whitegrid", palette="deep", font_scale=0.95)


def _season_labels(prevalence: pd.DataFrame) -> dict[int, str]:
    seasons = prevalence[["season_index", "season", "year"]].drop_duplicates()
    seasons["short"] = seasons["season"].str[0] + seasons["year"].astype(str).str[-2:]
    return dict(zip(seasons["season_index"], seasons["short"]))


def _label_color_map(axis: str, trend: pd.DataFrame) -> dict[str, tuple]:
    """One fixed color per label, shared across BOTH subplots (mens and
    womens). Built from every label the axis has -- not just whichever
    happen to be in a given subplot's top-N -- so the same label always
    gets the same color everywhere it appears, and a color never gets
    reused for a different label just because the other one wasn't in
    that particular subplot's top-N this time."""
    all_labels = sorted(trend[trend["axis"] == axis]["label"].unique())
    palette = sns.color_palette("tab10", n_colors=len(all_labels))
    return dict(zip(all_labels, palette))


def plot_axis_trends(axis: str, prevalence: pd.DataFrame, trend: pd.DataFrame, season_label: dict[int, str]):
    axis_trend = trend[trend["axis"] == axis]
    color_map = _label_color_map(axis, trend)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, line in zip(axes, ["mens", "womens"]):
        top_labels = (
            axis_trend[axis_trend["line"] == line]
            .reindex(axis_trend[axis_trend["line"] == line]["slope"].abs().sort_values(ascending=False).index)
            .head(TOP_N_PER_AXIS)["label"]
        )
        subset = prevalence[
            (prevalence["axis"] == axis) & (prevalence["line"] == line) & (prevalence["label"].isin(top_labels))
        ]
        sns.lineplot(
            data=subset,
            x="season_index",
            y="prevalence",
            hue="label",
            palette=color_map,
            marker="o",
            ax=ax,
            linewidth=2,
        )
        ax.set_xticks(sorted(season_label.keys()))
        ax.set_xticklabels([season_label[i] for i in sorted(season_label.keys())], rotation=45, ha="right")
        ax.set_title(f"{axis.replace('_', ' ').title()} -- {line}")
        ax.set_xlabel("")
        ax.set_ylabel("prevalence")
        ax.legend(title=None, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(f"Top {TOP_N_PER_AXIS} moving labels: {axis.replace('_', ' ')}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = FIGURES_DIR / f"trend_{axis}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_biggest_movers(trend: pd.DataFrame):
    significant = trend[trend["direction"] != "stable"].copy()
    significant["display_label"] = (
        significant["axis"] + ": " + significant["label"] + " (" + significant["line"] + ")"
    )

    rising = significant[significant["direction"] == "rising"].nlargest(TOP_N_MOVERS, "slope")
    falling = significant[significant["direction"] == "falling"].nsmallest(TOP_N_MOVERS, "slope")
    movers = pd.concat([rising, falling]).sort_values("slope")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(movers))))
    colors = ["#c0392b" if s < 0 else "#27ae60" for s in movers["slope"]]
    ax.barh(movers["display_label"], movers["slope"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("trend slope (prevalence change per season)")
    ax.set_title(
        f"Biggest movers across all axes (p < {0.1:.1g})", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / "biggest_movers.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run():
    prevalence = pd.read_csv(PREVALENCE_CSV)
    trend = pd.read_csv(TREND_CSV)
    season_label = _season_labels(prevalence)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for axis in sorted(prevalence["axis"].unique()):
        written.append(plot_axis_trends(axis, prevalence, trend, season_label))
    written.append(plot_biggest_movers(trend))

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    run()
