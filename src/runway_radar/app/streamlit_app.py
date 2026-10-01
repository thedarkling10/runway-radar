"""Runway Radar -- the public-facing dashboard.

Reads Phase 3's precomputed outputs directly (seasonal_prevalence.csv,
trend_summary.csv). 

Run with:
    uv run streamlit run src/runway_radar/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from runway_radar.aggregation.visuals import _season_labels
from runway_radar.attributes.color_palette import NAMED_COLORS

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "processed"

TOP_N_DEFAULT = 5
SIGNIFICANCE_THRESHOLD = 0.1  # matches aggregation/trend.py

# One font family for the whole app (serif, inspired by true Vogue articles),
# applied globally (including inside the Plotly charts) rather than just
# that one tab, so the whole thing reads as one consistent piece
FONT_FAMILY = "'Playfair Display', Georgia, serif"


def inject_global_font():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');
        html, body, .stApp, .stApp * {{
            font-family: {FONT_FAMILY} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# For the "color" axis specifically, a label's line should actually look like the label for easier eye detection
COLOR_HEX = {name: "#{:02x}{:02x}{:02x}".format(*rgb) for name, rgb in NAMED_COLORS.items()}


@st.cache_data
def load_data():
    prevalence = pd.read_csv(DATA_DIR / "seasonal_prevalence.csv")
    trend = pd.read_csv(DATA_DIR / "trend_summary.csv")
    colors = pd.read_csv(DATA_DIR / "colors.csv")
    looks = pd.read_csv(REPO_ROOT / "data" / "metadata" / "looks.csv")
    return prevalence, trend, colors, looks


def render_trends_tab(prevalence: pd.DataFrame, trend: pd.DataFrame, season_label: dict[int, str]):
    col1, col2 = st.columns(2)
    axis = col1.selectbox("Attribute axis", sorted(prevalence["axis"].unique()), key="trends_axis")
    line = col2.selectbox("Line", ["mens", "womens"], key="trends_line")

    axis_trend = trend[(trend["axis"] == axis) & (trend["line"] == line)]
    default_labels = (
        axis_trend.reindex(axis_trend["slope"].abs().sort_values(ascending=False).index)
        .head(TOP_N_DEFAULT)["label"]
        .tolist()
    )
    all_labels = sorted(axis_trend["label"].unique())
    selected_labels = st.multiselect(
        "Labels to show", all_labels, default=default_labels, key="trends_labels"
    )

    if not selected_labels:
        st.info("Pick at least one label to see its trend.")
        return

    fig = go.Figure()
    season_indices = sorted(season_label.keys())
    next_index = max(season_indices) + 1

    for label in selected_labels:
        series = prevalence[
            (prevalence["axis"] == axis) & (prevalence["line"] == line) & (prevalence["label"] == label)
        ].sort_values("season_index")

        trace_kwargs = {}
        if axis == "color" and label in COLOR_HEX:
            hex_color = COLOR_HEX[label]
            # A dark outline keeps light colors visible against the chart's white background
            trace_kwargs = dict(
                line=dict(color=hex_color, width=3),
                marker=dict(color=hex_color, size=8, line=dict(width=1, color="rgba(0,0,0,0.35)")),
            )

        fig.add_trace(
            go.Scatter(
                x=series["season_index"],
                y=series["prevalence"],
                mode="lines+markers",
                name=label,
                hovertemplate=f"{label}<br>%{{text}}: %{{y:.0%}}<extra></extra>",
                text=[season_label[i] for i in series["season_index"]],
                **trace_kwargs,
            )
        )

        trend_row = axis_trend[axis_trend["label"] == label]
        if not trend_row.empty and trend_row.iloc[0]["direction"] != "stable":
            last_point = series.iloc[-1]
            projected = trend_row.iloc[0]["projected_next_season"]
            fig.add_trace(
                go.Scatter(
                    x=[last_point["season_index"], next_index],
                    y=[last_point["prevalence"], projected],
                    mode="lines+markers",
                    line=dict(dash="dot", color="gray"),
                    marker=dict(symbol="diamond", size=9),
                    name=f"{label} (projected)",
                    hovertemplate=f"{label} -- projected next season: %{{y:.0%}}<extra></extra>",
                    showlegend=False,
                )
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=season_indices + [next_index],
        ticktext=[season_label[i] for i in season_indices] + ["next?"],
        tickangle=45,
    )
    fig.update_yaxes(tickformat=".0%", title="prevalence")
    fig.update_layout(
        height=520,
        legend=dict(
            orientation="h",
            y=1.08,
            yanchor="bottom",
            x=0.5,
            xanchor="center",
            entrywidth=140,
            entrywidthmode="pixels",
        ),
        margin=dict(t=60),
        font=dict(family=FONT_FAMILY),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Dotted segments with diamond markers are **trend extrapolation**, not a validated "
        "forecast -- only shown for labels whose trend is statistically distinguishable from "
        f"flat (p < {SIGNIFICANCE_THRESHOLD}) across this ~16-season series."
    )


def render_movers_tab(trend: pd.DataFrame):
    col1, col2, col3 = st.columns([1, 1, 1.6])
    line_filter = col1.selectbox("Line", ["both", "mens", "womens"], key="movers_line")
    axis_filter = col2.selectbox("Axis", ["all"] + sorted(trend["axis"].unique()), key="movers_axis")
    significant_only = col3.checkbox(
        f"Only statistically significant (p < {SIGNIFICANCE_THRESHOLD})", value=False, key="movers_sig_only"
    )

    subset = trend.copy()
    if line_filter != "both":
        subset = subset[subset["line"] == line_filter]
    if axis_filter != "all":
        subset = subset[subset["axis"] == axis_filter]
    if significant_only:
        subset = subset[subset["direction"] != "stable"]

    if subset.empty:
        st.info("No trends match this filter.")
        return

    subset = subset.copy()
    subset["display_label"] = subset["axis"] + ": " + subset["label"] + " (" + subset["line"] + ")"
    subset["significant"] = subset["direction"] != "stable"

    rising = subset[subset["slope"] > 0].nlargest(10, "slope")
    falling = subset[subset["slope"] < 0].nsmallest(10, "slope")
    movers = pd.concat([rising, falling]).sort_values("slope")

    if movers.empty:
        st.info("No trends match this filter.")
        return

    # Four separate traces purely so Plotly draws a proper legend -- solid
    # colors for trends that cleared the significance bar, muted/pastel
    # versions for ones that didn't. Diversity you can see, without
    # pretending every bar carries the same level of confidence.
    hover = (
        "%{y}<br>slope: %{x:.4f}<br>p-value: %{customdata[0]:.3f}"
        "<br>mean prevalence: %{customdata[1]:.0%}<extra></extra>"
    )
    trace_specs = [
        ("Rising (confident)", rising[rising["significant"]], "#27ae60"),
        ("Rising (tentative)", rising[~rising["significant"]], "#a9d9bd"),
        ("Falling (confident)", falling[falling["significant"]], "#c0392b"),
        ("Falling (tentative)", falling[~falling["significant"]], "#e8b4ad"),
    ]

    fig = go.Figure()
    for name, data, color in trace_specs:
        if data.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=data["slope"],
                y=data["display_label"],
                orientation="h",
                name=name,
                marker_color=color,
                customdata=data[["p_value", "mean_prevalence"]],
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        height=max(350, 35 * len(movers)) + 60,
        xaxis_title="trend slope (prevalence change per season)",
        yaxis=dict(categoryorder="array", categoryarray=movers["display_label"].tolist()),
        margin=dict(l=10, r=10, t=60),
        font=dict(family=FONT_FAMILY),
        legend=dict(
            orientation="h",
            y=1.0,
            yanchor="bottom",
            x=0.5,
            xanchor="center",
            entrywidth=170,
            entrywidthmode="pixels",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Top 10 rising + top 10 falling across the current filter, shown regardless of "
        "significance so every axis/line has something to look at. Muted colors are "
        f"tentative (didn't clear p < {SIGNIFICANCE_THRESHOLD}) -- real movement in this "
        "data, but not confidently distinguishable from season-to-season noise. Check the "
        '"Only statistically significant" box above to see confident trends only.'
    )


def render_gallery_tab(prevalence: pd.DataFrame, colors: pd.DataFrame, looks: pd.DataFrame, season_label: dict[int, str]):
    col1, col2 = st.columns(2)
    season_index = col1.selectbox(
        "Season", sorted(season_label.keys()), format_func=lambda i: season_label[i], key="gallery_season"
    )
    line = col2.selectbox("Line", ["mens", "womens"], key="gallery_line")

    season_looks = looks[(looks["season_index"] == season_index) & (looks["line"] == line)]
    season_colors = colors[colors["look_id"].isin(season_looks["look_id"])]

    st.subheader(f"{season_label[season_index]} -- {line} ({len(season_looks)} looks)")

    st.markdown("**Dominant colors this season:**")
    color_counts = season_colors["dominant_color_name"].value_counts().head(6)
    swatch_cols = st.columns(len(color_counts) or 1)
    for col, (name, count) in zip(swatch_cols, color_counts.items()):
        hex_sample = season_colors[season_colors["dominant_color_name"] == name]["dominant_hex"].iloc[0]
        col.markdown(
            f'<div style="background-color:{hex_sample}; height:50px; border-radius:6px; '
            f'border:1px solid #444;"></div><p style="text-align:center; font-size:0.8em;">'
            f"{name}<br>{count}/{len(season_looks)}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("**Top attribute per axis:**")
    season_prev = prevalence[(prevalence["season_index"] == season_index) & (prevalence["line"] == line)]
    for axis in sorted(season_prev["axis"].unique()):
        if axis == "color":
            continue
        top = season_prev[season_prev["axis"] == axis].nlargest(1, "prevalence").iloc[0]
        st.write(f"- **{axis.replace('_', ' ')}**: {top['label']} ({top['prevalence']:.0%})")


def render_tribute_tab():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

.tribute-masthead {
    font-family: 'Playfair Display', serif;
    text-align: center;
    font-size: 3.2em;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0;
}
.tribute-subtitle {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.05em;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin-top: 0.3em;
}
.tribute-rule {
    border: none;
    border-top: 2px solid #1a1a1a;
    width: 55%;
    margin: 1.3em auto;
}
.tribute-body {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05em;
    line-height: 1.75;
    column-count: 2;
    column-gap: 3em;
    text-align: justify;
    max-width: 920px;
    margin: 0 auto;
}
.tribute-body p { margin-bottom: 1em; }
.tribute-body p:first-of-type::first-letter {
    font-size: 3.6em;
    font-weight: 700;
    float: left;
    line-height: 0.8;
    padding-right: 0.09em;
    padding-top: 0.05em;
}
.tribute-pullquote {
    column-span: all;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.45em;
    text-align: center;
    border-top: 1px solid #ccc;
    border-bottom: 1px solid #ccc;
    padding: 0.7em 2em;
    margin: 0.4em 0 1em 0;
    color: #222;
}
</style>

<div class="tribute-masthead">RUNWAY RADAR</div>
<div class="tribute-subtitle">A Tribute &mdash; Eight Years of Alessandro Michele at Gucci</div>
<hr class="tribute-rule" />

<div class="tribute-body">

<p>In January 2015, Gucci handed its creative reins to a designer few
outside the industry had heard of. Alessandro Michele had spent over a
decade inside the house's ateliers, quietly shaping accessories and
ready-to-wear for other names on the label. He was given his first
runway show with days, not months, to prepare. What emerged was not a
continuation of Gucci's sleek, seductive inheritance from the Tom Ford
era &mdash; it was a rupture.</p>

<p>Michele's Gucci was maximalist where the house had been minimal,
romantic where it had been cool, androgynous where it had insisted on
binary glamour. Pussy-bow blouses sat beside oversized tailoring;
vintage florals collided with logo-emblazoned leather; a single look
might reference the 1970s, Renaissance portraiture, and a thrifted
cardigan all at once. Critics reached for the word "eclectic" so often
it became shorthand for the whole era &mdash; but underneath the
maximalism was a genuinely coherent worldview: that fashion didn't have
to choose between irony and sincerity, masculine and feminine, high and
low. It could hold all of it at once, unapologetically.</p>

<div class="tribute-pullquote">
"Fashion didn't have to choose between irony and sincerity,<br/>
masculine and feminine, high and low."
</div>

<p>The impact was immediate and, by luxury-industry standards, enormous.
Gucci's sales grew dramatically in Michele's first few years, and the
look he built &mdash; now widely shorthanded as "geek-chic" or
"grandmillennial" &mdash; rippled out across the industry, reshaping
what other houses believed their customers wanted. He also treated the
runway show itself as a canvas, not just a backdrop: a seven-part film
with Gus Van Sant when the pandemic made a conventional show
impossible; a meditation on the house's history for its 100th
anniversary; a runway staged on Hollywood Boulevard. When the usual
format stopped being possible, he didn't pause &mdash; he changed the
medium entirely.</p>

<p>Michele's final collection for Gucci, shown in September 2022, cast
only identical twins on the runway &mdash; a quiet, strange, deeply
personal closing statement from a designer who had spent eight years
insisting that identity itself could be plural. He departed that
November.</p>

<p>This project exists because that eight-year arc is genuinely worth
studying closely &mdash; not just admired, but actually looked at,
season by season, pattern by pattern, color by color. Runway Radar is
an attempt to do exactly that with data: to take 486 looks across all
sixteen of his collections and ask, quantitatively, what a creative era
this singular actually looked like from the inside. It is, in its own
small way, a form of close reading &mdash; and an homage to eight years
that changed what a fashion house could be.</p>

</div>
""",
        unsafe_allow_html=True,
    )


def render_methodology_tab():
    st.markdown(
        """
### How this works

CLIP zero-shot classification (no fine-tuning) detects 5 visual attributes
per look -- silhouette, neckline/collar, fabric/texture, pattern,
embellishment -- plus dominant color via k-means clustering, across 486
hand-curated looks spanning all 16 of Alessandro Michele's Gucci seasons.

### Validated, not just run

Every prediction was checked against independent blind hand-labeling on a
48-look sample. Final results, after fixing two real bugs found during
validation (a vocabulary coverage gap, and a default-label bias corrected
via per-label calibration):

| Axis | Agreement | vs. random chance |
|---|---|---|
| pattern | 60.9% | 4.9x |
| fabric_texture | 40.4% | 3.2x |
| silhouette | 37.5% | 3.0x |
| neckline_collar | 29.8% | 2.1x |
| embellishment | 27.9% | 2.0x |

All five axes clearly beat random guessing. Full methodology, including
the bugs found and fixed along the way, is documented in the repository.

### Known limitations

- These are per-image accuracy numbers, not a direct measure of
  season-level trend accuracy. Aggregating many looks per season
  tends to average out non-systematic noise, but this isn't separately
  validated.
- Two seasons (Spring 2021, Fall 2021) were presented as short films,
  not traditional runway shows.
- Trend lines are fit on ~16 points per series. Projections shown in the
  Trends tab are simple extrapolation from recent momentum, explicitly
  not a validated forecasting model.
- Source images are never redistributed here, only derived statistics
  and attribute labels, consistent with the project's sourcing policy.
"""
    )


def main():
    st.set_page_config(page_title="Runway Radar", layout="wide")
    inject_global_font()
    st.title("Runway Radar")
    st.caption(
        "Gucci under Alessandro Michele (2015-2023) -- trend detection from runway "
        "attribute classification, validated against independent human judgment."
    )

    prevalence, trend, colors, looks = load_data()
    season_label = _season_labels(prevalence)

    tab_tribute, tab_trends, tab_movers, tab_gallery, tab_methodology = st.tabs(
        ["Tribute", "Trends", "Biggest Movers", "Gallery", "Methodology"]
    )
    with tab_tribute:
        render_tribute_tab()
    with tab_trends:
        render_trends_tab(prevalence, trend, season_label)
    with tab_movers:
        render_movers_tab(trend)
    with tab_gallery:
        render_gallery_tab(prevalence, colors, looks, season_label)
    with tab_methodology:
        render_methodology_tab()


if __name__ == "__main__":
    main()
