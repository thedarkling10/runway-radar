# Runway Radar 🐍✨

A trend-forecasting dashboard that treats one designer's runway history as a
time series: **Gucci under Alessandro Michele (2015–2022)**. It detects
visual attributes (silhouette, embellishment, print, texture, color) in
runway looks season by season, and tracks how those attributes rise, peak,
and fade — in the same spirit as commercial trend-forecasting tools like
Heuritech or Stylumia, built as a solo portfolio project.

> Status: 🚧 early scaffold — see [Roadmap](#roadmap).

## Why this project

Built as a flagship portfolio piece aimed at AI/Data Science roles at the
intersection of machine learning and fashion (luxury houses' internal data
science teams, trend-forecasting vendors, resale/authentication platforms).
It's meant to demonstrate: a real, self-curated dataset; a defensible
zero-shot computer vision pipeline with honest validation; time-series
methods scoped correctly to small-N data; and a polished, demo-able product
around all of it.

## Methodology (summary)

1. **Data**: runway look images for Gucci, Michele era, collected manually
   from Gucci's own official show archive / official channels — not scraped,
   not bulk-automated.
2. **Attribute extraction**: CLIP zero-shot classification against a curated
   prompt vocabulary per attribute axis, plus k-means color palette
   extraction. Validated against a hand-labeled sample (see
   `src/runway_radar/validation/`).
3. **Trend aggregation**: per-season attribute prevalence, summarized with
   simple, interpretable trend regression — explicitly *not* deep-learning
   time-series forecasting, since there are only ~14 seasons of data. Any
   "next season" projection is framed as trend extrapolation, not prediction.
4. **Dashboard**: Streamlit + Plotly, deployed on Streamlit Community Cloud.

Full methodology and validation results will be filled in as each phase
lands — see `src/runway_radar/validation/labeling_notes.md`.

## Legal & sourcing notes

- **Primary source**: Gucci's own official YouTube channel (`@gucci`) —
  individual looks are manually screenshotted while watching the full show
  video at normal playback. No automated bulk scraping, no downloading or
  ripping of the video files themselves.
- **Fallback source**: when the official channel's video quality is
  genuinely too low to resolve garment detail (this happened for the Fall
  2015 menswear show), a small number of individual stills are instead
  manually screenshotted from third-party editorial coverage (e.g. WWD
  runway galleries). These are used strictly as **private, local-only
  reference material** — never committed to this repository, never
  redistributed, and not treated as the project's default sourcing method.
  Every image's actual origin (YouTube vs. editorial gallery) is recorded
  per-row in `data/metadata/looks.csv` for full provenance.
- Raw image files live only in `data/raw/` (git-ignored) and are **never**
  committed or redistributed under either sourcing method. This repo ships
  code, extracted attribute metadata, and aggregated statistics — not the
  source photography.
- The dashboard displays derived visualizations (attribute tags, color
  swatches, charts) rather than raw copyrighted images wherever practical.
- This is a personal, non-commercial, educational/portfolio project.

## Project layout

```
data/
  raw/         # collected images (git-ignored)
  metadata/    # season/year/source CSV (committed)
  processed/   # extracted attributes + aggregated time series (committed)
src/runway_radar/
  acquisition/ # helpers to register manually-collected images
  attributes/  # CLIP zero-shot extractor, color palette, prompt vocab
  aggregation/ # per-season stats, trend regression/smoothing
  validation/  # hand-label set + agreement notes
  app/         # Streamlit dashboard
notebooks/     # exploratory analysis
```

## Setup

```bash
uv sync
uv run streamlit run src/runway_radar/app/streamlit_app.py
```

## Roadmap

- [ ] Phase 0 — validate manual data collection on 2–3 seasons
- [ ] Phase 1 — full image collection + metadata across ~14 seasons
- [ ] Phase 2 — CLIP zero-shot attribute pipeline + hand-label validation
- [ ] Phase 3 — trend aggregation + regression/smoothing
- [ ] Phase 4 — Streamlit dashboard
- [ ] Phase 5 — polish, deploy, writeup
- [ ] Phase 6 (stretch) — CLIP-embedding linear probe vs. zero-shot
