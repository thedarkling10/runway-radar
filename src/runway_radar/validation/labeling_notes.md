# Attribute extraction: validation methodology and results

## Method

CLIP zero-shot classification (`clip_extractor.py`) was validated against
independent human judgment before being trusted for downstream trend
analysis. A stratified random sample of 48 looks (3 from each of the 16
season/line combinations available at the time) was hand-labeled blind —
through a purpose-built Streamlit tool (`validation/labeling_app.py`)
that never displays CLIP's predictions, so the human labels are an
independent check rather than something anchored to the model's output.
Labelers could mark a look `not_sure` when genuinely ambiguous; those
answers are excluded from scoring rather than counted as errors.

Agreement is computed per attribute axis as
`(matches) / (comparisons with a definite hand label)`, via
`agreement_report.py`.

## Initial result and diagnosis

The first validation pass (215 looks, 8 seasons) showed 26.5% overall
agreement, with real structure once broken down:

1. **Vocabulary gaps.** Several looks had no correct option available on
   one or more axes — no "solid/no pattern," no eyewear-as-embellishment,
   no plain shirt collar, no silk or cotton. Fixed by adding 7 labels
   across the 5 axes (see `vocab.py`).

2. **A default-label bias (the "hubness problem").** Certain labels —
   `applique_patches`, `brocade_jacquard`, `seventies_flared`,
   `turtleneck` — were winning CLIP's comparison on a large fraction of
   images regardless of actual content, because their text embeddings sit
   generically close to almost any image in this stylistically narrow,
   visually "maximalist" dataset. Fixed with a calibration step: each
   label's average similarity across the whole dataset is subtracted
   before picking a winner, correcting for this per-label baseline pull
   rather than comparing raw similarity directly. Full technical
   explanation, with worked numeric examples, in
   `docs/problems_and_fixes.md` and `docs/clip_pipeline_roadmap.md`
   (both local, not part of this repo's tracked history).

## Final results (486 looks, all 16 seasons, post-fix, revisit complete)

All 48 hand-labeled looks were revisited after the vocabulary expansion
so every comparison below reflects a genuine, considered human judgment
against the *current* label set — no confound adjustment needed.

| Axis | Agreement | N | Labels used | vs. random chance |
|---|---|---|---|---|
| pattern | 60.9% | 46 | 8/8 | 4.9x |
| fabric_texture | 40.4% | 47 | 8/8 | 3.2x |
| silhouette | 37.5% | 48 | 8/8 | 3.0x |
| neckline_collar | 29.8% | 47 | 7/7 | 2.1x |
| embellishment | 27.9% | 43 | 7/7 | 2.0x |
| **Overall** | **39.4%** | 231 | — | — |

Random chance is `1/(number of labels on that axis)` — roughly 12.5% for
an 8-label axis, 14.3% for a 7-label axis. **All five axes now clearly
beat chance**, including embellishment, which had been the persistent
weak point through every earlier check (12.2% raw on the first pass,
barely above its 14.3% chance floor). Revisiting the hand labels with
the `oversized_eyewear` option available more than doubled its agreement
(12.2% -> 27.9%) — most of its earlier weakness was a real vocabulary
gap, not the structural "small localized detail" limitation originally
hypothesized below. Worth remembering as a general lesson: a plausible-
sounding explanation for a weak result is still just a hypothesis until
it's actually tested.

## Known limitations

- **Embellishment remains the weakest axis** (27.9%) even though it now
  clearly beats chance — a bow, rhinestones, or a pin are still small,
  localized details relative to CLIP's single whole-image embedding, so
  some residual weakness here relative to the other axes is plausible
  even without a specific vocabulary gap to blame.
- **Human-human agreement was not measured** (no second independent
  labeler). Some of the disagreement above likely reflects genuine,
  inherent ambiguity in the task itself (e.g. "romantic bohemian" vs.
  "layered eclectic" on a borderline look) rather than model error —
  100% agreement would not be a realistic target even between two
  people.
- **Two seasons (Spring 2021, Fall 2021) were not traditional runway
  shows** — they were presented as short films ("Ouverture of Something
  That Never Ended" and "Gucci Aria" respectively). Looks were
  extracted from film stills rather than a runway walk; this may affect
  framing/composition consistency relative to other seasons (see
  per-row `notes` in `data/metadata/looks.csv`).
- **These numbers describe per-image classification accuracy**, not the
  accuracy of season-level trend statistics computed downstream in
  Phase 3. Calibration specifically targeted *systematic* bias (which
  would skew a trend's direction); the remaining disagreement looks more
  like scattered, non-systematic noise, which tends to average out
  better across a season's worth of looks than these per-image numbers
  might suggest — but this is a reasonable expectation, not something
  separately validated.

## What would improve this further

- A second independent human labeler, to establish a real human-human
  agreement ceiling to compare against.
- A targeted embellishment-specific approach (e.g. cropping to a
  detected subject region before re-classifying just that axis) rather
  than relying on the same whole-image embedding used for the other
  four axes — worth revisiting only if embellishment's gap to the other
  axes still matters once real trend statistics are computed in Phase 3.
