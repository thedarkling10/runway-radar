"""Streamlit app for hand-labeling a validation sample of looks.

Deliberately shows nothing from attributes.csv -- CLIP's predictions stay
hidden while you label, so your answers are an independent check rather
than something anchored to what the model already guessed.

Run with:
    uv run streamlit run src/runway_radar/validation/labeling_app.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pillow_avif  # noqa: F401 -- registers the AVIF codec with Pillow
import streamlit as st
from PIL import Image

from runway_radar.acquisition.register import RAW_DIR, read_metadata, season_slug
from runway_radar.attributes.vocab import ATTRIBUTE_AXES

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CSV = REPO_ROOT / "data" / "processed" / "validation_sample.csv"
HAND_LABELS_CSV = REPO_ROOT / "data" / "processed" / "hand_labels.csv"

SAMPLES_PER_GROUP = 3  # per (season, year, line) -- ~16 groups -> ~48 looks
RANDOM_SEED = 42  # fixed, so the sample is the same every time the app runs

AXES = list(ATTRIBUTE_AXES.keys())


def build_validation_sample() -> list[dict]:
    """Pick a fixed, reproducible sample stratified across every season/line
    group, so all 8 seasons and both lines are represented -- not just
    whichever happens to be biggest. Saved once and reused on later runs."""
    if SAMPLE_CSV.exists():
        with SAMPLE_CSV.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows = read_metadata()
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["season"], row["year"], row["line"])
        groups.setdefault(key, []).append(row)

    rng = random.Random(RANDOM_SEED)
    sample = []
    for key in sorted(groups):
        group_rows = groups[key]
        sample.extend(rng.sample(group_rows, min(SAMPLES_PER_GROUP, len(group_rows))))

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(sample[0].keys())
    with SAMPLE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)
    return sample


def load_hand_labels() -> dict[tuple[str, str], str]:
    if not HAND_LABELS_CSV.exists():
        return {}
    with HAND_LABELS_CSV.open(newline="", encoding="utf-8") as f:
        return {(r["look_id"], r["axis"]): r["hand_label"] for r in csv.DictReader(f)}


def save_hand_label(look_id: str, axis: str, label: str) -> None:
    existing = load_hand_labels()
    existing[(look_id, axis)] = label
    HAND_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HAND_LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["look_id", "axis", "hand_label"])
        writer.writeheader()
        for (lid, ax), lbl in sorted(existing.items()):
            writer.writerow({"look_id": lid, "axis": ax, "hand_label": lbl})


def image_path_for(row: dict) -> Path:
    return RAW_DIR / season_slug(row["season"], row["year"]) / row["line"] / row["image_filename"]


def is_fully_labeled(row: dict, hand_labels: dict) -> bool:
    return all((row["look_id"], axis) in hand_labels for axis in AXES)


def main():
    st.set_page_config(page_title="Runway Radar -- Hand Labeling", layout="centered")
    st.title("Hand-labeling: validate CLIP's predictions")
    st.caption(
        "Pick one label per axis, or 'not_sure' if genuinely ambiguous. "
    )

    sample = build_validation_sample()
    hand_labels = load_hand_labels()

    if "idx" not in st.session_state:
        remaining = [i for i, row in enumerate(sample) if not is_fully_labeled(row, hand_labels)]
        st.session_state.idx = remaining[0] if remaining else 0

    idx = st.session_state.idx
    row = sample[idx]

    done_count = sum(1 for r in sample if is_fully_labeled(r, hand_labels))
    st.progress(done_count / len(sample))
    st.write(f"**{done_count} / {len(sample)} looks fully labeled** -- viewing look {idx + 1} of {len(sample)}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", disabled=idx == 0, use_container_width=True):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with col2:
        if st.button("Next", disabled=idx == len(sample) - 1, use_container_width=True):
            st.session_state.idx = min(len(sample) - 1, idx + 1)
            st.rerun()

    st.subheader(f"{row['house']} -- {row['season']} {row['year']} ({row['line']})")

    image = Image.open(image_path_for(row)).convert("RGB")
    st.image(image, width=350)

    st.divider()
    for axis in AXES:
        options = list(ATTRIBUTE_AXES[axis].keys()) + ["not_sure"]
        current = hand_labels.get((row["look_id"], axis))
        default_index = options.index(current) if current in options else None

        with st.expander(f"What do these {axis.replace('_', ' ')} labels mean?"):
            for label, phrasings in ATTRIBUTE_AXES[axis].items():
                st.markdown(f"- **{label}**: {phrasings[0]}")
            st.markdown("- **not_sure**: genuinely ambiguous, ignored in the agreement report")

        choice = st.selectbox(
            axis.replace("_", " ").title(),
            options,
            index=default_index,
            key=f"{row['look_id']}_{axis}",
            placeholder="Pick one...",
        )
        if choice is not None and choice != current:
            save_hand_label(row["look_id"], axis, choice)


if __name__ == "__main__":
    main()
