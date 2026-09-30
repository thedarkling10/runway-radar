"""Compare hand labels against CLIP's predictions and report agreement.

"not_sure" hand labels are excluded -- an ambiguous look shouldn't count
as a strike against the model either way.

Run with:
    uv run python -m runway_radar.validation.agreement_report
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HAND_LABELS_CSV = REPO_ROOT / "data" / "processed" / "hand_labels.csv"
ATTRIBUTES_CSV = REPO_ROOT / "data" / "processed" / "attributes.csv"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    hand_labels = {(r["look_id"], r["axis"]): r["hand_label"] for r in load_csv(HAND_LABELS_CSV)}
    predictions = {(r["look_id"], r["axis"]): r["predicted_label"] for r in load_csv(ATTRIBUTES_CSV)}

    if not hand_labels:
        print("No hand labels found yet -- run the labeling app first:")
        print("  uv run streamlit run src/runway_radar/validation/labeling_app.py")
        return

    per_axis_correct: dict[str, int] = {}
    per_axis_total: dict[str, int] = {}
    mismatches: list[tuple] = []
    overall_correct = 0
    overall_total = 0

    for (look_id, axis), hand_label in hand_labels.items():
        if hand_label == "not_sure":
            continue
        prediction = predictions.get((look_id, axis))
        if prediction is None:
            continue

        per_axis_total[axis] = per_axis_total.get(axis, 0) + 1
        overall_total += 1
        if prediction == hand_label:
            per_axis_correct[axis] = per_axis_correct.get(axis, 0) + 1
            overall_correct += 1
        else:
            mismatches.append((look_id, axis, hand_label, prediction))

    print(f"{'Axis':<20} {'Agreement':>12} {'N':>6}")
    print("-" * 40)
    for axis in sorted(per_axis_total):
        correct = per_axis_correct.get(axis, 0)
        total = per_axis_total[axis]
        print(f"{axis:<20} {correct / total:>11.1%} {total:>6}")
    print("-" * 40)
    if overall_total:
        print(f"{'OVERALL':<20} {overall_correct / overall_total:>11.1%} {overall_total:>6}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches (look_id, axis, your_label, clip_label):")
        for m in mismatches:
            print(f"  {m[0]:<35} {m[1]:<18} you={m[2]:<22} clip={m[3]}")


if __name__ == "__main__":
    main()
