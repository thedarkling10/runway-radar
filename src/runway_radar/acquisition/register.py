"""Helpers for registering manually-collected runway look images.

Images are collected by hand from Gucci's official show archive/channels
(see README's "Legal & sourcing notes") and dropped into
``data/raw/{season_slug}/``. This module just keeps the metadata CSV in
sync with what's actually on disk -- it does not download or scrape
anything.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
METADATA_CSV = REPO_ROOT / "data" / "metadata" / "looks.csv"

FIELDNAMES = [
    "look_id",
    "house",
    "season",
    "year",
    "season_index",
    "line",
    "collection_name",
    "image_filename",
    "source_url",
    "access_date",
    "notes",
]


@dataclass
class LookRecord:
    look_id: str
    house: str
    season: str
    year: int
    season_index: int
    line: str  # "mens" or "womens" -- Gucci runs these as separate shows
    collection_name: str
    image_filename: str
    source_url: str
    access_date: str
    notes: str = ""


def read_metadata() -> list[dict]:
    if not METADATA_CSV.exists():
        return []
    with METADATA_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_metadata(rows: list[dict]) -> None:
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def add_look(record: LookRecord) -> None:
    """Append a single look record, skipping if look_id already exists."""
    rows = read_metadata()
    if any(r["look_id"] == record.look_id for r in rows):
        raise ValueError(f"look_id {record.look_id!r} already registered")
    rows.append(asdict(record))
    write_metadata(rows)


def season_slug(season: str, year: str | int) -> str:
    return f"{season}_{year}".replace(" ", "_").lower()


def season_dir(season: str, year: str | int, line: str) -> Path:
    d = RAW_DIR / season_slug(season, year) / line
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_consistency() -> list[str]:
    """Return a list of human-readable problems: missing files, orphaned
    files not registered in the metadata, and duplicate look_ids."""
    problems: list[str] = []
    rows = read_metadata()

    seen_ids = set()
    registered_paths = set()
    for row in rows:
        if row["look_id"] in seen_ids:
            problems.append(f"duplicate look_id: {row['look_id']}")
        seen_ids.add(row["look_id"])

        rel_path = Path(season_slug(row["season"], row["year"])) / row["line"] / row["image_filename"]
        registered_paths.add(rel_path)
        if not (RAW_DIR / rel_path).exists():
            problems.append(
                f"look_id {row['look_id']}: expected image not found at {RAW_DIR / rel_path}"
            )

    if RAW_DIR.exists():
        for img_path in RAW_DIR.rglob("*"):
            if img_path.is_dir() or img_path.name == ".gitkeep":
                continue
            rel_path = img_path.relative_to(RAW_DIR)
            if rel_path not in registered_paths:
                problems.append(f"orphaned image not in metadata: {img_path}")

    return problems


if __name__ == "__main__":
    issues = check_consistency()
    if not issues:
        print(f"OK - {len(read_metadata())} looks registered, no inconsistencies found.")
    else:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
