#!/usr/bin/env python3
"""Check distribution of raw or prepared procedural-letter datasets.

Examples:

  # Raw nested collection layout
  python scripts/check_letters_dataset_distribution.py --source_dir data/mujoco/letters

  # Prepared world-model dataset from prepare_letters_dataset.py
  python scripts/check_letters_dataset_distribution.py \
    --dataset_dir data/mujoco/letters_prepared/stage1_all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_MOTIONS = (
    "linear",
    "rotating",
    "random_contact",
    "random_no_contact",
    "mixed",
    "rl_coverage",
)


@dataclass
class EpisodeStats:
    """Per-episode summary statistics."""

    path: str
    split: str
    letter: str
    motion: str
    frames: int
    xy_translation: float
    xy_path_length: float
    yaw_rotation: float
    error: str = ""


def parse_tokens(value: str | None) -> list[str]:
    """Parse comma/space separated CLI selections."""
    if value is None:
        return []
    return [token.strip() for token in value.replace(",", " ").split() if token.strip()]


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_from_quat_wxyz(quat):
    """Compute z yaw from wxyz quaternions."""
    import numpy as np

    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_episode_stats(
    path: Path, split: str, letter: str, motion: str
) -> EpisodeStats:
    """Read one HDF5 episode summary."""
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as f:
            frames = int(f["action"].shape[0]) if "action" in f else 0
            if "env_state" not in f or f["env_state"].shape[0] == 0:
                return EpisodeStats(
                    str(path),
                    split,
                    letter,
                    motion,
                    frames,
                    0.0,
                    0.0,
                    0.0,
                    "missing env_state",
                )
            env_state = f["env_state"][()]
            frames = frames or int(env_state.shape[0])
            xy = env_state[:, :2]
            if len(xy) >= 2:
                xy_translation = float(np.linalg.norm(xy[-1] - xy[0]))
                xy_path_length = float(
                    np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
                )
            else:
                xy_translation = 0.0
                xy_path_length = 0.0
            yaw = yaw_from_quat_wxyz(env_state[:, 3:7])
            yaw_rotation = abs(wrap_angle(float(yaw[-1] - yaw[0]))) if len(yaw) else 0.0
            return EpisodeStats(
                path=str(path),
                split=split,
                letter=letter,
                motion=motion,
                frames=frames,
                xy_translation=xy_translation,
                xy_path_length=xy_path_length,
                yaw_rotation=yaw_rotation,
            )
    except Exception as exc:  # noqa: BLE001
        return EpisodeStats(
            str(path), split, letter, motion, 0, 0.0, 0.0, 0.0, repr(exc)
        )


def iter_prepared_dataset(dataset_dir: Path) -> list[tuple[Path, str, str, str]]:
    """Return episodes from a prepared flat dataset."""
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        items = []
        for episode in metadata.get("episodes", []):
            prepared_path = dataset_dir / episode["prepared_episode"]
            items.append(
                (
                    prepared_path,
                    episode.get("split", "unknown"),
                    episode.get("letter", "?"),
                    episode.get("motion", "unknown"),
                )
            )
        return items

    items = []
    for split in ("train", "val"):
        for path in sorted((dataset_dir / split).glob("episode_*.hdf5")):
            letter, motion = infer_letter_motion_from_path(path)
            items.append((path, split, letter, motion))
    return items


def iter_raw_source(source_dir: Path) -> list[tuple[Path, str, str, str]]:
    """Return episodes from raw nested letter/motion layout."""
    items = []
    for letter_dir in sorted(source_dir.iterdir() if source_dir.exists() else []):
        if not letter_dir.is_dir() or len(letter_dir.name) != 1:
            continue
        letter = letter_dir.name.upper()
        for motion_dir in sorted(letter_dir.iterdir()):
            if not motion_dir.is_dir() or motion_dir.name in {"videos", "rejected"}:
                continue
            for path in sorted(motion_dir.glob("episode_*.hdf5")):
                items.append((path, "raw", letter, motion_dir.name))
    return items


def infer_letter_motion_from_path(path: Path) -> tuple[str, str]:
    """Best-effort source inference for non-metadata prepared datasets."""
    try:
        target = path.resolve()
    except FileNotFoundError:
        target = path
    parts = target.parts
    if "letters" in parts:
        idx = parts.index("letters")
        if idx + 2 < len(parts):
            return parts[idx + 1].upper(), parts[idx + 2]
    return "?", "unknown"


def filter_items(
    items: Iterable[tuple[Path, str, str, str]],
    letters: set[str],
    motions: set[str],
) -> list[tuple[Path, str, str, str]]:
    """Apply optional letter/motion filters."""
    result = []
    for path, split, letter, motion in items:
        if letters and letter.upper() not in letters:
            continue
        if motions and motion not in motions:
            continue
        result.append((path, split, letter.upper(), motion))
    return result


def summarize(stats: list[EpisodeStats]) -> list[dict]:
    """Summarize per split/letter/motion."""
    import numpy as np

    groups: dict[tuple[str, str, str], list[EpisodeStats]] = defaultdict(list)
    for item in stats:
        groups[(item.split, item.letter, item.motion)].append(item)

    rows = []
    for (split, letter, motion), group in sorted(groups.items()):
        frames = np.array([item.frames for item in group], dtype=np.float64)
        translations = np.array(
            [item.xy_translation for item in group], dtype=np.float64
        )
        path_lengths = np.array(
            [item.xy_path_length for item in group], dtype=np.float64
        )
        rotations = np.array([item.yaw_rotation for item in group], dtype=np.float64)
        errors = sum(1 for item in group if item.error)
        rows.append(
            {
                "split": split,
                "letter": letter,
                "motion": motion,
                "episodes": len(group),
                "frames": int(frames.sum()),
                "mean_frames": float(frames.mean()) if len(frames) else 0.0,
                "mean_xy_translation": float(translations.mean())
                if len(translations)
                else 0.0,
                "mean_xy_path_length": float(path_lengths.mean())
                if len(path_lengths)
                else 0.0,
                "mean_yaw_rotation": float(rotations.mean()) if len(rotations) else 0.0,
                "errors": errors,
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print a compact distribution table."""
    if not rows:
        print("No episodes matched.")
        return
    headers = [
        "split",
        "letter",
        "motion",
        "episodes",
        "frames",
        "mean_frames",
        "mean_xy_translation",
        "mean_xy_path_length",
        "mean_yaw_rotation",
        "errors",
    ]
    widths = {header: len(header) for header in headers}
    formatted_rows = []
    for row in rows:
        formatted = {
            "split": str(row["split"]),
            "letter": str(row["letter"]),
            "motion": str(row["motion"]),
            "episodes": str(row["episodes"]),
            "frames": str(row["frames"]),
            "mean_frames": f"{row['mean_frames']:.1f}",
            "mean_xy_translation": f"{row['mean_xy_translation']:.4f}",
            "mean_xy_path_length": f"{row['mean_xy_path_length']:.4f}",
            "mean_yaw_rotation": f"{row['mean_yaw_rotation']:.4f}",
            "errors": str(row["errors"]),
        }
        formatted_rows.append(formatted)
        for header, value in formatted.items():
            widths[header] = max(widths[header], len(value))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset_dir", help="Prepared flat dataset directory.")
    group.add_argument(
        "--source_dir",
        default="data/mujoco/letters",
        help="Raw nested letters directory.",
    )
    parser.add_argument(
        "--letters", default=None, help="Optional letters filter, e.g. 'A B C'."
    )
    parser.add_argument("--motions", default=None, help="Optional motions filter.")
    parser.add_argument(
        "--json_out", default=None, help="Optional path for detailed JSON output."
    )
    parser.add_argument(
        "--csv_out", default=None, help="Optional path for summary CSV output."
    )
    parser.add_argument("--fail_if_empty", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    letters = {token.upper() for token in parse_tokens(args.letters)}
    motions = set(parse_tokens(args.motions))

    if args.dataset_dir:
        root = Path(args.dataset_dir)
        items = iter_prepared_dataset(root)
        label = f"prepared dataset {root}"
    else:
        root = Path(args.source_dir)
        items = iter_raw_source(root)
        label = f"raw source {root}"

    items = filter_items(items, letters=letters, motions=motions)
    if args.fail_if_empty and not items:
        raise RuntimeError(f"No episodes matched in {label}")

    stats = [
        read_episode_stats(path, split, letter, motion)
        for path, split, letter, motion in items
    ]
    rows = summarize(stats)

    print(f"Dataset distribution for {label}")
    print(f"Matched episodes: {len(stats)}")
    print(f"Total frames: {sum(item.frames for item in stats)}")
    print_table(rows)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "label": label,
                    "episodes": [asdict(item) for item in stats],
                    "summary": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"Wrote JSON: {out}")
    if args.csv_out:
        write_csv(Path(args.csv_out), rows)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
