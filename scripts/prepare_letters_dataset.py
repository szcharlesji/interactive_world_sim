#!/usr/bin/env python3
"""Prepare procedural-letter MuJoCo rollouts for world-model training.

The SimAlohaDataset loader expects a flat layout:

    <dataset_dir>/train/episode_*.hdf5
    <dataset_dir>/val/episode_*.hdf5

Raw letter collection is nested by letter and motion:

    data/mujoco/letters/<LETTER>/<MOTION>/episode_*.hdf5

This script creates a prepared dataset by symlinking or copying selected raw
letter episodes into train/val directories and writing metadata that preserves
the original letter/motion/source path for analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ALL_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MOTIONS = (
    "linear",
    "rotating",
    "random_contact",
    "random_no_contact",
    "mixed",
    "rl_coverage",
)


@dataclass(frozen=True)
class EpisodeRecord:
    """One selected source episode and its prepared split."""

    source_path: str
    split: str
    letter: str
    motion: str
    source_episode: str
    prepared_episode: str


def parse_tokens(value: str | None) -> list[str]:
    """Parse comma/space separated CLI selections."""
    if value is None:
        return []
    return [token.strip() for token in value.replace(",", " ").split() if token.strip()]


def resolve_letters(
    source_dir: Path, letters_arg: str | None, exclude_arg: str | None
) -> list[str]:
    """Resolve the requested letter list."""
    available = sorted(
        path.name.upper()
        for path in source_dir.iterdir()
        if path.is_dir() and len(path.name) == 1 and path.name.upper() in ALL_LETTERS
    )
    if letters_arg:
        requested = [token.upper() for token in parse_tokens(letters_arg)]
    else:
        requested = available

    exclude = {token.upper() for token in parse_tokens(exclude_arg)}
    letters = [letter for letter in requested if letter not in exclude]

    invalid = [letter for letter in letters if letter not in ALL_LETTERS]
    if invalid:
        raise ValueError(f"Invalid letter(s): {invalid}")

    missing = [letter for letter in letters if letter not in available]
    if missing:
        print(f"WARNING: requested letters not found under {source_dir}: {missing}")

    return [letter for letter in letters if letter in available]


def resolve_motions(
    source_dir: Path, letters: Iterable[str], motions_arg: str | None
) -> list[str]:
    """Resolve requested motion list."""
    if motions_arg:
        return parse_tokens(motions_arg)

    available: set[str] = set()
    for letter in letters:
        letter_dir = source_dir / letter
        if not letter_dir.exists():
            continue
        for path in letter_dir.iterdir():
            if path.is_dir() and path.name not in {"videos", "rejected"}:
                if list(path.glob("episode_*.hdf5")):
                    available.add(path.name)
    if available:
        return [motion for motion in DEFAULT_MOTIONS if motion in available] + sorted(
            available.difference(DEFAULT_MOTIONS)
        )
    return list(DEFAULT_MOTIONS)


def collect_sources(
    source_dir: Path,
    letters: Iterable[str],
    motions: Iterable[str],
    min_frames: int,
    max_episodes_per_group: int | None,
    seed: int,
) -> dict[tuple[str, str], list[Path]]:
    """Collect source HDF5 files grouped by (letter, motion)."""
    groups: dict[tuple[str, str], list[Path]] = {}
    rng = random.Random(seed)
    for letter in letters:
        for motion in motions:
            motion_dir = source_dir / letter / motion
            episodes = sorted(motion_dir.glob("episode_*.hdf5"))
            if min_frames > 1:
                episodes = [
                    path for path in episodes if hdf5_num_frames(path) >= min_frames
                ]
            if (
                max_episodes_per_group is not None
                and len(episodes) > max_episodes_per_group
            ):
                episodes = episodes[:]
                rng.shuffle(episodes)
                episodes = sorted(episodes[:max_episodes_per_group])
            if episodes:
                groups[(letter, motion)] = episodes
    return groups


def hdf5_num_frames(path: Path) -> int:
    """Return episode frame count, or 0 if unreadable."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if "action" in f:
                return int(f["action"].shape[0])
            if "env_state" in f:
                return int(f["env_state"].shape[0])
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not read frame count from {path}: {exc}")
    return 0


def split_group(
    paths: list[Path], val_ratio: float, min_val_per_group: int, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split one letter/motion group into train and validation episodes."""
    if len(paths) <= 1 or val_ratio <= 0.0:
        return paths, []

    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n_val = max(min_val_per_group, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Create one prepared episode file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel_src)


def prepare_dataset(args: argparse.Namespace) -> None:
    """Prepare train/val directories and metadata."""
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must not be the same as source_dir")

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"output_dir already exists: {output_dir}. Use --overwrite or choose a new path."
            )

    letters = resolve_letters(source_dir, args.letters, args.exclude_letters)
    motions = resolve_motions(source_dir, letters, args.motions)
    groups = collect_sources(
        source_dir=source_dir,
        letters=letters,
        motions=motions,
        min_frames=args.min_frames,
        max_episodes_per_group=args.max_episodes_per_group,
        seed=args.seed,
    )

    if not groups:
        raise RuntimeError(
            f"No episodes found under {source_dir} for letters={letters} motions={motions}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)

    records: list[EpisodeRecord] = []
    counters = {"train": 0, "val": 0}
    group_summaries = []
    for group_idx, ((letter, motion), paths) in enumerate(sorted(groups.items())):
        train_paths, val_paths = split_group(
            paths,
            val_ratio=args.val_ratio,
            min_val_per_group=args.min_val_per_group,
            seed=args.seed + group_idx,
        )
        group_summaries.append(
            {
                "letter": letter,
                "motion": motion,
                "source_episodes": len(paths),
                "train_episodes": len(train_paths),
                "val_episodes": len(val_paths),
            }
        )
        for split, split_paths in (("train", train_paths), ("val", val_paths)):
            for src in split_paths:
                epi_idx = counters[split]
                prepared_name = f"episode_{epi_idx:06d}.hdf5"
                dst = output_dir / split / prepared_name
                link_or_copy(src, dst, copy=args.copy)
                records.append(
                    EpisodeRecord(
                        source_path=str(src.resolve()),
                        split=split,
                        letter=letter,
                        motion=motion,
                        source_episode=src.name,
                        prepared_episode=f"{split}/{prepared_name}",
                    )
                )
                counters[split] += 1

    metadata = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "letters": letters,
        "motions": motions,
        "exclude_letters": parse_tokens(args.exclude_letters),
        "val_ratio": args.val_ratio,
        "min_val_per_group": args.min_val_per_group,
        "min_frames": args.min_frames,
        "max_episodes_per_group": args.max_episodes_per_group,
        "copy": args.copy,
        "seed": args.seed,
        "train_episodes": counters["train"],
        "val_episodes": counters["val"],
        "groups": group_summaries,
        "episodes": [asdict(record) for record in records],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    print("Prepared letter dataset")
    print(f"  source_dir: {source_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  letters:    {' '.join(letters)}")
    print(f"  motions:    {' '.join(motions)}")
    print(f"  train:      {counters['train']} episodes")
    print(f"  val:        {counters['val']} episodes")
    print(f"  metadata:   {output_dir / 'metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="data/mujoco/letters")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--letters",
        default=None,
        help="Letters to include, e.g. 'A B C'. Default: all available letters.",
    )
    parser.add_argument(
        "--exclude_letters",
        default=None,
        help="Letters to exclude from the selected/default set, e.g. 'Q X Z'.",
    )
    parser.add_argument(
        "--motions",
        default=None,
        help="Motions to include. Default: all available known motions.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--min_val_per_group", type=int, default=1)
    parser.add_argument("--min_frames", type=int, default=1)
    parser.add_argument("--max_episodes_per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlinking."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace output_dir if it exists."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {args.val_ratio}")
    if args.min_val_per_group < 0:
        raise ValueError("min_val_per_group must be non-negative")
    if args.min_frames < 1:
        raise ValueError("min_frames must be positive")
    prepare_dataset(args)


if __name__ == "__main__":
    main()
