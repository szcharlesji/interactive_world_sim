#!/usr/bin/env python3
"""Merge a continued stage-2 checkpoint with a stage-3 finetuned decoder.

Stage 2 trains the dynamics predictor while the autoencoder is frozen. Stage 3
finetunes the decoder while the encoder and dynamics predictor are frozen. If
stage 2 continues training after stage 3 was launched, the latest stage-2
checkpoint has the best dynamics but the stage-3 checkpoint has the best
decoder. This script creates a single checkpoint by taking the stage-2
checkpoint as the base and replacing only decoder.* tensors with those from the
stage-3 checkpoint.

The output directory is made checkpoint-load friendly by copying a .hydra/
configuration directory next to checkpoints/ when available.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"expected checkpoint dict at {path}, got {type(ckpt)}")
    if "state_dict" not in ckpt:
        raise KeyError(f"checkpoint missing state_dict: {path}")
    if not isinstance(ckpt["state_dict"], dict):
        raise TypeError(f"checkpoint state_dict is not a dict: {path}")
    return ckpt


def _run_dir_for_checkpoint(ckpt_path: Path) -> Path:
    # Expected Lightning/Hydra layout: <run_dir>/checkpoints/<file>.ckpt
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _copy_hydra_dir(src_ckpt: Path, out_run_dir: Path) -> bool:
    hydra_src = _run_dir_for_checkpoint(src_ckpt) / ".hydra"
    hydra_dst = out_run_dir / ".hydra"
    if not hydra_src.is_dir():
        return False
    shutil.copytree(hydra_src, hydra_dst, dirs_exist_ok=True)
    return True


def _maybe_patch_training_stage(out_run_dir: Path, training_stage: int) -> None:
    config_path = out_run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if "algorithm" in cfg and "training_stage" in cfg.algorithm:
        cfg.algorithm.training_stage = training_stage
        OmegaConf.save(cfg, config_path)


def _replace_decoder_weights(
    s2_state: dict[str, Any],
    s3_state: dict[str, Any],
) -> list[str]:
    decoder_keys = sorted(k for k in s2_state if k.startswith("decoder."))
    if not decoder_keys:
        raise RuntimeError("no decoder.* keys found in the stage-2 checkpoint")

    missing = [k for k in decoder_keys if k not in s3_state]
    if missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(
            f"stage-3 checkpoint is missing {len(missing)} decoder keys:\n  {preview}"
        )

    mismatched = [
        (k, tuple(s2_state[k].shape), tuple(s3_state[k].shape))
        for k in decoder_keys
        if s2_state[k].shape != s3_state[k].shape
    ]
    if mismatched:
        preview = "\n  ".join(f"{k}: {a} != {b}" for k, a, b in mismatched[:20])
        raise RuntimeError(
            f"decoder shape mismatch for {len(mismatched)} keys:\n  {preview}"
        )

    for key in decoder_keys:
        s2_state[key] = s3_state[key]
    return decoder_keys


def _default_out_ckpt(out: Path) -> Path:
    if out.suffix == ".ckpt":
        return out
    return out / "checkpoints" / "last.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one checkpoint with encoder+dynamics from a stage-2 ckpt "
            "and decoder from a stage-3 ckpt."
        )
    )
    parser.add_argument(
        "--stage2_ckpt",
        required=True,
        type=Path,
        help="Source checkpoint for encoder, dynamics, normalizer, and metadata.",
    )
    parser.add_argument(
        "--stage3_ckpt",
        required=True,
        type=Path,
        help="Source checkpoint for decoder.* weights.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help=(
            "Output .ckpt path or output run directory. If a directory is given, "
            "writes <out>/checkpoints/last.ckpt."
        ),
    )
    parser.add_argument(
        "--hydra_source",
        choices=["stage2", "stage3", "none"],
        default="stage2",
        help="Which source .hydra directory to copy next to the merged checkpoint.",
    )
    parser.add_argument(
        "--patch_hydra_training_stage",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help=(
            "If a .hydra/config.yaml is copied, set algorithm.training_stage to "
            "this value. Use 2 for rollout validation/inference."
        ),
    )
    parser.add_argument(
        "--keep_training_state",
        action="store_true",
        help=(
            "Keep optimizer/lr scheduler state from the stage-2 checkpoint. By "
            "default these are removed because the merged checkpoint is intended "
            "for validation/inference rather than exact training resume."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    stage2_ckpt = args.stage2_ckpt.expanduser().resolve()
    stage3_ckpt = args.stage3_ckpt.expanduser().resolve()
    out_ckpt = _default_out_ckpt(args.out.expanduser()).resolve()
    out_run_dir = _run_dir_for_checkpoint(out_ckpt)

    s2 = _load_checkpoint(stage2_ckpt)
    s3 = _load_checkpoint(stage3_ckpt)

    replaced_decoder_keys = _replace_decoder_weights(s2["state_dict"], s3["state_dict"])

    if not args.keep_training_state:
        s2["optimizer_states"] = []
        s2["lr_schedulers"] = []
        # Lightning may try to restore loops when resuming training. Drop the
        # fit loop state to make accidental resume less misleading.
        if "loops" in s2:
            s2.pop("loops")

    s2["merged_from"] = {
        "stage2_encoder_dynamics_ckpt": str(stage2_ckpt),
        "stage3_decoder_ckpt": str(stage3_ckpt),
        "rule": "base checkpoint is stage2; replaced only decoder.* tensors from stage3",
        "num_decoder_keys_replaced": len(replaced_decoder_keys),
    }

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(s2, out_ckpt)

    copied_hydra = False
    if args.hydra_source != "none":
        hydra_ckpt = stage2_ckpt if args.hydra_source == "stage2" else stage3_ckpt
        copied_hydra = _copy_hydra_dir(hydra_ckpt, out_run_dir)
        if copied_hydra:
            _maybe_patch_training_stage(out_run_dir, args.patch_hydra_training_stage)

    manifest = {
        "merged_checkpoint": str(out_ckpt),
        "stage2_ckpt": str(stage2_ckpt),
        "stage3_ckpt": str(stage3_ckpt),
        "hydra_source": args.hydra_source,
        "copied_hydra": copied_hydra,
        "patched_hydra_training_stage": (
            args.patch_hydra_training_stage if copied_hydra else None
        ),
        "keep_training_state": args.keep_training_state,
        "num_decoder_keys_replaced": len(replaced_decoder_keys),
    }
    manifest_path = out_run_dir / "merged_checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved merged checkpoint: {out_ckpt}")
    print(f"Replaced decoder tensors: {len(replaced_decoder_keys)}")
    if copied_hydra:
        print(f"Copied .hydra from {args.hydra_source} checkpoint run directory")
    elif args.hydra_source != "none":
        print(
            f"WARNING: no .hydra directory found next to {args.hydra_source} checkpoint; "
            "merged checkpoint was still written"
        )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
