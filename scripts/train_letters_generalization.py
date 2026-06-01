#!/usr/bin/env python3
"""Run local procedural-letter generalization training.

This is a thin orchestration layer around the existing Hydra entrypoint
(`main.py`) and the letter dataset preparation scripts. It keeps stage 1 and
stage 2 as separate commands while sharing one YAML experiment definition.

Typical usage:

    # Prepare all configured datasets.
    python scripts/train_letters_generalization.py prepare --dataset all

    # Train stage 1 on all letters.
    python scripts/train_letters_generalization.py stage1 --run all_letters

    # Train stage 2 on 20 train letters / 6 held-out validation letters, seeded
    # from the all-letter stage-1 checkpoint.
    python scripts/train_letters_generalization.py stage2 --run s1_all_s2_train20
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_CONFIG = "configurations/letters_generalization.yaml"


def parse_tokens(value: Any) -> list[str]:
    """Parse a YAML/CLI letter or motion selection into tokens."""
    if value is None:
        return []
    if isinstance(value, str):
        return [
            token.strip() for token in value.replace(",", " ").split() if token.strip()
        ]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(parse_tokens(item))
        return result
    return [str(value)]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load an OmegaConf YAML file as a plain dict with interpolations resolved."""
    cfg = OmegaConf.load(path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return resolved


def cfg_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get one top-level config value."""
    return cfg[key] if key in cfg else default


def resolve_letters(value: Any, cfg: dict[str, Any]) -> list[str]:
    """Resolve letter aliases used in the generalization config."""
    all_letters = [
        letter.upper()
        for letter in parse_tokens(cfg_get(cfg, "all_letters", ALL_LETTERS))
    ]
    heldout_letters = [
        letter.upper() for letter in parse_tokens(cfg_get(cfg, "heldout_letters", []))
    ]

    if isinstance(value, str):
        key = value.strip().lower()
        if key == "all":
            return all_letters
        if key == "heldout":
            return heldout_letters
        if key in {"complement_heldout", "train20", "non_heldout"}:
            heldout = set(heldout_letters)
            return [letter for letter in all_letters if letter not in heldout]

    letters = [letter.upper() for letter in parse_tokens(value)]
    invalid = [letter for letter in letters if letter not in ALL_LETTERS]
    if invalid:
        raise ValueError(f"Invalid letter selection {invalid} from {value!r}")
    return letters


def join_tokens(tokens: list[str]) -> str:
    """Join tokens for CLI args accepted by prepare/check scripts."""
    return " ".join(tokens)


def hydra_value(value: Any) -> str:
    """Format a Python value as a simple Hydra override value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def print_command(cmd: list[str], env_prefix: dict[str, str] | None = None) -> None:
    """Print a shell-copyable command."""
    pieces = []
    if env_prefix:
        pieces.extend(
            f"{key}={shlex.quote(value)}" for key, value in env_prefix.items()
        )
    pieces.extend(shlex.quote(part) for part in cmd)
    print("  " + " ".join(pieces))


def run_command(
    cmd: list[str],
    *,
    dry_run: bool,
    env: dict[str, str] | None = None,
    env_prefix: dict[str, str] | None = None,
) -> None:
    """Print and optionally run one command."""
    print_command(cmd, env_prefix=env_prefix)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def project_python(cfg: dict[str, Any]) -> str:
    """Return the Python executable to use for child processes."""
    configured = cfg_get(cfg, "python", None)
    return str(configured) if configured else sys.executable


def prepare_mode(cli_mode: str | None, cfg: dict[str, Any]) -> str:
    """Resolve dataset preparation mode."""
    mode = cli_mode or cfg_get(cfg_get(cfg, "prepare", {}), "mode", "if_missing")
    if mode not in {"if_missing", "always", "never"}:
        raise ValueError(
            f"Invalid prepare mode {mode!r}; expected if_missing|always|never"
        )
    return mode


def dataset_cfg(cfg: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Return one dataset config."""
    datasets = cfg_get(cfg, "datasets", {})
    if dataset_name not in datasets:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Available: {sorted(datasets)}"
        )
    ds = datasets[dataset_name]
    if not isinstance(ds, dict):
        raise TypeError(f"datasets.{dataset_name} must be a mapping")
    return ds


def dataset_output_dir(cfg: dict[str, Any], dataset_name: str) -> Path:
    """Return output dir for a prepared dataset."""
    ds = dataset_cfg(cfg, dataset_name)
    output_dir = ds.get("output_dir")
    if output_dir is None:
        output_dir = str(
            Path(
                str(
                    cfg_get(
                        cfg,
                        "prepared_root",
                        "data/mujoco/letters_prepared/generalization",
                    )
                )
            )
            / dataset_name
        )
    return Path(str(output_dir))


def maybe_prepare_dataset(
    cfg: dict[str, Any],
    dataset_name: str,
    *,
    mode: str,
    dry_run: bool,
) -> None:
    """Prepare one configured letter dataset if requested."""
    ds = dataset_cfg(cfg, dataset_name)
    out_dir = dataset_output_dir(cfg, dataset_name)
    metadata_path = out_dir / "metadata.json"

    if mode == "if_missing" and metadata_path.is_file():
        print(f"[prepare] Reusing {dataset_name}: {out_dir}")
        return
    if mode == "never":
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"--prepare never but metadata is missing for {dataset_name}: {metadata_path}"
            )
        print(f"[prepare] Using existing {dataset_name}: {out_dir}")
        return

    prepare_cfg = cfg_get(cfg, "prepare", {})
    source_dir = str(
        ds.get("source_dir", cfg_get(cfg, "source_dir", "data/mujoco/letters"))
    )
    seed = int(ds.get("seed", cfg_get(cfg, "seed", 42)))
    min_frames = int(ds.get("min_frames", prepare_cfg.get("min_frames", 1)))
    max_episodes = ds.get(
        "max_episodes_per_letter",
        cfg_get(
            cfg, "max_episodes_per_letter", cfg_get(cfg, "episodes_per_letter", 50)
        ),
    )
    motions = parse_tokens(ds.get("motions", cfg_get(cfg, "motions", ["rl_coverage"])))

    cmd = [
        project_python(cfg),
        "scripts/prepare_letters_dataset.py",
        "--source_dir",
        source_dir,
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--min_frames",
        str(min_frames),
    ]
    if max_episodes is not None:
        cmd.extend(["--max_episodes_per_group", str(int(max_episodes))])
    if motions:
        cmd.extend(["--motions", join_tokens(motions)])
    if bool(ds.get("copy", prepare_cfg.get("copy", False))):
        cmd.append("--copy")

    split = str(ds.get("split", "ratio")).lower()
    if split == "ratio":
        letters = resolve_letters(ds.get("letters", "all"), cfg)
        if letters:
            cmd.extend(["--letters", join_tokens(letters)])
        val_ratio = float(ds.get("val_ratio", prepare_cfg.get("val_ratio", 0.1)))
        min_val = int(
            ds.get("min_val_per_group", prepare_cfg.get("min_val_per_group", 1))
        )
        cmd.extend(["--val_ratio", str(val_ratio), "--min_val_per_group", str(min_val)])
    elif split in {"explicit", "explicit_letters", "heldout"}:
        train_letters = resolve_letters(
            ds.get("train_letters", "complement_heldout"), cfg
        )
        val_letters = resolve_letters(ds.get("val_letters", "heldout"), cfg)
        if not train_letters or not val_letters:
            raise ValueError(
                f"Dataset {dataset_name} requires non-empty train_letters and val_letters"
            )
        cmd.extend(["--train_letters", join_tokens(train_letters)])
        cmd.extend(["--val_letters", join_tokens(val_letters)])
        # Explicit letter splits do not use per-group random validation splits.
        cmd.extend(["--val_ratio", "0.0", "--min_val_per_group", "0"])
    else:
        raise ValueError(f"Unsupported split mode for dataset {dataset_name}: {split}")

    if mode == "always" or (mode == "if_missing" and out_dir.exists()):
        cmd.append("--overwrite")

    print(f"[prepare] Preparing {dataset_name}: {out_dir}")
    run_command(cmd, dry_run=dry_run)


def maybe_check_distribution(
    cfg: dict[str, Any], dataset_name: str, *, dry_run: bool
) -> None:
    """Run the prepared dataset distribution checker."""
    prepare_cfg = cfg_get(cfg, "prepare", {})
    if not bool(prepare_cfg.get("check_distribution", True)):
        return
    out_dir = dataset_output_dir(cfg, dataset_name)
    cmd = [
        project_python(cfg),
        "scripts/check_letters_dataset_distribution.py",
        "--dataset_dir",
        str(out_dir),
        "--fail_if_empty",
    ]
    print(f"[check] Checking distribution for {dataset_name}: {out_dir}")
    run_command(cmd, dry_run=dry_run)


def build_training_env(
    cfg: dict[str, Any], gpu: str | int | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Build environment for local training commands."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    prefix = "external/gym-aloha:."
    env["PYTHONPATH"] = (
        prefix if not existing_pythonpath else f"{prefix}:{existing_pythonpath}"
    )
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    env_prefix: dict[str, str] = {}
    if gpu is not None:
        gpu_str = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        env_prefix["CUDA_VISIBLE_DEVICES"] = gpu_str
    return env, env_prefix


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return last.ckpt or newest checkpoint under a run dir."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(
        ckpt_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def stage1_run_cfg(cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    """Return one stage-1 run config."""
    runs = cfg_get(cfg, "stage1_runs", {})
    if run_name not in runs:
        raise KeyError(f"Unknown stage1 run {run_name!r}. Available: {sorted(runs)}")
    run = runs[run_name]
    if not isinstance(run, dict):
        raise TypeError(f"stage1_runs.{run_name} must be a mapping")
    return run


def stage2_run_cfg(cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    """Return one stage-2 run config."""
    runs = cfg_get(cfg, "stage2_runs", {})
    if run_name not in runs:
        raise KeyError(f"Unknown stage2 run {run_name!r}. Available: {sorted(runs)}")
    run = runs[run_name]
    if not isinstance(run, dict):
        raise TypeError(f"stage2_runs.{run_name} must be a mapping")
    return run


def merged_stage_cfg(
    cfg: dict[str, Any], stage: str, run: dict[str, Any]
) -> dict[str, Any]:
    """Merge global stage defaults with per-run training overrides."""
    merged = dict(cfg_get(cfg, stage, {}))
    merged.update(run.get("training", {}))
    return merged


def append_common_hydra_args(
    cmd: list[str],
    *,
    cfg: dict[str, Any],
    run: dict[str, Any],
    stage_cfg: dict[str, Any],
    dataset_name: str,
    run_dir: Path,
    run_name: str,
    training_stage: int,
    wandb_mode: str,
) -> None:
    """Append Hydra overrides shared by stage 1 and stage 2."""
    obs_keys = parse_tokens(
        stage_cfg.get("obs_keys", cfg_get(cfg, "obs_keys", ["top_pov"]))
    )
    action_dim = int(stage_cfg.get("action_dim", cfg_get(cfg, "action_dim", 4)))
    dataset_dir = dataset_output_dir(cfg, dataset_name)

    cmd.extend(
        [
            f"+name={run_name}",
            "algorithm=latent_world_model",
            "experiment=exp_latent_dyn",
            "dataset=sim_aloha_dataset",
            f"dataset.dataset_dir={dataset_dir}",
            f"dataset.horizon={int(stage_cfg['horizon'])}",
            f"dataset.val_horizon={int(stage_cfg['val_horizon'])}",
            f"dataset.obs_keys={hydra_value(obs_keys)}",
            f"dataset.use_cache={hydra_value(stage_cfg.get('use_cache', True))}",
            f"hydra.run.dir={run_dir}",
            f"experiment.training.batch_size={int(stage_cfg['batch_size'])}",
            f"experiment.training.max_steps={int(stage_cfg['max_steps'])}",
            f"experiment.training.log_every_n_steps={int(stage_cfg.get('log_every_n_steps', 100))}",
            f"experiment.training.optim.accumulate_grad_batches={int(stage_cfg.get('accumulate_grad_batches', 1))}",
            f"experiment.training.checkpointing.every_n_train_steps={int(stage_cfg['checkpoint_every'])}",
            f"experiment.training.data.num_workers={int(stage_cfg.get('num_workers', 4))}",
            f"experiment.validation.batch_size={int(stage_cfg['val_batch_size'])}",
            f"experiment.validation.val_every_n_step={int(stage_cfg['val_every'])}",
            f"experiment.validation.limit_batch={stage_cfg.get('val_limit_batch', 1.0)}",
            f"experiment.validation.data.num_workers={int(stage_cfg.get('num_workers', 4))}",
            f"algorithm.latent_dim={int(stage_cfg.get('latent_dim', 512))}",
            f"algorithm.action_dim={action_dim}",
            f"algorithm.training_stage={training_stage}",
            "+experiment.training.checkpointing.save_last=True",
            f"wandb.mode={wandb_mode}",
        ]
    )

    for override in parse_tokens(run.get("extra_hydra_overrides", [])):
        cmd.append(override)


def add_resume_arg(
    cmd: list[str], run_dir: Path, *, resume: bool, stage_label: str
) -> None:
    """Add a Lightning checkpoint resume override if requested and present."""
    if not resume:
        print(f"[{stage_label}] Resume disabled")
        return
    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        print(
            f"[{stage_label}] Starting fresh; no checkpoint under {run_dir / 'checkpoints'}"
        )
        return
    print(f"[{stage_label}] Resuming from {ckpt}")
    cmd.append(f"load={ckpt}")


def train_stage1(
    cfg: dict[str, Any],
    run_name: str,
    *,
    cli_prepare_mode: str | None,
    gpu: str | None,
    wandb_mode: str | None,
    dry_run: bool,
    resume: bool | None,
) -> None:
    """Prepare/check data and run one stage-1 training job."""
    run = stage1_run_cfg(cfg, run_name)
    stage_cfg = merged_stage_cfg(cfg, "stage1", run)
    dataset_name = str(run["dataset"])
    mode = prepare_mode(cli_prepare_mode, cfg)
    maybe_prepare_dataset(cfg, dataset_name, mode=mode, dry_run=dry_run)
    maybe_check_distribution(cfg, dataset_name, dry_run=dry_run)

    run_dir = Path(
        str(
            run.get(
                "run_dir",
                Path(str(cfg_get(cfg, "output_root", "outputs/letters/generalization")))
                / run_name
                / "stage_1",
            )
        )
    )
    hydra_name = str(run.get("run_name", f"letters_{run_name}_stage_1"))
    selected_wandb_mode = wandb_mode or str(cfg_get(cfg, "wandb_mode", "online"))
    should_resume = bool(stage_cfg.get("resume", True) if resume is None else resume)

    cmd = [project_python(cfg), "main.py"]
    append_common_hydra_args(
        cmd,
        cfg=cfg,
        run=run,
        stage_cfg=stage_cfg,
        dataset_name=dataset_name,
        run_dir=run_dir,
        run_name=hydra_name,
        training_stage=1,
        wandb_mode=selected_wandb_mode,
    )
    add_resume_arg(cmd, run_dir, resume=should_resume, stage_label="stage1")

    env, env_prefix = build_training_env(
        cfg, gpu if gpu is not None else cfg_get(cfg, "gpu", 0)
    )
    print(f"[stage1] Launching run {run_name}: {run_dir}")
    run_command(cmd, dry_run=dry_run, env=env, env_prefix=env_prefix)


def copy_stage1_seed(
    ckpt: Path,
    *,
    stage1_run_dir: Path | None,
    stage2_run_dir: Path,
    dry_run: bool,
) -> Path:
    """Snapshot a stage-1 checkpoint and .hydra config into the stage-2 run dir."""
    source_hydra = None
    if stage1_run_dir is not None:
        candidate = stage1_run_dir / ".hydra"
        if candidate.is_dir():
            source_hydra = candidate
    if source_hydra is None:
        candidate = ckpt.parent.parent / ".hydra"
        if candidate.is_dir():
            source_hydra = candidate

    if source_hydra is None and not dry_run:
        raise FileNotFoundError(
            "Could not find stage-1 .hydra config next to checkpoint. "
            f"Checked {stage1_run_dir} and {ckpt.parent.parent}."
        )

    snapshot_dir = stage2_run_dir / "_s1_seed"
    snapshot_ckpt = snapshot_dir / "checkpoints" / ckpt.name
    print(f"[stage2] Stage-1 seed checkpoint: {ckpt}")
    print(f"[stage2] Snapshot seed checkpoint: {snapshot_ckpt}")
    if dry_run:
        return snapshot_ckpt

    snapshot_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_ckpt.exists():
        shutil.copy2(ckpt, snapshot_ckpt)
    if source_hydra is not None:
        snapshot_hydra = snapshot_dir / ".hydra"
        if snapshot_hydra.exists():
            shutil.rmtree(snapshot_hydra)
        shutil.copytree(source_hydra, snapshot_hydra)
    return snapshot_ckpt


def resolve_stage1_checkpoint(
    cfg: dict[str, Any],
    run: dict[str, Any],
    *,
    explicit_ckpt: str | None,
    dry_run: bool,
) -> tuple[Path, Path | None]:
    """Resolve the stage-1 checkpoint used to seed a stage-2 run."""
    if explicit_ckpt:
        ckpt = Path(explicit_ckpt)
        if not ckpt.exists() and not dry_run:
            raise FileNotFoundError(f"Explicit stage-1 checkpoint not found: {ckpt}")
        return ckpt, None

    stage1_name = str(run.get("stage1_run", "all_letters"))
    s1_run = stage1_run_cfg(cfg, stage1_name)
    s1_run_dir = Path(
        str(
            s1_run.get(
                "run_dir",
                Path(str(cfg_get(cfg, "output_root", "outputs/letters/generalization")))
                / stage1_name
                / "stage_1",
            )
        )
    )
    ckpt = latest_checkpoint(s1_run_dir)
    if ckpt is None:
        if dry_run:
            ckpt = s1_run_dir / "checkpoints" / "last.ckpt"
        else:
            raise FileNotFoundError(
                f"No stage-1 checkpoint found for stage1_run={stage1_name!r} under "
                f"{s1_run_dir / 'checkpoints'}"
            )
    return ckpt, s1_run_dir


def train_stage2(
    cfg: dict[str, Any],
    run_name: str,
    *,
    cli_prepare_mode: str | None,
    gpu: str | None,
    wandb_mode: str | None,
    dry_run: bool,
    resume: bool | None,
    stage1_ckpt: str | None,
) -> None:
    """Prepare/check data and run one stage-2 training job."""
    run = stage2_run_cfg(cfg, run_name)
    stage_cfg = merged_stage_cfg(cfg, "stage2", run)
    dataset_name = str(run["dataset"])
    mode = prepare_mode(cli_prepare_mode, cfg)
    maybe_prepare_dataset(cfg, dataset_name, mode=mode, dry_run=dry_run)
    maybe_check_distribution(cfg, dataset_name, dry_run=dry_run)

    run_dir = Path(
        str(
            run.get(
                "run_dir",
                Path(str(cfg_get(cfg, "output_root", "outputs/letters/generalization")))
                / run_name
                / "stage_2",
            )
        )
    )
    hydra_name = str(run.get("run_name", f"letters_{run_name}_stage_2"))
    selected_wandb_mode = wandb_mode or str(cfg_get(cfg, "wandb_mode", "online"))
    should_resume = bool(stage_cfg.get("resume", True) if resume is None else resume)

    seed_ckpt, s1_run_dir = resolve_stage1_checkpoint(
        cfg, run, explicit_ckpt=stage1_ckpt, dry_run=dry_run
    )
    if bool(stage_cfg.get("snapshot_stage1_ckpt", True)):
        seed_ckpt = copy_stage1_seed(
            seed_ckpt,
            stage1_run_dir=s1_run_dir,
            stage2_run_dir=run_dir,
            dry_run=dry_run,
        )
    else:
        print(f"[stage2] Loading stage-1 checkpoint directly: {seed_ckpt}")

    cmd = [project_python(cfg), "main.py"]
    append_common_hydra_args(
        cmd,
        cfg=cfg,
        run=run,
        stage_cfg=stage_cfg,
        dataset_name=dataset_name,
        run_dir=run_dir,
        run_name=hydra_name,
        training_stage=2,
        wandb_mode=selected_wandb_mode,
    )
    cmd.extend(
        [
            "algorithm.noise_scheduler.loss_weighting=uniform",
            "algorithm.sampling_strategy=terminal_only",
            f"algorithm.load_ae={seed_ckpt}",
        ]
    )
    add_resume_arg(cmd, run_dir, resume=should_resume, stage_label="stage2")

    env, env_prefix = build_training_env(
        cfg, gpu if gpu is not None else cfg_get(cfg, "gpu", 0)
    )
    print(f"[stage2] Launching run {run_name}: {run_dir}")
    run_command(cmd, dry_run=dry_run, env=env, env_prefix=env_prefix)


def list_config(cfg: dict[str, Any]) -> None:
    """Print configured datasets and runs."""
    print("Datasets:")
    for name in sorted(cfg_get(cfg, "datasets", {})):
        print(f"  {name}: {dataset_output_dir(cfg, name)}")
    print("Stage 1 runs:")
    for name, run in sorted(cfg_get(cfg, "stage1_runs", {}).items()):
        print(f"  {name}: dataset={run.get('dataset')} run_dir={run.get('run_dir')}")
    print("Stage 2 runs:")
    for name, run in sorted(cfg_get(cfg, "stage2_runs", {}).items()):
        print(
            f"  {name}: dataset={run.get('dataset')} "
            f"stage1_run={run.get('stage1_run')} run_dir={run.get('run_dir')}"
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add args shared by subcommands."""
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="YAML workflow config."
    )
    parser.add_argument(
        "--prepare",
        choices=["if_missing", "always", "never"],
        default=None,
        help="Override config prepare.mode.",
    )
    parser.add_argument("--gpu", default=None, help="Override config GPU index.")
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled", "dryrun"],
        default=None,
        help="Override config wandb_mode.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured datasets/runs.")
    add_common_args(list_parser)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare configured datasets."
    )
    add_common_args(prepare_parser)
    prepare_parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key to prepare, or 'all'. Default: all.",
    )

    stage1_parser = subparsers.add_parser(
        "stage1", help="Run one stage-1 training job."
    )
    add_common_args(stage1_parser)
    stage1_parser.add_argument("--run", default="all_letters", help="Stage-1 run key.")
    stage1_parser.add_argument(
        "--resume",
        choices=["0", "1"],
        default=None,
        help="Override stage1.resume.",
    )

    stage2_parser = subparsers.add_parser(
        "stage2", help="Run one stage-2 training job."
    )
    add_common_args(stage2_parser)
    stage2_parser.add_argument(
        "--run", default="s1_all_s2_train20", help="Stage-2 run key."
    )
    stage2_parser.add_argument(
        "--resume",
        choices=["0", "1"],
        default=None,
        help="Override stage2.resume.",
    )
    stage2_parser.add_argument(
        "--stage1_ckpt",
        default=None,
        help="Explicit stage-1 checkpoint path. Overrides stage2_runs.<run>.stage1_run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)

    if args.command == "list":
        list_config(cfg)
        return

    if args.command == "prepare":
        datasets = cfg_get(cfg, "datasets", {})
        if args.dataset == "all":
            names = sorted(datasets)
        else:
            names = [args.dataset]
        mode = prepare_mode(args.prepare, cfg)
        for name in names:
            maybe_prepare_dataset(cfg, name, mode=mode, dry_run=args.dry_run)
            maybe_check_distribution(cfg, name, dry_run=args.dry_run)
        return

    if args.command == "stage1":
        train_stage1(
            cfg,
            args.run,
            cli_prepare_mode=args.prepare,
            gpu=args.gpu,
            wandb_mode=args.wandb_mode,
            dry_run=args.dry_run,
            resume=None if args.resume is None else bool(int(args.resume)),
        )
        return

    if args.command == "stage2":
        train_stage2(
            cfg,
            args.run,
            cli_prepare_mode=args.prepare,
            gpu=args.gpu,
            wandb_mode=args.wandb_mode,
            dry_run=args.dry_run,
            resume=None if args.resume is None else bool(int(args.resume)),
            stage1_ckpt=args.stage1_ckpt,
        )
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
