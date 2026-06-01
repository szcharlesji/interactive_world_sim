# Procedural Letter World-Model Training Scripts

These scripts prepare the current nested letter collection layout for `SimAlohaDataset` and launch local single-GPU world-model training.

Raw collection layout:

```text
data/mujoco/letters/<LETTER>/<MOTION>/episode_*.hdf5
```

World-model dataset layout expected by `SimAlohaDataset`:

```text
<dataset_dir>/train/episode_*.hdf5
<dataset_dir>/val/episode_*.hdf5
```

`prepare_letters_dataset.py` creates that flat train/val layout with symlinks and a `metadata.json` mapping every prepared episode back to its source letter/motion.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/prepare_letters_dataset.py` | Create a flat `train/` and `val/` dataset from selected letters/motions. |
| `scripts/check_letters_dataset_distribution.py` | Print per-letter/per-motion episode counts, frame counts, translation, and rotation stats. |
| `scripts/train_letters_stage1_local.zsh` | Prepare/check the dataset and train latent world model stage 1 on a local GPU. |

## Environment

From repo root:

```bash
cd /home_shared/grail_charles/interactive_world_sim
conda activate iws
export PYTHONPATH=external/gym-aloha:.
```

## Check raw distribution

```bash
python scripts/check_letters_dataset_distribution.py \
  --source_dir data/mujoco/letters
```

Filter by letters/motions:

```bash
python scripts/check_letters_dataset_distribution.py \
  --source_dir data/mujoco/letters \
  --letters "A B C D" \
  --motions "rl_coverage"
```

## Prepare a dataset manually

All available letters/motions:

```bash
python scripts/prepare_letters_dataset.py \
  --source_dir data/mujoco/letters \
  --output_dir data/mujoco/letters_prepared/all_letters \
  --val_ratio 0.1 \
  --seed 42 \
  --overwrite
```

All letters except held-out letters:

```bash
python scripts/prepare_letters_dataset.py \
  --source_dir data/mujoco/letters \
  --output_dir data/mujoco/letters_prepared/no_QXZ \
  --exclude_letters "Q X Z" \
  --val_ratio 0.1 \
  --seed 42 \
  --overwrite
```

Explicit letters and motions:

```bash
python scripts/prepare_letters_dataset.py \
  --source_dir data/mujoco/letters \
  --output_dir data/mujoco/letters_prepared/ABCD_mixed_rl \
  --letters "A B C D" \
  --motions "mixed rl_coverage" \
  --val_ratio 0.1 \
  --seed 42 \
  --overwrite
```

Check prepared distribution:

```bash
python scripts/check_letters_dataset_distribution.py \
  --dataset_dir data/mujoco/letters_prepared/all_letters \
  --fail_if_empty
```

## Train stage 1 locally

Default: all available letters/motions, local GPU `0`.

```bash
./scripts/train_letters_stage1_local.zsh \
  --gpu 0 \
  --dataset_dir data/mujoco/letters_prepared/stage1_all \
  --run_dir outputs/letters/stage_1_all \
  --run_name letters_stage_1_all
```

Hold out letters from stage 1:

```bash
./scripts/train_letters_stage1_local.zsh \
  --gpu 0 \
  --dataset_dir data/mujoco/letters_prepared/stage1_no_QXZ \
  --run_dir outputs/letters/stage_1_no_QXZ \
  --run_name letters_stage_1_no_QXZ \
  --exclude_letters "Q X Z"
```

Train on explicit letters and only mixed/RL coverage data:

```bash
./scripts/train_letters_stage1_local.zsh \
  --gpu 1 \
  --letters "A B C D" \
  --motions "mixed rl_coverage" \
  --dataset_dir data/mujoco/letters_prepared/stage1_ABCD_mixed_rl \
  --run_dir outputs/letters/stage_1_ABCD_mixed_rl \
  --run_name letters_stage_1_ABCD_mixed_rl
```

Smoke test command construction without training:

```bash
./scripts/train_letters_stage1_local.zsh \
  --dry_run 1 \
  --gpu 0 \
  --wandb_mode disabled
```

## Important notes

- The prepared dataset uses symlinks by default, so it is cheap to create. Use `--copy` in `prepare_letters_dataset.py` if you need a standalone copy.
- `train_letters_stage1_local.zsh` defaults to `--prepare_dataset if_missing`. If you change the selected letters/motions but reuse the same `--dataset_dir`, either use a new dataset dir or pass `--prepare_dataset always`.
- Stage 1 uses `dataset=sim_aloha_dataset`, `dataset.obs_keys=[top_pov]`, `algorithm.action_dim=4`, `algorithm.training_stage=1`.
- The script resumes from `<run_dir>/checkpoints/last.ckpt` by default if it exists. Pass `--resume 0` to ignore it.
- For future stage-2 experiments, use the same prepared-dataset script to create separate dataset dirs for different letter sets, e.g. `stage2_all` vs `stage2_no_QXZ`.
