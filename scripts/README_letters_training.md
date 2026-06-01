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
| `scripts/train_letters_generalization.py` | YAML-driven local stage-1/stage-2 letter generalization workflow. |

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

Explicit train-letter / validation-letter split:

```bash
python scripts/prepare_letters_dataset.py \
  --source_dir data/mujoco/letters \
  --output_dir data/mujoco/letters_prepared/train20_val6 \
  --motions "rl_coverage" \
  --max_episodes_per_group 50 \
  --train_letters "A B C D E F G H I J K L M N O P Q R S T" \
  --val_letters "U V W X Y Z" \
  --val_ratio 0.0 \
  --min_val_per_group 0 \
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

## Letter generalization workflow

Use `configurations/letters_generalization.yaml` for the 26-letter / 20-letter / 6-heldout experiment. The default heldout split is `U V W X Y Z`; edit `heldout_letters` in the YAML if you want a different split.

List configured datasets and runs:

```bash
python scripts/train_letters_generalization.py list
```

Prepare the two configured datasets:

```bash
python scripts/train_letters_generalization.py prepare --dataset all
```

Stage 1 runs are separate:

```bash
# S1 sees all 26 letters.
python scripts/train_letters_generalization.py stage1 --run all_letters

# S1 sees only the 20 non-heldout letters; val is the six heldout letters.
python scripts/train_letters_generalization.py stage1 --run train20_val6
```

Stage 2 runs are separate and load the referenced stage-1 checkpoint:

```bash
# Baseline: S1 and S2 both use all letters.
python scripts/train_letters_generalization.py stage2 --run all_letters

# S1 all letters, S2 train on 20 letters and validate on six heldout letters.
python scripts/train_letters_generalization.py stage2 --run s1_all_s2_train20

# S1 and S2 both train on 20 letters and validate on six heldout letters.
python scripts/train_letters_generalization.py stage2 --run s1_train20_s2_train20
```

Useful overrides:

```bash
# Show commands without running training.
python scripts/train_letters_generalization.py stage1 --run all_letters --dry_run --wandb_mode disabled

# Force dataset relinking after changing heldout_letters or episode count.
python scripts/train_letters_generalization.py prepare --dataset all --prepare always

# Use a different local GPU.
python scripts/train_letters_generalization.py stage2 --run s1_all_s2_train20 --gpu 1
```

## Important notes

- The prepared dataset uses symlinks by default, so it is cheap to create. Use `--copy` in `prepare_letters_dataset.py` if you need a standalone copy.
- `train_letters_stage1_local.zsh` defaults to `--prepare_dataset if_missing`. If you change the selected letters/motions but reuse the same `--dataset_dir`, either use a new dataset dir or pass `--prepare_dataset always`.
- Stage 1 uses `dataset=sim_aloha_dataset`, `dataset.obs_keys=[top_pov]`, `algorithm.action_dim=4`, `algorithm.training_stage=1`.
- The script resumes from `<run_dir>/checkpoints/last.ckpt` by default if it exists. Pass `--resume 0` to ignore it.
- `train_letters_generalization.py` is stage-separated: run `stage1` first, then run the desired `stage2` command after a stage-1 checkpoint exists.
- Stage 2 snapshots the selected stage-1 checkpoint and `.hydra/` config into `<stage2_run_dir>/_s1_seed/` before training so `algorithm.load_ae` can resolve the correct autoencoder config.
