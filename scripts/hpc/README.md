# HPC training on NYU Torch

Single-job training of all 3 stages (S1 autoencoder → S2 dynamics → S3 finetune)
on the `h200_courant` partition with auto-resume across requeues.

## TL;DR

```bash
cd /scratch/$USER/interactive_world_sim
git pull origin main
sbatch scripts/hpc/train_all_stages.sbatch                 # bimanual_rope, full
sbatch scripts/hpc/train_all_stages.sbatch pusht           # pusht, full
sbatch scripts/hpc/train_all_stages.sbatch pusht mini      # pusht, mini (smoke test)
```

Monitor:

```bash
squeue -u $USER
tail -F logs/train_<jobid>.out
# wandb: https://wandb.ai/charlesji/interactive_world_sim
```

---

## Does "full" bundle all tasks together?

**No.** `full` vs `mini` is the *size* of the dataset for **one** task, not a
mix of tasks.

The data on Hugging Face is laid out per task:

```
data/full/pusht/...
data/full/single_grasp/...
data/full/bimanual_sweep/...
data/full/bimanual_rope/...
data/full/bimanual_box/...
data/full/single_chain_in_box/...

data/mini/pusht/...
data/mini/single_grasp/...
data/mini/bimanual_sweep/...
data/mini/bimanual_rope/...
```

The sbatch reads exactly one of these:

```bash
dataset.dataset_dir=data/$DATA_SPLIT/$TASK     # e.g. data/full/bimanual_rope
```

So one job = one task. If you want to train both `pusht` and `bimanual_rope`,
launch two jobs in parallel — they run as independent slurm allocations on
separate GPUs:

```bash
sbatch scripts/hpc/train_all_stages.sbatch pusht         full
sbatch scripts/hpc/train_all_stages.sbatch bimanual_rope full
```

Each one writes to its own `outputs/<task>/` tree and produces its own wandb
runs (`pusht_stage_1`, `pusht_stage_2`, …, `bimanual_rope_stage_1`, …).

---

## Prerequisites (one-time)

1. **Conda overlay built** at `$SCRATCH/iws.ext3`
   - Created via the Open OnDemand "Conda Overlay" tool, then populated via
     `apptainer exec --fakeroot --overlay $SCRATCH/iws.ext3:rw <image> bash`
     followed by `mamba env create -f conda_env.yaml` and
     `uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/`
     and `pip install -e .` inside the overlay.

2. **Data downloaded** to `data/full/<task>/` (or `data/mini/<task>/`)
   - Mini: `bash scripts/download_mini_data.sh` (few minutes, fine on a login
     node).
   - Full: launch as a CPU-only sbatch (no GPU needed) — the HF
     `snapshot_download` call is resumable.
   - Single task only: `python scripts/download_data_hf.py
     --repo yixuan1999/interactive-world-sim-data --local_dir data/full/pusht
     --repo_dir pusht`.

3. **Wandb API key**:
   ```bash
   mkdir -p ~/.secrets
   echo "YOUR_WANDB_KEY" > ~/.secrets/wandb_api_key
   chmod 600 ~/.secrets/wandb_api_key
   ```

4. **Logs dir**: the sbatch writes to `logs/`; just `mkdir -p logs` once.

---

## What the sbatch does

| Concern | Mechanism |
|---|---|
| Three stages in one job | `if [[ ! -f $stageN_done ]]` gates each stage; `touch _stageN_done` on clean exit |
| Cross-stage handoff | `algorithm.load_ae=$S{N-1}_CKPT` for S2/S3 (weights-only seed) |
| Mid-stage resume after requeue | `load=last.ckpt` (Lightning `trainer.fit ckpt_path`) restores step counter + optimizer |
| Guaranteed `last.ckpt` | `+experiment.training.checkpointing.save_last=True` injected via Hydra `+` prefix |
| Clean requeue before walltime | `--signal=B:USR1@180` → bash trap → SIGTERM python → `scontrol requeue` → exit 0 |
| Per-stage output isolation | `hydra.run.dir=outputs/<task>/stage_{1,2,3}` |
| Per-task config | `case "$TASK"` block at top selects `obs_keys`, `action_mode`, `action_dim` |

### Output tree per task

```
outputs/<task>/
├── _stage1_done                # state files: presence = completed
├── _stage2_done
├── _stage3_done
├── stage_1/                    # hydra.run.dir for S1
│   ├── .hydra/                 # full resolved config snapshot
│   ├── checkpoints/
│   │   ├── last.ckpt           # symlink to most-recent step ckpt (resume target)
│   │   └── *.ckpt              # every-10k-step snapshots
│   └── wandb/                  # local wandb cache
├── stage_2/
└── stage_3/
```

### Per-task defaults (edit in sbatch `case "$TASK"` block if wrong)

| TASK            | obs_keys           | action_mode    | action_dim |
|-----------------|--------------------|----------------|------------|
| pusht           | `[camera_1_color]` | bimanual_push  | 4          |
| single_grasp    | `[camera_0_color]` | single_arm     | 7          |
| bimanual_sweep  | `[camera_0_color]` | bimanual_push  | 4          |
| bimanual_rope   | `[camera_0_color]` | bimanual_3d    | 6          |

`pusht` is the only task whose values come straight from the README's worked
example; the others are extrapolated. If you hit a config error early in
training, double-check the `case "$TASK"` block against the upstream
`scripts/inference/{keyboard,aloha}/<task>_*.sh` for the right `action_mode`
and `action_dim`.

---

## Common commands

### Launch

```bash
# Default: bimanual_rope, full dataset
sbatch scripts/hpc/train_all_stages.sbatch

# Specific task + split
sbatch scripts/hpc/train_all_stages.sbatch pusht full
sbatch scripts/hpc/train_all_stages.sbatch single_grasp mini

# Parallel jobs for multiple tasks
for t in pusht bimanual_rope; do
  sbatch scripts/hpc/train_all_stages.sbatch $t full
done
```

### Monitor

```bash
squeue -u $USER                                         # queue state
tail -F logs/train_<jobid>.out                          # stdout live
tail -F logs/train_<jobid>.err                          # stderr live
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS    # post-mortem
nvidia-smi                                              # GPU usage on the node (ssh into it via squeue's NODELIST)
```

Wandb dashboard: <https://wandb.ai/charlesji/interactive_world_sim>

### Cancel / requeue

```bash
scancel <jobid>                  # cancel a running job
scontrol requeue <jobid>         # manually requeue
scontrol release <jobid>         # release a job held by --dependency
```

### Re-run a single stage from scratch

```bash
# Force S2 to restart from S1's checkpoint:
rm outputs/bimanual_rope/_stage2_done
rm -rf outputs/bimanual_rope/stage_2/checkpoints/*
sbatch scripts/hpc/train_all_stages.sbatch bimanual_rope full
```

### Start completely fresh

```bash
rm -rf outputs/bimanual_rope
sbatch scripts/hpc/train_all_stages.sbatch bimanual_rope full
```

### Merge continued S2 dynamics with S3 decoder

If S3 was launched from an older S2 checkpoint and S2 later kept improving,
create a single checkpoint with encoder+dynamics from the newer S2 checkpoint
and decoder from the S3 checkpoint. The wrapper below requests the `l40s_lpinto`
partition and runs the merge utility inside the project container:

```bash
sbatch scripts/hpc/merge_stage2_stage3_checkpoint.sbatch
```

The defaults are the PushT stage-2-c merge:

```bash
# S2 source: /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_c/checkpoints/last.ckpt
# S3 source: /scratch/$USER/interactive_world_sim/outputs/pusht/stage_3/checkpoints/last.ckpt
# output   : /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_s3_decoder/checkpoints/last.ckpt
```

You can also pass explicit paths:

```bash
sbatch scripts/hpc/merge_stage2_stage3_checkpoint.sbatch \
  /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_c/checkpoints/last.ckpt \
  /scratch/$USER/interactive_world_sim/outputs/pusht/stage_3/checkpoints/last.ckpt \
  /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_s3_decoder
```

Then visualize the merged checkpoint on validation episodes:

```bash
sbatch scripts/hpc/viz_stage2_trainset.sbatch \
  pusht \
  full \
  /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_s3_decoder/checkpoints/last.ckpt \
  "" \
  val
```

---

## Sbatch resource block (and how to change it)

```
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
```

- **time**: 48h is typical max. If `h200_courant` caps lower, edit `--time` —
  the requeue logic still works at any walltime.
- **mem**: 128G is generous. Drop to 64G if memory contention slows allocation.
- **gpu**: single H200. Multi-GPU would require code changes (the project uses
  `num_devices=1` in `experiment.yaml`).

### Bumping batch sizes for H200 (141 GB VRAM)

The defaults match the README (which assumed an A100/2080). H200 can easily
take larger batches:

| Stage | README default | H200-safe (estimate) |
|-------|---------------|----------------------|
| S1    | 4             | 16-32                |
| S2    | 4             | 16                   |
| S3    | 16            | 32-64                |

Edit `experiment.training.batch_size=N` in the relevant stage block of the
sbatch. Larger batches → fewer steps to converge (sometimes), but you may need
to lower `experiment.training.max_steps` accordingly.

---

## Troubleshooting

**`Key 'X' is not in struct`** — Hydra strict mode rejecting an override of a
key not pre-declared in the yaml. Prefix with `+`:
```
+experiment.training.foo=bar
```

**`FATAL: stage N marked done but no checkpoint in …`** — the state file
exists but the checkpoint dir is empty. Most likely cause: you `rm -rf`'d the
checkpoints but forgot the `_done` marker. Remove the marker:
```bash
rm outputs/<task>/_stage<N>_done
```

**Wandb errors at job start** — likely `~/.secrets/wandb_api_key` missing or
unreadable. The sbatch loads it into `$WANDB_API_KEY`.

**OOM partway through** — likely a batch-size bump that didn't account for
val mode allocating more memory than train. Lower the **validation** batch
size first (it's smaller in the defaults, but the renderer doubles peak
allocation):
```
experiment.validation.batch_size=2
```

**Job killed exactly at walltime with no requeue** — the SIGUSR1 trap didn't
fire. Check that `--signal=B:USR1@180` is in the sbatch header (the `B:`
prefix sends to bash, not the python child; the `@180` is the lead time).

**HDF5 read errors / DataLoader hangs** — the sbatch already exports
`HDF5_USE_FILE_LOCKING=FALSE`. If still failing, drop `num_workers` to 2 in
both `experiment.training.data.num_workers` and `.validation.data.num_workers`.

---

## What gets logged to wandb (per stage, per namespace)

After the recent metric-parity work, both `training/` and `validation/` carry
the same informative set per stage:

| Stage | both namespaces                                |
|-------|------------------------------------------------|
| S1    | `rec_loss, mse, ssim, psnr, uiqi`              |
| S2    | `loss, dyn_loss, fvd, mse, ssim, psnr, uiqi`   |
| S3    | `rec_loss, fvd, mse, ssim, psnr, uiqi`         |

Plus `epoch`, `trainer/global_step`, `lr_scheduler/*`, and
`validation_vis/video_*` for qualitative reconstruction checks.

The expensive training-side image metrics fire every
`algorithm.compute_train_metrics_every_n_steps` steps (defaults to the val
cadence). Disable entirely with `algorithm.compute_train_image_metrics=False`.

---

## Expected wall-clock

Rough estimates on a single H200 at the sbatch's default batch sizes
(`batch_size=4`):

| Stage | max_steps | est. time |
|-------|-----------|-----------|
| S1    | 1,000,005 | 2–4 days  |
| S2    | 1,000,005 | 4–6 days  |
| S3    | 1,000,005 | 2–4 days  |

So expect a full chain to take **8–14 days of GPU time** spread across
**4–8 requeues** at `--time=48:00:00`. The state files + `last.ckpt` resume
make this transparent.
