#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-s2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/s2_%j.out
#SBATCH --error=logs/s2_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 2 only — runs independently of an in-flight stage-1 job.
# Snapshots a stage-1 checkpoint at job start and trains the dynamics
# model from it. Safe to run in parallel with a still-going stage-1 job;
# writes to outputs/<task>/stage_2/ which is disjoint from stage_1.
#
# Usage:
#   sbatch scripts/hpc/train_stage2_only.sbatch [task] [data_split] [s1_ckpt]
#     task       = pusht | single_grasp | bimanual_sweep | bimanual_rope   (default: pusht)
#     data_split = mini | full                                              (default: full)
#     s1_ckpt    = absolute path to a stage-1 .ckpt                         (default: newest in stage_1/checkpoints/)
#
# Examples:
#   sbatch scripts/hpc/train_stage2_only.sbatch pusht
#   sbatch scripts/hpc/train_stage2_only.sbatch pusht full /scratch/$USER/interactive_world_sim/outputs/pusht/stage_1/checkpoints/epoch=4-step=15000.ckpt

set -euo pipefail

TASK="${1:-pusht}"
DATA_SPLIT="${2:-full}"
S1_CKPT_ARG="${3:-}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

S1_DIR=$REPO/outputs/$TASK/stage_1
S2_DIR=$REPO/outputs/$TASK/stage_2

mkdir -p logs "$S2_DIR"

case "$TASK" in
  pusht)          OBS_KEYS='[camera_1_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  single_grasp)   OBS_KEYS='[camera_0_color]'; ACT_MODE=single_arm;    ACT_DIM=7 ;;
  bimanual_sweep) OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  bimanual_rope)  OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_3d;   ACT_DIM=6 ;;
  *) echo "unknown task: $TASK"; exit 2 ;;
esac

# ---------- pick the stage-1 checkpoint (once, at job start) ----------
if [[ -n "$S1_CKPT_ARG" ]]; then
  S1_CKPT="$S1_CKPT_ARG"
elif [[ -f "$S1_DIR/checkpoints/last.ckpt" ]]; then
  S1_CKPT="$S1_DIR/checkpoints/last.ckpt"
elif compgen -G "$S1_DIR/checkpoints/*.ckpt" > /dev/null; then
  S1_CKPT=$(ls -t "$S1_DIR"/checkpoints/*.ckpt | head -1)
else
  echo "FATAL: no stage-1 checkpoint found in $S1_DIR/checkpoints/"
  exit 1
fi

# Snapshot the ckpt + the matching .hydra/ config into a self-contained
# subdir that mimics the run-dir layout (<dir>/checkpoints/<name>.ckpt next
# to <dir>/.hydra/). latent_world_model.py:_build_model resolves the load_ae
# config from `dirname(dirname(ckpt))/.hydra/config.yaml`, so this layout is
# required. The original S1 .hydra/ dir is copied (not symlinked) so the
# still-running stage-1 job can keep updating its own files without affecting
# us.
SNAPSHOT_DIR=$S2_DIR/_s1_seed
SNAPSHOT=$SNAPSHOT_DIR/checkpoints/$(basename "$S1_CKPT")
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "[$(date)] snapshotting $S1_CKPT -> $SNAPSHOT"
  mkdir -p "$SNAPSHOT_DIR/checkpoints"
  cp "$S1_CKPT" "$SNAPSHOT"
  if [[ -d "$S1_DIR/.hydra" ]]; then
    cp -r "$S1_DIR/.hydra" "$SNAPSHOT_DIR/.hydra"
  else
    echo "FATAL: expected $S1_DIR/.hydra/ to exist for load_ae config resolution"
    exit 1
  fi
fi
S1_CKPT="$SNAPSHOT"

# ---------- find latest s2 ckpt for in-progress resume ----------
RESUME_FLAG=""
if [[ -f "$S2_DIR/checkpoints/last.ckpt" ]]; then
  RESUME_FLAG="load=$S2_DIR/checkpoints/last.ckpt"
elif compgen -G "$S2_DIR/checkpoints/*.ckpt" > /dev/null; then
  RESUME_FLAG="load=$(ls -t "$S2_DIR"/checkpoints/*.ckpt | head -1)"
fi

# ---------- signal handler: clean requeue ----------
requeue_self() {
  echo "[$(date)] SIGUSR1 received — requeueing job $SLURM_JOB_ID"
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill -SIGTERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_self USR1

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  task=$TASK  split=$DATA_SPLIT"
echo "  S1 seed  : $S1_CKPT"
echo "  S2 dir   : $S2_DIR"
echo "  resume   : ${RESUME_FLAG:-<none>}"
echo "================================================================"
nvidia-smi || true

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" \
  /bin/bash -c "
    set -e
    source /ext3/env.sh
    conda activate iws
    cd $REPO
    export HDF5_USE_FILE_LOCKING=FALSE
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    export WANDB_API_KEY=\$(cat ~/.secrets/wandb_api_key 2>/dev/null || echo '')

    python main.py +name=${TASK}_stage_2 \
      algorithm=latent_world_model \
      experiment=exp_latent_dyn \
      dataset=real_aloha_dataset \
      dataset.dataset_dir=data/$DATA_SPLIT/$TASK \
      dataset.horizon=10 dataset.val_horizon=200 \
      dataset.obs_keys=$OBS_KEYS \
      dataset.action_mode=$ACT_MODE \
      hydra.run.dir=$S2_DIR \
      experiment.training.batch_size=64 \
      experiment.training.max_steps=1000005 \
      experiment.training.log_every_n_steps=100 \
      experiment.training.checkpointing.every_n_train_steps=10000 \
      experiment.training.data.num_workers=4 \
      experiment.validation.batch_size=2 \
      experiment.validation.val_every_n_step=500 \
      experiment.validation.limit_batch=1.0 \
      experiment.validation.data.num_workers=4 \
      algorithm.latent_dim=512 \
      algorithm.action_dim=$ACT_DIM \
      algorithm.noise_scheduler.loss_weighting=uniform \
      algorithm.sampling_strategy=terminal_only \
      algorithm.load_ae=$S1_CKPT \
      algorithm.training_stage=2 \
      +experiment.training.checkpointing.save_last=True \
      $RESUME_FLAG
  " &
TRAIN_PID=$!
wait "$TRAIN_PID"
echo "[$(date)] Stage 2 finished"
