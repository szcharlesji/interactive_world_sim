#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-s3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/s3_%j.out
#SBATCH --error=logs/s3_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 3 only — Autoencoder finetune.
# Snapshots a stage-2 checkpoint at job start and finetunes the
# autoencoder from it. Safe to run in parallel with a still-going
# stage-2 job; writes to outputs/<task>/stage_3/ which is disjoint
# from stage_2.
#
# Usage:
#   sbatch scripts/hpc/train_stage3_only.sbatch [task] [data_split] [s2_ckpt] [run_suffix]
#     task       = pusht | single_grasp | bimanual_sweep | bimanual_rope
#     data_split = mini | full
#     s2_ckpt    = absolute path to a stage-2 .ckpt (default: newest in stage_2/checkpoints/)
#     run_suffix = optional suffix for parallel runs (e.g. "b" -> stage_3_b/)
#
# Examples:
#   sbatch scripts/hpc/train_stage3_only.sbatch pusht
#   sbatch scripts/hpc/train_stage3_only.sbatch pusht full /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2/checkpoints/last.ckpt
#   sbatch -J iws-s3-b scripts/hpc/train_stage3_only.sbatch pusht full "" b

set -euo pipefail

TASK="${1:-pusht}"
DATA_SPLIT="${2:-full}"
S2_CKPT_ARG="${3:-}"
RUN_SUFFIX="${4:-}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

S2_DIR=$REPO/outputs/$TASK/stage_2
S3_DIR=$REPO/outputs/$TASK/stage_3${RUN_SUFFIX:+_$RUN_SUFFIX}

mkdir -p logs "$S3_DIR"

case "$TASK" in
  pusht)          OBS_KEYS='[camera_1_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  single_grasp)   OBS_KEYS='[camera_0_color]'; ACT_MODE=single_arm;    ACT_DIM=7 ;;
  bimanual_sweep) OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  bimanual_rope)  OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_3d;   ACT_DIM=6 ;;
  *) echo "unknown task: $TASK"; exit 2 ;;
esac

# ---------- pick the stage-2 checkpoint (once, at job start) ----------
if [[ -n "$S2_CKPT_ARG" ]]; then
  S2_CKPT="$S2_CKPT_ARG"
elif [[ -f "$S2_DIR/checkpoints/last.ckpt" ]]; then
  S2_CKPT="$S2_DIR/checkpoints/last.ckpt"
elif compgen -G "$S2_DIR/checkpoints/*.ckpt" > /dev/null; then
  S2_CKPT=$(ls -t "$S2_DIR"/checkpoints/*.ckpt | head -1)
else
  echo "FATAL: no stage-2 checkpoint found in $S2_DIR/checkpoints/"
  exit 1
fi

# Snapshot the ckpt + matching .hydra/ config into a self-contained subdir
# (same pattern as train_stage2_only.sbatch, but from stage_2 source).
# latent_world_model.py:_build_model resolves load_ae config from
# dirname(dirname(ckpt))/.hydra/config.yaml, so this layout is required.
SNAPSHOT_DIR=$S3_DIR/_s2_seed
SNAPSHOT=$SNAPSHOT_DIR/checkpoints/$(basename "$S2_CKPT")
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "[$(date)] snapshotting $S2_CKPT -> $SNAPSHOT"
  mkdir -p "$SNAPSHOT_DIR/checkpoints"
  cp "$S2_CKPT" "$SNAPSHOT"
  if [[ -d "$S2_DIR/.hydra" ]]; then
    cp -r "$S2_DIR/.hydra" "$SNAPSHOT_DIR/.hydra"
  else
    echo "FATAL: expected $S2_DIR/.hydra/ to exist for load_ae config resolution"
    exit 1
  fi
fi
S2_CKPT="$SNAPSHOT"

# ---------- find latest s3 ckpt for in-progress resume ----------
RESUME_FLAG=""
if [[ -f "$S3_DIR/checkpoints/last.ckpt" ]]; then
  RESUME_FLAG="load=$S3_DIR/checkpoints/last.ckpt"
elif compgen -G "$S3_DIR/checkpoints/*.ckpt" > /dev/null; then
  RESUME_FLAG="load=$(ls -t "$S3_DIR"/checkpoints/*.ckpt | head -1)"
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
echo "  task=$TASK  split=$DATA_SPLIT  suffix=${RUN_SUFFIX:-<none>}"
echo "  S2 seed  : $S2_CKPT"
echo "  S3 dir   : $S3_DIR"
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

    python main.py +name=${TASK}_stage_3${RUN_SUFFIX:+_$RUN_SUFFIX} \
      algorithm=latent_world_model \
      experiment=exp_latent_dyn \
      dataset=real_aloha_dataset \
      dataset.dataset_dir=data/$DATA_SPLIT/$TASK \
      dataset.horizon=1 dataset.val_horizon=200 \
      dataset.obs_keys=$OBS_KEYS \
      dataset.action_mode=$ACT_MODE \
      hydra.run.dir=$S3_DIR \
      experiment.training.batch_size=32 \
      experiment.training.max_steps=1000005 \
      experiment.training.log_every_n_steps=100 \
      experiment.training.checkpointing.every_n_train_steps=10000 \
      experiment.training.data.num_workers=4 \
      experiment.validation.batch_size=2 \
      experiment.validation.val_every_n_step=30000 \
      experiment.validation.limit_batch=1.0 \
      experiment.validation.data.num_workers=4 \
      algorithm.latent_dim=512 \
      algorithm.action_dim=$ACT_DIM \
      algorithm.noise_scheduler.loss_weighting=uniform \
      algorithm.sampling_strategy=terminal_only \
      algorithm.load_ae=$S2_CKPT \
      algorithm.training_stage=3 \
      +experiment.training.checkpointing.save_last=True \
      $RESUME_FLAG
  " &
TRAIN_PID=$!
wait "$TRAIN_PID"
echo "[$(date)] Stage 3 finished"
