#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --signal=B:USR1@180
#
# Train all three stages (S1 autoencoder -> S2 dynamics -> S3 finetune)
# in a SINGLE slurm job. Resumable: state files mark stage completion;
# inside a stage, last.ckpt is used to resume mid-run.
#
# Usage:
#   sbatch scripts/hpc/train_all_stages.sbatch [task] [data_split]
#     task       = pusht | single_grasp | bimanual_sweep | bimanual_rope   (default: bimanual_rope)
#     data_split = mini | full                                              (default: full)
#
# Single-stage runs:
#   sbatch scripts/hpc/train_all_stages.sbatch bimanual_rope full
#   # then to force re-running just one stage:
#   rm outputs/<task>/_stage2_done && sbatch ...
#
# Prereqs:
#   - $SCRATCH/iws.ext3 built (conda overlay with iws env)
#   - data downloaded to data/<split>/<task>/
#   - ~/.secrets/wandb_api_key contains your wandb API key (chmod 600)

set -euo pipefail

# ---------- args & paths ----------
TASK="${1:-bimanual_rope}"
DATA_SPLIT="${2:-full}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

RUN_BASE=$REPO/outputs/$TASK
S1_DIR=$RUN_BASE/stage_1
S2_DIR=$RUN_BASE/stage_2
S3_DIR=$RUN_BASE/stage_3
S1_DONE=$RUN_BASE/_stage1_done
S2_DONE=$RUN_BASE/_stage2_done
S3_DONE=$RUN_BASE/_stage3_done

mkdir -p logs "$S1_DIR" "$S2_DIR" "$S3_DIR"

# ---------- per-task config ----------
# Action mode / dim per README; obs key picks which camera view to train on.
# Override on the CLI by editing this block if your task isn't listed.
case "$TASK" in
  pusht)
    OBS_KEYS='[camera_1_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  single_grasp)
    OBS_KEYS='[camera_0_color]'; ACT_MODE=single_arm;    ACT_DIM=7 ;;
  bimanual_sweep)
    OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  bimanual_rope)
    OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_3d;   ACT_DIM=6 ;;
  *)
    echo "unknown task: $TASK"; exit 2 ;;
esac

# ---------- signal handler: clean requeue ----------
# Slurm sends SIGUSR1 180s before --time; trap it, requeue ourselves, exit 0.
# The state files + last.ckpt logic below restart at the right point next time.
requeue_self() {
  echo "[$(date)] SIGUSR1 received — requeueing job $SLURM_JOB_ID"
  if [[ -n "${TRAIN_PID:-}" ]]; then
    # Give the python process a moment to flush its last checkpoint
    kill -SIGTERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_self USR1

# ---------- shared apptainer runner ----------
# Args:
#   $1 = the python command (already fully expanded), as a single string
# Runs in background; sets $TRAIN_PID so the USR1 trap can SIGTERM it.
run_train() {
  local cmd="$1"
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
      $cmd
    " &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  local rc=$?
  unset TRAIN_PID
  return $rc
}

# ---------- helpers ----------
latest_ckpt() {
  # Echo last.ckpt if it exists, else the newest step-named checkpoint, else nothing.
  local dir="$1/checkpoints"
  if [[ -f "$dir/last.ckpt" ]]; then
    echo "$dir/last.ckpt"
  elif compgen -G "$dir/*.ckpt" > /dev/null; then
    ls -t "$dir"/*.ckpt | head -1
  fi
}

# Common flags injected into every stage:
#   +experiment.training.checkpointing.save_last=True  => guarantees last.ckpt for resume
#     (uses Hydra '+' prefix because save_last isn't pre-declared in base_pytorch.yaml)
#   hydra.run.dir=<stage dir>                          => keeps each stage's outputs in its own folder
COMMON_HYDRA_FLAGS='+experiment.training.checkpointing.save_last=True'

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  task=$TASK  split=$DATA_SPLIT"
echo "  S1_DIR=$S1_DIR  done=$( [[ -f $S1_DONE ]] && echo yes || echo no )"
echo "  S2_DIR=$S2_DIR  done=$( [[ -f $S2_DONE ]] && echo yes || echo no )"
echo "  S3_DIR=$S3_DIR  done=$( [[ -f $S3_DONE ]] && echo yes || echo no )"
echo "================================================================"
nvidia-smi || true

# ============================================================
# Stage 1 — Autoencoder
# ============================================================
if [[ ! -f $S1_DONE ]]; then
  RESUME_FLAG=""
  S1_LATEST=$(latest_ckpt "$S1_DIR")
  if [[ -n "$S1_LATEST" ]]; then
    RESUME_FLAG="load=$S1_LATEST"
    echo "[$(date)] Stage 1 resuming from $S1_LATEST"
  else
    echo "[$(date)] Stage 1 starting fresh"
  fi

  CMD_S1="python main.py +name=${TASK}_stage_1 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=real_aloha_dataset \
    dataset.dataset_dir=data/$DATA_SPLIT/$TASK \
    dataset.horizon=1 dataset.val_horizon=1 \
    dataset.obs_keys=$OBS_KEYS \
    dataset.action_mode=$ACT_MODE \
    hydra.run.dir=$S1_DIR \
    experiment.training.batch_size=32 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.validation.batch_size=10 \
    experiment.validation.val_every_n_step=6000 \
    experiment.validation.limit_batch=1.0 \
    algorithm.latent_dim=512 \
    algorithm.action_dim=$ACT_DIM \
    algorithm.training_stage=1 \
    $COMMON_HYDRA_FLAGS \
    $RESUME_FLAG"

  run_train "$CMD_S1"
  echo "[$(date)] Stage 1 finished cleanly"
  touch "$S1_DONE"
else
  echo "[$(date)] Stage 1 already done — skipping"
fi

# ============================================================
# Stage 2 — Dynamics
# ============================================================
if [[ ! -f $S2_DONE ]]; then
  S1_CKPT=$(latest_ckpt "$S1_DIR")
  if [[ -z "$S1_CKPT" ]]; then
    echo "FATAL: stage 1 marked done but no checkpoint in $S1_DIR/checkpoints"
    exit 1
  fi

  RESUME_FLAG=""
  S2_LATEST=$(latest_ckpt "$S2_DIR")
  if [[ -n "$S2_LATEST" ]]; then
    RESUME_FLAG="load=$S2_LATEST"
    echo "[$(date)] Stage 2 resuming from $S2_LATEST"
  else
    echo "[$(date)] Stage 2 starting fresh from S1 ckpt $S1_CKPT"
  fi

  CMD_S2="python main.py +name=${TASK}_stage_2 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=real_aloha_dataset \
    dataset.dataset_dir=data/$DATA_SPLIT/$TASK \
    dataset.horizon=10 dataset.val_horizon=200 \
    dataset.obs_keys=$OBS_KEYS \
    dataset.action_mode=$ACT_MODE \
    hydra.run.dir=$S2_DIR \
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
    algorithm.load_ae=$S1_CKPT \
    algorithm.training_stage=2 \
    $COMMON_HYDRA_FLAGS \
    $RESUME_FLAG"

  run_train "$CMD_S2"
  echo "[$(date)] Stage 2 finished cleanly"
  touch "$S2_DONE"
else
  echo "[$(date)] Stage 2 already done — skipping"
fi

# ============================================================
# Stage 3 — Autoencoder finetune
# ============================================================
if [[ ! -f $S3_DONE ]]; then
  S2_CKPT=$(latest_ckpt "$S2_DIR")
  if [[ -z "$S2_CKPT" ]]; then
    echo "FATAL: stage 2 marked done but no checkpoint in $S2_DIR/checkpoints"
    exit 1
  fi

  RESUME_FLAG=""
  S3_LATEST=$(latest_ckpt "$S3_DIR")
  if [[ -n "$S3_LATEST" ]]; then
    RESUME_FLAG="load=$S3_LATEST"
    echo "[$(date)] Stage 3 resuming from $S3_LATEST"
  else
    echo "[$(date)] Stage 3 starting fresh from S2 ckpt $S2_CKPT"
  fi

  CMD_S3="python main.py +name=${TASK}_stage_3 \
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
    $COMMON_HYDRA_FLAGS \
    $RESUME_FLAG"

  run_train "$CMD_S3"
  echo "[$(date)] Stage 3 finished cleanly"
  touch "$S3_DONE"
else
  echo "[$(date)] Stage 3 already done — skipping"
fi

echo "[$(date)] All stages complete for task=$TASK"
