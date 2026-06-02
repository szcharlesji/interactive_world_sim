#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-letters-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/letters_s1_%j.out
#SBATCH --error=logs/letters_s1_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 1 (autoencoder) for the letter generalization experiment on H200.
# Downloads the procedural-letter sim dataset from HuggingFace, then trains
# stage 1 only. Stage 2 is run separately with train_letters_stage2.sbatch.
#
# Usage:
#   sbatch scripts/hpc/train_letters_stage1.sbatch <run>
#
#   run = all_letters          Baseline: S1 on all 26 letters
#       | train20_val6         Heldout: S1 on 20 letters, val on 6 heldout
#
# Prereqs:
#   - Dataset uploaded to HF: szcharlesji/push-letters-sim
#     with subdirectories: all_letters_50/ and train20_val6_50/
#   - $SCRATCH/iws.ext3 built (conda overlay with iws env)
#   - ~/.secrets/wandb_api_key (chmod 600)
#   - Repo cloned to /scratch/$USER/interactive_world_sim

set -euo pipefail

RUN="${1:-all_letters}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HF_DATASET_REPO="szcharlesji/push-letters-sim"

case "$RUN" in
  all_letters)
    DATASET_SUBDIR="all_letters_50"
    ;;
  train20_val6)
    DATASET_SUBDIR="train20_val6_50"
    ;;
  *)
    echo "ERROR: unknown run '$RUN'"
    echo "  valid: all_letters | train20_val6"
    exit 2
    ;;
esac

S1_DIR=$REPO/outputs/letters/$RUN/stage_1
S1_DONE=$REPO/outputs/letters/$RUN/_stage1_done

mkdir -p logs "$S1_DIR"

# ---------- signal handler ----------
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

# ---------- apptainer runner ----------
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
  local dir="$1/checkpoints"
  if [[ -f "$dir/last.ckpt" ]]; then
    echo "$dir/last.ckpt"
  elif compgen -G "$dir/*.ckpt" > /dev/null; then
    ls -t "$dir"/*.ckpt | head -1
  fi
}

# ============================================================
# Download dataset
# ============================================================
DATA_DIR="data/letters/${DATASET_SUBDIR}"
if [[ ! -d "$DATA_DIR/train" ]]; then
  echo "[$(date)] Downloading $HF_DATASET_REPO/$DATASET_SUBDIR → $DATA_DIR"
  python scripts/download_data_hf.py \
    --repo "$HF_DATASET_REPO" \
    --local_dir "$DATA_DIR" \
    --repo_dir "$DATASET_SUBDIR"
else
  echo "[$(date)] Dataset already downloaded: $DATA_DIR"
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  run=$RUN  dataset=$DATA_DIR"
echo "  S1_DIR=$S1_DIR  done=$( [[ -f $S1_DONE ]] && echo yes || echo no )"
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

  CMD_S1="python main.py +name=letters_${RUN}_stage_1 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=sim_aloha_dataset \
    dataset.dataset_dir=$DATA_DIR \
    dataset.horizon=1 dataset.val_horizon=1 \
    dataset.obs_keys=[top_pov] \
    hydra.run.dir=$S1_DIR \
    experiment.training.batch_size=32 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.validation.batch_size=10 \
    experiment.validation.val_every_n_step=6000 \
    experiment.validation.limit_batch=1.0 \
    algorithm.latent_dim=512 \
    algorithm.action_dim=4 \
    algorithm.training_stage=1 \
    +experiment.training.checkpointing.save_last=True \
    $RESUME_FLAG"

  run_train "$CMD_S1"
  echo "[$(date)] Stage 1 finished cleanly"
  touch "$S1_DONE"
else
  echo "[$(date)] Stage 1 already done — skipping"
fi

echo "[$(date)] Stage 1 complete for run=$RUN"
