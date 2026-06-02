#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-letters-s2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/letters_s2_%j.out
#SBATCH --error=logs/letters_s2_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 2 (dynamics) for the letter generalization experiment on H200.
# Runs independently of stage 1 — snapshots the latest S1 checkpoint at
# job start. Safe to run in parallel with a still-going S1 job.
#
# Usage:
#   sbatch scripts/hpc/train_letters_stage2.sbatch <run>
#
#   run = all_letters          Baseline: S2 on all 26 letters
#       | s1_all_s2_train20    Generalization: S1 all, S2 on 20 letters
#       | s1_train20_s2_train20  Stronger test: S1+S2 both on 20 letters
#
# Prereqs:
#   - Stage 1 checkpoint exists at outputs/letters/<s1_run>/stage_1/checkpoints/
#   - Dataset uploaded to HF: szcharlesji/push-letters-sim
#   - $SCRATCH/iws.ext3 built (conda overlay with iws env)
#   - ~/.secrets/wandb_api_key (chmod 600)

set -euo pipefail

RUN="${1:-all_letters}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HF_DATASET_REPO="szcharlesji/push-letters-sim"

case "$RUN" in
  all_letters)
    S1_RUN="all_letters"
    DATASET_SUBDIR="all_letters_50"
    ;;
  s1_all_s2_train20)
    S1_RUN="all_letters"
    DATASET_SUBDIR="train20_val6_50"
    ;;
  s1_train20_s2_train20)
    S1_RUN="train20_val6"
    DATASET_SUBDIR="train20_val6_50"
    ;;
  *)
    echo "ERROR: unknown run '$RUN'"
    echo "  valid: all_letters | s1_all_s2_train20 | s1_train20_s2_train20"
    exit 2
    ;;
esac

S1_DIR=$REPO/outputs/letters/$S1_RUN/stage_1
S2_DIR=$REPO/outputs/letters/$RUN/stage_2
S2_DONE=$REPO/outputs/letters/$RUN/_stage2_done

mkdir -p logs "$S2_DIR"

# ---------- pick the stage-1 checkpoint (once, at job start) ----------
S1_CKPT_ARG="${2:-}"
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
# Download dataset for stage 2
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

# ============================================================
# Snapshot S1 ckpt + .hydra config
# ============================================================
SNAPSHOT_DIR=$S2_DIR/_s1_seed
SNAPSHOT=$SNAPSHOT_DIR/checkpoints/$(basename "$S1_CKPT")
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "[$(date)] snapshotting $S1_CKPT → $SNAPSHOT"
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

# ---------- find latest s2 ckpt for resume ----------
RESUME_FLAG=""
S2_LATEST=$(latest_ckpt "$S2_DIR")
if [[ -n "$S2_LATEST" ]]; then
  RESUME_FLAG="load=$S2_LATEST"
  echo "[$(date)] Stage 2 resuming from $S2_LATEST"
else
  echo "[$(date)] Stage 2 starting fresh from S1 ckpt $S1_CKPT"
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  run=$RUN  s1_run=$S1_RUN  dataset=$DATA_DIR"
echo "  S1 seed: $S1_CKPT"
echo "  S2 dir : $S2_DIR"
echo "  resume : ${RESUME_FLAG:-<none>}"
echo "================================================================"
nvidia-smi || true

# ============================================================
# Stage 2 — Dynamics
# ============================================================
if [[ ! -f $S2_DONE ]]; then
  CMD_S2="python main.py +name=letters_${RUN}_stage_2 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=sim_aloha_dataset \
    dataset.dataset_dir=$DATA_DIR \
    dataset.horizon=10 dataset.val_horizon=200 \
    dataset.obs_keys=[top_pov] \
    hydra.run.dir=$S2_DIR \
    experiment.training.batch_size=64 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.training.data.num_workers=4 \
    experiment.validation.batch_size=2 \
    experiment.validation.val_every_n_step=30000 \
    experiment.validation.limit_batch=1.0 \
    experiment.validation.data.num_workers=4 \
    algorithm.latent_dim=512 \
    algorithm.action_dim=4 \
    algorithm.noise_scheduler.loss_weighting=uniform \
    algorithm.sampling_strategy=terminal_only \
    algorithm.load_ae=$S1_CKPT \
    algorithm.training_stage=2 \
    +experiment.training.checkpointing.save_last=True \
    $RESUME_FLAG"

  run_train "$CMD_S2"
  echo "[$(date)] Stage 2 finished cleanly"
  touch "$S2_DONE"
else
  echo "[$(date)] Stage 2 already done — skipping"
fi

echo "[$(date)] Stage 2 complete for run=$RUN"
