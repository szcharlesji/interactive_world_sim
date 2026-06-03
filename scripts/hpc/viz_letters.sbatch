#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-letters-viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/letters_viz_%j.out
#SBATCH --error=logs/letters_viz_%j.err
#
# Stage 2 rollout visualization for the letter generalization experiment.
# Loads a checkpoint and runs the `validation` task on either train or val
# split, logging predicted-vs-GT rollout videos to a NEW wandb run.
#
# No training is performed.
#
# Usage:
#   sbatch scripts/hpc/viz_letters.sbatch <run> [ckpt] [viz_split]
#
#   run       = all_letters | s1_all_s2_train20 | s1_train20_s2_train20
#   ckpt      = absolute path to a compatible .ckpt (default: newest in stage_2/checkpoints/)
#   viz_split = train | val (default: val)
#
# Examples:
#   sbatch scripts/hpc/viz_letters.sbatch all_letters
#   sbatch scripts/hpc/viz_letters.sbatch all_letters "" train
#   sbatch scripts/hpc/viz_letters.sbatch s1_all_s2_train20 /path/to/ckpt val

set -euo pipefail

RUN="${1:-all_letters}"
S2_CKPT_ARG="${2:-}"
VIZ_SPLIT="${3:-val}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

S2_DIR=$REPO/outputs/letters/$RUN/stage_2

case "$RUN" in
  all_letters)
    DATA_DIR="data/letters/all_letters"
    ;;
  s1_all_s2_train20|s1_train20_s2_train20)
    DATA_DIR="data/letters/train20_val6"
    ;;
  *)
    echo "ERROR: unknown run '$RUN'"
    exit 2
    ;;
esac

case "$VIZ_SPLIT" in
  train) VIZ_DATA_DIR="$DATA_DIR/train"; VIZ_NAME=trainviz ;;
  val)   VIZ_DATA_DIR="$DATA_DIR/val";   VIZ_NAME=valviz ;;
  *) echo "unknown viz split: $VIZ_SPLIT (expected train or val)"; exit 2 ;;
esac

VIZ_DIR=$S2_DIR/$VIZ_NAME
mkdir -p logs "$VIZ_DIR"

# ---------- pick the stage-2 checkpoint ----------
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

# ---------- find the S1 seed ----------
if [[ -f "$S2_DIR/_s1_seed/checkpoints/last.ckpt" ]]; then
  S1_SEED_CKPT="$S2_DIR/_s1_seed/checkpoints/last.ckpt"
elif compgen -G "$S2_DIR/_s1_seed/checkpoints/*.ckpt" > /dev/null; then
  S1_SEED_CKPT=$(ls -t "$S2_DIR"/_s1_seed/checkpoints/*.ckpt | head -1)
else
  echo "FATAL: no S1 seed ckpt found in $S2_DIR/_s1_seed/checkpoints/"
  exit 1
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  run=$RUN  viz_split=$VIZ_SPLIT"
echo "  ckpt     : $S2_CKPT"
echo "  S1 seed  : $S1_SEED_CKPT"
echo "  viz dir  : $VIZ_DIR"
echo "  viz data : $VIZ_DATA_DIR"
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

    python main.py +name=letters_${RUN}_stage_2_${VIZ_NAME} \
      experiment.tasks=[validation] \
      algorithm=latent_world_model \
      experiment=exp_latent_dyn \
      dataset=sim_aloha_dataset \
      dataset.dataset_dir=$DATA_DIR \
      dataset.val_dataset_dir=$VIZ_DATA_DIR \
      dataset.horizon=10 dataset.val_horizon=200 \
      dataset.obs_keys=[top_pov] \
      hydra.run.dir=$VIZ_DIR \
      experiment.validation.batch_size=1 \
      experiment.validation.limit_batch=1 \
      experiment.validation.data.num_workers=4 \
      algorithm.latent_dim=512 \
      algorithm.action_dim=4 \
      algorithm.noise_scheduler.loss_weighting=uniform \
      algorithm.sampling_strategy=terminal_only \
      algorithm.load_ae=$S1_SEED_CKPT \
      algorithm.training_stage=2 \
      load=$S2_CKPT
  "
echo "[$(date)] Stage 2 $VIZ_SPLIT-set visualization finished"
