#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=a100
#SBATCH --job-name=iws-merge-s2s3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_s2s3_%j.out
#SBATCH --error=logs/merge_s2s3_%j.err
#
# Merge a continued stage-2 checkpoint with a stage-3 finetuned decoder inside
# the project container. The default arguments match the PushT stage_2_c merge:
#
#   encoder+dynamics from outputs/pusht/stage_2_c/checkpoints/last.ckpt
#   decoder          from outputs/pusht/stage_3/checkpoints/last.ckpt
#   output           to   outputs/pusht/stage_2_s3_decoder/checkpoints/last.ckpt
#
# Usage:
#   sbatch scripts/hpc/merge_stage2_stage3_checkpoint.sbatch [stage2_ckpt] [stage3_ckpt] [out_dir_or_ckpt]
#
# Examples:
#   sbatch scripts/hpc/merge_stage2_stage3_checkpoint.sbatch
#   sbatch scripts/hpc/merge_stage2_stage3_checkpoint.sbatch \
#     /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_c/checkpoints/last.ckpt \
#     /scratch/$USER/interactive_world_sim/outputs/pusht/stage_3/checkpoints/last.ckpt \
#     /scratch/$USER/interactive_world_sim/outputs/pusht/stage_2_s3_decoder

set -euo pipefail

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

STAGE2_CKPT="${1:-$REPO/outputs/pusht/stage_2_c/checkpoints/last.ckpt}"
STAGE3_CKPT="${2:-$REPO/outputs/pusht/stage_3/checkpoints/last.ckpt}"
OUT="${3:-$REPO/outputs/pusht/stage_2_s3_decoder}"

mkdir -p logs

if [[ ! -f "$STAGE2_CKPT" ]]; then
  echo "FATAL: stage-2 checkpoint not found: $STAGE2_CKPT"
  exit 1
fi
if [[ ! -f "$STAGE3_CKPT" ]]; then
  echo "FATAL: stage-3 checkpoint not found: $STAGE3_CKPT"
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME=singularity
else
  echo "FATAL: neither apptainer nor singularity found on PATH"
  exit 1
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  partition : a100"
echo "  runtime   : $CONTAINER_RUNTIME"
echo "  S2 ckpt   : $STAGE2_CKPT"
echo "  S3 ckpt   : $STAGE3_CKPT"
echo "  output    : $OUT"
echo "================================================================"
nvidia-smi || true

"$CONTAINER_RUNTIME" exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" \
  /bin/bash -c "
    set -e
    source /ext3/env.sh
    conda activate iws
    cd $REPO

    python scripts/merge_stage2_stage3_checkpoint.py \
      --stage2_ckpt $STAGE2_CKPT \
      --stage3_ckpt $STAGE3_CKPT \
      --out $OUT \
      --hydra_source stage2 \
      --patch_hydra_training_stage 2
  "

echo "[$(date)] Stage 2/3 checkpoint merge finished"
