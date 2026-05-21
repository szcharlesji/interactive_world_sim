#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-s2-viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s2viz_%j.out
#SBATCH --error=logs/s2viz_%j.err
#
# Stage 2 visualization on the TRAINING set.
# Loads a stage-2 checkpoint and runs the `validation` task pointed at the
# train episodes, logging predicted-vs-GT rollout videos to a NEW wandb run
# named <task>_stage_2[_suffix]_trainviz (separate from the training run).
#
# No training is performed. Reads from outputs/<task>/stage_2[_suffix]/checkpoints/
# and writes nothing back there except the disjoint trainviz/ run dir.
#
# Usage:
#   sbatch scripts/hpc/viz_stage2_trainset.sbatch [task] [data_split] [s2_ckpt] [run_suffix]
#     task       = pusht | single_grasp | bimanual_sweep | bimanual_rope   (default: pusht)
#     data_split = mini | full                                              (default: full)
#     s2_ckpt    = absolute path to a stage-2 .ckpt                         (default: newest in stage_2/checkpoints/)
#     run_suffix = stage_2 dir suffix to target (matches the trainer's suffix arg).
#                  Empty -> outputs/<task>/stage_2/.  Suffix "b" -> outputs/<task>/stage_2_b/.
#
# Examples:
#   sbatch scripts/hpc/viz_stage2_trainset.sbatch pusht
#   sbatch scripts/hpc/viz_stage2_trainset.sbatch pusht full "" b

set -euo pipefail

TASK="${1:-pusht}"
DATA_SPLIT="${2:-full}"
S2_CKPT_ARG="${3:-}"
RUN_SUFFIX="${4:-}"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

S2_DIR=$REPO/outputs/$TASK/stage_2${RUN_SUFFIX:+_$RUN_SUFFIX}
VIZ_DIR=$S2_DIR/trainviz

mkdir -p logs "$VIZ_DIR"

case "$TASK" in
  pusht)          OBS_KEYS='[camera_1_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  single_grasp)   OBS_KEYS='[camera_0_color]'; ACT_MODE=single_arm;    ACT_DIM=7 ;;
  bimanual_sweep) OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_push; ACT_DIM=4 ;;
  bimanual_rope)  OBS_KEYS='[camera_0_color]'; ACT_MODE=bimanual_3d;   ACT_DIM=6 ;;
  *) echo "unknown task: $TASK"; exit 2 ;;
esac

# ---------- pick the stage-2 checkpoint to visualize ----------
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

# ---------- find the S1 seed used by this stage-2 run ----------
# load_ae is only needed so the model builds (config resolves from
# dirname(dirname(ckpt))/.hydra/config.yaml); ckpt_path then overrides the
# weights with the full stage-2 state. The trainer snapshotted the S1 ckpt +
# .hydra into stage_2/_s1_seed/.
if [[ -f "$S2_DIR/_s1_seed/checkpoints/last.ckpt" ]]; then
  S1_SEED_CKPT="$S2_DIR/_s1_seed/checkpoints/last.ckpt"
elif compgen -G "$S2_DIR/_s1_seed/checkpoints/*.ckpt" > /dev/null; then
  S1_SEED_CKPT=$(ls -t "$S2_DIR"/_s1_seed/checkpoints/*.ckpt | head -1)
else
  echo "FATAL: no S1 seed ckpt found in $S2_DIR/_s1_seed/checkpoints/"
  echo "       (expected the snapshot created by train_stage2_only.sbatch)"
  exit 1
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  task=$TASK  split=$DATA_SPLIT  suffix=${RUN_SUFFIX:-<none>}"
echo "  S2 ckpt  : $S2_CKPT"
echo "  S1 seed  : $S1_SEED_CKPT"
echo "  viz dir  : $VIZ_DIR"
echo "  val data : data/$DATA_SPLIT/$TASK/train  (TRAINING episodes)"
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

    python main.py +name=${TASK}_stage_2${RUN_SUFFIX:+_$RUN_SUFFIX}_trainviz \
      experiment.tasks=[validation] \
      algorithm=latent_world_model \
      experiment=exp_latent_dyn \
      dataset=real_aloha_dataset \
      dataset.dataset_dir=data/$DATA_SPLIT/$TASK \
      dataset.val_dataset_dir=data/$DATA_SPLIT/$TASK/train \
      dataset.horizon=10 dataset.val_horizon=200 \
      dataset.obs_keys=$OBS_KEYS \
      dataset.action_mode=$ACT_MODE \
      hydra.run.dir=$VIZ_DIR \
      experiment.validation.batch_size=8 \
      experiment.validation.limit_batch=1 \
      experiment.validation.data.num_workers=4 \
      algorithm.latent_dim=512 \
      algorithm.action_dim=$ACT_DIM \
      algorithm.noise_scheduler.loss_weighting=uniform \
      algorithm.sampling_strategy=terminal_only \
      algorithm.load_ae=$S1_SEED_CKPT \
      algorithm.training_stage=2 \
      load=$S2_CKPT
  "
echo "[$(date)] Stage 2 train-set visualization finished"
