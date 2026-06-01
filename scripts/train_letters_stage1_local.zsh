#!/bin/zsh
set -euo pipefail

GPU=0
SOURCE_DIR="data/mujoco/letters"
DATASET_DIR="data/mujoco/letters_prepared/stage1_letters"
RUN_DIR="outputs/letters/stage_1"
RUN_NAME="letters_stage_1"
LETTERS=""
EXCLUDE_LETTERS=""
MOTIONS=""
PREPARE_DATASET="if_missing"  # if_missing | always | never
VAL_RATIO=0.1
SEED=42
MIN_FRAMES=1
BATCH_SIZE=16
MAX_STEPS=200005
CHECKPOINT_EVERY=10000
VAL_EVERY=6000
NUM_WORKERS=4
WANDB_MODE="online"
RESUME=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_letters_stage1_local.zsh [options]

Purpose:
  Prepare procedural-letter data for SimAlohaDataset and train only stage 1
  (autoencoder) of the latent world model on a local single GPU.

Data input:
  Raw episodes are expected under:
    data/mujoco/letters/<LETTER>/<MOTION>/episode_*.hdf5

  The script prepares a flat dataset at --dataset_dir:
    <dataset_dir>/train/episode_*.hdf5
    <dataset_dir>/val/episode_*.hdf5

Options:
  --gpu 0                         Local GPU index. Default: 0
  --source_dir PATH               Raw nested letters dir. Default: data/mujoco/letters
  --dataset_dir PATH              Prepared flat dataset dir. Default: data/mujoco/letters_prepared/stage1_letters
  --run_dir PATH                  Hydra output dir. Default: outputs/letters/stage_1
  --run_name NAME                 Wandb/Hydra run name. Default: letters_stage_1
  --letters "A B C"               Letters to include. Default: all available letters
  --exclude_letters "Q X Z"       Letters to exclude from selected/default set
  --motions "mixed rl_coverage"   Motions to include. Default: all available known motions
  --prepare_dataset MODE          if_missing | always | never. Default: if_missing
  --val_ratio 0.1                 Per letter/motion validation split. Default: 0.1
  --seed 42                       Dataset split seed. Default: 42
  --min_frames 1                  Ignore episodes shorter than this. Default: 1
  --batch_size 16                 Training batch size. Default: 16
  --max_steps 200005              Training max steps. Default: 200005
  --checkpoint_every 10000        Checkpoint interval. Default: 10000
  --val_every 6000                Validation interval. Default: 6000
  --num_workers 4                 Train/val dataloader workers. Default: 4
  --wandb_mode online             online | offline | disabled. Default: online
  --resume 0|1                    Resume from run_dir/checkpoints/last.ckpt if present. Default: 1
  --dry_run 0|1                   Print commands without running. Default: 0
  --help                          Show this help.

Examples:
  # Stage 1 on all available letters/motions, GPU 0
  ./scripts/train_letters_stage1_local.zsh --gpu 0

  # Stage 1 on all letters except held-out letters
  ./scripts/train_letters_stage1_local.zsh \
    --gpu 0 \
    --dataset_dir data/mujoco/letters_prepared/stage1_no_QXZ \
    --run_dir outputs/letters/stage_1_no_QXZ \
    --run_name letters_stage_1_no_QXZ \
    --exclude_letters "Q X Z"

  # Stage 1 on explicit letters and only mixed/RL data
  ./scripts/train_letters_stage1_local.zsh \
    --gpu 1 \
    --letters "A B C D" \
    --motions "mixed rl_coverage" \
    --dataset_dir data/mujoco/letters_prepared/stage1_ABCD_mixed_rl \
    --run_dir outputs/letters/stage_1_ABCD_mixed_rl
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --source_dir) SOURCE_DIR="$2"; shift 2 ;;
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --letters) LETTERS="$2"; shift 2 ;;
    --exclude_letters) EXCLUDE_LETTERS="$2"; shift 2 ;;
    --motions) MOTIONS="$2"; shift 2 ;;
    --prepare_dataset) PREPARE_DATASET="$2"; shift 2 ;;
    --val_ratio) VAL_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --min_frames) MIN_FRAMES="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --checkpoint_every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --val_every) VAL_EVERY="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --wandb_mode) WANDB_MODE="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$PREPARE_DATASET" in
  if_missing|always|never) ;;
  *) echo "ERROR: --prepare_dataset must be if_missing, always, or never" >&2; exit 1 ;;
esac

case "$WANDB_MODE" in
  online|offline|disabled|dryrun) ;;
  *) echo "ERROR: --wandb_mode must be online, offline, disabled, or dryrun" >&2; exit 1 ;;
esac

mkdir -p logs "$(dirname "$DATASET_DIR")" "$RUN_DIR"
export PYTHONPATH="external/gym-aloha:.${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

prepare_args=(
  python scripts/prepare_letters_dataset.py
  --source_dir "$SOURCE_DIR"
  --output_dir "$DATASET_DIR"
  --val_ratio "$VAL_RATIO"
  --seed "$SEED"
  --min_frames "$MIN_FRAMES"
)
if [[ -n "$LETTERS" ]]; then
  prepare_args+=(--letters "$LETTERS")
fi
if [[ -n "$EXCLUDE_LETTERS" ]]; then
  prepare_args+=(--exclude_letters "$EXCLUDE_LETTERS")
fi
if [[ -n "$MOTIONS" ]]; then
  prepare_args+=(--motions "$MOTIONS")
fi

metadata_path="$DATASET_DIR/metadata.json"
case "$PREPARE_DATASET" in
  always)
    prepare_args+=(--overwrite)
    ;;
  if_missing)
    if [[ -f "$metadata_path" ]]; then
      echo "[$(date +%H:%M:%S)] Reusing prepared dataset: $DATASET_DIR"
      prepare_args=()
    elif [[ -e "$DATASET_DIR" ]]; then
      prepare_args+=(--overwrite)
    fi
    ;;
  never)
    if [[ ! -f "$metadata_path" ]]; then
      echo "ERROR: --prepare_dataset never but metadata not found: $metadata_path" >&2
      exit 1
    fi
    prepare_args=()
    ;;
esac

if (( ${#prepare_args} > 0 )); then
  echo "[$(date +%H:%M:%S)] Preparing letters dataset"
  printf '  %q' "${prepare_args[@]}"
  echo ""
  if (( ! DRY_RUN )); then
    "${prepare_args[@]}"
  fi
fi

check_args=(
  python scripts/check_letters_dataset_distribution.py
  --dataset_dir "$DATASET_DIR"
  --fail_if_empty
)
echo "[$(date +%H:%M:%S)] Checking prepared dataset distribution"
printf '  %q' "${check_args[@]}"
echo ""
if (( ! DRY_RUN )); then
  "${check_args[@]}"
fi

resume_args=()
if (( RESUME )) && [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
  resume_args+=("load=$RUN_DIR/checkpoints/last.ckpt")
  echo "[$(date +%H:%M:%S)] Resuming Stage 1 from $RUN_DIR/checkpoints/last.ckpt"
else
  echo "[$(date +%H:%M:%S)] Starting Stage 1 fresh"
fi

train_args=(
  python main.py
  "+name=$RUN_NAME"
  algorithm=latent_world_model
  experiment=exp_latent_dyn
  dataset=sim_aloha_dataset
  "dataset.dataset_dir=$DATASET_DIR"
  dataset.horizon=1
  dataset.val_horizon=1
  "dataset.obs_keys=[top_pov]"
  dataset.use_cache=true
  "hydra.run.dir=$RUN_DIR"
  "experiment.training.batch_size=$BATCH_SIZE"
  "experiment.training.max_steps=$MAX_STEPS"
  experiment.training.log_every_n_steps=100
  "experiment.training.checkpointing.every_n_train_steps=$CHECKPOINT_EVERY"
  "experiment.training.data.num_workers=$NUM_WORKERS"
  experiment.validation.batch_size=10
  "experiment.validation.val_every_n_step=$VAL_EVERY"
  experiment.validation.limit_batch=1.0
  "experiment.validation.data.num_workers=$NUM_WORKERS"
  algorithm.latent_dim=512
  algorithm.action_dim=4
  algorithm.training_stage=1
  +experiment.training.checkpointing.save_last=True
  "wandb.mode=$WANDB_MODE"
  "${resume_args[@]}"
)

echo "[$(date +%H:%M:%S)] Launching local Stage 1 training on GPU $GPU"
printf '  CUDA_VISIBLE_DEVICES=%q' "$GPU"
printf ' %q' "${train_args[@]}"
echo ""

if (( ! DRY_RUN )); then
  CUDA_VISIBLE_DEVICES="$GPU" "${train_args[@]}"
fi
