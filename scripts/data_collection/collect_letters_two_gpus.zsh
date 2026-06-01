#!/bin/zsh
set -u

LETTERS="V W"
MOTIONS="linear rotating random_contact random_no_contact mixed rl_coverage"
GPUS="0 0 1 1 0 1"
EPISODE_STEPS=600
MOTION_SPEEDUP=2
RL_WARMUP_EPISODES=0
RL_CHECKPOINT_INTERVAL=1

usage() {
  cat <<'EOF'
Usage:
  ./scripts/data_collection/collect_letters_two_gpus.zsh [options]

Options:
  --letters "V W"          Letters to collect. Default: "V W"
  --motions "..."          Motions to collect. Default:
                            "linear rotating random_contact random_no_contact mixed rl_coverage"
  --gpus "0 0 1 1 0 1"     GPU per motion. Default: "0 0 1 1 0 1"
                            order follows --motions
  --episode_steps 600      Frames per saved episode/video. Default: 600
  --motion_speedup 2       Per-segment trajectory speedup. Default: 2
  --rl_warmup_episodes 0   For rl_coverage: full episode horizons to learn
                            before saving successful HDF5/MP4 episodes.
                            Default: 0
  --rl_checkpoint_interval 1
                            For rl_coverage: save policy checkpoint every N
                            full episode horizons. Default: 1
  --help                   Show this help.

Example:
  ./scripts/data_collection/collect_letters_two_gpus.zsh \
    --letters "V W Z" \
    --motions "rl_coverage" \
    --gpus "0" \
    --episode_steps 600 \
    --motion_speedup 2 \
    --rl_warmup_episodes 10 \
    --rl_checkpoint_interval 1
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --letters)
      LETTERS="$2"
      shift 2
      ;;
    --motions)
      MOTIONS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --episode_steps)
      EPISODE_STEPS="$2"
      shift 2
      ;;
    --motion_speedup)
      MOTION_SPEEDUP="$2"
      shift 2
      ;;
    --rl_warmup_episodes)
      RL_WARMUP_EPISODES="$2"
      shift 2
      ;;
    --rl_checkpoint_interval)
      RL_CHECKPOINT_INTERVAL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

letters=(${=LETTERS})
motions=(${=MOTIONS})
gpus=(${=GPUS})

if (( ${#letters} == 0 )); then
  echo "ERROR: --letters cannot be empty" >&2
  exit 1
fi

if (( ${#motions} == 0 )); then
  echo "ERROR: --motions cannot be empty" >&2
  exit 1
fi

for motion in "${motions[@]}"; do
  case "$motion" in
    linear|rotating|random_contact|random_no_contact|mixed|rl_coverage)
      ;;
    *)
      echo "ERROR: unknown motion '${motion}'. Valid: linear rotating random_contact random_no_contact mixed rl_coverage" >&2
      exit 1
      ;;
  esac
done

if (( ${#gpus} != ${#motions} )); then
  echo "ERROR: --gpus must have ${#motions} entries, got ${#gpus}: ${gpus[*]}" >&2
  exit 1
fi

mkdir -p logs

pids=()

cleanup() {
  trap - EXIT INT TERM
  if (( ${#pids} > 0 )); then
    echo "[$(date +%H:%M:%S)] Stopping ${#pids} collectors..."
    for pid in "${pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "${pids[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[$(date +%H:%M:%S)] Starting collection"
echo "  letters:        ${letters[*]}"
echo "  motions:        ${motions[*]}"
echo "  gpus:           ${gpus[*]}"
echo "  episode_steps:  ${EPISODE_STEPS}"
echo "  motion_speedup:       ${MOTION_SPEEDUP}"
echo "  rl_warmup_episodes:  ${RL_WARMUP_EPISODES}"
echo "  rl_checkpoint_interval: ${RL_CHECKPOINT_INTERVAL}"
echo "  log_dir:             logs"
echo ""
echo "GPU mapping:"
for (( i=1; i<=${#motions}; i++ )); do
  echo "  ${motions[$i]} -> GPU ${gpus[$i]}"
done

echo ""
for shape in "${letters[@]}"; do
  for (( i=1; i<=${#motions}; i++ )); do
    motion=${motions[$i]}
    gpu=${gpus[$i]}
    out_dir="data/mujoco/letters/${shape}/${motion}"
    out_log="logs/collect_${shape}_${motion}.out"
    err_log="logs/collect_${shape}_${motion}.err"

    mkdir -p "$out_dir"

    extra_args=()
    if [[ "$motion" == "rl_coverage" ]]; then
      extra_args=(
        --rl_warmup_episodes "$RL_WARMUP_EPISODES"
        --rl_checkpoint_interval "$RL_CHECKPOINT_INTERVAL"
      )
    fi

    echo "[$(date +%H:%M:%S)] Launching ${shape}/${motion} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu \
    MUJOCO_GL=egl \
    MUJOCO_EGL_DEVICE_ID=$gpu \
      python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
        --output_dir "$out_dir" \
        --motion_type "$motion" \
        --shape "$shape" \
        --headless \
        --episode_steps "$EPISODE_STEPS" \
        --motion_speedup "$MOTION_SPEEDUP" \
        "${extra_args[@]}" \
        > "$out_log" \
        2> "$err_log" &
    pids+=($!)
  done
done

echo ""
echo "[$(date +%H:%M:%S)] Started ${#pids} collectors. Logs in logs/"
echo "[$(date +%H:%M:%S)] Press Ctrl-C to stop all collectors."

wait "${pids[@]}"
pids=()
trap - EXIT INT TERM

echo "[$(date +%H:%M:%S)] All collectors exited."
