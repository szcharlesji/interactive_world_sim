#!/bin/zsh
set -u

LETTERS="V W"
GPUS="0 0 1 1"
EPISODE_STEPS=600
MOTION_SPEEDUP=2

usage() {
  cat <<'EOF'
Usage:
  ./scripts/data_collection/collect_letters_two_gpus.zsh [options]

Options:
  --letters "V W"          Letters to collect. Default: "V W"
  --gpus "0 0 1 1"         GPU per motion. Default: "0 0 1 1"
                            order: linear rotating random_contact random_no_contact
  --episode_steps 600      Frames per saved episode/video. Default: 600
  --motion_speedup 2       Per-segment trajectory speedup. Default: 2
  --help                   Show this help.

Example:
  ./scripts/data_collection/collect_letters_two_gpus.zsh \
    --letters "V W Z" \
    --gpus "0 0 1 1" \
    --episode_steps 600 \
    --motion_speedup 2
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --letters)
      LETTERS="$2"
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
motions=(linear rotating random_contact random_no_contact)
gpus=(${=GPUS})

if (( ${#letters} == 0 )); then
  echo "ERROR: --letters cannot be empty" >&2
  exit 1
fi

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
echo "  motion_speedup: ${MOTION_SPEEDUP}"
echo "  log_dir:        logs"
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
