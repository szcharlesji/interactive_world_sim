#!/bin/zsh
emulate zsh
set -u

print -r -- "collect_all_letters.zsh: starting"

LETTERS=(A B C D E F G H I J K L M N O P Q R S T U V W X Y Z)
MOTIONS=(linear rotating random_contact random_no_contact)

# Motion-to-GPU mapping by index:
#   linear            -> GPU 0
#   rotating          -> GPU 0
#   random_contact    -> GPU 1
#   random_no_contact -> GPU 1
GPUS=(0 0 1 1)

BATCH_SIZE=3
HOURS=3
TOTAL_SECONDS=$(( HOURS * 3600 ))
NUM_BATCHES=$(( (${#LETTERS} + BATCH_SIZE - 1) / BATCH_SIZE ))
STOP_GRACE_SECONDS=5
SAFETY_SECONDS=60
BATCH_SECONDS=$(( (TOTAL_SECONDS - NUM_BATCHES * STOP_GRACE_SECONDS - SAFETY_SECONDS) / NUM_BATCHES ))
DRY_RUN=${DRY_RUN:-0}

if (( BATCH_SECONDS <= 0 )); then
  print -r -- "ERROR: computed non-positive BATCH_SECONDS=${BATCH_SECONDS}. Increase HOURS or reduce overhead."
  exit 1
fi

mkdir -p logs

active_pids=()
TIMER_PID=""

stop_pids() {
  local ids=("$@")
  if (( ${#ids} == 0 )); then
    return
  fi

  for pid in "${ids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  sleep "$STOP_GRACE_SECONDS"

  for pid in "${ids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done

  wait "${ids[@]}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM ALRM
  echo "[$(date +%H:%M:%S)] Cleaning up active collectors..."
  if [[ -n "${TIMER_PID}" ]]; then
    kill "$TIMER_PID" 2>/dev/null || true
  fi
  stop_pids "${active_pids[@]}"
}

on_timeout() {
  echo "[$(date +%H:%M:%S)] Global time limit reached after ${HOURS} hours."
  cleanup
  exit 0
}

trap cleanup EXIT INT TERM
trap on_timeout ALRM

# Extra global watchdog. Batch time-slicing should finish within HOURS, but this
# guarantees the script and active collectors stop if anything hangs.
WATCH_PID=$$
( sleep "$TOTAL_SECONDS"; kill -ALRM "$WATCH_PID" 2>/dev/null ) &
TIMER_PID=$!

start_time=$(date +%s)
echo "[$(date +%H:%M:%S)] Collecting ${#LETTERS} letters x ${#MOTIONS} motions on GPUs ${GPUS[*]}"
echo "[$(date +%H:%M:%S)] ${NUM_BATCHES} batches, ${BATCH_SECONDS}s per batch, max ${HOURS}h total"
echo "[$(date +%H:%M:%S)] Reserved $((NUM_BATCHES * STOP_GRACE_SECONDS + SAFETY_SECONDS))s for shutdown/safety overhead"
echo "[$(date +%H:%M:%S)] DRY_RUN=${DRY_RUN}"

for (( b=1; b<=${#LETTERS}; b+=BATCH_SIZE )); do
  batch=("${LETTERS[@]:$((b-1)):$BATCH_SIZE}")
  batch_num=$(( (b - 1) / BATCH_SIZE + 1 ))
  active_pids=()

  echo "[$(date +%H:%M:%S)] Starting batch ${batch_num}/${NUM_BATCHES}: ${batch[*]}"

  for shape in "${batch[@]}"; do
    for (( i=1; i<=${#MOTIONS}; i++ )); do
      motion=${MOTIONS[$i]}
      gpu=${GPUS[$i]}
      out_dir="data/mujoco/letters/${shape}/${motion}"
      mkdir -p "$out_dir"

      echo "[$(date +%H:%M:%S)] Launching ${shape}/${motion} on GPU ${gpu} -> ${out_dir}"
      if (( DRY_RUN )); then
        continue
      fi

      CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
        python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
          --output_dir "$out_dir" \
          --motion_type "$motion" \
          --shape "$shape" \
          --headless \
          > "logs/collect_${shape}_${motion}.out" \
          2> "logs/collect_${shape}_${motion}.err" &
      active_pids+=($!)
    done
  done

  if (( DRY_RUN )); then
    echo "[$(date +%H:%M:%S)] DRY_RUN: batch ${batch_num}/${NUM_BATCHES} launch list complete."
    continue
  fi

  echo "[$(date +%H:%M:%S)] Batch ${batch_num}/${NUM_BATCHES} running ${#active_pids} collectors for ${BATCH_SECONDS}s"
  sleep "$BATCH_SECONDS"

  echo "[$(date +%H:%M:%S)] Stopping batch ${batch_num}/${NUM_BATCHES}: ${batch[*]}"
  stop_pids "${active_pids[@]}"
  active_pids=()
done

kill "$TIMER_PID" 2>/dev/null || true
trap - EXIT INT TERM ALRM

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "[$(date +%H:%M:%S)] All batches complete in ${elapsed}s."
