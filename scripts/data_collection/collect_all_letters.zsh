#!/bin/zsh

LETTERS=(A B C D E F G H I J K L M N O P Q R S T U V W X Y Z)
MOTIONS=(linear rotating random_contact random_no_contact)
GPUS=(0 0 1 1)
BATCH_SIZE=3
HOURS=3

mkdir -p logs

# Kill timer + all running jobs on exit/interrupt.
cleanup() {
  kill "$TIMER_PID" 2>/dev/null
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT INT TERM

# Hard stop after HOURS.
( sleep $(( HOURS * 3600 )); echo "Time limit reached."; kill $$ ) &
TIMER_PID=$!

for (( b=1; b<=${#LETTERS}; b+=BATCH_SIZE )); do
  batch=("${LETTERS[@]:$((b-1)):$BATCH_SIZE}")
  echo "[$(date +%H:%M:%S)] Starting batch: ${batch[*]}"

  batch_pids=()
  for shape in "${batch[@]}"; do
    for (( i=1; i<=${#MOTIONS}; i++ )); do
      motion=${MOTIONS[$i]}
      gpu=${GPUS[$i]}
      mkdir -p data/mujoco/letters/${shape}/${motion}
      CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
        python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
          --output_dir data/mujoco/letters/${shape}/${motion} \
          --motion_type ${motion} \
          --shape ${shape} \
          --headless \
          > logs/collect_${shape}_${motion}.out \
          2> logs/collect_${shape}_${motion}.err &
      batch_pids+=($!)
    done
  done

  wait "${batch_pids[@]}"
  echo "[$(date +%H:%M:%S)] Batch ${batch[*]} done."
done

echo "All 26 letters complete."
