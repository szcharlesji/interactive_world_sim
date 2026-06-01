#!/bin/zsh
set -euo pipefail
set +xv
setopt typeset_silent

LETTERS="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
MOTION="rl_coverage"
GPU="0"
SOURCE_ROOT="data/mujoco/letters"
EPISODE_STEPS=600
MOTION_SPEEDUP=2
RL_WARMUP_EPISODES=20
RL_CHECKPOINT_INTERVAL=10
EPISODES_PER_ROUND=1
MAX_PARALLEL=4
ROUND_SECONDS=0
TARGET_EPISODES=0
POLL_SECONDS=20
MUJOCO_EGL_DEVICE_ID_OVERRIDE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/data_collection/collect_letters_balanced.zsh [options]

Purpose:
  Balance data collection across letters while keeping the GPU busy. The script
  repeatedly launches collectors for the currently lowest-count letters, runs up
  to --max_parallel collectors concurrently, stops each letter after it gains
  --episodes_per_round successful saved episodes, then refills that slot with
  the next least-covered letter.

Options:
  --letters "A ... Z"          Letters to balance. Default: all 26 letters.
  --motion rl_coverage         Motion type. Default: rl_coverage
  --motions rl_coverage        Alias for --motion, for compatibility with other scripts.
  --gpu 0                      GPU for all active collectors. Default: 0
  --max_parallel 4             Concurrent collectors on this GPU. Default: 4
  --source_root PATH           Root output dir. Default: data/mujoco/letters
  --episode_steps 600          Frames per saved episode/video. Default: 600
  --motion_speedup 2           Per-segment trajectory speedup. Default: 2
  --rl_warmup_episodes 20      For rl_coverage: warmup horizons before saving.
                               Existing checkpoint warmup progress is resumed.
  --rl_checkpoint_interval 10  For rl_coverage: checkpoint interval. Default: 10
  --episodes_per_round 1       Stop a letter after this many new saved episodes,
                               then re-count and refill the slot. Default: 1
  --round_seconds 0            Optional max seconds per active collector round.
                               0 disables. Useful if a hard letter gets stuck.
  --target_episodes 0          Stop once every letter has at least this many
                               saved episodes. 0 runs until Ctrl-C.
  --poll_seconds 20            Episode-count polling interval. Default: 20
  --mujoco_egl_device_id ID    Override MUJOCO_EGL_DEVICE_ID. Default: same as --gpu
  --dry_run 0                  Print launch plan without launching. Default: 0
  --help                       Show this help.

Examples:
  # Balance all 26 letters with 4 concurrent RL collectors on GPU 0.
  ./scripts/data_collection/collect_letters_balanced.zsh \
    --gpu 0 \
    --max_parallel 4 \
    --episodes_per_round 1 \
    --rl_warmup_episodes 20 \
    --rl_checkpoint_interval 10

  # Stop when every letter has at least 100 saved episodes.
  ./scripts/data_collection/collect_letters_balanced.zsh \
    --gpu 0 \
    --max_parallel 4 \
    --target_episodes 100

  # Use a 30-minute cap per letter round to avoid one hard letter occupying a slot forever.
  ./scripts/data_collection/collect_letters_balanced.zsh \
    --gpu 0 \
    --max_parallel 4 \
    --round_seconds 1800
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --letters) LETTERS="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    --motions) MOTION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --source_root) SOURCE_ROOT="$2"; shift 2 ;;
    --episode_steps) EPISODE_STEPS="$2"; shift 2 ;;
    --motion_speedup) MOTION_SPEEDUP="$2"; shift 2 ;;
    --rl_warmup_episodes) RL_WARMUP_EPISODES="$2"; shift 2 ;;
    --rl_checkpoint_interval) RL_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --episodes_per_round) EPISODES_PER_ROUND="$2"; shift 2 ;;
    --round_seconds) ROUND_SECONDS="$2"; shift 2 ;;
    --target_episodes) TARGET_EPISODES="$2"; shift 2 ;;
    --poll_seconds) POLL_SECONDS="$2"; shift 2 ;;
    --mujoco_egl_device_id) MUJOCO_EGL_DEVICE_ID_OVERRIDE="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$LETTERS" ]]; then
  echo "ERROR: --letters cannot be empty" >&2
  exit 1
fi

case "$MOTION" in
  linear|rotating|random_contact|random_no_contact|mixed|rl_coverage) ;;
  *) echo "ERROR: unknown --motion '$MOTION'" >&2; exit 1 ;;
esac

if (( MAX_PARALLEL <= 0 )); then
  echo "ERROR: --max_parallel must be positive" >&2
  exit 1
fi
if (( EPISODES_PER_ROUND <= 0 )); then
  echo "ERROR: --episodes_per_round must be positive" >&2
  exit 1
fi
if (( ROUND_SECONDS < 0 || TARGET_EPISODES < 0 || POLL_SECONDS <= 0 )); then
  echo "ERROR: --round_seconds/--target_episodes must be non-negative and --poll_seconds positive" >&2
  exit 1
fi

letters=(${=LETTERS})
EGL_ID="${MUJOCO_EGL_DEVICE_ID_OVERRIDE:-$GPU}"
mkdir -p logs

typeset -A active_pid
typeset -A active_start_count
typeset -A active_target_count
typeset -A active_start_time
typeset -A active_out_log
typeset -A active_err_log

count_episodes() {
  local letter="$1"
  local dir="$SOURCE_ROOT/$letter/$MOTION"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -name 'episode_*.hdf5' -type f 2>/dev/null | wc -l | tr -d ' '
}

active_count() {
  local keys=(${(k)active_pid})
  echo ${#keys}
}

is_active() {
  local letter="$1"
  [[ -n "${active_pid[$letter]:-}" ]]
}

print_counts() {
  echo "[$(date +%H:%M:%S)] Current saved episode counts for motion=$MOTION"
  local letter count active_marker
  for letter in "${letters[@]}"; do
    count=$(count_episodes "$letter")
    active_marker=""
    if is_active "$letter"; then
      active_marker=" active->${active_target_count[$letter]}"
    fi
    echo "  $letter: $count$active_marker"
  done
}

pick_least_inactive_letter() {
  local best_letter=""
  local best_count=999999999
  local letter count
  for letter in "${letters[@]}"; do
    if is_active "$letter"; then
      continue
    fi
    count=$(count_episodes "$letter")
    if (( TARGET_EPISODES > 0 && count >= TARGET_EPISODES )); then
      continue
    fi
    if (( count < best_count )); then
      best_count=$count
      best_letter=$letter
    fi
  done
  if [[ -n "$best_letter" ]]; then
    echo "$best_letter $best_count"
  fi
}

all_reached_target() {
  if (( TARGET_EPISODES <= 0 )); then
    return 1
  fi
  local letter count
  for letter in "${letters[@]}"; do
    count=$(count_episodes "$letter")
    if (( count < TARGET_EPISODES )); then
      return 1
    fi
  done
  return 0
}

stop_letter() {
  local letter="$1"
  local pid="${active_pid[$letter]:-}"
  if [[ -n "$pid" ]]; then
    echo "[$(date +%H:%M:%S)] Stopping $letter collector pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  unset "active_pid[$letter]"
  unset "active_start_count[$letter]"
  unset "active_target_count[$letter]"
  unset "active_start_time[$letter]"
  unset "active_out_log[$letter]"
  unset "active_err_log[$letter]"
}

stop_all() {
  local keys=(${(k)active_pid})
  local letter
  for letter in "${keys[@]}"; do
    stop_letter "$letter"
  done
}

cleanup() {
  trap - EXIT INT TERM
  stop_all
}
trap cleanup EXIT INT TERM

launch_letter() {
  local letter="$1"
  local start_count="$2"
  local target_count=$(( start_count + EPISODES_PER_ROUND ))
  if (( TARGET_EPISODES > 0 && target_count > TARGET_EPISODES )); then
    target_count=$TARGET_EPISODES
  fi

  local out_dir="$SOURCE_ROOT/$letter/$MOTION"
  local out_log="logs/balanced_collect_${letter}_${MOTION}.out"
  local err_log="logs/balanced_collect_${letter}_${MOTION}.err"
  mkdir -p "$out_dir"

  local extra_args=()
  if [[ "$MOTION" == "rl_coverage" ]]; then
    extra_args=(
      --rl_warmup_episodes "$RL_WARMUP_EPISODES"
      --rl_checkpoint_interval "$RL_CHECKPOINT_INTERVAL"
    )
  fi

  echo "[$(date +%H:%M:%S)] Launching $letter ($start_count -> $target_count episodes) on GPU $GPU"

  if (( DRY_RUN )); then
    printf '  CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=%q python scripts/data_collection/sim_aloha_dataset_collection_scripted.py' "$GPU" "$EGL_ID"
    printf ' %q' --output_dir "$out_dir" --motion_type "$MOTION" --shape "$letter" --headless --episode_steps "$EPISODE_STEPS" --motion_speedup "$MOTION_SPEEDUP" "${extra_args[@]}"
    echo ""
    active_pid[$letter]="DRY_RUN"
    active_start_count[$letter]="$start_count"
    active_target_count[$letter]="$target_count"
    active_start_time[$letter]="$(date +%s)"
    active_out_log[$letter]="$out_log"
    active_err_log[$letter]="$err_log"
    return
  fi

  CUDA_VISIBLE_DEVICES=$GPU \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$EGL_ID \
    python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
      --output_dir "$out_dir" \
      --motion_type "$MOTION" \
      --shape "$letter" \
      --headless \
      --episode_steps "$EPISODE_STEPS" \
      --motion_speedup "$MOTION_SPEEDUP" \
      "${extra_args[@]}" \
      > "$out_log" \
      2> "$err_log" &

  active_pid[$letter]=$!
  active_start_count[$letter]="$start_count"
  active_target_count[$letter]="$target_count"
  active_start_time[$letter]="$(date +%s)"
  active_out_log[$letter]="$out_log"
  active_err_log[$letter]="$err_log"
}

fill_slots() {
  while (( $(active_count) < MAX_PARALLEL )); do
    if all_reached_target; then
      return
    fi
    local pick
    pick="$(pick_least_inactive_letter)"
    if [[ -z "$pick" ]]; then
      return
    fi
    local parts=(${=pick})
    local letter=${parts[1]}
    local count=${parts[2]}
    launch_letter "$letter" "$count"
    if (( DRY_RUN && $(active_count) >= MAX_PARALLEL )); then
      return
    fi
  done
}

check_active_collectors() {
  local keys=(${(k)active_pid})
  local letter pid curr_count target_count now elapsed
  now=$(date +%s)
  for letter in "${keys[@]}"; do
    pid="${active_pid[$letter]}"
    target_count="${active_target_count[$letter]}"

    if [[ "$pid" != "DRY_RUN" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] $letter collector exited; see ${active_out_log[$letter]} / ${active_err_log[$letter]}"
      wait "$pid" 2>/dev/null || true
      stop_letter "$letter"
      continue
    fi

    curr_count=$(count_episodes "$letter")
    echo "[$(date +%H:%M:%S)] $letter/$MOTION count: $curr_count / $target_count"

    if (( curr_count >= target_count )); then
      echo "[$(date +%H:%M:%S)] $letter reached round target ($curr_count episodes)"
      stop_letter "$letter"
      continue
    fi

    if (( ROUND_SECONDS > 0 )); then
      elapsed=$(( now - active_start_time[$letter] ))
      if (( elapsed >= ROUND_SECONDS )); then
        echo "[$(date +%H:%M:%S)] Round time limit reached for $letter (${elapsed}s)"
        stop_letter "$letter"
      fi
    fi
  done
}

echo "[$(date +%H:%M:%S)] Balanced parallel letter collection"
echo "  letters:                ${letters[*]}"
echo "  motion:                 $MOTION"
echo "  gpu:                    $GPU"
echo "  max_parallel:           $MAX_PARALLEL"
echo "  mujoco_egl_device_id:   $EGL_ID"
echo "  source_root:            $SOURCE_ROOT"
echo "  episode_steps:          $EPISODE_STEPS"
echo "  motion_speedup:         $MOTION_SPEEDUP"
echo "  rl_warmup_episodes:     $RL_WARMUP_EPISODES"
echo "  rl_checkpoint_interval: $RL_CHECKPOINT_INTERVAL"
echo "  episodes_per_round:     $EPISODES_PER_ROUND"
echo "  round_seconds:          $ROUND_SECONDS"
echo "  target_episodes:        $TARGET_EPISODES"
echo "  poll_seconds:           $POLL_SECONDS"
echo "  dry_run:                $DRY_RUN"
print_counts

fill_slots
if (( DRY_RUN )); then
  echo "[$(date +%H:%M:%S)] Dry run complete. Would launch $(active_count) collectors."
  trap - EXIT INT TERM
  exit 0
fi

while true; do
  if all_reached_target && (( $(active_count) == 0 )); then
    echo "[$(date +%H:%M:%S)] All letters reached target_episodes=$TARGET_EPISODES"
    print_counts
    break
  fi

  if (( $(active_count) == 0 )); then
    fill_slots
    if (( $(active_count) == 0 )); then
      echo "[$(date +%H:%M:%S)] No active collectors and no eligible letters remain."
      print_counts
      break
    fi
  fi

  sleep "$POLL_SECONDS"
  check_active_collectors
  fill_slots
  print_counts
done

trap - EXIT INT TERM
