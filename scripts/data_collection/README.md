# Sim Data Generation

This directory contains scripts for collecting MuJoCo/ALOHA simulation data for legacy PushT and procedural block-letter planar pushing.

The main collector is:

```bash
scripts/data_collection/sim_aloha_dataset_collection_scripted.py
```

It can collect scripted rollouts, mixed rollouts, and an online high-level RL coverage mode that tries to produce both translation and rotation within each short episode.

Quick recommendation for new procedural-letter data:

```bash
cd /home_shared/grail_charles/interactive_world_sim
conda activate iws
export PYTHONPATH=external/gym-aloha:.

./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "Y" \
  --motions "mixed rl_coverage" \
  --gpus "0 1" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10 \
  --rl_checkpoint_interval 1
```

Use `mixed` for diverse scripted trajectories and `rl_coverage` for online coverage-seeking trajectories. `--rl_warmup_episodes` and `--rl_checkpoint_interval` only affect `rl_coverage`; other motion types ignore them.

## 1. Environment setup

From the repo root:

```bash
cd /home_shared/grail_charles/interactive_world_sim
conda activate iws
export PYTHONPATH=external/gym-aloha:.
```

Initialize/install the MuJoCo ALOHA submodule if needed:

```bash
git submodule update --init --recursive
uv pip install -e external/gym-aloha/
pip install -e . --no-deps
```

For headless GPU rendering:

```bash
export MUJOCO_GL=egl
```

For a specific GPU, set both CUDA and MuJoCo EGL device IDs:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python ...
```

On some machines, `MUJOCO_EGL_DEVICE_ID` is interpreted after `CUDA_VISIBLE_DEVICES` remapping. If `MUJOCO_EGL_DEVICE_ID=<physical_gpu>` fails, try `MUJOCO_EGL_DEVICE_ID=0` while keeping `CUDA_VISIBLE_DEVICES=<physical_gpu>`.

If EGL fails, use CPU rendering as a fallback:

```bash
export MUJOCO_GL=osmesa
```

## 2. Quick smoke tests

Check that procedural letters load:

```bash
python - <<'PY'
from gym_aloha.env import AlohaEnv

for task in ["planar_push", "push_h", "push_a", "push_z"]:
    env = AlohaEnv(task=task)
    obs, info = env.reset(seed=0)
    raw = env._env.task.get_observation(env._env.physics)
    print(task, env.shape, raw["env_state"].shape, list(raw["images"].keys()))
PY
```

Expected camera key:

```text
['top_pov']
```

Check the IWS wrapper:

```bash
python - <<'PY'
from interactive_world_sim.environments import SimAlohaPlanarPushEnv

env = SimAlohaPlanarPushEnv(shape="H")
env.reset(seed=0)
obs = env.get_observations()
print(obs["env_state"])
print(obs["images"].keys())
PY
```

## 3. Motion types

Available `--motion_type` values:

```text
linear
rotating
random_contact
random_no_contact
mixed
rl_coverage
```

Summary:

| Motion type | What it does |
| --- | --- |
| `linear` | Pushes in one of four cardinal directions. |
| `rotating` | Uses coordinated contacts to rotate the object clockwise/counterclockwise. |
| `random_contact` | Uses exterior side/corner/tangential contacts. For letters, one arm is active to avoid pinching/hooking holes. |
| `random_no_contact` | Moves arms without moving the object; useful negative/no-op data. |
| `mixed` | Replans diverse sub-motions inside one saved episode: translation, rotation, random contact, and occasional no-contact. |
| `rl_coverage` | Online high-level RL/contextual-bandit policy over safe primitives. Rewards new `(x, y, theta)` coverage, translation, rotation, and penalizes edge/abort cases. |

`rl_coverage` is not raw-joint RL and it is not world-model training. It is an online data-collection policy. While the collector records HDF5 episodes, it also learns a small in-memory table that chooses which high-level primitive to run next. The safe planner/IK/controller still generate and execute the actual end-effector trajectories.

The `rl_coverage` action set is:

```text
push_right
push_left
push_up
push_down
rotate_cw
rotate_ccw
tangent_cw
tangent_ccw
```

The policy rewards:

- new object `(x, y, theta)` coverage
- object translation
- object rotation

and penalizes:

- little/no object movement
- getting too close to table edges
- object leaving table bounds
- object tipping/lifting/abort

Each collector process has its own learner. Multiple letters/GPU jobs do not share RL state. Restarting a process resets the learner, but saved HDF5 files remain valid.

If you want the policy to learn before writing training files, use:

```bash
--rl_warmup_episodes 10
```

During RL warmup, the policy still updates from each full episode horizon, but successful HDF5/MP4 episodes are discarded instead of saved. After the warmup count is reached, future successful episodes save normally. Warmup is useful when you want the online policy to explore for a while before producing the dataset.

`rl_coverage` checkpoints are saved automatically. By default, each collector writes and resumes from:

```text
<output_dir>/rl_policy_checkpoint.json
```

So if you restart the same command with the same `--output_dir`, the online policy resumes from the previous Q/coverage table and previously completed warmup count. Checkpoints are saved after every full episode horizon by default and immediately after abort penalties. Use:

```bash
--rl_checkpoint_interval 5
```

to save every 5 full episode horizons instead. Use `--rl_checkpoint_path <path>` for direct single-process runs if you want a custom checkpoint location. Do not share one checkpoint path between simultaneous collectors.

For `rl_coverage`, keep:

```bash
--episode_steps 600
```

or lower. `600` saved frames corresponds to a 20-second MP4 at 30 FPS, and the collector enforces `rl_coverage` episodes at `<=600` steps. Inside those 600 frames, the policy runs multiple shorter primitives. Saved `rl_coverage` episodes must contain both:

- object XY translation greater than about `1.5 cm`
- object yaw rotation greater than about `0.08 rad`

## 4. Collect legacy mesh PushT data

If `--shape` is omitted, the script uses the original mesh PushT environment.

```bash
MUJOCO_GL=egl python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/pusht/train \
  --motion_type random_contact \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2
```

The script runs until stopped. Only successful episodes are saved.

## 5. Collect one procedural letter

Procedural letters use the generic `planar_push` environment. Add `--shape A` through `--shape Z`.

Example for letter `Y` with random contact:

```bash
cd /home_shared/grail_charles/interactive_world_sim
conda activate iws
export PYTHONPATH=external/gym-aloha:.

CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/Y/random_contact \
  --motion_type random_contact \
  --shape Y \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2
```

Example for `mixed`:

```bash
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/Y/mixed \
  --motion_type mixed \
  --shape Y \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2
```

Example for online RL coverage:

```bash
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/Y/rl_coverage \
  --motion_type rl_coverage \
  --shape Y \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10 \
  --rl_checkpoint_interval 1
```

This automatically resumes from and saves to:

```text
data/mujoco/letters/Y/rl_coverage/rl_policy_checkpoint.json
```

Set `--rl_warmup_episodes 0` to save successful episodes immediately. Delete the checkpoint file or use a new `--output_dir` if you want a fresh RL policy. The direct collector runs until you stop it with `Ctrl-C`.

## 6. Collect multiple letters manually

Run each collector into a unique output directory. Do not run multiple processes into the same `--output_dir`.

Small parallel example:

```bash
mkdir -p logs

for shape in A H S Z; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
    --output_dir data/mujoco/letters/${shape}/rl_coverage \
    --motion_type rl_coverage \
    --shape ${shape} \
    --headless \
    --episode_steps 600 \
    --motion_speedup 2 \
    --rl_warmup_episodes 10 \
    > logs/collect_${shape}_rl_coverage.out \
    2> logs/collect_${shape}_rl_coverage.err &
done
```

Monitor:

```bash
jobs
nvidia-smi
htop
tail -F logs/collect_A_rl_coverage.out
```

## 7. Use the multi-process letter script

Use `collect_letters_two_gpus.zsh` for controlled multi-process collection. Despite the historical name, it can use any GPU IDs you pass through `--gpus`.

Important mental model:

```text
for each letter:
  for each motion:
    launch one collector on the GPU assigned to that motion
```

So `--gpus` maps to `--motions`, not directly to `--letters`.

Use `--motions "rl_coverage" --gpus "1"` to put all listed letters' RL collectors on GPU 1. Use multiple script invocations if you want to shard different letter groups across different GPUs.

Single-letter example:

```bash
cd /home_shared/grail_charles/interactive_world_sim
conda activate iws
export PYTHONPATH=external/gym-aloha:.

./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "Y" \
  --motions "rl_coverage" \
  --gpus "0" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10
```

Collect `mixed` and `rl_coverage` for several letters, with `mixed` on GPU 0 and `rl_coverage` on GPU 1:

```bash
./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "V W Y" \
  --motions "mixed rl_coverage" \
  --gpus "0 1" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10
```

Collect all supported modes for one letter:

```bash
./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "Y" \
  --motions "linear rotating random_contact random_no_contact mixed rl_coverage" \
  --gpus "0 0 1 1 0 1" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10
```

Important: the script launches:

```text
num_letters × num_motions
```

parallel collectors. For example, `--letters "A B C" --motions "mixed rl_coverage"` launches `6` processes:

```text
A/mixed        -> GPU 0
A/rl_coverage  -> GPU 1
B/mixed        -> GPU 0
B/rl_coverage  -> GPU 1
C/mixed        -> GPU 0
C/rl_coverage  -> GPU 1
```

### Shard different letters across different GPUs

To distribute letters across GPUs for the same motion type, run separate invocations in separate terminals or `tmux` panes.

GPU 0:

```bash
./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "A B C D" \
  --motions "rl_coverage" \
  --gpus "0" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10
```

GPU 1:

```bash
./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "E F G H" \
  --motions "rl_coverage" \
  --gpus "1" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10
```

This launches:

```text
GPU 0: A/B/C/D rl_coverage collectors
GPU 1: E/F/G/H rl_coverage collectors
```

Start with 1-2 letters per GPU if you are unsure about memory/rendering load, then scale up after checking `nvidia-smi` and logs.

If the script is not executable, run once:

```bash
chmod +x scripts/data_collection/collect_letters_two_gpus.zsh
```

or invoke it with `zsh`:

```bash
zsh scripts/data_collection/collect_letters_two_gpus.zsh --letters "Y" --motions "rl_coverage" --gpus "0"
```

### Run with `nohup`

For longer unattended runs, redirect the driver logs and let child collector logs go to `logs/collect_<LETTER>_<MOTION>.out|err`:

```bash
nohup ./scripts/data_collection/collect_letters_two_gpus.zsh \
  --letters "A B C D" \
  --motions "rl_coverage" \
  --gpus "0" \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 10 \
  > logs/driver_gpu0_rl_letters_ABCD.out \
  2> logs/driver_gpu0_rl_letters_ABCD.err &
```

Monitor:

```bash
nvidia-smi
pgrep -af sim_aloha_dataset_collection_scripted
tail -F logs/collect_A_rl_coverage.out
tail -F logs/collect_A_rl_coverage.err
```

## 8. Balanced all-letter collection

Use `collect_letters_balanced.zsh` when you want to keep the GPU busy while automatically focusing under-represented letters. Unlike `collect_letters_two_gpus.zsh`, this script does not launch every selected letter forever. Instead, it:

1. counts saved `episode_*.hdf5` files per letter for one motion type
2. launches the currently lowest-count letters up to `--max_parallel`
3. stops a letter once it gains `--episodes_per_round` new successful episodes
4. refills that slot with the next lowest-count inactive letter

By default it balances all 26 letters:

```bash
./scripts/data_collection/collect_letters_balanced.zsh \
  --gpu 0 \
  --max_parallel 4 \
  --motion rl_coverage \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 20 \
  --rl_checkpoint_interval 10 \
  --episodes_per_round 1
```

Increase `--max_parallel` if GPU utilization is low:

```bash
./scripts/data_collection/collect_letters_balanced.zsh \
  --gpu 0 \
  --max_parallel 6 \
  --motion rl_coverage \
  --episode_steps 600 \
  --motion_speedup 2 \
  --rl_warmup_episodes 20 \
  --rl_checkpoint_interval 10 \
  --episodes_per_round 1
```

Stop when every letter reaches a target count:

```bash
./scripts/data_collection/collect_letters_balanced.zsh \
  --gpu 0 \
  --max_parallel 4 \
  --target_episodes 100
```

Use a time cap to avoid one hard letter occupying a slot forever:

```bash
./scripts/data_collection/collect_letters_balanced.zsh \
  --gpu 0 \
  --max_parallel 4 \
  --round_seconds 1800
```

For unattended runs:

```bash
nohup ./scripts/data_collection/collect_letters_balanced.zsh \
  --gpu 0 \
  --max_parallel 4 \
  --target_episodes 100 \
  > logs/balanced_collect_all_letters_rl.out \
  2> logs/balanced_collect_all_letters_rl.err &
```

Monitor the driver and active child collectors:

```bash
tail -F logs/balanced_collect_all_letters_rl.out
tail -F logs/balanced_collect_<LETTER>_rl_coverage.out
tail -F logs/balanced_collect_<LETTER>_rl_coverage.err
```

## 9. Collect all letters with the batch script

Use:

```bash
./scripts/data_collection/collect_all_letters.zsh
```

The script:

- iterates over `A` through `Z`
- runs the motion types configured inside the script
- uses batched letters
- writes logs to `logs/`
- writes data to `data/mujoco/letters/<LETTER>/<MOTION>/`
- hard-stops after `HOURS` hours

Default settings inside the script may be edited directly:

```zsh
LETTERS=(A B C D E F G H I J K L M N O P Q R S T U V W X Y Z)
MOTIONS=(linear rotating random_contact random_no_contact)
GPUS=(0 0 1 1)
BATCH_SIZE=3
HOURS=3
```

Run:

```bash
conda activate iws
cd /home_shared/grail_charles/interactive_world_sim
export PYTHONPATH=external/gym-aloha:.
./scripts/data_collection/collect_all_letters.zsh
```

If you want `mixed` or `rl_coverage` in this older batch script, edit `MOTIONS` and `GPUS` inside `collect_all_letters.zsh` so the array lengths match.

## 10. Output layout

Successful episodes are saved as HDF5 files:

```text
data/mujoco/letters/Y/rl_coverage/
  episode_0.hdf5
  episode_1.hdf5
  videos/
    episode_0.mp4
    episode_1.mp4
```

Rejected/aborted trials are saved for debugging under:

```text
data/mujoco/letters/Y/rl_coverage/rejected/
```

Each saved episode contains observations, actions, robot bases, object state, and rendered camera frames. The saved videos are only for inspection; the HDF5 files are the training data.

Check generated files:

```bash
find data/mujoco/letters/Y/rl_coverage -maxdepth 1 -name 'episode_*.hdf5' | sort
find data/mujoco/letters/Y/rl_coverage/videos -name '*.mp4' | sort
```

Count episodes:

```bash
find data/mujoco/letters -name 'episode_*.hdf5' | wc -l
```

## 11. Behavior safeguards for procedural letters

For `--shape` collection, the code includes safeguards to avoid bad footage:

- grippers are fixed wide open; only XY positions change
- end-effectors are held close to the table
- object spawn is near the center
- random-contact and RL coverage use one active pusher for letter contacts to avoid pinching/hooking holes
- collection aborts and resets if:
  - the object tips
  - the object lifts
  - the procedural letter footprint leaves the MuJoCo table bounds

You may see messages like:

```text
Episode 3 aborted and saved as rejected episode 1: object tipped/lifted/out of table. Resetting.
```

This is expected; that trial is rejected and not saved as successful data.

## 12. Stop running collectors

If a script is running in your current terminal:

```text
Ctrl-C
```

If it does not stop, kill script and child collectors:

```bash
pkill -TERM -f collect_letters_two_gpus.zsh
pkill -TERM -f collect_all_letters.zsh
pkill -TERM -f sim_aloha_dataset_collection_scripted.py
```

Check remaining processes:

```bash
pgrep -af 'collect_letters_two_gpus|collect_all_letters|sim_aloha_dataset_collection_scripted'
```

Force kill if needed:

```bash
pkill -KILL -f collect_letters_two_gpus.zsh
pkill -KILL -f collect_all_letters.zsh
pkill -KILL -f sim_aloha_dataset_collection_scripted.py
```

The `grep` process from commands like `ps aux | grep collect_all_letters` is not the collector; ignore it.

## 13. Troubleshooting

### GPU/EGL issues

Use:

```bash
export MUJOCO_GL=egl
```

For a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/H/rl_coverage \
  --motion_type rl_coverage \
  --shape H \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2
```

If EGL fails:

```bash
MUJOCO_GL=osmesa python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/H/rl_coverage \
  --motion_type rl_coverage \
  --shape H \
  --headless \
  --episode_steps 600 \
  --motion_speedup 2
```

`osmesa` is CPU rendering.

### Low success rate

This is normal for random scripted data. The script saves only successful episodes.

For `rl_coverage`, a trial is successful only if it has both translation and rotation, so early rejected/unsaved episodes are expected while the online policy explores. If almost nothing is being saved after many attempts, lower `--motion_speedup` to `1.5` or `2`, try a different letter, or inspect `rejected/` videos for tipping/out-of-table behavior.

### RL policy does not remember after restart

`rl_coverage` currently learns online in memory inside each collector process. If you stop and restart the process, the Q/coverage table resets. The saved HDF5 data is unaffected.

### Flickering letter intersections

Procedural letters use separate visual and collision geoms to avoid z-fighting at stroke intersections. Restart collectors after code changes so generated XML files are refreshed.

### Multiple collectors overwrite files

Do not run two collectors into the same directory. Use separate directories per letter, motion type, and process.

Bad:

```bash
--output_dir data/mujoco/letters/H/rl_coverage
```

for multiple simultaneous `H/rl_coverage` jobs.

Good:

```bash
--output_dir data/mujoco/letters/H/rl_coverage_run0
--output_dir data/mujoco/letters/H/rl_coverage_run1
```
