# Sim Data Generation

This directory contains scripts for collecting MuJoCo/Aloha simulation data for PushT and procedural block-letter planar pushing.

## 1. Environment setup

From the repo root:

```bash
conda activate iws
cd /home_shared/grail_charles/interactive_world_sim
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

## 3. Collect legacy mesh PushT data

If `--shape` is omitted, the script uses the original mesh PushT environment.

```bash
MUJOCO_GL=egl python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/pusht/train \
  --motion_type random_contact \
  --headless
```

Available motion types:

```text
linear
rotating
random_contact
random_no_contact
```

The script runs until stopped. Only successful episodes are saved.

## 4. Collect one procedural letter

Procedural letters use the generic `planar_push` environment. Add `--shape A` through `--shape Z`.

Example for letter `H`:

```bash
MUJOCO_GL=egl python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/H/random_contact \
  --motion_type random_contact \
  --shape H \
  --headless
```

Examples for all motion types for one letter:

```bash
for motion in linear rotating random_contact random_no_contact; do
  MUJOCO_GL=egl python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
    --output_dir data/mujoco/letters/H/${motion} \
    --motion_type ${motion} \
    --shape H \
    --headless
done
```

Note: each process runs until stopped, so the loop above only advances after the current process exits.

## 5. Collect multiple letters manually

Run each collector into a unique output directory. Do not run multiple processes into the same `--output_dir`.

Small parallel example:

```bash
mkdir -p logs

for shape in A H S Z; do
  MUJOCO_GL=egl python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
    --output_dir data/mujoco/letters/${shape}/random_contact \
    --motion_type random_contact \
    --shape ${shape} \
    --headless \
    > logs/collect_${shape}_random_contact.out \
    2> logs/collect_${shape}_random_contact.err &
done
```

Monitor:

```bash
jobs
nvidia-smi
htop
tail -F logs/collect_A_random_contact.out
```

## 6. Collect all letters with the batch script

Use:

```bash
./scripts/data_collection/collect_all_letters.zsh
```

The script:

- iterates over `A` through `Z`
- runs motion types:
  - `linear`
  - `rotating`
  - `random_contact`
  - `random_no_contact`
- uses batched letters
- writes logs to `logs/`
- writes data to `data/mujoco/letters/<LETTER>/<MOTION>/`
- hard-stops after `HOURS` hours

Default settings inside the script:

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

If you want to use different GPUs, edit:

```zsh
GPUS=(0 0 1 1)
```

The GPU mapping is per motion type by index:

```text
linear            -> GPUS[1]
rotating          -> GPUS[2]
random_contact    -> GPUS[3]
random_no_contact -> GPUS[4]
```

## 7. Stop running collectors

If the batch script is running in your current terminal:

```text
Ctrl-C
```

If it does not stop, kill script and child collectors:

```bash
pkill -TERM -f collect_all_letters.zsh
pkill -TERM -f sim_aloha_dataset_collection_scripted.py
```

Check remaining processes:

```bash
pgrep -af 'collect_all_letters|sim_aloha_dataset_collection_scripted'
```

Force kill if needed:

```bash
pkill -KILL -f collect_all_letters.zsh
pkill -KILL -f sim_aloha_dataset_collection_scripted.py
```

If you know the parent PID:

```bash
kill -9 <PID>
pkill -KILL -f sim_aloha_dataset_collection_scripted.py
```

The `grep` process from commands like `ps aux | grep collect_all_letters` is not the collector; ignore it.

## 8. Output layout

Successful episodes are saved as HDF5 files:

```text
data/mujoco/letters/H/random_contact/
  episode_0.hdf5
  episode_1.hdf5
  videos/
    episode_0.mp4
    episode_1.mp4
```

Each saved episode contains observations, actions, robot bases, object state, and rendered camera frames. Failed trials are rejected and not saved.

Check generated files:

```bash
find data/mujoco/letters/H/random_contact -maxdepth 1 -name 'episode_*.hdf5' | sort
find data/mujoco/letters/H/random_contact/videos -name '*.mp4' | sort
```

Count episodes:

```bash
find data/mujoco/letters -name 'episode_*.hdf5' | wc -l
```

## 9. Behavior safeguards for procedural letters

For `--shape` collection, the code includes safeguards to avoid bad footage:

- grippers are fixed wide open; only XY positions change
- end-effectors are held close to the table
- object spawn is near the center
- random-contact motion uses one active pusher instead of pinching between both claws
- collection aborts and resets if:
  - the object tips or lifts
  - a claw overlaps/enters the letter footprint

You may see messages like:

```text
Episode 3 aborted: object tipped/lifted or claw overlapped the letter. Resetting.
```

This is expected; that trial is rejected and not saved.

## 10. Troubleshooting

### GPU/EGL issues

Use:

```bash
export MUJOCO_GL=egl
```

For a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/H/random_contact \
  --motion_type random_contact \
  --shape H \
  --headless
```

If EGL fails:

```bash
MUJOCO_GL=osmesa python scripts/data_collection/sim_aloha_dataset_collection_scripted.py \
  --output_dir data/mujoco/letters/H/random_contact \
  --motion_type random_contact \
  --shape H \
  --headless
```

`osmesa` is CPU rendering.

### Low success rate

This is normal for random scripted data. The script saves only successful episodes.

### Flickering letter intersections

Procedural letters use separate visual and collision geoms to avoid z-fighting at stroke intersections. Restart collectors after code changes so generated XML files are refreshed.

### Multiple collectors overwrite files

Do not run two collectors into the same directory. Use separate directories per letter, motion type, and process.

Bad:

```bash
--output_dir data/mujoco/letters/H/random_contact
```

for multiple simultaneous `H/random_contact` jobs.

Good:

```bash
--output_dir data/mujoco/letters/H/random_contact_run0
--output_dir data/mujoco/letters/H/random_contact_run1
```
