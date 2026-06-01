"""Usage:

Scripted Data Collection with Auto-Recording:
- By default, episodes are automatically recorded every 3 seconds
- Press "Q" to exit program
- Press "A" to toggle auto-recording on/off
- Manual controls (when auto-recording is OFF):
  - Press "C" to start recording
  - Press "S" to stop recording
  - Press "Backspace" to delete the previously recorded episode

The system uses scripted and online-RL policies:
1. Linear pushes: Coordinated pushing in different directions
2. Rotations: Rotating the object by gripping/pushing exterior points
3. Random contact: Random exterior side/corner/tangential contacts
4. Random exploration: Non-contact exploration motions
5. Mixed: Replans diverse interaction-heavy sub-motions within each episode
6. RL coverage: Online contextual-bandit primitive selection for translation + rotation coverage
"""

# %%
import os
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import transforms3d
from gym_aloha.constants import PUPPET_GRIPPER_POSITION_OPEN
from gym_aloha.env import AlohaEnv
from gym_aloha.planar_objects import get_planar_object_spec
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
from yixuan_utilities.kinematics_helper import KinHelper

from interactive_world_sim.utils.draw_utils import (
    concat_img_h,
    concat_img_v,
    plot_single_3d_pos_traj,
)
from interactive_world_sim.utils.motion_planner import (
    MotionPlanner,
    PlanarObjectGeometryAnalyzer,
    PlanarObjectInfo,
    TGeometryAnalyzer,
    TShapeInfo,
    WorkspaceConstraints,
    evaluate_random_contact_success,
    evaluate_random_motion_success,
    wrap_angle,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert

# Procedural letters have center z ~= 0.014 and half-height ~= 0.012, so the
# top surface is around z=0.026. Push just below the top to avoid catching the
# letter on the sloped lower part of the open claws.
PLANAR_PUSH_EEF_Z = 0.022
PLANAR_PUSH_GRIPPER = 1.0
DEFAULT_EPISODE_STEPS = 600
DEFAULT_MOTION_SPEEDUP = 5.0
MAX_RL_COVERAGE_EPISODE_STEPS = 600
DEFAULT_RL_WARMUP_EPISODES = 0
DEFAULT_RL_CHECKPOINT_INTERVAL = 1
RL_CHECKPOINT_FILENAME = "rl_policy_checkpoint.json"
TABLE_X_LIMIT = 0.61
TABLE_Y_LIMIT = 0.37
TABLE_EDGE_MARGIN = 0.03
SUPPORTED_MOTION_TYPES = {
    "linear",
    "rotating",
    "random_contact",
    "random_no_contact",
    "mixed",
    "rl_coverage",
}


def speed_up_trajectory(trajectory: np.ndarray, speedup: float) -> np.ndarray:
    """Resample a planned XY trajectory to make each executed step larger."""
    if speedup <= 1.0 or len(trajectory) <= 2:
        return trajectory
    num_steps = max(2, int(np.ceil(len(trajectory) / speedup)))
    indices = np.rint(np.linspace(0, len(trajectory) - 1, num_steps)).astype(int)
    return trajectory[indices]


def evaluate_collection_success(
    motion_type: str,
    init_pose: np.ndarray,
    final_pose: np.ndarray,
    actions: np.ndarray,
) -> bool:
    """Evaluate a full multi-segment collection episode.

    Individual planned segments may choose different push directions.  For full
    episodes with multiple segments, use direction-agnostic movement success for
    contact motions and no-movement success for random_no_contact.
    """
    if motion_type == "random_no_contact":
        return evaluate_random_motion_success(init_pose, final_pose, actions)
    if motion_type == "rl_coverage":
        initial_yaw = np.arctan2(init_pose[1, 0], init_pose[0, 0])
        final_yaw = np.arctan2(final_pose[1, 0], final_pose[0, 0])
        xy_translation = float(np.linalg.norm(final_pose[:2, 3] - init_pose[:2, 3]))
        yaw_rotation = abs(wrap_angle(final_yaw - initial_yaw))
        return (
            evaluate_random_contact_success(init_pose, final_pose, actions)
            and xy_translation > 0.015
            and yaw_rotation > 0.08
        )
    return evaluate_random_contact_success(init_pose, final_pose, actions)


def freeze_planar_grippers_open(env: AlohaEnv) -> None:
    """Force simulated gripper fingers to a fixed wide-open pusher state."""
    physics = env._env.physics
    physics.data.qpos[6:8] = PUPPET_GRIPPER_POSITION_OPEN
    physics.data.qpos[14:16] = PUPPET_GRIPPER_POSITION_OPEN
    if physics.data.ctrl.shape[0] >= 14:
        physics.data.ctrl[6] = PUPPET_GRIPPER_POSITION_OPEN
        physics.data.ctrl[13] = PUPPET_GRIPPER_POSITION_OPEN
    physics.forward()


def restore_planar_object_pose(env: AlohaEnv, object_pose: np.ndarray) -> None:
    """Restore the pushed object after moving ALOHA to a safe reset pose."""
    physics = env._env.physics
    joint_id = physics.model.name2id("push_object_joint", "joint")
    qpos_start = physics.model.jnt_qposadr[joint_id]
    dof_start = physics.model.jnt_dofadr[joint_id]
    physics.data.qpos[qpos_start : qpos_start + 7] = object_pose
    physics.data.qvel[dof_start : dof_start + 6] = 0.0
    physics.forward()


def env_state_to_mat(env_state: np.ndarray) -> np.ndarray:
    """Convert environment state to matrix."""
    position = env_state[:3]
    quaternion = env_state[3:]
    rot_matrix = transforms3d.quaternions.quat2mat(quaternion)
    world_t_obj = np.eye(4)
    world_t_obj[:3, :3] = rot_matrix
    world_t_obj[:3, 3] = position
    return world_t_obj


def is_planar_object_pose_valid(
    env_state: np.ndarray,
    min_flat_alignment: float = 0.97,
    max_z: float = 0.06,
) -> bool:
    """Return whether the pushed object is still flat and on the table."""
    world_t_obj = env_state_to_mat(env_state)
    z_axis = world_t_obj[:3, 2]
    alignment = float(np.dot(z_axis, np.array([0.0, 0.0, 1.0])))
    z = float(world_t_obj[2, 3])
    return alignment >= min_flat_alignment and z <= max_z


def extract_t_info(env_state: np.ndarray) -> TShapeInfo:
    """Extract T-shape information from environment state."""
    # env_state contains [x, y, z, qw, qx, qy, qz] for the T-shape
    world_t_obj = env_state_to_mat(env_state)
    position = world_t_obj[:3, 3]
    rot_matrix = world_t_obj[:3, :3]

    rotation_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # -pi/2 to pi/2

    # T-shape dimensions (scaled by 0.6 from XML)
    original_width = 0.2
    original_height = 0.2
    original_thickness = 0.05
    original_height_2 = 0.04
    scale = 0.8

    width = original_width * scale
    height = original_height * scale
    thickness = original_thickness * scale

    # Generate keypoints (simplified, we'll use geometric calculation)
    # For now, we'll use dummy keypoints since we don't have the exact mesh info
    keypoints = np.array(
        [
            [-width / 2, thickness / 2, original_height_2],  # top left
            [width / 2, thickness / 2, original_height_2],  # top right
            [-width / 2, -thickness / 2, original_height_2],  # top bar bottom side left
            [width / 2, -thickness / 2, original_height_2],  # top bar bottom side right
            [-thickness / 2, -thickness / 2, original_height_2],  # stem top left
            [thickness / 2, -thickness / 2, original_height_2],  # stem top right
            [
                -thickness / 2,
                thickness / 2 - height,
                original_height_2,
            ],  # stem bottom left
            [
                thickness / 2,
                thickness / 2 - height,
                original_height_2,
            ],  # stem bottom right
        ]
    )

    return TShapeInfo(
        center=position[:2],  # Only use x, y
        rotation=rotation_angle,
        keypoints=keypoints,
        width=width,
        height=height,
        thickness=thickness,
        pose=world_t_obj,
    )


def extract_planar_object_info(env_state: np.ndarray, shape: str) -> PlanarObjectInfo:
    """Extract generic planar object pose information from env state."""
    world_t_obj = env_state_to_mat(env_state)
    position = world_t_obj[:3, 3]
    rot_matrix = world_t_obj[:3, :3]
    rotation_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return PlanarObjectInfo(
        center=position[:2],
        rotation=rotation_angle,
        pose=world_t_obj,
        spec=get_planar_object_spec(shape),
    )


def planar_object_footprint_xy(env_state: np.ndarray, shape: str) -> np.ndarray:
    """Return world-frame XY corners for all procedural letter box strokes."""
    object_info = extract_planar_object_info(env_state, shape)
    local_points = []
    for part in object_info.spec.parts:
        hx, hy = part.size[:2]
        corners = np.array(
            [
                [-hx, -hy],
                [hx, -hy],
                [hx, hy],
                [-hx, hy],
            ],
            dtype=np.float64,
        )
        part_cos, part_sin = np.cos(part.yaw), np.sin(part.yaw)
        part_rot = np.array([[part_cos, -part_sin], [part_sin, part_cos]])
        local_points.append(np.array(part.pos[:2])[None] + corners @ part_rot.T)

    local_points_xy = np.concatenate(local_points, axis=0)
    obj_cos, obj_sin = np.cos(object_info.rotation), np.sin(object_info.rotation)
    obj_rot = np.array([[obj_cos, -obj_sin], [obj_sin, obj_cos]])
    return object_info.center[None] + local_points_xy @ obj_rot.T


def is_planar_object_on_table(env_state: np.ndarray, shape: str) -> bool:
    """Return whether the whole letter footprint remains on the MuJoCo table."""
    footprint = planar_object_footprint_xy(env_state, shape)
    x_min = -TABLE_X_LIMIT + TABLE_EDGE_MARGIN
    x_max = TABLE_X_LIMIT - TABLE_EDGE_MARGIN
    y_min = -TABLE_Y_LIMIT + TABLE_EDGE_MARGIN
    y_max = TABLE_Y_LIMIT - TABLE_EDGE_MARGIN
    return bool(
        (footprint[:, 0] >= x_min).all()
        and (footprint[:, 0] <= x_max).all()
        and (footprint[:, 1] >= y_min).all()
        and (footprint[:, 1] <= y_max).all()
    )


def get_current_arm_positions(
    obs: dict, kin_helper: KinHelper, world_t_bases: np.ndarray
) -> np.ndarray:
    """Get current end-effector positions for both arms in left base frame."""
    # Extract joint positions
    left_qpos = obs["qpos"][:7]
    right_qpos = obs["qpos"][7:]

    # Compute forward kinematics
    left_qpos_extended = np.concatenate([left_qpos, left_qpos[6:7]])
    right_qpos_extended = np.concatenate([right_qpos, right_qpos[6:7]])

    left_ee_pose = kin_helper.compute_fk_from_link_idx(
        left_qpos_extended, [kin_helper.sapien_eef_idx]
    )[0]
    right_ee_pose = kin_helper.compute_fk_from_link_idx(
        right_qpos_extended, [kin_helper.sapien_eef_idx]
    )[0]
    world_t_left_ee = world_t_bases[0] @ left_ee_pose
    world_t_right_ee = world_t_bases[1] @ right_ee_pose

    # Extract XY positions
    left_xy = world_t_left_ee[:2, 3]
    right_xy = world_t_right_ee[:2, 3]

    return np.concatenate([left_xy, right_xy])


def _points_inside_planar_object_strokes(
    points_xy: np.ndarray,
    object_info: PlanarObjectInfo,
    margin: float = 0.002,
) -> np.ndarray:
    """Return a mask for points whose XY projection overlaps letter material.

    The procedural letters are unions of box strokes.  Checking against that
    union catches actual claw/finger overlap while avoiding false positives from
    empty holes inside letters such as A, H, and O.
    """
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=bool)

    cos_r, sin_r = np.cos(object_info.rotation), np.sin(object_info.rotation)
    object_rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    object_local = (points_xy - object_info.center[None]) @ object_rot

    inside_any = np.zeros((len(points_xy),), dtype=bool)
    for part in object_info.spec.parts:
        part_cos, part_sin = np.cos(part.yaw), np.sin(part.yaw)
        part_rot = np.array([[part_cos, -part_sin], [part_sin, part_cos]])
        part_local = (object_local - np.array(part.pos[:2])[None]) @ part_rot
        hx, hy = part.size[:2]
        inside_part = (np.abs(part_local[:, 0]) <= hx + margin) & (
            np.abs(part_local[:, 1]) <= hy + margin
        )
        inside_any |= inside_part
    return inside_any


def _append_segment_samples(
    samples: list[np.ndarray], start_xy: np.ndarray, end_xy: np.ndarray, n: int = 7
) -> None:
    for alpha in np.linspace(0.0, 1.0, n):
        samples.append((1.0 - alpha) * start_xy + alpha * end_xy)


def _get_named_xyz(physics, kind: str, name: str) -> Optional[np.ndarray]:
    try:
        if kind == "site":
            return np.asarray(
                physics.named.data.site_xpos[name], dtype=np.float64
            ).copy()
        if kind == "geom":
            return np.asarray(
                physics.named.data.geom_xpos[name], dtype=np.float64
            ).copy()
    except (KeyError, ValueError):
        return None
    return None


def get_current_claw_xy_samples(
    env: AlohaEnv,
    obs: dict,
    kin_helper: KinHelper,
    world_t_bases: np.ndarray,
) -> np.ndarray:
    """Sample core XY points on both open grippers for top-overlap rejection.

    Deliberately do not sample fingertip contact spheres or the open span between
    fingers: those are expected to touch/partly overlap the letter during valid
    side pushes, including two-claw contact.
    """
    physics = env._env.physics
    samples: list[np.ndarray] = []

    for arm in ("left", "right"):
        gripper = _get_named_xyz(physics, "site", f"{arm}/gripper")
        finger_sites = [
            _get_named_xyz(physics, "site", f"{arm}/left_finger"),
            _get_named_xyz(physics, "site", f"{arm}/right_finger"),
        ]

        samples.extend(p[:2] for p in [gripper, *finger_sites] if p is not None)

        if gripper is not None:
            for finger in finger_sites:
                if finger is not None:
                    _append_segment_samples(samples, gripper[:2], finger[:2], n=4)

    if samples:
        return np.asarray(samples, dtype=np.float64)

    # Fallback for unexpected MJCF naming changes: preserve the old EEF-center
    # behavior instead of silently disabling the safety check.
    return get_current_arm_positions(obs, kin_helper, world_t_bases).reshape(2, 2)


def are_eefs_outside_object_footprint(
    obs: dict,
    shape: str,
    kin_helper: KinHelper,
    world_t_bases: np.ndarray,
    inset: float = 0.004,
) -> bool:
    """Return whether EEF centers are outside the object's outer envelope."""
    object_info = extract_planar_object_info(obs["env_state"], shape)
    analyzer = PlanarObjectGeometryAnalyzer(object_info)
    points = analyzer.all_contact_points
    x_min, y_min = points.min(axis=0) + inset
    x_max, y_max = points.max(axis=0) - inset
    eef_xy = get_current_arm_positions(obs, kin_helper, world_t_bases).reshape(2, 2)
    inside = (
        (eef_xy[:, 0] > x_min)
        & (eef_xy[:, 0] < x_max)
        & (eef_xy[:, 1] > y_min)
        & (eef_xy[:, 1] < y_max)
    )
    return not bool(inside.any())


def are_claws_clear_of_object_footprint(
    env: AlohaEnv,
    obs: dict,
    shape: str,
    kin_helper: KinHelper,
    world_t_bases: np.ndarray,
    material_margin: float = -0.006,
) -> bool:
    """Reject only deep/core claw overlap with letter material.

    A negative margin shrinks each stroke before testing, so normal boundary
    contact from one or both claws is allowed.  This guard is only meant to catch
    obvious cases where the gripper body is on top of the letter.
    """
    object_info = extract_planar_object_info(obs["env_state"], shape)
    claw_xy = get_current_claw_xy_samples(env, obs, kin_helper, world_t_bases)
    return not bool(
        _points_inside_planar_object_strokes(
            claw_xy,
            object_info,
            margin=material_margin,
        ).any()
    )


def is_planar_push_frame_valid(
    env: AlohaEnv,
    obs: dict,
    shape: str,
    kin_helper: KinHelper,
    world_t_bases: np.ndarray,
) -> bool:
    """Return whether a procedural-letter frame is safe to keep."""
    return is_planar_object_pose_valid(obs["env_state"]) and is_planar_object_on_table(
        obs["env_state"], shape
    )


def trajectory_to_joint_actions(
    target_xy: np.ndarray,
    world_t_bases: np.ndarray,
    kin_helper: KinHelper,
    curr_puppet_joint: np.ndarray,
    curr_vel: np.ndarray,
    dt: float,
    k_p: float = 50,
    k_v: float = 10,
    acc_lim: float = 1.0,
    vel_lim: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2D trajectory to joint actions using the existing PID controller."""
    assert target_xy.shape == (4,), f"action shape must be (4,), got {target_xy.shape}"
    puppet_target_state = np.zeros(14)
    target_xy_clip = target_xy.copy()

    for rob_i in range(2):  # left and right arms
        # Target end-effector pose
        world_t_ee_pose = np.eye(4)
        world_t_ee_pose[:2, 3] = target_xy[rob_i * 2 : rob_i * 2 + 2]
        world_t_ee_pose[2, 3] = PLANAR_PUSH_EEF_Z

        theta = np.pi * 5.0 / 12.0
        target_ee_pose = np.linalg.inv(world_t_bases[rob_i]) @ world_t_ee_pose
        target_ee_pose[:3, :3] = np.array(
            [
                [np.sin(theta), 0.0, np.cos(theta)],
                [0.0, 1.0, 0.0],
                [-np.cos(theta), 0.0, np.sin(theta)],
            ]
        )
        target_ee_pose[0, 3] = np.clip(target_ee_pose[0, 3], 0.25, 0.6)
        target_ee_pose[1, 3] = np.clip(target_ee_pose[1, 3], -0.2, 0.2)
        target_ee_pose[2, 3] = PLANAR_PUSH_EEF_Z

        world_t_ee_pose_clip = world_t_bases[rob_i] @ target_ee_pose
        target_xy_clip[rob_i * 2 : rob_i * 2 + 2] = world_t_ee_pose_clip[:2, 3]

        # Get current joint positions
        ik_init_joint = np.concatenate(
            [curr_puppet_joint[7 * rob_i : 7 * rob_i + 6], np.zeros(2)]
        )

        # Compute current end-effector pose
        curr_ee_pose = kin_helper.compute_fk_from_link_idx(
            ik_init_joint, [kin_helper.sapien_eef_idx]
        )[0]

        # PID control
        pid_ee_pose = target_ee_pose.copy()
        acceleration = k_p * (target_ee_pose[:3, 3] - curr_ee_pose[:3, 3]) + k_v * (
            np.zeros(3) - curr_vel[rob_i * 3 : rob_i * 3 + 3]
        )
        acceleration = np.clip(acceleration, -acc_lim, acc_lim)
        next_vel = acceleration * dt + curr_vel[rob_i * 3 : rob_i * 3 + 3]
        next_vel = np.clip(next_vel, -vel_lim, vel_lim)
        avg_vel = (next_vel + curr_vel[rob_i * 3 : rob_i * 3 + 3]) / 2.0
        pid_ee_pose[:3, 3] = curr_ee_pose[:3, 3] + avg_vel * dt
        curr_vel[rob_i * 3 : rob_i * 3 + 3] = next_vel

        # Inverse kinematics. Use a fixed wide-open gripper so only XY
        # position changes; openness never changes during planar pushing.
        ik_joint = kin_helper.compute_ik_from_mat(ik_init_joint, pid_ee_pose)
        puppet_target_state[7 * rob_i : 7 * rob_i + 6] = ik_joint[:6]
        puppet_target_state[7 * rob_i + 6] = PLANAR_PUSH_GRIPPER

    return puppet_target_state, target_xy_clip


def init_episode() -> dict:
    episode = {
        "robot_bases": [],
        "env_state": [],
        "obs": {
            "joint_pos": [],
            "ee_pos": [],
            "images": {},
        },
        "action": [],
    }
    return episode


def dict_list_to_np(episode: dict) -> dict:
    for key in list(episode.keys()):
        if isinstance(episode[key], list):
            episode[key] = np.stack(episode[key], axis=0)
        elif isinstance(episode[key], dict):
            episode[key] = dict_list_to_np(episode[key])
    return episode


def episode_has_frames(episode: dict) -> bool:
    return len(episode["env_state"]) > 0


def save_episode(episode: dict, output_dir: str, episode_id: int) -> None:
    ### create config dict
    config_dict: dict = {
        "obs": {"images": {}},
    }
    episode = dict_list_to_np(episode)

    # Get first camera name for video saving
    cam_names = list(episode["obs"]["images"].keys())
    first_cam_name = cam_names[0] if cam_names else None

    for cam_name in cam_names:
        cam_height, cam_width = episode["obs"]["images"][cam_name].shape[1:3]
        color_save_kwargs = {
            "chunks": (1, cam_height, cam_width, 3),  # (1, 480, 640, 3)
            # "compression": "gzip",
            # "compression_opts": 9,
            "dtype": "uint8",
        }
        config_dict["obs"]["images"][cam_name] = color_save_kwargs

    ### save episode data
    episode_path = os.path.join(output_dir, f"episode_{episode_id}.hdf5")
    save_dict_to_hdf5(episode, config_dict, str(episode_path))

    ### save video
    if first_cam_name is not None:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, f"episode_{episode_id}.mp4")
        video_frames = episode["obs"]["images"][first_cam_name]

        # Get video dimensions
        num_frames, height, width, channels = video_frames.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        # Write frames to video
        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    print(f"Episode {episode_id} saved to {output_dir}!")


def visualize_object_keypoints_on_camera(
    img: np.ndarray,
    obs: dict,
    env: AlohaEnv,
    camera_name: str,
    shape: Optional[str] = None,
) -> np.ndarray:
    """Visualize planar object contact/key points projected onto camera image."""
    if shape is None:
        object_info = extract_t_info(obs["env_state"])
        analyzer = TGeometryAnalyzer(object_info)
    else:
        object_info = extract_planar_object_info(obs["env_state"], shape)
        analyzer = PlanarObjectGeometryAnalyzer(object_info)

    keypoints_3d = analyzer.contact_points
    keypoints_3d = np.concatenate(
        [keypoints_3d, 0.02 * np.ones((keypoints_3d.shape[0], 1))], axis=-1
    )
    world_t_kypts = keypoints_3d

    # Get camera intrinsics and extrinsics
    img_shape = img.shape[:2]  # (H, W)
    cam_intrinsics = env.get_cam_intrinsic(camera_name, img_shape)
    cam_extrinsics = env.get_cam_extrinsic(camera_name)
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]
    fx = cam_intrinsics[0, 0]
    fy = cam_intrinsics[1, 1]

    colors = None

    # Project and draw keypoints
    img = plot_single_3d_pos_traj(
        img=img.copy(),
        cam_intrisics=(cx, cy, fx, fy),
        world_t_cam=cam_extrinsics,
        trajs=world_t_kypts[None, :],
        radius=5,
        colors=colors,
    )

    return img


def vis_obs(
    obs: dict,
    episode_id: int,
    is_recording: bool,
    env: Optional[AlohaEnv] = None,
    trajectory: Optional[np.ndarray] = None,
    shape: Optional[str] = None,
) -> np.ndarray:
    third_views = ["top_pov"]

    # Create visualization with T-shape keypoints overlay
    for view_name in third_views:
        img = obs["images"][view_name].copy()

        # Add T-shape keypoint visualization for top_pov camera
        if view_name == "top_pov" and env is not None:
            img = visualize_object_keypoints_on_camera(img, obs, env, "top_pov", shape)
            if trajectory is not None:
                trajectory = trajectory.reshape(-1, 2, 2)
                trajectory = np.transpose(trajectory, (1, 0, 2))
                traj_3d = np.concatenate(
                    [
                        trajectory,
                        0.02 * np.ones((trajectory.shape[0], trajectory.shape[1], 1)),
                    ],
                    axis=-1,
                )
                cam_intrinsics = env.get_cam_intrinsic(view_name, img.shape[:-1])
                cam_extrinsics = env.get_cam_extrinsic(view_name)
                cx = cam_intrinsics[0, 2]
                cy = cam_intrinsics[1, 2]
                fx = cam_intrinsics[0, 0]
                fy = cam_intrinsics[1, 1]
                img = plot_single_3d_pos_traj(
                    img=img.copy(),
                    cam_intrisics=(cx, cy, fx, fy),
                    world_t_cam=cam_extrinsics,
                    trajs=traj_3d,
                    radius=5,
                )

        images_with_keypoints = img

    vis_img = images_with_keypoints
    text = f"Episode: {episode_id}"
    if is_recording:
        text += ", Recording!"
    vis_img = cv2.putText(
        vis_img.copy(),
        text,
        (10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
        color=(255, 255, 255),
    )
    return vis_img


def save_rejected_episode(
    episode: dict,
    output_dir: str,
    rejected_episode_id: int,
    obs: dict,
    action: np.ndarray,
    kin_helper: KinHelper,
) -> int:
    """Save an aborted rollout/snapshot for debugging invalid letter episodes."""
    rejected_dir = os.path.join(output_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    episode = update_episode(episode, obs, action, kin_helper)
    save_episode(episode, rejected_dir, rejected_episode_id)
    return rejected_episode_id + 1


def update_episode(
    episode: dict, obs: dict, action: np.ndarray, kin_helper: KinHelper
) -> dict:
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    world_t_left_base = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_right_base = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    all_bases = np.stack([world_t_left_base, world_t_right_base])

    episode["robot_bases"].append(all_bases)
    episode["env_state"].append(obs["env_state"])
    episode["obs"]["joint_pos"].append(obs["qpos"])

    # compute FK for obs
    left_qpos = obs["qpos"][:7]
    left_qpos = np.concatenate([left_qpos, left_qpos[6:7]])
    right_qpos = obs["qpos"][7:]
    right_qpos = np.concatenate([right_qpos, right_qpos[6:7]])
    left_base_t_left_eef = kin_helper.compute_fk_from_link_idx(
        left_qpos, [kin_helper.sapien_eef_idx]
    )[0]
    right_base_t_right_eef = kin_helper.compute_fk_from_link_idx(
        right_qpos, [kin_helper.sapien_eef_idx]
    )[0]
    world_t_left_eef = world_t_left_base @ left_base_t_left_eef
    world_t_right_eef = world_t_right_base @ right_base_t_right_eef
    world_t_all_eef = np.stack([world_t_left_eef, world_t_right_eef])
    episode["obs"]["ee_pos"].append(world_t_all_eef)

    for camera_name in list(obs["images"].keys()):
        if camera_name not in episode["obs"]["images"]:
            episode["obs"]["images"][camera_name] = []
        img = obs["images"][camera_name]
        crop_img = center_crop(img, (128, 128))
        resize_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_AREA)
        episode["obs"]["images"][camera_name].append(resize_img)

    episode["action"].append(action)

    return episode


def generate_random_init_action(
    world_t_bases: np.ndarray,
    obs: Optional[dict] = None,
    shape: Optional[str] = None,
) -> np.ndarray:
    if shape is not None and obs is not None:
        object_info = extract_planar_object_info(obs["env_state"], shape)
        analyzer = PlanarObjectGeometryAnalyzer(object_info)
        points = analyzer.all_contact_points
        _, y_min = points.min(axis=0)
        _, y_max = points.max(axis=0)
        center_y = (y_min + y_max) / 2.0 + np.random.uniform(-0.015, 0.015)
        left_xy = np.array([-0.22, center_y])
        right_xy = np.array([0.22, center_y])
        left_xy[0] = np.clip(left_xy[0], -0.25, 0.25)
        right_xy[0] = np.clip(right_xy[0], -0.25, 0.25)
        left_xy[1] = np.clip(left_xy[1], -0.2, 0.2)
        right_xy[1] = np.clip(right_xy[1], -0.2, 0.2)
        return np.concatenate([left_xy, right_xy])

    rand_init_action = np.random.uniform(0, 1, size=(4,))
    rand_init_action[0] = rand_init_action[0] * 0.2 + 0.05
    rand_init_action[1] = rand_init_action[1] * 0.4 - 0.2
    rand_init_action[2] = rand_init_action[2] * 0.2 + 0.05
    rand_init_action[3] = rand_init_action[3] * 0.4 - 0.2
    left_base_t_left_action = np.eye(4)
    left_base_t_left_action[:2, 3] = rand_init_action[:2]
    right_base_t_right_action = np.eye(4)
    right_base_t_right_action[:2, 3] = rand_init_action[2:]
    world_t_left_action = world_t_bases[0] @ left_base_t_left_action
    world_t_right_action = world_t_bases[1] @ right_base_t_right_action
    action = np.concatenate([world_t_left_action[:2, 3], world_t_right_action[:2, 3]])
    return action


def task_reset(
    env: AlohaEnv,
    episode_id: int,
    kin_helper: KinHelper,
    k_p: float,
    k_v: float,
    acc_lim: float,
    vel_lim: float,
    headless: bool,
    shape: Optional[str] = None,
) -> None:
    env.reset(seed=int(time.time()))
    if shape is not None:
        freeze_planar_grippers_open(env)
    obs = env._env.task.get_observation(env._env.physics)
    reset_object_pose = obs["env_state"].copy() if shape is not None else None
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    left_base_mat = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    right_base_mat = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_bases = np.stack([left_base_mat, right_base_mat])
    rand_init_action = generate_random_init_action(world_t_bases, obs, shape)
    curr_vel = np.zeros(6)
    curr_puppet_joint = obs["qpos"][:14]
    dt = 1 / 10.0
    for _ in range(100):
        obs = env._env.task.get_observation(env._env.physics)
        curr_puppet_joint = obs["qpos"][:14]
        joint_actions, _ = trajectory_to_joint_actions(
            rand_init_action,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )
        env.step(joint_actions)
        if shape is not None:
            freeze_planar_grippers_open(env)
        obs = env._env.task.get_observation(env._env.physics)
        if not headless:
            vis_img = vis_obs(obs, episode_id, False, env, shape=shape)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Aloha Dataset Collection", vis_img)
            cv2.waitKey(1)

    if shape is not None and reset_object_pose is not None:
        restore_planar_object_pose(env, reset_object_pose)
        freeze_planar_grippers_open(env)


@click.command()
@click.option(
    "--output_dir", "-o", default=".", help="Directory to save demonstration dataset."
)
@click.option(
    "--motion_type",
    "-mt",
    default="random_no_contact",
    help=(
        "Motion type: linear, rotating, random_contact, random_no_contact, "
        "mixed, or rl_coverage."
    ),
)
@click.option(
    "--shape",
    "-s",
    default=None,
    help=(
        "Optional procedural block-letter shape (A-Z). If omitted, uses the "
        "legacy mesh PushT environment."
    ),
)
@click.option("--headless", "-h", is_flag=True, help="Run in headless mode.")
@click.option(
    "--episode_steps",
    default=DEFAULT_EPISODE_STEPS,
    type=int,
    show_default=True,
    help="Number of recorded frames per saved episode/video.",
)
@click.option(
    "--motion_speedup",
    default=DEFAULT_MOTION_SPEEDUP,
    type=float,
    show_default=True,
    help="Resample each planned motion segment by this factor before execution.",
)
@click.option(
    "--rl_warmup_episodes",
    default=DEFAULT_RL_WARMUP_EPISODES,
    type=int,
    show_default=True,
    help=(
        "For rl_coverage only: number of full episode horizons to use for "
        "online policy learning before saving successful HDF5/MP4 episodes."
    ),
)
@click.option(
    "--rl_checkpoint_path",
    default=None,
    type=str,
    help=(
        "For rl_coverage only: JSON checkpoint path for the online policy. "
        "Default: <output_dir>/rl_policy_checkpoint.json. Existing checkpoints "
        "are loaded automatically."
    ),
)
@click.option(
    "--rl_checkpoint_interval",
    default=DEFAULT_RL_CHECKPOINT_INTERVAL,
    type=int,
    show_default=True,
    help=(
        "For rl_coverage only: save the online policy checkpoint every N full "
        "episode horizons. Abort penalties are checkpointed immediately."
    ),
)
def main(
    output_dir: str,
    motion_type: str = "random_no_contact",
    shape: Optional[str] = None,
    headless: bool = False,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
    motion_speedup: float = DEFAULT_MOTION_SPEEDUP,
    rl_warmup_episodes: int = DEFAULT_RL_WARMUP_EPISODES,
    rl_checkpoint_path: Optional[str] = None,
    rl_checkpoint_interval: int = DEFAULT_RL_CHECKPOINT_INTERVAL,
) -> None:
    if motion_type not in SUPPORTED_MOTION_TYPES:
        valid = ", ".join(sorted(SUPPORTED_MOTION_TYPES))
        raise ValueError(f"Unknown motion_type '{motion_type}'. Valid: {valid}")
    if episode_steps <= 0:
        raise ValueError(f"episode_steps must be positive, got {episode_steps}")
    if motion_type == "rl_coverage" and episode_steps > MAX_RL_COVERAGE_EPISODE_STEPS:
        raise ValueError(
            "rl_coverage is designed for <=20s videos; use "
            f"--episode_steps {MAX_RL_COVERAGE_EPISODE_STEPS} or lower"
        )
    if motion_speedup <= 0.0:
        raise ValueError(f"motion_speedup must be positive, got {motion_speedup}")
    if rl_warmup_episodes < 0:
        raise ValueError(
            f"rl_warmup_episodes must be non-negative, got {rl_warmup_episodes}"
        )
    if rl_checkpoint_interval <= 0:
        raise ValueError(
            f"rl_checkpoint_interval must be positive, got {rl_checkpoint_interval}"
        )
    if motion_type != "rl_coverage" and rl_warmup_episodes > 0:
        print(
            "WARNING: --rl_warmup_episodes is only used by "
            f"--motion_type rl_coverage; ignoring value {rl_warmup_episodes}."
        )
    if motion_type != "rl_coverage" and rl_checkpoint_path is not None:
        print(
            "WARNING: --rl_checkpoint_path is only used by "
            f"--motion_type rl_coverage; ignoring value {rl_checkpoint_path}."
        )

    frequency = 10.0
    dt = 1 / frequency
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    kin_helper = KinHelper(robot_name="trossen_vx300s")

    # enter contexts
    shm_manager = SharedMemoryManager()
    shm_manager.__enter__()

    curr_vel = np.zeros(6)
    k_p, k_v = 50, 10  # PD control
    acc_lim = 10.0
    vel_lim = 0.04
    shape = shape.upper() if shape else None
    env = AlohaEnv("planar_push", shape=shape) if shape else AlohaEnv("pusht")
    cv2.setNumThreads(1)

    time.sleep(1.0)
    resolved_rl_checkpoint_path = None
    if motion_type == "rl_coverage":
        resolved_rl_checkpoint_path = (
            Path(rl_checkpoint_path)
            if rl_checkpoint_path
            else Path(output_dir) / RL_CHECKPOINT_FILENAME
        )

    print(
        f"Ready! episode_steps={episode_steps}, motion_speedup={motion_speedup}, "
        f"video_seconds={episode_steps / 30.0:.1f} at 30 FPS, "
        f"rl_warmup_episodes={rl_warmup_episodes}, "
        f"rl_checkpoint_path={resolved_rl_checkpoint_path}"
    )
    t_start = time.monotonic()
    iter_idx = 0
    episode_id = len(list(Path(output_dir).glob("episode_*.hdf5")))
    rejected_dir = Path(output_dir) / "rejected"
    rejected_episode_id = (
        len(list(rejected_dir.glob("episode_*.hdf5"))) if rejected_dir.exists() else 0
    )
    init_episode_id = episode_id
    stop = False
    is_recording = False
    episode = init_episode()

    # sample to a random init action
    task_reset(env, episode_id, kin_helper, k_p, k_v, acc_lim, vel_lim, headless, shape)

    # Initialize scripted policy components
    workspace_constraints = WorkspaceConstraints(
        x_min=-0.25,
        x_max=0.25,
        y_min=-0.2,
        y_max=0.2,
    )
    motion_planner = MotionPlanner(workspace_constraints)
    if motion_type == "rl_coverage" and resolved_rl_checkpoint_path is not None:
        if motion_planner.load_rl_checkpoint(resolved_rl_checkpoint_path):
            print(
                "Loaded RL coverage checkpoint from "
                f"{resolved_rl_checkpoint_path} "
                f"with {motion_planner.rl_total_updates} updates."
            )
        else:
            print(
                "No existing RL coverage checkpoint found; will save to "
                f"{resolved_rl_checkpoint_path}."
            )
    current_trajectory = None
    trajectory_step = 0
    episode_step = 0

    # Auto-recording setup
    auto_record = True  # Set to True for automatic data collection
    episode_start_time = None

    # compute world_t_bases
    obs = env._env.task.get_observation(env._env.physics)
    left_base = obs["left_base"][None]
    right_base = obs["right_base"][None]
    left_base_mat = pose_convert(left_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    right_base_mat = pose_convert(right_base, PoseType.POS_QUAT, PoseType.MAT)[0]
    world_t_bases = np.stack([left_base_mat, right_base_mat])
    trial_num = 0
    trajectory_steps = 0
    rl_warmup_completed = 0
    if motion_type == "rl_coverage":
        loaded_warmup_completed = int(
            motion_planner.loaded_rl_checkpoint_extra_state.get(
                "rl_warmup_completed", 0
            )
        )
        rl_warmup_completed = min(loaded_warmup_completed, rl_warmup_episodes)
        if loaded_warmup_completed > 0:
            print(
                f"Resumed RL warmup progress: {rl_warmup_completed}/"
                f"{rl_warmup_episodes}."
            )
    rl_checkpoint_events = 0

    def maybe_save_rl_checkpoint(reason: str, force: bool = False) -> None:
        nonlocal rl_checkpoint_events
        if motion_type != "rl_coverage" or resolved_rl_checkpoint_path is None:
            return
        rl_checkpoint_events += 1
        if force or rl_checkpoint_events % rl_checkpoint_interval == 0:
            motion_planner.save_rl_checkpoint(
                resolved_rl_checkpoint_path,
                extra_state={
                    "rl_warmup_completed": rl_warmup_completed,
                    "rl_warmup_episodes": rl_warmup_episodes,
                },
            )
            print(
                f"Saved RL coverage checkpoint ({reason}) to "
                f"{resolved_rl_checkpoint_path} "
                f"with {motion_planner.rl_total_updates} updates."
            )

    while not stop:
        # pump obs
        obs = env._env.task.get_observation(env._env.physics)  # noqa

        rl_warmup_active = (
            motion_type == "rl_coverage" and rl_warmup_completed < rl_warmup_episodes
        )

        if shape is not None and not is_planar_push_frame_valid(
            env, obs, shape, kin_helper, world_t_bases
        ):
            if motion_type == "rl_coverage":
                try:
                    motion_planner.end_rl_episode(
                        extract_planar_object_info(obs["env_state"], shape),
                        aborted=True,
                    )
                except Exception:
                    motion_planner.end_rl_episode(aborted=True)
            if rl_warmup_active:
                print(
                    "RL warmup rollout aborted: object tipped/lifted/out of table. "
                    "Policy received abort penalty; not saving rejected snapshot. "
                    "Resetting."
                )
            else:
                reject_action = get_current_arm_positions(
                    obs, kin_helper, world_t_bases
                )
                rejected_episode_id = save_rejected_episode(
                    episode if episode_has_frames(episode) else init_episode(),
                    output_dir,
                    rejected_episode_id,
                    obs,
                    reject_action,
                    kin_helper,
                )
                print(
                    f"Episode {episode_id} aborted and saved as rejected "
                    f"episode {rejected_episode_id - 1}: object tipped/lifted/out of table. "
                    "Resetting."
                )
            if motion_type == "rl_coverage":
                maybe_save_rl_checkpoint("abort", force=True)
            episode = init_episode()
            task_reset(
                env,
                episode_id,
                kin_helper,
                k_p,
                k_v,
                acc_lim,
                vel_lim,
                headless,
                shape,
            )
            is_recording = False
            episode_start_time = None
            current_trajectory = None
            trajectory_step = 0
            episode_step = 0
            if not rl_warmup_active:
                trial_num += 1
            continue

        # Auto-recording logic
        if auto_record and not is_recording:
            is_recording = True
            episode_start_time = time.monotonic()
            current_trajectory = None  # Force new trajectory generation
            trajectory_step = 0
            episode_step = 0
            if motion_type == "rl_coverage":
                motion_planner.reset_rl_episode()
            if rl_warmup_active:
                print(
                    f"Auto-started RL warmup episode {rl_warmup_completed + 1}/"
                    f"{rl_warmup_episodes}"
                )
            else:
                print(f"Auto-started recording episode {episode_id}")

        if auto_record and is_recording and episode_start_time is not None:
            # Check if the fixed episode horizon elapsed. Individual motion
            # segments can be shorter; the loop replans more segments until this
            # full recording horizon is reached.
            if episode_step >= episode_steps and episode_has_frames(episode):
                # Auto-save episode
                init_pose = env_state_to_mat(episode["env_state"][0])
                final_pose = env_state_to_mat(episode["env_state"][-1])
                actions = np.stack(episode["action"])
                if motion_type == "rl_coverage":
                    if shape is None:
                        final_object_info = extract_t_info(episode["env_state"][-1])
                    else:
                        final_object_info = extract_planar_object_info(
                            episode["env_state"][-1], shape
                        )
                    motion_planner.end_rl_episode(final_object_info)
                episode_success = evaluate_collection_success(
                    motion_type, init_pose, final_pose, actions
                )
                if rl_warmup_active:
                    rl_warmup_completed += 1
                    status = "successful" if episode_success else "not successful"
                    print(
                        f"RL warmup episode {rl_warmup_completed}/"
                        f"{rl_warmup_episodes} completed ({status}); "
                        "policy updated, not saving HDF5/MP4."
                    )
                    if rl_warmup_completed >= rl_warmup_episodes:
                        print(
                            "RL warmup complete; future successful episodes will save."
                        )
                else:
                    if episode_success:
                        print(f"Episode {episode_id} was successful!")
                        save_episode(episode, output_dir, episode_id)
                        episode_id += 1
                    else:
                        print(f"Episode {episode_id} was NOT successful!")
                    trial_num += 1
                    print(
                        "Current success rate: ",
                        float(episode_id - init_episode_id) / float(trial_num),
                    )
                if motion_type == "rl_coverage":
                    maybe_save_rl_checkpoint("episode")
                episode = init_episode()
                task_reset(
                    env,
                    episode_id,
                    kin_helper,
                    k_p,
                    k_v,
                    acc_lim,
                    vel_lim,
                    headless,
                    shape,
                )
                is_recording = False
                current_trajectory = None
                trajectory_step = 0
                episode_step = 0
                print(
                    f"One episode takes {time.monotonic() - episode_start_time} seconds"
                )
                episode_start_time = None

        # visualize
        vis_img = vis_obs(obs, episode_id, is_recording, env, current_trajectory, shape)
        if not headless:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Aloha Dataset Collection", vis_img)
            cv2.waitKey(1)

        # Generate scripted policy actions
        puppet_target_state = np.zeros(14)

        # Check if we need to generate a new faster motion segment. A saved
        # episode can contain multiple planned segments, so the video stays long
        # while per-frame displacement is larger.
        if current_trajectory is None or trajectory_step >= len(current_trajectory):
            success = False
            while not success:
                obs = env._env.task.get_observation(env._env.physics)

                # Extract pushed-object information.  `shape is None` keeps the
                # legacy mesh PushT geometry; any shape value uses the generic
                # procedural block-letter geometry.
                if shape is None:
                    object_info = extract_t_info(obs["env_state"])
                else:
                    object_info = extract_planar_object_info(obs["env_state"], shape)

                # Get current arm positions
                current_arm_pos = get_current_arm_positions(
                    obs, kin_helper, world_t_bases
                )

                # Plan a segment, then resample it to execute the same geometric
                # path in fewer steps.
                current_trajectory, success, _segment_success_fn, trajectory_steps = (
                    motion_planner.plan_episode(
                        object_info, current_arm_pos, motion_type
                    )
                )
                if success:
                    current_trajectory = speed_up_trajectory(
                        current_trajectory, motion_speedup
                    )
                    trajectory_steps = len(current_trajectory)
            trajectory_step = 0

        # Get target positions from trajectory
        if current_trajectory is not None and trajectory_step < len(current_trajectory):
            target_xy = current_trajectory[trajectory_step]
        else:
            # Fallback to current position
            target_xy = get_current_arm_positions(obs, kin_helper, world_t_bases)

        # Convert to joint actions using the existing PID control logic
        curr_puppet_joint = obs["qpos"][:14]
        puppet_target_state, target_xy_clip = trajectory_to_joint_actions(
            target_xy,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )

        trajectory_step += 1

        if is_recording:
            episode = update_episode(episode, obs, target_xy_clip, kin_helper)
            episode_step += 1

        # execute teleop command
        env.step(puppet_target_state)
        if shape is not None:
            freeze_planar_grippers_open(env)
        iter_idx += 1

    # exit contexts
    shm_manager.__exit__(None, None, None)


# %%
if __name__ == "__main__":
    main()
