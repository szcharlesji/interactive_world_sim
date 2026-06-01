"""Motion planner for generating human-like bimanual planar pushing trajectories.
Generates linear, rotating, random contact, random no-contact, and mixed motions.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gym_aloha.planar_objects import PlanarObjectSpec

from .trajectory_primitives import (
    BimanualCoordination,
    CurvePrimitive,
    LinearPrimitive,
    StabilizePrimitive,
    TrajectoryConfig,
)


@dataclass
class TShapeInfo:
    """Information about T-shape pose and keypoints."""

    center: np.ndarray  # (x, y) center of T
    rotation: float  # rotation angle in radians
    keypoints: np.ndarray  # Nx2 array of keypoint positions
    width: float  # width of T top bar
    height: float  # height of T stem
    thickness: float  # thickness of T
    pose: np.ndarray  # world_t_obj


@dataclass
class PlanarObjectInfo:
    """Shape-generic planar object pose and geometry spec."""

    center: np.ndarray
    rotation: float
    pose: np.ndarray
    spec: PlanarObjectSpec


@dataclass
class WorkspaceConstraints:
    """Workspace constraints for arm movements."""

    x_min: float = 0.2
    x_max: float = 0.8
    y_min: float = -0.3
    y_max: float = 0.3
    min_arm_distance: float = 0.1  # Minimum distance between arms
    min_t_distance: float = 0.05  # Minimum distance from T when not contacting


class TGeometryAnalyzer:
    """Analyzes T-shape geometry and computes manipulation targets."""

    def __init__(self, t_info: TShapeInfo):
        self.t_info = t_info
        self._compute_geometric_features()

    def _compute_geometric_features(self):
        """Compute key geometric features of the T."""
        center = self.t_info.center
        rotation = self.t_info.rotation

        # Rotation matrix
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        # Key points in local coordinates
        local_points = {
            "top_left": self.t_info.keypoints[0, :2],
            "top_right": self.t_info.keypoints[1, :2],
            "top_bar_bottom_side_left": self.t_info.keypoints[2, :2],
            "top_bar_bottom_side_right": self.t_info.keypoints[3, :2],
            "stem_top_left": self.t_info.keypoints[4, :2],
            "stem_top_right": self.t_info.keypoints[5, :2],
            "stem_bottom_left": self.t_info.keypoints[6, :2],
            "stem_bottom_right": self.t_info.keypoints[7, :2],
        }

        # Transform to world coordinates
        self.key_points = {}
        for name, local_pt in local_points.items():
            self.key_points[name] = center + R @ local_pt

        # Compute contact regions
        self._compute_contact_regions()

    def _compute_contact_regions(self):
        """Compute feasible contact points along T edges."""
        # Generate contact points along edges
        self.contact_points = []
        kypts_order = [
            "top_left",
            "top_right",
            "top_bar_bottom_side_right",
            "stem_top_right",
            "stem_bottom_right",
            "stem_bottom_left",
            "stem_top_left",
            "top_bar_bottom_side_left",
        ]

        for i in range(len(kypts_order)):
            last_kypt = self.key_points[kypts_order[i - 1]]
            curr_kypt = self.key_points[kypts_order[i]]
            side_len = np.linalg.norm(last_kypt - curr_kypt)
            weights = np.arange(0, side_len, 0.002) / side_len
            world_pts = last_kypt[None] + weights[:, None] * (
                curr_kypt[None] - last_kypt[None]
            )
            self.contact_points.append(world_pts)

        self.contact_points = np.concatenate(self.contact_points)

    def select_contact_point(self, side: str) -> np.ndarray:
        """Select contact points based on side preference.

        Args:
            side: "left", "right", "up", or "down" to select points on the specified side

        Returns:
            Selected contact points as (N, 2) array
        """
        if side not in ["left", "right", "up", "down"]:
            raise ValueError(
                f"Side must be 'left', 'right', 'up', or 'down', got '{side}'"
            )

        # Flatten all contact points from all edges into a single array
        all_points = self.contact_points.reshape(-1, 2)

        if len(all_points) == 0:
            return np.array([]).reshape(0, 2)

        # Use a tolerance for grouping similar coordinates
        tolerance = 0.002  # 2mm tolerance for coordinate grouping

        if side in ["left", "right"]:
            # Group by y-coordinate and select by x-coordinate
            return self._select_by_x_coordinate(all_points, side, tolerance)
        else:  # side in ["up", "down"]
            # Group by x-coordinate and select by y-coordinate
            return self._select_by_y_coordinate(all_points, side, tolerance)

    def _select_by_x_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        """Group by y-coordinate and select by x-coordinate (left/right)."""
        # Sort points by y coordinate first
        sorted_indices = np.argsort(all_points[:, 1])
        sorted_points = all_points[sorted_indices]

        selected_points = []
        current_y_group = []
        current_y_ref = None

        for point in sorted_points:
            x, y = point

            # Check if this point belongs to the current y group
            if current_y_ref is None or abs(y - current_y_ref) <= tolerance:
                current_y_group.append(point)
                current_y_ref = y if current_y_ref is None else current_y_ref
            else:
                # Process the current group and start a new one
                if current_y_group:
                    selected_point = self._select_point_from_group_by_x(
                        current_y_group, side
                    )
                    selected_points.append(selected_point)

                current_y_group = [point]
                current_y_ref = y

        # Process the last group
        if current_y_group:
            selected_point = self._select_point_from_group_by_x(current_y_group, side)
            selected_points.append(selected_point)

        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_by_y_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        """Group by x-coordinate and select by y-coordinate (up/down)."""
        # Sort points by x coordinate first
        sorted_indices = np.argsort(all_points[:, 0])
        sorted_points = all_points[sorted_indices]

        selected_points = []
        current_x_group = []
        current_x_ref = None

        for point in sorted_points:
            x, y = point

            # Check if this point belongs to the current x group
            if current_x_ref is None or abs(x - current_x_ref) <= tolerance:
                current_x_group.append(point)
                current_x_ref = x if current_x_ref is None else current_x_ref
            else:
                # Process the current group and start a new one
                if current_x_group:
                    selected_point = self._select_point_from_group_by_y(
                        current_x_group, side
                    )
                    selected_points.append(selected_point)

                current_x_group = [point]
                current_x_ref = x

        # Process the last group
        if current_x_group:
            selected_point = self._select_point_from_group_by_y(current_x_group, side)
            selected_points.append(selected_point)

        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_point_from_group_by_x(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        """Select the leftmost or rightmost point from a group of points with similar y values."""
        if not points:
            return np.array([])

        points_array = np.array(points)
        x_coords = points_array[:, 0]

        if side == "left":
            # Select point with minimum x coordinate
            min_x_idx = np.argmin(x_coords)
        else:  # side == "right"
            # Select point with maximum x coordinate
            min_x_idx = np.argmax(x_coords)

        return points_array[min_x_idx]

    def _select_point_from_group_by_y(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        """Select the topmost or bottommost point from a group of points with similar x values."""
        if not points:
            return np.array([])

        points_array = np.array(points)
        y_coords = points_array[:, 1]

        if side == "up":
            # Select point with maximum y coordinate
            max_y_idx = np.argmax(y_coords)
        else:  # side == "down"
            # Select point with minimum y coordinate
            max_y_idx = np.argmin(y_coords)

        return points_array[max_y_idx]

    def get_linear_push_waypoints(
        self, push_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        """Generate waypoints for linear pushing motions."""
        push_distance = 0.15
        approach_distance = 0.08

        if push_direction == "horizontal_right":
            # Push T horizontally to the right
            left_contacts = self.select_contact_point("left")
            left_contact = left_contacts[np.random.choice(len(left_contacts))]

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.cos(left_approach_theta), np.sin(left_approach_theta)]
            )
            left_approach = left_contact - left_approach_dir * approach_distance

            # Target points
            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.cos(left_target_theta), np.sin(left_target_theta)]
            )
            left_target = left_contact + left_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [],
            }

        elif push_direction == "horizontal_left":
            # Push T horizontally to the left
            right_contacts = self.select_contact_point("right")
            right_contact = right_contacts[np.random.choice(len(right_contacts))]

            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_dir = np.array(
                [np.cos(right_approach_theta), np.sin(right_approach_theta)]
            )
            right_approach = right_contact + right_approach_dir * approach_distance

            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_dir = np.array(
                [np.cos(right_target_theta), np.sin(right_target_theta)]
            )
            right_target = right_contact - right_target_dir * push_distance

            return {
                "left": [],
                "right": [right_approach, right_contact, right_target],
            }

        elif push_direction == "vertical_up":
            # Push T vertically upward
            up_contacts = self.select_contact_point("down")
            contacts = up_contacts[np.random.choice(len(up_contacts), 2)]
            left_contact = contacts[contacts[:, 0].argmin()]
            right_contact = contacts[contacts[:, 0].argmax()]

            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_contact - left_approach_dir * approach_distance
            right_approach = right_contact - right_approach_dir * approach_distance

            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.sin(left_target_theta), np.cos(left_target_theta)]
            )
            right_target_dir = np.array(
                [np.sin(right_target_theta), np.cos(right_target_theta)]
            )
            left_target = left_contact + left_target_dir * push_distance
            right_target = right_contact + right_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [right_approach, right_contact, right_target],
            }

        elif push_direction == "vertical_down":
            # Push T vertically downward
            down_contacts = self.select_contact_point("up")
            contacts = np.random.choice(len(down_contacts), 2)
            contacts = down_contacts[contacts]
            left_contact = contacts[contacts[:, 0].argmin()]
            right_contact = contacts[contacts[:, 0].argmax()]

            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_contact + left_approach_dir * approach_distance
            right_approach = right_contact + right_approach_dir * approach_distance

            left_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_target_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_target_dir = np.array(
                [np.sin(left_target_theta), np.cos(left_target_theta)]
            )
            right_target_dir = np.array(
                [np.sin(right_target_theta), np.cos(right_target_theta)]
            )
            left_target = left_contact - left_target_dir * push_distance
            right_target = right_contact - right_target_dir * push_distance

            return {
                "left": [left_approach, left_contact, left_target],
                "right": [right_approach, right_contact, right_target],
            }

        else:
            raise ValueError(f"Unknown push direction: {push_direction}")

    def get_rotation_waypoints(
        self, rotation_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        """Generate waypoints for rotating the T."""
        rotation_angle = np.pi / 6  # 30 degrees
        approach_distance = 0.06

        if rotation_direction == "clockwise":
            angle = self.t_info.rotation
            # T pointing down
            if angle > -np.pi / 4 and angle < np.pi / 4:
                left_grip = self.key_points["top_bar_bottom_side_left"]
                right_grip = self.key_points["top_right"]
            # T pointing right
            elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
                left_grip = self.key_points["top_left"]
                right_grip = self.key_points["stem_bottom_right"]
            # T pointing up
            elif (angle > 3 * np.pi / 4 and angle < np.pi) or (
                angle > -np.pi and angle < -3 * np.pi / 4
            ):
                left_grip = self.key_points["top_right"]
                right_grip = self.key_points["top_bar_bottom_side_left"]
            # T pointing left
            elif angle > -3 * np.pi / 4 and angle < -np.pi / 4:
                left_grip = self.key_points["stem_bottom_right"]
                right_grip = self.key_points["top_left"]
            else:
                raise ValueError(f"Unknown rotation direction: {rotation_direction}")

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_grip - left_approach_dir * approach_distance
            right_approach = right_grip + right_approach_dir * approach_distance

            # Rotation around center
            center = (left_grip + right_grip) / 2

            # Calculate rotated positions
            left_rel = left_grip - center
            right_rel = right_grip - center

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            R_rot = np.array([[cos_r, sin_r], [-sin_r, cos_r]])  # Clockwise rotation

            left_rotated = center + R_rot @ left_rel
            right_rotated = center + R_rot @ right_rel

            return {
                "left": [left_approach, left_grip, left_rotated],
                "right": [right_approach, right_grip, right_rotated],
            }

        elif rotation_direction == "counterclockwise":
            # Rotate T counterclockwise - flip right and left
            angle = self.t_info.rotation
            if angle > -np.pi / 4 and angle < np.pi / 4:
                left_grip = self.key_points["top_left"]
                right_grip = self.key_points["top_bar_bottom_side_right"]
            elif angle > np.pi / 4 and angle < 3 * np.pi / 4:
                left_grip = self.key_points["top_right"]
                right_grip = self.key_points["stem_bottom_left"]
            # pointing up
            elif (angle > 3 * np.pi / 4 and angle < np.pi) or (
                angle > -np.pi and angle < -3 * np.pi / 4
            ):
                left_grip = self.key_points["top_bar_bottom_side_right"]
                right_grip = self.key_points["top_left"]
            elif angle > -3 * np.pi / 4 and angle < -np.pi / 4:
                left_grip = self.key_points["stem_bottom_left"]
                right_grip = self.key_points["top_right"]
            else:
                raise ValueError(f"Unknown rotation direction: {rotation_direction}")

            # Approach points
            left_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            right_approach_theta = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
            left_approach_dir = np.array(
                [np.sin(left_approach_theta), np.cos(left_approach_theta)]
            )
            right_approach_dir = np.array(
                [np.sin(right_approach_theta), np.cos(right_approach_theta)]
            )
            left_approach = left_grip + left_approach_dir * approach_distance
            right_approach = right_grip - right_approach_dir * approach_distance

            # Rotation around center
            center = (left_grip + right_grip) / 2
            left_rel = left_grip - center
            right_rel = right_grip - center

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            R_rot = np.array(
                [[cos_r, -sin_r], [sin_r, cos_r]]
            )  # Counterclockwise rotation

            left_rotated = center + R_rot @ left_rel
            right_rotated = center + R_rot @ right_rel

            return {
                "left": [left_approach, left_grip, left_rotated],
                "right": [right_approach, right_grip, right_rotated],
            }

        else:
            raise ValueError(f"Unknown rotation direction: {rotation_direction}")


class PlanarObjectGeometryAnalyzer:
    """Shape-generic planar object geometry analyzer.

    Contact points are sampled from the perimeter of each box part in the
    object's compound block-letter spec.  The public methods intentionally match
    `TGeometryAnalyzer` so the existing `MotionPlanner` can operate on either.
    """

    def __init__(self, object_info: PlanarObjectInfo):
        self.t_info = object_info
        self.object_info = object_info
        self._compute_contact_regions()

    def _transform_local_points(self, points: np.ndarray) -> np.ndarray:
        cos_r, sin_r = (
            np.cos(self.object_info.rotation),
            np.sin(self.object_info.rotation),
        )
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        return self.object_info.center[None] + points @ rot.T

    def _compute_contact_regions(self) -> None:
        contact_points = []
        for part in self.object_info.spec.parts:
            hx, hy = part.size[:2]
            corners = np.array(
                [
                    [-hx, -hy],
                    [hx, -hy],
                    [hx, hy],
                    [-hx, hy],
                ]
            )
            part_cos, part_sin = np.cos(part.yaw), np.sin(part.yaw)
            part_rot = np.array([[part_cos, -part_sin], [part_sin, part_cos]])
            corners = np.array(part.pos[:2])[None] + corners @ part_rot.T
            for i in range(len(corners)):
                start = corners[i]
                end = corners[(i + 1) % len(corners)]
                side_len = max(np.linalg.norm(end - start), 1e-8)
                n = max(int(side_len / 0.004), 2)
                weights = np.linspace(0.0, 1.0, n, endpoint=False)
                contact_points.append(
                    start[None] + weights[:, None] * (end - start)[None]
                )
        local_contact_points = np.concatenate(contact_points, axis=0)
        self.all_contact_points = self._transform_local_points(local_contact_points)
        self.contact_points = self._filter_exterior_contact_points(
            self.all_contact_points
        )

    def _filter_exterior_contact_points(self, points: np.ndarray) -> np.ndarray:
        """Keep mostly exterior contacts to avoid planning into letter holes.

        Compound letters such as H/A/O have internal gaps where a two-finger
        gripper can hook the object.  For scripted data collection we prefer
        stable outer pushes, so random/contact planners sample from points near
        the outer envelope rather than every internal bar edge.
        """
        if len(points) <= 8:
            return points
        center = self.object_info.center
        radius = np.linalg.norm(points - center[None], axis=1)
        radial_threshold = np.quantile(radius, 0.60)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        bbox_margin = 0.018
        exterior_mask = (
            (radius >= radial_threshold)
            | (points[:, 0] <= x_min + bbox_margin)
            | (points[:, 0] >= x_max - bbox_margin)
            | (points[:, 1] <= y_min + bbox_margin)
            | (points[:, 1] >= y_max - bbox_margin)
        )
        exterior_points = points[exterior_mask]
        return exterior_points if len(exterior_points) >= 8 else points

    def select_contact_point(self, side: str) -> np.ndarray:
        if side not in ["left", "right", "up", "down"]:
            raise ValueError(
                f"Side must be 'left', 'right', 'up', or 'down', got '{side}'"
            )
        all_points = self.contact_points.reshape(-1, 2)
        if len(all_points) == 0:
            return np.array([]).reshape(0, 2)
        tolerance = 0.004
        if side in ["left", "right"]:
            return self._select_by_x_coordinate(all_points, side, tolerance)
        return self._select_by_y_coordinate(all_points, side, tolerance)

    def _select_by_x_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        sorted_points = all_points[np.argsort(all_points[:, 1])]
        selected_points = []
        current_y_group = []
        current_y_ref = None
        for point in sorted_points:
            _, y = point
            if current_y_ref is None or abs(y - current_y_ref) <= tolerance:
                current_y_group.append(point)
                current_y_ref = y if current_y_ref is None else current_y_ref
            else:
                selected_points.append(
                    self._select_point_from_group_by_x(current_y_group, side)
                )
                current_y_group = [point]
                current_y_ref = y
        if current_y_group:
            selected_points.append(
                self._select_point_from_group_by_x(current_y_group, side)
            )
        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_by_y_coordinate(
        self, all_points: np.ndarray, side: str, tolerance: float
    ) -> np.ndarray:
        sorted_points = all_points[np.argsort(all_points[:, 0])]
        selected_points = []
        current_x_group = []
        current_x_ref = None
        for point in sorted_points:
            x, _ = point
            if current_x_ref is None or abs(x - current_x_ref) <= tolerance:
                current_x_group.append(point)
                current_x_ref = x if current_x_ref is None else current_x_ref
            else:
                selected_points.append(
                    self._select_point_from_group_by_y(current_x_group, side)
                )
                current_x_group = [point]
                current_x_ref = x
        if current_x_group:
            selected_points.append(
                self._select_point_from_group_by_y(current_x_group, side)
            )
        return (
            np.array(selected_points) if selected_points else np.array([]).reshape(0, 2)
        )

    def _select_point_from_group_by_x(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        points_array = np.array(points)
        return points_array[
            np.argmin(points_array[:, 0])
            if side == "left"
            else np.argmax(points_array[:, 0])
        ]

    def _select_point_from_group_by_y(
        self, points: List[np.ndarray], side: str
    ) -> np.ndarray:
        points_array = np.array(points)
        return points_array[
            np.argmax(points_array[:, 1])
            if side == "up"
            else np.argmin(points_array[:, 1])
        ]

    def get_linear_push_waypoints(
        self, push_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        push_distance = 0.12
        approach_distance = 0.06
        direction_cfg = {
            "horizontal_right": ("left", np.array([1.0, 0.0]), "left"),
            "horizontal_left": ("right", np.array([-1.0, 0.0]), "right"),
            "vertical_up": ("down", np.array([0.0, 1.0]), "both"),
            "vertical_down": ("up", np.array([0.0, -1.0]), "both"),
        }
        if push_direction not in direction_cfg:
            raise ValueError(f"Unknown push direction: {push_direction}")
        contact_side, push_dir, arm_mode = direction_cfg[push_direction]
        contacts = self.select_contact_point(contact_side)
        if len(contacts) == 0:
            contacts = self.contact_points
        if arm_mode == "both":
            if len(contacts) >= 2:
                chosen = contacts[np.random.choice(len(contacts), 2, replace=False)]
                left_contact = chosen[chosen[:, 0].argmin()]
                right_contact = chosen[chosen[:, 0].argmax()]
            else:
                left_contact = contacts[0] + np.array([-0.02, 0.0])
                right_contact = contacts[0] + np.array([0.02, 0.0])
            return {
                "left": [
                    left_contact - push_dir * approach_distance,
                    left_contact,
                    left_contact + push_dir * push_distance,
                ],
                "right": [
                    right_contact - push_dir * approach_distance,
                    right_contact,
                    right_contact + push_dir * push_distance,
                ],
            }
        contact = contacts[np.random.choice(len(contacts))]
        waypoints = [
            contact - push_dir * approach_distance,
            contact,
            contact + push_dir * push_distance,
        ]
        return {
            "left": waypoints if arm_mode == "left" else [],
            "right": waypoints if arm_mode == "right" else [],
        }

    def get_rotation_waypoints(
        self, rotation_direction: str
    ) -> Dict[str, List[np.ndarray]]:
        if rotation_direction not in ["clockwise", "counterclockwise"]:
            raise ValueError(f"Unknown rotation direction: {rotation_direction}")
        points = self.contact_points
        center = self.object_info.center
        left_grip = points[np.argmin(points[:, 0])]
        right_grip = points[np.argmax(points[:, 0])]
        angle = -np.pi / 6 if rotation_direction == "clockwise" else np.pi / 6
        cos_r, sin_r = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        left_rotated = center + rot @ (left_grip - center)
        right_rotated = center + rot @ (right_grip - center)
        left_approach = (
            left_grip
            + (left_grip - center) / (np.linalg.norm(left_grip - center) + 1e-8) * 0.05
        )
        right_approach = (
            right_grip
            + (right_grip - center)
            / (np.linalg.norm(right_grip - center) + 1e-8)
            * 0.05
        )
        return {
            "left": [left_approach, left_grip, left_rotated],
            "right": [right_approach, right_grip, right_rotated],
        }


class CollisionChecker:
    """Checks for collisions between arms and with a pushed planar object."""

    def __init__(
        self, t_analyzer: TGeometryAnalyzer, constraints: WorkspaceConstraints
    ):
        self.t_analyzer = t_analyzer
        self.constraints = constraints

    def check_arm_collision(self, left_pos: np.ndarray, right_pos: np.ndarray) -> bool:
        """Check if arms are too close to each other."""
        distance = np.linalg.norm(left_pos - right_pos)
        return distance < self.constraints.min_arm_distance

    def check_workspace_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        x, y = pos
        return (
            self.constraints.x_min <= x <= self.constraints.x_max
            and self.constraints.y_min <= y <= self.constraints.y_max
        )

    def check_t_collision(
        self, start_pos: np.ndarray, end_pos: np.ndarray, allow_contact: bool = False
    ) -> bool:
        """Check collision with T-shape for a line segment from start_pos to end_pos."""
        min_dist = 0.0 if allow_contact else self.constraints.min_t_distance

        # Get all contact points (edges of T-shape)
        center = self.t_analyzer.t_info.center

        # Sample points along the line segment
        num_samples = 20  # Number of points to check along the line
        t_values = np.linspace(0, 1, num_samples)
        line_points = start_pos[None, :] + t_values[:, None] * (
            end_pos[None, :] - start_pos[None, :]
        )

        # Check distance from each point on the line to all T-shape points
        for line_point in line_points:
            distances = np.linalg.norm(center - line_point)
            min_distance_to_t = distances
            if min_distance_to_t < min_dist:
                return True

        return False

    def is_contact_point_feasible(
        self, contact_point: np.ndarray, arm_positions: List[np.ndarray]
    ) -> bool:
        """Check if a contact point is reachable without collisions."""
        # Check workspace bounds
        if not self.check_workspace_bounds(contact_point):
            return False

        # Check if other arm positions would cause collision
        for other_pos in arm_positions:
            if self.check_arm_collision(contact_point, other_pos):
                return False

        return True


def actions_in_range(actions: np.ndarray) -> bool:
    """Check if actions are within range."""
    N = actions.shape[0]
    l_act = actions[:, :2]
    r_act = actions[:, 2:]
    world_bases = np.array(
        [
            [
                [1.0, 0.0, 0.0, -0.469],
                [0.0, 1.0, 0.0, -0.019],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [-1.0, 0.0, 0.0, 0.469],
                [0.0, -1.0, 0.0, -0.019],
                [0.0, 0.0, 1.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    l_poses = np.tile(np.eye(4), (N, 1, 1))
    r_poses = np.tile(np.eye(4), (N, 1, 1))
    l_poses[:, :2, 3] = l_act
    r_poses[:, :2, 3] = r_act
    robot_t_l_act = np.tile(np.linalg.inv(world_bases[0]), (N, 1, 1)) @ l_poses
    robot_t_r_act = np.tile(np.linalg.inv(world_bases[1]), (N, 1, 1)) @ r_poses
    x_max = 0.6 + 0.02
    x_min = 0.25 - 0.02
    y_max = 0.2 + 0.02
    y_min = -0.2 - 0.02
    in_range = (
        (robot_t_l_act[:, 0, 3].max() <= x_max)
        and (robot_t_l_act[:, 0, 3].min() >= x_min)
        and (robot_t_r_act[:, 0, 3].max() <= x_max)
        and (robot_t_r_act[:, 0, 3].min() >= x_min)
        and (robot_t_l_act[:, 1, 3].max() <= y_max)
        and (robot_t_l_act[:, 1, 3].min() >= y_min)
        and (robot_t_r_act[:, 1, 3].max() <= y_max)
        and (robot_t_r_act[:, 1, 3].min() >= y_min)
    )
    return in_range


def evaluate_vertical_up_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if vertical upward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if y position increased
    y_movement = final_pos[1] - initial_pos[1]
    return is_flat and y_movement > 0.01 and actions_in_range(actions)


def evaluate_vertical_down_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if vertical downward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if y position decreased
    y_movement = initial_pos[1] - final_pos[1]
    return is_flat and y_movement > 0.01 and actions_in_range(actions)


def evaluate_horizontal_right_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if horizontal rightward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if x position increased
    x_movement = final_pos[0] - initial_pos[0]
    return is_flat and x_movement > 0.01 and actions_in_range(actions)


def evaluate_horizontal_left_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if horizontal leftward push was successful."""
    initial_rot = initial_pose[:3, :3]
    initial_pos = initial_pose[:3, 3]
    final_pos = final_pose[:3, 3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if x position decreased
    x_movement = initial_pos[0] - final_pos[0]
    return is_flat and x_movement > 0.01 and actions_in_range(actions)


def evaluate_rotation_cw_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if rotation was successful."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if there was significant rotation
    init_t_final_pose = np.linalg.inv(initial_pose) @ final_pose
    delta_angle = np.arctan2(init_t_final_pose[1, 0], init_t_final_pose[0, 0])
    return is_flat and delta_angle < -0.1 and actions_in_range(actions)


def evaluate_rotation_ccw_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if rotation was successful."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat

    # Check if there was significant rotation
    init_t_final_pose = np.linalg.inv(initial_pose) @ final_pose
    delta_angle = np.arctan2(init_t_final_pose[1, 0], init_t_final_pose[0, 0])
    return is_flat and delta_angle > 0.1 and actions_in_range(actions)


def evaluate_random_motion_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if random motion was successful (T is still flat and not moved)."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is not moved
    final_pos = final_pose[:3, 3]
    initial_pos = initial_pose[:3, 3]

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat
    rotation_matrix = final_rot @ initial_rot.T
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    is_moved = (angle > 0.1) or (np.linalg.norm(final_pos - initial_pos) > 0.01)
    return is_flat and not is_moved and actions_in_range(actions)


def evaluate_random_contact_success(
    initial_pose: np.ndarray, final_pose: np.ndarray, actions: np.ndarray
) -> bool:
    """Check if random motion was successful (T is still flat and moved)."""
    initial_rot = initial_pose[:3, :3]
    final_rot = final_pose[:3, :3]

    # Check if T is lying flat on the table
    table_normal = np.array([0, 0, 1])
    final_z_axis = final_rot[:, 2]
    final_alignment = np.dot(final_z_axis, table_normal)
    is_final_flat = final_alignment > 0.95

    # Check if T is not moved
    final_pos = final_pose[:3, 3]
    initial_pos = initial_pose[:3, 3]
    rotation_matrix = final_rot @ initial_rot.T

    # Check if T is initially lying flat on the table
    initial_z_axis = initial_rot[:, 2]
    initial_alignment = np.dot(initial_z_axis, table_normal)
    is_initial_flat = initial_alignment > 0.95

    is_flat = is_final_flat and is_initial_flat
    trace = np.trace(rotation_matrix)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    is_moved = (angle > 0.1) or (np.linalg.norm(final_pos - initial_pos) > 0.01)
    return is_flat and is_moved and actions_in_range(actions)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class RLCoverageTransition:
    """Bookkeeping for one high-level RL coverage action."""

    state_key: tuple[int, int, int]
    action_key: str
    start_center: np.ndarray
    start_rotation: float


class MotionPlanner:
    """Plans scripted planar pushing motions, including diverse mixed rollouts."""

    def __init__(self, workspace_constraints: WorkspaceConstraints = None):
        self.constraints = workspace_constraints or WorkspaceConstraints()
        self.config = TrajectoryConfig(
            noise_std=0.001,
        )
        self.eef_fingertip_offset = 0.08
        self.coordinator = BimanualCoordination(self.config)

        # Motion type probabilities
        self.motion_probabilities = {
            "linear": 0.3,
            "rotating": 0.3,
            "random_contact": 0.3,
            "random_no_contact": 0.1,
        }
        # Mixed rollouts should be interaction-heavy but still include a little
        # free-space motion so the world model sees both contact and non-contact
        # actions in the same episode.
        self.mixed_motion_probabilities = {
            "linear": 0.35,
            "rotating": 0.35,
            "random_contact": 0.25,
            "random_no_contact": 0.05,
        }
        self._last_mixed_motion_type: Optional[str] = None

        # Online high-level RL for coverage collection.  The learner chooses
        # between safe scripted translation/rotation/contact primitives and
        # updates a tabular contextual bandit from observed object-pose change.
        self.rl_actions = [
            "push_right",
            "push_left",
            "push_up",
            "push_down",
            "rotate_cw",
            "rotate_ccw",
            "tangent_cw",
            "tangent_ccw",
        ]
        self.rl_q_values: Dict[tuple[tuple[int, int, int], str], float] = {}
        self.rl_action_counts: Dict[tuple[tuple[int, int, int], str], int] = {}
        self.rl_pose_counts: Dict[tuple[int, int, int], int] = {}
        self.rl_theta_counts: Dict[int, int] = {}
        self.rl_total_updates = 0
        self.rl_epsilon = 0.20
        self.rl_ucb_bonus = 0.35
        self.rl_alpha = 0.25
        self._last_rl_transition: Optional[RLCoverageTransition] = None
        self.loaded_rl_checkpoint_extra_state: dict = {}
        self.reset_rl_episode()

    def plan_episode(
        self,
        t_info: TShapeInfo,
        current_arm_positions: np.ndarray,
        motion_type: str,
    ) -> tuple[np.ndarray, bool, callable, int]:
        """Plan a complete episode trajectory.

        Args:
            t_info: Information about T-shape
            current_arm_positions: Current positions (left_x, left_y, right_x, right_y)
            duration: Episode duration in seconds
            num_steps: Number of trajectory steps

        Returns:
            trajectory: Nx4 array of bimanual trajectory
        """
        # Initialize analyzers. Keep the original T-specific path for legacy
        # mesh PushT data, and use the procedural object analyzer for generic
        # block-letter shapes.
        if isinstance(t_info, PlanarObjectInfo):
            t_analyzer = PlanarObjectGeometryAnalyzer(t_info)
        else:
            t_analyzer = TGeometryAnalyzer(t_info)
        collision_checker = CollisionChecker(t_analyzer, self.constraints)

        # Generate trajectory based on motion type
        if motion_type == "linear":
            duration = 15.0
            num_steps = 150
            traj, success, success_fn = self._plan_linear_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "rotating":
            duration = 30.0
            num_steps = 300
            traj, success, success_fn = self._plan_rotating_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "random_contact":
            duration = 30.0
            num_steps = 300
            traj, success, success_fn = self._plan_random_contact_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "random_no_contact":
            duration = 15.0
            num_steps = 150
            traj, success, success_fn = self._plan_random_no_contact_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
                duration,
                num_steps,
            )
        elif motion_type == "mixed":
            traj, success, success_fn = self._plan_mixed_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
            )
            num_steps = len(traj)
        elif motion_type == "rl_coverage":
            traj, success, success_fn = self._plan_rl_coverage_motion(
                t_analyzer,
                collision_checker,
                current_arm_positions,
            )
            num_steps = len(traj)
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")
        return traj, success, success_fn, num_steps

    def _select_motion_type(self) -> str:
        """Select motion type based on probabilities."""
        return self._sample_weighted(self.motion_probabilities, fallback="linear")

    def _sample_weighted(self, weights: Dict[str, float], fallback: str) -> str:
        """Sample a key from unnormalized positive weights."""
        positive_items = [
            (key, float(weight)) for key, weight in weights.items() if weight > 0.0
        ]
        if not positive_items:
            return fallback
        total = sum(weight for _, weight in positive_items)
        threshold = random.random() * total
        cumulative = 0.0
        for key, weight in positive_items:
            cumulative += weight
            if threshold <= cumulative:
                return key
        return positive_items[-1][0]

    def _safe_linear_direction_weights(self, center: np.ndarray) -> Dict[str, float]:
        """Prefer inward/tangential pushes when the object is near a table edge."""
        x, y = center[:2]
        edge_margin = 0.06
        weights = {
            "horizontal_right": 1.0,
            "horizontal_left": 1.0,
            "vertical_up": 1.0,
            "vertical_down": 1.0,
        }

        if x > self.constraints.x_max - edge_margin:
            weights["horizontal_right"] = 0.0
            weights["horizontal_left"] += 2.0
        if x < self.constraints.x_min + edge_margin:
            weights["horizontal_left"] = 0.0
            weights["horizontal_right"] += 2.0
        if y > self.constraints.y_max - edge_margin:
            weights["vertical_up"] = 0.0
            weights["vertical_down"] += 2.0
        if y < self.constraints.y_min + edge_margin:
            weights["vertical_down"] = 0.0
            weights["vertical_up"] += 2.0

        if all(weight <= 0.0 for weight in weights.values()):
            return {
                "horizontal_right": 1.0,
                "horizontal_left": 1.0,
                "vertical_up": 1.0,
                "vertical_down": 1.0,
            }
        return weights

    def _linear_direction_to_vector(self, direction: str) -> np.ndarray:
        direction_vectors = {
            "horizontal_right": np.array([1.0, 0.0]),
            "horizontal_left": np.array([-1.0, 0.0]),
            "vertical_up": np.array([0.0, 1.0]),
            "vertical_down": np.array([0.0, -1.0]),
        }
        return direction_vectors[direction]

    def save_rl_checkpoint(
        self,
        checkpoint_path: str | Path,
        extra_state: Optional[dict] = None,
    ) -> None:
        """Save persistent online RL coverage state to a JSON checkpoint."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "collector_state": extra_state or {},
            "rl_actions": self.rl_actions,
            "rl_total_updates": int(self.rl_total_updates),
            "rl_q_values": [
                {
                    "state": list(state_key),
                    "action": action_key,
                    "value": float(value),
                }
                for (state_key, action_key), value in sorted(self.rl_q_values.items())
            ],
            "rl_action_counts": [
                {
                    "state": list(state_key),
                    "action": action_key,
                    "count": int(count),
                }
                for (state_key, action_key), count in sorted(
                    self.rl_action_counts.items()
                )
            ],
            "rl_pose_counts": [
                {"state": list(state_key), "count": int(count)}
                for state_key, count in sorted(self.rl_pose_counts.items())
            ],
            "rl_theta_counts": [
                {"theta_bin": int(theta_bin), "count": int(count)}
                for theta_bin, count in sorted(self.rl_theta_counts.items())
            ],
        }
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(path)

    def load_rl_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Load persistent online RL coverage state from a JSON checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            return False

        payload = json.loads(path.read_text())
        extra_state = payload.get("collector_state", {})
        self.loaded_rl_checkpoint_extra_state = (
            extra_state if isinstance(extra_state, dict) else {}
        )
        valid_actions = set(self.rl_actions)
        self.rl_q_values = {}
        for entry in payload.get("rl_q_values", []):
            action_key = str(entry["action"])
            if action_key not in valid_actions:
                continue
            state_key = tuple(int(value) for value in entry["state"])
            self.rl_q_values[(state_key, action_key)] = float(entry["value"])

        self.rl_action_counts = {}
        for entry in payload.get("rl_action_counts", []):
            action_key = str(entry["action"])
            if action_key not in valid_actions:
                continue
            state_key = tuple(int(value) for value in entry["state"])
            self.rl_action_counts[(state_key, action_key)] = int(entry["count"])

        self.rl_pose_counts = {}
        for entry in payload.get("rl_pose_counts", []):
            state_key = tuple(int(value) for value in entry["state"])
            self.rl_pose_counts[state_key] = int(entry["count"])

        self.rl_theta_counts = {
            int(entry["theta_bin"]): int(entry["count"])
            for entry in payload.get("rl_theta_counts", [])
        }
        self.rl_total_updates = int(payload.get("rl_total_updates", 0))
        self.reset_rl_episode()
        return True

    def reset_rl_episode(self) -> None:
        """Reset per-episode RL bookkeeping while keeping learned coverage stats."""
        self._last_rl_transition = None
        self._rl_episode_translation = 0.0
        self._rl_episode_rotation = 0.0
        self._rl_episode_segments = 0

    def end_rl_episode(
        self,
        t_info: Optional[TShapeInfo] = None,
        aborted: bool = False,
    ) -> None:
        """Finish the pending RL transition and clear per-episode state."""
        if t_info is not None:
            self.observe_rl_segment_end(t_info, aborted=aborted)
        elif aborted and self._last_rl_transition is not None:
            self._update_rl_value(self._last_rl_transition, reward=-6.0)
            self._last_rl_transition = None
        self.reset_rl_episode()

    def _rl_pose_key(self, center: np.ndarray, rotation: float) -> tuple[int, int, int]:
        x_bins = 10
        y_bins = 8
        theta_bins = 16
        x_alpha = (center[0] - self.constraints.x_min) / max(
            self.constraints.x_max - self.constraints.x_min, 1e-8
        )
        y_alpha = (center[1] - self.constraints.y_min) / max(
            self.constraints.y_max - self.constraints.y_min, 1e-8
        )
        theta_alpha = (wrap_angle(rotation) + np.pi) / (2.0 * np.pi)
        x_bin = int(np.clip(np.floor(x_alpha * x_bins), 0, x_bins - 1))
        y_bin = int(np.clip(np.floor(y_alpha * y_bins), 0, y_bins - 1))
        theta_bin = int(np.clip(np.floor(theta_alpha * theta_bins), 0, theta_bins - 1))
        return x_bin, y_bin, theta_bin

    def _action_to_direction(self, action_key: str) -> Optional[str]:
        return {
            "push_right": "horizontal_right",
            "push_left": "horizontal_left",
            "push_up": "vertical_up",
            "push_down": "vertical_down",
        }.get(action_key)

    def _is_translation_action(self, action_key: str) -> bool:
        return action_key.startswith("push_")

    def _is_rotation_action(self, action_key: str) -> bool:
        return action_key.startswith("rotate_") or action_key.startswith("tangent_")

    def _safe_rl_actions(self, center: np.ndarray) -> list[str]:
        """Filter high-level actions that would push outward near workspace edges."""
        direction_weights = self._safe_linear_direction_weights(center)
        actions = []
        for action in self.rl_actions:
            direction = self._action_to_direction(action)
            if direction is None or direction_weights[direction] > 0.0:
                actions.append(action)
        return actions or ["rotate_cw", "rotate_ccw", "tangent_cw", "tangent_ccw"]

    def _predicted_coverage_bonus(
        self, center: np.ndarray, rotation: float, action_key: str
    ) -> float:
        predicted_center = center.copy()
        predicted_rotation = rotation
        if action_key == "push_right":
            predicted_center += np.array([0.055, 0.0])
        elif action_key == "push_left":
            predicted_center += np.array([-0.055, 0.0])
        elif action_key == "push_up":
            predicted_center += np.array([0.0, 0.055])
        elif action_key == "push_down":
            predicted_center += np.array([0.0, -0.055])
        elif action_key in ["rotate_cw", "tangent_cw"]:
            predicted_rotation = wrap_angle(rotation - 0.35)
        elif action_key in ["rotate_ccw", "tangent_ccw"]:
            predicted_rotation = wrap_angle(rotation + 0.35)

        state_key = self._rl_pose_key(predicted_center, predicted_rotation)
        pose_count = self.rl_pose_counts.get(state_key, 0)
        theta_count = self.rl_theta_counts.get(state_key[2], 0)
        return 0.50 / np.sqrt(pose_count + 1.0) + 0.25 / np.sqrt(theta_count + 1.0)

    def _select_rl_action(self, t_analyzer: TGeometryAnalyzer) -> str:
        center = t_analyzer.t_info.center
        rotation = t_analyzer.t_info.rotation
        state_key = self._rl_pose_key(center, rotation)
        available_actions = self._safe_rl_actions(center)

        # Hard curriculum: force both translation and rotation attempts early in
        # every <=600-frame rollout so saved RL episodes contain both effects.
        if self._rl_episode_translation < 0.025 and self._rl_episode_segments <= 2:
            translation_actions = [
                action
                for action in available_actions
                if self._is_translation_action(action)
            ]
            if translation_actions:
                return self._sample_weighted(
                    {
                        action: self._safe_linear_direction_weights(center)[
                            self._action_to_direction(action)
                        ]
                        for action in translation_actions
                    },
                    fallback=translation_actions[0],
                )
        if self._rl_episode_rotation < 0.12 and self._rl_episode_segments <= 4:
            rotation_actions = [
                action
                for action in available_actions
                if self._is_rotation_action(action)
            ]
            if rotation_actions:
                return random.choice(rotation_actions)

        if random.random() < self.rl_epsilon:
            return random.choice(available_actions)

        log_total = np.log(self.rl_total_updates + 2.0)
        best_action = available_actions[0]
        best_score = -np.inf
        for action in available_actions:
            key = (state_key, action)
            count = self.rl_action_counts.get(key, 0)
            q_value = self.rl_q_values.get(key, 0.0)
            exploration = self.rl_ucb_bonus * np.sqrt(log_total / (count + 1.0))
            balance_bonus = 0.0
            if (
                self._is_translation_action(action)
                and self._rl_episode_translation < 0.04
            ):
                balance_bonus += 0.8
            if self._is_rotation_action(action) and self._rl_episode_rotation < 0.18:
                balance_bonus += 0.8
            score = (
                q_value
                + exploration
                + balance_bonus
                + self._predicted_coverage_bonus(center, rotation, action)
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _update_rl_value(self, transition: RLCoverageTransition, reward: float) -> None:
        key = (transition.state_key, transition.action_key)
        old_value = self.rl_q_values.get(key, 0.0)
        count = self.rl_action_counts.get(key, 0)
        step_size = max(self.rl_alpha, 1.0 / float(count + 1))
        self.rl_q_values[key] = old_value + step_size * (reward - old_value)
        self.rl_action_counts[key] = count + 1
        self.rl_total_updates += 1

    def observe_rl_segment_end(
        self,
        t_info: TShapeInfo,
        aborted: bool = False,
    ) -> None:
        """Update the RL table from the previous segment outcome."""
        if self._last_rl_transition is None:
            state_key = self._rl_pose_key(t_info.center, t_info.rotation)
            self.rl_pose_counts[state_key] = self.rl_pose_counts.get(state_key, 0) + 1
            self.rl_theta_counts[state_key[2]] = (
                self.rl_theta_counts.get(state_key[2], 0) + 1
            )
            return

        transition = self._last_rl_transition
        translation = float(np.linalg.norm(t_info.center - transition.start_center))
        rotation = abs(wrap_angle(t_info.rotation - transition.start_rotation))
        state_key = self._rl_pose_key(t_info.center, t_info.rotation)
        previous_pose_count = self.rl_pose_counts.get(state_key, 0)
        previous_theta_count = self.rl_theta_counts.get(state_key[2], 0)

        self._rl_episode_translation += translation
        self._rl_episode_rotation += rotation
        self.rl_pose_counts[state_key] = previous_pose_count + 1
        self.rl_theta_counts[state_key[2]] = previous_theta_count + 1

        reward = 0.0
        reward += 1.50 / np.sqrt(previous_pose_count + 1.0)
        reward += 0.75 / np.sqrt(previous_theta_count + 1.0)
        reward += 1.00 * min(translation / 0.08, 1.0)
        reward += 1.25 * min(rotation / 0.45, 1.0)
        if translation < 0.006 and rotation < 0.04:
            reward -= 0.5

        x, y = t_info.center[:2]
        edge_distance = min(
            x - self.constraints.x_min,
            self.constraints.x_max - x,
            y - self.constraints.y_min,
            self.constraints.y_max - y,
        )
        if edge_distance < 0.0:
            reward -= 5.0
        elif edge_distance < 0.025:
            reward -= 2.0 * (0.025 - edge_distance) / 0.025
        if aborted:
            reward -= 6.0

        self._update_rl_value(transition, reward)
        self._last_rl_transition = None

    def _plan_rl_coverage_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan one online-RL coverage segment.

        This is a contextual bandit over safe scripted primitives, not raw-joint
        RL.  That keeps collection stable while still adapting toward actions
        that create new object position/yaw coverage with both translation and
        rotation inside short episodes.
        """
        self.observe_rl_segment_end(t_analyzer.t_info)
        state_key = self._rl_pose_key(
            t_analyzer.t_info.center, t_analyzer.t_info.rotation
        )
        max_attempts = 16

        for _ in range(max_attempts):
            action = self._select_rl_action(t_analyzer)
            direction = self._action_to_direction(action)

            if direction is not None:
                num_steps = random.randint(100, 160)
                traj, success, success_fn = self._plan_linear_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    direction=direction,
                )
            elif action == "rotate_cw":
                num_steps = random.randint(120, 190)
                traj, success, success_fn = self._plan_rotating_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    direction="clockwise",
                )
            elif action == "rotate_ccw":
                num_steps = random.randint(120, 190)
                traj, success, success_fn = self._plan_rotating_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    direction="counterclockwise",
                )
            else:
                num_steps = random.randint(100, 170)
                contact_style = (
                    "tangential_cw" if action == "tangent_cw" else "tangential_ccw"
                )
                traj, success, success_fn = self._plan_random_contact_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    contact_style=contact_style,
                )

            if success and actions_in_range(traj):
                self._last_rl_transition = RLCoverageTransition(
                    state_key=state_key,
                    action_key=action,
                    start_center=t_analyzer.t_info.center.copy(),
                    start_rotation=float(t_analyzer.t_info.rotation),
                )
                self._rl_episode_segments += 1
                return traj, True, success_fn

            self._update_rl_value(
                RLCoverageTransition(
                    state_key=state_key,
                    action_key=action,
                    start_center=t_analyzer.t_info.center.copy(),
                    start_rotation=float(t_analyzer.t_info.rotation),
                ),
                reward=-1.0,
            )

        traj, success, success_fn = self._plan_random_contact_motion(
            t_analyzer,
            collision_checker,
            current_pos,
            duration=12.0,
            num_steps=120,
            contact_style="tangential_ccw",
        )
        if success:
            self._last_rl_transition = RLCoverageTransition(
                state_key=state_key,
                action_key="tangent_ccw",
                start_center=t_analyzer.t_info.center.copy(),
                start_rotation=float(t_analyzer.t_info.rotation),
            )
            self._rl_episode_segments += 1
        return traj, success, success_fn

    def _sample_mixed_motion_type(self, center: np.ndarray) -> str:
        """Sample a diverse mixed sub-motion with light anti-repetition and edge bias."""
        weights = dict(self.mixed_motion_probabilities)

        # Do not let one primitive dominate a long episode.
        if self._last_mixed_motion_type in weights:
            weights[self._last_mixed_motion_type] *= 0.35

        # Near edges, rotations and small contacts are safer than large linear
        # outward pushes, but keep some linear probability for inward pushes.
        x, y = center[:2]
        edge_margin = 0.06
        near_edge = (
            x > self.constraints.x_max - edge_margin
            or x < self.constraints.x_min + edge_margin
            or y > self.constraints.y_max - edge_margin
            or y < self.constraints.y_min + edge_margin
        )
        if near_edge:
            weights["linear"] *= 0.7
            weights["rotating"] *= 1.3
            weights["random_contact"] *= 1.2
            weights["random_no_contact"] *= 1.5

        return self._sample_weighted(weights, fallback="linear")

    def _plan_mixed_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan one diverse sub-segment for a mixed rollout.

        Mixed episodes are generated by repeatedly calling this method as the
        object moves.  Each call re-reads the current object pose, samples a new
        primitive, and uses boundary-aware direction choices so long episodes
        contain translation, rotation, contact, and occasional non-contact data.
        """
        center = t_analyzer.t_info.center
        max_attempts = 16

        for _ in range(max_attempts):
            sub_motion = self._sample_mixed_motion_type(center)

            if sub_motion == "linear":
                direction_weights = self._safe_linear_direction_weights(center)
                direction = self._sample_weighted(
                    direction_weights, fallback="horizontal_right"
                )
                num_steps = random.randint(120, 180)
                traj, success, success_fn = self._plan_linear_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    direction=direction,
                )
            elif sub_motion == "rotating":
                num_steps = random.randint(160, 260)
                traj, success, success_fn = self._plan_rotating_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    direction=random.choice(["clockwise", "counterclockwise"]),
                )
            elif sub_motion == "random_contact":
                direction_name = self._sample_weighted(
                    self._safe_linear_direction_weights(center),
                    fallback="horizontal_right",
                )
                contact_style = self._sample_weighted(
                    {"cardinal": 0.45, "tangential": 0.45, "corner": 0.10},
                    fallback="cardinal",
                )
                num_steps = random.randint(110, 220)
                traj, success, success_fn = self._plan_random_contact_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                    preferred_push_direction=self._linear_direction_to_vector(
                        direction_name
                    ),
                    contact_style=contact_style,
                )
            else:
                num_steps = random.randint(80, 150)
                traj, success, success_fn = self._plan_random_no_contact_motion(
                    t_analyzer,
                    collision_checker,
                    current_pos,
                    duration=num_steps / 10.0,
                    num_steps=num_steps,
                )

            if success and actions_in_range(traj):
                self._last_mixed_motion_type = sub_motion
                return traj, True, success_fn

        # Conservative fallback: a small no-contact move avoids hanging the
        # collector if all sampled interaction primitives are out of range.
        traj, success, success_fn = self._plan_random_no_contact_motion(
            t_analyzer,
            collision_checker,
            current_pos,
            duration=10.0,
            num_steps=100,
        )
        if success:
            self._last_mixed_motion_type = "random_no_contact"
        return traj, success, success_fn

    def _plan_linear_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
        direction: Optional[str] = None,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan linear pushing motion."""
        directions = [
            "horizontal_right",
            "horizontal_left",
            "vertical_up",
            "vertical_down",
        ]
        if direction is None:
            direction = random.choice(directions)

        waypoints = t_analyzer.get_linear_push_waypoints(direction)
        if direction == "horizontal_right":
            rand_noise = np.random.rand(3, 2) * 0.02
            for noise in rand_noise:
                waypoints["right"].append(current_pos[2:] + noise)
        elif direction == "horizontal_left":
            rand_noise = np.random.rand(3, 2) * 0.02
            for noise in rand_noise:
                waypoints["left"].append(current_pos[:2] + noise)

        for i, p in enumerate(waypoints["left"]):
            waypoints["left"][i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(waypoints["right"]):
            waypoints["right"][i][0] = p[0] + self.eef_fingertip_offset

        waypoints["left"] = [current_pos[:2]] + waypoints["left"]
        waypoints["right"] = [current_pos[2:]] + waypoints["right"]

        # Create curve primitives from waypoints
        left_primitive = CurvePrimitive(waypoints["left"], self.config)
        right_primitive = CurvePrimitive(waypoints["right"], self.config)

        if direction == "vertical_up":
            success_fn = evaluate_vertical_up_success
        elif direction == "vertical_down":
            success_fn = evaluate_vertical_down_success
        elif direction == "horizontal_right":
            success_fn = evaluate_horizontal_right_success
        elif direction == "horizontal_left":
            success_fn = evaluate_horizontal_left_success

        # Generate coordinated trajectory
        return (
            self.coordinator.coordinate(
                left_primitive, right_primitive, duration, num_steps, "simultaneous"
            ),
            True,
            success_fn,
        )

    def _plan_rotating_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
        direction: Optional[str] = None,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan rotating motion."""
        if direction is None:
            direction = random.choice(["clockwise", "counterclockwise"])

        waypoints = t_analyzer.get_rotation_waypoints(direction)
        for i, p in enumerate(waypoints["left"]):
            waypoints["left"][i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(waypoints["right"]):
            waypoints["right"][i][0] = p[0] + self.eef_fingertip_offset

        waypoints["left"] = [current_pos[:2]] + waypoints["left"]
        waypoints["right"] = [current_pos[2:]] + waypoints["right"]

        # Create curve primitives from waypoints
        left_primitive = CurvePrimitive(waypoints["left"], self.config)
        right_primitive = CurvePrimitive(waypoints["right"], self.config)

        if direction == "clockwise":
            success_fn = evaluate_rotation_cw_success
        elif direction == "counterclockwise":
            success_fn = evaluate_rotation_ccw_success
        else:
            raise ValueError(f"Unknown rotation direction: {direction}")

        # Generate coordinated trajectory
        return (
            self.coordinator.coordinate(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                "simultaneous",
                speed_profile="constant",
            ),
            True,
            success_fn,
        )

    def _plan_random_contact_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
        preferred_push_direction: Optional[np.ndarray] = None,
        contact_style: Optional[str] = None,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan random contact motion."""
        if isinstance(t_analyzer, PlanarObjectGeometryAnalyzer):
            return self._plan_random_contact_motion_planar_object(
                t_analyzer,
                current_pos,
                duration,
                num_steps,
                preferred_push_direction=preferred_push_direction,
                contact_style=contact_style,
            )

        # Randomly select contact points
        left_contact = random.choice(t_analyzer.contact_points)
        right_contact = random.choice(t_analyzer.contact_points)

        # Generate approach and contact waypoints
        approach_distance = 0.03
        left_approach = left_contact - approach_distance * np.random.rand(2)
        right_approach = right_contact - approach_distance * np.random.rand(2)

        # Create small push motions from contact points
        push_distance = 0.03
        left_push = left_contact + push_distance * np.random.rand(2)
        right_push = right_contact + push_distance * np.random.rand(2)

        left_waypoints = [
            left_approach,
            left_contact,
            left_push,
        ]
        right_waypoints = [
            right_approach,
            right_contact,
            right_push,
        ]

        for i, p in enumerate(left_waypoints):
            left_waypoints[i][0] = p[0] - self.eef_fingertip_offset
        for i, p in enumerate(right_waypoints):
            right_waypoints[i][0] = p[0] + self.eef_fingertip_offset

        left_waypoints = [current_pos[:2]] + left_waypoints
        right_waypoints = [current_pos[2:]] + right_waypoints

        left_primitive = CurvePrimitive(left_waypoints, self.config)
        right_primitive = CurvePrimitive(right_waypoints, self.config)

        success_fn = evaluate_random_contact_success

        return (
            self.coordinator.coordinate(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                "overlap",
                speed_profile="constant",
            ),
            True,
            success_fn,
        )

    def _plan_random_contact_motion_planar_object(
        self,
        t_analyzer: PlanarObjectGeometryAnalyzer,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
        preferred_push_direction: Optional[np.ndarray] = None,
        contact_style: Optional[str] = None,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan a one-arm exterior push for generic letters.

        Two-arm random contact often traps letters between claws or hooks holes
        in shapes like A/H/O.  For procedural letters, use one active pusher and
        diversify the contact geometry: cardinal side pushes for translation,
        tangential pushes for rotation, and occasional corner shoves.
        """
        center = t_analyzer.object_info.center
        contact_style = contact_style or self._sample_weighted(
            {"cardinal": 0.55, "tangential": 0.35, "corner": 0.10},
            fallback="cardinal",
        )

        def normalized(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                return fallback.copy()
            return vec / norm

        approach_distance = random.uniform(0.035, 0.065)
        push_distance = random.uniform(0.035, 0.085)

        cardinal_options = [
            (np.array([1.0, 0.0]), "left"),
            (np.array([-1.0, 0.0]), "right"),
            (np.array([0.0, 1.0]), "down"),
            (np.array([0.0, -1.0]), "up"),
        ]

        if contact_style == "cardinal":
            if preferred_push_direction is not None:
                preferred = normalized(preferred_push_direction, np.array([1.0, 0.0]))
                push_dir, side = max(
                    cardinal_options,
                    key=lambda option: float(np.dot(option[0], preferred)),
                )
            else:
                push_dir, side = random.choice(cardinal_options)
            contacts = t_analyzer.select_contact_point(side)
            if len(contacts) == 0:
                contacts = t_analyzer.contact_points
            contact = contacts[np.random.choice(len(contacts))].copy()
        elif contact_style == "corner":
            points = t_analyzer.contact_points
            radii = np.linalg.norm(points - center[None], axis=1)
            radius_threshold = np.quantile(radii, 0.80)
            candidates = points[radii >= radius_threshold]
            if len(candidates) == 0:
                candidates = points
            contact = candidates[np.random.choice(len(candidates))].copy()
            inward = normalized(center - contact, np.array([1.0, 0.0]))
            tangent = np.array([-inward[1], inward[0]])
            if random.random() < 0.5:
                tangent *= -1.0
            push_dir = normalized(0.75 * inward + 0.25 * tangent, inward)
        else:
            points = t_analyzer.contact_points
            contact = points[np.random.choice(len(points))].copy()
            radial = normalized(contact - center, np.array([1.0, 0.0]))
            if contact_style == "tangential_cw":
                push_dir = np.array([radial[1], -radial[0]])
            elif contact_style == "tangential_ccw":
                push_dir = np.array([-radial[1], radial[0]])
            else:
                push_dir = np.array([-radial[1], radial[0]])
                if random.random() < 0.5:
                    push_dir *= -1.0

        approach = contact - push_dir * approach_distance
        pushed = contact + push_dir * push_distance

        # Choose the active arm based on push direction and contact location.
        # Horizontal pushes should use the arm on the contacting side.  Mostly
        # vertical/tangential pushes choose the arm closer to that contact point.
        if push_dir[0] > 0.25:
            use_left_arm = True
        elif push_dir[0] < -0.25:
            use_left_arm = False
        elif abs(contact[0] - center[0]) > 0.01:
            use_left_arm = contact[0] <= center[0]
        else:
            use_left_arm = random.random() < 0.5

        if use_left_arm:
            active_waypoints = [current_pos[:2], approach, contact, pushed]
            for waypoint in active_waypoints[1:]:
                waypoint[0] -= self.eef_fingertip_offset
            left_primitive = CurvePrimitive(active_waypoints, self.config)
            right_primitive = StabilizePrimitive(current_pos[2:], self.config)
        else:
            active_waypoints = [current_pos[2:], approach, contact, pushed]
            for waypoint in active_waypoints[1:]:
                waypoint[0] += self.eef_fingertip_offset
            left_primitive = StabilizePrimitive(current_pos[:2], self.config)
            right_primitive = CurvePrimitive(active_waypoints, self.config)

        return (
            self.coordinator.coordinate(
                left_primitive,
                right_primitive,
                duration,
                num_steps,
                "simultaneous",
                speed_profile="constant",
            ),
            True,
            evaluate_random_contact_success,
        )

    def _plan_random_no_contact_motion(
        self,
        t_analyzer: TGeometryAnalyzer,
        collision_checker: CollisionChecker,
        current_pos: np.ndarray,
        duration: float,
        num_steps: int,
    ) -> tuple[np.ndarray, bool, callable]:
        """Plan random motion with no T contact."""
        # Generate random target positions away from T
        max_attempts = 20

        for attempt in range(max_attempts):
            # Random positions in workspace
            left_target = np.array(
                [
                    random.uniform(self.constraints.x_min, self.constraints.x_max),
                    random.uniform(self.constraints.y_min, self.constraints.y_max),
                ]
            )
            right_target = np.array(
                [
                    random.uniform(self.constraints.x_min, self.constraints.x_max),
                    random.uniform(self.constraints.y_min, self.constraints.y_max),
                ]
            )

            # Check constraints
            if (
                collision_checker.check_workspace_bounds(left_target)
                and collision_checker.check_workspace_bounds(right_target)
                and not collision_checker.check_arm_collision(left_target, right_target)
                and not collision_checker.check_t_collision(
                    current_pos[:2], left_target, allow_contact=False
                )
                and not collision_checker.check_t_collision(
                    current_pos[2:], right_target, allow_contact=False
                )
            ):
                # Create linear primitives
                left_primitive = LinearPrimitive(
                    current_pos[:2], left_target, self.config
                )
                right_primitive = LinearPrimitive(
                    current_pos[2:], right_target, self.config
                )

                # Use sequential coordination for exploration-like behavior
                sync_type = "overlap"
                success_fn = evaluate_random_motion_success

                return (
                    self.coordinator.coordinate(
                        left_primitive,
                        right_primitive,
                        duration,
                        num_steps,
                        sync_type,
                        speed_profile="variable",
                    ),
                    True,
                    success_fn,
                )
        return np.zeros((num_steps, 4)), False, None
