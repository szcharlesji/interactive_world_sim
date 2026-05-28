"""Backward-compatible PushT wrapper.

New code should import `SimAlohaPlanarPushEnv` from
`interactive_world_sim.environments.sim_aloha_planar_push_env` and pass
`shape="T"`, `shape="H"`, etc.  This module keeps the historical class name
available for callers that still expect `SimAlohaPushTEnv`.
"""

from .sim_aloha_planar_push_env import (
    SimAlohaPlanarPushEnv,
    mat_to_rot_6d,
    pos_quat_to_mat,
    rot_6d_to_mat,
)


class SimAlohaPushTEnv(SimAlohaPlanarPushEnv):
    """Compatibility alias for the procedural T planar pushing env."""

    def __init__(
        self,
        task: str = "planar_push",
        render_size: tuple[int, int] = (128, 128),
        delta_action: bool = False,
    ):
        super().__init__(
            shape="T",
            task=task,
            render_size=render_size,
            delta_action=delta_action,
        )
