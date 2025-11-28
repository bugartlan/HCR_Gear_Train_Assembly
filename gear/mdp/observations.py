from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def plug_pos(
    env: ManagerBasedRLEnv,
    offset: list[float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    """The position of the hole in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    plug_pos_w = gear_base.data.root_pos_w + torch.tensor(
        offset, device=gear_base.device
    )
    plug_quat_w = gear_base.data.root_quat_w

    plug_pos_b, plug_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        plug_pos_w,
        plug_quat_w,
    )
    return plug_pos_b


def gear_pos_wrt_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
) -> torch.Tensor:
    """The position of the hole in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    gear: Articulation = env.scene[gear_cfg.name]

    gear_pos_b, gear_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        gear.data.root_pos_w,
        gear.data.root_quat_w,
    )
    return torch.cat((gear_pos_b, gear_quat_b), dim=1)


def gear_vel_wrt_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
) -> torch.Tensor:
    """The velocity of the peg in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    gear: Articulation = env.scene[gear_cfg.name]

    gear_vel_b = quat_apply_inverse(robot.data.root_quat_w, gear.data.root_lin_vel_w)
    gear_ang_vel_b = quat_apply_inverse(
        robot.data.root_quat_w, gear.data.root_ang_vel_w
    )

    return torch.cat((gear_vel_b, gear_ang_vel_b), dim=1)
