from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hole_pos_wrt_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """The position of the hole in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]

    hole_pos_b, hole_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        hole.data.root_pos_w,
        hole.data.root_quat_w,
    )
    return hole_pos_b


def peg_pos_wrt_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
) -> torch.Tensor:
    """The position of the hole in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    peg: FrameTransformer = env.scene[peg_cfg.name]

    peg_pos_w = peg.data.target_pos_w[:, 0, :]
    peg_quat_w = peg.data.target_quat_w[:, 0, :]
    peg_pos_b, peg_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, peg_pos_w, peg_quat_w
    )
    return torch.cat((peg_pos_b, peg_quat_b), dim=1)


def peg_vel_wrt_robot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The velocity of the peg in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    peg: Articulation = env.scene[peg_cfg.name]

    peg_vel_b = quat_apply_inverse(robot.data.root_quat_w, peg.data.root_lin_vel_w)
    peg_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, peg.data.root_ang_vel_w)

    return torch.cat((peg_vel_b, peg_ang_vel_b), dim=1)
