# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat

from .utils import is_inserted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_insertion_success(
    env: ManagerBasedRLEnv,
    location_threshold: float = 0.001,
    hole_offset: list[float] = [0.0, 0.0, 0.0],
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for successful peg insertion."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w + torch.tensor(
        hole_offset, device=hole.data.root_pos_w.device
    )
    peg_w = peg.data.target_pos_w[:, 0, :]
    return is_inserted(peg_w, hole_pos_w, threshold=location_threshold).float()


def position_xy_error(
    env: ManagerBasedRLEnv,
    std: float,
    kernel: str = "exp",
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using exp-kernel."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w
    peg_w = peg.data.target_pos_w[:, 0, :]
    error = torch.norm(hole_pos_w[:, :2] - peg_w[:, :2], dim=1)

    if kernel == "tanh":
        return 1 - torch.tanh(error / std**2)
    elif kernel == "exp":
        return torch.exp(-error / std**2)
    else:
        return F.relu(1 - error / std**2)


def position_z_error(
    env: ManagerBasedRLEnv,
    std_xy: float = 1.0,
    std_z: float = 1.0,
    std_rz: float = 1.0,
    kernel: str = "exp",
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using exp-kernel."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w
    peg_pos_w = peg.data.target_pos_w[:, 0, :]
    xy_error = torch.norm(hole_pos_w[:, :2] - peg_pos_w[:, :2], dim=1)
    z_error = hole_pos_w[:, 2] - peg_pos_w[:, 2]

    peg_rz_w = matrix_from_quat(peg.data.target_quat_w[:, 0, :])[:, :, 2]
    hole_rz_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]
    alignment_error = 1 - torch.cosine_similarity(peg_rz_w, hole_rz_w, dim=1).pow(2)

    if kernel == "tanh":
        reward = 1 - torch.tanh(-z_error / std_z**2)
    else:
        reward = torch.exp(z_error / std_z**2)

    return (
        torch.exp(-xy_error / std_xy**2)
        * torch.exp(-alignment_error / std_rz**2)
        * reward
    )


def orientation_error(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    sign: float = 1.0,
    kernel: str = "exp",
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for aligning the peg axis with the hole axis."""
    peg: Articulation = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    # Peg z-axis in world frame: (num_envs, 3)
    peg_z_w = matrix_from_quat(peg.data.root_quat_w)[:, :, 2]
    # Hole z-axis in world frame: (num_envs, 3)
    hole_z_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]

    cos_error = sign * torch.cosine_similarity(peg_z_w, hole_z_w, dim=1)
    rad_error = torch.acos(cos_error)

    if kernel == "tanh":
        return 1 - torch.tanh(rad_error.pow(2) / std**2)
    elif kernel == "exp":
        return torch.exp(-rad_error.pow(2) / std**2)
    else:
        return F.relu(1 - rad_error.pow(2) / std**2)