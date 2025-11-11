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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hole_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    hole: RigidObject = env.scene[hole_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    hole_pos_w = hole.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    hole_ee_distance = torch.norm(hole_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(hole_ee_distance / std)


def axis_alignment(
    env: ManagerBasedRLEnv,
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

    return torch.abs(torch.cosine_similarity(peg_z_w, hole_z_w, dim=1))


def peg_hole_horizontal_distance(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for minimizing the horizontal distance between peg and hole."""
    peg: RigidObject = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    # Peg position in world frame: (num_envs, 3)
    peg_pos_w = peg.data.root_pos_w
    # Hole position in world frame: (num_envs, 3)
    hole_pos_w = hole.data.root_pos_w

    horizontal_distance = torch.norm(peg_pos_w[:, :2] - hole_pos_w[:, :2], dim=1)

    return horizontal_distance


def insertion_depth(
    env: ManagerBasedRLEnv,
    location_threshold: float = 0.002,
    orientation_threshold: float = 0.95,
    height_threshold: float = 0.075,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for maximizing the insertion depth of the peg into the hole."""
    peg: RigidObject = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    peg_z_w = matrix_from_quat(peg.data.root_quat_w)[:, :, 2]
    hole_z_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]

    peg_pos_w = peg.data.root_pos_w
    hole_pos_w = hole.data.root_pos_w

    # is_within_location_threshold = (
    #     torch.norm(peg_pos_w[:, :2] - hole_pos_w[:, :2], dim=1) < location_threshold
    # ).to(torch.float32)
    # is_within_orientation_threshold = (
    #     torch.abs(torch.cosine_similarity(peg_z_w, hole_z_w, dim=1))
    #     > orientation_threshold
    # ).to(torch.float32)
    # is_within_height_threshold = (
    #     (peg_pos_w[:, 2] - hole_pos_w[:, 2]) < height_threshold
    # ).to(torch.float32)
    # return (
    #     is_within_location_threshold
    #     * is_within_orientation_threshold
    #     * (10 * is_within_height_threshold + 1)
    #     * (hole_pos_w[:, 2] - peg_pos_w[:, 2])
    # )

    return hole_pos_w[:, 2] - peg_pos_w[:, 2]
