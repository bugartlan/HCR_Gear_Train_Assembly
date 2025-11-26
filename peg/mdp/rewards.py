# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_keypoints_distance(
    env: ManagerBasedRLEnv,
    length: float = 0.016,
    n_points: int = 5,
    std: float = 0.001,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    peg: FrameTransformer = env.scene[peg_cfg.name]  # Origin at peg bottom
    hole: Articulation = env.scene[hole_cfg.name]  # Origin at at hole entry

    peg_keypoints = [
        peg.data.target_pos_w[:, 0, :]
        + i
        * (length / (n_points - 1))
        * torch.tensor([0.0, 0.0, 1.0], device=peg.data.target_pos_w.device)
        for i in range(n_points)
    ][::-1]  # From top to bottom
    hole_keypoints = [
        hole.data.root_pos_w
        + i
        * (length / (n_points - 1))
        * torch.tensor([0.0, 0.0, -1.0], device=hole.data.root_pos_w.device)
        for i in range(n_points)
    ]

    distances = []
    for peg_pt, hole_pt in zip(peg_keypoints, hole_keypoints):
        distances.append(torch.exp(-(peg_pt - hole_pt).pow(2).sum(dim=1) / std**2))

    return torch.stack(distances, dim=0).mean(dim=0)


def success_bonus(
    env: ManagerBasedRLEnv,
    length: float = 0.016,
    std: float = 0.01,
    orientation_threshold: float = 0.01,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Give a bonus reward for successful peg insertion."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w
    hole_rot_z_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]
    peg_pos_w = peg.data.target_pos_w[:, 0, :]
    peg_rot_z_w = matrix_from_quat(peg.data.target_quat_w[:, 0, :])[:, :, 2]

    alignment = torch.cosine_similarity(peg_rot_z_w, hole_rot_z_w, dim=1).pow(2)
    distance = peg_pos_w[:, 2] - (hole_pos_w[:, 2] - length)
    return (
        alignment.ge(1 - orientation_threshold).float()
        * distance.le(length).float()
        * torch.exp(-(distance**2) / std**2)
    )


def position_xy_error(
    env: ManagerBasedRLEnv,
    std: float,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using exp-kernel."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w
    peg_w = peg.data.target_pos_w[:, 0, :]
    error = torch.norm(hole_pos_w[:, :2] - peg_w[:, :2], dim=1)

    return torch.exp(-error / std**2)


def position_z_error(
    env: ManagerBasedRLEnv,
    std_xy: float = 1.0,
    std_z: float = 1.0,
    std_rz: float = 1.0,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using exp-kernel."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w
    peg_pos_w = peg.data.target_pos_w[:, 0, :]
    xy_error = torch.sum((hole_pos_w[:, :2] - peg_pos_w[:, :2]).pow(2), dim=1)
    z_error = hole_pos_w[:, 2] - peg_pos_w[:, 2]

    peg_rz_w = matrix_from_quat(peg.data.target_quat_w[:, 0, :])[:, :, 2]
    hole_rz_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]
    alignment_error = 1 - torch.cosine_similarity(peg_rz_w, hole_rz_w, dim=1)

    return (
        torch.exp(-xy_error / std_xy**2)
        * torch.exp(-alignment_error / std_rz**2)
        * torch.exp(z_error / std_z**2)
    )


def orientation_error(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    sign: float = 1.0,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for aligning the peg axis with the hole axis."""
    peg: Articulation = env.scene[peg_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]
    # Peg z-axis in world frame: (num_envs, 3)
    peg_z_w = matrix_from_quat(peg.data.root_quat_w)[:, :, 2]
    # Hole z-axis in world frame: (num_envs, 3)
    hole_z_w = matrix_from_quat(hole.data.root_quat_w)[:, :, 2]

    error = 1 - sign * torch.cosine_similarity(peg_z_w, hole_z_w, dim=1)
    return torch.exp(-error / std**2)


def peg_slip(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize the agent if the peg is dropped below a certain height."""
    peg: Articulation = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    peg_pos_w = peg.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    dist = torch.norm(peg_pos_w - ee_pos_w, dim=1)
    return dist.ge(threshold).float() * dist
