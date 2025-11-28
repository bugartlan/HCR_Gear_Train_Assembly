# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def keypoints_distance(
    env: ManagerBasedRLEnv,
    offset: list[float],
    length: float = 0.016,
    n_points: int = 5,
    std: float = 0.001,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    gear: Articulation = env.scene[gear_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    device = gear_base.data.root_pos_w.device
    offset_tensor = torch.tensor(offset, device=device)
    plug_pos_w = gear_base.data.root_pos_w + offset_tensor

    gear_keypoints = [
        gear.data.root_pos_w
        + i * (length / (n_points - 1)) * torch.tensor([0.0, 0.0, 1.0], device=device)
        for i in range(n_points)
    ]
    plug_keypoints = [
        plug_pos_w
        + i * (length / (n_points - 1)) * torch.tensor([0.0, 0.0, 1.0], device=device)
        for i in range(n_points)
    ]

    rewards = []
    for gear_pt, plug_pt in zip(gear_keypoints, plug_keypoints):
        rewards.append(torch.exp(-(gear_pt - plug_pt).pow(2).sum(dim=1) / std**2))

    return torch.stack(rewards, dim=0).mean(dim=0)


def success_bonus(
    env: ManagerBasedRLEnv,
    offset: list[float],
    length: float = 0.016,
    std: float = 0.01,
    translation_xy_threshold: float = 0.001,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    gear: Articulation = env.scene[gear_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    device = gear_base.data.root_pos_w.device
    offset_tensor = torch.tensor(offset, device=device)
    plug_pos_w = gear_base.data.root_pos_w + offset_tensor

    xy_distance = torch.norm(gear.data.root_pos_w[:, :2] - plug_pos_w[:, :2], dim=1)
    z_distance = gear.data.root_pos_w[:, 2] - plug_pos_w[:, 2]
    return (
        xy_distance.le(translation_xy_threshold).float()
        * z_distance.le(length).float()
        * torch.exp(-(z_distance**2) / std**2)
    )


def slip(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize the agent if the peg is dropped below a certain height."""
    gear: Articulation = env.scene[gear_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    gear_pos_w = gear.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    dist = torch.norm(gear_pos_w - ee_pos_w, dim=1)
    return dist.ge(threshold).float() * dist
