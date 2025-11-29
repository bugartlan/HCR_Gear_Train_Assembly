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
from isaaclab.utils.math import matrix_from_quat, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def keypoint_distance(
    env: ManagerBasedRLEnv,
    length: float,
    n_points: int = 4,
    std: float = 0.001,
    offset1: list[float] = [0.0, 0.0, 0.0],
    offset2: list[float] = [0.0, 0.0, 0.0],
    quat1: list[float] = [1.0, 0.0, 0.0, 0.0],
    quat2: list[float] = [1.0, 0.0, 0.0, 0.0],
    asset1_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    asset2_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    asset1: Articulation | FrameTransformer = env.scene[asset1_cfg.name]
    asset2: Articulation | FrameTransformer = env.scene[asset2_cfg.name]

    n, _ = asset1.data.root_pos_w.shape
    device = asset1.data.root_pos_w.device
    offset1_tensor = torch.tensor(offset1, device=device)
    offset2_tensor = torch.tensor(offset2, device=device)
    quat1_tensor = torch.tensor(quat1, device=device).broadcast_to(n, 4)
    quat2_tensor = torch.tensor(quat2, device=device).broadcast_to(n, 4)
    vec_z = torch.tensor([0.0, 0.0, 1.0], device=device).broadcast_to(n, 3)

    asset1_pos_w = (
        asset1.data.root_pos_w
        if isinstance(asset1, Articulation)
        else asset1.data.target_pos_w[:, 0, :]
    )
    asset1_quat_w = (
        asset1.data.root_quat_w
        if isinstance(asset1, Articulation)
        else asset1.data.target_quat_w[:, 0, :]
    )
    asset2_pos_w = (
        asset2.data.root_pos_w
        if isinstance(asset2, Articulation)
        else asset2.data.target_pos_w[:, 0, :]
    )
    asset2_quat_w = (
        asset2.data.root_quat_w
        if isinstance(asset2, Articulation)
        else asset2.data.target_quat_w[:, 0, :]
    )
    asset1_pos_w = asset1_pos_w + offset1_tensor
    asset2_pos_w = asset2_pos_w + offset2_tensor
    asset1_ax_z_w = quat_apply(asset1_quat_w, quat_apply(quat1_tensor, vec_z))
    asset2_ax_z_w = quat_apply(asset2_quat_w, quat_apply(quat2_tensor, vec_z))
    asset1_keypoints = [
        asset1_pos_w + i * (length / (n_points - 1)) * asset1_ax_z_w
        for i in range(n_points)
    ]
    asset2_keypoints = [
        asset2_pos_w + i * (length / (n_points - 1)) * asset2_ax_z_w
        for i in range(n_points)
    ]

    rewards = []
    for asset1_pt, asset2_pt in zip(asset1_keypoints, asset2_keypoints):
        rewards.append(torch.exp(-(asset1_pt - asset2_pt).pow(2).sum(dim=1) / std**2))

    return torch.stack(rewards, dim=0).mean(dim=0)


def discrete_success_bonus(
    env: ManagerBasedRLEnv,
    offset: list[float] = [0.0, 0.0, 0.0],
    trans_xy_threshold: float = 0.001,
    trans_z_threshold: float = 0.001,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    gear: Articulation = env.scene[gear_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    device = gear_base.data.root_pos_w.device
    gear_base_pos_w = gear_base.data.root_pos_w + torch.tensor(offset, device=device)

    xy_dist = torch.norm(gear.data.root_pos_w[:, :2] - gear_base_pos_w[:, :2], dim=1)
    z_dist = torch.abs(gear.data.root_pos_w[:, 2] - gear_base_pos_w[:, 2])

    return xy_dist.le(trans_xy_threshold).float() * z_dist.le(trans_z_threshold).float()


def keypoints_distance(
    env: ManagerBasedRLEnv,
    offset: list[float],
    length: float = 0.016,
    n_points: int = 4,
    std: float = 0.001,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
) -> torch.Tensor:
    gear: Articulation = env.scene[gear_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    device = gear_base.data.root_pos_w.device
    offset_tensor = torch.tensor(offset, device=device)
    plug_pos_w = gear_base.data.root_pos_w + offset_tensor

    gear_ax_z_w = matrix_from_quat(gear.data.root_quat_w)[:, :, 2]
    plug_ax_z_w = matrix_from_quat(gear_base.data.root_quat_w)[:, :, 2]

    gear_keypoints = [
        gear.data.root_pos_w + i * (length / (n_points - 1)) * gear_ax_z_w
        for i in range(n_points)
    ]
    plug_keypoints = [
        plug_pos_w + i * (length / (n_points - 1)) * plug_ax_z_w
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
