import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .utils import distance, is_inserted


def completion(
    env: ManagerBasedRLEnv,
    location_threshold: float = 0.0001,
    hole_offset: list[float] = [0.0, 0.0, 0.0],
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg_bottom_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Reward the agent for successful peg insertion."""
    peg: FrameTransformer = env.scene[peg_cfg.name]
    hole: Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w + torch.tensor(
        hole_offset, device=hole.data.root_pos_w.device
    )
    peg_w = peg.data.target_pos_w[:, 0, :]
    return is_inserted(peg_w, hole_pos_w, threshold=location_threshold)


def dropped(
    env: ManagerBasedRLEnv,
    length: float,
    threshold: float = 0.1,
    held_asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate the episode if the peg is dropped out of the fingers.

    Args:
        env: The environment object.
        threshold: The distance threshold above which the episode is terminated.
        held_asset_cfg: The configuration for the held asset entity.
        ee_frame_cfg: The configuration for the end-effector frame entity.
    ) -> torch.Tensor:
    """
    held_asset: Articulation = env.scene[held_asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    dist = distance(
        env.num_envs,
        n_points=4,
        length=length,
        asset1=held_asset,
        asset2=ee_frame,
        device=env.device,
    )

    return dist.mean(dim=0) > threshold
