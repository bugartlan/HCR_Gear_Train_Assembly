import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def held_asset_dropped(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    medium_gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate the episode if the peg is dropped out of the fingers.

    Args:
        env: The environment object.
        threshold: The distance threshold above which the episode is terminated.
        medium_gear_cfg: The configuration for the medium gear entity.
        ee_frame_cfg: The configuration for the end-effector frame entity.
    ) -> torch.Tensor:
    """
    medium_gear: Articulation = env.scene[medium_gear_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    medium_gear_pos_w = medium_gear.data.root_pos_w
    ee_frame_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    return torch.norm(medium_gear_pos_w - ee_frame_pos_w, dim=1) > threshold
