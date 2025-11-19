import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .utils import is_inserted


def success(
    env: ManagerBasedRLEnv,
    location_threshold: float = 0.0001,
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
    return is_inserted(peg_w, hole_pos_w, threshold=location_threshold)


def peg_dropping(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate the episode if the peg is dropped out of the fingers.

    Args:
        env: The environment object.
        threshold: The distance threshold above which the episode is terminated.
        peg_cfg: The configuration for the peg entity.
        ee_frame_cfg: The configuration for the end-effector frame entity.
    ) -> torch.Tensor:
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    peg_pos_w = peg.data.root_pos_w
    ee_frame_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    return torch.norm(peg_pos_w - ee_frame_pos_w, dim=1) > threshold
