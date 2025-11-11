import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


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
