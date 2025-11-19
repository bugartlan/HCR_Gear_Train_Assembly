import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_mul


def reset_joints_selected(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_positions: dict,
    target_positions: dict = {},
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset selected joints of the robot to specified positions.

    Args:
        env: The environment object.
        env_ids: The environment IDs to reset.
        joint_positions: A dictionary mapping joint names to their desired positions.
        target_positions: A dictionary mapping joint names to their target positions.
        asset_cfg: The configuration for the scene entity.
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.clone()
    target_pos = robot.data.joint_pos.clone()

    # update the specified joints
    for key, pos in joint_positions.items():
        joints = robot.find_joints(key)  # returns ([indices], [names])
        joint_pos[env_ids[:, None], joints[0]] = pos

    for key, pos in target_positions.items():
        joints = robot.find_joints(key)
        target_pos[env_ids[:, None], joints[0]] = pos

    robot.write_joint_position_to_sim(joint_pos[env_ids], env_ids=env_ids)
    robot.set_joint_position_target(target_pos[env_ids], env_ids=env_ids)


def reset_peg_in_hand(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tf_pos: list = [0.0, 0.0, 0.0],
    tf_quat: list = [0.707, 0.0, 0.0, 0.707],
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the peg in hand to a specific position.

    Args:
        env: The environment object.
        env_ids: The environment IDs to reset.
        peg_cfg: The configuration for the peg entity.
        asset_cfg: The configuration for the robot entity.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    _ = robot.data.body_link_pos_w  # trigger internal update
    ee_frame.update(dt=0, force_recompute=True)

    root = peg.data.default_root_state.clone()
    root[:, :3] = ee_frame.data.target_pos_w[..., 0, :] + torch.tensor(
        tf_pos, device=root.device
    ).broadcast_to(ee_frame.data.target_pos_w[..., 0, :].shape)
    root[:, 3:7] = quat_mul(
        ee_frame.data.target_quat_w[..., 0, :],
        torch.tensor(tf_quat, device=root.device).broadcast_to(
            ee_frame.data.target_quat_w[..., 0, :].shape
        ),
    )
    root[:, 7:] = 0.0  # zero velocity
    peg.write_root_state_to_sim(root[env_ids], env_ids=env_ids)
