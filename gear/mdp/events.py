import torch
from isaaclab.assets import Articulation
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
    robot: Articulation = env.scene[asset_cfg.name]
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


def reset_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    small_gear_cfg: SceneEntityCfg = SceneEntityCfg("small_gear"),
    large_gear_cfg: SceneEntityCfg = SceneEntityCfg("large_gear"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("gear_base"),
):
    """Reset the gears in the scene to their initial positions.

    Args:
        env: The environment object.
        env_ids: The environment IDs to reset.
        small_gear: The configuration for the small gear entity.
        medium_gear: The configuration for the medium gear entity.
        large_gear: The configuration for the large gear entity.
    """
    small_gear: Articulation = env.scene[small_gear_cfg.name]
    large_gear: Articulation = env.scene[large_gear_cfg.name]
    gear_base: Articulation = env.scene[gear_base_cfg.name]

    fixed_state = gear_base.data.default_root_state.clone()
    gear_base.write_root_state_to_sim(fixed_state, env_ids)
    small_gear.write_root_state_to_sim(fixed_state, env_ids)
    large_gear.write_root_state_to_sim(fixed_state, env_ids)


def reset_held_gear(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tf_pos: list = [0.0, 0.0, 0.0],
    tf_quat: list = [1.0, 0.0, 0.0, 0.0],
    medium_gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_gear"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the small gear held by the robot to a specific position.

    Args:
        env: The environment object.
        env_ids: The environment IDs to reset.
        small_gear: The configuration for the small gear entity.
    """
    medium_gear: Articulation = env.scene[medium_gear_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    _ = robot.data.body_link_pos_w  # trigger internal update
    ee_frame.update(dt=0, force_recompute=True)

    root = medium_gear.data.default_root_state.clone()
    dp = torch.tensor(tf_pos, device=root.device).broadcast_to(root[:, :3].shape)
    dq = torch.tensor(tf_quat, device=root.device).broadcast_to(root[:, 3:7].shape)
    root[:, :3] = ee_frame.data.target_pos_w[..., 0, :] + dp
    root[:, 3:7] = quat_mul(root[:, 3:7], dq)
    root[:, 7:] = 0.0  # zero velocity
    medium_gear.write_root_state_to_sim(root[env_ids], env_ids=env_ids)
