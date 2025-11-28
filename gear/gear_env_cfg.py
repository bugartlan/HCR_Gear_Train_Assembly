# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise.noise_cfg import GaussianNoiseCfg

from . import mdp
from .assets import ROBOTIQ_GRIPPER_CENTER_OFFSET

##
# Scene definition
##

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
robot_selected_joints = SceneEntityCfg(name="robot", joint_names=JOINT_NAMES)


@configclass
class GearMeshSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # Gears
    small_gear: ArticulationCfg = MISSING
    medium_gear: ArticulationCfg = MISSING
    large_gear: ArticulationCfg = MISSING
    gear_base: ArticulationCfg = MISSING

    # robot
    robot: ArticulationCfg = MISSING

    # end effector frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg.copy(),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, ROBOTIQ_GRIPPER_CENTER_OFFSET],
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": robot_selected_joints},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": robot_selected_joints},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        wrench_ee = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", body_names="wrist_3_link")
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )
        plug_pos = ObsTerm(
            func=mdp.plug_pos, noise=GaussianNoiseCfg(mean=0.0, std=0.001)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

        gear_position = ObsTerm(func=mdp.gear_pos_wrt_robot)
        gear_velocity = ObsTerm(func=mdp.gear_vel_wrt_robot)

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    critic: PolicyCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_fixed_gears = EventTerm(func=mdp.reset_scene, mode="reset", params={})
    reset_held_gear = EventTerm(func=mdp.reset_held_gear, mode="reset", params={})

    close_gripper = EventTerm(
        func=mdp.reset_joints_selected,
        mode="reset",
        params={
            "joint_positions": {"Slider_.*": -0.012},
            "target_positions": {"Slider_.*": -0.025},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    keypoint_distance_baseline = RewTerm(
        func=mdp.keypoints_distance,
        weight=1.0,
        params={"n_points": 8, "std": 0.1},
    )

    keypoint_distance_coarse = RewTerm(
        func=mdp.keypoints_distance,
        weight=1.0,
        params={"n_points": 8, "std": 0.04},
    )

    keypoint_distance_fine = RewTerm(
        func=mdp.keypoints_distance,
        weight=1.0,
        params={"n_points": 8, "std": 0.01},
    )

    task_success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=10.0,
        params={"std": 0.01},
    )

    slip = RewTerm(func=mdp.slip, weight=-10.0, params={"threshold": 0.01})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    held_asset_dropped = DoneTerm(mdp.held_asset_dropped, params={"threshold": 0.08})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 100000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 100000},
    )

    joint_acc = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_acc", "weight": -1e-3, "num_steps": 100000},
    )

    joint_torque = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_torque", "weight": -1e-3, "num_steps": 100000},
    )


##
# Environment configuration
##


@configclass
class GearMeshEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GearMeshSceneCfg = GearMeshSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = (
            192  # Important to avoid interpenetration.
        )
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.gpu_max_num_partitions = 1
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
