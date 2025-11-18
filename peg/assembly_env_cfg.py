# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sim.spawners.from_files.from_files_cfg import (GroundPlaneCfg,
                                                             UsdFileCfg)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .assets import UR3e_ROBOTIQ_GRIPPER_CFG, custom_hole, custom_peg

##
# Scene definition
##

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


@configclass
class AssemblySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # TODO: find a better place to store the hardcoded parameters

    # robot
    robot: ArticulationCfg = UR3e_ROBOTIQ_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.13],
                ),
            ),
        ],
    )

    peg_bottom_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Peg/Peg",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/Peg",
                name="peg_bottom",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.036],
                ),
            ),
        ],
    )

    # peg
    peg: ArticulationCfg = custom_peg.replace(prim_path="/World/envs/env_.*/Peg")

    # hole
    hole: ArticulationCfg = custom_hole.replace(prim_path="/World/envs/env_.*/Hole")

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

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # TODO: make this a command
        hole_position = ObsTerm(func=mdp.hole_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    critic: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # TODO: maybe initialize the arm to be above the hole?

    reset_peg = EventTerm(func=mdp.reset_peg_in_hand, mode="reset", params={})

    close_gripper = EventTerm(
        func=mdp.reset_joints_selected,
        mode="reset",
        params={
            "joint_positions": {"Slider_.*": -0.021},
            "target_positions": {"Slider_.*": -0.025},
        },
    )

    reset_hole = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("hole", body_names="Hole"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    peg_orientation_tracking = RewTerm(
        func=mdp.orientation_error, weight=1.0, params={"std": 1.0, "kernel": "tanh"}
    )

    peg_orientation_tracking_fine_grained = RewTerm(
        func=mdp.orientation_error, weight=2.0, params={"std": 0.25, "kernel": "tanh"}
    )

    peg_position_xy_tracking = RewTerm(
        func=mdp.position_xy_error, weight=1.0, params={"std": 0.25, "kernel": "exp"}
    )

    peg_position_xy_tracking_fine_grained = RewTerm(
        func=mdp.position_xy_error, weight=2.0, params={"std": 0.1, "kernel": "tanh"}
    )

    peg_position_z_tracking = RewTerm(
        func=mdp.position_z_error,
        weight=4.0,
        params={"std_xy": 0.1, "std_z": 0.2, "std_rz": 0.25, "kernel": "exp"},
    )

    peg_position_z_tracking_fine_grained = RewTerm(
        func=mdp.position_z_error,
        weight=10.0,
        params={"std_xy": 0.1, "std_z": 0.1, "std_rz": 0.25, "kernel": "tanh"},
    )

    task_success_bonus = RewTerm(
        func=mdp.peg_insertion_success,
        weight=1000.0,
        params={"location_threshold": 0.0002},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    peg_dropping = DoneTerm(func=mdp.peg_dropping, params={"threshold": 0.1})

    success = DoneTerm(func=mdp.success, params={"location_threshold": 0.0002})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class AssemblyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: AssemblySceneCfg = AssemblySceneCfg(num_envs=4096, env_spacing=4.0)
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
        self.episode_length_s = 5
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


@configclass
class AssemblyEnvCfg_PLAY(AssemblyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False
