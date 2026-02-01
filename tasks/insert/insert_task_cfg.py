import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...assembly_env_cfg import AssemblyEnvCfg, AssemblySceneCfg
from ...assets import HoleSpec, PegSpec

target_marker_cfg = FRAME_MARKER_CFG.copy()
target_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
target_marker_cfg.prim_path = "/Visuals/TargetFrame"

held_marker_cfg = FRAME_MARKER_CFG.copy()
held_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
held_marker_cfg.prim_path = "/Visuals/HeldAssetFrame"


@configclass
class InsertSceneCfg(AssemblySceneCfg):
    fixed_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HoleSpec.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.4, 0.0, HoleSpec.height),
            rot=(0.0, 1.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    held_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PegSpec.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.2, 0.2, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Hole/Hole",
        debug_vis=False,
        visualizer_cfg=target_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Hole/Hole",
                name="target_frame",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            ),
        ],
    )

    held_asset_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Peg/Peg",
        debug_vis=False,
        visualizer_cfg=held_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/Peg",
                name="held_asset_frame",
                offset=OffsetCfg(pos=PegSpec.tip_pos, rot=PegSpec.tip_rot),
            ),
        ],
    )


@configclass
class InsertEnvCfg(AssemblyEnvCfg):
    scene: InsertSceneCfg = InsertSceneCfg(num_envs=1024, env_spacing=2.5)

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 15.0

        self.rewards.keypoint_distance_baseline.params["length"] = HoleSpec.height
        self.rewards.keypoint_distance_coarse.params["length"] = HoleSpec.height
        self.rewards.keypoint_distance_fine.params["length"] = HoleSpec.height
        self.rewards.task_success_bonus.params["length"] = HoleSpec.height
        self.rewards.slip.params["length"] = PegSpec.total_length
        self.terminations.dropped.params["length"] = PegSpec.total_length
