from isaaclab.assets import ArticulationCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from ... import mdp
from ...assets import (
    UR3e_ROBOTIQ_GRIPPER_CFG,
    factory_gear_base_cfg,
    factory_large_gear_cfg,
    factory_medium_gear_cfg,
    factory_small_gear_cfg,
)
from ...gear_env_cfg import GearMeshEnvCfg

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 1.0
    mass: float = 0.5


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 1.0
    mass: float = 0.05


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012


@configclass
class FactoryGearMeshEnvCfg(GearMeshEnvCfg):
    held_asset: HeldAssetCfg = MediumGear()
    fixed_asset: FixedAssetCfg = GearBase()

    def __post_init__(self):
        super().__post_init__()

        self.scene.small_gear = factory_small_gear_cfg
        self.scene.medium_gear = factory_medium_gear_cfg
        self.scene.large_gear = factory_large_gear_cfg
        self.scene.gear_base = factory_gear_base_cfg

        self.scene.robot = UR3e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.1,
            use_default_offset=True,
        )

        self.events.reset_held_gear.params["tf_pos"] = [
            -0.02,
            0.0,
            -self.held_asset.height + 0.002,
        ]
        self.events.reset_held_gear.params["tf_quat"] = [0.0, 0.0, 0.0, 1.0]
