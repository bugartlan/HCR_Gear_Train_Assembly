from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from ... import mdp
from ...assets import (
    GearTrainGearBase,
    GearTrainMediumGear,
    UR3e_ROBOTIQ_GRIPPER_CFG,
    geartrain_gear_base_cfg,
    geartrain_large_gear_cfg,
    geartrain_medium_gear_cfg,
    geartrain_small_gear_cfg,
)
from ...gear_env_cfg import GearMeshEnvCfg

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class GearTrainGearMeshEnvCfg(GearMeshEnvCfg):
    fixed_asset = GearTrainGearBase()
    held_asset = GearTrainMediumGear()

    def __post_init__(self):
        super().__post_init__()

        self.scene.small_gear = geartrain_small_gear_cfg
        self.scene.medium_gear = geartrain_medium_gear_cfg
        self.scene.large_gear = geartrain_large_gear_cfg
        self.scene.gear_base = geartrain_gear_base_cfg

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

        params = {"offset": self.fixed_asset.medium_gear_base_offset}
        self.observations.policy.plug_pos.params.update(params)
        self.observations.critic.plug_pos.params.update(params)
        params = {
            "length": self.held_asset.height,
            "offset": self.fixed_asset.medium_gear_base_offset,
        }
        self.rewards.keypoint_distance_baseline.params.update(params)
        self.rewards.keypoint_distance_coarse.params.update(params)
        self.rewards.keypoint_distance_fine.params.update(params)
        self.rewards.task_success_bonus.params.update(params)

        self.rewards

        gear_base_offsets = {
            "small_gear_base_offset": self.fixed_asset.small_gear_base_offset,
            "large_gear_base_offset": self.fixed_asset.large_gear_base_offset,
        }
        self.events.reset_fixed_gears.params["offsets"] = gear_base_offsets
        self.events.reset_held_gear.params.update(
            {
                "tf_pos": [
                    0.0,
                    0.0,
                    -self.held_asset.height + self.held_asset.held_offset,
                ],
                "tf_quat": [1.0, 0.0, 0.0, 0.0],
            }
        )


@configclass
class GearTrainGearMeshEnvCfg_VIDEO(GearTrainGearMeshEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.viewer = ViewerCfg(
            eye=(1.0, -0.2, 0.4), origin_type="asset_root", asset_name="robot"
        )
