from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from matplotlib import scale

from ...assets import ROBOTIQ_GRIPPER_CENTER_OFFSET, UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG
from . import joint_env_cfg


@configclass
class ChamferedPegInsertEnvCfg(joint_env_cfg.ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="robotiq_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, ROBOTIQ_GRIPPER_CENTER_OFFSET]
            ),
        )


@configclass
class ChamferedPegInsertEnvCfg_VIDEO(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.viewer = ViewerCfg(
            eye=(1.0, 0.0, 0.4), origin_type="asset_root", asset_name="robot"
        )


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
