import joint_env_cfg
from isaaclab.controllers.differential_ik_cfg import \
    DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import \
    DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from ...assets import ROBOTIQ_GRIPPER_CENTER_OFFSET


@configclass
class ChamferedPegInsertEnvCfg(joint_env_cfg.ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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
        body_name="ee_link",
        controller=DifferentialIKControllerCfg(command_type="pose", uses_relative=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, ROBOTIQ_GRIPPER_CENTER_OFFSET]))


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1