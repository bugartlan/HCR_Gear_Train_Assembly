from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assembly_env_cfg import AssemblyEnvCfg
from ...assets import (
    CustomHole,
    CustomPeg,
    UR3e_ROBOTIQ_GRIPPER_CFG,
    custom_hole,
    custom_peg,
)


@configclass
class ChamferedPegInsertEnvCfg(AssemblyEnvCfg):
    peg = CustomPeg()
    hole = CustomHole()

    def __post_init__(self):
        super().__post_init__()

        # Flipping the table and robot to better suit the real world setup
        # self.scene.table.init_state.pos = (0.225, 0.0, 0.0)
        # self.scene.table.init_state.rot = (0.707, 0.0, 0.0, -0.707)
        # self.scene.robot.init_state.pos = (0.8, 0.0, 0.0)
        # self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        # self.scene.hole.init_state.pos = (0.5, 0.0, self.hole.height)

        self.scene.robot = UR3e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.scene.hole = custom_hole.replace(prim_path="/World/envs/env_.*/Hole")
        self.scene.hole.init_state.pos = (0.3, 0.0, self.hole.height)

        self.scene.peg = custom_peg.replace(prim_path="/World/envs/env_.*/Peg")

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer_Peg"
        self.scene.peg_bottom_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Peg/Peg",
            debug_vis=False,
            visualizer_cfg=marker_cfg.copy(),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Peg/Peg",
                    name="peg_bottom",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, self.peg.total_length],
                    ),
                ),
            ],
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

        self.events.reset_peg.params = {
            "tf_pos": [0.0, 0.0, 0.005],
            "tf_quat": [0.707, 0.0, 0.0, 0.707],
        }
        self.events.reset_hole.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.0, 0.0),
            "z": (0.0, 0.0),
        }

        self.rewards.task_success_bonus.params["length"] = self.hole.height

        self.terminations.success.params["location_threshold"] = self.hole.height * 0.1
        self.terminations.success.params["hole_offset"] = [0.0, 0.0, -self.hole.height]


@configclass
class ChamferedPegInsertEnvCfg_VIDEO(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.viewer = ViewerCfg(
            eye=(1.0, -0.2, 0.4), origin_type="asset_root", asset_name="robot"
        )


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
