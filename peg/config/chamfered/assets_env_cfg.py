from isaaclab.envs.common import ViewerCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...assembly_env_cfg import AssemblyEnvCfg
from ...assets import CustomHole, CustomPeg, custom_peg


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

        self.scene.hole.init_state.pos = (0.3, 0.0, self.hole.height)
        self.scene.peg = custom_peg.replace(prim_path="{ENV_REGEX_NS}/Peg")
        self.scene.peg_bottom_frame.target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/Peg",
                name="peg_bottom",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, self.peg.total_length],
                ),
            ),
        ]

        self.events.reset_peg.params = {
            "tf_pos": [0.0, 0.0, 0.005],
            "tf_quat": [0.707, 0.0, 0.0, 0.707],
        }
        self.events.reset_hole.params["pose_range"] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (0.0, 0.0),
        }

        self.rewards.task_success_bonus.params["location_threshold"] = 0.0003
        self.rewards.task_success_bonus.params["hole_offset"] = [
            0.0,
            0.0,
            -self.hole.height,
        ]
        self.terminations.success.params["location_threshold"] = 0.0003
        self.terminations.success.params["hole_offset"] = [0.0, 0.0, -self.hole.height]


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.viewer = ViewerCfg(
            eye=(1.2, -0.2, 0.25), origin_type="asset_root", asset_name="robot"
        )
        self.scene.env_spacing = 2.5
        self.events.reset_hole.params["pose_range"] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (0.0, 0.0),
        }
