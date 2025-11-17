from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assembly_env_cfg import AssemblyEnvCfg
from ...assets import custom_peg


@configclass
class ChamferedPegInsertEnvCfg(AssemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.peg = custom_peg.replace(prim_path="{ENV_REGEX_NS}/Peg")

        self.scene.peg_bottom_frame.target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/Peg",
                name="peg_bottom",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.04],
                ),
            ),
        ]

        self.events.reset_peg.params = {
            "tf_pos": [0.0, 0.0, 0.005],
            "tf_quat": [0.7071, 0.0, 0.0, 0.7071],
        }
        self.rewards.task_success_bonus.params["hole_offset"] = [0.0, 0.0, -0.016]
        self.terminations.success.params["hole_offset"] = [0.0, 0.0, -0.016]


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.events.reset_hole = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("hole", body_names="Hole"),
            },
        )
