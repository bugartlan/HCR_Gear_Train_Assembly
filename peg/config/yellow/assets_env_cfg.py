from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assembly_env_cfg import AssemblyEnvCfg
from ...assets import factory_hole_8mm, factory_peg_8mm


@configclass
class YellowPegInsertEnvCfg(AssemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.peg = factory_peg_8mm.replace(prim_path="{ENV_REGEX_NS}/Peg")
        self.scene.hole = factory_hole_8mm.replace(prim_path="{ENV_REGEX_NS}/Hole")

        self.scene.peg_bottom_frame.prim_path = "{ENV_REGEX_NS}/Peg/forge_round_peg_8mm"
        self.scene.peg_bottom_frame.target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/forge_round_peg_8mm",
                name="peg_bottom",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.05],
                ),
            ),
        ]

        self.events.reset_peg = EventTerm(
            func=mdp.reset_peg_in_hand,
            mode="reset",
            params={
                "tf_pos": [0.0, 0.0, 0.0],
                "tf_quat": [1.0, 0.0, 0.0, 0.0],
            },
        )

        self.events.reset_hole = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("hole", body_names="forge_hole_8mm"),
            },
        )


@configclass
class YellowPegInsertEnvCfg_PLAY(YellowPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
