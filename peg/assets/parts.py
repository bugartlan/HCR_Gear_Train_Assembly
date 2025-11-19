import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .. import TASK_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class PegCfg:
    usd_path: str = ""
    diameter: float = 0.0
    effective_length: float = 0.0
    total_length: float = 0.0


@configclass
class HoleCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0


@configclass
class Peg8mm(PegCfg):
    usd_path: str = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter: float = 0.007986
    effective_length: float = 0.050
    total_length: float = 0.050


@configclass
class Hole8mm(HoleCfg):
    usd_path: str = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter: float = 0.0081
    height: float = 0.025


@configclass
class CustomPeg(PegCfg):
    usd_path: str = f"{TASK_DIR}/assets/USD/Peg_v1.usd"
    diameter: float = 0.0125
    effective_length: float = 0.016
    total_length: float = 0.046


@configclass
class CustomHole(HoleCfg):
    usd_path: str = f"{TASK_DIR}/assets/USD/Hole.usd"
    diameter: float = 0.013
    height: float = 0.016


CUSTOM_PEG = CustomPeg()
CUSTOM_HOLE = CustomHole()

factory_peg_8mm = ArticulationCfg(
    prim_path="/World/envs/env_.*/Peg",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_peg_8mm.usd",
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

factory_hole_8mm = ArticulationCfg(
    prim_path="/World/envs/env_.*/Hole",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_hole_8mm.usd",
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
        pos=(0.3, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

custom_peg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Peg",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CUSTOM_PEG.usd_path,
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

custom_hole = ArticulationCfg(
    prim_path="/World/envs/env_.*/Hole",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CUSTOM_HOLE.usd_path,
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
        pos=(0.3, 0.0, CUSTOM_HOLE.height),
        rot=(0.0, 1.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
