import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .. import TASK_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    effective_length: float = 0.0
    total_length: float = 0.0
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
class GearTrainGearBase(FixedAssetCfg):
    usd_path = f"{TASK_DIR}/assets/USD/GearBase.usd"
    height = 0.075
    small_gear_base_offset = [0.0, 0.055, 0.001]
    medium_gear_base_offset = [0.0, 0.03, 0.001]
    large_gear_base_offset = [0.0, -0.015, 0.001]


@configclass
class GearTrainMediumGear(HeldAssetCfg):
    usd_path = f"{TASK_DIR}/assets/USD/Gear_15.usd"
    diameter = 0.02  # Used for gripper width.
    height: float = 0.022
    held_offset: float = 0.005
    mass = 0.012


factory_small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
factory_medium_gear_usd = f"{ASSET_DIR}/factory_gear_medium.usd"
factory_large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"
factory_gear_base_usd = f"{ASSET_DIR}/factory_gear_base.usd"

factory_small_gear_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/SmallGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=factory_small_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

factory_medium_gear_cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/HeldAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=factory_medium_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


factory_large_gear_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/LargeGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=factory_large_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

factory_gear_base_cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/FixedAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=factory_gear_base_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

geartrain_small_gear_usd = f"{TASK_DIR}/assets/USD/spur_gear_am_10.usd"
geartrain_medium_gear_usd = f"{TASK_DIR}/assets/USD/Gear_15.usd"
geartrain_large_gear_usd = f"{TASK_DIR}/assets/USD/spur_gear_am_30.usd"
geartrain_gear_base_usd = f"{TASK_DIR}/assets/USD/GearBase.usd"

geartrain_small_gear_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/SmallGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=geartrain_small_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

geartrain_medium_gear_cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/HeldAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=geartrain_medium_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.4, 0.1), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

geartrain_large_gear_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/LargeGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=geartrain_large_gear_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

geartrain_gear_base_cfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/FixedAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=geartrain_gear_base_usd,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)
