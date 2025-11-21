import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .. import TASK_DIR

UR3e_ROBOTIQ_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TASK_DIR}/assets/USD/ur3e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": -math.pi / 2,
            "elbow_joint": math.pi / 2,
            "wrist_1_joint": -math.pi / 2,
            "wrist_2_joint": -math.pi / 2,
            "wrist_3_joint": 0.0,
            "Slider_.*": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=330.0,
            damping=36.0,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=150.0,
            damping=17.5,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=54.0,
            damping=14.5,
            friction=0.0,
            armature=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Slider_.*"],
            stiffness=1e6,
            damping=2e4,
            friction=0.0,
            armature=0.0,
        ),
    },
)

ROBOTIQ_GRIPPER_CENTER_OFFSET = (
    0.13  # Offset from gripper base to gripper center along gripper approach axis
)
