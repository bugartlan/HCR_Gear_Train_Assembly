import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .. import TASK_DIR

UR3e_ROBOTIQ_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TASK_DIR}/assets/USD/HCR_ClassRobot.usd",
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
            effort_limit_sim=50.0,
            velocity_limit_sim=1.0,
            stiffness=5000.0,
            damping=200.0,
            friction=0.0,
            armature=0.05,
        ),
    },
)

UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG = UR3e_ROBOTIQ_GRIPPER_CFG.copy()
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["shoulder"].stiffness = 1500.0
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["shoulder"].damping = 200.0
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["elbow"].stiffness = 800.0
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["elbow"].damping = 100.0
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["wrist"].stiffness = 300.0
UR3e_ROBOTIQ_GRIPPER_HIGH_PD_CFG.actuators["wrist"].damping = 80.0

ROBOTIQ_GRIPPER_CENTER_OFFSET = (
    0.14  # Offset from gripper base to gripper center along gripper approach axis
)
