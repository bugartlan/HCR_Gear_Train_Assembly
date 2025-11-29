# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Factory task configs for gear assembly.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments - GearMesh registration commented out to avoid conflicts
##

# gym.register(
#     id="Isaac-Assembly-GearMesh-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "isaaclab_tasks.manager_based.assembly.gear.config.factory.joint_env_cfg:FactoryGearMeshEnvCfg",
#         "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.assembly.gear.config.factory.agents.rsl_rl_ppo_cfg:GearMeshPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Assembly-GearMesh-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "isaaclab_tasks.manager_based.assembly.gear.config.factory.joint_env_cfg:FactoryGearMeshEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.assembly.gear.config.factory.agents.rsl_rl_ppo_cfg:GearMeshPPORunnerCfg",
#     },
# )
