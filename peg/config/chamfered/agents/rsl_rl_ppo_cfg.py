# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg,
                                RslRlPpoActorCriticRecurrentCfg,
                                RslRlPpoAlgorithmCfg)


@configclass
class PegInsertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    experiment_name = "ur3e_chamfered_peg_insert"
    wandb_project = "UR3e-Chamfered-Peg-Insertion"
    run_name = ""
    logger = "wandb"
    store_code_state = True
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class PegInsertPPORunnerRecurrentCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "ur3e_chamfered_peg_insert"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
    )
