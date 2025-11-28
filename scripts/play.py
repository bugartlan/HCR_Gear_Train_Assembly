import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--wandb_run", type=str, default="", help="Run from WandB.")
parser.add_argument("--wandb_model", type=str, default="", help="Model from WandB.")
parser.add_argument("--wandb", action="store_true", default=False, help="Select WandB run.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--convert", action="store_true", default=False, help="Convert to JIT & onnx.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# args_cli.headless = True
# args_cli.video = True
args_cli.wandb = bool(args_cli.wandb or (len(args_cli.wandb_run) and len(args_cli.wandb_model)))

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# suppress logs
if not hasattr(args_cli, "kit_args"):
    args_cli.kit_args = ""
args_cli.kit_args += " --/log/level=error"
args_cli.kit_args += " --/log/fileLogLevel=error"
args_cli.kit_args += " --/log/outputStreamLevel=error"

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ISAAC_PREFIXES = ("--/log/", "--/app/", "--/renderer=", "--/physics/")
hydra_args = [arg for arg in hydra_args if not arg.startswith(ISAAC_PREFIXES)]
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import shutil
import torch
from datetime import datetime

from robot_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # task_name = args_cli.task.split(":")[-1]
    # train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    ext_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_root_path = os.path.join(ext_path, "logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # specify directory for logging runs: {time-stamp}_{run_name}
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif args_cli.wandb:
        run_path = cli_args.get_wandb_run_name(args_cli.wandb_run)
        model_name = cli_args.get_wandb_model_name(args_cli.wandb_model)
        try:
            resume_path, _ = cli_args.pull_policy_from_wandb(log_root_path, run_path, model_name)
            print("\033[92m\n[INFO] added policy to load\033[0m")
            model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
            model_dir = os.path.join(log_dir, run_path.split("/")[-1])
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
        except Exception:
            raise ValueError(
                "\n\033[91m[ERROR] Unable to download from Weights and Biases, is the path and filename correct?\033[0m"
            )
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        Runner = OnPolicyRunner
    elif agent_cfg.ClassName == "DistillationRunner":
        Runner = DistillationRunner
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner = Runner(env, agent_cfg.to_dict(), log_dir=log_dir if args_cli.resume else None, device=agent_cfg.device)
    if agent_cfg.resume or args_cli.wandb:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.convert:
        policy_nn = runner.alg.policy

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy_jit.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()
