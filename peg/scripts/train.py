import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=12_000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--estimator", action="store_true", default=False, help="Learn estimator during training.")
parser.add_argument("--probe", action="store_true", default=False, help="Train probes.")
parser.add_argument("--wandb_run", type=str, default="", help="Run from WandB.")
parser.add_argument("--wandb_model", type=str, default="", help="Model from WandB.")
parser.add_argument("--wandb", action="store_true", default=False, help="Select WandB run.")
parser.add_argument("--server", action="store_true", default=False, help="Train on a headless server.")
parser.add_argument("--distributed", action="store_true", default=False, help="Train with multiple GPUs.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.video = True
if args_cli.video and args_cli.server:
    args_cli.log_videos_async = True
    args_cli.video = False
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
import shutil
import torch
from datetime import datetime

from robot_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

# import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
    else:
        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # type: ignore

    env_cfg_dict = env_cfg.to_dict()  # type: ignore
    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.wandb:
        run_path = cli_args.get_wandb_run_name(args_cli.wandb_run, args_cli.server)
        model_name = cli_args.get_wandb_model_name(args_cli.wandb_model, args_cli.server)
        try:
            resume_path, env_cfg_dict = cli_args.pull_policy_from_wandb(log_root_path, run_path, model_name)
            print("\033[92m\n[INFO] added policy to load\033[0m")
            model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
            model_dir = os.path.join(log_dir, run_path.split("/")[-1])
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
        except Exception:
            raise ValueError(
                "\n\033[91m[ERROR] Unable to download from Weights and Biases, is the path and filename correct?\033[0m"
            )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # type: ignore

    # create runner from rsl-rl
    if args_cli.probe:
        Runner = ProbeRunner
    elif agent_cfg.class_name == "OnPolicyRunner":
        Runner = OnPolicyRunner
    elif agent_cfg.class_name == "DistillationRunner":
        Runner = DistillationRunner
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner = Runner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)  # type: ignore
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or args_cli.wandb or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if isinstance(runner, ProbeRunner):
            runner.load_actor(resume_path)
        else:
            runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg_dict)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
