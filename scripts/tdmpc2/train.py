# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with TD-MPC."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with TD-MPC.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
	args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from isaaclab.envs import (
	DirectMARLEnv,
	DirectMARLEnvCfg,
	DirectRLEnvCfg,
	ManagerBasedRLEnvCfg,
	multi_agent_to_single_agent,
)
from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from omegaconf import OmegaConf, DictConfig

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')

from termcolor import colored
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
from common.parser import parse_cfg

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class EnvWrapper(gym.Env):
	def __init__(self, env: gym.Env):
		self.env = env
		self.t = 0

	def reset(self):
		obs, info = self.env.reset()
		self.t = 0
		return obs.squeeze(0)
	
	def step(self, action):
		self.t += 1
		action = action.unsqueeze(0)
		obs, reward, _, done, info = self.env.step(action)
		return obs.squeeze(0), reward.item(), self.t >= self.max_episode_length, info # type: ignore

	@property
	def action_space(self):
		return self.env.action_space

	@property
	def observation_space(self):
		return self.env.observation_space
	
	@property
	def max_episode_length(self):
		return self.env.unwrapped.max_episode_length # type: ignore
	
	def __getattr__(self, name):
		return getattr(self.env, name)


def set_seed(seed):
	"""Set random seeds for reproducibility."""
	if seed == -1:
		seed = np.random.randint(0, 10000)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	return seed


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: 
			video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			if video: 
				video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: 
			video.save(env_step)
	return np.nanmean(episode_rewards)


@hydra_task_config(args_cli.task, "tdmpc2_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: DictConfig):
	"""Train with TD-MPC agent."""
	assert torch.cuda.is_available()
	
	# Override configurations with non-hydra CLI arguments
	env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	agent_cfg = OmegaConf.create(agent_cfg)
	agent_cfg = parse_cfg(agent_cfg)

	# Set seeds
	seed = set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg.seed)
	env_cfg.seed = seed
	
	# Specify directory for logging experiments
	log_root_path = os.path.join("logs", "tdmpc", agent_cfg.task)
	log_root_path = os.path.abspath(log_root_path)
	print(f"[INFO] Logging experiment in directory: {log_root_path}")
	
	# Specify directory for logging runs
	log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	work_dir = Path(os.path.join(log_root_path, log_dir))
	
	# Create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
	env = EnvWrapper(env)

	agent_cfg.obs_shape = {"state": (int(env_cfg.observation_space), )} # type: ignore
	agent_cfg.action_shape = (int(env_cfg.action_space), ) # type: ignore
	agent_cfg.action_dim = int(env_cfg.action_space) # type: ignore
	print("obs_shape: ", agent_cfg.obs_shape)
	print("action_shape: ", agent_cfg.action_shape)
	print("action_dim: ", agent_cfg.action_dim)
	agent_cfg.device = "cuda"
	agent_cfg.task_title = agent_cfg.task.replace('-', ' ').title()
	
	# Wrap for video recording
	if args_cli.video:
		video_kwargs = {
			"video_folder": os.path.join(work_dir, "videos", "train"),
			"step_trigger": lambda step: step % args_cli.video_interval == 0,
			"video_length": args_cli.video_length,
			"disable_logger": True,
		}
		print("[INFO] Recording videos during training.")
		print_dict(video_kwargs, nesting=4)
		env = gym.wrappers.RecordVideo(env, **video_kwargs)
	
	# Initialize agent and buffer
	agent_cfg.work_dir = work_dir
	agent = TDMPC2(agent_cfg)
	buffer = Buffer(agent_cfg)

	print(colored('Work dir:', 'yellow', attrs=['bold']), agent_cfg.work_dir)
	
	# Dump configurations
	dump_yaml(os.path.join(work_dir, "params", "env.yaml"), env_cfg)
	dump_yaml(os.path.join(work_dir, "params", "agent.yaml"), OmegaConf.to_yaml(agent_cfg))

	trainer_cls = OfflineTrainer if agent_cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=agent_cfg,
		env=env,
		agent=agent,
		buffer=buffer,
		logger=Logger(agent_cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')
	
	# Close the simulator
	env.close()


if __name__ == '__main__':

	# Run the main function
	main() # type: ignore
	# Close sim app
	simulation_app.close()
