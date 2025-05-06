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
parser.add_argument("--video", "-v", action="store_true", default=False, help="Save videos during training.")
parser.add_argument("--num_envs", "-ne", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", "-t", type=str, default="Isaac-Quadcopter-Direct-v1", help="Name of the task.")
parser.add_argument("--seed", "-s", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--exp_name", "-en", type=str, default="default", help="Name of the experiment.")
parser.add_argument("--use_wandb", "-uw", action="store_true", default=False, help="Use wandb for logging.")
parser.add_argument("--save_model", "-sm", action="store_true", default=False, help="Save model.")
parser.add_argument("--update_multiplier", "-um", type=float, default=1.0, help="Update multiplier.")
parser.add_argument("--finetune_dynamics", "-fd", action="store_true", default=False, help="Finetune dynamics.")
parser.add_argument("--ckpt_path", "-cp", type=str, default=None, help="Path to the checkpoint file.")
parser.add_argument("--eval_only", action="store_true", help="Only run evaluation using the loaded model.")
parser.add_argument("--follow_robot", type=int, default=1, help="Which environment index to follow in viewer.")

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
from isaaclab.utils.io import dump_pickle, dump_yaml

from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
from cfg import parse_cfg
from omegaconf import OmegaConf, DictConfig
import logger

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cudnn.benchmark = True


class EnvWrapper(gym.Env):
	def __init__(self, env: gym.Env):
		self.env = env
		self.t = 0

	def reset(self):
		obs, info = self.env.reset()
		self.t = 0
		return obs['policy'].squeeze(0)
	
	def step(self, action):
		self.t += 1
		action = action.unsqueeze(0)
		obs, reward, terminated, time_outs, info = self.env.step(action)
		done = self.t >= self.max_episode_length or time_outs
		return obs['policy'].squeeze(0), reward.item(), done, info # type: ignore
	
	def __str__(self):
		return f"<{type(self).__name__}{self.env}>"

	def __repr__(self):
		return str(self)
		
	@property
	def cfg(self) -> object:
		"""Returns the configuration class instance of the environment."""
		return self.unwrapped.cfg

	@property
	def render_mode(self) -> str | None:
		"""Returns the :attr:`Env` :attr:`render_mode`."""
		return self.env.render_mode

	@property
	def observation_space(self) -> gym.Space:
		"""Returns the :attr:`Env` :attr:`observation_space`."""
		return self.env.observation_space

	@property
	def action_space(self) -> gym.Space:
		"""Returns the :attr:`Env` :attr:`action_space`."""
		return self.env.action_space

	@classmethod
	def class_name(cls) -> str:
		"""Returns the class name of the wrapper."""
		return cls.__name__

	@property
	def unwrapped(self) -> DirectRLEnv:
		"""Returns the base environment of the wrapper.

		This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
		"""
		return self.env.unwrapped # type: ignore
	
	@property
	def step_id(self):
		return self.env.step_id # type: ignore
	
	@property
	def max_episode_length(self):
		return self.env.unwrapped.max_episode_length # type: ignore
	
	def render(self):
		return self.env.unwrapped.render() # type: ignore
	
	def __getattr__(self, name):
		return getattr(self.env.unwrapped, name)


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
	return np.nanmean(episode_rewards), np.nanstd(episode_rewards)


@hydra_task_config(args_cli.task, "tdmpc_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: DictConfig):
	"""Train with TD-MPC agent."""
	assert torch.cuda.is_available()
	
	# Override configurations with non-hydra CLI arguments
	env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	agent_cfg = OmegaConf.create(agent_cfg)
	
	# Set seeds
	seed = set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg.seed)
	env_cfg.seed = seed
	agent_cfg.seed = seed

	# Set experiment name
	agent_cfg.exp_name = args_cli.exp_name
	agent_cfg.save_model = args_cli.save_model
	
	# Specify directory for logging experiments
	log_root_path = os.path.join("logs", "tdmpc", agent_cfg.task)
	log_root_path = os.path.abspath(log_root_path)
	print(f"[INFO] Logging experiment in directory: {log_root_path}")
	
	# Specify directory for logging runs
	log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	work_dir = Path(os.path.join(log_root_path, log_dir))
	
	# # # Set viewer tracking options
	env_cfg.viewer.origin_type = "asset_root"
	env_cfg.viewer.env_index = 0  # Always use first environment for tracking
	env_cfg.viewer.asset_name = "robot"

	# Create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

	agent_cfg.obs_shape = (int(env_cfg.observation_space), ) # type: ignore
	agent_cfg.action_shape = (int(env_cfg.action_space), ) # type: ignore
	agent_cfg.action_dim = int(env_cfg.action_space) # type: ignore
	print("obs_shape: ", agent_cfg.obs_shape)
	print("action_shape: ", agent_cfg.action_shape)
	print("action_dim: ", agent_cfg.action_dim)
	agent_cfg.device = "cuda"
	agent_cfg.task_title = agent_cfg.task.replace('-', ' ').title()

	# setup wandb
	agent_cfg.use_wandb = args_cli.use_wandb
	# agent_cfg.wandb_project = "isaaclab"
	# agent_cfg.wandb_entity = "chxing-university-of-pennsylvania"
	
	# setup video recording
	agent_cfg.save_video = args_cli.video

	# update multiplier
	agent_cfg.update_multiplier = args_cli.update_multiplier

	if args_cli.ckpt_path:
		agent_cfg.lr = 1e-5
		
	# Initialize agent and buffer
	agent = TDMPC(agent_cfg)
	buffer = ReplayBuffer(agent_cfg)
	
	if args_cli.ckpt_path:
		agent.load(args_cli.ckpt_path, dynamics_only=args_cli.finetune_dynamics)
		print(f"Loaded model from {args_cli.ckpt_path}")
	
	# Dump configurations
	dump_yaml(os.path.join(work_dir, "params", "env.yaml"), env_cfg)
	dump_yaml(os.path.join(work_dir, "params", "agent.yaml"), OmegaConf.to_yaml(agent_cfg))
	
	# Run training
	L = logger.Logger(work_dir, agent_cfg)
	episode_idx, start_time = 0, time.time()

	env = EnvWrapper(env)

	if args_cli.eval_only:
		# Run evaluation only
		print("Running evaluation only mode...")
		reward_mean, reward_std = evaluate(env, agent, agent_cfg.eval_episodes, step=0, env_step=0, video=L.video)
		print(f"Evaluation reward: {reward_mean:.2f} ± {reward_std:.2f}")
	else:		
		for step in range(0, agent_cfg.train_steps + agent_cfg.episode_length, agent_cfg.episode_length):
			# Collect trajectory
			obs = env.reset()
			episode = Episode(agent_cfg, obs)
			while not episode.done:
				action = agent.plan(obs, step=step, t0=episode.first)
				obs, reward, done, _ = env.step(action) # type: ignore
				episode += (obs, action, reward, done)
			assert len(episode) == agent_cfg.episode_length
			buffer += episode

			# Update model
			train_metrics = {}
			if step >= agent_cfg.seed_steps:
				num_updates = agent_cfg.seed_steps if step == agent_cfg.seed_steps else agent_cfg.episode_length
				num_updates = int(num_updates * args_cli.update_multiplier)
				for i in range(num_updates):
					train_metrics.update(agent.update(buffer, step+i))

			# Log training episode
			episode_idx += 1
			env_step = int(step * agent_cfg.action_repeat)
			common_metrics = {
				'episode': episode_idx,
				'step': step,
				'env_step': env_step,
				'total_time': time.time() - start_time,
				'episode_reward': episode.cumulative_reward
			}
			train_metrics.update(common_metrics)
			L.log(train_metrics, category='train')
			
			# Evaluate agent periodically
			if env_step % agent_cfg.eval_freq == 0:
				common_metrics['episode_reward'], common_metrics['episode_reward_std'] = \
					evaluate(env, agent, agent_cfg.eval_episodes, step, env_step, L.video)
				L.log(common_metrics, category='eval', agent=agent)

		L.finish(agent)
		print('Training completed successfully')
		
	# Close the simulator
	env.close()


if __name__ == '__main__':

	# Run the main function
	main() # type: ignore
	# Close sim app
	simulation_app.close()
