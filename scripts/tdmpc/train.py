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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cudnn.benchmark = True


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
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: 
				video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: 
			video.save(env_step)
	return np.nanmean(episode_rewards)


@hydra_task_config(args_cli.task, "tdmpc_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
	"""Train with TD-MPC agent."""
	assert torch.cuda.is_available()
	
	# Override configurations with non-hydra CLI arguments
	env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
	
	# Set seeds
	seed = set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg["seed"])
	env_cfg.seed = seed
	
	# Specify directory for logging experiments
	log_root_path = os.path.join("logs", "tdmpc", agent_cfg["task"])
	log_root_path = os.path.abspath(log_root_path)
	print(f"[INFO] Logging experiment in directory: {log_root_path}")
	
	# Specify directory for logging runs
	log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	work_dir = Path(os.path.join(log_root_path, log_dir))
	
	# Create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
	
	# Convert to single-agent instance if required
	if isinstance(env.unwrapped, DirectMARLEnv):
		env = multi_agent_to_single_agent(env)
	
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
	agent = TDMPC(agent_cfg)
	buffer = ReplayBuffer(agent_cfg)
	
	# Dump configurations
	dump_yaml(os.path.join(work_dir, "params", "env.yaml"), env_cfg)
	dump_yaml(os.path.join(work_dir, "params", "agent.yaml"), agent_cfg)
	dump_pickle(os.path.join(work_dir, "params", "env.pkl"), env_cfg)
	dump_pickle(os.path.join(work_dir, "params", "agent.pkl"), agent_cfg)
	
	# Run training
	L = logger.Logger(work_dir, agent_cfg)
	episode_idx, start_time = 0, time.time()
	
	for step in range(0, agent_cfg.train_steps + agent_cfg.episode_length, agent_cfg.episode_length):
		# Collect trajectory
		obs = env.reset()
		episode = Episode(agent_cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		assert len(episode) == agent_cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= agent_cfg.seed_steps:
			num_updates = agent_cfg.seed_steps if step == agent_cfg.seed_steps else agent_cfg.episode_length
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
			common_metrics['episode_reward'] = evaluate(env, agent, agent_cfg.eval_episodes, step, env_step, L.video)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')
	
	# Close the simulator
	env.close()


if __name__ == '__main__':
	# Run the main function
	main()
	# Close sim app
	simulation_app.close()
