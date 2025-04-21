# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, wrap_to_pi, matrix_from_euler

from matplotlib import pyplot as plt
from collections import deque

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    # observation_space = 12+1+4
    observation_space = (
        3 +  # linear velocity
        3 +  # angular velocity
        3 +  # relative desired position
        9 +  # attitude matrix
        4 +  # last actions
        1    # absolute height
    )
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.01
    ang_vel_reward_scale = -0.001
    approaching_goal_reward_scale = 0.0
    convergence_goal_reward_scale = 10.0
    yaw_reward_scale = 5.0
    new_goal_reward_scale = 0.0

    cmd_smoothness_reward_scale = -1
    cmd_body_rates_reward_scale = -0.1
    death_cost = -10.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.last_yaw = 0.0
        self.n_laps = torch.zeros(self.num_envs, device=self.device)
        self.prob_change = 0.05
        self.proximity_threshold = 0.1

        # Get mode
        if self.num_envs > 10:
            self.is_train = True
        else:
            self.is_train = False
            self.change_setpoint = True
            if self.change_setpoint:
                cfg.episode_length_s = 100.0
            else:
                cfg.episode_length_s = 5.0
            self.max_len_deque = 1000
            self.roll_history = deque(maxlen=self.max_len_deque)
            self.pitch_history = deque(maxlen=self.max_len_deque)
            self.yaw_history = deque(maxlen=self.max_len_deque)
            self.actions_history = deque(maxlen=self.max_len_deque)
            self.n_steps = 0
            self.rpy_fig, self.rpy_axes = plt.subplots(4, 1, figsize=(10, 8))
            self.roll_line, = self.rpy_axes[0].plot([], [], 'r', label="Roll")
            self.pitch_line, = self.rpy_axes[1].plot([], [], 'g', label="Pitch")
            self.yaw_line, = self.rpy_axes[2].plot([], [], 'b', label="Yaw")
            self.actions_lines = [self.rpy_axes[3].plot([], [], label=f"Motor {i+1}")[0] for i in range(cfg.action_space)]

            # Configure subplots
            for ax, title in zip(self.rpy_axes, ["Roll History", "Pitch History", "Yaw History", "Actions History"]):
                ax.set_title(title)
                ax.set_xlabel("Time Step")
                if any(angle in title for angle in ["Roll", "Pitch", "Yaw"]):
                    ax.set_ylabel("Angle (°)")
                elif title == "Actions History":
                    ax.set_ylabel("Action")
                ax.legend(loc="upper left")
                ax.grid(True)

            plt.tight_layout()
            plt.ion()  # interactive mode

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "arrpoaching_goal",
                "convergence_to_goal",
                "yaw",
                "cmd",
                "new_goal",
            ]
        }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w)

        quat_w = self._robot.data.root_quat_w
        rpy = euler_xyz_from_quat(quat_w)
        attitude_matrix = matrix_from_euler(
            torch.stack([
                torch.zeros_like(rpy[0]), 
                torch.zeros_like(rpy[1]), 
                rpy[2]
            ], dim=-1),
            convention="XYZ"
        )
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.last_yaw
        self.n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.unwrapped_yaw = yaw_w + 2 * np.pi * self.n_laps
        self.last_yaw = yaw_w

    
        obs = torch.cat(
            [
                self._robot.data.root_link_state_w[:, 2].unsqueeze(1),  # absolute height
                desired_pos_b,                                          # relative desired position
                attitude_matrix.view(attitude_matrix.shape[0], -1),           # attitude matrix
                self._robot.data.root_com_lin_vel_b,                    # linear velocity
                self._robot.data.root_com_ang_vel_b,                    # angular velocity
                self.last_actions,                                  # last actions
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        if not self.is_train:
            # RPY plots
            roll_w = wrap_to_pi(rpy[0])
            pitch_w = wrap_to_pi(rpy[1])

            self.roll_history.append(roll_w * 180.0 / np.pi)
            self.pitch_history.append(pitch_w * 180.0 / np.pi)
            self.yaw_history.append(self.unwrapped_yaw * 180.0 / np.pi)
            self.actions_history.append(self._actions.squeeze(0).cpu().numpy())

            self.n_steps += 1
            if self.n_steps >= self.max_len_deque:
                steps = np.arange(self.n_steps - self.max_len_deque, self.n_steps)
            else:
                steps = np.arange(self.n_steps)

            self.roll_line.set_data(steps, self.roll_history)
            self.pitch_line.set_data(steps, self.pitch_history)
            self.yaw_line.set_data(steps, self.yaw_history)

            for i in range(self.cfg.action_space):
                self.actions_lines[i].set_data(steps, np.array(self.actions_history)[:, i])

            for ax in self.rpy_axes:
                ax.relim()
                ax.autoscale_view()

            plt.draw()
            plt.pause(0.001)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        approaching = (self.last_distance_to_goal - distance_to_goal)
        convergence = (1 - torch.tanh(distance_to_goal / 0.8))
        # convergence = 0.5 * (1 - torch.tanh(distance_to_goal / 0.01 - 3))

        yaw_w_mapped = torch.exp(-10.0 * torch.abs(self.unwrapped_yaw))

        cmd_smoothness = torch.sum(torch.square(self._actions - self.last_actions), dim=1)
        cmd_body_rates_smoothness = torch.sum(torch.square(self._actions[:, 1:]), dim=1)

        close_to_goal = (distance_to_goal < self.proximity_threshold).to(self.device)
        change_setpoint = (torch.rand(self.num_envs, device=self.device) < self.prob_change)
        new_point = torch.logical_and(close_to_goal, change_setpoint)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,

            "arrpoaching_goal": approaching * self.cfg.approaching_goal_reward_scale * self.step_dt,
            "convergence_to_goal": convergence * self.cfg.convergence_goal_reward_scale * self.step_dt,

            "yaw": yaw_w_mapped * self.cfg.yaw_reward_scale * self.step_dt,

            "cmd": cmd_smoothness * self.cfg.cmd_smoothness_reward_scale * self.step_dt + \
                   cmd_body_rates_smoothness * self.cfg.cmd_body_rates_reward_scale * self.step_dt,
            
            "new_goal": new_point * self.cfg.new_goal_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.cfg.death_cost, reward)

        self.last_actions = self._actions.clone()
        self.last_distance_to_goal = distance_to_goal.clone()

        if True: #self.is_train:
            if torch.any(new_point):
                # Update goal position for environments that are close to the goal
                env_ids = torch.where(close_to_goal)[0]
                self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
                self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        elif self.change_setpoint:
            # Check if drone is within the proximity threshold
            close_to_goal = distance_to_goal < self.proximity_threshold
            
            if torch.any(close_to_goal):
                # Update goal position for environments that are close to the goal
                env_ids = torch.where(close_to_goal)[0]
                self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
                self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        default_root_state = self._robot.data.default_root_state[:, :2] + self._terrain.env_origins[:, :2]
        drone_pos = self._robot.data.root_link_pos_w[:, :2]

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #cond_h_min = torch.logical_and(self._robot.data.root_link_pos_w[:, 2] < 0.1, \
        #                               torch.sum(torch.square(drone_pos - default_root_state), dim=1) > 0.1)
        died = torch.logical_or(self._robot.data.root_link_pos_w[:, 2] < 0.1, self._robot.data.root_link_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.n_laps[env_ids] = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)