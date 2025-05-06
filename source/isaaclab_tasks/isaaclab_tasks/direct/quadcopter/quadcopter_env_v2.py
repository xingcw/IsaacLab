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
# from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

from matplotlib import pyplot as plt
from collections import deque

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

# GOAL_MARKER_CFG = VisualizationMarkersCfg(
#     markers={
#         # "sphere": sim_utils.UsdFileCfg(
#         #     usd_path="/home/lorenzo/Desktop/goal.usdc",
#         #     scale=(0.5, 0.5, 0.5),
#         # ),
#         "sphere": sim_utils.SphereCfg(
#             radius=0.01,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#         ),
#     }
# )

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
    episode_length_s = 20.0             # episode_length = episode_length_s / dt / decimation
    action_space = 4
    observation_space = (
        3 +     # linear velocity
        3 +     # angular velocity
        3 +     # relative desired position
        9 +     # attitude matrix
        4 +     # last actions
        1       # absolute height
    )
    state_space = 0
    debug_vis = True

    sim_rate_hz = 100
    policy_rate_hz = 50
    pd_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
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
    moment_scale = 0.01

    # Initialize variables
    eps_tanh = 1e-3
    beta = 1.0         # 1.0 for no smoothing, 0.0 for no update
    min_altitude = 0.1
    max_altitude = 2.0
    max_time_on_ground = 0.0

    # values to randomly initialize the drone
    min_roll_pitch = -torch.pi / 4.0
    max_roll_pitch =  torch.pi / 4.0
    min_yaw = -torch.pi
    max_yaw =  torch.pi
    min_lin_vel_xy = -0.2
    max_lin_vel_xy =  0.2
    min_lin_vel_z = -0.1
    max_lin_vel_z =  0.1
    min_ang_vel = -0.1
    max_ang_vel =  0.1

    # motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    # CTBR Parameters
    kp_omega = 1.0      # default taken from RotorPy, needs to be checked on hardware. 
    kd_omega = 0.0      # default taken from RotorPy, needs to be checked on hardware.
    body_rate_scale_xy = 7.0
    body_rate_scale_z = 3.0

    # Parameters from train.py or play.py
    use_simple_model = False
    thrust_to_weight = 1.9 if use_simple_model else 1.8
    prob_change = 0.5
    proximity_threshold = 0.1
    velocity_threshold = 100.0
    wait_time_s = 0.5
    max_time_no_approach = 6.0
    max_motor_noise_std = 50.0
    curriculum_start = 1000
    curriculum_end = 4000
    
    # reward scales
    lin_vel_reward_scale = -1.0
    ang_vel_reward_scale = -1.0
    approaching_goal_reward_scale = 1000.0
    convergence_goal_reward_scale = 1000.0
    yaw_reward_scale = 10.0
    new_goal_reward_scale = 0.0
    
    cmd_smoothness_reward_scale = -1.0
    cmd_body_rates_reward_scale = -1.0
    death_cost = -200.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.iteration = 0
        self.motor_noise_std = 0.0

        # Get train/test mode
        self.is_train = True

        # Initialize tensors
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_yaw = torch.zeros(self.num_envs, device=self.device)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._last_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._n_laps = torch.zeros(self.num_envs, device=self.device)
        self._previous_t = torch.zeros(self.num_envs, device=self.device)
        self._previous_t_change_point = torch.zeros(self.num_envs, device=self.device)
        self._episode_length_buf_zero = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.closest_distance_to_goal = -torch.ones(self.num_envs, device=self.device)
        self.first_approach = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Things necessary for motor dynamics
        r2o2 = np.sqrt(2.0) / 2.0
        self._rotor_positions = torch.cat(
            [
                self.cfg.arm_length * torch.FloatTensor([[ r2o2,  r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[ r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2,  r2o2, 0]]),
            ],
            dim=0).to(self.device)
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k = self.cfg.k_m / self.cfg.k_eta

        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                    ], 
                    dim=1,
                ).to(self.device),
                self.k * self._rotor_directions.view(1, -1),
            ],
            dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_to_goal",
                "lin_vel_reward",
                "lin_vel_penalty",
                "hover_reward",
                "alignment_reward",
                "smoothness",
                "height_reward",
                "low_alt_penalty"
            ]
        }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def update_iteration(self, iter):
        self.iteration = iter

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

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        return motor_speeds_des

    def _get_moment_from_ctbr(self, actions):
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, :2] = self.cfg.body_rate_scale_xy * actions[:, 1:3]
        omega_des[:, 2] = self.cfg.body_rate_scale_z * actions[:, 3]

        omega_err = self._robot.data.root_ang_vel_b - omega_des         # FIXME
        omega_dot_err = (omega_err - self._previous_omega_err) / self.cfg.pd_loop_rate_hz
        self._previous_omega_err = omega_err
        omega_dot = self.cfg.kp_omega * omega_err + self.cfg.kd_omega * omega_dot_err

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)    # actions come directly from the NN

        if self.cfg.use_simple_model:
            self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
            self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        else:
            self._actions = self.cfg.beta * self._actions + (1 - self.cfg.beta) * self._previous_action
            self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * self._robot_weight * self.cfg.thrust_to_weight

            # compute wrench from desired body rates and current body rates using PD controller
            # self._wrench_des[:,1:] = self._get_moment_from_ctbr(self._actions)          ##
            # self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)       ##
            self.pd_loop_counter = 0

    def _apply_action(self):
        if not self.cfg.use_simple_model:
            # Update PD loop at a lower rate
            if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0:
                self._wrench_des[:, 1:] = self._get_moment_from_ctbr(self._actions)     ##
                self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)   ##

            self.pd_loop_counter += 1

            motor_accel = (self._motor_speeds_des - self._motor_speeds) / self.cfg.tau_m
            self._motor_speeds += motor_accel * self.physics_dt

            # add noise to motor speeds
            if not self.is_train:
                self.motor_noise_std = self.cfg.max_motor_noise_std
            elif self.iteration <= self.cfg.curriculum_start:
                self.motor_noise_std = 0
            else:
                self.motor_noise_std = (self.iteration - self.cfg.curriculum_start) / (self.cfg.curriculum_end - self.cfg.curriculum_start) * self.cfg.max_motor_noise_std
                self.motor_noise_std = min(self.motor_noise_std, self.cfg.max_motor_noise_std)

            self._motor_speeds += torch.randn_like(self._motor_speeds) * self.motor_noise_std

            self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
            # self._motor_speeds = self._motor_speeds_des # assume no delay to simplify the simulation
            motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
            wrench = torch.matmul(motor_forces, self.f_to_TM.t())

            self._thrust[:, 0, 2] = wrench[:, 0]
            self._moment[:, 0, :] = wrench[:, 1:]

        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)


    def _get_rewards(self) -> torch.Tensor:
        pos = self._robot.data.root_link_pos_w
        vel = self._robot.data.root_com_lin_vel_b
        act = self._actions
        prev_act = self._previous_action

        dist = torch.norm(self._desired_pos_w - pos, dim=1)
        speed = torch.norm(vel, dim=1)
        smooth = torch.sum((act - prev_act)**2, dim=1)

        # Hover shaping
        z = pos[:, 2]
        low_alt_penalty = torch.where(z < 0.3, -2.0 * (0.3 - z), torch.zeros_like(z))
        z_bonus = 1.0 * torch.exp(-((z - 1.0) / 0.3)**2)

        # Velocity shaping
        sigma = 0.3
        r = 0.1
        vel_pen = -0.5 * speed * torch.exp(-dist / sigma)
        hover_bonus = 1.0 * torch.exp(-(dist / r) ** 2)

        # Goal alignment
        goal_dir = torch.nn.functional.normalize(self._desired_pos_w - pos, dim=1)
        align_reward = 0.5 * torch.sum(vel * goal_dir, dim=1)

        rewards = {
            "distance_to_goal": -1.0 * dist,
            "lin_vel_reward": 0.5 * torch.tanh(2 * speed),
            "lin_vel_penalty": vel_pen,
            "hover_reward": hover_bonus,
            "alignment_reward": 0.1 * align_reward,
            "smoothness": -0.05 * smooth,
            "height_reward": z_bonus,
            "low_alt_penalty": low_alt_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    
    # _get_observations is executed after _get_rewards
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(self._robot.data.root_link_state_w[:, :3],
                                                     self._robot.data.root_link_state_w[:, 3:7],
                                                     self._desired_pos_w)

        quat_w = self._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self._robot.data.root_link_state_w[:, 2].unsqueeze(1),  # absolute height
                desired_pos_b,                                          # relative desired position
                attitude_mat.view(attitude_mat.shape[0], -1),           # attitude matrix
                self._robot.data.root_com_lin_vel_b,                    # linear velocity
                self._robot.data.root_com_ang_vel_b,                    # angular velocity
                self._previous_action,                                  # last actions
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self._previous_yaw
        self._n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self._n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.unwrapped_yaw = yaw_w + 2 * np.pi * self._n_laps
        self._previous_yaw = yaw_w

        self._previous_action = self._actions.clone()

        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        episode_time = (self.episode_length_buf - self._episode_length_buf_zero) * self.cfg.sim.dt * self.cfg.decimation
        cond_h_min_time = torch.logical_and(
            self._robot.data.root_link_pos_w[:, 2] < self.cfg.min_altitude, \
            episode_time > self.cfg.max_time_on_ground
        )
        cond_max_h = self._robot.data.root_link_pos_w[:, 2] > self.cfg.max_altitude
        cond_not_converged = self.first_approach & ((episode_time - self._previous_t_change_point) > self.cfg.max_time_no_approach)
        died = cond_h_min_time | cond_max_h | cond_not_converged

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.is_train:
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
            extras["Metrics/motor_noise_std"] = self.motor_noise_std
            self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._episode_length_buf_zero = self.episode_length_buf.clone()

        self._actions[env_ids] = 0.0

        # Sample new desired position
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Reset joints state
        joint_pos = self._robot.data.default_joint_pos[env_ids]     # not important
        joint_vel = self._robot.data.default_joint_vel[env_ids]     #
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset robots state
        # [pos, quat, lin_vel, ang_vel] in local environment frame. Shape is (num_instances, 13)
        default_root_state = self._robot.data.default_root_state[env_ids]   
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self._n_laps[env_ids] = 0
        self._previous_t[env_ids] = 0.0
        self._previous_t_change_point[env_ids] = 0.0
        self.closest_distance_to_goal[env_ids] = -1.0
        self.first_approach[env_ids] = True

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
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