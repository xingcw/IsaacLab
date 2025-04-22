from collections import defaultdict
from typing import Any, NamedTuple
import torch
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST


class ExtendedTimeStepWrapper(gym.Env):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = torch.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_space

	def action_spec(self):
		return self._env.action_space

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper(object):
	def __init__(self, env, domain, task, action_repeat, modality):
		try: # pixels
			obs_shp = env.observation_spec().shape
			assert modality == 'pixels'
		except: # state
			obs_shp = []
			for v in env.observation_spec().values():
				try:
					shp = np.prod(v.shape)
				except:
					shp = 1
				obs_shp.append(shp)
			obs_shp = (np.sum(obs_shp, dtype=np.int32),)
			assert modality != 'pixels'
		act_shp = env.action_spec().shape
		obs_dtype = np.float32 if modality != 'pixels' else np.uint8
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf if modality != 'pixels' else env.observation_spec().minimum,
				dtype=obs_dtype),
			high=np.full(
				obs_shp,
				np.inf if modality != 'pixels' else env.observation_spec().maximum,
				dtype=obs_dtype),
			shape=obs_shp,
			dtype=obs_dtype,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			shape=act_shp,
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.ep_len = 1000//action_repeat
		self.modality = modality
		self.t = 0
	
	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		if self.modality != 'pixels':
			return np.concatenate([v.flatten() for v in obs.values()])
		return obs

	def reset(self):
		self.t = 0
		return self._obs_to_array(self.env.reset().observation)
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


class DefaultDictWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return obs, reward, done, defaultdict(float, info)


def wrap_env(env: gym.Env):

	env = ExtendedTimeStepWrapper(env)
	env = TimeStepToGymWrapper(env, domain, task, cfg.action_repeat, cfg.modality)
	env = DefaultDictWrapper(env)

	return env
