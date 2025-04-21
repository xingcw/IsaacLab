# train.py
import argparse
import os
from pathlib import Path

# Import AppLauncher first
from isaaclab.app import AppLauncher

# Parse args before importing environment-dependent modules
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD-MPC on Isaac Sim')

    # Isaac Lab args
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Direct-v1")
    parser.add_argument("--headless", action="store_true")
    
    # TD‑MPC hyperparameters (override YAML)
    parser.add_argument("--cfg_dir",      type=str, default="cfgs")
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--train_steps",  type=int, default=None)
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--action_repeat",  type=int, default=None)
    parser.add_argument("--eval_freq",      type=int, default=None)
    parser.add_argument("--eval_episodes",  type=int, default=None)
    parser.add_argument("--exp_name",       type=str, default="tdmpc_quad")
    parser.add_argument("--modality",       type=str, default="state")  # Use state modality for direct observations
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--obs_dim",        type=int, default=256)      # Match model dimension
    args = parser.parse_args()

    # Print all arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Launch Isaac Sim first
    app = AppLauncher(args)
    sim_app = app.app

    # Now import modules that depend on Isaac Sim
    import warnings
    warnings.filterwarnings("ignore")
    
    import random
    import time
    import gymnasium as gym
    import torch
    import numpy as np

    # Now it's safe to import these modules
    import isaaclab_tasks
    from isaaclab_tasks.direct.quadcopter.quadcopter_env_v1 import QuadcopterEnvCfg

    # Print all registered environments
    print("Registered environments:")
    for env_name in sorted(gym.envs.registry.keys()):
        if "Isaac-Quadcopter" in env_name:
            print(f"  - {env_name}")

    # tdmpc imports
    from tdmpc.src.cfg import parse_cfg
    from tdmpc.src.env import make_env
    from tdmpc.src.algorithm.tdmpc import TDMPC
    from tdmpc.src.algorithm.helper import Episode, ReplayBuffer
    from tdmpc.src.logger import Logger

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Helper function to safely convert tensors to NumPy arrays
    def safe_to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    # Helper function to ensure value is a CUDA tensor
    def ensure_cuda_tensor(x, dtype=torch.float):
        if isinstance(x, torch.Tensor):
            # If already a tensor, move to CUDA if needed
            if x.device.type != 'cuda':
                return x.to(device='cuda')
            return x
        else:
            # Convert to tensor and move to CUDA
            return torch.tensor(x, device='cuda', dtype=dtype)
        
    # Helper function to get shape safely
    def get_shape(x):
        if isinstance(x, torch.Tensor):
            return list(x.shape)
        elif isinstance(x, np.ndarray):
            return list(x.shape)
        elif hasattr(x, 'shape'):
            return list(x.shape)
        return 'N/A'

    # Helper function to check if done (handles both scalar and tensor values)
    def is_done(terminated, truncated):
        if isinstance(terminated, torch.Tensor) and terminated.numel() > 1:
            # If it's a tensor with multiple values, consider done if any env is done
            term_done = terminated.any().item()
        elif isinstance(terminated, torch.Tensor):
            term_done = terminated.item()
        else:
            term_done = terminated
            
        if isinstance(truncated, torch.Tensor) and truncated.numel() > 1:
            # If it's a tensor with multiple values, consider done if any env is done
            trunc_done = truncated.any().item()
        elif isinstance(truncated, torch.Tensor):
            trunc_done = truncated.item()
        else:
            trunc_done = truncated
            
        return term_done or trunc_done

    # Helper function to extract observation from dictionary if needed
    def extract_observation(obs):
        if isinstance(obs, dict):
            # print(f"Observation is a dictionary with keys: {obs.keys()}")
            if 'state' in obs:
                return safe_to_numpy(obs['state'])
            elif 'proprioceptive' in obs:
                return safe_to_numpy(obs['proprioceptive'])
            elif 'observations' in obs:
                return safe_to_numpy(obs['observations'])
            elif 'policy' in obs:
                return safe_to_numpy(obs['policy'])
            else:
                # Try to concatenate all numeric arrays in the observation
                numeric_arrays = []
                for k, v in obs.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        # Move tensor to CPU if it's on GPU
                        v_np = safe_to_numpy(v)
                        numeric_arrays.append(v_np.reshape(-1))
                if numeric_arrays:
                    return np.concatenate(numeric_arrays)
                # If we couldn't extract anything useful, just use the first value
                return safe_to_numpy(next(iter(obs.values())))
        return safe_to_numpy(obs)
    
    # Transform observation to match model's expected dimension
    TARGET_OBS_DIM = args.obs_dim  # The dimension that the TD-MPC model expects
    
    def transform_observation(obs, target_dim=TARGET_OBS_DIM):
        """Transform observation to the dimension expected by the model."""
        if isinstance(obs, torch.Tensor):
            # Convert tensor to numpy for processing
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.array(obs)
        
        # Flatten in case it's multi-dimensional
        obs_flat = obs_np.flatten()
        
        # Get current dimension
        current_dim = obs_flat.shape[0]
        
        # Print dimensions for debugging
        # print(f"Original obs shape: {obs_np.shape}, flattened: {obs_flat.shape}, target: {target_dim}")
        
        if current_dim == target_dim:
            # No need to transform
            return obs_flat
        elif current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim, dtype=obs_flat.dtype)
            padded[:current_dim] = obs_flat
            return padded
        else:
            # Truncate to match target dimensions
            return obs_flat[:target_dim]

    def evaluate(env, agent, num_episodes, step, env_step, video):
        """Evaluate a trained agent and optionally save a video."""
        episode_rewards = []
        for i in range(num_episodes):
            obs, _ = env.reset()
            extracted_obs = extract_observation(obs)
            # Transform observation to match model dimensions
            obs_transformed = transform_observation(extracted_obs)
            
            done = False
            ep_reward = 0
            t = 0
            if video: video.init(env, enabled=(i==0))
            while not done:
                action = agent.plan(obs_transformed, eval_mode=True, step=step, t0=t==0)
                # Prepare action for env - reshape if needed
                action_for_env = prepare_action_for_env(action)
                
                step_out = env.step(action_for_env)
                if len(step_out) == 5:
                    next_obs, reward, terminated, truncated, info = step_out
                    done = is_done(terminated, truncated)
                else:
                    next_obs, reward, done, _ = step_out
                    if isinstance(done, torch.Tensor) and done.numel() > 1:
                        # If it's a tensor with multiple values, consider done if any env is done
                        done = done.any().item()
                    elif isinstance(done, torch.Tensor):
                        done = done.item()
                
                # Extract and transform next observation
                extracted_next_obs = extract_observation(next_obs)
                obs_transformed = transform_observation(extracted_next_obs)
                
                # Ensure reward is a scalar value
                if isinstance(reward, torch.Tensor):
                    # Move to CPU first if it's a CUDA tensor
                    if reward.device.type == 'cuda':
                        reward = reward.cpu()
                    reward = reward.mean().item()
                
                ep_reward += reward
                if video: video.record(env)
                t += 1
            
            # Ensure episode_reward is a CPU scalar before adding to list
            if isinstance(ep_reward, torch.Tensor):
                if ep_reward.device.type == 'cuda':
                    ep_reward = ep_reward.cpu()
                ep_reward = ep_reward.item()
                
            episode_rewards.append(ep_reward)
            if video: video.save(env_step)
            
        # Convert all values to CPU scalars if needed
        episode_rewards = [r.cpu().item() if isinstance(r, torch.Tensor) and r.device.type == 'cuda' 
                          else r.item() if isinstance(r, torch.Tensor) 
                          else r for r in episode_rewards]
        
        return np.mean(episode_rewards)

    def train(cfg, env):
        """Training script for TD-MPC. Requires a CUDA-enabled device."""
        assert torch.cuda.is_available()
        set_seed(cfg.seed)
        work_dir = Path().cwd() / "logs" / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
        agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
        
        # Run training
        L = Logger(work_dir, cfg)
        episode_idx, start_time = 0, time.time()
        for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

            # Collect trajectory of fixed length
            print(f"Starting episode {episode_idx}, step {step}")
            
            # Get the initial observation first
            obs_dict, _ = env.reset()
            extracted_obs = extract_observation(obs_dict)
            
            # Transform observation to match model dimensions
            obs_transformed = transform_observation(extracted_obs)
            print(f"Transformed observation shape: {np.array(obs_transformed).shape}")
            
            # Create the episode with the transformed initial observation
            episode = Episode(cfg, obs_transformed)
            
            # Track if environment is done
            env_done = False
            real_episode_steps = 0
            
            # Collect fixed number of steps for consistent episode lengths
            while len(episode) < cfg.episode_length:
                real_episode_steps += 1
                
                # If environment is done, reset it
                if env_done:
                    # print(f"Environment terminated early at step {len(episode)}. Resetting...")
                    obs_dict, _ = env.reset()
                    extracted_obs = extract_observation(obs_dict)
                    obs_transformed = transform_observation(extracted_obs)
                    env_done = False
                
                # Get action from policy
                action = agent.plan(obs_transformed, step=step, t0=episode.first)
                
                # Ensure action is on CUDA
                action_cuda = ensure_cuda_tensor(action)
                
                # Prepare action for env - reshape if needed
                action_for_env = prepare_action_for_env(action_cuda)
                
                # Step the environment
                step_out = env.step(action_for_env)
                
                # Handle return values
                if len(step_out) == 5:
                    next_obs_dict, reward, terminated, truncated, info = step_out
                    # Safely handle tensor boolean operation
                    env_done = is_done(terminated, truncated)
                    
                    # Convert reward to scalar if it's a tensor
                    if isinstance(reward, torch.Tensor):
                        reward = reward.mean().item()  # Use mean of all environment rewards
                else:
                    next_obs_dict, reward, done, _ = step_out
                    if isinstance(done, torch.Tensor) and done.numel() > 1:
                        # If it's a tensor with multiple values, consider done if any env is done
                        env_done = done.any().item()
                    elif isinstance(done, torch.Tensor):
                        env_done = done.item()
                    else:
                        env_done = done
                    
                    # Convert reward to scalar if it's a tensor
                    if isinstance(reward, torch.Tensor):
                        reward = reward.mean().item()  # Use mean of all environment rewards
                    
                # Extract observation for agent
                extracted_next_obs = extract_observation(next_obs_dict)
                
                # Transform next observation to match model dimensions
                next_obs_transformed = transform_observation(extracted_next_obs)
                
                # Add to episode
                # Only set done=True on the final step of the episode
                is_last_step = (len(episode) == cfg.episode_length - 1)
                episode += (next_obs_transformed, action_cuda, reward, is_last_step)
                
                # Update observation for next step
                obs_transformed = next_obs_transformed
                
            # Verify we have collected enough steps
            print(f"Episode {episode_idx} collected {real_episode_steps} steps, len(episode)={len(episode)}")
            
            # Now ensure the episode length matches what's expected
            assert len(episode) == cfg.episode_length, f"Episode length {len(episode)} doesn't match expected {cfg.episode_length}"
            
            # Add to replay buffer
            buffer += episode

            # Update model
            train_metrics = {}
            if step >= cfg.seed_steps:
                num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
                for i in range(num_updates):
                    train_metrics.update(agent.update(buffer, step+i))

            # Log training episode
            episode_idx += 1
            env_step = int(step*cfg.action_repeat)
            common_metrics = {
                'episode': episode_idx,
                'step': step,
                'env_step': env_step,
                'total_time': time.time() - start_time,
                'episode_reward': episode.cumulative_reward}
            train_metrics.update(common_metrics)
            L.log(train_metrics, category='train')

            # Evaluate agent periodically
            if env_step % cfg.eval_freq == 0:
                common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
                L.log(common_metrics, category='eval')

        L.finish(agent)
        print('Training completed successfully')

    # --- Load & override TD‑MPC config ---
    # Use the directory path only, not the full file path
    cfg_dir = Path("/home/ftesshu/dream_sim2real/tdmpc/cfgs")
    if not (cfg_dir / "default.yaml").exists():
        # Try the tasks subdirectory as an alternative
        cfg_dir = Path("/home/ftesshu/dream_sim2real/tdmpc/cfgs/tasks")
    
    print(f"Loading config from directory: {cfg_dir}")
    cfg = parse_cfg(cfg_dir)
    
    # Set the task name
    cfg.task = args.task
    
    # Override other parameters if provided
    for key in ["seed", "train_steps", "episode_length", "action_repeat",
                "eval_freq", "eval_episodes", "exp_name", "modality"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    # --- Instantiate IsaacLab quadcopter env ---
    env_cfg = QuadcopterEnvCfg()  # defaults from Python dataclass
    if args.device:
        env_cfg.sim.device = args.device
    if args.headless:
        env_cfg.sim.backend = "HEADLESS"
    
    print(f"\nCreating environment: {args.task}")
    try:
        env = gym.make(args.task, cfg=env_cfg)
    except gym.error.NameNotFound as e:
        print(f"Error: {e}")
        # Try without the version suffix
        if "-v" in args.task:
            alt_task = args.task.split("-v")[0]
            print(f"Trying alternative task name: {alt_task}")
            env = gym.make(alt_task, cfg=env_cfg)
            # Update the task name in config
            cfg.task = alt_task
    
    # Print observation and action spaces for debugging
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Helper function to prepare action for env
    n_envs = env_cfg.scene.num_envs
    def prepare_action_for_env(action):
        # Ensure action is a torch tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=torch.device("cuda:0"))
            
        # Check if action needs reshaping - IsaacLab expects [batch_size, action_dim]
        return action.view(1, -1).repeat(n_envs, 1)

    
    # Check the observation type from a sample
    obs_sample, _ = env.reset()
    print(f"Sample observation type: {type(obs_sample)}")
    if isinstance(obs_sample, dict):
        print(f"Observation dictionary keys: {obs_sample.keys()}")
        for k, v in obs_sample.items():
            print(f"  {k}: {type(v)} shape={get_shape(v)}")

    # Extract and reshape observation for TDMPC
    test_obs = extract_observation(obs_sample)
    
    # Transform to match model dimensions
    transformed_test_obs = transform_observation(test_obs)
    obs_shape = tuple(np.array(transformed_test_obs).shape)
    
    # Fix the issue with OmegaConf ListConfig not supporting slicing
    # We need to ensure cfg.obs_shape is a tuple of ints, not an OmegaConf ListConfig
    # First, set the modality explicitly to "state" to avoid the problematic slicing code
    cfg.modality = "state"  # Make sure we're using state modality
    
    # Set shape as regular Python tuples to match the model expectation
    cfg.obs_shape = obs_shape
    
    print(f"Using observation shape: {cfg.obs_shape}")
    print(f"Using modality: {cfg.modality}")
    
    # Set action shape and dimension
    cfg.action_shape = tuple(env.action_space.shape)
    cfg.action_dim = env.action_space.shape[1]
    
    # --- Run TD‑MPC training ---
    train(cfg, env)

    # --- Clean up ---
    sim_app.close()
    env.close()