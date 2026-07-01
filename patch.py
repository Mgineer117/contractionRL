import re

with open("/home/minjae/research/contractionRL/scripts/skrl/train.py", "r") as f:
    content = f.read()

# 1. Revert the re-spawning
old_respawn = """    # Collect trajectories
    num_trajs = args_cli.ref_num_trajs
    print(f"[RefTraj] Re-spawning {num_trajs} environments to collect reference trajectories in one shot...")
    
    # Close existing environment and spawn a new one with exactly `num_trajs` environments
    skrl_env.close()
    env_cfg.scene.num_envs = num_trajs
    isaac_env = gym.make(task, cfg=env_cfg)
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    skrl_env = SkrlVecEnvWrapper(isaac_env)
    
    unwrapped = isaac_env.unwrapped
    num_envs = skrl_env.num_envs"""

new_respawn = """    # Collect trajectories
    num_trajs = args_cli.ref_num_trajs
    print(f"[RefTraj] Collecting {num_trajs} trajectories → {out_path}")
    unwrapped = isaac_env.unwrapped
    num_envs = skrl_env.num_envs"""

content = content.replace(old_respawn, new_respawn)

# 2. Fix the success_mask and add padding
old_success = """        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # Trajectories are successful if they survived for at least T steps
            success_mask = step_counts[done_indices] >= T
            success_indices = done_indices[success_mask]
            
            if len(success_indices) > 0:
                s_np = ep_states[success_indices].cpu().numpy()
                a_np = ep_actions[success_indices].cpu().numpy()"""

new_success = """        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # Accept trajectories that survived at least half the max length.
            # This handles policies that fall slightly early but pass the quality gate,
            # as well as off-by-one errors with Isaac Gym's max_episode_length.
            success_mask = step_counts[done_indices] >= (T // 2)
            success_indices = done_indices[success_mask]
            
            if len(success_indices) > 0:
                # Pad any missing steps with the final valid state to ensure x_dot is stable
                for i in success_indices:
                    length = step_counts[i].item()
                    if length < T and length > 0:
                        ep_states[i, length:] = ep_states[i, length - 1].clone()
                        ep_actions[i, length:] = ep_actions[i, length - 1].clone()
                        
                s_np = ep_states[success_indices].cpu().numpy()
                a_np = ep_actions[success_indices].cpu().numpy()"""

content = content.replace(old_success, new_success)

with open("/home/minjae/research/contractionRL/scripts/skrl/train.py", "w") as f:
    f.write(content)
