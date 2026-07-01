import re

with open("/home/minjae/research/contractionRL/scripts/skrl/train.py", "r") as f:
    content = f.read()

# 1. Initialize variables
old_init = """    all_states, all_actions = [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)
    
    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]
    
    ep_states = torch.zeros((num_envs, T, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T, u_dim), dtype=torch.float32, device=skrl_env.device)
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=skrl_env.device)"""

new_init = """    import tqdm
    all_states, all_actions, all_pos = [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)
    
    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]
    
    ep_states = torch.zeros((num_envs, T, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T, u_dim), dtype=torch.float32, device=skrl_env.device)
    ep_pos = torch.zeros((num_envs, T, 3), dtype=torch.float32, device=skrl_env.device)
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=skrl_env.device)
    
    pbar = tqdm.tqdm(total=num_trajs, desc="[RefTraj] Collecting")"""

content = content.replace(old_init, new_init)

# 2. Update loop logic
old_loop = """        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        ep_actions[valid_indices, step_counts[valid_indices]] = actions[valid_indices].float()
        step_counts[valid_indices] += 1
        
        obs_dict, _, terminated, truncated, _ = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)
        
        if done.any():
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
                a_np = ep_actions[success_indices].cpu().numpy()
                for i in range(len(success_indices)):
                    all_states.append(s_np[i])
                    all_actions.append(a_np[i])
                    if len(all_states) % 200 == 0:
                        print(f"[RefTraj]   {len(all_states)} / {num_trajs}")
                    if len(all_states) >= num_trajs:
                        break"""

new_loop = """        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        ep_actions[valid_indices, step_counts[valid_indices]] = actions[valid_indices].float()
        if hasattr(unwrapped, "_robot"):
            ep_pos[valid_indices, step_counts[valid_indices]] = unwrapped._robot.data.root_pos_w[valid_indices].float()
        
        step_counts[valid_indices] += 1
        
        obs_dict, _, terminated, truncated, _ = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)
        
        if done.any():
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
                        ep_pos[i, length:] = ep_pos[i, length - 1].clone()
                        
                s_np = ep_states[success_indices].cpu().numpy()
                a_np = ep_actions[success_indices].cpu().numpy()
                p_np = ep_pos[success_indices].cpu().numpy()
                for i in range(len(success_indices)):
                    if len(all_states) >= num_trajs:
                        break
                    all_states.append(s_np[i])
                    all_actions.append(a_np[i])
                    all_pos.append(p_np[i])
                    pbar.update(1)"""

content = content.replace(old_loop, new_loop)

# 3. Post collection logic
old_post = """    states_arr = np.stack(all_states[:num_trajs]).astype(np.float32)
    actions_arr = np.stack(all_actions[:num_trajs]).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print(f"[RefTraj] Saved → {out_path}  states{states_arr.shape}  actions{actions_arr.shape}")
    
    # Generate dynamics data via finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation
    # 4th-order central difference for velocity (matches generate_ref_traj.py)
    x_dot_arr = np.zeros_like(states_arr)"""

new_post = """    pbar.close()
    states_arr = np.stack(all_states[:num_trajs]).astype(np.float32)
    actions_arr = np.stack(all_actions[:num_trajs]).astype(np.float32)
    pos_arr = np.stack(all_pos[:num_trajs]).astype(np.float32)
    
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print(f"\\n[RefTraj] Saved → {out_path}  states{states_arr.shape}  actions{actions_arr.shape}")
    
    # Plot absolute position of 10 sampled trajectories
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for i in range(min(10, num_trajs)):
            plt.plot(pos_arr[i, :, 0], pos_arr[i, :, 1], label=f"Traj {i+1}")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Sampled Reference Trajectories (Absolute Position)")
        plt.legend()
        plot_path = os.path.join(out_dir, "position_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[RefTraj] Saved position plot → {plot_path}")
    except Exception as e:
        print(f"[RefTraj] Failed to generate position plot: {e}")
    
    # Generate dynamics data via finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"[RefTraj] Computing dynamics (x_dot) via 4th-order central difference (dt={dt:.3f})...")
    x_dot_arr = np.zeros_like(states_arr)"""

content = content.replace(old_post, new_post)

with open("/home/minjae/research/contractionRL/scripts/skrl/train.py", "w") as f:
    f.write(content)
