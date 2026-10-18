"""Excited nominal PhysX transitions for learning f(x) + B(x)u."""
from pathlib import Path

import numpy as np
import torch

from contractionRL.agents.utils.system_dynamics.state import wrap_diff


@torch.no_grad()
def collect_system_id(env, sample_count, controls_per_state):
    """Probe multiple controls from each state, then restore live episode state.

    One physics substep approximates a continuous-time derivative. Reference
    trajectories alone (especially equilibrium references) cannot identify B.
    No episode counters, reference IDs or held-disturbance timers are advanced.
    """
    if sample_count < 1 or controls_per_state < 2:
        raise ValueError('System identification needs positive samples and at least two controls per state')
    ids = torch.arange(env.num_envs, device=env.device)
    saved_x = env.get_physical_state().clone()
    saved_action, saved_gust = env._actions.clone(), env._gust.clone()
    records = {key: [] for key in ('x','u','x_next','x_dot','state_id')}
    count, batches = 0, 0
    margin = env.reference_margin
    low, high = env.X_MIN+margin, env.X_MAX-margin
    try:
        env._gust.zero_()  # Identify nominal dynamics; gusts remain enabled for RL.
        while count < sample_count:
            if batches >= max(20, 20 * (sample_count // env.num_envs + 1)):
                raise RuntimeError('Insufficient finite in-domain PhysX identification transitions')
            x = low + torch.rand(env.num_envs, env.x_dim, device=env.device)*(high-low)
            group = ids + batches*env.num_envs
            for _ in range(controls_per_state):
                env._write_state(ids, x)
                before = env.get_physical_state().clone()
                u = env.U_MIN + torch.rand(env.num_envs, env.u_dim, device=env.device)*(env.U_MAX-env.U_MIN)
                env._actions = u
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                after = env.get_physical_state().clone()
                valid = torch.isfinite(before).all(-1) & torch.isfinite(after).all(-1)
                valid &= ((before >= env.X_MIN) & (before <= env.X_MAX)).all(-1)
                valid &= ((after >= env.X_MIN) & (after <= env.X_MAX)).all(-1)
                derivative = wrap_diff(after-before, env.angle_idx) / env.physics_dt
                valid &= torch.isfinite(derivative).all(-1)
                for key,value in [('x',before),('u',u),('x_next',after),('x_dot',derivative),('state_id',group)]:
                    records[key].append(value[valid].clone())
                count += int(valid.sum())
            batches += 1
    finally:
        env._write_state(ids, saved_x)
        env._actions, env._gust = saved_action, saved_gust
    # Keep groups together in storage; callers randomly subsample their buffer.
    result = {key: torch.cat(values) for key,values in records.items()}
    if getattr(env.cfg,'log_dir',None):
        target = Path(env.cfg.log_dir)/'physx_system_id.npz'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite identification evidence: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target, **{key:value.cpu().numpy()[:,None] for key,value in result.items()},
                            dt=env.physics_dt, task=env.cfg.classic_task, model=env.model_version,
                            seed=env.cfg.seed, controls_per_state=controls_per_state,
                            source='excited_physx_transitions', disturbance_enabled=False)
    print(f'[PhysX] Collected {len(result["x"])} nominal identification transitions; '
          f'{controls_per_state} controls/state, dt={env.physics_dt:g}s',flush=True)
    return result
