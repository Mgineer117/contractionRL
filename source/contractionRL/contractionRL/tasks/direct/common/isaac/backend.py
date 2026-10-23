"""Backend selection shared by environment creation, training and playback."""

def dynamics_backend_for_task(task):
    """The registered task determines transitions; no algorithm setting can override it."""
    name = str(task).split(':')[-1]
    return 'analytical' if name.startswith('Isaac-Classic-') else 'physx'


def configure_backend(cfg, *, task):
    backend = dynamics_backend_for_task(task)
    if not hasattr(cfg, 'classic_task'):
        return backend
    if getattr(cfg, 'reference_mode', 'stabilizing') != 'stabilizing':
        raise ValueError('Paired Isaac tasks support reference_mode=stabilizing')
    dt = .025 if cfg.classic_task == 'classic-quadrotor-v0' else .03
    cfg.decimation = round(dt / .005) if backend == 'physx' else 1
    cfg.sim.dt = dt / cfg.decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.replicate_physics = backend == 'physx'
    cfg.sim.gravity = (0., 0., 0.) if backend == 'physx' and cfg.classic_task in ('classic-car-v0', 'classic-car-v1') else (0., 0., -9.81)
    if backend == 'physx':
        cfg.sim.physx.enable_external_forces_every_iteration = True
    return backend


def configure_backend_agent(agent_cfg, task, *, classic_derived=True):
    from contractionRL.agents.utils.RL.config import validate_experiment_config
    validate_experiment_config(agent_cfg)
    if not classic_derived:
        return
    backend = dynamics_backend_for_task(task)
    for block in ('cm', 'cmg', 'agent', 'empirical_dynamics'):
        for key in ('cm_data_path', 'dynamics_pretrain_data_path'):
            value = agent_cfg.get(block, {}).get(key)
            if value:
                from pathlib import Path
                p = Path(value)
                cache_task = task.split(':')[-1].replace('Isaac-Classic-', 'Isaac-', 1)
                canonical_body = cache_task in ('Isaac-Car-v0', 'Isaac-Car-v1', 'Isaac-Quadrotor-v0')
                folder = cache_task + ('-physx-v4' if canonical_body else '-physx-v2')
                prior_folders = tuple(cache_task + f'-physx-v{v}' for v in (1, 2, 3, 4))
                if backend == 'analytical' and p.parent.name in prior_folders:
                    agent_cfg[block][key] = str(p.parent.parent / p.name)
                elif backend == 'physx' and p.parent.name != folder:
                    parent = p.parent.parent if p.parent.name in prior_folders else p.parent
                    agent_cfg[block][key] = str(parent / folder / p.name)


def checkpoint_reference_bank(checkpoint):
    from pathlib import Path
    bank = Path(checkpoint).resolve().parent.parent / 'physx_reference_bank.npz'
    if not bank.is_file():
        raise FileNotFoundError(f'PhysX checkpoint requires its training reference bank: {bank}')
    return str(bank)
