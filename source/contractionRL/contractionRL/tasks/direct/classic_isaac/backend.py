"""Backend selection shared by environment creation, training and playback."""

def configure_backend(cfg, mode=None):
    from contractionRL.agents.utils.RL.config import EXPERIMENT_MODES
    if mode is not None and mode not in EXPERIMENT_MODES:
        raise ValueError(f'Unknown mode: {mode}')
    if not hasattr(cfg, 'classic_task'):
        cfg.mode = 'sim_to_real'
        return
    mode = mode or getattr(cfg, 'mode', None) or ('sim_to_real' if getattr(cfg, 'dynamics_backend', None) == 'physx' else 'analytical')
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f'Unknown mode: {mode}')
    backend = 'physx' if mode == 'sim_to_real' else 'analytical'
    if backend == 'physx' and getattr(cfg, 'reference_mode', 'stabilizing') != 'stabilizing':
        raise ValueError('PhysX uses measured nominal references; reference_mode=contractive is analytical-only')
    cfg.mode = mode
    dt = .025 if cfg.classic_task == 'classic-quadrotor-v0' else .03
    cfg.decimation = round(dt / .005) if backend == 'physx' else 1
    cfg.sim.dt = dt / cfg.decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.replicate_physics = backend == 'physx'
    cfg.sim.gravity = (0., 0., 0.) if backend == 'physx' and cfg.classic_task in ('classic-car-v0', 'classic-car-v1') else (0., 0., -9.81)
    if backend == 'physx':
        cfg.sim.physx.enable_external_forces_every_iteration = True


def configure_backend_agent(agent_cfg, mode, task, *, classic_derived=True):
    from contractionRL.agents.utils.RL.config import normalize_experiment_mode
    agent_cfg['mode'] = mode
    normalize_experiment_mode(agent_cfg)
    if not classic_derived:
        return
    # Old analytical metrics and dynamics have different state widths/physics.
    for block in ('cm', 'cmg', 'agent', 'empirical_dynamics'):
        for key in ('cm_data_path', 'dynamics_pretrain_data_path'):
            value = agent_cfg.get(block, {}).get(key)
            if value:
                from pathlib import Path
                p = Path(value)
                folder = task + '-physx-v1'
                if mode == 'analytical' and p.parent.name == folder:
                    agent_cfg[block][key] = str(p.parent.parent / p.name)
                elif mode == 'sim_to_real' and p.parent.name != folder:
                    agent_cfg[block][key] = str(p.parent / folder / p.name)


def checkpoint_reference_bank(checkpoint):
    from pathlib import Path
    bank = Path(checkpoint).resolve().parent.parent / 'physx_reference_bank.npz'
    if not bank.is_file():
        raise FileNotFoundError(f'PhysX checkpoint requires its training reference bank: {bank}')
    return str(bank)
