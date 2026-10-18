"""Fixed-bank analytical-car evaluation. Holdout is never used for selection."""
import argparse
import json
from pathlib import Path
import sys


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(root):
    sys.path[:0] = [str(root / 'code/source/contractionRL'), str(root / 'code/scripts/skrl')]
    import torch
    import contractionRL.tasks.direct.classic  # noqa: F401
    torch.set_num_threads(2)


def load_bank(root, name):
    import numpy as np
    with np.load(root / 'banks' / (name + '.npz')) as data:
        return {key: data[key].copy() for key in data.files}


def prepare(root):
    import gymnasium as gym
    import numpy as np
    import torch
    from contractionRL.tasks.direct.common.disturbance import BoundedGust
    banks = root / 'banks'
    banks.mkdir(parents=True, exist_ok=True)
    for name, ref_seed, gust_seed, action_seed in [
        ('validation', 20261017, 20261024, 20261025),
        ('holdout', 20261027, 20261028, 20261029),
    ]:
        target = banks / (name + '.npz')
        if target.exists():
            raise FileExistsError('Preserve prior bank: ' + str(target))
        torch.manual_seed(ref_seed)
        env = gym.make('classic-car-v0', num_envs=512, device='cpu', disturbance_scales=[0.] * 4).unwrapped
        try:
            env.set_fix_ref_trajectories(True)
            env.reset()
            bank = dict(x0=env.x_t.numpy().copy(), xref=env.xref.numpy().copy(),
                        uref=env.uref.numpy().copy(), parameters=env._reference_parameters.numpy().copy(),
                        seed=ref_seed, dt=env.dt, disturbance_enabled=False)
        finally:
            env.close()
        torch.manual_seed(gust_seed)
        gust = BoundedGust(512, [0., 0., .02, .05], .03, 'cpu')
        gust.reset(torch.arange(512))
        bank.update(gusts=torch.stack([gust.advance() for _ in range(500)], 1).numpy(),
                    gust_seed=gust_seed, action_seed=action_seed)
        np.savez_compressed(target, **bank)
        evaluate(root, 'lqr', name, root / 'baseline' / name)


def attach_metric(env, checkpoint):
    """Optional metric attachment affects reward only, never AUC or controls."""
    import torch
    from contractionRL.agents.utils.RL.models import MetricModel
    from contractionRL.agents.utils.system_dynamics.state import embedded_dim
    from train_utils import _resolve_symmetry_for_env
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['cmg']
    weights = [value for key, value in state.items()
               if key.startswith('ccm_gen.model.net.') and key.endswith('.weight')]
    if not weights:
        raise ValueError('Expected a bounded CCM metric checkpoint')
    angles = list(env.angle_idx)
    sym = _resolve_symmetry_for_env(env)
    width = weights[0].shape[1]
    if sym is not None and width == sym.single_dim():
        pass
    elif width == embedded_dim(env.num_dim_x, angles):
        sym = None
    else:
        raise ValueError('Checkpoint metric input width does not match the physical state or translation quotient')
    import yaml
    metric_cfg=yaml.safe_load((Path(checkpoint).parent.parent/'params/agent.yaml').read_text())['cm']
    lower,upper=float(metric_cfg['ccm_w_lb']),float(metric_cfg['ccm_w_ub'])
    metric = MetricModel(env.observation_space, env.action_space, 'cpu', hidden_dim=[w.shape[0] for w in weights],
                         angle_idx=angles, sym=sym, constrain_eigenvalues=True,
                         w_lb=lower, w_ub=upper, outputs_metric=True)
    metric.load_state_dict(state)
    metric.eval()
    metric.requires_grad_(False)
    env.set_ccm(metric.ccm_gen, lower, 'cpu')


def evaluate(root, checkpoint, name, output, metric_checkpoint=None, mean=False):
    import gymnasium as gym
    import numpy as np
    import torch
    from compare_baselines import install_bank, load_actor
    from contractionRL.agents.skrl.sdlqr import _care_gain
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_bank(root, name)
    n, steps, dt = 512, 500, .03
    assert len(data['x0']) == n and abs(float(data['dt']) - dt) < 1e-7
    assert data['gusts'].shape == (n, steps, 4)
    env = gym.make('classic-car-v0', num_envs=n, device='cpu', disturbance_scales=[0.] * 4).unwrapped
    actor = None
    try:
        import yaml
        beta = (float(yaml.safe_load((Path(checkpoint).parent.parent / 'params/agent.yaml').read_text())['agent'].get('action_bound_penalty_beta', 0.))
                if checkpoint != 'lqr' else 0.)
        env.set_action_bound_penalty(beta)
        if metric_checkpoint:
            attach_metric(env, metric_checkpoint)
        install_bank(env, data)
        if checkpoint != 'lqr':
            actor = load_actor(env, checkpoint)
        A = torch.tensor([[0., 0., 0., 1.], [0., 0., 1.5, 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
        B = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
        K0 = _care_gain(A, B, 1., 1., 4, 2)
        gusts = torch.as_tensor(data['gusts'])
        cursor = [0]
        def advance():
            value = gusts[:, cursor[0]]
            cursor[0] += 1
            return value
        env.disturbance.advance = advance
        noise = torch.Generator().manual_seed(int(data['action_seed']))
        errors = torch.full((n, steps + 1), float('nan'))
        errors[:, 0] = env.wrap_angles(env.x_t - env.xref[:, 0]).norm(dim=-1)
        alive = torch.ones(n, dtype=torch.bool)
        clipped, overflow, counts = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        with torch.no_grad():
            for k in range(steps):
                obs = env.construct_state_flat(env.x_t)
                if actor is None:
                    x, refs, controls = env.ref_window.split(obs)
                    yaw = refs[:, 0, 2]
                    rotation = torch.eye(4).expand(n, 4, 4).clone()
                    c, s = yaw.cos(), yaw.sin()
                    rotation[:, 0, 0], rotation[:, 0, 1] = c, s
                    rotation[:, 1, 0], rotation[:, 1, 1] = -s, c
                    error = env.wrap_angles(x - refs[:, 0])
                    action = controls[:, 0] - (K0[None] @ rotation @ error.unsqueeze(-1)).squeeze(-1)
                else:
                    action, _ = actor.compute({'observations': obs})
                    if not mean:
                        action = action + actor.log_std_parameter.clamp(-6.9, 2.).exp() * torch.randn(action.shape, generator=noise)
                invalid = ~torch.isfinite(action).all(-1)
                safe = action.nan_to_num(nan=0., posinf=0., neginf=0.)
                clipped += (((safe < env.U_MIN) | (safe > env.U_MAX)).any(-1) & alive).float()
                overflow += torch.where(alive, (safe - safe.clamp(env.U_MIN, env.U_MAX)).norm(dim=-1), 0.)
                counts += alive.float()
                _, _, terminated, truncated, info = env.step(safe)
                error = env.wrap_angles(info['x'] - torch.as_tensor(data['xref'][:, k + 1])).norm(dim=-1)
                errors[alive, k + 1] = error[alive]
                alive &= ~(invalid | terminated | (truncated & (k + 1 < steps)) | ~torch.isfinite(error))
        normalized = errors / errors[:, :1].clamp_min(1e-8)
        auc = torch.trapezoid(normalized, dx=dt, dim=1).numpy()
        complete = alive.numpy() & np.isfinite(auc)
        all_complete = bool(complete.all())
        result = dict(status='COMPLETED' if all_complete else 'INCOMPLETE', auc_mean=float(auc.mean()) if all_complete else None,
                      auc_stddev=float(auc.std(dtype=np.float64)) if all_complete else None,
                      episodes=n, completed_episodes=int(complete.sum()), failed_episodes=int((~complete).sum()),
                      episode_completion=float(complete.mean()), bank=name, checkpoint=str(checkpoint),
                      stochastic=actor is not None and not mean, disturbance=True,
                      final_worse_than_start=int(((normalized[:, -1] > 1).numpy() & complete).sum()),
                      action_clip_fraction=float((clipped / counts.clamp_min(1)).mean()),
                      action_overflow_l2_mean=float((overflow / counts.clamp_min(1)).mean()),
                      action_bound_penalty_beta=beta, reward_metric='CCM' if metric_checkpoint else 'identity (reward unused in ranking)',
                      reference_seed=int(data['seed']), gust_seed=int(data['gust_seed']), action_seed=int(data['action_seed']),
                      auc_definition='integral_0^15 ||e(t)||_2/||e(0)||_2 dt; lower is better')
        np.savez_compressed(output / 'episodes.npz', auc=auc, completed=complete, normalized_error=normalized.numpy(),
                            action_clip_fraction=(clipped / counts.clamp_min(1)).numpy(),
                            action_overflow_l2_mean=(overflow / counts.clamp_min(1)).numpy())
        baseline = root / 'baseline' / name / 'episodes.npz'
        if actor is not None and all_complete and baseline.exists():
            with np.load(baseline) as reference:
                if reference['completed'].all():
                    delta = auc - reference['auc']
                    se = float(delta.std(ddof=1) / np.sqrt(n))
                    result['paired_lqr_difference'] = dict(mean=float(delta.mean()), standard_error=se,
                        interval95=[float(delta.mean()) - 1.96 * se, float(delta.mean()) + 1.96 * se],
                        fraction_better=float((delta < 0).mean()))
        write_json(output / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('root', type=Path)
    ev = sub.add_parser('evaluate')
    ev.add_argument('root', type=Path)
    ev.add_argument('checkpoint')
    ev.add_argument('bank', choices=['validation', 'holdout'])
    ev.add_argument('output', type=Path)
    ev.add_argument('--metric-checkpoint')
    ev.add_argument('--mean', action='store_true', help='Diagnostic deterministic mean; default is stochastic policy')
    args = parser.parse_args()
    args.root = args.root.resolve()
    setup(args.root)
    if args.command == 'prepare':
        prepare(args.root)
    else:
        evaluate(args.root, args.checkpoint, args.bank, args.output, args.metric_checkpoint, args.mean)
