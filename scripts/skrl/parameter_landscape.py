"""Common-scenario return slices and finite-difference curvature of car policies.

Use --source-root to evaluate historical checkpoints with their archived code.
Only deployed mean-control weights/biases are perturbed; metric, critic,
normalization buffers and exploration variance remain fixed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.func import functional_call, vmap
import yaml


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def directions(size, count, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(count + 2, size, generator=generator, dtype=torch.float64)
    values /= values.norm(dim=1, keepdim=True)
    values[1] -= values[1].dot(values[0]) * values[0]
    values[1] /= values[1].norm()
    return values.float()


def curvature(returns, h):
    """Episode-paired differences; the caller retains all failed episodes."""
    plus, center, minus = [np.asarray(value, dtype=np.float64) for value in returns]
    return (plus - 2 * center + minus) / h**2


class MeanControl(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        if actor.control_mode != 'structured' or not actor.cl_actor.compact:
            raise ValueError('This evaluator requires matched compact structured-control actors')

    def features(self, observations):
        controller = self.actor.cl_actor
        x, refs, controls = controller.window.split(observations)
        return controller.tracking_features(observations), controller.tracking_features.error(x, refs[:, 0]), controls[:, 0]

    def forward(self, features, error, reference_control):
        # Compute parameter-independent reference features outside vmap: the
        # historical reference generator uses in-place tensor writes.
        controller = self.actor.cl_actor
        n = features.shape[0]
        w1 = controller.w1(features).reshape(n, controller.c, controller.x_dim)
        w2 = controller.w2(features).reshape(n, controller.u_dim, controller.c)
        return reference_control + (w2 @ torch.tanh(w1 @ error.unsqueeze(-1))).squeeze(-1)


def parameters(model):
    return {name: value for name, value in model.named_parameters()
            if value.requires_grad and name.rsplit('.', 1)[-1] in ('weight', 'bias')}


def unflatten(vector, template):
    result, offset = {}, 0
    for name, value in template.items():
        result[name] = vector[..., offset:offset + value.numel()].reshape(*vector.shape[:-1], *value.shape)
        offset += value.numel()
    if offset != vector.shape[-1]:
        raise ValueError('Parameter vector size mismatch')
    return result


def install_bank(env, bank, copies):
    env.set_fix_ref_trajectories(True)
    for key, name in [('x0', '_fixed_x0'), ('xref', '_fixed_xref'),
                      ('uref', '_fixed_uref'), ('parameters', '_fixed_parameters')]:
        target = getattr(env, name)
        value = torch.as_tensor(bank[key], device=env.device, dtype=target.dtype)
        value = value.repeat(copies, *([1] * (value.ndim - 1)))
        if value.shape != target.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid bank field: ' + key)
        target.copy_(value)
    env._ref_fixed[:] = True
    env.reset()
    torch.testing.assert_close(env.x_t, torch.as_tensor(bank['x0'], device=env.device).repeat(copies, 1), rtol=0, atol=0)


@torch.no_grad()
def rollout(env, model, vectors, template, bank, seed):
    """Evaluate a batch of policies, each on the same complete scenario bank."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.reset()
    batch, episodes = len(vectors), len(bank['x0'])
    reference = torch.as_tensor(bank['xref'], device=env.device).repeat(batch, 1, 1)
    initial = env.wrap_angles(env.x_t - reference[:, 0]).norm(dim=-1).double().clamp_min(1e-8)
    previous = torch.ones_like(initial)
    auc = torch.zeros_like(initial)
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    # An absorbing worst-box error prevents early termination from improving J.
    worst = (env.X_MAX - env.X_MIN).norm().double() / initial
    theta = unflatten(vectors, template)
    def mean(params, features, error, control):
        return functional_call(model, params, (features, error, control), strict=False)
    for k in range(env.max_episode_len):
        obs = env.construct_state_flat(env.x_t)
        features = [value.reshape(batch, episodes, -1) for value in model.features(obs)]
        action = vmap(mean, randomness='error')(theta, *features).reshape(env.num_envs, -1)
        valid = torch.isfinite(action).all(-1)
        _, _, terminated, truncated, info = env.step(action.nan_to_num())
        error = env.wrap_angles(info['x'] - reference[:, k + 1]).norm(dim=-1).double() / initial
        early = terminated | (truncated & (k + 1 < env.max_episode_len)) | ~valid | ~torch.isfinite(error)
        alive &= ~early
        current = torch.where(alive, error, worst)
        auc += .5 * env.dt * (previous + current)
        previous = current
    return -auc.reshape(batch, episodes).cpu().numpy(), alive.reshape(batch, episodes).cpu().numpy()


def plot(output, rows, axis, hs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    a, b = np.meshgrid(axis, axis, indexing='ij')
    surfaces = [np.load(output / name / 'surface.npz')['returns'].mean(-1) for name in rows]
    norm = Normalize(min(z.min() for z in surfaces), max(z.max() for z in surfaces))
    fig = plt.figure(figsize=(12, 5.2), layout='constrained')
    for i, (name, z) in enumerate(zip(rows, surfaces)):
        ax = fig.add_subplot(1, len(rows), i + 1, projection='3d')
        surface = ax.plot_surface(a, b, z, cmap='viridis', norm=norm, linewidth=0, antialiased=True)
        ax.set(xlabel='a (parameter L2 units)', ylabel='b (parameter L2 units)',
               zlabel='J (negative tracking AUC)', title=f'{name}: J(0,0)={rows[name]["center_return"]:.3f}')
        ax.set_zlim(norm.vmin, norm.vmax)
    fig.colorbar(surface, ax=fig.axes, shrink=.55, label='Common evaluation return J')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'return_surfaces.{ext}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for name in rows:
        history = output.parent / f'{name}_training_history.json'
        if history.exists():
            points = json.loads(history.read_text())['rows']
            ax.plot([point['global_step'] for point in points],
                    [point['Evaluation/Stability/auc_mean'] for point in points], label=name, alpha=.8)
    if ax.lines:
        ax.set(xlabel='Training steps', ylabel='Historical evaluation tracking AUC (lower is better)',
               yscale='log', title='Existing checkpoint training histories (seed 42)')
        ax.legend()
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'training_convergence.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(hs), figsize=(12, 4), layout='constrained', squeeze=False, sharey=True)
    for j, h in enumerate(hs):
        values = [np.abs(np.load(output / name / 'curvature.npz')['curvature'][j].mean(-1)) for name in rows]
        axes[0, j].boxplot(values, tick_labels=list(rows), showmeans=True)
        axes[0, j].set(title=f'h = {h:g}', ylabel='Absolute directional curvature')
        axes[0, j].set_yscale('symlog', linthresh=1e-4)
        axes[0, j].set_ylim(bottom=0)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'directional_curvature.{ext}', dpi=180)
    plt.close(fig)


def run(args):
    from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
    sys.path[:0] = [str(args.source_root / 'source/contractionRL'), str(args.source_root / 'scripts/skrl')]
    import gymnasium as gym
    try:
        import contractionRL.tasks.direct.common.tracking_tasks  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != 'contractionRL.tasks.direct.common.tracking_tasks':
            raise
        # Historical checkpoint analysis loads the original source snapshot.
        import contractionRL.tasks.direct.classic  # noqa: F401
    from compare_baselines import load_actor
    import contractionRL
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'progress.csv').write_text('Steps,ElapsedSeconds\n0,0\n')
    with np.load(args.bank) as archive:
        bank = {key: archive[key].copy() for key in archive.files}
    total = len(bank['x0'])
    if not 1 <= args.episodes <= total or bool(bank['disturbance_enabled']):
        raise ValueError('Require a nominal bank with sufficient scenarios')
    bank = {key: value[:args.episodes] if value.ndim and len(value) == total else value for key, value in bank.items()}
    np.savez_compressed(args.output / 'evaluation_bank.npz', **bank)
    axis = np.linspace(-args.radius, args.radius, args.grid)
    summary = dict(status='RUNNING', task='Isaac-Classic-Car-v0', backend='analytical',
        state='(x, x*, xi); no preview or history', policy='deterministic mean control',
        objective='J = -trapezoidal integral_0^15 ||wrapped(x-x*)||_2 / max(||wrapped(x0-x*0)||_2,1e-8) dt',
        failure_handling='Failed episodes retained with absorbing box-diagonal tracking error; completion masks saved separately',
        parameter_scope='All active deterministic actor weights and biases; fixed variance, metric, critic and buffers',
        perturbation_scale='Absolute Euclidean norm, identical across methods; no reward rescaling or filter normalization',
        source_root=str(args.source_root), imported_package=str(contractionRL.__file__),
        bank_source=str(args.bank), bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        scenario_indices=list(range(args.episodes)), evaluation_seed=args.seed, direction_seed=args.direction_seed,
        grid=args.grid, radius=args.radius, h=args.h, directions=args.directions, device=args.device,
        started_at=datetime.now(timezone.utc).isoformat(), methods={},
        limitations=['One training seed; local directional estimates are not a Hessian spectral norm or conditioning estimate.',
                     'Clipping, wrapping and termination can make finite differences nonsmooth; h dependence is reported.',
                     'A two-dimensional slice does not establish full-space flatness or a causal training benefit.'])
    write_json(args.output / 'summary.json', summary)
    expected_layout = None
    start = time.monotonic()
    for name, checkpoint in args.checkpoint:
        folder = args.output / name
        folder.mkdir()
        cfg = yaml.safe_load((checkpoint.parent.parent / 'params/agent.yaml').read_text())
        spec = cfg['models']['policy']
        if spec.get('encoder', 'mlp') == 'gru':
            raise ValueError('Checkpoint must use compact state without future/history windows')
        env = make_analytical_plant('classic-car-v0', num_envs=args.batch_size * args.episodes, device=args.device,
                       disturbance_scales=[0.] * 4).unwrapped
        try:
            if not env.ref_window.compact or not np.isclose(float(bank['dt']), env.dt):
                raise ValueError('State layout or timestep mismatch')
            install_bank(env, bank, args.batch_size)
            model = MeanControl(load_actor(env, checkpoint)).to(args.device).eval()
            template = parameters(model)
            layout = [(key, list(value.shape)) for key, value in template.items()]
            if expected_layout is not None and layout != expected_layout:
                raise ValueError('Methods must have identical parameter layouts')
            expected_layout = layout
            theta = torch.cat([value.detach().flatten() for value in template.values()])
            ds = directions(len(theta), args.directions, args.direction_seed).to(args.device)
            np.savez_compressed(folder / 'directions.npz', directions=ds.cpu().numpy())
            torch.save(dict(theta=theta.cpu(), layout=layout), folder / 'parameters.pt')
            row = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       network=spec['network'][0]['layers'], parameters=len(theta), parameter_norm=float(theta.norm()),
                       training_algorithm=cfg['agent']['class'], layout=layout)
            def evaluate_points(deltas, label):
                returns, complete = [], []
                for offset in range(0, len(deltas), args.batch_size):
                    delta = deltas[offset:offset + args.batch_size]
                    size = len(delta)
                    if size < args.batch_size:
                        delta = torch.cat([delta, delta[:1].expand(args.batch_size - size, -1)])
                    result, mask = rollout(env, model, theta + delta, template, bank, args.seed)
                    returns.append(result[:size]); complete.append(mask[:size])
                    write_json(args.output / 'state.json', dict(method=name, phase=label,
                        current_points=offset + size, target_points=len(deltas), elapsed_seconds=time.monotonic()-start,
                        observed_at=datetime.now(timezone.utc).isoformat()))
                    points_per_method = args.grid**2 + 2 * args.directions * len(args.h)
                    points = len(summary['methods']) * points_per_method + offset + size
                    if label == 'curvature':
                        points += args.grid**2
                    with (args.output / 'progress.csv').open('a') as progress:
                        progress.write(f'{points},{time.monotonic()-start}\n')
                    print(name, label, offset + size, '/', len(deltas), flush=True)
                return np.concatenate(returns), np.concatenate(complete)
            # Repeat the center and compare batched evaluation against direct actor execution.
            zero = torch.zeros(args.batch_size, len(theta), device=args.device)
            center, center_mask = rollout(env, model, theta + zero, template, bank, args.seed)
            repeated, repeated_mask = rollout(env, model, theta + zero, template, bank, args.seed)
            np.testing.assert_array_equal(center, repeated)
            np.testing.assert_allclose(center, np.repeat(center[:1], args.batch_size, axis=0), rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(center_mask, repeated_mask)
            obs = env.construct_state_flat(env.x_t)
            torch.testing.assert_close(model(*model.features(obs)), model.actor.compute({'observations': obs})[0], rtol=0, atol=0)
            row.update(center_return=float(center[0].mean()), center_episode_stddev=float(center[0].std()),
                       center_completion=float(center_mask[0].mean()), repeat_check='bitwise identical across repeated calls',
                       batch_slot_return_noise_max=float(np.abs(center-center[:1]).max()))
            deltas = torch.stack([float(a)*ds[0] + float(b)*ds[1] for a in axis for b in axis])
            returns, complete = evaluate_points(deltas, 'surface')
            midpoint = (args.grid**2 - 1) // 2
            np.testing.assert_allclose(returns[midpoint], center[0], rtol=1e-6, atol=1e-7)
            row['surface_completion_min'] = float(complete.mean(-1).min())
            np.savez_compressed(folder / 'surface.npz', axis=axis,
                returns=returns.reshape(args.grid, args.grid, -1), completed=complete.reshape(args.grid, args.grid, -1))
            deltas = torch.stack([sign*h*d for h in args.h for d in ds[2:] for sign in (1, -1)])
            returns, complete = evaluate_points(deltas, 'curvature')
            paired = returns.reshape(len(args.h), args.directions, 2, args.episodes)
            estimates = np.stack([curvature([paired[i, :, 0], center[0], paired[i, :, 1]], h)
                                  for i, h in enumerate(args.h)])
            np.savez_compressed(folder / 'curvature.npz', h=args.h, curvature=estimates,
                               returns=paired, center_returns=center[0], completed=complete.reshape(paired.shape))
            row['curvature'] = []
            for h, values in zip(args.h, estimates):
                absolute = np.abs(values.mean(-1))
                row['curvature'].append(dict(h=h, median=float(np.median(absolute)),
                    p90=float(np.quantile(absolute, .9)), maximum=float(absolute.max()),
                    paired_standard_error=(values.std(-1, ddof=1) / np.sqrt(args.episodes)).tolist()))
            summary['methods'][name] = row
            write_json(args.output / 'summary.json', summary)
        finally:
            env.close()
    plot(args.output, summary['methods'], axis, args.h)
    summary.update(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                   completed_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output / 'summary.json', summary)
    lines = ['# Isaac-Classic-Car parameter landscape', '',
             'Existing matched 256x256 PPO/C2RL checkpoints: 500,000 training steps, seed 42, compact state (x, x*, xi).', '',
             f'Each point uses the same {args.episodes} nominal scenarios. J is negative normalized tracking AUC; higher is better. '
             'Historical convergence plots use the original evaluation stream and are context, not the fixed landscape bank.', '',
             '| Method | Center J | Center completion | Median absolute curvature at h = ' + ', '.join(map(str, args.h)) + ' |',
             '|---|---:|---:|---|']
    for name, row in summary['methods'].items():
        medians = ', '.join(f'{item["median"]:.6g}' for item in row['curvature'])
        lines.append(f'| {name} | {row["center_return"]:.6g} | {row["center_completion"]:.1%} | {medians} |')
    lines += ['', 'Axes use shared orthonormal unit directions in all active mean-policy weights and biases, '
              'with identical absolute L2 scales across methods. These are unnormalized parameter slices, '
              'not filter-normalized slices. Exploration variances, critics, metrics and buffers remain fixed.', '',
              'Failed trajectories are retained using an absorbing box-diagonal error; masks are saved. '
              'Kinks can come from clipping, wrapping, termination, or that penalty. '
              'Curvature estimates and paired scenario standard errors are in summary.json and curvature.npz.', '',
              *summary['limitations']]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    target = len(summary['methods']) * (args.grid**2 + 2 * args.directions * len(args.h))
    (args.output / 'completed.csv').write_text(f'Steps,Status\n{target},COMPLETED\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--checkpoint', nargs=2, action='append', metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--grid', type=int, default=31)
    parser.add_argument('--radius', type=float, default=5.)
    parser.add_argument('--directions', type=int, default=32)
    parser.add_argument('--h', nargs='+', type=float, default=[.1, .3, 1.])
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--direction-seed', type=int, default=20261019)
    args = parser.parse_args()
    if (args.grid < 3 or args.grid % 2 == 0 or args.radius <= 0 or args.directions < 1
            or args.batch_size < 1 or args.episodes < 2 or args.threads < 1
            or not all(np.isfinite(h) and h > 0 for h in args.h)):
        parser.error('Require an odd grid >=3, positive scales/counts, and >=2 episodes')
    names = [name for name, _ in args.checkpoint]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ('.', '..') for name in names):
        parser.error('Require distinct, simple method names')
    args.checkpoint = [(name, Path(path).resolve()) for name, path in args.checkpoint]
    args.source_root, args.bank, args.output = args.source_root.resolve(), args.bank.resolve(), args.output.resolve()
    run(args)
