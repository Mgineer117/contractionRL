"""Cross identity/CCM reward landscapes along a matched car training replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.func import functional_call, vmap
import yaml

from parameter_landscape import MeanControl, directions, install_bank, parameters, unflatten, write_json

OBJECTIVES = ['identity', 'ccm', 'negative_auc']


def cached_evaluation(coefficients, cache, batch_size, evaluate, progress):
    """Reuse preview points when filling the full, predeclared grid."""
    keys = [tuple((int(i), float(c)) for i, c in item if c != 0) for item in coefficients]
    for begin in range(0, len(keys), batch_size):
        chunk = keys[begin:begin + batch_size]
        missing = list(dict.fromkeys(key for key in chunk if key not in cache))
        if missing:
            values, masks = evaluate(missing)
            for key, value, mask in zip(missing, values, masks):
                cache[key] = (value.copy(), mask.copy())
        progress(len(chunk), len(missing))
    return np.stack([cache[key][0] for key in keys]), np.stack([cache[key][1] for key in keys])


def load_metric(env, root, device):
    from contractionRL.agents.utils.RL.models import MetricModel
    cfg = yaml.safe_load((root / 'metric_config.yaml').read_text())['cm']
    metric = MetricModel(env.observation_space, env.action_space, device,
        hidden_dim=[128, 128], angle_idx=list(env.angle_idx), sym=None,
        constrain_eigenvalues=True, w_lb=cfg['w_lb'], w_ub=cfg['w_ub'], outputs_metric=True)
    metric.load_state_dict(torch.load(root / 'metric.pt', map_location=device, weights_only=False)['cmg'], strict=True)
    metric.double().requires_grad_(False).eval()
    env.set_ccm(metric.ccm_gen, cfg['w_lb'], device, tracking_scaler=1., control_scaler=0.)
    env.M = torch.zeros(env.num_envs, env.num_dim_x, env.num_dim_x, device=device, dtype=torch.float64)
    env.init_maha_error = torch.zeros(env.num_envs, device=device, dtype=torch.float64)
    return metric


def prepare_bank(root, episodes):
    from contractionRL.tasks.direct.common.disturbance import BoundedGust
    with np.load(root / 'evaluation_bank.npz') as archive:
        data = {key: archive[key].copy() for key in archive.files}
    total = len(data['x0'])
    data = {key: value[:episodes] if value.ndim and len(value) == total else value for key, value in data.items()}
    torch.manual_seed(20261024)
    gust = BoundedGust(episodes, [0., 0., .02, .05], .03, 'cpu')
    gust.reset(torch.arange(episodes))
    data['gusts'] = torch.stack([gust.advance() for _ in range(500)], dim=1).numpy()
    data['action_noise'] = torch.randn(episodes, 500, 2, generator=torch.Generator().manual_seed(20261025)).numpy()
    data = {key: value.astype(np.float64) if np.issubdtype(value.dtype, np.floating) else value
            for key, value in data.items()}
    return data


@torch.no_grad()
def rollout(env, model, vectors, template, bank, log_std, gamma):
    env.reset()
    batch, episodes = len(vectors), len(bank['x0'])
    ref = torch.as_tensor(bank['xref'], device=env.device, dtype=env.x_t.dtype).repeat(batch, 1, 1)
    gusts = torch.as_tensor(bank['gusts'], device=env.device, dtype=env.x_t.dtype).repeat(batch, 1, 1)
    noise = torch.as_tensor(bank['action_noise'], device=env.device, dtype=env.x_t.dtype).repeat(batch, 1, 1)
    initial = env.wrap_angles(env.x_t - ref[:, 0]).norm(dim=-1).double().clamp_min(1e-8)
    previous = torch.ones_like(initial)
    worst = (env.X_MAX - env.X_MIN).norm().double() / initial
    returns = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float64)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cursor = [0]
    def advance():
        return gusts[:, cursor[0]]
    env.disturbance.advance = advance
    theta = unflatten(vectors, template)
    def mean(params, features, error, control):
        return functional_call(model, params, (features, error, control), strict=False)
    for k in range(env.max_episode_len):
        cursor[0] = k
        previous_error = env.wrap_angles(env.x_t - ref[:, k])
        obs = env.construct_state_flat(env.x_t)
        features = [value.reshape(batch, episodes, -1) for value in model.features(obs)]
        action = vmap(mean, randomness='error')(theta, *features).reshape(env.num_envs, -1)
        action += log_std.clamp(-6.9, 2.).exp() * noise[:, k]
        finite = torch.isfinite(action).all(-1)
        _, ccm_reward, terminated, truncated, info = env.step(action.nan_to_num())
        next_error = env.wrap_angles(info['x'] - ref[:, k + 1])
        identity_reward = previous_error.square().sum(-1) - next_error.square().sum(-1)
        reward = torch.stack([identity_reward, ccm_reward.reshape(-1)], -1).double()
        if not torch.isfinite(reward[alive]).all():
            raise RuntimeError('Nonfinite actual reward on a live first episode')
        returns[:, :2] += torch.where(alive[:, None], reward, 0.) * gamma**k
        error = next_error.norm(dim=-1).double() / initial
        early = terminated | (truncated & (k + 1 < env.max_episode_len)) | ~finite | ~torch.isfinite(error)
        alive &= ~early
        current = torch.where(alive, error, worst)
        returns[:, 2] -= .5 * env.dt * (previous + current)
        previous = current
    return returns.reshape(batch, episodes, 3).cpu().numpy(), alive.reshape(batch, episodes).cpu().numpy()


def visualize(output, rows, scales, axis, hs, preview=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    times = sorted({row['step'] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for method in ['ppo', 'c2rl_ppo']:
        selected = [row for row in rows if row['method'] == method]
        axes[0].plot([r['step'] for r in selected], [-r['center_mean'][2] for r in selected], 'o-', label=method)
        for j, reward in enumerate(OBJECTIVES[:2], 1):
            for hi, h in enumerate(hs):
                ys = [r['curvature'][hi][reward]['median_absolute'] / scales[j-1] for r in selected]
                axes[j].plot([r['step'] for r in selected], ys, 'o-', label=f'{method}, h={h:g}')
    axes[0].set(title='Common tracking performance', xlabel='Training steps', ylabel='AUC (lower is better)', yscale='log')
    for j, reward in enumerate(OBJECTIVES[:2], 1):
        axes[j].set(title=f'{reward}: fixed-scale curvature', xlabel='Training steps', ylabel='Median absolute curvature')
        axes[j].set_yscale('symlog', linthresh=1e-5)
    for ax in axes:
        ax.legend(fontsize=7)
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'curvature_over_training.{extension}', dpi=180)
    plt.close(fig)
    a, b = np.meshgrid(axis, axis, indexing='ij')
    surface_times = sorted({row['step'] for row in rows if row['surface']})
    if surface_times:
        # One scale across rewards, policy centers AND time makes evolution comparable.
        surfaces = {(row['step'], row['method']):
                    np.load(output / row['path'] / 'surface.npz')['returns'].mean(2)[..., :2] / scales[:2]
                    for row in rows if row['surface']}
        low = min(z.min() for z in surfaces.values())
        high = max(z.max() for z in surfaces.values())
        norm = Normalize(low, high)
        for objective, name in enumerate(OBJECTIVES[:2]):
            fig = plt.figure(figsize=(max(8, 4 * len(surface_times)), 8), layout='constrained')
            for column, step in enumerate(surface_times):
                for index, method in enumerate(['ppo', 'c2rl_ppo']):
                    if (step, method) not in surfaces:
                        continue
                    ax = fig.add_subplot(2, len(surface_times), index*len(surface_times)+column+1, projection='3d')
                    ax.plot_surface(a, b, surfaces[step, method][..., objective], cmap='viridis', norm=norm)
                    ax.set(title=f'{method}, step {step:,}', xlabel='a', ylabel='b',
                           zlabel='Return / initial RMS', zlim=(low, high))
            fig.suptitle(f'{name} reward over training' + ('\nPreliminary 7×7 grid, 4 curvature directions' if preview else ''), fontsize=12)
            for extension in ['png', 'pdf']:
                fig.savefig(output / f'landscape_evolution_{name}.{extension}', dpi=160)
            plt.close(fig)
    for step in times:
        selected = [row for row in rows if row['step'] == step and row['surface']]
        if not selected:
            continue
        fig = plt.figure(figsize=(12, 10.5), layout='constrained')
        fig.set_constrained_layout_pads(hspace=.12)
        all_surfaces = [[np.load(output / row['path'] / 'surface.npz')['returns'].mean(2)[..., objective] / scales[objective]
                         for row in selected] for objective in range(2)]
        norm = Normalize(min(z.min() for values in all_surfaces for z in values),
                         max(z.max() for values in all_surfaces for z in values))
        for objective in range(2):
            surfaces = all_surfaces[objective]
            for method_index, (row, z) in enumerate(zip(selected, surfaces)):
                ax = fig.add_subplot(2, 2, objective*2 + method_index + 1, projection='3d')
                ax.plot_surface(a, b, z, cmap='viridis', norm=norm, linewidth=0)
                ax.set(title=f'{row["method"]}, {OBJECTIVES[objective]} reward, step {step}',
                       xlabel='a (parameter L2 units)', ylabel='b (parameter L2 units)', zlabel='Return / initial RMS')
                ax.set_zlim(norm.vmin, norm.vmax)
        for extension in ['png', 'pdf']:
            fig.savefig(output / f'reward_surfaces_{step:06d}.{extension}', dpi=180)
        plt.close(fig)


def analyze(args, preview=False, cache=None):
    from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
    root, output = args.root.resolve(), args.output.resolve()
    if preview:
        output = output.with_name(output.name + '-preview')
    if cache is None:
        cache = {}
    sys.path[:0] = [str(root / 'code/source/contractionRL'), str(root / 'code/scripts/skrl')]
    import gymnasium as gym
    try:
        import contractionRL.tasks.direct.common.tracking_tasks  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != 'contractionRL.tasks.direct.common.tracking_tasks':
            raise
        # Historical checkpoint analysis loads the original source snapshot.
        import contractionRL.tasks.direct.classic  # noqa: F401
    from compare_baselines import load_actor
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((root / 'plan.json').read_text())
    steps = [0, 64] if args.smoke else plan['analysis_checkpoints']
    surface_steps = [0] if args.smoke else plan['surface_steps']
    grid, count, episodes = (3, 2, 2) if args.smoke else (31, 32, 64)
    if preview:
        steps, grid, count = surface_steps, 7, 4
    hs, radius = [.1, .3, 1.], 5.
    bank = prepare_bank(root, episodes)
    np.savez_compressed(output / 'evaluation_bank.npz', **bank)
    axis = np.linspace(-radius, radius, 31)[::5] if preview else np.linspace(-radius, radius, grid)
    total = sum(2 * (2*count*len(hs) + (grid**2 if step in surface_steps else 0)) for step in steps)
    progress = 0
    (output / 'progress.csv').write_text('Steps,ElapsedSeconds\n0,0\n')
    rows, scales = [], None
    started = time.monotonic()
    layout = None
    summary = dict(status='RUNNING', objectives=OBJECTIVES, rows=rows, steps=steps, surface_steps=surface_steps,
        preview=preview, grid=grid, curvature_directions=count,
        scenarios=episodes, direction_seed=20261019, action_seed=20261025, gust_seed=20261024,
        gamma=.99, parameter_scope='All active mean-control weights/biases; exploration variance held at each checkpoint',
        evaluation='Stochastic first episodes, same fixed Gaussian draws and gusts across all points, checkpoints and methods',
        reward_definition='sum gamma^k [E(s_k)-E(s_k+1)]; E=eTe or eTM(x)e; exact archived training reward',
        failure_handling='Reward returns stop at first episode end; AUC uses absorbing box-error penalty. Masks retained; no episodes dropped.',
        calibration='One fixed RMS of each objective return at shared initialization, computed before inspecting later checkpoints',
        evaluation_precision='Float64 model, metric, plant state and rewards to limit cancellation in finite differences; training remains float32',
        limitations=['Single training seed; directional finite differences, not a Hessian norm or causal proof.',
                     'PPO optimizes a clipped surrogate; return geometry is only one part of its update geometry.',
                     'Conditional episode resets can diverge during training despite matched random streams.'])
    write_json(output / 'summary.json', summary)
    for step in steps:
        for method in ['ppo', 'c2rl_ppo']:
            folder = output / f'{step:06d}' / method
            folder.mkdir(parents=True)
            checkpoint = root / ('preflight' if args.smoke else 'training') / method / 'snapshots' / f'{step:06d}/checkpoints/agent.pt'
            env = make_analytical_plant('classic-car-v0', num_envs=args.batch_size*episodes, device=args.device,
                           disturbance_scales=[0.]*4).unwrapped
            try:
                for key, value in list(vars(env).items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        setattr(env, key, value.double())
                install_bank(env, bank, args.batch_size)
                actor = load_actor(env, checkpoint).to(args.device).double().eval()
                model = MeanControl(actor)
                metric = load_metric(env, root, args.device)
                template = parameters(model)
                current_layout = [(key, list(value.shape)) for key, value in template.items()]
                if layout is not None and layout != current_layout:
                    raise RuntimeError('Parameter layouts changed across checkpoints or methods')
                layout = current_layout
                theta = torch.cat([value.detach().flatten() for value in template.values()])
                ds = directions(len(theta), 2 if args.smoke else 32, 20261019).to(args.device, dtype=torch.float64)
                ds /= ds.norm(dim=1, keepdim=True)
                if not rows:
                    np.savez_compressed(output/'directions.npz', directions=ds.cpu().numpy())
                    summary['parameter_layout'] = current_layout
                vector = theta.repeat(args.batch_size, 1)
                center, mask = rollout(env, model, vector, template, bank, actor.log_std_parameter, .99)
                repeated, _ = rollout(env, model, vector, template, bank, actor.log_std_parameter, .99)
                np.testing.assert_array_equal(center, repeated)
                np.testing.assert_allclose(center, np.repeat(center[:1],args.batch_size,axis=0), rtol=1e-6,atol=1e-7)
                if scales is None:
                    scales = np.sqrt(np.mean(center[0]**2, axis=0))
                    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
                        raise RuntimeError('Degenerate fixed-scale calibration')
                    summary['fixed_initial_return_rms'] = dict(zip(OBJECTIVES, scales.tolist()))
                if step == 0 and method == 'c2rl_ppo':
                    first = np.load(output / '000000/ppo/curvature.npz')['center_returns']
                    np.testing.assert_allclose(center[0], first, rtol=1e-6, atol=1e-7)
                row_cache = cache.setdefault((step, method), {})
                row_cache[()] = (center[0], mask[0])
                def evaluate(coefficients, label):
                    nonlocal progress
                    def run_batch(chunk):
                        n = len(chunk)
                        offsets = torch.stack([sum(float(c)*ds[int(i)] for i,c in item) for item in chunk])
                        if n < args.batch_size:
                            offsets = torch.cat([offsets, offsets[:1].expand(args.batch_size-n, -1)])
                        result, valid = rollout(env, model, theta+offsets, template, bank, actor.log_std_parameter, .99)
                        return result[:n], valid[:n]
                    def record(n, evaluated):
                        nonlocal progress
                        progress += n
                        with (output / 'progress.csv').open('a') as stream:
                            stream.write(f'{progress},{time.monotonic()-started}\n')
                        write_json(output / 'state.json', dict(phase=label, method=method, checkpoint=step,
                            current_points=progress, target_points=total, preview=preview,
                            observed_at=datetime.now(timezone.utc).isoformat()))
                        print(step, method, label, progress, '/', total, 'new evaluations', evaluated, flush=True)
                    return cached_evaluation(coefficients, row_cache, args.batch_size, run_batch, record)
                surface = step in surface_steps
                if surface:
                    coords = [[(0,a),(1,b)] for a in axis for b in axis]
                    values, valid = evaluate(coords,'surface')
                    np.testing.assert_allclose(values[(grid**2-1)//2],center[0],rtol=1e-6,atol=1e-7)
                    np.savez_compressed(folder/'surface.npz',returns=values.reshape(grid,grid,episodes,3),completed=valid.reshape(grid,grid,episodes),axis=axis)
                coords = [[(d,sign*h)] for h in hs for d in range(2,count+2) for sign in [1,-1]]
                values,valid = evaluate(coords,'curvature')
                paired = values.reshape(len(hs),count,2,episodes,3)
                curv = np.stack([(paired[i,:,0]-2*center[0]+paired[i,:,1])/h**2 for i,h in enumerate(hs)])
                np.savez_compressed(folder/'curvature.npz',returns=paired,curvature=curv,center_returns=center[0],completed=valid.reshape(len(hs),count,2,episodes),h=hs)
                row = dict(step=step,method=method,path=str(folder.relative_to(output)),surface=surface,
                    center_mean=center[0].mean(0).tolist(),center_completion=float(mask[0].mean()),
                    batch_slot_noise_max=np.abs(center-center[:1]).max(axis=(0,1)).tolist(),
                    perturbation_completion_min=float(valid.mean(-1).min()),curvature=[])
                for i,h in enumerate(hs):
                    item = dict(h=h)
                    for j,name in enumerate(OBJECTIVES):
                        absolute = np.abs(curv[i,...,j].mean(-1))
                        item[name] = dict(median_absolute=float(np.median(absolute)),p90=float(np.quantile(absolute,.9)),
                                          maximum=float(absolute.max()),normalized_median=float(np.median(absolute)/scales[j]))
                    row['curvature'].append(item)
                rows.append(row)
                write_json(output/'summary.json',summary)
                if method == 'c2rl_ppo':
                    visualize(output, rows, scales, axis, hs, preview=preview)
            finally:
                env.close()
    visualize(output,rows,scales,axis,hs,preview=preview)
    summary.update(status='PRELIMINARY' if preview else 'COMPLETED',elapsed_seconds=time.monotonic()-started)
    write_json(output/'summary.json',summary)
    (output/'completed.csv').write_text(f'Steps,Status\n{total},COMPLETED\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cache = {}
    if not args.smoke:
        analyze(args, preview=True, cache=cache)
    analyze(args, cache=cache)
