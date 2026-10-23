"""Evaluate both archived rewards in one fixed plane defined by early training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from parameter_landscape import MeanControl, install_bank, parameters, write_json
from early_reward_landscape import load_metric, prepare_bank, rollout


def training_plane(initial, ppo, c2rl):
    first, second = ppo - initial, c2rl - initial
    lengths = [float(first.norm()), float(second.norm())]
    if min(lengths) < 1e-10:
        raise ValueError('Training displacements cannot define a two-dimensional plane')
    first = first / first.norm()
    cosine = float(first.dot(second / second.norm()))
    second = second - second.dot(first) * first
    if second.norm() < 1e-10:
        raise ValueError('Training displacements cannot define a two-dimensional plane')
    second /= second.norm()
    plane = torch.stack([first, second])
    torch.testing.assert_close(plane @ plane.T, torch.eye(2, dtype=plane.dtype), atol=1e-10, rtol=1e-10)
    return plane, dict(displacement_norms=lengths, displacement_cosine=cosine,
                      definition='d1 = PPO 0-to-20k displacement; d2 = orthogonalized C2RL 0-to-20k displacement')


@torch.no_grad()
def scan(args):
    from contractionRL.tasks.direct.common.tracking_tasks import make_analytical_plant
    root, output = args.root.resolve(), args.output.resolve()
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
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    bank = prepare_bank(root, 64)
    np.savez_compressed(output / 'evaluation_bank.npz', **bank)
    reference = json.loads((root / 'analysis-preview/summary.json').read_text())
    axis = np.linspace(-20., 20., 25)
    fine = np.linspace(-1., 1., 21)
    cuts = np.unique(np.concatenate([axis, fine]))
    summary = dict(status='RUNNING', purpose=__doc__, seed=42, machine='uiuc-cc',
        job_id=args.job_id, source_root=str(root), parameter_scope=reference['parameter_scope'],
        objectives=reference['objectives'], fixed_initial_return_rms=reference['fixed_initial_return_rms'],
        grid=25, radii=[5, 10, 20], grid_spacing=float(axis[1]-axis[0]),
        scenarios=64, precision='float64', gamma=.99, rows=[],
        limitations=['One training seed; a two-dimensional slice cannot characterize the full Hessian.',
            'Coarse 2D grid: 7, 13 and 25 samples per axis at radii 5, 10 and 20; no claim about sub-grid extrema.',
            'PPO optimizes a clipped surrogate; these figures show rollout-return objectives.',
            'Exploration variance stays at each checkpoint and can change between checkpoints.'],
        started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'summary.json', summary)
    env = make_analytical_plant('classic-car-v0', num_envs=64*args.batch_size, device='cpu', disturbance_scales=[0.]*4).unwrapped
    started = time.monotonic()
    progress = 0
    (output / 'progress.csv').write_text('Steps,ElapsedSeconds\n0,0\n')
    try:
        for key, value in list(vars(env).items()):
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(env, key, value.double())
        install_bank(env, bank, args.batch_size)
        def checkpoint(step, method):
            return root / f'training/{method}/snapshots/{step:06d}/checkpoints/agent.pt'
        def actor_vector(step, method):
            actor = load_actor(env, checkpoint(step, method)).double().eval()
            model = MeanControl(actor)
            template = parameters(model)
            return actor, model, template, torch.cat([p.detach().flatten() for p in template.values()])
        initial = actor_vector(0, 'ppo')[3]
        ds, metadata = training_plane(initial, actor_vector(20000, 'ppo')[3], actor_vector(20000, 'c2rl_ppo')[3])
        summary['plane'] = metadata
        np.savez_compressed(output / 'directions.npz', directions=ds.numpy())
        load_metric(env, root, 'cpu')
        total = 7*(len(axis)**2+2*(len(cuts)-len(axis)))
        summary['target_points'] = total
        for step in [0, 20000, 40000, 60000]:
            for method in (['ppo'] if step == 0 else ['ppo', 'c2rl_ppo']):
                actor, model, template, theta = actor_vector(step, method)
                folder = output / f'{step:06d}/{method}'
                folder.mkdir(parents=True)
                coordinates = [(float(a), float(b)) for a in axis for b in axis]
                coordinates += [(float(h), 0.) for h in cuts if h not in axis]
                coordinates += [(0., float(h)) for h in cuts if h not in axis]
                coordinates = list(dict.fromkeys(coordinates))
                values, masks = [], []
                for begin in range(0, len(coordinates), args.batch_size):
                    chunk = coordinates[begin:begin+args.batch_size]
                    coefficients = torch.tensor(chunk, dtype=torch.float64)
                    vectors = theta + coefficients @ ds
                    if len(chunk) < args.batch_size:
                        vectors = torch.cat([vectors, vectors[:1].expand(args.batch_size-len(chunk), -1)])
                    result, mask = rollout(env, model, vectors, template, bank, actor.log_std_parameter, .99)
                    values.extend(result[:len(chunk)])
                    masks.extend(mask[:len(chunk)])
                    progress += len(chunk)
                    state = dict(status='RUNNING', step=step, method=method, current_points=progress,
                        target_points=total, elapsed_seconds=time.monotonic()-started,
                        observed_at=datetime.now(timezone.utc).isoformat())
                    write_json(output / 'state.json', state)
                    with (output / 'progress.csv').open('a') as stream:
                        stream.write(f'{progress},{state["elapsed_seconds"]}\n')
                    print(json.dumps(state), flush=True)
                values, masks = np.stack(values), np.stack(masks)
                lookup = dict(zip(coordinates, range(len(coordinates))))
                center = values[lookup[0., 0.]]
                with np.load(root / f'analysis-preview/{step:06d}/{method}/curvature.npz') as original:
                    np.testing.assert_allclose(center, original['center_returns'], rtol=1e-6, atol=1e-7)
                np.savez_compressed(folder / 'surface.npz', axis=axis,
                    returns=values[:len(axis)**2].reshape(len(axis),len(axis),64,3),
                    completed=masks[:len(axis)**2].reshape(len(axis),len(axis),64))
                np.savez_compressed(folder / 'cuts.npz', axis=cuts, center_returns=center,
                    returns=np.stack([[values[lookup[float(h),0.]], values[lookup[0.,float(h)]]] for h in cuts]),
                    completed=np.stack([[masks[lookup[float(h),0.]], masks[lookup[0.,float(h)]]] for h in cuts]))
                summary['rows'].append(dict(step=step, method=method, path=str(folder.relative_to(output)),
                    surface=True, center_mean=center.mean(0).tolist(),
                    completion_min=float(masks.mean(1).min()),
                    checkpoint_sha256=hashlib.sha256(checkpoint(step, method).read_bytes()).hexdigest()))
                write_json(output / 'summary.json', summary)
        summary.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output / 'summary.json', summary)
        (output / 'completed.csv').write_text(f'Steps,Status\n{progress},COMPLETED\n')
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--job-id')
    scan(parser.parse_args())
