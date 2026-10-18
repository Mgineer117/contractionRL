"""Paired final-checkpoint AUC evaluation for the analytical Isaac car ablation.

Training uses Isaac-Car-v0. Evaluation uses its identical analytical plant without
rendering, on one persisted fixed bank. Failed episodes are excluded from AUC
and counted explicitly; paired differences use only common complete episodes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/contractionRL'))

import gymnasium as gym
import numpy as np
import torch
import yaml
import contractionRL.tasks.direct.classic  # register analytical plant
from contractionRL.agents.utils.RL.models import FullControlActorModel


def summarize(values):
    values = np.asarray(values)
    return {'auc_mean': float(values.mean()) if len(values) else None,
            'auc_stddev': float(values.std()) if len(values) else None}


@torch.no_grad()
def compare(checkpoints, output, *, seed=20261015, episodes=512, wandb_project=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output/'evaluation_bank.npz'
    if bank_path.exists():
        raise FileExistsError(f'Refusing to overwrite comparison evidence: {bank_path}')
    torch.manual_seed(seed)
    env = gym.make('classic-car-v0', num_envs=episodes, device='cpu',
                   disturbance_scales=[0.]*4).unwrapped
    env.set_fix_ref_trajectories(True)
    env.reset()
    assert not env.disturbance.enabled
    bank = {'x0': env.x_t.numpy().copy(), 'xref': env.xref.numpy().copy(),
            'uref': env.uref.numpy().copy(), 'parameters': env._reference_parameters.numpy().copy()}
    np.savez_compressed(bank_path, **bank, seed=seed, dt=env.dt, disturbance_enabled=False)
    rows, curves, masks, metric_states = {}, {}, {}, []
    initial_observation = None
    for mode in ('full', 'residual'):
        path = Path(checkpoints[mode]).resolve()
        cfg = yaml.safe_load((path.parent.parent/'params/agent.yaml').read_text())
        spec = cfg['models']['policy']
        assert spec.get('control_mode', 'full') == mode
        assert not cfg['agent'].get('use_state_norm', False), 'Apply the saved state scaler before evaluating normalized runs'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        metric_states.append(checkpoint['cmg'])
        network = spec['network'][0]
        actor = FullControlActorModel(env.observation_space, env.action_space, 'cpu',
                   hidden_dim=network['layers'], activation=network['activations'], control_mode=mode)
        actor.load_state_dict(checkpoint['policy'])
        actor.eval()
        env.reset()
        obs = env.construct_state_flat(env.x_t)
        if initial_observation is None: initial_observation = obs.clone()
        else: torch.testing.assert_close(obs, initial_observation, rtol=0, atol=0)
        np.testing.assert_array_equal(env.x_t.numpy(), bank['x0'])
        np.testing.assert_array_equal(env.xref.numpy(), bank['xref'])
        n, steps = episodes, env.max_episode_len
        errors = torch.full((n, steps+1), float('nan'))
        errors[:, 0] = env.wrap_angles(env.x_t-env.xref[:, 0]).norm(dim=-1)
        alive = torch.ones(n, dtype=torch.bool)
        failed = torch.zeros_like(alive)
        for k in range(steps):
            mean, _ = actor.compute({'observations': obs})
            _, _, terminated, truncated, info = env.step(mean.clamp(env.U_MIN, env.U_MAX))
            error = env.wrap_angles(info['x']-torch.as_tensor(bank['xref'][:, k+1])).norm(dim=-1)
            errors[alive, k+1] = error[alive]
            early = terminated | (truncated & (k+1 < steps)) | ~torch.isfinite(error)
            failed |= alive & early
            alive &= ~early
            obs = env.construct_state_flat(env.x_t)
        normalized = errors/errors[:, :1].clamp_min(1e-8)
        auc = torch.trapezoid(normalized, dx=env.dt, dim=1).numpy()
        valid = alive.numpy() & np.isfinite(auc)
        curves[mode], masks[mode] = auc, valid
        np.savez_compressed(output/f'{mode}_curves.npz', normalized_error=normalized.numpy(),
                            auc=auc, completed=valid)
        digest = hashlib.sha256()
        for key, value in sorted(checkpoint['cmg'].items()):
            digest.update(key.encode()); digest.update(value.numpy().tobytes())
        rows[mode] = {**summarize(auc[valid]), 'completed_episodes': int(valid.sum()),
                      'failed_episodes': int(failed.sum()), 'episodes': episodes,
                      'episode_completion': float(valid.mean()),
                      'checkpoint': str(path), 'metric_sha256': digest.hexdigest()}
    assert metric_states[0].keys() == metric_states[1].keys()
    for key in metric_states[0]:
        torch.testing.assert_close(metric_states[0][key], metric_states[1][key], rtol=1e-5, atol=1e-6)
    common = masks['full'] & masks['residual']
    difference = curves['residual'][common]-curves['full'][common]
    result = {'task': 'Isaac-Car-v0', 'backend': 'analytical', 'evaluation_seed': seed,
              'disturbance_enabled': False, 'matched_metric': True,
              'auc_definition': 'integral_0^15 ||wrapped_error(t)|| / ||wrapped_error(0)|| dt',
              'failure_handling': 'AUC only for complete first episodes; failures reported; paired difference uses common complete episodes',
              'variants': rows, 'common_complete_episodes': int(common.sum()),
              'paired_residual_minus_full': summarize(difference)}
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['| Control | AUC mean | AUC stddev | Complete episodes |', '|---|---:|---:|---:|']
    for mode, row in rows.items():
        lines.append(f"| {mode} | {row['auc_mean']} | {row['auc_stddev']} | {row['completed_episodes']}/{episodes} |")
    lines += ['', 'Lower AUC is better. AUC excludes failed episodes; the paired difference uses only common complete episodes.',
              f'Evaluation seed: {seed}. Disturbances: off. Same metric, evaluation trajectories and initial states.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    if wandb_project:
        import wandb
        for mode, row in rows.items():
            with wandb.init(project=wandb_project, name=f'{mode}-paired-final-evaluation',
                            group=output.parent.name, job_type='paired-evaluation',
                            config={'control_mode': mode, 'disturbance_enabled': False,
                                    'evaluation_seed': seed, 'checkpoint': row['checkpoint']}) as run:
                from contractionRL.agents.utils.RL.logging import log_metrics
                payload = {f'Stability/{key}': row[key] for key in
                           ('auc_mean', 'auc_stddev') if row[key] is not None}
                payload['Learning/episode_completion'] = row['episode_completion']
                log_metrics(payload, step=0)
                rows[mode]['evaluation_url'] = run.url
        (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    env.close()
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', required=True)
    parser.add_argument('--residual', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--wandb_project')
    args = parser.parse_args()
    compare({'full': args.full, 'residual': args.residual}, args.output,
            seed=args.seed, episodes=args.episodes, wandb_project=args.wandb_project)
