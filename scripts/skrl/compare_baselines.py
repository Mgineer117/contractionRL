"""Evaluate final Isaac analytical-car checkpoints on the persisted ablation bank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/contractionRL'))

import gymnasium as gym
import numpy as np
import torch
import yaml
import contractionRL.tasks.direct.classic
from contractionRL.agents.skrl.contraction_ppo import ContractionPPOPolicy
from contractionRL.agents.utils.RL.models import FullControlActorModel
from compare_residual import summarize


def load_actor(env, path):
    path = Path(path)
    cfg = yaml.safe_load((path.parent.parent/'params/agent.yaml').read_text())
    spec = cfg['models']['policy']
    from train_utils import configure_tracking_preview
    configure_tracking_preview(env, cfg)
    if spec['class'] == 'ContractionPPOPolicy':
        actor = ContractionPPOPolicy(env.observation_space, env.action_space, 'cpu', network=spec['network'],
                                     control_mode=spec.get('control_mode', 'full'))
    else:
        assert spec['class'] == 'GaussianMixin' and spec.get('backbone') == 'control'
        network = spec['network'][0]
        actor = FullControlActorModel(env.observation_space, env.action_space, 'cpu',
                                     hidden_dim=network['layers'], activation=network['activations'],
                                     control_mode=spec.get('control_mode', 'full'),
                                     encoder=spec.get('encoder', 'mlp'), encoder_hidden=spec.get('encoder_hidden', 64),
                                     encoder_stride=spec.get('encoder_stride', 1))
    state = torch.load(path, map_location='cpu', weights_only=False)
    if cfg['agent'].get('use_state_norm', False):
        from contractionRL.agents.utils.RL.env import RunningFeatureScaler
        model = getattr(actor, 'cl_actor', actor)
        model.feature_scaler = RunningFeatureScaler(model.tracking_features.output_dim)
    actor.load_state_dict(state['policy'])
    actor.eval()
    return actor


def install_bank(env, bank):
    """Install generator parameters as well as trajectories for compact observations."""
    assert not env.disturbance.enabled
    assert np.isclose(float(bank['dt']), env.dt)
    assert not bool(bank['disturbance_enabled'])
    env.set_fix_ref_trajectories(True)
    for key, target in [('x0', env._fixed_x0), ('xref', env._fixed_xref),
                        ('uref', env._fixed_uref), ('parameters', env._fixed_parameters)]:
        value = torch.as_tensor(bank[key], dtype=target.dtype, device=target.device)
        assert value.shape == target.shape and torch.isfinite(value).all(), key
        target.copy_(value)
    env._ref_fixed[:] = True
    env.reset()
    np.testing.assert_array_equal(env.x_t.numpy(), bank['x0'])
    np.testing.assert_array_equal(env.xref.numpy(), bank['xref'])
    np.testing.assert_array_equal(env._reference_parameters.numpy(), bank['parameters'])


@torch.no_grad()
def evaluate(env, actor, bank):
    env.reset()
    n, steps = env.num_envs, env.max_episode_len
    errors = torch.full((n, steps+1), float('nan'))
    errors[:, 0] = env.wrap_angles(env.x_t-env.xref[:, 0]).norm(dim=-1)
    alive = torch.ones(n, dtype=torch.bool)
    for k in range(steps):
        obs = env.construct_state_flat(env.x_t)
        mean, _ = actor.compute({'observations': obs})
        _, _, terminated, truncated, info = env.step(mean.clamp(env.U_MIN, env.U_MAX))
        error = env.wrap_angles(info['x']-torch.as_tensor(bank['xref'][:, k+1])).norm(dim=-1)
        errors[alive, k+1] = error[alive]
        early = terminated | (truncated & (k+1 < steps)) | ~torch.isfinite(error)
        alive &= ~early
    normalized = errors/errors[:, :1].clamp_min(1e-8)
    auc = torch.trapezoid(normalized, dx=env.dt, dim=1).numpy()
    complete = alive.numpy() & np.isfinite(auc)
    return normalized.numpy(), auc, complete


@torch.no_grad()
def compare(manifest, checkpoint, output):
    entry = json.loads(Path(manifest).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bank_path = Path(entry['evaluation_bank'])
    with np.load(bank_path) as loaded:
        bank = {k: loaded[k].copy() for k in loaded.files}
    n = len(bank['x0'])
    env = gym.make('classic-car-v0', num_envs=n, device='cpu', disturbance_scales=[0.]*4).unwrapped
    install_bank(env, bank)
    paths = dict(entry['previous_checkpoints'])
    paths['contraction_ppo'] = {'checkpoint': str(Path(checkpoint).resolve()),
        'training_disturbance': False, 'training_steps': 100000, 'training_num_envs': 128, 'training_seed': 42}
    rows, curves, masks = {}, {}, {}
    for name, metadata in paths.items():
        actor = load_actor(env, metadata['checkpoint'])
        normalized, auc, complete = evaluate(env, actor, bank)
        curves[name], masks[name] = auc, complete
        np.savez_compressed(output/f'{name}_curves.npz', normalized_error=normalized, auc=auc, completed=complete)
        rows[name] = {**metadata, **summarize(auc[complete]), 'episodes': n,
                      'completed_episodes': int(complete.sum()), 'failed_episodes': int((~complete).sum()),
                      'episode_completion': float(complete.mean())}
        print(name, rows[name], flush=True)
    # The prior ablation must reproduce before its numbers enter this comparison.
    previous = json.loads(Path(entry['previous_comparison']).read_text())
    for mode in ('full', 'residual'):
        for key in ('auc_mean', 'auc_stddev', 'episode_completion'):
            np.testing.assert_allclose(rows[f'c2rl_{mode}'][key], previous['variants'][mode][key], rtol=1e-6, atol=1e-6)
    paired = {}
    for name in entry['previous_checkpoints']:
        common = masks[name] & masks['contraction_ppo']
        paired[name] = {**summarize(curves['contraction_ppo'][common]-curves[name][common]),
                        'common_complete_episodes': int(common.sum())}
    result = {'task': 'Isaac-Car-v0', 'backend': 'analytical', 'evaluation_seed': int(bank['seed']),
              'evaluation_bank': str(bank_path), 'evaluation_bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              'disturbance_enabled': False, 'auc_definition': previous['auc_definition'],
              'failure_handling': previous['failure_handling'], 'variants': rows,
              'paired_contractionppo_minus_baseline': paired,
              'limitations': ['One training seed; stddev is across evaluation episodes, not training seeds.',
                             'Historical PPO trained with disturbances; C2RL and ContractionPPO trained without them.',
                             'Released ContractionPPO network/hyperparameters differ from local PPO/C2RL defaults.']}
    def save():
        (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    import wandb
    from contractionRL.agents.utils.RL.logging import log_metrics
    for name, row in rows.items():
        with wandb.init(project='isaac-car-baselines', name=f'{name}-paired-final-evaluation',
                        group=Path(manifest).parent.name, job_type='paired-evaluation',
                        config={**row, 'disturbance_enabled': False, 'evaluation_seed': int(bank['seed']),
                                'evaluation_bank_sha256': result['evaluation_bank_sha256']}) as run:
            payload = {f'Stability/{key}': row[key] for key in ('auc_mean', 'auc_stddev') if row[key] is not None}
            payload['Learning/episode_completion'] = row['episode_completion']
            log_metrics(payload, step=0)
            row['evaluation_url'] = run.url
    save()
    lines = ['| Algorithm | AUC mean | AUC stddev | Complete episodes | Training disturbance |',
             '|---|---:|---:|---:|---|']
    for name, row in rows.items():
        lines.append(f"| {name} | {row['auc_mean']} | {row['auc_stddev']} | {row['completed_episodes']}/{n} | {row['training_disturbance']} |")
    lines += ['', 'Lower AUC is better. Evaluation uses the exact saved 512-case bank and no disturbance.', *result['limitations']]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    env.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--new_checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.manifest, args.new_checkpoint, args.output)
