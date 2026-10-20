"""Isolated short replay of archived car PPO/C2RL, preserving early weights."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import sys
import time

import numpy as np
import torch
import yaml


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RandomStream:
    """Keep environment and action draws independent of optimizer randomness."""
    def __init__(self, seed, device):
        self.device = torch.device(device)
        self.devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        self.depth = 0
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            if self.devices:
                torch.cuda.manual_seed_all(seed)
            self.cpu = torch.get_rng_state()
            self.cuda = torch.cuda.get_rng_state(self.device) if self.devices else None

    @contextmanager
    def use(self):
        if self.depth:
            yield
            return
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu)
            if self.devices:
                torch.cuda.set_rng_state(self.cuda, self.device)
            self.depth += 1
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                if self.devices:
                    self.cuda = torch.cuda.get_rng_state(self.device)
                self.depth -= 1


def train(args):
    root = args.root.resolve()
    folder = root / args.output
    folder.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root / 'code/source/contractionRL'), str(root / 'code/scripts/skrl')]
    import train_utils
    try:
        from contractionRL.tasks.direct.common.analytical_env import BaseEnv
    except ModuleNotFoundError as error:
        if error.name != 'contractionRL.tasks.direct.common.analytical_env':
            raise
        # Replays deliberately execute the checkpoint's original source snapshot.
        from contractionRL.tasks.direct.classic.common.env_base import BaseEnv
    from skrl.agents.torch.ppo import PPO
    torch.set_num_threads(args.threads)
    config = yaml.safe_load((root / 'configs' / f'{args.algorithm}.yaml').read_text())
    config['trainer']['timesteps'] = args.steps
    config['agent']['experiment'].update(checkpoint_interval=10000, write_interval=500,
                                         wandb=False, directory=str(folder / 'training'))
    (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(status='STARTING', algorithm=args.algorithm, seed=42, steps=args.steps,
                    num_envs=args.num_envs, device=args.device, pid=os.getpid(),
                    job_id=os.environ.get('SLURM_JOB_ID'), started_at=datetime.now(timezone.utc).isoformat(),
                    snapshots=[], accepted_updates=0, skipped_updates=0)
    write_json(folder / 'state.json', metadata)
    (folder / 'progress.csv').write_text('Steps,ElapsedSeconds\n0,0\n')
    (folder / 'learning.csv').write_text('Steps,TrackingAUC\n')
    start = time.monotonic()
    environment_stream = RandomStream(1042, args.device)
    action_stream = RandomStream(2042, args.device)
    original_reset, original_step = BaseEnv.reset, BaseEnv.step
    original_act, original_track = PPO.act, PPO.track_data
    original_patches = train_utils.apply_agent_patches
    initialized = []
    first_reset = []

    def reset(env, *positional, **keywords):
        with environment_stream.use():
            result = original_reset(env, *positional, **keywords)
        if initialized and not first_reset:
            first_reset.append(state_hash(dict(x0=env.x_t, xref=env.xref, parameters=env._reference_parameters)))
            metadata['initial_scenario_sha256'] = first_reset[0]
        return result

    def step(env, *positional, **keywords):
        with environment_stream.use():
            return original_step(env, *positional, **keywords)

    def act(agent, *positional, **keywords):
        with action_stream.use():
            return original_act(agent, *positional, **keywords)

    def track(agent, tag, value):
        if tag == 'Policy / KL divergence':
            key = 'accepted_updates' if float(value) > 0 else 'skipped_updates'
            metadata[key] += 1
        return original_track(agent, tag, value)

    def apply(agent, **keywords):
        original_patches(agent, **keywords)
        inner = getattr(agent, '_rl_agent', agent)
        models = {name: inner.models[name] for name in ('policy', 'value')}
        common = root / args.initialization
        if common.exists():
            state = torch.load(common, map_location='cpu', weights_only=False)
            for name, model in models.items():
                model.load_state_dict(state[name], strict=True)
        else:
            if args.algorithm != 'ppo':
                raise RuntimeError('PPO must establish the shared initialization first')
            state = {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                     for name, model in models.items()}
            torch.save(state, common)
        metadata['initial_model_sha256'] = {name: state_hash(model.state_dict()) for name, model in models.items()}
        if args.algorithm == 'c2rl_ppo':
            frozen = torch.load(root / 'metric.pt', map_location='cpu', weights_only=False)['cmg']
            agent.models['cmg'].load_state_dict(frozen, strict=True)
            agent._restored_cmg_checkpoint = str(root / 'metric.pt')
            agent.freeze_cmg()
            metadata['metric_sha256'] = state_hash(frozen)

        def snapshot(step):
            path = folder / 'snapshots' / f'{step:06d}'
            (path / 'checkpoints').mkdir(parents=True, exist_ok=False)
            shutil.copytree(Path(agent.experiment_dir) / 'params', path / 'params')
            agent.save(str(path / 'checkpoints/agent.pt'))
            metadata['snapshots'].append(step)
            write_json(path / 'metadata.json', dict(step=step, seed=42, algorithm=args.algorithm,
                initial_model_sha256=metadata['initial_model_sha256'], accepted_updates=metadata['accepted_updates'],
                skipped_updates=metadata['skipped_updates']))

        snapshot(0)
        original_post = agent.post_interaction
        def post(*, timestep, timesteps):
            values = agent.tracking_data.get('Stability/auc_mean') or inner.tracking_data.get('Stability/auc_mean')
            latest_auc = float(values[-1]) if values else None
            original_post(timestep=timestep, timesteps=timesteps)
            count = timestep + 1
            if count % 10000 == 0 or count == timesteps:
                if count not in metadata['snapshots']:
                    snapshot(count)
            if count % 500 == 0 or count == timesteps:
                metadata.update(status='TRAINING', current_steps=count,
                                elapsed_seconds=time.monotonic()-start,
                                observed_at=datetime.now(timezone.utc).isoformat())
                write_json(folder / 'state.json', metadata)
                with (folder / 'progress.csv').open('a') as stream:
                    stream.write(f'{count},{time.monotonic()-start}\n')
            if count % 500 == 0 and latest_auc is not None:
                with (folder / 'learning.csv').open('a') as stream:
                    stream.write(f'{count},{latest_auc}\n')
        agent.post_interaction = post
        initialized.append(agent)
        torch.manual_seed(3042)
        np.random.seed(42)
        random.seed(42)

    BaseEnv.reset, BaseEnv.step = reset, step
    PPO.act, PPO.track_data = act, track
    train_utils.apply_agent_patches = apply
    original_directory = Path.cwd()
    sys.argv = [str(root / 'code/scripts/skrl/train.py'), '--classic', '--task', 'classic-car-v0',
                '--cfg', str(folder / 'config.yaml'), '--algorithm', args.algorithm, '--num_envs', str(args.num_envs),
                '--seed', '42', '--num_timesteps', str(args.steps), '--device', args.device,
                '--skip_final_eval', '--no_wandb']
    try:
        os.chdir(folder)
        runpy.run_path(sys.argv[0], run_name='__main__')
        if metadata['accepted_updates'] < 1:
            raise RuntimeError('Replay accepted no PPO policy updates')
        if args.algorithm == 'c2rl_ppo':
            actual = state_hash(initialized[0].models['cmg'].state_dict())
            if actual != metadata['metric_sha256']:
                raise RuntimeError('Frozen metric changed during replay')
        metadata.update(status='COMPLETED', current_steps=args.steps, elapsed_seconds=time.monotonic()-start)
        (folder / 'completed.csv').write_text(f'Steps,Status\n{args.steps},COMPLETED\n')
        return metadata
    except Exception as error:
        metadata.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(folder / 'state.json', metadata)
        os.chdir(original_directory)
        BaseEnv.reset, BaseEnv.step = original_reset, original_step
        PPO.act, PPO.track_data = original_act, original_track
        train_utils.apply_agent_patches = original_patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--algorithm', choices=['ppo', 'c2rl_ppo'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initialization', default='initialization.pt')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 16 or args.steps % 16 or args.num_envs < 1:
        parser.error('Require positive environments and steps divisible by the 16-step rollout')
    train(args)
