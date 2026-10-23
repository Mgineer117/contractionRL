"""One isolated W&B Bayesian trial using the existing analytical training pipeline."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import threading
import time
import traceback

import yaml

# Resolve models from this checkout before an editable installation can load them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/contractionRL"))
from car_bayes_runtime import evaluate_sweep

OBJECTIVE = 'Evaluation/Stability/validation_auc_mean'
BAD_OBJECTIVE = 1e6


def resolve_config(base, parameters):
    cfg = copy.deepcopy(base)
    low, high = map(float, parameters['ccm_bounds'].split(':'))
    if not 0 < low < high or not all(map(math.isfinite, (low, high))):
        raise ValueError('Invalid paired CCM spectral bounds')
    cfg['cm'].update(ccm_w_lb=low, ccm_w_ub=high)
    pair = parameters.get('encoder_pair')
    if pair is not None:
        pairs = {'null:null': (None, None), 'null:gru': (None, 'gru'), 'gru:gru': ('gru', 'gru')}
        if pair not in pairs:
            raise ValueError('Unknown policy/critic encoder pair')
        if any(key in parameters for key in ('models.policy.encoder', 'models.critic.encoder')):
            raise ValueError('encoder_pair cannot be combined with individual encoder overrides')
        cfg['models']['policy']['encoder'], cfg['models']['critic']['encoder'] = pairs[pair]
    for key, value in parameters.items():
        if key.startswith(('agent.', 'models.')):
            *parents, leaf = key.split('.')
            node = cfg
            for parent in parents:
                if parent not in node:
                    raise ValueError('Unknown parameter: ' + key)
                node = node[parent]
            if leaf not in node:
                raise ValueError('Unknown parameter: ' + key)
            node[leaf] = value
        elif key not in {'ccm_bounds', 'encoder_pair'}:
            raise ValueError('Unknown parameter: ' + key)
    from contractionRL.agents.utils.RL.config import validate_experiment_config
    validate_experiment_config(cfg)
    cfg['agent']['value_expansion_horizon'] = 10
    for key in ('use_value_norm', 'use_value_expansion'):
        if not isinstance(cfg['agent'][key], bool):
            raise ValueError(key + ' must be boolean')
    for key in ('rollouts', 'mini_batches', 'learning_epochs'):
        if not isinstance(cfg['agent'][key], int) or isinstance(cfg['agent'][key], bool) or cfg['agent'][key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if cfg['agent']['rollouts'] % cfg['agent']['mini_batches']:
        raise ValueError('mini_batches must divide rollouts')
    cfg['_tuned_parameters'] = [key for key in parameters if key.startswith(('agent.','models.'))]
    if pair is not None:
        cfg['_tuned_parameters'].extend(['models.policy.encoder', 'models.critic.encoder'])
    return cfg


def objective(result):
    value = result.get('auc_mean')
    if (result.get('status') != 'COMPLETED' or result.get('completed_episodes') != 512
            or value is None or not math.isfinite(value) or value < 0):
        raise ValueError('Validation requires 512 complete trajectories and a finite nonnegative AUC')
    return float(value)


def check_policy_updates(kls):
    if not kls or not all(math.isfinite(value) and value >= 0 for value in kls):
        raise RuntimeError('GPU check requires finite nonnegative PPO post-update KL measurements')
    changed = sum(value > 0 for value in kls)
    if not changed:
        raise RuntimeError('GPU check observed no PPO policy movement; refusing to launch a sweep agent')
    return dict(updates=len(kls), changed=changed)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def finish_run(run, result, step):
    succeeded = result['status'] == 'COMPLETED'
    metrics = {'Training/Learning/Search/failed': int(not succeeded), 'global_step': step}
    if succeeded and result['objective'] is not None:
        metrics[OBJECTIVE] = result['objective']
        run.summary[OBJECTIVE] = result['objective']
    run.log(metrics)
    run.summary['sweep_result_status'] = result['status']
    run.finish(exit_code=0 if succeeded else 1)


def progress(text, target):
    matches = re.findall(r'(?<!\d)([\d,]+)\s*/\s*' + f'{target:,}'.replace(',', ',?') + r'(?!\d)', text)
    return max([0] + [min(target, int(value.replace(',', ''))) for value in matches])


def trial(root, *, smoke=None, steps=None, checkpoint=None, smoke_evaluate=False):
    import wandb
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    run = wandb.init(project=plan['project'], entity=plan['entity'],
                     config=smoke, mode='disabled' if smoke is not None else 'online',
                     settings=wandb.Settings(console='off'))
    parameters = dict(run.config)
    from contractionRL.agents.utils.RL.config import load_agent_config
    cfg = resolve_config(load_agent_config(root / 'base_config.yaml'), parameters)
    metric_receipt = plan.get('pretrained_metrics', {}).get(parameters['ccm_bounds'])
    if plan.get('require_pretrained_metric') and not metric_receipt:
        raise ValueError('No pretrained metric for sampled CCM bounds')
    if metric_receipt:
        metric_path = root / metric_receipt['path']
        if hashlib.sha256(metric_path.read_bytes()).hexdigest() != metric_receipt['sha256']:
            raise ValueError('Pretrained metric checksum mismatch')
        cfg['agent']['cmg_checkpoint'] = str(metric_path)
        cfg['cmg']['ccm_epochs'] = 0
    identity = ('smoke-' + os.environ.get('CRL_SMOKE_NAME', str(os.getpid()))) if smoke is not None else run.id
    folder = root / ('preflight' if smoke is not None else 'trials') / identity
    folder.mkdir(parents=True, exist_ok=False)
    target = steps if steps is not None else plan['training_steps']
    cfg['agent']['experiment'].update(directory=str(folder / 'training'), write_interval=500,
                                        checkpoint_interval=0, wandb=False)
    cfg['trainer']['timesteps'] = target
    if smoke is not None and not metric_receipt:
        cfg['cmg']['ccm_epochs'] = 2
        cfg['cmg']['ccm_memory_size'] = 128
        cfg['cmg']['ccm_batch_size'] = 64
    config = folder / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    machine = os.environ.get('CRL_SWEEP_MACHINE', socket.gethostname())
    state = dict(status='STARTING', current_steps=0, target_steps=target, training_pid=os.getpid(),
                 source_sha256=plan.get('source_sha256'),
                 machine=machine, gpu_index=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 configuration=str(config), parameters=parameters, seed=plan['seed'],
                 started_at=datetime.now(timezone.utc).isoformat())
    entry = dict(id=plan['id']+'-'+identity, parent_id=plan['id'], name='Car Bayesian trial '+identity,
                 purpose='Matched500k-step analytical C2RL-PPO Bayesian search', machines=[machine],
                 repository=str(root / 'code'), paths=[str(folder)], configuration=str(config),
                 remote_trial_path=str(folder), remote_configuration=str(config),
                 job_ids=[os.environ.get('SLURM_JOB_ID', str(os.getpid()))], training_pid=os.getpid(),
                 slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=socket.gethostname(),
                 seed=plan['seed'], seeds=[plan['seed']],
                 target_steps=target, recorded_status='STARTING', dependencies=[plan['id']],
                 source_sha256=plan.get('source_sha256'),
                 process_pattern=' '.join(sys.argv), log_path=str(folder / 'rawtrain.log'),
                 wandb_url=run.url if smoke is None else None, parameters=parameters,
                 history=[dict(at=state['started_at'], status='STARTING')])
    write_json(folder / 'manifest.json', entry)
    write_json(folder / 'state.json', state)
    stop = threading.Event()
    start = time.monotonic()

    def watch():
        with (folder / 'progress.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Steps', 'WalltimeSeconds'])
            while not stop.is_set():
                log = folder / 'rawtrain.log'
                if log.exists():
                    with log.open('rb') as data:
                        data.seek(max(0, log.stat().st_size - 16384))
                        tail = data.read().decode(errors='replace')
                        state['current_steps'] = max(state['current_steps'], progress(tail, target))
                        state['phase'] = 'POLICY_TRAINING' if state['current_steps'] or 'Phase B' in tail else 'CCM_PRETRAINING'
                state.update(walltime_seconds=time.monotonic()-start, observed_at=datetime.now(timezone.utc).isoformat())
                write_json(folder / 'state.json', state)
                entry.update(recorded_status='RUNNING', current_steps=state['current_steps'],
                             phase=state.get('phase'), observed_at=state['observed_at'])
                write_json(folder / 'manifest.json', entry)
                writer.writerow([state['current_steps'], state['walltime_seconds']])
                stream.flush()
                stop.wait(10)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    result = dict(status='FAILED', objective=None, parameters=parameters)
    sys.path[:0] = [str(root / 'code/source/contractionRL'), str(root / 'code/scripts/skrl'), str(root)]
    import train_utils
    original_finish = train_utils.finish_wandb
    train_utils.install_wandb_scalar_hook()
    original_argv = sys.argv[:]
    original_directory = Path.cwd()
    smoke_kls = []
    original_track = None
    if smoke is not None:
        from skrl.agents.torch.ppo import PPO
        original_track = PPO.track_data

        def track_smoke(agent, tag, value):
            if tag == 'Policy / KL divergence':
                smoke_kls.append(float(value))
            return original_track(agent, tag, value)

        PPO.track_data = track_smoke
    # The owning sweep run stays open through fixed-bank evaluation.
    train_utils.finish_wandb = lambda args: None
    try:
        import contextlib
        state['status'] = 'TRAINING'
        os.chdir(folder)
        sys.argv = [str(root / 'code/scripts/skrl/train.py'), '--classic', '--task', 'classic-car-v0',
                    '--cfg', str(config), '--algorithm', 'c2rl_ppo',
                    '--num_envs', str(plan['num_envs']), '--seed', str(plan['seed']),
                    '--num_timesteps', str(target), '--device', 'cuda:0', '--skip_final_eval',
                    '--wandb_project', plan['project'], '--no_wandb']
        if checkpoint:
            sys.argv.extend(['--checkpoint', str(checkpoint)])
        with (folder / 'rawtrain.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            namespace = runpy.run_path(sys.argv[0], run_name='__main__')
            owner = namespace['runner'].agent
            if metric_receipt:
                import torch
                expected = torch.load(metric_path, map_location='cpu', weights_only=False)['cmg']
                for key, value in owner.models['cmg'].state_dict().items():
                    if not torch.equal(value.cpu(), expected[key]):
                        raise RuntimeError('Pretrained metric changed during policy training: ' + key)
                result['pretrained_metric'] = dict(**metric_receipt, unchanged_after_training=True,
                                                 loaded_from=owner._restored_cmg_checkpoint)
            if smoke is not None:
                result['policy_update_check'] = check_policy_updates(smoke_kls)
            final = Path(owner.experiment_dir) / 'checkpoints' / f'agent_{target}.pt'
            final.parent.mkdir(exist_ok=True)
            owner.save(str(final))
        state.update(status='EVALUATING', current_steps=target)
        result.update(checkpoint=str(final), checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                      actual_configuration=str(final.parent.parent / 'params/agent.yaml'))
        if smoke is None or smoke_evaluate:
            evaluation = evaluate_sweep.evaluate(root, str(final), 'validation', folder / 'evaluation/validation',
                                                  metric_checkpoint=str(final))
            score = objective(evaluation)
            result.update(status='COMPLETED', validation=evaluation, objective=score)
        else:
            result.update(status='COMPLETED')
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
        (folder / 'failure.txt').write_text(result['traceback'])
    finally:
        if original_track is not None:
            PPO.track_data = original_track
        train_utils.finish_wandb = original_finish
        sys.argv = original_argv
        os.chdir(original_directory)
        state['status'] = result['status']
        stop.set()
        watcher.join(timeout=15)
        write_json(folder / 'state.json', state)
        write_json(folder / 'result.json', result)
        entry.update(recorded_status=result['status'], current_steps=state['current_steps'], results=result,
                     observed_at=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'manifest.json', entry)
        finish_run(run, result, state['current_steps'])
    if result['status'] != 'COMPLETED':
        raise RuntimeError(result.get('error'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke-config', type=Path)
    parser.add_argument('--smoke-steps', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--smoke-evaluate', action='store_true')
    args, _ = parser.parse_known_args()
    sampled = yaml.safe_load(args.smoke_config.read_text()) if args.smoke_config else None
    trial(args.root, smoke=sampled, steps=args.smoke_steps if sampled is not None else None,
          checkpoint=args.checkpoint, smoke_evaluate=args.smoke_evaluate)
