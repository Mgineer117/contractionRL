"""GPU preflight, matched archived replay, then crossed early reward analysis."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_pair(root, directory, steps):
    states = [json.loads((root / directory / name / 'state.json').read_text()) for name in ['ppo', 'c2rl_ppo']]
    assert all(state['status'] == 'COMPLETED' and state['accepted_updates'] > 0 for state in states)
    assert states[0]['initial_model_sha256'] == states[1]['initial_model_sha256']
    assert states[0]['initial_scenario_sha256'] == states[1]['initial_scenario_sha256']
    expected = [0, 64] if steps == 64 else list(range(0, steps+1, 10000))
    assert all(state['snapshots'] == expected for state in states), 'Missing preserved checkpoints'
    return states


def run(root, phase, training_device):
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MPLBACKEND='Agg', CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='disabled')
    os.environ.pop('WANDB_SWEEP_ID', None)
    preflight_phase = 'GPU_PREFLIGHT' if training_device.startswith('cuda') else 'CPU_PREFLIGHT'
    state = dict(status=preflight_phase if phase=='train' else 'ANALYZING', started_at=datetime.now(timezone.utc).isoformat(),
                 job_id=os.environ.get('SLURM_JOB_ID'), pid=os.getpid())
    def update(**fields):
        state.update(fields, observed_at=datetime.now(timezone.utc).isoformat())
        write_json(root / 'state.json', state)
    def training_command(method, output, steps, environments):
        return [sys.executable, '-u', str(root / 'early_landscape_replay.py'), '--root', str(root),
                '--algorithm', method, '--output', output, '--steps', str(steps),
                '--num-envs', str(environments), '--device', training_device, '--threads', '2']
    def execute(command, log_name):
        with (root / log_name).open('x') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    update()
    try:
        if phase == 'analysis':
            state['training'] = dict(zip(['ppo', 'c2rl_ppo'], check_pair(root, 'training', 100000)))
            execute([sys.executable, '-u', str(root/'early_reward_landscape.py'), '--root', str(root),
                     '--output', str(root/'analysis'), '--batch-size', '16', '--device', 'cpu'], 'analysis.log')
            result = json.loads((root / 'analysis/summary.json').read_text())
            assert result['status'] == 'COMPLETED' and len(result['rows']) == 14
            update(status='COMPLETED', training_status='COMPLETED', analysis_status='COMPLETED')
            (root / 'completed.csv').write_text('Steps,Status\n100000,COMPLETED\n')
            return
        for method in ['ppo', 'c2rl_ppo']:
            execute(training_command(method, f'preflight/{method}', 64, 8), f'preflight-{method}.log')
        state['preflight'] = check_pair(root, 'preflight', 64)
        execute([sys.executable, '-u', str(root/'early_reward_landscape.py'), '--root', str(root),
                 '--output', str(root/'analysis-preflight'), '--smoke', '--batch-size', '2',
                 '--device', training_device], 'analysis-preflight.log')
        update(status='TRAINING', preflight_status='PASSED')
        processes = {}
        for method in ['ppo', 'c2rl_ppo']:
            log = (root / f'train-{method}.log').open('x')
            process = subprocess.Popen(training_command(method, f'training/{method}', 100000, 128),
                cwd=root, stdout=log, stderr=subprocess.STDOUT)
            processes[method] = (process, log)
        while any(process.poll() is None for process, _ in processes.values()):
            observed = {}
            for method, (process, _) in processes.items():
                path = root / 'training' / method / 'state.json'
                observed[method] = json.loads(path.read_text()) if path.exists() else dict(status='STARTING')
                observed[method]['pid'] = process.pid
            update(training=observed)
            time.sleep(10)
        for method, (process, log) in processes.items():
            log.close()
            if process.returncode:
                raise RuntimeError(f'{method} exited {process.returncode}; artifacts preserved; no automatic retry')
        state['training'] = dict(zip(['ppo', 'c2rl_ppo'], check_pair(root, 'training', 100000)))
        update(status='TRAINING_COMPLETED', training_status='COMPLETED')
        (root / 'training_completed.csv').write_text('Steps,Status\n100000,COMPLETED\n')
    except Exception as error:
        update(status='FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['train', 'analysis', 'all'], default='train')
    parser.add_argument('--training-device', default='cuda:0')
    args = parser.parse_args()
    for phase in (['train', 'analysis'] if args.phase=='all' else [args.phase]):
        run(args.root.resolve(), phase, args.training_device)
