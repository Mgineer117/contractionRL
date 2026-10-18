"""Register these finite ablations and publish data to the existing private dashboard."""
from pathlib import Path
import fcntl
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
DASH = Path('/home/minjae/research/experiment-dashboard')


def publish():
    paths = [ROOT / 'manifest.json', *sorted((ROOT / 'preflight').glob('*/manifest.json')),
             *sorted((ROOT / 'agents').glob('*/manifest.json')), *sorted((ROOT / 'trials').glob('*/manifest.json'))]
    entries = [json.loads(path.read_text()) for path in paths]
    with (DASH / 'work/refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads((DASH / 'registry.json').read_text())
        indexed = {entry['id']: i for i, entry in enumerate(registry)}
        for entry in entries:
            if entry['id'] in indexed:
                registry[indexed[entry['id']]] = entry
            else:
                indexed[entry['id']] = len(registry)
                registry.append(entry)
        (DASH / 'registry.json').write_text(json.dumps(registry, indent=2) + '\n')
        progress = json.loads((DASH / 'progress.json').read_text())
        for entry in entries:
            pid = entry.get('slurm_job_id') or entry.get('training_pid')
            if pid and not any(row.get('machine') == entry['machines'][0] and row.get('job_id') == str(pid) for row in progress):
                progress.append(dict(machine=entry['machines'][0], job_id=str(pid), seed=entry['seed'], target=entry['target_steps'],
                                     csv_glob=entry['remote_trial_path'] + '/progress.csv',
                                     result_glob=entry['remote_trial_path'] + '/result.json',
                                     log_path=entry['remote_trial_path'] + '/rawtrain.log',
                                     repository='contractionRL', environment='Isaac-Car analytical', algorithm='C2RL-PPO'))
        (DASH / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        path = DASH / 'public/status.json'
        snapshot = json.loads(path.read_text())
        snapshot['experiments'] = registry
        path.write_text(json.dumps(snapshot, indent=2) + '\n')
        env = dict(os.environ, PATH='/home/minjae/miniconda3/envs/contraction/bin:' + os.environ['PATH'])
        with (ROOT / 'dashboard-publish.log').open('a') as log:
            subprocess.run(['npm', 'run', 'build'], cwd=DASH, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(['bash', '-c', 'set -a\n. ./.env.runner\nset +a\npython3 scripts/runner.py --publish'],
                           cwd=DASH, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Registered and published', len(entries), 'experiment entries', flush=True)


if __name__ == '__main__':
    publish()
