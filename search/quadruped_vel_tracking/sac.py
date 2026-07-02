import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Reward / Total reward (mean)', 'goal': 'maximize'},
    'command': [
        'conda', 'run', '-n', 'env_isaaclab',
        'python', 'scripts/skrl/train.py',
        '--task', 'Quadruped-VelTracking-v0',
        '--algorithm', 'sac',
        '--headless',
        '${args}'
    ],
    'parameters': {'learning_rate': {'min': 1e-06, 'max': 0.001}, 'discount_factor': {'values': [0.9, 0.99, 0.999]}}
}

try:
    api = wandb.Api()
    entity = api.default_entity
except Exception:
    entity = "UIUC-LIRA"

sweep_id = wandb.sweep(sweep_config, project="contractionRL", entity=entity)
print(f"{entity}/contractionRL/{sweep_id}")
