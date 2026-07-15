import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Reward / Total reward (mean)', 'goal': 'maximize'},
    'command': [
        'conda', 'run', '-n', 'env_isaaclab',
        'python', 'scripts/skrl/train.py',
        '--task', 'Manipulator-PathTracking-v0',
        '--algorithm', 'c3m',
        '--headless',
        '${args}'
    ],
    'parameters': {
        'learning_rate': {'min': 1e-06, 'max': 0.001},
        'discount_factor': {'values': [0.9, 0.99, 0.999]},
        'c1_c2_scale': {'distribution': 'uniform', 'min': 0.01, 'max': 1.0},
        'agent.models.policy.backbone': {'values': ['control', 'control-squashed']},
    }
}

try:
    api = wandb.Api()
    entity = api.default_entity
except Exception:
    entity = "UIUC-LIRA"

sweep_id = wandb.sweep(sweep_config, project="contractionRL", entity=entity)
print(f"{entity}/contractionRL/{sweep_id}")
