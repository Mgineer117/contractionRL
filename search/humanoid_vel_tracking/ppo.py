import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'Reward / Total reward (mean)', 'goal': 'maximize'},
    'command': [
        'conda', 'run', '-n', 'env_isaaclab',
        'python', 'scripts/skrl/train.py',
        '--task', 'Humanoid-VelTracking-v0',
        '--algorithm', 'ppo',
        '--headless',
        '${args}'
    ],
    'parameters': {'ppo_lr': {'min': 1e-06, 'max': 0.001}, 'ppo_entropy_scale': {'values': [0.0, 0.001, 0.01]}, 'ppo_kl_threshold': {'min': 0.001, 'max': 0.1}, 'ppo_discount': {'values': [0.9, 0.99, 0.999]}, 'ppo_use_state_norm': {'values': ['True', 'False']}, 'ppo_use_value_norm': {'values': ['True', 'False']}, 'ppo_lambda': {'min': 0.9, 'max': 0.99}, 'ppo_activations': {'values': ['elu', 'relu', 'tanh']}, 'ppo_network_arch': {'values': ['128,128', '512,256,128', '256,256,256']}}
}

try:
    api = wandb.Api()
    entity = api.default_entity
except Exception:
    entity = "UIUC-LIRA"

sweep_id = wandb.sweep(sweep_config, project="contractionRL", entity=entity)
print(f"{entity}/contractionRL/{sweep_id}")
