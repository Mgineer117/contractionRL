"""Validate PPO-family options and expose conditional parameters to optimizers."""

PPO_FEATURE_FIELDS = {'use_state_norm','use_value_norm','use_reward_norm','rewards_shaper_scale',
    'use_value_expansion','value_expansion_horizon','value_expansion_batch_size','action_bound_penalty_beta',
    'reward_euclidean','reward_level'}


def validate_algorithm_config(cfg):
    algorithm = str(cfg.get('agent', {}).get('class', '')).lower().replace('_','-')
    if algorithm not in {'ppo','c2rl-ppo','contractionppo','contraction-ppo'}:
        return
    allowed_roots = {'seed','agent','models','memory','trainer','cm','cmg','empirical_dynamics',
                     'environment','parameter_usage','_tuned_parameters','network_architecture','dynamics_backend'}
    unknown = set(cfg) - allowed_roots
    if unknown:
        raise ValueError(f"{algorithm}: unsupported config sections {sorted(unknown)}")
    from skrl.agents.torch.ppo import PPO_CFG
    fields = set(PPO_CFG.__dataclass_fields__) | PPO_FEATURE_FIELDS | {'class'}
    if algorithm == 'c2rl-ppo':
        from contractionRL.agents.skrl.c2rl import C2RLPPOCfg
        fields.update(C2RLPPOCfg.__dataclass_fields__)
    elif algorithm != 'ppo':
        from contractionRL.agents.skrl.contraction_ppo import ContractionPPOCfg
        fields.update(ContractionPPOCfg.__dataclass_fields__)
    unknown = set(cfg.get('agent', {})) - fields
    if unknown:
        raise ValueError(f"{algorithm}: unsupported agent parameters {sorted(unknown)}")
    if algorithm != 'c2rl-ppo':
        for section in ('cm','cmg'):
            if cfg.get(section):
                raise ValueError(f"{section} is a C2RL feature and is not used by {algorithm}")
        if 'cmg' in cfg.get('models', {}):
            raise ValueError(f"models.cmg is not used by {algorithm}")
    if algorithm == 'c2rl-ppo':
        from contractionRL.agents.utils.RL.config import migrate_c2rl_metric_cfg
        from contractionRL.agents.skrl.c2rl import C2RLPPOCfg
        for section in ('cm','cmg'):
            normalized = migrate_c2rl_metric_cfg(cfg.get(section, {}))
            unknown = set(normalized) - set(C2RLPPOCfg.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unsupported {section} parameters {sorted(unknown)}")
    shared_model_fields = {'class','network','output','encoder','angle_idx','sym','x_dim'}
    role_fields = {
        'policy': shared_model_fields | {'control_mode','initial_log_std','clip_actions','clip_log_std','min_log_std','max_log_std'},
        'value': shared_model_fields | {'analytic_potential','w_lb','legacy_reference_critic'},
        'critic': shared_model_fields | {'analytic_potential','w_lb','legacy_reference_critic'},
        'dynamics': {'class','network'},
        'cmg': {'class','network'},
    }
    for role, block in cfg.get('models', {}).items():
        if role=='encoder':
            continue
        if role=='separate':
            if block is not True:
                raise ValueError('PPO tracking requires separate policy and critic models')
            continue
        allowed_roles = {'policy','critic','cmg','dynamics'} if algorithm=='c2rl-ppo' else {'policy','value','dynamics'}
        if role not in role_fields or role not in allowed_roles:
            raise ValueError(f"{algorithm}: unsupported model role {role}")
        expected_class = {'dynamics':'NeuralDynamics','cmg':'CMG'}.get(role)
        if expected_class is not None and block.get('class', expected_class) != expected_class:
            raise ValueError(f'models.{role}.class must be {expected_class}')
        unknown = set(block) - role_fields[role]
        if unknown:
            raise ValueError(f"Unsupported models.{role} parameters {sorted(unknown)}")
        for network in block.get('network', []):
            unknown = set(network) - {'name','input','layers','activations'}
            if unknown:
                raise ValueError(f"Unsupported models.{role}.network parameters {sorted(unknown)}")
    empirical_fields = {'num_controls_per_state','dynamics_lr','dynamics_lr_scheduler',
        'dynamics_lr_scheduler_kwargs','dynamics_batch_size','dynamics_pretrain_epochs',
        'dynamics_pretrain_data_path','emp_dynamics_memory_size','dynamics_val_frac','dynamics_early_stop_patience'}
    unknown = set(cfg.get('empirical_dynamics', {})) - empirical_fields
    if unknown:
        raise ValueError(f"Unsupported empirical_dynamics parameters {sorted(unknown)}")


def config_parameter_usage(cfg, env=None, *, task=None):
    """Report inactive leaves; inherited defaults may exist without being tunable."""
    from contractionRL.runners.contraction_runner import _unwrap_env
    raw = _unwrap_env(env) if env is not None else None
    algorithm = str(cfg.get('agent', {}).get('class', '')).lower().replace('_','-')
    a, models = cfg.get('agent', {}), cfg.get('models', {})
    enabled = any(isinstance(block,dict) and block.get('encoder')=='gru' for block in models.values())
    exact = getattr(raw, 'has_analytical_dynamics', None)
    if exact is None and task is not None:
        from contractionRL.tasks.direct.common.isaac.backend import dynamics_backend_for_task
        exact = dynamics_backend_for_task(task)=='analytical'
    c2rl = algorithm.startswith('c2rl')
    needs_model = c2rl or algorithm == 'c3m' or a.get('use_value_expansion',False)
    method = a.get('cmg_method','ccm' if c2rl else '')
    compact = getattr(getattr(raw, 'ref_window', None), 'compact', True)
    inactive, leaves = {}, []

    def visit(values, prefix=''):
        for key,value in values.items():
            path = f'{prefix}.{key}' if prefix else key
            if key.startswith('_') or key=='parameter_usage':
                continue
            if isinstance(value,dict) and value:
                visit(value,path)
            elif isinstance(value,list) and value and all(isinstance(row,dict) for row in value):
                for index,row in enumerate(value):
                    visit(row,f'{path}.{index}')
            else:
                leaves.append(path)
    visit(cfg)

    def mark(prefix, reason):
        for path in leaves:
            if path==prefix or path.startswith(prefix+'.'):
                inactive[path]=reason

    for role,block in models.items():
        if not isinstance(block,dict):
            continue
        for index,network in enumerate(block.get('network', [])):
            for key in ('name','input'):
                mark(f'models.{role}.network.{index}.{key}','Fixed model input/label; not an optimization parameter')
        mark(f'models.{role}.output','Fixed model output; not an optimization parameter')
        if role=='policy' and block.get('clip_log_std') is False:
            for key in ('min_log_std','max_log_std'):
                mark(f'models.{role}.{key}','Log-standard-deviation clipping is disabled')
    if algorithm in ('ppo','c2rl-ppo','contractionppo','contraction-ppo'):
        mark('empirical_dynamics.num_controls_per_state','Measured rollout collector records one applied control per transition')
    mark('environment','Recorded metadata; change the environment configuration instead')
    mark('dynamics_backend','The task selects the dynamics backend')
    if not enabled:
        mark('models.encoder','Neither policy nor critic uses GRU')
    if not a.get('use_value_expansion',False):
        for key in ('value_expansion_horizon','value_expansion_batch_size'):
            mark('agent.'+key,'Value expansion is disabled')
    if exact or not needs_model:
        reason = 'The environment supplies exact dynamics' if exact else 'This configuration does not use a dynamics model'
        mark('models.dynamics',reason); mark('empirical_dynamics',reason)
    elif not cfg.get('empirical_dynamics',{}).get('dynamics_lr_scheduler'):
        mark('empirical_dynamics.dynamics_lr_scheduler_kwargs','Dynamics scheduler is disabled')
    if not a.get('learning_rate_scheduler'):
        mark('agent.learning_rate_scheduler_kwargs','Learning-rate scheduler is disabled')
    if c2rl:
        other = 'cvstem' if method=='ccm' else 'ccm'
        for section in ('agent','cm','cmg'):
            for key in cfg.get(section,{}):
                if key.startswith(other+'_'):
                    mark(section+'.'+key,f'cmg_method is {method}')
        if method=='ccm':
            for key in cfg.get('cmg',{}):
                if key.startswith('regress_'):
                    mark('cmg.'+key,'CCM fits the metric directly; regression is disabled')
            for key in ('cm_solver','cm_data_path','cvstem_r_scaler'):
                mark('cm.'+key,'Only used by CV-STEM')
        if compact:
            for section in ('cm','agent'):
                for key in ('tracking_scaler','control_scaler','reward_level','residual_anchor_scale'):
                    mark(section+'.'+key,'Compact tracking uses its fixed energy-decrease reward')
    if compact and not c2rl:
        for key in ('reward_euclidean','reward_level'):
            mark('agent.'+key,'Compact PPO tracking uses its fixed energy-decrease reward')
    if raw is not None and not hasattr(raw,'set_action_bound_penalty'):
        mark('agent.action_bound_penalty_beta','Environment does not implement an action-bound penalty')
    return {'active':[path for path in leaves if path not in inactive], 'inactive':inactive}


def validate_optimization_parameters(cfg, parameters, env=None, *, task=None):
    """Reject sampled inactive options instead of wasting optimizer trials."""
    inactive = config_parameter_usage(cfg, env, task=task)['inactive']
    rejected = {path:reason for key in parameters for path,reason in inactive.items()
                if path==key or path.startswith(key+'.') or key.startswith(path+'.')}
    if rejected:
        details = '; '.join(f'{path}: {reason}' for path,reason in sorted(rejected.items()))
        raise ValueError('Optimization contains inactive parameters: '+details)
