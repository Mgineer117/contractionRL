"""Export safe nominal references from a trained velocity-tracking policy.

Example: python scripts/skrl/generate_ref_traj.py --task Quadruped-VelTracking-v0
--checkpoint logs/skrl/.../checkpoints/best_agent.pt --robot quadruped
--num_envs 8 --num_trajs 20 --out_dir data/quadruped --headless
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/contractionRL'))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', required=True, help='Trained agent checkpoint (.pt)')
    parser.add_argument('--robot', required=True, choices=['quadruped', 'humanoid', 'manipulator'])
    parser.add_argument('--num_envs', type=int, default=128, help='Parallel simulator environments')
    parser.add_argument('--num_trajs', type=int, default=2000, help='Target full-length trajectories before domain filtering')
    parser.add_argument('--out_dir', default=None, help='Output directory (default: checkout data/{robot})')
    parser.add_argument('--seed', type=int, default=42, help='Policy/environment generation seed')
    parser.add_argument('--min_ref_quality', type=float, default=None,
                        help='Minimum mean episode return; default matches training export; 0 disables this gate')
    parser.add_argument('--ref_oversample_factor', type=float, default=2.,
                        help='Candidate pool multiplier, matching training export (minimum 1)')
    return parser


def validate_args(args):
    if args.num_envs < 1 or args.num_trajs < 1:
        raise ValueError('--num_envs and --num_trajs must be positive')
    if not math.isfinite(args.ref_oversample_factor) or args.ref_oversample_factor < 1:
        raise ValueError('--ref_oversample_factor must be finite and at least 1')
    if args.min_ref_quality is not None and not math.isfinite(args.min_ref_quality):
        raise ValueError('--min_ref_quality must be finite')
    task = args.task.split(':')[-1]
    if not task.startswith(args.robot.capitalize() + '-VelTracking'):
        raise ValueError('--robot must match the velocity-tracking task')
    args.ref_num_trajs = args.num_trajs
    return args


def checkpoint_agent_config(fallback, checkpoint):
    """Restore recorded model and normalization settings before allocating models."""
    import yaml
    from train_utils import restore_checkpoint_model_config
    saved_config = Path(checkpoint).parent.parent / 'params/agent.yaml'
    config = yaml.safe_load(saved_config.read_text()) if saved_config.is_file() else copy.deepcopy(fallback)
    restore_checkpoint_model_config(config, checkpoint)
    return config



def check_checkpoint_environment(checkpoint, env_cfg):
    """Reject known plant/control mismatches without constructing YAML objects."""
    import warnings
    from numbers import Real
    import yaml
    saved_path = Path(checkpoint).parent.parent / 'params/env.yaml'
    result = dict(path=str(saved_path), checked=[], missing=[])
    if not saved_path.is_file():
        warnings.warn('Checkpoint has no params/env.yaml; export environment compatibility is unverified. '
                      'Use the original training control scales, actuator settings and timestep.', stacklevel=2)
        result['missing'] = ['params/env.yaml']
        return result
    # compose builds syntax nodes only: real Isaac records include opaque
    # NumPy/Gym Python tags that must never be instantiated here.
    tree = yaml.compose(saved_path.read_text(), Loader=yaml.SafeLoader)
    missing = object()

    def node_at(path):
        node = tree
        for part in path.split('.'):
            if not isinstance(node, yaml.MappingNode):
                return missing
            node = next((value for key, value in node.value if key.value == part), missing)
            if node is missing:
                return missing
        return node

    def plain(node):
        if isinstance(node, yaml.ScalarNode):
            if node.tag == 'tag:yaml.org,2002:null':
                return None
            if node.tag in ('tag:yaml.org,2002:float', 'tag:yaml.org,2002:int'):
                return float(node.value)
            if node.tag == 'tag:yaml.org,2002:bool':
                return node.value.lower() in ('true', 'yes', 'on')
            if node.tag == 'tag:yaml.org,2002:str':
                return node.value
        if isinstance(node, yaml.SequenceNode) and node.tag in (
                'tag:yaml.org,2002:seq', 'tag:yaml.org,2002:python/tuple'):
            return [plain(value) for value in node.value]
        if isinstance(node, yaml.MappingNode) and node.tag == 'tag:yaml.org,2002:map':
            return {plain(key): plain(value) for key, value in node.value}
        raise ValueError('Unsupported serialized object in checkpoint environment control metadata')

    def current_at(path):
        value = env_cfg
        for part in path.split('.'):
            value = value.get(part, missing) if isinstance(value, dict) else getattr(value, part, missing)
            if value is missing:
                return missing
        return value

    def equivalent(saved, current):
        if hasattr(current, 'tolist'):
            current = current.tolist()
        if isinstance(saved, dict) and isinstance(current, dict):
            return saved.keys() == current.keys() and all(equivalent(saved[key], current[key]) for key in saved)
        if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
            return len(saved) == len(current) and all(equivalent(a, b) for a, b in zip(saved, current))
        if isinstance(saved, Real) and isinstance(current, Real):
            return math.isfinite(saved) and math.isfinite(current) and math.isclose(saved, current, rel_tol=1e-6, abs_tol=1e-9)
        return saved == current

    def compare(path, current=missing, saved=missing):
        current = current_at(path) if current is missing else current
        if current is missing:
            return
        node = node_at(path)
        if saved is missing:
            if node is missing:
                result['missing'].append(path)
                return
            saved = plain(node)
        if not equivalent(saved, current):
            raise ValueError(f'Checkpoint environment mismatch for {path}: saved={saved!r}, current={current!r}. '
                             'Export with the original training environment/control settings.')
        result['checked'].append(path)

    for path in ('sim.dt', 'decimation', 'action_scale', 'action_scale_shoulder', 'action_scale_forearm',
                 'arm_joint_names', 'ee_body_name', 'robot_cfg.spawn.usd_path',
                 'robot_cfg.init_state.joint_pos'):
        compare(path)
    space = getattr(env_cfg, 'action_space', missing)
    if space is not missing:
        shape = getattr(space, 'shape', (space,) if isinstance(space, int) else missing)
        if shape is not missing:
            node = node_at('action_space._shape')
            if node is missing:
                node = node_at('action_space')
            if node is missing:
                result['missing'].append('action_space')
            else:
                saved = plain(node)
                compare('action_space', list(shape), [saved] if isinstance(saved, (int, float)) else saved)
    observation = getattr(env_cfg, 'observation_space', missing)
    if isinstance(observation, int):
        compare('observation_space', observation)
    actuators = current_at('robot_cfg.actuators')
    if isinstance(actuators, dict):
        node = node_at('robot_cfg.actuators')
        if isinstance(node, yaml.MappingNode):
            compare('robot_cfg.actuators', sorted(actuators), sorted(key.value for key, value in node.value))
        for name in actuators:
            for field in ('joint_names_expr', 'stiffness', 'damping', 'effort_limit_sim', 'velocity_limit_sim'):
                compare(f'robot_cfg.actuators.{name}.{field}')
    if result['missing']:
        warnings.warn('Checkpoint environment metadata is incomplete; compatibility is unverified for: '
                      + ', '.join(result['missing']) + '. Use the original training settings.', stacklevel=2)
    return result


def run_export(args, env_cfg, agent_cfg):
    import gymnasium as gym
    from skrl.utils import set_seed
    from contractionRL.agents.utils.RL.config import normalize_agent_cfg
    from skrl.envs.wrappers.torch import wrap_env
    from contractionRL.agents.utils.RL.runner import CLActorRunner
    from train_utils import _generate_ref_trajs

    validate_args(args)
    checkpoint = str(Path(args.checkpoint).resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    compatibility = check_checkpoint_environment(checkpoint, env_cfg)
    config = checkpoint_agent_config(agent_cfg, checkpoint)
    seed = int(getattr(args, 'seed', 42))
    set_seed(seed)
    config['seed'] = seed
    algorithm = str(config['agent']['class']).lower()
    if algorithm not in ('ppo', 'sac', 'mbpo'):
        raise ValueError('Reference export requires a trained velocity PPO, SAC or MBPO policy')
    config.setdefault('memory', {})['memory_size'] = -1 if algorithm == 'ppo' else 4096
    config.setdefault('trainer', {}).update(timesteps=0, headless=True, close_environment_at_exit=False)
    config['agent'].setdefault('experiment', {}).update(wandb=False, write_interval=0, checkpoint_interval=0)
    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = seed
    if getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    if hasattr(env_cfg, 'randomize_init'):
        env_cfg.randomize_init = True
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ROOT / 'data' / args.robot
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    config['agent']['experiment']['directory'] = str(out_dir / 'export-logs' / stamp)
    config_path = out_dir / f'export_config_{stamp}.json'
    config_path.write_text(json.dumps(dict(task=args.task, checkpoint=checkpoint,
        seed=seed, num_envs=args.num_envs, num_trajs=args.num_trajs, min_ref_quality=args.min_ref_quality,
        ref_oversample_factor=args.ref_oversample_factor, checkpoint_environment=compatibility, agent=config,
        environment=dict(seed=getattr(env_cfg, 'seed', seed), episode_length_s=env_cfg.episode_length_s, dt=env_cfg.sim.dt,
                         decimation=env_cfg.decimation, device=env_cfg.sim.device)), indent=2, default=str) + '\n')
    # Persist raw effective settings before simulator/model allocation.
    env_raw = None
    try:
        env_raw = gym.make(args.task, cfg=env_cfg)
        wrapped = wrap_env(env_raw, wrapper="isaaclab")
        normalize_agent_cfg(config, algorithm=algorithm, device=wrapped.device)
        runner = CLActorRunner(wrapped, config)
        bank = _generate_ref_trajs(task=args.task.split(':')[-1], runner=runner,
            isaac_env=env_raw, skrl_env=wrapped, env_cfg=env_cfg, args_cli=args,
            out_dir=str(out_dir), checkpoint_path=checkpoint)
        if bank is None:
            raise RuntimeError('Reference export rejected the policy; no new accepted reference bank was saved')
        return bank
    finally:
        if env_raw is not None:
            env_raw.close()


def main(argv=None):
    from isaaclab.app import AppLauncher
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    args.headless = True
    args.kit_args = (getattr(args, 'kit_args', '') or '') + ' --/app/hangDetector/enabled=false'
    sys.argv = [sys.argv[0]] + hydra_args
    launcher = AppLauncher(args)

    def export():
        import contractionRL.tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config
        @hydra_task_config(args.task, 'skrl_cfg_entry_point')
        def configured(env_cfg, agent_cfg):
            return run_export(args, env_cfg, agent_cfg)
        return configured()

    from isaac_entrypoint import run_isaac
    import omni.kit.app
    run_isaac(export, launcher.app, omni.kit.app.get_app())


if __name__ == '__main__':
    main()
