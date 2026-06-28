"""mjrl Runner — configure & instantiate mjrl components in a few lines.

Mirrors ``skrl.utils.runner.torch.Runner``:

    from mjrl.utils.runner.torch import Runner
    runner = Runner(env, cfg)   # cfg is a plain dict (loaded from YAML)
    runner.run()                # train / evaluate

``cfg`` layout::

    seed: 42
    agent:
      class: C3M | LQR
      device: cpu | cuda
      ...hyperparameters...
      cmg:   {hidden_dim: [...], activation: tanh, mode: deterministic}   # C3M
      actor: {hidden_dim: [...], activation: tanh, type: CL}              # C3M
    trainer:
      class: C3MTrainer | EvalTrainer
      num_agent: 4          # == num_worker for the sampler
      epochs: 30000
      eval_interval: 1000

Optional ``dynamics_model`` kwarg (NeuralDynamics):
  * Sets ``env.learned_dynamics_model`` and ``env.use_learned_dynamics = True``
    so that env simulation rollouts use the learned model.
  * Passes ``dynamics_model.get_f_and_B`` (autodiff-compatible) to agents
    instead of ``env.get_f_and_B``, enabling C3M Jacobians through the net.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from mjrl.agents.torch import C3M, LQR
from mjrl.models import CCM_Generator, CLActor
from mjrl.trainers.torch import C3MTrainer, EvalTrainer

_AGENTS = {"C3M": C3M, "LQR": LQR}
_TRAINERS = {"C3MTrainer": C3MTrainer, "EvalTrainer": EvalTrainer}

# sensible default trainer per agent
_DEFAULT_TRAINER = {"C3M": "C3MTrainer", "LQR": "EvalTrainer"}


class Runner:
    def __init__(
        self,
        env,
        cfg: dict[str, Any],
        *,
        verbose: bool = False,
        dynamics_model=None,
    ) -> None:
        self._env = env
        self._verbose = verbose
        self._cfg = copy.deepcopy(cfg)

        # Wire up learned dynamics on the env so simulation rollouts go through it
        if dynamics_model is not None:
            env.learned_dynamics_model = dynamics_model
            env.use_learned_dynamics = True

        seed = self._cfg.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed)

        self._agent = self._generate_agent(env, self._cfg, dynamics_model=dynamics_model)
        self._trainer = self._generate_trainer(env, self._cfg, self._agent)

    # ------------------------------------------------------------------ #
    @property
    def agent(self):
        return self._agent

    @property
    def trainer(self):
        return self._trainer

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    def _generate_agent(self, env, cfg: dict, dynamics_model=None):
        agent_cfg = dict(cfg.get("agent", {}))
        agent_class = agent_cfg.pop("class", None)
        if agent_class not in _AGENTS:
            raise ValueError(f"Unknown agent class {agent_class!r}. Choose from {list(_AGENTS)}")

        device = agent_cfg.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        x_dim = int(env.num_dim_x)
        u_dim = int(env.num_dim_control)

        # Prefer dynamics_model.get_f_and_B so that C3M's Jacobians flow through
        # the neural network.  Fall back to env.get_f_and_B for analytical envs.
        get_f_and_B = dynamics_model.get_f_and_B if dynamics_model is not None else env.get_f_and_B

        if agent_class == "LQR":
            agent = LQR(
                x_dim=x_dim, u_dim=u_dim,
                get_f_and_B=get_f_and_B,
                Q_scaler=float(agent_cfg.get("Q_scaler", 1.0)),
                R_scaler=float(agent_cfg.get("R_scaler", 0.0)),
                device=device,
            )

        elif agent_class == "C3M":
            cmg_cfg = dict(agent_cfg.get("cmg", {}))
            actor_cfg = dict(agent_cfg.get("actor", {}))
            CMG = CCM_Generator(
                x_dim=x_dim,
                hidden_dim=list(cmg_cfg.get("hidden_dim", [128, 128])),
                activation=cmg_cfg.get("activation", "tanh"),
                mode=cmg_cfg.get("mode", "deterministic"),
                device=device,
            )
            actor = CLActor(
                x_dim=x_dim, u_dim=u_dim,
                mode="deterministic",
                hidden_dim=list(actor_cfg.get("hidden_dim", [128, 128])),
                activation=actor_cfg.get("activation", "tanh"),
            )
            buffer_size = int(agent_cfg.get("buffer_size", 65536))
            data = env.get_rollout(buffer_size, mode="c3m")
            agent = C3M(
                x_dim=x_dim, u_dim=u_dim, CMG=CMG, actor=actor, data=data,
                get_f_and_B=get_f_and_B,
                W_lr=float(agent_cfg.get("W_lr", 3e-4)),
                u_lr=float(agent_cfg.get("u_lr", 3e-4)),
                lbd=float(agent_cfg.get("lbd", 0.5)),
                eps=float(agent_cfg.get("eps", 1e-2)),
                w_ub=float(agent_cfg.get("w_ub", 10.0)),
                w_lb=float(agent_cfg.get("w_lb", 0.1)),
                num_minibatch=int(agent_cfg.get("num_minibatch", 8)),
                minibatch_size=int(agent_cfg.get("minibatch_size", 256)),
                nupdates=int(cfg.get("trainer", {}).get("epochs", 30000)),
                cmg_updates_per_policy_update=int(agent_cfg.get("cmg_updates_per_policy_update", 1)),
                device=device,
            )

        agent.to_device(torch.device(device))
        # expose policy on env for render/bound diagnostics
        try:
            env.policy = agent
        except Exception:
            pass
        return agent

    def _generate_trainer(self, env, cfg: dict, agent):
        trainer_cfg = dict(cfg.get("trainer", {}))
        agent_class = cfg.get("agent", {}).get("class")
        trainer_class = trainer_cfg.pop("class", None) or _DEFAULT_TRAINER.get(agent_class)
        if trainer_class not in _TRAINERS:
            raise ValueError(f"Unknown trainer class {trainer_class!r}. Choose from {list(_TRAINERS)}")
        trainer_cfg.setdefault("seed", cfg.get("seed", 0))
        return _TRAINERS[trainer_class](env, agent, trainer_cfg)

    # ------------------------------------------------------------------ #
    def run(self, mode: str = "train"):
        return self._trainer.run()
