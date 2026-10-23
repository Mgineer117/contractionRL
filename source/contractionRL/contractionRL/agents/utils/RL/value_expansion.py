"""Finite-horizon MVE and TD-k critic fitting for the shared skrl PPO path.

Feinberg et al. (2018), https://arxiv.org/abs/1803.00101, Eq. 2 and Alg. 1.
The PPO adaptation retains real-data GAE/actor updates, replaces real-state
critic targets with MVE, and fits intermediate imagined states with TD-k.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from contractionRL.agents.utils.contraction.math_utils import bound_W, spd_inverse, action_bound_violation, tracking_energy
from contractionRL.agents.utils.system_dynamics.state import wrap_diff, clamp_physical_state, canonical_bound_state


class TrackingExpansionModel:
    """Side-effect-free tracking transitions; references never use plant noise."""

    def __init__(self, owner, env):
        self.owner = owner
        self.env = env = getattr(env, "plant", env)
        self.window = owner._window
        if not getattr(self.window, "compact", False):
            raise ValueError("Value expansion requires compact (x, xref, xi) observations")
        self.device = owner._rl_agent.device
        self.angles = owner._angle_idx
        analytical = getattr(env, "has_analytical_dynamics", hasattr(env, "_f_logic"))
        if hasattr(env, "_f_logic"):
            self.kind = "classic"
            self.dynamics = (lambda x: env.get_f_and_B(x, need_null=False)) if analytical else owner._get_f_and_B
        elif hasattr(env, "_refs"):
            self.kind = "physx"
            self.dynamics = owner._get_f_and_B
        elif hasattr(env, "_traj_buf"):
            self.kind = "robot"
            self.dynamics = owner._get_f_and_B
        else:
            raise ValueError("Value expansion requires a supported tracking environment and reference generator")
        self.dynamics_source = "analytical" if self.kind == "classic" and analytical else "learned"
        self.dt = float(env.step_dt if self.kind == "robot" else env.dt)
        disturbance = getattr(env, 'disturbance', None)
        disturbed = bool(getattr(disturbance, 'enabled', False))
        fresh_gust = self.dynamics_source == 'analytical' and disturbed
        persistent = disturbed and getattr(disturbance, 'max_hold', self.dt) > self.dt
        self.model_assumptions = {
            'nominal_dynamics': self.dynamics_source,
            'history_length': int(getattr(self.window, 'history_length', 0)),
            'disturbance_enabled': disturbed,
            'persistent_disturbance': persistent,
            'root_disturbance_conditioning': 'fresh_unconditional_gust' if fresh_gust else 'learned_state_action_noise' if disturbed else 'disabled',
            'conditional_history_bias_possible': fresh_gust and persistent,
            'latent_simulator_gust_or_dwell_used': False,
        }
        self.time_bound = float(self.window.time_bound)
        if (not math.isfinite(self.dt) or self.dt <= 0
                or not math.isfinite(self.time_bound) or self.time_bound <= 0):
            raise ValueError("Value expansion requires finite positive dt and episode horizon")
        low, high = env._state_bounds() if self.kind == "robot" else (env.X_MIN, env.X_MAX)
        self.low, self.high = [torch.as_tensor(v, device=self.device) for v in (low, high)]
        self.control_low, self.control_high = [torch.as_tensor(v, device=self.device)
                                              for v in (owner._u_min, owner._u_max)]
        if self.kind == "robot":
            self.position = slice(14, 17) if self.window.u_dim == 7 else slice(0, 2)
        else:
            self.position = slice(0, env.pos_dimension)
        self._gust = None
        self._member_ids = None

    def begin_rollout(self, roots, *, member_ids=None):
        """Sample a private gust path for each imagined branch; never read plant gust state."""
        self._gust = None
        self._member_ids = None
        if (getattr(self.env, 'transition_backend', None) == 'learned'
                and hasattr(self.env._dynamics_model, 'member_parameters')):
            if member_ids is None:
                member_ids = getattr(self.env, '_dynamics_member_ids', None)
            if member_ids is None or len(member_ids) != len(roots):
                raise ValueError('Transfer value expansion requires the dynamics member for each root environment')
            self._member_ids = member_ids.clone()
        disturbance = getattr(self.env, 'disturbance', None)
        if self.dynamics_source == 'analytical' and getattr(disturbance, 'enabled', False):
            from contractionRL.tasks.direct.common.disturbance import BoundedGust
            self._gust = BoundedGust(len(roots), disturbance.scales, self.dt,
                                    roots.device, max_hold=disturbance.max_hold)
            self._gust.reset(torch.arange(len(roots), device=roots.device))

    def _advance(self, x, control, *, stochastic=False):
        if getattr(self.env, 'transition_backend', None) == 'learned':
            from contractionRL.tasks.direct.common.transfer_dynamics import integrate_learned
            if self._member_ids is None and hasattr(self.env._dynamics_model, 'member_parameters'):
                self.begin_rollout(x)
            return integrate_learned(self.env._dynamics_model, x, control, self.env.cfg.sim.dt,
                                     self.env.cfg.decimation, stochastic=stochastic, member_ids=self._member_ids)
        exact_reference = self.kind == "classic" and getattr(self.env, "has_analytical_dynamics", True) and not stochastic
        dynamics = (lambda x: self.env.get_f_and_B(x, need_null=False)) if exact_reference else self.dynamics
        f, B, _ = dynamics(x)
        derivative = f + torch.bmm(B, control.unsqueeze(-1)).squeeze(-1)
        if (stochastic and self.dynamics_source == 'analytical'
                and getattr(getattr(self.env, 'disturbance', None), 'enabled', False)):
            if self._gust is None or len(self._gust.value) != len(x):
                self.begin_rollout(x)
            derivative = derivative + self._gust.advance()
        if (stochastic and self.dynamics_source == "learned"
                and getattr(getattr(self.env, "disturbance", None), "enabled", False)):
            derivative = derivative + self.owner._neural_dynamics.sample_disturbance(x, control)
        return x + self.dt * derivative

    def _energy(self, x, reference):
        metric = self.owner._ccm_gen
        if metric is None:
            return wrap_diff(x - reference, self.angles).square().sum(-1)
        raw, _ = metric(x)
        M = bound_W(raw, self.owner._cfg.w_lb, self.window.x_dim, getattr(metric, "bounded", False))
        if not getattr(metric, "outputs_metric", False):
            M = spd_inverse(M)
        return tracking_energy(x - reference, M, self.angles)

    def step(self, observations, actions):
        x, xr, xi = self.window.extract(observations)
        raw_actions = actions
        actions = actions.nan_to_num().clamp(self.control_low, self.control_high)
        raw = self._advance(x, actions, stochastic=True)
        nonfinite = ~torch.isfinite(raw).all(-1)
        raw = torch.where(torch.isfinite(raw), raw, x)
        canonical = canonical_bound_state(raw, self.low, self.high, self.angles)
        outside = (canonical < self.low) | (canonical > self.high)
        failed = nonfinite | outside[:, self.position].any(-1)
        task = getattr(self.env, "task", "")
        if self.kind == "classic":
            if task == "quadrotor":
                failed |= (raw[:, 2] <= .15) | (raw[:, 7:9].abs() >= math.pi / 3).any(-1)
            elif task in ("cartpole", "segway"):
                failed |= raw[:, 1].abs() >= self.high[1]
        elif self.kind == "physx":
            failed |= outside.any(-1)
            if self.env.quad:
                failed |= (raw[:, 2] <= .15) | (raw[:, 7:9].abs() >= math.pi / 3).any(-1)
        else:
            failed |= ((canonical < self.low - 1e-5) | (canonical > self.high + 1e-5)).any(-1)
            if self.window.u_dim != 7:
                failed |= raw[:, -1] > 3.
                if getattr(self.env.cfg, "terminate_on_fall", True):
                    failed |= (raw[:, -1] < self.env.cfg.base_height_min) | (raw[:, 5] > -.71)
        next_x = clamp_physical_state(raw, self.low, self.high, self.angles)
        next_x[:, self.position] = canonical[:, self.position]
        next_xi = xi.clone()
        p = self.window.parameter_dim
        next_xi[:, p] = (xi[:, p] + self.dt).clamp(max=self.time_bound)
        if self.kind == "classic":
            ur = self.window.current_control(xr, xi)
            next_xr = self._advance(xr, ur)
            next_xr = clamp_physical_state(next_xr, self.low, self.high, self.angles)
        elif self.kind == "physx":
            ids = xi[:, 0].long().clamp(0, len(self.env._refs) - 1)
            steps = (next_xi[:, p] / self.dt).round().long().clamp(0, self.env.max_episode_length)
            next_xr = self.env._refs[ids, steps]
        else:
            ids = xi[:, 0].long().clamp(0, self.env._traj_buf.num_trajs - 1)
            next_xr, _ = self.env._traj_buf.get(ids, (next_xi[:, p] / self.dt).round().long())
        reward = (self._energy(x, xr) - self._energy(next_x, next_xr)).unsqueeze(-1)
        from contractionRL.tasks.direct.common.transfer_dynamics import ood_reward_penalty
        reward = reward - ood_reward_penalty(self.env, x, actions).unsqueeze(-1)
        beta = getattr(self.owner._cfg, "action_bound_penalty", 0.0)
        if beta:
            reward = reward - beta * action_bound_violation(raw_actions, self.control_low, self.control_high).unsqueeze(-1)
        # PhysX observations also clip terminal positions; its reward keeps the excursion.
        observed_x = clamp_physical_state(next_x, self.low, self.high, self.angles) if self.kind == "physx" else next_x
        preview = None
        if self.window.preview_length:
            if self.kind == 'classic':
                reference, parameters = next_xr, next_xi.clone()
                points = []
                for index in range(self.window.preview_length):
                    advances = self.window.preview_start if index == 0 else self.window.preview_stride
                    for _ in range(advances):
                        ur = self.window.current_control(reference, parameters)
                        candidate = clamp_physical_state(self._advance(reference, ur), self.low, self.high, self.angles)
                        reference = torch.where((parameters[:, p] < self.time_bound - self.dt * 1e-4)[:, None],
                                                candidate, reference)
                        parameters[:, p] = (parameters[:, p] + self.dt).clamp(max=self.time_bound)
                    points.append(reference)
            else:
                ids = next_xi[:, 0].long()
                step = (next_xi[:, p] / self.dt).round().long()
                points = []
                for offset in range(1, self.window.preview_length + 1):
                    index = (step + offset).clamp(max=round(self.time_bound / self.dt))
                    points.append(self.env._refs[ids, index] if self.kind == 'physx'
                                  else self.env._traj_buf.get(ids, index)[0])
            preview = torch.stack(points, 1)
        history = None
        if self.window.history_length:
            row = torch.cat((x, actions, x.new_ones(len(x), 1)), -1)
            history = torch.cat((self.window.history(observations)[:, 1:], row[:, None]), 1)
        next_obs = self.window.flatten(observed_x, next_xr, None, xi=next_xi, preview=preview, history=history)
        agent = self.owner._rl_agent
        if getattr(agent, "contraction_metric", None) is not None:
            error, endpoint = x - xr, next_x - next_xr
            derivative = (endpoint - error) / self.dt
            features = agent.metric_features(observations)
            features_dot = (agent.metric_features(next_obs) - features) / self.dt
            _, _, penalty = agent._terms(error[:, agent.indices], derivative[:, agent.indices], features, features_dot)
            reward = reward + ((agent.cfg.epsilon_contraction - penalty.detach()) * agent.cfg.penalty_coef)[:, None]
        timeout = next_xi[:, p] >= self.time_bound - self.dt * 1e-4
        return next_obs, reward, failed[:, None], timeout[:, None]


@contextmanager
def expanded_critic_targets(targets):
    """Scope the skrl target hook to this synchronous update, restoring on error."""
    import skrl.agents.torch.ppo.ppo as ppo
    original = ppo.compute_gae

    def compute_gae(**kwargs):
        _, advantages = original(**kwargs)
        return targets.reshape_as(kwargs["rewards"]), advantages

    ppo.compute_gae = compute_gae
    try:
        yield
    finally:
        ppo.compute_gae = original


class ValueExpansion:
    def __init__(self, owner, model):
        self.owner, self.model = owner, model
        self.agent = owner._rl_agent
        self.horizon = owner._cfg.value_expansion_horizon
        configured_batch_size = getattr(owner._cfg, 'value_expansion_batch_size', None)
        self.batch_size = self.agent.cfg.rollouts if configured_batch_size is None else configured_batch_size
        for name, value in (("value_expansion_horizon", self.horizon),
                            ("value_expansion_batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.agent.state_space is not None:
            raise ValueError("Value expansion requires identical actor and critic observations")
        if {id(p) for p in self.agent.policy.parameters()} & {id(p) for p in self.agent.value.parameters()}:
            raise ValueError("Value expansion requires separate actor and critic parameters")
        if owner._raw_cfg.get("rewards_shaper") is not None:
            raise ValueError("Value expansion supports rewards_shaper_scale, not custom reward shapers")

    def _value(self, observations):
        values, _ = self.agent.value.act(
            {"observations": self.agent._observation_preprocessor(observations), "states": None}, role="value")
        return self.agent._value_preprocessor(values, inverse=True)

    @torch.no_grad()
    def imagine(self, roots, *, member_ids=None):
        """Return fixed targets at every depth, in physical (unnormalized) value units."""
        if hasattr(self.model, 'begin_rollout'):
            if member_ids is None:
                self.model.begin_rollout(roots)
            else:
                self.model.begin_rollout(roots, member_ids=member_ids)
        obs, path, rewards, continues, valid = roots, [], [], [], []
        alive = torch.ones(len(roots), 1, device=roots.device, dtype=torch.bool)
        gamma, scale = self.agent.cfg.discount_factor, self.owner._cfg.rewards_shaper_scale
        for _ in range(self.horizon):
            path.append(obs)
            valid.append(alive)
            actions, _ = self.agent.policy.act(
                {"observations": self.agent._observation_preprocessor(obs), "states": None}, role="policy")
            next_obs, reward, terminated, truncated = self.model.step(obs, actions)
            reward = reward * scale
            if self.agent.cfg.time_limit_bootstrap:
                reward = reward + gamma * self._value(next_obs) * (truncated & ~terminated)
            continuation = alive & ~(terminated | truncated)
            rewards.append(torch.where(alive, reward, 0.))
            continues.append(continuation)
            obs = torch.where(alive, next_obs, obs)
            alive = alive & continuation
        target = self._value(obs)
        targets = []
        for reward, continuation in zip(reversed(rewards), reversed(continues)):
            target = reward + gamma * continuation * target
            targets.append(target)
        targets = torch.stack(list(reversed(targets)))
        if not torch.isfinite(targets).all():
            raise RuntimeError("Value expansion produced non-finite critic targets")
        return torch.stack(path), targets, torch.stack(valid)

    def prepare(self):
        observations = self.agent.memory.get_tensor_by_name("observations")
        roots = observations.reshape(-1, observations.shape[-1])
        env = getattr(self.model, 'env', None)
        members = None
        if (getattr(env, 'transition_backend', None) == 'learned'
                and hasattr(env._dynamics_model, 'member_parameters')):
            assigned = getattr(env, '_dynamics_member_ids', None)
            if observations.ndim != 3 or assigned is None or len(assigned) != observations.shape[1]:
                raise ValueError('Transfer value expansion requires rollout memory ordered as time, environment, features')
            members = assigned.repeat(observations.shape[0])
        modules = [module for module in (self.agent.policy, self.agent.value, self.owner._ccm_gen,
                   getattr(self.agent, "contraction_metric", None)) if module is not None]
        if self.owner._neural_dynamics is not None:
            modules.append(self.owner._neural_dynamics)
        modes = [module.training for module in modules]
        devices = [roots.device.index] if roots.is_cuda else []
        real_targets, imagined_obs, imagined_targets = [], [], []
        try:
            for module in modules:
                module.eval()
            # Imagination must not advance real-policy exploration or retry RNG streams.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for start in range(0, len(roots), self.batch_size):
                    chunk_members = None if members is None else members[start:start + self.batch_size]
                    path, targets, valid = self.imagine(roots[start:start + self.batch_size], member_ids=chunk_members)
                    real_targets.append(targets[0])
                    mask = valid[1:].squeeze(-1)
                    imagined_obs.append(path[1:][mask])
                    imagined_targets.append(targets[1:][mask])
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)
        return torch.cat(real_targets), torch.cat(imagined_obs), torch.cat(imagined_targets)

    def fit_intermediate(self, observations, targets):
        """One critic-only TD-k step; actor/metric/dynamics receive no gradients."""
        if not len(observations):
            return
        agent = self.agent
        agent.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        for start in range(0, len(observations), self.batch_size):
            obs, target = observations[start:start + self.batch_size], targets[start:start + self.batch_size]
            with torch.autocast(device_type=agent._device_type, enabled=agent.cfg.mixed_precision):
                prediction, _ = agent.value.act(
                    {"observations": agent._observation_preprocessor(obs), "states": None}, role="value")
                target = agent._value_preprocessor(target)
                loss = F.mse_loss(prediction, target) * len(obs) / len(observations)
            agent.scaler.scale(agent.cfg.value_loss_scale * loss).backward()
            loss_total += loss.detach().item()
        from skrl import config
        if config.torch.is_distributed:
            agent.value.reduce_parameters()
        if agent.cfg.grad_norm_clip > 0:
            agent.scaler.unscale_(agent.optimizer)
            torch.nn.utils.clip_grad_norm_(agent.value.parameters(), agent.cfg.grad_norm_clip)
        agent.scaler.step(agent.optimizer)
        agent.scaler.update()
        agent.track_data("Loss / Value expansion loss", loss_total)


def install_value_expansion(owner, env):
    """Keep skrl's PPO optimizer loop and real-data GAE, adding MVE for its critic."""
    if getattr(owner, "_value_expansion", None) is not None:
        return
    expansion = ValueExpansion(owner, TrackingExpansionModel(owner, env))
    agent = owner._rl_agent
    original_update = agent.update

    def update(*, timestep, timesteps):
        targets, observations, intermediate_targets = expansion.prepare()
        with expanded_critic_targets(targets):
            original_update(timestep=timestep, timesteps=timesteps)
        expansion.fit_intermediate(observations, intermediate_targets)

    agent.update = update
    owner._value_expansion = expansion
