"""Deterministic evaluation and sampling deadlines for transfer-path control."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
import time

import torch


def evaluation_act(agent, observations, states=None):
    """Use the policy mean and bypass random warmup for evaluation/references."""
    deterministic = getattr(agent, 'act_deterministic', None)
    if callable(deterministic):
        return deterministic(observations, states)
    timestep = max(0, int(getattr(agent.cfg, "random_timesteps", 0)))
    device = torch.device(getattr(agent, "device", "cpu"))
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        actions, outputs = agent.act(observations, states, timestep=timestep, timesteps=max(timestep + 1, 1))
    return outputs.get("mean_actions", actions), outputs


def _policy_mean(agent, observations, states):
    # Sampling then restoring the global RNG would race with simulator disturbances.
    owner = getattr(agent, "_rl_agent", agent)
    policy = getattr(owner, "models", {}).get("policy")
    if policy is None:
        timestep = max(0, int(getattr(agent.cfg, "random_timesteps", 0)))
        actions, outputs = agent.act(observations, states, timestep=timestep, timesteps=timestep + 1)
        return outputs.get("mean_actions", actions)
    inputs = {"observations": observations, "states": states}
    for name, preprocessor in (("observations", "_observation_preprocessor"), ("states", "_state_preprocessor")):
        if inputs[name] is not None and callable(getattr(owner, preprocessor, None)):
            inputs[name] = getattr(owner, preprocessor)(inputs[name])
    device = observations.device
    with torch.autocast(device_type=device.type, enabled=bool(getattr(owner.cfg, "mixed_precision", False))):
        mean, outputs = policy.compute(inputs, role="policy")
        from contractionRL.agents.utils.RL.models import _TanhSquashMixin
        if isinstance(policy, _TanhSquashMixin):
            residual = outputs.get("residual")
            return (policy._rescale(torch.tanh(mean)) if residual is None
                    else policy._rescale_residual(torch.tanh(mean), residual)[0])
        if getattr(policy, "_g_clip_mean_actions", False):
            mean = torch.clamp(mean, policy._g_min_actions, policy._g_max_actions)
        return mean


class EvaluationControl:
    """Run one inference at a time; only the main thread touches the simulator.

    Transfer-path evaluation uses wall-clock deadlines separated by step_dt.
    Other tasks keep their existing synchronous evaluation behavior.
    """

    def __init__(self, env, agent):
        raw = env.unwrapped
        self._raw = raw
        task = getattr(getattr(raw, "spec", None), "id", "") or ""
        self.enabled = task.split(":")[-1].startswith("Isaac-Transfer-path-")
        self.agent = agent
        self._pool = None
        self._future = None
        if not self.enabled:
            return
        self.dt = float(raw.step_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Control sampling time must be finite and positive")
        self.device = torch.device(env.device)
        self._low = torch.as_tensor(env.action_space.low, device=self.device, dtype=torch.float32)
        self._high = torch.as_tensor(env.action_space.high, device=self.device, dtype=torch.float32)
        self._held = torch.zeros((env.num_envs, *env.action_space.shape), device=self.device).clamp(self._low, self._high)
        self._generation = torch.zeros(env.num_envs, dtype=torch.long)
        self._source_step = torch.full((env.num_envs,), -1, dtype=torch.long)
        self._steps = torch.zeros(env.num_envs, dtype=torch.long)
        self._delays = torch.zeros_like(self._steps)
        self._holds = torch.zeros_like(self._steps)
        self._step = 0
        self._deadline = None
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="control-inference")

    def __enter__(self):
        if self.enabled:
            self._previous_penalty = getattr(self._raw, '_ood_penalty_enabled', True)
            self._raw._ood_penalty_enabled = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                if exc_type is None and self._future is not None:
                    self._future.result()
        finally:
            if self.enabled:
                self._raw._ood_penalty_enabled = self._previous_penalty

    def _infer(self, observations, states, ready):
        context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with context, torch.no_grad():
            if ready is not None:
                self._stream.wait_event(ready)
                observations.record_stream(self._stream)
                if states is not None:
                    states.record_stream(self._stream)
            actions = _policy_mean(self.agent, observations, states).detach().to(self.device, dtype=torch.float32)
            if actions.shape != self._held.shape or not torch.isfinite(actions).all():
                raise ValueError("Controller returned invalid actions")
            actions = actions.clamp(self._low, self._high)
        if self._stream is not None:
            # A completed Future must mean the device result is ready as well.
            self._stream.synchronize()
        return actions, time.monotonic()

    def _receive(self, fresh, deadline):
        if self._future is None or not self._future.done():
            return
        actions, completed_at = self._future.result()
        if completed_at > deadline:
            return
        self._future = None
        valid = self._request_generation == self._generation
        if self._stream is not None:
            actions.record_stream(torch.cuda.current_stream(self.device))
        self._held = torch.where(valid.to(self.device).unsqueeze(-1), actions, self._held)
        self._source_step[valid] = self._request_step
        fresh |= valid

    def act(self, observations, states=None, *, active=None):
        if not self.enabled:
            with torch.no_grad():
                return evaluation_act(self.agent, observations, states)[0]
        started = time.monotonic()
        deadline = started + self.dt if self._deadline is None else max(started, self._deadline)
        fresh = torch.zeros_like(self._steps, dtype=torch.bool)
        self._receive(fresh, started)
        if self._future is None:
            observations = observations.detach().clone()
            states = states.detach().clone() if states is not None else None
            ready = None
            if self._stream is not None:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
            self._request_generation = self._generation.clone()
            self._request_step = self._step
            self._future = self._pool.submit(self._infer, observations, states, ready)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._receive(fresh, deadline)
        active = torch.ones_like(fresh) if active is None else active.detach().reshape(-1).bool().cpu()
        self._steps += active
        self._delays += active & (self._source_step != self._step)
        self._holds += active & ~fresh
        self._step += 1
        self._deadline = deadline + self.dt
        return self._held.clone()

    def after_step(self, done):
        if self.enabled:
            done = done.detach().reshape(-1).bool().cpu()
            self._generation += done
            self._held[done.to(self.device)] = torch.zeros_like(self._low).clamp(self._low, self._high)
            self._source_step[done] = -1

    def summary(self, *, num_envs=None):
        if not self.enabled:
            return {}
        steps = self._steps[:num_envs]
        delays = self._delays[:num_envs]
        holds = self._holds[:num_envs]
        total = int(steps.sum())
        return {"control_delay_steps": int(delays.sum()),
                "control_hold_steps": int(holds.sum()), "control_steps": total,
                "control_delay_steps_mean": float(delays.float().mean()),
                "control_delay_fraction": float(delays.sum()) / max(total, 1)}


def rollout_stability(env, agent, *, max_steps: int) -> dict:
    """Evaluate one episode per environment, including transfer control deadlines."""
    import torch

    agent.enable_training_mode(False)
    for model in getattr(agent, "models", {}).values():
        if model is not None and hasattr(model, "eval"):
            model.eval()

    from contractionRL.tasks.direct.common.transfer_dynamics import force_wrapper_reset
    force_wrapper_reset(env)
    obs, _ = env.reset()
    states = env.state() if hasattr(env, "state") else None
    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    computes_before = getattr(env, "_compute_count", None)

    with EvaluationControl(env, agent) as control:
        for _step in range(max_steps):
            actions = control.act(obs, states, active=~finished)
            obs, _r, terminated, truncated, _ = env.step(actions)
            states = env.state() if hasattr(env, "state") else None
            done = (terminated | truncated).view(num_envs)
            control.after_step(done)
            finished |= done
            if finished.all():
                break
        control_metrics = control.summary()

    if computes_before is not None and getattr(env, "_compute_count", None) == computes_before:
        print("  [WARN] Stability buffer did not complete; withholding previous evaluation metrics.")
        return control_metrics
    return {**env.stability_summary(), **control_metrics}
