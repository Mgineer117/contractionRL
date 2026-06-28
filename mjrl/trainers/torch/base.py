"""Base trainer for mjrl: shared evaluation via the parallel sampler.

Evaluation rolls out the current policy with an ``OnlineSampler`` whose worker
count equals ``num_agent`` (num_agent == num_worker) and reports tracking
metrics computed from the collected ``[x, xref, uref]`` observations.
"""

from __future__ import annotations

import numpy as np

from mjrl.utils.sampler import OnlineSampler


class BaseTrainer:
    def __init__(self, env, agent, cfg: dict):
        self.env = env
        self.agent = agent
        self.cfg = cfg or {}

        self.num_agent = int(self.cfg.get("num_agent", 1))
        self.seed = self.cfg.get("seed", 0)

        # dims for the sampler
        x_dim = int(getattr(env, "num_dim_x"))
        u_dim = int(getattr(env, "num_dim_control"))
        state_dim = int(np.prod(env.observation_space.shape))
        episode_len = int(getattr(env, "max_episode_len"))

        self.sampler = OnlineSampler(
            state_dim=state_dim,
            u_dim=u_dim,
            episode_len=episode_len,
            batch_size=self.num_agent * 3 * episode_len,
            num_agent=self.num_agent,
        )
        self.x_dim = x_dim

    # ------------------------------------------------------------------ #
    def evaluate(self, seed: int | None = None) -> dict:
        """Roll out the policy across ``num_agent`` workers and compute metrics."""
        self.agent.eval()
        memory, elapsed = self.sampler.collect_samples(
            self.env, self.agent, seed=self.seed if seed is None else seed
        )

        states = memory["states"]                # (N, state_dim)
        x = states[:, : self.x_dim]
        xref = states[:, self.x_dim : 2 * self.x_dim]
        err = np.linalg.norm(x - xref, axis=1)   # per-step tracking error

        rewards = memory["rewards"].reshape(-1)
        metrics = {
            "eval/tracking_error_mean": float(np.mean(err)),
            "eval/tracking_error_final": float(np.mean(err[-self.num_agent:])),
            "eval/reward_mean": float(np.nanmean(rewards)),
            "eval/num_samples": int(len(err)),
            "eval/num_workers": self.sampler.total_num_worker,
            "eval/collect_time_s": round(elapsed, 3),
            # performance_score: higher is better (negative mean tracking error)
            "eval/performance_score": float(-np.mean(err)),
        }
        return metrics

    def run(self):  # pragma: no cover - overridden
        raise NotImplementedError
