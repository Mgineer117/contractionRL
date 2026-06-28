"""Eval-only trainer for analytical controllers (LQR, SD-LQR).

These controllers have no learnable parameters, so ``run`` simply evaluates the
policy with the parallel sampler (num_agent == num_worker) and reports metrics.
"""

from __future__ import annotations

from mjrl.trainers.torch.base import BaseTrainer


class EvalTrainer(BaseTrainer):
    def __init__(self, env, agent, cfg: dict):
        super().__init__(env, agent, cfg)
        self.num_eval_rounds = int(cfg.get("num_eval_rounds", 1))

    def run(self):
        self.agent.eval()
        all_metrics = []
        for r in range(self.num_eval_rounds):
            metrics = self.evaluate(seed=self.seed + r * 7919)
            all_metrics.append(metrics)
            print(
                f"[{self.agent.name} eval {r + 1}/{self.num_eval_rounds}] "
                f"track_err_mean={metrics['eval/tracking_error_mean']:.4f} "
                f"track_err_final={metrics['eval/tracking_error_final']:.4f} "
                f"reward_mean={metrics['eval/reward_mean']:.4f} "
                f"workers={metrics['eval/num_workers']} "
                f"samples={metrics['eval/num_samples']}"
            )
        return all_metrics
