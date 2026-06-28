"""C3M trainer: supervised contraction-metric synthesis loop.

Repeatedly calls ``agent.learn()`` (each call does ``cmg_updates_per_policy_update``
metric updates + one controller update) for ``epochs`` epochs, periodically
evaluating with the parallel sampler.
"""

from __future__ import annotations

import time

from tqdm import tqdm

from mjrl.trainers.torch.base import BaseTrainer


class C3MTrainer(BaseTrainer):
    def __init__(self, env, agent, cfg: dict):
        super().__init__(env, agent, cfg)
        self.epochs = int(cfg.get("epochs", 30000))
        self.eval_interval = int(cfg.get("eval_interval", 1000))
        self.log_interval = int(cfg.get("log_interval", 200))

    def run(self):
        start = time.time()
        n_epochs = getattr(self.agent, "cmg_updates_per_policy_update", 1)
        eval_idx = 0
        self.agent.train()

        with tqdm(total=self.epochs, desc=f"{self.agent.name} training") as pbar:
            while pbar.n < self.epochs:
                step = pbar.n + 1
                loss_dict, _supp, update_time = self.agent.learn()
                pbar.update(n_epochs)

                if step % self.log_interval < n_epochs:
                    pbar.set_postfix(
                        loss=f"{loss_dict.get('C3M/loss/loss', float('nan')):.3g}",
                        pd=f"{loss_dict.get('C3M/loss/pd_loss', float('nan')):.3g}",
                    )

                if step >= self.eval_interval * (eval_idx + 1):
                    eval_idx += 1
                    self.agent.eval()
                    metrics = self.evaluate()
                    tqdm.write(
                        f"[eval @ {step}] "
                        f"track_err={metrics['eval/tracking_error_mean']:.4f} "
                        f"reward={metrics['eval/reward_mean']:.4f} "
                        f"workers={metrics['eval/num_workers']}"
                    )
                    self.agent.train()

        print(f"[C3M] training done in {(time.time() - start) / 60:.1f} min")
