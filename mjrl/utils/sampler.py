"""Parallel Monte-Carlo rollout sampler (ported from CAC-dev ``utils/sampler.py``).

Key change for mjrl: the number of parallel workers is taken directly from
``num_agent`` (the env-level "number of agents"), so **num_agent == num_worker**.
Each worker rolls out ``episodes_per_worker`` episodes of the env under the
policy and ships the transitions back through a multiprocessing queue.

Set ``num_agent`` to control parallelism; the legacy auto-derivation from
``batch_size`` is used only when ``num_agent`` is None.
"""

from __future__ import annotations

import random
import time
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


def temp_seed(seed, pid):
    """Per-worker seeding so parallel workers produce distinct trajectories."""
    rand_int = random.randint(0, 1_000_000)
    seed = seed + pid + rand_int
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


class OnlineSampler:
    def __init__(
        self,
        state_dim: int,
        u_dim: int,
        episode_len: int,
        batch_size: int,
        num_agent: int | None = None,
        episodes_per_worker: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.episode_len = episode_len
        self.batch_size = batch_size

        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len

        # num_agent IS the worker count. Fall back to batch-size derivation only
        # when it is not provided (preserves CAC-dev behaviour).
        if num_agent is not None:
            self.total_num_worker = max(1, int(num_agent))
        else:
            self.total_num_worker = ceil(self.batch_size / self.thread_batch_size)
        self.num_agent = self.total_num_worker

        torch.set_num_threads(1)  # avoid CPU oversubscription across workers

    def get_reset_data(self, batch_size):
        return dict(
            states=np.full((batch_size, self.state_dim), np.nan, dtype=np.float32),
            next_states=np.full((batch_size, self.state_dim), np.nan, dtype=np.float32),
            controls=np.full((batch_size, self.u_dim), np.nan, dtype=np.float32),
            rewards=np.full((batch_size, 1), np.nan, dtype=np.float32),
            terminations=np.full((batch_size, 1), np.nan, dtype=np.float32),
            truncations=np.full((batch_size, 1), np.nan, dtype=np.float32),
            logprobs=np.full((batch_size, 1), np.nan, dtype=np.float32),
            entropys=np.full((batch_size, 1), np.nan, dtype=np.float32),
        )

    def collect_samples(self, env, policy, seed: int | None = None):
        """Collect transitions in parallel across ``num_agent`` workers."""
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))
        policy.to_device(torch.device("cpu"))

        max_retries = 25
        all_collected_batches = []
        total_samples_so_far = 0

        with mp.Manager() as manager:
            queue = manager.Queue()
            for attempt in range(max_retries):
                processes = []
                worker_memories = [None] * self.total_num_worker
                current_seed = seed + (attempt * 1000) if seed is not None else None

                for i in range(self.total_num_worker):
                    p = mp.Process(
                        target=self.collect_trajectory,
                        args=(i, queue, env, policy, current_seed),
                    )
                    processes.append(p)
                    p.start()

                expected = len(processes)
                collected = 0
                start_wait = time.time()
                while collected < expected:
                    if time.time() - start_wait > 300:
                        print(f"[Warning] collection timeout on attempt {attempt + 1}")
                        break
                    try:
                        pid, data = queue.get(timeout=5.0)
                        if worker_memories[pid] is None:
                            worker_memories[pid] = data
                            collected += 1
                    except Empty:
                        continue

                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    p.join()

                valid = [wm for wm in worker_memories if wm is not None]
                all_collected_batches.extend(valid)
                total_samples_so_far += sum(len(wm["states"]) for wm in valid)

                if total_samples_so_far >= (0.8 * self.batch_size):
                    break
            else:
                raise RuntimeError(
                    f"Failed to collect sufficient samples. "
                    f"Total collected: {total_samples_so_far} (Threshold: {0.8 * self.batch_size})"
                )

            memory = {}
            for wm in all_collected_batches:
                for key, val in wm.items():
                    memory[key] = np.concatenate((memory[key], val), axis=0) if key in memory else val

        policy.to_device(device)
        return memory, time.time() - t_start

    def collect_trajectory(self, pid, queue, env, policy: nn.Module, seed: int | None = None):
        data = self.get_reset_data(batch_size=self.thread_batch_size + self.episode_len)
        seed = temp_seed(seed, pid)

        current_step = 0
        for ep in range(self.episodes_per_worker):
            obs, _ = env.reset(seed=seed + ep)
            for t in range(self.episode_len):
                with torch.no_grad():
                    a, metaData = policy(obs)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]
                    next_obs, rew, term, trunc, _ = env.step(a)
                    done = term or trunc

                data["states"][current_step + t] = obs
                data["next_states"][current_step + t] = next_obs
                data["controls"][current_step + t] = a
                data["rewards"][current_step + t] = rew
                data["terminations"][current_step + t] = term
                data["truncations"][current_step + t] = trunc
                data["logprobs"][current_step + t] = metaData["logprobs"].detach().numpy()
                data["entropys"][current_step + t] = metaData["entropy"].detach().numpy()

                if done:
                    current_step += t + 1
                    break
                obs = next_obs

        for k in data:
            data[k] = data[k][:current_step]
        return queue.put([pid, data]) if queue is not None else data
