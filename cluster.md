# UIUC Campus Cluster runbook

Everything needed to run this repo's sweeps on the UIUC Campus Cluster
(`cc-login.campuscluster.illinois.edu`, user `minjae5`). Written so a future
session can go from cold start to a running multi-partition sweep without
rediscovering any of it.

## 1. Access

Host alias `uiuc-cc` in `~/.ssh/config`, with `ControlMaster auto` /
`ControlPersist 12h` and a socket under `~/.ssh/sockets/`.

Login needs an Illinois NetID password **plus a Duo phone tap**, so an agent
can never authenticate on its own. The user must open the session once:

```bash
ssh uiuc-cc            # user runs this (or `! ssh uiuc-cc` inside Claude Code)
```

While that ControlMaster socket is alive (≤12 h), any further `ssh uiuc-cc`
reuses it with no auth. Always probe before assuming access:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 uiuc-cc echo CONNECTED
```

Never accept a password typed into chat — treat it as compromised and tell the
user to rotate it.

## 2. Environment

`wandb`, `python`, etc. are **not** on the login node's bare PATH. Every remote
command that touches them needs:

```bash
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate env_isaaclab
```

`wandb login --verify` reporting "not authenticated" is almost always this PATH
problem, not a credentials problem (`~/.netrc` already holds a valid key).

**This matters for job submission.** The generated `job.sbatch` contains no
conda activation — SLURM's default `--export=ALL` propagates the submitting
shell's environment instead. Submit from a shell where the env is already
activated, or the workers start without `wandb` on PATH.

## 3. File transfer — `scp` is broken here

`scp` fails against this login node (`scp: close remote: Failure`, an
SFTP-subsystem mismatch, not quota/permissions). Use:

```bash
ssh -o BatchMode=yes uiuc-cc "cat > path/to/dest" < local_file
```

Binary-safe. Verify afterwards (`py_compile`, a grep) — don't trust exit code
alone on this transport.

## 4. The cluster repo is a separate checkout

`~/contractionRL` is an independent clone that carries its own untracked files
(`cluster_runs/`, `scratch/`, `scratch_cfg/`, sometimes extra `data/*.npz`).
Before syncing, always:

```bash
ssh uiuc-cc "cd ~/contractionRL && git status --short && git log --oneline -2"
```

If tracked files are clean, the clean way to ship a fix is push-then-fast-forward:

```bash
# local
git push -u origin <branch>
# cluster
ssh uiuc-cc "cd ~/contractionRL && git fetch origin <branch> && git merge --ff-only origin/<branch>"
```

If the cluster has *modified tracked files* in a file you need to change, do
**not** overwrite it wholesale — download it, apply the same targeted edit to
the cluster's actual text (which drifts cosmetically from local), upload, and
`diff` to confirm only the intended change landed.

> ⚠️ The cluster's `origin` URL currently embeds a GitHub personal access token
> in plaintext (visible in `git remote -v`). That token should be rotated and
> replaced with a credential helper or SSH remote.

## 5. Quota is an INODE limit, not space

```bash
ssh uiuc-cc "quota -s | grep 'u/minjae5'"
```

`/u/minjae5`: 100 G soft / 103 G hard space, but **490 000 soft / 500 000 hard
file count** — the file count is what actually blocks writes. A tiny
`echo > file` failing with `Disk quota exceeded` while `df -h` looks fine is
always this.

Biggest consumers under `~/contractionRL`, and safe to delete (user has
authorized these categories — run artifacts only):

| Dir | Typical size | Notes |
|---|---|---|
| `wandb/` | ~2.7 G | worst offender by file count |
| `logs/` | tens of MB | per-run checkpoints + eval json |
| `search/logs/` | ~160 M | sweep dirs; holds `job.sbatch` + `STOP` — **don't delete while a sweep is live** |

```bash
ssh uiuc-cc "cd ~/contractionRL && nohup bash -c 'rm -rf logs wandb' > /tmp/crl_cleanup.log 2>&1 & disown"
```

Run it backgrounded — deleting tens of thousands of files over the network FS
takes minutes. Poll `quota -s` to know when writes are safe again; the inode
count clears before `rm` finishes reclaiming space.

**Never** delete `conda/`, `.conda/`, `.cache/` — large, but installed
packages, not artifacts, and never authorized. `~/scratch` (→ `/scratch/minjae5`)
is a different filesystem with no file-count limit; cleaning it does nothing
for this quota.

## 6. Job safety — never touch `csl` / `aeose_*`

This project's jobs are always named `crl-<algorithm>-<env>-<timestamp>`. Jobs
on the `csl` partition and any named `aeose_*` belong to a **different user**
sharing the account and must never be cancelled.

```bash
# this project's jobs only
squeue -u minjae5 -h -o '%i %j' | awk '$2 ~ /^crl-/ {print $1}'
```

Never `scancel -u $USER`. Prefer `search_cluster.sh --stop` (below) over raw
`scancel` for sweep jobs, because they self-resubmit.

## 7. Partitions

| Partition | Wall-time cap | `--account` needed | GPU | Safe agents/GPU | Notes |
|---|---|---|---|---|---|
| `scavenger` | 24 h | no | H100 / H200 / L40S (≥48 GB) | 2 | Largest, heterogeneous. `search_cluster.sh` auto-excludes GPUs too old for the installed torch (V100). |
| `eng-research-gpu` | 2 d | **`huytran1-ae-eng`** | **A10, 24 GB** | **1** | ~5 nodes × 8 A10. |
| `IllinoisComputes-GPU` | 3 d | **`huytran1-ae-eng`** | A100 | 2 | ~5 nodes × 4 A100. |
| `ic-express` | **8 h** | no | **H100 MIG `1g.20gb`, 20 GB** | **1** | One node, 16 slices but only **48 CPUs total** → at `--cpus-per-task=8`, max **6 concurrent jobs**. Use `--time=07:45:00`. |

### Sizing `--agents-per-gpu` — check GPU **memory**, not slice count

A single `c2rl-*-cvstem` trial can peak near **19 GB**: `regress_cmg`
(`ncm_synthesis.py`) moves the whole CM dataset onto the GPU and then runs a
batched `torch.linalg.eigh` per minibatch. So a 20 GB MIG slice or a 24 GB A10
fits exactly **one** trial, and packing 2 agents there OOMs the second one.

Confirmed 2026-08-01: at 2 agents/GPU, one ic-express job crash-looped 22×
(`torch.OutOfMemoryError`, 21 dead W&B runs in ~7 min) while all 16 jobs on
full-size GPUs were healthy. Read the real cap first:

```bash
sinfo -p <partition> -N -o '%.20P %.10n %.45G' | sort -u   # gres names carry the size
```

Note the MIG reporting trap: on a `1g.20gb` slice the OOM message reports
`total capacity 19.62 GiB` (your slice) but lists processes from the **whole
physical H100**, so unrelated 13–18 GB processes appear in the list. They are
other users on other slices and are *not* your problem — MIG partitions memory
strictly. Diagnose from your own slice's numbers only.

**A crash-looping worker dominates the W&B dashboard.** It produces a dead run
every ~20 s while a healthy worker takes many minutes per run, so "most runs
crashed" in the UI can mean one bad job out of twenty. Count restarts per job
before concluding the sweep is broken:

```bash
for f in $D/slurm_*.out; do echo "$(basename $f): $(grep -ac 're)starting' $f)"; done | sort -t: -k2 -rn
```

A healthy job shows exactly `agents_per_gpu` restarts; anything much higher is
a crash loop.

Partitions other than scavenger/ic-express reject jobs without `--account`
(`srun: error: You must specify an account`). Verify with
`sacctmgr show associations user=minjae5 format=account,partition -p`.

Standing preference: when work is PENDING on one partition and another has
capacity, **split across both** rather than draining one first.

## 8. Launching a sweep across several partitions

`search_cluster.sh` creates a **brand-new W&B sweep on every invocation**, so
running it once per partition gives you N *separate* sweeps sharing no search
state. To get **one** sweep with workers on several partitions: create it once,
then clone its generated `job.sbatch`.

### Step 1 — create the sweep on one partition

```bash
ssh uiuc-cc "source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate env_isaaclab && \
  cd ~/contractionRL/search && ./search_cluster.sh \
    --algorithm c2rl-ppo-cvstem --env car --partition scavenger \
    --num-jobs 6 --gpus-per-job 1 --agents-per-gpu 2 --no-probe --time 24:00:00 -y"
```

Pass `--agents-per-gpu` explicitly with `--no-probe`. The smoke-test
auto-heuristic sizes packing from GPU memory headroom, which is wrong for the
CPU-bound CV-STEM algorithms and once picked 16 agents/GPU on ic-express's
20 GB MIG slices, OOM-ing every job.

This prints the sweep id and the worker script path:
`search/logs/<algo>_<env>_<timestamp>/job.sbatch`.

### Step 2 — clone that script onto the other partitions

Keep the **same `--job-name`** so a single `--stop` halts every partition. Drop
the scavenger-specific `--exclude=` node list, fix `--time` to the new
partition's cap, and add `--account` where required.

```bash
D=~/contractionRL/search/logs/<algo>_<env>_<timestamp>
for spec in \
  "eng-research-gpu|24:00:00|huytran1-ae-eng" \
  "IllinoisComputes-GPU|24:00:00|huytran1-ae-eng" \
  "ic-express|07:45:00|" ; do
    IFS='|' read -r P T A <<< "$spec"
    OUT="$D/job_${P}.sbatch"
    sed -e "s|^#SBATCH --partition=.*|#SBATCH --partition=${P}|" \
        -e "s|^#SBATCH --time=.*|#SBATCH --time=${T}|" \
        -e "/^#SBATCH --exclude=/d" "$D/job.sbatch" > "$OUT"
    [[ -n "$A" ]] && sed -i "/^#SBATCH --partition=/a #SBATCH --account=${A}" "$OUT"
    for i in $(seq 6); do sbatch "$OUT"; done
done
```

`total workers = jobs × gpus_per_job × agents_per_gpu`. The USR1 trap
resubmits `$0`, so each clone keeps its own partition forever.

To add more workers to an already-running sweep, just `sbatch` the existing
`job.sbatch` again — never rerun `search_cluster.sh`.

### Step 3 — stop it

```bash
ssh uiuc-cc "cd ~/contractionRL/search && ./search_cluster.sh --stop <log-dir-name>"
```

Writes a `STOP` sentinel first (so an in-flight USR1 trap declines to
resubmit), then `scancel --name`s every job sharing the name. Plain `scancel`
risks a job resubmitting itself out from under you.

## 9. Storage hygiene in the training code

These are already handled in-repo; don't re-solve them:

- **Sweep trials write no checkpoints.** `train.py` sets
  `checkpoint_interval = 0` when `WANDB_SWEEP_ID` is in the environment, which
  in skrl gates *both* the periodic `agent_<step>.pt` and `best_agent.pt`.
- **Normal runs keep only the newest checkpoint.** `patch_prune_checkpoints`
  (`agents/skrl/agent_patches.py`) deletes stale `agent_*.pt` after each write,
  leaving the latest snapshot plus `best_agent.pt`.
- **No TensorBoard files are ever written.** `disable_tensorboard_files()`
  (`scripts/skrl/train_utils.py`, called from `train.py`) no-ops skrl's
  `EventFileWriter`; `play.py` sets `write_interval = 0` outright. The
  `SummaryWriter` object still exists only because the W&B scalar hook wraps
  its `add_scalar` — **do not** set `write_interval = 0` in training configs to
  "disable tensorboard", it would silently kill W&B scalar logging.
- **`data/**/*.npz` is version-controlled on purpose.** `cm_data_*.npz` is the
  CV-STEM/CCM synthesis result and costs a long serial cvxpy SDP solve;
  committing it is what stops every cluster checkout from re-running it. Watch
  the 100 MB per-file GitHub limit — an Isaac `dynamics_data.npz` can reach
  ~150 MB and would need git-lfs.
