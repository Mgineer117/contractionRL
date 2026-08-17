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

## 1a. Do NOT compute on the login node

The login node is shared by every user on the cluster, and admins email about
accounts that load it. Sysadmins can and do kill offending processes. Treat
`cc-login` as a control plane only.

**Allowed on the login node** — cheap, sub-second, control-plane only:

```bash
sbatch / squeue / scancel / sacct / sinfo    # job control
ls, cat, tail, head, grep, wc, stat, mkdir   # small file inspection
ssh uiuc-cc "cat > path" < local_file        # the file-transfer recipe in §3
```

**Never on the login node** — anything that computes, allocates real memory, or
runs for more than a second or two:

```bash
python -c "import numpy/yaml/torch ..."      # even a one-liner: imports are heavy
python scripts/anything.py                   # solves, builds, training, plotting
ast.parse / py_compile over the tree         # do it locally instead
np.load of a dataset                         # a CM .npz is 8-44 MB
pip install, conda create/activate + run
```

Two ways to run real work. Prefer the first:

```bash
# 1. Batch (preferred — survives a dropped ssh, and leaves a log)
sbatch --export=ALL,TASK=classic-car-v0 build_cm24.sbatch

# 2. A one-off interactive step, when you genuinely need the output now
srun --partition=secondary --time=00:10:00 --cpus-per-task=2 --mem=8G \
     --pty bash -lc 'cd ~/contractionRL && python -u scripts/whatever.py'
```

Rules of thumb that keep this honest:

* **Check files locally, not remotely.** Parsing yaml, `ast.parse`, inspecting an
  `.npz` — do it in the local checkout. The two checkouts diverge (§4), so when
  the question is really "what does the CLUSTER copy say?", `grep` it (cheap) or
  `srun` a job that prints it, rather than running a Python interpreter on login.
* **`grep` a log, don't post-process it.** `grep -h 'key ' *.out` is fine;
  piping a log through a Python script on login is not.
* **Reading a job's own output is free.** The `.out` files are already on disk;
  `tail`/`grep` on them costs nothing.
* `srun` on `secondary` (4 h cap) is the quickest interactive allocation; use
  `scavenger` when the step needs longer than that.

## 1b. Always `python -u` in an sbatch script

Python block-buffers stdout when it is a file, and `#SBATCH --output=` makes it
one. SLURM ends a job at the wall limit with `SIGTERM`, whose default handler
terminates the interpreter **without flushing** — so everything still in the
8 KB buffer is lost.

Measured cost of getting this wrong: two `find_uniform_lambda` jobs ran
**18 h 52 m** and left 109 bytes each — just the shell `echo` before Python
started. Nothing about how far the λ ladder got was recoverable, and it is not
recoverable after the fact either: `gdb -p <pid> -ex 'call fflush(0)'` on the
compute node fails with `ptrace: Operation not permitted`.

```bash
python -u scripts/find_uniform_lambda.py ...   # right
python    scripts/find_uniform_lambda.py ...   # a timeout erases the whole run
```

Two companions to it:

* **Print before a long call, not only after.** One joint CV-STEM solve at
  N=10000 runs ~15 h inside a single cvxpy call with no progress of its own. With
  output only on completion, a working job and a hung job look identical.
* **Do not let a meaningful exit code look like a crash.** `set -e` plus a script
  that returns 2 for "infeasible" marks the job FAILED. Capture it instead:

```bash
set -uo pipefail        # no -e
python -u scripts/find_uniform_lambda.py ... ; rc=$?
echo "[$(date)] done rc=$rc"
exit 0
```

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

## 5. Quota is an inode limit, not space

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

Standing preference: when work is pending on one partition and another has
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

### Troubleshooting: `Error: Sweep <id> not found`

Every worker restarting every few seconds with

```
wandb: ERROR Find detailed error logs at: .../wandb/debug-cli.<user>.log
Error: Sweep UIUC-LIRA/contractionRL-Search/<id> not found
```

means the sweep was **deleted server-side** (e.g. cleaned up in the W&B UI).
The sweep is the only shared state, so every job on every partition dies at the
next agent start — but jobs that already picked up a trial keep training to
completion, which makes the fleet look half-healthy. Confirm with:

```bash
python -c "import wandb; print(len(list(wandb.Api().project('contractionRL-Search', entity='UIUC-LIRA').sweeps())))"
```

`0` means everything was deleted. There is no recovery — `scancel` this
sweep's jobs and go back to Step 1 to create a new sweep. Deleting a sweep in
the UI to tidy up crashed runs will kill a healthy running search.

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
