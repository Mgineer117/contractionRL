"""Generate one c2rl-ppo sweep config per gamma.

Gamma is the TREATMENT: tuned around, never tuned. Each gamma gets its own
Bayesian search, so the estimand is "effect of gamma under OPTIMAL tuning" --
sharing one tuned config confounds it with "the defaults happen to suit low
gamma", the artifact that made an earlier segway reading wrong.

HELD FIXED by the ENV YAMLS and simply absent from the search space, so no trial
can vary them and the decision lives in exactly one place:
    use_state_norm / use_value_norm / use_reward_norm = false
    learning_rate_scheduler = null                      (no KLAdaptiveLR)
    caps_temporal_scale / caps_spatial_scale = 0.0      (regularizer fully off)

REMOVED entirely: the cm.* family and every residual_pretrain_* axis. The
contraction metric is an INPUT to this experiment -- built once per env by
build_cm_dataset from that env's yaml -- not something a trial tunes. (The
cm_build_if_missing machinery stays in the code at its False default, so
restoring a cm.* sweep later is a config edit, not a code change.)

SEARCHED, split into optimizer / PPO / architecture:
    learning_rate, kl_threshold, std_dev_annealing, entropy_loss_scale
    gae_lambda, ratio_clip, learning_epochs, mini_batches, rollouts, grad_norm_clip
    xref_encoder, xref_encoder_stride, net_hidden
"""
import pathlib

GAMMAS = ["0.01", "0.1", "0.5", "0.9", "0.99", "0.999"]
PIN_LEN = 500

TEMPLATE = """label: c2rl (rl=ppo, cm=cvstem) gamma={g}
algorithm: c2rl-ppo
num_envs: 1024

# Bayes, not grid: the optimizer x PPO x architecture cross product is far too
# large to enumerate, and a surrogate is the point.
method: bayes

metric:
  # The DISCOUNTED return -- the objective the agent is actually given, and the
  # only quantity whose maximizer is pi*_gamma. Tuning against undiscounted AUC
  # instead would select the policy that happens to track well while using gamma
  # as an algorithmic device, which severs the link between what the theory
  # proves and what the experiment measures. It is also NOT one of the reported
  # outcomes, so selecting on it does not make any reported effect definitional.
  #
  # gamma-dependence of this criterion is harmless: it selects hyperparameters
  # WITHIN a gamma and is never compared across them -- the cross-gamma
  # comparison is on the fixed outcome metrics (AUC, lambda, C, C/lambda).
  name: "Reward/discounted_return_mean"
  goal: maximize

runner:
  # A run that dies without writing the metric is IGNORED by bayes bookkeeping,
  # so nothing is learned and the same dead region gets resampled. The wrapper
  # turns a failure into a real datapoint. NEGATIVE because the goal is now
  # maximize: the poison value must be worse than any real score, and the
  # Mahalanobis reward is itself negative.
  wrapper: true
  bad_value: -1.0e6

extra_flags:
  - "--ref_offset"
  - "1"
  # PINNED, and this is the treatment protocol rather than a convenience. AUTO
  # sizes the window to 1/(1-gamma), which makes observation width a FUNCTION of
  # the treatment (19-wide at gamma=0.5 against 2504-wide at 0.999) -- a fat-hand
  # intervention no post-hoc analysis can undo. {pin} == max_episode_len for
  # car/car_weak/segway/cartpole, so every policy sees the whole reference
  # trajectory and has a Markov state regardless of its discount.
  - "--ref_length"
  - "{pin}"

parameters:
  # ── The treatment: fixed for this sweep ──────────────────────────────────── #
  # Tuned AROUND, never tuned. The window does not track it -- ref_length is
  # pinned above -- so the architecture is identical across all six sweeps.
  agent.discount_factor:
    value: {g}

  # NOT LISTED, deliberately: use_state_norm / use_value_norm / use_reward_norm
  # (false), learning_rate_scheduler (null) and caps_temporal_scale /
  # caps_spatial_scale (0.0). All twelve classic c2rl-ppo yamls already set
  # exactly these, so restating them here would be a second copy of the same
  # decision -- and a sweep parameter is the wrong place to keep it, since it
  # would then have to be edited in seven files instead of one whenever it
  # changes. They are absent from the search space, so no trial can vary them.

  # ── Searched: optimizer ──────────────────────────────────────────────────── #
  # Log-uniform over three decades. The per-env yamls pin 1.0e-5, chosen under a
  # KL-adaptive schedule that is now off, so it is not a trustworthy centre and
  # the range brackets it widely on purpose.
  agent.learning_rate:
    distribution: log_uniform_values
    min: 1.0e-6
    max: 1.0e-3
  # PPO's own early stop. With the LR fixed this is the ONLY trust-region
  # control, and the stop is per-MINIBATCH, not on the epoch mean.
  agent.kl_threshold:
    values: [0.004, 0.008, 0.016, 0.032, 0.064]
  # A live hypothesis for the high-gamma instability: policy std stayed at
  # 0.65-0.80 at high gamma against 0.22 at low gamma, and residual exploration
  # noise on an unstable plant is what turns a transient into an overshoot.
  agent.std_dev_annealing:
    values: [true, false]
  # 0.0 keeps the current behaviour in the ladder as a control.
  agent.entropy_loss_scale:
    values: [0.0, 0.001, 0.01]

  # ── Searched: PPO / GAE ──────────────────────────────────────────────────── #
  agent.gae_lambda:
    values: [0.9, 0.95, 0.975, 0.99, 1.0]
  agent.ratio_clip:
    values: [0.1, 0.2, 0.3]
  agent.learning_epochs:
    values: [2, 5, 10]
  agent.mini_batches:
    values: [4, 8, 16]
  # Steps per env per update. The per-env yamls disagree wildly here (car 4,
  # segway/cartpole 24), which is itself a reason to search rather than inherit.
  agent.rollouts:
    values: [4, 8, 16, 24]
  agent.grad_norm_clip:
    values: [0.5, 1.0, 5.0]

  # ── Searched: architecture ───────────────────────────────────────────────── #
  # "mlp" flattens the (strided) points; "gru"/"attn" consume them as a sequence
  # (nn_modules.PreviewSequenceEncoder). Both synthetic keys fan out to BOTH
  # models.policy.* and models.critic.* (train_utils.apply_wandb_sweep_overrides),
  # so a trial always uses the same encoder and stride on both sides.
  # models.policy.backbone stays pinned at `control`: the CLActor
  # u = uref + feedback form is load-bearing for the Mahalanobis reward, not a
  # tuning axis.
  xref_encoder:
    values: [mlp, gru, attn]
  # Sized for the PINNED {pin}-point window: keeps {keeps2} points respectively.
  # STRIDE 1 IS DELIBERATELY ABSENT. Attention is quadratic in sequence length,
  # so attn x stride-1 over all {pin} points dominated the cost of the entire
  # sweep -- measured at 1% of training after 25 min, i.e. a 40-130 h trial
  # against the wall clock, which is why the first launch finished zero trials.
  # Dropping it does NOT weaken the Markov argument: stride subsamples the window
  # for ENCODING while the window still SPANS all {pin} steps, so the effective
  # horizon the observation covers is unchanged. It only removes the densest
  # encoding of that same span.
  xref_encoder_stride:
    values: [5, 10, 25, 50]
  # Hidden layers, actor and critic together. Reaches models.*.network[0].layers
  # through a synthetic fanout, because `network` is a LIST and a dotted sweep
  # key cannot index it -- without that handler layer shape is unreachable from a
  # sweep and "architecture" would mean the encoder alone.
  net_hidden:
    values:
      - [128, 128]
      - [256, 256]
      - [512, 512]
      - [128, 128, 128]
      - [256, 256, 256]
"""

keeps2 = "/".join(str(PIN_LEN // s) for s in (5, 10, 25, 50))
for g in GAMMAS:
    p = pathlib.Path(f"search/configs/c2rl-ppo-g{g}.yaml")
    p.write_text(TEMPLATE.format(g=g, pin=PIN_LEN, keeps2=keeps2))
    print(f"wrote {p}")
