# Shared helpers + per-algorithm W&B sweep metric/parameter definitions.
# Sourced by both run_sweep.sh (generates the sweep yaml) and search.sh (the
# interactive launcher, which previews this same block before confirming).
#
# Expects ALGORITHM (and, for ALGORITHM=c2rl, RL and CM) already set in the
# caller's environment. metric_block/parameters_block print yaml fragments.

# `select`-based menu prompt; echoes the chosen option to stdout.
prompt_choice() {
    local prompt_text="$1"; shift
    local opts=("$@")
    echo "$prompt_text" >&2
    select opt in "${opts[@]}"; do
        if [[ -n "$opt" ]]; then
            echo "$opt"
            return
        fi
        echo "Invalid choice, try again." >&2
    done
}

metric_block() {
    case "$ALGORITHM" in
        # ── ppo / sac / c2rl (any --rl, any --cm) — pure RL algorithms, ──── #
        # ── optimize the task reward directly. ─────────────────────────── #
        ppo|sac|c2rl)
            cat <<'EOF'
metric:
  name: "Reward / Total reward (mean)"
  goal: maximize
EOF
            ;;
        # ── c3m  /  cvstem-lqr — certificate-based controllers, optimize ── #
        # ── the contraction certificate's own score. ────────────────────── #
        c3m|cvstem-lqr)
            cat <<'EOF'
metric:
  # contraction_score = contraction_rate / overshoot — higher is better.
  # StatManagerEnvWrapper injects a fresh Stability/* dict into info["log"]
  # every env.step() regardless of algorithm, so this is always populated
  # once the eval-buffer envs complete their first episode.
  name: "Stability/contraction_score_mean"
  goal: maximize
EOF
            ;;
        # ── lqr  /  sdlqr — analytical LQR-family controllers, optimize ─── #
        # ── the normalized-error AUC instead. ────────────────────────────── #
        lqr|sdlqr)
            cat <<'EOF'
metric:
  # AUC of the normalized error curve e(t)/e(0) — minimized, since it directly
  # measures the certified contraction quantity rather than a training proxy.
  # StatManagerEnvWrapper injects a fresh Stability/* dict into info["log"]
  # every env.step() regardless of algorithm, so this is always populated
  # once the eval-buffer envs complete their first episode.
  name: "Stability/auc_mean"
  goal: minimize
EOF
            ;;
    esac
}

parameters_block() {
    case "$ALGORITHM" in
        # ══ --algorithm ppo ═══════════════════════════════════════════════ #
        ppo)
            cat <<'EOF'
parameters:
  agent.discount_factor:
    values: [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
  agent.gae_lambda:
    distribution: uniform
    min: 0.9
    max: 1.0
  agent.ratio_clip:
    distribution: uniform
    values: [0.1, 0.2, 0.3]
  agent.entropy_loss_scale:
    values: [0.0, 0.001, 0.01]
  agent.models.policy.backbone:
    # mlp (plain MLP, u = MLP(obs)) | control (CLActor, u = uref + feedback(x-xref))
    values: [mlp, control]
EOF
            ;;
        # ══ --algorithm sac ═══════════════════════════════════════════════ #
        sac)
            cat <<'EOF'
parameters:
  agent.learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  agent.discount_factor:
    values: [0.9, 0.99, 0.999]
  agent.polyak:
    distribution: log_uniform_values
    min: 1e-3
    max: 1e-1
  agent.batch_size:
    values: [128, 256, 512, 1024]
  agent.initial_entropy_value:
    distribution: uniform
    min: 0.05
    max: 0.5
EOF
            ;;
        # ══ --algorithm c3m ═══════════════════════════════════════════════ #
        c3m)
            cat <<'EOF'
parameters:
  agent.lbd:
    distribution: uniform
    min: 0.01
    max: 3.0
  agent.eps:
    distribution: log_uniform_values
    min: 1e-3
    max: 1.0
  agent.W_lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  agent.c1_c2_scale:
    distribution: uniform
    min: 0.01
    max: 1.0
  agent.models.policy.backbone:
    values: [control, control-squashed]
  agent.actor_lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  agent.actor_architecture:
    values: [[64, 64], [128, 128], [512, 512], [128, 128, 128, 128]]
  # w_lb/w_ub bound the generated metric M(x) (w_lb·I ⪯ M ⪯ w_ub·I) and must
  # stay w_lb < w_ub — swept as a joint {w_lb, w_ub} pair (wandb's nested-
  # parameter form) rather than two independent flat parameters, so bayes
  # never samples an inverted/degenerate pair. c3m's fields live under
  # `agent:` (skrl_c3m_cfg.yaml), so the outer key here is "agent" —
  # flattens to the real "agent.w_lb"/"agent.w_ub" override keys.
  agent:
    values:
      - {w_lb: 0.5, w_ub: 1.5}
      - {w_lb: 1.0, w_ub: 3.0}
      - {w_lb: 0.1, w_ub: 1.0}
      - {w_lb: 1.0, w_ub: 10.0}
      - {w_lb: 0.1, w_ub: 10.0}
      - {w_lb: 1.0, w_ub: 100.0}
EOF
            ;;
        # ══ --algorithm lqr  /  --algorithm sdlqr ═══════════════════════════ #
        lqr|sdlqr)
            cat <<'EOF'
parameters:
  agent.Q_scaler:
    distribution: log_uniform_values
    min: 0.1
    max: 10.0
  agent.R_scaler:
    distribution: uniform
    min: 0.0
    max: 1.0
EOF
            ;;
        # ══ --algorithm cvstem-lqr ══════════════════════════════════════════ #
        cvstem-lqr)
            cat <<'EOF'
parameters:
  agent.r_scaler:
    distribution: log_uniform_values
    min: 0.1
    max: 10.0
  cm.lbd:
    distribution: uniform
    min: 0.01
    max: 3.0
  cm.cm_eps:
    distribution: log_uniform_values
    min: 1e-3
    max: 1.0
EOF
            ;;
        # ══ --algorithm c2rl (all --rl / --cm combinations) ═══════════════ #
        c2rl)
            # -- common to both --rl (ppo/sac) and both --cm (ccm/cvstem) -- #
            cat <<'EOF'
parameters:
  agent.discount_factor:
    distribution: uniform
    values: [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
  agent.models.policy.backbone:
    # mlp (plain MLP, u = MLP(obs)) | control (CLActor, u = uref + feedback(x-xref))
    values: [mlp, control]
  cm.cm_eps:
    distribution: log_uniform_values
    min: 1e-3
    max: 1.0
  cm.lbd:
    distribution: uniform
    min: 0.01
    max: 3.0
EOF
            # -- --rl ppo / --rl sac (RL sub-agent driving the policy) ----- #
            if [[ "$RL" == "ppo" ]]; then
                cat <<'EOF'
  agent.gae_lambda:
    distribution: uniform
    min: 0.9
    max: 1.0
EOF
            else
                cat <<'EOF'
  agent.polyak:
    distribution: log_uniform_values
    min: 1e-3
    max: 1e-1
EOF
            fi
            # -- --cm cvstem / --cm ccm (cmg_method — see c2rl.py module ---- #
            # -- docstring): cvstem's cvstem_r_scaler sets R in the SDP's --- #
            # -- Riccati term; ccm has no SDP, so cmg_regress_lr (the CMG's - #
            # -- own direct-synthesis learning rate) is swept instead. ----- #
            if [[ "$CM" == "cvstem" ]]; then
                cat <<'EOF'
  cm.cvstem_r_scaler:
    distribution: log_uniform_values
    min: 0.01
    max: 10.0
EOF
            else
                cat <<'EOF'
  cmg.cmg_regress_lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  # w_lb/w_ub bound W(x)'s eigenvalues (w_lb·I ⪯ W ⪯ w_ub·I) and must stay
  # w_lb < w_ub — sweeping them as two independent flat parameters risks
  # bayes sampling an inverted/degenerate pair. A categorical parameter
  # whose values are {w_lb, w_ub} dicts is wandb's nested-parameter form —
  # it flattens to the SAME "cm.w_lb"/"cm.w_ub" CLI flags as flat keys
  # would (see W&B sweep docs on nested parameters), but samples each pair
  # jointly so w_lb < w_ub always holds. ccm-only: w_lb/w_ub are used by
  # both cmg_method's synthesis (see c2rl.py module docstring), but cvstem
  # already fixes them indirectly via its cm_data_path cache key.
  cm:
    values:
      - {w_lb: 0.5, w_ub: 1.5}
      - {w_lb: 1.0, w_ub: 3.0}
      - {w_lb: 0.1, w_ub: 1.0}
      - {w_lb: 1.0, w_ub: 10.0}
      - {w_lb: 0.1, w_ub: 10.0}
      - {w_lb: 1.0, w_ub: 100.0}
EOF
            fi
            ;;
    esac
}
