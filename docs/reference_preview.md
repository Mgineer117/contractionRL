# GRU context for tracking policies and critics

The base tracking state is `(x, xref, xi)`. Policy and critic independently select `null` (no GRUs; base state only) or `gru` (base state plus separate past and future summaries). GRU settings belong in the shared `models.encoder` block:

```yaml
models:
  encoder:
    feature_dim: null  # null defaults to 3 * x_dim per GRU.
    encoder_window: null  # null uses ceil(1 / (1 - discount_factor)) steps per past/future window.
  policy:
    encoder: null  # Compact policy without past/future GRUs.
  value:
    encoder: gru  # Privileged critic with separate past/future GRUs.
```

This example uses an MLP policy and a privileged critic with two GRUs followed by its MLP. The actor ignores the extra observation windows; the critic uses them during learning to estimate values and guide policy updates. This keeps the deployed policy compact without requiring those context encoders. PPO and ContractionPPO name the critic role `value`; C2RL-PPO uses `critic`; SAC uses `critic_1`, with the other Q networks and target networks inheriting its encoder. Set both roles to `gru` to give both networks context; their weights remain independent.

All value/critic configs default to `encoder: gru`; policy settings are independent. The native quadruped, humanoid and manipulator velocity-tracking tasks currently expose flat observations without these past/future tracking windows. Their GRU configs therefore raise a context error until that support is implemented. An explicit `--critic_encoder null` disables their GRUs.

`feature_dim` is the output width of each GRU. A null value defaults to `3 * x_dim`. `encoder_window` is the number of control steps in each window. A null value resolves to `ceil(1 / (1 - discount_factor))`, rounded at integer boundaries. For gamma 0.9 this gives ten future and ten past rows. An explicit positive integer overrides that horizon. Each GRU has internal hidden width 64.

Future references cover steps k+1 through k+H, repeating the terminal reference beyond the episode. History covers the last H measured states and the controls actually applied from those states. Unavailable rows are masked. Resets clear only the affected environment's history; terminal observations retain the old episode's final history. The windows expose no future disturbances or hidden simulator gust state.

Future GRUs read farthest to nearest; history GRUs read oldest to newest. Hidden states start at zero for each window. Rollout memory stores raw windows, and every minibatch re-encodes them with current weights. Policy GRUs receive policy gradients; critic GRUs receive value or Q gradients. There is no persistent recurrent state across decisions.

The base features preserve relative positions and periodic angle embeddings. Sequence positions are relative to the current robot position, sequence angles use sine/cosine, and history includes applied controls. Raw physical coordinates remain available to dynamics and metric computations.

PPO, SAC, C2RL, ContractionPPO and MBPO support this context. C3M rejects GRU settings because its offline state sampler has no measured history. Training, playback, evaluation and value expansion use the same resolved context layout. Context changes neural inputs without changing initial distributions, dynamics, references, rewards, disturbances or termination rules. It does not establish disturbance observability or stability.

The public config no longer accepts `backbone`, `preview_steps`, `temporal_context`, `use_history`, `encoder_stride`, role-specific GRU widths, or a dynamics `mode`. Removed fields raise an error. The task selects exact dynamics for `classic-*` and `Isaac-Classic-*`, or learned approximate control-affine dynamics for model-dependent algorithms on PhysX `Isaac-*` tasks. Historical checkpoints containing removed fields require their archived source/config.
