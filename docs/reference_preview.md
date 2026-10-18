# Reference preview for compact tracking policies

Keep the compact generator state `(x, xref, xi)` and optionally append a deterministic nominal reference window. For analytical Isaac-Car and classic car, configure:

```yaml
models:
  policy:
    control_mode: structured
    preview_steps: 30
    encoder: gru
    encoder_hidden: 64
    encoder_stride: 1
  critic:
    encoder: gru
    embed_dim: 64
    encoder_stride: 1
```

`preview_steps: 30` adds the current reference and30future states at the control timestep, giving31points spanning0.9seconds for the car. The end of the episode repeats the terminal reference. `preview_steps: 0` preserves the prior compact MLP baseline. Set both encoders to `mlp` for a flattened-preview control experiment.

The observation's sorted flat layout is `[x, xi, xref, xrefs.flatten()]`. `u_ref` is reconstructed from `(xref, xi)` by the existing nominal generator. The physical dynamics, references, disturbance, reward and termination rules do not change when preview is enabled.

Actor and critic use independent encoders. Preview position coordinates are displacements from the current reference position; angles use sine/cosine, and the other channels use finite bound-based scaling. The context MLP reads current-state/xi features, the learned embedding and physical box distances. The existing relative-position convention is preserved, and non-position physical configuration is retained. Current reference configuration enters through the preview and the actor's explicit tracking error.

The actor mean is `u_ref + W2(context) @ tanh(W1(context) @ error)`, followed by unchanged learned Gaussian exploration. Applied controls are clipped by the plant. Each generator and the value network use the configured MLP widths.

The GRU runs over the future window from farthest to nearest and starts from a fresh hidden state at each decision. It is a preview encoder, not a persistent episode memory. PPO stores raw references and recomputes embeddings for each current-parameter update; actor encoders receive the policy loss and critic encoders receive the value loss. No truncated backpropagation across environment timesteps is required.

Playback and checkpoint evaluation restore the preview layout and encoder configuration. Analytical value expansion generates the same nominal preview at each imagined state. Native PhysX reference-bank tasks do not yet expose this optional preview configuration and reject it explicitly.

Past observation/action history is separate from reference preview. It may help estimate persistent unobserved disturbances, but should be tested independently; no future gust values are exposed by this feature.
# Future and measured-history context

Set `models.policy.temporal_context: true`, `models.policy.use_history: true`, and both actor/critic encoders to `gru`. Leave `preview_steps: 0`: this mode derives exactly `H = ceil(1 / (1 - gamma))` points from the discount, with numerical rounding at integer boundaries. At gamma 0.9 and dt 0.03 s this is 10 steps, or 0.3 s. Set `use_history: false` for the matched future-only ablation.

The raw observation stores current x, generator parameters/time xi, current xref, future references at k+1 through k+H, and history at k-H through k-1. History rows contain actual measured state, the clipped control applied from that state, and a validity flag. Current reference control is recovered exactly from xref and xi for residual feedforward. Raw physical observations remain available to dynamics, reward, and metric code.

Each actor and critic owns independent future and history GRUs: four encoders total. Each GRU has configurable internal width (64 in the experiment) and projects to x_dim (4 for the car). Future rows run farthest-first; measured history runs oldest-first. Hidden states start at zero for each window. Actor encoders receive the PPO policy gradient; critic encoders receive the value-loss gradient. Raw windows are stored in rollout memory and re-encoded with current weights in every minibatch. PPO retains the original action log-probabilities for its denominator.

Network position features are x.position-xref.position and the opposite displacement for current xref. Every future/history position and initial-reference position parameter is expressed relative to the current robot position. Other physical components remain physical, and sequence angles use sine/cosine. Existing finite-bound scaling applies consistently. World-boundary distances remain additional inputs because termination depends on the fixed world box.

History uses only past measurements/applied controls, with no future plant disturbances. Unavailable history is masked, and resets clear only the affected environment slots; terminal observations retain the old episode's final history. This changes observation features, not plant dynamics, references, reward, or termination. It provides no stability guarantee, and the H-step, x_dim-wide summaries need not be sufficient to identify a hidden disturbance.
