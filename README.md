# When immediate imitation is not enough

Code for the post [When immediate imitation is not enough](https://khanhptnk.github.io/machine-learning/future-aware-imitation):
two toy POMDPs where some imitation error is unavoidable, and where DAgger, PPO and a GRPO-style update learn to make
different mistakes. Everything runs on a CPU in under a minute with NumPy.

```sh
uv run reproduce.py   # ~1 min: 3 methods x 2 environments x 12 seeds; writes results/runs.json, results/summary.json
uv run plots.py       # figures/*.svg (light and dark versions) from results/runs.json
```

`results/` holds the JSON from the run behind the post, so `plots.py` works without re-running anything.

## The environments

Each episode samples a hidden bit `z ~ Bernoulli(0.5)` that the expert sees and the learner doesn't. The learner makes
9 binary decisions: one at a root observation that hides `z`, then 8 downstream. Each decision earns +1 if it matches the
expert's action and −1 if not. The policy is tabular (one Bernoulli logit per observation ID), so nothing learned
downstream can generalize back to the root.

- **Information reveal.** The expert plays `z` at every step. Root action 1 leads to downstream observations that reveal
  `z`; root action 0 leads to one that still hides it.
- **Hard expert.** A correct root action enters an easy corridor, the root mistake "1 when z = 0" a recoverable corridor,
  and the mistake "0 when z = 1" a hard corridor. The expert plays action 0 throughout the easy and recoverable corridors,
  and a fresh coin flip at every step of the hard one.

In both, either root action matches the expert with probability 1/2, so DAgger's root target is exactly 1/2. Action 1
leads to a future the learner can imitate.

## Results (12 seeds, mean ± standard deviation; 50,000 evaluation episodes per seed)

| Information reveal | P(root action 1) | Signed return | Errors / episode | Success (≤ 2 errors) |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.002 | 4.00 ± 0.02 | 2.499 ± 0.009 | 54.5% ± 0.2% |
| PPO | 0.986 ± 0.000 | 7.19 ± 0.01 | 0.905 ± 0.004 | 96.2% ± 0.1% |
| GRPO | 0.967 ± 0.001 | 7.01 ± 0.01 | 0.995 ± 0.006 | 96.0% ± 0.1% |

| Hard expert | P(root action 1) | Signed return | Errors / episode | Success (≤ 2 errors) |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.001 | 6.00 ± 0.02 | 1.499 ± 0.012 | 75.9% ± 0.3% |
| PPO | 0.970 ± 0.001 | 7.18 ± 0.01 | 0.911 ± 0.004 | 96.1% ± 0.1% |
| GRPO | 0.902 ± 0.003 | 6.83 ± 0.02 | 1.085 ± 0.008 | 92.9% ± 0.2% |

DAgger's numbers have closed forms: 54.49% success (reveal) and 75.88% (hard expert).

## Training details

- **DAgger:** 40 iterations × 3,000 learner roll-outs. Expert labels at every visited observation are aggregated over all
  iterations, and the policy is their label frequencies (pseudocount 1e-3).
- **PPO:** 120 iterations × 3,072 episodes; undiscounted reward-to-go; baseline = batch-mean return-to-go at the same
  observation; advantages divided by their global standard deviation; clipped surrogate with ε = 0.2; 4 full-batch
  gradient-ascent epochs with learning rate 0.065. No value network.
- **GRPO-style:** 120 iterations × 192 groups of 8 trajectories sharing `z`; each trajectory's total return is normalized
  within its group and assigned to all of its actions; same clipped update. No KL term or reference policy.
- In the clipped update, each observation's logit gets the **mean gradient over the samples at that observation**.

Seeds 0–11 for training and `default_rng(9000 + seed)` for evaluation.

## License

MIT
