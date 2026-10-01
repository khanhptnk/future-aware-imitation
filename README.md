# When immediate imitation is not enough

Code for the post [When immediate imitation is not enough](https://khanhptnk.github.io/machine-learning/future-aware-imitation):
toy environments for the three reasons imitation can be impossible (privileged information, a hard-to-imitate expert,
and limited capacity), comparing DAgger with **Agreement PPO**: PPO on the learner's own roll-outs with a reward of +1
when its action matches the expert's and −1 otherwise, credited with undiscounted returns. Agreement GRPO is the same
reward with a minimal GRPO-style update. Everything runs on a CPU in a few minutes with NumPy.

```sh
uv run reproduce.py            # ~1 min: 3 methods x 2 environments x 12 seeds; writes results/runs.json, summary.json
uv run distill.py              # ~2 min: deterministic teacher, student degree 0-7; writes results/distill.json
uv run distill_stochastic.py   # ~3 min: stochastic teacher, 4 distillation objectives; results/distill_stochastic.json
uv run plots.py                # figures/*.svg (light and dark versions) from results/*.json
```

`results/` holds the JSON from the run behind the post, so `plots.py` works without re-running anything.

## Cases 1 and 2: privileged information and a hard-to-imitate expert

Each episode samples a hidden bit `z ~ Bernoulli(0.5)` that the expert sees and the learner doesn't. The learner makes
9 binary decisions: one at a root observation that hides `z`, then 8 downstream. Each decision earns +1 if it matches the
expert's action and −1 if not. The policy is tabular (one Bernoulli logit per observation ID), so nothing learned
downstream can generalize back to the root.

- **Privileged information** (`reveal`). The expert plays `z` at every step. Root action 1 leads to downstream
  observations that reveal `z`; root action 0 leads to one that still hides it.
- **Hard-to-imitate expert** (`hard`). A correct root action enters an easy corridor, the root mistake "1 when z = 0" a
  recoverable corridor, and the mistake "0 when z = 1" a hard corridor. The expert plays action 0 throughout the easy and
  recoverable corridors, and a fresh coin flip at every step of the hard one.

In both, either root action matches the expert with probability 1/2, so DAgger's root target is exactly 1/2. Action 1
leads to a future the learner can imitate.

## Results (12 seeds, mean ± standard deviation; 50,000 evaluation episodes per seed)

| Privileged information | P(root action 1) | Signed return | Errors / episode | Success (≤ 2 errors) |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.002 | 4.00 ± 0.02 | 2.499 ± 0.009 | 54.5% ± 0.2% |
| Agreement PPO | 0.986 ± 0.000 | 7.19 ± 0.01 | 0.905 ± 0.004 | 96.2% ± 0.1% |
| Agreement GRPO | 0.967 ± 0.001 | 7.01 ± 0.01 | 0.995 ± 0.006 | 96.0% ± 0.1% |

| Hard-to-imitate expert | P(root action 1) | Signed return | Errors / episode | Success (≤ 2 errors) |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.001 | 6.00 ± 0.02 | 1.499 ± 0.012 | 75.9% ± 0.3% |
| Agreement PPO | 0.970 ± 0.001 | 7.18 ± 0.01 | 0.911 ± 0.004 | 96.1% ± 0.1% |
| Agreement GRPO | 0.902 ± 0.003 | 6.83 ± 0.02 | 1.085 ± 0.008 | 92.9% ± 0.2% |

DAgger's numbers have closed forms: 54.49% success (privileged information) and 75.88% (hard-to-imitate expert).

## Case 3: limited capacity (distillation into a smaller student)

`distill.py` and `distill_stochastic.py`. Nothing is hidden: the state is (branch, t). The teacher plays 0 at the root,
so its own path is branch 0, where it plays the parity of t; in branch 1 it plays 0. The student has a root logit and,
per branch, a logistic policy whose logit is a degree-k polynomial in t (Legendre features). Degree 7 fits the teacher
exactly; a smaller student makes at least ceil((7 − k) / 2) errors on the teacher's path, while leaving it at the root
costs one error.

**Deterministic teacher** (mean errors per episode, 12 seeds):

| Student degree k | 0 | 1–2 | 3–4 | 5–6 | 7 |
|---|---|---|---|---|---|
| DAgger | 4.00 | 3.82 | 3.37 | 2.46 | **0.00** |
| Agreement PPO | 1.01 | 1.01 | 1.01 | 1.01 | 0.95 |
| Agreement GRPO | 1.07 | 1.07 | 1.06 | 1.05 | 1.04 |

DAgger always follows the teacher (P(leave) = 0); Agreement PPO and GRPO leave its path in 98–99.8% of episodes below k = 7.

**Stochastic teacher** (0.9 on the action above in every state). Four objectives on the student's own roll-outs:
on-policy forward KL (DAgger with soft labels), per-token reverse KL with discount 0, reverse KL with returns
(sequence-level reverse KL), and Agreement PPO. Evaluation is exact (all 512 action sequences). At k = 1:

| Objective | P(leave) | Errors | Success | KL(P_S ‖ P_T) | KL(P_T ‖ P_S) |
|---|---|---|---|---|---|
| teacher | 0.10 | 0.90 | 94.7% | 0 | 0 |
| forward KL | 0.100 | 3.64 | 23.1% | 3.49 | 2.54 |
| reverse KL, γ = 0 | 0.103 | 3.61 | 23.7% | 3.47 | 2.55 |
| reverse KL, returns | 0.840 | 2.12 | 71.1% | 2.13 | 3.90 |
| Agreement PPO | 0.998 | 1.01 | 99.8% | 3.11 | 14.2 |

At k = 0 the exact optimum of the sequence-level reverse KL leaves with probability 0.869 at 2.162 nats; reverse KL
with returns reaches 0.866 and 2.162.

## Training details

- **DAgger:** 40 iterations × 3,000 learner roll-outs. Expert labels at every visited observation are aggregated over all
  iterations, and the policy is their label frequencies (pseudocount 1e-3).
- **Agreement PPO:** 120 iterations × 3,072 episodes; undiscounted reward-to-go; baseline = batch-mean return-to-go at the same
  observation; advantages divided by their global standard deviation; clipped surrogate with ε = 0.2; 4 full-batch
  gradient-ascent epochs with learning rate 0.065. No value network.
- **Agreement GRPO:** 120 iterations × 192 groups of 8 trajectories sharing `z`; each trajectory's total return is normalized
  within its group and assigned to all of its actions; same clipped update. No KL term or reference policy.
- In the clipped update, each observation's logit gets the **mean gradient over the samples at that observation**.

Seeds 0–11 for training and `default_rng(9000 + seed)` for evaluation.

## License

MIT
