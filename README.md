# When immediate imitation is not enough

Code for the post [When immediate imitation is not enough](https://machineslearner.com/machine-learning/future-aware-imitation):
toy environments for three reasons imitation can be impossible (privileged information, a stochastic expert, and
limited capacity), and how DAgger, AggreVaTe, LOLS and **Agreement PPO (APPO)** handle them.

APPO is PPO on the learner's own roll-outs with a reward of +1 when its action matches the expert's and −1 otherwise,
credited with undiscounted returns, with no entropy bonus or KL term. AGRPO is the same reward with a minimal GRPO-style
update; it behaves like APPO and isn't shown in the post. Everything is NumPy and
runs on a CPU.

The idea in one line: when the learner can't imitate the expert, the projection of the expert onto the learner's
policy class (the closest policy by the method's own measure) and the best policy in that class are in general not the
same policy. DAgger and AggreVaTe find the projection; LOLS and APPO find the best policy, because they score actions by
the learner's own future.

```sh
uv run reproduce.py            # ~10 min: cases 1 and 2, five methods, tuned; results/runs.json, results/tuning.json
uv run distill.py              # ~20 min: case 3, deterministic teacher, student degree 0-7; results/distill.json
uv run distill_stochastic.py   # ~30 min: case 3, stochastic teacher, six distillation objectives; results/distill_stochastic.json
uv run entropy_ablation.py     # ~10 min: APPO with an entropy bonus, swept; results/entropy.json
uv run plots.py                # figures/*.svg (light and dark versions) from results/*.json
uv run reproduce.py --note     # the original note's settings, which reproduce its tables
```

`results/` holds the JSON from the runs behind the post (every tuning trial included), so `plots.py` works without
re-running anything. `uv run checks.py` (~1 min) runs numerical checks of the implementations: the
updates against finite-difference gradients of their objectives, the value estimates against simulation and exact values,
the exact evaluations against Monte Carlo, every method's episode count against the budget, and the tuning picks.

## Protocol (`protocol.py`)

- **Equal budget:** every method simulates 368,640 episodes per training run, counting roll-ins and roll-outs alike
  (DAgger and APPO 120 iterations × 3,072 episodes; AggreVaTe 60 iterations, one expert roll-out per roll-in; LOLS 40
  iterations, two roll-outs per roll-in; AGRPO 240 iterations × 1,536).
- **Tuning:** each knob is searched separately in every setting on seeds 100–104, and the chosen value is retrained and
  reported on seeds 0–11. Learning rates for APPO and AGRPO: twelve values from 0.0075 to 16; for the RL distillation
  objectives, sixteen values from 0.0005 to 16. LOLS: β ∈ {0, 0.5}. DAgger and AggreVaTe fit their data exactly and have
  nothing to tune. Selection: the imitation objective J (expected agreement) for the ±1 methods, and the sequence-level
  reverse KL for the reverse-KL distillation objectives.
- **Evaluation:** 50,000 fresh episodes from `default_rng(9000 + seed)`, both sampling actions and playing each
  policy's more likely action ("greedy"); exact over all 512 action sequences with the stochastic teacher.

## The methods

All of them roll out the learner and query the expert at the states it visits; they differ in what credits an action.

| | Credit for an action | Who plays the future | Policy update |
|---|---|---|---|
| DAgger | matching the expert now | nobody | supervised fit |
| AggreVaTe | return from that step on | the expert | cost-sensitive classifier on aggregated values |
| LOLS | return from that step on, every action tried at the same state (two here) | the learner (β = 0) | cost-sensitive classifier on aggregated values |
| APPO, AGRPO | return from that step on | the learner, in the same episode | clipped policy gradient |

## The environments

Every environment has 9 binary decisions: a root decision, then 8 steps whose situation depends on the root action. Each
step earns +1 if the learner's action matches the expert's and −1 if not. The root has its own parameter.

- **Case 1, privileged information** (`reveal`): a hidden bit z is the expert's action at every step. Root action 1
  makes the later observations reveal z; root action 0 keeps it hidden.
- **Case 2, stochastic expert** (`hard`): at the root the expert flips a fair coin z and plays it. Matching it enters an
  easy corridor, the mistake "1 when z = 0" a recoverable corridor, the mistake "0 when z = 1" a hard corridor where the
  expert flips a fresh coin at every step.
- **Case 3, limited capacity** (`distill.py`): nothing is hidden. The teacher plays 0 at the root, then the parity of t
  in branch 0 (which needs a degree-7 polynomial) and 0 in branch 1. The student is a degree-k polynomial in t per branch,
  plus a root logit.

## Results (12 seeds; errors per episode, mean)

| | DAgger | AggreVaTe | LOLS | APPO | AGRPO |
|---|---|---|---|---|---|
| Case 1 (best possible: 0.5) | 2.50 | 2.50 (per-seed coin flip) | **0.50** | **0.50** | 0.52 |
| Case 2 (best possible: 0.5) | 1.50 | 1.16 (per-seed coin flip) | **0.50** | **0.50** | 0.53 |
| Case 3, degree-1 student (best possible: 1) | 3.82 | 4.00 | **1.00** | **1.00** | **1.00** |
| Case 3, degree-7 student (best possible: 0) | **0.00** | **0.00** | **0.00** | 0.06 (0 greedy) | 0.03 |

DAgger and AggreVaTe never prefer the action that makes the rest of the episode imitable: DAgger's root target is the
expert's action averaged over what the learner can't see, and AggreVaTe's estimated value of each root action assumes the
expert plays the rest. LOLS and APPO score actions by the learner's own future and are optimal in all three cases.
Exact values, standard deviations and the stochastic-teacher comparison are in the post and in `results/`.

**Stochastic teacher** (case 3, `distill_stochastic.py`: probability 0.9 on the teacher's action above in every state;
degree-1 student; exact evaluation):

| Objective | P(leave the teacher's path) | Errors | KL(P_S ‖ P_T) | KL(P_T ‖ P_S) |
|---|---|---|---|---|
| teacher | 0.10 | 0.90 | 0 | 0 |
| off-policy KD | 0.100 | 3.64 | 3.49 | 2.54 |
| on-policy forward KL | 0.100 | 3.64 | 3.49 | 2.54 |
| on-policy JSD (β = 0.5) | 0.100 | 3.61 | 3.48 | 2.55 |
| reverse KL, discount 0 (tuning stops it early) | 0.376 | 3.25 | 2.74 | 2.75 |
| reverse KL with returns | 0.840 | 2.12 | **2.13** | 3.90 |
| APPO | 1.000 | **1.00** | 3.15 | 50 |

The per-token objectives follow the teacher at the root; reverse KL with discount zero does too once trained to
convergence (reverse KL ≈ 3.48), but scores best when stopped partway, so its tuned step size is small. The objectives with
returns leave the teacher's path. At k = 0, reverse KL with returns reaches the exact optimum of the sequence-level reverse
KL (leave 0.869 at 2.162 nats; training: 0.866 and 2.162).

## Entropy bonus (`entropy_ablation.py`)

APPO has no entropy bonus. Adding one, swept from 0.003 to 1 at APPO's tuned learning rate, changes no decision up to 0.1
(errors rise by at most 0.02); from 0.3 on the policy gets noisier, while its root decisions mostly hold. With the stochastic teacher it trades
errors for KL but stays far from the distribution-matching objectives. With the stochastic teacher at degree 7, for example,
a coefficient of 0.3 lowers APPO's forward KL from 5.26 to 3.45 nats and raises its errors from 0.06 to 0.14, while
reverse KL with returns reaches 0.02. With a degree-1 student the bonus has no effect: the logits saturate within a few
iterations at the tuned learning rate, where the entropy gradient vanishes.

## License

MIT
