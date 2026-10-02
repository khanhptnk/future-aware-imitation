"""The experimental protocol shared by reproduce.py, distill.py and distill_stochastic.py.

  budget:     every method simulates the same number of episodes per training run, BUDGET, counting roll-ins and
              roll-outs alike (each method's `iters` is set from its episodes per iteration).
  tuning:     each method's hyperparameter grid is searched separately in every setting, on TUNE_SEEDS, and the setting
              with the best mean value of the selection metric is kept.
  final policy: the checkpoint the method converges to. Every training function records, for each iteration (round),
              the policy it played and that policy's own training loss on the round's data, with a standard error. At CHECKPOINTS checkpoints spaced by budget
              (every 18,432 episodes, whatever the method's iteration size), a run has converged once its training loss
              has not improved on its best by more than one standard error for PATIENCE checkpoints in a row, and the
              policy at that moment is reported. (Not the best-loss checkpoint: each round's loss is measured on the
              learner's own states, so picking the best round would select by performance.) A run that never
              converges reports its last checkpoint.
  over training: the same metrics for the policy played in every round, averaged over all rounds (each round uses the
              same number of episodes, so this is the average over the budget). Minus the best achievable value, it
              is the average regret in the online-learning sense, in expectation over each round's episodes.
  reporting:  the kept setting is retrained on REPORT_SEEDS; mean and standard deviation over those seeds are reported.
Evaluation seeds are 9000 + training seed, so tuning and reporting never share evaluation episodes either.
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

EPISODES = 3072  # roll-in episodes per iteration, for every method
BUDGET = 120 * EPISODES  # 368,640 simulated episodes per training run
TUNE_SEEDS = range(100, 105)
REPORT_SEEDS = range(12)
CHECKPOINTS = 20  # checkpoints per training run, at equal shares of the budget
PATIENCE = 5  # checkpoints without improvement (a quarter of the budget) that count as converged


@dataclass
class Method:
    train: Callable  # train(setting, rng, **kwargs) -> (final policy, [(policy, training loss, its standard error)]
    #                  after every iteration)
    episodes_per_iter: int  # simulated episodes per iteration (roll-ins + roll-outs)
    grid: dict = field(default_factory=dict)  # hyperparameter -> list of values to try
    fixed: dict = field(default_factory=dict)

    def kwargs(self, **hp):
        return {"iters": BUDGET // self.episodes_per_iter, **self.fixed, **hp}


def mean_sd(rows, metric):
    v = [r[metric] for r in rows]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def safe_evaluate(evaluate: Callable, setting, policy, seed) -> dict:
    """evaluate(), except that a diverged policy (any non-finite probability) scores NaN on every metric, so it loses
    tuning instead of being scored as if its NaN probabilities were actions."""
    out = evaluate(setting, policy, seed)
    return out if np.all(np.isfinite(policy)) else {k: float("nan") for k in out}


def checkpoints(history) -> list:
    step = len(history) // CHECKPOINTS
    assert step * CHECKPOINTS == len(history), (len(history), CHECKPOINTS)
    return list(range(step - 1, len(history), step))


def converged_checkpoint(history) -> tuple:
    """(index into history at the moment of convergence, converged?) by the patience rule on the training loss
    (non-finite losses are skipped)."""
    best, since = None, 0
    for i in checkpoints(history):
        policy, loss, se = history[i]
        if not (np.isfinite(loss) and np.all(np.isfinite(policy))):
            continue
        if best is None or loss < history[best][1] - se:
            best, since = i, 0
        else:
            since += 1
            if since >= PATIENCE:
                return i, True
    return checkpoints(history)[-1], False


def train_and_converge(m: Method, setting, seed: int, **hp):
    """Train once; return (the converged checkpoint's policy, its round, converged?, the history)."""
    _, history = m.train(setting, np.random.default_rng(seed), **m.kwargs(**hp))
    i, ok = converged_checkpoint(history)
    return history[i][0], i, ok, history


def tune_and_report(setting, methods: dict, evaluate: Callable, select: dict, label: str, shown: list,
                    curve: Callable = None):
    """For each method: if its grid has more than one point, try every point on TUNE_SEEDS and keep the one with the best
    mean of select[name] = (metric, +1 to maximize or -1 to minimize); then train the kept point on REPORT_SEEDS. Every
    run reports its converged checkpoint (train_and_converge). On REPORT_SEEDS, curve(setting, policy, seed) -> dict of
    metrics (default: evaluate) is averaged over the policies of all rounds and stored as over_training_<metric>.
    Returns (rows, tuning) for JSON output."""
    curve = curve or evaluate
    rows, tuning = [], {}
    for name, m in methods.items():
        t0 = time.time()
        metric, sign = select[name]
        keys = list(m.grid)
        points = [dict(zip(keys, v)) for v in itertools.product(*(m.grid[k] for k in keys))]
        trials = []
        if len(points) > 1:
            for hp in points:
                scores = [safe_evaluate(evaluate, setting, train_and_converge(m, setting, s, **hp)[0], s)[metric]
                          for s in TUNE_SEEDS]
                trials.append({"hp": hp, "score": float(np.mean(scores))})
            best = max(trials, key=lambda tr: sign * tr["score"] if np.isfinite(tr["score"]) else -np.inf)["hp"]  # diverged runs lose
        else:
            best = points[0]
        tuning[name] = {"metric": metric, "trials": trials, "best": best, "iters": m.kwargs()["iters"]}
        mine = []
        for s in REPORT_SEEDS:
            policy, chosen, ok, history = train_and_converge(m, setting, s, **best)
            ck = [safe_evaluate(curve, setting, h[0], s) for h in history]
            mine.append({"method": name, "seed": s, "hp": best, "policy": np.asarray(policy).tolist(),
                         "chosen_iter": chosen, "converged": ok, "last_policy": np.asarray(history[-1][0]).tolist(),
                         "history": [float(h[0][0]) for h in history], "loss": [float(h[1]) for h in history],
                         "episodes_per_iter": m.episodes_per_iter,
                         **safe_evaluate(evaluate, setting, policy, s),
                         **{"over_training_" + k: float(np.mean([c[k] for c in ck])) for k in ck[0]}})
        rows += mine
        print(f"{label} {name:20s} {str(best):16s} " + "  ".join(
            f"{k} {mean_sd(mine, k)[0]:.3f}±{mean_sd(mine, k)[1]:.3f}" for k in shown)
            + f"  converged {sum(r['converged'] for r in mine)}/{len(mine)} at iter {[r['chosen_iter'] for r in mine]}"
            + f"  over training: " + " ".join(f"{k} {np.mean([r['over_training_' + k] for r in mine]):.3f}"
                                              for k in shown if "over_training_" + k in mine[0])
            + f"  ({time.time() - t0:.0f}s)",
            flush=True)
    return rows, tuning


def summarize(rows, metrics):
    """{method: {metric: (mean, sd)}} over the rows of one setting."""
    out = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == name]
        out[name] = {k: mean_sd(mine, k) for k in metrics}
    return out
