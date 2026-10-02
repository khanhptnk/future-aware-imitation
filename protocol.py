"""The experimental protocol shared by reproduce.py, distill.py and distill_stochastic.py.

  budget:     every method simulates the same number of episodes per training run, BUDGET, counting roll-ins and
              roll-outs alike (each method's `iters` is set from its episodes per iteration).
  tuning:     each method's hyperparameter grid is searched separately in every setting, on TUNE_SEEDS, and the setting
              with the best mean value of the selection metric is kept.
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


@dataclass
class Method:
    train: Callable  # train(setting, rng, **kwargs) -> (policy, history)
    episodes_per_iter: int  # simulated episodes per iteration (roll-ins + roll-outs)
    grid: dict = field(default_factory=dict)  # hyperparameter -> list of values to try
    fixed: dict = field(default_factory=dict)

    def kwargs(self, **hp):
        return {"iters": BUDGET // self.episodes_per_iter, **self.fixed, **hp}


def mean_sd(rows, metric):
    v = [r[metric] for r in rows]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def tune_and_report(setting, methods: dict, evaluate: Callable, select: dict, label: str, shown: list):
    """For each method: if its grid has more than one point, try every point on TUNE_SEEDS and keep the one with the best
    mean of select[name] = (metric, +1 to maximize or -1 to minimize); then train the kept point on REPORT_SEEDS.
    Returns (rows, tuning) for JSON output."""
    rows, tuning = [], {}
    for name, m in methods.items():
        t0 = time.time()
        metric, sign = select[name]
        keys = list(m.grid)
        points = [dict(zip(keys, v)) for v in itertools.product(*(m.grid[k] for k in keys))]
        trials = []
        if len(points) > 1:
            for hp in points:
                scores = [evaluate(setting, m.train(setting, np.random.default_rng(s), **m.kwargs(**hp))[0], s)[metric]
                          for s in TUNE_SEEDS]
                trials.append({"hp": hp, "score": float(np.mean(scores))})
            best = max(trials, key=lambda tr: sign * tr["score"] if np.isfinite(tr["score"]) else -np.inf)["hp"]  # diverged runs lose
        else:
            best = points[0]
        tuning[name] = {"metric": metric, "trials": trials, "best": best, "iters": m.kwargs()["iters"]}
        mine = []
        for s in REPORT_SEEDS:
            policy, history = m.train(setting, np.random.default_rng(s), **m.kwargs(**best))
            mine.append({"method": name, "seed": s, "hp": best, "policy": np.asarray(policy).tolist(),
                         "history": [float(h) for h in history], "episodes_per_iter": m.episodes_per_iter,
                         **evaluate(setting, policy, s)})
        rows += mine
        print(f"{label} {name:20s} {str(best):16s} " + "  ".join(
            f"{k} {mean_sd(mine, k)[0]:.3f}±{mean_sd(mine, k)[1]:.3f}" for k in shown) + f"  ({time.time() - t0:.0f}s)",
            flush=True)
    return rows, tuning


def summarize(rows, metrics):
    """{method: {metric: (mean, sd)}} over the rows of one setting."""
    out = {}
    for name in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == name]
        out[name] = {k: mean_sd(mine, k) for k in metrics}
    return out
