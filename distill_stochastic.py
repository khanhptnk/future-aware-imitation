"""Distillation from a stochastic teacher: which on-policy objectives leave the teacher's path when the student can't follow it.

Same states and student as distill.py, but the teacher is a distribution: in every state it puts probability 0.9 on its
preferred action (the deterministic teacher's action in distill.py) and 0.1 on the other. Its preferred root action
leads into branch 0, whose parity pattern a degree-k student (k < 7) can't represent.

Four ways to train the student on its own roll-outs, with the teacher queried at every visited state:
  forward KL       on-policy cross-entropy to the teacher's distribution, i.e. DAgger with soft labels (GKD-style
                   forward KL, "on-policy SFT"); aggregated over iterations and fit exactly, as in distill.py.
  reverse KL (γ=0) per-token reward log pi_T(a|s) - log pi_S(a|s), each token credited only with its own reward
                   (on-policy distillation with discount zero).
  reverse KL       the same reward with undiscounted returns, i.e. the sequence-level reverse KL KL(P_S || P_T).
  ±1 agreement     PPO with +1 / -1 for agreeing with the teacher's preferred action, with returns.
The three RL variants share the clipped update of reproduce.py / distill.py.

Evaluation is exact: an episode has 2 * 2^8 = 512 action sequences, so every metric is a sum over all of them.

Run: uv run distill_stochastic.py        (CPU, a few minutes; writes results/distill_stochastic.json)
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np

from distill import DEGREES, N_STATES, SEEDS, TEACHER, Student, clipped_update, fit_logistic, rollout
from reproduce import H, returns_to_go

P_PREFERRED = 0.9
TEACHER_P1 = np.where(TEACHER == 1, P_PREFERRED, 1 - P_PREFERRED)  # teacher's P(action 1 | state)

# all 512 episodes: root action, then 8 actions in the chosen branch
_SEQS = np.array([(a0,) + rest for a0 in (0, 1) for rest in itertools.product((0, 1), repeat=H)])
_STATES = np.column_stack([np.zeros(len(_SEQS), int), 1 + _SEQS[:, :1] * H + np.arange(H)[None, :]])


def evaluate(p1: np.ndarray) -> dict:
    def logprob(q1):
        q = q1[_STATES]
        return np.log(np.where(_SEQS == 1, q, 1 - q)).sum(1)

    log_s, log_t = logprob(p1), logprob(TEACHER_P1)
    P_s, P_t = np.exp(log_s), np.exp(log_t)
    errors = (_SEQS != TEACHER[_STATES]).sum(1)
    return {
        "p_deviate": float(p1[0]),
        "errors": float(P_s @ errors),
        "success": float(P_s @ (errors <= 2)),
        "reverse_kl": float(P_s @ (log_s - log_t)),
        "forward_kl": float(P_t @ (log_t - log_s)),
    }


def forward_kl(student, rng, iters=40, episodes=3000, pseudocount=1e-3):
    n1, n = np.full(N_STATES, pseudocount), np.full(N_STATES, 2 * pseudocount)
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, _, _, _ = rollout(student.probs(theta), episodes, rng)
        np.add.at(n1, states.ravel(), TEACHER_P1[states].ravel())  # soft labels
        np.add.at(n, states.ravel(), 1)
        theta = fit_logistic(student, n1, n, theta)
    return student.probs(theta)


def rl(student, rng, reward, use_returns, iters=120, episodes=3072, lr=0.065, eps=0.2, epochs=4):
    """PPO pipeline of distill.ppo with a choice of per-step reward and of returns vs. each step's own reward."""
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        p1 = student.probs(theta)
        states, act, teacher, agree = rollout(p1, episodes, rng)
        if reward == "agreement":
            r = agree
        else:  # per-token reverse-KL reward: log pi_T(a|s) - log pi_S(a|s), student log-prob held fixed
            pt, ps = TEACHER_P1[states], p1[states]
            r = np.log(np.where(act == 1, pt, 1 - pt)) - np.log(np.where(act == 1, ps, 1 - ps))
        G = returns_to_go(r) if use_returns else r
        n = np.bincount(states.ravel(), minlength=N_STATES)
        adv = G - (np.bincount(states.ravel(), weights=G.ravel(), minlength=N_STATES) / np.maximum(n, 1))[states]
        adv = adv / (adv.std() + 1e-8)
        theta = clipped_update(student, theta, states, act, adv, lr, eps, epochs)
    return student.probs(theta)


METHODS = {
    "forward KL": forward_kl,
    "reverse KL (γ=0)": lambda s, rng: rl(s, rng, "reverse_kl", use_returns=False),
    "reverse KL": lambda s, rng: rl(s, rng, "reverse_kl", use_returns=True),
    "±1 agreement": lambda s, rng: rl(s, rng, "agreement", use_returns=True),
}
METRICS = ["p_deviate", "errors", "success", "reverse_kl", "forward_kl"]


def main():
    out = Path(__file__).parent / "results"
    runs, summary = [], {}
    print("teacher itself:", {m: round(v, 3) for m, v in evaluate(TEACHER_P1).items()})
    for k in DEGREES:
        student = Student(k)
        for method, train in METHODS.items():
            t0 = time.time()
            rows = [{"degree": k, "method": method, "seed": seed, "policy": (p1 := train(student, np.random.default_rng(seed))).tolist(),
                     **evaluate(p1)} for seed in SEEDS]
            runs += rows
            stats = {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
                     for m in METRICS}
            summary.setdefault(str(k), {})[method] = stats
            print(f"k={k} {method:17s} " + "  ".join(f"{m} {mu:.3f} ± {sd:.3f}" for m, (mu, sd) in stats.items())
                  + f"  ({time.time() - t0:.1f}s)", flush=True)
    (out / "distill_stochastic.json").write_text(json.dumps({"runs": runs, "summary": summary}))


if __name__ == "__main__":
    main()
