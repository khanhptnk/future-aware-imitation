"""Distillation from a stochastic teacher: which distillation objectives leave the teacher's path when the student can't
follow it.

Same states and student as distill.py, but the teacher is a distribution: in every state it puts probability 0.9 on its
preferred action (the deterministic teacher's action in distill.py) and 0.1 on the other. Its preferred root action
leads into branch 0, whose parity pattern a degree-k student (k < 7) can't represent.

Six ways to train the student, all with the budget of protocol.py:
  off-policy KD    forward KL (cross-entropy to the teacher's distribution) on episodes the teacher generates
                   (word-level KD on teacher samples); fit exactly.
  forward KL       the same on episodes the student generates, i.e. DAgger with soft labels (GKD's forward KL,
                   "on-policy SFT"); aggregated over iterations and fit exactly, as in distill.py.
  JSD              GKD's generalized Jensen-Shannon divergence (beta = 0.5) on the student's episodes; fit by Adam.
  reverse KL (γ=0) per-token reward log pi_T(a|s) - log pi_S(a|s), each token credited only with its own reward
                   (on-policy distillation with discount zero).
  reverse KL       the same reward with undiscounted returns, i.e. the sequence-level reverse KL KL(P_S || P_T).
  APPO             +1 / -1 for agreeing with the teacher's preferred action, with returns.
The three RL objectives share the clipped update of distill.py; their learning rates are tuned (protocol.py) over a grid
from 0.0005 to 16, APPO's by its own objective (errors against the preferred action) and the reverse-KL ones by the
sequence-level reverse KL. Reverse KL with discount zero scores best when stopped partway, before its root probability
reaches the teacher's, so its tuned step size is small; trained to convergence it follows the teacher.

Evaluation is exact: an episode has 2 * 2^8 = 512 action sequences, so every metric is a sum over all of them.

Run: uv run distill_stochastic.py        (CPU, ~15 min; writes results/distill_stochastic.json)
"""

import functools
import itertools
import json
from pathlib import Path

import numpy as np

from distill import DEGREES, N_STATES, TEACHER, Student, clipped_update, fit_logistic, rollout
from protocol import EPISODES, Method, summarize, tune_and_report
from reproduce import H, LR_GRID, returns_to_go, sigmoid

# smaller steps too: reverse KL with discount zero does best when stopped partway (see the post)
LR_GRID_DISTILL = [0.0005, 0.001, 0.002, 0.004] + LR_GRID

P_PREFERRED = 0.9
TEACHER_P1 = np.where(TEACHER == 1, P_PREFERRED, 1 - P_PREFERRED)  # teacher's P(action 1 | state)

# all 512 episodes: root action, then 8 actions in the chosen branch
_SEQS = np.array([(a0,) + rest for a0 in (0, 1) for rest in itertools.product((0, 1), repeat=H)])
_STATES = np.column_stack([np.zeros(len(_SEQS), int), 1 + _SEQS[:, :1] * H + np.arange(H)[None, :]])


def evaluate(student, p1: np.ndarray, seed=None) -> dict:
    def logprob(q1):
        q = q1[_STATES]
        return np.log(np.where(_SEQS == 1, q, 1 - q)).sum(1)

    log_s, log_t = logprob(np.asarray(p1)), logprob(TEACHER_P1)
    P_s, P_t = np.exp(log_s), np.exp(log_t)
    errors = (_SEQS != TEACHER[_STATES]).sum(1)
    return {
        "p_deviate": float(p1[0]),
        "errors": float(P_s @ errors),
        "success": float(P_s @ (errors <= 2)),
        "reverse_kl": float(P_s @ (log_s - log_t)),
        "forward_kl": float(P_t @ (log_t - log_s)),
    }


def off_policy_kd(student, rng, iters=120, episodes=EPISODES, pseudocount=1e-3):
    """Cross-entropy to the teacher's distribution on the states of teacher-generated episodes."""
    n1, n = np.full(N_STATES, pseudocount), np.full(N_STATES, 2 * pseudocount)
    for _ in range(iters):
        states, _, _, _ = rollout(TEACHER_P1, episodes, rng)
        np.add.at(n1, states.ravel(), TEACHER_P1[states].ravel())
        np.add.at(n, states.ravel(), 1)
    return student.probs(fit_logistic(student, n1, n, np.zeros(student.n_params))), []


def forward_kl(student, rng, iters=120, episodes=EPISODES, pseudocount=1e-3):
    """Cross-entropy to the teacher's distribution on the states of student-generated episodes, aggregated."""
    n1, n = np.full(N_STATES, pseudocount), np.full(N_STATES, 2 * pseudocount)
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, _, _, _ = rollout(student.probs(theta), episodes, rng)
        np.add.at(n1, states.ravel(), TEACHER_P1[states].ravel())  # soft labels
        np.add.at(n, states.ravel(), 1)
        theta = fit_logistic(student, n1, n, theta)
    return student.probs(theta), []


def jsd(student, rng, iters=120, episodes=EPISODES, beta=0.5, steps=60, lr=0.1):
    """GKD's generalized JSD, beta KL(P || M) + (1 - beta) KL(Q || M) with M = beta P + (1 - beta) Q, between the
    teacher P and the student Q at the states of student-generated episodes (aggregated). For a binary action its
    derivative with respect to the student's logit z is (1 - beta) (z - logit(m)) q (1 - q), with m = M(action 1).
    Each iteration takes `steps` Adam steps from the previous parameters."""
    n = np.zeros(N_STATES)
    theta, m1, v1, k = np.zeros(student.n_params), np.zeros(student.n_params), np.zeros(student.n_params), 0
    for _ in range(iters):
        states, _, _, _ = rollout(student.probs(theta), episodes, rng)
        np.add.at(n, states.ravel(), 1)
        w = n / n.sum()
        for _ in range(steps):
            z = student.phi @ theta
            q = sigmoid(z)
            m = beta * TEACHER_P1 + (1 - beta) * q
            g = student.phi.T @ (w * (1 - beta) * (z - np.log(m / (1 - m))) * q * (1 - q))
            k += 1
            m1, v1 = 0.9 * m1 + 0.1 * g, 0.999 * v1 + 0.001 * g * g
            theta = theta - lr * (m1 / (1 - 0.9 ** k)) / (np.sqrt(v1 / (1 - 0.999 ** k)) + 1e-8)
    return student.probs(theta), []


def rl(student, rng, reward, use_returns, iters=120, episodes=EPISODES, lr=0.065, eps=0.2, epochs=4):
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
    return student.probs(theta), []


METHODS = {
    "off-policy KD": Method(off_policy_kd, EPISODES),
    "forward KL": Method(forward_kl, EPISODES),
    "JSD": Method(jsd, EPISODES),
    "reverse KL (γ=0)": Method(functools.partial(rl, reward="reverse_kl", use_returns=False), EPISODES,
                               grid={"lr": LR_GRID_DISTILL}),
    "reverse KL": Method(functools.partial(rl, reward="reverse_kl", use_returns=True), EPISODES, grid={"lr": LR_GRID_DISTILL}),
    "APPO": Method(functools.partial(rl, reward="agreement", use_returns=True), EPISODES, grid={"lr": LR_GRID_DISTILL}),
}
SELECT = {name: ("reverse_kl", -1) for name in METHODS} | {"APPO": ("errors", -1)}
METRICS = ["p_deviate", "errors", "success", "reverse_kl", "forward_kl"]


def main():
    out = Path(__file__).parent / "results"
    runs, tuning, summary = [], {}, {}
    print("teacher itself:", {m: round(v, 3) for m, v in evaluate(None, TEACHER_P1).items()})
    for k in DEGREES:
        rows, tuning[k] = tune_and_report(Student(k), METHODS, evaluate, SELECT, f"k={k}", METRICS)
        runs += [{"degree": k, **r} for r in rows]
        summary[str(k)] = summarize(rows, METRICS)
    (out / "distill_stochastic.json").write_text(json.dumps({"runs": runs, "summary": summary, "tuning": tuning}))


if __name__ == "__main__":
    main()
