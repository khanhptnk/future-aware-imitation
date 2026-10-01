"""Distillation into a smaller student: DAgger vs PPO vs GRPO when the student can't represent the teacher.

The student sees everything the teacher sees. The only gap is model size. Each episode has a root decision and then
H = 8 steps in one of two branches, chosen by the root action. The state is (branch, t), fully observed.

  teacher: plays 0 at the root, so its own path is branch 0. In branch 0 it plays the parity of t (1, 0, 1, 0, ...); in
           branch 1 it plays 0. Its policy is a degree-7 polynomial in t per branch (enough to fit parity on 8 points).
  student: a separate root logit, plus a degree-k polynomial in t per branch: P(1 | branch, t) = sigmoid(w_branch . P(t)),
           with Legendre features P_0..P_k of t rescaled to [-1, 1]. A degree-k polynomial changes sign at most k times,
           so in branch 0 it makes at least ceil((7 - k) / 2) errors; branch 1 it fits for any k.

Copying the teacher at the root is free now but costs those errors later; deviating (root action 1) costs one error now
and none later. DAgger's root label is always 0, so it always copies.

Run: uv run distill.py        (CPU, a few minutes; writes results/distill.json)
"""

import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

from reproduce import H, T, returns_to_go, sigmoid

DEGREES = range(8)
SEEDS = range(12)
EVAL_EPISODES = 50_000
N_STATES = 1 + 2 * H  # 0: root; 1..8: branch 0 at t = 1..8; 9..16: branch 1
TEACHER = np.array([0] + [t % 2 for t in range(1, H + 1)] + [0] * H)  # teacher's action in each state


class Student:
    """Logits = Phi @ theta: theta = [root logit, w_branch0 (k + 1), w_branch1 (k + 1)]."""

    def __init__(self, k: int):
        u = (np.arange(1, H + 1) - (H + 1) / 2) / ((H - 1) / 2)  # t rescaled to [-1, 1]
        P = legendre.legvander(u, k)  # (H, k + 1)
        self.phi = np.zeros((N_STATES, 1 + 2 * (k + 1)))
        self.phi[0, 0] = 1
        self.phi[1:1 + H, 1:k + 2] = P
        self.phi[1 + H:, k + 2:] = P
        self.n_params = self.phi.shape[1]

    def probs(self, theta):
        return sigmoid(self.phi @ theta)


def rollout(p1: np.ndarray, n: int, rng: np.random.Generator):
    """Episodes under a student with P(action 1 | state) = p1[state]. Returns states, actions, teacher actions, rewards."""
    a0 = (rng.random(n) < p1[0]).astype(int)
    down = 1 + a0[:, None] * H + np.arange(H)[None, :]  # (n, H) state ids along the chosen branch
    states = np.column_stack([np.zeros(n, int), down])
    act = (rng.random((n, T)) < p1[states]).astype(int)
    act[:, 0] = a0
    teacher = TEACHER[states]
    return states, act, teacher, np.where(act == teacher, 1.0, -1.0)


def evaluate(p1, seed):
    rng = np.random.default_rng(9000 + seed)
    _, _, _, rew = rollout(p1, EVAL_EPISODES, rng)
    errors = (rew < 0).sum(1)
    return {"p_deviate": float(p1[0]), "signed_return": float(rew.sum(1).mean()), "errors": float(errors.mean()),
            "success": float((errors <= 2).mean())}


def fit_logistic(student, n1, n, theta, iters=50, ridge=1e-6):
    """Maximum likelihood for label counts (n1 ones out of n) per state, by damped Newton steps."""
    def loglik(th):
        z = student.phi @ th
        return (n1 * -np.logaddexp(0, -z) + (n - n1) * -np.logaddexp(0, z)).sum() - 0.5 * ridge * th @ th

    for _ in range(iters):
        p = student.probs(theta)
        grad = student.phi.T @ (n1 - n * p) - ridge * theta
        hess = student.phi.T @ (student.phi * (n * p * (1 - p))[:, None]) + ridge * np.eye(len(theta))
        step = np.linalg.solve(hess, grad)
        f0, s = loglik(theta), 1.0
        while loglik(theta + s * step) < f0 and s > 1e-6:
            s /= 2
        theta = theta + s * step
        if np.abs(s * step).max() < 1e-8:
            break
    return theta


def dagger(student, rng, iters=40, episodes=3000, pseudocount=1e-3):
    n1, n = np.full(N_STATES, pseudocount), np.full(N_STATES, 2 * pseudocount)
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, _, teacher, _ = rollout(student.probs(theta), episodes, rng)
        np.add.at(n1, states.ravel(), teacher.ravel())
        np.add.at(n, states.ravel(), 1)
        theta = fit_logistic(student, n1, n, theta)
    return student.probs(theta)


def clipped_update(student, theta, states, act, adv, lr, eps, epochs):
    """As in reproduce.py: each state's logit gets the mean clipped-surrogate gradient over its samples, then the chain
    rule through the features (with one-hot features this is exactly the tabular update)."""
    s, a, adv = states.ravel(), act.ravel(), adv.ravel()
    count = np.maximum(np.bincount(s, minlength=N_STATES), 1)
    p_old = student.probs(theta)[s]
    pi_old = np.where(a == 1, p_old, 1 - p_old)
    for _ in range(epochs):
        p = student.probs(theta)[s]
        ratio = np.where(a == 1, p, 1 - p) / pi_old
        active = ~(((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps)))
        g_state = np.bincount(s, weights=active * adv * ratio * (a - p), minlength=N_STATES) / count
        theta = theta + lr * student.phi.T @ g_state
    return theta


def ppo(student, rng, iters=120, episodes=3072, lr=0.065, eps=0.2, epochs=4):
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, act, _, rew = rollout(student.probs(theta), episodes, rng)
        G = returns_to_go(rew)
        n = np.bincount(states.ravel(), minlength=N_STATES)
        adv = G - (np.bincount(states.ravel(), weights=G.ravel(), minlength=N_STATES) / np.maximum(n, 1))[states]
        adv = adv / (adv.std() + 1e-8)
        theta = clipped_update(student, theta, states, act, adv, lr, eps, epochs)
    return student.probs(theta)


def grpo(student, rng, iters=120, groups=192, group_size=8, lr=0.065, eps=0.2, epochs=4):
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, act, _, rew = rollout(student.probs(theta), groups * group_size, rng)
        R = rew.sum(1).reshape(groups, group_size)
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)
        theta = clipped_update(student, theta, states, act, np.repeat(A.reshape(-1, 1), T, 1), lr, eps, epochs)
    return student.probs(theta)


METHODS = {"DAgger": dagger, "PPO": ppo, "GRPO": grpo}
METRICS = ["p_deviate", "signed_return", "errors", "success"]


def main():
    out = Path(__file__).parent / "results"
    runs, summary = [], {}
    for k in DEGREES:
        student = Student(k)
        for method, train in METHODS.items():
            t0 = time.time()
            rows = []
            for seed in SEEDS:
                p1 = train(student, np.random.default_rng(seed))
                rows.append({"degree": k, "method": method, "seed": seed, "policy": p1.tolist(), **evaluate(p1, seed)})
            runs += rows
            stats = {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
                     for m in METRICS}
            summary.setdefault(str(k), {})[method] = stats
            print(f"k={k} {method:6s} " + "  ".join(f"{m} {mu:.3f} ± {sd:.3f}" for m, (mu, sd) in stats.items())
                  + f"  ({time.time() - t0:.1f}s)", flush=True)
    (out / "distill.json").write_text(json.dumps({"runs": runs, "summary": summary}))


if __name__ == "__main__":
    main()
