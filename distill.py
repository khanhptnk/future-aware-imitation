"""Distillation into a smaller student: DAgger, AggreVaTe, LOLS, APPO and AGRPO when the student can't represent the
teacher.

The student sees everything the teacher sees. The only gap is model size. Each episode has a root decision and then
H = 8 steps in one of two branches, chosen by the root action. The state is (branch, t), fully observed.

  teacher: plays 0 at the root, so its own path is branch 0. In branch 0 it plays the parity of t (1, 0, 1, 0, ...); in
           branch 1 it plays 0. Its policy is a degree-7 polynomial in t per branch (enough to fit parity on 8 points).
  student: a separate root logit, plus a degree-k polynomial in t per branch: P(1 | branch, t) = sigmoid(w_branch . P(t)),
           with Legendre features P_0..P_k of t rescaled to [-1, 1]. A degree-k polynomial changes sign at most k times,
           so in branch 0 it makes at least ceil((7 - k) / 2) errors; branch 1 it fits for any k.

Copying the teacher at the root is free now but costs those errors later; deviating (root action 1) costs one error now
and none later. DAgger's root label is always 0, and AggreVaTe's teacher roll-outs value both root actions at the
teacher's 8 later agreements (plus +1 / -1 at the root), so both always copy. LOLS and APPO value the root actions by
the student's own later agreement, so they leave whenever the student can't fit branch 0.

Run: uv run distill.py        (CPU, ~20 min: tunes and reports at every student size; writes results/distill.json)
"""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

from protocol import EPISODES, Method, summarize, tune_and_report
from reproduce import H, LR_GRID, T, explore, returns_to_go, sigmoid

DEGREES = range(8)
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


def evaluate(student, p1, seed):
    """Metrics of the policy as trained (actions sampled) and played greedily, on the same 50,000 fresh episodes."""
    out = {"p_deviate": float(p1[0])}
    for prefix, policy in (("", p1), ("greedy_", (np.asarray(p1) > 0.5).astype(float))):
        rng = np.random.default_rng(9000 + seed)
        _, _, _, rew = rollout(policy, EVAL_EPISODES, rng)
        errors = (rew < 0).sum(1)
        out |= {prefix + "signed_return": float(rew.sum(1).mean()), prefix + "errors": float(errors.mean()),
                prefix + "success": float((errors <= 2).mean())}
    return out


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
    return student.probs(theta), []


def fit_cost_sensitive(student, q_sum, q_n, theta, pseudocount=1e-3):
    """The cost-sensitive classifier for the student's class. A binary cost-sensitive problem is a weighted
    classification problem (label = the action with the higher mean value, weight = samples x the value gap); fit it
    with the logistic loss and play it deterministically. Returns the new parameters and the policy."""
    mean = q_sum / np.maximum(q_n, 1)
    gap = np.where((q_n > 0).all(1), mean[:, 1] - mean[:, 0], 0.0)
    w = q_n.sum(1) * np.abs(gap)
    theta = fit_logistic(student, w * (gap > 0) + pseudocount, w + 2 * pseudocount, theta)
    return theta, (student.phi @ theta > 0).astype(float)


def aggrevate(student, rng, iters=40, episodes=3000):
    """AggreVaTe with learner roll-in (see reproduce.explore): one random action at one random step per episode, then
    the teacher finishes the episode."""
    q_sum, q_n = np.zeros((N_STATES, 2)), np.zeros((N_STATES, 2))
    theta, p1 = np.zeros(student.n_params), np.full(N_STATES, 0.5)
    for _ in range(iters):
        states, _, teacher, _ = rollout(p1, episodes, rng)
        s, a, q = explore(states, teacher, rng)
        np.add.at(q_sum, (s, a), q)
        np.add.at(q_n, (s, a), 1)
        theta, p1 = fit_cost_sensitive(student, q_sum, q_n, theta)
    return p1, []


def continue_from(a0, t, a, p1, teacher_rollout, rng):
    """Agreement from step t on, after taking action a at step t (the root action a0 of the roll-in decides the branch
    when t > 0) and then following the teacher (where teacher_rollout) or the student p1."""
    n = len(a0)
    states = np.column_stack([np.zeros(n, int), 1 + np.where(t == 0, a, a0)[:, None] * H + np.arange(H)[None, :]])
    teacher = TEACHER[states]
    act = np.where(teacher_rollout[:, None], teacher, (rng.random((n, T)) < p1[states]).astype(int))
    act[np.arange(n), t] = a
    return (np.where(act == teacher, 1.0, -1.0) * (np.arange(T)[None, :] >= t[:, None])).sum(1)


def lols(student, rng, beta=0.0, iters=40, episodes=3000):
    """LOLS (see reproduce.lols): both actions at one random step per episode, each followed by a roll-out with the
    teacher (probability beta) or the student."""
    q_sum, q_n = np.zeros((N_STATES, 2)), np.zeros((N_STATES, 2))
    theta, p1 = np.zeros(student.n_params), np.full(N_STATES, 0.5)
    for _ in range(iters):
        states, act, _, _ = rollout(p1, episodes, rng)
        t = rng.integers(0, T, episodes)
        s = states[np.arange(episodes), t]
        teacher_rollout = rng.random(episodes) < beta
        for a in (0, 1):
            q = continue_from(act[:, 0], t, np.full(episodes, a), p1, teacher_rollout, rng)
            np.add.at(q_sum[:, a], s, q)
            np.add.at(q_n[:, a], s, 1)
        theta, p1 = fit_cost_sensitive(student, q_sum, q_n, theta)
    return p1, []


def clipped_update(student, theta, states, act, adv, lr, eps, epochs, ent=0.0):
    """As in reproduce.py: each state's logit gets the mean clipped-surrogate gradient over its samples (plus, with
    ent > 0, the entropy bonus at every visited state), then the chain rule through the features (with one-hot features
    this is exactly the tabular update)."""
    s, a, adv = states.ravel(), act.ravel(), adv.ravel()
    count = np.maximum(np.bincount(s, minlength=N_STATES), 1)
    p_old = student.probs(theta)[s]
    pi_old = np.where(a == 1, p_old, 1 - p_old)
    for _ in range(epochs):
        p = student.probs(theta)[s]
        ratio = np.where(a == 1, p, 1 - p) / pi_old
        active = ~(((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps)))
        g_state = np.bincount(s, weights=active * adv * ratio * (a - p), minlength=N_STATES) / count
        z = student.phi @ theta
        g_state = g_state + ent * -z * sigmoid(z) * (1 - sigmoid(z)) * (np.bincount(s, minlength=N_STATES) > 0)
        theta = theta + lr * student.phi.T @ g_state
    return theta


def ppo(student, rng, iters=120, episodes=3072, lr=0.065, eps=0.2, epochs=4, ent=0.0):
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, act, _, rew = rollout(student.probs(theta), episodes, rng)
        G = returns_to_go(rew)
        n = np.bincount(states.ravel(), minlength=N_STATES)
        adv = G - (np.bincount(states.ravel(), weights=G.ravel(), minlength=N_STATES) / np.maximum(n, 1))[states]
        adv = adv / (adv.std() + 1e-8)
        theta = clipped_update(student, theta, states, act, adv, lr, eps, epochs, ent)
    return student.probs(theta), []


def grpo(student, rng, iters=120, groups=192, group_size=8, lr=0.065, eps=0.2, epochs=4):
    theta = np.zeros(student.n_params)
    for _ in range(iters):
        states, act, _, rew = rollout(student.probs(theta), groups * group_size, rng)
        R = rew.sum(1).reshape(groups, group_size)
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)
        theta = clipped_update(student, theta, states, act, np.repeat(A.reshape(-1, 1), T, 1), lr, eps, epochs)
    return student.probs(theta), []


METHODS = {  # the same budget and grids as reproduce.py; tuned separately at every student size
    "DAgger": Method(dagger, EPISODES, fixed={"episodes": EPISODES}),
    "AggreVaTe": Method(aggrevate, 2 * EPISODES, fixed={"episodes": EPISODES}),
    "LOLS": Method(lols, 3 * EPISODES, grid={"beta": [0.0, 0.5]}, fixed={"episodes": EPISODES}),
    "APPO": Method(ppo, EPISODES, grid={"lr": LR_GRID}, fixed={"episodes": EPISODES}),
    "AGRPO": Method(grpo, 192 * 8, grid={"lr": LR_GRID}),
}
SELECT = {name: ("signed_return", +1) for name in METHODS}
METRICS = ["p_deviate", "signed_return", "errors", "success", "greedy_signed_return", "greedy_errors", "greedy_success"]


def main():
    out = Path(__file__).parent / "results"
    runs, tuning, summary = [], {}, {}
    for k in DEGREES:
        rows, tuning[k] = tune_and_report(Student(k), METHODS, evaluate, SELECT, f"k={k}",
                                          ["p_deviate", "errors", "success", "greedy_errors"])
        runs += [{"degree": k, **r} for r in rows]
        summary[str(k)] = summarize(rows, METRICS)
    (out / "distill.json").write_text(json.dumps({"runs": runs, "summary": summary, "tuning": tuning}))


if __name__ == "__main__":
    main()
