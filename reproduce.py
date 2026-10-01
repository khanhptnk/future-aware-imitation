"""DAgger vs PPO vs GRPO on two toy POMDPs where some imitation error is unavoidable.

Every episode samples a hidden bit z ~ Bernoulli(0.5) that the expert sees and the learner doesn't. The learner makes
9 binary decisions: one at a root observation that hides z, then H = 8 downstream decisions. Each decision earns +1 if it
matches the expert's action and -1 if not. The policy is tabular: one Bernoulli logit per observation ID, so nothing
learned downstream can leak back to the root.

  reveal: the expert plays z everywhere. Root action 1 leads to downstream observations that reveal z (IDs 2 and 3);
          root action 0 leads to one downstream observation that still hides it (ID 1).
  hard:   a correct root action enters an easy corridor (ID 1). The root mistake "action 1 when z = 0" enters a
          recoverable corridor (ID 2), and the mistake "action 0 when z = 1" a hard one (ID 3). The expert plays action 0
          throughout the easy and recoverable corridors, and a fresh coin flip at every step of the hard one.

In both, the two root actions match the expert equally often (1/2), so a per-step imitation loss can't tell them
apart, but action 1 leaves a future that can be imitated.

Run: uv run reproduce.py        (CPU, ~1 min; writes results/runs.json and results/summary.json)
"""

import json
import time
from pathlib import Path

import numpy as np

H = 8  # downstream steps; 9 decisions in all
T = H + 1
SEEDS = range(12)
EVAL_EPISODES = 50_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Env:
    def __init__(self, name: str):
        assert name in ("reveal", "hard")
        self.name = name
        self.n_obs = 4

    def rollout(self, p1: np.ndarray, z: np.ndarray, rng: np.random.Generator):
        """Roll out the policy with P(action 1 | obs) = p1[obs] on episodes with hidden bits z.
        Returns obs, actions, expert actions, rewards, each of shape (n, T)."""
        n = len(z)
        a0 = (rng.random(n) < p1[0]).astype(int)
        if self.name == "reveal":
            down = np.where(a0 == 1, 2 + z, 1)
            expert_down = np.repeat(z[:, None], H, 1)
        else:
            down = np.select([a0 == z, a0 == 1], [1, 2], 3)  # 1 easy, 2 recoverable, 3 hard
            expert_down = np.where((down == 3)[:, None], rng.integers(0, 2, (n, H)), 0)
        obs = np.column_stack([np.zeros(n, int), np.repeat(down[:, None], H, 1)])
        act = np.column_stack([a0, (rng.random((n, H)) < p1[down][:, None]).astype(int)])
        expert = np.column_stack([z, expert_down])
        rew = np.where(act == expert, 1.0, -1.0)
        return obs, act, expert, rew


def evaluate(env: Env, p1: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(9000 + seed)
    z = rng.integers(0, 2, EVAL_EPISODES)
    _, _, _, rew = env.rollout(p1, z, rng)
    errors = (rew < 0).sum(1)
    return {
        "p_root_1": float(p1[0]),
        "signed_return": float(rew.sum(1).mean()),
        "errors": float(errors.mean()),
        "success": float((errors <= 2).mean()),
    }


def dagger(env: Env, rng, iters=40, episodes=3000, pseudocount=1e-3):
    """Roll out the learner, label every visited observation with the expert's action, aggregate the counts over all
    iterations, and refit the tabular policy by maximum likelihood (the label frequencies).
    Returns the final P(action 1 | obs) and the root's P(action 1) after each iteration."""
    counts = np.full((env.n_obs, 2), pseudocount)
    history = []
    for _ in range(iters):
        p1 = counts[:, 1] / counts.sum(1)
        obs, _, expert, _ = env.rollout(p1, rng.integers(0, 2, episodes), rng)
        np.add.at(counts, (obs.ravel(), expert.ravel()), 1)
        history.append(counts[0, 1] / counts[0].sum())
    return counts[:, 1] / counts.sum(1), history


def clipped_update(theta, obs, act, adv, lr, eps, epochs):
    """Full-batch gradient ascent on the PPO clipped surrogate. Each observation's logit gets the mean gradient over the
    samples at that observation (the normalization that reproduces the note's numbers)."""
    obs, act, adv = obs.ravel(), act.ravel(), adv.ravel()
    count = np.maximum(np.bincount(obs, minlength=len(theta)), 1)
    p_old = sigmoid(theta[obs])
    pi_old = np.where(act == 1, p_old, 1 - p_old)
    for _ in range(epochs):
        p = sigmoid(theta[obs])
        ratio = np.where(act == 1, p, 1 - p) / pi_old
        active = ~(((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps)))  # clipped samples: no gradient
        g = active * adv * ratio * (act - p)  # d/dlogit of ratio * A = ratio * A * (a - p)
        theta = theta + lr * np.bincount(obs, weights=g, minlength=len(theta)) / count
    return theta


def returns_to_go(rew):
    return np.flip(np.cumsum(np.flip(rew, 1), 1), 1)


def ppo(env: Env, rng, iters=120, episodes=3072, lr=0.065, eps=0.2, epochs=4):
    """No value network: the baseline for each sample is the batch-mean return-to-go of samples with the same
    observation, then all advantages are divided by their global standard deviation."""
    theta = np.zeros(env.n_obs)
    history = []
    for _ in range(iters):
        obs, act, _, rew = env.rollout(sigmoid(theta), rng.integers(0, 2, episodes), rng)
        G = returns_to_go(rew)
        n = np.bincount(obs.ravel(), minlength=env.n_obs)
        baseline = np.bincount(obs.ravel(), weights=G.ravel(), minlength=env.n_obs) / np.maximum(n, 1)
        adv = G - baseline[obs]
        adv = adv / (adv.std() + 1e-8)
        theta = clipped_update(theta, obs, act, adv, lr, eps, epochs)
        history.append(sigmoid(theta[0]))
    return sigmoid(theta), history


def grpo(env: Env, rng, iters=120, groups=192, group_size=8, lr=0.065, eps=0.2, epochs=4):
    """Groups of trajectories that share z. Each trajectory's total return is normalized within its group, and that one
    advantage is given to every action in the trajectory."""
    theta = np.zeros(env.n_obs)
    history = []
    for _ in range(iters):
        z = np.repeat(rng.integers(0, 2, groups), group_size)
        obs, act, _, rew = env.rollout(sigmoid(theta), z, rng)
        R = rew.sum(1).reshape(groups, group_size)
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)
        adv = np.repeat(A.reshape(-1, 1), T, 1)
        theta = clipped_update(theta, obs, act, adv, lr, eps, epochs)
        history.append(sigmoid(theta[0]))
    return sigmoid(theta), history


METHODS = {"DAgger": dagger, "PPO": ppo, "GRPO": grpo}
EPISODES_PER_ITER = {"DAgger": 3000, "PPO": 3072, "GRPO": 192 * 8}
METRICS = ["p_root_1", "signed_return", "errors", "success"]


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    runs, summary = [], {}
    for env_name in ("reveal", "hard"):
        env = Env(env_name)
        for method, train in METHODS.items():
            t0 = time.time()
            rows = []
            for seed in SEEDS:
                p1, history = train(env, np.random.default_rng(seed))
                rows.append({"env": env_name, "method": method, "seed": seed, "policy": p1.tolist(),
                             **evaluate(env, p1, seed), "episodes_per_iter": EPISODES_PER_ITER[method],
                             "root_history": [float(h) for h in history]})
            runs += rows
            stats = {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
                     for m in METRICS}
            summary.setdefault(env_name, {})[method] = stats
            print(f"{env_name:6s} {method:6s} " + "  ".join(f"{m} {mu:.4f} ± {sd:.4f}" for m, (mu, sd) in stats.items())
                  + f"  ({time.time() - t0:.1f}s)")
    (out / "runs.json").write_text(json.dumps(runs))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
