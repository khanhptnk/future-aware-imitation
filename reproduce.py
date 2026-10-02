"""DAgger, AggreVaTe, LOLS, APPO and AGRPO on two toy POMDPs where some imitation error is unavoidable.

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

APPO (Agreement PPO) is PPO with a reward of +1 when the learner's action matches the expert's and -1 otherwise,
credited with undiscounted returns; AGRPO uses the same reward with a minimal GRPO-style update.

Run: uv run reproduce.py           (CPU, a few minutes: tunes, then reports; writes results/runs.json, tuning.json)
     uv run reproduce.py --note    (the original note's settings and tables)
"""

import json
import sys
from pathlib import Path

import numpy as np

from protocol import EPISODES, REPORT_SEEDS, Method, mean_sd, tune_and_report

H = 8  # downstream steps; 9 decisions in all
T = H + 1
EVAL_EPISODES = 50_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Env:
    def __init__(self, name: str):
        assert name in ("reveal", "hard")
        self.name = name
        self.n_obs = 4

    def branch(self, a0: np.ndarray, z: np.ndarray, rng: np.random.Generator):
        """Given the root action, each episode's downstream observation and the expert's 8 downstream actions."""
        if self.name == "reveal":
            return np.where(a0 == 1, 2 + z, 1), np.repeat(z[:, None], H, 1)
        down = np.select([a0 == z, a0 == 1], [1, 2], 3)  # 1 easy, 2 recoverable, 3 hard
        return down, np.where((down == 3)[:, None], rng.integers(0, 2, (len(z), H)), 0)

    def rollout(self, p1: np.ndarray, z: np.ndarray, rng: np.random.Generator):
        """Roll out the policy with P(action 1 | obs) = p1[obs] on episodes with hidden bits z.
        Returns obs, actions, expert actions, rewards, each of shape (n, T)."""
        n = len(z)
        a0 = (rng.random(n) < p1[0]).astype(int)
        down, expert_down = self.branch(a0, z, rng)
        obs = np.column_stack([np.zeros(n, int), np.repeat(down[:, None], H, 1)])
        act = np.column_stack([a0, (rng.random((n, H)) < p1[down][:, None]).astype(int)])
        expert = np.column_stack([z, expert_down])
        rew = np.where(act == expert, 1.0, -1.0)
        return obs, act, expert, rew

    def continue_from(self, z, a0, t, a, p1, expert_rollout, rng):
        """Agreement collected from step t on, in episodes whose roll-in chose root action a0, after taking action a at
        step t and then following the expert (where expert_rollout) or the learner p1. Used by LOLS."""
        n = len(z)
        down, expert_down = self.branch(np.where(t == 0, a, a0), z, rng)
        expert = np.column_stack([z, expert_down])
        learner = (rng.random((n, T)) < p1[np.column_stack([np.zeros(n, int), np.repeat(down[:, None], H, 1)])])
        act = np.where(expert_rollout[:, None], expert, learner.astype(int))
        act[np.arange(n), t] = a
        agree = np.where(act == expert, 1.0, -1.0)
        return (agree * (np.arange(T)[None, :] >= t[:, None])).sum(1)


def evaluate(env: Env, p1: np.ndarray, seed: int) -> dict:
    """Metrics of the policy as trained (actions sampled) and played greedily (its more likely action), on the same
    50,000 fresh episodes."""
    out = {"p_root_1": float(p1[0])}
    for prefix, policy in (("", p1), ("greedy_", (np.asarray(p1) > 0.5).astype(float))):
        rng = np.random.default_rng(9000 + seed)
        _, _, _, rew = env.rollout(policy, rng.integers(0, 2, EVAL_EPISODES), rng)
        errors = (rew < 0).sum(1)
        out |= {prefix + "signed_return": float(rew.sum(1).mean()), prefix + "errors": float(errors.mean()),
                prefix + "success": float((errors <= 2).mean())}
    return out


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


def explore(obs, expert, rng):
    """AggreVaTe's exploration: in each roll-in episode, at one uniformly random step t, take a uniformly random action a,
    then let the expert finish the episode. The value of (o_t, a) is the agreement collected from t on: a's own +1/-1,
    plus 1 for every later step, because the expert always agrees with itself (in every environment here, whatever a
    was). Returns the observations, actions and values, one per episode."""
    n = len(obs)
    t, a = rng.integers(0, T, n), rng.integers(0, 2, n)
    i = np.arange(n)
    return obs[i, t], a, np.where(a == expert[i, t], 1.0, -1.0) + (T - 1 - t)


def aggrevate(env: Env, rng, iters=40, episodes=3000):
    """AggreVaTe (Ross & Bagnell, 2014) with learner roll-in, the same budget as DAgger. Values are aggregated over all
    iterations, and the policy is the cost-sensitive classifier for the tabular class: at each observation, the action
    with the higher mean value (observations never explored stay uniform)."""
    q_sum, q_n = np.zeros((env.n_obs, 2)), np.zeros((env.n_obs, 2))
    p1, history = np.full(env.n_obs, 0.5), []
    for _ in range(iters):
        obs, _, expert, _ = env.rollout(p1, rng.integers(0, 2, episodes), rng)
        o, a, q = explore(obs, expert, rng)
        np.add.at(q_sum, (o, a), q)
        np.add.at(q_n, (o, a), 1)
        mean = q_sum / np.maximum(q_n, 1)
        p1 = np.where((q_n > 0).all(1), (mean[:, 1] > mean[:, 0]).astype(float), 0.5)
        history.append(p1[0])
    return p1, history


def lols(env: Env, rng, beta=0.0, iters=40, episodes=3000):
    """LOLS (Chang et al., 2015) with learner roll-in. In each episode, at one uniformly random step t, try both actions;
    after each, roll out to the end with the expert (probability beta, one choice per episode) or the learner, and score
    the action by the agreement collected from t on. The policy is the same cost-sensitive classifier as aggrevate.
    beta = 0: learned roll-outs only; beta = 1 would be AggreVaTe's expert roll-outs, but trying both actions."""
    q_sum, q_n = np.zeros((env.n_obs, 2)), np.zeros((env.n_obs, 2))
    p1, history = np.full(env.n_obs, 0.5), []
    for _ in range(iters):
        z = rng.integers(0, 2, episodes)
        obs, act, _, _ = env.rollout(p1, z, rng)
        t = rng.integers(0, T, episodes)
        o = obs[np.arange(episodes), t]
        expert_rollout = rng.random(episodes) < beta
        for a in (0, 1):
            q = env.continue_from(z, act[:, 0], t, np.full(episodes, a), p1, expert_rollout, rng)
            np.add.at(q_sum[:, a], o, q)
            np.add.at(q_n[:, a], o, 1)
        mean = q_sum / np.maximum(q_n, 1)
        p1 = np.where((q_n > 0).all(1), (mean[:, 1] > mean[:, 0]).astype(float), 0.5)
        history.append(p1[0])
    return p1, history


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


LR_GRID = [0.0075, 0.015, 0.03, 0.065, 0.13, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
METHODS = {  # every method simulates protocol.BUDGET episodes; grids are searched per environment
    "DAgger": Method(dagger, EPISODES, fixed={"episodes": EPISODES}),
    "AggreVaTe": Method(aggrevate, 2 * EPISODES, fixed={"episodes": EPISODES}),  # roll-in + one expert roll-out
    "LOLS": Method(lols, 3 * EPISODES, grid={"beta": [0.0, 0.5]}, fixed={"episodes": EPISODES}),  # + two roll-outs
    "APPO": Method(ppo, EPISODES, grid={"lr": LR_GRID}, fixed={"episodes": EPISODES}),
    "AGRPO": Method(grpo, 192 * 8, grid={"lr": LR_GRID}),
}
SELECT = {name: ("signed_return", +1) for name in METHODS}  # the shared imitation objective J
NOTE = {  # the original note's settings, which reproduce its tables (python reproduce.py --note)
    "DAgger": lambda env, rng: dagger(env, rng, iters=40, episodes=3000),
    "PPO": lambda env, rng: ppo(env, rng, iters=120, episodes=3072, lr=0.065),
    "GRPO": lambda env, rng: grpo(env, rng, iters=120, groups=192, lr=0.065),
}


def main():
    if "--note" in sys.argv:
        for env_name in ("reveal", "hard"):
            for name, train in NOTE.items():
                rows = [evaluate(Env(env_name), train(Env(env_name), np.random.default_rng(s))[0], s) for s in REPORT_SEEDS]
                print(f"{env_name:6s} {name:6s} " + "  ".join(f"{k} {mean_sd(rows, k)[0]:.4f} ± {mean_sd(rows, k)[1]:.4f}"
                                                         for k in ("p_root_1", "signed_return", "errors", "success")))
        return
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    runs, tuning = [], {}
    for env_name in ("reveal", "hard"):
        rows, tuning[env_name] = tune_and_report(Env(env_name), METHODS, evaluate, SELECT, env_name,
                                                 ["p_root_1", "errors", "success", "greedy_errors", "greedy_success"])
        runs += [{"env": env_name, **r} for r in rows]
    (out / "runs.json").write_text(json.dumps(runs))
    (out / "tuning.json").write_text(json.dumps(tuning, indent=1))


if __name__ == "__main__":
    main()
