"""Numerical checks of the implementations. Each check prints PASS or FAIL with the largest discrepancy.

Run: uv run checks.py        (CPU, ~1 min)
"""

import json
from pathlib import Path

import numpy as np

import distill
import distill_stochastic as DS
import plots
import reproduce as R
from protocol import BUDGET, EPISODES

rng = np.random.default_rng(0)
FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILED.append(name)


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ---------------------------------------------------------------- 1. the clipped update follows the surrogate's gradient
def surrogate(logits_of, theta, theta_old, idx, act, adv, eps, n_states):
    """sum over states of the mean over that state's samples of min(rho A, clip(rho) A)."""
    p, p_old = R.sigmoid(logits_of(theta))[idx], R.sigmoid(logits_of(theta_old))[idx]
    rho = np.where(act == 1, p, 1 - p) / np.where(act == 1, p_old, 1 - p_old)
    obj = np.minimum(rho * adv, np.clip(rho, 1 - eps, 1 + eps) * adv)
    count = np.maximum(np.bincount(idx, minlength=n_states), 1)
    return (np.bincount(idx, weights=obj, minlength=n_states) / count).sum()


def check_update(name, update, logits_of, n_params, n_states):
    n = 400
    idx = rng.integers(0, n_states, n)
    act = rng.integers(0, 2, n)
    adv = rng.normal(size=n)
    theta0 = rng.normal(size=n_params) * 0.5
    lr, eps = 3.0, 0.2  # large enough that the second epoch clips some samples
    # two epochs: the first starts at theta_old (ratio 1), the second has some samples clipped
    theta2 = update(theta0, idx, act, adv, lr, eps, 2)
    g0 = num_grad(lambda th: surrogate(logits_of, th, theta0, idx, act, adv, eps, n_states), theta0)
    theta1 = theta0 + lr * g0
    g1 = num_grad(lambda th: surrogate(logits_of, th, theta0, idx, act, adv, eps, n_states), theta1)
    err = np.abs(theta2 - (theta1 + lr * g1)).max()
    p1, p0 = R.sigmoid(logits_of(theta1))[idx], R.sigmoid(logits_of(theta0))[idx]
    rho = np.where(act == 1, p1, 1 - p1) / np.where(act == 1, p0, 1 - p0)
    clipped = int((((adv > 0) & (rho > 1 + eps)) | ((adv < 0) & (rho < 1 - eps))).sum())
    check(name, err < 1e-6, f"max error {err:.1e}; {clipped} samples clipped in the second epoch")


check_update("tabular clipped update = gradient ascent on the per-observation-mean surrogate",
             lambda th, o, a, A, lr, eps, ep: R.clipped_update(th, o, a, A, lr, eps, ep), lambda th: th, 4, 4)
for k in (1, 4):
    st = distill.Student(k)
    check_update(f"feature clipped update (degree {k}) = gradient ascent on the same surrogate",
                 lambda th, s, a, A, lr, eps, ep, st=st: distill.clipped_update(st, th, s, a, A, lr, eps, ep),
                 lambda th, st=st: st.phi @ th, st.n_params, distill.N_STATES)


# ---------------------------------------------------------------- 2. entropy and JSD gradients
def bern_entropy(z):
    p = R.sigmoid(z)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


z = np.linspace(-4, 4, 9)
analytic = -z * R.sigmoid(z) * (1 - R.sigmoid(z))
numeric = np.array([(bern_entropy(x + 1e-6) - bern_entropy(x - 1e-6)) / 2e-6 for x in z])
check("entropy bonus gradient -theta p (1 - p)", np.abs(analytic - numeric).max() < 1e-7,
      f"max error {np.abs(analytic - numeric).max():.1e}")
# the entropy term in the tabular update: with zero advantages, one step adds lr * ent * dH/dlogit at visited states
th = rng.normal(size=4); obs = np.array([0, 0, 2, 3]); acts = np.array([0, 1, 1, 0])
step = R.clipped_update(th, obs, acts, np.zeros(4), 1e-3, 0.2, 1, ent=1.0) - th
expected = 1e-3 * -th * R.sigmoid(th) * (1 - R.sigmoid(th)) * np.array([1, 0, 1, 1])
check("entropy term enters the update at visited observations only", np.abs(step - expected).max() < 1e-12)


def jsd_objective(theta, st, w, beta=0.5):
    q = st.probs(theta); p = DS.TEACHER_P1; m = beta * p + (1 - beta) * q
    kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    return (w * (beta * kl(p, m) + (1 - beta) * kl(q, m))).sum()


st = distill.Student(3)
w = rng.random(distill.N_STATES); w /= w.sum()
theta = rng.normal(size=st.n_params) * 0.5
zz = st.phi @ theta; q = R.sigmoid(zz); m = 0.5 * DS.TEACHER_P1 + 0.5 * q
g_code = st.phi.T @ (w * 0.5 * (zz - np.log(m / (1 - m))) * q * (1 - q))  # the expression in distill_stochastic.jsd
g_num = num_grad(lambda t: jsd_objective(t, st, w), theta)
check("JSD gradient used by distill_stochastic.jsd", np.abs(g_code - g_num).max() < 1e-7,
      f"max error {np.abs(g_code - g_num).max():.1e}")


# MiniLLM's single-step part: d/dz E_{a ~ q}[log p(a) - log q(a)] = (logit p - z) q (1 - q)
zz = np.linspace(-3, 3, 7); pp = np.array([0.1, 0.9, 0.1, 0.9, 0.5, 0.1, 0.9])
f = lambda z: (R.sigmoid(z) * np.log(pp / R.sigmoid(z)) + (1 - R.sigmoid(z)) * np.log((1 - pp) / (1 - R.sigmoid(z))))
analytic = (np.log(pp / (1 - pp)) - zz) * R.sigmoid(zz) * (1 - R.sigmoid(zz))
numeric = (f(zz + 1e-6) - f(zz - 1e-6)) / 2e-6
check("MiniLLM single-step gradient (logit p - z) q (1 - q)", np.abs(analytic - numeric).max() < 1e-7,
      f"max error {np.abs(analytic - numeric).max():.1e}")


# ---------------------------------------------------------------- 3. AggreVaTe's value = an actual expert roll-out
for env_name in ("reveal", "hard"):
    env = R.Env(env_name)
    n = 20000
    z = rng.integers(0, 2, n)
    obs, act, expert, _ = env.rollout(np.full(4, 0.5), z, rng)
    t, a = rng.integers(0, R.T, n), rng.integers(0, 2, n)
    shortcut = np.where(a == expert[np.arange(n), t], 1.0, -1.0) + (R.T - 1 - t)
    # simulate: take a at t, then let the expert play the rest (continue_from with expert roll-outs)
    simulated = env.continue_from(z, act[:, 0], t, a, np.full(4, 0.5), np.ones(n, bool), rng)
    # at t > 0 in the hard corridor, continue_from redraws the expert's coin at step t; compare where it doesn't
    same_state = ~((t > 0) & (obs[np.arange(n), t] == 3)) if env_name == "hard" else np.ones(n, bool)
    err = np.abs(shortcut - simulated)[same_state].max()
    check(f"AggreVaTe value shortcut = simulated expert roll-out ({env_name})", err == 0,
          f"{same_state.sum()} samples, max difference {err}")
    if env_name == "hard":  # in the hard corridor the shortcut and the simulation agree in distribution
        hc = ~same_state
        check("  ... and in the hard corridor, in expectation", abs(shortcut[hc].mean() - simulated[hc].mean()) < 0.05,
              f"means {shortcut[hc].mean():.3f} vs {simulated[hc].mean():.3f}")
st = distill.Student(1)
n = 20000
states, act, teacher, _ = distill.rollout(np.full(distill.N_STATES, 0.5), n, rng)
t, a = rng.integers(0, R.T, n), rng.integers(0, 2, n)
shortcut = np.where(a == teacher[np.arange(n), t], 1.0, -1.0) + (R.T - 1 - t)
simulated = distill.continue_from(act[:, 0], t, a, np.full(distill.N_STATES, 0.5), np.ones(n, bool), rng)
check("AggreVaTe value shortcut = simulated teacher roll-out (distillation)", np.abs(shortcut - simulated).max() == 0)


# ---------------------------------------------------------------- 4. LOLS's learner roll-outs match exact Q^pi
def exact_q_root(env_name, p1, z, a):
    """Exact Q^pi((root, z), a): a's own +-1 plus 8 later steps by the learner."""
    root = 1.0 if a == z else -1.0
    if env_name == "reveal":
        o = 2 + z if a == 1 else 1
        acc = (p1[o] if z == 1 else 1 - p1[o])  # P(learner plays z) at o
        return root + 8 * (2 * acc - 1)
    down = 1 if a == z else (2 if a == 1 else 3)
    if down == 3:
        return root + 0.0  # against coin flips, expected agreement 0
    return root + 8 * (2 * (1 - p1[down]) - 1)  # expert plays 0 in the easy and recoverable corridors


for env_name, p1 in (("reveal", np.array([0.5, 0.3, 0.1, 0.8])), ("hard", np.array([0.5, 0.2, 0.3, 0.6]))):
    env, n, worst = R.Env(env_name), 200000, 0.0
    for zz in (0, 1):
        for a in (0, 1):
            q = env.continue_from(np.full(n, zz), np.zeros(n, int), np.zeros(n, int), np.full(n, a), p1,
                                  np.zeros(n, bool), rng).mean()
            worst = max(worst, abs(q - exact_q_root(env_name, p1, zz, a)))
    check(f"LOLS learner roll-outs = exact Q^pi at the root ({env_name})", worst < 0.05, f"max error {worst:.3f}")


# ---------------------------------------------------------------- 5. exact evaluations = Monte Carlo
p1 = np.array([0.7, 0.4, 0.2, 0.9])
d = plots.error_dist(p1)
_, _, _, rew = R.Env("reveal").rollout(p1, rng.integers(0, 2, 400000), rng)
mc = np.bincount((rew < 0).sum(1), minlength=len(d)) / 400000
check("figure 4's exact error distribution = Monte Carlo", abs(d.sum() - 1) < 1e-12 and np.abs(d - mc).max() < 0.005,
      f"max difference {np.abs(d - mc).max():.4f}")

st = distill.Student(2)
p1 = st.probs(rng.normal(size=st.n_params))
ev = DS.evaluate(st, p1)
states, act, teacher, rew = distill.rollout(p1, 400000, rng)
errors = (rew < 0).sum(1)
pt = DS.TEACHER_P1[states]; ps = p1[states]
log_ratio = (np.log(np.where(act == 1, ps, 1 - ps)) - np.log(np.where(act == 1, pt, 1 - pt))).sum(1)
mc = {"errors": errors.mean(), "success": (errors <= 2).mean(), "reverse_kl": log_ratio.mean()}
worst = max(abs(ev[k] - mc[k]) for k in mc)
check("exact stochastic-teacher evaluation = Monte Carlo (errors, success, reverse KL)", worst < 0.02,
      f"max difference {worst:.4f}")
states, act, _, _ = distill.rollout(DS.TEACHER_P1, 400000, rng)
pt = DS.TEACHER_P1[states]; ps = p1[states]
fwd = (np.log(np.where(act == 1, pt, 1 - pt)) - np.log(np.where(act == 1, ps, 1 - ps))).sum(1).mean()
check("exact forward KL = Monte Carlo over teacher episodes", abs(ev["forward_kl"] - fwd) < 0.02,
      f"{ev['forward_kl']:.4f} vs {fwd:.4f}")


# ---------------------------------------------------------------- 6. every method simulates the same budget
def count_episodes(module_patches, run):
    """Count episodes passed through roll-in and roll-out functions while run() trains one method."""
    n = [0]
    originals = []
    for owner, name, size in module_patches:
        f = getattr(owner, name)
        originals.append((owner, name, f))

        def wrapped(*args, f=f, size=size):
            n[0] += size(args)
            return f(*args)
        setattr(owner, name, wrapped)
    try:
        run()
    finally:
        for owner, name, f in originals:
            setattr(owner, name, f)
    return n[0]


def first_grid_point(m):
    return m.kwargs(**{k: v[0] for k, v in m.grid.items()})


# AggreVaTe's expert roll-out is the value shortcut checked in 3, so its roll-outs are not function calls:
# count one per roll-in episode (its Method charges 2 x EPISODES per iteration for exactly that).
for method, m in R.METHODS.items():
    n = count_episodes([(R.Env, "rollout", lambda a: len(a[2])), (R.Env, "continue_from", lambda a: len(a[1]))],
                       lambda: m.train(R.Env("reveal"), np.random.default_rng(0), **first_grid_point(m)))
    n *= 2 if method == "AggreVaTe" else 1
    check(f"budget, cases 1-2: {method} simulates {n:,} episodes", n == BUDGET)
for method, m in distill.METHODS.items():
    n = count_episodes([(distill, "rollout", lambda a: a[1]), (distill, "continue_from", lambda a: len(a[0]))],
                       lambda: m.train(distill.Student(1), np.random.default_rng(0), **first_grid_point(m)))
    n *= 2 if method == "AggreVaTe" else 1
    check(f"budget, case 3: {method} simulates {n:,} episodes", n == BUDGET)
for method, m in DS.METHODS.items():
    n = count_episodes([(DS, "rollout", lambda a: a[1])],
                       lambda: m.train(distill.Student(1), np.random.default_rng(0), **first_grid_point(m)))
    check(f"budget, stochastic teacher: {method} simulates {n:,} episodes", n == BUDGET)


# ---------------------------------------------------------------- fit_logistic reaches the maximum likelihood
worst = 0.0
for k in (0, 1, 3, 7):
    st = distill.Student(k)
    n = rng.integers(1, 5000, distill.N_STATES).astype(float)
    n1 = rng.binomial(n.astype(int), rng.random(distill.N_STATES)).astype(float) + 1e-3
    n += 2e-3
    th = distill.fit_logistic(st, n1, n, np.zeros(st.n_params))
    grad = st.phi.T @ (n1 - n * st.probs(th)) - 1e-6 * th
    worst = max(worst, np.abs(grad).max() / n.sum())
check("fit_logistic converges (relative gradient at the solution)", worst < 1e-8, f"{worst:.1e}")


# ---------------------------------------------------------------- training losses used for the convergence rule
labels = rng.integers(0, 2, 5000); where = rng.integers(0, 4, 5000); pp = np.array([0.3, 0.6, 0.9, 0.5])
direct = -np.where(labels == 1, np.log(pp[where]), np.log(1 - pp[where]))
n1, n = np.bincount(where, weights=labels, minlength=4), np.bincount(where, minlength=4).astype(float)
m, se = R.log_loss(n1, n, pp)
check("log_loss = mean cross-entropy over samples, with its standard error",
      abs(m - direct.mean()) < 1e-12 and abs(se - direct.std() / np.sqrt(len(direct))) < 1e-9, f"{m:.4f} vs {direct.mean():.4f}")
q = rng.normal(5, 2, 5000); act = rng.integers(0, 2, 5000); pol = np.array([1.0, 0.0, 1.0, 0.5])
qs, qq, qn = np.zeros((4, 2)), np.zeros((4, 2)), np.zeros((4, 2))
np.add.at(qs, (where, act), q); np.add.at(qq, (where, act), q * q); np.add.at(qn, (where, act), 1)
m, se = R.cost_sensitive_loss(qs, qq, qn, pol)
mean = qs / qn
direct = -sum(qn[s].sum() / qn.sum() * ((1 - pol[s]) * mean[s, 0] + pol[s] * mean[s, 1]) for s in range(4))
check("cost_sensitive_loss = minus the state-weighted mean value of the played actions", abs(m - direct) < 1e-12,
      f"{m:.4f} vs {direct:.4f}")
for name, hist in (("APPO", R.ppo(R.Env("reveal"), np.random.default_rng(0), iters=20, episodes=500, lr=1.0)[1]),
                   ("AggreVaTe", R.aggrevate(R.Env("reveal"), np.random.default_rng(0), iters=20, episodes=500)[1])):
    ok = all(len(h) == 3 and np.isfinite(h[1]) and h[2] >= 0 for h in hist)
    check(f"{name} records (policy, loss, standard error) every iteration", ok)


# ---------------------------------------------------------------- 7. the reported settings are the best finite trials
res = Path(__file__).parent / "results"
tunings = {"cases 1-2": json.loads((res / "tuning.json").read_text()),
           "case 3": json.loads((res / "distill.json").read_text())["tuning"],
           "case 3 stochastic": json.loads((res / "distill_stochastic.json").read_text())["tuning"]}
bad = []
for label, t in tunings.items():
    for setting, ms in t.items():
        for m, v in ms.items():
            if not v["trials"]:
                continue
            sign = -1 if v["metric"] in ("errors", "reverse_kl") else 1
            finite = [tr for tr in v["trials"] if np.isfinite(tr["score"])]
            best = max(finite, key=lambda tr: sign * tr["score"])
            if best["hp"] != v["best"] and sign * best["score"] > sign * [tr["score"] for tr in v["trials"]
                                                                      if tr["hp"] == v["best"]][0]:
                bad.append((label, setting, m))
check("every reported hyperparameter is the best finite tuning trial", not bad, str(bad) if bad else "")

print(f"\n{len(FAILED)} check(s) failed" if FAILED else "\nall checks passed")
