"""APPO with an entropy bonus: does it change what APPO learns?

APPO has no entropy bonus. This sweep adds one, ent * H(pi(. | o)) per visited observation, with coefficients
ENTS, at APPO's tuned learning rate in each setting (read from the tuning results), and reports the same metrics as the
main experiments on the reporting seeds. It is a sweep, not a tuning step: tuning the coefficient by APPO's own objective
would pick 0, since an entropy bonus can only cost agreement.

Run: uv run entropy_ablation.py        (CPU, ~10 min; after reproduce.py, distill.py and distill_stochastic.py;
                                        writes results/entropy.json)
"""

import functools
import json
from pathlib import Path

import numpy as np

import distill
import distill_stochastic
import reproduce
from protocol import REPORT_SEEDS, mean_sd

ENTS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def settings(results):
    tuning = json.loads((results / "tuning.json").read_text())
    det = json.loads((results / "distill.json").read_text())["tuning"]
    sto = json.loads((results / "distill_stochastic.json").read_text())["tuning"]
    for env in ("reveal", "hard"):
        yield (f"case {1 if env == 'reveal' else 2}", reproduce.Env(env), reproduce.ppo, reproduce.evaluate,
               tuning[env]["APPO"]["best"]["lr"])
    for k in (1, 7):
        yield f"case 3, degree {k}", distill.Student(k), distill.ppo, distill.evaluate, det[str(k)]["APPO"]["best"]["lr"]
    for k in (1, 7):
        yield (f"case 3 stochastic, degree {k}", distill.Student(k),
               functools.partial(distill_stochastic.rl, reward="agreement", use_returns=True),
               distill_stochastic.evaluate, sto[str(k)]["APPO"]["best"]["lr"])


def main():
    results = Path(__file__).parent / "results"
    out = []
    for name, setting, train, evaluate, lr in settings(results):
        for ent in ENTS:
            rows = [evaluate(setting, train(setting, np.random.default_rng(s), lr=lr, ent=ent)[0], s)
                    for s in REPORT_SEEDS]
            root = "p_root_1" if "p_root_1" in rows[0] else "p_deviate"
            row = {"setting": name, "lr": lr, "ent": ent, "root": mean_sd(rows, root)}
            row |= {k: mean_sd(rows, k) for k in ("errors", "success", "reverse_kl", "forward_kl") if k in rows[0]}
            out.append(row)
            print(f"{name:28s} lr {lr:<5} ent {ent:<6} root {row['root'][0]:.3f}±{row['root'][1]:.3f}  "
                  + "  ".join(f"{k} {row[k][0]:.3f}" for k in ("errors", "success", "reverse_kl", "forward_kl") if k in row),
                  flush=True)
    (results / "entropy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
