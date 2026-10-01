"""Figures for the post, each rendered twice (light and dark theme) as SVG, from the JSON files in results/.

Run: uv run plots.py        (after reproduce.py, distill.py, distill_stochastic.py; writes figures/*.svg)
"""
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from reproduce import H

for f in Path("/usr/share/fonts/truetype/ubuntu").glob("UbuntuMono*.ttf"):  # the blog's font, if installed
    font_manager.fontManager.addfont(str(f))
FONT = "Ubuntu Mono" if any(f.name == "Ubuntu Mono" for f in font_manager.fontManager.ttflist) else "DejaVu Sans Mono"

THEMES = {  # validated palette steps (dataviz reference palette), site surfaces #faf8f8 / #161618
    "light": dict(surface="#faf8f8", ink="#2b2b2b", ink2="#52514e", grid="#e5e5e5", ref="#6b6a66",
                  DAgger="#2a78d6", PPO="#eb6834", GRPO="#1baf7a"),
    "dark": dict(surface="#161618", ink="#ebebec", ink2="#c3c2b7", grid="#393639", ref="#a3a29a",
                 DAgger="#3987e5", PPO="#d95926", GRPO="#199e70"),
}
METHODS = ["DAgger", "PPO", "GRPO"]
LW = 1.6  # ~2px


def style(ax, c):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["grid"])
    ax.tick_params(colors=c["ink2"], labelsize=10, length=0, pad=6)
    ax.grid(True, color=c["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(c["ink2"])
    ax.yaxis.label.set_color(c["ink2"])


def new_fig(c, w=7.6, h=3.6, ncols=1):
    plt.rcParams.update({"font.family": FONT, "svg.fonttype": "path", "svg.hashsalt": "fai"})
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_alpha(0)
    return fig, np.atleast_1d(axes)


def save(fig, out):
    fig.savefig(out, format="svg", transparent=True, metadata={"Date": None})  # no timestamp: reruns are byte-identical
    plt.close(fig)


# ---------------------------------------------------------------- figure 1: the information-reveal environment
def fig_env(c, out):
    fig, (ax,) = new_fig(c, w=7.6, h=2.9)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, text, w=2.5, h=0.95):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="none", ec=c["ink2"], lw=1.1))
        ax.text(x, y, text, ha="center", va="center", color=c["ink"], fontsize=10, linespacing=1.4)

    def arrow(x0, y0, x1, y1, text, color, dy):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=LW, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="center", color=c["ink"], fontsize=9.5)

    box(1.45, 2.0, "root\nhides z", w=2.0)
    box(7.6, 3.15, "reveal-z  (8 steps)\nz is visible", w=3.6)
    box(7.6, 0.85, "hidden  (8 steps)\nz still hidden", w=3.6)
    arrow(2.5, 2.2, 5.75, 3.05, "action 1", c["PPO"], 0.35)
    arrow(2.5, 1.8, 5.75, 0.95, "action 0", c["DAgger"], -0.35)
    ax.text(4.1, 2.0, "either action matches\nthe expert with prob. 1/2", ha="center", va="center", color=c["ink2"],
            fontsize=9, linespacing=1.3)
    ax.text(7.6, 2.0, "expert plays a* = z at every step", ha="center", va="center", color=c["ink2"], fontsize=9)
    fig.tight_layout(pad=0.2)
    save(fig, out)


# ---------------------------------------------------------------- figure 2: the root decision during training
def fig_root(c, runs, out):
    fig, axes = new_fig(c, w=7.6, h=3.4, ncols=2)
    titles = {"reveal": "(a) information reveal",
              "hard": "(b) hard expert"}
    for ax, env in zip(axes, ("reveal", "hard")):
        style(ax, c)
        for m in METHODS:
            rs = [r for r in runs if r["env"] == env and r["method"] == m]
            hist = np.array([[0.5] + r["root_history"] for r in rs])
            x = np.arange(hist.shape[1]) * rs[0]["episodes_per_iter"] / 1000
            ax.fill_between(x, hist.min(0), hist.max(0), color=c[m], alpha=0.25, lw=0)
            ax.plot(x, hist.mean(0), color=c[m], lw=LW, label=m)
            ax.annotate(f" {m}", (x[-1], hist.mean(0)[-1]), color=c["ink"], fontsize=9.5, va="center",
                        xytext=(0, {"DAgger": 0, "PPO": 5, "GRPO": -5}[m] if env == "reveal" else 0),
                        textcoords="offset points")
        ax.set_ylim(0.45, 1.02)
        ax.set_xlim(0, 400)
        ax.set_xlabel("training episodes (thousands)")
        ax.set_title(titles[env], color=c["ink"], fontsize=10, loc="left")
    axes[0].set_ylabel("P(root action 1)")
    axes[0].legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="center right", bbox_to_anchor=(1.0, 0.35))
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- figure 3: errors per episode, information reveal
def error_dist(p1: list) -> np.ndarray:
    """Exact distribution of the number of expert disagreements in a 9-step reveal episode under policy p1."""
    dist = np.zeros(H + 2)
    binom = lambda q: np.array([comb(H, k) * q ** k * (1 - q) ** (H - k) for k in range(H + 1)])
    for z in (0, 1):
        for a0, pa in ((1, p1[0]), (0, 1 - p1[0])):
            o = 2 + z if a0 == 1 else 1
            q = 1 - p1[o] if z == 1 else p1[o]  # per-step error probability downstream
            dist[int(a0 != z):int(a0 != z) + H + 1] += 0.5 * pa * binom(q)
    return dist


def fig_errors(c, runs, out):
    fig, (ax,) = new_fig(c, w=7.6, h=3.2)
    style(ax, c)
    k = np.arange(H + 2)
    width = 0.27
    for i, m in enumerate(METHODS):
        d = np.mean([error_dist(r["policy"]) for r in runs if r["env"] == "reveal" and r["method"] == m], 0)
        ax.bar(k + (i - 1) * width, d, width=width - 0.04, color=c[m], label=m, lw=0)
    ax.axvspan(-0.5, 2.5, color=c["grid"], alpha=0.45, lw=0, zorder=0)
    ax.text(1.0, 0.62, "success: ≤ 2 errors", ha="center", color=c["ink2"], fontsize=9.5)
    ax.set_xticks(k)
    ax.set_xlim(-0.5, H + 1.5)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("expert disagreements in the episode (of 9 decisions)")
    ax.set_ylabel("fraction of episodes")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="upper right")
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- figure 4: distillation, deterministic teacher
def fig_distill(c, data, out):
    fig, (ax,) = new_fig(c, w=7.6, h=3.4)
    style(ax, c)
    ks = sorted(int(k) for k in data["summary"])
    ax.step(ks, [np.ceil((7 - k) / 2) for k in ks], where="mid", color=c["ref"], lw=1.1, ls=(0, (1, 2)))
    ax.annotate("fewest errors possible\nwhen copying the teacher", (2.55, 2.12), color=c["ink2"], fontsize=9)
    ax.axhline(1, color=c["ref"], lw=1.0, ls=(0, (4, 3)))
    ax.annotate("leaving the teacher's path: 1 error", (0.05, 0.7), color=c["ink2"], fontsize=9)
    for m in METHODS:
        mu = [data["summary"][str(k)][m]["errors"][0] for k in ks]
        ax.plot(ks, mu, color=c[m], lw=LW, marker="o", ms=5, label=m)
    ax.set_xticks(ks)
    ax.set_ylim(-0.15, 4.4)
    ax.set_xlabel("student size: polynomial degree k (the teacher needs 7)")
    ax.set_ylabel("disagreements per episode")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- figure 5: distillation, stochastic teacher
SOFT = [("forward KL", "on-policy forward KL", "DAgger", "-", "o"),
        ("reverse KL (γ=0)", "reverse KL, discount 0", "DAgger", (0, (4, 2)), "s"),
        ("reverse KL", "reverse KL with returns", "PPO", (0, (4, 2)), "s"),
        ("±1 agreement", "±1 agreement (PPO)", "PPO", "-", "o")]


def fig_distill_soft(c, data, out):
    fig, axes = new_fig(c, w=7.6, h=3.5, ncols=2)
    ks = sorted(int(k) for k in data["summary"])
    for ax, key, title in ((axes[0], "p_deviate", "(a) P(leave the teacher's path)"),
                           (axes[1], "errors", "(b) disagreements per episode")):
        style(ax, c)
        for name, lab, col, ls, mk in SOFT:
            ax.plot(ks, [data["summary"][str(k)][name][key][0] for k in ks], color=c[col], lw=LW, ls=ls,
                    marker=mk, ms=4.5, label=lab)
        ax.set_xticks(ks)
        ax.set_xlabel("student degree k")
        ax.set_title(title, color=c["ink"], fontsize=10, loc="left")
    axes[0].axhline(0.1, color=c["ref"], lw=0.9, ls=(0, (1, 2)))
    axes[0].annotate("teacher: 0.1", (0.0, 0.14), color=c["ink2"], fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 4.0)
    axes[1].legend(frameon=False, labelcolor=c["ink"], fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, out)


if __name__ == "__main__":
    runs = json.load(open("results/runs.json"))
    distill = json.load(open("results/distill.json"))
    distill_soft = json.load(open("results/distill_stochastic.json"))
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for theme, c in THEMES.items():
        fig_env(c, out / f"env-{theme}.svg")
        fig_root(c, runs, out / f"root-{theme}.svg")
        fig_errors(c, runs, out / f"errors-{theme}.svg")
        fig_distill(c, distill, out / f"distill-{theme}.svg")
        fig_distill_soft(c, distill_soft, out / f"distill-soft-{theme}.svg")
    print("wrote", sorted(p.name for p in out.glob("*.svg")))
