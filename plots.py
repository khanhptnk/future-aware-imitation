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

THEMES = {  # validated palette steps (keys DAgger / PPO / GRPO name the three color slots) (dataviz reference palette), site surfaces #faf8f8 / #161618
    "light": dict(surface="#faf8f8", ink="#2b2b2b", ink2="#52514e", grid="#e5e5e5", ref="#6b6a66",
                  DAgger="#2a78d6", PPO="#eb6834", GRPO="#1baf7a"),
    "dark": dict(surface="#161618", ink="#ebebec", ink2="#c3c2b7", grid="#393639", ref="#a3a29a",
                 DAgger="#3987e5", PPO="#d95926", GRPO="#199e70"),
}
METHODS = ["DAgger", "AggreVaTe", "LOLS", "APPO", "AGRPO"]
# Composite encoding: color = whose future gets credited (blue: none beyond the current step, or the expert's;
# orange: the learner's; aqua: AGRPO), line style / hatch = which algorithm within that family.
STYLE = {"DAgger": ("DAgger", "-", "o", None), "AggreVaTe": ("DAgger", (0, (4, 2)), "s", "////"),
         "LOLS": ("PPO", (0, (4, 2)), "s", "////"), "APPO": ("PPO", "-", "o", None), "AGRPO": ("GRPO", "-", "^", None)}
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


# ---------------------------------------------------------------- summary: task success in the three settings
SUMMARY_METHODS = ["DAgger", "AggreVaTe", "LOLS", "APPO"]


def bar(ax, x, h, width, m, c, label=None):
    col, _, _, hatch = STYLE[m]
    if hatch:  # outlined and hatched in the family color
        return ax.bar(x, h, width=width, facecolor=c["surface"], edgecolor=c[col], hatch=hatch, lw=1.2, label=label)
    return ax.bar(x, h, width=width, color=c[col], lw=0, label=label)


def fig_summary(c, runs, distill, out, degree=1):
    fig, (ax,) = new_fig(c, w=7.6, h=3.4)
    style(ax, c)
    ax.grid(axis="x", visible=False)
    plt.rcParams["hatch.linewidth"] = 1.0
    settings = ["privileged\ninformation", "hard-to-imitate\nexpert", f"limited capacity\n(student degree {degree})"]
    values = {m: [100 * np.mean([r["success"] for r in runs if r["env"] == env and r["method"] == m])
                  for env in ("reveal", "hard")] + [100 * distill["summary"][str(degree)][m]["success"][0]]
              for m in SUMMARY_METHODS}
    x, width = np.arange(3), 0.2
    for i, m in enumerate(SUMMARY_METHODS):
        bars = bar(ax, x + (i - 1.5) * width, values[m], width - 0.03, m, c, label=m)
        for b, v in zip(bars, values[m]):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color=c["ink"], fontsize=8.5)
    ax.set_xticks(x, settings)
    ax.tick_params(axis="x", colors=c["ink"])
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("task success (%)")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- environment diagrams: a root and two branches
def two_branch_diagram(c, out, root, up, down, up_label, down_label, middle, right):
    fig, (ax,) = new_fig(c, w=7.6, h=2.9)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, text, w, h=0.95):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="none", ec=c["ink2"], lw=1.1))
        ax.text(x, y, text, ha="center", va="center", color=c["ink"], fontsize=11, linespacing=1.4)

    def arrow(x0, y0, x1, y1, text, dy):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c["ink2"], lw=1.3, shrinkA=0,
                                                            shrinkB=0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="center", color=c["ink"], fontsize=10.5)

    box(1.35, 2.0, root, w=2.1)
    box(7.45, 3.15, up, w=4.4)
    box(7.45, 0.85, down, w=4.4)
    arrow(2.45, 2.2, 5.2, 3.05, up_label, 0.35)
    arrow(2.45, 1.8, 5.2, 0.95, down_label, -0.35)
    ax.text(3.85, 2.0, middle, ha="center", va="center", color=c["ink2"], fontsize=9.5, linespacing=1.3)
    ax.text(7.45, 2.0, right, ha="center", va="center", color=c["ink2"], fontsize=9.5, linespacing=1.3)
    fig.tight_layout(pad=0.2)
    save(fig, out)


DIAGRAMS = {
    "env": dict(root="root\nhides z", up="8 steps where z is visible", down="8 steps where z is still hidden",
                up_label="action 1", down_label="action 0",
                middle="either action matches\nthe expert half the time", right="the expert plays z at every step"),
    "env-hard": dict(root="root\nhides z", up="z = 1: easy corridor\nz = 0: recoverable corridor",
                     down="z = 0: easy corridor\nz = 1: hard corridor (coin flips)", up_label="action 1",
                     down_label="action 0", middle="either action matches\nthe expert half the time",
                     right="downstream, the expert plays 0\nexcept in the hard corridor"),
    "env-distill": dict(root="root\n(nothing hidden)", up="other path: teacher plays 0\neasy for any student",
                        down="teacher's path: plays parity of t\n1, 0, 1, 0, ...  needs degree 7",
                        up_label="action 1", down_label="action 0",
                        middle="the teacher\nplays 0 here", right="the student is a degree-k polynomial in t"),
}


# ---------------------------------------------------------------- the root decision during training
def line(ax, x, y, m, c, **kw):
    col, ls, mk, _ = STYLE[m]
    return ax.plot(x, y, color=c[col], ls=ls, lw=LW, label=m, **kw)


def fig_root(c, runs, env, out):
    fig, (ax,) = new_fig(c, w=7.6, h=3.3)
    style(ax, c)
    for m in METHODS:
        rs = [r for r in runs if r["env"] == env and r["method"] == m]
        hist = np.array([[0.5] + r["history"] for r in rs])
        x = np.arange(hist.shape[1]) * rs[0]["episodes_per_iter"] / 1000
        if m in ("APPO", "AGRPO", "DAgger"):
            ax.fill_between(x, hist.min(0), hist.max(0), color=c[STYLE[m][0]], alpha=0.2, lw=0)
        line(ax, x, hist.mean(0), m, c)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 370)
    ax.set_xlabel("simulated episodes (thousands; every method gets 369)")
    ax.set_ylabel({"reveal": "P(revealing action)", "hard": "P(recoverable side)"}[env] + "\n(mean over seeds)")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- errors per episode, information reveal
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
    plt.rcParams["hatch.linewidth"] = 1.0
    k = np.arange(H + 2)
    width = 0.2
    for i, m in enumerate(SUMMARY_METHODS):
        d = np.mean([error_dist(r["policy"]) for r in runs if r["env"] == "reveal" and r["method"] == m], 0)
        bar(ax, k + (i - 1.5) * width, d, width - 0.03, m, c, label=m)
    ax.axvspan(-0.5, 2.5, color=c["grid"], alpha=0.45, lw=0, zorder=0)
    ax.text(1.0, 0.93, "success: ≤ 2 errors", ha="center", color=c["ink2"], fontsize=9.5)
    ax.set_xticks(k)
    ax.set_xlim(-0.5, H + 1.5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("expert disagreements in the episode (of 9 decisions)")
    ax.set_ylabel("fraction of episodes")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="upper right")
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- distillation, deterministic teacher
def fig_distill(c, data, out):
    fig, (ax,) = new_fig(c, w=7.6, h=3.4)
    style(ax, c)
    ks = sorted(int(k) for k in data["summary"])
    ax.step(ks, [np.ceil((7 - k) / 2) for k in ks], where="mid", color=c["ref"], lw=1.1, ls=(0, (1, 2)))
    ax.annotate("dotted: fewest possible\nif it copies the teacher", (2.6, 1.3), color=c["ink2"], fontsize=9)
    ax.axhline(1, color=c["ref"], lw=1.0, ls=(0, (4, 3)))
    ax.annotate("leaving the teacher's path: 1 error", (0.05, 0.7), color=c["ink2"], fontsize=9)
    for m in METHODS:
        col, ls, mk, _ = STYLE[m]
        ax.plot(ks, [data["summary"][str(k)][m]["errors"][0] for k in ks], color=c[col], ls=ls, lw=LW, marker=mk,
                ms=4.5, label=m)
    ax.set_xticks(ks)
    ax.set_ylim(-0.15, 4.4)
    ax.set_xlabel("student size: polynomial degree k (the teacher needs 7)")
    ax.set_ylabel("disagreements per episode")
    ax.legend(frameon=False, labelcolor=c["ink"], fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2)
    fig.tight_layout()
    save(fig, out)


# ---------------------------------------------------------------- distillation, stochastic teacher
SOFT = [("off-policy KD", "off-policy KD", "DAgger", (0, (1, 1.5)), "v"),
        ("forward KL", "on-policy forward KL", "DAgger", "-", "o"),
        ("JSD", "on-policy JSD", "DAgger", (0, (6, 2)), "D"),
        ("reverse KL (γ=0)", "reverse KL, discount 0", "DAgger", (0, (4, 2)), "s"),
        ("reverse KL", "reverse KL with returns", "PPO", (0, (4, 2)), "s"),
        ("APPO", "APPO", "PPO", "-", "o")]


def fig_distill_soft(c, data, out):
    fig, axes = new_fig(c, w=7.6, h=4.3, ncols=2)
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
    axes[0].annotate("teacher: 0.1", (0.0, 0.135), color=c["ink2"], fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 4.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=c["ink"], fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, out)


if __name__ == "__main__":
    runs = json.load(open("results/runs.json"))
    distill = json.load(open("results/distill.json"))
    distill_soft = json.load(open("results/distill_stochastic.json"))
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for f in out.glob("*.svg"):
        f.unlink()
    for theme, c in THEMES.items():
        fig_summary(c, runs, distill, out / f"summary-{theme}.svg")
        for name, kw in DIAGRAMS.items():
            two_branch_diagram(c, out / f"{name}-{theme}.svg", **kw)
        fig_root(c, runs, "reveal", out / f"root-{theme}.svg")
        fig_root(c, runs, "hard", out / f"root-hard-{theme}.svg")
        fig_errors(c, runs, out / f"errors-{theme}.svg")
        fig_distill(c, distill, out / f"distill-{theme}.svg")
        fig_distill_soft(c, distill_soft, out / f"distill-soft-{theme}.svg")
    print("wrote", sorted(p.name for p in out.glob("*.svg")))
