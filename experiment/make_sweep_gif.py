"""Render an animated GIF of the f32258 steering sweep for the LessWrong post.

Three panels (male-stereotyped, female-stereotyped, context-override prompts).
A vertical marker sweeps strength -6 -> +6 with live P(' he')/P(' she') readouts,
holding briefly at -6, 0, and +6. Data = mean over 12 prompts per group from
results/steer_sweep.json.
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HE, SHE = "#2a78d6", "#eb6834"
STRENGTHS = [-6, -3, -1, 0, 1, 3, 6]

rows = json.load(open("experiment/results/steer_sweep.json"))
agg = defaultdict(lambda: [0.0, 0.0, 0])
for r in rows:
    if r["feature"] != "he_stereo_f32258":
        continue
    k = (r["group"], r["strength"])
    agg[k][0] += r["p_he"]; agg[k][1] += r["p_she"]; agg[k][2] += 1

def series(group):
    he = [agg[(group, s)][0] / agg[(group, s)][2] for s in STRENGTHS]
    she = [agg[(group, s)][1] / agg[(group, s)][2] for s in STRENGTHS]
    return np.array(he), np.array(she)

PANELS = [
    ("Male-stereotyped\n“The mechanic said that …”", series("male")),
    ("Female-stereotyped\n“The nurse said that …”", series("female")),
    ("Context-override\n“The mechanic tied her hair back …”", series("D_female")),
]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), dpi=110)
fig.subplots_adjust(left=0.06, right=0.955, top=0.74, bottom=0.16, wspace=0.32)
sup = fig.suptitle("", fontsize=13, fontweight="bold", y=0.97)

movers = []
for ax, (title, (he, she)) in zip(axes, PANELS):
    ax.plot(STRENGTHS, he, color=HE, lw=2, zorder=3)
    ax.plot(STRENGTHS, she, color=SHE, lw=2, zorder=3)
    ax.plot(STRENGTHS, he, "o", color=HE, ms=3.5, zorder=4)
    ax.plot(STRENGTHS, she, "o", color=SHE, ms=3.5, zorder=4)
    ymax = max(he.max(), she.max()) * 1.28
    ax.set_ylim(0, ymax); ax.set_xlim(-6.4, 6.4)
    ax.axvline(0, color="#c3c2b7", lw=1, ls=(0, (2, 3)))
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("steering strength", fontsize=9, color="#52514e")
    ax.tick_params(labelsize=8, colors="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", lw=0.7)
    ax.text(6.35, he[-1], " he", color=HE, fontsize=9, fontweight="bold", va="center")
    ax.text(6.35, she[-1], " she", color=SHE, fontsize=9, fontweight="bold", va="center")

    vline = ax.axvline(-6, color="#0b0b0b", lw=1, alpha=0.45, zorder=5)
    dhe, = ax.plot([], [], "o", color=HE, ms=8, mec="white", mew=1.6, zorder=6)
    dshe, = ax.plot([], [], "o", color=SHE, ms=8, mec="white", mew=1.6, zorder=6)
    txt = ax.text(0.03, 0.965, "", transform=ax.transAxes, fontsize=8.5,
                  va="top", family="monospace")
    movers.append((vline, dhe, dshe, txt, he, she))

axes[0].set_ylabel("mean next-token probability", fontsize=9, color="#52514e")

def interp(vals, s):
    return float(np.interp(s, STRENGTHS, vals))

# sweep path: hold at -6, sweep to 0, hold, sweep to +6, hold
def build_path():
    path, hold = [], 14
    path += [-6.0] * hold
    path += list(np.linspace(-6, 0, 36))
    path += [0.0] * hold
    path += list(np.linspace(0, 6, 36))
    path += [6.0] * hold
    return path

PATH = build_path()

def frame(i):
    s = PATH[i]
    sup.set_text(f"Steering f32258 (male-stereotype feature), Llama-3.2-1B layer 8"
                 f"   •   strength = {s:+.1f}")
    for vline, dhe, dshe, txt, he, she in movers:
        vh, vs = interp(he, s), interp(she, s)
        vline.set_xdata([s, s])
        dhe.set_data([s], [vh]); dshe.set_data([s], [vs])
        ratio = vh / max(vs, 1e-6)
        fav = f"he {ratio:.0f}:1" if ratio >= 1.5 else (
              f"she {1/max(ratio,1e-6):.0f}:1" if ratio <= 1/1.5 else "≈ even")
        txt.set_text(f"P(he)={vh:.3f}  P(she)={vs:.3f}  {fav}")
    return []

anim = FuncAnimation(fig, frame, frames=len(PATH), blit=False)
anim.save("experiment/results/fig3_sweep_animation.gif", writer=PillowWriter(fps=18))
print("saved experiment/results/fig3_sweep_animation.gif")
