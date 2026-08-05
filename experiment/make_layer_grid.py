"""Layer-grid view of one prediction forming (redesign of the Aquin dashboard
graph): 16 layer nodes colored by logit-lens confidence, red rings sized by
causal (ablation) drop, SAE read point marked. Data: results/trace_B07.json.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

INK1, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
GREEN, YELLOW, GRAY, RED = "#0ca30c", "#eda100", "#e7e6e0", "#d03b3b"
ORANGE = "#eb6834"

d = json.load(open("experiment/results/trace_B07.json"))
lens = {e["layer"]: e["top_tokens"][0] for e in d["logit_lens"]}
drop = {t["layer"]: t["drop"] for t in d["trace_results"]}

def status(p):
    if p >= 0.9: return GREEN
    if p >= 0.2: return YELLOW
    return GRAY

fig, ax = plt.subplots(figsize=(12.6, 6.8), dpi=110)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(-0.7, 6.55)
ax.set_ylim(-1.35, 4.05)
ax.set_aspect("equal")
ax.set_axis_off()

COLS = [1.3, 2.5, 3.7, 4.9]           # layer group columns
ROWS = [3.0, 2.0, 1.0, 0.0]           # 4 layers per column, top to bottom
IN_X, OUT_X = 0.0, 5.9
R = 0.245

def pos(layer):
    return COLS[layer // 4], ROWS[layer % 4]

# ---- edges (light, behind everything) ----
in_nodes = [(IN_X, 2.5), (IN_X, 1.5), (IN_X, 0.5)]
for ix, iy in in_nodes:
    for l in range(4):
        x, y = pos(l)
        ax.plot([ix, x], [iy, y], color=GRID, lw=0.6, zorder=1)
for l in range(12):
    x1, y1 = pos(l)
    for m in range(4):
        x2, y2 = pos((l // 4 + 1) * 4 + m)
        ax.plot([x1, x2], [y1, y2], color=GRID, lw=0.6, zorder=1)
out_y = 1.5
for l in range(12, 16):
    x1, y1 = pos(l)
    ax.plot([x1, OUT_X], [y1, out_y], color=GRID, lw=0.6, zorder=1)

# ---- input / output nodes ----
for (ix, iy), lab in zip(in_nodes, ["tok", "pos", "emb"]):
    ax.add_patch(Circle((ix, iy), R * 0.9, fc=INK1, ec="none", zorder=3))
    ax.text(ix, iy, lab, ha="center", va="center", color="white",
            fontsize=9, family="monospace", zorder=4)
ax.add_patch(Circle((OUT_X, out_y), R * 0.95, fc=GREEN, ec="none", zorder=3))
ax.text(OUT_X, out_y, "out", ha="center", va="center", color="white",
        fontsize=9, family="monospace", fontweight="bold", zorder=4)
ax.text(OUT_X, out_y - R - 0.16, "‘The’ · p=0.40", ha="center", va="top",
        fontsize=8, color=INK2, family="monospace")

# ---- layer nodes ----
for l in range(16):
    x, y = pos(l)
    p = lens[l]["prob"]
    tok = lens[l]["token"].strip()[:6]
    fc = status(p)
    # causal-drop ring
    dr = drop.get(l, 0.0)
    if dr > 0.005:
        lw = 1.2 + dr * 22          # 0.0088 -> ~1.4 ; 0.17 -> ~5
        alpha = 0.45 if dr <= 0.03 else 1.0
        ax.add_patch(Circle((x, y), R + 0.055, fc="none", ec=RED,
                            lw=lw, alpha=alpha, zorder=2))
    ax.add_patch(Circle((x, y), R, fc=fc, ec="none", zorder=3))
    tcol = "white" if fc == GREEN else (INK1 if fc == YELLOW else MUTED)
    ax.text(x, y, f"L{l}", ha="center", va="center", color=tcol,
            fontsize=9.5, family="monospace", fontweight="bold", zorder=4)
    yoff = 0.24 if l == 8 else 0.13   # clear the dashed SAE ring at L8
    ax.text(x, y - R - yoff, f"‘{tok}’", ha="center", va="top",
            fontsize=7.6, color=MUTED, family="monospace", zorder=4)

# ---- SAE read marker at L8 ----
x8, y8 = pos(8)
ax.add_patch(Circle((x8, y8), R + 0.14, fc="none", ec=ORANGE, lw=1.4,
                    ls=(0, (4, 3)), zorder=2))
ax.annotate("SAE reads here", (x8 - R - 0.16, y8 + R + 0.10),
            xytext=(x8 - 1.02, y8 + 0.72), fontsize=8.5, color=ORANGE,
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))

# ---- column labels ----
for x, lab in zip([IN_X] + COLS + [OUT_X], ["in", "L0–3", "L4–7", "L8–11", "L12–15", "out"]):
    ax.text(x, -0.75, lab, ha="center", va="center", fontsize=8.5,
            color=MUTED, family="monospace")

# ---- legend ----
items = [(0.30, GREEN, "none", "confident (p ≥ 0.9)"),
         (2.10, YELLOW, "none", "emerging (p ≥ 0.2)"),
         (3.80, GRAY, "#c3c2b7", "noise"),
         (4.60, "none", RED, "ring = causal drop (ablation)")]
for lx, fc, ec, lab in items:
    lw = 2.2 if fc == "none" else 0.6
    ax.add_patch(Circle((lx, -1.15), 0.075, fc=fc, ec=ec, lw=lw))
    ax.text(lx + 0.15, -1.15, lab, ha="left", va="center", fontsize=8.5, color=INK2)

ax.set_title("How the prediction forms — logit-lens top token per layer, hairdresser probe (llama-3.2-1b)",
             loc="left", fontsize=12, fontweight="bold", color=INK1, pad=14, x=0.02)

fig.savefig("experiment/results/fig_layer_grid.png", facecolor=SURFACE, bbox_inches="tight")
print("saved experiment/results/fig_layer_grid.png")
