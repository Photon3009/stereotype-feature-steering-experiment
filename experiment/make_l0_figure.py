"""Render the 'one SAE, three readings' L0 diagnostic figure for the post.

Log-scale gauge of L0 (active features out of 32,768) for the three states we
observed: raw inputs (dead, L0=3), invalid catalog norm applied (saturated,
L0=30,688 from saestats_neutral.json), reconstructed wikitext norm (alive,
L0~6,400 from find_features.py).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK1, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
CRIT, WARN, GOOD = "#d03b3b", "#ec835a", "#0ca30c"

ROWS = [  # (label, sublabel, L0, color, status, note)
    ("Raw activations",
     "norm file missing → no standardization applied", 3, CRIT, "DEAD",
     "0.009% of features fire — the SAE is silent"),
    ("Invalid norm applied",
     "the corrupted catalog file, as shipped", 30688, WARN, "SATURATED",
     "94% of features fire — a sparse autoencoder with no sparsity"),
    ("Reconstructed norm",
     "per-dim μ/σ recomputed from 22,503 wikitext tokens", 6400, GOOD, "ALIVE",
     "features become interpretable — and, downstream, causal"),
]
TOTAL = 32768
HEALTHY = (100, 1000)

fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=115)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.27, right=0.955, top=0.78, bottom=0.17)

ax.set_xscale("log")
ax.set_xlim(1, TOTAL * 1.25)
ax.set_ylim(-0.62, 2.72)
ax.invert_yaxis()

# healthy band
ax.axvspan(*HEALTHY, color=GOOD, alpha=0.10, zorder=0)
ax.text((HEALTHY[0] * HEALTHY[1]) ** 0.5, -0.52, "healthy zone\n(hundreds)",
        ha="center", va="top", fontsize=8.5, color="#006300", fontweight="bold",
        linespacing=1.3)
# total line
ax.axvline(TOTAL, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.text(TOTAL, -0.52, "all 32,768\nfeatures", ha="center", va="top",
        fontsize=8.5, color=MUTED, linespacing=1.3)

for y, (label, sub, l0, color, status, note) in enumerate(ROWS):
    # track
    ax.plot([1, TOTAL], [y, y], color=GRID, lw=7, solid_capstyle="round", zorder=1)
    # value bar
    ax.plot([1, l0], [y, y], color=color, lw=7, solid_capstyle="round", zorder=2)
    ax.plot([l0], [y], "o", color=color, ms=13, mec="white", mew=2, zorder=3)
    # value + status above the dot; align so it never runs off the axis
    ha = "center" if l0 < 15000 else "right"
    ax.annotate(f"L0 = {l0:,}  ·  {status}", (l0, y), xytext=(0, 14),
                textcoords="offset points", ha=ha, va="bottom", fontsize=11,
                fontweight="bold", color=color, family="monospace")
    # left labels
    ax.text(-0.015, y, label + "\n", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=11, fontweight="bold", color=INK1)
    ax.text(-0.015, y + 0.13, sub, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.8, color=MUTED, style="italic")
    # note below the row, left-anchored inside the track
    ax.text(1.3, y + 0.36, note, fontsize=8.3, color=INK2, ha="left", va="center")

ax.set_yticks([])
ax.set_xticks([1, 10, 100, 1000, 10000, 32768])
ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "32,768"], fontsize=9)
ax.tick_params(colors=INK2, length=0)
ax.set_xlabel("active features at the final token  (log scale)", fontsize=9.5, color=INK2)
for s in ax.spines.values():
    s.set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)

fig.suptitle("One SAE, three readings", x=0.045, ha="left",
             fontsize=16, fontweight="bold", color=INK1)
ax.set_title("The same layer-8 SAE on the same 12 prompts — the only difference is the normalization file  ·  llama-3.2-1b",
             loc="left", fontsize=9.5, color=INK2, pad=26, x=-0.36)

fig.savefig("experiment/results/fig_l0_diagnostics.png", facecolor=SURFACE, bbox_inches="tight")
print("saved experiment/results/fig_l0_diagnostics.png")
