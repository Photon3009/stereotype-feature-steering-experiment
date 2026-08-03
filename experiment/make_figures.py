"""Blog figures: dose-response curves for the two stereotype features.
Palette: validated 2-slot categorical (he=blue #2a78d6, she=orange #eb6834), light surface.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
HE, SHE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

def mean_by(rows, key_strength, group, prob):
    sel = [r for r in rows if r["strength"] == key_strength and r["group"] == group]
    return sum(r[prob] for r in sel) / len(sel)

def panel(ax, rows, group, title, strengths):
    he = [mean_by(rows, s, group, "p_he") for s in strengths]
    she = [mean_by(rows, s, group, "p_she") for s in strengths]
    ax.plot(strengths, he, color=HE, lw=2, marker="o", ms=5, label="P(' he')")
    ax.plot(strengths, she, color=SHE, lw=2, marker="o", ms=5, label="P(' she')")
    ax.annotate("he", (strengths[-1], he[-1]), xytext=(5, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center")
    ax.annotate("she", (strengths[-1], she[-1]), xytext=(5, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center")
    ax.axvline(0, color=BASELINE, lw=1, ls=":")
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("steering strength", fontsize=9)
    ax.set_ylim(bottom=0)

# --- Figure 1: llama f32258 ---
sweep = json.load(open("experiment/results/steer_sweep.json"))
rows = [r for r in sweep if r["feature"] == "he_stereo_f32258"]
S = sorted({r["strength"] for r in rows})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
panel(axes[0], rows, "male", "Male-stereotyped occupations\n(“The mechanic said that …”)", S)
panel(axes[1], rows, "female", "Female-stereotyped occupations\n(“The nurse said that …”)", S)
panel(axes[2], rows, "D_female", "Context-override\n(“The mechanic tied her hair back …”)", S)
axes[0].set_ylabel("mean next-token probability", fontsize=9)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Steering feature f32258 (male-stereotype) in Llama-3.2-1B-Instruct — layer 8 SAE",
             fontsize=11, color=INK, y=1.06)
fig.tight_layout()
fig.savefig("experiment/results/fig1_llama_dose_response.png", dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
print("saved fig1")

# --- Figure 2: lfm f9619 ---
rows2 = json.load(open("experiment/results/lfm_l9_f9619_sweep.json"))
S2 = sorted({r["strength"] for r in rows2})
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
panel(axes[0], rows2, "female", "Female-stereotyped occupations", S2)
panel(axes[1], rows2, "male", "Male-stereotyped occupations", S2)
axes[0].set_ylabel("mean next-token probability", fontsize=9)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Steering feature f9619 (female-stereotype) in LFM2.5-230M — layer 9 SAE",
             fontsize=11, color=INK, y=1.06)
fig.tight_layout()
fig.savefig("experiment/results/fig2_lfm_dose_response.png", dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
print("saved fig2")
