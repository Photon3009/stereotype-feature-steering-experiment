"""Generate every figure in the paper as PDF (vector, embeds cleanly in LaTeX).

Run after baseline_compare.py, override_sweep.py, specificity.py, sae_quality.py and
lfm_search.py. Each figure is skipped with a message if its input is missing, so this
can be run repeatedly while experiments land.

Outputs paper/figures/*.pdf
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("experiment/results")
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "figure.constrained_layout.use": True, "figure.constrained_layout.w_pad": 0.06,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 3.5,
})

# Okabe-Ito, colour-blind safe
C = {"sae": "#0072B2", "dim": "#D55E00", "unembed": "#009E73", "random": "#999999",
     "male": "#0072B2", "female": "#D55E00", "neutral": "#666666",
     "they": "#CC79A7", "llama": "#0072B2", "lfm": "#D55E00"}
RNG = np.random.default_rng(0)


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def bias(rows):
    return np.array([math.log(r["p_he"]) - math.log(r["p_she"]) for r in rows])


def mean_ci(x):
    """Same seeded bootstrap as the tables (paperstats.ci), so figure and table agree."""
    from paperstats import ci
    x = np.asarray(x)
    if len(x) < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    lo, hi = ci(x)
    return float(x.mean()), lo, hi


def curve(rows, group, key="strength", value=bias):
    """-> (strengths, mean, lo, hi) for one group."""
    ss = sorted({r[key] for r in rows})
    out = []
    for s in ss:
        sel = [r for r in rows if r[key] == s and r["group"] == group]
        if sel:
            out.append((s,) + mean_ci(value(sel)))
    if not out:
        return None
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def band(ax, c, color, label):
    if c is None:
        return
    s, m, lo, hi = c
    ax.plot(s, m, "-o", color=color, label=label)
    ax.fill_between(s, lo, hi, color=color, alpha=0.18, linewidth=0)


# ===================================================== Fig 1: dose-response + CI
sw = load("baseline_sweep.json")
if sw:
    rows = [r for r in sw if r["method"] in ("sae_f32258", "baseline")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))

    ax = axes[0]
    for g, lab in (("male", "male-stereotyped"), ("female", "female-stereotyped"),
                   ("neutral", "neutral")):
        band(ax, curve(rows, g), C[g], lab)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axhspan(-0.3, 0.3, color="k", alpha=0.06, linewidth=0)
    ax.set_xlabel("steering strength $s$")
    ax.set_ylabel(r"pronoun log-odds $b$")
    ax.set_title("(a) bias, with 95% CI")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for g in ("male", "female"):
        c = curve(rows, g, value=lambda rs: np.array([r["p_he"] for r in rs]))
        s, m, lo, hi = c
        ax.plot(s, m, "-o", color=C[g], label=f"{g}: P(' he')")
        c2 = curve(rows, g, value=lambda rs: np.array([r["p_she"] for r in rs]))
        ax.plot(c2[0], c2[1], "--s", color=C[g], alpha=0.6, label=f"{g}: P(' she')")
    ax.set_xlabel("steering strength $s$")
    ax.set_ylabel("probability")
    ax.set_title("(b) raw probabilities")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[2]
    for g in ("male", "female", "neutral"):
        c = curve(rows, g, value=lambda rs: np.array([r["p_they"] for r in rs]))
        if c:
            ax.plot(c[0], c[1], "-o", color=C[g], label=g)
    ax.set_xlabel("steering strength $s$")
    ax.set_ylabel("P(' they')")
    ax.set_title("(c) mass on ' they'")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig_dose_ci.pdf")
    plt.close(fig)
    print("wrote fig_dose_ci.pdf")
else:
    print("skip fig_dose_ci: baseline_sweep.json missing")

# ===================================================== Fig 2: capability frontier
# (a) male-group bias against WikiText cost -- the metric the submitted version used.
# (b) the occupation gap b(male) - b(female) against cost: the stereotype itself. A
#     direction that shifts every occupation toward ' she' wins (a) without moving (b).
cap = load("capability_baselines.json")
gf = load("gap_frontier.json")
if sw and cap:
    base_loss = [c for c in cap if c["method"] == "baseline"][0]["wikitext_loss"]
    pts_rows = list(sw) + (gf["rows"] if gf else [])
    pts_cap = list(cap) + ([c for c in gf["capability"] if c["method"] != "baseline"] if gf else [])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    labels = {"sae_f32258": ("SAE feature f32258", C["sae"]),
              "diff_in_means": ("difference-in-means", C["dim"]),
              "unembed_he_she": ("unembedding he$-$she", C["unembed"]),
              "random_s0": ("random direction", C["random"])}

    def grp(meth, s, g):
        return [r for r in pts_rows if r["method"] == meth and r["strength"] == s
                and r["group"] == g and r.get("set", "tpl") == "tpl"]

    def frontier(ax, value, ylabel, title, ref):
        for meth, (lab, col) in labels.items():
            pts = []
            for c in pts_cap:
                if c["method"] != meth:
                    continue
                m, f = grp(meth, c["strength"], "male"), grp(meth, c["strength"], "female")
                if not m or not f:
                    continue
                pts.append((100 * (c["wikitext_loss"] - base_loss) / base_loss,
                            value(bias(m), bias(f)), c["strength"]))
            if not pts:
                continue
            pts.sort()
            x, y, st = zip(*pts)
            ax.plot(x, y, "-o", color=col, label=lab)
            for xi, yi, si in pts:
                ax.annotate(f"{si:+.0f}", (xi, yi), textcoords="offset points",
                            xytext=(3, 3), fontsize=5.5, color=col)
        m0, f0 = bias(grp("baseline", 0.0, "male")), bias(grp("baseline", 0.0, "female"))
        ax.axhline(value(m0, f0), color="k", lw=0.8, ls=":", label="unsteered")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("WikiText loss increase (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    frontier(axes[0], lambda m, f: m.mean(), r"bias $b$, male-stereotyped",
             "(a) male-group bias vs. capability", None)
    axes[0].legend(frameon=False, fontsize=6)
    frontier(axes[1], lambda m, f: m.mean() - f.mean(), r"gap $b_{male} - b_{female}$",
             "(b) occupation gap vs. capability", None)
    fig.savefig(FIG / "fig_frontier.pdf")
    plt.close(fig)
    print("wrote fig_frontier.pdf")
else:
    print("skip fig_frontier: inputs missing")

# ===================================================== Fig 3: cross-model mirror
llama = load("steer_sweep.json")
lfm = load("lfm_l9_f9619_sweep.json")
if llama and lfm:
    lr = [r for r in llama if r["feature"] == "he_stereo_f32258"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, rows, title in ((axes[0], lr, "Llama-3.2-1B-Instruct, f32258\n(male-context feature)"),
                            (axes[1], lfm, "LFM2.5-230M, f9619\n(female-context feature)")):
        for g, lab in (("male", "male-stereotyped"), ("female", "female-stereotyped")):
            band(ax, curve(rows, g), C[g], lab)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("steering strength $s$")
        ax.set_title(title)
    axes[0].set_ylabel(r"pronoun log-odds $b$")
    axes[0].legend(frameon=False)
    fig.savefig(FIG / "fig_crossmodel.pdf")
    plt.close(fig)
    print("wrote fig_crossmodel.pdf")
else:
    print("skip fig_crossmodel: inputs missing")

# ===================================================== Fig 4: specificity
spec = load("specificity.json")
if spec:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))

    ax = axes[0]
    modes = ["all", "final_only", "except_final"]
    w = 0.26
    for i, g in enumerate(("male", "female")):
        for j, s in enumerate((-6.0, 6.0)):
            vals = []
            for m in modes:
                sel = [r for r in spec["position"] if r["mode"] == m
                       and r["strength"] == s and r["group"] == g]
                vals.append(bias(sel).mean() if sel else np.nan)
            ax.bar(np.arange(3) + (i * 2 + j - 1.5) * w, vals, w,
                   color=C[g], alpha=1.0 if s < 0 else 0.5,
                   label=f"{g}, s={s:+.0f}")
    b0 = [r for r in spec["position"] if r["strength"] == 0.0 and r["mode"] == "all"]
    unst = bias([r for r in b0 if r["group"] == "male"]).mean()
    ax.axhline(unst, color=C["male"], ls=":", lw=0.9, zorder=0)
    ax.annotate("unsteered (male)", (2.42, unst), fontsize=5.5, color=C["male"],
                ha="right", va="bottom")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["all pos.", "final only", "all but final"], fontsize=6)
    ax.set_ylabel(r"bias $b$")
    ax.set_title("(a) which positions matter")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=5.5,
              loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.2)

    ax = axes[1]
    for g in ("male", "female"):
        for s, ls in ((-6.0, "-o"), (6.0, "--s")):
            c = curve([r for r in spec["layer"] if r["strength"] == s], g, key="layer")
            if c:
                ax.plot(c[0], c[1], ls, color=C[g], alpha=1.0 if s < 0 else 0.55,
                        label=f"{g}, s={s:+.0f}")
    ax.axvline(8, color="k", lw=0.8, ls=":")
    ax.text(8.15, ax.get_ylim()[1] * 0.9, "SAE layer", fontsize=6)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("injection layer")
    ax.set_ylabel(r"bias $b$")
    ax.set_title("(b) layer specificity")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.55)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=5.5,
              loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.2)

    ax = axes[2]
    prefs = ["none", "neutral_instruction", "strong_instruction"]
    for i, g in enumerate(("male", "female")):
        vals = [bias([r for r in spec["prompting"] if r["prefix"] == p
                      and r["group"] == g]).mean() for p in prefs]
        ax.bar(np.arange(3) + (i - 0.5) * 0.35, vals, 0.35, color=C[g], label=g)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["no prompt", "neutral\ninstruction", "strong\ninstruction"], fontsize=6)
    ax.set_ylabel(r"bias $b$")
    ax.set_title("(c) prompting baseline")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=6,
              loc="upper right")
    fig.savefig(FIG / "fig_specificity.pdf")
    plt.close(fig)
    print("wrote fig_specificity.pdf")
else:
    print("skip fig_specificity: specificity.json missing")

# ===================================================== Fig 5: held-out + override
ho = load("heldout_sweep.json")
ov = load("override_sweep.json")
if ho or ov:
    ncol = (1 if ho else 0) + (1 if ov else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 2.4), squeeze=False)
    k = 0
    if ho:
        ax = axes[0][k]
        k += 1
        for meth, col, lab in (("sae_f32258", C["sae"], "SAE f32258"),
                               ("diff_in_means", C["dim"], "difference-in-means")):
            rows = [r for r in ho if r["method"] in (meth, "baseline")]
            for g, ls in (("male", "-o"), ("female", "--s")):
                c = curve(rows, g)
                if c:
                    ax.plot(c[0], c[1], ls, color=col, alpha=1.0 if g == "male" else 0.55,
                            label=f"{lab}, {g}")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("steering strength $s$")
        ax.set_ylabel(r"bias $b$")
        ax.set_title("held-out occupations \\& templates")
        ax.legend(frameon=False, fontsize=5.5)
    if ov:
        ax = axes[0][k]
        for g, col, lab in (("D_female", C["female"], "explicit ``her''"),
                            ("D_male", C["male"], "explicit ``his''")):
            rows = [r for r in ov if r["method"] == "sae_f32258"]
            c = curve(rows, g)
            if c:
                band(ax, c, col, lab)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("steering strength $s$")
        ax.set_ylabel(r"bias $b$")
        ax.set_title("context override ($n=12$ per side)")
        ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "fig_heldout_override.pdf")
    plt.close(fig)
    print("wrote fig_heldout_override.pdf")
else:
    print("skip fig_heldout_override: inputs missing")

# ===================================================== Fig 6: pre-flight
q = load("sae_quality.json")
if q:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax = axes[0]
    states = ["raw\ninputs", "corrupt\nfile", "reconstructed"]
    l0 = [3, 30688, q["mean_l0"]]
    ax.bar(range(3), l0, color=["#999999", "#D55E00", C["sae"]])
    ax.axhline(q["n_features"], color="k", ls=":", lw=0.8, zorder=0)
    # log axis: set both limits explicitly, or the headroom clips the bars
    ax.set_ylim(1, q["n_features"] * 8)
    ax.text(2.45, q["n_features"] * 1.5, "all features", fontsize=6, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(states, fontsize=6)
    ax.set_ylabel(r"mean $L_0$ (log scale)")
    ax.set_title("(a) same SAE, three norm files")

    ax = axes[1]
    f = np.array(q["per_passage_fvu"])
    ax.hist(f, bins=25, color=C["sae"])
    ax.axvline(np.median(f), color="k", ls="--", lw=0.9)
    ax.text(np.median(f) + 0.02, ax.get_ylim()[1] * 0.85,
            f"median {np.median(f):.2f}", fontsize=6)
    ax.set_xlabel("FVU per passage")
    ax.set_ylabel("passages")
    ax.set_title("(b) reconstruction quality")
    ax.grid(axis="x", alpha=0.15)

    ax = axes[2]
    names = list(q["template_activations"]["male"].keys())
    xs = np.arange(len(names))
    for i, g in enumerate(("male", "female", "neutral")):
        vals = [np.mean(q["template_activations"][g][n]) for n in names]
        ax.bar(xs + (i - 1) * 0.27, vals, 0.27, color=C[g], label=g)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("mean activation")
    ax.set_title("(c) selectivity of candidates")
    ax.annotate(r"$\approx$0", (len(names) - 1, 0.012), fontsize=5.5, ha="center",
                va="bottom", color="#555555")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "fig_preflight.pdf")
    plt.close(fig)
    print("wrote fig_preflight.pdf")
else:
    print("skip fig_preflight: sae_quality.json missing")

# ===================================================== Fig 7: LFM two-sided search
ls = load("lfm_search.json")
if ls:
    fig, axes = plt.subplots(1, len(ls["layers"]), figsize=(3.6 * len(ls["layers"]), 2.4),
                             squeeze=False)
    for ax, (layer, d) in zip(axes[0], sorted(ls["layers"].items(), reverse=True)):
        for side, col in (("he", C["male"]), ("she", C["female"])):
            cands = d["candidates"].get(side, [])
            xs = np.arange(len(cands))
            vals = [abs(c.get("swing_male", 0)) for c in cands]
            ax.bar(xs + (0 if side == "he" else 0.4), vals, 0.4, color=col,
                   label=f"{side}-side candidates")
            for x, c in zip(xs, cands):
                ax.annotate(f"f{c['feature']}", (x + (0 if side == "he" else 0.4),
                                                 abs(c.get("swing_male", 0))),
                            rotation=90, fontsize=5, ha="center", va="bottom")
        ax.set_xlabel("candidate rank")
        ax.set_ylabel(r"$|b|$ swing, male-stereotyped")
        ax.set_title(f"LFM2.5-230M, layer {layer}")
        ax.legend(frameon=False, fontsize=6)
    fig.savefig(FIG / "fig_lfm_search.pdf")
    plt.close(fig)
    print("wrote fig_lfm_search.pdf")
else:
    print("skip fig_lfm_search: lfm_search.json missing")


# ===================================================== placeholders for pending figures
# so that main.tex always compiles while experiments are still running
EXPECTED = ["fig_dose_ci.pdf", "fig_frontier.pdf", "fig_crossmodel.pdf",
            "fig_specificity.pdf", "fig_heldout_override.pdf", "fig_preflight.pdf",
            "fig_lfm_search.pdf"]
for name in EXPECTED:
    if (FIG / name).exists():
        continue
    fig, ax = plt.subplots(figsize=(7.2, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, f"[{name} pending: its experiment has not finished]",
            ha="center", va="center", fontsize=9, color="#D55E00",
            bbox=dict(boxstyle="round,pad=0.6", fc="none", ec="#D55E00", lw=0.8))
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote placeholder {name}")
