"""Generate every numeric table in the paper as a LaTeX fragment under paper/tables/.

main.tex inputs these, so no number is ever transcribed by hand. Fragments are
written only when their input JSON exists, and each writes a one-line placeholder
otherwise so the document still compiles while experiments are running.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("experiment/results")
TAB = Path("paper/tables")
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
NBOOT = 20000


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def bias(rows):
    return np.array([math.log(r["p_he"]) - math.log(r["p_she"]) for r in rows])


def ci(x, n=NBOOT):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    d = x[idx].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def paired(b0, b1):
    b0, b1 = np.asarray(b0), np.asarray(b1)
    if b0.shape != b1.shape or b0.size == 0:
        return float("nan"), (float("nan"), float("nan")), 1.0, float("nan")
    d = b1 - b0
    if np.allclose(d, 0):
        return 0.0, (0.0, 0.0), 1.0, 0.0
    p = float(stats.wilcoxon(b1, b0, alternative="two-sided", zero_method="wilcox").pvalue)
    lo, hi = ci(d)
    return float(d.mean()), (lo, hi), p, float(d.mean() / (d.std(ddof=1) + 1e-12))


def fmt_p(p):
    if p >= 0.05:
        return "n.s."
    if p <= 5e-4:
        return r"$\le$4.9e-4"
    return f"{p:.1e}".replace("e-0", "e-")


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return sorted(out, key=lambda r: r.get("id", ""))


def write(name, body):
    body = body.rstrip("\n").rstrip()
    if body.endswith("\\\\"):
        body = body[:-2].rstrip()
    (TAB / f"{name}.tex").write_text(body)
    print(f"wrote tables/{name}.tex")


def placeholder(name, why):
    write(name, "\\textit{[%s]}\n" % why)


# ======================================================== dose-response (Llama SAE)
def dose_table(SHOWN_STRENGTHS, GROUPS, name_out, drop_dz=False):
    if True:
      base = {g: bias(sel(sw, method="baseline", group=g)) for g in ("male", "female", "neutral")}
      lines = []
      for g, lab in GROUPS:
          strengths = sorted({r["strength"] for r in sw if r["method"] == "sae_f32258"})
          shown = [s for s in strengths if s in SHOWN_STRENGTHS]
          first = True
          for s in [0.0] + shown:
              rows = sel(sw, method="baseline" if s == 0 else "sae_f32258", group=g,
                         strength=0.0 if s == 0 else s)
              if not rows:
                  continue
              b = bias(rows)
              lo, hi = ci(b)
              he = np.mean([r["p_he"] for r in rows])
              she = np.mean([r["p_she"] for r in rows])
              they = np.mean([r["p_they"] for r in rows])
              if s == 0:
                  extra = "--- & ---" if drop_dz else "--- & --- & ---"
              else:
                  md, (clo, chi), p, dz = paired(base[g], b)
                  extra = f"${md:+.2f}$ $[{clo:+.2f},{chi:+.2f}]$ & {fmt_p(p)}"
                  if not drop_dz:
                      extra += f" & ${dz:+.2f}$"
              name = lab if first else ""
              first = False
              lines.append(f"{name} & ${s:+.0f}$ & {he:.3f} & {she:.3f} & {they:.4f} & "
                           f"${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & {extra} \\\\")
          lines.append("\\midrule")
      write(name_out, "\n".join(lines[:-1]))



sw = load("baseline_sweep.json")
if sw:
    dose_table((-6, -3, -1, 3, 6),
               (("male", "male-stereo"), ("female", "female-stereo")), "tab_dose",
               drop_dz=True)
    dose_table((-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8),
               (("male", "male-stereo"), ("female", "female-stereo"), ("neutral", "neutral")),
               "tab_dose_full")
else:
    placeholder("tab_dose", "baseline\\_sweep.json not yet produced")
    placeholder("tab_dose_full", "baseline\\_sweep.json not yet produced")

# ======================================================== baselines / frontier
cap = load("capability_baselines.json")
if sw and cap:
    base_loss = [c for c in cap if c["method"] == "baseline"][0]["wikitext_loss"]
    base_qa = [c for c in cap if c["method"] == "baseline"][0]["qa"]
    b0 = bias(sel(sw, method="baseline", group="male"))
    lo0, hi0 = ci(b0)
    names = {"sae_f32258": "SAE feature \\ff{32258}", "diff_in_means": "difference-in-means",
             "unembed_he_she": "unembedding he$-$she", "random_s0": "random direction"}

    def baseline_table(keep, out_name):
        lines = [f"unsteered & --- & ${b0.mean():+.2f}$ $[{lo0:+.2f},{hi0:+.2f}]$ & --- & "
                 f"{base_loss:.3f} & --- & {base_qa}/10 \\\\", "\\midrule"]
        for meth, lab in names.items():
            first = True
            for c in sorted([c for c in cap if c["method"] == meth], key=lambda c: c["strength"]):
                if keep is not None and c["strength"] not in keep:
                    continue
                r = sel(sw, method=meth, group="male", strength=c["strength"])
                if not r:
                    continue
                b = bias(r)
                lo, hi = ci(b)
                md, _, _, _ = paired(b0, b)
                dloss = 100 * (c["wikitext_loss"] - base_loss) / base_loss
                qa = f"{c['qa']}/10" if c.get("qa") is not None else "---"
                lines.append(f"{lab if first else ''} & ${c['strength']:+.0f}$ & "
                             f"${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & ${md:+.2f}$ & "
                             f"{c['wikitext_loss']:.3f} & ${dloss:+.1f}\\%$ & {qa} \\\\")
                first = False
            if not first:
                lines.append("\\midrule")
        write(out_name, "\n".join(lines[:-1]))

    baseline_table({-6.0, -8.0}, "tab_baselines")
    baseline_table(None, "tab_baselines_full")
else:
    placeholder("tab_baselines", "capability\\_baselines.json not yet produced")
    placeholder("tab_baselines_full", "capability\\_baselines.json not yet produced")

# ======================================================== held-out
ho = load("heldout_sweep.json")
if ho:
    lines = []
    names = {"sae_f32258": "SAE \\ff{32258}", "diff_in_means": "difference-in-means",
             "unembed_he_she": "unembedding he$-$she"}
    for g, lab in (("male", "male-stereo"), ("female", "female-stereo")):
        b0 = bias(sel(ho, method="baseline", group=g))
        lo, hi = ci(b0)
        lines.append(f"{lab} & unsteered & ${b0.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & --- & --- \\\\")
        for meth, mlab in names.items():
            for s in sorted({r["strength"] for r in ho if r["method"] == meth}):
                r = sel(ho, method=meth, group=g, strength=s)
                if not r:
                    continue
                b = bias(r)
                lo, hi = ci(b)
                md, _, p, _ = paired(b0, b)
                lines.append(f" & {mlab}, $s={s:+.0f}$ & ${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & "
                             f"${md:+.2f}$ & {fmt_p(p)} \\\\")
        lines.append("\\midrule")
    write("tab_heldout", "\n".join(lines[:-1]))
else:
    placeholder("tab_heldout", "heldout\\_sweep.json not yet produced")

# ======================================================== context override
ov = load("override_sweep.json")
if ov and any(r["strength"] == 0.0 for r in ov):
    lines = []
    for g, lab in (("D_female", "explicit ``her''"), ("D_male", "explicit ``his''")):
        b0 = bias(sel(ov, method="sae_f32258", group=g, strength=0.0))
        strengths = sorted({r["strength"] for r in ov if r["method"] == "sae_f32258"})
        first = True
        for s in strengths:
            r = sel(ov, method="sae_f32258", group=g, strength=s)
            if not r:
                continue
            b = bias(r)
            lo, hi = ci(b)
            wins = sum((x["p_he"] > x["p_she"]) if g == "D_female" else (x["p_she"] > x["p_he"])
                       for x in r)
            if s == 0:
                extra = "--- & ---"
            else:
                md, _, p, _ = paired(b0, b)
                extra = f"${md:+.2f}$ & {fmt_p(p)}"
            lines.append(f"{lab if first else ''} & ${s:+.0f}$ & "
                         f"${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$ & {wins}/{len(r)} & {extra} \\\\")
            first = False
        lines.append("\\midrule")
    write("tab_override", "\n".join(lines[:-1]))
else:
    placeholder("tab_override", "override\\_sweep.json not yet complete")

# ======================================================== ablation
ab = load("ablation.json")
if ab:
    lines = []
    names = {"no_ablation": "unmodified", "projout_sae_f32258": "project out \\ff{32258}",
             "projout_diff_in_means": "project out difference-in-means"}
    for st, slab in (("tpl", "discovery templates"), ("heldout", "held-out occupations")):
        for meth, mlab in names.items():
            row = [f"{slab if meth == 'no_ablation' else ''} & {mlab}"]
            for g in ("male", "female"):
                r = sel(ab, method=meth, set=st, group=g)
                if not r:
                    row.append("---")
                    continue
                b = bias(r)
                lo, hi = ci(b)
                row.append(f"${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$")
            if meth != "no_ablation":
                r0 = sel(ab, method="no_ablation", set=st, group="male")
                r1 = sel(ab, method=meth, set=st, group="male")
                md, _, p, _ = paired(bias(r0), bias(r1))
                row.append(f"${md:+.2f}$ ({fmt_p(p)})")
            else:
                row.append("---")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    write("tab_ablation", "\n".join(lines[:-1]))
else:
    placeholder("tab_ablation", "ablation.json not yet produced")

# ======================================================== two-sided search, both models
llama_s = load("llama_search.json")
lfm_s = load("lfm_search.json")


def search_rows(label, cands_by_side, ranks=None, best_row=True):
    out = []
    firstline = True
    for side, slab in (("he", "he-side"), ("she", "she-side")):
        cs = cands_by_side.get(side, [])
        if not cs:
            continue
        pick = list(range(len(cs))) if ranks is None else [r for r in ranks if r < len(cs)]
        for k in pick:
            c = cs[k]
            out.append(f"{label if firstline else ''} & {slab} & {k+1} & "
                       f"\\ff{{{c['feature']}}} & {c['act_male']:.2f} & {c['act_female']:.2f} & "
                       f"${c.get('swing_male', float('nan')):+.2f}$ & "
                       f"${c.get('swing_female', float('nan')):+.2f}$ \\\\")
            firstline = False
        if best_row and len(cs) > 1:
            best = max(cs, key=lambda c: abs(c.get("swing_male", 0)))
            bi = cs.index(best) + 1
            out.append(f" & {slab} & \\textit{{best of {len(cs)}}} & \\ff{{{best['feature']}}} "
                       f"(rank {bi}) & {best['act_male']:.2f} & {best['act_female']:.2f} & "
                       f"$\\mathbf{{{best.get('swing_male', float('nan')):+.2f}}}$ & "
                       f"${best.get('swing_female', float('nan')):+.2f}$ \\\\")
    return out


if llama_s or lfm_s:
    for out_name, ranks, layers in (("tab_search", (0, 1), ("9",)),
                                    ("tab_search_full", None, None)):
        lines = []
        if llama_s:
            lines += search_rows("Llama-3.2-1B, L8", llama_s["candidates"], ranks)
            lines.append("\\midrule")
        if lfm_s:
            for layer, d in sorted(lfm_s["layers"].items(), reverse=True):
                if layers is not None and layer not in layers:
                    continue
                lines += search_rows(f"LFM2.5-230M, L{layer}", d["candidates"], ranks)
                lines.append("\\midrule")
        write(out_name, "\n".join(lines[:-1]))
else:
    placeholder("tab_search", "search JSONs not yet produced")
    placeholder("tab_search_full", "search JSONs not yet produced")

# ======================================================== specificity
sp = load("specificity.json")
if sp:
    lines = []
    b0 = {g: bias([r for r in sp["position"] if r["strength"] == 0.0 and r["mode"] == "all"
                   and r["group"] == g]) for g in ("male", "female")}
    for mode, mlab in (("all", "all positions"), ("final_only", "final position only"),
                       ("except_final", "all but final")):
        row = [mlab]
        for s in (-6.0, 6.0):
            for g in ("male", "female"):
                r = [x for x in sp["position"] if x["mode"] == mode and x["strength"] == s
                     and x["group"] == g]
                row.append(f"${bias(r).mean():+.2f}$" if r else "---")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    lines.append("unsteered & \\multicolumn{4}{c}{"
                 f"male ${b0['male'].mean():+.2f}$, female ${b0['female'].mean():+.2f}$"
                 "} \\\\")
    write("tab_position", "\n".join(lines))

    lines = []
    for layer in sorted({r["layer"] for r in sp["layer"]}):
        row = [f"{layer}"]
        for s in (-6.0, 6.0):
            for g in ("male", "female"):
                r = [x for x in sp["layer"] if x["layer"] == layer and x["strength"] == s
                     and x["group"] == g]
                row.append(f"${bias(r).mean():+.2f}$" if r else "---")
        lines.append(" & ".join(row) + " \\\\")
    write("tab_layer", "\n".join(lines))

    lines = []
    for pref, plab in (("none", "no instruction"),
                       ("neutral_instruction", "gender-neutral instruction"),
                       ("strong_instruction", "explicit ``use they'' instruction")):
        row = [plab]
        for g in ("male", "female", "neutral"):
            r = [x for x in sp["prompting"] if x["prefix"] == pref and x["group"] == g]
            if r:
                b = bias(r)
                lo, hi = ci(b)
                row.append(f"${b.mean():+.2f}$ $[{lo:+.2f},{hi:+.2f}]$")
            else:
                row.append("---")
        lines.append(" & ".join(row) + " \\\\")
    write("tab_prompting", "\n".join(lines))

    lines = []
    for g, lab in (("male", "male-stereo"), ("female", "female-stereo"), ("neutral", "neutral")):
        r = [x for x in sp["readout"] if x["group"] == g]
        mass = np.mean([x["pronoun_mass"] for x in r])
        he_rank = int(np.median([x["he_rank"] for x in r]))
        she_rank = int(np.median([x["she_rank"] for x in r]))
        top1 = {}
        for x in r:
            top1[x["top5"][0][0]] = top1.get(x["top5"][0][0], 0) + 1
        common = ", ".join(f"``{k.strip()}'' ({v})" for k, v in
                           sorted(top1.items(), key=lambda kv: -kv[1])[:3])
        lines.append(f"{lab} & {mass:.3f} & {he_rank} & {she_rank} & {common} \\\\")
    write("tab_readout", "\n".join(lines))
else:
    for n in ("tab_position", "tab_layer", "tab_prompting", "tab_readout"):
        placeholder(n, "specificity.json not yet produced")

# ======================================================== geometry line
geo = load("geometry.json")
if geo:
    write("txt_geometry",
          f"cosine similarity ${geo['cos_sae_vs_diff_in_means']:.2f}$ with the "
          f"difference-in-means direction and ${geo['cos_sae_vs_unembed']:.2f}$ with the "
          f"unembedding direction; one dose unit perturbs the residual stream by "
          f"${100*geo['perturbation_fraction_of_resid_norm_per_unit_strength']:.1f}\\%$ "
          f"of its mean norm")
else:
    placeholder("txt_geometry", "geometry.json not yet produced")


# ======================================================== pre-flight + candidates
# These two were hand-written in main.tex and silently went stale when
# sae_quality.py was re-run with the BOS fix. Generating them removes that risk.
q = load("sae_quality.json")
if q:
    f = np.array(q["per_passage_fvu"])
    pct = 100 * q["mean_l0"] / q["n_features"]
    pct_bos = 100 * q["mean_l0_with_bos"] / q["n_features"]
    def thou(x):
        return f"{x:,.0f}".replace(",", "{,}")

    write("tab_preflight", "\n".join([
        # the two broken states are historical facts, not recomputable values
        "raw inputs, unstandardised    & 3        & 0.01\\% & ---   & dead \\\\",
        "corrupt file from the catalog & 30{,}688 & 93.7\\% & ---   & saturated \\\\",
        f"reconstructed from WikiText   & {thou(q['mean_l0'])} & {pct:.1f}\\% & "
        f"{np.median(f):.3f} & usable, but not sparse \\\\",
        "\\midrule",
        f"\\emph{{as above, BOS included}} & {thou(q['mean_l0_with_bos'])} & "
        f"{pct_bos:.1f}\\% & {q['mean_fvu_with_bos']:.1f} & unusable diagnostic",
    ]))

    VERDICT = {"f32258": "masculine-referent context",
               "f27420": "low-magnitude, no coherent theme",
               "f9392": "food (fires on \\emph{dietitian})",
               "f26265": "possessive determiner"}
    lines = []
    for k in ("f32258", "f27420", "f9392", "f26265"):
        a = [np.mean(q["template_activations"][g][k]) for g in ("male", "female", "neutral")]
        lines.append(f"\\ff{{{k[1:]}}} & {a[0]:.2f} & {a[1]:.2f} & {a[2]:.2f} & "
                     f"${q['vocab_push'][k]['he_minus_she']:+.3f}$ & "
                     f"{q['feature_density_pct'][k]:.1f} & {VERDICT[k]} \\\\")
    write("tab_candidates", "\n".join(lines))

    # numbers the prose quotes, so they cannot drift either
    write("txt_saestats", f"mean $L_0$ {q['mean_l0']:,.0f}".replace(",", "{,}")
          + f", {pct:.1f}\\% of features active, median FVU {np.median(f):.3f}, "
          + f"BOS inflation {q['mean_fvu_with_bos']/np.median(f):.0f}$\\times$")
else:
    placeholder("tab_preflight", "sae\\_quality.json not yet produced")
    placeholder("tab_candidates", "sae\\_quality.json not yet produced")
