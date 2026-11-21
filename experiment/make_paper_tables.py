"""Generate every numeric table in the paper as a LaTeX fragment under paper/tables/.

main.tex inputs these, so no number is ever transcribed by hand. Fragments are
written only when their input JSON exists, and each writes a one-line placeholder
otherwise so the document still compiles while experiments are running.
"""
import json
from pathlib import Path

import numpy as np

from paperstats import bias, ci, fmt_b, fmt_ci, fmt_p, holm, paired, wilcoxon_p

RES = Path("experiment/results")
TAB = Path("paper/tables")
TAB.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


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


# ======================================================== numbers quoted in prose
# Every interval the abstract and body quote is a macro defined here, computed by the
# same ci() as the tables. The submitted version typed them by hand and they drifted.
macros = {}


def mac(name, rows=None, b=None):
    b = bias(rows) if b is None else np.asarray(b)
    lo, hi = ci(b)
    macros[name] = fmt_b(b.mean())
    macros[name + "CI"] = fmt_ci(lo, hi)
    return b


def by_id(rows):
    return sorted(rows, key=lambda r: r.get("id", r.get("text", "")))


# primary family: pre-specified tests the text makes a claim from (label, b0, b1)
primary = []
if sw:
    b0m = mac("BaseMale", sel(sw, method="baseline", group="male"))
    b0f = mac("BaseFemale", sel(sw, method="baseline", group="female"))
    b1m = mac("SteerMale", sel(sw, method="sae_f32258", group="male", strength=-6.0))
    b1f = bias(sel(sw, method="sae_f32258", group="female", strength=6.0))
    primary += [("steer s=-6, male, discovery", b0m, b1m),
                ("steer s=+6, female, discovery", b0f, b1f)]
rob = load("robustness_sweep.json")
if rob:
    r0 = mac("RobBase", by_id([r for r in rob if r["group"] == "A" and r["strength"] == 0.0]))
    r1 = mac("RobSteer", by_id([r for r in rob if r["group"] == "A" and r["strength"] == -6.0]))
    macros["RobDelta"] = fmt_b((r1 - r0).mean())
    primary.append(("steer s=-6, male, varied predicate", r0, r1))
if ho:
    h0 = mac("HeldBase", sel(ho, method="baseline", group="male"))
    h1 = mac("HeldSteer", sel(ho, method="sae_f32258", group="male", strength=-6.0))
    primary.append(("steer s=-6, male, held-out", h0, h1))
if ab:
    for st, nm in (("tpl", "ProjTpl"), ("heldout", "ProjHeld")):
        a0 = bias(sel(ab, method="no_ablation", set=st, group="male"))
        a1 = mac(nm, sel(ab, method="projout_sae_f32258", set=st, group="male"))
        primary.append((f"project out, male, {st}", a0, a1))
if ov:
    o0 = bias(sel(ov, method="sae_f32258", group="D_female", strength=0.0))
    o1 = bias(sel(ov, method="sae_f32258", group="D_female", strength=6.0))
    primary.append(("override s=+6, explicit her", o0, o1))
st_rows = load("steer_sweep.json")
if st_rows:
    nc = [r for r in st_rows if r["feature"] == "she_stereo_f27420" and r["group"] == "female"]
    n0 = bias(by_id([r for r in nc if r["strength"] == 0.0]))
    n1 = bias(by_id([r for r in nc if r["strength"] == -6.0]))
    primary.append(("negative control f27420 s=-6, female", n0, n1))
lfm = load("lfm_l9_f9619_sweep.json")
if lfm:
    mac("LfmBaseMale", [r for r in lfm if r["group"] == "male" and r["strength"] == 0.0])
    mac("LfmBaseFemale", [r for r in lfm if r["group"] == "female" and r["strength"] == 0.0])

if cap:
    def dloss(meth, s):
        c = [c for c in cap if c["method"] == meth and c["strength"] == s][0]
        return f"$+{100 * (c['wikitext_loss'] - base_loss) / base_loss:.1f}\\%$"
    macros["DlossSAE"] = dloss("sae_f32258", -6.0)
    macros["DlossSAEEight"] = dloss("sae_f32258", -8.0)
    macros["DlossUnembed"] = dloss("unembed_he_she", -8.0)
    macros["DlossDiM"] = dloss("diff_in_means", -8.0)
    macros["DlossProj"] = dloss("projout_sae_f32258", 0.0)
if q:
    macros["BosInflation"] = f"{q['mean_fvu_with_bos'] / np.median(q['per_passage_fvu']):.0f}"
if lfm_s:
    he9 = lfm_s["layers"]["9"]["candidates"]["he"]
    macros["LfmRankOneSwing"] = f"{he9[0]['swing_male']:+.2f}"
    macros["LfmRankTwoSwing"] = f"{he9[1]['swing_male']:.2f}"

if primary:
    raw = [wilcoxon_p(a, b) for _, a, b in primary]
    adj = holm(raw)
    print("\nprimary family (Holm):")
    for (lab, a, b), p, q in zip(primary, raw, adj):
        print(f"  {lab:42s} n={len(a):3d}  d={np.mean(b - a):+.2f}  p={p:.1e}  p_holm={q:.1e}")
    macros["PrimaryN"] = str(len(primary))
    macros["PrimaryMaxHolm"] = f"{max(adj):.1e}".replace("e-0", "\\times10^{-") + "}"
    json.dump([{"test": l, "n": len(a), "p": p, "p_holm": q}
               for (l, a, _), p, q in zip(primary, raw, adj)],
              open(RES / "primary_family.json", "w"), indent=1)

(TAB / "macros.tex").write_text(
    "% generated by experiment/make_paper_tables.py -- do not edit\n" +
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
print(f"wrote tables/macros.tex ({len(macros)} macros)")


def gap_ci(m, f, n=20000):
    r = np.random.default_rng(0)
    m, f = np.asarray(m), np.asarray(f)
    d = (m[r.integers(0, len(m), (n, len(m)))].mean(1)
         - f[r.integers(0, len(f), (n, len(f)))].mean(1))
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ======================================================== camera-ready: expanded probes
# Unit of analysis is the occupation: b averaged over its frames, then CI over occupations.
def per_occ(rows):
    occ = {}
    for r in rows:
        occ.setdefault(r["occupation"], []).append(r)
    keys = sorted(occ)
    return keys, np.array([bias(occ[k]).mean() for k in keys])


xr = load("expanded_probes.json")
if xr:
    XN = {"baseline": "unsteered", "sae_f32258": "SAE \\ff{32258}",
          "unembed_he_she": "unembedding he$-$she", "diff_in_means": "difference-in-means",
          "projout_sae_f32258": "project out \\ff{32258}",
          "projout_diff_in_means": "project out diff.-in-means"}
    conds = []
    for r in xr:
        c = (r["method"], r["strength"])
        if c not in conds:
            conds.append(c)
    lines = []
    for subset, keep in (("all", lambda r: True), ("unseen", lambda r: not r["in_discovery"])):
        for c in conds:
            if subset == "unseen" and c[0] != "baseline" and c not in (("unembed_he_she", -8.0), ("sae_f32258", -6.0)):
                continue
            row = [f"{XN[c[0]]}" + (f", $s={c[1]:+.0f}$" if not c[0].startswith(("baseline", "projout")) else "")]
            for g in ("male", "female"):
                rows0 = [r for r in xr if r["method"] == "baseline" and r["group"] == g and keep(r)]
                rows1 = [r for r in xr if (r["method"], r["strength"]) == c and r["group"] == g and keep(r)]
                k0, b0 = per_occ(rows0)
                k1, b1 = per_occ(rows1)
                lo, hi = ci(b1)
                cell = f"{fmt_b(b1.mean())} {fmt_ci(lo, hi)}"
                if c[0] != "baseline":
                    cell += f" & {fmt_b((b1 - b0).mean())}"
                else:
                    cell += " & ---"
                row.append(cell)
                if subset == "all":
                    tag = {"baseline": "Base", "sae_f32258": "Sae", "unembed_he_she": "Unemb",
                           "diff_in_means": "DiM", "projout_sae_f32258": "ProjSae",
                           "projout_diff_in_means": "ProjDiM"}[c[0]]
                    sname = {-8.0: "MinusEight", -6.0: "MinusSix", -3.0: "MinusThree",
                             3.0: "PlusThree", 6.0: "PlusSix", 0.0: ""}[c[1]]
                    mac(f"X{tag}{sname}{g.capitalize()}", b=b1)
                    macros[f"XN{g.capitalize()}"] = str(len(k1))
            lines.append(" & ".join(row) + " \\\\")
        if subset == "all":
            n_unseen = len({r["occupation"] for r in xr if not r["in_discovery"] and r["group"] == "male"})
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{excluding the "
                         f"{len({r['occupation'] for r in xr if r['in_discovery']})} "
                         f"occupations that also appear in discovery}}}} \\\\")
            lines.append("\\midrule")
    write("tab_expanded", "\n".join(lines))
    macros["XFrames"] = str(len({r["frame"] for r in xr}))
    macros["XOverlap"] = str(len({r["occupation"] for r in xr if r["in_discovery"]}))
    # add to the primary family
    xc = next((c for c in (("sae_f32258", -6.0), ("unembed_he_she", -6.0))
               if any((r["method"], r["strength"]) == c for r in xr)), None)
    if xc:
        _, a = per_occ([r for r in xr if r["method"] == "baseline" and r["group"] == "male"])
        _, b = per_occ([r for r in xr if (r["method"], r["strength"]) == xc and r["group"] == "male"])
        primary.append((f"{xc[0]} s={xc[1]:+.0f}, male, expanded (per occupation)", a, b))
else:
    placeholder("tab_expanded", "expanded\\_probes.json not yet produced")

# ======================================================== camera-ready: generation
gr = load("generation_eval.json")
if gr:
    GN = {("baseline", 0.0): "unsteered", ("sae_f32258", -3.0): "SAE \\ff{32258}, $s=-3$",
          ("sae_f32258", -6.0): "SAE \\ff{32258}, $s=-6$",
          ("unembed_he_she", -6.0): "unembedding, $s=-6$",
          ("unembed_he_she", -8.0): "unembedding, $s=-8$",
          ("diff_in_means", -8.0): "diff.-in-means, $s=-8$",
          ("projout_sae_f32258", 0.0): "project out \\ff{32258}",
          ("projout_diff_in_means", 0.0): "project out diff.-in-means"}

    def he_share(rows):
        """Per occupation: share of continuations whose first gendered pronoun is male,
        among those with any gendered (he/she) pronoun. NaN if none."""
        occ = {}
        for r in rows:
            occ.setdefault(r["occupation"], []).append(r["first_pronoun"])
        keys = sorted(occ)
        out = []
        for k in keys:
            fp = occ[k]
            m, f = fp.count("male"), fp.count("female")
            out.append(m / (m + f) if m + f else np.nan)
        return keys, np.array(out)

    lines = []
    for fmt, flab in (("raw", "raw completion"), ("chat", "chat template")):
        first = True
        base = {g: [r for r in gr if r["fmt"] == fmt and r["group"] == g
                    and r["method"] == "baseline"] for g in ("male", "female")}
        for c, lab in GN.items():
            if not any((r["method"], r["strength"]) == c for r in gr):
                continue
            cells = [flab if first else "", lab]
            first = False
            for g in ("male", "female"):
                rows = [r for r in gr if r["fmt"] == fmt and r["group"] == g
                        and (r["method"], r["strength"]) == c]
                if not rows:
                    cells.append("---")
                    continue
                _, sh = he_share(rows)
                ok = sh[~np.isnan(sh)]
                lo, hi = ci(ok) if len(ok) > 1 else (np.nan, np.nan)
                cells.append(f"{100 * ok.mean():.0f}\\% [{100 * lo:.0f},{100 * hi:.0f}]")
                if g == "male":
                    tag = "Gen" + fmt.capitalize() + {"baseline": "Base", "sae_f32258": "Sae",
                                                      "unembed_he_she": "Unemb",
                                                      "diff_in_means": "DiM",
                                                      "projout_sae_f32258": "Proj",
                                                      "projout_diff_in_means": "ProjDiM"}[c[0]] + \
                          {-3.0: "Three", -6.0: "Six", -8.0: "Eight", 0.0: ""}[c[1]]
                    macros[tag] = f"{100 * ok.mean():.0f}\\%"
                    macros[tag + "CI"] = f"[{100 * lo:.0f},{100 * hi:.0f}]\\%"
            # gap in percentage points of he-share, male-stereo minus female-stereo
            sh = {}
            for g in ("male", "female"):
                _, x = he_share([r for r in gr if r["fmt"] == fmt and r["group"] == g
                                 and (r["method"], r["strength"]) == c])
                sh[g] = x[~np.isnan(x)]
            glo, ghi = gap_ci(sh["male"], sh["female"])
            gtag = "Gen" + fmt.capitalize() + "Gap" + {"baseline": "Base", "sae_f32258": "Sae",
                                                       "unembed_he_she": "Unemb",
                                                       "diff_in_means": "DiM",
                                                       "projout_sae_f32258": "Proj",
                                                       "projout_diff_in_means": "ProjDiM"}[c[0]] + \
                {-3.0: "Three", -6.0: "Six", -8.0: "Eight", 0.0: ""}[c[1]]
            macros[gtag] = f"{100 * (sh['male'].mean() - sh['female'].mean()):.0f}"
            macros[gtag + "CI"] = f"[{100 * glo:.0f},{100 * ghi:.0f}]"
            macros[gtag.replace("Gap", "Fem")] = f"{100 * sh['female'].mean():.0f}\\%"
            allr = [r for r in gr if r["fmt"] == fmt and (r["method"], r["strength"]) == c]
            n = len(allr)
            nonep = 100 * sum(r["first_pronoun"] == "none" for r in allr) / n
            theyp = 100 * sum(r["first_pronoun"] == "they" for r in allr) / n
            nll = np.mean([r["nll_unsteered"] for r in allr if r["nll_unsteered"] is not None])
            d2 = np.mean([r["distinct2"] for r in allr])
            cells += [f"{theyp:.0f}\\%", f"{nonep:.0f}\\%", f"{nll:.2f}", f"{d2:.2f}"]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write("tab_generation", "\n".join(lines[:-1]))
    macros["GenNOcc"] = str(len({r["occupation"] for r in gr if r["group"] == "male"}))
    macros["GenNPerCond"] = str(len([r for r in gr if r["method"] == "baseline"]))
    # primary: raw-format he-share, SAE s=-6 vs unsteered, male group, paired by occupation
    gen_c = next((c for c in (("sae_f32258", -6.0), ("unembed_he_she", -8.0))
                  if any((r["method"], r["strength"]) == c for r in gr)), None)
    for fmt in ("raw", "chat"):
        if gen_c is None:
            break
        k0, a = he_share([r for r in gr if r["fmt"] == fmt and r["group"] == "male" and r["method"] == "baseline"])
        k1, b = he_share([r for r in gr if r["fmt"] == fmt and r["group"] == "male"
                          and (r["method"], r["strength"]) == gen_c])
        m = ~np.isnan(a) & ~np.isnan(b)
        primary.append((f"generation he-share {gen_c[0]} s={gen_c[1]:+.0f}, male, {fmt}", a[m], b[m]))
    # metric validity: does next-token b predict the pronoun actually generated?
    if xr:
        from scipy.stats import spearmanr
        xk, xb = per_occ([r for r in xr if r["method"] == "baseline"])
        for fmt in ("raw", "chat"):
            gk, gs = he_share([r for r in gr if r["fmt"] == fmt and r["method"] == "baseline"])
            gmap = dict(zip(gk, gs))
            pairs = [(b, gmap[k]) for k, b in zip(xk, xb) if k in gmap and not np.isnan(gmap[k])]
            rho, pv = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            macros[f"Valid{fmt.capitalize()}Rho"] = f"{rho:.2f}"
            macros[f"Valid{fmt.capitalize()}N"] = str(len(pairs))
            print(f"metric validity ({fmt}): spearman rho={rho:.2f} p={pv:.1e} n={len(pairs)}")
else:
    placeholder("tab_generation", "generation\\_eval.json not yet produced")

# ======================================================== occupation gap (camera-ready)
# gap = mean b(male-stereo) - mean b(female-stereo): the occupation dependence itself.
# CI: independent bootstrap of the two group means (seeded as everywhere else).



def gap_mac(name, m, f):
    lo, hi = gap_ci(m, f)
    macros[name] = fmt_b(np.mean(m) - np.mean(f))
    macros[name + "CI"] = fmt_ci(lo, hi)


SN = {-8.0: "Eight", -6.0: "Six", -4.0: "Four", -10.0: "Ten", -12.0: "Twelve",
      -16.0: "Sixteen", -20.0: "Twenty", 0.0: ""}
MN = {"baseline": "Base", "sae_f32258": "Sae", "unembed_he_she": "Unemb",
      "diff_in_means": "DiM"}
gap_rows = []
if sw:
    for meth, st in (("baseline", 0.0), ("sae_f32258", -8.0), ("sae_f32258", -6.0),
                     ("sae_f32258", -4.0), ("unembed_he_she", -8.0),
                     ("unembed_he_she", -6.0), ("diff_in_means", -8.0)):
        m = bias(sel(sw, method=meth, group="male", strength=st))
        f = bias(sel(sw, method=meth, group="female", strength=st))
        gap_mac(f"Gap{MN[meth]}{SN[st]}", m, f)
        gap_rows.append(("discovery", meth, st, m, f))
gfr = load("gap_frontier.json")
if gfr:
    for meth in ("unembed_he_she", "diff_in_means"):
        for st in sorted({r["strength"] for r in gfr["rows"]}):
            rr = [r for r in gfr["rows"] if r["method"] == meth and r["strength"] == st
                  and r["set"] == "tpl"]
            m = bias(sel(rr, group="male"))
            f = bias(sel(rr, group="female"))
            gap_mac(f"Gap{MN[meth]}{SN[st]}", m, f)
            macros[f"B{MN[meth]}{SN[st]}Male"] = fmt_b(m.mean())
            gap_rows.append(("discovery", meth, st, m, f))
    gb = [c for c in gfr["capability"] if c["method"] == "baseline"][0]["wikitext_loss"]
    for c in gfr["capability"]:
        if c["method"] != "baseline":
            macros[f"Dloss{MN[c['method']]}{SN[c['strength']]}"] = \
                f"$+{100 * (c['wikitext_loss'] - gb) / gb:.1f}\\%$"
if ho:
    for meth, st in (("baseline", 0.0), ("sae_f32258", -6.0), ("unembed_he_she", -6.0),
                     ("diff_in_means", -6.0)):
        m = bias(sel(ho, method=meth, group="male", strength=st))
        f = bias(sel(ho, method=meth, group="female", strength=st))
        gap_mac(f"HeldGap{MN[meth]}{SN[st]}", m, f)
if ab:
    for st, nm in (("tpl", "Tpl"), ("heldout", "Held")):
        for meth, mn in (("no_ablation", "Base"), ("projout_sae_f32258", "Sae"),
                         ("projout_diff_in_means", "DiM")):
            m = bias(sel(ab, method=meth, set=st, group="male"))
            f = bias(sel(ab, method=meth, set=st, group="female"))
            gap_mac(f"ProjGap{nm}{mn}", m, f)
if xr:
    for c in (("baseline", 0.0), ("unembed_he_she", -8.0), ("unembed_he_she", -6.0),
              ("diff_in_means", -8.0)):
        _, m = per_occ([r for r in xr if (r["method"], r["strength"]) == c and r["group"] == "male"])
        _, f = per_occ([r for r in xr if (r["method"], r["strength"]) == c and r["group"] == "female"])
        gap_mac(f"XGap{MN[c[0]]}{SN[c[1]]}", m, f)
if gap_rows:
    lines = []
    for _, meth, st, m, f in sorted(gap_rows, key=lambda t: (list(MN).index(t[1]), -t[2])):
        lo, hi = gap_ci(m, f)
        lab = {"baseline": "unsteered", "sae_f32258": "SAE \\ff{32258}",
               "unembed_he_she": "unembedding he$-$she",
               "diff_in_means": "difference-in-means"}[meth]
        capr = [c for c in (list(cap or []) + (gfr["capability"] if gfr else []))
                if c["method"] == meth and c["strength"] == st]
        dl = (f"${100 * (capr[0]['wikitext_loss'] - base_loss) / base_loss:+.1f}\\%$"
              if capr and meth != "baseline" else "---")
        lines.append(f"{lab} & {'---' if meth == 'baseline' else f'${st:+.0f}$'} & "
                     f"{fmt_b(m.mean())} & {fmt_b(f.mean())} & "
                     f"{fmt_b(m.mean() - f.mean())} {fmt_ci(lo, hi)} & {dl} \\\\")
    write("tab_gap", "\n".join(lines))

rc = load("repro_check.json")
if rc:
    e = max(c["max_rel_err_p_he"] for c in rc)
    m, ex = f"{e:.1e}".split("e")
    macros["ReproMaxErr"] = f"{m}\\times10^{{{int(ex)}}}"

# re-emit the primary family and macros now that the camera-ready tests are in
if primary:
    raw = [wilcoxon_p(a, b) for _, a, b in primary]
    adj = holm(raw)
    print("\nprimary family incl. camera-ready (Holm):")
    for (lab, a, b), p, q in zip(primary, raw, adj):
        print(f"  {lab:46s} n={len(a):3d}  d={np.mean(b - a):+.3f}  p={p:.1e}  p_holm={q:.1e}")
    macros["PrimaryN"] = str(len(primary))
    macros["PrimaryMaxHolm"] = f"{max(adj):.1e}".replace("e-0", "\\times10^{-") + "}"
    json.dump([{"test": l, "n": len(a), "p": p, "p_holm": q}
               for (l, a, _), p, q in zip(primary, raw, adj)],
              open(RES / "primary_family.json", "w"), indent=1)
(TAB / "macros.tex").write_text(
    "% generated by experiment/make_paper_tables.py -- do not edit\n" +
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
print(f"wrote tables/macros.tex ({len(macros)} macros)")
