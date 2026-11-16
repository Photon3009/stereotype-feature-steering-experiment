"""Statistics for the paper: paired tests, bootstrap CIs, per-occupation win rates.

Runs over the per-prompt sweep artifacts (no model needed). The blog post reported
ratios of group-mean probabilities; a reviewer will ask for a paired test and an
interval, because 12 occupations is a small sample. Primary metric is the per-prompt
log-odds-style bias score

    b = log P(' he') - log P(' she')

which is symmetric around 0, additive, and pairable across steering strengths
(the same occupation appears at every strength).

Outputs results/stats.json and prints LaTeX-ready rows.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("experiment/results")
RNG = np.random.default_rng(0)
NBOOT = 20000


def bias(rows):
    return np.array([np.log(r["p_he"]) - np.log(r["p_she"]) for r in rows])


def boot_ci(x, fn=np.mean, n=NBOOT, alpha=0.05):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    d = fn(x[idx], axis=1)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def paired(x0, x1):
    """x1 vs x0, same prompts. Wilcoxon signed-rank + paired bootstrap on the mean diff."""
    d = np.asarray(x1) - np.asarray(x0)
    if np.allclose(d, 0):
        return {"n": len(d), "mean_diff": 0.0, "ci": [0.0, 0.0], "p_wilcoxon": 1.0, "cohens_dz": 0.0}
    w = stats.wilcoxon(x1, x0, alternative="two-sided", zero_method="wilcox")
    lo, hi = boot_ci(d)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci": [lo, hi],
            "p_wilcoxon": float(w.pvalue), "cohens_dz": float(d.mean() / (d.std(ddof=1) + 1e-12))}


def group_summary(rows):
    b = bias(rows)
    lo, hi = boot_ci(b)
    return {"n": len(rows),
            "mean_p_he": float(np.mean([r["p_he"] for r in rows])),
            "mean_p_she": float(np.mean([r["p_she"] for r in rows])),
            "mean_p_they": float(np.mean([r.get("p_they", np.nan) for r in rows])),
            "ratio_of_means": float(np.mean([r["p_he"] for r in rows]) / np.mean([r["p_she"] for r in rows])),
            "mean_bias": float(b.mean()), "bias_ci": [lo, hi],
            "frac_he_preferring": float(np.mean(b > 0))}


def key_of(r, has_id):
    return r["id"] if has_id else None


def analyze_sweep(rows, label, groups, strength_key="strength", method=None, out=None):
    if method is not None:
        rows = [r for r in rows if r.get("method") == method or r.get("feature") == method]
    strengths = sorted({r[strength_key] for r in rows})
    res = {}
    print(f"\n===== {label} " + (f"[{method}]" if method else ""))
    for g in groups:
        gr = [r for r in rows if r["group"] == g]
        if not gr:
            continue
        # pair prompts across strengths: by id when present, else by stable within-strength order
        has_id = "id" in gr[0] or "text" in gr[0]
        per_s = {}
        for s in strengths:
            sel = [r for r in gr if r[strength_key] == s]
            if has_id:
                sel = sorted(sel, key=lambda r: r.get("id", r.get("text")))
            per_s[s] = sel
        n0 = len(per_s[strengths[0]])
        if any(len(v) != n0 for v in per_s.values()):
            print(f"  !! {g}: unequal prompt counts across strengths, skipping paired tests")
            continue
        base = 0.0 if 0.0 in per_s else strengths[len(strengths) // 2]
        res[g] = {"baseline_strength": base, "by_strength": {}}
        for s in strengths:
            summ = group_summary(per_s[s])
            if s != base:
                summ["vs_baseline"] = paired(bias(per_s[base]), bias(per_s[s]))
            res[g]["by_strength"][str(s)] = summ
            v = summ.get("vs_baseline")
            extra = ""
            if v:
                extra = (f"  Δbias={v['mean_diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]"
                         f" p={v['p_wilcoxon']:.1e} dz={v['cohens_dz']:+.2f}")
            print(f"  {g:10s} s={s:+.1f}  he={summ['mean_p_he']:.3f} she={summ['mean_p_she']:.3f}"
                  f"  bias={summ['mean_bias']:+.2f} [{summ['bias_ci'][0]:+.2f},{summ['bias_ci'][1]:+.2f}]"
                  f"  he-pref {summ['frac_he_preferring']*100:.0f}%" + extra)
    if out is not None:
        out[label + (f"::{method}" if method else "")] = res
    return res


all_stats = {}

# ---- Llama f32258 (main) + f27420 (negative control) -----------------------
steer = json.load(open(OUT / "steer_sweep.json"))
analyze_sweep(steer, "llama_steer", ["male", "female", "D_female", "D_male"],
              method="he_stereo_f32258", out=all_stats)
analyze_sweep(steer, "llama_steer", ["male", "female", "D_female", "D_male"],
              method="she_stereo_f27420", out=all_stats)

# ---- Llama robustness (varied predicates, same occupations) ----------------
rob = json.load(open(OUT / "robustness_sweep.json"))
analyze_sweep(rob, "llama_robustness", ["A", "B"], out=all_stats)

# ---- LFM2.5-230M f9619 ----------------------------------------------------
lfm = json.load(open(OUT / "lfm_l9_f9619_sweep.json"))
analyze_sweep(lfm, "lfm_f9619", ["male", "female", "neutral"], out=all_stats)

# ---- new artifacts, if the baseline run has produced them ------------------
for fname, label, groups in [
    ("baseline_sweep.json", "baselines_main", ["male", "female", "neutral", "D_female", "D_male"]),
    ("heldout_sweep.json", "heldout", ["male", "female"]),
]:
    p = OUT / fname
    if not p.exists():
        print(f"\n(skipping {fname}: not produced yet)")
        continue
    rows = json.load(open(p))
    methods = sorted({r["method"] for r in rows} - {"baseline"})
    base_rows = [r for r in rows if r["method"] == "baseline"]
    for m in methods:
        merged = base_rows + [r for r in rows if r["method"] == m]
        analyze_sweep(merged, label, groups, method=None, out=all_stats)
        all_stats[f"{label}::{m}"] = all_stats.pop(label)

# ---- ablation -------------------------------------------------------------
p = OUT / "ablation.json"
if p.exists():
    abl = json.load(open(p))
    print("\n===== projection ablation (necessity test)")
    res = {}
    for m in sorted({r["method"] for r in abl}):
        for st in ("tpl", "heldout"):
            for g in ("male", "female"):
                sel = [r for r in abl if r["method"] == m and r["set"] == st and r["group"] == g]
                if not sel:
                    continue
                s = group_summary(sel)
                res[f"{m}|{st}|{g}"] = s
                print(f"  {m:24s} {st:8s} {g:7s} he={s['mean_p_he']:.3f} she={s['mean_p_she']:.3f} "
                      f"bias={s['mean_bias']:+.2f} [{s['bias_ci'][0]:+.2f},{s['bias_ci'][1]:+.2f}]")
    # paired: no_ablation vs each proj-out, on tpl male
    for m in sorted({r["method"] for r in abl} - {"no_ablation"}):
        for g in ("male", "female"):
            b0 = bias(sorted([r for r in abl if r["method"] == "no_ablation" and r["set"] == "tpl" and r["group"] == g], key=lambda r: r["id"]))
            b1 = bias(sorted([r for r in abl if r["method"] == m and r["set"] == "tpl" and r["group"] == g], key=lambda r: r["id"]))
            t = paired(b0, b1)
            res[f"paired|{m}|{g}"] = t
            print(f"  paired {m} [{g}]: Δbias={t['mean_diff']:+.2f} [{t['ci'][0]:+.2f},{t['ci'][1]:+.2f}] p={t['p_wilcoxon']:.1e}")
    all_stats["ablation"] = res

# ---- Holm-Bonferroni across the family of primary paired tests ---------------
def holm(stats_obj):
    """Collect every vs_baseline p-value, Holm-correct in place, report the family size."""
    entries = []

    def walk(node):
        if isinstance(node, dict):
            if "vs_baseline" in node and isinstance(node["vs_baseline"], dict):
                entries.append(node["vs_baseline"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(stats_obj)
    m = len(entries)
    order = sorted(range(m), key=lambda i: entries[i]["p_wilcoxon"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * entries[i]["p_wilcoxon"])
        running = max(running, adj)          # enforce monotonicity
        entries[i]["p_holm"] = running
        entries[i]["significant_holm_05"] = bool(running < 0.05)
    return m, sum(e["significant_holm_05"] for e in entries)


m, k = holm(all_stats)
print(f"\nHolm-Bonferroni over the family of {m} paired tests: "
      f"{k} remain significant at alpha=0.05")
all_stats["_holm_family_size"] = m
all_stats["_holm_n_significant"] = k

json.dump(all_stats, open(OUT / "stats.json", "w"), indent=1)
print("\nsaved experiment/results/stats.json")
