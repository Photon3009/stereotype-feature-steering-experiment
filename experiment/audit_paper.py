"""One-pass audit of paper/main.tex against the saved results.

Checks, in order:
  1. Every numeric claim in the prose has a matching value in the data.
  2. Every \\input'd table body has the column count its header declares.
  3. Every strength quoted in the prose appears in some table in the paper.
  4. Figure label text contains no raw LaTeX (matplotlib does not run a LaTeX pass).
  5. Captions contain no stale conditionals ("If X reproduces...") for questions
     the tables now answer.
  6. Structure: undefined refs, unreferenced floats, anonymity, page budget.

Run from the repo root. Prints a report; exits non-zero if anything fails.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RES = Path("experiment/results")
TEX = Path("paper/main.tex")
TAB = Path("paper/tables")
problems = []


def load(n):
    p = RES / n
    return json.load(open(p)) if p.exists() else None


def bias(rows):
    return np.array([math.log(r["p_he"]) - math.log(r["p_she"]) for r in rows])


# ---------------------------------------------------------------- number universe
# Every quantity the paper could legitimately quote, rounded the way it prints.
universe = set()


def add(*vals):
    for v in vals:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        for nd in (1, 2, 3, 4):
            universe.add(f"{abs(round(float(v), nd)):.{nd}f}".rstrip("0").rstrip("."))


def add_group(rows):
    if not rows:
        return
    b = bias(rows)
    add(b.mean(), np.mean([r["p_he"] for r in rows]), np.mean([r["p_she"] for r in rows]))
    if "p_they" in rows[0]:
        add(np.mean([r["p_they"] for r in rows]))
    lo, hi = np.quantile(b, [0.025, 0.975])
    add(lo, hi)


for fname, keys in [("baseline_sweep.json", ("method", "group", "strength")),
                    ("heldout_sweep.json", ("method", "group", "strength")),
                    ("override_sweep.json", ("method", "group", "strength")),
                    ("ablation.json", ("method", "set", "group")),
                    ("steer_sweep.json", ("feature", "group", "strength")),
                    ("robustness_sweep.json", ("group", "strength")),
                    ("lfm_l9_f9619_sweep.json", ("group", "strength"))]:
    d = load(fname)
    if not d:
        continue
    cells = {}
    for r in d:
        cells.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    for rows in cells.values():
        add_group(rows)
        # paired differences against the same cell's baseline are also quoted
    # deltas vs strength 0 / baseline method
    base = {}
    for k, rows in cells.items():
        if 0.0 in k or "baseline" in [str(x) for x in k] or "no_ablation" in [str(x) for x in k]:
            base[tuple(x for x in k if x not in (0.0, "baseline", "no_ablation"))] = bias(rows)
    for k, rows in cells.items():
        key = tuple(x for x in k if x not in (0.0, "baseline", "no_ablation"))
        for bk, bv in base.items():
            if len(bv) == len(rows):
                add((bias(rows) - bv).mean())

for c in (load("capability_baselines.json") or []):
    add(c["wikitext_loss"])
    b0 = [x for x in (load("capability_baselines.json") or []) if x["method"] == "baseline"]
    if b0:
        add(100 * (c["wikitext_loss"] - b0[0]["wikitext_loss"]) / b0[0]["wikitext_loss"])

q = load("sae_quality.json")
if q:
    f = np.array(q["per_passage_fvu"])
    add(q["mean_l0"], q["mean_fvu"], np.median(f), f.min(), f.max(), f.mean(),
        q["mean_l0_with_bos"], q["mean_fvu_with_bos"],
        100 * q["mean_l0"] / q["n_features"], q["mean_fvu_with_bos"] / np.median(f))
    for k, v in q["feature_density_pct"].items():
        add(v)
    for k, v in q["vocab_push"].items():
        add(v["he_minus_she"])
    for k, v in q["decoder_neighbours"].items():
        for _, c in v:
            add(c)
    for g, per in q["template_activations"].items():
        for k, vals in per.items():
            add(np.mean(vals))

g = load("geometry.json")
if g:
    add(*g.values(), 100 * g["perturbation_fraction_of_resid_norm_per_unit_strength"],
        8 * 100 * g["perturbation_fraction_of_resid_norm_per_unit_strength"])

for sf in ("llama_search.json", "lfm_search.json"):
    d = load(sf)
    if not d:
        continue
    blocks = [d["candidates"]] if "candidates" in d else [v["candidates"] for v in d["layers"].values()]
    for cb in blocks:
        for side in cb.values():
            for c in side:
                add(c.get("swing_male"), c.get("swing_female"), c["act_male"],
                    c["act_female"], c["act_neutral"])

# numbers that come from the generated tables are by construction fine
for t in TAB.glob("*.tex"):
    for m in re.finditer(r"[-+]?\d+\.\d+", t.read_text()):
        universe.add(m.group().lstrip("+-").rstrip("0").rstrip(".") or "0")

# ---------------------------------------------------------------- prose numbers
src = TEX.read_text()
body = src[src.index("\\begin{abstract}"):]
# strip tables, figures and the appendix probe lists: only audit prose
prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", body, flags=re.S)
prose = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", prose, flags=re.S)
prose = re.sub(r"\\cite[a-z]*\{[^}]*\}", " ", prose)
prose = re.sub(r"\\(?:ref|label|input)\{[^}]*\}", " ", prose)

STRUCTURAL = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "14", "16",
    "24", "36", "48", "100", "200", "230", "370", "2048", "32768", "20000", "95",
    "1.2", "3.2", "2.5", "1.1", "0.125", "4.9", "0.5", "1.0", "0.01", "0.05",
}
flagged = []
for m in re.finditer(r"(?<![\w.])[-+]?\d+\.\d+(?![\w])", prose):
    tok = m.group().lstrip("+-")
    norm = tok.rstrip("0").rstrip(".") or "0"
    if norm in STRUCTURAL or norm in universe or tok in universe:
        continue
    a = max(0, m.start() - 90)
    flagged.append((tok, re.sub(r"\s+", " ", prose[a:m.end() + 40])))
if flagged:
    problems.append(f"{len(flagged)} prose number(s) with no match in the data")
print("=" * 78)
print("1. PROSE NUMBERS NOT FOUND IN THE DATA")
print("=" * 78)
if flagged:
    for tok, ctx in flagged:
        print(f"  !! {tok}\n     ...{ctx}...")
else:
    print("  ok: every prose number matches a computed value")

# ---------------------------------------------------------------- column counts
print("\n" + "=" * 78)
print("2. TABLE COLUMN COUNTS")
print("=" * 78)
for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", src, re.S):
    spec, blockk = m.group(1), m.group(2)
    ncol = len(re.findall(r"[lcr]", spec))
    inp = re.search(r"\\input\{tables/([a-z_]+)\}", blockk)
    hdr = re.search(r"\\toprule\s*(.*?)\\\\", blockk, re.S)
    label = re.search(r"\\label\{([^}]+)\}", src[max(0, m.start() - 700):m.start()])
    name = label.group(1) if label else "?"
    if hdr:
        h = hdr.group(1)
        hcols = h.count("&") + 1
        for mc in re.finditer(r"\\multicolumn\{(\d+)\}", h):
            hcols += int(mc.group(1)) - 1
        if hcols != ncol and "cmidrule" not in blockk:
            problems.append(f"{name}: header {hcols} cols vs spec {ncol}")
            print(f"  !! {name}: header has {hcols} columns, spec declares {ncol}")
    if inp:
        f = TAB / f"{inp.group(1)}.tex"
        if f.exists():
            for i, line in enumerate(f.read_text().strip().split("\n")):
                if line.strip() in ("\\midrule", "") or "multicolumn" in line:
                    continue
                cols = line.count("&") + 1
                if cols != ncol:
                    problems.append(f"{name}: row {i+1} has {cols} cols vs {ncol}")
                    print(f"  !! {name}: generated row {i+1} has {cols} columns, spec {ncol}")
print("  (only mismatches shown)")

# ---------------------------------------------------------------- quoted strengths
print("\n" + "=" * 78)
print("3. STRENGTHS QUOTED IN PROSE THAT MUST APPEAR IN A TABLE")
print("=" * 78)
tabnums = " ".join(t.read_text() for t in TAB.glob("*.tex"))
quoted = {int(x) for x in re.findall(r"\$s\s*=\s*([-+]\d+)\$", prose)}
tabstr = {int(x) for x in re.findall(r"\$([-+]\d+)\$", tabnums)}
missing = sorted(s for s in quoted if s not in tabstr)
if missing:
    problems.append(f"strengths quoted but not tabulated: {missing}")
    print(f"  !! quoted in prose but in no table: {missing}")
else:
    print(f"  ok: all quoted strengths {sorted(quoted)} appear in tables")

# ---------------------------------------------------------------- figure labels
print("\n" + "=" * 78)
print("4. RAW LATEX IN FIGURE LABELS")
print("=" * 78)
figsrc = Path("experiment/make_paper_figures.py").read_text()
bad = []
for line in figsrc.splitlines():
    if any(k in line for k in ("set_title", "set_xlabel", "set_ylabel", "annotate", "label=")):
        # strip plot-format strings like "--s" (linestyle, not text) before checking
        checked = re.sub(r'"[-.:]{1,2}[a-zA-Z]?"', '""', line)
        # matplotlib renders $...$ mathtext, so those are fine; only non-math
        # escapes and literal double hyphens print as typed
        checked = re.sub(r"\$[^$]*\$", "", checked)
        for tok in ("\\%", "\\u", "--", "\\text"):
            if tok in checked and "matched $L_2$" not in checked:
                bad.append((tok, line.strip()))
for tok, line in bad:
    problems.append(f"figure label contains {tok!r}")
    print(f"  !! {tok!r} in: {line}")
if not bad:
    print("  ok: no raw LaTeX escapes in axis labels or titles")

# ---------------------------------------------------------------- stale captions
print("\n" + "=" * 78)
print("5. CONDITIONAL CAPTIONS FOR ANSWERED QUESTIONS")
print("=" * 78)
stale = [c[:90] for c in re.findall(r"\\caption\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", src)
         if re.match(r"\s*(If|Whether|Should)\b", c) or " if " in c[:60].lower()]
for c in stale:
    problems.append("conditional caption")
    print(f"  !! {c}...")
if not stale:
    print("  ok: no captions pose a question the table answers")

# ---------------------------------------------------------------- structure
print("\n" + "=" * 78)
print("6. STRUCTURE")
print("=" * 78)
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
for name, val in [("refs without a label", sorted(refs - labels)),
                  ("floats never referenced",
                   sorted(x for x in labels - refs if x.startswith(("tab:", "fig:"))))]:
    print(f"  {name}: {val or 'none'}")
    if val:
        problems.append(name)
log = Path("paper/main.log")
if log.exists():
    t = log.read_text(errors="ignore")
    und = len(re.findall(r"Reference .* undefined", t))
    print(f"  undefined references in last build: {und}")
    if und:
        problems.append("undefined references")
pdf = Path("paper/main.pdf")
if pdf.exists():
    raw = pdf.read_bytes()
    hits = sum(raw.lower().count(w) for w in (b"shivam", b"verma", b"gmail"))
    print(f"  identifying strings in PDF: {hits}")
    if hits:
        problems.append("PDF is not anonymous")

print("\n" + "=" * 78)
print(f"RESULT: {len(problems)} problem(s)" if problems else "RESULT: clean")
for p in problems:
    print(f"  - {p}")
sys.exit(1 if problems else 0)
