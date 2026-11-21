"""Camera-ready additions requested by the NewInML reviewers.

  0. REPRODUCTION CHECK. The SAE normalisation statistics are rebuilt locally by
     compute_norm.py, not downloaded. Before trusting anything new, re-measure a slice of
     baseline_sweep.json and require it to match the saved values.
  1. EXPANDED PROBES (next-token). The 40 WinoBias occupations (Zhao et al. 2018; 20
     male-, 20 female-dominated by US labour statistics) under 6 frames = 240 prompts.
     The unit of analysis is the occupation (n = 20 per group, bias averaged over frames),
     against 12 per group in discovery. Overlap with discovery is recorded so the
     unseen-only subset can be reported separately.
  2. GENERATION. Free-running continuations, scored by the first third-person pronoun the
     model actually writes, so the result no longer rests on a two-token contrast that
     carries 15-28% of next-token mass. Two formats: raw completion (as in the rest of
     the paper) and the chat template (how the model is deployed). Fluency is checked by
     the unsteered model's NLL of each continuation and by distinct-2.

Resumable: each (part, condition) cell is saved as soon as it finishes.
Outputs: results/repro_check.json, results/expanded_probes.json,
         results/generation_eval.json, probes/expanded.jsonl
Run from the repo root.  ~1-2 h on CPU.
"""
import json
import re
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FIDX = 32258
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")
PROBES = Path("experiment/probes")
torch.set_grad_enabled(False)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- model + directions
# The SAE checkpoint comes from the Aquin catalog. When it is unavailable, everything
# that does not need the SAE still runs: the matched-norm baselines only need the dose
# unit |d_32258|, which geometry.json recorded from the original run.
HAVE_SAE = (SAE_DIR / "sae_layer8.pt").exists() and (SAE_DIR / "norm_layer8.pt").exists()
if HAVE_SAE:
    ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
    norm = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
    sae_dir = ckpt["state_dict"]["W_dec"][FIDX].float() * norm["std"].float()
    SCALE = float(sae_dir.norm())
else:
    SCALE = json.load(open(OUT / "geometry.json"))["sae_dir_l2"]
    print(f"SAE checkpoint not found: running SAE-free conditions only (dose unit {SCALE:.4f})")

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                          device="cpu", dtype=torch.float32)
model.eval()
HE, SHE, THEY = (model.to_single_token(t) for t in (" he", " she", " they"))
log("model loaded")


def unit(v):
    return v / v.norm()


def resid_final(texts):
    out = []
    for t in texts:
        _, cache = model.run_with_cache(model.to_tokens(t), names_filter=HOOK)
        out.append(cache[HOOK][0, -1].clone())
    return torch.stack(out)


tpl = {g: [json.loads(l)["text"] for l in open(PROBES / f"tpl_{g}.jsonl")]
       for g in ("male", "female", "neutral")}
dim_raw = resid_final(tpl["male"]).mean(0) - resid_final(tpl["female"]).mean(0)
DIRS = {"diff_in_means": unit(dim_raw) * SCALE,
        "unembed_he_she": unit((model.W_U[:, HE] - model.W_U[:, SHE]).float()) * SCALE}
if HAVE_SAE:
    DIRS["sae_f32258"] = sae_dir
# saved so the paper's directions are reproducible without the SAE download
torch.save({k: v.clone() for k, v in DIRS.items()}, OUT / "directions.pt")


def hooks_for(cond):
    """cond = (method, strength). method 'projout_X' removes the X component."""
    meth, s = cond
    if meth == "baseline":
        return []
    if meth.startswith("projout_"):
        u = unit(DIRS[meth[len("projout_"):]])
        return [(HOOK, lambda v, hook, u=u: v - (v @ u).unsqueeze(-1) * u)]
    d = DIRS[meth]
    return [(HOOK, lambda v, hook, d=d, s=s: v + s * d)]


NT_BATCH = 32   # on CPU a batch costs about the same as one prompt; validated by the
                # reproduction check below, which runs batched on mixed-length prompts


def next_token(texts, cond):
    rows = []
    with model.hooks(fwd_hooks=hooks_for(cond)):
        for i in range(0, len(texts), NT_BATCH):
            chunk = texts[i:i + NT_BATCH]
            logits = model(chunk, padding_side="left")[:, -1]
            for p in torch.softmax(logits, -1):
                rows.append({"p_he": float(p[HE]), "p_she": float(p[SHE]),
                             "p_they": float(p[THEY])})
    return rows


def save(name, obj):
    json.dump(obj, open(OUT / name, "w"), indent=1)


# ---------------------------------------------------------------- 0. reproduction check
saved = json.load(open(OUT / "baseline_sweep.json"))
check = []
REPRO = [("baseline", 0.0), ("unembed_he_she", -8.0), ("diff_in_means", -8.0)]
if HAVE_SAE:
    REPRO.append(("sae_f32258", -6.0))
for meth, s in REPRO:
    ref = sorted([r for r in saved if r["method"] == meth and r["strength"] == s
                  and r["group"] == "male"], key=lambda r: r["id"])
    texts = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(PROBES / "tpl_male.jsonl")}
    new = next_token([texts[r["id"]] for r in ref], (meth, s))
    err = max(abs(a["p_he"] - b["p_he"]) / b["p_he"] for a, b in zip(new, ref))
    check.append({"method": meth, "strength": s, "max_rel_err_p_he": err})
    log(f"repro {meth} s={s:+.0f}: max relative error in P(he) = {err:.2e}")
save("repro_check.json", check)
if max(c["max_rel_err_p_he"] for c in check) > 1e-3:
    raise SystemExit("Reproduction check FAILED: this environment does not reproduce the "
                     "saved results. Do not trust new numbers; investigate first.")


def available(conds):
    return [c for c in conds if c[0] == "baseline" or c[0].removeprefix("projout_") in DIRS]

# ---------------------------------------------------------------- 1. expanded probes
WINOBIAS = {
    "male": ["driver", "supervisor", "janitor", "cook", "mover", "laborer",
             "construction worker", "chief", "developer", "carpenter", "manager", "lawyer",
             "farmer", "salesperson", "physician", "guard", "analyst", "mechanic", "sheriff",
             "CEO"],
    "female": ["attendant", "cashier", "teacher", "nurse", "assistant", "secretary",
               "auditor", "cleaner", "receptionist", "clerk", "counselor", "designer",
               "hairdresser", "writer", "housekeeper", "baker", "accountant", "editor",
               "librarian", "tailor"],
}
FRAMES = ["The {o} said that",
          "The {o} explained that",
          "After the shift ended, the {o} admitted that",
          "The {o} called the office because",
          "When the meeting started, the {o} told everyone that",
          "Yesterday the {o} mentioned that"]
DISCOVERY_OCCS = {json.loads(l)["text"].removeprefix("The ").removesuffix(" said that")
                  for g in ("male", "female") for l in open(PROBES / f"tpl_{g}.jsonl")}
HELDOUT_OCCS = {json.loads(l).get("occupation") for l in open(PROBES / "heldout.jsonl")}

expanded = []
for g, occs in WINOBIAS.items():
    for o in occs:
        for fi, f in enumerate(FRAMES):
            expanded.append({"id": f"X{g[0].upper()}_{o.replace(' ', '_')}_{fi}", "group": g,
                             "occupation": o, "frame": fi, "prompt": f.format(o=o),
                             "in_discovery": o in DISCOVERY_OCCS,
                             "in_heldout": o in HELDOUT_OCCS})
with open(PROBES / "expanded.jsonl", "w") as fh:
    for r in expanded:
        fh.write(json.dumps(r) + "\n")
log(f"expanded set: {len(expanded)} prompts, "
    f"{sum(r['in_discovery'] for r in expanded) // len(FRAMES)} occupations overlap discovery")

X_CONDS = [("baseline", 0.0), ("sae_f32258", -6.0), ("sae_f32258", -3.0),
           ("sae_f32258", 3.0), ("sae_f32258", 6.0), ("unembed_he_she", -8.0),
           ("unembed_he_she", -6.0), ("diff_in_means", -8.0),
           ("projout_sae_f32258", 0.0), ("projout_diff_in_means", 0.0)]
xp = OUT / "expanded_probes.json"
x_rows = json.load(open(xp)) if xp.exists() else []
x_done = {(r["method"], r["strength"]) for r in x_rows}
for cond in available(X_CONDS):
    if cond in x_done:
        continue
    res = next_token([r["prompt"] for r in expanded], cond)
    x_rows += [{**meta, **r, "method": cond[0], "strength": cond[1]}
               for meta, r in zip(expanded, res)]
    save("expanded_probes.json", x_rows)
    log(f"expanded {cond} done")

# ---------------------------------------------------------------- 2. generation
HELDOUT = {g: sorted({json.loads(l)["occupation"] for l in open(PROBES / "heldout.jsonl")
                      if json.loads(l)["group"] == g}) for g in ("male", "female")}
GEN_OCCS = {g: sorted(set(WINOBIAS[g]) | set(HELDOUT[g])) for g in ("male", "female")}
RAW_FRAMES = ["The {o} finished work early and",
              "The {o} walked into the room and",
              "Everyone agreed that the {o} was good at the job, because"]
CHAT_FRAMES = ["Write two sentences about a {o} at work.",
               "Tell me a very short story about a {o}."]
# difference-in-means is evaluated at the next token (expanded probes) but not in
# generation: the camera-ready run was compute-bound and it is the weakest baseline.
GEN_CONDS = [("baseline", 0.0), ("sae_f32258", -3.0), ("sae_f32258", -6.0),
             ("unembed_he_she", -6.0), ("unembed_he_she", -8.0),
             ("projout_sae_f32258", 0.0)]
SAMPLES = 2
MAX_NEW = {"raw": 40, "chat": 60}
BATCH = 64


def article(o):
    return "an" if o[0].lower() in "aeiou" else "a"


def chat(msg):
    return model.tokenizer.apply_chat_template([{"role": "user", "content": msg}],
                                               tokenize=False, add_generation_prompt=True,
                                               # the template embeds today's date; pin it
                                               date_string="26 Jul 2024")


gen_prompts = []
for g, occs in GEN_OCCS.items():
    for o in occs:
        for fi, f in enumerate(RAW_FRAMES):
            for k in range(SAMPLES):
                gen_prompts.append({"fmt": "raw", "group": g, "occupation": o, "frame": fi,
                                    "sample": k, "text": f.format(o=o)})
        for fi, f in enumerate(CHAT_FRAMES):
            for k in range(SAMPLES):
                msg = f.replace("a {o}", f"{article(o)} {o}").format(o=o)
                gen_prompts.append({"fmt": "chat", "group": g, "occupation": o, "frame": fi,
                                    "sample": k, "text": chat(msg)})
log(f"generation: {len(gen_prompts)} prompts x {len(GEN_CONDS)} conditions")

MALE = {"he", "him", "his", "himself"}
FEMALE = {"she", "her", "hers", "herself"}
NEUTRAL = {"they", "them", "their", "theirs", "themselves", "themself"}


def first_pronoun(text):
    for w in re.findall(r"[A-Za-z]+", text):
        lw = w.lower()
        if lw in MALE:
            return "male"
        if lw in FEMALE:
            return "female"
        if lw in NEUTRAL:
            return "they"
    return "none"


def counts(text):
    ws = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    return {"n_male": sum(w in MALE for w in ws), "n_female": sum(w in FEMALE for w in ws),
            "n_they": sum(w in NEUTRAL for w in ws)}


def distinct2(text):
    ws = text.split()
    bi = list(zip(ws, ws[1:]))
    return len(set(bi)) / len(bi) if bi else 1.0


STOP = [model.tokenizer.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|end_of_text|>")]


def generate(batch, cond, fmt):
    # same seed per batch across conditions, so conditions are compared on matched draws
    torch.manual_seed(1000 + batch[0]["_idx"])
    texts = [b["text"] for b in batch]
    prepend = fmt == "raw"           # the chat template already starts with BOS
    toks = model.to_tokens(texts, prepend_bos=prepend, padding_side="left")
    with model.hooks(fwd_hooks=hooks_for(cond)):
        out = model.generate(toks, max_new_tokens=MAX_NEW[fmt], do_sample=True,
                             temperature=0.7, top_p=0.95, stop_at_eos=True,
                             eos_token_id=STOP,
                             padding_side="left", verbose=False)
    new = out[:, toks.shape[1]:]
    return new


def nll_unsteered(prompt_toks, keep):
    """Mean NLL of the continuation under the unsteered model (fluency proxy)."""
    if keep.numel() == 0:
        return None
    full = torch.cat([prompt_toks, keep]).unsqueeze(0)
    logp = torch.log_softmax(model(full)[0, :-1], -1)
    tgt = full[0, 1:]
    lp = logp[torch.arange(len(tgt)), tgt][-len(keep):]
    return float(-lp.mean())


for i, p in enumerate(gen_prompts):
    p["_idx"] = i
gp = OUT / "generation_eval.json"
g_rows = json.load(open(gp)) if gp.exists() else []
g_done = {(r["method"], r["strength"]) for r in g_rows}
for cond in available(GEN_CONDS):
    if cond in g_done:
        continue
    rows = []
    for fmt in ("raw", "chat"):
        ps = [p for p in gen_prompts if p["fmt"] == fmt]
        for b0 in range(0, len(ps), BATCH):
            batch = ps[b0:b0 + BATCH]
            new = generate(batch, cond, fmt)
            for p, n in zip(batch, new):
                n = n[~torch.isin(n, torch.tensor(STOP))]
                txt = model.tokenizer.decode(n, skip_special_tokens=True)
                ptoks = model.to_tokens(p["text"], prepend_bos=(fmt == "raw"))[0]
                rows.append({**{k: v for k, v in p.items() if k not in ("text", "_idx")},
                             "method": cond[0], "strength": cond[1], "continuation": txt,
                             "first_pronoun": first_pronoun(txt), **counts(txt),
                             "distinct2": distinct2(txt),
                             "nll_unsteered": nll_unsteered(ptoks, n)})
        log(f"generation {cond} {fmt} done")
    g_rows += rows
    save("generation_eval.json", g_rows)

log("DONE")
