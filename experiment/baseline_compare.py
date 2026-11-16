"""Baseline comparison + held-out generalization + ablation, for the paper.

Adds the three things the blog post did not have:

  1. BASELINES at matched intervention norm. The SAE feature direction is compared
     against (a) difference-in-means between male- and female-stereotyped activations,
     (b) the raw unembedding gender direction W_U[:,' he'] - W_U[:,' she'], and
     (c) random directions. All rescaled to the same L2 norm as the SAE direction, so
     "strength s" means the same size perturbation for every method.
  2. HELD-OUT OCCUPATIONS. probes/heldout.jsonl uses 24 occupations that appear
     nowhere in discovery, under two template families unseen in discovery.
  3. PROJECTION ABLATION. Instead of adding a direction, remove the component along
     it (x -> x - (x.u)u) to ask whether the direction is *necessary*, i.e. whether
     the stereotype survives its removal (redundancy test).

Also records P(' they') everywhere, so we can distinguish "the bias was removed"
from "pronouns were suppressed".

Outputs: results/baseline_sweep.json, results/heldout_sweep.json,
         results/ablation.json, results/capability_baselines.json
"""
import json
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FIDX = 32258
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")

MAIN_STRENGTHS = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
# the SAE feature gets the full curve; the baselines only need the strengths that the
# frontier and the dose-response comparison actually use. Trimmed after the first run
# showed this machine is memory-bound, not compute-bound.
BASELINE_STRENGTHS = [-8.0, -6.0, -4.0, -2.0, 3.0, 6.0]
HELDOUT_STRENGTHS = [-6.0, -3.0, 3.0, 6.0]
CAP_STRENGTHS = [-8.0, -6.0, -4.0, -2.0]
RANDOM_SEEDS = [0, 1, 2]
RANDOM_STRENGTHS = [-6.0, 6.0]

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- model + SAE
ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
norm = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
W_dec = ckpt["state_dict"]["W_dec"].float()
norm_std = norm["std"].float()

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32
)
HE = model.to_single_token(" he")
SHE = model.to_single_token(" she")
THEY = model.to_single_token(" they")
log("model loaded")

# ---------------------------------------------------------------- probes
tpl = []
for g in ("male", "female", "neutral"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        tpl.append({"set": "tpl", "group": g, "id": r["id"], "text": r["text"]})
for r in map(json.loads, open("experiment/probes/probes.jsonl")):
    if r["group"] == "D":
        tpl.append({"set": "tpl", "group": "D_" + r["context_gender"],
                    "id": r["id"], "text": r["prompt"]})

heldout = [{"set": "heldout", "group": r["group"], "id": r["id"], "text": r["prompt"]}
           for r in map(json.loads, open("experiment/probes/heldout.jsonl"))]

log(f"{len(tpl)} main probes, {len(heldout)} held-out probes")


# ---------------------------------------------------------------- directions
def unit(v):
    return v / v.norm()


sae_dir = W_dec[FIDX] * norm_std          # the reference direction (raw resid space)
SCALE = float(sae_dir.norm())             # dose unit: |s * sae_dir| = |s| * SCALE


def resid_final(texts):
    outs = []
    for t in texts:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(t), names_filter=HOOK)
        outs.append(cache[HOOK][0, -1].clone())
    return torch.stack(outs)


male_acts = resid_final([p["text"] for p in tpl if p["group"] == "male"])
female_acts = resid_final([p["text"] for p in tpl if p["group"] == "female"])
dim_raw = male_acts.mean(0) - female_acts.mean(0)
log(f"diff-in-means computed, |dim| = {float(dim_raw.norm()):.2f}")

unembed_raw = model.W_U[:, HE] - model.W_U[:, SHE]

DIRECTIONS = {
    "sae_f32258": sae_dir,
    "diff_in_means": unit(dim_raw) * SCALE,
    "unembed_he_she": unit(unembed_raw.float()) * SCALE,
}
for seed in RANDOM_SEEDS:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(model.cfg.d_model, generator=g)
    DIRECTIONS[f"random_s{seed}"] = unit(r) * SCALE

# reference scale: how big is the perturbation relative to a typical residual?
resid_norm = float(torch.cat([male_acts, female_acts]).norm(dim=-1).mean())
cos_sae_dim = float(torch.nn.functional.cosine_similarity(sae_dir, dim_raw, dim=0))
cos_sae_unembed = float(torch.nn.functional.cosine_similarity(sae_dir, unembed_raw.float(), dim=0))
geometry = {
    "sae_dir_l2": SCALE,
    "dim_l2_raw": float(dim_raw.norm()),
    "mean_resid_l2_at_L8": resid_norm,
    "cos_sae_vs_diff_in_means": cos_sae_dim,
    "cos_sae_vs_unembed": cos_sae_unembed,
    "perturbation_fraction_of_resid_norm_per_unit_strength": SCALE / resid_norm,
}
log(f"geometry: {geometry}")
json.dump(geometry, open(OUT / "geometry.json", "w"), indent=1)


# ---------------------------------------------------------------- measurement
def add_hook(direction, s):
    def hook(value, hook, s=s, d=direction):
        return value + s * d
    return [(HOOK, hook)]


def proj_out_hook(direction):
    u = unit(direction)

    def hook(value, hook, u=u):
        return value - (value @ u).unsqueeze(-1) * u
    return [(HOOK, hook)]


def measure(probes, fwd_hooks, tag):
    rows = []
    for p in probes:
        toks = model.to_tokens(p["text"])
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            logits = model(toks)
        probs = torch.softmax(logits[0, -1], dim=-1)
        rows.append({**tag, "set": p["set"], "group": p["group"], "id": p["id"],
                     "p_he": float(probs[HE]), "p_she": float(probs[SHE]),
                     "p_they": float(probs[THEY])})
    return rows


def summarize(rows, groups=("male", "female")):
    out = []
    for g in groups:
        sel = [r for r in rows if r["group"] == g]
        if not sel:
            continue
        mhe = sum(r["p_he"] for r in sel) / len(sel)
        mshe = sum(r["p_she"] for r in sel) / len(sel)
        mthey = sum(r["p_they"] for r in sel) / len(sel)
        out.append(f"[{g}] he={mhe:.3f} she={mshe:.3f} they={mthey:.4f}")
    return "  ".join(out)


# ---------------------------------------------------------------- 1. main sweep
def resume(fname):
    p = OUT / fname
    if not p.exists():
        return [], set()
    rows = json.load(open(p))
    done = {(r["method"], r["strength"]) for r in rows}
    log(f"resuming {fname}: {len(rows)} rows, {len(done)} (method, strength) cells done")
    return rows, done


main_rows, done = resume("baseline_sweep.json")
if ("baseline", 0.0) not in done:
    main_rows += measure(tpl, [], {"method": "baseline", "strength": 0.0})
    log("s=0 " + summarize(main_rows))

for name, d in DIRECTIONS.items():
    if name.startswith("random"):
        strengths = RANDOM_STRENGTHS
    elif name == "sae_f32258":
        strengths = MAIN_STRENGTHS
    else:
        strengths = BASELINE_STRENGTHS
    for s in strengths:
        if (name, s) in done:
            continue
        rows = measure(tpl, add_hook(d, s), {"method": name, "strength": s})
        main_rows += rows
        log(f"{name} s={s:+.0f}  " + summarize(rows))
        json.dump(main_rows, open(OUT / "baseline_sweep.json", "w"), indent=1)
json.dump(main_rows, open(OUT / "baseline_sweep.json", "w"), indent=1)
log("saved baseline_sweep.json")

# ---------------------------------------------------------------- 2. held-out
held_rows, hdone = resume("heldout_sweep.json")
if ("baseline", 0.0) not in hdone:
    held_rows += measure(heldout, [], {"method": "baseline", "strength": 0.0})
    log("HELDOUT s=0 " + summarize(held_rows))
for name in ("sae_f32258", "diff_in_means", "unembed_he_she"):
    for s in HELDOUT_STRENGTHS:
        if (name, s) in hdone:
            continue
        rows = measure(heldout, add_hook(DIRECTIONS[name], s), {"method": name, "strength": s})
        held_rows += rows
        log(f"HELDOUT {name} s={s:+.0f}  " + summarize(rows))
        json.dump(held_rows, open(OUT / "heldout_sweep.json", "w"), indent=1)
json.dump(held_rows, open(OUT / "heldout_sweep.json", "w"), indent=1)
log("saved heldout_sweep.json")

# ---------------------------------------------------------------- 3. ablation
abl_rows = measure(tpl, [], {"method": "no_ablation", "strength": 0.0})
abl_rows += measure(heldout, [], {"method": "no_ablation", "strength": 0.0})
for name in ("sae_f32258", "diff_in_means"):
    abl_rows += measure(tpl, proj_out_hook(DIRECTIONS[name]), {"method": f"projout_{name}", "strength": 0.0})
    abl_rows += measure(heldout, proj_out_hook(DIRECTIONS[name]), {"method": f"projout_{name}", "strength": 0.0})
    sel = [r for r in abl_rows if r["method"] == f"projout_{name}" and r["set"] == "tpl"]
    log(f"PROJ-OUT {name}  tpl: " + summarize(sel))
json.dump(abl_rows, open(OUT / "ablation.json", "w"), indent=1)
log("saved ablation.json")

# ---------------------------------------------------------------- 4. capability
from datasets import load_dataset  # noqa: E402

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
texts = []
for row in ds:
    t = row["text"].strip()
    if len(t) > 300:
        texts.append(t[:800])
    if len(texts) >= 15:
        break

QA = [
    ("The capital of France is", "paris"),
    ("The chemical symbol for gold is", "au"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in our solar system is", "jupiter"),
    ("The author of Romeo and Juliet is William", "shakespeare"),
    ("The square root of 64 is", "8"),
    ("The opposite of hot is", "cold"),
    ("The first month of the year is", "january"),
    ("Photosynthesis converts sunlight into", "energy"),
    ("The currency of Japan is the", "yen"),
]


def wikitext_loss(fwd_hooks):
    losses = []
    for t in texts:
        toks = model.to_tokens(t)[:, :128]
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            losses.append(float(model(toks, return_type="loss")))
    return sum(losses) / len(losses)


def qa_score(fwd_hooks):
    correct = 0
    for prompt, answer in QA:
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            out = model.generate(model.to_tokens(prompt), max_new_tokens=8,
                                 temperature=0.0, verbose=False)
        text = model.to_string(out[0, 1:]).lower()
        if answer in text[len(prompt):]:
            correct += 1
    return correct


QA_CONFIGS = {("baseline", 0.0), ("sae_f32258", -6.0), ("diff_in_means", -8.0),
              ("unembed_he_she", -6.0), ("projout_sae_f32258", 0.0)}


def maybe_qa(method, strength, hooks):
    return qa_score(hooks) if (method, strength) in QA_CONFIGS else None


cap = [{"method": "baseline", "strength": 0.0, "wikitext_loss": wikitext_loss([]),
        "qa": qa_score([])}]
log(f"CAP baseline {cap[-1]}")
for name in ("sae_f32258", "diff_in_means", "unembed_he_she", "random_s0"):
    strengths = CAP_STRENGTHS if not name.startswith("random") else [-6.0]
    strengths = [x for x in strengths if x in MAIN_STRENGTHS or name == "sae_f32258"]
    for s in strengths:
        h = add_hook(DIRECTIONS[name], s)
        cap.append({"method": name, "strength": s, "wikitext_loss": wikitext_loss(h),
                    "qa": maybe_qa(name, s, h)})
        log(f"CAP {name} s={s:+.0f} {cap[-1]}")
    json.dump(cap, open(OUT / "capability_baselines.json", "w"), indent=1)

# projection-ablation capability cost too
for name in ("sae_f32258", "diff_in_means"):
    h = proj_out_hook(DIRECTIONS[name])
    cap.append({"method": f"projout_{name}", "strength": 0.0,
                "wikitext_loss": wikitext_loss(h),
                "qa": maybe_qa(f"projout_{name}", 0.0, h)})
    log(f"CAP projout_{name} {cap[-1]}")
json.dump(cap, open(OUT / "capability_baselines.json", "w"), indent=1)
log("saved capability_baselines.json — DONE")
