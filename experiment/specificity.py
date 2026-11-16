"""Specificity controls and two extra baselines.

Four questions a referee will ask that the sweep does not answer:

  1. POSITION. The sweep adds the direction at every token position, so the result is
     compatible with "we injected a global he-prior" rather than "we changed how the
     occupation is represented". We decompose: all positions / final position only /
     all positions except the final one.
  2. LAYER. Is layer 8 special, or would injecting this vector anywhere do the same?
     We inject the same direction at layers 2..14.
  3. READOUT VALIDITY. P(' he') = 0.18 means 82% of the mass is elsewhere. We report
     the top-5 next tokens per group and the share of mass on {' he',' she',' they'},
     so a reader can judge whether the metric tracks what the model actually does.
  4. A PROMPTING BASELINE. Before adding an interpretability method, does simply
     asking the model to avoid gendered assumptions work?

Outputs results/specificity.json
"""
import json
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

FIDX = 32258
LAYER = 8
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
nrm = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
sae_dir = ckpt["state_dict"]["W_dec"][FIDX].float() * nrm["std"].float()

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32
)
HE, SHE, THEY = (model.to_single_token(t) for t in (" he", " she", " they"))
log("model loaded")

probes = []
for g in ("male", "female", "neutral"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        probes.append({"group": g, "id": r["id"], "text": r["text"]})

results = {"position": [], "layer": [], "readout": [], "prompting": []}


def measure(hook_list, prompt_prefix=""):
    rows = []
    for p in probes:
        toks = model.to_tokens(prompt_prefix + p["text"])
        with torch.no_grad(), model.hooks(fwd_hooks=hook_list):
            logits = model(toks)
        probs = torch.softmax(logits[0, -1], dim=-1)
        rows.append({"group": p["group"], "id": p["id"],
                     "p_he": float(probs[HE]), "p_she": float(probs[SHE]),
                     "p_they": float(probs[THEY])})
    return rows


def brief(rows):
    out = []
    for g in ("male", "female", "neutral"):
        sel = [r for r in rows if r["group"] == g]
        mhe = sum(r["p_he"] for r in sel) / len(sel)
        mshe = sum(r["p_she"] for r in sel) / len(sel)
        out.append(f"[{g}] he={mhe:.3f} she={mshe:.3f}")
    return "  ".join(out)


# ------------------------------------------------------------------ 1. position
HOOK8 = f"blocks.{LAYER}.hook_resid_post"

def mk_pos_hook(mode, s):
    def hook(value, hook, s=s, mode=mode):
        v = value.clone()
        if mode == "all":
            v = v + s * sae_dir
        elif mode == "final_only":
            v[:, -1, :] = v[:, -1, :] + s * sae_dir
        elif mode == "except_final":
            v[:, :-1, :] = v[:, :-1, :] + s * sae_dir
        return v
    return [(HOOK8, hook)]


for mode in ("all", "final_only", "except_final"):
    for s in (-6.0, 0.0, 6.0):
        rows = measure([] if s == 0 else mk_pos_hook(mode, s))
        results["position"] += [{**r, "mode": mode, "strength": s} for r in rows]
        log(f"POSITION {mode:13s} s={s:+.0f}  {brief(rows)}")

# ------------------------------------------------------------------ 2. layer
for layer in (2, 4, 6, 8, 10, 12, 14):
    hk = f"blocks.{layer}.hook_resid_post"
    for s in (-6.0, 6.0):
        def hook(value, hook, s=s):
            return value + s * sae_dir
        rows = measure([(hk, hook)])
        results["layer"] += [{**r, "layer": layer, "strength": s} for r in rows]
        log(f"LAYER {layer:2d} s={s:+.0f}  {brief(rows)}")

# ------------------------------------------------------------------ 3. readout
for p in probes:
    toks = model.to_tokens(p["text"])
    with torch.no_grad():
        logits = model(toks)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 5)
    results["readout"].append({
        "group": p["group"], "id": p["id"], "text": p["text"],
        "top5": [[model.to_string(torch.tensor([i])), float(v)]
                 for v, i in zip(top.values, top.indices)],
        "pronoun_mass": float(probs[HE] + probs[SHE] + probs[THEY]),
        "he_rank": int((probs > probs[HE]).sum()) + 1,   # 1-indexed
        "she_rank": int((probs > probs[SHE]).sum()) + 1,
    })
for g in ("male", "female", "neutral"):
    sel = [r for r in results["readout"] if r["group"] == g]
    log(f"READOUT [{g}] mean pronoun mass={sum(r['pronoun_mass'] for r in sel)/len(sel):.3f}  "
        f"median he-rank={sorted(r['he_rank'] for r in sel)[len(sel)//2]}  "
        f"median she-rank={sorted(r['she_rank'] for r in sel)[len(sel)//2]}  "
        f"top1 examples={[r['top5'][0][0] for r in sel[:4]]}")

# ------------------------------------------------------------------ 4. prompting
PREFIXES = {
    "none": "",
    "neutral_instruction": ("Use gender-neutral language and do not assume a person's "
                            "gender from their occupation.\n\n"),
    "strong_instruction": ("Important: never infer gender from occupation. If gender is "
                           "not stated, use the singular pronoun \"they\".\n\n"),
}
for name, pref in PREFIXES.items():
    rows = measure([], prompt_prefix=pref)
    results["prompting"] += [{**r, "prefix": name} for r in rows]
    log(f"PROMPT {name:20s}  {brief(rows)}")

json.dump(results, open(OUT / "specificity.json", "w"), indent=1)
log("saved specificity.json")
