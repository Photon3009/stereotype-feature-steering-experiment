"""Robustness: f32258 steering sweep on the varied-predicate probes (Groups A/B),
which use different sentence structures than the 'said that' templates.
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FIDX = 32258
STRENGTHS = [-6.0, -3.0, 0.0, 3.0, 6.0]
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"

ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
n = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
direction = ckpt["state_dict"]["W_dec"][FIDX].float() * n["std"].float()

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)
HE, SHE = model.to_single_token(" he"), model.to_single_token(" she")

probes = [r for r in map(json.loads, open("experiment/probes/probes.jsonl")) if r["group"] in ("A", "B")]

results = []
for s in STRENGTHS:
    def hook(value, hook, s=s):
        return value + s * direction
    for p in probes:
        toks = model.to_tokens(p["prompt"])
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(HOOK, hook)] if s != 0 else []):
                logits = model(toks)
        probs = torch.softmax(logits[0, -1], dim=-1)
        results.append({"strength": s, "group": p["group"], "id": p["id"],
                        "p_he": float(probs[HE]), "p_she": float(probs[SHE])})
    for g in ("A", "B"):
        rows = [r for r in results if r["strength"] == s and r["group"] == g]
        mhe = sum(r["p_he"] for r in rows) / len(rows)
        mshe = sum(r["p_she"] for r in rows) / len(rows)
        print(f"s={s:+.0f} [{'male-occ A' if g == 'A' else 'female-occ B'}] he={mhe:.3f} she={mshe:.3f}", flush=True)

json.dump(results, open("experiment/results/robustness_sweep.json", "w"), indent=1)
print("saved experiment/results/robustness_sweep.json")
