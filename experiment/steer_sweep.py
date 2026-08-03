"""Dose-response steering sweep: add scaled SAE decoder directions to the
layer-8 residual stream and measure P(' he') vs P(' she') at the final position.
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
SAE_PATH = str(Path.home() / ".aquin/sae/llama-3.2-1b/sae_layer8.pt")
FEATURES = {"he_stereo_f32258": 32258, "she_stereo_f27420": 27420}
STRENGTHS = [-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0]

ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
W_dec = ckpt["state_dict"]["W_dec"].float()
n = torch.load(SAE_PATH.replace("sae_layer8", "norm_layer8"), map_location="cpu", weights_only=False)
norm_std = n["std"].float()

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)
HE, SHE = model.to_single_token(" he"), model.to_single_token(" she")

probes = []
for g in ("female", "male"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        probes.append({"group": g, "text": r["text"]})
for r in map(json.loads, open("experiment/probes/probes.jsonl")):
    if r["group"] == "D":
        probes.append({"group": "D_" + r["context_gender"], "text": r["prompt"]})

results = []
for fname, fidx in FEATURES.items():
    direction = W_dec[fidx] * norm_std  # map from normalized space back to raw resid space
    for s in STRENGTHS:
        def hook(value, hook, s=s, d=direction):
            return value + s * d
        gaps = {}
        for p in probes:
            toks = model.to_tokens(p["text"])
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(HOOK, hook)] if s != 0 else []):
                    logits = model(toks)
            probs = torch.softmax(logits[0, -1], dim=-1)
            results.append({"feature": fname, "strength": s, "group": p["group"],
                            "text": p["text"], "p_he": float(probs[HE]), "p_she": float(probs[SHE])})
        done = [r for r in results if r["feature"] == fname and r["strength"] == s]
        for g in ("female", "male", "D_female", "D_male"):
            rows = [r for r in done if r["group"] == g]
            if rows:
                mhe = sum(r["p_he"] for r in rows) / len(rows)
                mshe = sum(r["p_she"] for r in rows) / len(rows)
                gaps[g] = f"he={mhe:.3f} she={mshe:.3f}"
        print(f"{fname} s={s:+.0f}  " + "  ".join(f"[{g}] {v}" for g, v in gaps.items()), flush=True)

json.dump(results, open("experiment/results/steer_sweep.json", "w"), indent=1)
print("saved experiment/results/steer_sweep.json")
