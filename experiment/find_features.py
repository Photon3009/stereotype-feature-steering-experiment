"""Find SAE features that separate male/female-stereotyped occupations at the
final prompt position, using Aquin's downloaded model + SAE artifacts directly.

Runs on CPU (TransformerLens warns MPS may be silently incorrect on torch 2.7).
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
SAE_PATH = str(Path.home() / ".aquin/sae/llama-3.2-1b/sae_layer8.pt")

ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
b_pre, W_enc, b_enc = sd["b_pre"].float(), sd["W_enc"].float(), sd["b_enc"].float()

NORM_PATH = SAE_PATH.replace("sae_layer8.pt", "norm_layer8.pt")
try:
    n = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
    norm_mean, norm_std = n["mean"].float(), n["std"].float()
    print(f"using norm stats from {NORM_PATH}")
except FileNotFoundError:
    norm_mean, norm_std = torch.zeros(2048), torch.ones(2048)
    print("WARNING: no norm stats — running unnormalized")

def sae_encode(x):
    x = (x - norm_mean) / norm_std
    return torch.relu((x - b_pre) @ W_enc + b_enc)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)

def final_pos_feats(text):
    toks = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=f"blocks.{LAYER}.hook_resid_post")
    resid = cache[f"blocks.{LAYER}.hook_resid_post"][0, -1]
    return sae_encode(resid)

groups = {}
for g in ("female", "male", "neutral"):
    rows = [json.loads(l) for l in open(f"experiment/probes/tpl_{g}.jsonl")]
    acts = torch.stack([final_pos_feats(r["text"]) for r in rows])
    groups[g] = acts
    print(f"{g}: {acts.shape}, mean L0={float((acts > 0).float().sum(-1).mean()):.0f}")

mF, mM, mN = (groups[g].mean(0) for g in ("female", "male", "neutral"))

def report(name, delta, base):
    top = torch.argsort(delta, descending=True)[:10]
    print(f"\n=== {name} ===")
    for i in top:
        i = int(i)
        print(f"f{i}: delta={float(delta[i]):.3f}  gendered={float(base[i]):.3f}  neutral={float(mN[i]):.3f}")

report("female-stereotype candidates (F - N)", mF - mN, mF)
report("male-stereotype candidates (M - N)", mM - mN, mM)
report("F - M direct contrast", mF - mM, mF)

out = {
    "female_minus_neutral": [[int(i), float((mF - mN)[i])] for i in torch.argsort(mF - mN, descending=True)[:20]],
    "male_minus_neutral": [[int(i), float((mM - mN)[i])] for i in torch.argsort(mM - mN, descending=True)[:20]],
    "female_minus_male": [[int(i), float((mF - mM)[i])] for i in torch.argsort(mF - mM, descending=True)[:20]],
    "male_minus_female": [[int(i), float((mM - mF)[i])] for i in torch.argsort(mM - mF, descending=True)[:20]],
}
json.dump(out, open("experiment/results/final_pos_contrast.json", "w"), indent=1)
print("\nsaved experiment/results/final_pos_contrast.json")
