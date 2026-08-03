"""Rank SAE features by (a) how strongly their decoder direction pushes
' he' vs ' she' through the unembedding, and (b) whether they fire on our
occupation prompts. The intersection = stereotype steering candidates.
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
SAE_PATH = str(Path.home() / ".aquin/sae/llama-3.2-1b/sae_layer8.pt")

ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
b_pre, W_enc, b_enc, W_dec = sd["b_pre"].float(), sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float()
n = torch.load(SAE_PATH.replace("sae_layer8", "norm_layer8"), map_location="cpu", weights_only=False)
norm_mean, norm_std = n["mean"].float(), n["std"].float()

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)

he = model.to_single_token(" he")
she = model.to_single_token(" she")
gender_dir = model.W_U[:, he] - model.W_U[:, she]  # [d_model]
# decoder directions live in normalized space; map back through std before unembedding
proj = (W_dec * norm_std) @ gender_dir  # [n_features] >0 pushes ' he', <0 pushes ' she'

def sae_encode(x):
    x = (x - norm_mean) / norm_std
    return torch.relu((x - b_pre) @ W_enc + b_enc)

def final_pos(texts):
    outs = []
    for t in texts:
        toks = model.to_tokens(t)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=f"blocks.{LAYER}.hook_resid_post")
        outs.append(sae_encode(cache[f"blocks.{LAYER}.hook_resid_post"][0, -1]))
    return torch.stack(outs)

acts = {}
for g in ("female", "male", "neutral"):
    rows = [json.loads(l) for l in open(f"experiment/probes/tpl_{g}.jsonl")]
    acts[g] = final_pos([r["text"] for r in rows])

mF, mM, mN = acts["female"].mean(0), acts["male"].mean(0), acts["neutral"].mean(0)
active = (mF + mM + mN) / 3

# effect at the logit = activation * decoder projection
he_push = active * proj
she_push = active * (-proj)

print("=== active features pushing ' he' (activation x projection) ===")
for i in torch.argsort(he_push, descending=True)[:8]:
    i = int(i)
    print(f"f{i}: push={float(he_push[i]):+.3f}  act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f}  proj={float(proj[i]):+.3f}")

print("\n=== active features pushing ' she' ===")
for i in torch.argsort(she_push, descending=True)[:8]:
    i = int(i)
    print(f"f{i}: push={float(she_push[i]):+.3f}  act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f}  proj={float(proj[i]):+.3f}")

# the ideal stereotype feature: pushes she AND fires more on female occupations (or he/male)
stereo_f = (-proj) * (mF - mN)
stereo_m = proj * (mM - mN)
print("\n=== STEREOTYPE candidates: she-pushing x (F-N firing) ===")
for i in torch.argsort(stereo_f, descending=True)[:8]:
    i = int(i)
    print(f"f{i}: score={float(stereo_f[i]):.4f}  act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f}  proj={float(proj[i]):+.3f}")
print("\n=== STEREOTYPE candidates: he-pushing x (M-N firing) ===")
for i in torch.argsort(stereo_m, descending=True)[:8]:
    i = int(i)
    print(f"f{i}: score={float(stereo_m[i]):.4f}  act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f}  proj={float(proj[i]):+.3f}")

json.dump({
    "she_stereotype": [[int(i), float(stereo_f[i])] for i in torch.argsort(stereo_f, descending=True)[:15]],
    "he_stereotype": [[int(i), float(stereo_m[i])] for i in torch.argsort(stereo_m, descending=True)[:15]],
}, open("experiment/results/stereotype_candidates.json", "w"), indent=1)
print("\nsaved experiment/results/stereotype_candidates.json")
