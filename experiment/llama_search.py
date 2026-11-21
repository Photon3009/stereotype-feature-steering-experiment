"""Two-sided top-k feature search in Llama-3.2-1B-Instruct, mirroring lfm_search.py.

Necessary for a symmetric cross-model claim. In Llama we originally steered the
rank-1 he-side candidate (f32258, which worked) and the rank-1 she-side candidate
(f27420, which was flat) and concluded the model exposes a male-context handle.
In LFM2.5 the rank-1 he-side candidate was flat and we concluded the opposite
polarity. But rank-1 is not the whole ranking: in LFM2.5 the rank-2 he-side
candidate turns out to steer. Unless we search to the same depth in both models,
"mirror-image mechanism" is an artifact of where we stopped looking.

Same protocol as lfm_search.py: rank by the two-test score on each side, steer the
top TOPK of both sides, report the bias swing.

Outputs results/llama_search.json
"""
import json
import math
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")
TOPK = 8   # was 5; matched to the LFM search depth for the camera-ready
STRENGTHS = [-6.0, 6.0]

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
b_pre, W_enc, b_enc, W_dec = (sd[k].float() for k in ("b_pre", "W_enc", "b_enc", "W_dec"))
nrm = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
norm_mean, norm_std = nrm["mean"].float(), nrm["std"].float()

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32
)
HE, SHE, THEY = (model.to_single_token(t) for t in (" he", " she", " they"))
log("model loaded")

probes = []
for g in ("male", "female", "neutral"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        probes.append({"group": g, "id": r["id"], "text": r["text"]})


def sae_encode(x):
    return torch.relu((((x - norm_mean) / norm_std) - b_pre) @ W_enc + b_enc)


def resid_final(text):
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(text), names_filter=HOOK)
    return cache[HOOK][0, -1]


# ---- rank both sides with the same score function used for LFM2.5
gender_dir = model.W_U[:, HE] - model.W_U[:, SHE]
proj = (W_dec * norm_std) @ gender_dir.float()
acts = {}
for g in ("male", "female", "neutral"):
    acts[g] = torch.stack([sae_encode(resid_final(p["text"]))
                           for p in probes if p["group"] == g])
mM, mF, mN = acts["male"].mean(0), acts["female"].mean(0), acts["neutral"].mean(0)
log(f"L0 at final position: " +
    ", ".join(f"{g}={float((acts[g] > 0).float().sum(-1).mean()):.0f}" for g in acts))

score_he = (torch.clamp(proj, min=0) * torch.clamp(mM - mN, min=0)
            * torch.clamp(mM - mF, min=0))
score_she = (torch.clamp(-proj, min=0) * torch.clamp(mF - mN, min=0)
             * torch.clamp(mF - mM, min=0))

cands = {}
for side, score in (("he", score_he), ("she", score_she)):
    idx = torch.argsort(score, descending=True)[:TOPK]
    cands[side] = [{"feature": int(i), "score": float(score[i]),
                    "act_male": float(mM[i]), "act_female": float(mF[i]),
                    "act_neutral": float(mN[i]), "proj": float(proj[i])}
                   for i in idx if float(score[i]) > 0]
    log(f"top {side}-side: " + ", ".join(f"f{c['feature']}(s={c['score']:.5f})"
                                         for c in cands[side]))


def measure(direction=None, s=0.0):
    hooks = []
    if direction is not None and s != 0:
        def hook(value, hook, s=s, d=direction):
            return value + s * d
        hooks = [(HOOK, hook)]
    rows = []
    for p in probes:
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            logits = model(model.to_tokens(p["text"]))
        pr = torch.softmax(logits[0, -1], dim=-1)
        rows.append({"group": p["group"], "id": p["id"], "p_he": float(pr[HE]),
                     "p_she": float(pr[SHE]), "p_they": float(pr[THEY])})
    return rows


def bias_of(rows, g):
    sel = [r for r in rows if r["group"] == g]
    return sum(math.log(r["p_he"]) - math.log(r["p_she"]) for r in sel) / len(sel)


base = measure()
log(f"baseline: male b={bias_of(base,'male'):+.2f} female b={bias_of(base,'female'):+.2f} "
    f"neutral b={bias_of(base,'neutral'):+.2f}")

rows_all = [{**r, "side": "none", "feature": -1, "strength": 0.0} for r in base]
for side in ("he", "she"):
    for c in cands[side]:
        d = W_dec[c["feature"]] * norm_std
        per_s = {}
        for s in STRENGTHS:
            rr = measure(d, s)
            per_s[s] = rr
            rows_all += [{**r, "side": side, "feature": c["feature"], "strength": s}
                         for r in rr]
        c["swing_male"] = bias_of(per_s[max(STRENGTHS)], "male") - bias_of(per_s[min(STRENGTHS)], "male")
        c["swing_female"] = bias_of(per_s[max(STRENGTHS)], "female") - bias_of(per_s[min(STRENGTHS)], "female")
        log(f"{side}-side f{c['feature']}: b-swing male={c['swing_male']:+.2f} "
            f"female={c['swing_female']:+.2f}")
        json.dump({"layer": LAYER, "topk": TOPK, "strengths": STRENGTHS,
                   "candidates": cands, "rows": rows_all},
                  open(OUT / "llama_search.json", "w"), indent=1)
log("saved llama_search.json")
