"""Systematic two-sided feature search in LFM2.5-230M, at both available layers.

The cross-model claim in the paper is that LFM2.5 exposes a *female*-context handle
where Llama exposes a male-context one. As originally run, the male-side half of that
claim was a null result from a single candidate: "the best he-side feature steered
flat." Absence of evidence.

This script replaces that with a search: rank every SAE feature by the two-test score
on each side, steer the top TOPK of both sides, and report the full table. The claim
then becomes "the top-k he-side candidates are all flat, while the top she-side
candidate moves both groups" -- which is a bounded, checkable statement.

Runs at layers 9 and 5 (both public SAEs), which also gives the paper a second layer
in at least one model.

Outputs results/lfm_search.json
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAE_DIR = Path.home() / ".aquin/sae/lfm2.5-230m"
OUT = Path("experiment/results")
LAYERS = [9, 5]
TOPK = 8
STRENGTHS = [-6.0, -3.0, 3.0, 6.0]
D_MODEL = 1024

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-230M", torch_dtype=torch.float32)
model.eval()
layers = model.model.layers
log(f"model loaded: {len(layers)} layers")

HE = tok(" he", add_special_tokens=False).input_ids[0]
SHE = tok(" she", add_special_tokens=False).input_ids[0]
THEY = tok(" they", add_special_tokens=False).input_ids[0]
W_U = model.get_output_embeddings().weight

probes = []
for g in ("male", "female", "neutral"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        probes.append({"group": g, "id": r["id"], "text": r["text"]})

out = {"topk": TOPK, "strengths": STRENGTHS, "layers": {}}

for LAYER in LAYERS:
    ckpt = torch.load(SAE_DIR / f"sae_layer{LAYER}.pt", map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    b_pre, W_enc, b_enc, W_dec = (sd[k].float() for k in ("b_pre", "W_enc", "b_enc", "W_dec"))
    nrm = torch.load(SAE_DIR / f"norm_layer{LAYER}.pt", map_location="cpu", weights_only=False)
    norm_mean, norm_std = nrm["mean"].float(), nrm["std"].float()

    captured = {}

    def capture(module, inp, o):
        captured["r"] = (o[0] if isinstance(o, tuple) else o).detach()

    def resid_at(text):
        h = layers[LAYER].register_forward_hook(capture)
        try:
            with torch.no_grad():
                model(tok(text, return_tensors="pt").input_ids)
        finally:
            h.remove()
        return captured["r"][0]

    def sae_encode(x):
        return torch.relu((((x - norm_mean) / norm_std) - b_pre) @ W_enc + b_enc)

    # ---- rank features by the two-test score, both sides
    gender_dir = (W_U[HE] - W_U[SHE]).float()
    proj = (W_dec * norm_std) @ gender_dir
    acts = {}
    for g in ("male", "female", "neutral"):
        rows = [p for p in probes if p["group"] == g]
        acts[g] = torch.stack([sae_encode(resid_at(p["text"])[-1]) for p in rows])
    mM, mF, mN = acts["male"].mean(0), acts["female"].mean(0), acts["neutral"].mean(0)
    l0 = {g: float((acts[g] > 0).float().sum(-1).mean()) for g in acts}
    log(f"L{LAYER}: L0 {l0}")

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
        log(f"L{LAYER} top {side}-side: " +
            ", ".join(f"f{c['feature']}(s={c['score']:.4f})" for c in cands[side]))

    # ---- steer every candidate
    def steer_hook(module, inp, o):
        d = steer_hook.d * steer_hook.s
        if isinstance(o, tuple):
            return (o[0] + d,) + o[1:]
        return o + d

    def measure(fwd=None):
        rows = []
        h = layers[LAYER].register_forward_hook(steer_hook) if fwd else None
        try:
            for p in probes:
                with torch.no_grad():
                    logits = model(tok(p["text"], return_tensors="pt").input_ids).logits
                pr = torch.softmax(logits[0, -1], dim=-1)
                rows.append({"group": p["group"], "id": p["id"], "p_he": float(pr[HE]),
                             "p_she": float(pr[SHE]), "p_they": float(pr[THEY])})
        finally:
            if h:
                h.remove()
        return rows

    def bias_of(rows, g):
        import math
        sel = [r for r in rows if r["group"] == g]
        return sum(math.log(r["p_he"]) - math.log(r["p_she"]) for r in sel) / len(sel)

    base = measure(None)
    log(f"L{LAYER} baseline: male b={bias_of(base,'male'):+.2f} "
        f"female b={bias_of(base,'female'):+.2f} neutral b={bias_of(base,'neutral'):+.2f}")

    sweeps = []
    for side in ("he", "she"):
        for c in cands[side]:
            steer_hook.d = W_dec[c["feature"]] * norm_std
            per_s = {}
            for s in STRENGTHS:
                steer_hook.s = s
                rows = measure(True)
                per_s[s] = rows
                sweeps += [{**r, "layer": LAYER, "side": side, "feature": c["feature"],
                            "strength": s} for r in rows]
            # the summary a reader needs: how much did the swing move the bias?
            swing_m = bias_of(per_s[max(STRENGTHS)], "male") - bias_of(per_s[min(STRENGTHS)], "male")
            swing_f = bias_of(per_s[max(STRENGTHS)], "female") - bias_of(per_s[min(STRENGTHS)], "female")
            log(f"L{LAYER} {side}-side f{c['feature']}: "
                f"b-swing over [{min(STRENGTHS)},{max(STRENGTHS)}]  male={swing_m:+.2f}  female={swing_f:+.2f}")
            c["swing_male"] = swing_m
            c["swing_female"] = swing_f
    sweeps += [{**r, "layer": LAYER, "side": "none", "feature": -1, "strength": 0.0}
               for r in base]

    out["layers"][str(LAYER)] = {"l0": l0, "candidates": cands, "rows": sweeps}
    json.dump(out, open(OUT / "lfm_search.json", "w"), indent=1)
    log(f"L{LAYER} done, saved")

log("saved lfm_search.json")
