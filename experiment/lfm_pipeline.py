"""Full stereotype-feature pipeline for lfm2.5-230m (hf_only model, plain
transformers hooks): norm stats -> discovery -> steering sweep.

Usage: python experiment/lfm_pipeline.py [layer]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SAE_DIR = Path.home() / ".aquin/sae/lfm2.5-230m"
SAE_PATH = SAE_DIR / f"sae_layer{LAYER}.pt"
NORM_PATH = SAE_DIR / f"norm_layer{LAYER}.pt"
D_MODEL = 1024

tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-230M", torch_dtype=torch.float32)
model.eval()

layers = model.model.layers
print(f"model loaded: {len(layers)} layers, capturing layer {LAYER} output")

_captured = {}
def capture_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _captured["resid"] = h.detach()

def resid_at(text):
    handle = layers[LAYER].register_forward_hook(capture_hook)
    try:
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return _captured["resid"][0]  # [seq, d_model]

# ---------- 1. norm stats (compute if missing or catalog-invalid) ----------
def valid_norm(p):
    try:
        n = torch.load(p, map_location="cpu", weights_only=False)
        return isinstance(n, dict) and "mean" in n and n["mean"].shape[-1] == D_MODEL
    except Exception:
        return False

if not valid_norm(NORM_PATH):
    print("computing norm stats from wikitext...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > 200:
            texts.append(t[:1000])
        if len(texts) >= 200:
            break
    count = 0
    mean_acc = torch.zeros(D_MODEL, dtype=torch.float64)
    m2_acc = torch.zeros(D_MODEL, dtype=torch.float64)
    for i, t in enumerate(texts):
        resid = resid_at(t)[1:129].double()
        for x in resid:
            count += 1
            delta = x - mean_acc
            mean_acc += delta / count
            m2_acc += delta * (x - mean_acc)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)}", flush=True)
    std = (m2_acc / (count - 1)).sqrt()
    torch.save({"mean": mean_acc.float(), "std": std.float()}, NORM_PATH)
    print(f"saved {NORM_PATH} ({count} tokens)")

n = torch.load(NORM_PATH, map_location="cpu", weights_only=False)
norm_mean, norm_std = n["mean"].float(), n["std"].float()

ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
b_pre, W_enc, b_enc, W_dec = sd["b_pre"].float(), sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float()

def sae_encode(x):
    x = (x - norm_mean) / norm_std
    return torch.relu((x - b_pre) @ W_enc + b_enc)

# ---------- 2. discovery: final-position contrast x pronoun projection ----------
he_ids = tok(" he", add_special_tokens=False).input_ids
she_ids = tok(" she", add_special_tokens=False).input_ids
assert len(he_ids) == 1 and len(she_ids) == 1, f"multi-token pronouns: {he_ids} {she_ids}"
HE, SHE = he_ids[0], she_ids[0]
W_U = model.get_output_embeddings().weight  # [vocab, d_model]
gender_dir = (W_U[HE] - W_U[SHE]).float()
proj = (W_dec * norm_std) @ gender_dir

acts = {}
for g in ("female", "male", "neutral"):
    rows = [json.loads(l) for l in open(f"experiment/probes/tpl_{g}.jsonl")]
    acts[g] = torch.stack([sae_encode(resid_at(r["text"])[-1]) for r in rows])
mF, mM, mN = acts["female"].mean(0), acts["male"].mean(0), acts["neutral"].mean(0)
print(f"L0: F={float((acts['female'] > 0).float().sum(-1).mean()):.0f} N={float((acts['neutral'] > 0).float().sum(-1).mean()):.0f}")

stereo_m = torch.clamp(proj, min=0) * torch.clamp(mM - mN, min=0) * torch.clamp(mM - mF, min=0)
stereo_f = torch.clamp(-proj, min=0) * torch.clamp(mF - mN, min=0) * torch.clamp(mF - mM, min=0)

cands = {}
print("\n=== he-stereotype candidates ===")
for i in torch.argsort(stereo_m, descending=True)[:5]:
    i = int(i)
    print(f"f{i}: score={float(stereo_m[i]):.5f} act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f} proj={float(proj[i]):+.3f}")
cands["he"] = int(torch.argmax(stereo_m))
print("=== she-stereotype candidates ===")
for i in torch.argsort(stereo_f, descending=True)[:5]:
    i = int(i)
    print(f"f{i}: score={float(stereo_f[i]):.5f} act(F/M/N)={float(mF[i]):.2f}/{float(mM[i]):.2f}/{float(mN[i]):.2f} proj={float(proj[i]):+.3f}")
cands["she"] = int(torch.argmax(stereo_f))

# ---------- 3. steering sweep on top he-stereotype candidate ----------
FIDX = cands["he"]
direction = W_dec[FIDX] * norm_std
print(f"\n=== steering sweep on f{FIDX} ===")

def steer_hook(module, inp, out):
    if isinstance(out, tuple):
        return (out[0] + steer_hook.s * direction,) + out[1:]
    return out + steer_hook.s * direction

probes = []
for g in ("female", "male"):
    for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")):
        probes.append({"group": g, "text": r["text"]})

results = []
for s in (-6.0, -3.0, 0.0, 3.0, 6.0):
    steer_hook.s = s
    handle = layers[LAYER].register_forward_hook(steer_hook) if s != 0 else None
    try:
        for p in probes:
            ids = tok(p["text"], return_tensors="pt").input_ids
            with torch.no_grad():
                logits = model(ids).logits
            probs = torch.softmax(logits[0, -1], dim=-1)
            results.append({"feature": FIDX, "strength": s, "group": p["group"],
                            "p_he": float(probs[HE]), "p_she": float(probs[SHE])})
    finally:
        if handle:
            handle.remove()
    for g in ("female", "male"):
        rows = [r for r in results if r["strength"] == s and r["group"] == g]
        mhe = sum(r["p_he"] for r in rows) / len(rows)
        mshe = sum(r["p_she"] for r in rows) / len(rows)
        print(f"s={s:+.0f} [{g}] he={mhe:.3f} she={mshe:.3f}", flush=True)

json.dump({"layer": LAYER, "candidates": cands, "sweep": results},
          open(f"experiment/results/lfm_l{LAYER}_results.json", "w"), indent=1)
print(f"saved experiment/results/lfm_l{LAYER}_results.json")
