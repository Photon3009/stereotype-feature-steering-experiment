"""Extend the SAE-free baselines past s = -8, for the occupation-gap frontier.

The submitted frontier scored each direction by the male-stereotyped group's log-odds
alone. That rewards a direction that shifts *every* occupation toward ' she'. The
stereotype itself is the occupation dependence, gap = b(male-stereo) - b(female-stereo).
On that metric the SAE feature reaches a gap of +0.20 at s = -8, while the unembedding
direction was only measured down to s = -8 (gap +2.53). This script measures whether the
baselines close the gap at larger strengths, and at what WikiText cost, so the comparison
is measured rather than extrapolated. Needs no SAE: the dose unit comes from geometry.json.

Reproduces baseline_compare.py exactly: same directions, same WikiText passages, same
128-token truncation. A reproduction check on the unsteered loss runs first.
Output: results/gap_frontier.json.  ~10 min on CPU.
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

HOOK = "blocks.8.hook_resid_post"
OUT = Path("experiment/results")
PROBES = Path("experiment/probes")
STRENGTHS = [-10.0, -12.0, -16.0, -20.0]
torch.set_grad_enabled(False)

SCALE = json.load(open(OUT / "geometry.json"))["sae_dir_l2"]
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                          device="cpu", dtype=torch.float32)
HE, SHE, THEY = (model.to_single_token(t) for t in (" he", " she", " they"))


def unit(v):
    return v / v.norm()


tpl = []
for g in ("male", "female", "neutral"):
    for r in map(json.loads, open(PROBES / f"tpl_{g}.jsonl")):
        tpl.append({"set": "tpl", "group": g, "id": r["id"], "text": r["text"]})
held = [{"set": "heldout", "group": r["group"], "id": r["id"], "text": r["prompt"]}
        for r in map(json.loads, open(PROBES / "heldout.jsonl"))]


def resid_final(texts):
    out = []
    for t in texts:
        _, cache = model.run_with_cache(model.to_tokens(t), names_filter=HOOK)
        out.append(cache[HOOK][0, -1].clone())
    return torch.stack(out)


dim_raw = (resid_final([p["text"] for p in tpl if p["group"] == "male"]).mean(0)
           - resid_final([p["text"] for p in tpl if p["group"] == "female"]).mean(0))
DIRS = {"diff_in_means": unit(dim_raw) * SCALE,
        "unembed_he_she": unit((model.W_U[:, HE] - model.W_U[:, SHE]).float()) * SCALE}


def measure(probes, hooks, tag):
    rows = []
    with model.hooks(fwd_hooks=hooks):
        for i in range(0, len(probes), 32):
            chunk = probes[i:i + 32]
            logits = model([p["text"] for p in chunk], padding_side="left")[:, -1]
            for p, pr in zip(chunk, torch.softmax(logits, -1)):
                rows.append({**tag, "set": p["set"], "group": p["group"], "id": p["id"],
                             "p_he": float(pr[HE]), "p_she": float(pr[SHE]),
                             "p_they": float(pr[THEY])})
    return rows


ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
texts = []
for row in ds:
    t = row["text"].strip()
    if len(t) > 300:
        texts.append(t[:800])
    if len(texts) >= 15:
        break


def wikitext_loss(hooks):
    losses = []
    with model.hooks(fwd_hooks=hooks):
        for t in texts:
            losses.append(float(model(model.to_tokens(t)[:, :128], return_type="loss")))
    return sum(losses) / len(losses)


saved = [c for c in json.load(open(OUT / "capability_baselines.json"))
         if c["method"] == "baseline"][0]["wikitext_loss"]
base_loss = wikitext_loss([])
print(f"repro: unsteered WikiText loss {base_loss:.5f} vs saved {saved:.5f}", flush=True)
if abs(base_loss - saved) > 1e-3:
    raise SystemExit("Reproduction check FAILED: WikiText passages or model differ.")

rows, cap = [], [{"method": "baseline", "strength": 0.0, "wikitext_loss": base_loss}]
for name, d in DIRS.items():
    for s in STRENGTHS:
        h = [(HOOK, lambda v, hook, d=d, s=s: v + s * d)]
        rows += measure(tpl + held, h, {"method": name, "strength": s})
        cap.append({"method": name, "strength": s, "wikitext_loss": wikitext_loss(h)})
        print(f"{name} s={s:+.0f} loss={cap[-1]['wikitext_loss']:.3f}", flush=True)
        json.dump({"rows": rows, "capability": cap}, open(OUT / "gap_frontier.json", "w"),
                  indent=1)
print("DONE", flush=True)
