"""Context-override sweep on an enlarged probe set (n=12 per side, was n=4).

The blog post's most striking claim -- that amplifying the feature makes the model
contradict an explicit pronoun it has already read -- rested on 4 prompts per side.
With n=4 a two-sided Wilcoxon signed-rank test cannot return p < 0.125 no matter how
large the effect, so the claim was untestable rather than merely uncertain.
This rerun uses 12 counter-stereotypical prompts per side, and also records the
crossover strength per prompt (the smallest strength at which the stereotype wins).

Outputs results/override_sweep.json
"""
import json
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FIDX = 32258
STRENGTHS = [-6.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")

t0 = time.time()
ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
n = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
sae_dir = ckpt["state_dict"]["W_dec"][FIDX].float() * n["std"].float()

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32
)
HE, SHE, THEY = (model.to_single_token(t) for t in (" he", " she", " they"))

# same matched-norm difference-in-means baseline as baseline_compare.py
def resid_final(texts):
    outs = []
    for t in texts:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(t), names_filter=HOOK)
        outs.append(cache[HOOK][0, -1].clone())
    return torch.stack(outs)


tpl = {g: [r["text"] for r in map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl"))]
       for g in ("male", "female")}
dim_raw = resid_final(tpl["male"]).mean(0) - resid_final(tpl["female"]).mean(0)
DIRS = {"sae_f32258": sae_dir,
        "diff_in_means": dim_raw / dim_raw.norm() * sae_dir.norm()}

probes = list(map(json.loads, open("experiment/probes/override.jsonl")))
print(f"[{time.time()-t0:.0f}s] {len(probes)} override probes", flush=True)

rows = []
for name, d in DIRS.items():
    for s in STRENGTHS:
        def hook(value, hook, s=s, d=d):
            return value + s * d
        hooks = [] if s == 0 else [(HOOK, hook)]
        for p in probes:
            with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                logits = model(model.to_tokens(p["prompt"]))
            probs = torch.softmax(logits[0, -1], dim=-1)
            rows.append({"method": name, "strength": s, "group": p["group"], "id": p["id"],
                         "p_he": float(probs[HE]), "p_she": float(probs[SHE]),
                         "p_they": float(probs[THEY])})
        sel = [r for r in rows if r["method"] == name and r["strength"] == s]
        msg = []
        for g in ("D_female", "D_male"):
            gg = [r for r in sel if r["group"] == g]
            mhe = sum(r["p_he"] for r in gg) / len(gg)
            mshe = sum(r["p_she"] for r in gg) / len(gg)
            # "stereotype wins" = model prefers the stereotype-consistent pronoun
            # over the one written in the prompt
            wins = sum((r["p_he"] > r["p_she"]) if g == "D_female" else (r["p_she"] > r["p_he"])
                       for r in gg)
            msg.append(f"[{g}] he={mhe:.3f} she={mshe:.3f} stereo_wins={wins}/{len(gg)}")
        print(f"[{time.time()-t0:.0f}s] {name} s={s:+.1f}  " + "  ".join(msg), flush=True)
        json.dump(rows, open(OUT / "override_sweep.json", "w"), indent=1)
print("saved override_sweep.json")
