"""Reconstruct missing SAE norm stats (per-dim mean/std of layer-8 resid_post)
for llama-3.2-1b and save them where aquin's load_norm() looks for them.
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
OUT = Path.home() / ".aquin/sae/llama-3.2-1b" / f"norm_layer{LAYER}.pt"

try:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > 200:
            texts.append(t[:1000])
        if len(texts) >= 200:
            break
    print(f"corpus: {len(texts)} wikitext passages")
except Exception as e:
    raise SystemExit(f"could not load corpus: {e}")

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)

count = 0
mean_acc = torch.zeros(2048, dtype=torch.float64)
m2_acc = torch.zeros(2048, dtype=torch.float64)
for i, t in enumerate(texts):
    toks = model.to_tokens(t)[:, :128]
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    resid = cache[HOOK][0, 1:].double()  # drop BOS
    for x in resid:
        count += 1
        delta = x - mean_acc
        mean_acc += delta / count
        m2_acc += delta * (x - mean_acc)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(texts)} passages, {count} tokens")

std = (m2_acc / (count - 1)).sqrt()
print(f"tokens={count}  mean-norm={mean_acc.norm():.2f}  std range=[{std.min():.3f}, {std.max():.3f}]")
torch.save({"mean": mean_acc.float(), "std": std.float()}, OUT)
print(f"saved {OUT}")
