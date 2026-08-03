"""Capability check: does steering f32258 at -6 hurt the model?
(a) factual QA accuracy (keyword match), (b) mean next-token loss on wikitext.
"""
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FIDX = 32258
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"

ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
n = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
direction = ckpt["state_dict"]["W_dec"][FIDX].float() * n["std"].float()

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32)

QA = [
    ("The capital of France is", "paris"),
    ("The chemical symbol for gold is", "au"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in our solar system is", "jupiter"),
    ("The author of Romeo and Juliet is William", "shakespeare"),
    ("The square root of 64 is", "8"),
    ("The opposite of hot is", "cold"),
    ("The first month of the year is", "january"),
    ("Photosynthesis converts sunlight into", "energy"),
    ("The currency of Japan is the", "yen"),
]

def steer_hooks(s):
    if s == 0:
        return []
    def hook(value, hook, s=s):
        return value + s * direction
    return [(HOOK, hook)]

print("=== factual QA ===")
for s in (0.0, -6.0):
    correct = 0
    for prompt, answer in QA:
        with torch.no_grad(), model.hooks(fwd_hooks=steer_hooks(s)):
            out = model.generate(model.to_tokens(prompt), max_new_tokens=8, temperature=0.0, verbose=False)
        text = model.to_string(out[0, 1:]).lower()
        completion = text[len(prompt):]
        if answer in completion:
            correct += 1
    print(f"s={s:+.0f}: {correct}/{len(QA)} correct", flush=True)

print("=== wikitext next-token loss ===")
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
texts = []
for row in ds:
    t = row["text"].strip()
    if len(t) > 300:
        texts.append(t[:800])
    if len(texts) >= 20:
        break

for s in (0.0, -6.0):
    losses = []
    for t in texts:
        toks = model.to_tokens(t)[:, :128]
        with torch.no_grad(), model.hooks(fwd_hooks=steer_hooks(s)):
            loss = model(toks, return_type="loss")
        losses.append(float(loss))
    print(f"s={s:+.0f}: mean loss = {sum(losses)/len(losses):.4f}", flush=True)
