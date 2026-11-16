"""SAE pre-flight metrics and feature characterization.

Reviewers of an SAE paper will ask two things the blog post did not answer:

  1. Is the SAE actually reconstructing? L0 alone does not say. We report L0 *and*
     fraction of variance unexplained (FVU) on held-out wikitext, with the
     reconstructed normalization statistics, so the operating point is on record.
  2. What is f32258, in natural text? A feature named "male-stereotype" on the basis
     of 36 templates could be a "male person" feature, a "blue-collar" feature, or a
     token-level artifact. We report max-activating wikitext contexts, the top
     promoted/suppressed vocabulary tokens, and the nearest decoder neighbours
     (a redundancy indicator).

Outputs results/sae_quality.json
"""
import json
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_post"
FEATURES = {"f32258": 32258, "f27420": 27420, "f9392": 9392, "f26265": 26265}
SAE_DIR = Path.home() / ".aquin/sae/llama-3.2-1b"
OUT = Path("experiment/results")
N_PASSAGES = 100
CTX = 128
SKIP = 5          # drop the BOS position from all SAE statistics -- see paper 3.1

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


ckpt = torch.load(SAE_DIR / "sae_layer8.pt", map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
b_pre, W_enc, b_enc, W_dec, b_dec = (
    sd[k].float() for k in ("b_pre", "W_enc", "b_enc", "W_dec", "b_dec"))
nrm = torch.load(SAE_DIR / "norm_layer8.pt", map_location="cpu", weights_only=False)
norm_mean, norm_std = nrm["mean"].float(), nrm["std"].float()
log(f"SAE loaded: W_enc {tuple(W_enc.shape)}  W_dec {tuple(W_dec.shape)}")

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device="cpu", dtype=torch.float32
)
log("model loaded")


def encode(x):
    """x: [..., d_model] raw residual -> feature activations."""
    xn = (x - norm_mean) / norm_std
    return torch.relu((xn - b_pre) @ W_enc + b_enc)


def decode(z):
    return z @ W_dec + b_dec


from datasets import load_dataset  # noqa: E402

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
texts = []
for row in ds:
    t = row["text"].strip()
    if len(t) > 400:
        texts.append(t[:1200])
    if len(texts) >= N_PASSAGES:
        break
log(f"{len(texts)} wikitext passages")

# ------------------------------------------------ L0 / FVU + max-activating scan
l0s, fvus, n_tok = [], [], 0
tracked = {name: [] for name in FEATURES}          # (act, passage_idx, pos)
act_sums = {name: 0.0 for name in FEATURES}
act_nonzero = {name: 0 for name in FEATURES}
tok_cache = []

for pi, t in enumerate(texts):
    toks = model.to_tokens(t)[:, :CTX]
    tok_cache.append(toks)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    x = cache[HOOK][0][SKIP:]                       # [T, d_model], BOS dropped
    z = encode(x)                                   # [T, n_feat]
    xn = (x - norm_mean) / norm_std
    xhat = decode(z)
    fvu = ((xn - xhat) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum()
    l0s.append(float((z > 0).float().sum(-1).mean()))
    fvus.append(float(fvu))
    n_tok += x.shape[0]
    for name, idx in FEATURES.items():
        col = z[:, idx]
        act_sums[name] += float(col.sum())
        act_nonzero[name] += int((col > 0).sum())
        top = torch.topk(col, k=min(3, col.numel()))
        for a, p in zip(top.values.tolist(), top.indices.tolist()):
            if a > 0:
                tracked[name].append((a, pi, p + SKIP))
    if pi % 20 == 0:
        log(f"passage {pi}: L0={l0s[-1]:.0f} FVU={fvus[-1]:.3f}")

# the same computation WITH the BOS position, to quantify how badly it distorts
bos_l0s, bos_fvus = [], []
for toks in tok_cache[:20]:
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    x = cache[HOOK][0]
    z = encode(x)
    xn = (x - norm_mean) / norm_std
    bos_fvus.append(float(((xn - decode(z)) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum()))
    bos_l0s.append(float((z > 0).float().sum(-1).mean()))
log(f"WITH BOS included: mean L0 = {sum(bos_l0s)/len(bos_l0s):.0f}, "
    f"mean FVU = {sum(bos_fvus)/len(bos_fvus):.1f}  <-- distorted")

mean_l0 = sum(l0s) / len(l0s)
mean_fvu = sum(fvus) / len(fvus)
log(f"wikitext: mean L0 = {mean_l0:.0f} / {W_dec.shape[0]}  ({100*mean_l0/W_dec.shape[0]:.1f}%),"
    f"  mean FVU = {mean_fvu:.3f}  (explains {100*(1-mean_fvu):.1f}% of variance)")

# ------------------------------------------------ max-activating contexts
maxact = {}
for name in FEATURES:
    tracked[name].sort(reverse=True)
    entries = []
    for a, pi, p in tracked[name][:12]:
        toks = tok_cache[pi][0]
        lo = max(0, p - 8)
        entries.append({
            "act": a,
            "token": model.to_string(toks[p: p + 1]),
            "context": model.to_string(toks[lo: p + 1]),
        })
    maxact[name] = entries
    log(f"--- {name}: density {100*act_nonzero[name]/n_tok:.2f}% of wikitext tokens, "
        f"mean act {act_sums[name]/n_tok:.4f}")
    for e in entries[:6]:
        log(f"      {e['act']:6.2f}  ...{e['context']!r}")

# ------------------------------------------------ vocabulary push
HE, SHE = model.to_single_token(" he"), model.to_single_token(" she")
vocab_push = {}
for name, idx in FEATURES.items():
    d = W_dec[idx] * norm_std
    logits = d @ model.W_U                                   # [d_vocab]
    top = torch.topk(logits, 12)
    bot = torch.topk(-logits, 12)
    vocab_push[name] = {
        "promoted": [[model.to_string(torch.tensor([i])), float(logits[i])] for i in top.indices],
        "suppressed": [[model.to_string(torch.tensor([i])), float(logits[i])] for i in bot.indices],
        "he_minus_she": float(logits[HE] - logits[SHE]),
    }
    log(f"--- {name} promotes: " + ", ".join(repr(t) for t, _ in vocab_push[name]['promoted'][:8]))

# ------------------------------------------------ decoder neighbours (redundancy)
Wn = torch.nn.functional.normalize(W_dec, dim=-1)
neighbours = {}
for name, idx in FEATURES.items():
    cos = Wn @ Wn[idx]
    cos[idx] = -1
    top = torch.topk(cos, 10)
    neighbours[name] = [[int(i), float(c)] for i, c in zip(top.indices, top.values)]
    log(f"--- {name} nearest decoder neighbours: " +
        ", ".join(f"f{i}({c:.2f})" for i, c in neighbours[name][:5]))

# ------------------------------------------------ activation on the template sets
tpl_acts = {}
for g in ("male", "female", "neutral"):
    rows = list(map(json.loads, open(f"experiment/probes/tpl_{g}.jsonl")))
    per = {name: [] for name in FEATURES}
    for r in rows:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(r["text"]), names_filter=HOOK)
        z = encode(cache[HOOK][0, -1])
        for name, idx in FEATURES.items():
            per[name].append(float(z[idx]))
    tpl_acts[g] = per
    log(f"--- template acts [{g}]: " +
        "  ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in per.items()))

json.dump({
    "n_tokens": n_tok, "n_features": int(W_dec.shape[0]),
    "mean_l0": mean_l0, "mean_fvu": mean_fvu,
    "mean_l0_with_bos": sum(bos_l0s) / len(bos_l0s),
    "mean_fvu_with_bos": sum(bos_fvus) / len(bos_fvus),
    "per_passage_l0": l0s, "per_passage_fvu": fvus,
    "feature_density_pct": {k: 100 * act_nonzero[k] / n_tok for k in FEATURES},
    "max_activating": maxact, "vocab_push": vocab_push,
    "decoder_neighbours": neighbours, "template_activations": tpl_acts,
}, open(OUT / "sae_quality.json", "w"), indent=1)
log("saved sae_quality.json")
