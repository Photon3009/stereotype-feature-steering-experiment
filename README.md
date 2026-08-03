# Locating and Steering Stereotype Features in Small LLMs

A weekend-scale mechanistic interpretability experiment: using sparse autoencoders (SAEs),
we locate a single feature in **Llama-3.2-1B-Instruct** (layer-8 residual stream, feature
**f32258**) that causally implements occupation→gender stereotyping — and its mirror image
(**f9619**, a *female*-context feature) in **LFM2.5-230M**.

**Headline results:**

- Subtracting f32258 during the forward pass takes the he:she bias on male-stereotyped
  occupations from **15:1 to 1:1**, at ~2% perplexity cost (factual QA 9/10 vs 10/10).
- Adding it flips female-stereotyped occupations to "he" — and past strength ≈ +4 it
  **overrides an explicit "her" in the prompt**.
- The two models implement the *same* behavior with *opposite* mechanics (male-context
  knob vs female-context knob) — evidence against encoding-universality assumptions in
  debiasing methods.

![Animated steering sweep](experiment/results/fig3_sweep_animation.gif)

**Read the full write-up:** [experiment/lesswrong_post.md](experiment/lesswrong_post.md)
(the LessWrong post, including three failed discovery attempts and the broken-SAE
detective story). The raw chronological lab log is
[experiment/WALKTHROUGH.md](experiment/WALKTHROUGH.md), and the original plan is
[PROPOSAL.md](PROPOSAL.md).

---

## Repo layout

```
PROPOSAL.md                     original experiment proposal
experiment/
  WALKTHROUGH.md                chronological lab log: every command, result, and dead end
  lesswrong_post.md             the write-up (LessWrong post source)
  probes/                       all prompt sets (see below)
  results/                      every number in the post: raw sweep JSONs, trace
                                artifacts, and generated figures
  compute_norm.py               step 1 — rebuild the SAE's normalization statistics
  find_features.py              step 2 — SAE health check (L0) + activation contrasts
  pronoun_direction.py          step 3 — two-test feature discovery (the method that worked)
  steer_sweep.py                step 4 — main causal result: dose-response steering sweep
  robustness_sweep.py           step 5 — same sweep on held-out sentence forms
  capability_check.py           step 6 — QA accuracy + wikitext loss under steering
  lfm_pipeline.py               step 7 — full pipeline on the comparison model (LFM2.5-230M)
  make_figures.py               figures 4–5 (static dose-response curves)
  make_sweep_gif.py             figure 3 (animated steering sweep)
  make_l0_figure.py             figure 2 (L0 diagnostics)
  make_trace_figure.py          figure 1 (single-completion anatomy)
  run_traces.sh                 optional — batch `aquin trace` over probes (CLI artifacts)
  interactive_knob.html         optional — self-contained interactive dose-response explorer
                                (open in a browser; no server needed)
```

### Probes (`experiment/probes/`)

| File | Contents |
|---|---|
| `tpl_{male,female,neutral}.jsonl` | 36 matched templates "The {occupation} said that" — **only the occupation word differs**. The discovery set. |
| `probes.jsonl` | 40 varied-predicate prompts: A = male-stereotyped, B = female-stereotyped, C = neutral, D = context-override ("The mechanic tied **her** hair back before"). Held-out validation + steering measurement. |
| `behavior_probes.jsonl` | "Continue this story…" prompts for behavior-conditioned bucketing (failed attempt 2) |
| `locate_probes.jsonl` | A/B prompts reformatted for `aquin feature locate` (failed attempt 1) |

## Reproducing the experiment

### 0. Setup

Hardware: everything below ran on a MacBook (M4 Pro). Generation uses MPS where the
toolkit chooses to; **all TransformerLens analysis runs on CPU** (TransformerLens warns
MPS may be silently incorrect on torch 2.7.1). Expect a few hours total.

```bash
# Python 3.12 required (torchvision <0.23 has no 3.14 build)
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt

# Llama-3.2-1B-Instruct is gated on Hugging Face — accept the license, then:
./venv/bin/huggingface-cli login

# Download models + SAEs via the Aquin catalog
./venv/bin/aquin load model llama-3.2-1b        # ~2.5 GB
./venv/bin/aquin load sae llama-3.2-1b-l8       # 537 MB
./venv/bin/aquin load model lfm2.5-230m         # comparison model
./venv/bin/aquin load sae lfm2.5-230m-l9
```

> **Note:** as of aquin 3.0.5 the `llama-3.2-1b-l8` download warns
> *"norm invalid in catalog storage"*. That warning is real and it matters — it's why
> step 1 exists. If a later version ships a valid norm file, you can skip step 1
> (but still run step 2 to verify L0 looks healthy before trusting anything).

### 1–7. Run the pipeline (all scripts run from the repo root)

```bash
./venv/bin/python experiment/compute_norm.py        # 1. rebuild norm stats (~30 min, CPU)
./venv/bin/python experiment/find_features.py       # 2. L0 must jump from ~3 to ~6,400
./venv/bin/python experiment/pronoun_direction.py   # 3. discovery → stereotype_candidates.json
./venv/bin/python experiment/steer_sweep.py         # 4. main result → steer_sweep.json (slowest step)
./venv/bin/python experiment/robustness_sweep.py    # 5. held-out validation
./venv/bin/python experiment/capability_check.py    # 6. QA + wikitext loss at strength −6
./venv/bin/python experiment/lfm_pipeline.py 9      # 7. comparison model, layer 9
```

Checkpoints to compare against as you go:

- After step 2: `mean L0 ≈ 6,400` (if you see ~3, the norm file isn't being picked up).
- After step 3: f32258 should top the he-stereotype list; f9392 (the "dietitian" food
  feature) appears in raw contrasts but fails the output-push test.
- After step 4: at strength 0, male-stereotyped occupations ≈ 0.178 / 0.012 (he/she);
  at −6 ≈ 0.053 / 0.050. Exact values vary slightly with your norm statistics.
- Negative control: f27420 should move essentially nothing at any strength.

### 8. Regenerate the figures

```bash
./venv/bin/python experiment/make_figures.py        # fig1, fig2 (static curves)
./venv/bin/python experiment/make_sweep_gif.py      # fig3 (animation)
./venv/bin/python experiment/make_l0_figure.py      # L0 diagnostics
./venv/bin/python experiment/make_trace_figure.py   # completion anatomy (needs trace_B07.json)
```

Optional extras: `./experiment/run_traces.sh` regenerates the per-prompt `aquin trace`
artifacts (`results/trace_*.{json,png}`), and `experiment/interactive_knob.html` is a
self-contained interactive version of the dose-response curves — open it directly in a
browser.

## Results files → post tables

Every table in the write-up maps to a raw JSON in `experiment/results/`:

| File | Backs |
|---|---|
| `steer_sweep.json` | §5 steering tables (f32258 + f27420 control, 7 strengths × 4 groups) |
| `robustness_sweep.json` | §5 "different sentences, same knob" |
| `stereotype_candidates.json` | §4 discovery winners |
| `final_pos_contrast.json` | §3/§4 post-fix activation contrasts |
| `lfm_l9_f9619_sweep.json`, `lfm_l9_results.json` | §6 comparison model |
| `saestats_{male,female,neutral}.json` | §3 saturated-L0 reading (30,688/32,768) |
| `trace_*.json/png` | §1–2 per-prompt traces |

## Known caveats

- The reconstructed norm statistics come from wikitext, not the SAE's original training
  corpus; post-fix L0 (~6,400) is plausibly inflated. The causal results validate
  end-to-end regardless.
- Aquin 3.0.5 bugs we worked around: invalid llama-l8 norm in catalog;
  `feature locate --save` crash (`name 'Path' is not defined`); `feature locate`
  token-mean pooling dilutes final-position signals.
- Two small models, binary pronouns only, next-token metrics — see the post's
  Limitations section before quoting numbers.

## License

MIT — see [LICENSE](LICENSE).
