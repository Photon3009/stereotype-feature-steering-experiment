# Complete Experiment Walkthrough — Locating and Steering Stereotype Features

Every step, command, dataset, and result, in order. Written to be readable by
someone new to interpretability. This is the raw material for the LessWrong post.

## Concepts in plain words

- **Residual stream** — as a transformer reads text, each token carries a vector
  (2048 numbers in llama-3.2-1b) updated layer by layer: the model's working
  memory for that token.
- **SAE (sparse autoencoder)** — a translator that unpacks that dense vector into
  32,768 slots ("features"), each ideally meaning one thing. Only a few fire at
  a time.
- **Activation / L0** — how strongly a feature fires / how many features are
  non-zero at once. Healthy: hundreds. We saw 3 (broken) and ~6,400 (fixed).
- **Norm** — the SAE was trained on *standardized* inputs: per-dimension
  `(x − mean) / std`, like converting raw scores to "how far from average".
  The norm file stores those 2048 means and stds. Feed the SAE raw
  un-standardized numbers and it goes silent — that's the bug we hit.
- **Steering** — every feature is a direction in the residual stream. Steering
  adds `strength × direction` during the forward pass. If behavior changes
  proportionally with strength (a dose-response curve), the feature is causal.

## Phase 0 — Setup

| Command | Outcome |
|---|---|
| `pip install aquin==3.0.5` (in old venv) | FAILED: Python 3.14 venv; aquin needs torchvision <0.23 which has no 3.14 build |
| `uv venv venv --python 3.12` + `uv pip install aquin==3.0.5` | Works. Old venv kept as `venv-old-py314` |
| `aquin status` | Logged in; GPU = Apple Metal (MPS) |
| `aquin list sae` | llama-3.2-1b: 16 layers of SAEs; lfm2.5-230m: 14; **gpt2-small: none** → comparison model changed to lfm2.5-230m |
| `aquin load model llama-3.2-1b` | ~2.5GB download, ready in 3m35s |
| `aquin load sae llama-3.2-1b-l8` | 537MB. **Warning: "norm invalid in catalog storage"** — the seed of Phase 4 |

## Phase 1 — Probe data (experiment/probes/)

- `probes.jsonl` — 40 prompts: A = 12 male-stereotyped occupations, B = 12
  female-stereotyped, C = 8 neutral ("the person"), D = 8 context-override
  ("The mechanic tied **her** hair back before"). All end right before a pronoun.
- `tpl_{female,male,neutral}.jsonl` — 36 matched templates "The {occupation}
  said that" where ONLY the occupation word differs.
- `locate_probes.jsonl`, `behavior_probes.jsonl` — reformatted for `feature locate`.

## Phase 2 — Baseline: the model is biased

| Command | Result |
|---|---|
| `aquin prompt "The mechanic finished the repair and then"` | Completion invents "Master Technician Alex … He" |
| `aquin trace --prompt "The nurse said that" --layer 8 --check` | Logit lens: `' she'` = 31% at layer 14, `' he'` absent from top-5. (Also: instruct model chat-templated our prompt and replied conversationally) |
| behavior bucketing (Phase 3, attempt 2) | 11 completions used "he", 6 "she", on a balanced prompt set |

## Phase 3 — Three discovery attempts that failed informatively

1. `aquin feature locate --prompts locate_probes.jsonl --layer 8 --conditioning prompt`
   (A vs B as the two buckets) → top feature f3432; `aquin feature logit
   --feature 3432` showed random junk tokens. **Lesson: contrasting prompts that
   differ in topic finds topic features, not gender.** Also `--save` crashed
   (`name 'Path' is not defined`) — aquin bug.
2. Same with `--conditioning behavior` on `behavior_probes.jsonl` ("Continue this
   story…" + she/he reference words). Buckets by which pronoun the model actually
   generated — smarter design, but deltas ~0.5%. **Lesson: token-mean averaging
   dilutes a one-position signal.**
3. `./experiment/run_traces.sh` — `aquin trace` on 8 probes, collect features
   peaking at pronoun tokens (results/trace_*.json). Found f26265 firing on both
   "his" and "her" = grammar feature, not stereotype. Two male-only features:
   f2546, **f32258**. **Lesson: trace's top-20 cutoff is too thin alone, but its
   candidates seeded the cross-check in Phase 5.**

## Phase 4 — The broken norm and the fix

- `experiment/find_features.py` (first run): SAE encoding of our prompts fired
  only **L0 = 3** of 32,768 features → SAE effectively dead.
- Read aquin source (`compute/feature_analysis.py`): it standardizes inputs with
  a norm file at `~/.aquin/sae/llama-3.2-1b/norm_layer8.pt` — the exact file the
  download warning said was invalid.
- `experiment/compute_norm.py`: 200 wikitext passages (22,503 tokens) through the
  model on CPU; computed per-dimension mean/std of layer-8 residuals; saved in
  aquin's expected format/location.
- Re-run: **L0 = 3 → ~6,400**. (The lfm SAEs later downloaded WITH valid norms,
  so this is specific to the llama-l8 catalog entry.)

## Phase 5 — Discovery that worked

`experiment/pronoun_direction.py` on the 36 matched templates. Two tests
intersected per feature:
1. **Firing selectivity** at the final token: female vs male vs neutral occupations.
2. **Output push**: decoder direction projected onto the vocabulary — does it
   raise ' he' or ' she'? (Kills topic impostors: candidate f9392 turned out to
   be a *food* feature riding on "dietitian".)

Winners (results/stereotype_candidates.json):
- **f32258** — male-stereotype: act 0.38 male / 0.03 female / 0.16 neutral,
  pushes ' he'; independently flagged by Phase-3 traces. Two methods agree.
- f27420 — female-stereotype candidate (single-method; becomes the negative control).

## Phase 6 — Causal proof on llama-3.2-1b

`experiment/steer_sweep.py` — inject direction at layer 8, strengths −6…+6,
measure mean P(' he') / P(' she') (results/steer_sweep.json, fig1):

| f32258 strength | male-occ he:she | female-occ | context-override D |
|---|---|---|---|
| 0 | 0.178:0.012 (15:1) | she-favored 0.099:0.039 | she 13:1 (respects "her") |
| −6 | **0.053:0.050 (1:1) — gap gone** | — | — |
| +6 | 0.225:0.004 | **flips male: 0.163:0.014** | **flips male: he 0.033 vs she 0.013** |

- Robustness (`experiment/robustness_sweep.py`, varied-predicate probes):
  27:1 → ~1:1 at −6; female probes flip at +6. Same curve, different sentences.
- Coherence at −6: completions grammatical; mechanic story switched to "inform
  **them**" (spontaneously neutral).
- Capability (`experiment/capability_check.py`): factual QA 9/10 vs 10/10;
  wikitext loss 3.284 → 3.359 (+2.3%). Debiasing ≈ free.
- Negative control: f27420 flat everywhere → verification (two converging
  methods) is what separated a real knob from a dud.

## Phase 7 — Comparison model lfm2.5-230m

- `aquin load model lfm2.5-230m`; `aquin load sae lfm2.5-230m-l9` (+l5). Norm
  files valid this time.
- `experiment/lfm_pipeline.py 9` (plain HF hooks — architecture not
  TransformerLens-compatible): baseline bias smaller (male-occ 3:1 vs llama's
  15:1). He-side candidates weak (steering f12288 ≈ flat). She-side standout:
  **f9619** (0.32 female / 0.01 male / 0.04 neutral — 32× selective).
- f9619 sweep (results/lfm_l9_f9619_sweep.json, fig2): at +6 even *male*
  occupations go she-favored 26:1 (0.234:0.009); at −6 female occupations flip
  to he-favored. Monotonic both ways.

**Cross-model punchline:** both models learned the same stereotype, but one
implements it as a steerable *male-context* feature and the other as a
*female-context* feature — same behavior, mirror-image mechanics. The smaller
base-ish model is less biased at baseline than the larger instruct model.

## Bugs/feedback for the Aquin team

1. `llama-3.2-1b-l8` norm stats invalid in catalog (root cause of Phase 3–4 failures).
2. `feature locate --save` crashes: `name 'Path' is not defined`.
3. `feature locate` token-mean conditioning dilutes position-specific signals —
   a `--position last` option would help.
4. TransformerLens warns MPS may be silently incorrect on torch 2.7.1 — analyses
   ran on CPU.

## File inventory

- Scripts: `compute_norm.py`, `find_features.py`, `pronoun_direction.py`,
  `steer_sweep.py`, `robustness_sweep.py`, `capability_check.py`,
  `lfm_pipeline.py`, `make_figures.py`, `run_traces.sh`
- Probes: `probes/*.jsonl`
- Raw results: `results/*.json`, trace artifacts `results/trace_*.{json,png}`
- Figures: `results/fig1_llama_dose_response.png`, `results/fig2_lfm_dose_response.png`
