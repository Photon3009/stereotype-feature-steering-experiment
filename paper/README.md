# NewInML @ NeurIPS 2026 submission

Deadline **29 August 2026**, 11:59pm AoE. Non-archival, double-blind, 2–8 pages
excluding references, NeurIPS 2026 workshop template.

## Files

| path | what it is |
|---|---|
| `main.tex` | the paper. Every number comes from `\input{tables/*}`, generated from the result JSONs — nothing is transcribed by hand |
| `refs.bib` | bibliography |
| `tables/*.tex` | generated table bodies — **do not edit**, regenerate instead |
| `figures/*.pdf` | generated figures — **do not edit**, regenerate instead |
| `REVIEW_RISKS.md` | adversarial review: every vulnerability, its severity, and its current defence. Read this before submitting |

## Building the PDF

```bash
./paper/build.sh          # regenerate tables + figures, compile, print page counts
open paper/main.pdf
```

Needs `tectonic` (`brew install tectonic`, no sudo). The generators skip any table or
figure whose input JSON is missing and write a visible placeholder, so the document
always compiles even mid-experiment.

`paper/neurips_2026.sty` is a **local preview shim**, not the official style file. It
reproduces the NeurIPS page geometry (5.5in x 9in text block, 10pt Times) so the page
count is close to real, but heading sizes and the title block are approximations.
**Delete it and use the workshop's Overleaf template before submitting**; expect the page
count to shift by a few tenths either way.

For statistics only: `venv/bin/python experiment/stats.py`.

## Experiment scripts, in the order they must run

Pre-existing (produced the original results):

1. `compute_norm.py` — rebuild the SAE normalisation statistics
2. `find_features.py` — SAE health check + activation contrasts
3. `pronoun_direction.py` — two-test feature discovery
4. `steer_sweep.py` — main dose–response sweep
5. `robustness_sweep.py` — varied-predicate sweep
6. `capability_check.py` — QA + WikiText loss
7. `lfm_pipeline.py` — the comparison model

Added for the paper:

8. `baseline_compare.py` — matched-norm baselines (difference-in-means, unembedding,
   random), held-out occupations, projection ablation, capability per method.
   Also records P(' they'). **~1 h on CPU**
9. `override_sweep.py` — context override at n=12 per side (was n=4). **~25 min**
10. `specificity.py` — position decomposition, layer sweep, readout diagnostics,
    prompting baseline. **~30 min**
11. `llama_search.py` / `lfm_search.py` — symmetric two-sided top-k feature search in
    both models, needed for the cross-model claim. **~20 min / ~30 min**
12. `sae_quality.py` — L0, FVU, max-activating contexts, vocabulary push, decoder
    neighbours. **~5 min**
13. `stats.py` — paired Wilcoxon, bootstrap CIs, Holm correction. **instant, no model**

> Run these **one at a time**. Three concurrent fp32 copies of Llama-3.2-1B exhausted
> 24 GB and pushed the machine into 30 GB of swap, which made everything an order of
> magnitude slower than serial execution.

## Before you submit — checklist

- [ ] **Compile on the workshop Overleaf template.** Swap the placeholder
      `\usepackage{neurips_2026}` for the official style file. Leave `[final]` off:
      the style file must be in anonymous mode.
- [ ] **Confirm the main body is ≤ 8 pages** excluding references. The current estimate
      is 7.6–7.9 pages, but it comes from a word/row count, not a compile — there is no
      LaTeX toolchain on the machine this was written on. If Overleaf says you are over,
      cut in this order (each is a lift to the appendix, not a content loss):
      `tab:search` → appendix (~0.4 p), `fig:frontier` → appendix (~0.35 p),
      `tab:candidates` → appendix (~0.25 p), §Broader impact to three sentences (~0.15 p).
- [ ] **Anonymous code mirror.** Push the repo to `anonymous.4open.science` and put
      that link in the paper. Do not link the public repo, the personal site, or the
      toolkit.
- [ ] **Grep for de-anonymising strings** before upload:
      `grep -riE "shivam|iamshivamverma|aquin|photon3009|lesswrong" paper/`
- [ ] **Email the organisers** confirming that a publicly posted blog write-up of the
      same work is acceptable for a non-archival submission. It almost certainly is —
      the CFP allows concurrent submission and publication elsewhere — but get it on
      record rather than assume.
- [ ] **Re-read `REVIEW_RISKS.md` §5** and make sure every un-fixable limitation has a
      sentence in `\section{Limitations}`.

## Things that changed from the blog post, and why

These are corrections, not polish. Each was found while preparing the submission.

1. **The "mirror-image mechanism" claim was overturned.** The blog concluded LFM2.5 has
   no steerable he-side feature. That rested on testing rank 1 only; rank 2 (`f17579`)
   swings the bias by 3.16 and rank 5 (`f32504`) by 2.24. The systematic search is in
   `lfm_search.py` / `llama_search.py`, and the paper's cross-model claim is restated
   to match what the symmetric data supports.
2. **The "robustness" set was not held out.** It reuses the same 12 occupations as
   discovery, varying only the predicate. `probes/heldout.jsonl` adds 24 genuinely
   unseen occupations under two unseen template families.
3. **The context-override result had n=4**, where a two-sided Wilcoxon cannot reach
   p < 0.125 at any effect size. Now n=12 per side.
4. **FVU was never reported**, and computing it exposed two bugs: the decoder bias
   (`b_dec`, not `b_pre`) and BOS contamination. Including the BOS position inflates FVU
   by ~370× and makes the maximally activating context of every feature the same token.
   Correct value: median FVU 0.157 at L0 ≈ 3.6k.
5. **The LFM sweep is not monotonic** at the extreme negative end (b = +3.85 at s = −3
   versus +3.17 at s = −6), contrary to the blog's "monotonic both ways".
6. **No baselines existed.** Difference-in-means, unembedding, and random directions at
   matched L2 norm are now the core comparison. Useful geometric fact: the SAE direction
   has cosine similarity 0.16 with difference-in-means, so it is not a rediscovery of it.
7. **`f32258` is better described as a masculine-referent feature** than as a
   "stereotype feature". It promotes ` himself`/` his`/`His`/`他的` and max-activates on
   established-male-referent contexts. The stereotype is what happens when an occupation
   noun activates it.
8. **The output-push test is a filter, not the mechanism.** The direct logit path
   accounts for under a tenth of the measured effect; the rest is mediated downstream.
