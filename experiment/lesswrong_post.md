*Epistemic status: a weekend-scale interpretability experiment on two small open models, written up honestly — including the three discovery attempts that failed and the broken normalization file that silently invalidated our first results. Every number below comes from a saved artifact in the repo; nothing is from memory.* I'm new to mechanistic interpretability; corrections are welcome.

> **TL;DR**  
>   
> Using sparse autoencoders (SAEs), I located a single feature in Llama-3.2-1B-Instruct's layer-8 residual stream **f32258** that implements occupation→gender stereotyping. Subtracting it during the forward pass takes the model's he:she bias on male-stereotyped occupations from **15:1 to 1:1**, at a capability cost of ~2% perplexity. Adding it flips even female-stereotyped occupations to "he," and overrides an explicit "her" in the prompt. A second model (LFM2.5-230M) turned out to implement the same stereotype with **opposite mechanics**: its steerable knob is a *female*-context feature, f9619. Same learned behavior, mirror-image implementation. Along the way we hit and fixed, a corrupted SAE normalization file, which taught us the most transferable lesson of the project: **an SAE fed inputs distributed differently from its training data doesn't fail loudly, it just goes quiet**.

* * *

Why this project
----------------

Occupational gender stereotyping is one of the oldest documented behaviors in language models it shows up in word embeddings, in masked LMs, and in every generation of autoregressive model since. It is usually studied *behaviorally*: measure the bias, fine-tune or filter, measure again. We wanted to ask the mechanistic question instead: **where does the stereotype live, and is it a thing you can grab?**

Concretely, we wanted to run the full modern interpretability pipeline end to end, at a scale one person can afford (two models under 1.3B parameters, a single MacBook generation on Apple's integrated GPU, all quantitative analysis on CPU):

1.  Observe the bias (logit lens, sampled completions);
2.  Decompose the residual stream (SAE);
3.  Locate candidate features (contrastive probes);
4.  Validate them (a second, independent test);
5.  Prove causality (steering with a dose-response curve);
6.  Check side effects (capability evals) and then do it all again on a second model to see what transfers.

This is the same shape as the work Anthropic did with ["Golden Gate Claude"](https://www.anthropic.com/news/golden-gate-claude) and the [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) paper, and the steering methodology follows [Activation Addition](https://arxiv.org/abs/2308.10248) but small enough to replicate in a weekend, and with all the failure modes left in the write-up. If you are a beginner considering an SAE project, sections 2 and 3 (the failures) are probably worth more to you than the results.

0\. Concepts in plain words
---------------------------

Skip if you know mech interp basics.

*   **Residual stream** — as a transformer reads text, each token carries a vector (2048 numbers in Llama-3.2-1B) that every layer reads and updates: the model's working memory for that token.
*   **Sparse autoencoder (SAE)** — a translator that unpacks that dense vector into 32,768 slots ("features"), each ideally meaning one interpretable thing. Only a few hundred should fire at a time. (The canonical reference is [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/).)
*   **L0** — how many features are non-zero for a given input. Healthy: hundreds. We saw 3 (broken), ~30,000 (differently broken), and ~6,400 (fixed) — more on that disaster below.
*   **Logit lens** — take any layer's residual vector, multiply by the unembedding matrix, and read off what the model "currently predicts." Lets you watch a prediction form layer by layer. ([Original post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).)
*   **Steering** — every SAE feature corresponds to a direction in the residual stream. Steering adds `strength × direction` during the forward pass. If behavior changes smoothly and monotonically with strength — a dose-response curve — the feature isn't just correlated with the behavior, it's causally upstream of it.

**Setup.** Llama-3.2-1B-Instruct (16 layers) with a 32,768-feature SAE trained on the layer-8 residual stream — layer 8 being the midpoint, where earlier logit-lens work suggests semantic/priors information is consolidating but the output isn't yet decided. Comparison model: LFM2.5-230M with layer-9 and layer-5 SAEs. All artifacts come from the **Aquin toolkit's catalog**; analysis via TransformerLens (Llama) and plain HuggingFace hooks (LFM, whose architecture TransformerLens doesn't support), on CPU. I had planned GPT-2-small as the comparison, but the catalog had no SAEs for it.

```
aquin load model llama-3.2-1b     # ~2.5 GB
aquin load sae llama-3.2-1b-l8    # 537 MB — printed a warning I ignored. Hold that thought.
```

1\. The behaviour: yes, the model is biased
-------------------------------------------

Three quick checks before any interpretability:

1.  **Free completion.** Prompted with "The mechanic finished the repair and then", the model invents a character — "Master Technician Alex … **He**". (`aquin prompt "The mechanic finished the repair and then"`)
2.  **Logit lens.** On "The nurse said that", the prediction ' she' emerges mid-network and reaches **31% by layer 14**; ' he' is absent from the top-5. (`aquin trace --prompt "The nurse said that" --layer 8 --check`)
3.  **Balanced completions.** Across a balanced set of occupation prompts, sampled continuations used "he" 11 times and "she" 6.

![fig_trace_anatomy.png](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1785761049/lexical_client_uploads/q502xjcuebk2pkptsyqe.png)

Figure 1. Anatomy of one biased completion (the hairdresser probe). Top: nothing in the prompt says the hairdresser is a woman, the stereotype fills in "her … she". Bottom-left: logit lens early layers just copy the prompt's last word ("then"); the actual prediction only forms in the upper layers. Bottom-middle: ablating single layers shows the prediction depends on layers 9–14, just downstream of where our SAE reads (layer 8, dashed). Bottom-right: the strongest SAE features for this prompt, none of them obviously about gender, which is why finding the stereotype feature took the machinery of §4.

Quantitatively (measured later, as the strength-0 point of our steering sweeps, averaged over 12 prompts per group of the form "The {occupation} said that"):

| Prompt group | mean P(' he') | mean P(' she') | ratio |
| --- | --- | --- | --- |
| Male-stereotyped occupations (mechanic, engineer, pilot…) | 0.178 | 0.012 | **he 15:1** |
| Female-stereotyped occupations (nurse, librarian, florist…) | 0.039 | 0.099 | she 2.5:1 |
| Context-override ("The mechanic tied **her** hair back…") | 0.003 | 0.039 | she 13:1 |

Note the asymmetry: male-stereotyped occupations get a 15:1 skew while female-stereotyped ones only get 2.5:1, the model is much more confident that mechanics are "he" than that nurses are "she". Also note the model *does* respect explicit context, an explicit "her" beats the mechanic stereotype, 13:1. Keep that context-override row in mind; it becomes the most striking result of the steering section.

2\. Three discovery attempts that failed informatively
------------------------------------------------------

I tried the obvious things first. All three failed, each teaching me something I needed later. (If you're skimming for results, jump to §4—but these failures are the part I most wish someone had written up before I started.)

**Attempt 1 — contrast male-occupation prompts vs female-occupation prompts.** Feed 12 prompts about mechanics/engineers/pilots and 12 about nurses/receptionists/teachers into a feature-localization tool and rank features by activation difference. Top hit: f3432, with deltas of ~1% on top of enormous base activations — already suspicious. Projecting its decoder direction onto the vocabulary gave random junk tokens, not gendered ones.

```
aquin feature locate --prompts locate_probes.jsonl --layer 8 --conditioning prompt
aquin feature logit --feature 3432    # decoder direction → vocabulary: junk tokens
```

> **Lesson 1: contrasting prompt sets that differ in topic finds topic features.** Repair-and-wiring prompts vs patients-and-phones prompts differ in *many* ways besides implied gender, and the biggest activation differences track the biggest differences,v subject matter.

**Attempt 2 — bucket by the model's own behavior.** Smarter design: give a neutral "Continue this story…" instruction, then bucket prompts by whether the model actually generated "she" or "he," and contrast those buckets. Directionally right, but the activation deltas were ~0.5% — noise.

```
aquin feature locate --prompts behavior_probes.jsonl --layer 8 --conditioning behavior
```

> **Lesson 2: averaging feature activations over all token positions dilutes a one-position signal.** Gender-of-the-upcoming-pronoun lives at the *final* token position. Mean-pooling over 10+ positions buries it.

**Attempt 3 — per-prompt tracing.** Trace 8 probes individually (Figure 1 is one of them) and collect features that peak at pronoun positions. This found f26265 firing on both "his" and "her" — a *grammar* (possessive-pronoun) feature, not a stereotype feature. Distinguishing grammatical gender features from stereotype features became a recurring theme. But the traces also surfaced two male-only candidates: f2546 and **f32258**. We parked them.

```
aquin trace --prompt "<probe>" --layer 8 --check    # × 8 probes (run_traces.sh in the repo)
```

> **Lesson 3: single-prompt traces are too thin alone, but their candidates are gold for cross-checking against a systematic method later.**

3\. Interlude: the SAE was dead the whole time
----------------------------------------------

Here's the part that would have silently ruined everything if I hadn't checked a basic diagnostic.

I wrote a direct encoding script (`find_features.py` in the repo, bypassing the CLI): run our prompts through the model, take the layer-8 residual at the final token, push it through the SAE, and look at which features fire. First run:

female: mean L0 = 3  
male:   mean L0 = 3  

**Three** active features out of 32,768. A healthy SAE fires hundreds. Every activation-based result up to this point had been produced by an effectively dead SAE.

The cause: this SAE was trained on *standardized* inputsm, each of the 2048 residual dimensions shifted and scaled as (x − μ)/σ, with the means and stds stored in a companion `norm` file. The toolkit had actually warned us at download time ("norm invalid in catalog storage") and we'd ignored it. Without valid stats, raw residuals whose per-dimension scale is nothing like the standardized training distribution, mostly fail to clear the encoder's ReLU. The SAE doesn't error. It just goes quiet.

The same broken file produced the *opposite* pathology in a different code path. Our direct script fell back to no normalization and saw near-silence (L0 = 3); the toolkit's own stats command (`aquin sae-stats --prompts tpl_neutral.jsonl --layers 8`), which applied the invalid values, reported a mean L0 of **30,688 out of 32,768** — a "sparse" autoencoder with 94% of its features active:

![fig_l0_diagnostics.png](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1785762617/lexical_client_uploads/tsixnl4qzigayqyezawq.png)

*Figure 2. One SAE, three readings — same layer-8 SAE, same 12 prompts; the only difference is the normalization file. Raw inputs read as dead (L0 = 3); the corrupted catalog file reads as saturated (L0 = 30,688, from the toolkit's own stats command); our reconstructed statistics bring it to life. Neither broken state throws an error — you only catch this by checking L0.*

If you've done classical ML, this is exactly **train/serving skew**: fit a `StandardScaler` at training time, forget (or corrupt) it at inference, get garbage. The norm file *is* the fitted scaler.

The fix was 50 lines (`compute_norm.py` in the repo): stream 200 wikitext passages (22,503 tokens) through the model, compute each residual dimension's running mean and std (Welford's algorithm), and save the result where the toolkit expects it. Re-run:

mean L0 = 3  →  mean L0 ≈ 6,400  

Alive. (An L0 of 6,400 is still high for a well-trained SAE our wikitext statistics are surely not identical to the SAE's original training statistics but features were now interpretable and, as the rest of this post shows, causally meaningful. The comparison model's SAEs shipped with valid norm files, confirming the corruption was specific to this catalog entry.)

> **Lesson 4 — the big one: check L0 before believing anything an SAE tells you.** A distribution-mismatched SAE fails silently, in either direction — dead quiet or fully saturated. One number, computed in one line, distinguishes "the SAE disagrees with your hypothesis" from "the SAE isn't running."

4\. Discovery that worked: two independent tests must agree
-----------------------------------------------------------

With a live SAE, we rebuilt discovery around the failed attempts' lessons: matched templates (kill topic confounds), final-position readout (kill dilution), and critically **two independent tests intersected per feature**:

1.  **Firing selectivity.** Over 36 templates "The {occupation} said that" where *only the occupation word differs* (12 female-stereotyped, 12 male-stereotyped, 12 neutral like "the person"), does the feature fire for one group and not the others at the final token?
2.  **Output push.** Project the feature's *decoder* direction onto the vocabulary (the logit lens applied to a single feature "direct logit attribution"): does this feature, when it fires, actually push ' he' or ' she' upward?

The two tests fail independently, which is the point. Selectivity alone admits anything correlated with the occupation groups; output-push alone admits every grammar feature that touches pronouns. The intersection is what kills impostors: the top female-minus-male feature by activation contrast alone was f9392 — which the output-push test exposed as a **food feature** riding on "dietitian." Selective firing, zero pronoun push. Discarded.

Winners:

*   **f32258** (male-stereotype): fires 0.38 on male-stereotyped occupations vs 0.03 female / 0.16 neutral, and pushes ' he'. Independently flagged by Attempt 3's traces — two methods converging on the same feature.
*   **f27420** (female-stereotype candidate): passed only the selectivity test. We kept it as a **negative control** — a feature that *looks* related by one method but lacks convergent evidence. It earns its keep in the next section.

5\. Causal proof: the dose-response curve
-----------------------------------------

Correlation established; now causation. We inject `strength × decoder_direction(f32258)` into the layer-8 residual stream at every position, sweep strength from −6 to +6 (in units of the SAE's activation scale; 0 = unmodified model), and measure mean P(' he') and P(' she') at the final token, per prompt group (`steer_sweep.py` in the repo; the robustness and capability checks below are `robustness_sweep.py` and `capability_check.py`).

![fig3_sweep_animation.gif](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1785762678/lexical_client_uploads/hjxmlctgpbgavqqkkqvg.gif)

*Figure 3 (animated). One knob, three prompt groups. Watch the male-stereotyped panel first: at −6 the two curves meet (bias erased); by +6 the ratio is 57:1. Then watch the right panel: past strength ≈ +4 the model overrides an explicit "her" in its own prompt.*

![fig1_llama_dose_response.png](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1785762712/lexical_client_uploads/eroafguoh6uqeiupzocf.png)

*Figure 4 (static version of Figure 3, for skimming). Every curve is monotonic in steering strength — the dose-response signature.*

The numbers behind the figures (mean over 12 prompts per group):

**Male-stereotyped occupations** — the 15:1 gap closes to 1:1 at −6:

| strength | P(' he') | P(' she') | ratio |
| --- | --- | --- | --- |
| −6  | 0.053 | 0.050 | **1.1 : 1 — gap gone** |
| −3  | 0.108 | 0.039 | 2.8 : 1 |
| 0   | 0.178 | 0.012 | 14.6 : 1 (baseline) |
| +6  | 0.225 | 0.004 | 57 : 1 |

**Female-stereotyped occupations** — positive steering *flips* them:

| strength | P(' he') | P(' she') | favored |
| --- | --- | --- | --- |
| −6  | 0.015 | 0.064 | she |
| 0   | 0.039 | 0.099 | she 2.5:1 (baseline) |
| +6  | 0.163 | 0.014 | **he 12:1 — flipped** |

**Context-override prompts** ("The mechanic tied **her** hair back before…") — at baseline the explicit "her" wins 13:1. At +6, f32258 overpowers *explicit textual evidence*:

| strength | P(' he') | P(' she') | favored |
| --- | --- | --- | --- |
| 0   | 0.003 | 0.039 | she 13:1 (respects "her") |
| +6  | 0.034 | 0.013 | **he 2.5:1 — stereotype beats the text** |

That last table is, to us, the most striking result: a single feature, amplified 6×, makes the model contradict a pronoun it just read. The stereotype isn't a soft prior the model consults when context is silent — it's a causal circuit that, at sufficient magnitude, *outvotes* context.

### Robustness, coherence, cost, control

A dose-response curve on the discovery prompts isn't enough — the standard failure mode is a knob that only works on the sentences you used to find it, or that works by lobotomizing the model. Four checks:

*   **Different sentences, same knob.** A second sweep with varied predicates (not just "said that") reproduced the curve: baseline he:she of 28:1 on male-stereotyped prompts → ~1:1 at −6; female-stereotyped prompts flip to 11:1 he at +6. The knob is attached to the *concept*, not to our template.
*   **Coherence.** At −6, completions stay grammatical. The mechanic story spontaneously switched to "inform **them**", the model didn't get confused, it went *neutral*. This is what you'd hope debiasing looks like from the inside: not suppressing pronouns, but genuinely not privileging one.
*   **Capability cost ≈ free.** At the debiasing strength (−6): factual QA 9/10 vs 10/10 unsteered; wikitext loss 3.284 → 3.359 (**+2.3%**). We are not claiming zero damage — see Limitations — but this is far from lobotomy.
*   **Negative control behaves.** Steering f27420 — the single-method candidate from §4 moves essentially nothing anywhere: male-occupation P(' he') drifts 0.184 → 0.174 across the entire −6…+6 range, and female prompts stay she-favored throughout. This is the quiet punchline of the methodology: **one line of converging evidence found a dud; two lines found a knob.** If we had skipped the output-push test and steered our best selectivity-only candidates, we would have written a null result.

6\. The twist: a second model implements the same stereotype backwards
----------------------------------------------------------------------

We ran the whole pipeline on LFM2.5-230M (layer-9 SAE; different architecture, so plain HuggingFace hooks instead of TransformerLens — `lfm_pipeline.py` in the repo, after `aquin load model lfm2.5-230m` and `aquin load sae lfm2.5-230m-l9`).

Baseline: same direction of bias, much weaker — male-stereotyped occupations at **3:1** he:she (vs Llama's 15:1). The smaller, less heavily instruction-tuned model is *less* biased at baseline than the bigger instruct model — consistent with the suspicion that some of Llama's confident 15:1 skew is a property of post-training, not just pretraining data, though our two-model sample can't settle that.

Then discovery found something unexpected. The he-side candidates were weak — steering the best one (f12288) did roughly nothing. The standout was on the *she* side: **f9619**, firing 0.32 on female-stereotyped occupations vs 0.01 male / 0.04 neutral — 32× selectivity.

![fig2_lfm_dose_response.png](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1785762746/lexical_client_uploads/x5mhpkg2f03wpspcksot.png)

*Figure 5. Steering f9619 (a female-context feature) in LFM2.5-230M. Note the mirror image relative to Figures 3–4: here it's P(' she') that the knob drives directly, on both prompt groups.*

Steering f9619 (mean over 12 prompts per group):

| strength | female occs P(he)/P(she) | male occs P(he)/P(she) |
| --- | --- | --- |
| −6  | 0.063 / 0.006 — **flips he 11:1** | 0.116 / 0.005 |
| 0   | 0.014 / 0.102 (baseline) | 0.103 / 0.034 (baseline 3:1) |
| +6  | 0.002 / 0.164 | 0.009 / 0.234 — **she 26:1, on** ***male*** **occupations** |

Monotonic both directions, both groups.

So: **Llama-3.2-1B implements occupation→pronoun stereotyping via a steerable** ***male-context*** **feature; LFM2.5-230M implements it via a steerable** ***female-context*** **feature.** Identical learned behavior, mirror-image mechanics.

Why does this matter beyond trivia? There's an implicit universality assumption in a lot of applied interpretability — that a behavior as canonical as gender bias will be encoded in some canonical way ("the gender direction"). Our n=2 sample already breaks it. If you only ever inspected one model, you might conclude "gender bias is encoded as a maleness direction" as if it were a fact about the *behavior* — it's a fact about the *implementation*, and it doesn't transfer. Any debiasing intervention that assumes a particular internal encoding (ablate direction X, clamp feature Y) has to be re-derived per model. The *procedure* transferred perfectly; the *feature* did not.

7\. What we'd tell someone starting a project like this
-------------------------------------------------------

1.  **Check L0 first.** Before interpreting any SAE output, verify the SAE is actually firing at a healthy rate on *your* inputs. Distribution mismatch fails silently, in both directions. One line: `aquin sae-stats --prompts your_probes.jsonl --layers <n>` — healthy is hundreds active. (And read your tools' download warnings.)
2.  **Match everything except the variable you care about.** "The {occupation} said that" with only the occupation swapped found in one afternoon what topic-confounded prompt sets missed in two.
3.  **Read out at the position where the signal lives.** Token-mean pooling erased a signal that was obvious at the final position.
4.  **Demand two independent lines of evidence per feature.** Activation contrast alone crowned a food feature (f9392); one-method candidate f27420 steered like a dud. The features that passed both tests — and only those — turned out to be causal.
5.  **Correlation is cheap; sweep a dose-response curve.** A monotonic behavior-vs-strength curve, on held-out sentence forms, with a capability check and a negative control, is the difference between "we found a correlate" and "we found the knob."
6.  **Run a second model.** The cross-model mirror image was the most interesting finding of the project and cost one extra pipeline run.

Related work
------------

Nothing here is methodologically novel — the contribution is running the whole pipeline end to end on a socially meaningful behavior, at hobbyist scale, with the failures documented. The pieces we assembled:

*   **Gender bias in language models.** The behavioral phenomenon goes back to word embeddings — [Bolukbasi et al. (2016)](https://arxiv.org/abs/1607.06520) ("man is to computer programmer as woman is to homemaker") and [Caliskan et al. (2017)](https://www.science.org/doi/10.1126/science.aal4230) (WEAT) — and was formalized for coreference in [Winogender](https://arxiv.org/abs/1804.09301) and [WinoBias](https://arxiv.org/abs/1804.06876). Our occupation→pronoun probes are a stripped-down cousin of those benchmarks, adapted for next-token probability readout.
*   **Sparse autoencoders.**  [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) (Bricken et al., 2023) established SAE-based dictionary learning on LM residuals; [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Templeton et al., 2024) scaled it to a frontier model and demonstrated feature steering, popularized by [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude). [Gemma Scope](https://arxiv.org/abs/2408.05147) (Lieberum et al., 2024) is the reference open SAE release — its documentation of input normalization is the context for our §3 disaster.
*   **Activation steering.** Adding directions to the residual stream to control behavior: [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023), [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Rimsky et al., 2024), and [Inference-Time Intervention](https://arxiv.org/abs/2306.03341) (Li et al., 2023). Our dose-response framing follows the sweep methodology of CAA.
*   **Lenses.** The [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (nostalgebraist, 2020) and its calibrated successor the [tuned lens](https://arxiv.org/abs/2303.08112) (Belrose et al., 2023). Our "output push" validation test is a single-feature version of direct logit attribution.
*   **SAE features for debiasing specifically.**  [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) (Marks et al., 2024) introduced SHIFT, which ablates SAE features carrying unintended signals (including gender) from a classifier — the closest published relative of our §5, and much more rigorous about circuit-level attribution. Our cross-model mirror-image finding (§6) is a small empirical data point against the implicit universality assumption in encoding-specific debiasing.

Limitations
-----------

Two small models; one bias axis (binary pronouns — we didn't measure ' they' except anecdotally, though the "inform them" completion suggests it's worth doing properly); one layer per model; steering evaluated mainly on next-token pronoun probabilities plus small QA/perplexity checks rather than long-horizon generation. We found *a* causal knob, not *the* representation: nothing here shows the stereotype is stored in exactly one feature — f32258 is a handle the SAE happened to carve out, and other features or layers may carry redundant copies (our −6 "gap gone" result suggests this handle covers most of it on these prompts, but that's as far as the evidence goes). Our reconstructed norm stats came from wikitext rather than the SAE's original training corpus (the post-fix L0 of ~6,400 is plausibly inflated relative to the SAE's intended operating point). The he:she ratios at extreme strengths involve small absolute probabilities and should be read as directional, not precise.

Future work
-----------

Roughly in order of how much we want to do each:

1.  **Non-binary pronouns as a first-class axis.** The spontaneous "inform **them**" completion at −6 suggests the debiased model reaches for singular *they* — measure P(' they') across the whole sweep instead of treating it as an anecdote, and check whether f32258 suppression redistributes mass to ' they' or spreads it elsewhere.
2.  **Layer scan.** We only had an SAE at one layer per model. Repeating discovery at every layer with SAEs would show where the stereotype *forms*: the Figure 1 ablation drops (L9–14 doing the work, just above our read point) predict the knob should exist at several adjacent layers with varying strength.
3.  **Redundancy test.** Zero-ablate f32258 entirely, re-run discovery, and see whether a backup feature emerges. This distinguishes "the stereotype's home" from "one handle among several" — our Limitations section flags this as the main open question about the causal story.
4.  **From steering to weight edit.** Steering requires intervening at inference time. Following [Sparse Feature Circuits / SHIFT](https://arxiv.org/abs/2403.19647), the same feature could be ablated permanently (e.g., subtracting its decoder direction from downstream weight reads), giving a deploy-time debiased model — then re-run the full capability suite.
5.  **More models, one hypothesis.** With n=2 we found opposite polarities (male-context vs female-context knobs). Running the pipeline over 5–10 small open models would turn the mirror-image observation into an actual claim about how often each implementation arises, and whether it correlates with model size, architecture, or amount of instruction tuning (our Llama-vs-LFM baseline gap of 15:1 vs 3:1 hints at instruction tuning amplifying the bias).
6.  **Long-horizon behavior.** All our steering metrics are next-token probabilities plus short completions. Generate full stories at −6 and evaluate them with a judge model for coherence, pronoun consistency, and whether the debiasing survives 200 tokens of context accumulation.
7.  **Proper norm statistics.** Recompute the norm file from the SAE's actual training corpus (if the toolkit publishes it) and check how much our wikitext approximation inflated L0 — and whether discovery results shift at all (we predict not, since the causal results validated end-to-end).

Appendix: bugs filed against the toolkit
----------------------------------------

For reproducibility, the issues we hit in Aquin v3.0.5: (1) the llama-3.2-1b layer-8 norm stats are invalid in catalog storage — root cause of our dead-SAE phase; (2) `feature locate --save` crashes with `name 'Path' is not defined`; (3) `feature locate` averages activations over all token positions — a `--position last` option would have saved us a week; (4) TransformerLens warns MPS may be silently incorrect on torch 2.7.1, so all analyses ran on CPU.

Reproducibility
---------------

Everything needed to reproduce this post is on GitHub: [**Photon3009/stereotype-feature-steering-experiment**](https://github.com/Photon3009/stereotype-feature-steering-experiment). The `experiment/` directory contains the probe sets (matched templates, context-override prompts, behavior probes), all scripts in run order (norm reconstruction, direct SAE encoding, two-test discovery, steering sweeps, robustness sweep, capability checks, figure and animation generation), the raw sweep JSONs behind every table in this post, and the per-prompt trace artifacts. Total compute: a few hours on a laptop (MPS for generation, CPU for analysis).