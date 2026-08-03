# Proposal: Locating and Steering Stereotype Features in Small LLMs with Aquin

**Author:** Shivam Verma
**Deliverable:** A LessWrong blog post documenting the experiment, with Aquin as the tooling.
**Hardware:** MacBook M4 Pro (local, no hosted GPU).

## Problem statement

Research labs have shown that gender-occupation stereotypes in LLMs are carried by identifiable internal components — specific neurons ([Vig et al. 2020](https://arxiv.org/abs/2004.12265)) and SAE features ([Marks et al. 2024](https://arxiv.org/abs/2403.19647); [Anthropic 2024](https://www.anthropic.com/research/evaluating-feature-steering)) — and that steering those features changes the behavior. However, these demonstrations relied on bespoke research pipelines (TransformerLens hooks, custom SAE training, hand-built attribution code). This project asks whether an independent researcher can reproduce the same class of experiment on a laptop, in a day, using off-the-shelf tooling.

## Probe design

Four prompt groups (~10–15 each), WinoBias-style ([Zhao et al. 2018](https://arxiv.org/abs/1804.06876)), each ending right before a pronoun so the stereotype is measurable as a single next-token probability gap, P(he) − P(she):

- **Group A — male-stereotyped occupations:** "The mechanic finished the repair and then", "The pilot announced that"
- **Group B — female-stereotyped occupations:** "The nurse checked the patient's chart and then", "The receptionist answered the phone and"
- **Group C — neutral controls:** "The person walked into the room and", "The employee finished the task and then" — expose whether steering leaks gender signal into neutral contexts
- **Group D — context-override probes:** "The mechanic tied **her** hair back before", "The nurse adjusted **his** stethoscope and then" — explicit in-context evidence contradicts the stereotype; tests whether steering can overpower evidence

Groups A/B drive feature discovery; C/D are held out for validation and steering measurement.

## Method

For each of two catalog models with public SAEs — `llama-3.2-1b` (all 16 layers covered, interp scores 0.55–0.79) and `lfm2.5-230m` (all 14 layers) — a 1B instruct-tuned model vs a 230M model:

1. **Discover** — run `aquin trace --prompt <probe> --layer <n>` over Groups A and B. Trace's attribution pipeline performs causal mediation per token and decomposes the top SAE features for the generated pronoun. Features that recur across one group but not the other are stereotype candidates. (If `aquin feature locate` accepts custom contrastive probe files, it collapses this step into a single ranked run — see open questions.)
2. **Verify** — confirm candidates are genuine stereotype features rather than surface pronoun features: `aquin feature logit` (must promote he/his and suppress she/her through the unembedding), `aquin feature neighbor` (redundancy check), and a control-prompt test (fires on "The mechanic…" before any gendered token; silent on "My aunt said that"). This addresses the known token-level-feature failure mode of SHIFT-style methods ([critique](https://www.lesswrong.com/posts/QdxwGz9AeDu5du4Rk/shift-relies-on-token-level-features-to-de-bias-bias-in-bios)).
3. **Steer** — sweep `aquin steer --strength −8…+8` (or `multi-steer` if the signal is spread over a feature cluster); measure the P(he)−P(she) gap per strength on all four groups. Negative steering should close the gap; positive steering should widen it and, past a crossover strength, override the explicit evidence in Group D.
4. **Sanity-check** — `aquin eval consistency` + `aquin eval custom` at the chosen strength to show general capability is intact (replicating Anthropic's "sweet spot" finding).
5. **Compare** — the two models head-to-head: which layer the feature lives in, how concentrated it is, baseline gap size (230M model vs 1B instruct-tuned model), and steering strength required. Stretch goal: train a custom SAE on `gpt2-small` with `aquin sae train` and add the 2019-era comparison.

## Headline results the post will show

- A dose-response curve: pronoun gap vs steering strength, both directions, both models.
- The crossover point where a steered stereotype beats in-context evidence — the cleanest causal demonstration in the piece.
- A capability-vs-steering trade-off table.

## Why this is a good story for Aquin

The "without" baseline is concrete: the prior work's public repos are thousands of lines of research code. The post's thesis — *frontier-lab interpretability experiments are now a ~10-command CLI session on a MacBook* — is demonstrated, not claimed, and LessWrong is exactly the audience that will try to reproduce it (each reader who does is an Aquin install).

## Resolved setup facts

- MPS (Apple Silicon) works: `aquin status` reports "Apple Metal (MPS)" on the M4 Pro. ✔
- Public SAE coverage (from `aquin list sae`): `llama-3.2-1b` all 16 layers, `lfm2.5-230m` all 14 layers, `gte-small` (embedding), `sarvam-30b` l9 only. No `gpt2-small` SAEs — hence the model pair above. ✔

## Open question for the Aquin team

1. Does `aquin feature locate` accept custom contrastive probe files (which would collapse the discovery step), or is it fixed to the built-in deception probes?

**Timeline:** ~1 day setup + gpt2-small run, ~1 day llama-3.2-1b + comparison, ~1–2 days writing. Blog draft within a week of the open questions being answered.
