# benchcheck - Design & Threat Model

This document explains *why* the tool is built the way it is. It doubles as the
record of the statistical reasoning and the failure modes found while building
it - the parts worth being able to discuss.

## 1. Problem & threat model

**Claim under attack:** "Model M scores X% on benchmark B, therefore M has
capability X."

**The threat:** *benchmark contamination* - some of B's items were present in
M's training data, so M's score reflects memorization, not capability. We assume
no access to M's training set (the realistic case), so contamination must be
inferred from M's *behavior*.

**What we can observe**, in decreasing order of access:
- the benchmark text itself (always);
- M's generated continuations (any model, incl. hosted APIs);
- M's token log-probabilities (only open-weight models we run ourselves).

The tool is designed to extract signal from whatever level of access is
available - this is the Tier A / Tier B split.

**Out of scope (stated honestly):** we detect *memorization fingerprints*, not
contamination directly. A model can memorize without contamination (e.g. a fact
repeated across the web), and can be contaminated without strong memorization
(seen once, low capacity). The tool estimates a *correlate*, and the calibration
experiment quantifies how good a correlate it is.

## 2. Signals - and why each exists

Each signal targets a different observable consequence of memorization, so they
fail independently (the property the aggregator relies on).

- **`ngram_overlap`** (no model): the most direct evidence - the question's
  exact wording sitting in a candidate training corpus. Long n-grams (n=8) avoid
  chance collisions on common phrases. Weak alone (you rarely have the real
  training set) but valuable corroboration.
- **`verbatim_completion`** (generation): memorization's behavioral tell - feed
  a prefix, see if the model continues the *exact* original. We require a
  contiguous run of at least 3 words to count, because 1-2 shared common words happen by
  chance.
- **`perturbation_gap`** (logprobs): the flagship. Compare per-token logprob of
  the original vs. meaning-preserving paraphrases. Understanding is
  paraphrase-invariant; memorization is not. Per-token normalization prevents
  string length from confounding the comparison.
- **`shuffle_sensitivity`** (logprobs): for multiple-choice, genuine capability
  is invariant to option order; memorized "the answer is B" is not. We measure
  the variance of the probability assigned to the correct option across
  deterministic shuffles.

## 3. The aggregation problem - and three bugs that shaped it

The combination layer went through three failed designs before the current one.
Each failure was informative, so they're recorded here.

**Attempt 1 - empirical p-value, flag the bottom 10%.** Flagged the items in the
upper tail of the combined-score distribution. *Failure:* this structurally caps
detection at 10% of items, so at 30% true contamination recall was 0.33 by
construction. Worse, contaminated items pollute the very distribution they're
compared against. **Lesson:** an empirical-null tail test silently assumes
contamination is *rare*.

**Attempt 2 - Otsu split + fixed separation cutoff.** Find the threshold that
best separates a low and high cluster; flag if they're separated by >2 pooled
std-devs. *Failure:* Otsu *always* finds a split, even in unimodal noise, where
the "separation" is ~2.6 std-devs by chance. Clean benchmarks got 80% of items
flagged. **Lesson:** "a cluster exists" is not evidence; you must compare against
what noise produces.

**Attempt 3 - Otsu + Gaussian Monte-Carlo null.** Calibrate the separation
cutoff by running Otsu on Gaussian noise. *Failure:* the signals aren't
Gaussian - they're zero-inflated and discrete, which Otsu separates far more
than smooth noise, so the Gaussian null underestimated the floor. **Lesson:**
the null must match the data's actual marginal shape.

**Current design - per-signal Otsu + corroboration.** Two ideas:
1. *Per-signal* Otsu finds each signal's high cluster. Otsu (not median/MAD)
   because the scores are zero-inflated and can be contaminated past 50%, where
   robust location estimators break down (the median moves into the
   contaminated mass).
2. *Corroboration*: flag a question only if **2 or more independent signals** fire on
   it. This is the operationalization of the core insight - independent signals
   coincide on truly memorized items but only randomly on clean ones. If fewer
   than two signals are informative, corroboration is impossible and the tool
   flags nothing rather than trust a lone signal.

This is the only design that is simultaneously silent on clean data (0% false
positives) and sensitive across 10-70% contamination.

## 4. Honest reporting of the rate

The flagged fraction is reported as a **detected rate**, explicitly a *lower
bound*: because recall < 1, it systematically undershoots the true rate. The
bootstrap CI captures sampling *variance*, not this *bias* - so we don't claim
the CI covers the truth. The calibration experiment exists precisely to *measure*
the bias (via recall) so a user knows how to interpret the number.

## 5. Validation methodology

`calibration.py` is the experiment that turns "a script that prints numbers" into
"a validated tool":
1. Generate a synthetic benchmark with **known** ground-truth labels.
2. Construct a model (the mock) that has memorized a known subset.
3. Run the detector **blind**.
4. Report precision, recall, F1, and false-positive rate against ground truth.

The verdict is based on **detection quality**, not the rate point-estimate,
because precision/recall/FPR are what generalize to a real model.

## 6. The mock model

The full pipeline runs without downloading a multi-GB model because of a
deterministic `MockModel` that reproduces the *one* behavior under test:
memorized text gets inflated logprob and verbatim completions; everything else
is near-uniform noise. This is a testability decision - it lets the calibration
experiment and all 26 tests run in under a second and stay reproducible (no
RNG without an explicit seed). Swapping in a real model is a one-flag change
(`--model hf:gpt2`) thanks to the `Scorer`/`Generator` protocol split.

## 7. Performance

The log-prob checks dominate runtime: `shuffle_sensitivity` scores ~16 rendered
blocks per multiple-choice question. The optimization was batching - score all
blocks for a question in one padded forward pass instead of one at a time
(`HFModel.logprob_batch`). Two correctness traps showed up and are worth noting,
because batched scoring must be numerically identical to single scoring:
  - **Padding leakage:** with left padding, the first real token is predicted
    from a pad token. Fixed by only counting a position when both it and its
    predecessor are real (`mask[:,1:] * mask[:,:-1]`).
  - **Position embeddings:** gpt2 uses absolute position embeddings, so left
    padding shifts real tokens to the wrong positions. Fixed by passing
    `position_ids` derived from the attention mask.
After both fixes, batched and single scoring agree to <1e-3. On CPU the batch
gives ~25% wall-time improvement; on GPU the gain is larger.

### C++ n-gram scanner

The `ngram_overlap` check has a different cost profile: no model, but a large
reference corpus to scan. For a realistic contamination audit the corpus could
be millions of documents, and the work is per-token hashing plus set membership
over tens of millions of n-grams - exactly where Python's per-object overhead
hurts. `native/ngram_scan.cpp` (pybind11) does it with a rolling FNV-1a hash,
n-grams stored as 64-bit integers in one flat `unordered_set`, and no per-n-gram
Python allocation.

Measured (median of 7 trials, 80k-doc synthetic corpus, 7.44M distinct n-grams):

| | Build throughput | Scan throughput | Speedup vs Python |
|---|---|---|---|
| C++ | ~1.1M n-grams/s | ~1.0M query-n-grams/s | build 1.7x, scan 3.2x |

The speedup is real but bounded: Python's `set` is already C-backed, so the win
comes from removing tuple allocation and repeated hashing, not from algorithmic
change. I kept an identical pure-Python backend (`corpus_index.py`) and a test
asserting the two agree exactly - the C++ path is an optimization, never a
correctness dependency, and the tool runs fine if the extension isn't built.

## 8. What I'd do next

- Replace synthetic paraphrases with a held-out paraphrase model for stronger
  controls.
- Add a learned combiner (logistic regression over signals) trained on the
  calibration data, vs. the current unweighted vote.
- Run against larger open-weight models and more benchmarks, and publish a
  ranking by apparent contamination. The gpt2 / ARC-Easy run in FINDINGS.md is
  the first data point.
