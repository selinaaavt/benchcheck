# Findings: contamination in ARC-Easy as seen by gpt2

This is a real run of benchcheck against a real model and a real public
benchmark. It is not a synthetic demo.

## Setup

- **Model:** gpt2 (124M parameters), run on CPU.
- **Benchmark:** ARC-Easy test split (AI2 Reasoning Challenge), a multiple-choice
  grade-school science benchmark.
- **Checks run:** all four (gpt2 exposes token log-probabilities, so both the
  generation-based and log-prob-based checks apply).
- **Reference corpus for n-gram overlap:** a small sample corpus; the n-gram
  check is therefore the weakest contributor here and mostly abstains.

- **Questions analyzed:** 500
- **Flagged as likely contaminated:** 86 (17.2%), 95% CI 14.0%–20.6%
- **Throughput:** 0.25 questions/sec (4031 ms/question) on CPU, 500 questions in 2016 s

Mean score per check:

| Check | Mean score |
|---|---|
| `ngram_overlap` | 0.000 |
| `perturbation_gap` | 0.145 |
| `shuffle_sensitivity` | 0.639 |
| `verbatim_completion` | 0.010 |

Top flagged questions (and which checks fired):

- `Mercury_7271740` (score +2.04): perturbation_gap 0.88, shuffle_sensitivity 0.78, verbatim_completion 0.25
- `Mercury_7033548` (score +2.00): perturbation_gap 0.92, shuffle_sensitivity 0.39, verbatim_completion 0.33
- `MCAS_1999_8_9` (score +1.83): shuffle_sensitivity 0.58, verbatim_completion 0.36, perturbation_gap 0.30
- `Mercury_414097` (score +1.76): shuffle_sensitivity 0.70, verbatim_completion 0.38
- `ACTAAP_2007_7_18` (score +1.62): shuffle_sensitivity 0.76, verbatim_completion 0.33
- `Mercury_SC_401624` (score +1.53): shuffle_sensitivity 0.68, verbatim_completion 0.33
- `Mercury_402144` (score +1.46): shuffle_sensitivity 0.76, verbatim_completion 0.30
- `MEAP_2005_8_14` (score +1.45): shuffle_sensitivity 0.70, perturbation_gap 0.43, verbatim_completion 0.24

## How to read this

The flagged rate is a **lower bound** on true contamination. The detector only
flags a question when at least two independent checks agree, which keeps false
positives low but means borderline cases are missed (recall is below 1 in the
calibration experiment).

Every flagged question was selected independently by two or three of the checks.
The strongest signals came from `perturbation_gap` (gpt2 assigns higher
probability to the exact original wording than to a paraphrase) and
`shuffle_sensitivity` (its preferred answer changes when options are reordered).

## Caveats, stated plainly

- This detects **memorization fingerprints**, which correlate with contamination
  but are not identical to it. A model can be confident on a question for reasons
  other than having trained on it.
- gpt2 is a small, old model. It is a convenient first target, not a model
  anyone benchmarks seriously today. The point here is that the tool produces
  sensible, corroborated results on real inputs.
- The n-gram check needs a real candidate training corpus to be useful; with the
  small sample corpus used here it mostly contributes nothing.

## Reproduce

```bash
python -m examples.fetch_arc --split test --out examples/arc_easy_test.jsonl --limit 500
python -m benchcheck detect --dataset examples/arc_easy_test.jsonl \
    --model hf:gpt2 --timing --show-flagged --output examples/results_gpt2_arc500.json
```
