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

<!-- RESULTS_PLACEHOLDER -->

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
