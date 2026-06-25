# benchcheck

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/selinaaavt/benchcheck/blob/main/notebooks/benchcheck_colab.ipynb)

Estimates how much of an LLM's benchmark score comes from memorizing the test
questions instead of actually answering them.

> Want to run it on a modern model with a GPU? Click the Colab badge above — it
> clones the repo, fetches the benchmarks, and runs the detector on Qwen2.5-3B
> across four benchmarks in a few minutes on a free T4.

Benchmark questions are all over the public internet, and models train on the
public internet. So when a model scores well, some of that score can be
recall of questions it already saw during training ("contamination") rather
than capability. benchcheck measures how much, without needing access to the
model's training data.

## What it does

Given a benchmark (a list of questions) and a model, benchcheck runs four
independent checks for memorization, combines them, and reports which questions
look contaminated plus an overall rate with a confidence interval.

It works with whatever model access you have. With only text generation (e.g. a
hosted API) it runs two checks; with token log-probabilities (an open-weight
model you run yourself) it runs all four.

## The four checks

| Check | Needs | What it looks for |
|---|---|---|
| `ngram_overlap` | nothing | The question's exact wording appearing in a reference corpus the model may have trained on. |
| `verbatim_completion` | text generation | The model completing a question prefix with the original wording, word for word. |
| `perturbation_gap` | log-probs | The model being more confident on the exact original wording than on a paraphrase that means the same thing. |
| `shuffle_sensitivity` | log-probs | A multiple-choice answer that changes when the options are reordered (real understanding shouldn't care about order). |

No single check is reliable on its own. The detector only flags a question when
at least two checks agree on it. That requirement is what keeps it quiet on
clean benchmarks.

## Results on a real model

Running gpt2 against four public multiple-choice science benchmarks (1,400
questions total, on CPU), ranked by the fraction showing memorization
fingerprints:

| Benchmark | Questions | Flagged | Rate | 95% CI |
|---|---|---|---|---|
| ARC-Challenge | 300 | 56 | 18.7% | 14.3%–23.3% |
| ARC-Easy | 500 | 86 | 17.2% | 14.0%–20.6% |
| SciQ | 300 | 45 | 15.0% | 11.0%–19.0% |
| OpenBookQA | 300 | 40 | 13.3% | 9.7%–17.3% |

Each flagged question was independently picked out by at least two checks. The
rates are a lower bound (recall < 1). See [FINDINGS.md](FINDINGS.md) for the full
writeup, per-benchmark detail, and example questions.

## How the scoring works

1. Each check produces a per-question score (higher = more suspicious).
2. For each check, find its "high" cluster using Otsu thresholding. This is
   robust to most questions scoring zero and works at any contamination level,
   including above 50% where median-based thresholds fail.
3. Flag a question only if at least two checks put it in their high cluster.
4. Bootstrap a 95% confidence interval on the flagged fraction.

## Validation

`benchcheck calibrate` builds a synthetic benchmark with known answers, tells a
model to memorize a known subset, runs the detector blind, and scores it against
the truth. Results on a 200-question benchmark:

| True contamination | Precision | Recall | False positives (clean items) |
|---|---|---|---|
| 0% | — | — | 0% |
| 10% | 0.62 | 1.00 | 7% |
| 30% | 0.91 | 0.80 | 4% |
| 50% | 0.93 | 0.80 | 6% |

A clean benchmark is never flagged. Contamination in the 10–50% range is caught
with high precision. The flagged rate is a lower bound on the true rate, since
recall is below 1.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m examples.demo            # n-gram check on a toy example, no model
python -m benchcheck calibrate     # the validation experiment
python -m benchcheck signals       # list the checks

# Run on a real model (downloads gpt2 the first time)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets
python -m examples.fetch_arc --limit 200 --out examples/arc.jsonl
python -m benchcheck detect --dataset examples/arc.jsonl --model hf:gpt2 \
    --timing --show-flagged --output report.json

python -m pytest -q                # 26 tests
```

## Dataset format

JSONL, one question per line. `answer` and `choices` are optional:

```json
{"id": "q1", "prompt": "What is the capital of France?", "answer": "B", "choices": ["Lyon", "Paris", "Marseille", "Nice"]}
```

## Layout

```
benchcheck/
  models/      model backends: protocol, mock (no deps), HuggingFace
  signals/     the four checks + a registry
  perturb.py   paraphrase / shuffle helpers used as controls
  stats.py     combine checks, find clusters, require agreement, bootstrap CI
  pipeline.py  run checks over a dataset, with timing
  calibration.py  the validation experiment
  cli.py       detect / signals / calibrate
examples/      demos, the ARC fetcher, sample data
tests/         26 tests
```

See [DESIGN.md](DESIGN.md) for the threat model, why each check exists, and the
bugs found while building the scoring layer.
