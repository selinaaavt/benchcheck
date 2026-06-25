# benchcheck

**A lie detector for LLM benchmark scores.**

When a model scores 90% on a benchmark, how much is real capability and how much
is the model having *memorized the test questions* from its training data?
`benchcheck` estimates that. It looks for the statistical fingerprints of
memorization and reports how contaminated a benchmark looks — per question and
in aggregate, with confidence intervals and an honest account of its own error
rates.

## Why this matters

Benchmark questions live on the internet. Models train on the internet. So part
of any headline benchmark score can be memorization, not intelligence — and if
we can't tell the two apart, we can't tell whether models are actually
improving. This is an active, unsolved problem in LLM evaluation; `benchcheck`
is a clean, validated tool for measuring it.

## How it works

`benchcheck` collects several *independent* signals, each a weak piece of
evidence that one question was memorized, then combines them. The core insight
driving the whole design: **a single signal's tail is just noise, but real
contamination makes multiple independent signals fire on the *same* questions.**
Detection requires corroboration.

### The signals

| Signal | Tier | Needs | Idea |
|---|---|---|---|
| `ngram_overlap` | B | nothing | Does the question's exact wording appear in a reference corpus the model may have trained on? |
| `verbatim_completion` | B | text generation | Given a prefix, does the model regurgitate the rest of the question word-for-word? |
| `perturbation_gap` | A | token logprobs | A memorizing model is suspiciously *more confident* on the exact original wording than on a meaning-preserving paraphrase. |
| `shuffle_sensitivity` | A | token logprobs | Real understanding survives shuffling multiple-choice options; memorization is sensitive to the canonical order. |

The tool detects which capabilities your model exposes and runs whichever
signals apply (**graceful degradation**): API-only models get Tier B;
open-weight models you run locally also get the stronger Tier A signals.

### The statistics (the heart of the project)

1. **Combine** — z-score each signal across items (so signals with different
   natural ranges are comparable) and average.
2. **Flag** — for each signal, find its "high" cluster via per-signal Otsu
   thresholding (robust to zero-inflated scores and to *any* contamination
   fraction, even >50% where median/MAD methods break down). Flag a question
   only when **≥2 signals corroborate**. If fewer than two signals are
   informative, the tool refuses to guess and flags nothing.
3. **Quantify** — bootstrap a 95% confidence interval on the contamination rate.

## Validation

The headline experiment (`benchcheck calibrate`) *proves the detector works*: it
builds a synthetic benchmark with known ground truth, tells a model to "memorize"
a known subset, runs the detector blind, and scores it. Representative results
(200-item benchmark):

| True contamination | Precision | Recall | False-positive rate |
|---|---|---|---|
| 0% (clean) | — | — | **0.00** |
| 10% | 0.62 | 1.00 | 0.07 |
| 30% | **0.91** | 0.80 | 0.04 |
| 50% | **0.93** | 0.80 | 0.06 |

A clean benchmark is never flagged; realistic contamination (10–50%) is caught
with high precision. The tool is honest about its hard regimes (very low and
very high contamination) rather than hiding them.

## Quick start

```bash
cd ~/personal/benchcheck
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. No-model demo: n-gram overlap on a toy contaminated/clean pair
python -m examples.demo

# 2. Full pipeline end-to-end on the sample benchmark
python -m examples.detect_demo

# 3. The validation experiment
python -m benchcheck calibrate --frac 0.3

# 4. Run the detector on your own benchmark
python -m benchcheck detect --dataset examples/sample_benchmark.jsonl \
    --model mock --corpus examples/sample_corpus.txt --show-flagged

# List signals / run the tests
python -m benchcheck signals
python -m pytest -q
```

### Using a real model

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python -m benchcheck detect --dataset your_benchmark.jsonl --model hf:gpt2
```

## Dataset format

JSONL, one item per line. `answer`/`choices` are optional:

```json
{"id": "q1", "prompt": "What is the capital of France?", "answer": "B", "choices": ["Lyon", "Paris", "Marseille", "Nice"]}
```

## Design

See [DESIGN.md](DESIGN.md) for the threat model, why each signal exists, the
statistical reasoning, and the failure modes discovered during development.

## Layout

```
benchcheck/
  types.py          Item, SignalResult
  models/           backend abstraction: base protocol, mock, HuggingFace
  perturb.py        meaning-preserving rewrites used as statistical controls
  signals/          the four signals + registry
  stats.py          combine -> per-signal Otsu -> corroboration -> bootstrap CI
  pipeline.py       glue: model + signals + stats, with graceful degradation
  calibration.py    the validation experiment
  cli.py            `detect`, `signals`, `calibrate`
examples/           runnable demos + sample benchmark and corpus
tests/              26 tests across all modules
```
