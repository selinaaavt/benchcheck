#!/usr/bin/env bash
# Run gpt2 detection across several benchmarks, one JSON report each.
# Intended to run in the background (each run is ~10-20 min on CPU).
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p results
for b in arc_challenge openbookqa sciq; do
  echo "=== $b: starting ==="
  python -m benchcheck detect \
      --dataset "examples/${b}.jsonl" \
      --model hf:gpt2 \
      --corpus examples/sample_corpus.txt \
      --timing \
      --output "results/gpt2_${b}.json" 2>&1 \
    | grep -vE "Warning|attention|Loading|HF_TOKEN" | tail -12
  echo "=== $b: done ==="
done
echo "ALL DONE"
