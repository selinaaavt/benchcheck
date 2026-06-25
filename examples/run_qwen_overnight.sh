#!/usr/bin/env bash
# Overnight run: Qwen2.5-0.5B (a modern, 2024 open-weight model) across several
# benchmarks with ALL FOUR checks, matching the gpt2 methodology. Slow on CPU
# (~30s/item with the shuffle check), so this is meant to run unattended for
# hours. Each benchmark writes its own JSON, and a failure on one does not stop
# the others.
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p results
MODEL="Qwen/Qwen2.5-0.5B"
TAG="qwen2.5_0.5b"

for b in openbookqa sciq arc_easy arc_challenge; do
  echo "=== $(date +%H:%M) $b: starting ==="
  python -m benchcheck detect \
      --dataset "examples/${b}.jsonl" \
      --model "hf:${MODEL}" \
      --corpus examples/sample_corpus.txt \
      --timing \
      --output "results/${TAG}_${b}.json" 2>&1 \
    | grep -vE "Warning|attention mask|Loading|HF_TOKEN" | tail -10 \
    || echo "!!! $b FAILED, continuing"
  echo "=== $(date +%H:%M) $b: done ==="
done
echo "ALL DONE $(date +%H:%M)"
