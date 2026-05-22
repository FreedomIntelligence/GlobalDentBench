#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Minimal user-facing entry point for concurrent benchmark evaluation with breakpoint resume.

set -euo pipefail

# Edit these settings before running.
BENCHMARK_PATH="00-Data_and_Models/benchmarks/GlobalDentBench_newL1-L3-bench-small.json"
OUTPUT_PATH="00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
MODEL_CONFIGS=(
    "gpt-5.4"
    "gemini-3.1-pro-preview"
    "gemini-3-flash-preview"
)
JUDGE_CONFIG="llm"
CONCURRENCY=50
SAVE_EVERY=10

if [[ ! -f "evaluation_pipeline/01-EvaluateBenchmark.py" ]]; then
    echo "Please run this script from the GlobalDentBench project root:"
    echo "bash examples/03-evaluation.sh"
    exit 1
fi

if [[ ${#MODEL_CONFIGS[@]} -eq 0 ]]; then
    echo "Please add at least one model config name to MODEL_CONFIGS."
    exit 1
fi

MODEL_CONFIGS_CSV="$(IFS=,; echo "${MODEL_CONFIGS[*]}")"

python evaluation_pipeline/01-EvaluateBenchmark.py \
    --benchmark "$BENCHMARK_PATH" \
    --output "$OUTPUT_PATH" \
    --models "$MODEL_CONFIGS_CSV" \
    --judge-config "$JUDGE_CONFIG" \
    --concurrency "$CONCURRENCY" \
    --save-every "$SAVE_EVERY"
