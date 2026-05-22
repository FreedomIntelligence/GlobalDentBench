#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Minimal user-facing entry point for S0/S1/S2 risk analysis on evaluated CBQ answers.

set -euo pipefail

# Edit these settings before running.
BENCHMARK_PATH="00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
OUTPUT_PATH="00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
MODEL_CONFIGS=(
    # Leave empty to analyze all evaluated CBQ models.
    "gpt-5.4-nano"
    "deepseek-v4-flash-nothink"
)
JUDGE_CONFIG="llm"
CONCURRENCY=10
SAVE_EVERY=10

if [[ ! -f "evaluation_pipeline/03-RiskAnalysis.py" ]]; then
    echo "Please run this script from the GlobalDentBench project root:"
    echo "bash examples/04-riskAnalysis.sh"
    exit 1
fi

MODEL_CONFIGS_CSV="$(IFS=,; echo "${MODEL_CONFIGS[*]}")"

python evaluation_pipeline/03-RiskAnalysis.py \
    --benchmark "$BENCHMARK_PATH" \
    --output "$OUTPUT_PATH" \
    --models "$MODEL_CONFIGS_CSV" \
    --judge-config "$JUDGE_CONFIG" \
    --concurrency "$CONCURRENCY" \
    --save-every "$SAVE_EVERY"
