#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Minimal user-facing entry point for analyzing evaluated benchmark results.

set -euo pipefail

# Edit these settings before running.
BENCHMARK_PATH="00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
OUTPUT_DIR="00-Data_and_Models/result_analysis"
REPORT_NAME="GlobalDentBench_result_analysis"
MODEL_CONFIGS=(
    # Leave empty to analyze all evaluated models.
    # "gpt-5.4-nano"
    # "deepseek-v4-flash-nothink"
)

if [[ ! -f "evaluation_pipeline/02-ResultAnalysis.py" ]]; then
    echo "Please run this script from the GlobalDentBench project root:"
    echo "bash examples/05-resultAnalysis.sh"
    exit 1
fi

MODEL_CONFIGS_CSV="$(IFS=,; echo "${MODEL_CONFIGS[*]}")"

python evaluation_pipeline/02-ResultAnalysis.py \
    --benchmark "$BENCHMARK_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --report-name "$REPORT_NAME" \
    --models "$MODEL_CONFIGS_CSV"
