#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Minimal user-facing entry point for building a clean benchmark JSON from pipeline QA outputs.

set -euo pipefail

# Edit these settings before running.
QA_SUMMARY="00-Data_and_Models/construction_pipeline_outputs/02_qa_outputs/qa_run_summary.json"
OUTPUT_DIR="00-Data_and_Models/benchmarks"
BENCHMARK_NAME="GlobalDentBench"

if [[ ! -f "construction_pipeline/03-BuildBenchmark.py" ]]; then
    echo "Please run this script from the GlobalDentBench project root:"
    echo "bash examples/02-buildBenchmark.sh"
    exit 1
fi

python construction_pipeline/03-BuildBenchmark.py \
    --qa-summary "$QA_SUMMARY" \
    --output-dir "$OUTPUT_DIR" \
    --benchmark-name "$BENCHMARK_NAME"
