#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Minimal user-facing entry point for running the GlobalDentBench construction pipeline.

set -euo pipefail

# Edit these two paths before running.
INPUT_PATH="00-Data_and_Models/input_documents"
OUTPUT_ROOT="00-Data_and_Models/construction_pipeline_outputs"

# Optional settings.
DOCUMENT_TYPE="CaseReport"   # CaseReport for CBQ; also supports MCQ and SAQ.
RECURSIVE=1                  # 1 = scan subfolders, 0 = current folder only.
STAGE1_WORKERS=2
STAGE1_GPUS=2
STAGE2_WORKERS=2

if [[ ! -f "construction_pipeline/construction_pipeline.sh" ]]; then
    echo "Please run this script from the GlobalDentBench project root:"
    echo "bash examples/01-construction_pipeline.sh"
    exit 1
fi

RECURSIVE_ARG=()
if [[ "$RECURSIVE" -eq 1 ]]; then
    RECURSIVE_ARG=(--recursive)
fi

echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_ROOT"
echo "Type:   $DOCUMENT_TYPE"

bash construction_pipeline/construction_pipeline.sh \
    --input-path "$INPUT_PATH" \
    --document-type "$DOCUMENT_TYPE" \
    "${RECURSIVE_ARG[@]}" \
    --stage1-output-root "$OUTPUT_ROOT/01_markdown_and_metadata" \
    --stage2-output-root "$OUTPUT_ROOT/02_qa_outputs" \
    --buffer-dir "$OUTPUT_ROOT/buffer" \
    --progress-state "$OUTPUT_ROOT/buffer/progress_state.json" \
    --producer-done-signal "$OUTPUT_ROOT/buffer/producer_done.signal" \
    --stage1-workers "$STAGE1_WORKERS" \
    --stage1-gpus "$STAGE1_GPUS" \
    --stage1-gpu-mem 0.85 \
    --stage2-workers "$STAGE2_WORKERS"
