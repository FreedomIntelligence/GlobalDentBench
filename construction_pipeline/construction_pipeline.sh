#!/usr/bin/env bash
# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Launch the construction_pipeline producer-consumer workflow; 01-ToMarkdown.py writes to buffer, 02-QA_Generation.py monitors buffer and processes continuously.

set -euo pipefail

PROJECT_ROOT="$(pwd)"
STAGE1_OUTPUT_ROOT="00-Data_and_Models/construction_pipeline/01_markdown_and_metadata"
STAGE2_OUTPUT_ROOT="00-Data_and_Models/construction_pipeline/02_qa_outputs"
BUFFER_DIR="00-Data_and_Models/construction_pipeline/buffer"
CONFIG_TYPE="llm"
DOCUMENT_TYPE="auto"
TOKEN_COUNT_MODE="auto"
RECURSIVE=0
FORCE_REPROCESS=0
MODEL_OVERRIDE=""
TOKENIZER_SOURCE=""
INPUT_PATH=""
STAGE1_WORKER_COUNT=1
STAGE1_GPU_COUNT=1
STAGE1_GPU_MEM="0.9"
STAGE2_WORKER_COUNT=1
BUFFER_POLL_INTERVAL="2"
BUFFER_IDLE_EXIT_SECONDS="20"
PROGRESS_STATE=""
PRODUCER_DONE_SIGNAL=""

print_usage() {
    cat <<'EOF'
Usage:
  bash construction_pipeline/construction_pipeline.sh --input-path <file or directory> --document-type CaseReport

Common parameters:
  --input-path <path>          Required, input file or directory path
  --document-type <type>       Optional, auto / MCQ / SAQ / CaseReport, default auto
  --recursive                  Optional, recursively process subdirectories
  --force-reprocess            Optional, force reprocess already processed documents
  --config-type <name>         Optional, default llm
  --token-count-mode <mode>    Optional, auto / local / api, default auto
  --tokenizer-source <value>   Optional, for local mode can pass tokenizer.json path or model name
  --model <name>               Optional, override model name in config.json
  --buffer-dir <path>          Optional, default 00-Data_and_Models/construction_pipeline/buffer
  --stage1-workers <num>       Optional, parallel worker count for stage 01, default 1
  --stage1-gpus <num>          Optional, available GPU count for stage 01, default 1
  --stage1-gpu-mem <ratio>     Optional, GPU memory utilization ratio for DeepSeek-OCR2 in stage 01, default 0.9
  --stage2-workers <num>       Optional, concurrent worker count for stage 02, default 1
  --buffer-poll-interval <s>   Optional, buffer polling interval for stage 02, default 2 seconds
  --buffer-idle-exit <s>       Optional, auto-exit after buffer idle for N seconds in stage 02, default 20 seconds
  --progress-state <path>      Optional, overall progress state file, default <buffer-dir>/progress_state.json
  --producer-done-signal <path> Optional, stage 01 completion signal file, default <buffer-dir>/producer_done.signal
  --stage1-output-root <path>  Optional, default 00-Data_and_Models/construction_pipeline/01_markdown_and_metadata
  --stage2-output-root <path>  Optional, default 00-Data_and_Models/construction_pipeline/02_qa_outputs

Examples:
  bash construction_pipeline/construction_pipeline.sh --input-path 00-Data_and_Models/input_documents --recursive --document-type CaseReport
  bash construction_pipeline/construction_pipeline.sh --input-path 00-Data_and_Models/input_documents --document-type CaseReport --stage1-workers 1 --stage1-gpus 1 --stage1-gpu-mem 0.85 --stage2-workers 8
  bash construction_pipeline/construction_pipeline.sh --input-path 00-Data_and_Models/test.mmd --document-type SAQ --token-count-mode api --stage2-workers 12
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --document-type)
            DOCUMENT_TYPE="$2"
            shift 2
            ;;
        --recursive)
            RECURSIVE=1
            shift
            ;;
        --force-reprocess)
            FORCE_REPROCESS=1
            shift
            ;;
        --config-type)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --token-count-mode)
            TOKEN_COUNT_MODE="$2"
            shift 2
            ;;
        --tokenizer-source)
            TOKENIZER_SOURCE="$2"
            shift 2
            ;;
        --model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --buffer-dir)
            BUFFER_DIR="$2"
            shift 2
            ;;
        --stage1-workers)
            STAGE1_WORKER_COUNT="$2"
            shift 2
            ;;
        --stage1-gpus)
            STAGE1_GPU_COUNT="$2"
            shift 2
            ;;
        --stage1-gpu-mem)
            STAGE1_GPU_MEM="$2"
            shift 2
            ;;
        --stage2-workers)
            STAGE2_WORKER_COUNT="$2"
            shift 2
            ;;
        --buffer-poll-interval)
            BUFFER_POLL_INTERVAL="$2"
            shift 2
            ;;
        --buffer-idle-exit)
            BUFFER_IDLE_EXIT_SECONDS="$2"
            shift 2
            ;;
        --progress-state)
            PROGRESS_STATE="$2"
            shift 2
            ;;
        --producer-done-signal)
            PRODUCER_DONE_SIGNAL="$2"
            shift 2
            ;;
        --stage1-output-root)
            STAGE1_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --stage2-output-root)
            STAGE2_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_PATH" ]]; then
    echo "Missing --input-path"
    print_usage
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/construction_pipeline/01-ToMarkdown.py" || ! -f "$PROJECT_ROOT/construction_pipeline/02-QA_Generation.py" ]]; then
    echo "Please run this script from the project root directory. Current directory: $PROJECT_ROOT"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PROGRESS_STATE" ]]; then
    PROGRESS_STATE="$BUFFER_DIR/progress_state.json"
fi
if [[ -z "$PRODUCER_DONE_SIGNAL" ]]; then
    PRODUCER_DONE_SIGNAL="$BUFFER_DIR/producer_done.signal"
fi
rm -f "$PRODUCER_DONE_SIGNAL"

count_supported_inputs() {
    python - "$INPUT_PATH" "$RECURSIVE" <<'PY'
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
recursive = sys.argv[2] == "1"
supported = {
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".mmd", ".markdown",
    ".html", ".htm", ".xml", ".xlsx", ".xls", ".csv", ".tsv",
}
if input_path.is_file():
    print(1 if input_path.suffix.lower() in supported else 0)
    raise SystemExit
pattern = "**/*" if recursive else "*"
count = sum(1 for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in supported)
print(count)
PY
}

initialize_progress_state() {
    local total_inputs="$1"
    mkdir -p "$(dirname "$PROGRESS_STATE")"
    python - "$PROGRESS_STATE" "$total_inputs" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1])
total = int(sys.argv[2])
payload = {
    "total": total,
    "stage1_completed": 0,
    "stage2_completed": 0,
    "stage1_failed": 0,
    "stage2_failed": 0,
    "start_time": __import__("time").time(),
}
progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

monitor_progress() {
    python - "$PROGRESS_STATE" "$STAGE2_PID" <<'PY'
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

progress_path = Path(sys.argv[1])
stage2_pid = int(sys.argv[2])

def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def format_seconds(value: float | None) -> str:
    if value is None or value < 0 or value == float("inf"):
        return "--:--:--"
    total = int(value)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

with tqdm(total=1, desc="Overall progress", dynamic_ncols=True) as bar:
    bar.total = 1
    while True:
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        total = max(int(payload.get("total", 0) or 0), 1)
        start_time = float(payload.get("start_time", time.time()) or time.time())
        stage1_done = int(payload.get("stage1_completed", 0) or 0) + int(payload.get("stage1_failed", 0) or 0)
        stage2_done = int(payload.get("stage2_completed", 0) or 0) + int(payload.get("stage2_failed", 0) or 0)
        overall_done = stage1_done + stage2_done
        overall_total = max(total * 2, 1)
        elapsed = max(0.0, time.time() - start_time)
        rate = overall_done / elapsed if elapsed > 0 else 0.0
        remaining = (overall_total - overall_done) / rate if rate > 0 else None

        bar.total = overall_total
        bar.n = min(overall_done, overall_total)
        bar.set_postfix_str(
            f"01={stage1_done}/{total} | 02={stage2_done}/{total} | "
            f"elapsed={format_seconds(elapsed)} | remaining={format_seconds(remaining)}"
        )
        bar.refresh()

        if stage2_done >= total and not pid_alive(stage2_pid):
            break
        if not pid_alive(stage2_pid) and stage1_done >= total:
            break
        time.sleep(1.0)

    bar.n = bar.total
    bar.refresh()
PY
}

TOTAL_INPUTS="$(count_supported_inputs)"
if [[ "$TOTAL_INPUTS" -le 0 ]]; then
    echo "No processable files found in input path: $INPUT_PATH"
    exit 1
fi
initialize_progress_state "$TOTAL_INPUTS"

STAGE2_CMD=(
    python construction_pipeline/02-QA_Generation.py
    --output-root "$STAGE2_OUTPUT_ROOT"
    --config-type "$CONFIG_TYPE"
    --document-type "$DOCUMENT_TYPE"
    --token-count-mode "$TOKEN_COUNT_MODE"
    --buffer-dir "$BUFFER_DIR"
    --watch-buffer
    --worker-count "$STAGE2_WORKER_COUNT"
    --buffer-poll-interval "$BUFFER_POLL_INTERVAL"
    --buffer-idle-exit-seconds "$BUFFER_IDLE_EXIT_SECONDS"
    --progress-state "$PROGRESS_STATE"
    --producer-done-signal "$PRODUCER_DONE_SIGNAL"
)

if [[ -n "$TOKENIZER_SOURCE" ]]; then
    STAGE2_CMD+=(--tokenizer-source "$TOKENIZER_SOURCE")
fi

if [[ -n "$MODEL_OVERRIDE" ]]; then
    STAGE2_CMD+=(--model "$MODEL_OVERRIDE")
fi

if [[ "$FORCE_REPROCESS" -eq 1 ]]; then
    STAGE2_CMD+=(--force-reprocess)
fi

echo "========== Starting 02-QA_Generation.py consumer =========="
printf '%q ' "${STAGE2_CMD[@]}"
echo
"${STAGE2_CMD[@]}" &
STAGE2_PID=$!
monitor_progress &
PROGRESS_MONITOR_PID=$!

STAGE1_CMD=(
    python construction_pipeline/01-ToMarkdown.py
    --input-path "$INPUT_PATH"
    --output-root "$STAGE1_OUTPUT_ROOT"
    --config-type "$CONFIG_TYPE"
    --buffer-dir "$BUFFER_DIR"
    --worker-count "$STAGE1_WORKER_COUNT"
    --gpu-count "$STAGE1_GPU_COUNT"
    --ocr-gpu-memory-utilization "$STAGE1_GPU_MEM"
    --progress-state "$PROGRESS_STATE"
)

if [[ "$RECURSIVE" -eq 1 ]]; then
    STAGE1_CMD+=(--recursive)
fi

if [[ -n "$MODEL_OVERRIDE" ]]; then
    STAGE1_CMD+=(--model "$MODEL_OVERRIDE")
fi

echo "========== Running 01-ToMarkdown.py producer =========="
printf '%q ' "${STAGE1_CMD[@]}"
echo
"${STAGE1_CMD[@]}"
touch "$PRODUCER_DONE_SIGNAL"

echo "========== Waiting for 02-QA_Generation.py to process buffer =========="
wait "$STAGE2_PID"
wait "$PROGRESS_MONITOR_PID"

echo "========== All completed =========="
echo "Stage 1 output directory: $STAGE1_OUTPUT_ROOT"
echo "Stage 2 output directory: $STAGE2_OUTPUT_ROOT"
echo "Buffer directory: $BUFFER_DIR"
echo "Progress state file: $PROGRESS_STATE"
