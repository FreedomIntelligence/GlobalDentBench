# Towards Multimodal LLMs for Traditional Chinese Medicine

<div align="center">
<h3>
  GlobalDentBench
</h3>
</div>

A pipeline for building and evaluating a dental QA benchmark from raw documents.

Supported question types:

- `MCQ` — multiple-choice questions
- `SAQ` — short-answer questions
- `CBQ` — case-based questions

## Layout

```text
construction_pipeline/   # Convert documents → Markdown → QA JSON → benchmark file
evaluation_pipeline/     # Evaluate models, run risk analysis, summarize results
examples/                # Minimal user-facing entry scripts (01–05)
utils/                   # Shared LLM API wrapper and bundled DeepSeek-OCR2 runtime
config/config.json       # Runtime config template (no real secrets)
data/                    # Prompt YAML references
```

## Install

```bash
pip install -r requirements.txt
# Optional, only when running PDF/image OCR with the bundled DeepSeek-OCR2 runtime:
pip install -r requirements-ocr.txt
```

Install `pandoc` separately if you need Word/RTF/ODT/HTML conversion.

## Configure

- Edit `config/config.json` to set the model URLs and keys.

- For PDF/image OCR, set the local path to DeepSeek-OCR-2 weights:

  ```json
  {
    "deepseek_ocr2": {
      "runtime_root": "utils/deepseek_ocr2_runtime",
      "model_path": "/path/to/DeepSeek-OCR-2",
      "gpu_memory_utilization": 0.9
    }
  }
  ```

## Usage

Run from the project root. Edit each script's variables before running.

```bash
bash examples/01-construction_pipeline.sh   # Documents → QA JSON
bash examples/02-buildBenchmark.sh          # QA JSON → benchmark file
bash examples/03-evaluation.sh              # Run target models against the benchmark
bash examples/04-riskAnalysis.sh            # S0/S1/S2 clinical risk labels for CBQ
bash examples/05-resultAnalysis.sh          # Summary report by type / level / discipline
```

For advanced flags, see:

```bash
bash construction_pipeline/construction_pipeline.sh --help
```

## Output

Stage 1 + 2 outputs (under `OUTPUT_ROOT`):

```text
01_markdown_and_metadata/
02_qa_outputs/qa_run_summary.json     # MCQ / SAQ / CBQ
buffer/
```

Stage 3 produces a single benchmark JSON; stage 4–5 add evaluation, risk, and analysis outputs alongside it.
