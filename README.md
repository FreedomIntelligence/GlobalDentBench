# GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration

<div align="center">
<h3>
  GlobalDentBench
</h3>
</div>

<div align="center">
<h4>
  📃 <a href="https://github.com/FreedomIntelligence/GlobalDentBench" target="_blank">Paper (Coming soon)</a> ｜ 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/GlobalDentBench-OA" target="_blank">GlobalDentBench (OA)</a>
</h4>
</div>

## ⚡ Introduction
Hello! Welcome to the repository for GlobalDentBench!

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

##  📖 About Us
We are from:
- Faculty of Dentistry, The University of Hong Kong 香港大学牙医学院
- The Chinese University of Hong Kong, Shenzhen 香港中文大学（深圳）
- Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University 南方医科大学深圳口腔医院（坪山）
- Peking University 北京大学
- Peking-Tsinghua Center for Life Sciences 北大清华生命科学联合中心
- National Biomedical Imaging Center 国家生物医学成像中心
- New Cornerstone Science Laboratory 新基石科学实验室
- Mayo Clinic 梅奥诊所
- LMU University Hospital 德国慕尼黑大学医院 
- Freedom AI 深圳自由动脉科技有限公司


## ✨ Citation

If you use this code or refer to our method, please cite our paper. This is very important for us🤩:

> GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration.

---

## 📮 Contact

If you have any questions, please contact us🧐: [zhenyangcai@link.cuhk.edu.cn] or [junjiezhao@connect.hku.hk].

Due to copyright restrictions, only an open-access (OA) version of GlobalDentBench is provided. If you require the full dataset, please contact us via email to receive the original materials directory, which you can download and process according to the pipeline.

## 📄 License
This project is open-sourced under the MIT License.
