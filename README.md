# GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration

<div align="center">
<h3>
  GlobalDentBench
</h3>
</div>

<div align="center">
<h4>
  📃 <a href="https://arxiv.org/abs/2605.24636" target="_blank">Paper</a> ｜ 🤗 <a href="https://huggingface.co/datasets/FreedomIntelligence/GlobalDentBench-OA" target="_blank">GlobalDentBench (OA)</a>
</h4>
</div>

---

## ⚡ Introduction

Hello! Welcome to the repository for **GlobalDentBench**, a multinational and full-spectrum dental benchmark designed to evaluate the clinical reasoning robustness and safety of Large Language Models (LLMs) in dentistry.

<div align=center>
<img src="assets/figure1_benchmark_overview.jpg" width = "80%" alt="GlobalDentBench Overview" align=center/>
<p><em>Figure 1: Overview of GlobalDentBench covering global sources, 3 reasoning levels, 14 dental disciplines, and 3 data types.</em></p>
</div>

While LLMs show transformative potential in medical knowledge replication, real-world clinical environments demand higher cognitive reasoning under high stakes. **GlobalDentBench** bridges the gap between closed-form factual recall and authentic clinical decision-making through **8,978 questions** across **14 dental specialties**, spanning **88 countries and regions** across 6 continents. To ensure clinical validity and consistency, the benchmark was calibrated and verified by **6 senior dentists** with an average of **6.8 years of clinical practice**, contributing a total of **297 person-hours** of expert review and validation.

---

## ⚙️ Installation and Configuration

Install the environment.

```bash
pip install -r requirements.txt

# Optional, only when running PDF/image OCR with the bundled DeepSeek-OCR2 runtime:
pip install -r requirements-ocr.txt
```

*Note: Please install `pandoc` independently via your system package manager if you require Word/RTF/ODT/HTML document conversions.*

1. Edit fields in `config/config.json` to configure target model endpoint URLs and access keys.
2. For local PDF/image OCR capabilities, specify the path to your local DeepSeek-OCR-2 weights:

```json
{
  "deepseek_ocr2": {
    "runtime_root": "utils/deepseek_ocr2_runtime",
    "model_path": "/path/to/DeepSeek-OCR-2",
    "gpu_memory_utilization": 0.9
  }
}
```

---

## 🚀 Usage

### Option 1: Evaluate Your Model on GlobalDentBench-OA (Recommended)

If you only want to benchmark your model on GlobalDentBench, simply download the Open-Access benchmark subset (**GlobalDentBench-OA**) from Hugging Face and run the evaluation pipeline.

> **🔒 Copyright & Data Request Notice:** > Due to publishing copyright protections on authoritative textbooks and specific testing databases, access to the full raw source text corpus is restricted. **This repository provides an Open-Access (OA) test subset via HuggingFace**. If you require the full benchmark corpus for educational research, please email us to verify credentials and obtain the structural directory required for local pipeline compilation.

```bash
bash examples/03-evaluation.sh      # Run model inference and benchmark evaluation, please define the model name in config/config.json
bash examples/04-riskAnalysis.sh    # Clinical risk analysis (S0/S1/S2)
bash examples/05-resultAnalysis.sh  # Aggregate results by type / level / discipline
```

This workflow does **not** require running the benchmark construction pipeline.

---

### Option 2: Reproduce the Benchmark Construction Pipeline

For researchers interested in reproducing the GlobalDentBench construction framework or building new domain-specific benchmarks, the complete construction pipeline is also provided.

```bash
bash examples/01-construction_pipeline.sh   # Raw Documents → Intermediate QA JSON
bash examples/02-buildBenchmark.sh          # Intermediate QA JSON → Final benchmark file
```

The pipeline converts heterogeneous source documents (PDFs, XML files, images, textbooks, examinations, and case reports) into standardized benchmark samples with unified taxonomy and reasoning-level annotations.

---

### Generated Outputs (under your configured `OUTPUT_ROOT`)

```text
01_markdown_and_metadata/             # Cleaned markdown extractions & unified metadata
02_qa_outputs/qa_run_summary.json     # Extracted MCQ / SAQ / CBQ candidates
buffer/                               # Volatile processing files
```

---

## 📚 Benchmark Features

GlobalDentBench evaluates LLMs through a multi-dimensional clinical lens, shifting evaluation metrics from superficial statistics to rigorous, patient-centric criteria.

### 1. Hierarchical Cognitive Framework
The benchmark stratifies tasks into three progressive reasoning levels ($L1 \rightarrow L3$):
* **L1: Knowledge Recall:** Assessing direct retrieval of foundational dental knowledge without case-based reasoning.
* **L2: Routine Reasoning:** Assessing clinical logic based on core clinical presentation and typical diagnostic/therapeutic tracks.
* **L3: Individualized Reasoning:** Assessing complex, multi-step optimization requiring patient-specific constraints, spatial analysis, or non-standard clinical considerations.

### 2. Multi-Format Clinical Question Types
| Question Type | Samples | Primary Source Origin | Cognitive Focus |
| :--- | :---: | :--- | :--- |
| **MCQ** (Multiple-Choice Questions) | **3,679** | National dental qualification & licensure exams (US, UK, AU, CA, NZ, IN) | Standardized knowledge recall & foundational matching |
| **SAQ** (Short-Answer Questions) | **3,709** | Authoritative dental textbooks (e.g., *Diagnosis and Treatment Planning in Dentistry*) | Text-based structured response generation |
| **CBQ** (Case-Based Questions) | **1,590** | Peer-reviewed clinical case reports from high-impact journals (e.g., *JADA*) | Scenario-driven reasoning under clinical ambiguity |

### 3. Comprehensive 14-Discipline Taxonomy
GlobalDentBench provides fine-grained, specialty-aware evaluation across 14 distinct fields:
* *Anesthesia & Medical Emergencies (AME)*, *Basic Sciences & Preventive Dentistry (BSPD)*, *Caries, Tooth Defects & Trauma (CTDT)*, *Conventional Prosthodontics (CP)*, *Dentoalveolar Surgery (DS)*, *Maxillofacial Diseases & Surgery (MFDS)*, *Oral & Maxillofacial Radiology (OMR)*, *Oral Implantology (OI)*, *Oral Mucosal Diseases (OMD)*, *Orthodontics (Ortho)*, *Pediatric Dentistry (PD)*, *Pulp & Periapical Diseases (PPD)*, *Periodontal & Peri-implant Diseases (PP)*, *Systemic Health, Pharmacology & Safety (SHPS)*.

---

## 🤖 Construction Pipeline & Evaluation

To support scalability and absolute trustworthiness, GlobalDentBench combines an advanced **automated LLM agent pipeline** with a strict **Dentist-in-the-Loop validation framework** (accumulating 297 senior-dentist person-hours).

<div align=center>
<img src="assets/figure2_pipeline.jpg" width = "80%" alt="GlobalDentBench Pipeline" align=center/>
<p><em>Figure 2: Three-stage agent pipeline for benchmark construction and type-specific evaluation protocol.</em></p>
</div>

<details open>
<summary><h4>🛠️ Automated & Expert-Calibrated Framework</h4></summary>

* **Stage I: Document Normalization:** A *Reformat Agent* leverages OCR (including DeepSeek-OCR2) and Parsers to transform heterogeneous raw inputs (PDFs, XMLs, Images) into a unified intermediate Markdown representation.
* **Stage II: Type-aware Construction:** An *Extract Agent* builds type-specific QA structures embedded with a *Self-Correction Loop* (up to 3 validation iterations per item).
* **Stage III: Unified Tagging & Final Audit:** A *Tag Agent* utilizes a majority-voting consistency protocol to assign disciplines and reasoning levels. Senior dentists manually audited a critical subset, confirming a human-expert agreement rate of **99.98% for MCQs/SAQs** and **96.78% for complex CBQs**.
* **Evaluation Protocol:** MCQs are scored via exact match. SAQs and CBQs use a rubric-based automated judge framework spearheaded by *Gemini-3-Flash-Preview* (validated to have over 98% concordance with human expert grading).

</details>

<details>
<summary><h4>⚠️ Zero-Shot Clinical Risk Analysis</h4></summary>

Beyond final-answer accuracy, the evaluation pipeline embeds a safety-risk classifier to assess the downstream safety risks of LLM-generated treatment recommendations:
* **S0:** Clinically safe response.
* **S1:** Unsafe response with potential for *reversible* patient harm.
* **S2:** Unsafe response with potential for *irreversible* patient harm (e.g., severe failures concentrated in *Systemic Health, Pharmacology, and Safety*).

</details>

---

## 📂 Layout

```text
construction_pipeline/   # Convert documents → Markdown → QA JSON → benchmark file
evaluation_pipeline/     # Evaluate models, run risk analysis, summarize results
examples/                # Minimal user-facing entry scripts (01–05)
utils/                   # Shared LLM API wrapper and bundled DeepSeek-OCR2 runtime
config/config.json       # Runtime config template (no real secrets)
data/                    # Prompt YAML references
```

---

## 📖 About Us

This project is a collaborative effort brought together by:

- Faculty of Dentistry, The University of Hong Kong 香港大学牙医学院
- The Chinese University of Hong Kong, Shenzhen 香港中文大学（深圳）
- Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University 南方医科大学深圳口腔医院（坪山）
- Peking University 北京大学
- Peking-Tsinghua Center for Life Sciences 北大清华生命科学联合中心
- National Biomedical Imaging Center 国家生物医学成像中心
- New Cornerstone Science Laboratory 新基石科学实验室
- Mayo Clinic 梅奥诊所
- LMU University Hospital 德国慕尼黑大学医院
- Shenzhen Loop Area Institute 深圳河套学院
- Freedom AI 深圳自由动脉科技有限公司

---

## 📮 Contact & Data Access

If you have any questions or are interested in collaborating, feel free to reach out via:
📧 **zhenyangcai@link.cuhk.edu.cn** or **junjiezhao@connect.hku.hk**

---

## ✨ Citation

If you find this benchmark helpful, or use this pipeline framework to evaluate other clinical subfields, please kindly cite our work:

```bibtex
coming soon~
```

---

## 📄 License

This project is officially open-sourced under the **MIT License**.
