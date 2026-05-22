# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Build a document conversion agent that routes files by suffix, converts them to Markdown (.mmd), and generates document metadata JSON with `utils.llm_api.call_gpt`, including MCQ, SAQ, and CaseReport type hints, while storing stage outputs under `00-Data_and_Models/`, and delegating PDF/image OCR to the bundled DeepSeek-OCR2 vLLM runtime.

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_config
from utils.llm_api import call_gpt


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_deepseek_ocr2_config() -> Dict:
    try:
        config = load_config()
    except FileNotFoundError:
        return {}
    ocr_config = config.get("deepseek_ocr2", {})
    return ocr_config if isinstance(ocr_config, dict) else {}


def resolve_ocr_runtime_path(ocr_config: Dict) -> Path:
    return resolve_project_path(
        os.getenv("DEEPSEEK_OCR2_VLLM_ROOT")
        or str(ocr_config.get("runtime_root", "") or "")
        or "utils/deepseek_ocr2_runtime"
    )


def resolve_ocr_model_path(ocr_config: Dict) -> Path:
    return resolve_project_path(
        os.getenv("DEEPSEEK_OCR2_MODEL_PATH")
        or str(ocr_config.get("model_path", "") or "")
        or "00-Data_and_Models/DeepSeek-OCR-2"
    )


def resolve_ocr_gpu_memory_utilization(ocr_config: Dict) -> float:
    value = os.getenv("DEEPSEEK_OCR2_GPU_MEMORY_UTILIZATION") or ocr_config.get(
        "gpu_memory_utilization",
        0.9,
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.9


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".txt",
    ".md",
    ".mmd",
    ".markdown",
    ".html",
    ".htm",
    ".xml",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
}

PANDOC_SUFFIXES = {
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".txt",
    ".md",
    ".mmd",
    ".markdown",
    ".html",
    ".htm",
}

TABULAR_SUFFIXES = {".xlsx", ".xls", ".csv", ".tsv"}
XML_SUFFIXES = {".xml"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_CONSTRUCTION_PIPELINE_STAGE1_OUTPUT_ROOT = "00-Data_and_Models/construction_pipeline/01_markdown_and_metadata"
MCQ_TYPE = "MCQ"
SAQ_TYPE = "SAQ"
CASE_REPORT_TYPE = "CaseReport"
DEEPSEEK_OCR2_CONFIG = load_deepseek_ocr2_config()
DEEPSEEK_OCR2_VLLM_ROOT = resolve_ocr_runtime_path(DEEPSEEK_OCR2_CONFIG)
DEEPSEEK_OCR2_MODEL_PATH = resolve_ocr_model_path(DEEPSEEK_OCR2_CONFIG)
DEFAULT_STAGE1_BUFFER_DIR = "00-Data_and_Models/construction_pipeline/buffer"
DEFAULT_OCR_GPU_MEMORY_UTILIZATION = resolve_ocr_gpu_memory_utilization(DEEPSEEK_OCR2_CONFIG)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def locked_json_state(path: Path):
    ensure_parent_dir(path)
    with open(path, "a+", encoding="utf-8") as handle:
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw_text = handle.read().strip()
        try:
            payload = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        yield payload
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_progress_state(
    progress_state_path: Optional[Path],
    stage_key: str,
    increment: int = 1,
) -> None:
    if progress_state_path is None:
        return
    with locked_json_state(progress_state_path) as payload:
        payload.setdefault("total", 0)
        payload.setdefault("stage1_completed", 0)
        payload.setdefault("stage2_completed", 0)
        payload.setdefault("stage1_failed", 0)
        payload.setdefault("stage2_failed", 0)
        payload.setdefault(stage_key, 0)
        payload[stage_key] += increment


def log_message(message: str) -> None:
    tqdm.write(message)


def estimate_token_count(text: str) -> int:
    return len(text.split())


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def normalize_document_type(document_type: str) -> str:
    normalized = str(document_type or "").strip().lower()
    if normalized in {"exam", "mcq"}:
        return MCQ_TYPE
    if normalized == "saq":
        return SAQ_TYPE
    if normalized in {"case report", "casereport", "case_report"}:
        return CASE_REPORT_TYPE
    return str(document_type or "").strip()


def run_pandoc_to_markdown(input_path: Path) -> str:
    command = [
        "pandoc",
        str(input_path),
        "--to",
        "gfm",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pandoc conversion failed for `{input_path}`: {result.stderr.strip()}"
        )
    return clean_text(result.stdout)


def build_deepseek_ocr2_runtime_config(
    input_path: Path,
    output_path: Path,
    ocr_gpu_memory_utilization: float,
) -> str:
    return textwrap.dedent(
        f"""
        BASE_SIZE = 1024
        IMAGE_SIZE = 768
        CROP_MODE = True
        MIN_CROPS = 2
        MAX_CROPS = 6
        MAX_CONCURRENCY = 100
        NUM_WORKERS = 64
        PRINT_NUM_VIS_TOKENS = False
        SKIP_REPEAT = True
        GPU_MEMORY_UTILIZATION = {ocr_gpu_memory_utilization}
        MODEL_PATH = {json.dumps(str(DEEPSEEK_OCR2_MODEL_PATH))}
        INPUT_PATH = {json.dumps(str(input_path))}
        OUTPUT_PATH = {json.dumps(str(output_path))}
        PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

        from transformers import AutoTokenizer

        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        """
    ).strip() + "\n"


def patch_deepseek_ocr2_script_for_runtime(script_text: str) -> str:
    patched_text = script_text.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"] = \'0\'',
        'os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("DEEPSEEK_OCR2_GPU", "0")',
    )
    patched_text = patched_text.replace(
        "from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE",
        "from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, GPU_MEMORY_UTILIZATION",
    )
    patched_text = patched_text.replace(
        "gpu_memory_utilization=0.9,",
        "gpu_memory_utilization=GPU_MEMORY_UTILIZATION,",
    )
    patched_text = patched_text.replace(
        "gpu_memory_utilization=0.75,",
        "gpu_memory_utilization=GPU_MEMORY_UTILIZATION,",
    )
    return patched_text


def copy_runtime_entry(source: Path, target: Path) -> None:
    if target.exists():
        return
    if source.is_dir():
        shutil.copytree(source, target)
        return
    shutil.copy2(source, target)


def prepare_deepseek_ocr2_runtime(
    runtime_root: Path,
    input_path: Path,
    output_path: Path,
    is_pdf: bool,
    ocr_gpu_memory_utilization: float,
) -> Path:
    runtime_root.mkdir(parents=True, exist_ok=True)
    for entry_name in ("deepencoderv2", "process"):
        source = DEEPSEEK_OCR2_VLLM_ROOT / entry_name
        target = runtime_root / entry_name
        copy_runtime_entry(source, target)

    for file_name in ("deepseek_ocr2.py",):
        source = DEEPSEEK_OCR2_VLLM_ROOT / file_name
        target = runtime_root / file_name
        copy_runtime_entry(source, target)

    script_name = "run_pdf_ocr.py" if is_pdf else "run_dpsk_ocr2_image.py"
    script_source = DEEPSEEK_OCR2_VLLM_ROOT / script_name
    script_target = runtime_root / script_name
    script_target.write_text(
        patch_deepseek_ocr2_script_for_runtime(script_source.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    config_target = runtime_root / "config.py"
    config_target.write_text(
        build_deepseek_ocr2_runtime_config(
            input_path=input_path,
            output_path=output_path,
            ocr_gpu_memory_utilization=ocr_gpu_memory_utilization,
        ),
        encoding="utf-8",
    )
    return script_target


def prepare_deepseek_ocr2_pdf_batch_runtime(
    runtime_root: Path,
    output_path: Path,
    ocr_gpu_memory_utilization: float,
) -> Path:
    runtime_root.mkdir(parents=True, exist_ok=True)
    for entry_name in ("deepencoderv2", "process"):
        source = DEEPSEEK_OCR2_VLLM_ROOT / entry_name
        target = runtime_root / entry_name
        copy_runtime_entry(source, target)

    for file_name in ("deepseek_ocr2.py",):
        source = DEEPSEEK_OCR2_VLLM_ROOT / file_name
        target = runtime_root / file_name
        copy_runtime_entry(source, target)

    script_source = DEEPSEEK_OCR2_VLLM_ROOT / "run_pdf_ocr_multi_gpu.py"
    script_target = runtime_root / "run_pdf_ocr_multi_gpu.py"
    script_target.write_text(
        patch_deepseek_ocr2_script_for_runtime(script_source.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    runner_target = runtime_root / "run_pdf_batch_once.py"
    runner_target.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from config import MODEL_PATH, OUTPUT_PATH, PROMPT
            from run_pdf_ocr_multi_gpu import worker_process

            tasks = json.loads(Path("tasks.json").read_text(encoding="utf-8"))
            pdf_list = [item["runtime_input_path"] for item in tasks]
            worker_process(
                gpu_id=int(os.environ.get("DEEPSEEK_OCR2_GPU", "0")),
                pdf_list=pdf_list,
                output_base_path=OUTPUT_PATH,
                model_path=MODEL_PATH,
                prompt=PROMPT,
            )
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config_target = runtime_root / "config.py"
    config_target.write_text(
        textwrap.dedent(
            f"""
            BASE_SIZE = 1024
            IMAGE_SIZE = 768
            CROP_MODE = True
            MIN_CROPS = 2
            MAX_CROPS = 6
            MAX_CONCURRENCY = 100
            NUM_WORKERS = 64
            PRINT_NUM_VIS_TOKENS = False
            SKIP_REPEAT = True
            GPU_MEMORY_UTILIZATION = {ocr_gpu_memory_utilization}
            MODEL_PATH = {json.dumps(str(DEEPSEEK_OCR2_MODEL_PATH))}
            INPUT_PATH = ""
            OUTPUT_PATH = {json.dumps(str(output_path))}
            PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

            from transformers import AutoTokenizer

            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return runner_target


def read_deepseek_ocr2_markdown(output_dir: Path, input_path: Path, is_pdf: bool) -> str:
    if is_pdf:
        candidate = output_dir / input_path.name.replace("pdf", "mmd")
    else:
        candidate = output_dir / "result.mmd"

    if not candidate.exists():
        raise RuntimeError(f"DeepSeek-OCR2 output Markdown was not generated for `{input_path}`.")

    return clean_text(candidate.read_text(encoding="utf-8", errors="ignore"))


def deepseek_ocr2_to_markdown(
    input_path: Path,
    config_type: str,
    model: Optional[str] = None,
    gpu_id: int = 0,
    ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
) -> str:
    del config_type, model

    if not DEEPSEEK_OCR2_VLLM_ROOT.exists():
        raise FileNotFoundError(f"DeepSeek-OCR2 runtime directory not found: {DEEPSEEK_OCR2_VLLM_ROOT}")
    if not DEEPSEEK_OCR2_MODEL_PATH.exists():
        raise FileNotFoundError(f"DeepSeek-OCR2 model directory not found: {DEEPSEEK_OCR2_MODEL_PATH}")

    is_pdf = input_path.suffix.lower() in PDF_SUFFIXES
    if not is_pdf and input_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"DeepSeek-OCR2 only supports PDF and image files: {input_path}")

    with tempfile.TemporaryDirectory(prefix="deepseek_ocr2_runtime_", dir="/tmp") as runtime_dir_str:
        runtime_dir = Path(runtime_dir_str)
        output_dir = runtime_dir / "ocr_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        script_path = prepare_deepseek_ocr2_runtime(
            runtime_root=runtime_dir,
            input_path=input_path,
            output_path=output_dir,
            is_pdf=is_pdf,
            ocr_gpu_memory_utilization=ocr_gpu_memory_utilization,
        )

        command = [sys.executable, str(script_path)]
        result = subprocess.run(
            command,
            cwd=runtime_dir,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "DEEPSEEK_OCR2_GPU": str(gpu_id)},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DeepSeek-OCR2 failed for `{input_path}` with exit code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        return read_deepseek_ocr2_markdown(output_dir=output_dir, input_path=input_path, is_pdf=is_pdf)


def build_runtime_pdf_name(index: int, input_path: Path) -> str:
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in input_path.stem)
    safe_stem = safe_stem[:80] or "pdf"
    return f"{index:04d}_{safe_stem}.pdf"


def deepseek_ocr2_batch_pdf_to_markdown(
    input_paths: List[Path],
    gpu_id: int,
    ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
) -> Dict[Path, str]:
    if not input_paths:
        return {}
    if not DEEPSEEK_OCR2_VLLM_ROOT.exists():
        raise FileNotFoundError(f"DeepSeek-OCR2 runtime directory not found: {DEEPSEEK_OCR2_VLLM_ROOT}")
    if not DEEPSEEK_OCR2_MODEL_PATH.exists():
        raise FileNotFoundError(f"DeepSeek-OCR2 model directory not found: {DEEPSEEK_OCR2_MODEL_PATH}")

    with tempfile.TemporaryDirectory(prefix=f"deepseek_ocr2_pdf_gpu{gpu_id}_", dir="/tmp") as runtime_dir_str:
        runtime_dir = Path(runtime_dir_str)
        input_dir = runtime_dir / "pdf_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir = runtime_dir / "ocr_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[Dict[str, str]] = []
        for index, input_path in enumerate(input_paths):
            runtime_pdf_name = build_runtime_pdf_name(index, input_path)
            runtime_input_path = input_dir / runtime_pdf_name
            shutil.copy2(input_path, runtime_input_path)
            tasks.append(
                {
                    "source_path": str(input_path),
                    "runtime_input_path": str(runtime_input_path),
                    "runtime_pdf_name": runtime_pdf_name,
                }
            )

        (runtime_dir / "tasks.json").write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        runner_path = prepare_deepseek_ocr2_pdf_batch_runtime(
            runtime_root=runtime_dir,
            output_path=output_dir,
            ocr_gpu_memory_utilization=ocr_gpu_memory_utilization,
        )

        result = subprocess.run(
            [sys.executable, str(runner_path)],
            cwd=runtime_dir,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "DEEPSEEK_OCR2_GPU": str(gpu_id)},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DeepSeek-OCR2 PDF batch failed on GPU {gpu_id} with exit code {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        outputs: Dict[Path, str] = {}
        for task in tasks:
            runtime_pdf_name = task["runtime_pdf_name"]
            runtime_stem = Path(runtime_pdf_name).stem
            candidate = output_dir / runtime_stem / f"{runtime_stem}.mmd"
            if not candidate.exists():
                raise RuntimeError(
                    f"DeepSeek-OCR2 PDF batch did not generate Markdown for `{task['source_path']}`."
                )
            outputs[Path(task["source_path"])] = clean_text(
                candidate.read_text(encoding="utf-8", errors="ignore")
            )
        return outputs


def convert_with_pandoc_tool(input_path: str) -> str:
    return run_pandoc_to_markdown(Path(input_path))


def convert_pdf_with_builtin_tool(input_path: str) -> str:
    return convert_pdf_to_markdown(Path(input_path))


def convert_tabular_tool(input_path: str) -> str:
    return convert_tabular_to_markdown(Path(input_path))


def convert_xml_tool(input_path: str) -> str:
    return convert_xml_to_markdown(Path(input_path))


def convert_html_fallback_tool(input_path: str) -> str:
    return convert_plain_html_to_markdown(Path(input_path))


def deepseek_ocr2_tool(
    input_path: str,
    config_type: str,
    model: Optional[str] = None,
    gpu_id: int = 0,
    ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
) -> str:
    return deepseek_ocr2_to_markdown(
        input_path=Path(input_path),
        config_type=config_type,
        model=model,
        gpu_id=gpu_id,
        ocr_gpu_memory_utilization=ocr_gpu_memory_utilization,
    )


def get_sglang_tools() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "convert_with_pandoc",
                "description": "Convert Word, RTF, ODT, TXT, Markdown, or HTML documents into Markdown using pandoc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the source document.",
                        }
                    },
                    "required": ["input_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_pdf_with_builtin",
                "description": "Extract text from a PDF with a local parser when OCR is not required.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the source PDF file.",
                        }
                    },
                    "required": ["input_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_tabular_document",
                "description": "Convert Excel, CSV, or TSV data into Markdown tables.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the tabular file.",
                        }
                    },
                    "required": ["input_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_xml_document",
                "description": "Convert XML content into Markdown sections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the XML file.",
                        }
                    },
                    "required": ["input_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_html_with_fallback",
                "description": "Convert HTML to Markdown with a local parser fallback when pandoc is not suitable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the HTML file.",
                        }
                    },
                    "required": ["input_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "deepseek_ocr2_to_markdown",
                "description": "Use DeepSeek-OCR2 to convert PDF or image documents into Markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Absolute path to the source PDF or image file.",
                        },
                        "config_type": {
                            "type": "string",
                            "description": "Config section name used by utils.llm_api.call_gpt.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override.",
                        },
                        "gpu_id": {
                            "type": "integer",
                            "description": "GPU index assigned to this OCR task.",
                        },
                        "ocr_gpu_memory_utilization": {
                            "type": "number",
                            "description": "vLLM GPU memory utilization ratio for the OCR worker.",
                        },
                    },
                    "required": ["input_path", "config_type"],
                },
            },
        },
    ]


TOOL_REGISTRY: Dict[str, Callable] = {
    "convert_with_pandoc": convert_with_pandoc_tool,
    "convert_pdf_with_builtin": convert_pdf_with_builtin_tool,
    "convert_tabular_document": convert_tabular_tool,
    "convert_xml_document": convert_xml_tool,
    "convert_html_with_fallback": convert_html_fallback_tool,
    "deepseek_ocr2_to_markdown": deepseek_ocr2_tool,
}


def create_tool_call(function_name: str, arguments: Dict) -> Dict:
    return {
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def execute_tool_call(tool_call: Dict) -> str:
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    if function_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {function_name}")
    return TOOL_REGISTRY[function_name](**arguments)


def choose_conversion_tool_call(
    input_path: Path,
    ocr_backend: str,
    config_type: str,
    model: Optional[str] = None,
    gpu_id: int = 0,
    ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
) -> Tuple[Dict, str]:
    suffix = input_path.suffix.lower()
    input_path_str = str(input_path)

    if suffix in PDF_SUFFIXES:
        return (
            create_tool_call(
                "deepseek_ocr2_to_markdown",
                    {
                        "input_path": input_path_str,
                        "config_type": config_type,
                        "model": model,
                        "gpu_id": gpu_id,
                        "ocr_gpu_memory_utilization": ocr_gpu_memory_utilization,
                    },
                ),
                "deepseek_ocr2_to_markdown",
        )

    if suffix in IMAGE_SUFFIXES:
        return (
            create_tool_call(
                "deepseek_ocr2_to_markdown",
                {
                    "input_path": input_path_str,
                    "config_type": config_type,
                    "model": model,
                    "gpu_id": gpu_id,
                    "ocr_gpu_memory_utilization": ocr_gpu_memory_utilization,
                },
            ),
            "deepseek_ocr2_to_markdown",
        )

    if suffix in TABULAR_SUFFIXES:
        return (
            create_tool_call("convert_tabular_document", {"input_path": input_path_str}),
            "convert_tabular_document",
        )

    if suffix in XML_SUFFIXES:
        return (
            create_tool_call("convert_xml_document", {"input_path": input_path_str}),
            "convert_xml_document",
        )

    if suffix in {".html", ".htm"}:
        return (
            create_tool_call("convert_with_pandoc", {"input_path": input_path_str}),
            "convert_with_pandoc",
        )

    if suffix in PANDOC_SUFFIXES:
        return (
            create_tool_call("convert_with_pandoc", {"input_path": input_path_str}),
            "convert_with_pandoc",
        )

    raise ValueError(f"Unsupported file suffix: `{suffix}`")


def dataframe_to_markdown(dataframe: pd.DataFrame, title: str) -> str:
    dataframe = dataframe.fillna("")
    if dataframe.empty:
        return f"## {title}\n\n(No rows found)"

    table_markdown = dataframe.to_markdown(index=False)
    return f"## {title}\n\n{table_markdown}"


def convert_tabular_to_markdown(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    sections: List[str] = []

    if suffix in {".csv", ".tsv"}:
        separator = "," if suffix == ".csv" else "\t"
        dataframe = pd.read_csv(input_path, sep=separator)
        sections.append(dataframe_to_markdown(dataframe, input_path.stem))
    else:
        excel_file = pd.ExcelFile(input_path)
        for sheet_name in excel_file.sheet_names:
            dataframe = pd.read_excel(input_path, sheet_name=sheet_name)
            sections.append(dataframe_to_markdown(dataframe, sheet_name))

    return clean_text("\n\n".join(sections))


def convert_xml_to_markdown(input_path: Path) -> str:
    content = input_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "xml")
    lines: List[str] = [f"# {input_path.stem}", ""]

    for element in soup.find_all(recursive=True):
        if not getattr(element, "name", None):
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        lines.append(f"## {element.name}")
        lines.append("")
        lines.append(text)
        lines.append("")

    return clean_text("\n".join(lines))


def convert_plain_html_to_markdown(input_path: Path) -> str:
    content = input_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else input_path.stem
    body_text = soup.get_text("\n", strip=True)
    markdown = f"# {title}\n\n{body_text}"
    return clean_text(markdown)


def convert_document_to_markdown(
    input_path: Path,
    ocr_backend: str,
    config_type: str,
    model: Optional[str] = None,
    gpu_id: int = 0,
    ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
) -> Tuple[str, str]:
    tool_call, tool_name = choose_conversion_tool_call(
        input_path=input_path,
        ocr_backend=ocr_backend,
        config_type=config_type,
        model=model,
        gpu_id=gpu_id,
        ocr_gpu_memory_utilization=ocr_gpu_memory_utilization,
    )

    try:
        markdown_text = execute_tool_call(tool_call)
    except RuntimeError:
        if input_path.suffix.lower() in {".html", ".htm"} and tool_name == "convert_with_pandoc":
            fallback_tool_call = create_tool_call(
                "convert_html_with_fallback",
                {"input_path": str(input_path)},
            )
            markdown_text = execute_tool_call(fallback_tool_call)
            return markdown_text, "convert_html_with_fallback"
        raise

    return markdown_text, tool_name


def build_buffer_item_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode("utf-8")).hexdigest()


def enqueue_stage2_buffer_item(buffer_dir: Path, metadata_payload: Dict, metadata_path: Path) -> Path:
    new_dir = buffer_dir / "new"
    ensure_parent_dir(new_dir / "placeholder")
    item_id = build_buffer_item_id(str(metadata_payload.get("source_path", "") or metadata_path))
    payload = {
        "item_id": item_id,
        "source_path": str(metadata_payload.get("source_path", "") or ""),
        "markdown_path": str(metadata_payload.get("markdown_path", "") or ""),
        "metadata_path": str(metadata_path),
        "type": str(metadata_payload.get("Type", "") or ""),
    }
    temp_path = new_dir / f"{item_id}.json.tmp"
    final_path = new_dir / f"{item_id}.json"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(final_path)
    return final_path


def build_metadata_prompt(source_path: Path, markdown_text: str) -> str:
    truncated_markdown = truncate_to_token_limit(markdown_text, max_tokens=4096)
    return f"""
You are a document metadata extraction agent.

Task:
1. Read the source path and the Markdown content.
2. Extract the document's basic information.
3. Return a JSON object only.

Rules:
- If a field is missing, return an empty string.
- `Brief Description` should be 2-3 short sentences.
- `Type` must be one of:
  - "MCQ": directly reusable multiple-choice question content with clear options and answers.
  - "SAQ": book-like short-answer-question material where questions and answers may be organized in complex sections.
  - "CaseReport": papers, reports, articles, or documents that require further analysis and extraction.
- Use both the source path and the document content for inference.

Required JSON schema:
{{
  "Name": "",
  "Date": "",
  "Authors": "",
  "Brief Description": "",
  "Type": ""
}}

Source path:
{source_path}

Markdown content:
{truncated_markdown}
""".strip()


def infer_fallback_document_type(markdown_text: str) -> str:
    sample = markdown_text[:12000].lower()
    if "case report" in sample or "case presentation" in sample:
        return CASE_REPORT_TYPE
    if "multiple choice" in sample or "mcq" in sample:
        return MCQ_TYPE
    if "short answer" in sample or "saq" in sample:
        return SAQ_TYPE
    return ""


def build_fallback_metadata(source_path: Path, markdown_text: str, error_message: str) -> Dict[str, str]:
    return {
        "Name": source_path.stem,
        "Date": "",
        "Authors": "",
        "Brief Description": "Automatic metadata extraction failed. This fallback metadata was generated from the source filename so downstream processing can continue.",
        "Type": infer_fallback_document_type(markdown_text),
        "Metadata Warning": error_message[:1000],
    }


def generate_document_metadata(
    source_path: Path,
    markdown_text: str,
    config_type: str,
    model: str = None,
) -> Dict[str, str]:
    prompt = build_metadata_prompt(source_path, markdown_text)
    response = call_gpt(
        prompt=prompt,
        model=model,
        config_type=config_type,
        json_output=True,
        system_prompt="Return valid JSON only.",
        timeout=300,
    )

    if not isinstance(response, dict):
        error_message = f"Metadata extraction did not return JSON: {response}"
        log_message(f"{error_message} for `{source_path}`. Using fallback metadata.")
        return build_fallback_metadata(source_path, markdown_text, error_message)

    if response.get("error"):
        error_message = f"Metadata extraction returned invalid JSON: {response}"
        log_message(f"{error_message} for `{source_path}`. Using fallback metadata.")
        return build_fallback_metadata(source_path, markdown_text, error_message)

    return {
        "Name": str(response.get("Name", "") or ""),
        "Date": str(response.get("Date", "") or ""),
        "Authors": str(response.get("Authors", "") or ""),
        "Brief Description": str(response.get("Brief Description", "") or ""),
        "Type": normalize_document_type(str(response.get("Type", "") or "")),
    }


def build_output_paths(input_path: Path, input_root: Path, output_root: Path) -> Tuple[Path, Path]:
    relative_path = input_path.relative_to(input_root) if input_path.is_relative_to(input_root) else Path(input_path.name)
    markdown_path = output_root / relative_path.with_suffix(".mmd")
    metadata_path = output_root / relative_path.with_suffix(".json")
    return markdown_path, metadata_path


def write_text_file(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def write_json_file(path: Path, content: Dict) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_input_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def is_pdf_input(path: Path) -> bool:
    return path.suffix.lower() in PDF_SUFFIXES


def is_ocr_input(path: Path) -> bool:
    return path.suffix.lower() in PDF_SUFFIXES or path.suffix.lower() in IMAGE_SUFFIXES


class SGLangStyleMarkdownAgent:
    def __init__(
        self,
        output_root: Path,
        config_type: str,
        model: str = None,
        ocr_backend: str = "builtin",
        buffer_dir: Optional[Path] = None,
        gpu_count: int = 1,
        ocr_gpu_memory_utilization: float = DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
        progress_state_path: Optional[Path] = None,
    ):
        self.output_root = output_root
        self.config_type = config_type
        self.model = model
        self.ocr_backend = ocr_backend
        self.buffer_dir = buffer_dir
        self.gpu_count = max(1, gpu_count)
        self.ocr_gpu_memory_utilization = ocr_gpu_memory_utilization
        self.progress_state_path = progress_state_path

    def step_convert_to_markdown(self, input_path: Path, gpu_id: int = 0) -> Tuple[str, str]:
        return convert_document_to_markdown(
            input_path=input_path,
            ocr_backend=self.ocr_backend,
            config_type=self.config_type,
            model=self.model,
            gpu_id=gpu_id,
            ocr_gpu_memory_utilization=self.ocr_gpu_memory_utilization,
        )

    def step_extract_metadata(self, input_path: Path, markdown_text: str) -> Dict[str, str]:
        return generate_document_metadata(
            source_path=input_path,
            markdown_text=markdown_text,
            config_type=self.config_type,
            model=self.model,
        )

    def finalize_file(
        self,
        input_path: Path,
        input_root: Path,
        markdown_text: str,
        tool_name: str,
    ) -> Dict:
        markdown_path, metadata_path = build_output_paths(
            input_path=input_path,
            input_root=input_root,
            output_root=self.output_root,
        )
        write_text_file(markdown_path, markdown_text)

        metadata = self.step_extract_metadata(input_path, markdown_text)
        metadata_payload = {
            "source_path": str(input_path),
            "markdown_path": str(markdown_path),
            "suffix": input_path.suffix.lower(),
            "conversion_tool": tool_name,
            "markdown_token_estimate": estimate_token_count(markdown_text),
            **metadata,
        }
        write_json_file(metadata_path, metadata_payload)
        if self.buffer_dir is not None:
            enqueue_stage2_buffer_item(
                buffer_dir=self.buffer_dir,
                metadata_payload=metadata_payload,
                metadata_path=metadata_path,
            )
        return metadata_payload

    def run_file(self, input_path: Path, input_root: Path, gpu_id: int = 0) -> Dict:
        markdown_text, tool_name = self.step_convert_to_markdown(input_path, gpu_id=gpu_id)
        return self.finalize_file(
            input_path=input_path,
            input_root=input_root,
            markdown_text=markdown_text,
            tool_name=tool_name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert heterogeneous documents to Markdown and generate document metadata JSON."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="A file path or a directory path to process.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_CONSTRUCTION_PIPELINE_STAGE1_OUTPUT_ROOT,
        help="Root directory for generated .mmd and .json files.",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default="llm",
        help="The config section in config/config.json used by utils.llm_api.call_gpt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process supported files under the input directory.",
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default="deepseek_ocr2",
        choices=["deepseek_ocr2"],
        help="OCR backend for PDF and image documents.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Parallel worker count for stage 01 file processing.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPU slots available for OCR tasks. OCR files are assigned round-robin.",
    )
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default=None,
        help="Optional buffer directory used to enqueue stage-02 tasks as soon as each markdown file is ready.",
    )
    parser.add_argument(
        "--ocr-gpu-memory-utilization",
        type=float,
        default=DEFAULT_OCR_GPU_MEMORY_UTILIZATION,
        help="vLLM GPU memory utilization ratio for DeepSeek-OCR2 workers.",
    )
    parser.add_argument(
        "--progress-state",
        type=str,
        default=None,
        help="Optional shared JSON file used to track pipeline progress.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_root = Path(args.output_root).resolve()
    buffer_dir = Path(args.buffer_dir).resolve() if args.buffer_dir else None
    progress_state_path = Path(args.progress_state).resolve() if args.progress_state else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_root = input_path.parent if input_path.is_file() else input_path
    input_files = collect_input_files(input_path, recursive=args.recursive)
    if not input_files:
        raise ValueError(f"No supported files found under: {input_path}")

    pdf_input_files = [path for path in input_files if is_pdf_input(path)]
    remaining_input_files = [path for path in input_files if not is_pdf_input(path)]

    agent = SGLangStyleMarkdownAgent(
        output_root=output_root,
        config_type=args.config_type,
        model=args.model,
        ocr_backend=args.ocr_backend,
        buffer_dir=buffer_dir,
        gpu_count=args.gpu_count,
        ocr_gpu_memory_utilization=args.ocr_gpu_memory_utilization,
        progress_state_path=progress_state_path,
    )

    results = []
    failed_files = []
    if pdf_input_files:
        pdf_batches: List[List[Path]] = [[] for _ in range(max(1, args.gpu_count))]
        for index, file_path in enumerate(pdf_input_files):
            pdf_batches[index % max(1, args.gpu_count)].append(file_path)

        pdf_markdown_cache: Dict[Path, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(args.gpu_count, len(pdf_input_files)))) as executor:
            future_to_batch = {}
            for gpu_id, batch_paths in enumerate(pdf_batches):
                if not batch_paths:
                    continue
                future = executor.submit(
                    deepseek_ocr2_batch_pdf_to_markdown,
                    batch_paths,
                    gpu_id,
                    args.ocr_gpu_memory_utilization,
                )
                future_to_batch[future] = (gpu_id, batch_paths)

            for future in as_completed(future_to_batch):
                gpu_id, batch_paths = future_to_batch[future]
                try:
                    batch_outputs = future.result()
                    pdf_markdown_cache.update(batch_outputs)
                    log_message(f"GPU {gpu_id} PDF batch completed: {len(batch_outputs)} files")
                except Exception as exc:
                    log_message(f"GPU {gpu_id} PDF batch failed: {exc}")
                    for file_path in batch_paths:
                        failed_files.append({"source_path": str(file_path), "error": str(exc)})
                        update_progress_state(progress_state_path, "stage1_failed")

        with ThreadPoolExecutor(max_workers=max(1, args.worker_count)) as executor:
            future_to_file = {}
            for file_path in pdf_input_files:
                markdown_text = pdf_markdown_cache.get(file_path)
                if markdown_text is None:
                    continue
                future = executor.submit(
                    agent.finalize_file,
                    file_path,
                    input_root,
                    markdown_text,
                    "deepseek_ocr2_batch_pdf",
                )
                future_to_file[future] = file_path

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                log_message(f"Processing: {file_path}")
                try:
                    result = future.result()
                    results.append(result)
                    update_progress_state(progress_state_path, "stage1_completed")
                    log_message(f"Saved Markdown: {result['markdown_path']}")
                except Exception as exc:
                    failed_files.append({"source_path": str(file_path), "error": str(exc)})
                    update_progress_state(progress_state_path, "stage1_failed")
                    log_message(f"Failed Markdown conversion: {file_path}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, args.worker_count)) as executor:
        future_to_file = {}
        for index, file_path in enumerate(remaining_input_files):
            gpu_id = index % max(1, args.gpu_count)
            future = executor.submit(agent.run_file, file_path, input_root, gpu_id)
            future_to_file[future] = file_path

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            log_message(f"Processing: {file_path}")
            try:
                result = future.result()
                results.append(result)
                update_progress_state(progress_state_path, "stage1_completed")
                log_message(f"Saved Markdown: {result['markdown_path']}")
            except Exception as exc:
                failed_files.append({"source_path": str(file_path), "error": str(exc)})
                update_progress_state(progress_state_path, "stage1_failed")
                log_message(f"Failed Markdown conversion: {file_path}: {exc}")

    summary_path = output_root / "run_summary.json"
    write_json_file(summary_path, {"processed_files": results, "failed_files": failed_files})
    log_message(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


# Run from the project root:
# python construction_pipeline/01-ToMarkdown.py --input-path 00-Data_and_Models/your_input_file.pdf
# python construction_pipeline/01-ToMarkdown.py --input-path 00-Data_and_Models/your_input_dir --recursive --output-root 00-Data_and_Models/construction_pipeline/01_markdown_and_metadata
