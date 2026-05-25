# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Use `utils.llm_api.call_gpt` to analyze document length and generate, validate, and label dental QA JSON files from Markdown and metadata. Supports manually specified MCQ, SAQ, and Case Report document types, stores staged outputs under `00-Data_and_Models/`, and remains compatible with runtimes that only provide an `llm` configuration.

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tokenizers import Tokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_model_config, load_config
from utils.llm_api import call_gpt


LEGACY_EXAM_TYPE = "Exam"
MCQ_TYPE = "MCQ"
SAQ_TYPE = "SAQ"
CASE_REPORT_TYPE = "CaseReport"
DEFAULT_AGENT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_AGENT_MAX_READ_TOKENS = 32768
DEFAULT_CHUNK_RATIO = 0.9
DEFAULT_CHUNK_OVERLAP_TOKENS = 96
DEFAULT_CASE_OVERLAP_TOKENS = 160
DEFAULT_CONSTRUCTION_PIPELINE_STAGE1_OUTPUT_ROOT = Path("00-Data_and_Models/construction_pipeline/01_markdown_and_metadata").resolve()
DEFAULT_CONSTRUCTION_PIPELINE_STAGE2_OUTPUT_ROOT = "00-Data_and_Models/construction_pipeline/02_qa_outputs"
DEFAULT_STAGE2_BUFFER_DIR = "00-Data_and_Models/construction_pipeline/buffer"
DEFAULT_TOKEN_COUNT_MODE = "auto"
DEFAULT_API_COUNT_MAX_OUTPUT_TOKENS = 1
DEFAULT_FALLBACK_CHARS_PER_TOKEN = 4
DEFAULT_BUFFER_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_BUFFER_IDLE_EXIT_SECONDS = 15.0
CASE_KEY_POINT_COUNT = 5
MAX_CASE_REPORT_REFINEMENT_ROUNDS = 3
TAG_TAXONOMY_MAP = {
    1: "Caries, Tooth Defects & Trauma",
    2: "Pulp & Periapical Diseases",
    3: "Periodontal & Peri-implant Diseases",
    4: "Oral Mucosal Diseases",
    5: "Dentoalveolar Surgery",
    6: "Maxillofacial Diseases & Surgery",
    7: "Anesthesia & Medical Emergencies",
    8: "Oral & Maxillofacial Radiology",
    9: "Conventional Prosthodontics",
    10: "Oral Implantology",
    11: "Pediatric Dentistry",
    12: "Orthodontics",
    13: "Basic Sciences & Preventive Dentistry",
    14: "Systemic Health, Pharmacology & Safety",
}

CONSTRUCTION_PROMPT_PATH = PROJECT_ROOT / "data" / "construction_pipeline" / "prompt.yaml"


def _load_construction_prompts(prompt_path: Path = CONSTRUCTION_PROMPT_PATH) -> Dict:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Construction prompt YAML not found: {prompt_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read the construction prompt YAML. Install it with `pip install PyYAML`."
        ) from exc
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Construction prompt YAML must be a top-level mapping: {prompt_path}")
    return payload


CONSTRUCTION_PROMPTS = _load_construction_prompts()

CASE_REPORT_TAGGING_SYSTEM_PROMPT = CONSTRUCTION_PROMPTS["tag"]["question_tagging_for_caserepo"]["system_prompt"]
CASE_REPORT_TAGGING_USER_TEMPLATE = CONSTRUCTION_PROMPTS["tag"]["question_tagging_for_caserepo"]["user_template"]

QUESTION_TAGGING_SYSTEM_PROMPT = CONSTRUCTION_PROMPTS["tag"]["question_tagging_for_saq_or_mcq"]["system_prompt"]
QUESTION_TAGGING_USER_TEMPLATE = CONSTRUCTION_PROMPTS["tag"]["question_tagging_for_saq_or_mcq"]["user_template"]

CASE_REPORT_SYSTEM_PROMPT = CONSTRUCTION_PROMPTS["casequestion_extract_seed_qa_prompts"]["unified_extraction"]["system_prompt"]
CASE_REPORT_EXTRACTION_USER_TEMPLATE = CONSTRUCTION_PROMPTS["casequestion_extract_seed_qa_prompts"]["unified_extraction"]["user_template"]

CASE_REPORT_VALIDATION_SYSTEM_PROMPT = CONSTRUCTION_PROMPTS["casequestion_validate_qa_prompts"]["benchmark_validation"]["system_prompt"]
CASE_REPORT_VALIDATION_USER_TEMPLATE = CONSTRUCTION_PROMPTS["casequestion_validate_qa_prompts"]["benchmark_validation"]["user_template"]

SAQ_STRUCTURE_SYSTEM_PROMPT = (
    "You are a senior benchmark-construction expert for dental short-answer books. "
    "Your task is to inspect one Markdown chunk and summarize its question-answer organization faithfully. "
    "Focus on locating sections where complete questions and their answers likely coexist or can be paired locally within the same nearby span. "
    "Return valid JSON only."
)

SAQ_STRUCTURE_PROMPT = """
Analyze the following Markdown chunk from a dental SAQ book and summarize its structure.

OBJECTIVES:
1. Identify the major sections or subsection patterns in this chunk.
2. Judge whether each section mainly contains questions, mainly contains answers, or mixes both.
3. Highlight which section titles or local spans are the best extraction targets because they likely contain enough context to recover both question and answer together.

RULES:
- Be faithful to the source chunk only.
- Do not infer content outside this chunk.
- Prefer concise section titles copied from the chunk when possible.
- If a section has no explicit title, create a short descriptive placeholder.
- `recommended_for_extraction` must be 1 only when this section is a strong candidate for extracting complete SAQ pairs from a local span.

Return JSON only in this schema:
{{
  "chunk_summary": "",
  "global_pairing_hint": "",
  "recommended_section_titles": [],
  "sections": [
    {{
      "section_title": "",
      "section_type": "question_bank|answer_key|mixed_qa|explanation|other",
      "question_signal": "",
      "answer_signal": "",
      "pairing_hint": "",
      "recommended_for_extraction": 0
    }}
  ]
}}

Document metadata:
{metadata}

Chunk index:
{chunk_index} / {total_chunks}

Chunk content:
{chunk_text}
"""

SAQ_GENERATION_SYSTEM_PROMPT = (
    "You are extracting benchmark-ready dental short-answer questions from a book-like SAQ source. "
    "You must only emit complete question-answer pairs supported by the provided chunk and structure guidance. "
    "Return valid JSON only."
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json_file(path: Path, payload: Dict) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_document_type(document_type: Optional[str]) -> str:
    normalized = str(document_type or "").strip().lower()
    if normalized in {"exam", "mcq"}:
        return MCQ_TYPE
    if normalized == "saq":
        return SAQ_TYPE
    if normalized in {"case report", "casereport", "case_report"}:
        return CASE_REPORT_TYPE
    return str(document_type or "").strip()


def normalize_heading_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.strip().lower())


def split_markdown_sections(text: str) -> List[Dict]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
    sections: List[Dict] = []
    current_title = "Preamble"
    current_level = 0
    current_lines: List[str] = []

    def flush_section() -> None:
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        sections.append(
            {
                "title": current_title,
                "level": current_level,
                "text": section_text,
            }
        )

    for line in text.splitlines():
        match = heading_pattern.match(line)
        if match:
            flush_section()
            current_title = match.group(2).strip()
            current_level = len(match.group(1))
            current_lines = [line]
            continue
        current_lines.append(line)

    flush_section()
    if not sections:
        sections.append({"title": "Full Document", "level": 0, "text": text.strip()})
    return sections


def normalize_saq_structure_result(response) -> Dict:
    normalized = {
        "chunk_summary": "",
        "global_pairing_hint": "",
        "recommended_section_titles": [],
        "sections": [],
    }
    if not isinstance(response, dict):
        return normalized

    normalized["chunk_summary"] = str(response.get("chunk_summary", "") or "")
    normalized["global_pairing_hint"] = str(response.get("global_pairing_hint", "") or "")

    recommended_titles = response.get("recommended_section_titles", [])
    if isinstance(recommended_titles, list):
        normalized["recommended_section_titles"] = [
            str(title).strip() for title in recommended_titles if str(title).strip()
        ]

    sections = response.get("sections", [])
    if not isinstance(sections, list):
        return normalized

    for section in sections:
        if not isinstance(section, dict):
            continue
        normalized["sections"].append(
            {
                "section_title": str(section.get("section_title", "") or "").strip(),
                "section_type": str(section.get("section_type", "other") or "other").strip(),
                "question_signal": str(section.get("question_signal", "") or "").strip(),
                "answer_signal": str(section.get("answer_signal", "") or "").strip(),
                "pairing_hint": str(section.get("pairing_hint", "") or "").strip(),
                "recommended_for_extraction": 1 if section.get("recommended_for_extraction") == 1 else 0,
            }
        )
    return normalized


def build_saq_structure_context(structure_reports: List[Dict], max_reports: int = 8) -> str:
    compact_reports = []
    for report in structure_reports[:max_reports]:
        structure = report.get("structure", {})
        compact_reports.append(
            {
                "chunk_index": report.get("chunk_index", 0),
                "recommended_section_titles": structure.get("recommended_section_titles", []),
                "chunk_summary": structure.get("chunk_summary", ""),
                "global_pairing_hint": structure.get("global_pairing_hint", ""),
                "sections": [
                    {
                        "section_title": section.get("section_title", ""),
                        "section_type": section.get("section_type", "other"),
                        "recommended_for_extraction": section.get("recommended_for_extraction", 0),
                    }
                    for section in structure.get("sections", [])[:8]
                ],
            }
        )
    return json.dumps(compact_reports, ensure_ascii=False, indent=2)


def select_saq_extraction_segments(markdown_text: str, structure_reports: List[Dict]) -> List[Dict]:
    sections = split_markdown_sections(markdown_text)
    recommended_titles = []
    for report in structure_reports:
        structure = report.get("structure", {})
        recommended_titles.extend(structure.get("recommended_section_titles", []))
        for section in structure.get("sections", []):
            if section.get("recommended_for_extraction") == 1 and section.get("section_title"):
                recommended_titles.append(section["section_title"])

    normalized_titles = [normalize_heading_text(title) for title in recommended_titles if normalize_heading_text(title)]
    matched_sections = []
    seen_titles = set()
    if normalized_titles:
        for section in sections:
            normalized_section_title = normalize_heading_text(section.get("title", ""))
            if not normalized_section_title:
                continue
            if any(
                normalized_title == normalized_section_title
                or normalized_title in normalized_section_title
                or normalized_section_title in normalized_title
                for normalized_title in normalized_titles
            ):
                if normalized_section_title in seen_titles:
                    continue
                seen_titles.add(normalized_section_title)
                matched_sections.append(
                    {
                        "segment_type": "section",
                        "title": section.get("title", ""),
                        "text": section.get("text", ""),
                    }
                )

    if matched_sections:
        return matched_sections

    fallback_segments = []
    for report in structure_reports:
        structure = report.get("structure", {})
        has_recommended_section = any(
            section.get("recommended_for_extraction") == 1 for section in structure.get("sections", [])
        )
        if has_recommended_section or structure.get("recommended_section_titles"):
            fallback_segments.append(
                {
                    "segment_type": "read_chunk",
                    "title": f"Structure chunk {report.get('chunk_index', 0) + 1}",
                    "text": report.get("text", ""),
                }
            )

    if fallback_segments:
        return fallback_segments

    return [{"segment_type": "full_document", "title": "Full Document", "text": markdown_text}]


def load_qa_generation_config() -> Dict:
    config = load_config()
    return config.get("qa_generation", {})


def resolve_runtime_setting(cli_value, config_value, fallback_value):
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback_value


def load_tokenizer(tokenizer_source: Optional[str], config_type: str) -> Tokenizer:
    candidate = tokenizer_source
    if not candidate:
        config = get_model_config(config_type)
        candidate = config.get("tokenizer") or config.get("model")

    if not candidate:
        raise ValueError(
            "Tokenizer source is missing. Please provide `--tokenizer-source` or add `tokenizer` to the config section."
        )

    tokenizer_path = Path(candidate)
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    try:
        return Tokenizer.from_pretrained(candidate)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer from `{candidate}`. "
            "Provide a local tokenizer.json path with `--tokenizer-source` if needed."
        ) from exc


def split_text_into_units(text: str) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text]


def split_oversized_unit(unit_text: str, chunk_size_tokens: int) -> List[str]:
    estimated_char_limit = max(1, chunk_size_tokens * DEFAULT_FALLBACK_CHARS_PER_TOKEN)
    if len(unit_text) <= estimated_char_limit:
        return [unit_text]

    lines = [line.strip() for line in unit_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", unit_text)
        if sentence.strip()
    ]
    if len(sentences) > 1:
        return sentences

    words = unit_text.split()
    if not words:
        return [unit_text]

    word_chunk_size = max(1, estimated_char_limit // max(1, sum(len(word) for word in words) // len(words)))
    return [" ".join(words[index : index + word_chunk_size]) for index in range(0, len(words), word_chunk_size)]


def count_tokens_with_api(
    text: str,
    config_type: str,
    model: Optional[str],
    cache: Dict[str, int],
) -> int:
    if text in cache:
        return cache[text]

    _, usage = call_gpt(
        prompt=text,
        config_type=config_type,
        model=model,
        json_output=False,
        return_usage=True,
        max_tokens=DEFAULT_API_COUNT_MAX_OUTPUT_TOKENS,
        temperature=0,
    )
    token_count = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("total_tokens")
        or 0
    )
    if token_count <= 0 and text:
        token_count = max(1, len(text) // DEFAULT_FALLBACK_CHARS_PER_TOKEN)
    cache[text] = token_count
    return token_count


def count_tokens(
    text: str,
    tokenizer: Optional[Tokenizer],
    token_count_mode: str,
    config_type: str,
    model: Optional[str],
    api_token_count_cache: Dict[str, int],
) -> int:
    if token_count_mode == "local":
        if tokenizer is None:
            raise ValueError("`tokenizer` is required when token_count_mode is `local`.")
        return len(tokenizer.encode(text).ids)

    if token_count_mode == "api":
        return count_tokens_with_api(
            text=text,
            config_type=config_type,
            model=model,
            cache=api_token_count_cache,
        )

    raise ValueError(f"Unsupported token_count_mode: `{token_count_mode}`")


def resolve_token_count_mode(
    token_count_mode: str,
    tokenizer_source: Optional[str],
    config_type: str,
) -> str:
    if token_count_mode in {"local", "api"}:
        return token_count_mode

    config = get_model_config(config_type)
    candidate = tokenizer_source or config.get("tokenizer")
    return "local" if candidate else "api"


def create_token_chunks(
    text: str,
    tokenizer: Optional[Tokenizer],
    chunk_size_tokens: int,
    overlap_tokens: int,
    token_count_mode: str,
    config_type: str,
    model: Optional[str],
    api_token_count_cache: Dict[str, int],
) -> List[Dict]:
    if token_count_mode == "local":
        if tokenizer is None:
            raise ValueError("`tokenizer` is required when token_count_mode is `local`.")

        encoding = tokenizer.encode(text)
        token_count = len(encoding.ids)
        if token_count == 0:
            return [{"chunk_index": 0, "token_start": 0, "token_end": 0, "text": ""}]

        if token_count <= chunk_size_tokens:
            return [
                {
                    "chunk_index": 0,
                    "token_start": 0,
                    "token_end": token_count,
                    "text": text,
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0
        offsets = encoding.offsets
        step = max(1, chunk_size_tokens - overlap_tokens)

        while start < token_count:
            end = min(token_count, start + chunk_size_tokens)
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1] if end > start else len(text)
            chunk_text = text[start_char:end_char].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "token_start": start,
                        "token_end": end,
                        "text": chunk_text,
                    }
                )

            if end >= token_count:
                break

            start += step
            chunk_index += 1

        return chunks

    if token_count_mode != "api":
        raise ValueError(f"Unsupported token_count_mode: `{token_count_mode}`")

    total_token_count = count_tokens_with_api(
        text=text,
        config_type=config_type,
        model=model,
        cache=api_token_count_cache,
    )
    if total_token_count == 0 and not text:
        return [{"chunk_index": 0, "token_start": 0, "token_end": 0, "text": ""}]

    if total_token_count <= chunk_size_tokens:
        return [
            {
                "chunk_index": 0,
                "token_start": 0,
                "token_end": total_token_count,
                "text": text,
            }
        ]

    base_units = split_text_into_units(text)
    expanded_units: List[str] = []
    for unit in base_units:
        expanded_units.extend(split_oversized_unit(unit, chunk_size_tokens))

    chunks: List[Dict] = []
    current_units: List[str] = []
    current_token_count = 0
    chunk_index = 0
    token_start = 0

    for unit in expanded_units:
        unit_token_count = count_tokens_with_api(
            text=unit,
            config_type=config_type,
            model=model,
            cache=api_token_count_cache,
        )
        candidate_units = current_units + [unit]
        candidate_text = "\n\n".join(candidate_units).strip()
        candidate_token_count = count_tokens_with_api(
            text=candidate_text,
            config_type=config_type,
            model=model,
            cache=api_token_count_cache,
        )

        if current_units and candidate_token_count > chunk_size_tokens:
            chunk_text = "\n\n".join(current_units).strip()
            token_end = token_start + current_token_count
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": chunk_text,
                }
            )
            token_start = max(0, token_end - overlap_tokens)
            chunk_index += 1

            overlap_units: List[str] = []
            overlap_token_count = 0
            for previous_unit in reversed(current_units):
                previous_text = previous_unit.strip()
                if not previous_text:
                    continue
                previous_unit_token_count = count_tokens_with_api(
                    text=previous_text,
                    config_type=config_type,
                    model=model,
                    cache=api_token_count_cache,
                )
                if overlap_token_count + previous_unit_token_count > overlap_tokens and overlap_units:
                    break
                overlap_units.insert(0, previous_text)
                overlap_token_count += previous_unit_token_count
                if overlap_token_count >= overlap_tokens:
                    break

            current_units = overlap_units + [unit]
            current_text = "\n\n".join(current_units).strip()
            current_token_count = count_tokens_with_api(
                text=current_text,
                config_type=config_type,
                model=model,
                cache=api_token_count_cache,
            )
            continue

        current_units = candidate_units
        current_token_count = candidate_token_count if candidate_text else unit_token_count

    if current_units:
        chunks.append(
            {
                "chunk_index": chunk_index,
                "token_start": token_start,
                "token_end": token_start + current_token_count,
                "text": "\n\n".join(current_units).strip(),
            }
        )

    return chunks


def load_metadata_records(metadata_input: Path) -> List[Dict]:
    if metadata_input.is_file():
        payload = json.loads(read_text_file(metadata_input))
        if isinstance(payload, dict) and "processed_files" in payload:
            return payload["processed_files"]
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported metadata file format: {metadata_input}")

    records = []
    for json_path in sorted(metadata_input.rglob("*.json")):
        if json_path.name == "run_summary.json":
            continue
        records.append(json.loads(read_text_file(json_path)))
    return records


def load_existing_summary(summary_path: Path) -> Dict[str, List[Dict]]:
    if not summary_path.exists():
        return {}

    try:
        payload = json.loads(read_text_file(summary_path))
    except json.JSONDecodeError:
        return {}
    return load_existing_summary_from_payload(payload)


def load_resume_index(index_path: Path) -> Dict[str, Dict]:
    if not index_path.exists():
        return {}

    try:
        payload = json.loads(read_text_file(index_path))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    records = payload.get("records", payload)
    if not isinstance(records, dict):
        return {}

    return {
        str(source_path): record
        for source_path, record in records.items()
        if isinstance(source_path, str) and isinstance(record, dict)
    }


def write_resume_index(index_path: Path, records: Dict[str, Dict]) -> None:
    write_json_file(index_path, {"records": records})


def build_resume_record(
    metadata: Dict,
    output_path: Path,
    question_type: str,
    status: str,
    process_output_path: Optional[Path] = None,
) -> Dict:
    return {
        "source_path": str(metadata.get("source_path", "") or ""),
        "markdown_path": str(metadata.get("markdown_path", "") or ""),
        "question_type": question_type,
        "output_path": str(output_path),
        "process_output_path": str(process_output_path) if process_output_path else "",
        "status": status,
    }


def load_existing_result_for_summary(output_path: Path) -> Dict:
    payload = json.loads(read_text_file(output_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported QA result format: {output_path}")
    payload["output_path"] = str(output_path)
    return payload


def ensure_buffer_dirs(buffer_dir: Path) -> Dict[str, Path]:
    dirs = {
        "new": buffer_dir / "new",
        "processing": buffer_dir / "processing",
        "done": buffer_dir / "done",
        "failed": buffer_dir / "failed",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_buffer_item(item_path: Path) -> Dict:
    payload = json.loads(read_text_file(item_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid buffer item: {item_path}")
    return payload


def claim_next_buffer_item(buffer_dir: Path) -> Optional[Tuple[Path, Path, Dict]]:
    dirs = ensure_buffer_dirs(buffer_dir)
    for item_path in sorted(dirs["new"].glob("*.json")):
        processing_name = f"{item_path.stem}.{uuid.uuid4().hex}.json"
        processing_path = dirs["processing"] / processing_name
        try:
            item_path.replace(processing_path)
        except FileNotFoundError:
            continue
        payload = load_buffer_item(processing_path)
        return processing_path, item_path, payload
    return None


def finalize_buffer_item(processing_path: Path, final_dir: Path) -> Path:
    final_path = final_dir / processing_path.name
    processing_path.replace(final_path)
    return final_path


def merge_results_into_summary(existing_summary: Dict[str, List[Dict]], new_results: List[Dict]) -> Dict[str, List[Dict]]:
    merged_summary: Dict[str, List[Dict]] = {
        category: list(items) for category, items in existing_summary.items()
    }

    for result in new_results:
        question_type = str(result.get("question_type", "") or "Unknown")
        merged_summary.setdefault(question_type, [])

        result_identifier = str(result.get("output_path", "") or result.get("source_path", "") or "")
        replaced = False
        if result_identifier:
            for index, existing_item in enumerate(merged_summary[question_type]):
                existing_identifier = str(
                    existing_item.get("output_path", "") or existing_item.get("source_path", "") or ""
                )
                if existing_identifier and existing_identifier == result_identifier:
                    merged_summary[question_type][index] = result
                    replaced = True
                    break

        if not replaced:
            merged_summary[question_type].append(result)

    return merged_summary


def update_summary_file(summary_path: Path, new_results: List[Dict]) -> Dict[str, List[Dict]]:
    if not new_results:
        return load_existing_summary(summary_path)
    with locked_json_state(summary_path) as payload:
        existing_summary = load_existing_summary(summary_path) if payload == {} else load_existing_summary_from_payload(payload)
        merged_summary = merge_results_into_summary(existing_summary, new_results)
        payload.clear()
        payload.update(merged_summary)
        return merged_summary


def load_existing_summary_from_payload(payload: Dict) -> Dict[str, List[Dict]]:
    if not isinstance(payload, dict):
        return {}
    normalized_summary: Dict[str, List[Dict]] = {}

    legacy_items = payload.get("processed_files")
    if isinstance(legacy_items, list):
        for item in legacy_items:
            if not isinstance(item, dict):
                continue
            category = str(item.get("question_type", "") or "Unknown")
            normalized_summary.setdefault(category, []).append(item)

    for category, items in payload.items():
        if category == "processed_files":
            continue
        if not isinstance(category, str) or not isinstance(items, list):
            continue
        normalized_summary[category] = [item for item in items if isinstance(item, dict)]
    return normalized_summary


def build_qa_output_paths(metadata: Dict, output_root: Path) -> Tuple[Path, Optional[Path]]:
    markdown_path = resolve_markdown_path(metadata)
    relative_path: Path

    if markdown_path.is_relative_to(DEFAULT_CONSTRUCTION_PIPELINE_STAGE1_OUTPUT_ROOT):
        relative_path = markdown_path.relative_to(DEFAULT_CONSTRUCTION_PIPELINE_STAGE1_OUTPUT_ROOT)
    else:
        relative_path = Path(markdown_path.name)

    qa_output_path = output_root / relative_path.with_suffix(".qa.json")
    case_process_output_path = output_root / relative_path.with_suffix(".case_process.json")
    return qa_output_path, case_process_output_path


def resolve_markdown_path(record: Dict) -> Path:
    candidates = [
        record.get("markdown_path"),
        record.get("md_path"),
        record.get("mmd_path"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    raise FileNotFoundError(f"Markdown path not found for record: {record.get('source_path', '')}")


def build_exam_generation_prompt(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    source_name = metadata.get("Name") or Path(metadata.get("source_path", "Unknown")).stem
    return f"""
You are an MCQ extraction agent.

Your task is to convert the provided MCQ chunk into a JSON list of question objects.
The source is already an MCQ document. Preserve the source meaning and do not invent facts.

Rules:
- Return JSON only.
- The output must be a list.
- Extract complete questions only. Ignore incomplete or truncated fragments at the chunk boundaries.
- Avoid duplicates caused by overlap with neighboring chunks.
- Use `"from"` = "{source_name}" for every question.
- Generate MCQ objects only.
- Format:
  {{
    "from": "xxx",
    "question": "xxxx?",
    "options": {{
      "A": "Option A",
      "B": "Option B"
    }},
    "answer": "A",
    "reason": ""
  }}
- `answer` should be an option key such as A/B/C/D.
- Keep the wording faithful to the original text.

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Chunk index:
{chunk_index + 1} / {total_chunks}

Chunk text:
{chunk_text}
""".strip()


def build_exam_repair_prompt(metadata: Dict, chunk_text: str, broken_item: Dict) -> str:
    source_name = metadata.get("Name") or Path(metadata.get("source_path", "Unknown")).stem
    return f"""
You are repairing one invalid MCQ item.

Rules:
- Return JSON only.
- Return a single valid question object.
- Use `"from"` = "{source_name}".
- Stay faithful to the source text.
- The repaired item must be an MCQ and include `options`.
- Keep `reason` as an empty string if there is no reason to provide.

Broken item:
{json.dumps(broken_item, ensure_ascii=False, indent=2)}

Source chunk:
{chunk_text}
""".strip()


def build_exam_tagging_prompt(metadata: Dict, question_item: Dict) -> str:
    question_text = str(question_item.get("question", "") or "").strip()
    options = question_item.get("options")
    if isinstance(options, list) and options:
        question_text = (
            question_text
            + "\n"
            + "\n".join(f"{chr(65 + index)}. {str(option)}" for index, option in enumerate(options))
        ).strip()
    elif isinstance(options, dict) and options:
        question_text = (
            question_text
            + "\n"
            + "\n".join(f"{key}. {str(value)}" for key, value in options.items())
        ).strip()
    answer_text = str(question_item.get("answer", "") or "").strip()
    return QUESTION_TAGGING_USER_TEMPLATE.format(
        question=question_text,
        answer=answer_text,
    ).strip()


def build_saq_structure_prompt(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    return SAQ_STRUCTURE_PROMPT.format(
        metadata=json.dumps(metadata, ensure_ascii=False, indent=2),
        chunk_index=chunk_index + 1,
        total_chunks=total_chunks,
        chunk_text=chunk_text,
    ).strip()


def build_saq_generation_prompt(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    structure_context: str,
) -> str:
    source_name = metadata.get("Name") or Path(metadata.get("source_path", "Unknown")).stem
    return f"""
You are extracting dental short-answer questions from a book-like SAQ source.

Your task is to convert the provided chunk into a JSON list of SAQ objects.
Only keep complete question-answer pairs that are supported by this chunk and the structure guidance.

Rules:
- Return JSON only.
- The output must be a list.
- Every item must use this schema:
  {{
    "from": "{source_name}",
    "question": "xxxx?",
    "answer": "xxxx",
    "reason": ""
  }}
- Do not include `options`.
- Extract complete questions only. Ignore incomplete or truncated fragments at chunk boundaries.
- Only keep items whose answer can be recovered faithfully from the current chunk.
- If the chunk suggests a question exists but the answer is absent or ambiguous here, skip it.
- Avoid duplicates caused by overlap with neighboring chunks.
- Keep wording faithful to the source text.
- `reason` is optional and can be an empty string.

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Structure guidance:
{structure_context}

Chunk index:
{chunk_index + 1} / {total_chunks}

Chunk text:
{chunk_text}
""".strip()


def build_saq_repair_prompt(
    metadata: Dict,
    chunk_text: str,
    broken_item: Dict,
    structure_context: str,
) -> str:
    source_name = metadata.get("Name") or Path(metadata.get("source_path", "Unknown")).stem
    return f"""
You are repairing one invalid dental SAQ item.

Rules:
- Return JSON only.
- Return one valid SAQ object only.
- Use `"from"` = "{source_name}".
- The object must contain only: `from`, `question`, `answer`, `reason`.
- Do not include `options`.
- Stay faithful to the source chunk.
- If the original item is too broken to repair faithfully, return the closest valid object supported by the source chunk.

Structure guidance:
{structure_context}

Broken item:
{json.dumps(broken_item, ensure_ascii=False, indent=2)}

Source chunk:
{chunk_text}
""".strip()


def build_case_report_generation_prompt(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    chunk_header = (
        f"\n\nCHUNK INDEX:\n{chunk_index + 1} / {total_chunks}\n\n"
        f"DOCUMENT METADATA:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
    )
    return CASE_REPORT_EXTRACTION_USER_TEMPLATE.format(
        content=f"{chunk_header}\n{chunk_text}",
        key_point_count=CASE_KEY_POINT_COUNT,
    ).strip()


def build_case_report_validation_prompt(
    metadata: Dict,
    markdown_text: str,
    candidate: Dict,
) -> str:
    source_content = (
        f"DOCUMENT METADATA:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"{markdown_text}"
    )
    return CASE_REPORT_VALIDATION_USER_TEMPLATE.format(
        md_content=source_content,
        seed_question=json.dumps(candidate.get("seed_question", {}), ensure_ascii=False, indent=2),
        key_points=json.dumps(candidate.get("key_points", []), ensure_ascii=False, indent=2),
    ).strip()


def build_case_report_refinement_prompt(
    metadata: Dict,
    chunk_text: str,
    candidate: Dict,
    validation: Dict,
) -> str:
    return f"""
You are revising a case-report benchmark item after a strict quality audit.

Return JSON only with this exact schema:
{{
  "seed_question": {{
    "question": "",
    "location": "",
    "explanation": ""
  }},
  "key_points": [
    {{
      "content": "",
      "location": "",
      "explanation": ""
    }}
  ]
}}

Revision goals:
- Fix any hallucination or unsupported detail.
- Keep the question difficult but answerable.
- Make the question precise, self-contained, and clinically meaningful.
- Keep exactly ONE seed_question and exactly {CASE_KEY_POINT_COUNT} key_points.
- Ensure all key points directly support the seed question.
- If the original question is too broad or clinically debatable, revise it into a documented case-based reasoning question.
- Include only facts explicitly supported by the source chunk; omit missing demographics, labs, imaging, or history instead of guessing.
- Use real section titles/headings or nearby Markdown cues for locations; do not invent paragraph numbers.
- Stay faithful to the source chunk only.

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Original candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Validation feedback:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Source chunk:
{chunk_text}
""".strip()


def build_case_report_selection_prompt(metadata: Dict, candidates: List[Dict]) -> str:
    return f"""
You are selecting the single best benchmark item for a dental case report from multiple chunk-level candidates.

Return JSON only in this schema:
{{
  "selected_index": 0,
  "reason": ""
}}

Selection criteria:
- Prefer the candidate with the strongest clinical significance.
- Prefer the candidate with the clearest high-stakes dilemma.
- Prefer the candidate with the best support from its key points.
- Prefer the candidate with stronger correctness, difficulty, and answerability validation.

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
""".strip()


def build_case_report_tagging_prompt(metadata: Dict, item: Dict) -> str:
    del metadata
    question_text = str(item.get("seed_question", {}).get("question", "") or "").strip()
    return CASE_REPORT_TAGGING_USER_TEMPLATE.format(question_text=question_text).strip()


def is_non_empty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_exam_item_structure(item: Dict) -> Tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "Item is not a dict."
    if not is_non_empty_string(item.get("from")):
        return False, "Missing `from`."
    if not is_non_empty_string(item.get("question")):
        return False, "Missing `question`."
    if not is_non_empty_string(item.get("answer")):
        return False, "Missing `answer`."

    if "options" in item and item["options"] is not None:
        options = item["options"]
        if not isinstance(options, dict) or not options:
            return False, "Invalid `options`."
        non_empty_options = {
            key: value for key, value in options.items() if is_non_empty_string(key) and is_non_empty_string(value)
        }
        if len(non_empty_options) < 2:
            return False, "MCQ needs at least two options."
        answer = str(item["answer"]).strip()
        if answer not in non_empty_options:
            return False, "MCQ answer is not one of the option keys."

    if "reason" in item and item["reason"] is not None and not isinstance(item["reason"], str):
        return False, "`reason` must be a string."

    return True, ""


def validate_saq_item_structure(item: Dict) -> Tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "Item is not a dict."
    if not is_non_empty_string(item.get("from")):
        return False, "Missing `from`."
    if not is_non_empty_string(item.get("question")):
        return False, "Missing `question`."
    if not is_non_empty_string(item.get("answer")):
        return False, "Missing `answer`."
    if "options" in item:
        return False, "SAQ item must not contain `options`."
    if "reason" in item and item["reason"] is not None and not isinstance(item["reason"], str):
        return False, "`reason` must be a string."
    return True, ""


def validate_case_report_item_structure(item: Dict) -> Tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "Item is not a dict."

    seed_question = item.get("seed_question")
    if not isinstance(seed_question, dict):
        return False, "Missing `seed_question`."
    for field in ("question", "location", "explanation"):
        if not is_non_empty_string(seed_question.get(field)):
            return False, f"Invalid `seed_question.{field}`."

    key_points = item.get("key_points")
    if not isinstance(key_points, list) or len(key_points) != CASE_KEY_POINT_COUNT:
        return False, f"`key_points` must contain exactly {CASE_KEY_POINT_COUNT} items."

    for index, key_point in enumerate(key_points):
        if not isinstance(key_point, dict):
            return False, f"`key_points[{index}]` is not a dict."
        for field in ("content", "location", "explanation"):
            if not is_non_empty_string(key_point.get(field)):
                return False, f"Invalid `key_points[{index}].{field}`."

    return True, ""


def normalize_validation_result(response) -> Dict:
    if not isinstance(response, dict):
        response = {}

    normalized = {}
    for category in ("correctness", "difficulty", "answerability"):
        value = response.get(category, {})
        if not isinstance(value, dict):
            value = {}
        score = value.get("score", 0)
        normalized[category] = {
            "score": 1 if score == 1 else 0,
            "reasoning": str(value.get("reasoning", "") or ""),
        }

    normalized["overall_passed"] = int(
        all(normalized[category]["score"] == 1 for category in ("correctness", "difficulty", "answerability"))
    )
    normalized["total_score"] = sum(
        normalized[category]["score"] for category in ("correctness", "difficulty", "answerability")
    )
    return normalized


def deduplicate_exam_items(items: List[Dict]) -> List[Dict]:
    deduplicated = []
    seen_questions = set()

    for item in items:
        normalized_question = normalize_question_text(item.get("question", ""))
        if not normalized_question or normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        deduplicated.append(item)

    return deduplicated


def infer_dental_discipline_from_text(text: str) -> int:
    normalized = text.lower()
    keyword_map = [
        (12, ("orthodont", "malocclusion", "cephalometric", "class iii", "crossbite")),
        (10, ("implant", "peri-implant", "osseointegration", "bone graft", "zirconia crown")),
        (2, ("pulp", "periapical", "root canal", "endodont")),
        (6, ("maxillofacial", "orthognathic", "le fort", "bsso")),
        (8, ("radiograph", "cbct", "radiology", "panoramic")),
        (5, ("extraction", "dentoalveolar", "oral surgery")),
        (3, ("periodontal", "periodontitis", "gingival", "periodont")),
    ]
    for number, keywords in keyword_map:
        if any(keyword in normalized for keyword in keywords):
            return number
    return 0


def infer_reasoning_level_from_text(text: str) -> str:
    normalized = text.lower()
    if any(
        keyword in normalized
        for keyword in (
            "individualized",
            "patient-specific",
            "medical history",
            "systemic",
            "risk assessment",
            "management decision",
            "treatment choice",
            "justify",
            "evaluate",
            "trade-off",
            "tradeoff",
            "risk-benefit",
        )
    ):
        return "L3"
    if any(
        keyword in normalized
        for keyword in (
            "diagnosis",
            "differential diagnosis",
            "treatment planning",
            "clinical reasoning",
            "symptoms",
            "signs",
            "examination findings",
            "imaging",
        )
    ):
        return "L2"
    return "L1"


def parse_tagging_response(response, context_text: str = "") -> Dict:
    fallback = {
        "dental_discipline": {"number": infer_dental_discipline_from_text(context_text), "name": ""},
        "reasoning_level": infer_reasoning_level_from_text(context_text),
    }
    if fallback["dental_discipline"]["number"]:
        fallback["dental_discipline"]["name"] = TAG_TAXONOMY_MAP.get(fallback["dental_discipline"]["number"], "")

    if not isinstance(response, str):
        return fallback

    normalized_lines = [line.strip() for line in response.splitlines() if line.strip()]
    normalized_text = "\n".join(normalized_lines)

    discipline_number = 0
    discipline_match = re.search(
        r"(?:dental\s*discipline|taxonomy|label\s*1|category)?\D*\b([1-9]|1[0-4])\b",
        normalized_text,
        flags=re.IGNORECASE,
    )
    if discipline_match:
        discipline_number = int(discipline_match.group(1))
    else:
        lowered_response = normalized_text.lower()
        for number, name in TAG_TAXONOMY_MAP.items():
            if name.lower() in lowered_response:
                discipline_number = number
                break

    reasoning_match = re.search(r"\b(L[123])\b", normalized_text, flags=re.IGNORECASE)
    reasoning_level = reasoning_match.group(1).upper() if reasoning_match else fallback["reasoning_level"]

    if not discipline_number:
        discipline_number = fallback["dental_discipline"]["number"]
    if not discipline_number:
        return fallback

    return {
        "dental_discipline": {
            "number": discipline_number,
            "name": TAG_TAXONOMY_MAP.get(discipline_number, ""),
        },
        "reasoning_level": reasoning_level,
    }


def is_valid_tags(tags: Dict) -> bool:
    if not isinstance(tags, dict):
        return False
    discipline = tags.get("dental_discipline") or tags.get("taxonomy") or {}
    level = tags.get("reasoning_level") or tags.get("capability_level")
    return isinstance(discipline, dict) and bool(discipline.get("number")) and level in {"L1", "L2", "L3"}


def repair_exam_item_with_llm(
    metadata: Dict,
    chunk_text: str,
    broken_item: Dict,
    config_type: str,
    model: Optional[str],
) -> Optional[Dict]:
    prompt = build_exam_repair_prompt(metadata, chunk_text, broken_item)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt="Return valid JSON only.",
    )

    if not isinstance(response, dict):
        return None

    is_valid, _ = validate_exam_item_structure(response)
    return response if is_valid else None


def generate_exam_items_for_chunk(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    config_type: str,
    model: Optional[str],
) -> List[Dict]:
    prompt = build_exam_generation_prompt(metadata, chunk_text, chunk_index, total_chunks)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt="Return valid JSON only.",
    )

    if not isinstance(response, list):
        raise RuntimeError(f"Exam generation did not return a JSON list for chunk {chunk_index}.")

    validated_items = []
    for item in response:
        is_valid, _ = validate_exam_item_structure(item)
        if is_valid:
            validated_items.append(item)
            continue

        repaired_item = repair_exam_item_with_llm(
            metadata=metadata,
            chunk_text=chunk_text,
            broken_item=item if isinstance(item, dict) else {"raw_item": item},
            config_type=config_type,
            model=model,
        )
        if repaired_item is not None:
            validated_items.append(repaired_item)

    return deduplicate_exam_items(validated_items)


def analyze_saq_structure_for_chunk(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    config_type: str,
    model: Optional[str],
) -> Dict:
    prompt = build_saq_structure_prompt(metadata, chunk_text, chunk_index, total_chunks)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt=SAQ_STRUCTURE_SYSTEM_PROMPT,
    )
    return normalize_saq_structure_result(response)


def repair_saq_item_with_llm(
    metadata: Dict,
    chunk_text: str,
    broken_item: Dict,
    structure_context: str,
    config_type: str,
    model: Optional[str],
) -> Optional[Dict]:
    prompt = build_saq_repair_prompt(
        metadata=metadata,
        chunk_text=chunk_text,
        broken_item=broken_item,
        structure_context=structure_context,
    )
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt="Return valid JSON only.",
    )

    if not isinstance(response, dict):
        return None

    is_valid, _ = validate_saq_item_structure(response)
    return response if is_valid else None


def generate_saq_items_for_chunk(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    structure_context: str,
    config_type: str,
    model: Optional[str],
) -> List[Dict]:
    prompt = build_saq_generation_prompt(
        metadata=metadata,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        structure_context=structure_context,
    )
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt=SAQ_GENERATION_SYSTEM_PROMPT,
    )

    if not isinstance(response, list):
        raise RuntimeError(f"SAQ generation did not return a JSON list for chunk {chunk_index}.")

    validated_items = []
    for item in response:
        is_valid, _ = validate_saq_item_structure(item)
        if is_valid:
            validated_items.append(item)
            continue

        repaired_item = repair_saq_item_with_llm(
            metadata=metadata,
            chunk_text=chunk_text,
            broken_item=item if isinstance(item, dict) else {"raw_item": item},
            structure_context=structure_context,
            config_type=config_type,
            model=model,
        )
        if repaired_item is not None:
            validated_items.append(repaired_item)

    return deduplicate_exam_items(validated_items)


def generate_tags_for_item(
    metadata: Dict,
    question_item: Dict,
    config_type: str,
    model: Optional[str],
) -> Dict:
    prompt = build_exam_tagging_prompt(metadata, question_item)
    context_text = f"{prompt}\n{json.dumps(metadata, ensure_ascii=False)}"
    tags = parse_tagging_response("", context_text=context_text)
    for _ in range(3):
        response = call_gpt(
            prompt=prompt,
            config_type=config_type,
            model=model,
            json_output=False,
            system_prompt=QUESTION_TAGGING_SYSTEM_PROMPT,
        )
        tags = parse_tagging_response(response, context_text=context_text)
        if is_valid_tags(tags):
            return tags
    return tags


def generate_tags_for_case_report(
    metadata: Dict,
    item: Dict,
    config_type: str,
    model: Optional[str],
) -> Dict:
    prompt = build_case_report_tagging_prompt(metadata, item)
    context_text = f"{prompt}\n{json.dumps(item.get('key_points', []), ensure_ascii=False)}\n{json.dumps(metadata, ensure_ascii=False)}"
    tags = parse_tagging_response("", context_text=context_text)
    for _ in range(3):
        response = call_gpt(
            prompt=prompt,
            config_type=config_type,
            model=model,
            json_output=False,
            system_prompt=CASE_REPORT_TAGGING_SYSTEM_PROMPT,
        )
        tags = parse_tagging_response(response, context_text=context_text)
        if is_valid_tags(tags):
            return tags
    return tags


def generate_case_report_item_for_chunk(
    metadata: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    config_type: str,
    model: Optional[str],
) -> Optional[Dict]:
    prompt = build_case_report_generation_prompt(metadata, chunk_text, chunk_index, total_chunks)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt=CASE_REPORT_SYSTEM_PROMPT,
    )

    if not isinstance(response, dict):
        return None

    is_valid, _ = validate_case_report_item_structure(response)
    return response if is_valid else None


def validate_case_report_candidate(
    metadata: Dict,
    markdown_text: str,
    candidate: Dict,
    config_type: str,
    model: Optional[str],
) -> Dict:
    prompt = build_case_report_validation_prompt(metadata, markdown_text, candidate)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt=CASE_REPORT_VALIDATION_SYSTEM_PROMPT,
    )
    return normalize_validation_result(response)


def refine_case_report_candidate(
    metadata: Dict,
    chunk_text: str,
    candidate: Dict,
    validation: Dict,
    config_type: str,
    model: Optional[str],
) -> Optional[Dict]:
    prompt = build_case_report_refinement_prompt(metadata, chunk_text, candidate, validation)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt="Return valid JSON only.",
    )

    if not isinstance(response, dict):
        return None

    is_valid, _ = validate_case_report_item_structure(response)
    return response if is_valid else None


def select_best_case_report_candidate(
    metadata: Dict,
    candidates: List[Dict],
    config_type: str,
    model: Optional[str],
) -> Dict:
    if len(candidates) == 1:
        return candidates[0]

    selection_payload = [
        {
            "index": index,
            "seed_question": candidate["seed_question"],
            "key_points": candidate["key_points"],
            "validation": candidate["validation"],
        }
        for index, candidate in enumerate(candidates)
    ]

    prompt = build_case_report_selection_prompt(metadata, selection_payload)
    response = call_gpt(
        prompt=prompt,
        config_type=config_type,
        model=model,
        json_output=True,
        system_prompt="Return valid JSON only.",
    )

    if isinstance(response, dict):
        selected_index = response.get("selected_index")
        if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
            return candidates[selected_index]

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["validation"]["overall_passed"],
            candidate["validation"]["total_score"],
        ),
        reverse=True,
    )
    return ranked_candidates[0]


def analyze_document_processing(
    metadata: Dict,
    markdown_text: str,
    tokenizer: Optional[Tokenizer],
    token_count_mode: str,
    config_type: str,
    model: Optional[str],
    api_token_count_cache: Dict[str, int],
    agent_max_output_tokens: int,
    agent_max_read_tokens: int,
    chunk_ratio: float,
    overlap_tokens: int,
    case_overlap_tokens: int,
) -> Dict:
    doc_type = metadata.get("Type", "")
    token_count = count_tokens(
        text=markdown_text,
        tokenizer=tokenizer,
        token_count_mode=token_count_mode,
        config_type=config_type,
        model=model,
        api_token_count_cache=api_token_count_cache,
    )
    exam_threshold = int(agent_max_output_tokens * chunk_ratio)

    if doc_type == MCQ_TYPE:
        chunks = create_token_chunks(
            text=markdown_text,
            tokenizer=tokenizer,
            chunk_size_tokens=exam_threshold,
            overlap_tokens=overlap_tokens,
            token_count_mode=token_count_mode,
            config_type=config_type,
            model=model,
            api_token_count_cache=api_token_count_cache,
        )
        strategy = "single_chunk" if len(chunks) == 1 else "token_chunked_mcq"
        return {
            "type": doc_type,
            "token_count": token_count,
            "chunk_threshold": exam_threshold,
            "chunk_count": len(chunks),
            "strategy": strategy,
            "chunks": chunks,
        }

    if doc_type == SAQ_TYPE:
        chunks = create_token_chunks(
            text=markdown_text,
            tokenizer=tokenizer,
            chunk_size_tokens=agent_max_read_tokens,
            overlap_tokens=case_overlap_tokens,
            token_count_mode=token_count_mode,
            config_type=config_type,
            model=model,
            api_token_count_cache=api_token_count_cache,
        )
        strategy = "single_chunk" if len(chunks) == 1 else "token_chunked_saq_structure"
        return {
            "type": doc_type,
            "token_count": token_count,
            "chunk_threshold": agent_max_read_tokens,
            "chunk_count": len(chunks),
            "strategy": strategy,
            "chunks": chunks,
        }

    if doc_type == CASE_REPORT_TYPE:
        chunks = create_token_chunks(
            text=markdown_text,
            tokenizer=tokenizer,
            chunk_size_tokens=agent_max_read_tokens,
            overlap_tokens=case_overlap_tokens,
            token_count_mode=token_count_mode,
            config_type=config_type,
            model=model,
            api_token_count_cache=api_token_count_cache,
        )
        strategy = "single_chunk" if len(chunks) == 1 else "token_chunked_case_report"
        return {
            "type": doc_type,
            "token_count": token_count,
            "chunk_threshold": agent_max_read_tokens,
            "chunk_count": len(chunks),
            "strategy": strategy,
            "chunks": chunks,
        }

    return {
        "type": doc_type,
        "token_count": token_count,
        "chunk_threshold": 0,
        "chunk_count": 0,
        "strategy": "unsupported_document_type",
        "chunks": [],
    }


class QAGenerationAgent:
    def __init__(
        self,
        tokenizer: Optional[Tokenizer],
        config_type: str,
        model: Optional[str],
        output_root: Path,
        document_type_override: Optional[str],
        token_count_mode: str,
        agent_max_output_tokens: int,
        agent_max_read_tokens: int,
        chunk_ratio: float,
        overlap_tokens: int,
        case_overlap_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.config_type = config_type
        self.model = model
        self.output_root = output_root
        self.document_type_override = normalize_document_type(document_type_override)
        self.token_count_mode = token_count_mode
        self.agent_max_output_tokens = agent_max_output_tokens
        self.agent_max_read_tokens = agent_max_read_tokens
        self.chunk_ratio = chunk_ratio
        self.overlap_tokens = overlap_tokens
        self.case_overlap_tokens = case_overlap_tokens
        self.api_token_count_cache: Dict[str, int] = {}

    def process_exam_record(self, metadata: Dict, markdown_text: str) -> Dict:
        analysis = analyze_document_processing(
            metadata=metadata,
            markdown_text=markdown_text,
            tokenizer=self.tokenizer,
            token_count_mode=self.token_count_mode,
            config_type=self.config_type,
            model=self.model,
            api_token_count_cache=self.api_token_count_cache,
            agent_max_output_tokens=self.agent_max_output_tokens,
            agent_max_read_tokens=self.agent_max_read_tokens,
            chunk_ratio=self.chunk_ratio,
            overlap_tokens=self.overlap_tokens,
            case_overlap_tokens=self.case_overlap_tokens,
        )

        exam_items = []
        total_chunks = len(analysis["chunks"])
        for chunk in analysis["chunks"]:
            chunk_items = generate_exam_items_for_chunk(
                metadata=metadata,
                chunk_text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                total_chunks=total_chunks,
                config_type=self.config_type,
                model=self.model,
            )
            exam_items.extend(chunk_items)

        exam_items = deduplicate_exam_items(exam_items)
        final_items = []
        for item in exam_items:
            enriched_item = dict(item)
            enriched_item["question_type"] = MCQ_TYPE
            enriched_item["tags"] = generate_tags_for_item(
                metadata=metadata,
                question_item=enriched_item,
                config_type=self.config_type,
                model=self.model,
            )
            final_items.append(enriched_item)

        return {
            "source_path": metadata.get("source_path", ""),
            "markdown_path": metadata.get("markdown_path", ""),
            "Name": metadata.get("Name", ""),
            "Type": MCQ_TYPE,
            "analysis": {
                "token_count": analysis["token_count"],
                "chunk_threshold": analysis["chunk_threshold"],
                "chunk_count": analysis["chunk_count"],
                "strategy": analysis["strategy"],
            },
            "questions": final_items,
        }

    def process_saq_record(self, metadata: Dict, markdown_text: str) -> Dict:
        structure_analysis = analyze_document_processing(
            metadata=metadata,
            markdown_text=markdown_text,
            tokenizer=self.tokenizer,
            token_count_mode=self.token_count_mode,
            config_type=self.config_type,
            model=self.model,
            api_token_count_cache=self.api_token_count_cache,
            agent_max_output_tokens=self.agent_max_output_tokens,
            agent_max_read_tokens=self.agent_max_read_tokens,
            chunk_ratio=self.chunk_ratio,
            overlap_tokens=self.overlap_tokens,
            case_overlap_tokens=self.case_overlap_tokens,
        )

        structure_reports = []
        total_structure_chunks = len(structure_analysis["chunks"])
        for chunk in structure_analysis["chunks"]:
            structure = analyze_saq_structure_for_chunk(
                metadata=metadata,
                chunk_text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                total_chunks=total_structure_chunks,
                config_type=self.config_type,
                model=self.model,
            )
            structure_reports.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                    "text": chunk["text"],
                    "structure": structure,
                }
            )

        extraction_segments = select_saq_extraction_segments(markdown_text, structure_reports)
        generation_threshold = int(self.agent_max_output_tokens * self.chunk_ratio)
        generation_chunks = []
        for segment in extraction_segments:
            segment_chunks = create_token_chunks(
                text=segment["text"],
                tokenizer=self.tokenizer,
                chunk_size_tokens=generation_threshold,
                overlap_tokens=self.overlap_tokens,
                token_count_mode=self.token_count_mode,
                config_type=self.config_type,
                model=self.model,
                api_token_count_cache=self.api_token_count_cache,
            )
            for chunk in segment_chunks:
                generation_chunks.append(
                    {
                        "segment_type": segment.get("segment_type", "unknown"),
                        "segment_title": segment.get("title", ""),
                        "token_start": chunk["token_start"],
                        "token_end": chunk["token_end"],
                        "text": chunk["text"],
                    }
                )

        structure_context = build_saq_structure_context(structure_reports)
        saq_items = []
        total_generation_chunks = len(generation_chunks)
        for chunk_index, chunk in enumerate(generation_chunks):
            chunk_items = generate_saq_items_for_chunk(
                metadata=metadata,
                chunk_text=chunk["text"],
                chunk_index=chunk_index,
                total_chunks=total_generation_chunks,
                structure_context=structure_context,
                config_type=self.config_type,
                model=self.model,
            )
            saq_items.extend(chunk_items)

        saq_items = deduplicate_exam_items(saq_items)
        final_items = []
        for item in saq_items:
            enriched_item = dict(item)
            enriched_item["question_type"] = SAQ_TYPE
            enriched_item["tags"] = generate_tags_for_item(
                metadata=metadata,
                question_item=enriched_item,
                config_type=self.config_type,
                model=self.model,
            )
            final_items.append(enriched_item)

        return {
            "source_path": metadata.get("source_path", ""),
            "markdown_path": metadata.get("markdown_path", ""),
            "Name": metadata.get("Name", ""),
            "Type": SAQ_TYPE,
            "analysis": {
                "token_count": structure_analysis["token_count"],
                "structure_chunk_threshold": structure_analysis["chunk_threshold"],
                "structure_chunk_count": structure_analysis["chunk_count"],
                "structure_strategy": structure_analysis["strategy"],
                "generation_chunk_threshold": generation_threshold,
                "generation_chunk_count": total_generation_chunks,
                "extraction_segment_count": len(extraction_segments),
            },
            "structure_analysis": [
                {
                    "chunk_index": report["chunk_index"],
                    "token_start": report["token_start"],
                    "token_end": report["token_end"],
                    "structure": report["structure"],
                }
                for report in structure_reports
            ],
            "questions": final_items,
        }

    def process_case_report_record(self, metadata: Dict, markdown_text: str) -> Dict:
        analysis = analyze_document_processing(
            metadata=metadata,
            markdown_text=markdown_text,
            tokenizer=self.tokenizer,
            token_count_mode=self.token_count_mode,
            config_type=self.config_type,
            model=self.model,
            api_token_count_cache=self.api_token_count_cache,
            agent_max_output_tokens=self.agent_max_output_tokens,
            agent_max_read_tokens=self.agent_max_read_tokens,
            chunk_ratio=self.chunk_ratio,
            overlap_tokens=self.overlap_tokens,
            case_overlap_tokens=self.case_overlap_tokens,
        )

        candidates = []
        total_chunks = len(analysis["chunks"])
        for chunk in analysis["chunks"]:
            candidate = generate_case_report_item_for_chunk(
                metadata=metadata,
                chunk_text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                total_chunks=total_chunks,
                config_type=self.config_type,
                model=self.model,
            )
            if candidate is None:
                log_message(f"CaseReport chunk {chunk['chunk_index']}: candidate generation returned None")
                continue

            validation = validate_case_report_candidate(
                metadata=metadata,
                markdown_text=markdown_text,
                candidate=candidate,
                config_type=self.config_type,
                model=self.model,
            )

            final_candidate = candidate
            final_validation = validation
            refinement_rounds = 0
            while (
                final_validation["overall_passed"] != 1
                and refinement_rounds < MAX_CASE_REPORT_REFINEMENT_ROUNDS
            ):
                refined_candidate = refine_case_report_candidate(
                    metadata=metadata,
                    chunk_text=chunk["text"],
                    candidate=final_candidate,
                    validation=final_validation,
                    config_type=self.config_type,
                    model=self.model,
                )
                refinement_rounds += 1
                if refined_candidate is None:
                    break

                refined_validation = validate_case_report_candidate(
                    metadata=metadata,
                    markdown_text=markdown_text,
                    candidate=refined_candidate,
                    config_type=self.config_type,
                    model=self.model,
                )
                final_candidate = refined_candidate
                final_validation = refined_validation

            log_message(
                f"CaseReport chunk {chunk['chunk_index']}: "
                f"validation={final_validation.get('total_score', 0)}/3, "
                f"passed={final_validation.get('overall_passed', 0)}, "
                f"refinement_rounds={refinement_rounds}"
            )

            if final_validation["overall_passed"] != 1:
                continue

            candidates.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "seed_question": final_candidate["seed_question"],
                    "key_points": final_candidate["key_points"],
                    "validation": final_validation,
                    "refinement_rounds": refinement_rounds,
                }
            )

        if not candidates:
            raise RuntimeError("No valid Case Report candidate could be generated.")

        selected_candidate = select_best_case_report_candidate(
            metadata=metadata,
            candidates=candidates,
            config_type=self.config_type,
            model=self.model,
        )

        final_payload = {
            "from": metadata.get("Name") or Path(metadata.get("source_path", "Unknown")).stem,
            "question_type": CASE_REPORT_TYPE,
            "seed_question": selected_candidate["seed_question"],
            "key_points": selected_candidate["key_points"],
        }
        final_payload["tags"] = generate_tags_for_case_report(
            metadata=metadata,
            item=final_payload,
            config_type=self.config_type,
            model=self.model,
        )
        final_payload["_process"] = {
            "source_path": metadata.get("source_path", ""),
            "markdown_path": metadata.get("markdown_path", ""),
            "analysis": {
                "token_count": analysis["token_count"],
                "chunk_threshold": analysis["chunk_threshold"],
                "chunk_count": analysis["chunk_count"],
                "strategy": analysis["strategy"],
            },
            "candidates": candidates,
            "selected_candidate": selected_candidate,
        }
        return final_payload

    def process_record(self, metadata: Dict) -> Dict:
        markdown_path = resolve_markdown_path(metadata)
        markdown_text = normalize_text(read_text_file(markdown_path))
        doc_type = self.document_type_override or normalize_document_type(metadata.get("Type", ""))

        if doc_type == MCQ_TYPE:
            result = self.process_exam_record(metadata, markdown_text)
        elif doc_type == SAQ_TYPE:
            result = self.process_saq_record(metadata, markdown_text)
        elif doc_type == CASE_REPORT_TYPE:
            result = self.process_case_report_record(metadata, markdown_text)
        else:
            raise ValueError(f"Unsupported document Type: `{doc_type}`")

        output_path, process_output_path = build_qa_output_paths(metadata, self.output_root)
        if doc_type == CASE_REPORT_TYPE and "_process" in result:
            write_json_file(process_output_path, result["_process"])
            result = dict(result)
            result.pop("_process", None)
        write_json_file(output_path, result)
        result["output_path"] = str(output_path)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated QA JSON files from Markdown and metadata for MCQ, SAQ, and CaseReport documents."
    )
    parser.add_argument(
        "--metadata-input",
        type=str,
        required=False,
        help="A metadata JSON file, a run_summary.json file, or a directory containing metadata JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory for generated QA JSON files.",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default=None,
        help="Config section in config/config.json for LLM calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for LLM calls.",
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default="auto",
        choices=["auto", MCQ_TYPE, SAQ_TYPE, CASE_REPORT_TYPE],
        help="Manually force all input documents to be treated as MCQ, SAQ, or CaseReport. Use `auto` to follow metadata Type.",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=str,
        default=None,
        help="Tokenizer source name or local tokenizer.json path. Optional when using API-based token counting.",
    )
    parser.add_argument(
        "--token-count-mode",
        type=str,
        default=DEFAULT_TOKEN_COUNT_MODE,
        choices=["auto", "local", "api"],
        help="How to count tokens for chunking. `auto` prefers local tokenizer when available, otherwise falls back to API usage counting.",
    )
    parser.add_argument(
        "--agent-max-output-tokens",
        type=int,
        default=None,
        help="Maximum token generation length used to decide MCQ and SAQ extraction chunking.",
    )
    parser.add_argument(
        "--agent-max-read-tokens",
        type=int,
        default=None,
        help="Maximum reading length used to decide SAQ structure-analysis and Case Report chunking.",
    )
    parser.add_argument(
        "--chunk-ratio",
        type=float,
        default=None,
        help="MCQ chunk threshold ratio relative to the agent max output tokens.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=None,
        help="Token overlap between neighboring MCQ and SAQ generation chunks.",
    )
    parser.add_argument(
        "--case-overlap-tokens",
        type=int,
        default=None,
        help="Token overlap between neighboring SAQ structure-analysis and Case Report chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many metadata records to process.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess files even if the target QA output already exists.",
    )
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default=None,
        help="Optional stage-02 buffer directory. When provided with --watch-buffer, 02 will keep consuming new stage-01 items from this buffer.",
    )
    parser.add_argument(
        "--watch-buffer",
        action="store_true",
        help="Continuously watch the buffer directory and process new items as they arrive.",
    )
    parser.add_argument(
        "--buffer-poll-interval",
        type=float,
        default=DEFAULT_BUFFER_POLL_INTERVAL_SECONDS,
        help="Polling interval in seconds when watching the stage-02 buffer.",
    )
    parser.add_argument(
        "--buffer-idle-exit-seconds",
        type=float,
        default=DEFAULT_BUFFER_IDLE_EXIT_SECONDS,
        help="When watching the buffer, exit after this many idle seconds with no new tasks.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Parallel worker count for stage 02 processing.",
    )
    parser.add_argument(
        "--progress-state",
        type=str,
        default=None,
        help="Optional shared JSON file used to track pipeline progress.",
    )
    parser.add_argument(
        "--producer-done-signal",
        type=str,
        default=None,
        help="Optional file path. In buffer watch mode, exit only after this signal exists and the buffer stays empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_config = load_qa_generation_config()
    metadata_input = Path(args.metadata_input).resolve() if args.metadata_input else None
    buffer_dir = Path(args.buffer_dir).resolve() if args.buffer_dir else None
    progress_state_path = Path(args.progress_state).resolve() if args.progress_state else None
    producer_done_signal = Path(args.producer_done_signal).resolve() if args.producer_done_signal else None
    output_root = Path(
        resolve_runtime_setting(
            args.output_root,
            qa_config.get("output_root"),
            DEFAULT_CONSTRUCTION_PIPELINE_STAGE2_OUTPUT_ROOT,
        )
    ).resolve()

    if metadata_input is None and not args.watch_buffer:
        raise ValueError("Please provide `--metadata-input`, or use `--buffer-dir --watch-buffer`.")
    if metadata_input is not None and not metadata_input.exists():
        raise FileNotFoundError(f"Metadata input does not exist: {metadata_input}")
    if args.watch_buffer and buffer_dir is None:
        buffer_dir = Path(DEFAULT_STAGE2_BUFFER_DIR).resolve()

    config_type = resolve_runtime_setting(args.config_type, qa_config.get("config_type"), "llm")
    tokenizer_source = resolve_runtime_setting(
        args.tokenizer_source,
        qa_config.get("tokenizer_source"),
        None,
    )
    token_count_mode = resolve_token_count_mode(
        token_count_mode=args.token_count_mode,
        tokenizer_source=tokenizer_source,
        config_type=config_type,
    )
    agent_max_output_tokens = resolve_runtime_setting(
        args.agent_max_output_tokens,
        qa_config.get("agent_max_output_tokens"),
        DEFAULT_AGENT_MAX_OUTPUT_TOKENS,
    )
    agent_max_read_tokens = resolve_runtime_setting(
        args.agent_max_read_tokens,
        qa_config.get("agent_max_read_tokens"),
        DEFAULT_AGENT_MAX_READ_TOKENS,
    )
    chunk_ratio = resolve_runtime_setting(
        args.chunk_ratio,
        qa_config.get("chunk_ratio"),
        DEFAULT_CHUNK_RATIO,
    )
    overlap_tokens = resolve_runtime_setting(
        args.overlap_tokens,
        qa_config.get("overlap_tokens"),
        DEFAULT_CHUNK_OVERLAP_TOKENS,
    )
    case_overlap_tokens = resolve_runtime_setting(
        args.case_overlap_tokens,
        qa_config.get("case_overlap_tokens"),
        DEFAULT_CASE_OVERLAP_TOKENS,
    )

    tokenizer = load_tokenizer(tokenizer_source, config_type) if token_count_mode == "local" else None

    agent = QAGenerationAgent(
        tokenizer=tokenizer,
        config_type=config_type,
        model=args.model,
        output_root=output_root,
        document_type_override=None if args.document_type == "auto" else args.document_type,
        token_count_mode=token_count_mode,
        agent_max_output_tokens=agent_max_output_tokens,
        agent_max_read_tokens=agent_max_read_tokens,
        chunk_ratio=chunk_ratio,
        overlap_tokens=overlap_tokens,
        case_overlap_tokens=case_overlap_tokens,
    )

    resume_index_path = output_root / "qa_resume_index.json"
    resume_index = load_resume_index(resume_index_path)
    summary_path = output_root / "qa_run_summary.json"

    def handle_single_record(record: Dict, position_label: str) -> Optional[Dict]:
        source_path = str(record.get("source_path", "") or "")
        doc_type = normalize_document_type(
            None if args.document_type == "auto" else args.document_type
        ) or normalize_document_type(record.get("Type", ""))
        output_path, process_output_path = build_qa_output_paths(record, output_root)

        if output_path.exists() and not args.force_reprocess:
            log_message(f"{position_label} Skipped existing: {source_path}")
            existing_result = load_existing_result_for_summary(output_path)
            if source_path:
                resume_index[source_path] = build_resume_record(
                    metadata=record,
                    output_path=output_path,
                    question_type=str(existing_result.get("question_type", "") or doc_type or "Unknown"),
                    status="skipped_existing",
                    process_output_path=process_output_path if process_output_path.exists() else None,
                )
            update_summary_file(summary_path, [existing_result])
            update_progress_state(progress_state_path, "stage2_completed")
            return existing_result

        log_message(f"{position_label} Processing: {source_path}")
        result = agent.process_record(record)
        log_message(f"Saved QA JSON: {result['output_path']}")
        if source_path:
            resume_index[source_path] = build_resume_record(
                metadata=record,
                output_path=Path(result["output_path"]),
                question_type=str(result.get("question_type", "") or doc_type or "Unknown"),
                status="processed",
                process_output_path=process_output_path if process_output_path.exists() else None,
            )
        update_summary_file(summary_path, [result])
        update_progress_state(progress_state_path, "stage2_completed")
        return result

    results: List[Dict] = []

    if metadata_input is not None:
        records = load_metadata_records(metadata_input)
        if args.limit is not None:
            records = records[: args.limit]
        for index, record in enumerate(records, start=1):
            try:
                result = handle_single_record(record, f"[{index}/{len(records)}]")
                if result is not None:
                    results.append(result)
            except Exception as exc:
                import traceback
                source_path = str(record.get("source_path", "") or "")
                error_detail = f"{exc}\n{traceback.format_exc()}"
                if source_path:
                    resume_index[source_path] = {
                        "source_path": source_path,
                        "markdown_path": str(record.get("markdown_path", "") or ""),
                        "question_type": normalize_document_type(
                            None if args.document_type == "auto" else args.document_type
                        ) or normalize_document_type(record.get("Type", "")),
                        "output_path": str(build_qa_output_paths(record, output_root)[0]),
                        "process_output_path": str(build_qa_output_paths(record, output_root)[1]),
                        "status": "failed",
                        "error": error_detail,
                    }
                update_progress_state(progress_state_path, "stage2_failed")
                log_message(f"[{index}/{len(records)}] Failed: {source_path}: {error_detail}")
    else:
        ensure_buffer_dirs(buffer_dir)
        active_futures = {}
        idle_start = time.time()
        pending_dirs = ensure_buffer_dirs(buffer_dir)
        with ThreadPoolExecutor(max_workers=max(1, args.worker_count)) as executor:
            while True:
                while len(active_futures) < max(1, args.worker_count):
                    claimed = claim_next_buffer_item(buffer_dir)
                    if claimed is None:
                        break
                    processing_path, _, payload = claimed
                    metadata_path = Path(payload["metadata_path"]).resolve()
                    record = json.loads(read_text_file(metadata_path))
                    future = executor.submit(handle_single_record, record, "[buffer]")
                    active_futures[future] = (processing_path, payload, record)
                    idle_start = time.time()

                if active_futures:
                    done_futures, _ = wait(active_futures.keys(), timeout=args.buffer_poll_interval, return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        processing_path, payload, record = active_futures.pop(future)
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                            finalize_buffer_item(processing_path, pending_dirs["done"])
                            idle_start = time.time()
                        except Exception as exc:
                            failed_payload = {
                                **payload,
                                "error": str(exc),
                                "source_path": str(record.get("source_path", "") or payload.get("source_path", "")),
                            }
                            source_path = failed_payload["source_path"]
                            if source_path:
                                resume_index[source_path] = {
                                    "source_path": source_path,
                                    "markdown_path": str(record.get("markdown_path", "") or payload.get("markdown_path", "")),
                                    "question_type": normalize_document_type(
                                        None if args.document_type == "auto" else args.document_type
                                    ) or normalize_document_type(record.get("Type", "") or payload.get("type", "")),
                                    "output_path": str(payload.get("output_path", "") or ""),
                                    "process_output_path": str(payload.get("process_output_path", "") or ""),
                                    "status": "failed",
                                    "error": str(exc),
                                }
                            processing_path.write_text(json.dumps(failed_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                            finalize_buffer_item(processing_path, pending_dirs["failed"])
                            update_progress_state(progress_state_path, "stage2_failed")
                            log_message(f"[buffer] Failed: {failed_payload['source_path']}: {exc}")
                    continue

                has_pending_new = any(pending_dirs["new"].glob("*.json"))
                has_pending_processing = any(pending_dirs["processing"].glob("*.json"))
                producer_done = producer_done_signal is not None and producer_done_signal.exists()
                if (
                    producer_done
                    and not has_pending_new
                    and not has_pending_processing
                    and time.time() - idle_start >= args.buffer_idle_exit_seconds
                ):
                    break
                time.sleep(args.buffer_poll_interval)

    merged_summary = update_summary_file(summary_path, results)
    write_resume_index(resume_index_path, resume_index)
    log_message(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


# Run from the project root:
# python construction_pipeline/02-QA_Generation.py --metadata-input 00-Data_and_Models/construction_pipeline/01_markdown_and_metadata/run_summary.json --document-type MCQ --tokenizer-source 00-Data_and_Models/tokenizers/your_tokenizer.json
# python construction_pipeline/02-QA_Generation.py --metadata-input 00-Data_and_Models/construction_pipeline/01_markdown_and_metadata --document-type SAQ --tokenizer-source Qwen/Qwen2.5-7B-Instruct
