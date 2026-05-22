# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Build a clean benchmark JSON file with top-level MCQ, SAQ, and CBQ lists from qa_run_summary.json, filtering failed or incomplete records.

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA_SUMMARY_PATH = Path("00-Data_and_Models/construction_pipeline_outputs/02_qa_outputs/qa_run_summary.json")
DEFAULT_OUTPUT_DIR = Path("00-Data_and_Models/benchmarks")


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_benchmark_filename(benchmark_name: str) -> str:
    name = benchmark_name.strip() or "GlobalDentBench"
    return name if name.endswith(".json") else f"{name}.json"


def has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_failed_record(item: Dict) -> bool:
    status = str(item.get("status", "") or "").strip().lower()
    return status == "failed" or bool(item.get("error"))


def is_valid_mcq(item: Dict) -> bool:
    options = item.get("options")
    return (
        has_text(item.get("question"))
        and isinstance(options, (dict, list))
        and bool(options)
        and has_text(str(item.get("answer", "") or ""))
    )


def is_valid_saq(item: Dict) -> bool:
    return has_text(item.get("question")) and has_text(str(item.get("answer", "") or ""))


def is_valid_case_report(item: Dict) -> bool:
    seed_question = item.get("seed_question")
    key_points = item.get("key_points")
    return (
        isinstance(seed_question, dict)
        and has_text(seed_question.get("question"))
        and isinstance(key_points, list)
        and bool(key_points)
        and all(isinstance(point, dict) and has_text(point.get("content")) for point in key_points)
    )


def infer_question_type(category: str, item: Dict) -> str:
    return str(item.get("question_type") or item.get("Type") or category or "").strip()


def output_category(question_type: str) -> str:
    if question_type == "CaseReport":
        return "CBQ"
    if question_type in {"MCQ", "SAQ", "CBQ"}:
        return question_type
    return ""


def locate_case_process_path(item: Dict, qa_summary_path: Path) -> Optional[Path]:
    output_path = item.get("output_path")
    if output_path:
        qa_path = Path(str(output_path))
        if qa_path.exists():
            candidate = qa_path.with_suffix(".case_process.json")
            if candidate.exists():
                return candidate

    output_dir = qa_summary_path.parent
    source_name = str(item.get("from", "") or "").strip()
    for candidate in output_dir.glob("*.case_process.json"):
        try:
            payload = read_json(candidate)
        except Exception:
            continue
        selected = payload.get("selected_candidate", {})
        if selected.get("seed_question") == item.get("seed_question"):
            return candidate
        if source_name and source_name in candidate.name:
            return candidate
    return None


def case_validation_passed(process_path: Optional[Path]) -> Tuple[bool, str]:
    if process_path is None:
        return False, "missing_case_process"

    try:
        payload = read_json(process_path)
    except Exception as exc:
        return False, f"invalid_case_process: {exc}"

    selected = payload.get("selected_candidate", {})
    validation = selected.get("validation", {})
    if validation.get("overall_passed") == 1:
        return True, "passed"

    for candidate in payload.get("candidates", []):
        validation = candidate.get("validation", {})
        if validation.get("overall_passed") == 1:
            return True, "passed"
    return False, "validation_not_passed"


def record_is_complete(category: str, item: Dict, qa_summary_path: Path) -> Tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not_json_object"
    if is_failed_record(item):
        return False, "failed_record"

    question_type = infer_question_type(category, item)
    if question_type == "MCQ":
        return (True, "passed") if is_valid_mcq(item) else (False, "incomplete_mcq")
    if question_type == "SAQ":
        return (True, "passed") if is_valid_saq(item) else (False, "incomplete_saq")
    if question_type in {"CaseReport", "CBQ"}:
        if not is_valid_case_report(item):
            return False, "incomplete_case_report"
        return case_validation_passed(locate_case_process_path(item, qa_summary_path))

    return False, f"unsupported_type:{question_type}"


def clean_item(item: Dict) -> Dict:
    cleaned = dict(item)
    cleaned.pop("output_path", None)
    cleaned.pop("question_type", None)
    cleaned.pop("Type", None)
    cleaned.pop("status", None)
    cleaned.pop("error", None)
    return cleaned


def build_benchmark_payload(qa_summary_path: Path) -> Tuple[Dict, Dict]:
    summary = read_json(qa_summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Expected qa_run_summary.json to be a JSON object: {qa_summary_path}")

    benchmark = {
        "MCQ": [],
        "SAQ": [],
        "CBQ": [],
    }
    skipped: List[Dict] = []

    for category, records in summary.items():
        if not isinstance(records, list):
            continue
        for index, item in enumerate(records):
            passed, reason = record_is_complete(category, item, qa_summary_path)
            if not passed:
                skipped.append(
                    {
                        "category": category,
                        "index": index,
                        "reason": reason,
                        "source": item.get("from") if isinstance(item, dict) else "",
                    }
                )
                continue

            question_type = infer_question_type(category, item)
            category_name = output_category(question_type)
            if not category_name:
                skipped.append(
                    {
                        "category": category,
                        "index": index,
                        "reason": f"unsupported_type:{question_type}",
                        "source": item.get("from") if isinstance(item, dict) else "",
                    }
                )
                continue

            cleaned = clean_item(item)
            benchmark[category_name].append(cleaned)

    stats = {
        "total_kept": sum(len(items) for items in benchmark.values()),
        "total_skipped": len(skipped),
        "by_question_type": {
            "MCQ": len(benchmark["MCQ"]),
            "SAQ": len(benchmark["SAQ"]),
            "CBQ": len(benchmark["CBQ"]),
        },
        "skipped_reasons": {},
    }
    for skipped_item in skipped:
        reason = skipped_item["reason"]
        stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1

    return benchmark, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean benchmark JSON from qa_run_summary.json.")
    parser.add_argument(
        "--qa-summary",
        type=str,
        default=str(DEFAULT_QA_SUMMARY_PATH),
        help="Path to qa_run_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the benchmark JSON file will be saved.",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="GlobalDentBench",
        help="Benchmark name or output JSON filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_summary_path = resolve_project_path(args.qa_summary)
    output_dir = resolve_project_path(args.output_dir)
    benchmark_filename = normalize_benchmark_filename(args.benchmark_name)
    output_path = output_dir / benchmark_filename

    if not qa_summary_path.exists():
        raise FileNotFoundError(f"qa_run_summary.json not found: {qa_summary_path}")

    payload, stats = build_benchmark_payload(qa_summary_path=qa_summary_path)
    write_json(output_path, payload)

    print(f"Benchmark saved: {output_path}")
    print(f"Kept: {stats['total_kept']}")
    print(f"Skipped: {stats['total_skipped']}")
    print(f"By type: {stats['by_question_type']}")


if __name__ == "__main__":
    main()


# Run from the project root:
# python construction_pipeline/03-BuildBenchmark.py --qa-summary 00-Data_and_Models/construction_pipeline_outputs/02_qa_outputs/qa_run_summary.json --output-dir 00-Data_and_Models/benchmarks --benchmark-name GlobalDentBench
