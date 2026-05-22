# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Analyze evaluated GlobalDentBench results by question type, reasoning level, and dental discipline.

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BENCHMARK_PATH = "00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
DEFAULT_OUTPUT_DIR = "00-Data_and_Models/result_analysis"
DEFAULT_REPORT_NAME = "GlobalDentBench_result_analysis"
CATEGORIES = ("MCQ", "SAQ", "CBQ")
LEVELS = ("L1", "L2", "L3")
RISK_LEVELS = ("S0", "S1", "S2")


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_model_configs(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_models(benchmark: Dict) -> List[str]:
    models = set()
    for category in CATEGORIES:
        for item in benchmark.get(category, []):
            if isinstance(item, dict) and isinstance(item.get("models"), dict):
                models.update(str(model_name) for model_name in item["models"].keys())
    return sorted(models)


def normalize_option_letter(value) -> str:
    if isinstance(value, dict):
        if value.get("status") == "failed":
            return ""
        value = value.get("answer", "")
    text = str(value or "").strip().upper()
    if not text:
        return ""
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else text[:1]


def is_positive_judge_score(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "correct"}
    return False


def is_hit(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "hit", "correct"}
    return False


def score_mcq(item: Dict, model_result) -> float:
    reference = normalize_option_letter(item.get("answer"))
    prediction = normalize_option_letter(model_result)
    return 100.0 if reference and prediction and reference == prediction else 0.0


def score_saq(model_result) -> float:
    if not isinstance(model_result, dict):
        return 0.0
    judge = model_result.get("judge")
    if not isinstance(judge, dict):
        return 0.0
    return 100.0 if is_positive_judge_score(judge.get("score")) else 0.0


def score_cbq(model_result) -> float:
    if not isinstance(model_result, dict):
        return 0.0
    judge = model_result.get("judge")
    if not isinstance(judge, dict):
        return 0.0
    key_points = judge.get("key_points")
    if isinstance(key_points, list):
        return min(100.0, sum(20.0 for point in key_points if isinstance(point, dict) and is_hit(point.get("hit"))))
    if "score" in judge:
        score = judge.get("score")
        if isinstance(score, (int, float)):
            return float(score * 100 if 0 <= score <= 1 else score)
    return 0.0


def score_item(category: str, item: Dict, model_result) -> float:
    if category == "MCQ":
        return score_mcq(item, model_result)
    if category == "SAQ":
        return score_saq(model_result)
    if category == "CBQ":
        return score_cbq(model_result)
    raise ValueError(f"Unsupported category: {category}")


def get_level(item: Dict) -> str:
    tags = item.get("tags") if isinstance(item, dict) else {}
    level = (tags.get("reasoning_level") or tags.get("capability_level")) if isinstance(tags, dict) else None
    return level if level in LEVELS else "Unknown"


def get_dental_discipline(item: Dict) -> Tuple[str, str]:
    tags = item.get("tags") if isinstance(item, dict) else {}
    discipline = (tags.get("dental_discipline") or tags.get("taxonomy")) if isinstance(tags, dict) else {}
    if not isinstance(discipline, dict):
        return "Unknown", "Unknown"
    number = discipline.get("number")
    name = discipline.get("name") or "Unknown"
    label = f"{number}. {name}" if number else name
    return str(number or "Unknown"), label


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def round_optional(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


def normalize_risk_level(value) -> str:
    text = str(value or "").strip().upper()
    if text in RISK_LEVELS:
        return text
    if text == "0":
        return "S0"
    if text == "1":
        return "S0"
    if text == "2":
        return "S1"
    if text == "3":
        return "S2"
    match = re.search(r"\bS([012])\b", text)
    if match:
        return f"S{match.group(1)}"
    legacy_match = re.search(r"\b([123])\b", text)
    if legacy_match:
        return {"1": "S0", "2": "S1", "3": "S2"}[legacy_match.group(1)]
    return ""


def collect_score_records(benchmark: Dict, model_configs: List[str]) -> List[Dict]:
    records = []
    for category in CATEGORIES:
        for item_index, item in enumerate(benchmark.get(category, [])):
            if not isinstance(item, dict) or not isinstance(item.get("models"), dict):
                continue
            level = get_level(item)
            discipline_number, discipline_label = get_dental_discipline(item)
            for model_config in model_configs:
                if model_config not in item["models"]:
                    continue
                records.append(
                    {
                        "model": model_config,
                        "category": category,
                        "item_index": item_index,
                        "score": score_item(category, item, item["models"][model_config]),
                        "reasoning_level": level,
                        "dental_discipline_number": discipline_number,
                        "dental_discipline": discipline_label,
                    }
                )
    return records


def collect_risk_records(benchmark: Dict, model_configs: List[str]) -> List[Dict]:
    records = []
    for item_index, item in enumerate(benchmark.get("CBQ", [])):
        if not isinstance(item, dict) or not isinstance(item.get("models"), dict):
            continue
        for model_config in model_configs:
            model_result = item["models"].get(model_config)
            if not isinstance(model_result, dict):
                continue
            if not str(model_result.get("answer", "")).strip():
                continue
            risk = model_result.get("risk")
            risk_level = ""
            if not isinstance(risk, dict):
                risk = {}
            else:
                risk_level = normalize_risk_level(risk.get("risk_level") or risk.get("level"))
            records.append(
                {
                    "model": model_config,
                    "item_index": item_index,
                    "risk_level": risk_level if risk_level in RISK_LEVELS else "",
                }
            )
    return records


def make_score_row(group_keys: Dict, records: List[Dict], digits: int) -> Dict:
    row = dict(group_keys)
    category_means = []
    total_scores = []
    total_count = 0

    for category in CATEGORIES:
        scores = [record["score"] for record in records if record["category"] == category]
        category_mean = mean(scores)
        if category_mean is not None:
            category_means.append(category_mean)
        total_scores.extend(scores)
        total_count += len(scores)
        row[f"{category}_score"] = round_optional(category_mean, digits)
        row[f"{category}_n"] = len(scores)

    row["macro_average_score"] = round_optional(mean(category_means), digits)
    row["micro_average_score"] = round_optional(mean(total_scores), digits)
    row["n"] = total_count
    return row


def group_records(records: List[Dict], group_fields: Tuple[str, ...], digits: int) -> List[Dict]:
    grouped = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in group_fields)
        grouped[key].append(record)

    rows = []
    for key, group in grouped.items():
        group_keys = {field: value for field, value in zip(group_fields, key)}
        rows.append(make_score_row(group_keys, group, digits))
    return rows


def build_risk_distribution_rows(records: List[Dict], digits: int) -> List[Dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    rows = []
    for model, group in grouped.items():
        total = len(group)
        scored = sum(1 for record in group if record["risk_level"] in RISK_LEVELS)
        if scored == 0:
            continue
        row = {"model": model, "risk_scored_n": scored, "cbq_response_n": total}
        for level in RISK_LEVELS:
            count = sum(1 for record in group if record["risk_level"] == level)
            row[f"{level}_n"] = count
            row[f"{level}_percent"] = round((count / total) * 100, digits) if total else 0.0
        rows.append(row)
    return sort_rows(rows, ("model",))


def sort_value(row: Dict, field: str):
    value = row.get(field, "")
    if field == "dental_discipline_number":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 999
    return str(value)


def sort_rows(rows: List[Dict], fields: Tuple[str, ...]) -> List[Dict]:
    return sorted(rows, key=lambda row: tuple(sort_value(row, field) for field in fields))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fieldnames})


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_table(rows: List[Dict], fieldnames: List[str]) -> str:
    if not rows:
        return "_No records._"
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(format_value(row.get(field, "")) for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def truncate_text(value: str, max_width: int) -> str:
    if len(value) <= max_width:
        return value
    if max_width <= 3:
        return value[:max_width]
    return value[: max_width - 3] + "..."


def plain_text_table(rows: List[Dict], columns: List[Tuple[str, str]], max_widths: Optional[Dict[str, int]] = None) -> str:
    if not rows:
        return "No records."

    max_widths = max_widths or {}
    widths = {}
    for field, label in columns:
        values = [format_value(row.get(field, "")) for row in rows]
        width = max([len(label), *(len(value) for value in values)])
        widths[field] = min(width, max_widths.get(field, width))

    def is_numeric_field(field: str) -> bool:
        return field == "n" or field.endswith("_n") or "score" in field or "average" in field

    def format_cell(field: str, value: str) -> str:
        value = truncate_text(value, widths[field])
        return value.rjust(widths[field]) if is_numeric_field(field) else value.ljust(widths[field])

    header = " | ".join(format_cell(field, label) for field, label in columns)
    separator = "-+-".join("-" * widths[field] for field, _ in columns)
    body = [
        " | ".join(format_cell(field, format_value(row.get(field, ""))) for field, _ in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def build_report_payload(records: List[Dict], risk_records: List[Dict], digits: int) -> Dict:
    overall_rows = sort_rows(group_records(records, ("model",), digits), ("model",))
    level_rows = [
        row
        for row in sort_rows(group_records(records, ("model", "reasoning_level"), digits), ("model", "reasoning_level"))
        if row.get("reasoning_level") in {*LEVELS, "Unknown"}
    ]
    discipline_rows = sort_rows(
        group_records(records, ("model", "dental_discipline_number", "dental_discipline"), digits),
        ("model", "dental_discipline_number", "dental_discipline"),
    )
    payload = {
        "overall_by_question_type": overall_rows,
        "by_reasoning_level": level_rows,
        "by_dental_discipline": discipline_rows,
    }
    risk_rows = build_risk_distribution_rows(risk_records, digits) if risk_records else []
    if risk_rows:
        payload["risk_distribution"] = risk_rows
    return payload


def write_markdown_report(path: Path, benchmark_path: Path, model_configs: List[str], payload: Dict) -> None:
    overall_fields = [
        "model",
        "MCQ_score",
        "MCQ_n",
        "SAQ_score",
        "SAQ_n",
        "CBQ_score",
        "CBQ_n",
        "macro_average_score",
        "micro_average_score",
        "n",
    ]
    level_fields = [
        "model",
        "reasoning_level",
        "MCQ_score",
        "MCQ_n",
        "SAQ_score",
        "SAQ_n",
        "CBQ_score",
        "CBQ_n",
        "macro_average_score",
        "n",
    ]
    discipline_fields = [
        "model",
        "dental_discipline_number",
        "dental_discipline",
        "MCQ_score",
        "MCQ_n",
        "SAQ_score",
        "SAQ_n",
        "CBQ_score",
        "CBQ_n",
        "macro_average_score",
        "n",
    ]

    content = [
        "# GlobalDentBench Result Analysis",
        "",
        f"Benchmark: `{benchmark_path}`",
        f"Models: `{', '.join(model_configs)}`",
        "",
        "## Overall By Question Type",
        "",
        markdown_table(payload["overall_by_question_type"], overall_fields),
        "",
        "## By Reasoning Level",
        "",
        markdown_table(payload["by_reasoning_level"], level_fields),
        "",
        "## By Dental Discipline",
        "",
        markdown_table(payload["by_dental_discipline"], discipline_fields),
        "",
    ]
    if payload.get("risk_distribution"):
        risk_fields = [
            "model",
            "S0_percent",
            "S0_n",
            "S1_percent",
            "S1_n",
            "S2_percent",
            "S2_n",
            "risk_scored_n",
            "cbq_response_n",
        ]
        content.extend(
            [
                "## CBQ Risk Distribution",
                "",
                markdown_table(payload["risk_distribution"], risk_fields),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def print_console_summary(payload: Dict) -> None:
    columns = [
        ("model", "Model"),
        ("MCQ_score", "MCQ"),
        ("SAQ_score", "SAQ"),
        ("CBQ_score", "CBQ"),
        ("macro_average_score", "Macro"),
        ("micro_average_score", "Micro"),
        ("n", "N"),
    ]
    print("\nOverall by question type:")
    print(plain_text_table(payload["overall_by_question_type"], columns))

    level_columns = [
        ("model", "Model"),
        ("reasoning_level", "Reasoning"),
        ("MCQ_score", "MCQ"),
        ("SAQ_score", "SAQ"),
        ("CBQ_score", "CBQ"),
        ("macro_average_score", "Macro"),
        ("n", "N"),
    ]
    print("\nBy reasoning level:")
    print(plain_text_table(payload["by_reasoning_level"], level_columns))

    discipline_columns = [
        ("model", "Model"),
        ("dental_discipline", "Dental Discipline"),
        ("MCQ_score", "MCQ"),
        ("SAQ_score", "SAQ"),
        ("CBQ_score", "CBQ"),
        ("macro_average_score", "Macro"),
        ("n", "N"),
    ]
    print("\nBy dental discipline:")
    print(plain_text_table(payload["by_dental_discipline"], discipline_columns, max_widths={"dental_discipline": 44}))

    if payload.get("risk_distribution"):
        risk_columns = [
            ("model", "Model"),
            ("S0_percent", "S0 %"),
            ("S0_n", "S0 N"),
            ("S1_percent", "S1 %"),
            ("S1_n", "S1 N"),
            ("S2_percent", "S2 %"),
            ("S2_n", "S2 N"),
            ("risk_scored_n", "Scored"),
            ("cbq_response_n", "Total"),
        ]
        print("\nCBQ risk distribution:")
        print(plain_text_table(payload["risk_distribution"], risk_columns))


def run_analysis(args: argparse.Namespace) -> None:
    benchmark_path = resolve_project_path(args.benchmark)
    output_dir = resolve_project_path(args.output_dir)
    benchmark = read_json(benchmark_path)
    discovered_models = discover_models(benchmark)
    model_configs = parse_model_configs(args.models) or discovered_models
    if not model_configs:
        raise ValueError("No evaluated models were found. Run evaluation first or set --models.")

    records = collect_score_records(benchmark, model_configs)
    if not records:
        raise ValueError("No model results matched the requested model list.")

    risk_records = collect_risk_records(benchmark, model_configs)
    payload = build_report_payload(records, risk_records, args.digits)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.report_name}.json"
    markdown_path = output_dir / f"{args.report_name}.md"
    overall_csv_path = output_dir / f"{args.report_name}_overall_by_question_type.csv"
    level_csv_path = output_dir / f"{args.report_name}_by_reasoning_level.csv"
    discipline_csv_path = output_dir / f"{args.report_name}_by_dental_discipline.csv"
    risk_csv_path = output_dir / f"{args.report_name}_cbq_risk_distribution.csv"

    write_json(
        json_path,
        {
            "benchmark_path": str(benchmark_path),
            "models": model_configs,
            **payload,
        },
    )
    write_markdown_report(markdown_path, benchmark_path, model_configs, payload)
    write_csv(
        overall_csv_path,
        payload["overall_by_question_type"],
        ["model", "MCQ_score", "MCQ_n", "SAQ_score", "SAQ_n", "CBQ_score", "CBQ_n", "macro_average_score", "micro_average_score", "n"],
    )
    write_csv(
        level_csv_path,
        payload["by_reasoning_level"],
        ["model", "reasoning_level", "MCQ_score", "MCQ_n", "SAQ_score", "SAQ_n", "CBQ_score", "CBQ_n", "macro_average_score", "n"],
    )
    write_csv(
        discipline_csv_path,
        payload["by_dental_discipline"],
        ["model", "dental_discipline_number", "dental_discipline", "MCQ_score", "MCQ_n", "SAQ_score", "SAQ_n", "CBQ_score", "CBQ_n", "macro_average_score", "n"],
    )
    if payload.get("risk_distribution"):
        write_csv(
            risk_csv_path,
            payload["risk_distribution"],
            [
                "model",
                "S0_percent",
                "S0_n",
                "S1_percent",
                "S1_n",
                "S2_percent",
                "S2_n",
                "risk_scored_n",
                "cbq_response_n",
            ],
        )
    elif risk_csv_path.exists():
        risk_csv_path.unlink()

    print_console_summary(payload)
    print(f"\nSaved report files to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GlobalDentBench evaluated results.")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK_PATH, help="Evaluated benchmark JSON path.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for result analysis files.")
    parser.add_argument("--report-name", type=str, default=DEFAULT_REPORT_NAME, help="Base name for output report files.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model config names. Empty means all evaluated models.")
    parser.add_argument("--digits", type=int, default=2, help="Decimal places for reported scores.")
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()


# Run from the project root:
# python evaluation_pipeline/02-ResultAnalysis.py --benchmark 00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json --output-dir 00-Data_and_Models/result_analysis
