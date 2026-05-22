# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Add S0/S1/S2 clinical risk labels to evaluated CBQ model answers with concurrent LLM judging and breakpoint resume.

import argparse
import copy
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_api import call_gpt


DEFAULT_BENCHMARK_PATH = "00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
DEFAULT_OUTPUT_PATH = "00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
DEFAULT_PROMPT_PATH = "data/evaluation_pipeline/prompt.yaml"
DEFAULT_JUDGE_CONFIG = "llm"
RISK_LEVELS = {"S0", "S1", "S2"}


save_lock = threading.Lock()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_prompts(prompt_path: Path) -> Dict:
    if prompt_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML prompt files. Install it with `pip install PyYAML`.") from exc
        payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt YAML must be a top-level object: {prompt_path}")
        return payload
    return read_json(prompt_path)


def parse_model_configs(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_benchmark_shape(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark JSON must be a top-level object.")
    payload.setdefault("CBQ", [])
    if not isinstance(payload["CBQ"], list):
        raise ValueError("`CBQ` must be a list.")
    return payload


def initialize_output_payload(benchmark_path: Path, output_path: Path, force_reprocess: bool) -> Dict:
    if output_path.exists() and output_path.resolve() != benchmark_path.resolve() and not force_reprocess:
        return ensure_benchmark_shape(read_json(output_path))
    return ensure_benchmark_shape(read_json(benchmark_path))


def discover_cbq_models(payload: Dict) -> List[str]:
    models = set()
    for item in payload.get("CBQ", []):
        if isinstance(item, dict) and isinstance(item.get("models"), dict):
            models.update(str(model_name) for model_name in item["models"].keys())
    return sorted(models)


def build_key_point_block(key_points: List[Dict]) -> str:
    lines = []
    for index, point in enumerate(key_points, start=1):
        if isinstance(point, dict):
            content = point.get("content", "")
            location = point.get("location", "")
            explanation = point.get("explanation", "")
            lines.append(f"{index}. {content}\nLocation: {location}\nExplanation: {explanation}")
        else:
            lines.append(f"{index}. {point}")
    return "\n\n".join(lines)


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
    level_match = re.search(r"\bS([012])\b", text)
    if level_match:
        return f"S{level_match.group(1)}"
    numeric_match = re.search(r"\b([123])\b", text)
    if numeric_match:
        return {"1": "S0", "2": "S1", "3": "S2"}[numeric_match.group(1)]
    return ""


def normalize_risk_response(response) -> Dict:
    if not isinstance(response, dict):
        level = normalize_risk_level(response)
        return {
            "risk_level": level,
            "reason": str(response or "")[:500],
            "raw": response,
        }

    level = normalize_risk_level(response.get("risk_level") or response.get("level") or response.get("raw_content"))
    return {
        "risk_level": level,
        "reason": str(response.get("rationale") or response.get("reason") or ""),
        "raw": response,
    }


def evaluate_cbq_risk(item: Dict, model_result: Dict, judge_config: str, prompts: Dict, timeout: int) -> Dict:
    prompt_config = prompts["risk_analysis_prompts"]["cbq_risk_judge"]
    case_context = item.get("seed_question", {}).get("question", "")
    gold_answer = build_key_point_block(item.get("key_points", []))
    answer = model_result.get("answer", "") if isinstance(model_result, dict) else ""
    prompt = prompt_config["user_template"].format(
        rubric=prompt_config["rubric"],
        case_context=case_context,
        gold_answer=gold_answer,
        answer=answer,
    )
    response = call_gpt(
        prompt=prompt,
        config_type=judge_config,
        system_prompt=prompt_config.get("system_prompt"),
        json_output=True,
        timeout=timeout,
    )
    risk = normalize_risk_response(response)
    if risk["risk_level"] not in RISK_LEVELS:
        risk["status"] = "failed"
        risk["error"] = f"Invalid risk level from judge: {str(response)[:500]}"
    else:
        risk["status"] = "completed"
    risk["judge_config"] = judge_config
    risk["tagged_at"] = datetime.now(timezone.utc).isoformat()
    return risk


def build_pending_tasks(payload: Dict, model_configs: List[str], force_reprocess: bool) -> List[Tuple[int, str]]:
    tasks = []
    for index, item in enumerate(payload.get("CBQ", [])):
        if not isinstance(item, dict) or not isinstance(item.get("models"), dict):
            continue
        for model_config in model_configs:
            model_result = item["models"].get(model_config)
            if not isinstance(model_result, dict) or not str(model_result.get("answer", "")).strip():
                continue
            existing_risk = model_result.get("risk")
            if force_reprocess or not (isinstance(existing_risk, dict) and existing_risk.get("risk_level") in RISK_LEVELS):
                tasks.append((index, model_config))
    return tasks


def run_risk_analysis(args: argparse.Namespace) -> None:
    benchmark_path = resolve_project_path(args.benchmark)
    output_path = resolve_project_path(args.output)
    prompt_path = resolve_project_path(args.prompt)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Evaluated benchmark file not found: {benchmark_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts = load_prompts(prompt_path)
    output_payload = initialize_output_payload(benchmark_path, output_path, args.force_reprocess)
    model_configs = parse_model_configs(args.models) or discover_cbq_models(output_payload)
    if not model_configs:
        raise ValueError("No CBQ model results were found. Run evaluation first or set --models.")

    pending_tasks = build_pending_tasks(output_payload, model_configs, args.force_reprocess)
    if not pending_tasks:
        write_json_atomic(output_path, output_payload)
        print(f"No pending CBQ risk tasks. Output is up to date: {output_path}")
        return

    completed_since_save = 0
    print(f"Evaluated benchmark: {benchmark_path}")
    print(f"Output: {output_path}")
    print(f"Models: {model_configs}")
    print(f"Judge config: {args.judge_config}")
    print(f"Pending CBQ model-answer risk tasks: {len(pending_tasks)}")

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_task = {
            executor.submit(
                evaluate_cbq_risk,
                copy.deepcopy(output_payload["CBQ"][index]),
                copy.deepcopy(output_payload["CBQ"][index]["models"][model_config]),
                args.judge_config,
                prompts,
                args.timeout,
            ): (index, model_config)
            for index, model_config in pending_tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Risk analysis", dynamic_ncols=True):
            index, model_config = future_to_task[future]
            try:
                risk = future.result()
            except Exception as exc:
                risk = {
                    "status": "failed",
                    "risk_level": "",
                    "reason": "",
                    "error": str(exc),
                    "judge_config": args.judge_config,
                    "tagged_at": datetime.now(timezone.utc).isoformat(),
                }

            with save_lock:
                output_payload["CBQ"][index]["models"][model_config]["risk"] = risk
                completed_since_save += 1
                if args.save_every and completed_since_save >= args.save_every:
                    write_json_atomic(output_path, output_payload)
                    completed_since_save = 0

    write_json_atomic(output_path, output_payload)
    print(f"Saved final output with risk labels: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add S0/S1/S2 risk labels to evaluated CBQ model answers.")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK_PATH, help="Evaluated benchmark JSON path.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output JSON path. May be the same as --benchmark.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model config names. Empty means all evaluated CBQ models.")
    parser.add_argument("--judge-config", type=str, default=DEFAULT_JUDGE_CONFIG, help="Config name used by risk judge LLM calls.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent risk judge tasks.")
    parser.add_argument("--save-every", type=int, default=100, help="Save output after this many completed tasks.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each API call.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_PATH, help="Evaluation prompt file path.")
    parser.add_argument("--force-reprocess", action="store_true", help="Re-run all CBQ risk tasks even if output already contains risk labels.")
    return parser.parse_args()


def main() -> None:
    run_risk_analysis(parse_args())


if __name__ == "__main__":
    main()


# Run from the project root:
# python evaluation_pipeline/03-RiskAnalysis.py --benchmark 00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json --output 00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json --judge-config llm --concurrency 4
