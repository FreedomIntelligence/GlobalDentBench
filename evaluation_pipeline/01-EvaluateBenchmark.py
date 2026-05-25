# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Concurrently evaluate MCQ, SAQ, and CBQ benchmark items with configured LLM models, save results every N completed tasks, and support breakpoint resume.

import argparse
import copy
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_api import call_gpt


DEFAULT_BENCHMARK_PATH = "00-Data_and_Models/benchmarks/GlobalDentBench_newL1-L3-bench-small.json"
DEFAULT_OUTPUT_PATH = "00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json"
DEFAULT_PROMPT_PATH = "data/evaluation_pipeline/prompt.yaml"
DEFAULT_MODEL_CONFIGS = "llm"
DEFAULT_JUDGE_CONFIG = "llm"


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
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model config name is required.")
    return models


def ensure_benchmark_shape(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark JSON must be a top-level object.")
    normalized = {}
    for category in ("MCQ", "SAQ", "CBQ"):
        records = payload.get(category, [])
        if not isinstance(records, list):
            raise ValueError(f"`{category}` must be a list.")
        normalized[category] = records
    return normalized


def initialize_output_payload(benchmark_path: Path, output_path: Path, force_reprocess: bool) -> Dict:
    if output_path.exists() and not force_reprocess:
        return ensure_benchmark_shape(read_json(output_path))
    return ensure_benchmark_shape(read_json(benchmark_path))


def format_options(options) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        return "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(options))
    return str(options or "")


def extract_option_letter(response) -> str:
    text = str(response or "").strip()
    if not text:
        return ""
    match = re.search(r"\b([A-Z])\b", text.upper())
    return match.group(1) if match else text[:1].upper()


def normalize_judge_response(response, fallback_reason: str = "") -> Dict:
    if isinstance(response, dict):
        return response
    return {
        "score": 0,
        "reason": fallback_reason or f"Judge did not return JSON: {str(response)[:300]}",
    }


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


def evaluate_mcq(item: Dict, model_config: str, prompts: Dict, timeout: int) -> str:
    prompt_template = prompts["mcq_evaluation_prompts"]["test_llm"]["user_template"]
    system_prompt = prompts["mcq_evaluation_prompts"]["test_llm"].get("system_prompt")
    prompt = prompt_template.format(
        question=item.get("question", ""),
        options=format_options(item.get("options")),
    )
    response = call_gpt(
        prompt=prompt,
        config_type=model_config,
        system_prompt=system_prompt,
        json_output=False,
        timeout=timeout,
    )
    return extract_option_letter(response)


def evaluate_saq(item: Dict, model_config: str, judge_config: str, prompts: Dict, timeout: int) -> Dict:
    test_cfg = prompts["saq_evaluation_prompts"]["test_llm"]
    judge_cfg = prompts["saq_evaluation_prompts"]["judge_llm"]
    test_prompt = test_cfg["user_template"].format(
        cleaned_text=item.get("question", "")
    )
    answer = call_gpt(
        prompt=test_prompt,
        config_type=model_config,
        system_prompt=test_cfg.get("system_prompt"),
        json_output=False,
        timeout=timeout,
    )
    judge_prompt = judge_cfg["user_template"].format(
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        model_answer=answer,
    )
    judge = call_gpt(
        prompt=judge_prompt,
        config_type=judge_config,
        json_output=True,
        system_prompt=judge_cfg.get("system_prompt"),
        timeout=timeout,
    )
    return {
        "answer": str(answer or ""),
        "judge": normalize_judge_response(judge),
    }


def evaluate_cbq(item: Dict, model_config: str, judge_config: str, prompts: Dict, timeout: int) -> Dict:
    question_text = item.get("seed_question", {}).get("question", "")
    test_cfg = prompts["casequestion_evaluation_prompts"]["test_llm"]
    judge_cfg = prompts["casequestion_evaluation_prompts"]["judge_llm"]
    test_prompt = test_cfg["user_template"].format(
        question_text=question_text
    )
    answer = call_gpt(
        prompt=test_prompt,
        config_type=model_config,
        system_prompt=test_cfg.get("system_prompt"),
        json_output=False,
        timeout=timeout,
    )
    judge_prompt = judge_cfg["user_template"].format(
        question_text=question_text,
        kp_block=build_key_point_block(item.get("key_points", [])),
        model_answer=answer,
    )
    judge = call_gpt(
        prompt=judge_prompt,
        config_type=judge_config,
        json_output=True,
        system_prompt=judge_cfg.get("system_prompt"),
        timeout=timeout,
    )
    return {
        "answer": str(answer or ""),
        "judge": normalize_judge_response(judge),
    }


def evaluate_one(category: str, item: Dict, model_config: str, judge_config: str, prompts: Dict, timeout: int):
    if category == "MCQ":
        return evaluate_mcq(item, model_config, prompts, timeout)
    if category == "SAQ":
        return evaluate_saq(item, model_config, judge_config, prompts, timeout)
    if category == "CBQ":
        return evaluate_cbq(item, model_config, judge_config, prompts, timeout)
    raise ValueError(f"Unsupported category: {category}")


def build_pending_tasks(payload: Dict, model_configs: List[str], force_reprocess: bool) -> List[Tuple[str, int, str]]:
    tasks = []
    for category in ("MCQ", "SAQ", "CBQ"):
        for index, item in enumerate(payload[category]):
            if not isinstance(item, dict):
                continue
            models = item.setdefault("models", {})
            for model_config in model_configs:
                if force_reprocess or model_config not in models:
                    tasks.append((category, index, model_config))
    return tasks


def run_evaluation(args: argparse.Namespace) -> None:
    benchmark_path = resolve_project_path(args.benchmark)
    output_path = resolve_project_path(args.output)
    prompt_path = resolve_project_path(args.prompt)
    model_configs = parse_model_configs(args.models)
    prompts = load_prompts(prompt_path)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    output_payload = initialize_output_payload(benchmark_path, output_path, args.force_reprocess)
    pending_tasks = build_pending_tasks(output_payload, model_configs, args.force_reprocess)
    if not pending_tasks:
        write_json_atomic(output_path, output_payload)
        print(f"No pending tasks. Output is up to date: {output_path}")
        return

    completed_since_save = 0
    print(f"Benchmark: {benchmark_path}")
    print(f"Output: {output_path}")
    print(f"Models: {model_configs}")
    print(f"Judge config: {args.judge_config}")
    print(f"Pending model-item tasks: {len(pending_tasks)}")

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_task = {
            executor.submit(
                evaluate_one,
                category,
                copy.deepcopy(output_payload[category][index]),
                model_config,
                args.judge_config,
                prompts,
                args.timeout,
            ): (category, index, model_config)
            for category, index, model_config in pending_tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating", dynamic_ncols=True):
            category, index, model_config = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "error": str(exc),
                }

            with save_lock:
                output_payload[category][index].setdefault("models", {})[model_config] = result
                completed_since_save += 1
                if completed_since_save >= args.save_every:
                    write_json_atomic(output_path, output_payload)
                    completed_since_save = 0

    write_json_atomic(output_path, output_payload)
    print(f"Saved final output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GlobalDentBench benchmark with multiple configured LLMs.")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK_PATH, help="Input benchmark JSON path.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output JSON path with model results.")
    parser.add_argument("--models", type=str, default=DEFAULT_MODEL_CONFIGS, help="Comma-separated config names from config/config.json.")
    parser.add_argument("--judge-config", type=str, default=DEFAULT_JUDGE_CONFIG, help="Config name used by judge LLM calls.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent model-item tasks.")
    parser.add_argument("--save-every", type=int, default=100, help="Save output after this many completed tasks.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each API call.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_PATH, help="Evaluation prompt file path.")
    parser.add_argument("--force-reprocess", action="store_true", help="Re-run all model-item tasks even if output already contains results.")
    return parser.parse_args()


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()


# Run from the project root:
# python evaluation_pipeline/01-EvaluateBenchmark.py --benchmark 00-Data_and_Models/benchmarks/GlobalDentBench_newL1-L3-bench-small.json --output 00-Data_and_Models/evaluation_outputs/GlobalDentBench_newL1-L3-evaluated.json --models llm,gpt-5.4-nano --concurrency 4
