# Created: 2026-02-01
# Modified: 2026-05-22
# Purpose: Provide a minimal reusable API wrapper for chat-style LLM and VLM requests, with environment-variable overrides for open-source use.

import base64
import json
import os
import re
from typing import Any, Dict, Iterable, Optional

import requests

from utils import get_model_config
from utils.image_process import get_image


DEFAULT_IMAGE_SIZE_MB = 0.3
DEFAULT_TIMEOUT = 300


def normalize_api_url(config: Dict[str, Any]) -> str:
    if config.get("url"):
        return config["url"].rstrip("/")

    base_url = config.get("base_url", "").rstrip("/")
    if not base_url:
        return ""

    # Check if using responses API
    api_type = config.get("api_type", "chat")
    if api_type == "responses":
        if base_url.endswith("/responses"):
            return base_url
        return f"{base_url}/responses"

    # Default to chat/completions API
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def normalize_image_paths(image_paths) -> Iterable[str]:
    if image_paths is None:
        return []
    if isinstance(image_paths, str):
        return [image_paths]
    return image_paths


def build_user_content(prompt: str, image_paths=None, max_image_size_mb: float = DEFAULT_IMAGE_SIZE_MB):
    content = [{"type": "text", "text": prompt}]

    for image_path in normalize_image_paths(image_paths):
        if not image_path or not os.path.exists(image_path):
            continue

        image_buffer, image_type, _ = get_image(image_path, max_size_mb=max_image_size_mb)
        if image_buffer is None:
            continue

        image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_type};base64,{image_b64}"},
            }
        )

    return content


def build_payload(
    model: str,
    prompt: str,
    image_paths=None,
    system_prompt: Optional[str] = None,
    json_output: bool = False,
    stream: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    api_type: str = "chat",
    only_max_tokens: bool = False,
) -> Dict[str, Any]:
    # responses API uses different format
    if api_type == "responses":
        payload: Dict[str, Any] = {
            "model": model,
            "input": prompt,
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if system_prompt:
            payload["instructions"] = system_prompt
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if extra_body:
            payload.update(extra_body)
        return payload

    # Standard chat/completions API
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_content(prompt, image_paths)})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
        if not only_max_tokens:
            payload["max_completion_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    if stream:
        payload["stream"] = True
    if extra_body:
        payload.update(extra_body)

    return payload


class GPTClient:
    def __init__(self, url: str, api_key: str, model_name: str, timeout: int = DEFAULT_TIMEOUT, api_type: str = "chat"):
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.api_type = api_type

    def call(
        self,
        prompt,
        image_paths=None,
        json_output=False,
        stream=False,
        return_usage=False,
        system_prompt=None,
        max_tokens=None,
        temperature=None,
        reasoning_effort=None,
        extra_body=None,
        only_max_tokens=False,
    ):
        payload = build_payload(
            model=self.model_name,
            prompt=prompt,
            image_paths=image_paths,
            system_prompt=system_prompt,
            json_output=json_output,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            api_type=self.api_type,
            only_max_tokens=only_max_tokens,
        )

        try:
            response = requests.post(
                self.url,
                headers=build_headers(self.api_key),
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return self._generate_stream(response)

            response_json = response.json()
            content = self._extract_content(response_json)
            if json_output and isinstance(content, str):
                content = self._parse_json(content)

            if return_usage:
                return content, self._extract_usage(response_json)
            return content
        except Exception as exc:
            if stream:
                def error_generator():
                    yield f"Request failed: {exc}"

                return error_generator()

            error_message = f"Request failed: {exc}"
            if return_usage:
                return error_message, {"input_tokens": 0, "output_tokens": 0}
            return error_message

    def _generate_stream(self, response):
        try:
            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8")
                if not decoded_line.startswith("data: "):
                    continue

                data_str = decoded_line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data_json = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                content = self._extract_delta_content(data_json)
                if content:
                    yield content
        except Exception as exc:
            yield f"[Stream Error: {exc}]"

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]):
        # Prefer the output field from the responses API
        if "output" in response_json:
            output = response_json["output"]
            if isinstance(output, list) and output:
                # Iterate over output items and find type='message'
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        content_list = item.get("content", [])
                        if isinstance(content_list, list):
                            for content_item in content_list:
                                if isinstance(content_item, dict) and content_item.get("type") in ("output_text", "text"):
                                    text = content_item.get("text", "")
                                    if text:
                                        return text
            elif isinstance(output, str):
                return output

        # Standard chat/completions API
        choices = response_json.get("choices", [])
        if not choices:
            return f"Error: No choices in response. Raw: {response_json}"

        content = choices[0].get("message", {}).get("content", "")

        # If content is None or empty, try extracting it from alternate fields
        if content is None or content == "":
            if "text" in choices[0].get("message", {}):
                content = choices[0]["message"]["text"]

        return content if content else ""

    @staticmethod
    def _extract_delta_content(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("delta", {}).get("content", "")

    @staticmethod
    def _extract_usage(response_json: Dict[str, Any]) -> Dict[str, int]:
        usage = response_json.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    @staticmethod
    def _parse_json(content: str):
        try:
            clean_content = re.sub(
                r"^```json\s*|\s*```$",
                "",
                content.strip(),
                flags=re.DOTALL,
            )
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return {"error": "JSONDecodeError", "raw_content": content}


def call_gpt(
    prompt,
    model=None,
    image_paths=None,
    json_output=False,
    stream=False,
    config_type="llm",
    return_usage=False,
    system_prompt=None,
    max_tokens=None,
    temperature=None,
    reasoning_effort=None,
    timeout=DEFAULT_TIMEOUT,
    extra_body=None,
    only_max_tokens=False,
):
    config = get_model_config(config_type)
    api_key = (
        config.get("api_key", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("PRIMARY_API_KEY", "")
        or os.getenv("LLM_API_KEY", "")
    )
    runtime_config = {
        **config,
        "url": os.getenv("LLM_API_URL", "") or config.get("url", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "") or os.getenv("LLM_BASE_URL", "") or config.get("base_url", ""),
    }
    url = normalize_api_url(runtime_config)
    model_name = model or os.getenv("OPENAI_MODEL", "") or os.getenv("LLM_MODEL", "") or config.get("model", "") or config_type
    api_type = config.get("api_type", "chat")

    if not model_name:
        raise ValueError(
            f"Model is not configured. Please add `model` under `{config_type}` in config.json."
        )
    if not api_key:
        raise ValueError(
            f"API key is not configured. Please add `api_key` under `{config_type}` in config.json."
        )
    if not url:
        raise ValueError(
            f"URL is not configured. Please add `url` or `base_url` under `{config_type}` in config.json."
        )

    if max_tokens is None:
        max_tokens = config.get("max_tokens")
    if temperature is None:
        temperature = config.get("temperature")
    if reasoning_effort is None:
        reasoning_effort = config.get("reasoning_effort")
    config_extra_body = config.get("extra_body")
    if config_extra_body:
        if extra_body:
            extra_body = {**config_extra_body, **extra_body}
        else:
            extra_body = config_extra_body

    client = GPTClient(
        url=url,
        api_key=api_key,
        model_name=model_name,
        timeout=timeout,
        api_type=api_type,
    )
    return client.call(
        prompt=prompt,
        image_paths=image_paths,
        json_output=json_output,
        stream=stream,
        return_usage=return_usage,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        only_max_tokens=only_max_tokens,
    )
