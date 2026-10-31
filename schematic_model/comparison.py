"""Answer-only comparisons through operator-configured private inference endpoints.

No model loading, switching, tools, retention, or browser-selected destinations.
The HTTP server owns serialization with its normal chat generation lock.
"""

import copy
import math
import os
import re
import time
from typing import Any

from .agent import private_endpoint, request_json
from .deployment import DeploymentConfig, validate_request
from .inference import InferenceError
from .metrics import InferenceMetrics, safe_model_details


class ComparisonError(InferenceError):
    """Content-free explanation safe for the comparison error panel."""


class ComparisonRuntime:
    def __init__(self, config, transport=request_json):
        if not isinstance(config, dict) or set(config) - {"profiles", "max_tokens"}:
            raise ValueError("Expected comparison profiles and optional max_tokens")
        self.max_tokens = config.get("max_tokens", 1024)
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise ValueError("Comparison max_tokens must be between 1 and 4096")
        profiles = config.get("profiles")
        if not isinstance(profiles, list) or not 2 <= len(profiles) <= 4:
            raise ValueError("Configure two to four comparison profiles")
        self.profiles: dict[str, dict[str, Any]] = {}
        destinations = set()
        for item in profiles:
            required = {"id", "label", "endpoint", "model"}
            optional = {"api_key_env", "expected_adapter_sha256"}
            if not isinstance(item, dict) or not required <= set(item) or set(item) - required - optional:
                raise ValueError("Invalid comparison profile fields")
            item = copy.deepcopy(item)
            if not isinstance(item["id"], str) or not re.fullmatch(r"[a-z0-9-]{1,40}", item["id"]) or item["id"] in self.profiles:
                raise ValueError("Comparison profile IDs must be unique")
            for key in ("label", "model"):
                if not isinstance(item[key], str) or not re.fullmatch(r"[A-Za-z0-9_.+ @:/-]{1,128}", item[key]):
                    raise ValueError("Invalid comparison label or model")
            if not isinstance(item["endpoint"], str):
                raise ValueError("Comparison endpoint must be a URL")
            item["endpoint"] = private_endpoint(item["endpoint"])
            destination = (item["endpoint"], item["model"])
            if destination in destinations:
                raise ValueError("Comparison profiles must identify distinct served models")
            destinations.add(destination)
            if "api_key_env" in item and (not isinstance(item["api_key_env"], str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item["api_key_env"])):
                raise ValueError("Invalid credential environment name")
            if "expected_adapter_sha256" in item and (not isinstance(item["expected_adapter_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["expected_adapter_sha256"])):
                raise ValueError("Invalid expected adapter hash")
            self.profiles[item["id"]] = item
        self.transport = transport

    def describe(self):
        return {"enabled": True, "max_tokens": self.max_tokens, "temperature": 0,
                "profiles": [{key: item[key] for key in ("id", "label", "model")} for item in self.profiles.values()]}

    def complete(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"profile", "messages"}:
            raise ValueError("Expected profile and messages only")
        if not isinstance(payload["profile"], str) or payload["profile"] not in self.profiles:
            raise ValueError("Unknown comparison profile")
        profile = self.profiles[payload["profile"]]
        request = {"model": profile["model"], "messages": payload["messages"],
                   "temperature": 0, "max_tokens": self.max_tokens, "stream": False}
        config = DeploymentConfig(model="", served_model_name=profile["model"], max_tokens=self.max_tokens)
        validate_request(request, config)
        # Validation does not mutate messages; forward exactly the frozen input.
        key = os.environ.get(profile["api_key_env"]) if "api_key_env" in profile else None
        if "api_key_env" in profile and not key:
            raise RuntimeError("Comparison credential is unavailable")
        started = time.monotonic()
        response = self.transport(profile["endpoint"] + "/chat/completions", copy.deepcopy(request), key)
        seconds, scope = time.monotonic() - started, "upstream_wall_time"
        if not isinstance(response, dict) or response.get("model") != profile["model"]:
            raise ComparisonError("Comparison endpoint returned an unexpected model")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ComparisonError("Comparison endpoint returned an invalid answer")
        choice = choices[0]
        message = choice.get("message")
        if (choice.get("finish_reason") != "stop" or not isinstance(message, dict)
                or message.get("role") != "assistant" or message.get("tool_calls") or message.get("function_call")
                or not isinstance(message.get("content"), str) or not message["content"].strip()
                or len(message["content"]) > 200_000):
            raise ComparisonError("Comparison requires a complete plain answer; tools are unavailable")
        details = safe_model_details(response.get("model_details"))
        details["served_model_name"] = profile["model"]
        expected = profile.get("expected_adapter_sha256")
        if expected and details.get("adapter_sha256") != expected:
            raise ComparisonError("Comparison adapter identity did not match its configured hash")
        reported = response.get("metrics")
        if isinstance(reported, dict):
            duration = reported.get("inference_seconds")
            if (isinstance(duration, (int, float)) and type(duration) in (int, float) and math.isfinite(duration) and duration > 0
                    and reported.get("timing_scope") == "model_inference_including_prefill"):
                seconds, scope = duration, "model_inference_including_prefill"
        metrics = InferenceMetrics()
        metrics.add(response.get("usage"), seconds, scope)
        reasoning = message.get("reasoning_content")
        return {"profile": profile["id"], "model": profile["model"], "answer": message["content"],
                "model_details": details, "identity": "pinned_adapter_match" if expected else "endpoint_reported",
                "metrics": metrics.result(),
                "reasoning_content": reasoning[:20000] if isinstance(reasoning, str) else None}
