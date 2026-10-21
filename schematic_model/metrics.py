"""Content-free, bounded inference telemetry shared by native and gateway APIs."""
from __future__ import annotations

import math
import re

SCOPES = {"model_inference_including_prefill", "upstream_wall_time", "mixed"}


def safe_model_details(value):
    """Never forward provider metadata wholesale (it may contain local paths)."""
    if not isinstance(value, dict):
        return {}
    output = {}
    for key in ("served_model_name", "backend", "backend_version", "base_model_name"):
        item = value.get(key)
        if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.+ @:-]{1,128}", item):
            output[key] = item
    for key in ("model_fingerprint", "adapter_sha256", "base_model_sha256"):
        item = value.get(key)
        if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
            output[key] = item
    return output


def native_model_details(engine):
    provenance = getattr(engine, "provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    artifacts = provenance.get("model", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    packages = provenance.get("packages", {})
    if not isinstance(packages, dict):
        packages = {}
    details = {"served_model_name": engine.config.served_model_name,
               "backend": provenance.get("backend", "embedded"),
               "backend_version": packages.get("unsloth"),
               "model_fingerprint": getattr(engine, "fingerprint", None)}
    for artifact in artifacts.get("files", []):
        if artifact.get("name") == "adapter_model.safetensors":
            details["adapter_sha256"] = artifact.get("sha256")
    base = artifacts.get("base_model")
    if isinstance(base, dict):
        from pathlib import Path
        details["base_model_name"] = Path(base.get("path", "")).name
        details["base_model_sha256"] = base.get("sha256")
    return safe_model_details(details)


class InferenceMetrics:
    """One HTTP response's model steps; approval/search wait time is excluded."""
    def __init__(self):
        self.counts = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        self.seconds = 0.0
        self.scopes = set()
        self.steps = 0

    def add(self, usage, seconds, scope):
        self.steps += 1
        usage = usage if isinstance(usage, dict) else {}
        for key in self.counts:
            value = usage.get(key)
            if type(value) is not int or not 0 <= value <= 1_000_000_000:
                self.counts[key] = None
            elif self.counts[key] is not None:
                self.counts[key] += value
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            self.seconds = None
        elif self.seconds is not None:
            self.seconds += seconds
        self.scopes.add(scope if scope in SCOPES else None)

    def result(self):
        seconds = self.seconds if self.steps else None
        tokens = self.counts["completion_tokens"] if self.steps else None
        scope = next(iter(self.scopes)) if len(self.scopes) == 1 else "mixed" if self.scopes else None
        return {**(self.counts if self.steps else dict.fromkeys(self.counts)),
                "inference_seconds": round(seconds, 6) if seconds else None,
                "tokens_per_second": round(tokens / seconds, 2) if tokens is not None and seconds else None,
                "timing_scope": scope, "aggregation": "response", "model_steps": self.steps}
