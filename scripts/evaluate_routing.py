"""Generate real routing answers against a self-hosted endpoint; never execute tools."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import re
from pathlib import Path
import time

from schematic_model.agent import AGENT_TOOLS, private_endpoint, request_json
from schematic_model.training import sha256


def score(message, expected):
    calls = message.get("tool_calls") or []
    if expected["tool"] is None:
        text = message.get("content") or ""
        return not calls and bool(re.search(r"\b(which|select|specify|provide)\b", text.lower())) and any(word in text.lower() for word in ("part", "component", "reference", "designator"))
    if len(calls) != 1:
        return False
    function = calls[0].get("function", {})
    if function.get("name") != expected["tool"]:
        return False
    try:
        arguments = json.loads(function["arguments"])
    except (ValueError, KeyError, TypeError):
        return False
    if expected["tool"] == "web_search":
        return set(arguments) == {"query"} and expected["part"].casefold() in arguments["query"].casefold()
    return arguments == {"label": expected["refdes"]}


def run(cases_path, output, endpoint, model, max_tokens=256):
    if output.exists():
        raise ValueError("Evaluation output exists; preserve the previous run")
    endpoint = private_endpoint(endpoint)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    output.mkdir(parents=True)
    results = []
    for case in cases:
        messages = copy.deepcopy(case["messages"])
        for message in messages:
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if block["type"] == "image_url":
                        path = Path(block["image_url"]["url"])
                        block["image_url"]["url"] = 'data:' + (mimetypes.guess_type(path)[0] or 'image/png') + ';base64,' + base64.b64encode(path.read_bytes()).decode()
        started = time.monotonic()
        result = {"id": case["id"], "expected": case["expected"], "passed": False}
        try:
            response = request_json(endpoint + '/chat/completions', {"model": model, "messages": messages, "tools": AGENT_TOOLS, "max_tokens": max_tokens, "temperature": 0})
            choice = response["choices"][0]
            result.update(response=response, passed=choice.get("finish_reason") in {"stop", "tool_calls"} and score(choice["message"], case["expected"]))
        except Exception as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["elapsed_seconds"] = time.monotonic() - started
        results.append(result)
        with (output / 'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps({"id": result["id"], "passed": result["passed"], "error": result.get("error")}), flush=True)
    counts = {kind: {"correct": sum(r["passed"] for r in results if r["expected"]["kind"] == kind),
                     "total": sum(r["expected"]["kind"] == kind for r in results)} for kind in sorted({r["expected"]["kind"] for r in results})}
    report = {"cases_sha256": sha256(cases_path), "endpoint": endpoint, "model": model, "max_tokens": max_tokens,
              "temperature": 0, "counts": counts, "passed": all(r["passed"] for r in results),
              "scope": "Actual first-action generation; tools not executed, no search traffic; does not qualify schematic answer quality"}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', default='schematic')
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.cases, args.out, args.endpoint, args.model, args.max_tokens), indent=2))


if __name__ == '__main__':
    main()
