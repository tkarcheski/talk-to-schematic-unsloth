"""Generate real routing answers against a self-hosted endpoint; never execute tools."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import re
from pathlib import Path
import time

from schematic_model.agent import AGENT_TOOLS, private_endpoint, request_json


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


def prepare_cases(cases):
    """Snapshot request images; legacy cases get observed, not source-verified, hashes.

    Existing case files are never rewritten. Relative image paths retain their
    historical interpretation relative to the evaluator's working directory.
    """
    snapshots = {}
    prepared = []
    for case in cases:
        declared = case.get("image_sha256")
        pinned = "image_sha256" in case
        if pinned and (not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared)):
            raise ValueError("Case image_sha256 must be a SHA-256 hash")
        if pinned and (not isinstance(case.get("image"), str) or not case["image"]):
            raise ValueError("Pinned case requires its declared image path")
        messages = copy.deepcopy(case["messages"])
        bindings = []
        for message_index, message in enumerate(messages):
            if not isinstance(message.get("content"), list):
                continue
            for block_index, block in enumerate(message["content"]):
                if block["type"] != "image_url":
                    continue
                path = Path(block["image_url"]["url"]).resolve()
                if pinned and path != Path(case["image"]).resolve():
                    raise ValueError("Message image differs from the pinned case image")
                if path not in snapshots:
                    data = path.read_bytes()
                    snapshots[path] = (hashlib.sha256(data).hexdigest(), 'data:' +
                                       (mimetypes.guess_type(path)[0] or 'image/png') + ';base64,' +
                                       base64.b64encode(data).decode())
                observed, uri = snapshots[path]
                if pinned and observed != declared:
                    raise ValueError("Case image hash differs from the supplied image bytes")
                bindings.append({"message_index": message_index, "block_index": block_index, "sha256": observed})
                block["image_url"]["url"] = uri
        if not bindings:
            raise ValueError("Routing case requires at least one image")
        evidence = {"id": case["id"], "images": bindings,
                    "verification": "declared-hash-verified" if pinned else "legacy-unpinned-observed"}
        prepared.append((case, messages, evidence))
    return prepared


def run(cases_path, output, endpoint, model, max_tokens=256):
    if output.exists():
        raise ValueError("Evaluation output exists; preserve the previous run")
    endpoint = private_endpoint(endpoint)
    case_bytes = cases_path.read_bytes()
    cases_hash = hashlib.sha256(case_bytes).hexdigest()
    cases = [json.loads(line) for line in case_bytes.decode("utf-8").splitlines()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    prepared = prepare_cases(cases)  # Reject any changed pinned image before inference or output creation.
    image_evidence = [evidence for _, _, evidence in prepared]
    input_hash = hashlib.sha256(json.dumps({"cases_sha256": cases_hash, "image_evidence": image_evidence},
                                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True)
    results = []
    for case, messages, evidence in prepared:
        started = time.monotonic()
        result = {"id": case["id"], "expected": case["expected"], "passed": False, "image_evidence": evidence}
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
    report = {"cases_sha256": cases_hash, "evaluation_input_sha256": input_hash, "image_evidence": image_evidence,
              "image_identity_scope": "Declared hashes are verified when supplied; legacy hashes describe observed request bytes only, not historical provenance.",
              "endpoint": endpoint, "model": model, "max_tokens": max_tokens,
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
