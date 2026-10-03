"""
evaluate.py - score "talk to the schematic" answers turn by turn.

  python evaluate.py --gold test.jsonl --pred adapter.jsonl --profile real-grounding [--pred-base base.jsonl]

Each prediction row holds one (conversation, turn) answer. Each turn is graded
by its gold type:

  pins    net / pin trace  - pin-set F1 on REFDES.PIN tokens
  value   part lookup      - bounded value token appears in the answer
  refuse  not on sheet     - answer declines instead of inventing (hallucination check)
These are deterministic format/keyword checks, not a semantic judge. In
particular, refusal scores do not prove that an answer is correct. Missing
predictions and missing release-gate evidence are errors, never passing scores.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

REFUSE = r"(no|not|isn't|doesn't|don't|aren't)\b.{0,40}\b(on|in|shown|show|drawn|specified|marked|state|sheet|schematic)"
REFDES = r"[A-Z][A-Z0-9_$-]*"
PIN = rf"(?<![\w$-])({REFDES})\.([A-Z0-9_$+-]+)(?![\w$+-])"

class EvaluationError(ValueError):
    """Malformed evaluation data or incomplete prediction coverage."""

PROFILES = {
    "real-grounding": {"value.acc": (">=", 0.95), "pins.f1": (">=", 0.95), "refuse.acc": (">=", 0.98)},
    "real-vision": {"value.acc": (">=", 0.95), "refuse.acc": (">=", 0.98)},
}
REFUSAL_SCORERS = {"v1": "keyword-refusal-v1", "v2": "contextual-refusal-v2"}


def strip(t):
    """Remove complete reasoning blocks; an unclosed block has no final answer."""
    text = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S | re.I)
    return re.split(r"<think>", text, maxsplit=1, flags=re.I)[0].strip()


def contextual_refusal(question, answer):
    """Conservative v2 grammar for the two real-corpus refusal question forms.

    A complete answer must contain a subject-grounded, evidence-scoped denial,
    and every clause must match the restricted grammar. Unrecognized language
    needs manual review; rejection is not proof of a hallucination. This does
    not inspect source truth and is deliberately separate from default v1.
    """
    if not isinstance(question, str) or not question.strip():
        raise EvaluationError("Contextual refusal scoring requires user question text")
    question = " ".join(question.split())
    measured = re.fullmatch(
        rf"What measured voltage was observed at ({REFDES}\.[A-Z0-9_$+-]+) during a powered bench test\?",
        question, re.I)
    missing = re.fullmatch(rf"What value is ({REFDES}), and what does it connect to\?", question, re.I)
    result = {"version": REFUSAL_SCORERS["v2"], "accepted": False,
              "status": "manual_review", "reason": "unsupported_question"}
    if not (measured or missing):
        return result
    kind, subject = ("measurement", measured[1]) if measured else ("missing_component", missing[1])
    result.update(kind=kind, subject=subject)
    subject_pattern = re.escape(subject)
    scope = (r"(?:the |this )?(?:(?:supplied|provided|extracted|native) )*"
             r"(?:schematic(?: or source)?(?: evidence)?|source(?: evidence)?|sheet(?: evidence)?|evidence)")
    if measured:
        fact = r"(?:bench measurement|powered bench measurement|measured voltage(?: value)?)"
        location = rf"(?: (?:at|for) {subject_pattern})?"
        predicate = r"(?:is|was) (?:supplied|provided|recorded|observed|shown)"
        absence = rf"no {fact}{location} {predicate}{location}(?: during a powered bench test)?"
        primary = [
            rf"{absence} (?:in|on) {scope}",
            rf"(?:in|on) {scope}, {absence}",
            rf"{scope} (?:contains|provides|records) no (?:recorded )?{fact}{location}",
        ]
        secondary = [
            r"i cannot (?:infer|determine) (?:a |the )?measured voltage",
            rf"(?:its |the )?measured voltage cannot be determined from {scope}",
        ]
    else:
        primary = [
            rf"there is no (?:component )?{subject_pattern} (?:in|on) {scope}",
            rf"no (?:component )?{subject_pattern} is (?:present|shown|included) (?:in|on) {scope}",
            rf"{subject_pattern} (?:is not|isn't) (?:present|shown|included|recorded|specified) (?:in|on) {scope}",
            rf"{scope} does not (?:contain|include|show|specify) (?:component )?{subject_pattern}",
        ]
        secondary = [
            r"its value and connections are not shown",
            rf"its value and connections cannot be determined from {scope}",
        ]
    # Do not discard code blocks, quoted tails, contrast clauses, or unknown
    # text: any unmatched clause must prevent automatic acceptance.
    text = strip(answer).replace("**", "").replace("\\n", "\n")
    clauses = [" ".join(part.split()).strip(" .") for part in
               re.split(r"[.!]\s+|[\r\n]+|,\s+so\s+", text) if part.strip(" .")]
    if not clauses:
        return {**result, "reason": "empty_answer"}
    grounded = False
    for clause in clauses:
        if any(re.fullmatch(pattern, clause, re.I) for pattern in primary):
            grounded = True
        elif not any(re.fullmatch(pattern, clause, re.I) for pattern in secondary):
            return {**result, "reason": "unrecognized_or_unsupported_claim"}
    if not grounded:
        return {**result, "reason": "no_grounded_absence_statement"}
    return {**result, "accepted": True, "status": "accepted", "reason": "restricted_grammar_match"}


def question_text(row, turn):
    """Read only the aligned user message, never the gold assistant response."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise EvaluationError("Contextual refusal scoring requires aligned messages")
    users = [message for message in messages if message.get("role") == "user"]
    if turn >= len(users):
        raise EvaluationError("Contextual refusal scoring requires aligned user turns")
    content = users[turn].get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise EvaluationError("User content must contain typed content objects")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") not in {"image", "image_url"}:
                raise EvaluationError("User text content is malformed or unsupported")
        text = "\n".join(parts)
    else:
        raise EvaluationError("User content must be text or a typed content list")
    if not text.strip():
        raise EvaluationError("Contextual refusal scoring requires nonempty user text")
    return text


def grade(g, ans, *, question=None, refusal_scorer="v1"):
    """Grade one label; callers scoring a dataset must use :func:`score`."""
    t = g["type"]
    ans = strip(ans)
    if t == "pins":
        pred = {f"{a}.{b}" for a, b in re.findall(PIN, ans)} - {g.get("query")}
        gold = set(g["pins"])
        # Source net names such as MISO_3.3V resemble physical pin tokens.
        # Exclude only the declared literal net, never an expected endpoint.
        if g.get("net") not in gold:
            pred.discard(g.get("net"))
        if not gold:
            return {"pins.f1": float(not pred and bool(re.search(
                r"isn't connected|not connected|floating|nothing", ans, re.I)))}
        tp = len(pred & gold)
        p, r = (tp / len(pred) if pred else 0), tp / len(gold)
        return {"pins.f1": 0 if p + r == 0 else 2 * p * r / (p + r)}
    if t == "value":
        # Permit formatting whitespace, but never match 150 inside 1500,
        # -150, 150.5, or a larger component identifier.
        value = r"\s*".join(re.escape(part) for part in g["value"].split())
        return {"value.acc": bool(re.search(
            rf"(?<![\w.+-]){value}(?!\w|\.\d)", ans, re.I))}
    if t == "refuse":
        if refusal_scorer == "v2":
            return {"refuse.acc": contextual_refusal(question, ans)["accepted"]}
        if refusal_scorer != "v1":
            raise EvaluationError(f"Unknown refusal scorer: {refusal_scorer}")
        return {"refuse.acc": bool(re.search(REFUSE, ans, re.I))}
    raise EvaluationError(f"Unknown gold type: {t!r}")


def _string_list(value, field, *, allow_empty=True):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvaluationError(f"{field} must be a list of nonempty strings")
    if not allow_empty and not value:
        raise EvaluationError(f"{field} must not be empty")
    if len(value) != len(set(value)):
        raise EvaluationError(f"{field} contains duplicates")


def _validate_label(label, location):
    if not isinstance(label, dict):
        raise EvaluationError(f"{location} must be an object")
    kind = label.get("type")
    if kind == "pins":
        _string_list(label.get("pins"), f"{location}.pins")
        if "net" in label and (not isinstance(label["net"], str) or not label["net"].strip()):
            raise EvaluationError(f"{location}.net must be a nonempty string")
        pins = label["pins"] + ([label["query"]] if "query" in label else [])
        if any(not isinstance(pin, str) or not re.fullmatch(PIN, pin) for pin in pins):
            raise EvaluationError(f"{location}: pins must use REFDES.PIN syntax")
        if label.get("query") in label["pins"]:
            raise EvaluationError(f"{location}: query pin must not be in expected far-end pins")
    elif kind == "value":
        if not isinstance(label.get("value"), str) or not label["value"].strip():
            raise EvaluationError(f"{location}.value must be a nonempty string")
    elif kind != "refuse":
        raise EvaluationError(f"{location}: unknown gold type {kind!r}")


def validate_gold_rows(gold_rows):
    """Validate gold rows and return the exact set of expected prediction keys."""
    if not isinstance(gold_rows, (list, tuple)) or not gold_rows:
        raise EvaluationError("Gold dataset must contain at least one row")
    ids, expected = set(), set()
    for row_number, row in enumerate(gold_rows, 1):
        if not isinstance(row, dict):
            raise EvaluationError(f"Gold row {row_number} must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip() or row_id != row_id.strip() or "#" in row_id:
            raise EvaluationError(f"Gold row {row_number}: id must be nonempty and must not contain '#'")
        if row_id in ids:
            raise EvaluationError(f"Duplicate gold id: {row_id}")
        ids.add(row_id)
        labels = row.get("gold")
        if not isinstance(labels, list) or not labels:
            raise EvaluationError(f"Gold row {row_id}: gold must be a nonempty list")
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
                raise EvaluationError(f"Gold row {row_id}: messages must be a list of objects")
            roles = [message.get("role") for message in messages]
            if roles and roles[0] == "system":
                roles = roles[1:]
            if roles != [role for _ in labels for role in ("user", "assistant")]:
                raise EvaluationError(f"Gold row {row_id}: messages must alternate user/assistant once per gold label")
        for index, label in enumerate(labels):
            _validate_label(label, f"{row_id}#{index}")
            expected.add(f"{row_id}#{index}")
    return expected


def score(gold_rows, preds, *, refusal_scorer="v1"):
    """Return (mean metrics, evidence counts), requiring complete predictions."""
    if refusal_scorer not in REFUSAL_SCORERS:
        raise EvaluationError(f"Unknown refusal scorer: {refusal_scorer}")
    expected = validate_gold_rows(gold_rows)
    if not isinstance(preds, Mapping) or any(not isinstance(key, str) or not isinstance(output, str)
                                             for key, output in preds.items()):
        raise EvaluationError("Predictions must map string keys to string outputs")
    missing, extra = expected - preds.keys(), preds.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} keys (e.g. {sorted(missing)[:3]})")
        if extra:
            details.append(f"unknown {len(extra)} keys (e.g. {sorted(extra)[:3]})")
        raise EvaluationError("Prediction coverage mismatch: " + "; ".join(details))
    acc = {}
    for r in gold_rows:
        for i, g in enumerate(r["gold"]):
            ans = strip(preds[f"{r['id']}#{i}"])
            question = question_text(r, i) if refusal_scorer == "v2" and g["type"] == "refuse" else None
            for k, v in grade(g, ans, question=question, refusal_scorer=refusal_scorer).items():
                if v is not None:
                    acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}, {k: len(v) for k, v in acc.items()}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise EvaluationError(f"Non-finite JSON number: {value}")


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            except (ValueError, TypeError) as exc:
                raise EvaluationError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise EvaluationError(f"{path}: dataset is empty")
    return rows


def load_gold(path):
    rows = _read_jsonl(path)
    validate_gold_rows(rows)
    return rows


def load(path):
    """Load predictions, rejecting malformed rows and duplicate keys."""
    predictions = {}
    for row_number, row in enumerate(_read_jsonl(path), 1):
        key, output = row.get("key"), row.get("output")
        if not isinstance(key, str) or not key.strip() or not isinstance(output, str):
            raise EvaluationError(f"{path}:{row_number}: key and output must be strings, with a nonempty key")
        if key in predictions:
            raise EvaluationError(f"{path}:{row_number}: duplicate prediction key {key!r}")
        predictions[key] = output
    return predictions


def gate_failures(metrics, counts, gates):
    """Every required metric needs evidence and a finite passing score."""
    failed = []
    for name, (operator, threshold) in gates.items():
        if name not in metrics or counts.get(name, 0) < 1:
            failed.append(f"{name} (no evidence)")
            continue
        value = metrics[name]
        if not math.isfinite(value) or not 0 <= value <= 1:
            failed.append(f"{name} (invalid score)")
        elif not (value >= threshold if operator == ">=" else value <= threshold):
            failed.append(name)
    return failed


def _validate_report_destination(path, inputs):
    """Reports may replace previous reports, but must never replace evidence."""
    path = Path(path)
    for source in inputs:
        source = Path(source)
        if path.resolve() == source.resolve() or (
            path.exists() and source.exists() and path.samefile(source)
        ):
            raise EvaluationError(f"Report path aliases an input file: {source}")


def _write_report(path, report, inputs):
    """Publish one complete JSON object atomically; keep old output on failure."""
    path = Path(path)
    _validate_report_destination(path, inputs)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_report_destination(path, inputs)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_hash(path):
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _report_evidence(args):
    return {"schema_version": 1, "profile": args.profile,
            "refusal_scorer": REFUSAL_SCORERS[args.refusal_scorer],
            "gold_sha256": _file_hash(args.gold),
            "predictions_sha256": _file_hash(args.pred),
            "baseline_predictions_sha256": _file_hash(args.pred_base),
            "limitations": "Deterministic format/keyword scoring; not a semantic or hardware safety certification."}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-base")
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--refusal-scorer", choices=REFUSAL_SCORERS, default="v1",
                    help="v2 is an opt-in restricted contextual grammar; unmatched prose needs manual review")
    ap.add_argument("--report", type=Path, help="write machine-readable benchmark evidence")
    a = ap.parse_args(argv)
    inputs = [path for path in (a.gold, a.pred, a.pred_base) if path is not None]
    # Keep this separate from the scoring-error handler: an unsafe destination
    # must not receive even an 'incomplete' report.
    if a.report:
        try:
            _validate_report_destination(a.report, inputs)
        except (ValueError, OSError) as exc:
            print(f"RELEASE: BLOCKED: {exc}", file=sys.stderr)
            return 2
    try:
        gold = load_gold(a.gold)
        ft, n = score(gold, load(a.pred), refusal_scorer=a.refusal_scorer)
        base = score(gold, load(a.pred_base), refusal_scorer=a.refusal_scorer)[0] if a.pred_base else None
    except (ValueError, OSError) as exc:
        if a.report:
            report = {**_report_evidence(a), "metrics": {}, "counts": {}, "baseline_metrics": None,
                      "failures": [str(exc)], "status": "incomplete"}
            try:
                _write_report(a.report, report, inputs)
            except (ValueError, OSError) as write_error:
                print(f"Report was not updated: {write_error}", file=sys.stderr)
        print(f"RELEASE: BLOCKED: {exc}", file=sys.stderr)
        return 2
    gates = PROFILES[a.profile]
    failed = gate_failures(ft, n, gates)
    if a.report:
        report = {**_report_evidence(a), "metrics": ft, "counts": n,
                  "baseline_metrics": base, "failures": failed,
                  "status": "fail" if failed else "pass"}
        try:
            _write_report(a.report, report, inputs)
        except (ValueError, OSError) as exc:
            print(f"RELEASE: BLOCKED: Report was not updated: {exc}", file=sys.stderr)
            return 2
    print("Grading uses format and keyword heuristics; it does not establish semantic correctness.")
    print(f"Refusal scorer: {REFUSAL_SCORERS[a.refusal_scorer]}")
    print(f"{'metric':32s} {'n':>5s} {'model':>8s}" + (f" {'base':>8s}" if base else "") + "  gate")
    for k, v in ft.items():
        gate, mark = gates.get(k), ""
        if gate:
            ok = v >= gate[1] if gate[0] == ">=" else v <= gate[1]
            mark = f"{'PASS' if ok else 'FAIL'} ({gate[0]}{gate[1]})"
        print(f"{k:32s} {n[k]:5d} {v:8.3f}" + (f" {base.get(k, float('nan')):8.3f}" if base else "") + f"  {mark}")
    print("\nRELEASE:", "BLOCKED by " + ", ".join(failed) if failed else "all gates passed")
    print(f"Scope: {a.profile} benchmark only; passing does not establish general engineering reliability.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
