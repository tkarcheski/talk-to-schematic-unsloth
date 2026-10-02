"""
evaluate.py - score "talk to the schematic" answers turn by turn.

  python evaluate.py --gold data/test.jsonl --pred preds_ft.jsonl [--pred-base preds_base.jsonl]

predict.py writes one row per (conversation, turn) with the model's answer.
Each turn is graded by its gold type:

  codes   review turn      - finding recall / precision (keyword + refdes match)
  pins    net / pin trace  - pin-set F1 on REFDES.PIN tokens
  value   part lookup      - bounded value token appears in the answer
  number  calc / what-if   - final equation result within 2 %, including units
  yesno   presence         - correct yes/no
  refuse  not on sheet     - answer declines instead of inventing (hallucination check)
  refdes  fix suggestion   - answer names the right part
These are deterministic format/keyword checks, not a semantic judge. In
particular, review and refusal scores do not prove that an answer is correct.
Legacy numeric labels without a unit receive magnitude-only grading; the CLI
reports their count. Missing predictions and missing release-gate evidence
are errors, never passing scores.
"""
import argparse
import json
import math
import re
import sys
from collections.abc import Mapping

KEYWORDS = {  # code -> regex that a correct prose finding should match
    "LDO_VIN_EXCEEDED": r"abs.?max|exceed|input.*(rating|max)|rated",
    "LDO_DROPOUT": r"dropout|headroom|regulation",
    "LDO_CURRENT": r"current|mA.*(exceed|rating)",
    "LDO_THERMAL": r"thermal|dissipat|\bW\b|heat",
    "REG_OUTPUT_CAP": r"output cap|stabil|oscillat|minimum",
    "TVS_STANDOFF_LOW": r"stand.?off|conduct|TVS",
    "MISSING_INPUT_PROTECTION": r"TVS|ESD|surge|protection",
    "FUSE_UNDERSIZED": r"fuse|F1",
    "CAP_VOLTAGE_DERATING": r"derat|rated|voltage",
    "MISSING_DECOUPLING": r"decoupl|bypass|100 ?nF",
    "DECOUPLING_VALUE": r"decoupl|100 ?nF|value",
    "FLOATING_INPUT": r"float|ADDR|unconnected",
    "I2C_PULLUP_VALUE": r"pull.?up|rise|sink|weak|I2C",
    "I2C_PIN_SWAP": r"swap|SDA.*SCL|SCL.*SDA",
    "NET_LABEL_MISMATCH": r"SDA_1|label|not connected",
    "LED_REVERSED": r"revers|polarity|backwards|cathode",
    "LED_CURRENT": r"current|mA",
}
REFDES_OF = {"LDO_VIN_EXCEEDED": "U1", "LDO_DROPOUT": "U1", "LDO_CURRENT": "U1", "LDO_THERMAL": "U1",
             "REG_OUTPUT_CAP": "C2", "TVS_STANDOFF_LOW": "D1", "MISSING_INPUT_PROTECTION": "J1|D1|VIN",
             "FUSE_UNDERSIZED": "F1", "CAP_VOLTAGE_DERATING": "C1", "MISSING_DECOUPLING": "U2|C3",
             "DECOUPLING_VALUE": "C3", "FLOATING_INPUT": "U2", "I2C_PULLUP_VALUE": "R1|R2",
             "I2C_PIN_SWAP": "U2", "NET_LABEL_MISMATCH": "R1|U2", "LED_REVERSED": "D2", "LED_CURRENT": "R3|D2"}
REFUSE = r"(no|not|isn't|doesn't|don't|aren't)\b.{0,40}\b(on|in|shown|show|drawn|specified|marked|state|sheet|schematic)"
PIN = r"\b([A-Z]{1,2}\d+)\.([A-Z0-9_]+)\b"

UNITS = {
    "A": ("current", 1.0), "mA": ("current", 1e-3),
    "uA": ("current", 1e-6), "µA": ("current", 1e-6),
    "V": ("voltage", 1.0), "mV": ("voltage", 1e-3),
    "W": ("power", 1.0), "mW": ("power", 1e-3),
}
NUMBER_RESULT = re.compile(
    r"=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(mA|uA|µA|mV|mW|A|V|W)\b"
)
NEGATED_FINDING = re.compile(
    r"\b(?:no|without)\s+(?:[\w-]+\s+){0,3}"
    r"(?:problems?|issues?|faults?|concerns?|risk|violations?)\b"
    r"|\b(?:not|isn't|is not|doesn't|does not)\s+(?:[\w-]+\s+){0,2}"
    r"(?:exceed\w*|overheat\w*|revers\w*|float\w*|missing|swapped|too (?:high|low|weak))\b"
    r"|\b(?:within (?:its |the )?(?:rating|limits?)|adequate headroom|enough headroom)\b",
    re.I,
)
CLEAN_REVIEW = re.compile(
    r"\bno (?:\w+ )?(?:issues|problems|faults)\b|\blooks? (?:fine|good)\b"
    r"|\ball (?:check out|checks? pass)|\bevery check passes\b", re.I
)


class EvaluationError(ValueError):
    """Malformed evaluation data or incomplete prediction coverage."""

GATES = {
    "review.recall": (">=", 0.90),
    "review.false_alarm_on_clean": ("<=", 0.05),
    "pins.f1": (">=", 0.95),
    "number.acc": (">=", 0.95),
    "refuse.acc": (">=", 0.98),      # must not hallucinate parts or layout facts
}


def strip(t):
    """Remove complete reasoning blocks; an unclosed block has no final answer."""
    text = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S | re.I)
    return re.split(r"<think>", text, maxsplit=1, flags=re.I)[0].strip()


def _clauses(text):
    return [clause.strip() for clause in re.split(r"\n|(?<=[.!?])\s+(?=[A-Z])", text)
            if clause.strip()]


def lines_for(code, text):
    """Candidate claims with a matching part/keyword and no recognized denial.

    This intentionally remains a limited prose heuristic; compound sentences
    and paraphrases require human review or a separate structured benchmark.
    """
    return [line for line in _clauses(text)
            if re.search(rf"\b({REFDES_OF[code]})\b", line)
            and re.search(KEYWORDS[code], line, re.I)
            and not NEGATED_FINDING.search(line)]


def _number_correct(gold, answer):
    matches = NUMBER_RESULT.findall(answer)
    if not matches:
        return False
    value, unit = matches[-1]
    actual = float(value)
    if not math.isfinite(actual):
        return False
    expected = gold["value"]
    if "unit" in gold:
        expected_dimension, expected_scale = UNITS[gold["unit"]]
        actual_dimension, actual_scale = UNITS[unit]
        if expected_dimension != actual_dimension:
            return False
        actual = actual * actual_scale / expected_scale
    # A zero label requires zero; a nonzero absolute floor would accept an
    # energized circuit for a nominally nonconducting diode.
    tolerance = 0.02 * abs(expected) if expected else 0.0
    return abs(actual - expected) <= tolerance


def grade(g, ans):
    """Grade one label; callers scoring a dataset must use :func:`score`."""
    t = g["type"]
    ans = strip(ans)
    if t == "codes":
        gold = set(g["codes"])
        hit = {c for c in gold if lines_for(c, ans)}
        claims = {line for code in KEYWORDS for line in lines_for(code, ans)}
        claims.update(line for line in _clauses(ans)
                      if re.search(r"\b(?:problem|fault|fail|overheat|unsafe)\w*\b", line, re.I)
                      and re.search(r"\b[A-Z]{1,2}\d+\b", line)
                      and not NEGATED_FINDING.search(line))
        clean = bool(CLEAN_REVIEW.search(ans)) and not claims
        return {"review.recall": (len(hit) / len(gold)) if gold else None,
                "review.precision": (len(hit) / max(len(claims), len(hit))) if claims else float(clean),
                "review.false_alarm_on_clean": float(not clean) if not gold else None}
    if t == "pins":
        pred = {f"{a}.{b}" for a, b in re.findall(PIN, ans)} - {g.get("query")}
        gold = set(g["pins"])
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
    if t == "number":
        return {"number.acc": _number_correct(g, ans)}
    if t == "yesno":
        polarity = re.match(r"\s*(yes|no)\b", ans, re.I)
        if not polarity:
            return {"yesno.acc": False}
        said_yes = polarity[1].lower() == "yes"
        # Explicit contradictions are rejected. This is not a general natural
        # language entailment checker; the benchmark expects Yes/No first.
        rest = ans[polarity.end():]
        contradiction = re.search(r"\bno\b|\b(?:isn't|doesn't|not)\b", rest, re.I) if said_yes else re.search(r"\byes\b", rest, re.I)
        return {"yesno.acc": said_yes == g["yes"] and not contradiction}
    if t == "refuse":
        return {"refuse.acc": bool(re.search(REFUSE, ans, re.I))}
    if t == "refdes":
        return {"fix.names_part": bool(g["refdes"]) and all(
            re.search(rf"\b{re.escape(r)}\b", ans) for r in g["refdes"])}
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
    if kind == "codes":
        _string_list(label.get("codes"), f"{location}.codes")
        unknown = set(label["codes"]) - KEYWORDS.keys()
        if unknown:
            raise EvaluationError(f"{location}: unknown finding codes {sorted(unknown)}")
    elif kind == "pins":
        _string_list(label.get("pins"), f"{location}.pins")
        pins = label["pins"] + ([label["query"]] if "query" in label else [])
        if any(not isinstance(pin, str) or not re.fullmatch(PIN, pin) for pin in pins):
            raise EvaluationError(f"{location}: pins must use REFDES.PIN syntax")
        if label.get("query") in label["pins"]:
            raise EvaluationError(f"{location}: query pin must not be in expected far-end pins")
    elif kind == "value":
        if not isinstance(label.get("value"), str) or not label["value"].strip():
            raise EvaluationError(f"{location}.value must be a nonempty string")
    elif kind == "number":
        value = label.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise EvaluationError(f"{location}.value must be a finite number")
        if "unit" in label and (not isinstance(label["unit"], str) or label["unit"] not in UNITS):
            raise EvaluationError(f"{location}.unit must be one of {', '.join(UNITS)}")
    elif kind == "yesno":
        if not isinstance(label.get("yes"), bool):
            raise EvaluationError(f"{location}.yes must be a boolean")
    elif kind == "refdes":
        _string_list(label.get("refdes"), f"{location}.refdes", allow_empty=False)
        if any(not re.fullmatch(r"[A-Z]{1,2}\d+", ref) for ref in label["refdes"]):
            raise EvaluationError(f"{location}.refdes contains invalid reference designators")
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


def score(gold_rows, preds):
    """Return (mean metrics, evidence counts), requiring complete predictions."""
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
            for k, v in grade(g, ans).items():
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


def gate_failures(metrics, counts):
    """Every required metric needs evidence and a finite passing score."""
    failed = []
    for name, (operator, threshold) in GATES.items():
        if name not in metrics or counts.get(name, 0) < 1:
            failed.append(f"{name} (no evidence)")
            continue
        value = metrics[name]
        if not math.isfinite(value) or not 0 <= value <= 1:
            failed.append(f"{name} (invalid score)")
        elif not (value >= threshold if operator == ">=" else value <= threshold):
            failed.append(name)
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/test.jsonl")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-base")
    a = ap.parse_args(argv)
    try:
        gold = load_gold(a.gold)
        ft, n = score(gold, load(a.pred))
        base = score(gold, load(a.pred_base))[0] if a.pred_base else None
    except (EvaluationError, OSError) as exc:
        print(f"RELEASE: BLOCKED: {exc}", file=sys.stderr)
        return 2
    legacy_numbers = sum(label["type"] == "number" and "unit" not in label
                         for row in gold for label in row["gold"])
    if legacy_numbers:
        print(f"LIMITATION: {legacy_numbers} legacy numeric labels lack units; these receive magnitude-only grading.")
    print("Grading uses format and keyword heuristics; it does not establish semantic correctness.")
    print(f"{'metric':32s} {'n':>5s} {'model':>8s}" + (f" {'base':>8s}" if base else "") + "  gate")
    failed = gate_failures(ft, n)
    for k, v in ft.items():
        gate, mark = GATES.get(k), ""
        if gate:
            ok = v >= gate[1] if gate[0] == ">=" else v <= gate[1]
            mark = f"{'PASS' if ok else 'FAIL'} ({gate[0]}{gate[1]})"
        print(f"{k:32s} {n[k]:5d} {v:8.3f}" + (f" {base.get(k, float('nan')):8.3f}" if base else "") + f"  {mark}")
    print("\nRELEASE:", "BLOCKED by " + ", ".join(failed) if failed else "all gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
