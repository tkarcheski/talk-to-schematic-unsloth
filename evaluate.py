"""
evaluate.py - score "talk to the schematic" answers turn by turn.

  python evaluate.py --gold data/test.jsonl --pred preds_ft.jsonl [--pred-base preds_base.jsonl]

predict.py writes one row per (conversation, turn) with the model's answer.
Each turn is graded by its gold type:

  codes   review turn      - finding recall / precision (keyword + refdes match)
  pins    net / pin trace  - pin-set F1 on REFDES.PIN tokens
  value   part lookup      - value string appears in the answer
  number  calc / what-if   - first number within 2 %
  yesno   presence         - correct yes/no
  refuse  not on sheet     - answer declines instead of inventing (hallucination check)
  refdes  fix suggestion   - answer names the right part
"""
import argparse
import json
import re
import sys

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
PIN = r"\b([A-Z]{1,2}\d{1,2})\.([A-Z0-9_]+)\b"

GATES = {
    "review.recall": (">=", 0.90),
    "review.false_alarm_on_clean": ("<=", 0.05),
    "pins.f1": (">=", 0.95),
    "number.acc": (">=", 0.95),
    "refuse.acc": (">=", 0.98),      # must not hallucinate parts or layout facts
}


def strip(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def lines_for(code, text):
    """Lines of the answer that mention the code's part AND its keyword."""
    return [l for l in text.splitlines()
            if re.search(rf"\b({REFDES_OF[code]})\b", l) and re.search(KEYWORDS[code], l, re.I)]


def grade(g, ans):
    t = g["type"]
    if t == "codes":
        gold = set(g["codes"])
        hit = {c for c in gold if lines_for(c, ans)}
        bullets = [l for l in ans.splitlines() if l.strip().startswith("-")]
        return {"review.recall": (len(hit) / len(gold)) if gold else None,
                "review.precision": (len(hit) / max(len(bullets), len(hit))) if bullets else (1.0 if not gold else 0.0),
                "review.false_alarm_on_clean": (len(bullets) > 0) if not gold else None}
    if t == "pins":
        pred = {f"{a}.{b}" for a, b in re.findall(PIN, ans)} - {g.get("query")}
        gold = set(g["pins"])
        if not gold:
            return {"pins.f1": float(bool(re.search(r"isn't connected|not connected|floating|nothing", ans, re.I)))}
        tp = len(pred & gold)
        p, r = (tp / len(pred) if pred else 0), tp / len(gold)
        return {"pins.f1": 0 if p + r == 0 else 2 * p * r / (p + r)}
    if t == "value":
        return {"value.acc": re.sub(r"\s", "", g["value"]).lower() in re.sub(r"\s", "", ans).lower()}
    if t == "number":
        nums = [float(x) for x in re.findall(r"=\s*(-?\d+(?:\.\d+)?)\s*(?:mA|W|V)", ans)]
        v = g["value"]
        ok = any(abs(n - v) <= max(0.02 * abs(v), 0.01) for n in nums[-1:]) or (v == 0 and re.search(r"won't|not conduct", ans))
        return {"number.acc": bool(ok)}
    if t == "yesno":
        said_yes = bool(re.match(r"\s*yes", ans, re.I))
        return {"yesno.acc": said_yes == g["yes"]}
    if t == "refuse":
        return {"refuse.acc": bool(re.search(REFUSE, ans, re.I))}
    if t == "refdes":
        return {"fix.names_part": all(re.search(rf"\b{r}\b", ans) for r in g["refdes"])}
    return {}


def score(gold_rows, preds):
    acc = {}
    for r in gold_rows:
        for i, g in enumerate(r["gold"]):
            ans = strip(preds.get(f"{r['id']}#{i}", ""))
            for k, v in grade(g, ans).items():
                if v is not None:
                    acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}, {k: len(v) for k, v in acc.items()}


def load(path):
    return {r["key"]: r["output"] for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/test.jsonl")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-base")
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(a.gold)]
    ft, n = score(gold, load(a.pred))
    base = score(gold, load(a.pred_base))[0] if a.pred_base else None
    print(f"{'metric':32s} {'n':>5s} {'model':>8s}" + (f" {'base':>8s}" if base else "") + "  gate")
    failed = []
    for k, v in ft.items():
        gate, mark = GATES.get(k), ""
        if gate:
            ok = v >= gate[1] if gate[0] == ">=" else v <= gate[1]
            mark = f"{'PASS' if ok else 'FAIL'} ({gate[0]}{gate[1]})"
            failed += [] if ok else [k]
        print(f"{k:32s} {n[k]:5d} {v:8.3f}" + (f" {base.get(k, float('nan')):8.3f}" if base else "") + f"  {mark}")
    print("\nRELEASE:", "BLOCKED by " + ", ".join(failed) if failed else "all gates passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
