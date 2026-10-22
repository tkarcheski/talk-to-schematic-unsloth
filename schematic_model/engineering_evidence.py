"""Answer exact native-source questions without promoting role guesses to gold."""

import argparse
import json
from pathlib import Path

from .corpus import parse_eagle


REVIEW_REQUIREMENTS = {
    "component_role": ["reviewed part identity", "functional context"],
    "decoupling_adequacy": ["datasheet requirement and citation", "operating conditions", "PCB placement"],
    "voltage_compatibility": ["reviewed part identities", "datasheet ratings and citations", "operating conditions"],
    "design_defect": ["explicit design requirement", "supporting evidence", "human review"],
    "measured_voltage": ["recorded bench measurement"],
}


def answer(source: Path | bytes, question: dict) -> dict:
    """Return source-verified facts or explicit missing evidence, never role labels.

    Callers choose whether to retain this content. No network or output writes.
    Pages are explicit; cross-sheet topology is not inferred from net names.
    """
    if not isinstance(question, dict) or set(question) != {"kind", "page", "target"}:
        raise ValueError("Question requires kind, page, and target only")
    kind, page, target = question["kind"], question["page"], question["target"]
    if not isinstance(kind, str) or kind not in {"value", "net_members", "pad_net", *REVIEW_REQUIREMENTS}:
        raise ValueError("Unsupported question kind")
    if type(page) is not int or not isinstance(target, str) or not target or len(target) > 128:
        raise ValueError("Question requires a page number and bounded target identifier")
    evidence = parse_eagle(source)
    if not 1 <= page <= len(evidence["sheets"]):
        raise ValueError("Page does not exist")
    sheet = evidence["sheets"][page - 1]
    result = {"schema_version": 1, "source_sha256": evidence["sha256"], "page": page,
              "question": question, "evidence_mode": "native_source", "answer": None}
    if kind in REVIEW_REQUIREMENTS:
        return {**result, "status": "additional_evidence_required",
                "required_evidence": REVIEW_REQUIREMENTS[kind],
                "training_gold_eligible": False}
    values = {part["refdes"]: part["value"] for part in sheet["bom"]}
    if kind == "value":
        found = target in values and bool(values[target])
        value = values.get(target) if found else None
    elif kind == "net_members":
        found = target in sheet["nets"]
        value = sheet["nets"].get(target)
    else:
        matches = [name for name, pads in sheet["nets"].items() if target in pads]
        found = len(matches) == 1
        value = matches[0] if found else None
    return {**result, "status": "source_verified" if found else "not_recorded_on_page",
            "answer": value, "training_gold_eligible": found}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", required=True, choices=["value", "net_members", "pad_net", *REVIEW_REQUIREMENTS])
    parser.add_argument("--target", required=True)
    parser.add_argument("--page", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = answer(args.source, {"kind": args.kind, "target": args.target, "page": args.page})
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
