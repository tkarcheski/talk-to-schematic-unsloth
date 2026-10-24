"""Run the repository audit; publish only an exact-commit, sanitized model status."""

import argparse
import json
import os
from pathlib import Path
import re

from ai_pilled.checks import require_visible_index
from ai_pilled.full_audit import full_audit
from ai_pilled.runtime import CommandError, run
from ai_pilled.security import scan

REPOSITORY = "tkarcheski/talk-to-schematic-unsloth"
CONTEXT = "ai-pilled/model-audit"
PERSPECTIVES = {"audit-correctness", "audit-security", "audit-maintenance"}


def clean_head(root: Path) -> str:
    require_visible_index(root)
    head = run(["git", "rev-parse", "--verify", "HEAD"], root).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("Invalid commit identity")
    if run(["git", "status", "--porcelain", "--untracked-files=normal"], root):
        raise ValueError("Commit all source changes before the audit")
    pushed = os.environ.get("PRE_COMMIT_TO_REF") or os.environ.get("PRE_COMMIT_LOCAL_BRANCH")
    if pushed and run(["git", "rev-parse", "--verify", pushed + "^{commit}"], root).decode().strip() != head:
        raise ValueError("Check out the pushed commit before auditing")
    return head


def verify_remote(root: Path, head: str) -> None:
    origin = run(["git", "remote", "get-url", "origin"], root).decode().strip()
    allowed = {f"https://github.com/{REPOSITORY}", f"git@github.com:{REPOSITORY}"}
    if origin.removesuffix(".git") not in allowed:
        raise ValueError("Status publication requires this repository's GitHub origin")
    commit = json.loads(run(["gh", "api", "--hostname", "github.com", f"repos/{REPOSITORY}/commits/{head}"], root))
    if commit.get("sha") != head:
        raise ValueError("Commit must already exist on the verified origin")


def publish_status(root: Path, head: str, state: str) -> None:
    descriptions = {
        "pending": "Full audit and three model perspectives are running",
        "success": "Full audit and all three model perspectives passed",
        "failure": "Audit found defects; review private local findings",
        "error": "Audit incomplete; no passing review is available",
    }
    payload = {"state": state, "context": CONTEXT, "description": descriptions[state]}
    run(["gh", "api", "--hostname", "github.com", "--method", "POST",
         f"repos/{REPOSITORY}/statuses/{head}", "--input", "-"], root,
        input_data=json.dumps(payload).encode())


def model_checks_passed(report: dict) -> bool:
    checks = report.get("checks", [])
    perspectives = [item for item in checks if item.get("check") in PERSPECTIVES]
    return (
        report.get("status") == "pass"
        and report.get("metrics", {}).get("model_reviews_run") == 3
        and len(perspectives) == 3
        and {item["check"] for item in perspectives} == PERSPECTIVES
        and all(item.get("status") == "pass" and item.get("metrics", {}).get("model_invocations") == 1
                for item in perspectives)
    )


def audit_repository(root: Path, *, model_reviews: bool = False, publish: bool = False) -> int:
    head = ""
    pending = False
    try:
        model_reviews = model_reviews or publish
        head = clean_head(root)
        if publish:
            verify_remote(root, head)
            publish_status(root, head, "pending")
            pending = True
        report = full_audit(root, model_reviews=model_reviews).to_dict()
        passed = report.get("status") == "pass" and (not model_reviews or model_checks_passed(report))
        if clean_head(root) != head:
            raise ValueError("Commit changed during audit")
        after = scan(root, "worktree", patterns=True)
        if after.status != "pass" or after.snapshot != report.get("snapshot"):
            raise ValueError("Source changed after audit")
        if publish:
            publish_status(root, head, "success" if passed else
                           "failure" if report.get("status") == "fail" else "error")
        print(json.dumps({"commit": head, "audit": "pass" if passed else "blocked",
                          "model_reviews": "pass" if model_reviews and passed else
                          "blocked" if model_reviews else "not_run"}))
        return 0 if passed else 1
    except (CommandError, OSError, ValueError, KeyError, TypeError):
        if pending:
            try:
                publish_status(root, head, "error")
            except (CommandError, OSError, ValueError):
                pass
        print(json.dumps({"audit": "incomplete", "model_reviews": "not_certified"}))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-reviews", action="store_true")
    parser.add_argument("--publish-status", action="store_true",
                        help="Audit a commit already on origin, then publish its sanitized GitHub status")
    args = parser.parse_args()
    return audit_repository(Path.cwd(), model_reviews=args.model_reviews, publish=args.publish_status)


if __name__ == "__main__":
    raise SystemExit(main())
