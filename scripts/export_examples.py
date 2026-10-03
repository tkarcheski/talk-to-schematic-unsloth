"""Export complete simulated gold dialogues and unchanged real schematic images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(message: dict) -> str:
    blocks = message.get("content")
    if not isinstance(blocks, list):
        raise ValueError("Messages must contain typed content blocks")
    texts = [block.get("text") for block in blocks if block.get("type") == "text"]
    if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError("Every conversation message needs nonempty text")
    return "\n".join(texts)


def _fence(text: str) -> str:
    length = max((len(match[0]) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, length + 1)
    return fence + "text\n" + text + "\n" + fence


def _inside(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-directory corpus file: {relative}")
    return path


def _prepare(row: dict, data: Path) -> dict:
    messages, gold = row.get("messages"), row.get("gold")
    if not isinstance(messages, list) or not isinstance(gold, list) or len(gold) != 6:
        raise ValueError("Each selected example must contain six labelled turns")
    if len(messages) != 13 or [m.get("role") for m in messages] != ["system"] + ["user", "assistant"] * 6:
        raise ValueError("Each selected example must contain system plus six complete user/assistant pairs")
    for message in messages:
        _text(message)
    for label in gold:
        if not isinstance(label, dict) or label.get("type") not in {"value", "pins", "refuse"}:
            raise ValueError("Examples must retain supported factual gold labels")
    provenance = row.get("provenance", {})
    if provenance.get("evidence_mode") != "native_facts_supplied_in_prompt":
        raise ValueError("This export requires explicitly evidence-assisted source-fact dialogues")
    repository, commit = provenance.get("repository", ""), provenance.get("commit", "")
    if not re.fullmatch(r"adafruit/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Examples require an official source repository and full pinned commit")
    if row.get("family_id") != repository:
        raise ValueError("Distinct board families must match their source repository")
    if provenance.get("license") != "CC-BY-SA-3.0":
        raise ValueError("Example source license must be explicit")
    image_blocks = [block for message in messages for block in message["content"] if block.get("type") == "image"]
    if len(image_blocks) != 1:
        raise ValueError("Each example must have exactly one source schematic image")
    image_relative = image_blocks[0].get("image", "")
    image_path = _inside(data, image_relative)
    image_bytes = image_path.read_bytes()
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The original image must be a PNG")
    image_record = next((item for item in provenance.get("images", []) if item.get("path") == image_relative), None)
    if image_record is None or image_record.get("sha256") != digest(image_bytes):
        raise ValueError("Source image hash does not match frozen dataset provenance")
    source_dir = data / "sources" / repository.replace("/", "__")
    native = _inside(source_dir, provenance.get("source_path", ""))
    if digest(native.read_bytes()) != provenance.get("sha256"):
        raise ValueError("Native schematic hash does not match frozen dataset provenance")
    source_metadata = json.loads(_inside(source_dir, "source.json").read_text())
    if source_metadata.get("revision") != commit:
        raise ValueError("Source metadata commit differs from the selected dialogue")
    license_path = source_metadata.get("license_path", "")
    _inside(source_dir, license_path)
    prefix = f"https://github.com/{repository}/blob/{commit}/"
    return {"row": row, "repository": repository, "commit": commit,
            "native_sha256": provenance["sha256"], "image_sha256": digest(image_bytes),
            "image_bytes": image_bytes, "image_name": image_path.name,
            "source_url": prefix + quote(provenance["source_path"]),
            "license_url": prefix + quote(license_path)}


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_examples(data: Path, output: Path, images: Path, manifest: Path, *, count: int = 25,
                    overwrite: bool = False) -> dict:
    if count < 1:
        raise ValueError("Example count must be positive")
    if not overwrite and (output.exists() or manifest.exists() or images.exists()):
        raise ValueError("Example artifacts already exist; use --overwrite for an intentional refresh")
    rows, dataset_hashes = [], {}
    seen_ids = set()
    for split in ("train", "val", "test"):
        path = data / f"{split}.jsonl"
        content = path.read_bytes()
        dataset_hashes[split] = digest(content)
        for line in content.decode("utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != split or not isinstance(row.get("id"), str) or row["id"] in seen_ids:
                raise ValueError("Conversation identifiers must be unique and agree with their split")
            seen_ids.add(row["id"])
            rows.append(row)
    # Stable ordering gives the same 25 boards on every replay; it does not select
    # examples based on a model's correctness or merge related revisions as boards.
    candidates = sorted(rows, key=lambda row: (row.get("family_id", ""), row["id"]))
    selected, families = [], set()
    for row in candidates:
        family = row.get("family_id")
        if not isinstance(family, str) or not family:
            raise ValueError("Every row needs a board-family identifier")
        if family not in families:
            selected.append(row)
            families.add(family)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} distinct board conversations; only {len(selected)} available")
    prepared = [_prepare(row, data) for row in selected]
    lines = ["# Simulated conversations on real schematics", "",
             f"{count} distinct Adafruit board designs, six complete user/assistant turns each ({count * 6} user prompts).",
             "",
             "**These are simulated gold/reference conversations, not model transcripts.** Answers are taken verbatim "
             "from the frozen dataset and computed from the original native schematic evidence. They do not establish "
             "model accuracy, electrical correctness, or hardware safety. Each first prompt includes the extracted evidence "
             "shown below; these examples therefore demonstrate evidence-assisted tasks, not image-only reading.",
             "",
             "Images are unchanged copies of the corpus's circuit crops rendered from original EAGLE geometry. "
             "Adafruit attribution and source notices are retained. Original designs remain under CC BY-SA 3.0; "
             "each example links its exact source commit and license. The companion manifest records dataset, source, "
             "and image hashes.", ""]
    records = []
    for index, item in enumerate(prepared, 1):
        row = item["row"]
        image_target = images / item["image_name"]
        image_link = Path(os.path.relpath(image_target, output.parent)).as_posix()
        lines.extend([f"## {index}. {item['repository']}", "",
                      f"Conversation `{row['id']}` · dataset split `{row['split']}` · source commit `{item['commit']}`.", "",
                      f"[Original schematic]({item['source_url']}) · [CC BY-SA 3.0 license]({item['license_url']})", "",
                      f"![Original-source circuit crop for {item['repository']}]({quote(image_link, safe='/.-_')})", "",
                      "<details>", "<summary>Complete system instruction and original first-turn evidence prompt</summary>", "",
                      _fence(_text(row["messages"][0])), "", _fence(_text(row["messages"][1])), "", "</details>", ""])
        for turn in range(6):
            question = _text(row["messages"][1 + turn * 2])
            if turn == 0:
                question = question.rsplit("\n\n", 1)[-1]
            answer = _text(row["messages"][2 + turn * 2])
            lines.extend([f"**User {turn + 1}**", "", _fence(question), "",
                          f"**Assistant {turn + 1} — simulated gold**", "", _fence(answer), ""])
        records.append({"conversation_id": row["id"], "design_id": row["design_id"],
                        "family_id": row["family_id"], "split": row["split"], "turns": 6,
                        "source_url": item["source_url"], "source_commit": item["commit"],
                        "source_sha256": item["native_sha256"], "license": "CC-BY-SA-3.0",
                        "license_url": item["license_url"], "image": image_link,
                        "image_sha256": item["image_sha256"],
                        "render_method": row["provenance"]["render_method"],
                        "crop_policy": row["provenance"].get("crop_policy"),
                        "dialogue_sha256": digest(json.dumps(row["messages"], sort_keys=True).encode())})
    markdown = ("\n".join(lines).rstrip() + "\n").encode()
    report = {"version": 1, "kind": "simulated_gold_reference_not_model_outputs", "conversations": count,
              "distinct_board_families": len(families), "user_prompts": count * 6,
              "dataset_sha256": dataset_hashes, "markdown_sha256": digest(markdown), "records": records}
    # Validate every selected source before writing any output artifact.
    for item in prepared:
        _write(images / item["image_name"], item["image_bytes"])
    _write(manifest, (json.dumps(report, indent=2) + "\n").encode())
    _write(output, markdown)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=Path("docs/simulated_conversations.md"))
    parser.add_argument("--images", type=Path, default=Path("docs/examples/images"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/examples/manifest.json"))
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_examples(args.data, args.out, args.images, args.manifest,
                                 count=args.count, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "incomplete", "error": str(error)}))
        return 2
    print(json.dumps({"status": "complete", "conversations": result["conversations"],
                      "user_prompts": result["user_prompts"], "output": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
