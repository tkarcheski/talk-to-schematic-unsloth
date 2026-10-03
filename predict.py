"""Predict every gold-labelled turn with resumable, dataset-bound evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from evaluate import EvaluationError, load, load_gold
from schematic_model.inference import Client, InferenceError, convert_messages, image_fingerprints


def predict(dataset, output, model, client, *, resume=False, history="gold", max_tokens=1024):
    dataset, output = Path(dataset), Path(output)
    rows = load_gold(dataset)
    if history not in {"gold", "generated"} or max_tokens < 1:
        raise ValueError("History must be gold or generated; max_tokens must be positive")
    images = {}
    for row in rows:
        for reference, digest in image_fingerprints(row["messages"], dataset).items():
            if reference in images and images[reference] != digest:
                raise ValueError(f"Image bytes changed during prediction preflight: {reference}")
            images[reference] = digest
        convert_messages(row["messages"], dataset, image_hashes=images)
    binding = {"schema_version": 2, "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
               "images_sha256": images,
               "model": model, "base_url": client.base_url, "history": history,
               "max_tokens": max_tokens, "temperature": 0.0}
    metadata = output.with_suffix(output.suffix + ".meta.json")
    existing = {}
    if output.exists() or metadata.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume with the same dataset and settings")
        if not metadata.is_file() or json.loads(metadata.read_text()) != binding:
            raise ValueError("Resume metadata does not match dataset/images/model/settings")
        if output.exists() and output.stat().st_size:
            existing = load(output)
    expected = {f"{row['id']}#{i}" for row in rows for i in range(len(row["gold"]))}
    if set(existing) - expected:
        raise ValueError("Existing predictions contain keys outside this dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not metadata.exists():
        with metadata.open("x", encoding="utf-8") as handle:
            json.dump(binding, handle, indent=2)
            handle.write("\n")
    with output.open("a" if resume else "x", encoding="utf-8") as handle:
        for row in rows:
            messages = convert_messages(row["messages"], dataset, image_hashes=images)
            turn, conversation = 0, []
            for message in messages:
                if message["role"] != "assistant":
                    conversation.append(message)
                    continue
                key = f"{row['id']}#{turn}"
                if key in existing:
                    answer = existing[key]
                else:
                    result = client.complete(model, conversation, max_tokens=max_tokens)
                    answer = result["output"]
                    handle.write(json.dumps({"key": key, **result}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    existing[key] = answer
                conversation.append(message if history == "gold" else {"role": "assistant", "content": answer})
                turn += 1
    return {"conversations": len(rows), "predictions": len(existing), **binding}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--out", default="preds.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8888/v1")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--history", choices=["gold", "generated"], default="gold")
    args = parser.parse_args(argv)
    try:
        report = predict(args.data, args.out, args.model, Client(args.base_url, timeout=args.timeout),
                         resume=args.resume, history=args.history, max_tokens=args.max_tokens)
    except (EvaluationError, InferenceError, ValueError, OSError) as exc:
        print(f"Prediction incomplete: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
