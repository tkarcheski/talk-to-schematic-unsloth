"""Download a revision-pinned model into this project, without loading it."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--revision", default="3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636")
    parser.add_argument("--out", type=Path, default=Path("models/Qwen3.5-4B"))
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(args.model, revision=args.revision)
    if info.sha != args.revision:
        parser.error("revision must be an immutable commit hash")
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model, revision=args.revision, local_dir=args.out)
    provenance = {"model_id": args.model, "revision": info.sha, "format": "safetensors"}
    (args.out / "download-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
