"""Ask a local deployed vision model about a schematic image."""

import argparse
import json
from pathlib import Path
import sys

from schematic_model.inference import Client, InferenceError, SYSTEM, image_content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--ref", type=Path, help="verified parts or netlist evidence")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8888/v1")
    parser.add_argument("--save", type=Path, help="save a text transcript (exclusive creation)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--question", help="answer one question, then exit")
    args = parser.parse_args(argv)
    try:
        image = image_content(args.image)
        reference = args.ref.read_text() if args.ref else ""
        client = Client(args.base_url)
        if args.save and args.save.exists():
            raise ValueError("Transcript already exists; choose another path")
        history = [{"role": "system", "content": SYSTEM}]
        transcript = {"model": args.model, "image": str(args.image), "turns": []}
        while True:
            try:
                question = args.question or input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            content = question
            if len(history) == 1:
                content = [image, {"type": "text", "text": (f"Reference evidence:\n{reference}\n\n" if reference else "") + question}]
            candidate = history + [{"role": "user", "content": content}]
            try:
                result = client.complete(args.model, candidate, max_tokens=args.max_tokens)
            except InferenceError as exc:
                print(str(exc), file=sys.stderr)
                if args.question:
                    return 2
                continue
            print("assistant> " + result["output"])
            history = candidate + [{"role": "assistant", "content": result["output"]}]
            transcript["turns"].append({"question": question, **result})
            if args.question:
                break
        if args.save:
            with args.save.open("x", encoding="utf-8") as handle:
                json.dump(transcript, handle, indent=2)
                handle.write("\n")
    except (ValueError, OSError) as exc:
        print(f"Chat unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
