"""
predict.py - answer every turn of every test conversation through an
OpenAI-compatible server (vLLM / SGLang). Earlier turns use the gold answers
(teacher forcing) so each turn is graded independently.

  vllm serve outputs/hw-lora-merged --served-model-name hw --max-model-len 32768
  python predict.py --model hw --out preds_ft.jsonl
"""
import argparse
import base64
import json
import mimetypes
import re

from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/test.jsonl")
ap.add_argument("--model", default="hw")
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--out", default="preds.jsonl")
ap.add_argument("--max-tokens", type=int, default=4000)
args = ap.parse_args()
client = OpenAI(base_url=args.base_url, api_key="local")


def to_openai(content):
    out = []
    for c in content:
        if c["type"] == "image":
            mime = mimetypes.guess_type(c["image"])[0] or "image/png"
            b64 = base64.b64encode(open(c["image"], "rb").read()).decode()
            out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        else:
            out.append({"type": "text", "text": re.sub(r"<think>.*?</think>\s*", "", c["text"], flags=re.S)})
    return out


with open(args.out, "w") as fo:
    for line in open(args.data):
        r = json.loads(line)
        msgs = [{"role": m["role"], "content": to_openai(m["content"])} for m in r["messages"]]
        turn = 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            resp = client.chat.completions.create(model=args.model, messages=msgs[:i],
                                                  temperature=0.0, max_tokens=args.max_tokens)
            fo.write(json.dumps({"key": f"{r['id']}#{turn}", "output": resp.choices[0].message.content}) + "\n")
            turn += 1
        print(r["id"])
