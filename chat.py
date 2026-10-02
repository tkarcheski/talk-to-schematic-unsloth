"""
chat.py - talk to a schematic with your fine-tuned model.

  vllm serve outputs/hw-lora-merged --served-model-name hw --max-model-len 32768
  python chat.py path/to/schematic.png [--ref parts.txt]

--ref is optional datasheet/parts text (the same "Reference data" block used in
training). In production, fill it from your parts DB for the parts on the sheet.
"""
import argparse
import base64
import mimetypes
import re

from openai import OpenAI

from build_dataset import SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--ref", help="text file with datasheet/parts reference lines")
ap.add_argument("--model", default="hw")
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--show-thinking", action="store_true")
args = ap.parse_args()

client = OpenAI(base_url=args.base_url, api_key="local")
mime = mimetypes.guess_type(args.image)[0] or "image/png"
img = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(open(args.image, 'rb').read()).decode()}"}}
ref = open(args.ref).read().strip() if args.ref else ""
history = [{"role": "system", "content": SYSTEM}]
print("Ask about the schematic (blank line to quit). Try: 'Review this schematic for problems.'")
while True:
    q = input("\nyou> ").strip()
    if not q:
        break
    if len(history) == 1:   # first turn carries the image and reference data
        text = (f"Reference data:\n{ref}\n\n" if ref else "") + q
        history.append({"role": "user", "content": [img, {"type": "text", "text": text}]})
    else:
        history.append({"role": "user", "content": q})
    out = client.chat.completions.create(model=args.model, messages=history, temperature=0.2, max_tokens=4000)
    full = out.choices[0].message.content
    answer = re.sub(r"<think>.*?</think>\s*", "", full, flags=re.S)
    if args.show_thinking and answer != full:
        print("\n(thinking)\n" + re.search(r"<think>(.*?)</think>", full, re.S).group(1).strip())
    print("\nassistant> " + answer)
    history.append({"role": "assistant", "content": answer})   # drop reasoning from history, as in training
