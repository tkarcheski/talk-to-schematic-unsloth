"""make_mock_preds.py - smoke-test evaluate.py without a GPU.
  perfect : echoes the gold answers
  naive   : an overconfident model - says 'looks fine', invents values for missing parts
  python make_mock_preds.py perfect > p.jsonl
"""
import json
import random
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "perfect"
rng = random.Random(0)
for line in open("data/test.jsonl"):
    r = json.loads(line)
    answers = [c["text"] for m in r["messages"] if m["role"] == "assistant" for c in m["content"]]
    for i, (a, g) in enumerate(zip(answers, r["gold"])):
        if mode == "naive":
            if g["type"] == "codes":
                a = "The design looks fine to me."
            elif g["type"] == "refuse":
                a = "It is a 10k resistor, 1%, connected to U2.VDD."
            elif g["type"] == "pins" and g["pins"] and rng.random() < 0.4:
                a = "It connects to " + ", ".join(g["pins"][:-1] or ["U1.VIN"]) + "."
        print(json.dumps({"key": f"{r['id']}#{i}", "output": a}))
