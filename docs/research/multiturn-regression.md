<!-- cspell:words multiturn -->

# Multi-turn navigation regression

The first-action routing benchmark does not detect unsolicited navigation after
successful tool execution. `scripts.evaluate_multiturn` executes bounded offline
viewer tools, returns their results to the model, and scores the complete turn.
Web search returns an explicitly simulated empty result; it never uses the network.
This measures routing and turn termination, not electrical diagnosis quality.

The immutable local snapshot `tmp/agent-multiturn-v2` contains 281 training rows
(108 source-grounded conversations/tool-ending prefixes and 173 original replay
rows), 63 validation rows, and 21 held-out conversations with 84 user turns.
Seven validation IC values are absent from the training-family component pool.
Original family boundaries and original test data remain unchanged.

Value, review, and search prompts use ordinary wording without instructions to
keep the viewport still. Clearing UI focus preserves an unambiguous conversational
reference: a following search must use that part, not demand unnecessary
clarification. Genuine ambiguous-reference coverage remains separate future work.

**The preserved v1 snapshot has invalid clarification gold and is diagnostic only.
Do not train on it or use its aggregate score for qualification.** It incorrectly
assumed clearing focus erased the preceding conversational reference.

Reproduce v2 from the original project directory (run the checked-out modules):

```sh
python -m schematic_model.multiturn_data --corpus data/real-v3 \
  --replay data/training/real-crop-v1 --out tmp/agent-multiturn-v2
python -m scripts.evaluate_multiturn \
  --cases tmp/agent-multiturn-v2/rollout-cases.jsonl \
  --out results/agent-multiturn-v2-baseline \
  --endpoint http://127.0.0.1:8892/v1 --model schematic
```

The builder refuses existing outputs and verifies source/image hashes before
publishing. The local v2 case SHA-256 is
`7bc8bedb4f0dcb08ed8f07a18264b56c109d73585a90e2cea7f2ed31b6e5542e`.
Use the same cases and generation settings for baseline and candidate. Failed
or interrupted turns retain their denominator; dependent turns are marked blocked.
A repeated navigation call fails viewport invariance even if its box is unchanged.

`build_page_change_cases` builds a separate actual two-page PyPortal workflow:

```sh
python -m scripts.build_page_change_cases \
  --source 'data/real/sources/adafruit__Adafruit-PyPortal-PCB/Adafruit PyPortal.sch' \
  --metadata data/real/sources/adafruit__Adafruit-PyPortal-PCB/source.json \
  --out tmp/agent-page-change-v1
```

It pins native source, license, rendered images, and per-page component evidence.
Only valued components are selected; missing IC values are never inferred from
part names. The three training candidates are not merged into v2. PyPortal is an
existing training family, so this workflow is a training-family diagnostic and
must not be counted in the held-out score. Neither dataset supplies invented
measured voltages, operating specifications, faults, or model-generated answers.
