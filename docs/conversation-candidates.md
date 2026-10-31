# Compare standalone and conversation training

Build a separate pair of candidates from existing four-question, image-only source rows:

```sh
python -m schematic_model.conversation_candidates \
  --train-source data/training/source/train.jsonl \
  --validation-source data/training/source/val.jsonl \
  --out outputs/conversation-pair --families 24 --accumulation 4
```

Both arms contain six target questions per selected source: two component values,
a return to the first component, a corrected reference to the second, and two
existing refusal questions. Standalone examples state the component explicitly.
Conversation examples contain progressively longer histories. Each pair has the
same target answer and image; earlier answers are context, not additional labels.

Train **both** with `--last-response-only`, the same starting adapter, seed,
learning rate, accumulation, image size, epochs, and step limit. This is an opt-in
training setting recorded in the manifest and checked on resume. Existing
training behavior is unchanged without the flag. The installed Unsloth collator
must support `last_response_only`; an unsupported version fails instead of
silently changing supervision.

This construction matters for chat templates that insert empty thinking tokens
only into the last answer. Supervising every answer in a six-turn conversation
does not necessarily match six single-turn examples, even with identical answer
text. Prefix examples put every supervised target in the same template position
and give both arms the same number of image presentations. Still inspect actual
processor/collator labels, termination tokens, sequence lengths, and total
supervised tokens before training; matching text alone is insufficient.

The builder verifies source image hashes, inherits source labels without
claiming new source review, rejects overlapping training/validation families and
identities, and copies a private immutable input snapshot into a fresh ignored
directory. Both arms retain the same frozen validation conversations. The
manifest records source hashes, selected rows, answer/label inventories, and
recommended budgets. Context length and prior-answer exposure deliberately
differ; generated-answer evaluation is needed to establish their effect.

For final-answer-only validation loss, the final validation answer receives loss;
evaluate the full frozen conversations separately to measure values, refusals,
follow-ups, and completion. The builder does not run training or qualify a model.
