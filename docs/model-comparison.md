# Compare answers on one schematic

Open a schematic in Ask the Schematic, then choose **Compare models**. The
comparison captures that page image, evidence mode and viewer context. Both
models receive the same question, temperature and answer budget. Requests run
in sequence; the comparison never moves the schematic or executes tools/search.
Normal chat and its history remain separate. Matching image bytes, temperature
and token budgets does not establish preprocessing parity: endpoints may resize
images, tokenize them or apply chat templates differently. Inspect those settings
separately before attributing a response difference to training.

Choose **Identical history for both** to isolate how models answer the same
input. After each pair, explicitly select A or B's answer as shared context for
the next turn. This is a context choice, not a correctness vote or a gold label.
Choose **Each model's own answers** to observe how independent conversations
develop. A failed pair commits neither history and keeps successful output
visible; retrying reruns the pair from the last complete turn. Source/model/history
changes require **New comparison**. Workspace instructions apply to subsequent
pairs, including existing conversations.

Per-answer details show the actual configured model, endpoint-reported artifact
identity, available usage/timing, and redacted request/response traces. A profile
label is an operator label, not proof of weights. If both endpoints report the
same artifact, the UI identifies that limitation. No quality score is inferred
from response speed or text length.

The composer stays editable during generation. **Stop** cancels browser waiting
and prevents the next lane from starting; it cannot forcibly cancel an upstream
model. Wait for the service to become idle before retrying. Content is held only
in session memory. Nothing is exported, retained as training data or published.
Each comparison is bounded to 20 attempts.

## Operator configuration

The gateway accepts an optional private JSON configuration. Example:

```json
{
  "max_tokens": 1024,
  "profiles": [
    {"id": "baseline", "label": "Baseline", "endpoint": "http://127.0.0.1:8892/v1", "model": "baseline"},
    {"id": "candidate", "label": "Candidate", "endpoint": "http://127.0.0.1:8894/v1", "model": "candidate"}
  ]
}
```

Endpoints must be explicitly configured literal private/loopback IP addresses.
Browser requests cannot change endpoints, credentials, models or token budgets.
Each endpoint must already serve its configured model and return that exact
model name. Use upstream inference endpoints, not the comparison gateway itself;
recursive requests are rejected by its generation lock. Optional profile fields are `api_key_env` (a server environment
variable containing the credential) and `expected_adapter_sha256` (require the
endpoint's adapter hash to match). Redirects and environment proxies are disabled.
The gateway sends the frozen schematic/context only to these configured services;
the operator controls their content retention and any external networking.

Use the Compose startup path with a private configuration file outside the image:

```sh
export SCHEMATIC_COMPARISON_CONFIG=/private/config/comparison.json
docker compose -f compose.yaml -f compose.comparison.yaml up --build
```

The standard Compose inference service serves only one model. This override does
**not** start a second model, download weights, change the default adapter or
manage residency. Its network remains private. Configure reachable services
within that network, or an explicitly managed local backend that already serves
both model IDs. If authentication is required, provide the named environment
variables to the gateway through a private operator configuration.

There is one generation lock for this gateway's chat and comparison requests.
That is not a reservation across independent servers or other GPU jobs. On a
single GPU, do not start two resident native engines that exceed the resource
budget. Sequential model/adapter residency needs an independently verified
lifecycle; this feature does not implement it. Unavailable endpoints fail
visibly and there is no alternate/cloud fallback.

CPU and scripted-browser tests validate transport, isolation and interaction.
They do not establish that distinct real models fit together or that a candidate
is better. Correctness annotations, review queues and training-feedback export
are separate follow-up work for [issue #25](https://github.com/tkarcheski/talk-to-schematic-unsloth/issues/25).


## Review saved checkpoint answers

Choose **Compare models → Saved benchmarks**. This works with no loaded model,
no open schematic and no live comparison configuration. Select the exact benchmark
dataset JSONL and, for each run, all three files produced by
`python -m schematic_model.benchmark`: `run.jsonl`, `run.jsonl.meta.json` and
`run.jsonl.report.json`. The files must retain their matching sidecar names.
Optional A/B operator labels (for example, v1 and v3) make the lanes easier to
recognize; the recorded adapter hashes remain visible. Labels are not verified
model identities. Files are read in the browser; no content is uploaded, stored in browser storage,
published or converted into training labels. Reloading the page clears them.
Use HTTPS or localhost for browser SHA-256 support.

Import requires version 1 failure-accounted benchmark reports, identical dataset
and image hashes, history mode and generation settings, and exact turn-key
coverage. Prediction bytes must match the report hash, and the report provenance
must equal the metadata. Duplicate JSON fields, repeated/missing/unknown turn
keys, contradictory generated-history failures, incomplete successes and partial
text in scored failure outputs are rejected. A failed import preserves the last
valid comparison. Limits are 16 MiB per file, 64 MiB total, 5000 turns and 64 turns
per conversation. These checks establish file consistency; they do not authenticate
a report or prove that inference used the declared model/preprocessing.

Browse real questions with **All turns**, **Any failed or blocked turn**, either
lane's exclusive failures, or **Different complete text**. Each pair includes the
dataset reference answer, recorded finish reason, available tokens/time/rate and
expandable prior context. Gold history uses the identical dataset reference
answers. Generated history shows each run's own earlier answers, including
blocked follow-ups after a failure. Neither mode imports anything into live chat.

Scores are the report's **aggregate recorded metrics**, with denominators. The
browser does not recompute semantic correctness or assign per-answer verdicts.
A text difference may be equally correct. A recorded stop does not establish
correctness or clean termination; a response may still contain spurious role
markers or other malformed text.
Artifact hashes are recorded identities; identical adapters are flagged. Base
model/runtime differences are flagged because they prevent attribution to
training alone. Raw model/source paths and raw error messages are not displayed.
A failed generation has no scored answer. Supplemental partial-output sidecars
are not imported: the current format does not bind them to the report hash, so
attaching a partial to a model would imply evidence that the files do not supply.

**Inspect source image** accepts deliberately selected original PNG files, up to
16 MiB, 40 megapixels and 32768 pixels on either axis per image. Choose multiple
files once to browse their matching conversations without picking a file for each
question. Each image must match a hash recorded for the dataset. The session cache
is bounded to 32 images, 64 MiB of source bytes and 80 megapixels in total; matching duplicates do not
consume extra cache entries. A rejected batch preserves existing previews.
**Clear previews** releases them. Dataset paths are never fetched or resolved, and an open
schematic is never substituted for the recorded image. Without a selected match,
the view explicitly says the source pixels have not been verified in this
session. The original image preview does not establish how readable the model's resized input was.

Run `npm run test:saved-benchmark` for CPU integrity tests and
`npm run test:browser-saved` for scripted desktop/mobile browser checks. The latter
uses artificial responses and does not measure a model. To review existing local
predictions in the same browser check, deliberately set `SCHEMATIC_SAVED_DATASET`,
`SCHEMATIC_SAVED_A` and `SCHEMATIC_SAVED_B` to the dataset and two prediction files.
Their matching sidecars are required. Optional `SCHEMATIC_SAVED_LABEL_A` and
`SCHEMATIC_SAVED_LABEL_B` provide operator labels for that inspection. `SCHEMATIC_SAVED_IMAGES` optionally
accepts a JSON array of explicitly selected original PNG paths for hash-checked
source previews. The test does not resolve paths found in dataset content. Screenshots
stay in the local test-output
directory; review them for private content before any publication. Reading such
existing predictions is distinct from running new inference or re-scoring them.
