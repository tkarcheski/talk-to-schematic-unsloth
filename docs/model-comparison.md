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
is better. Saved benchmark import, correctness annotations, review queues and
training-feedback export are separate follow-up work for [issue #25](https://github.com/tkarcheski/talk-to-schematic-unsloth/issues/25).
