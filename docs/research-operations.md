<!-- cspell:words finetune -->
# Research and training operations

The local scheduler runs reviewed research/training adapters serially. It uses a private SQLite ledger, a process lock, GPU inventory, finite budgets, and cooperative checkpoint yields. It does not start an unbounded worker. The existing interactive model must be handed off before a scheduler-owned GPU job can start; the scheduler never kills someone else's model.

```sh
python scripts/research_scheduler.py --ledger results/jobs.sqlite3 submit research public-research
python scripts/research_scheduler.py --ledger results/jobs.sqlite3 --config /absolute/operator-config.json run-once
python scripts/research_scheduler.py --ledger results/jobs.sqlite3 status
python scripts/training_dashboard.py --run outputs/real-crop-v1-r16-e3 --ledger results/jobs.sqlite3 --port 8890
```

Dashboard: `http://127.0.0.1:8890`. Read-only, loopback-only, with training metrics, GPU process owners, queued/running/blocked/completed/failed states, handoff reasons, elapsed work, checkpoint steps and source/proposal counts when an adapter reports them. No raw prompts, tool content, or child stdout/stderr enter the ledger. Child failures expose exit status; detailed sensitive diagnostics require an explicit operator-run adapter invocation.

Example operator configuration:

```json
{
  "budgets": {"research": 300, "training": 1800, "proposal": 120},
  "reserve_mb": 2048,
  "allow_proposals": false,
  "adapters": {
    "public-research": {
      "argv": ["/absolute/unsloth/python", "-m", "scripts.research_agent_step", "--plan", "/absolute/public-research.json", "--output", "/absolute/review/candidates.json"],
      "cwd": "/absolute/project",
      "kind": "research", "ready": true, "gpu_mb": 16000, "cooperative": true
    },
    "train-checkpoint": {
      "argv": ["/absolute/unsloth/python", "-m", "scripts.training_slice", "--plan", "/absolute/train-plan.json"],
      "cwd": "/absolute/project",
      "kind": "training", "ready": false, "gpu_mb": 18000, "cooperative": true
    }
  }
}
```

Set training ready only after verifying the integrated training deadline callback and data/model fit. Training plan fields: absolute `data`, `out`, `model`; positive bounded `max_steps`, `save_steps`, `max_seq`, `rank`, `gradient_accumulation`; optional boolean `continue_adapter` (the `model` path is the existing adapter), `lr` (default 5e-5), `max_image_size` (default 1024), and `finetune_vision` (default false). The wrapper binds the plan and train/validation file hashes to a sibling marker outside its output directory, resumes the latest checkpoint from that same plan, and refuses unrelated existing output. Dataset images and model weights still require operator immutability. Use a fresh output for a changed plan.

Research plan:

```json
{
  "public_topic": "Adafruit open hardware multi-page schematics with a published license",
  "backend": "native-owned",
  "model": "/absolute/local-model-or-adapter",
  "model_endpoint": "http://127.0.0.1:8892",
  "search_endpoint": "http://127.0.0.1:8894",
  "allowed_hosts": ["learn.adafruit.com", "github.com", "cdn-learn.adafruit.com"],
  "allow_public_search": true
}
```

`native-owned` loads a local model via the Unsloth library inside the short-lived scheduler child, makes one bounded query, writes a candidate proposal, then exits and releases residency. It does not start a server or stop other processes. `model_endpoint` is unused for this backend. `native-unsloth` instead uses an already-resident loopback compatible endpoint during an explicitly coordinated handoff. `studio` uses the installed Studio endpoint with `UNSLOTH_API_KEY`; specify adapter `credential_env: "UNSLOTH_API_KEY"` to forward only that named secret. Studio status must identify the exact resident model and show no loading. No backend silently falls back to external AI.

`research_agent_step` uses only an operator-reviewed public topic, obtains a short query from the local model, calls the configured loopback SearXNG service under explicit public-search policy, and retains allowlisted candidate URLs. Search still contacts the public internet. It does not send private schematic/chat history. Candidate licenses remain unknown and require review; discovery is not proof of multi-page coverage or suitability for training.

Verified on 2026-10-07: one real research call through native Unsloth on port 8892 and SearXNG on 8894 completed and produced six Adafruit/GitHub candidates. The model query was `Adafruit open hardware multi-page schematic PDF published open hardware license`. Every candidate remained review-required, `ready_for_training: false`. Studio source/routes were inspected, but authenticated live Studio access was unavailable (401). Native-owned loading and training time-share lifecycle are implemented but were not GPU-tested in this task; the GPU was handed to the separate training evaluation.

After license/source review, `python -m scripts.research_step --plan PLAN --output DIRECTORY` downloads up to eight reviewed sources. Plan fields are `allowed_hosts` and `sources`; each source has `url`, `license`, `license_url`, `public: true`. Downloads pin public DNS, refuse redirects/query strings and unapproved hosts, limit size, preserve hashes, and resume already verified sources across slices. A listed license is an operator assertion: read the actual license and verify redistribution rights before collection/publication. This adapter creates source proposals; PDF extraction/example generation/evaluation are separate work.

Explicit publication adapter:

```sh
python -m scripts.dataset_proposal_pr --repo /absolute/public-worktree --manifest /absolute/public-worktree/data/public-proposals/review/proposal.json --branch research/reviewed-sources --github-repo owner/repository
```

Publication requires `publication_approved: true`, every source `review_status: "approved-for-publication"`, reviewed license/provenance, matching hashes, clean tracked index/worktree, exact approved public `origin/main` HEAD, and matching explicit GitHub origin. Only reviewed manifest/source paths are staged; staged bytes are rehashed; commit ancestry is checked. The adapter opens a **draft** PR and never merges/closes it. It is not run by default; scheduler proposal jobs require `allow_proposals: true`. Review all source bytes for private content: public flags cannot prove classification. Git or publication failure leaves reviewable local state, never automatic reset/cleanup. No actual PR was created by this adapter during verification.

Default selection balances accumulated runtime divided by per-class budgets, skips GPU-blocked jobs when CPU work is runnable, and retries resource blocks on the next run-once. Optional config `advisor: {"enabled": true, "endpoint": "http://127.0.0.1:8888", "model": "EXACT_RESIDENT_ID"}` asks authenticated local Studio to select only among already eligible opaque job IDs. It cannot introduce commands or bypass gates. The advisor requires an available resident model; automatic advisor model loading/switching is intentionally absent. Deterministic fallback is used for a mismatched recommendation, while endpoint/auth failures stop dispatch visibly.

Adapters receive `SCHEMATIC_DEADLINE` (Unix epoch seconds), `SCHEMATIC_BUDGET_SECONDS`, `SCHEMATIC_JOB_ID`, and private `SCHEMATIC_RESULT_PATH`. Return 75 after checkpoint yield, 0 on completion, other codes on failure. Result JSON accepts only bounded integer `checkpoint_step`, `source_count`, `proposal_count`; no raw error content. Budgets are cooperative, not destructive hard timeouts: training stops at optimizer boundaries; native model loading/generation cannot be interrupted safely mid-operation. A late adapter is blocked after it returns. A stale running row blocks new work until the operator verifies child/GPU state.

All GPU controllers must honor handoff: inventory checks are not an OS-level reservation. CPU adapters must not load GPU models. Configured adapters execute with operator privileges and are not a sandbox. Keep configuration, ledger, artifacts, and credentials private; never point an adapter at arbitrary model-authored shell commands. Studio resident inference and scheduler-owned training cannot automatically share the same occupied GPU without an explicit lifecycle handoff.

Use dashboard `--follow` to display the newest training manifest under the selected run's parent directory, including new fit/candidate runs. This is filesystem-root-bounded discovery, not an arbitrary HTTP path selector.

For explicitly approved baseline desktop/Studio residents, scheduler configuration may contain `co_residents` entries with exact `pid` (integer), `/proc/PID/stat` field22 `starttime` (string), `/proc/PID/exe` `exe` path, and `max_mb` (integer). The GPU probe permits only these exact identities within each declared memory cap; new/stale processes and growth beyond a cap block the job. Available memory must still cover the job plus reserve. Never infer consent from low memory use. Baselines are machine-session-specific and must be approved again after restart. Root approved the current three baseline resident processes during this session; no universal baseline is shipped.

A finite worker can dispatch multiple slices and wait briefly for an operator handoff:

```sh
python scripts/research_scheduler.py --ledger results/jobs.sqlite3 --config /absolute/operator-config.json worker --max-jobs 4 --max-idle-polls 3 --idle-seconds 10
```

`--max-jobs` counts executed slices, including checkpoint yields; it is limited to 100. Idle/resource-blocked polling is separately bounded (at most 100 waits, each at most 60 seconds). No daemon or infinite retry starts implicitly. Cooperative workload duration still follows the adapter deadline contract.

While a `native-owned` research child already owns its model, it may make one additional bounded inference selecting a next job from scheduler-supplied opaque IDs/kinds. It writes only an exact matching ID to the private result file. The scheduler persists that recommendation in its SQLite ledger and consumes it on the next dispatch **after rechecking readiness, permissions, GPU identities, and memory**. Invalid or no-longer-eligible choices fall back to deterministic ordering. This avoids loading a separate advisor during training and never lets the model introduce an executable or read private training data. The research child ends normally, releasing its model; any delayed GPU release blocks subsequent dispatch until inventory is clear. Mocked tests cover this lifecycle/advice contract; live owned-model loading still requires an explicit GPU handoff and validation.

Always invoke research/training modules from the project root with `python -m scripts.research_agent_step` and `python -m scripts.training_slice`; this makes the repository package importable even when the managed Studio Python environment does not have the project installed. CPU tests launch these real module help commands and import the deployment engine in a subprocess. Wrapper failures report only a fixed code (`invalid_configuration`, `io_failure`, `inference_failure`, `deadline_reached`, or `unexpected_failure`) through the private result file; no exception message is persisted.

Further live verification on 2026-10-07: the operator's training run yielded a complete checkpoint at step 26/68 after approximately 123 seconds, then released GPU memory. A scheduler-dispatched `native-owned` research child completed in 33.48 seconds, recorded six candidates and one proposal, and its GPU PID disappeared after normal exit. Training subsequently resumed checkpoint 26 with a 300-second deadline. An earlier import-failure job remains honestly failed in the ledger. This verifies the research ownership lifecycle and the trainer's yield/release/resume path; the `training_slice` wrapper itself has not yet been exercised with a live GPU run. Finite worker and persisted advisory selection have CPU/mock coverage, not live model validation.
