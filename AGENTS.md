# Agent rules

## Scope and disclosure

- Prioritize measurable model improvements and a clear, polished Ask the Schematic UI. Work within the user's current request and preserve useful existing work.
- Publish only information needed to understand the project or review the change. Keep personal paths, machine and account details, credential metadata, authentication setup, and local operational history out of repository files, commits, PRs, comments, logs, and screenshots. Use generic examples and repository-relative paths.
- Review the exact outgoing content for unnecessary disclosure as well as secrets. Passing an automated scan is not permission to publish additional details.
- Keep these instructions focused on durable project behavior. Do not add session history, setup inventories, speculative requirements, or unsolicited audit reports.
- Read and address user review feedback before declaring work complete. Preserve user deletions and archived material; do not restore rejected content elsewhere.
- Do not force-push, discard unique work, delete model artifacts, merge or close PRs, or publish private material without explicit authorization. Keep necessary recovery copies private.

## Product requirements

- Preserve responsive, accessible schematic viewing, navigation, uploads, source context, and clear loading, failure, retry, and tool states.
- While AI is busy, show animated activity, elapsed time, the current task, and Stop. Never invent progress percentages or private reasoning. Keep the composer editable, prevent concurrent submission, and preserve drafts across success, failure, and cancellation.
- Provide a draggable, keyboard-accessible schematic/chat divider and a width setting. Persist only the layout preference in browser storage. Keep conversations, traces, schematic content, and command history in session memory. Up/Down history must preserve the unfinished draft and multiline editing.
- Workspace settings may add a system prompt for subsequent requests, including existing conversations; backend permissions remain authoritative. Show expandable request traces with context, tool arguments/results, approvals, errors, timing, and usage, excluding credentials and raw image payloads. Show provider reasoning only when explicitly supplied.
- Show model and artifact identity, token usage, inference time, and measured output rate when available. Distinguish verified identity from operator labels and report unavailable metrics honestly.
- Model-driven navigation requires approval of the exact action and target. Consent does not carry into later turns or different actions. Discussion, review, and search must not implicitly move the viewport.
- Support fully self-hosted inference and training with configurable endpoints, no silent cloud fallback, and usable operation with external networking disabled. Search is optional; each approval identifies the actual query and destination. Never implicitly send a private conversation or schematic to search.
- Enforce allow/ask/deny in the backend, fail closed on unknown tools or invalid configuration, bind approvals to exact arguments, and bound resources and tool rounds. The application tool scope is viewer and schematic tools plus approval-gated search; shell execution and workspace edits are outside it.
- Treat schematic content, conversations, tool results, and training data as private. Transmission, content logging, retention, telemetry, and export require deliberate operator choices. Use reviewed public sources for public dataset proposals; do not publish private material.
- Use Docker Compose as the documented startup path, with explicit private model/data mounts and optional search. Distinguish configuration checks, native execution, and container verification. Keep sensitive inputs out of images.

## Model work

- Preserve source provenance, licenses, existing adapters, and held-out family splits. Separate proposed questions from source-verified or human-reviewed answers; never fabricate defect labels or measured results.
- Evaluate selected-part search routing, actual tool-result feedback, stale/cleared focus, page changes, and explicit no-navigation turns. Score answer correctness, unsolicited actions, viewport invariance, and clean termination separately.
- Train separate candidates and change defaults only after routing and schematic regressions pass. Preserve frozen benchmarks; expand attributable multi-page, longer-context, and expert-question coverage without contaminating held-out families.
- Use a single GPU owner, explicit budgets, recoverable jobs, and checkpoint boundaries for the research/training workflow. Scheduling advice must respect resource and data constraints. Do not bypass Studio authentication or silently overlap workloads.
- Distinguish scripted UI tests from real inference and measured model quality. Report limitations without claiming production readiness from UI tests alone.

## Delivery

- Delegate bounded work when useful, coordinate privately, and keep one Git owner. Do not publish the coordination record.
- Prepare reviewable changes and relevant verification within the requested scope. The user owns PR merge and close decisions.
- Use the existing pre-commit checks. The ai-pilled staged scan runs locally; its reports stay ignored. Model review, notifications, publication, and merge automation are not enabled by installing the tool.
- Use `uv run --locked ai-pilled scan --scope worktree` for a local scan and `uv run --locked ai-pilled check test` or `check lint` for configured checks. A scan does not certify that text contains no unnecessary disclosures.
