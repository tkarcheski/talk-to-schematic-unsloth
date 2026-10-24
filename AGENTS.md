# Agent rules

## Scope and disclosure

- Prioritize measurable model improvements and a clear, polished Ask the Schematic UI. Work within the user's current request and preserve useful existing work.
- Finish the current deliverable before starting another speculative feature. Inspect existing branches, PRs, and worktrees before creating replacements. Reuse accepted implementation; do not rebuild it in another checkout.
- Publish only information needed to understand the project or review the change. Keep personal paths, machine and account details, credential metadata, authentication setup, and local operational history out of repository files, commits, PRs, comments, logs, and screenshots. Use generic examples and repository-relative paths.
- Review the exact outgoing content for unnecessary disclosure as well as secrets. Passing an automated scan is not permission to publish additional details.
- Keep these instructions focused on durable project behavior. Do not add session history, setup inventories, speculative requirements, or unsolicited audit reports.
- Read and address user review feedback before declaring work complete. Preserve user deletions and archived material; do not restore rejected content elsewhere.
- Do not force-push, delete model artifacts, merge or close PRs, or publish private material without explicit authorization. A cleanup request permits removing verified merged, duplicated, or remotely preserved development checkouts. Identify unique source and experimental evidence before removal; keep only necessary, compact private recovery records, not copies of entire environments.

## Workspace lifecycle

- The primary checkout stays on clean, current `main`. Keep private datasets, adapters, experiment evidence, and service state in its ignored artifact directories or explicit operator-selected storage; never make a disposable worktree their only copy.
- Allow at most two linked development worktrees for the repository. Reuse them across PRs. Use the persistent sibling worktree directory, not `/tmp`, for ongoing development. Disposable audit clones must be removed in `finally`, including on failure or cancellation.
- Before editing a linked checkout, claim it with `python scripts/check_worktrees.py claim --owner <agent> --reference '#<issue-or-pr>' --hours 24`. Claims expire within 72 hours and identify the exact branch. Renew only while work is active. Run `python scripts/check_worktrees.py check` at session start and before push; resolve failures rather than weakening the gate.
- Coordinate privately with one Git owner per shared index. Use explicit file and GPU handoffs. Do not use inactivity as permission to overwrite an active owner's changes. A lease records responsibility; it does not grant runtime or publication permission.
- Each change ends in one of three states: merged; a pushed, reviewable PR with a clear next decision; or deliberately rejected with unique work accounted for. A submitted PR does not justify retaining its working copy. Remove its checkout and reproducible caches once its remote commit and retained artifacts are verified; recreate it only for actual review changes.
- Remove obsolete local branches after proving merge, patch equivalence, or a preserved remote/recovery reference. Recheck dirty state and active process references immediately before removal. Never run blanket `git clean`, delete model/data directories, or leave indefinite backup worktrees.
- A handoff includes the PR and exact head, evidence, unresolved blockers, and the resulting local worktree inventory. Do not claim completion while abandoned checkouts, source not pushed, or stale claims remain.

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

- Delegate bounded work when useful; share an existing checkout with non-overlapping file ownership rather than allocating a worktree per agent. Do not publish coordination records.
- Keep each PR focused on one usable outcome and its tests. Prefer independent PRs; declare unavoidable base dependencies and test the combined result. Do not split validation from the behavior it validates merely to reduce a diff count.
- Before recommending merge, read user feedback, inspect the complete diff against its actual base, and verify the candidate with current `main`. Fix concrete audit findings. Separate implemented behavior, fixture tests, real execution, and unverified claims in the PR; test counts alone are not acceptance evidence.
- Run the configured ai-pilled audits automatically through hooks, CI, and the trusted post-push worker. The committed-head audit must fail on failed, missing, unavailable, or stale checks. Pre-push and CI run deterministic checks; the local worker adds independent correctness, security, and maintenance model reviews before publishing the required merge status. A failing model audit may be published as a PR for review, but cannot qualify for merge. Never disable a merge gate to ship, substitute an old report, or mark an unexecuted check successful.
- Only publish the generic status and exact commit identity from an audit. Reports, prompts, source excerpts, machine details, and private artifacts stay ignored/local. Audit code runs only for trusted internal changes; external PRs must not execute on a credential-bearing workstation.
- GitHub must require the configured CI audit, current-head model-audit status, an up-to-date base, and resolved conversations. The user retains merge/close decisions. Automated audits must never merge, approve their own work, enable auto-merge, or discard failed changes to manufacture a passing result.
- Use `uv run --locked python scripts/audit_repository.py` for deterministic checks, `--model-reviews` for local committed-head review, and `--publish-status` to publish the bounded result after the commit exists remotely. Read `docs/quality-gates.md` for the exact audit contract and limitations. A passing scan or model review is evidence, not a guarantee of correctness.
