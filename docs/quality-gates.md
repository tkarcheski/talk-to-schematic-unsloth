# Automated delivery audits

Every push and pull request runs the required GitHub check `audit`. It runs the
configured ai-pilled security scan, CPU tests with coverage, lint, type checking,
dead-code checks, Python and npm vulnerability audits, then spelling, formatting,
configuration validation, browser state tests, and package build. The CPU suite
runs once under coverage; the coverage gate reads that run's data. Missing tools,
unavailable advisory services, incomplete evidence, and failing checks block.

Python advisory coverage includes every PyPI package in `uv.lock`, including
transitive development and audit-tool dependencies. The local project and the
explicitly pinned ai-pilled Git source have no registry advisory record and are
excluded from that query. Any other source, conflicting versions, or changed lock
blocks until audit support is reviewed. npm auditing includes development and
optional dependencies. These gates cover repository locks, not separately managed
GPU environments. Registry queries send public dependency names and versions.

Install the repository hooks after `uv sync --locked` and `npm ci --ignore-scripts`:

```sh
uv run --locked python -m pre_commit install
```

Commits retain staged checks and model review. Before a push, the hook requires a
clean committed checkout and runs the full deterministic audit. Push one proposal
branch at a time. A passing hook allows publishing a reviewable PR; it does not
create the required remote model-review status or establish merge eligibility.

The independent required status `ai-pilled/model-audit` is published by the trusted
local audit worker after the commit exists on GitHub. The worker checks changed
PR heads hourly and runs three isolated model perspectives: correctness, security,
and maintenance. Each review must finish and pass against the same source snapshot:

```sh
uv run --locked python scripts/audit_repository.py --publish-status
```

This command verifies the repository origin and exact commit, posts a pending
status, runs all deterministic gates and all three model reviews, checks that
source and commit are unchanged, and only then posts success. Defects publish
failure; missing evidence publishes error. Publication failure also returns a
nonzero exit. The worker must use a clean checkout and remove it after the audit;
it must not execute untrusted contribution commands with access to credentials.
Only commit identity and generic status text are published. Reports and findings
stay under ignored `.ai-pilled/`; do not upload them as CI artifacts or comments.

GitHub-hosted CI receives no model credentials and does not claim a model review.
Without the trusted worker's passing status for the current head, merging remains
blocked. PRs with failing model audits remain available for review and fixes; they
are never merge eligible. Neither an audit pass nor a model review grants merge
permission or establishes measured schematic-model quality. Human review remains
required.

For a local deterministic audit use `uv run --locked python
scripts/audit_repository.py`; add `--model-reviews` to require the model
perspectives without publishing. Commit source changes first. Inspect private
findings through `uv run --locked python -m ai_pilled dashboard`.
