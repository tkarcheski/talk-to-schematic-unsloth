<!-- cspell:words worktrees multiturn reflogs tmpfiles subtests -->

# Git recovery and development baseline — 2026-10-20

The primary checkout was detached at `f1ff646`, fourteen commits behind
`origin/main`. Eleven registered worktrees under `/tmp` had lost all tracked
files except `.gitignore`: 1,767 missing files in total. Their branch tips and
Git objects survived. `git fsck --full --no-reflogs` found no corrupt or missing
objects, and reported seventeen dangling trees. There were no stashes or open
GitHub pull requests at audit time.

## Preservation and recovery

- Archived the complete Git metadata, original instructions, worktree inventory,
  and remaining temporary-worktree contents under the private, ignored directory
  `tmp/git-recovery-20261020/` in the primary repository. This is a local recovery
  copy, not an independent off-machine backup.
- Inventoried the dangling trees and extracted sixteen blob versions not
  reachable from named refs into that archive. Historical drafts remain
  available for inspection; they were not blindly reapplied to newer code.
- Restored only missing tracked files from each worktree's index. All eleven
  worktrees then had clean status.
- Relocated those worktrees to the persistent sibling directory
  `../talk-to-schematic-worktrees/`. A cross-filesystem move required filesystem
  relocation followed by `git worktree repair`; every relocated checkout was
  verified clean. All branch names and tips were preserved.
- Created `codex/recovery-20261020` from `origin/main` at `f4a2bad` in the primary
  checkout. Updated its instructions for the user's autonomous recovery, model,
  and UI development authorization, while retaining product requirements and
  the documented SSH setup.

The host's `/usr/lib/tmpfiles.d/tmp.conf` specifies ten-day cleanup for `/tmp`.
That is consistent with the loss pattern, but no deletion log was examined
to establish the actual cause. Persistent worktree locations avoid
depending on the operating system's temporary-directory retention policy.

## Branch disposition

| Work | Finding and disposition |
|---|---|
| Application, viewer consent, research loop, multi-page datasets, multi-turn evaluation | Already present on current `main`, including squashed or rebased equivalents. Preserve historical branches; do not merge duplicate implementations. |
| `codex/agentic-integration` | Its final tree differs from current `main` only by added CI checks and absence of newer multi-turn files. Salvaged the CI checks alone, preserving the newer evaluation work. |
| Old backend, UI, routing, and dashboard branches | Intermediate feature implementations consolidated in the application/research commits. Retained intact for reference. |
| `codex/production-schematic-model` | The unmerged follow-up diagnosis is already covered, with more careful causal qualifications, in `docs/validation/v3-regression.md`. Retained rather than restoring obsolete status files. |
| `claude/docs` and the pending CI branch | Historical documentation and build changes have newer replacements on `main`. Retained; no wholesale replay of outdated code or deleted generators. |
| Dangling drafts | Preserved locally, including CLI, agent, evaluator, dashboard, tests, and documentation versions. Selected comparisons confirmed older implementations; not all drafts warrant integration. |

## Repairs on the recovered baseline

The salvaged CI configuration adds browser agent/navigation regressions and an
isolated gateway container job. All three Compose configurations validate
locally. Docker daemon access is denied for this account, so the image build
and container execution have not been verified locally.

UI interaction repairs prevent IME confirmation from submitting partial prompts
or navigating history, preserve focus when a response finishes, make approval
progress text accurate for both viewer and external actions, and distinguish
completed, stopped, and failed requests. Scripted browser tests cover these
behaviors. They do not establish real-model quality or a completed visual redesign.

Evaluation repairs reject affirmative measurement claims that previously passed
the refusal keyword check. Explicit absence of measured evidence can still pass
when an answer also discusses a nominal rail voltage. This remains a narrow
deterministic check, not semantic adjudication. Reports now separate attempted
failures from blocked dependent turns and retain HTTP status without saving
upstream error bodies, URLs, or headers. All turns remain in the denominator.

## Model and UI priorities

1. Diagnose the preserved multi-turn baseline's four HTTP 422 failures before
   launching another training run. Its 76/84 result consists of 76 passing turns,
   four unclassified serving/generation failures, and four blocked dependent
   turns. HTTP status alone cannot distinguish malformed model output,
   incomplete generation, context overflow, or another serving fault. Add safe
   typed error codes at the serving boundary and rerun the frozen cases.
2. Preserve the original adapter. The routing continuation candidate improves
   the total from 24/28 to 25/28 but regresses viewer selection from 7/7 to 4/7;
   it remains rejected. Test clean tool-turn termination and follow-up coverage
   with validation-selected checkpoints before considering promotion.
3. Refine the UI visually after these interaction repairs: simplify redundant
   labels and consolidate competing mobile layout rules, then inspect rendered
   desktop/mobile behavior while retaining viewing, source context, drafts,
   approval controls, metrics, and request traces.

No model weights, frozen datasets, or benchmark snapshots were changed during
recovery. No new training run, model promotion, or publication was performed.

## Verification

The final changes pass 487 CPU tests and 76 subtests, all twelve browser state
tests, and the scripted agent and navigation browser suites. Ruff passes, and
both the source distribution and wheel build successfully. All three Compose
configurations validate. Container execution remains limited as described above.
