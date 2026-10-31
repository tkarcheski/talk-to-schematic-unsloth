# Private experiment memory

<!-- cspell:words opencode frontmatter -->

The recorder keeps exact commands and evidence local so an agent can distinguish
what was proposed, attempted, completed, or changed. It complements the existing
research scheduler: the scheduler owns resource scheduling; this recorder owns
explicit command evidence. It does not start a model, choose a provider, run a
recorded command automatically, or replace the GPU ownership rules.

From the project root, explicitly opt in to recording a bounded CPU command:

```sh
python -m schematic_model.experiment_memory run --timeout 60 \
  --input tests/test_training.py -- python -m pytest tests/test_training.py -q
python -m schematic_model.experiment_memory list
python -m schematic_model.experiment_memory show EXPERIMENT_ID
```

The command is an argv array, passed without a shell. Shell expansion and command
substitution are not implicit. Supply `--cwd` for a project subdirectory; input
and artifact paths are relative to that directory. Declare additional `--input`
files for scripts, configuration, datasets, and model manifests that matter.
Use `--artifact PATH` to record an expected output file's before/after hash.
Unchanged preexisting artifacts are identified; absent outputs remain marked
missing. Hashes do not establish source correctness or scientific validity.

An explicit rerun supplies the command again:

```sh
python -m schematic_model.experiment_memory run --parent EXPERIMENT_ID \
  --relation rerun --input tests/test_training.py \
  -- python -m pytest tests/test_training.py -q
```

`rerun` requires matching argv, cwd, and declared input hashes. `variant` links a
changed experiment to its parent. Neither operation reads a command from the
record and executes it. Always inspect the parent first. Matching declared
inputs does not prove that undeclared files, dependencies, executable resolution,
or inherited environment values are unchanged.

Records live in ignored `outputs/experiments/`. Each fresh ID owns an append-only
`events.jsonl`, with a durable attempted event before launch and a terminal event
after collection. Hard interruption can leave an incomplete attempt; reading it
never reports success. Nonzero exit, launch failure, timeout, interruption, and
artifact-recording failure remain distinct. A successful process exit is not a
successful model experiment. Records are not a cryptographically authenticated
or tamper-proof audit log.

Directories are private (`0700`) and files private (`0600`), independent of umask.
Each stdout/stderr file retains at most 1 MiB by default; excess bytes are drained
and counted. `--output-limit` changes this cap up to 16 MiB per stream. The timeout
is at most one hour and kills the command's process group; it is not suitable for
training that must save recoverable checkpoints. Output caps cover captured logs, not files written by the
command. The child runs with operator permissions and inherited environment;
there is no sandbox, environment dump, automatic cleanup, or retention service.
Commands and output may contain private data: opting into recording deliberately
retains them. Never pass credentials in argv or publish these local records.

An optional [OpenCode agent prompt](examples/opencode/debug-harness.md) can be
copied deliberately to `.opencode/agents/debug-harness.md`. Its minimal Markdown
frontmatter uses the description/mode fields documented by both
[OpenCode V1](https://docs.opencode.ai/docs/agents/) and
[OpenCode V2](https://opencode.ai/v2/docs/agents). It inherits the operator's model
and permissions; installing or invoking the prompt does not enforce a local
provider. Configure a self-hosted model separately and keep cloud fallback
explicitly disabled. No OpenCode runtime integration has been verified by the
recorder's CPU tests. The prompt can also be used manually with another agent.
