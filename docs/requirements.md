# Requirements and acceptance evidence

The deliverable is a reproducible local schematic assistant: a fine-tuned model,
an installable Python package, a real schematic chat workspace, and reusable AI
skills. The target machine has one RTX 4090 with 24 GB VRAM. An experimental
checkpoint is available, but model qualification remains open.

## Required behavior

| Requirement | Acceptance condition | Current evidence or remaining work |
|---|---|---|
| Installable package | Locked uv install, wheel install outside the checkout, working CLI and packaged assets | CPU package and isolated wheel checks pass; native GPU versions are recorded separately |
| Real schematic corpus | At least 100 attributable schematics with pinned sources, licenses, image hashes, and board-family splits | 120 different Adafruit board repositories; [source inventory](../corpora/adafruit-120.json) |
| Detailed simulated chats | At least 25 complete conversations, clearly separated from actual model transcripts | [25 conversations and 150 user prompts](simulated_conversations.md), derived from source facts |
| Train on the 4090 | Actual Unsloth training completes with finite loss and saved artifacts that reload successfully; memory use recorded | First real candidate completed three epochs at 12.43 GiB peak allocated memory; larger-image training is being checked |
| Read values from images | Fixed held-out evaluation, every requested answer accounted for, literal-value accuracy at least 95% | First candidate fails: 27/40 at 1,024 pixels; validation-only 2,048-pixel result is 24/26 |
| Trace native connections | Physical-pad endpoint evaluation with F1 at least 95%; provenance states when native evidence is supplied | First candidate passes the narrow native-evidence benchmark; image-only connection tracing is not established |
| Respect missing evidence | Refusal accuracy at least 98%, no incomplete generation hidden, separate audit of unsupported claims | Initial candidate passes keyword refusal checks; semantic robustness needs broader held-out questions |
| Compare candidates fairly | Pinned model/data hashes, identical resolution and answer budgets, original baseline retained | Failure-accounting benchmark and resumable experiment queue; [first comparison](validation/model-v1/summary.json) |
| Usable local chat | Select a real schematic, inspect its source, zoom, ask the loaded model, see failures accurately | 120-item catalog, desktop/mobile browser checks, actual three-turn conversation at 2,048 pixels |
| Reusable AI skills | Packaged reading, tracing, and evaluation instructions with evidence boundaries | Three packaged skills; export is verified and refuses to overwrite existing directories |
| Publish a model | Reproducible adapter and honest model card; qualified status only after gates pass | [Experimental release](validation/hugging-face-experimental-release.json) published and hash verified; qualified release pending |
| Maintainable development | Small reviewed commits, formatting, spelling, regression tests, regular pushes | Local hooks run; remote CI activation is pending GitHub workflow credential scope |

## Evaluation and data boundaries

Training, validation, and test sets are separated by board repository. Test
families and their original files remain frozen. New training views can add
source-verified labels only from training families. A new renderer gets a new
dataset version; its effect is measured separately from model updates.

Native-evidence questions provide extracted component and net facts to the
model. Passing them does not demonstrate independent visual reading. Image-only
questions provide no extracted answer facts. Source visibility flags establish
that a label is intended to be drawn; they do not prove that it remains legible
after overlap or downscaling. Confirmed ambiguous training labels are excluded
explicitly, with reasons recorded.

Every requested benchmark turn stays in the denominator. Truncated or failed
generations have empty scored answers and explicit failure records. Gold-history
and generated-history runs are reported separately. Deterministic scores are
accompanied by source review: a correct nominal value with an invented qualifier
or an unsupported extra claim must not be presented as engineering correctness.

## Runtime and operations

CPU tools must import and run without Unsloth or a GPU. Actual model operations
use the installed Unsloth environment without upgrading its managed packages.
Training and inference share one GPU and run sequentially. A bounded fit test
precedes a larger training configuration; an out-of-memory failure is reported
before changing resolution, precision, model, or other settings.

The service targets one local user and binds to loopback. HTTP concurrency is
bounded, GPU generation is serialized, and request size, context, and output
length are limited. Public multi-user hosting, authentication, and distributed
training are outside this prototype's verified deployment scope.

Artifact manifests bind source data, image bytes, model files, settings, and
installed versions. Fresh outputs cannot overwrite their inputs. Resuming a
benchmark requires matching provenance; interrupted and completed attempts
remain auditable. Deployment bundles and wheel assets are checked independently
of the source checkout.

## Open acceptance work

- Improve image-only value reading with a balanced, reviewed training set and
  larger-image experiments that fit the 4090.
- Expand generated-history evaluation and novel questions beyond repeated
  template wording, while retaining the existing benchmark for comparison.
- Document measured limitations of visual connection reasoning and distinguish
  them from the supported native-source tracing workflow.
- Activate remote CI when the existing GitHub credential can publish workflow
  files. Local checks continue to run before pushes.
- Publish a qualified checkpoint only after the evidence supports that label.

KiCad import, a general electrical-rules checker, and a full schematic review
pipeline are later work. No benchmark here establishes fault-detection coverage
or replaces an electrical design review.
