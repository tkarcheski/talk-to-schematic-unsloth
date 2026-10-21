# Search routing continuation: candidate rejected

A separate adapter continued the existing v1 weights for 68 optimizer steps on 269 conversations: 96 source-grounded routing lessons and 173 original replay conversations. Settings were rank 16, learning rate 0.00005, gradient accumulation 4, 1 epoch, 8192 context, 1024 maximum image side, and frozen vision layers. Original weights were not changed.

The local Unsloth run saved a complete checkpoint and yielded at step 26 after 123 seconds. A scheduler-owned research process then loaded the local model, produced six public-source review candidates in 33 seconds, exited, and released GPU residency. Training resumed from checkpoint 26 and completed step 68. This verifies that handoff sequence; it does not establish unattended scheduler reliability or authenticated Studio API control.

| Frozen routing category | Original v1 | Candidate |
|---|---:|---:|
| Explicit web search | 7/7 | 7/7 |
| Follow-up search for selected part | 7/7 | 7/7 |
| Viewer selection | 7/7 | 4/7 |
| Clarification with no selected part | 3/7 | 7/7 |
| Total | 24/28 | 25/28 |

The original baseline report incorrectly required a question mark in clarification responses. A versioned rescore accepts explicit requests to specify the component; all original predictions and the initial report are preserved. Neither model was regenerated for this scoring correction.

The three candidate viewer failures contain the correct `show_label` call followed by invented viewer/user conversation text. They use only 43–57 output tokens, and all three still fail with a 1024-token allowance. Strict parsing correctly rejects them. Do not trim the generated text, execute its valid-looking prefix, or count these as successful actions.

The original image-only benchmark remains 24/26 for values and 26/26 for refusals, with zero incomplete generations. This preserves v1 performance but remains below the 95% value-accuracy release threshold. Native-evidence validation preserves 26/26 for values, pins, and refusals, with zero incomplete generations. The candidate is not promoted because viewer behavior regressed.

Local evidence is under `results/agent-routing-v1-*`: original and revised baseline, candidate first-action predictions, raw public-case diagnostics, training logs, checkpoint metadata, and original-task comparisons. These artifacts contain public evaluation cases; private workspace conversations were not used. Candidate adapter SHA-256: `d4e87d43e8d5a99682abc90d15f2d4296d84bfebd17d87c9419d2428aeb67ec3`.

The next training experiment should test earlier checkpoints and, if necessary, strengthen clean tool-call termination using prefixes derived only from training families. Preserve text replay and clarification examples, keep validation boards out of training, and repeat both action and schematic-quality gates. The existing parser contract remains unchanged.
