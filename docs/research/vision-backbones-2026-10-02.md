# Vision model selection for a 24 GB RTX 4090

Research date: October 2, 2026. **Keep Qwen3.5-4B with BF16 LoRA as the working foundation.** Improve readable, balanced schematic supervision before changing backbones. The first alternative to test is Qwen3-VL-4B-Instruct; Gemma 4 E4B-it is a second candidate if the first comparison leaves a meaningful quality gap. Neither alternative has been loaded or evaluated in this project.

| Candidate | Verified specification | Proposed role |
|---|---|---|
| Qwen3.5-4B | Approximately 4.66 billion stored parameters; official Unsloth vision fine-tuning support | Current foundation, with measured local training evidence |
| Qwen3-VL-4B-Instruct | Approximately 4.44 billion stored parameters; official Unsloth vision training notebook | First challenger at comparable weight size |
| Gemma 4 E4B-it | Approximately 8.00 billion stored parameters despite the “E4B” effective-size name; official Unsloth vision fine-tuning support | Second challenger, subject to measured memory fit |

These specifications come from official model metadata and training documentation, not project benchmarks: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5 training guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-VL training notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_%288B%29-Vision.ipynb), [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it), and [Gemma 4 training guide](https://unsloth.ai/docs/models/gemma-4/train). The inspected repositories declare Apache 2.0 licenses. Unsloth discourages Qwen3.5 four-bit QLoRA because of quantization differences; BF16 LoRA remains the preferred baseline. Vendor memory estimates do not establish fit for this project's images and sequence lengths.

## Measured local evidence

The machine has an RTX 4090 with 24,564 MiB of GPU memory. A Qwen3.5-4B BF16 LoRA smoke test completed two optimizer steps on eight selected extreme training rows at a 2,048-pixel image limit, sequence limit 8,192, rank 16 and gradient accumulation 4. The selected rows reached 4,032 visual tokens and 7,794 total training tokens. Peak PyTorch allocation was 12,112,464,896 bytes (11.28 GiB); peak reservation was 12,318,670,848 bytes (11.47 GiB). The fit gate passed. Evidence: [recorded fit results](../validation/expanded-v3-fit-2048.json).

This is bounded compatibility and memory evidence, not proof that every full-run allocation fits, that training improves quality, or that an alternative model fits. PyTorch reservation also excludes memory held by other processes. The full training run starts from the pinned base; the smoke adapter remains a separate test artifact. Existing validation results also show that image resolution materially affects value reading; that makes input detail and label quality immediate priorities. Training and serving preprocessing must be compared explicitly rather than assumed identical from the same pixel-limit flag.

## Comparison and retraining policy

The hypothesis is that a compact model with readable inputs, balanced labels and authoritative tools can outperform a larger backbone on this narrow task. Test it with matched validation questions, image settings, output budgets and training snapshots. Inspect actual image-token grids, assistant supervision masks, trainable LoRA modules and clean adapter reloads. Measure value/reference association, pin/net correctness, unsupported claims, abstention, generation failures, latency and memory. Keep image-only, OCR-assisted and native-CAD-grounded results separate; crop selection must not use hidden answer coordinates. Preserve existing holdouts and reserve new board families for independent confirmation.

Fine-tune stable schematic-reading and engineering behaviors. Retrieve changing component specifications from versioned manufacturer datasheets, CAD files, BOMs and netlists, with source citations. Retrain on reviewed, immutable data snapshots that retain old-task examples, source hashes, licenses and label provenance. Do not train blindly on the model's own predictions. Promote a candidate only after old-task regression checks and new-family evaluation, publishing its exact base revision, data hashes, recipe and results alongside the adapter. Consider larger hardware only when measured quality or memory limits justify it; newer release dates and active-parameter names alone do not.
