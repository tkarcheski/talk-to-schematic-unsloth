# Training status

Last updated 2026-10-02. Every number here comes from files in this repository; the source is named next to it.

## Where things stand

- **v3 finished training** (666 of 666 steps, 3 epochs). Output: `outputs/expanded-v3-r16-e3-2048`.
- **v3 tests worse than v1** on both validation sets. **Do not publish v3.**

| Validation set | v3 values correct | v1 values correct | Release target |
|---|---|---|---|
| v3 render (`data/real-v3/vision/val.jsonl`) | 14/26 (0.538) | 23/26 (0.885) | 0.95 |
| frozen v2 (`data/training/real-crop-v1/vision/val.jsonl`) | 16/26 (0.615) | 24/26 (0.923) | 0.95 |

Both models refused correctly on every question about parts that don't exist (26/26).

- In each set, v3 failed to finish **9 answers**: it hit the 1,024-token limit without stopping, where v1 finished every answer. The failure records keep no partial text, so *why* it didn't stop (for example, repeating itself) is unconfirmed.
- Scoring checks format and keywords only. It is a rough measure, not proof of correctness. Answers have not yet been reviewed by hand.
- Reports: `results/v3-vs-v1-v3-render-val-2048-report.json`, `results/v3-vs-v1-val-vision-2048-report.json`.

## The training, in simple terms

**The starting point.** We take Qwen3.5-4B, a small AI model that can look at pictures and answer questions about them. It already has general knowledge from its original training. We don't change the whole model. We train a small add-on (a "LoRA adapter", about 155 MB) that adjusts how it answers.

**The lessons.** We show it real schematic pictures from 86–87 open-source hardware board families, with questions and correct answers. The v3 set (`data/training/real-expanded-v3`) has 888 practice conversations:

- **801 picture-only conversations** (887 questions). The model sees only the image. Mostly "Read the value shown next to R2" (692 questions).
- **87 picture-plus-text conversations** (522 questions, 6 per board). The model also gets the board's wiring list as text, so these test reading that text, not the picture.

There are six kinds of question:

1. **Read a value.** "Read the value shown next to R2", "What is the specified value of R2?" Answer: the label, like "10k". This is most of the set.
2. **Pads on a net.** "Which physical component pads connect to net SCL?" Follow the wiring.
3. **Trace a pad.** "Trace U1.4: which other pads share its net?" The same idea from the other direction.
4. **A part that doesn't exist.** "What value is R9999?" Right answer: it isn't on this schematic. This teaches the model not to make things up.
5. **Something a drawing can't tell you.** "What voltage was measured at JP1.8 during a bench test?" Right answer: a schematic can't show that.
6. **The same value questions in other phrasings**, such as "What value is recorded for R1?"

**Then the test.** We ask the same kinds of questions about 13 board families the model never saw in training (no overlap with the training boards), and score its answers.

## Are we teaching fundamentals?

**No.** Nothing in the training teaches what a resistor, capacitor, power rail, CPU, FPGA or IC *is*, or how to recognize their symbols. No training answer contains words like "resistor", "capacitor", "regulator", "FPGA" or "pull-up". The lessons teach finding labels and following wiring. Whatever the model understands about the parts themselves comes from its original training, and we never test that.

In plain terms: the model is taught to read labels on a map without being taught what the landmarks are.

## Why did v3 get worse? (not yet proven)

Missing fundamentals **does not** explain the drop by itself: v1 had no fundamentals either and still scored 0.89–0.92. What actually changed between v1 and v3:

- **About 5× more training data** (888 conversations vs 173 for v1), almost all the same "read the value" question.
- **Three passes** over that set, 666 training steps.
- **Training loss fell to almost zero** (0.0000337 at the last step), with no check against unseen boards during training.

The leading explanation is that v3 memorized the training answers instead of learning to read labels in general. That is a hypothesis. Tests that could confirm or rule it out:

- Score earlier checkpoints of the same run (`checkpoint-300`, `-450`, `-600`) on the same validation sets. If earlier checkpoints score better, that points to overfitting.
- Re-run the 9 unfinished answers and keep the partial text, to see what the model was doing.

## What adding fundamentals could look like (suggestion)

A "basics" layer of lessons, built from the same boards:

- **Symbols:** "What kind of part is this symbol?" for resistors, capacitors, inductors, diodes, transistors, ICs and connectors.
- **Concepts:** what a power rail (VCC, 3V3, GND) is, what a net is, what a pin and a reference designator (U1, R5) are.
- **Roles:** "Which part is the main processor?", "Which parts form the power supply?", "Is U3 a regulator, a CPU or an FPGA?"
- **Simple reasoning:** "Is this resistor a pull-up?", "Which capacitors decouple the 3V3 rail?"

The answers would need checking against each board's source files, not guessed. And because this adds more training data, it should come with a way to catch memorization, such as checking against unseen boards during training. Otherwise it could repeat v3's problem.
