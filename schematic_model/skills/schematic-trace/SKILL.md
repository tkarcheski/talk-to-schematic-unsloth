---
name: schematic-trace
description: Trace pins, net labels, connectors, and signal paths through electronic schematics using native connectivity or visible wire evidence. Use when asked what connects to a pin or net, or to check a suspected wiring mismatch.
---

# Trace schematic connectivity

Prefer native connectivity when supplied. Run `schematic-model inspect SOURCE.sch --out evidence.json`; in a repository checkout, use `uv run --locked schematic-model`. Read the requested sheet's `nets` and `bom` from the result. The EAGLE parser reports physical package pads; do not silently substitute the symbol's logical pin label.

For a pin trace, find the exact `REFDES.PAD` token and report its net plus the other endpoints. For a net trace, report the complete endpoint set on the relevant sheet. Preserve case and punctuation, including `+`, `-`, `$`, and underscores. Quote unusual identifiers so punctuation is not mistaken for prose.

Use the source revision and sheet number in the answer. Explicitly scope a sheet-local result; do not infer unseen hierarchical ports or cross-sheet connections. A parser rejection of an unsupported construct means the extraction is incomplete, not that the circuit is disconnected.

When tracing only an image, inspect junction dots, crossing wires, net labels, and connector numbering. Follow labels as well as drawn segments. If a crossing or label is unreadable, describe the ambiguity instead of choosing a connection. A nearby component is not necessarily on the net.

End with the exact connection and its evidence. Distinguish a confirmed wiring mismatch from a hypothesis about intended function. A schematic connection does not prove continuity on an assembled board.
