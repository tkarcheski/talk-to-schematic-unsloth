# Search routing regression

The reported failure substitutes a viewer movement for an explicit request to search for the focused part. Routing lessons use the UI's serialized focused-label context and native source BOM, with contrasting search, show, and ambiguous-reference tasks. Follow-up examples include the preceding viewer call and result. Search targets contain the source part number; no fetched results or electrical conclusions are invented.

Inputs are the existing pinned Adafruit corpus (`corpora/adafruit-120.json`, CC BY-SA 3.0 with source hashes and notices) and the original v1 training snapshot. Native source and image hashes, plus reproduced SVG geometry, are checked before adding a row. Original board-family ownership is preserved. Validation part numbers are absent from all eligible training ICs; original test boards are excluded. Generated outputs remain local and record artifact hashes.

The continuation path uses the installed Unsloth loader's existing-adapter support (loading existing adapter weights with `is_trainable=True`, followed by model patching). An explicit `--continue-adapter` skips adding a second adapter, binds source hashes and rank/target modules, trains only existing LoRA parameters, and starts a fresh optimizer in a separate output directory. It is not optimizer resume. CPU tests cannot establish GPU fit or actual learning; a two-step fit and real pre/post generations are required before accepting a candidate.

Acceptance requires correct first-action routing on the frozen cases (including the reported wording), no unfinished generations, and a separate comparison on the original image-only and native-evidence validation sets. A routing score alone does not qualify a model for schematic review.
