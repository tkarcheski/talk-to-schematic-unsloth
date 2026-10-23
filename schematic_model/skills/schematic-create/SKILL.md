---
name: schematic-create
description: Generate small instructional resistor/capacitor EAGLE schematics from explicit components and nets, and verify their extracted facts. Use for controlled synthetic examples, not unrestricted circuit design or PCB fabrication.
---

Use the installed `schematic-model` package and its local Python modules. This
workflow needs no model download, external service, or GPU. A local model may
propose a specification, but the executable validator must accept it first.

Prepare JSON with exactly `family`, `components`, and `nets`. Example:

```json
{
  "family": "rc-filter",
  "components": [
    {"refdes": "R1", "kind": "R", "value": "10k"},
    {"refdes": "C1", "kind": "C", "value": "100nF"}
  ],
  "nets": {"IN": ["R1.1"], "OUT": ["R1.2", "C1.1"], "GND": ["C1.2"]}
}
```

Every pad must belong to exactly one net, including explicitly named isolated
pads. Only R/C parts with pads 1 and 2 are supported, with at most 12 parts.

```sh
python -m schematic_model.synthetic_circuit spec.json --out outputs/example.sch
schematic-model inspect outputs/example.sch --out outputs/example-facts.json --render outputs/example.png
```

The generator re-imports its XML and compares component values and physical-pad
net membership with the specification. Inspect the rendered result for label
overlap and ambiguous connections before using it as visual training material.
Existing outputs are protected. Keep private inputs and generated artifacts in
an ignored output directory; publishing requires a deliberate operator choice.

For paired examples, change one value or one endpoint assignment and regenerate
to a new destination. Verify the intended changed fact and unchanged facts.
Keep all related variants under the same family and dataset split; never use
renaming or layout changes to move a family across splits. The generator does
not assign splits or add examples to existing datasets.

Report these as synthetic source-verified facts. The generated packages are
instructional identifiers, not manufacturing footprints. Native round trips do
not establish electrical suitability, model quality, or design-defect labels.
