---
name: schematic-evidence
description: Separate exact EAGLE schematic facts from engineering conclusions that require datasheets, operating assumptions, measurements, or human review. Use when preparing grounded answers or training labels.
---

Use the local `schematic-model` package to inspect the native source. Select the
page explicitly; matching net names across sheets alone do not prove topology.

```sh
python -m schematic_model.engineering_evidence circuit.sch --kind value --target R1 --page 1
python -m schematic_model.engineering_evidence circuit.sch --kind net_members --target VCC --page 1
python -m schematic_model.engineering_evidence circuit.sch --kind pad_net --target R1.1 --page 1
```

Only `source_verified` responses have extracted answers eligible for native
fact labels. Record the source hash, source license, family split, page, and
evidence mode alongside any accepted example. The source hash alone does not
establish redistribution permission or image readability. A missing target is
`not_recorded_on_page`, not proof of absence from every sheet or the hardware.

Queries for `component_role`, `decoupling_adequacy`, `voltage_compatibility`,
`design_defect`, and `measured_voltage` return required evidence with no answer.
Do not turn them into gold labels through a heuristic or a model's confidence.
For example, a capacitor between a rail and ground is an observed connection;
adequate decoupling needs component requirements and placement evidence.

Keep proposed questions separate from source-verified answers and reviewed
engineering conclusions. Reviewed conclusions need actual supporting citations
and explicit assumptions; this tool does not certify those conclusions. Do not
invent a measurement, defect, part identity, or reviewer approval.

Commands print source content locally. Save, transmit, or publish it only when
the operator deliberately chooses that action. No cloud fallback is required.
