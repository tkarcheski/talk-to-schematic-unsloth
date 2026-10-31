<!-- cspell:words Pynt Titano QSPI EVK MicroMod MacroPad MEMENTO FreeSoC -->

# Multi-page evidence and longer-context candidates

Research only; none of these rows are included in the routing retrain. Preserve related board revisions as one family. Existing independent-vendor diagnostic sources must remain evaluation-only unless their role is deliberately changed.

## Accepted expansion

The reproducible review queue now contains **10 families, 12 source schematics, 27 populated sheets, and 318 cross-page endpoint questions**, up from two families, three schematics, eight sheets, and 102 questions. Eight new families add power, cellular/satellite tracking, GNSS, camera, display/audio, and programmable-controller circuits. The largest source grows from 194 to 312 unique physical reference designators. These counts describe native circuit structure, not measured model reasoning quality.

| Family | Populated sheets per source | Unique physical components per source | Cross-page questions |
|---|---:|---:|---:|
| [Tsunami Qwiic](https://github.com/sparkfun/SparkFun_Tsunami_Super_WAV_Trigger_Qwiic/tree/336eb78afcebdcea91bf6caa554670940aae7bd5) | 2 | 194 | 16 |
| [PyPortal / Pynt](https://github.com/adafruit/Adafruit-PyPortal-PCB/tree/90d9aa7ba3f892fd4adb31815c99ede26c10757c) | 3 / 3 | 76 / 81 | 86 |
| [MicroMod Asset Tracker](https://github.com/sparkfun/MicroMod_Asset_Tracker/tree/d0f4731a7c3d8cb791db75c5734e7c261e82f5de) | 2 | 206 | 41 |
| [RTK EVK](https://github.com/sparkfun/SparkFun_RTK_EVK/tree/5704e26d9ca373bed4e0504b0571e29be5153b1c) | 2 | 312 | 18 |
| [Artemis Global Tracker](https://github.com/sparkfun/SparkFun_Artemis_Global_Tracker/tree/3212a30d17952d89d7fb301f154e82907fe01a70) | 2 | 148 | 22 |
| [RTK Facet variants](https://github.com/sparkfun/SparkFun_RTK_Facet/tree/47aa0abf364b5d41a1568778cb3abd4f2d598c22) | 2 / 2 | 131 / 113 | 37 |
| [MacroPad RP2040](https://github.com/adafruit/Adafruit-MacroPad-RP2040-PCB/tree/fdd7f2cb3bc2b3c7a9c0765780387647ea872141) | 2 | 96 | 26 |
| [Fruit Jam](https://github.com/adafruit/Adafruit-Fruit-Jam-PCB/tree/252dc09456ee447b3d7eeb4fa99877898503e81a) | 2 | 134 | 33 |
| [MEMENTO](https://github.com/adafruit/Adafruit-MEMENTO-PCB/tree/8f1e07b4baa4e43749793e3ebebd12005a5a8f96) | 2 | 121 | 19 |
| [FreeSoC2](https://github.com/sparkfun/FreeSoc2/tree/7a7a325278c0fb96ba51e05cd0fb847a73feb79f) | 3 | 109 | 20 |

All selected sources retain their existing deterministic train partition. Related PyPortal and Facet revisions stay together. RTK mosaic-X5 is excluded because its test assignment would separate substantially reused circuitry from the training-partition EVK. BeagleBone Blue is excluded pending clarification of conflicting schematic and repository license notices. Sources without reviewed license evidence or supported physical-pad mappings are not admitted.

The accepted queue uses a 2,048-token answer reserve, above the measured longest canonical endpoint answer of 1,880 tokens. All 318 focused text variants fit 8,192 tokens; **232 full-evidence variants exceed that limit**. Full input reaches 21,712 tokens before the answer reserve. Nothing is truncated, and image tokens are not included in these text measurements. The accepted candidate hash is `742073ac036670ea1fc7ba85b331a86b8ee220631931159abb521d0d6eb76b47`.

All 27 source pages have attributed PNG/SVG review assets with recorded hashes. The extracted candidate answers were checked against pinned native endpoints; visual labels and narrative engineering answers remain unreviewed. Existing training, validation, and test snapshots were not modified. See [build instructions](build-multipage-candidates.md) and the [pinned source plan](multipage-source-plan.json).

## Earlier source investigation

| Source | Verified native sheets | License / status |
|---|---:|---|
| [SparkFun Tsunami Qwiic](https://github.com/sparkfun/SparkFun_Tsunami_Super_WAV_Trigger_Qwiic/tree/336eb78afcebdcea91bf6caa554670940aae7bd5) | 2, both populated | CC BY-SA 4.0; pinned in existing SparkFun manifest; train family |
| [Adafruit PyPortal](https://github.com/adafruit/Adafruit-PyPortal-PCB/tree/90d9aa7ba3f892fd4adb31815c99ede26c10757c) | 3, all populated | CC BY-SA 3.0; local source hash verified; keep Pynt/Titano variants in this same family |
| [SparkFun RA6M5](https://github.com/sparkfun/SparkFun_Thing_Plus_RA6M5/tree/b655d23b286b797c6edc95dcd18786b6abdef8cd) | 2, both populated | CC BY-SA 4.0 hardware; existing independent-vendor diagnostic, not training |
| [Adafruit Voice Bonnet](https://github.com/adafruit/Adafruit-Voice-Bonnet-PCB/tree/9a0927aa41c9a2b6cb3f4e198425ffd6a30149b2) | 2 raw XML sheets | Excluded: embedded GND1 deviceset prevents current parser from resolving physical pads |

Artemis Development Kit was also checked at `4d5afc951a0887709fc2ba3bf365e2c71c42151a`: its source parses but has only one sheet, so it does not count toward multi-page coverage. An empty second sheet also does not qualify.

The Tsunami candidate has 194 sheet-local BOM entries, 16 net names shared across pages, and 37,438 characters of full native context. Local source-derived candidate questions ask for every physical pad on each shared net, with explicit page citations. This is native-evidence reasoning, not visual wire tracing. The pinned local tokenizer counts 16,446 tokens for this native JSON alone, already above the current 8,192-token serving context before tool definitions, image tokens, or an answer reserve. This is an explicit over-budget case, not a successful long-context run.

Expert question proposals, requiring source citations and human review before using narrative answers as gold:

- PyPortal: trace the ESP32 read-back path through the buffer to the MCU across sheets; distinguish the DNP bypass resistor from an installed connection. Any tri-state timing claim needs the buffer datasheet.
- PyPortal: produce a silent-speaker fault-isolation plan using the audio coupling path and shutdown signal on different sheets. Identify measurements to take without inventing their values.
- RA6M5: trace the shared I2C bus, onboard fuel gauge, pull-ups, and alert interrupt across sheets. Does unplugging the external cable remove the onboard device from the drawn bus?
- RA6M5: enumerate the QSPI flash endpoints and chip-select pull-up, respecting resistor-array gates that share one reference across sheets.
- RA6M5: trace reference-enable and analog-reference supply paths. Separate connectivity from noise, settling, and measured-reference performance.
- Voice Bonnet (blocked): identify capture/playback clock and data paths from the codec's perspective, then trace analog power and return filtering. Keep symbol pin names separate from physical pads until import succeeds.

For longer-context evaluation, freeze short and full native-context variants of the same questions, retain all pages and conflicting irrelevant net names, and score exact page/endpoint sets. Count tokens with the pinned tokenizer; include tool definitions, conversation history, image tokens, and the answer reserve in the actual context budget. Over-budget requests must produce an explicit bounded failure, never silently truncate the evidence. A text-token count alone does not establish combined image/text fit.
