<!-- cspell:words Pynt Titano QSPI -->

# Multi-page evidence and longer-context candidates

Research only; none of these rows are included in the routing retrain. Preserve related board revisions as one family. Existing independent-vendor diagnostic sources must remain evaluation-only unless their role is deliberately changed.

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
