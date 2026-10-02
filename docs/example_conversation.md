# Example training conversation

Design `D0012` — image: `docs/example_schematic.png`

**Reference data (from parts DB, sent with the first question)**

```
- LM1117-3.3: Vin abs-max 15 V, dropout 1.2 V, Iout max 800 mA, Pd max 1.20 W (SOT-223), Cout min 10 uF
- SMAJ5.0A: TVS stand-off (Vwm) 5 V
- LIS2DH12: VDD decoupling 100 nF; ADDR strap must be tied to GND or VDD
- Design rules: caps >= 1.5x applied voltage; fuse >= 1.5x input current; I2C pull-up >= 1.1k (3 mA sink), <= 4.7k at 400 kHz or <= 10k at 100 kHz; LED Vf 2.0 V, 1-20 mA; regulator needs Vin_min - dropout >= 3.4 V.
```

**Engineer:** Where does U2.SDA go?

**Model:** 

U2.SDA is on SDA, together with R1.2.

**Engineer:** What value is F1?

**Model:** 

F1 is a fuse, 0.5 A.

**Engineer:** Do a design review of this page.

**Model:** 

I found 3 issues:
- [critical] D1: D1 SMAJ5.0A stand-off 5 V is below the 12 V input, so it will conduct continuously.
- [major] C2, U1: C2 0.1 uF is below the 10 uF minimum output capacitance for LM1117-3.3; risk of oscillation.
- [minor] C3: C3 is 10nF; the LIS2DH12 datasheet calls for 100 nF at VDD.

**Engineer:** If I change R3 to 1k, what's the LED current?

**Model:** 

I = (3.3 - 2.0) V / 1k = 1.30 mA.

**Engineer:** What tolerance are the pull-up resistors?

**Model:** 

Tolerance isn't marked on R1 or R2 in this schematic.

**Engineer:** What should I change for C3?

**Model:** 

<think>
The issue: C3 is 10nF; the LIS2DH12 datasheet calls for 100 nF at VDD. Fix it at the source.
</think>

Change C3 to 100 nF (add a separate bulk cap if more capacitance is wanted).

**Gold labels used for grading:**

```json
[
 {
  "type": "pins",
  "pins": [
   "R1.2"
  ],
  "query": "U2.SDA"
 },
 {
  "type": "value",
  "value": "0.5 A"
 },
 {
  "type": "codes",
  "codes": [
   "DECOUPLING_VALUE",
   "REG_OUTPUT_CAP",
   "TVS_STANDOFF_LOW"
  ]
 },
 {
  "type": "number",
  "value": 1.3
 },
 {
  "type": "refuse"
 },
 {
  "type": "refdes",
  "refdes": [
   "C3"
  ]
 }
]
```
