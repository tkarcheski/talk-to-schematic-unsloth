"""
hwsynth.py - synthetic schematic generator + rules engine (the "teacher").

Every design is generated from parameters, so we always know the TRUE
netlist, BOM and list of design faults. build_dataset.py turns that truth
into multi-turn "talk to the schematic" conversations.

Circuit (one sheet):
  J1 (12 V or 5 V USB in) -> F1 fuse -> D1 TVS + C1 bulk cap -> U1 LDO -> 3V3 (C2 out cap)
  3V3 -> R1/R2 I2C pull-ups -> U2 sensor (+ C3 decoupling, ADDR strap pin)
  3V3 -> R3 -> D2 status LED

Design fault library (v2, 17 codes) - chosen from common real-world misses:
  power:      LDO_VIN_EXCEEDED, LDO_DROPOUT, LDO_CURRENT, LDO_THERMAL, REG_OUTPUT_CAP
  protection: TVS_STANDOFF_LOW, MISSING_INPUT_PROTECTION, FUSE_UNDERSIZED, CAP_VOLTAGE_DERATING
  IC support: MISSING_DECOUPLING, DECOUPLING_VALUE, FLOATING_INPUT
  interface:  I2C_PULLUP_VALUE, I2C_PIN_SWAP, NET_LABEL_MISMATCH
  indicator:  LED_REVERSED, LED_CURRENT
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use("Agg")
import schemdraw
import schemdraw.elements as elm

# --------------------------------------------------------------------------
# Parts database (simplified, representative datasheet values)
# --------------------------------------------------------------------------
LDOS = {
    "AP2112K-3.3":  {"vin_max": 6.0,  "iout_max": 0.6, "pd_max": 0.40, "pkg": "SOT-23-5", "dropout": 0.25, "cout_min": 1.0},
    "TLV75533PDBV": {"vin_max": 5.5,  "iout_max": 0.5, "pd_max": 0.45, "pkg": "SOT-23-5", "dropout": 0.24, "cout_min": 1.0},
    "LM1117-3.3":   {"vin_max": 15.0, "iout_max": 0.8, "pd_max": 1.20, "pkg": "SOT-223",  "dropout": 1.20, "cout_min": 10.0},
    "AZ1117CH-3.3": {"vin_max": 18.0, "iout_max": 1.0, "pd_max": 1.00, "pkg": "SOT-223",  "dropout": 1.30, "cout_min": 4.7},
}
TVS = {"SMAJ5.0A": 5.0, "SMAJ12A": 12.0, "SMAJ15A": 15.0, "SMAJ24A": 24.0}
CAP_RATINGS = [6.3, 10, 16, 25, 35, 50]
PULLUPS = [470, 1000, 2200, 4700, 10000, 47000]
LED_RES = [47, 150, 330, 1000, 4700, 100000]
FUSES = [0.25, 0.5, 1.0, 2.0]
SENSORS = ["BME280", "TMP117", "SHT40", "LIS2DH12"]
VIN_MIN = {12.0: 10.8, 5.0: 4.5}   # 12 V +/-10 %, USB 5 V spec minimum

SEVERITY = {
    # design faults
    "LDO_VIN_EXCEEDED": "critical", "TVS_STANDOFF_LOW": "critical", "NET_LABEL_MISMATCH": "critical",
    "I2C_PIN_SWAP": "critical",
    "CAP_VOLTAGE_DERATING": "major", "LDO_THERMAL": "major", "LDO_CURRENT": "major", "LDO_DROPOUT": "major",
    "REG_OUTPUT_CAP": "major", "MISSING_DECOUPLING": "major", "I2C_PULLUP_VALUE": "major",
    "LED_REVERSED": "major", "MISSING_INPUT_PROTECTION": "major", "FLOATING_INPUT": "major",
    "FUSE_UNDERSIZED": "major",
    "DECOUPLING_VALUE": "minor", "LED_CURRENT": "minor",
}

@dataclass
class Design:
    design_id: str
    rev: str = "A"
    vin: float = 12.0
    i_load: float = 0.25
    i2c_khz: int = 400
    fuse_a: float = 1.0
    has_tvs: bool = True
    tvs: str = "SMAJ15A"
    c1_uf: float = 10.0
    c1_v: float = 25.0
    ldo: str = "LM1117-3.3"
    c2_uf: float = 10.0
    c2_v: float = 10.0
    r1: int = 2200
    r2: int = 2200
    sensor: str = "BME280"
    has_c3: bool = True
    c3_nf: int = 100
    addr_tie: str = "GND"          # GND | 3V3 | float
    i2c_swapped: bool = False
    r3: int = 330
    led_reversed: bool = False
    sda_label_mismatch: bool = False

    def to_dict(self):
        return asdict(self)


def fmt_ohm(r: int) -> str:
    return f"{r/1000:g}k" if r >= 1000 else f"{r}"


def fmt_cap_nf(nf: int) -> str:
    return f"{nf/1000:g}uF" if nf >= 1000 else f"{nf}nF"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def sample_design(design_id: str, rng: random.Random, fault_rate: float = 0.30) -> Design:
    """Each fault knob flips independently with probability ~fault_rate/4,
    so roughly a quarter of designs come out clean."""
    d = Design(design_id=design_id)
    d.vin = rng.choice([5.0, 12.0, 12.0])
    d.i_load = rng.choice([0.03, 0.05, 0.08, 0.1] if d.vin > 6 else [0.05, 0.1, 0.15, 0.2])
    d.i2c_khz = rng.choice([100, 400, 400])
    d.sensor = rng.choice(SENSORS)
    p = fault_rate / 4

    if d.vin > 6:
        d.ldo = rng.choice(["LM1117-3.3", "AZ1117CH-3.3"])
        d.tvs = rng.choice(["SMAJ15A", "SMAJ24A"])
        d.c1_v = rng.choice([25, 35, 50])
    else:
        d.ldo = rng.choice(["AP2112K-3.3", "TLV75533PDBV"])
        d.tvs = rng.choice(["SMAJ5.0A", "SMAJ12A"])
        d.c1_v = rng.choice([10, 16, 25])
    d.c1_uf = rng.choice([4.7, 10.0, 22.0])
    d.c2_uf = rng.choice([v for v in (1.0, 10.0, 22.0) if v >= LDOS[d.ldo]["cout_min"]])
    d.c2_v = rng.choice([10, 16])
    d.r1 = d.r2 = rng.choice([1500, 2200, 4700] if d.i2c_khz == 400 else [2200, 4700, 10000])
    d.r3 = rng.choice([150, 330, 1000])
    d.fuse_a = rng.choice([0.5, 1.0]) if d.i_load <= 0.25 else 1.0
    d.addr_tie = rng.choice(["GND", "3V3"])

    # ---- fault injection knobs ----
    if rng.random() < p and d.vin > 6:
        d.ldo = rng.choice(["AP2112K-3.3", "TLV75533PDBV"])
    if rng.random() < p and d.vin < 6:
        d.ldo = rng.choice(["LM1117-3.3", "AZ1117CH-3.3"])        # USB 5 V + 1117 = dropout
    if rng.random() < p and d.vin > 6:
        d.tvs = "SMAJ5.0A"
    if rng.random() < p:
        d.has_tvs = False
    if rng.random() < p:
        d.c1_v = rng.choice([c for c in CAP_RATINGS if c < 1.5 * d.vin] or [6.3])
    if rng.random() < p:
        d.i_load = rng.choice([0.25, 0.5, 0.7, 0.9])
    if rng.random() < p:
        d.c2_uf = 0.1 if LDOS[d.ldo]["cout_min"] <= 1 else rng.choice([0.1, 1.0])
    if rng.random() < p:
        d.fuse_a = 0.25
    if rng.random() < p:
        d.has_c3 = False
    elif rng.random() < p:
        d.c3_nf = rng.choice([10, 10000])
    if rng.random() < p:
        d.addr_tie = "float"
    if rng.random() < p:
        bad = rng.choice([470, 10000, 47000] if d.i2c_khz == 400 else [470, 47000])
        if rng.random() < 0.5:
            d.r1 = bad
        else:
            d.r2 = bad
    if rng.random() < p:
        d.led_reversed = True
    if rng.random() < p:
        d.r3 = rng.choice([47, 100000])
    if rng.random() < p * 0.7:
        d.sda_label_mismatch = True
    elif rng.random() < p * 0.7:
        d.i2c_swapped = True
    return d


# --------------------------------------------------------------------------
# Rules engine = ground-truth design findings
# --------------------------------------------------------------------------
def check_design(d: Design) -> list[dict]:
    f = []
    ldo = LDOS[d.ldo]
    vmin = VIN_MIN[d.vin]

    def add(code, refs, why):
        f.append({"code": code, "severity": SEVERITY[code], "refdes": refs, "why": why})

    if not d.has_tvs:
        add("MISSING_INPUT_PROTECTION", ["J1"],
            "External input J1 has no TVS/ESD protection; surges and hot-plug transients reach C1 and U1 directly.")
    elif TVS[d.tvs] < d.vin:
        add("TVS_STANDOFF_LOW", ["D1"],
            f"D1 {d.tvs} stand-off {TVS[d.tvs]:g} V is below the {d.vin:g} V input, so it will conduct continuously.")
    i_in = d.i_load + 0.005
    if d.fuse_a < 1.5 * i_in:
        add("FUSE_UNDERSIZED", ["F1"],
            f"F1 {d.fuse_a:g} A is under 1.5x the {i_in*1000:.0f} mA input current; nuisance blows or slow-blow fatigue.")
    if d.c1_v < 1.5 * d.vin:
        add("CAP_VOLTAGE_DERATING", ["C1"],
            f"C1 is rated {d.c1_v:g} V on the {d.vin:g} V VIN net; 1.5x derating needs >= {1.5*d.vin:g} V.")
    if d.vin > ldo["vin_max"]:
        add("LDO_VIN_EXCEEDED", ["U1"], f"U1 {d.ldo} abs-max input is {ldo['vin_max']:g} V but VIN is {d.vin:g} V.")
    elif vmin - ldo["dropout"] < 3.4:
        add("LDO_DROPOUT", ["U1"],
            f"At VIN min {vmin:g} V, U1 {d.ldo} (dropout {ldo['dropout']:g} V) can only hold "
            f"{vmin - ldo['dropout']:.2f} V; 3V3 falls out of regulation.")
    if d.i_load > ldo["iout_max"]:
        add("LDO_CURRENT", ["U1"],
            f"3V3 load {d.i_load*1000:.0f} mA exceeds U1 {d.ldo} rating of {ldo['iout_max']*1000:.0f} mA.")
    pd = (d.vin - 3.3) * d.i_load
    if pd > ldo["pd_max"]:
        add("LDO_THERMAL", ["U1"],
            f"Pd = ({d.vin:g} - 3.3) V x {d.i_load*1000:.0f} mA = {pd:.2f} W, above the {ldo['pd_max']:.2f} W limit for {ldo['pkg']}.")
    if d.c2_uf < ldo["cout_min"]:
        add("REG_OUTPUT_CAP", ["C2", "U1"],
            f"C2 {d.c2_uf:g} uF is below the {ldo['cout_min']:g} uF minimum output capacitance for {d.ldo}; risk of oscillation.")
    if not d.has_c3:
        add("MISSING_DECOUPLING", ["U2"], f"U2 {d.sensor} VDD has no local decoupling capacitor (expected 100 nF at the pin).")
    elif d.c3_nf != 100:
        add("DECOUPLING_VALUE", ["C3"],
            f"C3 is {fmt_cap_nf(d.c3_nf)}; the {d.sensor} datasheet calls for 100 nF at VDD.")
    if d.addr_tie == "float":
        add("FLOATING_INPUT", ["U2"], "U2 ADDR strap pin is unconnected, so the I2C address is undefined.")
    for ref, r in (("R1", d.r1), ("R2", d.r2)):
        hi = 4700 if d.i2c_khz == 400 else 10000
        if r < 1100:
            add("I2C_PULLUP_VALUE", [ref], f"{ref} = {fmt_ohm(r)} sinks {3.3/r*1000:.1f} mA when low, above the 3 mA I2C limit.")
        elif r > hi:
            add("I2C_PULLUP_VALUE", [ref], f"{ref} = {fmt_ohm(r)} pull-up is too weak for {d.i2c_khz} kHz I2C (max ~{fmt_ohm(hi)}).")
    if d.i2c_swapped:
        add("I2C_PIN_SWAP", ["U2"], "The wire labeled SDA lands on U2 pin SCL and SCL lands on SDA; the bus cannot work.")
    if d.led_reversed:
        add("LED_REVERSED", ["D2"], "D2 is drawn with its cathode toward R3/3V3 and anode to GND, so it never lights.")
    i_led = (3.3 - 2.0) / d.r3 * 1000
    if not d.led_reversed and (i_led > 20 or i_led < 1):
        add("LED_CURRENT", ["R3", "D2"], f"LED current (3.3 - 2.0) V / {fmt_ohm(d.r3)} = {i_led:.2f} mA is outside 1-20 mA.")
    if d.sda_label_mismatch:
        add("NET_LABEL_MISMATCH", ["R1", "U2"],
            "R1 connects to net label 'SDA' but U2 pin SDA is labeled 'SDA_1', so the pull-up is not connected.")
    return f


# --------------------------------------------------------------------------
# Netlist + BOM
# --------------------------------------------------------------------------
def bom(d: Design) -> list[dict]:
    items = [
        {"refdes": "J1", "type": "connector", "value": "2-pin", "rating": ""},
        {"refdes": "F1", "type": "fuse", "value": f"{d.fuse_a:g} A", "rating": ""},
        {"refdes": "C1", "type": "capacitor", "value": f"{d.c1_uf:g} uF", "rating": f"{d.c1_v:g} V"},
        {"refdes": "U1", "type": "ldo_regulator", "value": d.ldo, "rating": LDOS[d.ldo]["pkg"]},
        {"refdes": "C2", "type": "capacitor", "value": f"{d.c2_uf:g} uF", "rating": f"{d.c2_v:g} V"},
        {"refdes": "R1", "type": "resistor", "value": fmt_ohm(d.r1), "rating": ""},
        {"refdes": "R2", "type": "resistor", "value": fmt_ohm(d.r2), "rating": ""},
        {"refdes": "U2", "type": "sensor_ic", "value": d.sensor, "rating": ""},
        {"refdes": "R3", "type": "resistor", "value": fmt_ohm(d.r3), "rating": ""},
        {"refdes": "D2", "type": "led", "value": "LED", "rating": ""},
    ]
    if d.has_tvs:
        items.append({"refdes": "D1", "type": "tvs_diode", "value": d.tvs, "rating": f"{TVS[d.tvs]:g} V stand-off"})
    if d.has_c3:
        items.append({"refdes": "C3", "type": "capacitor", "value": fmt_cap_nf(d.c3_nf).replace("nF", " nF").replace("uF", " uF"), "rating": "10 V"})
    return sorted(items, key=lambda x: x["refdes"])


def netlist(d: Design) -> dict[str, list[str]]:
    gnd = ["J1.2", "C1.2", "U1.GND", "C2.2", "U2.GND"]
    v33 = ["U1.VOUT", "C2.1", "R1.1", "R2.1", "U2.VDD", "R3.1"]
    vin = ["F1.2", "C1.1", "U1.VIN"]
    if d.has_tvs:
        gnd.append("D1.A")
        vin.append("D1.K")
    if d.has_c3:
        gnd.append("C3.2")
        v33.append("C3.1")
    gnd.append("D2.A" if d.led_reversed else "D2.K")
    nets = {"VIN_RAW": ["J1.1", "F1.1"], "VIN": vin, "GND": gnd, "3V3": v33,
            "LED_A": ["R3.2", "D2.K" if d.led_reversed else "D2.A"]}
    if d.addr_tie == "GND":
        gnd.append("U2.ADDR")
    elif d.addr_tie == "3V3":
        v33.append("U2.ADDR")
    else:
        nets["U2_ADDR_NC"] = ["U2.ADDR"]
    sda_pin, scl_pin = ("U2.SCL", "U2.SDA") if d.i2c_swapped else ("U2.SDA", "U2.SCL")
    nets["SCL"] = ["R2.2", scl_pin]
    if d.sda_label_mismatch:
        nets["SDA"] = ["R1.2"]
        nets["SDA_1"] = [sda_pin]
    else:
        nets["SDA"] = ["R1.2", sda_pin]
    return {k: sorted(v) for k, v in sorted(nets.items())}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(d: Design, path: str, dpi: int = 110) -> None:
    sd = schemdraw.Drawing(show=False)
    sd.config(fontsize=10, unit=2.2)

    # ---- Block A: input + regulator ----
    sd += elm.Dot(open=True).at((0, 0)).label("J1-1", loc="left")
    sd += elm.Line().right(0.6)
    sd += elm.Fuse().right().label(f"F1\n{d.fuse_a:g} A")
    sd += elm.Line().right(0.6).label("VIN", loc="top")
    sd += elm.Dot()
    if d.has_tvs:
        sd.push()
        sd += elm.Zener().down().reverse().label(f"D1\n{d.tvs}", loc="bottom")
        sd += elm.Ground()
        sd.pop()
    sd += elm.Line().right(2.6)
    sd += elm.Dot()
    sd.push()
    sd += elm.Capacitor().down().label(f"C1\n{d.c1_uf:g}uF\n{d.c1_v:g}V", loc="bottom")
    sd += elm.Ground()
    sd.pop()
    sd += elm.Line().right(1.2)
    u1 = elm.Ic(pins=[elm.IcPin(name="VIN", side="left"), elm.IcPin(name="GND", side="bot"),
                      elm.IcPin(name="VOUT", side="right")],
                size=(2.4, 1.6)).anchor("VIN").label(f"U1\n{d.ldo}\n{LDOS[d.ldo]['pkg']}", loc="top", ofst=0.15)
    sd += u1
    sd += elm.Ground().at(u1.GND)
    sd += elm.Line().right(1.0).at(u1.VOUT)
    sd += elm.Dot()
    sd.push()
    sd += elm.Capacitor().down().label(f"C2\n{d.c2_uf:g}uF\n{d.c2_v:g}V", loc="bottom")
    sd += elm.Ground()
    sd.pop()
    sd += elm.Line().right(1.2)
    sd += elm.Vdd().label("3V3")
    sd += elm.Dot(open=True).at((0, -3.0)).label("J1-2", loc="left")
    sd += elm.Ground()

    # ---- Block B: I2C sensor ----
    y = -9.5
    left_top, left_bot = ("SDA", "SCL") if d.i2c_swapped else ("SCL", "SDA")
    u2 = elm.Ic(pins=[elm.IcPin(name=left_bot, side="left"), elm.IcPin(name=left_top, side="left"),
                      elm.IcPin(name="VDD", side="top"), elm.IcPin(name="GND", side="bot"),
                      elm.IcPin(name="ADDR", side="right")],
                size=(3.6, 3.0), pinspacing=1.2).at((10.5, y))
    sd += u2
    sd += elm.Label().at((u2.GND[0] - 2.6, u2.GND[1] - 0.7)).label(f"U2  {d.sensor}", halign="left", fontsize=10)
    sd += elm.Ground().at(u2.GND)
    sd += elm.Line().up(0.8).at(u2.VDD)
    sd += elm.Dot()
    sd.push()
    sd += elm.Vdd().label("3V3")
    sd.pop()
    if d.has_c3:
        sd += elm.Line().right(3.8)
        sd += elm.Capacitor().down().label(f"C3\n{fmt_cap_nf(d.c3_nf)}", loc="bottom")
        sd += elm.Ground()
    # ADDR strap
    sd += elm.Line().right(0.9).at(u2.ADDR)
    if d.addr_tie == "GND":
        sd += elm.Ground()
    elif d.addr_tie == "3V3":
        sd += elm.Vdd().label("3V3")

    # pull-ups - wires go to the physical pin positions (top-left / bottom-left)
    top_pin = getattr(u2, left_top)
    bot_pin = getattr(u2, left_bot)
    for ref, val, x, pin, net in (("R1", d.r1, 3.0, bot_pin, "SDA"), ("R2", d.r2, 5.0, top_pin, "SCL")):
        yy = pin[1]
        sd += elm.Vdd().at((x, yy + 3.2)).label("3V3")
        sd += elm.Resistor().down().at((x, yy + 3.2)).toy(yy).label(f"{ref}\n{fmt_ohm(val)}", loc="bottom")
        sd += elm.Dot()
        if net == "SDA" and d.sda_label_mismatch:
            sd += elm.Line().left(1.0).at((x, yy)).label("SDA", loc="left")
            sd += elm.Line().left(2.2).at(pin).label("SDA_1", loc="top")
        else:
            sd += elm.Line().at((x, yy)).to(pin).label(net, loc="top")

    # ---- Block C: status LED ----
    x0 = 18.5
    sd += elm.Vdd().at((x0, -1.0)).label("3V3")
    sd += elm.Resistor().down().at((x0, -1.0)).label(f"R3\n{fmt_ohm(d.r3)}", loc="bottom")
    led = elm.LED().down().label("D2", loc="bottom")
    sd += led.reverse() if d.led_reversed else led
    sd += elm.Ground()

    # ---- title block ----
    vmin = VIN_MIN[d.vin]
    sd += elm.Label().at((16.5, -13.5)).label(
        f"DESIGN {d.design_id}  REV {d.rev}\n"
        f"NOTES: VIN = {d.vin:g} V nominal ({vmin:g} V min). 3V3 load = {d.i_load*1000:.0f} mA.\n"
        f"I2C bus = {d.i2c_khz} kHz.", halign="left", fontsize=9)
    sd.save(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close("all")


def datasheet_context(*designs: Design) -> str:
    """Datasheet excerpts for the parts on the sheet. In production this comes
    from your parts DB / RAG over datasheets - the model reasons from these
    numbers, not from memory."""
    lines = []
    for ldo in sorted({x.ldo for x in designs}):
        p = LDOS[ldo]
        lines.append(f"{ldo}: Vin abs-max {p['vin_max']:g} V, dropout {p['dropout']:g} V, Iout max {p['iout_max']*1000:.0f} mA, "
                     f"Pd max {p['pd_max']:.2f} W ({p['pkg']}), Cout min {p['cout_min']:g} uF")
    for tvs in sorted({x.tvs for x in designs if x.has_tvs}):
        lines.append(f"{tvs}: TVS stand-off (Vwm) {TVS[tvs]:g} V")
    for s in sorted({x.sensor for x in designs}):
        lines.append(f"{s}: VDD decoupling 100 nF; ADDR strap must be tied to GND or VDD")
    lines.append("Design rules: caps >= 1.5x applied voltage; fuse >= 1.5x input current; I2C pull-up >= 1.1k (3 mA sink), "
                 "<= 4.7k at 400 kHz or <= 10k at 100 kHz; LED Vf 2.0 V, 1-20 mA; regulator needs Vin_min - dropout >= 3.4 V.")
    return "\n".join("- " + l for l in lines)


