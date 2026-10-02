"""
build_dataset.py - build "talk to the schematic" training data for Unsloth.

Each example is ONE schematic image + a multi-turn conversation:
  - review:       "Review this sheet"            -> findings with refdes + severity
  - connectivity: "What's on the SDA net?"       -> exact pins (REFDES.PIN)
  - pin trace:    "Where does U2 ADDR go?"       -> net + far-end pins
  - lookup:       "What is C1 rated?"            -> value / rating as drawn
  - calc:         "What's the LED current?"      -> worked number
  - what-if:      "If R3 were 1k?"               -> recomputed number
  - fix:          "How would you fix that?"      -> concrete change
  - presence:     "Is there a TVS on the input?" -> yes/no with refdes
  - grounding:    "What value is C7?" / "What's the trace width?" -> says it is not on the sheet

Answers are computed from the true netlist/BOM/rules, so every turn has a
machine-checkable gold label (used by evaluate.py).

Only the LAST assistant turn carries a <think> block - this matches inference,
where Qwen chat templates drop reasoning from earlier turns.

Usage:
  python build_dataset.py --n-designs 3000 --out data
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import warnings

from PIL import Image, ImageFilter

import hwsynth as hs
from evaluate import validate_gold_rows

SYSTEM = (
    "You are a hardware engineer's schematic assistant. Answer only from the schematic image and the "
    "reference data provided. Cite reference designators and pins as REFDES.PIN (e.g. U2.SDA). "
    "Show the numbers when you calculate. If the answer is not on the schematic, say so plainly - never guess."
)

FIX = {
    "LDO_VIN_EXCEEDED": "Replace U1 with a regulator rated above the input (e.g. AZ1117CH-3.3, 18 V max) or add a buck pre-regulator.",
    "LDO_DROPOUT": "Use a low-dropout part such as AP2112K-3.3 (0.25 V dropout), or raise the input voltage.",
    "LDO_CURRENT": "Use a regulator rated above the 3V3 load current, or move to a buck converter.",
    "LDO_THERMAL": "Reduce Vin - Vout (pre-regulate), reduce load, or switch to a buck converter; a bigger package alone may not be enough.",
    "REG_OUTPUT_CAP": "Increase C2 to at least the regulator's minimum output capacitance (10-22 uF is a safe choice).",
    "TVS_STANDOFF_LOW": "Choose a TVS whose stand-off voltage is above the maximum input, e.g. SMAJ15A on a 12 V rail.",
    "MISSING_INPUT_PROTECTION": "Add a TVS (D1) from VIN to GND right after F1, stand-off above the max input voltage.",
    "FUSE_UNDERSIZED": "Raise F1 to at least 1.5x the maximum input current.",
    "CAP_VOLTAGE_DERATING": "Use a C1 rated at least 1.5x the input voltage.",
    "MISSING_DECOUPLING": "Add a 100 nF capacitor from U2.VDD to GND, placed at the pin.",
    "DECOUPLING_VALUE": "Change C3 to 100 nF (add a separate bulk cap if more capacitance is wanted).",
    "FLOATING_INPUT": "Tie U2.ADDR to GND or 3V3 to set the I2C address.",
    "I2C_PULLUP_VALUE": "Use 2.2k pull-ups at 400 kHz (4.7k at 100 kHz).",
    "I2C_PIN_SWAP": "Swap the connections so the SDA net lands on U2.SDA and SCL on U2.SCL.",
    "NET_LABEL_MISMATCH": "Rename the U2 pin label from SDA_1 to SDA (or draw the wire) so R1 actually pulls up U2.SDA.",
    "LED_REVERSED": "Flip D2 so the anode faces R3 and the cathode goes to GND.",
    "LED_CURRENT": "Pick R3 for 2-10 mA, e.g. 330 ohm gives about 3.9 mA.",
}

UNPOLARIZED = ("R", "C", "F")
NOT_ON_SHEET = [
    ("What trace width is used for the VIN net?", "Trace width is a PCB layout property; this schematic doesn't show it."),
    ("Which PCB layer is the SDA line routed on?", "Layer assignment isn't on a schematic. You'd need the PCB layout to answer that."),
    ("What dielectric is C2 (X7R, X5R)?", "The sheet shows C2's capacitance and voltage only; the dielectric isn't specified."),
    ("What's the manufacturer part number for R1?", "Only R1's value is drawn; no manufacturer part number is shown on this sheet."),
    ("What tolerance are the pull-up resistors?", "Tolerance isn't marked on R1 or R2 in this schematic."),
    ("What's the board's operating temperature range?", "The schematic doesn't state an operating temperature range."),
]


# --------------------------------------------------------------------------
def pins_on(nets, pin):
    for n, ps in nets.items():
        if pin in ps:
            return n, [p for p in ps if p != pin]
    return None, []


def canonical_pairs(nets):
    out = set()
    for pins in nets.values():
        norm = sorted({p.split(".")[0] if p[0] in UNPOLARIZED else p for p in pins})
        for i in range(len(norm)):
            for k in range(i + 1, len(norm)):
                if norm[i].split(".")[0] != norm[k].split(".")[0]:
                    out.add((norm[i], norm[k]))
    return sorted([list(p) for p in out])


def augment(src, dst, rng):
    with Image.open(src) as source:
        im = source.convert("RGB")
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.4:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    if rng.random() < 0.4:
        im = Image.blend(im, Image.new("RGB", im.size, rng.choice([(250, 245, 225), (235, 235, 235)])), 0.25)
    im.save(dst, quality=rng.randint(60, 90))


# --------------------------------------------------------------------------
# Turn generators: each returns (question, answer, think, gold)
# --------------------------------------------------------------------------
def t_review(d, ctx, rng):
    f = ctx["findings"]
    q = rng.choice(["Review this schematic for problems.", "Can you sanity-check this sheet before layout?",
                    "Anything wrong with this design?", "Do a design review of this page."])
    if not f:
        a = ("I don't see any problems against the provided rules: protection, regulator limits, decoupling, "
             "I2C pull-ups, strap pins and LED current all check out. Layout, thermal relief and firmware "
             "aren't covered by a schematic review.")
    else:
        order = {"critical": 0, "major": 1, "minor": 2}
        lines = [f"- [{x['severity']}] {', '.join(x['refdes'])}: {x['why']}" for x in sorted(f, key=lambda x: order[x["severity"]])]
        a = f"I found {len(f)} issue{'s' if len(f) > 1 else ''}:\n" + "\n".join(lines)
    think = ("Walk the sheet block by block against the reference data: input protection and fuse, C1 derating, "
             "U1 input/dropout/current/thermal, C2 minimum, U2 decoupling and ADDR strap, I2C pull-ups and pin "
             "mapping, net labels, LED polarity and current.\n" + ("\n".join(x["why"] for x in f) or "Every check passes."))
    return q, a, think, {"type": "codes", "codes": sorted({x["code"] for x in f})}


def t_fix(d, ctx, rng):
    f = ctx["findings"]
    if not f:
        return None
    x = rng.choice(f)
    refs = ", ".join(x["refdes"])
    q = rng.choice([f"How would you fix the {refs} issue?", f"What should I change for {refs}?"])
    # A component can have several faults; identify which one is being fixed.
    q += f" Specifically: {x['why']}"
    a = f"For {refs}: {FIX[x['code']]}"
    gold = list(x["refdes"])
    return q, a, f"The issue: {x['why']} Fix it at the source.", {"type": "refdes", "refdes": gold}


def t_net(d, ctx, rng):
    nets = ctx["nets"]
    name = rng.choice([n for n in ("VIN", "3V3", "SDA", "SCL", "GND") if n in nets])
    pins = nets[name]
    q = rng.choice([f"What's connected to the {name} net?", f"List everything on {name}.", f"Trace {name} for me."])
    a = f"{name} connects {', '.join(pins)}."
    if name == "SDA" and d.sda_label_mismatch:
        a += " Note U2's SDA pin is labeled SDA_1, a different net, so it is not on SDA."
    if name in ("SDA", "SCL") and d.i2c_swapped:
        a += f" Careful: the {name} wire lands on U2's {'SCL' if name == 'SDA' else 'SDA'} pin - the bus is swapped."
    think = f"Follow every wire and label named {name}, including power/ground symbols, and list the pins it touches."
    return q, a, think, {"type": "pins", "pins": pins}


def t_pin(d, ctx, rng):
    nets = ctx["nets"]
    pin = rng.choice(["U2.ADDR", "U2.SDA", "U2.SCL", "U1.VOUT", "U1.VIN", "R3.2", "D2.K", "D2.A"])
    net, others = pins_on(nets, pin)
    q = f"Where does {pin} go?"
    if not others:
        a = f"{pin} isn't connected to anything else" + (" - the ADDR strap is floating." if pin == "U2.ADDR" else ".")
    else:
        a = f"{pin} is on {net}, together with {', '.join(others)}."
    return q, a, f"Find {pin} on the symbol and follow its wire.", {"type": "pins", "pins": others, "query": pin}


def t_lookup(d, ctx, rng):
    item = rng.choice(ctx["bom"])
    q = rng.choice([f"What is {item['refdes']}?", f"What value is {item['refdes']}?", f"Read {item['refdes']} off the sheet."])
    a = f"{item['refdes']} is a {item['type'].replace('_', ' ')}, {item['value']}" + (f", {item['rating']}" if item["rating"] else "") + "."
    return q, a, f"Read the text next to {item['refdes']}.", {"type": "value", "value": item["value"]}


def t_missing_part(d, ctx, rng):
    present = {b["refdes"] for b in ctx["bom"]}
    ref = rng.choice([r for r in ("C4", "C7", "R5", "U3", "D3", "L1", "Q1") if r not in present]
                     + ([] if d.has_tvs else ["D1"]) + ([] if d.has_c3 else ["C3"]))
    q = rng.choice([f"What value is {ref}?", f"Where is {ref} connected?"])
    a = f"There's no {ref} on this sheet."
    return q, a, f"Search the sheet for {ref}: not present.", {"type": "refuse"}


def t_not_on_sheet(d, ctx, rng):
    q, a = rng.choice(NOT_ON_SHEET)
    return q, a, "That is not schematic information.", {"type": "refuse"}


def t_presence(d, ctx, rng):
    kind = rng.choice(["tvs", "decoupling"])
    if kind == "tvs":
        q = "Is there surge/ESD protection on the input?"
        a = (f"Yes, D1 ({d.tvs}, {hs.TVS[d.tvs]:g} V stand-off) clamps VIN to GND after F1." if d.has_tvs
             else "No. There's no TVS on VIN; J1 feeds F1 and C1 directly.")
        gold = {"type": "yesno", "yes": d.has_tvs}
    else:
        q = "Does U2 have a decoupling cap?"
        a = (f"Yes, C3 ({hs.fmt_cap_nf(d.c3_nf)}) from U2.VDD to GND." if d.has_c3
             else "No. Nothing is drawn between U2.VDD and GND.")
        gold = {"type": "yesno", "yes": d.has_c3}
    return q, a, "Look for the part at that location.", gold


def t_calc(d, ctx, rng):
    ldo = hs.LDOS[d.ldo]
    vmin = hs.VIN_MIN[d.vin]
    kind = rng.choice(["led", "pd", "dropout", "sink"])
    if kind == "led":
        forward = 1.3 / d.r3 * 1000
        v = 0.0 if d.led_reversed else forward
        unit = "mA"
        q = "What's the LED current?"
        a = f"Forward-biased, I = (3.3 - 2.0) V / {hs.fmt_ohm(d.r3)} = {forward:.4f} mA, assuming 2.0 V Vf."
        if d.led_reversed:
            a += " But D2 is drawn reversed, so the idealized current as drawn is I = 0.0000 mA."
    elif kind == "pd":
        v = (d.vin - 3.3) * d.i_load
        unit = "W"
        q = "How much power does U1 dissipate?"
        a = f"Pd = ({d.vin:g} - 3.3) V x {d.i_load*1000:.0f} mA = {v:.4f} W; the {d.ldo} limit is {ldo['pd_max']:.2f} W."
    elif kind == "dropout":
        v = vmin - ldo["dropout"]
        unit = "V"
        q = "Does U1 have enough headroom at minimum input?"
        a = (f"At {vmin:g} V min, {vmin:g} - {ldo['dropout']:g} V dropout = {v:.4f} V available. "
             + ("That's enough for 3.3 V." if v >= 3.4 else "That's not enough to hold 3.3 V."))
    else:
        r = min(d.r1, d.r2)
        v = 3.3 / r * 1000
        unit = "mA"
        q = "How much current do the I2C pull-ups sink when the line is low?"
        a = f"Worst case is the {hs.fmt_ohm(r)} pull-up: 3.3 V / {hs.fmt_ohm(r)} = {v:.4f} mA (I2C limit 3 mA)."
    return q, a, "Pull the inputs from the sheet and the reference data, then compute. " + a, {"type": "number", "value": round(v, 4), "unit": unit}


def t_whatif(d, ctx, rng):
    new = rng.choice([v for v in (150, 330, 1000, 2200) if v != d.r3])
    forward = 1.3 / new * 1000
    v = 0.0 if d.led_reversed else forward
    q = f"If I change R3 to {hs.fmt_ohm(new)}, what's the LED current?"
    a = f"Forward-biased, I = (3.3 - 2.0) V / {hs.fmt_ohm(new)} = {forward:.4f} mA."
    if d.led_reversed:
        a += " D2 is still reversed, so changing only R3 leaves the idealized current at I = 0.0000 mA. Fix D2's polarity first."
    return q, a, f"Keep the 2.0 V Vf and 3.3 V rail, swap in the new resistor. {a}", {"type": "number", "value": round(v, 4), "unit": "mA"}


GENERATORS = [t_net, t_pin, t_lookup, t_calc, t_whatif, t_presence, t_missing_part, t_not_on_sheet]


def build_conversation(d, rng):
    ctx = {"findings": hs.check_design(d), "nets": hs.netlist(d), "bom": hs.bom(d)}
    turns = []
    if rng.random() < 0.6:                       # most chats include a review
        turns.append(t_review(d, ctx, rng))
        if ctx["findings"] and rng.random() < 0.7:
            turns.append(t_fix(d, ctx, rng))
    for g in rng.sample(GENERATORS, rng.randint(2, 4)):
        turns.append(g(d, ctx, rng))
    turns = [t for t in turns if t]
    # The review and its dependent fix stay first. rng.sample already varies
    # the independent question order without moving a fix ahead of its review.
    return turns[:6], ctx


def to_messages(img, d, turns, use_think):
    ref = hs.datasheet_context(d)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    for i, (q, a, think, _) in enumerate(turns):
        content = []
        if i == 0:
            content.append({"type": "image", "image": img})
            q = f"Reference data:\n{ref}\n\n{q}"
        content.append({"type": "text", "text": q})
        msgs.append({"role": "user", "content": content})
        last = i == len(turns) - 1
        text = f"<think>\n{think}\n</think>\n\n{a}" if (last and use_think) else a
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return msgs


def _validate_settings(n_designs, seed, chats_per_design, think_ratio, dpi):
    for name, value, minimum in (("n_designs", n_designs, 3),
                                 ("chats_per_design", chats_per_design, 1), ("dpi", dpi, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(think_ratio, bool) or not isinstance(think_ratio, (int, float)) or not math.isfinite(think_ratio) or not 0 <= think_ratio <= 1:
        raise ValueError("think_ratio must be between 0 and 1")


def _check_destination(out, overwrite):
    if out.is_symlink():
        raise ValueError("output must not be a symbolic link")
    if out.exists():
        if not out.is_dir():
            raise ValueError("output must be a directory")
        if not overwrite:
            raise FileExistsError(f"{out} exists; use --overwrite to replace a generated dataset")
        if any(out.iterdir()) and not (
            all((out / f"{split}.jsonl").is_file() for split in ("train", "val", "test"))
            and (out / "images").is_dir()
        ):
            raise ValueError("refusing to replace a non-dataset directory; choose a new --out directory")
        owned = {"train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "images"}
        unexpected = sorted(path.name for path in out.iterdir() if path.name not in owned)
        if unexpected:
            raise ValueError(f"output contains unrelated paths {unexpected}; generate into a dedicated dataset directory")


def _seeded_rng(seed, *parts):
    # Separate sampling from chat generation so changing the chat count or
    # augmentation does not silently change every later schematic.
    value = json.dumps([seed, *parts], separators=(",", ":")).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(value).digest(), "big"))


def _write_dataset(stage, *, n_designs, seed, chats_per_design, think_ratio, dpi):
    rng = random.Random(seed)
    img_dir = stage / "images"
    img_dir.mkdir()
    ids = [f"D{i:04d}" for i in range(n_designs)]
    rng.shuffle(ids)
    n = max(1, n_designs // 10)
    split_of = {**{i: "test" for i in ids[:n]}, **{i: "val" for i in ids[n:2 * n]}, **{i: "train" for i in ids[2 * n:]}}
    stats = {split: {"designs": 0, "conversations": 0, "turns": 0, "clean_designs": 0}
             for split in ("train", "val", "test")}
    designs = []
    with ExitStack() as stack:
        files = {split: stack.enter_context((stage / f"{split}.jsonl").open("w", encoding="utf-8"))
                 for split in stats}
        for did in sorted(ids):
            split = split_of[did]
            design_rng = _seeded_rng(seed, "design", did)
            rate = design_rng.choice([0.0, 0.3, 0.3, 0.45])
            design = hs.sample_design(did, design_rng, fault_rate=rate)
            base = img_dir / f"{did}.png"
            hs.render(design, str(base), dpi=dpi)
            context = {"findings": hs.check_design(design), "nets": hs.netlist(design), "bom": hs.bom(design)}
            designs.append({"design_id": did, "split": split, "parameters": design.to_dict(),
                            "image": base.relative_to(stage).as_posix(), "fault_rate": rate,
                            "netlist": context["nets"], "bom": context["bom"], "findings": context["findings"]})
            stats[split]["designs"] += 1
            stats[split]["clean_designs"] += int(not context["findings"])
            for chat_index in range(chats_per_design if split == "train" else 1):
                chat_rng = _seeded_rng(seed, "chat", did, chat_index)
                image_path = base
                if split == "train" and chat_rng.random() < 0.35:
                    image_path = base.with_name(f"{did}_aug{chat_index}.jpg")
                    augment(base, image_path, chat_rng)
                turns, ctx = build_conversation(design, chat_rng)
                use_think = split != "train" or chat_rng.random() < think_ratio
                row = {"schema_version": 2, "id": f"{did}-c{chat_index}", "design_id": did, "split": split,
                       "messages": to_messages(image_path.relative_to(stage).as_posix(), design, turns, use_think),
                       "gold": [turn[3] for turn in turns],
                       "design_truth": {"findings": ctx["findings"], "pairs": canonical_pairs(ctx["nets"])}}
                validate_gold_rows([row])
                files[split].write(json.dumps(row, sort_keys=True) + "\n")
                stats[split]["conversations"] += 1
                stats[split]["turns"] += len(turns)
    hashes = {path.relative_to(stage).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(stage.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": 2,
        "kind": "synthetic_schematic_conversations",
        "generator": "build_dataset.py",
        "image_path_base": "dataset_directory",
        "seed": seed,
        "settings": {"n_designs": n_designs, "chats_per_design": chats_per_design,
                     "think_ratio": think_ratio, "dpi": dpi},
        "limitations": ["One synthetic topology with simplified component rules; not a verified hardware design.",
                        "Numerical answers use the stated idealized supply, load, and diode assumptions."],
        "splits": stats,
        "designs": designs,
        "sha256": hashes,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate_dataset(out, *, n_designs=3000, seed=7, chats_per_design=2,
                     think_ratio=0.8, dpi=110, overwrite=False):
    """Stage a complete dataset, then replace the destination with rollback.

    Existing data survives rendering/validation failures. Overwrite explicitly
    replaces the whole dataset directory, including old augmented images.
    """
    _validate_settings(n_designs, seed, chats_per_design, think_ratio, dpi)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    destination = Path(out).expanduser().absolute()
    _check_destination(destination, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.staging-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        stage = temporary / "dataset"
        stage.mkdir()
        manifest = _write_dataset(stage, n_designs=n_designs, seed=seed, chats_per_design=chats_per_design,
                                  think_ratio=think_ratio, dpi=dpi)
        # Recheck after generation in case another writer created the target.
        _check_destination(destination, overwrite)
        existed = destination.exists()
        backup_container = None
        if existed:
            # Keep the backup outside automatic stage cleanup so a failed
            # rollback never deletes the user's previous dataset.
            backup_container = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
            backup = backup_container / "dataset"
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if existed:
                try:
                    os.replace(backup, destination)
                except OSError as restore_error:
                    raise OSError(f"dataset publish and rollback failed; previous data preserved at {backup}") from restore_error
                backup_container.rmdir()
            raise
        if backup_container is not None:
            try:
                shutil.rmtree(backup_container)
            except OSError:
                warnings.warn(f"dataset published, but old backup remains at {backup_container}", stacklevel=2)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate validated synthetic schematic conversations.")
    ap.add_argument("--n-designs", type=int, default=3000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chats-per-design", type=int, default=2)
    ap.add_argument("--think-ratio", type=float, default=0.8,
                    help="fraction of training conversations with final-turn reasoning (0..1)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing generated dataset after staging succeeds")
    args = ap.parse_args(argv)
    try:
        manifest = generate_dataset(args.out, n_designs=args.n_designs, seed=args.seed,
                                    chats_per_design=args.chats_per_design, think_ratio=args.think_ratio,
                                    dpi=args.dpi, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print({split: stats["conversations"] for split, stats in manifest["splits"].items()})
    return 0


if __name__ == "__main__":
    main()
