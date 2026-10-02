"""Synthetic-data truth, reproducibility, and safe publication tests."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import build_dataset as dataset
import evaluate
import hwsynth


def fast_render(design, path, dpi=110):
    """A valid, distinct image for staging tests that do not inspect symbols."""
    color = hashlib.sha256(design.design_id.encode()).digest()[:3]
    Image.new("RGB", (32, 32), tuple(color)).save(path)


class DesignTruthTests(unittest.TestCase):
    def test_invalid_design_parameters_are_rejected(self):
        invalid = [{"r3": 0}, {"vin": 9}, {"i_load": -1}, {"c2_uf": math.nan},
                   {"has_tvs": "yes"}, {"ldo": "unknown"}, {"sensor": "unknown"},
                   {"addr_tie": "unknown"}, {"i2c_khz": 200}, {"r1": 2200.5}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                hwsynth.Design("test", **values)
        with self.assertRaises(ValueError):
            hwsynth.Design("../escape")

    def test_mutated_design_is_validated_at_public_boundaries(self):
        design = hwsynth.Design("test")
        design.r3 = 0
        for operation in (hwsynth.check_design, hwsynth.bom, hwsynth.netlist, hwsynth.datasheet_context):
            with self.subTest(operation=operation.__name__), self.assertRaises(ValueError):
                operation(design)

    def test_fault_rate_is_validated(self):
        for rate in (-0.1, 1.1, math.nan, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                hwsynth.sample_design("test", random.Random(0), rate)

    def test_reversed_led_current_is_zero_for_calculation_and_what_if(self):
        design = hwsynth.Design("reversed", led_reversed=True)
        rng = unittest.mock.Mock()
        rng.choice.return_value = "led"
        calculation = dataset.t_calc(design, {}, rng)
        rng.choice.return_value = 1000
        what_if = dataset.t_whatif(design, {}, rng)
        for turn in (calculation, what_if):
            self.assertEqual(turn[3], {"type": "number", "value": 0, "unit": "mA"})
            self.assertIn("I = 0.0000 mA", turn[1])
            self.assertTrue(evaluate.grade(turn[3], turn[1])["number.acc"])

    def test_each_generated_answer_satisfies_its_gold(self):
        rng = random.Random(572)
        count = 0
        seen_types, seen_codes = set(), set()
        for index in range(200):
            design = hwsynth.sample_design(f"test{index}", rng, fault_rate=rng.choice([0, 0.3, 0.8]))
            seen_codes.update(item["code"] for item in hwsynth.check_design(design))
            for _ in range(3):
                turns, _ = dataset.build_conversation(design, rng)
                evaluate.validate_gold_rows([{"id": "test", "gold": [turn[3] for turn in turns]}])
                review_seen = False
                for _, answer, _, gold in turns:
                    kind = gold["type"]
                    seen_types.add(kind)
                    if kind == "codes":
                        review_seen = True
                    if kind == "refdes":
                        self.assertTrue(review_seen, "fix appeared before the review it references")
                        self.assertTrue(gold["refdes"])
                    if kind == "number":
                        self.assertIn(gold["unit"], {"mA", "W", "V"})
                    for metric, value in evaluate.grade(gold, answer).items():
                        if value is not None:
                            expected = 0 if metric == "review.false_alarm_on_clean" else 1
                            self.assertEqual(value, expected, (design.design_id, gold, answer, metric))
                    count += 1
        self.assertGreater(count, 2000)
        self.assertEqual(seen_types, {"codes", "refdes", "pins", "number", "value", "yesno", "refuse"})
        self.assertEqual(seen_codes, set(hwsynth.SEVERITY))

    def test_netlist_tracks_fault_topology(self):
        design = hwsynth.Design("faults", i2c_swapped=True, sda_label_mismatch=True,
                                led_reversed=True, has_c3=False, has_tvs=False, addr_tie="float")
        nets = hwsynth.netlist(design)
        self.assertEqual(nets["SDA"], ["R1.2"])
        self.assertEqual(nets["SDA_1"], ["U2.SCL"])
        self.assertEqual(nets["SCL"], ["R2.2", "U2.SDA"])
        self.assertEqual(nets["U2_ADDR_NC"], ["U2.ADDR"])
        self.assertIn("D2.K", nets["LED_A"])
        pins = [pin for net in nets.values() for pin in net]
        self.assertEqual(len(pins), len(set(pins)))
        self.assertFalse(any(pin.startswith(("D1.", "C3.")) for pin in pins))
        self.assertFalse({"D1", "C3"} & {item["refdes"] for item in hwsynth.bom(design)})

    def test_actual_renderer_writes_a_readable_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schematic.png"
            hwsynth.render(hwsynth.Design("render-smoke"), str(path), dpi=60)
            with Image.open(path) as image:
                image.load()
                self.assertGreater(image.width, 500)
                self.assertGreater(image.height, 300)
                self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 2)


class DatasetPublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.out = self.root / "data"
        self.renderer = patch.object(hwsynth, "render", side_effect=fast_render)
        self.renderer.start()
        self.addCleanup(self.renderer.stop)

    def generate(self, out=None, **kwargs):
        return dataset.generate_dataset(out or self.out, n_designs=10, **kwargs)

    def test_manifest_paths_hashes_and_disjoint_splits(self):
        manifest = self.generate(seed=19)
        self.assertEqual(manifest, json.loads((self.out / "manifest.json").read_text()))
        self.assertEqual(manifest["seed"], 19)
        self.assertEqual(manifest["settings"]["n_designs"], 10)
        ids = {}
        for split in ("train", "val", "test"):
            rows = evaluate.load_gold(self.out / f"{split}.jsonl")
            ids[split] = {row["design_id"] for row in rows}
            self.assertEqual(len(ids[split]), manifest["splits"][split]["designs"])
            self.assertEqual(len(rows), manifest["splits"][split]["conversations"])
            for row in rows:
                for message in row["messages"]:
                    for content in message["content"]:
                        if content["type"] == "image":
                            self.assertTrue(content["image"].startswith("images/"))
                            self.assertTrue((self.out / content["image"]).is_file())
                            if split != "train":
                                self.assertTrue(content["image"].endswith(".png"))
        self.assertFalse(ids["train"] & ids["val"] | ids["train"] & ids["test"] | ids["val"] & ids["test"])
        for relative, digest in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((self.out / relative).read_bytes()).hexdigest(), digest)
        for truth in manifest["designs"]:
            design = hwsynth.Design(**truth["parameters"])
            self.assertEqual(hwsynth.netlist(design), truth["netlist"])
            self.assertEqual(hwsynth.bom(design), truth["bom"])
            self.assertEqual(hwsynth.check_design(design), truth["findings"])

    def test_generation_is_reproducible_and_chat_count_does_not_change_designs(self):
        first = self.generate(seed=11)
        second = self.generate(self.root / "again", seed=11)
        self.assertEqual(first, second)
        more_chats = self.generate(self.root / "more", seed=11, chats_per_design=4)
        self.assertEqual(first["designs"], more_chats["designs"])

    def test_existing_output_requires_explicit_overwrite(self):
        original = self.generate()
        with self.assertRaises(FileExistsError):
            self.generate(seed=42)
        self.assertEqual(original, json.loads((self.out / "manifest.json").read_text()))
        replaced = self.generate(seed=42, overwrite=True)
        self.assertEqual(replaced["seed"], 42)
        self.assertFalse(list(self.root.glob(".data.backup-*")))

    def test_render_failure_preserves_existing_dataset(self):
        self.generate()
        original = {path.relative_to(self.out): path.read_bytes() for path in self.out.rglob("*") if path.is_file()}
        with patch.object(hwsynth, "render", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.generate(overwrite=True)
        actual = {path.relative_to(self.out): path.read_bytes() for path in self.out.rglob("*") if path.is_file()}
        self.assertEqual(actual, original)
        self.assertFalse(list(self.root.glob(".data.staging-*")))

    def test_publication_failure_rolls_back_existing_dataset(self):
        self.generate(seed=7)
        original = (self.out / "manifest.json").read_bytes()
        replace = os.replace

        def fail_publish(source, target):
            if Path(source).name == "dataset" and ".staging-" in str(source):
                raise OSError("publish failed")
            return replace(source, target)

        with patch.object(dataset.os, "replace", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.generate(seed=42, overwrite=True)
        self.assertEqual((self.out / "manifest.json").read_bytes(), original)

    def test_failed_rollback_keeps_backup_available(self):
        self.generate(seed=7)
        original = (self.out / "manifest.json").read_bytes()
        replace = os.replace

        def fail_after_backup(source, target):
            if Path(source).name == "dataset":
                raise OSError("destination unavailable")
            return replace(source, target)

        with patch.object(dataset.os, "replace", side_effect=fail_after_backup):
            with self.assertRaisesRegex(OSError, "previous data preserved"):
                self.generate(seed=42, overwrite=True)
        backups = list(self.root.glob(".data.backup-*/dataset/manifest.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_invalid_arguments_do_not_touch_existing_output(self):
        self.generate()
        original = (self.out / "manifest.json").read_bytes()
        for kwargs in ({"n_designs": -1}, {"n_designs": 2}, {"chats_per_design": 0},
                       {"think_ratio": math.nan}, {"think_ratio": 1.1}, {"dpi": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dataset.generate_dataset(self.out, overwrite=True, **kwargs)
        self.assertEqual((self.out / "manifest.json").read_bytes(), original)

    def test_overwrite_refuses_unrelated_directory(self):
        self.out.mkdir()
        (self.out / "important.txt").write_text("keep me")
        with self.assertRaisesRegex(ValueError, "non-dataset"):
            self.generate(overwrite=True)
        self.assertEqual((self.out / "important.txt").read_text(), "keep me")

    def test_overwrite_refuses_dataset_with_unrelated_subdirectory(self):
        self.generate()
        (self.out / "real").mkdir()
        (self.out / "real" / "source.txt").write_text("preserve real corpus")
        with self.assertRaisesRegex(ValueError, "unrelated paths"):
            self.generate(overwrite=True)
        self.assertEqual((self.out / "real" / "source.txt").read_text(), "preserve real corpus")


if __name__ == "__main__":
    unittest.main()
