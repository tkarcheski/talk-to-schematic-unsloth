"""Adversarial grading checks and evaluation boundary tests; no model required."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import evaluate


def gold_row(*labels, row_id="design-c0"):
    return {"id": row_id, "gold": list(labels)}


class GradingTests(unittest.TestCase):
    def test_negative_presence_requires_explicit_no(self):
        label = {"type": "yesno", "yes": False}
        for answer in ("", "I cannot tell.", "Yesterday I checked.", "Not sure."):
            with self.subTest(answer=answer):
                self.assertFalse(evaluate.grade(label, answer)["yesno.acc"])
        self.assertTrue(evaluate.grade(label, "No. There is no TVS on VIN.")["yesno.acc"])

    def test_presence_rejects_explicit_contradictions(self):
        self.assertFalse(evaluate.grade({"type": "yesno", "yes": True},
                                        "Yes, there is no capacitor.")["yesno.acc"])
        self.assertFalse(evaluate.grade({"type": "yesno", "yes": False},
                                        "No. Actually yes.")["yesno.acc"])
        self.assertTrue(evaluate.grade({"type": "yesno", "yes": True},
                                       "Yes, C3 decouples U2.")["yesno.acc"])

    def test_value_is_a_bounded_token(self):
        label = {"type": "value", "value": "150"}
        for answer in ("1500 ohms", "-150 ohms", "150.5 ohms", "R150", "0.150"):
            with self.subTest(answer=answer):
                self.assertFalse(evaluate.grade(label, answer)["value.acc"])
        self.assertTrue(evaluate.grade(label, "R3 is 150.")["value.acc"])

    def test_value_permits_spacing_but_not_larger_units(self):
        label = {"type": "value", "value": "100 nF"}
        self.assertTrue(evaluate.grade(label, "C3 = 100nF.")["value.acc"])
        self.assertTrue(evaluate.grade(label, "C3 = 100  nF.")["value.acc"])
        self.assertFalse(evaluate.grade(label, "C3 = 1100 nF.")["value.acc"])
        self.assertFalse(evaluate.grade({"type": "value", "value": "1 A"},
                                        "F1 = 1 Ah.")["value.acc"])

    def test_numeric_dimensions_and_conversion(self):
        label = {"type": "number", "value": 1.3, "unit": "mA"}
        self.assertTrue(evaluate.grade(label, "I = 1.3 mA.")["number.acc"])
        self.assertTrue(evaluate.grade(label, "I = 0.0013 A.")["number.acc"])
        self.assertFalse(evaluate.grade(label, "I = 1.3 W.")["number.acc"])
        self.assertFalse(evaluate.grade(label, "I = 1.3 A.")["number.acc"])

    def test_numeric_final_result_and_relative_tolerance(self):
        label = {"type": "number", "value": 0.013, "unit": "mA"}
        self.assertTrue(evaluate.grade(label, "I = 0.013 mA.")["number.acc"])
        self.assertFalse(evaluate.grade(label, "I = 0.01 mA.")["number.acc"])
        self.assertFalse(evaluate.grade(label, "I = 0.013 mA. Correction: I = 1 mA.")["number.acc"])
        self.assertFalse(evaluate.grade(label, "I = 1e999 mA.")["number.acc"])

    def test_zero_requires_explicit_zero(self):
        label = {"type": "number", "value": 0, "unit": "mA"}
        for answer in ("It won't turn off; it draws 99 mA.", "It won't conduct.", "I = 0.01 mA."):
            with self.subTest(answer=answer):
                self.assertFalse(evaluate.grade(label, answer)["number.acc"])
        self.assertTrue(evaluate.grade(label, "D2 is reversed, so I = 0 mA.")["number.acc"])

    def test_legacy_numeric_labels_remain_magnitude_only(self):
        self.assertTrue(evaluate.grade({"type": "number", "value": 1.3},
                                       "I = 1.3 W.")["number.acc"])

    def test_negated_review_finding_is_not_a_hit(self):
        label = {"type": "codes", "codes": ["LDO_THERMAL"]}
        for answer in ("- U1 has no thermal problems.", "U1 is not overheating."):
            with self.subTest(answer=answer):
                self.assertEqual(evaluate.grade(label, answer)["review.recall"], 0)
        self.assertEqual(evaluate.grade(label, "U1 will overheat.")["review.recall"], 1)

    def test_clean_review_detects_prose_false_alarm(self):
        label = {"type": "codes", "codes": []}
        for answer in ("U1 will overheat and fail.", "No issues found. U1 will overheat.", ""):
            with self.subTest(answer=answer):
                self.assertEqual(evaluate.grade(label, answer)["review.false_alarm_on_clean"], 1)
        self.assertEqual(evaluate.grade(label, "No issues found.")["review.false_alarm_on_clean"], 0)

    def test_empty_fix_labels_never_pass(self):
        self.assertFalse(evaluate.grade({"type": "refdes", "refdes": []}, "")["fix.names_part"])
        self.assertFalse(evaluate.grade({"type": "refdes", "refdes": ["R1"]}, "Use 2.2k.")["fix.names_part"])
        self.assertTrue(evaluate.grade({"type": "refdes", "refdes": ["R1"]}, "Change R1 to 2.2k.")["fix.names_part"])

    def test_floating_pin_rejects_invented_far_end(self):
        label = {"type": "pins", "pins": [], "query": "U2.ADDR"}
        self.assertFalse(evaluate.grade(label, "U2.ADDR is floating but connected to R1.2.")["pins.f1"])
        self.assertTrue(evaluate.grade(label, "U2.ADDR is floating.")["pins.f1"])

    def test_pin_f1_counts_extra_pins_and_long_designators(self):
        label = {"type": "pins", "pins": ["R101.2"], "query": "U2.SDA"}
        self.assertEqual(evaluate.grade(label, "U2.SDA connects to R101.2.")["pins.f1"], 1)
        self.assertAlmostEqual(evaluate.grade(label, "R101.2 and R2.2.")["pins.f1"], 2 / 3)

    def test_reasoning_does_not_supply_the_answer(self):
        label = {"type": "yesno", "yes": True}
        self.assertFalse(evaluate.grade(label, "<think>Yes, C3 is present.</think>Uncertain.")["yesno.acc"])
        self.assertFalse(evaluate.grade(label, "<think>Yes, C3 is present.")["yesno.acc"])


class ValidationTests(unittest.TestCase):
    def test_requires_nonempty_gold(self):
        for rows in ([], [gold_row()]):
            with self.subTest(rows=rows), self.assertRaises(evaluate.EvaluationError):
                evaluate.score(rows, {})

    def test_rejects_missing_and_unknown_prediction_keys(self):
        rows = [gold_row({"type": "yesno", "yes": False})]
        for predictions in ({}, {"wrong#0": "No."}, {"design-c0#0": "No.", "extra#0": "No."}):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(evaluate.EvaluationError, "coverage"):
                evaluate.score(rows, predictions)

    def test_blank_is_present_but_fails_presence(self):
        metrics, counts = evaluate.score([gold_row({"type": "yesno", "yes": False})], {"design-c0#0": ""})
        self.assertEqual(metrics["yesno.acc"], 0)
        self.assertEqual(counts["yesno.acc"], 1)

    def test_rejects_duplicate_gold_ids(self):
        row = gold_row({"type": "refuse"})
        with self.assertRaisesRegex(evaluate.EvaluationError, "Duplicate gold id"):
            evaluate.score([row, row], {"design-c0#0": "Not on the sheet."})

    def test_rejects_malformed_labels(self):
        labels = [
            {"type": "unknown"}, {"type": "yesno", "yes": "false"},
            {"type": "number", "value": True}, {"type": "number", "value": math.nan},
            {"type": "number", "value": 1, "unit": "bananas"},
            {"type": "refdes", "refdes": []}, {"type": "value", "value": ""},
            {"type": "pins", "pins": ["not-a-pin"]},
            {"type": "pins", "pins": ["U2.SDA"], "query": "U2.SDA"},
            {"type": "codes", "codes": ["IMAGINARY_FAULT"]},
            {"type": "codes", "codes": ["LDO_THERMAL", "LDO_THERMAL"]},
        ]
        for label in labels:
            with self.subTest(label=label), self.assertRaises(evaluate.EvaluationError):
                evaluate.score([gold_row(label)], {"design-c0#0": "answer"})

    def test_rejects_messages_gold_misalignment(self):
        row = gold_row({"type": "refuse"})
        row["messages"] = [{"role": "assistant", "content": []}]
        with self.assertRaisesRegex(evaluate.EvaluationError, "alternate"):
            evaluate.score([row], {"design-c0#0": "Not on the sheet."})

    def test_predictions_must_be_strings(self):
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.score([gold_row({"type": "refuse"})], {"design-c0#0": None})

    def test_required_gates_need_evidence(self):
        self.assertEqual(len(evaluate.gate_failures({}, {})), len(evaluate.GATES))
        metrics = {name: 0 if operator == "<=" else 1 for name, (operator, _) in evaluate.GATES.items()}
        counts = {name: 1 for name in metrics}
        self.assertEqual(evaluate.gate_failures(metrics, counts), [])
        counts["pins.f1"] = 0
        self.assertIn("pins.f1 (no evidence)", evaluate.gate_failures(metrics, counts))
        counts["pins.f1"] = 1
        metrics["pins.f1"] = math.nan
        self.assertIn("pins.f1 (invalid score)", evaluate.gate_failures(metrics, counts))


class JsonlAndCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def write(self, name, rows):
        path = self.path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return str(path)

    def test_duplicate_prediction_keys_are_rejected(self):
        path = self.write("pred.jsonl", [{"key": "x#0", "output": "yes"}] * 2)
        with self.assertRaisesRegex(evaluate.EvaluationError, "duplicate prediction"):
            evaluate.load(path)

    def test_malformed_jsonl_reports_location(self):
        path = self.path / "pred.jsonl"
        for text in ('{"key":"x#0","key":"x#1","output":"yes"}\n',
                     '{"key":"x#0","output":NaN}\n', "[]\n", "\n", "broken\n"):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaisesRegex(evaluate.EvaluationError, r"pred.jsonl:1"):
                evaluate.load(path)

    def test_empty_cli_inputs_are_blocked(self):
        gold = self.write("gold.jsonl", [])
        pred = self.write("pred.jsonl", [])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = evaluate.main(["--gold", gold, "--pred", pred])
        self.assertEqual(result, 2)
        self.assertIn("BLOCKED", stderr.getvalue())

    def test_cli_missing_metric_evidence_is_blocked(self):
        gold = self.write("gold.jsonl", [gold_row({"type": "yesno", "yes": True})])
        pred = self.write("pred.jsonl", [{"key": "design-c0#0", "output": "Yes."}])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = evaluate.main(["--gold", gold, "--pred", pred])
        self.assertEqual(result, 1)
        self.assertIn("no evidence", stdout.getvalue())

    def test_cli_can_pass_complete_evidence_and_report_legacy_units(self):
        labels = [
            {"type": "codes", "codes": ["LDO_THERMAL"]},
            {"type": "codes", "codes": []},
            {"type": "pins", "pins": ["R1.2"]},
            {"type": "number", "value": 1.3},
            {"type": "refuse"},
        ]
        answers = ["U1 will overheat.", "No issues found.", "R1.2.", "I = 1.3 mA.",
                   "Trace width isn't shown on the schematic."]
        gold = self.write("gold.jsonl", [gold_row(*labels)])
        pred = self.write("pred.jsonl", [{"key": f"design-c0#{i}", "output": answer}
                                          for i, answer in enumerate(answers)])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = evaluate.main(["--gold", gold, "--pred", pred])
        self.assertEqual(result, 0)
        self.assertIn("all gates passed", stdout.getvalue())
        self.assertIn("1 legacy numeric labels lack units", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
