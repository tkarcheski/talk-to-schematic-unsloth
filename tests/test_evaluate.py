"""Adversarial grading checks and evaluation boundary tests; no model required."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate


def gold_row(*labels, row_id="design-c0"):
    return {"id": row_id, "gold": list(labels)}


class GradingTests(unittest.TestCase):
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

    def test_floating_pin_rejects_invented_far_end(self):
        label = {"type": "pins", "pins": [], "query": "U2.ADDR"}
        self.assertFalse(evaluate.grade(label, "U2.ADDR is floating but connected to R1.2.")["pins.f1"])
        self.assertTrue(evaluate.grade(label, "U2.ADDR is floating.")["pins.f1"])

    def test_pin_f1_counts_extra_pins_and_long_designators(self):
        label = {"type": "pins", "pins": ["R101.2"], "query": "U2.SDA"}
        self.assertEqual(evaluate.grade(label, "U2.SDA connects to R101.2.")["pins.f1"], 1)
        self.assertAlmostEqual(evaluate.grade(label, "R101.2 and R2.2.")["pins.f1"], 2 / 3)

    def test_real_eda_reference_and_pad_names(self):
        label = {"type": "pins", "pins": ["USB.VBUS", "U$1.P$2", "CN_2.D+"]}
        rows = [gold_row(label)]
        metrics, _ = evaluate.score(rows, {"design-c0#0": "USB.VBUS, U$1.P$2, CN_2.D+."})
        self.assertEqual(metrics["pins.f1"], 1)

    def test_literal_net_name_is_not_an_extra_pin(self):
        label = {"type": "pins", "pins": ["JP2.5", "U5.6"], "net": "MISO_3.3V"}
        self.assertEqual(evaluate.grade(label, "On this sheet, MISO_3.3V connects JP2.5, U5.6.")["pins.f1"], 1)
        trace = {"type": "pins", "pins": ["JP2.5"], "query": "U5.6", "net": "MISO_3.3V"}
        self.assertEqual(evaluate.grade(trace, "U5.6 is on MISO_3.3V with JP2.5.")["pins.f1"], 1)
        self.assertLess(evaluate.grade(label, "MISO_3.3V connects JP2.5, U5.6, U9.2.")["pins.f1"], 1)

    def test_net_metadata_does_not_hide_an_expected_endpoint(self):
        label = {"type": "pins", "pins": ["U5.6"], "net": "U5.6"}
        self.assertEqual(evaluate.grade(label, "U5.6.")["pins.f1"], 1)

    def test_reasoning_does_not_supply_the_answer(self):
        label = {"type": "value", "value": "10k"}
        self.assertFalse(evaluate.grade(label, "<think>R1 is 10k.</think>Uncertain.")["value.acc"])
        self.assertFalse(evaluate.grade(label, "<think>R1 is 10k.")["value.acc"])


class ValidationTests(unittest.TestCase):
    def test_requires_nonempty_gold(self):
        for rows in ([], [gold_row()]):
            with self.subTest(rows=rows), self.assertRaises(evaluate.EvaluationError):
                evaluate.score(rows, {})

    def test_rejects_missing_and_unknown_prediction_keys(self):
        rows = [gold_row({"type": "refuse"})]
        for predictions in ({}, {"wrong#0": "No."}, {"design-c0#0": "No.", "extra#0": "No."}):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(evaluate.EvaluationError, "coverage"):
                evaluate.score(rows, predictions)

    def test_blank_answer_is_counted_as_a_failure(self):
        metrics, counts = evaluate.score([gold_row({"type": "refuse"})], {"design-c0#0": ""})
        self.assertEqual(metrics["refuse.acc"], 0)
        self.assertEqual(counts["refuse.acc"], 1)

    def test_rejects_duplicate_gold_ids(self):
        row = gold_row({"type": "refuse"})
        with self.assertRaisesRegex(evaluate.EvaluationError, "Duplicate gold id"):
            evaluate.score([row, row], {"design-c0#0": "Not on the sheet."})

    def test_rejects_malformed_labels(self):
        labels = [
            {"type": "unknown"}, {"type": "yesno", "yes": True},
            {"type": "number", "value": 1.3}, {"type": "value", "value": ""},
            {"type": "pins", "pins": ["not-a-pin"]},
            {"type": "pins", "pins": ["U2.SDA"], "net": None},
            {"type": "pins", "pins": ["U2.SDA"], "query": "U2.SDA"},
            {"type": "codes", "codes": []},
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
        gates = evaluate.PROFILES["real-grounding"]
        self.assertEqual(len(evaluate.gate_failures({}, {}, gates)), len(gates))
        metrics = {name: 1 for name in gates}
        counts = {name: 1 for name in metrics}
        self.assertEqual(evaluate.gate_failures(metrics, counts, gates), [])
        counts["pins.f1"] = 0
        self.assertIn("pins.f1 (no evidence)", evaluate.gate_failures(metrics, counts, gates))
        counts["pins.f1"] = 1
        metrics["pins.f1"] = math.nan
        self.assertIn("pins.f1 (invalid score)", evaluate.gate_failures(metrics, counts, gates))

    def test_profile_specific_missing_evidence(self):
        metrics = {"value.acc": 1, "refuse.acc": 1}
        counts = {name: 1 for name in metrics}
        self.assertEqual(evaluate.gate_failures(metrics, counts, evaluate.PROFILES["real-vision"]), [])
        self.assertEqual(evaluate.gate_failures(metrics, counts, evaluate.PROFILES["real-grounding"]),
                         ["pins.f1 (no evidence)"])


class ContextualRefusalTests(unittest.TestCase):
    question = "What measured voltage was observed at D1.C during a powered bench test?"

    def test_validation_and_adversarial_contract_matrix(self):
        path = Path(__file__).parents[1] / "docs/validation/refusal-rule-proposal.json"
        cases = json.loads(path.read_text())["cases"]
        self.assertGreaterEqual(len(cases), 24)
        for case in cases:
            with self.subTest(case=case["id"]):
                result = evaluate.contextual_refusal(case["question"], case["answer"])
                self.assertEqual(result["accepted"], case["expected_restricted_rule_acceptance"])
                self.assertEqual(result["version"], "contextual-refusal-v2")
                self.assertEqual(result["status"], "accepted" if result["accepted"] else "manual_review")

    def test_unsupported_questions_and_prose_require_manual_review(self):
        for question, answer in (
            ("What is the trace width?", "Trace width is not shown in the schematic."),
            (self.question, "The evidence is silent about that."),
            (self.question, "No bench measurement is supplied in the schematic. Unknown extra claim."),
            (self.question, "Its measured voltage cannot be determined from the schematic."),
        ):
            with self.subTest(question=question, answer=answer):
                result = evaluate.contextual_refusal(question, answer)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["status"], "manual_review")

    def test_default_v1_is_unchanged_and_v2_rejects_contradictory_tails(self):
        answer = "No measurement is supplied in the schematic, but D1.C measured 3.3 V."
        label = {"type": "refuse"}
        self.assertTrue(evaluate.grade(label, answer)["refuse.acc"])
        self.assertFalse(evaluate.grade(label, answer, question=self.question,
                                        refusal_scorer="v2")["refuse.acc"])

    def test_context_uses_only_aligned_user_text(self):
        row = gold_row({"type": "value", "value": "10K"}, {"type": "refuse"})
        row["messages"] = [
            {"role": "system", "content": "Not a question."},
            {"role": "user", "content": [{"type": "text", "text": "What value is R1?"}]},
            {"role": "assistant", "content": self.question.replace("D1.C", "U9.1")},
            {"role": "user", "content": [{"type": "image", "image": "not-opened.png"},
                                         {"type": "text", "text": self.question}]},
            {"role": "assistant", "content": "Do not use this gold answer as question context."},
        ]
        self.assertEqual(evaluate.question_text(row, 1), self.question)
        answer = "No measured voltage for D1.C is recorded in the supplied schematic evidence."
        metrics, counts = evaluate.score([row], {"design-c0#0": "10K", "design-c0#1": answer},
                                        refusal_scorer="v2")
        self.assertEqual(metrics["refuse.acc"], 1)
        self.assertEqual(counts["refuse.acc"], 1)
        row["messages"][3]["content"][-1]["text"] = self.question.replace("D1.C", "U9.1")
        metrics, _ = evaluate.score([row], {"design-c0#0": "10K", "design-c0#1": answer},
                                   refusal_scorer="v2")
        self.assertEqual(metrics["refuse.acc"], 0)

    def test_v2_requires_well_formed_context_and_known_version(self):
        row = gold_row({"type": "refuse"})
        predictions = {"design-c0#0": "No bench measurement is supplied in the schematic."}
        with self.assertRaisesRegex(evaluate.EvaluationError, "aligned messages"):
            evaluate.score([row], predictions, refusal_scorer="v2")
        with self.assertRaisesRegex(evaluate.EvaluationError, "Unknown refusal scorer"):
            evaluate.score([row], predictions, refusal_scorer="future")
        for content in (None, [], [{"type": "text", "text": 3}], ["plain fragment"],
                        [{"type": "audio", "text": self.question}]):
            row["messages"] = [{"role": "user", "content": content},
                               {"role": "assistant", "content": "irrelevant"}]
            with self.subTest(content=content), self.assertRaises(evaluate.EvaluationError):
                evaluate.score([row], predictions, refusal_scorer="v2")


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
            result = evaluate.main(["--gold", gold, "--pred", pred, "--profile", "real-vision"])
        self.assertEqual(result, 2)
        self.assertIn("BLOCKED", stderr.getvalue())

    def test_cli_missing_metric_evidence_is_blocked(self):
        gold = self.write("gold.jsonl", [gold_row({"type": "refuse"})])
        pred = self.write("pred.jsonl", [{"key": "design-c0#0", "output": "Not on the schematic."}])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = evaluate.main(["--gold", gold, "--pred", pred, "--profile", "real-grounding"])
        self.assertEqual(result, 1)
        self.assertIn("no evidence", stdout.getvalue())

    def test_cli_can_pass_complete_evidence(self):
        labels = [
            {"type": "value", "value": "10k"},
            {"type": "pins", "pins": ["R1.2"]},
            {"type": "refuse"},
        ]
        answers = ["R1 is 10k.", "R1.2.", "Trace width isn't shown on the schematic."]
        gold = self.write("gold.jsonl", [gold_row(*labels)])
        pred = self.write("pred.jsonl", [{"key": f"design-c0#{i}", "output": answer}
                                          for i, answer in enumerate(answers)])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = evaluate.main(["--gold", gold, "--pred", pred, "--profile", "real-grounding"])
        self.assertEqual(result, 0)
        self.assertIn("all gates passed", stdout.getvalue())

    def test_json_report_records_profile_denominators_and_failures(self):
        labels = [{"type": "value", "value": "100 nF"}, {"type": "refuse"}]
        gold = self.write("gold.jsonl", [gold_row(*labels)])
        pred = self.write("pred.jsonl", [{"key": "design-c0#0", "output": "C3 is 100 nF."},
                                         {"key": "design-c0#1", "output": "Not on the schematic."}])
        report_path = self.path / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = evaluate.main(["--gold", gold, "--pred", pred, "--profile", "real-grounding",
                                    "--report", str(report_path)])
        self.assertEqual(status, 1)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["profile"], "real-grounding")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["failures"], ["pins.f1 (no evidence)"])
        self.assertEqual(report["counts"], {"value.acc": 1, "refuse.acc": 1})
        self.assertEqual(len(report["gold_sha256"]), 64)
        self.assertEqual(report["refusal_scorer"], "keyword-refusal-v1")

    def test_v2_cli_report_records_explicit_version(self):
        row = gold_row({"type": "value", "value": "10K"}, {"type": "refuse"})
        row["messages"] = [{"role": "user", "content": "What value is R1?"},
                           {"role": "assistant", "content": "10K"},
                           {"role": "user", "content": ContextualRefusalTests.question},
                           {"role": "assistant", "content": "Not used by scorer."}]
        gold = self.write("gold.jsonl", [row])
        pred = self.write("pred.jsonl", [{"key": "design-c0#0", "output": "10K"},
                                         {"key": "design-c0#1", "output":
                                          "No bench measurement is supplied in the schematic."}])
        report = self.path / "report.json"
        self.assertEqual(self.run_report(gold, pred, report, "--refusal-scorer", "v2"), 0)
        self.assertEqual(json.loads(report.read_text())["refusal_scorer"], "contextual-refusal-v2")
        row.pop("messages")
        self.write("gold.jsonl", [row])
        self.assertEqual(self.run_report(gold, pred, report, "--refusal-scorer", "v2"), 2)
        incomplete = json.loads(report.read_text())
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(incomplete["refusal_scorer"], "contextual-refusal-v2")

    def report_fixture(self):
        gold = self.write("gold.jsonl", [gold_row({"type": "value", "value": "10k"}, {"type": "refuse"})])
        predictions = [{"key": "design-c0#0", "output": "R1 is 10k."},
                       {"key": "design-c0#1", "output": "Not shown on the schematic."}]
        pred = self.write("pred.jsonl", predictions)
        return gold, pred, predictions

    def run_report(self, gold, pred, report, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evaluate.main(["--gold", gold, "--pred", pred, "--profile", "real-vision",
                                  "--report", str(report), *extra])

    def test_incomplete_predictions_replace_a_stale_passing_report(self):
        gold, pred, predictions = self.report_fixture()
        report = self.path / "report.json"
        self.assertEqual(self.run_report(gold, pred, report), 0)
        self.assertEqual(json.loads(report.read_text())["status"], "pass")
        self.write("pred.jsonl", predictions[:-1])
        self.assertEqual(self.run_report(gold, pred, report), 2)
        incomplete = json.loads(report.read_text())
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertIn("missing 1 keys", incomplete["failures"][0])
        self.assertEqual(incomplete["metrics"], {})
        self.assertEqual(incomplete["counts"], {})

    def test_malformed_or_missing_inputs_replace_stale_report(self):
        gold, pred, _ = self.report_fixture()
        report = self.path / "report.json"
        for invalid in (b"broken JSON\n", b"\xff\n", b""):
            report.write_text('{"status":"pass"}\n')
            Path(pred).write_bytes(invalid)
            self.assertEqual(self.run_report(gold, pred, report), 2)
            self.assertEqual(json.loads(report.read_text())["status"], "incomplete")
        Path(pred).unlink()
        report.write_text('{"status":"pass"}\n')
        self.assertEqual(self.run_report(gold, pred, report), 2)
        self.assertEqual(json.loads(report.read_text())["status"], "incomplete")

    def test_report_cannot_alias_gold_predictions_or_baseline(self):
        gold, pred, _ = self.report_fixture()
        baseline = self.path / "baseline.jsonl"
        baseline.write_bytes(Path(pred).read_bytes())
        inputs = [Path(gold), Path(pred), baseline]
        original = {path: path.read_bytes() for path in inputs}
        for source in inputs:
            for alias_kind in ("direct", "symlink", "hardlink"):
                with self.subTest(source=source.name, alias=alias_kind):
                    if alias_kind == "direct":
                        report = source
                    else:
                        report = self.path / f"{source.name}.{alias_kind}"
                        if alias_kind == "symlink":
                            report.symlink_to(source)
                        else:
                            os.link(source, report)
                    self.assertEqual(self.run_report(gold, pred, report, "--pred-base", str(baseline)), 2)
                    self.assertEqual({path: path.read_bytes() for path in inputs}, original)

    def test_unsafe_report_path_is_rejected_even_when_input_is_invalid(self):
        gold, pred, _ = self.report_fixture()
        Path(pred).write_text("malformed\n")
        original = Path(gold).read_bytes()
        self.assertEqual(self.run_report(gold, pred, gold), 2)
        self.assertEqual(Path(gold).read_bytes(), original)

    def test_failed_atomic_report_replacement_preserves_previous_output(self):
        gold, pred, _ = self.report_fixture()
        report = self.path / "report.json"
        original = b'{"status":"incomplete","failures":["previous run"]}\n'
        report.write_bytes(original)
        gold_before, pred_before = Path(gold).read_bytes(), Path(pred).read_bytes()
        with patch("evaluate.os.replace", side_effect=OSError("simulated publish failure")):
            self.assertEqual(self.run_report(gold, pred, report), 2)
        self.assertEqual(report.read_bytes(), original)
        self.assertEqual((Path(gold).read_bytes(), Path(pred).read_bytes()), (gold_before, pred_before))
        self.assertFalse(list(self.path.glob(".report.json.*.tmp")))

    def test_report_is_complete_before_atomic_publication(self):
        gold, pred, _ = self.report_fixture()
        report = self.path / "report.json"
        replace = os.replace
        publications = []

        def inspect_publication(source, target):
            payload = json.loads(Path(source).read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(Path(source).parent, report.parent)
            publications.append(payload)
            return replace(source, target)

        with patch("evaluate.os.replace", side_effect=inspect_publication):
            self.assertEqual(self.run_report(gold, pred, report), 0)
        self.assertEqual(len(publications), 1)
        self.assertEqual(json.loads(report.read_text()), publications[0])


if __name__ == "__main__":
    unittest.main()
