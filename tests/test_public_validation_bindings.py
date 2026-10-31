"""Public document hashes must stay usable without rewriting historical evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/validation"
V1 = "generalization-eval-plan.json"
V2 = "generalization-eval-plan-v2.json"
ORIGINAL_REVIEW = "generalization-readability.json"
SERVING_REVIEW = "generalization-readability-2048.json"
SNAPSHOT = "generalization-snapshot-v2.json"
TRAINING = "real-crop-v1-training.json"

# Original document fingerprints contain no machine paths. Payload fingerprints
# pin all non-path evidence before public-copy bindings were repaired: targets,
# labels, observations, dimensions, measured metrics and frozen artifact hashes.
HISTORICAL = {
    V1: ("a920982e21966f9505ac98c4898fb34dc3ccc0564de728fc8b5ada28ecfef39c",
         "4115c498ba0a7fb206f6fa3cb30e8cff0cc1cca9606d1d8ea4527bf7d25f62f1"),
    V2: ("2a4c0e7d0fce6689f81d08d52120b350db2bef9c5c6a832f54c720c9bb6c5263",
         "8ee37068b04395e6853ac2c875b95ecbb0e5bbdd0e241c5ff0a1942d2c956db5"),
    ORIGINAL_REVIEW: ("9e7c68d7ba06e3820d28ce9b51a3259c2c1a0042ea0f3637fd3499ed3675bc5d",
                      "070316b7cad3cb47413be325ef5d73268faedfe25ce0e056b43f249c496adef2"),
    SERVING_REVIEW: ("5c1172b8f750490520fc73f6a2008bccf293bd24bbd8a7c2b4c4fa6376b011a5",
                     "8985403078e875aa39bca7f37008340c317bc159f3145b81790a03d02046d42e"),
    SNAPSHOT: ("f3920a64b593f451c520e3d1f0b998623035761882a25f5c8bc7f65102f93068",
               "97869138c4ff125ec00ff57c75e41d6f691feae1b286282aa7610e4a156f24d0"),
    TRAINING: ("df61990aefae75d2e18ff942773feff99b09fc4e5b48ac2a0eeee16563fe51b7",
               "0f83f58501f0c6a8a9a4d07fa9c017e2ad4b8dc1f6c72db406e1af7bed81ff64"),
}
HISTORICAL_PLANS = {
    V1: "010e037e864ec0a7c12ff85a38aa31167f1354a391e9edf4df11ae0af3321e6a",
    V2: "31354cbf66e43bfe8dd10753b009ab0296fb269ce5790b6bba5e9a3a4da464dd",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    # This is the canonical encoding used by prepare_generalization_eval.py.
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def read(name):
    return json.loads((DIRECTORY / name).read_bytes())


def file_hash(relative):
    path = Path(relative)
    assert not path.is_absolute() and ".." not in path.parts
    return digest((ROOT / path).read_bytes())


@pytest.mark.parametrize("name", [V1, V2])
def test_current_public_plan_hash_uses_current_canonical_content(name):
    plan = read(name)
    assert plan["plan_sha256"] == digest(canonical({
        key: value for key, value in plan.items() if key != "plan_sha256"
    }))
    assert plan["public_copy"]["historical_plan_sha256"] == HISTORICAL_PLANS[name]
    assert plan["plan_sha256"] != HISTORICAL_PLANS[name]


def test_public_plan_override_bindings_match_files_consumed_by_builder():
    prior, plan = read(V1), read(V2)
    inputs = {entry["path"]: entry["sha256"] for entry in plan["inputs"]}
    for path, fingerprint in inputs.items():
        if path.startswith("docs/"):
            assert fingerprint == file_hash(path)
    for override in plan["value_overrides"]:
        for path_key, hash_key in (("prior_plan_path", "prior_plan_sha256"),
                                   ("overlap_report_path", "overlap_report_sha256")):
            assert override[hash_key] == file_hash(override[path_key]) == inputs[override[path_key]]
        original = next(case for case in prior["cases"]
                        if case["category"] == "value" and case["board_id"] == override["board_id"])
        assert original["target"] == {
            "refdes": override["original_refdes"], "literal_value": override["original_value"]
        }
        assert all(original[key] == override[key] for key in ("source_sha256", "image_sha256"))


def test_public_reviews_and_snapshot_link_current_documents_with_historical_bindings():
    original, serving, snapshot = read(ORIGINAL_REVIEW), read(SERVING_REVIEW), read(SNAPSHOT)
    original_inputs = original["inputs"]
    assert original_inputs["plan_file_sha256"] == file_hash(original_inputs["plan_path"])
    assert original_inputs["plan_content_sha256"] == read(V1)["plan_sha256"]
    inputs = serving["inputs"]
    for prefix, name in (("original", V1), ("final", V2)):
        assert inputs[prefix + "_plan_file_sha256"] == file_hash(inputs[prefix + "_plan_path"])
        assert inputs[prefix + "_plan_content_sha256"] == read(name)["plan_sha256"]
    assert inputs["prior_original_resolution_report_sha256"] == file_hash(
        inputs["prior_original_resolution_report_path"])
    assert snapshot["plan_file_sha256"] == file_hash(snapshot["plan_file"])
    assert snapshot["plan_sha256"] == read(V2)["plan_sha256"]
    assert snapshot["serving_readability_review_sha256"] == file_hash(snapshot["serving_readability_review"])
    historical = snapshot["public_copy"]["historical_document_bindings"]
    assert historical["plan_sha256"] == HISTORICAL_PLANS[V2]
    assert historical["plan_file_sha256"] == HISTORICAL[V2][0]
    assert historical["serving_readability_review_sha256"] == HISTORICAL[SERVING_REVIEW][0]


@pytest.mark.parametrize("review_name,plan_name", [(ORIGINAL_REVIEW, V1), (SERVING_REVIEW, V2)])
def test_historical_review_targets_still_match_public_plan(review_name, plan_name):
    cases = {case["id"]: case for case in read(plan_name)["cases"] if case["category"] == "value"}
    reviewed = read(review_name)["reviewed_targets"]
    assert {target["id"] for target in reviewed} == set(cases)
    for target in reviewed:
        case = cases[target["id"]]
        for key in ("board_id", "family_id", "target", "source_sha256", "image_sha256", "page"):
            assert target[key] == case[key]


def without_paths(value):
    if isinstance(value, dict):
        return {key: without_paths(item) for key, item in value.items()
                if key not in {"path", "native_source_path", "frozen_image_path", "plan_sha256"}}
    if isinstance(value, list):
        return [without_paths(item) for item in value]
    return value


@pytest.mark.parametrize("name", HISTORICAL)
def test_original_targets_measurements_and_frozen_identities_are_unchanged(name):
    document = copy.deepcopy(read(name))
    public = document.pop("public_copy")
    assert public["historical_file_sha256"] == HISTORICAL[name][0]
    if name in (ORIGINAL_REVIEW, SERVING_REVIEW):
        document["inputs"] = public["historical_inputs"]
    elif name == V2:
        historical_inputs = {item["path"]: item["sha256"] for item in public["historical_inputs"]}
        for item in document["inputs"]:
            if item["path"] in historical_inputs:
                item["sha256"] = historical_inputs[item["path"]]
        for override in document["value_overrides"]:
            override["prior_plan_sha256"] = historical_inputs[override["prior_plan_path"]]
            override["overlap_report_sha256"] = historical_inputs[override["overlap_report_path"]]
    elif name == SNAPSHOT:
        document.update(public["historical_document_bindings"])
    assert digest(canonical(without_paths(document))) == HISTORICAL[name][1]


@pytest.mark.parametrize("name", HISTORICAL)
def test_public_copy_has_no_absolute_filesystem_references(name):
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and (key == "path" or key.endswith("_path")):
                    assert not Path(item).is_absolute() and not item.startswith("file:")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(read(name))
