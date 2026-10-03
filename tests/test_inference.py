import io
import hashlib
import json
from email.message import Message
from unittest.mock import patch
from urllib.request import HTTPHandler, build_opener
from urllib.response import addinfourl

import pytest
from PIL import Image

from predict import predict
from schematic_model.inference import Client, InferenceError, _RejectRedirects, convert_messages, resolve_image


def response(content="answer", reason="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                             "finish_reason": reason}]}).encode())


@pytest.mark.parametrize("content,reason", [(None, "stop"), ("", "stop"), ("partial", "length"),
                                            ("<think>unfinished", "stop")])
def test_incomplete_response_is_not_a_prediction(content, reason):
    with patch("schematic_model.inference._OPENER.open", return_value=response(content, reason)):
        with pytest.raises(InferenceError):
            Client().complete("local", [])


def test_success_removes_reasoning_and_keeps_final_answer():
    with patch("schematic_model.inference._OPENER.open", return_value=response("<think>notes</think>Answer")):
        assert Client().complete("local", [])["output"] == "Answer"


def test_credentials_are_headers_not_url():
    with pytest.raises(ValueError):
        Client("http://user:secret@localhost/v1")
    with patch("schematic_model.inference._OPENER.open", return_value=response()) as send:
        Client(api_key="test-only").complete("local", [])
        assert send.call_args.args[0].get_header("Authorization") == "Bearer test-only"


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("destination", ["http://other-host.invalid/collect", "http://localhost:8888/elsewhere"])
def test_redirects_never_forward_authenticated_requests(code, destination):
    class RedirectTransport(HTTPHandler):
        """In-memory urllib transport: no listening socket or network request."""
        def __init__(self):
            super().__init__()
            self.requests = []

        def http_open(self, request):
            self.requests.append(request)
            headers = Message()
            headers["Location"] = destination
            result = addinfourl(io.BytesIO(b""), headers, request.full_url, code=code)
            result.msg = "Redirect"
            return result

    transport = RedirectTransport()
    opener = build_opener(transport, _RejectRedirects())
    with patch("schematic_model.inference._OPENER", opener):
        with pytest.raises(InferenceError, match="redirects are disabled"):
            Client(api_key="dummy-test-key").complete("local", [])
    assert len(transport.requests) == 1
    assert transport.requests[0].full_url == "http://localhost:8888/v1/chat/completions"
    assert transport.requests[0].get_header("Authorization") == "Bearer dummy-test-key"


def make_dataset(tmp_path):
    Image.new("RGB", (12, 12), "white").save(tmp_path / "sheet.png")
    rows = [{"id": "real-1", "gold": [{"type": "value", "value": "10k"}, {"type": "refuse"}],
             "messages": [
                 {"role": "user", "content": [{"type": "image", "image": "sheet.png"},
                                               {"type": "text", "text": "Read R1"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "R1 is 10k"}]},
                 {"role": "user", "content": [{"type": "text", "text": "What is R99?"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "R99 is not on the sheet"}]},
             ]}]
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(rows[0]) + "\n")
    return dataset, rows


class FakeClient:
    base_url = "http://localhost:8888/v1"

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def complete(self, model, messages, **kwargs):
        self.calls.append(list(messages))
        if len(self.calls) == self.fail_at:
            raise InferenceError("simulated interruption")
        return {"output": "model answer", "finish_reason": "stop"}


def test_image_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    dataset, rows = make_dataset(tmp_path)
    monkeypatch.chdir("/")
    assert resolve_image("sheet.png", dataset) == tmp_path / "sheet.png"
    block = convert_messages(rows[0]["messages"], dataset)[0]["content"][0]
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_ambiguous_image_paths_fail_before_output_creation(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    dataset, _ = make_dataset(nested)
    Image.new("RGB", (12, 12), "black").save(tmp_path / "sheet.png")
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError, match="Ambiguous"):
        predict(dataset, output, "local", FakeClient())
    assert not output.exists()
    assert not output.with_suffix(".jsonl.meta.json").exists()


def test_same_file_alias_is_not_an_ambiguous_image(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    dataset, _ = make_dataset(nested)
    (tmp_path / "sheet.png").symlink_to(nested / "sheet.png")
    assert resolve_image("sheet.png", dataset) == nested / "sheet.png"


def test_resume_keeps_completed_turns_after_failure(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(InferenceError):
        predict(dataset, output, "local", FakeClient(fail_at=2))
    first_line = output.read_text()
    client = FakeClient()
    report = predict(dataset, output, "local", client, resume=True)
    assert report["predictions"] == 2
    assert len(client.calls) == 1
    assert output.read_text().startswith(first_line)
    assert len(output.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="metadata"):
        predict(dataset, output, "different-model", client, resume=True)


def test_resume_rejects_changed_images_without_touching_outputs(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(InferenceError):
        predict(dataset, output, "local", FakeClient(fail_at=2))
    metadata = output.with_suffix(".jsonl.meta.json")
    before = output.read_bytes(), metadata.read_bytes()
    assert json.loads(metadata.read_text())["images_sha256"] == {
        "sheet.png": hashlib.sha256((tmp_path / "sheet.png").read_bytes()).hexdigest()
    }
    Image.new("RGB", (12, 12), "black").save(tmp_path / "sheet.png")
    client = FakeClient()
    with pytest.raises(ValueError, match="metadata.*images"):
        predict(dataset, output, "local", client, resume=True)
    assert not client.calls
    assert (output.read_bytes(), metadata.read_bytes()) == before


def test_all_referenced_images_are_bound_in_metadata(tmp_path):
    dataset, rows = make_dataset(tmp_path)
    Image.new("RGB", (12, 12), "black").save(tmp_path / "second.png")
    rows[0]["messages"][2]["content"].append({"type": "image", "image": "second.png"})
    dataset.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "predictions.jsonl"
    predict(dataset, output, "local", FakeClient())
    binding = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
    assert binding["images_sha256"] == {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("sheet.png", "second.png")
    }


def test_image_change_after_preflight_is_rejected_before_request(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    output = tmp_path / "predictions.jsonl"
    client = FakeClient()
    original_convert = convert_messages
    conversions = []

    def mutate_between_validation_and_request(*args, **kwargs):
        conversions.append(True)
        if len(conversions) == 2:
            Image.new("RGB", (12, 12), "black").save(tmp_path / "sheet.png")
        return original_convert(*args, **kwargs)

    with patch("predict.convert_messages", side_effect=mutate_between_validation_and_request):
        with pytest.raises(ValueError, match="Image bytes changed"):
            predict(dataset, output, "local", client)
    assert not client.calls
    assert output.read_bytes() == b""


def test_generated_history_uses_actual_previous_prediction(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    client = FakeClient()
    predict(dataset, tmp_path / "predictions.jsonl", "local", client, history="generated")
    assert client.calls[1][1] == {"role": "assistant", "content": "model answer"}


def test_bad_images_do_not_create_output(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    (tmp_path / "sheet.png").write_text("not an image")
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(OSError):
        predict(dataset, output, "local", FakeClient())
    assert not output.exists()
    assert not output.with_suffix(".jsonl.meta.json").exists()
