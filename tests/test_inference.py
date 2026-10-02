import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from predict import predict
from schematic_model.inference import Client, InferenceError, convert_messages, resolve_image


def response(content="answer", reason="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                             "finish_reason": reason}]}).encode())


@pytest.mark.parametrize("content,reason", [(None, "stop"), ("", "stop"), ("partial", "length"),
                                            ("<think>unfinished", "stop")])
def test_incomplete_response_is_not_a_prediction(content, reason):
    with patch("schematic_model.inference.urlopen", return_value=response(content, reason)):
        with pytest.raises(InferenceError):
            Client().complete("local", [])


def test_success_removes_reasoning_and_keeps_final_answer():
    with patch("schematic_model.inference.urlopen", return_value=response("<think>notes</think>Answer")):
        assert Client().complete("local", [])["output"] == "Answer"


def test_credentials_are_headers_not_url():
    with pytest.raises(ValueError):
        Client("http://user:secret@localhost/v1")
    with patch("schematic_model.inference.urlopen", return_value=response()) as send:
        Client(api_key="test-only").complete("local", [])
        assert send.call_args.args[0].get_header("Authorization") == "Bearer test-only"


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
