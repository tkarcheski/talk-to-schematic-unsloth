import io
import json
from email.message import Message
from unittest.mock import patch
from urllib.request import HTTPHandler, build_opener
from urllib.response import addinfourl

import pytest
from PIL import Image

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


def test_image_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    dataset, rows = make_dataset(tmp_path)
    monkeypatch.chdir("/")
    assert resolve_image("sheet.png", dataset) == tmp_path / "sheet.png"
    block = convert_messages(rows[0]["messages"], dataset)[0]["content"][0]
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_ambiguous_image_paths_are_rejected(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    dataset, _ = make_dataset(nested)
    Image.new("RGB", (12, 12), "black").save(tmp_path / "sheet.png")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_image("sheet.png", dataset)


def test_same_file_alias_is_not_an_ambiguous_image(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    dataset, _ = make_dataset(nested)
    (tmp_path / "sheet.png").symlink_to(nested / "sheet.png")
    assert resolve_image("sheet.png", dataset) == nested / "sheet.png"
