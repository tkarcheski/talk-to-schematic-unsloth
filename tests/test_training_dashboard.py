"""Real HTTP requests must not expose private state through a rebound host."""

from contextlib import closing
from http.client import HTTPConnection
import json
import threading

import pytest

from scripts import training_dashboard as dashboard


@pytest.fixture
def running_dashboard(tmp_path, monkeypatch):
    reads = []

    def snapshot(run, log):
        reads.append((run, log))
        return {"status": "fixture"}

    monkeypatch.setattr(dashboard, "snapshot", snapshot)
    with dashboard.make_server(tmp_path, port=0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server.server_port, reads
        finally:
            server.shutdown()
            worker.join(timeout=5)


def request(port, path, hosts):
    with closing(HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.putrequest("GET", path, skip_host=True)
        for host in hosts:
            connection.putheader("Host", host)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.read()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST"])
def test_local_authority_can_read_dashboard(running_dashboard, host):
    port, reads = running_dashboard
    status, body = request(port, "/api/state", [f"{host}:{port}"])
    assert status == 200
    assert json.loads(body) == {"status": "fixture", "jobs": []}
    assert len(reads) == 1
    assert request(port, "/", [f"{host}:{port}"])[0] == 200


@pytest.mark.parametrize("path", ["/", "/api/state"])
@pytest.mark.parametrize("hosts", [[], ["attacker.example:{port}"], ["localhost"],
                                  ["127.0.0.1:1"], ["localhost:{port}", "attacker.example"],
                                  ["localhost:{port}@attacker.example"], ["localhost:{port},attacker.example"]])
def test_bad_or_ambiguous_host_is_rejected_before_reading_state(running_dashboard, path, hosts):
    port, reads = running_dashboard
    status, body = request(port, path, [host.format(port=port) for host in hosts])
    assert status == 403
    assert b"fixture" not in body
    assert reads == []
