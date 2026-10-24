"""Probe an actual CPU gateway container without invoking model inference."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    origin = 'http://127.0.0.1:8893'
    for attempt in range(30):
        try:
            with urlopen(origin + '/health', timeout=2) as response:
                health = json.load(response)
            break
        except (URLError, TimeoutError):
            if attempt == 29:
                raise
            time.sleep(1)
    assert health['provider'] == 'self_hosted'
    with urlopen(origin + '/', timeout=2) as response:
        assert response.status == 200
        assert response.headers['Cache-Control'] == 'no-store'
        assert b'<html' in response.read().lower()
    with urlopen(origin + '/api/agent/config', timeout=2) as response:
        config = json.load(response)
    assert config['enabled'] and config['network'] == 'disabled'
    request = Request(origin + '/api/agent/chat', data=b'{}',
                      headers={'Content-Type': 'application/json', 'Origin': 'http://untrusted.invalid'})
    try:
        urlopen(request, timeout=2)
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError('Gateway accepted a cross-origin request')
    print('CPU container startup, UI, privacy headers and origin policy passed; inference untested.')


if __name__ == '__main__':
    main()
