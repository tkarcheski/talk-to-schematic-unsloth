"""Public-source research primitives; no private conversation inputs or cloud fallback."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import time
from urllib.parse import urlsplit


ALLOWED_LICENSES = {'CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0', 'CERN-OHL-P-2.0', 'CERN-OHL-W-2.0', 'CERN-OHL-S-2.0'}


def fetch_public(source, hosts, *, max_bytes=8_000_000):
    """Download an operator-approved public source. Redirects are refused, DNS pinned."""
    if set(source) != {'url', 'license', 'license_url', 'public'} or source['public'] is not True:
        raise ValueError('public provenance required')
    if source['license'] not in ALLOWED_LICENSES:
        raise ValueError('license needs review')
    target = urlsplit(source['url'])
    license_url = urlsplit(source['license_url'])
    if target.scheme != 'https' or target.hostname not in hosts or target.username or target.password or target.port not in (None, 443):
        raise ValueError('source outside HTTPS allowlist')
    if target.query or target.fragment or license_url.scheme != 'https' or license_url.hostname not in hosts or license_url.username or license_url.password or license_url.query or license_url.fragment or license_url.port not in (None, 443):
        raise ValueError('query/fragment or license provenance not allowed')
    addresses = socket.getaddrinfo(target.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('nonpublic destination')
    address = addresses[0][4][0]
    connection = http.client.HTTPSConnection(target.hostname, timeout=20)
    raw = socket.create_connection((address, 443), timeout=20)
    connection.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=target.hostname)
    try:
        connection.request('GET', target.path or '/', headers={'User-Agent': 'AskTheSchematic-public-research/1'})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('source response rejected; redirects require separate review')
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError('source too large')
        return data, {**source, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    finally:
        connection.close()


def studio_recommend(endpoint, model, candidates, *, key_env='UNSLOTH_API_KEY'):
    """Local Studio advisory inference only; accepts opaque queue metadata, never documents."""
    target = urlsplit(endpoint)
    if target.scheme != 'http' or target.hostname not in ('127.0.0.1', '::1') or target.username or target.password or target.query or target.fragment:
        raise ValueError('Studio must be an explicit loopback endpoint')
    key = os.environ.get(key_env)
    if not key:
        raise ValueError('Studio authentication required')
    if any(set(c) != {'id', 'kind'} or c['kind'] not in ('research', 'training', 'proposal') or not isinstance(c['id'], str) or len(c['id']) != 32 or any(x not in '0123456789abcdef' for x in c['id']) for c in candidates):
        raise ValueError('only opaque queue metadata allowed')
    # Operator must first verify residency; this helper intentionally has no load/unload API.
    status = local_json(endpoint, '/api/inference/status', key=key)
    if status.get('loading') or status.get('model_identifier') != model:
        raise ValueError('Studio resident model handoff required')
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Choose one job ID from this queue. Output only its ID: ' + json.dumps(candidates)}], 'max_tokens': 64, 'stream': False}
    body = local_json(endpoint, '/api/inference/chat/completions', payload, key=key)
    answer = body['choices'][0]['message']['content'].strip()
    return answer if answer in {c['id'] for c in candidates} else None


def local_json(endpoint, path, payload=None, *, key=None):
    """Loopback-only JSON request, no proxies, redirects or fallback."""
    target = urlsplit(endpoint)
    if target.scheme != 'http' or target.hostname not in ('127.0.0.1', '::1') or target.username or target.password or target.path not in ('', '/') or target.query or target.fragment:
        raise ValueError('explicit loopback origin required')
    remaining = float(os.environ.get('SCHEMATIC_DEADLINE', time.time() + 30)) - time.time()
    if remaining < 1:
        raise TimeoutError('cooperative deadline reached')
    connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=min(30, remaining))
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    try:
        connection.request('POST' if payload is not None else 'GET', path,
                           body=json.dumps(payload) if payload is not None else None, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('local endpoint request rejected')
        body = response.read(128_001)
        if len(body) > 128_000:
            raise ValueError('local response too large')
        return json.loads(body)
    finally:
        connection.close()


SAFE_FAILURE_CODES = {'authentication_required', 'invalid_configuration', 'io_failure',
                      'deadline_reached', 'inference_failure', 'unexpected_failure'}


def report_failure(error):
    """Write only a fixed classification, never exception text, paths or credentials."""
    if isinstance(error, TimeoutError):
        code = 'deadline_reached'
    elif isinstance(error, ValueError):
        code = 'invalid_configuration'
    elif isinstance(error, OSError):
        code = 'io_failure'
    elif type(error).__name__ == 'InferenceError':
        code = 'inference_failure'
    else:
        code = 'unexpected_failure'
    destination = os.environ.get('SCHEMATIC_RESULT_PATH')
    if destination:
        from pathlib import Path
        Path(destination).write_text(json.dumps({'error_code': code}))
    return code
