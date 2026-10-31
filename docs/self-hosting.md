<!-- cspell:words venv urlopen noexec SEARXNG -->
# Self-hosted deployment with Docker Compose

The default Compose stack runs the single-user browser workspace and native
Unsloth server with existing local base weights. It does not download models or
use a cloud fallback. The gateway-only stack connects to an existing private
OpenAI-compatible inference server without starting a GPU service.

## Start the private workspace

Place a complete model at `models/base`, or set `SCHEMATIC_BASE_DIR` to its
reviewed directory, then run:

```sh
docker compose up --build
```

To serve an adapter, explicitly add its overlay:

```sh
export SCHEMATIC_ADAPTER_DIR=/path/to/adapter
export SCHEMATIC_BASE_DIR=/path/to/base-model
docker compose -f compose.yaml -f compose.adapter.yaml up --build
```

Only adapter mode copies the adapter into temporary storage and rebinds its base
reference. Original model files remain read-only. Stop with the same Compose
file arguments followed by `down`.

Native serving accepts `SCHEMATIC_CONTEXT_LIMIT`, `SCHEMATIC_MAX_TOKENS`,
`SCHEMATIC_MAX_IMAGE_SIZE`, `SCHEMATIC_MIN_FREE_VRAM_GB`, `SCHEMATIC_MODEL_NAME`,
and `SCHEMATIC_FOUR_BIT` (`true` or `false`). Defaults remain 8192 context tokens,
1024 output tokens, 2048 image pixels and a 10 GiB minimum free-memory check.
These are configuration defaults, not a guarantee that a model fits. Match the
operator's gateway configuration to the native model name and budgets; increasing
one server's limits does not qualify the other server or the selected model.

For an already-running local/private endpoint, edit an operator-owned agent
configuration and use the standalone Linux gateway stack:

```sh
export SCHEMATIC_AGENT_CONFIG=/path/to/agent.json
docker compose -f compose.gateway.yaml up --build
```

This stack uses host networking to reach host-loopback inference, while binding
the UI to loopback only. It has no GPU reservation or native model container.
Unlike the default stack, its networking is not isolated by an internal Docker
network; endpoint restrictions and host network policy remain authoritative.

If the endpoint requires a key, set `api_key_env` in the agent configuration to
its environment variable name. Store only that variable in a private environment
file outside the repository, set `SCHEMATIC_AGENT_ENV_FILE` to that file, and add
`-f compose.gateway-auth.yaml` after `-f compose.gateway.yaml`. Credentials are
not automatically inherited from the shell. Do not commit the environment file
or publish expanded Compose configuration.

Requires Docker daemon access, NVIDIA Container Toolkit, a CUDA 13 compatible
driver and at least 10 GiB free GPU memory (actual needs depend on context and
model). Stop competing inference workloads deliberately before loading another
model. Include complete processor/tokenizer files and Safetensors shards, readable
by UID 10001. Resolve Hugging Face snapshot symlinks into a dedicated directory
first if they reference files outside the mount. Never mount the entire home or
model cache.

Open **http://127.0.0.1:8893** and upload a schematic. No example corpus is bundled
in these images. To use the prepared corpus with component region maps:

```sh
export SCHEMATIC_CORPUS_DIR=/path/to/reviewed-corpus
docker compose -f compose.yaml -f compose.examples.yaml up --build
```

Use `real-v3` for region-aware viewer commands; an older corpus may lack those
coordinates. The gateway can open before inference finishes loading. Check
both containers with `docker compose ps`; the inference health check reports the
actual loaded state, while the UI gateway initially reports its endpoint as
configured. First startup hashes model weights and loads them; allow several
minutes. A failed load does not fall back to another provider.

```sh
docker compose ps
docker compose exec gateway /app/.venv/bin/python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8892/health').read().decode())"
docker compose down
```

Builds fetch OS packages, Python dependencies and container images. Runtime
inference is offline. For an air-gapped installation, build and validate the
images on an approved build machine, transfer them using Docker's image export
and import commands, and start with `docker compose up --no-build --pull never`.
Weights must already be present on the destination. Source, model and image
licensing remain the operator's responsibility.

## What is isolated

- Gateway and inference share one network namespace so the inference server can
  remain loopback-only on port 8892. Only UI port 8893 is published, on host
  `127.0.0.1`. Both use an internal Docker network without an external route.
- Models and configuration are read-only bind mounts. Startup copies the adapter
  into ephemeral `/tmp` and rewrites its base reference to `/models/base`; it
  never changes the original adapter. The 4 GiB temporary allocation also holds
  compilation caches, and the GPU service has 2 GiB shared memory.
- Containers run without root, with all Linux capabilities removed and privilege
  escalation disabled. No Docker socket, host credentials, host IPC, or host
  filesystem access is supplied. CUDA compilation needs executable temporary
  storage; the temporary mounts intentionally do not use `noexec`.
- No application conversations, uploads or training exports are retained by this
  deployment. Docker log retention is disabled because upstream exceptions may
  contain content. Startup output is still visible when attached to Compose;
  don't redirect it into a shared log without reviewing your retention policy.
  Swap, host access and browser memory remain host-level responsibilities.
- Browser Host and Origin validation stays enabled. Keep the published port and
  gateway port equal when changing them. This package is not a public multi-user
  deployment; add a separately reviewed authentication/TLS boundary before
  exposing it beyond loopback.

The fixed private subnet is `172.30.91.0/24`. If that conflicts with your network,
change the subnet and service addresses together, including the search endpoint
in `config/compose-agent-search.json`. No shell or workspace write tool is enabled.

## Optional web search

Search is disabled in the default configuration. The separate overlay starts
SearXNG and gives **only the search container** an outbound network. Its internal
address is `172.30.91.3:8080`; no search port is published. A self-hosted search
service still sends approved queries to public search engines. The browser shows
the exact query and configured search-service destination before allowing one
request; do not approve queries containing sensitive design information.

Generate a new secret in your shell, then explicitly opt in:

```sh
export SEARXNG_SECRET="$(openssl rand -hex 32)"
docker compose -f compose.yaml -f compose.search.yaml up --build
```

The secret is required by Compose and is passed only to SearXNG. Do not commit it
or paste expanded `docker compose config` output into a public issue. This overlay
changes `web_search` from `deny` to `ask`, enables SearXNG's JSON format, disables
its metrics, and keeps its cache ephemeral. Requests are subject to upstream
search-engine behavior and availability. Stop the same stack with:

```sh
docker compose -f compose.yaml -f compose.search.yaml down
```

## Reproducibility and verification

The gateway installs the committed `uv.lock` using Python 3.13. The GPU image pins
core packages in `docker/requirements-inference.txt`, including CUDA 13 PyTorch
and Unsloth. Its complete
transitive GPU environment is **not yet locked**. Python, CUDA and SearXNG images are pinned to digests read from their official
Docker Hub registry metadata on 2026-10-07. uv is pinned to release 0.12.19.
Review replacements through Dockerfile build arguments (`PYTHON_IMAGE`, `UV_IMAGE`,
`CUDA_IMAGE`) and `SEARXNG_IMAGE`; build once, test and retain the resulting images.

CI builds and starts the CPU gateway container, then checks its UI, health,
privacy headers and origin rejection without model inference. Passing that job
does not validate the GPU image, model quality, memory fit, or optional search.
The GPU image and model-specific serving require separate runtime verification.
The dependency gate audits locked runtime/development registry packages and npm;
the pinned Git audit tool requires source review, and GPU extras outside `uv.lock`
are not covered by that registry audit. Advisory queries require networking;
unavailable audit services fail the gate rather than returning a clean result.

After building, verify loaded inference, an actual schematic question, viewer
tool use, search deny/allow behavior, and absence of runtime egress with search
disabled before introducing sensitive data. Do not enable diagnostics that retain
private content merely to make these checks convenient.

The network and GPU declarations follow Docker's
[internal network](https://docs.docker.com/reference/compose-file/networks/) and
[GPU reservation](https://docs.docker.com/compose/how-tos/gpu-support/)
documentation. Optional search follows the upstream
[SearXNG container guide](https://docs.searxng.org/admin/installation-docker.html)
and [settings reference](https://docs.searxng.org/admin/settings/settings.html).
