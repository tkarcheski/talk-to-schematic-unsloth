# Agent configuration and data boundaries

The gateway connects the schematic workspace to an operator-selected, self-hosted OpenAI-compatible vision model. It never falls back to a cloud model. Configure it with `config/agent.example.json`; configuration is read at startup, not supplied by the browser.

The permission vocabulary follows [OpenCode's allow / ask / deny model](https://opencode.ai/docs/permissions/). This is a separate implementation, not an OpenCode configuration loader.

| Capability | Default | Execution boundary |
|---|---|---|
| Show a label, region, or full sheet | Allow | Browser viewer; no file mutation |
| Arithmetic | Allow | Bounded numeric operations in the gateway |
| Web search | Disabled without a search endpoint | Once configured, every exact query requires approval |
| Shell commands and file edits | Unavailable | Outside the first release |
| Unknown tools | Deny | Rejected by the registry |

Set `permission` by tool name, with `"*": "deny"` as the default. Search cannot inherit `allow`. A search approval displays the query and configured destination; it cannot authorize different arguments or be replayed. Denial returns a tool result to the model without sending the query. The gateway sends only the approved query to search, never the whole conversation. Search results are untrusted content.

Inference endpoints must use a literal private or loopback IP, an explicit port, and an HTTP(S) URL without credentials, query, or fragment. Redirects and environment proxies are disabled. Credentials may be read from the environment variable named by `api_key_env`; keep them out of configuration files and Git. Remote image URLs are rejected; uploads are rendered in the browser and sent to the configured inference endpoint when you ask a question.

## Sensitive data

The application does not persist chats in browser storage, export them to training, or log request bodies. Pending approvals retain conversation data in server memory for up to five minutes, with at most eight pending approvals and a 32 MiB aggregate serialized-state limit. Python memory overhead is additional; expiry does not guarantee physical memory erasure. Conversations remain in browser memory until reset or the page is closed. Downloads, screenshots, and training exports require separate deliberate handling.

The selected inference and search services are trusted infrastructure. Their logs, retention, transport encryption, telemetry, and any upstream connections are the operator's responsibility. A private IP does not prove that a service is offline. A self-hosted search service normally queries external search engines. Leave the search endpoint unset for offline use; the Compose default also isolates runtime networking.

## Deployment scope and limits

This release targets one trusted user on loopback. Host and Origin checks prevent cross-origin browser requests; they are not user authentication. Approvals are single-use bearer tokens, not tenant accounts. Do not expose the unauthenticated service directly to a LAN or the internet. Container wildcard binding is explicit and must be paired with a loopback-only published port.

The agent runs at most six model steps per request by default (configurable from one to twelve), with bounded messages, responses, HTTP concurrency, and output tokens. The browser also bounds viewer rounds. Stop cancels browser waiting; it does not forcibly interrupt an already running model or approved search. The progress bar shows activity and elapsed time, not an estimated completion percentage or private model reasoning.

Actual tool use depends on the model. The v1 adapter was not trained on viewer or search commands, and its image-only accuracy remains below the project's qualification gate. Test your chosen model and review its answers before relying on them.
