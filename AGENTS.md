# Agent rules

## Get approval before changing direction

- Do not make changes without the user's approval of the direction. This covers code, data, config, Git (commits, pushes, branches), training runs, publications and board posts.
- When you see a problem, raise it to the user. Do not work around it, and do not copy a workaround a previous agent used.
- Propose the direction, wait for approval, then act. Approval covers only what was approved; ask again for anything beyond it.

## Product requirements approved by the user

- The application name is **Ask the Schematic**.
- Deliver a working, production-quality application for viewing schematics and chatting with AI. Preserve responsive, accessible viewing, navigation, uploads, source context, and clear loading, failure, retry, and tool-execution states.
- While AI is busy, show an obvious animated progress bar, elapsed time, current activity, and a Stop control. Use indeterminate activity when actual completion progress is unavailable; never fabricate percentages or expose private model reasoning as progress.
- Let the user configure the schematic/chat split with a draggable, keyboard-accessible divider and a width setting. Remember only the layout preference in browser storage; keep conversations and schematic content out of persistent browser storage.
- Show the model's name and version/artifact identity, token usage, elapsed inference time, and measured output tokens per second when supplied by the backend. State timing scope and unavailable fields honestly; distinguish operator labels from verified model identity.
- Keep the chat composer editable while inference runs. Disable concurrent submission, not drafting; never erase text typed during a pending request when that request completes, fails, or is stopped.
- Provide session command history in the composer: Up recalls older prompts, Down moves forward, and the current unfinished draft is restored at the newest position. Respect multiline editing and keep prompt history in memory only.
- Let users edit an additional system prompt in Workspace settings, including recurring schematic-review feedback. Apply changes to the next request, including existing conversations; server tool and data permissions remain authoritative.
- Show a detailed, expandable request trace with model-visible context, tool arguments/results, approvals, errors, timing, and usage. Show reasoning text only when explicitly returned by the self-hosted provider, never invented private reasoning. Keep traces in session memory and exclude credentials and raw image payloads.
- Add a real-model regression for selected-part web-search requests being incorrectly routed into unrelated viewer actions. Preserve selected-component context, provide a real self-hosted search service, and retrain a separate candidate on source-grounded routing examples. Preserve the current adapter and benchmark splits; switch defaults only after routing and schematic regression checks pass.
- Do not move, zoom, highlight, or select schematic parts merely because the model proposes it during discussion, review, or search. Bind any model-driven viewer action to the user's approval of the exact action; prior navigation consent must not authorize later turns or different targets.
- Expand real-model evaluation to multi-turn conversations with actual tool-result feedback, stale/cleared focus, page changes, and explicit no-navigation turns. Score unsolicited actions, viewport invariance, and clean tool-turn termination separately from answer correctness. Continue retraining separate candidates without weakening these gates.
- Continue dataset work on attributable multi-page schematics, longer-context tests, and expert hardware-design questions. Keep proposed questions distinct from source-verified or human-reviewed answers; do not manufacture defect labels or mix held-out board families into training.
- Provide a training/research dashboard and scripts for collecting public sources, preparing dataset PRs, and sharing the local GPU between self-hosted research-agent inference and Unsloth training. Use a single GPU owner, explicit budgets, recoverable job state, and cooperative checkpoint boundaries. AI scheduling recommendations must pass resource, provenance, and privacy checks; never silently overlap workloads or bypass Studio authentication.
- Research and dataset publication automation may operate only on reviewed public-source material. Sensitive schematics, conversations, and training data must not enter public PRs. Keep merge/close decisions with the user.
- Support fully self-hosted AI end to end. Inference and the training loop must work without a cloud AI account or external inference service. Self-hosted model endpoints must be configurable; never silently fall back to a cloud provider.
- Treat schematics, conversations, tool results, and training data as highly sensitive. External transmission, telemetry, logging of content, retention, and training-data export must be deliberate operator choices, not hidden defaults. Do not commit private data or credentials.
- Use OpenCode-inspired declarative configuration and explicit tool permissions. Distinguish allow, ask, and deny; enforce decisions in the backend, fail closed on unknown tools or invalid configuration, and document the effective policy.
- Provide agentic interaction, including schematic/viewer tool calls and configurable web search. Design tools to be extensible. External search is optional and disabled until configured; approving a search must identify the actual query and destination. Do not send a full private conversation or schematic to search implicitly.
- Fully self-hosted operation and optional external tools must coexist: the application must remain usable with external network access disabled. A self-hosted search service can still contact the public internet; make that boundary explicit.
- Broad tool access is a product objective, not blanket permission for arbitrary host execution or unrestricted data egress. Bring major tool-scope/security conflicts to the user before expanding access. Bind approvals to the exact action and arguments; enforce resource and tool-round limits.
- The user selected the first-release tool scope: viewer commands, schematic tools, and approval-gated web search with an expandable tool registry. Shell execution and workspace file edits are outside this first release.
- Provide Docker Compose as the documented application startup path, including self-hosted inference and an optional search service. Mount model/data inputs explicitly and keep them private; do not bundle sensitive data into images. Validate the Compose configuration and distinguish container verification from native-process verification.
- Distinguish UI previews and scripted responses from actual model inference. Verify the real serving path and report remaining model-quality limitations; passing UI tests does not qualify a model or establish production readiness.

## Delivery and collaboration

- The user authorized sub-agents and isolated Git worktree checkouts for this effort. Give each agent bounded ownership and coordinate integration through the private message board; do not publish that board.
- Review all open PRs and post useful comments with findings, context, and verification. Prepare reviewable PRs for completed work. The user will merge or close PRs; agents must not do either on the user's behalf.
- Bring major conflicts to the user, including incompatible branch changes, deletion of useful evidence, security tradeoffs, and changes to sensitive-data handling. Preserve unmerged work until its disposition is approved.
- Record requirement changes here as the user makes decisions. Separate requirements from implemented capabilities and verified results; do not describe planned controls as already enforced.
