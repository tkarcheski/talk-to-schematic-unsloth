# Chat template fixture

`unsloth-qwen3.5-4b-3764fa35.jinja` is the `chat_template` from
[unsloth/Qwen3.5-4B at 3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636](https://huggingface.co/unsloth/Qwen3.5-4B/blob/3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636/chat_template.jinja),
the pinned base model revision. It is byte-identical to `chat_template` in that
revision's `tokenizer_config.json`. Qwen3.5 is released under the Apache License 2.0.

`tests/test_view_tools.py` renders it to prove that the viewer tool wire format
in `schematic_model/view_tools.py` produces the same prompt text as native tool
messages.
