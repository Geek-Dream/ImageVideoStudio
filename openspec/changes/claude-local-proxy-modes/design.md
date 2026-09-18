# Design: Claude Local Proxy Modes

## Architecture

The model launcher stores `proxy_mode` as an integer (`0` disabled, `1` Codex, `2` Claude) and accepts legacy `codex_proxy` preferences as mode `1`. Mode `1` starts the existing `codex_proxy.py`; mode `2` starts the new `claude_proxy.py`; both bind `127.0.0.1:8848` and forward to the llama-server backend on `127.0.0.1:8846`. Before either proxy starts, the current proxy PID is stopped and the shared public port is checked.

`claude_proxy.py` accepts Anthropic `POST /v1/messages` requests, maps system/user/assistant/tool-result blocks into llama.cpp chat-completions messages, maps OpenAI tool calls back into Anthropic `tool_use` blocks, and emits Anthropic-compatible non-stream and SSE responses. It remains standard-library-only and writes the same bounded per-process proxy log style as the Codex adapter.

`claude-local` is an executable project launcher. It locates `/Users/wl/.cc-switch/cc-switch.db` (with an environment override), reads the provider record whose name/model identifies the local Qwen3.6-35B entry, and uses its base URL, model, and API key. It does not read or mutate the active-provider selection. It starts/reuses the local llama backend and Claude proxy through the project launcher, exports `ANTHROPIC_BASE_URL=http://127.0.0.1:8848`, and `exec`s `claude` with all user arguments.

## Interfaces

- `proxy_mode`: `0 | 1 | 2` in model preferences and `/api/llm/start` request payload.
- `claude_proxy.py --listen 8848 --target 127.0.0.1:8846`.
- `claude-local [claude arguments...]`.
- Claude proxy endpoints: `POST /v1/messages`, `GET /health`.

## Compatibility and failure handling

- Existing saved `codex_proxy: true` is interpreted as `proxy_mode=1`.
- Metal/vMLX models keep their existing restrictions; Claude mode is offered only for llama.cpp models that support the adapter.
- A second proxy never starts while the first owns `8848`; the launcher stops the recorded proxy and verifies port release first.
- Unsupported Anthropic features return an explicit HTTP 400/501 JSON error instead of leaking malformed `<tool_code>` text.

## Verification

2026-08-31: `python3 -m unittest discover -s tests -v` passed all 28 tests. `python3 -m py_compile claude_proxy.py claude_local_config.py llm.py gen.py`, `zsh -n claude-local`, and `git diff --check` passed. A temporary-port HTTP smoke test confirmed `/v1/messages` is forwarded as `/v1/chat/completions` and the response is converted back to Anthropic format. Existing listeners on `8846`, `8848`, and `8860` were left running and unchanged.
