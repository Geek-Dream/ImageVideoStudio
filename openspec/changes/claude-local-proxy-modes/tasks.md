# Claude Local Proxy Modes Implementation Tasks

- [x] Add contract tests for proxy mode normalization, Claude message conversion, tool results, SSE events, and CC Switch lookup.
- [x] Implement `claude_proxy.py` with Anthropic Messages request/response conversion, streaming, health endpoint, and bounded process logs.
- [x] Update `llm.py` to persist `proxy_mode`, preserve legacy preferences, and enforce one shared proxy on `8848`.
- [x] Update `gen.py` model settings UI and submit payload to use the three-mode control and show active mode.
- [x] Add the `claude-local` launcher and CC Switch configuration reader.
- [x] Run focused tests and the existing test suite; record verification results in this change.
