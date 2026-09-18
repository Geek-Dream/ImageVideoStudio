# Execution Notes

1. Work on one numbered task at a time and check it off only after its tests pass.
2. Do not modify existing tests to make an implementation pass; change implementation instead.
3. Follow red-green-refactor: run each new test once while it fails for the intended reason before writing production code.
4. Do not commit or push during implementation; repository flow owns delivery.
5. Keep edits within this change's stated files; report unrelated worktree changes without reverting them.
6. If the same failure persists through two attempted fixes, stop and report the exact command and output.

## Interface contract

- `proxy_mode` is numeric: `0=off`, `1=Codex`, `2=Claude`; legacy boolean `codex_proxy=true` maps to `1`.
- The public proxy is always `127.0.0.1:8848`; the llama backend is `127.0.0.1:8846` when a proxy is selected.
- Claude request conversion must preserve system text, conversation order, image blocks, tool definitions, tool-use IDs, stop reason, and usage counts where the upstream response provides them.
- The CC Switch active-provider field is ignored; only the named local Qwen provider's endpoint/model/key are consumed.
