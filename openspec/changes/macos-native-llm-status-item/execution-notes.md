# Execution Notes

1. Work on one numbered task at a time and check it off only after its tests pass.
2. Do not modify existing tests to make an implementation pass; change implementation instead.
3. Follow red-green-refactor: run each new test once while it fails for the intended reason before writing production code.
4. Do not commit or push during implementation; repository flow owns delivery.
5. Keep edits within this change's stated files; report unrelated worktree changes without reverting them.
6. If the same failure persists through two attempted fixes, stop and report the exact command and output.

## Interface Contract

- `format_status(stats)` is a pure function returning the user-facing full status string; no-model returns the waiting text. Item removal is controlled only by the enable switch/process lifecycle.
- `stats` follows `llm.stats()`: `loading`, `model`, and `activity` may be absent or malformed; mapping must never raise.
- `activity.state` values are `prompt`, `gen`, `tool`, `idle`; unknown values map to an 异常 status. Completed requests must return `idle` even when older log lines contain generation timings.
- macOS lifecycle is controlled by `start.sh`; non-Darwin execution must be a no-op.
