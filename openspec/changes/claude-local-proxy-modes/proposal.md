# Claude Local Proxy Modes

## Why

ImageVideoStudio currently exposes a boolean Codex proxy switch. The local Qwen model also needs to serve Claude Code, but Claude Code uses Anthropic's `/v1/messages` contract rather than the existing OpenAI Responses contract. A boolean switch cannot select the two mutually exclusive adapters, and both adapters must own the same public model port.

## What Changes

- Replace the model-page "enable proxy" checkbox with a three-mode control: disabled, Codex proxy, or Claude proxy.
- Keep `8848` as the single public proxy port. Starting one proxy stops the other; the llama.cpp backend remains on `8846` while a proxy is active.
- Add `claude_proxy.py` for Anthropic Messages requests, tool calls, and streaming translation to llama.cpp.
- Add a `claude-local` launcher that reads the CC Switch local Qwen configuration while ignoring CC Switch's active-provider toggle.
- Preserve the existing Codex proxy behavior and the `8850` ComfyUI video port.

## Non-goals

- No changes to `codex_proxy.py` protocol behavior.
- No migration of the ComfyUI image/video ports.
- No requirement for Claude Code to use the cloud CC Switch provider.
