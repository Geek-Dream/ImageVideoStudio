#!/usr/bin/env python3
"""Anthropic Messages -> llama.cpp chat-completions compatibility proxy."""
import argparse
import json
import os
import socket
import threading
import time
import uuid
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(BASE, "codex_proxy_log")
MAX_PROCESS_LOGS = 6
_PROCESS_LOG = None
_PROCESS_LOG_LOCK = threading.Lock()


def _open_process_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    files = [os.path.join(log_dir, n) for n in os.listdir(log_dir)
             if n.startswith("claude_proxy_") and n.endswith(".log")]
    files.sort(key=lambda p: os.path.getmtime(p))
    while len(files) >= MAX_PROCESS_LOGS:
        try:
            os.remove(files.pop(0))
        except OSError:
            pass
    path = os.path.join(log_dir, f"claude_proxy_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}_{os.getpid()}.log")
    return path, open(path, "a", encoding="utf-8", buffering=1)


def _log(message):
    if _PROCESS_LOG is None:
        return
    try:
        with _PROCESS_LOG_LOCK:
            _PROCESS_LOG.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            _PROCESS_LOG.flush()
    except Exception:
        pass


def _block_text(block):
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    if block.get("type") == "text":
        return str(block.get("text", ""))
    content = block.get("content")
    if isinstance(content, list):
        return "".join(_block_text(item) for item in content)
    return str(content or "")


def _content_for_openai(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _block_text(content)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        typ = block.get("type")
        if typ == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif typ == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                media = source.get("media_type", "application/octet-stream")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}})
            elif source.get("type") == "url" and source.get("url"):
                parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
    if not parts:
        return ""
    return parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts


def anthropic_to_openai(request):
    """Convert an Anthropic Messages payload to llama.cpp chat-completions."""
    out = {"model": request.get("model", "local-qwen"), "messages": []}
    system = request.get("system")
    system_text = _block_text(system) if isinstance(system, (str, dict)) else "".join(_block_text(b) for b in (system or []))
    if system_text:
        out["messages"].append({"role": "system", "content": system_text})
    for message in request.get("messages") or []:
        role = message.get("role", "user")
        content = message.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "assistant":
            text_parts = [_block_text(b) for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            calls = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({"id": block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}", "type": "function",
                                  "function": {"name": block.get("name", ""),
                                               "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False, separators=(",", ":"))}})
            item = {"role": "assistant", "content": "\n".join(x for x in text_parts if x) or None}
            if calls:
                item["tool_calls"] = calls
            out["messages"].append(item)
        else:
            normal = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out["messages"].append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""),
                                            "content": _block_text(block.get("content", ""))})
                else:
                    normal.append(block)
            if normal:
                out["messages"].append({"role": "user", "content": _content_for_openai(normal)})
    tools = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append({"type": "function", "function": {"name": tool.get("name", ""),
                     "description": tool.get("description", ""), "parameters": tool.get("input_schema") or {"type": "object"}}})
    if tools:
        out["tools"] = tools
    for key in ("temperature", "top_p"):
        if key in request:
            out[key] = request[key]
    if "stop_sequences" in request:
        out["stop"] = request["stop_sequences"]
    if "max_tokens" in request:
        out["max_tokens"] = request["max_tokens"]
    out["stream"] = bool(request.get("stream", False))
    return out


def _stop_reason(finish_reason):
    return {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}.get(finish_reason, "end_turn")


def openai_to_anthropic(response, model=None):
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (TypeError, ValueError):
            args = {}
        blocks.append({"type": "tool_use", "id": call.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                       "name": fn.get("name", ""), "input": args})
    usage = response.get("usage") or {}
    return {"id": response.get("id", f"msg_{uuid.uuid4().hex}"), "type": "message", "role": "assistant",
            "model": model or response.get("model", "local-qwen"), "content": blocks or [{"type": "text", "text": ""}],
            "stop_reason": _stop_reason(choice.get("finish_reason")), "stop_sequence": None,
            "usage": {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                      "output_tokens": int(usage.get("completion_tokens", 0) or 0)}}


def openai_error_to_anthropic(response, status):
    error = response.get("error") if isinstance(response, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return {"error": {"type": "api_error" if int(status or 500) >= 500 else "invalid_request_error",
                       "message": str(message or "上游 llama.cpp 请求失败")}}


class _StreamEncoder:
    def __init__(self, model, message_id):
        self.model, self.message_id = model, message_id
        self.started = False
        self.blocks = {}
        self.next_index = 0
        self.finished = False

    def _start(self):
        if self.started:
            return []
        self.started = True
        return [{"type": "message_start", "message": {"id": self.message_id, "type": "message", "role": "assistant",
                "model": self.model, "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}}]

    def feed(self, chunk):
        if self.finished:
            return []
        events = self._start()
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            if "text" not in self.blocks:
                self.blocks["text"] = self.next_index; self.next_index += 1
                events.append({"type": "content_block_start", "index": self.blocks["text"], "content_block": {"type": "text", "text": ""}})
            events.append({"type": "content_block_delta", "index": self.blocks["text"], "delta": {"type": "text_delta", "text": delta["content"]}})
        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            key = f"tool:{index}"
            fn = call.get("function") or {}
            if key not in self.blocks:
                self.blocks[key] = self.next_index; self.next_index += 1
                events.append({"type": "content_block_start", "index": self.blocks[key], "content_block": {
                    "type": "tool_use", "id": call.get("id", f"toolu_{uuid.uuid4().hex[:12]}"), "name": fn.get("name", ""), "input": {}}})
            if fn.get("arguments"):
                events.append({"type": "content_block_delta", "index": self.blocks[key], "delta": {
                    "type": "input_json_delta", "partial_json": fn["arguments"]}})
        if choice.get("finish_reason"):
            reason = _stop_reason(choice["finish_reason"])
            for index in self.blocks.values():
                events.append({"type": "content_block_stop", "index": index})
            usage = chunk.get("usage") or {}
            events.append({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                           "usage": {"output_tokens": int(usage.get("completion_tokens", 0) or 0)}})
            events.append({"type": "message_stop"})
            self.finished = True
        return events

    def finish(self):
        if self.finished:
            return []
        self.finished = True
        events = self._start()
        for index in self.blocks.values():
            events.append({"type": "content_block_stop", "index": index})
        events.extend([{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 0}},
                       {"type": "message_stop"}])
        return events


def openai_stream_to_anthropic(chunks, model, message_id=None):
    encoder = _StreamEncoder(model, message_id or f"msg_{uuid.uuid4().hex}")
    for chunk in chunks:
        for event in encoder.feed(chunk):
            yield event
    for event in encoder.finish():
        yield event


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "claude-local-proxy/1.0"

    def log_message(self, fmt, *args):
        _log("http " + (fmt % args))

    def _json(self, status, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "proxy": "claude", "target": f"{self.server.up_host}:{self.server.up_port}"})
        else:
            self._json(404, {"error": {"type": "not_found_error", "message": "Not found"}})

    def do_POST(self):
        if self.path.split("?", 1)[0].rstrip("/") not in ("/v1/messages", "/messages"):
            self._json(404, {"error": {"type": "not_found_error", "message": "Use /v1/messages"}}); return
        try:
            length = int(self.headers.get("Content-Length", "0")); request = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps(anthropic_to_openai(request), ensure_ascii=False).encode()
            self._forward(body, request)
        except Exception as exc:
            _log(f"request_error {type(exc).__name__}: {exc}")
            self._json(400, {"error": {"type": "invalid_request_error", "message": str(exc)}})

    def _forward(self, body, request):
        conn = HTTPConnection(self.server.up_host, self.server.up_port, timeout=600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        headers["Content-Type"] = "application/json"; headers["Content-Length"] = str(len(body)); headers["Connection"] = "close"
        _log(f"request claude -> llama path=/v1/messages model={request.get('model','')} stream={bool(request.get('stream'))} bytes={len(body)}")
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            resp = conn.getresponse(); ctype = (resp.getheader("Content-Type") or "").lower()
            if request.get("stream") and "event-stream" in ctype:
                self.send_response(resp.status); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.end_headers()
                encoder = _StreamEncoder(request.get("model", "local-qwen"), f"msg_{uuid.uuid4().hex}")
                while True:
                    line = resp.readline()
                    if not line: break
                    if not line.startswith(b"data:"): continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]": break
                    try: chunk = json.loads(raw.decode())
                    except (ValueError, UnicodeDecodeError): continue
                    for event in encoder.feed(chunk): self.wfile.write(_sse(event["type"], event)); self.wfile.flush()
                for event in encoder.finish(): self.wfile.write(_sse(event["type"], event)); self.wfile.flush()
                return
            raw = resp.read()
            if "json" in ctype:
                try:
                    parsed = json.loads(raw.decode())
                    converted = openai_error_to_anthropic(parsed, resp.status) if resp.status >= 400 else openai_to_anthropic(parsed, request.get("model"))
                    raw = json.dumps(converted, ensure_ascii=False).encode()
                except (ValueError, UnicodeDecodeError): pass
            self.send_response(resp.status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Claude Code Anthropic Messages compatibility proxy")
    parser.add_argument("--listen", type=int, default=8848)
    parser.add_argument("--target", default="127.0.0.1:8846")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    global _PROCESS_LOG
    path, _PROCESS_LOG = _open_process_log(args.log_dir)
    host, port = args.target.rsplit(":", 1)
    srv = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    srv.up_host, srv.up_port = host, int(port)
    _log(f"process_start pid={os.getpid()} listen=127.0.0.1:{args.listen} target={args.target} log={path}")
    print(f"Claude proxy: 127.0.0.1:{args.listen} -> {args.target}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if _PROCESS_LOG: _PROCESS_LOG.close()


if __name__ == "__main__":
    main()
