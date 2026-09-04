#!/usr/bin/env python3
# ============================================================
# codex_proxy.py · Codex 工具兼容翻译代理
# (纯标准库,跨平台 Win/macOS/Linux)
#
# 这个代理兼容 Codex 的 OpenAI 兼容请求：
#   - 支持 /v1/responses 与 /v1/chat/completions；
#   - 把 namespace 等包装工具摊平成 llama.cpp 能识别的 function 工具；
#   - 把 system/developer/instructions 合并到 llama.cpp 模板要求的位置；
#   - 把模型泄漏的受支持 <tool_code>/<tool_call> 文本安全转换为 Responses function_call；
#   - 其余响应和 SSE 流原样转发。
#
# 代理占公共端口(默认 8848)，llama-server 退到内部端口(默认 8846)，所以
# Codex、网页和已有客户端都不用改地址。
#
# 使用: python3 codex_proxy.py [--listen 8848] [--target 127.0.0.1:8846]
# ============================================================
import argparse, ast, html, json, os, re, socket, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
from urllib.parse import urlsplit

DROP_TOOL_TYPES = ("web_search", "file_search")   # 本地用不了的联网类内置工具, 直接丢弃省 token
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex_proxy_log")
MAX_PROCESS_LOGS = 6
_PROCESS_LOG = None
_PROCESS_LOG_LOCK = threading.Lock()

# Codex normally sends this catalogue itself.  A stale/partially restored
# session can omit ``tools`` while still requesting ``tool_choice=auto``;
# without a catalogue the local model has no structured way to call anything.
# Keep this fallback limited to tools executed by the Codex client, rather than
# pretending that MCP or network tools are available upstream.
FALLBACK_TOOL_CATALOG = [
    {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a command in a PTY, returning output or a session ID for an ongoing interaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
                "login": {"type": "boolean", "description": "True runs with login shell semantics; defaults to true."},
                "max_output_tokens": {"type": "number", "description": "Output token budget."},
                "shell": {"type": "string", "description": "Shell binary to launch."},
                "tty": {"type": "boolean", "description": "Allocate a PTY."},
                "workdir": {"type": "string", "description": "Working directory."},
                "yield_time_ms": {"type": "number", "description": "Wait before yielding output."},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_stdin",
        "description": "Writes characters to an existing unified exec session and returns recent output.",
        "parameters": {
            "type": "object",
            "properties": {
                "chars": {"type": "string", "description": "Bytes to write to stdin; defaults to empty."},
                "max_output_tokens": {"type": "number", "description": "Output token budget."},
                "session_id": {"type": "number", "description": "Running exec session identifier."},
                "yield_time_ms": {"type": "number", "description": "Wait before yielding output."},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apply_patch",
        "description": "Apply a patch to files in the workspace.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "view_image",
        "description": "View a local image file from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Local image path."}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_mcp_resources",
        "description": "Lists resources provided by MCP servers.",
        "parameters": {
            "type": "object",
            "properties": {
                "cursor": {"type": "string"},
                "server": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_mcp_resource_templates",
        "description": "Lists parameterized resource templates provided by MCP servers.",
        "parameters": {
            "type": "object",
            "properties": {
                "cursor": {"type": "string"},
                "server": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_mcp_resource",
        "description": "Read a specific MCP resource by server and URI.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "uri": {"type": "string"},
            },
            "required": ["server", "uri"],
            "additionalProperties": False,
        },
    },
]


def _open_process_log(log_dir):
    """Create one process log and retain only the six newest proxy logs."""
    os.makedirs(log_dir, exist_ok=True)
    prefix = "codex_proxy_"
    suffix = ".log"
    existing = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    existing.sort(key=lambda path: os.path.getmtime(path))
    while len(existing) >= MAX_PROCESS_LOGS:
        oldest = existing.pop(0)
        try:
            os.remove(oldest)
        except OSError:
            pass
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{prefix}{stamp}_{time.time_ns()}_{os.getpid()}{suffix}")
    handle = open(path, "a", encoding="utf-8", buffering=1)
    return path, handle


def _process_log(message):
    """Write a timestamped process event without interrupting proxy traffic."""
    global _PROCESS_LOG
    if _PROCESS_LOG is None:
        return
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{time.time_ns() % 1_000_000_000:09d}] {message}\n"
    try:
        with _PROCESS_LOG_LOCK:
            _PROCESS_LOG.write(line)
            _PROCESS_LOG.flush()
    except (OSError, ValueError):
        pass


def _tool_spec_map(tools):
    """Return the tools declared by this request, including nested namespaces."""
    result = {}
    if not isinstance(tools, list):
        return result
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        nested = tool.get("tools")
        if isinstance(nested, list):
            result.update(_tool_spec_map(nested))
            continue
        core = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = core.get("name")
        if isinstance(name, str) and name:
            result[name] = core
    return result


def _matches_schema(value, schema):
    if not isinstance(schema, dict):
        return True
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    if isinstance(expected, str) and expected in type_matches and not type_matches[expected]:
        return False
    if isinstance(expected, list) and not any(_matches_schema(value, dict(schema, type=item)) for item in expected):
        return False
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list) and any(key not in value for key in required):
            return False
        properties = schema.get("properties")
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False and any(key not in properties for key in value):
                return False
            if any(key in properties and not _matches_schema(child, properties[key]) for key, child in value.items()):
                return False
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_matches_schema(item, schema["items"]) for item in value)
    return True


def _schema_accepts_arguments(spec, arguments):
    """Validate model-generated arguments against the request's tool schema."""
    if not isinstance(arguments, dict):
        return False
    schema = spec.get("parameters") or spec.get("input_schema") or spec.get("schema") or {}
    return _matches_schema(arguments, schema)


def parse_text_tool_call(text, tools):
    """Parse the model's textual tool-call fallbacks safely.

    The result is deliberately restricted to tools present in the request and
    to JSON-like arguments. Unrecognised markup remains ordinary assistant text.
    """
    if not isinstance(text, str):
        return None
    if "<tool_call" in text.lower():
        return _parse_xml_tool_call(text, tools)
    if "<tool_code" not in text.lower():
        return None
    # Some llama.cpp stop configurations consume the closing tag. Accept EOF
    # only when the entire remaining payload is still a complete literal.
    match = re.search(r"<tool_code\s*>(.*?)(?:</tool_code\s*>|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    value = None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(raw)
            break
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments", value.get("input"))
    specs = _tool_spec_map(tools)
    spec = specs.get(name) if isinstance(name, str) else None
    if spec is None or not _schema_accepts_arguments(spec, arguments):
        return None
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return {"name": name, "arguments": encoded}


def _parse_xml_tool_call(text, tools):
    """Parse ``<tool_call><function=NAME>`` markup with parameter tags."""
    match = re.search(r"<tool_call\s*>(.*?)</tool_call\s*>", text,
                      flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    function = re.search(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function\s*>",
                         match.group(1), flags=re.IGNORECASE | re.DOTALL)
    if not function:
        return None
    name = function.group(1).strip()
    spec = _tool_spec_map(tools).get(name)
    if spec is None:
        return None
    schema = spec.get("parameters") or spec.get("input_schema") or spec.get("schema") or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    arguments = {}
    for parameter in re.finditer(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter\s*>",
                                 function.group(2), flags=re.IGNORECASE | re.DOTALL):
        key = parameter.group(1).strip()
        value = html.unescape(parameter.group(2)).strip()
        parameter_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
        expected = parameter_schema.get("type") if isinstance(parameter_schema, dict) else None
        if expected in ("number", "integer", "boolean", "null", "array", "object"):
            try:
                value = json.loads(value)
            except (ValueError, TypeError, json.JSONDecodeError):
                return None
        arguments[key] = value
    if not _schema_accepts_arguments(spec, arguments):
        return None
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return {"name": name, "arguments": encoded}


def _response_function_call(response, call):
    call_id = "call_" + uuid.uuid4().hex
    item = {
        "id": "fc_" + uuid.uuid4().hex,
        "type": "function_call",
        "status": "completed",
        "arguments": call["arguments"],
        "call_id": call_id,
        "name": call["name"],
    }
    response = dict(response)
    response["output"] = [item]
    response["status"] = "completed"
    return response, item


def convert_text_tool_response(response, tools):
    """Convert a non-streaming llama.cpp text response when it contains a call."""
    if not isinstance(response, dict):
        return response
    output = response.get("output")
    if not isinstance(output, list):
        return response
    # Rebuilding mixed reasoning/message event streams would risk dropping
    # opaque reasoning state. Restrict fallback conversion to one text item.
    if len(output) != 1:
        return response
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                call = parse_text_tool_call(part.get("text", ""), tools)
                if call:
                    return _response_function_call(response, call)[0]
    return response


def _sse_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def convert_stream_tool_response(body, tools):
    """Convert a buffered Responses SSE body containing textual tool markup."""
    if not isinstance(body, bytes) or not any(marker in body.lower() for marker in (b"tool_code", b"tool_call")):
        return body
    completed = None
    text = ""
    for raw in body.splitlines():
        if not raw.startswith(b"data: "):
            continue
        try:
            data = json.loads(raw[6:].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if data.get("type") == "response.output_text.delta":
            text += data.get("delta", "")
        elif data.get("type") == "response.output_text.done":
            text = data.get("text", text)
        elif data.get("type") == "response.completed":
            completed = data.get("response")
    call = parse_text_tool_call(text, tools)
    if not call or not isinstance(completed, dict):
        return body
    converted, item = _response_function_call(completed, call)
    response_id = converted.get("id", "resp_" + uuid.uuid4().hex)
    added = {"type": "response.output_item.added", "output_index": 0, "item": dict(item, status="in_progress", arguments="")}
    done = {"type": "response.output_item.done", "output_index": 0, "item": item}
    delta = {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": item["id"], "delta": item["arguments"]}
    completed_event = {"type": "response.completed", "response": converted}
    return b"".join([
        _sse_event("response.created", {"type": "response.created", "response": {"id": response_id, "object": "response", "status": "in_progress"}}),
        _sse_event("response.in_progress", {"type": "response.in_progress", "response": {"id": response_id, "object": "response", "status": "in_progress"}}),
        _sse_event("response.output_item.added", added),
        _sse_event("response.function_call_arguments.delta", delta),
        _sse_event("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "output_index": 0, "item_id": item["id"], "arguments": item["arguments"]}),
        _sse_event("response.output_item.done", done),
        _sse_event("response.completed", completed_event),
    ])


def _possible_tool_markup(text):
    """Return true while a streamed delta may be starting textual tool markup."""
    lower = text.lower()
    if re.search(r"<(?:tool_code|tool_call)\b", lower):
        return True
    # Tokenizers can split the opening tag across deltas (e.g. ``<tool`` +
    # ``_call>``). Keep any suffix that is a prefix of a supported marker.
    if "<" not in lower:
        return False
    fragment = "<" + lower.rsplit("<", 1)[-1]
    return any(marker.startswith(fragment) for marker in ("<tool_code", "<tool_call"))


def proxy_label():
    """两个入口共用实现，但日志保留实际启动入口名称。"""
    return os.environ.get("LLM_PROXY_NAME", "codex_proxy")


def _content_to_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            inner = item.get("content")
            if isinstance(inner, str) and inner.strip():
                parts.append(inner.strip())
        return "\n\n".join(parts).strip()
    return ""


def normalize_messages(messages, instructions=None):
    """llama.cpp 的部分模板要求 system 只能在最开头, developer 也要并到最前面的 system。"""
    if not isinstance(messages, list):
        return messages, instructions, False
    preamble = []
    rest = []
    changed = False
    if isinstance(instructions, str) and instructions.strip():
        preamble.append(instructions.strip())
        instructions = None
        changed = True
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            rest.append(msg)
            continue
        role = msg.get("role")
        # Codex function_call_output may contain structured image/content
        # blocks (notably from view_image). llama.cpp Responses accepts tool
        # outputs as input text only, so serialize non-text results before
        # forwarding. Keep ordinary string outputs byte-for-byte unchanged.
        if msg.get("type") == "function_call_output" and not isinstance(msg.get("output"), str):
            copy = dict(msg)
            output = msg.get("output")
            if isinstance(output, list):
                text_parts = []
                for part in output:
                    if isinstance(part, dict):
                        if isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                        elif part.get("type") in ("image", "input_image", "image_url"):
                            text_parts.append("[图片工具结果：已返回图片内容]")
                        else:
                            text_parts.append(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
                    else:
                        text_parts.append(str(part))
                copy["output"] = "\n".join(text_parts)
            else:
                copy["output"] = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
            rest.append(copy)
            changed = True
            continue
        if role in ("system", "developer"):
            txt = _content_to_text(msg.get("content"))
            if txt:
                preamble.append(txt)
            if role == "developer" or idx != 0 or len(preamble) > 1:
                changed = True
            continue
        rest.append(msg)
    if not preamble:
        return messages, instructions, changed
    merged = {"role": "system", "content": "\n\n".join(preamble)}
    new_messages = [merged] + rest
    if new_messages != messages:
        changed = True
    return new_messages, instructions, changed


def _contains_image(messages):
    """Detect OpenAI multimodal image parts without inspecting binary data."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    return True
    return False


def _is_title_request(request):
    """Recognize Codex's internal title-generation call.

    Title generation deliberately has no tools and asks for a strict JSON
    schema. Injecting the fallback catalogue there makes the model spend the
    turn on a tool call instead of returning the requested title.
    """
    if not isinstance(request, dict):
        return False
    text = request.get("text")
    if not isinstance(text, dict):
        return False
    fmt = text.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return False
    prompt = ""
    value = request.get("input")
    if isinstance(value, list):
        prompt = "\n".join(_content_to_text(item.get("content")) for item in value if isinstance(item, dict))
    return bool(re.search(r"(?:task\s+title|任务标题|标题).{0,80}(?:36|five|单行)", prompt, re.IGNORECASE | re.DOTALL)
                or re.search(r"Generate a concise", str(request.get("text")), re.IGNORECASE))


def _should_inject_fallback_tools(request, msg_key):
    """Return whether a missing tool catalogue is an accidental omission."""
    if not isinstance(request, dict) or not msg_key:
        return False
    # Require an explicit auto choice. Ordinary OpenAI-compatible callers
    # often omit both fields intentionally and must remain tool-free.
    if request.get("tool_choice") != "auto":
        return False
    if _is_title_request(request) or _contains_image(request.get(msg_key)):
        return False
    messages = request.get(msg_key)
    if not isinstance(messages, list) or not any(
        isinstance(item, dict) and item.get("role") == "user" for item in messages
    ):
        return False
    # Treat both an absent field and an explicit empty list as the same stale
    # session failure. A non-empty catalogue is always authoritative.
    return not request.get("tools")


def _coerce_function_tool(t, wrap_function):
    core = t.get("function") if isinstance(t.get("function"), dict) else None
    if not core:
        core = {k: v for k, v in t.items() if k not in ("type", "strict", "tools", "namespace")}
    if isinstance(core.get("function"), dict) and "name" not in core:
        core = dict(core["function"])
    out = {"type": "function"}
    spec = {k: v for k, v in core.items() if k in ("name", "description", "parameters")}
    # Codex-compatible clients use both OpenAI's `parameters` spelling and
    # Responses/MCP-style `input_schema` / `schema`. Dropping the latter made
    # every tool look parameterless to Qwen, so it copied the tool catalogue
    # instead of emitting a valid call.
    if "parameters" not in spec:
        for schema_key in ("input_schema", "inputSchema", "schema"):
            schema = core.get(schema_key)
            if isinstance(schema, dict):
                spec["parameters"] = schema
                break
    if "parameters" not in spec:
        spec["parameters"] = {"type": "object", "properties": {}}
    if wrap_function:
        out["function"] = spec
    else:
        out.update(spec)
    return out


TOOL_PROTOCOL = """Tool execution protocol:
- The <tools> catalogue is input only. Never repeat, summarize, translate, or explain it.
- When a tool is needed, call it immediately using exactly one <tool_call><function=NAME>...</function></tool_call> block from the provided tool list.
- Include every required parameter with a concrete value. Do not emit a tool call with blank parameters.
- Do not write a plan or a natural-language answer instead of an available required tool call.
- Never emit <tool_code>, Markdown code fences, Python-style function syntax, or any other textual tool-call format. They are invalid and will not be executed.
- Network commands are bounded operations: use curl/wget with --connect-timeout 5 and --max-time 15 (never an unbounded request).
- Keep command output small. For HTML or JSON, extract the relevant fields and cap output with head -c 20000; never return a whole web page to context.
- On macOS use grep -E or rg; grep -P is unavailable. Do not retry a hanging request with another search engine automatically.
- After two search sources fail, time out, or produce no useful result, stop searching and report the limitation. Summarize evidence and finish the user's task.
- Poll an existing process only when its session is still running and polling can advance it. Do not launch a new equivalent command while one is active.
"""

LOOP_RECOVERY_PROTOCOL = """Tool loop recovery is active for this turn.
You have retried the exact same exec_command while its earlier process was
still running. This is an autonomous no-progress loop, not a new user request.
The user's original task is still active and must be completed.

For this response, first call write_stdin for the existing session ID below
with empty chars and a reasonable yield_time_ms. Do not start the same command
again. If the poll completes, use its result and continue the original task. If
it is still running, report the active session status only when further polling
cannot advance the task; do not relaunch it. If it failed, diagnose the failure
and use a materially different valid approach on the next step.

Do not end the task merely because loop recovery is active. Do not describe
this internal recovery policy to the user unless the pending command prevents a
useful answer.
"""

FAILED_TOOL_RECOVERY_PROTOCOL = """Tool loop recovery is active for this turn.
You have repeated the exact same tool call after it already failed or was
rejected. This is an autonomous no-progress loop, not a new user request.
The user's original task is still active and must be completed.

Do not retry that exact command or argument set. Read the previous tool result,
then use a materially different valid approach, or complete the parts that are
already supported by completed evidence. Keep using tools when they can advance
the task. Do not end the task merely because this recovery notice is active.
"""


def _canonical_call_signature(item):
    """Return a stable signature for a Responses API function_call item."""
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = item.get("arguments", "")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    try:
        encoded = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(args)
    return name, encoded


def _last_user_text(messages):
    for item in reversed(messages or []):
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
            return _content_to_text(item.get("content"))
    return ""


def _explicit_repeat_limit(text):
    """Read a deliberately requested repeat count from the latest user text.

    This is not a general natural-language interpreter. It only recognizes a
    number next to an explicit call/run/retry/repeat instruction, which keeps a
    user asking for (for example) ten identical API calls in control.
    """
    if not isinstance(text, str):
        return None
    number = r"([0-9]{1,3}|[一二三四五六七八九十两]+)"
    action = r"(?:调用|执行|运行|请求|重试|重复|call|run|retry|repeat)"
    patterns = (
        rf"{action}.{{0,20}}?{number}\s*(?:次|遍|回|times?|x\b)",
        rf"{number}\s*(?:次|遍|回|times?|x\b).{{0,20}}?{action}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        if raw.isdigit():
            value = int(raw)
        else:
            values = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            value = values.get(raw)
        if value and 1 <= value <= 100:
            return value
    return None


def repeated_running_tool_call(messages):
    """Detect an agent repeatedly starting the same still-running command.

    Codex sends the complete current turn back on every Responses request.  A
    model that ignores a ``Process running with session ID`` result can otherwise
    keep launching the same command indefinitely.  Scope the check to the most
    recent user message, so separate user turns and legitimate repeated work
    are not affected.
    """
    if not isinstance(messages, list):
        return False, 0, None, None, None
    last_user = -1
    for index, item in enumerate(messages):
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
            last_user = index
    turn = messages[last_user + 1:]
    calls = [item for item in turn if _canonical_call_signature(item)]
    if len(calls) < 3:
        return False, len(calls), _explicit_repeat_limit(_last_user_text(messages)), None, None
    signature = _canonical_call_signature(calls[-1])
    repeated = [item for item in calls if _canonical_call_signature(item) == signature]
    if len(repeated) < 3:
        return False, len(repeated), _explicit_repeat_limit(_last_user_text(messages)), None, None
    call_ids = {item.get("call_id") for item in repeated if item.get("call_id")}
    running = 0
    latest_output = ""
    latest_call_id = calls[-1].get("call_id")
    for item in turn:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        if item.get("call_id") not in call_ids:
            continue
        output = str(item.get("output") or "").lower()
        if item.get("call_id") == latest_call_id:
            latest_output = output
        if "process running with session id" in output or "process running with session" in output:
            running += 1
    # A user may explicitly ask for N identical calls.  The breaker fires only
    # when that requested budget has been consumed; otherwise, use a small
    # no-progress ceiling to stop autonomous retries.
    requested_limit = _explicit_repeat_limit(_last_user_text(messages))
    limit = requested_limit if requested_limit is not None else 6
    # Recovery is needed only when the *latest* model action is another copy
    # of the same still-running command. Once it polls through write_stdin,
    # this condition clears and the task continues normally.
    session = None
    match = re.search(r"session\s+id\s+(\d+)", latest_output, flags=re.IGNORECASE)
    if match:
        session = match.group(1)
    failed = any(marker in latest_output for marker in (
        "rejected:", "timed out", "exit code: 1", "exit code: 2", "command not found",
    ))
    state = "running" if session is not None else ("failed" if failed else None)
    looped = len(repeated) >= limit and ((running >= 2 and state == "running") or state == "failed")
    return looped, len(repeated), requested_limit, session, state


def _add_loop_recovery(messages, session_id, repeats, requested_limit, state):
    if not isinstance(messages, list):
        return messages
    out = list(messages)
    system = next((m for m in out if isinstance(m, dict) and m.get("role") == "system"), None)
    budget = (f"The user explicitly allowed {requested_limit} repeats; that budget is exhausted."
              if requested_limit is not None else
              f"No repeat count was requested by the user; {repeats} no-progress retries were observed.")
    if state == "running":
        note = LOOP_RECOVERY_PROTOCOL + f"\nExisting session ID: {session_id}\n{budget}\n"
    else:
        note = FAILED_TOOL_RECOVERY_PROTOCOL + f"\n{budget}\n"
    if system:
        old = system.get("content")
        if isinstance(old, str) and note.strip() not in old:
            system["content"] = (old + "\n\n" + note).strip()
        return out
    return [{"role": "system", "content": note}] + out


def _tool_name(tool):
    if not isinstance(tool, dict):
        return ""
    if isinstance(tool.get("function"), dict):
        return str(tool["function"].get("name") or "")
    return str(tool.get("name") or "")


def _add_tool_protocol(messages):
    """Append a compact, model-facing tool policy after Codex's long prompt."""
    if not isinstance(messages, list):
        return messages
    out = list(messages)
    system = next((m for m in out if isinstance(m, dict) and m.get("role") == "system"), None)
    if system:
        old = system.get("content")
        if isinstance(old, str) and TOOL_PROTOCOL not in old:
            system["content"] = (old + "\n\n" + TOOL_PROTOCOL).strip()
        return out
    return [{"role": "system", "content": TOOL_PROTOCOL}] + out


def flatten_tools(tools, wrap_function):
    """把 namespace 等包装的工具摊平成 llama.cpp 能接受的 function 结构。
    wrap_function=True -> chat/completions 需要 {type:function,function:{...}}
    wrap_function=False -> responses 需要 {type:function,name,...}"""
    if not isinstance(tools, list):
        return tools, False
    out, changed = [], False
    seen = set()
    for t in tools:
        if not isinstance(t, dict):
            out.append(t)
            continue
        typ = t.get("type")
        if typ in DROP_TOOL_TYPES:
            changed = True
            continue
        subs = t.get("tools")
        # Harmony 的 namespace 外层有时也带 name，必须先看 tools，不能误判成函数。
        if isinstance(subs, list):
            flat, _ = flatten_tools(subs, wrap_function)
            out.extend(flat)
            changed = True
            continue
        if typ == "function" or "name" in t or "parameters" in t or isinstance(t.get("function"), dict):
            tool = _coerce_function_tool(t, wrap_function)
            name = tool.get("name") or (tool.get("function") or {}).get("name", "")
            if name and name in seen:
                changed = True
                continue
            if name:
                seen.add(name)
            if tool != t:
                changed = True
            out.append(tool)
            continue
        out.append(t)
    return out, changed


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CodexProxy/1.0"

    def log_message(self, fmt, *args):     # 默认访问日志关掉, 走自己的 LOG
        pass

    def _log(self, msg):
        t = time.strftime("%H:%M:%S")
        line = f"{t} {proxy_label()}: {msg}"
        print(line, flush=True)
        _process_log(line)

    def _begin_trace(self):
        self._trace_id = "req_" + uuid.uuid4().hex[:12]
        self._trace_step = 0
        self._trace_started = time.monotonic()

    def _trace_log(self, message):
        self._trace_step = getattr(self, "_trace_step", 0) + 1
        self._log(f"[{getattr(self, '_trace_id', 'req_unknown')} step={self._trace_step}] {message}")

    def _trace_done(self, status=None, detail=""):
        started = getattr(self, "_trace_started", None)
        elapsed = (time.monotonic() - started) * 1000 if started is not None else 0
        code = f"status={status}" if status is not None else "status=unknown"
        suffix = f" {detail}" if detail else ""
        self._trace_log(f"返回客户端 proxy -> client {code} elapsed_ms={elapsed:.1f}{suffix}")

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        self._begin_trace()
        self._trace_log(f"请求进入 client -> proxy method=GET path={self.path}")
        self._request_tools = []
        self._forward(b"")

    def do_POST(self):
        self._begin_trace()
        body = self._read_body()
        try:
            original_request = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            original_request = {}
        if not isinstance(original_request, dict):
            original_request = {}
        self._request_tools = original_request.get("tools", []) if isinstance(original_request.get("tools"), list) else []
        tool_names = [_tool_name(tool) for tool in self._request_tools if _tool_name(tool)]
        msg_count = len(original_request.get("messages", [])) if isinstance(original_request.get("messages"), list) else (
            len(original_request.get("input", [])) if isinstance(original_request.get("input"), list) else 0)
        self._trace_log(
            f"请求进入 client -> proxy method=POST path={self.path} bytes={len(body)} "
            f"model={original_request.get('model', '<missing>')} stream={bool(original_request.get('stream', False))} "
            f"tool_choice={original_request.get('tool_choice', '<default>')} messages={msg_count} "
            f"tools={len(tool_names)} names={','.join(tool_names[:20]) or '<none>'}"
        )
        # 只对 LLM 相关路径做工具摊平
        if self.path.rstrip("/") in ("/v1/responses", "/v1/chat/completions"):
            body = self._maybe_rewrite(body)
        try:
            request = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            request = {}
        # Parse fallback tool text only against the effective catalogue that
        # was actually sent upstream. Image requests and dropped tool types
        # must remain tool-free after rewriting.
        self._request_tools = request.get("tools", []) if isinstance(request, dict) and isinstance(request.get("tools"), list) else []
        self._trace_log(f"请求准备转发 path={self.path} rewritten_bytes={len(body)}")
        self._forward(body)

    def _maybe_rewrite(self, body):
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        if not isinstance(req, dict):
            return body
        original_tools = req.get("tools") if isinstance(req.get("tools"), list) else []
        changed = False
        wrap_function = self.path.rstrip("/") != "/v1/responses"
        msg_key = "messages" if isinstance(req.get("messages"), list) else ("input" if isinstance(req.get("input"), list) else None)
        if msg_key:
            msgs, instr, msg_changed = normalize_messages(req.get(msg_key), req.get("instructions"))
            req[msg_key] = msgs
            if instr is None:
                req.pop("instructions", None)
            else:
                req["instructions"] = instr
            changed = changed or msg_changed
        has_image = _contains_image(req.get(msg_key) if msg_key else None)
        if has_image:
            # Vision requests may still need local tools afterwards (for
            # example, inspect a screenshot and then edit the source). Keep
            # the catalogue available instead of forcing a tool-free reply.
            if _should_inject_fallback_tools(req, msg_key) or req.get("tool_choice") == "auto":
                fallback = json.loads(json.dumps(FALLBACK_TOOL_CATALOG))
                fallback, _ = flatten_tools(fallback, wrap_function)
                req["tools"] = fallback
                req[msg_key] = _add_tool_protocol(req[msg_key])
                changed = True
            if msg_key:
                msgs = req[msg_key]
                system = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "system"), None)
                note = "When an image is provided, analyze it directly. You may still use the provided local tools when needed."
                if system:
                    old = system.get("content") or ""
                    system["content"] = (old + "\n\n" + note).strip()
                else:
                    req[msg_key] = [{"role": "system", "content": note}] + msgs
                changed = True
            self._trace_log("请求改写: 检测到图片输入，保留/注入本地工具并追加视觉提示")
        elif _should_inject_fallback_tools(req, msg_key):
            # Copy the catalogue per request so flattening or future request
            # normalization cannot mutate the process-wide defaults.
            fallback = json.loads(json.dumps(FALLBACK_TOOL_CATALOG))
            fallback, _ = flatten_tools(fallback, wrap_function)
            req["tools"] = fallback
            req[msg_key] = _add_tool_protocol(req[msg_key])
            changed = True
            names = [_tool_name(tool) for tool in fallback if _tool_name(tool)]
            self._log(f"工具目录兜底: 原请求未声明 tools，注入 {len(fallback)} 个本地工具")
            self._trace_log(
                f"请求改写: fallback_tools_injected count={len(fallback)} "
                f"names={','.join(names)}"
            )
        elif "tools" in req:
            looped, repeats, requested_limit, session_id, loop_state = (repeated_running_tool_call(req[msg_key])
                                                                          if msg_key else (False, 0, None, None, None))
            if looped:
                # This is deliberately a narrow recovery path: only an
                # identical call exhausts the user's explicit repeat budget
                # (or reaches the no-progress ceiling) after a single user
                # message, with the executor explicitly saying it is still
                # running. Remove only exec_command for this request, so the
                # model must poll the existing session through write_stdin and
                # can then continue the original task.
                flat, _ = flatten_tools(req.get("tools"), wrap_function)
                has_poller = loop_state == "running" and any(_tool_name(tool) == "write_stdin" for tool in flat)
                if has_poller:
                    req["tools"] = [tool for tool in flat if _tool_name(tool) != "exec_command"]
                # The normal Codex schema includes write_stdin. If a client
                # omits it, retain its tools and make the loop visible rather
                # than silently ending a task that cannot be polled here.
                req[msg_key] = _add_loop_recovery(req[msg_key], session_id, repeats, requested_limit, loop_state)
                changed = True
                budget = f"用户要求 {requested_limit} 次" if requested_limit is not None else "无用户重复指令"
                if has_poller:
                    action = f"改为轮询 session {session_id}"
                elif loop_state == "failed":
                    action = "保留工具并要求模型改用不同路径"
                else:
                    action = "客户端未提供 write_stdin，已提示模型改用其他路径"
                self._log(f"检测到工具无进展重试 {repeats} 次（{budget}），{action}")
                self._trace_log(f"请求改写: 工具无进展恢复 repeats={repeats} action={action}")
            else:
                flat, tool_changed = flatten_tools(req.get("tools"), wrap_function)
                # vMLX/Qwen benefits from an explicit final protocol after the
                # large Codex developer prompt. This is prompt-level rather than
                # tool_choice="required": vMLX can stall indefinitely on the
                # latter when parsing a Responses request.
                if msg_key and flat:
                    req[msg_key] = _add_tool_protocol(req[msg_key])
                    changed = True
                if tool_changed:
                    req["tools"] = flat
                    changed = True
                    names = [_tool_name(tool) for tool in flat if _tool_name(tool)]
                    self._trace_log(
                        f"请求改写: tools 扁平化 {len(original_tools)} -> {len(flat)} "
                        f"names={','.join(names[:20]) or '<none>'}"
                    )
        force = getattr(self.server, "force_settings", None) or {}
        if force:
            thinking = bool(force.get("thinking", False))
            req["enable_thinking"] = thinking
            ctk = req.get("chat_template_kwargs")
            ctk = dict(ctk) if isinstance(ctk, dict) else {}
            ctk["enable_thinking"] = thinking
            req["chat_template_kwargs"] = ctk
            req["temperature"] = float(force.get("temperature", 0.0))
            req["repetition_penalty"] = float(force.get("repetition_penalty", 1.0))
            limit = max(1, int(force.get("max_tokens", 2048)))
            token_key = "max_output_tokens" if self.path.rstrip("/") == "/v1/responses" else "max_tokens"
            try:
                requested = int(req.get(token_key, limit))
            except (TypeError, ValueError):
                requested = limit
            req[token_key] = min(max(1, requested), limit)
            changed = True
            self._log(
                f"强制启动参数 thinking={thinking} temperature={req['temperature']} "
                f"{token_key}<={limit}"
            )
            self._trace_log("请求改写: 应用 force-settings")
        if changed:
            self._log(f"改写请求兼容 llama.cpp ({self.path.rstrip('/')})")
            self._trace_log(
                f"请求改写完成 proxy -> llama.cpp path={self.path.rstrip('/')} "
                f"messages={len(req.get(msg_key, [])) if msg_key and isinstance(req.get(msg_key), list) else 0} "
                f"tools={len(req.get('tools', [])) if isinstance(req.get('tools'), list) else 0}"
            )
            if os.environ.get("CODEX_PROXY_DUMP"):
                self._dump("request", req)
            return json.dumps(req).encode("utf-8")
        return body

    def _dump(self, tag, obj):
        try:
            d = os.environ.get("CODEX_PROXY_DUMP")
            if d:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{tag}_{time.time_ns()}.json"), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    def _forward(self, body):
        """Forward to llama-server and translate fallback Responses tool text."""
        conn = HTTPConnection(self.server.up_host, self.server.up_port, timeout=600)
        try:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "connection",
                                            "accept-encoding", "transfer-encoding")}
            if body:
                headers["Content-Length"] = str(len(body))
            headers["Connection"] = "close"
            self._trace_log(
                f"转发请求 proxy -> llama.cpp target={self.server.up_host}:{self.server.up_port} "
                f"method={self.command} path={self.path or '/'} bytes={len(body)}"
            )
            conn.request(self.command, self.path or "/", body=body, headers=headers)
            resp = conn.getresponse()
            content_type = (resp.getheader("Content-Type") or "").lower()
            response_path = self.path.rstrip("/") == "/v1/responses"
            self._trace_log(
                f"收到上游响应 llama.cpp -> proxy status={resp.status} content_type={content_type or '<none>'} "
                f"responses_api={response_path}"
            )
            if response_path and "event-stream" in content_type:
                self._forward_responses_sse(resp)
                self._trace_done(resp.status, "response_type=SSE")
                return
            if response_path and "json" in content_type:
                raw = resp.read()
                converted = raw
                try:
                    obj = json.loads(raw.decode("utf-8"))
                    rewritten = convert_text_tool_response(obj, getattr(self, "_request_tools", []))
                    if rewritten != obj:
                        converted = json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        item = next((item for item in rewritten.get("output", [])
                                     if isinstance(item, dict) and item.get("type") == "function_call"), None)
                        if item:
                            self._trace_log(
                                f"工具转换 llama.cpp message/output_text -> proxy parser -> Codex function_call "
                                f"name={item.get('name', '<unknown>')} call_id={item.get('call_id', '<none>')} "
                                f"arguments_bytes={len(str(item.get('arguments', '')))}"
                            )
                except (ValueError, UnicodeDecodeError):
                    pass
                if converted != raw:
                    self._log("将模型文本工具调用转换为 Responses function_call")
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(converted)))
                self.end_headers()
                self.wfile.write(converted)
                self.wfile.flush()
                self._trace_done(resp.status, f"response_type=JSON bytes={len(converted)}")
                return
            self.send_response(resp.status)
            up_cl = resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
                    continue
                self.send_header(k, v)
            if up_cl is None:
                # 流式(chunked/SSE): 用 read1() 逐块转发 — 上游一到数据立刻转发,
                # 绝不攒批(之前用 read(65536) 会憋满 64KB 才吐, 客户端看到"一大块一大块蹦").
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read1(16384)
                    if not chunk:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                        break
                    self.wfile.write(("%X\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                    self.wfile.flush()
            else:
                self.send_header("Content-Length", up_cl)
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            self._trace_done(resp.status, f"response_type=passthrough bytes={up_cl or 'chunked'}")
        except Exception as e:
            self._trace_log(f"转发失败 proxy -> llama.cpp error={type(e).__name__}: {e}")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                msg = f"{proxy_label()} error: {e}".encode("utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _upstream_headers(self, resp, content_length=None, chunked=False):
        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
                continue
            self.send_header(key, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        elif content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _write_chunk(self, data):
        if data:
            self.wfile.write(("%X\r\n" % len(data)).encode() + data + b"\r\n")
            self.wfile.flush()

    def _forward_responses_sse(self, resp):
        """Stream progress text immediately; buffer only possible text-tool markup."""
        if not getattr(self, "_request_tools", []):
            self._trace_log("SSE 透传: 请求未声明工具，原样转发")
            self._upstream_headers(resp, chunked=True)
            while True:
                chunk = resp.read1(16384)
                if not chunk:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                self._write_chunk(chunk)
        buffered = bytearray()
        passthrough = False
        arguments_done = False
        headers_sent = False

        def send_headers():
            nonlocal headers_sent
            if not headers_sent:
                self._upstream_headers(resp, chunked=True)
                headers_sent = True

        def send_event(event):
            send_headers()
            self._write_chunk(event)

        while True:
            lines = []
            while True:
                line = resp.readline()
                if not line:
                    break
                lines.append(line)
                if line in (b"\n", b"\r\n"):
                    break
            if not lines:
                break
            event = b"".join(lines)
            data = None
            for line in lines:
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:].decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        pass
            if passthrough:
                if isinstance(data, dict) and data.get("type") == "response.function_call_arguments.done":
                    arguments_done = True
                if (isinstance(data, dict)
                        and data.get("type") == "response.output_item.done"
                        and isinstance(data.get("item"), dict)
                        and data["item"].get("type") == "function_call"
                        and not arguments_done):
                    item = data["item"]
                    done = {
                        "type": "response.function_call_arguments.done",
                        "output_index": data.get("output_index", 0),
                        "item_id": item.get("id"),
                        "arguments": item.get("arguments", ""),
                    }
                    self._write_chunk(_sse_event("response.function_call_arguments.done", done))
                    self._trace_log(
                        f"SSE 原生工具调用: 补齐 response.function_call_arguments.done "
                        f"name={item.get('name', '<unknown>')} item_id={item.get('id', '<none>')}"
                    )
                    arguments_done = True
                self._write_chunk(event)
                continue
            # Flush ordinary assistant text as soon as it arrives. If the
            # model starts a textual tool-call protocol, keep that event in
            # the buffer so the existing safe converter can handle it.
            event_text = ""
            if isinstance(data, dict) and data.get("type") == "response.output_text.delta":
                event_text = str(data.get("delta") or "")
            possible_tool_markup = _possible_tool_markup(event_text)
            if event_text and not possible_tool_markup and not buffered:
                send_event(event)
                continue
            buffered.extend(event)
            if isinstance(data, dict) and data.get("type") == "response.output_item.added":
                item = data.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    self._trace_log(
                        f"SSE 原生工具调用: detected function_call name={item.get('name', '<unknown>')} "
                        f"call_id={item.get('call_id', '<none>')}，开始透传"
                    )
                    send_headers()
                    self._write_chunk(bytes(buffered))
                    buffered.clear()
                    passthrough = True
        if passthrough:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        raw = bytes(buffered)
        if not raw and headers_sent:
            self.wfile.write(b"0\\r\\n\\r\\n")
            self.wfile.flush()
            return
        converted = convert_stream_tool_response(raw, getattr(self, "_request_tools", []))
        if converted != raw:
            self._log("将模型文本工具调用转换为 Responses function_call")
            self._trace_log("SSE 工具转换 llama.cpp text stream -> proxy parser -> Codex function_call")
        if headers_sent:
            # Plain text events were already sent. A non-converted remainder
            # is safe to stream as-is; converted tool markup remains buffered.
            if converted == raw and converted:
                self._write_chunk(converted)
            self.wfile.write(b"0\\r\\n\\r\\n")
            self.wfile.flush()
        else:
            self._upstream_headers(resp, content_length=len(converted))
            self.wfile.write(converted)
            self.wfile.flush()


def upstream_alive(target):
    """TCP 直连探测(与 svc.py port_up 同款): llama-server 加载中 /health 是 503,
    HTTP 语义探测会误判, TCP 能连上就说明进程在、端口在听。"""
    try:
        with socket.create_connection((target.hostname, target.port), timeout=1.5):
            return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Codex 工具兼容翻译代理")
    ap.add_argument("--listen", type=int, default=8847, help="本代理监听端口(默认 8847)")
    ap.add_argument("--target", default="127.0.0.1:8848", help="上游 llama-server 地址(默认 127.0.0.1:8848)")
    ap.add_argument("--dump", default=None, help="调试: 把改写后的请求 JSON 存到该目录")
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="进程日志目录(默认 codex_proxy_log)")
    ap.add_argument("--force-settings", default=None,
                    help="JSON: 强制 thinking/temperature/max_tokens，防止客户端覆盖启动页设置")
    args = ap.parse_args()

    global _PROCESS_LOG
    try:
        log_path, _PROCESS_LOG = _open_process_log(args.log_dir)
    except OSError as e:
        print(f"{proxy_label()}: 无法创建日志文件: {e}", flush=True)
        sys.exit(2)

    host, port = args.target.split(":")
    target = urlsplit(f"http://{args.target}")
    if args.dump:
        os.environ["CODEX_PROXY_DUMP"] = args.dump

    # 立即占用公共端口。大模型加载可能远超过代理原先的 30 秒等待，
    # 因此在上游尚未监听时保留代理，由请求转发返回暂不可用即可。
    srv = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    srv.target = args.target
    srv.up_host, srv.up_port = args.target.split(":")
    try:
        srv.force_settings = json.loads(args.force_settings) if args.force_settings else None
    except Exception as e:
        _process_log(f"process_error force_settings_invalid error={type(e).__name__}: {e}")
        _PROCESS_LOG.close()
        _PROCESS_LOG = None
        print(f"{proxy_label()}: --force-settings JSON 无效: {e}", flush=True)
        sys.exit(2)
    _process_log(
        f"process_start pid={os.getpid()} listen=127.0.0.1:{args.listen} "
        f"target={args.target} log={log_path} max_logs={MAX_PROCESS_LOGS}"
    )
    print(f"{proxy_label()}: 监听 127.0.0.1:{args.listen} → {args.target} (Codex 工具兼容已启用)", flush=True)

    # 上游哨兵：只在曾连接到上游后才执行退出逻辑，避免模型加载阶段
    # 因端口尚未监听而误杀代理。
    def watchdog():
        seen_upstream = False
        while True:
            time.sleep(2)
            if upstream_alive(target):
                seen_upstream = True
                continue
            # Keep the public endpoint alive while llama.cpp loads/restarts.
            # Requests receive a normal 502/503 during the outage, and new
            # requests work automatically once the backend is listening again.
    threading.Thread(target=watchdog, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _process_log(f"process_stop pid={os.getpid()}")
        try:
            _PROCESS_LOG.close()
        except (AttributeError, OSError, ValueError):
            pass
        _PROCESS_LOG = None


if __name__ == "__main__":
    main()
