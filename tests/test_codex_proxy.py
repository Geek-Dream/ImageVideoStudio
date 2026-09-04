import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import codex_proxy


TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "yield_time_ms": {"type": "number"},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    }
]


def text_response(text):
    return {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def sse_response(text):
    events = [
        ("response.created", {"type": "response.created", "response": {"id": "resp_test", "object": "response", "status": "in_progress"}}),
        ("response.output_item.added", {"type": "response.output_item.added", "item": {"id": "msg_test", "type": "message", "status": "in_progress", "role": "assistant", "content": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_test", "delta": text}),
        ("response.output_text.done", {"type": "response.output_text.done", "item_id": "msg_test", "text": text}),
        ("response.completed", {"type": "response.completed", "response": text_response(text)}),
    ]
    return b"".join(codex_proxy._sse_event(name, data) for name, data in events)


class ToolCodeParsingTests(unittest.TestCase):
    def test_parses_historical_python_dict_format(self):
        text = """Let me check.\n<tool_code>\n{'name': 'exec_command', 'arguments': {'cmd': \"sysctl hw.memsize\"}}\n</tool_code>"""
        call = codex_proxy.parse_text_tool_call(text, TOOLS)
        self.assertEqual(call["name"], "exec_command")
        self.assertEqual(json.loads(call["arguments"]), {"cmd": "sysctl hw.memsize"})

    def test_parses_json_format(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd","yield_time_ms":1000}}</tool_code>'
        call = codex_proxy.parse_text_tool_call(text, TOOLS)
        self.assertEqual(json.loads(call["arguments"])["yield_time_ms"], 1000)

    def test_parses_complete_payload_when_closing_tag_is_consumed(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd"}}'
        call = codex_proxy.parse_text_tool_call(text, TOOLS)
        self.assertEqual(json.loads(call["arguments"]), {"cmd": "pwd"})

    def test_rejects_truncated_payload_without_closing_tag(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd"}'
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_unknown_tool(self):
        text = '<tool_code>{"name":"delete_everything","arguments":{"cmd":"pwd"}}</tool_code>'
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_missing_required_argument(self):
        text = '<tool_code>{"name":"exec_command","arguments":{}}</tool_code>'
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_wrong_argument_type(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":42}}</tool_code>'
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_additional_argument(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd","unsafe":true}}</tool_code>'
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))


class ResponseConversionTests(unittest.TestCase):
    def test_converts_non_streaming_text_response(self):
        source = text_response('<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd"}}</tool_code>')
        result = codex_proxy.convert_text_tool_response(source, TOOLS)
        self.assertEqual(result["output"][0]["type"], "function_call")
        self.assertEqual(result["output"][0]["name"], "exec_command")
        self.assertTrue(result["output"][0]["call_id"].startswith("call_"))

    def test_preserves_normal_text(self):
        source = text_response("Memory is 32 GB")
        self.assertIs(codex_proxy.convert_text_tool_response(source, TOOLS), source)

    def test_preserves_existing_function_call(self):
        source = {"output": [{"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"}]}
        self.assertIs(codex_proxy.convert_text_tool_response(source, TOOLS), source)

    def test_converts_stream_to_function_events(self):
        text = '<tool_code>{"name":"exec_command","arguments":{"cmd":"pwd"}}</tool_code>'
        result = codex_proxy.convert_stream_tool_response(sse_response(text), codex_proxy.FALLBACK_TOOL_CATALOG).decode()
        self.assertIn("response.function_call_arguments.done", result)
        self.assertIn('"name":"exec_command"', result)
        self.assertNotIn("tool_code", result)

    def test_preserves_invalid_stream(self):
        source = sse_response('<tool_code>{"name":"exec_command","arguments":{}}</tool_code>')
        self.assertIs(codex_proxy.convert_stream_tool_response(source, TOOLS), source)


class XmlToolCallContractTests(unittest.TestCase):
    def test_parses_parameter_format_and_xml_entities(self):
        text = """<tool_call>
<function=exec_command>
<parameter=cmd>printf '&lt;ok&gt; &amp; done'</parameter>
<parameter=yield_time_ms>1000</parameter>
</function>
</tool_call>"""
        call = codex_proxy.parse_text_tool_call(text, TOOLS)
        self.assertEqual(call["name"], "exec_command")
        self.assertEqual(json.loads(call["arguments"]), {"cmd": "printf '<ok> & done'", "yield_time_ms": 1000})

    def test_rejects_unknown_function(self):
        text = "<tool_call><function=unknown><parameter=cmd>pwd</parameter></function></tool_call>"
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_missing_required_parameter(self):
        text = "<tool_call><function=exec_command></function></tool_call>"
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_rejects_invalid_typed_parameter(self):
        text = "<tool_call><function=exec_command><parameter=cmd>pwd</parameter><parameter=yield_time_ms>soon</parameter></function></tool_call>"
        self.assertIsNone(codex_proxy.parse_text_tool_call(text, TOOLS))

    def test_converts_xml_stream_to_function_events(self):
        text = """<tool_call>
  <function=view_image>
  <parameter=path>
  /tmp/screenshot_tk.png
  </parameter>
  </function>
  </tool_call>"""
        result = codex_proxy.convert_stream_tool_response(sse_response(text), codex_proxy.FALLBACK_TOOL_CATALOG).decode()
        self.assertIn("response.function_call_arguments.done", result)
        self.assertIn('"name":"view_image"', result)
        self.assertNotIn("<tool_call>", result)

    def test_stream_markup_split_across_deltas_is_buffered(self):
        chunks = ["<tool", "_call><function=exec_command><parameter=cmd>pwd</parameter></function></tool_call>"]
        events = [
            ("response.created", {"type": "response.created", "response": {"id": "resp_test", "object": "response", "status": "in_progress"}}),
            ("response.output_item.added", {"type": "response.output_item.added", "item": {"id": "msg_test", "type": "message", "status": "in_progress", "role": "assistant", "content": []}}),
        ]
        for chunk in chunks:
            events.append(("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_test", "delta": chunk}))
        text = "".join(chunks)
        events.extend([
            ("response.output_text.done", {"type": "response.output_text.done", "item_id": "msg_test", "text": text}),
            ("response.completed", {"type": "response.completed", "response": text_response(text)}),
        ])
        body = b"".join(codex_proxy._sse_event(name, data) for name, data in events)
        result = codex_proxy.convert_stream_tool_response(body, TOOLS).decode()
        self.assertIn("response.function_call_arguments.done", result)
        self.assertNotIn("<tool_call>", result)


class FallbackToolRewriteTests(unittest.TestCase):
    def rewrite(self, request, path="/v1/responses"):
        handler = codex_proxy.Handler.__new__(codex_proxy.Handler)
        handler.path = path
        handler.server = SimpleNamespace(force_settings=None)
        return json.loads(handler._maybe_rewrite(json.dumps(request).encode("utf-8")))

    def test_injects_tools_for_user_request_missing_catalogue(self):
        request = {
            "model": "gpt-local",
            "input": [{"role": "user", "content": "查看一下电脑内存"}],
            "tool_choice": "auto",
            "stream": True,
        }
        rewritten = self.rewrite(request)
        names = {tool["name"] for tool in rewritten["tools"]}
        self.assertIn("exec_command", names)
        self.assertIn("write_stdin", names)
        self.assertTrue(any("Tool execution protocol" in item.get("content", "")
                            for item in rewritten["input"] if item.get("role") == "system"))

    def test_does_not_inject_tools_for_title_schema_request(self):
        request = {
            "model": "gpt-local",
            "input": [
                {"role": "system", "content": "You are Codex."},
                {"role": "user", "content": "Generate a concise, single-line task title of at most 36 characters. User prompt: 查看一下电脑内存"},
            ],
            "tool_choice": "auto",
            "text": {
                "verbosity": "low",
                "format": {"type": "json_schema", "name": "codex_output_schema",
                           "schema": {"type": "object"}},
            },
        }
        rewritten = self.rewrite(request)
        self.assertNotIn("tools", rewritten)

    def test_does_not_inject_tools_for_image_request(self):
        request = {
            "model": "gpt-local",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "描述这张图"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ]}],
            "tool_choice": "auto",
        }
        rewritten = self.rewrite(request)
        self.assertIn("tools", rewritten)
        self.assertTrue(any(tool.get("name") == "view_image" for tool in rewritten["tools"]))

    def test_does_not_inject_when_tool_choice_is_omitted(self):
        request = {
            "model": "gpt-local",
            "input": [{"role": "user", "content": "普通无工具问答"}],
        }
        rewritten = self.rewrite(request)
        self.assertNotIn("tools", rewritten)


class ProcessLogTests(unittest.TestCase):
    def test_process_log_rotation_keeps_six_newest_proxy_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(6):
                path = os.path.join(directory, f"codex_proxy_old_{index}.log")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(str(index))
                os.utime(path, (index + 1, index + 1))
                paths.append(path)
            unrelated = os.path.join(directory, "other.log")
            with open(unrelated, "w", encoding="utf-8") as handle:
                handle.write("keep")
            path, handle = codex_proxy._open_process_log(directory)
            handle.close()
            logs = sorted(name for name in os.listdir(directory) if name.startswith("codex_proxy_") and name.endswith(".log"))
            self.assertEqual(len(logs), 6)
            self.assertFalse(os.path.exists(paths[0]))
            self.assertTrue(os.path.exists(unrelated))
            self.assertTrue(os.path.exists(path))

    def test_process_log_writes_timestamped_message(self):
        old_log = codex_proxy._PROCESS_LOG
        try:
            with tempfile.TemporaryDirectory() as directory:
                path, handle = codex_proxy._open_process_log(directory)
                codex_proxy._PROCESS_LOG = handle
                codex_proxy._process_log("trace test message")
                handle.close()
                with open(path, encoding="utf-8") as reader:
                    content = reader.read()
                self.assertIn("trace test message", content)
                self.assertRegex(content, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[\d{9}\] ")
        finally:
            codex_proxy._PROCESS_LOG = old_log


if __name__ == "__main__":
    unittest.main()
