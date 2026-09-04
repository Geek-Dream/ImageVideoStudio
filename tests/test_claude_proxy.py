import json
import unittest

import claude_proxy


class ClaudeRequestConversionTests(unittest.TestCase):
    def test_converts_anthropic_messages_and_tools_to_openai_chat(self):
        request = {
            "model": "Qwen3.6-35B",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "You are local."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "查看内存"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "我来检查。"},
                    {"type": "tool_use", "id": "toolu_1", "name": "exec_command",
                     "input": {"cmd": "free -h"}},
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                                   "content": "Mem: 32Gi"}]},
            ],
            "tools": [{"name": "exec_command", "description": "Run command",
                        "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
            "stream": False,
        }
        converted = claude_proxy.anthropic_to_openai(request)
        self.assertEqual(converted["model"], "Qwen3.6-35B")
        self.assertEqual(converted["messages"][0], {"role": "system", "content": "You are local."})
        assistant = converted["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"cmd": "free -h"})
        self.assertEqual(converted["messages"][3]["role"], "tool")
        self.assertEqual(converted["tools"][0]["function"]["name"], "exec_command")

    def test_maps_sampling_parameters_without_inventing_stop(self):
        converted = claude_proxy.anthropic_to_openai({
            "model": "qwen", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.4, "top_p": 0.8,
        })
        self.assertEqual(converted["temperature"], 0.4)
        self.assertEqual(converted["top_p"], 0.8)
        self.assertNotIn("stop", converted)

    def test_converts_openai_response_to_anthropic_tool_use(self):
        source = {
            "id": "chatcmpl_1",
            "model": "Qwen3.6-35B",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "先执行命令。", "tool_calls": [{
                    "id": "toolu_1", "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"free -h"}'},
                }],
            }}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        converted = claude_proxy.openai_to_anthropic(source, "Qwen3.6-35B")
        self.assertEqual(converted["stop_reason"], "tool_use")
        self.assertEqual(converted["content"][0], {"type": "text", "text": "先执行命令。"})
        self.assertEqual(converted["content"][1]["type"], "tool_use")
        self.assertEqual(converted["content"][1]["input"], {"cmd": "free -h"})
        self.assertEqual(converted["usage"], {"input_tokens": 12, "output_tokens": 7})

    def test_converts_openai_stream_to_anthropic_events(self):
        chunks = [
            {"id": "chatcmpl_1", "model": "qwen", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "chatcmpl_1", "model": "qwen", "choices": [{"index": 0, "delta": {"content": "你好"}, "finish_reason": None}]},
            {"id": "chatcmpl_1", "model": "qwen", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
        events = list(claude_proxy.openai_stream_to_anthropic(chunks, "qwen", "msg_1"))
        self.assertEqual(events[0]["type"], "message_start")
        self.assertTrue(any(e["type"] == "content_block_delta" and e["delta"]["text"] == "你好" for e in events))
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_preserves_upstream_error_as_anthropic_error(self):
        converted = claude_proxy.openai_error_to_anthropic({"error": {"message": "model loading"}}, 503)
        self.assertEqual(converted, {"error": {"type": "api_error", "message": "model loading"}})


if __name__ == "__main__":
    unittest.main()
