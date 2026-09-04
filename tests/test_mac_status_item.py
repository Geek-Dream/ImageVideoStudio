"""Executable contract for the macOS native status item."""
import unittest

from mac_status_item import format_status


class MacStatusTextTests(unittest.TestCase):
    def test_no_model_waits(self):
        self.assertEqual(format_status({"running": False, "loading": False, "model": None}), "他在干嘛？发呆等你启动模型~🥳")

    def test_loading(self):
        self.assertEqual(format_status({"loading": True, "model": {"id": "qwen"}}), "他在干嘛？发呆中~😶")

    def test_prompt_progress(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "prompt", "progress": 0.426}}), "他在干嘛？阅读内容~🤨 · 43%")

    def test_generation_tokens(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "gen", "n": 128}}, now=0), "他在干嘛？组织语言~🥲 · 已生成 128 token")

    def test_idle(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "idle"}}), "他在干嘛？发呆中~😶")

    def test_unknown_activity_is_error(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "weird"}}), "他在干嘛？状态异常~😶")

    def test_malformed_progress_does_not_raise(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "prompt", "progress": "bad"}}), "他在干嘛？阅读内容~🤨 · 0%")

    def test_tool(self):
        self.assertEqual(format_status({"running": True, "model": {"id": "qwen"}, "activity": {"state": "tool"}}, now=9), "他在干嘛？尝试使用工具~🔧")


if __name__ == "__main__":
    unittest.main()
