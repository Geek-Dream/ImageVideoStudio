import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import llm
from claude_local_config import find_local_qwen_provider, provider_environment


class ProxyModeTests(unittest.TestCase):
    def test_normalizes_numeric_modes_and_legacy_boolean(self):
        self.assertEqual(llm.normalize_proxy_mode(0), 0)
        self.assertEqual(llm.normalize_proxy_mode(1), 1)
        self.assertEqual(llm.normalize_proxy_mode(2), 2)
        self.assertEqual(llm.normalize_proxy_mode(True), 1)
        self.assertEqual(llm.normalize_proxy_mode(False), 0)
        self.assertEqual(llm.normalize_proxy_mode("2"), 2)
        self.assertEqual(llm.normalize_proxy_mode(99), 0)

    def test_reads_mode_from_running_state_with_legacy_fallback(self):
        self.assertEqual(llm.running_proxy_mode({"proxy_mode": 2, "codex_proxy": True}), 2)
        self.assertEqual(llm.running_proxy_mode({"codex_proxy": True}), 1)
        self.assertEqual(llm.running_proxy_mode({}), 0)

    def test_proxy_health_uses_backend_port_when_config_omits_it(self):
        with mock.patch.object(llm, "_cfg", return_value={"llm_port": 8848}):
            self.assertEqual(llm._health_port({"proxy_mode": 2}, {"port": 8848}), 8846)

    def test_saves_claude_mode_and_legacy_codex_flag(self):
        old_path = llm.PREFS_JSON
        try:
            with tempfile.TemporaryDirectory() as directory:
                llm.PREFS_JSON = os.path.join(directory, "prefs.json")
                llm.save_pref("MODEL", True, 0.4, 128, proxy_mode=2)
                with open(llm.PREFS_JSON, encoding="utf-8") as handle:
                    saved = json.load(handle)
                self.assertEqual(saved["MODEL"]["proxy_mode"], 2)
                self.assertFalse(saved["MODEL"]["codex_proxy"])
        finally:
            llm.PREFS_JSON = old_path


class CCSwitchProviderTests(unittest.TestCase):
    def test_reads_local_qwen_claude_provider_without_using_current_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cc-switch.db")
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE providers (id TEXT, app_type TEXT, name TEXT, settings_config TEXT, is_current INTEGER)")
            settings = {"env": {"ANTHROPIC_AUTH_TOKEN": "local-no-key", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8848", "ANTHROPIC_MODEL": "stale-mistral"}, "model": "Qwen3.6-35B"}
            db.execute("INSERT INTO providers VALUES (?, ?, ?, ?, ?)", ("local", "claude", "本地模型Qwen3.6-35B", json.dumps(settings), 0))
            db.commit(); db.close()
            provider = find_local_qwen_provider(path)
            self.assertEqual(provider["name"], "本地模型Qwen3.6-35B")
            env = provider_environment(provider)
            self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8848")
            self.assertEqual(env["ANTHROPIC_MODEL"], "Qwen3.6-35B")
            self.assertEqual(env["ANTHROPIC_DEFAULT_SONNET_MODEL"], "Qwen3.6-35B")

    def test_local_qwen_forces_local_endpoint_over_stale_cc_switch_url(self):
        provider = {
            "settings": {
                "env": {
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:15721",
                    "ANTHROPIC_MODEL": "Qwen3.6-35B",
                },
                "model": "Qwen3.6-35B",
            }
        }
        env = provider_environment(provider)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8848")

    def test_claude_local_ignores_user_settings_that_point_to_cc_switch(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "claude-local"), encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn('exec claude --setting-sources project,local "$@"', script)


if __name__ == "__main__":
    unittest.main()
