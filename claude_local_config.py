#!/usr/bin/env python3
"""Read the local Claude/Qwen provider from CC Switch without changing state."""
import json
import os
import sys
import sqlite3

DEFAULT_DB = os.path.expanduser("~/.cc-switch/cc-switch.db")


def find_local_qwen_provider(db_path=None):
    path = db_path or os.environ.get("CC_SWITCH_DB") or DEFAULT_DB
    if not os.path.exists(path):
        raise FileNotFoundError(f"CC Switch 数据库不存在: {path}")
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT id, app_type, name, settings_config, is_current FROM providers WHERE app_type='claude' ORDER BY name").fetchall()
    finally:
        conn.close()
    for provider_id, app_type, name, settings_config, is_current in rows:
        if "qwen" in name.lower() or "本地模型" in name or "qwen" in settings_config.lower():
            try: settings = json.loads(settings_config or "{}")
            except (TypeError, ValueError) as exc: raise ValueError(f"CC Switch provider 配置不是 JSON: {name}") from exc
            return {"id": provider_id, "app_type": app_type, "name": name, "settings": settings, "is_current": bool(is_current)}
    raise LookupError("CC Switch 中没有找到 Claude 的本地 Qwen provider")


def provider_environment(provider):
    settings = provider.get("settings") or {}
    env = dict(settings.get("env") or {})
    # CC Switch stores the selected model at the provider root; old env
    # snapshots can contain a stale model name and must not win.
    if settings.get("model"):
        model = str(settings["model"])
        env["ANTHROPIC_MODEL"] = model
        for tier in ("HAIKU", "SONNET", "OPUS", "FABLE"):
            env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] = model
            env[f"ANTHROPIC_DEFAULT_{tier}_MODEL_NAME"] = model
    # This launcher is explicitly local: never inherit a stale CC Switch
    # endpoint (typically 15721) from the selected provider record.
    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8848"
    env.setdefault("ANTHROPIC_AUTH_TOKEN", "local-no-key")
    return {str(k): str(v) for k, v in env.items() if v is not None}


def main():
    provider = find_local_qwen_provider()
    env = provider_environment(provider)
    if "ANTHROPIC_MODEL" not in env:
        raise RuntimeError("CC Switch 本地 Qwen provider 没有设置 ANTHROPIC_MODEL 或 model")
    json.dump({"provider": provider["name"], "env": env}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
