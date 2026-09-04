#!/usr/bin/env python3
"""Harmony 兼容代理入口。

实现与 codex_proxy.py 共用，保留两个文件名是为了兼容不同客户端和旧配置。
"""
import os
import runpy


if __name__ == "__main__":
    os.environ["LLM_PROXY_NAME"] = "harmony_proxy"
    runpy.run_path(os.path.join(os.path.dirname(__file__), "codex_proxy.py"), run_name="__main__")
