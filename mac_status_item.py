#!/usr/bin/env python3
"""Independent macOS menu-bar status item for the running language model."""
import argparse
import logging
import os
import sys
import time

GEN_EMOJIS = ("🥲", "😅", "🤪", "😴", "🤬")
TOOL_EMOJIS = ("🧰", "🪚", "🛠︎", "🔧", "⛏")

def _cycle_emoji(options, now=None):
    return options[int((time.monotonic() if now is None else now) // 3) % len(options)]


def format_status(stats, now=None):
    """Convert llm.stats() data to the full user-facing status, or None when hidden."""
    if not isinstance(stats, dict):
        return "模型状态异常"
    model = stats.get("model")
    if not model:
        return "他在干嘛？发呆等你启动模型~🥳"
    if stats.get("paused"):
        return None
    if stats.get("loading"):
        return "他在干嘛？发呆中~😶"
    if not stats.get("running"):
        return None
    activity = stats.get("activity") or {}
    state = activity.get("state") if isinstance(activity, dict) else None
    if state == "prompt":
        try:
            progress = max(0, min(100, round(float(activity.get("progress", 0)) * 100)))
        except (TypeError, ValueError):
            progress = 0
        return f"他在干嘛？阅读内容~🤨 · {progress}%"
    if state == "gen":
        try:
            count = max(0, int(activity.get("n", 0)))
        except (TypeError, ValueError):
            count = 0
        return f"他在干嘛？组织语言~{_cycle_emoji(GEN_EMOJIS, now)} · 已生成 {count} token"
    if state == "tool":
        return f"他在干嘛？尝试使用工具~{_cycle_emoji(TOOL_EMOJIS, now)}"
    if state == "idle":
        return "他在干嘛？发呆中~😶"
    return "他在干嘛？状态异常~😶"


def short_title(full_status):
    if full_status is None:
        return ""
    if "阅读内容" in full_status:
        return "阅读🤨"
    if "组织语言" in full_status:
        return "组织语言" + full_status.split("~", 1)[1].split(" ", 1)[0]
    if "使用工具" in full_status or "尝试使用工具" in full_status:
        return "使用工具" + full_status.rsplit("~", 1)[-1]
    if "发呆中" in full_status:
        return "发呆😶"
    if "发呆等你启动模型" in full_status:
        return "等待模型🥳"
    return "异常😶"


class StatusController:
    """AppKit adapter kept separate from formatting so it can be tested headlessly."""
    def __init__(self, interval, project_root):
        self.interval = interval
        self.project_root = project_root
        self.status_item = None
        self.menu = None
        self.last_status = None
        self.target = None

    def update(self, stats):
        from AppKit import NSMenuItem, NSVariableStatusItemLength
        text = format_status(stats, time.monotonic())
        if text is None:
            if self.status_item is not None:
                self.bar.removeStatusItem_(self.status_item)
                self.status_item = None
            self.last_status = None
            return
        self.last_status = text
        if self.status_item is None:
            self.status_item = self.bar.statusItemWithLength_(NSVariableStatusItemLength)
            self.menu = self._build_menu()
            self.status_item.setMenu_(self.menu)
        self.status_item.button().setTitle_(short_title(text))
        self.menu.itemAtIndex_(0).setTitle_(text)

    def _build_menu(self):
        from AppKit import NSMenu, NSMenuItem
        menu = NSMenu.alloc().init()
        menu.addItemWithTitle_action_keyEquivalent_(self.last_status or "模型状态异常", None, "")
        menu.addItem_(NSMenuItem.separatorItem())
        quit_item = menu.addItemWithTitle_action_keyEquivalent_("退出状态栏", "terminate:", "q")
        if self.target is not None:
            quit_item.setTarget_(self.target)
        return menu

    def poll(self):
        try:
            import llm
            self.update(llm.stats())
        except Exception:
            logging.exception("status poll failed")
            if self.status_item is not None and self.menu is not None:
                self.menu.itemAtIndex_(0).setTitle_("状态暂时不可用")


def run(interval=1.0, project_root=None):
    project_root = project_root or os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    logging.basicConfig(filename=os.path.join(project_root, "mac_status_item.log"), level=logging.INFO)
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSStatusBar
        from Foundation import NSObject, NSTimer, NSRunLoop, NSRunLoopCommonModes
    except ImportError:
        logging.exception("PyObjC/AppKit is unavailable")
        return 0

    class AppDelegate(NSObject):
        def applicationDidFinishLaunching_(self, _notification):
            self.controller = StatusController(interval, project_root)
            self.controller.target = self
            self.controller.bar = NSStatusBar.systemStatusBar()
            # Create the waiting item immediately; the first stats read may race service startup.
            self.controller.update({"model": {"waiting": True}, "running": True, "activity": {"state": "idle"}})
            self.controller.poll()
            self.timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                interval, self, "tick:", None, True)
            NSRunLoop.currentRunLoop().addTimer_forMode_(self.timer, NSRunLoopCommonModes)

        def tick_(self, _timer):
            self.controller.poll()

        def refresh_(self, _sender):
            self.controller.poll()

        def terminate_(self, _sender):
            NSApplication.sharedApplication().terminate_(self)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit(0)
    raise SystemExit(run(args.interval, args.project_root))
