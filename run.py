#!/usr/bin/env python3
# ============================================================
# run.py · ImageVideoStudio 统一启动器(命令行菜单)
#   用法: python3 run.py
#   一个入口管所有本地功能,输入数字回车即执行:
#     1. 开网页控制台(gen.py,生图/生视频/语言模型/连载都在网页里)
#     2. 模板漫画连载(comic_gen.py,读 my_story/story.txt)
#     3. 质量批量出图(quality_test.py,prompts/ 全模型×全提示词)
#     4. 视频补帧到 60 帧(interp60.sh,需给一个 mp4 路径)
#     0. 退出
# 说明: 2/3 脚本自己会起生图服务(内存互斥,会先停语言/视频模型);
#       1 起的是网页后台,关掉终端不影响;4 是纯 ffmpeg,不占模型内存。
# ============================================================
import os, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))

MENU = """
================ ImageVideoStudio 统一启动器 ================
  1. 开网页控制台   (http://127.0.0.1:8860 ,功能最全)
  2. 模板漫画连载   (读 my_story/story.txt)
  3. 质量批量出图   (prompts/ 全模型×全提示词)
  4. 视频补帧 60 帧 (给一个 mp4)
  0. 退出
=========================================================="""

def run_web():
    """开网页控制台: 复用 start.sh(检查环境→起服务→开网页)。"""
    subprocess.call(["bash", os.path.join(BASE, "start.sh")])

def run_script(script):
    """跑一个自带起服务逻辑的 python 脚本(comic_gen / quality_test)。"""
    subprocess.call([sys.executable, os.path.join(BASE, script)])

def run_interp():
    path = input("输入要补帧的 mp4 路径: ").strip().strip('"').strip("'")
    if not path:
        print("没填路径,取消。")
        return
    if not os.path.isfile(path):
        print(f"找不到文件: {path}")
        return
    subprocess.call(["bash", os.path.join(BASE, "interp60.sh"), path])

ACTIONS = {"1": run_web, "2": lambda: run_script("comic_gen.py"),
           "3": lambda: run_script("quality_test.py"), "4": run_interp}

def main():
    while True:
        print(MENU)
        choice = input("选 [0-4]: ").strip()
        if choice == "0":
            print("拜拜。")
            return
        action = ACTIONS.get(choice)
        if action is None:
            print("无效选择,重新输。")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\n已中断,回主菜单。")
        input("\n---- 结束,按回车回主菜单 ----")

if __name__ == "__main__":
    main()
