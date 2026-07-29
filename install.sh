#!/bin/bash
# ============================================================
# ImageVideoStudio · 一键安装 ComfyUI(Mac)
# 顺序: 官网 git clone → 国内镜像 → 提示你手动下载后指定文件夹
# 用法: ./install.sh
# ============================================================
set -e
cd "$(dirname "$0")"
BASE="$(pwd)"
COMFY_DIR="$BASE/ComfyUI"          # 默认装到项目目录下
CONFIG="$BASE/config.json"

echo "================================================"
echo "  ImageVideoStudio · 安装 ComfyUI"
echo "================================================"

# ---------- 前置检查 ----------
command -v git >/dev/null || { echo "✘ 没装 git,请先安装 Xcode 命令行工具: xcode-select --install"; exit 1; }
command -v python3 >/dev/null || { echo "✘ 没装 python3,请到 https://www.python.org 下载安装"; exit 1; }
echo "✔ git / python3 就绪 (python3: $(python3 --version 2>&1))"

# ---------- 已存在则跳过下载 ----------
if [ -f "$COMFY_DIR/main.py" ]; then
    echo "✔ 检测到已存在 ComfyUI: $COMFY_DIR"
else
    echo "------------------------------------------------"
    echo "  第 1 步:下载 ComfyUI 源码"
    echo "------------------------------------------------"
    OK=0
    # ① 官网
    echo "[1/2] 尝试官网 github.com ..."
    if git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"; then
        OK=1
    else
        echo "  官网失败,换国内镜像..."
        rm -rf "$COMFY_DIR"
        # ② 国内镜像
        echo "[2/2] 尝试镜像 gitclone.com ..."
        if git clone --depth 1 https://gitclone.com/github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"; then
            OK=1
        fi
    fi
    # ③ 都不行 → 让用户自己下
    if [ "$OK" != "1" ]; then
        rm -rf "$COMFY_DIR"
        echo "------------------------------------------------"
        echo "  ✘ 自动下载都失败了(多半是网络问题)"
        echo "  请你手动下载 ComfyUI,二选一:"
        echo "    网页: https://github.com/comfyanonymous/ComfyUI  (Code → Download ZIP,解压)"
        echo "    镜像: https://gitclone.com/github.com/comfyanonymous/ComfyUI"
        echo "------------------------------------------------"
        printf "下载解压后,把 ComfyUI 文件夹的完整路径粘贴到这里: "
        read -r USER_DIR
        USER_DIR="${USER_DIR%/}"
        if [ -f "$USER_DIR/main.py" ]; then
            COMFY_DIR="$USER_DIR"
            echo "✔ 使用你指定的目录: $COMFY_DIR"
        else
            echo "✘ 这个目录里没有 main.py,不对。装好后重跑 ./install.sh"; exit 1
        fi
    fi
fi

# ---------- 指定自定义目录(如果已经装过别处) ----------
printf "ComfyUI 安装目录 [%s](直接回车用它,或粘贴你已有的路径): " "$COMFY_DIR"
read -r CUSTOM
if [ -n "$CUSTOM" ]; then
    CUSTOM="${CUSTOM%/}"
    [ -f "$CUSTOM/main.py" ] && COMFY_DIR="$CUSTOM" || echo "⚠ 该路径无 main.py,仍用 $COMFY_DIR"
fi

# ---------- 依赖 ----------
echo "------------------------------------------------"
echo "  第 2 步:创建虚拟环境并安装依赖(约几分钟)"
echo "------------------------------------------------"
if [ ! -d "$COMFY_DIR/venv" ]; then
    python3 -m venv "$COMFY_DIR/venv"
fi
PY="$COMFY_DIR/venv/bin/python"
"$PY" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple || "$PY" -m pip install --upgrade pip
# Mac(Apple Silicon/Intel)直接用官方 torch(自带 MPS 加速)
"$PY" -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple || "$PY" -m pip install torch torchvision
"$PY" -m pip install -r "$COMFY_DIR/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple || "$PY" -m pip install -r "$COMFY_DIR/requirements.txt"

# ---------- 写入配置 ----------
echo "------------------------------------------------"
echo "  第 3 步:写入配置 config.json"
echo "------------------------------------------------"
python3 - "$CONFIG" "$COMFY_DIR" <<'PYEOF'
import json, sys, os
cfg_path, comfy = sys.argv[1], sys.argv[2]
cfg = {}
if os.path.exists(cfg_path):
    try: cfg = json.load(open(cfg_path))
    except Exception: cfg = {}
cfg.setdefault("img_port", 8849); cfg.setdefault("vid_port", 8850); cfg.setdefault("port", 8860)
cfg["comfy_dir"] = comfy
json.dump(cfg, open(cfg_path, "w"), ensure_ascii=False, indent=2)
print("✔ 已写入 comfy_dir =", comfy)
PYEOF

echo "================================================"
echo "  ✅ 安装完成!"
echo "  下一步:"
echo "    1. 把图片模型放进 models/image/(看该文件夹 README.html 有推荐+下载链接)"
echo "    2. 运行 ./start.sh 启动生图服务"
echo "    3. 运行 python3 gen.py 打开操作台"
echo "================================================"
