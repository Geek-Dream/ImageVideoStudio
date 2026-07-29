#!/bin/bash
# ============================================================
# ImageVideoStudio · 启动生图服务(ComfyUI,Mac)
# 作用: 生成模型路径映射 → 启动 ComfyUI(端口 8849)
# 用法: ./start.sh        (启动生图服务)
#       ./start.sh stop   (停止)
# 视频服务请直接用 ComfyUI 网页(见 models/video/README.html)
# ============================================================
cd "$(dirname "$0")"
BASE="$(pwd)"
CONFIG="$BASE/config.json"
PID_FILE="$BASE/comfy.pid"
LOG="$BASE/comfy.log"

# ---------- 读配置 ----------
read_cfg() { python3 -c "import json,os;print(json.load(open('$CONFIG')).get('$1',''))" 2>/dev/null; }
COMFY_DIR="$(read_cfg comfy_dir)"
IMG_PORT="$(read_cfg img_port)"; IMG_PORT="${IMG_PORT:-8849}"

# ---------- 停止 ----------
running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
if [ "$1" = "stop" ]; then
    if running; then
        kill "$(cat "$PID_FILE")" 2>/dev/null; sleep 2
        running && kill -9 "$(cat "$PID_FILE")" 2>/dev/null
        rm -f "$PID_FILE"; echo "✔ 已停止生图服务"
    else echo "○ 生图服务本来就没在跑"; fi
    exit 0
fi

# ---------- 校验 ----------
if [ -z "$COMFY_DIR" ] || [ ! -f "$COMFY_DIR/main.py" ]; then
    echo "✘ 没找到 ComfyUI,请先运行 ./install.sh"
    exit 1
fi
if running; then
    echo "○ 生图服务已在运行: http://127.0.0.1:$IMG_PORT"
    exit 0
fi

# ---------- 内存提示(仅提示,不强制) ----------
MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
echo "本机内存: ${MEM_GB}GB"
if [ "$MEM_GB" -lt 16 ]; then
    echo "⚠ 内存偏小,建议关闭其他大软件,模型选小一点的"
fi

# ---------- 生成模型路径映射 ----------
mkdir -p "$BASE/output/comfy" "$BASE/models/image/controlnet" "$BASE/models/video"
YAML="$BASE/extra_model_paths.yaml"
cat > "$YAML" <<EOF
# 本文件由 start.sh 自动生成,把项目的 models/ 目录挂进 ComfyUI
ivs:
  base_path: $BASE/models
  checkpoints: image
  controlnet: image/controlnet
  vae: image
  clip: image
  diffusion_models: video
  text_encoders: video
EOF
echo "✔ 模型路径映射已生成: $YAML"

# ---------- 选 python ----------
if [ -x "$COMFY_DIR/venv/bin/python" ]; then
    PY="$COMFY_DIR/venv/bin/python"
else
    PY="python3"
fi
echo "使用解释器: $PY"

# ---------- 启动 ----------
echo "正在启动生图服务(端口 $IMG_PORT)… 日志: $LOG"
( cd "$COMFY_DIR" && nohup "$PY" main.py \
    --port "$IMG_PORT" \
    --output-directory "$BASE/output/comfy" \
    --extra-model-paths-config "$YAML" \
    > "$LOG" 2>&1 & echo $! > "$PID_FILE" )

# ---------- 等待就绪 ----------
for _ in $(seq 1 48); do
    code=$(curl -s --max-time 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$IMG_PORT" 2>/dev/null)
    [ "$code" = "200" ] && break
    if ! running; then
        echo "✘ 启动失败,日志末尾:"; tail -15 "$LOG"; rm -f "$PID_FILE"; exit 1
    fi
    sleep 5
done
echo "✔ 生图服务已就绪: http://127.0.0.1:$IMG_PORT"
echo "  现在运行 python3 gen.py 打开操作台(或直接刷新已打开的网页)"
