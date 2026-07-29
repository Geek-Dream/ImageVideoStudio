#!/bin/bash
# ============================================================
# ImageVideoStudio · 一键启动(Mac)
#   ./start.sh        检查环境 → 启动生图服务 → 自动打开网页
#   ./start.sh stop   停止生图服务
# 没有模型时:不开网页,直接告诉你去哪下模型。
# ============================================================
cd "$(dirname "$0")"
BASE="$(pwd)"
CONFIG="$BASE/config.json"
PID_FILE="$BASE/comfy.pid"
LOG="$BASE/comfy.log"

read_cfg() { python3 -c "import json,os;print(json.load(open('$CONFIG')).get('$1',''))" 2>/dev/null; }
write_cfg() { python3 -c "
import json,os
cfg=json.load(open('$CONFIG')) if os.path.exists('$CONFIG') else {}
cfg['$1']='$2'
json.dump(cfg,open('$CONFIG','w'),ensure_ascii=False,indent=2)"; }

COMFY_DIR="$(read_cfg comfy_dir)"
IMG_PORT="$(read_cfg img_port)"; IMG_PORT="${IMG_PORT:-8849}"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
port_up() { curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$IMG_PORT" 2>/dev/null; }

# ---------- 停止 ----------
if [ "$1" = "stop" ]; then
    if running; then
        kill "$(cat "$PID_FILE")" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8; do running || break; sleep 1; done
        running && kill -9 "$(cat "$PID_FILE")" 2>/dev/null
        rm -f "$PID_FILE"; echo "✔ 生图服务已停止"
    else echo "○ 生图服务本来就没在跑"; fi
    exit 0
fi

echo "================================================"
echo "  ImageVideoStudio · 一键启动"
echo "================================================"

# ---------- 第 1 步:找 ComfyUI ----------
if [ -z "$COMFY_DIR" ] || [ ! -f "$COMFY_DIR/main.py" ]; then
    for c in "$COMFYUI_DIR" "$BASE/ComfyUI" "$HOME/ComfyUI" "$HOME/Desktop/ComfyUI" "$HOME/开发/video_ai/ComfyUI"; do
        if [ -n "$c" ] && [ -f "$c/main.py" ]; then COMFY_DIR="$c"; break; fi
    done
fi
if [ -z "$COMFY_DIR" ] || [ ! -f "$COMFY_DIR/main.py" ]; then
    echo "✘ 没找到 ComfyUI,请先运行: ./install.sh"
    exit 1
fi
write_cfg comfy_dir "$COMFY_DIR"
echo "✔ ComfyUI: $COMFY_DIR"

# ---------- 第 2 步:检查模型(没有就不开网页) ----------
shopt -s nullglob
MODELS=( "$BASE"/models/image/*.safetensors "$BASE"/models/image/*.ckpt )
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "------------------------------------------------"
    echo "✘ models/image/ 里还没有图片模型,先不开网页。"
    echo "  请打开 models/image/README.html,里面有推荐模型和下载链接,"
    echo "  下载的 .safetensors 文件放进 models/image/ 后,再跑 ./start.sh"
    exit 1
fi
echo "✔ 检测到 ${#MODELS[@]} 个图片模型"

# ---------- 第 3 步:启动生图服务(若没在跑) ----------
if running || port_up; then
    echo "✔ 生图服务已在运行: http://127.0.0.1:$IMG_PORT"
else
    MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
    [ "$MEM_GB" -lt 16 ] && echo "⚠ 本机内存 ${MEM_GB}GB 偏小,建议关闭其他大软件"
    mkdir -p "$BASE/output/comfy" "$BASE/models/image/controlnet" "$BASE/models/image/diffusion" "$BASE/models/image/encoder" "$BASE/models/image/vae" "$BASE/models/video"
    YAML="$BASE/extra_model_paths.yaml"
    cat > "$YAML" <<EOF
# 本文件由 start.sh 自动生成,把项目的 models/ 目录挂进 ComfyUI
ivs:
  base_path: $BASE/models
  checkpoints: image
  controlnet: image/controlnet
  diffusion_models: image/diffusion
  text_encoders: image/encoder
  clip: image/encoder
  vae: image/vae
ivs_video:
  base_path: $BASE/models
  diffusion_models: video
  text_encoders: video
  vae: video
EOF
    if [ -x "$COMFY_DIR/venv/bin/python" ]; then PY="$COMFY_DIR/venv/bin/python"; else PY="python3"; fi
    echo "正在启动生图服务(端口 $IMG_PORT,首次加载模型要 1~2 分钟)…"
    ( cd "$COMFY_DIR" && nohup "$PY" main.py \
        --port "$IMG_PORT" \
        --output-directory "$BASE/output/comfy" \
        --extra-model-paths-config "$YAML" \
        > "$LOG" 2>&1 & echo $! > "$PID_FILE" )
    for _ in $(seq 1 48); do
        port_up && break
        if ! running; then
            echo "✘ 启动失败,日志末尾:"; tail -15 "$LOG"; rm -f "$PID_FILE"; exit 1
        fi
        sleep 5
    done
    port_up || { echo "✘ 等待超时,看日志: $LOG"; exit 1; }
    echo "✔ 生图服务已就绪"
fi

# ---------- 第 4 步:打开网页操作台 ----------
echo "------------------------------------------------"
echo "🎨 正在打开网页操作台(Ctrl+C 只关网页,服务继续跑;"
echo "   想彻底停止服务: ./start.sh stop)"
echo "------------------------------------------------"
exec python3 "$BASE/gen.py"
