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

# ---------- 停止(函数,stop 命令和退出菜单共用) ----------
stop_svc() {
    if running; then
        kill "$(cat "$PID_FILE")" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8; do running || break; sleep 1; done
        running && kill -9 "$(cat "$PID_FILE")" 2>/dev/null
        rm -f "$PID_FILE"
    fi
    # 兜底: 端口还占着(服务不是本次脚本起的)也一起停
    PIDS=$(lsof -ti :"$IMG_PORT" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null
        for _ in 1 2 3 4 5; do port_up || break; sleep 1; done
    fi
    if port_up; then echo "⚠ 端口 $IMG_PORT 还被占着,可能没停干净"; else echo "✔ 生图服务已彻底停止,内存已释放"; fi
}

if [ "$1" = "stop" ]; then
    stop_svc
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
# 用 gen.py 的 list_models() 数真实可用的模型(单文件 checkpoint + GGUF 三件套都算)
MODEL_INFO="$(python3 -c "
import sys; sys.path.insert(0,'$BASE')
import gen
ms = gen.list_models()
print(len(ms))
for m in ms: print(' - ' + m['name'])
" 2>/dev/null)"
MODEL_COUNT="$(echo "$MODEL_INFO" | head -1)"
if [ -z "$MODEL_COUNT" ] || [ "$MODEL_COUNT" = "0" ]; then
    echo "------------------------------------------------"
    echo "✘ 一个可用的图片模型都没有,先不开网页。"
    echo "  至少得有一个模型才能生成。请打开 models/image/README.html,"
    echo "  里面有推荐模型和下载链接;下载的模型文件放进 models/image/ 后,再跑 ./start.sh"
    exit 1
fi
echo "✔ 检测到 $MODEL_COUNT 个可用模型:"
echo "$MODEL_INFO" | tail -n +2

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
  loras: image/loras
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
echo "🎨 正在打开网页操作台(在网页那里按 Ctrl+C 或关闭网页后,"
echo "   回到这里会问你退出的方式)"
echo "------------------------------------------------"
trap '' INT   # 本脚本忽略 Ctrl+C,保证网页被杀后菜单能弹出来
( trap - INT; exec python3 "$BASE/gen.py" )  # 子进程把 Ctrl+C 重置回默认,网页才能被 Ctrl+C 杀掉(否则继承"忽略"会按不动)
trap - INT    # 恢复默认

# 网页关闭后,问用户怎么退
echo ""
echo "================================================"
echo "  网页已关闭。生图服务还在后台跑着(模型仍占内存)。"
echo "================================================"
echo "  1) 退出       —— 只退出本脚本,生图服务继续跑(下次 ./start.sh 直接用)"
echo "  2) 彻底退出   —— 连生图服务(ComfyUI)一起关,释放内存"
echo "------------------------------------------------"
while true; do
    read -r -p "选 1 或 2 后回车: " ans || { ans=1; echo; }  # 终端断开时按 1 处理
    case "$ans" in
        1) echo "✔ 已退出。生图服务仍在后台;想彻底关随时跑: ./start.sh stop"; break;;
        2) stop_svc; break;;
        *) echo "  请输入 1 或 2";;
    esac
done
