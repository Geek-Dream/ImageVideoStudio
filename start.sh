#!/bin/bash
# ============================================================
# ImageVideoStudio · 控制台(Mac)
#   ./start.sh          菜单: 1)启动 2)关闭服务 3)Emoji开关 4/5)图片测试 6/7)视频测试
#   ./start.sh start    直接启动/打开网页(幂等:已运行则只开网页)
#   ./start.sh status   查看运行状态(网页+三类模型+内存占用)
#   ./start.sh stop     彻底关闭: 停全部模型+网页控制台(后台生图worker不受影响,要停用 killtest 或选4)
#   ./start.sh mon      监控后台生图进度(只读,Ctrl+C退出不影响生图)
#   ./start.sh killtest 停止后台生图worker(唯一停法之一;另一个是网页⚙️里杀死)
# 说明: 模型启动/切换在网页里点选;模型运行期间由 caffeinate 自动防睡眠,
#       模型一停防睡眠自动解除,电脑恢复正常休眠。
#       测试场后台生图worker独立于网页:关网页/关本脚本/选2彻底关闭都杀不死它,
#       只能选 4 / ./start.sh killtest / 网页⚙️暂停·杀死。
# ============================================================
cd "$(dirname "$0")"
BASE="$(pwd)"

# Codex 兼容代理开启时，8848 是代理，llama-server 使用内部后端端口。
# 读取配置，避免“彻底关闭”漏掉后端端口。
project_ports() {
    local backend
    backend="$(python3 - "$BASE/config.json" <<'PYEOF'
import json, sys
try:
    c = json.load(open(sys.argv[1], encoding="utf-8"))
    print(int(c.get("llm_backend_port", int(c.get("llm_port", 8848)) - 2)))
except Exception:
    print(8846)
PYEOF
)"
    echo "8848 8849 8850 8860 ${backend:-8846}"
}

mac_status_enabled() {
    python3 - "$BASE/config.json" <<'PYEOF'
import json, sys
try:
    print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("mac_status_item_enabled", True) else "0")
except Exception:
    print("1")
PYEOF
}

set_mac_status_enabled() {
    python3 - "$BASE/config.json" "$1" <<'PYEOF'
import json, sys
path, value = sys.argv[1], sys.argv[2]
try: data = json.load(open(path, encoding="utf-8"))
except Exception: data = {}
data["mac_status_item_enabled"] = value == "1"
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2); f.write("\n")
PYEOF
}

# 后台生图 worker 是否在跑(读 test_jobs/worker.pid,进程探活)
test_worker_pid() {
    local pidf="$BASE/test_jobs/worker.pid"
    [ -f "$pidf" ] || return 1
    local pid; pid="$(cat "$pidf" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    return 1
}

video_test_worker_pid() {
    local pidf="$BASE/video_test_jobs/worker.pid"
    [ -f "$pidf" ] || return 1
    local pid; pid="$(cat "$pidf" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    return 1
}

# 选项4: 只读监控后台生图进度。Ctrl+C 退出只停监控,不动 worker。
do_test_monitor() {
    local st="$BASE/test_jobs/status.json"
    if ! test_worker_pid >/dev/null; then
        echo "⚫ 后台生图没在跑(网页⚙️测试场建好任务点「一键生图」就会起)"
        [ -f "$st" ] && { echo "—— 上次记录 ——"; cat "$st"; echo; }
        return
    fi
    echo "🟢 后台生图监控中(每2秒刷新;Ctrl+C 只退出监控,不影响生图)…"
    while true; do
        python3 - "$st" <<'PYEOF'
import sys, json, os, time
f = sys.argv[1]
try:
    s = json.load(open(f, encoding="utf-8"))
except Exception:
    print("\r(还没有进度记录)        ", end="", flush=True); raise SystemExit
done, total = s.get("done", 0), s.get("total", 0)
ok, fail = s.get("ok", 0), s.get("fail", 0)
el = int((s.get("ts", 0) or time.time()) - (s.get("t0", 0) or time.time()))
state = {"running": "🟢生成中", "paused": "⏸已暂停", "idle": "⚪待命"}.get(s.get("state"), s.get("state", ""))
line = f"\r{state} {done}/{total}张 ✓{ok} ✗{fail}"
if s.get("folder"): line += f" 📁{s['folder']}"
if s.get("msg"):    line += f" · {s['msg']}"
line += f" · 本张{el}秒   "
print(line, end="", flush=True)
PYEOF
        sleep 2
    done
}

# 选项4: 停止后台生图 worker。先写 kill_all 让它体面退出,3秒后还在就强杀。
do_test_stop() {
    local pid; pid="$(test_worker_pid)"
    if [ -z "$pid" ]; then
        echo "⚫ 后台生图本来就没在跑"
        return
    fi
    echo "⏳ 停止后台生图 worker (pid $pid)…"
    mkdir -p "$BASE/test_jobs"
    echo '{"cmd":"kill_all"}' > "$BASE/test_jobs/control.json"
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        echo "   未自行退出,强杀…"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    rm -f "$BASE/test_jobs/worker.pid"
    if kill -0 "$pid" 2>/dev/null; then
        echo "❌ 没杀掉,手动执行: kill -9 $pid"
    else
        echo "✔ 后台生图 worker 已停止(已生成的图片保留在 output/imgtest/)"
    fi
}

# 选项5: 只读监控后台视频测试。Ctrl+C 只退出监控。
do_video_test_monitor() {
    local st="$BASE/video_test_jobs/status.json"
    if ! video_test_worker_pid >/dev/null; then
        echo "⚫ 后台视频测试没在跑"
        [ -f "$st" ] && { echo "—— 上次记录 ——"; cat "$st"; echo; }
        return
    fi
    echo "🟢 后台视频测试监控中(每2秒刷新;Ctrl+C不影响生成)…"
    while true; do
        python3 - "$st" <<'PYEOF'
import sys, json
try:
    s=json.load(open(sys.argv[1],encoding="utf-8"))
except Exception:
    print("\r(还没有进度记录)       ",end="",flush=True); raise SystemExit
line=f"\r{s.get('state','')} {s.get('done',0)}/{s.get('total',0)}段 ✓{s.get('ok',0)} ✗{s.get('fail',0)}"
if s.get('folder'): line+=f" 📁{s['folder']}"
if s.get('msg'): line+=f" · {s['msg']}"
if s.get('segment_elapsed') is not None: line+=f" · 本段{s['segment_elapsed']}秒/保护剩{s.get('watchdog_left',0)}秒"
print(line+"   ",end="",flush=True)
PYEOF
        sleep 2
    done
}

# 选项6: 停止所有后台视频测试脚本，不删除已经生成的视频。
do_video_test_stop() {
    local pid; pid="$(video_test_worker_pid)"
    if [ -z "$pid" ]; then
        echo "⚫ 后台视频测试本来就没在跑"
        return
    fi
    echo "⏳ 停止后台视频测试 worker (pid $pid)…"
    mkdir -p "$BASE/video_test_jobs"
    echo '{"cmd":"kill_all"}' > "$BASE/video_test_jobs/control.json"
    sleep 4
    if kill -0 "$pid" 2>/dev/null; then
        echo "   未自行退出,强杀 worker…"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    rm -f "$BASE/video_test_jobs/worker.pid"
    if kill -0 "$pid" 2>/dev/null; then
        echo "❌ 没杀掉,手动执行: kill -9 $pid"
    else
        echo "✔ 后台视频测试已停止(已生成的视频保留在 output/vidtest/)"
    fi
}

print_mac_status() {
    [ "$(uname -s 2>/dev/null)" = "Darwin" ] || return 0
    if [ "$(mac_status_enabled)" = "1" ]; then
        mpid=""; [ -f "$BASE/mac_status_item.pid" ] && mpid="$(cat "$BASE/mac_status_item.pid" 2>/dev/null)"
        mpstate="$(ps -p "$mpid" -o stat= 2>/dev/null | tr -d ' ')"
        if [ -n "$mpid" ] && kill -0 "$mpid" 2>/dev/null && [[ "$mpstate" != Z* ]]; then
            echo "Emoji状态：🟢 已启用"
        else
            echo "Emoji状态：⚫ 已停止"
        fi
    else
        echo "Emoji状态：未启用"
    fi
}

do_status() {
    print_mac_status
    python3 - "$BASE" <<'PYEOF'
import sys, subprocess
sys.path.insert(0, sys.argv[1])
import svc

def rss(pid):
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                           stderr=subprocess.DEVNULL).strip())
    except Exception:
        return 0

# 网页控制台(8860)
web_pids = subprocess.run(["lsof", "-ti:8860"], capture_output=True, text=True).stdout.split()
web_rss = sum(rss(p) for p in web_pids)
if web_pids:
    print(f"🖥  网页控制台(:8860): 🟢 运行中 pid {'/'.join(web_pids)} · 内存 {web_rss//1024}MB")
else:
    print("🖥  网页控制台(:8860): ⚫ 未运行")

# 三类模型服务(生图/生视频/语言)
total = web_rss
for t in ("img", "vid", "llm"):
    s = svc.svc_status(t)
    pid = open(svc.pid_path(t)).read().strip() if s["alive_pid"] else None
    r = rss(pid) if pid else 0
    total += r
    mark = "🟢 运行中" if s["running"] else ("🟡 加载中" if s["alive_pid"] else "⚫ 已停止")
    extra = f" · 内存 {r//1024}MB" if r else ""
    print(f"   {s['name']}(:{s['port']}): {mark}{extra}")
print(f"—— 项目总内存占用: {total//1024}MB ——")

# 后台生图 worker(测试场,独立于网页)
import os, json
wf = os.path.join(sys.argv[1], "test_jobs", "worker.pid")
alive = False
if os.path.exists(wf):
    try:
        pid = int(open(wf).read().strip())
        os.kill(pid, 0); alive = True
    except Exception:
        alive = False
if alive:
    extra = ""
    sf = os.path.join(sys.argv[1], "test_jobs", "status.json")
    try:
        s = json.load(open(sf, encoding="utf-8"))
        extra = f" · {s.get('done',0)}/{s.get('total',0)}张 ✓{s.get('ok',0)} ✗{s.get('fail',0)}"
    except Exception:
        pass
    print(f"   🧪 测试场后台生图: 🟢 运行中{extra}(选4监控/选5停止)")
else:
    print("   🧪 测试场后台生图: ⚫ 未运行")

vf = os.path.join(sys.argv[1], "video_test_jobs", "worker.pid")
valive = False
if os.path.exists(vf):
    try:
        vpid = int(open(vf).read().strip()); os.kill(vpid, 0); valive = True
    except Exception:
        pass
if valive:
    extra = ""
    try:
        s = json.load(open(os.path.join(sys.argv[1], "video_test_jobs", "status.json"), encoding="utf-8"))
        extra = f" · {s.get('done',0)}/{s.get('total',0)}段 ✓{s.get('ok',0)} ✗{s.get('fail',0)}"
    except Exception:
        pass
    print(f"   🎬 测试场后台生视频: 🟢 运行中{extra}(选6监控/选7停止)")
else:
    print("   🎬 测试场后台生视频: ⚫ 未运行")
PYEOF
}

# 启动网页控制台但不占用当前终端。网页进程独立运行，只有选 2/stop 才会停止。
start_web_background() {
    if lsof -ti:8860 >/dev/null 2>&1; then
        echo "🎨 ImageVideoStudio 已在后台运行: http://127.0.0.1:8860"
        open "http://127.0.0.1:8860" >/dev/null 2>&1 || true
        return 0
    fi
    echo "🚀 正在后台启动 ImageVideoStudio…"
    nohup python3 "$BASE/gen.py" >>"$BASE/web.log" 2>&1 </dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "🎨 网页已转入后台: http://127.0.0.1:8860 (PID $pid)"
}

# macOS 原生菜单栏状态项独立于网页运行;守护进程只在检测到语言模型时显示菜单项。
start_mac_status_background() {
    [ "$(uname -s 2>/dev/null)" = "Darwin" ] || return 0
    [ "$(mac_status_enabled)" = "1" ] || return 0
    local pid_file="$BASE/mac_status_item.pid"
    if [ -f "$pid_file" ]; then
        local old_pid; old_pid="$(cat "$pid_file" 2>/dev/null)"
        local old_state; old_state="$(ps -p "$old_pid" -o stat= 2>/dev/null | tr -d ' ')"
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null && [[ "$old_state" != Z* ]]; then
            return 0
        fi
        rm -f "$pid_file"
    fi
    nohup python3 "$BASE/mac_status_item.py" >>"$BASE/mac_status_item.log" 2>&1 </dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    printf '%s\n' "$pid" > "$pid_file"
}

stop_mac_status_background() {
    [ "$(uname -s 2>/dev/null)" = "Darwin" ] || return 0
    local pid_file="$BASE/mac_status_item.pid"
    [ -f "$pid_file" ] || return 0
    local pid; pid="$(cat "$pid_file" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

toggle_mac_status() {
    [ "$(uname -s 2>/dev/null)" = "Darwin" ] || { echo "⚪ Emoji 状态栏仅支持 macOS"; return 0; }
    if [ "$(mac_status_enabled)" = "1" ]; then
        set_mac_status_enabled 0; stop_mac_status_background; echo "🔕 Emoji 状态栏: 已停用"
    else
        set_mac_status_enabled 1; start_mac_status_background; echo "🔔 Emoji 状态栏: 已启用"
    fi
}

do_stop() {
    stop_mac_status_background
    echo "⏳ 停止全部模型服务(生图/生视频/语言)…"
    python3 -c "import sys; sys.path.insert(0,'$BASE'); import svc; svc.stop_all()"
    echo "⏳ 关闭网页控制台…"
    lsof -ti:8860 | xargs kill 2>/dev/null
    # 注意: 测试场后台生图 worker 不在本选项范围——后台生图的意义就是关啥都杀不死它,
    # 要停只能选4(./start.sh killtest)或网页⚙️里杀死。
    sleep 1
    left=""
    for p in $(project_ports); do
        [ -n "$(lsof -ti:$p)" ] && left="$left $p"
    done
    if [ -n "$left" ]; then
        echo "⚠ 端口有残留:$left ,强制清理…"
        for p in $left; do lsof -ti:$p | xargs kill -9 2>/dev/null; done
        # 后端端口可能没有出现在公共端口残留列表中，单独再清一次。
        for p in $(project_ports); do lsof -ti:$p | xargs kill -9 2>/dev/null; done
        sleep 1
    fi
    # 最终确认: 四端口全灭 = 项目内存占用 0
    ok=1
    for p in $(project_ports); do
        [ -n "$(lsof -ti:$p)" ] && ok=0
    done
    if [ "$ok" = "1" ]; then
        echo "✔ 已彻底关闭: 8846/8848/8849/8850/8860 全部清零,电脑恢复正常睡眠"
        if test_worker_pid >/dev/null; then
            echo "   🧪 后台生图 worker 仍在跑(它不属于彻底关闭范围;要停: 选4 或 网页⚙️里杀)"
        fi
    else
        echo "❌ 仍有端口未释放,请手动执行: lsof -ti:8846,8848,8849,8850,8860 | xargs kill -9"
    fi
}

case "$1" in
    stop)     do_stop; exit 0;;
    status)   do_status; exit 0;;
    emoji)    toggle_mac_status; exit 0;;
    start)    start_mac_status_background; start_web_background; exit 0;;
    mon)      do_test_monitor; exit 0;;
    killtest) do_test_stop; exit 0;;
    monvid)   do_video_test_monitor; exit 0;;
    killvid)  do_video_test_stop; exit 0;;
esac

# 无参数: 上面显示状态,下面七个选项
do_status
echo
echo "  1) 启动/打开网页操作台"
echo "  2) 彻底关闭(停全部模型+网页;后台生图不受影响,要停选4)"
echo "  3) 启用/停用 Emoji 原生状态栏"
echo "  4) 监控后台生图(只读,持续打印进度;Ctrl+C退出不影响生图)"
echo "  5) 停止后台生图脚本(测试场worker)"
echo "  6) 监控后台生视频(只读,Ctrl+C退出不影响生成)"
echo "  7) 停止后台所有视频生成脚本(测试场worker)"
echo
read -r -p "选 [1-7,回车=1]: " c
case "$c" in
    ""|1) start_mac_status_background; start_web_background;;
    2)    do_stop;;
    3)    toggle_mac_status;;
    4)    do_test_monitor;;
    5)    do_test_stop;;
    6)    do_video_test_monitor;;
    7)    do_video_test_stop;;
    *)    echo "无效选择"; exit 1;;
esac
