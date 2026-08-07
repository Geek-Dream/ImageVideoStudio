#!/bin/bash
# ============================================================
# ImageVideoStudio · 控制台(Mac)
#   ./start.sh          菜单: 上面显示运行状态,1)启动 2)彻底关闭 3)监控后台生图 4)停止后台生图
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

# 后台生图 worker 是否在跑(读 test_jobs/worker.pid,进程探活)
test_worker_pid() {
    local pidf="$BASE/test_jobs/worker.pid"
    [ -f "$pidf" ] || return 1
    local pid; pid="$(cat "$pidf" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    return 1
}

# 选项3: 只读监控后台生图进度。Ctrl+C 退出只停监控,不动 worker。
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

do_status() {
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
    print(f"   🧪 测试场后台生图: 🟢 运行中{extra}(选3监控/选4停止)")
else:
    print("   🧪 测试场后台生图: ⚫ 未运行")
PYEOF
}

do_stop() {
    echo "⏳ 停止全部模型服务(生图/生视频/语言)…"
    python3 -c "import sys; sys.path.insert(0,'$BASE'); import svc; svc.stop_all()"
    echo "⏳ 关闭网页控制台…"
    lsof -ti:8860 | xargs kill 2>/dev/null
    # 注意: 测试场后台生图 worker 不在本选项范围——后台生图的意义就是关啥都杀不死它,
    # 要停只能选4(./start.sh killtest)或网页⚙️里杀死。
    sleep 1
    left=""
    for p in 8848 8849 8850 8860; do
        [ -n "$(lsof -ti:$p)" ] && left="$left $p"
    done
    if [ -n "$left" ]; then
        echo "⚠ 端口有残留:$left ,强制清理…"
        for p in $left; do lsof -ti:$p | xargs kill -9 2>/dev/null; done
        sleep 1
    fi
    # 最终确认: 四端口全灭 = 项目内存占用 0
    ok=1
    for p in 8848 8849 8850 8860; do
        [ -n "$(lsof -ti:$p)" ] && ok=0
    done
    if [ "$ok" = "1" ]; then
        echo "✔ 已彻底关闭: 8848/8849/8850/8860 全部清零,电脑恢复正常睡眠"
        if test_worker_pid >/dev/null; then
            echo "   🧪 后台生图 worker 仍在跑(它不属于彻底关闭范围;要停: 选4 或 网页⚙️里杀)"
        fi
    else
        echo "❌ 仍有端口未释放,请手动执行: lsof -ti:8848,8849,8850,8860 | xargs kill -9"
    fi
}

case "$1" in
    stop)     do_stop; exit 0;;
    status)   do_status; exit 0;;
    start)    exec python3 "$BASE/gen.py";;
    mon)      do_test_monitor; exit 0;;
    killtest) do_test_stop; exit 0;;
esac

# 无参数: 上面显示状态,下面四个选项
do_status
echo
echo "  1) 启动/打开网页操作台"
echo "  2) 彻底关闭(停全部模型+网页;后台生图不受影响,要停选4)"
echo "  3) 监控后台生图(只读,持续打印进度;Ctrl+C退出不影响生图)"
echo "  4) 停止后台生图脚本(测试场worker)"
echo
read -r -p "选 [1-4,回车=1]: " c
case "$c" in
    ""|1) exec python3 "$BASE/gen.py";;
    2)    do_stop;;
    3)    do_test_monitor;;
    4)    do_test_stop;;
    *)    echo "无效选择"; exit 1;;
esac
