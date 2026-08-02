#!/bin/bash
# ============================================================
# ImageVideoStudio · 控制台(Mac)
#   ./start.sh          菜单: 上面显示运行状态,1)启动 2)彻底关闭
#   ./start.sh start    直接启动/打开网页(幂等:已运行则只开网页)
#   ./start.sh status   查看运行状态(网页+三类模型+内存占用)
#   ./start.sh stop     彻底关闭: 停全部模型+网页控制台,项目内存清零
# 说明: 模型启动/切换在网页里点选;模型运行期间由 caffeinate 自动防睡眠,
#       模型一停防睡眠自动解除,电脑恢复正常休眠。
# ============================================================
cd "$(dirname "$0")"
BASE="$(pwd)"

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
PYEOF
}

do_stop() {
    echo "⏳ 停止全部模型服务(生图/生视频/语言)…"
    python3 -c "import sys; sys.path.insert(0,'$BASE'); import svc; svc.stop_all()"
    echo "⏳ 关闭网页控制台…"
    lsof -ti:8860 | xargs kill 2>/dev/null
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
        echo "✔ 已彻底关闭: 8848/8849/8850/8860 全部清零,项目内存占用 = 0,电脑恢复正常睡眠"
    else
        echo "❌ 仍有端口未释放,请手动执行: lsof -ti:8848,8849,8850,8860 | xargs kill -9"
    fi
}

case "$1" in
    stop)   do_stop; exit 0;;
    status) do_status; exit 0;;
    start)  exec python3 "$BASE/gen.py";;
esac

# 无参数: 上面显示状态,下面两个选项
do_status
echo
echo "  1) 启动/打开网页操作台"
echo "  2) 彻底关闭(停全部模型+网页,项目内存清零)"
echo
read -r -p "选 [1/2,回车=1]: " c
case "$c" in
    ""|1) exec python3 "$BASE/gen.py";;
    2)    do_stop;;
    *)    echo "无效选择"; exit 1;;
esac
