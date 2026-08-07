#!/usr/bin/env python3
# ============================================================
# llm.py · 语言模型(llama-server)启动/参数记忆/GPU 内存上限管理
# 模型定义在 llm_models.json(随项目);每模型三选项(思考/温度/max_tokens)
#   记忆在 llm_prefs.json(本机私有)。
# GPU 上限: 读 sysctl iogpu.wired_limit_mb,不够就用 osascript 弹 macOS
#   原生密码框提权设置(重启后系统自动还原,无需手动复位)。
# 内存互斥: 启动前经 svc.stop_others 把生图/生视频停死透(32GB 同时只跑一类)。
# 依赖: 仅标准库。import svc(svc 不 import 本模块,仅 start_svc 内懒加载,无环)。
# ============================================================
import json, os, re, subprocess, time, urllib.request
import svc

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(BASE, "llm_models.json")
PREFS_JSON = os.path.join(BASE, "llm_prefs.json")
CUR_FILE = os.path.join(BASE, "current_llm.json")

def _cfg():
    try:
        return json.load(open(os.path.join(BASE, "config.json")))
    except Exception:
        return {}

# ---------------- 模型定义 + 参数记忆 ----------------
def _prefs():
    try:
        return json.load(open(PREFS_JSON))
    except Exception:
        return {}

def list_models():
    """三模型定义,各并入 remembered prefs(没有则用 defaults)。"""
    try:
        ms = json.load(open(MODELS_JSON))["models"]
    except Exception:
        return []
    pr = _prefs()
    out = []
    for m in ms:
        d = dict(m.get("defaults", {}))
        d.update(pr.get(m["key"], {}))           # 记忆覆盖默认
        m = dict(m)
        m["prefs"] = d
        m["exists"] = os.path.exists(os.path.join(BASE, m["gguf"]))
        out.append(m)
    return out

def _find(model_id):
    for m in list_models():
        if m["id"] == model_id:
            return m
    return None

def save_pref(key, thinking, temp, max_tokens):
    pr = _prefs()
    pr[key] = {"thinking": bool(thinking), "temp": float(temp), "max_tokens": int(max_tokens)}
    try:
        json.dump(pr, open(PREFS_JSON, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------- GPU 内存上限 ----------------
def gpu_limit():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"],
                                           stderr=subprocess.DEVNULL).strip())
    except Exception:
        return 0

def ensure_gpu_limit(need_mb):
    """不够就弹 macOS 原生密码框提权设置。返回 {ok,cur,set,error,manual}。"""
    cur = gpu_limit()
    if cur >= need_mb:
        return {"ok": True, "cur": cur, "set": False}
    cmd = f"sysctl -w iogpu.wired_limit_mb={need_mb}"
    try:
        r = subprocess.run(["osascript", "-e",
                            f'do shell script "{cmd}" with administrator privileges'],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and gpu_limit() >= need_mb:
            return {"ok": True, "cur": gpu_limit(), "set": True}
        return {"ok": False, "cur": gpu_limit(), "error": "已取消授权或验证失败",
                "manual": f"sudo {cmd}"}
    except Exception as e:
        return {"ok": False, "cur": cur, "error": str(e), "manual": f"sudo {cmd}"}

# ---------------- 启动参数(严格按 Qwen3 start.sh 骨架) ----------------
def build_args(m, thinking, temp, max_tokens):
    c = _cfg()
    ctx = int(c.get("llm_ctx", 32768))
    port = int(c.get("llm_port", 8848))
    reasoning = "on" if (m.get("is_reasoning") and thinking) else "off"
    args = ["-m", os.path.join(BASE, m["gguf"]),
            "-ngl", "999",
            "-c", str(ctx),
            "-n", str(int(max_tokens)),
            "--temp", str(float(temp)),
            "--repeat-penalty", "1.1",   # 防"一句话反复说"死循环(蒸馏小模型尤其需要,默认1.0=关)
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--jinja",
            "--reasoning", reasoning]
    mm = m.get("mmproj")
    if mm and os.path.exists(os.path.join(BASE, mm)):   # 无视觉文件自动跳过
        args += ["--mmproj", os.path.join(BASE, mm),
                 "--image-min-tokens", "1024"]   # Qwen-VL 系官方建议值,低了图片识别精度差(启动日志警告)
    ct = m.get("chat_template")
    if ct and os.path.exists(os.path.join(BASE, ct)):
        args += ["--chat-template-file", os.path.join(BASE, ct)]
    return args

# ---------------- 启动 / 当前状态 ----------------
def start_llm(model_id, thinking, temp, max_tokens):
    """异步启动(加载 1-2 分钟,立即返回,前端轮询 /api/llm/current)。"""
    m = _find(model_id)
    if not m:
        return {"ok": False, "error": f"未知语言模型: {model_id}"}
    if not m["exists"]:
        return {"ok": False, "error": f"模型文件缺失: {m['gguf']}(检查 models/llm 软链)"}
    st = svc.svc_status("llm")
    cur = current()
    if st["running"] and cur.get("model", {}).get("id") == model_id:
        return {"ok": True, "already": True, "pid": None}
    # 互斥: 停掉生图/生视频 + 可能在跑的另一个语言模型
    svc.stop_others("llm")
    if svc.svc_status("llm")["alive_pid"]:
        svc.stop_svc("llm")
    # GPU 上限(按需弹密码框)
    g = ensure_gpu_limit(int(m.get("gpu_mb", 0)))
    if not g["ok"]:
        return {"ok": False, "error": f"GPU 内存上限未设置(需 {m['gpu_mb']}MB): {g.get('error','')}",
                "manual": g.get("manual")}
    c = _cfg()
    binx = c.get("llm_bin", "llama-server")
    args = build_args(m, thinking, temp, max_tokens)
    try:
        logf = open(svc.log_path("llm"), "ab")
        p = subprocess.Popen([binx] + args, cwd=BASE, stdout=logf, stderr=subprocess.STDOUT)
        open(svc.pid_path("llm"), "w").write(str(p.pid))
        svc.keep_awake(p.pid)   # 语言模型服务期间禁止系统睡眠(锁屏/合盖接电源不断活)
    except FileNotFoundError:
        return {"ok": False, "error": f"找不到 {binx}(需 brew install llama.cpp)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    save_pref(m["key"], thinking, temp, max_tokens)
    try:
        try:
            logsz = os.path.getsize(svc.log_path("llm"))
        except Exception:
            logsz = 0
        json.dump({"id": m["id"], "name": m["name"], "thinking": bool(thinking),
                   "temp": float(temp), "max_tokens": int(max_tokens),
                   "port": int(c.get("llm_port", 8848)), "pid": p.pid, "t0": time.time(),
                   "log_size": logsz, "paused": False},
                  open(CUR_FILE, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {"ok": True, "pid": p.pid, "loading": True}

def current():
    """当前语言模型状态: running=就绪可聊, loading=进程在但模型还在加载。"""
    st = svc.svc_status("llm")
    info = {}
    try:
        info = json.load(open(CUR_FILE))
    except Exception:
        info = {}
    if not st["alive_pid"]:
        return {"running": False, "loading": False, "model": None, "port": st["port"]}
    return {"running": st["running"], "loading": (st["alive_pid"] and not st["port_up"]),
            "model": info or None, "port": st["port"]}

def log_tail(n=15):
    try:
        lines = open(svc.log_path("llm"), "rb").read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""

# ---------------- 运行页统计 / 暂停 / 恢复 ----------------
def _session_tokens(offset):
    """从 llm.log 的 offset 之后累计本次会话的 输入/输出 token(print_timing 行)。"""
    pt = gt = 0
    try:
        with open(svc.log_path("llm"), "rb") as f:
            f.seek(max(0, int(offset)))
            txt = f.read().decode("utf-8", "replace")
        for m in re.finditer(r"prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens", txt):
            pt += int(m.group(1))
        for m in re.finditer(r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) tokens", txt):
            gt += int(m.group(1))
    except Exception:
        pass
    return pt, gt

def _activity():
    """解析 llm.log 尾部,回答"它现在在干嘛":
    idle(空闲等指令) / prompt(消化输入,带进度%) / gen(生成回复,带速度)。
    依据: llama-server 空闲打印 'all slots are idle';接任务打印 'processing task';
    prompt 阶段周期性打印 progress;任务结束 print_timing 给 tok/s。"""
    try:
        with open(svc.log_path("llm"), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 32768))
            txt = f.read().decode("utf-8", "replace")
    except Exception:
        return {"state": "unknown"}
    idle_at = txt.rfind("all slots are idle")
    pp = None
    for m in re.finditer(r"prompt processing, n_tokens =\s*(\d+), progress =\s*([\d.]+).*?([\d.]+) tokens per second", txt):
        pp = m
    ge = None
    for m in re.finditer(r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) (?:tokens|runs)[^\n]*?([\d.]+) tokens per second", txt):
        ge = m
    tg = None  # 新版 llama-server 生成中周期性打印: n_decoded = N, tg = X t/s
    for m in re.finditer(r"n_decoded =\s*(\d+),\s*tg =\s*([\d.]+) t/s", txt):
        tg = m
    task_at = max(txt.rfind("processing task"), pp.start() if pp else -1,
                  ge.start() if ge else -1, tg.start() if tg else -1)
    last_tps = float(ge.group(2)) if ge else (float(tg.group(2)) if tg else (float(pp.group(3)) if pp else None))
    if idle_at >= task_at:
        return {"state": "idle", "tps": last_tps}
    if pp and pp.start() > (ge.start() if ge else -1) and pp.start() > (tg.start() if tg else -1) and float(pp.group(2)) < 0.99:
        return {"state": "prompt", "progress": float(pp.group(2)),
                "n": int(pp.group(1)), "tps": float(pp.group(3))}
    if tg and tg.start() > (ge.start() if ge else -1):  # 生成中: 用实时 tg 速度(而非 prefill 速度)
        r = {"state": "gen", "tps": float(tg.group(2)), "n": int(tg.group(1))}
        if pp:
            r["read_tps"] = float(pp.group(3))  # 顺带带上本任务读输入的速度
        return r
    return {"state": "gen", "tps": last_tps}

def stats():
    """语言运行页数据: API 地址 + 启用时长 + 本次 token + 运行/加载/暂停态。"""
    c = _cfg()
    port = int(c.get("llm_port", 8848))
    st = svc.svc_status("llm")
    try:
        info = json.load(open(CUR_FILE))
    except Exception:
        info = {}
    paused = bool(info.get("paused"))
    if paused:
        elapsed = int(info.get("paused_elapsed", 0))
    elif info.get("t0") and st["alive_pid"]:
        elapsed = int(time.time() - info["t0"])
    else:
        elapsed = 0
    pt, gt = _session_tokens(info.get("log_size", 0)) if info else (0, 0)
    act = _activity() if (st["alive_pid"] and not paused) else {"state": "stopped"}
    return {"running": st["running"], "loading": (st["alive_pid"] and not st["port_up"]),
            "paused": paused, "model": (info or None), "port": port,
            "api_url": f"http://127.0.0.1:{port}/v1",
            "elapsed_sec": elapsed, "prompt_tokens": pt, "gen_tokens": gt,
            "activity": act}

def pause_llm():
    """暂停: 停掉 llama 进程释放内存,但保留模型与参数,可 resume 原样拉起。"""
    st = svc.svc_status("llm")
    if not st["alive_pid"]:
        return {"ok": False, "error": "语言模型未在运行"}
    try:
        info = json.load(open(CUR_FILE))
    except Exception:
        info = {}
    elapsed = int(time.time() - info.get("t0", time.time()))
    svc.stop_svc("llm")
    info["paused"] = True
    info["paused_elapsed"] = elapsed
    try:
        json.dump(info, open(CUR_FILE, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return {"ok": True, "paused": True}

def resume_llm():
    """恢复: 用 current_llm.json 里记录的模型与参数原样重启。"""
    try:
        info = json.load(open(CUR_FILE))
    except Exception:
        return {"ok": False, "error": "没有可恢复的模型记录"}
    if not info.get("id"):
        return {"ok": False, "error": "没有可恢复的模型记录"}
    return start_llm(info["id"], bool(info.get("thinking", True)),
                     float(info.get("temp", 0.7)), int(info.get("max_tokens", 16384)))

def close_llm():
    """彻底关闭: 停进程 + 删掉状态记录(current_llm.json)。
    与"暂停"不同——暂停要保留记录供"恢复"原样拉起;关闭则清除。
    否则 stats() 会一直读到 paused=true,每次打开页面都被强制跳回"已暂停"页。"""
    r = svc.stop_svc("llm")
    try:
        os.remove(CUR_FILE)
    except OSError:
        pass
    return r
