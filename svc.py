#!/usr/bin/env python3
# ============================================================
# svc.py · 三类本地服务(生图/生视频/语言)的统一启停与状态
# 设计:
#   - 只有 gen.py import 本模块;本模块不 import gen,避免循环依赖。
#   - 状态 = PID文件探测(kill 0) AND 端口探测 双保险:
#     进程被外部杀死→PID没了→网页立刻显示已停止(满足"kill后网页要显示停止")。
#   - 内存互斥: 32GB 内存同时只跑一类大模型,启一个前必须先把别的停死透
#     (阻塞等 PID 消失 + 端口释放),否则 swap 双占拖死整机(教训见 PLAN.md)。
# 依赖: 仅标准库。
# ============================================================
import json, os, signal, socket, subprocess, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE, "config.json")
YAML_FILE = os.path.join(BASE, "extra_model_paths.yaml")
COMFY_OUT = os.path.join(BASE, "output", "comfy")

def _cfg():
    try:
        return json.load(open(CONFIG_FILE))
    except Exception:
        return {}

# 服务注册表: pid/log 都在项目根,端口从 config.json 读(有默认)
def _services():
    c = _cfg()
    return {
        "img": {"port": int(c.get("img_port", 8849)), "pid": os.path.join(BASE, "comfy.pid"),
                "log": os.path.join(BASE, "comfy.log"), "kind": "comfy", "name": "生图服务"},
        "vid": {"port": int(c.get("vid_port", 8850)), "pid": os.path.join(BASE, "comfy_vid.pid"),
                "log": os.path.join(BASE, "comfy_vid.log"), "kind": "comfy", "name": "生视频服务"},
        "llm": {"port": int(c.get("llm_port", 8848)), "pid": os.path.join(BASE, "llm.pid"),
                "log": os.path.join(BASE, "llm.log"), "kind": "llm", "name": "语言模型"},
    }

# ---------------- 基础探测 ----------------
def _read_pid(pid_file):
    try:
        return int(open(pid_file).read().strip())
    except Exception:
        return None

def pid_alive(pid_file):
    """PID 文件存在且进程活着(kill 0 不抛错)。"""
    pid = _read_pid(pid_file)
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def port_up(port, timeout=0.4):
    """TCP 直连探测端口。用原始 socket 而非 urllib: urllib 会读 http_proxy 环境变量,
    本机挂代理时探测 127.0.0.1 也被绕去走代理,每次白等满超时(start.sh 状态页卡顿的元凶)。"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except Exception:
        return False

def _comfy_child_alive(port):
    """ComfyUI 会在部分 macOS/venv 启动方式下由父进程派生真正监听端口的子进程。
    父 PID 退出但 ComfyUI 仍正常工作时,用端口上的命令行做一次窄匹配恢复状态。"""
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL).split()
        for raw in out:
            pid = int(raw)
            cmd = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], stderr=subprocess.DEVNULL).decode(errors="ignore")
            if "main.py" in cmd and f"--port {port}" in cmd and "extra-model-paths-config" in cmd:
                return True
    except Exception:
        pass
    return False

def svc_status(stype):
    """双保险状态: PID+端口;兼容 ComfyUI 派生子进程实际监听端口的情况。"""
    s = _services()[stype]
    ap = pid_alive(s["pid"])
    pu = port_up(s["port"])
    if not ap and s["kind"] == "comfy" and pu:
        ap = _comfy_child_alive(s["port"])
    return {"type": stype, "name": s["name"], "port": s["port"],
            "alive_pid": ap, "port_up": pu, "running": ap and pu}


def comfy_health(stype="img", timeout=3):
    """ComfyUI API 真健康: PID+端口+HTTP JSON 都要过,避免端口假 ready。"""
    st = svc_status(stype)
    logf = log_path(stype)
    detail = {**st, "api_ok": False, "log": logf}
    if not st["alive_pid"]:
        detail["error"] = f"{st['name']}进程未运行"
        return detail
    if not st["port_up"]:
        detail["error"] = f"{st['name']}端口 {st['port']} 未监听"
        return detail
    last_err = ""
    for path in ("/system_stats", "/queue"):
        try:
            url = f"http://127.0.0.1:{st['port']}{path}"
            json.load(urllib.request.urlopen(url, timeout=timeout))
            detail.update(api_ok=True, running=True, path=path, error="")
            return detail
        except Exception as e:
            last_err = str(e)
    detail["error"] = f"{st['name']}端口已开但 API 未就绪: {last_err}"
    return detail

def all_status():
    return {t: svc_status(t) for t in ("img", "vid", "llm")}

def pid_path(stype):
    """某类服务的 PID 文件路径(llm.py 启动后写这里,svc_status 才认)。"""
    return _services()[stype]["pid"]

def log_path(stype):
    return _services()[stype]["log"]

# ---------------- ComfyUI 模型路径配置(img/vid 共用一份 yaml) ----------------
def ensure_yaml():
    """把项目 models/ 目录挂进 ComfyUI(image 段 + video 段),幂等。"""
    c = _cfg()
    base = BASE
    txt = f"""# 本文件由 svc.py 自动生成,把项目的 models/ 目录挂进 ComfyUI
ivs:
  base_path: {base}/models
  checkpoints: image
  controlnet: image/controlnet
  diffusion_models: image/diffusion
  text_encoders: image/encoder
  clip: image/encoder
  vae: image/vae
  loras: image/loras
  ipadapter: image/ipadapter
  clip_vision: image/clip_vision
ivs_video:
  base_path: {base}/models
  diffusion_models: video
  text_encoders: video
  vae: video
  loras: video
"""
    try:
        if not os.path.exists(YAML_FILE) or open(YAML_FILE).read() != txt:
            open(YAML_FILE, "w").write(txt)
    except Exception:
        pass

# ---------------- 睡眠保护 ----------------
def keep_awake(pid):
    """模型进程活着期间禁止系统睡眠,但允许屏保和显示器息屏。
    ``-i`` 只阻止空闲系统睡眠,不会申请显示器唤醒或
    ``PreventUserIdleDisplaySleep`` 断言,因此屏保/黑屏仍按系统设置工作。
    caffeinate -w 跟随进程: 进程一死断言自动消失,电脑立刻恢复正常睡眠。
    仅 macOS 有 caffeinate,其他系统静默跳过。"""
    try:
        subprocess.Popen(["caffeinate", "-i", "-w", str(pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# ---------------- 启动 ----------------
def _launch_comfy(stype):
    """后台启动一个 ComfyUI 实例(8849 生图 / 8850 生视频),参数与旧 start.sh 一致。"""
    c = _cfg()
    comfy_dir = c.get("comfy_dir", "")
    if not comfy_dir or not os.path.exists(os.path.join(comfy_dir, "main.py")):
        return {"ok": False, "error": "未找到 ComfyUI,请先运行 ./install.sh"}
    s = _services()[stype]
    ensure_yaml()
    os.makedirs(COMFY_OUT, exist_ok=True)
    py = os.path.join(comfy_dir, "venv", "bin", "python")
    if not os.path.exists(py):
        py = "python3"
    logf = open(s["log"], "ab")
    try:
        args = [py, "main.py", "--port", str(s["port"]),
                "--output-directory", COMFY_OUT,
                "--extra-model-paths-config", YAML_FILE]
        # 两个 ComfyUI 实例不能抢同一个 comfyui.db；视频实例单独存自己的状态。
        if stype == "vid":
            args += ["--database-url", "sqlite:///" + os.path.join(BASE, "comfyui_vid.db")]
        p = subprocess.Popen(
            args,
            cwd=comfy_dir, stdout=logf, stderr=subprocess.STDOUT,
            # ComfyUI must survive after the worker/terminal that launched it
            # exits; otherwise a long model load can be killed with the shell.
            start_new_session=True)
        open(s["pid"], "w").write(str(p.pid))
        keep_awake(p.pid)   # 生图/生视频跑图期间禁止系统睡眠
        return {"ok": True, "pid": p.pid}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def start_svc(stype):
    """启动某类服务。除 llm 外交由本函数;llm 的启动由 llm.py 负责(P4 接入)。"""
    s = _services()[stype]
    if svc_status(stype)["running"]:
        return {"ok": True, "already": True}
    stop_others(except_t=stype)          # 内存互斥:先把别的停死透
    if s["kind"] == "comfy":
        return _launch_comfy(stype)
    # llm: 延迟 import,避免 P1 阶段 llm.py 未就绪时影响 img/vid
    try:
        import llm as _llm
        return _llm.start_llm()
    except Exception as e:
        return {"ok": False, "error": f"语言模型启动模块未就绪: {e}"}

# ---------------- 停止 ----------------
def _pids_on_port(port):
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL)
        return [int(x) for x in out.split()]
    except Exception:
        return []

def _llm_ports():
    """语言模型启用 Codex 代理时，8848 是代理，后端默认退到 8846。"""
    c = _cfg()
    public = int(c.get("llm_port", 8848))
    backend = int(c.get("llm_backend_port", public - 2))
    return tuple(dict.fromkeys((public, backend)))

def _kill_pid(pid, sig):
    try:
        os.kill(int(pid), sig)
    except (OSError, TypeError, ValueError):
        pass

def _pid_alive_value(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False

def stop_svc(stype):
    """彻底停止一类服务: TERM→等候→KILL→端口兜底，阻塞直到死透。"""
    s = _services()[stype]
    pid = _read_pid(s["pid"])
    pids = []
    if pid and pid_alive(s["pid"]):
        pids.append(pid)
    # Codex 兼容代理有自己的 PID 文件；llm.pid 记录的是 8846 后端，
    # 只杀其中一个会留下另一个进程继续占内存。
    proxy_file = os.path.join(BASE, "llm_proxy.pid") if stype == "llm" else ""
    proxy_pid = _read_pid(proxy_file) if proxy_file else None
    if proxy_pid:
        pids.append(proxy_pid)
    # vMLX keeps its own PID marker because it runs behind the settings proxy.
    # Include it even when llm.pid was removed or became stale.
    vmlx_file = os.path.join(BASE, "vmlx.pid") if stype == "llm" else ""
    vmlx_pid = _read_pid(vmlx_file) if vmlx_file else None
    if vmlx_pid:
        pids.append(vmlx_pid)
    for child in dict.fromkeys(pids):
        _kill_pid(child, signal.SIGTERM)
    for _ in range(8):
        if not any(_pid_alive_value(child) for child in pids):
            break
        time.sleep(1)
    for child in dict.fromkeys(pids):
        if _pid_alive_value(child):
            _kill_pid(child, signal.SIGKILL)
    if pids:
        time.sleep(1)
    try: os.remove(s["pid"])
    except OSError: pass
    if proxy_file:
        try: os.remove(proxy_file)
        except OSError: pass
    if vmlx_file:
        try: os.remove(vmlx_file)
        except OSError: pass
    # 兜底: 端口还占着(服务不是本程序起的,或代理/后端子进程)也一起停。
    ports = _llm_ports() if stype == "llm" else (s["port"],)
    for port in ports:
        for child in _pids_on_port(port):
            _kill_pid(child, signal.SIGTERM)
    # 阻塞等端口真正释放(最多 8s),确保内存还回来了。
    for _ in range(6):
        if not any(port_up(port) for port in ports): break
        time.sleep(1)
    for port in ports:
        if port_up(port):
            for child in _pids_on_port(port):
                _kill_pid(child, signal.SIGKILL)
    return {"ok": True, "running": any(port_up(port) for port in ports)}

def stop_others(except_t):
    """内存互斥:停掉 except_t 以外的所有服务,全部死透才返回。"""
    for t in ("img", "vid", "llm"):
        if t == except_t:
            continue
        st = svc_status(t)
        if st["alive_pid"] or st["port_up"]:
            stop_svc(t)

def stop_all():
    for t in ("img", "vid", "llm"):
        stop_svc(t)
    return {"ok": True}
