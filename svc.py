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
import json, os, signal, subprocess, time, urllib.request

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

def port_up(port, timeout=2):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=timeout)
        return True
    except Exception:
        return False

def svc_status(stype):
    """双保险状态: alive_pid=进程在, port_up=端口通, running=两者皆真。"""
    s = _services()[stype]
    ap = pid_alive(s["pid"])
    pu = port_up(s["port"])
    return {"type": stype, "name": s["name"], "port": s["port"],
            "alive_pid": ap, "port_up": pu, "running": ap and pu}

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
    """模型进程活着期间禁止系统睡眠: 锁屏/屏保照常(它们本来就不停进程),
    但系统不会休眠,CPU/GPU 一直干活; 接电源时合盖也不断活(-s)。
    caffeinate -w 跟随进程: 进程一死断言自动消失,电脑立刻恢复正常睡眠。
    仅 macOS 有 caffeinate,其他系统静默跳过。"""
    try:
        subprocess.Popen(["caffeinate", "-is", "-w", str(pid)],
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
        p = subprocess.Popen(
            [py, "main.py", "--port", str(s["port"]),
             "--output-directory", COMFY_OUT,
             "--extra-model-paths-config", YAML_FILE],
            cwd=comfy_dir, stdout=logf, stderr=subprocess.STDOUT)
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

def stop_svc(stype):
    """彻底停止一类服务: TERM→等8s→KILL→lsof 端口兜底。阻塞直到死透才返回。"""
    s = _services()[stype]
    pid = _read_pid(s["pid"])
    if pid and pid_alive(s["pid"]):
        try: os.kill(pid, signal.SIGTERM)
        except Exception: pass
        for _ in range(8):
            if not pid_alive(s["pid"]): break
            time.sleep(1)
        if pid_alive(s["pid"]):
            try: os.kill(pid, signal.SIGKILL)
            except Exception: pass
            time.sleep(1)
    try: os.remove(s["pid"])
    except OSError: pass
    # 兜底: 端口还占着(服务不是本程序起的,或 llm 子进程)也一起停
    for p in _pids_on_port(s["port"]):
        try: os.kill(p, signal.SIGTERM)
        except Exception: pass
    # 阻塞等端口真正释放(最多 6s),确保内存还回来了
    for _ in range(6):
        if not port_up(s["port"]): break
        time.sleep(1)
    return {"ok": True, "running": port_up(s["port"])}

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
