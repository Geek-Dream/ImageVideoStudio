#!/usr/bin/env python3
# ============================================================
# llm.py · 语言模型(llama-server)启动/参数记忆/GPU 内存上限管理
# 模型定义在 llm_models.json(随项目);每模型四选项(思考/温度/max_tokens/MTP 加速)
#   记忆在 llm_prefs.json(本机私有)。
# GPU 上限: 读 sysctl iogpu.wired_limit_mb,不够就用 osascript 弹 macOS
#   原生密码框提权设置(重启后系统自动还原,无需手动复位)。
# 内存互斥: 启动前经 svc.stop_others 把生图/生视频停死透(32GB 同时只跑一类)。
# 依赖: 仅标准库。import svc(svc 不 import 本模块,仅 start_svc 内懒加载,无环)。
# ============================================================
import json, os, re, shutil, subprocess, sys, time, urllib.request
import threading
import svc

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(BASE, "llm_models.json")
PREFS_JSON = os.path.join(BASE, "llm_prefs.json")
CUR_FILE = os.path.join(BASE, "current_llm.json")
PROXY_PID_FILE = os.path.join(BASE, "llm_proxy.pid")
VMLX_PID_FILE = os.path.join(BASE, "vmlx.pid")
VMLX_PORT_DEFAULT = 8848
VMLX_PYTHON_DEFAULT = os.path.join(BASE, ".venv-vmlx", "bin", "python")
FAST_MTP_LLAMA_BIN = os.path.join(BASE, "fastmtp-llama.cpp", "build-llvm", "bin", "llama-server")
FAST_MTP_LLAMA_LIB = os.path.join(BASE, "fastmtp-llama.cpp", "build-llvm", "bin")
GPU_LIMIT_FAILURE_COOLDOWN = 300
_GPU_LIMIT_LOCK = threading.Lock()
_GPU_LIMIT_LAST_FAILURE = 0.0

# 官方 GPT-OSS Metal 环境与权重全部隔离在项目 metal/ 下。
# 用户可把官方仓库和转换后的 model.bin 放到这里，也可在 config.json 覆盖路径。
METAL_DIR = os.path.join(BASE, "metal")
METAL_REPO_DEFAULT = os.path.join(METAL_DIR, "gpt-oss")
METAL_VENV_DEFAULT = os.path.join(METAL_DIR, ".venv")
METAL_MODEL_DEFAULT = os.path.join(METAL_DIR, "models", "gpt-oss-20b", "metal", "model.bin")

def _cfg():
    try:
        return json.load(open(os.path.join(BASE, "config.json")))
    except Exception:
        return {}

def _abs_path(value, fallback):
    """解析 config 中的 Metal 路径；相对路径固定相对于项目根目录。"""
    value = value or fallback
    return value if os.path.isabs(value) else os.path.join(BASE, value)

def _vmlx_paths(m=None):
    c = _cfg(); m = m or {}
    model_dir = _abs_path(m.get("model_dir"), os.path.join(BASE, "models", "llm", "Qwen3.6-35B-A3B-MXFP4-CRACK-MTP"))
    py = _abs_path(c.get("vmlx_python"), VMLX_PYTHON_DEFAULT)
    return {"model_dir": model_dir, "python": py}

def vmlx_status(m=None):
    p = _vmlx_paths(m)
    return {"ready": os.path.isdir(p["model_dir"]) and os.path.isfile(p["python"]),
            "model_dir": p["model_dir"], "python": p["python"]}

def _metal_paths(m=None):
    c = _cfg()
    m = m or {}
    repo = _abs_path(m.get("metal_repo") or c.get("metal_repo"), METAL_REPO_DEFAULT)
    venv = _abs_path(c.get("metal_venv"), METAL_VENV_DEFAULT)
    checkpoint = _abs_path(m.get("metal_model") or c.get("metal_checkpoint"), METAL_MODEL_DEFAULT)
    configured_python = c.get("metal_python")
    # 绝对路径/带目录的路径按文件处理；单独的 python3 名称保留给 PATH 查找。
    if configured_python and (os.path.isabs(configured_python) or os.sep in configured_python):
        py = _abs_path(configured_python, os.path.join(venv, "bin", "python"))
    else:
        py = configured_python or os.path.join(venv, "bin", "python")
    return {"repo": repo, "venv": venv, "python": py, "checkpoint": checkpoint}

def metal_status(m=None):
    """返回 Metal 配置状态，不下载模型；用于模型卡置灰和初始化页面。"""
    p = _metal_paths(m)
    repo_ok = os.path.isdir(p["repo"]) and os.path.isdir(os.path.join(p["repo"], "gpt_oss"))
    python_ok = bool(shutil.which(p["python"]) if not os.path.isabs(p["python"]) else
                     (os.path.isfile(p["python"]) and os.access(p["python"], os.X_OK)))
    package_ok = False
    package_error = ""
    if python_ok and repo_ok:
        try:
            r = subprocess.run([p["python"], "-c", "import gpt_oss; import mlx"], cwd=p["repo"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=12)
            package_ok = r.returncode == 0
            package_error = (r.stderr or "").strip()[-300:]
        except Exception as e:
            package_error = str(e)
    checkpoint_ok = os.path.isfile(p["checkpoint"])
    ready = repo_ok and python_ok and package_ok and checkpoint_ok
    missing = []
    if not repo_ok: missing.append("官方 gpt-oss 源码")
    if not python_ok: missing.append("Metal 虚拟环境")
    elif not package_ok: missing.append("gpt_oss/MLX 依赖")
    if not checkpoint_ok: missing.append("Metal model.bin 权重")
    return {"ready": ready, "repo_ok": repo_ok, "python_ok": python_ok,
            "package_ok": package_ok, "checkpoint_ok": checkpoint_ok,
            "missing": missing, "package_error": package_error,
            "repo": p["repo"], "venv": p["venv"], "python": p["python"],
            "checkpoint": p["checkpoint"], "workspace": METAL_DIR}

def metal_init(install=True):
    """创建隔离目录/虚拟环境，并在官方源码已放置时安装 Metal 依赖。
    不下载模型权重；源码不存在时只创建工作区并返回放置路径。"""
    os.makedirs(METAL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METAL_MODEL_DEFAULT), exist_ok=True)
    readme = os.path.join(METAL_DIR, "README.md")
    if not os.path.exists(readme):
        try:
            open(readme, "w", encoding="utf-8").write(
                "# GPT-OSS Metal\n\n"
                "把官方 openai/gpt-oss 源码放到 `gpt-oss/`，把转换后的\n"
                "`gpt-oss-20b/metal/model.bin` 放到 `models/gpt-oss-20b/metal/model.bin`。\n"
                "ImageVideoStudio 不会自动下载模型权重。\n")
        except OSError:
            pass
    p = _metal_paths()
    if not os.path.isfile(p["python"]):
        try:
            r = subprocess.run([sys.executable, "-m", "venv", p["venv"]], cwd=METAL_DIR,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
            if r.returncode != 0:
                return {"ok": False, "error": "创建 Metal 虚拟环境失败", "output": (r.stdout or "")[-1200:], "status": metal_status()}
        except Exception as e:
            return {"ok": False, "error": f"创建 Metal 虚拟环境失败: {e}", "status": metal_status()}
    p = _metal_paths()
    if not (os.path.isdir(p["repo"]) and os.path.isdir(os.path.join(p["repo"], "gpt_oss"))):
        return {"ok": True, "initialized": True, "installed": False,
                "message": "工作区已创建。请把官方 gpt-oss 源码放入 metal/gpt-oss，再次点击初始化安装依赖。",
                "status": metal_status()}
    if install and not metal_status().get("package_ok"):
        try:
            r = subprocess.run([p["python"], "-m", "pip", "install", "-e", ".[metal]"], cwd=p["repo"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
            if r.returncode != 0:
                return {"ok": False, "error": "安装官方 Metal 依赖失败", "output": (r.stdout or "")[-2000:], "status": metal_status()}
        except Exception as e:
            return {"ok": False, "error": f"安装官方 Metal 依赖失败: {e}", "status": metal_status()}
    st = metal_status()
    return {"ok": st["ready"], "initialized": True, "installed": st["package_ok"],
            "message": "Metal 环境已准备好" if st["ready"] else "运行环境已初始化，但还缺少 model.bin 权重。",
            "status": st}

# ---------------- 模型定义 + 参数记忆 ----------------
def _prefs():
    try:
        return json.load(open(PREFS_JSON))
    except Exception:
        return {}

def normalize_proxy_mode(value):
    """Return the shared proxy selector: 0=off, 1=Codex, 2=Claude.
    Existing boolean ``codex_proxy`` preferences remain readable as mode 1.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        mode = int(value)
    except (TypeError, ValueError):
        return 0
    return mode if mode in (0, 1, 2) else 0

def running_proxy_mode(info):
    """Read a persisted running mode, falling back to legacy Codex state."""
    if not isinstance(info, dict):
        return 0
    return normalize_proxy_mode(info.get("proxy_mode", info.get("codex_proxy", False)))

def _health_port(info, status):
    """Return the model server port used for readiness checks.

    When a protocol adapter owns the public port, its health response is not
    llama.cpp's health schema; probe the backend instead.  Keep the default
    backend relationship in one place so omitted ``llm_backend_port`` config
    cannot accidentally probe the adapter.
    """
    public = int(_cfg().get("llm_port", (status or {}).get("port", 8848)))
    if running_proxy_mode(info) > 0:
        return int(_cfg().get("llm_backend_port", public - 2))
    return int((status or {}).get("port", public))

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
        d.setdefault("mtp", bool(m.get("has_mtp", False)))   # 有 MTP 头的模型默认开启加速
        d.setdefault("codex_proxy", False)                   # codex 工具兼容代理默认关(开源项目:别人可不开)
        d["proxy_mode"] = normalize_proxy_mode(d.get("proxy_mode", d.get("codex_proxy", False)))
        d.setdefault("supports_codex_proxy", bool(m.get("supports_codex_proxy", m.get("backend") not in ("metal", "vmlx"))))
        d.setdefault("ctx", 32768)                           # 上下文大小默认 32K(可在启动参数页调整)
        d.setdefault("parallel", 1)                          # 单会话:避免多个 slot 争抢 KV cache
        d.setdefault("budget", -1)                           # 思考长度上限: -1=不限, N>0=最多想N个token
        d.setdefault("reasoning_level", "high")              # 思考档位: high/medium/low(同步给 DSH 用)
        m = dict(m)
        m["prefs"] = d
        if m.get("backend") == "metal":
            st = metal_status(m)
            m["metal_status"] = st
            m["exists"] = bool(st["ready"])
            m["missing_reason"] = "、".join(st["missing"])
        elif m.get("backend") == "vmlx":
            st = vmlx_status(m)
            m["vmlx_status"] = st
            m["exists"] = st["ready"]
            m["missing_reason"] = "、".join(x for x, ok in (("vMLX 虚拟环境", os.path.isfile(st["python"])), ("MXFP4 模型目录", os.path.isdir(st["model_dir"]))) if not ok)
        else:
            gguf = m.get("gguf")
            m["exists"] = bool(gguf and os.path.exists(os.path.join(BASE, gguf)))
        out.append(m)
    return out

def _find(model_id):
    for m in list_models():
        if m["id"] == model_id:
            return m
    return None

def save_pref(key, thinking, temp, max_tokens, mtp=None, codex_proxy=None, ctx=None, budget=None, reasoning_level=None, parallel=1, proxy_mode=None):
    pr = _prefs()
    pr[key] = {"thinking": bool(thinking), "temp": float(temp), "max_tokens": int(max_tokens)}
    if mtp is not None:                          # 保留旧记录兼容;新记录带上 MTP 开关
        pr[key]["mtp"] = bool(mtp)
    if codex_proxy is not None:
        pr[key]["codex_proxy"] = bool(codex_proxy)
    if proxy_mode is not None:
        pr[key]["proxy_mode"] = normalize_proxy_mode(proxy_mode)
        pr[key]["codex_proxy"] = normalize_proxy_mode(proxy_mode) == 1
    if ctx is not None:
        pr[key]["ctx"] = int(ctx)
    pr[key]["parallel"] = max(1, min(8, int(parallel)))
    if budget is not None:
        pr[key]["budget"] = int(budget)
    if reasoning_level is not None:
        pr[key]["reasoning_level"] = str(reasoning_level)
    try:
        with open(PREFS_JSON, "w", encoding="utf-8") as handle:
            json.dump(pr, handle, ensure_ascii=False, indent=2)
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
    """Set the GPU limit at most once per cooldown after a failed auth attempt."""
    global _GPU_LIMIT_LAST_FAILURE
    cur = gpu_limit()
    if cur >= need_mb:
        return {"ok": True, "cur": cur, "set": False}
    cmd = f"sysctl -w iogpu.wired_limit_mb={need_mb}"
    with _GPU_LIMIT_LOCK:
        now = time.monotonic()
        if now - _GPU_LIMIT_LAST_FAILURE < GPU_LIMIT_FAILURE_COOLDOWN:
            return {"ok": False, "cur": cur, "error": "刚才的管理员授权失败，已暂缓重复弹窗",
                    "manual": f"sudo {cmd}"}
        try:
            r = subprocess.run(["osascript", "-e",
                                f'do shell script "{cmd}" with administrator privileges'],
                               capture_output=True, text=True, timeout=180)
            if r.returncode == 0 and gpu_limit() >= need_mb:
                _GPU_LIMIT_LAST_FAILURE = 0.0
                return {"ok": True, "cur": gpu_limit(), "set": True}
            _GPU_LIMIT_LAST_FAILURE = now
            return {"ok": False, "cur": gpu_limit(), "error": "已取消授权或验证失败",
                    "manual": f"sudo {cmd}"}
        except Exception as e:
            _GPU_LIMIT_LAST_FAILURE = now
            return {"ok": False, "cur": cur, "error": str(e), "manual": f"sudo {cmd}"}

# ---------------- 启动参数(严格按 Qwen3 start.sh 骨架) ----------------
def build_args(m, thinking, temp, max_tokens, mtp=False, codex_proxy=False, ctx=None, budget=-1, parallel=1):
    c = _cfg()
    # 上下文大小: 用户选的 ctx 优先, 没选用 config.json 的 llm_ctx, 再没有用 32K。
    if ctx is None:
        ctx = int(c.get("llm_ctx", 32768))
    else:
        ctx = int(ctx)
    ctx = max(8192, min(131072, ctx))
    parallel = max(1, min(8, int(parallel)))
    # 端口策略: 不开代理 → llama-server 直接占公共端口(llm_port,如 8848), 一切照旧;
    #           开了 codex 代理 → 代理占公共端口做透明转发, llama-server 退到内部端口(llm_backend_port),
    #             codex/网页/现有配置全部不用改, 照常连 8848。
    pub = int(c.get("llm_port", 8848))
    port = int(c.get("llm_backend_port", pub - 2)) if codex_proxy else pub
    reasoning = "on" if (m.get("is_reasoning") and thinking) else "off"
    args = ["-m", os.path.join(BASE, m["gguf"]),
            "-ngl", "auto",
            "-c", str(ctx),
            "-np", str(parallel),
            "-n", str(int(max_tokens)),
            "--temp", str(float(temp)),
            "--top-k", "20",
            "--top-p", "0.95" if thinking else "0.8",
            "--repeat-penalty", "1.1",   # 防"一句话反复说"死循环(蒸馏小模型尤其需要,默认1.0=关)
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--jinja",
            "--reasoning", reasoning]
    if m.get("is_reasoning") and thinking and budget is not None and int(budget) >= 0:
        # 思考长度上限: 限制"想多久再答"; -1=不限(不加参数), 0=不想, N>0=最多N个思考token
        args += ["--reasoning-budget", str(int(budget))]
    if mtp and m.get("has_mtp"):   # MTP 收益取决于模型、后端和接受率，不能假定一定提速
        args += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(int(m.get("spec_draft_n_max", 3)))]
        draft = m.get("spec_draft_model")
        if draft and os.path.exists(os.path.join(BASE, draft)):
            args += ["--spec-draft-model", os.path.join(BASE, draft)]
    mm = m.get("mmproj")
    if mm and os.path.exists(os.path.join(BASE, mm)):   # 无视觉文件自动跳过
        args += ["--mmproj", os.path.join(BASE, mm),
                 "--image-min-tokens", "1024"]   # Qwen-VL 系官方建议值,低了图片识别精度差(启动日志警告)
    ct = m.get("chat_template")
    if ct and os.path.exists(os.path.join(BASE, ct)):
        args += ["--chat-template-file", os.path.join(BASE, ct)]
    return args

# ---------------- 启动 / 当前状态 ----------------
# Codex 兼容代理: 统一处理 Responses API、工具结构和消息格式，
# 让 llama-server 能正确识别 MCP/记忆库工具。纯标准库, 跨平台。
def _proxy_pid():
    try:
        return int(open(PROXY_PID_FILE).read().strip())
    except Exception:
        return 0

def _vmlx_pid():
    try:
        return int(open(VMLX_PID_FILE).read().strip())
    except Exception:
        return 0

def _vmlx_args(m, thinking, temp, max_tokens, mtp, ctx, port, parallel=1):
    p = _vmlx_paths(m); c = _cfg()
    args = [p["python"], "-m", "vmlx_engine.cli", "serve", p["model_dir"],
            "--host", "127.0.0.1", "--port", str(port), "--served-model-name", m["id"],
            "--max-num-seqs", str(max(1, min(8, int(parallel)))), "--max-prompt-tokens", str(ctx),
            # 32GB Apple unified memory: chunk prefill and keep cache bounded so
            # a long conversation does not create a one-shot Metal allocation.
            "--prefill-batch-size", "128", "--prefill-step-size", "512",
            "--completion-batch-size", "128", "--cache-memory-mb", "2048",
            # vMLX defaults to 8-token streaming batches.  Non-MTP sessions
            # favor interactive output; MTP keeps batching to avoid client
            # backpressure reducing speculative-decoding throughput.
            "--stream-interval", "8" if mtp else "1",
            "--max-tokens", str(int(max_tokens)), "--default-temperature", str(float(temp)),
            "--default-repetition-penalty", "1.0" if mtp else "1.1",
            "--default-enable-thinking", "true" if thinking else "false"]
    # vMLX's OpenAI-compatible endpoint only returns structured tool calls
    # when automatic tool choice is explicitly enabled. Without this flag,
    # Codex/MCP schemas are injected into the prompt and Qwen emits raw
    # <tool_call> markup instead of a callable Responses API item.
    args += ["--enable-auto-tool-choice"]
    if mtp and m.get("has_mtp"):
        args += ["--native-mtp-depth", str(int(m.get("vmlx_mtp_depth", 1))), "--native-mtp-sampling-policy", "compatible-only"]
    else:
        args += ["--disable-native-mtp"]
    return args

def _health_ready(port):
    """端口监听不等于模型已加载；llama/vMLX 都用 /health 作为就绪信号。"""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/health")
        with urllib.request.urlopen(req, timeout=0.8) as r:
            data = json.loads(r.read().decode("utf-8", "replace") or "{}")
        return data.get("status") in ("ok", "healthy") and data.get("model_loaded", True) is not False
    except Exception:
        return False

def _stop_proxy():
    pid = _proxy_pid()
    if pid:
        try:
            subprocess.run(["kill", str(pid)], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    try:
        os.remove(PROXY_PID_FILE)
    except OSError:
        pass

def _start_proxy(c, force_settings=None, entry="codex_proxy.py", proxy_mode=1):
    """Start exactly one adapter on the shared public model port."""
    try:
        pub = int(c.get("llm_port", 8848))
        back = int(c.get("llm_backend_port", pub - 2))
        logf = open(svc.log_path("llm"), "ab")
        entry = "claude_proxy.py" if normalize_proxy_mode(proxy_mode) == 2 else entry
        command = [sys.executable, os.path.join(BASE, entry),
                   "--listen", str(pub), "--target", f"127.0.0.1:{back}"]
        # Keep rewritten requests locally for diagnosis of client protocol
        # mismatches. The adapter uses nanosecond names so concurrent requests
        # do not overwrite one another.
        dump_dir = os.path.join(BASE, ".proxy-debug")
        if entry == "codex_proxy.py":
            command += ["--dump", dump_dir]
        if force_settings:
            command += ["--force-settings", json.dumps(force_settings)]
        # Keep the proxy alive when the caller is a short-lived CLI/API
        # process (for example start.sh launching a detached model).
        p = subprocess.Popen(command, cwd=BASE, stdout=logf,
                             stderr=subprocess.STDOUT, start_new_session=True)
        open(PROXY_PID_FILE, "w").write(str(p.pid))
        return p.pid
    except Exception:
        return None

def _sync_dsh_reasoning(level):
    """把思考档位同步到 DSH 配置(local-qwen 的 reasoning 字段), 让 DSH 用的就是页面选的档位。
    尽力而为: DSH 没装/没配 local-qwen/文件不存在都静默跳过, 不影响本地模型启动。"""
    try:
        path = os.path.expanduser("~/.dsh/settings.yaml")
        if not os.path.exists(path):
            return
        txt = open(path, encoding="utf-8").read()
        if "local-qwen" not in txt:
            return
        # 只替换 local-qwen 段里的 reasoning: 行(块缩进下的)
        import re
        new, n = re.subn(r"(local-qwen:.*?reasoning:\s*)(high|medium|low)",
                         lambda m: m.group(1) + level, txt, count=1, flags=re.S)
        if n:
            open(path, "w", encoding="utf-8").write(new)
    except Exception:
        pass

def _metal_args(m, max_tokens, temp, ctx, reasoning_level, thinking):
    """官方 responses_api Metal 服务启动参数；上下文/输出在请求中传递。"""
    p = _metal_paths(m)
    c = _cfg()
    port = int(c.get("llm_port", 8848))
    return [p["python"], "-m", "gpt_oss.responses_api.serve",
            "--checkpoint", p["checkpoint"], "--port", str(port),
            "--inference-backend", "metal"]

def _ensure_mac_status_item():
    """Keep the native status supervisor independent from the browser launcher."""
    if sys.platform != "darwin":
        return
    if _cfg().get("mac_status_item_enabled", True) is False:
        return
    pid_file = os.path.join(BASE, "mac_status_item.pid")
    try:
        if os.path.isfile(pid_file):
            pid = int(open(pid_file).read().strip())
            os.kill(pid, 0)
            state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
            if state and not state.startswith("Z"):
                return
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(BASE, "mac_status_item.log"), "ab") as logf:
            child = subprocess.Popen([sys.executable, os.path.join(BASE, "mac_status_item.py")],
                                     cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        with open(pid_file, "w") as f:
            f.write(str(child.pid))
    except Exception:
        pass

def start_llm(model_id, thinking, temp, max_tokens, mtp=False, codex_proxy=False, ctx=None, budget=-1, reasoning_level="high", parallel=1, proxy_mode=None):
    """异步启动(加载 1-2 分钟,立即返回,前端轮询 /api/llm/current)。"""
    proxy_mode = normalize_proxy_mode(codex_proxy if proxy_mode is None else proxy_mode)
    codex_proxy = proxy_mode > 0
    m = _find(model_id)
    if not m:
        return {"ok": False, "error": f"未知语言模型: {model_id}"}
    if mtp and m.get("mtp_compatible") is False and not os.path.isfile(FAST_MTP_LLAMA_BIN):
        return {"ok": False, "error": "该模型需要官方 FastMTP 补丁版 llama.cpp，但项目内补丁版尚未编译；请先完成 FastMTP 初始化。",
                "mtp_compatible": False, "mtp_note": m.get("mtp_note", "")}
    if not m["exists"]:
        if m.get("backend") == "metal":
            st = m.get("metal_status") or metal_status(m)
            return {"ok": False, "error": "官方 Metal 尚未配置: " + ("、".join(st.get("missing") or []) or "请先初始化 Metal 环境"),
                    "metal_status": st}
        if m.get("backend") == "vmlx":
            st = m.get("vmlx_status") or vmlx_status(m)
            return {"ok": False, "error": "vMLX 模型未就绪: " + (m.get("missing_reason") or "请检查 .venv-vmlx 和模型目录"), "vmlx_status": st}
        return {"ok": False, "error": f"模型文件缺失: {m['gguf']}(检查 models/llm 软链)"}
    st = svc.svc_status("llm")
    cur = current()
    if st["running"] and cur.get("model", {}).get("id") == model_id and running_proxy_mode(cur.get("model")) == proxy_mode:
        _ensure_mac_status_item()
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
    is_metal = m.get("backend") == "metal"
    is_vmlx = m.get("backend") == "vmlx"
    binx = c.get("llm_bin", "llama-server")
    use_fastmtp_bin = bool(mtp and m.get("id") == "qwen3.8-27b-hauhaucs-aggressive-fastmtp" and os.path.isfile(FAST_MTP_LLAMA_BIN))
    if use_fastmtp_bin:
        binx = FAST_MTP_LLAMA_BIN
    if ctx is None:
        ctx = int(c.get("llm_ctx", 32768))
    ctx = max(8192, min(131072, int(ctx)))
    parallel = max(1, min(8, int(parallel)))
    pub = int(c.get("llm_port", VMLX_PORT_DEFAULT))
    backend_port = int(c.get("llm_backend_port", pub - 2))
    # Native vMLX is public by default. In tool-proxy mode it moves to the
    # backend port and vmlx_proxy owns the public endpoint.
    service_port = backend_port if (is_vmlx and codex_proxy) else pub
    args = (_metal_args(m, max_tokens, temp, ctx, reasoning_level, thinking)
            if is_metal else (_vmlx_args(m, thinking, temp, max_tokens, mtp, ctx, service_port, parallel)
                              if is_vmlx else build_args(m, thinking, temp, max_tokens, mtp, codex_proxy, ctx, budget, parallel)))
    try:
        logf = open(svc.log_path("llm"), "ab")
        command = args if (is_metal or is_vmlx) else [binx] + args
        cwd = _metal_paths(m)["repo"] if is_metal else BASE
        env = None
        if is_vmlx:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["VMLX_DISABLE_TQ_KV"] = "1"
            # Qwen3.6 hybrid (Mamba + attention) can OOM on a one-shot
            # prefill even around 10K tokens on a 32GB M1 Max. Force the
            # proven chunked path so external Responses clients cannot take
            # the high-peak one-shot route.
            env["VMLX_ALLOW_HYBRID_CHUNKED_PREFILL"] = "1"
            env["VMLINUX_ALLOW_HYBRID_CHUNKED_PREFILL"] = "1"
            env["VMLX_HYBRID_MIN_CHUNK"] = "256"
        elif use_fastmtp_bin:
            env = os.environ.copy()
            env["DYLD_LIBRARY_PATH"] = FAST_MTP_LLAMA_LIB + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
        p = subprocess.Popen(command, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        open(svc.pid_path("llm"), "w").write(str(p.pid))
        if is_vmlx:
            open(VMLX_PID_FILE, "w").write(str(p.pid))
        svc.keep_awake(p.pid)   # 语言模型服务期间禁止系统睡眠(锁屏/合盖接电源不断活)
    except FileNotFoundError:
        return {"ok": False, "error": ("找不到 Metal Python 环境，请先初始化 Metal" if is_metal
                                         else f"找不到 {binx}(需 brew install llama.cpp)")}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    save_pref(m["key"], thinking, temp, max_tokens, mtp, proxy_mode == 1, ctx, budget, reasoning_level, parallel, proxy_mode)
    if m.get("is_reasoning"):
        _sync_dsh_reasoning(reasoning_level)   # 同步思考档位给 DSH(尽力而为)
    proxy_force = ({"thinking": bool(thinking), "temperature": float(temp),
                    "max_tokens": int(max_tokens), "repetition_penalty": 1.1}
                   if (codex_proxy and is_vmlx) else None)
    proxy_pid = (_start_proxy(c, force_settings=proxy_force, entry="vmlx_proxy.py", proxy_mode=proxy_mode) if (codex_proxy and is_vmlx)
                 else (_start_proxy(c, proxy_mode=proxy_mode) if (codex_proxy and not is_metal and not is_vmlx) else None))
    try:
        try:
            logsz = os.path.getsize(svc.log_path("llm"))
        except Exception:
            logsz = 0
        json.dump({"id": m["id"], "name": m["name"], "backend": m.get("backend", "llama"),
                   "thinking": bool(thinking),
                   "temp": float(temp), "max_tokens": int(max_tokens),
                   "mtp": bool(mtp and m.get("has_mtp")),
                   "codex_proxy": bool(codex_proxy and proxy_pid),
                   "proxy_mode": int(proxy_mode if proxy_pid else 0),
                   "settings_proxy": bool(is_vmlx and proxy_pid),
                   "ctx": int(ctx), "parallel": int(parallel), "budget": int(budget),
                   "reasoning_level": str(reasoning_level),
                   "port": int(c.get("llm_port", 8848)), "pid": p.pid, "t0": time.time(),
                   "checkpoint": _metal_paths(m)["checkpoint"] if is_metal else "",
                   "log_size": logsz, "paused": False},
                  open(CUR_FILE, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    _ensure_mac_status_item()
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
    health_port = _health_port(info, st)
    ready = bool(st["port_up"] and _health_ready(health_port))
    return {"running": bool(st["alive_pid"] and ready), "loading": bool(st["alive_pid"] and not ready),
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
    # print_timing/release 表示最近一次请求已经结束;没有该边界时旧日志会被误报为生成中。
    done_at = max(txt.rfind("stop processing"), txt.rfind("total time ="), txt.rfind("all slots are idle"))
    if done_at >= task_at:
        return {"state": "idle", "tps": last_tps if 'last_tps' in locals() else None}
    tool_at = max(txt.rfind("tool call"), txt.rfind("function call"), txt.rfind("工具调用"))
    if tool_at >= task_at:
        return {"state": "tool"}
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
    health_port = _health_port(info, st)
    ready = bool(st["port_up"] and _health_ready(health_port))
    return {"running": bool(st["alive_pid"] and ready), "loading": bool(st["alive_pid"] and not ready),
            "paused": paused, "model": (info or None), "port": port,
            "api_url": f"http://127.0.0.1:{port}/v1/{'responses' if info.get('backend') == 'metal' else ''}".rstrip('/'),
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
    _stop_proxy()                       # Codex/vMLX 代理跟随暂停
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
                     float(info.get("temp", 0.7)), int(info.get("max_tokens", 16384)),
                     bool(info.get("mtp", False)), bool(info.get("codex_proxy", False)),
                     int(info.get("ctx", 32768)), int(info.get("budget", -1)),
                     str(info.get("reasoning_level", "high")), int(info.get("parallel", 1)),
                     int(info.get("proxy_mode", 1 if info.get("codex_proxy") else 0)))

def close_llm():
    """彻底关闭: 停进程 + 删掉状态记录(current_llm.json)。
    与"暂停"不同——暂停要保留记录供"恢复"原样拉起;关闭则清除。
    否则 stats() 会一直读到 paused=true,每次打开页面都被强制跳回"已暂停"页。"""
    r = svc.stop_svc("llm")
    _stop_proxy()                       # Codex/vMLX 代理跟随关闭
    try:
        os.remove(CUR_FILE)
    except OSError:
        pass
    return r
