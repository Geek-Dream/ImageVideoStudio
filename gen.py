#!/usr/bin/env python3
# ============================================================
# ImageVideoStudio · 生图/生视频小助手(网页版)
# 用法: python3 gen.py   然后浏览器打开 http://127.0.0.1:8860
# 依赖: 仅 Python3 标准库,无需 pip 安装任何东西
# 原理: 本程序只是一个"好看的操作台",真正画图的是 ComfyUI。
#        第一次用请先运行 ./install.sh 装好 ComfyUI,再 ./start.sh 启动。
# ============================================================
import json, os, re, shutil, sys, time, threading, urllib.request, webbrowser, subprocess, platform
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import svc  # 三类服务(生图/生视频/语言)统一启停与状态;svc 不回 import 本模块,无循环
import vidwf  # 视频 I2V 工作流(打 8850);同样不回 import 本模块
import llm   # 语言模型启动/参数记忆/GPU 上限;llm→svc,svc 仅函数内懒加载 llm,无环

BASE = os.path.dirname(os.path.abspath(__file__))          # 项目目录(gen.py 所在)
CONFIG_FILE = os.path.join(BASE, "config.json")            # 首跑自动生成的配置
MODELS_IMG = os.path.join(BASE, "models", "image")         # 图片模型放这里
MODELS_CN  = os.path.join(MODELS_IMG, "controlnet")        # 图片辅助模型(ControlNet)放这里
MODELS_VID = os.path.join(BASE, "models", "video")         # 视频模型放这里
OUT_DIR    = os.path.join(BASE, "output", "images")        # 成品图(网页可直接看)
COMFY_OUT  = os.path.join(BASE, "output", "comfy")         # ComfyUI 的临时输出(start.sh 指定)
MODELS_JSON= os.path.join(BASE, "models.json")             # 已知模型的友好名称/参数(可选)

IMG_EXTS = (".safetensors", ".ckpt", ".pt", ".pth")  # 仅单文件 checkpoint;GGUF 需多文件配套,本工具不支持
NEG_DEFAULT = "blurry, low quality, worst quality, watermark, text"

# ---------------- 配置读写 ----------------
def default_config():
    return {
        "comfy_dir": "",        # ComfyUI 安装目录;留空=自动找
        "img_port": 8849,       # 生图 ComfyUI 端口
        "vid_port": 8850,       # 生视频 ComfyUI 端口
        "port": 8860,           # 本网页端口
        "llm_port": 8848,       # 语言模型 llama-server 端口
        "llm_bin": "llama-server",  # llama-server 可执行文件名(PATH 里找)
        "llm_ctx": 32768,       # 语言模型上下文长度
    }

def load_config():
    cfg = default_config()
    if os.path.exists(CONFIG_FILE):
        try:
            cfg.update(json.load(open(CONFIG_FILE)))
        except Exception:
            pass
    if not cfg.get("comfy_dir"):
        cfg["comfy_dir"] = find_comfy()
    save_config(cfg)
    return cfg

def save_config(cfg):
    try:
        json.dump(cfg, open(CONFIG_FILE, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass

def find_comfy():
    """按优先级找 ComfyUI: 环境变量 -> 常见目录。找到返回路径,找不到返回空。"""
    env = os.environ.get("COMFYUI_DIR", "")
    cands = [env,
             os.path.join(BASE, "ComfyUI"),
             os.path.expanduser("~/ComfyUI"),
             os.path.expanduser("~/Desktop/ComfyUI"),
             os.path.expanduser("~/开发/video_ai/ComfyUI")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "main.py")):
            return c
    return ""

CFG = load_config()
IMG_PORT, VID_PORT, SELF_PORT = CFG["img_port"], CFG["vid_port"], CFG["port"]

# ---------------- 模型扫描 ----------------
DIFF_DIR = os.path.join(MODELS_IMG, "diffusion")   # GGUF 主模型(unet)放这里
ENC_DIR  = os.path.join(MODELS_IMG, "encoder")     # GGUF 文本编码器放这里
VAE_DIR  = os.path.join(MODELS_IMG, "vae")         # GGUF 的 VAE 放这里
LOCAL_JSON = os.path.join(BASE, "models.local.json")
NOTES_JSON = os.path.join(BASE, "model_notes.json")   # 模型说明书(关键字→擅长领域/速度/内存)

SAMPLER_DEFAULT = {"steps": 25, "cfg": 6.0, "sampler_name": "euler_ancestral", "scheduler": "normal"}

def registry():
    """合并 models.json(随项目发布) + models.local.json(本机私有,不上传)。
    返回 (friendly文件名→显示信息, gguf模型定义列表)。"""
    friendly, ggufs = {}, []
    for fp in (MODELS_JSON, LOCAL_JSON):
        if os.path.exists(fp):
            try:
                d = json.load(open(fp))
                friendly.update(d.get("models", {}))
                ggufs.extend(d.get("gguf_models", []))
            except Exception:
                pass
    return friendly, ggufs

def match_note(*texts):
    """按模型文件名/显示名(小写)匹配 model_notes.json 里的关键字,返回说明书条目(没有就 None)。
    以后拖新模型进来,只要在 model_notes.json 加一条对应关键字即可自动配上描述。"""
    if not os.path.exists(NOTES_JSON):
        return None
    try:
        notes = json.load(open(NOTES_JSON)).get("notes", [])
    except Exception:
        return None
    hay = " ".join(t.lower() for t in texts if t)
    for n in notes:
        if any(k.lower() in hay for k in n.get("keys", [])):
            return n
    return None

def gguf_ok(e):
    """GGUF 模型三件套(unet/编码器/vae)都在才算可用。"""
    try:
        return (os.path.exists(os.path.join(DIFF_DIR, e["unet"])) and
                os.path.exists(os.path.join(ENC_DIR, e["clip"])) and
                os.path.exists(os.path.join(VAE_DIR, e["vae"])))
    except Exception:
        return False

def list_models():
    """可选模型 = models/image 里的单文件 checkpoint + 三件套齐全的 GGUF 模型。"""
    friendly, ggufs = registry()
    out = []
    if os.path.isdir(MODELS_IMG):
        for fn in sorted(os.listdir(MODELS_IMG)):
            fp = os.path.join(MODELS_IMG, fn)
            if os.path.isfile(fp) and fn.lower().endswith(IMG_EXTS):
                meta = friendly.get(fn, {})
                m = {"id": fn, "kind": "checkpoint",
                     "name": meta.get("name", fn),
                     "sec": int(meta.get("sec", 60)),
                     "desc": meta.get("desc", "单文件模型(checkpoint)")}
                note = match_note(fn, m["name"])
                if note: m.update({"field": note.get("field"), "speed": note.get("speed"),
                                   "time": note.get("time"), "mem": note.get("mem"),
                                   "detail": note.get("detail")})
                out.append(m)
    for e in ggufs:
        if gguf_ok(e):
            m = {"id": e["id"], "kind": "gguf",
                 "name": e.get("name", e["id"]),
                 "sec": int(e.get("sec", 120)),
                 "desc": e.get("desc", "GGUF 模型")}
            note = match_note(e["id"], m["name"], e.get("unet", ""))
            if note: m.update({"field": note.get("field"), "speed": note.get("speed"),
                               "time": note.get("time"), "mem": note.get("mem"),
                               "detail": note.get("detail")})
            out.append(m)
    return out

def find_entry(model_id):
    """按 id 找到完整模型定义(生成时用)。"""
    fp = os.path.join(MODELS_IMG, model_id)
    if os.path.isfile(fp) and model_id.lower().endswith(IMG_EXTS):
        return {"id": model_id, "kind": "checkpoint", "sampler": SAMPLER_DEFAULT}
    _, ggufs = registry()
    for e in ggufs:
        if e.get("id") == model_id and gguf_ok(e):
            return {**e, "kind": "gguf"}
    return None

def has_controlnet():
    if not os.path.isdir(MODELS_CN):
        return False
    return any(f.lower().endswith(IMG_EXTS) for f in os.listdir(MODELS_CN))

def first_controlnet():
    for f in sorted(os.listdir(MODELS_CN)):
        if f.lower().endswith(IMG_EXTS):
            return f
    return ""

# ---------------- ComfyUI 工作流(checkpoint / GGUF 双架构) ----------------
def _enc_nodes(e):
    """按模型架构返回 (加载器节点dict, clip接线, vae接线, model接线)。"""
    if e["kind"] == "gguf":
        # 用大编号 100/101/102,避免与 build_wf 里 5~11 的功能节点撞号
        nodes = {"100": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": e["unet"]}},
                 "101": {"class_type": e.get("clip_loader", "CLIPLoaderGGUF"),
                         "inputs": {"clip_name": e["clip"], "type": e["clip_type"]}},
                 "102": {"class_type": "VAELoader", "inputs": {"vae_name": e["vae"]}}}
        model_src = ["100", 0]
        if e.get("lora"):  # 模型挂了 LoRA 补丁(如 flux 的 NSFW 补丁),串在 unet 后面
            nodes["103"] = {"class_type": "LoraLoaderModelOnly",
                            "inputs": {"lora_name": e["lora"],
                                       "strength_model": e.get("lora_strength", 1.0),
                                       "model": ["100", 0]}}
            model_src = ["103", 0]
        return (nodes, ["101", 0], ["102", 0], model_src)
    return ({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": e["id"]}}},
            ["1", 1], ["1", 2], ["1", 0])

def build_wf(e, pos, neg, w, h, seed, mode="t2i", ref=None, mask=None,
             strength=0.6, scale=2.0, ctype="openpose"):
    """e 是 find_entry() 返回的模型定义。五种玩法: t2i/i2i/inpaint/upscale/pose。"""
    loaders, clip_src, vae_src, model_src = _enc_nodes(e)
    cfg = e.get("sampler", SAMPLER_DEFAULT)
    wf = dict(loaders)
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": clip_src}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": clip_src}}
    pos_out = ["3", 0]
    if e.get("flux_guidance"):  # flux 系正向要过 FluxGuidance
        wf["3g"] = {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["3", 0]}}
        pos_out = ["3g", 0]

    if mode == "inpaint":  # 局部重绘: 只重画涂抹区域
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
        wf["8"] = {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["6", 0], "vae": vae_src, "mask": ["7", 0], "grow_mask_by": 6}}
        wf["9"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["8", 0]}}
        wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae_src}}
        wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["10", 0]}}
        return wf

    if mode == "upscale":  # 放大: 先拉伸再低幅度重绘补细节
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["6", 0]}}
        wf["8"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": vae_src}}
        wf["9"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": strength, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["8", 0]}}
        wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae_src}}
        wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["10", 0]}}
        return wf

    if mode == "pose":  # 姿势控制: 仅单文件 checkpoint(SDXL 系配 union controlnet)
        if e["kind"] != "checkpoint":
            raise ValueError("姿势控制只支持单文件模型(如 wai / SDXL 系)")
        cn = first_controlnet()
        if not cn:
            raise ValueError("未找到 ControlNet 辅助模型,请先放入 models/image/controlnet/")
        wf["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn}}
        wf["8"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["7", 0], "type": ctype}}
        wf["9"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "control_net": ["8", 0], "image": ["6", 0], "strength": strength, "start_percent": 0.0, "end_percent": 1.0}}
        wf["10"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": model_src, "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["5", 0]}}
        wf["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": vae_src}}
        wf["12"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["11", 0]}}
        return wf

    if mode == "i2i" and ref:  # 以图生图: 参考图编码进 latent,按 strength 重绘
        wf["5"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": vae_src}}
        wf["7"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": strength, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["6", 0]}}
        wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": vae_src}}
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["8", 0]}}
        return wf

    # t2i 普通文生图
    wf["5"] = {"class_type": e.get("latent", "EmptyLatentImage"), "inputs": {"width": w, "height": h, "batch_size": 1}}
    wf["6"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["5", 0]}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae_src}}
    wf["8"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["7", 0]}}
    return wf

# ---------------- ComfyUI 通信 ----------------
def port_running(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2)
        return True
    except Exception:
        return False

def comfy_get(path, timeout=10):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{IMG_PORT}{path}", timeout=timeout))

TASKS = {}
LOCK = threading.Lock()

def submit(model, pos, neg, w, h, name, mode="t2i", ref=None, mask=None, strength=0.6, scale=2.0, ctype="openpose"):
    import random
    e = find_entry(model)
    if not e:
        raise ValueError(f"模型不可用: {model}(文件缺失,检查 models/ 目录)")
    seed = random.randint(0, 2**31 - 1)
    wf = build_wf(e, pos, neg, w, h, seed, mode, ref, mask, strength, scale, ctype)
    req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    with LOCK:
        TASKS[pid] = {"name": name, "model": model, "t0": time.time(), "done": False, "error": "", "url": ""}
    threading.Thread(target=wait_done, args=(pid,), daemon=True).start()
    return pid

def wait_done(pid):
    deadline = time.time() + 1800
    while time.time() < deadline:
        time.sleep(3)
        try:
            hist = comfy_get(f"/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            continue
        st = hist[pid].get("status", {})
        if st.get("status_str") == "error":
            with LOCK: TASKS[pid]["error"] = "生成失败:多半是缺模型/缺辅助模型,或提示词有问题"
            return
        if st.get("completed"):
            fname = None
            for out in hist[pid].get("outputs", {}).values():
                for img in out.get("images", []):
                    fname = os.path.join(COMFY_OUT, img.get("subfolder", ""), img["filename"])
            if fname and os.path.exists(fname):
                os.makedirs(OUT_DIR, exist_ok=True)
                dest = os.path.join(OUT_DIR, TASKS[pid]["name"] + ".png")
                shutil.copy2(fname, dest)
                try: os.remove(fname)
                except OSError: pass
                with LOCK:
                    TASKS[pid]["done"] = True
                    TASKS[pid]["url"] = "/out/" + TASKS[pid]["name"] + ".png"
            else:
                with LOCK: TASKS[pid]["error"] = "找不到输出文件"
            return
    with LOCK: TASKS[pid]["error"] = "超时(30分钟)"

# ---------------- 预定批量(quality_test.py 的网页版) ----------------
# 逻辑同 quality_test.py: 逐模型逐提示词提交→轮询等真完成(不是入队就算完)
# →归档 output/organized/<模型名>/。全程后台线程跑,网页轮询 /api/batch/status。
BATCH = {"running": False, "done": 0, "total": 0, "ok": 0, "fail": 0,
         "current": "", "log": [], "stop": False, "started": 0.0, "finished": False}
ORG_DIR = os.path.join(BASE, "output", "organized")
PROMPT_DIR = os.path.join(BASE, "prompts")

def _safe(s):
    return re.sub(r"[^\w.-]+", "_", s)

def batch_config():
    """批量配置: 全部可用模型 + 公共负向。提示词改成「模型→风格→提示词组」结构,
    网页端自己组织(每个模型一张大卡,卡内多个风格子卡),不再从 prompts/ 预填。"""
    neg = NEG_DEFAULT
    negp = os.path.join(PROMPT_DIR, "_negative.txt")
    if os.path.exists(negp):
        neg = open(negp, encoding="utf-8").read().strip()
    return {"models": list_models(), "neg": neg,
            "org_dir": "output/organized/<模型>/<风格>/"}

def batch_import(folder):
    """从一个文件夹导入风格: 每个 .txt = 一个风格(文件名=风格名),文件内空行分隔多条提示词。
    _ 开头的文件忽略(如 _negative.txt)。返回 [{name, prompts:[...]}]。"""
    if not os.path.isdir(folder):
        return {"error": f"文件夹不存在: {folder}"}
    styles = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt") or fn.startswith("_"):
            continue
        txt = open(os.path.join(folder, fn), encoding="utf-8").read()
        prompts = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        if prompts:
            styles.append({"name": fn[:-4], "prompts": prompts})
    if not styles:
        return {"error": f"{folder} 里没读到任何提示词(.txt,空行分隔多条)"}
    return {"styles": styles}

def _blog(msg):
    BATCH["log"].append(msg)
    BATCH["log"] = BATCH["log"][-120:]

def batch_start(models, neg, w, h):
    """models = [{id,name,styles:[{name,prompts:[...],ref}]},...](网页已按模型独立组织好)。"""
    if BATCH["running"]:
        return {"error": "已有批量任务在跑(先停止或等它结束)"}
    total = sum(len(s["prompts"]) for m in models for s in m["styles"])
    if not models or total == 0:
        return {"error": "至少留一个模型和一条提示词"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    BATCH.update(running=True, done=0, total=total, ok=0, fail=0,
                 current="", log=[], stop=False, started=time.time(), finished=False)
    threading.Thread(target=_batch_run, args=(models, neg, w, h), daemon=True).start()
    return {"ok": True}

def _batch_run(models, neg, w, h):
    """逐模型→逐风格→逐提示词提交→真等完成→归档 output/organized/<模型>/<风格>/图片N.png。
    风格带 ref 就走 i2i 0.75 垫图,否则纯文字 t2i。"""
    for m in models:
        mid, mname = m["id"], m["name"]
        for s in m["styles"]:
            sname, ref = s["name"], (s.get("ref") or None)
            ddir = os.path.join(ORG_DIR, _safe(mname), _safe(sname))
            for idx, ptext in enumerate(s["prompts"], 1):
                if BATCH["stop"]:
                    break
                BATCH["current"] = f"{mname} | {sname} | 图{idx}"
                _blog(f"[{BATCH['done']+1}/{BATCH['total']}] {mname} | {sname} | 图{idx}")
                name = _safe(f"{mid}_{sname}_{idx}_{int(time.time())}")
                try:
                    pid = submit(mid, ptext, neg, w, h, name,
                                 "i2i" if ref else "t2i", ref, None, 0.75)
                except Exception as e:
                    BATCH["fail"] += 1; BATCH["done"] += 1
                    _blog(f"  ✗ 提交失败: {e}")
                    continue
                t0 = time.time(); ok = False; err = ""
                while time.time() - t0 < 1800 and not BATCH["stop"]:
                    t = TASKS.get(pid, {})
                    if t.get("done"): ok = True; break
                    if t.get("error"): err = t["error"]; break
                    time.sleep(3)
                if ok:
                    src = os.path.join(OUT_DIR, name + ".png")
                    os.makedirs(ddir, exist_ok=True)
                    shutil.copy2(src, os.path.join(ddir, f"图片{idx}.png"))
                    BATCH["ok"] += 1
                    _blog(f"  ✓ 完成({int(time.time()-t0)}秒)")
                else:
                    BATCH["fail"] += 1
                    _blog(f"  ✗ {'已停止' if BATCH['stop'] else (err or '超时')}")
                BATCH["done"] += 1
    if BATCH["stop"]:
        _blog("⏹ 用户停止")
    BATCH.update(running=False, finished=True, current="")
    _blog(f"🏁 结束: 成功 {BATCH['ok']}/{BATCH['total']},产物在 output/organized/<模型>/<风格>/")

# ---------------- 漫画连载(校园热血番·comic_gen 网页版) ----------------
COMIC = {"running": False, "done": 0, "total": 0, "ok": 0, "fail": 0,
         "current": "", "log": [], "stop": False, "finished": False, "last": ""}
STORY_FILE = os.path.join(BASE, "my_story", "story.txt")
COMIC_OUT = os.path.join(BASE, "output", "my_story_comic")

def comic_story():
    """解析 my_story/story.txt → 分镜列表(文件里几格就是几格,不自动补齐)。"""
    panels = []
    if os.path.exists(STORY_FILE):
        text = open(STORY_FILE, encoding="utf-8").read()
        pat = (r"### Image (\d+)"
               r"\s*\nPrompt:\s*(.*?)"
               r"\s*\n\s*Dialogue:\s*(.*?)"
               r"(?=\n\n|\n### Image |$)")
        for n, p, d in re.findall(pat, text, re.DOTALL):
            panels.append({"num": int(n), "prompt": p.strip(), "dialogue": d.strip()})
        panels.sort(key=lambda x: x["num"])
    return {"panels": panels, "models": list_models(), "out_dir": "output/my_story_comic/",
            "story_file": "my_story/story.txt"}

def _clog(msg):
    COMIC["log"].append(msg)
    COMIC["log"] = COMIC["log"][-120:]

def _add_dialogue(image_path, dialogue, out_path):
    """底部叠加台词气泡。中文必须用 PingFang 等中文字体(Helvetica 会出豆腐块)。"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    h = img.size[1]
    bh = min(h * 0.18, 180)
    bx, by = 30, h - bh - 10
    bw = img.size[0] - 60
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=15,
                           fill=(20, 15, 30, 180), outline=(255, 255, 245, 255), width=2)
    font = None
    for fp in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
               "/System/Library/Fonts/STHeiti Medium.ttc",
               "/System/Library/Fonts/Supplemental/Songti.ttc"):
        try:
            font = ImageFont.truetype(fp, 22)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    lines, cur = [], ""   # 中文按字断行,兼容英文按词
    for ch in dialogue:
        test = cur + ch
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] > bw - 40 and cur:
            lines.append(cur); cur = ch
        else:
            cur = test
    if cur:
        lines.append(cur)
    tx = bx + 20
    ty = by + (bh - len(lines) * 26) / 2
    for ln in lines:
        draw.text((tx, ty), ln, fill=(255, 255, 245, 255), font=font)
        ty += 26
    img.save(out_path, "PNG")

def comic_start(model, panels, ref, strength, w, h):
    if COMIC["running"]:
        return {"error": "已有连载任务在跑(先停止或等它结束)"}
    if not panels:
        return {"error": "至少留一格分镜"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    COMIC.update(running=True, done=0, total=len(panels), ok=0, fail=0,
                 current="", log=[], stop=False, finished=False, last="")
    threading.Thread(target=_comic_run, args=(model, panels, ref, strength, w, h), daemon=True).start()
    return {"ok": True}

def _comic_run(model, panels, ref, strength, w, h):
    os.makedirs(COMIC_OUT, exist_ok=True)
    try:
        import PIL  # noqa: F401
        can_text = True
    except Exception:
        can_text = False
        _clog("⚠ Pillow 未安装(pip3 install Pillow),本批跳过台词叠加")
    for p in panels:
        if COMIC["stop"]:
            break
        num = p["num"]
        COMIC["current"] = f"第 {num:03d} 格"
        _clog(f"[{COMIC['done']+1}/{COMIC['total']}] 第 {num:03d} 格 | {p['dialogue'][:40]}")
        name = _safe(f"comic_{num:03d}_{int(time.time())}")
        try:
            pid = submit(model, p["prompt"], NEG_DEFAULT, w, h, name,
                         "i2i" if ref else "t2i", ref, None, strength)
        except Exception as e:
            COMIC["fail"] += 1; COMIC["done"] += 1
            _clog(f"  ✗ 提交失败: {e}")
            continue
        t0 = time.time(); ok = False; err = ""
        while time.time() - t0 < 1800 and not COMIC["stop"]:
            t = TASKS.get(pid, {})
            if t.get("done"): ok = True; break
            if t.get("error"): err = t["error"]; break
            time.sleep(3)
        if ok:
            src = os.path.join(OUT_DIR, name + ".png")
            dst = os.path.join(COMIC_OUT, f"{num:03d}.png")
            shutil.copy2(src, dst)
            if can_text and p["dialogue"] and p["dialogue"] != "(no dialogue)":
                try:
                    _add_dialogue(dst, p["dialogue"], dst)
                except Exception as e:
                    _clog(f"  ⚠ 台词叠加失败(图已存): {e}")
            COMIC["ok"] += 1; COMIC["last"] = f"{num:03d}.png"
            _clog(f"  ✓ 完成({int(time.time()-t0)}秒)")
        else:
            COMIC["fail"] += 1
            _clog(f"  ✗ {'已停止' if COMIC['stop'] else (err or '超时')}")
        COMIC["done"] += 1
    if COMIC["stop"]:
        _clog("⏹ 用户停止")
    COMIC.update(running=False, finished=True, current="")
    _clog(f"🏁 结束: 成功 {COMIC['ok']}/{COMIC['total']},产物在 output/my_story_comic/")

# ---------------- 自定义连载(角色设定→分镜画布→生成) ----------------
# 与模板连载(story.txt)并行的一套:用户先在网页钉死每个主角的「设定图」并逐张审核,
# 再搭任意张分镜画布(左场景/右逐句台词/穿搭/尺寸风格可继承上一张),最后逐格真等画完。
# 每格用「选定主角的设定图」做 i2i 参考垫图,其余角色靠文字锚点;台词每句一个独立气泡。
CC = {"running": False, "done": 0, "total": 0, "ok": 0, "fail": 0,
      "current": "", "log": [], "stop": False, "finished": False, "last": "",
      "sheet": {"running": False, "done": False, "error": "", "url": "", "name": ""}}
CC_PROJ = os.path.join(BASE, "my_story", "custom_project.json")
CC_OUT = os.path.join(BASE, "output", "custom_comic")

def _cc_model():
    """自定义连载固定用动漫最强的 wai v17(风格统一),没有就退到第一个可用模型。"""
    ms = list_models()
    for m in ms:
        if "waiIllustriousSDXL_v170" in m["id"]:
            return m["id"]
    return ms[0]["id"] if ms else ""

def cc_project():
    """读自定义项目(角色+分镜)。没有就返回空骨架。"""
    if os.path.exists(CC_PROJ):
        try:
            d = json.load(open(CC_PROJ, encoding="utf-8"))
            return {"chars": d.get("chars", []), "panels": d.get("panels", [])}
        except Exception:
            pass
    return {"chars": [], "panels": []}

def cc_save(chars, panels):
    os.makedirs(os.path.dirname(CC_PROJ), exist_ok=True)
    json.dump({"chars": chars, "panels": panels},
              open(CC_PROJ, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"ok": True}

def _cc_sheet_prompt(desc, role):
    """角色设定图(Character Design Sheet): 立绘+多角度+穿搭拆解,方便当后续垫图。"""
    who = "1boy" if role == "male" else "1girl"
    return (f"masterpiece, best quality, character design sheet, {who}, {desc}, "
            f"full body, multiple views, front view, side view, outfit breakdown, "
            f"simple background, reference sheet, anime style")

def cc_sheet_gen(role, desc, ref):
    """生成一张角色设定图。有照片 ref → i2i 0.6 重绘成动漫;否则纯文字 t2i。"""
    if CC["sheet"]["running"]:
        return {"error": "上一张设定图还在画,等它结束"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    if not desc.strip() and not ref:
        return {"error": "先写点外貌服装特征,或上传照片"}
    CC["sheet"] = {"running": True, "done": False, "error": "", "url": "", "name": ""}
    threading.Thread(target=_cc_sheet_run, args=(role, desc, ref), daemon=True).start()
    return {"ok": True}

def _cc_sheet_run(role, desc, ref):
    name = _safe(f"cc_sheet_{int(time.time())}")
    try:
        pid = submit(_cc_model(), _cc_sheet_prompt(desc, role), NEG_DEFAULT, 832, 1216,
                     name, "i2i" if ref else "t2i", ref, None, 0.6)
    except Exception as e:
        CC["sheet"].update(running=False, error=str(e))
        return
    t0 = time.time()
    while time.time() - t0 < 1800:
        t = TASKS.get(pid, {})
        if t.get("done"):
            CC["sheet"].update(running=False, done=True, name=name,
                               url="/out/" + name + ".png")
            return
        if t.get("error"):
            CC["sheet"].update(running=False, error=t["error"])
            return
        time.sleep(3)
    CC["sheet"].update(running=False, error="超时(30分钟)")

def cc_start(chars, panels):
    if CC["running"]:
        return {"error": "已有自定义连载在跑(先停止或等它结束)"}
    if not panels:
        return {"error": "至少搭一格分镜画布"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    CC.update(running=True, done=0, total=len(panels), ok=0, fail=0,
              current="", log=[], stop=False, finished=False, last="")
    threading.Thread(target=_cc_run, args=(chars, panels), daemon=True).start()
    return {"ok": True}

def _cclog(msg):
    CC["log"].append(msg)
    CC["log"] = CC["log"][-120:]

def _cc_panel_prompt(chars, p):
    """拼单格提示词: 风格 + 场景/站位/剧情 + 当格穿搭 + 各角色文字锚点。"""
    parts = []
    if p.get("style"):
        parts.append(p["style"])
    if p.get("scene"):
        parts.append(p["scene"])
    if p.get("outfit"):
        parts.append(p["outfit"])
    for c in chars:
        if c.get("anchor"):
            parts.append(c["anchor"])
    parts.append("masterpiece, best quality, anime style, comic panel")
    return ", ".join(x.strip().strip(",") for x in parts if x and x.strip())

def _cc_ref_upload(local_png):
    """把 OUT_DIR 里的设定图上传给 ComfyUI 当 i2i 参考,返回服务器文件名。"""
    fp = os.path.join(OUT_DIR, local_png)
    if not os.path.exists(fp):
        return None
    boundary = "----ccupload"
    with open(fp, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{local_png}\"\r\nContent-Type: image/png\r\n\r\n").encode() + data + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n"
            f"true\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read()).get("name")
    except Exception:
        return None

def _add_bubbles(image_path, dialogues, out_path):
    """每句台词一个独立气泡,从底部往上堆叠,带「角色名:」前缀。dialogues=[{who,text}]"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    font = None
    for fp in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
               "/System/Library/Fonts/STHeiti Medium.ttc",
               "/System/Library/Fonts/Supplemental/Songti.ttc"):
        try:
            font = ImageFont.truetype(fp, 22)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    # 自底向上逐句叠: 每句先按宽度断行,算好气泡高,再画
    y = H - 14
    margin, bw = 24, W - 48
    for d in reversed(dialogues):
        label = (d["who"] + ":" if d.get("who") else "") + d["text"]
        lines, cur = [], ""
        for ch in label:
            test = cur + ch
            bb = font.getbbox(test)
            if (bb[2] - bb[0]) > bw - 36 and cur:
                lines.append(cur); cur = ch
            else:
                cur = test
        if cur:
            lines.append(cur)
        bh = len(lines) * 27 + 18
        y -= bh
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle([margin, y, margin + bw, y + bh], radius=13,
                             fill=(20, 15, 30, 185), outline=(255, 255, 245, 255), width=2)
        img = Image.alpha_composite(img, overlay)
        od2 = ImageDraw.Draw(img)
        ty = y + 9
        for ln in lines:
            od2.text((margin + 18, ty), ln, fill=(255, 255, 245, 255), font=font)
            ty += 27
        y -= 10  # 气泡之间的缝
    img.convert("RGB").save(out_path, "PNG")

def _cc_run(chars, panels):
    os.makedirs(CC_OUT, exist_ok=True)
    try:
        import PIL  # noqa: F401
        can_text = True
    except Exception:
        can_text = False
        _cclog("⚠ Pillow 未安装,本批跳过台词气泡")
    model = _cc_model()
    ref_cache = {}  # 设定图本地名→ComfyUI 名,同一张设定图只上传一次
    for p in panels:
        if CC["stop"]:
            break
        num = p["num"]
        CC["current"] = f"第 {num:03d} 格"
        _cclog(f"[{CC['done']+1}/{CC['total']}] 第 {num:03d} 格")
        # 垫图: 本格选定的主角设定图(有才垫,没有就纯文字)
        ref = None
        ridx = p.get("ref_char")
        if ridx is not None and 0 <= int(ridx) < len(chars):
            sheet = chars[int(ridx)].get("sheet", "")
            if sheet:
                if sheet not in ref_cache:
                    ref_cache[sheet] = _cc_ref_upload(sheet)
                ref = ref_cache[sheet]
        name = _safe(f"cc_{num:03d}_{int(time.time())}")
        try:
            pid = submit(model, _cc_panel_prompt(chars, p), NEG_DEFAULT,
                         int(p.get("w", 1024)), int(p.get("h", 720)), name,
                         "i2i" if ref else "t2i", ref, None, 0.6)
        except Exception as e:
            CC["fail"] += 1; CC["done"] += 1
            _cclog(f"  ✗ 提交失败: {e}")
            continue
        t0 = time.time(); ok = False; err = ""
        while time.time() - t0 < 1800 and not CC["stop"]:
            t = TASKS.get(pid, {})
            if t.get("done"): ok = True; break
            if t.get("error"): err = t["error"]; break
            time.sleep(3)
        if ok:
            src = os.path.join(OUT_DIR, name + ".png")
            dst = os.path.join(CC_OUT, f"{num:03d}.png")
            shutil.copy2(src, dst)
            dlg = [{"who": d.get("who", ""), "text": d.get("text", "")}
                   for d in p.get("dialogues", []) if d.get("text", "").strip()]
            if can_text and dlg:
                try:
                    _add_bubbles(dst, dlg, dst)
                except Exception as e:
                    _cclog(f"  ⚠ 气泡叠加失败(图已存): {e}")
            CC["ok"] += 1; CC["last"] = f"{num:03d}.png"
            _cclog(f"  ✓ 完成({int(time.time()-t0)}秒)")
        else:
            CC["fail"] += 1
            _cclog(f"  ✗ {'已停止' if CC['stop'] else (err or '超时')}")
        CC["done"] += 1
    if CC["stop"]:
        _cclog("⏹ 用户停止")
    CC.update(running=False, finished=True, current="")
    _cclog(f"🏁 结束: 成功 {CC['ok']}/{CC['total']},产物在 output/custom_comic/")

# ---------------- 网页 ----------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>ImageVideoStudio · 生图/生视频小助手</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
/* ============ 主题变量:浅色(默认)/ 深色(body.dark) ============ */
:root{
  --bg:radial-gradient(1200px 800px at 85% -10%,#e7ecff 0%,transparent 55%),
       radial-gradient(1000px 700px at -10% 110%,#ffeef6 0%,transparent 50%),
       linear-gradient(160deg,#f4f6fb 0%,#f6f4fa 50%,#f2f7f5 100%);
  --card:#ffffff; --card-sel:#f2f5ff; --border:#e6e8f0; --border-hi:#5b8cff;
  --text:#1d2230; --muted:#5a6274; --faint:#9aa1b0;
  --input:#fbfcfe; --accent:#4f7cff; --accent2:#8a5cff;
  --grad:linear-gradient(120deg,#4f7cff,#8a5cff);
  --btn-grad:linear-gradient(120deg,#4f7cff 0%,#6a5cff 55%,#8a5cff 100%);
  --shadow-sm:0 1px 2px rgba(50,60,120,.06),0 1px 8px rgba(50,60,120,.05);
  --shadow-md:0 8px 28px rgba(70,90,200,.14),0 2px 6px rgba(70,90,200,.08);
  --chip-bg:#eef2ff; --chip-bd:#d6defa; --chip-tx:#3b5bdb;
  --box-bg:#fffaf0; --box-bd:#ffe3a6; --box-tx:#5a4a1e;
  --mem-bg:#fff7e8; --mem-bd:#f0e0b8; --mem-tx:#8a5a00;
  --on:#16a34a; --off:#a0a6b3;
  --glass:rgba(255,255,255,.72);
}
body.dark{
  --bg:radial-gradient(1200px 800px at 85% -10%,#1d2b4d 0%,transparent 55%),
       radial-gradient(1000px 700px at -10% 110%,#2a1530 0%,transparent 50%),
       linear-gradient(160deg,#0d1017 0%,#12151d 50%,#0f1a14 100%);
  --card:#1a1e28; --card-sel:#233052; --border:#2b3040; --border-hi:#6f9bff;
  --text:#e7eaf2; --muted:#a6aebf; --faint:#7d8494;
  --input:#20242f; --accent:#6f9bff; --accent2:#a78bff;
  --shadow-sm:0 1px 2px rgba(0,0,0,.5),0 1px 10px rgba(0,0,0,.35);
  --shadow-md:0 10px 34px rgba(0,0,0,.55),0 3px 10px rgba(80,110,230,.18);
  --chip-bg:#262c40; --chip-bd:#3a4560; --chip-tx:#93b4ff;
  --box-bg:#241f10; --box-bd:#4a3f1f; --box-tx:#e8d9a8;
  --mem-bg:#241f10; --mem-bd:#4a3f1f; --mem-tx:#d9b25f;
  --on:#4ade80; --off:#6b7280;
  --glass:rgba(22,26,36,.72);
}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{font-family:-apple-system,'PingFang SC','SF Pro Text',sans-serif;max-width:900px;margin:0 auto;
  padding:34px 18px 70px;background:var(--bg);background-attachment:fixed;color:var(--text);min-height:100vh;
  transition:background .35s,color .35s}
h1{font-size:27px;text-align:center;margin:4px 0 6px;letter-spacing:.6px;font-weight:800;
  background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
h2{font-size:15.5px;margin:28px 0 13px;color:var(--muted);display:flex;align-items:center;gap:9px;
  font-weight:700;letter-spacing:.3px;text-transform:none}
h2::before{content:'';width:4px;height:16px;border-radius:3px;background:var(--grad);box-shadow:0 0 8px rgba(111,140,255,.5)}
.card{background:var(--glass);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid var(--border);border-radius:16px;padding:17px 19px;margin:13px 0;cursor:pointer;
  box-shadow:var(--shadow-sm);transition:transform .18s cubic-bezier(.2,.7,.3,1.2),box-shadow .18s,border-color .18s}
.card:hover{border-color:var(--border-hi);box-shadow:var(--shadow-md);transform:translateY(-3px)}
.card.sel{border-color:var(--border-hi);background:var(--card-sel);box-shadow:var(--shadow-md)}
.card.dis{opacity:.5;cursor:not-allowed}
.card.dis:hover{transform:none;box-shadow:var(--shadow-sm)}
.dot{font-size:12px;font-weight:600}.on{color:var(--on)}.off{color:var(--off)}
button{background:var(--btn-grad);color:#fff;border:0;border-radius:12px;padding:12px 28px;
  font-size:15px;font-weight:600;letter-spacing:.2px;cursor:pointer;
  box-shadow:0 4px 14px rgba(90,110,255,.32),inset 0 1px 0 rgba(255,255,255,.25);
  transition:filter .15s,transform .12s,box-shadow .15s}
button:hover{filter:brightness(1.07);box-shadow:0 6px 20px rgba(90,110,255,.4),inset 0 1px 0 rgba(255,255,255,.25)}
button:active{transform:scale(.96)}
button:disabled{background:#a8adba;box-shadow:none;cursor:not-allowed}
.back{background:var(--card);color:var(--muted);box-shadow:var(--shadow-sm);border:1px solid var(--border)}
.back:hover{background:var(--card-sel);color:var(--text);box-shadow:var(--shadow-md)}
textarea,input[type=text],input[type=number]{width:100%;box-sizing:border-box;border:1px solid var(--border);
  border-radius:10px;padding:10px 12px;font-size:14px;font-family:inherit;background:var(--input);color:var(--text);
  transition:border-color .15s,box-shadow .15s}
textarea:focus,input:focus,select:focus{outline:0;border-color:var(--border-hi);box-shadow:0 0 0 4px rgba(95,135,255,.18)}
textarea{height:58px;resize:vertical;line-height:1.5}
select{border:1px solid var(--border);border-radius:10px;padding:8px 11px;font-size:14px;background:var(--input);
  color:var(--text);cursor:pointer}
label{font-size:13px;color:var(--muted);display:block;margin:12px 0 5px;font-weight:600}
.bar{height:9px;background:var(--border);border-radius:999px;overflow:hidden;margin-top:11px}
.bar>i{display:block;height:100%;background:var(--grad);width:0;transition:width .5s;border-radius:999px;
  box-shadow:0 0 8px rgba(111,140,255,.6)}
img.out{max-width:100%;border-radius:14px;margin-top:13px;box-shadow:var(--shadow-md)}
img.mask{filter:blur(18px)}
.row{display:flex;gap:11px;align-items:center;flex-wrap:wrap}
.small{font-size:12.5px;color:var(--faint);line-height:1.55}
.drop{border:2px dashed var(--border);border-radius:14px;padding:20px;text-align:center;color:var(--faint);
  font-size:13px;cursor:pointer;background:var(--input);transition:.18s}
.drop:hover{border-color:var(--border-hi);color:var(--accent)}
.drop.over{border-color:var(--border-hi);background:var(--card-sel);color:var(--accent);border-style:solid;transform:scale(1.01)}
.drop img{max-width:100%;max-height:150px;border-radius:9px;margin-top:10px}
.box{background:var(--box-bg);border:1px solid var(--box-bd);color:var(--box-tx);border-radius:14px;
  padding:16px 18px;margin:13px 0;line-height:1.75;box-shadow:var(--shadow-sm)}
code{background:var(--chip-bg);border-radius:6px;padding:2px 8px;font-size:13px;color:var(--chip-tx);
  border:1px solid var(--chip-bd)}
input[type=range]{accent-color:var(--accent);height:22px}
input[type=radio],input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 6px}
.tag{padding:4px 12px;border-radius:999px;background:var(--chip-bg);border:1px solid var(--chip-bd);
  font-size:12px;cursor:pointer;user-select:none;color:var(--chip-tx);transition:.15s}
.tag:hover{background:var(--card-sel);transform:translateY(-1px)}
.tag.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tag.neg.on{background:#e0556b;border-color:#e0556b}
.delcard{float:right;color:#e0556b;cursor:pointer;font-weight:700;padding:0 6px;font-size:16px;line-height:1;transition:.15s}
.delcard:hover{color:#ff2d55;transform:scale(1.2)}
.szbox{display:inline-flex;align-items:center;justify-content:center;border:1.5px solid var(--border-hi);border-radius:5px;
  background:var(--card-sel);color:var(--accent);font-size:10px;margin-left:10px;vertical-align:middle;overflow:hidden;
  text-align:center;line-height:1.1}
.mini{background:var(--card);color:var(--muted);box-shadow:var(--shadow-sm);border:1px solid var(--border);
  padding:8px 14px;font-size:13px;border-radius:9px}
.mini:hover{background:var(--card-sel);color:var(--text)}
.pbox{background:var(--input);border:1px solid var(--border);border-radius:11px;padding:11px 13px;margin-top:11px;
  font-size:13px;line-height:1.65;word-break:break-word}
.cardthumb{float:right;width:84px;height:84px;object-fit:cover;border-radius:9px;margin:0 0 6px 10px;
  box-shadow:var(--shadow-md)}
/* 速度指示:wifi 三竖条,绿=快/黄=一般/红=慢 */
.wifi{float:right;display:inline-flex;align-items:flex-end;gap:2px;height:18px;margin:2px 0 0 8px}
.wifi i{width:5px;border-radius:1.5px;background:var(--border)}
.wifi i:nth-child(1){height:7px}.wifi i:nth-child(2){height:13px}.wifi i:nth-child(3){height:18px}
.wifi.fast i{background:#22c55e}
.wifi.mid i:nth-child(-n+2){background:#eab308}
.wifi.slow i:nth-child(1){background:#ef4444}
.mfield{display:inline-block;margin:4px 0 2px;padding:3px 10px;border-radius:999px;font-size:12px;
  background:var(--chip-bg);color:var(--chip-tx);border:1px solid var(--chip-bd)}
.mmem{margin-top:6px;font-size:12px;color:var(--mem-tx);background:var(--mem-bg);border:1px solid var(--mem-bd);
  border-radius:8px;padding:4px 10px;display:inline-block}
.tabs{display:flex;gap:8px;margin:8px 0 4px}
.tab{padding:8px 18px;border-radius:999px;background:var(--glass);border:1px solid var(--border);
  color:var(--muted);font-size:13.5px;cursor:pointer;transition:.15s;user-select:none}
.tab:hover{border-color:var(--border-hi);color:var(--text)}
.tab.on{background:var(--grad);color:#fff;border-color:transparent;font-weight:600}
.logbox{background:var(--input);border:1px solid var(--border);border-radius:11px;padding:10px 13px;
  margin-top:12px;font-size:12.5px;font-family:ui-monospace,Menlo,monospace;line-height:1.65;
  max-height:260px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}
#themeBtn{position:fixed;top:16px;right:16px;z-index:99;width:42px;height:42px;padding:0;border-radius:999px;
  font-size:18px;line-height:1;display:flex;align-items:center;justify-content:center;
  background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  color:var(--text);box-shadow:var(--shadow-sm);border:1px solid var(--border);
  transition:transform .2s,box-shadow .2s}
#themeBtn:hover{transform:rotate(15deg) scale(1.08);box-shadow:var(--shadow-md)}
</style></head><body>
<button id="themeBtn" onclick="toggleTheme()" title="切换黑夜模式">🌙</button>
<h1>🎨 ImageVideoStudio · 生图/生视频小助手</h1>
<div id="app"></div>
<script>
const $ = s => document.querySelector(s);
let ST = {step:0, model:null, mname:'', msec:0, items:[], pids:[], hasCN:false};
// 全局兜底: 图片拖到上传区以外时,阻止浏览器直接打开图片导致页面跳走
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>e.preventDefault());
const NEG_DEF = %NEG%;
const SIZES = [[232,304,'小图·竖(约230x300,草图快)'],[304,232,'小图·横(草图快)'],[256,256,'小图·方(草图快)'],[512,512,'中方'],[1024,576,'横屏16:9 场景'],[1024,1024,'方形1:1 半身/头像'],[768,1152,'竖屏2:3 全身'],[1152,768,'横屏3:2 宽场景']];
// 提示词快捷标签(英文): 点一下自动加进对应框,再点一下取消
const POS_TAGS=['masterpiece','best quality','ultra detailed','8k','high resolution','sharp focus','cinematic lighting','photorealistic','anime style','depth of field','soft lighting','vibrant colors'];
const NEG_TAGS=['low quality','worst quality','blurry','bad anatomy','extra limbs','extra fingers','missing fingers','deformed hands','poorly drawn face','duplicate','watermark','text','cropped','jpeg artifacts','low resolution'];
async function j(u, opt){ const r = await fetch(u, opt); return r.json(); }
// 黑夜模式: 跟随系统/浏览器(prefers-color-scheme);手动切换过才用 localStorage 记住
const __mq = (window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null);
function applyTheme(d){ document.body.classList.toggle('dark', d); const b=$('#themeBtn'); if(b) b.textContent=d?'☀️':'🌙'; }
function savedTheme(){ try{ return localStorage.getItem('ivs_theme'); }catch(e){ return null; } }
function toggleTheme(){ const d=!document.body.classList.contains('dark'); try{localStorage.setItem('ivs_theme',d?'dark':'light');}catch(e){} applyTheme(d); }
(function(){
  const t=savedTheme();                                  // null=没手动选过
  applyTheme(t ? (t==='dark') : (__mq ? __mq.matches : false));
  if(__mq && __mq.addEventListener) __mq.addEventListener('change', e=>{ if(!savedTheme()) applyTheme(e.matches); });
})();

async function home(){
  ST = {step:0, model:null, mname:'', msec:0, items:[], pids:[], hasCN:ST.hasCN, kind:ST.kind};
  const st = await j('/api/svc/state');
  const s = await j('/api/status');
  let lc={paused:false}; try{ lc=await j('/api/llm/stats'); }catch(e){}
  if(!s.comfy_installed){
    $('#app').innerHTML = `<div class="box"><b>😅 还没检测到 ComfyUI</b><br><br>
      ImageVideoStudio 本身不画图,真正干活的是开源的 ComfyUI。请二选一:<br><br>
      <b>① 自动安装(推荐)</b>:打开终端,进入本项目目录,运行 <code>./install.sh</code><br>
      <b>② 你电脑里已经装过</b>:把 ComfyUI 文件夹路径填进 <code>config.json</code> 的 <code>comfy_dir</code>,或设置环境变量 <code>COMFYUI_DIR</code><br><br>
      装好后点下方按钮刷新。</div>
      <p><button onclick="home()">🔄 重新检测</button></p>`;
    return;
  }
  // 三卡片: 图片/视频/语言。点未启动的卡→自动停其他服务并启动,再进对应页
  let imgCard;
  if(st.img.running){
    imgCard = s.has_model
      ? `<div class="card" onclick="pickImg()"><b>🖼️ 图片生成</b> <span class="dot on">● 服务运行中·已检测 ${s.model_count} 个模型</span><div class="small">点击进入生图</div></div>`
      : `<div class="box"><b>🖼️ 生图服务已运行,但还没检测到图片模型</b><br>把图片模型(.safetensors)放进 <code>models/image/</code> 后点刷新。<p><button onclick="home()">🔄 重新检测</button></p></div>`;
  } else {
    imgCard = `<div class="card" onclick="startAndGo('img',1)"><b>🖼️ 图片生成</b> <span class="dot off">○ 未启动</span><div class="small">点击启动并进入(会先停掉其他服务腾内存)</div></div>`;
  }
  const vidCard = st.vid.running
    ? `<div class="card" onclick="goStep(10)"><b>🎬 视频生成</b> <span class="dot on">● 服务运行中</span><div class="small">点击进入图生视频</div></div>`
    : `<div class="card" onclick="startAndGo('vid',10)"><b>🎬 视频生成</b> <span class="dot off">○ 未启动</span><div class="small">点击启动并进入(会先停掉其他服务腾内存)</div></div>`;
  let llmCard;
  if(st.llm.running) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot on">● 运行中</span><div class="small">点击进入</div></div>`;
  else if(st.llm.alive_pid) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot off">◐ 加载中…</span><div class="small">点击进入查看</div></div>`;
  else if(lc.paused) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot off">⏸ 已暂停</span><div class="small">点击进入,可恢复</div></div>`;
  else llmCard = `<div class="card" onclick="goStep(20)"><b>💬 语言模型</b> <span class="dot off">○ 未启动</span><div class="small">点击选模型并启动</div></div>`;
  $('#app').innerHTML = `<h2>选一个开始</h2>` + imgCard + vidCard + llmCard;
}
function goStep(n){ ST.step=n; render(); }
async function startAndGo(type, nextStep){
  $('#app').innerHTML = `<div class="box">🚀 正在启动${type==='img'?'生图':'生视频'}服务…<br><span class="small">会先自动停掉其他服务腾内存,首次加载模型约 1~2 分钟,请稍候。</span></div>`;
  const r = await j('/api/svc/start?type='+type);
  if(r.error){ $('#app').innerHTML = `<div class="box">启动失败:${esc(r.error)}</div><p><button onclick="home()">返回</button></p>`; return; }
  let n=0;
  const t=setInterval(async()=>{
    let s; try{ s=await j('/api/svc/status?type='+type); }catch(e){ return; } n++;
    if(s.running){ clearInterval(t); ST.step=nextStep; render(); }
    else if(!s.alive_pid && n>3){ clearInterval(t); $('#app').innerHTML='<div class="box">启动失败,请查看项目目录日志(comfy.log / comfy_vid.log)。</div><p><button onclick="home()">返回</button></p>'; }
    else if(n>80){ clearInterval(t); $('#app').innerHTML='<div class="box">启动超时,请查看日志。</div><p><button onclick="home()">返回</button></p>'; }
  },3000);
}
function pickImg(){ ST.step=1; render(); }

async function render(){
  if(ST.step===1){
    const ms = await j('/api/models');
    ST.hasCN = (await j('/api/status')).has_cn;
    $('#app').innerHTML = `<h2>第二步:选生图模型</h2>` + tabHtml(0) + `
      <p class="small">右上角竖条 = 生成速度(<span style="color:#22c55e">绿</span>快 / <span style="color:#eab308">黄</span>一般 / <span style="color:#ef4444">红</span>慢)</p>` + ms.map(m=>`
      <div class="card" onclick="pickModel('${m.id}','${m.name}',${m.sec},'${m.kind}')">
        <span class="wifi ${m.speed||'mid'}" title="生成速度"><i></i><i></i><i></i></span>
        <b>${m.name}</b> <span class="small">${m.time||('约'+m.sec+'秒')}/张</span>
        ${m.field?`<div><span class="mfield">🎯 ${m.field}</span></div>`:''}
        <div class="small">${m.detail||m.desc}</div>
        ${m.mem?`<div><span class="mmem">💾 启动约占内存 ${m.mem}</span></div>`:''}
      </div>`).join('') + `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===3){
    let cards='';
    for(let i=0;i<ST.items.length;i++){ if(!ST.items[i]) ST.items[i]=newItem(); cards += cardHTML(i); }
    $('#app').innerHTML = `<h2>第三步:每张图单独设置</h2>
      <p class="small">模型:${ST.mname}(约${ST.msec}秒/张) · 不想要某张就点卡片右上角 ✕</p>${cards}
      <p><button class="back" onclick="addCard()">＋ 加一张图</button></p>
      <p><button onclick="startGen()">🚀 开始生成</button>
      <button class="back" onclick="ST.step=1;render()">← 返回</button></p>`;
    for(let i=0;i<ST.items.length;i++){
      if(ST.items[i].mode==='inpaint') initMaskCanvas(i);
      if(ST.items[i].mode==='pose'&&ST.items[i].poseSrc==='draw') initPoseCanvas(i);
      drawSzBox(i);
    }
  }
  if(ST.step===4){
    let cards='';
    for(let i=0;i<ST.items.length;i++){
      cards += `<div class="card" style="cursor:default" id="g${i}">
        <b>第 ${i+1} 张</b> <span class="small" id="st${i}">排队中…</span>
        <div class="bar"><i id="b${i}"></i></div><div id="img${i}"></div>
        <div class="row" id="acts${i}" style="display:none;margin-top:8px">
          <button class="back" onclick="viewPrompt(${i})">🔍 查看提示词</button>
          <button class="back" onclick="editPrompt(${i})">✏️ 编辑提示词重生成</button>
        </div>
        <div id="pe${i}"></div></div>`;
    }
    $('#app').innerHTML = `<h2>生成中…</h2>
      <p class="row"><label style="margin:0"><input type="checkbox" id="mask" onchange="toggleMask(this.checked)"> 给图片打遮(模糊) — 默认不打遮</label></p>
      ${cards}<p id="again" style="display:none">
        <button onclick="editRound()">✏️ 编辑生图(回到设置接着改)</button>
        <button onclick="newRound()">➕ 继续生图(全新一轮)</button>
        <button class="back" onclick="home()">🏠 回首页</button></p>`;
    pollAll();
  }
  if(ST.step===10){
    const vm = await j('/api/vid/models');
    if(!ST.vid) ST.vid={unet:vm.unets[0].id,lora:'none',pos:'',neg:'',w:640,h:352,frames:49,fps:24,stg:false,file:null,imgName:null,pid:null,url:null};
    const v=ST.vid;
    const unetCards=vm.unets.map(u=>`<div class="card ${v.unet===u.id?'sel':''}" onclick="vSet('unet','${u.id}',1)"><b>${u.name}</b> <span class="mfield">${u.tag}</span><div class="small">${u.desc}</div></div>`).join('');
    const loraOpts=vm.loras.map(l=>`<label style="display:block;margin:6px 0"><input type="radio" name="vlora" ${v.lora===l.id?'checked':''} onchange="vSet('lora','${l.id}')"> <b>${l.name}</b> <span class="small">${l.desc}</span></label>`).join('');
    const imgHtml = v.file
      ? `<b style="color:#18a058">✔ ${esc(v.file.name)}</b><br><img src="${URL.createObjectURL(v.file)}" alt="">`
      : `把源图拖到这里,或点击选择<br><span class="small">视频会让这张图动起来</span>`;
    $('#app').innerHTML = `<h2>图生视频</h2>
      <p class="small">选视频模型 → 上传源图 → 写"<b>动作</b>"(不是场景) → 开始</p>
      ${unetCards}
      <label>源图</label>
      <div class="drop" ondragover="event.preventDefault();this.classList.add('over')"
           ondragleave="this.classList.remove('over')" ondrop="vDrop(event)"
           onclick="$('#vfile').click()">${imgHtml}</div>
      <input type="file" id="vfile" accept="image/*" style="display:none" onchange="vSetFile(this.files[0])">
      <label>动作提示词(描述要做的动作,如: she turns around and smiles)</label>
      <textarea id="vpos" placeholder="写动作,不写场景">${esc(v.pos)}</textarea>
      <label>风格 LoRA(可叠加,白话说明)</label>${loraOpts}
      <div class="row">
        <span><label>分辨率</label><select onchange="vRes(this.value)">
          <option value="640x352" ${v.w===640?'selected':''}>640 × 352</option>
          <option value="512x288" ${v.w===512?'selected':''}>512 × 288(更快)</option></select></span>
        <span><label>帧数 <span id="vfnum">${v.frames}</span></label>
          <input type="range" min="25" max="97" step="8" value="${v.frames}" oninput="vFrames(this.value)"></span>
      </div>
      <p class="row"><label style="margin:0"><input type="checkbox" ${v.stg?'checked':''} onchange="vSet('stg',this.checked)"> STG 增强(动作更有力,稍慢)</label></p>
      <p><button onclick="vStart()">🚀 开始生成</button>
      <button class="back" onclick="home()">← 返回</button></p>
      <p class="small">提示: 新放进 models/video/ 的模型要重启视频服务才会被识别。</p>`;
  }
  if(ST.step===11){
    $('#app').innerHTML = `<h2>生成视频中…</h2>
      <div class="box">🎬 正在生成,约 1~3 分钟… <span id="vel">0s</span></div>
      <div class="bar"><i id="vbar"></i></div>
      <div id="vout" style="margin-top:14px"></div>
      <p id="vacts" style="display:none">
        <button onclick="ST.step=10;render()">🔁 再做一段</button>
        <button class="back" onclick="home()">🏠 回首页</button></p>`;
    vPoll();
  }
  if(ST.step===20){
    const ms = await j('/api/llm/models');
    ST._llmModels = ms;
    $('#app').innerHTML = `<h2>选语言模型</h2>` + ms.map(m=>`
      <div class="card ${m.exists?'':'dis'}" onclick="${m.exists?`llmPick('${m.id}')`:''}">
        <b>${m.name}</b> <span class="mfield">${m.tag}</span>
        <div class="small">${m.desc}</div>
        <div><span class="mmem">💾 需 GPU 上限 ${m.gpu_mb}MB</span></div>
        ${m.exists?'':'<div class="small" style="color:#d03050">模型文件缺失(检查 models/llm 软链)</div>'}
      </div>`).join('') + `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===21){
    const m=ST.llmModel; const p=m.prefs||{thinking:true,temp:0.7,max_tokens:8192};
    const thinkHtml = m.is_reasoning
      ? `<p class="row"><label style="margin:0"><input type="checkbox" id="lthink" ${p.thinking?'checked':''}> 思考模式(先推理再回答,更聪明但慢)</label></p>`
      : `<p class="small">此模型为快速应答型,无思考模式。</p>`;
    $('#app').innerHTML = `<h2>${esc(m.name)} · 启动参数</h2>
      ${thinkHtml}
      <label>温度(越高越发散) <span id="ltv">${p.temp}</span></label>
      <input type="range" id="ltemp" min="0" max="1.5" step="0.1" value="${p.temp}" oninput="$('#ltv').textContent=this.value">
      <label>最大回复长度(max_tokens)</label>
      <select id="lmax">${[4096,8192,16384].map(n=>`<option value="${n}" ${p.max_tokens===n?'selected':''}>${n}</option>`).join('')}</select>
      <p><button onclick="llmStart()">🚀 启动</button>
      <button class="back" onclick="goStep(20)">← 返回</button></p>`;
  }
  if(ST.step===22){
    const c = await j('/api/llm/stats');
    if(!c.model){
      $('#app').innerHTML = `<div class="box">语言模型未在运行。</div>
        <p><button onclick="goStep(20)">选择模型</button>
        <button class="back" onclick="home()">🏠 回首页</button></p>`;
      return;
    }
    if(c.paused){
      $('#app').innerHTML = `<h2>语言模型已暂停</h2>
        <div class="card" style="cursor:default"><b>💬 ${esc(c.model.name)}</b> <span class="dot off">⏸ 已暂停</span>
          <div class="small">进程已停止,内存已释放;点恢复可原样拉起</div></div>
        <div class="box">🔌 API 地址:<code>${esc(c.api_url)}</code> <span class="small">(暂停中,未监听)</span></div>
        <div class="row" style="gap:26px;margin:14px 0">
          <span>⏱ 本次已运行 <b>${fmtTime(c.elapsed_sec)}</b></span>
          <span>🧮 token: 输入 <b>${c.prompt_tokens}</b> · 输出 <b>${c.gen_tokens}</b></span></div>
        <p style="text-align:center;margin:20px 0"><button onclick="llmResume()" style="padding:13px 40px;font-size:17px">▶ 恢复</button></p>
        <p class="row">
          <button class="back" onclick="llmStop(0)">🔄 更换模型</button>
          <button class="back" onclick="llmStop(20)">🔄 更换语言模型</button>
          <button class="back" onclick="llmStop(0)">⏻ 关闭模型</button></p>`;
      return;
    }
    if(c.loading){
      $('#app').innerHTML = `<h2>语言模型加载中…</h2>
        <div class="box">⏳ ${esc(c.model.name)} 正在加载,约 1~2 分钟…</div>`;
      setTimeout(()=>{ if(ST.step===22) render(); },4000);
      return;
    }
    const port=c.port||8848;
    $('#app').innerHTML = `<h2>语言模型运行中</h2>
      <div class="card" style="cursor:default"><b>💬 ${esc(c.model.name)}</b> <span class="dot on">● 运行中</span>
        <div class="small">温度 ${c.model.temp} · max_tokens ${c.model.max_tokens}${c.model.thinking?' · 思考模式':''}</div></div>
      <div class="box">🔌 API 地址:<code>${esc(c.api_url)}</code>
        <button class="back" style="margin-left:8px" onclick="llmCopyApi('${c.api_url}')">复制</button></div>
      <div class="row" style="gap:26px;margin:14px 0">
        <span>⏱ 已启用 <b id="llmEl">${fmtTime(c.elapsed_sec)}</b></span>
        <span>🧮 token: 输入 <b id="llmPT">${c.prompt_tokens}</b> · 输出 <b id="llmGT">${c.gen_tokens}</b></span></div>
      <div class="row" style="margin:0 0 6px"><span>⚡ <b id="llmAct">${fmtAct(c.activity)}</b></span></div>
      <p style="text-align:center;margin:20px 0"><button onclick="llmPause()" style="padding:13px 40px;font-size:17px">⏸ 暂停</button></p>
      <p class="row">
        <button onclick="window.open('http://127.0.0.1:${port}')">💬 打开聊天网页</button></p>
      <p class="row">
        <button class="back" onclick="llmStop(0)">🔄 更换模型</button>
        <button class="back" onclick="llmStop(20)">🔄 更换语言模型</button>
        <button class="back" onclick="llmStop(0)">⏻ 关闭模型</button></p>`;
    llmTick();
  }
  if(ST.step===30){ // 预定批量: 模型大卡→风格子卡→提示词组(每模型独立,每风格可垫一张图)
    const cfg = await j('/api/batch/config');
    if(!ST.bstyles) ST.bstyles=null; // 导入的公共风格集 [{name,prompts:[...]}],null=未导入
    const mcards = cfg.models.map(m=>bModelCard(m)).join('');
    const body = `
      <div class="box">每个模型一张大卡,卡里多个<b>风格子卡</b>;一个风格=一组提示词(一行一条)=多张图。
        逐张真等画完才跑下一张,归档到 <code>${esc(cfg.org_dir)}</code> 下的 <code>图片N.png</code><br>
        <span class="small">默认全选 + 每模型一个空白「普通」风格;改这里只本次生效</span></div>
      <div class="box">📂 从文件夹导入风格(每个 .txt = 一个风格,文件内空行分隔多条提示词):
        <div class="row" style="margin-top:6px">
          <input id="bfolder" type="text" placeholder="文件夹路径,如 /Users/wl/Desktop/ImageVideoStudio/prompts_sfw"
            style="flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px">
          <button class="back" onclick="bImport()">导入到所有模型</button></div>
        <span class="small" id="bimpMsg"></span></div>
      <div class="row" style="align-items:center;gap:10px">
        <label style="margin:0">图片尺寸</label>
        <select id="bsize" style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
          <option value="640x360">640×360(测试小图,快)</option>
          <option value="1024x720" selected>1024×720(默认)</option>
          <option value="832x1216">832×1216(竖图)</option>
        </select></div>
      <label>公共负向提示词</label><textarea id="bneg" style="height:44px">${esc(cfg.neg)}</textarea>
      ${mcards}
      <p><button onclick="batchStart()">🚀 开始批量</button></p>`;
    $('#app').innerHTML = `<h2>第二步:预定批量</h2>` + tabHtml(1) + body +
      `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===31){ // 批量进度: 只显示进度和日志,不显示图片,告知图片所在文件夹
    let b; try{ b=await j('/api/batch/status'); }catch(e){ return; }
    const pct=b.total?Math.round(b.done/b.total*100):0;
    $('#app').innerHTML = `<h2>批量生成${b.running?'中…':'已结束'}</h2>
      <div class="card" style="cursor:default">
        <span style="font-size:22px;font-weight:700">${b.done}/${b.total}</span> 张 ·
        <span style="color:var(--on)">✓ ${b.ok}</span> · <span style="color:#e0556b">✗ ${b.fail}</span>
        ${b.current?`<div class="small" style="margin-top:4px">正在画:${esc(b.current)}</div>`:''}
        <div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div></div>
      <div class="logbox" id="blog">${esc(b.log.slice(-18).join('\n'))}</div>
      <div class="box">📁 图片在 <code>output/organized/&lt;模型&gt;/&lt;风格&gt;/图片N.png</code> 里,本页不显示图片;关掉网页后台照样继续跑。</div>
      <p>${b.running?'<button class="back" onclick="batchStop()">⏹ 停止批量</button>':'<button onclick="goStep(30)">🔁 再来一批</button>'}
      <button class="back" onclick="home()">🏠 回首页</button></p>`;
    const lb=$('#blog'); if(lb) lb.scrollTop=lb.scrollHeight;
    if(b.running) setTimeout(()=>{ if(ST.step===31) render(); },2500);
  }
  if(ST.step===40){ // 漫画连载: 解析 story.txt 分镜,可选上传参考照片固定主角
    const cfg = await j('/api/comic/story');
    let body;
    if(!cfg.panels.length){
      body = `<div class="box">${esc(cfg.story_file)} 里没有分镜,按 "### Image 编号 / Prompt: / Dialogue:" 格式写几格再回来。</div>`;
    } else {
      const opts = cfg.models.map(m=>`<option value="${m.id}" ${/waiIllustriousSDXL_v170/.test(m.id)?'selected':''}>${esc(m.name)}</option>`).join('');
      const rows = cfg.panels.map((p,i)=>`
      <div class="card" style="cursor:default;padding:12px 14px">
        <label style="margin:0;cursor:pointer"><input type="checkbox" class="cp" data-i="${i}" data-num="${p.num}" checked> <b>第 ${String(p.num).padStart(3,'0')} 格</b></label>
        <textarea class="cpp" data-i="${i}" style="height:52px;margin-top:6px">${esc(p.prompt)}</textarea>
        <input class="cpd" data-i="${i}" type="text" value="${esc(p.dialogue)}" placeholder="台词(可改)"
          style="width:100%;margin-top:6px;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px;box-sizing:border-box">
      </div>`).join('');
      body = `
      <div class="box">共 <b>${cfg.panels.length}</b> 格 · 逐格真等画完,完成即叠加台词归档到 <code>${esc(cfg.out_dir)}</code><br>
      <span class="small">主文件:${esc(cfg.story_file)} · 这里的修改只本次生效;上传你的照片后每格按 0.75 重绘,主角就是你的动漫形象,不怕侵权</span></div>
      <label>生图模型(默认动漫最强)</label><select id="cmodel" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${opts}</select>
      <label>参考照片(可选,固定主角长相)</label>
      <div class="row"><input type="file" accept="image/*" onchange="comicUp(this.files[0])"><span class="small" id="crefName">${ST.comicRef?'✔ 已上传':'未上传则纯文字生成'}</span></div>
      <label>分镜(提示词和台词都可改)</label>${rows}
      <p><button onclick="comicStart()">🚀 开始连载</button></p>`;
    }
    $('#app').innerHTML = `<h2>第二步:漫画连载</h2>` + tabHtml(2) + ccTabHtml(0) + body +
      `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===41){ // 连载进度: 进度+日志+最新一格预览
    let b; try{ b=await j('/api/comic/status'); }catch(e){ return; }
    const pct=b.total?Math.round(b.done/b.total*100):0;
    $('#app').innerHTML = `<h2>连载生成${b.running?'中…':'已结束'}</h2>
      <div class="card" style="cursor:default">
        <span style="font-size:22px;font-weight:700">${b.done}/${b.total}</span> 格 ·
        <span style="color:var(--on)">✓ ${b.ok}</span> · <span style="color:#e0556b">✗ ${b.fail}</span>
        ${b.current?`<div class="small" style="margin-top:4px">正在画:${esc(b.current)}</div>`:''}
        <div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div></div>
      ${b.last?`<div class="card" style="cursor:default"><div class="small">最新一格:</div><img src="/comic_out/${b.last}?t=${Date.now()}" style="max-width:100%;border-radius:10px"></div>`:''}
      <div class="logbox" id="clog">${esc(b.log.slice(-15).join('\n'))}</div>
      <div class="box">📁 漫画在 <code>output/my_story_comic/</code>,关掉网页后台照样继续跑。</div>
      <p>${b.running?'<button class="back" onclick="comicStop()">⏹ 停止连载</button>':'<button onclick="goStep(40)">🔁 再跑一次</button>'}
      <button class="back" onclick="home()">🏠 回首页</button></p>`;
    const lc2=$('#clog'); if(lc2) lc2.scrollTop=lc2.scrollHeight;
    if(b.running) setTimeout(()=>{ if(ST.step===41) render(); },2500);
  }
  if(ST.step===50){ // 自定义连载·第1步 角色设定: 钉死每个主角的设定图并逐张审核
    const P=ST.cc||{chars:[]};
    const cards=P.chars.map((c,i)=>`
      <div class="card" style="cursor:default">
        <div class="row" style="justify-content:space-between"><b>${c.role==='male'?'🧑 男主':'👩 女主'} ${esc(c.name||('#'+(i+1)))}</b>
          <span class="delcard" onclick="ccDelChar(${i})">✕</span></div>
        <label>外貌/服装特征(文字锚点,英文更稳)</label>
        <textarea id="cdesc${i}" style="height:56px">${esc(c.desc||'')}</textarea>
        <div class="row" style="margin-top:6px">
          <input type="file" accept="image/*" onchange="ccUpChar(${i},this.files[0])" style="flex:1">
          <button class="back" onclick="ccSheet(${i})" ${P.sheetRunning?'disabled':''}>${c.sheetUrl?'🔄 重新生成':'🎨 生成设定图'}</button></div>
        <div class="small" id="cup${i}">${c.ref?'✔ 已传照片,按 0.6 漫改成设定图':'不传照片则纯文字生成'}</div>
        ${c.sheetUrl?`<div style="margin-top:8px"><img src="${c.sheetUrl}?t=${Date.now()}" style="max-width:220px;border-radius:10px;border:1px solid var(--border)">
          <div class="row" style="margin-top:6px">
            <input type="text" id="cfix${i}" placeholder="补充指导(如:发型再短一点)" style="flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
            <button class="back" onclick="ccFix(${i})">完善重画</button>
            <button onclick="ccApprove(${i})">${c.approved?'✔ 已通过':'✔ 通过'}</button></div></div>`
        :`<div class="small" style="margin-top:6px">还没设定图,点「生成设定图」</div>`}
      </div>`).join('');
    $('#app').innerHTML = `<h2>自定义连载 · 1/3 角色设定</h2>` + tabHtml(2) + ccTabHtml(1) + `
      <div class="box">先把主角「钉死」:每个角色写特征或传你的照片 → 生成设定图 → 审核通过。全部通过后下一步。<span class="small">(角色不定死,后面几十格会画崩)</span></div>
      <div class="row"><span>男主</span><input type="number" id="nmale" min="0" max="2" value="${P.nmale??1}" style="width:60px">
        <span>女主</span><input type="number" id="nfemale" min="0" max="8" value="${P.nfemale??1}" style="width:60px">
        <button class="back" onclick="ccBuildChars()">生成角色卡</button>
        <span class="small">男最多2 · 女最多8</span></div>
      ${P.sheetRunning?'<div class="box">🎨 正在画设定图,约 1 分钟,画好自动出现…</div>':''}
      ${cards||'<div class="box">先定人数,点「生成角色卡」</div>'}
      <div class="row"><button class="back" onclick="ccSave(50)">💾 保存进度</button>
      <button onclick="ccToPanels()">下一步:分镜画布 →</button></div>`;
    if(P.sheetRunning) setTimeout(()=>{ if(ST.step===50) ccPollSheet(); },2500);
  }
  if(ST.step===51){ // 自定义连载·第2步 分镜画布: 左场景/右台词+穿搭,可增删,风格可继承上一张
    const P=ST.cc||{chars:[],panels:[]};
    const cnames=P.chars.map(c=>c.name||'?');
    const whoOpts=(sel)=>cnames.map(n=>`<option ${n===sel?'selected':''}>${esc(n)}</option>`).join('');
    const panels=P.panels.map((p,i)=>{
      const dlg=p.dialogues.map((d,j)=>`
        <div class="row" style="margin:4px 0">
          <select data-p="${i}" data-j="${j}" class="cdwho" style="width:90px">${whoOpts(d.who)}</select>
          <input type="text" class="cdtext" data-p="${i}" data-j="${j}" value="${esc(d.text||'')}" placeholder="台词" style="flex:1;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
          <span class="delcard" onclick="ccDelDlg(${i},${j})">✕</span></div>`).join('');
      const refOpts=P.chars.map((c,ci)=>`<option value="${ci}" ${p.ref_char===ci?'selected':''}>${esc(c.name||('#'+(ci+1)))}</option>`).join('');
      return `<div class="card" style="cursor:default">
        <div class="row" style="justify-content:space-between"><b>第 ${i+1} 格</b>
          <span class="delcard" onclick="ccDelPanel(${i})">✕ 删除</span></div>
        <div class="row" style="align-items:flex-start;gap:12px">
          <div style="flex:1.2">
            <label>场景/站位/剧情</label>
            <textarea class="cpscene" data-p="${i}" style="height:96px" placeholder="英文更稳: 谁在左谁在右,做什么,什么氛围">${esc(p.scene||'')}</textarea></div>
          <div style="flex:1">
            <label>台词(每人独立气泡)</label>${dlg||'<div class="small">无台词</div>'}
            <button class="back" style="padding:5px 12px;font-size:12px" onclick="ccAddDlg(${i})">＋ 加一句</button>
            <label style="margin-top:6px">当格穿搭(可空)</label>
            <input type="text" class="cpoutfit" data-p="${i}" value="${esc(p.outfit||'')}" placeholder="如: 白色球衣" style="width:100%;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)"></div>
        </div>
        <div class="row" style="margin-top:6px;flex-wrap:wrap">
          <span>垫图主角 <select class="cpref" data-p="${i}">${refOpts}</select></span>
          <span>尺寸 <select class="cpsize" data-p="${i}">
            <option ${p.w===1024&&p.h===720?'selected':''}>1024x720</option>
            <option ${p.w===832&&p.h===1216?'selected':''}>832x1216</option>
            <option ${p.w===1024&&p.h===1024?'selected':''}>1024x1024</option></select></span>
          <span>风格 <input type="text" class="cpstyle" data-p="${i}" value="${esc(p.style||'')}" placeholder="留空=继承上一张" style="width:200px;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)"></span></div>
      </div>`;}).join('');
    $('#app').innerHTML = `<h2>自定义连载 · 2/3 分镜画布</h2>` + tabHtml(2) + ccTabHtml(1) + `
      <div class="box">要几格就加几格(比如连载 79 格)。每格:左边场景站位剧情,右边逐句台词+当格穿搭。风格留空就沿用上一张。</div>
      <div class="row"><button class="back" onclick="ccAddPanel()">＋ 新增画布</button>
        <span class="small">共 ${P.panels.length} 格 · 约 ${Math.round(P.panels.length*1.5)} 分钟</span></div>
      ${panels||'<div class="box">点「＋ 新增画布」开始搭</div>'}
      <div class="row"><button class="back" onclick="ccSave(51)">💾 保存进度</button>
      <button class="back" onclick="goStep(50)">← 角色设定</button>
      <button onclick="ccStartGen()">🚀 开始生成(${P.panels.length}格)</button></div>`;
  }
  if(ST.step===52){ // 自定义连载·进度: 进度+日志+最新一格
    let b; try{ b=await j('/api/cc/status'); }catch(e){ return; }
    const pct=b.total?Math.round(b.done/b.total*100):0;
    $('#app').innerHTML = `<h2>自定义连载${b.running?'生成中…':'已结束'}</h2>` + tabHtml(2) + ccTabHtml(1) + `
      <div class="card" style="cursor:default">
        <span style="font-size:22px;font-weight:700">${b.done}/${b.total}</span> 格 ·
        <span style="color:var(--on)">✓ ${b.ok}</span> · <span style="color:#e0556b">✗ ${b.fail}</span>
        ${b.current?`<div class="small" style="margin-top:4px">正在画:${esc(b.current)}</div>`:''}
        <div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div></div>
      ${b.last?`<div class="card" style="cursor:default"><div class="small">最新一格:</div><img src="/cc_out/${b.last}?t=${Date.now()}" style="max-width:100%;border-radius:10px"></div>`:''}
      <div class="logbox" id="cclog">${esc(b.log.slice(-15).join('\n'))}</div>
      <div class="box">📁 漫画在 <code>output/custom_comic/</code>,关掉网页后台照样继续跑。</div>
      <p>${b.running?'<button class="back" onclick="ccStop()">⏹ 停止</button>':'<button onclick="goStep(51)">🔁 回分镜再改</button>'}
      <button class="back" onclick="home()">🏠 回首页</button></p>`;
    const lc3=$('#cclog'); if(lc3) lc3.scrollTop=lc3.scrollHeight;
    if(b.running) setTimeout(()=>{ if(ST.step===52) render(); },2500);
  }
}
function newItem(){ return {mode:'t2i',w:1024,h:576,pos:'',neg:NEG_DEF,defPos:'masterpiece, best quality',strength:0.6,scale:2,ctype:'openpose',refFile:null,strokes:[],brush:30,poseSrc:'draw',joints:null,resultUrl:null}; }
// ---- 视频 I2V 交互 ----
function vSet(k,val,rerender){ ST.vid[k]=val; if(rerender) render(); }
function vRes(s){ const[a,b]=s.split('x'); ST.vid.w=+a; ST.vid.h=+b; }
function vFrames(n){ n=Math.round(n); ST.vid.frames=n; $('#vfnum').textContent=n; }
function vSetFile(f){ if(!f) return; ST.vid.file=f; render(); }
function vDrop(e){ e.preventDefault(); e.currentTarget.classList.remove('over'); const f=e.dataTransfer.files[0]; if(f) vSetFile(f); }
async function vStart(){
  const v=ST.vid; v.pos=$('#vpos').value;
  if(!v.file){ alert('请先上传一张源图'); return; }
  $('#app').innerHTML = `<div class="box">📤 上传源图并提交…</div>`;
  try{
    const fd=new FormData(); fd.append('image', v.file, v.file.name); fd.append('overwrite','true');
    const ud=await (await fetch('/api/vid/upload',{method:'POST',body:fd})).json();
    if(ud.error) throw new Error(ud.error);
    const d=await (await fetch('/api/vid/gen',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:'vid_'+Date.now(),unet:v.unet,pos:v.pos,neg:v.neg,image:ud.name,
        w:v.w,h:v.h,frames:v.frames,fps:v.fps,lora:v.lora,lora_strength:0.8,stg:v.stg,steps:8})})).json();
    if(d.error) throw new Error(d.error);
    v.pid=d.pid; ST.step=11; render();
  }catch(e){ $('#app').innerHTML = `<div class="box">提交失败:${esc(e.message)}</div><p><button onclick="ST.step=10;render()">← 返回</button></p>`; }
}
async function vPoll(){
  const v=ST.vid; const t0=Date.now();
  const t=setInterval(async()=>{
    if(ST.step!==11){ clearInterval(t); return; }
    let d; try{ d=await j('/api/vid/poll?pid='+v.pid); }catch(e){ return; }
    const el=Math.round((Date.now()-t0)/1000); if($('#vel')) $('#vel').textContent=el+'s';
    if($('#vbar')) $('#vbar').style.width=Math.min(95,el/1.5)+'%';
    if(d.error){ clearInterval(t); $('#vout').innerHTML=`<div class="box">生成失败:${esc(d.error)}</div>`; $('#vacts').style.display='block'; return; }
    if(d.done){ clearInterval(t); if($('#vbar')) $('#vbar').style.width='100%';
      $('#vout').innerHTML=`<video controls autoplay loop src="${d.url}" style="max-width:100%;border-radius:10px"></video>`;
      $('#vacts').style.display='block'; }
  },2500);
}
// ---- 语言模型交互 ----
function llmPick(id){ ST.llmModel=(ST._llmModels||[]).find(m=>m.id===id); ST.step=21; render(); }
async function llmStart(){
  const m=ST.llmModel;
  const thinking=m.is_reasoning?($('#lthink')?$('#lthink').checked:m.prefs.thinking):false;
  const temp=+($('#ltemp')?$('#ltemp').value:m.prefs.temp);
  const max_tokens=+($('#lmax')?$('#lmax').value:m.prefs.max_tokens);
  $('#app').innerHTML = `<div class="box">🚀 正在启动 ${esc(m.name)}…<br><span class="small">会先停掉其他服务腾内存;若 GPU 上限不足,会弹一次 macOS 密码框。</span></div>`;
  const d=await (await fetch('/api/llm/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:m.id,thinking,temp,max_tokens})})).json();
  if(d.error){ let h=`<div class="box">启动失败:${esc(d.error)}</div>`;
    if(d.manual) h+=`<div class="box">可在终端手动运行(<code>!</code>前缀)后重试:<br><code>${esc(d.manual)}</code></div>`;
    h+=`<p><button onclick="goStep(21)">← 返回</button></p>`; $('#app').innerHTML=h; return; }
  ST.step=22; render();
}
async function llmStop(next){
  $('#app').innerHTML = `<div class="box">⏳ 正在停止语言模型…</div>`;
  ST.llmPaused=false;
  await j('/api/svc/stop?type=llm');
  if(next===0) home(); else goStep(next);
}
function fmtTime(s){ s=Math.max(0,Math.floor(s)); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return (h?h+'小时':'')+(h||m?m+'分':'')+x+'秒'; }
function fmtAct(a){ // "它正在干嘛"实时状态: 空闲/消化输入(带进度)/生成回复(带速度)
  if(!a) return '…';
  if(a.state==='idle') return '💤 空闲 · 等指令'+(a.tps?`(上次生成 ${a.tps.toFixed(1)} tok/s)`:'');
  if(a.state==='prompt') return `📥 消化输入中 ${Math.round((a.progress||0)*100)}% · ${(a.tps||0).toFixed(0)} tok/s`;
  if(a.state==='gen') return '✍️ 生成回复中…'+(a.tps?` · ${a.tps.toFixed(1)} tok/s`:'');
  return '…'; }
function llmCopyApi(u){ if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(u).then(()=>alert('已复制: '+u),()=>alert(u)); } else alert(u); }
function llmTick(){
  if(ST._llmTimer) clearInterval(ST._llmTimer);
  ST._llmTimer=setInterval(async()=>{
    if(ST.step!==22){ clearInterval(ST._llmTimer); ST._llmTimer=null; return; }
    let s; try{ s=await j('/api/llm/stats'); }catch(e){ return; }
    if(s.paused){ clearInterval(ST._llmTimer); ST._llmTimer=null; render(); return; }
    if($('#llmEl')) $('#llmEl').textContent=fmtTime(s.elapsed_sec);
    if($('#llmPT')) $('#llmPT').textContent=s.prompt_tokens;
    if($('#llmGT')) $('#llmGT').textContent=s.gen_tokens;
    if($('#llmAct')) $('#llmAct').textContent=fmtAct(s.activity);
  },1000);
}
async function llmPause(){ ST.llmPaused=true; await j('/api/llm/pause'); render(); }
async function llmResume(){
  $('#app').innerHTML=`<div class="box">▶ 正在恢复语言模型…</div>`;
  ST.llmPaused=false;
  await (await fetch('/api/llm/resume',{method:'POST'})).json();
  ST.step=22; render();
}
// ---- 预定批量交互 ----
function tabHtml(active){ // 图片区顶部面包屑切换: 单模型 / 预定批量 / 漫画连载
  return `<div class="tabs">
    <span class="tab ${active===0?'on':''}" onclick="goStep(1)">🖼️ 单模型生成</span>
    <span class="tab ${active===1?'on':''}" onclick="goStep(30)">📦 预定批量</span>
    <span class="tab ${active===2?'on':''}" onclick="goStep(40)">📖 漫画连载</span></div>`;
}
// ---- 预定批量: 模型大卡 → 风格子卡 → 提示词组 ----
function bModelCard(m){ // 一个模型一张大卡,内含风格子卡容器
  return `<div class="card bmodel" data-id="${m.id}" data-name="${esc(m.name)}" style="cursor:default;padding:14px 16px">
    <label style="margin:0;cursor:pointer;font-size:16px"><input type="checkbox" class="bm" checked> <b>🧠 ${esc(m.name)}</b></label>
    <div class="bstyles" style="margin-top:10px">${bStyleCard({name:'普通',prompts:[]})}</div>
    <button class="back" style="margin-top:8px;padding:6px 14px;font-size:13px" onclick="bAddStyle(this)">＋ 新增风格</button></div>`;
}
function bStyleCard(s){ // 一个风格子卡: 风格名 + 提示词组(一行一条) + 垫图 + 删除
  return `<div class="card bstyle" data-ref="" style="cursor:default;padding:10px 12px;margin-top:8px;background:var(--input)">
    <div class="row" style="gap:8px;align-items:center">
      <input class="bs-name" type="text" value="${esc(s.name||'')}" placeholder="风格名,如 哥特 / 像素"
        style="flex:1;padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:13px">
      <button class="back" style="padding:6px 10px;font-size:12px" onclick="bDelStyle(this)">🗑 删风格</button></div>
    <textarea class="bs-prompts" placeholder="提示词组:一行一条,一条画一张(可多条)" style="height:64px;margin-top:8px">${esc((s.prompts||[]).join('\n'))}</textarea>
    <div class="row" style="margin-top:6px;gap:8px;align-items:center">
      <span class="small">垫图(可选):</span>
      <input type="file" accept="image/*" style="font-size:12px" onchange="bUpRef(this,this.files[0])">
      <span class="small bs-refname">不传=纯文字生成</span></div></div>`;
}
function bAddStyle(btn){ // 在所属模型大卡里追加一个空白风格子卡
  btn.insertAdjacentHTML('beforebegin', bStyleCard({name:'',prompts:[]}));
}
function bDelStyle(btn){ // 删除一个风格子卡(至少保留一个则随便删)
  const card=btn.closest('.bstyle'); const wrap=card.parentElement;
  if(wrap.querySelectorAll('.bstyle').length<=1){ alert('每个模型至少留一个风格'); return; }
  card.remove();
}
async function bUpRef(input,f){ // 给某个风格传一张垫图(i2i 0.75)
  if(!f) return;
  const fd=new FormData(); fd.append('image',f,f.name); fd.append('overwrite','true');
  const ud=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
  if(ud.error||!ud.name){ alert('垫图上传失败:'+(ud.error||'未知')); return; }
  const card=input.closest('.bstyle'); card.dataset.ref=ud.name;
  card.querySelector('.bs-refname').textContent='✔ 垫图:'+f.name;
}
async function bImport(){ // 从文件夹导入风格集,套用到所有模型大卡
  const folder=($('#bfolder').value||'').trim();
  if(!folder){ alert('先填文件夹路径'); return; }
  const r=await (await fetch('/api/batch/import',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({folder})})).json();
  if(r.error){ $('#bimpMsg').textContent='✗ '+r.error; return; }
  document.querySelectorAll('.bmodel .bstyles').forEach(w=>{ w.innerHTML=r.styles.map(s=>bStyleCard(s)).join(''); });
  $('#bimpMsg').textContent=`✔ 已导入 ${r.styles.length} 个风格(共 ${r.styles.reduce((a,s)=>a+s.prompts.length,0)} 条)到每个模型`;
}
async function batchStart(){
  const size=($('#bsize').value||'1024x720').split('x');
  const models=[];
  document.querySelectorAll('.bmodel').forEach(mc=>{
    if(!mc.querySelector('.bm').checked) return;
    const styles=[];
    mc.querySelectorAll('.bstyle').forEach(sc=>{
      const name=sc.querySelector('.bs-name').value.trim()||'普通';
      const prompts=sc.querySelector('.bs-prompts').value.split('\n').map(x=>x.trim()).filter(Boolean);
      if(prompts.length) styles.push({name,prompts,ref:sc.dataset.ref||''});
    });
    if(styles.length) models.push({id:mc.dataset.id,name:mc.dataset.name,styles});
  });
  if(!models.length){ alert('至少勾一个模型,且每个勾选模型至少一条提示词'); return; }
  const total=models.reduce((a,m)=>a+m.styles.reduce((x,s)=>x+s.prompts.length,0),0);
  if(!confirm(`将生成 ${models.length} 个模型共 ${total} 张图,开始?`)) return;
  const r=await (await fetch('/api/batch/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({models,neg:$('#bneg').value,w:+size[0],h:+size[1]})})).json();
  if(r.error){ alert(r.error); return; }
  ST.step=31; render();
}
async function batchStop(){ await (await fetch('/api/batch/stop',{method:'POST'})).json(); }
async function comicUp(f){ // 上传参考照片(你的照片→主角动漫形象,i2i 0.75 重绘)
  if(!f) return;
  const fd=new FormData(); fd.append('image',f,f.name); fd.append('overwrite','true');
  const ud=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
  if(ud.error||!ud.name){ alert('上传失败:'+(ud.error||'未知')); return; }
  ST.comicRef=ud.name; if($('#crefName')) $('#crefName').textContent='✔ 已上传:'+f.name;
}
async function comicStart(){
  const model=$('#cmodel').value;
  const panels=[...document.querySelectorAll('.cp:checked')].map(c=>{const i=c.dataset.i;
    return {num:+c.dataset.num,
      prompt:(document.querySelector('.cpp[data-i="'+i+'"]')||{}).value||'',
      dialogue:(document.querySelector('.cpd[data-i="'+i+'"]')||{}).value||''};})
    .filter(p=>p.prompt.trim());
  if(!panels.length){ alert('至少留一格分镜'); return; }
  const r=await (await fetch('/api/comic/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model,panels,ref:ST.comicRef||'',strength:0.75,w:1024,h:720})})).json();
  if(r.error){ alert(r.error); return; }
  ST.step=41; render();
}
async function comicStop(){ await (await fetch('/api/comic/stop',{method:'POST'})).json(); }
// ---- 自定义连载交互(角色设定→分镜画布→生成) ----
function ccTabHtml(active){ // 漫画连载内部: 模板 / 自定义 切换
  return `<div class="tabs" style="margin-top:4px">
    <span class="tab ${active===0?'on':''}" onclick="goStep(40)">📄 模板连载</span>
    <span class="tab ${active===1?'on':''}" onclick="ccEnter()">🎨 自定义连载</span></div>`;
}
async function ccEnter(){ // 进自定义连载: 载入已存进度,有进度回对应步,否则从角色设定开始
  let p; try{ p=await j('/api/cc/project'); }catch(e){ p={chars:[],panels:[]}; }
  p.chars=p.chars||[]; p.panels=p.panels||[];
  const cs=await j('/api/cc/status');
  if(cs && cs.running){ ST.cc=p; ST.step=52; render(); return; }
  ST.cc=p;
  ST.step = p.chars.length ? (p.panels.length?51:50) : 50;
  render();
}
async function ccSave(backStep){ // 把当前角色+分镜存到 my_story/custom_project.json
  const P=ST.cc||{chars:[],panels:[]};
  if(ST.step===50) ccCollectChars();
  if(ST.step===51) ccCollectPanels();
  await fetch('/api/cc/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({chars:P.chars,panels:P.panels})});
  if(backStep!=null && backStep!==ST.step){ ST.step=backStep; }
  render();
}
function ccCollectChars(){ // 把 50 页输入读回 ST.cc.chars
  const P=ST.cc; if(!P) return;
  P.nmale=+($('#nmale')?$('#nmale').value:1); P.nfemale=+($('#nfemale')?$('#nfemale').value:1);
  P.chars.forEach((c,i)=>{ const d=$('#cdesc'+i); if(d) c.desc=d.value; });
}
function ccBuildChars(){ // 按男/女人数重建角色卡(保留已填的)
  const P=ST.cc||{chars:[]}; ccCollectChars();
  const nm=Math.min(2,Math.max(0,+$('#nmale').value||0));
  const nf=Math.min(8,Math.max(0,+$('#nfemale').value||0));
  const old=P.chars, chars=[];
  for(let i=0;i<nm;i++) chars.push(old.filter(c=>c.role==='male')[i]||{role:'male',name:'男主'+(i+1),desc:'',ref:'',sheet:'',sheetUrl:'',approved:false});
  for(let i=0;i<nf;i++) chars.push(old.filter(c=>c.role==='female')[i]||{role:'female',name:'女主'+(i+1),desc:'',ref:'',sheet:'',sheetUrl:'',approved:false});
  P.chars=chars; P.panels=P.panels||[]; ST.cc=P; ccSave(50);
}
function ccDelChar(i){ ST.cc.chars.splice(i,1); ccSave(50); }
async function ccUpChar(i,f){ // 上传角色照片(真人→漫改设定图,不侵权)
  if(!f) return;
  const fd=new FormData(); fd.append('image',f,f.name); fd.append('overwrite','true');
  const ud=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
  if(ud.error||!ud.name){ alert('上传失败:'+(ud.error||'未知')); return; }
  ST.cc.chars[i].ref=ud.name; if($('#cup'+i)) $('#cup'+i).textContent='✔ 已传照片,按 0.6 漫改成设定图';
}
async function ccSheet(i){ // 生成/重画某角色设定图
  const c=ST.cc.chars[i]; const d=$('#cdesc'+i); if(d) c.desc=d.value;
  const extra=$('#cfix'+i)?$('#cfix'+i).value.trim():'';
  const desc=(c.desc||'')+(extra?(', '+extra):'');
  const r=await (await fetch('/api/cc/sheet',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({role:c.role,desc,ref:c.ref||''})})).json();
  if(r.error){ alert(r.error); return; }
  ST.cc.sheetRunning=true; ST.cc.sheetFor=i; ccSave(50);
}
function ccFix(i){ ccSheet(i); } // 按补充指导重画(提示词已在 ccSheet 里拼上)
async function ccPollSheet(){ // 轮询设定图是否画完,画完贴回对应角色
  const s=await j('/api/cc/status'); const sh=(s&&s.sheet)||{};
  if(sh.running){ if(ST.step===50){ ST.cc.sheetRunning=true; render(); } return; }
  ST.cc.sheetRunning=false;
  const i=ST.cc.sheetFor;
  if(sh.error){ alert('设定图失败:'+sh.error); render(); return; }
  if(i!=null && sh.url){ ST.cc.chars[i].sheet=sh.name+'.png'; ST.cc.chars[i].sheetUrl=sh.url; ST.cc.chars[i].approved=false; }
  ccSave(50);
}
function ccApprove(i){ ST.cc.chars[i].approved=!ST.cc.chars[i].approved; ccSave(50); }
function ccToPanels(){ // 去第2步: 要求全部角色都有设定图且通过审核
  const P=ST.cc; ccCollectChars();
  if(!P.chars.length){ alert('先生成角色卡'); return; }
  const bad=P.chars.find(c=>!c.sheetUrl||!c.approved);
  if(bad){ alert(`「${bad.name||'?'}」还没${!bad.sheetUrl?'生成设定图':'点✔通过'},全部通过才能下一步`); return; }
  P.panels=P.panels||[]; ccSave(51);
}
function ccCollectPanels(){ // 把 51 页输入读回 ST.cc.panels(风格留空→继承上一张)
  const P=ST.cc; if(!P) return;
  P.panels.forEach((p,i)=>{
    const sc=document.querySelector('.cpscene[data-p="'+i+'"]'); if(sc) p.scene=sc.value;
    const of=document.querySelector('.cpoutfit[data-p="'+i+'"]'); if(of) p.outfit=of.value;
    const rf=document.querySelector('.cpref[data-p="'+i+'"]'); if(rf) p.ref_char=+rf.value;
    const sz=document.querySelector('.cpsize[data-p="'+i+'"]');
    if(sz){ const[a,b]=sz.value.split('x'); p.w=+a; p.h=+b; }
    const st=document.querySelector('.cpstyle[data-p="'+i+'"]'); if(st) p.style=st.value;
    p.dialogues.forEach((d,j)=>{
      const w=document.querySelector('.cdwho[data-p="'+i+'"][data-j="'+j+'"]'); if(w) d.who=w.value;
      const t=document.querySelector('.cdtext[data-p="'+i+'"][data-j="'+j+'"]'); if(t) d.text=t.value;
    });
  });
  // 风格继承: 留空的格子沿用它前面最近一张非空风格
  let last=''; P.panels.forEach(p=>{ if(p.style&&p.style.trim()) last=p.style.trim(); else if(last) p.style=last; });
}
function ccAddPanel(){ // 新增画布: 默认继承上一张的尺寸/风格/垫图主角
  const P=ST.cc; ccCollectPanels();
  const prev=P.panels[P.panels.length-1];
  P.panels.push({num:P.panels.length+1,scene:'',outfit:'',dialogues:[],
    ref_char:prev?prev.ref_char:0,w:prev?prev.w:1024,h:prev?prev.h:720,style:prev?prev.style:''});
  ccSave(51);
}
function ccDelPanel(i){ ccCollectPanels(); ST.cc.panels.splice(i,1); ST.cc.panels.forEach((p,k)=>p.num=k+1); ccSave(51); }
function ccAddDlg(i){ ccCollectPanels(); ST.cc.panels[i].dialogues.push({who:(ST.cc.chars[0]||{}).name||'',text:''}); ccSave(51); }
function ccDelDlg(i,j){ ccCollectPanels(); ST.cc.panels[i].dialogues.splice(j,1); ccSave(51); }
async function ccStartGen(){ // 校验并启动自定义连载
  const P=ST.cc; ccCollectPanels();
  if(!P.panels.length){ alert('先＋新增画布'); return; }
  if(P.panels.some(p=>!p.scene||!p.scene.trim())){ alert('有格子没写场景描述'); return; }
  await fetch('/api/cc/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({chars:P.chars,panels:P.panels})});
  const r=await (await fetch('/api/cc/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({chars:P.chars,panels:P.panels})})).json();
  if(r.error){ alert(r.error); return; }
  ST.step=52; render();
}
async function ccStop(){ await (await fetch('/api/cc/stop',{method:'POST'})).json(); }
function pickModel(id,name,sec,kind){ ST.model=id; ST.mname=name; ST.msec=sec; ST.kind=kind; if(!ST.items.length) ST.items.push(newItem()); ST.step=3; render(); }
function addCard(){ for(let i=0;i<ST.items.length;i++) collect(i); ST.items.push(newItem()); render(); }
function delCard(i){ if(ST.items.length<=1){ alert('至少留一张'); return; } for(let j=0;j<ST.items.length;j++) collect(j); ST.items.splice(i,1); render(); }
function collect(i){
  const it=ST.items[i]; if(!it) return;
  if($('#pos'+i)) it.pos=$('#pos'+i).value;
  if($('#defpos'+i)) it.defPos=$('#defpos'+i).value;
  if($('#neg'+i)) it.neg=$('#neg'+i).value;
  if($('#str'+i)) it.strength=+$('#str'+i).value;
  if($('#scale'+i)) it.scale=+$('#scale'+i).value;
  if($('#ctype'+i)) it.ctype=$('#ctype'+i).value;
  if($('#brush'+i)) it.brush=+$('#brush'+i).value;
}
function setMode(i){
  collect(i); const it=ST.items[i];
  it.mode=$('#mode'+i).value;
  it.strength={i2i:0.6,upscale:0.35,pose:0.8}[it.mode]||0.6;
  render();
}
function tagRow(i,tags,target){ // 渲染一排可点标签,target='pos'/'neg'
  const cur=((target==='pos'?ST.items[i].pos:ST.items[i].neg)||'').split(',').map(s=>s.trim()).filter(Boolean);
  return `<div class="tags">${tags.map(t=>`<span class="tag${target==='neg'?' neg':''}${cur.includes(t)?' on':''}" onclick="tagToggle(${i},'${t}','${target}')">${t}</span>`).join('')}</div>`;
}
function tagToggle(i,word,target){ // 点标签: 没有就加,有了就取消
  collect(i); const it=ST.items[i];
  const arr=((target==='pos'?it.pos:it.neg)||'').split(',').map(s=>s.trim()).filter(Boolean);
  const k=arr.indexOf(word);
  if(k>=0) arr.splice(k,1); else arr.push(word);
  if(target==='pos') it.pos=arr.join(', '); else it.neg=arr.join(', ');
  render();
}
function sizeRow(i,it){
  let sizeOpts = SIZES.map(s=>`<option value="${s[0]}x${s[1]}" ${it.w===s[0]&&it.h===s[1]?'selected':''}>${s[0]} x ${s[1]} ${s[2]}</option>`).join('');
  const isCust = !SIZES.some(s=>s[0]===it.w&&s[1]===it.h);
  if(isCust) sizeOpts += `<option value="cust" selected>自定义 ${it.w} x ${it.h}</option>`;
  else sizeOpts += `<option value="cust">自定义…</option>`;
  return `<label>尺寸</label><div class="row"><select onchange="sizeSel(${i},this)">${sizeOpts}</select>
    <span id="cust${i}" style="display:${isCust?'inline':'none'}">
      <input type="number" id="cw${i}" value="${it.w}" style="width:70px" min="192" max="1280" onchange="custSize(${i})"> x
      <input type="number" id="ch${i}" value="${it.h}" style="width:70px" min="192" max="1280" onchange="custSize(${i})"></span>
    <span class="szbox" id="szbox${i}" title="按出图比例画的大小示意"></span></div>`;
}
function drawSzBox(i){ const it=ST.items[i]; const b=$('#szbox'+i); if(!b) return;
  const max=96, s=Math.min(max/it.w, max/it.h);
  b.style.width=Math.max(8,Math.round(it.w*s))+'px'; b.style.height=Math.max(8,Math.round(it.h*s))+'px';
  b.textContent=it.w+'×'+it.h;
}
function custSize(i){ const it=ST.items[i];
  it.w=Math.min(1280,Math.max(192,+$('#cw'+i).value||1024));
  it.h=Math.min(1280,Math.max(192,+$('#ch'+i).value||1024));
  drawSzBox(i);
}
function refRow(i,tip){
  const it=ST.items[i];
  const inner = it.refFile
    ? `<b style="color:#18a058">✔ ${it.refFile.name}</b><br><img src="${URL.createObjectURL(it.refFile)}" alt="">`
    : `把图片<b>拖到这里</b>,或<b>点击选择</b><br><span class="small">支持 PNG / JPG / WebP 等常见图片格式</span>`;
  return `<label>${tip}</label>
    <div class="drop" id="drop${i}"
         ondragover="event.preventDefault();this.classList.add('over')"
         ondragleave="this.classList.remove('over')"
         ondrop="dropRef(${i},event)"
         onclick="$('#ref${i}').click()">${inner}</div>
    <input type="file" id="ref${i}" accept="image/*" style="display:none" onchange="refPick(${i},this)">`;
}
function refPick(i,el){ collect(i); setRef(i,el.files[0]||null); }
function dropRef(i,e){
  e.preventDefault();
  const dz=e.currentTarget; dz.classList.remove('over');
  const f=e.dataTransfer.files && e.dataTransfer.files[0];
  if(!f) return;
  if(!f.type.startsWith('image/')){ alert('拖进来的不是图片文件!'); return; }
  collect(i); setRef(i,f);
}
function setRef(i,f){ ST.items[i].refFile=f; ST.items[i].strokes=[]; render(); }
function strRow(i,label,tip){
  const v=ST.items[i].strength;
  return `<div class="row" style="margin-top:6px"><span style="font-size:13px">${label} <b id="strV${i}">${v.toFixed(2)}</b></span>
    <input type="range" id="str${i}" min="0.1" max="0.95" step="0.05" value="${v}" style="flex:1"
           oninput="$('#strV${i}').textContent=(+this.value).toFixed(2)"><span class="small">${tip}</span></div>`;
}
function maskRow(i){
  return `<label>涂抹要重画的区域(红=重画,其余不动)</label><div class="row">
      <span style="font-size:13px">笔刷</span>
      <input type="range" id="brush${i}" min="10" max="80" step="5" value="${ST.items[i].brush}" style="width:120px">
      <button class="back" style="padding:4px 12px;font-size:13px" onclick="clearMask(${i})">清除涂抹</button></div>
    <canvas id="mcanvas${i}" style="max-width:100%;border-radius:8px;border:1px dashed #bbb;cursor:crosshair;touch-action:none"></canvas>`;
}
function initMaskCanvas(i){
  const it=ST.items[i]; const cv=document.getElementById('mcanvas'+i); if(!cv) return;
  if(!it.refFile){ cv.style.display='none'; return; }
  const img=new Image();
  img.onload=()=>{
    const sc=Math.min(1, 780/img.width);
    cv.width=Math.round(img.width*sc); cv.height=Math.round(img.height*sc);
    it._img=img; it._sc=sc; redrawMask(i);
    cv.onpointerdown=e=>{ it._draw=true; it._lx=null; maskDraw(i,e); };
    cv.onpointermove=e=>{ if(it._draw) maskDraw(i,e); };
    cv.onpointerup=cv.onpointerleave=()=>{ it._draw=false; it._lx=null; };
  };
  img.src=URL.createObjectURL(it.refFile);
}
function maskDraw(i,e){
  const it=ST.items[i]; const cv=document.getElementById('mcanvas'+i);
  const r=cv.getBoundingClientRect();
  const x=(e.clientX-r.left)*(cv.width/r.width), y=(e.clientY-r.top)*(cv.height/r.height);
  if(it._lx!==null) it.strokes.push({x0:it._lx,y0:it._ly,x1:x,y1:y});
  else it.strokes.push({x0:x,y0:y,x1:x,y1:y});
  it._lx=x; it._ly=y; redrawMask(i);
}
function redrawMask(i){
  const it=ST.items[i]; const cv=document.getElementById('mcanvas'+i);
  if(!cv||!it._img) return;
  const c=cv.getContext('2d');
  c.drawImage(it._img,0,0,cv.width,cv.height);
  c.strokeStyle='rgba(255,40,40,.75)'; c.lineCap='round'; c.lineJoin='round'; c.lineWidth=it.brush;
  it.strokes.forEach(s=>{ c.beginPath(); c.moveTo(s.x0,s.y0); c.lineTo(s.x1,s.y1); c.stroke(); });
}
function clearMask(i){ ST.items[i].strokes=[]; redrawMask(i); }
function exportMask(i){
  const it=ST.items[i];
  const off=document.createElement('canvas');
  off.width=it._img.width; off.height=it._img.height;
  const c=off.getContext('2d');
  c.fillStyle='#000'; c.fillRect(0,0,off.width,off.height);
  c.strokeStyle='#fff'; c.lineCap='round'; c.lineJoin='round'; c.lineWidth=it.brush/it._sc;
  it.strokes.forEach(s=>{ c.beginPath(); c.moveTo(s.x0/it._sc,s.y0/it._sc); c.lineTo(s.x1/it._sc,s.y1/it._sc); c.stroke(); });
  return new Promise(res=>off.toBlob(res,'image/png'));
}
// ---- 火柴人骨架编辑器(openpose 18 关节,黑底彩条,union controlnet 认这个格式) ----
const POSE_BONES=[[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,0],[0,14],[14,16],[0,15],[15,17]];
const POSE_COLS=['#ff0000','#ff5500','#ffaa00','#ffff00','#aaff00','#55ff00','#00ff00','#00ff55','#00ffaa','#00ffff','#00aaff','#0055ff','#0000ff','#5500ff','#aa00ff','#ff00ff','#ff00aa'];
const POSE_JOINT_NAMES=['鼻','颈','右肩','右肘','右腕','左肩','左肘','左腕','右胯','右膝','右踝','左胯','左膝','左踝','右眼','左眼','右耳','左耳'];
function defaultPose(){ // 站立小人,归一化坐标(0~1)
  return [[.5,.12],[.5,.24],[.40,.26],[.37,.38],[.35,.50],[.60,.26],[.63,.38],[.65,.50],
          [.44,.52],[.44,.68],[.44,.86],[.56,.52],[.56,.68],[.56,.86],
          [.48,.11],[.52,.11],[.46,.11],[.54,.11]];
}
function setPoseSrc(i,s){ collect(i); ST.items[i].poseSrc=s; render(); }
function resetPose(i){ ST.items[i].joints=defaultPose(); drawPose(i); }
function drawPose(i){
  const it=ST.items[i]; const cv=document.getElementById('pcanvas'+i); if(!cv) return;
  if(!it.joints) it.joints=defaultPose();
  const c=cv.getContext('2d');
  c.fillStyle='#000'; c.fillRect(0,0,cv.width,cv.height);
  const lw=Math.max(3,cv.width/110), jr=lw*0.9;
  const pts=it.joints.map(p=>[p[0]*cv.width,p[1]*cv.height]);
  c.lineCap='round';
  POSE_BONES.forEach((b,k)=>{ c.strokeStyle=POSE_COLS[k]; c.lineWidth=lw;
    c.beginPath(); c.moveTo(pts[b[0]][0],pts[b[0]][1]); c.lineTo(pts[b[1]][0],pts[b[1]][1]); c.stroke(); });
  pts.forEach((p,k)=>{ c.fillStyle='#fff'; c.beginPath(); c.arc(p[0],p[1],jr,0,7); c.fill(); });
}
function initPoseCanvas(i){
  const it=ST.items[i]; const cv=document.getElementById('pcanvas'+i); if(!cv) return;
  if(!it.joints) it.joints=defaultPose();
  const sc=Math.min(1, 780/it.w);
  cv.width=Math.round(it.w*sc); cv.height=Math.round(it.h*sc);
  drawPose(i);
  let drag=-1;
  const loc=e=>{ const r=cv.getBoundingClientRect();
    return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]; };
  cv.onpointerdown=e=>{ const [x,y]=loc(e); let best=0.05;
    it.joints.forEach((p,k)=>{ const d=Math.hypot(p[0]-x,(p[1]-y)*cv.height/cv.width); if(d<best){best=d;drag=k;} });
    cv.setPointerCapture(e.pointerId); };
  cv.onpointermove=e=>{ if(drag<0) return; const [x,y]=loc(e);
    it.joints[drag]=[Math.min(1,Math.max(0,x)),Math.min(1,Math.max(0,y))]; drawPose(i); };
  cv.onpointerup=()=>{ drag=-1; };
}
function exportPose(i){ // 按出图分辨率导出黑底骨架 PNG
  const it=ST.items[i]; if(!it.joints) it.joints=defaultPose();
  const off=document.createElement('canvas'); off.width=it.w; off.height=it.h;
  const c=off.getContext('2d');
  c.fillStyle='#000'; c.fillRect(0,0,off.width,off.height);
  const lw=Math.max(4,off.width/110);
  const pts=it.joints.map(p=>[p[0]*off.width,p[1]*off.height]);
  c.lineCap='round';
  POSE_BONES.forEach((b,k)=>{ c.strokeStyle=POSE_COLS[k]; c.lineWidth=lw;
    c.beginPath(); c.moveTo(pts[b[0]][0],pts[b[0]][1]); c.lineTo(pts[b[1]][0],pts[b[1]][1]); c.stroke(); });
  return new Promise(res=>off.toBlob(res,'image/png'));
}
function cardHTML(i){
  const it=ST.items[i];
  let mid='';
  if(it.mode==='t2i') mid=sizeRow(i,it);
  if(it.mode==='i2i') mid=refRow(i,'上传参考图:保持画风构图,按提示词改内容(尺寸跟随参考图)')+strRow(i,'重绘幅度','小=更像原图,大=变化更大');
  if(it.mode==='inpaint') mid=refRow(i,'上传要修改的原图(尺寸跟随原图)')+maskRow(i);
  if(it.mode==='upscale') mid=refRow(i,'上传要放大的图')+
    `<label>放大倍数</label><select id="scale${i}"><option value="1.5" ${it.scale===1.5?'selected':''}>1.5 倍</option><option value="2" ${it.scale===2?'selected':''}>2 倍</option></select>`+
    strRow(i,'重绘幅度','建议0.3~0.5,太大细节会变样');
  if(it.mode==='pose'){
    const srcRow=`<label>骨架从哪来</label><div class="row">
      <label style="margin:0"><input type="radio" name="psrc${i}" ${it.poseSrc==='draw'?'checked':''} onchange="setPoseSrc(${i},'draw')"> 摆火柴人(推荐)</label>
      <label style="margin:0"><input type="radio" name="psrc${i}" ${it.poseSrc!=='draw'?'checked':''} onchange="setPoseSrc(${i},'upload')"> 上传骨架/线稿图</label></div>`;
    if(it.poseSrc==='draw'){
      mid=srcRow+
        `<div class="row"><button class="back" style="padding:4px 12px;font-size:13px" onclick="resetPose(${i})">重置姿势</button>
         <span class="small">拖动圆点摆姿势,黑底彩条就是给 AI 的骨架,不用上传</span></div>
         <canvas id="pcanvas${i}" style="max-width:100%;border-radius:8px;border:1px dashed #bbb;cursor:grab;touch-action:none;background:#000"></canvas>`+
        sizeRow(i,it)+strRow(i,'控制强度','越大越严格照骨架来');
    }else{
      mid=srcRow+refRow(i,'上传骨架图(黑底彩色火柴人)或线稿图')+sizeRow(i,it)+
        `<label>控制类型</label><select id="ctype${i}"><option value="openpose" ${it.ctype==='openpose'?'selected':''}>骨架图(openpose)</option><option value="hed/pidi/scribble/ted" ${it.ctype==='hed/pidi/scribble/ted'?'selected':''}>涂鸦/线稿(scribble)</option></select>`+
        strRow(i,'控制强度','越大越严格照图来');
    }
  }
  const posLabel={inpaint:'正向提示词:涂抹区域里要画什么(可留空)',upscale:'正向提示词(可留空)',pose:'正向提示词:想要什么画面(可留空)'}[it.mode]||'正向提示词:想要什么画面(可留空)';
  return `<div class="card" style="cursor:default">${it.resultUrl?`<img class="cardthumb" src="${it.resultUrl}" title="上次生成的图">`:''}<b>第 ${i+1} 张</b>${ST.items.length>1?`<span class="delcard" onclick="delCard(${i})" title="删掉这张">✕</span>`:''}
    <label>玩法</label><select id="mode${i}" onchange="setMode(${i})">
      <option value="t2i" ${it.mode==='t2i'?'selected':''}>普通文生图</option>
      <option value="i2i" ${it.mode==='i2i'?'selected':''}>以图生图(参考画风)</option>
      <option value="inpaint" ${it.mode==='inpaint'?'selected':''}>局部重绘(涂哪改哪)</option>
      <option value="upscale" ${it.mode==='upscale'?'selected':''}>放大变清晰</option>
      ${(ST.kind==='checkpoint'&&ST.hasCN)?`<option value="pose" ${it.mode==='pose'?'selected':''}>姿势控制(骨架/线稿)</option>`:''}
    </select>${mid}
    <label>默认提示词(质量词,自动加在最前面,可改可清空)</label><textarea id="defpos${i}" style="height:38px">${it.defPos}</textarea>
    <label>${posLabel} — 点下面标签快速加</label>${tagRow(i,POS_TAGS,'pos')}<textarea id="pos${i}" placeholder="不填也行,比如: a cat, 1girl, school uniform">${it.pos}</textarea>
    <label>负向提示词:不想要什么(可留空,有默认值) — 点下面标签快速加</label>${tagRow(i,NEG_TAGS,'neg')}<textarea id="neg${i}">${it.neg}</textarea></div>`;
}
function sizeSel(i, sel){
  $('#cust'+i).style.display = sel.value==='cust'?'inline':'none';
  if(sel.value!=='cust'){ const [w,h]=sel.value.split('x').map(Number); ST.items[i].w=w; ST.items[i].h=h; }
  drawSzBox(i);
}
async function uploadFile(file, fname){
  const fd=new FormData(); fd.append('image', file, fname||file.name); fd.append('overwrite','true');
  const r=await fetch('/api/upload',{method:'POST',body:fd}); const d=await r.json();
  if(d.error) throw new Error('上传失败: '+d.error);
  return d.name;
}
async function startGen(){
  for(let i=0;i<ST.items.length;i++){
    const it=ST.items[i]; collect(i);
    if($('#cust'+i) && $('#cust'+i).style.display!=='none'){
      it.w=Math.min(1280,Math.max(192,+$('#cw'+i).value||1024));
      it.h=Math.min(1280,Math.max(192,+$('#ch'+i).value||1024));
    }
    // 正向/负向都可留空: 默认提示词自动拼在最前,全空才兜底
    const dp=(it.defPos||'').trim(), up=it.pos.trim();
    it.pos=dp?(up?dp+', '+up:dp):up;
    if(!it.pos) it.pos='masterpiece, best quality';
    if(!it.neg.trim()) it.neg=NEG_DEF;
    if(it.mode!=='t2i' && !(it.mode==='pose'&&it.poseSrc==='draw') && !it.refFile){ alert('第 '+(i+1)+' 张还没上传图片!'); return; }
    if(it.mode==='inpaint' && !it.strokes.length){ alert('第 '+(i+1)+' 张还没涂抹要重画的区域!'); return; }
    try{
      it.ref=null; it.mask=null;
      if(it.mode==='pose'&&it.poseSrc==='draw'){
        it.ctype='openpose';
        const blob=await exportPose(i); it.ref=await uploadFile(blob,'pose-'+Date.now()+'-'+i+'.png');
      }else if(it.mode!=='t2i') it.ref=await uploadFile(it.refFile);
      if(it.mode==='inpaint'){ const blob=await exportMask(i); it.mask=await uploadFile(blob,'mask-'+Date.now()+'-'+i+'.png'); }
    }catch(e){ alert('第 '+(i+1)+' 张: '+e.message); return; }
  }
  ST.pids=[];
  for(let i=0;i<ST.items.length;i++){
    const it=ST.items[i];
    const name=`img-${Date.now()}-${String(i+1).padStart(2,'0')}`;
    const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:ST.model,mode:it.mode,pos:it.pos,neg:it.neg,w:it.w,h:it.h,name,
        ref:it.ref,mask:it.mask,strength:it.strength,scale:it.scale,ctype:it.ctype})});
    if(r.error){ alert('提交失败: '+r.error); return; }
    ST.pids.push(r.pid);
  }
  ST.step=4; render();
}
async function pollAll(){
  let left=ST.items.length;
  ST.pids.forEach(async (pid,i)=>{
    while(true){
      await new Promise(r=>setTimeout(r,2000));
      let r; try{ r=await j('/api/poll?pid='+pid); }catch(e){ continue; }
      if(r.done){ $('#b'+i).style.width='100%'; $('#st'+i).textContent='✅ 完成';
        ST.items[i].resultUrl=r.url;
        $('#img'+i).innerHTML=`<img class="out" src="${r.url}">`;
        if($('#mask').checked) $('#img'+i+' img').classList.add('mask');
        $('#acts'+i).style.display='flex';
        left--; if(left<=0) $('#again').style.display='block'; return; }
      if(r.error){ $('#b'+i).style.background='#d03050'; $('#st'+i).textContent='❌ '+r.error;
        left--; if(left<=0) $('#again').style.display='block'; return; }
      if(r.state==='running'){
        const el=Math.round(r.run_elapsed);
        const pct=Math.min(95, Math.round(r.run_elapsed/ST.msec*100));
        $('#b'+i).style.width=pct+'%';
        $('#st'+i).textContent=`🎨 生成中… ${el}秒(约${ST.msec}秒)`;
      }else{
        $('#b'+i).style.width='0%';
        $('#st'+i).textContent=`⏳ 排队中(前面还有${r.pos}张)`;
      }
    }
  });
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function viewPrompt(i){ // 结果页: 查看这张图用的提示词(只读)
  const it=ST.items[i];
  $('#pe'+i).innerHTML=`<div class="pbox"><b>正向:</b><div class="small">${esc(it.pos||'(空)')}</div>
    <b style="margin-top:6px;display:inline-block">负向:</b><div class="small">${esc(it.neg||'(空)')}</div>
    <div class="row" style="margin-top:6px"><button class="back" onclick="$('#pe${i}').innerHTML=''">收起</button></div></div>`;
}
function editPrompt(i){ // 结果页: 编辑提示词,同尺寸重新生成
  const it=ST.items[i];
  $('#pe'+i).innerHTML=`<div class="pbox"><b>正向提示词(改完点重新生成,尺寸不变):</b>
    <textarea id="pepos${i}" rows="2">${esc(it.pos)}</textarea>
    <b>负向提示词:</b><textarea id="peneg${i}" rows="2">${esc(it.neg)}</textarea>
    <div class="row" style="margin-top:6px">
      <button onclick="regenOne(${i})">🎨 按新提示词重新生成这张</button>
      <button class="back" onclick="$('#pe${i}').innerHTML=''">取消</button></div></div>`;
}
async function regenOne(i){ // 只重生成第 i 张(同尺寸同模式,只换提示词)
  const it=ST.items[i];
  const posEl=$('#pepos'+i), negEl=$('#peneg'+i);
  if(posEl) it.pos=posEl.value.trim();
  if(negEl) it.neg=negEl.value.trim();
  if(!it.pos) it.pos='masterpiece, best quality';
  if(!it.neg) it.neg=NEG_DEF;
  try{
    it.ref=null; it.mask=null;
    if(it.mode==='pose'&&it.poseSrc==='draw'){
      it.ctype='openpose';
      const blob=await exportPose(i); it.ref=await uploadFile(blob,'pose-'+Date.now()+'-'+i+'.png');
    }else if(it.mode!=='t2i'&&it.refFile) it.ref=await uploadFile(it.refFile);
    if(it.mode==='inpaint'&&it.strokes.length){ const blob=await exportMask(i); it.mask=await uploadFile(blob,'mask-'+Date.now()+'-'+i+'.png'); }
  }catch(e){ alert('第 '+(i+1)+' 张: '+e.message); return; }
  const name=`img-${Date.now()}-${String(i+1).padStart(2,'0')}`;
  const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:ST.model,mode:it.mode,pos:it.pos,neg:it.neg,w:it.w,h:it.h,name,
      ref:it.ref,mask:it.mask,strength:it.strength,scale:it.scale,ctype:it.ctype})});
  if(r.error){ alert('提交失败: '+r.error); return; }
  $('#pe'+i).innerHTML=''; $('#acts'+i).style.display='none'; $('#img'+i).innerHTML='';
  $('#b'+i).style.width='0%'; $('#st'+i).textContent='排队中…';
  pollOne(i,r.pid);
}
async function pollOne(i,pid){ // 单张轮询(重生成用)
  while(true){
    await new Promise(r=>setTimeout(r,2000));
    let r; try{ r=await j('/api/poll?pid='+pid); }catch(e){ continue; }
    if(r.done){ $('#b'+i).style.width='100%'; $('#st'+i).textContent='✅ 完成';
      ST.items[i].resultUrl=r.url;
      $('#img'+i).innerHTML=`<img class="out" src="${r.url}">`;
      if($('#mask')&&$('#mask').checked) $('#img'+i+' img').classList.add('mask');
      $('#acts'+i).style.display='flex'; return; }
    if(r.error){ $('#b'+i).style.background='#d03050'; $('#st'+i).textContent='❌ '+r.error; return; }
    if(r.state==='running'){
      const el=Math.round(r.run_elapsed);
      const pct=Math.min(95, Math.round(r.run_elapsed/ST.msec*100));
      $('#b'+i).style.width=pct+'%';
      $('#st'+i).textContent=`🎨 生成中… ${el}秒(约${ST.msec}秒)`;
    }else{
      $('#b'+i).style.width='0%';
      $('#st'+i).textContent=`⏳ 排队中(前面还有${r.pos}张)`;
    }
  }
}
function editRound(){ ST.step=3; render(); } // 回到设置页,卡片/缩略图/骨架都在,可接着改
function newRound(){ // 同模型全新一轮(保留刚生成的缩略图在首页卡片上)
  for(let i=0;i<ST.items.length;i++) collect(i);
  const thumbs=ST.items.map(it=>it.resultUrl).filter(Boolean);
  ST.items=[newItem()]; ST.items[0].resultUrl=thumbs[thumbs.length-1]||null;
  ST.step=3; render();
}
function toggleMask(on){ document.querySelectorAll('img.out').forEach(im=>im.classList.toggle('mask',on)); }
// 死亡检测: 在服务页时每 5s 查一次,发现进程被杀(关网页不影响后台,但 kill 进程)就提示并回首页
setInterval(async()=>{
  if(![1,3,4,10,11,22,31,41,52].includes(ST.step)) return;
  let st; try{ st=await j('/api/svc/state'); }catch(e){ return; }
  let dead=null;
  if([1,3,4,31,41,52].includes(ST.step) && !st.img.alive_pid) dead='生图服务';
  else if([10,11].includes(ST.step) && !st.vid.alive_pid) dead='生视频服务';
  else if(ST.step===22 && !ST.llmPaused && !st.llm.alive_pid) dead='语言模型';
  if(dead){ alert('⚠ '+dead+'已停止(进程被关闭)。'); home(); }
},5000);
// 启动时按当前运行状态恢复页面: llm 在跑→语言页, img 在跑→生图, vid 在跑→视频, 否则首页
(async()=>{
  try{
    const bs=await j('/api/batch/status');
    if(bs && bs.running){ ST.step=31; render(); return; } // 批量在跑→直接回进度页
    const cs=await j('/api/comic/status');
    if(cs && cs.running){ ST.step=41; render(); return; } // 连载在跑→回连载进度页
    const ccs=await j('/api/cc/status');
    if(ccs && ccs.running){ ST.step=52; render(); return; } // 自定义连载在跑→回进度页
    const st=await j('/api/svc/state');
    if(st.llm.alive_pid){ ST.step=22; render(); return; }
    const lc=await j('/api/llm/stats');
    if(lc.paused){ ST.llmPaused=true; ST.step=22; render(); return; }
    if(st.img.running){ ST.step=1; render(); return; }
    if(st.vid.running){ ST.step=10; render(); return; }
  }catch(e){}
  home();
})();
</script></body></html>""".replace("%NEG%", json.dumps(NEG_DEFAULT)).replace("%VID%", str(VID_PORT))

# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _qarg(self, key, default=""):
        """从 query string 取一个参数(?type=vid → 'vid')。"""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get(key, [default])[0]
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            ms = list_models()
            self._send(200, json.dumps({
                "comfy_installed": bool(CFG.get("comfy_dir")),
                "img_running": port_running(IMG_PORT),
                "vid_running": port_running(VID_PORT),
                "has_model": len(ms) > 0,
                "model_count": len(ms),
                "has_cn": has_controlnet(),
            }))
        elif self.path == "/api/models":
            self._send(200, json.dumps(list_models()))
        elif self.path == "/api/start":
            self._send(200, json.dumps(start_comfy()))
        elif self.path == "/api/svc/state":
            # 一把抓三类服务状态(网页加载时据此恢复到对应页面)
            self._send(200, json.dumps(svc.all_status()))
        elif self.path == "/api/vid/models":
            self._send(200, json.dumps({"unets": vidwf.list_unets(), "loras": vidwf.list_loras()}))
        elif self.path == "/api/llm/models":
            self._send(200, json.dumps(llm.list_models()))
        elif self.path == "/api/llm/current":
            self._send(200, json.dumps(llm.current()))
        elif self.path == "/api/llm/stats":
            self._send(200, json.dumps(llm.stats()))
        elif self.path == "/api/batch/config":
            self._send(200, json.dumps(batch_config()))
        elif self.path == "/api/batch/status":
            self._send(200, json.dumps(BATCH))
        elif self.path == "/api/comic/story":
            self._send(200, json.dumps(comic_story()))
        elif self.path == "/api/comic/status":
            self._send(200, json.dumps(COMIC))
        elif self.path == "/api/cc/project":
            self._send(200, json.dumps(cc_project()))
        elif self.path == "/api/cc/status":
            self._send(200, json.dumps({k: CC[k] for k in
                                        ("running", "done", "total", "ok", "fail", "current",
                                         "log", "finished", "last", "sheet")}))
        elif self.path.startswith("/api/vid/poll"):
            self._send(200, json.dumps(vidwf.poll_vid(self._qarg("pid"))))
        elif self.path.startswith("/api/svc/status"):
            t = self._qarg("type", "img")
            self._send(200, json.dumps(svc.svc_status(t)))
        elif self.path.startswith("/api/svc/start"):
            t = self._qarg("type", "img")
            self._send(200, json.dumps(svc.start_svc(t)))
        elif self.path.startswith("/api/svc/stop"):
            t = self._qarg("type", "img")
            self._send(200, json.dumps(svc.stop_svc(t)))
        elif self.path.startswith("/api/poll"):
            pid = self.path.split("pid=")[-1]
            with LOCK: t = TASKS.get(pid)
            if not t:
                self._send(404, json.dumps({"error": "未知任务"}))
            elif t["done"] or t["error"]:
                self._send(200, json.dumps({"done": t["done"], "error": t["error"], "url": t["url"]}))
            else:
                state, pos, run_elapsed = "queued", 0, 0
                try:
                    q = comfy_get("/queue")
                    run_ids = [e[1] for e in q.get("queue_running", [])]
                    pen_ids = [e[1] for e in q.get("queue_pending", [])]
                    if pid in run_ids:
                        state = "running"
                        with LOCK:
                            if not t.get("t_start"): t["t_start"] = time.time()
                            run_elapsed = time.time() - t["t_start"]
                    elif pid in pen_ids:
                        pos = pen_ids.index(pid) + 1
                except Exception:
                    pass
                self._send(200, json.dumps({"done": False, "error": "", "url": "",
                                            "state": state, "pos": pos, "run_elapsed": run_elapsed}))
        elif self.path.startswith("/out/"):
            fp = os.path.join(OUT_DIR, os.path.basename(self.path[5:]))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "image/png")
            else: self._send(404, "{}")
        elif self.path.startswith("/vout/"):
            fp = os.path.join(vidwf.OUT_VID, os.path.basename(self.path[6:]))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "video/mp4")
            else: self._send(404, "{}")
        elif self.path.startswith("/comic_out/"):
            fp = os.path.join(COMIC_OUT, os.path.basename(self.path[11:].split("?")[0]))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "image/png")
            else: self._send(404, "{}")
        elif self.path.startswith("/cc_out/"):
            fp = os.path.join(CC_OUT, os.path.basename(self.path[8:].split("?")[0]))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "image/png")
            else: self._send(404, "{}")
        else:
            self._send(404, "{}")
    def do_POST(self):
        if self.path == "/api/gen":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pid = submit(d["model"], d["pos"], d["neg"], int(d["w"]), int(d["h"]), d["name"],
                             d.get("mode", "t2i"), d.get("ref"), d.get("mask"),
                             float(d.get("strength", 0.6)), float(d.get("scale", 2.0)), d.get("ctype", "openpose"))
                self._send(200, json.dumps({"pid": pid}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/upload":
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}/upload/image", data=body,
                                             headers={"Content-Type": self.headers["Content-Type"]})
                self._send(200, urllib.request.urlopen(req, timeout=120).read())
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/vid/upload":
            # 视频源图必须传到生视频实例(8850),不能复用 8849 的 /api/upload
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                req = urllib.request.Request(f"http://127.0.0.1:{VID_PORT}/upload/image", data=body,
                                             headers={"Content-Type": self.headers["Content-Type"]})
                self._send(200, urllib.request.urlopen(req, timeout=120).read())
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/vid/gen":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pid = vidwf.submit_vid(d["name"], unet_id=d["unet"], pos=d.get("pos", ""), neg=d.get("neg", ""),
                                       image_name=d["image"], w=int(d["w"]), h=int(d["h"]),
                                       frames=int(d["frames"]), fps=float(d.get("fps", 24)),
                                       lora_id=d.get("lora", "none"), lora_strength=float(d.get("lora_strength", 0.8)),
                                       use_stg=bool(d.get("stg", False)), steps=int(d.get("steps", 8)))
                self._send(200, json.dumps({"pid": pid}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/llm/start":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r = llm.start_llm(d["id"], bool(d.get("thinking", True)),
                                  float(d.get("temp", 0.7)), int(d.get("max_tokens", 16384)))
                self._send(200, json.dumps(r))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/llm/pause":
            self._send(200, json.dumps(llm.pause_llm()))
        elif self.path == "/api/llm/resume":
            self._send(200, json.dumps(llm.resume_llm()))
        elif self.path == "/api/batch/start":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                # models 已是网页组织好的嵌套结构 [{id,name,styles:[{name,prompts,ref}]}]
                self._send(200, json.dumps(batch_start(d.get("models", []),
                                                       d.get("neg", NEG_DEFAULT),
                                                       int(d.get("w", 1024)), int(d.get("h", 720)))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/batch/import":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(batch_import(d.get("folder", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/batch/stop":
            BATCH["stop"] = True
            self._send(200, json.dumps({"ok": True}))
        elif self.path == "/api/comic/start":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                panels = [{"num": int(p["num"]), "prompt": p["prompt"], "dialogue": p.get("dialogue", "")}
                          for p in d.get("panels", []) if p.get("prompt", "").strip()]
                self._send(200, json.dumps(comic_start(d["model"], panels, d.get("ref") or None,
                                                       float(d.get("strength", 0.75)),
                                                       int(d.get("w", 1024)), int(d.get("h", 720)))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/comic/stop":
            COMIC["stop"] = True
            self._send(200, json.dumps({"ok": True}))
        elif self.path == "/api/cc/save":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(cc_save(d.get("chars", []), d.get("panels", []))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cc/sheet":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(cc_sheet_gen(d.get("role", "male"),
                                                        d.get("desc", ""), d.get("ref") or None)))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cc/start":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                panels = [p for p in d.get("panels", []) if p.get("scene", "").strip()]
                self._send(200, json.dumps(cc_start(d.get("chars", []), panels)))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cc/stop":
            CC["stop"] = True
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, "{}")

def start_comfy():
    """调用 start.sh 后台启动 ComfyUI(若已装)。"""
    if not CFG.get("comfy_dir"):
        return {"ok": False, "error": "未安装 ComfyUI"}
    script = os.path.join(BASE, "start.sh")
    if not os.path.exists(script):
        return {"ok": False, "error": "缺少 start.sh"}
    try:
        subprocess.Popen(["bash", script], cwd=BASE,
                         stdout=open(os.path.join(BASE, "comfy.log"), "ab"),
                         stderr=subprocess.STDOUT)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    for d in (MODELS_IMG, MODELS_CN, MODELS_VID, OUT_DIR, COMFY_OUT):
        os.makedirs(d, exist_ok=True)
    if not CFG.get("comfy_dir"):
        print("⚠ 未检测到 ComfyUI,网页会提示你先运行 ./install.sh")
    # 若我们自己的服务已在跑,直接开网页退出,不再起第二个(避免 Address already in use)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{SELF_PORT}/api/svc/state", timeout=1)
        already = True
    except Exception:
        already = False
    if already:
        print(f"🎨 ImageVideoStudio 已在运行: http://127.0.0.1:{SELF_PORT}")
        if "--no-browser" not in sys.argv:
            webbrowser.open(f"http://127.0.0.1:{SELF_PORT}")
        sys.exit(0)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", SELF_PORT), H)
    except OSError as e:
        print(f"❌ 端口 {SELF_PORT} 被其他程序占用({e})。")
        print(f"   若是旧实例:打开 http://127.0.0.1:{SELF_PORT} 即可;要重启先执行  lsof -ti:{SELF_PORT} | xargs kill")
        sys.exit(1)
    print(f"🎨 ImageVideoStudio 已启动: http://127.0.0.1:{SELF_PORT}  (Ctrl+C 退出)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{SELF_PORT}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
