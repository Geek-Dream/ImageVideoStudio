#!/usr/bin/env python3
# ============================================================
# ImageVideoStudio · 生图/生视频小助手(网页版)
# 用法: python3 gen.py   然后浏览器打开 http://127.0.0.1:8860
# 依赖: 仅 Python3 标准库,无需 pip 安装任何东西
# 原理: 本程序只是一个"好看的操作台",真正画图的是 ComfyUI。
#        第一次用请先运行 ./install.sh 装好 ComfyUI,再 ./start.sh 启动。
# ============================================================
import json, os, shutil, sys, time, threading, urllib.request, webbrowser, subprocess, platform
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

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
        return ({"100": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": e["unet"]}},
                 "101": {"class_type": e.get("clip_loader", "CLIPLoaderGGUF"),
                         "inputs": {"clip_name": e["clip"], "type": e["clip_type"]}},
                 "102": {"class_type": "VAELoader", "inputs": {"vae_name": e["vae"]}}},
                ["101", 0], ["102", 0], ["100", 0])
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

# ---------------- 网页 ----------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>ImageVideoStudio · 生图/生视频小助手</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:880px;margin:0 auto;padding:28px 16px 60px;
  background:linear-gradient(160deg,#eef1f8 0%,#f7f5fa 45%,#f2f7f5 100%);color:#23262e;min-height:100vh}
h1{font-size:24px;text-align:center;margin:6px 0 4px;letter-spacing:.5px;
  background:linear-gradient(90deg,#4f7cff,#9b5cff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
h2{font-size:16px;margin:26px 0 12px;color:#444a57;display:flex;align-items:center;gap:8px}
h2::before{content:'';width:4px;height:16px;border-radius:2px;background:linear-gradient(180deg,#4f7cff,#9b5cff)}
.card{background:#fff;border:1px solid #e8e9f0;border-radius:14px;padding:16px 18px;margin:12px 0;cursor:pointer;
  box-shadow:0 1px 3px rgba(60,70,120,.06);transition:transform .15s,box-shadow .15s,border-color .15s}
.card:hover{border-color:#4f7cff;box-shadow:0 6px 20px rgba(80,100,220,.13);transform:translateY(-2px)}
.card.sel{border-color:#4f7cff;background:#f2f5ff}
.card.dis{opacity:.45;cursor:not-allowed}
.card.dis:hover{transform:none;box-shadow:0 1px 3px rgba(60,70,120,.06)}
.dot{font-size:12px}.on{color:#18a058}.off{color:#999}
button{background:linear-gradient(90deg,#4f7cff,#6a5cff);color:#fff;border:0;border-radius:10px;padding:11px 26px;
  font-size:15px;cursor:pointer;box-shadow:0 3px 10px rgba(90,110,255,.28);transition:filter .15s,transform .1s}
button:hover{filter:brightness(1.08)} button:active{transform:scale(.97)}
button:disabled{background:#aab;box-shadow:none}
.back{background:#eceef4;color:#444a57;box-shadow:none}
.back:hover{background:#e0e3ec}
textarea,input[type=text],input[type=number]{width:100%;box-sizing:border-box;border:1px solid #d8dae4;border-radius:8px;
  padding:9px 11px;font-size:14px;font-family:inherit;background:#fbfbfe;transition:border-color .15s,box-shadow .15s}
textarea:focus,input:focus,select:focus{outline:0;border-color:#4f7cff;box-shadow:0 0 0 3px rgba(79,124,255,.15)}
textarea{height:56px;resize:vertical}
select{border:1px solid #d8dae4;border-radius:8px;padding:7px 10px;font-size:14px;background:#fbfbfe;cursor:pointer}
label{font-size:13px;color:#5a6070;display:block;margin:10px 0 4px;font-weight:500}
.bar{height:10px;background:#ecedf3;border-radius:6px;overflow:hidden;margin-top:10px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#4f7cff,#9b5cff);width:0;transition:width .5s;border-radius:6px}
img.out{max-width:100%;border-radius:12px;margin-top:12px;box-shadow:0 4px 18px rgba(50,60,110,.14)}
img.mask{filter:blur(18px)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.small{font-size:12px;color:#9aa0ae}
.drop{border:2px dashed #c6c9d6;border-radius:12px;padding:18px;text-align:center;color:#9aa0ae;font-size:13px;
  cursor:pointer;background:#fafbfe;transition:.15s}
.drop:hover{border-color:#8fa5ff;color:#5a70c0}
.drop.over{border-color:#4f7cff;background:#eef3ff;color:#4f7cff;border-style:solid}
.drop img{max-width:100%;max-height:150px;border-radius:8px;margin-top:10px}
.box{background:#fff9ec;border:1px solid #ffe1a8;border-radius:14px;padding:16px 18px;margin:12px 0;line-height:1.7}
code{background:#eef0f6;border-radius:5px;padding:2px 7px;font-size:13px;color:#4a5fc1}
input[type=range]{accent-color:#4f7cff}
input[type=radio],input[type=checkbox]{accent-color:#4f7cff;width:15px;height:15px}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 6px}
.tag{padding:3px 11px;border-radius:999px;background:#eef2ff;border:1px solid #d3dcee;font-size:12px;
  cursor:pointer;user-select:none;color:#3b5bdb;transition:.15s}
.tag:hover{background:#dbe4ff;transform:translateY(-1px)}
.tag.on{background:#4f7cff;color:#fff;border-color:#4f7cff}
.tag.neg.on{background:#e0556b;border-color:#e0556b}
.delcard{float:right;color:#d03050;cursor:pointer;font-weight:700;padding:0 6px;font-size:16px;line-height:1}
.delcard:hover{color:#ff2d55}
.szbox{display:inline-flex;align-items:center;justify-content:center;border:1.5px solid #4f7cff;border-radius:4px;
  background:#eef3ff;color:#4a5fc1;font-size:10px;margin-left:10px;vertical-align:middle;overflow:hidden;text-align:center;line-height:1.1}
.mini{background:#eceef4;color:#444a57;box-shadow:none;padding:7px 13px;font-size:13px;border-radius:8px}
.mini:hover{background:#e0e3ec}
.pbox{background:#f7f8fc;border:1px solid #e6e8f0;border-radius:10px;padding:10px 12px;margin-top:10px;
  font-size:13px;line-height:1.6;word-break:break-word}
.cardthumb{float:right;width:84px;height:84px;object-fit:cover;border-radius:8px;margin:0 0 6px 10px;
  box-shadow:0 2px 8px rgba(50,60,110,.18)}
/* 速度指示:wifi 三竖条,绿=快/黄=一般/红=慢 */
.wifi{float:right;display:inline-flex;align-items:flex-end;gap:2px;height:18px;margin:2px 0 0 8px}
.wifi i{width:5px;border-radius:1.5px;background:#dfe3ec}
.wifi i:nth-child(1){height:7px}.wifi i:nth-child(2){height:13px}.wifi i:nth-child(3){height:18px}
.wifi.fast i{background:#22c55e}
.wifi.mid i:nth-child(-n+2){background:#eab308}
.wifi.slow i:nth-child(1){background:#ef4444}
.mfield{display:inline-block;margin:4px 0 2px;padding:2px 9px;border-radius:999px;font-size:12px;
  background:#eef2ff;color:#3b5bdb;border:1px solid #d3dcee}
.mmem{margin-top:6px;font-size:12px;color:#8a5a00;background:#fff7e8;border:1px solid #f0e0b8;
  border-radius:7px;padding:4px 9px;display:inline-block}
</style></head><body>
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

async function home(){
  ST = {step:0, model:null, mname:'', msec:0, items:[], pids:[], hasCN:ST.hasCN};
  const s = await j('/api/status');
  if(!s.comfy_installed){
    $('#app').innerHTML = `<div class="box"><b>😅 还没检测到 ComfyUI</b><br><br>
      ImageVideoStudio 本身不画图,真正干活的是开源的 ComfyUI。请二选一:<br><br>
      <b>① 自动安装(推荐)</b>:打开终端,进入本项目目录,运行 <code>./install.sh</code><br>
      <b>② 你电脑里已经装过</b>:把 ComfyUI 文件夹路径填进 <code>config.json</code> 的 <code>comfy_dir</code>,或设置环境变量 <code>COMFYUI_DIR</code><br><br>
      装好后点下方按钮刷新。</div>
      <p><button onclick="home()">🔄 重新检测</button></p>`;
    return;
  }
  let imgCard;
  if(!s.img_running){
    imgCard = `<div class="card" onclick="startSvc()"><b>🖼️ 图片生成</b>
      <span class="dot off">○ 生图服务未启动</span>
      <div class="small">点这里自动启动(或在终端跑 ./start.sh)</div></div>`;
  } else if(!s.has_model){
    imgCard = `<div class="box"><b>🖼️ 生图服务已运行,但还没检测到图片模型</b><br><br>
      请把一个<b>图片模型文件</b>(.safetensors)放进 <code>models/image/</code> 文件夹,
      不知道该下哪个就看该文件夹里的 <code>README.html</code>(里面有推荐和下载链接)。<br>
      放好后点下方刷新。</div>
      <p><button onclick="home()">🔄 重新检测</button></p>`;
  } else {
    imgCard = `<div class="card" onclick="pickImg()"><b>🖼️ 图片生成</b>
      <span class="dot on">● 服务运行中,已检测到 ${s.model_count} 个模型</span></div>`;
  }
  $('#app').innerHTML = `<h2>第一步:生成图片还是视频?</h2>` + imgCard + `
    <div class="card ${s.vid_running?'':'dis'}" onclick="${s.vid_running?'pickVid()':''}">
      <b>🎬 视频生成</b> <span class="dot ${s.vid_running?'on':'off'}">${s.vid_running?'● 服务运行中':'○ 服务未启动(见 models/video/README.html)'}</span>
      <div class="small">视频参数多,点这个会打开 ComfyUI 网页操作</div></div>`;
}
async function startSvc(){
  $('#app').innerHTML = `<div class="box">🚀 正在启动生图服务(第一次要加载模型,等 1~2 分钟)…</div>`;
  await j('/api/start');
  let n=0;
  const t=setInterval(async()=>{
    const s=await j('/api/status'); n++;
    if(s.img_running){ clearInterval(t); home(); }
    else if(n>60){ clearInterval(t); $('#app').innerHTML='<div class="box">启动超时,请在终端手动跑 ./start.sh 看报错。</div><p><button onclick="home()">返回</button></p>'; }
  },3000);
}
function pickImg(){ ST.step=1; render(); }
function pickVid(){ window.open('http://127.0.0.1:'+%VID%); }

async function render(){
  if(ST.step===1){
    const ms = await j('/api/models');
    ST.hasCN = (await j('/api/status')).has_cn;
    $('#app').innerHTML = `<h2>第二步:选生图模型</h2>
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
}
function newItem(){ return {mode:'t2i',w:1024,h:576,pos:'',neg:NEG_DEF,defPos:'masterpiece, best quality',strength:0.6,scale:2,ctype:'openpose',refFile:null,strokes:[],brush:30,poseSrc:'draw',joints:null,resultUrl:null}; }
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
home();
</script></body></html>""".replace("%NEG%", json.dumps(NEG_DEFAULT)).replace("%VID%", str(VID_PORT))

# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
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
    srv = ThreadingHTTPServer(("127.0.0.1", SELF_PORT), H)
    print(f"🎨 ImageVideoStudio 已启动: http://127.0.0.1:{SELF_PORT}  (Ctrl+C 退出)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{SELF_PORT}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
