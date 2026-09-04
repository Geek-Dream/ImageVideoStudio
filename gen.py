#!/usr/bin/env python3
# ============================================================
# ImageVideoStudio · 生图/生视频小助手(网页版)
# 用法: python3 gen.py   然后浏览器打开 http://127.0.0.1:8860
# 依赖: 仅 Python3 标准库,无需 pip 安装任何东西
# 原理: 本程序只是一个"好看的操作台",真正画图的是 ComfyUI。
#        第一次用请先运行 ./install.sh 装好 ComfyUI,再 ./start.sh 启动。
# ============================================================
import json, os, re, shutil, sys, time, threading, urllib.request, urllib.parse, urllib.error, webbrowser, subprocess, platform
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# 本模块所有 urllib 调用都只打 127.0.0.1(ComfyUI/tts/自检),一律不走代理。
# 否则终端挂 http_proxy 时,urllib 连本地也绕去代理,代理对关闭端口迟迟不拒,
# 每次白等满超时——首页 /api/tts/state 卡2秒(用户感知4秒)就是这个。装空 ProxyHandler 全局禁用。
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

import svc  # 三类服务(生图/生视频/语言)统一启停与状态;svc 不回 import 本模块,无循环
import vidwf  # 视频 I2V 工作流(打 8850);同样不回 import 本模块
import llm   # 语言模型启动/参数记忆/GPU 上限;llm→svc,svc 仅函数内懒加载 llm,无环
import comic_engine as ce  # 漫画引擎(画风预设/解析/四型气泡/条漫拼接);纯 PIL+stdlib,无环

BASE = os.path.dirname(os.path.abspath(__file__))          # 项目目录(gen.py 所在)
CONFIG_FILE = os.path.join(BASE, "config.json")            # 首跑自动生成的配置
MODELS_IMG = os.path.join(BASE, "models", "image")         # 图片模型放这里
MODELS_CN  = os.path.join(MODELS_IMG, "controlnet")        # 图片辅助模型(ControlNet)放这里
MODELS_IPA = os.path.join(MODELS_IMG, "ipadapter")         # IPAdapter 角色一致性模型放这里
MODELS_CV  = os.path.join(MODELS_IMG, "clip_vision")       # CLIP 视觉编码器(IPAdapter 的眼睛)放这里
MODELS_VID = os.path.join(BASE, "models", "video")         # 视频模型放这里
OUT_DIR    = os.path.join(BASE, "output", "images")        # 成品图(网页可直接看)
HQ_OUT     = os.path.join(BASE, "output", "high_quality") # 高质量工作流归档
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
        # GPT-OSS Metal 独立环境(源码、虚拟环境、权重均在项目 metal/ 下)
        "metal_repo": "metal/gpt-oss",
        "metal_venv": "metal/.venv",
        "metal_checkpoint": "metal/models/gpt-oss-20b/metal/model.bin",
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
SD35_FILE = "sd3.5_medium.safetensors"               # SD3.5 Medium 放在 diffusion/ 下
SD35_CLIPS = ("clip_l.safetensors", "clip_g.safetensors", "t5xxl_fp8_e4m3fn.safetensors")
# SD3.5 的 VAE 不可使用 Flux 的 split_files/vae/ae.safetensors；这里使用从
# sd3.5_medium.safetensors 提取出的专用 VAE。
SD35_VAE = "sd3.5_medium_vae.safetensors"
LOCAL_JSON = os.path.join(BASE, "models.local.json")
NOTES_JSON = os.path.join(BASE, "model_notes.json")   # 模型说明书(关键字→擅长领域/速度/内存)

SAMPLER_DEFAULT = {"steps": 20, "cfg": 6.0, "sampler_name": "euler_ancestral", "scheduler": "normal"}

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


def sd35_ok():
    """SD3.5 Medium = diffusion 主模型 + 三文本编码器 + VAE 都在才展示。"""
    return (os.path.exists(os.path.join(DIFF_DIR, SD35_FILE)) and
            all(os.path.exists(os.path.join(ENC_DIR, c)) for c in SD35_CLIPS) and
            os.path.exists(os.path.join(VAE_DIR, SD35_VAE)))

def list_models():
    """可选模型 = SD3.5 专用三件套 + models/image 里的单文件 checkpoint + 三件套齐全的 GGUF。"""
    friendly, ggufs = registry()
    out = []
    if sd35_ok():
        meta = friendly.get(SD35_FILE, {})
        m = {"id": SD35_FILE, "kind": "sd3", "name": meta.get("name", "SD3.5 Medium"),
             "sec": int(meta.get("sec", 180)),
             "desc": meta.get("desc", "SD3.5 Medium 专用流·三编码器·文生图/图生图")}
        note = match_note(SD35_FILE, m["name"])
        if note: m.update({"field": note.get("field"), "speed": note.get("speed"),
                           "time": note.get("time"), "mem": note.get("mem"),
                           "detail": note.get("detail")})
        out.append(m)
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
            m = {"id": e["id"], "kind": "gguf", "hq_lock": bool(e.get("clip_type") == "sdxl"),
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
    if model_id == SD35_FILE and sd35_ok():
        return {"id": SD35_FILE, "kind": "sd3", "sampler":
                {"steps": 28, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal"}}
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

def first_ipa():
    """第一个 IPAdapter 模型(优先 plus-face: 锁脸比锁风格更准)。"""
    if not os.path.isdir(MODELS_IPA):
        return ""
    fs = [f for f in sorted(os.listdir(MODELS_IPA)) if f.lower().endswith(".safetensors")]
    for f in fs:
        if "face" in f.lower():
            return f
    return fs[0] if fs else ""

def first_clipvision():
    """第一个 CLIP 视觉编码器(IPAdapter 编码参考图必需)。"""
    if not os.path.isdir(MODELS_CV):
        return ""
    for f in sorted(os.listdir(MODELS_CV)):
        if f.lower().endswith(".safetensors"):
            return f
    return ""

# ---------------- ComfyUI 工作流(checkpoint / GGUF 双架构) ----------------
def _enc_nodes(e):
    """按模型架构返回 (加载器节点dict, clip接线, vae接线, model接线)。"""
    if e["kind"] == "sd3":
        nodes = {
            "100": {"class_type": "UNETLoader", "inputs": {
                "unet_name": SD35_FILE, "weight_dtype": "default"}},
            "101": {"class_type": "TripleCLIPLoader", "inputs": {
                "clip_name1": SD35_CLIPS[0], "clip_name2": SD35_CLIPS[1],
                "clip_name3": SD35_CLIPS[2]}},
            "102": {"class_type": "VAELoader", "inputs": {"vae_name": SD35_VAE}},
        }
        return (nodes, ["101", 0], ["102", 0], ["100", 0])
    if e["kind"] == "gguf":
        # 用大编号 100/101/102,避免与 build_wf 里 5~11 的功能节点撞号
        unet_loader = e.get("unet_loader", "UnetLoaderGGUF")
        unet_inputs = {"unet_name": e["unet"]}
        if unet_loader == "UNETLoader":  # 普通 safetensors 分体(非 GGUF)要多一个 dtype
            unet_inputs["weight_dtype"] = e.get("weight_dtype", "default")
        nodes = {"100": {"class_type": unet_loader, "inputs": unet_inputs}}
        if e.get("clip2"):  # 双文本编码器(如 SDXL 分体): clip=clip_g, clip2=clip_l
            nodes["101"] = {"class_type": "DualCLIPLoader",
                            "inputs": {"clip_name1": e["clip"], "clip_name2": e["clip2"],
                                       "type": e["clip_type"]}}
        else:
            nodes["101"] = {"class_type": e.get("clip_loader", "CLIPLoaderGGUF"),
                            "inputs": {"clip_name": e["clip"], "type": e["clip_type"]}}
        nodes["102"] = {"class_type": "VAELoader", "inputs": {"vae_name": e["vae"]}}
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
             strength=0.6, scale=2.0, ctype="openpose", ipa=None, ipa_weight=0.3, batch=1):
    """e 是 find_entry() 返回的模型定义。五种玩法: t2i/i2i/inpaint/upscale/pose。
    ipa=参考人物图(服务端文件名)时,叠加 IPAdapter 角色一致性(可与其他玩法组合)。
    ipa_weight 控制锁脸力度(本机 M1 Max/MPS 实测): 配大头照参考图,0.3 干净且锁脸最稳,
    0.35 仍可用, >=0.4 会出排线/彩虹色边杂纹。参考图务必用大头照/半身照,全身照锁不住脸。"""
    loaders, clip_src, vae_src, model_src = _enc_nodes(e)
    cfg = e.get("sampler", SAMPLER_DEFAULT)
    wf = dict(loaders)
    if e["kind"] == "sd3":
        wf["1s"] = {"class_type": "ModelSamplingSD3", "inputs": {"shift": 3.0, "model": model_src}}
        model_src = ["1s", 0]
        wf["3"] = {"class_type": "CLIPTextEncodeSD3", "inputs": {
            "clip_l": pos, "clip_g": pos, "t5xxl": pos, "empty_padding": "none", "clip": clip_src}}
        wf["4"] = {"class_type": "CLIPTextEncodeSD3", "inputs": {
            "clip_l": neg, "clip_g": neg, "t5xxl": neg, "empty_padding": "none", "clip": clip_src}}
    else:
        wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": clip_src}}
        wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": clip_src}}
    pos_out = ["3", 0]
    if e.get("flux_guidance"):  # flux 系正向要过 FluxGuidance
        wf["3g"] = {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["3", 0]}}
        pos_out = ["3g", 0]

    if ipa:  # 角色一致性(IPAdapter): 参考人物的脸/形象经 CLIP 视觉编码后注入交叉注意力,只换 model 接线
        if not (e["kind"] == "checkpoint" or e.get("clip_type") == "sdxl"):
            raise ValueError("角色一致性(IPAdapter)需要 SDXL 模型")
        ipam, cv = first_ipa(), first_clipvision()
        if not ipam or not cv:
            raise ValueError("未找到 IPAdapter 模型或 CLIP 视觉编码器,检查 models/image/ipadapter 和 clip_vision/")
        wf["1c"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": cv}}
        wf["1i"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipam}}
        wf["1l"] = {"class_type": "LoadImage", "inputs": {"image": ipa}}
        wf["1a"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": model_src, "ipadapter": ["1i", 0], "clip_vision": ["1c", 0], "image": ["1l", 0],
            "weight": ipa_weight, "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}}
        model_src = ["1a", 0]

    if mode == "inpaint":  # 局部重绘: 只重画涂抹区域
        if e["kind"] == "sd3":
            raise ValueError("SD3.5 Medium 当前只支持文生图和图生图，不支持局部重绘")
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
        wf["8"] = {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["6", 0], "vae": vae_src, "mask": ["7", 0], "grow_mask_by": 6}}
        wf["9"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["8", 0]}}
        wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae_src}}
        wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["10", 0]}}
        return wf

    if mode == "upscale":  # 放大: 先拉伸再低幅度重绘补细节
        if e["kind"] == "sd3":
            raise ValueError("SD3.5 Medium 当前只支持文生图和图生图，不支持放大重绘")
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
        # 先把参考图等比缩放到目标 W×H(crop=center 铺满居中裁,不拉伸),
        # 否则 latent 尺寸=参考图尺寸,输出会无视请求的 W×H(构图被压成 ref 竖版半身)。
        wf["5b"] = {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": w, "height": h, "crop": "center", "image": ["5", 0]}}
        wf["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5b", 0], "vae": vae_src}}
        wf["7"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": strength, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["6", 0]}}
        wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": vae_src}}
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["8", 0]}}
        return wf

    # t2i 普通文生图(batch>1: 同一提示词一次出N张不同seed,共享CLIP编码,省时间;仅本路径支持批量)
    wf["5"] = {"class_type": "EmptySD3LatentImage" if e["kind"] == "sd3" else e.get("latent", "EmptyLatentImage"),
               "inputs": {"width": w, "height": h, "batch_size": max(1, int(batch))}}
    wf["6"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": model_src, "positive": pos_out, "negative": ["4", 0], "latent_image": ["5", 0]}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae_src}}
    wf["8"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["7", 0]}}
    return wf

# ---------------- ComfyUI 通信 ----------------
def port_running(port):
    return svc.port_up(port)   # 复用 socket 直连版:不吃代理、端口关立即返回(首页/api/status 卡慢的元凶)

def comfy_get(path, timeout=10):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{IMG_PORT}{path}", timeout=timeout))

def comfy_post(path, payload, timeout=10):   # 提交类调用(取消/删队列用),与 comfy_get 同实例
    req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

TASKS = {}
LOCK = threading.Lock()

class ComfyServiceUnavailable(RuntimeError):
    """ComfyUI 不可连接或尚未就绪。"""


def _submit_prompt(req, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            last = e
            if attempt < retries:
                time.sleep(1 + attempt)
    raise ComfyServiceUnavailable(f"生图服务 {IMG_PORT} 不可用: {last}") from last


def submit(model, pos, neg, w, h, name, mode="t2i", ref=None, mask=None, strength=0.6, scale=2.0, ctype="openpose", ipa=None, ipa_weight=0.3, batch=1, timeout=1800, archive=None):
    import random
    e = find_entry(model)
    if not e:
        raise ValueError(f"模型不可用: {model}(文件缺失,检查 models/ 目录)")
    seed = random.randint(0, 2**31 - 1)
    batch = max(1, min(4, int(batch)))          # 钳 1~4: 32GB内存,SDXL batch>4 易 swap 反而更慢
    if mode != "t2i":
        batch = 1                                # 批量只走 t2i;i2i/垫图/inpaint 等 latent 来自参考图,保持逐张
    wf = build_wf(e, pos, neg, w, h, seed, mode, ref, mask, strength, scale, ctype, ipa, ipa_weight, batch)
    req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = _submit_prompt(req)["prompt_id"]
    with LOCK:
        TASKS[pid] = {"name": name, "model": model, "t0": time.time(), "done": False,
                      "error": "", "url": "", "batch": batch, "files": [],
                      "timeout": max(60, int(timeout)), "archive": archive or ""}
    threading.Thread(target=wait_done, args=(pid,), daemon=True).start()
    return pid

def wait_done(pid):
    deadline = time.time() + TASKS.get(pid, {}).get("timeout", 1800)
    while time.time() < deadline:
        time.sleep(1)
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
            batch = TASKS[pid].get("batch", 1)
            srcs = []
            for out in hist[pid].get("outputs", {}).values():
                for img in out.get("images", []):
                    f = os.path.join(COMFY_OUT, img.get("subfolder", ""), img["filename"])
                    if os.path.exists(f):
                        srcs.append(f)
            if srcs:
                os.makedirs(OUT_DIR, exist_ok=True)
                name = TASKS[pid]["name"]
                files = []
                if batch <= 1 or len(srcs) == 1:      # 单张: 保持原名 name.png(老调用方零影响)
                    dest = os.path.join(OUT_DIR, name + ".png")
                    shutil.copy2(srcs[-1], dest)
                    files.append(dest)
                else:                                    # 批量: name_1.png .. name_N.png
                    for i, f in enumerate(srcs, 1):
                        dest = os.path.join(OUT_DIR, f"{name}_{i}.png")
                        shutil.copy2(f, dest)
                        files.append(dest)
                for f in srcs:
                    try: os.remove(f)
                    except OSError: pass
                archive = TASKS[pid].get("archive", "")
                if archive:
                    folder = re.sub(r"[^\w.-]+", "_", str(archive), flags=re.UNICODE).strip("._")[:80] or "round"
                    adir = os.path.join(HQ_OUT, folder)
                    os.makedirs(adir, exist_ok=True)
                    for f in files:
                        try: shutil.copy2(f, os.path.join(adir, os.path.basename(f)))
                        except OSError: pass
                with LOCK:
                    TASKS[pid]["done"] = True
                    TASKS[pid]["files"] = files
                    TASKS[pid]["url"] = "/out/" + os.path.basename(files[0])
            else:
                with LOCK: TASKS[pid]["error"] = "找不到输出文件"
            return
    timeout_min = max(1, round(TASKS.get(pid, {}).get("timeout", 1800) / 60))
    with LOCK: TASKS[pid]["error"] = f"超时({timeout_min}分钟)"

# ---------------- 预定批量(batch_images.py 的网页版) ----------------
# 逻辑同 batch_images.py: 逐模型逐提示词提交→轮询等真完成(不是入队就算完)
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
                    time.sleep(1)
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

# ---- 自定义参考图库(用户自己传图当 i2i 参考,可多张上传/点选/删除) ----
REFS_DIR    = os.path.join(BASE, "refs")
REFS_ACTIVE = os.path.join(REFS_DIR, ".active")   # 记住当前选用的一张(空=纯文字)
REF_EXTS    = (".png", ".jpg", ".jpeg", ".webp")

def list_refs():
    """refs/ 里的参考图 + 当前选用的一张。"""
    os.makedirs(REFS_DIR, exist_ok=True)
    fs = [f for f in sorted(os.listdir(REFS_DIR)) if f.lower().endswith(REF_EXTS)]
    active = ""
    try:
        active = open(REFS_ACTIVE, encoding="utf-8").read().strip()
    except Exception:
        pass
    if active not in fs:
        active = ""
    return {"refs": [{"name": f, "url": "/refs/" + urllib.parse.quote(f)} for f in fs],
            "active": active}

def ref_save(fname, data):
    os.makedirs(REFS_DIR, exist_ok=True)
    fn = os.path.basename(fname) or f"ref_{int(time.time())}.png"
    with open(os.path.join(REFS_DIR, fn), "wb") as f:
        f.write(data)
    return {"ok": True, "name": fn}

def ref_set_active(name):
    os.makedirs(REFS_DIR, exist_ok=True)
    name = os.path.basename(name or "")
    with open(REFS_ACTIVE, "w", encoding="utf-8") as f:
        f.write(name)
    return {"ok": True, "active": name}

def ref_delete(name):
    fn = os.path.basename(name or "")
    fp = os.path.join(REFS_DIR, fn)
    if os.path.exists(fp):
        os.remove(fp)
    try:
        if open(REFS_ACTIVE, encoding="utf-8").read().strip() == fn:
            ref_set_active("")
    except Exception:
        pass
    return {"ok": True}

def ref_to_comfy(name):
    """把 refs/<name> 上传给 ComfyUI 当 i2i 参考,返回服务器文件名(失败 None)。"""
    fp = os.path.join(REFS_DIR, os.path.basename(name or ""))
    if not os.path.exists(fp):
        return None
    boundary = "----refupload"
    data = open(fp, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{os.path.basename(fp)}\"\r\nContent-Type: image/png\r\n\r\n").encode() + data + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n"
            f"true\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"http://127.0.0.1:{IMG_PORT}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read()).get("name")
    except Exception:
        return None


# ---------------- 语音模型(TTS · Audio8) ----------------
# 独立微服务 tts_server.py,用 ~/.venvs/audio8 的 python 跑(transformers<5,与 ComfyUI 5.x 隔离)。
# 0.6B 模型跑 CPU(~2.5GB),不参与内存互斥,可与图/视频/语言共存;所以启停不动其它服务。
TTS_PORT   = int(os.environ.get("TTS_PORT", "8851"))
TTS_PY     = os.path.expanduser("~/.venvs/audio8/bin/python")
TTS_SCRIPT = os.path.join(BASE, "tts_server.py")
TTS_PID    = os.path.join(BASE, "tts.pid")
TTS_VOICES = os.path.join(BASE, "models", "tts", "voices")
TTS_OUT    = os.path.join(BASE, "output", "tts")

def _tts_pid_alive():
    try:
        pid = int(open(TTS_PID).read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _tts_port_up():
    """8851 端口是否在监听。socket 直连:不吃代理、端口关立即返回(原来用urllib,挂代理时等满超时)。"""
    return svc.port_up(TTS_PORT)

def tts_state():
    """TTS 状态: running=端口通, ready=模型已加载可合成, error=加载失败信息(503加载中不算错)。"""
    ap = _tts_pid_alive()
    pu = _tts_port_up()
    ready, err = False, ""
    if pu:
        try:
            r = json.load(urllib.request.urlopen(f"http://127.0.0.1:{TTS_PORT}/health", timeout=3))
            ready = bool(r.get("ok"))
        except urllib.error.HTTPError as e:
            if e.code == 500:
                try: err = json.load(e).get("error", "模型加载失败")
                except Exception: err = "模型加载失败"
        except Exception:
            pass
    return {"alive_pid": ap, "running": pu, "ready": ready, "error": err}

def tts_start():
    if _tts_port_up():
        return {"ok": True, "already": True}
    if not os.path.exists(TTS_PY):
        return {"ok": False, "error": "缺少语音运行环境 ~/.venvs/audio8(未安装)"}
    if not os.path.exists(TTS_SCRIPT):
        return {"ok": False, "error": "缺少 tts_server.py"}
    logf = open(os.path.join(BASE, "tts.log"), "ab")
    p = subprocess.Popen([TTS_PY, TTS_SCRIPT], cwd=BASE, stdout=logf, stderr=subprocess.STDOUT)
    open(TTS_PID, "w").write(str(p.pid))
    return {"ok": True, "pid": p.pid}

def tts_stop():
    import signal
    try:
        os.kill(int(open(TTS_PID).read().strip()), signal.SIGTERM)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{TTS_PORT}"], stderr=subprocess.DEVNULL)
        for x in out.split():
            try: os.kill(int(x), signal.SIGTERM)
            except Exception: pass
    except Exception:
        pass
    try: os.remove(TTS_PID)
    except OSError: pass
    for _ in range(6):
        if not _tts_port_up(): break
        time.sleep(1)
    return {"ok": True, "running": _tts_port_up()}

def tts_voices():
    """直接扫声音库目录(不等服务起),与 tts_server 的 list_voices 同规则。"""
    out = []
    if os.path.isdir(TTS_VOICES):
        for name in sorted(os.listdir(TTS_VOICES)):
            d = os.path.join(TTS_VOICES, name)
            wav, txt = os.path.join(d, "ref.wav"), os.path.join(d, "ref.txt")
            if os.path.isfile(wav) and os.path.isfile(txt):
                try: rt = open(txt, encoding="utf-8").read().strip()
                except Exception: rt = ""
                out.append({"name": name, "ref_text": rt,
                            "has_sample": os.path.isfile(os.path.join(d, "sample.wav"))})
    return out

def tts_learn(name, ref_text, fname, data):
    """学一段声音: 原始字节→ffmpeg 抽音轨转 44.1k 单声道 wav→voices/<名>/{ref.wav,ref.txt}。"""
    name = (name or "").strip().replace("/", "_").replace("\\", "_").replace("..", "_")
    ref_text = (ref_text or "").strip()
    if not name: return {"error": "给这个声音起个名字"}
    if not ref_text: return {"error": "参考文本不能为空(必须和音频里说的内容一致)"}
    if not data: return {"error": "没收到音频数据"}
    d = os.path.join(TTS_VOICES, name)
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(fname or "")[1].lower() or ".bin"
    src = os.path.join(d, "src" + ext)
    with open(src, "wb") as f: f.write(data)
    wav = os.path.join(d, "ref.wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "1", wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.exists(wav):
        return {"error": "抽音轨失败,确认上传的是能放出声的视频/音频"}
    trimmed = False
    try:   # 参考音超 12s 会爆模型 2048 prompt 上限 → 切到说话最密片段
        import tts_server as _ts
        cut = _ts.trimmed_ref(d)
        if cut != wav and os.path.exists(cut):
            shutil.move(cut, wav)
            trimmed = True
    except Exception:
        pass
    with open(os.path.join(d, "ref.txt"), "w", encoding="utf-8") as f: f.write(ref_text)
    msg = {"ok": True, "name": name}
    if trimmed:
        msg["note"] = "音频较长,已自动截取说话最清晰的约12秒片段;参考文本请与该片段内容一致"
    return msg

def tts_delvoice(name):
    d = os.path.join(TTS_VOICES, os.path.basename((name or "").strip()))
    if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}

def tts_speak(text, voice):
    """转发给 8851 微服务合成,返回 {"ok","wav"} 或 {"error"}。"""
    if not (text or "").strip(): return {"error": "台词不能为空"}
    body = json.dumps({"text": text, "voice": voice or None}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{TTS_PORT}/tts", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        try: return {"error": json.load(e).get("error", f"合成失败({e.code})")}
        except Exception: return {"error": f"合成失败({e.code})"}
    except Exception as e:
        return {"error": f"语音服务没响应: {e}"}

def tts_progress():
    """转发 8851 的实时进度;服务没起就回静止值,前端轮询不炸。"""
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{TTS_PORT}/progress", timeout=3))
    except Exception:
        return {"step": 0, "max": 0, "busy": False}

def tts_rename(old, new):
    """重命名声音: 纯目录改名(8851 每次都现扫目录/现读 ref,不用动服务)。"""
    old = os.path.basename((old or "").strip())
    new = (new or "").strip().replace("/", "_").replace("\\", "_").replace("..", "_")
    new = os.path.basename(new)
    if not old or not new: return {"error": "名字不能为空"}
    src = os.path.join(TTS_VOICES, old)
    dst = os.path.join(TTS_VOICES, new)
    if not os.path.isdir(src): return {"error": "原声音不存在"}
    if new == old: return {"ok": True, "name": new}
    if os.path.exists(dst): return {"error": "这个名字已被占用"}
    os.rename(src, dst)
    return {"ok": True, "name": new}

def tts_sample(name, text=""):
    """给声音库某个声音生成试听(转发 8851 /sample);text 自定义试听句,空=默认句。"""
    name = os.path.basename((name or "").strip())
    if not name: return {"error": "缺声音名"}
    if not os.path.isfile(os.path.join(TTS_VOICES, name, "ref.wav")):
        return {"error": "声音不存在"}
    body = json.dumps({"voice": name, "text": (text or "").strip()}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{TTS_PORT}/sample", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        try: return {"error": json.load(e).get("error", f"试听生成失败({e.code})")}
        except Exception: return {"error": f"试听生成失败({e.code})"}
    except Exception as e:
        return {"error": f"语音服务没响应: {e}"}

# ---------------- 分段配音工作台(30秒/段·可排序·局部重录·合并锁定) ----------------
# 单次合成上限 1024 帧≈23秒≈180字,30秒段必须内部切块再 ffmpeg 拼接。8851 不动,这里只做编排。
WB = {"segs": [], "next_id": 1, "merged": None, "busy": False}

def _wb_chunk_text(text, maxlen=140):
    """把一段台词按句切成 ≤maxlen 字的小块(单次合成 1024 帧≈23秒,140字≈770帧留足余量)。
    优先在 。!?!;…\\n 处断;单句超长就硬切。"""
    parts = [p.strip() for p in re.split(r"(?<=[。!?!;…\n])", text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= maxlen:
            cur += p
        else:
            if cur: chunks.append(cur)
            while len(p) > maxlen:
                chunks.append(p[:maxlen]); p = p[maxlen:]
            cur = p
    if cur: chunks.append(cur)
    return chunks

def _wb_concat(wavs, out):
    """ffmpeg concat 把多段 wav 拼成一个(同模型同参数,直接 -c copy 不重编码)。"""
    lst = out + ".list"
    with open(lst, "w", encoding="utf-8") as f:
        for w in wavs: f.write("file '%s'\n" % w)
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try: os.remove(lst)
    except OSError: pass
    return r.returncode == 0 and os.path.exists(out)

def wb_state():
    """工作台全貌;正在跑的那段补上 8851 实时帧进度(本请求另起线程查,不卡合成线程)。"""
    segs = [dict(s) for s in WB["segs"]]
    for s in segs:
        if s["status"] == "run":
            p = tts_progress()
            s["frame"], s["frame_max"] = p.get("step", 0), p.get("max", 0)
    return {"segs": segs, "merged": WB["merged"], "busy": WB["busy"], "locked": bool(WB["merged"])}

def _wb_kick():
    if not WB["busy"]:
        threading.Thread(target=_wb_worker, daemon=True).start()

def _wb_worker():
    """串行取 wait 段合成(8851 自身也有锁,并发无意义)。好一段即标记 done,前端立即可听。"""
    WB["busy"] = True
    try:
        while True:
            nxt = next((s for s in WB["segs"] if s["status"] == "wait"), None)
            if nxt is None: break
            nxt["status"] = "run"; nxt["err"] = ""
            r = _wb_synth_seg(nxt)
            if r.get("ok"):
                nxt["status"] = "done"
            else:
                nxt["status"] = "err"; nxt["err"] = r.get("error", "合成失败")
    finally:
        WB["busy"] = False

def _wb_synth_seg(seg):
    """合成一整段: 内部切块逐块 synth(每块≤23秒防爆 1024 帧),再拼成单个 wav。"""
    chunks = _wb_chunk_text(seg["text"])
    seg["chunk_count"] = len(chunks); seg["chunks_done"] = 0
    wavs = []
    for c in chunks:
        if WB["merged"]: return {"error": "已合并锁定"}
        r = tts_speak(c, seg["voice"])
        if not (r.get("ok") and r.get("wav")):
            return {"error": r.get("error", "合成失败")}
        wavs.append(os.path.join(TTS_OUT, r["wav"]))
        seg["chunks_done"] += 1
    if len(wavs) == 1:
        final = wavs[0]
    else:
        final = os.path.join(TTS_OUT, "seg_%d_%d.wav" % (seg["id"], int(time.time())))
        if not _wb_concat(wavs, final):
            return {"error": "段落拼接失败"}
        for w in wavs:
            try: os.remove(w)
            except OSError: pass
    seg["wav"] = os.path.basename(final)
    return {"ok": True}

def wb_add(text, voice):
    text = (text or "").strip()
    if not text: return {"error": "台词不能为空"}
    if WB["merged"]: return {"error": "已合并锁定,不能再加(点重新开始)"}
    WB["segs"].append({"id": WB["next_id"], "text": text, "voice": voice or "",
                       "status": "wait", "wav": None, "err": "", "chunks_done": 0, "chunk_count": 1})
    WB["next_id"] += 1
    _wb_kick()
    return {"ok": True}

def wb_edit(sid, text=None, voice=None):
    """改台词或音色(text/voice 传谁改谁);改了就把这段打回 wait 重录。"""
    for s in WB["segs"]:
        if s["id"] == sid:
            if s["status"] == "run": return {"error": "这段正在合成,等它跑完再改"}
            changed = False
            if text is not None:
                text = text.strip()
                if not text: return {"error": "台词不能为空"}
                if text != s["text"]: s["text"] = text; changed = True
            if voice is not None and voice != s["voice"]:
                s["voice"] = voice; changed = True
            if changed:
                s["status"] = "wait"; s["wav"] = None; s["err"] = ""
            break
    _wb_kick()
    return {"ok": True}

def wb_del(sid):
    for i, s in enumerate(WB["segs"]):
        if s["id"] == sid:
            if s["status"] == "run": return {"error": "这段正在合成,等它跑完再删"}
            WB["segs"].pop(i)
            break
    return {"ok": True}

def wb_regen(sid):
    for s in WB["segs"]:
        if s["id"] == sid:
            if s["status"] == "run": return {"error": "这段正在合成"}
            s["status"] = "wait"; s["wav"] = None; s["err"] = ""
            break
    _wb_kick()
    return {"ok": True}

def wb_split(sid, start, end):
    """片段式重录: 把这段按选中字范围拆成 前/选中/后 最多3小段,各自重新合成(音频按整句语气生成,无法按字切开,故拆开重录)。"""
    for i, s in enumerate(WB["segs"]):
        if s["id"] == sid:
            if s["status"] == "run": return {"error": "这段正在合成,等它跑完再拆"}
            t = s["text"]
            a, b = max(0, int(start)), min(len(t), int(end))
            if a >= b: return {"error": "先在台词框里选中要重录的那几个字"}
            pieces = [x for x in (t[:a].strip(), t[a:b].strip(), t[b:].strip()) if x]
            if len(pieces) < 2: return {"error": "选中范围太靠边缘,没必要拆"}
            new = []
            for p in pieces:
                new.append({"id": WB["next_id"], "text": p, "voice": s["voice"],
                            "status": "wait", "wav": None, "err": "", "chunks_done": 0, "chunk_count": 1})
                WB["next_id"] += 1
            WB["segs"][i:i+1] = new
            break
    _wb_kick()
    return {"ok": True}

def wb_order(ids):
    pos = {int(i): k for k, i in enumerate(ids)}
    WB["segs"].sort(key=lambda s: pos.get(s["id"], 10**9))
    return {"ok": True}

def wb_merge():
    """按当前顺序把所有 done 段拼成一条最终 wav;合并即锁定(不能再改台词/音色/顺序)。"""
    if WB["merged"]: return {"error": "已经合并过了"}
    if any(s["status"] in ("wait", "run") for s in WB["segs"]):
        return {"error": "还有段落没生成完"}
    if any(s["status"] == "err" for s in WB["segs"]):
        return {"error": "有失败段落,先重录或删除再合并"}
    dones = [s for s in WB["segs"] if s["status"] == "done" and s["wav"]]
    if not dones: return {"error": "没有已生成的段落"}
    name = "wb_merged_%d.wav" % int(time.time())
    out = os.path.join(TTS_OUT, name)
    if not _wb_concat([os.path.join(TTS_OUT, s["wav"]) for s in dones], out):
        return {"error": "合并失败"}
    WB["merged"] = name
    return {"ok": True, "wav": name}

def wb_reset():
    if WB["busy"]: return {"error": "正在合成,等它跑完再重开"}
    WB["segs"] = []; WB["merged"] = None
    return {"ok": True}


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
    d = {"panels": panels, "models": list_models(), "out_dir": "output/my_story_comic/",
         "story_file": "my_story/story.txt"}
    d.update(list_refs())   # 参考图库: refs 列表 + active(当前选用)
    d["styles"] = [{"key": k, "name": n} for k, (n, _) in ce.STYLE_PRESETS.items()]  # 画风预设
    return d

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

def comic_start(model, panels, ref, strength, w, h, style="", custom_style="", cmode="i2i"):
    if COMIC["running"]:
        return {"error": "已有连载任务在跑(先停止或等它结束)"}
    if not panels:
        return {"error": "至少留一格分镜"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    COMIC.update(running=True, done=0, total=len(panels), ok=0, fail=0,
                 current="", log=[], stop=False, finished=False, last="")
    threading.Thread(target=_comic_run, args=(model, panels, ref, strength, w, h,
                                              style, custom_style, cmode), daemon=True).start()
    return {"ok": True}

def _comic_run(model, panels, ref, strength, w, h, style="", custom_style="", cmode="i2i"):
    os.makedirs(COMIC_OUT, exist_ok=True)
    try:
        import PIL  # noqa: F401
        can_text = True
    except Exception:
        can_text = False
        _clog("⚠ Pillow 未安装(pip3 install Pillow),本批跳过台词叠加")
    # ref 是 refs/ 图库里的本地文件名,先上传给 ComfyUI 拿服务器名(失败则纯文字)
    comfy_ref = None
    if ref:
        comfy_ref = ref_to_comfy(ref)
        if comfy_ref:
            _clog(f"✔ 参考图已就位: {ref} · {'IPAdapter 锁脸(只锁长相)' if cmode=='ipa' else 'i2i 垫图(贴整张画风)'}")
        else:
            _clog("⚠ 参考图上传失败,改纯文字生成")
    # 画风: 自定义框优先,否则按预设 key(空/未知→油亮)
    style_str = ce.style_prefix(style, custom_style)
    style_name = custom_style.strip()[:30] if custom_style.strip() else \
        ce.STYLE_PRESETS.get(style, ce.STYLE_PRESETS["glossy"])[0]
    _clog(f"画风: {style_name}")
    for p in panels:
        if COMIC["stop"]:
            break
        num = p["num"]
        COMIC["current"] = f"第 {num:03d} 格"
        _clog(f"[{COMIC['done']+1}/{COMIC['total']}] 第 {num:03d} 格 | {p['dialogue'][:40]}")
        name = _safe(f"comic_{num:03d}_{int(time.time())}")
        try:
            if comfy_ref and cmode == "ipa":
                # IPAdapter 锁脸: 只锁主角长相,构图/动作交给提示词;权重走 submit 默认(0.3 实测最稳)
                pid = submit(model, style_str + ", " + p["prompt"], NEG_DEFAULT, w, h, name,
                             "t2i", None, None, strength, ipa=comfy_ref)
            else:
                pid = submit(model, style_str + ", " + p["prompt"], NEG_DEFAULT, w, h, name,
                             "i2i" if comfy_ref else "t2i", comfy_ref, None, strength)
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
            dlg = ce.parse_dialogue(p["dialogue"]) \
                if p["dialogue"] and p["dialogue"] != "(no dialogue)" else []
            if can_text and dlg:
                try:
                    ce.add_dialogue(dst, dlg, dst)
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
    # 竖版单格 → 竖向拼成条漫长页 page_01.png…
    try:
        pages = ce.assemble_pages(COMIC_OUT, COMIC["total"])
        if pages:
            _clog("🧩 条漫 " + str(len(pages)) + " 页: " + ", ".join(os.path.basename(x) for x in pages))
    except Exception as e:
        _clog(f"⚠ 条漫拼接失败(单格仍在): {e}")
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

def cc_panels():
    """output/custom_comic 里可拖气泡的干净分镜(NNN.png) → [{name,url}]。"""
    out = []
    if os.path.isdir(CC_OUT):
        for f in sorted(os.listdir(CC_OUT)):
            if len(f) == 7 and f.endswith(".png") and f[:3].isdigit():
                out.append({"name": f, "url": "/cc_out/" + f})
    return {"panels": out}

def cc_flatten(name, bubbles):
    """把 bubbles 压平到干净格 CC_OUT/<name> → 存 <base>_flat.png(原图保留可再改)。"""
    src = os.path.join(CC_OUT, os.path.basename(name))
    if not os.path.exists(src):
        return {"error": "找不到这格: " + name}
    outname = os.path.splitext(os.path.basename(name))[0] + "_flat.png"
    ce.render_bubbles(src, bubbles, os.path.join(CC_OUT, outname))
    return {"ok": True, "name": outname, "url": "/cc_out/" + outname}

# ---------------- 图片测试场(⚙️齿轮·独立后台 worker) ----------------
# 生图循环在 test_worker.py 独立进程里(start_new_session 脱离),关网页/杀 gen.py 都不停;
# 全靠 test_jobs/ 下文件通信,这里只读状态/写任务/写控制令,不在本进程跑生图。
TEST_JOBS   = os.path.join(BASE, "test_jobs")
TEST_QUEUE  = os.path.join(TEST_JOBS, "queue")
TEST_PROMPT = os.path.join(TEST_JOBS, "prompts")
TEST_STATUS = os.path.join(TEST_JOBS, "status.json")
TEST_CTRL   = os.path.join(TEST_JOBS, "control.json")
TEST_PIDF   = os.path.join(TEST_JOBS, "worker.pid")
TEST_OUT    = os.path.join(BASE, "output", "imgtest")

VTEST_JOBS   = os.path.join(BASE, "video_test_jobs")
VTEST_QUEUE  = os.path.join(VTEST_JOBS, "queue")
VTEST_PROMPT = os.path.join(VTEST_JOBS, "prompts")
VTEST_STATUS = os.path.join(VTEST_JOBS, "status.json")
VTEST_CTRL   = os.path.join(VTEST_JOBS, "control.json")
VTEST_PIDF   = os.path.join(VTEST_JOBS, "worker.pid")
VTEST_OUT    = os.path.join(BASE, "output", "vidtest")
VTEST_SPAWN_LOCK = threading.Lock()

def _test_safe_folder(raw, fallback):
    """生成跨平台安全的目录名：保留文字、数字、下划线和连字符。"""
    for value in (raw, fallback, "model"):
        name = re.sub(r"[^\w-]+", "_", str(value or "").strip(), flags=re.UNICODE)
        name = re.sub(r"_+", "_", name).strip("_")[:80].rstrip("_")
        if name:
            return name
    return "model"


def _test_clean_models(models):
    """清洗测试场目录名，并避免两个模型写进同一个目录。"""
    clean = []
    used = set()
    for index, model in enumerate(models, 1):
        mm = dict(model)
        base = _test_safe_folder(mm.get("folder"), mm.get("name") or mm.get("id") or f"model_{index}")
        folder = base
        suffix = 2
        while folder.casefold() in used:
            tail = f"_{suffix}"
            folder = base[:80 - len(tail)].rstrip("_") + tail
            suffix += 1
        used.add(folder.casefold())
        mm["folder"] = folder
        clean.append(mm)
    return clean


def _test_safe_suffix(raw):
    """统一命名后缀；保留用户输入的 - 或 _，没有分隔符时默认加 _。"""
    raw_text = str(raw or "").strip()
    name = re.sub(r"[^\w-]+", "_", raw_text, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"-+", "-", name).strip("_-")[:40].rstrip("_-")
    if not name:
        return ""
    return ("-" if raw_text.startswith("-") else "_") + name


def _video_folder_taken(folder, reserved=None):
    """判断视频任务文件夹是否已被旧任务或磁盘上的成片占用。"""
    key = str(folder or "").casefold()
    if not key:
        return True
    if key in (reserved or set()):
        return True
    if os.path.isdir(os.path.join(VTEST_OUT, folder)):
        return True
    if not os.path.isdir(VTEST_QUEUE):
        return False
    for filename in os.listdir(VTEST_QUEUE):
        if not filename.endswith(".json"):
            continue
        try:
            task = json.load(open(os.path.join(VTEST_QUEUE, filename), encoding="utf-8"))
        except Exception:
            continue
        for model in task.get("models", []):
            actual = model.get("output_folder") or model.get("folder")
            if str(actual or "").casefold() == key:
                return True
    return False


def _video_unique_folder(candidate, reserved):
    """同名任务自动变成 _v2、_v3；已有 _vN 时继续递增。"""
    candidate = _test_safe_folder(candidate, "video")
    if not _video_folder_taken(candidate, reserved):
        reserved.add(candidate.casefold())
        return candidate
    match = re.match(r"^(.*)_v(\d+)$", candidate, flags=re.I)
    root, version = (match.group(1), int(match.group(2)) + 1) if match else (candidate, 2)
    while True:
        folder = _test_safe_folder(f"{root}_v{version}", "video")
        if not _video_folder_taken(folder, reserved):
            reserved.add(folder.casefold())
            return folder
        version += 1


def _video_allocate_folders(models, suffix=""):
    """给每个模型分配本任务独占的成片文件夹，避免不同任务互相覆盖。"""
    reserved = set()
    allocated = []
    for model in models:
        item = dict(model)
        base = _test_safe_folder(item.get("base_folder") or item.get("folder"),
                                 item.get("name") or item.get("id") or "video")
        candidate = base + suffix if suffix and not base.endswith(suffix) else base
        item["base_folder"] = base
        item["output_folder"] = _video_unique_folder(candidate, reserved)
        item["folder"] = base
        allocated.append(item)
    return allocated


def _video_safe_remove(path, roots):
    """只允许删除视频任务自己拥有的文件或目录。"""
    if not path:
        return
    absolute = os.path.realpath(path)
    if not any(absolute == root or absolute.startswith(root + os.sep) for root in roots):
        return
    if os.path.isdir(absolute) and not os.path.islink(absolute):
        shutil.rmtree(absolute, ignore_errors=True)
    elif os.path.exists(absolute):
        try:
            os.remove(absolute)
        except OSError:
            pass


def _video_delete_outputs(task):
    """删除最终成片、分段工作区和旧版 _segments 目录，空模型目录也一并移除。"""
    output_root = os.path.realpath(VTEST_OUT)
    work_root = os.path.realpath(os.path.join(VTEST_JOBS, "work"))
    roots = (output_root, work_root)
    for path in task.get("outputs", []):
        _video_safe_remove(os.path.join(BASE, path), roots)
    segment_paths = []
    for group in task.get("segments", []) or []:
        segment_paths.extend(group if isinstance(group, list) else [group])
    for path in segment_paths:
        if not isinstance(path, str):
            continue
        absolute = os.path.realpath(os.path.join(BASE, path))
        parent = os.path.dirname(absolute)
        if absolute.startswith(work_root + os.sep) or os.path.basename(parent).endswith("_segments"):
            _video_safe_remove(parent, roots)
        else:
            _video_safe_remove(absolute, roots)
    for path in task.get("legacy_work_dirs", []):
        _video_safe_remove(os.path.join(BASE, path), roots)
    _video_safe_remove(os.path.join(BASE, task.get("work_dir", "")), roots)
    for folder in task.get("output_dirs", []):
        absolute = os.path.realpath(os.path.join(BASE, folder))
        if absolute.startswith(output_root + os.sep) and os.path.isdir(absolute):
            try:
                os.rmdir(absolute)
            except OSError:
                pass


def _pidfile_alive(pidfile):
    try:
        pid = int(open(pidfile).read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _queue_has_running(queue_dir):
    if not os.path.isdir(queue_dir):
        return False
    for fn in os.listdir(queue_dir):
        if not fn.endswith(".json"):
            continue
        try:
            if json.load(open(os.path.join(queue_dir, fn), encoding="utf-8")).get("state") == "running":
                return True
        except Exception:
            pass
    return False


def _comfy_queue_busy(stype):
    """检查真实 ComfyUI 队列，覆盖普通生图/生视频页面提交的任务。"""
    try:
        status = svc.svc_status(stype)
        if not status.get("port_up"):
            return False
        data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{status['port']}/queue", timeout=3))
        return bool(data.get("queue_running") or data.get("queue_pending"))
    except Exception:
        return False


def _test_conflict(want):
    """图片测试和视频测试绝不同时跑，避免两套大模型一起占满内存。"""
    if want == "image":
        other_name, pidf, queue, own_pidf, own_svc, other_svc = (
            "视频", VTEST_PIDF, VTEST_QUEUE, TEST_PIDF, "img", "vid")
    else:
        other_name, pidf, queue, own_pidf, own_svc, other_svc = (
            "图片", TEST_PIDF, TEST_QUEUE, VTEST_PIDF, "vid", "img")
    if _pidfile_alive(pidf) or _queue_has_running(queue):
        return {"error": f"{other_name}测试后台任务正在运行，请先停止它再启动当前测试",
                "error_code": "test_worker_conflict", "conflict": other_name + "测试"}
    if _comfy_queue_busy(other_svc):
        return {"error": f"后台还有{other_name}生成任务正在计算，请等它完成或手动停止后再启动当前测试",
                "error_code": "test_worker_conflict", "conflict": other_name + "生成"}
    # 同一种测试 worker 活着时允许继续追加计划；否则普通页面提交的同类任务也要先跑完。
    if not _pidfile_alive(own_pidf) and _comfy_queue_busy(own_svc):
        label = "图片" if want == "image" else "视频"
        return {"error": f"后台还有普通{label}任务正在计算，请等它完成后再启动测试",
                "error_code": "test_worker_conflict", "conflict": label + "生成"}
    return None

def _test_worker_alive():
    return _pidfile_alive(TEST_PIDF)

def _test_spawn():
    """worker 没在跑就派生一个(脱离本进程组,独立存活)。"""
    if _test_worker_alive():
        return
    for d in (TEST_QUEUE, TEST_PROMPT, TEST_OUT):
        os.makedirs(d, exist_ok=True)
    logf = open(os.path.join(TEST_JOBS, "worker.log"), "ab")
    p = subprocess.Popen([sys.executable, os.path.join(BASE, "test_worker.py")],
                         stdout=logf, stderr=subprocess.STDOUT,
                         cwd=BASE, start_new_session=True)
    open(TEST_PIDF, "w").write(str(p.pid))

def test_state():
    """测试中心首页数据: 模型清单 + refs + 任务队列 + worker 实时状态。"""
    tasks = []
    if os.path.isdir(TEST_QUEUE):
        for fn in sorted(os.listdir(TEST_QUEUE)):
            if fn.endswith(".json"):
                try:
                    tasks.append(json.load(open(os.path.join(TEST_QUEUE, fn), encoding="utf-8")))
                except Exception:
                    pass
    status = {}
    try:
        status = json.load(open(TEST_STATUS, encoding="utf-8"))
    except Exception:
        pass
    return {"models": list_models(), "refs": list_refs()["refs"], "tasks": tasks,
            "status": status, "worker_alive": _test_worker_alive(),
            "out_base": "output/imgtest/", "prompt_dir": "test_jobs/prompts/"}

def test_create(cfg):
    """建一个测试任务: 写 queue/NNN.json + prompts/NNN.txt,然后确保 worker 在跑(排队)。"""
    conflict = _test_conflict("image")
    if conflict:
        return conflict
    models = cfg.get("models", [])
    prompts = [p.strip() for p in cfg.get("prompts", []) if p.strip()]
    if not models:
        return {"error": "至少选一个模型"}
    if not prompts:
        return {"error": "至少写一条提示词"}
    for d in (TEST_QUEUE, TEST_PROMPT, TEST_OUT):
        os.makedirs(d, exist_ok=True)
    existing = [f for f in os.listdir(TEST_QUEUE) if f.endswith(".json")]
    nid = f"{(max([int(f[:3]) for f in existing if f[:3].isdigit()] or [0]) + 1):03d}"
    pf = os.path.join(TEST_PROMPT, nid + ".txt")
    open(pf, "w", encoding="utf-8").write("\n".join(prompts) + "\n")
    clean_models = _test_clean_models(models)
    task = {"id": nid, "state": "queued", "created": time.time(),
            "models": clean_models, "canvas": cfg.get("canvas", {"w": 832, "h": 1216}),
            "per_prompt": max(1, min(10, int(cfg.get("per_prompt", 1)))),
            "prompts_file": os.path.relpath(pf, BASE),
            "folder_suffix": _test_safe_suffix(cfg.get("folder_suffix")),
            "file_suffix": _test_safe_suffix(cfg.get("file_suffix")),
            "pad_ref": cfg.get("pad_ref") or "",
            "face_lock": cfg.get("face_lock") or {}}
    json.dump(task, open(os.path.join(TEST_QUEUE, nid + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    _test_spawn()
    return {"ok": True, "id": nid}

def test_rerun(tid):
    """原样重跑: 把已有任务(配置+提示词)复制成一个新任务进队列。"""
    conflict = _test_conflict("image")
    if conflict:
        return conflict
    tid = re.sub(r"\D", "", tid or "")
    src = os.path.join(TEST_QUEUE, tid + ".json")
    if not os.path.exists(src):
        return {"error": "找不到任务 " + tid}
    t = json.load(open(src, encoding="utf-8"))
    t["models"] = _test_clean_models(t.get("models", []))
    pf = os.path.join(BASE, t.get("prompts_file", ""))
    prompts = open(pf, encoding="utf-8").read() if os.path.exists(pf) else ""
    existing = [f for f in os.listdir(TEST_QUEUE) if f.endswith(".json")]
    nid = f"{(max([int(f[:3]) for f in existing if f[:3].isdigit()] or [0]) + 1):03d}"
    npf = os.path.join(TEST_PROMPT, nid + ".txt")
    open(npf, "w", encoding="utf-8").write(prompts)
    t.update({"id": nid, "state": "queued", "created": time.time(),
              "prompts_file": os.path.relpath(npf, BASE)})
    t.pop("started", None); t.pop("finished", None)
    json.dump(t, open(os.path.join(TEST_QUEUE, nid + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    _test_spawn()
    return {"ok": True, "id": nid}

def test_control(cmd):
    """写控制令给 worker: pause/resume/kill_curr(杀当前任务)/kill_all(连worker一起杀)。"""
    os.makedirs(TEST_JOBS, exist_ok=True)
    if cmd not in ("pause", "resume", "kill_curr", "kill_all"):
        return {"error": "未知控制令: " + cmd}
    json.dump({"cmd": cmd}, open(TEST_CTRL, "w", encoding="utf-8"))
    return {"ok": True, "cmd": cmd}

def test_delete(tid):
    """删任务: 只删 queue/NNN.json + prompts/NNN.txt(配置提示词),图片一律保留。"""
    tid = re.sub(r"\D", "", tid or "")
    for p in (os.path.join(TEST_QUEUE, tid + ".json"),
              os.path.join(TEST_PROMPT, tid + ".txt")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    return {"ok": True}

def test_prompts_get(tid):
    tid = re.sub(r"\D", "", tid or "")
    pf = os.path.join(TEST_PROMPT, tid + ".txt")
    if not os.path.exists(pf):
        return {"error": "没有这个任务的提示词"}
    return {"ok": True, "id": tid, "path": "test_jobs/prompts/" + tid + ".txt",
            "text": open(pf, encoding="utf-8").read()}

def test_prompts_save(tid, text):
    tid = re.sub(r"\D", "", tid or "")
    pf = os.path.join(TEST_PROMPT, tid + ".txt")
    if not os.path.exists(pf):
        return {"error": "任务不存在(可能已删除)"}
    open(pf, "w", encoding="utf-8").write(text)
    return {"ok": True}


# ---------------- 视频测试场(独立后台 worker) ----------------
def _video_test_worker_alive():
    return _pidfile_alive(VTEST_PIDF)


def _video_test_has_queued():
    """队列里是否还有真正等着运行的视频任务。"""
    if not os.path.isdir(VTEST_QUEUE):
        return False
    for filename in os.listdir(VTEST_QUEUE):
        if not filename.endswith(".json"):
            continue
        try:
            task = json.load(open(os.path.join(VTEST_QUEUE, filename), encoding="utf-8"))
            if task.get("state") == "queued":
                return True
        except Exception:
            continue
    return False


def _video_test_spawn():
    with VTEST_SPAWN_LOCK:
        if _video_test_worker_alive():
            return
        for d in (VTEST_QUEUE, VTEST_PROMPT, VTEST_OUT):
            os.makedirs(d, exist_ok=True)
        logf = open(os.path.join(VTEST_JOBS, "worker.log"), "ab")
        p = subprocess.Popen([sys.executable, os.path.join(BASE, "video_test_worker.py")],
                             stdout=logf, stderr=subprocess.STDOUT,
                             cwd=BASE, start_new_session=True)
        open(VTEST_PIDF, "w").write(str(p.pid))


def _video_test_wake_queue():
    """worker 意外退出或前一个任务被删后，自动接着跑剩余队列。"""
    if _video_test_worker_alive() or not _video_test_has_queued():
        return False
    if _test_conflict("video"):
        return False
    _video_test_spawn()
    return True


def _read_video_task_state(filename):
    try:
        return json.load(open(os.path.join(VTEST_QUEUE, filename), encoding="utf-8")).get("state")
    except Exception:
        return ""


def _video_task_total(task):
    """为旧任务补算总成片数，方便断点页面显示。"""
    prompts_file = os.path.join(BASE, task.get("prompts_file", ""))
    try:
        prompt_count = len([line for line in open(prompts_file, encoding="utf-8") if line.strip()])
    except Exception:
        prompt_count = 1
    ref_count = 1 if task.get("mode") == "t2v" else max(1, len(task.get("refs") or []))
    return (len(task.get("models") or []) * prompt_count * ref_count *
            max(1, len(task.get("variants") or [])) * max(1, int(task.get("copies", 1))))


def _video_timing(task, progress):
    """根据任务实际开始时间和已完成成片数，计算本次任务的平均速度。"""
    started = float(task.get("started") or 0)
    finished = float(task.get("finished") or 0)
    now = finished if finished and task.get("state") in ("done", "partial", "failed", "killed") else time.time()
    elapsed = max(0, now - started) if started else 0
    done = max(0, int(progress.get("done") or 0))
    total = max(0, int(progress.get("total") or task.get("total") or _video_task_total(task)))
    average = elapsed / done if done else 0
    return {"elapsed_sec": round(elapsed), "done": done, "total": total,
            "average_sec": round(average) if average else 0,
            "remaining_sec": round(average * max(0, total - done)) if average else 0}


def video_test_state():
    # 页面轮询时顺手修复“有排队任务但 worker 已退出”的状态。
    _video_test_wake_queue()
    tasks = []
    model_samples = {}
    worker_alive = _video_test_worker_alive()
    if os.path.isdir(VTEST_QUEUE):
        for fn in sorted(os.listdir(VTEST_QUEUE)):
            if fn.endswith(".json"):
                try:
                    task = json.load(open(os.path.join(VTEST_QUEUE, fn), encoding="utf-8"))
                    view = dict(task)
                    resumable_state = task.get("state") in (
                        "running", "paused", "interrupted", "killed", "partial", "failed")
                    view["resume_available"] = bool((not worker_alive) and resumable_state and
                                                    (task.get("started") or task.get("outputs") or
                                                     task.get("checkpoint")))
                    if view["resume_available"] and task.get("state") in ("running", "paused"):
                        view["state"] = "interrupted"
                    progress = dict(task.get("progress") or {})
                    if not progress:
                        progress = {"done": len(task.get("outputs") or []),
                                    "total": task.get("total") or _video_task_total(task)}
                    elif not progress.get("total"):
                        progress["total"] = task.get("total") or _video_task_total(task)
                    view["progress"] = progress
                    view["timing"] = _video_timing(task, progress)
                    for sample in task.get("timings") or []:
                        try:
                            if not sample.get("ok", True):
                                continue
                            count = max(1, int(sample.get("segment_count") or 1))
                            elapsed = float(sample.get("elapsed_sec") or 0) / count
                            if elapsed > 0:
                                model_samples.setdefault(str(sample.get("model_id") or ""), []).append(elapsed)
                        except Exception:
                            continue
                    tasks.append(view)
                except Exception:
                    pass
    status = {}
    try:
        status = json.load(open(VTEST_STATUS, encoding="utf-8"))
    except Exception:
        pass
    models = []
    for model in vidwf.list_unets():
        item = dict(model)
        samples = model_samples.get(str(item.get("id") or ""), [])
        if samples:
            item["actual_average_sec"] = round(sum(samples) / len(samples))
            item["actual_samples"] = len(samples)
        models.append(item)
    return {"models": models, "loras": vidwf.list_loras(),
            "refs": list_refs()["refs"], "tasks": tasks, "status": status,
            "worker_alive": _video_test_worker_alive(), "out_base": "output/vidtest/",
            "work_base": "video_test_jobs/work/", "prompt_dir": "video_test_jobs/prompts/"}


def _video_test_variants(raw):
    out = []
    for v in (raw or [])[:30]:
        try:
            w = max(64, min(1920, (int(v.get("w", 360)) // 16) * 16))
            h = max(64, min(1920, (int(v.get("h", 640)) // 16) * 16))
            duration = max(0.5, min(300.0, float(v.get("duration", 2))))
            fps = max(1.0, min(120.0, float(v.get("fps", 24))))
            item = {"w": w, "h": h, "duration": duration, "fps": fps}
            if item not in out:
                out.append(item)
        except Exception:
            pass
    return out or [{"w": 360, "h": 640, "duration": 2, "fps": 24}]


def _video_source_frames(duration, target_fps):
    """高帧率是成片目标；模型统一最多按24fps生成，再由 ffmpeg 补帧。"""
    raw = max(9, round(float(duration) * min(24.0, float(target_fps))))
    return max(9, round((raw - 1) / 8) * 8 + 1)


def video_test_create(cfg):
    conflict = _test_conflict("video")
    if conflict:
        return conflict
    models = _test_clean_models(cfg.get("models", []))
    prompts = [str(p).strip() for p in cfg.get("prompts", []) if str(p).strip()]
    mode = "t2v" if cfg.get("mode") == "t2v" else "i2v"
    refs = []
    for name in cfg.get("refs", []):
        name = os.path.basename(str(name))
        if name and os.path.isfile(os.path.join(REFS_DIR, name)) and name not in refs:
            refs.append(name)
    if not models:
        return {"error": "至少选一个视频模型"}
    if not prompts:
        return {"error": "至少写一条提示词"}
    if mode == "i2v" and not refs:
        return {"error": "图生视频至少选择一张垫图"}
    variants = _video_test_variants(cfg.get("variants"))
    for d in (VTEST_QUEUE, VTEST_PROMPT, VTEST_OUT):
        os.makedirs(d, exist_ok=True)
    existing = [f for f in os.listdir(VTEST_QUEUE) if f.endswith(".json")]
    nid = f"{(max([int(f[:3]) for f in existing if f[:3].isdigit()] or [0]) + 1):03d}"
    pf = os.path.join(VTEST_PROMPT, nid + ".txt")
    open(pf, "w", encoding="utf-8").write("\n".join(prompts) + "\n")
    folder_suffix = _test_safe_suffix(cfg.get("folder_suffix"))
    models = _video_allocate_folders(models, folder_suffix)
    raw_loras = cfg.get("loras")
    if raw_loras is None:
        raw_loras = [cfg.get("lora", "none")]
    if isinstance(raw_loras, str):
        raw_loras = [raw_loras]
    known_loras = {str(item.get("id")) for item in vidwf.list_loras() if item.get("id") != "none"}
    selected_loras = []
    for item in raw_loras or []:
        item = str(item)
        if item in known_loras and item not in selected_loras:
            selected_loras.append(item)
    task = {"id": nid, "state": "queued",
            "created": time.time(), "models": models,
            "mode": mode, "refs": refs, "variants": variants,
            "copies": max(1, min(10, int(cfg.get("copies", 1)))),
            "prompts_file": os.path.relpath(pf, BASE),
            "folder_suffix": folder_suffix,
            "file_suffix": _test_safe_suffix(cfg.get("file_suffix")),
            "lora": selected_loras[0] if selected_loras else "none",
            "loras": selected_loras,
            "stg": bool(cfg.get("stg", False)),
            "interpolate": bool(cfg.get("interpolate", False)),
            "native_audio": bool(cfg.get("native_audio", False)),
            "style_2d": bool(cfg.get("style_2d", False)), "auto_segment": True}
    json.dump(task, open(os.path.join(VTEST_QUEUE, nid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    _video_test_spawn()
    return {"ok": True, "id": nid, "waiting": False}


def video_test_rerun(tid):
    conflict = _test_conflict("video")
    if conflict:
        return conflict
    tid = re.sub(r"\D", "", tid or "")
    src = os.path.join(VTEST_QUEUE, tid + ".json")
    if not os.path.exists(src):
        return {"error": "找不到视频任务 " + tid}
    t = json.load(open(src, encoding="utf-8"))
    pf = os.path.join(BASE, t.get("prompts_file", ""))
    prompts = open(pf, encoding="utf-8").read() if os.path.exists(pf) else ""
    existing = [f for f in os.listdir(VTEST_QUEUE) if f.endswith(".json")]
    nid = f"{(max([int(f[:3]) for f in existing if f[:3].isdigit()] or [0]) + 1):03d}"
    npf = os.path.join(VTEST_PROMPT, nid + ".txt")
    open(npf, "w", encoding="utf-8").write(prompts)
    base_models = []
    for model in t.get("models", []):
        item = dict(model)
        item["folder"] = item.get("base_folder") or item.get("folder") or item.get("name")
        item.pop("output_folder", None)
        base_models.append(item)
    t["models"] = _video_allocate_folders(_test_clean_models(base_models), t.get("folder_suffix", ""))
    t.update({"id": nid, "state": "queued", "created": time.time(),
              "prompts_file": os.path.relpath(npf, BASE), "outputs": [], "segments": [],
              "completed_items": [], "checkpoint": {}, "progress": {},
              "resume_available": False})
    for key in ("started", "finished", "ok", "fail", "total", "error", "error_code"):
        t.pop(key, None)
    json.dump(t, open(os.path.join(VTEST_QUEUE, nid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    _video_test_spawn()
    return {"ok": True, "id": nid}


def video_test_control(cmd):
    os.makedirs(VTEST_JOBS, exist_ok=True)
    if cmd not in ("pause", "resume", "kill_curr", "kill_all"):
        return {"error": "未知控制令: " + cmd}
    json.dump({"cmd": cmd}, open(VTEST_CTRL, "w", encoding="utf-8"))
    return {"ok": True, "cmd": cmd}


def video_test_resume(tid):
    """把上次中断的任务重新排回视频队列，worker 会按断点跳过已完成成片。"""
    conflict = _test_conflict("video")
    if conflict:
        return conflict
    tid = re.sub(r"\D", "", tid or "")
    path = os.path.join(VTEST_QUEUE, tid + ".json")
    if not os.path.exists(path):
        return {"error": "找不到视频任务 " + tid}
    try:
        task = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        return {"error": "读取视频任务失败: " + str(exc)}
    if _video_test_worker_alive():
        return {"error": "视频后台正在运行，请等当前队列处理到这个任务"}
    if task.get("state") == "done":
        return {"error": "这个视频任务已经完成"}
    task.update(state="queued", resume_requested=True, resume_available=False,
                resumed_at=time.time())
    json.dump(task, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    _video_test_spawn()
    return {"ok": True, "id": tid}


def video_test_delete(tid):
    tid = re.sub(r"\D", "", tid or "")
    task_path = os.path.join(VTEST_QUEUE, tid + ".json")
    task = {}
    if os.path.exists(task_path):
        try:
            task = json.load(open(task_path, encoding="utf-8"))
        except Exception:
            task = {}
    if task.get("state") in ("queued", "running") and _video_test_worker_alive():
        return {"error": "任务正在后台运行，请先暂停并停止视频 worker 后再删除", "error_code": "video_task_running"}
    _video_delete_outputs(task)
    for p in (task_path, os.path.join(VTEST_PROMPT, tid + ".txt")):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    _video_test_wake_queue()
    return {"ok": True}


def video_test_prompts_get(tid):
    tid = re.sub(r"\D", "", tid or "")
    pf = os.path.join(VTEST_PROMPT, tid + ".txt")
    if not os.path.exists(pf):
        return {"error": "没有这个视频任务的提示词"}
    return {"ok": True, "id": tid, "path": "video_test_jobs/prompts/" + tid + ".txt",
            "text": open(pf, encoding="utf-8").read()}


def video_test_prompts_save(tid, text):
    tid = re.sub(r"\D", "", tid or "")
    pf = os.path.join(VTEST_PROMPT, tid + ".txt")
    if not os.path.exists(pf):
        return {"error": "视频任务不存在(可能已删除)"}
    open(pf, "w", encoding="utf-8").write(text)
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

def cc_start(chars, panels, style="", custom_style=""):
    if CC["running"]:
        return {"error": "已有自定义连载在跑(先停止或等它结束)"}
    if not panels:
        return {"error": "至少搭一格分镜画布"}
    if not svc.svc_status("img")["running"]:
        return {"error": "生图服务未运行(先回首页点图片卡启动)"}
    CC.update(running=True, done=0, total=len(panels), ok=0, fail=0,
              current="", log=[], stop=False, finished=False, last="")
    threading.Thread(target=_cc_run, args=(chars, panels, style, custom_style), daemon=True).start()
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

def _cc_run(chars, panels, style="", custom_style=""):
    os.makedirs(CC_OUT, exist_ok=True)
    try:
        import PIL  # noqa: F401
        can_text = True
    except Exception:
        can_text = False
        _cclog("⚠ Pillow 未安装,本批跳过台词气泡")
    model = _cc_model()
    # 画风: 自定义框优先,否则按预设 key(空/未知→油亮);前置做基底,每格 cpstyle 拼在后
    style_str = ce.style_prefix(style, custom_style)
    style_name = custom_style.strip()[:30] if custom_style.strip() else \
        ce.STYLE_PRESETS.get(style, ce.STYLE_PRESETS["glossy"])[0]
    _cclog(f"画风: {style_name}")
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
            pid = submit(model, style_str + ", " + _cc_panel_prompt(chars, p), NEG_DEFAULT,
                         int(p.get("w", 832)), int(p.get("h", 1216)), name,
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
            # 气泡不在生成时焊死: 保持干净格,拖到「气泡编辑器」里加,导出时才压平(ce.render_bubbles)
            CC["ok"] += 1; CC["last"] = f"{num:03d}.png"
            _cclog(f"  ✓ 完成({int(time.time()-t0)}秒)")
        else:
            CC["fail"] += 1
            _cclog(f"  ✗ {'已停止' if CC['stop'] else (err or '超时')}")
        CC["done"] += 1
    if CC["stop"]:
        _cclog("⏹ 用户停止")
    # 竖版单格 → 竖向拼成条漫长页 page_01.png…(尺寸不一时按最宽格居中)
    sizes = {(int(p.get("w", 832)), int(p.get("h", 1216))) for p in panels}
    if len(sizes) > 1:
        _cclog("ℹ 各格尺寸不一,条漫已按最宽格居中拼接")
    try:
        pages = ce.assemble_pages(CC_OUT, CC["total"])
        if pages:
            _cclog("🧩 条漫 " + str(len(pages)) + " 页: " + ", ".join(os.path.basename(x) for x in pages))
    except Exception as e:
        _cclog(f"⚠ 条漫拼接失败(单格仍在): {e}")
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
  color-scheme:dark;   /* 让原生控件(音频播放器/滚动条/表单)也跟着变深色 */
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
  background:var(--grad);background-size:220% 100%;-webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;animation:hueSlide 9s ease-in-out infinite alternate}
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
  font-size:15px;font-weight:600;letter-spacing:.2px;cursor:pointer;position:relative;overflow:hidden;
  box-shadow:0 4px 14px rgba(90,110,255,.32),inset 0 1px 0 rgba(255,255,255,.25);
  transition:filter .15s,transform .12s,box-shadow .15s}
button::after{content:'';position:absolute;top:0;left:-90%;width:45%;height:100%;pointer-events:none;
  background:linear-gradient(105deg,transparent,rgba(255,255,255,.38),transparent);
  transform:skewX(-20deg);transition:left .55s ease}
button:hover::after{left:140%}
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
  box-shadow:0 0 8px rgba(111,140,255,.6);position:relative;overflow:hidden}
.bar>i.indet{width:34%!important;animation:barSlide 1.6s ease-in-out infinite}
@keyframes barSlide{0%{transform:translateX(-115%)}55%,100%{transform:translateX(315%)}}
.bar>i::after{content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent);
  animation:barShine 1.6s linear infinite}
#chatbox{max-height:52vh;overflow-y:auto;padding:4px 2px}
.chatrow{display:flex;margin:11px 0}
.chatrow.me{justify-content:flex-end}
.bubble{max-width:82%;padding:11px 15px;border-radius:16px;font-size:14px;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--border);
  box-shadow:var(--shadow-sm)}
.chatrow.me .bubble{background:var(--grad);color:#fff;border-color:transparent;border-radius:16px 16px 5px 16px}
.bubble.ai{border-radius:16px 16px 16px 5px}
.bubble .think{margin-bottom:8px;font-size:12px;color:var(--faint)}
.bubble img.out{margin-top:9px}
.chatinput{position:sticky;bottom:10px;display:flex;gap:9px;margin-top:13px;background:var(--glass);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);padding:10px;border-radius:15px;border:1px solid var(--border)}
.chatinput textarea{flex:1;height:46px;resize:none}
img.out{max-width:100%;border-radius:14px;margin-top:13px;box-shadow:var(--shadow-md)}
img.mask{filter:blur(18px)}
.row{display:flex;gap:11px;align-items:center;flex-wrap:wrap}
.small{font-size:12.5px;color:var(--faint);line-height:1.55}
.vid-workspace{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:22px;align-items:start}
.vid-main{min-width:0}
.vid-side{position:sticky;top:18px;min-width:0;border-left:1px solid var(--border);padding-left:18px}
.vid-side-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:2px 0 10px}
.vid-side-head b{font-size:14px}.vid-side-head button{padding:6px 10px;border-radius:8px;font-size:12px}
.vid-task-group{margin:13px 0 6px;color:var(--muted);font-size:12px;font-weight:700}
.vid-task{border:1px solid var(--border);background:var(--card);border-radius:8px;padding:10px 11px;margin:8px 0;box-shadow:var(--shadow-sm)}
.vid-task:hover{border-color:var(--border-hi)}
.vid-task-title{font-size:13px;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.vid-task-meta{font-size:11.5px;color:var(--faint);margin:4px 0 8px;line-height:1.45}
.vid-task-actions{display:flex;gap:7px}.vid-task-actions button{min-height:30px;padding:5px 10px;border-radius:7px;font-size:12px;box-shadow:none}
.vid-state{font-size:11px;font-weight:700}.vid-state.running{color:var(--on)}.vid-state.queued{color:var(--accent)}.vid-state.failed{color:#d03050}
.vid-empty{padding:18px 10px;text-align:center;color:var(--faint);font-size:12px;border:1px dashed var(--border);border-radius:8px}
.vid-detail{max-width:760px;margin:0 auto}.vid-detail video{display:block;width:100%;max-height:70vh;background:#000;border-radius:8px}
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
.proxy-switch{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;padding:3px;margin:8px 0 5px;
  background:var(--input);border:1px solid var(--border);border-radius:8px;box-shadow:inset 0 1px 2px rgba(20,30,70,.08)}
.proxy-switch button{min-width:0;padding:8px 10px;border:0;border-radius:6px;background:transparent;color:var(--muted);
  font-size:13px;font-weight:600;box-shadow:none;transform:none;transition:background .16s,color .16s,box-shadow .16s}
.proxy-switch button::after{display:none}
.proxy-switch button:hover{background:var(--card-sel);color:var(--text);box-shadow:none;filter:none}
.proxy-switch button.active{background:var(--accent);color:#fff;box-shadow:0 2px 7px rgba(70,100,220,.28)}
.proxy-switch button.active:hover{background:var(--accent);color:#fff;filter:brightness(1.05)}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 6px}
.tag{padding:4px 12px;border-radius:999px;background:var(--chip-bg);border:1px solid var(--chip-bd);
  font-size:12px;cursor:pointer;user-select:none;color:var(--chip-tx);transition:.15s}
.tag:hover{background:var(--card-sel);transform:translateY(-1px)}
.tag.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tag.neg.on{background:#e0556b;border-color:#e0556b}
.tagx{margin-left:5px;color:var(--faint);font-weight:700;cursor:pointer;opacity:.55}
.tagx:hover{color:#ff2d55;opacity:1}
.tag.on .tagx{color:rgba(255,255,255,.85)}
.tagadd{border-style:dashed;color:var(--faint);font-weight:700}
.tagadd:hover{color:var(--accent)}
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
.mtime{font-weight:600}
.mtime.g{color:#22c55e}.mtime.y{color:#eab308}.mtime.r{color:#ef4444}
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
#testBtn{position:fixed;top:16px;right:66px;z-index:99;width:42px;height:42px;padding:0;border-radius:999px;
  font-size:18px;line-height:1;display:flex;align-items:center;justify-content:center;
  background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  color:var(--text);box-shadow:var(--shadow-sm);border:1px solid var(--border);cursor:pointer;
  transition:transform .2s,box-shadow .2s}
#testBtn:hover{transform:rotate(40deg) scale(1.08);box-shadow:var(--shadow-md)}

/* ---- 动效层: 白天模式动态渐变背景 + 入场动画 + 加载 ---- */
/* 白天: 两团柔和色块缓慢漂移(黑夜模式自动变淡变慢,不抢戏) */
body::before{content:'';position:fixed;inset:-22%;z-index:-1;pointer-events:none;
  background:
    radial-gradient(38% 32% at 22% 28%,rgba(124,156,255,.34) 0%,transparent 70%),
    radial-gradient(34% 30% at 78% 22%,rgba(255,170,210,.30) 0%,transparent 70%),
    radial-gradient(40% 36% at 55% 82%,rgba(120,230,200,.26) 0%,transparent 70%),
    radial-gradient(30% 28% at 85% 70%,rgba(255,214,140,.28) 0%,transparent 70%);
  filter:blur(12px);animation:blobA 26s ease-in-out infinite alternate}
body::after{content:'';position:fixed;inset:-22%;z-index:-1;pointer-events:none;
  background:
    radial-gradient(30% 26% at 70% 58%,rgba(150,140,255,.24) 0%,transparent 70%),
    radial-gradient(26% 24% at 28% 74%,rgba(255,190,160,.22) 0%,transparent 70%);
  filter:blur(16px);animation:blobB 33s ease-in-out infinite alternate}
body.dark::before,body.dark::after{opacity:.13;animation-duration:38s,46s}
@keyframes blobA{0%{transform:translate(0,0) rotate(0deg) scale(1)}
  50%{transform:translate(3%,-2.5%) rotate(4deg) scale(1.06)}
  100%{transform:translate(-3%,2%) rotate(-3deg) scale(1.03)}}
@keyframes blobB{0%{transform:translate(0,0) scale(1)}
  100%{transform:translate(-4%,-3%) scale(1.09)}}
@keyframes hueSlide{from{background-position:0% 0}to{background-position:100% 0}}
@keyframes barShine{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
/* 内容入场: 轻微上浮淡入(进度页轮询刷新时等同柔和闪新) */
@keyframes fadeUp{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
#app .card,#app .box,#app .logbox,#app .tabs{animation:fadeUp .38s cubic-bezier(.2,.7,.3,1) both}
/* 加载转圈(配 <span class="spin"></span> 用) */
.spin{display:inline-block;width:15px;height:15px;border:2.5px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:rot .8s linear infinite;vertical-align:-3px}
@keyframes rot{to{transform:rotate(360deg)}}
/* ============ 主题3: 极客·流星雨(body.tech) ============ */
body.tech{
  color-scheme:dark;
  --bg:linear-gradient(180deg,#14161b 0%,#191c22 52%,#12141a 100%);
  --card:#20242d; --card-sel:#223141; --border:#313947; --border-hi:#22d3ee;
  --text:#dbe3ee; --muted:#96a2b4; --faint:#68717f;
  --input:#1a1e26; --accent:#22d3ee; --accent2:#818cf8;
  --grad:linear-gradient(120deg,#22d3ee,#818cf8);
  --btn-grad:linear-gradient(120deg,#0e9fb8 0%,#4f5fd0 60%,#818cf8 100%);
  --shadow-sm:0 1px 2px rgba(0,0,0,.5),0 1px 10px rgba(0,0,0,.35);
  --shadow-md:0 10px 34px rgba(0,0,0,.55),0 0 18px rgba(34,211,238,.14);
  --chip-bg:#1e2f3d; --chip-bd:#2e4a5e; --chip-tx:#7ce7f4;
  --box-bg:#22241a; --box-bd:#4a4a22; --box-tx:#e8d9a8;
  --mem-bg:#242019; --mem-bd:#4f4433; --mem-tx:#d9b25f;
  --on:#4ade80; --off:#6b7280;
  --glass:rgba(24,28,36,.72);
}
/* 科技感: 按钮发光 + 卡片描边光 */
body.tech button{box-shadow:0 0 14px rgba(34,211,238,.28),inset 0 1px 0 rgba(255,255,255,.16)}
body.tech button:hover{box-shadow:0 0 24px rgba(34,211,238,.5),inset 0 1px 0 rgba(255,255,255,.16)}
body.tech .back{box-shadow:var(--shadow-sm)}
body.tech .back:hover{box-shadow:0 0 14px rgba(34,211,238,.22)}
body.tech .card:hover{box-shadow:var(--shadow-md),0 0 20px rgba(34,211,238,.18)}
body.tech .card.sel{box-shadow:var(--shadow-md),0 0 22px rgba(34,211,238,.3)}
body.tech #themePanel .thopt:hover{box-shadow:0 0 12px rgba(34,211,238,.3)}
body.tech::before,body.tech::after{opacity:.06}
/* ============ 主题: GitHub · 夜间科技(body.github) ============ */
body.github{
  color-scheme:dark;
  --bg:radial-gradient(900px 560px at 88% -12%,rgba(255,174,52,.22),transparent 58%),
       radial-gradient(760px 520px at -8% 110%,rgba(255,107,0,.16),transparent 56%),
       linear-gradient(150deg,#0b0b0f 0%,#16110b 52%,#090806 100%);
  --card:#1c1710; --card-sel:#2a1d0e; --border:#3a2a18; --border-hi:#f59e0b;
  --text:#f5e7c8; --muted:#c6a46c; --faint:#856b42;
  --input:#120f0b; --accent:#f59e0b; --accent2:#ff6b00;
  --grad:linear-gradient(120deg,#ffd166,#f59e0b 52%,#ff6b00);
  --btn-grad:linear-gradient(120deg,#ffcc4d 0%,#f59e0b 48%,#ff6b00 100%);
  --shadow-sm:0 1px 2px rgba(0,0,0,.55),0 0 10px rgba(245,158,11,.08);
  --shadow-md:0 10px 34px rgba(0,0,0,.62),0 0 22px rgba(245,158,11,.2);
  --chip-bg:#2b1f0f; --chip-bd:#5a3b11; --chip-tx:#ffd166;
  --box-bg:#271b0c; --box-bd:#6b420d; --box-tx:#f2d28a;
  --mem-bg:#271b0c; --mem-bd:#6b420d; --mem-tx:#ffc65a;
  --on:#fbbf24; --off:#7c6a4a;
  --glass:rgba(18,14,8,.78);
}
body.github button{box-shadow:0 0 15px rgba(245,158,11,.34),0 0 30px rgba(255,107,0,.12),inset 0 1px 0 rgba(255,255,255,.22)}
body.github button:hover{filter:brightness(1.12);box-shadow:0 0 24px rgba(245,158,11,.62),0 0 46px rgba(255,107,0,.24),inset 0 1px 0 rgba(255,255,255,.25)}
body.github .back:hover,body.github .mini:hover,body.github .tab:hover,body.github .tag:hover,body.github .drop:hover,body.github #themePanel .thopt:hover{box-shadow:0 0 16px rgba(245,158,11,.28)}
body.github .card:hover{box-shadow:var(--shadow-md),0 0 24px rgba(245,158,11,.28);border-color:var(--border-hi)}
body.github .card.sel,body.github .tab.on,body.github .tag.on{box-shadow:0 0 24px rgba(245,158,11,.34)}
body.github textarea:focus,body.github input:focus,body.github select:focus{box-shadow:0 0 0 4px rgba(245,158,11,.2),0 0 18px rgba(245,158,11,.2)}
body.github::before,body.github::after{opacity:.07}
body.github #fxLayer .grid{opacity:.62}
body.github #fxLayer .github-hole-canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
/* ============ 主题4: 绵羊·暖白(body.sheep) ============ */
body.sheep{
  --bg:linear-gradient(165deg,#f4f0e6 0%,#f0ead9 52%,#e8efe2 100%);
  --card:#fffdf6; --card-sel:#f4edda; --border:#ddd4c0; --border-hi:#b58a4a;
  --text:#3c382d; --muted:#6e6854; --faint:#a29b85;
  --input:#fbf8ee; --accent:#c08a34; --accent2:#7fa06b;
  --grad:linear-gradient(120deg,#c08a34,#7fa06b);
  --btn-grad:linear-gradient(120deg,#cf9a45 0%,#b07f2e 60%,#8fa96f 100%);
  --shadow-sm:0 1px 2px rgba(110,90,40,.08),0 2px 10px rgba(110,90,40,.07);
  --shadow-md:0 10px 30px rgba(120,95,40,.16),0 2px 8px rgba(120,95,40,.1);
  --chip-bg:#f3edda; --chip-bd:#e2d8ba; --chip-tx:#8a6a26;
  --box-bg:#f8f2de; --box-bd:#e6d9ae; --box-tx:#6d5a26;
  --mem-bg:#f7f0da; --mem-bd:#e4d5ac; --mem-tx:#8a6a20;
  --on:#4c9a52; --off:#a8a493;
  --glass:rgba(255,253,245,.8);
}
body.sheep::before,body.sheep::after{opacity:.35}
/* ============ 主题5: 向日葵·农家乐(body.sunny，覆盖原薄荷主题) ============ */
body.sunny{
  --bg:linear-gradient(180deg,#fff1bc 0%,#ffe4a0 38%,#f6c875 67%,#b8753e 100%);
  --card:#fffdf5; --card-sel:#fff0bd; --border:#d9a84f; --border-hi:#a95d20;
  --text:#352214; --muted:#604023; --faint:#735234;
  --input:#fff8e8; --accent:#c8752d; --accent2:#e49a27;
  --grad:linear-gradient(120deg,#f4b52d,#d97824 58%,#9c542f);
  --btn-grad:linear-gradient(120deg,#f3bd3c 0%,#df8327 58%,#a95b32 100%);
  --shadow-sm:0 1px 2px rgba(111,64,24,.12),0 4px 14px rgba(111,64,24,.1);
  --shadow-md:0 12px 32px rgba(94,52,20,.18),0 3px 12px rgba(94,52,20,.12);
  --chip-bg:#fff0c2; --chip-bd:#e8c879; --chip-tx:#8d5a24;
  --box-bg:#fffaf0; --box-bd:#dbae59; --box-tx:#4f351c;
  --mem-bg:#fff0c9; --mem-bd:#e3bd6b; --mem-tx:#8b5d25;
  --on:#6c9a42; --off:#ae9879;
  --glass:rgba(255,253,245,.97);
}
body.sunny::before{opacity:.08;background:radial-gradient(42% 28% at 50% 0%,rgba(255,255,226,.9),transparent 72%)}
body.sunny::after{opacity:.16;background:linear-gradient(180deg,transparent 42%,rgba(126,69,32,.22) 100%)}
body.sunny .card:hover{box-shadow:var(--shadow-md),0 0 18px rgba(220,133,35,.2);border-color:var(--border-hi)}
body.sunny button:hover,body.sunny .tab.on{box-shadow:0 5px 18px rgba(199,111,34,.24)}
body.sunny .small{color:var(--muted)}
body.sunny h2,body.sunny h3{color:#3f2918;text-shadow:0 1px 0 rgba(255,255,255,.75)}
/* 农场壁纸：太阳缓慢横向移动，田垄和作物保持克制的装饰感 */
#fxLayer .farm-sun{position:absolute;top:7vh;left:0;width:58px;height:58px;border-radius:50%;
  background:radial-gradient(circle at 35% 32%,#fff8bf 0 12%,#ffd85a 42%,#f3a52b 100%);
  box-shadow:0 0 0 9px rgba(255,198,54,.14),0 0 34px rgba(255,176,35,.45);
  animation:sunWander 42s ease-in-out infinite alternate}
#fxLayer .farm-sun::before{content:'';position:absolute;inset:-16px;border-radius:50%;
  background:repeating-conic-gradient(from 0deg,rgba(255,193,47,.5) 0 7deg,transparent 7deg 22deg);z-index:-1}
@keyframes sunWander{from{transform:translateX(10vw) translateY(0)}to{transform:translateX(78vw) translateY(5vh)}}
#fxLayer .farm-hills{position:absolute;left:-8%;right:-8%;bottom:22%;height:30%;
  background:linear-gradient(145deg,transparent 0 34%,rgba(132,91,44,.3) 35% 58%,transparent 59%),
    linear-gradient(18deg,#a96b39 0 40%,#c58a4a 41% 68%,transparent 69%);border-radius:50% 50% 0 0/35% 35% 0 0;opacity:.18}
#fxLayer .farm-field{position:absolute;inset:auto -5% -5% -5%;height:32%;
  background:repeating-linear-gradient(172deg,rgba(102,56,29,.36) 0 3px,transparent 3px 27px),
    linear-gradient(180deg,#bd7b3c,#8e522f);transform:perspective(360px) rotateX(8deg);opacity:.2}
#fxLayer .farm-crop{position:absolute;bottom:18%;font-size:clamp(20px,2.5vw,34px);filter:drop-shadow(0 3px 3px rgba(73,39,18,.28));animation:cropSway 5.5s ease-in-out infinite alternate}
#fxLayer .farm-crop:nth-of-type(2){animation-delay:-1.7s}#fxLayer .farm-crop:nth-of-type(3){animation-delay:-3.1s}
@keyframes cropSway{from{transform:rotate(-2deg) translateY(1px)}to{transform:rotate(2deg) translateY(-3px)}}
#fxLayer .farm-animal{position:absolute;bottom:9%;font-size:clamp(24px,3vw,40px);filter:drop-shadow(0 4px 4px rgba(54,29,16,.3));opacity:.58;animation:animalWander 36s ease-in-out infinite alternate}
#fxLayer .farm-animal.pig{right:12%;animation-delay:-13s}#fxLayer .farm-animal.chicken{left:18%;font-size:clamp(18px,2vw,28px);animation-delay:-6s}
@keyframes animalWander{from{translate:-3vw 0}to{translate:7vw -3px}}
/* ============ 主题5: 深海·夜海(body.ocean) ============ */
body.ocean{
  color-scheme:dark;
  --bg:linear-gradient(180deg,#04121f 0%,#062233 42%,#020d16 100%);
  --card:#0a2331; --card-sel:#0e3044; --border:#14425a; --border-hi:#2aa7c9;
  --text:#d6edf6; --muted:#8fb9c9; --faint:#5a8291;
  --input:#07202e; --accent:#2fb9dd; --accent2:#3f7fd6;
  --grad:linear-gradient(120deg,#2fb9dd,#3f7fd6);
  --btn-grad:linear-gradient(120deg,#1b93b8 0%,#2f6fc0 60%,#3f8fd6 100%);
  --shadow-sm:0 1px 2px rgba(0,0,0,.5),0 1px 10px rgba(0,0,0,.4);
  --shadow-md:0 10px 34px rgba(0,0,0,.6),0 0 18px rgba(47,185,221,.16);
  --chip-bg:#0d3040; --chip-bd:#17506a; --chip-tx:#7fe0f4;
  --box-bg:#0d2c33; --box-bd:#1a5560; --box-tx:#b9e6d9;
  --mem-bg:#0f2b3a; --mem-bd:#1d5468; --mem-tx:#6fc7e8;
  --on:#3fd98f; --off:#5a7280;
  --glass:rgba(8,32,44,.74);
}
body.ocean::before,body.ocean::after{opacity:.08}
/* ============ 主题: 太阳系 · 八大行星(body.solar) ============ */
body.solar{
  color-scheme:dark;
  --bg:radial-gradient(900px 680px at 72% 46%,rgba(21,35,70,.32),transparent 65%),
       linear-gradient(160deg,#03050b 0%,#070b16 48%,#020309 100%);
  --card:#111725; --card-sel:#17243a; --border:#26344b; --border-hi:#63a7ff;
  --text:#e8eef9; --muted:#a9b8cf; --faint:#71829c;
  --input:#0c1220; --accent:#63a7ff; --accent2:#f0a84b;
  --grad:linear-gradient(120deg,#63a7ff,#8b7cf6 56%,#f0a84b);
  --btn-grad:linear-gradient(120deg,#3f83d8 0%,#6d62d5 58%,#d78b35 100%);
  --shadow-sm:0 1px 2px rgba(0,0,0,.58),0 2px 12px rgba(24,58,110,.18);
  --shadow-md:0 12px 36px rgba(0,0,0,.64),0 0 22px rgba(99,167,255,.18);
  --chip-bg:#17243a; --chip-bd:#2c4569; --chip-tx:#91c2ff;
  --box-bg:#191b23; --box-bd:#4c4c57; --box-tx:#d8d3c6;
  --mem-bg:#211b17; --mem-bd:#58402c; --mem-tx:#efbd7a;
  --on:#54dc91; --off:#657087;
  --glass:rgba(12,18,30,.82);
}
body.solar::before,body.solar::after{opacity:.04}
body.solar .card:hover{box-shadow:var(--shadow-md),0 0 24px rgba(99,167,255,.2)}
body.solar textarea:focus,body.solar input:focus,body.solar select:focus{box-shadow:0 0 0 4px rgba(99,167,255,.18)}
/* ============ 装饰层(流星雨/绵羊/深海,按主题由 JS 注入) ============ */
#fxLayer{position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden}
/* —— 极客: 网格 + 星星 + 流星 —— */
#fxLayer .grid{position:absolute;inset:0;
  background-image:linear-gradient(rgba(34,211,238,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(34,211,238,.05) 1px,transparent 1px);
  background-size:44px 44px}
#fxLayer .stars{position:absolute;inset:0;animation:starTwinkle 3.2s ease-in-out infinite alternate;
  background-image:
    radial-gradient(1.6px 1.6px at 12% 18%,rgba(255,255,255,.95),transparent 60%),
    radial-gradient(1.2px 1.2px at 26% 42%,rgba(190,225,255,.8),transparent 60%),
    radial-gradient(1.8px 1.8px at 41% 9%,rgba(255,255,255,.9),transparent 60%),
    radial-gradient(1.1px 1.1px at 55% 30%,rgba(190,225,255,.75),transparent 60%),
    radial-gradient(1.5px 1.5px at 68% 14%,rgba(255,255,255,.9),transparent 60%),
    radial-gradient(1.2px 1.2px at 79% 38%,rgba(190,225,255,.8),transparent 60%),
    radial-gradient(1.7px 1.7px at 88% 7%,rgba(255,255,255,.95),transparent 60%),
    radial-gradient(1.1px 1.1px at 8% 55%,rgba(190,225,255,.7),transparent 60%),
    radial-gradient(1.4px 1.4px at 94% 52%,rgba(255,255,255,.85),transparent 60%),
    radial-gradient(1.2px 1.2px at 48% 58%,rgba(190,225,255,.7),transparent 60%)}
@keyframes starTwinkle{from{opacity:.5}to{opacity:1}}
#fxLayer .meteor{position:absolute;width:150px;height:2px;border-radius:2px;opacity:0;
  background:linear-gradient(90deg,transparent,rgba(120,200,255,.45) 55%,rgba(190,235,255,.95));
  filter:drop-shadow(0 0 5px rgba(140,215,255,.8));
  animation:meteorFly linear infinite}
@keyframes meteorFly{
  0%{opacity:0;transform:rotate(135deg) translateX(0)}
  3%{opacity:1}
  24%{opacity:1;transform:rotate(135deg) translateX(46vmin)}
  27%,100%{opacity:0;transform:rotate(135deg) translateX(46vmin)}}
/* —— 绵羊: 云 + 小羊(纯CSS画的) —— */
#fxLayer .cloud{position:absolute;width:110px;height:32px;border-radius:999px;
  background:rgba(255,255,255,.92);
  box-shadow:26px -13px 0 5px rgba(255,255,255,.92),56px -3px 0 1px rgba(255,255,255,.92);
  animation:cloudDrift 52s linear infinite;opacity:.9}
#fxLayer .cloud.c1{top:10%}
#fxLayer .cloud.c2{top:24%;animation-duration:70s;animation-delay:-26s;scale:.68}
#fxLayer .cloud.c3{top:5%;animation-duration:60s;animation-delay:-44s;scale:.48}
@keyframes cloudDrift{from{translate:-160px 0}to{translate:110vw 0}}
#fxLayer .sheeppos{position:absolute;bottom:4.5%}
#fxLayer .sheepy{position:relative;width:86px;height:64px;animation:graze 4.6s ease-in-out infinite}
#fxLayer .sheepy .wool{position:absolute;left:0;top:2px;width:72px;height:44px;
  border-radius:48% 52% 50% 50%/60% 62% 40% 38%;background:#fdfcf5;
  box-shadow:-7px -4px 0 2px #fdfcf5,7px -5px 0 2px #fdfcf5,0 -8px 0 3px #f8f5ea,
    0 10px 12px rgba(96,84,52,.16)}
#fxLayer .sheepy .shead{position:absolute;right:0;top:14px;width:24px;height:26px;
  border-radius:46% 54% 52% 48%;background:#55493b;box-shadow:inset -2px -3px 0 rgba(0,0,0,.18)}
#fxLayer .sheepy .shead::before{content:'';position:absolute;left:-3px;top:-6px;width:15px;height:10px;
  border-radius:50%;background:#fdfcf5}
#fxLayer .sheepy .sear{position:absolute;right:15px;top:-2px;width:10px;height:5px;border-radius:50%;
  background:#463c2f;transform:rotate(-26deg)}
#fxLayer .sheepy .seye{position:absolute;right:6px;top:9px;width:5px;height:5px;border-radius:50%;background:#fff}
#fxLayer .sheepy .seye::after{content:'';position:absolute;left:1.6px;top:1.4px;width:2.2px;height:2.2px;
  border-radius:50%;background:#241f18}
#fxLayer .sheepy .sleg{position:absolute;bottom:3px;width:5px;height:13px;border-radius:3px;background:#4a4034}
#fxLayer .sheepy .sl1{left:15px}#fxLayer .sheepy .sl2{left:45px}
@keyframes graze{0%,100%{transform:translateY(0) rotate(0)}30%{transform:translateY(-5px) rotate(-1deg)}
  55%{transform:translateY(0)}74%{transform:translateY(-3px) rotate(1deg)}}
/* 绵羊: 整只羊在草地上缓慢飘来飘去(叠加内部 graze 吃草) */
@keyframes sheepWander{from{transform:translateX(-7vw)}to{transform:translateX(15vw)}}
/* —— 深海: 海面光 + 气泡 + 鱼群 + 大鱼吃小鱼 + 鲸鱼定时冲屏 + 北极星跳入 —— */
#fxLayer .sea{position:absolute;inset:0;
  background:radial-gradient(62% 42% at 50% -6%,rgba(96,196,236,.18) 0%,transparent 70%),
    linear-gradient(180deg,rgba(12,64,94,.14) 0%,transparent 42%,rgba(0,8,14,.38) 100%)}
#fxLayer .bub{position:absolute;bottom:-4vh;border-radius:50%;opacity:.5;
  background:radial-gradient(circle at 32% 30%,rgba(255,255,255,.55),rgba(255,255,255,.07));
  animation:bubUp linear infinite}
@keyframes bubUp{from{transform:translateY(0)}to{transform:translateY(-110vh)}}
#fxLayer .fish{position:absolute;line-height:1;will-change:transform;
  filter:drop-shadow(0 2px 5px rgba(0,10,18,.55))}
@keyframes swimR{0%{transform:translate(-9vw,0) scaleX(-1)}25%{transform:translate(20vw,-12px) scaleX(-1)}
  50%{transform:translate(48vw,7px) scaleX(-1)}75%{transform:translate(76vw,-10px) scaleX(-1)}
  100%{transform:translate(109vw,0) scaleX(-1)}}
@keyframes swimL{0%{transform:translate(109vw,0)}25%{transform:translate(76vw,11px)}
  50%{transform:translate(48vw,-9px)}75%{transform:translate(20vw,9px)}100%{transform:translate(-9vw,0)}}
#fxLayer .squid{opacity:.92}
#fxLayer .prey{animation:preyRun 26s linear infinite}
@keyframes preyRun{0%{transform:translate(109vw,0);opacity:0}5%{opacity:1}
  68%{transform:translate(31vw,-6px);opacity:1}73%{transform:translate(25vw,0);opacity:0}
  100%{transform:translate(25vw,0);opacity:0}}
#fxLayer .shark{animation:sharkChase 26s linear infinite;filter:drop-shadow(0 3px 8px rgba(0,12,22,.7))}
@keyframes sharkChase{0%{transform:translate(122vw,5px);opacity:0}5%{opacity:1}
  71%{transform:translate(28vw,2px);opacity:1}82%{transform:translate(8vw,0);opacity:1}
  100%{transform:translate(-16vw,-4px);opacity:0}}
#fxLayer .whale{top:11%;font-size:72px;animation:whaleRush 48s linear infinite}
@keyframes whaleRush{0%,86%{transform:translate(-32vw,6vh) scale(.4) scaleX(-1);opacity:0}
  89%{opacity:.9}95%{transform:translate(50vw,2vh) scale(1.6) scaleX(-1);opacity:1}
  100%{transform:translate(124vw,-2vh) scale(2.1) scaleX(-1);opacity:0}}
#fxLayer .polar{font-size:26px;filter:drop-shadow(0 0 9px rgba(255,240,170,.95));
  animation:polarSwim 38s ease-in-out infinite}
@keyframes polarSwim{0%{transform:translate(0,-9vh) scale(.5) rotate(0);opacity:0}
  7%{opacity:1}18%{transform:translate(-4vw,32vh) scale(1) rotate(180deg)}
  24%{transform:translate(-9vw,34vh) rotate(160deg)}62%{transform:translate(-30vw,28vh) rotate(200deg)}
  100%{transform:translate(-74vw,22vh) rotate(180deg);opacity:0}}
/* —— 太阳系: 太阳 + 八条椭圆轨道 + 八大行星 —— */
#fxLayer .solar-space{position:absolute;inset:0;background:
  radial-gradient(1px 1px at 7% 14%,#fff,transparent),radial-gradient(1px 1px at 18% 74%,#a8c9ff,transparent),
  radial-gradient(1.5px 1.5px at 29% 32%,#fff,transparent),radial-gradient(1px 1px at 43% 81%,#e4edff,transparent),
  radial-gradient(1px 1px at 58% 9%,#fff,transparent),radial-gradient(1.5px 1.5px at 67% 67%,#b7d3ff,transparent),
  radial-gradient(1px 1px at 78% 24%,#fff,transparent),radial-gradient(1px 1px at 91% 72%,#d8e6ff,transparent),
  radial-gradient(1.5px 1.5px at 96% 11%,#fff,transparent);background-size:310px 250px;opacity:.64}
#fxLayer .solar-system{position:absolute;left:72%;top:50%;width:min(88vmin,760px);aspect-ratio:1;
  transform:translate(-50%,-50%) rotateX(61deg);transform-style:preserve-3d;opacity:.82}
#fxLayer .solar-sun{position:absolute;left:50%;top:50%;width:clamp(38px,7vmin,66px);aspect-ratio:1;border-radius:50%;
  transform:translate(-50%,-50%) rotateX(-61deg);z-index:20;
  background:radial-gradient(circle at 34% 30%,#fffbd0 0 8%,#ffd45d 26%,#f69a26 62%,#d64a10 100%);
  box-shadow:0 0 18px 7px rgba(255,184,62,.72),0 0 54px 20px rgba(255,112,24,.33),0 0 110px 38px rgba(255,74,15,.1);
  animation:solarPulse 5s ease-in-out infinite alternate}
@keyframes solarPulse{from{filter:brightness(.95)}to{filter:brightness(1.16)}}
#fxLayer .orbit{--size:30%;--dur:12s;position:absolute;left:50%;top:50%;width:var(--size);aspect-ratio:1;
  border:1px solid rgba(154,183,224,.2);border-radius:50%;transform-style:preserve-3d;
  animation:solarOrbit var(--dur) linear infinite}
@keyframes solarOrbit{from{transform:translate(-50%,-50%) rotateZ(0)}to{transform:translate(-50%,-50%) rotateZ(360deg)}}
#fxLayer .planet{--planet:8px;position:absolute;left:50%;top:0;width:var(--planet);height:var(--planet);
  border-radius:50%;transform:translate(-50%,-50%) rotateX(-61deg);box-shadow:inset -2px -2px 3px rgba(0,0,0,.48),0 0 5px rgba(255,255,255,.2)}
#fxLayer .mercury{--size:19%;--dur:9s}.mercury .planet{--planet:5px;background:#a7a39c}
#fxLayer .venus{--size:27%;--dur:14s}.venus .planet{--planet:8px;background:linear-gradient(135deg,#f4d08b,#b96e35)}
#fxLayer .earth{--size:36%;--dur:20s}.earth .planet{--planet:9px;background:radial-gradient(circle at 38% 35%,#79bf65 0 18%,#2e85cf 23% 70%,#173c7b)}
#fxLayer .mars{--size:45%;--dur:27s}.mars .planet{--planet:7px;background:linear-gradient(135deg,#e38258,#913820)}
#fxLayer .jupiter{--size:57%;--dur:39s}.jupiter .planet{--planet:18px;background:repeating-linear-gradient(180deg,#e2bf91 0 3px,#a96c4b 3px 5px,#f0d4ac 5px 8px)}
#fxLayer .saturn{--size:69%;--dur:52s}.saturn .planet{--planet:15px;background:linear-gradient(180deg,#ead19c,#a88754)}
#fxLayer .saturn .planet::after{content:'';position:absolute;left:50%;top:50%;width:25px;height:8px;border:2px solid rgba(231,208,159,.82);border-radius:50%;transform:translate(-50%,-50%) rotate(-18deg)}
#fxLayer .uranus{--size:81%;--dur:68s}.uranus .planet{--planet:11px;background:linear-gradient(135deg,#b8f0ed,#4aa5b4)}
#fxLayer .neptune{--size:94%;--dur:86s}.neptune .planet{--planet:11px;background:linear-gradient(135deg,#5b9bff,#203a9d)}
@media(max-width:720px){#fxLayer .solar-system{left:70%;top:35%;width:94vmin;opacity:.66}}
/* ============ 自绘提示框(替代原生 alert/confirm/prompt) ============ */
.dlgmask{position:fixed;inset:0;z-index:1000;background:rgba(8,12,20,.42);
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
  display:flex;align-items:center;justify-content:center;animation:dlgFade .16s ease both}
.dlg{width:min(420px,88vw);background:var(--card);border:1px solid var(--border-hi);border-radius:16px;
  padding:20px 22px;box-shadow:var(--shadow-md);color:var(--text);
  animation:dlgPop .24s cubic-bezier(.2,.9,.3,1.25) both}
.dlg .dmsg{font-size:14.5px;line-height:1.7;word-break:break-word}
.dlg .dbtns{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}
.dlg .dbtns button{padding:9px 22px;font-size:14px}
.dlg input{margin-top:12px}
@keyframes dlgFade{from{opacity:0}to{opacity:1}}
@keyframes dlgPop{from{opacity:0;transform:scale(.85) translateY(12px)}to{opacity:1;transform:scale(1) translateY(0)}}
/* ============ 主题选择面板 ============ */
#themePanel{position:fixed;top:66px;right:16px;z-index:99;background:var(--glass);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow-md);padding:8px;
  animation:dlgPop .2s cubic-bezier(.2,.9,.3,1.2) both}
#themePanel .thopt{padding:9px 16px;border-radius:10px;font-size:13.5px;cursor:pointer;
  color:var(--text);transition:.15s;white-space:nowrap}
#themePanel .thopt:hover{background:var(--card-sel);transform:translateX(-2px)}
#themePanel .thopt.cur{background:var(--grad);color:#fff;font-weight:600}
#themePanel .thgroup{margin:8px 7px 4px;color:var(--faint);font-size:10px;font-weight:700;letter-spacing:1.2px}
#themePanel .thgroup:first-child{margin-top:3px}
/* ---- 视频测试表单: 紧凑、可扫描、移动端不挤成一团 ---- */
.test-hub-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.test-entry{min-height:112px;display:flex;flex-direction:column;justify-content:center;gap:5px}
.vt-page h2{margin-top:10px}
.vt-page .box{border-radius:8px;padding:14px 16px;margin:11px 0}
.vt-page .card{border-radius:8px;padding:14px 16px}
.vt-panel-title{font-size:13px;font-weight:700;color:var(--text);margin-bottom:8px}
.vt-help{margin:8px 0 14px}
.vt-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;align-items:end;margin:12px 0}
.vt-form-grid>span{min-width:0}
.vt-choice-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:7px}
.vt-choice-grid-short{grid-template-columns:repeat(auto-fill,minmax(92px,1fr))}
#app:has([id^="vt"]) .vt-choice-grid>label.vt-choice{display:flex;margin:0;min-height:42px}
#app:has([id^="vt"]) .vt-choice small{display:block;color:var(--faint);font-size:11px;margin-top:2px}
.vt-toggle{display:flex!important;align-items:center;gap:7px;margin:0!important;padding:10px;border:1px solid var(--border);border-radius:7px;background:var(--input);color:var(--text)!important}
#app:has([id^="vt"]) input[type=text],#app:has([id^="vt"]) input[type=number],#app:has([id^="vt"]) select,
.vt-page input[type=text],.vt-page input[type=number],.vt-page select{min-height:40px;border-radius:7px;padding:9px 11px}
#app:has([id^="vt"]) textarea,.vt-page textarea{min-height:120px;border-radius:7px;padding:11px 12px}
#app:has([id^="vt"]) .box>div>label{display:inline-flex;align-items:center;gap:7px;margin:4px 6px 4px 0;padding:8px 10px;border:1px solid var(--border);border-radius:7px;background:var(--input);color:var(--text);font-weight:500;cursor:pointer;transition:.15s}
#app:has([id^="vt"]) .box>div>label:has(input:checked){border-color:var(--border-hi);background:var(--card-sel);color:var(--accent);box-shadow:0 0 0 2px rgba(95,135,255,.12)}
#app:has([id^="vt"]) input[type=checkbox]{flex:0 0 auto}
#app:has([id^="vt"]) .row>span{flex:1 1 180px;min-width:150px}
#app:has([id^="vt"]) .row>span>input,#app:has([id^="vt"]) .row>span>select{width:100%;margin-top:4px}
.vt-page .row,#app:has([id^="vt"]) .row{gap:14px 16px}
.vt-page .row button{min-height:40px;padding:9px 14px}
#app:has([id^="vt"]) p{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-top:18px}
#app:has([id^="vt"]) p>button{margin:0}
.vt-page .test-entry:hover{transform:translateY(-2px)}
.vt-name-list{display:grid;gap:9px;margin:10px 0 16px}
.vt-name-row{display:grid;grid-template-columns:minmax(180px,.8fr) minmax(260px,1.2fr);gap:14px;align-items:center;
  padding:11px 13px;background:var(--input);border:1px solid var(--border);border-radius:9px}
.vt-name-model{display:flex;align-items:center;gap:10px;min-width:0}
.vt-name-index{display:inline-flex;align-items:center;justify-content:center;flex:0 0 30px;height:30px;border-radius:7px;
  background:var(--chip-bg);border:1px solid var(--chip-bd);color:var(--accent);font:700 12px ui-monospace,monospace}
.vt-name-model strong{display:block;color:var(--text);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vt-name-model small{display:block;margin-top:2px;color:var(--faint);font-size:11px}
.vt-name-control label{margin:0 0 4px;font-size:11px;color:var(--faint);font-weight:600}
.vt-name-control{min-width:0}
.vt-name-control-line{display:flex;align-items:center;gap:8px}
.vt-name-control-line .vt-pretty-input{flex:1;min-width:0}
.vt-rename-btn{flex:0 0 auto;min-height:38px!important;padding:8px 11px!important;border-radius:7px!important;font-size:12px!important}
.vt-pretty-input{display:flex;align-items:stretch;overflow:hidden;min-height:43px;background:var(--input);
  border:1px solid var(--border);border-radius:8px;box-shadow:inset 0 1px 2px rgba(0,0,0,.08);
  transition:border-color .16s,box-shadow .16s,background .16s}
.vt-pretty-input:focus-within{border-color:var(--border-hi);background:var(--card-sel);
  box-shadow:0 0 0 3px rgba(95,135,255,.15),inset 0 1px 2px rgba(0,0,0,.05)}
.vt-field-icon{display:flex;align-items:center;justify-content:center;flex:0 0 40px;border-right:1px solid var(--border);
  background:var(--chip-bg);color:var(--accent);font-size:15px;line-height:1;user-select:none}
.vt-pretty-input input{min-width:0!important;min-height:41px!important;border:0!important;border-radius:0!important;
  background:transparent!important;box-shadow:none!important;padding:9px 11px!important;color:var(--text)}
.vt-input-shell{position:relative}
.vt-input-shell input{padding-right:30px!important}
.vt-folder-tail{position:absolute;right:11px;top:50%;transform:translateY(-50%);color:var(--faint);font-size:12px;pointer-events:none}
.vt-name-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:4px 0 8px}
.vt-name-option{padding:12px;background:var(--card);border:1px solid var(--border);border-radius:9px}
.vt-name-option label{margin:0 0 5px;color:var(--muted);font-size:12px}
.vt-name-option .small{margin-top:5px}
.vt-setting-note{margin:-4px 0 4px;padding:9px 11px;border-left:3px solid var(--accent);background:var(--input);border-radius:6px}
.vt-name-option-actions{display:flex;gap:7px;margin-top:7px}
.vt-name-option-actions button{min-height:32px;padding:5px 11px;font-size:12px;border-radius:7px}
.vt-nav-actions{display:flex;gap:16px;align-items:center;margin-top:18px}
.vt-nav-actions button{min-width:112px}
@media(max-width:620px){
  .test-hub-grid{grid-template-columns:1fr}
  .vt-form-grid{grid-template-columns:1fr}
  .vt-name-row,.vt-name-options{grid-template-columns:1fr}
  .vt-name-control-line{align-items:stretch}.vt-name-control-line .vt-pretty-input{min-width:0}
  .vt-rename-btn{min-width:72px!important}
  #app:has([id^="vt"]) .row>span{flex-basis:100%;min-width:0}
  .vt-page .row button{flex:1 1 auto}
  #app:has([id^="vt"]) p{gap:9px}
}
@media(max-width:820px){.vid-workspace{grid-template-columns:1fr}.vid-side{position:static;border-left:0;border-top:1px solid var(--border);padding:16px 0 0}}
/* 主题过渡：从右上角主题按钮所在的屏幕右上角向左展开，禁用浏览器默认淡入 */
::view-transition-old(root){animation:none}
::view-transition-new(root){animation:themeReveal .56s cubic-bezier(.4,0,.2,1) both}
@keyframes themeReveal{
  from{clip-path:circle(0 at 100% 0)}
  to{clip-path:circle(150% at 100% 0)}
}
/* ============ 翻书翻页 + 模型卡片掉落 + 点击反馈 ============ */
#app.flipin{animation:pageIn .45s cubic-bezier(.25,.8,.3,1) both;transform-origin:left center}
@keyframes pageIn{from{opacity:0;transform:perspective(1400px) rotateY(-7deg) translateX(30px)}
  to{opacity:1;transform:perspective(1400px) rotateY(0) translateX(0)}}
@keyframes cardDrop{from{opacity:0;transform:translateY(-52px) rotate(-1.2deg) scale(.96)}
  to{opacity:1;transform:translateY(0) rotate(0) scale(1)}}
#app .card.drop-in{animation:cardDrop .55s cubic-bezier(.2,.85,.3,1.12) both}
.card:active{transform:scale(.975);transition-duration:.06s}
/* 尊重系统减弱动效设置 */
@media (prefers-reduced-motion:reduce){
  body::before,body::after,h1,.bar>i::after,.spin{animation:none!important}
  #app .card,#app .box,#app .logbox,#app .tabs{animation:none!important}
  #app.flipin,#app .card.drop-in{animation:none!important}
  ::view-transition-old(root),::view-transition-new(root){animation:none!important}
  #fxLayer .github-hole-canvas{display:none!important}
  #fxLayer .meteor,#fxLayer .cloud,#fxLayer .sheepy,#fxLayer .sheeppos,#fxLayer .stars{animation:none!important}
  #fxLayer .fish,#fxLayer .bub{animation:none!important;display:none}
  #fxLayer .orbit,#fxLayer .solar-sun{animation:none!important}
  #fxLayer .meteor{display:none}
}
</style></head><body>
<button id="themeBtn" onclick="toggleTheme()" title="切换主题">🌙</button>
<button id="testBtn" onclick="goStep(70)" title="测试场(图片/语音)">⚙️</button>
<h1>🎨 ImageVideoStudio · 生图/生视频小助手</h1>
<div id="app"></div>
<script>
const $ = s => document.querySelector(s);
let ST = {step:0, model:null, mname:'', msec:0, items:[], pids:[], hasCN:false, test:null, testEditId:''};
// 全局兜底: 图片拖到上传区以外时,阻止浏览器直接打开图片导致页面跳走
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>e.preventDefault());
const NEG_DEF = %NEG%;
const SIZES = [[640,960,'竖屏·最小可用(SDXL 下限,再小会糊)'],[768,1152,'竖屏2:3 全身'],[832,1216,'竖屏·漫画标准'],[1024,1024,'方形1:1 半身/头像'],[1152,768,'横屏3:2 宽场景'],[1024,576,'横屏16:9 场景']];
// 提示词快捷标签(英文): 点一下自动加进对应框,再点一下取消
const POS_TAGS=['masterpiece','best quality','ultra detailed','8k','high resolution','sharp focus','cinematic lighting','photorealistic','anime style','depth of field','soft lighting','vibrant colors'];
const NEG_TAGS=['low quality','worst quality','blurry','bad anatomy','extra limbs','extra fingers','missing fingers','deformed hands','poorly drawn face','duplicate','watermark','text','cropped','jpeg artifacts','low resolution'];
async function j(u, opt){ const r = await fetch(u, opt); return r.json(); }
// ============ 自绘提示框: 替代原生 alert/confirm/prompt(跟随主题) ============
function _dlgOpen(inner){
  const mask=document.createElement('div'); mask.className='dlgmask';
  mask.innerHTML=`<div class="dlg">${inner}</div>`;
  document.body.appendChild(mask);
  return mask;
}
function cAlert(msg){
  return new Promise(res=>{
    const mask=_dlgOpen(`<div class="dmsg">${esc(msg)}</div><div class="dbtns"><button class="dok">知道了</button></div>`);
    const done=()=>{ mask.remove(); res(); };
    mask.querySelector('.dok').onclick=done;
    mask.onmousedown=e=>{ if(e.target===mask) done(); };
    mask.querySelector('.dok').focus();
  });
}
function cConfirm(msg, okText){
  return new Promise(res=>{
    const mask=_dlgOpen(`<div class="dmsg">${esc(msg)}</div><div class="dbtns">
      <button class="back dcancel">取消</button><button class="dok">${esc(okText||'确定')}</button></div>`);
    const done=v=>{ mask.remove(); res(v); };
    mask.querySelector('.dok').onclick=()=>done(true);
    mask.querySelector('.dcancel').onclick=()=>done(false);
    mask.onmousedown=e=>{ if(e.target===mask) done(false); };
    mask.querySelector('.dok').focus();
  });
}
function cPrompt(msg, def){
  return new Promise(res=>{
    const mask=_dlgOpen(`<div class="dmsg">${esc(msg)}</div>
      <input class="dinp" type="text" value="${esc(def||'')}">
      <div class="dbtns"><button class="back dcancel">取消</button><button class="dok">确定</button></div>`);
    const inp=mask.querySelector('.dinp');
    const done=v=>{ mask.remove(); res(v); };
    mask.querySelector('.dok').onclick=()=>done(inp.value);
    mask.querySelector('.dcancel').onclick=()=>done(null);
    mask.onmousedown=e=>{ if(e.target===mask) done(null); };
    inp.onkeydown=e=>{ if(e.key==='Enter') done(inp.value); if(e.key==='Escape') done(null); };
    inp.focus(); inp.select();
  });
}
// ============ 主题: 白天/黑夜/极客流星雨/绵羊暖白 ============
const THEMES=[
  ['light','☀️ 白天 · 清透','日间'],['sunny','🌻 农家乐 · 向日葵','日间'],['sheep','🐑 绵羊 · 暖白','日间'],
  ['dark','🌙 黑夜 · 护眼','夜间'],['ocean','🌊 深海 · 夜海','夜间'],['solar','🪐 太阳系 · 八大行星','夜间'],
  ['tech','🌠 极客 · 流星雨','科技'],['github','🐙 GitHub · 夜间科技','科技']
];
const THEME_ICON={light:'🌙',dark:'☀️',tech:'🌠',sheep:'🐑',sunny:'🌻',ocean:'🌊',solar:'🪐',github:'🐙'};
const THEME_CLASSES=THEMES.map(t=>t[0]).filter(k=>k!=='light');
const __mq = (window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null);
const __reduceMotion=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)');
const githubBlackHoleFx={
  raf:0,canvas:null,ctx:null,particles:[],mouse:{x:0,y:0,until:0},hole:null,last:0,resize:null,move:null,
  stop(){
    if(this.raf) cancelAnimationFrame(this.raf);
    if(this.resize) window.removeEventListener('resize',this.resize);
    if(this.move) window.removeEventListener('pointermove',this.move);
    if(this.canvas) this.canvas.remove();
    this.raf=0;this.canvas=this.ctx=this.resize=this.move=null;this.particles=[];this.hole=null;
  },
  start(fx){
    this.stop();
    if(__reduceMotion&&__reduceMotion.matches) return;
    const canvas=document.createElement('canvas'); canvas.className='github-hole-canvas'; fx.appendChild(canvas);
    const ctx=canvas.getContext('2d'); if(!ctx) return;
    this.canvas=canvas;this.ctx=ctx;
    const reset=(p,far)=>{
      const a=Math.random()*Math.PI*2,d=far?Math.max(innerWidth,innerHeight)*(.38+Math.random()*.38):Math.random()*Math.max(innerWidth,innerHeight);
      p.x=innerWidth/2+Math.cos(a)*d;p.y=innerHeight/2+Math.sin(a)*d;
      p.vx=(Math.random()-.5)*.28;p.vy=(Math.random()-.5)*.28;p.size=1+Math.random()*1.8;p.alpha=.35+Math.random()*.65;
    };
    const fit=()=>{
      const ratio=Math.min(devicePixelRatio||1,1.5); canvas.width=Math.round(innerWidth*ratio);canvas.height=Math.round(innerHeight*ratio);
      ctx.setTransform(ratio,0,0,ratio,0,0);
    };
    fit(); this.resize=fit; window.addEventListener('resize',fit);
    const h={x:innerWidth*.68,y:innerHeight*.38,tx:innerWidth*.68,ty:innerHeight*.38,next:performance.now()+6000}; this.hole=h;
    const count=innerWidth<700?48:72;
    for(let i=0;i<count;i++){const p={};reset(p,false);this.particles.push(p);}
    this.move=e=>{this.mouse.x=e.clientX;this.mouse.y=e.clientY;this.mouse.until=performance.now()+800;};
    window.addEventListener('pointermove',this.move,{passive:true});
    const draw=now=>{
      if(!this.canvas||document.hidden){this.raf=requestAnimationFrame(draw);return;}
      if(now-this.last<32){this.raf=requestAnimationFrame(draw);return;} this.last=now;
      if(now>h.next){h.tx=innerWidth*(.18+Math.random()*.64);h.ty=innerHeight*(.18+Math.random()*.58);h.next=now+8000+Math.random()*6000;}
      h.x+=(h.tx-h.x)*.006;h.y+=(h.ty-h.y)*.006;
      ctx.clearRect(0,0,innerWidth,innerHeight);
      for(const p of this.particles){
        const dx=h.x-p.x,dy=h.y-p.y,d2=dx*dx+dy*dy+1800,force=Math.min(.95,4200/d2);
        p.vx+=dx*force*.018-dy*force*.008;p.vy+=dy*force*.018+dx*force*.008;
        if(now<this.mouse.until){const mx=this.mouse.x-p.x,my=this.mouse.y-p.y,md=Math.hypot(mx,my);if(md<165){const mf=(1-md/165)*.032;p.vx+=mx*mf;p.vy+=my*mf;}}
        p.vx*=.986;p.vy*=.986;p.x+=p.vx;p.y+=p.vy;
        if(Math.hypot(dx,dy)<24||p.x<-20||p.x>innerWidth+20||p.y<-20||p.y>innerHeight+20) reset(p,true);
        ctx.fillStyle=`rgba(255,${150+Math.round(p.alpha*75)},${40+Math.round(p.alpha*100)},${p.alpha})`;
        ctx.fillRect(p.x,p.y,p.size,p.size);
      }
      const disk=ctx.createRadialGradient(h.x,h.y,12,h.x,h.y,68);disk.addColorStop(0,'rgba(0,0,0,1)');disk.addColorStop(.3,'rgba(0,0,0,.96)');disk.addColorStop(.55,'rgba(255,107,0,.34)');disk.addColorStop(.72,'rgba(255,209,102,.14)');disk.addColorStop(1,'rgba(255,175,40,0)');
      ctx.fillStyle=disk;ctx.beginPath();ctx.arc(h.x,h.y,68,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#020202';ctx.beginPath();ctx.arc(h.x,h.y,22,0,Math.PI*2);ctx.fill();
      this.raf=requestAnimationFrame(draw);
    };
    this.raf=requestAnimationFrame(draw);
  }
};
function buildFx(name){ // 按主题注入装饰层(极客=网格+星星+流星;绵羊=云+小羊)
  githubBlackHoleFx.stop();
  let fx=document.getElementById('fxLayer');
  if(!fx){ fx=document.createElement('div'); fx.id='fxLayer'; document.body.appendChild(fx); }
  if(name==='tech'){
    fx.innerHTML='<div class="grid"></div><div class="stars"></div>'+
      [[6,4,'0s','7s'],[22,16,'2.6s','9s'],[38,6,'5.4s','8s'],[55,24,'1.2s','10.5s'],
       [70,10,'6.8s','6.5s'],[86,30,'3.8s','11s'],[14,42,'-4.2s','8.5s'],[31,58,'-7.1s','9.8s'],
       [49,38,'-2.4s','7.4s'],[67,54,'-5.6s','10.8s'],[91,64,'-8.3s','8.9s']].map(m=>
      `<div class="meteor" style="left:${m[0]}%;top:${m[1]}%;animation-delay:${m[2]};animation-duration:${m[3]}"></div>`).join('');
  }else if(name==='github'){
    fx.innerHTML='<div class="grid"></div><div class="stars"></div>';
    githubBlackHoleFx.start(fx);
  }else if(name==='sheep'){
    const sheep=`<div class="wool"></div><div class="shead"><i class="sear"></i><i class="seye"></i></div><i class="sleg sl1"></i><i class="sleg sl2"></i>`;
    fx.innerHTML='<div class="cloud c1"></div><div class="cloud c2"></div><div class="cloud c3"></div>';
      [['5%','1','0s','34s'],['70%','.74','1.4s','47s'],['36%','.56','2.6s','40s']].map(s=>
      `<div class="sheeppos" style="left:${s[0]};scale:${s[1]};animation:sheepWander ${s[3]} ease-in-out infinite alternate;animation-delay:-${parseFloat(s[3])/2}s"><div class="sheepy" style="animation-delay:${s[2]}">${sheep}</div></div>`).join('');
  }else if(name==='sunny'){
    fx.innerHTML='<div class="farm-sun"></div><div class="farm-hills"></div><div class="farm-field"></div>'+
      '<span class="farm-crop" style="left:10%">🌻</span><span class="farm-crop" style="left:47%">🥕</span><span class="farm-crop" style="left:76%">🌻</span>'+
      '<span class="farm-animal chicken">🐔</span><span class="farm-animal pig">🐖</span>';
  }else if(name==='ocean'){
    fx.innerHTML='<div class="sea"></div><div class="stars"></div>'+
      [[8,'9px','13s','0s'],[20,'6px','17s','-6s'],[46,'8px','15s','-3s'],[70,'5px','19s','-9s'],[88,'7px','14s','-4s']].map(b=>
      `<span class="bub" style="left:${b[0]}%;width:${b[1]};height:${b[1]};animation-duration:${b[2]};animation-delay:${b[3]}"></span>`).join('')+
      [['🐟','30%','22px','26s','0s',1],['🐠','48%','18px','32s','-9s',0],['🐟','62%','15px','24s','-14s',1],['🐡','22%','16px','38s','-5s',0],['🐠','74%','20px','29s','-19s',1]].map(f=>
      `<span class="fish" style="top:${f[1]};font-size:${f[2]};animation:${f[5]?'swimR':'swimL'} ${f[3]} linear infinite;animation-delay:${f[4]}">${f[0]}</span>`).join('')+
      `<span class="fish squid" style="top:56%;font-size:30px;animation:swimL 55s linear infinite;animation-delay:-20s">🦑</span>`+
      `<span class="fish prey" style="top:40%;font-size:16px">🐟</span>`+
      `<span class="fish shark" style="top:38.5%;font-size:42px">🦈</span>`+
      `<span class="fish whale">🐋</span>`+
      `<span class="fish polar" style="left:62%">⭐</span>`;
  }else if(name==='solar'){
    const planets=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune'];
    fx.innerHTML='<div class="solar-space"></div><div class="solar-system"><div class="solar-sun"></div>'+
      planets.map((p,i)=>`<div class="orbit ${p}" style="animation-delay:-${i*4.7}s"><i class="planet"></i></div>`).join('')+'</div>';
  }else fx.innerHTML='';
}
function applyTheme(name){
  if(!THEMES.some(t=>t[0]===name)) name='light';
  document.body.classList.remove(...THEME_CLASSES);
  if(name!=='light') document.body.classList.add(name);
  const b=$('#themeBtn'); if(b){ b.textContent=THEME_ICON[name]; b.title='切换主题(当前: '+THEMES.find(t=>t[0]===name)[1]+')'; }
  buildFx(name);
  const p=document.getElementById('themePanel');
  if(p) p.querySelectorAll('.thopt').forEach(o=>o.classList.toggle('cur', o.dataset.t===name));
}
function savedTheme(){ try{ return localStorage.getItem('ivs_theme'); }catch(e){ return null; } }
function setTheme(name){
  try{localStorage.setItem('ivs_theme',name);}catch(e){}
  const p=document.getElementById('themePanel'); if(p) p.remove();   // 点选项后收起面板
  const b=$('#themeBtn'); const r=b?b.getBoundingClientRect():{left:innerWidth-59,top:16,width:42,height:42};
  const cx=r.left+r.width/2, cy=r.top+r.height/2;
  const rx=Math.max(cx,innerWidth-cx), ry=Math.max(cy,innerHeight-cy);
  // 明确把揭幕中心锁定在右上角主题按钮，避免浏览器默认从中心淡入
  document.documentElement.style.setProperty('--theme-cx',`${cx}px`);
  document.documentElement.style.setProperty('--theme-cy',`${cy}px`);
  document.documentElement.style.setProperty('--theme-rx',`${rx}px`);
  document.documentElement.style.setProperty('--theme-ry',`${ry}px`);
  if(document.startViewTransition){
    document.startViewTransition(()=>applyTheme(name));
    return;
  }
  applyTheme(name);
}
function toggleTheme(){ // 点开/收起主题选择面板
  let p=document.getElementById('themePanel');
  if(p){ p.remove(); return; }
  const cur=[...document.body.classList].find(c=>THEME_CLASSES.includes(c))||'light';
  p=document.createElement('div'); p.id='themePanel';
  p.innerHTML=['日间','夜间','科技'].map(g=>`<div class="thgroup">${g}</div>`+
    THEMES.filter(t=>t[2]===g).map(t=>`<div class="thopt ${t[0]===cur?'cur':''}" data-t="${t[0]}" onclick="setTheme('${t[0]}')">${t[1]}</div>`).join('')).join('');
  document.body.appendChild(p);
  setTimeout(()=>document.addEventListener('mousedown',function h(e){ if(!p.contains(e.target)){ p.remove(); document.removeEventListener('mousedown',h); } }),0);
}
(function(){
  const t=savedTheme();                                  // null=没手动选过
  applyTheme(t || (__mq && __mq.matches ? 'dark' : 'light'));
  if(__mq && __mq.addEventListener) __mq.addEventListener('change', e=>{ if(!savedTheme()) applyTheme(e.matches?'dark':'light'); });
  if(__reduceMotion&&__reduceMotion.addEventListener) __reduceMotion.addEventListener('change', e=>{
    if(e.matches) githubBlackHoleFx.stop();
    else if(document.body.classList.contains('github')) buildFx('github');
  });
})();

async function home(){
  ST = {step:0, model:null, mname:'', msec:0, items:[], pids:[], hasCN:ST.hasCN, kind:ST.kind};
  const st = await j('/api/svc/state');
  const s = await j('/api/status');
  let lc={paused:false}; try{ lc=await j('/api/llm/stats'); }catch(e){}
  let ts={}; try{ ts=await j('/api/tts/state'); }catch(e){}
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
    : `<div class="card" onclick="enterVideo()"><b>🎬 视频生成</b> <span class="dot off">○ 未启动</span><div class="small">点击检查后台任务并进入；空闲时会启动视频服务</div></div>`;
  let llmCard;
  if(st.llm.running) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot on">● 运行中</span><div class="small">点击进入</div></div>`;
  else if(st.llm.alive_pid) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot off">◐ 加载中…</span><div class="small">点击进入查看</div></div>`;
  else if(lc.paused) llmCard = `<div class="card" onclick="goStep(22)"><b>💬 语言模型</b> <span class="dot off">⏸ 已暂停</span><div class="small">点击进入,可恢复</div></div>`;
  else llmCard = `<div class="card" onclick="goStep(20)"><b>💬 语言模型</b> <span class="dot off">○ 未启动</span><div class="small">点击选模型并启动</div></div>`;
  // 语音卡: 轻量(CPU),不参与内存互斥,点进去直接进页(页面里可启停)
  const ttsCard = ts.ready
    ? `<div class="card" onclick="goStep(60)"><b>🎙 语音模型</b> <span class="dot on">● 就绪</span><div class="small">点击进入 学声音/配音</div></div>`
    : (ts.running
      ? `<div class="card" onclick="goStep(60)"><b>🎙 语音模型</b> <span class="dot off">◐ 加载中…</span><div class="small">点击进入查看</div></div>`
      : `<div class="card" onclick="goStep(60)"><b>🎙 语音模型</b> <span class="dot off">○ 未启动</span><div class="small">点击进入(很轻,不影响其它服务)</div></div>`);
  $('#app').innerHTML = `<h2>选一个开始</h2>` + imgCard + vidCard + llmCard + ttsCard;
}
function goStep(n){ ST.step=n; render(); }
function enterVideo(){ ST._videoTaskOnly=false; startAndGo('vid',10); }
function vidTaskStateLabel(s){ return ({running:'运行中',queued:'排队中',done:'已完成',failed:'失败'})[s]||s||'未知'; }
function vidTaskCard(t){
  const state=t.state||'done', active=state==='running'||state==='queued';
  const title=esc(t.name||'未命名视频'), engine=t.engine==='video'?'普通视频模型':'视频';
  const id=encodeURIComponent(t.id||''), url=encodeURIComponent(t.url||''), name=encodeURIComponent(t.name||'视频');
  const action=active
    ? `<button onclick="vidOpenTask('${id}','${t.engine}', '${url}', '${state}', '${name}')">查看进度</button>`
    : (t.url?`<button onclick="vidOpenTask('${id}','${t.engine}', '${url}', '${state}', '${name}')">查看成片</button>`:'');
  return `<div class="vid-task"><div class="vid-task-title">${title}</div><div class="vid-task-meta"><span class="vid-state ${state}">${vidTaskStateLabel(state)}</span> · ${engine}${t.error?`<br><span style="color:#d03050">${esc(t.error)}</span>`:''}</div><div class="vid-task-actions">${action}</div></div>`;
}
async function vidTasksPanel(){
  const el=$('#vidTasksPanel'); if(!el)return;
  let d; try{ d=await j('/api/vid/tasks'); }catch(e){ el.innerHTML='<div class="vid-empty">任务状态暂时读不到，点刷新重试。</div>'; return; }
  const groups=[['running','正在运行'],['queued','排队中'],['done','已完成'],['failed','失败']];
  let html='';
  for(const [state,label] of groups){ const rows=(d.tasks||[]).filter(t=>t.state===state); if(rows.length) html+=`<div class="vid-task-group">${label} · ${rows.length}</div>`+rows.map(vidTaskCard).join(''); }
  el.innerHTML=html||'<div class="vid-empty">还没有视频任务。<br>生成过的视频会自动出现在这里。</div>';
  if(window.__vidTaskTimer) clearTimeout(window.__vidTaskTimer);
  if(ST.step===10) window.__vidTaskTimer=setTimeout(vidTasksPanel,5000);
}
function vidOpenTask(id,engine,url,state,name){
  id=decodeURIComponent(id); url=decodeURIComponent(url); name=decodeURIComponent(name);
  ST.vid=ST.vid||{}; ST.vid.pid=id; ST.vid.url=url||null; ST.vid.taskName=name;
  if(state==='done' && url){ ST.step=12; render(); }
  else { ST.step=11; render(); }
}
async function startAndGo(type, nextStep){
  $('#app').innerHTML = `<div class="box">🚀 正在启动${type==='img'?'生图':'生视频'}服务…<br><span class="small">会先自动停掉其他服务腾内存,首次加载模型约 1~2 分钟。</span><div class="bar" style="margin-top:12px"><i class="indet"></i></div><div class="small" style="margin-top:8px">已等待 <b id="waitSec">0</b> 秒</div></div>`;
  const r = await j('/api/svc/start?type='+type);
  if(r.error){ $('#app').innerHTML = `<div class="box">启动失败:${esc(r.error)}</div><p><button onclick="home()">返回</button></p>`; return; }
  let n=0;
  const t=setInterval(async()=>{
    let s; try{ s=await j('/api/svc/status?type='+type); }catch(e){ return; } n++;
    const ws=$('#waitSec'); if(ws) ws.textContent=n*3;
    if(s.running){ clearInterval(t); ST.step=nextStep; render(); }
    else if(!s.alive_pid && n>3){ clearInterval(t); $('#app').innerHTML='<div class="box">启动失败,请查看项目目录日志(comfy.log / comfy_vid.log)。</div><p><button onclick="home()">返回</button></p>'; }
    else if(n>80){ clearInterval(t); $('#app').innerHTML='<div class="box">启动超时,请查看日志。</div><p><button onclick="home()">返回</button></p>'; }
  },3000);
}
function pickImg(){ ST.step=1; render(); }

let __lastStep=-99;   // 翻书动效: 只有切换页面(step变化)才翻,同页轮询重渲不翻
async function render(){
  if(ST.step!==__lastStep){
    __lastStep=ST.step;
    const ap=$('#app'); ap.classList.remove('flipin'); void ap.offsetWidth; ap.classList.add('flipin');
  }
  if(ST.step===1){
    const ms = await j('/api/models');
    ST.hasCN = (await j('/api/status')).has_cn;
    $('#app').innerHTML = `<h2>第二步:选生图模型</h2>` + tabHtml(0) + `
      <p class="small">右上角竖条 = 生成速度;时间着色:<span class="mtime g">绿 &lt;3分钟</span> / <span class="mtime y">黄 3~5分钟</span> / <span class="mtime r">红 &gt;5分钟·慢</span></p>` + ms.map((m,mi)=>`
      <div class="card drop-in" style="animation-delay:${mi*80}ms" onclick="pickModel('${m.id}','${m.name}',${m.sec},'${m.kind}')">
        <span class="wifi ${m.speed||'mid'}" title="生成速度"><i></i><i></i><i></i></span>
        <b>${m.name}</b> <span class="small mtime ${secCls(m.sec)}">${m.time||('约'+m.sec+'秒')}/张</span>
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
  if(ST.step===90){ // 高质量图片: 先生成脸部参考,再锁脸做服装/身材版本
    const ms=await j('/api/models');
    const cps=ms.filter(m=>m.kind==='checkpoint'||m.hq_lock);
    const h=ST.hq||(ST.hq={});
    if(!h.model && cps.length){
      const pref=cps.find(m=>/lustify/i.test(m.id+' '+m.name))||cps.find(m=>/waiNSFW/i.test(m.id+' '+m.name))||cps.find(m=>/realvis|cyber/i.test(m.id+' '+m.name))||cps[0];
      h.model=pref.id;
    }
    if(!h.facePrompt) h.facePrompt='adult East Asian woman, elegant soft oval face, delicate refined facial features, almond-shaped eyes, defined nose, naturally full lips, long loose black hair flowing over the shoulders, calm confident expression, wearing an elegant high-collar red silk qipao, ethereal Chinese fairy-like beauty, luxury editorial fashion portrait, neutral porcelain skin tone, cool soft studio lighting, balanced white balance, clean light-gray background, photorealistic, highly detailed';
    if(!h.neg) h.neg=NEG_DEF+', blurry face, asymmetrical eyes, deformed hands, bad anatomy, yellow skin, orange skin, warm color cast, oversaturated skin, hair bun, updo, plastic doll face';
    const opts=cps.map(m=>`<option value="${m.id}" ${h.model===m.id?'selected':''}>${esc(m.name)}</option>`).join('');
    $('#app').innerHTML=`<h2>高质量图片</h2>${tabHtml(3)}
      <div class="box"><b>这个功能分两步:</b>先单独生成一张脸并保存,确认后再锁住这张脸生成旗袍和不同身材比例的版本。每个版本都会单独归档。</div>
      <label>生图模型(锁脸需要 SDXL 单文件模型)</label><select id="hqmodel" onchange="ST.hq.model=this.value">${opts}</select>
      <label>脸部参考提示词(系统已预填,可以修改)</label><textarea id="hqfacep" style="height:92px">${esc(h.facePrompt)}</textarea>
      <label>负向提示词</label><textarea id="hqneg" style="height:56px">${esc(h.neg)}</textarea>
      <div class="row"><span><label>脸部参考尺寸</label><select onchange="ST.hq.faceSize=this.value"><option value="1024x1024">1024×1024</option><option value="832x1216">832×1216</option></select></span>
      <span class="small" style="align-self:flex-end">先生成脸,满意后再进入身材参数</span></div>
      <p><button onclick="hqStartFace()">🎭 先生成脸部参考</button><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===91){
    const h=ST.hq;
    $('#app').innerHTML=`<h2>高质量图片 · 确认脸部</h2>${tabHtml(3)}
      <div class="box">这张脸会作为后续图片的参考。满意就锁定,不满意可以重新生成。</div>
      ${h.faceUrl?`<img class="out" src="${h.faceUrl}?t=${Date.now()}" style="max-width:520px;width:100%;display:block;margin:auto">`:'<div class="box"><span class="spin"></span> 正在生成脸部参考…</div>'}
      ${h.faceError?`<div class="box" style="color:#d03050">生成失败:${esc(h.faceError)}</div>`:''}
      <p><button onclick="hqStartFace()">🔁 重新生成脸</button>
      ${h.faceUrl?'<button onclick="hqAcceptFace()">🔒 锁定这张脸,设置身材</button>':''}
      <button class="back" onclick="goStep(90)">← 返回修改提示词</button></p>`;
    if(h.facePid&&!h.faceDone&&!h.facePolling) hqPollFace();
  }
  if(ST.step===92){
    const h=ST.hq;
    $('#app').innerHTML=`<h2>高质量图片 · 身材和服装</h2>${tabHtml(3)}
      <div class="row" style="align-items:flex-start"><img class="cardthumb" src="${h.faceUrl}" style="width:120px;height:120px;object-fit:cover"><div class="small">🔒 脸部已锁定<br>后面每次调整参数都会生成新文件,旧文件不会覆盖。</div></div>
      <label>服装和场景提示词(已预填旗袍)</label><textarea id="hqbpos" style="height:100px">${esc(h.bodyPrompt||'full-body adult East Asian woman with long loose black hair, wearing an elegant red silk qipao, graceful Chinese fairy-like beauty, luxury editorial fashion photography, ornate Chinese interior, neutral porcelain skin tone, cool soft studio lighting, balanced white balance, realistic skin texture')}</textarea>
      <div class="row" style="flex-wrap:wrap"><span><label>胸围比例</label><select id="hqbust" onchange="ST.hq.bust=this.value"><option value="natural">自然</option><option value="full">丰满</option><option value="very_full">夸张</option></select></span>
      <span><label>腰部比例</label><select id="hqwaist" onchange="ST.hq.waist=this.value"><option value="natural">自然</option><option value="slim">纤细</option><option value="very_slim">明显纤细</option></select></span>
      <span><label>臀胯比例</label><select id="hqhip" onchange="ST.hq.hip=this.value"><option value="natural">自然</option><option value="full">丰满</option><option value="very_full">夸张</option></select></span>
      <span><label>生成数量</label><select id="hqcount" onchange="ST.hq.count=+this.value">${[1,2,3,4].map(n=>`<option value="${n}">${n} 张</option>`).join('')}</select></span></div>
      <label>尺寸</label><select id="hqsize" onchange="ST.hq.size=this.value"><option value="832x1216">832×1216 竖图(推荐)</option><option value="1024x1536">1024×1536 高质量</option><option value="1024x1024">1024×1024 半身</option></select>
      <p><button onclick="hqStartVariants()">🚀 生成这一组并归档</button><button class="back" onclick="goStep(91)">← 重新选脸</button></p>`;
    if(h.bust) $('#hqbust').value=h.bust; if(h.waist) $('#hqwaist').value=h.waist; if(h.hip) $('#hqhip').value=h.hip;
    if(h.count) $('#hqcount').value=h.count; if(h.size) $('#hqsize').value=h.size;
  }
  if(ST.step===93){
    const h=ST.hq; const done=h.variants.filter(v=>v.done).length;
    const rows=h.variants.map((v,i)=>`<div class="card" style="cursor:default"><b>${esc(v.name)}</b> <span class="small">${v.done?'✅ 完成':v.error?'❌ '+esc(v.error):'⏳ 生成中…'}</span>${v.url?`<img class="out" src="${v.url}?t=${Date.now()}"><button class="back" onclick="hqChoose(${i})">${h.selected===i?'✅ 已选择':'选择这张'}</button><div class="small">已归档: output/high_quality/${esc(h.session)}/${esc(v.name)}.png</div>`:''}</div>`).join('');
    $('#app').innerHTML=`<h2>高质量图片 · 结果</h2>${tabHtml(3)}<div class="box">本组完成 ${done}/${h.variants.length} 张。每次调整参数重新生成,都会追加新文件。</div>${rows}<p><button onclick="goStep(92)">🔁 继续调整再生成</button><button class="back" onclick="home()">🏠 回首页</button></p>`;
    if(!h.variantPolling && done<h.variants.length) hqPollVariants();
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
    $('#app').innerHTML = `<h2>生成中… <button class="back" id="stopBtn" style="float:right;padding:5px 14px;font-size:13px" onclick="stopAll()">⏹ 停止</button></h2>
      <p class="row"><label style="margin:0"><input type="checkbox" id="mask" onchange="toggleMask(this.checked)"> 给图片打遮(模糊) — 默认不打遮</label></p>
      ${cards}<p id="again" style="display:none">
        <button onclick="editRound()">✏️ 编辑生图(回到设置接着改)</button>
        <button onclick="newRound()">➕ 继续生图(全新一轮)</button>
        <button class="back" onclick="home()">🏠 回首页</button></p>`;
    pollAll();
  }
  if(ST.step===10){
    const vm = await j('/api/vid/models');
    if(!ST.vid) ST.vid={unet:vm.unets[0].id,lora:'none',loras:[],pos:'',neg:'',aspect:'9:16',w:360,h:640,frames:49,fps:24,duration:2,stg:false,interpolate:false,style2d:false,textOnly:false,audioMode:'none',audioFile:null,file:null,imgName:null,pid:null,url:null};
    const v=ST.vid;
    if(!Array.isArray(v.loras)) v.loras=v.lora&&v.lora!=='none'?[v.lora]:[];
    if(!v.aspect){ const legacy={'360x640':'9:16','640x360':'16:9','512x512':'1:1','432x544':'4:5','480x640':'3:4','640x480':'4:3'}[v.w+'x'+v.h]; v.aspect=legacy||'custom'; v.customRes=!legacy; }
    if(v.duration==null) v.duration=Math.max(1,Math.round(v.frames/(v.fps||24)));
    if(!v.audioMode) v.audioMode='none';
    const unetCards=vm.unets.map((u,ui)=>`<div class="card drop-in ${v.unet===u.id?'sel':''}" data-vmodel="${u.id}" style="animation-delay:${ui*90}ms;${u.ready===false?'opacity:.62':''}" onclick="vPickModel('${u.id}')"><b>${u.name}</b> <span class="mfield">${u.tag}</span>${u.ready===false?'<span class="small" style="color:#e0556b"> · 正在安装</span>':''}<div class="small">${u.desc}</div></div>`).join('');
    const loraOpts=vm.loras.filter(l=>l.id!=='none').map(l=>`<label style="display:block;margin:6px 0"><input type="checkbox" name="vlora" value="${l.id}" ${v.loras.includes(l.id)?'checked':''} onchange="vToggleLora('${l.id}',this.checked)"> <b>${l.name}</b> <span class="small">${l.desc}</span></label>`).join('');
    const imgHtml = v.file
      ? `<b style="color:#18a058">✔ ${esc(v.file.name)}</b><br><img src="${URL.createObjectURL(v.file)}" alt="">`
      : `把源图拖到这里,或点击选择<br><span class="small">视频会让这张图动起来</span>`;
    $('#app').innerHTML = `<h2>视频生成</h2><div class="vid-workspace"><section class="vid-main">
      <p class="small">选择一个视频模型，上传源图，再用动作提示词告诉模型画面怎么动。</p>
      <label class="vt-toggle"><input type="checkbox" ${v.style2d?'checked':''} onchange="vSetStyle2d(this.checked)"> 2D 动漫模式（自动加动漫 LoRA 和 2D 提示词）</label>
      ${unetCards}
      <div class="small">2D 模式默认建议使用动漫图片垫图；勾选下面的“纯文字”后可不上传图片，但画风稳定性会差一些。</div>
      <label class="vt-toggle"><input type="checkbox" ${v.textOnly?'checked':''} onchange="vSetTextOnly(this.checked)"> 纯文字生成（不使用垫图）</label>
      <label>源图</label>
      <div class="drop" style="${v.textOnly?'opacity:.45;pointer-events:none':''}" ondragover="event.preventDefault();this.classList.add('over')"
           ondragleave="this.classList.remove('over')" ondrop="vDrop(event)"
           onclick="$('#vfile').click()">${imgHtml}</div>
      <input type="file" id="vfile" accept="image/*" style="display:none" onchange="vSetFile(this.files[0])">
      <label>动作提示词（告诉模型画面怎么动）</label>
      <textarea id="vpos" placeholder="例如：她慢慢转身看向镜头，头发和衣角随风摆动，轻轻呼吸">${esc(v.pos)}</textarea>
      <div class="small">常用写法：<b>slowly turns around</b>（慢慢转身）、<b>walks forward</b>（向前走）、<b>hair and clothes move naturally</b>（头发和衣服自然摆动）。想要 LTX 生成现场声，可以加 <b>footsteps, breathing, clothes friction</b>（脚步、喘息、衣物摩擦）。</div>
      <label>效果 LoRA（可多选，按勾选顺序叠加）</label>${loraOpts}<div class="small">动漫风=尽量保留二次元画风；动作增强=动作更明显；兽人向=兽人/特殊题材。它们不是同一种效果，叠得越多也越容易互相干扰。空间放大器、VAE、文本编码器都是系统配件，不在这里勾选。</div>
      <div class="row" style="align-items:flex-end;flex-wrap:wrap">
        <span><label>画面比例</label><select id="vres" onchange="vRes(this.value)">
          ${[['360x640','9:16 竖屏(推荐)'],['640x360','16:9 横屏'],['512x512','1:1 方形'],['432x544','4:5 竖向'],['480x640','3:4 竖屏'],['640x480','4:3 横屏'],['custom','自定义尺寸']].map(s=>`<option value="${s[0]}" ${(s[0]==='custom' ? (v.customRes?'selected':'') : (v.w+'x'+v.h===s[0]&&!v.customRes?'selected':''))}>${s[1]}${s[0]==='custom'?'':' · '+s[0].replace('x','×')}</option>`).join('')}</select><span id="vAspectNote" class="small">当前 ${v.aspect||((v.w/v.h)<1?'竖屏':'横屏')} · ${v.w}×${v.h}</span></span>
        <span id="vcustom" style="display:${v.customRes?'inline-flex':'none'};gap:6px;align-items:flex-end"><span><label>宽</label><input id="vwidth" type="number" min="64" step="16" value="${v.w}" onchange="vCustomSize()"></span><span><label>高</label><input id="vheight" type="number" min="64" step="16" value="${v.h}" onchange="vCustomSize()"></span></span>
        <span><label>视频时长</label><select onchange="vDuration(this.value)">${[1,2,3,4,5,6,8,10,15,20,30,60].map(n=>`<option value="${n}" ${+v.duration===n?'selected':''}>${n} 秒</option>`).join('')}</select></span>
        <span><label>帧率</label><select onchange="vFps(this.value)">${[[24,'24 fps'],[29.97,'30 fps (29.97)'],[59.94,'59 fps (59.94)'],[90,'90 fps']].map(s=>`<option value="${s[0]}" ${+v.fps===s[0]?'selected':''}>${s[1]}</option>`).join('')}</select></span>
        <span><label>总帧数 <span id="vfnum">${v.frames}</span></label><input type="range" min="9" max="241" step="8" value="${v.frames}" oninput="vFrames(this.value)"></span>
      </div>
      <p class="small">源图不会自动决定视频横竖；当前输出 <b>${v.aspect||'自定义'} · ${v.w}×${v.h}</b> · ${v.duration}秒 · ${v.frames}帧 · ${v.fps}fps。超过6秒会自动均分成小段（例如8秒=4+4），后台算完再拼成一个成片；超过24fps时先按24fps生成，再自动补帧。</p>
      <p class="row" style="flex-wrap:wrap"><label style="margin:0"><input type="checkbox" ${v.interpolate?'checked':''} onchange="vSet('interpolate',this.checked)"> 生成后补帧到 60fps</label>
        <label style="margin:0"><input type="checkbox" ${v.stg?'checked':''} onchange="vSet('stg',this.checked)"> STG 增强(动作更有力,稍慢)</label></p>
      <div class="box" style="margin-top:8px"><b>声音</b><br><label style="margin:4px 0"><input type="radio" name="vaudio" value="none" ${v.audioMode==='none'?'checked':''} onchange="vAudioMode(this.value)"> 不添加声音</label>
        <label style="margin:4px 0"><input type="radio" name="vaudio" value="native" ${v.audioMode==='native'?'checked':''} onchange="vAudioMode(this.value)"> LTX 原生声音（按提示词同时生成）</label>
        <label style="margin:4px 0"><input type="radio" name="vaudio" value="upload" ${v.audioMode==='upload'?'checked':''} onchange="vAudioMode(this.value)"> 添加音频(上传环境声/配音)</label>
        <input id="vaudiofile" type="file" accept="audio/*,video/*,.mp3,.wav,.m4a,.aac" style="display:${v.audioMode==='upload'?'block':'none'};margin-top:6px" onchange="vSetAudio(this.files[0])">
        <span id="vaudioname" class="small">${v.audioMode==='native'?'LTX 会按动作和场景同时生成现场声音。':(v.audioFile?'已选择: '+esc(v.audioFile.name):'上传音频会在成片后混入。')}</span></div>
      <p><button onclick="vStart()">🚀 开始生成</button>
      <button class="back" onclick="home()">← 返回</button></p>
      <p class="small">提示: 新放进 models/video/ 的模型要重启视频服务才会被识别。</p></section>
      <aside class="vid-side"><div class="vid-side-head"><b>后台任务与历史成片</b><button class="back" onclick="vidTasksPanel()">刷新</button></div>
      <div class="small">这里仅查看任务和硬盘里的成片，不会加载模型，也不会额外占模型内存。</div><div id="vidTasksPanel"><div class="vid-empty">正在读取…</div></div></aside></div>`;
    vidTasksPanel();
  }
  if(ST.step===11){
    $('#app').innerHTML = `<h2>生成视频中…</h2>
      <div class="box">🎬 正在生成… <span id="vel">0s</span><div id="vstep" class="small"></div></div>
      <div class="bar"><i id="vbar"></i></div>
      <div id="vout" style="margin-top:14px"></div>
      <p><button class="back" onclick="vCancel()">⏹ 停止当前视频</button></p>
      <p id="vacts" style="display:none">
        <button onclick="ST.step=10;render()">🔁 再做一段</button>
        <button class="back" onclick="home()">🏠 回首页</button></p>`;
    vPoll();
  }
  if(ST.step===12){
    const v=ST.vid||{};
    $('#app').innerHTML=`<div class="vid-detail"><h2>历史成片</h2>
      <div class="box">这是硬盘里已经生成好的视频。播放它不会启动 H3 或普通视频模型，也不会占用模型内存。</div>
      ${v.url?`<div class="small" style="margin-bottom:8px">${esc(v.taskName||'视频成片')}</div><video controls playsinline src="${esc(v.url)}"></video>`:'<div class="box">找不到这个视频文件。</div>'}
      <p class="row"><button onclick="goStep(10)">← 返回视频生成</button><button class="back" onclick="home()">回首页</button></p></div>`;
  }
  if(ST.step===20){
    const ms = await j('/api/llm/models');
    ST._llmModels = ms;
    const metalBox = '';
    $('#app').innerHTML = `<h2>选语言模型</h2>${metalBox}` + ms.map((m,mi)=>`
      <div class="card drop-in ${m.exists?'':'dis'}" style="animation-delay:${mi*90}ms" onclick="${m.exists?`llmPick('${m.id}')`:''}">
        <b>${esc(m.name)}</b> <span class="mfield">${esc(m.tag||'')}</span> <span class="small">${m.backend==='metal'?'官方 Metal':(m.backend==='vmlx'?'vMLX / MLX Metal':'llama.cpp')}</span>
        <div class="small">${esc(m.desc||'')}</div>
        ${m.author_usage?`<div class="small" style="margin-top:5px;color:var(--muted)">作者建议：${esc(m.author_usage)}</div>`:''}
        <div><span class="mmem">💾 需 GPU 上限 ${m.gpu_mb}MB</span></div>
        ${m.exists?'':`<div class="small" style="color:#d03050">${m.backend==='metal'?'未配置: '+esc(m.missing_reason||'请初始化 Metal'):m.backend==='vmlx'?'未配置: '+esc(m.missing_reason||'请检查 vMLX 环境'):'模型文件缺失(检查 models/llm 软链)'}</div>`}
      </div>`).join('') + `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===21){
    const m=ST.llmModel; const isMetal=m.backend==='metal'; const isVmlx=m.backend==='vmlx'; const p=m.prefs||{thinking:true,temp:0.7,max_tokens:8192,mtp:false,codex_proxy:false,proxy_mode:0,ctx:32768,parallel:1,budget:-1,reasoning_level:'high'};
    const proxyMode=Number.isInteger(+p.proxy_mode)?Math.max(0,Math.min(2,+p.proxy_mode)):(p.codex_proxy?1:0);
    const _b=p.budget??-1, _c=+p.ctx||32768, _ctxPresets=[16384,32768,65536], _customCtx=!_ctxPresets.includes(_c);
    const thinkHtml = m.is_reasoning
      ? `<div class="box"><b>🧠 思考设置</b>
        <div class="row" style="margin:8px 0;align-items:center;gap:8px;flex-wrap:wrap">
          <span id="lthinkFast" class="small" style="color:${p.thinking?'var(--muted)':'var(--text)'};font-weight:${p.thinking?'500':'700'}">极速模式</span>
          <label style="position:relative;display:inline-block;width:38px;height:22px;cursor:pointer;vertical-align:middle">
            <input type="checkbox" id="lthink" ${p.thinking?'checked':''} style="position:absolute;opacity:0;width:0;height:0" onchange="const on=this.checked; const ex=document.getElementById('lthinkExtra'); const mode=document.getElementById('lthinkMode'); const fast=document.getElementById('lthinkFast'); const deep=document.getElementById('lthinkDeep'); const track=document.getElementById('lthinkTrack'); const knob=document.getElementById('lthinkKnob'); if(ex) ex.style.display=on?'block':'none'; if(mode) mode.textContent=on?'深度思考模式':'极速模式'; if(fast){ fast.style.color=on?'var(--muted)':'var(--text)'; fast.style.fontWeight=on?'500':'700'; } if(deep){ deep.style.color=on?'var(--accent)':'var(--muted)'; deep.style.fontWeight=on?'700':'500'; } if(track) track.style.background=on?'var(--accent)':'var(--border)'; if(knob) knob.style.left=on?'19px':'3px';">
            <span id="lthinkTrack" style="position:absolute;inset:0;background:${p.thinking?'var(--accent)':'var(--border)'};border-radius:999px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.04);transition:.2s"></span>
            <span id="lthinkKnob" style="position:absolute;top:3px;left:${p.thinking?'19px':'3px'};width:16px;height:16px;background:var(--card);border:1px solid rgba(0,0,0,.05);border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.14);transition:.2s"></span>
          </label>
          <span id="lthinkDeep" class="small" style="color:${p.thinking?'var(--accent)':'var(--muted)'};font-weight:${p.thinking?'700':'500'}">深度思考模式</span>
          <span class="small">当前: <b id="lthinkMode">${p.thinking?'深度思考模式':'极速模式'}</b></span>
        </div>
        <div class="small" style="margin-top:2px">💡 关闭 = 极速模式; 开启 = 深度思考模式。极速更快更稳,深度思考更慢但适合难题。</div>
        <div id="lthinkExtra" style="display:${p.thinking?'block':'none'};margin-top:8px">
          <label style="display:block;margin-top:2px">思考档位(想得多深)
            <select id="llevel">
              <option value="high" ${(p.reasoning_level||'high')==='high'?'selected':''}>high · 高(深度思考,最慢)</option>
              <option value="medium" ${(p.reasoning_level||'high')==='medium'?'selected':''}>medium · 中(推荐)</option>
              <option value="low" ${(p.reasoning_level||'high')==='low'?'selected':''}>low · 低(快,思考浅)</option>
            </select></label>
          <label style="display:block;margin-top:6px">思考长度(回答前最多想多少 token)
            <select id="lbud">
              <option value="-1" ${_b===-1?'selected':''}>-1 · 不限(随便想,默认)</option>
              <option value="2048" ${_b===2048?'selected':''}>2048 · 简短思考(快,推荐)</option>
              <option value="8192" ${_b===8192?'selected':''}>8192 · 深度思考(慢)</option>
            </select>
            <label style="margin-left:10px"><input type="checkbox" id="lbudc" onchange="document.getElementById('lbudget').style.display=this.checked?'inline-block':'none'"> 自定义</label>
            <input type="number" id="lbudget" min="-1" max="32768" step="256" placeholder="0=不思考 · 正数=限N个token" style="display:none;width:200px"></label>
          <div class="small" style="margin-top:2px">💡 -1=不限随便想 · 0=不思考直接答 · 正数=最多想N个token(如 1024/2048/4096)</div>
        </div>
      </div>`
      : `<div class="box"><b>🧠 思考设置</b><p class="small" style="margin:4px 0">此模型为快速应答型,无思考模式。</p></div>`;
    const mtpLabel = m.mtp_warning ? 'MTP（实验性，不建议开启）' : 'MTP 加速（每步尝试预判多个 token）';
    const mtpHtml = !isMetal && m.has_mtp
      ? `<div style="margin:6px 0"><label style="margin:0"><input type="checkbox" id="lmtp" ${p.mtp?'checked':''}> ${mtpLabel}</label>${m.mtp_warning?`<div class="small" style="color:#b54708;margin-top:5px">⚠ ${esc(m.mtp_warning)}</div>`:''}</div>`
      : '';
    const mtpNoteHtml = m.mtp_compatible===false ? `<div class="small" style="color:#a15c00;margin:5px 0">MTP 暂不可用：${esc(m.mtp_note||'侧车与主模型词表不兼容')}。当前只启动主模型，避免加载失败。</div>` : '';
    const cpxHtml = m.supports_codex_proxy ? `<div class="box" style="margin:4px 0"><label style="margin:0">代理模式</label><div class="proxy-switch" role="group" aria-label="代理模式"><button type="button" data-proxy-mode="0" class="${proxyMode===0?'active':''}" onclick="selectProxyMode(0)">关闭</button><button type="button" data-proxy-mode="1" class="${proxyMode===1?'active':''}" onclick="selectProxyMode(1)">Codex</button><button type="button" data-proxy-mode="2" class="${proxyMode===2?'active':''}" onclick="selectProxyMode(2)">Claude</button></div><input type="hidden" id="lcpx" value="${proxyMode}"><div id="lcpxMode" class="small">当前：${['关闭','Codex 代理','Claude 代理'][proxyMode]}</div><div class="small" style="margin-top:5px">${isVmlx?'普通 OpenAI 请求可直连 vMLX；选择代理后为工具请求增加格式适配，后端转到内部端口。':'Codex 使用 Responses API，Claude 使用 Anthropic Messages API，代理会把对应请求转换为 llama.cpp 接口。'} 公共地址固定使用 8848，同一时间只能运行一种代理。</div></div>` : '';
    const metalHint = isMetal ? `<div class="box"><b>官方 Metal 运行提示</b><div class="small" style="margin-top:5px">此模型走官方 GPT-OSS Responses API 和 Apple Metal 参考实现，不使用 llama.cpp/MTP/Codex 代理。需要先完成 Metal 初始化，并准备转换后的 model.bin。</div></div>` : '';
    const vmlxHint = isVmlx ? `<div class="box"><b>vMLX / MLX Metal</b><div class="small" style="margin-top:5px">此模型使用项目内隔离的 .venv-vmlx，实体读取 models/llm 下的 MXFP4 权重；与 llama.cpp/其他模型互斥运行。作者推荐 temperature 1.0、top_p 0.95、top_k 20；本机实测为稳定性优先，默认温度 0.4 且关闭原生 MTP。当前版本开启 MTP 后容易重复循环，稳定使用请保持关闭。</div></div>` : '';
    $('#app').innerHTML = `<h2>${esc(m.name)} · 启动参数</h2>
      ${metalHint}
      ${vmlxHint}
      ${m.author_source?`<div class="small" style="margin:5px 0;color:var(--muted)">作者资料：${esc(m.author_source)}</div>`:''}
      ${thinkHtml}
      <div class="box"><b>📏 上下文与输出</b>
        <label style="display:block;margin-top:2px">上下文大小(对话总容量: 历史+思考+回答)
          <select id="lctx" onchange="document.getElementById('lctxcust').style.display=this.value==='custom'?'inline-block':'none'">
            <option value="16384" ${_c===16384?'selected':''}>16384 · 省内存(普通聊天)</option>
            <option value="32768" ${_c===32768?'selected':''}>32768 · 推荐(32GB 机器)</option>
            <option value="65536" ${_c===65536?'selected':''}>65536 · 长上下文(更吃内存)</option>
            <option value="custom" ${_customCtx?'selected':''}>自定义</option>
          </select>
          <input type="number" id="lctxcust" min="8192" max="131072" step="4096" value="${_customCtx?_c:''}" placeholder="自定义 8192~131072" style="display:${_customCtx?'inline-block':'none'};width:200px" oninput="if(document.getElementById('lctx').value!=='custom') document.getElementById('lctx').value='custom'"></label>
        <div class="small" style="margin-top:2px">💡 越大越能记住长会话,越吃内存;32GB 机器建议 16K~32K</div>
        <label style="display:block;margin-top:8px">并行会话数
          <select id="lparallel">
            <option value="1" ${(+p.parallel||1)===1?'selected':''}>1 · 最省内存（推荐）</option>
            <option value="2" ${(+p.parallel||1)===2?'selected':''}>2 · 两个请求并行</option>
            <option value="4" ${(+p.parallel||1)===4?'selected':''}>4 · 多请求并行</option>
            <option value="custom" ${![1,2,4].includes(+p.parallel)?'selected':''}>自定义</option>
          </select>
          <input type="number" id="lparallelcust" min="1" max="8" step="1" value="${![1,2,4].includes(+p.parallel)?Math.max(1,Math.min(8,+p.parallel||1)):''}" placeholder="1~8" style="display:${![1,2,4].includes(+p.parallel)?'inline-block':'none'};width:90px" oninput="if(document.getElementById('lparallel').value!=='custom') document.getElementById('lparallel').value='custom'">
        </label>
        <div class="small" style="margin-top:2px">并行越高越容易占满统一内存；32GB 机器建议 1。</div>
        <label style="display:block;margin-top:8px">温度(越低越严谨) <span id="ltv">${p.temp}</span>
          <input type="range" id="ltemp" min="0" max="1.5" step="0.1" value="${p.temp}" oninput="$('#ltv').textContent=this.value"></label>
        <div class="small" style="margin-top:2px">💡 此模型建议温度 <b>${esc(m.recommended_temp||'0.4~0.7')}</b>；越低越稳定严谨，过高更容易重复或跑偏。</div>
        <label style="display:block;margin-top:8px">最大回复长度(max_tokens,单次回复上限)
          <input type="number" id="lmax" min="512" max="65536" step="512" value="${p.max_tokens}" style="width:150px"></label>
      </div>
      <div class="box"><b>⚡ 加速与兼容</b>
        ${mtpHtml}
        ${mtpNoteHtml}
        ${cpxHtml}
      </div>
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
        <div class="card" style="cursor:default"><b>💬 ${esc(c.model.name)}</b> <span class="dot off">◐ 加载中</span>
          <div class="small" style="margin-top:5px">模型进内存+预热约 1~2 分钟,本页每 4 秒自动刷新;就绪后这里会出现「💬 打开聊天网页」按钮。</div>
          <div class="bar" style="margin-top:10px"><i class="indet"></i></div></div>
        <div class="box"><span class="small">💡 直接开 8848 看到的「The model is loading」静态文字页是 llama.cpp 二进制自带的,改不了样式——在这里等,有动画有提示。</span></div>`;
      setTimeout(()=>{ if(ST.step===22) render(); },4000);
      return;
    }
    const port=c.port||8848;
    const backendLabel=c.model.backend==='metal'?'官方 Metal':(c.model.backend==='vmlx'?'vMLX / MLX Metal':'llama.cpp');
    $('#app').innerHTML = `<h2>语言模型运行中</h2>
      <div class="card" style="cursor:default"><b>💬 ${esc(c.model.name)}</b> <span class="dot on">● 运行中</span> <span class="small">${backendLabel}</span>
        <div class="small">温度 ${c.model.temp} · max_tokens ${c.model.max_tokens}${c.model.ctx?' · 上下文'+Math.round(c.model.ctx/1024)+'K':''}${c.model.thinking?' · 思考模式':''}${c.model.thinking&&c.model.budget>=0?' · 思考限'+c.model.budget+'t':''}${c.model.mtp?' · MTP加速':''}${c.model.proxy_mode===2?' · Claude代理':(c.model.proxy_mode===1?' · Codex代理':'')}</div></div>
      <div class="box">🔌 API 地址:<code>${esc(c.api_url)}</code>
        <button class="back" style="margin-left:8px" onclick="llmCopyApi('${c.api_url}')">复制</button></div>
      <div class="row" style="gap:26px;margin:14px 0">
        <span>⏱ 已启用 <b id="llmEl">${fmtTime(c.elapsed_sec)}</b></span>
        <span>🧮 token: 输入 <b id="llmPT">${c.prompt_tokens}</b> · 输出 <b id="llmGT">${c.gen_tokens}</b></span></div>
      <div class="row" style="margin:0 0 6px"><span>⚡ <b id="llmAct">${fmtAct(c.activity)}</b></span></div>
      <p style="text-align:center;margin:20px 0"><button onclick="llmPause()" style="padding:13px 40px;font-size:17px">⏸ 暂停</button></p>
      ${c.model.backend==='vmlx'
        ? `<div class="box"><b>vMLX / MLX Metal 接口说明</b><div class="small" style="margin-top:5px">此模型由 vMLX 提供原生 OpenAI 兼容 API：<code>${esc(c.api_url)}</code>。默认直连；若启动页开启 vMLX 工具兼容代理，则由代理适配 Codex/MCP 请求。它不接入 ImageVideoStudio 的控制台聊天或“说画图”工作流；直接访问 8848 根地址或旧客户端的 <code>/props</code> 路径会出现 404，不代表模型故障。</div></div>`
        : `<p class="row"><button onclick="window.open('http://127.0.0.1:${port}')">💬 打开聊天网页</button><button onclick="goStep(23)">🎨 控制台聊天(说「画xx」自动生图)</button></p>`}
      <p class="row">
        <button class="back" onclick="llmStop(0)">🔄 更换模型</button>
        <button class="back" onclick="llmStop(20)">🔄 更换语言模型</button>
        <button class="back" onclick="llmStop(0)">⏻ 关闭模型</button></p>`;
    llmTick();
  }
  if(ST.step===23){ // 控制台聊天: 打字走语言模型;说「画xx」自动暂停LLM去生图,生完一键恢复
    if(!ST.chat) ST.chat={msgs:[],busy:false,genModel:'',size:'1024x576',llmPaused:false};
    const C=ST.chat;
    const c=await j('/api/llm/stats');
    const ms=await j('/api/models');
    if(!C.genModel&&ms.length) C.genModel=ms[0].id;
    if(!c.model){ $('#app').innerHTML=`<div class="box">语言模型没在跑,先去启动一个。</div><p><button onclick="goStep(20)">去选语言模型</button><button class="back" onclick="home()">🏠 回首页</button></p>`; return; }
    const banner = c.paused
      ? `<div class="box">⏸ 语言模型已暂停(内存让给了生图)。现在不能聊天,但说「画xx」仍能直接生图。</div>`
      : (c.loading ? `<div class="box"><span class="spin"></span> 语言模型加载/恢复中,约 1~2 分钟…</div>` : '');
    const rows=C.msgs.map(m=>{
      if(m.role==='user') return `<div class="chatrow me"><div class="bubble">${esc(m.text)}</div></div>`;
      return `<div class="chatrow"><div class="bubble ai">${m.think?`<details class="think"><summary>💭 思考过程(点开看)</summary><div class="small" style="white-space:pre-wrap;margin-top:5px">${esc(m.think)}</div></details>`:''}${esc(m.text)}${m.img?`<img class="out" src="${m.img}">`:''}${m.html||''}</div></div>`;
    }).join('');
    $('#app').innerHTML=`<h2>🎨 控制台聊天 · ${esc(c.model.name)}</h2>
      ${banner}
      <div class="row" style="margin:2px 0 10px;gap:8px">
        <span class="small">生图模型</span>
        <select onchange="ST.chat.genModel=this.value">${ms.map(m=>`<option value="${m.id}" ${C.genModel===m.id?'selected':''}>${esc(m.name)}</option>`).join('')}</select>
        <span class="small">尺寸</span>
        <select onchange="ST.chat.size=this.value">${[['1024x576','横 1024×576'],['832x1216','竖 832×1216'],['1024x1024','方 1024×1024']].map(s=>`<option value="${s[0]}" ${C.size===s[0]?'selected':''}>${s[1]}</option>`).join('')}</select>
        <span class="small">💡 消息里写「画…」就会用上面的模型自动出图贴进对话</span></div>
      <div id="chatbox">${rows||'<div class="box">开聊吧。试试说「画一只在雨里跑的猫」。</div>'}</div>
      <div class="chatinput">
        <textarea id="chatin" placeholder="打字聊天;说「画xx」自动生图(Enter 发送,Shift+Enter 换行)" ${C.busy?'disabled':''}
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();chatSend()}"></textarea>
        <button onclick="chatSend()" ${C.busy?'disabled':''}>${C.busy?'…':'发送'}</button></div>
      <p><button class="back" onclick="goStep(22)">← 回语言模型页</button><button class="back" onclick="home()">🏠 回首页</button></p>`;
    const cb=$('#chatbox'); if(cb) cb.scrollTop=cb.scrollHeight;
    if(!C.busy){ const t=$('#chatin'); if(t) t.focus(); }
  }
  if(ST.step===60){ // 语音模型: 学声音(传音频/视频+原话) → 声音库 → 输台词生成配音
    const ts = await j('/api/tts/state');
    const vs = (await j('/api/tts/voices')).voices || [];
    if(!ts.running){
      $('#app').innerHTML = `<h2>🎙 语音模型</h2>
        <div class="box">语音服务还没启动。它很小(0.6B,跑 CPU),<b>不会</b>停掉生图/视频/语言,可放心共存。</div>
        ${ts.error?`<div class="box" style="color:#d03050">上次加载失败:${esc(ts.error)}</div>`:''}
        <p><button onclick="ttsStart()">🚀 启动语音服务</button>
        <button class="back" onclick="home()">← 返回</button></p>`;
      return;
    }
    if(!ts.ready){
      $('#app').innerHTML = `<h2>🎙 语音模型加载中…</h2>
        <div class="box"><span class="spin"></span> Audio8 正在加载,约 10~20 秒…</div>`;
      setTimeout(()=>{ if(ST.step===60) render(); },3000);
      return;
    }
    const voiceOpts = [`<label style="display:block;margin:6px 0"><input type="radio" name="ttsv" value="" checked> <b>🎲 随机音色</b> <span class="small">每次随机一个声音</span></label>`]
      .concat(vs.map(v=>`<label style="display:block;margin:6px 0"><input type="radio" name="ttsv" value="${esc(v.name)}"> <b>🎤 ${esc(v.name)}</b> <span class="small">${esc(v.ref_text)}</span></label>`)).join('');
    const voiceList = vs.length ? vs.map(v=>{
      const en = encodeURIComponent(v.name);
      return `<div class="card" style="cursor:default"><b>🎤 ${esc(v.name)}</b>
        <button class="back" style="float:right;padding:4px 10px" onclick="ttsRename('${en}')">✏️ 重命名</button>
        <div class="small">参考文本:${esc(v.ref_text)}</div>
        ${v.has_sample
          ? `<p style="margin:8px 0 0"><audio controls preload="none" src="/api/tts/vsample/${en}?t=${Date.now()}" style="width:100%"></audio></p>`
          : `<p style="margin:8px 0 0" class="small">还没有试听</p>`}
        <div class="row" style="margin:8px 0 0;gap:6px;align-items:center">
          <input id="smptext-${en}" type="text" placeholder="试听想说的话(留空=默认那句)"
            style="flex:1;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px">
          <button class="back" onclick="ttsSample('${en}',this)">🔊 ${v.has_sample?'重生成':'生成'}试听</button></div>
        <p style="margin:6px 0 0"><button class="back" onclick="ttsDel('${en}')">🗑 删除</button></p></div>`;
    }).join('')
      : `<div class="box">还没有声音。先在下面传一段录音/视频学一个。</div>`;
    $('#app').innerHTML = `<h2>🎙 语音模型 <span class="dot on">● 就绪</span></h2>
      <h3>① 学一个声音</h3>
      <div class="box">传一段<b>能听清人声</b>的录音或视频(mp4/mp3/wav/m4a),再填<b>这段里说的原话</b>(必须一字不差,否则学不像)。</div>
      <div class="drop" onclick="$('#ttsfile').click()">点击选择音频/视频文件<br><span class="small" id="ttsfn">未选择</span></div>
      <input type="file" id="ttsfile" accept="audio/*,video/*,.mp4,.m4a,.mp3,.wav" style="display:none" onchange="$('#ttsfn').textContent=this.files[0]?this.files[0].name:'未选择'">
      <label>这段音频里说的原话(参考文本)</label>
      <textarea id="ttstext" placeholder="一字不差地填这段音频里说的话"></textarea>
      <label>给这个声音起名</label>
      <input id="ttsname" type="text" placeholder="如: 我的声音" style="width:100%;padding:10px;border-radius:8px">
      <p><button onclick="ttsLearn()">📚 学习并存入声音库</button></p>
      <h3>② 声音库(${vs.length})</h3>${voiceList}
      <h3>③ 输台词 → 生成配音</h3>
      <label>选音色</label>${voiceOpts}
      <label>台词</label>
      <div class="box" style="font-size:12px;line-height:1.7">
        <b>🎭 台词符号口诀</b>(符号只管"停顿/长短/收尾",改不了字的声调)<br>
        · <b>,</b> 小停顿(换口气)　<b>……</b> 断断续续/发颤(喘气、犹豫)　<b>——</b> 拖长音(贴在字后面)<br>
        · <b>?</b> 升调=疑问/惊讶(二声)　<b>!</b> 重音=大叫/强调(四声)<br>
        · <b>声调靠选字</b>:啊?=á二声惊讶;啊!=à四声大叫;哦?=质疑;嗯……=抿嘴害羞;咦=二声/意=四声<br>
        · <b>喘气写法</b>:呼……呼……=小喘　哈……哈……=大喘　呼呼呼呼=快喘　呼——=喘不上气<br>
        · <b>完整实例(跑步遇老鼠)</b>:呼……呼……呼——哈……哈……啊?!老鼠!呼呼呼呼——啊——!哈——哈……呼……嗯……<br>
        · 参考音越生动,克隆越有感情;平读参考音情绪有限
      </div>
      <textarea id="ttssay" placeholder="想让它说的话"></textarea>
      <p><button onclick="ttsSpeak()">🔊 生成配音</button>
      <button onclick="goStep(61)">🎬 分段配音工作台</button>
      <button class="back" onclick="ttsStop()">⏻ 关闭语音服务</button>
      <button class="back" onclick="home()">🏠 回首页</button></p>
      <div id="ttsresult"></div>`;
  }
  if(ST.step===61){ // 分段配音工作台: 逐段后台合成→好一段听一段→拖拽/上下排序→改台词/局部重录→合并锁定
    const ts = await j('/api/tts/state');
    if(!ts.running || !ts.ready){
      $('#app').innerHTML = `<h2>🎬 分段配音工作台</h2>
        <div class="box">语音服务${ts.running?'还在加载中':'还没启动'}。先回语音页启动它(0.6B 跑 CPU,不影响别的)。</div>
        <p><button onclick="goStep(60)">← 回语音页</button> <button class="back" onclick="home()">🏠 回首页</button></p>`;
      if(ts.running && !ts.ready) setTimeout(()=>{ if(ST.step===61) render(); },3000);
      return;
    }
    const wb = await j('/api/wb/state');
    const vs = (await j('/api/tts/voices')).voices || [];
    const vopts = (sel)=>`<option value="" ${!sel?'selected':''}>🎲 随机音色</option>`
      + vs.map(v=>`<option value="${esc(v.name)}" ${v.name===sel?'selected':''}>🎤 ${esc(v.name)}</option>`).join('');
    const segCard = (s, ro)=>{
      let body;
      if(s.status==='run'){
        const cp = s.frame_max? (s.frame/s.frame_max):0;
        const pct = Math.min(99, Math.round(((s.chunks_done+cp)/(s.chunk_count||1))*100));
        body = `<div class="bar"><i style="width:${pct}%"></i></div>
          <div class="small">🔊 合成中 第${s.chunks_done+1}/${s.chunk_count}小段 · ${s.frame||0}帧</div>`;
      }else if(s.status==='wait'){ body = `<div class="small">⏳ 排队中…</div>`; }
      else if(s.status==='err'){ body = `<div class="small" style="color:#d03050">✗ 失败:${esc(s.err||'合成失败')}</div>`; }
      else { body = `<audio controls preload="none" src="/tts_out/${encodeURIComponent(s.wav)}" style="width:100%"></audio>`; }
      if(ro){
        return `<div class="card" style="cursor:default"><b>#${s.id}</b> <span class="small">${esc(s.voice||'随机音色')}</span>
          <div class="small" style="margin:4px 0">${esc(s.text)}</div>${body}</div>`;
      }
      return `<div class="card wbcard" data-sid="${s.id}" style="cursor:default"
          ondragover="event.preventDefault()" ondrop="wbDrop(event,${s.id})">
        <div class="row" style="gap:6px;align-items:center">
          <span draggable="true" ondragstart="wbDrag(event,${s.id})" style="cursor:grab;font-size:16px" title="按住拖动排序">⠿</span>
          <b>#${s.id}</b>
          <button class="back" style="padding:3px 8px" title="上移" onclick="wbMove(${s.id},-1)">↑</button>
          <button class="back" style="padding:3px 8px" title="下移" onclick="wbMove(${s.id},1)">↓</button>
          <select onchange="wbVoice(${s.id},this.value)" style="padding:5px 8px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${vopts(s.voice)}</select>
          <span style="flex:1"></span>
          <button class="back" style="padding:4px 9px" title="整段重录" onclick="wbRegen(${s.id})">🔁</button>
          <button class="back" style="padding:4px 9px" title="选中几个字→拆分重录" onclick="wbSplit(${s.id})">✂️</button>
          <button class="back" style="padding:4px 9px" title="删除这段" onclick="wbDel(${s.id})">🗑</button>
        </div>
        <textarea id="wbt-${s.id}" style="margin-top:6px">${esc(s.text)}</textarea>
        <div class="row" style="gap:6px;margin-top:4px;align-items:center">
          <button class="back" style="padding:5px 10px" onclick="wbSave(${s.id})">💾 保存台词</button>
          <span class="small">${s.text.length}字 · 改台词/换音色后会自动重录</span>
        </div>
        <div id="wbstatus-${s.id}">${body}</div>
      </div>`;
    };
    if(wb.merged){
      const url = '/tts_out/'+encodeURIComponent(wb.merged);
      $('#app').innerHTML = `<h2>🎬 分段配音工作台 <span class="dot on">● 已合并锁定</span></h2>
        <div class="card" style="cursor:default;border:2px solid var(--accent)">
          <b>🔒 最终配音(顺序已锁定,不能再改台词和音色)</b>
          <p><audio controls autoplay src="${url}" style="width:100%"></audio></p>
          <p><a href="${url}" download="${esc(wb.merged)}"><button>⬇ 下载最终 wav</button></a></p>
        </div>
        <h3>各段(只读)</h3>${wb.segs.map(s=>segCard(s,true)).join('')}
        <p><button class="back" onclick="wbReset()">🔓 重新开始(清空)</button>
        <button class="back" onclick="goStep(60)">← 回语音页</button></p>`;
      return;
    }
    const anyBusy = wb.segs.some(s=>s.status==='wait'||s.status==='run');
    const allDone = wb.segs.length>0 && wb.segs.every(s=>s.status==='done');
    $('#app').innerHTML = `<h2>🎬 分段配音工作台 <span class="dot on">● 就绪</span></h2>
      <div class="box">把长台词<b>拆成一段段</b>(每段约30秒内),后台逐段合成,<b>好一段就能听一段</b>;
      可拖动/上下<b>排序</b>、改<b>台词</b>、删了<b>重录</b>、选中几个字<b>局部重录</b>;满意后<b>合并</b>成一条长配音(合并后锁定不能改)。</div>
      <div class="card" style="cursor:default">
        <label>新的一段台词</label>
        <textarea id="wbnew" placeholder="这一段想说的话(约30秒内;太长会自动切块合成,不会断)"></textarea>
        <div class="row" style="gap:8px;margin-top:6px;align-items:center">
          <select id="wbnewv" style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${vopts('')}</select>
          <button onclick="wbAdd()">＋ 加进列表(后台合成)</button>
        </div>
      </div>
      <h3>段落(${wb.segs.length}) <span id="wbBusyLabel" class="small">${anyBusy?'· 后台合成中…':''}</span></h3>
      ${wb.segs.length ? wb.segs.map(s=>segCard(s,false)).join('') : '<div class="box">还没有段落,先在上面加一段。</div>'}
      <p><button id="wbMergeBtn" onclick="wbMerge()" ${allDone?'':'disabled'}>🔒 全部满意,合并成一条</button>
      <button class="back" onclick="wbReset()">🗑 清空重开</button>
      <button class="back" onclick="goStep(60)">← 回语音页</button></p>
      ${allDone?'':'<div class="small">全部段落生成完成后才能合并。</div>'}`;
    wbMaybePoll(anyBusy);
  }
  if(ST.step===30){ // 预定批量: 模型大卡→风格子卡→提示词组(每模型独立,每风格可垫一张图)
    const cfg = await j('/api/batch/config');
    ST.bmodels = cfg.models; // 可选模型清单;默认一张卡都不加,用户自己「＋ 新增模型」
    const mopts = cfg.models.map(m=>`<option value="${m.id}">${esc(m.name)}</option>`).join('');
    const body = `
      <div class="box">先在下方选模型点「＋ 新增模型」,把要用的加进来(<b>默认一个都不加</b>);每个模型卡里多个<b>风格子卡</b>;一个风格=一组提示词(一行一条)=多张图。
        逐张真等画完才跑下一张,归档到 <code>${esc(cfg.org_dir)}</code> 下的 <code>图片N.png</code><br>
        <span class="small">每个新增的模型自带一个空白「普通」风格;改这里只本次生效</span></div>
      <div class="box">📂 从文件夹导入风格(每个 .txt = 一个风格,文件内空行分隔多条提示词):
        <div class="row" style="margin-top:6px">
          <input id="bfolder" type="text" placeholder="风格文件夹路径,如项目下的 prompts_sfw"
            style="flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px">
          <button class="back" onclick="bImport()">导入到已新增的模型</button></div>
        <span class="small" id="bimpMsg"></span></div>
      <div class="row" style="align-items:center;gap:10px">
        <label style="margin:0">图片尺寸 <span class="small">(最小 640×960)</span></label>
        <select id="bsize" style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
          <option value="640x960">640×960(最小可用,快)</option>
          <option value="1024x720" selected>1024×720(默认)</option>
          <option value="832x1216">832×1216(竖图)</option>
        </select></div>
      <label>公共负向提示词</label><textarea id="bneg" style="height:44px">${esc(cfg.neg)}</textarea>
      <div id="bmodels"></div>
      <div class="row" style="margin-top:10px;gap:8px;align-items:center">
        <select id="bpick" style="flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${mopts}</select>
        <button class="back" onclick="bAddModel()">＋ 新增模型</button></div>
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
      const refGrid = cfg.refs.length ? cfg.refs.map(r=>`
        <div style="position:relative;display:inline-block;margin:4px;cursor:pointer;border:3px solid ${r.name===cfg.active?'var(--on)':'var(--border)'};border-radius:10px;overflow:hidden;line-height:0"
             onclick="refUse('${encodeURIComponent(r.name)}')" title="点选当参考图">
          <img src="${r.url}" style="width:84px;height:84px;object-fit:cover;display:block">
          <span style="position:absolute;top:2px;right:3px;width:18px;height:18px;line-height:18px;text-align:center;font-size:12px;color:#fff;background:#e0556b;border-radius:50%"
                onclick="event.stopPropagation();refDel('${encodeURIComponent(r.name)}')" title="删除">✕</span>
          ${r.name===cfg.active?'<span style="position:absolute;bottom:2px;left:3px;font-size:11px;color:#fff;background:var(--on);border-radius:6px;padding:0 5px;line-height:16px">参考中</span>':''}
        </div>`).join('')
        : '<div class="small">还没传参考图;上面选图上传(可多张),点图选用,不选就是纯文字生成</div>';
      body = `
      <div class="box">共 <b>${cfg.panels.length}</b> 格 · 逐格真等画完,完成即叠加台词归档到 <code>${esc(cfg.out_dir)}</code><br>
      <span class="small">主文件:${esc(cfg.story_file)} · 这里的修改只本次生效;选用参考图后每格按 0.75 以图生图,主角/画风就贴你那张图</span></div>
      <label>生图模型(默认动漫最强)</label><select id="cmodel" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${opts}</select>
      <label>画风(注入每格提示词;想蝙蝠侠那种黑暗风就选「黑暗蝙蝠侠风」)</label>
      <select id="cstyle" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
        ${(cfg.styles||[]).map(s=>`<option value="${s.key}">${esc(s.name)}</option>`).join('')}
      </select>
      <input id="ccustom" type="text" placeholder="自定义画风(英文,填了就覆盖上面预设;如: dark comic book style, gritty noir)"
        style="width:100%;margin-top:6px;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px;box-sizing:border-box">
      <label>参考图库(自己传图当主角/画风参考 · 点图选用 · ${cfg.active?'当前: '+esc(cfg.active):'当前未选用=纯文字'})</label>
      <div class="row"><input type="file" accept="image/*" multiple onchange="refUp(this)"><span class="small">可一次选多张</span>
        ${cfg.active?'<button class="back" style="padding:4px 10px" onclick="refClear()">✕ 不用参考图</button>':''}</div>
      <div>${refGrid}</div>
      <label>主角一致性(选了参考图才生效)</label>
      <select id="cmode" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
        <option value="i2i">垫图 · 贴整张画风+构图(强度0.75)</option>
        <option value="ipa">锁脸 · 只锁主角长相,构图/动作交给提示词(大头照参考最稳,仅单文件模型)</option>
      </select>
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
    let gstyles=[]; try{ gstyles=(await j('/api/comic/story')).styles||[]; }catch(e){}
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
            <option ${p.w===832&&p.h===1216?'selected':''}>832x1216</option>
            <option ${p.w===1024&&p.h===720?'selected':''}>1024x720</option>
            <option ${p.w===1024&&p.h===1024?'selected':''}>1024x1024</option></select></span>
          <span>风格 <input type="text" class="cpstyle" data-p="${i}" value="${esc(p.style||'')}" placeholder="留空=继承上一张" style="width:200px;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)"></span></div>
      </div>`;}).join('');
    $('#app').innerHTML = `<h2>自定义连载 · 2/3 分镜画布</h2>` + tabHtml(2) + ccTabHtml(1) + `
      <div class="box">要几格就加几格(比如连载 79 格)。每格:左边场景站位剧情,右边逐句台词+当格穿搭。风格留空就沿用上一张。</div>
      <div class="card" style="cursor:default">
        <label style="margin:0">全局画风(做每格基底,当格「风格」拼在它后面;想蝙蝠侠那种黑暗风选「黑暗蝙蝠侠风」)</label>
        <div class="row" style="margin-top:6px;flex-wrap:wrap">
          <select onchange="ST.cc.gstyle=this.value" style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
            ${gstyles.map(s=>`<option value="${s.key}" ${s.key===P.gstyle?'selected':''}>${esc(s.name)}</option>`).join('')}
          </select>
          <input type="text" value="${esc(P.gcustom||'')}" oninput="ST.cc.gcustom=this.value" placeholder="自定义画风(英文,填了覆盖预设)" style="flex:1;min-width:200px;padding:7px 9px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
        </div></div>
      <div class="row"><button class="back" onclick="ccAddPanel()">＋ 新增画布</button>
        <span class="small">共 ${P.panels.length} 格 · 约 ${Math.round(P.panels.length*1.5)} 分钟</span></div>
      ${panels||'<div class="box">点「＋ 新增画布」开始搭</div>'}
      <div class="row"><button class="back" onclick="ccSave(51)">💾 保存进度</button>
      <button class="back" onclick="goStep(50)">← 角色设定</button>
      <button class="back" onclick="goStep(55)">🎈 气泡编辑器</button>
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
  if(ST.step===55){ // 气泡编辑器: 选干净分镜 → 拖气泡/改台词/换字体/选类型 → 导出压平成最终图
    const pans=(await j('/api/cc/panels')).panels||[];
    const fonts=(await j('/api/fonts')).fonts||[];
    const chars=(await j('/api/cc/project')).chars||[];
    const cnames=chars.map(c=>c.name).filter(Boolean);
    const B=ST.bub||(ST.bub={cur:'',list:[],seq:1,sel:null,flat:''});
    ST.bubFonts={}; fonts.forEach(f=>ST.bubFonts[f.file]=f.name);
    if(!document.getElementById('bubcss')){ const c=document.createElement('style'); c.id='bubcss'; c.textContent=`
.bub{position:absolute;line-height:1.3;z-index:5}
.bubhandle{position:absolute;top:-16px;left:0;cursor:grab;color:#fff;background:#5b6cff;border-radius:5px;padding:0 5px;font-size:12px;line-height:15px;user-select:none;touch-action:none}
.bubtext{outline:none;cursor:text;padding:8px 10px;border-radius:12px;text-align:center;white-space:pre-wrap;word-break:break-word;background:rgba(255,255,255,.92);color:#1c1a26;border:2px solid #1c1a26}
.bubtext:empty:before{content:'点我改台词';color:#9aa}
.bubtext.sel{box-shadow:0 0 0 2px #5b6cff}
.bubtext.love{border-color:#ff6ea0}
.bubtext.shout{border:3px dashed #1c1a26;border-radius:4px}
.bubtext.narration{background:rgba(22,20,30,.92);color:#ffecaa;border:none;text-align:left}
`; document.head.appendChild(c); }
    const face=fonts.map(f=>`@font-face{font-family:'${f.name}';src:url('/fonts/${f.file}')}`).join('\n');
    if(!document.getElementById('bubface')){ const s=document.createElement('style'); s.id='bubface'; document.head.appendChild(s); }
    document.getElementById('bubface').textContent=face;
    if(!B.cur){
      $('#app').innerHTML=`<h2>🎈 气泡编辑器</h2>`+tabHtml(2)+ccTabHtml(1)+`
        <div class="box">选一格干净分镜,在上面<strong>拖气泡、点字改台词、换字体/类型</strong>;「💾导出这一页」把气泡压平成最终图。气泡里按<strong>回车=新气泡</strong>(Shift+回车才换行)。</div>
        <div class="grid">${pans.map(p=>`<div class="card" onclick="bubOpen('${p.name}')"><img src="${p.url}" style="width:100%;border-radius:8px;pointer-events:none"><div class="small" style="margin-top:4px">${p.name}</div></div>`).join('')||'<div class="box">还没有干净分镜。新版「自定义连载」出图不带气泡,专供这里编辑;先去生成几格。</div>'}</div>
        <p><button class="back" onclick="goStep(51)">← 回分镜</button></p>`;
      return;
    }
    const cur=bubCur();
    const whoOpts=`<option value="">(不带名字)</option>`+cnames.map(n=>`<option ${cur&&cur.who===n?'selected':''}>${esc(n)}</option>`).join('');
    const typeOpts=[['speech','对白'],['love','爱心'],['shout','喊叫'],['narration','旁白']].map(t=>`<option value="${t[0]}" ${cur&&cur.type===t[0]?'selected':''}>${t[1]}</option>`).join('');
    const fontOpts=`<option value="">默认字体</option>`+fonts.map(f=>`<option value="${f.file}" ${cur&&cur.font===f.file?'selected':''}>${esc(f.name)}</option>`).join('');
    $('#app').innerHTML=`<h2>🎈 气泡编辑器 · ${esc(B.cur)}</h2>`+tabHtml(2)+ccTabHtml(1)+`
      <div class="card" style="cursor:default">
        <div class="row" style="flex-wrap:wrap;gap:10px;align-items:center">
          <button class="back" onclick="bubAdd()">＋ 加气泡</button>
          <span>类型 <select onchange="bubSetType(this.value)">${typeOpts}</select></span>
          <span>角色 <select onchange="bubSetWho(this.value)">${whoOpts}</select></span>
          <span>字体 <select onchange="bubSetFont(this.value)">${fontOpts}</select></span>
          <button class="back" onclick="bubDel()">🗑 删选中</button>
          <button onclick="bubExport()">💾 导出这一页</button>
          <button class="back" onclick="bubBack()">← 换一格</button>
        </div>
        <div class="small" style="margin-top:6px">${cur?'已选中气泡,上面的类型/角色/字体作用于它':'点气泡选中;拖 ⠿ 手柄移动;点字直接改;回车=新气泡。'}</div>
      </div>
      <div id="bstage" style="position:relative;display:inline-block;max-width:100%;line-height:0">
        <img id="bimg" src="/cc_out/${B.cur}?t=${Date.now()}" style="max-width:100%;border-radius:10px;display:block" onload="bubSync()">
      </div>
      ${B.flat?`<div class="card" style="cursor:default"><div class="small">✅ 已导出(气泡已压平):</div><img src="${B.flat}&t=${Date.now()}" style="max-width:100%;border-radius:10px"><div class="small" style="margin-top:4px">存成带 _flat 后缀的新图,在 output/custom_comic/;原图仍是干净的,可继续改。</div></div>`:''}
      <p><button class="back" onclick="bubBack()">← 换一格</button><button class="back" onclick="home()">🏠 回首页</button></p>`;
    bubSync();
  }
  // ================= 图片测试场(⚙️齿轮) =================
  if(ST.step===70){ // 测试中心: 选类型 + 任务队列管理(暂停/杀/编辑/删除)
    const d = await j('/api/test/state');
    const vd = await j('/api/video-test/state');
    const st = d.status||{};
    const statLine = d.worker_alive
      ? `<div class="card" style="cursor:default"><b>🟢 后台生图中</b> <span class="small">${esc(st.msg||'')}</span>
         <div class="small" style="margin-top:4px">${st.done||0}/${st.total||0} 张 · ✓${st.ok||0} · ✗${st.fail||0} ${st.folder?`· 📁 ${esc(st.folder)}`:''}</div>
         ${st.state==='paused'?'<div class="small" style="color:#e8d9a8">⏸ 已暂停</div>':''}
         ${ST.testMsg?`<div class="small" style="color:#e8d9a8">${esc(ST.testMsg)}</div>`:''}
         <div class="row" style="margin-top:8px">
           ${st.state==='paused'
             ? `<button class="back" onclick="testCtl('resume')">▶ 继续</button>`
             : `<button class="back" onclick="testCtl('pause')">⏸ 暂停</button>`}
           <button class="back" onclick="cConfirm('杀掉正在跑的这个任务?已生成的图保留,队列里后面的任务会接着跑','杀掉').then(ok=>{if(ok)testCtl('kill_curr')})">⏹ 杀当前任务</button>
           <button class="back" onclick="cConfirm('连后台worker一起杀?队列里剩余任务也会停','全部杀掉').then(ok=>{if(ok)testCtl('kill_all')})">🛑 杀全部</button>
         </div></div>`
      : `<div class="box">后台没在跑。建好任务点「一键生图」就会自动起后台进程——<b>关这个网页、甚至关掉 start.sh,它都照跑</b>;想停只能来这里暂停/杀死,或在终端 <code>./start.sh</code> 选 4。</div>`;
    const vst=vd.status||{};
    const vcurrent=(vd.tasks||[]).find(t=>String(t.id)===String(vst.task))||{};
    const vtiming=vcurrent.timing||{};
    const vtimingLine=vtiming.average_sec
      ? `<div class="small" style="margin-top:5px">本次实际平均约 <b>${videoFmtTime(vtiming.average_sec)}</b>/条 · 预计还需 ${videoFmtTime(vtiming.remaining_sec)}</div>`
      : '';
    const videoStatLine=vd.worker_alive
      ? `<div class="card" style="cursor:default"><b>🟢 后台生视频中</b> <span class="small">${esc(vst.msg||'')}</span>
         <div class="small" style="margin-top:4px">${vst.done||0}/${vst.total||0} 条成片 · ✓${vst.ok||0} · ✗${vst.fail||0}${vst.folder?` · 📁 ${esc(vst.folder)}`:''}</div>
         ${vtimingLine}
         <div class="row" style="margin-top:8px">
           ${vst.state==='paused'?`<button class="back" onclick="videoTestCtl('resume')">▶ 继续</button>`:`<button class="back" onclick="videoTestCtl('pause')">⏸ 暂停</button>`}
           <button class="back" onclick="cConfirm('停止当前视频任务?已完成的视频会保留','停止').then(ok=>{if(ok)videoTestCtl('kill_curr')})">⏹ 杀当前任务</button>
           <button class="back" onclick="cConfirm('停止全部视频测试?队列后面的任务不会继续跑','全部停止').then(ok=>{if(ok)videoTestCtl('kill_all')})">🛑 杀全部</button>
         </div></div>`
      : `<div class="box">后台视频测试没在跑。创建任务后会自动启动；终端 <code>./start.sh</code> 选 5 可监控，选 6 可停止。</div>`;
    const taskRows = (d.tasks||[]).map(t=>{
      const nm = (t.models||[]).map(m=>esc(m.folder||m.name||m.id)).join('、');
      const badge = {queued:'⏳ 排队中',running:'🟢 生成中',done:'✅ 完成',partial:'⚠️ 部分成功',failed:'❌ 失败',killed:'⏹ 被杀'}[t.state]||t.state;
      return `<div class="card" style="cursor:default">
        <b>任务 ${esc(t.id)}</b> <span class="small">${badge}</span>
        <div class="small" style="margin-top:4px">${(t.models||[]).length}模型:${nm} · 每提示词${t.per_prompt}张 · 画布${t.canvas.w}×${t.canvas.h}${t.pad_ref?' · 垫图':''}${(t.face_lock||{}).enabled?' · 锁脸':''}</div>
        <div class="row" style="margin-top:8px">
          <button class="back" onclick="testEditPrompts('${t.id}')">✏️ 提示词</button>
          ${['done','partial','failed','killed'].includes(t.state)?`<button class="back" onclick="testRerun('${t.id}')">🔁 原样重跑</button>
          <button class="back" onclick="testRerunEdit('${t.id}')">📝 改配置重跑</button>`:''}
          <button class="back" onclick="cConfirm('删任务 ${esc(t.id)}?只删配置和提示词,已生成的图片保留','删除').then(ok=>{if(ok)testDel('${t.id}')})">🗑 删除</button>
        </div></div>`;
    }).join('');
    const videoRows=(vd.tasks||[]).map(t=>{
      const nm=(t.models||[]).map(m=>esc(m.folder||m.name||m.id)).join('、');
      const badge={queued:'⏳ 排队中',running:'🟢 生成中',done:'✅ 完成',partial:'⚠️ 部分成功',failed:'❌ 失败',killed:'⏹ 被杀'}[t.state]||t.state;
      const vars=(t.variants||[]).map(v=>`${v.w}×${v.h}/${v.duration}秒/${v.fps}fps`).join('、');
      const outputLinks=(t.outputs||[]).slice(-4).map(p=>{const rel=String(p).replace(/^output\/vidtest\//,'');return `<a href="/vidtest/${encodeURI(rel)}" target="_blank">${esc(rel.split('/').pop())}</a>`}).join(' · ');
      return `<div class="card" style="cursor:default"><b>视频任务 ${esc(t.id)}</b> <span class="small">${badge}</span>
        <div class="small" style="margin-top:4px">${(t.models||[]).length}模型:${nm} · ${t.mode==='t2v'?'纯文字':'垫图'+(t.refs||[]).length+'张'} · ${vars} · 每组${t.copies||1}段</div>
        ${outputLinks?`<div class="small" style="margin-top:5px">成品：${outputLinks}</div>`:''}
        <div class="row" style="margin-top:8px"><button class="back" onclick="videoTestEditPrompts('${t.id}')">✏️ 提示词</button>
          ${['done','partial','failed','killed'].includes(t.state)?`<button class="back" onclick="videoTestRerun('${t.id}')">🔁 原样重跑</button><button class="back" onclick="videoTestRerunEdit('${t.id}')">📝 改配置重跑</button>`:''}
          <button class="back" onclick="cConfirm('删除视频任务 ${esc(t.id)}?成品视频会保留','删除').then(ok=>{if(ok)videoTestDel('${t.id}')})">🗑 删除</button></div></div>`;
    }).join('');
    $('#app').innerHTML = `<div class="vt-page test-hub"><h2>⚙️ 测试场</h2>
      <div class="test-hub-grid">
        <div class="card test-entry" onclick="goStep(88)"><b>🖼️ 图片测试</b><div class="small">${d.worker_alive?`<span class="dot on">● 运行中</span> · ${st.done||0}/${st.total||0} 张`:'后台空闲'}</div><div class="small">图片任务、提示词和图片重试独立管理</div></div>
        <div class="card test-entry" onclick="goStep(89)"><b>🎬 视频测试</b><div class="small">${vd.worker_alive?`<span class="dot on">● 运行中</span> · ${vst.done||0}/${vst.total||0} 条`:'后台空闲'}</div><div class="small">视频任务、成片文件夹和视频重试独立管理</div></div>
      </div>
      <div class="box">图片和视频的任务页面已经分开；内存保护仍然会阻止两种后台同时运行。</div>
      <p><button class="back" onclick="home()">🏠 回首页</button></p></div>`;
  }
  if(ST.step===88){ // 图片测试任务管理页
    const d=await j('/api/test/state'), st=d.status||{};
    const badge={queued:'⏳ 排队中',running:'🟢 生成中',done:'✅ 完成',partial:'⚠️ 部分成功',failed:'❌ 失败',killed:'⏹ 被杀'};
    const rows=(d.tasks||[]).map(t=>{
      const nm=(t.models||[]).map(m=>esc(m.folder||m.name||m.id)).join('、');
      return `<div class="card" style="cursor:default"><b>图片任务 ${esc(t.id)}</b> <span class="small">${badge[t.state]||t.state}</span>
        <div class="small">${(t.models||[]).length}模型：${nm} · 每提示词${t.per_prompt}张 · ${t.canvas.w}×${t.canvas.h}</div>
        <div class="row" style="margin-top:8px"><button class="back" onclick="testEditPrompts('${t.id}')">✏️ 图片提示词</button>
          ${['done','partial','failed','killed'].includes(t.state)?`<button class="back" onclick="testRerun('${t.id}')">🔁 图片原样重跑</button><button class="back" onclick="testRerunEdit('${t.id}')">📝 图片改配置重跑</button>`:''}
          <button class="back" onclick="cConfirm('删除图片任务 ${esc(t.id)}？已生成图片会保留','删除').then(ok=>{if(ok)testDel('${t.id}')})">🗑 删除</button></div></div>`;
    }).join('');
    const controls=d.worker_alive?`<div class="card" style="cursor:default"><b>🟢 图片后台运行中</b><div class="small">${esc(st.msg||'')} · ${st.done||0}/${st.total||0} 张</div><div class="row" style="margin-top:8px">${st.state==='paused'?`<button class="back" onclick="testCtl('resume')">▶ 继续</button>`:`<button class="back" onclick="testCtl('pause')">⏸ 暂停</button>`}<button class="back" onclick="testCtl('kill_curr')">⏹ 杀当前图片任务</button><button class="back" onclick="testCtl('kill_all')">🛑 停止图片后台</button></div></div>`:'<div class="box">图片后台当前空闲。</div>';
    $('#app').innerHTML=`<div class="vt-page"><h2>🖼️ 图片测试任务</h2><div class="row"><button onclick="testNew()">＋ 新建图片测试</button><button class="back" onclick="goStep(70)">← 测试场首页</button></div>${controls}<h3>图片任务列表</h3>${rows||'<div class="box">还没有图片任务。</div>'}<p><button class="back" onclick="render()">🔄 刷新</button></p></div>`;
  }
  if(ST.step===89){ // 视频测试任务管理页，和图片页面完全分开
    const vd=await j('/api/video-test/state'), s=vd.status||{};
    const badge={queued:'⏳ 排队中',waiting_engine:'⏳ 等当前视频后台结束',running:'🟢 生成中',done:'✅ 完成',partial:'⚠️ 部分成功',failed:'❌ 失败',killed:'⏹ 被停止',interrupted:'⏸ 上次未完成',paused:'⏸ 已暂停'};
    const rows=(vd.tasks||[]).map(t=>{
      const nm=(t.models||[]).map(m=>esc(m.output_folder||m.folder||m.name||m.id)).join('、');
      const vars=(t.variants||[]).map(v=>`${v.w}×${v.h}/${v.duration}秒/${v.fps}fps`).join('、');
      const links=(t.outputs||[]).slice(-6).map(p=>{const rel=String(p).replace(/^output\/vidtest\//,'');return `<a href="/vidtest/${encodeURI(rel)}" target="_blank">${esc(rel.split('/').pop())}</a>`}).join(' · ');
      const canRetry=['done','partial','failed','killed'].includes(t.state);
      const progress=t.progress||{};
      const timing=t.timing||{};
      const resumeLine=t.resume_available
        ? `<div class="small" style="margin-top:6px;color:var(--accent)">上次停在：已完成 <b>${progress.done||0}/${progress.total||'?'}</b> 条${progress.model?`，当前模型：${esc(progress.model)}`:''}。正在生成的这一条会从头重做。</div>`
        : '';
      const timingLine=timing.average_sec
        ? `<div class="small" style="margin-top:5px">本次实际平均：约 <b>${videoFmtTime(timing.average_sec)}</b>/条 · 已运行 ${videoFmtTime(timing.elapsed_sec)} · 按当前速度还需约 ${videoFmtTime(timing.remaining_sec)}</div>`
        : (timing.elapsed_sec?`<div class="small" style="margin-top:5px">本次已运行：${videoFmtTime(timing.elapsed_sec)}，完成第一条后会显示实际平均速度。</div>`:'');
      return `<div class="card" style="cursor:default"><b>视频任务 ${esc(t.id)}</b> <span class="small">${badge[t.state]||t.state}</span>
        <div class="small">${(t.models||[]).length}模型：${nm} · ${t.mode==='t2v'?'纯文字':'垫图'+(t.refs||[]).length+'张'} · ${vars} · 每组${t.copies||1}条</div>
        ${(t.output_dirs||[]).length?`<div class="small">成片文件夹：${t.output_dirs.map(esc).join('、')}</div>`:''}
        ${links?`<div class="small" style="margin-top:5px">最近成片：${links}</div>`:''}
        ${resumeLine}
        ${timingLine}
        <div class="row" style="margin-top:8px"><button class="back" onclick="videoTestEditPrompts('${t.id}')">✏️ 视频提示词</button>
          ${t.resume_available?`<button onclick="videoTestResume('${t.id}')">▶ 从断点继续</button>`:''}
          ${canRetry?`<button class="back" onclick="videoTestRerun('${t.id}')">🔁 视频原样重跑</button><button class="back" onclick="videoTestRerunEdit('${t.id}')">📝 视频改配置重跑</button>`:''}
          <button class="back" onclick="cConfirm('删除视频任务 ${esc(t.id)}？成片和分段工作文件都会删除','删除').then(ok=>{if(ok)videoTestDel('${t.id}')})">🗑 删除成片和临时文件</button></div></div>`;
    }).join('');
    const controls=vd.worker_alive?`<div class="card" style="cursor:default"><b>🟢 视频后台运行中</b><div class="small">${esc(s.msg||'')} · ${s.done||0}/${s.total||0} 条</div><div class="row" style="margin-top:8px">${s.state==='paused'?`<button class="back" onclick="videoTestCtl('resume')">▶ 继续</button>`:`<button class="back" onclick="videoTestCtl('pause')">⏸ 暂停</button>`}<button class="back" onclick="videoTestCtl('kill_curr')">⏹ 杀当前视频任务</button><button class="back" onclick="videoTestCtl('kill_all')">🛑 停止视频后台</button></div></div>`:'<div class="box">视频后台当前空闲。</div>';
    $('#app').innerHTML=`<div class="vt-page"><h2>🎬 视频测试任务</h2><div class="row"><button onclick="videoTestNew()">＋ 新建视频测试</button><button class="back" onclick="goStep(70)">← 测试场首页</button></div>${controls}<h3>视频任务列表</h3>${rows||'<div class="box">还没有视频任务。</div>'}<p><button class="back" onclick="render()">🔄 刷新</button></p></div>`;
  }
  if(ST.step===71){ // 向导①: 选模型(多选)
    const d = await j('/api/test/state');
    const T = ST.test;
    T.all = d.models;   // 可选模型清单,供「全选」用
    const cards = d.models.map((m,mi)=>{
      const on = T.models.find(x=>x.id===m.id);
      return `<div class="card drop-in ${on?'sel':''}" style="animation-delay:${mi*70}ms" onclick="testToggleModel('${m.id}','${esc(m.name)}',${m.sec||60},this)">
        <b>${esc(m.name)}</b> <span class="small mtime ${secCls(m.sec)}">${m.time||('约'+m.sec+'秒')}/张</span>
        <div class="small">${esc(m.desc||'')}</div></div>`;
    }).join('');
    $('#app').innerHTML = `<h2>图片测试 ① 选模型</h2>
      <div class="box">点选一个或多个模型(可全选对比同一套提示词)。已选 <b id="tcount">${T.models.length}</b> 个。<button class="back" style="padding:4px 12px;font-size:13px;margin-left:8px" onclick="testSelectAll()">☑ 全选/清空</button></div>
      ${cards}
      <div class="box" id="torder">${testOrderHtml()}</div>
      <p><button class="back" onclick="goStep(70)">← 取消</button>
      <button onclick="if(ST.test.models.length)goStep(72);else cAlert('至少选一个模型')">下一步 →</button></p>`;
  }
  if(ST.step===72){ // 向导②: 每个模型的安全输出文件夹名
    const T = ST.test;
    const rows = T.models.map((m,i)=>`
      <div class="row" style="margin-top:6px;align-items:center">
        <span style="flex:0 0 180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(m.name)}</span>
        <input id="tf${i}" type="text" value="${esc(m.folder)}" placeholder="文件夹名"
          style="flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text);font-size:13px">
      </div>`).join('');
    $('#app').innerHTML = `<h2>图片测试 ② 输出文件夹</h2>
      <div class="box">每个模型的图存到 <code>output/imgtest/&lt;文件夹名&gt;/</code>。系统会自动把空格、点号和其他符号改成下划线，避免文件夹名称报错。</div>
      ${rows}
      <div class="box" style="margin-top:10px"><b>批量追加名称</b>
        <div class="row" style="margin-top:8px;flex-wrap:wrap;align-items:flex-end">
          <span><label>所有文件夹末尾追加</label><input id="tfolderSuffix" value="${esc(T.folder_suffix||'')}" placeholder="例如 _v2"></span>
          <span><label>所有图片文件名末尾追加</label><input id="tfileSuffix" value="${esc(T.file_suffix||'')}" placeholder="例如 _v2"></span>
        </div>
        <div class="small">输入 <b>v2</b> 或 <b>_v2</b> 都可以。示例：文件夹变成 <code>模型名_v2</code>，图片变成 <code>p01_1_v2.png</code>。</div>
      </div>
      <p><button class="back" onclick="goStep(71)">← 上一步</button>
      <button onclick="testSetFolders(${T.models.length})">下一步 →</button></p>`;
  }
  if(ST.step===73){ // 向导③: 统一画布
    const T = ST.test;
    $('#app').innerHTML = `<h2>图片测试 ③ 画布尺寸</h2>
      <div class="box">所有模型用同一个画布。选预设或自定义。</div>
      <div class="row" style="flex-wrap:wrap;gap:8px">
        ${[[640,960,'640×960 竖·快'],[1024,720,'1024×720 横'],[832,1216,'832×1216 竖·标准'],[1216,832,'1216×832 横·宽'],[1024,1024,'1024×1024 方']].map(c=>
          `<button class="back" onclick="testSetCanvas(${c[0]},${c[1]})">${c[2]}</button>`).join('')}
      </div>
      <div class="row" style="margin-top:10px;align-items:center">
        <span class="small">自定义</span>
        <input id="cw" type="number" value="${T.canvas.w}" style="width:90px;padding:7px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)"> ×
        <input id="ch" type="number" value="${T.canvas.h}" style="width:90px;padding:7px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
        <button class="back" onclick="testSetCanvas(+document.getElementById('cw').value,+document.getElementById('ch').value)">用这个</button>
      </div>
      <div class="small" id="tcur" style="margin-top:6px">当前:<b>${T.canvas.w}×${T.canvas.h}</b></div>
      <p><button class="back" onclick="goStep(72)">← 上一步</button>
      <button onclick="goStep(74)">下一步 →</button></p>`;
  }
  if(ST.step===74){ // 向导④: 统一提示词 + 每条张数
    const T = ST.test;
    $('#app').innerHTML = `<h2>图片测试 ④ 提示词 & 数量</h2>
      <div class="box"><b>一行一条提示词</b>,所有模型共用这套。每条提示词生成 1~10 张(不同 seed)。<br>
      <span class="small">提示词也会存成 <code>test_jobs/prompts/&lt;任务号&gt;.txt</code>,之后能直接改文件,也能在队列页网页改。</span></div>
      <label>提示词(一行一条)</label>
      <textarea id="tpro" style="height:140px" placeholder="masterpiece, 1girl, ...&#10;masterpiece, 1boy, ...">${esc(T.prompts)}</textarea>
      <div class="row" style="margin-top:8px;align-items:center">
        <span>每条提示词生成</span>
        <input id="tper" type="number" min="1" max="10" value="${T.per}" style="width:70px;padding:7px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">
        <span>张</span></div>
      <p><button class="back" onclick="goStep(73)">← 上一步</button>
      <button onclick="testSetPrompts()">下一步 →</button></p>`;
  }
  if(ST.step===75){ // 向导⑤: 垫图(可选)
    const d = await j('/api/test/state');
    const T = ST.test;
    const imgs = (d.refs||[]).map(r=>`
      <div class="card tpad ${T.pad_ref===r.name?'sel':''}" style="width:110px" onclick="testSetPad('${r.name}',this)">
        <img src="${r.url}" style="width:100%;border-radius:8px;pointer-events:none"><div class="small">${esc(r.name)}</div></div>`).join('');
    $('#app').innerHTML = `<h2>图片测试 ⑤ 垫图(画风参考)</h2>
      <div class="box">选一张 refs/ 里的图,所有模型按它的画风生成(i2i 0.75);不选就是纯文字生图。${(!d.refs||!d.refs.length)?'<br><span class="small">refs/ 现在是空的,可先去首页图床上传,或直接下一步跳过。</span>':''}</div>
      <div class="row" style="flex-wrap:wrap;gap:8px">${imgs}</div>
      <div class="small" id="tpadcur" style="margin-top:6px">${T.pad_ref?'已选垫图: <b>'+esc(T.pad_ref)+'</b>':'当前: <b>不垫图(纯文字)</b>'}</div>
      <div class="row" style="margin-top:6px"><button class="back" onclick="testSetPad('',null)">✖ 不垫图</button></div>
      <p><button class="back" onclick="goStep(74)">← 上一步</button>
      <button onclick="goStep(76)">下一步 →</button></p>`;
  }
  if(ST.step===76){ // 向导⑥: 脸部锁图(可选)
    const d = await j('/api/test/state');
    const T = ST.test;
    const fl = T.face_lock;
    const mopts = T.models.map(m=>`<option value="${m.id}" ${fl.gen_model===m.id?'selected':''}>${esc(m.name)}</option>`).join('');
    $('#app').innerHTML = `<h2>图片测试 ⑥ 脸部锁图(可选)</h2>
      <div class="box">开了锁脸: 系统<b>先用你选的模型生成一张脸</b>(存 refs/),之后所有模型都锁住这张脸生成(IPAdapter)。适合测不同模型画同一个人的差别。</div>
      <div class="row" style="align-items:center;gap:10px">
        <span>锁脸</span>
        <button class="back" id="tlockbtn" onclick="testSetLock()">${fl.enabled?'🔒 已开(点我关)':'○ 已关(点我开)'}</button>
      </div>
      <div id="tlockform" style="margin-top:10px;${fl.enabled?'':'display:none'}">
        <label>用哪个模型生成锁脸图</label>
        <select id="tlockm" style="width:100%;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--input);color:var(--text)">${mopts}</select>
        <label style="margin-top:8px">锁脸图提示词(这张脸长什么样)</label>
        <textarea id="tlockp" style="height:70px" placeholder="masterpiece, 1girl, long hair, detailed face, looking at viewer">${esc(fl.prompt)}</textarea>
      </div>
      <p><button class="back" onclick="goStep(75)">← 上一步</button>
      <button onclick="testLockNext()">下一步 →</button></p>`;
  }
  if(ST.step===77){ // 向导⑦: 确认 + 一键生图
    const T = ST.test;
    const plist = T.prompts.split('\n').filter(x=>x.trim());
    const total = T.models.length * plist.length * T.per;
    $('#app').innerHTML = `<h2>图片测试 ⑦ 确认</h2>
      <div class="card" style="cursor:default">
        <b>${T.models.length} 个模型</b> × <b>${plist.length} 条提示词</b> × 每条 <b>${T.per} 张</b> = <b>${total} 张图</b>
        <div class="small" style="margin-top:6px">模型:${T.models.map(m=>esc(m.folder||m.name)).join('、')}</div>
        <div class="small">画布:${T.canvas.w}×${T.canvas.h} · ${T.pad_ref?'垫图:'+esc(T.pad_ref):'不垫图'} · ${T.face_lock.enabled?'锁脸':'不锁脸'} · 文件夹后缀:${esc(T.folder_suffix||'无')} · 图片后缀:${esc(T.file_suffix||'无')}</div>
        <div class="small" style="margin-top:6px">提示词:<br>${plist.map(p=>'· '+esc(p)).join('<br>')}</div>
      </div>
      <div class="box">点「一键生图」后任务进队列,后台 worker 逐个跑。<b>这时你关网页、关 start.sh 都不影响</b>;想停回测试场暂停/杀死,或终端 <code>./start.sh</code> 选 4。</div>
      <p><button class="back" onclick="goStep(76)">← 上一步</button>
      <button onclick="testGo()">🚀 一键生图</button></p>`;
  }
  if(ST.step===78){ // 进度页(轮询 status)
    const d = await j('/api/test/state');
    const st = d.status||{};
    const pct = st.total?Math.round((st.done||0)/st.total*100):0;
    const el = st.t0?Math.round((st.ts||0)-st.t0):0;
    $('#app').innerHTML = `<h2>后台生图${d.worker_alive?'中…':'(已停)'}</h2>
      <div class="card" style="cursor:default">
        <span style="font-size:22px;font-weight:700">${st.done||0}/${st.total||0}</span> 张 ·
        <span style="color:var(--on)">✓ ${st.ok||0}</span> · <span style="color:#e0556b">✗ ${st.fail||0}</span>
        ${st.msg?`<div class="small" style="margin-top:4px">${esc(st.msg)}</div>`:''}
        ${st.folder?`<div class="small">📁 ${esc(st.folder)}</div>`:''}
        <div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div>
        <div class="small" style="margin-top:4px">${st.state==='paused'?'⏸ 已暂停':'🟢 运行中'} · 本张已用 ${el} 秒</div>
        <div class="row" style="margin-top:8px">
          ${st.state==='paused'
            ? `<button class="back" onclick="testCtl('resume')">▶ 继续</button>`
            : `<button class="back" onclick="testCtl('pause')">⏸ 暂停</button>`}
          <button class="back" onclick="cConfirm('杀掉正在跑的这个任务?已生成的图保留,队列里后面的任务会接着跑','杀掉').then(ok=>{if(ok)testCtl('kill_curr')})">⏹ 杀当前任务</button>
          <button class="back" onclick="cConfirm('连后台worker一起杀?队列里剩余任务也会停','全部杀掉').then(ok=>{if(ok)testCtl('kill_all')})">🛑 杀全部</button>
        </div></div>
      <div class="card" style="cursor:default"><div class="small" style="white-space:pre-wrap;font-family:monospace">${(st.log||[]).slice(-12).map(esc).join('\n')}</div></div>
      <p><button class="back" onclick="goStep(70)">← 回测试场</button>
      <button class="back" onclick="render()">🔄 刷新</button></p>`;
    if(d.worker_alive) setTimeout(()=>{ if(ST.step===78) render(); },2500);
  }
  if(ST.step===79){ // 网页编辑某任务的提示词
    const r = await j('/api/test/prompts?f='+encodeURIComponent(ST.testEditId||''));
    $('#app').innerHTML = `<h2>✏️ 图片任务 ${esc(ST.testEditId||'')} 提示词</h2>
      <div class="box">直接改,<b>一行一条</b>,保存即生效(正在跑的那张不受影响,下一张起用新词)。也可直接改文件 <code>${esc(r.path||'')}</code>。</div>
      <textarea id="tpe" style="height:200px">${esc(r.text||'')}</textarea>
      <p><button class="back" onclick="goStep(ST.testReturnStep||88)">← 图片任务</button>
      <button onclick="testSavePrompts()">💾 保存</button></p>`;
  }
  if(ST.step===80){ // 视频测试① 选模型
    const d=await j('/api/video-test/state'); const V=ST.videoTest; V.all=d.models; V.loras=d.loras||[];
    const cards=(d.models||[]).map((m,i)=>{const on=V.models.some(x=>x.id===m.id);const avg=m.actual_average_sec||0;const speed=avg?`本机平均约${avg}秒/段（${m.actual_samples}条成片）`:`登记约${m.sec||300}秒/段`;return `<div class="card drop-in ${on?'sel':''}" data-vtmodel="${m.id}" style="animation-delay:${i*70}ms" onclick="videoTestToggleModel('${m.id}','${esc(m.name)}',${avg||m.sec||300},this)"><b>${esc(m.name)}</b> <span class="small mtime ${secCls(avg||m.sec)}">${speed}</span><div class="small">${esc(m.desc||'')}${m.t2v_experimental?' · 纯文字为试验模式':''}</div></div>`}).join('');
    $('#app').innerHTML=`<h2>视频测试 ① 选模型</h2><div class="box">可单选或多选。默认按预计生成时间从快到慢排队。已选 <b id="vtcount">${V.models.length}</b> 个。<button class="back" onclick="videoTestSelectAll()">☑ 全选/清空</button></div>${cards}<div class="box" id="vtorder">${videoTestOrderHtml()}</div><p><button class="back" onclick="goStep(89)">← 取消</button><button onclick="if(ST.videoTest.models.length)goStep(81);else cAlert('至少选一个视频模型')">下一步 →</button></p>`;
  }
  if(ST.step===81){ // 视频测试② 输出命名
    const V=ST.videoTest;
    videoTestSyncFolders();
    const rows=V.models.map((m,i)=>`<div class="vt-name-row"><div class="vt-name-model"><span class="vt-name-index">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(m.name)}</strong><small>${m.custom_name?'已自定义名称':'默认模型名称'}</small></div></div><div class="vt-name-control"><label for="vtf${i}">输出文件夹名称</label><div class="vt-name-control-line"><div class="vt-pretty-input vt-input-shell"><span class="vt-field-icon">📁</span><input id="vtf${i}" value="${esc(m.renaming?m.base_folder:m.folder)}" ${m.renaming?'':'disabled'} placeholder="模型默认名称"><span class="vt-folder-tail">/</span></div><button class="back vt-rename-btn" onclick="videoTestRename(${i})">${m.renaming?'保存':'重命名'}</button></div></div></div>`).join('');
    $('#app').innerHTML=`<div class="vt-page"><h2>视频测试 ② 输出命名</h2><div class="box"><b>最终成片</b> 会放到 <code>output/vidtest/</code> 下；临时分段会放到任务工作目录。模型默认文件夹已锁定，点击右侧「重命名」后才能修改。</div><div class="vt-panel-title">本任务的模型文件夹</div><div class="vt-name-list">${rows}</div><div class="vt-panel-title">批量命名（可选）</div><div class="vt-name-options"><div class="vt-name-option"><label for="vtFolderSuffixInput">文件夹追加后缀</label><div class="vt-pretty-input"><span class="vt-field-icon">＋</span><input id="vtFolderSuffixInput" value="${esc(V.folder_suffix||'')}" placeholder="例如：_舞蹈"></div><div class="vt-name-option-actions"><button class="back" onclick="videoTestApplySuffix('folder')">应用</button><button class="back" onclick="videoTestRemoveSuffix('folder')">删除</button></div></div><div class="vt-name-option"><label for="vtFileSuffixInput">成片文件名追加后缀</label><div class="vt-pretty-input"><span class="vt-field-icon">Aa</span><input id="vtFileSuffixInput" value="${esc(V.file_suffix||'')}" placeholder="例如：_成片01"></div><div class="vt-name-option-actions"><button class="back" onclick="videoTestApplySuffix('file')">应用</button><button class="back" onclick="videoTestRemoveSuffix('file')">删除</button></div></div></div><div class="small vt-help">点击「应用」会替换当前后缀；需要多层命名时，直接在输入框中完整填写。</div><div class="vt-nav-actions"><button class="back" onclick="goStep(80)">← 上一步</button><button onclick="videoTestSetNames()">下一步 →</button></div></div>`;
  }
  if(ST.step===82){ // 视频测试③ T2V/I2V + 多垫图
    const d=await j('/api/video-test/state'); const V=ST.videoTest;
    const refs=(d.refs||[]).map(r=>`<div class="card vtref ${V.refs.includes(r.name)?'sel':''}" style="width:120px" onclick="videoTestToggleRef('${r.name}',this)"><img src="${r.url}" style="width:100%;border-radius:6px;pointer-events:none"><div class="small">${esc(r.name)}</div></div>`).join('');
    $('#app').innerHTML=`<h2>视频测试 ③ 生成方式</h2><div class="row" style="gap:8px"><button class="${V.mode==='i2v'?'':'back'}" onclick="videoTestMode('i2v')">🖼️ 用图片做视频</button><button class="${V.mode==='t2v'?'':'back'}" onclick="videoTestMode('t2v')">📝 只写文字做视频</button></div><div class="box" style="margin-top:10px">${V.mode==='i2v'?'用一张或多张图片做参考，提示词主要写人物怎么动。通常更容易保持人物和画风。':'不上传图片，直接写人物、场景和动作。文字越具体，画面越容易稳定。'}</div><div id="vtrefs" style="display:${V.mode==='i2v'?'block':'none'}"><div class="row" style="flex-wrap:wrap;gap:8px">${refs||'<div class="box">参考图库里还没有图片，请先上传一张图片。</div>'}</div><div class="small">已选 <b id="vtrefcount">${V.refs.length}</b> 张</div></div><p><button class="back" onclick="goStep(81)">← 上一步</button><button onclick="videoTestRefsNext()">下一步 →</button></p>`;
  }
  if(ST.step===83){ // 视频测试④ 批量参数
    const V=ST.videoTest; if(!Array.isArray(V.lora_ids))V.lora_ids=V.lora&&V.lora!=='none'?[V.lora]:[]; if(V.style_2d&&!V.lora_ids.includes('anime'))V.lora_ids.unshift('anime'); const loras=(V.loras||[]).filter(l=>l.id!=='none').map(l=>`<label class="vt-toggle" style="display:block;margin:6px 0"><input type="checkbox" name="vtlora" value="${l.id}" ${V.lora_ids.includes(l.id)?'checked':''}> <b>${esc(l.name)}</b> <span class="small">${esc(l.desc||'')}</span></label>`).join('');
    const sizes=[[144,256,'144p 竖'],[180,320,'180p 竖'],[240,432,'240p 竖'],[360,640,'360p 竖'],[480,848,'480p 竖'],[720,1280,'720p 竖'],[1080,1920,'1080p 竖'],[256,144,'144p 横'],[320,180,'180p 横'],[432,240,'240p 横'],[640,360,'360p 横'],[848,480,'480p 横'],[1280,720,'720p 横'],[1920,1080,'1080p 横']];
    const sizeHtml=sizes.map(s=>{const k=s[0]+'x'+s[1];return `<label class="vt-choice"><input type="checkbox" ${V.sizes.includes(k)?'checked':''} onchange="videoTestToggleArray('sizes','${k}',this.checked)"> <span>${s[2]}<small>${s[0]}×${s[1]}</small></span></label>`}).join('');
    const durHtml=[1,2,3,4,5,6,8,10,15,20,30,60].map(n=>`<label class="vt-choice"><input type="checkbox" ${V.durations.includes(n)?'checked':''} onchange="videoTestToggleArray('durations',${n},this.checked)"><span>${n}秒</span></label>`).join('');
    const fpsHtml=[[24,'24'],[29.97,'29.97'],[59.94,'59.94'],[90,'90']].map(n=>`<label class="vt-choice"><input type="checkbox" ${V.fpses.includes(n[0])?'checked':''} onchange="videoTestToggleArray('fpses',${n[0]},this.checked)"><span>${n[1]} fps</span></label>`).join('');
    const style2dHtml=`<div class="box vt-param-panel"><label class="vt-toggle"><input id="vt2d" type="checkbox" ${V.style_2d?'checked':''}> 2D 动漫模式</label><div class="small">自动勾选“动漫/幻想风”，并在你的镜头和动作提示词后追加 2D、赛璐璐、手绘线稿等约束，同时排除 3D、写实和真人摄影感。</div></div>`;
    $('#app').innerHTML=`<h2>视频测试 ④ 批量参数</h2><div class="box vt-param-panel"><div class="vt-panel-title">输出尺寸（可多选）</div><div class="vt-choice-grid">${sizeHtml}</div><div class="vt-form-grid"><span><label>自定义宽</label><input id="vtcw" type="number" min="64" max="1920" step="16" value="${V.custom.w}"></span><span><label>自定义高</label><input id="vtch" type="number" min="64" max="1920" step="16" value="${V.custom.h}"></span><button class="back" onclick="videoTestAddCustomSize()">＋ 加入自定义尺寸</button></div><div class="small">已选：${V.sizes.map(esc).join('、')||'无'}</div><div class="small">720p/1080p 会先用本机安全尺寸生成，全部拼好后再统一放大到目标尺寸。</div></div><div class="box vt-param-panel"><div class="vt-panel-title">成片时长（可多选）</div><div class="vt-choice-grid vt-choice-grid-short">${durHtml}</div><div class="small">超过6秒会自动均分，例如8秒=4+4、10秒=5+5；全部完成后再拼成一个成片。</div></div><div class="box vt-param-panel"><div class="vt-panel-title">最终帧率（可多选）</div><div class="vt-choice-grid vt-choice-grid-short">${fpsHtml}</div><div class="small">29.97/59.94/90fps 都先按最高24fps生成，再后期补到目标帧率。</div></div><div class="vt-form-grid"><span><label>每个组合生成几条成片</label><input id="vtcopies" type="number" min="1" max="10" value="${V.copies}"></span><span><label>风格 LoRA</label><select id="vtlora">${loras}</select></span><label class="vt-toggle"><input id="vtstg" type="checkbox" ${V.stg?'checked':''}> STG 时序稳定（LTX）</label><label class="vt-toggle"><input id="vtinterp" type="checkbox" ${V.interpolate?'checked':''}> 低帧率也补到60fps</label></div><div class="small vt-setting-note"><b>大白话：</b>STG 时序稳定是让画面更连贯、少闪烁；LoRA 里的「动作增强」是让动作幅度更大。它们不是一个东西，可以同时开。</div><p><button class="back" onclick="goStep(82)">← 上一步</button><button onclick="videoTestParamsNext()">下一步 →</button></p>`;
    $('#app h2').insertAdjacentHTML('afterend',style2dHtml);
    $('#vt2d').onchange=()=>{ V.style_2d=$('#vt2d').checked; if(V.style_2d&&!V.lora_ids.includes('anime'))V.lora_ids.unshift('anime'); if(!V.style_2d)V.lora_ids=V.lora_ids.filter(x=>x!=='anime'); render(); };
    const nativeAudioNote=$('.vt-setting-note');
    if(nativeAudioNote) nativeAudioNote.insertAdjacentHTML('beforebegin', `<label class="vt-toggle"><input id="vtnativeaudio" type="checkbox" ${V.native_audio?'checked':''}> LTX 原生声音</label>`);
    const oldLora=$('#vtlora');
    if(oldLora){ const wrap=oldLora.parentElement; wrap.innerHTML=`<label>效果 LoRA（可多选）</label>${loras}<div class="small">动漫风：尽量保留二次元画风；动作增强：动作更明显；兽人向：兽人/特殊题材。叠加越多，互相干扰的可能性越大。</div><div class="small">空间放大器只是把视频变大，不改变人物和画风；VAE、文本编码器都是模型运行所需的固定配件，都不是需要手动勾选的效果。</div>`; }
    if(nativeAudioNote) nativeAudioNote.innerHTML=`<b>大白话：</b>打开「LTX 原生声音」后，LTX 会根据提示词同时生成喘息、喊叫、脚步、衣物摩擦等现场声音，不是后期硬塞音频。STG 时序稳定是让画面更连贯、少闪烁；LoRA 里的「动作增强」是让动作幅度更大。它们不是一个东西，可以同时开。`;
  }
  if(ST.step===84){
    const V=ST.videoTest;
    $('#app').innerHTML=`<h2>视频测试 ⑤ 提示词</h2><div class="box">一行就是一条测试内容，所有模型都会用同一行。用图片做视频时，主要写“怎么动”；只写文字时，要把“谁、在哪里、做什么”一起写清楚。</div><textarea id="vtprompts" style="height:180px" placeholder="她慢慢转身看向镜头，头发和衣角随风摆动，现场有轻微脚步声和呼吸声">${esc(V.prompts)}</textarea><div class="small">常用动作：slowly turns around（慢慢转身）、walks forward（向前走）、runs（奔跑）、looks at the camera（看向镜头）。动漫画面可加：2D anime、cel shading、flat colors、non-photorealistic；不想变真人可加负面词：realistic face、photorealistic、live action。</div><p><button class="back" onclick="goStep(83)">← 上一步</button><button onclick="videoTestPromptsNext()">下一步 →</button></p>`;
  }
  if(ST.step===85){
    const V=ST.videoTest; const prompts=V.prompts.split('\n').filter(x=>x.trim()); const refs=V.mode==='t2v'?1:V.refs.length; const total=V.models.length*prompts.length*refs*V.variants.length*V.copies;
    const sourceParts=V.models.reduce((sum,m)=>sum+V.variants.reduce((n,v)=>n+Math.max(1,Math.ceil(v.duration/6)),0),0)*prompts.length*refs*V.copies;
    const estimateSec=videoEstimateSeconds(V.models,V.variants,prompts.length,refs,V.copies);
    $('#app').innerHTML=`<h2>视频测试 ⑥ 确认</h2><div class="card" style="cursor:default"><b>共 ${total} 条最终成片</b><div class="small">后台预计拆成约 <b>${sourceParts}</b> 个安全分段逐个生成，再自动拼接。</div><div class="small">按模型登记速度估算：约 <b>${videoFmtTime(estimateSec)}</b>（实际会受分辨率、显存、当前电脑负载影响）</div><div class="small">${V.models.length}模型 × ${prompts.length}提示词 × ${V.mode==='t2v'?'纯文字':V.refs.length+'张垫图'} × ${V.variants.length}组参数 × 每组${V.copies}条</div><div class="small" style="margin-top:5px">执行顺序：${V.models.map(m=>esc(m.name)).join(' → ')}</div><div class="small">成片参数：${V.variants.map(v=>`${v.w}×${v.h}/${v.duration}秒/${v.fps}fps`).join('、')}</div></div><div class="box">每个分段都有超时保护；超时会自动缩短再试。分段即时落盘，最终成片另存，关闭网页不影响后台。</div><p><button class="back" onclick="goStep(84)">← 上一步</button><button onclick="videoTestGo()">🚀 启动视频测试</button></p>`;
    const nativeConfirm=$('#app .card');
    if(nativeConfirm) nativeConfirm.insertAdjacentHTML('beforeend', `<div class="small">LTX 原生声音：<b>${V.native_audio?'开':'关'}</b></div>`);
  }
  if(ST.step===86){
    const d=await j('/api/video-test/state'); const s=d.status||{}; const pct=s.total?Math.round((s.done||0)/s.total*100):0;
    const current=(d.tasks||[]).find(t=>String(t.id)===String(s.task))||{}; const tm=current.timing||{};
    const timingLine=tm.average_sec?`<div class="small" style="margin-top:5px">本次实际平均约 <b>${videoFmtTime(tm.average_sec)}</b>/条 · 已运行 ${videoFmtTime(tm.elapsed_sec)} · 预计还需 ${videoFmtTime(tm.remaining_sec)}</div>`:`<div class="small" style="margin-top:5px">完成第一条成片后，页面会计算这台电脑的实际平均速度。</div>`;
    $('#app').innerHTML=`<div class="vt-page"><h2>后台生视频${d.worker_alive?'中…':'(已停)'}</h2><div class="card" style="cursor:default"><b style="font-size:22px">${s.done||0}/${s.total||0}</b> 条成片 · <span style="color:var(--on)">✓${s.ok||0}</span> · <span style="color:#e0556b">✗${s.fail||0}</span><div class="small">${esc(s.msg||'')}</div>${s.segment_elapsed!=null?`<div class="small">本段已计算 ${s.segment_elapsed} 秒 · 超时保护剩余 ${s.watchdog_left||0} 秒 · ComfyUI:${esc(s.comfy_state||'running')}</div>`:''}${timingLine}${s.folder?`<div class="small">📁 ${esc(s.folder)}</div>`:''}<div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div><div class="row" style="margin-top:8px">${s.state==='paused'?`<button class="back" onclick="videoTestCtl('resume')">▶ 继续</button>`:`<button class="back" onclick="videoTestCtl('pause')">⏸ 暂停</button>`}<button class="back" onclick="videoTestCtl('kill_curr')">⏹ 杀当前任务</button><button class="back" onclick="videoTestCtl('kill_all')">🛑 杀全部</button></div></div><div class="card" style="cursor:default"><div class="small" style="white-space:pre-wrap;font-family:monospace">${(s.log||[]).slice(-12).map(esc).join('\n')}</div></div><p><button class="back" onclick="goStep(89)">← 视频任务</button><button class="back" onclick="render()">🔄 刷新</button></p></div>`;
    if(d.worker_alive) setTimeout(()=>{if(ST.step===86)render()},2500);
  }
  if(ST.step===87){
    const r=await j('/api/video-test/prompts?f='+encodeURIComponent(ST.videoTestEditId||''));
    $('#app').innerHTML=`<div class="vt-page"><h2>✏️ 视频任务 ${esc(ST.videoTestEditId||'')} 提示词</h2><div class="box">一行一条。保存后，尚未开始的下一段视频会读取新内容。</div><textarea id="vtpe" style="height:200px">${esc(r.text||'')}</textarea><p><button class="back" onclick="goStep(ST.videoReturnStep||89)">← 视频任务</button><button onclick="videoTestSavePrompts()">💾 保存</button></p></div>`;
  }
}
function newItem(){ return {mode:'t2i',w:1024,h:576,pos:'',neg:NEG_DEF,defPos:'masterpiece, best quality',strength:0.6,scale:2,ctype:'openpose',refFile:null,strokes:[],brush:30,poseSrc:'draw',joints:null,resultUrl:null}; }
// ---- 视频 I2V 交互 ----
function vSet(k,val,rerender){ ST.vid[k]=val; if(rerender) render(); }
function vToggleLora(id,on){ const v=ST.vid; if(!Array.isArray(v.loras))v.loras=[]; const i=v.loras.indexOf(id); if(on&&i<0)v.loras.push(id); if(!on&&i>=0)v.loras.splice(i,1); v.lora=v.loras[0]||'none'; }
function vSetStyle2d(on){ const v=ST.vid; v.style2d=!!on; if(on&&!v.loras.includes('anime'))v.loras.unshift('anime'); if(!on){ const i=v.loras.indexOf('anime'); if(i>=0)v.loras.splice(i,1); } v.lora=v.loras[0]||'none'; render(); }
function vSetTextOnly(on){ ST.vid.textOnly=!!on; render(); }
function vPickModel(id){
  ST.vid.unet=id;
  document.querySelectorAll('[data-vmodel]').forEach(el=>el.classList.toggle('sel',el.dataset.vmodel===id));
}
function vRes(s){
  if(s==='custom'){
    ST.vid.aspect='custom';
    ST.vid.customRes=true;
    const box=$('#vcustom'); if(box) box.style.display='inline-flex';
    return;
  }
  const[a,b]=s.split('x'); ST.vid.aspect=({'360x640':'9:16','640x360':'16:9','512x512':'1:1','432x544':'4:5','480x640':'3:4','640x480':'4:3'})[s]||s;
  ST.vid.customRes=false; ST.vid.w=+a; ST.vid.h=+b;
  const box=$('#vcustom'); if(box) box.style.display='none';
  const note=$('#vAspectNote'); if(note) note.textContent=`当前 ${ST.vid.aspect} · ${ST.vid.w}×${ST.vid.h}`;
}
function vCustomSize(){
  const w=Math.max(64,Math.round(+($('#vwidth')||{}).value||ST.vid.w));
  const h=Math.max(64,Math.round(+($('#vheight')||{}).value||ST.vid.h));
  ST.vid.w=Math.round(w/16)*16; ST.vid.h=Math.round(h/16)*16; ST.vid.customRes=true; ST.vid.aspect='custom';
  if($('#vwidth')) $('#vwidth').value=ST.vid.w; if($('#vheight')) $('#vheight').value=ST.vid.h;
  const note=$('#vAspectNote'); if(note) note.textContent=`当前自定义 · ${ST.vid.w}×${ST.vid.h}`;
}
function vFrameFor(sec,fps){
  const raw=Math.max(9,Math.round(+sec*+fps));
  return Math.max(9,Math.round((raw-1)/8)*8+1);
}
function vSyncFrames(){ ST.vid.frames=vFrameFor(ST.vid.duration,ST.vid.fps); const n=$('#vfnum'); if(n) n.textContent=ST.vid.frames; }
function vDuration(n){ ST.vid.duration=+n; vSyncFrames(); }
function vFps(n){ ST.vid.fps=+n; vSyncFrames(); }
function vFrames(n){ n=Math.round(n); ST.vid.frames=n; ST.vid.duration=Math.max(0.1,n/(ST.vid.fps||24)); const d=document.querySelector('select[onchange="vDuration(this.value)"]'); if(d) d.value=''; const x=$('#vfnum'); if(x) x.textContent=n; }
function vSetFile(f){ if(!f) return; ST.vid.file=f; render(); }
function vAudioMode(mode){ ST.vid.audioMode=mode; const el=$('#vaudiofile'); if(el) el.style.display=mode==='upload'?'block':'none'; const n=$('#vaudioname'); if(n)n.textContent=mode==='native'?'LTX 会按动作和场景同时生成现场声音。':(mode==='upload'?(ST.vid.audioFile?'已选择: '+ST.vid.audioFile.name:'上传音频会在成片后混入。'):'不生成声音。'); }
function vSetAudio(f){ if(!f) return; ST.vid.audioFile=f; const n=$('#vaudioname'); if(n) n.textContent='已选择: '+f.name; }
function vDrop(e){ e.preventDefault(); e.currentTarget.classList.remove('over'); const f=e.dataTransfer.files[0]; if(f) vSetFile(f); }
async function vStart(){
  const v=ST.vid; v.pos=$('#vpos').value;
  if(!v.textOnly&&!v.file){ cAlert('请先上传一张源图，或勾选“纯文字生成”'); return; }
  if(v.audioMode==='upload' && !v.audioFile){ cAlert('你选择了添加音频,请先选择一个音频文件'); return; }
  const sourceFps=Math.min(24,+v.fps||24), sourceFrames=vFrameFor(v.duration,sourceFps);
  $('#app').innerHTML = `<div class="box">📤 上传源图并提交…</div>`;
  try{
    let ud={name:''};
    if(v.file && !v.textOnly){
      const fd=new FormData(); fd.append('image', v.file, v.file.name); fd.append('overwrite','true');ud=await (await fetch('/api/vid/upload',{method:'POST',body:fd})).json();
      if(ud.error) throw new Error(ud.error);
    }
    let audioName='';
    if(v.audioMode==='upload' && v.audioFile){
      const au=await (await fetch('/api/vid/audio?fname='+encodeURIComponent(v.audioFile.name),{method:'POST',body:v.audioFile})).json();
      if(au.error) throw new Error(au.error); audioName=au.name||'';
    }
    const d=await (await fetch('/api/vid/gen',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:'vid_'+Date.now(),unet:v.unet,pos:v.pos,neg:v.neg,image:v.textOnly?'':ud.name,
        w:v.w,h:v.h,duration:v.duration,frames:sourceFrames,fps:sourceFps,lora:(v.loras[0]||'none'),loras:v.loras,lora_strength:0.8,stg:v.stg,steps:8,
        interpolate:!!v.interpolate,interpolate_fps:(+v.fps>sourceFps?+v.fps:(v.interpolate?60:0)),audio:audioName,native_audio:v.audioMode==='native',style_2d:!!v.style2d})})).json();
    if(d.error) throw new Error(d.error);
    v.pid=d.pid; ST.step=11; render();
  }catch(e){ $('#app').innerHTML = `<div class="box">提交失败:${esc(e.message)}</div><p><button onclick="ST.step=10;render()">← 返回</button></p>`; }
}
async function vCancel(){
  const v=ST.vid;if(!v.pid)return;
  const r=await (await fetch('/api/vid/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pid:v.pid})})).json();
  if(r.error)cAlert('停止失败：'+r.error);else cAlert('停止指令已发送');
}
async function vPoll(){
  const v=ST.vid; const t0=Date.now();
  const t=setInterval(async()=>{
    if(ST.step!==11){ clearInterval(t); return; }
    let d; try{ d=await j('/api/vid/poll?pid='+v.pid); }catch(e){ return; }
    const el=Math.round(d.run_elapsed!=null?d.run_elapsed:(Date.now()-t0)/1000); if($('#vel')) $('#vel').textContent=fmtTime(el);
    if($('#vbar')) $('#vbar').style.width=Math.min(95,el/1.5)+'%';
    if(d.error){ clearInterval(t); $('#vout').innerHTML=`<div class="box">生成失败:${esc(d.error)}</div>`; $('#vacts').style.display='block'; return; }
    if(d.done){ clearInterval(t); if($('#vbar')) $('#vbar').style.width='100%';
      $('#vout').innerHTML=`<video controls autoplay loop src="${d.url}" style="max-width:100%;border-radius:10px"></video>`;
      $('#vacts').style.display='block'; }
  },2500);
}
// ---- 语言模型交互 ----
function llmPick(id){ ST.llmModel=(ST._llmModels||[]).find(m=>m.id===id); ST.step=21; render(); }
function syncProxyMode(v){ const n=Math.max(0,Math.min(2,Math.round(+v||0))); const input=$('#lcpx'); if(input) input.value=n; document.querySelectorAll('[data-proxy-mode]').forEach(btn=>btn.classList.toggle('active',+btn.dataset.proxyMode===n)); const el=$('#lcpxMode'); if(el) el.textContent='当前：'+['关闭','Codex 代理','Claude 代理'][n]; }
function selectProxyMode(v){ syncProxyMode(v); }
async function llmStart(){
  const m=ST.llmModel;
  const thinking=m.is_reasoning?($('#lthink')?$('#lthink').checked:!!m.prefs.thinking):false;
  const temp=+($('#ltemp')?$('#ltemp').value:m.prefs.temp);
  const max_tokens=Math.min(65536,Math.max(512,Math.round(+($('#lmax')?$('#lmax').value:m.prefs.max_tokens)||16384)));
  const mtp=(m.backend==='metal'||!m.has_mtp)?false:(($('#lmtp')?$('#lmtp').checked:!!m.prefs.mtp));
  const proxy_mode=(m.backend==='metal')?0:(($('#lcpx')?Math.max(0,Math.min(2,Math.round(+$('#lcpx').value||0))):Number.isInteger(+m.prefs.proxy_mode)?+m.prefs.proxy_mode:(m.prefs.codex_proxy?1:0)));
  const codex_proxy=proxy_mode===1;
  const csel=$('#lctx')?$('#lctx').value:String(m.prefs.ctx||32768);
  const ccust=csel==='custom';
  const ctx=ccust
    ? Math.min(131072,Math.max(8192,Math.round(+($('#lctxcust')?$('#lctxcust').value:0)||32768)))
    : csel;
  const psel=$('#lparallel')?$('#lparallel').value:String(m.prefs.parallel||1);
  const parallel=psel==='custom'
    ? Math.min(8,Math.max(1,Math.round(+($('#lparallelcust')?$('#lparallelcust').value:1)||1)))
    : Math.min(8,Math.max(1,parseInt(psel,10)||1));
  const bv=$('#lbudget')?$('#lbudget').value:'';
  const budget=m.is_reasoning
    ? (($('#lbudc')&&$('#lbudc').checked)
        ? (bv===''?-1:Math.max(-1,Math.round(+bv)))
        : +($('#lbud')?$('#lbud').value:(m.prefs.budget??-1)))
    : -1;
  const reasoning_level=m.is_reasoning?($('#llevel')?$('#llevel').value:(m.prefs.reasoning_level||'high')):'high';
  $('#app').innerHTML = `<div class="box">🚀 正在启动 ${esc(m.name)}…<br><span class="small">会先停掉其他服务腾内存;若 GPU 上限不足,会弹一次 macOS 密码框。</span><div class="bar" style="margin-top:12px"><i class="indet"></i></div></div>`;
  const d=await (await fetch('/api/llm/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:m.id,thinking,temp,max_tokens,mtp,codex_proxy,proxy_mode,ctx,parallel,budget,reasoning_level})})).json();
  if(d.error){ let h=`<div class="box">启动失败:${esc(d.error)}</div>`;
    if(d.manual) h+=`<div class="box">可在终端手动运行(<code>!</code>前缀)后重试:<br><code>${esc(d.manual)}</code></div>`;
    h+=`<p><button onclick="goStep(21)">← 返回</button></p>`; $('#app').innerHTML=h; return; }
  ST.step=22; render();
}
async function llmMetalInit(){
  $('#app').innerHTML=`<div class="box">正在初始化官方 Metal 环境…<br><span class="small">会创建 metal/.venv；如果已放入官方源码，会安装 gpt-oss[metal] 依赖。不会下载 model.bin。</span><div class="bar" style="margin-top:12px"><i class="indet"></i></div></div>`;
  try{
    const d=await j('/api/llm/metal/init',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    let h=`<div class="box">${esc(d.message||d.error||'Metal 初始化完成')}</div>`;
    if(d.output) h+=`<div class="box"><pre style="white-space:pre-wrap;max-height:220px;overflow:auto">${esc(d.output)}</pre></div>`;
    h+=`<p><button onclick="goStep(20)">返回语言模型</button></p>`;
    $('#app').innerHTML=h;
  }catch(e){ $('#app').innerHTML=`<div class="box">Metal 初始化失败:${esc(e)}</div><p><button onclick="goStep(20)">返回</button></p>`; }
}
async function llmStop(next){
  $('#app').innerHTML = `<div class="box"><span class="spin"></span> 正在停止语言模型…</div>`;
  ST.llmPaused=false;
  await j('/api/svc/stop?type=llm');
  if(next===0) home(); else goStep(next);
}
function fmtTime(s){ s=Math.max(0,Math.floor(s)); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return (h?h+'小时':'')+(h||m?m+'分':'')+x+'秒'; }
function videoFmtTime(s){
  s=Number(s)||0;
  if(s<=0) return '暂时无法估算';
  return fmtTime(s);
}
function videoEstimateSeconds(models,variants,promptCount,refCount,copies){
  return (models||[]).reduce((sum,m)=>sum+(variants||[]).reduce((part,v)=>{
    const limit=6;
    return part+Math.ceil(Number(v.duration||2)/limit)*(Number(m.sec)||300);
  },0),0)*Math.max(1,promptCount||1)*Math.max(1,refCount||1)*Math.max(1,copies||1);
}
function fmtAct(a){ // "它正在干嘛"实时状态: 空闲/消化输入(带进度)/生成回复(带速度)
  if(!a) return '…';
  if(a.state==='idle') return '💤 空闲 · 等指令'+(a.tps?`(上次生成 ${a.tps.toFixed(1)} tok/s)`:'');
  if(a.state==='prompt') return `📥 读取输入中 ${Math.round((a.progress||0)*100)}% · 读取 ${(a.tps||0).toFixed(0)} tok/s`;
  if(a.state==='gen') return `✍️ 写入回复中… 已写 ${a.n||0} token · 写入 ${(a.tps||0).toFixed(1)} tok/s`+(a.read_tps?` · 读取 ${a.read_tps.toFixed(0)} tok/s`:'');
  return '…'; }
function llmCopyApi(u){ if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(u).then(()=>cAlert('已复制: '+u),()=>cAlert(u)); } else cAlert(u); }
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
async function llmPause(){ ST.llmPaused=true; await j('/api/llm/pause',{method:'POST'}); render(); }
async function llmResume(){
  $('#app').innerHTML=`<div class="box">▶ 正在恢复语言模型…</div>`;
  ST.llmPaused=false;
  await (await fetch('/api/llm/resume',{method:'POST'})).json();
  ST.step=22; render();
}
// ---- 控制台聊天(语言模型 + 说「画xx」自动生图) ----
function isDrawReq(t){
  return /^\s*(帮我|给我|麻烦|请|帮忙)?\s*(画|绘|生图|出图)/.test(t)
      || /(画|绘)\s*(一)?\s*(张|只|个|幅|条|朵|块|本)/.test(t)
      || /(生成|出)\s*(一)?\s*(张|个|份)?\s*(图|照片|图片|插画)/.test(t)
      || /^\s*(draw|paint|sketch|generate\s+(an?\s+)?(image|picture|photo))/i.test(t);
}
function resumeBtn(){ return `<div style="margin-top:9px"><button class="back" onclick="chatResume()">▶ 恢复语言模型继续聊</button></div>`; }
async function chatResume(){ ST.chat.llmPaused=false; await j('/api/llm/resume',{method:'POST'}); render(); }
async function chatSend(){
  const C=ST.chat; if(!C||C.busy) return;
  const ta=$('#chatin'); const text=(ta.value||'').trim(); if(!text) return;
  ta.value='';
  C.msgs.push({role:'user',text});
  if(isDrawReq(text)){ C.busy=true; render(); await chatDraw(text); C.busy=false; render();
    setTimeout(()=>{const t=$('#chatin'); if(t) t.focus();},60); return; }
  // 普通聊天需要语言模型在线
  const st=await j('/api/llm/stats');
  if(!st.model || st.paused || st.loading){
    C.msgs.push({role:'assistant',text: st.paused?'⏸ 语言模型已暂停(内存让给了生图)。点下面按钮恢复,或仍可说「画xx」直接生图。':'⏳ 语言模型加载中,稍等几秒再发…',
      html: st.paused?resumeBtn():''});
    render(); return;
  }
  C.busy=true;
  C.msgs.push({role:'assistant',text:'✍️ 思考中…',draft:true}); render();
  const hist=C.msgs.filter(m=>!m.draft).map(m=>({role:m.role,content:m.img?(m.text+'(已生成配图)'):m.text}));
  try{
    const r=await j('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:hist})});
    C.msgs=C.msgs.filter(m=>!m.draft);
    if(r.error) C.msgs.push({role:'assistant',text:'⚠ 调用失败:'+r.error});
    else C.msgs.push({role:'assistant',text:r.reply||'(空回复)',think:r.think||''});
  }catch(e){ C.msgs=C.msgs.filter(m=>!m.draft); C.msgs.push({role:'assistant',text:'⚠ 请求出错:'+e}); }
  C.busy=false; render();
  setTimeout(()=>{const t=$('#chatin'); if(t) t.focus();},60);
}
async function chatDraw(text){
  const C=ST.chat;
  C.msgs.push({role:'assistant',text:'🎨 收到生图请求,准备中…'});
  render();
  const last=()=>C.msgs[C.msgs.length-1];
  const upd=(t,html)=>{ const m=last(); m.text=t; if(html!==undefined)m.html=html; render(); };
  // 1. 提示词: 语言模型在就翻成英文,不在就剥掉触发词用原文
  let prompt='';
  const st0=await j('/api/llm/stats');
  const llmOn = st0.model && !st0.paused && !st0.loading;
  if(llmOn){
    upd('🎨 请语言模型把描述翻成英文提示词…');
    try{
      const tr=await j('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({max_tokens:300,temperature:0.3,messages:[
          {role:'system',content:'You are an SDXL prompt translator. Rewrite the user scene description into a single English text-to-image prompt: comma-separated tag phrases, you may add quality words. Output ONLY the prompt itself, no explanation, no quotes, no newline.'},
          {role:'user',content:text}]})});
      prompt=((tr.reply||'').replace(/\n+/g,' ')).trim();
    }catch(e){ prompt=''; }
  }
  if(!prompt){
    prompt=text.replace(/^\s*(帮我|给我|麻烦|请|帮忙)?\s*(画|绘|生图|出图|生成)\s*(一)?\s*(张|只|个|幅|条|朵|块|本|份)?\s*/,'').trim()||text;
  }
  // 2. 生图服务没在跑 → 暂停语言模型腾内存(32G 同时只能跑一类)
  let paused=false;
  let st=await j('/api/svc/state');
  if(!st.img.running){
    const ok=await cConfirm('生图服务没在跑。这台 32G 内存同一时刻只能跑一类模型:先「暂停语言模型」去生图,生完图卡片上可一键恢复继续聊。现在去生图?','去生图');
    if(!ok){ upd('🎨 已取消生图,继续聊吧。',''); return; }
    upd('⏸ 正在暂停语言模型…');
    await j('/api/llm/pause',{method:'POST'});
    paused=true;
    upd('🚀 正在启动生图服务(加载模型约 1~2 分钟)…');
    await j('/api/svc/start?type=img');
    for(let n=0;n<100;n++){
      await new Promise(r=>setTimeout(r,3000));
      st=await j('/api/svc/state');
      if(st.img.running) break;
      if(!st.img.alive_pid && n>3){ upd('⚠ 生图服务启动失败,请看 comfy.log。', resumeBtn()); return; }
    }
    if(!st.img.running){ upd('⚠ 生图服务启动超时。', resumeBtn()); return; }
  }
  // 3. 提交生图
  upd('🎨 提交生图:「'+prompt+'」');
  const wh=C.size.split('x');
  const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:C.genModel,pos:prompt,neg:NEG_DEF,w:+wh[0],h:+wh[1],name:'chat_'+Date.now()})});
  if(r.error){ upd('⚠ 提交失败:'+r.error, paused?resumeBtn():''); return; }
  // 4. 轮询出图
  for(let n=0;n<600;n++){
    await new Promise(r2=>setTimeout(r2,2000));
    let p; try{ p=await j('/api/poll?pid='+r.pid); }catch(e){ continue; }
    if(p.done){ const m=last(); m.text='✅ 画好了:'+prompt; m.img=p.url; m.html=paused?resumeBtn():''; render(); return; }
    if(p.error){ upd('⚠ 生图失败:'+p.error, paused?resumeBtn():''); return; }
    if(n%6===0) upd('🎨 生成中… '+Math.round(p.run_elapsed||0)+'s「'+prompt.slice(0,50)+'…」');
  }
  upd('⚠ 生图超时(20分钟)。', paused?resumeBtn():'');
}
// ---- 语音模型交互 ----
async function ttsStart(){
  $('#app').innerHTML=`<div class="box"><span class="spin"></span> 正在启动语音服务…</div>`;
  const r=await j('/api/tts/start');
  if(r.error){ $('#app').innerHTML=`<div class="box">启动失败:${esc(r.error)}</div><p><button onclick="goStep(60)">← 返回</button></p>`; return; }
  render();
}
async function ttsStop(){ await j('/api/tts/stop'); goStep(60); }
async function ttsLearn(){
  const f=$('#ttsfile').files[0];
  const name=$('#ttsname').value.trim(), text=$('#ttstext').value.trim();
  if(!f){ cAlert('先选一个音频/视频文件'); return; }
  if(!name||!text){ cAlert('名字和参考文本都要填'); return; }
  $('#app').innerHTML=`<div class="box"><span class="spin"></span> 正在学习「${esc(name)}」并生成试听,约 10~20 秒…</div>`;
  const r=await (await fetch('/api/tts/learn?name='+encodeURIComponent(name)+'&text='+encodeURIComponent(text)+'&fname='+encodeURIComponent(f.name),{method:'POST',body:f})).json();
  if(r.error){ cAlert('学习失败:'+r.error); }
  else if(r.sample_error){ cAlert('声音已存,但试听生成失败(可在声音库点"生成试听"重试):'+r.sample_error); }
  render();
}
async function ttsDel(name){
  if(!await cConfirm('删除这个声音?','删除')) return;
  await (await fetch('/api/tts/delvoice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:decodeURIComponent(name)})})).json();
  render();
}
async function ttsSample(name, btn){
  const nm=decodeURIComponent(name);
  const inp=document.getElementById('smptext-'+name);
  const text=inp?inp.value.trim():'';
  if(btn){ btn.disabled=true; btn.textContent='⏳ 生成中…'; }
  const r=await (await fetch('/api/tts/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:nm,text})})).json();
  if(r.error){ cAlert('试听生成失败:'+r.error); if(btn){ btn.disabled=false; btn.textContent='🔊 试听'; } return; }
  render();
}
async function ttsRename(name){
  const nm=decodeURIComponent(name);
  const nn=await cPrompt('给「'+nm+'」换个名字:', nm);
  if(nn===null||!nn.trim()) return;
  const r=await (await fetch('/api/tts/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old:nm,new:nn.trim()})})).json();
  if(r.error){ cAlert('重命名失败:'+r.error); }
  render();
}
async function ttsSpeak(){
  const text=$('#ttssay').value.trim();
  const voice=(document.querySelector('input[name=ttsv]:checked')||{}).value||'';
  if(!text){ cAlert('台词不能为空'); return; }
  const box=$('#ttsresult');
  box.innerHTML=`<div class="card" style="cursor:default"><b>🔊 正在合成…</b> <span class="small" id="ttspt">0%</span>
    <div class="bar"><i id="ttspb"></i></div></div>`;
  // 真进度: 每 400ms 轮询一次(生成是逐帧的,服务器报真实帧数)
  let done=false;
  const timer=setInterval(async()=>{
    if(done) return;
    try{ const p=await j('/api/tts/progress');
      const pct = p.max? Math.min(99, Math.round(p.step/p.max*100)) : 0;
      const b=$('#ttspb'), t=$('#ttspt');
      if(b) b.style.width=pct+'%'; if(t) t.textContent=pct+'% ('+p.step+' 帧)';
    }catch(e){}
  },400);
  let r;
  try{
    r=await (await fetch('/api/tts/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,voice})})).json();
  }catch(e){
    r={error:'语音服务没响应: '+e};
  }finally{
    done=true; clearInterval(timer);   // 必须清掉,否则每报错一次就永久多一个 400ms 轮询(越用越卡的元凶)
  }
  if(r.error){ box.innerHTML=`<div class="box" style="color:#d03050">合成失败:${esc(r.error)}</div>`; return; }
  const url='/tts_out/'+encodeURIComponent(r.wav);
  box.innerHTML=`<div class="card" style="cursor:default"><b>✅ 合成完成</b>
    <p><audio controls autoplay src="${url}" style="width:100%"></audio></p>
    <p><a href="${url}" download="${esc(r.wav)}"><button class="back">⬇ 下载 wav</button></a></p></div>`;
}
// ---- 分段配音工作台(step 61) ----
async function wbPost(path,obj){
  try{ return await (await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)})).json(); }
  catch(e){ return {error:'请求失败: '+e}; }
}
function wbStatusHtml(s){
  if(s.status==='run'){
    const cp=s.frame_max?(s.frame/s.frame_max):0;
    const pct=Math.min(99,Math.round(((s.chunks_done+cp)/(s.chunk_count||1))*100));
    return `<div class="bar"><i style="width:${pct}%"></i></div><div class="small">🔊 合成中 第${s.chunks_done+1}/${s.chunk_count}小段 · ${s.frame||0}帧</div>`;
  }
  if(s.status==='wait') return '<div class="small">⏳ 排队中…</div>';
  if(s.status==='err') return `<div class="small" style="color:#d03050">✗ 失败:${esc(s.err||'合成失败')}</div>`;
  return `<audio controls preload="none" src="/tts_out/${encodeURIComponent(s.wav)}" style="width:100%"></audio>`;
}
function wbMaybePoll(busy){ // 只更新每段状态，不整页重画，避免屏幕和播放器反复闪烁
  if(!busy) return;
  setTimeout(async()=>{
    if(ST.step!==61) return;
    let wb; try{ wb=await j('/api/wb/state'); }catch(e){ wbMaybePoll(true); return; }
    if(ST.step!==61) return;
    for(const s of (wb.segs||[])){
      const box=$('#wbstatus-'+s.id);
      if(box) box.innerHTML=wbStatusHtml(s);
    }
    const stillBusy=(wb.segs||[]).some(s=>s.status==='wait'||s.status==='run');
    const allDone=(wb.segs||[]).length>0&&(wb.segs||[]).every(s=>s.status==='done');
    const label=$('#wbBusyLabel'); if(label) label.textContent=stillBusy?'· 后台合成中…':'';
    const merge=$('#wbMergeBtn'); if(merge) merge.disabled=!allDone;
    wbMaybePoll(stillBusy);
  },1500);
}
async function wbAdd(){
  const text=$('#wbnew').value.trim();
  const voice=$('#wbnewv').value;
  if(!text){ cAlert('台词不能为空'); return; }
  const r=await wbPost('/api/wb/add',{text,voice});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbSave(id){
  const text=$('#wbt-'+id).value.trim();
  const r=await wbPost('/api/wb/edit',{id,text});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbVoice(id,val){ const r=await wbPost('/api/wb/edit',{id,voice:val}); if(r.error){cAlert(r.error);} render(); }
async function wbDel(id){
  if(!await cConfirm('删掉这一段 #'+id+' ?','删除')) return;
  const r=await wbPost('/api/wb/del',{id});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbRegen(id){ const r=await wbPost('/api/wb/regen',{id}); if(r.error){cAlert(r.error); return;} render(); }
async function wbSplit(id){
  const ta=$('#wbt-'+id);
  const a=ta.selectionStart, b=ta.selectionEnd;
  if(a===b){ cAlert('先在台词框里用鼠标选中要重录的那几个字,再点✂️'); return; }
  const r=await wbPost('/api/wb/split',{id,start:a,end:b});
  if(r.error){ cAlert(r.error); return; }
  render();
}
function wbDrag(e,id){ e.dataTransfer.setData('text/wbsid', String(id)); e.dataTransfer.effectAllowed='move'; }
async function wbDrop(e,targetId){
  e.preventDefault();
  const dragId=+e.dataTransfer.getData('text/wbsid');
  if(!dragId||dragId===targetId) return;
  const ids=[...document.querySelectorAll('.wbcard')].map(x=>+x.dataset.sid);
  const from=ids.indexOf(dragId), to=ids.indexOf(targetId);
  if(from<0||to<0) return;
  ids.splice(to,0,ids.splice(from,1)[0]);   // 移到目标那一格的位置
  const r=await wbPost('/api/wb/order',{ids});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbMove(id,dir){
  const ids=[...document.querySelectorAll('.wbcard')].map(x=>+x.dataset.sid);
  const i=ids.indexOf(id), k=i+dir;
  if(i<0||k<0||k>=ids.length) return;
  const t=ids[i]; ids[i]=ids[k]; ids[k]=t;
  const r=await wbPost('/api/wb/order',{ids});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbMerge(){
  if(!await cConfirm('合并后顺序锁定、不能再改台词和音色了。确定合并?','合并')) return;
  const r=await wbPost('/api/wb/merge',{});
  if(r.error){ cAlert(r.error); return; }
  render();
}
async function wbReset(){
  if(!await cConfirm('清空整个工作台(所有段落和合并结果都从列表移除),重新开始?','清空')) return;
  const r=await wbPost('/api/wb/reset',{});
  if(r.error){ cAlert(r.error); return; }
  render();
}
// ---- 预定批量交互 ----
function tabHtml(active){ // 图片区顶部面包屑切换: 单模型 / 预定批量 / 漫画连载
  return `<div class="tabs">
    <span class="tab ${active===0?'on':''}" onclick="goStep(1)">🖼️ 单模型生成</span>
    <span class="tab ${active===3?'on':''}" onclick="hqOpen()">✨ 高质量图片</span>
    <span class="tab ${active===1?'on':''}" onclick="goStep(30)">📦 预定批量</span>
    <span class="tab ${active===2?'on':''}" onclick="goStep(40)">📖 漫画连载</span></div>`;
}
// ---- 预定批量: 模型大卡 → 风格子卡 → 提示词组 ----
function bModelCard(m){ // 一个模型一张大卡(用户「＋ 新增模型」才出现),内含风格子卡容器;卡片顺序=执行顺序
  return `<div class="card bmodel" data-id="${m.id}" data-name="${esc(m.name)}" data-sec="${m.sec||60}" style="cursor:default;padding:14px 16px">
    <div class="row" style="justify-content:space-between;align-items:center">
      <b style="font-size:16px">🧠 ${esc(m.name)}</b> <span class="small mtime ${secCls(m.sec)}">${m.time||('约'+(m.sec||60)+'秒')}/张</span>
      <span><button class="back" style="padding:6px 8px;font-size:12px" title="提前执行" onclick="bMoveModel(this,-1)">↑</button>
      <button class="back" style="padding:6px 8px;font-size:12px" title="延后执行" onclick="bMoveModel(this,1)">↓</button>
      <button class="back" style="padding:6px 10px;font-size:12px" onclick="bDelModel(this)">🗑 删除模型</button></span></div>
    <div class="bstyles" style="margin-top:10px">${bStyleCard({name:'普通',prompts:[]})}</div>
    <button class="back" style="margin-top:8px;padding:6px 14px;font-size:13px" onclick="bAddStyle(this)">＋ 新增风格</button></div>`;
}
function bAddModel(){ // 把下拉选中的模型加成一张大卡;默认按速度插位(生图快的排前面,可点卡片↑↓手动调)
  const sel=$('#bpick'); const id=sel?sel.value:'';
  const m=(ST.bmodels||[]).find(x=>x.id===id);
  if(!m){ cAlert('先在上面选一个模型'); return; }
  const sec=+m.sec||60;
  const before=[...document.querySelectorAll('#bmodels .bmodel')].find(c=>(+c.dataset.sec||60)>sec);
  if(before) before.insertAdjacentHTML('beforebegin', bModelCard(m));
  else $('#bmodels').insertAdjacentHTML('beforeend', bModelCard(m));
  bRefreshPicker();
}
function bMoveModel(btn,dir){ // 手动调执行顺序:卡片上下移动,提交时按 DOM 顺序逐个跑
  const c=btn.closest('.bmodel'); if(!c) return;
  const sib=dir<0?c.previousElementSibling:c.nextElementSibling;
  if(sib&&sib.classList.contains('bmodel')){
    if(dir<0) c.parentNode.insertBefore(c,sib); else c.parentNode.insertBefore(sib,c);
  }
}
function bDelModel(btn){ btn.closest('.bmodel').remove(); bRefreshPicker(); }
function bRefreshPicker(){ // 下拉里去掉已添加的模型,加光后提示
  const sel=$('#bpick'); if(!sel) return;
  const added=[...document.querySelectorAll('.bmodel')].map(x=>x.dataset.id);
  const opts=(ST.bmodels||[]).filter(m=>!added.includes(m.id));
  sel.innerHTML = opts.length ? opts.map(m=>`<option value="${m.id}">${esc(m.name)}</option>`).join('')
                              : '<option value="">(全部模型都已添加)</option>';
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
  if(wrap.querySelectorAll('.bstyle').length<=1){ cAlert('每个模型至少留一个风格'); return; }
  card.remove();
}
async function bUpRef(input,f){ // 给某个风格传一张垫图(i2i 0.75)
  if(!f) return;
  const fd=new FormData(); fd.append('image',f,f.name); fd.append('overwrite','true');
  const ud=await (await fetch('/api/upload',{method:'POST',body:fd})).json();
  if(ud.error||!ud.name){ cAlert('垫图上传失败:'+(ud.error||'未知')); return; }
  const card=input.closest('.bstyle'); card.dataset.ref=ud.name;
  card.querySelector('.bs-refname').textContent='✔ 垫图:'+f.name;
}
async function bImport(){ // 从文件夹导入风格集,套用到所有模型大卡
  const folder=($('#bfolder').value||'').trim();
  if(!folder){ cAlert('先填文件夹路径'); return; }
  const r=await (await fetch('/api/batch/import',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({folder})})).json();
  if(r.error){ $('#bimpMsg').textContent='✗ '+r.error; return; }
  const wraps=document.querySelectorAll('.bmodel .bstyles');
  if(!wraps.length){ $('#bimpMsg').textContent='✗ 先「＋ 新增模型」再导入'; return; }
  wraps.forEach(w=>{ w.innerHTML=r.styles.map(s=>bStyleCard(s)).join(''); });
  $('#bimpMsg').textContent=`✔ 已导入 ${r.styles.length} 个风格(共 ${r.styles.reduce((a,s)=>a+s.prompts.length,0)} 条)到 ${wraps.length} 个模型`;
}
async function batchStart(){
  const size=($('#bsize').value||'1024x720').split('x');
  const models=[];
  document.querySelectorAll('.bmodel').forEach(mc=>{
    const styles=[];
    mc.querySelectorAll('.bstyle').forEach(sc=>{
      const name=sc.querySelector('.bs-name').value.trim()||'普通';
      const prompts=sc.querySelector('.bs-prompts').value.split('\n').map(x=>x.trim()).filter(Boolean);
      if(prompts.length) styles.push({name,prompts,ref:sc.dataset.ref||''});
    });
    if(styles.length) models.push({id:mc.dataset.id,name:mc.dataset.name,styles});
  });
  if(!models.length){ cAlert('先「＋ 新增模型」,并给至少一个模型写上提示词'); return; }
  const total=models.reduce((a,m)=>a+m.styles.reduce((x,s)=>x+s.prompts.length,0),0);
  if(!await cConfirm(`将生成 ${models.length} 个模型共 ${total} 张图,开始?`,'开跑')) return;
  const r=await (await fetch('/api/batch/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({models,neg:$('#bneg').value,w:+size[0],h:+size[1]})})).json();
  if(r.error){ cAlert(r.error); return; }
  ST.step=31; render();
}
async function batchStop(){ await (await fetch('/api/batch/stop',{method:'POST'})).json(); }
async function refUp(input){ // 参考图库: 一次传多张,存 refs/ 并自动选用最后一张
  const fs=[...input.files]; if(!fs.length) return;
  for(const f of fs){
    const r=await (await fetch('/api/refs/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f})).json();
    if(r.error){ cAlert('上传失败:'+r.error); return; }
    if(f===fs[fs.length-1]) await (await fetch('/api/refs/active',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:r.name||f.name})})).json();
  }
  render();
}
async function refUse(name){ // 点选某张当参考图(漫画图库;改名避开普通/批量的 refPick(i,el))
  await (await fetch('/api/refs/active',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:decodeURIComponent(name)})})).json();
  render();
}
async function refClear(){ // 不用参考图(纯文字生成)
  await (await fetch('/api/refs/active',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:''})})).json();
  render();
}
async function refDel(name){ // 删除一张参考图
  if(!await cConfirm('删除这张参考图?','删除')) return;
  await (await fetch('/api/refs/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:decodeURIComponent(name)})})).json();
  render();
}
async function comicStart(){
  const model=$('#cmodel').value;
  const panels=[...document.querySelectorAll('.cp:checked')].map(c=>{const i=c.dataset.i;
    return {num:+c.dataset.num,
      prompt:(document.querySelector('.cpp[data-i="'+i+'"]')||{}).value||'',
      dialogue:(document.querySelector('.cpd[data-i="'+i+'"]')||{}).value||''};})
    .filter(p=>p.prompt.trim());
  if(!panels.length){ cAlert('至少留一格分镜'); return; }
  const ra=await j('/api/refs');   // 当前选用的参考图(refs/ 本地名,空=纯文字)
  const r=await (await fetch('/api/comic/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model,panels,ref:ra.active||'',strength:0.75,w:832,h:1216,
      style:($('#cstyle')||{}).value||'',custom_style:($('#ccustom')||{}).value||'',
      cmode:($('#cmode')||{}).value||'i2i'})})).json();
  if(r.error){ cAlert(r.error); return; }
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
  if(ud.error||!ud.name){ cAlert('上传失败:'+(ud.error||'未知')); return; }
  ST.cc.chars[i].ref=ud.name; if($('#cup'+i)) $('#cup'+i).textContent='✔ 已传照片,按 0.6 漫改成设定图';
}
async function ccSheet(i){ // 生成/重画某角色设定图
  const c=ST.cc.chars[i]; const d=$('#cdesc'+i); if(d) c.desc=d.value;
  const extra=$('#cfix'+i)?$('#cfix'+i).value.trim():'';
  const desc=(c.desc||'')+(extra?(', '+extra):'');
  const r=await (await fetch('/api/cc/sheet',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({role:c.role,desc,ref:c.ref||''})})).json();
  if(r.error){ cAlert(r.error); return; }
  ST.cc.sheetRunning=true; ST.cc.sheetFor=i; ccSave(50);
}
function ccFix(i){ ccSheet(i); } // 按补充指导重画(提示词已在 ccSheet 里拼上)
async function ccPollSheet(){ // 轮询设定图是否画完,画完贴回对应角色
  const s=await j('/api/cc/status'); const sh=(s&&s.sheet)||{};
  if(sh.running){ if(ST.step===50){ ST.cc.sheetRunning=true; render(); } return; }
  ST.cc.sheetRunning=false;
  const i=ST.cc.sheetFor;
  if(sh.error){ cAlert('设定图失败:'+sh.error); render(); return; }
  if(i!=null && sh.url){ ST.cc.chars[i].sheet=sh.name+'.png'; ST.cc.chars[i].sheetUrl=sh.url; ST.cc.chars[i].approved=false; }
  ccSave(50);
}
function ccApprove(i){ ST.cc.chars[i].approved=!ST.cc.chars[i].approved; ccSave(50); }
function ccToPanels(){ // 去第2步: 要求全部角色都有设定图且通过审核
  const P=ST.cc; ccCollectChars();
  if(!P.chars.length){ cAlert('先生成角色卡'); return; }
  const bad=P.chars.find(c=>!c.sheetUrl||!c.approved);
  if(bad){ cAlert(`「${bad.name||'?'}」还没${!bad.sheetUrl?'生成设定图':'点✔通过'},全部通过才能下一步`); return; }
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
    ref_char:prev?prev.ref_char:0,w:prev?prev.w:832,h:prev?prev.h:1216,style:prev?prev.style:''});
  ccSave(51);
}
function ccDelPanel(i){ ccCollectPanels(); ST.cc.panels.splice(i,1); ST.cc.panels.forEach((p,k)=>p.num=k+1); ccSave(51); }
function ccAddDlg(i){ ccCollectPanels(); ST.cc.panels[i].dialogues.push({who:(ST.cc.chars[0]||{}).name||'',text:''}); ccSave(51); }
function ccDelDlg(i,j){ ccCollectPanels(); ST.cc.panels[i].dialogues.splice(j,1); ccSave(51); }
async function ccStartGen(){ // 校验并启动自定义连载
  const P=ST.cc; ccCollectPanels();
  if(!P.panels.length){ cAlert('先＋新增画布'); return; }
  if(P.panels.some(p=>!p.scene||!p.scene.trim())){ cAlert('有格子没写场景描述'); return; }
  await fetch('/api/cc/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({chars:P.chars,panels:P.panels})});
  const r=await (await fetch('/api/cc/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({chars:P.chars,panels:P.panels,style:P.gstyle||'',custom_style:P.gcustom||''})})).json();
  if(r.error){ cAlert(r.error); return; }
  ST.step=52; render();
}
async function ccStop(){ await (await fetch('/api/cc/stop',{method:'POST'})).json(); }
// ---- 图片测试场(⚙️) ----
function secCls(sec){ sec=+sec||60; return sec<180?'g':(sec<=300?'y':'r'); }  // <3分钟绿 / 3~5分钟黄 / >5分钟红
function testSafeFolderName(value){
  const clean=String(value||'').trim().replace(/[^\p{L}\p{N}_-]+/gu,'_').replace(/_+/g,'_').replace(/^[_-]+|[_-]+$/g,'');
  return clean.slice(0,80).replace(/[_-]+$/,'')||'model';
}
function testSafeSuffix(value){const raw=String(value||'').trim();const s=raw.replace(/[^\p{L}\p{N}_-]+/gu,'_').replace(/_+/g,'_').replace(/-+/g,'-').replace(/^[_-]+|[_-]+$/g,'').slice(0,40).replace(/[_-]+$/,'');return s?((raw.startsWith('-')?'-':'_')+s):'';}
function testDefault(){ return {models:[],canvas:{w:832,h:1216},prompts:'',per:1,pad_ref:'',folder_suffix:'',file_suffix:'',face_lock:{enabled:false,gen_model:'',prompt:''}}; }
function testNew(){ ST.test=testDefault(); ST.step=71; render(); }
function testToggleModel(id,name,sec,el){
  const T=ST.test; const i=T.models.findIndex(x=>x.id===id);
  if(i>=0) T.models.splice(i,1);
  else { T.models.push({id:id,name:name,folder:testSafeFolderName(name),sec:sec||60});
         T.models.sort((a,b)=>(a.sec||60)-(b.sec||60)); }   // 默认生图快的排前面
  if(!T.face_lock.gen_model && T.models.length) T.face_lock.gen_model=T.models[0].id;
  if(el) el.classList.toggle('sel', i<0);           // 原地切换选中态,不整页重渲(避免闪烁)
  const c=document.getElementById('tcount'); if(c) c.textContent=T.models.length;
  const o=document.getElementById('torder'); if(o) o.innerHTML=testOrderHtml();
}
function testOrderHtml(){ // 执行顺序列表(只原地刷新这个框)
  const T=ST.test;
  if(!T.models.length) return '执行顺序:还没选模型';
  return '执行顺序(默认生图快的排前面,按住 ⠿ 拖动或点 ↑↓ 调):' + T.models.map((m,i)=>
    `<div class="small tord" draggable="true" ondragstart="testDrag(event,${i})" ondragover="event.preventDefault()" ondrop="testDrop(event,${i})"
       style="margin-top:4px;padding:3px 6px;border-radius:6px;cursor:grab"><span style="cursor:grab;color:var(--faint)" title="按住拖动换位">⠿</span> <b>${i+1}.</b> ${esc(m.name)} <span class="mtime ${secCls(m.sec)}">${m.time||('约'+(m.sec||60)+'秒')}</span>
     <button class="back" style="padding:2px 8px;font-size:12px" onclick="testMove(${i},-1)">↑</button>
     <button class="back" style="padding:2px 8px;font-size:12px" onclick="testMove(${i},1)">↓</button></div>`).join('');
}
function testMove(i,dir){
  const T=ST.test; const j2=i+dir;
  if(j2<0||j2>=T.models.length) return;
  const t=T.models[i]; T.models[i]=T.models[j2]; T.models[j2]=t;
  const o=document.getElementById('torder'); if(o) o.innerHTML=testOrderHtml();
}
function testSelectAll(){ // 全选/清空: 一键把可选模型全加进来(再点清空)
  const T=ST.test; const all=T.all||[];
  if(!all.length) return;
  const allOn=all.every(m=>T.models.find(x=>x.id===m.id));
  T.models = allOn ? [] : all.map(m=>({id:m.id,name:m.name,folder:testSafeFolderName(m.name),sec:m.sec||60}));
  if(!allOn) T.models.sort((a,b)=>(a.sec||60)-(b.sec||60));   // 默认生图快的排前面
  if(!T.face_lock.gen_model && T.models.length) T.face_lock.gen_model=T.models[0].id;
  render();
}
function testDrag(e,i){ e.dataTransfer.setData('text/tord',String(i)); e.dataTransfer.effectAllowed='move'; }
function testDrop(e,to){ // 拖动换位: 把 from 项插到 to 项位置
  e.preventDefault();
  const from=+e.dataTransfer.getData('text/tord'); const T=ST.test;
  if(isNaN(from)||from===to||from<0||from>=T.models.length) return;
  T.models.splice(to,0,T.models.splice(from,1)[0]);
  const o=document.getElementById('torder'); if(o) o.innerHTML=testOrderHtml();
}
function testSetFolders(n){
  const T=ST.test;
  for(let i=0;i<n;i++){
    const el=document.getElementById('tf'+i);
    if(el&&T.models[i]) T.models[i].folder=testSafeFolderName(el.value||T.models[i].name);
  }
  T.folder_suffix=testSafeSuffix((document.getElementById('tfolderSuffix')||{}).value||'');
  T.file_suffix=testSafeSuffix((document.getElementById('tfileSuffix')||{}).value||'');
  goStep(73);
}
function testSetCanvas(w,h){
  ST.test.canvas={w:w,h:h};
  const c=document.getElementById('tcur'); if(c) c.innerHTML='当前:<b>'+w+'×'+h+'</b>';
  const cw=document.getElementById('cw'),ch=document.getElementById('ch');
  if(cw) cw.value=w; if(ch) ch.value=h;
}
function testSetPrompts(){
  const T=ST.test;
  T.prompts=document.getElementById('tpro').value;
  T.per=Math.min(10,Math.max(1,parseInt(document.getElementById('tper').value)||1));
  if(!T.prompts.split('\n').some(x=>x.trim())){ cAlert('至少写一行提示词'); return; }
  goStep(75);
}
function testSetPad(name,el){
  ST.test.pad_ref=name||'';
  document.querySelectorAll('.tpad').forEach(x=>x.classList.remove('sel'));
  if(el) el.classList.add('sel');
  const c=document.getElementById('tpadcur');
  if(c) c.innerHTML = ST.test.pad_ref ? '已选垫图: <b>'+esc(ST.test.pad_ref)+'</b>' : '当前: <b>不垫图(纯文字)</b>';
}
function testSetLock(){
  const fl=ST.test.face_lock; fl.enabled=!fl.enabled;
  if(fl.enabled && !fl.gen_model && ST.test.models.length) fl.gen_model=ST.test.models[0].id;
  const b=document.getElementById('tlockbtn'); if(b) b.textContent=fl.enabled?'🔒 已开(点我关)':'○ 已关(点我开)';
  const f=document.getElementById('tlockform'); if(f) f.style.display=fl.enabled?'':'none';
}
function testLockNext(){
  const fl=ST.test.face_lock;
  if(fl.enabled){
    fl.gen_model=(document.getElementById('tlockm')||{}).value||fl.gen_model;
    fl.prompt=(document.getElementById('tlockp')||{}).value||fl.prompt;
  }
  goStep(77);
}
async function testGo(){
  const T=ST.test;
  const body={models:T.models,canvas:T.canvas,per_prompt:T.per,
    prompts:T.prompts.split('\n').map(x=>x.trim()).filter(Boolean),
    pad_ref:T.pad_ref,face_lock:T.face_lock,folder_suffix:T.folder_suffix,file_suffix:T.file_suffix};
  const r=await (await fetch('/api/test/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(r.ok){ ST.step=78; render(); } else { cAlert('创建失败: '+(r.error||'?')); }
}
async function testCtl(cmd){
  if(cmd==='pause') ST.testMsg='⏸ 暂停指令已发送——正在生成的这张会跑完,下一张开始前停住(图不能中途掐断)';
  else ST.testMsg='';
  await (await fetch('/api/test/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cmd:cmd})})).json();
  render();
}
async function testRerun(id){ // 原样重跑: 同配置复制成一个新任务进队列
  if(!await cConfirm('按原配置把任务 '+id+' 重新跑一遍?(会生成一个新任务进队列)','重跑')) return;
  const r=await (await fetch('/api/test/rerun',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id})})).json();
  if(r.ok) render(); else cAlert('重跑失败: '+(r.error||'?'));
}
async function testRerunEdit(id){ // 改配置重跑: 把旧任务配置填回向导,逐步确认/修改后生成新任务
  const d=await j('/api/test/state');
  const t=(d.tasks||[]).find(x=>x.id===id);
  if(!t){ cAlert('找不到任务 '+id); return; }
  const p=await j('/api/test/prompts?f='+encodeURIComponent(id));
  ST.test={models:(t.models||[]).map(m=>({...m})),canvas:{...t.canvas},prompts:p.text||'',
    per:t.per_prompt||1,pad_ref:t.pad_ref||'',
    folder_suffix:t.folder_suffix||'',file_suffix:t.file_suffix||'',
    face_lock:t.face_lock||{enabled:false,gen_model:'',prompt:''}};
  ST.step=71; render();
}
async function testDel(id){ await (await fetch('/api/test/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id})})).json(); render(); }
function testEditPrompts(id){ ST.testEditId=id; ST.testReturnStep=ST.step===88?88:70; goStep(79); }
async function testSavePrompts(){
  const text=document.getElementById('tpe').value;
  await (await fetch('/api/test/prompts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:ST.testEditId,text:text})})).json();
  goStep(ST.testReturnStep||88);
}
// ---- 视频测试场 ----
function videoTestDefault(){return {models:[],all:[],loras:[],lora_ids:[],mode:'i2v',refs:[],sizes:['360x640'],custom:{w:360,h:640},durations:[2],fpses:[24],variants:[{w:360,h:640,duration:2,fps:24}],copies:1,lora:'none',stg:false,interpolate:false,native_audio:true,style_2d:false,prompts:'',folder_suffix:'',file_suffix:''};}
function videoTestNew(){ST.videoTest=videoTestDefault();ST.step=80;render();}
function videoTestToggleModel(id,name,sec,el){const V=ST.videoTest,i=V.models.findIndex(x=>x.id===id);if(i>=0)V.models.splice(i,1);else{const base=testSafeFolderName(name);V.models.push({id,name,sec:sec||300,folder:base,base_folder:base,custom_name:false});V.models.sort((a,b)=>(a.sec||300)-(b.sec||300));}if(el)el.classList.toggle('sel',i<0);const c=$('#vtcount');if(c)c.textContent=V.models.length;const o=$('#vtorder');if(o)o.innerHTML=videoTestOrderHtml();}
function videoTestOrderHtml(){const V=ST.videoTest;if(!V.models.length)return '执行顺序：还没选模型';return '执行顺序（默认快的在前，可拖动或点 ↑↓）：'+V.models.map((m,i)=>`<div class="small" draggable="true" ondragstart="videoTestDrag(event,${i})" ondragover="event.preventDefault()" ondrop="videoTestDrop(event,${i})" style="margin-top:4px;cursor:grab">⠿ <b>${i+1}.</b> ${esc(m.name)} · 约${m.sec||300}秒 <button class="back" style="padding:2px 8px" onclick="videoTestMove(${i},-1)">↑</button><button class="back" style="padding:2px 8px" onclick="videoTestMove(${i},1)">↓</button></div>`).join('');}
function videoTestMove(i,d){const V=ST.videoTest,j=i+d;if(j<0||j>=V.models.length)return;[V.models[i],V.models[j]]=[V.models[j],V.models[i]];const o=$('#vtorder');if(o)o.innerHTML=videoTestOrderHtml();}
function videoTestDrag(e,i){e.dataTransfer.setData('text/vtord',String(i));}
function videoTestDrop(e,to){e.preventDefault();const V=ST.videoTest,from=+e.dataTransfer.getData('text/vtord');if(isNaN(from)||from===to)return;V.models.splice(to,0,V.models.splice(from,1)[0]);const o=$('#vtorder');if(o)o.innerHTML=videoTestOrderHtml();}
function videoTestSelectAll(){const V=ST.videoTest,all=V.all||[];const on=all.length&&all.every(m=>V.models.some(x=>x.id===m.id));V.models=on?[]:all.map(m=>{const base=testSafeFolderName(m.name);return{id:m.id,name:m.name,sec:m.sec||300,folder:base,base_folder:base,custom_name:false}}).sort((a,b)=>a.sec-b.sec);render();}
function videoTestSyncFolders(){const V=ST.videoTest;for(const m of V.models){if(!m.base_folder)m.base_folder=testSafeFolderName(m.folder||m.name);m.folder=testSafeFolderName(m.base_folder+(V.folder_suffix||''));}}
function videoTestRename(i){const V=ST.videoTest,m=V.models[i];if(!m)return;if(m.renaming){const e=$('#vtf'+i);m.base_folder=testSafeFolderName(e?e.value:m.base_folder||m.name);m.custom_name=true;m.renaming=false;videoTestSyncFolders();render();return;}m.renaming=true;render();setTimeout(()=>{const e=$('#vtf'+i);if(e){e.focus();e.select()}},0);}
function videoTestApplySuffix(kind){const V=ST.videoTest,id=kind==='folder'?'#vtFolderSuffixInput':'#vtFileSuffixInput';const value=testSafeSuffix(($(id)||{}).value||'');if(!value){cAlert('先输入要应用的后缀');return;}if(kind==='folder'){V.folder_suffix=value;videoTestSyncFolders();}else V.file_suffix=value;render();}
function videoTestRemoveSuffix(kind){const V=ST.videoTest;if(kind==='folder'){V.folder_suffix='';videoTestSyncFolders();}else V.file_suffix='';render();}
function videoTestSetNames(){const V=ST.videoTest;V.models.forEach((m,i)=>{if(m.renaming){const e=$('#vtf'+i);m.base_folder=testSafeFolderName(e?e.value:m.base_folder||m.name);m.custom_name=true;m.renaming=false;}});videoTestSyncFolders();goStep(82);}
function videoTestMode(mode){ST.videoTest.mode=mode;render();}
function videoTestToggleRef(name,el){const V=ST.videoTest,i=V.refs.indexOf(name);if(i>=0)V.refs.splice(i,1);else V.refs.push(name);if(el)el.classList.toggle('sel',i<0);const c=$('#vtrefcount');if(c)c.textContent=V.refs.length;}
function videoTestRefsNext(){const V=ST.videoTest;if(V.mode==='i2v'&&!V.refs.length){cAlert('图生视频至少选择一张垫图');return;}goStep(83);}
function videoTestToggleArray(key,value,on){const a=ST.videoTest[key];const i=a.indexOf(value);if(on&&i<0)a.push(value);if(!on&&i>=0)a.splice(i,1);}
function videoTestAddCustomSize(){const V=ST.videoTest;let w=Math.max(64,Math.min(1920,Math.floor(+$('#vtcw').value/16)*16));let h=Math.max(64,Math.min(1920,Math.floor(+$('#vtch').value/16)*16));V.custom={w,h};const k=w+'x'+h;if(!V.sizes.includes(k))V.sizes.push(k);render();}
function videoTestParamsNext(){const V=ST.videoTest;if(!V.sizes.length||!V.durations.length||!V.fpses.length){cAlert('尺寸、时长、帧率都至少选一项');return;}V.copies=Math.max(1,Math.min(10,+($('#vtcopies')||{}).value||1));V.lora_ids=[...document.querySelectorAll('input[name="vtlora"]:checked')].map(x=>x.value);V.style_2d=!!($('#vt2d')||{}).checked;if(V.style_2d&&!V.lora_ids.includes('anime'))V.lora_ids.unshift('anime');V.lora=V.lora_ids[0]||'none';V.stg=!!($('#vtstg')||{}).checked;V.interpolate=!!($('#vtinterp')||{}).checked;V.native_audio=!!($('#vtnativeaudio')||{}).checked;V.variants=[];for(const s of V.sizes){const [w,h]=s.split('x').map(Number);for(const duration of V.durations)for(const fps of V.fpses)V.variants.push({w,h,duration,fps});}if(V.variants.length>100){cAlert('参数组合超过100组，请少选一些尺寸、时长或帧率');return;}goStep(84);}
function videoTestPromptsNext(){const V=ST.videoTest;V.prompts=($('#vtprompts')||{}).value||'';if(!V.prompts.split('\n').some(x=>x.trim())){cAlert('至少写一行提示词');return;}goStep(85);}
async function videoTestGo(){const V=ST.videoTest;const body={models:V.models,mode:V.mode,refs:V.refs,variants:V.variants,copies:V.copies,lora:(V.lora_ids[0]||'none'),loras:V.lora_ids,stg:V.stg,interpolate:V.interpolate,native_audio:V.native_audio,style_2d:!!V.style_2d,folder_suffix:V.folder_suffix,file_suffix:V.file_suffix,prompts:V.prompts.split('\n').map(x=>x.trim()).filter(Boolean)};const r=await (await fetch('/api/video-test/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();if(r.ok){ST.step=86;render()}else cAlert((r.error_code==='test_worker_conflict'?'为了防止内存爆满，当前不能启动：':'创建失败：')+(r.error||'?'));}
async function videoTestCtl(cmd){await fetch('/api/video-test/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cmd})});render();}
async function videoTestResume(id){const r=await (await fetch('/api/video-test/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();if(r.ok){cAlert('已加入视频队列，前面已完成的成片会自动跳过。','继续任务');render();}else cAlert('继续失败：'+(r.error||'?'));}
async function videoTestRerun(id){if(!await cConfirm('按原配置重新跑视频任务 '+id+'？','重跑'))return;const r=await (await fetch('/api/video-test/rerun',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();if(r.ok)render();else cAlert('重跑失败：'+(r.error||'?'));}
async function videoTestRerunEdit(id){const d=await j('/api/video-test/state');const t=(d.tasks||[]).find(x=>x.id===id);if(!t){cAlert('找不到视频任务 '+id);return;}const p=await j('/api/video-test/prompts?f='+encodeURIComponent(id));const V=videoTestDefault();V.models=(t.models||[]).map(m=>{const base=m.base_folder||m.folder||testSafeFolderName(m.name);return{...m,base_folder:base,folder:base,output_folder:'',custom_name:base!==testSafeFolderName(m.name)}});V.all=d.models||[];V.loras=d.loras||[];V.mode=t.mode||'i2v';V.refs=[...(t.refs||[])];V.variants=(t.variants||[]).map(v=>({...v}));V.sizes=[...new Set(V.variants.map(v=>v.w+'x'+v.h))];V.durations=[...new Set(V.variants.map(v=>v.duration))];V.fpses=[...new Set(V.variants.map(v=>v.fps))];V.copies=t.copies||1;V.lora_ids=[...(t.loras||((t.lora&&t.lora!=='none')?[t.lora]:[]))];V.style_2d=!!t.style_2d;if(V.style_2d&&!V.lora_ids.includes('anime'))V.lora_ids.unshift('anime');V.lora=V.lora_ids[0]||'none';V.stg=!!t.stg;V.interpolate=!!t.interpolate;V.native_audio=!!t.native_audio;V.folder_suffix=t.folder_suffix||'';V.file_suffix=t.file_suffix||'';V.prompts=p.text||'';ST.videoTest=V;ST.step=80;render();}
async function videoTestDel(id){const r=await (await fetch('/api/video-test/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();if(!r.ok&&r.error)cAlert('删除失败：'+r.error);else render();}
function videoTestEditPrompts(id){ST.videoTestEditId=id;ST.videoReturnStep=ST.step===89?89:70;goStep(87);}
async function videoTestSavePrompts(){const text=($('#vtpe')||{}).value||'';await fetch('/api/video-test/prompts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:ST.videoTestEditId,text})});goStep(ST.videoReturnStep||89);}
// ---- 气泡编辑器(可拖拽) ----
function bubStage(){ return document.getElementById('bstage'); }
function bubCur(){ return ((ST.bub&&ST.bub.list)||[]).find(x=>x.id===ST.bub.sel)||null; }
async function bubOpen(name){ ST.bub={cur:name,list:[],seq:1,sel:null,flat:''}; render(); }
function bubBack(){ if(ST.bub){ ST.bub.cur=''; ST.bub.flat=''; } render(); }
function bubAdd(){
  const B=ST.bub; const n=B.list.length;
  const nb={id:B.seq++,x:0.3+((n*0.07)%0.3),y:0.12+((n*0.1)%0.5),w:0.34,text:'',who:'',type:'speech',font:''};
  B.list.push(nb); B.sel=nb.id; bubSync();
  const st=bubStage(); const ft=st&&st.querySelector(`.bub[data-id="${nb.id}"] .bubtext`); if(ft) ft.focus();
}
function bubDel(){ const B=ST.bub; if(B.sel==null) return; B.list=B.list.filter(x=>x.id!==B.sel); B.sel=null; bubSync(); }
function bubSetType(t){ const b=bubCur(); if(b){ b.type=t; bubSync(); } }
function bubSetFont(f){ const b=bubCur(); if(b){ b.font=f; bubSync(); } }
function bubSetWho(w){ const b=bubCur(); if(b){ b.who=w; bubSync(); } }
function bubSync(){
  const st=bubStage(); if(!st||!ST.bub) return;
  st.querySelectorAll('.bub').forEach(e=>e.remove());
  const img=document.getElementById('bimg'); const D=img?(img.clientWidth||0):0;
  const B=ST.bub;
  B.list.forEach(b=>{
    const d=document.createElement('div'); d.className='bub'; d.dataset.id=b.id;
    d.style.left=(b.x*100)+'%'; d.style.top=(b.y*100)+'%'; d.style.width=(b.w*100)+'%';
    const fam=(b.font&&ST.bubFonts&&ST.bubFonts[b.font])?`'${ST.bubFonts[b.font]}'`:'inherit';
    d.innerHTML=`<div class="bubhandle" data-id="${b.id}">⠿</div><div class="bubtext ${b.type} ${b.id===B.sel?'sel':''}" contenteditable="true" data-id="${b.id}" style="font-family:${fam};font-size:${D?D/28:16}px">${esc(b.text)}</div>`;
    const hd=d.querySelector('.bubhandle'); const ft=d.querySelector('.bubtext');
    hd.addEventListener('pointerdown', bubDragStart);
    ft.addEventListener('focus', ()=>{ B.sel=b.id; st.querySelectorAll('.bubtext').forEach(x=>x.classList.remove('sel')); ft.classList.add('sel'); });
    ft.addEventListener('input', ()=>{ b.text=ft.innerText; });
    ft.addEventListener('keydown', (ev)=>{ if(ev.key==='Enter'&&!ev.shiftKey){ ev.preventDefault(); bubAdd(); } });
    st.appendChild(d);
  });
}
function bubDragStart(e){
  e.preventDefault();
  const id=+e.target.dataset.id; const b=((ST.bub&&ST.bub.list)||[]).find(x=>x.id===id); if(!b) return;
  ST.bub.sel=id;
  const st=bubStage(); const r=st.getBoundingClientRect();
  const sx=e.clientX, sy=e.clientY, ox=b.x, oy=b.y;
  function mv(ev){ b.x=Math.min(0.97,Math.max(0,ox+(ev.clientX-sx)/r.width)); b.y=Math.min(0.97,Math.max(0,oy+(ev.clientY-sy)/r.height)); const el=st.querySelector(`.bub[data-id="${id}"]`); if(el){ el.style.left=(b.x*100)+'%'; el.style.top=(b.y*100)+'%'; } }
  function up(){ window.removeEventListener('pointermove',mv); window.removeEventListener('pointerup',up); }
  window.addEventListener('pointermove',mv); window.addEventListener('pointerup',up);
}
async function bubExport(){
  const B=ST.bub; if(!B||!B.cur) return;
  const bubbles=B.list.map(b=>({x:b.x,y:b.y,w:b.w,text:b.text,who:b.who,type:b.type,font:b.font}));
  const r=await j('/api/cc/flatten',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:B.cur,bubbles})});
  if(r.error){ cAlert('导出失败: '+r.error); return; }
  B.flat=r.url; render();
}
function pickModel(id,name,sec,kind){ ST.model=id; ST.mname=name; ST.msec=sec; ST.kind=kind; if(!ST.items.length) ST.items.push(newItem()); ST.step=3; render(); }
function hqOpen(){
  ST.hq=ST.hq||{session:'hq_'+Date.now(),model:'',faceSize:'1024x1024',count:1,bust:'natural',waist:'natural',hip:'natural',size:'832x1216',batchNo:0,variants:[],selected:-1};
  ST.step=90; render();
}
function hqSlug(s){ return String(s||'').replace(/[^A-Za-z0-9_-]+/g,'_').replace(/^[_-]+|[_-]+$/g,'').slice(0,50)||'value'; }
function hqStartFace(){
  const h=ST.hq||{}; h.facePrompt=($('#hqfacep')||{}).value||h.facePrompt||''; h.neg=($('#hqneg')||{}).value||h.neg||NEG_DEF;
  h.model=($('#hqmodel')||{}).value||h.model; h.faceSize=h.faceSize||'1024x1024';
  const wh=h.faceSize.split('x');
  if(!h.model){ cAlert('没有可用的 SDXL 单文件模型'); return; }
  h.faceUrl=''; h.faceRef=''; h.faceDone=false; h.faceError=''; h.facePolling=false;
  const name=h.session+'_face_'+Date.now();
  j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:h.model,pos:h.facePrompt,neg:h.neg,w:+wh[0],h:+wh[1],name,archive:h.session})}).then(r=>{
    if(r.error){ h.faceError=r.error; render(); return; }
    h.facePid=r.pid; ST.step=91; render();
  }).catch(e=>{ h.faceError=String(e); render(); });
  ST.step=91; render();
}
async function hqPollFace(){
  const h=ST.hq; if(!h||!h.facePid||h.facePolling) return; h.facePolling=true;
  while(ST.step===91 && !h.faceDone){
    await new Promise(r=>setTimeout(r,1800));
    let r; try{ r=await j('/api/poll?pid='+h.facePid); }catch(e){ continue; }
    if(r.error){ h.faceError=r.error; h.facePolling=false; render(); return; }
    if(r.done){ h.faceUrl=r.url; h.faceDone=true; h.facePolling=false; render(); return; }
  }
  h.facePolling=false;
}
async function hqAcceptFace(){
  const h=ST.hq; if(!h||!h.faceUrl) return;
  try{
    const blob=await (await fetch(h.faceUrl)).blob();
    h.faceRef=await uploadFile(new File([blob],'hq-face.png',{type:'image/png'}),'hq-face-'+h.session+'.png');
    ST.step=92; render();
  }catch(e){ cAlert('锁脸参考上传失败: '+e.message); }
}
async function hqStartVariants(){
  const h=ST.hq; if(!h.faceRef){ cAlert('请先锁定脸部参考图'); return; }
  h.bodyPrompt=(($('#hqbpos')||{}).value||'').trim(); h.bust=($('#hqbust')||{}).value||'natural'; h.waist=($('#hqwaist')||{}).value||'natural'; h.hip=($('#hqhip')||{}).value||'natural'; h.count=+(($('#hqcount')||{}).value||1); h.size=(($('#hqsize')||{}).value||'832x1216');
  const bust={natural:'natural bust proportion',full:'fuller bust proportion',very_full:'very full bust proportion'}[h.bust];
  const waist={natural:'natural waist',slim:'slim waist',very_slim:'defined slim waist'}[h.waist];
  const hip={natural:'natural hip proportion',full:'fuller hips',very_full:'very full hips'}[h.hip];
  const wh=h.size.split('x'); h.batchNo=(h.batchNo||0)+1; h.variants=[]; ST.step=93; render();
  for(let i=0;i<h.count;i++){
    const name=h.session+'_r'+h.batchNo+'_v'+(i+1)+'_bust-'+hqSlug(h.bust)+'_waist-'+hqSlug(h.waist)+'_hip-'+hqSlug(h.hip);
    const pos='masterpiece, best quality, high resolution, photorealistic, neutral skin tone, balanced white balance, '+bust+', '+waist+', '+hip+', '+h.bodyPrompt;
    const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:h.model,pos,neg:h.neg,w:+wh[0],h:+wh[1],name,ipa:h.faceRef,ipa_weight:0.3,archive:h.session})});
    const v={name,pid:r.pid||'',done:false,error:r.error||'',url:''}; h.variants.push(v); render();
    if(r.error) break;
  }
  if(!h.variants.length) h.variants=[{name:'提交失败',done:false,error:'没有提交成功',url:''}];
  h.variantPolling=false; render();
}
function hqPollVariants(){
  const h=ST.hq; if(!h||h.variantPolling) return; h.variantPolling=true;
  const tick=async()=>{
    if(ST.step!==93){ h.variantPolling=false; return; }
    let active=false;
    for(const v of h.variants){
      if(v.done||v.error||!v.pid) continue; active=true;
      let r; try{ r=await j('/api/poll?pid='+v.pid); }catch(e){ continue; }
      if(r.error) v.error=r.error; else if(r.done){ v.done=true; v.url=r.url; }
    }
    render();
    if(active) setTimeout(tick,2000); else h.variantPolling=false;
  };
  tick();
}
function hqChoose(i){ const h=ST.hq; h.selected=i; render(); }
function addCard(){ for(let i=0;i<ST.items.length;i++) collect(i); ST.items.push(newItem()); render(); }
function delCard(i){ if(ST.items.length<=1){ cAlert('至少留一张'); return; } for(let j=0;j<ST.items.length;j++) collect(j); ST.items.splice(i,1); render(); }
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
function getTags(target){ // 读该组提示词标签(用户自定义存浏览器本地,默认用预置)
  const def=(target==='neg'?NEG_TAGS:POS_TAGS);
  try{ const s=localStorage.getItem('ivs_tags_'+target); if(s){ const a=JSON.parse(s); if(Array.isArray(a)&&a.length) return a; } }catch(e){}
  return def.slice();
}
function saveTags(target,arr){ try{ localStorage.setItem('ivs_tags_'+target, JSON.stringify(arr)); }catch(e){} }
function tagRow(i,tags,target){ // 渲染一排可点标签,target='pos'/'neg';tags 参数忽略,统一读 getTags(可自定义增删)
  const cur=((target==='pos'?ST.items[i].pos:ST.items[i].neg)||'').split(',').map(s=>s.trim()).filter(Boolean);
  const list=getTags(target);
  return `<div class="tags" id="tagrow-${target}-${i}">${list.map(t=>`<span class="tag${target==='neg'?' neg':''}${cur.includes(t)?' on':''}" onclick="tagToggle(${i},'${t}','${target}',this)">${esc(t)}<span class="tagx" title="删除这个标签" onclick="tagDel(event,${i},'${t}','${target}')">✕</span></span>`).join('')}<span class="tag tagadd" title="自定义新增一个标签" onclick="tagAdd(${i},'${target}')">＋</span></div>`;
}
function refreshTagRow(i,target){ const o=document.getElementById(`tagrow-${target}-${i}`); if(o) o.outerHTML=tagRow(i,null,target); }
function tagToggle(i,word,target,el){ // 点标签: 没有就加,有了就取消(原地更新,不整页重渲 → 不闪烁)
  collect(i); const it=ST.items[i];
  const arr=((target==='pos'?it.pos:it.neg)||'').split(',').map(s=>s.trim()).filter(Boolean);
  const k=arr.indexOf(word);
  if(k>=0) arr.splice(k,1); else arr.push(word);
  const v=arr.join(', ');
  if(target==='pos') it.pos=v; else it.neg=v;
  const ta=$('#'+target+i); if(ta) ta.value=v;   // 原地更新输入框内容
  if(el) el.classList.toggle('on', k<0);          // 原地切换标签高亮
}
async function tagAdd(i,target){ // 自定义新增标签(去掉引号防破坏内联事件)
  const w=await cPrompt(target==='neg'?'新增一个负向提示词标签:':'新增一个正向提示词标签:','');
  if(w===null) return; const w2=w.trim().replace(/['"\\]/g,''); if(!w2) return;
  const arr=getTags(target);
  if(arr.includes(w2)){ cAlert('这个标签已经有了'); return; }
  arr.push(w2); saveTags(target,arr); refreshTagRow(i,target);
}
function tagDel(e,i,word,target){ // 删除标签(只删标签,不动输入框里已写的词)
  e.stopPropagation();
  const arr=getTags(target); const k=arr.indexOf(word);
  if(k>=0){ arr.splice(k,1); saveTags(target,arr); }
  refreshTagRow(i,target);
}
function sizeRow(i,it){
  let sizeOpts = SIZES.map(s=>`<option value="${s[0]}x${s[1]}" ${it.w===s[0]&&it.h===s[1]?'selected':''}>${s[0]} x ${s[1]} ${s[2]}</option>`).join('');
  const isCust = !SIZES.some(s=>s[0]===it.w&&s[1]===it.h);
  if(isCust) sizeOpts += `<option value="cust" selected>自定义 ${it.w} x ${it.h}</option>`;
  else sizeOpts += `<option value="cust">自定义…</option>`;
  return `<label>尺寸 <span class="small">(最小 640×960,SDXL 再小会糊)</span></label><div class="row"><select onchange="sizeSel(${i},this)">${sizeOpts}</select>
    <span id="cust${i}" style="display:${isCust?'inline':'none'}">
      <input type="number" id="cw${i}" value="${it.w}" style="width:70px" min="640" max="1280" onchange="custSize(${i})"> x
      <input type="number" id="ch${i}" value="${it.h}" style="width:70px" min="640" max="1280" onchange="custSize(${i})"></span>
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
    <input type="file" id="ref${i}" accept="image/*" style="display:none" onchange="refPick(${i},this)">
    <div class="row" style="margin-top:6px">
      <button type="button" class="back" style="padding:4px 12px;font-size:13px"
              onclick="event.stopPropagation();refLibToggle(${i})">📚 从参考图库选</button>
      <span class="small">图库(漫画连载那套可传多张)里点一张直接当垫图</span></div>
    <div id="reflib${i}" style="display:none;margin-top:4px;line-height:0"></div>`;
}
function refPick(i,el){ collect(i); setRef(i,el.files[0]||null); }
function dropRef(i,e){
  e.preventDefault();
  const dz=e.currentTarget; dz.classList.remove('over');
  const f=e.dataTransfer.files && e.dataTransfer.files[0];
  if(!f) return;
  if(!f.type.startsWith('image/')){ cAlert('拖进来的不是图片文件!'); return; }
  collect(i); setRef(i,f);
}
function setRef(i,f){ ST.items[i].refFile=f; ST.items[i].strokes=[]; render(); }
async function refLibToggle(i){ // 展开/收起参考图库,列出缩略图供点选当垫图
  const box=$('#reflib'+i); if(!box) return;
  if(box.style.display!=='none'){ box.style.display='none'; return; }
  box.style.display='block'; box.innerHTML='<span class="small">载入图库…</span>';
  let d; try{ d=await j('/api/refs'); }catch(e){ box.innerHTML='<span class="small">图库载入失败</span>'; return; }
  if(!d.refs||!d.refs.length){ box.innerHTML='<span class="small">图库是空的 → 到「漫画连载」页可一次传多张参考图</span>'; return; }
  box.innerHTML=d.refs.map(r=>`<img src="${r.url}" title="${esc(r.name)}" loading="lazy"
    style="width:72px;height:72px;object-fit:cover;border-radius:8px;border:2px solid var(--border);margin:3px;cursor:pointer;display:inline-block"
    onclick="refLibPick(${i},'${encodeURIComponent(r.name)}')">`).join('');
}
async function refLibPick(i,name){ // 从图库选一张 → 拉成 File 喂给现有 i2i 垫图流程
  const fname=decodeURIComponent(name);
  try{
    const blob=await (await fetch('/refs/'+name)).blob();
    collect(i); setRef(i,new File([blob],fname,{type:blob.type||'image/png'}));
  }catch(e){ cAlert('读取图库图片失败:'+e.message); }
}
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
    if(it.mode!=='t2i' && !(it.mode==='pose'&&it.poseSrc==='draw') && !it.refFile){ cAlert('第 '+(i+1)+' 张还没上传图片!'); return; }
    if(it.mode==='inpaint' && !it.strokes.length){ cAlert('第 '+(i+1)+' 张还没涂抹要重画的区域!'); return; }
    try{
      it.ref=null; it.mask=null;
      if(it.mode==='pose'&&it.poseSrc==='draw'){
        it.ctype='openpose';
        const blob=await exportPose(i); it.ref=await uploadFile(blob,'pose-'+Date.now()+'-'+i+'.png');
      }else if(it.mode!=='t2i') it.ref=await uploadFile(it.refFile);
      if(it.mode==='inpaint'){ const blob=await exportMask(i); it.mask=await uploadFile(blob,'mask-'+Date.now()+'-'+i+'.png'); }
    }catch(e){ cAlert('第 '+(i+1)+' 张: '+e.message); return; }
  }
  ST.pids=[];
  for(let i=0;i<ST.items.length;i++){
    const it=ST.items[i];
    const name=`img-${Date.now()}-${String(i+1).padStart(2,'0')}`;
    const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:ST.model,mode:it.mode,pos:it.pos,neg:it.neg,w:it.w,h:it.h,name,
        ref:it.ref,mask:it.mask,strength:it.strength,scale:it.scale,ctype:it.ctype})});
    if(r.error){ cAlert('提交失败: '+r.error); return; }
    ST.pids.push(r.pid);
  }
  saveImgJobs();   // 存档: 刷新页面也能找回这批任务继续看进度
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
        left--; if(left<=0){ $('#again').style.display='block'; const sb=$('#stopBtn'); if(sb) sb.style.display='none'; clearImgJobs(); } return; }
      if(r.error){ $('#b'+i).style.background='#d03050'; $('#st'+i).textContent='❌ '+r.error;
        left--; if(left<=0){ $('#again').style.display='block'; const sb2=$('#stopBtn'); if(sb2) sb2.style.display='none'; clearImgJobs(); } return; }
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
function saveImgJobs(){ // 把当前这批生图任务存进浏览器,刷新后可找回继续看进度
  try{
    const items=ST.items.map(it=>({mode:it.mode,pos:it.pos,neg:it.neg,w:it.w,h:it.h,
      strength:it.strength,scale:it.scale,ctype:it.ctype,resultUrl:it.resultUrl||null}));
    localStorage.setItem('ivs_jobs_img', JSON.stringify({pids:ST.pids, items, model:ST.model, mname:ST.mname, msec:ST.msec, t0:Date.now()}));
  }catch(e){}
}
function clearImgJobs(){ try{ localStorage.removeItem('ivs_jobs_img'); }catch(e){} }
async function stopAll(){ // 停止当前这批生图(停掉正在跑的 + 删掉排队的)
  if(!ST.pids.length) return;
  if(!await cConfirm(`确定停止这批生图吗?正在跑的和排队的 ${ST.pids.length} 个任务都会被取消。`,'停止')) return;
  const r=await j('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pids:ST.pids})});
  if(r.error){ cAlert('停止失败: '+r.error); return; }
  ST.pids.forEach((p,i)=>{ const st=$('#st'+i); if(st && st.textContent.indexOf('✅')<0){ const b=$('#b'+i); if(b) b.style.background='#8a8f98'; st.textContent='⏹ 已停止'; } });
  const sb=$('#stopBtn'); if(sb) sb.style.display='none';
  const ag=$('#again'); if(ag) ag.style.display='block';
  clearImgJobs();
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
  }catch(e){ cAlert('第 '+(i+1)+' 张: '+e.message); return; }
  const name=`img-${Date.now()}-${String(i+1).padStart(2,'0')}`;
  const r=await j('/api/gen',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:ST.model,mode:it.mode,pos:it.pos,neg:it.neg,w:it.w,h:it.h,name,
      ref:it.ref,mask:it.mask,strength:it.strength,scale:it.scale,ctype:it.ctype})});
  if(r.error){ cAlert('提交失败: '+r.error); return; }
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
  if(dead){ cAlert('⚠ '+dead+'已停止(进程被关闭)。'); home(); }
},5000);
// 启动时按当前运行状态恢复页面: llm 在跑→语言页, img 在跑→生图, vid 在跑→视频, 否则首页
(async()=>{
  try{
    // 恢复刷新前提交的生图任务(任务还在 ComfyUI 后台跑,前端找回继续看进度)
    try{
      const d=JSON.parse(localStorage.getItem('ivs_jobs_img')||'null');
      if(d && d.pids && d.pids.length){
        ST.model=d.model; ST.mname=d.mname||''; ST.msec=d.msec||60;
        ST.items=d.items.map(it=>Object.assign(newItem(), it));
        ST.pids=d.pids; ST.step=4; render(); return;
      }
    }catch(e){}
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
    const wb=await j('/api/wb/state');
    if(wb && wb.busy){ ST.step=61; render(); return; } // 配音工作台后台还在合成→回工作台看进度
  }catch(e){}
  home();
})();
</script></body></html>""".replace("%NEG%", json.dumps(NEG_DEFAULT)).replace("%VID%", str(VID_PORT))

# ---------------- HTTP ----------------
def video_tasks_snapshot():
    """统一列出普通视频任务和磁盘成片；此函数只读，不会启动模型。"""
    tasks = vidwf.list_tasks()
    known_urls = {str(task.get("url") or "") for task in tasks if task.get("url")}
    try:
        files = [os.path.join(vidwf.OUT_VID, name) for name in os.listdir(vidwf.OUT_VID)
                 if name.lower().endswith(".mp4")]
    except OSError:
        files = []
    for path in sorted(files, key=lambda p: os.path.getmtime(p), reverse=True):
        name = os.path.basename(path)
        url = "/vout/" + urllib.parse.quote(name)
        if url in known_urls:
            continue
        tasks.append({"id": "file:" + name, "engine": "file", "name": os.path.splitext(name)[0],
                      "state": "done", "error": "", "url": url,
                      "created": os.path.getmtime(path)})
    order = {"running": 0, "queued": 1, "done": 2, "failed": 3}
    tasks.sort(key=lambda t: (order.get(t.get("state"), 9), -float(t.get("created") or 0)))
    return {"tasks": tasks}

def _responses_parts(response):
    """提取 GPT-OSS Responses API 的正文和推理摘要，兼容不同版本 schema。"""
    answer, think = [], []
    for item in response.get("output", []) or []:
        typ = item.get("type")
        parts = item.get("content") or item.get("summary") or []
        if isinstance(parts, str):
            parts = [{"text": parts}]
        target = think if typ == "reasoning" else answer if typ in ("message", "output_text") else None
        if target is None:
            continue
        for part in parts:
            value = part if isinstance(part, str) else (part.get("text") or part.get("value") or "")
            if value:
                target.append(value)
    if not answer and response.get("output_text"):
        answer.append(str(response["output_text"]))
    return "\n".join(answer).strip(), "\n".join(think).strip()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _qarg(self, key, default=""):
        """从 query string 取一个参数(?type=vid → 'vid')。"""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get(key, [default])[0]
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 前端轮询时切页/刷新/关网页,客户端先断连,属正常,不刷屏
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
        elif self.path == "/api/vid/tasks":
            self._send(200, json.dumps(video_tasks_snapshot()))
        elif self.path == "/api/llm/models":
            self._send(200, json.dumps(llm.list_models()))
        elif self.path == "/api/llm/metal/status":
            self._send(200, json.dumps(llm.metal_status()))
        elif self.path == "/api/llm/current":
            self._send(200, json.dumps(llm.current()))
        elif self.path == "/api/llm/stats":
            self._send(200, json.dumps(llm.stats()))
        elif self.path == "/api/tts/state":
            self._send(200, json.dumps(tts_state()))
        elif self.path == "/api/tts/voices":
            self._send(200, json.dumps({"voices": tts_voices()}))
        elif self.path == "/api/tts/start":
            self._send(200, json.dumps(tts_start()))
        elif self.path == "/api/tts/stop":
            self._send(200, json.dumps(tts_stop()))
        elif self.path == "/api/tts/progress":
            self._send(200, json.dumps(tts_progress()))
        elif self.path == "/api/wb/state":
            self._send(200, json.dumps(wb_state()))
        elif self.path.startswith("/api/tts/vsample/"):
            # 声音库试听音频: models/tts/voices/<名>/sample.wav
            nm = os.path.basename(urllib.parse.unquote(self.path[len("/api/tts/vsample/"):].split("?")[0]))
            fp = os.path.join(TTS_VOICES, nm, "sample.wav")
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "audio/wav")
            else: self._send(404, "{}")
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
        elif self.path == "/api/cc/panels":
            self._send(200, json.dumps(cc_panels()))
        elif self.path == "/api/fonts":
            self._send(200, json.dumps({"fonts": ce.list_fonts()}))
        elif self.path == "/api/test/state":
            self._send(200, json.dumps(test_state()))
        elif self.path.startswith("/api/test/prompts"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self._send(200, json.dumps(test_prompts_get(q.get("f", [""])[0])))
        elif self.path == "/api/video-test/state":
            self._send(200, json.dumps(video_test_state()))
        elif self.path.startswith("/api/video-test/prompts"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            self._send(200, json.dumps(video_test_prompts_get(q.get("f", [""])[0])))
        elif self.path == "/api/refs":
            self._send(200, json.dumps(list_refs()))
        elif self.path.startswith("/refs/"):
            fp = os.path.join(REFS_DIR, os.path.basename(urllib.parse.unquote(self.path[6:].split("?")[0])))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "image/png")
            else: self._send(404, "{}")
        elif self.path.startswith("/fonts/"):
            fp = os.path.join(ce.FONTS_DIR, os.path.basename(urllib.parse.unquote(self.path[7:].split("?")[0])))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "font/ttf")
            else: self._send(404, "{}")
        elif self.path.startswith("/api/vid/poll"):
            token = self._qarg("pid")
            self._send(200, json.dumps(vidwf.poll_vid(token)))
        elif self.path.startswith("/api/svc/status"):
            t = self._qarg("type", "img")
            self._send(200, json.dumps(svc.svc_status(t)))
        elif self.path.startswith("/api/svc/start"):
            t = self._qarg("type", "img")
            self._send(200, json.dumps(svc.start_svc(t)))
        elif self.path.startswith("/api/svc/stop"):
            t = self._qarg("type", "img")
            # llm 的"关闭"要走 close_llm: 停进程外还清掉 current_llm.json,
            # 否则 paused 标记残留,下次打开页面被强制跳回"已暂停"页
            r = llm.close_llm() if t == "llm" else svc.stop_svc(t)
            self._send(200, json.dumps(r))
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
            # 图片地址允许带 ?t=时间防缓存,查询参数不能参与文件名匹配。
            rel = self.path[5:].split("?", 1)[0]
            fp = os.path.join(OUT_DIR, os.path.basename(urllib.parse.unquote(rel)))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "image/png")
            else: self._send(404, "{}")
        elif self.path.startswith("/vout/"):
            rel = self.path[6:].split("?", 1)[0]
            fp = os.path.join(vidwf.OUT_VID, os.path.basename(urllib.parse.unquote(rel)))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "video/mp4")
            else: self._send(404, "{}")
        elif self.path.startswith("/vidtest/"):
            rel = urllib.parse.unquote(self.path[len("/vidtest/"):].split("?", 1)[0]).lstrip("/")
            fp = os.path.realpath(os.path.join(VTEST_OUT, rel))
            if fp.startswith(os.path.realpath(VTEST_OUT) + os.sep) and os.path.isfile(fp):
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
        elif self.path.startswith("/tts_out/"):
            fp = os.path.join(TTS_OUT, os.path.basename(self.path[9:].split("?")[0]))
            if os.path.exists(fp):
                with open(fp, "rb") as f: self._send(200, f.read(), "audio/wav")
            else: self._send(404, "{}")
        else:
            self._send(404, "{}")
    def do_POST(self):
        if self.path == "/api/gen":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pid = submit(d["model"], d["pos"], d["neg"], int(d["w"]), int(d["h"]), d["name"],
                             d.get("mode", "t2i"), d.get("ref"), d.get("mask"),
                             float(d.get("strength", 0.6)), float(d.get("scale", 2.0)), d.get("ctype", "openpose"),
                             d.get("ipa"), float(d.get("ipa_weight", 0.3)), int(d.get("batch", 1)),
                             int(d.get("timeout", 1800)), d.get("archive"))
                self._send(200, json.dumps({"pid": pid}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cancel":
            # 停止一批生图: 先 interrupt 停掉正在跑的那张,再从队列删掉所有排队任务
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pids = d.get("pids") or ([d["pid"]] if d.get("pid") else [])
                try: comfy_post("/interrupt", {})
                except Exception: pass
                try: comfy_post("/queue", {"delete": pids})
                except Exception: pass
                with LOCK:
                    for p in pids:
                        if p in TASKS: TASKS[p]["error"] = "已停止"
                self._send(200, json.dumps({"ok": True, "stopped": len(pids)}))
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
        elif self.path.startswith("/api/vid/audio"):
            # 音频不上传给 ComfyUI,先落到本项目临时目录,生成结束后由 vidwf 用 ffmpeg 混入。
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if not body: raise ValueError("没有收到音频文件")
                os.makedirs(vidwf.VID_AUDIO, exist_ok=True)
                raw = os.path.basename(self._qarg("fname", "audio.bin"))
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(raw)[0])[:80] or "audio"
                ext = os.path.splitext(raw)[1].lower() or ".bin"
                name = stem + "_" + str(int(time.time()*1000)) + ext
                with open(os.path.join(vidwf.VID_AUDIO, name), "wb") as f: f.write(body)
                self._send(200, json.dumps({"ok": True, "name": name}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/vid/gen":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                audio_name = os.path.basename(str(d.get("audio", "")))
                audio_path = os.path.join(vidwf.VID_AUDIO, audio_name) if audio_name else ""
                if audio_path and not os.path.isfile(audio_path):
                    raise ValueError("找不到上传的音频文件")
                service = svc.comfy_health("vid")
                if not service.get("api_ok"):
                    started = svc.start_svc("vid")
                    if not started.get("ok"):
                        raise ValueError(started.get("error", "生视频服务启动失败"))
                    deadline = time.time() + 240
                    while time.time() < deadline:
                        if svc.comfy_health("vid", timeout=3).get("api_ok"):
                            break
                        time.sleep(3)
                    else:
                        raise ValueError("生视频服务启动后一直没有准备好")
                common = dict(unet_id=d["unet"], pos=d.get("pos", ""), neg=d.get("neg", ""),
                              image_name=d.get("image", ""), w=int(d["w"]), h=int(d["h"]),
                              frames=int(d["frames"]), fps=float(d.get("fps", 24)),
                              lora_id=d.get("lora", "none"), lora_strength=float(d.get("lora_strength", 0.8)),
                              lora_ids=d.get("loras"), use_stg=bool(d.get("stg", False)),
                              steps=int(d.get("steps", 8)), style_2d=bool(d.get("style_2d", False)),
                              native_audio=bool(d.get("native_audio", False)))
                post = {"interpolate": bool(d.get("interpolate", False)),
                        "interpolate_fps": float(d.get("interpolate_fps", 0) or 0),
                        "audio_path": audio_path, "native_audio": bool(d.get("native_audio", False))}
                # 长片统一拆成均衡小段（例如 8 秒=4+4），模型不会一次计算过大的 latent。
                if float(d.get("duration", 0) or 0) > 6.0:
                    pid = vidwf.submit_long_vid(d["name"], float(d["duration"]), postprocess=post, **common)
                else:
                    pid = vidwf.submit_vid(d["name"], postprocess=post, **common)
                self._send(200, json.dumps({"pid": pid}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/vid/cancel":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                token = str(d.get("pid", ""))
                vidwf.cancel_vid(token)
                self._send(200, json.dumps({"ok": True}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/llm/start":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r = llm.start_llm(d["id"], bool(d.get("thinking", True)),
                                  float(d.get("temp", 0.7)), int(d.get("max_tokens", 16384)),
                                  bool(d.get("mtp", False)), bool(d.get("codex_proxy", False)),
                                  int(d.get("ctx", 32768)), int(d.get("budget", -1)),
                                  str(d.get("reasoning_level", "high")), int(d.get("parallel", 1)),
                                  d.get("proxy_mode"))
                self._send(200, json.dumps(r))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/llm/metal/init":
            try:
                self._send(200, json.dumps(llm.metal_init()))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)}))
        elif self.path == "/api/llm/pause":
            self._send(200, json.dumps(llm.pause_llm()))
        elif self.path == "/api/llm/resume":
            self._send(200, json.dumps(llm.resume_llm()))
        elif self.path == "/api/chat":
            # 控制台聊天页代理: 转发给 llama-server 的 OpenAI 兼容接口
            # (浏览器从 8860 直连 8848 会被 CORS 挡,走后端转一手;非流式,一次拿全)
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                port = int(load_config().get("llm_port", 8848))
                try:
                    current = llm.current().get("model") or {}
                except Exception:
                    current = {}
                is_metal = current.get("backend") == "metal"
                is_vmlx = current.get("backend") == "vmlx"
                if is_metal:
                    # 官方 GPT-OSS 服务使用 Responses API；保持历史消息并将正文放入 input。
                    body_obj = {"model": "gpt-oss-20b", "input": d.get("messages", []),
                                "max_output_tokens": int(d.get("max_tokens", current.get("max_tokens", 2048))),
                                "temperature": float(d.get("temperature", current.get("temp", 0.7))),
                                "reasoning": {"effort": current.get("reasoning_level", "low")}}
                    endpoint = "/v1/responses"
                else:
                    body_obj = {"messages": d.get("messages", []),
                                "model": current.get("id") or "local-model",
                                "max_tokens": int(d.get("max_tokens", current.get("max_tokens", 2048))),
                                "temperature": float(d.get("temperature", current.get("temp", 0.7)))}
                    if is_vmlx:
                        # vMLX allows per-request overrides; the project console must
                        # explicitly carry the launch-page thinking switch instead of
                        # falling back to a client/library default.
                        enabled = bool(current.get("thinking", False))
                        body_obj["enable_thinking"] = enabled
                        body_obj["chat_template_kwargs"] = {"enable_thinking": enabled}
                    endpoint = "/v1/chat/completions"
                body = json.dumps(body_obj).encode()
                req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=body,
                                             headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=600).read())
                if is_metal:
                    reply, think = _responses_parts(r)
                else:
                    msg = (r.get("choices") or [{}])[0].get("message") or {}
                    # 推理模型会把思考过程放 reasoning_content,正文在 content
                    reply = (msg.get("content") or "").strip()
                    think = (msg.get("reasoning_content") or "").strip()
                self._send(200, json.dumps({"reply": reply, "think": think}))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/tts/speak":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(tts_speak(d.get("text", ""), d.get("voice", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path.startswith("/api/tts/learn"):
            # 学声音: body 就是音频/视频原始字节(前端直接发 File,不走 multipart),名字+参考文本走 query
            try:
                data = self.rfile.read(int(self.headers["Content-Length"]))
                r = tts_learn(self._qarg("name"), self._qarg("text"), self._qarg("fname"), data)
                # 学习成功且模型就绪 → 顺手生成试听(失败不阻塞,页面可再点"生成试听")
                if r.get("ok") and tts_state().get("ready"):
                    sr = tts_sample(r["name"])
                    if sr.get("error"): r["sample_error"] = sr["error"]
                self._send(200, json.dumps(r))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/tts/delvoice":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(tts_delvoice(d.get("name", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/tts/sample":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(tts_sample(d.get("name", ""), d.get("text", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/tts/rename":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(tts_rename(d.get("old", ""), d.get("new", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path.startswith("/api/wb/"):
            # 分段配音工作台: add/edit/del/regen/split/order/merge/reset 一个入口分发
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                op = self.path[len("/api/wb/"):]
                if   op == "add":   r = wb_add(d.get("text", ""), d.get("voice", ""))
                elif op == "edit":  r = wb_edit(int(d.get("id", 0)), d.get("text"), d.get("voice"))
                elif op == "del":   r = wb_del(int(d.get("id", 0)))
                elif op == "regen": r = wb_regen(int(d.get("id", 0)))
                elif op == "split": r = wb_split(int(d.get("id", 0)), d.get("start", 0), d.get("end", 0))
                elif op == "order": r = wb_order(d.get("ids", []))
                elif op == "merge": r = wb_merge()
                elif op == "reset": r = wb_reset()
                else:               r = {"error": "unknown wb op"}
                self._send(200, json.dumps(r))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
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
                                                       int(d.get("w", 832)), int(d.get("h", 1216)),
                                                       d.get("style", ""), d.get("custom_style", ""),
                                                       d.get("cmode", "i2i"))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/comic/stop":
            COMIC["stop"] = True
            self._send(200, json.dumps({"ok": True}))
        elif self.path.startswith("/api/refs/upload"):
            # 参考图库上传: body 就是图片原始字节(前端直接发 File,不走 multipart)
            try:
                name = self._qarg("name", f"ref_{int(time.time())}.png")
                data = self.rfile.read(int(self.headers["Content-Length"]))
                self._send(200, json.dumps(ref_save(name, data)))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/refs/active":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(ref_set_active(d.get("name", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/refs/delete":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(ref_delete(d.get("name", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
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
                self._send(200, json.dumps(cc_start(d.get("chars", []), panels,
                                                    d.get("style", ""), d.get("custom_style", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cc/flatten":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(cc_flatten(d.get("name", ""), d.get("bubbles", []))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/cc/stop":
            CC["stop"] = True
            self._send(200, json.dumps({"ok": True}))
        elif self.path == "/api/test/create":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(test_create(d)))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/test/control":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(test_control(d.get("cmd", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/test/delete":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(test_delete(d.get("id", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/test/rerun":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(test_rerun(d.get("id", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/test/prompts":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(test_prompts_save(d.get("id", ""), d.get("text", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/create":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_create(d)))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/control":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_control(d.get("cmd", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/resume":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_resume(d.get("id", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/delete":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_delete(d.get("id", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/rerun":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_rerun(d.get("id", ""))))
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}))
        elif self.path == "/api/video-test/prompts":
            try:
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, json.dumps(video_test_prompts_save(d.get("id", ""), d.get("text", ""))))
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
