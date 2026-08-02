#!/usr/bin/env python3
# ============================================================
# vidwf.py · 视频 I2V(图生视频)工作流构建 + 任务提交/轮询
# 打法: LTX-2.3 底座 + 蒸馏LoRA(必串) + 可选风格LoRA,CFG 1.0,8步,原生分辨率。
#   提示词写"动作"不写"场景"(0成本最大提升,见 PLAN.md 工作流研究)。
# 通信: 一律打生视频 ComfyUI(默认 8850),与生图(8849)完全隔离。
# 关键坑(都已规避):
#   - LTXVImgToVideo 的 strength 必填,漏了提交报 400
#   - prompt+seed 相同会命中执行缓存返回空 outputs → 一律随机种子
#   - 模型列表在 ComfyUI 启动时缓存,新放模型须重启 8850 才认
# 依赖: 仅标准库。本模块不 import gen。
# ============================================================
import json, os, random, shutil, threading, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
COMFY_OUT = os.path.join(BASE, "output", "comfy")     # ComfyUI 临时输出(与 8849 同目录)
OUT_VID = os.path.join(BASE, "output", "videos")      # 成品视频(网页可播)

def _vid_port():
    try:
        return int(json.load(open(os.path.join(BASE, "config.json"))).get("vid_port", 8850))
    except Exception:
        return 8850

# ---- 模型/LoRA 注册表(文件名须与 8850 object_info 下拉枚举完全一致) ----
DISTILLED = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"  # 蒸馏LoRA,必串
GEMMA     = "gemma_3_12B_it_fp4_mixed.safetensors"      # 文本编码器1(主)
LTX_ENC   = "ltx-2-3-22b-text_encoder.safetensors"      # 文本编码器2
LTX_VAE   = "ltx-2-3-22b-VAE.safetensors"               # LTX 专用 VAE

UNETS = [
    {"id": "eros",    "unet": "10Eros_v1.4-Q4_K_M.gguf",                "name": "10Eros v1.4", "tag": "动漫专精", "desc": "动漫 NSFW 最强,横评第一"},
    {"id": "pink",    "unet": "PinkCherry_FineTune_Q5_K_M_v1_7-alpha.gguf", "name": "PinkCherry", "tag": "真人最强", "desc": "真人向,与 10Eros 不分上下(16G)"},
    {"id": "sulphur", "unet": "sulphur_dev-Q4_K_M.gguf",                "name": "Sulphur", "tag": "凑合能用", "desc": "通用底模,动漫/真人都还行"},
]
# 风格 LoRA(叠在蒸馏 LoRA 之后,strength 0.8)
LORAS = [
    {"id": "none",   "lora": None, "name": "不叠加", "desc": "只用底模自带效果"},
    {"id": "motion", "lora": "LTX2.3-NSFWMOTION_00750.safetensors", "name": "动作增强", "desc": "打击感/运动更猛(1.1G)"},
    {"id": "furry",  "lora": "ltx-2-2.3-i2v-nsfw-furry-multi-purpose-sex-lora.safetensors", "name": "兽人向", "desc": "furry 题材专用(2.5G)"},
    {"id": "anime",  "lora": "Fantasy_Anime_LTX23.safetensors", "name": "动漫风", "desc": "动漫画风增强(353M)"},
]

def list_unets():
    return UNETS

def list_loras():
    return LORAS

def _find(seq, key, val):
    for e in seq:
        if e.get(key) == val:
            return e
    return None

# ---------------- 工作流构建 ----------------
def build_vid_wf(unet_id, pos, neg, image_name, w, h, frames, fps,
                 lora_id="none", lora_strength=0.8, use_stg=False, steps=8):
    """返回可直接 POST 给 8850 /prompt 的节点图。节点编号用大号防撞。"""
    ue = _find(UNETS, "id", unet_id)
    if not ue:
        raise ValueError(f"未知视频模型: {unet_id}")
    le = _find(LORAS, "id", lora_id) or LORAS[0]
    seed = random.randint(0, 2**31 - 1)   # 随机种子,防 ComfyUI 执行缓存空转
    n = {}
    # 底座 + 蒸馏LoRA(必串) + 可选风格LoRA
    n["100"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": ue["unet"]}}
    n["101"] = {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["100", 0], "lora_name": DISTILLED, "strength_model": 1.0}}
    model_src = ["101", 0]
    if le["lora"]:
        n["102"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["101", 0], "lora_name": le["lora"], "strength_model": float(lora_strength)}}
        model_src = ["102", 0]
    # 双文本编码器(gemma + ltx-2-3, type=ltxv)
    n["103"] = {"class_type": "DualCLIPLoader",
                "inputs": {"clip_name1": GEMMA, "clip_name2": LTX_ENC, "type": "ltxv"}}
    n["104"] = {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["103", 0]}}
    n["105"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["103", 0]}}
    n["106"] = {"class_type": "LTXVConditioning",
                "inputs": {"positive": ["104", 0], "negative": ["105", 0], "frame_rate": float(fps)}}
    n["107"] = {"class_type": "VAELoader", "inputs": {"vae_name": LTX_VAE}}
    # 源图预处理(压缩降噪) → I2V(内部自带 VAEEncode,strength 必填)
    n["108"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    n["109"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["108", 0], "img_compression": 35}}
    n["110"] = {"class_type": "LTXVImgToVideo",
                "inputs": {"positive": ["106", 0], "negative": ["106", 1], "vae": ["107", 0],
                           "image": ["109", 0], "width": int(w), "height": int(h),
                           "length": int(frames), "batch_size": 1, "strength": 1.0}}
    # 采样: 随机噪声 + euler + (CFG 1.0 或 STG) + LTX 调度(8步)
    n["111"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    n["112"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    n["113"] = {"class_type": "LTXVScheduler",
                "inputs": {"steps": int(steps), "max_shift": 2.05, "base_shift": 0.95,
                           "stretch": True, "terminal": 0.1, "latent": ["110", 2]}}
    if use_stg:
        cnt = int(steps) + 1  # per-step 列表长度须等于 sigmas 数(steps+1,末位为0)
        cfg_s = ", ".join(["1.0"] * cnt)
        stg = ["2.0", "1.5"] + ["1.0"] * (cnt - 2)
        n["114"] = {"class_type": "LTX2STGGuider",
                    "inputs": {"model": model_src, "positive": ["110", 0], "negative": ["110", 1],
                               "sigmas": ["113", 0], "cfg_per_step": cfg_s,
                               "stg_scale_per_step": ", ".join(stg[:cnt]),
                               "stg_rescale_per_step": ", ".join(["1.0"] * cnt)}}
    else:
        n["114"] = {"class_type": "CFGGuider",
                    "inputs": {"model": model_src, "positive": ["110", 0], "negative": ["110", 1], "cfg": 1.0}}
    n["115"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["111", 0], "guider": ["114", 0], "sampler": ["112", 0],
                           "sigmas": ["113", 0], "latent_image": ["110", 2]}}
    # 分块解码(省显存) → 组视频 → 存 mp4
    n["116"] = {"class_type": "LTXVTiledVAEDecode",
                "inputs": {"vae": ["107", 0], "latents": ["115", 0],
                           "horizontal_tiles": 1, "vertical_tiles": 1, "overlap": 1, "last_frame_fix": False}}
    n["117"] = {"class_type": "CreateVideo", "inputs": {"images": ["116", 0], "fps": float(fps)}}
    n["118"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["117", 0], "filename_prefix": "ivs_vid", "format": "mp4", "codec": "h264"}}
    return n

# ---------------- 提交 / 轮询 ----------------
VTASKS = {}
VLOCK = threading.Lock()

def _get(path, timeout=10):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{_vid_port()}{path}", timeout=timeout))

def submit_vid(name, **kw):
    """构建工作流并提交到 8850,返回 prompt_id。name 是成品文件名(不含扩展名)。"""
    wf = build_vid_wf(**kw)
    req = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    with VLOCK:
        VTASKS[pid] = {"name": name, "t0": time.time(), "done": False, "error": "", "url": ""}
    threading.Thread(target=_wait_done, args=(pid,), daemon=True).start()
    return pid

def _wait_done(pid):
    deadline = time.time() + 2400   # 视频较慢,允许 40 分钟
    while time.time() < deadline:
        time.sleep(4)
        try:
            hist = _get(f"/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            continue
        st = hist[pid].get("status", {})
        if st.get("status_str") == "error":
            with VLOCK:
                VTASKS[pid]["error"] = "生成失败:多半是缺模型/源图问题,或提示词被拦"
            return
        if st.get("completed"):
            fname = None
            for out in hist[pid].get("outputs", {}).values():
                for key in ("videos", "gifs", "images"):
                    for v in out.get(key, []):
                        fname = os.path.join(COMFY_OUT, v.get("subfolder", ""), v["filename"])
            if fname and os.path.exists(fname):
                os.makedirs(OUT_VID, exist_ok=True)
                dest = os.path.join(OUT_VID, VTASKS[pid]["name"] + ".mp4")
                shutil.copy2(fname, dest)
                try: os.remove(fname)
                except OSError: pass
                with VLOCK:
                    VTASKS[pid]["done"] = True
                    VTASKS[pid]["url"] = "/vout/" + VTASKS[pid]["name"] + ".mp4"
            else:
                with VLOCK:
                    VTASKS[pid]["error"] = "找不到输出视频文件"
            return
    with VLOCK:
        VTASKS[pid]["error"] = "超时(40分钟)"

def poll_vid(pid):
    """返回 {done,error,url,state,pos,run_elapsed},供 /api/vid/poll。"""
    with VLOCK:
        t = VTASKS.get(pid)
    if not t:
        return {"error": "未知任务"}
    if t["done"] or t["error"]:
        return {"done": t["done"], "error": t["error"], "url": t["url"]}
    state, pos, run_elapsed = "queued", 0, 0
    try:
        q = _get("/queue")
        run_ids = [e[1] for e in q.get("queue_running", [])]
        pen_ids = [e[1] for e in q.get("queue_pending", [])]
        if pid in run_ids:
            state = "running"
            with VLOCK:
                if not t.get("t_start"):
                    t["t_start"] = time.time()
                run_elapsed = time.time() - t["t_start"]
        elif pid in pen_ids:
            pos = pen_ids.index(pid) + 1
    except Exception:
        pass
    return {"done": False, "error": "", "url": "", "state": state, "pos": pos, "run_elapsed": run_elapsed}
