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
import json, os, random, shutil, subprocess, threading, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
COMFY_OUT = os.path.join(BASE, "output", "comfy")     # ComfyUI 临时输出(与 8849 同目录)
OUT_VID = os.path.join(BASE, "output", "videos")      # 成品视频(网页可播)
VID_AUDIO = os.path.join(BASE, "output", "video_audio")

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
LTX_AUDIO_VAE = "ltx-2-3-22b-audio_vae_with_vocoder.safetensors"  # LTX 原生音效 VAE

# ---- Sulphur 2 (smthemex 工作流) 专用组件 ----
SM_GGUF       = "sulphur_distil-Q6_K.gguf"              # Sulphur 2 主模型(GGUF)
SM_CLIP_GGUF  = "gemma-3-12b-it-qat-Q4_0.gguf"          # smthemex GGUF 文本编码器
SM_CONNECTOR  = "connector.safetensors"                  # smthemex 连接器
SM_VIDEO_VAE  = "ltx-2.3-22b-distilled_video_vae.safetensors"  # 蒸馏视频 VAE(通用)
SM_AUDIO_VAE  = "ltx-2.3-22b-distilled_audio_vae.safetensors"  # 蒸馏音频 VAE(通用)
SM_DISTILLED  = DISTILLED                                # 蒸馏 LoRA(smthemex 也需串)

ANIME_POSITIVE = (
    "2D anime, hand-drawn animation, clean line art, cel shading, flat colors, "
    "anime character design, illustrated background, consistent 2D proportions, "
    "non-photorealistic, no 3D rendering"
)
ANIME_NEGATIVE = (
    "3D, 3D CGI, CGI, realistic rendering, photorealistic, live action, "
    "plastic skin, doll-like face, game character render, uncanny valley"
)

UNETS = [
    {"id": "eros",    "unet": "10Eros_v1.4-Q4_K_M.gguf",                "name": "10Eros v1.4", "tag": "半写实/人物", "desc": "把图片变成视频；真人和半写实更稳，纯动漫长视频可能慢慢变真人脸。", "sec": 240, "t2v": True},
    {"id": "pink",    "unet": "PinkCherry_FineTune_Q5_K_M_v1_7-alpha.gguf", "name": "PinkCherry", "tag": "人物微调", "desc": "人物向微调版；具体偏真人还是动漫要用同一张图实测，alpha 版可能不够稳定。", "sec": 300, "t2v": True},
    {"id": "sulphur2", "unet": SM_GGUF, "name": "Sulphur 2", "tag": "无审查底座", "desc": "Sulphur 2 无审查视频模型，基于 LTX 2.3 深度微调，12.5万视频样本训练，画质和动态全面提升。走 smthemex 工作流。", "sec": 180, "t2v": True, "sm": True},
]
# 风格 LoRA(叠在蒸馏 LoRA 之后,strength 0.8)
LORAS = [
    {"id": "none",   "lora": None, "name": "不叠加", "desc": "不额外挂效果，只用视频底座自己的风格。"},
    {"id": "motion", "lora": "LTX2.3-NSFWMOTION_00750.safetensors", "name": "动作增强", "desc": "让跑步、打斗、转身等动作幅度更明显；不负责把真人变动漫。"},
    {"id": "furry",  "lora": "ltx-2-2.3-i2v-nsfw-furry-multi-purpose-sex-lora.safetensors", "name": "兽人向", "desc": "给兽人、动物拟人和特殊成人题材用；普通人物不建议默认打开。"},
    {"id": "anime",  "lora": "Fantasy_Anime_LTX23.safetensors", "name": "动漫/幻想风", "desc": "尽量加强二次元、幻想、赛璐璐画面；是画风外挂，不是独立视频模型。"},
    {"id": "ai_anime", "lora": "h-anime4.comfy.safetensors", "name": "AI动漫风", "desc": "LTX2.3 动漫风格 AI 短视频生成 LoRA，偏 2.5D 动漫画面，适合动漫角色动态化。"},
    {"id": "retro90", "lora": "anime90s-step00053000.comfy.safetensors", "name": "90年代复古动画", "desc": "90年代复古动画风格 LoRA，赛璐璐手绘质感、怀旧色调，经典日本老动画画风。"},
    {"id": "dmd",    "lora": "LTX2.3_DMD_v2_avgrank86_audio160_L80-D20.safetensors", "name": "DMD加速(实验)", "desc": "DaSiWa DMD 蒸馏加速 LoRA，可大幅减少采样步数；与 Sulphur 2 自带蒸馏可能冲突，建议先用原版底座测试。"},
]

def list_unets():
    return [dict(model) for model in UNETS]

def list_loras():
    return LORAS

def balanced_segments(duration, max_segment=6.0):
    """把总时长均分成若干段，每段不超过安全上限。"""
    duration = max(0.001, float(duration))
    max_segment = max(0.5, float(max_segment))
    count = max(1, int((duration + max_segment - 0.000001) // max_segment))
    base = duration / count
    parts = [round(base, 3) for _ in range(count)]
    parts[-1] = round(duration - sum(parts[:-1]), 3)
    return parts

def apply_style_prompt(pos, neg="", style_2d=False):
    """2D 模式统一追加风格约束，保留用户原始镜头和动作描述。"""
    pos = str(pos or "").strip()
    neg = str(neg or "").strip()
    if not style_2d:
        return pos, neg
    pos = ", ".join(part for part in (pos, ANIME_POSITIVE) if part)
    neg = ", ".join(part for part in (neg, ANIME_NEGATIVE) if part)
    return pos, neg

def _find(seq, key, val):
    for e in seq:
        if e.get(key) == val:
            return e
    return None

# ---------------- 工作流构建 ----------------
def build_vid_wf(unet_id, pos, neg, image_name, w, h, frames, fps,
                 lora_id="none", lora_ids=None, lora_strength=0.8, use_stg=False, steps=8,
                 native_audio=False, style_2d=False):
    """返回可直接 POST 给 8850 /prompt 的节点图。节点编号用大号防撞。"""
    ue = _find(UNETS, "id", unet_id)
    if not ue:
        raise ValueError(f"未知视频模型: {unet_id}")
    # Sulphur 2 走 smthemex 工作流
    if ue.get("sm"):
        return build_vid_wf_sm(pos, neg, image_name, w, h, frames, fps,
                               lora_id=lora_id, lora_ids=lora_ids, lora_strength=lora_strength,
                               use_stg=use_stg, steps=steps, native_audio=native_audio, style_2d=style_2d)
    # LTX 节点要求宽高按 32 对齐。
    size_step = 32
    w = max(64, (int(w) // size_step) * size_step)
    h = max(64, (int(h) // size_step) * size_step)
    frames = max(9, int(frames))
    fps = max(1.0, float(fps))
    # 兼容旧任务的单个 lora_id；新任务可按顺序传多个 lora_ids。
    if lora_ids is None:
        lora_ids = [lora_id]
    elif isinstance(lora_ids, str):
        lora_ids = [lora_ids]
    lora_ids = [str(x) for x in lora_ids if str(x) and str(x) != "none"]
    if style_2d and "anime" not in lora_ids:
        lora_ids.insert(0, "anime")
    pos, neg = apply_style_prompt(pos, neg, style_2d)
    seed = random.randint(0, 2**31 - 1)   # 随机种子,防 ComfyUI 执行缓存空转
    n = {}
    # 底座 + 蒸馏LoRA(必串) + 可选风格LoRA
    n["100"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": ue["unet"]}}
    n["101"] = {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["100", 0], "lora_name": DISTILLED, "strength_model": 1.0}}
    model_src = ["101", 0]
    for index, selected_id in enumerate(lora_ids, start=1):
        le = _find(LORAS, "id", selected_id)
        if not le or not le.get("lora"):
            continue
        node_id = str(129 + index)
        n[node_id] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model_src, "lora_name": le["lora"],
                                 "strength_model": float(lora_strength)}}
        model_src = [node_id, 0]
    # 双文本编码器(gemma + ltx-2-3, type=ltxv)
    n["103"] = {"class_type": "DualCLIPLoader",
                "inputs": {"clip_name1": GEMMA, "clip_name2": LTX_ENC, "type": "ltxv"}}
    n["104"] = {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["103", 0]}}
    n["105"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["103", 0]}}
    n["106"] = {"class_type": "LTXVConditioning",
                "inputs": {"positive": ["104", 0], "negative": ["105", 0], "frame_rate": float(fps)}}
    n["107"] = {"class_type": "VAELoader", "inputs": {"vae_name": LTX_VAE}}
    # 有源图走 I2V；无源图使用空视频 latent，是真正的 T2V。
    if image_name:
        n["108"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        n["109"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["108", 0], "img_compression": 35}}
        n["110"] = {"class_type": "LTXVImgToVideo",
                    "inputs": {"positive": ["106", 0], "negative": ["106", 1], "vae": ["107", 0],
                               "image": ["109", 0], "width": int(w), "height": int(h),
                               "length": int(frames), "batch_size": 1, "strength": 1.0}}
        positive_src, negative_src, latent_src = ["110", 0], ["110", 1], ["110", 2]
    else:
        n["110"] = {"class_type": "EmptyLTXVLatentVideo",
                    "inputs": {"width": int(w), "height": int(h), "length": int(frames), "batch_size": 1}}
        positive_src, negative_src, latent_src = ["106", 0], ["106", 1], ["110", 0]
    # 采样: 随机噪声 + euler + (CFG 1.0 或 STG) + LTX 调度(8步)
    n["111"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    n["112"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    n["113"] = {"class_type": "LTXVScheduler",
                "inputs": {"steps": int(steps), "max_shift": 2.05, "base_shift": 0.95,
                           "stretch": True, "terminal": 0.1, "latent": latent_src}}
    sample_latent = latent_src
    audio_decode_src = None
    if native_audio:
        # LTX 2.3 原生音效分支：声音不是后期贴上去，而是和视频 latent 一起采样。
        n["119"] = {"class_type": "LTXVAudioVAELoader",
                    "inputs": {"ckpt_name": LTX_AUDIO_VAE}}
        n["120"] = {"class_type": "LTXVEmptyLatentAudio",
                     "inputs": {"frames_number": int(frames), "frame_rate": int(round(fps)),
                                "batch_size": 1, "audio_vae": ["119", 0]}}
        n["121"] = {"class_type": "LTXVConcatAVLatent",
                     "inputs": {"video_latent": latent_src, "audio_latent": ["120", 0]}}
        sample_latent = ["121", 0]
        n["113"]["inputs"]["latent"] = sample_latent
    if use_stg:
        cnt = int(steps) + 1  # per-step 列表长度须等于 sigmas 数(steps+1,末位为0)
        cfg_s = ", ".join(["1.0"] * cnt)
        stg = ["2.0", "1.5"] + ["1.0"] * (cnt - 2)
        n["114"] = {"class_type": "LTX2STGGuider",
                    "inputs": {"model": model_src, "positive": positive_src, "negative": negative_src,
                               "sigmas": ["113", 0], "cfg_per_step": cfg_s,
                               "stg_scale_per_step": ", ".join(stg[:cnt]),
                               "stg_rescale_per_step": ", ".join(["1.0"] * cnt)}}
    else:
        n["114"] = {"class_type": "CFGGuider",
                    "inputs": {"model": model_src, "positive": positive_src, "negative": negative_src, "cfg": 1.0}}
    n["115"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["111", 0], "guider": ["114", 0], "sampler": ["112", 0],
                               "sigmas": ["113", 0], "latent_image": sample_latent}}
    # 分块解码(省显存) → 组视频 → 存 mp4
    video_sample_src = ["115", 0]
    if native_audio:
        n["122"] = {"class_type": "LTXVSeparateAVLatent",
                     "inputs": {"av_latent": ["115", 0]}}
        video_sample_src = ["122", 0]
        audio_decode_src = ["122", 1]
        n["123"] = {"class_type": "LTXVAudioVAEDecode",
                     "inputs": {"samples": audio_decode_src, "audio_vae": ["119", 0]}}
    n["116"] = {"class_type": "LTXVTiledVAEDecode",
                "inputs": {"vae": ["107", 0], "latents": video_sample_src,
                           "horizontal_tiles": 1, "vertical_tiles": 1, "overlap": 1, "last_frame_fix": False}}
    create_inputs = {"images": ["116", 0], "fps": float(fps)}
    if native_audio:
        create_inputs["audio"] = ["123", 0]
    n["117"] = {"class_type": "CreateVideo", "inputs": create_inputs}
    n["118"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["117", 0], "filename_prefix": "ivs_vid", "format": "mp4", "codec": "h264"}}
    return n


# ---------------- Sulphur 2 (smthemex) 工作流 ----------------
def build_vid_wf_sm(pos, neg, image_name, w, h, frames, fps,
                    lora_id="none", lora_ids=None, lora_strength=0.8, use_stg=False, steps=8,
                    native_audio=False, style_2d=False):
    """Sulphur 2 专用工作流，走 smthemex (ComfyUI_LTX2_SM) 节点。"""
    # LTX 节点要求宽高按 32 对齐。
    size_step = 32
    w = max(64, (int(w) // size_step) * size_step)
    h = max(64, (int(h) // size_step) * size_step)
    frames = max(9, int(frames))
    fps = max(1.0, float(fps))
    # 兼容旧任务的单个 lora_id
    if lora_ids is None:
        lora_ids = [lora_id]
    elif isinstance(lora_ids, str):
        lora_ids = [lora_ids]
    lora_ids = [str(x) for x in lora_ids if str(x) and str(x) != "none"]
    if style_2d and "anime" not in lora_ids:
        lora_ids.insert(0, "anime")
    pos, neg = apply_style_prompt(pos, neg, style_2d)
    seed = random.randint(0, 2**31 - 1)
    # 取第一个风格 LoRA（smthemex 模型节点只支持单 LoRA 槽位）
    style_lora = "none"
    if lora_ids:
        le = _find(LORAS, "id", lora_ids[0])
        if le and le.get("lora"):
            style_lora = le["lora"]
    n = {}
    # 200: 加载模型 (GGUF + 蒸馏 LoRA + 可选风格 LoRA)
    n["200"] = {"class_type": "LTX2_SM_Model",
                "inputs": {"dit": "none", "gguf": SM_GGUF,
                           "distilled_lora": SM_DISTILLED, "lora": style_lora,
                           "sampling_mode": "distilled", "offload": True}}
    # 201: 加载 GGUF 文本编码器 + 连接器
    n["201"] = {"class_type": "LTX2_SM_Clip",
                "inputs": {"clip": SM_CLIP_GGUF, "connector": SM_CONNECTOR, "infer_device": "cpu"}}
    # 202: 编码提示词
    n["202"] = {"class_type": "LTX2_SM_ENCODER",
                "inputs": {"clip": ["201", 0], "prompt": pos, "negative_prompt": neg,
                           "enhance_prompt": False, "save_emb": True,
                           "streaming_prefetch_count": 1}}
    # 203: 加载视频 VAE
    n["203"] = {"class_type": "LTX2_SM_VAE",
                "inputs": {"vae": SM_VIDEO_VAE}}
    # 204: 生成潜空间 (I2V 或 T2V)
    latents_inputs = {"width": int(w), "height": int(h), "num_frames": int(frames),
                      "frame_rate": float(fps), "strength": 1.0,
                      "audio_start_time": 0.0, "audio_max_duration": 0.0,
                      "encoder": ["203", 1]}
    if image_name:
        n["205"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        latents_inputs["image"] = ["205", 0]
    if native_audio:
        n["206"] = {"class_type": "LTX2_SM_AUDIO_VAE",
                    "inputs": {"audio_vae": SM_AUDIO_VAE}}
        latents_inputs["a_encoder"] = ["206", 1]
    n["204"] = {"class_type": "LTX2_LATENTS", "inputs": latents_inputs}
    # 207: 采样
    sampler_inputs = {"model": ["200", 0], "latents": ["204", 0],
                      "steps": int(steps), "seed": seed,
                      "video_cfg_guidance_scale": 1.0, "video_stg_guidance_scale": 0.0,
                      "video_rescale_scale": 0.0, "a2v_guidance_scale": 1.0,
                      "video_skip_step": 0, "video_stg_blocks": -1,
                      "audio_cfg_guidance_scale": 1.0, "audio_stg_guidance_scale": 0.0,
                      "audio_rescale_scale": 0.0, "v2a_guidance_scale": 1.0,
                      "audio_skip_step": 0, "audio_stg_blocks": -1,
                      "block_group_size": 2, "spatial_upsampler": "none",
                      "positive": ["202", 0], "negative": ["202", 1]}
    if use_stg:
        sampler_inputs["video_stg_guidance_scale"] = 2.0
    n["207"] = {"class_type": "LTX2_SM_KSampler", "inputs": sampler_inputs}
    # 208: 视频解码
    n["208"] = {"class_type": "LTX2_DECO_VIDEO",
                "inputs": {"decoder": ["203", 0], "latent": ["207", 0], "tile": True}}
    # 209: 组视频 + 保存
    create_inputs = {"images": ["208", 0], "fps": float(fps)}
    if native_audio:
        n["210"] = {"class_type": "LTX2_DECO_AUDIO",
                    "inputs": {"a_decoder": ["206", 0], "audio_latents": ["207", 1]}}
        create_inputs["audio"] = ["210", 0]
    n["209"] = {"class_type": "CreateVideo", "inputs": create_inputs}
    n["211"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["209", 0], "filename_prefix": "ivs_vid", "format": "mp4", "codec": "h264"}}
    return n


# ---------------- 提交 / 轮询 ----------------
VTASKS = {}
VLOCK = threading.Lock()

def _submit_prompt_raw(wf):
    req = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]

def _history_video(pid):
    """取 ComfyUI 历史里的第一个视频文件。"""
    hist = _get(f"/history/{pid}")
    item = hist.get(pid)
    if not item:
        return None, False, ""
    st = item.get("status", {})
    if st.get("status_str") == "error":
        # 新版 ComfyUI 会把节点异常放在 messages；取一条短错误给测试场展示。
        detail = ""
        for message in reversed(st.get("messages") or []):
            text = json.dumps(message, ensure_ascii=False)
            marker = 'exception_message'
            if marker in text:
                try:
                    detail = str(message[1].get("exception_message") or "")
                except Exception:
                    detail = text
                break
        detail = " ".join(detail.split())[:360]
        return None, True, "ComfyUI 工作流执行失败" + ("：" + detail if detail else "")
    if not st.get("completed"):
        return None, False, ""
    for out in item.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for value in out.get(key, []):
                path = os.path.join(COMFY_OUT, value.get("subfolder", ""), value["filename"])
                if os.path.exists(path):
                    return path, True, ""
    return None, True, "找不到输出视频文件"

def _wait_prompt_file(pid, timeout=2400):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(4)
        try:
            path, finished, error = _history_video(pid)
        except Exception:
            continue
        if error:
            return None, error
        if finished:
            return path, ""
    return None, "超时(40分钟)"

def _run_postprocess(dest, pp):
    interpolate_fps = float(pp.get("interpolate_fps") or (60 if pp.get("interpolate") else 0))
    audio = pp.get("audio_path")
    native_audio = bool(pp.get("native_audio"))
    if not (interpolate_fps or audio or native_audio):
        return
    processed = dest + ".processed.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", dest]
    vf = (f"minterpolate=fps={interpolate_fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
          if interpolate_fps else None)
    if audio and os.path.isfile(audio):
        cmd += ["-i", audio]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-map", "0:v:0"]
    if audio and os.path.isfile(audio):
        cmd += ["-map", "1:a:0?", "-shortest", "-c:a", "aac", "-b:a", "192k"]
    elif native_audio:
        cmd += ["-map", "0:a:0?", "-af", "volume=20dB", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", processed]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=900)
    if result.returncode != 0 or not os.path.exists(processed):
        raise RuntimeError((result.stderr or b"").decode(errors="replace")[-500:])
    os.replace(processed, dest)

def submit_long_vid(name, duration, postprocess=None, **kw):
    """普通视频页的长片入口：均分成不超过6秒的小段，再拼成一个成片。"""
    parts = balanced_segments(duration, 6.0)
    parent = "long_" + str(int(time.time() * 1000))
    with VLOCK:
        VTASKS[parent] = {"name": name, "t0": time.time(), "done": False, "error": "", "url": "",
                          "postprocess": postprocess or {}, "state": "queued", "parts": parts, "part_done": 0}
    threading.Thread(target=_run_long_vid, args=(parent, parts, kw), daemon=True).start()
    return parent

def _run_long_vid(parent, parts, kw):
    work = os.path.join(OUT_VID, ".segments", parent)
    os.makedirs(work, exist_ok=True)
    paths = []
    try:
        with VLOCK:
            VTASKS[parent]["state"] = "running"
        total = sum(parts)
        continue_image = kw.get("image_name", "")
        for index, seconds in enumerate(parts):
            with VLOCK:
                if VTASKS.get(parent, {}).get("error") == "已停止":
                    return
                VTASKS[parent]["part_done"] = index
            fps = float(kw.get("fps", 24))
            frames = max(9, round(seconds * min(24.0, fps)))
            frames = max(9, round((frames - 1) / 8) * 8 + 1)
            child_kw = dict(kw, image_name=continue_image, frames=frames, fps=min(24.0, fps))
            pid = _submit_prompt_raw(build_vid_wf(**child_kw))
            with VLOCK:
                VTASKS[parent]["child"] = pid
            source, error = _wait_prompt_file(pid)
            if error:
                raise RuntimeError(error)
            target = os.path.join(work, f"segment_{index + 1:03d}.mp4")
            shutil.copy2(source, target)
            try: os.remove(source)
            except OSError: pass
            paths.append(target)
            # 后续片段接上一段最后一帧；没有源图的纯文字模式则保持每段独立生成。
            if continue_image:
                last_frame = os.path.join(work, f"last_{index + 1:03d}.png")
                probe = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-0.05", "-i", target, "-frames:v", "1", last_frame],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
                if probe.returncode == 0 and os.path.isfile(last_frame):
                    try:
                        boundary = "----ivs-long"
                        raw = open(last_frame, "rb").read()
                        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(last_frame)}\"\r\nContent-Type: image/png\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
                        request = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/upload/image", data=body,
                                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                        continue_image = json.load(urllib.request.urlopen(request, timeout=120)).get("name") or continue_image
                    except Exception:
                        pass
            with VLOCK:
                VTASKS[parent]["part_done"] = index + 1
        dest = os.path.join(OUT_VID, str(VTASKS[parent]["name"]) + ".mp4")
        concat = os.path.join(work, "concat.txt")
        with open(concat, "w", encoding="utf-8") as handle:
            for path in paths:
                safe_path = os.path.abspath(path).replace("'", "'\\''")
                handle.write("file '" + safe_path + "'\n")
        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", dest],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=900)
        if result.returncode != 0 or not os.path.exists(dest):
            raise RuntimeError((result.stderr or b"").decode(errors="replace")[-500:])
        _run_postprocess(dest, VTASKS[parent].get("postprocess") or {})
        with VLOCK:
            VTASKS[parent].update(done=True, state="done", url="/vout/" + os.path.basename(dest))
    except Exception as exc:
        with VLOCK:
            if parent in VTASKS and not VTASKS[parent].get("error"):
                VTASKS[parent].update(error=str(exc), state="failed")

def list_tasks():
    """返回普通视频任务的可序列化快照。只读队列，不启动或加载任何模型。"""
    with VLOCK:
        rows = []
        for pid, task in VTASKS.items():
            state = "failed" if task.get("error") else ("done" if task.get("done") else (task.get("state") or "queued"))
            child = str(task.get("child") or "")
            rows.append({
                "id": pid, "engine": "video", "name": task.get("name", ""),
                "state": state, "error": task.get("error", ""),
                "url": task.get("url", ""), "created": float(task.get("t0") or 0),
                "started": float(task.get("t_start") or 0),
                "progress": {"done": int(task.get("part_done") or 0), "total": len(task.get("parts") or [])},
                "child": child,
            })
    # ComfyUI 是唯一能准确告诉我们“正在跑/排队”的地方；查询失败时保留 queued。
    try:
        q = _get("/queue", timeout=3)
        running = {str(e[1]) for e in q.get("queue_running", [])}
        pending = {str(e[1]) for e in q.get("queue_pending", [])}
        for row in rows:
            if row["state"] in ("done", "failed"):
                continue
            if row["id"] in running or row.get("child") in running:
                row["state"] = "running"
            elif row["id"] in pending or row.get("child") in pending:
                row["state"] = "queued"
    except Exception:
        pass
    return rows

def _get(path, timeout=10):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{_vid_port()}{path}", timeout=timeout))

def submit_vid(name, postprocess=None, **kw):
    """构建工作流并提交到 8850,返回 prompt_id。name 是成品文件名(不含扩展名)。"""
    wf = build_vid_wf(**kw)
    req = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    with VLOCK:
        VTASKS[pid] = {"name": name, "t0": time.time(), "done": False, "error": "", "url": "",
                       "postprocess": postprocess or {}}
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
            detail = ""
            for message in reversed(st.get("messages") or []):
                try:
                    detail = str(message[1].get("exception_message") or "")
                except Exception:
                    continue
                if detail:
                    break
            detail = " ".join(detail.split())[:360]
            with VLOCK:
                VTASKS[pid]["error"] = "ComfyUI 工作流执行失败" + ("：" + detail if detail else "")
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
                pp = VTASKS[pid].get("postprocess") or {}
                interpolate_fps = float(pp.get("interpolate_fps") or (60 if pp.get("interpolate") else 0))
                native_audio = bool(pp.get("native_audio"))
                if interpolate_fps or pp.get("audio_path") or native_audio:
                    processed = dest + ".processed.mp4"
                    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", dest]
                    vf = (f"minterpolate=fps={interpolate_fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
                          if interpolate_fps else None)
                    audio = pp.get("audio_path")
                    if audio and os.path.isfile(audio):
                        cmd += ["-i", audio]
                    if vf:
                        cmd += ["-vf", vf]
                    cmd += ["-map", "0:v:0"]
                    if audio and os.path.isfile(audio):
                        cmd += ["-map", "1:a:0?", "-shortest", "-c:a", "aac", "-b:a", "192k"]
                    elif native_audio:
                        cmd += ["-map", "0:a:0?", "-af", "volume=20dB", "-c:a", "aac", "-b:a", "192k"]
                    else:
                        cmd += ["-an"]
                    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", processed]
                    try:
                        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=900)
                        if r.returncode != 0 or not os.path.exists(processed):
                            raise RuntimeError((r.stderr or b"").decode(errors="replace")[-500:])
                        os.replace(processed, dest)
                    except Exception as e:
                        with VLOCK:
                            VTASKS[pid]["error"] = "视频后处理失败: " + str(e)
                        try: os.remove(processed)
                        except OSError: pass
                        return
                with VLOCK:
                    VTASKS[pid]["done"] = True
                    VTASKS[pid]["url"] = "/vout/" + VTASKS[pid]["name"] + ".mp4"
                audio_path = pp.get("audio_path")
                if audio_path:
                    try: os.remove(audio_path)
                    except OSError: pass
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
        child = ""
        with VLOCK:
            child = str(t.get("child") or "")
        q = _get("/queue")
        run_ids = [e[1] for e in q.get("queue_running", [])]
        pen_ids = [e[1] for e in q.get("queue_pending", [])]
        if pid in run_ids or child in run_ids:
            state = "running"
            with VLOCK:
                if not t.get("t_start"):
                    t["t_start"] = time.time()
                run_elapsed = time.time() - t["t_start"]
        elif pid in pen_ids or child in pen_ids:
            pos = (pen_ids.index(pid if pid in pen_ids else child) + 1)
    except Exception:
        pass
    return {"done": False, "error": "", "url": "", "state": state, "pos": pos, "run_elapsed": run_elapsed}


def cancel_vid(pid):
    """中断正在运行的视频，并从 ComfyUI 待执行队列移除。"""
    with VLOCK:
        child = str((VTASKS.get(pid) or {}).get("child") or "")
    target_ids = [x for x in (pid, child) if x]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/interrupt",
                                     data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass
    try:
        body = json.dumps({"delete": target_ids}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{_vid_port()}/queue",
                                     data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass
    with VLOCK:
        if pid in VTASKS:
            VTASKS[pid]["error"] = "已停止"
            VTASKS[pid]["state"] = "failed"
