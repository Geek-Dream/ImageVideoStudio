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
def known_models():
    """读 models.json(随项目发布) + models.local.json(本机私有,不上传),按文件名给出友好名称/参数。"""
    merged = {}
    for fp in (MODELS_JSON, os.path.join(BASE, "models.local.json")):
        if os.path.exists(fp):
            try:
                merged.update(json.load(open(fp)).get("models", {}))
            except Exception:
                pass
    return merged

def list_models():
    """扫描 models/image 里的单文件模型,每个就是一个可选模型。"""
    known = known_models()
    out = []
    if not os.path.isdir(MODELS_IMG):
        return out
    for fn in sorted(os.listdir(MODELS_IMG)):
        fp = os.path.join(MODELS_IMG, fn)
        if not os.path.isfile(fp) or not fn.lower().endswith(IMG_EXTS):
            continue
        meta = known.get(fn, {})
        out.append({
            "id": fn,
            "name": meta.get("name", fn),
            "sec": int(meta.get("sec", 60)),
            "desc": meta.get("desc", "单文件模型(checkpoint)"),
        })
    return out

def has_controlnet():
    if not os.path.isdir(MODELS_CN):
        return False
    return any(f.lower().endswith(IMG_EXTS) for f in os.listdir(MODELS_CN))

def first_controlnet():
    for f in sorted(os.listdir(MODELS_CN)):
        if f.lower().endswith(IMG_EXTS):
            return f
    return ""

# ---------------- ComfyUI 工作流(通用 checkpoint) ----------------
def sampler_cfg():
    return {"steps": 25, "cfg": 6.0, "sampler_name": "euler_ancestral", "scheduler": "normal"}

def build_wf(model, pos, neg, w, h, seed, mode="t2i", ref=None, mask=None,
             strength=0.6, scale=2.0, ctype="openpose"):
    """model 即 checkpoint 文件名。loader 固定 CheckpointLoaderSimple(适合 SDXL/SD1.5 等单文件模型)。"""
    ckpt = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}}}
    clip, vae, mdl = ["1", 1], ["1", 2], ["1", 0]
    cfg = sampler_cfg()
    wf = dict(ckpt)
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": clip}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": clip}}
    pos_out = ["3", 0]

    if mode == "inpaint":
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
        wf["8"] = {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["6", 0], "vae": vae, "mask": ["7", 0], "grow_mask_by": 6}}
        wf["9"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": mdl, "positive": pos_out, "negative": ["4", 0], "latent_image": ["8", 0]}}
        wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae}}
        wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["10", 0]}}
        return wf

    if mode == "upscale":
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["6", 0]}}
        wf["8"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": vae}}
        wf["9"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": strength, "model": mdl, "positive": pos_out, "negative": ["4", 0], "latent_image": ["8", 0]}}
        wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae}}
        wf["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["10", 0]}}
        return wf

    if mode == "pose":
        cn = first_controlnet()
        if not cn:
            raise ValueError("未找到 ControlNet 辅助模型,请先放入 models/image/controlnet/")
        wf["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["7"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn}}
        wf["8"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["7", 0], "type": ctype}}
        wf["9"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "control_net": ["8", 0], "image": ["6", 0], "strength": strength, "start_percent": 0.0, "end_percent": 1.0}}
        wf["10"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": mdl, "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["5", 0]}}
        wf["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": vae}}
        wf["12"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["11", 0]}}
        return wf

    if mode == "i2i" and ref:
        wf["5"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": vae}}
        wf["7"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": strength, "model": mdl, "positive": pos_out, "negative": ["4", 0], "latent_image": ["6", 0]}}
        wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": vae}}
        wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "ivs", "images": ["8", 0]}}
        return wf

    # t2i 普通文生图
    wf["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    wf["6"] = {"class_type": "KSampler", "inputs": {**cfg, "seed": seed, "denoise": 1.0, "model": mdl, "positive": pos_out, "negative": ["4", 0], "latent_image": ["5", 0]}}
    wf["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae}}
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
    seed = random.randint(0, 2**31 - 1)
    wf = build_wf(model, pos, neg, w, h, seed, mode, ref, mask, strength, scale, ctype)
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
body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:860px;margin:20px auto;padding:0 16px;background:#f6f6f8;color:#222}
h1{font-size:22px} h2{font-size:17px;margin:18px 0 10px}
.card{background:#fff;border:1px solid #e2e2e6;border-radius:10px;padding:14px 16px;margin:10px 0;cursor:pointer}
.card:hover{border-color:#4f7cff}.card.sel{border-color:#4f7cff;background:#eef3ff}
.card.dis{opacity:.45;cursor:not-allowed}
.dot{font-size:12px}.on{color:#18a058}.off{color:#999}
button{background:#4f7cff;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:15px;cursor:pointer}
button:disabled{background:#aab}.back{background:#e8e8ec;color:#333}
textarea,input[type=text],input[type=number]{width:100%;box-sizing:border-box;border:1px solid #ccc;border-radius:6px;padding:8px;font-size:14px;font-family:inherit}
textarea{height:56px} select{border:1px solid #ccc;border-radius:6px;padding:6px;font-size:14px}
label{font-size:13px;color:#555;display:block;margin:8px 0 3px}
.bar{height:10px;background:#eee;border-radius:5px;overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;background:#4f7cff;width:0;transition:width .5s}
img.out{max-width:100%;border-radius:8px;margin-top:10px}
img.mask{filter:blur(18px)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.small{font-size:12px;color:#888}
.box{background:#fff7e6;border:1px solid #ffd591;border-radius:10px;padding:14px 16px;margin:10px 0}
code{background:#eee;border-radius:4px;padding:1px 6px;font-size:13px}
</style></head><body>
<h1>🎨 ImageVideoStudio · 生图/生视频小助手</h1>
<div id="app"></div>
<script>
const $ = s => document.querySelector(s);
let ST = {step:0, model:null, mname:'', msec:0, count:1, items:[], pids:[], hasCN:false};
const NEG_DEF = %NEG%;
const SIZES = [[1024,576,'横屏16:9 场景'],[1024,1024,'方形1:1 半身/头像'],[768,1152,'竖屏2:3 全身'],[1152,768,'横屏3:2 宽场景']];
async function j(u, opt){ const r = await fetch(u, opt); return r.json(); }

async function home(){
  ST = {step:0, model:null, mname:'', msec:0, count:1, items:[], pids:[], hasCN:ST.hasCN};
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
    $('#app').innerHTML = `<h2>第二步:选生图模型</h2>` + ms.map(m=>`
      <div class="card" onclick="pickModel('${m.id}','${m.name}',${m.sec})">
        <b>${m.name}</b> <span class="small">约${m.sec}秒/张</span><div class="small">${m.desc}</div>
      </div>`).join('') + `<p><button class="back" onclick="home()">← 返回</button></p>`;
  }
  if(ST.step===2){
    let opts=''; for(let i=1;i<=10;i++) opts+=`<option ${i===1?'selected':''}>${i}</option>`;
    $('#app').innerHTML = `<h2>第三步:生成几张?(最多10张)</h2>
      <p class="small">模型:${ST.mname}(约${ST.msec}秒/张)</p>
      <select id="cnt" style="font-size:16px;padding:8px">${opts}</select> 张
      <p style="margin-top:16px"><button onclick="pickCount()">下一步 →</button>
      <button class="back" onclick="ST.step=1;render()">← 返回</button></p>`;
  }
  if(ST.step===3){
    let cards='';
    for(let i=0;i<ST.count;i++){
      if(!ST.items[i]) ST.items[i]={mode:'t2i',w:1024,h:576,pos:'',neg:NEG_DEF,strength:0.6,scale:2,ctype:'openpose',refFile:null,strokes:[],brush:30};
      cards += cardHTML(i);
    }
    $('#app').innerHTML = `<h2>第四步:每张图单独设置</h2>${cards}
      <p><button onclick="startGen()">🚀 开始生成</button>
      <button class="back" onclick="ST.step=2;render()">← 返回</button></p>`;
    for(let i=0;i<ST.count;i++) if(ST.items[i].mode==='inpaint') initMaskCanvas(i);
  }
  if(ST.step===4){
    let cards='';
    for(let i=0;i<ST.count;i++){
      cards += `<div class="card" style="cursor:default" id="g${i}">
        <b>第 ${i+1} 张</b> <span class="small" id="st${i}">排队中…</span>
        <div class="bar"><i id="b${i}"></i></div><div id="img${i}"></div></div>`;
    }
    $('#app').innerHTML = `<h2>生成中…</h2>
      <p class="row"><label style="margin:0"><input type="checkbox" id="mask" onchange="toggleMask(this.checked)"> 给图片打遮(模糊) — 默认不打遮</label></p>
      ${cards}<p id="again" style="display:none"><button onclick="home()">🏠 回首页再来一轮</button></p>`;
    pollAll();
  }
}
function pickModel(id,name,sec){ ST.model=id; ST.mname=name; ST.msec=sec; ST.step=2; render(); }
function pickCount(){ ST.count=+$('#cnt').value; ST.items=[]; ST.step=3; render(); }
function collect(i){
  const it=ST.items[i]; if(!it) return;
  if($('#pos'+i)) it.pos=$('#pos'+i).value;
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
function sizeRow(i,it){
  let sizeOpts = SIZES.map(s=>`<option value="${s[0]}x${s[1]}" ${it.w===s[0]&&it.h===s[1]?'selected':''}>${s[0]} x ${s[1]} ${s[2]}</option>`).join('');
  const isCust = !SIZES.some(s=>s[0]===it.w&&s[1]===it.h);
  if(isCust) sizeOpts += `<option value="cust" selected>自定义 ${it.w} x ${it.h}</option>`;
  else sizeOpts += `<option value="cust">自定义…</option>`;
  return `<label>尺寸</label><div class="row"><select onchange="sizeSel(${i},this)">${sizeOpts}</select>
    <span id="cust${i}" style="display:${isCust?'inline':'none'}">
      <input type="number" id="cw${i}" value="${it.w}" style="width:80px" min="256" max="1280"> x
      <input type="number" id="ch${i}" value="${it.h}" style="width:80px" min="256" max="1280"></span></div>`;
}
function refRow(i,tip){
  const fn = ST.items[i].refFile ? `已选: ${ST.items[i].refFile.name}` : '';
  return `<label>${tip}</label><input type="file" id="ref${i}" accept="image/*" onchange="refPick(${i},this)"><span class="small">${fn}</span>`;
}
function refPick(i,el){ collect(i); ST.items[i].refFile=el.files[0]||null; ST.items[i].strokes=[]; render(); }
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
function cardHTML(i){
  const it=ST.items[i];
  let mid='';
  if(it.mode==='t2i') mid=sizeRow(i,it);
  if(it.mode==='i2i') mid=refRow(i,'上传参考图:保持画风构图,按提示词改内容(尺寸跟随参考图)')+strRow(i,'重绘幅度','小=更像原图,大=变化更大');
  if(it.mode==='inpaint') mid=refRow(i,'上传要修改的原图(尺寸跟随原图)')+maskRow(i);
  if(it.mode==='upscale') mid=refRow(i,'上传要放大的图')+
    `<label>放大倍数</label><select id="scale${i}"><option value="1.5" ${it.scale===1.5?'selected':''}>1.5 倍</option><option value="2" ${it.scale===2?'selected':''}>2 倍</option></select>`+
    strRow(i,'重绘幅度','建议0.3~0.5,太大细节会变样');
  if(it.mode==='pose') mid=refRow(i,'上传骨架图或线稿图(尺寸在下面选)')+sizeRow(i,it)+
    `<label>控制类型</label><select id="ctype${i}"><option value="openpose" ${it.ctype==='openpose'?'selected':''}>骨架图(openpose)</option><option value="hed/pidi/scribble/ted" ${it.ctype==='hed/pidi/scribble/ted'?'selected':''}>涂鸦/线稿(scribble)</option></select>`+
    strRow(i,'控制强度','越大越严格照图来');
  const posLabel={inpaint:'涂抹区域里要画什么(必填)',upscale:'正向提示词(可留空)',pose:'想要什么画面(必填)'}[it.mode]||'想要什么画面(正向提示词,必填)';
  return `<div class="card" style="cursor:default"><b>第 ${i+1} 张</b>
    <label>玩法</label><select id="mode${i}" onchange="setMode(${i})">
      <option value="t2i" ${it.mode==='t2i'?'selected':''}>普通文生图</option>
      <option value="i2i" ${it.mode==='i2i'?'selected':''}>以图生图(参考画风)</option>
      <option value="inpaint" ${it.mode==='inpaint'?'selected':''}>局部重绘(涂哪改哪)</option>
      <option value="upscale" ${it.mode==='upscale'?'selected':''}>放大变清晰</option>
      ${ST.hasCN?`<option value="pose" ${it.mode==='pose'?'selected':''}>姿势控制(骨架/线稿)</option>`:''}
    </select>${mid}
    <label>${posLabel}</label><textarea id="pos${i}" placeholder="例: a cat, masterpiece, best quality">${it.pos}</textarea>
    <label>不想要什么(负向提示词)</label><textarea id="neg${i}">${it.neg}</textarea></div>`;
}
function sizeSel(i, sel){
  $('#cust'+i).style.display = sel.value==='cust'?'inline':'none';
  if(sel.value!=='cust'){ const [w,h]=sel.value.split('x').map(Number); ST.items[i].w=w; ST.items[i].h=h; }
}
async function uploadFile(file, fname){
  const fd=new FormData(); fd.append('image', file, fname||file.name); fd.append('overwrite','true');
  const r=await fetch('/api/upload',{method:'POST',body:fd}); const d=await r.json();
  if(d.error) throw new Error('上传失败: '+d.error);
  return d.name;
}
async function startGen(){
  for(let i=0;i<ST.count;i++){
    const it=ST.items[i]; collect(i);
    if($('#cust'+i) && $('#cust'+i).style.display!=='none'){
      it.w=Math.min(1280,Math.max(256,+$('#cw'+i).value||1024));
      it.h=Math.min(1280,Math.max(256,+$('#ch'+i).value||1024));
    }
    if(it.mode==='upscale' && !it.pos.trim()) it.pos='masterpiece, best quality, highres, ultra detailed';
    if(!it.pos.trim()){ alert('第 '+(i+1)+' 张的正向提示词还没填!'); return; }
    if(!it.neg.trim()) it.neg=NEG_DEF;
    if(it.mode!=='t2i' && !it.refFile){ alert('第 '+(i+1)+' 张还没上传图片!'); return; }
    if(it.mode==='inpaint' && !it.strokes.length){ alert('第 '+(i+1)+' 张还没涂抹要重画的区域!'); return; }
    try{
      it.ref=null; it.mask=null;
      if(it.mode!=='t2i') it.ref=await uploadFile(it.refFile);
      if(it.mode==='inpaint'){ const blob=await exportMask(i); it.mask=await uploadFile(blob,'mask-'+Date.now()+'-'+i+'.png'); }
    }catch(e){ alert('第 '+(i+1)+' 张: '+e.message); return; }
  }
  ST.pids=[];
  for(let i=0;i<ST.count;i++){
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
  let left=ST.count;
  ST.pids.forEach(async (pid,i)=>{
    while(true){
      await new Promise(r=>setTimeout(r,2000));
      let r; try{ r=await j('/api/poll?pid='+pid); }catch(e){ continue; }
      if(r.done){ $('#b'+i).style.width='100%'; $('#st'+i).textContent='✅ 完成';
        $('#img'+i).innerHTML=`<img class="out" src="${r.url}">`;
        if($('#mask').checked) $('#img'+i+' img').classList.add('mask');
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
