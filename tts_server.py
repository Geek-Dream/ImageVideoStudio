#!/usr/bin/env python3
# ============================================================
# tts_server.py · Audio8 语音微服务(独立进程,端口 8851)
#   必须用 ~/.venvs/audio8/bin/python 跑(transformers 4.57,与 ComfyUI 的 5.x 隔离)
#   gen.py 通过 HTTP 转发调用,自己不加载模型。
#   接口:
#     GET  /health            → {"ok":true} 就绪探测
#     GET  /voices            → 声音库列表(扫 models/tts/voices/*/ref.wav+ref.txt)
#     POST /learn  {name, ref_text} + 已有 ref.wav → 登记声音(音轨由 gen.py 提前抽好)
#     POST /tts    {text, voice?} → {"wav": "<路径>"} 无 voice=随机音色,有=克隆
#   内存: 0.6B 模型跑 CPU(~2.5GB),不触发项目内存互斥,可与其它服务共存。
# ============================================================
import os, json, time, threading, traceback, shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models", "tts", "Audio8")
VOICES_DIR = os.path.join(BASE, "models", "tts", "voices")
OUT_DIR = os.path.join(BASE, "output", "tts")
PORT = int(os.environ.get("TTS_PORT", "8851"))

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

_model = None
_proc = None
_lock = threading.Lock()   # 生成是重活,串行,避免并发把 CPU 打爆
_err = None
_prog = {"step": 0, "max": 0, "busy": False}   # 真实生成进度(逐帧回调)
SAMPLE_TEXT = "你好,这是我的声音,以后就用它给你说话了。"
MAX_REF_SEC = 12   # 克隆参考音最长秒数:参考音要占模型 prompt(上限2048),12s≈250 token,留足文本空间

def _speech_segments(wav, max_sec=MAX_REF_SEC):
    """ffmpeg silencedetect 找出所有非静音(说话)段,按时间顺序拼到 max_sec 为止。
    返回 [(起点秒, 时长秒), ...];找不到 → [(0, max_sec)]。适配"长音频里零星几句人声"。"""
    import subprocess, re
    dur = 0.0
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", wav], capture_output=True, text=True)
        dur = float(out.stdout.strip())
    except Exception:
        pass
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav,
                        "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    segs, prev = [], 0.0                      # 非静音区间 = 静音与静音之间
    for i, s in enumerate(starts):
        if s > prev:
            segs.append((prev, s))
        prev = ends[i] if i < len(ends) else s
    if dur > prev:
        segs.append((prev, dur))
    segs = [(a, b) for a, b in segs if b - a >= 0.8]   # 至少 0.8 秒才算有效说话段
    if not segs:
        return [(0.0, min(max_sec, dur or max_sec))]
    picked, total = [], 0.0                   # 按时间顺序拼接,凑满 max_sec
    for a, b in segs:
        if total >= max_sec:
            break
        take = min(b - a, max_sec - total)
        picked.append((a, take))
        total += take
    return picked

def trimmed_ref(voice_dir):
    """返回 ≤MAX_REF_SEC 的参考音路径:把所有说话段拼出来,缓存在 ref_trim.wav(源更新自动重切)。"""
    import subprocess, soundfile as sf
    src = os.path.join(voice_dir, "ref.wav")
    dst = os.path.join(voice_dir, "ref_trim.wav")
    try:
        f = sf.SoundFile(src)
        dur = len(f) / f.samplerate
    except Exception:
        dur = 0
    if dur and dur <= MAX_REF_SEC:
        return src
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
        return dst
    segs = _speech_segments(src)
    if len(segs) == 1:
        a, t = segs[0]
        cmd = ["ffmpeg", "-y", "-ss", f"{a:.2f}", "-t", f"{t:.2f}",
               "-i", src, "-ar", "44100", "-ac", "1", dst]
    else:   # 多段: 逐段 atrim 再 concat
        fc = "".join(f"[0:a]atrim=start={a:.2f}:end={a + t:.2f},asetpts=PTS-STARTPTS[s{i}];"
                     for i, (a, t) in enumerate(segs))
        fc += "".join(f"[s{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=0:a=1[out]"
        cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", fc, "-map", "[out]",
               "-ar", "44100", "-ac", "1", dst]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst if os.path.exists(dst) else src

class _Tracker:
    """generate 的 stopping_criteria 每生成一帧调一次 → 真实进度(任何可调用对象都行)。"""
    def __call__(self, input_ids, scores, **kw):
        _prog["step"] += 1
        return False

def load_model():
    global _model, _proc, _err
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        _proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        _model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                           dtype=torch.float32).eval()
        _err = None
    except Exception:
        _err = traceback.format_exc()
        _model = None

def list_voices():
    """声音库: voices/<名字>/ 下有 ref.wav + ref.txt 才算一个完整声音。"""
    out = []
    if os.path.isdir(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            d = os.path.join(VOICES_DIR, name)
            wav = os.path.join(d, "ref.wav")
            txt = os.path.join(d, "ref.txt")
            if os.path.isfile(wav) and os.path.isfile(txt):
                out.append({"name": name,
                            "ref_text": open(txt, encoding="utf-8").read().strip()})
    return out

def synth(text, voice=None):
    """核心合成。voice=None 随机音色;否则用 voices/<voice>/ 的参考音克隆。"""
    import torch, soundfile as sf
    kwargs = {"text": [text]}
    if voice:
        d = os.path.join(VOICES_DIR, voice)
        wav = os.path.join(d, "ref.wav")
        txt = os.path.join(d, "ref.txt")
        if not (os.path.isfile(wav) and os.path.isfile(txt)):
            raise ValueError(f"声音不存在: {voice}")
        kwargs["reference_audio"] = [trimmed_ref(d)]   # 参考音超 12s 自动切说话段,防爆模型 2048 prompt 上限
        kwargs["reference_text"] = [open(txt, encoding="utf-8").read().strip()]
    inputs = _proc(**kwargs, return_tensors="pt")
    # max_new_tokens=1024 是上限,实际说到 EOS 就停;按台词长度估帧数(约5.5帧/字),进度条才走得满
    est = max(60, int(len(text) * 5.5))
    _prog.update(step=0, max=min(est, 1024), busy=True)
    try:
        with torch.inference_mode():
            output = _model.generate(**inputs, max_new_tokens=1024, temperature=0.8,
                                     top_p=0.95, top_k=50, do_sample=True,
                                     stopping_criteria=[_Tracker()],
                                     return_dict_in_generate=True)
            waveforms, lengths = _model.decode_audio(output.codes)
    finally:
        _prog["busy"] = False
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    name = f"tts_{int(time.time()*10)}.wav"
    path = os.path.join(OUT_DIR, name)
    sf.write(path, audio, _model.config.codec_sample_rate)
    return name

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # 静默,日志由 gen.py 那边管
        pass

    def do_GET(self):
        if self.path == "/health":
            if _model is not None:
                self._send(200, {"ok": True})
            elif _err:
                self._send(500, {"ok": False, "error": _err[-500:]})
            else:
                self._send(503, {"ok": False, "loading": True})
        elif self.path == "/voices":
            self._send(200, {"voices": list_voices()})
        elif self.path == "/progress":
            self._send(200, dict(_prog))
        else:
            self._send(404, {"error": "unknown"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, {"error": "bad json"}); return
        if self.path == "/tts":
            text = (body.get("text") or "").strip()
            voice = body.get("voice") or None
            if not text:
                self._send(400, {"error": "text 不能为空"}); return
            if _model is None:
                self._send(503, {"error": "模型未就绪", "detail": (_err or "加载中")[-300:]}); return
            with _lock:
                try:
                    name = synth(text, voice)
                    self._send(200, {"ok": True, "wav": name})
                except Exception as e:
                    self._send(500, {"error": str(e)[:300]})
        elif self.path == "/sample":
            # 给某个已登记的声音生成试听,存成 voices/<voice>/sample.wav 供网页试听;可自定义试听句
            voice = (body.get("voice") or "").strip()
            text = (body.get("text") or "").strip() or SAMPLE_TEXT
            d = os.path.join(VOICES_DIR, voice)
            if not voice or not os.path.isfile(os.path.join(d, "ref.wav")):
                self._send(404, {"error": f"声音不存在: {voice}"}); return
            if _model is None:
                self._send(503, {"error": "模型未就绪", "detail": (_err or "加载中")[-300:]}); return
            with _lock:
                try:
                    name = synth(text, voice)
                    shutil.move(os.path.join(OUT_DIR, name), os.path.join(d, "sample.wav"))
                    self._send(200, {"ok": True})
                except Exception as e:
                    self._send(500, {"error": str(e)[:300]})
        else:
            self._send(404, {"error": "unknown"})

if __name__ == "__main__":
    print(f"[tts] 加载 Audio8 模型(CPU)… port={PORT}", flush=True)
    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    print(f"[tts] HTTP 服务已起: http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
