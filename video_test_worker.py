#!/usr/bin/env python3
"""视频测试场独立 worker：按任务号、模型、提示词、垫图和参数顺序逐条生成。"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

try:
    from PIL import Image, ImageStat
except ImportError:
    Image = ImageStat = None

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc
import vidwf

JOBS = os.path.join(BASE, "video_test_jobs")
QUEUE = os.path.join(JOBS, "queue")
STATUS_F = os.path.join(JOBS, "status.json")
CTRL_F = os.path.join(JOBS, "control.json")
PID_F = os.path.join(JOBS, "worker.pid")
IMAGE_PID_F = os.path.join(BASE, "test_jobs", "worker.pid")
OUT_BASE = os.path.join(BASE, "output", "vidtest")
WORK_BASE = os.path.join(JOBS, "work")
REFS = os.path.join(BASE, "refs")

for directory in (QUEUE, os.path.join(JOBS, "prompts"), OUT_BASE, WORK_BASE):
    os.makedirs(directory, exist_ok=True)

_log = []


def write_status(**values):
    status = {}
    try:
        status = json.load(open(STATUS_F, encoding="utf-8"))
    except Exception:
        pass
    status.update(values)
    status.update(pid=os.getpid(), ts=time.time())
    tmp = STATUS_F + ".tmp"
    json.dump(status, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, STATUS_F)


def log(message):
    line = time.strftime("[%H:%M:%S] ") + message
    print(line, flush=True)
    _log.append(line)
    write_status(log=_log[-30:])


def load_task(path):
    return json.load(open(path, encoding="utf-8"))


def save_task(path, task):
    tmp = path + ".tmp"
    json.dump(task, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, path)


def next_task():
    for filename in sorted(os.listdir(QUEUE)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(QUEUE, filename)
        try:
            task = load_task(path)
        except Exception:
            continue
        if task.get("state") == "queued":
            return path, task
    return None, None


def read_prompts(task):
    path = os.path.join(BASE, task.get("prompts_file", ""))
    if not os.path.isfile(path):
        return []
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()]


def read_control():
    try:
        return json.load(open(CTRL_F, encoding="utf-8")).get("cmd", "")
    except Exception:
        return ""


def clear_control():
    try:
        os.remove(CTRL_F)
    except OSError:
        pass


def pidfile_alive(path):
    try:
        pid = int(open(path).read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def safe_name(value, fallback="video"):
    name = re.sub(r"[^\w]+", "_", str(value or "").strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", name).strip("_")[:80] or fallback


def frame_count(duration, fps):
    raw = max(9, round(float(duration) * float(fps)))
    return max(9, round((raw - 1) / 8) * 8 + 1)


def result_state(stop, ok, fail):
    if stop:
        return "killed"
    if ok and fail:
        return "partial"
    return "done" if ok else "failed"


def handle_control():
    paused = False
    while True:
        cmd = read_control()
        if cmd == "kill_all":
            return "all"
        if cmd == "kill_curr":
            clear_control()
            return "curr"
        if cmd == "pause":
            if not paused:
                paused = True
                write_status(state="paused", msg="已暂停，将在这里等待继续")
                log("⏸ 已暂停")
            time.sleep(1)
            continue
        if cmd == "resume":
            clear_control()
            write_status(state="running", msg="")
            if paused:
                log("▶ 继续")
        return ""


def ensure_video_service():
    health = svc.comfy_health("vid")
    if health.get("api_ok"):
        return {"ok": True}
    started = svc.start_svc("vid")
    if not started.get("ok"):
        return {"ok": False, "error": started.get("error", "生视频服务启动失败")}
    deadline = time.time() + 240
    last = "视频服务还没准备好"
    while time.time() < deadline:
        health = svc.comfy_health("vid", timeout=3)
        if health.get("api_ok"):
            return {"ok": True}
        last = health.get("error", last)
        time.sleep(3)
    return {"ok": False, "error": last}


def release_video_service(reason=""):
    """释放视频模型占用的统一内存；有外部队列时不强行关闭。"""
    try:
        port = svc.svc_status("vid")["port"]
        queue = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=5))
        if queue.get("queue_running") or queue.get("queue_pending"):
            log("⚠ 视频服务仍有其他队列任务，暂不释放 ComfyUI 内存")
            return False
    except Exception:
        # 服务已经退出时，目标已经达到；否则不要因为探测失败误杀进程。
        if not svc.svc_status("vid").get("running"):
            return True
        return False
    stopped = svc.stop_svc("vid")
    if stopped.get("ok"):
        log("🧹 " + (reason or "视频队列结束") + "，已释放视频模型内存")
        write_status(service_ready=False)
        return True
    return False


def upload_local(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到垫图: " + path)
    boundary = "----ivsvideotest"
    raw = open(path, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(path)}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    port = svc.svc_status("vid")["port"]
    request = urllib.request.Request(f"http://127.0.0.1:{port}/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    result = json.load(urllib.request.urlopen(request, timeout=120))
    return result.get("name") or os.path.basename(path)


def upload_ref(filename):
    return upload_local(os.path.join(REFS, os.path.basename(filename)))


def wait_video(prompt_id, timeout=1800):
    started = time.time()
    deadline = time.time() + timeout
    while True:
        cmd = read_control()
        if cmd in ("kill_curr", "kill_all"):
            vidwf.cancel_vid(prompt_id)
            if cmd == "kill_curr":
                clear_control()
            return None, ("all" if cmd == "kill_all" else "curr")
        result = vidwf.poll_vid(prompt_id)
        write_status(segment_elapsed=int(time.time() - started), comfy_state=result.get("state", "running"),
                     queue_position=result.get("pos", 0), watchdog_left=max(0, int(deadline - time.time())))
        if result.get("error"):
            raise RuntimeError(result["error"])
        if result.get("done"):
            name = os.path.basename(result.get("url", ""))
            path = os.path.join(vidwf.OUT_VID, name)
            if os.path.isfile(path):
                return path, ""
            raise RuntimeError("视频完成了，但找不到输出文件")
        if time.time() >= deadline:
            vidwf.cancel_vid(prompt_id)
            raise TimeoutError(f"单段超过{round(timeout / 60)}分钟，已自动中断")
        time.sleep(2)


def safe_generation_size(model_id, target_w, target_h):
    """高分辨率成片先低分辨率生成，拼接后统一放大，避免模型阶段爆内存。"""
    max_side = 832
    step = 32
    w, h = int(target_w), int(target_h)
    if max(w, h) <= max_side:
        return max(64, w // step * step), max(64, h // step * step)
    if h >= w:
        gh = max_side
        gw = max(64, round((gh * w / h) / step) * step)
    else:
        gw = max_side
        gh = max(64, round((gw * h / w) / step) * step)
    return gw, gh


def segment_durations(model_id, duration):
    """按安全上限计算段数，再均分时长，避免出现 6+2 这种短尾段。"""
    return vidwf.balanced_segments(duration, 6.0)


def item_key(model_index, prompt_index, ref_index, variant_index, copy_index):
    return f"{model_index}:{prompt_index}:{ref_index}:{variant_index}:{copy_index}"


def max_output_index(out_dir):
    """从已有成片名恢复本模型的编号，避免断点续跑覆盖旧文件。"""
    highest = 0
    if not os.path.isdir(out_dir):
        return highest
    for filename in os.listdir(out_dir):
        match = re.search(r"_(\d+)\.mp4$", filename, flags=re.I)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def checkpoint_summary(task, total=0):
    completed = task.get("completed_items") or []
    done = len(completed) if completed else len(task.get("outputs") or [])
    point = task.get("checkpoint") or {}
    return {
        "done": done,
        "total": int(total or task.get("total") or 0),
        "model": point.get("model", ""),
        "copy": int(point.get("copy") or 0),
        "segment_done": int(point.get("segment_done") or 0),
        "segment_total": int(point.get("segment_total") or 0),
    }


def extract_first_frame(video_path, image_path):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-ss", "0.04", "-i", video_path, "-frames:v", "1", image_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
    if result.returncode or not os.path.isfile(image_path):
        raise RuntimeError("提取颜色基准帧失败: " + result.stderr.decode(errors="replace")[-300:])
    return image_path


def stabilize_frame_color(image_path, anchor_path, output_path):
    """只校正接续帧的整体色偏，亮度和画面内容保持不变。"""
    if Image is None or not anchor_path or not os.path.isfile(anchor_path):
        return image_path
    try:
        with Image.open(anchor_path) as source:
            anchor = source.convert("YCbCr")
            anchor.thumbnail((256, 256))
            anchor_mean = ImageStat.Stat(anchor).mean
        with Image.open(image_path) as source:
            frame = source.convert("YCbCr")
        sample = frame.copy()
        sample.thumbnail((256, 256))
        frame_mean = ImageStat.Stat(sample).mean
        # Cb/Cr control blue-red chroma. A bounded correction avoids changing
        # deliberate scene colors while stopping a small cast accumulating.
        cb_shift = max(-32.0, min(32.0, (anchor_mean[1] - frame_mean[1]) * 0.9))
        cr_shift = max(-32.0, min(32.0, (anchor_mean[2] - frame_mean[2]) * 0.9))
        y_channel, cb_channel, cr_channel = frame.split()
        cb_channel = cb_channel.point(lambda value: max(0, min(255, round(value + cb_shift))))
        cr_channel = cr_channel.point(lambda value: max(0, min(255, round(value + cr_shift))))
        Image.merge("YCbCr", (y_channel, cb_channel, cr_channel)).convert("RGB").save(output_path)
        return output_path
    except Exception as exc:
        log("  ⚠ 接续帧色调校正失败，继续使用原末帧: " + str(exc))
        return image_path


def extract_last_frame(video_path, image_path, anchor_path=None):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-sseof", "-0.12", "-i", video_path, "-frames:v", "1", image_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
    if result.returncode or not os.path.isfile(image_path):
        raise RuntimeError("提取上一段末帧失败: " + result.stderr.decode(errors="replace")[-300:])
    stable_path = os.path.splitext(image_path)[0] + "_color_stable.png"
    return upload_local(stabilize_frame_color(image_path, anchor_path, stable_path))


def finish_long_video(segment_paths, destination, target_w, target_h, duration,
                      source_fps, target_fps, force_interpolate=False,
                      stabilize_color=False, native_audio=False):
    """无损拼分段，再统一裁时长、放大和补帧。"""
    if not segment_paths:
        raise RuntimeError("没有可拼接的视频分段")
    work_dir = os.path.dirname(segment_paths[0])
    concat_file = os.path.join(work_dir, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as handle:
        for path in segment_paths:
            handle.write("file '" + os.path.abspath(path).replace("'", "'\\''") + "'\n")
    joined = os.path.join(work_dir, "joined.mp4")
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-f", "concat", "-safe", "0", "-i", concat_file,
                             "-c", "copy", joined], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=900)
    if result.returncode or not os.path.isfile(joined):
        raise RuntimeError("拼接分段失败: " + result.stderr.decode(errors="replace")[-400:])
    filters = []
    if stabilize_color:
        smoothing = max(12, round(float(source_fps) * 2))
        filters.append(f"normalize=smoothing={smoothing}:independence=0.35:strength=0.28")
    # 最终输出严格使用用户选的尺寸；高分辨率是后期 Lanczos 放大。
    filters.append(f"scale={int(target_w)}:{int(target_h)}:flags=lanczos")
    # 分段帧数按 8n+1 对齐，拼接后可能比目标短零点几秒；克隆末帧兜底再精确裁切。
    filters.append("tpad=stop_mode=clone:stop_duration=1")
    final_fps = float(target_fps)
    if final_fps > source_fps or force_interpolate:
        final_fps = max(final_fps, 60 if force_interpolate else final_fps)
        filters.append(f"minterpolate=fps={final_fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", joined,
           "-t", f"{float(duration):g}", "-vf", ",".join(filters),
           "-map", "0:v:0"]
    if native_audio:
        # 旧 LTX 音频输出通常偏小，保留音轨并做约 +20dB 增益；没有音轨时不让整条任务失败。
        cmd += ["-map", "0:a:0?"]
        if native_audio != "plain":
            cmd += ["-af", "volume=20dB"]
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", destination]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
    if result.returncode or not os.path.isfile(destination):
        raise RuntimeError("最终放大/补帧失败: " + result.stderr.decode(errors="replace")[-500:])
    return destination


def run_task(task_path, task):
    work_root = os.path.join(WORK_BASE, safe_name(task.get("id"), "task"))
    os.makedirs(work_root, exist_ok=True)
    resuming = bool(task.get("resume_requested") or task.get("outputs") or
                    task.get("completed_items") or task.get("checkpoint"))
    if not resuming:
        task.update(outputs=[], segments=[], completed_items=[], checkpoint={}, output_dirs=[], timings=[])
    else:
        task.setdefault("outputs", [])
        task.setdefault("segments", [])
        task.setdefault("completed_items", [])
        task.setdefault("output_dirs", [])
        task.setdefault("timings", [])
    task.update(state="running", started=time.time(), resumed=bool(resuming),
                work_dir=os.path.relpath(work_root, BASE))
    task.pop("resume_requested", None)
    save_task(task_path, task)
    prompts = read_prompts(task)
    models = task.get("models") or []
    refs = task.get("refs") or []
    ref_items = [""] if task.get("mode") == "t2v" else refs
    variants = task.get("variants") or [{"w": 360, "h": 640, "duration": 2, "fps": 24}]
    copies = max(1, int(task.get("copies", 1)))
    total = len(models) * len(prompts) * len(ref_items) * len(variants) * copies
    completed = set(str(value) for value in task.get("completed_items") or [])
    legacy_remaining = len(task.get("outputs") or []) if not completed else 0
    done = ok = len(completed) if completed else legacy_remaining
    fail = 0
    stop = ""
    uploaded = {}
    used_folders = set()
    task["total"] = total
    task["progress"] = checkpoint_summary(task, total)
    save_task(task_path, task)
    write_status(task=task["id"], state="running", done=done, total=total, ok=ok, fail=0,
                 folder="", msg="", finished=False)
    log(f"📋 视频任务 {task['id']}: 共 {total} 条最终成片" +
        (f"，从断点继续，已完成 {done} 条" if resuming and done else ""))

    for model_index, model in enumerate(models, 1):
        if stop:
            break
        # Different video bases/workflows can retain MPS allocations in the
        # same ComfyUI process. Restart between model families so their memory
        # does not accumulate. Uploaded reference names are process-local.
        if model_index > 1:
            released = release_video_service(f"切换到 {model.get('name', model.get('id', '下一个模型'))}")
            if released:
                uploaded.clear()
                service = ensure_video_service()
                if not service.get("ok"):
                    raise RuntimeError(service.get("error", "切换视频模型时服务启动失败"))
                write_status(service_ready=True)
        # The API allocates a unique folder per task. Keep this fallback for
        # older queued JSON files created before output_folder was introduced.
        folder = safe_name(model.get("output_folder") or model.get("folder"),
                           model.get("name") or model.get("id"))
        base_folder = folder
        index = 2
        while folder.casefold() in used_folders:
            folder = f"{base_folder[:74]}_{index}"
            index += 1
        used_folders.add(folder.casefold())
        out_dir = os.path.join(OUT_BASE, folder)
        os.makedirs(out_dir, exist_ok=True)
        if os.path.relpath(out_dir, BASE) not in task["output_dirs"]:
            task["output_dirs"].append(os.path.relpath(out_dir, BASE))
        output_index = max_output_index(out_dir)
        for prompt_index, prompt in enumerate(prompts, 1):
            for ref_index, ref_name in enumerate(ref_items, 1):
                if stop:
                    break
                image_name = ""
                if ref_name:
                    if ref_name not in uploaded:
                        if not ensure_video_service().get("ok"):
                            raise RuntimeError("生视频服务启动失败")
                        uploaded[ref_name] = upload_ref(ref_name)
                    image_name = uploaded[ref_name]
                for variant_index, variant in enumerate(variants, 1):
                    for copy_index in range(1, copies + 1):
                        # 工作流错误会在上一条把 stop 置为 error；不能被下一轮控制检查重置。
                        if stop:
                            break
                        key = item_key(model_index, prompt_index, ref_index,
                                       variant_index, copy_index)
                        if key in completed:
                            continue
                        if legacy_remaining > 0:
                            completed.add(key)
                            task["completed_items"].append(key)
                            legacy_remaining -= 1
                            continue
                        stop = handle_control()
                        if stop:
                            break
                        w, h = int(variant["w"]), int(variant["h"])
                        duration, target_fps = float(variant["duration"]), float(variant["fps"])
                        source_fps = min(24.0, target_fps)
                        gen_w, gen_h = safe_generation_size(model["id"], w, h)
                        parts = segment_durations(model["id"], duration)
                        ref_label = f"垫图{ref_index}" if ref_name else "纯文字"
                        current = (f"{model.get('name', model['id'])} | 提示词{prompt_index} | {ref_label} | "
                                   f"成片{w}x{h} {duration:g}秒 {target_fps:g}fps | {len(parts)}段生成")
                        task["checkpoint"] = {
                            "item_key": key, "model_index": model_index,
                            "model": model.get("name", model["id"]),
                            "prompt": prompt_index, "ref": ref_index,
                            "variant": variant_index, "copy": copy_index,
                            "segment_done": 0, "segment_total": len(parts),
                        }
                        task["progress"] = checkpoint_summary(task, total)
                        save_task(task_path, task)
                        write_status(cur_model=model.get("name", model["id"]), cur_prompt=prompt_index,
                                     done=done, total=total, ok=ok, fail=fail,
                                     folder=f"output/vidtest/{folder}", msg=current, t0=time.time())
                        log(f"[{done + 1}/{total}] {current}")
                        item_started = time.time()
                        item_succeeded = False
                        try:
                            native_audio = bool(task.get("native_audio", False))
                            selected_loras = task.get("loras") or task.get("lora", "none")
                            output_index += 1
                            file_suffix = str(task.get("file_suffix") or "")
                            ref_part = f"r{ref_index:02d}" if ref_name else "t2v"
                            filename = (f"p{prompt_index:02d}_{ref_part}_{w}x{h}_{duration:g}s_{target_fps:g}fps_"
                                        f"v{variant_index:02d}{file_suffix}_{output_index:02d}.mp4")
                            destination = os.path.join(out_dir, filename)
                            # Intermediate segments never sit beside final MP4s.
                            segment_dir = os.path.join(work_root, folder, f"{output_index:03d}")
                            os.makedirs(segment_dir, exist_ok=True)
                            segment_paths = []
                            continue_image = image_name
                            color_anchor = (os.path.join(REFS, os.path.basename(ref_name))
                                            if ref_name else "")
                            segment_index = 0
                            while segment_index < len(parts):
                                segment_stop = handle_control()
                                if segment_stop:
                                    stop = segment_stop
                                    break
                                part_duration = parts[segment_index]
                                frames = frame_count(part_duration, source_fps)
                                write_status(msg=(current + f" · 分段 {segment_index + 1}/{len(parts)} "
                                                  f"({part_duration:g}秒/{frames}帧)"), t0=time.time())
                                log(f"  ↳ 分段 {segment_index + 1}/{len(parts)}: {part_duration:g}秒，{gen_w}x{gen_h}，{frames}帧")
                                temp_name = safe_name(
                                    f"vtest_{task['id']}_{model['id']}_{int(time.time() * 1000)}", "vtest")
                                style_2d = bool(task.get("style_2d", False))
                                segment_prompt = prompt
                                service = ensure_video_service()
                                if not service.get("ok"):
                                    raise RuntimeError(service.get("error", "生视频服务启动失败"))
                                pid = vidwf.submit_vid(
                                    temp_name, unet_id=model["id"], pos=segment_prompt,
                                    neg="",
                                    image_name=continue_image, w=gen_w, h=gen_h, frames=frames, fps=source_fps,
                                    lora_id=(selected_loras[0] if isinstance(selected_loras, list) and selected_loras else "none"),
                                    lora_ids=selected_loras, lora_strength=0.8,
                                    use_stg=bool(task.get("stg")),
                                    steps=8, native_audio=native_audio,
                                    style_2d=style_2d, postprocess={})
                                try:
                                    source, killed = wait_video(
                                        pid, timeout=2400)
                                except TimeoutError:
                                    if part_duration > 1.05:
                                        first = round(part_duration / 2, 3)
                                        parts[segment_index:segment_index + 1] = [first, round(part_duration - first, 3)]
                                        log(f"  ⚠ 本段超时，自动拆成 {parts[segment_index]:g}+{parts[segment_index + 1]:g} 秒重试")
                                        continue
                                    raise
                                if killed:
                                    stop = killed
                                    break
                                segment_path = os.path.join(segment_dir, f"segment_{segment_index + 1:03d}.mp4")
                                shutil.move(source, segment_path)
                                segment_paths.append(segment_path)
                                task["checkpoint"].update(
                                    segment_done=segment_index + 1,
                                    segment_total=len(parts))
                                task["progress"] = checkpoint_summary(task, total)
                                save_task(task_path, task)
                                if segment_index < len(parts) - 1:
                                    if not color_anchor:
                                        color_anchor = os.path.join(segment_dir, "color_anchor.png")
                                        extract_first_frame(segment_path, color_anchor)
                                    last_frame = os.path.join(segment_dir, f"last_{segment_index + 1:03d}.png")
                                    continue_image = extract_last_frame(segment_path, last_frame, color_anchor)
                                segment_index += 1
                            if stop:
                                break
                            write_status(msg=current + " · 正在拼接/放大/补帧", t0=time.time())
                            finish_long_video(segment_paths, destination, w, h, duration, source_fps,
                                              target_fps, bool(task.get("interpolate")),
                                              False, native_audio)
                            task["outputs"].append(os.path.relpath(destination, BASE))
                            task.setdefault("segments", []).append(
                                [os.path.relpath(path, BASE) for path in segment_paths])
                            completed.add(key)
                            task["completed_items"].append(key)
                            task["checkpoint"] = {}
                            ok += 1
                            item_succeeded = True
                            log(f"  ✓ 完成 → {os.path.relpath(destination, BASE)}")
                        except Exception as exc:
                            fail += 1
                            message = str(exc)
                            log("  ✗ " + message)
                            if any(word in message for word in ("Connection refused", "服务", "API 未就绪")):
                                task["error_code"] = "service_unavailable"
                                task["error"] = message
                                stop = "error"
                            elif message.startswith("ComfyUI 工作流执行失败"):
                                # 这是工作流、插件或模型文件问题；同一个配置继续刷只会重复失败。
                                task["error_code"] = "workflow_error"
                                task["error"] = message
                                stop = "error"
                        task["timings"].append({
                            "model_id": model["id"],
                            "model": model.get("name", model["id"]),
                            "elapsed_sec": round(time.time() - item_started),
                            "segment_count": len(parts),
                            "video_duration": duration,
                            "ok": item_succeeded,
                        })
                        task["timings"] = task["timings"][-500:]
                        done += 1
                        task["progress"] = checkpoint_summary(task, total)
                        save_task(task_path, task)
                        write_status(done=done, ok=ok, fail=fail)
                    if stop:
                        break
            if stop:
                break

    if done >= total and not stop:
        task["checkpoint"] = {}
    task.update(state=("failed" if stop == "error" else result_state(stop, ok, fail)),
                finished=time.time(), ok=ok, fail=fail, total=total)
    task["progress"] = checkpoint_summary(task, total)
    save_task(task_path, task)
    write_status(done=done, ok=ok, fail=fail, state=task["state"], finished=True)
    log(f"🏁 视频任务 {task['id']} 结束：成功 {ok}/{total}，状态 {task['state']}")
    release_video_service("本视频任务结束")
    return stop


def main():
    open(PID_F, "w").write(str(os.getpid()))
    clear_control()
    write_status(running=True, state="idle", msg="视频 worker 启动", finished=False)
    log(f"🔧 视频测试 worker 启动 pid={os.getpid()}")
    if pidfile_alive(IMAGE_PID_F):
        write_status(running=False, state="blocked", msg="图片测试正在运行，视频测试没有启动", finished=True)
        log("✗ 图片测试 worker 正在运行，为保护内存，视频测试退出")
        try: os.remove(PID_F)
        except OSError: pass
        return
    for filename in sorted(os.listdir(QUEUE)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(QUEUE, filename)
        try:
            task = load_task(path)
            if task.get("state") in ("running", "paused"):
                task["state"] = "interrupted"
                task["resume_available"] = True
                task["interrupted_at"] = time.time()
                save_task(path, task)
        except Exception:
            pass
    service = ensure_video_service()
    if not service.get("ok"):
        reason = service.get("error", "生视频服务不可用")
        for filename in sorted(os.listdir(QUEUE)):
            if filename.endswith(".json"):
                path = os.path.join(QUEUE, filename)
                try:
                    task = load_task(path)
                    if task.get("state") == "queued":
                        task.update(state="failed", finished=time.time(), error_code="service_unavailable", error=reason)
                        save_task(path, task)
                except Exception:
                    pass
        write_status(running=False, state="error", msg=reason, service_ready=False, finished=True)
        log("✗ " + reason)
        try: os.remove(PID_F)
        except OSError: pass
        return
    write_status(service_ready=True)
    ending = ""
    while True:
        if read_control() == "kill_all":
            ending = "killed"
            break
        path, task = next_task()
        if not path:
            break
        result = run_task(path, task)
        if result in ("all", "error"):
            ending = result
            break
    # Also clean up when a task exits through a service/workflow error or a
    # kill command, so a failed batch cannot leave a large MPS model resident.
    release_video_service("视频 worker 退出")
    try: os.remove(PID_F)
    except OSError: pass
    if ending in ("killed", "all"):
        write_status(running=False, state="killed", msg="视频 worker 已停止", finished=True)
    elif ending == "error":
        write_status(running=False, state="error", msg="视频服务出错，worker 已退出", finished=True)
    else:
        write_status(running=False, state="done", msg="视频队列已跑完", finished=True)
    log("🎉 视频 worker 退出")


if __name__ == "__main__":
    main()
