#!/usr/bin/env python3
# ============================================================
# test_worker.py · 图片测试场·独立后台生图进程
#   由 gen.py 网页「⚙️测试场」派生(start_new_session 脱离进程组):
#     关网页 / 杀 gen.py / 退出 start.sh 都不会停它。
#   停止只有两条路: start.sh 选4 / 网页测试场里暂停·杀死(都靠写 control.json)。
#   通信全靠 test_jobs/ 下的文件,重启谁都能接着读:
#     queue/NNN.json   任务(配置+提示词文件路径),state: queued/running/done/killed
#     prompts/NNN.txt  该任务的统一提示词(每行一条,可网页改也可直接改 txt)
#     status.json      实时进度(当前任务/模型/提示词/第几张/耗时/输出文件夹)
#     control.json     控制令: {"cmd":"pause|resume|kill_curr|kill_all"}
#     worker.pid       进程号(存活用)
#   图片落 output/imgtest/<每个模型自定义文件夹>/ ; 纯生图,不配气泡(测试用)。
#   队列跑空 → 打印「生图结束」自动退出。
# ============================================================
import os, re, json, time, shutil, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc
import gen as genmod
import comic_gen as cg   # ensure_img_service(内存互斥起生图)

JOBS     = os.path.join(BASE, "test_jobs")
QUEUE    = os.path.join(JOBS, "queue")
STATUS_F = os.path.join(JOBS, "status.json")
CTRL_F   = os.path.join(JOBS, "control.json")
PID_F    = os.path.join(JOBS, "worker.pid")
OUT_BASE = os.path.join(BASE, "output", "imgtest")
REFS     = os.path.join(BASE, "refs")

for d in (QUEUE, os.path.join(JOBS, "prompts"), OUT_BASE, REFS):
    os.makedirs(d, exist_ok=True)

# --------------- 状态 / 控制 ---------------

def _safe(s):
    return re.sub(r"[^\w.-]+", "_", s or "")

def write_status(**kw):
    st = {}
    if os.path.exists(STATUS_F):
        try:
            st = json.load(open(STATUS_F, encoding="utf-8"))
        except Exception:
            st = {}
    st.update(kw)
    st["pid"] = os.getpid()
    st["ts"] = time.time()
    tmp = STATUS_F + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, STATUS_F)

def read_control():
    try:
        return json.load(open(CTRL_F, encoding="utf-8")).get("cmd", "")
    except Exception:
        return ""

def clear_control():
    try:
        if os.path.exists(CTRL_F):
            os.remove(CTRL_F)
    except OSError:
        pass

_log = []
def log(msg):
    line = time.strftime("[%H:%M:%S] ") + msg
    print(line, flush=True)
    _log.append(line)
    write_status(log=_log[-30:])

# --------------- 任务读写 ---------------

def load_task(path):
    return json.load(open(path, encoding="utf-8"))

def save_task(path, t):
    tmp = path + ".tmp"
    json.dump(t, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, path)

def next_task():
    """取队列里第一个 state=queued 的任务(按文件名序=提交顺序)。"""
    for fn in sorted(os.listdir(QUEUE)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(QUEUE, fn)
        try:
            t = load_task(p)
        except Exception:
            continue
        if t.get("state") == "queued":
            return p, t
    return None, None

def read_prompts(t):
    pf = os.path.join(BASE, t.get("prompts_file", ""))
    if not os.path.exists(pf):
        return []
    return [l.strip() for l in open(pf, encoding="utf-8") if l.strip()]

# --------------- 控制处理(每张图前/暂停时调用) ---------------

def handle_control(task_path, t):
    """返回 'stop_all' / 'stop_curr' / '' 。pause 会在这里阻塞等到 resume/kill。"""
    paused = False
    while True:
        cmd = read_control()
        if cmd == "kill_all":
            return "stop_all"
        if cmd == "kill_curr":
            clear_control()
            return "stop_curr"
        if cmd == "pause":
            if not paused:
                paused = True
                write_status(state="paused", msg="已暂停")
                log("⏸ 已暂停(网页点继续/start.sh 不管)")
            time.sleep(1)
            continue
        if cmd == "resume":
            clear_control()
            if paused:
                log("▶ 继续")
            write_status(state="running", msg="")
            return ""
        if paused:                      # 控制令被外部清掉 → 当作继续
            write_status(state="running", msg="")
            return ""
        return ""

# --------------- 单张生成(复刻 comic_gen.gen_one 的等待) ---------------

def gen_wait(prompt, model_id, w, h, name, mode="t2i", ref=None, ipa=None, ipa_weight=0.3, batch=1):
    """提交并等完成,返回 (文件路径列表, 秒数)。batch>1(仅t2i)一次出N张。"""
    if ipa:
        pid = genmod.submit(model_id, prompt, genmod.NEG_DEFAULT, w, h, name,
                            ipa=ipa, ipa_weight=ipa_weight)
    elif ref:
        pid = genmod.submit(model_id, prompt, genmod.NEG_DEFAULT, w, h, name,
                            "i2i", ref, None, 0.75)
    else:
        pid = genmod.submit(model_id, prompt, genmod.NEG_DEFAULT, w, h, name, batch=batch)
    t0 = time.time()
    while time.time() - t0 < 1800:
        cmd = read_control()            # 等图期间也响应 kill_all(暂停不打断已提交的)
        if cmd == "kill_all":
            raise RuntimeError("KILL_ALL")
        t = genmod.TASKS.get(pid, {})
        if t.get("done"):
            files = t.get("files") or []
            if files:
                return files, time.time() - t0
            raise Exception("完成了但找不到输出文件")
        if t.get("error"):
            raise Exception(t["error"])
        time.sleep(1)
    raise Exception("超时(30分钟)")

# --------------- 锁图: 先用选定模型 t2i 出一张脸,传 ComfyUI 当 IPA 参考 ---------------

def make_lock_image(t):
    fl = t.get("face_lock") or {}
    lock_png = os.path.join(REFS, f"test_{t['id']}_lock.png")
    if not os.path.exists(lock_png):
        mp = fl.get("prompt") or "masterpiece, best quality, portrait of a person, detailed face, looking at viewer"
        log(f"🔒 锁图: 先用 {fl.get('gen_model')} 生成锁脸图…")
        src, sec = gen_wait(mp, fl.get("gen_model"), t["canvas"]["w"], t["canvas"]["h"],
                            _safe(f"testlock_{t['id']}_{int(time.time())}"))
        shutil.copy2(src[0], lock_png)
        log(f"  ✓ 锁脸图已生成({int(sec)}秒) → refs/{os.path.basename(lock_png)}")
    return genmod.ref_to_comfy(os.path.basename(lock_png))

# --------------- 跑一个任务 ---------------

def run_task(task_path, t):
    t["state"] = "running"; t["started"] = time.time()
    save_task(task_path, t)
    prompts = read_prompts(t)
    models  = t.get("models", [])
    per     = int(t.get("per_prompt", 1))
    W, H    = t["canvas"]["w"], t["canvas"]["h"]
    total   = len(models) * len(prompts) * per
    done = ok = fail = 0

    # 垫图: 整任务传一次(所有模型共用这一套画风参考)
    pad_server = None
    if t.get("pad_ref"):
        pad_server = genmod.ref_to_comfy(os.path.basename(t["pad_ref"]))
        if pad_server:
            log(f"🖼 垫图: {os.path.basename(t['pad_ref'])} 已上传,全员按它的画风走(i2i 0.75)")
        else:
            log("⚠ 垫图上传失败,改纯文字")
    # 锁图: 选定则先生成锁脸图,之后全员 ipa 锁脸
    ipa_server = None
    if (t.get("face_lock") or {}).get("enabled"):
        ipa_server = make_lock_image(t)
        if ipa_server:
            log("🔒 锁脸已启用,后续全员按锁脸图生成")
        else:
            log("⚠ 锁脸图上传失败,改不锁")

    write_status(task=t["id"], state="running", done=0, total=total, ok=0, fail=0,
                 folder="", msg="", finished=False)
    log(f"📋 任务 {t['id']}: {len(models)}模型 × {len(prompts)}提示词 × {per}张 = {total}张")

    stop = ""
    for m in models:
        if stop:
            break
        folder = _safe(m.get("folder") or m.get("name") or m["id"])
        ddir = os.path.join(OUT_BASE, folder)
        os.makedirs(ddir, exist_ok=True)
        for pi, ptext in enumerate(prompts, 1):
            if stop:
                break
            # 纯 t2i(无垫图无锁脸)一批出 per 张(共享CLIP编码,省时);垫图/锁脸 submit 会钳回1,逐张
            chunk = per if (not pad_server and not ipa_server) else 1
            i = 1
            while i <= per:
                if stop:
                    break
                r = handle_control(task_path, t)
                if r == "stop_all":
                    stop = "all"; break
                if r == "stop_curr":
                    stop = "curr"; break
                n = min(chunk, per - i + 1)         # 本批张数
                rng = f"第{i}张" if n == 1 else f"第{i}~{i+n-1}张"
                cur = f"{m.get('name', m['id'])} | 提示词{pi} | {rng}"
                write_status(cur_model=m.get("name", m["id"]), cur_prompt=pi, cur_img=i,
                             done=done, total=total, ok=ok, fail=fail,
                             folder=f"output/imgtest/{folder}", msg=cur, t0=time.time())
                log(f"[{done+1}/{total}] {cur}" + (" [批量]" if n > 1 else ""))
                name = _safe(f"test_{t['id']}_{m['id']}_p{pi}_{i}_{int(time.time())}")
                try:
                    files, sec = gen_wait(ptext, m["id"], W, H, name,
                                          ref=pad_server, ipa=ipa_server, batch=n)
                    for k, f in enumerate(files):
                        shutil.copy2(f, os.path.join(ddir, f"p{pi:02d}_{i+k}.png"))
                    ok += len(files)
                    log(f"  ✓ 完成{len(files)}张({int(sec)}秒) → output/imgtest/{folder}/p{pi:02d}_{i}" + (f"~{i+len(files)-1}.png" if len(files) > 1 else ".png"))
                    done += len(files)
                except RuntimeError:
                    stop = "all"; break
                except Exception as e:
                    fail += n; done += n
                    log(f"  ✗ {e}")
                write_status(done=done, ok=ok, fail=fail)
                i += n
            if stop:
                break

    t["state"] = "killed" if stop else "done"
    t["finished"] = time.time()
    save_task(task_path, t)
    write_status(done=done, ok=ok, fail=fail)
    log(f"{'⏹ 任务被杀' if stop else '🏁 任务结束'}: {t['id']} 成功 {ok}/{total}")
    return stop

# --------------- 主循环: 队列跑空自动退出 ---------------

def main():
    open(PID_F, "w").write(str(os.getpid()))
    clear_control()
    write_status(running=True, state="idle", msg="worker 启动", finished=False)
    log(f"🔧 测试场 worker 启动 pid={os.getpid()}")
    # 续跑: 上次被杀时正在 running 的任务拉回 queued,下次启动接着跑
    # (暂停后杀 worker / 选4 中断都算;已完成图保留,该任务从头重跑)
    for fn in sorted(os.listdir(QUEUE)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(QUEUE, fn)
        try:
            t = load_task(p)
        except Exception:
            continue
        if t.get("state") == "running":
            t["state"] = "queued"
            save_task(p, t)
            log(f"↩ 任务 {t.get('id')} 上次被中断,已重新排队")
    if not cg.ensure_img_service():
        write_status(running=False, state="error", msg="生图服务起不来", finished=True)
        return
    stop_all = ""
    while True:
        if read_control() == "kill_all":
            stop_all = "all"; break
        path, t = next_task()
        if not path:
            break                          # 队列空了 → 正常结束
        r = run_task(path, t)
        if r == "all":
            stop_all = "all"; break
    try:
        os.remove(PID_F)
    except OSError:
        pass
    if stop_all:
        write_status(running=False, state="killed", msg="已被杀死", finished=True)
        log("🛑 worker 被杀死")
    else:
        write_status(running=False, state="done", msg="生图结束", finished=True)
        log("🎉 生图结束(队列已空),worker 退出")

if __name__ == "__main__":
    main()
