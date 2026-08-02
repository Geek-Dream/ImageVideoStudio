#!/usr/bin/env python3
# ============================================================
# quality_test.py · 图片模型质量批量测试(逐张真等完成 + 自动归档)
#   用法: python3 quality_test.py
#   提示词: 全部在 prompts/ 文件夹,一个 .txt 一张图,随时改;
#           改完重跑本脚本即可。_negative.txt 是公共负向提示词。
#   产物:   output/organized/<模型名>/<提示词名>.png
# 流程: 起生图服务(会先停语言/视频模型腾内存) → 逐模型逐提示词提交
#       → 轮询等真正完成(不是入队就算完) → 归档。
# ============================================================
import os, re, shutil, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc
import gen as genmod

PROMPT_DIR = os.path.join(BASE, "prompts")
ORG_DIR = os.path.join(BASE, "output", "organized")
W, H = 1024, 720            # 出图尺寸(720p)
POLL_EVERY = 3              # 轮询间隔(秒)
TIMEOUT = 1800              # 单张超时(秒)

def load_prompts():
    """prompts/*.txt → [(名字, 正向提示词)] + 公共负向。下划线开头的是配置不算图。"""
    neg = genmod.NEG_DEFAULT
    negp = os.path.join(PROMPT_DIR, "_negative.txt")
    if os.path.exists(negp):
        neg = open(negp, encoding="utf-8").read().strip()
    out = []
    for fn in sorted(os.listdir(PROMPT_DIR)):
        if fn.endswith(".txt") and not fn.startswith("_"):
            txt = open(os.path.join(PROMPT_DIR, fn), encoding="utf-8").read().strip()
            if txt:
                out.append((fn[:-4], txt))
    return out, neg

def wait_service():
    print("启动生图服务(会先停掉语言/视频模型腾内存)…")
    svc.start_svc("img")
    for _ in range(120):
        if svc.svc_status("img")["running"]:
            print("✔ 生图服务就绪")
            return True
        time.sleep(3)
    print("❌ 生图服务启动超时,看 comfy.log")
    return False

def safe(s):
    return re.sub(r"[^\w.-]+", "_", s)

def run_one(model_id, pname, pos, neg, dest_dir):
    """提交一张并【轮询等真完成】,成功后复制到归档目录。返回 (是否成功, 耗时秒/错误)。"""
    name = safe(f"{model_id}_{pname}_{int(time.time())}")
    try:
        pid = genmod.submit(model_id, pos, neg, W, H, name)
    except Exception as e:
        return False, str(e)
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        t = genmod.TASKS.get(pid, {})
        if t.get("done"):
            src = os.path.join(genmod.OUT_DIR, name + ".png")
            if not os.path.exists(src):
                return False, "完成了但找不到输出文件"
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dest_dir, pname + ".png"))
            return True, int(time.time() - t0)
        if t.get("error"):
            return False, t["error"]
        time.sleep(POLL_EVERY)
    return False, f"超时({TIMEOUT}s)"

def main():
    prompts, neg = load_prompts()
    models = genmod.list_models()
    if not prompts:
        print("prompts/ 里没有提示词文件"); return
    if not models:
        print("没有可用图片模型(检查 models/image/)"); return
    total = len(models) * len(prompts)
    print("=" * 56)
    print(f"质量测试: {len(models)} 模型 × {len(prompts)} 提示词 = {total} 张")
    print("逐张生成: 一张真正画完才提交下一张,完成即归档")
    print("=" * 56)
    if not wait_service():
        return
    n = ok = 0
    report = []
    for m in models:
        mdir = os.path.join(ORG_DIR, safe(m["name"]))
        for pname, pos in prompts:
            n += 1
            print(f"\n[{n}/{total}] {m['name']} | {pname}")
            good, info = run_one(m["id"], pname, pos, neg, mdir)
            if good:
                ok += 1
                print(f"  ✓ 完成({info}秒)→ {os.path.join(mdir, pname + '.png')}")
            else:
                print(f"  ✗ 失败: {info}")
            report.append(f"{'OK ' if good else 'FAIL'}\t{m['name']}\t{pname}\t{info}")
    os.makedirs(ORG_DIR, exist_ok=True)
    with open(os.path.join(ORG_DIR, "_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"质量测试报告 · {time.strftime('%Y-%m-%d %H:%M')}\n成功 {ok}/{total}\n\n" + "\n".join(report))
    print("\n" + "=" * 56)
    print(f"全部结束: 成功 {ok}/{total}")
    print(f"产物: {ORG_DIR}/<模型名>/   报告: {ORG_DIR}/_report.txt")

if __name__ == "__main__":
    main()
