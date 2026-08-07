#!/usr/bin/env python3
# ============================================================
# comic_gen.py · 模板漫画连载一键生成(健康向)
#   用法: python3 comic_gen.py   (网页版: 图片区 → 漫画连载 标签)
#   输入: my_story/story.txt (### Image 编号 / Prompt: 英文 / Dialogue: 中文)
#   可选: my_story/ref.png 存在 → 每格按 0.5 以图生图,固定男主形象(尺寸已按目标 W×H 校正)
#   产物: output/my_story_comic/001.png ~ NNN.png 竖版单格(多角色左右漫画气泡)
#         + page_01.png… 条漫长页(每页 PER_PAGE 格竖向拼接)
# 流程: 起生图服务(内存互斥会停语言模型) → 逐张真等完成 → Pillow 加台词 → 归档
# 引擎: 画风预设/解析/四型气泡/条漫拼接 在 comic_engine.py(与网页版 gen.py 共用)
# ============================================================
import os, re, json, time, shutil, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc
import gen as genmod

# 漫画引擎抽到独立模块,避免与 gen.py 循环 import(gen.py 也 import comic_engine)
from comic_engine import STYLE_PREFIX, parse_story, add_dialogue, assemble_pages

STORY   = os.path.join(BASE, "my_story", "story.txt")
REF     = os.path.join(BASE, "my_story", "ref.png")   # 可选: 男主真人参考图
OUT_DIR = os.path.join(BASE, "output", "my_story_comic")
W, H    = 832, 1216   # 竖版分镜(条漫),拼页时竖向堆叠

# --------------- 1. 提交一张并等完成 ---------------

def upload_ref(local_png):
    """把本地参考图传进 ComfyUI,返回服务端文件名(作为 i2i 垫图)。"""
    import urllib.request, uuid
    bd = uuid.uuid4().hex
    data = open(local_png, "rb").read()
    body  = f'--{bd}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode()
    body += (f'--{bd}\r\nContent-Disposition: form-data; name="image"; filename="ref.png"\r\n'
             f'Content-Type: image/png\r\n\r\n').encode() + data + b'\r\n'
    body += f'--{bd}--\r\n'.encode()
    req = urllib.request.Request("http://127.0.0.1:8849/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={bd}"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read()).get("name")

def gen_one(prompt, model_name=None, ref_name=None, style_prefix=None, ipa_name=None, ipa_weight=0.3):
    """走正轨 genmod.submit(登记 TASKS + wait_done 线程) → 轮询等完成 → 返回输出路径。
    model_name/style_prefix 缺省时用默认(waiIllustrious + 全局 STYLE_PREFIX),向后兼容。
    ipa_name=角色参考图(服务端文件名)时走 IPAdapter 锁脸,与 i2i 垫图互斥(ipa 优先)。"""
    model_name = model_name or "waiIllustriousSDXL_v170.safetensors"
    prompt = (style_prefix or STYLE_PREFIX) + ", " + prompt   # 注入参考画风,角色穿扮由故事提示词定
    name = re.sub(r"[^\w.-]+", "_", f"comic_{int(time.time()*10)}")
    if ipa_name:
        pid = genmod.submit(model_name, prompt, genmod.NEG_DEFAULT, W, H, name,
                            ipa=ipa_name, ipa_weight=ipa_weight)
    elif ref_name:
        pid = genmod.submit(model_name, prompt, genmod.NEG_DEFAULT, W, H, name,
                            "i2i", ref_name, None, 0.5)
    else:
        pid = genmod.submit(model_name, prompt, genmod.NEG_DEFAULT, W, H, name)
    t0 = time.time()
    while time.time() - t0 < 1800:
        t = genmod.TASKS.get(pid, {})
        if t.get("done"):
            src = os.path.join(genmod.OUT_DIR, name + ".png")
            if os.path.exists(src):
                return src
            raise Exception("完成了但找不到输出文件")
        if t.get("error"):
            raise Exception(t["error"])
        time.sleep(5)
    raise Exception("Timeout")

# --------------- 2. 服务/参考图(批量级,一次性) ---------------

def ensure_img_service():
    """起生图服务(内存互斥: 会先停语言模型)。返回是否就绪。"""
    print("启动生图服务…")
    svc.start_svc("img")
    for _ in range(60):
        if svc.svc_status("img")["running"]:
            print("✔ 生图服务就绪")
            return True
        time.sleep(3)
    print("⚠ 生图服务可能未就绪,请检查 comfy.log")
    return False

def upload_ref_once(ref_path=REF):
    """有本地参考图就上传一次,返回服务端文件名;没有/失败返回 None(纯文字)。"""
    if not os.path.exists(ref_path):
        return None
    try:
        name = upload_ref(ref_path)
        print(f"✔ 参考图已上传: {name}")
        return name
    except Exception as e:
        print(f"⚠ 参考图上传失败,改纯文字生成: {e}")
        return None

# --------------- 3. 跑一个故事(批量/网页复用) ---------------

def run_story(story_path, out_dir, ref_name=None, model_name=None, style_prefix=None, ipa_name=None):
    """生成一个故事: 逐格真等完成→加台词气泡→归档→拼条漫。返回 (成功, 失败, 页列表)。
    model_name/style_prefix 可指定本故事用的模型与画风(缺省走 gen_one 默认)。
    ipa_name=角色参考图(IPAdapter 锁脸,治"男主一会一个样")。"""
    images = parse_story(story_path)
    os.makedirs(out_dir, exist_ok=True)
    ok = failed = 0
    for num, prompt, dialogues in images:
        fname = f"{num:03d}.png"
        out   = os.path.join(out_dir, fname)
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            ok += 1
            continue
        preview = dialogues[0][1][:36] if dialogues else prompt[:36]
        print(f"  [{num:03d}/{len(images)}] {preview}...")
        try:
            src = gen_one(prompt, model_name=model_name, ref_name=ref_name, style_prefix=style_prefix, ipa_name=ipa_name)
            shutil.copy2(src, out)
            if dialogues:
                add_dialogue(out, dialogues, out)
            ok += 1
        except Exception as e:
            print(f"  ✗ [{num:03d}] {e}")
            failed += 1
        time.sleep(2)
    try:
        pages = assemble_pages(out_dir, len(images))
    except Exception as e:
        print(f"  ⚠ 条漫拼接失败(单格仍在): {e}")
        pages = []
    return ok, failed, pages

# --------------- 4. 主函数(单故事: my_story/story.txt) ---------------

def main():
    images = parse_story(STORY)
    use_ref = os.path.exists(REF)
    print("=" * 56)
    print(f"漫画故事: {STORY}")
    print(f"输出目录: {OUT_DIR}")
    print(f"图数:     {len(images)} 张")
    print(f"模型:     waiIllustriousSDXL_v170")
    print(f"男主形象: {'ref.png 以图生图 0.5' if use_ref else '纯文字提示词'}")
    print("=" * 56)
    ensure_img_service()
    ref_name = upload_ref_once() if use_ref else None
    ok, failed, pages = run_story(STORY, OUT_DIR, ref_name)
    print("\n" + "=" * 56)
    print(f"漫画生成完毕! 成功 {ok}/{len(images)},失败 {failed}")
    print(f"单格: {OUT_DIR}/001.png ~ {len(images):03d}.png")
    if pages:
        print(f"条漫: {len(pages)} 页 → " + ", ".join(os.path.basename(x) for x in pages))
    print("=" * 56)

if __name__ == "__main__":
    main()
