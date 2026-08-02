#!/usr/bin/env python3
# ============================================================
# comic_gen.py · 校园热血番漫画连载一键生成(健康向)
#   用法: python3 comic_gen.py   (网页版: 图片区 → 漫画连载 标签)
#   产物: output/my_story_comic/001.png ~ NNN.png (带台词气泡)
# 流程: 起生图服务(内存互斥会停语言模型) → 逐张真等完成 → Pillow 加台词 → 归档
# 注意: 台词是中文,必须用 PingFang 等中文字体,Helvetica 会出豆腐块。
# ============================================================
import os, re, json, time, shutil, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc
import gen as genmod

STORY   = os.path.join(BASE, "my_story", "story.txt")
OUT_DIR = os.path.join(BASE, "output", "my_story_comic")
W, H    = 1024, 720

# --------------- 1. 解析 story.txt ---------------

def parse_story(path):
    """story.txt → [(num, prompt, dialogue)]"""
    text = open(path, encoding="utf-8").read()
    images = []
    pattern = (
        r"### Image (\d+)"
        r"\s*\nPrompt:\s*(.*?)"
        r"\s*\n\s*Dialogue:\s*(.*?)"
        r"(?=\n\n|\n### Image |$)"
    )
    for num_str, prompt, dialogue in re.findall(pattern, text, re.DOTALL):
        images.append((int(num_str), prompt.strip(), dialogue.strip()))
    images.sort()
    return images

# --------------- 2. 提交一张并等完成 ---------------

def gen_one(prompt, model_name="waiIllustriousSDXL_v170.safetensors"):
    """走正轨 genmod.submit(登记 TASKS + wait_done 线程) → 轮询等完成 → 返回输出路径。"""
    name = re.sub(r"[^\w.-]+", "_", f"comic_{int(time.time())}")
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

# --------------- 3. Pillow 加台词 ---------------

def add_dialogue(image_path, dialogue, out_path):
    """底部叠加台词气泡。"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    h = img.size[1]
    bh = min(h * 0.18, 180)
    bx, by = 30, h - bh - 10
    bw = img.size[0] - 60
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=15,
                           fill=(20, 15, 30, 180),
                           outline=(255, 255, 245, 255), width=2)
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
    words = dialogue.split()
    lines, cur = [], ""
    for w in words:
        test = cur + " " + w if cur else w
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] > bw - 20:
            lines.append(cur); cur = w
        else:
            cur = test
    lines.append(cur)
    tx = bx + 20
    ty = by + (bh - len(lines) * 26) / 2
    for ln in lines:
        draw.text((tx, ty), ln, fill=(255, 255, 245, 255), font=font)
        ty += 26
    img.save(out_path, "PNG")

# --------------- 4. 主函数 ---------------

def main():
    images = parse_story(STORY)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 56)
    print(f"漫画故事: {STORY}")
    print(f"输出目录: {OUT_DIR}")
    print(f"图数:     {len(images)} 张")
    print(f"模型:     waiIllustriousSDXL_v170")
    print("=" * 56)

    # 起生图服务
    print("启动生图服务…")
    svc.start_svc("img")
    for _ in range(60):
        if svc.svc_status("img")["running"]:
            print("✔ 生图服务就绪")
            break
        time.sleep(3)
    else:
        print("⚠ 生图服务可能未就绪,请检查 comfy.log")

    ok = failed = 0
    for i, (num, prompt, dialogue) in enumerate(images):
        fname = f"{num:03d}.png"
        out   = os.path.join(OUT_DIR, fname)
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            print(f"  [{num:03d}/{len(images)}] SKIP")
            ok += 1
            continue
        print(f"\n[{num:03d}/{len(images)}] {dialogue[:70]}...")
        try:
            src = gen_one(prompt)
            shutil.copy2(src, out)
            add_dialogue(out, dialogue, out)
            print(f"  ✓ ({os.path.getsize(out)//1024}KB)")
            ok += 1
        except Exception as e:
            print(f"  ✗ {e}")
            failed += 1
        time.sleep(3)

    print("\n" + "=" * 56)
    print(f"漫画生成完毕!")
    print(f"  成功: {ok}/{len(images)}")
    print(f"  失败: {failed}")
    print(f"产物: {OUT_DIR}/001.png ~ 0{len(images)}.png")
    print("=" * 56)

if __name__ == "__main__":
    main()
