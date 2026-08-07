#!/usr/bin/env python3
# ============================================================
# comic_engine.py · 漫画引擎(纯 PIL + stdlib,无 gen/svc 依赖)
#   被 comic_gen.py(命令行)与 gen.py(网页漫画连载)共用,避免循环 import。
#   内容: 画风预设 / story 与台词解析 / 四型气泡(柔影融合) / 条漫拼接。
#   不碰生图提交、服务启停、参考图上传(那些留在 comic_gen.py / gen.py)。
# ============================================================
import os, re

# 每页拼几个分镜(条漫)
PER_PAGE = 4

# 默认画风(油亮动漫风,从参考图提取的中性画风;角色穿扮由故事提示词决定,保证健康向)
STYLE_PREFIX = ("masterpiece, best quality, very aesthetic, absurdres, "
                "glossy anime cel shading, shiny skin, soft smooth skin gradient, "
                "detailed glossy hair, fine hair strand highlights, "
                "large expressive eyes, sparkling detailed irises, delicate blush, "
                "vibrant saturated colors, soft rim lighting, screentones, clean lineart")

# 画风预设: key → (中文名, 英文提示词前缀)。custom 自定义框非空时优先于预设。
STYLE_PRESETS = {
    "glossy": ("油亮动漫风", STYLE_PREFIX),
    "dark":   ("黑暗蝙蝠侠风",
               "masterpiece, best quality, dark comic book style, dramatic chiaroscuro, "
               "heavy shadows, gritty noir, deep blacks, bold inking, halftone dots, "
               "moody cinematic lighting"),
    "fresh":  ("清新日常风",
               "masterpiece, best quality, soft watercolor, gentle pastel colors, "
               "warm sunlight, slice of life, clean simple lineart, airy atmosphere, "
               "light screentones"),
    "pixel":  ("像素风",
               "pixel art style, retro game aesthetic, crisp pixel clusters, "
               "limited vibrant color palette, clean dithering, sharp sprite outlines, "
               "detailed 2d game illustration, charming pixel scene"),
    "ink":    ("水墨线条风",
               "traditional chinese ink wash painting, sumi-e style, bold black brush strokes, "
               "minimalist flowing lineart, monochrome with subtle color accent, "
               "rice paper texture, elegant negative space"),
}

def style_prefix(key="", custom=""):
    """返回要前置到提示词的画风串。custom 非空优先;key 空/未知 → glossy。"""
    if custom and custom.strip():
        return custom.strip()
    return STYLE_PRESETS.get(key, STYLE_PRESETS["glossy"])[1]

# 气泡配色(白底微透 225 → 融入画面;描边细 2)
PINK   = (255, 110, 160, 255)   # 爱心粉
DARK   = (28, 26, 38, 255)      # 描边/文字深色
WHITE  = (255, 255, 255, 225)   # 气泡白底(微透)
GOLD   = (255, 236, 170, 255)   # 旁白彩字
INKBOX = (22, 20, 30, 222)      # 旁白黑框

# 气泡柔影(让气泡像"长在"画面上,而不是浮在上面)
SHADOW_OFF   = (5, 9)           # 阴影偏移(右下)
SHADOW_BLUR  = 9                # 阴影高斯模糊半径
SHADOW_COLOR = (18, 14, 26, 255)

# --------------- 1. 解析 story.txt / 台词 ---------------

def parse_dialogue(dialogue):
    """「名字:台词」… → [(谁, 台词)];无「」整段一个气泡;空 → []"""
    pairs = re.findall(r"「([^:：」]+)[:：]([^」]*)」", dialogue)
    if pairs:
        return [(w.strip(), l.strip()) for w, l in pairs]
    if dialogue.strip():  # 兼容旧格式: 整段一个气泡
        return [("", dialogue.strip())]
    return []

def parse_story(path):
    """story.txt → [(num, prompt, dialogues)],dialogues = [(谁, 台词), ...](可为空名字)"""
    text = open(path, encoding="utf-8").read()
    images = []
    pattern = (
        r"### Image (\d+)"
        r"\s*\nPrompt:\s*(.*?)"
        r"\s*\n\s*Dialogue:\s*(.*?)"
        r"(?=\n\n|\n### Image |$)"
    )
    for num_str, prompt, dialogue in re.findall(pattern, text, re.DOTALL):
        images.append((int(num_str), prompt.strip(), parse_dialogue(dialogue)))
    images.sort()
    return images

# --------------- 2. Pillow 气泡绘制 helper ---------------

# 字体库(气泡编辑器自选字体): 用户把 .ttf/.otf/.ttc 放这,前端预览与压平共用同一文件
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

def list_fonts():
    """fonts/ 里的字体 → [{'file','name'}](file 压平用,name 前端 @font-face 用);没有就 []。"""
    out = []
    if os.path.isdir(FONTS_DIR):
        for f in sorted(os.listdir(FONTS_DIR)):
            if f.lower().endswith((".ttf", ".otf", ".ttc")):
                out.append({"file": f, "name": os.path.splitext(f)[0]})
    return out

def _load_font(size, font_file=None):
    """优先用 fonts/ 里指定的字体;未指定/加载失败退到系统中文字体。"""
    from PIL import ImageFont
    cands = [os.path.join(FONTS_DIR, font_file)] if font_file else []
    cands += ["/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti.ttc"]
    for fp in cands:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _wrap_cjk(draw, text, font, max_w):
    """按显示宽度逐字换行(中文没有空格,不能按词分)。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if draw.textlength(cur + ch, font=font) > max_w and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    lines.append(cur)
    return lines

def _heart_pts(cx, cy, s, n=48):
    """心形曲线采样点,(cx,cy) 中心,s≈高度(px)。"""
    import math
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * s / 34.0, cy - y * s / 34.0))
    return pts

def _heart(draw, cx, cy, s, color=PINK):
    draw.polygon(_heart_pts(cx, cy, s), fill=color)

def _vertical_cols(text, max_per_col):
    """文本切成竖排列(每列 ≤max_per_col 字),从右往左读。"""
    cols, cur = [], ""
    for ch in text:
        if ch == "\n":
            if cur:
                cols.append(cur); cur = ""
            continue
        cur += ch
        if len(cur) >= max_per_col:
            cols.append(cur); cur = ""
    if cur:
        cols.append(cur)
    return cols or [""]

def _spiky_pts(x0, y0, x1, y1, spikes=14, jag=0.80):
    """围绕矩形的锯齿爆炸形多边形点。"""
    import math
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2 + 8, (y1 - y0) / 2 + 8
    pts = []
    for i in range(spikes * 2):
        ang = math.pi * i / spikes
        r = 1.0 if i % 2 == 0 else jag
        pts.append((cx + rx * r * math.cos(ang), cy + ry * r * math.sin(ang)))
    return pts

def _draw_vbubble(draw, side, margin, y_top, text, font, Wd, kind):
    """竖排气泡: kind='speech' 圆角白底 / 'shout' 锯齿爆炸;带指向人物的尾巴。返回底部 y。"""
    line_h = int(font.size * 1.26)
    col_w  = int(font.size * 1.42)
    pad = 14
    cols = _vertical_cols(text, 12)
    longest = max(len(c) for c in cols)
    bw = len(cols) * col_w + pad * 2
    bh = longest * line_h + pad * 2
    x0 = margin if side == "left" else Wd - margin - bw
    y0 = y_top
    x1, y1 = x0 + bw, y0 + bh
    # 尾巴(先画,根部压进气泡底边里,不会悬空): 从底边靠内侧指向画面中央(人物)
    if side == "left":
        b0, b1, apex = (x1 - 40, y1 - 8), (x1 - 8, y1 - 8), (x1 + 16, y1 + 36)
    else:
        b0, b1, apex = (x0 + 8, y1 - 8), (x0 + 40, y1 - 8), (x0 - 16, y1 + 36)
    draw.polygon([b0, b1, apex], fill=WHITE)
    if kind == "shout":
        draw.polygon(_spiky_pts(x0, y0, x1, y1), fill=WHITE, outline=DARK, width=2)
    else:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=WHITE, outline=DARK, width=2)
    draw.line([b0, apex], fill=DARK, width=2)
    draw.line([b1, apex], fill=DARK, width=2)
    # 竖排文字(从右往左排,逐字居中)
    cx = x1 - pad - col_w // 2
    for c in cols:
        cy = y0 + pad + font.size // 2
        for ch in c:
            draw.text((cx, cy), ch, font=font, fill=DARK, anchor="mm")
            cy += line_h
        cx -= col_w
    return y1 + 46

def _draw_narration(draw, x, y_bottom, text, font, Wd):
    """旁白/转场: 黑底圆角框+彩字,横排,锚底边向上叠(放画面下角落,避开顶部对白)。返回新的底 y。"""
    pad = 14
    line_h = int(font.size * 1.4)
    lines = _wrap_cjk(draw, text, font, int(Wd * 0.52) - pad * 2)
    bw = int(max(draw.textlength(ln, font=font) for ln in lines)) + pad * 2
    bh = len(lines) * line_h + pad * 2
    y0 = y_bottom - bh
    draw.rounded_rectangle([x, y0, x + bw, y_bottom], radius=10, fill=INKBOX)
    ty = y0 + pad
    for ln in lines:
        draw.text((x + pad, ty), ln, font=font, fill=GOLD)
        ty += line_h
    return y0 - 14

def _draw_love(draw, margin, y_top, text, font, Wd):
    """爱心气泡: 白底横排+粉描边,字左右两侧粉色爱心,尾巴朝下指向人物。返回底部 y。"""
    pad = 18
    line_h = int(font.size * 1.4)
    lines = _wrap_cjk(draw, text, font, int(Wd * 0.5))
    tw = int(max(draw.textlength(ln, font=font) for ln in lines))
    hs = int(font.size * 1.15)                 # 爱心大小
    bw = tw + (hs + 16) * 2 + pad * 2
    bh = len(lines) * line_h + pad * 2
    x0 = (Wd - bw) // 2
    y0 = y_top
    x1, y1 = x0 + bw, y0 + bh
    cxm = Wd // 2
    b0, b1, apex = (cxm - 16, y1 - 2), (cxm + 16, y1 - 2), (cxm, y1 + 42)
    draw.polygon([b0, b1, apex], fill=WHITE)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=WHITE, outline=PINK, width=2)
    draw.line([b0, apex], fill=PINK, width=2)
    draw.line([b1, apex], fill=PINK, width=2)
    ty = y0 + pad
    for ln in lines:
        wln = draw.textlength(ln, font=font)
        draw.text(((Wd - wln) / 2, ty), ln, font=font, fill=DARK)
        ty += line_h
    cy = (y0 + y1) / 2
    _heart(draw, x0 + pad + hs / 2, cy, hs)
    _heart(draw, x1 - pad - hs / 2, cy, hs)
    _heart(draw, x0 + 8, y0 - 12, hs * 0.55)   # 上方飘两颗小爱心
    _heart(draw, x1 - 8, y0 - 18, hs * 0.4)
    return y1 + 48

def add_dialogue(image_path, dialogues, out_path, shadow=True):
    """漫画式对话四型气泡(柔影融合版):
    「旁白:」→黑框彩字(下角落); 带♡→爱心气泡; !结尾→锯齿喊叫; 其余→竖排白底气泡。
    所有气泡画到同一透明 overlay → 派生一层高斯柔影垫底 → overlay 盖上,气泡像长在画面上。"""
    from PIL import Image, ImageDraw, ImageFilter
    base = Image.open(image_path).convert("RGBA")
    Wd, Ht = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(max(24, Wd // 28))
    margin = 24
    y_side = {"left": margin + 6, "right": margin + 6}
    y_narr = Ht - margin - 6   # 旁白锚画面下角落,向上叠,避开顶部对白气泡
    toggle = 0
    for who, text in dialogues:
        love = ("♡" in who) or ("♡" in text)
        who_c = who.replace("♡", "").strip()
        text_c = text.replace("♡", "").strip()
        label = f"{who_c}: {text_c}" if who_c else text_c
        if who_c == "旁白":
            y_narr = _draw_narration(draw, margin, y_narr, text_c, font, Wd)
        elif love:
            yy = _draw_love(draw, margin, max(y_side["left"], y_side["right"]), label, font, Wd)
            y_side["left"] = y_side["right"] = yy
        else:
            kind = "shout" if text_c.endswith(("！", "!")) else "speech"
            side = "left" if toggle % 2 == 0 else "right"
            toggle += 1
            y_side[side] = _draw_vbubble(draw, side, margin, y_side[side], label, font, Wd, kind)
    if shadow:
        alpha = overlay.split()[3]                       # 气泡的不透明度形状
        sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sh.paste(Image.new("RGBA", base.size, SHADOW_COLOR), SHADOW_OFF, alpha)
        sh = sh.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))   # 只糊阴影,不糊字
        base = Image.alpha_composite(base, sh)           # 阴影垫在气泡下
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(out_path, "PNG")

# --------------- 3. 竖向拼成条漫长页 ---------------

def assemble_pages(out_dir, count, per_page=PER_PAGE):
    """把 001..N 单格竖向拼成条漫 page_01.png…(米白底+间隔条+页边距)。返回页文件列表。"""
    from PIL import Image
    files = [os.path.join(out_dir, f"{i:03d}.png")
             for i in range(1, count + 1)
             if os.path.exists(os.path.join(out_dir, f"{i:03d}.png"))]
    if not files:
        return []
    gutter, pm, bg = 18, 28, (245, 242, 238)   # 间隔条 / 页边距 / 米白纸感
    pages = []
    for p in range(0, len(files), per_page):
        imgs = [Image.open(f).convert("RGB") for f in files[p:p + per_page]]
        w = max(im.width for im in imgs)
        total_h = sum(im.height for im in imgs) + gutter * (len(imgs) - 1)
        canvas = Image.new("RGB", (w + pm * 2, total_h + pm * 2), bg)
        yy = pm
        for im in imgs:
            canvas.paste(im, (pm + (w - im.width) // 2, yy))
            yy += im.height + gutter
        name = os.path.join(out_dir, f"page_{p // per_page + 1:02d}.png")
        canvas.save(name, "PNG")
        pages.append(name)
    return pages

# --------------- 4. 气泡编辑器: 按显式坐标压平(可拖拽气泡的导出端) ---------------

def _draw_placed(draw, b, font, W, H):
    """在显式位置画一个气泡(编辑器导出用)。b={x,y,w,text,who,type};x/w 相对图宽、y 相对图高(0~1),
    高度随文字自适应。type: speech对白(圆角+尾) / shout喊叫(锯齿) / love爱心 / narration旁白(黑框)。"""
    pad = 14
    line_h = int(font.size * 1.32)
    bw = max(40, int(b.get("w", 0.3) * W))
    x0 = int(b.get("x", 0.05) * W)
    y0 = int(b.get("y", 0.05) * H)
    who = (b.get("who") or "").strip()
    text = (b.get("text") or "").replace("♡", "").replace("♥", "").strip()  # ♡仅作类型标记,不印出来(系统字体无此字形)
    typ = b.get("type") or "speech"
    label = text if typ == "narration" else (f"{who}: {text}" if who else text)
    lines = _wrap_cjk(draw, label, font, bw - pad * 2)
    bh = len(lines) * line_h + pad * 2
    x1, y1 = x0 + bw, y0 + bh
    if typ == "narration":                      # 旁白: 黑框彩字,无尾巴
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=INKBOX)
        ty = y0 + pad
        for ln in lines:
            draw.text((x0 + pad, ty), ln, font=font, fill=GOLD)
            ty += line_h
        return
    # 尾巴朝下(简化: 拖动气泡定位即够,尾巴固定向下)
    cxm = (x0 + x1) // 2
    b0, b1, apex = (cxm - 14, y1 - 2), (cxm + 14, y1 - 2), (cxm, y1 + 34)
    edge = PINK if typ == "love" else DARK
    draw.polygon([b0, b1, apex], fill=WHITE)
    if typ == "shout":
        draw.polygon(_spiky_pts(x0, y0, x1, y1), fill=WHITE, outline=DARK, width=2)
    else:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=WHITE, outline=edge, width=2)
    draw.line([b0, apex], fill=edge, width=2)
    draw.line([b1, apex], fill=edge, width=2)
    ty = y0 + pad                                # 文字水平居中
    for ln in lines:
        wln = draw.textlength(ln, font=font)
        draw.text(((x0 + x1 - wln) / 2, ty), ln, font=font, fill=DARK)
        ty += line_h
    if typ == "love":                            # 爱心: 上方飘两颗小爱心点缀
        _heart(draw, x0 + 14, y0 - 10, font.size * 0.55)
        _heart(draw, x1 - 14, y0 - 16, font.size * 0.4)

def render_bubbles(image_path, bubbles, out_path, shadow=True):
    """气泡编辑器导出: 把一组显式坐标的气泡(柔影融合)压平到图上,返回 out_path。
    bubbles 元素见 _draw_placed;每个可带 font(fonts/ 里的文件名)选字体。"""
    from PIL import Image, ImageDraw, ImageFilter
    base = Image.open(image_path).convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(18, W // 28)
    for b in bubbles:
        _draw_placed(draw, b, _load_font(size, b.get("font")), W, H)
    if shadow:
        alpha = overlay.split()[3]
        sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sh.paste(Image.new("RGBA", base.size, SHADOW_COLOR), SHADOW_OFF, alpha)
        sh = sh.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
        base = Image.alpha_composite(base, sh)
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(out_path, "PNG")
    return out_path
