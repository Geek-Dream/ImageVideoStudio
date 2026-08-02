#!/usr/bin/env python3
"""
安全的质量测试脚本 - 用于生成符合内容政策的图片
7个模型 × 5个场景 × 4种风格 = 140张图
调用 ComfyUI (127.0.0.1:8849) 的文生图接口。
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import svc, gen as genmod

# ──────────────────────────────────────────────
# 7个模型
# ──────────────────────────────────────────────
MODELS = [
    "waiIllustriousSDXL_v170.safetensors",
    "waiNSFWIllustrious_v140.safetensors",
    "NoobAI-XL-Vpred-v1.0.safetensors",
    "RealVisXL_V5.0_Lightning_fp16.safetensors",
    "CyberRealisticXLPlay_V10.0_FP16.safetensors",
    "zimage",
    "flux",
]

# 安全的负面提示词
NEG = "blurry, low quality, worst quality, watermark, text, deformed, ugly, nsfw"

# ──────────────────────────────────────────────
# 5个场景 × 4种风格 = 20张提示词 (安全内容)
# ──────────────────────────────────────────────
PROMPTS = {
    # 场景1: 人物互动 (安全内容)
    "s1_pixel": (
        "A fantasy character with magical powers, glowing aura, pixel art style, "
        "detailed background, 720p resolution"
    ),
    "s1_furry": (
        "A cute furry character with anthropomorphic features, fantasy setting, "
        "furry style, bright colors, 720p resolution"
    ),
    "s1_gothic": (
        "A mysterious character in dark fantasy setting, gothic style, dramatic "
        "lighting, detailed anatomy, 720p resolution"
    ),
    "s1_mlp": (
        "A cartoon character with bright colors and friendly expression, MLP "
        "cartoon style, soft lighting, 720p resolution"
    ),

    # 场景2: 室内场景
    "s2_pixel": (
        "A cozy interior with warm lighting, pixel art style, detailed furniture, "
        "720p resolution"
    ),
    "s2_furry": (
        "A friendly furry character in a cozy room, furry style, soft lighting, "
        "720p resolution"
    ),
    "s2_gothic": (
        "A dark and mysterious interior with gothic elements, gothic style, "
        "dramatic lighting, 720p resolution"
    ),
    "s2_mlp": (
        "A cheerful cartoon interior with pastel colors, MLP cartoon style, "
        "bright colors, 720p resolution"
    ),

    # 场景3: 自然风景
    "s3_pixel": (
        "A beautiful landscape with mountains and forests, pixel art style, "
        "detailed trees, 720p resolution"
    ),
    "s3_furry": (
        "A furry character exploring nature, furry style, natural background, "
        "720p resolution"
    ),
    "s3_gothic": (
        "A dark forest with mysterious atmosphere, gothic style, dramatic lighting, "
        "720p resolution"
    ),
    "s3_mlp": (
        "A whimsical garden with colorful flowers, MLP cartoon style, bright colors, "
        "720p resolution"
    ),

    # 场景4: 城市环境
    "s4_pixel": (
        "A modern cityscape with skyscrapers, pixel art style, detailed buildings, "
        "720p resolution"
    ),
    "s4_furry": (
        "A furry character in a city environment, furry style, urban background, "
        "720p resolution"
    ),
    "s4_gothic": (
        "A dark city with gothic architecture, gothic style, dramatic lighting, "
        "720p resolution"
    ),
    "s4_mlp": (
        "A colorful city with cartoon-style buildings, MLP cartoon style, "
        "bright colors, 720p resolution"
    ),

    # 场景5: 抽象艺术
    "s5_pixel": (
        "Abstract art with geometric patterns, pixel art style, colorful shapes, "
        "720p resolution"
    ),
    "s5_furry": (
        "Abstract furry character with artistic elements, furry style, creative "
        "design, 720p resolution"
    ),
    "s5_gothic": (
        "Dark abstract art with gothic elements, gothic style, mysterious "
        "atmosphere, 720p resolution"
    ),
    "s5_mlp": (
        "Colorful abstract art with MLP style, bright colors, playful design, "
        "720p resolution"
    ),
}

SCENE_NAMES = {
    "s1": "人物互动",
    "s2": "室内场景",
    "s3": "自然风景",
    "s4": "城市环境",
    "s5": "抽象艺术",
}
STYLE_NAMES = {
    "pixel": "像素风格",
    "furry": "Furry风格",
    "gothic": "哥特黑暗风",
    "mlp": "MLP花园风格",
}

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
def start_img():
    """启动生图服务。"""
    st = svc.svc_status("img")
    if not st["running"]:
        svc.start_svc("img")
        print("  启动生图服务...")
        time.sleep(8)

def generate_image(prompt, model_id):
    """调用 ComfyUI 生成单张图片，正确引用工作流中的节点。"""
    model_info = genmod.find_entry(model_id)
    if not model_info:
        model_info = genmod.find_entry("waiIllustriousSDXL_v170.safetensors")
        print(f"  ⚠ 模型 {model_id} 未找到，回退到默认模型")

    wf = genmod.build_wf(
        model_info, pos=prompt, neg=NEG,
        w=1024, h=720, seed=int(time.time() * 1000), mode="t2i"
    )
    # 正确: 保存节点["7"](VAEDecode输出),不是["3"](CLIPTextEncode输出)
    wf["99"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "gen", "images": ["7", 0]}}

    url = f"http://127.0.0.1:{genmod.IMG_PORT}/prompt"
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result

def main():
    print("=" * 60)
    print("安全质量测试 · 7模型 × 5场景 × 4风格 = 140张图")
    print("=" * 60)

    start_img()

    total = len(MODELS) * len(PROMPTS)  # 20张/模型 × 7模型 = 140
    count = 0
    success = 0
    failed = 0

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"模型: {model}")
        print(f"{'='*60}")

        for scene_key, prompt in sorted(PROMPTS.items()):
            count += 1
            scene_id = scene_key[:2]   # "s1"
            style = scene_key[3:]      # "pixel"
            scene_name = SCENE_NAMES.get(scene_id, scene_id)
            style_name = STYLE_NAMES.get(style, style)
            print(f"[{count}/{total}] {model} | {scene_name}({style_name})")
            print(f"  {prompt[:90]}...")

            try:
                result = generate_image(prompt, model)
                print(f"  ✓ 成功")
                success += 1
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                failed += 1
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"完成! 成功 {success} 张, 失败 {failed} 张, 共 {count} 张")
    print(f"输出目录: {genmod.OUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
