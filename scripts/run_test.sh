#!/bin/bash

# 运行质量测试脚本
# 你需要先确保 ComfyUI 在 8849 端口运行

echo "开始运行质量测试..."
echo "请确保 ComfyUI 已在 8849 端口运行"

# 检查 ComfyUI 是否运行
if ! curl -s http://127.0.0.1:8849 > /dev/null; then
    echo "错误: ComfyUI 没有在 8849 端口运行"
    echo "请先启动 ComfyUI"
    exit 1
fi

echo "正在运行测试脚本..."
python3 /Users/wl/Desktop/ImageVideoStudio/scripts/quality_test_safe.py

echo "测试完成!"
