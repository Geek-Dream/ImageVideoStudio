#!/bin/bash
# 补帧到60帧 —— 业界标准玩法(ffmpeg minterpolate 运动补偿插值)
# 原理: 模型只生24帧,这里用运动估计把中间帧算出来补到60帧,丝滑且几秒搞定
# 用法: ./interp60.sh 输入.mp4 [输出.mp4]   (不填输出则自动加"_60帧"后缀)
set -e
IN="$1"
if [ -z "$IN" ]; then echo "用法: $0 输入.mp4 [输出.mp4]"; exit 1; fi
if [ ! -f "$IN" ]; then echo "找不到文件: $IN"; exit 1; fi
OUT="${2:-${IN%.*}_60帧.mp4}"
ffmpeg -hide_banner -y -i "$IN" \
  -vf "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
  -c:v libx264 -crf 18 -preset medium -c:a copy "$OUT"
echo "补帧完成 -> $OUT"
