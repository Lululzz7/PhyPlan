#!/usr/bin/env bash

# Qwen3-VL 点云推理脚本（单样本或批量评估）
# 用法（单样本）：直接修改下方变量或以环境变量方式覆盖，然后执行本脚本
# 用法（批量）：设置 DATA=/path/to/annotations.json[l]（见 docs/pointcloud_inputs_zh.md）

set -euo pipefail

########################################
# 基本配置（可按需覆盖为环境变量）
########################################

# 模型路径或 HuggingFace ID
MODEL=${MODEL:-"/e2e-data/embodied-research-data/large_model/huggingface/hub/Qwen3VL-8B"}

# 注意力实现（可选：flash_attention_2/sdpa/eager；留空使用模型默认）
ATTN_IMPL=${ATTN_IMPL:-"flash_attention_2"}

# 精度：优先 bfloat16；如设备不支持可改为 fp16
USE_BF16=${USE_BF16:-"true"}   # true/false
USE_FP16=${USE_FP16:-"false"}   # true/false

# 生成参数
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

########################################
# 点云与视觉输入（单样本模式）
########################################

# 单点云文件：.npy/.npz/.ply/.pcd；(N,3) 或 (N,6)（颜色将统一归一到 [0,1]）
POINT_CLOUD=${POINT_CLOUD:-"/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/qwen-vl-finetune/demo/points/scene0000_01_vh_clean.ply"}

# 文本提示（无需写 <point> 占位符）
PROMPT=${PROMPT:-"Please describe the object category, structure, and key parts in detail based on the point cloud."}

# 可选视觉输入（绝对路径）；如有多张图像/视频，用逗号分隔
IMAGE=${IMAGE:-""}
IMAGES=${IMAGES:-""}
VIDEOS=${VIDEOS:-""}

# 点云骨干配置（需与训练一致）
POINT_BACKBONE=${POINT_BACKBONE:-"PointBERT"}
POINT_BACKBONE_CONFIG=${POINT_BACKBONE_CONFIG:-"PointTransformer_8192point_2layer"}
POINT_USE_COLOR=${POINT_USE_COLOR:-"true"}   # true=保留 RGB；false=仅 xyz
POINT_NUM=${POINT_NUM:-0}                  # 超出则远点采样到该点数

# 可选：加载训练时的骨干与 projector 权重
POINT_BACKBONE_CKPT=${POINT_BACKBONE_CKPT:-"/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/checkpoints/point_bert_v1.2.pt"}  # 如 /path/to/point_bert_v1.2.pt
POINT_PROJ_CKPT=${POINT_PROJ_CKPT:-"/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/outputs/pc_from_scratch/point_proj.bin"}          # 如 /path/to/point_proj_extracted.pt

# 适配器前向开关：true=在 forward 中计算嵌入；false=离线预先提取嵌入
PC_ADAPTER_FORWARD=${PC_ADAPTER_FORWARD:-"false"}

########################################
# 批量评估模式（设置 DATA 即启用）
########################################
# 标注文件（json/jsonl，字段结构同训练）；若设置则忽略单样本的 POINT_CLOUD/PROMPT
DATA=${DATA:-"qwen-vl-finetune/demo/pointcloud_qa_min.json"}
SAVE_PATH=${SAVE_PATH:-""}   # 可选：保存结果 JSONL 路径；留空则按脚本默认

########################################
# 构造命令并执行
########################################

ENTRY=qwenvl/train/infer_qwen_pointcloud.py
[[ -f "$ENTRY" ]] || { echo "[ERROR] 找不到入口脚本：$ENTRY"; exit 1; }

python_args=("$ENTRY" "--model" "$MODEL" "--max-new-tokens" "$MAX_NEW_TOKENS" "--temperature" "$TEMPERATURE" "--top-p" "$TOP_P" "--use-generate")

# 精度
if [[ "$USE_BF16" == "true" ]]; then
  python_args+=("--bf16")
fi
if [[ "$USE_FP16" == "true" ]]; then
  python_args+=("--fp16")
fi

# 注意力实现
if [[ -n "$ATTN_IMPL" ]]; then
  python_args+=("--attn-implementation" "$ATTN_IMPL")
fi

# 点云骨干配置
python_args+=("--point-backbone" "$POINT_BACKBONE" "--point-backbone-config" "$POINT_BACKBONE_CONFIG" "--point-num" "$POINT_NUM")
if [[ "$POINT_USE_COLOR" == "true" ]]; then
  python_args+=("--point-use-color")
fi
if [[ -n "$POINT_BACKBONE_CKPT" ]]; then
  python_args+=("--point-backbone-ckpt" "$POINT_BACKBONE_CKPT")
fi
if [[ -n "$POINT_PROJ_CKPT" ]]; then
  python_args+=("--point-proj-ckpt" "$POINT_PROJ_CKPT")
fi
if [[ "$PC_ADAPTER_FORWARD" == "true" ]]; then
  python_args+=("--pc-adapter-forward")
fi

if [[ -n "$DATA" ]]; then
  # 批量评估模式
  python_args+=("--data" "$DATA")
  if [[ -n "$SAVE_PATH" ]]; then
    python_args+=("--save-path" "$SAVE_PATH")
  fi
else
  # 单样本模式：要求至少提供点云文件
  if [[ -z "$POINT_CLOUD" ]]; then
    # 若未通过环境变量提供，检查是否在 CLI 中已传入 --point-cloud
    if [[ "$*" != *"--point-cloud"* ]]; then
      echo "[ERROR] 未提供 POINT_CLOUD。请设置 POINT_CLOUD=/abs/path/to/points.npy 或在命令行传入 --point-cloud，或使用批量模式 DATA=/path/to/annotations.json[l]"
      exit 2
    fi
  else
    python_args+=("--point-cloud" "$POINT_CLOUD")
  fi
  # 提示可由环境变量或 CLI 提供；若环境变量存在则加入
  if [[ -n "$PROMPT" ]]; then
    python_args+=("--prompt" "$PROMPT")
  fi

  # 可选视觉输入
  if [[ -n "$IMAGE" ]]; then
    python_args+=("--image" "$IMAGE")
  fi
  if [[ -n "$IMAGES" ]]; then
    python_args+=("--images" "$IMAGES")
  fi
  if [[ -n "$VIDEOS" ]]; then
    python_args+=("--videos" "$VIDEOS")
  fi
fi

echo "[INFO] Running: python ${python_args[*]} $@"
python "${python_args[@]}" "$@"

# 备注：
# - 若使用 .ply/.pcd，请确保已安装 open3d。
# - 点云仅首步注入；后续由缓存驱动增量解码。
# - 点云骨干与 YAML 配置需与训练一致，否则 token 长度可能不匹配。
