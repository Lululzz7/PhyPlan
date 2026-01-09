#!/usr/bin/env bash
set -euo pipefail
set -x

# 使用 train_qwen_pointcloud.py 进行仅 LLM 的全参微调（2D 图文 QA），启用 DeepSpeed ZeRO-2。
# 默认数据：qwen-vl-finetune/demo/AD_train_2D.jsonl（由转换脚本生成）。
# 可用环境变量覆盖参数，或直接编辑本脚本。

ROOT_DIR=$(cd "$(dirname "$0")"/../.. && pwd)

# ===== 基本参数（可通过环境变量覆盖） =====
MODEL=${MODEL:-/e2e-data/embodied-research-data/large_model/huggingface/hub/Qwen3VL-8B}                             # 预训练模型或本地路径
TRAIN_FILE=${TRAIN_FILE:-"${ROOT_DIR}/qwen-vl-finetune/demo/AD_train_2D.jsonl"}
OUT=${OUT:-"${ROOT_DIR}/outputs/run_2d_llm_qwen"}                  # 输出目录

EPOCHS=${EPOCHS:-3}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-32}
LR=${LR:-3e-6}
BF16=${BF16:-0}    # 1: 启用 bf16；0: 关闭
FP16=${FP16:-1}    # 1: 启用 fp16；0: 关闭（与 bf16 二选一）

# DeepSpeed ZeRO-2 配置（默认使用仓库内配置）
DS_CONFIG=${DS_CONFIG:-"${ROOT_DIR}/qwen-vl-finetune/scripts/zero3_offload.json"}

# 分布式（单机多卡）参数
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "${OUT}"

# ===== 组装 train_qwen_pointcloud.py 参数 =====
ARGS=(
  "${ROOT_DIR}/qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py"
  --model "${MODEL}"
  --train-file "${TRAIN_FILE}"
  --output-dir "${OUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_PER_DEVICE}"
  --lr "${LR}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --train-llm                       # 仅训练 LLM（language_model + lm_head）
  --deepspeed "${DS_CONFIG}"
  --lr-scheduler cosine
  --warmup-ratio 0.03
  --save-every-steps 1000
  --save-total-limit 2
)

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi
if [[ "${FP16}" == "1" ]]; then
  ARGS+=(--fp16)
fi

echo "[train_2d_llm_zero2_pc] 模型: ${MODEL}"
echo "[train_2d_llm_zero2_pc] 训练集: ${TRAIN_FILE}"
echo "[train_2d_llm_zero2_pc] 输出目录: ${OUT}"
echo "[train_2d_llm_zero2_pc] DeepSpeed 配置: ${DS_CONFIG}"
echo "[train_2d_llm_zero2_pc] 仅训练 LLM（视觉/连接/点云均冻结）"

# ===== 启动（单机多卡） =====
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  "${ARGS[@]}"

