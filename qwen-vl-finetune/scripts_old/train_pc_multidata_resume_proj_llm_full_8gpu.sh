#!/usr/bin/env bash
set -euo pipefail
set -x

# 说明：
# - 以“根目录下的整模型权重 + 根目录下的 point_proj.bin”为初始化，继续进行全参数训练
# - 仍然同时全参训练 LLM（--train-llm）与点云 projector（--train-point-projector）；backbone 默认不训练
# - 不再进行 LoRA 合并步骤；直接把根目录作为 --model 使用
# - 可选：通过 DS_CONFIG 环境变量开启 DeepSpeed（未设置则默认禁用以避免不兼容）

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROOT_DIR=$(cd "$(dirname "$0")"/../.. && pwd)
DEMO_DIR="${ROOT_DIR}/qwen-vl-finetune/demo"

# ===== 基础路径（可覆盖） =====
# 根目录（整模型分片 + 配置所在处）
MODEL_DIR=${MODEL_DIR:-"/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/outputs/pc_multidata_resume_proj_llm_full_8gpu_cont9/checkpoint-1000"}
# 继续训练的输出目录
OUT=${OUT:-"${MODEL_DIR}_cont9_e-5"}
# 训练数据（已合并）
DATA=${DATA:-"${DEMO_DIR}/pointcloud_qa_combined.json"}

# 点云 projector/backbone 初始化
POINT_PROJ_CKPT=${POINT_PROJ_CKPT:-"${MODEL_DIR}/point_proj.bin"}
POINT_BACKBONE=${POINT_BACKBONE:-PointBERT}
POINT_CFG=${POINT_CFG:-PointTransformer_8192point_2layer}
POINT_BACKBONE_CKPT=${POINT_BACKBONE_CKPT:-/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/checkpoints/point_bert_v1.2.pt}
POINT_USE_COLOR=${POINT_USE_COLOR:-1}
TRAIN_BACKBONE=${TRAIN_BACKBONE:-0}   # 1: 同时训练 backbone；0: 仅 projector

# ===== 训练超参（与之前一致，按需覆盖） =====
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-32}
EPOCHS=${EPOCHS:-9}
LR=${LR:-1e-5}
BF16=${BF16:-1}
GRAD_CKPT=${GRAD_CKPT:-0}     # 默认为 0，视环境需要可启用
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_STEPS=${WARMUP_STEPS:-0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}

# 保存策略
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}
SAVE_PROJ=${SAVE_PROJ:-"${OUT}/point_proj.bin"}
SAVE_PROJ_INTERVAL=${SAVE_PROJ_INTERVAL:-500}

# DeepSpeed（可选）：设置 DS_CONFIG=.../zero2.json 或 .../zero3_offload.json 生效
DS_CONFIG=${DS_CONFIG:-${ROOT_DIR}/qwen-vl-finetune/scripts/zero2.json}

mkdir -p "${OUT}"

# ===== 组装训练参数：LLM 全参 + projector 全参 =====
ARGS=(
  "${ROOT_DIR}/qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py"
  --model "${MODEL_DIR}"
  --train-file "${DATA}"
  --output-dir "${OUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_PER_DEVICE}"
  --lr "${LR}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-steps "${WARMUP_STEPS}"
  --warmup-ratio "${WARMUP_RATIO}"
  --save-every-steps "${SAVE_EVERY_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --point-backbone "${POINT_BACKBONE}"
  --point-backbone-config "${POINT_CFG}"
  --point-proj-ckpt "${POINT_PROJ_CKPT}"
  --save-point-proj-ckpt "${SAVE_PROJ}"
  --save-point-proj-interval "${SAVE_PROJ_INTERVAL}"
  --train-point-projector
  --train-llm
)

if [[ "${POINT_USE_COLOR}" == "1" ]]; then
  ARGS+=(--point-use-color)
fi

if [[ "${TRAIN_BACKBONE}" == "1" ]]; then
  ARGS+=(--train-point-backbone --point-backbone-ckpt "${POINT_BACKBONE_CKPT}")
fi

if [[ -n "${DS_CONFIG}" ]]; then
  echo "[cont_from_root] Enable DeepSpeed: ${DS_CONFIG}"
  ARGS+=(--deepspeed "${DS_CONFIG}")
else
  echo "[cont_from_root] DeepSpeed disabled (set DS_CONFIG to enable)"
fi

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi

if [[ "${GRAD_CKPT}" == "1" ]]; then
  ARGS+=(--grad-ckpt)
fi

# 打印训练计划
array_has_flag() { local f="$1"; for t in "${ARGS[@]}"; do [[ "$t" == "$f" ]] && return 0; done; return 1; }
TRAIN_PARTS=()
array_has_flag --train-point-projector && TRAIN_PARTS+=(point_projector)
array_has_flag --train-point-backbone && TRAIN_PARTS+=(point_backbone)
array_has_flag --train-llm && TRAIN_PARTS+=(llm[full])
echo "[cont_from_root] 可训练模块: ${TRAIN_PARTS[*]:-无}"
echo "[cont_from_root] 初始化整模型(根目录): ${MODEL_DIR}"
echo "[cont_from_root] 初始化 projector: ${POINT_PROJ_CKPT}"
echo "[cont_from_root] 输出目录: ${OUT}"

# 启动（单机八卡）
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  "${ARGS[@]}"

