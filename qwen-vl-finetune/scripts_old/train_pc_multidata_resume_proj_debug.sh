#!/usr/bin/env bash
set -euo pipefail
set -x

# 统一启用可扩展段分配器，缓解优化器步碎片引发的 OOM
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 单机单卡：多数据合并，加载已有 projector 断点，仅训练 projector，并周期性/最终仅保存 projector 权重。

ROOT_DIR=$(cd "$(dirname "$0")"/../.. && pwd)
DEMO_DIR="${ROOT_DIR}/qwen-vl-finetune/demo"

# 四份训练数据（绝对路径或仓库内 demo），合并到一个 JSON
SCANQA_JSON=${SCANQA_JSON:-"${DEMO_DIR}/pointcloud_qa_scanqa_train.json"}
SCENE_CAPTION_JSON=${SCENE_CAPTION_JSON:-"${DEMO_DIR}/pointcloud_qa_scene_caption.json"}
SCENE30K_JSON=${SCENE30K_JSON:-"${DEMO_DIR}/pointcloud_qa_scene30k.json"}
SQA3D_JSON=${SQA3D_JSON:-"${DEMO_DIR}/pointcloud_qa_sqa3d.json"}
COMBINED_JSON=${COMBINED_JSON:-"${DEMO_DIR}/pointcloud_qa_combined.json"}

# 模型与输出（统一权重路径）
MODEL=${MODEL:-/e2e-data/embodied-research-data/large_model/huggingface/hub/Qwen3VL-8B}
OUT=${OUT:-"${ROOT_DIR}/outputs/pc_multidata_resume_proj_single"}
SAVE_PROJ=${SAVE_PROJ:-"${OUT}/point_proj.bin"}

# projector 断点（统一为指定 resume 路径，可通过环境变量覆盖）
PROJ_CKPT=${PROJ_CKPT:-"/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/outputs/pc_multidata_resume_proj_8gpu/point_proj.bin"}

# 点云骨干配置
POINT_BACKBONE=${POINT_BACKBONE:-PointBERT}
POINT_CFG=${POINT_CFG:-PointTransformer_8192point_2layer}
POINT_USE_COLOR=${POINT_USE_COLOR:-1}
POINT_BACKBONE_CKPT=${POINT_BACKBONE_CKPT:-/e2e-data/evad-tech-vla/luzheng/Qwen3-VL-main/checkpoints/point_bert_v1.2.pt}

# 训练超参（单卡；统一每卡 batch=1，梯度累计=16）
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}
BF16=${BF16:-1}
GRAD_CKPT=${GRAD_CKPT:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}

# 学习率调度：默认使用余弦退火 + 3% 预热，可通过环境变量覆盖
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_STEPS=${WARMUP_STEPS:-0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}

mkdir -p "${OUT}"

# 合并四份 JSON（均为数组）到一个文件
python3 - "$SCANQA_JSON" "$SCENE_CAPTION_JSON" "$SCENE30K_JSON" "$SQA3D_JSON" "$COMBINED_JSON" << 'PY'
import json, sys
in_files = sys.argv[1:5]
out_file = sys.argv[5]
merged = []
for p in in_files:
    with open(p, 'r', encoding='utf-8') as f:
        arr = json.load(f)
        if isinstance(arr, list):
            merged.extend(arr)
        else:
            raise ValueError(f"Input {p} is not a JSON array")
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f"Merged {len(merged)} records -> {out_file}")
PY

# 组装训练参数（加载 projector，仅训练 projector；周期性与最终只保存 projector）
ARGS=(
  "${ROOT_DIR}/qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py"
  --model "${MODEL}"
  --train-file "${COMBINED_JSON}"
  --output-dir "${OUT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_PER_DEVICE}"
  --lr "${LR}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --lr-scheduler "${LR_SCHEDULER}"
  --warmup-steps "${WARMUP_STEPS}"
  --warmup-ratio "${WARMUP_RATIO}"
  --save-every-steps 500
  --save-total-limit 0
  --point-backbone "${POINT_BACKBONE}"
  --point-backbone-config "${POINT_CFG}"
  --point-proj-ckpt "${PROJ_CKPT}"
  --save-point-proj-ckpt "${SAVE_PROJ}"
  --save-point-proj-interval 500
  --train-point-projector
)

if [[ "${POINT_USE_COLOR}" == "1" ]]; then
  ARGS+=(--point-use-color)
fi

if [[ -n "${POINT_BACKBONE_CKPT}" ]]; then
  ARGS+=(--point-backbone-ckpt "${POINT_BACKBONE_CKPT}")
fi

if [[ "${BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi

if [[ "${GRAD_CKPT}" == "1" ]]; then
  ARGS+=(--grad-ckpt)
fi

# 启动前打印训练计划与保存目标（便于调试）
array_has_flag() { local f="$1"; for t in "${ARGS[@]}"; do [[ "$t" == "$f" ]] && return 0; done; return 1; }
get_flag_value() { local f="$1"; for ((i=0;i<${#ARGS[@]};i++)); do if [[ "${ARGS[i]}" == "$f" ]]; then echo "${ARGS[i+1]}"; return 0; fi; done; return 1; }

TRAIN_PARTS=()
array_has_flag --train-point-projector && TRAIN_PARTS+=(point_projector)
array_has_flag --train-point-backbone && TRAIN_PARTS+=(point_backbone)
array_has_flag --train-connector && TRAIN_PARTS+=(connector)
array_has_flag --train-visual && TRAIN_PARTS+=(visual)
array_has_flag --train-llm && TRAIN_PARTS+=(llm)

SAVE_TARGETS=()
val=$(get_flag_value --save-point-proj-ckpt) && [[ -n "$val" ]] && SAVE_TARGETS+=("point_proj -> $val") || true
val=$(get_flag_value --save-point-backbone-ckpt) && [[ -n "$val" ]] && SAVE_TARGETS+=("point_backbone -> $val") || true

INIT_PROJ=$(get_flag_value --point-proj-ckpt) || true
if [[ -n "${INIT_PROJ:-}" ]]; then
  echo "[train_pc] 初始 projector ckpt: ${INIT_PROJ}"
fi

echo "[train_pc] 可训练模块: ${TRAIN_PARTS[*]:-无}"
echo "[train_pc] 将保存的权重: ${SAVE_TARGETS[*]:-无}"

python "${ARGS[@]}"
