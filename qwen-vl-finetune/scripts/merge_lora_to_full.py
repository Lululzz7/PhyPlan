#!/usr/bin/env python3
"""
将 LoRA 适配器合并到基础模型，导出“整模型”目录，便于后续全量微调或直接部署。

用法示例：
  python qwen-vl-finetune/scripts/merge_lora_to_full.py \
    --base-model Qwen/Qwen3-VL-7B-Instruct \
    --adapter-dir /path/to/old_run_or_checkpoint \
    --out-dir /path/to/merged

可选参数：
  --attn-implementation {flash_attention_2,sdpa,eager}  # 默认 flash_attention_2
  --dtype {auto,bf16,fp16,fp32}                          # 默认 auto
  --no-processor                                         # 不保存处理器/分词器配置
  --trust-remote-code                                    # 如需信任远程代码

输出目录将至少包含：
  - model.safetensors 或 pytorch_model.bin
  - 处理器/分词器配置（除非使用 --no-processor）
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_dtype(s: str):
    s = (s or "auto").lower()
    # HF 接口期望 torch.dtype 或 None；"auto" 用 None 表示让框架自行选择
    if s == "auto":
        return None
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def has_model_weights(dir_path: Path) -> bool:
    # 同时支持单文件与分片命名
    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model*.bin",
    ]
    for pat in patterns:
        if list(dir_path.glob(pat)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into a full model directory")
    ap.add_argument("--base-model", type=str, required=True, help="基础模型名称或路径，例如 Qwen/Qwen3-VL-7B-Instruct")
    ap.add_argument("--adapter-dir", type=str, required=True, help="LoRA 适配器目录（可为 run 根目录或 checkpoint-* 目录）")
    ap.add_argument("--out-dir", type=str, required=True, help="合并后整模型的输出目录")
    ap.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"], help="注意力实现后端")
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"], help="权重加载精度")
    ap.add_argument("--no-processor", action="store_true", help="不保存处理器/分词器配置")
    ap.add_argument("--trust-remote-code", action="store_true", help="信任远程代码（如模型需）")
    args = ap.parse_args()

    base_id = args.base_model
    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_model_weights(out_dir):
        print(f"[Info] 已存在整模型权重，跳过合并：{out_dir}")
        return

    try:
        from peft import PeftModel
    except Exception as e:
        print("[Error] 需要安装 peft 才能合并 LoRA：", e)
        sys.exit(1)

    dtype = parse_dtype(args.dtype)

    # 加载基础模型与适配器
    print(f"[Info] 加载基础模型: {base_id}")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_id,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    print(f"[Info] 加载 LoRA 适配器: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    # 合并并导出
    print("[Info] 合并 LoRA 到整模型...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    if not args.no_processor:
        try:
            proc = AutoProcessor.from_pretrained(base_id, trust_remote_code=bool(args.trust_remote_code))
            proc.save_pretrained(str(out_dir))
        except Exception as e:
            print(f"[Warn] 保存处理器失败（可忽略）：{e}")

    # 校验输出（分片或单文件）
    if not has_model_weights(out_dir):
        raise RuntimeError(f"未发现整模型权重文件（可能是写入权限或磁盘空间问题），请检查输出目录：{out_dir}")
    print(f"[Success] 合并完成，整模型已保存：{out_dir}")


if __name__ == "__main__":
    main()
