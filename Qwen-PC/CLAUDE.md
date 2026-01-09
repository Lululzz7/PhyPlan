# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

- Core libraries
  - pip install "transformers>=4.57.0"
  - pip install qwen-vl-utils==0.0.14
  - Optional (video): pip install 'qwen-vl-utils[decord]'
  - Optional (Flash-Attention 2): pip install -U flash-attn --no-build-isolation

- Web demo (Gradio UI)
  - pip install -r requirements_web_demo.txt
  - python web_demo_mm.py --backend vllm
    - Flags: --checkpoint-path, --cpu-only, --flash-attn2, --share, --inbrowser, --server-port, --server-name, --backend {hf|vllm}, --tensor-parallel-size, --gpu-memory-utilization
  - Example (HF backend): python web_demo_mm.py --backend hf -c Qwen/Qwen3-VL-8B-Instruct

- Training (HF Trainer + optional DeepSpeed)
  - torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) qwen-vl-finetune/qwenvl/train/train_qwen.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_use public_dataset1,public_dataset2 \
    --output_dir ./output \
    --bf16
  - Add DeepSpeed: --deepspeed qwen-vl-finetune/scripts/zero3.json
  - Enable LoRA: set --lora_enable and related r/alpha/dropout flags in TrainingArguments

- Evaluation (MMMU)
  - python evaluation/mmmu/run_mmmu.py infer --model-path /path/to/model --data-dir /data/mmmu --output-file results.jsonl

- vLLM serving (FP8 example; adjust to your hardware)
  - vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 --tensor-parallel-size 8 --mm-encoder-tp-mode data --enable-expert-parallel --async-scheduling --media-io-kwargs '{"video": {"num_frames": -1}}' --host 0.0.0.0 --port 22002

- Lint/format (Ruff)
  - pip install ruff
  - ruff check qwen-vl-utils/src
  - ruff format qwen-vl-utils/src
  - Style: line-length 119, double quotes, isort via Ruff

## Architecture overview

- Preprocessing toolkit: qwen-vl-utils
  - qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    - Image resize within pixel budgets via smart_resize; supports http/file/base64/PIL
      - smart_resize(height,width,factor,...) at qwen-vl-utils/src/qwen_vl_utils/vision_process.py:56
      - fetch_image(...) at qwen-vl-utils/src/qwen_vl_utils/vision_process.py:93
    - Video loading via backends (torchcodec/decord/torchvision), frame sampling, budgeted resizing
      - get_video_reader_backend() at qwen-vl-utils/src/qwen_vl_utils/vision_process.py:389
      - fetch_video(...) at qwen-vl-utils/src/qwen_vl_utils/vision_process.py:403
    - Conversations -> tensors and kwargs
      - process_vision_info(...) at qwen-vl-utils/src/qwen_vl_utils/vision_process.py:501
    - Env vars: MODEL_SEQ_LEN, FORCE_QWENVL_VIDEO_READER, TORCHCODEC_NUM_THREADS

- Fine-tuning pipeline (HF Trainer with custom FA2 patches)
  - Training entrypoint
    - qwen-vl-finetune/qwenvl/train/train_qwen.py
      - Model selection (Qwen2-VL/2.5-VL/3-VL/MoE) at qwen-vl-finetune/qwenvl/train/train_qwen.py:103–134
      - Processor init at qwen-vl-finetune/qwenvl/train/train_qwen.py:137–139
      - Attention monkey‑patch for packed/flattened data at qwen-vl-finetune/qwenvl/train/train_qwen.py:141–144
      - LoRA config (q/k/v/o) at qwen-vl-finetune/qwenvl/train/train_qwen.py:170–177
      - Trainable toggles (vision/merger/LLM) at qwen-vl-finetune/qwenvl/train/train_qwen.py:67–90
      - Trainer orchestration and resume/save at qwen-vl-finetune/qwenvl/train/train_qwen.py:186–203
  - Attention/optimizer patches
    - qwen-vl-finetune/qwenvl/train/trainer.py
      - Varlen FA2 wrapper at qwen-vl-finetune/qwenvl/train/trainer.py:33–109
      - Qwen2‑VL forward using MRoPE at qwen-vl-finetune/qwenvl/train/trainer.py:111–158
      - Qwen3‑VL forward using RoPE at qwen-vl-finetune/qwenvl/train/trainer.py:162–200
      - Causal mask passthrough at qwen-vl-finetune/qwenvl/train/trainer.py:203–213
      - Optimizer grouping (vision/projector lrs) at qwen-vl-finetune/qwenvl/train/trainer.py:316–491
      - Patch application at qwen-vl-finetune/qwenvl/train/trainer.py:494–511
  - Data pipeline
    - qwen-vl-finetune/qwenvl/data/data_processor.py
      - Update processor pixel budgets at qwen-vl-finetune/qwenvl/data/data_processor.py:44–137
      - Build messages with <image>/<video> placeholders at qwen-vl-finetune/qwenvl/data/data_processor.py:140–199
      - Tokenize + label extraction at qwen-vl-finetune/qwenvl/data/data_processor.py:202–241
      - LazySupervisedDataset and collators at qwen-vl-finetune/qwenvl/data/data_processor.py:244–309, 535–603, 605–676

- Evaluation
  - evaluation/mmmu/run_mmmu.py — benchmark entrypoint; see README for usage

- Web demo (local UI)
  - web_demo_mm.py — Gradio app; supports HF and vLLM backends
    - vLLM path sets VLLM_WORKER_MULTIPROC_METHOD and uses AutoProcessor
      - _load_model_processor() at web_demo_mm.py:66–105
    - Streaming HF generation via TextIteratorStreamer
      - call_local_model(..., backend='hf') at web_demo_mm.py:219–239
    - vLLM generate with SamplingParams
      - call_local_model(..., backend='vllm') at web_demo_mm.py:206–218

## Important docs and configs

- README.md — quickstart, multi‑image/video inference, processor budgets, vLLM/SGLang serving, offline inference
- qwen-vl-utils/pyproject.toml — package metadata and Ruff settings (line‑length 119, double quotes, isort via Ruff)
- DeepSpeed JSON configs — qwen-vl-finetune/scripts/zero2.json, zero3.json, zero3_offload.json (use with --deepspeed)

## Notes and gotchas

- Flash‑Attention 2
  - pip install -U flash-attn --no-build-isolation
  - Use only with bf16/fp16; enable via attn_implementation="flash_attention_2"

- Video backends
  - Default prefers torchcodec if available, then decord, else torchvision
  - Override via env: FORCE_QWENVL_VIDEO_READER={torchcodec|decord|torchvision}

- Processor pixel budgets
  - When using qwen-vl-utils preprocessing, pass do_resize=False to HF processors to avoid double‑resizing

- Environment
  - This environment may not be a git repo by default; CI configs may be absent