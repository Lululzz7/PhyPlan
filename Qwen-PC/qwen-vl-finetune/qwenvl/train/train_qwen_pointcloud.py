 

import os
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys as _sys  # 提前导入 sys

project_root = Path(__file__).resolve().parent.parent.parent
repo_root = Path(__file__).resolve().parents[3]
if str(project_root) not in _sys.path:
    _sys.path.insert(0, str(project_root))

import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    Qwen3VLForConditionalGeneration,
    TrainerCallback,
)
import types as _types

from qwenvl.train.argument import TrainingArguments 
from qwenvl.train.modeling_qwen3_vl_pointcloud import Qwen3VLModel_PointCloud
from qwenvl.train.model.point_adapter import create_pointcloud_adapter
from qwenvl.data.data_processor import preprocess_qwen_visual, _build_messages
from qwenvl.data.rope2d import get_rope_index_3

try:
    from qwen_vl_utils.vision_process import process_vision_info
except Exception:
    import sys as _sys2
    _sys2.path.append(str(repo_root / 'qwen-vl-utils' / 'src'))
    from qwen_vl_utils.vision_process import process_vision_info

class PointCloudVisualDataset(Dataset):
    """支持点云+图像/视频/文本的联合数据集。

    每条样本 JSON 示例：
    {
      "data_path": "/abs/base/dir",           # 可选：图/视频相对路径的基准目录
      "image": ["images/xxx.jpg"],             # 可选
      "video": [],                               # 可选
      "point_cloud": "points/xxx.npy",        # 可选；npy/npz，形如 (N,3) 或 (N,6)
      "conversations": [
        {"from": "human", "value": "<image> 请描述场景"},
        {"from": "assistant", "value": "..."}
      ]
    }
    """

    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        processor: AutoProcessor,
        pc_adapter_fn,
        pc_token_len: int,
        hidden_size: int,
        use_color: bool = False,
        point_num: Optional[int] = 0,
        embeds_out_device: Optional[torch.device] = None,
    ):
        self.anns = annotations
        self.processor = processor
        self.pc_adapter_fn = pc_adapter_fn
        self.pc_token_len = pc_token_len
        self.hidden_size = hidden_size
        self.use_color = use_color
        self.point_num = point_num
        self.embeds_out_device = embeds_out_device or torch.device('cpu')

    def __len__(self):
        return len(self.anns)

    def _load_points(self, path_or_none: Optional[str]) -> Optional[torch.Tensor]:
            if not path_or_none:
                return None
            p = Path(path_or_none)
            if not p.exists():
                raise FileNotFoundError(f"Point cloud file not found: {p}")

            if p.suffix.lower() == '.ply':
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(p))
                xyz = np.asarray(pcd.points, dtype=np.float32)
                if pcd.has_colors():
                    rgb = np.asarray(pcd.colors, dtype=np.float32)
                    arr = np.concatenate([xyz, rgb], axis=1)
                else:
                    arr = xyz
            else:
                arr = np.load(str(p), allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    key = list(arr.keys())[0]
                    arr = arr[key]
            arr = arr.astype(np.float32)
            if not self.use_color and arr.shape[-1] > 3:
                arr = arr[:, :3]
            xyz = arr[:, :3]
            centroid = np.mean(xyz, axis=0)
            xyz = xyz - centroid
            m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
            if m > 0:
                xyz = xyz / m
            if arr.shape[-1] > 3:
                rgb = arr[:, 3:]
                if rgb.max() > 1.0:
                    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
                arr = np.concatenate([xyz, rgb], axis=1)
            else:
                arr = xyz
            if isinstance(self.point_num, int) and self.point_num > 0 and arr.shape[0] > self.point_num:
                n = arr.shape[0]
                xyz_np = arr[:, :3]
                centroids = np.zeros((self.point_num,), dtype=np.int64)
                distance = np.ones((n,), dtype=np.float64) * 1e10
                farthest = np.random.randint(0, n)
                for i in range(self.point_num):
                    centroids[i] = farthest
                    centroid = xyz_np[farthest]
                    dist = np.sum((xyz_np - centroid) ** 2, axis=1)
                    mask = dist < distance
                    distance[mask] = dist[mask]
                    farthest = int(np.argmax(distance))
                arr = arr[centroids]
            t = torch.from_numpy(arr).float()
            return t
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.anns[idx]
        sources = sample if isinstance(sample, list) else [sample]

        # 1) 文本预处理（生成 input_ids/labels）
        data_dict = preprocess_qwen_visual(sources, self.processor)

        try:
            base_path = Path(sample.get("data_path", "")) if isinstance(sample, dict) else Path("")
            messages = _build_messages(sample if isinstance(sample, dict) else sample[0], base_path)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=getattr(self.processor.image_processor, 'patch_size', 14)
            )
            if image_inputs is not None:
                img_pack = self.processor(images=image_inputs, return_tensors="pt", do_resize=False)
                data_dict.update({
                    "pixel_values": img_pack.get("pixel_values"),
                    "image_grid_thw": img_pack.get("image_grid_thw"),
                })
            if video_inputs is not None and hasattr(self.processor, 'video_processor') and self.processor.video_processor is not None:
                vid_pack = self.processor(videos=video_inputs, return_tensors="pt", do_resize=False, **(video_kwargs or {}))
                data_dict.update({
                    "pixel_values_videos": vid_pack.get("pixel_values_videos"),
                    "video_grid_thw": vid_pack.get("video_grid_thw"),
                })
        except Exception:
            pass

        try:
            merge_size = getattr(self.processor.image_processor, "merge_size", 2)
            img_thw = data_dict.get("image_grid_thw", None)
            vid_thw = data_dict.get("video_grid_thw", None)
            second_per_grid_ts = None
            if vid_thw is not None and hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
                vps = getattr(self.processor.video_processor, "temporal_patch_size", 1)
                fps = getattr(self.processor.video_processor, "fps", 1.0) or 1.0
                num_videos = vid_thw.shape[0]
                spgt = float(vps) / float(fps)
                second_per_grid_ts = torch.tensor([spgt] * int(num_videos), dtype=torch.float)

            position_ids, _ = get_rope_index_3(
                merge_size,
                data_dict["input_ids"],
                image_grid_thw=img_thw,
                video_grid_thw=vid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            data_dict["position_ids"] = position_ids
        except Exception:
            pass

        pc_path = sample.get("point_cloud", None) if isinstance(sample, dict) else None
        pc_tensor = self._load_points(pc_path)
        if getattr(self, "use_pc_adapter_forward", False):
            data_dict["point_clouds"] = pc_tensor if pc_tensor is not None else None
        else:
            if pc_tensor is not None:
                pc_embeds = self.pc_adapter_fn(pc_tensor.unsqueeze(0), out_device=self.embeds_out_device)
                pc_embeds = pc_embeds.squeeze(0)
            else:
                pc_embeds = torch.zeros(self.pc_token_len, self.hidden_size)
            data_dict["point_cloud_embeds"] = pc_embeds

        return data_dict


class PointCloudCollator:
    """数据整理：对齐 input_ids/labels，并拼接视觉张量与点云嵌入。"""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # 是否在前向中计算点云嵌入（训练点云分支时为 True）
        self.use_pc_adapter_forward: bool = False
        # 点云前缀 token 数（用于在线嵌入模式下为标签前缀填充 -100）
        self.pc_token_len: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}

        # input_ids（按 tokenizer pad）
        input_ids = [f["input_ids"][0] for f in features]
        batch_tokens = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
        batch.update({k: v for k, v in batch_tokens.items()})
        max_len = batch["input_ids"].size(1)
        # labels：仅按文本最大长度 pad/truncate 到 max_len；
        # 点云前缀的 -100 由模型前向在注入点云时统一补充，避免重复处理与长度不一致。
        labels_list = [f["labels"][0] for f in features]
        padded_labels = []
        for lab in labels_list:
            if lab.size(0) < max_len:
                pad = torch.full((max_len - lab.size(0),), -100, dtype=lab.dtype)
                lab = torch.cat([lab, pad], dim=0)
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.stack(padded_labels, dim=0)

        # 为上层模型提供 `logits_to_keep`，仅保留与文本长度匹配的 logits（自动忽略点云前缀部分），
        # 以确保与未补前缀的 labels 对齐、计算损失与日志均严格一致。
        batch["logits_to_keep"] = int(max_len)

        # position_ids：按最长序列对齐到形状 [3, B, max_len]；若不存在则由模型内部计算
        if all("position_ids" in f for f in features):
            pos_list = [f["position_ids"] for f in features]  # each: [3, 1, Li]
            pos_padded = []
            for pos in pos_list:
                # 截断或右侧用常数1填充（与官方 DataCollator pad_and_cat 一致）
                if pos.size(-1) > max_len:
                    pos = pos[..., : max_len]
                elif pos.size(-1) < max_len:
                    pad_len = max_len - pos.size(-1)
                    pos = torch.nn.functional.pad(pos, (0, pad_len), value=1)
                pos_padded.append(pos)
            batch["position_ids"] = torch.cat(pos_padded, dim=1)  # [3, B, L]

        # 点云：根据开关选择传入嵌入或原始点云列表
        if getattr(self, "use_pc_adapter_forward", False):
            batch["point_clouds"] = [f.get("point_clouds", None) for f in features]
        else:
            pc_list = [f["point_cloud_embeds"] for f in features]
            batch["point_cloud_embeds"] = torch.stack(pc_list, dim=0)

        # 可选视觉：pixel_values / image_grid_thw / pixel_values_videos / video_grid_thw
        if any("pixel_values" in f for f in features):
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features if "pixel_values" in f], dim=0)
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features if "image_grid_thw" in f], dim=0)

        if any("pixel_values_videos" in f for f in features):
            batch["pixel_values_videos"] = torch.cat([f["pixel_values_videos"] for f in features if "pixel_values_videos" in f], dim=0)
            batch["video_grid_thw"] = torch.cat([f["video_grid_thw"] for f in features if "video_grid_thw" in f], dim=0)

        return batch

def read_annotations(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with Point Clouds")
    parser.add_argument("--model", type=str, required=True, help="预训练模型名称或路径，如 Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-file", type=str, required=True, help="训练标注文件（json/jsonl）")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    # 学习率调度与预热
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=[
                            "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
                        ], help="学习率调度器类型，默认 cosine")
    parser.add_argument("--warmup-steps", type=int, default=0, help="学习率预热步数（>0 优先生效）")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="学习率预热比例（当 warmup-steps=0 时生效）")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    # 梯度累计步数（用于降低单步显存峰值，同时保持有效批量）
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累计步数（>=1）")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪阈值，默认 1.0；设置<=0 以禁用裁剪")
    parser.add_argument("--grad-ckpt", action="store_true")
    # Checkpoint保存频率与策略
    parser.add_argument("--save-every-steps", type=int, default=None,
                        help="每隔多少个step保存一次checkpoint；不传则使用默认1000。")
    parser.add_argument("--force-save-checkpoints", action="store_true",
                        help="即使未训练Qwen3-VL基础模块，也按save-every-steps保存checkpoint。")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="最多保留的checkpoint数量（循环覆盖）；默认2。设置<=0则不限制（谨慎使用）。")
    # 统一训练开关：若未显式传入 train-*- 开关，则该模块不训练（冻结）
    parser.add_argument("--train-visual", action="store_true", help="训练视觉编码器（visual 模块）")
    parser.add_argument("--train-connector", action="store_true", help="训练视觉与语言的连接模块（visual.merger）")
    parser.add_argument("--train-llm", action="store_true", help="训练语言模型（language_model）与输出头（lm_head）")
    # 统一训练路径：移除扁平/打包，采用逐样本整理
    # 点云适配器参数
    parser.add_argument("--point-backbone", type=str, default="PointBERT")
    parser.add_argument("--point-backbone-config", type=str, default="PointTransformer_8192point_2layer")
    parser.add_argument("--point-use-color", action="store_true")
    parser.add_argument("--point-num", type=int, default=0, help="采样点数量（>0 启用，<=0 不采样）")
    parser.add_argument("--pc-embeds-device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="离线点云嵌入输出设备：cuda 可减少主机→设备搬运（显存允许时建议）")
    parser.add_argument("--point-backbone-ckpt", type=str, default=None,
                        help="PointBERT/PointTransformer骨干权重文件，如 point_bert_v1.2.pt")
    parser.add_argument("--point-proj-ckpt", type=str, default=None,
                        help="projector 权重文件，可为 HF 分片或提取后的 .pt")
    parser.add_argument("--save-point-proj-ckpt", type=str, default=None,
                        help="训练结束后单独保存 point_proj.* 权重的路径（仅当启用 --train-point-projector 且存在 pc_adapter_model 时生效）")
    parser.add_argument("--save-point-proj-interval", type=int, default=None,
                        help="周期性保存 projector 权重的步数间隔，如 500。仅 rank0 写入，命名为 point_proj_step-<global_step>.bin")
    parser.add_argument("--save-point-backbone-ckpt", type=str, default=None,
                        help="训练结束后单独保存 point_backbone 权重（PointBERT），保存为与官方兼容的 {state_dict: {module.point_encoder.*}} 结构，仅当启用 --train-point-backbone 时生效")
    # 点云训练开关（细粒度）：可分别控制 PointBERT 与 projector（含 align_mlp）
    # 若均未开启，则使用离线嵌入并冻结
    parser.add_argument("--train-point-backbone", action="store_true", help="训练点云骨干（PointBERT）")
    parser.add_argument("--train-point-projector", action="store_true", help="训练点云投影器（point_proj + 对齐层）")

    # LLM 低秩适配（LoRA），显著降低显存占用；仅当启用 --train-llm 时可用
    parser.add_argument("--lora-enable", action="store_true", help="启用 LoRA 训练 LLM（仅训练低秩适配层）")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA r（秩）")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-adapter-path", type=str, default=None,
                        help="继续以 LoRA 微调时传入已训练的 adapter 路径，自动加载该 LoRA 适配器进行续训")
    parser.add_argument("--merge-lora-on-save", action="store_true",
                        help="在 LoRA 场景下训练结束时自动合并 LoRA 并导出完整模型到 output_dir/merged，同时保留 adapter 文件")
    # DeepSpeed 支持：传入 DeepSpeed 配置（如 qwen-vl-finetune/scripts/zero3_offload.json）以启用 ZeRO
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径（例如 qwen-vl-finetune/scripts/zero3_offload.json），不传则禁用 DeepSpeed")
    # 断点续训（严格恢复 optimizer/scheduler/RNG/global_step 等状态）
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="从给定 checkpoint 目录严格续训（例如 outputs/run/checkpoint-3500）；不传则从头本轮训练")

    args = parser.parse_args()

    # 注意：在分布式/多卡训练中，不要手动将模型 .to(cuda:0)。
    # 交由 HF Trainer/Accelerate 管理设备放置，避免所有进程都占用 GPU:0。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型与处理器
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation="flash_attention_2",
        dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
    )

    processor = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, fix_mistral_regex=True)
    # 对齐 special tokens，避免提示不一致
    try:
        tok = processor.tokenizer if hasattr(processor, 'tokenizer') else tokenizer
        for k in ("pad_token_id", "eos_token_id", "bos_token_id"):
            v = getattr(tok, k, None)
            if v is not None:
                setattr(qwen.config, k, v)
                if hasattr(qwen, 'generation_config') and getattr(qwen.generation_config, k, None) is not None:
                    setattr(qwen.generation_config, k, v)
    except Exception:
        pass

    # 接入 FA2/varlen 注意力与优化器分组补丁（与 train_qwen.py 一致）
    try:
        from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # monkey-patch attention/mask
        import qwenvl.train.trainer  # register custom Trainer.create_optimizer
        replace_qwen2_vl_attention_class()
    except Exception:
        pass

    # 2) 替换底层 model 为 PointCloud 版本（权重兼容）
    pc_model = Qwen3VLModel_PointCloud(qwen.model.config)
    pc_model.load_state_dict(qwen.model.state_dict(), strict=True)
    qwen.model = pc_model

    # 训练模块开关（统一：未显式 train-* 的模块全部冻结）。注意：若启用 LoRA，则不解冻 LLM 的基础权重，只训练 LoRA 层。
    for _, p in qwen.model.visual.named_parameters():
        p.requires_grad = False
    for _, p in qwen.model.visual.merger.named_parameters():
        p.requires_grad = False
    for _, p in qwen.model.language_model.named_parameters():
        p.requires_grad = False
    # Freeze lm_head by parameters to ensure it's not trained unless --train-llm
    for p in qwen.lm_head.parameters():
        p.requires_grad = False

    if args.train_visual:
        for _, p in qwen.model.visual.named_parameters():
            p.requires_grad = True
    if args.train_connector:
        for _, p in qwen.model.visual.merger.named_parameters():
            p.requires_grad = True
    # LLM：LoRA 或全量微调（二选一）
    if args.train_llm:
        if args.lora_enable:
            try:
                from peft import LoraConfig, get_peft_model, TaskType, PeftModel  # type: ignore
                print("[train_pc] 启用 LoRA 训练 LLM（仅训练低秩适配层；支持从头或显式加载续训）")
                # 冻结所有基础权重，仅训练 LoRA 注入的层
                for p in qwen.model.language_model.parameters():
                    p.requires_grad = False
                # 显式加载已训练适配器续训；若未提供路径，则从头注入新的 LoRA 适配器
                lora_cfg = None
                if args.lora_adapter_path:
                    qwen = PeftModel.from_pretrained(qwen, args.lora_adapter_path)
                else:
                    lora_cfg = LoraConfig(
                        r=int(args.lora_r or 64),
                        lora_alpha=int(args.lora_alpha or 128),
                        lora_dropout=float(args.lora_dropout or 0.05),
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ],
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                    )
                    qwen = get_peft_model(qwen, lora_cfg)
                # 诊断打印：LoRA 注入详情（目标模块与注入位置示例）
                try:
                    is_rank0 = True
                    try:
                        import torch.distributed as dist
                        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
                    except Exception:
                        is_rank0 = True
                    if is_rank0:
                        # 打印 LoRA 目标模块：优先用新建适配器的配置，否则从现有适配器结构推断
                        if lora_cfg is not None:
                            tgts = list(getattr(lora_cfg, 'target_modules', []))
                            print(f"[train_pc] LoRA 目标模块: {tgts}")
                        else:
                            inferred_tgts = set()
                            for name, _m in qwen.named_modules():
                                if ('.lora_A.' in name) or ('.lora_B.' in name):
                                    tail = name.split('.lora_')[0].split('.')[-1]
                                    inferred_tgts.add(tail)
                            tgts = sorted(list(inferred_tgts)) if inferred_tgts else ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                            print(f"[train_pc] LoRA 目标模块(推断): {tgts}")
                        print("[train_pc] LoRA 对应部分: 自注意力(q/k/v/o投影)、MLP(gate/up/down投影)")
                        found = {k: set() for k in tgts}
                        samples = {k: [] for k in tgts}
                        for name, _m in qwen.named_modules():
                            if ('.lora_A.' in name) or ('.lora_B.' in name):
                                parent = name.split('.lora_')[0]
                                tail = parent.split('.')[-1]
                                if tail in found:
                                    found[tail].add(parent)
                                    if len(samples[tail]) < 3:
                                        samples[tail].append(parent)
                        try:
                            num_layers = int(getattr(getattr(qwen, 'config', None), 'text_config', getattr(qwen, 'config', None)).num_hidden_layers)
                        except Exception:
                            num_layers = None
                        for k in tgts:
                            cnt = len(found.get(k, []))
                            smp = samples.get(k, [])
                            if num_layers is not None:
                                print(f"[train_pc] LoRA注入 {k}: {cnt} 层（文本层数≈{num_layers}），示例: {smp}")
                            else:
                                print(f"[train_pc] LoRA注入 {k}: {cnt} 处，示例: {smp}")
                except Exception:
                    pass
                # LoRA 模式下：不训练 lm_head 的基础权重，仅训练注入的 LoRA 低秩层
                # 上文已在初始化阶段将 lm_head 参数设为 requires_grad=False，这里保持冻结即可。
                try:
                    base = getattr(qwen, "base_model", None) or getattr(qwen, "model", None)
                    if base is not None and hasattr(base, "lm_head"):
                        for p in base.lm_head.parameters():
                            p.requires_grad = False
                    else:
                        for p in qwen.lm_head.parameters():
                            p.requires_grad = False
                except Exception:
                    pass
                # 记录可训练参数数量，便于诊断
                try:
                    trainable = sum(p.requires_grad for p in qwen.parameters())
                    total = sum(1 for _ in qwen.parameters())
                    print(f"[train_pc] Trainable params: {trainable}/{total} after LoRA")
                except Exception:
                    pass
            except Exception as _e:
                print(f"[警告] LoRA 注入失败：{_e}；回退为全量微调。")
                for _, p in qwen.model.language_model.named_parameters():
                    p.requires_grad = True
                for p in qwen.lm_head.parameters():
                    p.requires_grad = True
        else:
            for _, p in qwen.model.language_model.named_parameters():
                p.requires_grad = True
            for p in qwen.lm_head.parameters():
                p.requires_grad = True


    # 3) 构建点云适配器（输出维度与 Qwen3-VL 文本隐藏维一致）
    hidden_size = qwen.config.text_config.hidden_size
    # 适配器：默认离线提取；如开启训练点云 encoder，则并入前向、参与训练
    train_point_any = bool(args.train_point_backbone or args.train_point_projector)
    if train_point_any:
        pc_adapter_fn, pc_meta, pc_adapter_model = create_pointcloud_adapter(
            hidden_size=hidden_size,
            point_backbone=args.point_backbone,
            point_backbone_config_name=args.point_backbone_config,
            use_color=args.point_use_color,
            point_backbone_ckpt=args.point_backbone_ckpt,
            point_proj_ckpt=args.point_proj_ckpt,
            device=torch.device('cpu'),  # 先放 CPU，交由 Trainer/Accelerate 在各进程迁移
            eval_mode=False,
            mm_use_point_start_end=False,
            return_model=True,
        )
        # 将适配器模块注入模型以支持在 forward 中端到端计算嵌入
        qwen.model.pc_adapter_model = pc_adapter_model
        # 细粒度训练标志传入模型前向
        qwen.model.pc_train_backbone = bool(args.train_point_backbone)
        qwen.model.pc_train_projector = bool(args.train_point_projector)
        # requires_grad 与 train/eval 模式按需设置
        for p in pc_adapter_model.point_backbone.parameters():
            p.requires_grad = bool(args.train_point_backbone)
        for p in pc_adapter_model.point_proj.parameters():
            p.requires_grad = bool(args.train_point_projector)
        if hasattr(pc_adapter_model, "align_mlp"):
            for p in pc_adapter_model.align_mlp.parameters():
                p.requires_grad = bool(args.train_point_projector)
        # module 级别 train()/eval()
        pc_adapter_model.point_backbone.train() if args.train_point_backbone else pc_adapter_model.point_backbone.eval()
        pc_adapter_model.point_proj.train() if args.train_point_projector else pc_adapter_model.point_proj.eval()
        if hasattr(pc_adapter_model, "align_mlp"):
            pc_adapter_model.align_mlp.train() if args.train_point_projector else pc_adapter_model.align_mlp.eval()
    else:
        pc_adapter_fn, pc_meta, pc_adapter_model = create_pointcloud_adapter(
            hidden_size=hidden_size,
            point_backbone=args.point_backbone,
            point_backbone_config_name=args.point_backbone_config,
            use_color=args.point_use_color,
            point_backbone_ckpt=args.point_backbone_ckpt,
            point_proj_ckpt=args.point_proj_ckpt,
            device=torch.device('cpu'),  # 先放 CPU，避免主进程占用 GPU:0
            eval_mode=True,
            mm_use_point_start_end=False,
            return_model=True,
        )
        # 注入用于一致性检查（即便不训练点云模块，也能检查权重加载完整性）
        qwen.model.pc_adapter_model = pc_adapter_model
        qwen.model.pc_train_backbone = False
        qwen.model.pc_train_projector = False

    # ========== 训练开始前的概要信息打印与权重完整性检查（中文日志） ==========
    try:
        train_parts = []
        if bool(getattr(qwen.model, 'pc_train_backbone', False)):
            train_parts.append('point_backbone')
        if bool(getattr(qwen.model, 'pc_train_projector', False)):
            train_parts.append('point_projector(+align_mlp)')
        if args.train_visual:
            train_parts.append('qwen.visual')
        if args.train_connector:
            train_parts.append('qwen.connector')
        if args.train_llm:
            train_parts.append('qwen.llm+lm_head')

        if not train_parts:
            train_parts_desc = '无（全部冻结；点云模块以评估模式使用）'
        else:
            train_parts_desc = ', '.join(train_parts)

        save_targets = []
        if args.save_point_proj_ckpt and bool(getattr(qwen.model, 'pc_train_projector', False)):
            save_targets.append(f"point_proj(+align_mlp) -> {args.save_point_proj_ckpt}")
        if args.save_point_backbone_ckpt and bool(getattr(qwen.model, 'pc_train_backbone', False)):
            save_targets.append(f"point_backbone -> {args.save_point_backbone_ckpt}")
        # 基础 Qwen3-VL 组件只在其参与训练或强制保存时保存
        train_any_qwen_component = bool(args.train_visual or args.train_connector or args.train_llm)
        hf_ckpt_policy = (
            '启用（按步保存）' if (train_any_qwen_component or args.force_save_checkpoints) else '禁用'
        )

        print("==== 训练概要 ====")
        print(f"可训练模块: {train_parts_desc}")
        print(f"HF 整模型 checkpoint: {hf_ckpt_policy}")
        if save_targets:
            print("将额外保存:")
            for t in save_targets:
                print(f" - {t}")
        else:
            print("未配置额外的 projector/backbone 保存路径。")

        # 进一步打印：各可训练部分的参数规模统计（元素数量），帮助确认实际参与训练的组件
        try:
            # LoRA 低秩层参数元素总数（通过名称包含 .lora_A./.lora_B. 识别）
            lora_elems = 0
            lora_tensors = 0
            for n, p in qwen.named_parameters():
                if p.requires_grad and ('.lora_A.' in n or '.lora_B.' in n):
                    lora_elems += int(p.numel())
                    lora_tensors += 1
            # lm_head 参数元素数
            lm_head_elems = 0
            try:
                base = getattr(qwen, 'base_model', None) or getattr(qwen, 'model', None)
                lm = getattr(base, 'lm_head', None) or getattr(qwen, 'lm_head', None)
                if lm is not None:
                    for p in lm.parameters():
                        if p.requires_grad:
                            lm_head_elems += int(p.numel())
            except Exception:
                pass
            # projector/backbone 参数元素数
            proj_elems = back_elems = 0
            pc_m = getattr(qwen.model, 'pc_adapter_model', None)
            if pc_m is not None:
                try:
                    for p in pc_m.point_proj.parameters():
                        if p.requires_grad:
                            proj_elems += int(p.numel())
                except Exception:
                    pass
                try:
                    for p in pc_m.point_backbone.parameters():
                        if p.requires_grad:
                            back_elems += int(p.numel())
                except Exception:
                    pass
            print(f"[train_pc] 参数规模(元素): LoRA={lora_elems}({lora_tensors}张量) lm_head={lm_head_elems} projector={proj_elems} backbone={back_elems}")
        except Exception:
            pass

        # 若提供了 projector 初始化权重，则进行完整性检查（加载是否覆盖到所有参数、是否存在新建层）
        if args.point_proj_ckpt and hasattr(qwen.model, 'pc_adapter_model') and qwen.model.pc_adapter_model is not None:
            pc_m = qwen.model.pc_adapter_model
            # projector 覆盖情况
            exp_proj = set(pc_m.point_proj.state_dict().keys())
            loaded_proj = set()
            try:
                rpt = getattr(pc_m, 'point_proj_load_report', None)
                if isinstance(rpt, dict) and 'loaded_keys' in rpt:
                    loaded_proj = set(rpt['loaded_keys'])
            except Exception:
                loaded_proj = set()
            # 若上游未暴露报告，尽力从 ckpt 粗略解析已覆盖的键
            if not loaded_proj:
                try:
                    sd = torch.load(args.point_proj_ckpt, map_location='cpu')
                    if isinstance(sd, dict) and 'state_dict' in sd:
                        sd = sd['state_dict']
                    if isinstance(sd, dict):
                        for k in sd.keys():
                            if k.startswith('point_proj.'):
                                rel = k.split('point_proj.', 1)[1]
                                loaded_proj.add(rel)
                            elif k in exp_proj:
                                loaded_proj.add(k)
                except Exception:
                    pass

            missing_proj = sorted(list(exp_proj - loaded_proj))
            if missing_proj:
                print(f"[检查] 已加载 projector ckpt，但缺失 {len(missing_proj)} 个参数键：")
                for mk in missing_proj:
                    print(f" - {mk}")
                print("[错误] 使用 projector 时必须完整加载其权重。请检查 ckpt 与当前 projector 结构是否匹配。")
                raise SystemExit(1)
            else:
                print("[检查] projector 权重已全部从 ckpt 加载。")

            # 对齐层（align_mlp）初始化情况
            if hasattr(pc_m, 'align_mlp') and isinstance(pc_m.align_mlp, torch.nn.Linear):
                align_loaded = bool(getattr(pc_m, '_align_mlp_loaded_from_ckpt', False))
                if align_loaded:
                    print("[检查] align_mlp 已从 ckpt 加载。")
                else:
                    print("[警告] ckpt 中未包含 align_mlp；该层已按默认方式初始化。")
            else:
                print("[检查] 无需 align_mlp（恒等映射）。")
        elif args.point_proj_ckpt:
            print("[检查] 提供了 projector ckpt，但未能找到适配器模型进行校验。")
    except Exception as _e:
        try:
            print(f"[警告] 打印训练概要/校验失败: {_e}")
        except Exception:
            pass


    # 4) 读取标注并构建数据集/整理器
    anns = read_annotations(args.train_file)
    # 选择离线嵌入输出设备
    embeds_out_device = torch.device(
        "cuda" if (args.pc_embeds_device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    dataset = PointCloudVisualDataset(
        annotations=anns,
        processor=processor,
        pc_adapter_fn=pc_adapter_fn,
        pc_token_len=pc_meta["point_token_len"],
        hidden_size=hidden_size,
        use_color=args.point_use_color,
        point_num=args.point_num,
        embeds_out_device=embeds_out_device,
    )
    # 标记前向是否集成适配器：影响数据整理行为
    dataset.use_pc_adapter_forward = bool(train_point_any)
    # 选择整理器：统一使用普通整理器
    collator = PointCloudCollator(tokenizer)
    collator.use_pc_adapter_forward = bool(train_point_any)
    try:
        collator.pc_token_len = int(pc_meta.get("point_token_len", 0))
    except Exception:
        collator.pc_token_len = 0

    # 5) 训练参数与 Trainer
    # 仅当 Qwen3-VL 的三个组件（visual / connector / llm）任一被训练时，才保存 Qwen3-VL 权重；
    # 否则禁用模型权重的保存与周期性 checkpoint（避免产生大量 safetensors 分片）。
    train_any_qwen_component = bool(args.train_visual or args.train_connector or args.train_llm)

    # DDP 未用参数处理：LoRA 场景常见存在未参与反传的参数（仅注入的低秩层参与训练），
    # 为避免 "Expected to have finished reduction..." 错误，在启用 LoRA 时强制开启 find_unused_parameters。
    ddp_find_unused = (True if bool(getattr(args, 'lora_enable', False)) else (False if bool(args.train_llm) and bool(args.grad_ckpt) else True))

    grad_ckpt_effective = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=grad_ckpt_effective,
        max_grad_norm=(None if getattr(args, "max_grad_norm", 1.0) is not None and getattr(args, "max_grad_norm", 1.0) <= 0 else getattr(args, "max_grad_norm", 1.0)),
        # 启用/禁用未用参数检测（LoRA 强制启用以避免 DDP 报错）
        ddp_find_unused_parameters=ddp_find_unused,
        gradient_accumulation_steps=max(1, int(getattr(args, 'grad_accum_steps', 1))),

        logging_steps=10,
        save_steps=(args.save_every_steps if args.save_every_steps is not None else 1000),
        save_total_limit=(None if (hasattr(args, 'save_total_limit') and args.save_total_limit is not None and args.save_total_limit <= 0) else args.save_total_limit),
        # 保存策略：训练基础模块时保存；或显式强制保存
        save_strategy=("steps" if (train_any_qwen_component or args.force_save_checkpoints) else "no"),
        remove_unused_columns=False,  # 保留额外键（如 pixel_values / point_cloud_embeds）
        # 学习率调度（预热 + 余弦退火）
        lr_scheduler_type=str(getattr(args, 'lr_scheduler', 'cosine')),
        warmup_steps=int(getattr(args, 'warmup_steps', 0)),
        warmup_ratio=float(getattr(args, 'warmup_ratio', 0.0)),
        # 强制写入 TensorBoard 事件日志到输出目录下的 runs/
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "runs"),
        # DeepSpeed（ZeRO）：如传入配置文件则启用
        deepspeed=(str(args.deepspeed) if getattr(args, 'deepspeed', None) else None),
    )

    # 训练步内解码：在 loss 计算前打印并保存 pred/gt（贪心）。
    class LoggingTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None):
            # 分布式设备日志：每个进程仅打印一次自身 local_rank 与模型设备
            if not hasattr(self, "_dist_device_logged"):
                try:
                    import torch.distributed as dist
                    is_dist = dist.is_available() and dist.is_initialized()
                    rank = dist.get_rank() if is_dist else 0
                    world_size = dist.get_world_size() if is_dist else 1
                except Exception:
                    rank, world_size = 0, 1
                import os as _os_  # 局部别名，避免污染
                local_rank_env = _os_.environ.get("LOCAL_RANK")
                try:
                    local_rank = int(local_rank_env) if local_rank_env is not None else rank
                except Exception:
                    local_rank = rank
                try:
                    dev = next(model.parameters()).device
                except Exception:
                    dev = "unknown"
                print(f"[dist] rank={rank}/{world_size} local_rank={local_rank} device={dev}")
                setattr(self, "_dist_device_logged", True)
            labels = inputs.get("labels", None)
            # 前向，保留 logits 与 loss（遵循 HF 规范）
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)

            # 仅在 rank0 打印与保存；格式统一、可读性更强
            try:
                is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
            except Exception:
                is_rank0 = True
            if is_rank0 and labels is not None and logits is not None:
                try:
                    tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                    if tok is not None:
                        with torch.no_grad():
                            # teacher-forcing 预测：对每个位置的下一 token 进行贪心选择
                            pred_ids_all = torch.argmax(logits, dim=-1)
                        B = labels.shape[0]
                        save_path = Path(self.args.output_dir) / "preds.jsonl"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        max_print_chars = 256
                        for bi in range(B):
                            mask = labels[bi] != -100
                            if not mask.any():
                                continue
                            gt_ids = labels[bi][mask].detach().cpu().tolist()
                            pd_ids = pred_ids_all[bi][mask].detach().cpu().tolist()

                            # token-level准确率（严格比对）
                            correct = sum(int(x == y) for x, y in zip(pd_ids, gt_ids))
                            total = len(gt_ids)
                            acc = (float(correct) / float(total)) if total > 0 else 0.0

                            gt_text = tok.decode(gt_ids, skip_special_tokens=True)
                            pd_text = tok.decode(pd_ids, skip_special_tokens=True)
                            # 规范化打印：裁剪长文本，保留可读性
                            def _clip(s: str, n: int) -> str:
                                return (s[:n] + "…") if len(s) > n else s

                            step_print = int(getattr(self.state, "global_step", 0)) + 1
                            loss_val = float(getattr(outputs, "loss", torch.tensor(0.0)).detach().item())
                            print(f"[train] step={step_print} sample={bi} loss={loss_val:.4f} acc={acc:.3f}")
                            print(f"  gt:   {_clip(gt_text, max_print_chars)}")
                            print(f"  pred: {_clip(pd_text, max_print_chars)}")

                            # 结构化保存：JSONL 一行一条，字段规范
                            try:
                                rec = {
                                    "step": step_print,
                                    "sample_index": int(bi),
                                    "loss": loss_val,
                                    "token_acc": acc,
                                    "prediction": pd_text,
                                    "ground_truth": gt_text,
                                }
                                with save_path.open("a", encoding="utf-8") as fw:
                                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            except Exception:
                                pass
                except Exception:
                    pass
            # 返回 loss（若模型已计算则直接取用）
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                return loss
            return super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    # 用带日志的 Trainer 运行训练
    trainer = LoggingTrainer(
        model=qwen,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # 在训练开始前，尝试将 DDP 设为静态图，进一步规避重入反传触发的重复就绪标记。
    # 仅在同时训练 LLM 且启用梯度检查点时尝试。

    # ============ Projector 周期性保存（仅 rank0） ============
    class SaveProjectorCallback(TrainerCallback):
        def __init__(self, interval: int, out_dir: str):
            self.interval = int(interval)
            self.out_dir = out_dir

        def on_step_end(self, args, state, control, **kwargs):
            if self.interval is None or self.interval <= 0:
                return
            # 仅 rank0 执行写入
            try:
                import torch.distributed as dist
                is_dist = dist.is_available() and dist.is_initialized()
                rank = dist.get_rank() if is_dist else 0
            except Exception:
                rank = 0
            if rank != 0:
                return

            global_step = int(getattr(state, "global_step", 0))
            if global_step > 0 and (global_step % self.interval == 0):
                model = kwargs.get("model", None)
                if model is None:
                    return
                # 仅当训练 projector 时保存
                try:
                    pc_adapt = getattr(model, "model", model).pc_adapter_model if hasattr(getattr(model, "model", model), "pc_adapter_model") else None
                    train_proj = bool(getattr(getattr(model, "model", model), "pc_train_projector", False))
                    train_backbone = bool(getattr(getattr(model, "model", model), "pc_train_backbone", False))
                except Exception:
                    pc_adapt, train_proj, train_backbone = None, False, False
                if pc_adapt is None:
                    return
                import os
                os.makedirs(self.out_dir, exist_ok=True)
                # 保存 projector（如参与训练）
                if train_proj:
                    try:
                        sd = {f"point_proj.{k}": v.detach().cpu() for k, v in pc_adapt.point_proj.state_dict().items()}
                        if hasattr(pc_adapt, "align_mlp") and isinstance(pc_adapt.align_mlp, torch.nn.Linear):
                            sd.update({f"align_mlp.{k}": v.detach().cpu() for k, v in pc_adapt.align_mlp.state_dict().items()})
                        out_path = os.path.join(self.out_dir, f"point_proj_step-{global_step}.bin")
                        torch.save({"state_dict": sd}, out_path)
                        print(f"[Info] Periodic saved point_proj to: {out_path}")
                    except Exception as e:
                        print(f"[Warn] Periodic save point_proj failed: {e}")
                # 保存点云骨干（如参与训练），以 PointBERT 兼容前缀导出
                if train_backbone:
                    try:
                        bk_sd = {}
                        for k, v in pc_adapt.point_backbone.state_dict().items():
                            bk_sd[f"module.point_encoder.{k}"] = v.detach().cpu()
                        out_path_bk = os.path.join(self.out_dir, f"point_backbone_step-{global_step}.pt")
                        torch.save({"state_dict": bk_sd}, out_path_bk)
                        print(f"[Info] Periodic saved point_backbone to: {out_path_bk}")
                    except Exception as e:
                        print(f"[Warn] Periodic save point_backbone failed: {e}")

    if args.save_point_proj_interval is not None and args.save_point_proj_interval > 0:
        trainer.add_callback(SaveProjectorCallback(args.save_point_proj_interval, args.output_dir))

    # 训练损失曲线记录：按 logging_steps 追加到 loss.jsonl / loss.csv（仅 rank0）
    class LossCurveCallback(TrainerCallback):
        def __init__(self, out_dir: str):
            self.out_dir = out_dir

        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                import torch.distributed as dist
                is_dist = dist.is_available() and dist.is_initialized()
                rank = dist.get_rank() if is_dist else 0
            except Exception:
                rank = 0
            if rank != 0:
                return
            if logs is None:
                return
            loss_val = logs.get("loss", None)
            if loss_val is None:
                return
            step = int(getattr(state, "global_step", 0))
            lr = logs.get("learning_rate", None)
            epoch = logs.get("epoch", None)
            rec = {
                "step": step,
                "loss": float(loss_val),
                "learning_rate": (float(lr) if isinstance(lr, (int, float)) else lr),
                "epoch": epoch,
                "time": float(time.time()),
            }
            try:
                os.makedirs(self.out_dir, exist_ok=True)
                # JSONL 追加
                with open(os.path.join(self.out_dir, "loss.jsonl"), "a", encoding="utf-8") as fw:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # CSV 追加（简洁）
                csv_path = os.path.join(self.out_dir, "loss.csv")
                if not os.path.exists(csv_path):
                    with open(csv_path, "w", encoding="utf-8") as fw:
                        fw.write("step,loss,learning_rate,epoch,time\n")
                with open(csv_path, "a", encoding="utf-8") as fw:
                    fw.write(f"{step},{float(loss_val)},{'' if lr is None else lr},{'' if epoch is None else epoch},{int(rec['time'])}\n")
            except Exception:
                pass

    trainer.add_callback(LossCurveCallback(args.output_dir))
    
    
    # 训练：支持显式从 checkpoint 恢复（严格续训），否则从头开始本轮训练
    if getattr(args, "resume_from_checkpoint", None):
        print(f"[Info] Resume from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    # 仅当训练了 Qwen3-VL 的基础组件时，才保存完整模型权重与处理器/分词器配置
    if train_any_qwen_component or args.force_save_checkpoints:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    # 可选：LoRA 合并并导出整模型（仅 rank0 执行；保留 adapter 文件，整模型另存于 output_dir/merged）
    try:
        do_merge = bool(getattr(args, 'merge_lora_on_save', False)) and bool(getattr(args, 'lora_enable', False)) and bool(getattr(args, 'train_llm', False))
        if do_merge:
            # 仅 rank0 执行，避免并发写入
            try:
                import torch.distributed as dist
                is_dist = dist.is_available() and dist.is_initialized()
                rank = dist.get_rank() if is_dist else 0
            except Exception:
                rank = 0
            if rank == 0:
                try:
                    from peft import PeftModel  # type: ignore
                    is_peft = isinstance(qwen, PeftModel)
                except Exception:
                    is_peft = hasattr(qwen, 'merge_and_unload')
                if is_peft:
                    merged_dir = os.path.join(args.output_dir, 'merged')
                    os.makedirs(merged_dir, exist_ok=True)
                    try:
                        merged_model = qwen.merge_and_unload()
                        merged_model.save_pretrained(merged_dir)
                        # 同步导出处理器/分词器配置，确保 merged 目录可独立推理
                        processor.save_pretrained(merged_dir)
                        print(f"[Info] 已合并 LoRA 并导出完整模型到: {merged_dir}")
                    except Exception as e:
                        print(f"[Warn] LoRA 合并或保存失败: {e}")
                else:
                    print("[Info] 当前模型非 PEFT 包装或未启用 LoRA，跳过合并。")
    except Exception:
        pass

    # 可选：单独导出训练后的 projector / backbone 权重，供后续训练/推理加载
    try:
        # 仅 rank0 执行最终保存，避免 DDP 重复写入
        try:
            import torch.distributed as dist
            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
        except Exception:
            rank = 0
        if rank == 0:
            if args.save_point_proj_ckpt and hasattr(qwen.model, "pc_adapter_model") and qwen.model.pc_adapter_model is not None and args.train_point_projector:
                proj_sd = {f"point_proj.{k}": v.detach().cpu() for k, v in qwen.model.pc_adapter_model.point_proj.state_dict().items()}
                # 若存在非 Identity 的 align_mlp，一并保存，确保后续可完整加载
                if hasattr(qwen.model.pc_adapter_model, "align_mlp") and isinstance(qwen.model.pc_adapter_model.align_mlp, torch.nn.Linear):
                    align_sd = {f"align_mlp.{k}": v.detach().cpu() for k, v in qwen.model.pc_adapter_model.align_mlp.state_dict().items()}
                    proj_sd.update(align_sd)
                torch.save({"state_dict": proj_sd}, args.save_point_proj_ckpt)
                print(f"[Info] Saved trained point_proj(+align_mlp if any) to: {args.save_point_proj_ckpt}")
            if args.save_point_backbone_ckpt and hasattr(qwen.model, "pc_adapter_model") and qwen.model.pc_adapter_model is not None and args.train_point_backbone:
                bk_sd = {}
                for k, v in qwen.model.pc_adapter_model.point_backbone.state_dict().items():
                    bk_sd[f"module.point_encoder.{k}"] = v.detach().cpu()
                torch.save({"state_dict": bk_sd}, args.save_point_backbone_ckpt)
                print(f"[Info] Saved trained point_backbone to: {args.save_point_backbone_ckpt}")
    except Exception as e:
        print(f"[Warn] Failed to save point modules: {e}")


if __name__ == "__main__":
    main()
