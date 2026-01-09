"""
推理脚本：使用训练好的含点云输入的 Qwen3-VL 模型进行文本生成（支持仅点云+文字；可选图像/视频）。

用法示例：
  python infer_qwen_pointcloud.py \
    --model /path/to/finetuned/checkpoint \
    --point-cloud /path/to/point.npy \
    --prompt "请根据点云描述场景" \
    --max-new-tokens 64 --bf16

注意：
  - 点云嵌入采用“前置拼接 tokens”的方式注入到序列中；与 PointLLM 的 "<point>" 文本占位符替换不同（无需在文本中写入 <point>）。
  - 颜色点云（N,6，RGB 在 arr[:,3:]）将统一归一化到 [0,1]（若检测到 >1 的值则按 255 缩放并裁剪到 [0,1]）。
  - 该脚本在第一步前置注入点云，随后逐步增量解码（不再重复注入点云）。
  - 依赖训练时的点云骨干配置（PointBERT YAML 与 ckpt），需与训练保持一致。
"""

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
import sys as _sys
_sys.path.append(str(project_root))

from qwenvl.train.modeling_qwen3_vl_pointcloud import Qwen3VLModel_PointCloud
from qwenvl.train.model.point_adapter import create_pointcloud_adapter


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """从 logits 采样下一个 token（支持温度与 nucleus sampling）。"""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        # nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = torch.nonzero(cumulative > top_p, as_tuple=False)
        if cutoff.numel() > 0:
            k = cutoff[0, 0].item()
            mask = torch.zeros_like(sorted_probs)
            mask[..., : k + 1] = 1.0
            probs = torch.zeros_like(probs)
            probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * mask)
            probs /= probs.sum() + 1e-9
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


def main():
    parser = argparse.ArgumentParser(description="Inference for Qwen3-VL with Point Cloud input")
    parser.add_argument("--model", type=str, required=True, help="训练好的模型路径或 HuggingFace ID")
    parser.add_argument("--point-cloud", type=str, default=None, help="点云文件路径（.npy/.npz/.ply/.pcd），形如 (N,3) 或 (N,6)")
    parser.add_argument("--image", type=str, default=None, help="可选图像路径（jpg/png 等），单张")
    parser.add_argument("--images", type=str, default=None, help="可选多张图像，逗号分隔路径列表")
    parser.add_argument("--videos", type=str, default=None, help="可选多段视频，逗号分隔路径列表")
    # 提示语在单样本模式必需；批量评估模式可省略
    parser.add_argument("--prompt", type=str, required=False, help="用户文本提示（单样本模式必填；批量评估可省略）")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    # 注意力实现：适配非 FA2 环境（可选：flash_attention_2/sdpa/eager）
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="注意力实现后端（默认为模型默认设置；无 FA2 环境可选 sdpa/eager）",
    )
    # 生成方式：官方 generate（默认开启）或手写采样
    parser.add_argument("--use-generate", action="store_true", default=True,
                        help="使用官方 generate 接口进行生成（默认开启，更稳健，功能更全）")
    parser.add_argument("--no-generate", dest="use_generate", action="store_false",
                        help="关闭 generate，改为使用手写增量采样（不建议，功能较少）")
    # 点云骨干配置（需与训练一致）
    parser.add_argument("--point-backbone", type=str, default="PointBERT")
    parser.add_argument("--point-backbone-config", type=str, default="PointTransformer_8192point_2layer")
    parser.add_argument("--point-use-color", action="store_true")
    parser.add_argument("--point-num", type=int, default=8192, help="推理时可选采样点数（>0 启用远点采样）")
    parser.add_argument("--point-backbone-ckpt", type=str, default=None,
                        help="PointBERT/PointTransformer骨干权重文件，如 point_bert_v1.2.pt")
    parser.add_argument("--point-proj-ckpt", type=str, default=None,
                        help="projector 权重文件，可为 HF 分片或提取后的 .pt")
    # 前向集成开关：在 forward 中计算点云嵌入（推理不训练）
    parser.add_argument("--pc-adapter-forward", action="store_true")
    # 批量评估：读取 JSON/JSONL 标注，逐条生成并保存预测与 groundtruth
    parser.add_argument("--data", type=str, default=None, help="标注文件（json/jsonl），字段同训练")
    parser.add_argument("--save-path", type=str, default=None, help="保存结果 JSONL 的路径；默认与数据同目录下生成 eval_results.jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="仅评估前 N 条（0 表示全部）")

    args = parser.parse_args()
    # 参数有效性检查：当未提供批量数据 --data 时，要求 --prompt 与 --point-cloud 至少提供点云文件
    if args.data is None:
        if args.prompt is None:
            raise SystemExit("[ERROR] --prompt 为必填参数（单样本模式）。若使用批量评估，请提供 --data 以读取标注文件。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    # 1) 加载模型与处理器
    if args.attn_implementation:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, attn_implementation=args.attn_implementation, torch_dtype=torch_dtype
        ).to(device)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch_dtype
        ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    # 替换底层模型为点云前置版本（权重兼容）
    pc_model = Qwen3VLModel_PointCloud(model.model.config).to(device)
    pc_model.load_state_dict(model.model.state_dict(), strict=True)
    model.model = pc_model

    # 兼容 generate：为 prepare_inputs_for_generation 轻量补丁，首步保留点云键
    # 说明：Qwen3-VL 的 prepare_inputs_for_generation 不会主动保留额外 kwargs；
    # 这里仅在首步（past_key_values 为空或 cache_position 为 0）将点云键回填到返回的 model_inputs。
    import types as _types
    _orig_pifg = model.prepare_inputs_for_generation

    def _pifg_with_pc(self,
                      input_ids,
                      past_key_values=None,
                      attention_mask=None,
                      inputs_embeds=None,
                      cache_position=None,
                      position_ids=None,
                      use_cache=True,
                      pixel_values=None,
                      pixel_values_videos=None,
                      image_grid_thw=None,
                      video_grid_thw=None,
                      **kwargs):
        mi = _orig_pifg(input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        use_cache=use_cache,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        **kwargs)
        # 首步注入点云：仅当没有缓存或缓存位置为 0
        is_first_step = (past_key_values is None) or (
            cache_position is not None and isinstance(cache_position, torch.Tensor) and cache_position.numel() > 0 and cache_position[0].item() == 0
        )
        if is_first_step:
            pc_embeds_kw = kwargs.get("point_cloud_embeds", None)
            pc_clouds_kw = kwargs.get("point_clouds", None)
            if pc_embeds_kw is not None:
                mi["point_cloud_embeds"] = pc_embeds_kw
            if pc_clouds_kw is not None:
                mi["point_clouds"] = pc_clouds_kw
        return mi

    model.prepare_inputs_for_generation = _types.MethodType(_pifg_with_pc, model)

    # 允许在 generate 的 kwargs 中携带 point_cloud_embeds/point_clouds：
    # Transformers 的 _validate_model_kwargs 会检查 forward 的签名是否接受这些键；
    # 为避免报错，这里将 forward 替换为可接受 **kwargs 的版本，并原样传递给原始 forward，
    # 使得底层 Qwen3VLModel_PointCloud 能正确消费这些键。
    _orig_forward = model.forward
    def _forward_accept_kwargs(self, *args, **kwargs):
        return _orig_forward(*args, **kwargs)
    model.forward = _types.MethodType(_forward_accept_kwargs, model)

    # 修补 transformers 的签名缓存，允许在 generate 中携带点云相关键
    try:
        sig_cols = set(getattr(model, "_signature_columns", []))
        prep_cols = set(getattr(model, "_prepare_signature_columns", []))
        extra = {"point_cloud_embeds", "point_clouds"}
        setattr(model, "_signature_columns", sig_cols | extra)
        setattr(model, "_prepare_signature_columns", prep_cols | extra)
    except Exception:
        pass

    # 2) 点云适配器（与训练配置一致，仅在使用点云时构建）
    hidden_size = model.config.text_config.hidden_size
    adapter_fn = None
    pc_adapter_model = None
    if args.point_cloud or args.data:
        if args.pc_adapter_forward:
            adapter_fn, meta, pc_adapter_model = create_pointcloud_adapter(
                hidden_size=hidden_size,
                point_backbone=args.point_backbone,
                point_backbone_config_name=args.point_backbone_config,
                use_color=args.point_use_color,
                point_backbone_ckpt=args.point_backbone_ckpt,
                point_proj_ckpt=args.point_proj_ckpt,
                device=device,
                eval_mode=True,
                return_model=True,
            )
            model.model.pc_adapter_model = pc_adapter_model
            model.model.pc_adapter_trainable = False
        else:
            adapter_fn, meta = create_pointcloud_adapter(
                hidden_size=hidden_size,
                point_backbone=args.point_backbone,
                point_backbone_config_name=args.point_backbone_config,
                use_color=args.point_use_color,
                point_backbone_ckpt=args.point_backbone_ckpt,
                point_proj_ckpt=args.point_proj_ckpt,
                device=device,
                eval_mode=True,
            )

    # 3) 加载点云并生成嵌入（严格遵循 PointLLM 预处理：use_color/归一化/可选采样）
    def _load_points_array(path: Path) -> Optional[np.ndarray]:
        suf = path.suffix.lower()
        if suf in {".npy", ".npz"}:
            arr = np.load(str(path))
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[list(arr.keys())[0]]
            arr = arr.astype(np.float32)
        elif suf in {".ply", ".pcd"}:
            try:
                import open3d as o3d
            except Exception:
                return None
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pts.size == 0:
                return None
            if len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0:
                cols = np.asarray(pcd.colors, dtype=np.float32)
                arr = np.concatenate([pts, cols], axis=1)
            else:
                arr = pts
        else:
            return None
        if (not args.point_use_color) and arr.shape[-1] > 3:
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
        if isinstance(args.point_num, int) and args.point_num > 0 and arr.shape[0] > args.point_num:
            n = arr.shape[0]
            xyz_np = arr[:, :3]
            centroids = np.zeros((args.point_num,), dtype=np.int64)
            distance = np.ones((n,), dtype=np.float64) * 1e10
            farthest = np.random.randint(0, n)
            for i in range(args.point_num):
                centroids[i] = farthest
                c = xyz_np[farthest]
                dist = np.sum((xyz_np - c) ** 2, axis=1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = int(np.argmax(distance))
            arr = arr[centroids]
        return arr.astype(np.float32)

    pc_embeds = None
    point_clouds = None
    if args.point_cloud:
        pc_path = Path(args.point_cloud)
        arr = _load_points_array(pc_path)
        if arr is not None:
            pc_tensor = torch.from_numpy(arr).float().unsqueeze(0)
            if args.pc_adapter_forward:
                point_clouds = pc_tensor
            else:
                with torch.no_grad():
                    pc_embeds = adapter_fn(pc_tensor, out_device=device)

    # 4) 文本编码
    # 如果提供 --data，则进行批量评估；否则走单条推理
    if args.data:
        # 读取标注
        def _read_anns(path: str):
            p = Path(path)
            if p.suffix == ".jsonl":
                out = []
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        out.append(json.loads(line))
                return out
            else:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)

        from qwenvl.data.data_processor import _build_messages
        import json
        anns = _read_anns(args.data)
        save_path = Path(args.save_path) if args.save_path else (Path(args.data).with_name("eval_results.jsonl"))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        count = 0
        with save_path.open("w", encoding="utf-8") as fw:
            for idx, sample in enumerate(anns):
                base_path = Path(sample.get("data_path", "")) if isinstance(sample, dict) else Path("")
                messages = _build_messages(sample if isinstance(sample, dict) else sample[0], base_path)
                # 提取 ground truth（最后一个 assistant）并构造 prompt（移除最后一个 assistant）
                last_ass_idx = None
                gt = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "assistant":
                        last_ass_idx = i
                        gt = "".join([c.get("text", "") for c in messages[i]["content"] if c.get("type") == "text"])
                        break
                prompt_msgs = messages if last_ass_idx is None else messages[:last_ass_idx]

                inputs = processor.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # 图像/视频
                pixel_values = None
                image_grid_thw = None
                pixel_values_videos = None
                video_grid_thw = None
                try:
                    from qwen_vl_utils.vision_process import process_vision_info
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        prompt_msgs,
                        return_video_kwargs=True,
                        image_patch_size=getattr(processor.image_processor, 'patch_size', 14),
                    )
                    if image_inputs is not None:
                        pack_img = processor(images=image_inputs, return_tensors="pt", do_resize=False)
                        pixel_values = pack_img.get("pixel_values").to(device)
                        image_grid_thw = pack_img.get("image_grid_thw").to(device)
                    if video_inputs is not None and hasattr(processor, 'video_processor') and processor.video_processor is not None:
                        pack_vid = processor(videos=video_inputs, return_tensors="pt", do_resize=False, **(video_kwargs or {}))
                        pixel_values_videos = pack_vid.get("pixel_values_videos").to(device)
                        video_grid_thw = pack_vid.get("video_grid_thw").to(device)
                except Exception:
                    pass

                # 点云
                pc_path = sample.get("point_cloud", None) if isinstance(sample, dict) else None
                pc_embeds_eval = None
                point_clouds_eval = None
                if pc_path:
                    arr = _load_points_array(Path(pc_path))
                    if arr is not None:
                        pc = torch.from_numpy(arr).float().unsqueeze(0)
                        if args.pc_adapter_forward:
                            point_clouds_eval = pc
                        else:
                            with torch.no_grad():
                                pc_embeds_eval = adapter_fn(pc, out_device=device)

                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": (args.temperature != 1.0 or args.top_p < 1.0),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "pixel_values_videos": pixel_values_videos,
                    "video_grid_thw": video_grid_thw,
                    "point_cloud_embeds": pc_embeds_eval,
                    "point_clouds": point_clouds_eval,
                }
                try:
                    with torch.no_grad():
                        seq = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
                    new_ids = seq[0, input_ids.shape[1]:]
                    pred = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
                except ValueError as e:
                    # 兼容性回退：若 transformers 校验拒绝 point_cloud_* 相关键，则使用手写增量采样
                    if "model_kwargs" in str(e) and "point_cloud_embeds" in str(e):
                        past_key_values = None
                        cache_position = None
                        # 先进行首步前向（注入点云前缀）
                        outputs = model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            point_cloud_embeds=pc_embeds_eval,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            pixel_values_videos=pixel_values_videos,
                            video_grid_thw=video_grid_thw,
                            point_clouds=point_clouds_eval,
                        )
                        logits = model.lm_head(outputs.last_hidden_state)
                        past_key_values = outputs.past_key_values
                        # 估计点云前缀长度（用于 cache_position）
                        pc_len_est = int(pc_embeds_eval.shape[1]) if pc_embeds_eval is not None else int(meta.get("point_token_len", 0))
                        cache_position = torch.tensor([pc_len_est + input_ids.shape[1]], device=device)

                        new_tokens: List[int] = []
                        for _ in range(args.max_new_tokens):
                            next_id = sample_next_token(logits[:, -1, :].squeeze(0), temperature=args.temperature, top_p=args.top_p)
                            new_tokens.append(next_id)
                            step_ids = torch.tensor([[next_id]], device=device)
                            step_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
                            with torch.no_grad():
                                step_out = model.model(
                                    input_ids=step_ids,
                                    attention_mask=step_mask,
                                    past_key_values=past_key_values,
                                    cache_position=cache_position,
                                    point_cloud_embeds=None,
                                )
                                step_logits = model.lm_head(step_out.last_hidden_state)
                            logits = step_logits
                            past_key_values = step_out.past_key_values
                            cache_position = cache_position + 1
                        gen_ids = torch.tensor(new_tokens, device=device)
                        pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    else:
                        raise

                last_user_text = None
                for j in range(len(prompt_msgs) - 1, -1, -1):
                    if prompt_msgs[j]["role"] == "user":
                        last_user_text = " ".join([
                            c.get("text", "") for c in prompt_msgs[j]["content"] if c.get("type") == "text"
                        ])
                        break
                print(f"[{idx}] Q: {last_user_text or ''}\n    pred: {pred}\n    gt:   {gt or ''}")

                rec = {
                    "index": int(idx),
                    "point_cloud": pc_path,
                    "question": last_user_text,
                    "prediction": pred,
                    "ground_truth": gt,
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

                count += 1
                if args.max_samples and count >= args.max_samples:
                    break

        print(f"已保存 {count} 条结果到: {str(save_path)}")
        return

    # 单条推理路径：
    # 5) 预填充（前置注入点云嵌入）
    past_key_values = None
    cache_position = None
    # 可选视觉张量（支持多图/视频）
    pixel_values = None
    image_grid_thw = None
    pixel_values_videos = None
    video_grid_thw = None
    try:
        # 统一从 messages 中解析图像/视频
        from qwen_vl_utils.vision_process import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=getattr(processor.image_processor, 'patch_size', 14),
        )
        if image_inputs is not None:
            pack_img = processor(images=image_inputs, return_tensors="pt", do_resize=False)
            pixel_values = pack_img.get("pixel_values").to(device)
            image_grid_thw = pack_img.get("image_grid_thw").to(device)
        if video_inputs is not None and hasattr(processor, 'video_processor') and processor.video_processor is not None:
            pack_vid = processor(videos=video_inputs, return_tensors="pt", do_resize=False, **(video_kwargs or {}))
            pixel_values_videos = pack_vid.get("pixel_values_videos").to(device)
            video_grid_thw = pack_vid.get("video_grid_thw").to(device)
    except Exception:
        # 无视觉或未安装依赖时忽略
        pass

    # 5) 预填充（前置注入点云嵌入）
    # 计算点云前缀长度，用于后续增量解码的位置更新
    pc_len = 0
    if pc_embeds is not None:
        pc_len = int(pc_embeds.shape[1])
    elif point_clouds is not None and pc_adapter_model is not None:
        # 使用适配器在 forward 中计算嵌入时，依据 meta 中的 point_token_len 估计长度
        try:
            # 当使用 return_model=True 构建适配器时，meta 已就绪
            from typing import Any
            # meta 定义于上面的适配器创建分支；若不可用，则兜底为 0
            pc_len = int(meta.get("point_token_len", 0))  # type: ignore[name-defined]
        except Exception:
            pc_len = 0

    if getattr(args, "use_generate", False):
        # 使用官方 generate 流（更稳健，自动维护缓存/位置）
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": (args.temperature != 1.0 or args.top_p < 1.0),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            # 点云仅在首次调用中注入；generate 内部后续步将基于缓存继续
            "point_cloud_embeds": pc_embeds,
            "point_clouds": point_clouds,
        }
        with torch.no_grad():
            sequences = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
        # 仅解码新生成部分（不包含提示）
        new_ids = sequences[0, input_ids.shape[1] :]
        text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        print("[Response]", text_out)
    else:
        # 手写增量采样（轻量定制）
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            point_cloud_embeds=pc_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            point_clouds=point_clouds,
        )
        logits = model.lm_head(outputs.last_hidden_state)  # (1, P+L, V)

        new_tokens: List[int] = []
        past_key_values = outputs.past_key_values
        # 包含点云前缀长度，确保后续位置索引与缓存一致
        cache_position = torch.tensor([pc_len + input_ids.shape[1]], device=device)

        for _ in range(args.max_new_tokens):
            next_id = sample_next_token(logits[:, -1, :].squeeze(0), temperature=args.temperature, top_p=args.top_p)
            new_tokens.append(next_id)

            step_ids = torch.tensor([[next_id]], device=device)
            step_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)

            with torch.no_grad():
                step_out = model.model(
                    input_ids=step_ids,
                    attention_mask=step_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    point_cloud_embeds=None,
                )
                step_logits = model.lm_head(step_out.last_hidden_state)  # (1, 1, V)

            logits = step_logits
            past_key_values = step_out.past_key_values
            cache_position = cache_position + 1

        # 解码输出
        gen_ids = torch.tensor(new_tokens, device=device)
        text_out = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        print("[Response]", text_out)


if __name__ == "__main__":
    main()
