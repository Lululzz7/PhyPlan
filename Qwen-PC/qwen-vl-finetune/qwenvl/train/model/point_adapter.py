from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

# 复用现有 PointLLM 的点云骨干与投影逻辑
from .pointllm import PointLLMLlamaModel, PointLLMConfig


def _infer_point_proj_out_dim_from_ckpt(ckpt_path: Optional[str]) -> Optional[int]:
    """
    推断 PointLLM checkpoint 中 point_proj 最后一层 Linear 的输出维度。

    支持以下格式：
    - 纯 state_dict: {"point_proj.*": Tensor, ...}
    - 包装格式: {"state_dict": {...}}
    返回：若成功解析，返回 out_features；否则返回 None。
    """
    if not ckpt_path:
        return None
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            return None
        candidates: List[Tuple[int, torch.Tensor]] = []
        for k, v in sd.items():
            if not k.startswith("point_proj.") or not k.endswith(".weight"):
                continue
            parts = k.split(".")
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[1])
            except Exception:
                continue
            candidates.append((idx, v))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        last_weight = candidates[-1][1]
        if last_weight.ndim >= 2:
            return int(last_weight.shape[0])
        return None
    except Exception:
        return None


def create_pointcloud_adapter(
    hidden_size: int,
    point_backbone: str = "PointBERT",
    point_backbone_config_name: str = "PointTransformer_8192point_2layer",
    use_color: bool = False,
    point_backbone_ckpt: Optional[str] = None,
    point_proj_ckpt: Optional[str] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
    mm_use_point_start_end: bool = False,
    return_model: bool = False,
) -> Tuple[Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor], Dict[str, Any]]:
    """
    构建一个点云特征适配器（函数），用于将原始点云张量转换为 Qwen3-VL 可用的嵌入表示。

    返回：
        - adapter(point_clouds) -> torch.Tensor: (B, P, hidden_size)
        - meta: 包含 point_token_len、backbone_output_dim、hidden_size 等信息

    说明：
        - 该适配器内部复用 PointLLMLlamaModel 的点云骨干与投影层；不改任何权重命名，保持权重兼容。
        - point_clouds 支持 BxNxD 的张量或长度为 B 的 list（每个元素 N_i x D）。若为 list，则要求各样本输出的 token 数相同以便 stack。
    """
    # 若提供了 PointLLM ckpt，则使用 ckpt 的 point_proj 输出维度来构建投影层，
    # 随后通过对齐 MLP 将其映射到 Qwen3-VL 的 hidden_size。
    # 根据 projector 的权重推断其输出维度（而非骨干权重）
    proj_out_dim_ckpt = _infer_point_proj_out_dim_from_ckpt(point_proj_ckpt)
    pll_hidden_for_build = proj_out_dim_ckpt if proj_out_dim_ckpt is not None else hidden_size

    cfg = PointLLMConfig(
        hidden_size=pll_hidden_for_build,
        point_backbone=point_backbone,
        point_backbone_config_name=point_backbone_config_name,
        use_color=use_color,
        mm_use_point_start_end=mm_use_point_start_end,
    )

    model = PointLLMLlamaModel(cfg)
    # 初始化阶段一律放在 CPU，避免在分布式尚未就绪时占用 GPU:0 导致 OOM；
    # 随后由上层 Trainer/Accelerate 在各自进程统一迁移到对应 GPU。
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    if point_backbone_ckpt is not None:
        # Load encoder from backbone checkpoint
        model.load_point_backbone_checkpoint(point_backbone_ckpt)
    # Optionally load projector from a separate checkpoint (e.g., HF LLM shards or extracted file)
    if point_proj_ckpt is not None:
        try:
            model.load_point_proj_checkpoint(point_proj_ckpt)
        except Exception:
            # 若加载失败不影响后续流程；可由调用方单独处理
            pass

    if eval_mode:
        model.eval()
        # 避免训练时误更新点云骨干/投影（可按需放开）
        for p in model.point_backbone.parameters():
            p.requires_grad = False
        for p in model.point_proj.parameters():
            p.requires_grad = False

    # Build an optional alignment layer to match Qwen hidden size
    # 优先采用 ckpt 解析出的输出维度；否则从模块实例中推断
    proj_out_dim: Optional[int] = proj_out_dim_ckpt
    if proj_out_dim is None:
        try:
            if isinstance(model.point_proj, nn.Sequential):
                last = None
                for m in model.point_proj.modules():
                    if isinstance(m, nn.Linear):
                        last = m
                if last is not None:
                    proj_out_dim = int(last.weight.shape[0])
            elif isinstance(model.point_proj, nn.Linear):
                proj_out_dim = int(model.point_proj.out_features)
        except Exception:
            proj_out_dim = None
    if proj_out_dim is None:
        proj_out_dim = int(model.point_backbone_config.get("project_output_dim", hidden_size))

    # 对齐到 Qwen3-VL hidden_size
    if proj_out_dim != hidden_size:
        model.align_mlp = nn.Linear(proj_out_dim, hidden_size)
    else:
        model.align_mlp = nn.Identity()

    # 若未提供 projector ckpt，则对 point_proj/align_mlp（如为线性层）进行明确初始化
    if point_proj_ckpt is None:
        def _init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.point_proj.apply(_init_linear)
        if isinstance(model.align_mlp, nn.Linear):
            _init_linear(model.align_mlp)

    # 若提供的 ckpt 中包含 align_mlp.*，则一并加载（与 point_proj 同源文件）
    model._align_mlp_loaded_from_ckpt = False  # type: ignore[attr-defined]
    if point_proj_ckpt is not None and isinstance(model.align_mlp, nn.Linear):
        try:
            sd = torch.load(point_proj_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                align_sd = {}
                for k, v in sd.items():
                    if k.startswith("align_mlp."):
                        rel = k.split("align_mlp.", 1)[1]
                        align_sd[rel] = v
                if align_sd:
                    model.align_mlp.load_state_dict(align_sd, strict=False)
                    model._align_mlp_loaded_from_ckpt = True  # type: ignore[attr-defined]
        except Exception:
            pass

    meta = {
        "point_token_len": model.point_backbone_config["point_token_len"],
        "backbone_output_dim": model.point_backbone_config["backbone_output_dim"],
        "project_output_dim": proj_out_dim,
        "hidden_size": hidden_size,
    }

    @torch.no_grad()
    def _extract(point_clouds: Union[torch.Tensor, List[torch.Tensor]],
                 out_dtype: Optional[torch.dtype] = None,
                 out_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        将点云转为嵌入：
        - 输入：
            point_clouds: (B, N, D) 或 list[Tensor{N_i, D}]
        - 输出：
            (B, P, hidden_size)，其中 P 为 point_token_len 或骨干产生的 token 数
        """
        mdl_device = next(model.parameters()).device
        if out_device is None:
            out_device = mdl_device

        def _to_dev(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=mdl_device)

        # 提取 backbone 特征
        if isinstance(point_clouds, list):
            feats = []
            for pc in point_clouds:
                pc = _to_dev(pc)
                f = model.point_backbone(pc.unsqueeze(0))[0]  # (P, C)
                feats.append(f)
            # 投影到 hidden_size
            feats = [model.point_proj(f) for f in feats]  # list[(P, proj_out_dim)]
            feats = [model.align_mlp(f) for f in feats]   # list[(P, hidden)]
            # 要求各样本 P 一致，便于 stack
            lens = [f.shape[0] for f in feats]
            if len(set(lens)) != 1:
                raise ValueError(f"Batch 内各样本点云 token 长度不一致: {lens}")
            embeds = torch.stack(feats, dim=0)  # (B, P, hidden)
        else:
            pc = _to_dev(point_clouds)
            feats = model.point_backbone(pc)              # (B, P, C)
            embeds = model.point_proj(feats)              # (B, P, proj_out_dim)
            embeds = model.align_mlp(embeds)              # (B, P, hidden)

        if out_dtype is not None:
            embeds = embeds.to(dtype=out_dtype)
        embeds = embeds.to(device=out_device)
        return embeds

    if return_model:
        return _extract, meta, model
    return _extract, meta


def extract_pointcloud_embeds(
    point_clouds: Union[torch.Tensor, List[torch.Tensor]],
    hidden_size: int,
    point_backbone: str = "PointBERT",
    point_backbone_config_name: str = "PointTransformer_8192point_2layer",
    use_color: bool = False,
    point_backbone_ckpt: Optional[str] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
    mm_use_point_start_end: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    便捷函数：一次性构建适配器并提取嵌入。
    返回 (B, P, hidden_size) 的点云嵌入，可直接传给 Qwen3VLModel_PointCloud.forward 的 point_cloud_embeds。
    """
    adapter, _ = create_pointcloud_adapter(
        hidden_size=hidden_size,
        point_backbone=point_backbone,
        point_backbone_config_name=point_backbone_config_name,
        use_color=use_color,
        point_backbone_ckpt=point_backbone_ckpt,
        device=device,
        eval_mode=eval_mode,
        mm_use_point_start_end=mm_use_point_start_end,
    )
    return adapter(point_clouds, out_dtype=out_dtype)
