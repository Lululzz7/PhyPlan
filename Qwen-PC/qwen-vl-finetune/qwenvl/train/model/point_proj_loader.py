import torch
from typing import Dict, Any, Optional


def load_point_proj_checkpoint(point_proj_module: torch.nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    """
    从包含完整 PointLLM 权重的 ckpt 中提取并加载 point_proj.* 权重到给定的 point_proj 子模块。

    用法示例：
        from qwenvl.train.model.point_proj_loader import load_point_proj_checkpoint
        load_point_proj_checkpoint(model.point_proj, "/path/to/pointllm_full_ckpt.pt")

    说明：
    - ckpt 可以是原始 state_dict（dict[str, Tensor]），也可以是 {"state_dict": ...} 包装。
    - 自动截断前缀 "point_proj."，将其余部分作为子模块相对键，以匹配 point_proj_module 的参数结构。
    - 返回结果包含加载报告，便于调试（missing_keys / unexpected_keys）。
    """
    if checkpoint_path is None:
        raise ValueError("checkpoint_path 不能为空")

    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("无效的 checkpoint 格式：应为 state_dict 或包含 'state_dict' 的字典")

    proj_sd = {}
    for k, v in sd.items():
        if "point_proj" in k:
            # 取子模块相对路径（去掉前缀 'point_proj.'）
            parts = k.split("point_proj.", 1)
            rel = parts[1] if len(parts) == 2 else None
            if rel:
                proj_sd[rel] = v

    if not proj_sd:
        # 兼容部分权重直接以子模块相对键保存（不含前缀）
        for k, v in sd.items():
            if k in dict(point_proj_module.state_dict().items()):
                proj_sd[k] = v

    if not proj_sd:
        raise RuntimeError("在 checkpoint 中未找到 point_proj.* 相关权重")

    report = point_proj_module.load_state_dict(proj_sd, strict=False)
    return {
        "missing_keys": getattr(report, "missing_keys", []),
        "unexpected_keys": getattr(report, "unexpected_keys", []),
        "loaded_keys": list(proj_sd.keys()),
    }


__all__ = ["load_point_proj_checkpoint"]

