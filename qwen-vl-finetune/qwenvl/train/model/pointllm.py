"""
Complete PointLLM module (non-overwriting copy).

This file provides a self-contained implementation of PointLLM that:
- Builds a point cloud backbone (PointBERT/PointTransformer) and a projector.
- Supports loading both backbone and point_proj weights from a PointLLM checkpoint.
- Integrates point cloud tokens into Llama embeddings during forward (optional usage).

It does not alter your existing pointllm.py; it is provided as an additional
complete reference implementation so you can import and test without replacing
current files.
"""

from __future__ import annotations

import os
import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# Local utils
from ..utils import cfg_from_yaml_file

logger = logging.getLogger(__name__)


def _load_point_proj_checkpoint_into_module(point_proj_module: nn.Module, checkpoint_path: str) -> dict:
    """
    Load point_proj.* weights from a PointLLM checkpoint into the provided module.

    - Accepts either a raw state_dict (dict[str, Tensor]) or a wrapper {"state_dict": ...}.
    - Strips the "point_proj." prefix to match the submodule's state_dict keys.
    - Returns a report dict with missing/unexpected/loaded keys for diagnostics.
    """
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must not be None")

    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("Invalid checkpoint format: expect state_dict or dict with 'state_dict'")

    proj_sd = {}
    for k, v in sd.items():
        if "point_proj" in k:
            parts = k.split("point_proj.", 1)
            rel = parts[1] if len(parts) == 2 else None
            if rel:
                proj_sd[rel] = v

    if not proj_sd:
        # Fallback: some checkpoints may store submodule keys without the prefix
        current_keys = set(point_proj_module.state_dict().keys())
        for k, v in sd.items():
            if k in current_keys:
                proj_sd[k] = v

    if not proj_sd:
        raise RuntimeError("No point_proj.* weights found in checkpoint")

    report = point_proj_module.load_state_dict(proj_sd, strict=False)
    return {
        "missing_keys": getattr(report, "missing_keys", []),
        "unexpected_keys": getattr(report, "unexpected_keys", []),
        "loaded_keys": list(proj_sd.keys()),
    }


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"


class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info("Using point backbone: %s", self.point_backbone_type)

        if self.point_backbone_type == "PointBERT":
            # Local import to avoid heavy deps at import time
            from qwenvl.train.model import PointTransformer

            # Load PointBERT yaml config colocated with this file
            point_bert_config_name = getattr(
                config, "point_backbone_config_name", "PointTransformer_8192point_2layer"
            )
            point_bert_config_addr = os.path.join(
                os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml"
            )
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6

            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info("Point dims: %s", self.point_backbone.point_dims)

            backbone_out = (
                point_bert_config.model.trans_dim
                if not use_max_pool
                else point_bert_config.model.trans_dim * 2
            )
            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": backbone_out,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": (
                    point_bert_config.model.num_group + 1 if not use_max_pool else 1
                ),
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get("projection_hidden_layer", 0),
                "use_max_pool": use_max_pool,
            }
            if self.point_backbone_config["projection_hidden_layer"] > 0:
                self.point_backbone_config["projection_hidden_dim"] = (
                    point_bert_config.model.projection_hidden_dim
                )
            logger.info(
                "Use max pool: %s; point token len: %s",
                use_max_pool,
                self.point_backbone_config["point_token_len"],
            )
        else:
            raise ValueError(f"Unsupported point backbone: {self.point_backbone_type}")

        # Build projector from backbone output -> model hidden size
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info("Backbone output dim: %s", backbone_output_dim)
        logger.info(
            "Projection hidden layers: %s",
            self.point_backbone_config["projection_hidden_layer"],
        )

        if self.point_backbone_config["projection_hidden_layer"] > 0:
            projection_layers: List[nn.Module] = []
            last_dim = backbone_output_dim
            hidden_dims = self.point_backbone_config["projection_hidden_dim"]
            for h in hidden_dims:
                projection_layers.append(nn.Linear(last_dim, h))
                projection_layers.append(nn.GELU())
                last_dim = h
            projection_layers.append(
                nn.Linear(last_dim, self.point_backbone_config["project_output_dim"])
            )
            self.point_proj = nn.Sequential(*projection_layers)
        else:
            self.point_proj = nn.Linear(
                backbone_output_dim, self.point_backbone_config["project_output_dim"]
            )
        logger.info(
            "Point projector output dim: %s",
            self.point_backbone_config["project_output_dim"],
        )

        self.fix_pointnet = False
        self.fix_llm = False

    def load_point_backbone_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load point backbone (PointBERT) weights from a PointLLM checkpoint."""
        ckpt = self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path
        if ckpt is None:
            logger.info("No backbone checkpoint provided; skip loading point backbone.")
            return
        self.point_backbone.load_checkpoint(ckpt)

    def load_point_proj_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load point_proj weights from a PointLLM checkpoint (point_proj.*)."""
        ckpt = self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path
        if ckpt is None:
            logger.info("No projector checkpoint provided; skip loading point_proj.")
            return
        try:
            report = _load_point_proj_checkpoint_into_module(self.point_proj, ckpt)
            # Expose loading report for downstream diagnostics
            try:
                self.point_proj_load_report = report  # type: ignore[attr-defined]
                self.point_proj_loaded_ckpt_path = ckpt  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.info(
                "Loaded point_proj from %s; loaded=%d, missing=%s, unexpected=%s",
                ckpt,
                len(report["loaded_keys"]),
                report["missing_keys"],
                report["unexpected_keys"],
            )
        except Exception as e:
            logger.warning("Failed to load point_proj: %s", e)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Optional path that injects point features into the language embeddings by
        replacing placeholder tokens in input_ids.

        Notes:
        - If you only need features (embeddings), you can call
          self.point_backbone(x) and then self.point_proj(...) directly.
        """

        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, "point_backbone", None)
        point_backbone_config = getattr(self, "point_backbone_config", None)

        if (
            point_backbone is not None
            and (input_ids is None or input_ids.shape[1] != 1 or self.training)
            and point_clouds is not None
        ):
            # Compute point features (trainable or frozen depending on fix_pointnet)
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if isinstance(point_clouds, list):
                    point_features_list: List[torch.Tensor] = []
                    for pc in point_clouds:
                        feat = self.point_backbone(pc.unsqueeze(0))[0]
                        point_features_list.append(feat)
                    point_features = point_features_list
                else:
                    point_features = self.point_backbone(point_clouds)

            # Project to model hidden size
            if isinstance(point_clouds, list):
                point_features = [self.point_proj(f) for f in point_features]
            else:
                point_features = self.point_proj(point_features)

            # Dummy for shape-consistent no-op
            dummy = torch.zeros(
                point_backbone_config["point_token_len"],
                point_backbone_config["backbone_output_dim"],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy = self.point_proj(dummy)

            new_input_embeds = []
            cur_point_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0.0 * dummy).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue

                cur_point_features = (
                    point_features[cur_point_idx]
                    if isinstance(point_features, list)
                    else point_features[cur_point_idx]
                    if point_features.ndim == 3
                    else point_features
                )
                cur_point_features = cur_point_features.to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0]

                if point_backbone_config["mm_use_point_start_end"]:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (
                        cur_input_ids == point_backbone_config["point_end_token"]
                    ).sum():
                        raise ValueError(
                            "The number of point start tokens and end tokens should match."
                        )
                    start_positions = torch.where(
                        cur_input_ids == point_backbone_config["point_start_token"]
                    )[0]
                    for pos in start_positions:
                        if (
                            cur_input_ids[pos + num_patches + 1]
                            != point_backbone_config["point_end_token"]
                        ):
                            raise ValueError(
                                "The point end token should follow immediately after inserted tokens."
                            )
                        if orig_embeds_params is not None:
                            cur_new = torch.cat(
                                (
                                    cur_input_embeds[:pos].detach(),
                                    cur_input_embeds[pos : pos + 1],
                                    cur_point_features,
                                    cur_input_embeds[pos + num_patches + 1 : pos + num_patches + 2],
                                    cur_input_embeds[pos + num_patches + 2 :].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new = torch.cat(
                                (
                                    cur_input_embeds[: pos + 1],
                                    cur_point_features,
                                    cur_input_embeds[pos + num_patches + 1 :],
                                ),
                                dim=0,
                            )
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new)
                else:
                    masked_indices = torch.where(
                        cur_input_ids == point_backbone_config["point_patch_token"]
                    )[0]
                    if masked_indices.numel() == 0:
                        new_input_embeds.append(cur_input_embeds)
                        cur_point_idx += 1
                        continue
                    mask_index_start = masked_indices[0]
                    expected = torch.arange(
                        mask_index_start,
                        mask_index_start + num_patches,
                        device=masked_indices.device,
                        dtype=masked_indices.dtype,
                    )
                    if (masked_indices != expected).any():
                        raise ValueError("Point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_point_features,
                                cur_input_embeds[mask_index_start + num_patches :].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_point_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new)
                    cur_point_idx += 1

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config: LlamaConfig):
        # Intentionally bypass LlamaForCausalLM __init__ to set custom model
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds,
            **kwargs,
        )

    # Tokenizer helpers retained for compatibility with training/inference setups
    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = (
            point_backbone_config["mm_use_point_start_end"]
        ) = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)
        point_backbone_config["default_point_patch_token"] = default_point_patch_token
        point_backbone_config["point_patch_token"] = tokenizer.convert_tokens_to_ids(
            [default_point_patch_token]
        )[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens(
                [default_point_start_token, default_point_end_token], special_tokens=True
            )
            point_backbone_config["default_point_start_token"] = default_point_start_token
            point_backbone_config["default_point_end_token"] = default_point_end_token
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_start_token]
            )[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_end_token]
            )[0]

    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = (
            point_backbone_config["mm_use_point_start_end"]
        ) = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config["default_point_patch_token"] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        point_backbone_config["point_patch_token"] = tokenizer.convert_tokens_to_ids(
            [default_point_patch_token]
        )[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config["default_point_start_token"] = default_point_start_token
            point_backbone_config["default_point_end_token"] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens(
                [default_point_start_token, default_point_end_token], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_start_token]
            )[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_end_token]
            )[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # Only tune input embeddings for the new tokens by default
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [
                        self.get_input_embeddings().weight.data.clone().to(device=device)
                    ]
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False
                    print(
                        f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable."
                    )
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")


# Register for Auto classes
AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
