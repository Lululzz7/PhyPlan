# Qwen3‑VL 点云输入说明与示例（中文）

本文档说明在本仓库中如何为 Qwen3‑VL 集成点云输入，涵盖训练与推理的输入规范、预处理流程、以及完整的示例。

## 模态与输入总览
- 支持的模态：文本（必需）、图像（可选，多张）、视频（可选，多段）、点云（可选，单个）
- 训练使用消息模板与占位符（`<image>` / `<video>`）描述多模态对话；点云路径在样本字典中以 `point_cloud` 指定。
- 推理支持命令行脚本（推荐）或 Python 代码；点云作为“前缀嵌入”一次性注入到序列开头，随后使用缓存进行增量解码。

## 点云输入规范
- 文件格式：`.npy` 或 `.npz`（数组键取第一个），形状 `(N, 3)` 或 `(N, 6)`
  - `(N, 3)`: 仅 `xyz`
  - `(N, 6)`: `xyzRGB`（或 `xyz` + 其他 3 维属性），若开启 `use_color` 会保留颜色通道。
- 预处理（遵循 PointLLM）：
  - 可选去色：若未开启 `use_color`，仅保留前三维 `xyz`。
  - 单位球归一化：减去质心后，按最大半径缩放至单位球。
  - 远点采样（Farthest Point Sampling）：若 `N > point_num`（如 8192），将点数下采样到固定值，保证全局覆盖。
- 嵌入生成策略：
  - 离线（默认）：先用适配器将点云变为 `(B, P, hidden_size)` 嵌入，再在前向中作为前缀拼接。
  - 在线：在前向中调用点云骨干 + projector 生成嵌入（可选择训练骨干或投影器）。
- 前缀长度 `P`：由 PointBERT 配置决定（例如 `PointTransformer_8192point_2layer.yaml` 中，`P=num_group+1` 当未 max‑pool；`P=1` 当启用 max‑pool）。

## 训练数据格式（JSON/JSONL）
每条样本示例：
```json
{
  "data_path": "/abs/base/dir",
  "image": ["images/room.jpg"],
  "video": [],
  "point_cloud": "points/scene.npy",
  "conversations": [
    {"from": "human", "value": "<image> 请根据点云与图像描述场景"},
    {"from": "assistant", "value": "……"}
  ]
}
```
说明：
- `conversations` 使用 `<image>` 或 `<video>` 占位符依次消费 `image`/`video` 列表；数量必须匹配，否则会报错。
- 标签 `labels` 通过增量模板自动生成，仅对 assistant 段落进行监督；点云前缀在模型前向中被忽略（`-100`），避免重复注入。
- 视觉张量：使用处理器打包为 `pixel_values`/`image_grid_thw`（图像）与 `pixel_values_videos`/`video_grid_thw`（视频）。
- 位置编码：通过 mRoPE（3 通道）统一计算，视频使用时间戳方式（参考 `get_rope_index_3` 的行为）。

## 训练命令示例
最简用法（单卡）：
```
python qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py \
  --model Qwen/Qwen3-VL-7B-Instruct \
  --train-file /path/to/data.jsonl \
  --output-dir ./outputs/pc_run \
  --epochs 1 --batch-size 1 --bf16 \
  --point-backbone PointBERT \
  --point-backbone-config PointTransformer_8192point_2layer \
  --point-backbone-ckpt /path/to/point_bert_v1.2.pt \
  --point-proj-ckpt /path/to/point_proj_extracted.pt \
  --save-point-proj-ckpt /path/to/save_point_proj_after_training.pt \
  --save-point-backbone-ckpt /path/to/save_point_backbone_after_training.pt \
  --point-use-color --point-num 8192
```
分布式：
```
torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
  qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py \
  --model Qwen/Qwen3-VL-7B-Instruct \
  --train-file /path/to/data.jsonl \
  --output-dir ./outputs/pc_run --epochs 1 --batch-size 1 --bf16 \
  --point-backbone PointBERT --point-backbone-config PointTransformer_8192point_2layer \
  --point-backbone-ckpt /path/to/point_bert_v1.2.pt --point-proj-ckpt /path/to/point_proj_extracted.pt \
  --point-use-color --point-num 8192
```
可训练模块开关：
- `--train-visual` 训练视觉编码器
- `--train-connector` 训练视觉与语言连接（merger）
- `--train-llm` 训练语言模型与输出头
- 点云细粒度：`--train-point-backbone`（骨干）、`--train-point-projector`（projector + 对齐层）

## 推理（命令行脚本）
标准用法：
```
python qwen-vl-finetune/qwenvl/train/infer_qwen_pointcloud.py \
  --model /path/to/checkpoint \
  --point-cloud /path/to/points.npy \
  --prompt "请根据点云描述场景" \
  --bf16 --use-generate \
  --point-backbone PointBERT \
  --point-backbone-config PointTransformer_8192point_2layer \
  --point-backbone-ckpt /path/to/point_bert_v1.2.pt \
  --point-proj-ckpt /path/to/point_proj_extracted.pt \
  --point-use-color --point-num 8192
```
可选视觉：
```
  --image /path/to/img.jpg \
  --images /abs/a.jpg,/abs/b.jpg \
  --videos /abs/a.mp4,/abs/b.mp4
```
在线嵌入（在前向中生成）：
```
  --pc-adapter-forward
```
注意：点云嵌入只在首步注入，后续由缓存驱动；脚本已补丁 `prepare_inputs_for_generation` 以确保首步保留点云键。

## 推理（Python 示例）
```python
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwenvl.train.modeling_qwen3_vl_pointcloud import Qwen3VLModel_PointCloud
from qwenvl.train.model.point_adapter import create_pointcloud_adapter

model_path = "Qwen/Qwen3-VL-7B-Instruct"

# 1) 加载模型并替换为点云前置版本
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
pc_model = Qwen3VLModel_PointCloud(model.model.config).cuda()
pc_model.load_state_dict(model.model.state_dict(), strict=True)
model.model = pc_model
processor = AutoProcessor.from_pretrained(model_path)

# 2) 构建点云适配器（离线嵌入）
adapter_fn, meta = create_pointcloud_adapter(
    hidden_size=model.config.text_config.hidden_size,
    point_backbone="PointBERT",
    point_backbone_config_name="PointTransformer_8192point_2layer",
    use_color=True,
    device=torch.device("cuda"),
    eval_mode=True,
)

# 3) 加载并预处理点云（与脚本一致：保留颜色、单位球归一化、远点采样）
arr = np.load("/path/to/points.npy")
if isinstance(arr, np.lib.npyio.NpzFile):
    arr = arr[list(arr.keys())[0]]
# 若不使用颜色
# arr = arr[:, :3] if arr.shape[-1] > 3 else arr
xyz = arr[:, :3]
centroid = np.mean(xyz, axis=0)
xyz = xyz - centroid
m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
xyz = xyz / m if m > 0 else xyz
arr = np.concatenate([xyz, arr[:, 3:]], axis=1) if arr.shape[-1] > 3 else xyz
# 远点采样到 8192
point_num = 8192
if arr.shape[0] > point_num:
    n = arr.shape[0]
    xyz_np = arr[:, :3]
    centroids = np.zeros((point_num,), dtype=np.int64)
    distance = np.ones((n,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(point_num):
        centroids[i] = farthest
        centroid = xyz_np[farthest]
        dist = np.sum((xyz_np - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    arr = arr[centroids]
pc = torch.from_numpy(arr).float().unsqueeze(0)
pc_embeds = adapter_fn(pc, out_device=torch.device("cuda"))  # (1, P, hidden)

# 4) 构造消息并生成
messages = [{"role": "user", "content": [{"type": "text", "text": "请根据点云描述场景"}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
with torch.no_grad():
    sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, point_cloud_embeds=pc_embeds)
new_ids = sequences[0, input_ids.shape[1]:]
print(processor.tokenizer.decode(new_ids, skip_special_tokens=True))
```

## 视觉处理与位置编码（参考）
- 视觉打包：
  - 图像：`pixel_values`、`image_grid_thw`
  - 视频：`pixel_values_videos`、`video_grid_thw`
- 位置编码：
  - 使用 `get_rope_index_3` 计算 3 通道 mRoPE 位置（文本 / 视觉 patch），视频采用时间戳序列，保证与 Qwen3‑VL 官方逻辑一致。

## 点云嵌入与标签处理
- 嵌入注入：点云嵌入在首步前置拼入 `inputs_embeds` 的开头；`attention_mask` 前置 1；`position_ids` 右移并拼接点云位置；`visual_pos_masks` 对应位置扩展。
- 标签忽略：若存在 `labels`，会在前缀长度处拼接 `-100`，避免点云前缀参与损失。
- 缓存与步进：`generate()` 仅首步注入点云，后续基于缓存与 `cache_position` 前进。

## 依赖与环境
- `transformers>=4.57.0`
- `qwen-vl-utils==0.0.14`
- `timm==0.4.12`（PointBERT 需要）
- `easydict`、`pyyaml`（读取 PointBERT YAML 配置）
- 推荐：GPU 环境，`--bf16` 以节省显存；如无 FA2 环境可用 `--attn-implementation sdpa/eager`。

## 常见问题与排查
- 点云长度不一致：批内各样本点云 token 数必须一致（如都为 `P=num_group+1`）；否则适配器会报错。
- 占位符不匹配：`<image>/<video>` 数量必须与提供的文件数一致；否则会报错。
- 首步注入：仅首步注入点云；若自定义采样循环，需正确维护 `cache_position` 为 `pc_len + 输入长度`。
- 无点云样本：自动用零前缀填充，且标签前缀被忽略。
