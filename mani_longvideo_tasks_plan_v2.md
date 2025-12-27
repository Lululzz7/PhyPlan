# mani_longvideo 多模态任务划分规划 v2（面向长视频：JSON + 关键帧 + 均匀抽帧 + Step 间片段 + 累积前缀长视频）

本文档目标：在 `mani_longvideo.py` 生成的单视频输出目录内，利用以下数据源：

- `causal_plan_with_keyframes.json`（计划文本 + 关键帧标注）
- 每个step的关键帧图像：`steps[*].critical_frames[*].keyframe_image_path`
- 整个视频的均匀抽帧图像：`sampled_frames/`（默认 50 帧，`sample_###_ts_YY.YYs.jpg`）
- 两个Step间视频片段：相邻 step 的“尾关键帧”之间的 `last_frame_segments/*.mp4`
- 累积前缀长视频：从视频开始到每个 step 尾关键帧的 `cumulative_last_frame_segments/*.mp4`

系统性地构建**可落地生成**的多任务多模态 QA/监督数据。

与 v1 相比：v2 在保留 `Task_01~Task_17`（对齐 PhyPlan 17 类）基础上，新增一组**视频/时间维度任务**（`Task_18~Task_27`），并将每个任务明确绑定其“证据形态”（文本/关键帧/多图/视频片段/累积前缀）。

---

## 1. 单视频输出目录（Item）约定

以 `causal_spafa_plan_dataset_long/P01_01_part1/` 为例，一个 Item（单视频输出目录）通常包含：

```
<item_dir>/
  causal_plan_with_keyframes.json
  causal_plan.json
  run_summary.json
  sampled_frames/
    sample_001_ts_0.00s.jpg
    ...
    sample_050_ts_XXX.XXs.jpg
  01_<step_slug>/
    frame_007_ts_25.19s.jpg            # 关键帧（命名来自 frame_index + timestamp）
    ...
  02_<step_slug>/
    ...
```

因此，v2 对每个任务都给出“证据来源优先级 + fallback 规则”，以确保数据生成稳定且不产生低质量样本。

---

## 2. `causal_plan_with_keyframes.json` 的核心 Schema（当前实际产物）

顶层：
- `high_level_goal: str`
- `steps: List[Step]`

Step：
- `step_id: int`
- `step_goal: str`
- `rationale: str`
- `preconditions: List[str]`
- `expected_effects: List[str]`
- `tool_and_material_usage.tools: List[str]`
- `tool_and_material_usage.materials: List[str]`
- `causal_challenge_question: str`
- `expected_challenge_outcome: str`
- `failure_handling.reason: str`
- `failure_handling.recovery_strategy: str`
- `critical_frames: List[CriticalFrame]`

CriticalFrame：
- `frame_index: int`（1-based，对齐 `sampled_frames` 的帧序号）
- `keyframe_image_path: str`（绝对或相对；建议归一为绝对路径）
- `action_description: str`
- `state_change_description: str`
- `spatial_preconditions: List[{relation:str, objects:[str], truth:bool}]`
- `affordance_preconditions: List[{object_name:str, affordance_types:[str], reasons:str}]`
- `causal_chain: {agent, action, patient, causal_effect_on_patient, causal_effect_on_environment}`
- `affordance_hotspot: {description, affordance_type, mechanism}`

---

## 3. 多模态证据形态（必须显式写入 meta）

为避免“字段复述任务”和“视觉/视频核验任务”混杂，v2 统一为每条样本记录：

- `meta.evidence_type` ∈
  - `text_only`
  - `keyframe_single`
  - `images_uniform_scene`（从 `sampled_frames/` 选 4–8 张）
  - `images_uniform_clip`（从某个视频片段均匀选 6–12 张）
  - `video_clip`（直接给 mp4；仅适用于视频模型或后续会抽帧的管线）
  - `video_prefix`（累积前缀长视频 mp4）

建议在 `meta` 中同时记录“证据来源”，便于追溯：

- `meta.evidence_source` ∈ `sampled_frames|keyframes|last_frame_segments|cumulative_last_frame_segments`
- `meta.evidence_files: List[str]`（最终喂给模型的图片列表，或 mp4 路径列表）

同时为视频证据提供统一结构：

- Step 间片段：`last_frame_segments/segment_stepXX_to_stepYY.mp4`
  - 语义：从 stepXX 的尾关键帧时刻到 stepYY 的尾关键帧时刻（弱对齐“中间过程”）
  - manifest：`last_frame_segments/segments_manifest.json`

- 累积前缀：`cumulative_last_frame_segments/segment_start_to_stepXX_last.mp4`
  - 语义：从视频开始到 stepXX 的尾关键帧时刻（强对齐“前缀进度”）
  - manifest：`cumulative_last_frame_segments/segments_manifest.json`

---

## 4. 输出样本结构（建议）

延续 `generate_phyplan_api.py` 的 ShareGPT 风格，但扩展 `meta` 以覆盖视频：

```json
{
  "id": "uuid",
  "image": ["/abs/path/img1.jpg", "/abs/path/img2.jpg"],
  "video": "/abs/path/clip.mp4",
  "conversations": [
    {"from": "human", "value": "<英文问句>"},
    {"from": "gpt", "value": "<英文回答>"}
  ],
  "meta": {
    "task_name": "Task_20_Step_Boundary_Localization",
    "item_type": "TypeA|TypeB|TypeC|TypeD",
    "evidence_type": "video_clip",
    "source_path": ".../causal_plan_with_keyframes.json",
    "step_index": 2,
    "frame_index": 18,
    "segment_label": "segment_step02_to_step03",
    "step_goal": "...",
    "high_level_goal": "...",
    "fields": {"...": "..."}
  }
}
```

约定：
- `image` 与 `video` 可二选一或同时存在。
- 如果当前训练/推理模型不支持 `video`，则把 mp4 预抽帧成 `image` 序列，并把 `meta.evidence_type` 设为 `images_uniform_clip`。

---

## 5. 任务体系 v2

### 5.1 对齐 PhyPlan 的 17 类任务（Task_01~Task_17）

这些任务以“计划结构 + 关键帧”为核心监督信号；v2 只对每个任务补充“推荐证据形态”，并修正字段映射以匹配当前 `causal_plan_with_keyframes.json` 实际产物。

#### Task_01_Macro_Anchor_Extraction（场景锚点/关键可交互对象）
- **任务定义**：抽取任务相关的稳定锚点对象（工具/材料/关键实体）。
- **字段（JSONPath）**：
  - `steps[*].tool_and_material_usage.tools[*]`
  - `steps[*].tool_and_material_usage.materials[*]`
  - `steps[*].critical_frames[*].spatial_preconditions[*].objects[*]`
  - `steps[*].critical_frames[*].affordance_preconditions[*].object_name`
  - `steps[*].critical_frames[*].causal_chain.agent`
  - `steps[*].critical_frames[*].causal_chain.patient`
- **证据形态**：`images_uniform_scene`（从 `sampled_frames/` 选 4–8 张，覆盖环境）

#### Task_02_Transient_Geometric_Verification（瞬时空间关系验证）
- **任务定义**：描述/核验某关键帧中的空间关系。
- **字段**：`critical_frames[*].spatial_preconditions[*].relation/objects/truth`
- **证据形态**：`keyframe_single`
- **注意**：若训练目标是“视觉核验（true/false/uncertain）”，应使用 v2 新增的 `Task_27_Visual_Spatial_Relation_Check`，不要与本任务混用。

#### Task_03_Micro_Affordance_Visual_Semantics（微观可供性热点语义）
- **任务定义**：描述关键帧中的可供性热点区域 + 类别 + 物理机制。
- **字段**：`critical_frames[*].affordance_hotspot.description/affordance_type/mechanism`
- **证据形态**：`keyframe_single`

#### Task_04_Entity_Role_Identification（工具/材料角色区分）
- **字段**：`steps[*].tool_and_material_usage.tools/materials`
- **证据形态**：`keyframe_single`（该 step 最早关键帧）或 `images_uniform_scene`

#### Task_05_State_Evolution_Description（动作-状态变化描述）
- **字段**：`critical_frames[*].action_description/state_change_description`
- **证据形态**：`keyframe_single`

#### Task_06_Holistic_Causal_Chain_Analysis（因果链分析）
- **字段**：`critical_frames[*].causal_chain.*` + `critical_frames[*].affordance_preconditions` + `spatial_preconditions`
- **证据形态**：`keyframe_single`

#### Task_07_Scene_Goal_Derivation（场景高阶目标）
- **字段**：`high_level_goal`
- **证据形态**：`images_uniform_scene`

#### Task_08_Strategic_Rationale_Justification（步骤动机/必要性解释）
- **字段**：`steps[*].rationale`（可附 `high_level_goal` 与 `step_goal` 作为上下文）
- **证据形态**：`keyframe_single`（该 step 最早关键帧）

#### Task_09_Precondition_Statement（步骤前置条件陈述）
- **字段**：`steps[*].preconditions`
- **证据形态**：`keyframe_single`（该 step 最早关键帧）

#### Task_10_Step_Execution_Statement（步骤执行动作描述）
- **字段**：`steps[*].step_goal`（当前产物未包含 `navigation_and_manipulation`，因此默认复述/细化 goal）
- **证据形态**：`images_uniform_scene` 或 `images_uniform_clip`（更适合给出动作细节）

#### Task_11_Expected_Physical_Effects（期望效果/后置状态）
- **字段**：`steps[*].expected_effects`
- **证据形态**：`keyframe_single`（该 step 尾关键帧更合适；也可用该 step 最后一张关键帧图像）

#### Task_12_Inter_Step_Dependency_Analysis（跨步依赖分析）
- **字段**：`steps[i].expected_effects` ↔ `steps[i+1].preconditions`
- **证据形态**：`keyframe_single`（step i 的尾关键帧，或 step i 的最早关键帧）
- **约束**：必须通过词项重合/包含检测后才生成，避免牵强解释。

#### Task_13_Next_Action_Prediction（下一步/下一动作预测：基于计划）
- **字段**：v2 推荐拆成两种版本（二选一，别混在同一任务名下）：
  - **(A) 下一步目标预测**：标签为 `steps[i+1].step_goal`
  - **(B) 下一微动作预测**：若未来在 JSON 中补充 `predicted_next_actions` 再启用
- **证据形态**：`keyframe_single` 或 `video_prefix`

#### Task_14_Counterfactual_Prediction（反事实挑战与结果）
- **字段**：`steps[*].causal_challenge_question` + `expected_challenge_outcome`
- **证据形态**：`keyframe_single`

#### Task_15_Failure_Recovery_Protocol（失败模式与恢复策略）
- **字段**：`steps[*].failure_handling.reason/recovery_strategy`
- **证据形态**：`keyframe_single`（更建议给“容易失败”的关键帧）

#### Task_16_Physical_Feasibility_Verification（物理可行性核验）
- **字段**：`critical_frames[*].spatial_preconditions` + `affordance_preconditions`
- **证据形态**：`keyframe_single`

#### Task_17_Holistic_Step_Synthesis_Why_How（步级综合：Why/How）
- **字段**：Why=`steps[*].rationale`；How=`critical_frames[*].causal_chain.*` 或 `affordance_hotspot.mechanism`
- **证据形态**：`keyframe_single`（选择同时含“机制/因果链”的关键帧）

---

### 5.2 新增：面向长视频的“时间/视频证据”任务（Task_18~Task_27）

这些任务的核心价值：把 `sampled_frames`（全局上下文）、`last_frame_segments`（step 间过程）和 `cumulative_last_frame_segments`（前缀进度）引入监督，使数据集不仅能“读懂计划文本”，还能“用视频证据做定位/核验/对齐”。

#### Task_18_Visual_Precondition_Check（视觉前置条件核验）
- **任务定义**：给定 step i 的前置条件列表，要求基于图像/视频证据判断哪些已满足、哪些不满足，并说明证据。
- **字段**：`steps[i].preconditions[*]` + `steps[i].step_goal` + `high_level_goal`
- **证据形态（推荐）**：
  - `video_prefix`：`cumulative_last_frame_segments/segment_start_to_step{i-1}_last.mp4`（若 i=1，则用 `sampled_frames` 前若干帧）
  - 或 `images_uniform_scene`：从 `sampled_frames/` 取前 6–10 张
- **输出形式**：自然英文段落，逐条覆盖 preconditions（不要 bullet），明确“可见/不可见/不确定”。

#### Task_19_Visual_Effect_Check（视觉后置效果核验）
- **任务定义**：给定 step i 的期望效果，结合 step i 的“尾关键帧”及其附近过程，核验效果是否达成并指出证据。
- **字段**：`steps[i].expected_effects[*]` + `steps[i].step_goal`
- **证据形态（推荐）**：
  - `keyframe_single`：step i 的最后一张关键帧 `critical_frames[-1].keyframe_image_path`
  - 可加 `images_uniform_clip`：来自 `last_frame_segments/segment_step{i}_to_step{i+1}.mp4` 的均匀抽帧（若存在）

#### Task_20_Step_Boundary_Localization（Step 边界定位/转折点解释）
- **任务定义**：在相邻 step 的尾帧片段中，定位“从 step i 过渡到 step i+1 的关键转折”（不要求精确时间戳；用“发生了什么变化/出现了什么新对象/动作开始”描述即可）。
- **字段**：`steps[i].step_goal` + `steps[i+1].step_goal`
- **证据形态**：`video_clip`（`last_frame_segments/segment_step{i}_to_step{i+1}.mp4`）或其 `images_uniform_clip`
- **输出形式**：两段：第一段描述转折点现象；第二段说明为何它代表 step 切换。

#### Task_21_Keyframe_Justification（关键帧选择理由）
- **任务定义**：解释为什么某张关键帧“足够关键”（对应的动作/状态变化/空间条件/可供性机制）。
- **字段**：`critical_frames[j].action_description/state_change_description/spatial_preconditions/affordance_hotspot`
- **证据形态**：`keyframe_single`
- **价值**：把“关键帧=信息瓶颈”转化为可监督的“解释任务”，提高关键帧利用效率。

#### Task_22_Plan_Execution_Alignment（计划-执行一致性判别）
- **任务定义**：判断给定视频证据是否与某个 step_goal 一致（match/partial/mismatch），并给出依据。
- **字段**：`steps[i].step_goal`（可附 `expected_effects` 作为参考）
- **证据形态（推荐）**：
  - `video_clip`：如果你能得到 step 内片段则最好；在当前工具链下可先用 `last_frame_segments` 作为近似
  - 或 `images_uniform_clip`：从 `last_frame_segments/segment_step{i}_to_step{i+1}.mp4` 均匀抽帧
- **负样本构造（推荐）**：
  - 用 step i 的证据去配 step k（k≠i）作为 mismatch/partial。

#### Task_23_Goal_Recognition_From_Prefix（前缀目标识别）
- **任务定义**：仅给视频从开始到 step i 的累积前缀，让模型推断全局 `high_level_goal`。
- **字段**：`high_level_goal`
- **证据形态**：`video_prefix`（`cumulative_last_frame_segments/segment_start_to_step{i}_last.mp4`）或其抽帧序列
- **难度控制**：i 从小到大逐渐增加信息量，可用于 curriculum。

#### Task_24_Next_Step_Goal_Prediction_From_Prefix（前缀预测下一步目标）
- **任务定义**：给“已完成到 step i 的累积前缀”，预测下一步 `steps[i+1].step_goal`。
- **字段**：标签为 `steps[i+1].step_goal`；输入可附 `steps[i].step_goal` 作为已知上下文
- **证据形态**：`video_prefix`
- **注意**：这是 v2 推荐的 `Task_13` 的“视觉版本”，二者不要混同。

#### Task_25_Progress_Summary_From_Prefix（前缀进度总结）
- **任务定义**：给累积前缀，让模型总结“完成了哪些步骤/当前所处阶段/关键对象状态”。
- **字段**：可用 `steps[0..i].step_goal` 作为弱监督参考；也可只把它写入 `meta.fields` 供评测
- **证据形态**：`video_prefix`
- **输出形式**：自然英文段落，避免清单式 bullet（但允许逗号分隔的短并列）。

#### Task_26_Temporal_Order_Check（时间顺序判别）
- **任务定义**：给两个动作/状态变化描述，要求判断哪个更早发生，并指出视频证据。
- **字段（构造方式）**：
  - 从 `steps[i].critical_frames[a].action_description/state_change_description` 与 `steps[k].critical_frames[b]...` 抽取两条候选
  - 标签来自其在视频中的时间顺序（可用 `frame_index` 或关键帧文件名的 `ts_XXs`）
- **证据形态**：`images_uniform_scene`（全局抽帧）或 `video_prefix`
- **价值**：让模型学习长视频的时间一致性，而不是只看静态关键帧。

#### Task_27_Visual_Spatial_Relation_Check（视觉空间关系真假核验）
- **任务定义**：给定关键帧图像 + 一个空间关系陈述，判断该关系在图像中是否成立。
- **字段**：
  - 输入：`spatial_preconditions[k].relation/objects`
  - 标签：`spatial_preconditions[k].truth`
- **证据形态**：`keyframe_single`
- **负样本构造（推荐）**：
  - 对同一关键帧：随机抽一个不合理的对象对（例如把 `objects` 替换为同 step 的其他对象），标签设为 false（注意：此类负样本是“弱负样本”，需要在 `meta.neg_sample=true` 标注）。
  - 对同一 `(relation, objects)`：把 `truth` 取反作为对照（同样标注 `meta.neg_sample=true`）。

---


## 6. 质量约束与自动检查

1) **字段完整性**：缺关键字段则跳过生成（或写入 `meta.missing_fields` 并标记降级）。
2) **媒体存在性**：图片/mp4 不存在则跳过；或写入占位路径但必须在 `meta` 标注 `missing_media=true`。
3) **不臆造**：当证据不足，答案必须表达不确定性，而不是编造。
4) **语言规范**：自然英文段落；避免 bullet/标题/模板口吻；`verbatim` 型任务（如 Task_07）可以原样返回。
5) **一致性**：`meta.step_index/frame_index/segment_label` 必须能回溯到源数据。

---

## 7. Item 类型（用于与现有管线对齐）

为了与 `generate_phyplan_api.py` 的处理习惯保持一致，建议把样本按证据与粒度分为：

- `TypeA`：关键帧驱动（单图）——主要覆盖 Task_02/03/05/06/16/17 + 部分步级任务
- `TypeB`：均匀抽帧驱动（多图场景）——主要覆盖 Task_01/07/10/26（全局时间顺序）
- `TypeC`：Step 间片段驱动（mp4 或抽帧序列）——主要覆盖 Task_20/22/19
- `TypeD`：累积前缀驱动（mp4 或抽帧序列）——主要覆盖 Task_18/23/24/25/26

---

## 8. 取图/抽帧规则（落地细节）

### 8.1 `images_uniform_scene`（从 `sampled_frames/` 选图）

- 输入（优先级从高到低）：
  1) `<item_dir>/sampled_frames/` 下的 `sample_*.jpg`（长度通常为 50）
  2) 若 sampled_frames 缺失：用 `causal_plan_with_keyframes.json` 内所有 step 的关键帧作为“场景覆盖代理”（推荐用每个 step 的最早关键帧 `critical_frames[0].keyframe_image_path`）。
- 选取策略（两种输入都适用）：用等距采样索引（包含首尾），例如 k=8：
  - `idx = round(linspace(1, N, k))`（1-based），再映射到文件路径。
- 目的：覆盖全局环境与任务阶段变化，避免只取前几帧。

### 8.2 `images_uniform_clip`（从 mp4 片段抽帧）

如果你的下游不支持 `video` 字段，建议对 mp4 片段预抽帧：

- 建议命名：
  - `<item_dir>/clip_frames/<segment_label>/clip_001.jpg ... clip_012.jpg`
- 建议抽帧方式（ffmpeg 示例，12 帧）：
  - `ffmpeg -i <clip.mp4> -vf fps=12/\<duration_sec\> -q:v 2 clip_%03d.jpg`
- 更稳健做法：先从 `segments_manifest.json` 拿到 `duration_sec`，再决定抽帧数与 fps。

### 8.3 Step 级任务用“哪张关键帧”

推荐约定（确保一致性）：

- “描述/计划类”（Task_04/08/09/10/14/15）：优先用该 step 的**最早关键帧**（`critical_frames[0]`）
- “效果/完成状态类”（Task_11/19）：优先用该 step 的**最后关键帧**（`critical_frames[-1]`）
- “瞬时物理/机制类”（Task_02/03/05/06/16/17/21/27）：用对应 `critical_frames[j]`

---


## 9. 任务卡片（逐任务：字段 + 多模态来源 + QA 范例）

说明：以下每张任务卡都包含：

- **字段来源（JSONPath）**：构造 `meta.fields` 与 prompt 的唯一允许来源
- **多模态证据来源**：必须严格从指定路径取图/取片段
- **样本构造规则**：如何在 step/frame/segment 上取样
- **QA 范例**：使用 `causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json` 中的真实字段（图片/片段路径按约定写出）

为保证范例可读性，下面默认：

- `ITEM_DIR = causal_spafa_plan_dataset_long/P01_01_part1`
- `SOURCE_JSON = <ITEM_DIR>/causal_plan_with_keyframes.json`

### Task_01_Macro_Anchor_Extraction

- **字段（JSONPath）**：
  - `steps[*].tool_and_material_usage.tools[*]`
  - `steps[*].tool_and_material_usage.materials[*]`
  - `steps[*].critical_frames[*].spatial_preconditions[*].objects[*]`
  - `steps[*].critical_frames[*].affordance_preconditions[*].object_name`
  - `steps[*].critical_frames[*].causal_chain.agent`
  - `steps[*].critical_frames[*].causal_chain.patient`
- **证据来源（严格优先级）**：
  1) `images_uniform_scene` from `<ITEM_DIR>/sampled_frames/sample_*.jpg`（等距取 4–8 张）
  2) 若 `sampled_frames/` 缺失：改用“每步最早关键帧集合”作为 `images_uniform_scene`（仍等距采样到 4–8 张）
- **样本构造规则**：每个 item 生成 1 条（去重、统一小写/下划线规范化可选）。
- **meta.fields（建议最小集）**：`key_objects_for_planning`（去重后对象列表）
- **范例（基于 P01_01_part1）**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... (8 images)
fields.key_objects_for_planning ~= ["light_switch","refrigerator","cucumber","carrot","knife","cutting_board","sink","faucet","water","stovetop","frying_pan","box_grater"]
Q: Which stable objects are the task-relevant anchors for planning in this scene?
A: The task-relevant anchors include the light switch, refrigerator, cucumber and carrot, the cutting board and knife, the sink and faucet (with water), the stovetop and frying pan, and the box grater.
```

### Task_02_Transient_Geometric_Verification

- **字段（JSONPath）**：`steps[i].critical_frames[j].spatial_preconditions[k].relation/objects/truth`
- **证据来源**：`keyframe_single` = `steps[i].critical_frames[j].keyframe_image_path`
- **样本构造规则**：
  - 对每个关键帧，最多抽 1–2 条 spatial 关系（否则关系爆炸）。
  - 本任务的回答建议是“关系自然语言复述”，不输出 truth 判别；判别用 `Task_27`。
- **meta.fields（建议最小集）**：`relation`, `objects`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.relation = "contact"
fields.objects = ["hand","light_switch"]
Q: What is the precise spatial relationship between hand and light_switch in this frame?
A: In this frame, the hand is in contact with the light_switch.
```

### Task_03_Micro_Affordance_Visual_Semantics

- **字段（JSONPath）**：`critical_frames[j].affordance_hotspot.description/affordance_type/mechanism`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每个关键帧最多 1 条。
- **meta.fields（建议最小集）**：`affordance_type`, `description`, `mechanism`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.affordance_type = "pressable_surface"
fields.description = "The raised surface of the rocker switch designed for finger contact."
fields.mechanism = "The surface provides a target for applying force, which is then mechanically transferred ... powering the lights."
Q: Which specific region affords pressable_surface, and how does it visually appear and physically function?
A: The raised surface of the rocker switch is the pressable region; it provides a clear finger-contact target so applied force transfers to the internal toggle mechanism that completes the circuit and turns on the lights.
```

### Task_04_Entity_Role_Identification

- **字段（JSONPath）**：`steps[i].tool_and_material_usage.tools/materials`（可附 `steps[i].step_goal`）
- **证据来源**：`keyframe_single`（该 step 最早关键帧）
- **样本构造规则**：每个 step 至少 1 条（若 tools/materials 均为空则跳过）。
- **meta.fields（建议最小集）**：`tools`, `materials`, `step_goal`
- **范例（Step 2）**：

```text
Image: <ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg
fields.tools = ["refrigerator"]
fields.materials = ["cucumber","carrot"]
fields.step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: In the step Retrieve a carrot and a cucumber from the refrigerator, which items function as tools, and which are the materials being acted upon?
A: The refrigerator functions as the tool or container being accessed, while the cucumber and carrot are the materials being grasped and removed.
```

### Task_05_State_Evolution_Description

- **字段（JSONPath）**：`critical_frames[j].action_description/state_change_description`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每个关键帧最多 1 条。
- **meta.fields（建议最小集）**：`action_description`, `state_change_description`
- **范例（Step 4, frame 20）**：

```text
Image: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg
fields.action_description = "The person holds a cucumber with both hands under a stream of water ... rubbing its surface to clean it."
fields.state_change_description = "The surface of the cucumber is being cleaned as water and friction from the hands remove any dirt."
Q: Given the overall goal the mission, what ongoing action is occurring, and what immediate state change does it cause?
A: The person is holding and rubbing the cucumber under running water, which immediately causes the cucumber’s surface to be cleaned as contaminants are removed.
```

### Task_06_Holistic_Causal_Chain_Analysis

- **字段（JSONPath）**：
  - `critical_frames[j].causal_chain.agent/action/patient/causal_effect_on_patient/causal_effect_on_environment`
  - （可选支撑）`critical_frames[j].spatial_preconditions`, `critical_frames[j].affordance_preconditions`, `critical_frames[j].affordance_hotspot.mechanism`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每个关键帧最多 1 条。
- **meta.fields（建议最小集）**：`agent`, `action`, `patient`, `eff_pat`, `eff_env`, `mechanism`, `spatial_preconditions`, `affordance_preconditions`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.agent = "hand"
fields.action = "is applying downward force"
fields.patient = "light_switch"
fields.eff_pat = "is undergoing a rotational pivot, which completes an internal electrical circuit."
fields.eff_env = "The room's ambient light level is increasing from dark to illuminated."
Q: Could you explain how hand is applying downward force light_switch in this keyframe, focusing on the spatial setup, the affordance-level mechanism, and the immediate effects?
A: The hand is positioned within reach of the wall-mounted light_switch and makes direct contact with it, allowing force to be applied to the switch’s pressable surface. This contact-based spatial setup enables the hand to actuate the toggleable control.

By pressing the rocker surface, the applied force transfers into the internal toggle mechanism, causing the switch to pivot and complete the electrical circuit. As an immediate effect, the light_switch changes state and the room’s ambient lighting increases, making the environment illuminated.
```

### Task_07_Scene_Goal_Derivation

- **字段（JSONPath）**：`high_level_goal`
- **证据来源（严格优先级）**：同 `Task_01` 的 `images_uniform_scene` 规则（优先 sampled_frames，其次关键帧代理）。
- **样本构造规则**：每个 item 1 条。
- **meta.fields（建议最小集）**：`high_level_goal`
- **范例**：

```text
Images (scene): <ITEM_DIR>/sampled_frames/sample_001_ts_0.00s.jpg ... (8 images)
fields.high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board."
Q: Given the current scene, what is the appropriate high-level goal?
A: Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board.
```

### Task_08_Strategic_Rationale_Justification

- **字段（JSONPath）**：`steps[i].rationale`（可附 `high_level_goal`, `step_goal`）
- **证据来源**：`keyframe_single`（该 step 最早关键帧）
- **样本构造规则**：每个 step 1 条。
- **meta.fields（建议最小集）**：`rationale`, `step_goal`, `high_level_goal`
- **范例（Step 1）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.rationale = "This initial step is essential to enable all subsequent actions ... require adequate lighting for safety and efficiency."
Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, why is the step Enter the kitchen and turn on the light to illuminate the workspace strategically necessary?
A: This initial step is essential to enable all subsequent actions. Navigating the environment, locating items, and performing tasks like chopping require adequate lighting for safety and efficiency.
```

### Task_09_Precondition_Statement

- **字段（JSONPath）**：`steps[i].preconditions[*]`
- **证据来源**：`keyframe_single`（该 step 最早关键帧）
- **样本构造规则**：每个 step 1 条。
- **meta.fields（建议最小集）**：`preconditions`, `step_goal`, `high_level_goal`
- **范例（Step 2）**：

```text
Image: <ITEM_DIR>/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg
fields.preconditions = [
  "The person is inside the illuminated kitchen.",
  "The refrigerator contains the required vegetables."
]
Q: Given the overall goal the mission, before starting Retrieve a carrot and a cucumber from the refrigerator, what objective preconditions must be satisfied?
A: Before starting this step, the person must already be inside the illuminated kitchen, and the refrigerator must contain the required vegetables.
```

### Task_10_Step_Execution_Statement

- **字段（JSONPath）**：`steps[i].step_goal`
- **证据来源（两种严格模式二选一，建议用 meta.task_variant 区分）**：
  - `task_variant=text_restatement`：`keyframe_single`（该 step 最早关键帧）
  - `task_variant=video_micro_actions`：`images_uniform_clip`（用该 step 的“执行片段”抽帧；见下）
- **执行片段定义（弱监督、可落地）**：
  - 若存在 `last_frame_segments`：
    - step 1：`segment_start_to_step01.mp4`（脚本会生成该 label）
    - step i（i>1）：`segment_step{i-1:02d}_to_step{i:02d}.mp4`
- **样本构造规则**：
  - text_restatement 模式：答案仅复述 step_goal（不生成额外动作，避免幻觉）。
  - video_micro_actions 模式：允许列出 2–5 个短微动作，但必须可从抽帧中观察到；看不清则写 “not clearly observable”。
- **meta.fields（建议最小集）**：`step_goal`, `task_variant`
- **范例（text_restatement, Step 3）**：

```text
Image: <ITEM_DIR>/03_gather_a_cutting_board_and_a_knife_and_place_them_on_the_countertop/frame_014_ts_46.80s.jpg
fields.step_goal = "Gather a cutting board and a knife and place them on the countertop."
Q: Given the overall goal the mission, for the step Gather a cutting board and a knife and place them on the countertop, what specific execution actions are required?
A: Gather a cutting board and a knife and place them on the countertop.
```

### Task_11_Expected_Physical_Effects

- **字段（JSONPath）**：`steps[i].expected_effects[*]`
- **证据来源**：`keyframe_single`（该 step 最后一张关键帧 `critical_frames[-1]`）
- **样本构造规则**：每个 step 1 条。
- **meta.fields（建议最小集）**：`expected_effects`, `step_goal`, `high_level_goal`
- **范例（Step 4）**：

```text
Image: <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
fields.expected_effects = [
  "cucumber is clean",
  "carrot is clean",
  "cucumber is on countertop",
  "carrot is on countertop",
  "Vegetables are ready for cutting."
]
Q: Given the overall goal the mission, upon completion of Wash the cucumber and carrot under running water and place them on the countertop, what physical effects should be expected for the environment and objects?
A: Upon completion of this step, the cucumber and carrot should be clean, both vegetables should be on the countertop, and the vegetables should be ready for cutting.
```

### Task_12_Inter_Step_Dependency_Analysis

- **字段（JSONPath）**：
  - `steps[i].expected_effects[*]`
  - `steps[i+1].preconditions[*]`
  - （上下文）`steps[i].step_goal`, `steps[i+1].step_goal`, `high_level_goal`
- **证据来源**：`keyframe_single`（建议用 step i 的最后关键帧）
- **样本构造规则**：仅当 `expected_effects` 与下一步 `preconditions` 存在词项重合/包含时生成。
- **meta.fields（建议最小集）**：`step_n_goal`, `step_n_effects`, `step_next_goal`, `step_next_preconditions`
- **范例（Step 1 → Step 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.step_n_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.step_n_effects = ["The person is inside the kitchen.", "The kitchen is illuminated.", "Objects and surfaces ... are visible ..."]
fields.step_next_goal = "Retrieve a carrot and a cucumber from the refrigerator."
fields.step_next_preconditions = ["The person is inside the illuminated kitchen.", "The refrigerator contains the required vegetables."]
Q: Given the overall goal the mission, how does the outcome of Enter the kitchen and turn on the light to illuminate the workspace satisfy the preconditions for Retrieve a carrot and a cucumber from the refrigerator?
A: Entering the kitchen ensures the person is already inside the kitchen, and turning on the light makes the kitchen illuminated, which directly satisfies the next step’s precondition that the person is inside the illuminated kitchen.
```

### Task_13_Next_Action_Prediction（v2 推荐用“下一步目标”版本）

- **字段（JSONPath）**：
  - 输入侧：`steps[i].step_goal`（可选）
  - 标签侧：`steps[i+1].step_goal`
- **证据来源**：
  - 计划版本：`keyframe_single`（step i 最后关键帧）
  - 视觉版本（更难）：用 `Task_24`（prefix）而不是这里
- **样本构造规则**：严格只预测“下一步 step_goal”，不要生成微动作（微动作需单独任务名/字段）。
- **meta.fields（建议最小集）**：`current_step_goal`, `next_step_goal`
- **范例（Step 1 → Step 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.current_step_goal = "Enter the kitchen and turn on the light to illuminate the workspace."
fields.next_step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
Q: Given the overall goal the mission, what is the next planned action?
A: Retrieve a carrot and a cucumber from the refrigerator.
```

### Task_14_Counterfactual_Prediction

- **字段（JSONPath）**：`steps[i].causal_challenge_question`, `steps[i].expected_challenge_outcome`
- **证据来源**：`keyframe_single`（step i 任一关键帧；默认最早关键帧）
- **样本构造规则**：每个 step 0–1 条（字段为空则跳过）。
- **meta.fields（建议最小集）**：`question`, `outcome`, `step_goal`
- **范例（Step 7）**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg
fields.question = "What if the cutting board was slippery on the countertop?"
fields.outcome = "The board would slide during the application of downward cutting force ... The person would need to place a damp cloth under the board to increase friction."
Q: Given the overall goal the mission and the step goal Slice the cucumber into circular pieces on the cutting board, What if the cutting board was slippery on the countertop?
A: The board would slide during the application of downward cutting force. This would make the cutting action unstable and dangerous, increasing the risk of the knife slipping and causing injury. The person would need to place a damp cloth under the board to increase friction.
```

### Task_15_Failure_Recovery_Protocol

- **字段（JSONPath）**：`steps[i].failure_handling.reason`, `steps[i].failure_handling.recovery_strategy`
- **证据来源**：`keyframe_single`（step i 任一关键帧）
- **样本构造规则**：每个 step 0–1 条（failure_handling 缺失则跳过）。
- **meta.fields（建议最小集）**：`reason`, `recovery_strategy`, `step_goal`
- **范例（Step 7）**：

```text
Image: <ITEM_DIR>/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg
fields.reason = "The slices are of inconsistent thickness."
fields.recovery_strategy = "Slow down the cutting motion ... to achieve more uniform slices."
Q: Given the overall goal the mission, in the step Slice the cucumber into circular pieces on the cutting board, why might it fail due to The slices are of inconsistent thickness, and what recovery strategy should be applied?
A: It might fail because the slices can end up uneven, so the person should slow down, pay closer attention to the spacing and knife angle, and then continue cutting to produce more uniform slices.
```

### Task_16_Physical_Feasibility_Verification

- **字段（JSONPath）**：`critical_frames[j].spatial_preconditions[*]`, `critical_frames[j].affordance_preconditions[*]`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每个关键帧最多 1 条。
- **meta.fields（建议最小集）**：`spatial_preconditions`, `affordance_preconditions`, `step_goal`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.spatial_preconditions includes: contact(hand, light_switch)=true; is_mounted_on(light_switch, wall)=true; is_within_reach_of(light_switch, hand)=true
fields.affordance_preconditions includes: light_switch is pressable/toggleable because its rocker design supports actuation
Q: Given the overall goal the mission, is the step Enter the kitchen and turn on the light to illuminate the workspace physically feasible now, based on spatial and affordance evidence?
A: Spatially, the hand is within reach of and in contact with the wall-mounted light_switch, which supports the pressing interaction. In terms of affordances, the light_switch is pressable and toggleable due to its rocker design, so applying a downward force can feasibly change its state.
```

### Task_17_Holistic_Step_Synthesis_Why_How

- **字段（JSONPath）**：
  - Why：`steps[i].rationale`
  - How：从该 step 的某个关键帧取 `causal_chain.*` 或 `affordance_hotspot.mechanism`
- **证据来源**：`keyframe_single`（选择 mechanism 最丰富的关键帧）
- **样本构造规则**：每个 step 最多 1 条（优先选含 `affordance_hotspot.mechanism` 的帧）。
- **meta.fields（建议最小集）**：`rationale`, `mechanism`, `step_goal`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.rationale = "This initial step is essential to enable all subsequent actions ..."
fields.mechanism = "The surface provides a target for applying force ... powering the lights."
Q: Given the overall goal the mission, why is the step Enter the kitchen and turn on the light to illuminate the workspace necessary, and how is it physically achieved?
A: This step is necessary because later navigation and manipulation require sufficient lighting for safety and efficiency. It is achieved by pressing the rocker switch’s pressable surface so the applied force actuates the internal toggle mechanism, completing the circuit and illuminating the room.
```

### Task_18_Visual_Precondition_Check

- **字段（JSONPath）**：`steps[i].preconditions[*]`, `steps[i].step_goal`
- **证据来源**：`video_prefix`（推荐）或 `images_uniform_scene`（前缀帧子集）
  - 前缀片段：`cumulative_last_frame_segments/segment_start_to_step{i-1:02d}_last.mp4`（i=1 时可退化为 sampled_frames 前若干帧）
- **样本构造规则**：
  - 严格允许输出三态：`satisfied / not satisfied / not directly observable`。
  - 对“内部状态/容器内物体”类 precondition，如果证据未展示，必须选择 `not directly observable`。
- **meta.fields（建议最小集）**：`preconditions`, `step_goal`, `prefix_end_step`
- **范例（检查 Step 2 前置条件；前缀到 Step 1 尾帧）**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step01_last.mp4
fields.step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
fields.preconditions = ["The person is inside the illuminated kitchen.", "The refrigerator contains the required vegetables."]
Q: Before starting Retrieve a carrot and a cucumber from the refrigerator, which listed preconditions are satisfied based on the provided video prefix, and which are not directly observable?
A: The prefix supports that the person is inside an illuminated kitchen. Whether the refrigerator contains the required vegetables is not directly observable from this prefix unless the refrigerator interior or drawer contents are clearly shown.
```

### Task_19_Visual_Effect_Check

- **字段（JSONPath）**：`steps[i].expected_effects[*]`, `steps[i].step_goal`
- **证据来源**：
  - 主证据：step i 的最后关键帧 `critical_frames[-1].keyframe_image_path`
  - 可选：step i 的执行片段抽帧（见 Task_10 的“执行片段定义”）
- **样本构造规则**：对每条 expected_effect，回答必须明确“可见支持/不可见/不确定”。
- **meta.fields（建议最小集）**：`expected_effects`, `step_goal`, `keyframe_path`
- **范例（Step 4）**：

```text
Image (step-end keyframe): <ITEM_DIR>/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg
fields.expected_effects = ["cucumber is clean", "carrot is clean", "cucumber is on countertop", "carrot is on countertop", "Vegetables are ready for cutting."]
Q: Based on the provided keyframe, which expected effects of Wash the cucumber and carrot under running water and place them on the countertop are supported by visual evidence, and which are uncertain?
A: The keyframe can support whether the cucumber and carrot are on the countertop if their placement is visible. Claims like the vegetables being clean or ready for cutting may be partially supported but can be uncertain if cleanliness is not directly visible.
```

### Task_20_Step_Boundary_Localization

- **字段（JSONPath）**：`steps[i].step_goal`, `steps[i+1].step_goal`
- **证据来源**：`video_clip` = `last_frame_segments/segment_step{i:02d}_to_step{i+1:02d}.mp4`（或其抽帧）
- **样本构造规则**：
  - 让模型描述“过渡现象”，而不是输出时间戳（避免标注噪声）。
  - 回答必须同时引用 step i 与 step i+1 的语义（例如“从完成 X 到开始 Y 的转折”）。
- **meta.fields（建议最小集）**：`prev_step_goal`, `next_step_goal`, `segment_label`
- **范例（Step 2 → Step 3）**：

```text
Video clip: <ITEM_DIR>/last_frame_segments/segment_step02_to_step03.mp4
fields.prev_step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
fields.next_step_goal = "Gather a cutting board and a knife and place them on the countertop."
Q: In this transition clip, what observable change marks the shift from finishing the refrigerator retrieval step to starting the tool-gathering step?
A: The shift is marked when the person stops interacting with the refrigerator and begins handling the cutting board and knife, indicating the retrieval phase has ended and the countertop tool setup has started.
```

### Task_21_Keyframe_Justification

- **字段（JSONPath）**：
  - `critical_frames[j].action_description`
  - `critical_frames[j].state_change_description`
  - （可选）`spatial_preconditions`, `affordance_hotspot`
- **证据来源**：`keyframe_single`
- **样本构造规则**：每个关键帧最多 1 条。
- **meta.fields（建议最小集）**：`action_description`, `state_change_description`, `frame_index`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.action_description = "... pressing a rocker-style light switch."
fields.state_change_description = "... ambient light level increases ..."
Q: Why is this keyframe a critical moment for the step, and what decisive state change does it capture?
A: This keyframe is critical because it captures the actuation of the light switch, which is the decisive interaction in the step. It also captures the state transition where pressing the switch causes the kitchen to change from dark to illuminated.
```

### Task_22_Plan_Execution_Alignment

- **字段（JSONPath）**：`steps[i].step_goal`（可选 `expected_effects`）
- **证据来源**：`video_clip`/`images_uniform_clip`（使用“执行片段定义”，让 clip 尽量对齐某个 step）
- **样本构造规则**：
  - 输出三分类：`match / partial match / mismatch`。
  - **强约束**：回答必须给出“可见证据”或明确“不清晰/不可判断”。
- **meta.fields（建议最小集）**：`step_goal`, `label`, `segment_label`, `neg_sample`
- **范例（正样本：Step 4 对齐其执行片段）**：

```text
Video clip: <ITEM_DIR>/last_frame_segments/segment_step03_to_step04.mp4
fields.step_goal = "Wash the cucumber and carrot under running water and place them on the countertop."
label = match
Q: Does the clip align with the step goal Wash the cucumber and carrot under running water and place them on the countertop?
A: Match. The clip shows the vegetables being handled at the sink under running water and then being moved toward a countertop, which corresponds to washing and placing them.
```

### Task_23_Goal_Recognition_From_Prefix

- **字段（JSONPath）**：`high_level_goal`
- **证据来源**：`video_prefix`（前缀到 step i 尾帧）
- **样本构造规则**：
  - i 从小到大做 curriculum（前缀越短越难）。
  - 不把 `high_level_goal` 放进输入字段（避免泄漏），只作为 label。
- **meta.fields（建议最小集）**：`high_level_goal`, `prefix_end_step`
- **范例（前缀到 Step 2）**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
label.high_level_goal = "Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board."
Q: Based only on this video prefix, what is the most appropriate high-level goal?
A: Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board.
```

### Task_24_Next_Step_Goal_Prediction_From_Prefix

- **字段（JSONPath）**：标签为 `steps[i+1].step_goal`；可选输入 `steps[i].step_goal`
- **证据来源**：`video_prefix`（前缀到 step i 尾帧）
- **样本构造规则**：严格只预测下一步 step_goal，不能输出其它步骤文本。
- **meta.fields（建议最小集）**：`current_step_goal`, `next_step_goal`, `prefix_end_step`
- **范例（前缀到 Step 2，预测 Step 3）**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step02_last.mp4
fields.current_step_goal = "Retrieve a carrot and a cucumber from the refrigerator."
label.next_step_goal = "Gather a cutting board and a knife and place them on the countertop."
Q: Given this progress so far, what is the next step goal?
A: Gather a cutting board and a knife and place them on the countertop.
```

### Task_25_Progress_Summary_From_Prefix

- **字段（JSONPath）**：用于评测的弱监督参考：`steps[0..i].step_goal`（不要塞进 prompt）
- **证据来源**：`video_prefix`
- **样本构造规则**：
  - 输出必须是自然段落，强调“到目前为止完成了什么 + 当前状态”。
  - 不允许凭空列出未观察到的工具或对象。
- **meta.fields（建议最小集）**：`prefix_end_step`, `completed_step_goals`（仅写入 meta，不写入 prompt）
- **范例（前缀到 Step 4）**：

```text
Video prefix: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step04_last.mp4
meta.fields.completed_step_goals = [
  "Enter the kitchen and turn on the light to illuminate the workspace.",
  "Retrieve a carrot and a cucumber from the refrigerator.",
  "Gather a cutting board and a knife and place them on the countertop.",
  "Wash the cucumber and carrot under running water and place them on the countertop."
]
Q: Summarize what has been accomplished so far in this video prefix and describe the current situation.
A: So far, the workspace has been lit, vegetables have been retrieved from the refrigerator, basic tools have been gathered, and the vegetables have been washed and moved into position for preparation. The scene is now set up for the next stage of cutting and cooking.
```

### Task_26_Temporal_Order_Check

- **字段（JSONPath）**：从两个关键帧抽取：`action_description`/`state_change_description` + 对应 `frame_index` 或 `ts` 用作标签依据
- **证据来源（严格优先级）**：
  1) `video_prefix` = `cumulative_last_frame_segments/segment_start_to_stepXX_last.mp4`（若已生成）
  2) 若无 `video_prefix`：用 `images_uniform_scene`（优先 sampled_frames，否则关键帧代理）
- **样本构造规则**：
  - 题目只给两段描述（A/B），要求判断哪一个先发生。
  - 标签来自 `frame_index` 或关键帧文件名的 `ts_XXs`。
- **meta.fields（建议最小集）**：`event_a`, `event_b`, `label`（A earlier / B earlier）
- **范例（Step 1 frame 2 vs Step 4 frame 20）**：

```text
Evidence: <ITEM_DIR>/cumulative_last_frame_segments/segment_start_to_step04_last.mp4
event_a = "A person's right hand ... pressing a rocker-style light switch."
event_b = "The person holds a cucumber ... under a stream of water ... rubbing its surface to clean it."
label = A earlier
Q: Which event happened earlier in the video, A or B?
A: Event A happened earlier.
```

### Task_27_Visual_Spatial_Relation_Check

- **字段（JSONPath）**：`spatial_preconditions[k].relation/objects/truth`
- **证据来源**：`keyframe_single`
- **样本构造规则**：
  - 把 `truth` 作为分类标签（强监督），这是 v2 中最“高质量可控”的视觉判别任务之一。
  - 如果构造弱负样本（改 objects 或反转 truth），必须写入 `meta.neg_sample=true`。
- **meta.fields（建议最小集）**：`relation`, `objects`, `truth`, `neg_sample`
- **范例（Step 1, frame 2）**：

```text
Image: <ITEM_DIR>/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg
fields.relation = "contact"
fields.objects = ["hand","light_switch"]
label.truth = true
Q: In this image, is the hand in contact with the light_switch?
A: Yes.
```

---

## 10. 当前无约束 QA 示例（逐任务，完全自由文本）

本节给出“**不引入受控标签输出**（例如不要求 Yes/No/JSON labels）”时，每个任务在当前风格下的**具体问答范例**。

重要说明：

- 下面示例的字段取自 `causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json`。
- 关键帧图片路径为 JSON 中记录的 `keyframe_image_path`（绝对路径）。
- 对需要 `sampled_frames/`、`last_frame_segments/`、`cumulative_last_frame_segments/` 的任务：如果你的 item 中尚未生成这些产物，可按本文档第 1 节与第 9 节的脚本/规则生成；示例中仍按标准路径约定书写。

为保持示例可复用，统一定义：

- `SOURCE_JSON = causal_spafa_plan_dataset_long/P01_01_part1/causal_plan_with_keyframes.json`
- `HIGH_LEVEL_GOAL = "Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board."`

### Task_01_Macro_Anchor_Extraction（自由文本示例）

- Evidence（`images_uniform_scene`）：
  - 若无 `sampled_frames/`：可用每步最早关键帧做代理，例如：
    - `/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
    - `/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg`
    - `/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/03_gather_a_cutting_board_and_a_knife_and_place_them_on_the_countertop/frame_014_ts_46.80s.jpg`
    - `/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg`
- Q: Which stable objects are the task-relevant anchors for planning in this scene?
- A: The task-relevant anchors include the light switch, refrigerator, vegetables such as the cucumber and carrot, the cutting board and knife, and the sink and faucet used for washing.

### Task_02_Transient_Geometric_Verification（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: What is the precise spatial relationship between hand and light_switch in this frame?
- A: In this frame, the hand is in contact with the light_switch.

### Task_03_Micro_Affordance_Visual_Semantics（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Which specific region affords pressable_surface, and how does it visually appear and physically function?
- A: The raised rocker surface on the light switch is the pressable region; it provides a clear finger-contact area so applied force transfers to the internal toggle mechanism that completes the circuit and turns on the lights.

### Task_04_Entity_Role_Identification（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg`
- Q: In the step Retrieve a carrot and a cucumber from the refrigerator, which items function as tools, and which are the materials being acted upon?
- A: The refrigerator functions as the tool or container being accessed, while the cucumber and carrot are the materials being grasped and removed.

### Task_05_State_Evolution_Description（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_020_ts_68.39s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, what ongoing action is occurring, and what immediate state change does it cause?
- A: The person is holding and rubbing the cucumber under running water, which immediately causes the cucumber’s surface to be cleaned as water and friction remove contaminants.

### Task_06_Holistic_Causal_Chain_Analysis（自由文本示例，双段）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg`
- Q: Could you explain how knife_blade is applying downward force cucumber in this keyframe, focusing on the spatial setup, the affordance-level mechanism, and the immediate effects?
- A: The knife blade is positioned against the stabilized cucumber while the left hand holds the cucumber steady, creating the spatial setup needed for a controlled cutting interaction.

The blade’s cutting edge concentrates the applied force to exceed the cucumber’s shear strength, which causes the cucumber to fracture and separate. As an immediate effect, a new circular slice is created on the cutting board.

### Task_07_Scene_Goal_Derivation（自由文本示例，verbatim）

- Evidence（`images_uniform_scene`）：同 Task_01
- Q: Given the current scene, what is the appropriate high-level goal?
- A: Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board.

### Task_08_Strategic_Rationale_Justification（自由文本示例，verbatim）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, why is the step Enter the kitchen and turn on the light to illuminate the workspace strategically necessary?
- A: This initial step is essential to enable all subsequent actions. Navigating the environment, locating items, and performing tasks like chopping require adequate lighting for safety and efficiency.

### Task_09_Precondition_Statement（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/02_retrieve_a_carrot_and_a_cucumber_from_the_refrigerator/frame_008_ts_25.19s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, before starting Retrieve a carrot and a cucumber from the refrigerator, what objective preconditions must be satisfied?
- A: Before starting this step, the person must already be inside the illuminated kitchen, and the refrigerator must contain the required vegetables.

### Task_10_Step_Execution_Statement（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/03_gather_a_cutting_board_and_a_knife_and_place_them_on_the_countertop/frame_014_ts_46.80s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, for the step Gather a cutting board and a knife and place them on the countertop, what specific execution actions are required?
- A: The person needs to retrieve a cutting board and a knife and then place both items onto the countertop to set up the workspace for cutting.

### Task_11_Expected_Physical_Effects（自由文本示例）

- Evidence（`keyframe_single`，step 尾帧）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, upon completion of Wash the cucumber and carrot under running water and place them on the countertop, what physical effects should be expected for the environment and objects?
- A: Upon completion of this step, the cucumber and carrot should be clean, both vegetables should be on the countertop, and the vegetables should be ready for cutting.

### Task_12_Inter_Step_Dependency_Analysis（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, how does the outcome of Enter the kitchen and turn on the light to illuminate the workspace satisfy the preconditions for Retrieve a carrot and a cucumber from the refrigerator?
- A: Entering the kitchen ensures the person is already inside the kitchen, and turning on the light makes the kitchen illuminated, which directly satisfies the next step’s precondition that the person is inside the illuminated kitchen.

### Task_13_Next_Action_Prediction（自由文本示例：下一步目标）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, what is the next planned action?
- A: Retrieve a carrot and a cucumber from the refrigerator.

### Task_14_Counterfactual_Prediction（自由文本示例，verbatim）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_032_ts_111.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board and the step goal Slice the cucumber into circular pieces on the cutting board, What if the cutting board was slippery on the countertop?
- A: The board would slide during the application of downward cutting force. This would make the cutting action unstable and dangerous, increasing the risk of the knife slipping and causing injury. The person would need to place a damp cloth under the board to increase friction.

### Task_15_Failure_Recovery_Protocol（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/07_slice_the_cucumber_into_circular_pieces_on_the_cutting_board/frame_039_ts_136.79s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, in the step Slice the cucumber into circular pieces on the cutting board, why might it fail due to The slices are of inconsistent thickness, and what recovery strategy should be applied?
- A: It might fail because the slices can end up uneven, so the person should slow down, pay closer attention to spacing and knife angle, and then continue cutting to produce more uniform slices.

### Task_16_Physical_Feasibility_Verification（自由文本示例，双段）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, is the step Enter the kitchen and turn on the light to illuminate the workspace physically feasible now, based on spatial and affordance evidence?
- A: Spatially, the hand is within reach of and in contact with the wall-mounted light_switch, which supports the pressing interaction.

- In terms of affordances, the light_switch is pressable and toggleable due to its rocker design, so applying a downward force can feasibly change its state.

### Task_17_Holistic_Step_Synthesis_Why_How（自由文本示例，双段）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Given the overall goal Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board, why is the step Enter the kitchen and turn on the light to illuminate the workspace necessary, and how is it physically achieved?
- A: This step is necessary because later navigation and manipulation require sufficient lighting for safety and efficiency.

- It is achieved by pressing the rocker switch’s pressable surface so the applied force actuates the internal toggle mechanism, completing the circuit and illuminating the room.

### Task_18_Visual_Precondition_Check（自由文本示例）

- Evidence（`video_prefix`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/cumulative_last_frame_segments/segment_start_to_step01_last.mp4`
- Q: Before starting Retrieve a carrot and a cucumber from the refrigerator, which listed preconditions are supported by the provided video prefix, and which ones are not directly observable?
- A: The prefix supports that the person is inside an illuminated kitchen. Whether the refrigerator contains the required vegetables is not directly observable from this prefix unless the interior contents are clearly shown.

### Task_19_Visual_Effect_Check（自由文本示例）

- Evidence（`keyframe_single`，step 尾帧）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg`
- Q: Based on the provided keyframe, which expected effects of Wash the cucumber and carrot under running water and place them on the countertop are supported by visual evidence, and which are uncertain?
- A: The keyframe supports that the cucumber and carrot are on the countertop. Whether they are fully clean or ready for cutting can be uncertain if cleanliness and readiness are not directly visible from the image.

### Task_20_Step_Boundary_Localization（自由文本示例）

- Evidence（`video_clip`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/last_frame_segments/segment_step02_to_step03.mp4`
- Q: In this transition clip, what observable change marks the shift from finishing the refrigerator retrieval step to starting the tool-gathering step?
- A: The shift is marked when interaction transitions away from accessing or leaving the refrigerator and toward handling the cutting board and knife on the countertop, indicating the retrieval phase has ended and the workspace setup has begun.

### Task_21_Keyframe_Justification（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/01_enter_the_kitchen_and_turn_on_the_light_to_illuminate_the_workspace/frame_002_ts_3.59s.jpg`
- Q: Why is this keyframe a critical moment for the step, and what decisive state change does it capture?
- A: This keyframe is critical because it captures the actuation of the light switch, which is the decisive interaction in the step, and it captures the state transition where pressing the switch causes the kitchen to change from dark to illuminated.

### Task_22_Plan_Execution_Alignment（自由文本示例）

- Evidence（`video_clip`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/last_frame_segments/segment_step03_to_step04.mp4`
- Q: Does the clip align with the step goal Wash the cucumber and carrot under running water and place them on the countertop?
- A: The clip appears to align if it shows the vegetables being handled at the sink under running water and then being placed onto a countertop; if those actions are not clearly visible, the alignment would be uncertain.

### Task_23_Goal_Recognition_From_Prefix（自由文本示例）

- Evidence（`video_prefix`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/cumulative_last_frame_segments/segment_start_to_step02_last.mp4`
- Q: Based only on this video prefix, what is the most appropriate high-level goal?
- A: Prepare for cooking by turning on the light, gathering vegetables and tools from around the kitchen, washing the vegetables, and chopping them on a cutting board.

### Task_24_Next_Step_Goal_Prediction_From_Prefix（自由文本示例）

- Evidence（`video_prefix`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/cumulative_last_frame_segments/segment_start_to_step02_last.mp4`
- Q: Given this progress so far, what is the next step goal?
- A: Gather a cutting board and a knife and place them on the countertop.

### Task_25_Progress_Summary_From_Prefix（自由文本示例）

- Evidence（`video_prefix`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/cumulative_last_frame_segments/segment_start_to_step04_last.mp4`
- Q: Summarize what has been accomplished so far in this video prefix and describe the current situation.
- A: So far, the workspace has been lit, vegetables have been retrieved, and the basic tools for preparation have been gathered, with the vegetables washed and moved into position. The scene is now prepared for continued cutting and cooking steps.

### Task_26_Temporal_Order_Check（自由文本示例）

- Evidence（`video_prefix`，需生成）：`causal_spafa_plan_dataset_long/P01_01_part1/cumulative_last_frame_segments/segment_start_to_step04_last.mp4`
- Q: Which event happened earlier in the video: (A) pressing the light switch or (B) washing the cucumber under running water?
- A: Pressing the light switch happened earlier.

### Task_27_Visual_Spatial_Relation_Check（自由文本示例）

- Evidence（`keyframe_single`）：`/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long/P01_01_part1/04_wash_the_cucumber_and_carrot_under_running_water_and_place_them_on_the_countertop/frame_025_ts_86.39s.jpg`
- Q: In this image, is the carrot on the countertop?
- A: Yes, the carrot is on the countertop in this frame.
