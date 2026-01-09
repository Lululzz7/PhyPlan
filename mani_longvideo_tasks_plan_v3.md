# mani_longvideo 多模态任务规划 v3（Best Edition）

目标：在 `mani_longvideo.py` 生成的**单视频输出目录（item）**内，基于“计划 JSON + 关键帧 + 全局均匀抽帧 + 片段/前缀视频”构建**可落地批量生成**、**训练信号可控**、**视觉不可被轻易绕过**的多模态 QA/监督数据。

v3 相比 v2 的核心变化：

- 将任务体系拆为两条数据线：
  - **MM-Core（多模态核心线）**：必须依赖图像/视频证据才能完成，作为训练主数据。
  - **Text-Plan（规划文本线，可选）**：偏解释/策略/反事实的纯文本任务，默认不进入多模态主训练集，避免“视觉退化为可选输入”。
- 大幅减少“字段复述型”任务比例，优先引入：
  - **可判别/可校验**（Yes/No/Uncertain、多选、检索、排序）
  - **可构造难负样本**（hard negatives）
  - **对齐学习**（step ↔ frame/clip）
- 明确新增一种关键证据形态：**Step 对齐片段（step-aligned clip）**，用于稳定支撑对齐/核验类任务。

---

## 1. 输入产物（item 目录）

在 `causal_spafa_plan_dataset_long/<item_dir>/` 内，默认可用：

- `causal_plan_with_keyframes.json`（计划文本 + 关键帧标注）
- 关键帧图片：`steps[*].critical_frames[*].keyframe_image_path`
- 全局均匀抽帧：`sampled_frames/`（默认 50 帧）
- Step 间尾帧片段：`last_frame_segments/*.mp4`
- 累积前缀视频：`cumulative_last_frame_segments/*.mp4`

v3 推荐额外生成（强烈建议）：

- **Step 对齐片段**：`step_segments/segment_step{XX}.mp4`
  - 语义：尽量覆盖 step 的“开始→结束”（见第 3.2 节的生成规则）。

---

## 2. 样本输出结构（ShareGPT 扩展）

建议统一为：

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
    "task_name": "MM_06_StepGoal_Frame_Retrieval",
    "evidence_type": "images_uniform_clip",
    "evidence_source": "sampled_frames|keyframes|step_segments|last_frame_segments|cumulative_last_frame_segments",
    "evidence_files": ["..."],
    "source_json": ".../causal_plan_with_keyframes.json",
    "item_dir": "...",
    "step_index": 3,
    "frame_index": 20,
    "segment_label": "segment_step03",
    "label": {"...": "..."},
    "neg_sample": false,
    "missing_media": false
  }
}
```

约定：

- `image` 与 `video` 可二选一或同时存在。
- 若下游不支持 `video`，应在生成阶段对 mp4 预抽帧，并把 `evidence_type` 设置为 `images_uniform_clip`。

---

## 3. 证据形态（Evidence Types）与取证规则

### 3.1 Evidence Types（统一枚举）

`meta.evidence_type` ∈

- `keyframe_single`：单张关键帧
- `images_uniform_scene`：全局场景多图（从 `sampled_frames/` 采样 6–10 张）
- `images_uniform_clip`：某个 clip 均匀抽帧 8–16 张
- `video_clip`：mp4 片段（若训练支持视频输入）
- `video_prefix`：累积前缀 mp4

证据来源字段：

- `meta.evidence_source` ∈ `sampled_frames|keyframes|step_segments|last_frame_segments|cumulative_last_frame_segments`
- `meta.evidence_files`：最终喂给模型的图片路径列表或 mp4 路径

### 3.2 Step 对齐片段（step_segments）生成规则（强烈建议）

核心目的：让“对齐/核验/执行描述”任务有更可靠证据，减少 `last_frame_segments` 带来的语义错位。

对每个 step i：

- **优先**用该 step 的关键帧时间范围：
  - `start_ts`：`critical_frames[0]` 的时间戳
  - `end_ts`：`critical_frames[-1]` 的时间戳
  - 可加 padding（例如前后各 0.5–1.0 秒）
- 若关键帧无时间戳但文件名含 `ts_XX.YYs`，从文件名解析。
- 若仍不可得：退化为 `last_frame_segments/segment_step{i}_to_step{i+1}.mp4`（并在 `meta` 标记 `weak_alignment=true`）。

建议写入 manifest：

- `step_segments/segments_manifest.json`：记录 `segment_label, start_ts, end_ts, duration_sec`。

---

## 4. 任务体系 v3

### 4.1 MM-Core（多模态核心线，推荐训练主线）

设计原则：

- **必须看证据**才能稳定回答。
- 优先**判别/多选/检索**，其次才是生成式描述。
- 每个 keyframe/segment 设定“最多采样 K 个任务”的上限（建议 K=2~3），避免同源重复。

#### 任务卡片通用约定（适用于所有 MM_*）

- **输入字段来源约束**：prompt 中允许出现的结构化字段只能来自：
  - `high_level_goal`
  - `steps[i].step_goal`
  - `steps[i].preconditions[*]`
  - `steps[i].expected_effects[*]`
  - （少量任务允许）`critical_frames[j].action_description/state_change_description`
  - 其它 JSON 字段默认只允许写入 `meta`，不得直接出现在 prompt 中，避免标签泄漏。
- **答案格式约束**：MM-Core 优先固定格式（Yes/No/Uncertain、Answer: k、P1/E1 行式），便于自动质检与稳定训练。
- **证据文件排序**：
  - `images_uniform_scene`：按时间从早到晚排列。
  - `images_uniform_clip`：按 clip 时间从早到晚排列。
  - 检索任务候选（4 张/3 段）：按候选序号排列。
- **meta.label**：只放训练所需最小标签（分类/多选/索引/结构化判定），其余字段放 `meta` 的旁路字段（如 `meta.debug`）。

#### MM_01_Visible_Anchors_In_Scene（场景可见锚点抽取）

- **目标**：从全局多图中列出“肉眼可见且任务相关”的锚点对象（工具/材料/关键实体），允许不确定。
- **证据**：`images_uniform_scene`（从 `sampled_frames/` 等距取 6–10 张）
- **输入字段**：仅 `high_level_goal`（可选）
- **输出格式（英文，半结构化）**：
  - 第一行：`Visible anchors:` 逗号分隔对象名
  - 第二行：`Uncertain:` 逗号分隔（可空）
- **meta.label schema**：无（生成式），但建议保留 `meta.label.visible_anchors_count` 作为统计。
- **prompt template（英文）**：

  `Based on the images, list the task-relevant objects that are clearly visible in the scene. If an object is only partially visible or unclear, put it under Uncertain.`

- **minimal example**：

  Q: Based on the images, list the task-relevant objects that are clearly visible in the scene...

  A:

  `Visible anchors: refrigerator, cutting_board, knife, sink, faucet`

  `Uncertain: carrot, cucumber`
- **质检要点**：不得直接复述 JSON 的 tools/materials 作为唯一来源；若对象在多图中不可见，应进 `Uncertain`。

#### MM_02_Spatial_Relation_Check（空间关系真值判别，强监督）

- **目标**：判断关键帧中某空间关系是否成立。
- **证据**：`keyframe_single`
- **标签来源**：`critical_frames[*].spatial_preconditions[*].truth`
- **问句模板（英文）**：
  - `In this image, is <objA> <relation> <objB>? Answer with Yes/No/Uncertain.`
- **答案格式（英文）**：必须以 `Yes.` / `No.` / `Uncertain.` 开头，后接 1 句证据描述。
- **meta.label schema**：
  - `label.truth` ∈ `true|false|uncertain`
  - `meta.fields.relation: str`
  - `meta.fields.objects: [str, str]`
- **minimal example**：

  Q: In this image, is the hand in contact with the light_switch? Answer with Yes/No/Uncertain.

  A: Yes. The fingers are touching the switch surface.
- **负样本（可选）**：
  - 轻度：交换 objects 顺序（若 relation 非对称）
  - 难度：替换为同类物体（knife ↔ spoon）但需保证图中确实存在/不存在
  - 所有合成负样本必须写 `meta.neg_sample=true`

#### MM_03_Affordance_Type_MultiChoice（可供性类型多选）

- **目标**：给定关键帧与对象局部语义，选择最合理的 affordance 类型（避免“纯文本讲机制”的幻觉）。
- **证据**：`keyframe_single`
- **标签来源**：`critical_frames[*].affordance_hotspot.affordance_type`
- **形式**：四选一（A/B/C/D）。
  - 候选集合建议从全体 affordance_type 里挑高频 20 个，再按语义近邻采 hard negatives。
- **答案格式**：`Answer: <A|B|C|D>.` + 1 句理由。
- **meta.label schema**：
  - `label.choice` ∈ `A|B|C|D`
  - `meta.fields.choices: {A: str, B: str, C: str, D: str}`
  - `meta.fields.object_name: str`（若可得）
- **minimal example**：

  Q: Which affordance type best matches the highlighted interaction target?
  A) pressable_surface  B) graspable_handle  C) cuttable_surface  D) pourable_opening

  A: Answer: A. The finger is positioned to press the flat switch surface.

#### MM_04_Action_And_StateChange_Caption（动作与状态变化描述，合并版）

- **目标**：用 2 句描述关键帧：正在发生的动作 + 立即可见的状态变化。
- **证据**：`keyframe_single`
- **标签来源**：`action_description` + `state_change_description`
- **输出格式**：
  - Sentence 1: `Action: ...`
  - Sentence 2: `State change: ...`
- **meta.label schema**：无（生成式），但建议保留 `meta.fields.frame_index` 便于回溯。
- **minimal example**：

  Q: Describe the ongoing action and the immediate visible state change.

  A:
  `Action: The person is rinsing a cucumber under running water.`
  `State change: Water is flowing over the cucumber surface, indicating it is being cleaned.`
- **去幻觉约束**：禁止引入图中不可见的对象；若状态变化不可见，用 `Not clearly visible` 表达。

#### MM_05_BeforeAfter_Visual_Change（同一步骤首尾对比：可见变化）

- **目标**：给同一步骤的“最早关键帧 + 最晚关键帧”，要求总结 2–3 条可见变化。
- **证据**：两张图：`critical_frames[0]` 与 `critical_frames[-1]`
- **输出格式**：
  - `Change 1: ...`
  - `Change 2: ...`
  - `Change 3: ...`（可选）
- **价值**：比单帧 state_change 更稳，迫使模型做视觉对比。
- **meta.label schema**：无（生成式）。
- **minimal example**：

  Q: Compare the first and last images. List 2–3 visible changes.

  A:
  `Change 1: The vegetables move from the sink area to the countertop.`
  `Change 2: The cutting board becomes present in the workspace.`

#### MM_06_StepGoal_Frame_Retrieval（Step 目标→关键帧检索，强监督）

- **目标**：给一个 step_goal 与 4 张候选帧（1 正 3 负），选最匹配。
- **证据**：`image` 列表（4 张）
- **正样本**：来自 step i 的关键帧
- **负样本（hard）**：
  - 优先来自相邻 step（i-1/i+1）的关键帧
  - 其次来自同场景相似动作的其他 item
- **输出格式**：`Answer: <1|2|3|4>.` + 1 句证据理由。
- **meta.label schema**：
  - `label.answer_index` ∈ `1|2|3|4`
  - `meta.fields.step_goal: str`
  - `meta.fields.positive_source: {step_index:int, frame_index:int}`
  - `meta.fields.negative_sources: List[{item_dir:str, step_index:int, frame_index:int}]`
- **minimal example**：

  Q: The step goal is: "Gather a cutting board and a knife and place them on the countertop." Which image best matches this step?
  A: Answer: 2. Image 2 clearly shows the cutting board and knife being placed on the counter.
- **质检**：若 4 张都高度相似导致不可判别，允许 `Answer: Uncertain.` 并丢弃该样本（宁缺毋滥）。

#### MM_07_StepGoal_Clip_Retrieval（Step 目标→片段检索，强监督）

- **目标**：给一个 step_goal 与 3 段候选 clip（或每段抽 8 帧），选最匹配。
- **证据**：优先 `step_segments/segment_stepXX.mp4`；否则 `images_uniform_clip`。
- **输出格式**：同 MM_06。
- **meta.label schema**：
  - `label.answer_index` ∈ `1|2|3`
  - `meta.fields.step_goal: str`
  - `meta.fields.candidate_segments: List[str]`
- **minimal example**：

  Q: The step goal is: "Wash the cucumber and carrot under running water." Which clip best matches?
  A: Answer: 1. Clip 1 shows washing under a running faucet.

#### MM_08_Precondition_Check_Structured（视觉前置条件核验，结构化）

- **目标**：对 step i 的 preconditions 做逐条核验。
- **证据**：
  - 优先 `video_prefix` 到 step i-1 结束；或 `images_uniform_scene`（取前 8–12 张更偏早期）
- **输出格式（英文，固定行格式，利于质检）**：

  `P1: <satisfied|unsatisfied|uncertain> - <brief evidence>`

  `P2: ...`

- **meta.label schema**：
  - `label.statuses: List[satisfied|unsatisfied|uncertain]`（与 preconditions 同长度）
  - `meta.fields.preconditions: List[str]`（仅写 meta，不写 prompt）
- **prompt template（英文）**：

  `Check each precondition based only on the visual evidence. For each line, output: Pk: satisfied|unsatisfied|uncertain - brief evidence.`

- **minimal example**：

  A:
  `P1: satisfied - The countertop and workspace are clearly lit.`
  `P2: uncertain - The knife is not clearly visible in this prefix.`

- **注意**：许多 precondition 本质不可见（如“刀已消毒”），应该大量出现 `uncertain`，这是合理的。

#### MM_09_Effect_Check_Structured（视觉后置效果核验，结构化）

- **目标**：对 step i 的 expected_effects 做逐条核验。
- **证据**：
  - `keyframe_single`（step 尾关键帧） +（可选）`step_segments/segment_stepXX` 的抽帧
- **输出格式**：同 MM_08，用 `E1/E2/...`。
- **meta.label schema**：同 MM_08，把 `P` 改为 `E`。
- **minimal example**：

  A:
  `E1: satisfied - The cucumber and carrot are on the countertop.`
  `E2: uncertain - Cleanliness is not directly observable.`

#### MM_10_Step_Boundary_Keyframe_Selection（Step 边界帧选择 + 解释）

- **目标**：在过渡片段抽帧序列中，选出“最像从 step i 转到 step i+1 的边界帧”。
- **证据**：`images_uniform_clip`（来自 `last_frame_segments/segment_step{i}_to_step{i+1}.mp4`，抽 8–16 帧）
- **输出格式**：
  - `Answer: Frame <k>.`
  - `Reason: ...`（必须同时引用 prev_step_goal 与 next_step_goal 的语义）
- **meta.label schema**：
  - `label.boundary_frame_index` ∈ `[1..N]`（N=抽帧数）
  - `meta.fields.prev_step_goal: str`
  - `meta.fields.next_step_goal: str`
- **minimal example**：

  A: Answer: Frame 6. Reason: The person stops handling the refrigerator and starts picking up the cutting board, marking the transition.

#### MM_11_Plan_Execution_Alignment（计划-执行对齐判别，带不确定项）

- **目标**：判断给定证据是否与某个 step_goal 一致。
- **证据**：优先 `step_segments/segment_stepXX`（否则弱对齐片段并标记）
- **标签**：`match | partial_match | mismatch | uncertain`
- **输出格式**：
  - 第一词必须是：`Match.` / `Partial match.` / `Mismatch.` / `Uncertain.`
  - 后接 1–2句证据解释。
- **负样本**：同 v2，但要求 hard negatives（相邻 step、相似动作）占比 ≥ 50%。
- **meta.label schema**：
  - `label.alignment` ∈ `match|partial_match|mismatch|uncertain`
  - `meta.fields.step_goal: str`
  - `meta.fields.segment_label: str`
  - `meta.fields.weak_alignment: bool`（若使用 last_frame_segments 近似）
- **minimal example**：

  A: Partial match. The clip shows water running and vegetables being handled, but the final placement on the countertop is not clearly shown.

#### MM_12_Temporal_Order_Check_HardPairs（时间顺序判断：难样本对）

- **目标**：给两个事件描述（A/B），判断谁先发生。
- **证据**：优先 `video_prefix`（或全局 `images_uniform_scene`）
- **事件来源**：建议从两个关键帧的 `action_description` 中抽取，但要做“常识捷径”过滤：
  - 过滤掉过于模板化强先验对（如“开灯 vs 切菜”）
  - 多采相邻阶段事件对（如“拿刀 vs 放刀”、“打开冰箱 vs 关闭冰箱”）
- **输出格式**：`Answer: A earlier.` / `Answer: B earlier.` / `Answer: Uncertain.` + 1 句证据。
- **meta.label schema**：
  - `label.order` ∈ `A_earlier|B_earlier|uncertain`
  - `meta.fields.event_a: str`
  - `meta.fields.event_b: str`
  - `meta.fields.event_a_time: float`（可选，仅写 meta）
  - `meta.fields.event_b_time: float`（可选，仅写 meta）
- **minimal example**：

  Q: Which event happened earlier, A or B?
  A: Answer: B earlier. The faucet interaction appears before the vegetables are moved to the cutting board.

---

### 4.2 Text-Plan（规划文本线，可选，不建议混入多模态主线）

这条线的价值是提升“语言规划解释能力”，但会显著提高视觉可绕过风险。
默认：**只在你确实要训练一个“会讲计划”的通用模型时启用**，并与 MM-Core 分开混料。

建议保留的 text-only 任务（命名示例）：

- TP_01_HighLevelGoal_Verbatim（直接输出 `high_level_goal`）
- TP_02_Rationale_Justification（输出 `steps[i].rationale`）
- TP_03_Counterfactual_QA（`causal_challenge_question` ↔ `expected_challenge_outcome`）
- TP_04_Failure_Recovery（失败原因与恢复策略）

启用约束：

- `meta.evidence_type=text_only`
- 输出尽量 `verbatim` 或强约束格式，避免“自由发挥扩写”造成风格漂移。

---

## 5. 采样配比与课程学习（强烈建议）

推荐训练主线（MM-Core）配比（可按 item 统计后再微调）：

- MM_02（关系判别）+ MM_06/07（检索对齐）合计 35–50%
- MM_08/09（前置/效果核验）15–25%
- MM_11（对齐分类）10–20%
- MM_04/05（描述/对比）10–15%
- MM_10（边界选择）5–10%
- MM_12（时序判断）5–10%

课程学习建议：

- 先训练单帧强监督（MM_02/MM_03/MM_06）→ 再加入 clip/前缀任务（MM_07/MM_10/MM_11/MM_12）
- 对 prefix 任务，限制最大时长（例如 cap 到最近 10–20 秒的窗口）以防训练成本爆炸与重复。

---

## 6. 质量约束与自动检查（必须做）

### 6.1 媒体存在性

- `image/video` 路径不存在：直接跳过，或写 `missing_media=true` 并丢弃于训练集。

### 6.2 可判别性过滤（降低噪声）

- 检索任务（MM_06/MM_07）：若人工/规则判断候选过于相似或不可区分，丢弃。
- 对齐任务（MM_11）：若证据弱对齐（只有 `last_frame_segments`）可保留但打 `weak_alignment=true`，训练时降权。

### 6.3 防“视觉绕过”

- MM-Core 的 prompt 不得直接泄漏标签字段（例如不要把 `truth`、`affordance_type`、正确帧索引写进输入）。
- 任何从 JSON 直接读出即可回答的任务，必须移到 Text-Plan 线或改成判别/检索形式。

### 6.4 语言与格式

- 问句与答句统一英文。
- 答案格式要固定（Yes/No/Uncertain；Answer: X；P1/E1 行格式），便于自动质检与稳定训练。

---

## 7. 与 v2 的映射（便于迁移）

- v2 Task_27 → MM_02（核心保留，扩大占比）
- v2 Task_03 → MM_03（改成多选，减少幻觉）
- v2 Task_05 + Task_21 → MM_04（合并）
- v2 Task_18/19 → MM_08/MM_09（改成结构化核验）
- v2 Task_20 → MM_10（改成选帧任务）
- v2 Task_22 → MM_11（加入 uncertain + hard negatives + step_segments 优先）
- v2 Task_26 → MM_12（强化 hard pairs，避免常识捷径）
- v2 Task_07/08/09/14/15/17 → TP_*（默认不混入多模态主线）

---

## 8. 实施建议（落地优先级）

建议按以下顺序实现生成器：

1) 先做 `MM_02/MM_06/MM_04`（只依赖 keyframes + JSON，最稳、最强监督）
2) 再做 `MM_08/MM_09`（需要 prefix/step_segments 更好）
3) 最后做 `MM_10/MM_11/MM_07/MM_12`（依赖 clip，且需要负样本策略与过滤器）
