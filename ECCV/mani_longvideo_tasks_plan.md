# mani_longvideo.py 字段→任务分化规划（对齐 PhyPlan/PhyPlan-API 任务体系）

本文档的目标：参考 `generate_phyplan_api.py`/`generate_phyplan_api_typea.py` 的“多任务 QA”划分方式，基于 `mani_longvideo.py` **最终 JSON（仅按你给定的 schema 字段）**，系统性地拆解可生成的任务集合，并为每个任务列出：

- **任务定义（做什么）**
- **数据源头字段（JSONPath 级别）**
- **可选派生字段（由源字段计算得到，但不引入新信息）**
- **示例：从示例 JSON 片段抽取到的字段 + 一个问答/监督样本样例**

注意：
- 本文档**不依赖** `causal_spafa_plan_dataset_long/` 中的任何实际样本。
- 示例均为**虚构**，只用于说明字段映射与样本形态。

---

## 0. 输入数据 Schema（你提供的字段）

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
- `frame_index: int`（1-based）
- `action_description: str`
- `state_change_description: str`
- `spatial_preconditions: List[{relation:str, objects:[str], truth:bool}]`
- `affordance_preconditions: List[{object_name:str, affordance_types:[str], reasons:str}]`
- `causal_chain: {agent, action, patient, causal_effect_on_patient, causal_effect_on_environment}`
- `affordance_hotspot: {description, affordance_type, mechanism}`

---

## 1. 示例 JSON 片段（虚构，仅用于举例）

下面的片段将被后续每个任务的示例引用：

```json
{
  "high_level_goal": "Prepare a simple salad by washing and slicing a cucumber.",
  "steps": [
    {
      "step_id": 1,
      "step_goal": "Retrieve a cucumber and bring it to the sink.",
      "rationale": "The cucumber must be accessible at the sink before it can be washed.",
      "preconditions": [
        "The cucumber is present in the environment.",
        "The agent has a free hand to grasp the cucumber."
      ],
      "expected_effects": [
        "The cucumber is held in the agent's hand.",
        "The cucumber is positioned over the sink basin."
      ],
      "tool_and_material_usage": {
        "tools": ["sink"],
        "materials": ["cucumber"]
      },
      "causal_challenge_question": "What if the cucumber is wet and slippery before grasping?",
      "expected_challenge_outcome": "The grasp may fail due to reduced friction, requiring a different grip or slower motion.",
      "failure_handling": {
        "reason": "The cucumber slips from the hand.",
        "recovery_strategy": "Reposition the hand to increase contact area and re-grasp the cucumber more firmly."
      },
      "critical_frames": [
        {
          "frame_index": 7,
          "action_description": "The right hand grasps the cucumber and lifts it off the counter.",
          "state_change_description": "The cucumber transitions from resting on the counter to being supported by the hand.",
          "spatial_preconditions": [
            {"relation": "hand is in contact with cucumber", "objects": ["hand", "cucumber"], "truth": true},
            {"relation": "cucumber is on counter", "objects": ["cucumber", "counter"], "truth": true}
          ],
          "affordance_preconditions": [
            {"object_name": "cucumber", "affordance_types": ["graspable"], "reasons": "Its elongated shape allows a power grasp."}
          ],
          "causal_chain": {
            "agent": "hand",
            "action": "applies grasping force",
            "patient": "cucumber",
            "causal_effect_on_patient": "the cucumber is constrained and lifted against gravity",
            "causal_effect_on_environment": "the cucumber leaves the supporting surface"
          },
          "affordance_hotspot": {
            "description": "the mid-section of the cucumber's surface",
            "affordance_type": "graspable_surface",
            "mechanism": "normal force and friction at the contact patch prevent slip during lifting"
          }
        }
      ]
    },
    {
      "step_id": 2,
      "step_goal": "Wash the cucumber under running water.",
      "rationale": "Removing surface contaminants improves food safety before slicing.",
      "preconditions": [
        "The cucumber is at the sink.",
        "Water can be dispensed from the faucet."
      ],
      "expected_effects": [
        "The cucumber's exterior is rinsed.",
        "The cucumber is still intact and ready to be sliced."
      ],
      "tool_and_material_usage": {
        "tools": ["faucet", "sink"],
        "materials": ["cucumber", "water"]
      },
      "causal_challenge_question": "What if the water pressure is too high?",
      "expected_challenge_outcome": "The cucumber may be pushed out of the hand or water may splash, requiring reduced flow.",
      "failure_handling": {
        "reason": "Water splashes and the cucumber becomes harder to hold.",
        "recovery_strategy": "Reduce flow rate and stabilize the cucumber against the sink to increase support."
      },
      "critical_frames": [
        {
          "frame_index": 18,
          "action_description": "The hand holds the cucumber under the faucet while water flows over it.",
          "state_change_description": "Water wets the cucumber's surface and carries away visible residue.",
          "spatial_preconditions": [
            {"relation": "cucumber is under faucet", "objects": ["cucumber", "faucet"], "truth": true}
          ],
          "affordance_preconditions": [
            {"object_name": "faucet", "affordance_types": ["dispensable_water"], "reasons": "It can release water when actuated."}
          ],
          "causal_chain": {
            "agent": "water_stream",
            "action": "impinges on and rinses",
            "patient": "cucumber_surface",
            "causal_effect_on_patient": "contaminants are dislodged and transported",
            "causal_effect_on_environment": "water drains into the sink"
          },
          "affordance_hotspot": {
            "description": "the faucet outlet",
            "affordance_type": "water_outlet",
            "mechanism": "gravity-driven flow provides shear forces that remove residue"
          }
        }
      ]
    }
  ]
}
```

---

## 2. 任务体系总览（对齐 generate_phyplan_api 的 17 类）

为了与 `generate_phyplan_api.py` 的任务集合一致，这里仍采用 `Task_01`~`Task_17` 的编号方式。

在 **仅有上述字段** 的前提下，我们将任务分为 4 个粒度：

- **场景级**：围绕 `high_level_goal`（可选融合所有 step 文本）
- **步级**：围绕 `steps[i]` 的宏观字段（goal/rationale/preconditions/effects/tools/materials/challenge/failure）
- **帧级**：围绕 `steps[i].critical_frames[j]` 的微观字段（action/state/spatial/affordance/causal/hotspot）
- **跨步级**：依赖相邻 steps 的信息（依赖关系）

---

## 3. 17 类任务：字段来源 + 示例

下面每个任务都给出：
- **数据源头字段（JSONPath）**：严格从 schema 抽取
- **样本形态**：建议的输入/输出（可做 QA，也可做监督学习）
- **示例**：从“示例 JSON 片段”中抽取出的字段

### Task_01_Macro_Anchor_Extraction（场景锚点/关键可交互对象）

**任务定义**：从整段计划中抽取“规划锚点对象”（可交互对象/关键实体）。

**数据源头字段（JSONPath）**（可并集去重）：
- `steps[*].tool_and_material_usage.tools[*]`
- `steps[*].tool_and_material_usage.materials[*]`
- `steps[*].critical_frames[*].spatial_preconditions[*].objects[*]`
- `steps[*].critical_frames[*].affordance_preconditions[*].object_name`
- `steps[*].critical_frames[*].causal_chain.agent`
- `steps[*].critical_frames[*].causal_chain.patient`
- `steps[*].critical_frames[*].affordance_hotspot.affordance_type`（也可作为“对象属性/可供性标签”）

**可选派生字段（不引入新信息）**：
- `key_objects_for_planning = unique(flatten(...))`

**示例（从示例 JSON 抽取）**：
- 抽取到的锚点（示意）：`["sink", "cucumber", "counter", "faucet", "water"]`
- QA 示例：
  - Q：Based on the scene, which objects are the key interactable anchors for planning?
  - A：The key anchors include the sink, the faucet, the counter, and the cucumber, as well as water as the relevant material during rinsing.

---

### Task_02_Transient_Geometric_Verification（瞬时空间关系验证）

**任务定义**：对某一关键帧中的空间关系进行“精确关系陈述/验证”。

**数据源头字段（JSONPath）**：
- `steps[i].critical_frames[j].spatial_preconditions[k].relation`
- `steps[i].critical_frames[j].spatial_preconditions[k].objects[0..]`
- `steps[i].critical_frames[j].spatial_preconditions[k].truth`
- （定位）`steps[i].critical_frames[j].frame_index`

**样本形态**：
- 输入：`objects + relation`（可选加 `action_description` 作为上下文）
- 输出：关系表述或 `truth`（二分类/三分类）

**示例**：
- 字段：`relation="hand is in contact with cucumber"`, `objects=["hand","cucumber"]`, `truth=true`
- QA 示例：
  - Q：What is the precise spatial relationship between hand and cucumber in this frame?
  - A：At this keyframe, the hand is in direct contact with the cucumber.

---

### Task_03_Micro_Affordance_Visual_Semantics（微观可供性热点语义）

**任务定义**：描述关键帧中“可供性热点区域 + 功能类别 + 物理机制”。

**数据源头字段（JSONPath）**：
- `steps[i].critical_frames[j].affordance_hotspot.description`
- `steps[i].critical_frames[j].affordance_hotspot.affordance_type`
- `steps[i].critical_frames[j].affordance_hotspot.mechanism`
- （定位）`frame_index`

**示例**：
- 字段：`affordance_type="graspable_surface"`, `description="the mid-section..."`, `mechanism="normal force and friction..."`
- QA 示例：
  - Q：Which specific region affords graspable_surface, and how does it function?
  - A：The graspable region is the mid-section of the cucumber’s surface; contact normal forces and friction at that patch prevent slip during lifting.

---

### Task_04_Entity_Role_Identification（工具/材料角色区分）

**任务定义**：区分“主动工具”与“被作用材料”。

**数据源头字段（JSONPath）**：
- `steps[i].tool_and_material_usage.tools[*]`
- `steps[i].tool_and_material_usage.materials[*]`
- （可选上下文）`steps[i].step_goal`

**示例**：
- 字段：`tools=["sink"]`, `materials=["cucumber"]`
- QA 示例：
  - Q：In this step, which items function as tools, and which are the materials being acted upon?
  - A：The sink functions as the tool/environmental support, while the cucumber is the material being acted upon.

---

### Task_05_State_Evolution_Description（动作-状态变化描述）

**任务定义**：把“正在发生的动作”与“紧随其后的状态变化”组织为因果句。

**数据源头字段（JSONPath）**：
- `steps[i].critical_frames[j].action_description`
- `steps[i].critical_frames[j].state_change_description`
- （定位）`frame_index`

**示例**：
- 字段：`action_description="The right hand grasps..."`, `state_change_description="The cucumber transitions..."`
- QA 示例：
  - Q：Describe the ongoing action and the immediate resulting state change.
  - A：The right hand grasps the cucumber and lifts it off the counter, which causes the cucumber to transition from being supported by the counter to being supported by the hand.

---

### Task_06_Holistic_Causal_Chain_Analysis（因果链分析：谁对谁做了什么→产生什么后果）

**任务定义**：从因果链字段生成“物理因果解释”（强调 agent/action/patient、空间条件、可供性机制、效果）。

**数据源头字段（JSONPath）**：
- `steps[i].critical_frames[j].causal_chain.agent`
- `steps[i].critical_frames[j].causal_chain.action`
- `steps[i].critical_frames[j].causal_chain.patient`
- `steps[i].critical_frames[j].causal_chain.causal_effect_on_patient`
- `steps[i].critical_frames[j].causal_chain.causal_effect_on_environment`
- （支撑信息）`spatial_preconditions[*]`、`affordance_preconditions[*]`、`affordance_hotspot.mechanism`

**示例**：
- 字段：`agent=hand`, `action=applies grasping force`, `patient=cucumber`, `eff_pat=...`, `eff_env=...`, `mechanism=...`
- QA 示例：
  - Q：Explain the complete physical causal chain driving this interaction.
  - A：The hand applies grasping force to the cucumber, leveraging friction at the graspable surface so the cucumber becomes constrained and can be lifted against gravity. This relies on direct hand–cucumber contact and leads to the cucumber leaving the supporting counter surface.

---

### Task_07_Scene_Goal_Derivation（场景高阶目标）

**任务定义**：输出/改写整体高阶目标。

**数据源头字段（JSONPath）**：
- `high_level_goal`

**示例**：
- 字段：`"Prepare a simple salad by washing and slicing a cucumber."`
- QA 示例：
  - Q：Given the scene context, what is the logical high-level goal?
  - A：The logical high-level goal is to prepare a simple salad by washing and slicing a cucumber.

---

### Task_08_Strategic_Rationale_Justification（步骤动机/必要性解释）

**任务定义**：解释某一步为什么必要、如何服务整体目标。

**数据源头字段（JSONPath）**：
- `high_level_goal`
- `steps[i].step_goal`
- `steps[i].rationale`

**示例**：
- 字段：`step_goal="Retrieve a cucumber..."`, `rationale="The cucumber must be accessible..."`
- QA 示例：
  - Q：Given the overall goal, why is this step strategically necessary?
  - A：This step is necessary because the cucumber must be brought to the sink before washing, which directly supports the overall goal of preparing the salad.

---

### Task_09_Precondition_Statement（步骤前置条件陈述）

**任务定义**：列出开始该步前必须满足的条件。

**数据源头字段（JSONPath）**：
- `steps[i].step_goal`
- `steps[i].preconditions[*]`

**示例**：
- 字段：`preconditions=["The cucumber is present...", "The agent has a free hand..."]`
- QA 示例：
  - Q：What mandatory preconditions must be met before initiating this step?
  - A：Initialization requires that the cucumber is present in the environment and that the agent has a free hand available to grasp it.

---

### Task_10_Step_Execution_Statement（步骤执行动作描述）

**任务定义**：给出“如何执行这一步”的动作描述。

**数据源头字段（JSONPath）**：
- `steps[i].step_goal`
- （可选更细粒度执行序列）`steps[i].critical_frames[*].action_description`（按 `frame_index` 排序）

**示例**：
- 字段：`step_goal="Retrieve a cucumber..."`, `action_description="The right hand grasps..."`
- QA 示例：
  - Q：Detail the specific execution actions required for this step.
  - A：To execute this step, the agent grasps the cucumber from the counter, lifts it, and transports it to a position over the sink basin.

---

### Task_11_Expected_Physical_Effects（期望效果/后置状态）

**任务定义**：描述完成该步后预期发生的宏观物理结果（包含空间/状态结果）。

**数据源头字段（JSONPath）**：
- `steps[i].expected_effects[*]`
- （可选上下文）`steps[i].step_goal`

**示例**：
- 字段：`expected_effects=["The cucumber is held...", "...over the sink basin."]`
- QA 示例：
  - Q：Upon completion of this step, what physical effects should be expected?
  - A：Upon completion, the cucumber is held by the agent and is positioned over the sink basin, establishing a ready state for washing.

---

### Task_12_Inter_Step_Dependency_Analysis（跨步依赖分析）

**任务定义**：解释“上一步的效果”如何满足“下一步的前置条件”。

**数据源头字段（JSONPath）**：
- `steps[i].step_goal`
- `steps[i].expected_effects[*]`
- `steps[i+1].step_goal`
- `steps[i+1].preconditions[*]`

**可选派生字段**：
- `dependency_pairs = match(expected_effects, next_preconditions)`（可用词项重合/包含做启发式匹配，不引入新事实）

**示例**：
- 上一步效果：`"The cucumber is positioned over the sink basin."`
- 下一步前置：`"The cucumber is at the sink."`
- QA 示例：
  - Q：How does the outcome of the previous step satisfy the preconditions for the next step?
  - A：Placing the cucumber over the sink basin establishes that it is at the sink, which satisfies the next step’s requirement that the cucumber is positioned for washing under the faucet.

---

### Task_13_Next_Action_Prediction（下一步/下一动作预测：严格基于计划，不做臆测）

**任务定义**：在不引入臆测的前提下，给出“计划中下一步要做什么”。

**数据源头字段（JSONPath）**：
- `steps[i+1].step_goal`（作为“下一动作/下一子目标”的**标签**）
- （可选上下文）`steps[i].expected_effects[*]`, `steps[i].step_goal`, `high_level_goal`

**示例**：
- 标签：下一步 `step_goal="Wash the cucumber under running water."`
- QA 示例：
  - Q：Given the current state, what is the next planned action?
  - A：The next planned action is to wash the cucumber under running water.

---

### Task_14_Counterfactual_Prediction（反事实挑战与结果）

**任务定义**：对“what-if”挑战问题给出预期物理后果（直接使用字段或自然化改写）。

**数据源头字段（JSONPath）**：
- `steps[i].causal_challenge_question`
- `steps[i].expected_challenge_outcome`

**示例**：
- 字段：`question="What if the cucumber is wet and slippery..."`, `outcome="The grasp may fail..."`
- QA 示例：
  - Q：What if the cucumber is wet and slippery before grasping?
  - A：The grasp may fail due to reduced friction, requiring a different grip strategy or slower motion.

---

### Task_15_Failure_Recovery_Protocol（失败模式与恢复策略）

**任务定义**：当某失败发生时，给出恢复流程。

**数据源头字段（JSONPath）**：
- `steps[i].failure_handling.reason`
- `steps[i].failure_handling.recovery_strategy`

**示例**：
- 字段：`reason="The cucumber slips..."`, `strategy="Reposition the hand..."`
- QA 示例：
  - Q：If the action fails due to the cucumber slipping from the hand, what is the protocol?
  - A：If failure occurs because the cucumber slips, the protocol is to reposition the hand to increase contact area and re-grasp more firmly.

---

### Task_16_Physical_Feasibility_Verification（物理可行性核验：空间+可供性条件）

**任务定义**：分别核验空间拓扑条件与可供性/功能条件是否满足该帧动作。

**数据源头字段（JSONPath）**：
- `steps[i].critical_frames[j].spatial_preconditions[*]`
- `steps[i].critical_frames[j].affordance_preconditions[*]`
- （可选上下文）`action_description`

**示例**：
- 空间条件：`hand is in contact with cucumber`, `cucumber is on counter`
- 可供性条件：`cucumber` 是 `graspable`
- QA 示例：
  - Q：Verify the spatial and affordance-based conditions required for this interaction.
  - A：Spatially, the hand must be in direct contact with the cucumber while the cucumber is positioned on a stable support surface prior to lifting. In terms of affordances, the cucumber must be graspable, which is supported by its elongated geometry enabling a stable hand grip.

---

### Task_17_Holistic_Step_Synthesis_Why_How（步级综合：Why/How 两段式）

**任务定义**：将“战略动机(Why)”与“物理机制(How)”拆成两段说明。

**数据源头字段（JSONPath）**：
- Why：`steps[i].rationale`
- How（从该步某个关键帧抽取）：
  - `steps[i].critical_frames[*].affordance_hotspot.mechanism`
  - `steps[i].critical_frames[*].causal_chain.*`
  - `steps[i].critical_frames[*].spatial_preconditions[*]`

**示例**：
- Why：`"The cucumber must be accessible at the sink..."`
- How：`mechanism="normal force and friction..."` + 因果链字段
- QA 示例：
  - Q：Explain both the strategic reasoning (Why) and the physical mechanism (How) for this step.
  - A：(Why) The cucumber must be moved to the sink so washing can be performed as part of the overall preparation goal. (How) The hand applies grasping force at a graspable surface and relies on friction to prevent slip, allowing the cucumber to be lifted off the counter and transported.

---

## 5. 数据集生成“详细规划”（不改代码的落地方案）

下面给出一个与 `generate_phyplan_api.py` 风格一致的规划，强调：
- **严格溯源字段**（meta.fields 记录源字段）
- **分粒度采样**（场景/步/帧/关系/跨步）
- **平衡与去重**（避免某类样本过多）

### 5.1 输入解析与规范化

1) 读取单条 plan JSON（符合 schema）。
2) 标准化：
   - `steps` 按 `step_id` 升序
   - `critical_frames` 按 `frame_index` 升序
3) 建立索引：
   - `scene_ctx = {high_level_goal}`
   - `step_ctx[i] = {step_goal, rationale, preconditions, expected_effects, tools, materials, challenge, failure}`
   - `frame_ctx[i][j] = {frame_index, action_description, state_change, spatial_preconditions, affordance_preconditions, causal_chain, affordance_hotspot}`

### 5.2 候选样本生成（按任务）

建议“候选→过滤→抽样”的三段式：

- 场景级：
  - Task_01（1 条/plan）、Task_07（1 条/plan）
- 步级：
  - Task_04/08/09/10/11/14/15（每 step 至少 1 条）
- 帧级：
  - Task_03/05/06/16（每 critical_frame 至少 1 条）
- 关系级：
  - Task_02（每条 `spatial_preconditions` 可生成 1 条；如太多可抽样）
- 跨步级：
  - Task_12（对相邻 steps 做依赖检测，命中才生成；至少 0~N 条）
  - Task_13（对每个 step，若存在下一步则生成 1 条）

### 5.3 质量约束（强烈建议）

- **字段完整性**：任何任务所需字段缺失则跳过（或填空但打标）。
- **去重**：
  - 同一 step 内，若多个 critical_frame 的 `action_description` 高度相似，可只保留最早或抽样。
  - 关系级任务对同 `(relation, objects)` 去重。
- **平衡**：限制每个 plan 对每个任务的最大样本数（例如：关系级最多 10 条/plan）。
- **跨步一致性**：Task_12 仅在 `expected_effects` 与下一步 `preconditions` 存在词项重合/包含时生成，避免牵强解释。

### 5.4 输出建议（与 generate_phyplan_api 对齐）

建议每条样本都带 `meta.fields` 以便溯源，例如：

```json
{
  "id": "uuid",
  "conversations": [
    {"from": "human", "value": "<question>"},
    {"from": "gpt", "value": "<answer>"}
  ],
  "meta": {
    "task_name": "Task_06_Holistic_Causal_Chain_Analysis",
    "step_index": 1,
    "frame_index": 7,
    "fields": {
      "agent": "hand",
      "action": "applies grasping force",
      "patient": "cucumber",
      "mechanism": "normal force and friction ...",
      "spatial": [
        {"relation": "hand is in contact with cucumber", "objects": ["hand", "cucumber"], "truth": true}
      ]
    }
  }
}
```

---

## 6. 覆盖检查（确保“完整全面”）

从字段覆盖角度，以上任务集合至少覆盖：
- 顶层：`high_level_goal`（Task_07、Task_08 等）
- 步级：`step_goal/rationale/preconditions/expected_effects/tools/materials/challenge/failure`（Task_04/08/09/10/11/14/15）
- 帧级：`frame_index/action/state/spatial/affordance/causal/hotspot`（Task_02/03/05/06/16）
- 跨步：`expected_effects` ↔ `preconditions`（Task_12）与下一步目标（Task_13）

如果你后续希望“强对齐 PhyPlan 多模态版本”（加入图片），可在不改变 schema 的前提下，通过外部路径规则将 `frame_index` 映射到关键帧图片路径，再把图片路径作为 sample 的 `image` 字段补充进输出。
