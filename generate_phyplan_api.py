# -*- coding: utf-8 -*-
"""
PhyPlan 任务生成（API 版，带进度日志）

目标：
- 任务与提示逻辑严格对齐 scripts/libero/generate_phyplan.py 与 AGENTS.md。
- 唯一差异：调用模型的方式改为直接 API（与 mani_longvideo.py 一致），并提供直观进度日志。

特性：
- 抑制 httpx/httpcore/openai 的冗余日志，保留清晰的阶段进度：Start/Done、Step、关键帧、API 调用耗时、缓冲写盘统计。
- 输出格式沿用 ShareGPT JSONL（每任务目录 data.jsonl）。
"""

import os
import json
import uuid
import logging
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 降噪：与 mani_longvideo.py 保持一致，屏蔽 HTTP 库冗余日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1) 模型 API 配置（与 mani_longvideo.py 一致）
# ==============================================================================

@dataclass
class ApiConfig:
    api_key: str = os.environ.get('API_KEY', 'EMPTY')
    api_base_url: str = os.environ.get('API_BASE_URL', 'http://model.mify.ai.srv/v1')
    model_provider_id: str = os.environ.get('MODEL_PROVIDER_ID', 'vertex_ai')
    model_name: str = os.environ.get('MODEL_NAME', 'gemini-3-pro-preview')
    # 可选参数
    max_tokens: int = int(os.environ.get('MAX_TOKENS', '8192'))
    request_images_limit: int = int(os.environ.get('REQUEST_IMAGES_LIMIT', '1000000'))


def initialize_api_client(cfg: ApiConfig):
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base_url,
            default_headers={"X-Model-Provider-Id": cfg.model_provider_id}
        )
        logger.info(">>> [INFO] OpenAI 兼容客户端初始化成功")
        return client
    except Exception as e:
        logger.warning(f"初始化 OpenAI 客户端失败：{e}. 将回退为本地占位处理。")
        return None


# ==============================================================================
# 2) 提示模板（保持与 generate_phyplan.py 完全一致）
# ==============================================================================

SYSTEM_PROMPT = """You are an expert Embodied AI Analyst and Physics Consultant.
Your task is to synthesize structured data fields into high-quality, natural language answers for a QA dataset.

### Core Objectives:
1. **Naturalness**: Do not just list the fields. Wield the data into fluent, professional English sentences. Use logical connectors (e.g., "consequently," "enabled by," "specifically") to show relationships between fields.
2. **Strict Grounding**: The output must be based on the provided Input Data. Do not hallucinate external objects, physics, or intents not present in the JSON.
3. **Detail & Rigor**: Include ALL technical details provided in the input (e.g., specific tool names, precise spatial relations). Do not simplify or summarize if it loses precision.
4. **Professional Tone**: Keep the language objective and academic. Avoid flowery adjectives (e.g., "amazing," "carefully") unless they are in the source text. Avoid conversational fillers (e.g., "Here is the answer").

### Output Format:
- Return ONLY the final answer text.
- Unless specified otherwise, combine information into cohesive paragraphs rather than bullet points.
"""

TASK_PROMPTS = {
    # ==========================================
    # 支柱一：感知与锚点 (Perception & Anchoring)
    # ==========================================
    
    "Task_01_Macro_Anchor_Extraction": """Input Data:
Scene Context: {scene_desc}
Key Objects: {key_objects}

Instruction: The user asks: "Based on the scene, which objects are the key interactable anchors for planning?"
Draft a response that lists the items from `Key Objects`. Integrate them into a complete sentence (e.g., "The key anchors identified in this scene include X, Y, and Z."). Ensure the list is exhaustive based on the input.""",

    "Task_03_Micro_Affordance_Visual_Semantics": """Input Data:
Affordance Type: {aff_type}
Hotspot Description: {desc}

Instruction: The user asks: "Identify the specific region involved and its functional role."
Synthesize the `Hotspot Description` and `Affordance Type` into a descriptive statement.
Example format: "The [Affordance Type] is characterized by [Description], serving as the functional interface." """,

    "Task_04_Entity_Role_Identification": """Input Data:
Tools: {tools}
Materials: {materials}

Instruction: The user asks: "Distinguish between the active tools and the passive materials in this interaction."
Formulate a sentence that clearly assigns roles.
Example format: "The active tool utilized is the [Tools], while the material being acted upon is the [Materials]." """,

    # ==========================================
    # 支柱二：物理动力学 (Physical Dynamics)
    # ==========================================

    "Task_06_Holistic_Causal_Chain_Analysis": """Input Data:
Agent: {agent}
Action: {action}
Patient: {patient}
Physical Basis (Affordance): {aff_pre}
Mechanism: {mechanism}
Spatial Condition: {spatial}
Effect on Patient: {eff_pat}
Effect on Environment: {eff_env}

Instruction: The user asks: "Analyze the complete physical causal chain driving this interaction."
Compose a comprehensive, academic-style paragraph.
Flow: Start with the Agent interacting with the Patient. Explain that this interaction is grounded in the [Physical Basis] and mechanically driven by [Mechanism]. Note that it relies on the [Spatial Condition]. Conclude with the specific effects on the Patient and the Environment.""",

    "Task_05_State_Evolution_Description": """Input Data:
Ongoing Action: {action_desc}
Resulting State Change: {state_change}

Instruction: The user asks: "Describe the ongoing action and the immediate resulting state change."
Combine these two fields into a single cause-and-effect statement.
Example format: "The actor is currently [Ongoing Action], which directly results in [Resulting State Change]." """,

    "Task_02_Transient_Geometric_Verification": """Input Data:
Object A: {obj_a}
Object B: {obj_b}
Spatial Relation: {relation}

Instruction: The user asks: "What is the precise spatial relationship between these objects at this moment?"
State the relationship strictly using the provided term.
Example format: "At this keyframe, [Object A] is positioned [Spatial Relation] relative to [Object B]." """,

    # ==========================================
    # 支柱三：逻辑规划 (Logical Planning)
    # ==========================================

    "Task_07_Scene_Goal_Derivation": """Input Data:
High-Level Goal: {high_level_goal}

Instruction: The user asks: "Given the scene context, what is the logical high-level goal?"
State the goal in a complete sentence: "The logical high-level goal is to [High-Level Goal]." """,

    "Task_10_Step_Execution_Statement": """Input Data:
Global Goal: {global_goal}
Step Goal: {step_goal}
Execution Actions: {actions}

Instruction: The user asks: "Detail the specific execution actions required for this step."
Convert the `Execution Actions` into a procedural description.
Format: "To execute this step, the agent must [Actions]." If `Actions` are empty, use the `Step Goal` as the description.""",

    "Task_08_Strategic_Rationale_Justification": """Input Data:
Global Goal: {global_goal}
Step Goal: {step_goal}
Rationale: {rationale}

Instruction: The user asks: "Why is this specific step necessary within the global plan?"
Provide the justification using the `Rationale`.
Format: "This step is foundational because [Rationale], which facilitates the progression towards [Global Goal]." """,

    "Task_09_Precondition_Statement": """Input Data:
Step Goal: {step_goal}
Preconditions: {preconditions}

Instruction: The user asks: "What mandatory preconditions must be met before initiating this step?"
Present the `Preconditions` as a requirement statement.
Format: "Initialization of this step requires the following conditions: [List of conditions]." """,

    "Task_11_Expected_Physical_Effects": """Input Data:
Expected Effects: {macro_eff}
Final Spatial State: {spatial_post}
Final Affordance State: {affordance_post}

Instruction: The user asks: "What are the expected physical outcomes and final states after this step?"
Combine the macro and micro effects into a cohesive response.
Format: "Upon completion, the expected effects are: [Expected Effects]. Specifically, the final spatial state involves [Final Spatial State], and the object's affordance transitions to [Final Affordance State]." """,

    "Task_12_Inter_Step_Dependency_Analysis": """Input Data:
Step N Goal: {step_n_goal}
Step N Effect: {step_n_effect}
Step N+1 Goal: {step_next_goal}
Step N+1 Precondition: {step_next_precondition}

Instruction: The user asks: "Analyze the logical dependency between the previous step and the current step."
Explain how the previous output allows the current input.
Format: "The execution of '[Step N Goal]' results in [Step N Effect]. This outcome directly satisfies the precondition required for '[Step N+1 Goal]', specifically by establishing [connection between Effect and Precondition]." """,

    # ==========================================
    # 支柱四：鲁棒性与预测 (Robustness & Prediction)
    # ==========================================

    "Task_14_Counterfactual_Prediction": """Input Data:
Challenge Question: {question}
Expected Outcome: {outcome}

Instruction: The user asks: "{question}"
Provide the predicted physical consequence based on the `Expected Outcome`.
Format: "Under such conditions, the expected outcome is that [Outcome]." """,

    "Task_13_Next_Action_Prediction": """Input Data:
Next Actions: {next_actions}

Instruction: The user asks: "Given the current state, what are the logical next micro-actions?"
List the actions in a flowing sentence.
Format: "The logical next actions are to [Action 1], [Action 2], and [Action N]." """,

    "Task_15_Failure_Recovery_Protocol": """Input Data:
Failure Reason: {reason}
Recovery Strategy: {strategy}

Instruction: The user asks: "If the action fails due to [Reason], what is the protocol?"
Synthesize a recovery directive.
Format: "If failure occurs due to [Reason], the protocol dictates that the agent must [Strategy]." """,

    # ==========================================
    # 支柱五：验证与综合 (Verification & Synthesis)
    # ==========================================

    "Task_16_Physical_Feasibility_Verification": """Input Data:
Step Goal: {step_goal}
Spatial Preconditions: {spatial_preconditions}
Affordance Preconditions: {affordance_preconditions}

Instruction: The user asks: "Verify the spatial and affordance-based conditions required for this step."
Produce two distinct paragraphs.
Paragraph 1 (Spatial): Detail the required spatial relationships.
Paragraph 2 (Affordance): Detail the required object properties and reasons.
Constraint: Ensure the distinction between spatial topology and object properties is clear.""",

    "Task_17_Holistic_Step_Synthesis_Why_How": """Input Data:
Step Goal: {step_goal}
Strategic Rationale: {rationale}
Physical Mechanism: {mechanism}

Instruction: The user asks: "Explain both the strategic reasoning ('Why') and the physical mechanism ('How') for this step."
Produce two distinct paragraphs.
Paragraph 1 (Strategy): Explain the `Strategic Rationale`.
Paragraph 2 (Mechanism): Explain the `Physical Mechanism`.
Constraint: Do not mix the strategic intent with the physical execution details."""
}

# ==============================================================================
# 2.1) QA 生成辅助：根据任务生成英文疑问句与回答提示
# ==============================================================================

def _strip_quotes_punct(s: Optional[str]) -> str:
    try:
        import re
        t = (s or "").strip()
        # remove surrounding quotes/brackets
        t = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", t)
        # remove trailing sentence punctuation
        t = re.sub(r"[\.;:!?]+$", "", t)
        # collapse whitespace
        t = re.sub(r"\s+", " ", t)
        return t
    except Exception:
        return s or ""

def _finalize_question(q: str) -> str:
    try:
        import re
        s = (q or "").strip()
        # Remove double spaces around punctuation
        s = re.sub(r"\s+([,;:])", r"\1", s)
        s = re.sub(r"([,;:])\s+", r"\1 ", s)
        # Ensure ends with '?'
        s = s.rstrip(" .!;")
        if not s.endswith("?"):
            s = s + "?"
        return s
    except Exception:
        return q or ""

def build_user_question(task_name: str, fields: Dict[str, Any]) -> str:
    sg_raw = fields.get('step_goal')
    hl_raw = fields.get('high_level_goal') or fields.get('global_goal')
    sg = _strip_quotes_punct(sg_raw)
    hl = _strip_quotes_punct(hl_raw)
    if task_name == 'Task_04_Entity_Role_Identification':
        q = f"In the step {sg or 'this step'}, which items function as tools, and which are the materials being acted upon"
        return _finalize_question(q)
    if task_name == 'Task_08_Strategic_Rationale_Justification':
        q = f"Given the overall goal {hl or 'the mission'}, why is the step {sg or 'this step'} strategically necessary"
        return _finalize_question(q)
    if task_name == 'Task_09_Precondition_Statement':
        q = f"Given the overall goal {hl or 'the mission'}, before starting {sg or 'this step'}, what objective preconditions must be satisfied"
        return _finalize_question(q)
    if task_name == 'Task_12_Inter_Step_Dependency_Analysis':
        s1 = _strip_quotes_punct(fields.get('step_n_goal') or 'the previous step')
        s2 = _strip_quotes_punct(fields.get('step_next_goal') or 'the next step')
        q = f"Given the overall goal {hl or 'the mission'}, how does the outcome of {s1} satisfy the preconditions for {s2}"
        return _finalize_question(q)
    if task_name == 'Task_14_Counterfactual_Prediction':
        q0 = _strip_quotes_punct(fields.get('question') or 'What if the condition were different')
        q = f"Given the overall goal {hl or 'the mission'} and the step goal {sg or 'this step'}, {q0}"
        return _finalize_question(q)
    if task_name == 'Task_13_Next_Action_Prediction':
        q = f"Given the overall goal {hl or 'the mission'}, what are the most logical next micro-actions for the step {sg or 'this step'}"
        return _finalize_question(q)
    if task_name == 'Task_15_Failure_Recovery_Protocol':
        reason = _strip_quotes_punct(fields.get('reason') or 'a potential failure')
        q = f"Given the overall goal {hl or 'the mission'}, in the step {sg or 'this step'}, why might it fail due to {reason}, and what recovery strategy should be applied"
        return _finalize_question(q)
    if task_name == 'Task_03_Micro_Affordance_Visual_Semantics':
        aff = _strip_quotes_punct(fields.get('aff_type') or 'this affordance')
        q = f"Which specific region affords {aff}, and how does it visually appear and physically function"
        return _finalize_question(q)
    if task_name == 'Task_06_Holistic_Causal_Chain_Analysis':
        agent = _strip_quotes_punct(fields.get('agent') or 'the agent')
        action = _strip_quotes_punct(fields.get('action') or 'acting on')
        patient = _strip_quotes_punct(fields.get('patient') or 'the object')
        q = f"Could you explain how {agent} is {action} {patient} in this keyframe, focusing on the spatial setup, the affordance-level mechanism, and the immediate effects"
        return _finalize_question(q)
    if task_name == 'Task_05_State_Evolution_Description':
        q = f"Given the overall goal {hl or 'the mission'}, what ongoing action is occurring, and what immediate state change does it cause"
        return _finalize_question(q)
    if task_name == 'Task_02_Transient_Geometric_Verification':
        a = _strip_quotes_punct(fields.get('obj_a') or 'the first object')
        b = _strip_quotes_punct(fields.get('obj_b') or 'the second object')
        q = f"What is the precise spatial relationship between {a} and {b} in this frame"
        return _finalize_question(q)
    if task_name == 'Task_11_Expected_Physical_Effects':
        q = f"Given the overall goal {hl or 'the mission'}, upon completion of {sg or 'this step'}, what physical effects should be expected for the environment and objects"
        return _finalize_question(q)
    if task_name == 'Task_16_Physical_Feasibility_Verification':
        q = f"Given the overall goal {hl or 'the mission'}, is the step {sg or 'this step'} physically feasible now, based on spatial and affordance evidence"
        return _finalize_question(q)
    if task_name == 'Task_17_Holistic_Step_Synthesis_Why_How':
        q = f"Given the overall goal {hl or 'the mission'}, why is the step {sg or 'this step'} necessary, and how is it physically achieved"
        return _finalize_question(q)
    if task_name == 'Task_01_Macro_Anchor_Extraction':
        q = "Which stable objects are the task-relevant anchors for planning in this scene"
        return _finalize_question(q)
    if task_name == 'Task_07_Scene_Goal_Derivation':
        q = "Given the current scene, what is the appropriate high-level goal"
        return _finalize_question(q)
    if task_name == 'Task_10_Step_Execution_Statement':
        q = f"Given the overall goal {hl or 'the mission'}, for the step {sg or 'this step'}, what specific execution actions are required"
        return _finalize_question(q)
    return _finalize_question("Could you provide a concise, grounded answer based on the provided context")

def build_answer_prompt(task_name: str, fields: Dict[str, Any]) -> str:
    ctx = json.dumps(fields, ensure_ascii=False)
    tn = task_name
    # 任务指令（去模板化，自然语言合成）
    if tn == 'Task_01_Macro_Anchor_Extraction':
        instr = (
            "Provide a natural English listing of the task‑relevant anchors from the context, with items inline (commas or semicolons). Do not use bullets, commentary, or decorative phrasing."
        )
    elif tn == 'Task_02_Transient_Geometric_Verification':
        instr = (
            "Describe the precise spatial relationship between the two objects using only the given relation and names. Do not invent details; do not use bullets; avoid filler."
        )
    elif tn == 'Task_03_Micro_Affordance_Visual_Semantics':
        instr = (
            "Describe the specific region affording the function, its visual characteristics, and its physical role, using only the provided fields. Do not use bullets or decorative phrases."
        )
    elif tn == 'Task_04_Entity_Role_Identification':
        instr = (
            "Identify which items are the tools and which are the materials in a natural sentence. No bullets; avoid filler and decorative phrasing."
        )
    elif tn == 'Task_05_State_Evolution_Description':
        instr = (
            "Describe the ongoing action and the immediate state change, linking them naturally (e.g., 'thus', 'which results in'). Do not use bullets; avoid filler and decoration."
        )
    elif tn == 'Task_06_Holistic_Causal_Chain_Analysis':
        instr = (
            "Produce two natural paragraphs: (1) what the agent is doing to the patient and under which spatial setup, and how the affordance‑level conditions support it; "
            "(2) the mechanism at the affordance hotspot and the immediate effects on the patient and the environment. Use only the given fields; avoid filler; do not limit length."
        )
    elif tn == 'Task_07_Scene_Goal_Derivation':
        instr = "Return the high‑level goal verbatim without any added words."
    elif tn == 'Task_08_Strategic_Rationale_Justification':
        instr = "Return the rationale verbatim without any added words."
    elif tn == 'Task_09_Precondition_Statement':
        instr = (
            "Present all required preconditions from the context in a natural paragraph. Cover every item; list them inline and separate them naturally (e.g., with semicolons). Do not use bullets or labels; avoid commentary or decorative phrases."
        )
    elif tn == 'Task_10_Step_Execution_Statement':
        instr = (
            "Describe the execution actions naturally. If explicit actions are missing, restate the step goal as the action. Do not limit length; avoid filler."
        )
    elif tn == 'Task_11_Expected_Physical_Effects':
        instr = (
            "Summarize the expected effects in a natural paragraph without bullets. Cover all effects inline (separated naturally), and if final spatial or affordance states are provided, add natural sentences describing them. Do not limit length; avoid filler or decoration."
        )
    elif tn == 'Task_12_Inter_Step_Dependency_Analysis':
        instr = (
            "Explain how a specific effect from the previous step satisfies a specific precondition of the next step, using only the given phrases. Do not use bullets; avoid filler; do not limit length."
        )
    elif tn == 'Task_13_Next_Action_Prediction':
        instr = (
            "Provide the logical next micro‑actions in a natural sentence. List items inline separated naturally; do not use bullets; avoid filler or decorative language."
        )
    elif tn == 'Task_14_Counterfactual_Prediction':
        instr = "Return the expected outcome verbatim without any added words."
    elif tn == 'Task_15_Failure_Recovery_Protocol':
        instr = (
            "Combine the failure reason and the recovery strategy into a natural sentence. Use only the given fields; do not invent details; avoid filler; do not limit length."
        )
    elif tn == 'Task_16_Physical_Feasibility_Verification':
        instr = (
            "Write two natural paragraphs: (1) spatial conditions drawn from the context; (2) affordance conditions with object names, types and reasons. Do not limit length; avoid filler."
        )
    elif tn == 'Task_17_Holistic_Step_Synthesis_Why_How':
        instr = (
            "Write two natural paragraphs: (1) why the step is necessary (rationale); (2) how it is physically achieved (mechanism). Do not limit length; avoid filler or decorative language."
        )
    else:
        instr = "Answer concisely and objectively using only the given fields."

    return (
        "Context (Fields):\n" + ctx + "\n\n" +
        "Instruction: " + instr + " Produce a natural, fluent English answer; avoid rigid templates, labels, and speculation."
    )


# ==============================================================================
# 3) 工具：读取图片为 base64 并构造 image_url 内容
# ==============================================================================

def _read_image_as_base64(path: str) -> Optional[str]:
    try:
        import base64
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.warning(f"读取图片失败：{path}，原因：{e}")
        return None


def _build_image_contents(image_paths: List[str], limit: int) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    count = 0
    for p in image_paths:
        if count >= max(0, limit):
            break
        b64 = _read_image_as_base64(p)
        if not b64:
            continue
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        count += 1
    return contents


# ==============================================================================
# 4) 主类：API 版本生成器（逻辑与 generate_phyplan.py 等价）
# ==============================================================================

class PhyPlanAPIGenerator:
    def __init__(self, output_dir: str = "phyplan_output_api", api_config: ApiConfig = None, stream_save: bool = True, resume: bool = True, force: bool = False, processed_keys: Optional[set] = None, live_combined: bool = True, balanced: bool = True):
        self.output_dir = output_dir
        self.data_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.data_buffer_A: Dict[str, List[Dict[str, Any]]] = {}
        self.data_buffer_B: Dict[str, List[Dict[str, Any]]] = {}
        self.api_config = api_config or ApiConfig()
        self.client = initialize_api_client(self.api_config)
        self.stream_save = stream_save
        self.resume = resume
        self.force = force
        self.processed_keys = processed_keys or set()
        self.live_combined = live_combined
        # 平衡策略：限制 TypeA 的帧级任务数量，避免某类任务过多
        self.balanced = balanced
        self.balance_caps = {
            'Task_03_Micro_Affordance_Visual_Semantics': 1,
            'Task_05_State_Evolution_Description': 1,
            'Task_06_Holistic_Causal_Chain_Analysis': 1,
            'Task_02_Transient_Geometric_Verification': 1,
            'Task_16_Physical_Feasibility_Verification': 1,
            'Task_17_Holistic_Step_Synthesis_Why_How': 1,
        }
        # 全局候选池（用于随机选择生成，避免总是取“第一帧/第一步”）
        self.global_caps_A: Dict[str, int] = {
            # 帧级任务（原本可多条 → 每 JSON 每类最多 2 条）
            'Task_03_Micro_Affordance_Visual_Semantics': 2,
            'Task_05_State_Evolution_Description': 2,
            'Task_06_Holistic_Causal_Chain_Analysis': 2,
            'Task_02_Transient_Geometric_Verification': 2,
            'Task_11_Expected_Physical_Effects': 2,
            'Task_16_Physical_Feasibility_Verification': 2,
            'Task_17_Holistic_Step_Synthesis_Why_How': 2,
            # 步级任务（原本可多条 → 每 JSON 每类最多 2 条）
            'Task_04_Entity_Role_Identification': 2,
            'Task_08_Strategic_Rationale_Justification': 2,
            'Task_09_Precondition_Statement': 2,
            'Task_12_Inter_Step_Dependency_Analysis': 2,
            'Task_13_Next_Action_Prediction': 2,
            'Task_14_Counterfactual_Prediction': 2,
            'Task_15_Failure_Recovery_Protocol': 2,
        }
        self.global_caps_B: Dict[str, int] = {
            # 步级任务（原本可多条 → 每 JSON 每类最多 2 条）
            'Task_10_Step_Execution_Statement': 2,
            'Task_08_Strategic_Rationale_Justification': 2,
            'Task_09_Precondition_Statement': 2,
            'Task_11_Expected_Physical_Effects': 2,
            'Task_12_Inter_Step_Dependency_Analysis': 2,
            'Task_15_Failure_Recovery_Protocol': 2,
        }
        self.global_candidates_A: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.global_caps_A.keys()}
        self.global_candidates_B: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.global_caps_B.keys()}
        self._init_directories()

    def _init_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for task_name in TASK_PROMPTS.keys():
            os.makedirs(os.path.join(self.output_dir, task_name), exist_ok=True)
        self.data_buffer = {k: [] for k in TASK_PROMPTS.keys()}
        self.data_buffer_A = {k: [] for k in TASK_PROMPTS.keys()}
        self.data_buffer_B = {k: [] for k in TASK_PROMPTS.keys()}

    def _accept_balance(self, task_name: str, counter: Dict[str, int]) -> bool:
        if not self.balanced:
            return True
        cap = self.balance_caps.get(task_name)
        if cap is None:
            return True
        cur = counter.get(task_name, 0)
        if cur >= cap:
            return False
        counter[task_name] = cur + 1
        return True

    # --- LLM 调用（带耗时进度日志） ---
    def call_llm(self, prompt: str, images: Union[str, List[str]]) -> str:
        if self.client is None or self.api_config.api_key == 'EMPTY':
            logger.warning("API 客户端不可用或 API_KEY=EMPTY，将返回占位文本。")
            return "[API Disabled] " + (prompt[:256] if isinstance(prompt, str) else "")

        # 按需：仅传入文本（问题 + 字段上下文），不再传图片内容给模型
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_list: List[str] = []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info(f">>> [INFO] 调用模型: model={self.api_config.model_name}, fields_only=true, max_tokens={self.api_config.max_tokens}")
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=self.api_config.model_name,
                messages=messages,
                max_tokens=self.api_config.max_tokens,
                temperature=0.3,
                top_p=0.9,
                presence_penalty=0
            )
            
            dt = time.time() - t0
            if not (resp and getattr(resp, 'choices', None)):
                logger.info(f"!!! [ERROR] API 响应为空或无 choices，用时 {dt:.2f}s")
                return ""
            choice = resp.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                logger.info(f"!!! [ERROR] API 响应缺少 message.content，用时 {dt:.2f}s")
                return ""
            out = choice.message.content or ""
            logger.info(f">>> [SUCCESS] API 调用完成，用时 {dt:.2f}s，输出长度={len(out)}")
            return out
        except Exception as e:
            logger.error(f"模型 API 调用失败：{e}")
            return ""

    def call_llm_custom(self, system_prompt: str, user_text: str, max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        """自定义系统提示与用户文本（仅文本，不传图），用于结构化校验等场景。"""
        if self.client is None or self.api_config.api_key == 'EMPTY':
            logger.warning("API 客户端不可用或 API_KEY=EMPTY，将返回占位文本。")
            return "[API Disabled] " + (user_text[:256] if isinstance(user_text, str) else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]
        try:
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=self.api_config.model_name,
                messages=messages,
                max_tokens=max_tokens or min(1024, self.api_config.max_tokens),
                temperature=temperature,
                top_p=0.9,
                presence_penalty=0
            )
            dt = time.time() - t0
            if not (resp and getattr(resp, 'choices', None)):
                logger.info(f"!!! [ERROR] 自定义 API 响应为空或无 choices，用时 {dt:.2f}s")
                return ""
            choice = resp.choices[0]
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                logger.info(f"!!! [ERROR] 自定义 API 响应缺少 message.content，用时 {dt:.2f}s")
                return ""
            out = choice.message.content or ""
            logger.info(f">>> [SUCCESS] 自定义 API 调用完成，用时 {dt:.2f}s，输出长度={len(out)}")
            return out
        except Exception as e:
            logger.error(f"自定义模型 API 调用失败：{e}")
            return ""

    def polish_question(self, raw_question: str, fields: Dict[str, Any]) -> str:
        ctx = json.dumps(fields, ensure_ascii=False)
        polish_instr = (
            "Context (Fields):\n" + ctx + "\n\n"
            "Instruction: Rewrite the following question into one natural English sentence. "
            "Preserve its meaning and any explicit mentions of the overall goal or step goal when present; use only the information implied by the context; "
            "avoid line breaks, bullets, dashes used as bullets, and quotes/labels. "
            "Return only the rewritten question.\n"
            "Question:\n" + (raw_question or "")
        )
        out = self.call_llm(polish_instr, images=[])
        try:
            import re
            out = re.sub(r"\s+", " ", out or "").strip()
        except Exception:
            pass
        return out or (raw_question or "")

    def _sanitize_text(self, text: str) -> str:
        try:
            import re
            s = (text or "").replace("\n", " ")
            # Normalize spaces
            s = re.sub(r"\s+", " ", s)
            # Remove spaces around hyphens inside words: "open - plan" -> "open-plan"
            s = re.sub(r"(?<=\w)\s*[-–]\s*(?=\w)", "-", s)
            # Normalize punctuation spacing: remove space before , ; : . ! ? and ensure single space after
            s = re.sub(r"\s+([,;:\.!\?])", r"\1", s)
            s = re.sub(r"([,;:\.!\?])(\S)", r"\1 \2", s)
            # Collapse spaces again
            s = re.sub(r"\s+", " ", s)
            return s.strip()
        except Exception:
            return text or ""

    def _sanitize_answer(self, task_name: str, text: str) -> str:
        """Sanitize formatting while preserving task-required paragraph structure.
        - Robustly removes bullet markers (e.g., '1.', '-', '*') at the start of lines.
        - Normalizes whitespace/newlines.
        - Preserves paragraph structure for specific tasks.
        """
        if not text:
            return ""
            
        try:
            import re
            # 1. Pre-cleaning: Remove bullet points/numbering at the start of lines ONLY
            # Matches: Start of line -> optional whitespace -> (bullet char OR number+dot) -> whitespace
            # This avoids deleting hyphens inside sentences.
            cleaned_text = re.sub(r'(?m)^\s*([\-\*•\>]+|\d+\.)\s+', '', text)

            if task_name in {'Task_06_Holistic_Causal_Chain_Analysis', 'Task_16_Physical_Feasibility_Verification', 'Task_17_Holistic_Step_Synthesis_Why_How'}:
                # Logic for preserving paragraphs (Task requires 2 paragraphs):
                # Split by 2+ newlines to identify logical paragraphs
                paragraphs = re.split(r'\n\s*\n', cleaned_text)
                # Clean each paragraph individually
                paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in paragraphs if p.strip()]
                
                # Fallback: If task requires 2 paragraphs but LLM generated 1 big block,
                # we return it as is (sanitized) rather than forcing a split which might break semantics.
                return "\n\n".join(paragraphs)
            else:
                # Logic for single paragraph:
                # Collapse all whitespace (newlines, tabs, spaces) into a single space
                return re.sub(r'\s+', ' ', cleaned_text).strip()
        except Exception:
            # Fallback to simple strip if regex fails
            return text.strip()

    def polish_answer(self, task_name: str, raw_answer: str, fields: Dict[str, Any]) -> str:
        """Second-pass polishing to enforce natural, concise, non-template English
        while strictly adhering to the provided data fields."""
        
        # 1. Skip polishing for verbatim tasks to ensure exact matches
        if task_name in {
            'Task_07_Scene_Goal_Derivation',
            'Task_14_Counterfactual_Prediction'
        }:
            return self._sanitize_answer(task_name, raw_answer)

        # 2. Prepare Context
        ctx = json.dumps(fields, ensure_ascii=False)
        
        # 3. Define Structural Constraints
        # For complex tasks, we guide the split based on content logic.
        if task_name in {'Task_06_Holistic_Causal_Chain_Analysis', 'Task_16_Physical_Feasibility_Verification', 'Task_17_Holistic_Step_Synthesis_Why_How'}:
            structure_instr = (
                "**Structure Requirement**: Produce exactly **TWO** distinct paragraphs.\n"
                "- Paragraph 1: Synthesize the setup, context, or physical conditions.\n"
                "- Paragraph 2: Explain the mechanism, execution, or resulting effects.\n"
                "Ensure a logical transition between them."
            )
        else:
            structure_instr = (
                "**Structure Requirement**: Produce a **SINGLE** coherent paragraph.\n"
                "Weave all data points into a continuous narrative flow without line breaks."
            )

        # 4. Construct the Prompt (Optimized for fidelity and naturalness)
        instr = (
            f"You are an expert Embodied AI Analyst. Your task is to synthesize structured data into a natural, rigorous, and high-quality English response.\n\n"
            f"### INPUT DATA (Ground Truth):\n{ctx}\n\n"
            f"### DRAFT ANSWER (For Reference):\n{raw_answer}\n\n"
            f"### INSTRUCTIONS:\n"
            f"Rewrite the draft based strictly on the Input Data to improve fluency and professional tone.\n"
            f"1. **Strict Fidelity**: Use ONLY the entities, actions, and relationships present in the 'Input Data'. Do NOT hallucinate external details or adjectives (e.g., do not add 'carefully' or 'gently' unless specified).\n"
            f"2. **Natural Flow**: Avoid robotic listing (e.g., 'The tool is X. The material is Y.'). Instead, use syntactic integration (e.g., 'The agent uses X to manipulate Y...').\n"
            f"3. **Objective Tone**: Maintain a clinical, factual tone. Avoid conversational fillers, subjective evaluation, or meta-commentary (e.g., 'Here is the answer').\n"
            f"4. **No Formatting**: Do NOT use bullet points, lists, bold text, or labels.\n"
            f"{structure_instr}\n\n"
            f"### POLISHED OUTPUT:"
        )

        # 5. Call LLM
        out = self.call_llm(instr, images=[])
        
        # 6. Fallback if LLM fails, otherwise sanitize the output
        return self._sanitize_answer(task_name, out if out and out.strip() else raw_answer)

    def _defluff_text(self, text: str) -> str:
        """Remove common filler/meta/decorative openings and stock phrases."""
        try:
            import re
            s = text or ""
            # Remove common leading fillers
            patterns = [
                r"^\s*(In summary|In conclusion|To summarize|Overall|In general|Generally),\s*",
                r"^\s*(In this (scene|image|frame|step)),\s*",
                r"^\s*(It should be noted that|Note that)\s*",
            ]
            for pat in patterns:
                s = re.sub(pat, "", s, flags=re.IGNORECASE)
            # Remove repeated generic starters inside sentences
            s = re.sub(r"\s+(Overall|In summary|In conclusion),\s*", ", ", s, flags=re.IGNORECASE)
            # Collapse whitespace and fix punctuation spacing
            s = self._sanitize_text(s)
            return s
        except Exception:
            return text or ""

    def create_sharegpt_entry(self, images: Union[str, List[str]], user_q: str, assistant_a: str, meta: Optional[Dict[str, Any]] = None) -> Dict:
        return {
            "id": str(uuid.uuid4()),
            "image": images,
            "conversations": [
                {"from": "human", "value": user_q},
                {"from": "gpt", "value": assistant_a}
            ],
            "meta": meta or {}
        }

    def save_to_buffer(self, task_name: str, entry: Dict):
        if task_name in self.data_buffer:
            self.data_buffer[task_name].append(entry)
            img = entry.get('image')
            src = 'list' if isinstance(img, list) else 'single'
            meta = entry.get('meta', {})
            src_file = meta.get('source_path', '')
            step_idx = meta.get('step_index', '')
            frame_idx = meta.get('frame_index', '')
            logger.info(f"[Buffered] task={task_name} src={src_file} step={step_idx} frame={frame_idx} (image_source={src})")
            # 实时保存到磁盘（像 mani_longvideo.py 一样边生成边落盘）
            try:
                file_path = os.path.join(self.output_dir, task_name, "data.jsonl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                # 在保存日志中直观展示任务与元信息
                meta_str = json.dumps(meta, ensure_ascii=False)
                logger.info(f"[Saved ] task={task_name} src={src_file} -> {file_path}\n[Meta ] {meta_str}")
            except Exception as e:
                logger.warning(f"[SaveWarn] {task_name} 追加写入失败：{e}")
            # 路由到 A/B 缓冲区，便于最终合并输出
            item_type = (meta.get('item_type') or '').upper()
            if item_type == 'TYPEA':
                self.data_buffer_A[task_name].append(entry)
            elif item_type == 'TYPEB':
                self.data_buffer_B[task_name].append(entry)
            # 实时更新合并 JSON
            if getattr(self, 'live_combined', False):
                try:
                    self._write_combined(task_name)
                except Exception as e:
                    logger.warning(f"[LiveCombinedWarn] 更新合并文件失败 task={task_name}: {e}")

    def _write_combined(self, task_name: str):
        a_entries = self.data_buffer_A.get(task_name, [])
        b_entries = self.data_buffer_B.get(task_name, [])
        count = len(a_entries) + len(b_entries)
        combined = {"count": count, "A": a_entries, "B": b_entries}
        out_json = os.path.join(self.output_dir, task_name, "data.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info(f"[CombinedLive] {task_name} -> {out_json} (count={count}, A={len(a_entries)}, B={len(b_entries)})")

    def hydrate_existing_buffers(self):
        """在启动时加载已存在的数据进入 A/B 缓冲，确保实时合并包含历史样本。"""
        for task_name in TASK_PROMPTS.keys():
            # 优先从 data.json 读取；若不存在，则尝试 data.jsonl
            path_json = os.path.join(self.output_dir, task_name, 'data.json')
            path_jsonl = os.path.join(self.output_dir, task_name, 'data.jsonl')
            a_count = b_count = 0
            try:
                if os.path.exists(path_json):
                    with open(path_json, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                    a_entries = obj.get('A', []) or []
                    b_entries = obj.get('B', []) or []
                    for e in a_entries:
                        self.data_buffer_A[task_name].append(e)
                        self.data_buffer[task_name].append(e)
                    for e in b_entries:
                        self.data_buffer_B[task_name].append(e)
                        self.data_buffer[task_name].append(e)
                    a_count = len(a_entries)
                    b_count = len(b_entries)
                elif os.path.exists(path_jsonl):
                    with open(path_jsonl, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                meta = entry.get('meta', {})
                                itype = (meta.get('item_type') or '').upper()
                                if itype == 'TYPEA':
                                    self.data_buffer_A[task_name].append(entry)
                                    self.data_buffer[task_name].append(entry)
                                    a_count += 1
                                elif itype == 'TYPEB':
                                    self.data_buffer_B[task_name].append(entry)
                                    self.data_buffer[task_name].append(entry)
                                    b_count += 1
                                else:
                                    # 未标注类型的条目，放入总缓冲但不计入 A/B
                                    self.data_buffer[task_name].append(entry)
                            except Exception:
                                continue
                if a_count or b_count:
                    logger.info(f"[Hydrate] task={task_name} preload A={a_count} B={b_count}")
            except Exception as e:
                logger.warning(f"[HydrateWarn] 预加载 {task_name} 失败: {e}")

    def _key_from_meta(self, meta: Dict[str, Any]) -> str:
        return f"{meta.get('item_type','')}|{meta.get('source_path','')}|{meta.get('task_name','')}|{int(meta.get('step_index',-1))}|{int(meta.get('frame_index',-1))}|{int(meta.get('sub_index',0))}"

    def _meta(self, item_type: str, source_path: str, task_name: str, step_index: int, frame_index: int, sub_index: int, step_goal: Optional[str], image_path: Optional[str]) -> Dict[str, Any]:
        return {
            "item_type": item_type,
            "source_path": source_path,
            "task_name": task_name,
            "step_index": step_index,
            "frame_index": frame_index,
            "sub_index": sub_index,
            "step_goal": step_goal,
            "image_path": image_path,
        }

    def emit_sample(self, task_name: str, images: Union[str, List[str]], prompt: str, user_q: str, meta: Dict[str, Any], resume: bool = True, force: bool = False, fields: Optional[Dict[str, Any]] = None):
        key = self._key_from_meta(meta)
        if resume and not force and hasattr(self, 'processed_keys') and key in getattr(self, 'processed_keys', set()):
            logger.info(f"[SkipSample] {task_name} key={key}")
            return
        logger.info(
            f">>> [Gen ] task={task_name} src={meta.get('source_path','')} step={meta.get('step_index','')} frame={meta.get('frame_index','')} sub={meta.get('sub_index','')}"
        )
        # 将问句与字段指令合并为一次调用（压缩“问句润色+回答生成”为单次调用）
        # 问句仅做本地轻量清洗，保留 goal 关键信息
        polished_q = self._sanitize_text(user_q)
        full_prompt = f"User Question:\n{polished_q}\n\n" + (prompt or "")
        ans = self.call_llm(full_prompt, images)
        # 第二次润色：由模型将回答转化为自然英文（不使用模板/标签/不合理标点），并做基础清洗
        polished_ans = self.polish_answer(task_name, ans, fields or {}) if fields is not None else self._sanitize_text(ans)
        # Final pass: sanitize per-task structure (bullets vs single paragraph)
        polished_ans = self._sanitize_answer(task_name, polished_ans)
        # 去除冗余套话/装饰性开头，修复连字符与标点空格
        polished_ans = self._defluff_text(polished_ans)
        polished_q = self._sanitize_text(polished_q)
        # 直接保存（根据需求取消独立验证步骤）
        entry = self.create_sharegpt_entry(images, polished_q, polished_ans, meta=meta)
        self.save_to_buffer(task_name, entry)
        if hasattr(self, 'processed_keys'):
            self.processed_keys.add(key)

    # ---------------------------------------------------------
    # 依赖关系判定（与 generate_phyplan.py 一致）
    # ---------------------------------------------------------
    def _normalize_terms(self, text: str) -> set:
        if not isinstance(text, str) or not text.strip():
            return set()
        import re
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        stop = {"the","a","an","and","or","to","of","in","on","at","is","are","be","by","for","with","then","into","from","that","this"}
        return {t for t in tokens if t not in stop and len(t) >= 3}

    def _extract_effect_terms(self, step: Dict) -> set:
        terms = set()
        for e in step.get('expected_effects', []) or []:
            terms |= self._normalize_terms(e if isinstance(e, str) else str(e))
        for sp in step.get('spatial_postconditions_detail', []) or []:
            if isinstance(sp, dict):
                for k in ('relation','objects'):
                    v = sp.get(k)
                    if isinstance(v, str):
                        terms |= self._normalize_terms(v)
                    elif isinstance(v, list):
                        for item in v:
                            terms |= self._normalize_terms(item if isinstance(item, str) else str(item))
        for ap in step.get('affordance_postconditions_detail', []) or []:
            if isinstance(ap, dict):
                for k in ('object_name','affordance_types','reasons'):
                    v = ap.get(k)
                    if isinstance(v, str):
                        terms |= self._normalize_terms(v)
                    elif isinstance(v, list):
                        for item in v:
                            terms |= self._normalize_terms(item if isinstance(item, str) else str(item))
        return terms

    def _extract_precondition_terms(self, step: Dict) -> set:
        terms = set()
        for p in step.get('preconditions', []) or []:
            terms |= self._normalize_terms(p if isinstance(p, str) else str(p))
        return terms

    def _has_dependency(self, prev_step: Dict, next_step: Dict) -> bool:
        eff = self._extract_effect_terms(prev_step)
        pre = self._extract_precondition_terms(next_step)
        if not eff or not pre:
            return False
        if eff & pre:
            return True
        for e in eff:
            for p in pre:
                if e in p or p in e:
                    return True
        return False

    # =========================================================================
    # Type A 处理逻辑（逐帧）
    # =========================================================================
    def _process_type_a(self, data: Dict, source_path: Optional[str] = None):
        global_goal = data.get('high_level_goal', 'N/A')
        steps = data.get('steps', [])
        steps_count = len(steps) if isinstance(steps, list) else 0

        balance_counter: Dict[str, int] = {}

        for i, step in enumerate(steps):
            try:
                step_goal = step.get('step_goal', 'Unknown Step')
                critical_frames = step.get('critical_frames', [])
                if not critical_frames:
                    continue
                # 帧级任务按 frame_index 从小到大排序，步级锚点使用最早帧
                def _to_int(v):
                    try:
                        return int(v)
                    except Exception:
                        return 0
                critical_frames = sorted(critical_frames, key=lambda fr: _to_int(fr.get('frame_index', 0)))
                base_img = critical_frames[0].get('keyframe_image_path', 'missing.jpg')
                logger.info(f">>> [INFO] TypeA Step {i+1}/{len(steps)}: {step_goal}")

                # Task 3: Entity Role（加入全局候选池）
                tm = step.get('tool_and_material_usage', {})
                fields3 = {"tools": tm.get('tools'), "materials": tm.get('materials'), "step_goal": step_goal, "high_level_goal": global_goal}
                p3 = build_answer_prompt('Task_04_Entity_Role_Identification', fields3)
                user_q3 = build_user_question('Task_04_Entity_Role_Identification', fields3)
                meta3 = self._meta('TypeA', source_path or '', 'Task_04_Entity_Role_Identification', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_04_Entity_Role_Identification'].append({"images": base_img, "prompt": p3, "user_q": user_q3, "meta": meta3, "fields": fields3})

                # Task 8: Rationale（加入全局候选池）
                fields9 = {"global_goal": global_goal, "step_goal": step_goal, "rationale": step.get('rationale')}
                p9 = build_answer_prompt('Task_08_Strategic_Rationale_Justification', fields9)
                user_q9 = build_user_question('Task_08_Strategic_Rationale_Justification', fields9)
                meta9 = self._meta('TypeA', source_path or '', 'Task_08_Strategic_Rationale_Justification', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_08_Strategic_Rationale_Justification'].append({"images": base_img, "prompt": p9, "user_q": user_q9, "meta": meta9, "fields": fields9})

                # Task 9: Preconditions（加入全局候选池）
                fields10 = {"step_goal": step_goal, "preconditions": step.get('preconditions'), "high_level_goal": global_goal}
                p10 = build_answer_prompt('Task_09_Precondition_Statement', fields10)
                user_q10 = build_user_question('Task_09_Precondition_Statement', fields10)
                meta10 = self._meta('TypeA', source_path or '', 'Task_09_Precondition_Statement', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_09_Precondition_Statement'].append({"images": base_img, "prompt": p10, "user_q": user_q10, "meta": meta10, "fields": fields10})

                # Task 12: Step Dependency（仅当步骤数>1时生成跨步依赖）
                if steps_count > 1 and i < len(steps) - 1:
                    next_step = steps[i+1]
                    if self._has_dependency(step, next_step):
                        fields12 = {"step_n_goal": step_goal, "step_n_effect": step.get('expected_effects'), "step_next_goal": next_step.get('step_goal'), "step_next_precondition": next_step.get('preconditions'), "high_level_goal": global_goal}
                        p12 = build_answer_prompt('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        user_q12 = build_user_question('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        meta12 = self._meta('TypeA', source_path or '', 'Task_12_Inter_Step_Dependency_Analysis', i+1, -1, 0, step_goal, base_img)
                        self.global_candidates_A['Task_12_Inter_Step_Dependency_Analysis'].append({"images": base_img, "prompt": p12, "user_q": user_q12, "meta": meta12, "fields": fields12})
                elif steps_count <= 1:
                    logger.info("[TypeA] 单步骤短程任务：跳过跨步依赖（Task 12）生成")

                # Task 13: Counterfactual（如存在）
                if 'causal_challenge_question' in step:
                    fields14c = {"question": step.get('causal_challenge_question'), "outcome": step.get('expected_challenge_outcome'), "high_level_goal": global_goal, "step_goal": step_goal}
                    p13 = build_answer_prompt('Task_14_Counterfactual_Prediction', fields14c)
                    user_q13 = build_user_question('Task_14_Counterfactual_Prediction', fields14c)
                    meta14c = self._meta('TypeA', source_path or '', 'Task_14_Counterfactual_Prediction', i+1, -1, 0, step_goal, base_img)
                    self.global_candidates_A['Task_14_Counterfactual_Prediction'].append({"images": base_img, "prompt": p13, "user_q": user_q13, "meta": meta14c, "fields": fields14c})

                # Task 13: Next Action（加入全局候选池）
                fields13 = {"next_actions": step.get('predicted_next_actions'), "step_goal": step_goal, "high_level_goal": global_goal}
                p14 = build_answer_prompt('Task_13_Next_Action_Prediction', fields13)
                user_q14 = build_user_question('Task_13_Next_Action_Prediction', fields13)
                meta13 = self._meta('TypeA', source_path or '', 'Task_13_Next_Action_Prediction', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_13_Next_Action_Prediction'].append({"images": base_img, "prompt": p14, "user_q": user_q14, "meta": meta13, "fields": fields13})

                # Task 15: Failure Recovery（加入全局候选池）
                fh = step.get('failure_handling', {})
                if isinstance(fh, dict):
                    fields15 = {"reason": fh.get('reason'), "strategy": fh.get('recovery_strategy'), "step_goal": step_goal, "high_level_goal": global_goal}
                    p15 = build_answer_prompt('Task_15_Failure_Recovery_Protocol', fields15)
                    user_q15 = build_user_question('Task_15_Failure_Recovery_Protocol', fields15)
                    meta15 = self._meta('TypeA', source_path or '', 'Task_15_Failure_Recovery_Protocol', i+1, -1, 0, step_goal, base_img)
                    self.global_candidates_A['Task_15_Failure_Recovery_Protocol'].append({"images": base_img, "prompt": p15, "user_q": user_q15, "meta": meta15, "fields": fields15})

                # --- Frame Level：收集候选（全局随机采样，避免总取首帧/首关系）---
                for frame in critical_frames:
                    img = frame.get('keyframe_image_path', 'missing.jpg')
                    logger.info(f"  -> 关键帧: {img}")

                    # Task 3: Micro Hotspot → 加入全局候选池
                    ah = frame.get('affordance_hotspot', {})
                    fields2 = {"aff_type": ah.get('affordance_type'), "description": ah.get('description'), "causal_role": ah.get('causal_role'), "step_goal": step_goal, "high_level_goal": global_goal}
                    p2 = build_answer_prompt('Task_03_Micro_Affordance_Visual_Semantics', fields2)
                    user_q2 = build_user_question('Task_03_Micro_Affordance_Visual_Semantics', fields2)
                    meta2 = self._meta('TypeA', source_path or '', 'Task_03_Micro_Affordance_Visual_Semantics', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if self._accept_balance('Task_03_Micro_Affordance_Visual_Semantics', balance_counter):
                        self.global_candidates_A['Task_03_Micro_Affordance_Visual_Semantics'].append({"images": img, "prompt": p2, "user_q": user_q2, "meta": meta2, "fields": fields2})

                    # Task 4: Causal Chain → 加入全局候选池
                    cc = frame.get('causal_chain', {})
                    fields4 = {
                        "agent": cc.get('agent'), "action": cc.get('action'), "patient": cc.get('patient'),
                        "aff_pre": frame.get('affordance_preconditions', []),
                        "mechanism": cc.get('causal_affordance_focus_detail'),
                        "spatial": cc.get('causal_spatial_precondition'),
                        "eff_pat": cc.get('causal_effect_on_patient'), "eff_env": cc.get('causal_effect_on_environment'),
                        "high_level_goal": global_goal,
                    }
                    p4 = build_answer_prompt('Task_06_Holistic_Causal_Chain_Analysis', fields4)
                    user_q4 = build_user_question('Task_06_Holistic_Causal_Chain_Analysis', fields4)
                    meta4 = self._meta('TypeA', source_path or '', 'Task_06_Holistic_Causal_Chain_Analysis', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if self._accept_balance('Task_06_Holistic_Causal_Chain_Analysis', balance_counter):
                        self.global_candidates_A['Task_06_Holistic_Causal_Chain_Analysis'].append({"images": img, "prompt": p4, "user_q": user_q4, "meta": meta4, "fields": fields4})

                    # Task 5: State Evolution → 加入全局候选池
                    fields5 = {"action_desc": frame.get('action_description'), "state_change": frame.get('state_change_description'), "high_level_goal": global_goal}
                    p5 = build_answer_prompt('Task_05_State_Evolution_Description', fields5)
                    user_q5 = build_user_question('Task_05_State_Evolution_Description', fields5)
                    meta5 = self._meta('TypeA', source_path or '', 'Task_05_State_Evolution_Description', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if self._accept_balance('Task_05_State_Evolution_Description', balance_counter):
                        self.global_candidates_A['Task_05_State_Evolution_Description'].append({"images": img, "prompt": p5, "user_q": user_q5, "meta": meta5, "fields": fields5})

                    # Task 6: Transient Geometric Verification → 加入全局候选池（每个关系都加入，由全局采样挑选）
                    sp_list = frame.get('spatial_preconditions', []) or []
                    for j, sp in enumerate(sp_list):
                        objs = sp.get('objects')
                        relation = sp.get('relation')
                        objs_strs: List[str] = []
                        if isinstance(objs, list):
                            for o in objs:
                                objs_strs.append(o if isinstance(o, str) else str(o))
                        if len(objs_strs) >= 2 and isinstance(relation, str):
                            fields6 = {"obj_a": objs_strs[0], "obj_b": objs_strs[1], "relation": relation, "high_level_goal": global_goal}
                            p6 = build_answer_prompt('Task_02_Transient_Geometric_Verification', fields6)
                            user_q6 = build_user_question('Task_02_Transient_Geometric_Verification', fields6)
                            meta6 = self._meta('TypeA', source_path or '', 'Task_02_Transient_Geometric_Verification', i+1, frame.get('frame_index') or 0, j+1, step_goal, img)
                            if self._accept_balance('Task_02_Transient_Geometric_Verification', balance_counter):
                                self.global_candidates_A['Task_02_Transient_Geometric_Verification'].append({"images": img, "prompt": p6, "user_q": user_q6, "meta": meta6, "fields": fields6})
                        elif isinstance(relation, str) and relation.strip():
                            joined_objs = ", ".join(objs_strs) if objs_strs else "N/A"
                            fields6f = {"objects": joined_objs, "relation": relation, "high_level_goal": global_goal}
                            p6_fallback = build_answer_prompt('Task_02_Transient_Geometric_Verification', fields6f)
                            user_q6f = build_user_question('Task_02_Transient_Geometric_Verification', {"obj_a": joined_objs, "obj_b": "", "relation": relation, "high_level_goal": global_goal})
                            meta6f = self._meta('TypeA', source_path or '', 'Task_02_Transient_Geometric_Verification', i+1, frame.get('frame_index') or 0, j+1, step_goal, img)
                            if self._accept_balance('Task_02_Transient_Geometric_Verification', balance_counter):
                                self.global_candidates_A['Task_02_Transient_Geometric_Verification'].append({"images": img, "prompt": p6_fallback, "user_q": user_q6f, "meta": meta6f, "fields": fields6f})

                    # Task 16: Physical Feasibility → 加入全局候选池
                    sp_pre = frame.get('spatial_preconditions', [])
                    af_pre = frame.get('affordance_preconditions', [])
                    if (sp_pre and len(sp_pre) > 0) or (af_pre and len(af_pre) > 0):
                        fields16 = {"step_goal": step_goal, "spatial_preconditions": sp_pre, "affordance_preconditions": af_pre, "high_level_goal": global_goal}
                        p16 = build_answer_prompt('Task_16_Physical_Feasibility_Verification', fields16)
                        user_q16 = build_user_question('Task_16_Physical_Feasibility_Verification', fields16)
                        meta16 = self._meta('TypeA', source_path or '', 'Task_16_Physical_Feasibility_Verification', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                        if self._accept_balance('Task_16_Physical_Feasibility_Verification', balance_counter):
                            self.global_candidates_A['Task_16_Physical_Feasibility_Verification'].append({"images": img, "prompt": p16, "user_q": user_q16, "meta": meta16, "fields": fields16})

                    # Task 17: Holistic Synthesis → 加入全局候选池
                    cc2 = frame.get('causal_chain', {})
                    mech = cc2.get('causal_affordance_focus_detail') if isinstance(cc2, dict) else None
                    if isinstance(mech, str) and mech.strip():
                        fields17 = {"step_goal": step_goal, "rationale": step.get('rationale', ''), "mechanism": mech, "high_level_goal": global_goal}
                        p17 = build_answer_prompt('Task_17_Holistic_Step_Synthesis_Why_How', fields17)
                        user_q17 = build_user_question('Task_17_Holistic_Step_Synthesis_Why_How', fields17)
                        meta17 = self._meta('TypeA', source_path or '', 'Task_17_Holistic_Step_Synthesis_Why_How', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                        if self._accept_balance('Task_17_Holistic_Step_Synthesis_Why_How', balance_counter):
                            self.global_candidates_A['Task_17_Holistic_Step_Synthesis_Why_How'].append({"images": img, "prompt": p17, "user_q": user_q17, "meta": meta17, "fields": fields17})

                # Task 11: Expected Physical Effects（TypeA 收集步级候选，最终全局采样一次）
                try:
                    fields11_step = {
                        "expected_effects": step.get('expected_effects', []),
                        "spatial_post": step.get('spatial_postconditions_detail', []),
                        "affordance_post": step.get('affordance_postconditions_detail', []),
                        "step_goal": step_goal,
                        "high_level_goal": global_goal,
                    }
                    p11s = build_answer_prompt('Task_11_Expected_Physical_Effects', fields11_step)
                    user_q11s = build_user_question('Task_11_Expected_Physical_Effects', fields11_step)
                    meta11s = self._meta('TypeA', source_path or '', 'Task_11_Expected_Physical_Effects', i+1, -1, 0, step_goal, base_img)
                    self.global_candidates_A['Task_11_Expected_Physical_Effects'].append({"images": base_img, "prompt": p11s, "user_q": user_q11s, "meta": meta11s, "fields": fields11_step})
                except Exception as e:
                    logger.warning(f"[TypeA] Task 11 步级候选收集失败: {e}")

            except Exception as e:
                logger.error(f"Error processing Type A step: {e}", exc_info=True)

        # --- 全局候选采样并发射（TypeA）---
        try:
            rng = random.Random(hash(source_path or ''))
            for tname, cap in self.global_caps_A.items():
                candidates = self.global_candidates_A.get(tname, [])
                if not candidates:
                    continue
                pick = rng.sample(candidates, k=min(cap, len(candidates)))
                for cand in pick:
                    self.emit_sample(tname, cand["images"], cand["prompt"], cand["user_q"], cand["meta"], resume=self.resume, force=self.force, fields=cand.get("fields"))
            # 清空候选池，避免影响后续条目
            self.global_candidates_A = {k: [] for k in self.global_caps_A.keys()}
        except Exception as e:
            logger.warning(f"[TypeA] 全局候选采样失败: {e}")

    # =========================================================================
    # Type B 处理逻辑（多帧）
    # =========================================================================
    def _process_type_b(self, data: Dict, source_path: Optional[str] = None):
        global_goal = data.get('high_level_goal', 'Unknown Goal')
        frames_dir = data.get('sample_frames_dir', '')

        scene_images: List[str] = []
        if isinstance(frames_dir, str) and os.path.exists(frames_dir):
            try:
                # 自然命名排序（如 img2.jpg 在 img10.jpg 之前）
                file_names = [
                    f for f in os.listdir(frames_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ]
                file_names.sort(key=_natural_key)
                scene_images = [os.path.join(frames_dir, f) for f in file_names]
            except Exception:
                # 回退为普通排序
                all_files = sorted([
                    os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ])
                scene_images = all_files
        if not scene_images:
            scene_images = ["missing_scene_context.jpg"]

        try:
            # Task 1: Macro Anchor (QA)
            fields1 = {"scene_desc": data.get('scene_description'), "key_objects": data.get('key_objects_for_planning')}
            p1 = build_answer_prompt('Task_01_Macro_Anchor_Extraction', fields1)
            user_q1 = build_user_question('Task_01_Macro_Anchor_Extraction', fields1)
            meta1 = self._meta('TypeB', source_path or '', 'Task_01_Macro_Anchor_Extraction', 0, -1, 0, None, None)
            self.emit_sample('Task_01_Macro_Anchor_Extraction', scene_images, p1, user_q1, meta1, resume=self.resume, force=self.force, fields=fields1)

            # Task 7: Scene Goal (QA)
            fields7 = {"scene_desc": data.get('scene_description'), "high_level_goal": global_goal}
            p7 = build_answer_prompt('Task_07_Scene_Goal_Derivation', fields7)
            user_q7 = build_user_question('Task_07_Scene_Goal_Derivation', fields7)
            meta7 = self._meta('TypeB', source_path or '', 'Task_07_Scene_Goal_Derivation', 0, -1, 0, None, None)
            self.emit_sample('Task_07_Scene_Goal_Derivation', scene_images, p7, user_q7, meta7, resume=self.resume, force=self.force, fields=fields7)

            steps = data.get('steps', [])
            for i, step in enumerate(steps):
                step_goal = step.get('step_goal')
                logger.info(f">>> [INFO] TypeB Step {i+1}/{len(steps)}: {step_goal}")

                # Task 10: Step Execution（加入候选池）
                fields8 = {"global_goal": global_goal, "step_goal": step_goal, "actions": step.get('navigation_and_manipulation')}
                p8 = build_answer_prompt('Task_10_Step_Execution_Statement', fields8)
                user_q8 = build_user_question('Task_10_Step_Execution_Statement', fields8)
                meta8 = self._meta('TypeB', source_path or '', 'Task_10_Step_Execution_Statement', i+1, -1, 0, step_goal, None)
                self.global_candidates_B['Task_10_Step_Execution_Statement'].append({"images": scene_images, "prompt": p8, "user_q": user_q8, "meta": meta8, "fields": fields8})

                # Task 8: Rationale（加入候选池）
                fields9 = {"global_goal": global_goal, "step_goal": step_goal, "rationale": step.get('rationale')}
                p9 = build_answer_prompt('Task_08_Strategic_Rationale_Justification', fields9)
                user_q9 = build_user_question('Task_08_Strategic_Rationale_Justification', fields9)
                meta9 = self._meta('TypeB', source_path or '', 'Task_08_Strategic_Rationale_Justification', i+1, -1, 0, step_goal, None)
                self.global_candidates_B['Task_08_Strategic_Rationale_Justification'].append({"images": scene_images, "prompt": p9, "user_q": user_q9, "meta": meta9, "fields": fields9})

                # Task 9: Preconditions（加入候选池）
                fields10 = {"step_goal": step_goal, "preconditions": step.get('preconditions'), "high_level_goal": global_goal}
                p10 = build_answer_prompt('Task_09_Precondition_Statement', fields10)
                user_q10 = build_user_question('Task_09_Precondition_Statement', fields10)
                meta10 = self._meta('TypeB', source_path or '', 'Task_09_Precondition_Statement', i+1, -1, 0, step_goal, None)
                self.global_candidates_B['Task_09_Precondition_Statement'].append({"images": scene_images, "prompt": p10, "user_q": user_q10, "meta": meta10, "fields": fields10})

                # Task 11: Expected Effects（加入候选池）
                fields11 = {
                    "expected_effects": step.get('expected_effects'),
                    "spatial_post": step.get('spatial_postconditions_detail'),
                    "affordance_post": step.get('affordance_postconditions_detail'),
                    "step_goal": step_goal,
                    "high_level_goal": global_goal,
                }
                p11 = build_answer_prompt('Task_11_Expected_Physical_Effects', fields11)
                user_q11 = build_user_question('Task_11_Expected_Physical_Effects', fields11)
                meta11 = self._meta('TypeB', source_path or '', 'Task_11_Expected_Physical_Effects', i+1, -1, 0, step_goal, None)
                self.global_candidates_B['Task_11_Expected_Physical_Effects'].append({"images": scene_images, "prompt": p11, "user_q": user_q11, "meta": meta11, "fields": fields11})

                # Task 12: Dependency（仅在存在真实依赖时生成）
                if i < len(steps) - 1:
                    next_step = steps[i+1]
                    if self._has_dependency(step, next_step):
                        fields12 = {"step_n_goal": step_goal, "step_n_effect": step.get('expected_effects'), "step_next_goal": next_step.get('step_goal'), "step_next_precondition": next_step.get('preconditions')}
                        p12 = build_answer_prompt('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        user_q12 = build_user_question('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        meta12 = self._meta('TypeB', source_path or '', 'Task_12_Inter_Step_Dependency_Analysis', i+1, -1, 0, step_goal, None)
                        self.global_candidates_B['Task_12_Inter_Step_Dependency_Analysis'].append({"images": scene_images, "prompt": p12, "user_q": user_q12, "meta": meta12, "fields": fields12})

                # Task 15: Failure Recovery（QA，兼容旧格式）
                fh = step.get('failure_handling', {})
                if isinstance(fh, dict):
                    fields15 = {"reason": fh.get('reason'), "strategy": fh.get('recovery_strategy'), "step_goal": step_goal}
                    p15 = build_answer_prompt('Task_15_Failure_Recovery_Protocol', fields15)
                    user_q15 = build_user_question('Task_15_Failure_Recovery_Protocol', fields15)
                    meta15 = self._meta('TypeB', source_path or '', 'Task_15_Failure_Recovery_Protocol', i+1, -1, 0, step_goal, None)
                    self.global_candidates_B['Task_15_Failure_Recovery_Protocol'].append({"images": scene_images, "prompt": p15, "user_q": user_q15, "meta": meta15, "fields": fields15})
                elif isinstance(fh, list):
                    for item in fh:
                        reason = "failure"
                        strategy = str(item)
                        if isinstance(item, str) and ":" in item:
                            reason, strategy = item.split(":", 1)
                        fields15b = {"reason": reason.strip(), "strategy": strategy.strip(), "step_goal": step_goal}
                        p15b = build_answer_prompt('Task_15_Failure_Recovery_Protocol', fields15b)
                        user_q15b = build_user_question('Task_15_Failure_Recovery_Protocol', fields15b)
                        meta15b = self._meta('TypeB', source_path or '', 'Task_15_Failure_Recovery_Protocol', i+1, -1, 1, step_goal, None)
                        self.global_candidates_B['Task_15_Failure_Recovery_Protocol'].append({"images": scene_images, "prompt": p15b, "user_q": user_q15b, "meta": meta15b, "fields": fields15b})

            # --- 全局候选采样并发射（TypeB）---
            try:
                rng = random.Random(hash(source_path or ''))
                for tname, cap in self.global_caps_B.items():
                    candidates = self.global_candidates_B.get(tname, [])
                    if not candidates:
                        continue
                    pick = rng.sample(candidates, k=min(cap, len(candidates)))
                    for cand in pick:
                        self.emit_sample(tname, cand["images"], cand["prompt"], cand["user_q"], cand["meta"], resume=self.resume, force=self.force, fields=cand.get("fields"))
                # 清空候选池
                self.global_candidates_B = {k: [] for k in self.global_caps_B.keys()}
            except Exception as e:
                logger.warning(f"[TypeB] 全局候选采样失败: {e}")
        except Exception as e:
            logger.error(f"Error processing Type B: {e}", exc_info=True)

    # =========================================================================
    # 公共：类型判断并路由
    # =========================================================================
    def process_entry(self, raw_data: Dict, source_path: Optional[str] = None):
        is_type_b = "scene_description" in raw_data and "sample_frames_dir" in raw_data
        is_type_a = "steps" in raw_data and any("critical_frames" in s for s in raw_data.get("steps", []))
        if is_type_b:
            self._process_type_b(raw_data, source_path)
        elif is_type_a:
            self._process_type_a(raw_data, source_path)
        else:
            logger.warning("Skipping entry: Unknown data format.")

    def flush_to_disk(self):
        # 输出合并 JSON：count + A/B 分开数组（终局写一次，确保完整性）
        logger.info("Writing combined JSON outputs per task...")
        for task_name in TASK_PROMPTS.keys():
            try:
                self._write_combined(task_name)
            except Exception as e:
                logger.warning(f"[CombinedWarn] 写入 {task_name} 失败: {e}")
        total = sum(len(v) for v in self.data_buffer.values())
        per_task = {k: len(v) for k, v in self.data_buffer.items() if v}
        logger.info(f"Summary: total buffered={total}, per-task={per_task}")


# ==============================================================================
# 5) 批处理入口（与 generate_phyplan.py 一致，增加每项 Start/Done）
# ==============================================================================

def load_json(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON {path}: {e}")
        return {}


def _natural_key(s: str):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def scan_type_a_root(root: str) -> List[str]:
    results: List[str] = []
    if not os.path.isdir(root):
        return results
    names = sorted(os.listdir(root), key=_natural_key)
    for name in names:
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        p_key = os.path.join(sub, 'causal_plan_with_keyframes.json')
        p_base = os.path.join(sub, 'causal_plan.json')
        if os.path.exists(p_key):
            results.append(p_key)
        elif os.path.exists(p_base):
            results.append(p_base)
    return results


def scan_type_b_root(root: str) -> List[str]:
    results: List[str] = []
    if not os.path.isdir(root):
        return results
    names = sorted(os.listdir(root), key=_natural_key)
    for name in names:
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        p = os.path.join(sub, 'plan.json')
        if os.path.exists(p):
            results.append(p)
    return results


# ======================= 断点进度管理（逐项级别） ============================
def _progress_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "_progress")

def _progress_file(output_dir: str, kind: str) -> str:
    return os.path.join(_progress_dir(output_dir), f"{kind}_done.jsonl")

def load_progress(output_dir: str) -> Dict[str, set]:
    os.makedirs(_progress_dir(output_dir), exist_ok=True)
    res = {"typeA": set(), "typeB": set()}
    for kind in ("typeA", "typeB"):
        p = _progress_file(output_dir, kind)
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        path = obj.get("path")
                        if isinstance(path, str) and path:
                            res[kind].add(path)
                    except Exception:
                        continue
        except Exception:
            continue
    return res

def mark_progress(output_dir: str, kind: str, path: str):
    try:
        os.makedirs(_progress_dir(output_dir), exist_ok=True)
        pf = _progress_file(output_dir, kind)
        with open(pf, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"path": path, "ts": time.time()}, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[ProgressWarn] 记录进度失败: kind={kind}, path={path}, err={e}")

def clear_progress(output_dir: str):
    try:
        pd = _progress_dir(output_dir)
        if os.path.isdir(pd):
            for name in os.listdir(pd):
                fp = os.path.join(pd, name)
                try:
                    os.remove(fp)
                except Exception:
                    pass
            logger.info(f"[Progress] 已清空进度文件夹: {pd}")
    except Exception as e:
        logger.warning(f"[ProgressWarn] 清空进度失败: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate PhyPlan tasks via direct model API (no sglang/qwen3vl).')
    parser.add_argument('--typeA-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long', help='Root folder for TypeA data')
    parser.add_argument('--typeB-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/generated_plans_output_high_videos', help='Root folder for TypeB data')
    parser.add_argument('--output-dir', default='/e2e-data/evad-tech-vla/luzheng/ICML/phyplan_output_api', help='Output directory for generated tasks')
    parser.add_argument('--limit', type=int, default=0, help='Optional limit on number of items per dataset')
    parser.add_argument('--types', default='A,B', help='Which data types to process: A, B, or A,B')
    parser.add_argument('--stream-save', action='store_true', default=True, help='Enable streaming save (append on each sample)')
    parser.add_argument('--live-combined', action='store_true', default=True, help='Write combined data.json (count + A/B arrays) on every save')
    # 断点重续
    parser.add_argument('--resume', action='store_true', default=True, help='Enable resume: skip items already processed')
    parser.add_argument('--restart', action='store_true', default=False, help='Clear progress files and start over')
    parser.add_argument('--force', action='store_true', default=False, help='Force reprocess even if marked done')
    # API 覆盖项（可选）
    parser.add_argument('--api-key', default=os.environ.get('API_KEY', 'sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s'))
    parser.add_argument('--api-base', default=os.environ.get('API_BASE_URL', 'http://model.mify.ai.srv/v1'))
    parser.add_argument('--provider', default=os.environ.get('MODEL_PROVIDER_ID', 'vertex_ai'))
    parser.add_argument('--model', default=os.environ.get('MODEL_NAME', 'gemini-3-pro-preview'))
    parser.add_argument('--max-images', type=int, default=int(os.environ.get('REQUEST_IMAGES_LIMIT', '1000000')))
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('MAX_TOKENS', '8192')))
    args = parser.parse_args()

    api_cfg = ApiConfig(
        api_key=args.api_key,
        api_base_url=args.api_base,
        model_provider_id=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        request_images_limit=args.max_images,
    )

    # 加载已写入的样本级键（用于断点重续）
    def key_from_meta(meta: Dict[str, Any]) -> str:
        return f"{meta.get('item_type','')}|{meta.get('source_path','')}|{meta.get('task_name','')}|{int(meta.get('step_index',-1))}|{int(meta.get('frame_index',-1))}|{int(meta.get('sub_index',0))}"

    def load_existing_sample_keys(output_dir: str) -> set:
        keys = set()
        try:
            for task_name in TASK_PROMPTS.keys():
                p = os.path.join(output_dir, task_name, 'data.jsonl')
                if not os.path.exists(p):
                    continue
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            meta = obj.get('meta', {})
                            if meta:
                                keys.add(key_from_meta(meta))
                        except Exception:
                            continue
        except Exception:
            pass
        return keys

    existing_keys = set()
    if args.resume and not args.restart:
        existing_keys = load_existing_sample_keys(args.output_dir)
        logging.info(f"[Resume] 已加载样本级进度键：{len(existing_keys)} 条")

    generator = PhyPlanAPIGenerator(output_dir=args.output_dir, api_config=api_cfg, stream_save=args.stream_save, resume=args.resume, force=args.force, processed_keys=existing_keys, live_combined=args.live_combined)
    # 预加载历史样本，保证实时合并包含历史数据
    try:
        generator.hydrate_existing_buffers()
    except Exception as e:
        logging.warning(f"[HydrateWarn] preload failed: {e}")

    a_items = scan_type_a_root(args.typeA_root)
    b_items = scan_type_b_root(args.typeB_root)

    if args.limit and args.limit > 0:
        a_items = a_items[:args.limit]
        b_items = b_items[:args.limit]

    logging.info(f"Discovered TypeA items: {len(a_items)} from {args.typeA_root}")
    logging.info(f"Discovered TypeB items: {len(b_items)} from {args.typeB_root}")

    # 标准化为绝对路径，确保进度文件与跳过判断稳定
    a_items = [os.path.abspath(p) for p in a_items]
    b_items = [os.path.abspath(p) for p in b_items]

    # 断点重续控制
    prog = load_progress(args.output_dir)
    if args.restart:
        clear_progress(args.output_dir)
        prog = {"typeA": set(), "typeB": set()}

    process_types = set([t.strip().upper() for t in args.types.split(',') if t.strip()])

    if 'A' in process_types:
        for jp in a_items:
            if args.resume and not args.force and jp in prog.get('typeA', set()):
                logging.info(f"[Skip ] TypeA item 已处理: {jp}")
                continue
            logging.info(f"[Start] TypeA item: {jp}")
            data = load_json(jp)
            if not data:
                continue
            generator.process_entry(data, source_path=jp)
            logging.info(f"[Done ] TypeA item: {jp}")
            mark_progress(args.output_dir, 'typeA', jp)

    if 'B' in process_types:
        for jp in b_items:
            if args.resume and not args.force and jp in prog.get('typeB', set()):
                logging.info(f"[Skip ] TypeB item 已处理: {jp}")
                continue
            logging.info(f"[Start] TypeB item: {jp}")
            data = load_json(jp)
            if not data:
                continue
            sdir = data.get('sample_frames_dir')
            if isinstance(sdir, str):
                data['sample_frames_dir'] = os.path.abspath(sdir)
            generator.process_entry(data, source_path=jp)
            logging.info(f"[Done ] TypeB item: {jp}")
            mark_progress(args.output_dir, 'typeB', jp)

    generator.flush_to_disk()
