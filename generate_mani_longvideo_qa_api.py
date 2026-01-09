# -*- coding: utf-8 -*-
"""Mani-LongVideo 多任务 QA 生成（API 版，带进度日志）

目标：
- 任务与生成逻辑尽可能严格对齐 `generate_phyplan_api.py`。
- 仅差异：
  1) 任务集合与 prompt（对齐 `mani_longvideo_tasks_plan.md` 的 Task_01 ~ Task_17）
  2) 输入数据字段 schema（对齐 mani_longvideo 产出的 plan JSON）
- 仅处理 TypeA（新 mani-longvideo schema）。

输出：
- ShareGPT JSONL（每任务目录 `data.jsonl`）
- 同步合并 `data.json`（count + A/B，两段；本脚本默认写入 A 段）

注意：
- 与 `generate_phyplan_api.py` 一致：默认 fields-only，不把图片内容传入模型；仅在样本里保留 image path。
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1) 模型 API 配置（与 generate_phyplan_api.py 对齐）
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
    max_retries: int = int(os.environ.get('MAX_RETRIES', '3'))
    retry_backoff_sec: float = float(os.environ.get('RETRY_BACKOFF_SEC', '1.5'))


def initialize_api_client(cfg: ApiConfig):
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base_url,
            default_headers={"X-Model-Provider-Id": cfg.model_provider_id},
        )
        logger.info(">>> [INFO] OpenAI 兼容客户端初始化成功")
        return client
    except Exception as e:
        logger.warning(f"初始化 OpenAI 客户端失败：{e}. 将回退为本地占位处理。")
        return None


# ==============================================================================
# 2) 提示模板（对齐 mani_longvideo_tasks_plan.md 的 17 类）
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


TASK_PROMPTS: Dict[str, str] = {
    # ==========================================
    # 支柱一：感知与锚点 (Perception & Anchoring)
    # ==========================================
    "Task_01_Macro_Anchor_Extraction": """Input Data:
High-Level Goal: {high_level_goal}
Key Objects: {key_objects}
Affordance Tags (Optional): {affordance_tags}

Instruction: The user asks: "Based on the scene, which objects are the key interactable anchors for planning?"
Draft a response that lists the items from `Key Objects` and integrates them into a complete sentence. Ensure the list is exhaustive based on the input.""",

    # ==========================================
    # 支柱二：物理动力学 (Physical Dynamics)
    # ==========================================
    "Task_02_Transient_Geometric_Verification": """Input Data:
Spatial Relation: {relation}
Objects: {objects}
Truth Value: {truth}

Instruction: The user asks: "What is the precise spatial relationship between these objects in this keyframe?"
State the relationship strictly using the provided relation. If `Truth Value` is false, explicitly state that the relation does NOT hold.""",

    "Task_03_Micro_Affordance_Visual_Semantics": """Input Data:
Affordance Type: {aff_type}
Hotspot Description: {desc}
Mechanism: {mechanism}

Instruction: The user asks: "Identify the specific region involved and its functional role."
Synthesize the `Hotspot Description`, `Affordance Type`, and `Mechanism` into a rigorous descriptive statement.""",

    "Task_04_Entity_Role_Identification": """Input Data:
Tools: {tools}
Materials: {materials}

Instruction: The user asks: "Distinguish between the active tools and the passive materials in this interaction."
Formulate a sentence that clearly assigns roles.""",

    "Task_05_State_Evolution_Description": """Input Data:
Ongoing Action: {action_desc}
Resulting State Change: {state_change}

Instruction: The user asks: "Describe the ongoing action and the immediate resulting state change."
Combine these two fields into a single cause-and-effect statement.""",

    "Task_06_Holistic_Causal_Chain_Analysis": """Input Data:
Agent: {agent}
Action: {action}
Patient: {patient}
Physical Basis (Affordance Preconditions): {aff_pre}
Mechanism (Hotspot): {mechanism}
Spatial Condition: {spatial}
Effect on Patient: {eff_pat}
Effect on Environment: {eff_env}

Instruction: The user asks: "Analyze the complete physical causal chain driving this interaction."
Produce two distinct paragraphs.
Paragraph 1: Describe the interaction and the spatial/affordance setup.
Paragraph 2: Describe the hotspot mechanism and the immediate effects on the patient and environment.""",

    "Task_07_Scene_Goal_Derivation": """Input Data:
High-Level Goal: {high_level_goal}

Instruction: The user asks: "Given the scene context, what is the logical high-level goal?"
State the goal in one complete sentence grounded in the provided High-Level Goal.""",

    "Task_08_Strategic_Rationale_Justification": """Input Data:
Global Goal: {global_goal}
Step Goal: {step_goal}
Rationale: {rationale}

Instruction: The user asks: "Why is this specific step necessary within the global plan?"
Provide the justification using the `Rationale` and explicitly link it to the Global Goal.""",

    "Task_09_Precondition_Statement": """Input Data:
Step Goal: {step_goal}
Preconditions: {preconditions}

Instruction: The user asks: "What mandatory preconditions must be met before initiating this step?"
Present the `Preconditions` as a requirement statement in a natural paragraph.""",

    "Task_10_Step_Execution_Statement": """Input Data:
Step Goal: {step_goal}
Execution Actions: {actions}

Instruction: The user asks: "Detail the specific execution actions required for this step."
Convert the `Execution Actions` into a procedural description. If `Actions` are empty, use the `Step Goal` as the description.""",

    "Task_11_Expected_Physical_Effects": """Input Data:
Expected Effects: {macro_eff}

Instruction: The user asks: "What are the expected physical outcomes and final states after this step?"
Summarize all expected effects in a cohesive paragraph; emphasize spatial and state/affordance outcomes when present.""",

    "Task_12_Inter_Step_Dependency_Analysis": """Input Data:
Step N Goal: {step_n_goal}
Step N Effect: {step_n_effect}
Step N+1 Goal: {step_next_goal}
Step N+1 Precondition: {step_next_precondition}

Instruction: The user asks: "Analyze the logical dependency between the previous step and the current step."
Explain how the previous output allows the current input using only the given phrases.""",

    "Task_13_Next_Action_Prediction": """Input Data:
Next Step Goal: {next_step_goal}

Instruction: The user asks: "Given the current state, what is the next planned action?"
Answer by stating the Next Step Goal as the planned next action, without adding extra micro-actions.""",

    "Task_14_Counterfactual_Prediction": """Input Data:
Challenge Question: {question}
Expected Outcome: {outcome}

Instruction: The user asks: "{question}"
Provide the predicted physical consequence based on the `Expected Outcome`.""",

    "Task_15_Failure_Recovery_Protocol": """Input Data:
Failure Reason: {reason}
Recovery Strategy: {strategy}

Instruction: The user asks: "If the action fails due to the described reason, what is the protocol?"
Synthesize a recovery directive grounded in the given fields.""",

    "Task_16_Physical_Feasibility_Verification": """Input Data:
Step Goal: {step_goal}
Spatial Preconditions: {spatial_preconditions}
Affordance Preconditions: {affordance_preconditions}

Instruction: The user asks: "Verify the spatial and affordance-based conditions required for this interaction."
Produce two distinct paragraphs.
Paragraph 1 (Spatial): Detail the required spatial relationships.
Paragraph 2 (Affordance): Detail the required object properties and reasons.""",

    "Task_17_Holistic_Step_Synthesis_Why_How": """Input Data:
Step Goal: {step_goal}
Strategic Rationale: {rationale}
Physical Mechanism: {mechanism}
Causal Chain: {causal_chain}
Spatial Preconditions: {spatial}

Instruction: The user asks: "Explain both the strategic reasoning ('Why') and the physical mechanism ('How') for this step."
Produce two distinct paragraphs: (1) strategy grounded in rationale; (2) mechanism grounded in the physical mechanism, spatial preconditions, and causal chain.""",
}


# ==============================================================================
# 2.1) QA 生成辅助：根据任务生成英文疑问句与回答提示（对齐 generate_phyplan_api.py）
# ==============================================================================


def _strip_quotes_punct(s: Optional[str]) -> str:
    try:
        import re

        t = (s or "").strip()
        t = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", t)
        t = re.sub(r"[\.;:!?]+$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t
    except Exception:
        return s or ""


def _finalize_question(q: str) -> str:
    try:
        import re

        s = (q or "").strip()
        s = re.sub(r"\s+([,;:])", r"\1", s)
        s = re.sub(r"([,;:])\s+", r"\1 ", s)
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

    if task_name == 'Task_01_Macro_Anchor_Extraction':
        return _finalize_question("Based on the scene, which objects are the key interactable anchors for planning")
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
        q = f"Given the overall goal {hl or 'the mission'}, what is the next planned action after the step {sg or 'this step'}"
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
        objects = fields.get('objects')
        a = b = None
        if isinstance(objects, list) and objects:
            a = objects[0] if len(objects) >= 1 else None
            b = objects[1] if len(objects) >= 2 else None
        a = _strip_quotes_punct(a or 'the first object')
        b = _strip_quotes_punct(b or 'the second object')
        q = f"What is the precise spatial relationship between {a} and {b} in this keyframe"
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
    if task_name == 'Task_10_Step_Execution_Statement':
        q = f"Given the overall goal {hl or 'the mission'}, for the step {sg or 'this step'}, what specific execution actions are required"
        return _finalize_question(q)
    if task_name == 'Task_07_Scene_Goal_Derivation':
        return _finalize_question("Given the scene context, what is the logical high-level goal")

    return _finalize_question("Could you provide a concise, grounded answer based on the provided context")


def build_answer_prompt(task_name: str, fields: Dict[str, Any]) -> str:
    ctx = json.dumps(fields, ensure_ascii=False)
    tn = task_name
    if tn == 'Task_01_Macro_Anchor_Extraction':
        instr = (
            "Provide a natural English listing of the task‑relevant anchors from the context, with items inline (commas or semicolons). Do not use bullets, commentary, or decorative phrasing."
        )
    elif tn == 'Task_02_Transient_Geometric_Verification':
        instr = (
            "Describe the precise spatial relationship using only the given relation and object names. If the truth value is false, explicitly state that the relation does not hold. Do not invent details; do not use bullets; avoid filler."
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
            "Summarize the expected effects in a natural paragraph without bullets. Cover all effects inline (separated naturally). Do not limit length; avoid filler or decoration."
        )
    elif tn == 'Task_12_Inter_Step_Dependency_Analysis':
        instr = (
            "Explain how a specific effect from the previous step satisfies a specific precondition of the next step, using only the given phrases. Do not use bullets; avoid filler; do not limit length."
        )
    elif tn == 'Task_13_Next_Action_Prediction':
        instr = (
            "State the next planned action using only the Next Step Goal. Do not add any micro-actions or extra details."
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
            "Write two natural paragraphs: (1) why the step is necessary (rationale); (2) how it is physically achieved (mechanism), grounded in causal chain and spatial evidence. Do not limit length; avoid filler or decorative language."
        )
    else:
        instr = "Answer concisely and objectively using only the given fields."

    return "Context (Fields):\n" + ctx + "\n\n" + "Instruction: " + instr + " Return only the final answer text."


# ==============================================================================
# 3) 工具：读取图片为 base64 并构造 image_url 内容（保持与原脚本一致，当前默认不传图）
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
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        count += 1
    return contents


# ==============================================================================
# 4) 主类：API 版本生成器（逻辑与 generate_phyplan_api.py 对齐，仅 TypeA）
# ==============================================================================


class ManiLongVideoAPIGenerator:
    def __init__(
        self,
        output_dir: str = "mani_longvideo_qa_output_api",
        api_config: ApiConfig = None,
        stream_save: bool = True,
        resume: bool = True,
        force: bool = False,
        processed_keys: Optional[set] = None,
        live_combined: bool = True,
        balanced: bool = True,
        strict: bool = True,
        full_relations: bool = False,
        min_output_chars: int = 24,
    ):
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
        self.balanced = balanced
        self.strict = strict
        self.full_relations = full_relations
        self.min_output_chars = min_output_chars

        # 平衡策略（对齐 generate_phyplan_api.py）
        self.balance_caps = {
            'Task_03_Micro_Affordance_Visual_Semantics': 1,
            'Task_05_State_Evolution_Description': 1,
            'Task_06_Holistic_Causal_Chain_Analysis': 1,
            'Task_02_Transient_Geometric_Verification': 1,
            'Task_16_Physical_Feasibility_Verification': 1,
            'Task_17_Holistic_Step_Synthesis_Why_How': 1,
        }

        # 全局候选池（对齐 generate_phyplan_api.py global_caps_A）
        self.global_caps_A: Dict[str, int] = {
            'Task_03_Micro_Affordance_Visual_Semantics': 2,
            'Task_05_State_Evolution_Description': 2,
            'Task_06_Holistic_Causal_Chain_Analysis': 2,
            'Task_02_Transient_Geometric_Verification': 2,
            'Task_11_Expected_Physical_Effects': 2,
            'Task_16_Physical_Feasibility_Verification': 2,
            'Task_17_Holistic_Step_Synthesis_Why_How': 2,
            'Task_04_Entity_Role_Identification': 2,
            'Task_08_Strategic_Rationale_Justification': 2,
            'Task_09_Precondition_Statement': 2,
            'Task_12_Inter_Step_Dependency_Analysis': 2,
            'Task_13_Next_Action_Prediction': 2,
            'Task_14_Counterfactual_Prediction': 2,
            'Task_15_Failure_Recovery_Protocol': 2,
            # 场景级（新 schema 需要）
            'Task_01_Macro_Anchor_Extraction': 1,
            'Task_07_Scene_Goal_Derivation': 1,
            'Task_10_Step_Execution_Statement': 2,
        }
        self.global_caps_B: Dict[str, int] = {}
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

    def _clean_str_list(self, xs: Any) -> List[str]:
        if not isinstance(xs, list):
            return []
        out: List[str] = []
        for x in xs:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    def _nonempty_str(self, s: Any) -> Optional[str]:
        if isinstance(s, str) and s.strip():
            return s.strip()
        return None

    # --- LLM 调用（对齐 generate_phyplan_api.py） ---
    def call_llm(self, prompt: str, images: Union[str, List[str]]):
        if self.client is None or self.api_config.api_key == 'EMPTY':
            logger.warning("API 客户端不可用或 API_KEY=EMPTY，将返回占位文本。")
            return "[API Disabled] " + (prompt[:256] if isinstance(prompt, str) else "")

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        # 保持与原脚本一致：当前不传图
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(getattr(self.api_config, 'max_retries', 3)))):
            try:
                logger.info(
                    f">>> [INFO] 调用模型: model={self.api_config.model_name}, fields_only=true, max_tokens={self.api_config.max_tokens}, attempt={attempt+1}"
                )
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.api_config.model_name,
                    messages=messages,
                    max_tokens=self.api_config.max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    presence_penalty=0,
                )
                dt = time.time() - t0
                if not (resp and getattr(resp, 'choices', None)):
                    logger.info(f"!!! [ERROR] API 响应为空或无 choices，用时 {dt:.2f}s")
                    raise RuntimeError("Empty response or missing choices")
                choice = resp.choices[0]
                if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                    logger.info(f"!!! [ERROR] API 响应缺少 message.content，用时 {dt:.2f}s")
                    raise RuntimeError("Missing message.content")
                out = choice.message.content or ""
                logger.info(f">>> [SUCCESS] API 调用完成，用时 {dt:.2f}s，输出长度={len(out)}")
                return out
            except Exception as e:
                last_err = e
                logger.error(f"模型 API 调用失败（attempt={attempt+1}）：{e}")
                if attempt + 1 < max(1, int(getattr(self.api_config, 'max_retries', 3))):
                    time.sleep(float(getattr(self.api_config, 'retry_backoff_sec', 1.5)) * (attempt + 1))
        if last_err is not None:
            logger.error(f"模型 API 调用最终失败：{last_err}")
        return ""

    def call_llm_custom(self, system_prompt: str, user_text: str, max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        if self.client is None or self.api_config.api_key == 'EMPTY':
            logger.warning("API 客户端不可用或 API_KEY=EMPTY，将返回占位文本。")
            return "[API Disabled] " + (user_text[:256] if isinstance(user_text, str) else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(getattr(self.api_config, 'max_retries', 3)))):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.api_config.model_name,
                    messages=messages,
                    max_tokens=max_tokens or min(1024, self.api_config.max_tokens),
                    temperature=temperature,
                    top_p=0.9,
                    presence_penalty=0,
                )
                dt = time.time() - t0
                if not (resp and getattr(resp, 'choices', None)):
                    logger.info(f"!!! [ERROR] 自定义 API 响应为空或无 choices，用时 {dt:.2f}s")
                    raise RuntimeError("Empty response or missing choices")
                choice = resp.choices[0]
                if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                    logger.info(f"!!! [ERROR] 自定义 API 响应缺少 message.content，用时 {dt:.2f}s")
                    raise RuntimeError("Missing message.content")
                out = choice.message.content or ""
                logger.info(f">>> [SUCCESS] 自定义 API 调用完成，用时 {dt:.2f}s，输出长度={len(out)}")
                return out
            except Exception as e:
                last_err = e
                logger.error(f"自定义模型 API 调用失败（attempt={attempt+1}）：{e}")
                if attempt + 1 < max(1, int(getattr(self.api_config, 'max_retries', 3))):
                    time.sleep(float(getattr(self.api_config, 'retry_backoff_sec', 1.5)) * (attempt + 1))
        if last_err is not None:
            logger.error(f"自定义模型 API 调用最终失败：{last_err}")
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
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"(?<=\w)\s*[-–]\s*(?=\w)", "-", s)
            s = re.sub(r"\s+([,;:\.!\?])", r"\1", s)
            s = re.sub(r"([,;:\.!\?])(\S)", r"\1 \2", s)
            s = re.sub(r"\s+", " ", s)
            return s.strip()
        except Exception:
            return text.strip() if isinstance(text, str) else ""

    def _sanitize_answer(self, task_name: str, text: str) -> str:
        """Sanitize formatting while preserving task-required paragraph structure.
        - Removes bullet markers/numbering only at start of lines.
        - Normalizes whitespace/newlines.
        - Preserves paragraph breaks for 2-paragraph tasks.
        """
        if not text:
            return ""

        try:
            import re

            cleaned_text = re.sub(r'(?m)^\s*([\-\*•\>]+|\d+\.)\s+', '', text)
            cleaned_text = cleaned_text.strip().replace("\r\n", "\n").replace("\r", "\n")
            cleaned_text = re.sub(r"```[a-zA-Z]*\s*", "", cleaned_text)
            cleaned_text = cleaned_text.replace("```", "")

            if task_name in {'Task_06_Holistic_Causal_Chain_Analysis', 'Task_16_Physical_Feasibility_Verification', 'Task_17_Holistic_Step_Synthesis_Why_How'}:
                paragraphs = re.split(r'\n\s*\n', cleaned_text)
                paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in paragraphs if p.strip()]
                return "\n\n".join(paragraphs)
            return re.sub(r'\s+', ' ', cleaned_text).strip()
        except Exception:
            return text.strip()

    def polish_answer(self, task_name: str, raw_answer: str, fields: Dict[str, Any]) -> str:
        if task_name in {'Task_07_Scene_Goal_Derivation', 'Task_14_Counterfactual_Prediction'}:
            return self._sanitize_answer(task_name, raw_answer)

        ctx = json.dumps(fields, ensure_ascii=False)
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

        instr = (
            "You are an expert Embodied AI Analyst. Your task is to synthesize structured data into a natural, rigorous, and high-quality English response.\n\n"
            f"### INPUT DATA (Ground Truth):\n{ctx}\n\n"
            f"### DRAFT ANSWER (For Reference):\n{raw_answer}\n\n"
            "### INSTRUCTIONS:\n"
            "Rewrite the draft based strictly on the Input Data to improve fluency and professional tone.\n"
            "1. **Strict Fidelity**: Use ONLY the entities, actions, and relationships present in the 'Input Data'. Do NOT hallucinate external details.\n"
            "2. **Natural Flow**: Avoid robotic listing; integrate the facts into a coherent narrative.\n"
            "3. **No Formatting**: Do NOT use bullet points, lists, bold text, or labels.\n"
            f"{structure_instr}\n\n"
            "### POLISHED OUTPUT:"
        )
        out = self.call_llm(instr, images=[])
        return self._sanitize_answer(task_name, out if out and out.strip() else raw_answer)

    def _defluff_text(self, text: str) -> str:
        try:
            import re

            s = text or ""
            patterns = [
                r"^\s*(In summary|In conclusion|To summarize|Overall|In general|Generally),\s*",
                r"^\s*(In this (scene|image|frame|step)),\s*",
                r"^\s*(It should be noted that|Note that)\s*",
            ]
            for pat in patterns:
                s = re.sub(pat, "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+(Overall|In summary|In conclusion),\s*", ", ", s, flags=re.IGNORECASE)
            return self._sanitize_text(s)
        except Exception:
            return text or ""

    def create_sharegpt_entry(self, images: Union[str, List[str]], user_q: str, assistant_a: str, meta: Optional[Dict[str, Any]] = None) -> Dict:
        return {
            "id": str(uuid.uuid4()),
            "image": images,
            "conversations": [
                {"from": "human", "value": user_q},
                {"from": "gpt", "value": assistant_a},
            ],
            "meta": meta or {},
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

            if getattr(self, 'stream_save', True):
                try:
                    file_path = os.path.join(self.output_dir, task_name, "data.jsonl")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    meta_str = json.dumps(meta, ensure_ascii=False)
                    logger.info(f"[Saved ] task={task_name} src={src_file} -> {file_path}\n[Meta ] {meta_str}")
                except Exception as e:
                    logger.warning(f"[SaveWarn] {task_name} 追加写入失败：{e}")

            item_type = (meta.get('item_type') or '').upper()
            if item_type == 'TYPEA':
                self.data_buffer_A[task_name].append(entry)
            elif item_type == 'TYPEB':
                self.data_buffer_B[task_name].append(entry)

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
        for task_name in TASK_PROMPTS.keys():
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
                                    self.data_buffer[task_name].append(entry)
                            except Exception:
                                continue
                if a_count or b_count:
                    logger.info(f"[Hydrate] task={task_name} preload A={a_count} B={b_count}")
            except Exception as e:
                logger.warning(f"[HydrateWarn] 预加载 {task_name} 失败: {e}")

    def _key_from_meta(self, meta: Dict[str, Any]) -> str:
        return (
            f"{meta.get('item_type','')}|{meta.get('source_path','')}|{meta.get('task_name','')}|"
            f"{int(meta.get('step_index',-1))}|{int(meta.get('frame_index',-1))}|{int(meta.get('sub_index',0))}"
        )

    def _meta(
        self,
        item_type: str,
        source_path: str,
        task_name: str,
        step_index: int,
        frame_index: int,
        sub_index: int,
        step_goal: Optional[str],
        image_path: Optional[str],
    ) -> Dict[str, Any]:
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

    def emit_sample(
        self,
        task_name: str,
        images: Union[str, List[str]],
        prompt: str,
        user_q: str,
        meta: Dict[str, Any],
        resume: bool = True,
        force: bool = False,
        fields: Optional[Dict[str, Any]] = None,
    ):
        key = self._key_from_meta(meta)
        if resume and not force and hasattr(self, 'processed_keys') and key in getattr(self, 'processed_keys', set()):
            logger.info(f"[SkipSample] {task_name} key={key}")
            return
        logger.info(
            f">>> [Gen ] task={task_name} src={meta.get('source_path','')} step={meta.get('step_index','')} frame={meta.get('frame_index','')} sub={meta.get('sub_index','')}"
        )
        polished_q = self._sanitize_text(user_q)
        full_prompt = f"User Question:\n{polished_q}\n\n" + (prompt or "")
        ans = self.call_llm(full_prompt, images)
        if not isinstance(ans, str) or not ans.strip():
            logger.warning(f"[DropEmpty] task={task_name} empty raw answer")
            return
        polished_ans = self.polish_answer(task_name, ans, fields or {}) if fields is not None else self._sanitize_text(ans)
        polished_ans = self._sanitize_answer(task_name, polished_ans)
        polished_ans = self._defluff_text(polished_ans)
        polished_q = self._sanitize_text(polished_q)

        if not polished_ans or len(polished_ans.strip()) < int(getattr(self, 'min_output_chars', 24)):
            logger.warning(
                f"[DropShort] task={task_name} output too short len={len(polished_ans.strip()) if polished_ans else 0}"
            )
            return

        # Avoid training on the placeholder text when API is disabled.
        if polished_ans.strip().startswith('[API Disabled]'):
            logger.warning(f"[DropPlaceholder] task={task_name} API disabled placeholder")
            return
        entry = self.create_sharegpt_entry(images, polished_q, polished_ans, meta=meta)
        self.save_to_buffer(task_name, entry)
        if hasattr(self, 'processed_keys'):
            self.processed_keys.add(key)

    # ---------------------------------------------------------
    # 依赖关系判定（对齐 generate_phyplan_api.py）
    # ---------------------------------------------------------
    def _normalize_terms(self, text: str) -> set:
        if not isinstance(text, str) or not text.strip():
            return set()
        import re

        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "is", "are", "be", "by", "for", "with", "then", "into", "from", "that", "this"}
        return {t for t in tokens if t not in stop and len(t) >= 3}

    def _extract_effect_terms(self, step: Dict) -> set:
        terms = set()
        for e in self._clean_str_list(step.get('expected_effects')):
            terms |= self._normalize_terms(e)
        return terms

    def _extract_precondition_terms(self, step: Dict) -> set:
        terms = set()
        for p in self._clean_str_list(step.get('preconditions')):
            terms |= self._normalize_terms(p)
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
    # Type A 处理逻辑（新 mani-longvideo schema）
    # =========================================================================
    def _process_type_a(self, data: Dict, source_path: Optional[str] = None):
        global_goal = data.get('high_level_goal', 'N/A')
        steps = data.get('steps', [])
        steps_count = len(steps) if isinstance(steps, list) else 0
        if not isinstance(steps, list):
            return

        # 对齐 mani_longvideo_tasks_plan.md：steps 按 step_id 升序
        def _to_int(v):
            try:
                return int(v)
            except Exception:
                return 0

        steps = sorted(steps, key=lambda s: _to_int((s or {}).get('step_id', 0)) if isinstance(s, dict) else 0)

        balance_counter: Dict[str, int] = {}

        # Scene-level: Task_01 / Task_07（新 schema）
        try:
            anchors = set()
            affordance_tags = set()
            for step in steps:
                if not isinstance(step, dict):
                    continue
                tm = step.get('tool_and_material_usage', {})
                for t in tm.get('tools', []) or []:
                    if isinstance(t, str) and t.strip():
                        anchors.add(t.strip())
                for m in tm.get('materials', []) or []:
                    if isinstance(m, str) and m.strip():
                        anchors.add(m.strip())
                for frame in step.get('critical_frames', []) or []:
                    if not isinstance(frame, dict):
                        continue
                    for sp in frame.get('spatial_preconditions', []) or []:
                        if isinstance(sp, dict):
                            for o in sp.get('objects', []) or []:
                                if isinstance(o, str) and o.strip():
                                    anchors.add(o.strip())
                    for ap in frame.get('affordance_preconditions', []) or []:
                        if isinstance(ap, dict):
                            on = ap.get('object_name')
                            if isinstance(on, str) and on.strip():
                                anchors.add(on.strip())
                            for at in ap.get('affordance_types', []) or []:
                                if isinstance(at, str) and at.strip():
                                    affordance_tags.add(at.strip())

                    cc = frame.get('causal_chain', {})
                    if isinstance(cc, dict):
                        ag = cc.get('agent')
                        pt = cc.get('patient')
                        if isinstance(ag, str) and ag.strip():
                            anchors.add(ag.strip())
                        if isinstance(pt, str) and pt.strip():
                            anchors.add(pt.strip())

                    hs = frame.get('affordance_hotspot', {})
                    if isinstance(hs, dict):
                        aft = hs.get('affordance_type')
                        if isinstance(aft, str) and aft.strip():
                            affordance_tags.add(aft.strip())

            # 用首步首帧作为场景代表图
            base_img_scene = 'missing.jpg'
            try:
                if steps and isinstance(steps[0], dict):
                    cfs0 = steps[0].get('critical_frames', []) or []
                    if cfs0 and isinstance(cfs0[0], dict):
                        base_img_scene = cfs0[0].get('keyframe_image_path', 'missing.jpg')
            except Exception:
                pass

            fields1 = {
                "high_level_goal": global_goal,
                "key_objects": sorted(list(anchors)),
                "affordance_tags": sorted(list(affordance_tags)),
            }
            if (not getattr(self, 'strict', True)) or fields1.get('key_objects'):
                p1 = build_answer_prompt('Task_01_Macro_Anchor_Extraction', fields1)
                user_q1 = build_user_question('Task_01_Macro_Anchor_Extraction', fields1)
                meta1 = self._meta('TypeA', source_path or '', 'Task_01_Macro_Anchor_Extraction', 0, -1, 0, None, base_img_scene)
                self.global_candidates_A['Task_01_Macro_Anchor_Extraction'].append({"images": base_img_scene, "prompt": p1, "user_q": user_q1, "meta": meta1, "fields": fields1})

            fields7 = {"high_level_goal": global_goal}
            if (not getattr(self, 'strict', True)) or self._nonempty_str(global_goal):
                p7 = build_answer_prompt('Task_07_Scene_Goal_Derivation', fields7)
                user_q7 = build_user_question('Task_07_Scene_Goal_Derivation', fields7)
                meta7 = self._meta('TypeA', source_path or '', 'Task_07_Scene_Goal_Derivation', 0, -1, 0, None, base_img_scene)
                self.global_candidates_A['Task_07_Scene_Goal_Derivation'].append({"images": base_img_scene, "prompt": p7, "user_q": user_q7, "meta": meta7, "fields": fields7})
        except Exception as e:
            logger.warning(f"[TypeA] Scene-level candidates failed: {e}")

        for i, step in enumerate(steps):
            try:
                if not isinstance(step, dict):
                    continue
                step_goal = step.get('step_goal', 'Unknown Step')
                if getattr(self, 'strict', True) and not self._nonempty_str(step_goal):
                    continue
                critical_frames = step.get('critical_frames', [])
                if not critical_frames:
                    continue

                def _to_int(v):
                    try:
                        return int(v)
                    except Exception:
                        return 0

                critical_frames = sorted(critical_frames, key=lambda fr: _to_int((fr or {}).get('frame_index', 0)))
                base_img = (critical_frames[0] or {}).get('keyframe_image_path', 'missing.jpg')
                logger.info(f">>> [INFO] TypeA Step {i+1}/{len(steps)}: {step_goal}")

                # Task 4: Tools/Materials
                tm = step.get('tool_and_material_usage', {})
                fields4 = {"tools": tm.get('tools'), "materials": tm.get('materials'), "step_goal": step_goal, "high_level_goal": global_goal}
                p4 = build_answer_prompt('Task_04_Entity_Role_Identification', fields4)
                user_q4 = build_user_question('Task_04_Entity_Role_Identification', fields4)
                meta4 = self._meta('TypeA', source_path or '', 'Task_04_Entity_Role_Identification', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_04_Entity_Role_Identification'].append({"images": base_img, "prompt": p4, "user_q": user_q4, "meta": meta4, "fields": fields4})

                # Task 8: Rationale
                rationale = step.get('rationale')
                if getattr(self, 'strict', True) and not self._nonempty_str(rationale):
                    rationale = None
                fields8 = {"global_goal": global_goal, "step_goal": step_goal, "rationale": rationale}
                p8 = build_answer_prompt('Task_08_Strategic_Rationale_Justification', fields8)
                user_q8 = build_user_question('Task_08_Strategic_Rationale_Justification', fields8)
                meta8 = self._meta('TypeA', source_path or '', 'Task_08_Strategic_Rationale_Justification', i+1, -1, 0, step_goal, base_img)
                if not getattr(self, 'strict', True) or rationale:
                    self.global_candidates_A['Task_08_Strategic_Rationale_Justification'].append({"images": base_img, "prompt": p8, "user_q": user_q8, "meta": meta8, "fields": fields8})

                # Task 9: Preconditions
                preconditions = self._clean_str_list(step.get('preconditions'))
                fields9 = {"step_goal": step_goal, "preconditions": preconditions, "high_level_goal": global_goal}
                p9 = build_answer_prompt('Task_09_Precondition_Statement', fields9)
                user_q9 = build_user_question('Task_09_Precondition_Statement', fields9)
                meta9 = self._meta('TypeA', source_path or '', 'Task_09_Precondition_Statement', i+1, -1, 0, step_goal, base_img)
                if not getattr(self, 'strict', True) or preconditions:
                    self.global_candidates_A['Task_09_Precondition_Statement'].append({"images": base_img, "prompt": p9, "user_q": user_q9, "meta": meta9, "fields": fields9})

                # Task 10: Step execution (from critical frame action descriptions)
                actions = [
                    (fr or {}).get('action_description')
                    for fr in critical_frames
                    if isinstance(fr, dict) and isinstance((fr or {}).get('action_description'), str)
                ]
                fields10 = {"step_goal": step_goal, "actions": actions, "high_level_goal": global_goal}
                p10 = build_answer_prompt('Task_10_Step_Execution_Statement', fields10)
                user_q10 = build_user_question('Task_10_Step_Execution_Statement', fields10)
                meta10 = self._meta('TypeA', source_path or '', 'Task_10_Step_Execution_Statement', i+1, -1, 0, step_goal, base_img)
                self.global_candidates_A['Task_10_Step_Execution_Statement'].append({"images": base_img, "prompt": p10, "user_q": user_q10, "meta": meta10, "fields": fields10})

                # Task 11: Expected effects
                expected_effects = self._clean_str_list(step.get('expected_effects'))
                fields11_step = {"macro_eff": expected_effects, "step_goal": step_goal, "high_level_goal": global_goal}
                p11 = build_answer_prompt('Task_11_Expected_Physical_Effects', fields11_step)
                user_q11 = build_user_question('Task_11_Expected_Physical_Effects', fields11_step)
                meta11 = self._meta('TypeA', source_path or '', 'Task_11_Expected_Physical_Effects', i+1, -1, 0, step_goal, base_img)
                if not getattr(self, 'strict', True) or expected_effects:
                    self.global_candidates_A['Task_11_Expected_Physical_Effects'].append({"images": base_img, "prompt": p11, "user_q": user_q11, "meta": meta11, "fields": fields11_step})

                # Task 12: Dependency (if exists)
                if steps_count > 1 and i < len(steps) - 1:
                    next_step = steps[i + 1]
                    if isinstance(next_step, dict) and self._has_dependency(step, next_step):
                        fields12 = {
                            "step_n_goal": step_goal,
                            "step_n_effect": step.get('expected_effects'),
                            "step_next_goal": next_step.get('step_goal'),
                            "step_next_precondition": next_step.get('preconditions'),
                            "high_level_goal": global_goal,
                        }
                        p12 = build_answer_prompt('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        user_q12 = build_user_question('Task_12_Inter_Step_Dependency_Analysis', fields12)
                        meta12 = self._meta('TypeA', source_path or '', 'Task_12_Inter_Step_Dependency_Analysis', i+1, -1, 0, step_goal, base_img)
                        self.global_candidates_A['Task_12_Inter_Step_Dependency_Analysis'].append({"images": base_img, "prompt": p12, "user_q": user_q12, "meta": meta12, "fields": fields12})
                elif steps_count <= 1:
                    logger.info("[TypeA] 单步骤短程任务：跳过跨步依赖（Task 12）生成")

                # Task 14: Counterfactual
                fields14 = {
                    "question": step.get('causal_challenge_question'),
                    "outcome": step.get('expected_challenge_outcome'),
                    "high_level_goal": global_goal,
                    "step_goal": step_goal,
                }
                p14 = build_answer_prompt('Task_14_Counterfactual_Prediction', fields14)
                user_q14 = build_user_question('Task_14_Counterfactual_Prediction', fields14)
                meta14 = self._meta('TypeA', source_path or '', 'Task_14_Counterfactual_Prediction', i+1, -1, 0, step_goal, base_img)
                if not getattr(self, 'strict', True) or (self._nonempty_str(fields14.get('question')) and self._nonempty_str(fields14.get('outcome'))):
                    self.global_candidates_A['Task_14_Counterfactual_Prediction'].append({"images": base_img, "prompt": p14, "user_q": user_q14, "meta": meta14, "fields": fields14})

                # Task 13: Next step goal (label)
                if i < len(steps) - 1 and isinstance(steps[i + 1], dict):
                    fields13 = {"next_step_goal": (steps[i + 1] or {}).get('step_goal'), "step_goal": step_goal, "high_level_goal": global_goal}
                    p13 = build_answer_prompt('Task_13_Next_Action_Prediction', fields13)
                    user_q13 = build_user_question('Task_13_Next_Action_Prediction', fields13)
                    meta13 = self._meta('TypeA', source_path or '', 'Task_13_Next_Action_Prediction', i+1, -1, 0, step_goal, base_img)
                    self.global_candidates_A['Task_13_Next_Action_Prediction'].append({"images": base_img, "prompt": p13, "user_q": user_q13, "meta": meta13, "fields": fields13})

                # Task 15: Failure recovery
                fh = step.get('failure_handling', {})
                if isinstance(fh, dict):
                    fields15 = {"reason": fh.get('reason'), "strategy": fh.get('recovery_strategy'), "step_goal": step_goal, "high_level_goal": global_goal}
                    p15 = build_answer_prompt('Task_15_Failure_Recovery_Protocol', fields15)
                    user_q15 = build_user_question('Task_15_Failure_Recovery_Protocol', fields15)
                    meta15 = self._meta('TypeA', source_path or '', 'Task_15_Failure_Recovery_Protocol', i+1, -1, 0, step_goal, base_img)
                    if not getattr(self, 'strict', True) or (self._nonempty_str(fields15.get('reason')) and self._nonempty_str(fields15.get('strategy'))):
                        self.global_candidates_A['Task_15_Failure_Recovery_Protocol'].append({"images": base_img, "prompt": p15, "user_q": user_q15, "meta": meta15, "fields": fields15})

                # --- Frame Level candidates ---
                for frame in critical_frames:
                    if not isinstance(frame, dict):
                        continue
                    img = frame.get('keyframe_image_path', 'missing.jpg')
                    logger.info(f"  -> 关键帧: {img}")

                    # Task 3: Micro Hotspot
                    ah = frame.get('affordance_hotspot', {})
                    fields3 = {
                        "aff_type": (ah or {}).get('affordance_type'),
                        "desc": (ah or {}).get('description'),
                        "mechanism": (ah or {}).get('mechanism'),
                        "step_goal": step_goal,
                        "high_level_goal": global_goal,
                    }
                    p3 = build_answer_prompt('Task_03_Micro_Affordance_Visual_Semantics', fields3)
                    user_q3 = build_user_question('Task_03_Micro_Affordance_Visual_Semantics', fields3)
                    meta3 = self._meta('TypeA', source_path or '', 'Task_03_Micro_Affordance_Visual_Semantics', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if (not getattr(self, 'strict', True) or self._nonempty_str(fields3.get('aff_type')) or self._nonempty_str(fields3.get('desc'))):
                        if self._accept_balance('Task_03_Micro_Affordance_Visual_Semantics', balance_counter):
                            self.global_candidates_A['Task_03_Micro_Affordance_Visual_Semantics'].append({"images": img, "prompt": p3, "user_q": user_q3, "meta": meta3, "fields": fields3})

                    # Task 6: Causal chain
                    cc = frame.get('causal_chain', {})
                    fields6 = {
                        "agent": (cc or {}).get('agent'),
                        "action": (cc or {}).get('action'),
                        "patient": (cc or {}).get('patient'),
                        "aff_pre": frame.get('affordance_preconditions', []),
                        "mechanism": (ah or {}).get('mechanism'),
                        "spatial": frame.get('spatial_preconditions', []),
                        "eff_pat": (cc or {}).get('causal_effect_on_patient'),
                        "eff_env": (cc or {}).get('causal_effect_on_environment'),
                        "high_level_goal": global_goal,
                        "step_goal": step_goal,
                    }
                    p6 = build_answer_prompt('Task_06_Holistic_Causal_Chain_Analysis', fields6)
                    user_q6 = build_user_question('Task_06_Holistic_Causal_Chain_Analysis', fields6)
                    meta6 = self._meta('TypeA', source_path or '', 'Task_06_Holistic_Causal_Chain_Analysis', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if not getattr(self, 'strict', True) or (self._nonempty_str(fields6.get('agent')) and self._nonempty_str(fields6.get('action')) and self._nonempty_str(fields6.get('patient'))):
                        if self._accept_balance('Task_06_Holistic_Causal_Chain_Analysis', balance_counter):
                            self.global_candidates_A['Task_06_Holistic_Causal_Chain_Analysis'].append({"images": img, "prompt": p6, "user_q": user_q6, "meta": meta6, "fields": fields6})

                    # Task 5: state evolution
                    fields5 = {"action_desc": frame.get('action_description'), "state_change": frame.get('state_change_description'), "high_level_goal": global_goal, "step_goal": step_goal}
                    p5 = build_answer_prompt('Task_05_State_Evolution_Description', fields5)
                    user_q5 = build_user_question('Task_05_State_Evolution_Description', fields5)
                    meta5 = self._meta('TypeA', source_path or '', 'Task_05_State_Evolution_Description', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                    if not getattr(self, 'strict', True) or (self._nonempty_str(fields5.get('action_desc')) and self._nonempty_str(fields5.get('state_change'))):
                        if self._accept_balance('Task_05_State_Evolution_Description', balance_counter):
                            self.global_candidates_A['Task_05_State_Evolution_Description'].append({"images": img, "prompt": p5, "user_q": user_q5, "meta": meta5, "fields": fields5})

                    # Task 2: transient geometry (pick one relation)
                    sp_list = frame.get('spatial_preconditions', []) or []
                    if isinstance(sp_list, list) and sp_list:
                        # choose one relation by default; optionally emit all relations
                        for j, sp in enumerate(sp_list):
                            if not isinstance(sp, dict):
                                continue
                            objs = sp.get('objects', []) or []
                            if isinstance(objs, list) and len(objs) >= 2:
                                fields2 = {
                                    "relation": sp.get('relation'),
                                    "objects": objs,
                                    "truth": sp.get('truth'),
                                    "high_level_goal": global_goal,
                                    "step_goal": step_goal,
                                }
                                p2 = build_answer_prompt('Task_02_Transient_Geometric_Verification', fields2)
                                user_q2 = build_user_question('Task_02_Transient_Geometric_Verification', fields2)
                                meta2 = self._meta('TypeA', source_path or '', 'Task_02_Transient_Geometric_Verification', i+1, frame.get('frame_index') or 0, j, step_goal, img)
                                if self._accept_balance('Task_02_Transient_Geometric_Verification', balance_counter):
                                    self.global_candidates_A['Task_02_Transient_Geometric_Verification'].append({"images": img, "prompt": p2, "user_q": user_q2, "meta": meta2, "fields": fields2})
                                if not getattr(self, 'full_relations', False):
                                    break

                    # Task 16: feasibility
                    sp_pre = frame.get('spatial_preconditions', [])
                    af_pre = frame.get('affordance_preconditions', [])
                    if (sp_pre and len(sp_pre) > 0) or (af_pre and len(af_pre) > 0):
                        fields16 = {"step_goal": step_goal, "spatial_preconditions": sp_pre, "affordance_preconditions": af_pre, "high_level_goal": global_goal}
                        p16 = build_answer_prompt('Task_16_Physical_Feasibility_Verification', fields16)
                        user_q16 = build_user_question('Task_16_Physical_Feasibility_Verification', fields16)
                        meta16 = self._meta('TypeA', source_path or '', 'Task_16_Physical_Feasibility_Verification', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                        if self._accept_balance('Task_16_Physical_Feasibility_Verification', balance_counter):
                            self.global_candidates_A['Task_16_Physical_Feasibility_Verification'].append({"images": img, "prompt": p16, "user_q": user_q16, "meta": meta16, "fields": fields16})

                    # Task 17: holistic synthesis (why/how)
                    mech = (ah or {}).get('mechanism')
                    if isinstance(mech, str) and mech.strip():
                        fields17 = {
                            "step_goal": step_goal,
                            "rationale": step.get('rationale', ''),
                            "mechanism": mech,
                            "causal_chain": cc if isinstance(cc, dict) else {},
                            "spatial": frame.get('spatial_preconditions', []),
                            "high_level_goal": global_goal,
                        }
                        p17 = build_answer_prompt('Task_17_Holistic_Step_Synthesis_Why_How', fields17)
                        user_q17 = build_user_question('Task_17_Holistic_Step_Synthesis_Why_How', fields17)
                        meta17 = self._meta('TypeA', source_path or '', 'Task_17_Holistic_Step_Synthesis_Why_How', i+1, frame.get('frame_index') or 0, 0, step_goal, img)
                        if self._accept_balance('Task_17_Holistic_Step_Synthesis_Why_How', balance_counter):
                            self.global_candidates_A['Task_17_Holistic_Step_Synthesis_Why_How'].append({"images": img, "prompt": p17, "user_q": user_q17, "meta": meta17, "fields": fields17})

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
                    self.emit_sample(
                        tname,
                        cand["images"],
                        cand["prompt"],
                        cand["user_q"],
                        cand["meta"],
                        resume=self.resume,
                        force=self.force,
                        fields=cand.get("fields"),
                    )
            self.global_candidates_A = {k: [] for k in self.global_caps_A.keys()}
        except Exception as e:
            logger.warning(f"[TypeA] 全局候选采样失败: {e}")

    def process_entry(self, raw_data: Dict, source_path: Optional[str] = None):
        self._process_type_a(raw_data, source_path)

    def flush_to_disk(self):
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
# 5) 批处理入口（仅 TypeA）
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


def load_existing_sample_keys(output_dir: str) -> set:
    keys = set()

    def key_from_meta(meta: Dict[str, Any]) -> str:
        return f"{meta.get('item_type','')}|{meta.get('source_path','')}|{meta.get('task_name','')}|{int(meta.get('step_index',-1))}|{int(meta.get('frame_index',-1))}|{int(meta.get('sub_index',0))}"

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Mani-LongVideo 17 tasks via direct model API (TypeA only).')
    parser.add_argument('--typeA-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long', help='Root folder for TypeA data')
    parser.add_argument('--output-dir', default='/e2e-data/evad-tech-vla/luzheng/ICML/mani_longvideo_qa_output_api', help='Output directory')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--stream-save', action='store_true', default=True)
    parser.add_argument('--live-combined', action='store_true', default=True)
    parser.add_argument('--strict', action='store_true', default=True, help='Drop samples with missing required fields')
    parser.add_argument('--no-strict', action='store_true', default=False, help='Allow samples with missing fields')
    parser.add_argument('--full-relations', action='store_true', default=False, help='Emit all spatial relations for Task_02 instead of one per frame')
    parser.add_argument('--min-output-chars', type=int, default=int(os.environ.get('MIN_OUTPUT_CHARS', '24')))
    parser.add_argument('--api-key', default=os.environ.get('API_KEY', 'EMPTY'))
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

    existing_keys = set()
    if args.resume and not args.restart:
        existing_keys = load_existing_sample_keys(args.output_dir)
        logging.info(f"[Resume] 已加载样本级进度键：{len(existing_keys)} 条")

    generator = ManiLongVideoAPIGenerator(
        output_dir=args.output_dir,
        api_config=api_cfg,
        stream_save=args.stream_save,
        resume=args.resume,
        force=args.force,
        processed_keys=existing_keys,
        live_combined=args.live_combined,
        strict=(False if args.no_strict else args.strict),
        full_relations=args.full_relations,
        min_output_chars=args.min_output_chars,
    )

    try:
        generator.hydrate_existing_buffers()
    except Exception as e:
        logging.warning(f"[HydrateWarn] preload failed: {e}")

    a_items = scan_type_a_root(args.typeA_root)
    if args.limit and args.limit > 0:
        a_items = a_items[: args.limit]
    logging.info(f"Discovered TypeA items: {len(a_items)} from {args.typeA_root}")

    a_items = [os.path.abspath(p) for p in a_items]

    prog = load_progress(args.output_dir)
    if args.restart:
        clear_progress(args.output_dir)
        prog = {"typeA": set(), "typeB": set()}

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

    generator.flush_to_disk()
