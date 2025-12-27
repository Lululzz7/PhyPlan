# -*- coding: utf-8 -*-
"""
PhyPlan 任务生成（单阶段·高质量 API 版） - 完整版

主要特性：
- Single-Stage Generation: 1个样本 = 1次 API 调用。通过精细的 Prompt Engineering 一次性生成高质量答案。
- High Fidelity: 强制模型使用学术语调、自然段落，严禁 Bullet Points 和模板套话。
- Full Coverage: 包含 Type A (帧级) 和 Type B (场景级) 所有 17 类物理推理任务。
- Robustness: 包含断点续传、实时落盘、自动平衡采样逻辑。
"""

import os
import json
import uuid
import logging
import time
import random
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

# ==============================================================================
# 0) 日志与基础配置
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 屏蔽 HTTP 库的冗余日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 所有任务名称定义（用于目录创建与路由）
ALL_TASK_KEYS = [
    "Task_01_Macro_Anchor_Extraction",
    "Task_02_Transient_Geometric_Verification",
    "Task_03_Micro_Affordance_Visual_Semantics",
    "Task_04_Entity_Role_Identification",
    "Task_05_State_Evolution_Description",
    "Task_06_Holistic_Causal_Chain_Analysis",
    "Task_07_Scene_Goal_Derivation",
    "Task_08_Strategic_Rationale_Justification",
    "Task_09_Precondition_Statement",
    "Task_10_Step_Execution_Statement",
    "Task_11_Expected_Physical_Effects",
    "Task_12_Inter_Step_Dependency_Analysis",
    "Task_13_Next_Action_Prediction",
    "Task_14_Counterfactual_Prediction",
    "Task_15_Failure_Recovery_Protocol",
    "Task_16_Physical_Feasibility_Verification",
    "Task_17_Holistic_Step_Synthesis_Why_How"
]

# ==============================================================================
# 1) 模型 API 配置
# ==============================================================================

@dataclass
class ApiConfig:
    api_key: str = os.environ.get('API_KEY', 'EMPTY')
    api_base_url: str = os.environ.get('API_BASE_URL', 'http://model.mify.ai.srv/v1')
    model_provider_id: str = os.environ.get('MODEL_PROVIDER_ID', 'vertex_ai')
    model_name: str = os.environ.get('MODEL_NAME', 'gemini-3-pro-preview')
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
        logger.info(f">>> [INFO] API Client Initialized: {cfg.model_name}")
        return client
    except Exception as e:
        logger.warning(f"初始化 OpenAI 客户端失败：{e}. 将无法进行生成。")
        return None

# ==============================================================================
# 2) 核心 Prompt (System Prompt + Task Builders)
# ==============================================================================

# 升级版 System Prompt：强制单阶段高质量输出
SYSTEM_PROMPT = """You are an expert Embodied AI Analyst and Physics Consultant.
Your task is to synthesize structured input data into a single, high-quality, natural English response for a QA dataset.

### CRITICAL STYLE GUIDELINES (STRICT ADHERENCE REQUIRED):
1.  **Natural Synthesis**: Do NOT list fields explicitly. Do NOT use bullet points, numbered lists, or labels (like "Step Goal: ..."). Wield the data into fluent, professional English sentences.
2.  **No Filler**: Avoid conversational fillers (e.g., "Here is the answer", "In this scene", "It is important to note"). Start directly with the substance.
3.  **Strict Grounding**: Use ONLY the provided Input Data. Do not hallucinate external details or adjectives not present in the source.
4.  **Academic Tone**: Maintain an objective, clinical, and precise tone.
5.  **Structure**: Unless specified otherwise in the instruction, write in a continuous paragraph. Do NOT use line breaks within a logical block.

Your output must be the **final, polished answer** ready for training, with no need for further editing."""

def _strip_quotes_punct(s: Optional[str]) -> str:
    try:
        t = (s or "").strip()
        t = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", t)
        t = re.sub(r"[\.;:!?]+$", "", t)
        t = re.sub(r"\s+", " ", t)
        return t
    except Exception:
        return s or ""

def _finalize_question(q: str) -> str:
    try:
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
    """根据任务类型构建自然语言问句"""
    sg_raw = fields.get('step_goal')
    hl_raw = fields.get('high_level_goal') or fields.get('global_goal')
    sg = _strip_quotes_punct(sg_raw)
    hl = _strip_quotes_punct(hl_raw)
    
    q = ""
    if task_name == 'Task_04_Entity_Role_Identification':
        q = f"In the step {sg or 'this step'}, which items function as tools, and which are the materials being acted upon"
    elif task_name == 'Task_08_Strategic_Rationale_Justification':
        q = f"Given the overall goal {hl or 'the mission'}, why is the step {sg or 'this step'} strategically necessary"
    elif task_name == 'Task_09_Precondition_Statement':
        q = f"Given the overall goal {hl or 'the mission'}, before starting {sg or 'this step'}, what objective preconditions must be satisfied"
    elif task_name == 'Task_12_Inter_Step_Dependency_Analysis':
        s1 = _strip_quotes_punct(fields.get('step_n_goal') or 'the previous step')
        s2 = _strip_quotes_punct(fields.get('step_next_goal') or 'the next step')
        q = f"Given the overall goal {hl or 'the mission'}, how does the outcome of {s1} satisfy the preconditions for {s2}"
    elif task_name == 'Task_14_Counterfactual_Prediction':
        q0 = _strip_quotes_punct(fields.get('question') or 'What if the condition were different')
        q = f"Given the overall goal {hl or 'the mission'} and the step goal {sg or 'this step'}, {q0}"
    elif task_name == 'Task_13_Next_Action_Prediction':
        q = f"Given the overall goal {hl or 'the mission'}, what are the most logical next micro-actions for the step {sg or 'this step'}"
    elif task_name == 'Task_15_Failure_Recovery_Protocol':
        reason = _strip_quotes_punct(fields.get('reason') or 'a potential failure')
        q = f"Given the overall goal {hl or 'the mission'}, in the step {sg or 'this step'}, why might it fail due to {reason}, and what recovery strategy should be applied"
    elif task_name == 'Task_03_Micro_Affordance_Visual_Semantics':
        aff = _strip_quotes_punct(fields.get('aff_type') or 'this affordance')
        q = f"Which specific region affords {aff}, and how does it visually appear and physically function"
    elif task_name == 'Task_06_Holistic_Causal_Chain_Analysis':
        agent = _strip_quotes_punct(fields.get('agent') or 'the agent')
        action = _strip_quotes_punct(fields.get('action') or 'acting on')
        patient = _strip_quotes_punct(fields.get('patient') or 'the object')
        q = f"Could you explain how {agent} is {action} {patient} in this keyframe, focusing on the spatial setup, the affordance-level mechanism, and the immediate effects"
    elif task_name == 'Task_05_State_Evolution_Description':
        q = f"Given the overall goal {hl or 'the mission'}, what ongoing action is occurring, and what immediate state change does it cause"
    elif task_name == 'Task_02_Transient_Geometric_Verification':
        a = _strip_quotes_punct(fields.get('obj_a') or 'the first object')
        b = _strip_quotes_punct(fields.get('obj_b') or 'the second object')
        q = f"What is the precise spatial relationship between {a} and {b} in this frame"
    elif task_name == 'Task_11_Expected_Physical_Effects':
        q = f"Given the overall goal {hl or 'the mission'}, upon completion of {sg or 'this step'}, what physical effects should be expected for the environment and objects"
    elif task_name == 'Task_16_Physical_Feasibility_Verification':
        q = f"Given the overall goal {hl or 'the mission'}, is the step {sg or 'this step'} physically feasible now, based on spatial and affordance evidence"
    elif task_name == 'Task_17_Holistic_Step_Synthesis_Why_How':
        q = f"Given the overall goal {hl or 'the mission'}, why is the step {sg or 'this step'} necessary, and how is it physically achieved"
    elif task_name == 'Task_01_Macro_Anchor_Extraction':
        q = "Which stable objects are the task-relevant anchors for planning in this scene"
    elif task_name == 'Task_07_Scene_Goal_Derivation':
        q = "Given the current scene, what is the appropriate high-level goal"
    elif task_name == 'Task_10_Step_Execution_Statement':
        q = f"Given the overall goal {hl or 'the mission'}, for the step {sg or 'this step'}, what specific execution actions are required"
    else:
        q = "Could you provide a concise, grounded answer based on the provided context"
    
    return _finalize_question(q)

def build_answer_prompt(task_name: str, fields: Dict[str, Any]) -> str:
    """
    单阶段高质量 Prompt 构建器：
    将数据上下文、任务指令和严格的格式约束融合，确保一次生成即成品。
    """
    ctx = json.dumps(fields, ensure_ascii=False)
    
    base_instr = (
        "Using the **Input Data** below, answer the user's question. "
        "Strictly adhere to the following structure and tone requirements:"
    )
    
    specific_instr = ""
    
    # --- Group 1: 复杂结构任务 (强制分段) ---
    if task_name == 'Task_06_Holistic_Causal_Chain_Analysis':
        specific_instr = (
            "Produce exactly **TWO distinct paragraphs**:\n"
            "1.  **Context & Setup**: Describe the agent's interaction, spatial setup, and affordance basis.\n"
            "2.  **Mechanism & Effect**: Explain the specific physical mechanism at the hotspot and the immediate effects on the patient/environment.\n"
            "Combine all fields naturally. Do NOT use bullet points."
        )
    elif task_name == 'Task_16_Physical_Feasibility_Verification':
        specific_instr = (
            "Produce exactly **TWO distinct paragraphs**:\n"
            "1.  **Spatial Verification**: Verify the spatial topology conditions.\n"
            "2.  **Affordance Verification**: Verify object properties and affordance states.\n"
            "Ensure clear separation between spatial logic and object property logic."
        )
    elif task_name == 'Task_17_Holistic_Step_Synthesis_Why_How':
        specific_instr = (
            "Produce exactly **TWO distinct paragraphs**:\n"
            "1.  **Strategic Rationale (Why)**: Explain the strategic reason for this step.\n"
            "2.  **Physical Mechanism (How)**: Explain the physical execution details.\n"
            "Do NOT mix strategic intent with execution details."
        )
        
    # --- Group 2: 逐字返回任务 (Verbatim) ---
    elif task_name in {'Task_07_Scene_Goal_Derivation', 'Task_14_Counterfactual_Prediction', 'Task_08_Strategic_Rationale_Justification'}:
        specific_instr = "Return the answer content **verbatim** (word-for-word) from the input data, without adding any introductory or concluding phrases."
        
    # --- Group 3: 单段落融合任务 (Single Paragraph Synthesis) ---
    elif task_name == 'Task_01_Macro_Anchor_Extraction':
        specific_instr = "List the anchors in a single, natural English sentence (e.g., 'The key anchors are X, Y, and Z.'). Do not use a vertical list."
    elif task_name == 'Task_04_Entity_Role_Identification':
        specific_instr = "Write a single sentence identifying which items are tools and which are materials. Integrate them syntactically (e.g., 'The agent uses [Tool] to act upon [Material]')."
    elif task_name == 'Task_12_Inter_Step_Dependency_Analysis':
        specific_instr = "Explain the dependency in a single logical paragraph: How does the effect of the previous step explicitly satisfy the precondition of the next step?"
    elif task_name == 'Task_15_Failure_Recovery_Protocol':
        specific_instr = "Combine the failure reason and recovery strategy into a single conditional statement (e.g., 'If failure occurs due to [Reason], the agent must [Strategy]')."
    elif task_name == 'Task_11_Expected_Physical_Effects':
        specific_instr = "Synthesize all expected effects (macro, spatial, affordance) into one cohesive, flowing paragraph. Do not isolate them."
    elif task_name == 'Task_10_Step_Execution_Statement':
        specific_instr = "Describe the execution actions naturally in a single paragraph. If no actions are listed, use the step goal."
        
    # --- Group 4: 默认处理 (General Natural Response) ---
    else:
        specific_instr = (
            "Synthesize all fields into a single, fluent, academic paragraph. "
            "Weave the data points together naturally using logical connectors. "
            "Do NOT use bullet points or list formatting."
        )

    return (
        f"### INPUT DATA:\n{ctx}\n\n"
        f"### INSTRUCTION:\n{base_instr}\n\n"
        f"{specific_instr}\n\n"
        f"### ANSWER:"
    )

# ==============================================================================
# 3) 工具函数
# ==============================================================================

def _read_image_as_base64(path: str) -> Optional[str]:
    try:
        import base64
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.warning(f"读取图片失败：{path}，原因：{e}")
        return None

def _natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ==============================================================================
# 4) 主类：API 版本生成器（单阶段逻辑）
# ==============================================================================

class PhyPlanAPIGenerator:
    def __init__(self, output_dir: str = "phyplan_output_api", api_config: ApiConfig = None, stream_save: bool = True, resume: bool = True, force: bool = False, processed_keys: Optional[set] = None, live_combined: bool = True, balanced: bool = True):
        self.output_dir = output_dir
        # 初始化所有 Buffer
        self.data_buffer: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ALL_TASK_KEYS}
        self.data_buffer_A: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ALL_TASK_KEYS}
        self.data_buffer_B: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ALL_TASK_KEYS}
        
        self.api_config = api_config or ApiConfig()
        self.client = initialize_api_client(self.api_config)
        self.stream_save = stream_save
        self.resume = resume
        self.force = force
        self.processed_keys = processed_keys or set()
        self.live_combined = live_combined
        self.balanced = balanced

        # 采样配置：每种类型每条数据最多生成的样本数
        self.global_caps_A = {
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
        }
        self.balance_caps = {
            'Task_03_Micro_Affordance_Visual_Semantics': 1,
            'Task_05_State_Evolution_Description': 1,
            'Task_06_Holistic_Causal_Chain_Analysis': 1,
            'Task_02_Transient_Geometric_Verification': 1,
            'Task_16_Physical_Feasibility_Verification': 1,
            'Task_17_Holistic_Step_Synthesis_Why_How': 1,
        }
        self.global_caps_B = {
            'Task_10_Step_Execution_Statement': 2,
            'Task_08_Strategic_Rationale_Justification': 2,
            'Task_09_Precondition_Statement': 2,
            'Task_11_Expected_Physical_Effects': 2,
            'Task_12_Inter_Step_Dependency_Analysis': 2,
            'Task_15_Failure_Recovery_Protocol': 2,
        }
        
        self.global_candidates_A = {k: [] for k in ALL_TASK_KEYS}
        self.global_candidates_B = {k: [] for k in ALL_TASK_KEYS}
        self._init_directories()

    def _init_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for task_name in ALL_TASK_KEYS:
            os.makedirs(os.path.join(self.output_dir, task_name), exist_ok=True)

    def _accept_balance(self, task_name: str, counter: Dict[str, int]) -> bool:
        if not getattr(self, 'balanced', False):
            return True
        cap = self.balance_caps.get(task_name)
        if cap is None:
            return True
        cur = counter.get(task_name, 0)
        if cur >= cap:
            return False
        counter[task_name] = cur + 1
        return True

    def hydrate_existing_buffers(self):
        """预加载已存在的数据以支持 live_combined，但不加载全部内容以省内存"""
        logger.info("Hydrating buffers from disk (Counting only)...")
        for task_name in ALL_TASK_KEYS:
            path_json = os.path.join(self.output_dir, task_name, 'data.json')
            if os.path.exists(path_json):
                try:
                    with open(path_json, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                    # 仅把数据放回 buffer A/B 列表，注意内存消耗
                    # 为了性能，这里我们只保留 '计数' 逻辑如果需要，
                    # 但为了 strict logic adherence，我们还是 append 进去，假设内存够用。
                    # 如果数据量巨大，应改为仅 append 少量元数据。
                    a_entries = obj.get('A', [])
                    b_entries = obj.get('B', [])
                    self.data_buffer_A[task_name].extend(a_entries)
                    self.data_buffer_B[task_name].extend(b_entries)
                    self.data_buffer[task_name].extend(a_entries + b_entries)
                except Exception:
                    pass

    # --- LLM 调用 ---
    def call_llm(self, full_prompt: str, images: Union[str, List[str]]) -> str:
        """单次调用，Fields-Only 模式"""
        if self.client is None or self.api_config.api_key == 'EMPTY':
            return "[API Disabled] " + full_prompt[:100]

        user_content = [{"type": "text", "text": full_prompt}]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
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
            out = resp.choices[0].message.content or ""
            logger.info(f">>> [API] Done in {dt:.2f}s, len={len(out)}")
            return out
        except Exception as e:
            logger.error(f"API 调用失败：{e}")
            return ""

    def _sanitize_text(self, text: str) -> str:
        try:
            s = (text or "").replace("\n", " ")
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"(?<=\w)\s*[-–]\s*(?=\w)", "-", s)
            s = re.sub(r"\s+([,;:\.!\?])", r"\1", s)
            s = re.sub(r"([,;:\.!\?])(\S)", r"\1 \2", s)
            s = re.sub(r"\s+", " ", s)
            return s.strip()
        except Exception:
            return text or ""

    def _final_clean_answer(self, task_name: str, text: str) -> str:
        """单阶段生成的后处理清洗"""
        if not text: return ""
        
        # 1. 移除 Markdown 加粗/斜体符号
        s = text.replace("**", "").replace("*", "")
        
        # 2. 移除常见的废话开场白
        patterns = [
            r"^\s*(In summary|In conclusion|To summarize|Overall|In general),\s*",
            r"^\s*(Based on the input data|According to the input),\s*",
            r"^\s*(Here is|Here's) (the|a) (answer|response|description).*?[:\.]\s*",
            r"^Sure, here is.*?[:\.]\s*"
        ]
        for pat in patterns:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
            
        # 3. 处理段落
        if task_name in {'Task_06_Holistic_Causal_Chain_Analysis', 'Task_16_Physical_Feasibility_Verification', 'Task_17_Holistic_Step_Synthesis_Why_How'}:
            paragraphs = re.split(r'\n\s*\n', s)
            cleaned_paras = [self._sanitize_text(p) for p in paragraphs if p.strip()]
            return "\n\n".join(cleaned_paras)
        else:
            return self._sanitize_text(s)

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
            meta = entry.get('meta', {})
            item_type = (meta.get('item_type') or '').upper()
            if item_type == 'TYPEA':
                self.data_buffer_A[task_name].append(entry)
            elif item_type == 'TYPEB':
                self.data_buffer_B[task_name].append(entry)
            
            if self.stream_save:
                try:
                    file_path = os.path.join(self.output_dir, task_name, "data.jsonl")
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if self.live_combined:
                        self._write_combined(task_name)
                    logger.info(f"[Saved] {task_name} (step={meta.get('step_index')})")
                except Exception as e:
                    logger.warning(f"保存失败: {e}")

    def _write_combined(self, task_name: str):
        a = self.data_buffer_A.get(task_name, [])
        b = self.data_buffer_B.get(task_name, [])
        combined = {"count": len(a)+len(b), "A": a, "B": b}
        out_json = os.path.join(self.output_dir, task_name, "data.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

    def emit_sample(self, task_name: str, images: Union[str, List[str]], prompt: str, user_q: str, meta: Dict[str, Any], resume: bool = True, force: bool = False, fields: Optional[Dict[str, Any]] = None):
        key = f"{meta.get('item_type')}|{meta.get('source_path')}|{task_name}|{meta.get('step_index')}|{meta.get('frame_index')}|{meta.get('sub_index')}"
        if resume and not force and key in self.processed_keys:
            logger.info(f"[Skip] {task_name} key exists")
            return

        final_user_q = self._sanitize_text(user_q)
        final_prompt = f"User Question: {final_user_q}\n\n" + (prompt or "")

        raw_ans = self.call_llm(final_prompt, images)
        final_ans = self._final_clean_answer(task_name, raw_ans)

        if final_ans and len(final_ans) > 5 and "[API Disabled]" not in final_ans:
            entry = self.create_sharegpt_entry(images, final_user_q, final_ans, meta=meta)
            self.save_to_buffer(task_name, entry)
            self.processed_keys.add(key)
        else:
            logger.warning(f"[Drop] 生成内容无效 task={task_name}")

    def _meta(self, item_type, source_path, task_name, step_index, frame_index, sub_index, step_goal, image_path):
        return {
            "item_type": item_type, "source_path": source_path, "task_name": task_name,
            "step_index": step_index, "frame_index": frame_index, "sub_index": sub_index,
            "step_goal": step_goal, "image_path": image_path,
        }
        
    def _normalize_terms(self, text: str) -> set:
        if not text: return set()
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        stop = {"the","a","an","to","of","in","is"}
        return {t for t in tokens if t not in stop and len(t) >= 3}

    def _has_dependency(self, prev_step: Dict, next_step: Dict) -> bool:
        eff = set()
        for e in prev_step.get('expected_effects', []) or []: eff |= self._normalize_terms(str(e))
        pre = set()
        for p in next_step.get('preconditions', []) or []: pre |= self._normalize_terms(str(p))
        return bool(eff & pre)

    # 辅助：统一添加候选
    def _add_cand(self, container, task_name, fields, img, meta_args):
        p = build_answer_prompt(task_name, fields)
        uq = build_user_question(task_name, fields)
        meta = self._meta(*meta_args, image_path=img)
        container[task_name].append({"images": img, "prompt": p, "user_q": uq, "meta": meta, "fields": fields})

    # =========================================================================
    # Type A / Type B 处理器
    # =========================================================================

    def _process_type_a(self, data: Dict, source_path: str):
        global_goal = data.get('high_level_goal', 'N/A')
        steps = data.get('steps', [])
        
        balance_counter: Dict[str, int] = {}

        for i, step in enumerate(steps):
            step_goal = step.get('step_goal', 'Unknown')
            critical_frames = sorted(step.get('critical_frames', []), key=lambda x: int(x.get('frame_index',0)))
            if not critical_frames: continue
            base_img = critical_frames[0].get('keyframe_image_path', 'missing.jpg')
            
            # 1. 步级任务 (Step Level)
            tm = step.get('tool_and_material_usage', {})
            self._add_cand(self.global_candidates_A, 'Task_04_Entity_Role_Identification',
                           {"tools": tm.get('tools'), "materials": tm.get('materials'), "step_goal": step_goal, "high_level_goal": global_goal},
                           base_img, ('TypeA', source_path, 'Task_04_Entity_Role_Identification', i+1, -1, 0, step_goal))

            self._add_cand(self.global_candidates_A, 'Task_08_Strategic_Rationale_Justification',
                           {"global_goal": global_goal, "step_goal": step_goal, "rationale": step.get('rationale')},
                           base_img, ('TypeA', source_path, 'Task_08_Strategic_Rationale_Justification', i+1, -1, 0, step_goal))
                           
            self._add_cand(self.global_candidates_A, 'Task_09_Precondition_Statement',
                           {"step_goal": step_goal, "preconditions": step.get('preconditions'), "high_level_goal": global_goal},
                           base_img, ('TypeA', source_path, 'Task_09_Precondition_Statement', i+1, -1, 0, step_goal))

            if len(steps) > 1 and i < len(steps) - 1:
                next_step = steps[i+1]
                if self._has_dependency(step, next_step):
                    self._add_cand(self.global_candidates_A, 'Task_12_Inter_Step_Dependency_Analysis',
                                   {"step_n_goal": step_goal, "step_n_effect": step.get('expected_effects'), "step_next_goal": next_step.get('step_goal'), "step_next_precondition": next_step.get('preconditions'), "high_level_goal": global_goal},
                                   base_img, ('TypeA', source_path, 'Task_12_Inter_Step_Dependency_Analysis', i+1, -1, 0, step_goal))

            self._add_cand(self.global_candidates_A, 'Task_13_Next_Action_Prediction',
                           {"next_actions": step.get('predicted_next_actions'), "step_goal": step_goal, "high_level_goal": global_goal},
                           base_img, ('TypeA', source_path, 'Task_13_Next_Action_Prediction', i+1, -1, 0, step_goal))

            if 'failure_handling' in step:
                fh = step['failure_handling']
                if isinstance(fh, dict):
                    self._add_cand(self.global_candidates_A, 'Task_15_Failure_Recovery_Protocol',
                                   {"reason": fh.get('reason'), "strategy": fh.get('recovery_strategy'), "step_goal": step_goal, "high_level_goal": global_goal},
                                   base_img, ('TypeA', source_path, 'Task_15_Failure_Recovery_Protocol', i+1, -1, 0, step_goal))

            # Task 11 在 TypeA 中被视为步级任务
            self._add_cand(self.global_candidates_A, 'Task_11_Expected_Physical_Effects',
                           {"expected_effects": step.get('expected_effects'), "spatial_post": step.get('spatial_postconditions_detail'), "affordance_post": step.get('affordance_postconditions_detail'), "step_goal": step_goal, "high_level_goal": global_goal},
                           base_img, ('TypeA', source_path, 'Task_11_Expected_Physical_Effects', i+1, -1, 0, step_goal))

            # 2. 帧级任务 (Frame Level)
            for frame in critical_frames:
                img = frame.get('keyframe_image_path', 'missing.jpg')
                fid = frame.get('frame_index', 0)
                
                ah = frame.get('affordance_hotspot', {})
                if self._accept_balance('Task_03_Micro_Affordance_Visual_Semantics', balance_counter):
                    self._add_cand(self.global_candidates_A, 'Task_03_Micro_Affordance_Visual_Semantics',
                                   {"aff_type": ah.get('affordance_type'), "description": ah.get('description'), "step_goal": step_goal, "high_level_goal": global_goal},
                                   img, ('TypeA', source_path, 'Task_03_Micro_Affordance_Visual_Semantics', i+1, fid, 0, step_goal))

                cc = frame.get('causal_chain', {})
                if self._accept_balance('Task_06_Holistic_Causal_Chain_Analysis', balance_counter):
                    self._add_cand(self.global_candidates_A, 'Task_06_Holistic_Causal_Chain_Analysis',
                                   {"agent": cc.get('agent'), "action": cc.get('action'), "patient": cc.get('patient'), "aff_pre": frame.get('affordance_preconditions'), "mechanism": cc.get('causal_affordance_focus_detail'), "spatial": cc.get('causal_spatial_precondition'), "eff_pat": cc.get('causal_effect_on_patient'), "eff_env": cc.get('causal_effect_on_environment'), "high_level_goal": global_goal},
                                   img, ('TypeA', source_path, 'Task_06_Holistic_Causal_Chain_Analysis', i+1, fid, 0, step_goal))
                
                if self._accept_balance('Task_05_State_Evolution_Description', balance_counter):
                    self._add_cand(self.global_candidates_A, 'Task_05_State_Evolution_Description',
                                   {"action_desc": frame.get('action_description'), "state_change": frame.get('state_change_description'), "high_level_goal": global_goal},
                                   img, ('TypeA', source_path, 'Task_05_State_Evolution_Description', i+1, fid, 0, step_goal))

                for j, sp in enumerate(frame.get('spatial_preconditions', []) or []):
                    objs = sp.get('objects', [])
                    rel = sp.get('relation')
                    if len(objs) >= 2 and rel:
                        if self._accept_balance('Task_02_Transient_Geometric_Verification', balance_counter):
                            self._add_cand(self.global_candidates_A, 'Task_02_Transient_Geometric_Verification',
                                           {"obj_a": str(objs[0]), "obj_b": str(objs[1]), "relation": rel, "high_level_goal": global_goal},
                                           img, ('TypeA', source_path, 'Task_02_Transient_Geometric_Verification', i+1, fid, j+1, step_goal))

                if self._accept_balance('Task_16_Physical_Feasibility_Verification', balance_counter):
                    self._add_cand(self.global_candidates_A, 'Task_16_Physical_Feasibility_Verification',
                                   {"step_goal": step_goal, "spatial_preconditions": frame.get('spatial_preconditions'), "affordance_preconditions": frame.get('affordance_preconditions'), "high_level_goal": global_goal},
                                   img, ('TypeA', source_path, 'Task_16_Physical_Feasibility_Verification', i+1, fid, 0, step_goal))

                mech = frame.get('causal_chain', {}).get('causal_affordance_focus_detail')
                if mech:
                    if self._accept_balance('Task_17_Holistic_Step_Synthesis_Why_How', balance_counter):
                        self._add_cand(self.global_candidates_A, 'Task_17_Holistic_Step_Synthesis_Why_How',
                                       {"step_goal": step_goal, "rationale": step.get('rationale'), "mechanism": mech, "high_level_goal": global_goal},
                                       img, ('TypeA', source_path, 'Task_17_Holistic_Step_Synthesis_Why_How', i+1, fid, 0, step_goal))

        # 3. 采样并生成
        self._flush_candidates(self.global_candidates_A, self.global_caps_A, source_path)

    def _process_type_b(self, data: Dict, source_path: str):
        global_goal = data.get('high_level_goal', 'Unknown')
        frames_dir = data.get('sample_frames_dir', '')
        
        # 模拟图片扫描 (真实环境需要 os.listdir)
        scene_images = []
        if frames_dir and os.path.isdir(frames_dir):
            try:
                files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('jpg','png'))], key=_natural_key)
                scene_images = files
            except Exception: pass
        if not scene_images: scene_images = ["missing_scene.jpg"]
        
        # 1. 场景级任务（直接生成，不参与全局抽样 caps）
        fields1 = {"scene_desc": data.get('scene_description'), "key_objects": data.get('key_objects_for_planning')}
        p1 = build_answer_prompt('Task_01_Macro_Anchor_Extraction', fields1)
        uq1 = build_user_question('Task_01_Macro_Anchor_Extraction', fields1)
        meta1 = self._meta('TypeB', source_path, 'Task_01_Macro_Anchor_Extraction', 0, -1, 0, None)
        self.emit_sample('Task_01_Macro_Anchor_Extraction', scene_images, p1, uq1, meta1, resume=self.resume, force=self.force, fields=fields1)

        fields7 = {"scene_desc": data.get('scene_description'), "high_level_goal": global_goal}
        p7 = build_answer_prompt('Task_07_Scene_Goal_Derivation', fields7)
        uq7 = build_user_question('Task_07_Scene_Goal_Derivation', fields7)
        meta7 = self._meta('TypeB', source_path, 'Task_07_Scene_Goal_Derivation', 0, -1, 0, None)
        self.emit_sample('Task_07_Scene_Goal_Derivation', scene_images, p7, uq7, meta7, resume=self.resume, force=self.force, fields=fields7)

        # 2. 步骤循环 (Type B)
        steps = data.get('steps', [])
        for i, step in enumerate(steps):
            step_goal = step.get('step_goal')
            
            self._add_cand(self.global_candidates_B, 'Task_10_Step_Execution_Statement',
                           {"global_goal": global_goal, "step_goal": step_goal, "actions": step.get('navigation_and_manipulation')},
                           scene_images, ('TypeB', source_path, 'Task_10_Step_Execution_Statement', i+1, -1, 0, step_goal))
            
            self._add_cand(self.global_candidates_B, 'Task_08_Strategic_Rationale_Justification',
                           {"global_goal": global_goal, "step_goal": step_goal, "rationale": step.get('rationale')},
                           scene_images, ('TypeB', source_path, 'Task_08_Strategic_Rationale_Justification', i+1, -1, 0, step_goal))

            self._add_cand(self.global_candidates_B, 'Task_09_Precondition_Statement',
                           {"step_goal": step_goal, "preconditions": step.get('preconditions'), "high_level_goal": global_goal},
                           scene_images, ('TypeB', source_path, 'Task_09_Precondition_Statement', i+1, -1, 0, step_goal))

            self._add_cand(self.global_candidates_B, 'Task_11_Expected_Physical_Effects',
                           {"expected_effects": step.get('expected_effects'), "spatial_post": step.get('spatial_postconditions_detail'), "affordance_post": step.get('affordance_postconditions_detail'), "step_goal": step_goal, "high_level_goal": global_goal},
                           scene_images, ('TypeB', source_path, 'Task_11_Expected_Physical_Effects', i+1, -1, 0, step_goal))

            if len(steps) > 1 and i < len(steps) - 1:
                next_step = steps[i+1]
                if self._has_dependency(step, next_step):
                    self._add_cand(self.global_candidates_B, 'Task_12_Inter_Step_Dependency_Analysis',
                                   {"step_n_goal": step_goal, "step_n_effect": step.get('expected_effects'), "step_next_goal": next_step.get('step_goal'), "step_next_precondition": next_step.get('preconditions')},
                                   scene_images, ('TypeB', source_path, 'Task_12_Inter_Step_Dependency_Analysis', i+1, -1, 0, step_goal))
            
            if 'failure_handling' in step:
                fh = step['failure_handling']
                if isinstance(fh, dict):
                    self._add_cand(self.global_candidates_B, 'Task_15_Failure_Recovery_Protocol',
                                   {"reason": fh.get('reason'), "strategy": fh.get('recovery_strategy'), "step_goal": step_goal},
                                   scene_images, ('TypeB', source_path, 'Task_15_Failure_Recovery_Protocol', i+1, -1, 0, step_goal))
                elif isinstance(fh, list):
                    for idx, item in enumerate(fh):
                        reason, strategy = "failure", str(item)
                        if isinstance(item, str) and ":" in item:
                            reason, strategy = item.split(":", 1)
                        self._add_cand(self.global_candidates_B, 'Task_15_Failure_Recovery_Protocol',
                                       {"reason": reason.strip(), "strategy": strategy.strip(), "step_goal": step_goal},
                                       scene_images, ('TypeB', source_path, 'Task_15_Failure_Recovery_Protocol', i+1, -1, idx+1, step_goal))

        # 3. 采样并生成
        self._flush_candidates(self.global_candidates_B, self.global_caps_B, source_path)

    def _flush_candidates(self, candidates, caps, seed_source):
        rng = random.Random(hash(seed_source or ''))
        for tname, cap in caps.items():
            cands = candidates.get(tname, [])
            if not cands: continue
            pick = rng.sample(cands, k=min(cap, len(cands)))
            for c in pick:
                self.emit_sample(tname, c['images'], c['prompt'], c['user_q'], c['meta'], resume=self.resume, force=self.force, fields=c['fields'])
            candidates[tname] = [] # Clear logic

    def process_entry(self, raw_data: Dict, source_path: str):
        if "steps" in raw_data and any("critical_frames" in s for s in raw_data.get("steps", [])):
            self._process_type_a(raw_data, source_path)
        elif "scene_description" in raw_data:
            self._process_type_b(raw_data, source_path)

    def flush_to_disk(self):
        logger.info("Flushing final JSON outputs...")
        for task_name in ALL_TASK_KEYS:
            self._write_combined(task_name)

# ==============================================================================
# 5) 文件扫描与进度管理辅助
# ==============================================================================

def load_json(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Load failed {path}: {e}")
        return {}

def scan_type_a_root(root: str) -> List[str]:
    results: List[str] = []
    if not os.path.isdir(root): return results
    for name in sorted(os.listdir(root), key=_natural_key):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub): continue
        p1 = os.path.join(sub, 'causal_plan_with_keyframes.json')
        p2 = os.path.join(sub, 'causal_plan.json')
        if os.path.exists(p1): results.append(p1)
        elif os.path.exists(p2): results.append(p2)
    return results

def scan_type_b_root(root: str) -> List[str]:
    results: List[str] = []
    if not os.path.isdir(root): return results
    for name in sorted(os.listdir(root), key=_natural_key):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub): continue
        p = os.path.join(sub, 'plan.json')
        if os.path.exists(p): results.append(p)
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
        if not os.path.exists(p): continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("path"): res[kind].add(obj["path"])
                    except Exception: pass
        except Exception: pass
    return res

def mark_progress(output_dir: str, kind: str, path: str):
    try:
        os.makedirs(_progress_dir(output_dir), exist_ok=True)
        with open(_progress_file(output_dir, kind), 'a', encoding='utf-8') as f:
            f.write(json.dumps({"path": path, "ts": time.time()}, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Mark progress failed: {e}")

def load_existing_sample_keys(output_dir: str) -> set:
    keys = set()
    try:
        for task_name in ALL_TASK_KEYS:
            p = os.path.join(output_dir, task_name, 'data.jsonl')
            if not os.path.exists(p): continue
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        meta = json.loads(line).get('meta', {})
                        if meta:
                            k = f"{meta.get('item_type')}|{meta.get('source_path')}|{task_name}|{meta.get('step_index')}|{meta.get('frame_index')}|{meta.get('sub_index')}"
                            keys.add(k)
                    except Exception: pass
    except Exception: pass
    return keys

# ==============================================================================
# 6) 主程序入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PhyPlan Single-Stage High-Quality Generator')
    parser.add_argument('--typeA-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/causal_spafa_plan_dataset_long')
    parser.add_argument('--typeB-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/generated_plans_output_high_videos')
    parser.add_argument('--output-dir', default='/e2e-data/evad-tech-vla/luzheng/ICML/phyplan_output_api_single_stage')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--types', default='A,B')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--force', action='store_true', default=False)
    
    # API 配置
    parser.add_argument('--api-key', default=os.environ.get('API_KEY', 'sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s'))
    parser.add_argument('--api-base', default=os.environ.get('API_BASE_URL', 'http://model.mify.ai.srv/v1'))
    parser.add_argument('--model', default=os.environ.get('MODEL_NAME', 'gemini-3-pro-preview'))
    
    args = parser.parse_args()

    # 初始化配置
    api_cfg = ApiConfig(api_key=args.api_key, api_base_url=args.api_base, model_name=args.model)
    
    # 加载断点信息
    processed_samples = set()
    if args.resume and not args.force:
        processed_samples = load_existing_sample_keys(args.output_dir)
        logger.info(f"Resuming with {len(processed_samples)} existing samples.")

    # 初始化生成器
    gen = PhyPlanAPIGenerator(output_dir=args.output_dir, api_config=api_cfg, resume=args.resume, force=args.force, processed_keys=processed_samples)
    gen.hydrate_existing_buffers()

    # 扫描文件
    a_items = scan_type_a_root(args.typeA_root)
    b_items = scan_type_b_root(args.typeB_root)
    if args.limit > 0:
        a_items = a_items[:args.limit]
        b_items = b_items[:args.limit]
    
    a_items = [os.path.abspath(p) for p in a_items]
    b_items = [os.path.abspath(p) for p in b_items]

    logger.info(f"Found TypeA: {len(a_items)}, TypeB: {len(b_items)}")

    # 文件级进度
    prog = load_progress(args.output_dir)
    target_types = set([t.strip().upper() for t in args.types.split(',')])

    # 处理 Type A
    if 'A' in target_types:
        for jp in a_items:
            if args.resume and not args.force and jp in prog['typeA']:
                logger.info(f"[Skip] File done: {jp}")
                continue
            logger.info(f"[Start] TypeA: {jp}")
            data = load_json(jp)
            if data:
                gen.process_entry(data, jp)
                mark_progress(args.output_dir, 'typeA', jp)

    # 处理 Type B
    if 'B' in target_types:
        for jp in b_items:
            if args.resume and not args.force and jp in prog['typeB']:
                logger.info(f"[Skip] File done: {jp}")
                continue
            logger.info(f"[Start] TypeB: {jp}")
            data = load_json(jp)
            if data:
                # 规范化路径
                if 'sample_frames_dir' in data and not os.path.isabs(data['sample_frames_dir']):
                    data['sample_frames_dir'] = os.path.join(os.path.dirname(jp), data['sample_frames_dir'])
                gen.process_entry(data, jp)
                mark_progress(args.output_dir, 'typeB', jp)

    # 最终落盘
    gen.flush_to_disk()
    logger.info("All tasks completed.")
