# -*- coding: utf-8 -*-
"""
Causal-SPaFA-Plan Dataset Generation Script
Version: 6.9 (Definitive Fix for API Response Indexing)

This script processes a long video, uniformly samples frames, sends them to an
LMM, and parses the response. It follows the same structure and formatting as
the medium/short variants, and ensures correct API response handling by
indexing `choices[0]` before reading `message.content`.
"""

import base64
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
import json
import os
import re
import time
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

# Suppress OpenAI's internal httpx logging to keep the console clean
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==============================================================================
# === 1. CONFIGURATION PARAMETERS ==============================================
# ==============================================================================

# *** NEW CONFIGURATION: ABSOLUTE PATH TO VIDEO FOLDER ***
# Please modify this path to the absolute path of the folder containing your videos.

@dataclass
class ScriptConfig:
    """Centralized configuration for the script."""
    # --- API Credentials ---
    API_KEY: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    API_BASE_URL: str = "http://model.mify.ai.srv/v1"
    MODEL_PROVIDER_ID: str = "vertex_ai"
    # MODEL_PROVIDER_ID: str = "volcengine_maas"
    MODEL_NAME: str = "gemini-2.5-pro"
    # MODEL_NAME: str = "doubao-1-5-thinking-vision-pro-250428"

    # --- Input/Output Paths ---
    VIDEO_PATH: str = "example.mp4" # This will be updated dynamically in the loop
    OUTPUT_BASE_FOLDER: str = "causal_spafa_plan_dataset_long"
    # OUTPUT_BASE_FOLDER: str = "causal_spafa_plan_dataset_seed"

    # --- Video Processing ---
    MAX_FRAMES_TO_SAMPLE: int = 50
    RESIZE_DIMENSION: Tuple[int, int] = None  # e.g., (1280, 720) or None
    # JPEG_QUALITY: int = 100
    JPEG_QUALITY: int = 100
    # --- Script Behavior ---
    VERBOSE_LOGGING: bool = True
    # Overlay frame index/timestamp onto images sent to the API to avoid off-by-one confusion
    EMBED_INDEX_ON_API_IMAGES: bool = True

# Instantiate the configuration
# The VIDEO_PATH will be overwritten for each file in the folder.
PLANNING_CONFIG = ScriptConfig(
    VIDEO_PATH="placeholder.mp4", 
    OUTPUT_BASE_FOLDER="causal_spafa_plan_dataset_long",
    API_KEY="sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s",
    API_BASE_URL="http://model.mify.ai.srv/v1",
    MODEL_PROVIDER_ID="vertex_ai",
    MODEL_NAME="gemini-2.5-pro",
    VERBOSE_LOGGING=True,
)

SELECTION_CONFIG = ScriptConfig(
    VIDEO_PATH="placeholder.mp4",
    OUTPUT_BASE_FOLDER="causal_spafa_plan_dataset_long",
    API_KEY="sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s",
    API_BASE_URL="http://model.mify.ai.srv/v1",
    MODEL_PROVIDER_ID="vertex_ai",
    MODEL_NAME="gemini-2.5-pro",
    VERBOSE_LOGGING=True,
)

# ==============================================================================
# === 2. DATA STRUCTURE DEFINITIONS (SCHEMA) ===================================
# ==============================================================================

@dataclass
class AffordanceHotspot:
    """Semantic description of the causal focus point of an interaction."""
    description: str
    affordance_type: str = ""
    mechanism: str = ""

@dataclass
class CausalChain:
    """Core causal reasoning structure for a single keyframe."""
    agent: str
    action: str
    patient: str
    causal_effect_on_patient: str
    causal_effect_on_environment: str

@dataclass
class SpatialPrecondition:
    """A single, visually verifiable spatial relationship as a precondition."""
    relation: str
    objects: List[str]
    truth: bool

@dataclass
class AffordancePrecondition:
    """The functional state of an object as a precondition."""
    object_name: str
    affordance_types: List[str]
    reasons: str = ""

@dataclass
class CriticalFrameAnnotation:
    """Complete annotation for a single pivotal moment."""
    frame_index: int
    action_description: str
    spatial_preconditions: List[SpatialPrecondition]
    affordance_preconditions: List[AffordancePrecondition]
    causal_chain: CausalChain
    affordance_hotspot: AffordanceHotspot
    state_change_description: str = ""
    keyframe_image_path: str = None

@dataclass
class FailureHandling:
    """Describes a potential failure and the strategy to recover from it."""
    reason: str
    recovery_strategy: str

@dataclass
class ToolAndMaterialUsage:
    """Lists the specific tools and materials utilized in this step."""
    tools: List[str]
    materials: List[str]

@dataclass
class PlanningStep:
    """A single step in the hierarchical plan."""
    step_id: int
    step_goal: str
    rationale: str
    preconditions: List[str]
    expected_effects: List[str]
    critical_frames: List[CriticalFrameAnnotation]
    tool_and_material_usage: ToolAndMaterialUsage = None
    causal_challenge_question: str = ""
    expected_challenge_outcome: str = ""
    failure_handling: FailureHandling = None

# ==============================================================================
# === 3. CORE UTILITY FUNCTIONS ================================================
# ==============================================================================

def initialize_api_client(config: ScriptConfig) -> Any:
    """Initializes and returns the OpenAI API client."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
            default_headers={"X-Model-Provider-Id": config.MODEL_PROVIDER_ID}
        )
        print(">>> [SUCCESS] OpenAI client initialized successfully.")
        return client
    except ImportError:
        print("!!! [FATAL] 'openai' library not found. Please run 'pip install openai'.")
        return None
    except Exception as e:
        print(f"!!! [FATAL] Failed to initialize OpenAI client: {e}")
        return None

def process_video_to_frames(config: ScriptConfig) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Extracts, resizes, and base64-encodes frames uniformly from a video."""
    if cv2 is None:
        print("!!! [FATAL] 'opencv-python' (cv2) is not installed. Please run 'pip install opencv-python'.")
        return [], None
    print(f"\n>>> [INFO] Starting video processing for: {config.VIDEO_PATH}")
    if not os.path.exists(config.VIDEO_PATH):
        print(f"!!! [ERROR] Video file not found: {config.VIDEO_PATH}")
        return [], None
    video_capture = cv2.VideoCapture(config.VIDEO_PATH)
    if not video_capture.isOpened():
        print(f"!!! [ERROR] Cannot open video file: {config.VIDEO_PATH}")
        return [], None
    frame_data_list, original_dimensions = [], None
    try:
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_dimensions = (width, height)
        if total_frames == 0 or fps == 0:
            print("!!! [ERROR] Video has 0 frames or 0 FPS.")
            return [], original_dimensions
        print(f">>> [INFO] Video Details: {total_frames} frames, {fps:.2f} FPS, ({width}x{height})")
        frame_indices = [int(i * total_frames / config.MAX_FRAMES_TO_SAMPLE) for i in range(config.MAX_FRAMES_TO_SAMPLE)]
        for i, frame_idx in enumerate(frame_indices):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success: continue
            if config.RESIZE_DIMENSION:
                frame = cv2.resize(frame, config.RESIZE_DIMENSION)
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
            frame_data_list.append({
                "base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                "timestamp_sec": frame_idx / fps,
                "original_frame_index": frame_idx
            })
    finally:
        video_capture.release()
    print(f">>> [SUCCESS] Video processing complete. Extracted {len(frame_data_list)} frames.")
    return frame_data_list, original_dimensions

def save_keyframe_images(config: ScriptConfig, annotations: List[CriticalFrameAnnotation], step_output_path: str, all_frame_data: List[Dict[str, Any]]):
    """Extracts and saves the original keyframe images.

    Note: `annotation.frame_index` is treated as 1-based. Internally converted
    to 0-based when indexing `all_frame_data`.
    """
    print(f"  -> Saving {len(annotations)} keyframe images...")
    if cv2 is None:
        print("    !!! [FATAL] 'opencv-python' (cv2) is not installed. Cannot save keyframe images.")
        return
    video_capture = cv2.VideoCapture(config.VIDEO_PATH)
    if not video_capture.isOpened():
        print(f"    !!! [ERROR] Cannot re-open video to save keyframes: {config.VIDEO_PATH}")
        return
    try:
        for anno in annotations:
            try:
                # Convert 1-based index to 0-based for internal lookup
                if anno.frame_index is None:
                    print("    !!! [WARNING] Missing frame_index. Skipping.")
                    continue
                idx0 = int(anno.frame_index) - 1
                if idx0 < 0 or idx0 >= len(all_frame_data):
                    print(f"    !!! [WARNING] Invalid 1-based frame_index {anno.frame_index}. Skipping.")
                    continue
                frame_info = all_frame_data[idx0]
                original_frame_idx = frame_info["original_frame_index"]
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
                success, frame = video_capture.read()
                if not success:
                    print(f"    !!! [WARNING] Failed to capture frame at index {original_frame_idx}.")
                    continue
                if config.RESIZE_DIMENSION:
                    frame = cv2.resize(frame, config.RESIZE_DIMENSION)
                filename = f"frame_{anno.frame_index:03d}_ts_{frame_info['timestamp_sec']:.2f}s.jpg"
                filepath = os.path.join(step_output_path, filename)
                cv2.imwrite(filepath, frame)
                anno.keyframe_image_path = os.path.abspath(filepath)
            except Exception as e:
                print(f"    !!! [ERROR] Error saving keyframe for index {anno.frame_index}: {e}")
    finally:
        video_capture.release()

def save_sampled_frames_jpegs(sampled_frames: List[Dict[str, Any]], output_dir: str):
    """Save all uniformly sampled frames (JPEG) to `output_dir`.

    Naming starts at 1 (not 0): `sample_001_ts_YY.YYs.jpg`.
    Frames are written directly from their base64-encoded JPEG buffers.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"!!! [ERROR] Failed to create frames output dir '{output_dir}': {e}")
        return

    count = 0
    for i, frame in enumerate(sampled_frames):
        try:
            ts = float(frame.get("timestamp_sec", 0.0))
            idx1 = i + 1  # 1-based numbering for saved filenames
            name = f"sample_{idx1:03d}_ts_{ts:.2f}s.jpg"
            path = os.path.join(output_dir, name)
            data = base64.b64decode(frame["base64"]) if isinstance(frame.get("base64"), str) else None
            if not data:
                print(f"    !!! [WARNING] Missing base64 for frame {i}. Skipping.")
                continue
            with open(path, "wb") as f:
                f.write(data)
            count += 1
        except Exception as e:
            print(f"    !!! [WARNING] Failed to save sampled frame {i}: {e}")
    print(f"  -> Saved {count}/{len(sampled_frames)} sampled frames to: {output_dir}")

def build_index_manifest(sampled_frames: List[Dict[str, Any]]) -> str:
    """Return a textual manifest mapping 1-based index to timestamp seconds."""
    lines = ["Frame Index Manifest (1-based):"]
    for i, frame in enumerate(sampled_frames, start=1):
        ts = float(frame.get("timestamp_sec", 0.0))
        lines.append(f"- Frame {i}: t={ts:.2f}s")
    return "\n".join(lines)

def _overlay_index_on_base64_image(b64_img: str, index_1based: int, timestamp_sec: float) -> str:
    """Overlay index and timestamp onto a base64 JPEG image and return new base64."""
    try:
        import numpy as np
        data = base64.b64decode(b64_img)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64_img
        text = f"Frame {index_1based:02d}  t={timestamp_sec:.2f}s"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return b64_img
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return b64_img

def build_api_content(sampled_frames: List[Dict[str, Any]], embed_index: bool) -> List[Dict[str, Any]]:
    """Build message content list: manifest + alternating text + image items in 1-based order."""
    content: List[Dict[str, Any]] = []
    manifest = build_index_manifest(sampled_frames)
    content.append({"type": "text", "text": manifest})
    for i, frame in enumerate(sampled_frames, start=1):
        ts = float(frame.get("timestamp_sec", 0.0))
        b64 = frame.get("base64")
        if embed_index and isinstance(b64, str):
            b64 = _overlay_index_on_base64_image(b64, i, ts)
        content.append({"type": "text", "text": f"Frame {i}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content

def sanitize_filename(text: str) -> str:
    """Cleans a string to be a valid folder/file name."""
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '_', text)
    return text

def extract_json_from_response(response_text: str) -> str:
    """Extracts a JSON string from the model's response using multiple strategies."""
    if not isinstance(response_text, str):
        raise ValueError("Input to extract_json_from_response was not a string.")

    # Strategy 1: Look for a markdown ```json ... ``` block
    match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
    if match:
        print(">>> [INFO] Strategy 1: Found JSON within a markdown block.")
        return match.group(1).strip()

    # Strategy 2: If no markdown, find the first '{' and last '}'
    print(">>> [INFO] Strategy 1 failed. Trying Strategy 2: Find first '{' and last '}' characters.")
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return response_text[start_brace : end_brace + 1].strip()

    raise ValueError("Could not find a valid JSON structure in the model's response.")

# ==============================================================================
# === 4. PROMPT ENGINEERING ====================================================
# ==============================================================================

# *** SYSTEM PROMPT ***
system_prompt = """
You are a highly advanced AI acting as a Physical Interaction Analyst and Causal Planner. Your primary mission is to deconstruct observed actions in video frames into their fundamental causal, spatial, and affordance-based physical principles.
You must analyze key moments from a continuous action sequence to produce a hierarchical task plan. This plan must explain not just *what* happened, but precisely *how* and *why* it is happening from a physical standpoint by inferring the dynamics implied within each key moment.
Your output MUST be a single, syntactically flawless JSON object. The level of detail and adherence to the causal schema is paramount.
"""

def create_planning_user_prompt(num_frames: int, image_dimensions: Tuple[int, int]) -> str:
    """Generates the full, detailed user prompt for the LMM API call."""
    return f"""
You are a world-class AI, a doctorate-level expert in physics, robotics, and cognitive science, acting as a **Physical Interaction Analyst and Causal Planner**. Your primary mission is to deconstruct observed human actions from video frames into their most fundamental causal, kinetic, and physical principles. You must think step-by-step with extreme precision, logical rigor, and unwavering adherence to the specified JSON schema. Your output is not just a description; it is a scientific annotation.

Analyze the provided {num_frames} frames, which are uniformly sampled from a continuous video of a task. Your task is to reverse-engineer the high-level goal and generate a deeply detailed, **hierarchical causal plan**. This plan must be broken down into a reasonable number of fine-grained, logical steps.

Treat the frames as the ONLY source of truth. Use conservative language when uncertain and prefer generic object naming (e.g., "container", "bottle", "tool") over guessing brands or invisible states.

Your response MUST be a single, syntactically flawless JSON object. No extra text, no apologies, no explanations outside of the JSON structure. The JSON validity is a critical, non-negotiable part of the task.

**Detailed JSON Schema to Follow:**
{{
  "high_level_goal": "A single, comprehensive English sentence capturing the ultimate purpose and outcome of the entire task.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "A highly specific, action-oriented description of the sub-goal for this step. (e.g., 'Transfer raw fish fillets from packaging to the prepared baking tray').",
      
      "rationale": "The high-level reasoning for this entire step. Explain *why* this step is a necessary and logical component of the overall plan, linking it to previous or future steps.",
      
      "preconditions": [
          "A list of **macro-level, abstract** states that must be true before this step can begin. (e.g., 'Oven is preheated', 'All ingredients are gathered')."
      ],
      "expected_effects": [
          "A list of **macro-level, abstract** states that will be true after this step is fully completed. IMPORTANT: explicitly include (1) spatial outcomes (e.g., 'X is inside/on/next to Y', 'X is aligned with Y', 'X is held by hand') and (2) affordance/state outcomes (e.g., 'object is opened/closed/peeled/ready_to_use', 'liquid transferred', 'container emptied') whenever applicable."
      ],

      "tool_and_material_usage": {{
        "tools": ["A list of all tools being actively used in this step. (e.g., 'knife', 'cutting board')."],
        "materials": ["A list of all materials being acted upon or consumed in this step. (e.g., 'cucumber', 'olive oil')."]
      }},

      "causal_challenge_question": "A specific, insightful 'what-if' question that challenges the core causal or physical understanding of this step. (e.g., 'What if the knife was dull?').",
      "expected_challenge_outcome": "The predicted outcome for the causal challenge question, explaining the physical consequences.",

      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }},

      "critical_frames": [
        {{
          "action_description": "A rich, detailed, and objective description of what the actor is physically doing in this **single frame**. Use precise verbs and describe the observable motion.",
          "state_change_description": "A concise description of the observable change of state that has *just occurred or is occurring* in this frame as a result of the action. (e.g., 'A piece of cucumber is now fully severed from the main body.')",

          "spatial_preconditions": [
            {{
              "relation": "A visually verifiable spatial/physical relationship that is true *at the moment of this frame*. Be highly detailed and mechanistic (contact type, alignment, support, containment, relative pose, occlusion, reachability). Prefer multiple precise constraints over a single vague statement.",
              "objects": ["object_a", "object_b"],
              "truth": true
            }}
          ],
          "affordance_preconditions": [
            {{ "object_name": "knife", "affordance_types": ["cutting_active", "sharp"], "reasons": "The knife is visibly cleaving the cucumber, which implies it is sharp enough for the task." }}
          ],

          "causal_chain": {{
              "agent": "The primary entity that initiates force or action (e.g., 'hand', 'knife_blade'). Be specific.",
              "action": "A concise verb phrase describing the core physical action (e.g., 'is applying downward force', 'is rotating').",
              "patient": "The primary entity being directly acted upon (e.g., 'cucumber', 'kettle').",
              "causal_effect_on_patient": "A description of the **ONGOING physical state change** on the `patient` at this instant, using physics terminology. e.g., 'is undergoing plastic deformation and fracture'.",
              "causal_effect_on_environment": "A description of the **ONGOING observable effect** on the surrounding environment. e.g., 'a slice of the cucumber is beginning to separate'."
          }},
          "affordance_hotspot": {{
              "description": "A detailed semantic description of the specific functional part of an object that is the causal focus. e.g., 'the sharp cutting edge of the knife'.",
              "affordance_type": "A specific category of affordance being acted upon (e.g., 'cutting_edge', 'grabbable_handle').",
              "mechanism": "A detailed utilization mechanism: how the hotspot is used to transmit force/motion/constraint to produce the causal effect (e.g., edge concentrates stress to induce fracture; handle provides torque leverage; rim guides pouring flow)."
          }}
        }},
        ...
      ]
    }},
    ...
  ]
}}

**CRITICAL INSTRUCTIONS TO FOLLOW AT ALL COSTS:**

1.  **Extreme Detail and Objectivity:** Every description must be highly detailed, objective, and grounded in visual evidence.
2.  **Scientific Causal Reasoning:** All fields within `causal_chain` MUST be plausible and consistent with the principles of physics and dynamics.
3.  **Focus on the Critical Frame:** `spatial_preconditions` and `affordance_preconditions` MUST comprehensively describe the state of the world **in that specific frame**. `spatial_preconditions` must be highly detailed and mechanistic (contact type, alignment/pose, support, containment, relative positioning, occlusion, reachability/hand-object coupling).
4.  **Principle of Pivotal Moments & Each Step with Multiple Critical Frames:** A single `step` can and often should contain multiple `critical_frames`. There are usually 1-3 critical frames. Identify ALL distinct and meaningful pivotal moments.
5.  **Infer Dynamics from a Snapshot:** Your descriptions must infer motion, force, and consequence from a single, static key frame.
6.  **Complete All Fields:** Ensure every single field in the schema is filled with a meaningful and accurate value.
7.  **Critical Frames :** In this Stage 1 planning phase, DO NOT include any image references or frame indices in the JSON. Only provide the textual fields for `critical_frames` (action_description, state_change_description, spatial_preconditions, affordance_preconditions, causal_chain, affordance_hotspot). Frame selection happens later.
8.  **Grounding & Non‑Hallucination:** Base all facts strictly on what is visible in the provided frames. Do not invent objects, brands, or states that are not visually supported. If uncertain, use generic terms (e.g., "container", "bottle").
9.  **Consistency Requirement:** Use consistent object names across steps and frames (e.g., do not switch between "pan" and "tray" unless you are sure they are different). Prefer a stable canonical name.
10. **JSON Structure is Paramount:** Adhere strictly to the schema. Note that `causal_chain` and `affordance_hotspot` are **SIBLING** objects; `affordance_hotspot` is **NOT** inside `causal_chain`.
11. **High requirements for planning:** The overall plan must be detailed and comprehensive, representing all events implied by the frames. Each sub-step must be specific and detailed.
12. **Step with fine granularity:** If a frame implies multiple micro-events, split them into separate sub-steps. Each step should be precise, not a broad summary.
13. **The completeness of the plan and the detail of the sub-plans:** It is essential to ensure that the planning and labeling is a comprehensive plan encompassing all provided images. The sub-steps should be detailed, thorough, and logically structured, so that when combined, they present a complete plan comprised of all images.
14. **Principle of Visual Anchorability:** While you are not selecting frames now, every critical_frame you describe must correspond to a visually distinct and unambiguous moment. Your descriptions should be "anchor-able" to a plausible visual snapshot. Do not describe events that are inherently invisible or highly ambiguous from a third-person perspective.
---

TEMPORAL ALIGNMENT REQUIREMENTS (DO NOT IGNORE):
1) The ordering of your `steps` MUST strictly follow the chronological order of the {num_frames} frames as provided (earliest frames first, latest frames last).
2) Do NOT reorder events out of time; ensure that earlier events described in earlier frames appear in earlier steps.
3) Within each step, maintain descriptions consistent with the earlier-to-later progression implied by the frames.

Now, based on the uniformly sampled frames I have provided from a continuous video, and adhering strictly to all constraints (including temporal alignment) and the highest standards of quality, generate the complete and detailed JSON output for this video.
"""

# ==============================================================================
# === 5. MAIN EXECUTION LOGIC ==================================================
# ==============================================================================

def _filter_plan_remove_keyframe_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied plan with keyframe-specific fields removed.

    Removes 'keyframe_image_path' and 'frame_index' from each item in 'critical_frames'.
    """
    def _clean_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
        new_frame = {k: v for k, v in frame.items() if k not in ("keyframe_image_path", "frame_index")}
        return new_frame

    cleaned = {
        "high_level_goal": plan.get("high_level_goal"),
        "steps": []
    }
    for step in plan.get("steps", []):
        step_copy = {k: v for k, v in step.items() if k != "critical_frames"}
        cfs = step.get("critical_frames", [])
        step_copy["critical_frames"] = [_clean_frame(cf) for cf in cfs]
        cleaned["steps"].append(step_copy)
    return cleaned

def _create_frame_selection_prompt(plan_json_str: str, num_frames: int) -> str:
    """Build a Stage 2 prompt to QA+improve the plan and align keyframes.

    Stage 2 MUST:
    - Verify and improve Stage 1 planning using ONLY the provided frames.
    - Assign a valid 1-based `frame_index` to EVERY critical frame.
    - Preserve ALL fields from the Stage 1 schema (do not remove any field).
    - Output strict JSON only (no markdown, no extra text).
    """
    return f"""
You are a meticulous, world-class vision-grounded physical interaction analyst and causal planner.

You are given:
1) {num_frames} uniformly sampled video frames in strict chronological order (Frame 1 is earliest).
2) A DRAFT plan JSON produced in Stage 1.

Your mission in THIS Stage 2 is to produce a FINAL, HIGH-QUALITY, FULLY-GROUNDED plan and perfect keyframe alignment.

A. Vision-grounded QA & plan improvement (mandatory)
- Treat the Stage 1 plan as a DRAFT that may contain omissions, hallucinations, or non-visual claims.
- You MUST verify every step and every critical frame against the provided images.
- If any text is inaccurate, incomplete, ambiguous, or not visually supportable, you MUST revise it so it becomes fully supported by the frames.
- If Stage 1 missed important events implied by the frames, you MUST add the missing steps/critical frames.
- If Stage 1 contains events not supported by the frames, you MUST correct or remove those events.

Quality requirements (non-negotiable)
- Keep object naming consistent across steps/frames (use a canonical name).
- Avoid hidden-state claims (e.g., "turned on", "heated", "cooked") unless there is strong visual evidence; if not visible, describe only what is directly observed.
- For every step, ensure `expected_effects` explicitly captures spatial outcomes and affordance/state outcomes that are visually grounded.
- For every critical frame, ensure `spatial_preconditions` is comprehensive and mechanistic (contact type, alignment/pose, support, containment, relative positioning, reachability/hand-object coupling). Do not collapse the spatial story into a single vague line.
- For every critical frame, ensure `affordance_hotspot.mechanism` concretely explains the physical utilization mechanism (force transmission, leverage, constraint, guidance) consistent with the image.

B. Perfect keyframe alignment (mandatory)
- For EVERY `critical_frames` entry you output, you MUST set `frame_index` to the single best-matching frame number in [1, {num_frames}].
- The chosen frame MUST satisfy the conjunctive constraints of that critical frame.
- If a draft critical frame cannot be matched to any frame, you MUST rewrite the critical frame so it is anchorable, then assign a frame_index.
- Within each step, `frame_index` values MUST be non-decreasing.
- Across steps, earlier steps MUST not be assigned later frames than subsequent steps.

C. Step creation and renumbering (allowed and required when needed)
- You MAY add/remove/merge/split steps to improve completeness and correctness.
- You MUST output step_ids as a continuous sequence starting at 1.
- The ordering of steps MUST follow the chronological order of the frames.

D. Strict schema and field preservation (critical)
- You MUST preserve ALL fields from the Stage 1 schema in your final output. Do not omit any field.
- Do NOT add any new top-level fields or any extra audit/debug fields.
- Do NOT output `confidence`, `checks`, `matched_relations`, `unmatched_relations`, or any other non-schema keys.
- Output STRICT JSON ONLY. No markdown fences, no commentary, no extra text.

E. Grounding guidance (to avoid hallucination while preserving all fields)
- In `expected_effects`, explicitly emphasize the step's spatial outcomes (object relationships/placements) and affordance/state outcomes (opened/closed/held/emptied/ready_to_use, etc.) whenever applicable.
- For fields that require reasoning but cannot be directly verified (e.g., rationale, expected_effects, failure_handling, challenge question/outcome), keep them plausible and conservative, consistent with visible evidence.
- Never invent specific objects, brands, or states not visually supported.


REQUIRED OUTPUT FORMAT (strict JSON only):

{{
  "high_level_goal": "A single, comprehensive English sentence capturing the ultimate purpose and outcome of the entire task.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "A highly specific, action-oriented description of the sub-goal for this step. (e.g., 'Transfer raw fish fillets from packaging to the prepared baking tray').",
      
      "rationale": "The high-level reasoning for this entire step. Explain *why* this step is a necessary and logical component of the overall plan, linking it to previous or future steps.",
      
      "preconditions": [
          "A list of **macro-level, abstract** states that must be true before this step can begin. (e.g., 'Oven is preheated', 'All ingredients are gathered')."
      ],
      "expected_effects": [
          "A list of **macro-level, abstract** states that will be true after this step is fully completed. IMPORTANT: explicitly include (1) spatial outcomes (e.g., 'X is inside/on/next to Y', 'X is aligned with Y', 'X is held by hand') and (2) affordance/state outcomes (e.g., 'object is opened/closed/peeled/ready_to_use', 'liquid transferred', 'container emptied') whenever applicable."
      ],

      "tool_and_material_usage": {{
        "tools": ["A list of all tools being actively used in this step. (e.g., 'knife', 'cutting board')."],
        "materials": ["A list of all materials being acted upon or consumed in this step. (e.g., 'cucumber', 'olive oil')."]
      }},

      "causal_challenge_question": "A specific, insightful 'what-if' question that challenges the core causal or physical understanding of this step. (e.g., 'What if the knife was dull?').",
      "expected_challenge_outcome": "The predicted outcome for the causal challenge question, explaining the physical consequences.",

      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }},

      "critical_frames": [
        {{
          "frame_index": 1,
          "action_description": "A rich, detailed, and objective description of what the actor is physically doing in this **single frame**. Use precise verbs and describe the observable motion.",
          "state_change_description": "A concise description of the observable change of state that has *just occurred or is occurring* in this frame as a result of the action. (e.g., 'A piece of cucumber is now fully severed from the main body.')",

          "spatial_preconditions": [
            {{
              "relation": "A visually verifiable spatial/physical relationship that is true *at the moment of this frame*. Be highly detailed and mechanistic (contact type, alignment, support, containment, relative pose, occlusion, reachability). Prefer multiple precise constraints over a single vague statement.",
              "objects": ["object_a", "object_b"],
              "truth": true
            }}
          ],
          "affordance_preconditions": [
            {{ "object_name": "knife", "affordance_types": ["cutting_active", "sharp"], "reasons": "The knife is visibly cleaving the cucumber, which implies it is sharp enough for the task." }}
          ],

          "causal_chain": {{
              "agent": "The primary entity that initiates force or action (e.g., 'hand', 'knife_blade'). Be specific.",
              "action": "A concise verb phrase describing the core physical action (e.g., 'is applying downward force', 'is rotating').",
              "patient": "The primary entity being directly acted upon (e.g., 'cucumber', 'kettle').",
              "causal_effect_on_patient": "A description of the **ONGOING physical state change** on the `patient` at this instant, using physics terminology. e.g., 'is undergoing plastic deformation and fracture'.",
              "causal_effect_on_environment": "A description of the **ONGOING observable effect** on the surrounding environment. e.g., 'a slice of the cucumber is beginning to separate'."
          }},
          "affordance_hotspot": {{
              "description": "A detailed semantic description of the specific functional part of an object that is the causal focus. e.g., 'the sharp cutting edge of the knife'.",
              "affordance_type": "A specific category of affordance being acted upon (e.g., 'cutting_edge', 'grabbable_handle').",
              "mechanism": "A detailed utilization mechanism: how the hotspot is used to transmit force/motion/constraint to produce the causal effect (e.g., edge concentrates stress to induce fracture; handle provides torque leverage; rim guides pouring flow)."
          }}
        }},
        ...
      ]
    }},
    ...
  ]
}}

Reference DRAFT plan JSON from Stage 1 (QA and improve it; do not echo it separately):
{plan_json_str}
"""


def _validate_stage2_plan(plan: Dict[str, Any], num_frames: int) -> Tuple[bool, List[str]]:
    """Validate Stage 2 output plan structure and alignment constraints."""
    errors: List[str] = []
    if not isinstance(plan, dict):
        return False, ["Top-level output is not a JSON object."]

    if "high_level_goal" not in plan:
        errors.append("Missing required top-level field: high_level_goal")
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("Missing or empty required top-level field: steps")
        return False, errors

    required_step_fields = {
        "step_id",
        "step_goal",
        "rationale",
        "preconditions",
        "expected_effects",
        "tool_and_material_usage",
        "causal_challenge_question",
        "expected_challenge_outcome",
        "failure_handling",
        "critical_frames",
    }
    forbidden_keys = {"confidence", "checks", "matched_relations", "unmatched_relations"}

    required_tool_usage_fields = {"tools", "materials"}
    required_failure_fields = {"reason", "recovery_strategy"}

    step_ids: List[int] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step at index {idx} is not an object")
            continue

        missing = required_step_fields - set(step.keys())
        if missing:
            errors.append(f"Step {step.get('step_id', idx)} missing fields: {sorted(missing)}")
        if forbidden_keys & set(step.keys()):
            errors.append(
                f"Step {step.get('step_id', idx)} contains forbidden keys: {sorted(list(forbidden_keys & set(step.keys())))}"
            )

        sid = step.get("step_id")
        if not isinstance(sid, int):
            errors.append(f"Step at index {idx} has non-int step_id")
            continue
        step_ids.append(sid)

        # Validate nested required objects
        tool_usage = step.get("tool_and_material_usage")
        if not isinstance(tool_usage, dict) or (required_tool_usage_fields - set(tool_usage.keys())):
            errors.append(f"Step {sid} tool_and_material_usage missing tools/materials")
        failure = step.get("failure_handling")
        if not isinstance(failure, dict) or (required_failure_fields - set(failure.keys())):
            errors.append(f"Step {sid} failure_handling missing reason/recovery_strategy")

        cfs = step.get("critical_frames")
        if not isinstance(cfs, list) or len(cfs) == 0:
            errors.append(f"Step {sid} critical_frames missing or empty")
            continue

        prev_idx = -1
        for j, cf in enumerate(cfs):
            if not isinstance(cf, dict):
                errors.append(f"Step {sid} critical_frame at index {j} is not an object")
                continue
            if forbidden_keys & set(cf.keys()):
                errors.append(
                    f"Step {sid} critical_frame {j} contains forbidden keys: {sorted(list(forbidden_keys & set(cf.keys())))}"
                )
            fi = cf.get("frame_index")
            if not isinstance(fi, int):
                errors.append(f"Step {sid} critical_frame {j} missing/non-int frame_index")
                continue
            if fi < 1 or fi > num_frames:
                errors.append(f"Step {sid} critical_frame {j} frame_index out of range: {fi}")
            if fi < prev_idx:
                errors.append(f"Step {sid} critical_frames frame_index not non-decreasing: {prev_idx} -> {fi}")
            prev_idx = fi



    # Enforce step_id is exactly [1..N]
    if step_ids:
        expected = list(range(1, len(step_ids) + 1))
        if sorted(step_ids) != expected:
            errors.append(f"step_id must be continuous 1..N. Got: {sorted(step_ids)}")

    # Across-steps macro temporal order check
    try:
        # sort by step_id to evaluate ordering
        by_id = {s.get("step_id"): s for s in steps if isinstance(s, dict) and isinstance(s.get("step_id"), int)}
        prev_min = -1
        for sid in range(1, len(by_id) + 1):
            step = by_id.get(sid)
            if not step:
                continue
            cfs = step.get("critical_frames", [])
            frame_indices = [
                cf.get("frame_index") for cf in cfs if isinstance(cf, dict) and isinstance(cf.get("frame_index"), int)
            ]
            if not frame_indices:
                continue
            m = min(frame_indices)
            if m < prev_min:
                errors.append("Across-steps temporal order violated (min frame_index decreases)")
                break
            prev_min = m
    except Exception as e:
        errors.append(f"Temporal order validation error: {e}")

    return len(errors) == 0, errors


def _build_alignment_spec_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact spec used for alignment verification (no extra fields)."""
    spec_steps: List[Dict[str, Any]] = []
    for step in plan.get("steps", []) if isinstance(plan, dict) else []:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id")
        cfs_out: List[Dict[str, Any]] = []
        for cf in step.get("critical_frames", []) if isinstance(step.get("critical_frames"), list) else []:
            if not isinstance(cf, dict):
                continue
            cfs_out.append(
                {
                    "frame_index": cf.get("frame_index"),
                    "action_description": cf.get("action_description", ""),
                    "state_change_description": cf.get("state_change_description", ""),
                    "spatial_preconditions": cf.get("spatial_preconditions", []),
                    "affordance_preconditions": cf.get("affordance_preconditions", []),
                    "causal_chain": cf.get("causal_chain", {}),
                    "affordance_hotspot": cf.get("affordance_hotspot", {}),
                }
            )
        spec_steps.append({"step_id": step_id, "critical_frames": cfs_out})
    return {"high_level_goal": plan.get("high_level_goal", ""), "steps": spec_steps}


def _create_alignment_verification_prompt(alignment_spec_json: str, num_frames: int) -> str:
    """Build a prompt for the Stage 2B alignment verification round."""
    return f"""
You are a strict keyframe alignment VERIFIER.

You are given:
1) {num_frames} uniformly sampled frames in chronological order.
2) A compact alignment specification listing each critical frame's constraints AND its currently assigned `frame_index`.

Task:
- Verify whether each critical frame's current `frame_index` is truly the best-matching frame among ALL provided frames.
- If not, choose a better `frame_index` in [1, {num_frames}] that maximally satisfies the critical frame constraints.
- You MUST NOT rewrite, paraphrase, or modify any text fields; only decide indices.
- Keep the number of steps and the number/order of critical frames within each step exactly the same as the spec.
- Enforce non-decreasing `frame_index` within each step.
- Enforce macro temporal consistency across steps.

Output strict JSON only, with NO extra keys, using this exact schema:
{{
  "steps": [
    {{
      "step_id": <int>,
      "critical_frames": [
        {{"frame_index": <int>}},
        ...
      ]
    }},
    ...
  ]
}}

Alignment specification (read-only):
{alignment_spec_json}
"""


def _apply_alignment_updates_in_place(plan: Dict[str, Any], updates: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Apply verification `frame_index` updates onto a plan in place."""
    errors: List[str] = []
    if not isinstance(plan, dict) or not isinstance(updates, dict):
        return False, ["Plan or updates is not a dict."]
    if not isinstance(updates.get("steps"), list):
        return False, ["Updates missing steps list."]

    plan_steps = plan.get("steps", [])
    if not isinstance(plan_steps, list):
        return False, ["Plan missing steps list."]
    plan_by_id = {
        int(s.get("step_id")): s
        for s in plan_steps
        if isinstance(s, dict) and isinstance(s.get("step_id"), int)
    }

    for upd_step in updates.get("steps", []):
        if not isinstance(upd_step, dict):
            errors.append("Update step is not an object")
            continue
        sid = upd_step.get("step_id")
        if not isinstance(sid, int) or sid not in plan_by_id:
            errors.append(f"Unknown or invalid step_id in updates: {sid}")
            continue
        upd_cfs = upd_step.get("critical_frames")
        plan_cfs = plan_by_id[sid].get("critical_frames")
        if not isinstance(upd_cfs, list) or not isinstance(plan_cfs, list):
            errors.append(f"Missing critical_frames list for step_id {sid}")
            continue
        if len(upd_cfs) != len(plan_cfs):
            errors.append(f"critical_frames length mismatch for step_id {sid}: {len(upd_cfs)} vs {len(plan_cfs)}")
            continue
        for i, upd_cf in enumerate(upd_cfs):
            if not isinstance(upd_cf, dict) or "frame_index" not in upd_cf:
                errors.append(f"Step {sid} update critical_frame {i} missing frame_index")
                continue
            fi = upd_cf.get("frame_index")
            if not isinstance(fi, int):
                errors.append(f"Step {sid} update critical_frame {i} non-int frame_index")
                continue
            plan_cfs[i]["frame_index"] = fi

    return len(errors) == 0, errors


def process_single_video(video_file_path: str):
    """Processes a single video file using the defined pipeline with resume support.

    Resume behavior:
    - If final output `causal_plan_with_keyframes.json` exists, skip this video.
    - If only `causal_plan.json` exists, skip Stage 1 and run Stage 2 only.
    - Otherwise, run Stage 1 then Stage 2.
    """
    # Update global configs for the current video
    PLANNING_CONFIG.VIDEO_PATH = video_file_path
    SELECTION_CONFIG.VIDEO_PATH = video_file_path

    print(f"\n==============================================================================")
    print(f"=== PROCESSING VIDEO: {video_file_path}")
    print(f"==============================================================================")

    # Determine output paths and resume flags early
    video_filename_base, _ = os.path.splitext(os.path.basename(PLANNING_CONFIG.VIDEO_PATH))
    video_output_folder = os.path.join(PLANNING_CONFIG.OUTPUT_BASE_FOLDER, video_filename_base)
    sampled_frames_dir = os.path.join(video_output_folder, "sampled_frames")
    try:
        os.makedirs(video_output_folder, exist_ok=True)
    except Exception as e:
        print(f"!!! [ERROR] Failed to create video output folder '{video_output_folder}': {e}")
        return

    stage1_path = os.path.join(video_output_folder, "causal_plan.json")
    stage2_path = os.path.join(video_output_folder, "causal_plan_with_keyframes.json")

    # If Stage 2 already completed, skip entirely
    if os.path.exists(stage2_path):
        print(f">>> [INFO] Final plan already exists. Skipping: {video_output_folder}")
        return

    # Decide whether to resume at Stage 2
    resume_stage2_only = os.path.exists(stage1_path)

    print(">>> [INFO] Script started for single video.")

    # Initialize planning client only if running Stage 1
    planning_client = None
    if not resume_stage2_only:
        planning_client = initialize_api_client(PLANNING_CONFIG)
        if not planning_client:
            return  # Skip this video if client fails

    # 0) Extract frames (always re-extract for reproducibility)
    sampled_frames, original_dims = process_video_to_frames(PLANNING_CONFIG)
    if not sampled_frames:
        print(f"!!! [ERROR] No frames extracted for {video_file_path}. Skipping.")
        return

    # Persist the sampled frames alongside outputs
    save_sampled_frames_jpegs(sampled_frames, sampled_frames_dir)

    response_content = None
    filtered_plan = None
    high_level_goal = None
    steps_data = None

    if not resume_stage2_only:
        # 1) First call: generate plan JSON only (no keyframe images/paths and no frame_index)
        print("\n>>> [INFO] Building API request payload (Stage 1: plan only)...")
        try:
            user_prompt = create_planning_user_prompt(len(sampled_frames), original_dims)
            # Build content with an index manifest and 1-based per-image markers
            user_content = [{"type": "text", "text": user_prompt}] + build_api_content(sampled_frames, getattr(PLANNING_CONFIG, 'EMBED_INDEX_ON_API_IMAGES', True))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            print(f">>> [SUCCESS] API payload built with {len(sampled_frames)} frames.")
        except Exception as e:
            print(f"!!! [FATAL] Failed to build API request: {e}")
            return

        try:
            print(f"\n>>> [INFO] Sending request to model '{PLANNING_CONFIG.MODEL_NAME}' (Stage 1)...")
            start_time = time.time()
            response = planning_client.chat.completions.create(model=PLANNING_CONFIG.MODEL_NAME, messages=messages, max_tokens=30000)
            end_time = time.time()

            if not (response and response.choices and len(response.choices) > 0):
                print("!!! [ERROR] API response is invalid or does not contain any 'choices'.")
                if response:
                    print(">>> [DEBUG] Full invalid response object:", response)
                return
            first_choice = response.choices[0]
            if not hasattr(first_choice, 'message') or not hasattr(first_choice.message, 'content'):
                print("!!! [ERROR] The first choice object is missing 'message' or 'content' attributes.")
                print(">>> [DEBUG] First choice object:", first_choice)
                return
            response_content = first_choice.message.content
            if PLANNING_CONFIG.VERBOSE_LOGGING and response_content:
                print("\n>>> [DEBUG] Raw API Response Content (Stage 1):")
                print(response_content)
            print(f">>> [SUCCESS] Stage 1 API call in {end_time - start_time:.2f}s.")
        except Exception as e:
            print(f"\n!!! [FATAL] API call error (Stage 1): {e}")
            import traceback
            traceback.print_exc()
            return

        if not response_content:
            print("\n!!! [ERROR] No content extracted from API response (Stage 1).")
            return

        # Parse and save plan without keyframe images and indices
        try:
            print("\n>>> [INFO] Parsing JSON response (Stage 1)...")
            clean_json_string = extract_json_from_response(response_content)
            plan_data = json.loads(clean_json_string)

            high_level_goal = plan_data.get('high_level_goal', 'No Goal Provided')
            steps_data = plan_data.get('steps', [])
            print(f">>> [SUCCESS] JSON parsed. High-Level Goal: {high_level_goal}")

            print(f">>> [IO] Output folder: {video_output_folder}")

            # Save plan without keyframe fields
            final_plan_to_save = {
                "high_level_goal": high_level_goal,
                "steps": steps_data
            }
            filtered_plan = _filter_plan_remove_keyframe_fields(final_plan_to_save)
            plan_json_path = os.path.join(video_output_folder, "causal_plan.json")
            with open(plan_json_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_plan, f, indent=4, ensure_ascii=False)
            print(f"\n>>> [SUCCESS] Stage 1 plan saved (no keyframe images/paths/indices) to: {plan_json_path}")

            run_summary = {
                "source_video": os.path.basename(PLANNING_CONFIG.VIDEO_PATH),
                "processing_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "models_used": {
                    "planning": PLANNING_CONFIG.MODEL_NAME,
                    "selection": SELECTION_CONFIG.MODEL_NAME
                },
                "config_planning": asdict(PLANNING_CONFIG),
                "config_selection": asdict(SELECTION_CONFIG),
                "stages": ["plan_only", "frame_selection"]
            }
            summary_json_path = os.path.join(video_output_folder, "run_summary.json")
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(run_summary, f, indent=4, ensure_ascii=False)
            print(f">>> [SUCCESS] Run summary saved to: {summary_json_path}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n!!! [FATAL] JSON Parsing Error: {e}")
            error_log_path = os.path.join(PLANNING_CONFIG.OUTPUT_BASE_FOLDER, f"error_response_{video_filename_base}.txt")
            os.makedirs(PLANNING_CONFIG.OUTPUT_BASE_FOLDER, exist_ok=True)
            with open(error_log_path, 'w', encoding='utf-8') as f:
                if response_content:
                    f.write(response_content)
                else:
                    f.write("Response content was empty or None.")
            print(f">>> [INFO] Problematic response saved to: {error_log_path}")
            return
    else:
        # Resume Stage 2 using existing causal_plan.json
        try:
            with open(stage1_path, 'r', encoding='utf-8') as f:
                filtered_plan = json.load(f)
            high_level_goal = filtered_plan.get('high_level_goal', 'No Goal Provided')
            steps_data = filtered_plan.get('steps', [])
            print(f"\n>>> [INFO] Resume mode: Loaded existing Stage 1 plan from: {stage1_path}")
        except Exception as e:
            print(f"\n!!! [FATAL] Failed to load existing plan for resume: {e}")
            return

    # 2) Second call: provide plan JSON + images, QA/improve plan, and align keyframes
    try:
        print(">>> [INFO] Stage 2: QA+improve planning and align keyframes...")
        plan_json_text = json.dumps(filtered_plan, ensure_ascii=False)
        base_prompt = _create_frame_selection_prompt(plan_json_text, len(sampled_frames))
        embed_index = getattr(SELECTION_CONFIG, 'EMBED_INDEX_ON_API_IMAGES', True)
        base_user_content = [{"type": "text", "text": base_prompt}] + build_api_content(sampled_frames, embed_index)
        system_msg = {
            "role": "system",
            "content": "You QA and improve a draft plan grounded on the provided frames and align keyframes. Output strict JSON only."
        }

        selection_client = initialize_api_client(SELECTION_CONFIG)
        if not selection_client:
            return

        max_attempts = int(os.environ.get("STAGE2_MAX_RETRIES", "3"))
        stage2_max_tokens = int(os.environ.get("STAGE2_MAX_TOKENS", "30000"))
        last_content = None
        last_errors: List[str] = []
        stage2_plan = None
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                user_content = base_user_content
            else:
                err_text = "\n".join(f"- {e}" for e in last_errors[:30])
                prev = (last_content or "")
                prev = prev[:12000]  # prevent context blow-up
                repair = (
                    "Your previous output was invalid or failed validation. "
                    "Return a corrected FINAL plan JSON that satisfies ALL requirements, "
                    "preserves ALL schema fields, contains no extra keys, and ensures all frame_index values are valid and temporally consistent.\n"
                    f"Validation errors (fix all):\n{err_text}\n\n"
                    "Previous output (for reference; correct it):\n"
                    f"{prev}\n\n"
                    "Now output ONLY the corrected strict JSON." 
                )
                user_content = [{"type": "text", "text": repair + "\n\n" + base_prompt}] + build_api_content(sampled_frames, embed_index)

            sel_messages = [system_msg, {"role": "user", "content": user_content}]
            print(f">>> [INFO] Sending request (Stage 2 attempt {attempt}/{max_attempts})...")
            sel_start = time.time()
            sel_resp = selection_client.chat.completions.create(
                model=SELECTION_CONFIG.MODEL_NAME,
                messages=sel_messages,
                max_tokens=stage2_max_tokens
            )
            sel_end = time.time()
            if not (sel_resp and sel_resp.choices and len(sel_resp.choices) > 0):
                last_errors = ["Stage 2 response invalid or missing choices."]
                continue
            sel_choice = sel_resp.choices[0]
            if not hasattr(sel_choice, 'message') or not hasattr(sel_choice.message, 'content'):
                last_errors = ["Stage 2 missing message content."]
                continue
            sel_content = sel_choice.message.content
            last_content = sel_content
            if SELECTION_CONFIG.VERBOSE_LOGGING:
                print("\n>>> [DEBUG] Stage 2 raw content:")
                print(sel_content)
            print(f">>> [SUCCESS] Stage 2 API call in {sel_end - sel_start:.2f}s.")

            try:
                sel_json_str = extract_json_from_response(sel_content)
                sel_data = json.loads(sel_json_str)
            except Exception as e:
                last_errors = [f"JSON parse error: {e}"]
                continue

            candidate = sel_data
            ok, validation_errors = _validate_stage2_plan(candidate, len(sampled_frames))
            if not ok:
                last_errors = validation_errors
                msg = " | ".join(validation_errors[:8])
                print(f"!!! [WARNING] Stage 2 validation failed: {msg}")
                continue

            # Stage 2B: alignment verification round (indices only)
            # Default is DISABLED. Enable via env var ENABLE_STAGE2B=1.
            enable_stage2b = os.environ.get("ENABLE_STAGE2B", "0").strip().lower() in {"1", "true", "yes", "y"}
            try:
                verify_rounds = int(os.environ.get("STAGE2_VERIFY_ROUNDS", "1" if enable_stage2b else "0"))
            except Exception:
                verify_rounds = 1 if enable_stage2b else 0
            try:
                verify_max_tokens = int(os.environ.get("STAGE2_VERIFY_MAX_TOKENS", "6000"))
            except Exception:
                verify_max_tokens = 6000

            verified_plan = candidate
            for vr in range(max(0, verify_rounds)):
                align_spec = _build_alignment_spec_from_plan(verified_plan)
                align_spec_json = json.dumps(align_spec, ensure_ascii=False)
                verify_prompt = _create_alignment_verification_prompt(align_spec_json, len(sampled_frames))
                verify_user_content = [{"type": "text", "text": verify_prompt}] + build_api_content(sampled_frames, embed_index)
                verify_messages = [
                    {
                        "role": "system",
                        "content": "You are a strict keyframe alignment verifier. Output strict JSON only."
                    },
                    {"role": "user", "content": verify_user_content}
                ]
                print(f">>> [INFO] Stage 2B: verifying alignment (round {vr+1}/{verify_rounds})...")
                v_start = time.time()
                v_resp = selection_client.chat.completions.create(
                    model=SELECTION_CONFIG.MODEL_NAME,
                    messages=verify_messages,
                    max_tokens=verify_max_tokens
                )
                v_end = time.time()
                if not (v_resp and v_resp.choices and len(v_resp.choices) > 0):
                    last_errors = ["Stage 2B response invalid or missing choices."]
                    verified_plan = None
                    break
                v_choice = v_resp.choices[0]
                if not hasattr(v_choice, 'message') or not hasattr(v_choice.message, 'content'):
                    last_errors = ["Stage 2B missing message content."]
                    verified_plan = None
                    break
                v_content = v_choice.message.content
                if SELECTION_CONFIG.VERBOSE_LOGGING:
                    print("\n>>> [DEBUG] Stage 2B raw content:")
                    print(v_content)
                print(f">>> [SUCCESS] Stage 2B API call in {v_end - v_start:.2f}s.")

                try:
                    v_json_str = extract_json_from_response(v_content)
                    v_data = json.loads(v_json_str)
                except Exception as e:
                    last_errors = [f"Stage 2B JSON parse error: {e}"]
                    verified_plan = None
                    break

                ok_apply, apply_errors = _apply_alignment_updates_in_place(verified_plan, v_data)
                if not ok_apply:
                    last_errors = ["Stage 2B update apply failed"] + apply_errors
                    verified_plan = None
                    break

                ok2, val2 = _validate_stage2_plan(verified_plan, len(sampled_frames))
                if not ok2:
                    last_errors = ["Stage 2B caused validation failure"] + val2
                    verified_plan = None
                    break

            if verified_plan is None:
                msg = " | ".join(last_errors[:8]) if last_errors else "Unknown verification error"
                print(f"!!! [WARNING] Stage 2B verification failed: {msg}")
                continue

            candidate = verified_plan

            stage2_plan = candidate
            break

        if stage2_plan is None:
            msg = " | ".join(last_errors[:12]) if last_errors else "Unknown error"
            print(f"!!! [FATAL] Stage 2 failed after {max_attempts} attempts: {msg}")
            return

        high_level_goal = stage2_plan.get("high_level_goal", high_level_goal)
        steps_data = stage2_plan.get("steps", [])

        # Reconstruct directories and save selected keyframe images
        # using the Stage 2 plan (which may have modified/added steps).
        per_step_annotations: Dict[int, List[CriticalFrameAnnotation]] = {}
        for step_json in steps_data:
            step_id = int(step_json.get('step_id', 0))
            step_goal = step_json.get('step_goal', 'unnamed_step')
            step_folder_name = f"{step_id:02d}_{sanitize_filename(step_goal)}"
            step_output_path = os.path.join(video_output_folder, step_folder_name)
            os.makedirs(step_output_path, exist_ok=True)
            print(f"\n  -> Stage 2 Saving Step {step_id}: '{step_goal}'")

            cf_src_list = step_json.get('critical_frames', [])
            critical_frame_annotations: List[CriticalFrameAnnotation] = []
            for j, frame in enumerate(cf_src_list):
                chosen_idx1 = frame.get("frame_index")
                if not isinstance(chosen_idx1, int):
                    print(f"    !!! [WARNING] Missing/invalid frame_index for critical frame #{j} in step {step_id}. Skipping.")
                    continue
                chosen_idx0 = chosen_idx1 - 1
                if chosen_idx0 < 0 or chosen_idx0 >= len(sampled_frames):
                    print(f"    !!! [WARNING] Selected 1-based index {chosen_idx1} out of range. Skipping.")
                    continue

                critical_frame_annotations.append(
                    CriticalFrameAnnotation(
                        frame_index=chosen_idx1,
                        action_description=frame.get('action_description', ''),
                        spatial_preconditions=[SpatialPrecondition(**sp) for sp in frame.get('spatial_preconditions', [])],
                        affordance_preconditions=[AffordancePrecondition(**ap) for ap in frame.get('affordance_preconditions', [])],
                        causal_chain=CausalChain(**frame.get('causal_chain', {})),
                        affordance_hotspot=AffordanceHotspot(**frame.get('affordance_hotspot', {})),
                        state_change_description=frame.get('state_change_description', '')
                    )
                )

            # Save images for selected frames
            if critical_frame_annotations:
                save_keyframe_images(SELECTION_CONFIG, critical_frame_annotations, step_output_path, sampled_frames)
            per_step_annotations[int(step_id)] = critical_frame_annotations

        print("\n>>> [SUCCESS] Stage 2: Keyframe images saved according to selected indices.")

        # Build and save augmented plan with selected indices and image paths
        processed_planning_steps_2: List[PlanningStep] = []
        for step_json in steps_data:
            step_id = step_json.get('step_id', 0)
            tool_usage_data = step_json.get('tool_and_material_usage')
            tool_usage_obj = ToolAndMaterialUsage(**tool_usage_data) if tool_usage_data else None
            failure_handling_data = step_json.get('failure_handling')
            failure_handling_obj = FailureHandling(**failure_handling_data) if failure_handling_data else None

            # Ensure full field preservation even if model outputs null/missing.
            if tool_usage_obj is None:
                tool_usage_obj = ToolAndMaterialUsage(tools=[], materials=[])
            if failure_handling_obj is None:
                failure_handling_obj = FailureHandling(reason="", recovery_strategy="")

            reconstructed = PlanningStep(
                step_id=step_id,
                step_goal=step_json.get('step_goal', ''),
                rationale=step_json.get('rationale', ''),
                preconditions=step_json.get('preconditions', []),
                expected_effects=step_json.get('expected_effects', []),
                critical_frames=per_step_annotations.get(int(step_id), []),
                tool_and_material_usage=tool_usage_obj,
                causal_challenge_question=step_json.get('causal_challenge_question', ''),
                expected_challenge_outcome=step_json.get('expected_challenge_outcome', ''),
                failure_handling=failure_handling_obj
            )
            processed_planning_steps_2.append(reconstructed)

        final_plan_with_keyframes = {
            "high_level_goal": high_level_goal,
            "steps": [asdict(step) for step in processed_planning_steps_2]
        }
        with open(stage2_path, 'w', encoding='utf-8') as f:
            json.dump(final_plan_with_keyframes, f, indent=4, ensure_ascii=False)
        print(f">>> [SUCCESS] Stage 2 augmented plan saved to: {stage2_path}")

        # Cleanup: remove sampled_frames folder after successful Stage 2 usage
        try:
            import shutil
            if os.path.isdir(sampled_frames_dir):
                shutil.rmtree(sampled_frames_dir)
                print(f">>> [INFO] Removed sampled_frames directory: {sampled_frames_dir}")
        except Exception as e:
            print(f"!!! [WARNING] Unable to remove sampled_frames directory: {e}")

    except Exception as e:
        print(f"\n!!! [FATAL] Error during processing: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n>>> [INFO] Finished processing video: {video_file_path}")

def main():
    """Iterates through all videos in the INPUT_VIDEO_DIRECTORY and processes them sequentially."""
    print(">>> [INFO] Batch Processing Script started.")
    # Allow overriding via environment variables for portability
    input_dir = os.environ.get("INPUT_VIDEO_DIRECTORY", "/e2e-data/embodied-research-data/luzheng/kitchen/long")
    output_base = os.environ.get("OUTPUT_BASE_FOLDER", PLANNING_CONFIG.OUTPUT_BASE_FOLDER)
    PLANNING_CONFIG.OUTPUT_BASE_FOLDER = output_base
    SELECTION_CONFIG.OUTPUT_BASE_FOLDER = output_base
    print(f">>> [INFO] Input Directory: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"!!! [FATAL] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Allowed video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    # Get list of video files
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(video_extensions)]
    
    # Sort files alphabetically to ensure sequential processing "from front to back"
    video_files.sort()

    if not video_files:
        print("!!! [WARNING] No video files found in the directory.")
        sys.exit(0)

    print(f">>> [INFO] Found {len(video_files)} videos to process.")

    for i, filename in enumerate(video_files):
        full_path = os.path.join(input_dir, filename)
        print(f"\n\n##############################################################################")
        print(f"### BATCH PROGRESS: Video {i+1} of {len(video_files)}")
        print(f"### Filename: {filename}")
        print(f"##############################################################################")
        
        try:
            process_single_video(full_path)
        except Exception as e:
            print(f"!!! [ERROR] Unhandled exception while processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            print(">>> [INFO] Continuing to next video...")

    print("\n>>> [INFO] All videos in the folder have been processed.")

if __name__ == "__main__":
    main()
