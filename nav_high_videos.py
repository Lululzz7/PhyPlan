# -*- coding: utf-8 -*-
from openai import OpenAI
import base64
import os
import time
import cv2
import json
import re
from datetime import datetime
from collections import defaultdict 

# ==============================================================================
# --- 1. 初始化与配置 (Initialization and Configuration) ---
# ==============================================================================
print(">>> [INFO] 脚本开始执行：初始化客户端和配置...")

# --- 用户配置区 (User Configuration Area) ---

# 1.1. API凭证和地址
API_KEY="sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s"
API_BASE_URL="http://model.mify.ai.srv/v1"

# 1.2. 指定包含所有视频的根文件夹
input_video_directory = "/e2e-data/embodied-research-data/EmbodiedData/data/ScanNet_v2/videos"

# 1.3. 输出设置
output_base_folder = "generated_plans_output_high_videos" # 存储所有生成计划的主文件夹

# 1.4. 视频帧提取参数
extraction_config = {
    "total_max_frames": 50,
    "resize_dim": None, 
    "jpeg_quality": 100
}

# 1.5. 调试选项
VERBOSE_LOGGING = True 

# --- 配置结束 ---
# 移除费用定价配置

# 初始化OpenAI客户端
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        default_headers={"X-Model-Provider-Id": "vertex_ai"}
    )
    print(">>> [SUCCESS] OpenAI客户端初始化成功。")
except Exception as e:
    print(f"!!! [FATAL] 初始化OpenAI客户端失败: {e}")
    exit()

# ==============================================================================
# --- 2. 核心功能函数 (Core Function Definitions) ---
# ==============================================================================

# 【新增修改】增加 save_output_dir 参数，默认为 None
def process_multiple_videos(video_paths, total_max_frames, resize_dim, jpeg_quality, save_output_dir=None):
    """
    处理一个或多个视频，根据它们的时长按比例动态地提取总共 total_max_frames 数量的帧。
    如果提供了 save_output_dir，会将提取的帧保存为图片文件。
    返回: 一个统一的Base64字符串列表: [base64_frame, ...]
    """
    print(f"\n>>> [INFO] 开始处理 {len(video_paths)} 个视频文件...")
    
    # 【新增修改】如果需要保存图片，先确保目录存在
    if save_output_dir:
        os.makedirs(save_output_dir, exist_ok=True)
        print(f">>> [INFO] 提取的图片将被保存至: {save_output_dir}")

    video_metadata = []
    total_frames_all_videos = 0
    for path in video_paths:
        if not os.path.exists(path):
            print(f"!!! [WARNING] 视频文件未找到，已跳过: {path}")
            continue
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"!!! [WARNING] 无法打开视频文件，已跳过: {path}")
            continue
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            video_metadata.append({'path': path, 'total_frames': frame_count})
            total_frames_all_videos += frame_count
        cap.release()
    
    if not video_metadata:
        print("!!! [ERROR] 在此场景组中未找到任何有效的视频文件。")
        return []

    print(f">>> [INFO] 视频预扫描完成。总帧数: {total_frames_all_videos}")

    frames_to_extract_per_video = []
    remaining_frames = total_max_frames
    for i, meta in enumerate(video_metadata):
        if i < len(video_metadata) - 1:
            proportion = meta['total_frames'] / total_frames_all_videos
            num_frames = round(proportion * total_max_frames)
            num_frames = max(1, num_frames)
            frames_to_extract_per_video.append(num_frames)
            remaining_frames -= num_frames
        else:
            frames_to_extract_per_video.append(max(0, remaining_frames))
    
    print("\n>>> [INFO] 帧分配计划:")
    for meta, num in zip(video_metadata, frames_to_extract_per_video):
        print(f"  - 从 '{os.path.basename(meta['path'])}' (总帧数: {meta['total_frames']}) 提取 {num} 帧")

    all_base64_frames = []
    global_frame_counter = 0 # 【新增修改】用于给保存的图片连续编号

    for meta, num_frames_to_extract in zip(video_metadata, frames_to_extract_per_video):
        if num_frames_to_extract == 0:
            continue
            
        print(f"\n>>> [PROGRESS] 正在从 '{os.path.basename(meta['path'])}' 提取 {num_frames_to_extract} 帧...")
        video = cv2.VideoCapture(meta['path'])
        
        indices = [int(j * meta['total_frames'] / num_frames_to_extract) for j in range(num_frames_to_extract)]

        for frame_index in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video.read()
            if not success:
                continue

            if resize_dim:
                frame = cv2.resize(frame, resize_dim)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            
            # 【新增修改】保存图片逻辑
            if save_output_dir:
                # 格式: scene_frame_001.jpg
                save_filename = f"scene_frame_{global_frame_counter:03d}.jpg"
                save_full_path = os.path.join(save_output_dir, save_filename)
                cv2.imwrite(save_full_path, frame, encode_param)
                global_frame_counter += 1

            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            all_base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        
    print(f"\n>>> [SUCCESS] 所有视频处理完成。总共提取了 {len(all_base64_frames)} 帧图像。")
    return all_base64_frames


def format_plan_to_text(plan_data):
    """
    将解析后的JSON规划数据格式化为人类可读的、纯英文的TXT文件内容。
    """
    lines = []
    lines.append("="*80)
    lines.append(f"Task Plan: {plan_data.get('high_level_goal', 'N/A')}")
    lines.append("="*80)
    lines.append("\n")

    for step in plan_data.get('steps', []):
        lines.append(f"--- Step {step.get('step_id', 'N/A')}: {step.get('step_goal', 'N/A')} ---")
        lines.append(f"\n  [Rationale]")
        lines.append(f"  {step.get('rationale', 'N/A')}\n")
        
        lines.append(f"  [Navigation & Manipulation]")
        lines.append(f"  {step.get('navigation_and_manipulation', 'N/A')}\n")

        lines.append(f"  [Preconditions]")
        for pre in step.get('preconditions', ['N/A']):
            lines.append(f"  - {pre}")
        lines.append("\n")

        lines.append(f"  [Expected Effects]")
        for effect in step.get('expected_effects', ['N/A']):
            lines.append(f"  - {effect}")
        lines.append("\n")

        lines.append(f"  [Causal Challenge]")
        causal_question = step.get('causal_challenge_question')
        expected_outcome = step.get('expected_challenge_outcome')
        lines.append(f"  - Question: {causal_question if isinstance(causal_question, str) and causal_question.strip() else 'N/A'}")
        lines.append(f"  - Expected Outcome: {expected_outcome if isinstance(expected_outcome, str) and expected_outcome.strip() else 'N/A'}")
        lines.append("\n")

        lines.append(f"  [Failure Handling]")
        fh = step.get('failure_handling')
        if isinstance(fh, dict):
            lines.append(f"  - Reason: {fh.get('reason', 'N/A')}")
            lines.append(f"  - Recovery: {fh.get('recovery_strategy', 'N/A')}")
        elif isinstance(fh, list):
            for handling in fh:
                lines.append(f"  - {handling}")
        elif isinstance(fh, str) and fh.strip():
            lines.append(f"  - {fh}")
        else:
            lines.append(f"  - N/A")
        lines.append("\n")
    
    return "\n".join(lines)

# 移除费用计算函数

# ==============================================================================
# --- 3. 系统与用户提示 (System and User Prompts) ---
# ==============================================================================

# --- System Prompt ---
# This prompt establishes the AI's role as a versatile Embodied AI Planner.
system_prompt = """
You are an advanced Embodied AI Planner. Your mission is to act as the core reasoning engine for a versatile household robot whose primary perception modality is a 3D point cloud. Your task is to:
1.  **Perceive and Model the World:** First, explicitly summarize your understanding of the 3D scene. This includes a detailed overall description and a list of key objects relevant for planning.
2.  **Propose a Goal:** Based on your scene analysis, propose a meaningful and complex new plannig goal that is appropriate for the function of the space.
3.  **Generate a Plan:** Generate a detailed, multi-stage, step-by-step plan to achieve this new plannig goal. The plan must be logically sound, physically plausible, and robustly executable by a robot with real-world perception limitations.

Your output MUST be a single, syntactically flawless JSON object. Logical coherence, grounded in your initial perception, is paramount.
"""

# --- User Prompt Template ---
# This is the definitive prompt, refined to produce plans with a more naturalistic and intrinsic planning language.
def create_planning_user_prompt(num_frames):
    return f"""
Analyze the provided {num_frames} video frames which depict a tour of an indoor scene. First, build a mental model of the environment. Then, based on your understanding, propose and generate a detailed plan for a new, actionable task.

The proposed task must be a **long-horizon, complex task** that realistically requires **approximately 5 to 8 distinct logical steps**. Think creatively about the potential uses of the space and its objects to propose a diverse range of tasks.

Crucially, the plan must focus on interactions with large, stable objects easily perceivable in a 3D environment and avoid tasks centered on small, trivial items.

Please note that your plan must be strictly and closely related to the indoor environment described in these video frames! It must be strictly based on this environment and the "key_objects_for_planning" guidelines. Furthermore, it must be comprehensive and reasonable.

Please do not provide your "thinking" responses; instead, give the final, accurate answer.

Your response MUST be a single JSON object adhering to the following strict schema:

{{
  "scene_description": "A detailed, holistic summary of the environment. Describe the rooms shown, their layout, key furniture, and the overall state.",
  "key_objects_for_planning": [
      "A list of objects identified in the scene that are crucial for planning. STRICTLY ENUMERATE ONLY LARGE, PERCEPTION-FRIENDLY OBJECTS."
  ],
  "high_level_goal": "A single, highly descriptive English sentence for the NEWLY PROPOSED long-horizon task, reflecting a plausible use-case for the space.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "A clear and concise description of the sub-goal for this specific step.",
      
      "rationale": "The reasoning behind this step. Why is this action necessary for the new high-level goal?",
      "navigation_and_manipulation": "A paragraph of direct, first-person action statements outlining the execution logic. For navigation, describe the intended path implicitly referencing the environment (e.g., 'Go past the sofa into the kitchen'). Do not explicitly mention using objects as landmarks. For manipulation, state the high-level objective.",
      
      "preconditions": [
          "A list of essential states or conditions that MUST be true before this step can begin."
      ],
      "expected_effects": [
          "A list of states or conditions that WILL become true after this step is successfully completed."
      ],

      "causal_challenge_question": "A specific, insightful 'what-if' question that challenges the core causal or physical understanding of this step.",
      "expected_challenge_outcome": "The predicted outcome for the causal challenge question, explaining the physical consequences.",

      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }},
    ...
  ]
}}

**CRITICAL INSTRUCTIONS:**
-   **Perceive First, Then Plan:** The `scene_description` and `key_objects_for_planning` fields must be completed first. Your subsequent `high_level_goal` and `steps` must be logically consistent with this initial analysis.
-   **POINT CLOUD ROBUSTNESS IS NON-NEGOTIABLE:** Your plan must not rely on seeing or touching small, geometrically insignificant objects.
-   **Promote Task Diversity and Plausibility:** Based on your scene analysis, propose a task that logically fits the primary function of that space. Do not default to simple reorganization if a more functional task is possible.
-   **Strive for Complexity and Depth:** The final plan should have between 5 and 8 logically distinct steps that build upon each other.
-   **Naturalistic Planning Language:** **This is a key instruction.** Within the `navigation_and_manipulation` field, the language must reflect an agent's internal thought process. It should be a direct statement of actions. **AVOID meta-language like 'using X as a landmark'.**
    -   **Correct (Natural):** "Navigate past the dining table and approach the refrigerator."
    -   **Incorrect (Explanatory):** "Navigate to the refrigerator, using the dining table as a landmark."
-   **High-Level Manipulation:** Manipulation tasks must be described as single, high-level actions targeting large objects.
-   **High Requirements For Planning:** Ensure your overall planning is coherent, logical, detailed, and complete, aligning with the behavioral planning of an advanced embodied intelligent robot in an indoor setting. It can incorporate the robot's thought process. Because modeling and perceiving small objects in 3D space is limited, descriptions of operations are better suited to high-level actions rather than fine, delicate manipulations. Navigation descriptions can be more detailed. However, the overall plan and "navigation and manipulation" must be detailed, complete, and logical.
---
**GOLD-STANDARD EXAMPLE:**

**(Imagine the video frames showed a tour of a dining area with a table, chairs, and a vacuum cleaner left in the corner. The table is bare but might be dusty.)**

{{
  "scene_description": "The video shows a well-lit dining area adjacent to a living room space. The central feature is a large wooden dining table, surrounded by four dining chairs. Two chairs are properly tucked in, while the other two are pulled out. In the corner of the dining area, a stand-up vacuum cleaner is present. The table surface is clear of items but appears dusty. The area needs to be prepared to be usable for a meal.",
  "key_objects_for_planning": [
    "dining table", 
    "dining chairs", 
    "vacuum cleaner",
    "kitchen doorway",
    "living room area"
  ],
  "high_level_goal": "Prepare the dining area for a meal by clearing obstructions, cleaning the table, and arranging the chairs.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "Relocate the vacuum cleaner to its storage location.",
      "rationale": "The vacuum cleaner is not part of the dining setup and is a functional obstruction. Removing it first is a necessary clearing step before preparing the area for its intended use.",
      "navigation_and_manipulation": "Go to the corner where the vacuum cleaner is standing. Grasp the vacuum cleaner. Carry it out of the dining area to the designated storage closet and place it inside.",
      "preconditions": [
        "Agent is in the dining area.",
        "The vacuum cleaner is in the corner."
      ],
      "expected_effects": [
        "The dining area is free of non-dining-related obstructions.",
        "The vacuum cleaner is properly stored."
      ],
      "causal_challenge_question": "What if the vacuum cleaner is too heavy or awkward for the robot to lift safely?",
      "expected_challenge_outcome": "The robot may fail to lift or may destabilize. It should switch to a safer strategy such as pushing/rolling it along the floor or using a two-handed stabilized carry if feasible.",
      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }},
    {{
      "step_id": 2,
      "step_goal": "Wipe down the entire surface of the dining table.",
      "rationale": "A clean surface is essential for dining. This step ensures the table is hygienic and ready for place settings.",
      "navigation_and_manipulation": "Approach the dining table. Execute a 'surface wipe' action across the entire tabletop.",
      "preconditions": [
        "The dining area is clear of obstructions (Step 1 complete).",
        "The table surface is bare."
      ],
      "expected_effects": [
        "The dining table surface is clean and free of dust."
      ],
      "causal_challenge_question": "What if the table surface is wet or has low friction, making the wiping motion unstable?",
      "expected_challenge_outcome": "The robot's end-effector may slip and fail to apply consistent contact force, leaving areas unclean. It should reduce speed, increase normal force gradually, and use overlapping passes to ensure coverage.",
      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }},
    {{
      "step_id": 3,
      "step_goal": "Tuck in the two chairs that are already at the table.",
      "rationale": "Arranging the existing chairs properly is the first step in creating an organized and accessible seating arrangement.",
      "navigation_and_manipulation": "Go to the first correctly-placed but pulled-out chair. Push the chair neatly under the table edge. Repeat for the second chair on the same side.",
      "preconditions": [
        "The table is clean (Step 2 complete).",
        "Two chairs are at the table but not tucked in."
      ],
      "expected_effects": [
        "The first two dining chairs are now neatly tucked under the table."
      ],
      "causal_challenge_question": "What if a chair leg catches on the floor or rug while pushing it under the table?",
      "expected_challenge_outcome": "The chair may stop abruptly or rotate, causing misalignment. The robot should adjust the push point, apply force more centrally, and reposition the chair by small corrective moves.",
      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }},
    {{
      "step_id": 4,
      "step_goal": "Move the third chair to an empty place at the table.",
      "rationale": "To complete the seating for four, the remaining chairs must be moved from their scattered positions to the table.",
      "navigation_and_manipulation": "Go to the third dining chair located away from the table. Grasp the chair. Move it to an empty spot on the opposite side of the dining table and tuck it in.",
      "preconditions": [
        "Two chairs are already tucked in (Step 3 complete)."
      ],
      "expected_effects": [
        "Three of the four chairs are now correctly arranged at the table."
      ],
      "causal_challenge_question": "What if the chair is blocked by a narrow passage between the table and another large object?",
      "expected_challenge_outcome": "The chair may collide or get stuck. The robot should choose a wider path, rotate the chair to reduce its footprint, and perform incremental moves to avoid collisions.",
      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }},
    {{
      "step_id": 5,
      "step_goal": "Move the final chair to complete the dining arrangement.",
      "rationale": "Placing the last chair finalizes the seating setup, making the dining area fully prepared and symmetrical.",
      "navigation_and_manipulation": "Go to the final dining chair. Grasp the chair. Move it to the last remaining empty spot at the dining table and tuck it in, ensuring alignment with the other chairs.",
      "preconditions": [
        "Three chairs are arranged at the table (Step 4 complete)."
      ],
      "expected_effects": [
        "All four dining chairs are arranged neatly around the dining table.",
        "The dining area is fully prepared for a meal.",
        "The high-level goal is achieved."
      ],
      "causal_challenge_question": "What if the final chair placement leaves insufficient clearance for a person to sit due to table-chair spacing?",
      "expected_challenge_outcome": "The seating may be unusable even if visually aligned. The robot should back the chair out slightly and re-tuck it with a consistent gap that provides knee clearance.",
      "failure_handling": {{
        "reason": "A description of a likely and plausible failure mode for this step. (e.g., 'The coffee grounds spill over the filter during pouring').",
        "recovery_strategy": "A concise, actionable strategy to mitigate or recover from the described failure. (e.g., 'Discard the spilled grounds, clean the area, and restart the pour with a slower, more controlled motion.')"
      }}
    }}
  ]
}}
Now, based on the frames of the scene tour I have provided, and adhering strictly to all constraints, first output your understanding of the environment, then propose a new, diverse, and long-horizon (5-8 step) goal, and generate the complete JSON output for your new plan.
"""


# ==============================================================================
# --- 4. 【核心修改 + 断点续传】: 主执行流程 (Main Execution Flow) ---
# ==============================================================================

if __name__ == "__main__":
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # 移除总费用统计

    print(f">>> [SETUP] 正在扫描视频文件夹: {input_video_directory}")
    if not os.path.isdir(input_video_directory):
        print(f"!!! [FATAL] 视频文件夹不存在: {input_video_directory}")
        exit()
        
    scene_groups = defaultdict(list)
    for filename in os.listdir(input_video_directory):
        if filename.endswith('.mp4'):
            scene_id = filename.split('_')[0]
            full_path = os.path.join(input_video_directory, filename)
            scene_groups[scene_id].append(full_path)
    
    for scene_id in scene_groups:
        scene_groups[scene_id].sort()

    if not scene_groups:
        print("!!! [FATAL] 在指定文件夹中未找到任何 .mp4 视频文件。")
        exit()

    sorted_scene_ids = sorted(scene_groups.keys())
    print(f">>> [SETUP] 发现 {len(sorted_scene_ids)} 个独立的场景。")

    completed_scenes = set()
    if os.path.exists(output_base_folder):
        for scene_id in sorted_scene_ids:
            scene_output_folder = os.path.join(output_base_folder, scene_id)
            summary_file = os.path.join(scene_output_folder, 'run_summary.json')
            if os.path.exists(summary_file):
                completed_scenes.add(scene_id)
    
    if completed_scenes:
        print(f">>> [RESUME] 检测到 {len(completed_scenes)} 个已完成的场景，将在本次运行中跳过它们。")

    print("="*80)
    print(">>> [BATCH] 准备开始批量处理...")

    # --- 步骤 B: 循环处理每一个场景组 ---
    for scene_id in sorted_scene_ids:
        
        if scene_id in completed_scenes:
            print(f"\n>>> [BATCH SKIP] 跳过已完成的场景: '{scene_id}'")
            print("-"*80)
            continue

        video_paths = scene_groups[scene_id]
        print(f"\n>>> [BATCH] 开始处理场景: '{scene_id}' (包含 {len(video_paths)} 个视频)")
        print(f">>> [BATCH] 视频列表: {[os.path.basename(p) for p in video_paths]}")
        print("-"*80)

        # 【新增修改】提前确定输出文件夹路径，用于保存图片
        video_output_folder = os.path.join(output_base_folder, scene_id)
        images_output_dir = os.path.join(video_output_folder, "sampled_frames") # 将图片保存在子文件夹中

        # 步骤 4.1: 从一个或多个视频中动态提取帧
        # 【新增修改】传入 save_output_dir
        base64_frames = process_multiple_videos(
            video_paths=video_paths,
            total_max_frames=extraction_config["total_max_frames"],
            resize_dim=extraction_config["resize_dim"],
            jpeg_quality=extraction_config["jpeg_quality"],
            save_output_dir=images_output_dir 
        )

        if not base64_frames:
            print(f"!!! [WARNING] 场景 '{scene_id}' 未能提取任何帧，跳过此场景。")
            print("="*80)
            continue

        num_frames = len(base64_frames)

        # 步骤 4.2: 构建API请求
        model_to_use = "gemini-3-pro-preview" 
        print(f"\n>>> [INFO] 正在构建API请求负载... (模型: {model_to_use})")
        
        user_prompt = create_planning_user_prompt(num_frames)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in base64_frames)
                ],
            }
        ]
        
        # 步骤 4.3: 调用API并获取响应
        response_content = None
        usage_info = None
        try:
            print("\n>>> [INFO] 正在发送API请求...")
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
            )
            response_content = response.choices[0].message.content
            usage_info = response.usage
            end_time = time.time()
            print(f">>> [SUCCESS] API调用成功！耗时: {end_time - start_time:.2f} 秒。")
        except Exception as e:
            print(f"!!! [FATAL] API调用错误: {e}")
            exit()

        prompt_tokens = 0
        completion_tokens = 0
        if usage_info:
            prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_info, 'completion_tokens', 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        # 步骤 4.4: 解析、保存和结构化结果
        if response_content:
            # print(response_content) # 可选打印

            try:
                json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_content)
                clean_response = json_match.group(1) if json_match else response_content

                plan_data = json.loads(clean_response)

                steps = plan_data.get('steps', [])
                if isinstance(steps, list):
                    for step in steps:
                        if isinstance(step, dict):
                            step.setdefault('causal_challenge_question', '')
                            step.setdefault('expected_challenge_outcome', '')

                high_level_goal = plan_data.get('high_level_goal', 'No Goal Provided')
                print(f"\n>>> [SUCCESS] 成功解析JSON响应。")
                print(f">>> [GOAL] 目标: {high_level_goal}")

                # 确保主输出文件夹存在（虽然保存图片时可能已经创建了，但加上这行更稳健）
                os.makedirs(video_output_folder, exist_ok=True)

                # 计划文件中记录采样帧文件夹的绝对路径
                plan_data["sample_frames_dir"] = os.path.abspath(images_output_dir)
                plan_json_path = os.path.join(video_output_folder, "plan.json")
                with open(plan_json_path, 'w', encoding='utf-8') as f:
                    json.dump(plan_data, f, indent=4, ensure_ascii=False)
                
                plan_text_content = format_plan_to_text(plan_data)
                plan_text_path = os.path.join(video_output_folder, "new_plan.txt")
                with open(plan_text_path, 'w', encoding='utf-8') as f:
                    f.write(plan_text_content)

                run_summary = {
                    "source_scene_id": scene_id,
                    "source_videos": [os.path.basename(p) for p in video_paths],
                    "processing_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "extraction_parameters": extraction_config,
                    "model_used": model_to_use,
                    "saved_images_directory": images_output_dir, # 【新增修改】记录图片保存位置
                    "api_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                summary_json_path = os.path.join(video_output_folder, "run_summary.json")
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(run_summary, f, indent=4, ensure_ascii=False)
                print(f">>> [SUCCESS] 结果保存至: {video_output_folder}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"\n!!! [ERROR] JSON解析失败: {e}")
                error_log_path = os.path.join(output_base_folder, f"error_response_{scene_id}.txt")
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(response_content)
        else:
            print(f"\n!!! [ERROR] 未收到有效响应。")
        
        print("="*80)

    print("\n>>> [INFO] 所有场景处理完毕。脚本执行结束。")
    # 移除累计总开销打印
