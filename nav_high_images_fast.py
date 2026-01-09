# -*- coding: utf-8 -*-
"""
Parallel runner for nav_high_images.py

Goal: Accelerate generation via multiprocessing without changing any prompts
or extraction parameters. Avoid duplicate work by skipping already completed
scenes.
"""

import os
import re
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import nav_high_images as base
import time
import json


def _should_skip(scene_index: int) -> bool:
    scene_output_folder = os.path.join(base.output_base_folder, str(scene_index))
    summary_file = os.path.join(scene_output_folder, 'run_summary.json')
    return os.path.exists(summary_file)


def _worker(scene_index: int, scene_path: str):
    output_folder_name = str(scene_index)
    scene_output_folder = os.path.join(base.output_base_folder, output_folder_name)
    images_output_dir = os.path.join(scene_output_folder, "sampled_frames")

    # Use original extraction parameters strictly
    cfg = base.extraction_config
    base64_frames = base.process_image_folder(
        scene_path=scene_path,
        total_max_frames=cfg["total_max_frames"],
        resize_dim=cfg["resize_dim"],
        jpeg_quality=cfg["jpeg_quality"],
        save_output_dir=images_output_dir,
    )

    if not base64_frames:
        print(f"!!! [WARNING] 场景 '{os.path.basename(scene_path)}' 未能提取任何帧，跳过此场景。")
        return

    num_frames = len(base64_frames)
    # Keep consistent with nav_high_images.py unless explicitly overridden.
    model_to_use = os.environ.get("MODEL_TO_USE", "gemini-3-pro-preview")
    user_prompt = base.create_planning_user_prompt(num_frames)
    messages = [
        {"role": "system", "content": base.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in base64_frames)
            ],
        }
    ]

    response_content = None
    usage_info = None
    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    base_delay = float(os.environ.get("RETRY_DELAY_SEC", "3"))
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = base.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
            )
            response_content = getattr(response.choices[0].message, 'content', None)
            usage_info = getattr(response, 'usage', None)
            end_time = time.time()
            print(f">>> [SUCCESS] 场景 {scene_index} API调用成功！耗时: {end_time - start_time:.2f} 秒。")
        except Exception as e:
            print(f"!!! [FATAL] 场景 {scene_index} API调用错误(第{attempt}次): {e}")
            response_content = None

        # Validate response_content by attempting JSON extraction + basic keys
        valid = False
        if response_content:
            try:
                json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_content)
                clean_response = json_match.group(1) if json_match else response_content
                tmp = json.loads(clean_response)
                valid = (tmp.get('high_level_goal') is not None and isinstance(tmp.get('steps'), list))
            except Exception as e:
                print(f"!!! [WARN] 场景 {scene_index} JSON校验失败(第{attempt}次): {e}")
                valid = False
        if valid:
            break
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"!!! [RETRY] 场景 {scene_index} 输出为空或不可用，{delay:.1f}s 后重试...")
            time.sleep(delay)
    if not response_content:
        print(f"!!! [FATAL] 场景 {scene_index} API调用多次失败，跳过。")
        return

    # 移除费用计算：仅记录tokens
    prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
    completion_tokens = getattr(usage_info, 'completion_tokens', 0)

    if response_content:
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
            print(f">>> [GOAL] 场景 {scene_index} 目标: {high_level_goal}")

            os.makedirs(scene_output_folder, exist_ok=True)

            # 在计划JSON中记录采样帧文件夹的绝对路径
            plan_data["sample_frames_dir"] = os.path.abspath(images_output_dir)
            plan_json_path = os.path.join(scene_output_folder, "plan.json")
            with open(plan_json_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=4, ensure_ascii=False)

            plan_text_content = base.format_plan_to_text(plan_data)
            plan_text_path = os.path.join(scene_output_folder, "new_plan.txt")
            with open(plan_text_path, 'w', encoding='utf-8') as f:
                f.write(plan_text_content)

            run_summary = {
                "source_scene_id_original": os.path.basename(scene_path),
                "processed_as_index": scene_index,
                "source_directory": scene_path,
                "processing_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "extraction_parameters": base.extraction_config,
                "model_used": model_to_use,
                "saved_images_directory": images_output_dir,
                "api_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            summary_json_path = os.path.join(scene_output_folder, "run_summary.json")
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(run_summary, f, indent=4, ensure_ascii=False)
            print(f">>> [SUCCESS] 场景 {scene_index} 结果已保存至: {scene_output_folder}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n!!! [ERROR] 场景 {scene_index} JSON解析失败: {e}")
            error_log_path = os.path.join(base.output_base_folder, f"error_response_{scene_index}.txt")
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(response_content or "")
    else:
        print(f"\n!!! [ERROR] 场景 {scene_index} 未收到有效响应。")


def main():
    print(f">>> [SETUP] 正在扫描场景根目录: {base.input_base_directory}")
    if not os.path.isdir(base.input_base_directory):
        print(f"!!! [FATAL] 场景根目录不存在: {base.input_base_directory}")
        return

    try:
        scene_paths = sorted([
            os.path.join(base.input_base_directory, d)
            for d in os.listdir(base.input_base_directory)
            if os.path.isdir(os.path.join(base.input_base_directory, d))
        ])
    except Exception as e:
        print(f"!!! [FATAL] 扫描场景文件夹时出错: {e}")
        return

    if not scene_paths:
        print(f"!!! [FATAL] 在 {base.input_base_directory} 中未找到任何场景子文件夹。")
        return

    completed_scenes = {i for i in range(len(scene_paths)) if _should_skip(i)}
    to_schedule = [(i, p) for i, p in enumerate(scene_paths) if i not in completed_scenes]

    if not to_schedule:
        print(">>> [INFO] 所有场景已处理，无需重复生成。")
        return

    max_workers = int(os.environ.get("CONCURRENCY", max(os.cpu_count() or 1, 2)))
    print(f">>> [INFO] 并行处理进程数: {max_workers}，待处理场景数: {len(to_schedule)}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, idx, path): (idx, path) for idx, path in to_schedule}
        for fut in as_completed(futures):
            idx, path = futures[fut]
            try:
                fut.result()
                print(f">>> [DONE] 场景 {idx}: {os.path.basename(path)}")
            except Exception as e:
                print(f"!!! [ERROR] 场景 {idx} 失败: {e}")


if __name__ == "__main__":
    main()
