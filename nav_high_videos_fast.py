# -*- coding: utf-8 -*-
"""
Parallel runner for nav_high_videos.py

Goal: Accelerate generation via multiprocessing without changing any prompts
or extraction parameters. Avoid duplicate work by skipping already completed
scenes.
"""

import os
import re
import json
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nav_high_videos as base
import time
import json


def _should_skip(scene_id: str) -> bool:
    scene_output_folder = os.path.join(base.output_base_folder, scene_id)
    summary_file = os.path.join(scene_output_folder, 'run_summary.json')
    return os.path.exists(summary_file)


def _worker(scene_id: str, video_paths):
    video_output_folder = os.path.join(base.output_base_folder, scene_id)
    images_output_dir = os.path.join(video_output_folder, "sampled_frames")

    cfg = base.extraction_config
    base64_frames = base.process_multiple_videos(
        video_paths=video_paths,
        total_max_frames=cfg["total_max_frames"],
        resize_dim=cfg["resize_dim"],
        jpeg_quality=cfg["jpeg_quality"],
        save_output_dir=images_output_dir,
    )

    if not base64_frames:
        print(f"!!! [WARNING] 场景 '{scene_id}' 未能提取任何帧，跳过此场景。")
        return

    num_frames = len(base64_frames)
    # Keep consistent with nav_high_videos.py unless explicitly overridden.
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
            print(f">>> [SUCCESS] 场景 {scene_id} API调用成功！耗时: {end_time - start_time:.2f} 秒。")
        except Exception as e:
            print(f"!!! [FATAL] 场景 {scene_id} API调用错误(第{attempt}次): {e}")
            response_content = None

        valid = False
        if response_content:
            try:
                json_match = re.search(r'```json\s*([\s\S]+?)\s*```', response_content)
                clean_response = json_match.group(1) if json_match else response_content
                tmp = json.loads(clean_response)
                valid = (tmp.get('high_level_goal') is not None and isinstance(tmp.get('steps'), list))
            except Exception as e:
                print(f"!!! [WARN] 场景 {scene_id} JSON校验失败(第{attempt}次): {e}")
                valid = False
        if valid:
            break
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"!!! [RETRY] 场景 {scene_id} 输出为空或不可用，{delay:.1f}s 后重试...")
            time.sleep(delay)
    if not response_content:
        print(f"!!! [FATAL] 场景 {scene_id} API调用多次失败，跳过。")
        return

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
            print(f">>> [GOAL] 场景 {scene_id} 目标: {high_level_goal}")

            os.makedirs(video_output_folder, exist_ok=True)

            # 在计划JSON中记录采样帧文件夹的绝对路径
            plan_data["sample_frames_dir"] = os.path.abspath(images_output_dir)
            plan_json_path = os.path.join(video_output_folder, "plan.json")
            with open(plan_json_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=4, ensure_ascii=False)

            plan_text_content = base.format_plan_to_text(plan_data)
            plan_text_path = os.path.join(video_output_folder, "new_plan.txt")
            with open(plan_text_path, 'w', encoding='utf-8') as f:
                f.write(plan_text_content)

            # 移除费用计算：仅记录tokens
            prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_info, 'completion_tokens', 0)
            run_summary = {
                "source_scene_id": scene_id,
                "source_videos": [os.path.basename(p) for p in video_paths],
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
            summary_json_path = os.path.join(video_output_folder, "run_summary.json")
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(run_summary, f, indent=4, ensure_ascii=False)
            print(f">>> [SUCCESS] 场景 {scene_id} 结果保存至: {video_output_folder}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n!!! [ERROR] 场景 {scene_id} JSON解析失败: {e}")
            error_log_path = os.path.join(base.output_base_folder, f"error_response_{scene_id}.txt")
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(response_content or "")
    else:
        print(f"\n!!! [ERROR] 场景 {scene_id} 未收到有效响应。")


def main():
    print(f">>> [SETUP] 正在扫描视频文件夹: {base.input_video_directory}")
    if not os.path.isdir(base.input_video_directory):
        print(f"!!! [FATAL] 视频文件夹不存在: {base.input_video_directory}")
        return

    scene_groups = defaultdict(list)
    for filename in os.listdir(base.input_video_directory):
        if filename.endswith('.mp4'):
            scene_id = filename.split('_')[0]
            full_path = os.path.join(base.input_video_directory, filename)
            scene_groups[scene_id].append(full_path)

    for scene_id in scene_groups:
        scene_groups[scene_id].sort()

    if not scene_groups:
        print("!!! [FATAL] 在指定文件夹中未找到任何 .mp4 视频文件。")
        return

    sorted_scene_ids = sorted(scene_groups.keys())
    to_schedule = [(sid, scene_groups[sid]) for sid in sorted_scene_ids if not _should_skip(sid)]

    if not to_schedule:
        print(">>> [INFO] 所有场景已处理，无需重复生成。")
        return

    max_workers = int(os.environ.get("CONCURRENCY", max(os.cpu_count() or 1, 2)))
    print(f">>> [INFO] 并行处理进程数: {max_workers}，待处理场景数: {len(to_schedule)}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, sid, paths): (sid, paths) for sid, paths in to_schedule}
        for fut in as_completed(futures):
            sid, paths = futures[fut]
            try:
                fut.result()
                print(f">>> [DONE] 场景 {sid}")
            except Exception as e:
                print(f"!!! [ERROR] 场景 {sid} 失败: {e}")


if __name__ == "__main__":
    main()
