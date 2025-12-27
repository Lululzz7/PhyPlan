# -*- coding: utf-8 -*-
"""
Parallel runner for mani_mediumvideo.py

Goal: Accelerate generation via multiprocessing without changing any prompts
or configuration values. Avoid duplicate work by skipping already completed
outputs.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

import mani_mediumvideo as base


def _should_skip(video_path: str, output_base: str) -> bool:
    name, _ = os.path.splitext(os.path.basename(video_path))
    video_output_folder = os.path.join(output_base, name)
    stage2_path = os.path.join(video_output_folder, "causal_plan_with_keyframes.json")
    return os.path.exists(stage2_path)


def _worker(video_path: str, output_base: str):
    base.PLANNING_CONFIG.OUTPUT_BASE_FOLDER = output_base
    base.SELECTION_CONFIG.OUTPUT_BASE_FOLDER = output_base

    name, _ = os.path.splitext(os.path.basename(video_path))
    video_output_folder = os.path.join(output_base, name)
    stage2_path = os.path.join(video_output_folder, "causal_plan_with_keyframes.json")

    def _valid_stage2(path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not (isinstance(data.get('steps'), list) and data.get('high_level_goal') is not None):
                return False
            for step in data.get('steps', []):
                if not isinstance(step, dict):
                    return False
                required = [
                    'step_id',
                    'step_goal',
                    'rationale',
                    'preconditions',
                    'expected_effects',
                    'tool_and_material_usage',
                    'causal_challenge_question',
                    'expected_challenge_outcome',
                    'failure_handling',
                    'critical_frames',
                ]
                if any(k not in step for k in required):
                    return False
            return True
        except Exception:
            return False

    max_retries = int(os.environ.get("MAX_RETRIES", "3"))
    base_delay = float(os.environ.get("RETRY_DELAY_SEC", "3"))

    for attempt in range(1, max_retries + 1):
        print(f">>> [RUN] {os.path.basename(video_path)} attempt {attempt}/{max_retries}")
        base.process_single_video(video_path)
        if _valid_stage2(stage2_path):
            print(f">>> [OK] {os.path.basename(video_path)} completed.")
            return
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"!!! [RETRY] Stage 2 invalid or missing. Sleeping {delay:.1f}s then retry...")
            time.sleep(delay)
    print(f"!!! [FAIL] {os.path.basename(video_path)} failed after {max_retries} attempts.")


def main():
    input_dir = os.environ.get("INPUT_VIDEO_DIRECTORY", "/e2e-data/embodied-research-data/luzheng/medium")
    output_base = os.environ.get("OUTPUT_BASE_FOLDER", base.PLANNING_CONFIG.OUTPUT_BASE_FOLDER)

    if not os.path.exists(input_dir):
        print(f"!!! [FATAL] Input directory does not exist: {input_dir}")
        return

    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    try:
        video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]
    except Exception as e:
        print(f"!!! [FATAL] Failed to list directory '{input_dir}': {e}")
        return
    video_files.sort()

    full_paths = [os.path.join(input_dir, f) for f in video_files]
    to_run = [p for p in full_paths if not _should_skip(p, output_base)]

    if not to_run:
        print(">>> [INFO] Nothing to process. All videos already completed.")
        return

    max_workers = int(os.environ.get("CONCURRENCY", max(os.cpu_count() or 1, 2)))
    print(f">>> [INFO] Parallel processing with {max_workers} workers. Items: {len(to_run)}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, p, output_base): p for p in to_run}
        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                fut.result()
                print(f">>> [DONE] {os.path.basename(vid)}")
            except Exception as e:
                print(f"!!! [ERROR] {os.path.basename(vid)} failed: {e}")


if __name__ == "__main__":
    main()
