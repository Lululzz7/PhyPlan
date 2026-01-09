# -*- coding: utf-8 -*-
"""Parallel runner for mani_longvideo.py.

目标：只做并行调度，不引入与 `mani_longvideo.py` 不一致的逻辑。

- 抽帧/提示词/Stage1+Stage2/校验与重试/断点续跑逻辑：完全复用 `mani_longvideo.py`。
- 本脚本仅负责：扫描目录、跳过已完成样本、并行调用 `process_single_video()`。

可用环境变量（与 `mani_longvideo.py` 对齐）：
- INPUT_VIDEO_DIRECTORY
- OUTPUT_BASE_FOLDER

并行相关（仅本脚本使用）：
- CONCURRENCY
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import mani_longvideo as base


def _should_skip(video_path: str, output_base: str) -> bool:
    name, _ = os.path.splitext(os.path.basename(video_path))
    video_output_folder = os.path.join(output_base, name)
    stage2_path = os.path.join(video_output_folder, "causal_plan_with_keyframes.json")
    return os.path.exists(stage2_path)


def _worker(video_path: str, output_base: str):
    # 与 base.main() 一致：将输出根目录写回 config
    base.PLANNING_CONFIG.OUTPUT_BASE_FOLDER = output_base
    base.SELECTION_CONFIG.OUTPUT_BASE_FOLDER = output_base

    # 关键：只调用一次，所有“重试/校验/续跑”都由 base.process_single_video 负责。
    base.process_single_video(video_path)


def main():
    input_dir = os.environ.get("INPUT_VIDEO_DIRECTORY", "/e2e-data/embodied-research-data/luzheng/kitchen/long")
    output_base = os.environ.get("OUTPUT_BASE_FOLDER", base.PLANNING_CONFIG.OUTPUT_BASE_FOLDER)

    # 与 base.main() 一致：将输出根目录写回 config
    base.PLANNING_CONFIG.OUTPUT_BASE_FOLDER = output_base
    base.SELECTION_CONFIG.OUTPUT_BASE_FOLDER = output_base

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
