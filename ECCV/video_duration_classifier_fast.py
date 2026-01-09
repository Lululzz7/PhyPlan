# -*- coding: utf-8 -*-
"""
Parallel runner for video_duration_classifier.py

Goal: Accelerate processing via multiprocessing without changing any
classification parameters or behavior. Adds duplicate-avoidance and
resume safety with lock files, while preserving original CLI options.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import video_duration_classifier as base


def _lock_dir(output_base: str) -> str:
    path = os.path.join(os.path.abspath(output_base), "__locks__")
    os.makedirs(path, exist_ok=True)
    return path


def _acquire_lock(lock_root: str, filename: str) -> Tuple[bool, str]:
    lock_path = os.path.join(lock_root, f"{filename}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True, lock_path
    except FileExistsError:
        return False, lock_path


def _release_lock(lock_path: str):
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def _worker(video_path: str, destinations: Dict[str, str], config: base.ScriptConfig, overwrite: bool, lock_root: str):
    filename = os.path.basename(video_path)

    # Skip if already processed and not overwriting
    if not overwrite:
        existing = set()
        for d in destinations.values():
            try:
                if os.path.exists(os.path.join(d, filename)):
                    existing.add(d)
            except Exception:
                pass
        if existing:
            return (filename, "skipped", 0.0)

    ok, lock_path = _acquire_lock(lock_root, filename)
    if not ok:
        # Someone else is processing this file
        return (filename, "locked_skip", 0.0)

    try:
        category, duration = base.process_video(video_path, destinations, config)
        return (filename, category, duration)
    finally:
        _release_lock(lock_path)


def main():
    args = base.parse_args()
    config = base.ScriptConfig(
        source_folder=os.path.abspath(args.source_folder),
        output_directory=os.path.abspath(args.output_directory),
    )

    if not os.path.isdir(config.source_folder):
        print(f"[ERROR] 输入文件夹不存在: {config.source_folder}")
        return

    destinations = base.ensure_output_directories(config)

    processed_files = set()
    if not args.overwrite:
        processed_files = base.scan_existing_outputs(destinations)
    else:
        print("[INFO] 开启覆盖模式，将重新处理所有文件。")

    # Collect candidates
    candidates = []
    for video_path in base.collect_video_paths(config):
        filename = os.path.basename(video_path)
        if filename in processed_files and not args.overwrite:
            continue
        candidates.append(video_path)

    if not candidates:
        print("[INFO] 无需处理：已全部完成或无有效视频。")
        return

    lock_root = _lock_dir(config.output_directory)
    max_workers = int(os.environ.get("CONCURRENCY", max(os.cpu_count() or 1, 2)))
    print(f"[INFO] 并行处理进程数: {max_workers}，待处理文件数: {len(candidates)}")

    counters = {"short": 0, "medium": 0, "long": 0, "discard": 0, "error": 0, "skipped": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_worker, path, destinations, config, args.overwrite, lock_root): path
            for path in candidates
        }
        for fut in as_completed(futures):
            path = futures[fut]
            fname = os.path.basename(path)
            try:
                filename, category, duration = fut.result()
                if category in counters:
                    counters[category] += 1
                elif category == "locked_skip":
                    counters["skipped"] += 1
                print(f"[DONE] {filename}: {category} ({duration:.2f}s)")
            except Exception as e:
                counters["error"] += 1
                print(f"[ERROR] {fname} 失败: {e}")

    print("\n[INFO] 全部完成: ")
    for key, value in counters.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

