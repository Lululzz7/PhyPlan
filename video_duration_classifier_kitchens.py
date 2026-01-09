"""Video duration filtering and categorization utility with File-System based Resume.

This script scans videos from a provided source directory.
It supports two layouts:
  1) Flat folder containing videos directly.
  2) EPIC-KITCHENS layout: <root>/Pxx/videos/*.MP4 (and similar extensions).

The script checks if the video already exists in the output directories to support resume.
"""

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Set

import cv2


@dataclass
class ScriptConfig:
    """Centralized configuration for video processing."""

    # 支持 EPIC-KITCHENS 根目录（包含 Pxx 子目录，每个子目录下有 videos 子文件夹）
    # 仍支持传入一个直接包含视频文件的扁平目录。
    source_folder: str = "/e2e-data/embodied-research-data/HumanData/EPIC-KITCHENS"
    output_directory: str = "/e2e-data/embodied-research-data/luzheng/kitchen"
    short_folder: str = "short"
    medium_folder: str = "medium"
    long_folder: str = "long"
    # 新增：将不符合要求的视频也归档，以免下次重复扫描计算
    discard_folder: str = "discard" 
    
    supported_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
    min_duration_seconds: float = 8.0
    short_upper_seconds: float = 20.0
    medium_upper_seconds: float = 90.0
    max_duration_seconds: float = 180.0
    reencode_fourcc: str = "mp4v"


CONFIG = ScriptConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter, trim, and categorize videos by duration."
    )
    parser.add_argument(
        "source_folder",
        nargs="?",
        default=CONFIG.source_folder,
        help="Absolute path to directory containing input videos.",
    )
    parser.add_argument(
        "--output-directory",
        dest="output_directory",
        default=CONFIG.output_directory,
        help="Directory where categorized folders will be created.",
    )
    # 添加覆盖选项，如果想强制重新处理已存在的文件
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directories.",
    )
    return parser.parse_args()


def ensure_output_directories(config: ScriptConfig) -> Dict[str, str]:
    base_dir = os.path.abspath(config.output_directory)
    destinations = {
        "short": os.path.join(base_dir, config.short_folder),
        "medium": os.path.join(base_dir, config.medium_folder),
        "long": os.path.join(base_dir, config.long_folder),
        "discard": os.path.join(base_dir, config.discard_folder), # 建立丢弃文件夹
    }
    for _, path in destinations.items():
        os.makedirs(path, exist_ok=True)
    return destinations


def scan_existing_outputs(destinations: Dict[str, str]) -> Set[str]:
    """
    扫描所有输出文件夹，收集已处理过的文件名。
    这是实现“断点续传”的核心，无需txt文件，直接看结果。
    """
    existing_files = set()
    print("[INFO] 正在扫描已输出的文件以支持断点续传...")
    
    for category, folder_path in destinations.items():
        if not os.path.exists(folder_path):
            continue
            
        # 获取文件夹内所有文件名
        files = os.listdir(folder_path)
        count = 0
        for f in files:
            # 存入集合，用于快速查找
            existing_files.add(f)
            # 如果是分段文件（如 name_part1.mp4），同时将原始文件名也标记为已处理
            # 这样可以避免同一源视频被重复处理。
            try:
                stem, ext = os.path.splitext(f)
                if "_part" in stem:
                    base_stem = stem.split("_part")[0]
                    original_name = base_stem + ext
                    existing_files.add(original_name)
            except Exception:
                pass
            count += 1
        
        if count > 0:
            print(f"  - 在 '{category}' 中发现 {count} 个文件")
            
    print(f"[INFO] 共发现 {len(existing_files)} 个已处理文件，将自动跳过。")
    return existing_files


def collect_video_paths(config: ScriptConfig) -> Iterable[str]:
    """收集所有待处理的视频路径。

    兼容两种目录结构：
      1) 扁平结构：source_folder 直接包含视频文件。
      2) EPIC-KITCHENS 结构：source_folder 下存在若干子文件夹（如 P01、P02、...），每个子文件夹内的 `videos/` 子目录包含视频文件。
    """
    root = config.source_folder
    if not os.path.exists(root):
        return

    # 1) 先扫描扁平结构（source_folder 顶层的文件）
    direct_files: Iterable[str] = []
    try:
        entries = sorted(os.listdir(root))
    except Exception:
        entries = []

    for name in entries:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            extension = os.path.splitext(candidate)[1].lower()
            if extension in config.supported_extensions:
                yield candidate

    # 2) 再扫描 EPIC-KITCHENS 结构：<root>/<participant>/videos/*
    for name in entries:
        participant_dir = os.path.join(root, name)
        if not os.path.isdir(participant_dir):
            continue
        videos_dir = os.path.join(participant_dir, "videos")
        if not os.path.isdir(videos_dir):
            continue
        try:
            video_entries = sorted(os.listdir(videos_dir))
        except Exception:
            video_entries = []
        for vname in video_entries:
            candidate = os.path.join(videos_dir, vname)
            if not os.path.isfile(candidate):
                continue
            extension = os.path.splitext(candidate)[1].lower()
            if extension in config.supported_extensions:
                yield candidate


def compute_video_duration(video_path: str) -> Tuple[float, float]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        # 无法打开的文件，通常建议跳过
        return 0.0, 0.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0:
        return 0.0, 0.0

    duration = frame_count / fps
    return duration, fps


def categorize_duration(duration: float, config: ScriptConfig) -> str:
    if duration < config.min_duration_seconds:
        return "discard"
    if duration <= config.short_upper_seconds:
        return "short"
    if duration <= config.medium_upper_seconds:
        return "medium"
    if duration <= config.max_duration_seconds:
        return "long"
    return "long"


def trim_video(
    input_path: str,
    output_path: str,
    fps: float,
    max_duration: float,
    config: ScriptConfig,
) -> float:
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        return 0.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = int(max_duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*config.reencode_fourcc)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_written = 0
    while frames_written < frame_limit:
        success, frame = capture.read()
        if not success:
            break
        writer.write(frame)
        frames_written += 1

    capture.release()
    writer.release()

    if frames_written == 0:
        return 0.0

    return frames_written / fps


def process_video(
    video_path: str,
    destinations: Dict[str, str],
    config: ScriptConfig,
) -> Tuple[str, float]:
    
    duration, fps = compute_video_duration(video_path)
    filename = os.path.basename(video_path)
    
    if duration <= 0 or fps <= 0:
        print(f"[ERROR] 无法读取: {filename}")
        return "error", 0.0

    # 判断分类
    action = categorize_duration(duration, config) # 可能是 'discard', 'short', 'medium', 'long'

    # 针对 'discard' 也要进行处理（移动到 discard 文件夹），
    # 这样下次运行 scan_existing_outputs 时就能发现它，从而跳过计算
    if action == "discard":
        output_path = os.path.join(destinations["discard"], filename)
        print(f"[INFO] 归类为丢弃 (<{config.min_duration_seconds}s): {filename}")
        # 对于丢弃的文件，通常不需要剪辑，直接移动或复制占位即可
        # 这里选择复制，保留源文件不动
        shutil.copy2(video_path, output_path)
        return "discard", duration

    # 处理正常分类
    needs_segment = duration > config.max_duration_seconds
    effective_duration = min(duration, config.max_duration_seconds)
    
    # 注意：分类逻辑保持不变；如果被剪辑/分段，它依然属于 long
    category = categorize_duration(effective_duration, config)
    long_dir = destinations[category]

    print(f"[INFO] 处理中 ({category}): {filename} [{duration:.2f}s]")

    # 对于 long 且时长超过 max_duration_seconds 的视频，改为每隔 180s 截取并保存一个分段视频。
    # 最后不足 180s 的部分丢弃。其他保持不变。
    if needs_segment and category == "long":
        # 读取基本信息
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"[ERROR] 无法打开用于分段: {filename}")
            return "error", 0.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_limit = int(config.max_duration_seconds * fps)

        # 可保存的完整分段数量（丢弃不足 180s 的最后一段）
        full_segments = frame_count // frame_limit
        if full_segments <= 0:
            # 理论上不会进入此分支，因为 needs_segment 已保证时长>max。
            capture.release()
            return category, effective_duration

        stem, ext = os.path.splitext(filename)
        fourcc = cv2.VideoWriter_fourcc(*config.reencode_fourcc)

        current_segment_idx = 1
        frames_written_in_segment = 0
        frames_written_total_target = full_segments * frame_limit
        frames_processed = 0

        # 初始化第一个分段写入器
        part_name = f"{stem}_part{current_segment_idx}{ext}"
        output_path = os.path.join(long_dir, part_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while frames_processed < frames_written_total_target:
            success, frame = capture.read()
            if not success:
                # 读帧失败：如果当前分段未满，删除该分段
                try:
                    writer.release()
                except Exception:
                    pass
                if frames_written_in_segment < frame_limit:
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except Exception:
                        pass
                break

            writer.write(frame)
            frames_written_in_segment += 1
            frames_processed += 1

            if frames_written_in_segment >= frame_limit:
                # 完成一个分段
                writer.release()
                current_segment_idx += 1
                frames_written_in_segment = 0
                if current_segment_idx > full_segments:
                    # 已达到目标分段数
                    break
                # 打开下一个分段写入器
                part_name = f"{stem}_part{current_segment_idx}{ext}"
                output_path = os.path.join(long_dir, part_name)
                if os.path.exists(output_path):
                    os.remove(output_path)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        capture.release()
        try:
            writer.release()
        except Exception:
            pass

        return category, effective_duration

    # 不需要分段，直接复制到分类目录（保持原行为）
    output_path = os.path.join(long_dir, filename)
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.copy2(video_path, output_path)
    return category, duration


def main() -> None:
    args = parse_args()
    # 默认提高 OpenCV FFmpeg 的读取尝试上限，以提升多流视频读取的稳定性
    os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "100000")

    config = ScriptConfig(
        source_folder=os.path.abspath(args.source_folder),
        output_directory=os.path.abspath(args.output_directory),
        # 继承其他配置...
    )

    if not os.path.isdir(config.source_folder):
        print(f"[ERROR] 输入文件夹不存在: {config.source_folder}")
        return

    # 1. 确保输出目录存在
    destinations = ensure_output_directories(config)
    
    # 2. 【关键步骤】扫描已存在的输出文件
    # 如果不开启 overwrite，则加载已存在的文件列表
    processed_files = set()
    if not args.overwrite:
        processed_files = scan_existing_outputs(destinations)
    else:
        print("[INFO] 开启覆盖模式，将重新处理所有文件。")

    counters = {"short": 0, "medium": 0, "long": 0, "discard": 0, "error": 0, "skipped": 0}

    # 3. 遍历源文件
    for video_path in collect_video_paths(config):
        filename = os.path.basename(video_path)
        
        # 【关键步骤】检查是否存在
        if filename in processed_files:
            # 为了避免刷屏，可以注释掉下面这行
            # print(f"[SKIP] 跳过已存在文件: {filename}")
            counters["skipped"] += 1
            continue

        # 开始处理
        category, _ = process_video(video_path, destinations, config)
        if category in counters:
            counters[category] += 1

    print("\n[INFO] 全部完成: ")
    for key, value in counters.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
