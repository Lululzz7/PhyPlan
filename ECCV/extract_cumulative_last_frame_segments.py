# -*- coding: utf-8 -*-
"""
为每个 step 保存“从视频开头到该 step 尾帧（最后一张关键帧）”的完整片段。

使用示例：
    python extract_cumulative_last_frame_segments.py --video-output-dir <单个视频的输出目录>

说明：
- 仅消费既有产物（causal_plan_with_keyframes.json、run_summary.json、关键帧图片），不修改原生成脚本。
- 默认输出目录：<video-output-dir>/cumulative_last_frame_segments
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract segments from video start to each step's last keyframe (尾帧)."
    )
    parser.add_argument("--video-output-dir", required=True, help="包含 causal_plan_with_keyframes.json 的单视频输出目录")
    parser.add_argument("--source-video", help="显式指定源视频路径，覆盖 run_summary.json 解析")
    parser.add_argument(
        "--output-dir",
        help="片段输出目录，默认 <video-output-dir>/cumulative_last_frame_segments",
    )
    parser.add_argument("--overwrite", action="store_true", help="若目标文件已存在则覆盖")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg 可执行文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    return text


def parse_timestamp_from_filename(path: str) -> Optional[float]:
    name = os.path.basename(path)
    m = re.search(r"_ts_([0-9]+(?:\.[0-9]+)?)s\.jpg$", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _resolve_to_existing(path: str, base_dir: str) -> Optional[str]:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(base_dir, path)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return None


def resolve_source_video(video_dir: str, override: Optional[str]) -> str:
    if override:
        resolved = _resolve_to_existing(override, video_dir)
        if resolved:
            return resolved
        raise FileNotFoundError(f"--source-video 指定路径不存在: {override}")

    summary_path = os.path.join(video_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("未找到 run_summary.json，无法推断源视频路径，请使用 --source-video")
    data = load_json(summary_path)

    tried: List[str] = []

    config = data.get("config_planning", {})
    cand = config.get("VIDEO_PATH")
    if cand:
        tried.append(cand)
        resolved = _resolve_to_existing(cand, video_dir)
        if resolved:
            return resolved

    cand = data.get("source_video")
    if cand:
        tried.append(cand)
        resolved = _resolve_to_existing(cand, video_dir)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "未能定位源视频。已尝试: " + ", ".join(tried) + "。请使用 --source-video 指定绝对路径。"
    )


def find_keyframe_path(
    video_dir: str, step_id: int, step_goal: str, frame_index: int, existing_path: Optional[str]
) -> Optional[str]:
    if existing_path and os.path.exists(existing_path):
        return existing_path
    step_folder = f"{step_id:02d}_{sanitize_filename(step_goal)}"
    pattern = os.path.join(video_dir, step_folder, f"frame_{frame_index:03d}_ts_*s.jpg")
    candidates = sorted(glob(pattern))
    if candidates:
        return candidates[0]
    return None


def collect_last_keyframes(plan_path: str, video_dir: str) -> List[Dict[str, Any]]:
    data = load_json(plan_path)
    steps = sorted(data.get("steps", []), key=lambda s: s.get("step_id", 0))
    records: List[Dict[str, Any]] = []
    for step in steps:
        step_id = int(step.get("step_id", 0))
        critical_frames = step.get("critical_frames", []) or []
        if not critical_frames:
            print(f"[WARN] Step {step_id} 缺少 critical_frames，跳过。")
            continue
        last_cf = critical_frames[-1]
        frame_index = last_cf.get("frame_index")
        keyframe_path = last_cf.get("keyframe_image_path")
        if frame_index is None:
            print(f"[WARN] Step {step_id} 缺少 frame_index，跳过。")
            continue
        step_goal = step.get("step_goal", "")
        keyframe_path = find_keyframe_path(video_dir, step_id, step_goal, int(frame_index), keyframe_path)
        if not keyframe_path:
            print(f"[WARN] Step {step_id} 未找到关键帧图片（尾帧），跳过。")
            continue
        ts = parse_timestamp_from_filename(keyframe_path)
        if ts is None:
            print(f"[WARN] Step {step_id} 无法从文件名解析时间戳，跳过。文件: {keyframe_path}")
            continue
        records.append(
            {
                "step_id": step_id,
                "frame_index": int(frame_index),
                "timestamp_sec": ts,
                "keyframe_path": keyframe_path,
            }
        )
    return records


def build_segments(last_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if not last_frames:
        return segments
    ordered = sorted(last_frames, key=lambda x: x["step_id"])
    for item in ordered:
        segments.append(
            {
                "label": f"segment_start_to_step{item['step_id']:02d}_last",
                "start_sec": 0.0,
                "end_sec": item["timestamp_sec"],
                "step_id": item["step_id"],
                "last_keyframe": item,
            }
        )
    return segments


def run_ffmpeg(
    ffmpeg_bin: str, src: str, start: float, duration: float, dst: str, overwrite: bool, dry_run: bool
) -> bool:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        src,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        dst,
    ]
    if overwrite:
        cmd.insert(1, "-y")
    else:
        cmd.insert(1, "-n")
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 失败: {e}")
        return False


def save_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "segments_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 清单已保存: {path}")


def main() -> None:
    args = parse_args()
    video_dir = os.path.abspath(args.video_output_dir)
    plan_path = os.path.join(video_dir, "causal_plan_with_keyframes.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"未找到规划文件: {plan_path}")

    source_video = resolve_source_video(video_dir, args.source_video)
    if not os.path.exists(source_video):
        raise FileNotFoundError(f"源视频不存在: {source_video}")

    last_frames = collect_last_keyframes(plan_path, video_dir)
    if not last_frames:
        print("[WARN] 未找到任何可用的尾帧，退出。")
        return

    segments = build_segments(last_frames)
    if not segments:
        print("[WARN] 没有可裁剪的片段，退出。")
        return

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(video_dir, "cumulative_last_frame_segments")
    os.makedirs(output_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "source_video": source_video,
        "video_output_dir": video_dir,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "segments": [],
    }

    for seg in segments:
        start = 0.0
        end = float(seg["end_sec"])
        duration = end - start
        if duration <= 0:
            print(f"[WARN] 片段 {seg['label']} 时长非正，跳过。")
            continue
        outfile = os.path.join(output_dir, f"{seg['label']}.mp4")
        if os.path.exists(outfile) and not args.overwrite:
            print(f"[INFO] 已存在且未开启覆盖，跳过生成: {outfile}")
            status = "skipped_exists"
        else:
            ok = run_ffmpeg(args.ffmpeg_bin, source_video, start, duration, outfile, args.overwrite, args.dry_run)
            status = "ok" if ok else "failed"
        manifest["segments"].append(
            {
                "label": seg["label"],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration,
                "step_id": seg["step_id"],
                "last_keyframe": seg["last_keyframe"],
                "output_path": outfile,
                "status": status,
            }
        )

    save_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
