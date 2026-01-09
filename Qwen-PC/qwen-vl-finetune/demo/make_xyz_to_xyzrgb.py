#!/usr/bin/env python3
"""
批量将 (N,3) 点云扩展为 (N,6)，并保存到目标 demo 路径。

用法示例：
  1) 转换单个文件：
     python qwen-vl-finetune/demo/make_xyz_to_xyzrgb.py \
       --input qwen-vl-finetune/demo/points/sample_xyz.npy \
       --output-dir qwen-vl-finetune/demo/points

  2) 转换目录下所有 .npy：
     python qwen-vl-finetune/demo/make_xyz_to_xyzrgb.py \
       --input qwen-vl-finetune/demo/points \
       --glob "*.npy" \
       --output-dir qwen-vl-finetune/demo/points

说明：
  - 支持 .npy / .npz（.npz 取第一个数组）
  - (N,3) → (N,6)，后三维默认填充 0；若原始为 (N,6) 则直接复制；其他维度报错
"""

import argparse
import sys
from pathlib import Path
import numpy as np


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    elif path.suffix.lower() == ".npz":
        npz = np.load(str(path))
        keys = list(npz.keys())
        if not keys:
            raise ValueError(f"npz 文件无数组: {path}")
        arr = npz[keys[0]]
    else:
        raise ValueError(f"不支持的文件类型: {path.suffix}")
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"文件不是 numpy 数组: {path}")
    return arr


def _convert_xyz_to_xyzrgb(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"数组维度应为 2D，实际为 {arr.ndim}D")
    d = arr.shape[1]
    if d == 6:
        # 已经是 (N,6)，直接返回 float32 拷贝
        return arr.astype(np.float32, copy=False)
    if d != 3:
        raise ValueError(f"只支持扩展 (N,3) 到 (N,6)，实际通道维为 {d}")
    xyz = arr[:, :3]
    rgb = np.full_like(xyz, fill_value)
    out = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="扩展 (N,3) 点云为 (N,6) 并保存")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件或目录（.npy/.npz 或包含这些文件的目录）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录，转换后的 .npy 会写入此目录")
    parser.add_argument("--glob", type=str, default="*.npy",
                        help="当 --input 为目录时的匹配模式（默认 *.npy）")
    parser.add_argument("--suffix", type=str, default="_xyzrgb.npy",
                        help="输出文件名后缀（默认 _xyzrgb.npy）")
    parser.add_argument("--fill", type=float, default=0.0,
                        help="扩展后三维的填充值（默认 0.0）")

    args = parser.parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if in_path.is_file():
        targets = [in_path]
    elif in_path.is_dir():
        targets = sorted(in_path.glob(args.glob))
    else:
        print(f"[错误] 输入路径不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    if not targets:
        print(f"[提示] 未匹配到文件: {in_path} ({args.glob})")
        sys.exit(0)

    for src in targets:
        try:
            arr = _load_array(src)
            arr6 = _convert_xyz_to_xyzrgb(arr, fill_value=args.fill)
            dst = out_dir / (src.stem + args.suffix)
            np.save(str(dst), arr6)
            print(f"[OK] {src} -> {dst}  shape={arr6.shape}")
        except Exception as e:
            print(f"[跳过] {src}: {e}")


if __name__ == "__main__":
    main()

