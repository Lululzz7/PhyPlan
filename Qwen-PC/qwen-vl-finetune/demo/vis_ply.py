#!/usr/bin/env python3
# 固定路径版：可视化指定点云（支持体素/随机下采样与颜色归一化）

import numpy as np
import open3d as o3d
from pathlib import Path

# 可按需修改以下参数
PLY_PATH = "/home/luzheng/3DVLM/Qwen3-VL-main/qwen-vl-finetune/demo/scene0000_01_vh_clean.ply"
VOXEL_SIZE = 0       # 体素下采样体素大小（>0 启用，建议 0.02 降载）
MAX_POINTS = 0           # 随机下采样至最多点数（>0 启用，如 200000）
NORMALIZE_COLORS = False  # 若颜色为 0–255，归一化到 [0,1]
SAVE_OUT = None          # 可选：保存处理后点云路径（.ply/.pcd）；None 则不保存

in_path = Path(PLY_PATH)
if not in_path.exists():
    raise FileNotFoundError(f"输入文件不存在: {in_path}")

print(f"[Info] 读取点云: {in_path}")
pcd = o3d.io.read_point_cloud(str(in_path))
n0 = len(pcd.points)
has_colors = (len(pcd.colors) == n0 and n0 > 0)
print(f"[Info] 点数: {n0}; 含颜色: {has_colors}")

# 边界盒
if n0 > 0:
    pts = np.asarray(pcd.points)
    print(f"[BBox] min={pts.min(axis=0)}  max={pts.max(axis=0)}")

# 体素下采样
if VOXEL_SIZE and VOXEL_SIZE > 0:
    print(f"[Downsample] Voxel size = {VOXEL_SIZE}")
    pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL_SIZE))
    print(f"[Downsample] 点数: {len(pcd.points)}")

# 随机下采样
if MAX_POINTS and MAX_POINTS > 0 and len(pcd.points) > MAX_POINTS:
    m = len(pcd.points)
    idx = np.random.choice(m, size=MAX_POINTS, replace=False)
    pts = np.asarray(pcd.points)[idx]
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(pcd.colors) == m:
        cols = np.asarray(pcd.colors)[idx]
        pcd.colors = o3d.utility.Vector3dVector(cols)
    print(f"[Downsample-Rand] 点数: {len(pcd.points)}")

# 颜色归一化：若显式要求或检测到颜色范围>1
if len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0:
    cols = np.asarray(pcd.colors)
    if NORMALIZE_COLORS or (cols.max() > 1.0):
        cols = np.clip(cols / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        print("[Color] 已将颜色归一化到 [0,1]")

# 可选保存
if SAVE_OUT:
    out_path = Path(SAVE_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"[Save] 已保存处理后点云到: {out_path}")

# 坐标系 + 可视化
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="点云可视化")
