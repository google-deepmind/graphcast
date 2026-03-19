# ─────────────────────────────────────────────────────────────────────────────
# 10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly_LocalAffine.py
# GraphCast / GenCast ReadOut Selective Storm Guidance - DATA ONLY VERSION
# 
# 主要功能：
# - 执行 rollout 推理（baseline 和 guided）
# - 保存数据为 NetCDF 格式（在 rollout_data/ 子目录）
# - 跳过所有可视化和视频生成
# - 支持多种 Guidance 模式（互斥选择）：
#   * none: Baseline only（无 guidance）
#   * direct_optim: 直接优化 x_t（原 readout-based guidance）
#   * input_manipulation: 输入条件修改（Intensity Scaling + Spatial Shift）
#   * local_affine: 局部仿射变换优化（优化 warp 参数）
# 
# 配套使用：
# - 运行此脚本生成数据
# - 运行 10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py 进行可视化
# 
# 使用方式：
# 1. 修改脚本底部的参数配置区（CONFIGURATION SECTION）
# 2. 选择 GUIDANCE_METHOD（四选一）
# 3. 运行脚本：python 10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly_LocalAffine.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# 安全护栏：未指定时默认启用 "cuda,cpu"（避免 jax.debug.print 找不到 CPU 后端报错）
import os as _os

if "JAX_PLATFORMS" not in _os.environ:
    _os.environ["JAX_PLATFORMS"] = "cuda,cpu"

# 内存优化：限制 JAX 内存预分配
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in _os.environ:
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "XLA_PYTHON_CLIENT_ALLOCATOR" not in _os.environ:
    _os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import dataclasses
import gc
import glob
import json
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import dask
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy import ndimage

# GraphCast / GenCast
from google.cloud import storage
from graphcast.denoiser import ReadOutDenoiserArchitectureConfig

from graphcast.era5_dataset_12_01_PureJax import (
    DateMergedERA5TyphoonSizeDataset,
    data_merge_make_circular_one_hot_varradius_cpu,
)
from graphcast.gencast import ReadOutNoiseConfig, ReadOutSamplerConfig
from graphcast.vis import (
    generate_comparison_gifs_parallel,
    visualize_onehot_readout_simple,
)
from graphcast.vis_guidance import (
    create_all_variable_comparisons,
    create_readout_comparison_plot,
)
from torch.utils.data import DataLoader, Subset

from graphcast import (
    checkpoint,
    gencast,
    nan_cleaning,
    normalization,
    xarray_jax,
)


# =========================
# Timing Logger: 时间记录工具
# =========================
class TimingLogger:
    """时间记录器，支持嵌套时间记录和自动保存"""
    
    def __init__(self, output_path: Optional[str] = None):
        self.records = []
        self.stack = []  # 用于嵌套记录
        self.output_path = output_path
        self.start_time = time.time()
        self.current_section = None
    
    def start(self, section_name: str):
        """开始记录一个时间段"""
        try:
            t = time.time()
            self.stack.append({
                "name": section_name,
                "start_time": t,
                "subsections": []
            })
            return self
        except Exception as e:
            # 如果出现异常，打印警告但不中断程序
            print(f"[TimingLogger Warning] Error starting section '{section_name}': {e}")
            return self
    
    def end(self, section_name: Optional[str] = None):
        """结束当前时间段"""
        if not self.stack:
            # 如果栈为空，说明没有对应的start，可能是异常情况，直接返回
            return None
        
        try:
            t = time.time()
            current = self.stack[-1]
            duration = t - current["start_time"]
            
            # 如果有子时间段，计算纯时间（总时间 - 子时间段时间）
            subsections_time = sum(sub["duration"] for sub in current["subsections"])
            pure_time = max(0, duration - subsections_time)  # 确保非负
            
            record = {
                "name": current["name"],
                "duration": duration,
                "pure_time": pure_time,
                "subsections": current["subsections"],
                "start_time": current["start_time"],
                "end_time": t
            }
            
            # 从栈中移除
            self.stack.pop()
            
            # 如果有父时间段，添加到父时间段的子时间段中
            if self.stack:
                self.stack[-1]["subsections"].append(record)
            else:
                # 顶层记录
                self.records.append(record)
            
            return record
        except Exception as e:
            # 如果出现异常，打印警告但不中断程序
            print(f"[TimingLogger Warning] Error ending section: {e}")
            if self.stack:
                self.stack.pop()  # 至少清理栈
            return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack:
            # 自动结束未完成的时间段
            while self.stack:
                self.end()
        return False
    
    def save(self, output_path: Optional[str] = None):
        """保存时间记录到文件"""
        save_path = output_path or self.output_path
        if not save_path:
            return
        
        try:
            # 清理未完成的记录（如果有）
            while self.stack:
                print(f"[TimingLogger Warning] Cleaning up {len(self.stack)} unfinished timing records")
                self.end()
            
            total_time = time.time() - self.start_time
            
            # 保存为文本格式
            txt_path = save_path if save_path.endswith('.txt') else save_path.replace('.json', '.txt')
            # 确保目录存在
            txt_dir = os.path.dirname(txt_path)
            if txt_dir:
                os.makedirs(txt_dir, exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Timing Log\n")
                f.write("=" * 80 + "\n")
                f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n")
                f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"Total Duration: {total_time:.2f}s ({total_time/60:.2f}min)\n")
                f.write("\n")
                
                for record in self.records:
                    self._write_record(f, record, indent=0)
            
            # 保存为JSON格式（便于后续分析）
            json_path = save_path if save_path.endswith('.json') else save_path.replace('.txt', '.json')
            json_data = {
                "start_time": self.start_time,
                "end_time": time.time(),
                "total_duration": total_time,
                "records": self.records
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n[Timing Logger] Saved timing log to {txt_path} and {json_path}")
        except Exception as e:
            print(f"[TimingLogger Error] Failed to save timing log: {e}")
            import traceback
            traceback.print_exc()
    
    def record_duration(self, section_name: str, duration: float, subsections: Optional[List[Dict]] = None):
        """手动记录一个时间段（用于多进程场景，时间已在worker中计算）
        
        Args:
            section_name: 时间段名称
            duration: 持续时间（秒）
            subsections: 可选的子时间段列表
        """
        try:
            t = time.time()
            record = {
                "name": section_name,
                "duration": duration,
                "pure_time": duration - sum(sub.get("duration", 0) for sub in (subsections or [])),
                "subsections": subsections or [],
                "start_time": t - duration,  # 反推开始时间
                "end_time": t
            }
            
            # 如果有父时间段，添加到父时间段的子时间段中
            if self.stack:
                self.stack[-1]["subsections"].append(record)
            else:
                # 顶层记录
                self.records.append(record)
            
            return record
        except Exception as e:
            print(f"[TimingLogger Warning] Error recording duration for '{section_name}': {e}")
            return None
    
    def _write_record(self, f, record, indent=0):
        """递归写入记录"""
        prefix = "  " * indent
        f.write(f"{prefix}--- {record['name']} ---\n")
        f.write(f"{prefix}  Duration: {record['duration']:.2f}s")
        if record['pure_time'] > 0.01:  # 只显示有意义的纯时间
            f.write(f" (pure: {record['pure_time']:.2f}s)")
        f.write("\n")
        
        if record['subsections']:
            for sub in record['subsections']:
                self._write_record(f, sub, indent + 1)


# =========================
# Manual Guidance: 坐标转换与 Mask 生成
# =========================
def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """将经纬度转换为 181x360 网格索引
    
    Args:
        lat: 纬度，范围 -90 到 90
        lon: 经度，范围 0 到 360 (或 -180 到 180)
    
    Returns:
        (row, col): 网格索引，row ∈ [0, 180], col ∈ [0, 359]
    """
    # lat: -90 到 90 → row: 0 到 180
    row = int(lat + 90)
    row = max(0, min(180, row))
    # lon: 0 到 360 (或 -180 到 180) → col: 0 到 359
    col = int(lon) % 360
    return row, col


def make_manual_guidance_mask(
    batch_size: int,
    targets: List[Dict],
    height: int = 181,
    width: int = 360,
) -> torch.Tensor:
    """从用户指定的坐标生成 guidance mask
    
    Args:
        batch_size: 批次大小
        targets: 目标位置列表，支持两种格式：
            - 经纬度：[{"lat": 25.0, "lon": 120.0, "radius": 5}, ...]
            - 网格索引：[{"row": 115, "col": 120, "radius": 5}, ...]
        height: 网格高度（默认 181）
        width: 网格宽度（默认 360）
    
    Returns:
        one_hot: 形状为 (batch_size, height, width, 2) 的 one-hot tensor
    """
    centers_with_r = []
    for t in targets:
        if "row" in t and "col" in t:
            # 直接使用网格索引
            row, col = t["row"], t["col"]
        else:
            # 从经纬度转换
            row, col = latlon_to_grid(t["lat"], t["lon"])
        radius = t.get("radius", 5)
        centers_with_r.append((row, col, radius))
    
    # 复用现有函数生成 one-hot mask
    return data_merge_make_circular_one_hot_varradius_cpu(
        batch_size, height, width, centers_with_r
    )


def _targets_slug(targets: List[Dict]) -> str:
    """生成目标位置的简短描述字符串"""
    parts = []
    for t in targets:
        if "row" in t and "col" in t:
            parts.append(f"r{t['row']}c{t['col']}rad{t.get('radius', 5)}")
        else:
            parts.append(f"lat{t['lat']:.1f}lon{t['lon']:.1f}rad{t.get('radius', 5)}")
    return "__".join(parts)


# =========================
# Selective Storm Guidance: 风暴提取与选择
# =========================
# 全局缓存：{end_date: List[Dict]} - 缓存风暴提取结果
_storms_cache = {}


def extract_storms_from_end_date(
    ctx: Context,
    end_date: str,
) -> List[Dict]:
    """从 end_date 提取所有风暴信息（带缓存）
    
    Args:
        ctx: Context 对象
        end_date: 结束日期字符串，如 "2017-09-08 00:00:00"
    
    Returns:
        storms: List of dicts, each containing:
            - storm_id: 自动分配的 ID（0, 1, 2, ...）
            - lat: 纬度
            - lon: 经度
            - rsize: 半径（度）
            - row: 网格行索引
            - col: 网格列索引
            - min_row, max_row, min_col, max_col: bounding box 边界
    """
    # 检查缓存
    if end_date in _storms_cache:
        print(f"    [Extract Storms] Using cached storms for end_date: {end_date}")
        return _storms_cache[end_date]
    
    # 加载 end_date 的数据
    print(f"    [Extract Storms] Loading data for end_date: {end_date}...")
    t0 = time.time()
    wanted_loader = build_wanted_subloader(ctx.eval_ds, [end_date], batch_size=1)
    it = iter(wanted_loader)
    print(f"    [Extract Storms] DataLoader created, fetching data from zarr/GCS (this may take 30-60s)...")
    t1 = time.time()
    _, _, _, one_hot_original, _ = next(it)
    print(f"    [Extract Storms] Data loaded in {time.time() - t1:.2f}s (total: {time.time() - t0:.2f}s)")
    
    # 从 one_hot_original 提取风暴位置
    print(f"    [Extract Storms] Extracting storm regions from one-hot mask...")
    one_hot_np = one_hot_original.numpy() if hasattr(one_hot_original, "numpy") else np.asarray(one_hot_original)
    storm_mask = one_hot_np[0, ..., 1]  # (H, W)，取第一个 batch
    
    # 使用 scipy.ndimage.label 提取连通区域（识别风暴）
    labeled, num_features = ndimage.label(storm_mask)
    print(f"    [Extract Storms] Found {num_features} storm region(s)")
    
    storms = []
    for storm_id in range(1, num_features + 1):
        # 找到这个风暴的所有像素
        rows, cols = np.where(labeled == storm_id)
        if len(rows) == 0:
            continue
        
        # 计算中心位置
        center_row = int(rows.mean())
        center_col = int(cols.mean())
        
        # 计算 bounding box
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())
        
        # 估算半径（使用 bounding box 的对角线的一半）
        radius = max(max_row - min_row, max_col - min_col) // 2
        radius = max(1, radius)  # 至少为 1
        
        # 转换为 lat/lon
        lat = center_row - 90
        lon = center_col
        
        storms.append({
            "storm_id": storm_id - 1,  # 从 0 开始
            "lat": float(lat),
            "lon": float(lon),
            "rsize": float(radius),
            "row": center_row,
            "col": center_col,
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
        })
    
    print(f"    [Extract Storms] Extraction complete, {len(storms)} storm(s) identified")
    
    # 缓存结果
    _storms_cache[end_date] = storms
    print(f"    [Extract Storms] Cached storms for end_date: {end_date}")
    
    return storms


def print_storms_list(storms: List[Dict]):
    """列出所有风暴供用户查看"""
    print("\n=== Available Storms at End Date ===")
    for storm in storms:
        print(f"  Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, "
              f"rsize={storm['rsize']:.1f}, row={storm['row']}, col={storm['col']}")
    print()


# 全局缓存：{(start_date, end_date): storm_id}
_storm_id_cache = {}


def get_storm_id_for_date_pair(
    start_date: str,
    end_date: str,
    config_storm_id: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> int:
    """获取 storm_id，支持配置优先、缓存、IO输入
    
    Args:
        start_date: 输入时间点
        end_date: 目标时间点
        config_storm_id: 配置中指定的 storm_id（如果提供）
        ctx: Context 对象（用于提取风暴信息）
    
    Returns:
        selected_id: 选定的 storm_id
    """
    cache_key = (start_date, end_date)
    
    # 情况A：配置中已指定
    if config_storm_id is not None:
        _storm_id_cache[cache_key] = config_storm_id
        print(f"    [Storm Selection] Using configured storm_id: {config_storm_id}")
        return config_storm_id
    
    # 情况B：检查缓存
    if cache_key in _storm_id_cache:
        cached_id = _storm_id_cache[cache_key]
        print(f"    [Storm Selection] Using cached storm_id: {cached_id}")
        return cached_id
    
    # 情况C：需要 IO 输入（仅一次）
    if ctx is None:
        raise ValueError("ctx is required when config_storm_id is None and cache is empty")
    
    storms = extract_storms_from_end_date(ctx, end_date)
    
    if len(storms) == 0:
        raise ValueError(f"No storms found at end_date: {end_date}")
    
    print_storms_list(storms)
    
    if len(storms) == 1:
        print(f"    [Storm Selection] Only one storm found, selecting Storm ID 0")
        selected_id = 0
    else:
        selected_id = input(f"    [Storm Selection] Select storm ID (0-{len(storms)-1}): ")
        try:
            selected_id = int(selected_id)
            if selected_id < 0 or selected_id >= len(storms):
                raise ValueError(f"Invalid storm ID: {selected_id}")
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")
    
    # 缓存选择
    _storm_id_cache[cache_key] = selected_id
    print(f"    [Storm Selection] Selected storm_id {selected_id} cached for ({start_date}, {end_date})")
    return selected_id


def extract_storms_from_one_hot(one_hot: torch.Tensor) -> List[Dict]:
    """从 one_hot mask 提取所有风暴信息（通用函数）
    
    Args:
        one_hot: (B, H, W, 2) 或 (H, W, 2) 的 one-hot tensor
    
    Returns:
        storms: List of dicts, each containing storm information
    """
    one_hot_np = one_hot.numpy() if hasattr(one_hot, "numpy") else np.asarray(one_hot)
    if one_hot_np.ndim == 4:
        storm_mask = one_hot_np[0, ..., 1]  # (H, W)，取第一个 batch
    else:
        storm_mask = one_hot_np[..., 1]  # (H, W)
    
    # 使用 scipy.ndimage.label 提取连通区域
    labeled, num_features = ndimage.label(storm_mask)
    
    storms = []
    for storm_id in range(1, num_features + 1):
        rows, cols = np.where(labeled == storm_id)
        if len(rows) == 0:
            continue
        
        center_row = int(rows.mean())
        center_col = int(cols.mean())
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())
        radius = max(max_row - min_row, max_col - min_col) // 2
        radius = max(1, radius)
        
        lat = center_row - 90
        lon = center_col
        
        storms.append({
            "storm_id": storm_id - 1,
            "lat": float(lat),
            "lon": float(lon),
            "rsize": float(radius),
            "row": center_row,
            "col": center_col,
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
        })
    
    return storms


def match_storm_between_dates(
    start_storms: List[Dict],
    end_storm: Dict,
    distance_threshold: float = 10.0,  # 度
) -> Optional[int]:
    """匹配 start_date 和 end_date 的风暴（通过位置距离）
    
    Args:
        start_storms: start_date 的所有风暴列表
        end_storm: end_date 的选定风暴
        distance_threshold: 最大匹配距离（度）
    
    Returns:
        matched_storm_id: start_date 中对应的 storm_id，如果未找到则返回 None
    """
    end_lat, end_lon = end_storm["lat"], end_storm["lon"]
    
    best_match_id = None
    min_distance = float('inf')
    
    for start_storm in start_storms:
        start_lat, start_lon = start_storm["lat"], start_storm["lon"]
        # 简单的欧氏距离（度）
        distance = np.sqrt((start_lat - end_lat)**2 + (start_lon - end_lon)**2)
        
        if distance < distance_threshold and distance < min_distance:
            min_distance = distance
            best_match_id = start_storm["storm_id"]
    
    if best_match_id is not None:
        print(f"    [Storm Matching] Matched end_date storm (lat={end_lat:.1f}, lon={end_lon:.1f}) "
              f"to start_date storm_id {best_match_id} (lat={start_storms[best_match_id]['lat']:.1f}, "
              f"lon={start_storms[best_match_id]['lon']:.1f}, distance={min_distance:.2f}°)")
    else:
        print(f"    [Storm Matching] Warning: No matching storm found in start_date for "
              f"end_date storm (lat={end_lat:.1f}, lon={end_lon:.1f})")
    
    return best_match_id


def build_final_target_mask(
    one_hot_start: torch.Tensor,
    start_storms: List[Dict],
    mode: str,  # 新增：模式参数 "steer_one" 或 "delete_one"
    selected_storm_id_in_start: Optional[int],
    manual_target: Optional[Dict] = None,  # "steer_one" 模式需要，"delete_one" 模式不需要
    batch_size: int = 1,
    height: int = 181,
    width: int = 360,
) -> torch.Tensor:
    """构建最终的 target mask，支持两种模式：
    
    模式1 "steer_one":
    - 从 start_date 提取所有风暴
    - 将选定的风暴替换为 manual target（移动位置）
    - 保留其他风暴
    
    模式2 "delete_one":
    - 从 start_date 提取所有风暴
    - 删除选定的风暴
    - 保留其他风暴
    
    Args:
        one_hot_start: start_date 的原始 one-hot mask (B, H, W, 2)
        start_storms: start_date 的所有风暴列表
        mode: 模式选择 "steer_one" 或 "delete_one"
        selected_storm_id_in_start: 要操作的风暴 ID（在 start_date 中）
        manual_target: 用户指定的目标位置 {"lat": ..., "lon": ..., "radius": ...}（仅 "steer_one" 模式需要）
        batch_size: 批次大小
        height: 网格高度
        width: 网格宽度
    
    Returns:
        one_hot_final: (B, H, W, 2) 的最终 one-hot mask
    """
    # 从 start_date 的 one_hot 提取所有风暴的 mask
    one_hot_np = one_hot_start.numpy() if hasattr(one_hot_start, "numpy") else np.asarray(one_hot_start)
    storm_mask_start = one_hot_np[0, ..., 1]  # (H, W)
    labeled_start, num_features_start = ndimage.label(storm_mask_start)
    
    # 创建新的 mask（初始化为 0）
    final_mask = np.zeros((height, width), dtype=np.int64)
    
    # 添加所有 start_date 的风暴，除了要操作的那个
    for storm in start_storms:
        storm_id_in_labeled = storm["storm_id"] + 1  # labeled 从 1 开始
        if selected_storm_id_in_start is not None and storm["storm_id"] == selected_storm_id_in_start:
            # 跳过要操作的风暴
            if mode == "steer_one":
                print(f"    [Build Target Mask] Skipping start_date storm_id {storm['storm_id']} "
                      f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f}) - will be replaced by manual target")
            elif mode == "delete_one":
                print(f"    [Build Target Mask] Removing start_date storm_id {storm['storm_id']} "
                      f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f})")
            continue
        # 添加这个风暴的区域
        final_mask |= (labeled_start == storm_id_in_labeled)
        print(f"    [Build Target Mask] Keeping start_date storm_id {storm['storm_id']} "
              f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f})")
    
    # 如果是 "steer_one" 模式，添加 manual target storm
    if mode == "steer_one":
        if manual_target is None:
            raise ValueError("manual_target is required for 'steer_one' mode")
        if "row" in manual_target and "col" in manual_target:
            target_row, target_col = manual_target["row"], manual_target["col"]
        else:
            target_row, target_col = latlon_to_grid(manual_target["lat"], manual_target["lon"])
        target_radius = manual_target.get("radius", 5)
        
        # 创建圆形 mask
        yy = np.arange(height)[:, None]
        xx = np.arange(width)[None, :]
        target_circle = ((yy - target_row)**2 + (xx - target_col)**2 <= target_radius**2)
        final_mask |= target_circle
        
        print(f"    [Build Target Mask] Added manual target at (lat={manual_target.get('lat', 'N/A')}, "
              f"lon={manual_target.get('lon', 'N/A')}, row={target_row}, col={target_col}, radius={target_radius})")
    elif mode == "delete_one":
        # "delete_one" 模式不需要添加任何东西，只需要保留其他storm
        num_kept = len(start_storms) - (1 if selected_storm_id_in_start is not None else 0)
        print(f"    [Build Target Mask] Deleted storm_id {selected_storm_id_in_start}, kept {num_kept} other storm(s)")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'steer_one' or 'delete_one'")
    
    # 转换为 one-hot
    labels = np.broadcast_to(final_mask, (batch_size, height, width))
    labels_t = torch.from_numpy(labels)
    one_hot_final = torch.nn.functional.one_hot(labels_t, num_classes=2)
    
    return one_hot_final


def compute_bounding_box_mask(
    selected_storm: Dict,
    target_storm: Dict,  # 用户指定的目标位置
    batch_size: int,
    height: int = 181,
    width: int = 360,
    padding: int = 5,  # 在 bounding box 周围添加 padding
) -> torch.Tensor:
    """计算包含选定原始风暴和目标风暴的 bounding box mask
    
    Args:
        selected_storm: 选定的原始风暴信息（从 end_date 提取）
        target_storm: 目标风暴信息（从 manual_targets 生成）
        batch_size: 批次大小
        height: 网格高度
        width: 网格宽度
        padding: bounding box 周围的 padding（网格点）
    
    Returns:
        guidance_mask: (B, H, W) binary mask
    """
    # 获取选定原始风暴的边界
    storm1_min_row = selected_storm.get("min_row", selected_storm["row"] - int(selected_storm["rsize"]))
    storm1_max_row = selected_storm.get("max_row", selected_storm["row"] + int(selected_storm["rsize"]))
    storm1_min_col = selected_storm.get("min_col", selected_storm["col"] - int(selected_storm["rsize"]))
    storm1_max_col = selected_storm.get("max_col", selected_storm["col"] + int(selected_storm["rsize"]))
    
    # 目标风暴的边界（从 manual_targets 生成）
    if "row" in target_storm and "col" in target_storm:
        target_row, target_col = target_storm["row"], target_storm["col"]
    else:
        target_row, target_col = latlon_to_grid(target_storm["lat"], target_storm["lon"])
    target_radius = target_storm.get("radius", 5)
    
    storm2_min_row = target_row - target_radius
    storm2_max_row = target_row + target_radius
    storm2_min_col = target_col - target_radius
    storm2_max_col = target_col + target_radius
    
    # 计算联合 bounding box
    min_row = max(0, min(storm1_min_row, storm2_min_row) - padding)
    max_row = min(height - 1, max(storm1_max_row, storm2_max_row) + padding)
    min_col = max(0, min(storm1_min_col, storm2_min_col) - padding)
    max_col = min(width - 1, max(storm1_max_col, storm2_max_col) + padding)
    
    # 生成 mask
    mask = np.zeros((batch_size, height, width), dtype=np.float32)
    mask[:, min_row:max_row+1, min_col:max_col+1] = 1.0
    
    print(f"    [Bounding Box] rows [{min_row}-{max_row}], cols [{min_col}-{max_col}], "
          f"size: {mask.sum().item()} pixels")
    
    return torch.from_numpy(mask)


# =========================
# Loss 监控工具：保存和可视化
# =========================
def plot_loss_curves(loss_history: Dict[int, List[float]], output_path: str):
    """绘制每个 denoising step 的 loss 下降曲线
    
    Args:
        loss_history: {step: [loss_0, loss_1, ..., loss_n]}
        output_path: 保存图片的完整路径
    """
    import matplotlib.pyplot as plt
    
    num_steps = len(loss_history)
    if num_steps == 0:
        print("[Loss Curve] No loss data to plot")
        return
    
    # 创建子图：每个 step 一个
    fig, axes = plt.subplots(
        num_steps, 1,
        figsize=(12, 4 * num_steps),
        squeeze=False
    )
    axes = axes.flatten()
    
    for idx, (step, losses) in enumerate(sorted(loss_history.items())):
        ax = axes[idx]
        iterations = list(range(len(losses)))
        
        # 绘制 loss 曲线
        ax.plot(iterations, losses, marker='o', linewidth=2, markersize=4, color='#2E86AB')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title(f'Denoising Step {step} - Guidance Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 标注最高点和最低点
        max_loss = max(losses)
        min_loss = min(losses)
        max_idx = losses.index(max_loss)
        min_idx = losses.index(min_loss)
        
        # 最高点（红色）
        ax.plot(max_idx, max_loss, 'ro', markersize=10, zorder=5)
        ax.text(max_idx, max_loss, f'  Max: {max_loss:.6f}',
                fontsize=11, color='red', fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left')
        
        # 最低点（绿色）
        ax.plot(min_idx, min_loss, 'go', markersize=10, zorder=5)
        ax.text(min_idx, min_loss, f'  Min: {min_loss:.6f}',
                fontsize=11, color='green', fontweight='bold',
                verticalalignment='top', horizontalalignment='left')
        
        # 添加统计信息
        reduction = max_loss - min_loss
        reduction_pct = (reduction / max_loss * 100) if max_loss > 0 else 0
        ax.text(0.02, 0.98, 
                f'Reduction: {reduction:.6f} ({reduction_pct:.2f}%)\nIterations: {len(losses)}',
                transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置 x 轴标签
        if idx == num_steps - 1:
            ax.set_xlabel('Optimization Iteration', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Loss Curve] Saved to {output_path}")


def save_loss_summary(loss_history: Dict[int, List[float]], output_path: str):
    """保存 loss 统计摘要到文本文件
    
    Args:
        loss_history: {step: [loss_0, loss_1, ..., loss_n]}
        output_path: 保存文本文件的完整路径
    """
    if len(loss_history) == 0:
        print("[Loss Summary] No loss data to save")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Guidance Loss Summary\n")
        f.write("=" * 70 + "\n\n")
        
        # 整体统计
        all_losses = []
        for losses in loss_history.values():
            all_losses.extend(losses)
        
        if all_losses:
            f.write("Overall Statistics:\n")
            f.write(f"  Total Steps with Guidance: {len(loss_history)}\n")
            f.write(f"  Total Optimization Iterations: {len(all_losses)}\n")
            f.write(f"  Global Min Loss: {min(all_losses):.8f}\n")
            f.write(f"  Global Max Loss: {max(all_losses):.8f}\n")
            f.write(f"  Global Mean Loss: {sum(all_losses)/len(all_losses):.8f}\n")
            f.write("\n")
            f.write("-" * 70 + "\n\n")
        
        # 每个 step 的详细统计
        for step in sorted(loss_history.keys()):
            losses = loss_history[step]
            f.write(f"Denoising Step {step}:\n")
            f.write(f"  Optimization Iterations:   {len(losses)}\n")
            f.write(f"  Initial Loss (iter 0):     {losses[0]:.8f}\n")
            f.write(f"  Final Loss (iter {len(losses)-1}):      {losses[-1]:.8f}\n")
            f.write(f"  Max Loss:                  {max(losses):.8f} (at iter {losses.index(max(losses))})\n")
            f.write(f"  Min Loss:                  {min(losses):.8f} (at iter {losses.index(min(losses))})\n")
            f.write(f"  Loss Reduction:            {losses[0] - losses[-1]:.8f}\n")
            f.write(f"  Reduction Percentage:      {(1 - losses[-1]/losses[0])*100:.2f}%\n")
            f.write(f"  Mean Loss:                 {sum(losses)/len(losses):.8f}\n")
            f.write(f"  Std Dev:                   {np.std(losses):.8f}\n")
            
            # 收敛性分析（检查最后几次迭代的变化）
            if len(losses) >= 5:
                last_5 = losses[-5:]
                last_5_change = max(last_5) - min(last_5)
                f.write(f"  Last 5 iterations range:   {last_5_change:.8f} (convergence indicator)\n")
            
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("Note: Loss values < 0 indicate no optimization was performed.\n")
        f.write("=" * 70 + "\n")
    
    print(f"[Loss Summary] Saved to {output_path}")


# =========================
# Mask 可视化：Target Storm vs Guidance Mask
# =========================
def visualize_mask_comparison(
    one_hot_target: torch.Tensor,  # target storm 的 one-hot mask (B, H, W, 2)
    guidance_mask: torch.Tensor,    # bounding box mask (B, H, W)
    output_path: str,
    selected_storm_info: Optional[Dict] = None,
    target_storm_info: Optional[Dict] = None,
    sample_idx: int = 0,
):
    """并排可视化 target storm mask 和 guidance mask，用于检查 mask 是否正确
    
    Args:
        one_hot_target: target storm 的 one-hot mask，shape (B, H, W, 2)
        guidance_mask: bounding box mask，shape (B, H, W)
        output_path: 保存图片的完整路径
        selected_storm_info: 选定的原始风暴信息（用于标题显示）
        target_storm_info: 目标风暴信息（用于标题显示）
        sample_idx: 使用哪个 batch 样本（默认 0）
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    
    # 转换为 numpy
    one_hot_np = one_hot_target.numpy() if hasattr(one_hot_target, "numpy") else np.asarray(one_hot_target)
    guidance_np = guidance_mask.numpy() if hasattr(guidance_mask, "numpy") else np.asarray(guidance_mask)
    
    # 取指定 batch 样本
    target_mask = one_hot_np[sample_idx, ..., 1]  # (H, W)，取正类通道
    bbox_mask = guidance_np[sample_idx, ...]          # (H, W)
    
    # 验证数据（调试信息）
    print(f"    [Mask Debug] target_mask shape: {target_mask.shape}, min: {target_mask.min():.4f}, max: {target_mask.max():.4f}, sum: {target_mask.sum():.4f}")
    print(f"    [Mask Debug] bbox_mask shape: {bbox_mask.shape}, min: {bbox_mask.min():.4f}, max: {bbox_mask.max():.4f}, sum: {bbox_mask.sum():.4f}")
    
    # 创建坐标（标准 181x360 网格）
    height, width = target_mask.shape
    lats = np.linspace(-90, 90, height)
    lons = np.linspace(0, 360, width)
    
    # 处理经度循环（0-360）
    target_mask_cyclic, lons_cyclic = add_cyclic_point(target_mask, coord=lons)
    bbox_mask_cyclic, _ = add_cyclic_point(bbox_mask, coord=lons)
    
    # 创建图形：1行2列，使用 central_longitude=180 来正确处理 0-360 经度范围
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        subplot_kw={'projection': projection}
    )
    
    def setup_ax(ax):
        """设置单个子图的地图背景"""
        ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
    
    # 左图：Target Storm One-hot Mask
    setup_ax(ax1)
    target_masked = np.ma.masked_where(target_mask_cyclic < 0.01, target_mask_cyclic)
    im1 = ax1.contourf(
        lons_cyclic, lats, target_masked,  # 使用 1D 数组，contourf 会自动广播
        levels=np.linspace(0.01, 1.0, 10),
        cmap="Reds", alpha=0.85,
        transform=ccrs.PlateCarree()
    )
    ax1.contour(
        lons_cyclic, lats, target_mask_cyclic,
        levels=[0.1, 0.5, 0.9],
        colors="red", linewidths=1.5,
        transform=ccrs.PlateCarree()
    )
    title1 = "Target Storm One-hot Mask"
    if target_storm_info:
        lat_val = target_storm_info.get('lat', 'N/A')
        lon_val = target_storm_info.get('lon', 'N/A')
        if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)):
            title1 += f"\n(lat={lat_val:.1f}°, lon={lon_val:.1f}°)"
    ax1.set_title(title1, fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)
    
    # 右图：Guidance Bounding Box Mask
    setup_ax(ax2)
    # 绘制 bounding box mask（蓝色区域）
    im2 = ax2.contourf(
        lons_cyclic, lats, bbox_mask_cyclic,  # 使用 1D 数组
        levels=[0, 0.5, 1.0],
        colors=["white", "blue"],
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    # 叠加显示 target mask 的轮廓（红色虚线）
    ax2.contour(
        lons_cyclic, lats, target_mask_cyclic,
        levels=[0.5],
        colors="red", linewidths=2,
        transform=ccrs.PlateCarree(),
        linestyles="--"
    )
    title2 = "Guidance Bounding Box Mask"
    if selected_storm_info:
        storm_id = selected_storm_info.get('storm_id', 'N/A')
        title2 += f"\n(Selected Storm ID: {storm_id})"
    ax2.set_title(title2, fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)
    
    # 总标题
    suptitle = "Mask Comparison: Target Storm vs Guidance Region"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [Mask Visualization] Saved to {output_path}")


# =========================
# Baseline 缓存管理（磁盘缓存）
# =========================
def _get_baseline_cache_path(
    cache_dir: str,
    want_time: str,
    manual_targets: List[Dict],
    readout_collect_steps_for_vis: Tuple[int, ...],
) -> str:
    """生成 baseline 缓存文件路径
    
    Args:
        cache_dir: 缓存目录
        want_time: 时间点字符串
        manual_targets: 目标位置列表
        readout_collect_steps_for_vis: readout 收集的步骤
    
    Returns:
        缓存文件路径
    """
    time_slug = _short_time_label(want_time)
    targets_slug = _targets_slug(manual_targets)
    steps_slug = "-".join(str(s) for s in sorted(readout_collect_steps_for_vis))
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"baseline_{time_slug}__{targets_slug}__steps{steps_slug}.pkl"
    return os.path.join(cache_dir, cache_file)


def _load_baseline_cache(cache_path: str) -> Optional[Tuple[xr.Dataset, Dict[str, xr.Dataset], xr.Dataset]]:
    """从磁盘加载 baseline 缓存
    
    Returns:
        (preds_A, readouts_A, gt_A) 如果成功加载，否则 None
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        print(f"    [Baseline Cache] Loading from {cache_path}...")
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        print(f"    [Baseline Cache] Successfully loaded")
        return result
    except Exception as e:
        print(f"    [Baseline Cache] Failed to load: {e}")
        return None


def _save_baseline_cache(
    cache_path: str,
    preds_A: xr.Dataset,
    readouts_A: Dict[str, xr.Dataset],
    gt_A: xr.Dataset,
):
    """保存 baseline 结果到磁盘
    
    Args:
        cache_path: 缓存文件路径
        preds_A: baseline 预测结果
        readouts_A: baseline readout 结果
        gt_A: ground truth
    """
    try:
        print(f"    [Baseline Cache] Saving to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump((preds_A, readouts_A, gt_A), f)
        print(f"    [Baseline Cache] Successfully saved")
    except Exception as e:
        print(f"    [Baseline Cache] Failed to save: {e}")


# =========================
# Rollout专用工具函数
# =========================
def load_forcings_for_rollout(
    full_ds: xr.Dataset,
    start_time: str,
    num_steps: int,
    task_config,
) -> xr.Dataset:
    """
    [ROLLOUT ONLY] 从完整ERA5 dataset中加载rollout所需的所有forcings
    
    Rollout需要多步的forcings数据（如TOA incident solar radiation等外部输入）。
    此函数从原始dataset中提取指定时间范围的forcings。
    
    Args:
        full_ds: 原始ERA5 dataset (xr.Dataset)
        start_time: 开始时间字符串，如 "2017-09-07 00:00:00"
        num_steps: rollout步数（例如：10步 = 5天，每步12h）
        task_config: 任务配置（包含forcing_variables）
    
    Returns:
        forcings_extended: (batch=1, time=num_steps, lat, lon, ...)
                          时间坐标为timedelta: [12h, 24h, ..., num_steps*12h]
    
    Example:
        >>> forcings = load_forcings_for_rollout(
        ...     ctx.eval_ds.ds, "2017-09-07 00:00:00", 10, ctx.task_config
        ... )
        >>> forcings.dims  # {'batch': 1, 'time': 10, 'lat': 181, 'lon': 360, ...}
    """
    from graphcast import data_utils
    
    start_ts = pd.Timestamp(start_time)
    
    # 构建需要的时间点列表（每12小时一个）
    time_points = [
        start_ts + pd.Timedelta(hours=12 * (i + 1))
        for i in range(num_steps)
    ]
    
    print(f"    [Rollout Forcings] Loading {num_steps} time steps from dataset...")
    print(f"    [Rollout Forcings] Time range: {time_points[0]} to {time_points[-1]}")
    
    # 从dataset中选择这些时间点
    try:
        forcings_window = full_ds.sel(time=time_points).load()
    except KeyError as e:
        ds_min = pd.Timestamp(full_ds.time.min().values)
        ds_max = pd.Timestamp(full_ds.time.max().values)
        raise ValueError(
            f"部分时间点不在dataset范围内。\n"
            f"需要: {time_points[0]} 到 {time_points[-1]}\n"
            f"Dataset范围: {ds_min} 到 {ds_max}\n"
            f"原始错误: {e}"
        )
    
    # 重命名坐标（从原始ERA5格式转换为GenCast格式）
    # 原始dataset使用 longitude/latitude，但add_derived_vars需要lon/lat
    rename_dict = {}
    if "longitude" in forcings_window.coords or "longitude" in forcings_window.dims:
        rename_dict["longitude"] = "lon"
    if "latitude" in forcings_window.coords or "latitude" in forcings_window.dims:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        forcings_window = forcings_window.rename(rename_dict)
    
    # 下采样到1度分辨率（原始ERA5是0.25度，需要每4个点采样一次）
    # 这与Dataset的transform_sample保持一致
    if "lat" in forcings_window.dims and "lon" in forcings_window.dims:
        lat_size = forcings_window.sizes.get("lat", 0)
        lon_size = forcings_window.sizes.get("lon", 0)
        # 如果维度大于1度分辨率（181x360），则进行下采样
        if lat_size > 181 or lon_size > 360:
            print(f"    [Rollout Forcings] Downsampling from {lat_size}x{lon_size} to 1° resolution...")
            forcings_window = forcings_window.isel(
                lat=slice(0, None, 4),  # 每4个点采样一次
                lon=slice(0, None, 4)
            )
            # 反转lat维度（从南到北 → 从北到南）
            forcings_window = forcings_window.isel(lat=slice(None, None, -1))
            print(f"    [Rollout Forcings] Downsampled to {forcings_window.sizes.get('lat', 0)}x{forcings_window.sizes.get('lon', 0)}")
    
    # 添加batch维度（add_derived_vars需要batch维度）
    if "batch" not in forcings_window.dims:
        forcings_window = forcings_window.expand_dims("batch", axis=0)
    
    # 添加datetime坐标（add_derived_vars需要datetime坐标）
    if "datetime" not in forcings_window.coords:
        # 从time坐标创建datetime坐标
        time_values = forcings_window.coords["time"].values
        if isinstance(time_values[0], np.datetime64):
            datetime_values = time_values
        else:
            # 如果time是timedelta，需要转换为绝对时间
            datetime_values = np.array([start_ts + pd.Timedelta(hours=12 * (i + 1)) 
                                       for i in range(num_steps)], dtype="datetime64[ns]")
        
        # 创建datetime坐标（需要与time维度匹配）
        if "batch" in forcings_window.dims:
            datetime_coord = (("batch", "time"), datetime_values[np.newaxis, :])
        else:
            datetime_coord = ("time", datetime_values)
        forcings_window = forcings_window.assign_coords(datetime=datetime_coord)
    
    # 添加derived variables（year_progress_sin, day_progress_sin等）
    # 这些变量需要从datetime计算，不能直接从dataset读取
    if set(task_config.forcing_variables) & data_utils._DERIVED_VARS:
        data_utils.add_derived_vars(forcings_window)
    
    # 添加TISR（如果需要）
    if "toa_incident_solar_radiation" in task_config.forcing_variables:
        data_utils.add_tisr_var(forcings_window)
    
    # 只提取forcing变量（如tisr, year_progress_sin等）
    forcing_vars = task_config.forcing_variables
    forcings_only = forcings_window[list(forcing_vars)]
    
    # 转换time坐标为timedelta（相对于start_time）
    time_deltas = [np.timedelta64(12 * (i + 1), 'h') for i in range(num_steps)]
    forcings_only = forcings_only.assign_coords(time=time_deltas)
    
    # 删除datetime坐标（rollout中不需要）
    if "datetime" in forcings_only.coords:
        forcings_only = forcings_only.drop_vars("datetime", errors="ignore")
    
    print(f"    [Rollout Forcings] Loaded successfully: {forcings_only.dims}")
    
    return forcings_only


# =========================
# Direct Intensity Scaling: 局部变量缩放
# =========================
def apply_local_intensity_scale(
    preds: xr.Dataset,
    scale_factor: float,
    center_lat: float,
    center_lon: float,
    radius: float,
    variables_to_scale: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    对预测结果在指定圆形区域的指定变量进行局部相对缩放
    
    这个函数实现"想法1：直接缩放输入的Weather State"，通过局部放大/缩小
    特定区域的气象变量值，测试模型对输入强度的响应。
    
    使用相对缩放（相对于区域平均值）：
    - 计算圆形区域内的平均值作为基准
    - 新值 = 基准值 + (原值 - 基准值) × scale_factor
    - 这样保持物理合理性，不会产生极端值
    
    Args:
        preds: 预测结果 (batch, time, lat, lon, ...) 或 (batch, time, lat, lon, level, ...)
        scale_factor: 缩放系数（例如 1.2 表示增强20%的偏差，0.8表示减弱20%的偏差）
        center_lat: 圆形区域中心纬度
        center_lon: 圆形区域中心经度
        radius: 圆形区域半径（度）
        variables_to_scale: 要缩放的变量列表，如果为None则缩放所有变量
    
    Returns:
        scaled_preds: 缩放后的预测结果
    
    Example:
        >>> scaled = apply_local_intensity_scale(
        ...     preds, scale_factor=1.2, 
        ...     center_lat=18.0, center_lon=293.0, radius=5.0,
        ...     variables_to_scale=['u_component_of_wind', 'v_component_of_wind']
        ... )
    """
    scaled_preds = preds.copy(deep=True)  # 深拷贝，避免修改原始数据
    
    # 获取lat和lon坐标
    lats = preds.coords['lat'].values
    lons = preds.coords['lon'].values
    
    # 创建网格
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # 计算距离（简单的欧氏距离，适用于小范围）
    distance = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    
    # 创建圆形mask
    circle_mask = distance <= radius
    
    print(f"    [Direct Intensity] Circle region: center=({center_lat:.1f}°, {center_lon:.1f}°), "
          f"radius={radius:.1f}°, pixels={circle_mask.sum()}")
    
    # 确定要缩放的变量
    if variables_to_scale is None:
        vars_to_process = list(preds.data_vars)
    else:
        vars_to_process = [v for v in variables_to_scale if v in preds.data_vars]
    
    # 对每个变量进行相对缩放（使用 xarray 的维度安全方法）
    scaled_count = 0
    for var_name in vars_to_process:
        var_data = scaled_preds[var_name]
        
        # 只处理有 lat 和 lon 维度的变量
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            # 创建一个 xarray DataArray 包装 mask，让 xarray 自动处理维度对齐
            mask_da = xr.DataArray(
                circle_mask,
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons}
            )
            
            # 相对缩放：计算圆形区域内的平均值作为基准
            # 使用 where 来只计算区域内的值，然后取平均
            region_data = var_data.where(mask_da)
            region_mean = region_data.mean()
            
            # 相对缩放公式：新值 = 基准值 + (原值 - 基准值) × scale_factor
            # 这样：
            # - 如果原值 = 基准值，缩放后还是基准值（无变化）
            # - 如果原值 > 基准值，缩放后会增强偏差（放大）
            # - 如果原值 < 基准值，缩放后会减弱偏差（缩小）
            scaled_value = region_mean + (var_data - region_mean) * scale_factor
            
            # 只在圆形区域内应用缩放，区域外保持原值
            scaled_preds[var_name] = xr.where(mask_da, scaled_value, var_data)
            scaled_count += 1
            
            # 打印调试信息（只打印第一个变量的信息，避免输出过多）
            if scaled_count == 1:
                # 提取标量值（region_mean 已经是标量，因为 mean() 减少了所有维度）
                try:
                    region_mean_scalar = float(region_mean.values.item())
                except (AttributeError, ValueError):
                    region_mean_scalar = float(region_mean.values)
                print(f"    [Direct Intensity] Relative scaling: region_mean={region_mean_scalar:.2f}, "
                      f"scale_factor={scale_factor:.2f}")
    
    print(f"    [Direct Intensity] Scaled {scaled_count} variable(s) using relative scaling (factor {scale_factor:.2f})")
    
    return scaled_preds


# =========================
# Spatial Shift: 区域空间位移
# =========================
def apply_local_spatial_shift(
    preds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    radius: float,
    delta_lat: float,
    delta_lon: float,
    variables_to_shift: Optional[List[str]] = None,
    interpolation_method: str = "linear",
) -> xr.Dataset:
    """
    对预测结果在指定圆形区域进行空间位移
    
    这个函数实现"想法2：区域空间位移"，将指定圆形区域的气象数据"搬移"到
    另一个位置，目标位置被覆盖，源位置的"洞"使用周围数据插值填充。
    
    操作流程：
    1. 定义源圆形区域（center, radius）
    2. 计算目标区域位置（center + delta）
    3. 提取源区域数据并移动到目标位置
    4. 使用插值填充源位置的"洞"
    
    Args:
        preds: 预测结果 (batch, time, lat, lon, ...) 或 (batch, time, lat, lon, level, ...)
        center_lat: 源圆形区域中心纬度
        center_lon: 源圆形区域中心经度
        radius: 圆形区域半径（度）
        delta_lat: 纬度方向位移量（度，正值向北）
        delta_lon: 经度方向位移量（度，正值向东）
        variables_to_shift: 要位移的变量列表，如果为None则位移所有变量
        interpolation_method: 插值方法 ("linear", "nearest", "cubic")
    
    Returns:
        shifted_preds: 位移后的预测结果
    
    Example:
        >>> shifted = apply_local_spatial_shift(
        ...     preds, 
        ...     center_lat=18.0, center_lon=293.0, radius=5.0,
        ...     delta_lat=2.0, delta_lon=3.0,  # 向北2度，向东3度
        ...     variables_to_shift=['u_component_of_wind', 'v_component_of_wind']
        ... )
    """
    from scipy.interpolate import griddata
    
    shifted_preds = preds.copy(deep=True)  # 深拷贝，避免修改原始数据
    
    # 获取lat和lon坐标
    lats = preds.coords['lat'].values
    lons = preds.coords['lon'].values
    
    # 创建网格
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # 计算源区域和目标区域的mask
    distance_from_source = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    source_mask = distance_from_source <= radius
    
    target_center_lat = center_lat + delta_lat
    target_center_lon = center_lon + delta_lon
    distance_from_target = np.sqrt((lat_grid - target_center_lat)**2 + (lon_grid - target_center_lon)**2)
    target_mask = distance_from_target <= radius
    
    print(f"    [Spatial Shift] Source region: center=({center_lat:.1f}°, {center_lon:.1f}°), "
          f"radius={radius:.1f}°, pixels={source_mask.sum()}")
    print(f"    [Spatial Shift] Target region: center=({target_center_lat:.1f}°, {target_center_lon:.1f}°), "
          f"delta=({delta_lat:.1f}°, {delta_lon:.1f}°), pixels={target_mask.sum()}")
    
    # 确定要位移的变量
    if variables_to_shift is None:
        vars_to_process = list(preds.data_vars)
    else:
        vars_to_process = [v for v in variables_to_shift if v in preds.data_vars]
    
    # 对每个变量进行位移操作
    shifted_count = 0
    for var_name in vars_to_process:
        var_data = shifted_preds[var_name]
        
        # 只处理有 lat 和 lon 维度的变量
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            # 获取变量的所有维度
            dims = var_data.dims
            
            # 直接获取 numpy 数组并处理
            var_values = var_data.values
            
            # 处理不同维度的情况
            # 可能是 (batch, time, lat, lon) 或 (batch, time, lat, lon, level) 或 (batch, time, level, lat, lon)
            if 'level' in dims:
                # 有level维度，需要对每个level分别处理
                # 找到各个维度的位置
                level_axis = dims.index('level')
                lat_axis = dims.index('lat')
                lon_axis = dims.index('lon')
                num_levels = var_data.sizes['level']
                
                print(f"    [Spatial Shift] Processing variable '{var_name}' with level dimension")
                print(f"      Original shape: {var_values.shape}, dims: {dims}")
                print(f"      level_axis={level_axis}, lat_axis={lat_axis}, lon_axis={lon_axis}")
                
                # 对每个level分别处理
                for level_idx in range(num_levels):
                    # 构建索引切片
                    idx = [slice(None)] * var_values.ndim
                    idx[level_axis] = level_idx
                    
                    # 提取这个 level 的数据
                    var_slice = var_values[tuple(idx)]
                    print(f"      Level {level_idx}: extracted shape = {var_slice.shape}")
                    
                    # 计算提取level后，lat和lon在新数组中的位置
                    # 如果level在lat之前，lat的索引减1
                    lat_pos_in_slice = lat_axis if lat_axis < level_axis else lat_axis - 1
                    lon_pos_in_slice = lon_axis if lon_axis < level_axis else lon_axis - 1
                    
                    print(f"      After level extraction: lat at axis {lat_pos_in_slice}, lon at axis {lon_pos_in_slice}")
                    
                    # 使用 moveaxis 将 lat 和 lon 移到最后两个维度
                    # 目标：(..., lat, lon) 格式
                    var_slice_moved = np.moveaxis(var_slice, [lat_pos_in_slice, lon_pos_in_slice], [-2, -1])
                    print(f"      After moveaxis: {var_slice_moved.shape}")
                    
                    # 处理这个 level 的数据
                    shifted_slice = _shift_2d_field(
                        var_slice_moved,
                        source_mask,
                        target_mask,
                        lats,
                        lons,
                        interpolation_method
                    )
                    print(f"      After shift: {shifted_slice.shape}")
                    
                    # 将 lat 和 lon 移回原来的位置
                    shifted_slice_moved_back = np.moveaxis(shifted_slice, [-2, -1], [lat_pos_in_slice, lon_pos_in_slice])
                    print(f"      After moveaxis back: {shifted_slice_moved_back.shape}")
                    
                    # 将结果写回
                    var_values[tuple(idx)] = shifted_slice_moved_back
                
                # 一次性更新整个变量
                shifted_preds[var_name] = (dims, var_values)
            else:
                # 没有level维度，直接处理
                print(f"    [Spatial Shift] Processing variable '{var_name}' without level dimension")
                print(f"      Original shape: {var_values.shape}, dims: {dims}")
                
                # 找到 lat 和 lon 的位置
                lat_axis = dims.index('lat')
                lon_axis = dims.index('lon')
                print(f"      lat_axis={lat_axis}, lon_axis={lon_axis}")
                
                # 使用 moveaxis 将 lat 和 lon 移到最后两个维度
                var_values_moved = np.moveaxis(var_values, [lat_axis, lon_axis], [-2, -1])
                print(f"      After moveaxis: {var_values_moved.shape}")
                
                # 处理数据
                shifted_values = _shift_2d_field(
                    var_values_moved,
                    source_mask,
                    target_mask,
                    lats,
                    lons,
                    interpolation_method
                )
                print(f"      After shift: {shifted_values.shape}")
                
                # 将 lat 和 lon 移回原来的位置
                shifted_values_moved_back = np.moveaxis(shifted_values, [-2, -1], [lat_axis, lon_axis])
                print(f"      After moveaxis back: {shifted_values_moved_back.shape}")
                
                shifted_preds[var_name] = (dims, shifted_values_moved_back)
            
            shifted_count += 1
    
    print(f"    [Spatial Shift] Shifted {shifted_count} variable(s) using {interpolation_method} interpolation")
    
    return shifted_preds


def _shift_2d_field(
    field_2d: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """
    对单个2D场进行空间位移（辅助函数）
    
    处理流程：
    1. 保存源区域的数据
    2. 将源区域数据复制到目标区域（覆盖）
    3. 使用插值填充源区域的"洞"
    
    Args:
        field_2d: 2D数据场，shape可能是 (lat, lon) 或 (batch, time, lat, lon)
        source_mask: 源区域mask (lat, lon)
        target_mask: 目标区域mask (lat, lon)
        lats: 纬度坐标
        lons: 经度坐标
        interpolation_method: 插值方法
    
    Returns:
        shifted_field: 位移后的2D场
    """
    from scipy.interpolate import griddata
    
    # 处理不同维度的情况
    original_shape = field_2d.shape
    
    # 如果是4D (batch, time, lat, lon)，需要对每个batch和time分别处理
    if len(original_shape) == 4:
        batch_size, time_size = original_shape[0], original_shape[1]
        result = np.zeros_like(field_2d)
        
        for b in range(batch_size):
            for t in range(time_size):
                result[b, t, :, :] = _shift_single_2d(
                    field_2d[b, t, :, :],
                    source_mask,
                    target_mask,
                    lats,
                    lons,
                    interpolation_method
                )
        return result
    
    # 如果是3D (time, lat, lon) 或 (batch, lat, lon)
    elif len(original_shape) == 3:
        first_dim = original_shape[0]
        result = np.zeros_like(field_2d)
        
        for i in range(first_dim):
            result[i, :, :] = _shift_single_2d(
                field_2d[i, :, :],
                source_mask,
                target_mask,
                lats,
                lons,
                interpolation_method
            )
        return result
    
    # 如果是2D (lat, lon)
    elif len(original_shape) == 2:
        return _shift_single_2d(
            field_2d,
            source_mask,
            target_mask,
            lats,
            lons,
            interpolation_method
        )
    
    else:
        raise ValueError(f"Unsupported field shape: {original_shape}")


def _shift_single_2d(
    field: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    interpolation_method: str = "linear",
) -> np.ndarray:
    """
    对单个2D场 (lat, lon) 进行空间位移的核心实现
    
    使用点对点插值的方法：
    1. 从源区域的网格点提取数据
    2. 将这些数据插值到目标区域的网格点
    3. 对源区域（现在是空的）使用周围数据插值填充
    
    Args:
        field: (lat, lon) 的2D场，或可以squeeze成2D的更高维数组
        source_mask: 源区域mask (lat, lon)
        target_mask: 目标区域mask (lat, lon)
        lats: 纬度坐标
        lons: 经度坐标
        interpolation_method: 插值方法
    
    Returns:
        shifted: 位移后的2D场，形状与输入field相同
    """
    from scipy.interpolate import griddata
    
    # 保存原始形状
    original_shape = field.shape
    
    # 确保 field 是 2D 的
    # 策略：使用 numpy.squeeze 移除所有大小为1的维度，然后检查是否为 (lat, lon)
    field_squeezed = np.squeeze(field)
    
    # 检查 squeeze 后的形状
    if field_squeezed.ndim == 2:
        # 检查是否是 (lat, lon) 的正确大小
        if field_squeezed.shape[0] == len(lats) and field_squeezed.shape[1] == len(lons):
            field_2d = field_squeezed
        else:
            raise ValueError(f"Field 2D shape {field_squeezed.shape} doesn't match expected (lat={len(lats)}, lon={len(lons)})")
    elif field_squeezed.ndim > 2:
        # squeeze 后仍然大于 2D，说明有多个非1的维度
        # 尝试提取最后两个维度（假设是 lat, lon）
        if field_squeezed.shape[-2] == len(lats) and field_squeezed.shape[-1] == len(lons):
            # 如果只有3个维度且第一个是batch/time，提取第一个
            if field_squeezed.ndim == 3 and field_squeezed.shape[0] == 1:
                field_2d = field_squeezed[0]
            else:
                raise ValueError(f"Cannot reduce field to 2D. Shape after squeeze: {field_squeezed.shape} (original: {original_shape})")
        else:
            raise ValueError(f"Field dimensions don't match lat/lon. Shape after squeeze: {field_squeezed.shape}, expected last two dims: ({len(lats)}, {len(lons)})")
    elif field_squeezed.ndim < 2:
        raise ValueError(f"Field must be at least 2D after squeeze, got shape {field_squeezed.shape} (original: {original_shape})")
    else:
        field_2d = field_squeezed
    
    # 最终验证
    if field_2d.shape != (len(lats), len(lons)):
        raise ValueError(f"Final field_2d shape {field_2d.shape} doesn't match expected ({len(lats)}, {len(lons)}). Original: {original_shape}")
    
    # 确保是 C-contiguous 的 numpy 数组（避免 view 问题）
    field_2d = np.ascontiguousarray(field_2d)
    
    # 创建网格
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # 步骤1：创建结果数组（先复制原始数据）
    shifted = field_2d.copy()
    
    # 步骤2：将数据从源区域"搬移"到目标区域
    # 策略：对于目标区域的每个点，计算它应该从field的哪个"源位置"获取数据
    # 然后从原始field_2d在该源位置插值取值
    
    # 计算位移量（通过源区域和目标区域的中心点差异）
    source_center_lat = lat_grid[source_mask].mean()
    source_center_lon = lon_grid[source_mask].mean()
    target_center_lat = lat_grid[target_mask].mean()
    target_center_lon = lon_grid[target_mask].mean()
    
    delta_lat = target_center_lat - source_center_lat
    delta_lon = target_center_lon - source_center_lon
    
    # 获取目标区域的点坐标
    target_lat_coords = lat_grid[target_mask]
    target_lon_coords = lon_grid[target_mask]
    
    # 对于目标区域的每个点，计算它的"源位置"（反向映射）
    # 如果一个点在目标位置是(lat_t, lon_t)，它的数据应该来自(lat_t - delta_lat, lon_t - delta_lon)
    source_lat_for_target = target_lat_coords - delta_lat
    source_lon_for_target = target_lon_coords - delta_lon
    
    # 从原始field_2d在这些源位置插值
    # 使用整个field作为插值源（不只是source_mask的点）
    all_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    all_values = field_2d.ravel()
    
    query_points = np.column_stack([source_lat_for_target, source_lon_for_target])
    
    try:
        if interpolation_method == "nearest":
            method = "nearest"
        elif interpolation_method == "cubic":
            method = "cubic"
        else:  # linear
            method = "linear"
        
        # 从原始field插值获取目标区域的值
        target_values = griddata(
            all_points,
            all_values,
            query_points,
            method=method,
            fill_value=np.nan
        )
        
        # 如果有NaN，使用最近邻作为fallback
        if np.any(np.isnan(target_values)):
            nan_mask = np.isnan(target_values)
            target_values[nan_mask] = griddata(
                all_points,
                all_values,
                query_points[nan_mask],
                method="nearest"
            )
        
        # 将插值结果写入目标区域
        shifted[target_mask] = target_values
        
    except Exception as e:
        print(f"    [Spatial Shift] Warning: Target interpolation failed ({e}), using field mean")
        shifted[target_mask] = field_2d.mean()
    
    # 步骤3：填充源区域的"洞"（使用周围数据插值）
    # 关键：只填充"源区域中不与目标重叠的部分"，保留已经移动过去的数据
    # 创建插值所需的点集（排除源区域和目标区域的重叠部分）
    # 我们希望从"未被改动"的区域插值
    unchanged_mask = ~(source_mask | target_mask)  # 既不是源也不是目标
    
    # 定义需要填充的"洞"：源区域中不与目标重叠的部分
    source_hole_mask = source_mask & ~target_mask  # 源区域 - 重叠部分 = 真正的洞
    
    # 如果有需要填充的洞，并且有足够的未改动区域用于插值
    if source_hole_mask.sum() > 0 and unchanged_mask.sum() > 0:
        known_points = np.column_stack([
            lat_grid[unchanged_mask].ravel(),
            lon_grid[unchanged_mask].ravel()
        ])
        known_values = field_2d[unchanged_mask].ravel()  # 使用原始值
        
        # 只对"洞"进行插值（不包括与目标重叠的部分）
        source_hole_points = np.column_stack([
            lat_grid[source_hole_mask].ravel(),
            lon_grid[source_hole_mask].ravel()
        ])
        
        try:
            if interpolation_method == "nearest":
                method = "nearest"
            elif interpolation_method == "cubic":
                method = "cubic"
            else:  # linear
                method = "linear"
            
            interpolated_values = griddata(
                known_points,
                known_values,
                source_hole_points,
                method=method,
                fill_value=np.nan
            )
            
            # 如果有NaN，使用最近邻作为fallback
            if np.any(np.isnan(interpolated_values)):
                nan_mask = np.isnan(interpolated_values)
                interpolated_values[nan_mask] = griddata(
                    known_points,
                    known_values,
                    source_hole_points[nan_mask],
                    method="nearest"
                )
            
            # 只填充洞，不覆盖与目标重叠的部分
            shifted[source_hole_mask] = interpolated_values
            
        except Exception as e:
            print(f"    [Spatial Shift] Warning: Source hole filling failed ({e}), using mean fill")
            shifted[source_hole_mask] = known_values.mean()
    elif source_hole_mask.sum() > 0:
        # 如果有洞但没有足够的未改动区域，使用简单填充
        print(f"    [Spatial Shift] Warning: Not enough unchanged area for interpolation, using simple mean fill")
        shifted[source_hole_mask] = field_2d.mean()
    
    # 恢复原始形状（如果需要）
    if original_shape != shifted.shape:
        shifted = shifted.reshape(original_shape)
    
    return shifted


def _get_next_inputs_rollout(
    prev_inputs: xr.Dataset,
    next_frame: xr.Dataset,
) -> xr.Dataset:
    """
    [ROLLOUT ONLY] 更新输入队列：保留最新的num_inputs个时间步
    
    在autoregressive rollout中，每预测一步后需要将预测结果加入输入队列，
    并丢弃最旧的时间步，实现"滑动窗口"效果。
    
    Args:
        prev_inputs: 之前的输入 (batch, time=2, lat, lon, ...)
                    例如: [t-12h, t]
        next_frame: 新预测的frame (batch, time=1, lat, lon, ...)，已merge了forcings
                    例如: [t+12h]
    
    Returns:
        new_inputs: 更新后的输入 (batch, time=2, lat, lon, ...)
                   例如: [t, t+12h]
    
    Implementation:
        类似于队列的push操作：
        [t-12h, t] + [t+12h] → concat → [t-12h, t, t+12h] → tail(2) → [t, t+12h]
    """
    # 找出需要从predictions复制到inputs的变量
    # （排除只在forcings中的变量，如tisr）
    next_inputs_keys = list(
        set(next_frame.keys()).intersection(set(prev_inputs.keys()))
    )
    next_inputs = next_frame[next_inputs_keys]
    
    # 拼接：[prev_inputs, next_inputs] 并保留最后num_inputs个时间步
    num_inputs = prev_inputs.dims["time"]
    new_inputs = xr.concat(
        [prev_inputs, next_inputs],
        dim="time",
        data_vars="different"
    ).tail(time=num_inputs)
    
    return new_inputs


# =========================
# Utils: 索引、拼批、类型转换
# =========================
def detect_want_times(ds: DateMergedERA5TyphoonSizeDataset, want_times: List[str]):
    report = []
    time2idx = build_time2idx_map(ds)
    ds_time_min = pd.Timestamp(ds.ds["time"].min().values)
    for t in want_times:
        ts = pd.Timestamp(t)
        # 用日期（忽略 hour）来匹配，因为 DateMergedERA5TyphoonSizeDataset 使用 merge_same_day
        date_key = ts.normalize()  # 归一化到 00:00:00
        in_tracks = date_key in time2idx
        has_full_window = (ts - np.timedelta64(24, "h") >= ds_time_min)
        idxs = time2idx.get(date_key, [])
        n_samples = len(idxs)
        reason = None
        if not in_tracks:
            reason = "not in tracks (该日期无轨迹或不在 eval 年份范围内)"
        elif not has_full_window:
            reason = "no full window (t-24h 越界)"
        elif n_samples == 0:
            reason = "no sample (tracks 分组为空)"
        report.append({
            "time": str(ts),
            "in_tracks": in_tracks,
            "has_full_window": has_full_window,
            "n_samples": n_samples,
            "ok": in_tracks and has_full_window and n_samples > 0,
            "reason": reason
        })
    return report

def build_time2idx_map(ds: DateMergedERA5TyphoonSizeDataset) -> Dict[pd.Timestamp, List[int]]:
    """按日期（而非精确时刻）构建 time -> index 映射，因为 merge_same_day 后 hour 不固定"""
    time2idx = defaultdict(list)
    for i, r in ds.tracks.reset_index().iterrows():
        # 只用 year/month/day，忽略 hour（归一化到日期）
        ts = pd.Timestamp(int(r["year"]), int(r["month"]), int(r["day"]))
        time2idx[ts].append(i)
    return time2idx

def indices_for_times(
    ds: DateMergedERA5TyphoonSizeDataset,
    times: List[str | np.datetime64 | pd.Timestamp],
    require_full_window: bool = True
) -> List[int]:
    time2idx = build_time2idx_map(ds)
    indices = []
    ds_time_min = pd.Timestamp(ds.ds["time"].min().values)
    for t in times:
        ts = pd.Timestamp(t)
        if require_full_window and (ts - np.timedelta64(24, "h") < ds_time_min):
            continue
        # 用日期（忽略 hour）来匹配
        date_key = ts.normalize()
        hit = time2idx.get(date_key, [])
        if hit:
            indices.append(hit[0])   # 只取第一个 index，避免重复
    return indices

def custom_collate_fn(batch):
    inputs_list, targets_list, forcings_list, one_hot_list, ts_list = zip(*batch)
    inputs = xr.concat(inputs_list, dim="batch")
    targets = xr.concat(targets_list, dim="batch")
    forcings = xr.concat(forcings_list, dim="batch")
    one_hot = torch.cat(one_hot_list, dim=0)
    return inputs, targets, forcings, one_hot, ts_list

def _to_np(a):
    if hasattr(a, "block_until_ready"):
        a.block_until_ready()
    try:
        return np.asarray(jax.device_get(a))
    except Exception:
        return np.asarray(a)

def xr_to_numpy(obj):
    if isinstance(obj, xr.DataArray):
        out = obj.copy(deep=False)
        out.data = _to_np(out.data)
        for c in out.coords:
            try:
                out.coords[c].data = _to_np(out.coords[c].data)
            except Exception:
                pass
        return out
    if isinstance(obj, xr.Dataset):
        out = obj.copy(deep=False)
        for v in list(out.data_vars):
            da = out[v].copy(deep=False)
            da.data = _to_np(da.data)
            for c in da.coords:
                try:
                    da.coords[c].data = _to_np(da.coords[c].data)
                except Exception:
                    pass
            out[v] = da
        for c in list(out.coords):
            try:
                out.coords[c].data = _to_np(out.coords[c].data)
            except Exception:
                pass
        return out
    if isinstance(obj, dict):
        return {k: xr_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [xr_to_numpy(v) for v in obj]
        return type(obj)(t)
    return _to_np(obj)


# =========================
# 轻量命名帮助（SAVE_MODE 路由）
# =========================
def _short_time_label(ts: str) -> str:
    t = pd.Timestamp(ts)
    return f"{t:%Y%m%d_%H}"

def _param_slug(
    readout_collect_steps_for_vis: Tuple[int, ...],
    inner_idxs: List[int],
    inner_steps_map: Dict[int, int],  # 新增：per-step optimization 次数
    max_opt_steps: int,                # 新增：最大次数
    inner_lr: float,
    strength: float,
    random_seed: int = 42,             # 新增：random seed
    guidance_method: str = "direct_optim",  # 新增：guidance方法
    intensity_scaling_configs: Optional[List[Dict]] = None,  # 新增：缩放配置
    shift_configs: Optional[List[Dict]] = None,              # 新增：位移配置
    warp_configs: Optional[Dict] = None,                     # 新增：warp配置
) -> str:
    """
    生成参数标识符字符串（用于文件夹命名）
    
    文件夹命名示例：
    - 基础: idx[4-8-12]__steps[0-15x1]__lr0.005__gsteps[10-15]__strength0.7__seed42
    - +scale: ...seed42__scale[s0x3_s1x2.5]  (step 0放大3x, step 1放大2.5x)
    - +shift: ...seed42__shift[s0d+5_+3_s1d-2_+1]  (step 0向北5度东3度, step 1向南2度东1度)
    - 完整: ...seed42__scale[s0x3]__shift[s0d-3_-2]
    """
    idxs_part = "-".join(str(int(x)) for x in inner_idxs) if inner_idxs else "none"
    gsteps_part = "-".join(str(int(x)) for x in readout_collect_steps_for_vis)
    lr_part = f"{inner_lr:g}"
    
    # 生成 steps_map 的摘要字符串（例如：4-8x40_9-16x1）
    if inner_steps_map:
        # 按 step 排序，找出连续的范围
        sorted_steps = sorted(inner_steps_map.keys())
        ranges = []
        if sorted_steps:
            start = sorted_steps[0]
            prev_step = sorted_steps[0]
            prev_opt = inner_steps_map[prev_step]
            
            for step in sorted_steps[1:]:
                opt = inner_steps_map[step]
                if step == prev_step + 1 and opt == prev_opt:
                    # 连续且相同 opt_steps，继续范围
                    prev_step = step
                else:
                    # 范围结束，记录
                    ranges.append(f"{start}-{prev_step}x{prev_opt}")
                    start = step
                    prev_step = step
                    prev_opt = opt
            # 最后一个范围
            ranges.append(f"{start}-{prev_step}x{prev_opt}")
        steps_map_str = "_".join(ranges)
    else:
        steps_map_str = f"max{max_opt_steps}"
    
    # 构建基础 slug
    if guidance_method == "none":
        # Baseline only - 只需要基础信息
        base_slug = f"method[baseline]__seed{random_seed}"
    elif guidance_method == "input_manipulation":
        # Input Manipulation - 不需要 lr/strength/steps
        base_slug = f"method[input_manip]__seed{random_seed}"
    elif guidance_method == "local_affine":
        # Local Affine - 简化参数（含 steps_map 以支持 sweep）
        base_slug = f"method[affine]__steps[{steps_map_str}]__lr{lr_part}__seed{random_seed}"
    else:
        # direct_optim - 完整参数
        base_slug = f"method[direct_optim]__idx[{idxs_part}]__steps[{steps_map_str}]__lr{lr_part}__gsteps[{gsteps_part}]__strength{strength:g}__seed{random_seed}"
    
    # 添加额外的 scale 和 shift 参数
    extra_parts = []
    
    # Intensity Scaling 参数摘要
    if intensity_scaling_configs:
        scale_summaries = []
        for cfg in intensity_scaling_configs:
            step_idx = cfg.get("step_idx", 0)
            scale_factor = cfg.get("scale_factor", 1.0)
            # 简化格式：s{step_idx}x{scale_factor}
            scale_summaries.append(f"s{step_idx}x{scale_factor:g}")
        extra_parts.append("scale[" + "_".join(scale_summaries) + "]")
    
    # Spatial Shift 参数摘要
    if shift_configs:
        shift_summaries = []
        for cfg in shift_configs:
            step_idx = cfg.get("step_idx", 0)
            delta_lat = cfg.get("delta_lat", 0.0)
            delta_lon = cfg.get("delta_lon", 0.0)
            # 简化格式：s{step_idx}d{delta_lat:+g}_{delta_lon:+g}
            # 使用 :+g 会保留符号（+/-）
            shift_summaries.append(f"s{step_idx}d{delta_lat:+g}_{delta_lon:+g}")
        extra_parts.append("shift[" + "_".join(shift_summaries) + "]")
    
    # Warp 参数摘要
    if warp_configs and warp_configs.get("enabled", False):
        # 提取参数
        center_lat = warp_configs.get("center_lat", 0.0)
        center_lon = warp_configs.get("center_lon", 0.0)
        radius = warp_configs.get("radius", 0.0)
        warp_lr = warp_configs.get("learning_rate", 1e-2)
        reg_weight = warp_configs.get("regularization_weight", 1e-3)
        
        # 构建摘要：warp[c{lat}_{lon}_r{radius}_trans_rot_scale_lr{lr}_reg{reg}]
        warp_summary = f"warp[c{center_lat:.1f}_{center_lon:.1f}_r{radius:g}"
        
        # 添加 optimize 标志
        opt_parts = []
        if warp_configs.get("optimize_translation", False):
            opt_parts.append("trans")
        if warp_configs.get("optimize_rotation", False):
            opt_parts.append("rot")
        if warp_configs.get("optimize_scale", False):
            opt_parts.append("scale")
        if opt_parts:
            warp_summary += "_" + "_".join(opt_parts)
        
        # 添加学习率和正则化权重（如果与默认值不同，或总是显示）
        warp_summary += f"_lr{warp_lr:g}_reg{reg_weight:g}"
        warp_summary += "]"
        extra_parts.append(warp_summary)
    
    if extra_parts:
        return base_slug + "__" + "__".join(extra_parts)
    else:
        return base_slug

def decide_output_dirs(
    *,
    output_dir: str,
    save_mode: str,
    want_time: str,
    manual_targets: List[Dict],
    readout_collect_steps_for_vis: Tuple[int, ...],
    inner_idxs: List[int],
    inner_steps_map: Dict[int, int],  # 新增：per-step optimization 次数
    max_opt_steps: int,                # 新增：最大次数
    inner_lr: float,
    strength: float,
    guidance_mode: str = "steer_one",  # 新增：guidance 模式
    random_seed: int = 42,             # 新增：random seed
    guidance_method: str = "direct_optim",  # 新增：guidance方法
    intensity_scaling_configs: Optional[List[Dict]] = None,  # 新增：缩放配置
    shift_configs: Optional[List[Dict]] = None,              # 新增：位移配置
    warp_configs: Optional[Dict] = None,                     # 新增：warp配置
) -> Dict[str, str]:
    """
    根据 SAVE_MODE 决定输出目录（适配 manual guidance 模式）
    seed 放在更高层级的文件夹名称中，这样不同的 seed 不会共享 baseline
    """
    time_slug = _short_time_label(want_time)
    targets_slug = _targets_slug(manual_targets)
    # param_slug 不再包含 seed（seed 已经在 root 中）
    param_slug_no_seed = _param_slug(
        readout_collect_steps_for_vis, inner_idxs, inner_steps_map, max_opt_steps, inner_lr, strength, 
        random_seed=42,  # 使用固定的42，仅用于生成B目录名
        guidance_method=guidance_method,
        intensity_scaling_configs=intensity_scaling_configs,
        shift_configs=shift_configs,
        warp_configs=warp_configs,
    )
    # 实际的 param_slug 仍然包含 seed（用于返回值）
    param_slug = _param_slug(
        readout_collect_steps_for_vis, inner_idxs, inner_steps_map, max_opt_steps, inner_lr, strength, 
        random_seed,
        guidance_method=guidance_method,
        intensity_scaling_configs=intensity_scaling_configs,
        shift_configs=shift_configs,
        warp_configs=warp_configs,
    )

    if save_mode == "by_dates":
        # 新结构：time 提升一级，相同 time 的所有实验共享 A_no_guidance
        # seed 放在 root 路径中，time 作为第二级目录
        root = os.path.join(output_dir, f"seed{random_seed}", f"time_{time_slug}")
        # A_no_guidance 在 time 目录下，所有 target 共享
        out_A = os.path.join(root, "A_no_guidance")
        # B 目录在 target 子目录下，不同参数配置通过 param_slug 区分
        target_dir = os.path.join(root, f"target_{targets_slug}__mode_{guidance_mode}")
        out_B = os.path.join(target_dir, f"B_{param_slug_no_seed}")
    elif save_mode == "by_params":
        root = os.path.join(output_dir, f"seed{random_seed}", f"params_{param_slug_no_seed}", f"time_{time_slug}__target_{targets_slug}__mode_{guidance_mode}")
        out_A = os.path.join(root, "A_no_guidance")
        out_B = os.path.join(root, "B_guided_manual")
    else:
        raise ValueError(f"Unknown SAVE_MODE: {save_mode}")

    return {"root": root, "out_A": out_A, "out_B": out_B, "param_slug": param_slug, "time_slug": time_slug, "targets_slug": targets_slug}


# =========================
# IO helpers
# =========================
def get_gcs_bucket(bucket_name: str, prefix: str = "gencast/"):
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket(bucket_name)
    return gcs_bucket, prefix

def open_era5_zarr(url: str, years: tuple[int, int], step_12h: bool = True) -> xr.Dataset:
    ds = xr.open_zarr(url, chunks="auto", decode_times=True)
    ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    if step_12h:
        ds = ds.isel(time=slice(0, None, 2))
    return ds

def build_datasets(
    ds_train: xr.Dataset,
    ds_eval: xr.Dataset,
    task_config,
    tracks_folder: str,
    years_train: tuple[int, int],
    years_eval: tuple[int, int],
):
    eval_dataset  = DateMergedERA5TyphoonSizeDataset(ds_eval,  task_config, tracks_folder, years_eval[0],  years_eval[1])
    return None, eval_dataset

def build_wanted_subloader(
    eval_dataset: DateMergedERA5TyphoonSizeDataset,
    want_times: List[str],
    batch_size: int = 1
):
    idxs = indices_for_times(eval_dataset, want_times, require_full_window=True)
    if not idxs:
        raise ValueError("给定的日期一个也没匹配到（可能越界或 tracks 中无此时间）。")
    subset = Subset(eval_dataset, idxs)
    subloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=custom_collate_fn
    )
    return subloader


# =========================
# 模型构建 / JIT
# =========================
def load_checkpoint_and_stats(gcs_bucket, dir_prefix: str):
    params_file_options = [
        name for blob in gcs_bucket.list_blobs(prefix=dir_prefix + "params/")
        if (name := blob.name.removeprefix(dir_prefix + "params/"))
    ]
    model_name = next(f for f in params_file_options if "Mini" in f)
    print("Loading model:", model_name)

    with gcs_bucket.blob(dir_prefix + "params/" + f"{model_name}").open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    with gcs_bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/min_by_level.nc").open("rb") as f:
        min_by_level = xr.load_dataset(f).compute()

    return ckpt, (diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level)

def make_predictor_wrapped(
    task_config,
    sampler_config,
    noise_config,
    noise_encoder_config,
    denoiser_architecture_config,
    diffs_stddev_by_level,
    mean_by_level,
    stddev_by_level,
    min_by_level,
):
    def construct_wrapped_gencast():
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config
        )
        predictor = normalization.InputsAndResiduals(
            predictor=predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
            ReadOut_flag=True
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
            ReadOut_flag=True
        )
        return predictor
    return construct_wrapped_gencast

def build_jitted_functions(params, state, construct_wrapped_gencast):
    @hk.transform_with_state
    def readout_guided_inference_fn(inputs, targets_template, forcings, guidance_cfg):
        predictor = construct_wrapped_gencast()
        return predictor.readout_guided_inference_vis(inputs, targets_template, forcings, guidance_cfg=guidance_cfg)

    readout_inference_wGuide_fn_jitted = jax.jit(
        lambda rng, inputs, targets_template, forcings, guidance_cfg:
        readout_guided_inference_fn.apply(params, state, rng, inputs, targets_template, forcings, guidance_cfg)[0]
    )
    readout_inference_wGuide_fn_pmap = xarray_jax.pmap(readout_inference_wGuide_fn_jitted, dim="sample")
    return readout_guided_inference_fn, readout_inference_wGuide_fn_pmap


# =========================
# 推理（strength=0 无引导；>0 引导）
# =========================
def _run_once_with_guidance(
    *,
    params,
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    eval_forcings: xr.Dataset,
    readout_guided_inference_fn,
    target_readout_one_hot: Optional[torch.Tensor],
    readout_collect_steps_for_vis: Tuple[int, ...],
    guidance_strength: float,
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],  # 新增：per-step optimization 次数
    guide_max_opt_steps: int,                    # 新增：最大次数（用于 JAX 编译）
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,
    guidance_mask: Optional[torch.Tensor] = None,  # 新增：选择性 mask (B, H, W)
    random_seed: int = 42,  # 新增：random seed
):
    rng = jax.random.PRNGKey(random_seed)
    state_g = {}

    inner_idxs = [] if guidance_strength == 0.0 else list(guide_inner_opt_step_idxs)
    inner_steps_map = {} if guidance_strength == 0.0 else dict(guide_inner_opt_steps_map)
    max_opt_steps = 1 if guidance_strength == 0.0 else int(guide_max_opt_steps)

    guidance_cfg = {
        "steps": tuple(readout_collect_steps_for_vis),
        "strength": float(guidance_strength),
        "normalize_grad": bool(guide_normalize_grad),
        "eps": float(guide_eps),
        "inner_opt_step_idxs": inner_idxs,
        "inner_opt_steps_map": inner_steps_map,  # 新增
        "max_opt_steps": max_opt_steps,           # 新增
        "inner_opt_lr": float(guide_inner_opt_lr),
        "loss_type": str(guide_loss_type),
        "target_readout": None,
        "guidance_mask": None,  # 新增
        "warp_configs": warp_configs,  # ✅ 添加这一行
    }
    if target_readout_one_hot is not None:
        one_hot_np = target_readout_one_hot.numpy() if hasattr(target_readout_one_hot, "numpy") else np.asarray(target_readout_one_hot)
        guidance_cfg["target_readout"] = jnp.asarray(one_hot_np)
    
    # 新增：传递 guidance_mask
    if guidance_mask is not None:
        mask_np = guidance_mask.numpy() if hasattr(guidance_mask, "numpy") else np.asarray(guidance_mask)
        guidance_cfg["guidance_mask"] = jnp.asarray(mask_np)

    targets_tmpl = eval_targets * np.nan
    (preds, readouts, loss_history), _ = readout_guided_inference_fn.apply(
        params, state_g, rng, eval_inputs, targets_tmpl, eval_forcings, guidance_cfg
    )
    preds_host = xr_to_numpy(preds)
    readouts_host = {k: xr_to_numpy(v) for k, v in readouts.items()}
    gt_host = xr_to_numpy(eval_targets.isel(time=0, batch=0))
    
    # 立即释放 GPU 上的原始数据（在转换为 numpy 后）
    del preds, readouts
    
    # 转换 loss_history 为 numpy 并提取有效值
    loss_history_np = _to_np(loss_history)  # shape (20, max_opt_steps)
    del loss_history  # 释放原始 loss_history
    loss_history_dict = {}
    for step in guide_inner_opt_step_idxs:
        step_losses = loss_history_np[step, :]
        # 根据 guide_inner_opt_steps_map 获取该 step 的有效 optimization 次数
        valid_count = guide_inner_opt_steps_map.get(step, max_opt_steps)
        # 只取前 valid_count 个值（这些才是真正执行的优化步）
        valid_losses = step_losses[:valid_count]
        # 过滤掉可能的 -1.0（未触发 guidance 的 step）
        valid_losses = valid_losses[valid_losses >= 0]
        if len(valid_losses) > 0:
            loss_history_dict[int(step)] = valid_losses.tolist()
    del loss_history_np  # 释放 loss_history_np
    
    return preds_host, readouts_host, gt_host, loss_history_dict


def _run_rollout_with_guidance(
    *,
    params,
    eval_inputs: xr.Dataset,
    eval_targets_template: xr.Dataset,
    forcings_extended: xr.Dataset,
    readout_guided_inference_fn,
    num_steps: int,
    guidance_on_first_step: bool,
    
    # Guidance Method（新增：模式选择）
    guidance_method: str = "direct_optim",  # "none" | "direct_optim" | "input_manipulation" | "local_affine"
    
    # Guidance相关参数（仅 direct_optim 和 local_affine 使用）
    target_readout_one_hot: Optional[torch.Tensor],
    guidance_mask: Optional[torch.Tensor],
    readout_collect_steps_for_vis: Tuple[int, ...],
    guidance_strength: float,
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],
    guide_max_opt_steps: int,
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,
    timing_logger: Optional[TimingLogger] = None,  # 新增：时间记录器
    random_seed: int = 42,  # 新增：random seed
    
    # Input Manipulation 参数（仅 input_manipulation 使用）
    intensity_scaling_configs: Optional[List[Dict]] = None,  # 缩放配置列表
    shift_configs: Optional[List[Dict]] = None,  # 位移配置列表
    
    # Local Affine 参数（仅 local_affine 使用）
    warp_configs: Optional[Dict] = None,  # warp 配置
) -> Tuple[List[xr.Dataset], List[Dict[str, xr.Dataset]], Dict[int, List[float]]]:
    """
    [ROLLOUT ONLY] 执行multi-step autoregressive rollout，可选在第一步使用guidance
    
    这是rollout的核心函数，实现手动循环的多步预测：
    - 每步使用前2个时间步作为输入，预测下一个12h
    - 预测结果被加入输入队列，丢弃最旧的时间步（滑动窗口）
    - 第一步可选使用guidance（selective storm guidance）
    - 后续步骤使用无guidance的标准推理
    
    Args:
        params: 模型参数
        eval_inputs: 初始输入 (batch, time=2, lat, lon, ...)，如 [t-12h, t]
        eval_targets_template: 单步target模板 (batch, time=1, ...)，用于构建每步的template
        forcings_extended: 扩展的forcings (batch, time=num_steps, ...)
        readout_guided_inference_fn: 推理函数（haiku transform）
        num_steps: rollout总步数（例如：10步 = 5天）
        guidance_on_first_step: 是否在第一步使用guidance
        
        target_readout_one_hot: guidance目标mask (B, H, W, 2)
        guidance_mask: guidance区域mask (B, H, W)，用于局部loss计算
        readout_collect_steps_for_vis: 收集readout的denoising steps
        guidance_strength: guidance强度（0.0 = 无guidance）
        guide_inner_opt_step_idxs: 在哪些denoising steps做优化
        guide_inner_opt_steps_map: 每个step的优化次数映射
        guide_max_opt_steps: 最大优化次数（用于JAX编译）
        guide_inner_opt_lr: 优化学习率
        guide_loss_type: loss类型（"readout_l2" / "xt_l2"）
        guide_normalize_grad: 是否归一化梯度
        guide_eps: 数值稳定性epsilon
    
    Returns:
        predictions_list: 每一步的预测结果 List[xr.Dataset]，长度为num_steps
        readouts_list: 每一步的readout结果 List[Dict[str, xr.Dataset]]
        loss_history: 第一步的guidance loss history（如果使用guidance）
    
    Example:
        >>> preds_list, readouts_list, loss_hist = _run_rollout_with_guidance(
        ...     params=ctx.params,
        ...     eval_inputs=eval_inputs,  # [t-12h, t]
        ...     num_steps=10,
        ...     guidance_on_first_step=True,
        ...     ...
        ... )
        >>> len(preds_list)  # 10
        >>> preds_list[0].dims  # {'batch': 1, 'time': 1, ...}  # t+12h
        >>> preds_list[1].dims  # {'batch': 1, 'time': 1, ...}  # t+24h
    """
    
    # 初始化
    current_inputs = eval_inputs  # (batch, time=2, lat, lon, ...)
    predictions_list = []
    readouts_list = []
    loss_history_all = {}
    
    print(f"\n[Rollout] Starting {num_steps}-step autoregressive rollout...")
    print(f"[Rollout] Guidance on first step: {guidance_on_first_step}")
    print(f"[Rollout] Initial inputs time coords: {current_inputs.coords['time'].values}")
    
    for step_idx in range(num_steps):
        step_label = f"Step {step_idx + 1}/{num_steps}"
        step_name = f"Rollout step {step_idx}"
        if step_idx == 0 and guidance_on_first_step:
            step_name += " (with guidance)"
        
        print(f"\n  --- {step_label} ---")
        
        # 记录每个step的时间
        step_timing_started = False
        if timing_logger:
            try:
                timing_logger.start(step_name)
                step_timing_started = True
            except Exception as e:
                print(f"  [TimingLogger Warning] Failed to start timing for {step_name}: {e}")
        
        # 提取当前步的forcings
        current_forcings = forcings_extended.isel(time=slice(step_idx, step_idx + 1))
        
        # 决定是否使用guidance（仅第一步）
        use_guidance_this_step = (step_idx == 0) and guidance_on_first_step
        
        # 根据 guidance_method 确定是否激活guidance
        if guidance_method == "none":
            # Baseline only, 强制不使用guidance
            use_guidance_this_step = False
            current_strength = 0.0
        elif guidance_method == "input_manipulation":
            # Input Manipulation 不在 denoising 过程中使用 guidance
            use_guidance_this_step = False
            current_strength = 0.0
        else:
            # direct_optim 或 local_affine 使用 guidance
            current_strength = guidance_strength if use_guidance_this_step else 0.0
        
        if use_guidance_this_step:
            print(f"  [Rollout] Using GUIDANCE (strength={current_strength}, method={guidance_method})")
        else:
            if guidance_method == "input_manipulation":
                print(f"  [Rollout] Standard inference (input_manipulation mode)")
            elif guidance_method == "none":
                print(f"  [Rollout] Standard inference (baseline only)")
            else:
                print(f"  [Rollout] Standard inference (no guidance)")
        
        # 构建当前步的targets_template（单步，填充NaN）
        current_targets_template = eval_targets_template * np.nan
        # 更新time坐标为当前步的timedelta
        time_coord = np.timedelta64(12 * (step_idx + 1), 'h')
        current_targets_template = current_targets_template.assign_coords(time=[time_coord])
        
        # 构建guidance配置
        rng = jax.random.PRNGKey(random_seed + step_idx)  # 每步不同的seed
        state_g = {}
        
        inner_idxs = [] if not use_guidance_this_step else list(guide_inner_opt_step_idxs)
        inner_steps_map = {} if not use_guidance_this_step else dict(guide_inner_opt_steps_map)
        max_opt_steps = 1 if not use_guidance_this_step else int(guide_max_opt_steps)
        
        # 当不使用guidance时，必须使用"xt_l2"而不是"readout_l2"（因为readout_l2需要target_readout）
        current_loss_type = guide_loss_type if use_guidance_this_step else "xt_l2"
        
        guidance_cfg = {
            "steps": tuple(readout_collect_steps_for_vis),
            "strength": float(current_strength),
            "normalize_grad": bool(guide_normalize_grad),
            "eps": float(guide_eps),
            "inner_opt_step_idxs": inner_idxs,
            "inner_opt_steps_map": inner_steps_map,
            "max_opt_steps": max_opt_steps,
            "inner_opt_lr": float(guide_inner_opt_lr),
            "loss_type": str(current_loss_type),  # 使用current_loss_type而不是guide_loss_type
            "target_readout": None,
            "guidance_mask": None,
            "warp_configs": warp_configs if use_guidance_this_step else None,  # 新增：local_affine warp配置
        }
        
        if use_guidance_this_step and target_readout_one_hot is not None:
            one_hot_np = target_readout_one_hot.numpy() if hasattr(target_readout_one_hot, "numpy") else np.asarray(target_readout_one_hot)
            guidance_cfg["target_readout"] = jnp.asarray(one_hot_np)
            
            if guidance_mask is not None:
                mask_np = guidance_mask.numpy() if hasattr(guidance_mask, "numpy") else np.asarray(guidance_mask)
                guidance_cfg["guidance_mask"] = jnp.asarray(mask_np)
        
        # 调用模型推理
        (preds, readouts, loss_history), _ = readout_guided_inference_fn.apply(
            params, state_g, rng,
            current_inputs, current_targets_template, current_forcings, guidance_cfg
        )
        
        # 转换为host（从GPU/TPU移到CPU）
        preds_host = xr_to_numpy(preds)
        readouts_host = {k: xr_to_numpy(v) for k, v in readouts.items()}
        
        # 立即释放 GPU 上的原始数据（在转换为 numpy 后）
        del preds, readouts
        
        # ===== [Direct Intensity Scaling] 在指定步骤进行局部缩放 =====
        if intensity_scaling_configs is not None and len(intensity_scaling_configs) > 0:
            # 查找当前step的配置
            step_config = None
            for config in intensity_scaling_configs:
                if config.get("step_idx") == step_idx:
                    step_config = config
                    break
            
            if step_config is not None:
                print(f"  [Direct Intensity] Step {step_idx}: Applying local intensity scaling")
                preds_host = apply_local_intensity_scale(
                    preds_host,
                    scale_factor=step_config["scale_factor"],
                    center_lat=step_config["center_lat"],
                    center_lon=step_config["center_lon"],
                    radius=step_config["radius"],
                    variables_to_scale=step_config.get("variables", None),
                )
                print(f"  [Direct Intensity] ✓ Scaling applied successfully")
        
        # ===== [Spatial Shift] 在指定步骤进行区域空间位移 =====
        if shift_configs is not None and len(shift_configs) > 0:
            # 查找当前step的配置
            step_config = None
            for config in shift_configs:
                if config.get("step_idx") == step_idx:
                    step_config = config
                    break
            
            if step_config is not None:
                print(f"  [Spatial Shift] Step {step_idx}: Applying local spatial shift")
                preds_host = apply_local_spatial_shift(
                    preds_host,
                    center_lat=step_config["center_lat"],
                    center_lon=step_config["center_lon"],
                    radius=step_config["radius"],
                    delta_lat=step_config["delta_lat"],
                    delta_lon=step_config["delta_lon"],
                    variables_to_shift=step_config.get("variables", None),
                    interpolation_method=step_config.get("interpolation_method", "linear"),
                )
                print(f"  [Spatial Shift] ✓ Shift applied successfully")
        
        # 保存结果
        predictions_list.append(preds_host)
        readouts_list.append(readouts_host)
        
        # 保存loss history（仅第一步有guidance时）
        if use_guidance_this_step and guide_inner_opt_step_idxs:
            loss_history_np = _to_np(loss_history)  # shape (20, max_opt_steps)
            del loss_history  # 释放原始 loss_history（在转换为 numpy 后）
            loss_history_dict = {}
            for step in guide_inner_opt_step_idxs:
                step_losses = loss_history_np[step, :]
                valid_count = guide_inner_opt_steps_map.get(step, max_opt_steps)
                valid_losses = step_losses[:valid_count]
                valid_losses = valid_losses[valid_losses >= 0]
                if len(valid_losses) > 0:
                    loss_history_dict[int(step)] = valid_losses.tolist()
            loss_history_all = loss_history_dict
            print(f"  [Rollout] Guidance loss recorded for {len(loss_history_dict)} denoising steps")
            del loss_history_np  # 释放 loss_history_np
        elif not use_guidance_this_step:
            # 如果没有使用 guidance，立即释放 loss_history
            del loss_history
        
        # 构建下一步的输入（关键的rollout逻辑！）
        if step_idx < num_steps - 1:
            # 合并预测和forcings（next_frame包含所有下一步需要的变量）
            next_frame = xr.merge([preds_host, current_forcings])
            
            # 更新输入队列：[t-12h, t] → [t, t+12h]
            current_inputs = _get_next_inputs_rollout(current_inputs, next_frame)
            
            # 释放不再需要的中间变量
            del next_frame
            
            print(f"  [Rollout] Updated inputs for next step")
            print(f"            New input time coords: {current_inputs.coords['time'].values}")
        else:
            print(f"  [Rollout] Last step, no input update needed")
        
        # 每几个 step 进行一次轻量级内存清理（避免过于频繁）
        if step_idx > 0 and step_idx % 3 == 0:
            gc.collect()
        
        # 结束step的时间记录
        if timing_logger and step_timing_started:
            try:
                timing_logger.end(step_name)
            except Exception as e:
                print(f"  [TimingLogger Warning] Failed to end timing for {step_name}: {e}")
    
    print(f"\n[Rollout] ✓ Completed {num_steps}-step rollout successfully")
    print(f"[Rollout] Total predictions: {len(predictions_list)}")
    print(f"[Rollout] Total readouts: {len(readouts_list)}")
    
    # 最终内存清理：释放不再需要的中间变量
    del current_inputs, current_targets_template, current_forcings
    if 'guidance_cfg' in locals():
        del guidance_cfg
    gc.collect()
    
    return predictions_list, readouts_list, loss_history_all


def _visualize_and_export(
    *,
    gt_list: List[xr.Dataset],
    predictions_list: List[xr.Dataset],
    readouts_list: List[Dict[str, xr.Dataset]],
    one_hot_torch: torch.Tensor,
    out_dir: str,
    epoch: int = 0,
    visual_vars: Optional[List[str]] = None,
    step_prefix: Optional[str] = "step00_",  # 添加步骤前缀参数，默认为step00_
    timing_logger: Optional[TimingLogger] = None,  # 新增：时间记录器
    vis_n_procs: int = 1,  # 新增：可视化参数
    vis_contourf_levels: int = 36,
    vis_coastlines_resolution: str = "50m",
    vis_draw_gridlabels: bool = True,
    vis_add_borders: bool = True,
    vis_dpi: int = 240,
):
    """
    可视化单步或多步预测结果
    
    Args:
        step_prefix: 文件名前缀，默认为"step00_"（单步预测）
        timing_logger: 可选的时间记录器
    """
    from graphcast.vis import generate_comparison_gifs_parallel_with_prefix
    
    os.makedirs(out_dir, exist_ok=True)
    gt_frames = [g for g in gt_list]
    forecast_frames = [p.isel(time=0, batch=0) for p in predictions_list]
    readout_frames = {
        key: [ro[key].isel(time=0, batch=0) for ro in readouts_list]
        for key in readouts_list[0].keys()
    }
    # 使用传入的 visual_vars；如果为 None，则画全部变量
    all_vars = list(gt_frames[0].data_vars.keys())
    if visual_vars is not None:
        variable_list = [v for v in all_vars if v in visual_vars]
    else:
        variable_list = all_vars
    variable_list_wLevel = [v for v in variable_list if 'level' in gt_frames[0][v].dims]
    level_dict = {v: 500 for v in variable_list_wLevel}

    generate_comparison_gifs_parallel_with_prefix(
        gt_frames, forecast_frames, readout_frames,
        variables=variable_list,
        output_dir=out_dir,
        epoch=epoch,
        level_dict=level_dict,
        duration=500,
        n_procs=vis_n_procs,
        step_prefix=step_prefix,  # 传递步骤前缀
        contourf_levels=vis_contourf_levels,
        coastlines_resolution=vis_coastlines_resolution,
        draw_gridlabels=vis_draw_gridlabels,
        add_borders=vis_add_borders,
        dpi=vis_dpi,
        timing_logger=timing_logger,  # 传递时间记录器
    )

    visualize_onehot_readout_simple(
        readout_frames_dict=readout_frames,
        one_hot=one_hot_torch,
        output_dir=out_dir,
        epoch=epoch,
        ts=str(gt_frames[0].datetime.values) if 'datetime' in gt_frames[0].coords else "unknown-ts",
        mark=None,
        steps=list(readout_frames.keys()),  # 使用实际的 readout steps
    )


# =========================
# Rollout 数据保存：保存为 NetCDF 供后续画图
# =========================
def _save_rollout_data_netcdf(
    *,
    predictions_list: List[xr.Dataset],
    readouts_list: List[Dict[str, xr.Dataset]],
    gt_list: List[xr.Dataset],
    one_hot_torch: torch.Tensor,
    out_dir: str,
    metadata: Optional[Dict] = None,
    save_readout_steps: Optional[List[int]] = None,  # 新增：指定哪些步骤需要保存readout（例如[0]表示只保存第一步）
    shift_configs: Optional[List[Dict]] = None,  # 新增：shift配置，用于保存可视化信息
) -> str:
    """
    保存 rollout 数据为 NetCDF 格式（高效压缩，保留所有元数据）
    
    文件结构：
        rollout_data/
            predictions_step00.nc
            predictions_step01.nc
            ...
            readouts_step00_denoising10.nc  # 只保存有guidance的步骤
            readouts_step00_denoising15.nc
            ...
            gt_step00.nc
            gt_step01.nc
            ...
            one_hot.npy
            metadata.json
    
    Args:
        predictions_list: 每一步的预测结果
        readouts_list: 每一步的readout结果
        gt_list: 每一步的GT数据
        one_hot_torch: one-hot mask
        out_dir: 输出目录
        metadata: 额外的元数据（如 epoch, visual_vars 等）
        save_readout_steps: 需要保存readout的步骤索引列表（例如[0]表示只保存第一步）。
                           如果为None，保存所有步骤；如果为空列表[]，不保存任何readout。
    
    Returns:
        data_dir: 保存的数据目录路径
    """
    import json
    
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "rollout_data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n[Saving Rollout Data] Saving to {data_dir}...")
    t0 = time.time()
    
    # 保存 predictions
    print(f"  [Save] Saving {len(predictions_list)} predictions...")
    for idx, pred in enumerate(predictions_list):
        pred_path = os.path.join(data_dir, f"predictions_step{idx:02d}.nc")
        pred.to_netcdf(
            pred_path,
            engine='h5netcdf',
            encoding={var: {'zlib': True, 'complevel': 5} for var in pred.data_vars}
        )
    
    # 保存 readouts（只保存指定的步骤）
    if len(readouts_list) > 0:
        if save_readout_steps is None:
            # 如果未指定，保存所有步骤（向后兼容）
            steps_to_save = list(range(len(readouts_list)))
            print(f"  [Save] Saving readouts for all {len(readouts_list)} steps (save_readout_steps=None)...")
        elif len(save_readout_steps) == 0:
            # 如果为空列表，不保存readout
            steps_to_save = []
            print(f"  [Save] Skipping readout saving (save_readout_steps=[])...")
        else:
            # 只保存指定的步骤
            steps_to_save = save_readout_steps
            print(f"  [Save] Saving readouts only for steps {steps_to_save} (with guidance)...")
        
        saved_count = 0
        for rollout_idx, readouts in enumerate(readouts_list):
            if rollout_idx in steps_to_save:
                for step_name, readout_ds in readouts.items():
                    readout_path = os.path.join(
                        data_dir, 
                        f"readouts_step{rollout_idx:02d}_denoising{step_name}.nc"
                    )
                    readout_ds.to_netcdf(
                        readout_path,
                        engine='h5netcdf',
                        encoding={var: {'zlib': True, 'complevel': 5} for var in readout_ds.data_vars}
                    )
                    saved_count += 1
        print(f"  [Save] Saved {saved_count} readout files for {len(steps_to_save)} step(s)")
    
    # 保存 GT
    print(f"  [Save] Saving {len(gt_list)} ground truth frames...")
    for idx, gt in enumerate(gt_list):
        gt_path = os.path.join(data_dir, f"gt_step{idx:02d}.nc")
        gt.to_netcdf(
            gt_path,
            engine='h5netcdf',
            encoding={var: {'zlib': True, 'complevel': 5} for var in gt.data_vars}
        )
    
    # 保存 one_hot（numpy 格式，因为简单）
    one_hot_np = one_hot_torch.numpy() if hasattr(one_hot_torch, "numpy") else np.asarray(one_hot_torch)
    one_hot_path = os.path.join(data_dir, "one_hot.npy")
    np.save(one_hot_path, one_hot_np)
    
    # 保存元数据（JSON）
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata_serializable = {}
    if metadata:
        for k, v in metadata.items():
            if k == "loss_history" and isinstance(v, dict):
                # loss_history 已经是可序列化的
                metadata_serializable[k] = v
            elif isinstance(v, (list, tuple)):
                metadata_serializable[k] = list(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                metadata_serializable[k] = v
            else:
                metadata_serializable[k] = str(v)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    # 保存 shift regions 信息（用于可视化）
    if shift_configs and len(shift_configs) > 0:
        shift_regions_path = os.path.join(data_dir, "shift_regions.json")
        shift_regions_data = []
        for cfg in shift_configs:
            shift_regions_data.append({
                "step_idx": cfg.get("step_idx", 0),
                "center_lat": cfg.get("center_lat", 0.0),
                "center_lon": cfg.get("center_lon", 0.0),
                "radius": cfg.get("radius", 5.0),
                "delta_lat": cfg.get("delta_lat", 0.0),
                "delta_lon": cfg.get("delta_lon", 0.0),
                # 计算目标中心位置
                "target_lat": cfg.get("center_lat", 0.0) + cfg.get("delta_lat", 0.0),
                "target_lon": cfg.get("center_lon", 0.0) + cfg.get("delta_lon", 0.0),
            })
        with open(shift_regions_path, 'w') as f:
            json.dump(shift_regions_data, f, indent=2)
        print(f"  [Save] Saved shift regions info for visualization: {shift_regions_path}")
    
    # 计算总大小
    total_size = sum(
        os.path.getsize(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
    ) / (1024 * 1024)
    
    print(f"    -> Saved in {time.time() - t0:.2f}s, total size: {total_size:.2f}MB")
    print(f"    -> Data directory: {data_dir}")
    return data_dir


def _visualize_rollout_steps(
    *,
    predictions_list: List[xr.Dataset],
    readouts_list: List[Dict[str, xr.Dataset]],
    gt_list: List[xr.Dataset],  # 直接接受已加载的 GT 数据
    one_hot_torch: torch.Tensor,
    out_dir: str,
    epoch: int = 0,
    visual_vars: Optional[List[str]] = None,
    timing_logger: Optional[TimingLogger] = None,
    vis_n_procs: int = 1,
    vis_contourf_levels: int = 36,
    vis_coastlines_resolution: str = "50m",
    vis_draw_gridlabels: bool = True,
    vis_add_borders: bool = True,
    vis_dpi: int = 240,
):
    """
    [ROLLOUT ONLY] 可视化rollout的所有步骤，为每一步添加前缀
    
    Args:
        predictions_list: 每一步的预测结果
        readouts_list: 每一步的readout结果
        gt_list: 每一步的GT数据（已预先加载）
        one_hot_torch: one-hot mask
        out_dir: 输出目录
        epoch: epoch编号
        visual_vars: 要可视化的变量列表
        timing_logger: 时间记录器
        vis_n_procs: 可视化并行进程数
        vis_contourf_levels: contourf levels
        vis_coastlines_resolution: coastlines resolution
        vis_draw_gridlabels: 是否绘制grid labels
        vis_add_borders: 是否添加borders
        vis_dpi: DPI设置
    """
    from graphcast.vis import generate_comparison_gifs_parallel_with_prefix
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[Rollout Visualization] Using pre-loaded GT data for {len(gt_list)} steps...")
    
    # 准备forecast_frames和readout_frames
    forecast_frames = [p.isel(time=0, batch=0) for p in predictions_list]
    
    readout_frames = {}
    if len(readouts_list) > 0:
        all_readout_steps = list(readouts_list[0].keys())
        for step in all_readout_steps:
            readout_frames[step] = [
                readouts_list[rollout_idx][step].isel(time=0, batch=0) 
                for rollout_idx in range(len(readouts_list))
            ]
    
    # 使用传入的 visual_vars
    if len(gt_list) > 0 and not all(np.isnan(g.to_array().values).all() for g in gt_list):
        all_vars = list(gt_list[0].data_vars.keys())
    else:
        all_vars = list(forecast_frames[0].data_vars.keys())
    
    if visual_vars is not None:
        variable_list = [v for v in all_vars if v in visual_vars]
    else:
        variable_list = all_vars
    
    variable_list_wLevel = [v for v in variable_list if 'level' in forecast_frames[0][v].dims]
    level_dict = {v: 500 for v in variable_list_wLevel}
    
    # 生成步骤前缀列表：["step00_", "step01_", ..., "step09_"]
    num_steps = len(predictions_list)  # 从 predictions_list 获取步数
    step_prefix_list = [f"step{idx:02d}_" for idx in range(num_steps)]
    
    print(f"\n[Rollout Visualization] Generating visualizations for {num_steps} steps...")
    if timing_logger:
        timing_logger.start("PNG generation")
    
    generate_comparison_gifs_parallel_with_prefix(
        gt_list, forecast_frames, readout_frames,
        variables=variable_list,
        output_dir=out_dir,
        epoch=epoch,
        level_dict=level_dict,
        duration=500,
        n_procs=vis_n_procs,
        step_prefix=step_prefix_list,  # 传递步骤前缀列表
        contourf_levels=vis_contourf_levels,
        coastlines_resolution=vis_coastlines_resolution,
        draw_gridlabels=vis_draw_gridlabels,
        add_borders=vis_add_borders,
        dpi=vis_dpi,
        timing_logger=timing_logger,  # 传递时间记录器
    )
    
    if readout_frames:
        # 从 gt_list 或 predictions_list 获取时间戳（用于文件命名）
        if len(gt_list) > 0 and 'datetime' in gt_list[0].coords:
            ts_str = str(gt_list[0].datetime.values)
        elif len(predictions_list) > 0 and 'datetime' in predictions_list[0].coords:
            ts_str = str(predictions_list[0].datetime.values)
        else:
            ts_str = "rollout"
        
        visualize_onehot_readout_simple(
            readout_frames_dict=readout_frames,
            one_hot=one_hot_torch,
            output_dir=out_dir,
            epoch=epoch,
            ts=ts_str,
            mark=None,
            steps=list(readout_frames.keys()),
        )
    
    if timing_logger:
        timing_logger.end("PNG generation")


# =========================
# Rollout Video Generation: PNG拼接和Video生成
# =========================
def check_single_rollout_completed(
    out_dir: str,
    num_rollout_steps: int,
    min_steps_required: int = 3,
) -> bool:
    """
    检查单个目录的rollout数据是否已完成
    
    Args:
        out_dir: 输出目录（A_no_guidance 或 B_guided）
        num_rollout_steps: rollout总步数
        min_steps_required: 至少需要多少个step的NetCDF文件才算完成
    
    Returns:
        is_completed: 是否已完成
    """
    if not os.path.isdir(out_dir):
        print(f"    [Check Single] Directory does not exist: {out_dir}")
        return False
    
    data_dir = os.path.join(out_dir, "rollout_data")
    if not os.path.isdir(data_dir):
        print(f"    [Check Single] rollout_data directory does not exist: {data_dir}")
        return False
    
    # 检查必需的NetCDF文件
    required_files = []
    for step_idx in range(num_rollout_steps):
        pred_file = os.path.join(data_dir, f"predictions_step{step_idx:02d}.nc")
        gt_file = os.path.join(data_dir, f"gt_step{step_idx:02d}.nc")
        required_files.extend([pred_file, gt_file])
    
    # 统计存在的文件
    found_files = sum(1 for f in required_files if os.path.isfile(f))
    total_required = len(required_files)
    min_files_required = min_steps_required * 2  # 每个step需要predictions和gt两个文件
    
    is_completed = found_files >= min_files_required
    
    print(f"    [Check Single] {out_dir}:")
    print(f"      Found {found_files}/{total_required} files (need {min_files_required} for {min_steps_required} steps)")
    print(f"      Completed: {is_completed}")
    
    return is_completed


def check_rollout_completed(
    out_A: str,
    out_B: str,
    num_rollout_steps: int,
    visual_vars: Optional[List[str]] = None,
    min_steps_required: int = 3,  # 至少需要多少个step的NetCDF文件才算完成
) -> Tuple[bool, List[str]]:
    """
    检测rollout是否已完成（通过检查NetCDF数据文件）
    
    这是 DataOnly 版本，检查 rollout_data/ 目录下的 .nc 文件：
    - predictions_step00.nc, predictions_step01.nc, ...
    - gt_step00.nc, gt_step01.nc, ...
    - metadata.json
    
    Args:
        out_A: baseline输出目录
        out_B: guided输出目录
        num_rollout_steps: rollout总步数
        visual_vars: 要检查的变量列表（如果None，检查所有变量，但这里主要用于兼容性）
        min_steps_required: 至少需要多少个step的NetCDF文件才算完成
    
    Returns:
        (is_completed, found_variables): 
            is_completed: 是否已完成
            found_variables: 找到的变量列表（这里返回空列表，因为DataOnly版本不按变量检查）
    """
    # 检查目录是否存在
    has_A = os.path.isdir(out_A)
    has_B = os.path.isdir(out_B)
    
    print(f"    [Check] out_A exists: {has_A}, out_B exists: {has_B}")
    
    if not has_A:
        print(f"    [Check] Baseline directory not found: {out_A}")
        return False, []
    
    if not has_B:
        print(f"    [Check] Guided directory not found: {out_B}")
        # 如果B不存在，检查A是否完成（可能baseline已完成但guided未运行）
        # 这种情况下不应该跳过，因为guided还没运行
        return False, []
    
    # 检查 rollout_data 目录
    data_dir_A = os.path.join(out_A, "rollout_data")
    data_dir_B = os.path.join(out_B, "rollout_data")
    
    has_data_A = os.path.isdir(data_dir_A)
    has_data_B = os.path.isdir(data_dir_B)
    
    print(f"    [Check] rollout_data dirs - A: {has_data_A}, B: {has_data_B}")
    
    if not has_data_A:
        print(f"    [Check] Baseline rollout_data directory not found: {data_dir_A}")
        return False, []
    
    if not has_data_B:
        print(f"    [Check] Guided rollout_data directory not found: {data_dir_B}")
        return False, []
    
    # 检查必需的NetCDF文件
    required_files_A = []
    required_files_B = []
    
    # 检查 predictions 和 gt 文件
    for step_idx in range(num_rollout_steps):
        pred_file_A = os.path.join(data_dir_A, f"predictions_step{step_idx:02d}.nc")
        pred_file_B = os.path.join(data_dir_B, f"predictions_step{step_idx:02d}.nc")
        gt_file_A = os.path.join(data_dir_A, f"gt_step{step_idx:02d}.nc")
        gt_file_B = os.path.join(data_dir_B, f"gt_step{step_idx:02d}.nc")
        
        required_files_A.extend([pred_file_A, gt_file_A])
        required_files_B.extend([pred_file_B, gt_file_B])
    
    # 检查 metadata.json（可选，但如果有会更可靠）
    metadata_A = os.path.join(data_dir_A, "metadata.json")
    metadata_B = os.path.join(data_dir_B, "metadata.json")
    
    # 统计存在的文件
    found_files_A = sum(1 for f in required_files_A if os.path.isfile(f))
    found_files_B = sum(1 for f in required_files_B if os.path.isfile(f))
    
    has_metadata_A = os.path.isfile(metadata_A)
    has_metadata_B = os.path.isfile(metadata_B)
    
    total_required = len(required_files_A)  # A和B应该一样多
    min_files_required = min_steps_required * 2  # 每个step需要predictions和gt两个文件
    
    print(f"    [Check] Found {found_files_A}/{total_required} files in A, {found_files_B}/{total_required} files in B")
    print(f"    [Check] Metadata - A: {has_metadata_A}, B: {has_metadata_B}")
    
    # 判断是否完成：需要A和B都有足够的文件
    is_completed_A = found_files_A >= min_files_required
    is_completed_B = found_files_B >= min_files_required
    
    is_completed = is_completed_A and is_completed_B
    
    if is_completed:
        print(f"    [Check] ✓ Rollout data completed: A has {found_files_A} files, B has {found_files_B} files")
    else:
        if not is_completed_A:
            print(f"    [Check] ✗ Baseline incomplete: {found_files_A}/{total_required} files (need {min_files_required})")
        if not is_completed_B:
            print(f"    [Check] ✗ Guided incomplete: {found_files_B}/{total_required} files (need {min_files_required})")
    
    # 返回空列表作为 found_variables（DataOnly版本不按变量检查）
    return is_completed, []


def create_comparison_video_from_rollout(
    out_A: str,
    out_B: str,
    output_video_dir: str,
    num_rollout_steps: int,
    variables: List[str],
    epoch: int = 0,
    fps: float = 2.0,  # 帧率
    video_format: str = "mp4",
    timing_logger: Optional[TimingLogger] = None,
):
    """
    将rollout的PNG拼接并生成comparison video
    
    对于每个step和每个variable：
    1. 读取baseline和guided的PNG
    2. 上下拼接（baseline在上，guided在下）
    3. 将所有step的拼接frame组合成video
    
    Args:
        out_A: baseline输出目录
        out_B: guided输出目录
        output_video_dir: 视频输出目录
        num_rollout_steps: rollout总步数
        variables: 变量列表
        epoch: epoch编号
        fps: 视频帧率
        video_format: 视频格式（"mp4" 或 "avi"）
    """
    import cv2
    from PIL import Image
    
    os.makedirs(output_video_dir, exist_ok=True)
    
    for var in variables:
        var_dir_A = os.path.join(out_A, var)
        var_dir_B = os.path.join(out_B, var)
        
        if not os.path.isdir(var_dir_A) or not os.path.isdir(var_dir_B):
            print(f"  [Video] Skipping {var}: directory not found")
            continue
        
        # 记录每个变量的视频生成时间
        if timing_logger:
            timing_logger.start(f"Video: {var}")
        
        # 收集所有step的PNG文件
        frames = []
        missing_steps = []
        
        for step_idx in range(num_rollout_steps):
            step_prefix = f"step{step_idx:02d}_"
            pattern = f"{step_prefix}{var}_frame{step_idx}_epoch*.png"
            
            # 使用glob查找文件（可能因为路径中的方括号失败）
            search_path_A = os.path.join(var_dir_A, pattern)
            search_path_B = os.path.join(var_dir_B, pattern)
            files_A = sorted(glob.glob(search_path_A))
            files_B = sorted(glob.glob(search_path_B))
            
            # 如果glob失败，使用os.listdir + 正则表达式作为fallback
            pattern_re = re.compile(rf"^{step_prefix}{re.escape(var)}_frame{step_idx}_epoch\d+\.png$")
            
            if len(files_A) == 0 and os.path.isdir(var_dir_A):
                all_files_A = [f for f in os.listdir(var_dir_A) if f.endswith('.png')]
                matched_files_A = [f for f in all_files_A if pattern_re.match(f)]
                if matched_files_A:
                    files_A = sorted([os.path.join(var_dir_A, f) for f in matched_files_A])
            
            if len(files_B) == 0 and os.path.isdir(var_dir_B):
                all_files_B = [f for f in os.listdir(var_dir_B) if f.endswith('.png')]
                matched_files_B = [f for f in all_files_B if pattern_re.match(f)]
                if matched_files_B:
                    files_B = sorted([os.path.join(var_dir_B, f) for f in matched_files_B])
            
            if len(files_A) == 0 or len(files_B) == 0:
                missing_steps.append(step_idx)
                continue
            
            # 读取两张图片
            try:
                img_A = cv2.imread(files_A[0], cv2.IMREAD_COLOR)  # 明确指定读取为彩色
                img_B = cv2.imread(files_B[0], cv2.IMREAD_COLOR)
                
                if img_A is None or img_B is None:
                    print(f"  [Video] Warning: Failed to read images for {var} step {step_idx}")
                    print(f"    files_A[0]={files_A[0]}, exists={os.path.exists(files_A[0])}")
                    print(f"    files_B[0]={files_B[0]}, exists={os.path.exists(files_B[0])}")
                    missing_steps.append(step_idx)
                    continue
                
                # 验证图片格式
                if len(img_A.shape) != 3 or img_A.shape[2] != 3:
                    print(f"  [Video] Warning: img_A has unexpected shape {img_A.shape} for {var} step {step_idx}")
                    missing_steps.append(step_idx)
                    continue
                if len(img_B.shape) != 3 or img_B.shape[2] != 3:
                    print(f"  [Video] Warning: img_B has unexpected shape {img_B.shape} for {var} step {step_idx}")
                    missing_steps.append(step_idx)
                    continue
                
                # 确保数据类型是uint8
                if img_A.dtype != np.uint8:
                    img_A = (img_A * 255).astype(np.uint8) if img_A.max() <= 1.0 else img_A.astype(np.uint8)
                if img_B.dtype != np.uint8:
                    img_B = (img_B * 255).astype(np.uint8) if img_B.max() <= 1.0 else img_B.astype(np.uint8)
                
                # 确保两张图片宽度相同（如果不同，resize）
                if img_A.shape[1] != img_B.shape[1]:
                    target_width = max(img_A.shape[1], img_B.shape[1])
                    img_A = cv2.resize(img_A, (target_width, img_A.shape[0]), interpolation=cv2.INTER_LINEAR)
                    img_B = cv2.resize(img_B, (target_width, img_B.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # 上下拼接（baseline在上，guided在下）
                combined = np.vstack([img_A, img_B])
                
                # 验证拼接后的frame格式
                if combined.dtype != np.uint8:
                    combined = combined.astype(np.uint8)
                if len(combined.shape) != 3 or combined.shape[2] != 3:
                    print(f"  [Video] Warning: Combined frame has unexpected shape {combined.shape} for {var} step {step_idx}")
                    missing_steps.append(step_idx)
                    continue
                
                frames.append(combined)
                
            except Exception as e:
                print(f"  [Video] Error processing {var} step {step_idx}: {e}")
                missing_steps.append(step_idx)
                continue
        
        if len(frames) == 0:
            print(f"  [Video] No valid frames found for {var}, skipping")
            continue
        
        if missing_steps:
            print(f"  [Video] {var}: Missing steps {missing_steps}, using {len(frames)} frames")
        
        # 获取视频尺寸
        height, width = frames[0].shape[:2]
        print(f"  [Video] {var}: {len(frames)} frames, size={width}x{height}")
        
        # 创建视频写入器
        video_filename = f"{var}_comparison_epoch{epoch}.{video_format}"
        video_path = os.path.join(output_video_dir, video_filename)
        
        # 直接使用PIL创建GIF（最可靠的方法）
        # 因为cv2和ffmpeg在这个系统上都有问题，直接使用PIL
        try:
            from PIL import Image
            
            # 将frames转换为PIL Images
            pil_frames = []
            for idx, frame in enumerate(frames):
                # cv2使用BGR，PIL需要RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 确保数据类型正确
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8) if frame_rgb.max() <= 1.0 else frame_rgb.astype(np.uint8)
                pil_img = Image.fromarray(frame_rgb)
                pil_frames.append(pil_img)
            
            # 保存为GIF（虽然叫video，但GIF是最可靠的动画格式）
            gif_path = video_path.replace('.mp4', '.gif').replace('.avi', '.gif')
            duration_ms = int(1000 / fps)  # 转换为毫秒
            
            if len(pil_frames) > 0:
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration_ms,
                    loop=0
                )
                
                if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                    file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
                    print(f"  [Video] Saved {var} comparison animation: {gif_path} ({len(frames)} frames, {fps} fps, size={file_size_mb:.2f}MB)")
                    if timing_logger:
                        timing_logger.end(f"Video: {var}")
                    continue
                else:
                    print(f"  [Video] Error: GIF file not created for {var}")
            else:
                print(f"  [Video] Error: No frames to save for {var}")
                
        except ImportError:
            print(f"  [Video] Error: PIL not available for {var}")
        except Exception as e:
            print(f"  [Video] Error: PIL/GIF creation failed for {var}: {e}")
        
        # 如果PIL失败，跳过这个变量（不应该发生，因为PIL是标准库）
        print(f"  [Video] Error: Failed to create animation for {var}, skipping...")
        if timing_logger:
            timing_logger.end(f"Video: {var}")
        continue


# =========================
# ERA5 Batch 本地缓存
# =========================

def _era5_cache_time_slug(want_time: str) -> str:
    """把 want_time 字符串转成合法的文件夹名，例如 '2019-08-28 00:00:00' → '20190828_00'"""
    ts = pd.Timestamp(want_time)
    return f"{ts:%Y%m%d_%H}"


def _era5_batch_cache_dir(cache_root: str, want_time: str) -> str:
    return os.path.join(cache_root, _era5_cache_time_slug(want_time))


def load_era5_batch_cache(cache_root: str, want_time: str):
    """
    尝试从缓存加载初始 batch（eval_inputs, eval_targets, eval_forcings, one_hot_original, ts）。
    返回 tuple 或 None（缓存不存在）。
    """
    if cache_root is None:
        return None
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    batch_path = os.path.join(cache_dir, "batch.pt")
    if not os.path.exists(batch_path):
        return None
    try:
        data = torch.load(batch_path, map_location="cpu", weights_only=False)
        print(f"    [ERA5 Cache] Loaded batch from {batch_path}")
        return data["eval_inputs"], data["eval_targets"], data["eval_forcings"], data["one_hot_original"], data["ts"]
    except Exception as e:
        print(f"    [ERA5 Cache] Failed to load batch cache: {e}, will re-download")
        return None


def save_era5_batch_cache(cache_root: str, want_time: str,
                          eval_inputs, eval_targets, eval_forcings, one_hot_original, ts):
    """保存初始 batch 到缓存"""
    if cache_root is None:
        return
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    os.makedirs(cache_dir, exist_ok=True)
    batch_path = os.path.join(cache_dir, "batch.pt")
    torch.save({
        "eval_inputs": eval_inputs,
        "eval_targets": eval_targets,
        "eval_forcings": eval_forcings,
        "one_hot_original": one_hot_original,
        "ts": ts,
        "want_time": want_time,
    }, batch_path)
    print(f"    [ERA5 Cache] Saved batch to {batch_path}")


def load_era5_forcings_cache(cache_root: str, want_time: str, num_rollout_steps: int):
    """
    尝试从缓存加载 forcings_extended。
    只有缓存的步数 >= num_rollout_steps 时才命中，否则返回 None（需重新下载）。
    """
    if cache_root is None:
        return None
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    # 寻找步数 >= num_rollout_steps 的缓存文件
    if not os.path.isdir(cache_dir):
        return None
    for fname in os.listdir(cache_dir):
        if not fname.startswith("forcings_N") or not fname.endswith(".pt"):
            continue
        cached_n = int(fname[len("forcings_N"):-len(".pt")])
        if cached_n >= num_rollout_steps:
            fpath = os.path.join(cache_dir, fname)
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                forcings = data["forcings_extended"]
                # 如果缓存步数更多，截取前 num_rollout_steps 步
                if cached_n > num_rollout_steps:
                    forcings = forcings.isel(time=slice(0, num_rollout_steps))
                print(f"    [ERA5 Cache] Loaded forcings (N={cached_n}) from {fpath}")
                return forcings
            except Exception as e:
                print(f"    [ERA5 Cache] Failed to load forcings cache: {e}, will re-download")
                return None
    return None


def save_era5_forcings_cache(cache_root: str, want_time: str, num_rollout_steps: int, forcings_extended):
    """保存 forcings_extended 到缓存"""
    if cache_root is None:
        return
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"forcings_N{num_rollout_steps}.pt")
    torch.save({"forcings_extended": forcings_extended, "want_time": want_time, "num_steps": num_rollout_steps}, fpath)
    print(f"    [ERA5 Cache] Saved forcings (N={num_rollout_steps}) to {fpath}")


def load_era5_gt_cache(cache_root: str, want_time: str, num_rollout_steps: int):
    """
    尝试从缓存加载 gt_list（长度 = num_rollout_steps 的 list[xr.Dataset | None]）。
    只有缓存步数 >= num_rollout_steps 时才命中。
    """
    if cache_root is None:
        return None
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    if not os.path.isdir(cache_dir):
        return None
    for fname in os.listdir(cache_dir):
        if not fname.startswith("gt_N") or not fname.endswith(".pt"):
            continue
        cached_n = int(fname[len("gt_N"):-len(".pt")])
        if cached_n >= num_rollout_steps:
            fpath = os.path.join(cache_dir, fname)
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                gt_list = data["gt_list"]
                # 如果缓存步数更多，截取
                if cached_n > num_rollout_steps:
                    gt_list = gt_list[:num_rollout_steps]
                print(f"    [ERA5 Cache] Loaded GT list (N={cached_n}) from {fpath}")
                return gt_list
            except Exception as e:
                print(f"    [ERA5 Cache] Failed to load GT cache: {e}, will re-download")
                return None
    return None


def save_era5_gt_cache(cache_root: str, want_time: str, num_rollout_steps: int, gt_list: list):
    """保存 gt_list 到缓存"""
    if cache_root is None:
        return
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"gt_N{num_rollout_steps}.pt")
    torch.save({"gt_list": gt_list, "want_time": want_time, "num_steps": num_rollout_steps}, fpath)
    print(f"    [ERA5 Cache] Saved GT list (N={num_rollout_steps}) to {fpath}")


# =========================
# 上下文：一次性加载 + 复用
# =========================
@dataclasses.dataclass
class GlobalInputs:
    # —— 保存/调度 —— #
    save_mode: str = "by_dates"             # "by_dates" | "by_params"
    output_dir: str = "/fs/.../general_guidance_output"
    baseline_cache_dir: Optional[str] = None  # Baseline 缓存目录，None 表示禁用缓存
    epoch: int = 0
    only_guided: bool = False
    only_baseline: bool = False  # 如果为 True，只运行 baseline，跳过 guided

    # —— Guidance 固定项（非 sweep 的那部分） —— #
    guidance_strength: float = 0.7
    readout_collect_steps_for_vis: Tuple[int, ...] = (10, 15)
    guide_loss_type: str = "readout_l2"
    guide_normalize_grad: bool = True
    guide_eps: float = 1e-6

    # —— 数据与模型 —— #
    gcs_bucket: str = "dm_graphcast"
    gcs_dir_prefix: str = "gencast/"
    era5_zarr_url: str = (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    )
    years_train: Tuple[int, int] = (2020, 2021)
    years_eval: Tuple[int, int] = (2022, 2023)
    tracks_folder: str = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/tracks"
    step_12h: bool = True
    dask_num_workers: int = 2
    from_cache: bool = False
    cache_file: str = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/debug_result/eval_batch_cache_from_train.pt"
    # ERA5 batch 缓存目录：None 表示禁用缓存
    # 启用后，第一次加载时自动保存预处理后的 batch 到本地，下次直接读取
    era5_cache_dir: Optional[str] = None

    full_model_path: Optional[str] = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/debug_result/09-05_training/09-05_training_checkpoint_19k.pt"


@dataclasses.dataclass
class Context:
    params: dict
    readout_guided_inference_fn_factory: any  # 根据 readout_collect_steps_for_vis 生成 apply_fn
    eval_ds: DateMergedERA5TyphoonSizeDataset
    task_config: any


def prepare_context(G: GlobalInputs) -> Context:
    t_start_total = time.time()
    
    print("\n[Step 1/4] Loading checkpoint and stats from GCS...")
    t0 = time.time()
    gcs_bucket_h, dir_prefix = get_gcs_bucket(G.gcs_bucket, G.gcs_dir_prefix)
    ckpt, (diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level) = load_checkpoint_and_stats(
        gcs_bucket_h, dir_prefix
    )
    print(f"    -> Done in {time.time() - t0:.2f}s")
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    base_cfg = ckpt.denoiser_architecture_config
    tbd_spt_cfg = dataclasses.replace(
        base_cfg.sparse_transformer_config,
        attention_type="triblockdiag_mha",
        mask_type="full"
    )
    cfg_dict = dataclasses.asdict(base_cfg)
    cfg_dict["sparse_transformer_config"] = tbd_spt_cfg
    cfg_dict["ReadOut_flag"] = True

    task_config = ckpt.task_config
    noise_config = ReadOutNoiseConfig(**ckpt.noise_config.__dict__, ReadOut_flag=True)
    denoiser_architecture_config = ReadOutDenoiserArchitectureConfig(**cfg_dict)
    noise_encoder_config = ckpt.noise_encoder_config

    params = ckpt.params
    state = {}

    print("\n[Step 2/4] Loading ERA5 data from Zarr...")
    t0 = time.time()
    dask.config.set(scheduler="threads", num_workers=G.dask_num_workers)
    ds_train = open_era5_zarr(G.era5_zarr_url, G.years_train, step_12h=G.step_12h)
    ds_eval  = open_era5_zarr(G.era5_zarr_url, G.years_eval,  step_12h=G.step_12h)
    print(f"    -> Done in {time.time() - t0:.2f}s")
    print(f"    Train size: {ds_train.sizes['time']}  | Eval size: {ds_eval.sizes['time']}")

    print("\n[Step 3/4] Building datasets...")
    t0 = time.time()
    _, eval_ds = build_datasets(
        ds_train, ds_eval, task_config, G.tracks_folder,
        G.years_train, G.years_eval,
    )
    print(f"    -> Done in {time.time() - t0:.2f}s")
    print(f"    Total patches (eval): {len(eval_ds)}")

    if G.full_model_path and os.path.exists(G.full_model_path):
        print(f"\n[Step 4/4] Loading full trained model from {G.full_model_path}...")
        t0 = time.time()
        full_params = torch.load(G.full_model_path, weights_only=False)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), full_params)
        print(f"    -> Done in {time.time() - t0:.2f}s")
    else:
        print("\n[Step 4/4] Using pretrained model params (no full_model_path)")

    def make_apply_fn_for_steps(readout_collect_steps_for_vis: Tuple[int, ...]):
        sampler_config = ReadOutSamplerConfig(**ckpt.sampler_config.__dict__,
                                              selected_denoising_step=list(readout_collect_steps_for_vis))
        construct_wrapped = make_predictor_wrapped(
            task_config, sampler_config, noise_config, noise_encoder_config,
            denoiser_architecture_config, diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level
        )
        readout_guided_inference_fn, _ = build_jitted_functions(params, state, construct_wrapped)
        return readout_guided_inference_fn

    ctx = Context(
        params=params,
        readout_guided_inference_fn_factory=make_apply_fn_for_steps,
        eval_ds=eval_ds,
        task_config=task_config,
    )
    print(f"\n[prepare_context] Total time: {time.time() - t_start_total:.2f}s\n")
    return ctx


def run_or_load_baseline(
    ctx: Context,
    *,
    want_time: str,
    manual_targets: List[Dict],
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    eval_forcings: xr.Dataset,
    one_hot_original: torch.Tensor,
    readout_collect_steps_for_vis: Tuple[int, ...],
    baseline_cache_dir: Optional[str],
    output_dir: str,
    save_mode: str,
    epoch: int,
    guidance_mode: str = "steer_one",  # 新增：guidance 模式（用于目录结构）
    visual_vars: Optional[List[str]] = None,
) -> Tuple[xr.Dataset, Dict[str, xr.Dataset], xr.Dataset, str]:
    """
    运行或加载 baseline 结果
    
    Returns:
        (preds_A, readouts_A, gt_A, out_A)
    """
    # 生成 apply_fn
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    
    # 确定 out_A 目录（用于可视化）
    # 注意：baseline 不依赖 guidance_mode，但为了目录结构一致性，需要传递
    paths = decide_output_dirs(
        output_dir=output_dir,
        save_mode=save_mode,
        want_time=want_time,
        manual_targets=manual_targets,
        readout_collect_steps_for_vis=readout_collect_steps_for_vis,
        inner_idxs=[],  # baseline 不需要这些
        inner_steps_map={},
        max_opt_steps=1,
        inner_lr=1.0,
        strength=0.0,
        guidance_mode=guidance_mode,  # 传递 guidance_mode 以保持目录结构一致
        intensity_scaling_configs=None,
        shift_configs=None,
    )
    out_A = paths["out_A"]
    
    # 如果启用了缓存，尝试加载
    if baseline_cache_dir is not None:
        cache_path = _get_baseline_cache_path(
            baseline_cache_dir, want_time, manual_targets, readout_collect_steps_for_vis
        )
        cached = _load_baseline_cache(cache_path)
        if cached is not None:
            preds_A, readouts_A, gt_A = cached
            print(f"    [Baseline] Using cached results, skipping inference")
            # 可视化（如果需要）
            if not (os.path.isdir(out_A) and os.listdir(out_A)):
                print("    Visualizing baseline results...")
                t_vis = time.time()
                # 使用原始的 one_hot_original（真实风暴位置）进行可视化
                _visualize_and_export(
                    gt_list=[gt_A], predictions_list=[preds_A], readouts_list=[readouts_A],
                    one_hot_torch=one_hot_original,
                    out_dir=out_A,
                    epoch=epoch,
                    visual_vars=visual_vars,
                )
                print(f"    -> Visualization done in {time.time() - t_vis:.2f}s")
            else:
                print("    [A_no_guidance] exists -> skip visualization")
            return preds_A, readouts_A, gt_A, out_A
    
    # 缓存不存在或未启用，运行 baseline
    print("\n[Baseline] Running inference (will be cached if cache_dir is set)...")
    t0 = time.time()
    preds_A, readouts_A, gt_A, _ = _run_once_with_guidance(
        params=ctx.params,
        eval_inputs=eval_inputs, eval_targets=eval_targets, eval_forcings=eval_forcings,
        readout_guided_inference_fn=apply_fn,
        target_readout_one_hot=None,  # baseline 不需要 guidance mask
        readout_collect_steps_for_vis=tuple(readout_collect_steps_for_vis),
        guidance_strength=0.0,
        guide_inner_opt_step_idxs=[],
        guide_inner_opt_steps_map={},
        guide_max_opt_steps=1,
        guide_inner_opt_lr=1.0,
        guide_loss_type="xt_l2",  # baseline 使用 xt_l2，不需要 readout_l2
        guide_normalize_grad=True,
        guide_eps=1e-6,
    )
    print(f"    -> Inference done in {time.time() - t0:.2f}s")
    
    # 保存缓存（如果启用了）
    if baseline_cache_dir is not None:
        cache_path = _get_baseline_cache_path(
            baseline_cache_dir, want_time, manual_targets, readout_collect_steps_for_vis
        )
        _save_baseline_cache(cache_path, preds_A, readouts_A, gt_A)
    
    # 可视化
    if not (os.path.isdir(out_A) and os.listdir(out_A)):
        print("    Visualizing baseline results...")
        t_vis = time.time()
        # 使用原始的 one_hot_original（真实风暴位置）进行可视化
        _visualize_and_export(
            gt_list=[gt_A], predictions_list=[preds_A], readouts_list=[readouts_A],
            one_hot_torch=one_hot_original,
            out_dir=out_A,
            epoch=epoch,
            visual_vars=visual_vars,
        )
        print(f"    -> Visualization done in {time.time() - t_vis:.2f}s")
    else:
        print("    [A_no_guidance] exists -> skip visualization")
    
    return preds_A, readouts_A, gt_A, out_A


def run_once_with_context(
    ctx: Context,
    *,
    # —— 保存/调度 —— #
    save_mode: str,
    output_dir: str,
    epoch: int,
    only_guided: bool,
    only_baseline: bool,  # 如果为 True，只运行 baseline，跳过 guided

    # —— 时间（只需单个时间点）—— #
    want_time: str,  # start_date
    end_date: Optional[str] = None,  # 新增：目标时间点（如果 None，则从 start_date + 12h 计算）

    # —— Manual Guidance 目标 —— #
    manual_targets: List[Dict],

    # —— Selective Storm Guidance 参数 —— #
    guidance_mode: str = "steer_one",  # 新增：模式选择 "steer_one" 或 "delete_one"
    selected_storm_id: Optional[int] = None,  # 新增：要引导的风暴 ID（如果 None，则要求 IO 输入）
    guidance_mask_padding: int = 5,  # 新增：bounding box 的 padding
    use_guidance_mask: bool = True,  # 新增：是否使用 mask 限制 loss 区域

    # —— Guidance 参数（含 sweep）—— #
    guidance_strength: float,
    readout_collect_steps_for_vis: Tuple[int, ...],
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],  # 新增：per-step optimization 次数
    guide_max_opt_steps: int,                    # 新增：最大次数（用于 JAX 编译）
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,

    # —— 可视化 —— #
    visual_vars: Optional[List[str]] = None,
    
    # —— Baseline 缓存（可选）—— #
    baseline_preds: Optional[xr.Dataset] = None,
    baseline_readouts: Optional[Dict[str, xr.Dataset]] = None,
    baseline_gt: Optional[xr.Dataset] = None,
    baseline_out_A: Optional[str] = None,
    
    # —— 预加载的数据（可选，避免重复加载）—— #
    eval_inputs: Optional[xr.Dataset] = None,
    eval_targets: Optional[xr.Dataset] = None,
    eval_forcings: Optional[xr.Dataset] = None,
    one_hot_original: Optional[torch.Tensor] = None,
) -> Dict[str, str | bool]:
    t_run_start = time.time()
    
    # 计算 end_date（如果未提供，从 start_date + 12h 计算）
    if end_date is None:
        start_ts = pd.Timestamp(want_time)
        end_ts = start_ts + pd.Timedelta(hours=12)
        end_date = str(end_ts)
    print(f"\n[Selective Guidance] start_date: {want_time}, end_date: {end_date}")
    
    # 获取 storm_id（配置优先 → 缓存 → IO输入）
    storm_id = get_storm_id_for_date_pair(want_time, end_date, selected_storm_id, ctx)
    
    # 提取选定风暴信息
    storms = extract_storms_from_end_date(ctx, end_date)
    if storm_id < 0 or storm_id >= len(storms):
        raise ValueError(f"Invalid storm_id: {storm_id}, available: 0-{len(storms)-1}")
    selected_storm = storms[storm_id]
    print(f"    [Selected Storm] ID {storm_id}: lat={selected_storm['lat']:.1f}, "
          f"lon={selected_storm['lon']:.1f}, rsize={selected_storm['rsize']:.1f}")
    
    # 加载数据（如果未提供）
    if eval_inputs is None or eval_targets is None or eval_forcings is None or one_hot_original is None:
        # 检查日期可用（只检查单个时间点）
        check = detect_want_times(ctx.eval_ds, [want_time])
        for r in check:
            print(f"  - {r['time']}: ok={r['ok']}  | in_tracks={r['in_tracks']}  | full_window={r['has_full_window']}  | n_samples={r['n_samples']} "
                  f"{'| reason='+r['reason'] if r['reason'] else ''}")
        if any(not r["ok"] for r in check):
            raise ValueError("指定的时间不可用，请修正后再运行。")

        # 加载单个时间点的数据
        print("\n[Run Step 1] Loading eval batch...")
        t0 = time.time()
        wanted_loader = build_wanted_subloader(ctx.eval_ds, [want_time], batch_size=1)
        it = iter(wanted_loader)
        eval_inputs, eval_targets, eval_forcings, one_hot_original, ts = next(it)
        print(f"    -> Done in {time.time() - t0:.2f}s")
    else:
        print("\n[Run Step 1] Using pre-loaded eval batch (skipping data loading)")
    
    # 从 start_date 的 one_hot_original 提取所有风暴
    print(f"\n[Run Step 1.2] Extracting storms from start_date...")
    start_storms = extract_storms_from_one_hot(one_hot_original)
    print(f"    [Start Storms] Found {len(start_storms)} storm(s) in start_date")
    for storm in start_storms:
        print(f"      Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, rsize={storm['rsize']:.1f}")
    
    # 匹配 start_date 和 end_date 的风暴
    print(f"\n[Run Step 1.3] Matching storms between start_date and end_date...")
    selected_storm_id_in_start = match_storm_between_dates(start_storms, selected_storm, distance_threshold=10.0)
    
    # 根据模式构建 target mask
    print(f"\n[Run Step 1.4] Building final target mask (mode: {guidance_mode})...")
    batch_size = eval_inputs.sizes["batch"]
    
    if guidance_mode == "steer_one":
        # steer_one 模式：移动选定的storm
        one_hot_for_guidance = build_final_target_mask(
            one_hot_original,
            start_storms,
            mode="steer_one",
            selected_storm_id_in_start=selected_storm_id_in_start,
            manual_target=manual_targets[0],
            batch_size=batch_size,
        )
        num_other_storms = len(start_storms) - (1 if selected_storm_id_in_start is not None else 0)
        print(f"    [Steer One Mode] Final target mask contains: manual target + {num_other_storms} other storm(s) from start_date")
        target_storm = manual_targets[0]  # 用户指定的目标
        
    elif guidance_mode == "delete_one":
        # delete_one 模式：删除选定的storm
        one_hot_for_guidance = build_final_target_mask(
            one_hot_original,
            start_storms,
            mode="delete_one",
            selected_storm_id_in_start=selected_storm_id_in_start,
            manual_target=None,  # delete_one 模式不需要 manual_target
            batch_size=batch_size,
        )
        num_kept_storms = len(start_storms) - (1 if selected_storm_id_in_start is not None else 0)
        print(f"    [Delete One Mode] Final target mask contains: {num_kept_storms} storm(s) from start_date (deleted storm_id {selected_storm_id_in_start})")
        # delete_one 模式下，target_storm 用于 bounding box 计算（使用原始storm位置）
        if selected_storm_id_in_start is not None and selected_storm_id_in_start < len(start_storms):
            deleted_storm = start_storms[selected_storm_id_in_start]
            target_storm = {
                "lat": deleted_storm["lat"],
                "lon": deleted_storm["lon"],
                "radius": int(deleted_storm["rsize"])
            }
        else:
            target_storm = None
    else:
        raise ValueError(f"Unknown guidance_mode: {guidance_mode}. Must be 'steer_one' or 'delete_one'")
    
    one_hot_for_vis = one_hot_original  # 原始 mask 用于可视化对比
    print(f"    [Guidance Mode: {guidance_mode}] targets: {manual_targets if guidance_mode == 'steer_one' else 'N/A (delete_one mode)'}")
    
    # 计算 bounding box mask（包含选定原始风暴和目标风暴）
    if target_storm is not None and use_guidance_mask:
        guidance_mask = compute_bounding_box_mask(
            selected_storm, target_storm, batch_size, padding=guidance_mask_padding
        )
        print(f"    [Guidance Mask] ENABLED with padding={guidance_mask_padding}")
    else:
        guidance_mask = None
        if not use_guidance_mask:
            print(f"    [Guidance Mask] DISABLED - using global loss calculation")
        else:
            print(f"    [Guidance Mask] DISABLED - target_storm is None")

    # 目录路由（SAVE_MODE）
    paths = decide_output_dirs(
        output_dir=output_dir,
        save_mode=save_mode,
        want_time=want_time,
        manual_targets=manual_targets,
        readout_collect_steps_for_vis=readout_collect_steps_for_vis,
        inner_idxs=list(guide_inner_opt_step_idxs),
        inner_steps_map=dict(guide_inner_opt_steps_map),
        max_opt_steps=int(guide_max_opt_steps),
        inner_lr=float(guide_inner_opt_lr),
        strength=float(guidance_strength),
        guidance_mode=guidance_mode,  # 新增：传递 guidance_mode
        intensity_scaling_configs=None,
        shift_configs=None,
    )
    root = paths["root"]; out_A = paths["out_A"]; out_B = paths["out_B"]
    print("[SAVE_MODE]", save_mode, "| root:", root)
    os.makedirs(root, exist_ok=True)

    # 快速可视化：保存 mask 对比图（用于检查）
    # print("\n[Run Step 1.5] Visualizing mask comparison...")
    # mask_vis_path = os.path.join(root, "mask_comparison.png")
    # visualize_mask_comparison(
    #     one_hot_target=one_hot_for_guidance,
    #     guidance_mask=guidance_mask,
    #     output_path=mask_vis_path,
    #     selected_storm_info=selected_storm,
    #     target_storm_info=manual_targets[0] if manual_targets else None,
    # )

    # 针对本次 readout_collect_steps_for_vis 生成 apply_fn
    print("\n[Run Step 2] Building JIT apply_fn...")
    t0 = time.time()
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    print(f"    -> Done in {time.time() - t0:.2f}s")

    # baseline（如果提供了缓存结果，直接使用；否则运行）
    if not only_guided:
        if baseline_preds is not None and baseline_readouts is not None and baseline_gt is not None:
            # 使用提供的 baseline 结果（来自缓存）
            print("\n[Run Step 3a] Using cached baseline results (skipping inference)...")
            preds_A, readouts_A, gt_A = baseline_preds, baseline_readouts, baseline_gt
            # 使用提供的 out_A 路径，如果提供了的话
            if baseline_out_A is not None:
                out_A = baseline_out_A
        else:
            # 运行 baseline（向后兼容，或缓存未启用时）
            print("\n[Run Step 3a] Running Baseline (No Guidance) inference...")
            t0 = time.time()
            preds_A, readouts_A, gt_A, _ = _run_once_with_guidance(
                params=ctx.params,
                eval_inputs=eval_inputs, eval_targets=eval_targets, eval_forcings=eval_forcings,
                readout_guided_inference_fn=apply_fn,
                target_readout_one_hot=one_hot_for_guidance,
                readout_collect_steps_for_vis=tuple(readout_collect_steps_for_vis),
                guidance_strength=0.0,
                guide_inner_opt_step_idxs=list(guide_inner_opt_step_idxs),
                guide_inner_opt_steps_map=dict(guide_inner_opt_steps_map),
                guide_max_opt_steps=int(guide_max_opt_steps),
                guide_inner_opt_lr=float(guide_inner_opt_lr),
                guide_loss_type="xt_l2",  # baseline 使用 xt_l2，不需要 readout_l2
                guide_normalize_grad=guide_normalize_grad,
                guide_eps=guide_eps,
                guidance_mask=None,  # baseline 不使用 mask
            )
            print(f"    -> Inference done in {time.time() - t0:.2f}s")
        
        # 可视化可以选择跳过（如果目录已存在且非空）
        if not (os.path.isdir(out_A) and os.listdir(out_A)):
            print("    Visualizing baseline results...")
            t_vis = time.time()
            _visualize_and_export(
                gt_list=[gt_A], predictions_list=[preds_A], readouts_list=[readouts_A],
                one_hot_torch=one_hot_for_vis,
                out_dir=out_A,
                epoch=epoch,
                visual_vars=visual_vars,
            )
            print(f"    -> Visualization done in {time.time() - t_vis:.2f}s")
        else:
            print("    [A_no_guidance] exists -> skip visualization (inference done for comparison)")

    # guided（如果 only_baseline=True，则跳过）
    if not only_baseline:
        print("\n[Run Step 3b] Running Guided inference...")
        t0 = time.time()
        preds_B, readouts_B, gt_B, loss_history_B = _run_once_with_guidance(
            params=ctx.params,
            eval_inputs=eval_inputs, eval_targets=eval_targets, eval_forcings=eval_forcings,
            readout_guided_inference_fn=apply_fn,
            target_readout_one_hot=one_hot_for_guidance,
            readout_collect_steps_for_vis=tuple(readout_collect_steps_for_vis),
            guidance_strength=float(guidance_strength),
            guide_inner_opt_step_idxs=list(guide_inner_opt_step_idxs),
            guide_inner_opt_steps_map=dict(guide_inner_opt_steps_map),
            guide_max_opt_steps=int(guide_max_opt_steps),
            guide_inner_opt_lr=float(guide_inner_opt_lr),
            guide_loss_type=guide_loss_type,
            guide_normalize_grad=guide_normalize_grad,
            guide_eps=guide_eps,
            guidance_mask=guidance_mask,  # 新增：传递 bounding box mask
        )
        print(f"    -> Inference done in {time.time() - t0:.2f}s")
        
        # 保存和可视化 loss
        if loss_history_B:
            print("\n[Run Step 3c] Saving loss history...")
            loss_curve_path = os.path.join(out_B, "loss_curves.png")
            loss_summary_path = os.path.join(out_B, "loss_summary.txt")
            loss_json_path = os.path.join(out_B, "loss_history.json")
            
            plot_loss_curves(loss_history_B, loss_curve_path)
            save_loss_summary(loss_history_B, loss_summary_path)
            
            # 保存 JSON 格式（方便后续分析）
            with open(loss_json_path, 'w') as f:
                json.dump(loss_history_B, f, indent=2)
            print(f"[Loss JSON] Saved to {loss_json_path}")
        else:
            print("\n[Run Step 3c] No loss history to save (strength=0 or no optimization)")
        
        print("    Visualizing guided results...")
        t_vis = time.time()
        _visualize_and_export(
            gt_list=[gt_B], predictions_list=[preds_B], readouts_list=[readouts_B],
            one_hot_torch=one_hot_for_guidance,  # 用 guidance mask 可视化
            out_dir=out_B,
            epoch=epoch,
            visual_vars=visual_vars,
        )
        print(f"    -> Visualization done in {time.time() - t_vis:.2f}s")
    else:
        print("\n[Run Step 3b] Skipping Guided inference (only_baseline=True)")
        preds_B, readouts_B, gt_B, loss_history_B = None, None, None, None

    # ==== 创建对比可视化（需要 baseline 和 guided 都存在） ====
    if not only_guided and not only_baseline:
        print("\n[Run Step 4] Creating comparison visualizations...")
        t_comp = time.time()
        # 每个 B_ setting 有自己的 comparison 目录
        comparison_dir = os.path.join(out_B, "comparison_visualizations")
        os.makedirs(comparison_dir, exist_ok=True)

        # 构建 readout_frames dict 格式
        readout_frames_A = {
            key: [readouts_A[key].isel(time=0, batch=0)]
            for key in readouts_A.keys()
        }
        readout_frames_B = {
            key: [readouts_B[key].isel(time=0, batch=0)]
            for key in readouts_B.keys()
        }

        # Readout 对比：原始风暴位置 vs guidance 目标位置
        one_hot_original_np = one_hot_for_vis.numpy() if hasattr(one_hot_for_vis, "numpy") else np.asarray(one_hot_for_vis)
        one_hot_guidance_np = one_hot_for_guidance.numpy() if hasattr(one_hot_for_guidance, "numpy") else np.asarray(one_hot_for_guidance)
        create_readout_comparison_plot(
            no_guidance_readout_frames_dict=readout_frames_A,
            guided_readout_frames_dict=readout_frames_B,
            one_hot_t1=one_hot_original_np,
            one_hot_t2=one_hot_guidance_np,
            output_dir=comparison_dir,
            epoch=epoch,
            ts=str(gt_B.datetime.values) if 'datetime' in gt_B.coords else None,
            steps=list(readout_frames_A.keys()),
        )

        # 天气变量对比 - 使用 visual_vars 过滤
        all_vars = list(gt_B.data_vars.keys())
        if visual_vars is not None:
            variable_list = [v for v in all_vars if v in visual_vars]
        else:
            variable_list = all_vars
        variable_list_wLevel = [v for v in variable_list if 'level' in gt_B[v].dims]
        level_dict = {v: 500 for v in variable_list_wLevel}

        create_all_variable_comparisons(
            no_guidance_predictions=[preds_A],
            guided_predictions=[preds_B],
            gt_frames=[gt_B],
            output_dir=comparison_dir,
            epoch=epoch,
            level_dict=level_dict,
            variables=variable_list,
        )
        print(f"    -> Comparison visualizations done in {time.time() - t_comp:.2f}s")
        print(f"    Saved to {comparison_dir}")
    elif only_baseline:
        print("\n[Run Step 4] Skipping comparison visualizations (only_baseline=True, no guided results to compare)")

    print(f"\n[run_once_with_context] Total time: {time.time() - t_run_start:.2f}s")
    return {
        "root": root,
        "out_A": out_A,
        "out_B": out_B,
        "time_slug": paths["time_slug"],
        "targets_slug": paths["targets_slug"],
        "param_slug": paths["param_slug"],
    }


def run_rollout_with_context(
    ctx: Context,
    *,
    # —— 保存/调度 —— #
    save_mode: str,
    output_dir: str,
    epoch: int,
    only_guided: bool,
    only_baseline: bool,

    # —— 时间 —— #
    want_time: str,
    end_date: Optional[str] = None,

    # —— Rollout 专用参数 —— #
    num_rollout_steps: int,  # rollout步数（例如：10步 = 5天）
    guidance_on_first_step: bool,  # 是否在第一步使用guidance

    # —— Manual Guidance 目标 —— #
    manual_targets: List[Dict],

    # —— Selective Storm Guidance 参数 —— #
    guidance_mode: str = "steer_one",
    selected_storm_id: Optional[int] = None,
    guidance_mask_padding: int = 5,
    use_guidance_mask: bool = True,

    # —— Guidance 参数 —— #
    guidance_strength: float,
    readout_collect_steps_for_vis: Tuple[int, ...],
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],
    guide_max_opt_steps: int,
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,

    # —— 可视化 —— #
    visual_vars: Optional[List[str]] = None,
    vis_n_procs: int = 1,
    vis_contourf_levels: int = 36,
    vis_coastlines_resolution: str = "50m",
    vis_draw_gridlabels: bool = True,
    vis_add_borders: bool = True,
    vis_dpi: int = 240,
    
    # —— 预加载的数据（必须提供）—— #
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    forcings_extended: xr.Dataset,  # rollout专用：扩展的forcings
    one_hot_original: torch.Tensor,
    
    # —— 时间记录（可选）—— #
    timing_logger: Optional[TimingLogger] = None,
    
    # —— Random Seed —— #
    random_seed: int = 42,
    
    # —— Guidance Method —— #
    guidance_method: str = "direct_optim",
    
    # —— Input Manipulation —— #
    intensity_scaling_configs: Optional[List[Dict]] = None,
    shift_configs: Optional[List[Dict]] = None,
    
    # —— Local Affine —— #
    warp_configs: Optional[Dict] = None,

    # —— ERA5 本地缓存 —— #
    era5_cache_dir: Optional[str] = None,
) -> Dict[str, str | bool]:
    """
    [ROLLOUT ONLY] 执行multi-step rollout with optional first-step guidance
    
    这是rollout的高层封装函数，类似于 run_once_with_context 但支持多步预测：
    - Baseline路径：无guidance，rollout num_rollout_steps步
    - Guided路径：第一步使用guidance，后续步骤继续rollout
    - 生成对比可视化
    
    与 run_once_with_context 的主要区别：
    1. 接受 forcings_extended (time=num_rollout_steps) 而非单步forcings
    2. 使用 _run_rollout_with_guidance 而非 _run_once_with_guidance
    3. 返回的predictions_list是多步结果
    
    Args:
        ctx: Context对象
        num_rollout_steps: rollout步数（例如：10步 = 5天，每步12h）
        guidance_on_first_step: 是否在第一步使用guidance（后续步骤无guidance）
        forcings_extended: 扩展的forcings (batch, time=num_rollout_steps, ...)
        ... (其他参数与 run_once_with_context 相同)
    
    Returns:
        result_dict: 包含输出目录路径等信息
    
    Example:
        >>> res = run_rollout_with_context(
        ...     ctx,
        ...     num_rollout_steps=10,
        ...     guidance_on_first_step=True,
        ...     forcings_extended=forcings_extended,  # (B, 10, H, W, ...)
        ...     ...
        ... )
    """
    t_run_start = time.time()
    
    # 如果没有提供logger，创建一个临时的（不保存）
    if timing_logger is None:
        timing_logger = TimingLogger()
    
    print(f"\n[ROLLOUT Context] Starting {num_rollout_steps}-step rollout")
    print(f"[ROLLOUT Context] Guidance on first step: {guidance_on_first_step}")
    
    # ===== 预先加载所有 GT 数据用于可视化（支持本地缓存）=====
    print(f"\n[Rollout GT Loading] Pre-loading GT data for {num_rollout_steps} steps...")
    if timing_logger:
        timing_logger.start("GT loading")
    
    gt_list_for_visualization = []
    start_ts = pd.Timestamp(want_time)
    
    # 构建所有 rollout 步骤的目标时间点
    target_time_strs = []
    for step_idx in range(num_rollout_steps):
        target_time = start_ts + pd.Timedelta(hours=12 * (step_idx + 1))
        target_time_str = str(target_time)
        target_time_strs.append(target_time_str)
    
    # 尝试从缓存加载 GT
    _gt_cached = load_era5_gt_cache(era5_cache_dir, want_time, num_rollout_steps)
    if _gt_cached is not None:
        gt_list_for_visualization = _gt_cached
        print(f"  [GT Loading] Loaded {len(gt_list_for_visualization)} GT frames from cache")
    else:
        # 检查哪些时间点可用
        checks = detect_want_times(ctx.eval_ds, target_time_strs)
        available_indices = []
        
        for step_idx, check in enumerate(checks):
            if check["ok"]:
                available_indices.append(step_idx)
            else:
                print(f"  [Warning] Step {step_idx+1}: GT data not available for {target_time_strs[step_idx]} ({check.get('reason', 'unknown')})")
        
        # 批量加载可用的时间点
        gt_dict = {}
        if available_indices:
            available_time_strs = [target_time_strs[i] for i in available_indices]
            print(f"  [GT Loading] Batch loading {len(available_indices)}/{num_rollout_steps} time points from GCS...")
            
            try:
                wanted_loader = build_wanted_subloader(ctx.eval_ds, available_time_strs, batch_size=1)
                it = iter(wanted_loader)
                
                # 按顺序读取所有可用的GT数据
                for batch_idx, (_, eval_targets_batch, _, _, ts_list) in enumerate(it):
                    if batch_idx < len(available_indices):
                        step_idx = available_indices[batch_idx]
                        gt_frame = eval_targets_batch.isel(time=0, batch=0)
                        gt_dict[step_idx] = gt_frame
                        print(f"  [Step {step_idx+1}/{num_rollout_steps}] Loaded GT for {target_time_strs[step_idx]}")
                
                print(f"  [GT Loading] Successfully loaded {len(gt_dict)}/{len(available_indices)} GT frames")
                
            except Exception as e:
                print(f"  [GT Loading Error] Batch loading failed: {e}")
                print(f"  [GT Loading] Falling back to sequential loading...")
                gt_dict = {}
                # 回退到顺序加载
                for step_idx in available_indices:
                    try:
                        wanted_loader = build_wanted_subloader(ctx.eval_ds, [target_time_strs[step_idx]], batch_size=1)
                        it = iter(wanted_loader)
                        _, eval_targets_step, _, _, _ = next(it)
                        gt_frame = eval_targets_step.isel(time=0, batch=0)
                        gt_dict[step_idx] = gt_frame
                        print(f"  [Step {step_idx+1}/{num_rollout_steps}] Loaded GT (sequential fallback)")
                    except Exception as e2:
                        print(f"  [Warning] Step {step_idx+1}: Failed to load GT: {e2}")
        
        # 按顺序构建 gt_list（处理缺失的时间点）
        for step_idx in range(num_rollout_steps):
            if step_idx in gt_dict:
                gt_list_for_visualization.append(gt_dict[step_idx])
            else:
                # 使用 eval_targets 作为模板，填充 NaN
                gt_frame = eval_targets.isel(time=0, batch=0) * np.nan
                gt_list_for_visualization.append(gt_frame)
                print(f"  [Step {step_idx+1}/{num_rollout_steps}] Using NaN for missing GT data")
        
        # 保存到本地缓存
        save_era5_gt_cache(era5_cache_dir, want_time, num_rollout_steps, gt_list_for_visualization)
    
    if timing_logger:
        timing_logger.end("GT loading")
    print(f"[Rollout GT Loading] ✓ Pre-loaded {len(gt_list_for_visualization)} GT frames")
    
    # 计算 end_date（如果未提供）
    if end_date is None:
        start_ts = pd.Timestamp(want_time)
        end_ts = start_ts + pd.Timedelta(hours=12)
        end_date = str(end_ts)
    print(f"\n[Selective Guidance] start_date: {want_time}, end_date: {end_date}")
    
    # 获取 storm_id（配置优先 → 缓存 → IO输入）
    if timing_logger:
        timing_logger.start("Get storm_id")
    storm_id = get_storm_id_for_date_pair(want_time, end_date, selected_storm_id, ctx)
    if timing_logger:
        timing_logger.end("Get storm_id")
    
    # 提取选定风暴信息
    if timing_logger:
        timing_logger.start("Extract storms from end_date")
    storms = extract_storms_from_end_date(ctx, end_date)
    if timing_logger:
        timing_logger.end("Extract storms from end_date")
    if storm_id < 0 or storm_id >= len(storms):
        raise ValueError(f"Invalid storm_id: {storm_id}, available: 0-{len(storms)-1}")
    selected_storm = storms[storm_id]
    print(f"    [Selected Storm] ID {storm_id}: lat={selected_storm['lat']:.1f}, "
          f"lon={selected_storm['lon']:.1f}, rsize={selected_storm['rsize']:.1f}")
    
    # 从 start_date 的 one_hot_original 提取所有风暴
    print(f"\n[Run Step 1] Extracting storms from start_date...")
    if timing_logger:
        timing_logger.start("Step 1: Extract storms from start_date")
    start_storms = extract_storms_from_one_hot(one_hot_original)
    if timing_logger:
        timing_logger.end("Step 1: Extract storms from start_date")
    print(f"    [Start Storms] Found {len(start_storms)} storm(s) in start_date")
    for storm in start_storms:
        print(f"      Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, rsize={storm['rsize']:.1f}")
    
    # 匹配 start_date 和 end_date 的风暴
    print(f"\n[Run Step 2] Matching storms between start_date and end_date...")
    if timing_logger:
        timing_logger.start("Step 2: Match storms")
    selected_storm_id_in_start = match_storm_between_dates(start_storms, selected_storm, distance_threshold=10.0)
    if timing_logger:
        timing_logger.end("Step 2: Match storms")
    
    # 根据模式构建 target mask
    print(f"\n[Run Step 3] Building final target mask (mode: {guidance_mode})...")
    if timing_logger:
        timing_logger.start("Step 3: Build target mask")
    batch_size = eval_inputs.sizes["batch"]
    
    if guidance_mode == "steer_one":
        one_hot_for_guidance = build_final_target_mask(
            one_hot_original, start_storms, mode="steer_one",
            selected_storm_id_in_start=selected_storm_id_in_start,
            manual_target=manual_targets[0], batch_size=batch_size,
        )
        target_storm = manual_targets[0]
    elif guidance_mode == "delete_one":
        one_hot_for_guidance = build_final_target_mask(
            one_hot_original, start_storms, mode="delete_one",
            selected_storm_id_in_start=selected_storm_id_in_start,
            manual_target=None, batch_size=batch_size,
        )
        if selected_storm_id_in_start is not None and selected_storm_id_in_start < len(start_storms):
            deleted_storm = start_storms[selected_storm_id_in_start]
            target_storm = {
                "lat": deleted_storm["lat"],
                "lon": deleted_storm["lon"],
                "radius": int(deleted_storm["rsize"])
            }
        else:
            target_storm = None
    else:
        raise ValueError(f"Unknown guidance_mode: {guidance_mode}")
    
    one_hot_for_vis = one_hot_original
    
    # 计算 bounding box mask
    if target_storm is not None and use_guidance_mask:
        guidance_mask = compute_bounding_box_mask(
            selected_storm, target_storm, batch_size, padding=guidance_mask_padding
        )
        print(f"    [Guidance Mask] ENABLED with padding={guidance_mask_padding}")
    else:
        guidance_mask = None
        print(f"    [Guidance Mask] DISABLED")
    
    if timing_logger:
        timing_logger.end("Step 3: Build target mask")

    # 目录路由
    if timing_logger:
        timing_logger.start("Step 3.5: Decide output dirs")
    paths = decide_output_dirs(
        output_dir=output_dir, save_mode=save_mode, want_time=want_time,
        manual_targets=manual_targets, readout_collect_steps_for_vis=readout_collect_steps_for_vis,
        inner_idxs=list(guide_inner_opt_step_idxs), inner_steps_map=dict(guide_inner_opt_steps_map),
        max_opt_steps=int(guide_max_opt_steps), inner_lr=float(guide_inner_opt_lr),
        strength=float(guidance_strength), guidance_mode=guidance_mode,
        random_seed=random_seed,
        guidance_method=guidance_method,
        intensity_scaling_configs=intensity_scaling_configs,
        shift_configs=shift_configs,
        warp_configs=warp_configs,
    )
    root = paths["root"]; out_A = paths["out_A"]; out_B = paths["out_B"]
    if timing_logger:
        timing_logger.end("Step 3.5: Decide output dirs")
    print("[SAVE_MODE]", save_mode, "| root:", root)
    os.makedirs(root, exist_ok=True)
    
    # ===== 检测是否已完成rollout推理和可视化 =====
    if not only_guided and not only_baseline:
        print(f"\n[Check] Checking if rollout is already completed...")
        print(f"  out_A: {out_A}")
        print(f"  out_B: {out_B}")
        print(f"  num_rollout_steps: {num_rollout_steps}")
        if timing_logger:
            timing_logger.start("Check rollout completed")
        is_completed, found_vars = check_rollout_completed(
            out_A, out_B, num_rollout_steps, visual_vars, min_steps_required=3
        )
        if timing_logger:
            timing_logger.end("Check rollout completed")
        if is_completed:
            print(f"\n[Skip] Rollout already completed for this setting!")
            print(f"  NetCDF data files found in both A_no_guidance and B_guided directories")
            print(f"  Skipping inference (DataOnly mode - no video generation)")
            
            # 保存时间记录到文件
            if timing_logger and timing_logger.output_path:
                timing_logger.save()
            
            return {
                "root": root,
                "out_A": out_A,
                "out_B": out_B,
                "time_slug": paths["time_slug"],
                "targets_slug": paths["targets_slug"],
                "param_slug": paths["param_slug"],
                "num_rollout_steps": num_rollout_steps,
                "skipped": True,  # 标记为跳过
            }

    # 生成 apply_fn
    print("\n[Run Step 4] Building JIT apply_fn...")
    if timing_logger:
        timing_logger.start("Step 4: Build JIT apply_fn")
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    if timing_logger:
        timing_logger.end("Step 4: Build JIT apply_fn")

    # ===== Baseline: 无guidance rollout =====
    if not only_guided:
        # 检查 baseline 是否已完成
        print("\n[Check] Checking if baseline rollout is already completed...")
        # 检查至少80%的步骤（或至少3步，取较大值）
        min_steps_to_check = max(3, int(num_rollout_steps * 0.8))
        baseline_completed = check_single_rollout_completed(
            out_A, num_rollout_steps, min_steps_required=min_steps_to_check
        )
        if baseline_completed:
            print(f"  [Skip] Baseline rollout already completed, skipping inference")
            print(f"  Data directory: {os.path.join(out_A, 'rollout_data')}")
            # 跳过 baseline 推理，但继续运行 guided（如果需要）
        else:
            print(f"  [Run] Baseline rollout not found or incomplete, will run inference")
            print("\n[Run Step 5a] Running Baseline Rollout (No Guidance)...")
            if timing_logger:
                timing_logger.start("Step 5a: Baseline rollout")
            preds_A_list, readouts_A_list, _ = _run_rollout_with_guidance(
                params=ctx.params,
                eval_inputs=eval_inputs,
                eval_targets_template=eval_targets,
                forcings_extended=forcings_extended,
                readout_guided_inference_fn=apply_fn,
                num_steps=num_rollout_steps,
                guidance_on_first_step=False,  # 无guidance
                guidance_method="none",  # Baseline
                target_readout_one_hot=None,
                guidance_mask=None,
                readout_collect_steps_for_vis=readout_collect_steps_for_vis,
                guidance_strength=0.0,
                guide_inner_opt_step_idxs=[],
                guide_inner_opt_steps_map={},
                guide_max_opt_steps=1,
                guide_inner_opt_lr=1.0,
                guide_loss_type="xt_l2",
                guide_normalize_grad=True,
                guide_eps=1e-6,
                timing_logger=timing_logger,
                random_seed=random_seed,
                intensity_scaling_configs=None,  # Baseline 不使用 intensity scaling
                shift_configs=None,  # Baseline 不使用 spatial shift
                warp_configs=None,  # Baseline 不使用 warp
            )
            if timing_logger:
                timing_logger.end("Step 5a: Baseline rollout")
            
            # 保存所有rollout数据（替代可视化）
            # Baseline没有guidance，所以不保存readout
            print("\n[Run Step 5a-save] Saving baseline rollout data...")
            if timing_logger:
                timing_logger.start("Step 5a-save: Baseline data saving")
            _save_rollout_data_netcdf(
                predictions_list=preds_A_list,
                readouts_list=readouts_A_list,
                gt_list=gt_list_for_visualization,
                one_hot_torch=one_hot_original,
                out_dir=out_A,
                metadata={
                    "epoch": epoch,
                    "visual_vars": visual_vars,
                    "num_steps": num_rollout_steps,
                    "type": "baseline",
                    "want_time": want_time,
                    "end_date": end_date,
                    "guidance_mode": guidance_mode,
                },
                save_readout_steps=None,  # Baseline没有guidance，不保存readout
                shift_configs=None,    # Baseline 不使用 spatial shift
            )
            if timing_logger:
                timing_logger.end("Step 5a-save: Baseline data saving")

    # ===== Guided: 第一步guidance + 后续rollout =====
    if not only_baseline:
        print("\n[Run Step 5b] Running Guided Rollout (First-step Guidance)...")
        if timing_logger:
            timing_logger.start("Step 5b: Guided rollout")
        preds_B_list, readouts_B_list, loss_history_B = _run_rollout_with_guidance(
            params=ctx.params,
            eval_inputs=eval_inputs,
            eval_targets_template=eval_targets,
            forcings_extended=forcings_extended,
            readout_guided_inference_fn=apply_fn,
            num_steps=num_rollout_steps,
            guidance_on_first_step=guidance_on_first_step,  # 第一步用guidance
            guidance_method=guidance_method,
            target_readout_one_hot=one_hot_for_guidance if guidance_on_first_step else None,
            guidance_mask=guidance_mask if guidance_on_first_step else None,
            readout_collect_steps_for_vis=readout_collect_steps_for_vis,
            guidance_strength=guidance_strength,
            guide_inner_opt_step_idxs=list(guide_inner_opt_step_idxs),
            guide_inner_opt_steps_map=dict(guide_inner_opt_steps_map),
            guide_max_opt_steps=int(guide_max_opt_steps),
            guide_inner_opt_lr=float(guide_inner_opt_lr),
            guide_loss_type=guide_loss_type,
            guide_normalize_grad=guide_normalize_grad,
            guide_eps=guide_eps,
            timing_logger=timing_logger,
            random_seed=random_seed,
            intensity_scaling_configs=intensity_scaling_configs,  # 使用传入的配置
            shift_configs=shift_configs,  # 使用传入的配置
            warp_configs=warp_configs,  # 使用传入的配置
        )
        if timing_logger:
            timing_logger.end("Step 5b: Guided rollout")
        
        # 保存loss history
        if loss_history_B:
            print("\n[Run Step 5c] Saving loss history...")
            if timing_logger:
                timing_logger.start("Step 5c: Save loss history")
            loss_curve_path = os.path.join(out_B, "loss_curves.png")
            loss_summary_path = os.path.join(out_B, "loss_summary.txt")
            loss_json_path = os.path.join(out_B, "loss_history.json")
            
            plot_loss_curves(loss_history_B, loss_curve_path)
            save_loss_summary(loss_history_B, loss_summary_path)
            
            with open(loss_json_path, 'w') as f:
                json.dump(loss_history_B, f, indent=2)
            print(f"[Loss JSON] Saved to {loss_json_path}")
            if timing_logger:
                timing_logger.end("Step 5c: Save loss history")
        
        # 保存所有rollout数据（替代可视化）
        # 只保存有guidance的步骤的readout（第一步）
        print("\n[Run Step 5b-save] Saving guided rollout data...")
        if timing_logger:
            timing_logger.start("Step 5b-save: Guided data saving")
        
        # 确定哪些步骤有guidance（只有第一步，如果guidance_on_first_step为True）
        readout_steps_to_save = []
        if guidance_on_first_step:
            readout_steps_to_save = [0]  # 只保存第一步的readout
            print(f"  [Save] Only saving readout for step 0 (with guidance)")
        else:
            print(f"  [Save] No guidance on first step, skipping readout saving")
        
        _save_rollout_data_netcdf(
            predictions_list=preds_B_list,
            readouts_list=readouts_B_list,
            gt_list=gt_list_for_visualization,
            one_hot_torch=one_hot_for_guidance,
            out_dir=out_B,
            metadata={
                "epoch": epoch,
                "visual_vars": visual_vars,
                "num_steps": num_rollout_steps,
                "type": "guided",
                "loss_history": loss_history_B,
                "want_time": want_time,
                "end_date": end_date,
                "guidance_mode": guidance_mode,
                "guidance_strength": guidance_strength,
                "guidance_on_first_step": guidance_on_first_step,  # 保存guidance信息
                "readout_steps_saved": readout_steps_to_save,  # 保存哪些步骤的readout被保存了
                "intensity_scaling_configs": intensity_scaling_configs,  # 保存scaling配置
                "shift_configs": shift_configs,  # 保存shift配置
            },
            save_readout_steps=readout_steps_to_save,  # 只保存有guidance的步骤
            shift_configs=shift_configs,  # 传递shift配置用于保存可视化信息
        )
        if timing_logger:
            timing_logger.end("Step 5b-save: Guided data saving")
    
    # ===== 跳过视频生成（在可视化脚本中处理） =====
    print("\n[Run] Skipping visualization and video generation (use visualization script separately)")
    
    # 保存时间记录到文件
    if timing_logger and timing_logger.output_path:
        timing_logger.save()
    
    print(f"\n[run_rollout_with_context] Total time: {time.time() - t_run_start:.2f}s")
    return {
        "root": root,
        "out_A": out_A,
        "out_B": out_B,
        "time_slug": paths["time_slug"],
        "targets_slug": paths["targets_slug"],
        "param_slug": paths["param_slug"],
        "num_rollout_steps": num_rollout_steps,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[INFO] GenCast Storm Guidance - Data Generation Script")
    print("=" * 80)

    # srun --jobid=6286809 --overlap -pty bash
    # srun --jobid=6285504 --overlap -pty bash
    

    # ═════════════════════════════════════════════════════════════════════════════
    # CONFIGURATION SECTION - 修改这里的参数来配置实验
    # ═════════════════════════════════════════════════════════════════════════════

    # -------------------------------------------------------------------------
    # 0. 根目录配置（Root Path - 修改此处适配不同机器）
    # -------------------------------------------------------------------------
    # 所有本地路径均基于此根目录，迁移时只需修改这一行
    ROOT_DIR: str = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl"

    # -------------------------------------------------------------------------
    # 0. Guidance Method Selection（核心模式选择 - 互斥）
    # -------------------------------------------------------------------------
    # 可选值（四选一，互斥）:
    #   - "none": Baseline only，无任何 guidance
    #   - "direct_optim": 直接优化 x_t（原 readout-based guidance）
    #   - "input_manipulation": 输入条件修改（Intensity Scaling + Spatial Shift）
    #   - "local_affine": 局部仿射变换优化（优化 warp 参数）- 即将实现
    GUIDANCE_METHOD: str = "none"
    
    # -------------------------------------------------------------------------
    # 1. 输出变量配置（仅用于元数据）
    # -------------------------------------------------------------------------
    VISUAL_VARS = [
        "mean_sea_level_pressure",
        "u_component_of_wind",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "mean_sea_level_pressure",
        "10m_v_component_of_wind",
        "total_precipitation_12hr",
        "10m_u_component_of_wind",
        "vertical_velocity",
        "specific_humidity",
        "temperature",
        "sea_surface_temperature",
    ]
    
    # -------------------------------------------------------------------------
    # 2. Guidance 模式配置（针对 direct_optim 模式）
    # -------------------------------------------------------------------------
    GUIDANCE_MODE: str = "steer_one"  # "steer_one" 或 "delete_one"
    SELECTED_STORM_IDS: List[Optional[int]] = [1]  # 要引导的风暴 ID (None = 需要手动输入)
    GUIDANCE_MASK_PADDING: int = 2
    USE_GUIDANCE_MASK: bool = False
    
    # -------------------------------------------------------------------------
    # 3. 目标位置配置 - Storm Dorian
    # -------------------------------------------------------------------------
    # MANUAL_TARGETS: List[Dict] = [
    #     # {"lat": 15.0, "lon": 296.0, "radius": 3},
    #      {"lat": 25.0, "lon": 296.0, "radius": 3},
    # ]
    # Affine_Center_Lat, Affine_Center_Lon = 1,1
    # SWEEP_WANT_TIMES: List[str] = ["2019-08-28 00:00:00"]  
    # END_DATES: List[Optional[str]] = ["2019-08-29 00:00:00"]
    


    # -------------------------------------------------------------------------
    # 3. 目标位置配置 - Storm Irma 2017-09-07
    # -------------------------------------------------------------------------
    MANUAL_TARGETS: List[Dict] = [
        # {"lat": 19.0, "lon": 294.0, "radius": 3}, # Same Loc. Stronger.
        {"lat": 24.0, "lon": 288.0, "radius": 3}, # Same Loc. Stronger.
        # {"lat": 14.0, "lon": 294.0, "radius": 3}, # Steered 
    ]
    Affine_Center_Lat, Affine_Center_Lon = 19.0, 294.0
    SWEEP_WANT_TIMES: List[str] = ["2017-09-07 00:00:00"]  
    END_DATES: List[Optional[str]] = ["2017-09-08 00:00:00"]


    # -------------------------------------------------------------------------
    # 5. Rollout 配置
    # -------------------------------------------------------------------------
    ENABLE_ROLLOUT: bool = True
    ROLLOUT_DAYS: int = 1
    NUM_ROLLOUT_STEPS: int = ROLLOUT_DAYS * 2
    GUIDANCE_ON_FIRST_STEP: bool = True
    ONLY_BASELINE: bool = True
    
    # =========================================================================
    # MODE-SPECIFIC CONFIGURATIONS（根据 GUIDANCE_METHOD 选择）
    # =========================================================================
    if GUIDANCE_METHOD == "direct_optim":
        # ---------------------------------------------------------------------
        # Mode 1: Direct Optimization 配置
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] direct_optim (直接优化 x_t)")
        
        # 格式: [(start_step, end_step, opt_steps), ...]
        SWEEP_GUIDE_OPT_CONFIGS = [
            [(1, 1, 1)],  # 示例：step 1 做 1 次优化
        ]
        
        READOUT_COLLECT_STEPS_FOR_VIS = (19)
        SWEEP_GUIDE_LR = [0.000000000000000001]  # 学习率
        SWEEP_RANDOM_SEEDS = [789, 1000, 100, xxx, xxx, 123, 213]  # Random seeds
        
        # 不使用 input_manipulation
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        WARP_CONFIGS = None
        
    elif GUIDANCE_METHOD == "input_manipulation":
        # ---------------------------------------------------------------------
        # Mode 2: Input Manipulation 配置
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] input_manipulation (输入条件修改)")
        
        # Intensity Scaling 配置
        INTENSITY_SCALING_VARIABLES: List[str] = [
            "u_component_of_wind",
            "v_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation_12hr",
            "10m_u_component_of_wind",
            "vertical_velocity",
        ]
        
        INTENSITY_SCALING_CONFIGS: List[Dict] = [
            {
                "step_idx": 0,
                "scale_factor": 3.0,
                "center_lat": 18.0,
                "center_lon": 296.0,
                "radius": 5.0,
                "variables": INTENSITY_SCALING_VARIABLES,
            },
            {
                "step_idx": 1,
                "scale_factor": 3.0,
                "center_lat": 19.0,
                "center_lon": 295.0,
                "radius": 5.0,
                "variables": INTENSITY_SCALING_VARIABLES,
            },
        ]
        
        # Spatial Shift 配置
        SHIFT_VARIABLES: List[str] = [
            "u_component_of_wind",
            "v_component_of_wind",
            "mean_sea_level_pressure",
            "10m_v_component_of_wind",
            "total_precipitation_12hr",
            "10m_u_component_of_wind",
            "vertical_velocity",
            "specific_humidity",
            "temperature",
            "sea_surface_temperature",
        ]
        
        SHIFT_CONFIGS: List[Dict] = [
            {
                "step_idx": 1,
                "center_lat": 17.0,
                "center_lon": 294.0,
                "radius": 7.0,
                "delta_lat": 4.0,
                "delta_lon": -1.0,
                "variables": SHIFT_VARIABLES,
                "interpolation_method": "linear",
            },
        ]
        
        # 不使用 direct_optim 的优化参数（但仍需提供默认值）
        SWEEP_GUIDE_OPT_CONFIGS = [[(1, 1, 1)]]  # 占位，不会被使用
        READOUT_COLLECT_STEPS_FOR_VIS = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
        SWEEP_GUIDE_LR = [0.000000000000000001]
        SWEEP_RANDOM_SEEDS = [789]
        WARP_CONFIGS = None
        
    elif GUIDANCE_METHOD == "local_affine":
        # ---------------------------------------------------------------------
        # Mode 3: Local Affine Warp 配置（即将实现）
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] local_affine (局部仿射变换优化)")
        
        # Storm Irma
        WARP_CONFIGS = {
            "enabled": True,
            "center_lat": Affine_Center_Lat, 
            "center_lon": Affine_Center_Lon,
            "radius": 8.0,
            "init_translation": [0.0, 0.0],  # 初始平移 [dlat, dlon]
            "init_rotation": 0.0,            # 初始旋转（弧度）
            "init_scale": [1.0, 1.0],        # 初始缩放 [sx, sy]
            "optimize_translation": True,
            "optimize_rotation": False,
            "optimize_scale": False,
            "learning_rate": 5e-2,
            # "learning_rate": 1e-10,
            "regularization_weight": 1e-3,
        }
        
        # 仍需 denoising 相关配置
        # SWEEP_GUIDE_OPT_CONFIGS = [[(5, 6, 16)]]
        SWEEP_GUIDE_OPT_CONFIGS = [[(3, 4, 75)]]
        READOUT_COLLECT_STEPS_FOR_VIS = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
        SWEEP_GUIDE_LR = [1.0]  # warp 的学习率
        SWEEP_RANDOM_SEEDS = [789]
        
        # 不使用 input_manipulation
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        
    elif GUIDANCE_METHOD == "none":
        # ---------------------------------------------------------------------
        # Mode: None（Baseline only）
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] none (Baseline only, 无 guidance)")
        
        # 提供默认值（不会被使用）
        SWEEP_GUIDE_OPT_CONFIGS = [[(1, 1, 1)]]
        READOUT_COLLECT_STEPS_FOR_VIS = (19,)  # 只收最后一个 denoising step
        SWEEP_GUIDE_LR = [0.000000000000000001]
        SWEEP_RANDOM_SEEDS = [789]
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        WARP_CONFIGS = None
        
    else:
        raise ValueError(f"Unknown GUIDANCE_METHOD: {GUIDANCE_METHOD}. "
                        f"Must be one of: 'none', 'direct_optim', 'input_manipulation', 'local_affine'")
    
    # -------------------------------------------------------------------------
    # 8. Guidance 优化超参解析（通用）
    # -------------------------------------------------------------------------
    def parse_guide_config(config):
        """从配置派生出 step_idxs, steps_map, max_opt_steps"""
        steps_map = {s: n for start, end, n in config for s in range(start, end + 1)}
        step_idxs = sorted(steps_map.keys())
        max_opt_steps = max(steps_map.values()) if steps_map else 1
        return step_idxs, steps_map, max_opt_steps
    
    # -------------------------------------------------------------------------
    # 9. 可视化参数（仅用于元数据）
    # -------------------------------------------------------------------------
    VIS_CONTOURF_LEVELS: int = 24
    VIS_COASTLINES_RESOLUTION: str = "110m"
    VIS_DRAW_GRIDLABELS: bool = False
    VIS_ADD_BORDERS: bool = False
    VIS_DPI: int = 150
    VIS_N_PROCS: int = 1
    
    # ═════════════════════════════════════════════════════════════════════════════
    # END OF CONFIGURATION SECTION
    # ═════════════════════════════════════════════════════════════════════════════


    # -------------------------------------------------------------------------
    # 10. 全局配置（数据与模型路径）
    # -------------------------------------------------------------------------
    G = GlobalInputs(
        save_mode="by_dates",
        output_dir=os.path.join(ROOT_DIR, "0305-01—LocalAffine_RollOut"),
        baseline_cache_dir=os.path.join(ROOT_DIR, "0305-01—LocalAffine_RollOut", "_baseline_cache"),
        epoch=0,
        only_guided=False,
        only_baseline=ONLY_BASELINE,
        guidance_strength=0.7,
        readout_collect_steps_for_vis=READOUT_COLLECT_STEPS_FOR_VIS,
        guide_loss_type="readout_l2",
        guide_normalize_grad=True,
        guide_eps=1e-6,
        gcs_bucket="dm_graphcast",
        gcs_dir_prefix="gencast/",
        era5_zarr_url="gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        years_train=(2010, 2015),
        years_eval=(2016, 2020),
        tracks_folder=os.path.join(ROOT_DIR, "tracks"),
        step_12h=True,
        dask_num_workers=2,
        from_cache=False,
        cache_file=os.path.join(ROOT_DIR, "debug_result", "eval_batch_cache_from_train.pt"),
        full_model_path=os.path.join(ROOT_DIR, "debug_result", "12-09_readout9steps_dynW1.0_MaxPosWeight25_lr1e4_Y-Range-1959-2016", "checkpoints", "checkpoint_epoch_020000.pt"),
        era5_cache_dir=os.path.join(ROOT_DIR, "_era5_batch_cache"),
    )
    
    # -------------------------------------------------------------------------
    # 打印配置摘要
    # -------------------------------------------------------------------------
    print(f"\n[Config] Guidance Method: {GUIDANCE_METHOD}")
    print(f"[Config] Rollout: {ENABLE_ROLLOUT}, Days: {ROLLOUT_DAYS}, Steps: {NUM_ROLLOUT_STEPS}")
    print(f"[Config] Guidance Mode (for direct_optim): {GUIDANCE_MODE}, First Step Only: {GUIDANCE_ON_FIRST_STEP}")
    if GUIDANCE_METHOD == "input_manipulation":
        print(f"[Config] Intensity Scaling: {len(INTENSITY_SCALING_CONFIGS)} operations")
        print(f"[Config] Spatial Shift: {len(SHIFT_CONFIGS)} operations")
    elif GUIDANCE_METHOD == "local_affine":
        print(f"[Config] Warp enabled: {WARP_CONFIGS is not None and WARP_CONFIGS.get('enabled', False)}")
    print(f"[Config] Visual Vars: {VISUAL_VARS}")
    print("=" * 80)


    # ═════════════════════════════════════════════════════════════════════════════
    # MAIN EXECUTION LOOP
    # ═════════════════════════════════════════════════════════════════════════════
    
    # 准备上下文（加载模型和数据集）
    print("\n[Main] Preparing context...")
    t_prepare = time.time()
    ctx = prepare_context(G)
    print(f"[Main] Context ready in {time.time() - t_prepare:.2f}s\n")


    # 遍历时间点
    for i, want_time in enumerate(SWEEP_WANT_TIMES):
        print(f"\n{'='*80}")
        print(f"[TIME] {want_time}")
        print(f"[TARGETS] {len(MANUAL_TARGETS)} target(s): {MANUAL_TARGETS}")
        print(f"{'='*80}")
        
        end_date = END_DATES[i] if i < len(END_DATES) else (END_DATES[0] if END_DATES else None)
        selected_storm_id = SELECTED_STORM_IDS[i] if i < len(SELECTED_STORM_IDS) else (SELECTED_STORM_IDS[0] if SELECTED_STORM_IDS else None)
        
        # 加载数据（每个时间点只加载一次，支持本地缓存）
        print("\n[Data] Loading eval batch from ERA5...")
        t0 = time.time()
        _batch_cached = load_era5_batch_cache(G.era5_cache_dir, want_time)
        if _batch_cached is not None:
            eval_inputs, eval_targets, eval_forcings, one_hot_original, ts = _batch_cached
            print(f"[Data] Loaded from cache in {time.time() - t0:.2f}s")
        else:
            wanted_loader = build_wanted_subloader(ctx.eval_ds, [want_time], batch_size=1)
            eval_inputs, eval_targets, eval_forcings, one_hot_original, ts = next(iter(wanted_loader))
            print(f"[Data] Downloaded from GCS in {time.time() - t0:.2f}s")
            save_era5_batch_cache(G.era5_cache_dir, want_time,
                                  eval_inputs, eval_targets, eval_forcings, one_hot_original, ts)
        
        # 加载扩展的 forcings（用于 rollout，支持本地缓存）
        if ENABLE_ROLLOUT:
            print(f"[Data] Loading extended forcings for {NUM_ROLLOUT_STEPS} steps...")
            t_forcings = time.time()
            _forcings_cached = load_era5_forcings_cache(G.era5_cache_dir, want_time, NUM_ROLLOUT_STEPS)
            if _forcings_cached is not None:
                forcings_extended = _forcings_cached
                print(f"[Data] Forcings loaded from cache in {time.time() - t_forcings:.2f}s")
            else:
                forcings_extended = load_forcings_for_rollout(
                    ctx.eval_ds.ds, want_time, NUM_ROLLOUT_STEPS, ctx.task_config
                )
                print(f"[Data] Forcings downloaded from GCS in {time.time() - t_forcings:.2f}s")
                save_era5_forcings_cache(G.era5_cache_dir, want_time, NUM_ROLLOUT_STEPS, forcings_extended)
        else:
            forcings_extended = None
        
        # 处理目标位置
        targets_to_process = (
            [{"lat": 0.0, "lon": 0.0, "radius": 1}]  # 占位符（delete_one 模式不需要）
            if GUIDANCE_MODE == "delete_one" and len(MANUAL_TARGETS) == 0
            else MANUAL_TARGETS
        )
        
        # 遍历每个目标位置
        for target_idx, manual_target in enumerate(targets_to_process):
            print(f"\n[TARGET {target_idx+1}/{len(targets_to_process)}] {manual_target}")
            current_targets = [manual_target]
            
            # 运行或加载 baseline（每个 (want_time, manual_target) 组合只运行一次）
            # 注意：baseline 实际上不依赖目标位置，但为了目录结构清晰，我们为每个目标单独运行
            # [ROLLOUT] 在rollout模式下，不需要单步预测的baseline，rollout有自己的baseline
            if not G.only_guided and not (ENABLE_ROLLOUT and forcings_extended is not None):
                print(f"\n[Baseline] Running or loading baseline for target {target_idx+1}...")
                preds_A, readouts_A, gt_A, out_A = run_or_load_baseline(
                    ctx=ctx,
                    want_time=want_time,
                    manual_targets=current_targets,  # 传入单个目标的列表
                    eval_inputs=eval_inputs,
                    eval_targets=eval_targets,
                    eval_forcings=eval_forcings,
                    one_hot_original=one_hot_original,
                    readout_collect_steps_for_vis=G.readout_collect_steps_for_vis,
                    baseline_cache_dir=G.baseline_cache_dir,
                    output_dir=G.output_dir,
                    save_mode=G.save_mode,
                    epoch=G.epoch,
                    guidance_mode=GUIDANCE_MODE,  # 新增：传递 guidance_mode
                    visual_vars=VISUAL_VARS,
                )
            else:
                preds_A, readouts_A, gt_A, out_A = None, None, None, None
            
            # 参数循环（复用 baseline 结果）
            config_count = 0  # 用于跟踪配置数量，定期清理内存
            for guide_config_idx, guide_config in enumerate(SWEEP_GUIDE_OPT_CONFIGS):
                step_idxs, steps_map, max_opt_steps = parse_guide_config(guide_config)
                for lr_idx, lr in enumerate(SWEEP_GUIDE_LR):
                    for seed_idx, random_seed in enumerate(SWEEP_RANDOM_SEEDS):
                        config_count += 1
                        # 生成 tag：显示配置摘要（包含目标位置信息和seed）
                        config_str = "_".join(f"{s}-{e}x{n}" for s, e, n in guide_config)
                        target_slug = _targets_slug(current_targets)
                        tag = f"time[{_short_time_label(want_time)}]__target_{target_slug}__cfg[{config_str}]_lr{lr}__seed{random_seed}"
                        print(f"\n===== [RUN] {tag} =====")
                        print(f"  Target: {manual_target}")
                        print(f"  Random Seed: {random_seed}")
                        print(f"  step_idxs: {step_idxs}")
                        print(f"  steps_map: {steps_map}")
                        print(f"  max_opt_steps: {max_opt_steps}")
                    
                        # ===== [ROLLOUT] 根据配置选择单步或多步推理 =====
                        if ENABLE_ROLLOUT and forcings_extended is not None:
                            print(f"\n[Mode] ROLLOUT ({NUM_ROLLOUT_STEPS} steps)")
                            
                            # 创建时间记录器（为每个配置创建）
                            # 先决定输出目录，以便保存timing log
                            temp_paths = decide_output_dirs(
                                output_dir=G.output_dir, save_mode=G.save_mode, want_time=want_time,
                                manual_targets=current_targets, readout_collect_steps_for_vis=G.readout_collect_steps_for_vis,
                                inner_idxs=list(step_idxs), inner_steps_map=dict(steps_map),
                                max_opt_steps=int(max_opt_steps), inner_lr=float(lr),
                                strength=float(G.guidance_strength), guidance_mode=GUIDANCE_MODE,
                                random_seed=random_seed,
                                guidance_method=GUIDANCE_METHOD,
                                intensity_scaling_configs=INTENSITY_SCALING_CONFIGS,
                                shift_configs=SHIFT_CONFIGS,
                                warp_configs=WARP_CONFIGS,
                            )
                            timing_log_path = os.path.join(temp_paths["root"], "timing_log.txt")
                            timing_logger = TimingLogger(output_path=timing_log_path)
                            
                            res = run_rollout_with_context(
                            ctx,
                            save_mode=G.save_mode,
                            output_dir=G.output_dir,
                            epoch=G.epoch,
                            only_guided=G.only_guided,
                            only_baseline=G.only_baseline,
                            want_time=want_time,
                            end_date=end_date,
                            # Rollout专用参数
                            num_rollout_steps=NUM_ROLLOUT_STEPS,
                            guidance_on_first_step=GUIDANCE_ON_FIRST_STEP,
                            # Guidance参数
                            manual_targets=current_targets,
                            guidance_mode=GUIDANCE_MODE,
                            selected_storm_id=selected_storm_id,
                            guidance_mask_padding=GUIDANCE_MASK_PADDING,
                            use_guidance_mask=USE_GUIDANCE_MASK,
                            guidance_strength=G.guidance_strength,
                            readout_collect_steps_for_vis=G.readout_collect_steps_for_vis,
                            guide_inner_opt_step_idxs=step_idxs,
                            guide_inner_opt_steps_map=steps_map,
                            guide_max_opt_steps=max_opt_steps,
                            guide_inner_opt_lr=float(lr),
                            guide_loss_type=G.guide_loss_type,
                            guide_normalize_grad=G.guide_normalize_grad,
                            guide_eps=G.guide_eps,
                            visual_vars=VISUAL_VARS,
                            # 可视化参数
                            vis_n_procs=VIS_N_PROCS,
                            vis_contourf_levels=VIS_CONTOURF_LEVELS,
                            vis_coastlines_resolution=VIS_COASTLINES_RESOLUTION,
                            vis_draw_gridlabels=VIS_DRAW_GRIDLABELS,
                            vis_add_borders=VIS_ADD_BORDERS,
                            vis_dpi=VIS_DPI,
                            # 预加载的数据
                            eval_inputs=eval_inputs,
                            eval_targets=eval_targets,
                            forcings_extended=forcings_extended,  # 使用扩展的forcings
                            one_hot_original=one_hot_original,
                            # 时间记录器
                            timing_logger=timing_logger,
                            # Random seed
                            random_seed=random_seed,
                            # Guidance Method
                            guidance_method=GUIDANCE_METHOD,
                            # Direct Intensity Scaling
                            intensity_scaling_configs=INTENSITY_SCALING_CONFIGS,
                            # Spatial Shift
                            shift_configs=SHIFT_CONFIGS,
                            # Local Affine Warp
                            warp_configs=WARP_CONFIGS,
                            # ERA5 本地缓存
                            era5_cache_dir=G.era5_cache_dir,
                        )
                        else:
                            print(f"\n[Mode] SINGLE-STEP (original behavior)")
                            res = run_once_with_context(
                                ctx,
                                save_mode=G.save_mode,
                                output_dir=G.output_dir,
                                epoch=G.epoch,
                                only_guided=G.only_guided,
                                only_baseline=G.only_baseline,
                                want_time=want_time,
                                end_date=end_date,
                                manual_targets=current_targets,  # 传入单个目标的列表
                                guidance_mode=GUIDANCE_MODE,  # 新增：传递模式
                                selected_storm_id=selected_storm_id,
                                guidance_mask_padding=GUIDANCE_MASK_PADDING,
                                use_guidance_mask=USE_GUIDANCE_MASK,
                                guidance_strength=G.guidance_strength,
                                readout_collect_steps_for_vis=G.readout_collect_steps_for_vis,
                                guide_inner_opt_step_idxs=step_idxs,
                                guide_inner_opt_steps_map=steps_map,
                                guide_max_opt_steps=max_opt_steps,
                                guide_inner_opt_lr=float(lr),
                                guide_loss_type=G.guide_loss_type,
                                guide_normalize_grad=G.guide_normalize_grad,
                                guide_eps=G.guide_eps,
                                visual_vars=VISUAL_VARS,
                                # 传入 baseline 结果（如果存在）
                                baseline_preds=preds_A,
                                baseline_readouts=readouts_A,
                                baseline_gt=gt_A,
                                baseline_out_A=out_A,
                                # 传入预加载的数据（避免重复加载）
                                eval_inputs=eval_inputs,
                                eval_targets=eval_targets,
                                eval_forcings=eval_forcings,
                                one_hot_original=one_hot_original,
                            )
                        print("[RESULT]", json.dumps(res, indent=2, ensure_ascii=False))
                        
                        # ===== 内存清理：每个配置运行后立即清理 =====
                        # 释放结果变量
                        del res
                        if 'timing_logger' in locals():
                            del timing_logger
                        
                        # 强制垃圾回收
                        gc.collect()
                        
                        # 每 3 个配置清理一次 JAX 缓存（避免过于频繁影响性能）
                        if config_count % 3 == 0:
                            try:
                                jax.clear_backends()
                                print(f"  [Memory] Cleared JAX backends cache (after {config_count} configs)")
                            except Exception as e:
                                print(f"  [Memory] Warning: Failed to clear JAX backends: {e}")
                        
                        # 每 5 个配置进行一次更彻底的内存清理
                        if config_count % 5 == 0:
                            try:
                                # 尝试清理 GPU 缓存（如果使用 GPU）
                                _ = jnp.array(0).block_until_ready()
                                print(f"  [Memory] Performed deep memory cleanup (after {config_count} configs)")
                            except Exception as e:
                                pass  # 忽略错误，不影响主流程
            
            # ===== 每个 target 处理完成后进行清理 =====
            # 释放 baseline 结果（如果不再需要）
            if not G.only_guided:
                del preds_A, readouts_A, gt_A
            del current_targets
            gc.collect()
            print(f"  [Memory] Cleaned up after target {target_idx+1}/{len(targets_to_process)}")
        
        # 清理：每个时间点处理完成后
        del eval_inputs, eval_targets, eval_forcings, one_hot_original
        if ENABLE_ROLLOUT and forcings_extended is not None:
            del forcings_extended
        gc.collect()
        try:
            jax.clear_backends()
        except:
            pass
    
    print(f"\n{'='*80}")
    print("[INFO] All experiments completed successfully!")
    print(f"{'='*80}\n")


