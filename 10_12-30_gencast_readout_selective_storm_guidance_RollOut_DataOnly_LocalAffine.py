# ─────────────────────────────────────────────────────────────────────────────
# 10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly_LocalAffine.py
# GraphCast / GenCast ReadOut Selective Storm Guidance - DATA ONLY VERSION
# 
# Main features:
# - Run rollout inference (baseline and guided)
# - Save data in NetCDF format (under rollout_data/ subdirectory)
# - Skip all visualization and video generation
# - Support multiple Guidance modes (mutually exclusive):
#   * none: Baseline only (no guidance)
#   * direct_optim: Directly optimize x_t (original readout-based guidance)
#   * input_manipulation: Input condition modification (Intensity Scaling + Spatial Shift)
#   * local_affine: Local affine transform optimization (optimize warp parameters)
# 
# Usage workflow:
# - Run this script to generate data
# - Run 10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py for visualization
# 
# How to use:
# 1. Modify the parameter configuration section at the bottom of the script (CONFIGURATION SECTION)
# 2. Select GUIDANCE_METHOD (one of four options)
# 3. Run the script: python 10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly_LocalAffine.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# Safety guard: default to "cuda,cpu" if not specified (avoids jax.debug.print failing to find CPU backend)
import os as _os

if "JAX_PLATFORMS" not in _os.environ:
    _os.environ["JAX_PLATFORMS"] = "cuda,cpu"

# Memory optimization: limit JAX memory pre-allocation
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
# Timing Logger
# =========================
class TimingLogger:
    """Timing recorder supporting nested timing and auto-save."""
    
    def __init__(self, output_path: Optional[str] = None):
        self.records = []
        self.stack = []  # for nested timing
        self.output_path = output_path
        self.start_time = time.time()
        self.current_section = None
    
    def start(self, section_name: str):
        """Start recording a timed section."""
        try:
            t = time.time()
            self.stack.append({
                "name": section_name,
                "start_time": t,
                "subsections": []
            })
            return self
        except Exception as e:
            # Print warning on exception but do not interrupt the program
            print(f"[TimingLogger Warning] Error starting section '{section_name}': {e}")
            return self
    
    def end(self, section_name: Optional[str] = None):
        """End the current timed section."""
        if not self.stack:
            # Stack is empty - no matching start; return gracefully
            return None
        
        try:
            t = time.time()
            current = self.stack[-1]
            duration = t - current["start_time"]
            
            # If there are sub-sections, compute pure time (total - sub-section time)
            subsections_time = sum(sub["duration"] for sub in current["subsections"])
            pure_time = max(0, duration - subsections_time)  # ensure non-negative
            
            record = {
                "name": current["name"],
                "duration": duration,
                "pure_time": pure_time,
                "subsections": current["subsections"],
                "start_time": current["start_time"],
                "end_time": t
            }
            
            # Remove from stack
            self.stack.pop()
            
            # If there is a parent section, append to its sub-sections
            if self.stack:
                self.stack[-1]["subsections"].append(record)
            else:
                # Top-level record
                self.records.append(record)
            
            return record
        except Exception as e:
            # Print warning on exception but do not interrupt the program
            print(f"[TimingLogger Warning] Error ending section: {e}")
            if self.stack:
                self.stack.pop()  # at least clean up the stack
            return None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack:
            # Auto-end any unfinished sections
            while self.stack:
                self.end()
        return False
    
    def save(self, output_path: Optional[str] = None):
        """Save timing records to file."""
        save_path = output_path or self.output_path
        if not save_path:
            return
        
        try:
            # Clean up any unfinished records
            while self.stack:
                print(f"[TimingLogger Warning] Cleaning up {len(self.stack)} unfinished timing records")
                self.end()
            
            total_time = time.time() - self.start_time
            
            # Save as text format
            txt_path = save_path if save_path.endswith('.txt') else save_path.replace('.json', '.txt')
            # Ensure directory exists
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
            
            # Save as JSON format (for downstream analysis)
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
        """Manually record a timed section (for multi-process scenarios where time was measured in the worker).
        
        Args:
            section_name: Name of the timed section
            duration: Duration in seconds
            subsections: Optional list of sub-sections
        """
        try:
            t = time.time()
            record = {
                "name": section_name,
                "duration": duration,
                "pure_time": duration - sum(sub.get("duration", 0) for sub in (subsections or [])),
                "subsections": subsections or [],
                "start_time": t - duration,  # back-compute start time
                "end_time": t
            }
            
            # If there is a parent section, append to its sub-sections
            if self.stack:
                self.stack[-1]["subsections"].append(record)
            else:
                # Top-level record
                self.records.append(record)
            
            return record
        except Exception as e:
            print(f"[TimingLogger Warning] Error recording duration for '{section_name}': {e}")
            return None
    
    def _write_record(self, f, record, indent=0):
        """Recursively write a record."""
        prefix = "  " * indent
        f.write(f"{prefix}--- {record['name']} ---\n")
        f.write(f"{prefix}  Duration: {record['duration']:.2f}s")
        if record['pure_time'] > 0.01:  # only show meaningful pure time
            f.write(f" (pure: {record['pure_time']:.2f}s)")
        f.write("\n")
        
        if record['subsections']:
            for sub in record['subsections']:
                self._write_record(f, sub, indent + 1)


# =========================
# Manual Guidance: Coordinate Conversion and Mask Generation
# =========================
def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """Convert lat/lon to 181x360 grid indices.
    
    Args:
        lat: Latitude, range -90 to 90
        lon: Longitude, range 0 to 360 (or -180 to 180)
    
    Returns:
        (row, col): Grid indices, row ∈ [0, 180], col ∈ [0, 359]
    """
    # lat: -90 to 90 → row: 0 to 180
    row = int(lat + 90)
    row = max(0, min(180, row))
    # lon: 0 to 360 (or -180 to 180) → col: 0 to 359
    col = int(lon) % 360
    return row, col


def make_manual_guidance_mask(
    batch_size: int,
    targets: List[Dict],
    height: int = 181,
    width: int = 360,
) -> torch.Tensor:
    """Generate guidance mask from user-specified coordinates.
    
    Args:
        batch_size: Batch size
        targets: List of target locations, supporting two formats:
            - lat/lon: [{"lat": 25.0, "lon": 120.0, "radius": 5}, ...]
            - grid index: [{"row": 115, "col": 120, "radius": 5}, ...]
        height: Grid height (default 181)
        width: Grid width (default 360)
    
    Returns:
        one_hot: one-hot tensor of shape (batch_size, height, width, 2)
    """
    centers_with_r = []
    for t in targets:
        if "row" in t and "col" in t:
            # Use grid indices directly
            row, col = t["row"], t["col"]
        else:
            # Convert from lat/lon
            row, col = latlon_to_grid(t["lat"], t["lon"])
        radius = t.get("radius", 5)
        centers_with_r.append((row, col, radius))
    
    # Reuse existing function to generate one-hot mask
    return data_merge_make_circular_one_hot_varradius_cpu(
        batch_size, height, width, centers_with_r
    )


def _targets_slug(targets: List[Dict]) -> str:
    """Generate a short description string for target locations."""
    parts = []
    for t in targets:
        if "row" in t and "col" in t:
            parts.append(f"r{t['row']}c{t['col']}rad{t.get('radius', 5)}")
        else:
            parts.append(f"lat{t['lat']:.1f}lon{t['lon']:.1f}rad{t.get('radius', 5)}")
    return "__".join(parts)


# =========================
# Selective Storm Guidance: Storm Extraction and Selection
# =========================
# Global cache: {end_date: List[Dict]} - caches storm extraction results
_storms_cache = {}


def extract_storms_from_end_date(
    ctx: Context,
    end_date: str,
) -> List[Dict]:
    """Extract all storm information from end_date (with caching).
    
    Args:
        ctx: Context object
        end_date: End date string, e.g. "2017-09-08 00:00:00"
    
    Returns:
        storms: List of dicts, each containing:
            - storm_id: Auto-assigned ID (0, 1, 2, ...)
            - lat: Latitude
            - lon: Longitude
            - rsize: Radius (degrees)
            - row: Grid row index
            - col: Grid column index
            - min_row, max_row, min_col, max_col: Bounding box boundaries
    """
    # Check cache
    if end_date in _storms_cache:
        print(f"    [Extract Storms] Using cached storms for end_date: {end_date}")
        return _storms_cache[end_date]
    
    # Load data for end_date
    print(f"    [Extract Storms] Loading data for end_date: {end_date}...")
    t0 = time.time()
    wanted_loader = build_wanted_subloader(ctx.eval_ds, [end_date], batch_size=1)
    it = iter(wanted_loader)
    print(f"    [Extract Storms] DataLoader created, fetching data from zarr/GCS (this may take 30-60s)...")
    t1 = time.time()
    _, _, _, one_hot_original, _ = next(it)
    print(f"    [Extract Storms] Data loaded in {time.time() - t1:.2f}s (total: {time.time() - t0:.2f}s)")
    
    # Extract storm locations from one_hot_original
    print(f"    [Extract Storms] Extracting storm regions from one-hot mask...")
    one_hot_np = one_hot_original.numpy() if hasattr(one_hot_original, "numpy") else np.asarray(one_hot_original)
    storm_mask = one_hot_np[0, ..., 1]  # (H, W), take first batch
    
    # Use scipy.ndimage.label to extract connected regions (identify storms)
    labeled, num_features = ndimage.label(storm_mask)
    print(f"    [Extract Storms] Found {num_features} storm region(s)")
    
    storms = []
    for storm_id in range(1, num_features + 1):
        # Find all pixels belonging to this storm
        rows, cols = np.where(labeled == storm_id)
        if len(rows) == 0:
            continue
        
        # Compute center location
        center_row = int(rows.mean())
        center_col = int(cols.mean())
        
        # Compute bounding box
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())
        
        # Estimate radius (half the diagonal of the bounding box)
        radius = max(max_row - min_row, max_col - min_col) // 2
        radius = max(1, radius)  # at least 1
        
        # Convert to lat/lon
        lat = center_row - 90
        lon = center_col
        
        storms.append({
            "storm_id": storm_id - 1,  # 0-indexed
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
    
    # Cache results
    _storms_cache[end_date] = storms
    print(f"    [Extract Storms] Cached storms for end_date: {end_date}")
    
    return storms


def print_storms_list(storms: List[Dict]):
    """Print all storms for user inspection."""
    print("\n=== Available Storms at End Date ===")
    for storm in storms:
        print(f"  Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, "
              f"rsize={storm['rsize']:.1f}, row={storm['row']}, col={storm['col']}")
    print()


# Global cache: {(start_date, end_date): storm_id}
_storm_id_cache = {}


def get_storm_id_for_date_pair(
    start_date: str,
    end_date: str,
    config_storm_id: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> int:
    """Get storm_id with priority: config > cache > user input.
    
    Args:
        start_date: Input time point
        end_date: Target time point
        config_storm_id: Storm ID specified in config (if provided)
        ctx: Context object (used to extract storm info)
    
    Returns:
        selected_id: Selected storm_id
    """
    cache_key = (start_date, end_date)
    
    # Case A: already specified in config
    if config_storm_id is not None:
        _storm_id_cache[cache_key] = config_storm_id
        print(f"    [Storm Selection] Using configured storm_id: {config_storm_id}")
        return config_storm_id
    
    # Case B: check cache
    if cache_key in _storm_id_cache:
        cached_id = _storm_id_cache[cache_key]
        print(f"    [Storm Selection] Using cached storm_id: {cached_id}")
        return cached_id
    
    # Case C: requires user input (only once)
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
    
    # Cache selection
    _storm_id_cache[cache_key] = selected_id
    print(f"    [Storm Selection] Selected storm_id {selected_id} cached for ({start_date}, {end_date})")
    return selected_id


def extract_storms_from_one_hot(one_hot: torch.Tensor) -> List[Dict]:
    """Extract all storm information from a one_hot mask (general purpose).
    
    Args:
        one_hot: one-hot tensor of shape (B, H, W, 2) or (H, W, 2)
    
    Returns:
        storms: List of dicts, each containing storm information
    """
    one_hot_np = one_hot.numpy() if hasattr(one_hot, "numpy") else np.asarray(one_hot)
    if one_hot_np.ndim == 4:
        storm_mask = one_hot_np[0, ..., 1]  # (H, W), take first batch
    else:
        storm_mask = one_hot_np[..., 1]  # (H, W)
    
    # Use scipy.ndimage.label to extract connected regions
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
    distance_threshold: float = 10.0,  # degrees
) -> Optional[int]:
    """Match storms between start_date and end_date by spatial distance.
    
    Args:
        start_storms: List of all storms at start_date
        end_storm: Selected storm at end_date
        distance_threshold: Maximum matching distance (degrees)
    
    Returns:
        matched_storm_id: Corresponding storm_id in start_date, or None if not found
    """
    end_lat, end_lon = end_storm["lat"], end_storm["lon"]
    
    best_match_id = None
    min_distance = float('inf')
    
    for start_storm in start_storms:
        start_lat, start_lon = start_storm["lat"], start_storm["lon"]
        # Simple Euclidean distance (degrees)
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
    mode: str,  # mode parameter "steer_one" or "delete_one"
    selected_storm_id_in_start: Optional[int],
    manual_target: Optional[Dict] = None,  # required for "steer_one", not needed for "delete_one"
    batch_size: int = 1,
    height: int = 181,
    width: int = 360,
) -> torch.Tensor:
    """Build the final target mask, supporting two modes:
    
    Mode 1 "steer_one":
    - Extract all storms from start_date
    - Replace selected storm with manual target (move location)
    - Keep all other storms
    
    Mode 2 "delete_one":
    - Extract all storms from start_date
    - Delete selected storm
    - Keep all other storms
    
    Args:
        one_hot_start: Original one-hot mask at start_date (B, H, W, 2)
        start_storms: List of all storms at start_date
        mode: Mode selection "steer_one" or "delete_one"
        selected_storm_id_in_start: ID of the storm to operate on (in start_date)
        manual_target: User-specified target location {"lat": ..., "lon": ..., "radius": ...} (only for "steer_one")
        batch_size: Batch size
        height: Grid height
        width: Grid width
    
    Returns:
        one_hot_final: Final one-hot mask of shape (B, H, W, 2)
    """
    # Extract storm mask from start_date one_hot
    one_hot_np = one_hot_start.numpy() if hasattr(one_hot_start, "numpy") else np.asarray(one_hot_start)
    storm_mask_start = one_hot_np[0, ..., 1]  # (H, W)
    labeled_start, num_features_start = ndimage.label(storm_mask_start)
    
    # Create new mask (initialized to 0)
    final_mask = np.zeros((height, width), dtype=np.int64)
    
    # Add all start_date storms except the one to operate on
    for storm in start_storms:
        storm_id_in_labeled = storm["storm_id"] + 1  # labeled starts at 1
        if selected_storm_id_in_start is not None and storm["storm_id"] == selected_storm_id_in_start:
            # Skip the storm to operate on
            if mode == "steer_one":
                print(f"    [Build Target Mask] Skipping start_date storm_id {storm['storm_id']} "
                      f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f}) - will be replaced by manual target")
            elif mode == "delete_one":
                print(f"    [Build Target Mask] Removing start_date storm_id {storm['storm_id']} "
                      f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f})")
            continue
        # Add this storm's region
        final_mask |= (labeled_start == storm_id_in_labeled)
        print(f"    [Build Target Mask] Keeping start_date storm_id {storm['storm_id']} "
              f"(lat={storm['lat']:.1f}, lon={storm['lon']:.1f})")
    
    # If in "steer_one" mode, add the manual target storm
    if mode == "steer_one":
        if manual_target is None:
            raise ValueError("manual_target is required for 'steer_one' mode")
        if "row" in manual_target and "col" in manual_target:
            target_row, target_col = manual_target["row"], manual_target["col"]
        else:
            target_row, target_col = latlon_to_grid(manual_target["lat"], manual_target["lon"])
        target_radius = manual_target.get("radius", 5)
        
        # Create circular mask
        yy = np.arange(height)[:, None]
        xx = np.arange(width)[None, :]
        target_circle = ((yy - target_row)**2 + (xx - target_col)**2 <= target_radius**2)
        final_mask |= target_circle
        
        print(f"    [Build Target Mask] Added manual target at (lat={manual_target.get('lat', 'N/A')}, "
              f"lon={manual_target.get('lon', 'N/A')}, row={target_row}, col={target_col}, radius={target_radius})")
    elif mode == "delete_one":
        # "delete_one" mode: nothing to add, just keep other storms
        num_kept = len(start_storms) - (1 if selected_storm_id_in_start is not None else 0)
        print(f"    [Build Target Mask] Deleted storm_id {selected_storm_id_in_start}, kept {num_kept} other storm(s)")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'steer_one' or 'delete_one'")
    
    # Convert to one-hot
    labels = np.broadcast_to(final_mask, (batch_size, height, width))
    labels_t = torch.from_numpy(labels)
    one_hot_final = torch.nn.functional.one_hot(labels_t, num_classes=2)
    
    return one_hot_final


def compute_bounding_box_mask(
    selected_storm: Dict,
    target_storm: Dict,  # user-specified target location
    batch_size: int,
    height: int = 181,
    width: int = 360,
    padding: int = 5,  # padding around the bounding box
) -> torch.Tensor:
    """Compute a bounding box mask covering the selected original storm and the target storm.
    
    Args:
        selected_storm: Selected original storm info (extracted from end_date)
        target_storm: Target storm info (generated from manual_targets)
        batch_size: Batch size
        height: Grid height
        width: Grid width
        padding: Padding around the bounding box (grid points)
    
    Returns:
        guidance_mask: (B, H, W) binary mask
    """
    # Get boundaries of the selected original storm
    storm1_min_row = selected_storm.get("min_row", selected_storm["row"] - int(selected_storm["rsize"]))
    storm1_max_row = selected_storm.get("max_row", selected_storm["row"] + int(selected_storm["rsize"]))
    storm1_min_col = selected_storm.get("min_col", selected_storm["col"] - int(selected_storm["rsize"]))
    storm1_max_col = selected_storm.get("max_col", selected_storm["col"] + int(selected_storm["rsize"]))
    
    # Boundaries of the target storm (generated from manual_targets)
    if "row" in target_storm and "col" in target_storm:
        target_row, target_col = target_storm["row"], target_storm["col"]
    else:
        target_row, target_col = latlon_to_grid(target_storm["lat"], target_storm["lon"])
    target_radius = target_storm.get("radius", 5)
    
    storm2_min_row = target_row - target_radius
    storm2_max_row = target_row + target_radius
    storm2_min_col = target_col - target_radius
    storm2_max_col = target_col + target_radius
    
    # Compute joint bounding box
    min_row = max(0, min(storm1_min_row, storm2_min_row) - padding)
    max_row = min(height - 1, max(storm1_max_row, storm2_max_row) + padding)
    min_col = max(0, min(storm1_min_col, storm2_min_col) - padding)
    max_col = min(width - 1, max(storm1_max_col, storm2_max_col) + padding)
    
    # Generate mask
    mask = np.zeros((batch_size, height, width), dtype=np.float32)
    mask[:, min_row:max_row+1, min_col:max_col+1] = 1.0
    
    print(f"    [Bounding Box] rows [{min_row}-{max_row}], cols [{min_col}-{max_col}], "
          f"size: {mask.sum().item()} pixels")
    
    return torch.from_numpy(mask)


# =========================
# Loss Monitor: Saving and Visualization
# =========================
def plot_loss_curves(loss_history: Dict[int, List[float]], output_path: str):
    """Plot loss descent curves for each denoising step.
    
    Args:
        loss_history: {step: [loss_0, loss_1, ..., loss_n]}
        output_path: Full path to save the figure
    """
    import matplotlib.pyplot as plt
    
    num_steps = len(loss_history)
    if num_steps == 0:
        print("[Loss Curve] No loss data to plot")
        return
    
    # Create subplots: one per step
    fig, axes = plt.subplots(
        num_steps, 1,
        figsize=(12, 4 * num_steps),
        squeeze=False
    )
    axes = axes.flatten()
    
    for idx, (step, losses) in enumerate(sorted(loss_history.items())):
        ax = axes[idx]
        iterations = list(range(len(losses)))
        
        # Plot loss curve
        ax.plot(iterations, losses, marker='o', linewidth=2, markersize=4, color='#2E86AB')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title(f'Denoising Step {step} - Guidance Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate highest and lowest points
        max_loss = max(losses)
        min_loss = min(losses)
        max_idx = losses.index(max_loss)
        min_idx = losses.index(min_loss)
        
        # Highest point (red)
        ax.plot(max_idx, max_loss, 'ro', markersize=10, zorder=5)
        ax.text(max_idx, max_loss, f'  Max: {max_loss:.6f}',
                fontsize=11, color='red', fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left')
        
        # Lowest point (green)
        ax.plot(min_idx, min_loss, 'go', markersize=10, zorder=5)
        ax.text(min_idx, min_loss, f'  Min: {min_loss:.6f}',
                fontsize=11, color='green', fontweight='bold',
                verticalalignment='top', horizontalalignment='left')
        
        # Add statistics
        reduction = max_loss - min_loss
        reduction_pct = (reduction / max_loss * 100) if max_loss > 0 else 0
        ax.text(0.02, 0.98, 
                f'Reduction: {reduction:.6f} ({reduction_pct:.2f}%)\nIterations: {len(losses)}',
                transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set x-axis label
        if idx == num_steps - 1:
            ax.set_xlabel('Optimization Iteration', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Loss Curve] Saved to {output_path}")


def save_loss_summary(loss_history: Dict[int, List[float]], output_path: str):
    """Save loss statistics summary to a text file.
    
    Args:
        loss_history: {step: [loss_0, loss_1, ..., loss_n]}
        output_path: Full path to save the text file
    """
    if len(loss_history) == 0:
        print("[Loss Summary] No loss data to save")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Guidance Loss Summary\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall statistics
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
        
        # Detailed statistics per step
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
            
            # Convergence analysis (check variation in the last few iterations)
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
# Mask Visualization: Target Storm vs Guidance Mask
# =========================
def visualize_mask_comparison(
    one_hot_target: torch.Tensor,  # target storm one-hot mask (B, H, W, 2)
    guidance_mask: torch.Tensor,    # bounding box mask (B, H, W)
    output_path: str,
    selected_storm_info: Optional[Dict] = None,
    target_storm_info: Optional[Dict] = None,
    sample_idx: int = 0,
):
    """Side-by-side visualization of target storm mask and guidance mask to verify correctness.
    
    Args:
        one_hot_target: Target storm one-hot mask, shape (B, H, W, 2)
        guidance_mask: Bounding box mask, shape (B, H, W)
        output_path: Full path to save the figure
        selected_storm_info: Selected original storm info (for title display)
        target_storm_info: Target storm info (for title display)
        sample_idx: Which batch sample to use (default 0)
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    
    # Convert to numpy
    one_hot_np = one_hot_target.numpy() if hasattr(one_hot_target, "numpy") else np.asarray(one_hot_target)
    guidance_np = guidance_mask.numpy() if hasattr(guidance_mask, "numpy") else np.asarray(guidance_mask)
    
    # Take the specified batch sample
    target_mask = one_hot_np[sample_idx, ..., 1]  # (H, W), take positive class channel
    bbox_mask = guidance_np[sample_idx, ...]          # (H, W)
    
    # Validate data (debug info)
    print(f"    [Mask Debug] target_mask shape: {target_mask.shape}, min: {target_mask.min():.4f}, max: {target_mask.max():.4f}, sum: {target_mask.sum():.4f}")
    print(f"    [Mask Debug] bbox_mask shape: {bbox_mask.shape}, min: {bbox_mask.min():.4f}, max: {bbox_mask.max():.4f}, sum: {bbox_mask.sum():.4f}")
    
    # Create coordinates (standard 181x360 grid)
    height, width = target_mask.shape
    lats = np.linspace(-90, 90, height)
    lons = np.linspace(0, 360, width)
    
    # Handle longitude wrap-around (0-360)
    target_mask_cyclic, lons_cyclic = add_cyclic_point(target_mask, coord=lons)
    bbox_mask_cyclic, _ = add_cyclic_point(bbox_mask, coord=lons)
    
    # Create figure: 1 row x 2 cols, using central_longitude=180 for 0-360 longitude range
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        subplot_kw={'projection': projection}
    )
    
    def setup_ax(ax):
        """Set up the map background for a single subplot."""
        ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
    
    # Left panel: Target Storm One-hot Mask
    setup_ax(ax1)
    target_masked = np.ma.masked_where(target_mask_cyclic < 0.01, target_mask_cyclic)
    im1 = ax1.contourf(
        lons_cyclic, lats, target_masked,  # use 1D arrays; contourf broadcasts automatically
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
    
    # Right panel: Guidance Bounding Box Mask
    setup_ax(ax2)
    # Draw bounding box mask (blue region)
    im2 = ax2.contourf(
        lons_cyclic, lats, bbox_mask_cyclic,  # use 1D arrays
        levels=[0, 0.5, 1.0],
        colors=["white", "blue"],
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    # Overlay target mask contour (red dashed line)
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
    
    # Overall title
    suptitle = "Mask Comparison: Target Storm vs Guidance Region"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [Mask Visualization] Saved to {output_path}")


# =========================
# Baseline Cache Management (disk cache)
# =========================
def _get_baseline_cache_path(
    cache_dir: str,
    want_time: str,
    manual_targets: List[Dict],
    readout_collect_steps_for_vis: Tuple[int, ...],
) -> str:
    """Generate baseline cache file path.
    
    Args:
        cache_dir: Cache directory
        want_time: Time point string
        manual_targets: List of target locations
        readout_collect_steps_for_vis: Readout collection steps
    
    Returns:
        Cache file path
    """
    time_slug = _short_time_label(want_time)
    targets_slug = _targets_slug(manual_targets)
    steps_slug = "-".join(str(s) for s in sorted(readout_collect_steps_for_vis))
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"baseline_{time_slug}__{targets_slug}__steps{steps_slug}.pkl"
    return os.path.join(cache_dir, cache_file)


def _load_baseline_cache(cache_path: str) -> Optional[Tuple[xr.Dataset, Dict[str, xr.Dataset], xr.Dataset]]:
    """Load baseline cache from disk.
    
    Returns:
        (preds_A, readouts_A, gt_A) if successful, otherwise None
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
    """Save baseline results to disk.
    
    Args:
        cache_path: Cache file path
        preds_A: Baseline prediction results
        readouts_A: Baseline readout results
        gt_A: Ground truth
    """
    try:
        print(f"    [Baseline Cache] Saving to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump((preds_A, readouts_A, gt_A), f)
        print(f"    [Baseline Cache] Successfully saved")
    except Exception as e:
        print(f"    [Baseline Cache] Failed to save: {e}")


# =========================
# Rollout-specific Utility Functions
# =========================
def load_forcings_for_rollout(
    full_ds: xr.Dataset,
    start_time: str,
    num_steps: int,
    task_config,
) -> xr.Dataset:
    """
    [ROLLOUT ONLY] Load all forcings required for rollout from the full ERA5 dataset.
    
    Rollout requires multi-step forcing data (e.g. TOA incident solar radiation).
    This function extracts forcings for the specified time range from the raw dataset.
    
    Args:
        full_ds: Raw ERA5 dataset (xr.Dataset)
        start_time: Start time string, e.g. "2017-09-07 00:00:00"
        num_steps: Number of rollout steps (e.g. 10 steps = 5 days at 12h per step)
        task_config: Task configuration (contains forcing_variables)
    
    Returns:
        forcings_extended: (batch=1, time=num_steps, lat, lon, ...)
                          Time coordinate in timedelta: [12h, 24h, ..., num_steps*12h]
    
    Example:
        >>> forcings = load_forcings_for_rollout(
        ...     ctx.eval_ds.ds, "2017-09-07 00:00:00", 10, ctx.task_config
        ... )
        >>> forcings.dims  # {'batch': 1, 'time': 10, 'lat': 181, 'lon': 360, ...}
    """
    from graphcast import data_utils
    
    start_ts = pd.Timestamp(start_time)
    
    # Build list of required time points (one every 12 hours)
    time_points = [
        start_ts + pd.Timedelta(hours=12 * (i + 1))
        for i in range(num_steps)
    ]
    
    print(f"    [Rollout Forcings] Loading {num_steps} time steps from dataset...")
    print(f"    [Rollout Forcings] Time range: {time_points[0]} to {time_points[-1]}")
    
    # Select these time points from the dataset
    try:
        forcings_window = full_ds.sel(time=time_points).load()
    except KeyError as e:
        ds_min = pd.Timestamp(full_ds.time.min().values)
        ds_max = pd.Timestamp(full_ds.time.max().values)
        raise ValueError(
            f"Some time points are outside the dataset range.\n"
            f"Required: {time_points[0]} to {time_points[-1]}\n"
            f"Dataset range: {ds_min} to {ds_max}\n"
            f"Original error: {e}"
        )
    
    # Rename coordinates (convert from raw ERA5 format to GenCast format)
    # Raw dataset uses longitude/latitude, but add_derived_vars expects lon/lat
    rename_dict = {}
    if "longitude" in forcings_window.coords or "longitude" in forcings_window.dims:
        rename_dict["longitude"] = "lon"
    if "latitude" in forcings_window.coords or "latitude" in forcings_window.dims:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        forcings_window = forcings_window.rename(rename_dict)
    
    # Downsample to 1-degree resolution (raw ERA5 is 0.25°, sample every 4 points)
    # This is consistent with the dataset's transform_sample
    if "lat" in forcings_window.dims and "lon" in forcings_window.dims:
        lat_size = forcings_window.sizes.get("lat", 0)
        lon_size = forcings_window.sizes.get("lon", 0)
        # Downsample if resolution is finer than 1° (181x360)
        if lat_size > 181 or lon_size > 360:
            print(f"    [Rollout Forcings] Downsampling from {lat_size}x{lon_size} to 1° resolution...")
            forcings_window = forcings_window.isel(
                lat=slice(0, None, 4),  # sample every 4 points
                lon=slice(0, None, 4)
            )
            # Flip lat dimension (south-to-north → north-to-south)
            forcings_window = forcings_window.isel(lat=slice(None, None, -1))
            print(f"    [Rollout Forcings] Downsampled to {forcings_window.sizes.get('lat', 0)}x{forcings_window.sizes.get('lon', 0)}")
    
    # Add batch dimension (required by add_derived_vars)
    if "batch" not in forcings_window.dims:
        forcings_window = forcings_window.expand_dims("batch", axis=0)
    
    # Add datetime coordinate (required by add_derived_vars)
    if "datetime" not in forcings_window.coords:
        # Create datetime coordinate from time coordinate
        time_values = forcings_window.coords["time"].values
        if isinstance(time_values[0], np.datetime64):
            datetime_values = time_values
        else:
            # If time is timedelta, convert to absolute time
            datetime_values = np.array([start_ts + pd.Timedelta(hours=12 * (i + 1)) 
                                       for i in range(num_steps)], dtype="datetime64[ns]")
        
        # Create datetime coordinate (must match the time dimension)
        if "batch" in forcings_window.dims:
            datetime_coord = (("batch", "time"), datetime_values[np.newaxis, :])
        else:
            datetime_coord = ("time", datetime_values)
        forcings_window = forcings_window.assign_coords(datetime=datetime_coord)
    
    # Add derived variables (year_progress_sin, day_progress_sin, etc.)
    # These must be computed from datetime and cannot be read directly from the dataset
    if set(task_config.forcing_variables) & data_utils._DERIVED_VARS:
        data_utils.add_derived_vars(forcings_window)
    
    # Add TISR (if needed)
    if "toa_incident_solar_radiation" in task_config.forcing_variables:
        data_utils.add_tisr_var(forcings_window)
    
    # Extract only forcing variables (e.g. tisr, year_progress_sin)
    forcing_vars = task_config.forcing_variables
    forcings_only = forcings_window[list(forcing_vars)]
    
    # Convert time coordinate to timedelta (relative to start_time)
    time_deltas = [np.timedelta64(12 * (i + 1), 'h') for i in range(num_steps)]
    forcings_only = forcings_only.assign_coords(time=time_deltas)
    
    # Drop datetime coordinate (not needed during rollout)
    if "datetime" in forcings_only.coords:
        forcings_only = forcings_only.drop_vars("datetime", errors="ignore")
    
    print(f"    [Rollout Forcings] Loaded successfully: {forcings_only.dims}")
    
    return forcings_only


# =========================
# Direct Intensity Scaling: Local Variable Scaling
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
    Apply local relative scaling to selected variables within a circular region.
    
    Implements "Idea 1: Direct scaling of input Weather State" — locally amplifying
    or attenuating meteorological variables to test model sensitivity to input intensity.
    
    Uses relative scaling (relative to regional mean):
    - Compute mean within the circular region as baseline
    - new_value = baseline + (original - baseline) × scale_factor
    - This preserves physical plausibility and avoids extreme values
    
    Args:
        preds: Prediction results (batch, time, lat, lon, ...) or (batch, time, lat, lon, level, ...)
        scale_factor: Scale factor (e.g. 1.2 = amplify deviation by 20%, 0.8 = reduce by 20%)
        center_lat: Center latitude of the circular region
        center_lon: Center longitude of the circular region
        radius: Radius of the circular region (degrees)
        variables_to_scale: List of variables to scale; if None, scale all variables
    
    Returns:
        scaled_preds: Scaled prediction results
    
    Example:
        >>> scaled = apply_local_intensity_scale(
        ...     preds, scale_factor=1.2, 
        ...     center_lat=18.0, center_lon=293.0, radius=5.0,
        ...     variables_to_scale=['u_component_of_wind', 'v_component_of_wind']
        ... )
    """
    scaled_preds = preds.copy(deep=True)  # deep copy to avoid modifying original
    
    # Get lat and lon coordinates
    lats = preds.coords['lat'].values
    lons = preds.coords['lon'].values
    
    # Create grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Compute distance (simple Euclidean distance, suitable for small regions)
    distance = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    
    # Create circular mask
    circle_mask = distance <= radius
    
    print(f"    [Direct Intensity] Circle region: center=({center_lat:.1f}°, {center_lon:.1f}°), "
          f"radius={radius:.1f}°, pixels={circle_mask.sum()}")
    
    # Determine variables to scale
    if variables_to_scale is None:
        vars_to_process = list(preds.data_vars)
    else:
        vars_to_process = [v for v in variables_to_scale if v in preds.data_vars]
    
    # Apply relative scaling to each variable (using xarray dimension-safe methods)
    scaled_count = 0
    for var_name in vars_to_process:
        var_data = scaled_preds[var_name]
        
        # Only process variables with lat and lon dimensions
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            # Wrap mask in xarray DataArray to allow automatic dimension alignment
            mask_da = xr.DataArray(
                circle_mask,
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons}
            )
            
            # Relative scaling: compute region mean as baseline
            # Use where to restrict computation to the region, then take mean
            region_data = var_data.where(mask_da)
            region_mean = region_data.mean()
            
            # Relative scaling formula: new_value = baseline + (original - baseline) × scale_factor
            # Effect:
            # - original == baseline → no change
            # - original > baseline → deviation amplified
            # - original < baseline → deviation reduced
            scaled_value = region_mean + (var_data - region_mean) * scale_factor
            
            # Apply scaling only inside the circular region; keep original outside
            scaled_preds[var_name] = xr.where(mask_da, scaled_value, var_data)
            scaled_count += 1
            
            # Print debug info (only for the first variable to avoid excessive output)
            if scaled_count == 1:
                # Extract scalar (region_mean is already scalar since mean() reduces all dims)
                try:
                    region_mean_scalar = float(region_mean.values.item())
                except (AttributeError, ValueError):
                    region_mean_scalar = float(region_mean.values)
                print(f"    [Direct Intensity] Relative scaling: region_mean={region_mean_scalar:.2f}, "
                      f"scale_factor={scale_factor:.2f}")
    
    print(f"    [Direct Intensity] Scaled {scaled_count} variable(s) using relative scaling (factor {scale_factor:.2f})")
    
    return scaled_preds


# =========================
# Spatial Shift: Local Spatial Displacement
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
    Apply spatial displacement to a circular region of the prediction.
    
    Implements "Idea 2: Regional Spatial Shift" — moves meteorological data
    within a circular region to a new location, overwriting the target area
    and filling the source "hole" with interpolation from surrounding values.
    
    Workflow:
    1. Define source circular region (center, radius)
    2. Compute target region position (center + delta)
    3. Extract source region data and copy to target location
    4. Fill the source location "hole" with interpolation
    
    Args:
        preds: Prediction results (batch, time, lat, lon, ...) or (batch, time, lat, lon, level, ...)
        center_lat: Center latitude of the source circular region
        center_lon: Center longitude of the source circular region
        radius: Radius of the circular region (degrees)
        delta_lat: Latitude displacement (degrees, positive = northward)
        delta_lon: Longitude displacement (degrees, positive = eastward)
        variables_to_shift: List of variables to shift; if None, shift all variables
        interpolation_method: Interpolation method ("linear", "nearest", "cubic")
    
    Returns:
        shifted_preds: Shifted prediction results
    
    Example:
        >>> shifted = apply_local_spatial_shift(
        ...     preds, 
        ...     center_lat=18.0, center_lon=293.0, radius=5.0,
        ...     delta_lat=2.0, delta_lon=3.0,  # shift 2 degrees north, 3 degrees east
        ...     variables_to_shift=['u_component_of_wind', 'v_component_of_wind']
        ... )
    """
    from scipy.interpolate import griddata
    
    shifted_preds = preds.copy(deep=True)  # deep copy to avoid modifying original
    
    # Get lat and lon coordinates
    lats = preds.coords['lat'].values
    lons = preds.coords['lon'].values
    
    # Create grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Compute source and target region masks
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
    
    # Determine variables to shift
    if variables_to_shift is None:
        vars_to_process = list(preds.data_vars)
    else:
        vars_to_process = [v for v in variables_to_shift if v in preds.data_vars]
    
    # Apply spatial shift to each variable
    shifted_count = 0
    for var_name in vars_to_process:
        var_data = shifted_preds[var_name]
        
        # Only process variables with lat and lon dimensions
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            # Get all dimensions of the variable
            dims = var_data.dims
            
            # Get numpy array directly
            var_values = var_data.values
            
            # Handle different dimension layouts:
            # Could be (batch, time, lat, lon), (batch, time, lat, lon, level), or (batch, time, level, lat, lon)
            if 'level' in dims:
                # Has level dimension: process each level separately
                # Find axis positions
                level_axis = dims.index('level')
                lat_axis = dims.index('lat')
                lon_axis = dims.index('lon')
                num_levels = var_data.sizes['level']
                
                print(f"    [Spatial Shift] Processing variable '{var_name}' with level dimension")
                print(f"      Original shape: {var_values.shape}, dims: {dims}")
                print(f"      level_axis={level_axis}, lat_axis={lat_axis}, lon_axis={lon_axis}")
                
                # Process each level separately
                for level_idx in range(num_levels):
                    # Build index slices
                    idx = [slice(None)] * var_values.ndim
                    idx[level_axis] = level_idx
                    
                    # Extract data for this level
                    var_slice = var_values[tuple(idx)]
                    print(f"      Level {level_idx}: extracted shape = {var_slice.shape}")
                    
                    # Recompute lat/lon axis positions after removing level axis
                    # If level was before lat, lat index decreases by 1
                    lat_pos_in_slice = lat_axis if lat_axis < level_axis else lat_axis - 1
                    lon_pos_in_slice = lon_axis if lon_axis < level_axis else lon_axis - 1
                    
                    print(f"      After level extraction: lat at axis {lat_pos_in_slice}, lon at axis {lon_pos_in_slice}")
                    
                    # Move lat and lon to the last two dimensions
                    # Target format: (..., lat, lon)
                    var_slice_moved = np.moveaxis(var_slice, [lat_pos_in_slice, lon_pos_in_slice], [-2, -1])
                    print(f"      After moveaxis: {var_slice_moved.shape}")
                    
                    # Shift this level
                    shifted_slice = _shift_2d_field(
                        var_slice_moved,
                        source_mask,
                        target_mask,
                        lats,
                        lons,
                        interpolation_method
                    )
                    print(f"      After shift: {shifted_slice.shape}")
                    
                    # Move lat and lon back to their original positions
                    shifted_slice_moved_back = np.moveaxis(shifted_slice, [-2, -1], [lat_pos_in_slice, lon_pos_in_slice])
                    print(f"      After moveaxis back: {shifted_slice_moved_back.shape}")
                    
                    # Write back
                    var_values[tuple(idx)] = shifted_slice_moved_back
                
                # Update the entire variable at once
                shifted_preds[var_name] = (dims, var_values)
            else:
                # No level dimension: process directly
                print(f"    [Spatial Shift] Processing variable '{var_name}' without level dimension")
                print(f"      Original shape: {var_values.shape}, dims: {dims}")
                
                # Find lat and lon axis positions
                lat_axis = dims.index('lat')
                lon_axis = dims.index('lon')
                print(f"      lat_axis={lat_axis}, lon_axis={lon_axis}")
                
                # Move lat and lon to the last two dimensions
                var_values_moved = np.moveaxis(var_values, [lat_axis, lon_axis], [-2, -1])
                print(f"      After moveaxis: {var_values_moved.shape}")
                
                # Process data
                shifted_values = _shift_2d_field(
                    var_values_moved,
                    source_mask,
                    target_mask,
                    lats,
                    lons,
                    interpolation_method
                )
                print(f"      After shift: {shifted_values.shape}")
                
                # Move lat and lon back to their original positions
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
    Spatially shift a single 2D field (helper function).
    
    Workflow:
    1. Save source region data
    2. Copy source region data to target region (overwrite)
    3. Fill source region "hole" with interpolation
    
    Args:
        field_2d: 2D data field, shape may be (lat, lon) or (batch, time, lat, lon)
        source_mask: Source region mask (lat, lon)
        target_mask: Target region mask (lat, lon)
        lats: Latitude coordinates
        lons: Longitude coordinates
        interpolation_method: Interpolation method
    
    Returns:
        shifted_field: Shifted 2D field
    """
    from scipy.interpolate import griddata
    
    # Handle different dimension layouts
    original_shape = field_2d.shape
    
    # 4D (batch, time, lat, lon): process each batch and time step separately
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
    
    # 3D (time, lat, lon) or (batch, lat, lon)
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
    
    # 2D (lat, lon)
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
    Core implementation of spatial displacement for a single (lat, lon) 2D field.
    
    Uses point-to-point interpolation:
    1. Extract data from source region grid points
    2. Interpolate extracted data onto target region grid points
    3. Fill the emptied source region with interpolation from surrounding values
    
    Args:
        field: (lat, lon) 2D field, or higher-dimensional array that can be squeezed to 2D
        source_mask: Source region mask (lat, lon)
        target_mask: Target region mask (lat, lon)
        lats: Latitude coordinates
        lons: Longitude coordinates
        interpolation_method: Interpolation method
    
    Returns:
        shifted: Shifted 2D field, same shape as input
    """
    from scipy.interpolate import griddata
    
    # Save original shape
    original_shape = field.shape
    
    # Ensure field is 2D:
    # Strategy: use numpy.squeeze to remove all size-1 dimensions, then check for (lat, lon)
    field_squeezed = np.squeeze(field)
    
    # Check shape after squeeze
    if field_squeezed.ndim == 2:
        # Verify it matches (lat, lon)
        if field_squeezed.shape[0] == len(lats) and field_squeezed.shape[1] == len(lons):
            field_2d = field_squeezed
        else:
            raise ValueError(f"Field 2D shape {field_squeezed.shape} doesn't match expected (lat={len(lats)}, lon={len(lons)})")
    elif field_squeezed.ndim > 2:
        # Still >2D after squeeze: has multiple non-1 dimensions
        # Try extracting the last two dimensions (assumed to be lat, lon)
        if field_squeezed.shape[-2] == len(lats) and field_squeezed.shape[-1] == len(lons):
            # If 3D and the first dimension is batch/time with size 1, extract it
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
    
    # Final validation
    if field_2d.shape != (len(lats), len(lons)):
        raise ValueError(f"Final field_2d shape {field_2d.shape} doesn't match expected ({len(lats)}, {len(lons)}). Original: {original_shape}")
    
    # Ensure C-contiguous numpy array (avoid view issues)
    field_2d = np.ascontiguousarray(field_2d)
    
    # Create grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Step 1: Create result array (start from copy of original)
    shifted = field_2d.copy()
    
    # Step 2: "Move" data from source region to target region
    # Strategy: for each point in the target region, determine which "source position"
    # in the field it should get data from, then interpolate from original field_2d
    
    # Compute displacement from center difference between source and target regions
    source_center_lat = lat_grid[source_mask].mean()
    source_center_lon = lon_grid[source_mask].mean()
    target_center_lat = lat_grid[target_mask].mean()
    target_center_lon = lon_grid[target_mask].mean()
    
    delta_lat = target_center_lat - source_center_lat
    delta_lon = target_center_lon - source_center_lon
    
    # Get coordinates of points in the target region
    target_lat_coords = lat_grid[target_mask]
    target_lon_coords = lon_grid[target_mask]
    
    # For each target point, compute its "source position" (inverse mapping)
    # A point at (lat_t, lon_t) in the target should get data from (lat_t - delta_lat, lon_t - delta_lon)
    source_lat_for_target = target_lat_coords - delta_lat
    source_lon_for_target = target_lon_coords - delta_lon
    
    # Interpolate from the original field_2d at those source positions
    # Use the entire field as interpolation source (not just source_mask points)
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
        
        # Interpolate from the original field to obtain target region values
        target_values = griddata(
            all_points,
            all_values,
            query_points,
            method=method,
            fill_value=np.nan
        )
        
        # Fall back to nearest-neighbor for any NaN values
        if np.any(np.isnan(target_values)):
            nan_mask = np.isnan(target_values)
            target_values[nan_mask] = griddata(
                all_points,
                all_values,
                query_points[nan_mask],
                method="nearest"
            )
        
        # Write interpolated values into the target region
        shifted[target_mask] = target_values
        
    except Exception as e:
        print(f"    [Spatial Shift] Warning: Target interpolation failed ({e}), using field mean")
        shifted[target_mask] = field_2d.mean()
    
    # Step 3: Fill the source region "hole" using interpolation from surrounding data
    # Key: only fill the part of the source region not overlapping with the target,
    # preserving data that was already moved
    unchanged_mask = ~(source_mask | target_mask)  # neither source nor target
    
    # Define the "hole": source region minus the overlap with target
    source_hole_mask = source_mask & ~target_mask  # true hole
    
    # Fill the hole if it exists and there is enough surrounding data for interpolation
    if source_hole_mask.sum() > 0 and unchanged_mask.sum() > 0:
        known_points = np.column_stack([
            lat_grid[unchanged_mask].ravel(),
            lon_grid[unchanged_mask].ravel()
        ])
        known_values = field_2d[unchanged_mask].ravel()  # use original values
        
        # Interpolate only the hole (excluding the overlap with target)
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
            
            # Fall back to nearest-neighbor for any NaN values
            if np.any(np.isnan(interpolated_values)):
                nan_mask = np.isnan(interpolated_values)
                interpolated_values[nan_mask] = griddata(
                    known_points,
                    known_values,
                    source_hole_points[nan_mask],
                    method="nearest"
                )
            
            # Fill only the hole, not the part overlapping with the target
            shifted[source_hole_mask] = interpolated_values
            
        except Exception as e:
            print(f"    [Spatial Shift] Warning: Source hole filling failed ({e}), using mean fill")
            shifted[source_hole_mask] = known_values.mean()
    elif source_hole_mask.sum() > 0:
        # Has a hole but not enough unchanged area for interpolation
        print(f"    [Spatial Shift] Warning: Not enough unchanged area for interpolation, using simple mean fill")
        shifted[source_hole_mask] = field_2d.mean()
    
    # Restore original shape if needed
    if original_shape != shifted.shape:
        shifted = shifted.reshape(original_shape)
    
    return shifted


def _get_next_inputs_rollout(
    prev_inputs: xr.Dataset,
    next_frame: xr.Dataset,
) -> xr.Dataset:
    """
    [ROLLOUT ONLY] Update the input queue: retain the most recent num_inputs time steps.
    
    In autoregressive rollout, after each prediction step, the result must be appended
    to the input queue and the oldest step discarded — a "sliding window" effect.
    
    Args:
        prev_inputs: Previous inputs (batch, time=2, lat, lon, ...)
                    e.g. [t-12h, t]
        next_frame: Newly predicted frame (batch, time=1, lat, lon, ...) with forcings merged
                    e.g. [t+12h]
    
    Returns:
        new_inputs: Updated inputs (batch, time=2, lat, lon, ...)
                   e.g. [t, t+12h]
    
    Implementation:
        Queue-like push operation:
        [t-12h, t] + [t+12h] → concat → [t-12h, t, t+12h] → tail(2) → [t, t+12h]
    """
    # Find variables to copy from predictions into inputs
    # (exclude variables that only exist in forcings, e.g. tisr)
    next_inputs_keys = list(
        set(next_frame.keys()).intersection(set(prev_inputs.keys()))
    )
    next_inputs = next_frame[next_inputs_keys]
    
    # Concatenate and keep only the last num_inputs time steps
    num_inputs = prev_inputs.dims["time"]
    new_inputs = xr.concat(
        [prev_inputs, next_inputs],
        dim="time",
        data_vars="different"
    ).tail(time=num_inputs)
    
    return new_inputs


# =========================
# Utils: Indexing, Batching, Type Conversion
# =========================
def detect_want_times(ds: DateMergedERA5TyphoonSizeDataset, want_times: List[str]):
    report = []
    time2idx = build_time2idx_map(ds)
    ds_time_min = pd.Timestamp(ds.ds["time"].min().values)
    for t in want_times:
        ts = pd.Timestamp(t)
        # Match by date (ignoring hour), because DateMergedERA5TyphoonSizeDataset uses merge_same_day
        date_key = ts.normalize()  # normalize to 00:00:00
        in_tracks = date_key in time2idx
        has_full_window = (ts - np.timedelta64(24, "h") >= ds_time_min)
        idxs = time2idx.get(date_key, [])
        n_samples = len(idxs)
        reason = None
        if not in_tracks:
            reason = "not in tracks (no track for this date, or date not in eval year range)"
        elif not has_full_window:
            reason = "no full window (t-24h out of bounds)"
        elif n_samples == 0:
            reason = "no sample (track group is empty)"
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
    """Build a time -> index mapping by date (not exact timestamp), since merge_same_day makes the hour non-fixed."""
    time2idx = defaultdict(list)
    for i, r in ds.tracks.reset_index().iterrows():
        # Use only year/month/day, ignore hour (normalize to date)
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
        # Match by date (ignore hour)
        date_key = ts.normalize()
        hit = time2idx.get(date_key, [])
        if hit:
            indices.append(hit[0])   # take only the first index to avoid duplicates
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
# Lightweight Naming Helpers (SAVE_MODE routing)
# =========================
def _short_time_label(ts: str) -> str:
    t = pd.Timestamp(ts)
    return f"{t:%Y%m%d_%H}"

def _param_slug(
    readout_collect_steps_for_vis: Tuple[int, ...],
    inner_idxs: List[int],
    inner_steps_map: Dict[int, int],  # per-step optimization count
    max_opt_steps: int,                # maximum count
    inner_lr: float,
    strength: float,
    random_seed: int = 42,             # random seed
    guidance_method: str = "direct_optim",  # guidance method
    intensity_scaling_configs: Optional[List[Dict]] = None,  # scaling configs
    shift_configs: Optional[List[Dict]] = None,              # shift configs
    warp_configs: Optional[Dict] = None,                     # warp configs
) -> str:
    """
    Generate a parameter identifier string (used for folder naming).
    
    Folder naming examples:
    - Basic: idx[4-8-12]__steps[0-15x1]__lr0.005__gsteps[10-15]__strength0.7__seed42
    - +scale: ...seed42__scale[s0x3_s1x2.5]  (step 0 amplified 3x, step 1 amplified 2.5x)
    - +shift: ...seed42__shift[s0d+5_+3_s1d-2_+1]  (step 0: north 5°/east 3°, step 1: south 2°/east 1°)
    - Full: ...seed42__scale[s0x3]__shift[s0d-3_-2]
    """
    idxs_part = "-".join(str(int(x)) for x in inner_idxs) if inner_idxs else "none"
    gsteps_part = "-".join(str(int(x)) for x in readout_collect_steps_for_vis)
    lr_part = f"{inner_lr:g}"
    
    # Generate steps_map summary string (e.g. 4-8x40_9-16x1)
    if inner_steps_map:
        # Sort steps and find consecutive ranges
        sorted_steps = sorted(inner_steps_map.keys())
        ranges = []
        if sorted_steps:
            start = sorted_steps[0]
            prev_step = sorted_steps[0]
            prev_opt = inner_steps_map[prev_step]
            
            for step in sorted_steps[1:]:
                opt = inner_steps_map[step]
                if step == prev_step + 1 and opt == prev_opt:
                    # Consecutive with same opt_steps: extend range
                    prev_step = step
                else:
                    # Range ended: record
                    ranges.append(f"{start}-{prev_step}x{prev_opt}")
                    start = step
                    prev_step = step
                    prev_opt = opt
            # Last range
            ranges.append(f"{start}-{prev_step}x{prev_opt}")
        steps_map_str = "_".join(ranges)
    else:
        steps_map_str = f"max{max_opt_steps}"
    
    # Build base slug
    if guidance_method == "none":
        # Baseline only - only basic info needed
        base_slug = f"method[baseline]__seed{random_seed}"
    elif guidance_method == "input_manipulation":
        # Input Manipulation - no lr/strength/steps needed
        base_slug = f"method[input_manip]__seed{random_seed}"
    elif guidance_method == "local_affine":
        # Local Affine - simplified params (include steps_map to support sweep)
        base_slug = f"method[affine]__steps[{steps_map_str}]__lr{lr_part}__seed{random_seed}"
    else:
        # direct_optim - full params
        base_slug = f"method[direct_optim]__idx[{idxs_part}]__steps[{steps_map_str}]__lr{lr_part}__gsteps[{gsteps_part}]__strength{strength:g}__seed{random_seed}"
    
    # Add extra scale and shift parameters
    extra_parts = []
    
    # Intensity Scaling parameter summary
    if intensity_scaling_configs:
        scale_summaries = []
        for cfg in intensity_scaling_configs:
            step_idx = cfg.get("step_idx", 0)
            scale_factor = cfg.get("scale_factor", 1.0)
            # Compact format: s{step_idx}x{scale_factor}
            scale_summaries.append(f"s{step_idx}x{scale_factor:g}")
        extra_parts.append("scale[" + "_".join(scale_summaries) + "]")
    
    # Spatial Shift parameter summary
    if shift_configs:
        shift_summaries = []
        for cfg in shift_configs:
            step_idx = cfg.get("step_idx", 0)
            delta_lat = cfg.get("delta_lat", 0.0)
            delta_lon = cfg.get("delta_lon", 0.0)
            # Compact format: s{step_idx}d{delta_lat:+g}_{delta_lon:+g}
            # :+g preserves the sign (+/-)
            shift_summaries.append(f"s{step_idx}d{delta_lat:+g}_{delta_lon:+g}")
        extra_parts.append("shift[" + "_".join(shift_summaries) + "]")
    
    # Warp parameter summary
    if warp_configs and warp_configs.get("enabled", False):
        # Extract parameters
        center_lat = warp_configs.get("center_lat", 0.0)
        center_lon = warp_configs.get("center_lon", 0.0)
        radius = warp_configs.get("radius", 0.0)
        warp_lr = warp_configs.get("learning_rate", 1e-2)
        reg_weight = warp_configs.get("regularization_weight", 1e-3)
        
        # Build summary: warp[c{lat}_{lon}_r{radius}_trans_rot_scale_lr{lr}_reg{reg}]
        warp_summary = f"warp[c{center_lat:.1f}_{center_lon:.1f}_r{radius:g}"
        
        # Add optimize flags
        opt_parts = []
        if warp_configs.get("optimize_translation", False):
            opt_parts.append("trans")
        if warp_configs.get("optimize_rotation", False):
            opt_parts.append("rot")
        if warp_configs.get("optimize_scale", False):
            opt_parts.append("scale")
        if opt_parts:
            warp_summary += "_" + "_".join(opt_parts)
        
        # Add learning rate and regularization weight
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
    inner_steps_map: Dict[int, int],  # per-step optimization count
    max_opt_steps: int,                # maximum count
    inner_lr: float,
    strength: float,
    guidance_mode: str = "steer_one",  # guidance mode
    random_seed: int = 42,             # random seed
    guidance_method: str = "direct_optim",  # guidance method
    intensity_scaling_configs: Optional[List[Dict]] = None,  # scaling configs
    shift_configs: Optional[List[Dict]] = None,              # shift configs
    warp_configs: Optional[Dict] = None,                     # warp configs
) -> Dict[str, str]:
    """
    Determine output directories based on SAVE_MODE (adapted for manual guidance mode).
    Seed is placed at a higher folder level so different seeds do not share baseline.
    """
    time_slug = _short_time_label(want_time)
    targets_slug = _targets_slug(manual_targets)
    # param_slug without seed (seed is already in root path)
    param_slug_no_seed = _param_slug(
        readout_collect_steps_for_vis, inner_idxs, inner_steps_map, max_opt_steps, inner_lr, strength, 
        random_seed=42,  # fixed 42 used only for generating B directory name
        guidance_method=guidance_method,
        intensity_scaling_configs=intensity_scaling_configs,
        shift_configs=shift_configs,
        warp_configs=warp_configs,
    )
    # Actual param_slug still includes seed (used in return value)
    param_slug = _param_slug(
        readout_collect_steps_for_vis, inner_idxs, inner_steps_map, max_opt_steps, inner_lr, strength, 
        random_seed,
        guidance_method=guidance_method,
        intensity_scaling_configs=intensity_scaling_configs,
        shift_configs=shift_configs,
        warp_configs=warp_configs,
    )

    if save_mode == "by_dates":
        # New structure: time promoted one level; all experiments with same time share A_no_guidance
        # Seed is in root path; time is the second-level directory
        root = os.path.join(output_dir, f"seed{random_seed}", f"time_{time_slug}")
        # A_no_guidance is under the time directory, shared by all targets
        out_A = os.path.join(root, "A_no_guidance")
        # B directory is under the target subdirectory, differentiated by param_slug
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
        raise ValueError("None of the given dates matched (possibly out of range or not in tracks).")
    subset = Subset(eval_dataset, idxs)
    subloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=custom_collate_fn
    )
    return subloader


# =========================
# Model Construction / JIT
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
# Inference (strength=0: no guidance; >0: guided)
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
    guide_inner_opt_steps_map: Dict[int, int],  # per-step optimization count
    guide_max_opt_steps: int,                    # maximum count (for JAX compilation)
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,
    guidance_mask: Optional[torch.Tensor] = None,  # selective mask (B, H, W)
    random_seed: int = 42,  # random seed
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
        "inner_opt_steps_map": inner_steps_map,
        "max_opt_steps": max_opt_steps,
        "inner_opt_lr": float(guide_inner_opt_lr),
        "loss_type": str(guide_loss_type),
        "target_readout": None,
        "guidance_mask": None,
        "warp_configs": warp_configs,
    }
    if target_readout_one_hot is not None:
        one_hot_np = target_readout_one_hot.numpy() if hasattr(target_readout_one_hot, "numpy") else np.asarray(target_readout_one_hot)
        guidance_cfg["target_readout"] = jnp.asarray(one_hot_np)
    
    # Pass guidance_mask if provided
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
    
    # Release GPU data immediately after converting to numpy
    del preds, readouts
    
    # Convert loss_history to numpy and extract valid values
    loss_history_np = _to_np(loss_history)  # shape (20, max_opt_steps)
    del loss_history
    loss_history_dict = {}
    for step in guide_inner_opt_step_idxs:
        step_losses = loss_history_np[step, :]
        # Get valid optimization count for this step from guide_inner_opt_steps_map
        valid_count = guide_inner_opt_steps_map.get(step, max_opt_steps)
        # Take only the first valid_count values (the ones actually executed)
        valid_losses = step_losses[:valid_count]
        # Filter out possible -1.0 values (steps where guidance was not triggered)
        valid_losses = valid_losses[valid_losses >= 0]
        if len(valid_losses) > 0:
            loss_history_dict[int(step)] = valid_losses.tolist()
    del loss_history_np
    
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
    
    # Guidance Method (mode selection)
    guidance_method: str = "direct_optim",  # "none" | "direct_optim" | "input_manipulation" | "local_affine"
    
    # Guidance parameters (used by direct_optim and local_affine only)
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
    timing_logger: Optional[TimingLogger] = None,  # timing recorder
    random_seed: int = 42,  # random seed
    
    # Input Manipulation parameters (input_manipulation only)
    intensity_scaling_configs: Optional[List[Dict]] = None,  # scaling config list
    shift_configs: Optional[List[Dict]] = None,  # shift config list
    
    # Local Affine parameters (local_affine only)
    warp_configs: Optional[Dict] = None,  # warp config
) -> Tuple[List[xr.Dataset], List[Dict[str, xr.Dataset]], Dict[int, List[float]]]:
    """
    [ROLLOUT ONLY] Execute a multi-step autoregressive rollout with optional guidance on the first step.
    
    This is the core rollout function implementing manual loop multi-step prediction:
    - Each step uses the previous 2 time steps as input to predict the next 12h
    - The prediction is appended to the input queue; the oldest step is discarded (sliding window)
    - Optional guidance (selective storm guidance) on the first step
    - Subsequent steps use standard inference without guidance
    
    Args:
        params: Model parameters
        eval_inputs: Initial inputs (batch, time=2, lat, lon, ...), e.g. [t-12h, t]
        eval_targets_template: Single-step target template (batch, time=1, ...) for each step
        forcings_extended: Extended forcings (batch, time=num_steps, ...)
        readout_guided_inference_fn: Inference function (haiku transform)
        num_steps: Total rollout steps (e.g. 10 steps = 5 days)
        guidance_on_first_step: Whether to apply guidance on the first step
        
        target_readout_one_hot: Guidance target mask (B, H, W, 2)
        guidance_mask: Guidance region mask (B, H, W) for local loss computation
        readout_collect_steps_for_vis: Denoising steps to collect readouts from
        guidance_strength: Guidance strength (0.0 = no guidance)
        guide_inner_opt_step_idxs: Denoising steps to optimize
        guide_inner_opt_steps_map: Per-step optimization count mapping
        guide_max_opt_steps: Maximum optimization count (for JAX compilation)
        guide_inner_opt_lr: Optimization learning rate
        guide_loss_type: Loss type ("readout_l2" / "xt_l2")
        guide_normalize_grad: Whether to normalize gradients
        guide_eps: Numerical stability epsilon
    
    Returns:
        predictions_list: Per-step prediction results List[xr.Dataset], length = num_steps
        readouts_list: Per-step readout results List[Dict[str, xr.Dataset]]
        loss_history: Guidance loss history for the first step (if guidance was used)
    
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
    
    # Initialize
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
        
        # Track timing for each step
        step_timing_started = False
        if timing_logger:
            try:
                timing_logger.start(step_name)
                step_timing_started = True
            except Exception as e:
                print(f"  [TimingLogger Warning] Failed to start timing for {step_name}: {e}")
        
        # Extract forcings for the current step
        current_forcings = forcings_extended.isel(time=slice(step_idx, step_idx + 1))
        
        # Decide whether to use guidance (first step only)
        use_guidance_this_step = (step_idx == 0) and guidance_on_first_step
        
        # Determine whether guidance is active based on guidance_method
        if guidance_method == "none":
            # Baseline only: force guidance off
            use_guidance_this_step = False
            current_strength = 0.0
        elif guidance_method == "input_manipulation":
            # Input Manipulation does not use guidance in the denoising process
            use_guidance_this_step = False
            current_strength = 0.0
        else:
            # direct_optim or local_affine: use guidance
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
        
        # Build targets_template for current step (single step, filled with NaN)
        current_targets_template = eval_targets_template * np.nan
        # Update time coordinate to the current step's timedelta
        time_coord = np.timedelta64(12 * (step_idx + 1), 'h')
        current_targets_template = current_targets_template.assign_coords(time=[time_coord])
        
        # Build guidance config
        rng = jax.random.PRNGKey(random_seed + step_idx)  # different seed per step
        state_g = {}
        
        inner_idxs = [] if not use_guidance_this_step else list(guide_inner_opt_step_idxs)
        inner_steps_map = {} if not use_guidance_this_step else dict(guide_inner_opt_steps_map)
        max_opt_steps = 1 if not use_guidance_this_step else int(guide_max_opt_steps)
        
        # When not using guidance, must use "xt_l2" (readout_l2 requires target_readout)
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
            "loss_type": str(current_loss_type),  # use current_loss_type, not guide_loss_type
            "target_readout": None,
            "guidance_mask": None,
            "warp_configs": warp_configs if use_guidance_this_step else None,  # local_affine warp config
        }
        
        if use_guidance_this_step and target_readout_one_hot is not None:
            one_hot_np = target_readout_one_hot.numpy() if hasattr(target_readout_one_hot, "numpy") else np.asarray(target_readout_one_hot)
            guidance_cfg["target_readout"] = jnp.asarray(one_hot_np)
            
            if guidance_mask is not None:
                mask_np = guidance_mask.numpy() if hasattr(guidance_mask, "numpy") else np.asarray(guidance_mask)
                guidance_cfg["guidance_mask"] = jnp.asarray(mask_np)
        
        # Call model inference
        (preds, readouts, loss_history), _ = readout_guided_inference_fn.apply(
            params, state_g, rng,
            current_inputs, current_targets_template, current_forcings, guidance_cfg
        )
        
        # Transfer to host (GPU/TPU → CPU)
        preds_host = xr_to_numpy(preds)
        readouts_host = {k: xr_to_numpy(v) for k, v in readouts.items()}
        
        # Release GPU data immediately after converting to numpy
        del preds, readouts
        
        # ===== [Direct Intensity Scaling] Apply local scaling at specified steps =====
        if intensity_scaling_configs is not None and len(intensity_scaling_configs) > 0:
            # Find config for the current step
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
                print(f"  [Direct Intensity] Scaling applied successfully")
        
        # ===== [Spatial Shift] Apply local spatial displacement at specified steps =====
        if shift_configs is not None and len(shift_configs) > 0:
            # Find config for the current step
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
                print(f"  [Spatial Shift] Shift applied successfully")
        
        # Save results
        predictions_list.append(preds_host)
        readouts_list.append(readouts_host)
        
        # Save loss history (only for the first step with guidance)
        if use_guidance_this_step and guide_inner_opt_step_idxs:
            loss_history_np = _to_np(loss_history)  # shape (20, max_opt_steps)
            del loss_history  # release after converting to numpy
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
            del loss_history_np  # release loss_history_np
        elif not use_guidance_this_step:
            # If guidance was not used, immediately release loss_history
            del loss_history
        
        # Build inputs for the next step (core rollout logic)
        if step_idx < num_steps - 1:
            # Merge predictions and forcings (next_frame contains all variables for next step)
            next_frame = xr.merge([preds_host, current_forcings])
            
            # Update input queue: [t-12h, t] → [t, t+12h]
            current_inputs = _get_next_inputs_rollout(current_inputs, next_frame)
            
            # Release intermediate variables
            del next_frame
            
            print(f"  [Rollout] Updated inputs for next step")
            print(f"            New input time coords: {current_inputs.coords['time'].values}")
        else:
            print(f"  [Rollout] Last step, no input update needed")
        
        # Light memory cleanup every few steps (avoid doing it too frequently)
        if step_idx > 0 and step_idx % 3 == 0:
            gc.collect()
        
        # End timing for this step
        if timing_logger and step_timing_started:
            try:
                timing_logger.end(step_name)
            except Exception as e:
                print(f"  [TimingLogger Warning] Failed to end timing for {step_name}: {e}")
    
    print(f"\n[Rollout] Completed {num_steps}-step rollout successfully")
    print(f"[Rollout] Total predictions: {len(predictions_list)}")
    print(f"[Rollout] Total readouts: {len(readouts_list)}")
    
    # Final memory cleanup
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
    step_prefix: Optional[str] = "step00_",  # file name prefix, default "step00_"
    timing_logger: Optional[TimingLogger] = None,  # timing recorder
    vis_n_procs: int = 1,  # visualization parameters
    vis_contourf_levels: int = 36,
    vis_coastlines_resolution: str = "50m",
    vis_draw_gridlabels: bool = True,
    vis_add_borders: bool = True,
    vis_dpi: int = 240,
):
    """
    Visualize single-step or multi-step prediction results.
    
    Args:
        step_prefix: File name prefix, default "step00_" (single-step prediction)
        timing_logger: Optional timing recorder
    """
    from graphcast.vis import generate_comparison_gifs_parallel_with_prefix
    
    os.makedirs(out_dir, exist_ok=True)
    gt_frames = [g for g in gt_list]
    forecast_frames = [p.isel(time=0, batch=0) for p in predictions_list]
    readout_frames = {
        key: [ro[key].isel(time=0, batch=0) for ro in readouts_list]
        for key in readouts_list[0].keys()
    }
    # Use provided visual_vars; if None, plot all variables
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
        step_prefix=step_prefix,  # pass step prefix
        contourf_levels=vis_contourf_levels,
        coastlines_resolution=vis_coastlines_resolution,
        draw_gridlabels=vis_draw_gridlabels,
        add_borders=vis_add_borders,
        dpi=vis_dpi,
        timing_logger=timing_logger,  # pass timing recorder
    )

    visualize_onehot_readout_simple(
        readout_frames_dict=readout_frames,
        one_hot=one_hot_torch,
        output_dir=out_dir,
        epoch=epoch,
        ts=str(gt_frames[0].datetime.values) if 'datetime' in gt_frames[0].coords else "unknown-ts",
        mark=None,
        steps=list(readout_frames.keys()),  # use actual readout steps
    )


# =========================
# Rollout Data Saving: Save as NetCDF for downstream visualization
# =========================
def _save_rollout_data_netcdf(
    *,
    predictions_list: List[xr.Dataset],
    readouts_list: List[Dict[str, xr.Dataset]],
    gt_list: List[xr.Dataset],
    one_hot_torch: torch.Tensor,
    out_dir: str,
    metadata: Optional[Dict] = None,
    save_readout_steps: Optional[List[int]] = None,  # specify which steps to save readouts (e.g. [0] = first step only)
    shift_configs: Optional[List[Dict]] = None,  # shift configs for saving visualization info
) -> str:
    """
    Save rollout data in NetCDF format (efficiently compressed, with full metadata).
    
    File structure:
        rollout_data/
            predictions_step00.nc
            predictions_step01.nc
            ...
            readouts_step00_denoising10.nc  # only guidance steps saved
            readouts_step00_denoising15.nc
            ...
            gt_step00.nc
            gt_step01.nc
            ...
            one_hot.npy
            metadata.json
    
    Args:
        predictions_list: Per-step prediction results
        readouts_list: Per-step readout results
        gt_list: Per-step ground truth data
        one_hot_torch: One-hot mask
        out_dir: Output directory
        metadata: Additional metadata (e.g. epoch, visual_vars)
        save_readout_steps: List of step indices whose readouts should be saved
                           (e.g. [0] = first step only). None = save all; [] = save none.
    
    Returns:
        data_dir: Path to the saved data directory
    """
    import json
    
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "rollout_data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n[Saving Rollout Data] Saving to {data_dir}...")
    t0 = time.time()
    
    # Save predictions
    print(f"  [Save] Saving {len(predictions_list)} predictions...")
    for idx, pred in enumerate(predictions_list):
        pred_path = os.path.join(data_dir, f"predictions_step{idx:02d}.nc")
        pred.to_netcdf(
            pred_path,
            engine='h5netcdf',
            encoding={var: {'zlib': True, 'complevel': 5} for var in pred.data_vars}
        )
    
    # Save readouts (only for specified steps)
    if len(readouts_list) > 0:
        if save_readout_steps is None:
            # If not specified, save all steps (backward compatible)
            steps_to_save = list(range(len(readouts_list)))
            print(f"  [Save] Saving readouts for all {len(readouts_list)} steps (save_readout_steps=None)...")
        elif len(save_readout_steps) == 0:
            # Empty list: skip readout saving
            steps_to_save = []
            print(f"  [Save] Skipping readout saving (save_readout_steps=[])...")
        else:
            # Save only specified steps
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
    
    # Save GT
    print(f"  [Save] Saving {len(gt_list)} ground truth frames...")
    for idx, gt in enumerate(gt_list):
        gt_path = os.path.join(data_dir, f"gt_step{idx:02d}.nc")
        gt.to_netcdf(
            gt_path,
            engine='h5netcdf',
            encoding={var: {'zlib': True, 'complevel': 5} for var in gt.data_vars}
        )
    
    # Save one_hot as numpy (simple format)
    one_hot_np = one_hot_torch.numpy() if hasattr(one_hot_torch, "numpy") else np.asarray(one_hot_torch)
    one_hot_path = os.path.join(data_dir, "one_hot.npy")
    np.save(one_hot_path, one_hot_np)
    
    # Save metadata (JSON)
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata_serializable = {}
    if metadata:
        for k, v in metadata.items():
            if k == "loss_history" and isinstance(v, dict):
                # loss_history is already serializable
                metadata_serializable[k] = v
            elif isinstance(v, (list, tuple)):
                metadata_serializable[k] = list(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                metadata_serializable[k] = v
            else:
                metadata_serializable[k] = str(v)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    # Save shift regions info (for visualization)
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
                # Compute target center location
                "target_lat": cfg.get("center_lat", 0.0) + cfg.get("delta_lat", 0.0),
                "target_lon": cfg.get("center_lon", 0.0) + cfg.get("delta_lon", 0.0),
            })
        with open(shift_regions_path, 'w') as f:
            json.dump(shift_regions_data, f, indent=2)
        print(f"  [Save] Saved shift regions info for visualization: {shift_regions_path}")
    
    # Compute total size
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
    gt_list: List[xr.Dataset],  # accept pre-loaded GT data directly
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
    [ROLLOUT ONLY] Visualize all rollout steps with per-step filename prefixes.
    
    Args:
        predictions_list: Per-step prediction results
        readouts_list: Per-step readout results
        gt_list: Per-step GT data (pre-loaded)
        one_hot_torch: One-hot mask
        out_dir: Output directory
        epoch: Epoch number
        visual_vars: List of variables to visualize
        timing_logger: Timing recorder
        vis_n_procs: Number of parallel visualization processes
        vis_contourf_levels: Number of contourf levels
        vis_coastlines_resolution: Coastline resolution
        vis_draw_gridlabels: Whether to draw grid labels
        vis_add_borders: Whether to add borders
        vis_dpi: DPI setting
    """
    from graphcast.vis import generate_comparison_gifs_parallel_with_prefix
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[Rollout Visualization] Using pre-loaded GT data for {len(gt_list)} steps...")
    
    # Prepare forecast_frames and readout_frames
    forecast_frames = [p.isel(time=0, batch=0) for p in predictions_list]
    
    readout_frames = {}
    if len(readouts_list) > 0:
        all_readout_steps = list(readouts_list[0].keys())
        for step in all_readout_steps:
            readout_frames[step] = [
                readouts_list[rollout_idx][step].isel(time=0, batch=0) 
                for rollout_idx in range(len(readouts_list))
            ]
    
    # Use provided visual_vars
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
    
    # Generate step prefix list: ["step00_", "step01_", ..., "step09_"]
    num_steps = len(predictions_list)  # get step count from predictions_list
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
        step_prefix=step_prefix_list,  # pass step prefix list
        contourf_levels=vis_contourf_levels,
        coastlines_resolution=vis_coastlines_resolution,
        draw_gridlabels=vis_draw_gridlabels,
        add_borders=vis_add_borders,
        dpi=vis_dpi,
        timing_logger=timing_logger,  # pass timing recorder
    )
    
    if readout_frames:
        # Get timestamp from gt_list or predictions_list (for file naming)
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
# Rollout Video Generation: PNG assembly and video generation
# =========================
def check_single_rollout_completed(
    out_dir: str,
    num_rollout_steps: int,
    min_steps_required: int = 3,
) -> bool:
    """
    Check whether the rollout data for a single directory has been completed.
    
    Args:
        out_dir: Output directory (A_no_guidance or B_guided)
        num_rollout_steps: Total rollout steps
        min_steps_required: Minimum number of step NetCDF files required to be considered complete
    
    Returns:
        is_completed: Whether the rollout is complete
    """
    if not os.path.isdir(out_dir):
        print(f"    [Check Single] Directory does not exist: {out_dir}")
        return False
    
    data_dir = os.path.join(out_dir, "rollout_data")
    if not os.path.isdir(data_dir):
        print(f"    [Check Single] rollout_data directory does not exist: {data_dir}")
        return False
    
    # Check required NetCDF files
    required_files = []
    for step_idx in range(num_rollout_steps):
        pred_file = os.path.join(data_dir, f"predictions_step{step_idx:02d}.nc")
        gt_file = os.path.join(data_dir, f"gt_step{step_idx:02d}.nc")
        required_files.extend([pred_file, gt_file])
    
    # Count existing files
    found_files = sum(1 for f in required_files if os.path.isfile(f))
    total_required = len(required_files)
    min_files_required = min_steps_required * 2  # each step requires predictions and gt files
    
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
    min_steps_required: int = 3,  # minimum step NetCDF files required to be considered complete
) -> Tuple[bool, List[str]]:
    """
    Detect whether the rollout has been completed (by checking NetCDF data files).
    
    This is the DataOnly version: checks .nc files under rollout_data/:
    - predictions_step00.nc, predictions_step01.nc, ...
    - gt_step00.nc, gt_step01.nc, ...
    - metadata.json
    
    Args:
        out_A: Baseline output directory
        out_B: Guided output directory
        num_rollout_steps: Total rollout steps
        visual_vars: Variables to check (if None, check all; mainly for compatibility)
        min_steps_required: Minimum number of step NetCDF files required to be considered complete
    
    Returns:
        (is_completed, found_variables): 
            is_completed: Whether the rollout is complete
            found_variables: Found variable list (returns empty list in DataOnly version)
    """
    # Check whether directories exist
    has_A = os.path.isdir(out_A)
    has_B = os.path.isdir(out_B)
    
    print(f"    [Check] out_A exists: {has_A}, out_B exists: {has_B}")
    
    if not has_A:
        print(f"    [Check] Baseline directory not found: {out_A}")
        return False, []
    
    if not has_B:
        print(f"    [Check] Guided directory not found: {out_B}")
        # B doesn't exist: baseline may be done but guided hasn't run yet
        # Don't skip in this case since guided still needs to run
        return False, []
    
    # Check rollout_data directory
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
    
    # Check required NetCDF files
    required_files_A = []
    required_files_B = []
    
    # Check predictions and gt files
    for step_idx in range(num_rollout_steps):
        pred_file_A = os.path.join(data_dir_A, f"predictions_step{step_idx:02d}.nc")
        pred_file_B = os.path.join(data_dir_B, f"predictions_step{step_idx:02d}.nc")
        gt_file_A = os.path.join(data_dir_A, f"gt_step{step_idx:02d}.nc")
        gt_file_B = os.path.join(data_dir_B, f"gt_step{step_idx:02d}.nc")
        
        required_files_A.extend([pred_file_A, gt_file_A])
        required_files_B.extend([pred_file_B, gt_file_B])
    
    # Check metadata.json (optional, but improves reliability)
    metadata_A = os.path.join(data_dir_A, "metadata.json")
    metadata_B = os.path.join(data_dir_B, "metadata.json")
    
    # Count existing files
    found_files_A = sum(1 for f in required_files_A if os.path.isfile(f))
    found_files_B = sum(1 for f in required_files_B if os.path.isfile(f))
    
    has_metadata_A = os.path.isfile(metadata_A)
    has_metadata_B = os.path.isfile(metadata_B)
    
    total_required = len(required_files_A)  # A and B should have the same count
    min_files_required = min_steps_required * 2  # each step needs predictions and gt files
    
    print(f"    [Check] Found {found_files_A}/{total_required} files in A, {found_files_B}/{total_required} files in B")
    print(f"    [Check] Metadata - A: {has_metadata_A}, B: {has_metadata_B}")
    
    # Both A and B must have enough files
    is_completed_A = found_files_A >= min_files_required
    is_completed_B = found_files_B >= min_files_required
    
    is_completed = is_completed_A and is_completed_B
    
    if is_completed:
        print(f"    [Check] Rollout data completed: A has {found_files_A} files, B has {found_files_B} files")
    else:
        if not is_completed_A:
            print(f"    [Check] Baseline incomplete: {found_files_A}/{total_required} files (need {min_files_required})")
        if not is_completed_B:
            print(f"    [Check] Guided incomplete: {found_files_B}/{total_required} files (need {min_files_required})")
    
    # Return empty list as found_variables (DataOnly version does not check by variable)
    return is_completed, []


def create_comparison_video_from_rollout(
    out_A: str,
    out_B: str,
    output_video_dir: str,
    num_rollout_steps: int,
    variables: List[str],
    epoch: int = 0,
    fps: float = 2.0,  # frame rate
    video_format: str = "mp4",
    timing_logger: Optional[TimingLogger] = None,
):
    """
    Assemble rollout PNG frames into a comparison video.
    
    For each step and each variable:
    1. Read baseline and guided PNGs
    2. Stack vertically (baseline on top, guided on bottom)
    3. Combine all step frames into a video
    
    Args:
        out_A: Baseline output directory
        out_B: Guided output directory
        output_video_dir: Video output directory
        num_rollout_steps: Total rollout steps
        variables: Variable list
        epoch: Epoch number
        fps: Video frame rate
        video_format: Video format ("mp4" or "avi")
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
        
        # Track timing for each variable's video generation
        if timing_logger:
            timing_logger.start(f"Video: {var}")
        
        # Collect PNG files for all steps
        frames = []
        missing_steps = []
        
        for step_idx in range(num_rollout_steps):
            step_prefix = f"step{step_idx:02d}_"
            pattern = f"{step_prefix}{var}_frame{step_idx}_epoch*.png"
            
            # Use glob to find files (may fail if path contains brackets)
            search_path_A = os.path.join(var_dir_A, pattern)
            search_path_B = os.path.join(var_dir_B, pattern)
            files_A = sorted(glob.glob(search_path_A))
            files_B = sorted(glob.glob(search_path_B))
            
            # If glob fails, use os.listdir + regex as fallback
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
            
            # Read both images
            try:
                img_A = cv2.imread(files_A[0], cv2.IMREAD_COLOR)  # explicitly read as color
                img_B = cv2.imread(files_B[0], cv2.IMREAD_COLOR)
                
                if img_A is None or img_B is None:
                    print(f"  [Video] Warning: Failed to read images for {var} step {step_idx}")
                    print(f"    files_A[0]={files_A[0]}, exists={os.path.exists(files_A[0])}")
                    print(f"    files_B[0]={files_B[0]}, exists={os.path.exists(files_B[0])}")
                    missing_steps.append(step_idx)
                    continue
                
                # Validate image format
                if len(img_A.shape) != 3 or img_A.shape[2] != 3:
                    print(f"  [Video] Warning: img_A has unexpected shape {img_A.shape} for {var} step {step_idx}")
                    missing_steps.append(step_idx)
                    continue
                if len(img_B.shape) != 3 or img_B.shape[2] != 3:
                    print(f"  [Video] Warning: img_B has unexpected shape {img_B.shape} for {var} step {step_idx}")
                    missing_steps.append(step_idx)
                    continue
                
                # Ensure data type is uint8
                if img_A.dtype != np.uint8:
                    img_A = (img_A * 255).astype(np.uint8) if img_A.max() <= 1.0 else img_A.astype(np.uint8)
                if img_B.dtype != np.uint8:
                    img_B = (img_B * 255).astype(np.uint8) if img_B.max() <= 1.0 else img_B.astype(np.uint8)
                
                # Ensure both images have the same width (resize if different)
                if img_A.shape[1] != img_B.shape[1]:
                    target_width = max(img_A.shape[1], img_B.shape[1])
                    img_A = cv2.resize(img_A, (target_width, img_A.shape[0]), interpolation=cv2.INTER_LINEAR)
                    img_B = cv2.resize(img_B, (target_width, img_B.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Stack vertically (baseline on top, guided on bottom)
                combined = np.vstack([img_A, img_B])
                
                # Validate combined frame format
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
        
        # Get video dimensions
        height, width = frames[0].shape[:2]
        print(f"  [Video] {var}: {len(frames)} frames, size={width}x{height}")
        
        # Create video writer
        video_filename = f"{var}_comparison_epoch{epoch}.{video_format}"
        video_path = os.path.join(output_video_dir, video_filename)
        
        # Use PIL to create GIF (most reliable method on this system)
        try:
            from PIL import Image
            
            # Convert frames to PIL Images
            pil_frames = []
            for idx, frame in enumerate(frames):
                # cv2 uses BGR; PIL needs RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Ensure correct data type
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8) if frame_rgb.max() <= 1.0 else frame_rgb.astype(np.uint8)
                pil_img = Image.fromarray(frame_rgb)
                pil_frames.append(pil_img)
            
            # Save as GIF (most reliable animation format)
            gif_path = video_path.replace('.mp4', '.gif').replace('.avi', '.gif')
            duration_ms = int(1000 / fps)  # convert to milliseconds
            
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
        
        # If PIL fails, skip this variable (should not happen since PIL is standard)
        print(f"  [Video] Error: Failed to create animation for {var}, skipping...")
        if timing_logger:
            timing_logger.end(f"Video: {var}")
        continue


# =========================
# ERA5 Batch Local Cache
# =========================

def _era5_cache_time_slug(want_time: str) -> str:
    """Convert want_time string to a valid directory name, e.g. '2019-08-28 00:00:00' -> '20190828_00'."""
    ts = pd.Timestamp(want_time)
    return f"{ts:%Y%m%d_%H}"


def _era5_batch_cache_dir(cache_root: str, want_time: str) -> str:
    return os.path.join(cache_root, _era5_cache_time_slug(want_time))


def load_era5_batch_cache(cache_root: str, want_time: str):
    """
    Try to load the initial batch from cache (eval_inputs, eval_targets, eval_forcings, one_hot_original, ts).
    Returns a tuple or None if cache does not exist.
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
    """Save initial batch to cache."""
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
    Try to load forcings_extended from cache.
    Only hits cache if cached steps >= num_rollout_steps; otherwise returns None (re-download needed).
    """
    if cache_root is None:
        return None
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    # Find a cache file with steps >= num_rollout_steps
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
                # If cached steps exceed request, trim to num_rollout_steps
                if cached_n > num_rollout_steps:
                    forcings = forcings.isel(time=slice(0, num_rollout_steps))
                print(f"    [ERA5 Cache] Loaded forcings (N={cached_n}) from {fpath}")
                return forcings
            except Exception as e:
                print(f"    [ERA5 Cache] Failed to load forcings cache: {e}, will re-download")
                return None
    return None


def save_era5_forcings_cache(cache_root: str, want_time: str, num_rollout_steps: int, forcings_extended):
    """Save forcings_extended to cache."""
    if cache_root is None:
        return
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"forcings_N{num_rollout_steps}.pt")
    torch.save({"forcings_extended": forcings_extended, "want_time": want_time, "num_steps": num_rollout_steps}, fpath)
    print(f"    [ERA5 Cache] Saved forcings (N={num_rollout_steps}) to {fpath}")


def load_era5_gt_cache(cache_root: str, want_time: str, num_rollout_steps: int):
    """
    Try to load gt_list from cache (list[xr.Dataset | None] of length num_rollout_steps).
    Only hits cache if cached steps >= num_rollout_steps.
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
                # Trim if cached steps exceed request
                if cached_n > num_rollout_steps:
                    gt_list = gt_list[:num_rollout_steps]
                print(f"    [ERA5 Cache] Loaded GT list (N={cached_n}) from {fpath}")
                return gt_list
            except Exception as e:
                print(f"    [ERA5 Cache] Failed to load GT cache: {e}, will re-download")
                return None
    return None


def save_era5_gt_cache(cache_root: str, want_time: str, num_rollout_steps: int, gt_list: list):
    """Save gt_list to cache."""
    if cache_root is None:
        return
    cache_dir = _era5_batch_cache_dir(cache_root, want_time)
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"gt_N{num_rollout_steps}.pt")
    torch.save({"gt_list": gt_list, "want_time": want_time, "num_steps": num_rollout_steps}, fpath)
    print(f"    [ERA5 Cache] Saved GT list (N={num_rollout_steps}) to {fpath}")


# =========================
# Context: One-time Load + Reuse
# =========================
@dataclasses.dataclass
class GlobalInputs:
    # —— Save / scheduling —— #
    save_mode: str = "by_dates"             # "by_dates" | "by_params"
    output_dir: str = "/fs/.../general_guidance_output"
    baseline_cache_dir: Optional[str] = None  # Baseline cache dir; None disables cache
    epoch: int = 0
    only_guided: bool = False
    only_baseline: bool = False  # If True, run baseline only, skip guided

    # —— Guidance fixed parameters (non-sweep part) —— #
    guidance_strength: float = 0.7
    readout_collect_steps_for_vis: Tuple[int, ...] = (10, 15)
    guide_loss_type: str = "readout_l2"
    guide_normalize_grad: bool = True
    guide_eps: float = 1e-6

    # —— Data and model —— #
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
    # ERA5 batch cache dir: None disables caching.
    # When enabled, preprocessed batch is saved on first load and reused afterward.
    era5_cache_dir: Optional[str] = None

    full_model_path: Optional[str] = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/debug_result/09-05_training/09-05_training_checkpoint_19k.pt"


@dataclasses.dataclass
class Context:
    params: dict
    readout_guided_inference_fn_factory: any  # generates apply_fn based on readout_collect_steps_for_vis
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
    guidance_mode: str = "steer_one",  # guidance mode (used for directory structure)
    visual_vars: Optional[List[str]] = None,
) -> Tuple[xr.Dataset, Dict[str, xr.Dataset], xr.Dataset, str]:
    """
    Run or load baseline results.
    
    Returns:
        (preds_A, readouts_A, gt_A, out_A)
    """
    # Generate apply_fn
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    
    # Determine out_A directory (for visualization)
    # Note: baseline does not depend on guidance_mode, but pass it for consistent directory structure
    paths = decide_output_dirs(
        output_dir=output_dir,
        save_mode=save_mode,
        want_time=want_time,
        manual_targets=manual_targets,
        readout_collect_steps_for_vis=readout_collect_steps_for_vis,
        inner_idxs=[],  # baseline does not need these
        inner_steps_map={},
        max_opt_steps=1,
        inner_lr=1.0,
        strength=0.0,
        guidance_mode=guidance_mode,  # pass guidance_mode for consistent directory structure
        intensity_scaling_configs=None,
        shift_configs=None,
    )
    out_A = paths["out_A"]
    
    # If cache is enabled, try to load
    if baseline_cache_dir is not None:
        cache_path = _get_baseline_cache_path(
            baseline_cache_dir, want_time, manual_targets, readout_collect_steps_for_vis
        )
        cached = _load_baseline_cache(cache_path)
        if cached is not None:
            preds_A, readouts_A, gt_A = cached
            print(f"    [Baseline] Using cached results, skipping inference")
            # Visualize if needed
            if not (os.path.isdir(out_A) and os.listdir(out_A)):
                print("    Visualizing baseline results...")
                t_vis = time.time()
                # Use original one_hot_original (real storm location) for visualization
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
    
    # Cache not found or disabled - run baseline
    print("\n[Baseline] Running inference (will be cached if cache_dir is set)...")
    t0 = time.time()
    preds_A, readouts_A, gt_A, _ = _run_once_with_guidance(
        params=ctx.params,
        eval_inputs=eval_inputs, eval_targets=eval_targets, eval_forcings=eval_forcings,
        readout_guided_inference_fn=apply_fn,
        target_readout_one_hot=None,  # baseline does not need guidance mask
        readout_collect_steps_for_vis=tuple(readout_collect_steps_for_vis),
        guidance_strength=0.0,
        guide_inner_opt_step_idxs=[],
        guide_inner_opt_steps_map={},
        guide_max_opt_steps=1,
        guide_inner_opt_lr=1.0,
        guide_loss_type="xt_l2",  # baseline uses xt_l2, no readout_l2 needed
        guide_normalize_grad=True,
        guide_eps=1e-6,
    )
    print(f"    -> Inference done in {time.time() - t0:.2f}s")
    
    # Save cache if enabled
    if baseline_cache_dir is not None:
        cache_path = _get_baseline_cache_path(
            baseline_cache_dir, want_time, manual_targets, readout_collect_steps_for_vis
        )
        _save_baseline_cache(cache_path, preds_A, readouts_A, gt_A)
    
    # Visualize
    if not (os.path.isdir(out_A) and os.listdir(out_A)):
        print("    Visualizing baseline results...")
        t_vis = time.time()
        # Use original one_hot_original (real storm location) for visualization
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
    # —— Save / scheduling —— #
    save_mode: str,
    output_dir: str,
    epoch: int,
    only_guided: bool,
    only_baseline: bool,  # If True, run baseline only, skip guided

    # —— Time (single time point only) —— #
    want_time: str,  # start_date
    end_date: Optional[str] = None,  # target time point (if None, computed as start_date + 12h)

    # —— Manual Guidance targets —— #
    manual_targets: List[Dict],

    # —— Selective Storm Guidance parameters —— #
    guidance_mode: str = "steer_one",  # mode selection: "steer_one" or "delete_one"
    selected_storm_id: Optional[int] = None,  # storm ID to guide (if None, requires IO input)
    guidance_mask_padding: int = 5,  # bounding box padding
    use_guidance_mask: bool = True,  # whether to use mask to restrict loss region

    # —— Guidance parameters (including sweep) —— #
    guidance_strength: float,
    readout_collect_steps_for_vis: Tuple[int, ...],
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],  # per-step optimization count
    guide_max_opt_steps: int,                    # max count (used for JAX compilation)
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,

    # —— Visualization —— #
    visual_vars: Optional[List[str]] = None,
    
    # —— Baseline cache (optional) —— #
    baseline_preds: Optional[xr.Dataset] = None,
    baseline_readouts: Optional[Dict[str, xr.Dataset]] = None,
    baseline_gt: Optional[xr.Dataset] = None,
    baseline_out_A: Optional[str] = None,
    
    # —— Pre-loaded data (optional, to avoid repeated loading) —— #
    eval_inputs: Optional[xr.Dataset] = None,
    eval_targets: Optional[xr.Dataset] = None,
    eval_forcings: Optional[xr.Dataset] = None,
    one_hot_original: Optional[torch.Tensor] = None,
) -> Dict[str, str | bool]:
    t_run_start = time.time()
    
    # Compute end_date if not provided (start_date + 12h)
    if end_date is None:
        start_ts = pd.Timestamp(want_time)
        end_ts = start_ts + pd.Timedelta(hours=12)
        end_date = str(end_ts)
    print(f"\n[Selective Guidance] start_date: {want_time}, end_date: {end_date}")
    
    # Get storm_id (config priority -> cache -> IO input)
    storm_id = get_storm_id_for_date_pair(want_time, end_date, selected_storm_id, ctx)
    
    # Extract selected storm info
    storms = extract_storms_from_end_date(ctx, end_date)
    if storm_id < 0 or storm_id >= len(storms):
        raise ValueError(f"Invalid storm_id: {storm_id}, available: 0-{len(storms)-1}")
    selected_storm = storms[storm_id]
    print(f"    [Selected Storm] ID {storm_id}: lat={selected_storm['lat']:.1f}, "
          f"lon={selected_storm['lon']:.1f}, rsize={selected_storm['rsize']:.1f}")
    
    # Load data if not pre-provided
    if eval_inputs is None or eval_targets is None or eval_forcings is None or one_hot_original is None:
        # Check date availability (single time point only)
        check = detect_want_times(ctx.eval_ds, [want_time])
        for r in check:
            print(f"  - {r['time']}: ok={r['ok']}  | in_tracks={r['in_tracks']}  | full_window={r['has_full_window']}  | n_samples={r['n_samples']} "
                  f"{'| reason='+r['reason'] if r['reason'] else ''}")
        if any(not r["ok"] for r in check):
            raise ValueError("Specified time is not available, please correct and re-run.")

        # Load single time point data
        print("\n[Run Step 1] Loading eval batch...")
        t0 = time.time()
        wanted_loader = build_wanted_subloader(ctx.eval_ds, [want_time], batch_size=1)
        it = iter(wanted_loader)
        eval_inputs, eval_targets, eval_forcings, one_hot_original, ts = next(it)
        print(f"    -> Done in {time.time() - t0:.2f}s")
    else:
        print("\n[Run Step 1] Using pre-loaded eval batch (skipping data loading)")
    
    # Extract all storms from start_date one_hot_original
    print(f"\n[Run Step 1.2] Extracting storms from start_date...")
    start_storms = extract_storms_from_one_hot(one_hot_original)
    print(f"    [Start Storms] Found {len(start_storms)} storm(s) in start_date")
    for storm in start_storms:
        print(f"      Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, rsize={storm['rsize']:.1f}")
    
    # Match storms between start_date and end_date
    print(f"\n[Run Step 1.3] Matching storms between start_date and end_date...")
    selected_storm_id_in_start = match_storm_between_dates(start_storms, selected_storm, distance_threshold=10.0)
    
    # Build target mask based on mode
    print(f"\n[Run Step 1.4] Building final target mask (mode: {guidance_mode})...")
    batch_size = eval_inputs.sizes["batch"]
    
    if guidance_mode == "steer_one":
        # steer_one mode: move selected storm
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
        target_storm = manual_targets[0]  # user-specified target
        
    elif guidance_mode == "delete_one":
        # delete_one mode: remove selected storm
        one_hot_for_guidance = build_final_target_mask(
            one_hot_original,
            start_storms,
            mode="delete_one",
            selected_storm_id_in_start=selected_storm_id_in_start,
            manual_target=None,  # delete_one mode does not need manual_target
            batch_size=batch_size,
        )
        num_kept_storms = len(start_storms) - (1 if selected_storm_id_in_start is not None else 0)
        print(f"    [Delete One Mode] Final target mask contains: {num_kept_storms} storm(s) from start_date (deleted storm_id {selected_storm_id_in_start})")
        # In delete_one mode, target_storm is used for bounding box computation (original storm location)
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
    
    one_hot_for_vis = one_hot_original  # original mask for visualization comparison
    print(f"    [Guidance Mode: {guidance_mode}] targets: {manual_targets if guidance_mode == 'steer_one' else 'N/A (delete_one mode)'}")
    
    # Compute bounding box mask (covering selected original storm and target storm)
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

    # Directory routing (SAVE_MODE)
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
        guidance_mode=guidance_mode,  # pass guidance_mode
        intensity_scaling_configs=None,
        shift_configs=None,
    )
    root = paths["root"]; out_A = paths["out_A"]; out_B = paths["out_B"]
    print("[SAVE_MODE]", save_mode, "| root:", root)
    os.makedirs(root, exist_ok=True)

    # Generate apply_fn for this readout_collect_steps_for_vis
    print("\n[Run Step 2] Building JIT apply_fn...")
    t0 = time.time()
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    print(f"    -> Done in {time.time() - t0:.2f}s")

    # Baseline: use cached results if provided, otherwise run
    if not only_guided:
        if baseline_preds is not None and baseline_readouts is not None and baseline_gt is not None:
            # Use provided baseline results (from cache)
            print("\n[Run Step 3a] Using cached baseline results (skipping inference)...")
            preds_A, readouts_A, gt_A = baseline_preds, baseline_readouts, baseline_gt
            # Use provided out_A path if given
            if baseline_out_A is not None:
                out_A = baseline_out_A
        else:
            # Run baseline (backward compatibility, or when cache is disabled)
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
                guide_loss_type="xt_l2",  # baseline uses xt_l2, no readout_l2 needed
                guide_normalize_grad=guide_normalize_grad,
                guide_eps=guide_eps,
                guidance_mask=None,  # baseline does not use mask
            )
            print(f"    -> Inference done in {time.time() - t0:.2f}s")
        
        # Visualization can be skipped if the directory already exists and is non-empty
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

    # Guided: skip if only_baseline=True
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
            guidance_mask=guidance_mask,  # pass bounding box mask
        )
        print(f"    -> Inference done in {time.time() - t0:.2f}s")
        
        # Save and visualize loss
        if loss_history_B:
            print("\n[Run Step 3c] Saving loss history...")
            loss_curve_path = os.path.join(out_B, "loss_curves.png")
            loss_summary_path = os.path.join(out_B, "loss_summary.txt")
            loss_json_path = os.path.join(out_B, "loss_history.json")
            
            plot_loss_curves(loss_history_B, loss_curve_path)
            save_loss_summary(loss_history_B, loss_summary_path)
            
            # Save as JSON (convenient for downstream analysis)
            with open(loss_json_path, 'w') as f:
                json.dump(loss_history_B, f, indent=2)
            print(f"[Loss JSON] Saved to {loss_json_path}")
        else:
            print("\n[Run Step 3c] No loss history to save (strength=0 or no optimization)")
        
        print("    Visualizing guided results...")
        t_vis = time.time()
        _visualize_and_export(
            gt_list=[gt_B], predictions_list=[preds_B], readouts_list=[readouts_B],
            one_hot_torch=one_hot_for_guidance,  # visualize with guidance mask
            out_dir=out_B,
            epoch=epoch,
            visual_vars=visual_vars,
        )
        print(f"    -> Visualization done in {time.time() - t_vis:.2f}s")
    else:
        print("\n[Run Step 3b] Skipping Guided inference (only_baseline=True)")
        preds_B, readouts_B, gt_B, loss_history_B = None, None, None, None

    # ==== Create comparison visualizations (requires both baseline and guided) ====
    if not only_guided and not only_baseline:
        print("\n[Run Step 4] Creating comparison visualizations...")
        t_comp = time.time()
        # Each B_ setting has its own comparison directory
        comparison_dir = os.path.join(out_B, "comparison_visualizations")
        os.makedirs(comparison_dir, exist_ok=True)

        # Build readout_frames dict format
        readout_frames_A = {
            key: [readouts_A[key].isel(time=0, batch=0)]
            for key in readouts_A.keys()
        }
        readout_frames_B = {
            key: [readouts_B[key].isel(time=0, batch=0)]
            for key in readouts_B.keys()
        }

        # Readout comparison: original storm location vs guidance target location
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

        # Weather variable comparison - filter with visual_vars
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
    # —— Save / scheduling —— #
    save_mode: str,
    output_dir: str,
    epoch: int,
    only_guided: bool,
    only_baseline: bool,

    # —— Time —— #
    want_time: str,
    end_date: Optional[str] = None,

    # —— Rollout-specific parameters —— #
    num_rollout_steps: int,  # number of rollout steps (e.g. 10 steps = 5 days)
    guidance_on_first_step: bool,  # whether to apply guidance on the first step

    # —— Manual Guidance targets —— #
    manual_targets: List[Dict],

    # —— Selective Storm Guidance parameters —— #
    guidance_mode: str = "steer_one",
    selected_storm_id: Optional[int] = None,
    guidance_mask_padding: int = 5,
    use_guidance_mask: bool = True,

    # —— Guidance parameters —— #
    guidance_strength: float,
    readout_collect_steps_for_vis: Tuple[int, ...],
    guide_inner_opt_step_idxs: List[int],
    guide_inner_opt_steps_map: Dict[int, int],
    guide_max_opt_steps: int,
    guide_inner_opt_lr: float,
    guide_loss_type: str,
    guide_normalize_grad: bool,
    guide_eps: float,

    # —— Visualization —— #
    visual_vars: Optional[List[str]] = None,
    vis_n_procs: int = 1,
    vis_contourf_levels: int = 36,
    vis_coastlines_resolution: str = "50m",
    vis_draw_gridlabels: bool = True,
    vis_add_borders: bool = True,
    vis_dpi: int = 240,
    
    # —— Pre-loaded data (required) —— #
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    forcings_extended: xr.Dataset,  # rollout-specific: extended forcings
    one_hot_original: torch.Tensor,
    
    # —— Timing logger (optional) —— #
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

    # —— ERA5 local cache —— #
    era5_cache_dir: Optional[str] = None,
) -> Dict[str, str | bool]:
    """
    [ROLLOUT ONLY] Execute multi-step rollout with optional first-step guidance.
    
    High-level rollout wrapper similar to run_once_with_context but supports multi-step prediction:
    - Baseline path: no guidance, rollout for num_rollout_steps steps
    - Guided path: apply guidance on first step, then continue rollout
    - Generate comparison visualizations
    
    Key differences from run_once_with_context:
    1. Accepts forcings_extended (time=num_rollout_steps) instead of single-step forcings
    2. Uses _run_rollout_with_guidance instead of _run_once_with_guidance
    3. Returns predictions_list as multi-step results
    
    Args:
        ctx: Context object
        num_rollout_steps: rollout steps (e.g. 10 steps = 5 days at 12h per step)
        guidance_on_first_step: whether to apply guidance on first step (no guidance on subsequent steps)
        forcings_extended: extended forcings (batch, time=num_rollout_steps, ...)
        ... (other args same as run_once_with_context)
    
    Returns:
        result_dict: dict containing output directory paths and other info
    
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
    
    # If no logger provided, create a temporary one (no save)
    if timing_logger is None:
        timing_logger = TimingLogger()
    
    print(f"\n[ROLLOUT Context] Starting {num_rollout_steps}-step rollout")
    print(f"[ROLLOUT Context] Guidance on first step: {guidance_on_first_step}")
    
    # ===== Pre-load all GT data for visualization (with local cache support) =====
    print(f"\n[Rollout GT Loading] Pre-loading GT data for {num_rollout_steps} steps...")
    if timing_logger:
        timing_logger.start("GT loading")
    
    gt_list_for_visualization = []
    start_ts = pd.Timestamp(want_time)
    
    # Build target timestamps for all rollout steps
    target_time_strs = []
    for step_idx in range(num_rollout_steps):
        target_time = start_ts + pd.Timedelta(hours=12 * (step_idx + 1))
        target_time_str = str(target_time)
        target_time_strs.append(target_time_str)
    
    # Try loading GT from cache
    _gt_cached = load_era5_gt_cache(era5_cache_dir, want_time, num_rollout_steps)
    if _gt_cached is not None:
        gt_list_for_visualization = _gt_cached
        print(f"  [GT Loading] Loaded {len(gt_list_for_visualization)} GT frames from cache")
    else:
        # Check which time points are available
        checks = detect_want_times(ctx.eval_ds, target_time_strs)
        available_indices = []
        
        for step_idx, check in enumerate(checks):
            if check["ok"]:
                available_indices.append(step_idx)
            else:
                print(f"  [Warning] Step {step_idx+1}: GT data not available for {target_time_strs[step_idx]} ({check.get('reason', 'unknown')})")
        
        # Batch-load available time points
        gt_dict = {}
        if available_indices:
            available_time_strs = [target_time_strs[i] for i in available_indices]
            print(f"  [GT Loading] Batch loading {len(available_indices)}/{num_rollout_steps} time points from GCS...")
            
            try:
                wanted_loader = build_wanted_subloader(ctx.eval_ds, available_time_strs, batch_size=1)
                it = iter(wanted_loader)
                
                # Read all available GT data in order
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
                # Fall back to sequential loading
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
        
        # Build gt_list in order (handle missing time points)
        for step_idx in range(num_rollout_steps):
            if step_idx in gt_dict:
                gt_list_for_visualization.append(gt_dict[step_idx])
            else:
                # Use eval_targets as template, fill with NaN
                gt_frame = eval_targets.isel(time=0, batch=0) * np.nan
                gt_list_for_visualization.append(gt_frame)
                print(f"  [Step {step_idx+1}/{num_rollout_steps}] Using NaN for missing GT data")
        
        # Save to local cache
        save_era5_gt_cache(era5_cache_dir, want_time, num_rollout_steps, gt_list_for_visualization)
    
    if timing_logger:
        timing_logger.end("GT loading")
    print(f"[Rollout GT Loading] ✓ Pre-loaded {len(gt_list_for_visualization)} GT frames")
    
    # Compute end_date if not provided
    if end_date is None:
        start_ts = pd.Timestamp(want_time)
        end_ts = start_ts + pd.Timedelta(hours=12)
        end_date = str(end_ts)
    print(f"\n[Selective Guidance] start_date: {want_time}, end_date: {end_date}")
    
    # Get storm_id (config priority -> cache -> IO input)
    if timing_logger:
        timing_logger.start("Get storm_id")
    storm_id = get_storm_id_for_date_pair(want_time, end_date, selected_storm_id, ctx)
    if timing_logger:
        timing_logger.end("Get storm_id")
    
    # Extract selected storm info
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
    
    # Extract all storms from start_date one_hot_original
    print(f"\n[Run Step 1] Extracting storms from start_date...")
    if timing_logger:
        timing_logger.start("Step 1: Extract storms from start_date")
    start_storms = extract_storms_from_one_hot(one_hot_original)
    if timing_logger:
        timing_logger.end("Step 1: Extract storms from start_date")
    print(f"    [Start Storms] Found {len(start_storms)} storm(s) in start_date")
    for storm in start_storms:
        print(f"      Storm ID {storm['storm_id']}: lat={storm['lat']:.1f}, lon={storm['lon']:.1f}, rsize={storm['rsize']:.1f}")
    
    # Match storms between start_date and end_date
    print(f"\n[Run Step 2] Matching storms between start_date and end_date...")
    if timing_logger:
        timing_logger.start("Step 2: Match storms")
    selected_storm_id_in_start = match_storm_between_dates(start_storms, selected_storm, distance_threshold=10.0)
    if timing_logger:
        timing_logger.end("Step 2: Match storms")
    
    # Build target mask based on mode
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
    
    # Compute bounding box mask
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

    # Directory routing
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
    
    # ===== Check if rollout inference and visualization are already completed =====
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
            
            # Save timing log to file
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
                "skipped": True,  # marked as skipped
            }

    # Generate apply_fn
    print("\n[Run Step 4] Building JIT apply_fn...")
    if timing_logger:
        timing_logger.start("Step 4: Build JIT apply_fn")
    apply_fn = ctx.readout_guided_inference_fn_factory(tuple(readout_collect_steps_for_vis))
    if timing_logger:
        timing_logger.end("Step 4: Build JIT apply_fn")

    # ===== Baseline: rollout without guidance =====
    if not only_guided:
        # Check if baseline is already completed
        print("\n[Check] Checking if baseline rollout is already completed...")
        # Check at least 80% of steps (or at least 3, whichever is larger)
        min_steps_to_check = max(3, int(num_rollout_steps * 0.8))
        baseline_completed = check_single_rollout_completed(
            out_A, num_rollout_steps, min_steps_required=min_steps_to_check
        )
        if baseline_completed:
            print(f"  [Skip] Baseline rollout already completed, skipping inference")
            print(f"  Data directory: {os.path.join(out_A, 'rollout_data')}")
            # Skip baseline inference but continue to guided if needed
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
                guidance_on_first_step=False,  # no guidance
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
                intensity_scaling_configs=None,  # Baseline does not use intensity scaling
                shift_configs=None,  # Baseline does not use spatial shift
                warp_configs=None,  # Baseline does not use warp
            )
            if timing_logger:
                timing_logger.end("Step 5a: Baseline rollout")
            
            # Save all rollout data (replaces visualization)
            # Baseline has no guidance, so do not save readout
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
                save_readout_steps=None,  # Baseline has no guidance, no readout to save
                shift_configs=None,    # Baseline does not use spatial shift
            )
            if timing_logger:
                timing_logger.end("Step 5a-save: Baseline data saving")

    # ===== Guided: first-step guidance + subsequent rollout =====
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
            guidance_on_first_step=guidance_on_first_step,  # apply guidance on first step
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
            intensity_scaling_configs=intensity_scaling_configs,
            shift_configs=shift_configs,
            warp_configs=warp_configs,
        )
        if timing_logger:
            timing_logger.end("Step 5b: Guided rollout")
        
        # Save loss history
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
        
        # Save all rollout data (replaces visualization)
        # Only save readout for steps with guidance (first step only)
        print("\n[Run Step 5b-save] Saving guided rollout data...")
        if timing_logger:
            timing_logger.start("Step 5b-save: Guided data saving")
        
        # Determine which steps have guidance (only first step if guidance_on_first_step is True)
        readout_steps_to_save = []
        if guidance_on_first_step:
            readout_steps_to_save = [0]  # only save readout for step 0
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
                "guidance_on_first_step": guidance_on_first_step,
                "readout_steps_saved": readout_steps_to_save,
                "intensity_scaling_configs": intensity_scaling_configs,
                "shift_configs": shift_configs,
            },
            save_readout_steps=readout_steps_to_save,  # only save steps with guidance
            shift_configs=shift_configs,  # pass shift configs for saving visualization info
        )
        if timing_logger:
            timing_logger.end("Step 5b-save: Guided data saving")
    
    # ===== Skip video generation (handled in visualization script) =====
    print("\n[Run] Skipping visualization and video generation (use visualization script separately)")
    
    # Save timing log to file
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
    # CONFIGURATION SECTION - Modify parameters here to configure the experiment
    # ═════════════════════════════════════════════════════════════════════════════

    # -------------------------------------------------------------------------
    # 0. Root directory configuration (Root Path - modify here for different machines)
    # -------------------------------------------------------------------------
    # All local paths are based on this root directory; only change this line when migrating
    ROOT_DIR: str = "/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl"

    # -------------------------------------------------------------------------
    # 0. Guidance Method Selection (core mode selection - mutually exclusive)
    # -------------------------------------------------------------------------
    # Options (choose one, mutually exclusive):
    #   - "none": Baseline only, no guidance
    #   - "direct_optim": Directly optimize x_t (original readout-based guidance)
    #   - "input_manipulation": Input condition modification (Intensity Scaling + Spatial Shift)
    #   - "local_affine": Local affine transform optimization (optimize warp parameters)
    GUIDANCE_METHOD: str = "none"
    
    # -------------------------------------------------------------------------
    # 1. Output variable configuration (used for metadata only)
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
    # 2. Guidance mode configuration (for direct_optim mode)
    # -------------------------------------------------------------------------
    GUIDANCE_MODE: str = "steer_one"  # "steer_one" or "delete_one"
    SELECTED_STORM_IDS: List[Optional[int]] = [1]  # storm ID to guide (None = manual input required)
    GUIDANCE_MASK_PADDING: int = 2
    USE_GUIDANCE_MASK: bool = False
    
    # -------------------------------------------------------------------------
    # 3. Target location configuration - Storm Dorian
    # -------------------------------------------------------------------------
    # MANUAL_TARGETS: List[Dict] = [
    #     # {"lat": 15.0, "lon": 296.0, "radius": 3},
    #      {"lat": 25.0, "lon": 296.0, "radius": 3},
    # ]
    # Affine_Center_Lat, Affine_Center_Lon = 1,1
    # SWEEP_WANT_TIMES: List[str] = ["2019-08-28 00:00:00"]  
    # END_DATES: List[Optional[str]] = ["2019-08-29 00:00:00"]
    


    # -------------------------------------------------------------------------
    # 3. Target location configuration - Storm Irma 2017-09-07
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
    # 5. Rollout configuration
    # -------------------------------------------------------------------------
    ENABLE_ROLLOUT: bool = True
    ROLLOUT_DAYS: int = 4
    NUM_ROLLOUT_STEPS: int = ROLLOUT_DAYS * 2
    GUIDANCE_ON_FIRST_STEP: bool = True
    ONLY_BASELINE: bool = True
    
    # =========================================================================
    # MODE-SPECIFIC CONFIGURATIONS (select based on GUIDANCE_METHOD)
    # =========================================================================
    if GUIDANCE_METHOD == "direct_optim":
        # ---------------------------------------------------------------------
        # Mode 1: Direct Optimization configuration
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] direct_optim (directly optimize x_t)")
        
        # Format: [(start_step, end_step, opt_steps), ...]
        SWEEP_GUIDE_OPT_CONFIGS = [
            [(1, 1, 1)],  # example: step 1 does 1 optimization iteration
        ]
        
        READOUT_COLLECT_STEPS_FOR_VIS = (19)
        SWEEP_GUIDE_LR = [0.000000000000000001]  # learning rate
        SWEEP_RANDOM_SEEDS = [789, 1000, 100, xxx, xxx, 123, 213]  # Random seeds
        
        # Not using input_manipulation
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        WARP_CONFIGS = None
        
    elif GUIDANCE_METHOD == "input_manipulation":
        # ---------------------------------------------------------------------
        # Mode 2: Input Manipulation configuration
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] input_manipulation (input condition modification)")
        
        # Intensity Scaling configuration
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
        
        # Spatial Shift configuration
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
        
        # Not using direct_optim optimization parameters (but still need to provide defaults)
        SWEEP_GUIDE_OPT_CONFIGS = [[(1, 1, 1)]]  # placeholder, will not be used
        READOUT_COLLECT_STEPS_FOR_VIS = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
        SWEEP_GUIDE_LR = [0.000000000000000001]
        SWEEP_RANDOM_SEEDS = [789]
        WARP_CONFIGS = None
        
    elif GUIDANCE_METHOD == "local_affine":
        # ---------------------------------------------------------------------
        # Mode 3: Local Affine Warp configuration
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] local_affine (local affine transform optimization)")
        
        # Storm Irma
        WARP_CONFIGS = {
            "enabled": True,
            "center_lat": Affine_Center_Lat, 
            "center_lon": Affine_Center_Lon,
            "radius": 8.0,
            "init_translation": [0.0, 0.0],  # initial translation [dlat, dlon]
            "init_rotation": 0.0,            # initial rotation (radians)
            "init_scale": [1.0, 1.0],        # initial scale [sx, sy]
            "optimize_translation": True,
            "optimize_rotation": False,
            "optimize_scale": False,
            "learning_rate": 5e-2,
            # "learning_rate": 1e-10,
            "regularization_weight": 1e-3,
        }
        
        # Denoising-related configuration still required
        # SWEEP_GUIDE_OPT_CONFIGS = [[(5, 6, 16)]]
        SWEEP_GUIDE_OPT_CONFIGS = [[(3, 4, 75)]]
        READOUT_COLLECT_STEPS_FOR_VIS = (2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
        SWEEP_GUIDE_LR = [1.0]  # warp learning rate
        SWEEP_RANDOM_SEEDS = [789]
        
        # Not using input_manipulation
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        
    elif GUIDANCE_METHOD == "none":
        # ---------------------------------------------------------------------
        # Mode: None (Baseline only)
        # ---------------------------------------------------------------------
        print(f"\n[Guidance Method] none (Baseline only, no guidance)")
        
        # Provide defaults (will not be used)
        SWEEP_GUIDE_OPT_CONFIGS = [[(1, 1, 1)]]
        READOUT_COLLECT_STEPS_FOR_VIS = (19,)  # only collect the last denoising step
        SWEEP_GUIDE_LR = [0.000000000000000001]
        SWEEP_RANDOM_SEEDS = [789]
        INTENSITY_SCALING_CONFIGS = []
        SHIFT_CONFIGS = []
        WARP_CONFIGS = None
        
    else:
        raise ValueError(f"Unknown GUIDANCE_METHOD: {GUIDANCE_METHOD}. "
                        f"Must be one of: 'none', 'direct_optim', 'input_manipulation', 'local_affine'")
    
    # -------------------------------------------------------------------------
    # 8. Guidance optimization hyperparameter parsing (general)
    # -------------------------------------------------------------------------
    def parse_guide_config(config):
        """Derive step_idxs, steps_map, max_opt_steps from config."""
        steps_map = {s: n for start, end, n in config for s in range(start, end + 1)}
        step_idxs = sorted(steps_map.keys())
        max_opt_steps = max(steps_map.values()) if steps_map else 1
        return step_idxs, steps_map, max_opt_steps
    
    # -------------------------------------------------------------------------
    # 9. Visualization parameters (for metadata only)
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
    # 10. Global configuration (data and model paths)
    # -------------------------------------------------------------------------
    G = GlobalInputs(
        save_mode="by_dates",
        output_dir=os.path.join(ROOT_DIR, "0305-02—LocalAffine_RollOut"),
        baseline_cache_dir=os.path.join(ROOT_DIR, "0305-02—LocalAffine_RollOut", "_baseline_cache"),
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
    # Print configuration summary
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
    
    # Prepare context (load model and dataset)
    print("\n[Main] Preparing context...")
    t_prepare = time.time()
    ctx = prepare_context(G)
    print(f"[Main] Context ready in {time.time() - t_prepare:.2f}s\n")


    # Iterate over time points
    for i, want_time in enumerate(SWEEP_WANT_TIMES):
        print(f"\n{'='*80}")
        print(f"[TIME] {want_time}")
        print(f"[TARGETS] {len(MANUAL_TARGETS)} target(s): {MANUAL_TARGETS}")
        print(f"{'='*80}")
        
        end_date = END_DATES[i] if i < len(END_DATES) else (END_DATES[0] if END_DATES else None)
        selected_storm_id = SELECTED_STORM_IDS[i] if i < len(SELECTED_STORM_IDS) else (SELECTED_STORM_IDS[0] if SELECTED_STORM_IDS else None)
        
        # Load data (once per time point, with local cache support)
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
        
        # Load extended forcings (for rollout, with local cache support)
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
        
        # Process target locations
        targets_to_process = (
            [{"lat": 0.0, "lon": 0.0, "radius": 1}]  # placeholder (delete_one mode does not need targets)
            if GUIDANCE_MODE == "delete_one" and len(MANUAL_TARGETS) == 0
            else MANUAL_TARGETS
        )
        
        # Iterate over each target location
        for target_idx, manual_target in enumerate(targets_to_process):
            print(f"\n[TARGET {target_idx+1}/{len(targets_to_process)}] {manual_target}")
            current_targets = [manual_target]
            
            # Run or load baseline (only once per (want_time, manual_target) combination)
            # Note: baseline does not actually depend on target location, but for clear directory structure
            # [ROLLOUT] In rollout mode, single-step baseline is not needed; rollout has its own baseline
            if not G.only_guided and not (ENABLE_ROLLOUT and forcings_extended is not None):
                print(f"\n[Baseline] Running or loading baseline for target {target_idx+1}...")
                preds_A, readouts_A, gt_A, out_A = run_or_load_baseline(
                    ctx=ctx,
                    want_time=want_time,
                    manual_targets=current_targets,  # pass list with single target
                    eval_inputs=eval_inputs,
                    eval_targets=eval_targets,
                    eval_forcings=eval_forcings,
                    one_hot_original=one_hot_original,
                    readout_collect_steps_for_vis=G.readout_collect_steps_for_vis,
                    baseline_cache_dir=G.baseline_cache_dir,
                    output_dir=G.output_dir,
                    save_mode=G.save_mode,
                    epoch=G.epoch,
                    guidance_mode=GUIDANCE_MODE,  # pass guidance_mode
                    visual_vars=VISUAL_VARS,
                )
            else:
                preds_A, readouts_A, gt_A, out_A = None, None, None, None
            
            # Parameter loop (reuse baseline results)
            config_count = 0  # track config count for periodic memory cleanup
            for guide_config_idx, guide_config in enumerate(SWEEP_GUIDE_OPT_CONFIGS):
                step_idxs, steps_map, max_opt_steps = parse_guide_config(guide_config)
                for lr_idx, lr in enumerate(SWEEP_GUIDE_LR):
                    for seed_idx, random_seed in enumerate(SWEEP_RANDOM_SEEDS):
                        config_count += 1
                        # Generate tag: show config summary (including target location and seed)
                        config_str = "_".join(f"{s}-{e}x{n}" for s, e, n in guide_config)
                        target_slug = _targets_slug(current_targets)
                        tag = f"time[{_short_time_label(want_time)}]__target_{target_slug}__cfg[{config_str}]_lr{lr}__seed{random_seed}"
                        print(f"\n===== [RUN] {tag} =====")
                        print(f"  Target: {manual_target}")
                        print(f"  Random Seed: {random_seed}")
                        print(f"  step_idxs: {step_idxs}")
                        print(f"  steps_map: {steps_map}")
                        print(f"  max_opt_steps: {max_opt_steps}")
                    
                        # ===== [ROLLOUT] Select single-step or multi-step inference based on config =====
                        if ENABLE_ROLLOUT and forcings_extended is not None:
                            print(f"\n[Mode] ROLLOUT ({NUM_ROLLOUT_STEPS} steps)")
                            
                            # Create timing logger (one per config)
                            # Determine output directory first to save timing log
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
                            # Rollout-specific parameters
                            num_rollout_steps=NUM_ROLLOUT_STEPS,
                            guidance_on_first_step=GUIDANCE_ON_FIRST_STEP,
                            # Guidance parameters
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
                            # Visualization parameters
                            vis_n_procs=VIS_N_PROCS,
                            vis_contourf_levels=VIS_CONTOURF_LEVELS,
                            vis_coastlines_resolution=VIS_COASTLINES_RESOLUTION,
                            vis_draw_gridlabels=VIS_DRAW_GRIDLABELS,
                            vis_add_borders=VIS_ADD_BORDERS,
                            vis_dpi=VIS_DPI,
                            # Pre-loaded data
                            eval_inputs=eval_inputs,
                            eval_targets=eval_targets,
                            forcings_extended=forcings_extended,  # use extended forcings
                            one_hot_original=one_hot_original,
                            # Timing logger
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
                            # ERA5 local cache
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
                                manual_targets=current_targets,  # pass list with single target
                                guidance_mode=GUIDANCE_MODE,  # pass mode
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
                                # Pass baseline results if available
                                baseline_preds=preds_A,
                                baseline_readouts=readouts_A,
                                baseline_gt=gt_A,
                                baseline_out_A=out_A,
                                # Pass pre-loaded data (avoid repeated loading)
                                eval_inputs=eval_inputs,
                                eval_targets=eval_targets,
                                eval_forcings=eval_forcings,
                                one_hot_original=one_hot_original,
                            )
                        print("[RESULT]", json.dumps(res, indent=2, ensure_ascii=False))
                        
                        # ===== Memory cleanup: immediately after each config run =====
                        # Release result variables
                        del res
                        if 'timing_logger' in locals():
                            del timing_logger
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Clear JAX cache every 3 configs (avoid too frequent calls)
                        if config_count % 3 == 0:
                            try:
                                jax.clear_backends()
                                print(f"  [Memory] Cleared JAX backends cache (after {config_count} configs)")
                            except Exception as e:
                                print(f"  [Memory] Warning: Failed to clear JAX backends: {e}")
                        
                        # Perform deeper memory cleanup every 5 configs
                        if config_count % 5 == 0:
                            try:
                                # Try to clear GPU cache (if using GPU)
                                _ = jnp.array(0).block_until_ready()
                                print(f"  [Memory] Performed deep memory cleanup (after {config_count} configs)")
                            except Exception as e:
                                pass  # ignore errors, do not interrupt main flow
            
            # ===== Cleanup after each target is processed =====
            # Release baseline results if no longer needed
            if not G.only_guided:
                del preds_A, readouts_A, gt_A
            del current_targets
            gc.collect()
            print(f"  [Memory] Cleaned up after target {target_idx+1}/{len(targets_to_process)}")
        
        # Cleanup after each time point is processed
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


