"""
vis_guidance.py
用于 guidance 实验的可视化对比函数（无 guidance vs 有 guidance 上下对比）
"""

import os
from typing import Dict, List, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point


def create_readout_comparison_plot(
    no_guidance_readout_frames_dict: Dict[int, List[xr.Dataset]],
    guided_readout_frames_dict: Dict[int, List[xr.Dataset]],
    one_hot_t1: np.ndarray,
    one_hot_t2: np.ndarray,
    output_dir: str,
    epoch: int = 0,
    ts: Optional[str] = None,
    steps: Optional[List[int]] = None,
    sample_idx: int = 0,
    cmap_mask: str = "Reds",
    cmap_prob: str = "RdBu_r",
):
    """
    创建无 guidance vs 有 guidance 的 readout 对比图。
    
    布局：
    - 第一列：Mask（第一行显示 t1 的 GT，第二行显示 t2 的 Guidance Target）
    - 后续列：各 step 的 readout
    - 第一行：No Guidance（使用 t1 时刻的 inputs 预测）
    - 第二行：With Guidance（使用 t2 时刻的 storm location 作为 guidance target）
    
    Args:
        one_hot_t1: t1 时刻的 one-hot mask（No Guidance 行显示）
        one_hot_t2: t2 时刻的 one-hot mask（Guidance Target，With Guidance 行显示）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定 steps，从 dict keys 中获取
    if steps is None:
        steps = sorted(no_guidance_readout_frames_dict.keys())
    
    # —— 内部工具函数 —— #
    def as_2d_latlon(da: xr.DataArray):
        if "level" in da.dims:
            da = da.mean("level")
        for d in ("batch", "sample", "time"):
            if d in da.dims and da.sizes[d] == 1:
                da = da.isel({d: 0})
        if "time" in da.dims:
            da = da.isel(time=0)
        
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(f"DataArray 缺少 lat/lon 维：dims = {da.dims}")
        
        lats = da["lat"].values
        lons = da["lon"].values
        arr = np.asarray(da.values)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2:])
        return arr, lats, lons

    def make_prob_2d(ds: xr.Dataset):
        u = ds["u_component_of_wind"]
        v = ds["v_component_of_wind"]
        u2d, lats, lons = as_2d_latlon(u)
        v2d, _, _ = as_2d_latlon(v)
        logits = np.stack([u2d, v2d], axis=-1)
        exps = np.exp(logits - logits.max(-1, keepdims=True))
        probs = exps / exps.sum(-1, keepdims=True)
        return probs[..., 1], lats, lons

    # —— 准备两个 mask —— #
    mask_t1 = one_hot_t1[sample_idx, ..., 1]
    mask_t1 = np.asarray(mask_t1, dtype=np.float32)
    if mask_t1.ndim > 2:
        mask_t1 = np.squeeze(mask_t1)
    
    mask_t2 = one_hot_t2[sample_idx, ..., 1]
    mask_t2 = np.asarray(mask_t2, dtype=np.float32)
    if mask_t2.ndim > 2:
        mask_t2 = np.squeeze(mask_t2)

    # —— 计算两组 readout 的概率图 —— #
    no_guid_probs = []
    guid_probs = []
    lats, lons = None, None
    
    for step in steps:
        prob_ng, step_lats, step_lons = make_prob_2d(no_guidance_readout_frames_dict[step][sample_idx])
        prob_g, _, _ = make_prob_2d(guided_readout_frames_dict[step][sample_idx])
        if lats is None:
            lats, lons = step_lats, step_lons
        no_guid_probs.append(prob_ng)
        guid_probs.append(prob_g)

    # —— 添加循环点避免拼缝 —— #
    mask_t1_plot, lons_plot = add_cyclic_point(mask_t1, coord=lons)
    mask_t2_plot, _ = add_cyclic_point(mask_t2, coord=lons)
    no_guid_plots = [add_cyclic_point(p, coord=lons)[0] for p in no_guid_probs]
    guid_plots = [add_cyclic_point(p, coord=lons)[0] for p in guid_probs]

    # —— 布局：2 行 x (1 + len(steps)) 列 —— #
    n_cols = 1 + len(steps)  # mask + steps
    n_rows = 2  # No Guidance + With Guidance
    
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        subplot_kw={'projection': projection}
    )
    
    lon_min, lon_max = (0, 360) if (lons.min() >= 0 and lons.max() > 180) else (-180, 180)
    
    def setup_ax(ax):
        ax.set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False

    # —— 创建网格坐标（所有图共用）—— #
    lon_mesh, lat_mesh = np.meshgrid(lons_plot, lats)
    
    # —— 绘制 mask（每行显示不同的 mask） —— #
    masks_and_titles = [
        (mask_t1_plot, "GT @ t1\n(No Guidance)"),
        (mask_t2_plot, "Target @ t2\n(With Guidance)"),
    ]
    for row_idx, (mask_plot, title) in enumerate(masks_and_titles):
        ax = axs[row_idx, 0]
        setup_ax(ax)
        mask_masked = np.ma.masked_where(mask_plot < 0.01, mask_plot)
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, mask_masked,
            cmap=cmap_mask, alpha=0.85,
            vmin=0.01, vmax=1.0,
            transform=ccrs.PlateCarree(),
            shading='nearest'  # 使用 nearest 避免颜色断层
        )
        ax.contour(
            lon_mesh, lat_mesh, mask_plot,
            levels=[0.1, 0.5, 0.9],
            colors="red", linewidths=1.2,
            transform=ccrs.PlateCarree()
        )
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)

    # —— 绘制 readout 对比 —— #
    for col_idx, step in enumerate(steps):
        # Row 0: No Guidance
        ax_ng = axs[0, col_idx + 1]
        setup_ax(ax_ng)
        im_ng = ax_ng.pcolormesh(
            lon_mesh, lat_mesh, no_guid_plots[col_idx],
            cmap=cmap_prob, alpha=0.85,
            vmin=0.0, vmax=1.0,  # 概率值范围
            transform=ccrs.PlateCarree(),
            shading='nearest'  # 使用 nearest 避免颜色断层
        )
        ax_ng.set_title(f"Step {step}\n(No Guidance)", fontsize=10, fontweight="bold")
        plt.colorbar(im_ng, ax=ax_ng, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)
        
        # Row 1: With Guidance
        ax_g = axs[1, col_idx + 1]
        setup_ax(ax_g)
        im_g = ax_g.pcolormesh(
            lon_mesh, lat_mesh, guid_plots[col_idx],
            cmap=cmap_prob, alpha=0.85,
            vmin=0.0, vmax=1.0,  # 概率值范围
            transform=ccrs.PlateCarree(),
            shading='nearest'  # 使用 nearest 避免颜色断层
        )
        ax_g.set_title(f"Step {step}\n(With Guidance)", fontsize=10, fontweight="bold")
        plt.colorbar(im_g, ax=ax_g, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)

    # —— 标题 & 保存 —— #
    supt = f"Readout Comparison: No Guidance vs With Guidance - Epoch {epoch}"
    if ts is not None:
        supt += f"\nTime: {str(ts).split('.')[0]}"
    plt.suptitle(supt, fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"readout_comparison_epoch{epoch}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Comparison] Saved readout comparison to {out_path}")


def create_variable_comparison_plot(
    var_name: str,
    no_guidance_predictions: List[xr.Dataset],
    guided_predictions: List[xr.Dataset],
    gt_frames: List[xr.Dataset],
    output_dir: str,
    epoch: int = 0,
    level: Optional[int] = None,
    frame_idx: int = 0,
    sample_idx: int = 0,
):
    """
    创建单个天气变量的无 guidance vs 有 guidance 对比图。
    
    布局：
    - 第一行：GT | No Guidance Prediction | With Guidance Prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def extract_2d(ds, var: str, level_val: Optional[int] = None):
        # 如果是 Dataset
        if isinstance(ds, xr.Dataset):
            da = ds[var]
        else:
            da = ds
        
        if "batch" in da.dims:
            da = da.isel(batch=sample_idx)
        if "time" in da.dims:
            da = da.isel(time=0)
        if "level" in da.dims and level_val is not None:
            da = da.sel(level=level_val, method="nearest")
        elif "level" in da.dims:
            da = da.mean("level")
        
        arr = np.asarray(da.values)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2:])
        lats = da["lat"].values
        lons = da["lon"].values
        return arr, lats, lons

    # 提取数据
    gt_arr, lats, lons = extract_2d(gt_frames[frame_idx], var_name, level)
    ng_arr, _, _ = extract_2d(no_guidance_predictions[frame_idx], var_name, level)
    g_arr, _, _ = extract_2d(guided_predictions[frame_idx], var_name, level)

    # 添加循环点
    gt_plot, lons_plot = add_cyclic_point(gt_arr, coord=lons)
    ng_plot, _ = add_cyclic_point(ng_arr, coord=lons)
    g_plot, _ = add_cyclic_point(g_arr, coord=lons)

    # 统一 colorbar 范围
    vmin = min(gt_plot.min(), ng_plot.min(), g_plot.min())
    vmax = max(gt_plot.max(), ng_plot.max(), g_plot.max())

    # —— 布局：1 行 x 3 列（GT | No Guidance | With Guidance）—— #
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(
        1, 3,
        figsize=(18, 5),
        subplot_kw={'projection': projection}
    )
    
    lon_min, lon_max = (0, 360) if (lons.min() >= 0 and lons.max() > 180) else (-180, 180)
    
    def setup_ax(ax):
        ax.set_extent([lon_min, lon_max, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False

    # 创建网格坐标
    lon_mesh, lat_mesh = np.meshgrid(lons_plot, lats)
    
    titles = ["Ground Truth", "No Guidance", "With Guidance"]
    data_plots = [gt_plot, ng_plot, g_plot]
    
    for ax, title, data in zip(axs, titles, data_plots):
        setup_ax(ax)
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, data,
            cmap="RdBu_r", alpha=0.85,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading='nearest'  # 使用 nearest 避免颜色断层
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8)

    level_str = f" @ level {level}" if level else ""
    plt.suptitle(f"{var_name}{level_str} - Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # 保存到子目录
    var_dir = os.path.join(output_dir, var_name)
    os.makedirs(var_dir, exist_ok=True)
    out_path = os.path.join(var_dir, f"{var_name}_comparison_epoch{epoch}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Comparison] Saved {var_name} comparison to {out_path}")


def create_all_variable_comparisons(
    no_guidance_predictions: List[xr.Dataset],
    guided_predictions: List[xr.Dataset],
    gt_frames: List[xr.Dataset],
    output_dir: str,
    epoch: int = 0,
    level_dict: Optional[Dict[str, int]] = None,
    frame_idx: int = 0,
    variables: Optional[List[str]] = None,
):
    """
    为指定的天气变量创建对比图。
    
    Args:
        variables: 要绘制的变量列表。如果为 None，则绘制全部变量。
    """
    if level_dict is None:
        level_dict = {}
    
    if variables is None:
        variables = list(gt_frames[0].data_vars.keys())
    
    for var in variables:
        level = level_dict.get(var, None)
        try:
            create_variable_comparison_plot(
                var_name=var,
                no_guidance_predictions=no_guidance_predictions,
                guided_predictions=guided_predictions,
                gt_frames=gt_frames,
                output_dir=output_dir,
                epoch=epoch,
                level=level,
                frame_idx=frame_idx,
            )
        except Exception as e:
            print(f"[Comparison] Warning: Failed to create comparison for {var}: {e}")

