import io
import os
from multiprocessing import Pool, cpu_count

import cartopy.crs as ccrs

# Ensure matplotlib uses a non-interactive backend
# plt.switch_backend('agg')
# Define the function to visualize one-hot readout with geographical context
# This function assumes readout_frames_dict contains the readout frames and one-hot encoded masks
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from PIL import Image
from tqdm.auto import tqdm


def visualize_onehot_readout_simple(
    readout_frames_dict: dict,
    one_hot: np.ndarray,
    output_dir: str,
    epoch: int,
    mark: None = None,
    ts: None = None,
    steps: list = [10, 15],
    sample_idx: int = 0,
    cmap_mask: str = "Reds",
    cmap_prob: str = "RdBu_r",
    max_cols: int = 3,
):
    """
    可视化：第一张 = one-hot 正类 mask；后续依次为各个 denoising step 的正类概率读出。
    支持任意数量的 steps（4-20 个），每行最多 max_cols 张图。
    严格以数据自带坐标系为准绘制（兼容经度 0..360 或 -180..180）。
    
    Args:
        readout_frames_dict: {step: [xr.Dataset, ...], ...} 各 step 的 readout 结果
        one_hot: np.ndarray, shape (batch, H, W, 2) 的 one-hot 掩码
        output_dir: 输出目录
        epoch: 当前 epoch
        mark: 可选的外部 mark（暂时保留，但动态模式下不使用）
        ts: 时间戳（用于标题）
        steps: denoising step 列表，例如 [5, 10, 15, 18] 或 [0, 2, 4, ..., 18]
        sample_idx: 使用哪个 batch 样本
        cmap_mask: mask 的 colormap
        cmap_prob: 概率图的 colormap
        max_cols: 每行最多的图数量，默认 3
    """

    os.makedirs(output_dir, exist_ok=True)

    # —— 内部工具：把 xarray 变量挤成 2D(lat,lon) 并取坐标 —— #
    def as_2d_latlon(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 处理常见维度
        if "level" in da.dims:
            da = da.mean("level")
        # 优先挤掉 batch（或 sample）与 time 维
        for d in ("batch", "sample", "time"):
            if d in da.dims and da.sizes[d] == 1:
                da = da.isel({d: 0})
        # 若仍有 time 维，取第 0 帧（有些数据不把长度=1的 time 自动挤掉）
        if "time" in da.dims:
            da = da.isel(time=0)

        # 断言必须有 lat/lon
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(f"DataArray 缺少 lat/lon 维：dims = {da.dims}")

        lats = da["lat"].values
        lons = da["lon"].values
        arr = da.values  # 期望是 2D

        # 安全兜底：如果还有多余的前置维，尝试 squeeze
        arr = np.asarray(arr)
        if arr.ndim > 2:
            # 将除了最后两个维度外的都 squeeze 掉
            new_shape = arr.shape[-2:]
            arr = arr.reshape(new_shape)
        if arr.ndim != 2:
            raise ValueError(f"期望 2D (lat,lon)，但得到形状 {arr.shape}")

        return arr, lats, lons

    # —— softmax(u,v) → 正类概率，返回 2D —— #
    def make_prob_2d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = ds["u_component_of_wind"]
        v = ds["v_component_of_wind"]
        u2d, lats, lons = as_2d_latlon(u)
        v2d, _, _       = as_2d_latlon(v)
        # logits: [..., 2]
        logits = np.stack([u2d, v2d], axis=-1)
        exps   = np.exp(logits - logits.max(-1, keepdims=True))
        probs  = exps / exps.sum(-1, keepdims=True)
        pos    = probs[..., 1]  # 正类
        return pos, lats, lons

    # —— 取 mask（正类通道）并保证 2D —— #
    mask = one_hot[sample_idx, ..., 1]
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"期望 mask 为 2D (lat,lon)，但得到形状 {mask.shape}")

    # —— 计算所有 step 的概率图 —— #
    probs_list = []
    lats = None
    lons = None
    for step in steps:
        prob, step_lats, step_lons = make_prob_2d(readout_frames_dict[step][sample_idx])
        if lats is None:
            lats = step_lats
            lons = step_lons
        probs_list.append(prob)
    
    print(f"[Visualize] mask shape: {mask.shape}, num steps: {len(steps)}")

    # —— 坐标一致性检查 —— #
    if mask.shape != probs_list[0].shape:
        raise ValueError(f"mask 形状 {mask.shape} 与 prob 形状 {probs_list[0].shape} 不一致。")

    # —— 避免 0/360 拼缝：给 lon 加循环点 —— #
    mask_plot, lons_plot = add_cyclic_point(mask, coord=lons)
    probs_plot = [add_cyclic_point(p, coord=lons)[0] for p in probs_list]

    # —— 计算动态布局：1(mask) + len(steps)(readouts) —— #
    n_total = 1 + len(steps)  # mask + 各 step 的 readout
    n_cols = min(max_cols, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols  # 向上取整

    # —— 设定显示投影（太平洋居中） —— #
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(
        n_rows, n_cols, 
        figsize=(6 * n_cols, 5 * n_rows), 
        subplot_kw={'projection': projection}
    )
    
    # 确保 axs 是二维数组，方便索引
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)

    # —— 设定地图范围 —— #
    lon_min, lon_max = (0, 360) if (lons.min() >= 0 and lons.max() > 180) else (-180, 180)
    
    def setup_ax(ax):
        """设置单个子图的地图背景"""
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
    
    # —— 绘制第一张图：mask —— #
    ax0 = axs[0, 0]
    setup_ax(ax0)
    mask_masked = np.ma.masked_where(mask_plot < 0.01, mask_plot)
    im0 = ax0.pcolormesh(
        lon_mesh, lat_mesh, mask_masked,
        cmap=cmap_mask, alpha=0.85,
        vmin=0.01, vmax=1.0,
        transform=ccrs.PlateCarree(),
        shading='nearest'  # 使用 nearest 避免颜色断层
    )
    # 保留等高线用于显示关键阈值
    ax0.contour(
        lon_mesh, lat_mesh, mask_plot,
        levels=[0.1, 0.5, 0.9],
        colors="red", linewidths=1.2,
        transform=ccrs.PlateCarree()
    )
    ax0.set_title("One-hot Mask (GT)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=ax0, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8, label="Mask")

    # —— 绘制后续图：各 step 的 readout —— #
    
    for i, (step, prob_plot) in enumerate(zip(steps, probs_plot)):
        # 计算当前图在 grid 中的位置（从第 2 张开始，即索引 1）
        idx = i + 1
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axs[row, col]
        setup_ax(ax)
        
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, prob_plot,
            cmap=cmap_prob, alpha=0.85,
            vmin=0.0, vmax=1.0,  # 概率值范围
            transform=ccrs.PlateCarree(),
            shading='nearest'  # 使用 nearest 避免颜色断层
        )
        ax.set_title(f"Readout @ Step {step}", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08, shrink=0.8, label="Prob")

    # —— 隐藏多余的子图 —— #
    for idx in range(n_total, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axs[row, col].set_visible(False)

    # —— 标题 & 保存 —— #
    supt = f"Readout Analysis - Epoch {epoch} ({len(steps)} steps: {steps})"
    if ts is not None:
        supt += f"\nTime: {str(ts).split('.')[0]}"
    plt.suptitle(supt, fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"onehot_readout_epoch{epoch}_geo.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] saved geographical plot ({n_rows}x{n_cols} grid) to {out_path}")


# def visualize_onehot_readout_simple(
#     readout_frames_dict: dict,
#     one_hot: np.ndarray,
#     output_dir: str,
#     epoch: int,
#     mark: None = None,
#     ts: None = None,
#     steps: list = [10, 15],
#     sample_idx: int = 0,
#     cmap_mask: str = "Reds",  # 改用Reds以便更好地与地图叠加
#     cmap_prob: str = "RdBu_r",
# ):
    
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. 提取 mask (H, W)
#     mask = one_hot[sample_idx, ..., 1]
#     mask = np.array(mask, dtype=np.float32)

#     # 2. softmax → 正类概率
#     def make_prob(ds):
#         u = ds["u_component_of_wind"]
#         v = ds["v_component_of_wind"]
#         if "level" in u.dims:
#             u = u.mean("level"); v = v.mean("level")
#         for d in ("batch","time"):
#             if d in u.dims and u.sizes[d]==1:
#                 u = u.isel({d:0}); v = v.isel({d:0})
#         if "time" in u.dims:
#             u = u.isel(time=0); v = v.isel(time=0)
#         logits = np.stack([u.values, v.values], axis=-1)
#         exps   = np.exp(logits - logits.max(-1, keepdims=True))
#         probs  = exps / exps.sum(-1, keepdims=True)
#         return probs[...,1]

#     prob1 = make_prob(readout_frames_dict[steps[0]][sample_idx])
#     prob2 = make_prob(readout_frames_dict[steps[1]][sample_idx])
#     if mark is not None:
#         prob2 = mark

#     # left right flip
#     # prob1 = np.flip(prob1, axis=1)
#     # prob2 = np.flip(prob2, axis=1)
#     # mask = np.flip(mask, axis=1)

#     # 获取经纬度坐标
#     lats = np.linspace(90, -90, 181)
#     lons = np.linspace(0, 359, 360)

#     # 3. 绘图
#     projection = ccrs.PlateCarree(central_longitude=180)  # 设置中央经线
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), 
#                            subplot_kw={'projection': projection})  # 使用新投影
    
#     # 对每个子图设置显示范围
#     for ax in axs:
#         ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
    
#     # A. mask with map background
#     # 先添加地理特征作为背景
#     axs[0].add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
#     axs[0].add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
#     axs[0].add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
#     axs[0].add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
    
#     # 然后叠加mask，只显示非零值
#     mask_masked = np.ma.masked_where(mask < 0.01, mask)  # 隐藏接近0的值
#     im0 = axs[0].contourf(lons, lats, mask_masked, levels=np.linspace(0.01, 1, 10), 
#                          cmap=cmap_mask, alpha=0.8, transform=ccrs.PlateCarree())
    
#     # 添加mask的轮廓线以更清晰显示边界
#     axs[0].contour(lons, lats, mask, levels=[0.1, 0.5, 0.9], 
#                   colors=['red'], linewidths=[1, 2, 3], 
#                   transform=ccrs.PlateCarree())
    
#     gl0 = axs[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#     gl0.top_labels = False
#     gl0.right_labels = False
#     axs[0].set_global()
#     axs[0].set_title("One-hot Mask with Geography", fontsize=12, fontweight='bold')
    
#     # B. readout @step1
#     axs[1].add_feature(cfeature.COASTLINE, linewidth=0.5)
#     axs[1].add_feature(cfeature.BORDERS, linewidth=0.3)
#     axs[1].add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
#     axs[1].add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
#     im1 = axs[1].contourf(lons, lats, prob1[0], levels=20, 
#                          cmap=cmap_prob, alpha=0.8, transform=ccrs.PlateCarree())
    
#     gl1 = axs[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#     gl1.top_labels = False
#     gl1.right_labels = False
#     axs[1].set_global()
#     axs[1].set_title(f"Readout @{steps[0]}", fontsize=12, fontweight='bold')
    
#     # C. readout @step2
#     axs[2].add_feature(cfeature.COASTLINE, linewidth=0.5)
#     axs[2].add_feature(cfeature.BORDERS, linewidth=0.3)
#     axs[2].add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
#     axs[2].add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
#     im2 = axs[2].contourf(lons, lats, prob2[0], levels=20, 
#                          cmap=cmap_prob, alpha=0.8, transform=ccrs.PlateCarree())
    
#     gl2 = axs[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#     gl2.top_labels = False
#     gl2.right_labels = False
#     axs[2].set_global()
#     axs[2].set_title(f"Readout @{steps[1]}", fontsize=12, fontweight='bold')

#     # 4. 添加 colorbar
#     cbar0 = plt.colorbar(im0, ax=axs[0], orientation='horizontal', 
#                         fraction=0.046, pad=0.08, shrink=0.8)
#     cbar0.set_label("Mask Intensity", fontsize=10)
    
#     cbar1 = plt.colorbar(im1, ax=axs[1], orientation='horizontal', 
#                         fraction=0.046, pad=0.08, shrink=0.8)
#     cbar1.set_label("Probability", fontsize=10)
    
#     cbar2 = plt.colorbar(im2, ax=axs[2], orientation='horizontal', 
#                         fraction=0.046, pad=0.08, shrink=0.8)
#     cbar2.set_label("Probability", fontsize=10)

#     plt.suptitle(f'Readout Analysis - Epoch {epoch} - Time: {str(ts).split(".")[0]}', 
#             fontsize=14, fontweight='bold')
#     plt.tight_layout()
    
#     out_path = os.path.join(output_dir, f"onehot_readout_epoch{epoch}_geo.png")
#     fig.savefig(out_path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     print(f"[Visualize] saved geographical plot to {out_path}")





# def visualize_onehot_readout_simple(
#     readout_frames_dict: dict,
#     one_hot: np.ndarray,
#     output_dir: str,
#     epoch: int,
#     steps: list = [10, 15],
#     sample_idx: int = 0,
#     cmap_mask: str = "gray",
#     cmap_prob: str = "RdBu_r",
# ):
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. 提取 mask (H, W)
#     mask = one_hot[sample_idx, ..., 1]

#     # 2. softmax → 正类概率（和前面一样的处理）
#     def make_prob(ds):
#         u = ds["u_component_of_wind"]
#         v = ds["v_component_of_wind"]
#         if "level" in u.dims:
#             u = u.mean("level"); v = v.mean("level")
#         for d in ("batch","time"):
#             if d in u.dims and u.sizes[d]==1:
#                 u = u.isel({d:0}); v = v.isel({d:0})
#         if "time" in u.dims:
#             u = u.isel(time=0); v = v.isel(time=0)
#         logits = np.stack([u.values, v.values], axis=-1)
#         exps   = np.exp(logits - logits.max(-1, keepdims=True))
#         probs  = exps / exps.sum(-1, keepdims=True)
#         return probs[...,1]  # (H,W) 正类概率

#     prob1 = make_prob(readout_frames_dict[steps[0]][sample_idx])
#     prob2 = make_prob(readout_frames_dict[steps[1]][sample_idx])

#     # 3. 绘图
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     # A. mask
#     im0 = axs[0].imshow(mask, cmap=cmap_mask, vmin=0, vmax=1, origin='lower')
#     axs[0].set_title("One-hot Mask")
#     axs[0].axis('off')
#     # B. readout @step1
#     im1 = axs[1].imshow(prob1[0], cmap=cmap_prob, origin='lower')
#     axs[1].set_title(f"Readout @{steps[0]}")
#     axs[1].axis('off')
#     # C. readout @step2
#     im2 = axs[2].imshow(prob2[0], cmap=cmap_prob, origin='lower')
#     axs[2].set_title(f"Readout @{steps[1]}")
#     axs[2].axis('off')

#     # 4. 统一 colorbar（用读出图的第一个）
#     # cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.05, pad=0.07)
#     # cbar.set_label("Probability")

#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"onehot_readout_epoch{epoch}.png")
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"[Visualize] saved to {out_path}")





def visualize_onehot_readout(
    readout_frames_dict: dict,
    one_hot: np.ndarray,
    output_dir: str,
    epoch: int,
    steps: list = [10, 15],
    sample_idx: int = 0,
    level: int = None,
    cmap_mask: str = "gray",
    cmap_prob: str = "RdBu_r",
):
    """
    可视化 one_hot 掩码 与 两个 ReadOut 步骤下的概率场。

    Args:
      readout_frames_dict: {step: [xr.Dataset, xr.Dataset, ...], ...}
      one_hot: np.ndarray, shape (batch, H, W, 2)
      output_dir: str, 保存图像的目录
      epoch: int, 当前 epoch，用于文件名后缀
      steps: list of int, ReadOut 步骤（默认 [10,15]）
      sample_idx: int, 使用 one_hot 中的哪个样本（默认 0）
      level: Optional[int], 若变量有多层，指定 level
      cmap_mask: str, 掩码的 colormap
      cmap_prob: str, 概率场的 colormap
    """
    os.makedirs(output_dir, exist_ok=True)

    # —— 1. 提取二值掩码 —— 
    mask = one_hot[sample_idx, ..., 1]  # (H, W)

    def make_prob_field(ds: xr.Dataset) -> xr.DataArray:
        u = ds["u_component_of_wind"]
        v = ds["v_component_of_wind"]

        # 1. collapse level
        if "level" in u.dims:
            u = u.mean("level")
            v = v.mean("level")

        # 2. drop singleton batch/time
        for d in ("batch","time"):
            if d in u.dims and u.sizes[d] == 1:
                u = u.isel({d: 0})
                v = v.isel({d: 0})

        # 3. (可选) 如果还有 time 维度，选第一帧
        if "time" in u.dims:
            u = u.isel(time=0)
            v = v.isel(time=0)

        # 4. softmax → 正类概率
        logits = np.stack([u.values, v.values], axis=-1)
        exps   = np.exp(logits - logits.max(-1, keepdims=True))
        probs  = exps / exps.sum(-1, keepdims=True)
        data   = probs[..., 1]

        # 5. 动态维度和坐标
        out_dims  = tuple(d for d in u.dims)            # e.g. ('lat','lon')
        out_coords = {d: u.coords[d] for d in out_dims}

        return xr.DataArray(data, dims=out_dims, coords=out_coords)

    # 生成两个步骤的概率场
    prob1 = make_prob_field(readout_frames_dict[steps[0]][sample_idx])
    prob2 = make_prob_field(readout_frames_dict[steps[1]][sample_idx])

    # —— 3. 并排绘图 —— 
    fig, axs = plt.subplots(
        1, 3,
        figsize=(15, 5),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # A. one_hot 掩码
    axs[0].imshow(mask, cmap=cmap_mask, origin='lower')
    axs[0].set_title("One-hot Mask")
    axs[0].axis('off')


    # B. ReadOut @ step1
    m1 = prob1.plot(
        ax=axs[1],
        transform=ccrs.PlateCarree(),
        cmap=cmap_prob,
        add_colorbar=False
    )
    axs[1].coastlines(linewidth=0.5)
    axs[1].set_title(f"Readout @{steps[0]}")

    # C. ReadOut @ step2
    m2 = prob2.plot(
        ax=axs[2],
        transform=ccrs.PlateCarree(),
        cmap=cmap_prob,
        add_colorbar=False
    )
    axs[2].coastlines(linewidth=0.5)
    axs[2].set_title(f"Readout @{steps[1]}")

    # 然后用 m1 来做 colorbar
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    fig.colorbar(m1, cax=cax, orientation='horizontal', label='Probability')

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = os.path.join(output_dir, f"onehot_readout_epoch{epoch}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Visualize] saved to {out_path}")



def _make_gif_for_var_ori(args):
    """
    Worker function: generates one GIF for a single variable.
    args is a tuple: (var, gt_frames, forecast_frames, readout_frames_dict,
                     output_dir, level_dict, duration, epoch)
    """
    (var, gt_frames, forecast_frames_dict, readout_frames_dict,
     output_dir, level_dict, duration, epoch) = args

    steps = sorted(readout_frames_dict.keys())  # e.g. [10, 15]
    N = len(gt_frames)
    proj = ccrs.PlateCarree()
    level = level_dict.get(var) if level_dict else None

    # --- compute global color scale (for readout rows) ---
    all_vals = []
    for ds in gt_frames + forecast_frames_dict + \
              readout_frames_dict[steps[0]] + readout_frames_dict[steps[1]]:
        da = ds[var]
        if 'level' in da.dims:
            if level is None:
                raise ValueError(f"'{var}' has levels; provide level_dict['{var}']")
            da = da.sel(level=level)
        all_vals.append(da.values.ravel())
    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[~np.isnan(all_vals)]
    glob_vmin, glob_vmax = np.percentile(all_vals, [1, 99])
    glob_diff = max(abs(glob_vmin - np.median(all_vals)),
                    abs(glob_vmax - np.median(all_vals)))

    # --- compute first-row color scale (for GT & Forecast only) ---
    first_vals = []
    for ds in gt_frames + forecast_frames_dict:
        da = ds[var]
        if 'level' in da.dims:
            da = da.sel(level=level)
        first_vals.append(da.values.ravel())
    first_vals = np.concatenate(first_vals)
    first_vals = first_vals[~np.isnan(first_vals)]
    f_vmin, f_vmax = np.percentile(first_vals, [1, 99])
    f_diff = max(abs(f_vmin - np.median(first_vals)),
                 abs(f_vmax - np.median(first_vals)))

    pil_frames = []
    for idx in range(N):
        fig = plt.figure(figsize=(12, 12))
        # build 3×3 grid of axes
        axs = []
        for i in range(9):
            if i in (3, 6):
                ax = fig.add_subplot(3, 3, i+1)
            else:
                ax = fig.add_subplot(3, 3, i+1, projection=proj)
            axs.append(ax)

        # extract the data arrays for this frame
        gt   = gt_frames[idx][var]
        fc   = forecast_frames_dict[idx][var]
        ro10 = readout_frames_dict[steps[0]][idx][var]
        ro15 = readout_frames_dict[steps[1]][idx][var]
        if level is not None:
            gt   = gt.sel(level=level)
            fc   = fc.sel(level=level)
            ro10 = ro10.sel(level=level)
            ro15 = ro15.sel(level=level)

        # --- first row: GT, Forecast, Forecast−GT with its own scale ---
        im1 = gt.plot(
            ax=axs[0], transform=proj,
            cmap='RdBu_r', vmin=f_vmin, vmax=f_vmax,
            add_colorbar=False
        )
        axs[0].coastlines(linewidth=0.5)
        axs[0].set_title('GT', fontsize=10)

        im2 = fc.plot(
            ax=axs[1], transform=proj,
            cmap='RdBu_r', vmin=f_vmin, vmax=f_vmax,
            add_colorbar=False
        )
        axs[1].coastlines(linewidth=0.5)
        axs[1].set_title('Forecast', fontsize=10)

        im3 = (fc - gt).plot(
            ax=axs[2], transform=proj,
            cmap='RdBu_r', vmin=-f_diff, vmax=f_diff,
            add_colorbar=False
        )
        axs[2].coastlines(linewidth=0.5)
        axs[2].set_title('Forecast−GT', fontsize=10)

        # blank spacer
        axs[3].axis('off')

        # --- second row: Readout @step0 & its error with global scale ---
        im5 = ro10.plot(
            ax=axs[4], transform=proj,
            cmap='RdBu_r', vmin=glob_vmin, vmax=glob_vmax,
            add_colorbar=False
        )
        axs[4].coastlines(linewidth=0.5)
        axs[4].set_title(f'Readout @{steps[0]}', fontsize=10)

        im6 = (ro10 - gt).plot(
            ax=axs[5], transform=proj,
            cmap='RdBu_r', vmin=-glob_diff, vmax=glob_diff,
            add_colorbar=False
        )
        axs[5].coastlines(linewidth=0.5)
        axs[5].set_title(f'Readout{steps[0]}−GT', fontsize=10)

        # blank spacer
        axs[6].axis('off')

        # --- third row: Readout @step1 & its error with global scale ---
        im8 = ro15.plot(
            ax=axs[7], transform=proj,
            cmap='RdBu_r', vmin=glob_vmin, vmax=glob_vmax,
            add_colorbar=False
        )
        axs[7].coastlines(linewidth=0.5)
        axs[7].set_title(f'Readout @{steps[1]}', fontsize=10)

        im9 = (ro15 - gt).plot(
            ax=axs[8], transform=proj,
            cmap='RdBu_r', vmin=-glob_diff, vmax=glob_diff,
            add_colorbar=False
        )
        axs[8].coastlines(linewidth=0.5)
        axs[8].set_title(f'Readout{steps[1]}−GT', fontsize=10)

        # --- colorbars ---
        # first-row colorbar (above the plot)
        cax_f = fig.add_axes([0.1, 0.93, 0.25, 0.02])
        fig.colorbar(im1, cax=cax_f, orientation='horizontal', label=var)

        # global data colorbar (bottom left)
        cax1 = fig.add_axes([0.1, 0.05, 0.6, 0.02])
        fig.colorbar(im5, cax=cax1, orientation='horizontal', label=var)

        # global error colorbar (bottom right)
        cax2 = fig.add_axes([0.75, 0.05, 0.2, 0.02])
        fig.colorbar(im6, cax=cax2, orientation='horizontal', label='Error')

        # title and layout
        plt.suptitle(
            f'{var}' + (f', level={level}' if level is not None else '') +
            f' — frame {idx}',
            fontsize=12
        )
        fig.subplots_adjust(
            top=0.90, bottom=0.10,
            left=0.02, right=0.98,
            hspace=0.3, wspace=0.2
        )

        # save to buffer and append to frames
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # write out GIF
    gif_path = os.path.join(output_dir, f'{var}_{epoch}.gif')
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0
    )
    return var


def _make_gif_for_var(args):
    """
    Worker: 为单个变量逐帧生成静态图（PNG）
    布局：单行三图 [GT | Forecast | Forecast−GT]
    改动点：添加经纬度网格/标签；更高分辨率；50m 边界；自动经度范围；PNG 输出。
    """
    import os

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from cartopy.util import add_cyclic_point

    (var, gt_frames, forecast_frames_dict, readout_frames_dict,
     output_dir, level_dict, duration, epoch) = args

    # —— 基本设置 —— #
    N = len(gt_frames)
    proj = ccrs.PlateCarree(central_longitude=180)  # 太平洋居中
    data_crs = ccrs.PlateCarree()
    level = (level_dict or {}).get(var, None)

    # —— 工具：严格选层 & 去多余维 —— #
    def select_level_and_squeeze(da: xr.DataArray) -> xr.DataArray:
        x = da
        if "level" in x.dims:
            if level is None:
                raise ValueError(f"[{var}] 具有 'level' 维，请在 level_dict['{var}'] 指定具体层。")
            x = x.sel(level=level)
        # 去掉长度=1的 batch/sample/time；若仍有 time，则取第 0 帧
        for d in ("batch", "sample", "time"):
            if d in x.dims and x.sizes[d] == 1:
                x = x.isel({d: 0})
        if "time" in x.dims:
            x = x.isel(time=0)
        if "lat" not in x.dims or "lon" not in x.dims:
            raise ValueError(f"[{var}] 缺少 lat/lon 维：dims={tuple(x.dims)}")
        # 保证 2D
        arr = np.asarray(x.values)
        if arr.ndim > 2:
            x = xr.DataArray(arr.reshape(arr.shape[-2:]), dims=("lat", "lon"),
                             coords={"lat": x["lat"].values, "lon": x["lon"].values})
        return x

    # —— 工具：补经度循环点（避免 0/360 拼缝） —— #
    def to_cyclic(da: xr.DataArray) -> xr.DataArray:
        lons = da["lon"].values
        vals_c, lons_c = add_cyclic_point(da.values, coord=lons)
        coords = {d: da.coords[d].values for d in da.dims}
        coords["lon"] = lons_c
        return xr.DataArray(vals_c, dims=da.dims, coords=coords, attrs=da.attrs)

    # —— 依据首帧 GT 的经度判定 extent —— #
    gt0_da = select_level_and_squeeze(gt_frames[0][var])
    lons0 = gt0_da["lon"].values
    if np.nanmin(lons0) >= 0 and np.nanmax(lons0) > 180:
        LON_EXTENT = [0, 360]
    else:
        LON_EXTENT = [-180, 180]
    LAT_EXTENT = [-90, 90]

    os.makedirs(output_dir, exist_ok=True)

    for idx in range(N):
        # 数据抽取：GT 与 Forecast（注意：不用 readout）
        gt_da = select_level_and_squeeze(gt_frames[idx][var])
        fc_da = select_level_and_squeeze(forecast_frames_dict[idx][var])

        # 转为无拼缝的 2D 场
        gt_cyc = to_cyclic(gt_da)
        fc_cyc = to_cyclic(fc_da)
        df_cyc = fc_cyc - gt_cyc

        lats = gt_cyc["lat"].values
        lons_c = gt_cyc["lon"].values
        gt_arr = gt_cyc.values
        fc_arr = fc_cyc.values
        df_arr = df_cyc.values

        # —— 行内归一化 —— #
        gtp = np.concatenate([gt_arr.ravel(), fc_arr.ravel()])
        gtp = gtp[~np.isnan(gtp)]
        if gtp.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(gtp, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                eps = 1e-6 if vmin == 0 else max(1e-6, abs(vmin) * 1e-6)
                vmin, vmax = vmin - eps, vmax + eps

        dmax = np.nanmax(np.abs(df_arr))
        if not np.isfinite(dmax) or dmax == 0:
            dmax = max(abs(vmin), abs(vmax)) or 1.0

        # —— 绘图：单行三图 —— #
        fig = plt.figure(figsize=(18, 6))  # 更大画布
        axs = [fig.add_subplot(1, 3, j + 1, projection=proj) for j in range(3)]

        # 统一公共绘制函数
        def draw_panel(ax, data, title, vmin, vmax):
            # 创建网格坐标用于 pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lons_c, lats)
            im = ax.pcolormesh(
                lon_mesh, lat_mesh, data, cmap="RdBu_r",
                vmin=vmin, vmax=vmax, transform=data_crs,
                shading='nearest'  # 使用 nearest 避免颜色断层
            )
            ax.set_title(title, fontsize=11)
            ax.set_extent(LON_EXTENT + LAT_EXTENT, crs=data_crs)
            # 更细的底图线条
            ax.coastlines(resolution="50m", linewidth=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
            # 经纬网格线与标签
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                              linewidth=0.5, color="gray", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            # 字体略小，避免与图像挤在一起
            try:
                gl.xlabel_style = {"size": 8}
                gl.ylabel_style = {"size": 8}
            except Exception:
                pass
            return im

        im0 = draw_panel(axs[0], gt_arr, "GT", vmin, vmax)
        im1 = draw_panel(axs[1], fc_arr, "Forecast", vmin, vmax)
        im2 = draw_panel(axs[2], df_arr, "Forecast − GT", -dmax, dmax)

        # 两个 colorbar：GT/Forecast、Diff
        cax1 = fig.add_axes([0.12, 0.10, 0.32, 0.028])
        fig.colorbar(im1, cax=cax1, orientation="horizontal", label=var)
        cax2 = fig.add_axes([0.56, 0.10, 0.32, 0.028])
        fig.colorbar(im2, cax=cax2, orientation="horizontal", label="Diff")

        # 标题与布局
        ttl = f"{var}" + (f", level={level}" if level is not None else "") + f" — frame {idx}"
        plt.suptitle(ttl, fontsize=12)
        fig.subplots_adjust(top=0.86, bottom=0.20, left=0.06, right=0.98, wspace=0.18)

        # 保存 PNG（高 DPI）
        out_path = os.path.join(output_dir, f"{var}_frame{idx}_epoch{epoch}.png")
        plt.savefig(out_path, dpi=240, bbox_inches="tight")  # 提高 dpi
        plt.close(fig)

    return var












def generate_comparison_gifs_parallel(
    gt_frames, forecast_frames, readout_frames_dict,
    variables, output_dir, epoch,level_dict=None, duration=500,
    n_procs=None
):
    os.makedirs(output_dir, exist_ok=True)
    if n_procs is None:
        n_procs = max(1, cpu_count() - 1)

    # build args list - 为每个变量创建子目录
    args_list = []
    for var in variables:
        var_dir = os.path.join(output_dir, var)
        os.makedirs(var_dir, exist_ok=True)
        args_list.append(
            (var, gt_frames, forecast_frames, readout_frames_dict, var_dir, level_dict or {}, duration, epoch)
        )

    # 当 n_procs=1 时，直接顺序执行，避免 multiprocessing 在 JAX 多线程环境下的死锁问题
    if n_procs == 1:
        for args in tqdm(args_list, desc="Generating GIFs"):
            _make_gif_for_var(args)
        print("All GIFs generated sequentially.")
    else:
        with Pool(processes=n_procs) as pool:
            for finished_var in tqdm(pool.imap_unordered(_make_gif_for_var, args_list),
                                      total=len(variables), desc="Generating GIFs"):
                pass  # each var yields its name when done
        print("All GIFs generated in parallel.")


# ===== NEW FUNCTIONS WITH STEP PREFIX SUPPORT (for rollout visualization) =====

def _make_gif_for_var_with_prefix(args):
    """
    [NEW] Worker: 为单个变量逐帧生成静态图（PNG）with step prefix support
    布局：单行三图 [GT | Forecast | Forecast−GT]
    支持为每帧添加步骤前缀（如 step00_, step01_, ...）
    """
    import os
    import time

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from cartopy.util import add_cyclic_point

    # 解包参数，支持可选的 timing_logger
    if len(args) == 15:
        (var, gt_frames, forecast_frames_dict, readout_frames_dict,
            output_dir, level_dict, duration, epoch, step_prefix,
            contourf_levels, coastlines_resolution, draw_gridlabels, add_borders, dpi, timing_logger) = args
    else:
        (var, gt_frames, forecast_frames_dict, readout_frames_dict,
         output_dir, level_dict, duration, epoch, step_prefix,
         contourf_levels, coastlines_resolution, draw_gridlabels, add_borders, dpi) = args
        timing_logger = None

    # —— 基本设置 —— #
    N = len(gt_frames)
    proj = ccrs.PlateCarree(central_longitude=180)  # 太平洋居中
    data_crs = ccrs.PlateCarree()
    level = (level_dict or {}).get(var, None)

    # —— 工具：严格选层 & 去多余维 —— #
    def select_level_and_squeeze(da: xr.DataArray) -> xr.DataArray:
        x = da
        if "level" in x.dims:
            if level is None:
                raise ValueError(f"[{var}] 具有 'level' 维，请在 level_dict['{var}'] 指定具体层。")
            x = x.sel(level=level)
        # 去掉长度=1的 batch/sample/time；若仍有 time，则取第 0 帧
        for d in ("batch", "sample", "time"):
            if d in x.dims and x.sizes[d] == 1:
                x = x.isel({d: 0})
        if "time" in x.dims:
            x = x.isel(time=0)
        if "lat" not in x.dims or "lon" not in x.dims:
            raise ValueError(f"[{var}] 缺少 lat/lon 维：dims={tuple(x.dims)}")
        # 保证 2D
        arr = np.asarray(x.values)
        if arr.ndim > 2:
            x = xr.DataArray(arr.reshape(arr.shape[-2:]), dims=("lat", "lon"),
                             coords={"lat": x["lat"].values, "lon": x["lon"].values})
        return x

    # —— 工具：补经度循环点（避免 0/360 拼缝） —— #
    def to_cyclic(da: xr.DataArray) -> xr.DataArray:
        lons = da["lon"].values
        vals_c, lons_c = add_cyclic_point(da.values, coord=lons)
        coords = {d: da.coords[d].values for d in da.dims}
        coords["lon"] = lons_c
        return xr.DataArray(vals_c, dims=da.dims, coords=coords, attrs=da.attrs)

    # —— 依据首帧 GT 的经度判定 extent —— #
    gt0_da = select_level_and_squeeze(gt_frames[0][var])
    lons0 = gt0_da["lon"].values
    if np.nanmin(lons0) >= 0 and np.nanmax(lons0) > 180:
        LON_EXTENT = [0, 360]
    else:
        LON_EXTENT = [-180, 180]
    LAT_EXTENT = [-90, 90]

    os.makedirs(output_dir, exist_ok=True)

    # 记录整个变量的生成时间
    var_start_time = time.time()
    frame_times = []  # 记录每帧的生成时间

    for idx in range(N):
        frame_start_time = time.time()
        
        # 数据抽取：GT 与 Forecast（注意：不用 readout）
        gt_da = select_level_and_squeeze(gt_frames[idx][var])
        fc_da = select_level_and_squeeze(forecast_frames_dict[idx][var])

        # 转为无拼缝的 2D 场
        gt_cyc = to_cyclic(gt_da)
        fc_cyc = to_cyclic(fc_da)
        df_cyc = fc_cyc - gt_cyc

        lats = gt_cyc["lat"].values
        lons_c = gt_cyc["lon"].values
        gt_arr = gt_cyc.values
        fc_arr = fc_cyc.values
        df_arr = df_cyc.values

        # —— 行内归一化 —— #
        gtp = np.concatenate([gt_arr.ravel(), fc_arr.ravel()])
        gtp = gtp[~np.isnan(gtp)]
        if gtp.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(gtp, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                eps = 1e-6 if vmin == 0 else max(1e-6, abs(vmin) * 1e-6)
                vmin, vmax = vmin - eps, vmax + eps

        dmax = np.nanmax(np.abs(df_arr))
        if not np.isfinite(dmax) or dmax == 0:
            dmax = max(abs(vmin), abs(vmax)) or 1.0

        # —— 绘图：单行三图 —— #
        fig = plt.figure(figsize=(18, 6))  # 更大画布
        axs = [fig.add_subplot(1, 3, j + 1, projection=proj) for j in range(3)]

        # 统一公共绘制函数
        def draw_panel(ax, data, title, vmin, vmax):
            # 创建网格坐标用于 pcolormesh
            lon_mesh, lat_mesh = np.meshgrid(lons_c, lats)
            im = ax.pcolormesh(
                lon_mesh, lat_mesh, data, cmap="RdBu_r",
                vmin=vmin, vmax=vmax, transform=data_crs,
                shading='nearest'  # 使用 nearest 避免颜色断层
            )
            ax.set_title(title, fontsize=11)
            ax.set_extent(LON_EXTENT + LAT_EXTENT, crs=data_crs)
            # 更细的底图线条
            ax.coastlines(resolution=coastlines_resolution, linewidth=0.6)
            if add_borders:
                ax.add_feature(cfeature.BORDERS.with_scale(coastlines_resolution), linewidth=0.4)
            # 经纬网格线与标签
            gl = ax.gridlines(draw_labels=draw_gridlabels, dms=True, x_inline=False, y_inline=False,
                              linewidth=0.5, color="gray", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            # 字体略小，避免与图像挤在一起
            if draw_gridlabels:
                try:
                    gl.xlabel_style = {"size": 8}
                    gl.ylabel_style = {"size": 8}
                except Exception:
                    pass
            return im

        im0 = draw_panel(axs[0], gt_arr, "GT", vmin, vmax)
        im1 = draw_panel(axs[1], fc_arr, "Forecast", vmin, vmax)
        im2 = draw_panel(axs[2], df_arr, "Forecast − GT", -dmax, dmax)

        # 两个 colorbar：GT/Forecast、Diff
        cax1 = fig.add_axes([0.12, 0.10, 0.32, 0.028])
        fig.colorbar(im1, cax=cax1, orientation="horizontal", label=var)
        cax2 = fig.add_axes([0.56, 0.10, 0.32, 0.028])
        fig.colorbar(im2, cax=cax2, orientation="horizontal", label="Diff")

        # 标题与布局
        ttl = f"{var}" + (f", level={level}" if level is not None else "") + f" — frame {idx}"
        plt.suptitle(ttl, fontsize=12)
        fig.subplots_adjust(top=0.86, bottom=0.20, left=0.06, right=0.98, wspace=0.18)

        # 保存 PNG with step prefix
        if step_prefix:
            # step_prefix 可以是字符串（所有步骤用同一个前缀）或列表（每个步骤不同前缀）
            if isinstance(step_prefix, list) and idx < len(step_prefix):
                prefix = step_prefix[idx]
            elif isinstance(step_prefix, str):
                prefix = step_prefix
            else:
                prefix = ""
            filename = f"{prefix}{var}_frame{idx}_epoch{epoch}.png"
        else:
            filename = f"{var}_frame{idx}_epoch{epoch}.png"
        
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # 记录每帧的生成时间
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        
        # 如果提供了 timing_logger，记录每帧的时间（仅在单进程模式下有效）
        if timing_logger is not None:
            try:
                timing_logger.start(f"PNG: {var} frame {idx}")
                timing_logger.end(f"PNG: {var} frame {idx}")
            except Exception:
                pass  # 忽略错误，避免影响主流程

    # 计算总时间
    var_total_time = time.time() - var_start_time
    
    # 返回变量名和时间信息
    return {
        "var": var,
        "total_time": var_total_time,
        "num_frames": N,
        "frame_times": frame_times,
        "avg_frame_time": var_total_time / N if N > 0 else 0.0
    }


def generate_comparison_gifs_parallel_with_prefix(
    gt_frames, forecast_frames, readout_frames_dict,
    variables, output_dir, epoch, level_dict=None, duration=500,
    n_procs=None, step_prefix=None,
    contourf_levels=36, coastlines_resolution="50m", 
    draw_gridlabels=True, add_borders=True, dpi=240,
    timing_logger=None  # 新增：时间记录器
):
    """
    [NEW] 支持步骤前缀的可视化函数（用于rollout）
    
    Args:
        step_prefix: 可选，可以是：
            - None: 不添加前缀（保持原有行为）
            - str: 所有步骤使用同一个前缀（如 "step00_"）
            - List[str]: 每个步骤使用不同的前缀（如 ["step00_", "step01_", ...]）
        contourf_levels: contourf 的等级数（默认：36）
        coastlines_resolution: 海岸线分辨率（默认："50m"）
        draw_gridlabels: 是否绘制经纬网格标签（默认：True）
        add_borders: 是否添加国界线（默认：True）
        dpi: 输出图像 DPI（默认：240）
        timing_logger: 可选的时间记录器（TimingLogger 对象）
    """
    import time
    os.makedirs(output_dir, exist_ok=True)
    if n_procs is None:
        n_procs = max(1, cpu_count() - 1)

    # build args list - 为每个变量创建子目录
    args_list = []
    for var in variables:
        var_dir = os.path.join(output_dir, var)
        os.makedirs(var_dir, exist_ok=True)
        # 在单进程模式下传递 timing_logger，多进程模式下不传递（因为无法序列化）
        if n_procs == 1 and timing_logger is not None:
            args_list.append(
                (var, gt_frames, forecast_frames, readout_frames_dict, var_dir, 
                    level_dict or {}, duration, epoch, step_prefix,
                    contourf_levels, coastlines_resolution, draw_gridlabels, add_borders, dpi, timing_logger)
            )
        else:
            args_list.append(
                (var, gt_frames, forecast_frames, readout_frames_dict, var_dir, 
                 level_dict or {}, duration, epoch, step_prefix,
                 contourf_levels, coastlines_resolution, draw_gridlabels, add_borders, dpi)
            )

    # 当 n_procs=1 时，直接顺序执行，避免 multiprocessing 在 JAX 多线程环境下的死锁问题
    if n_procs == 1:
        all_timing_info = []
        for args in tqdm(args_list, desc="Generating GIFs"):
            result = _make_gif_for_var_with_prefix(args)
            if isinstance(result, dict):
                all_timing_info.append(result)
        print("All GIFs generated sequentially.")
        
        # 记录时间信息到 timing_logger
        if timing_logger is not None and all_timing_info:
            try:
                timing_logger.start("PNG generation by variable")
                for info in all_timing_info:
                    var_name = info.get("var", "unknown")
                    total_time = info.get("total_time", 0.0)
                    num_frames = info.get("num_frames", 0)
                    frame_times = info.get("frame_times", [])
                    
                    # 记录每个变量的总时间（包含所有帧）
                    timing_logger.start(f"PNG: {var_name} (total)")
                    # 添加每帧的子记录
                    for idx, frame_time in enumerate(frame_times):
                        timing_logger.start(f"PNG: {var_name} frame {idx}")
                        timing_logger.end(f"PNG: {var_name} frame {idx}")
                    timing_logger.end(f"PNG: {var_name} (total)")
                
                timing_logger.end("PNG generation by variable")
            except Exception as e:
                print(f"[TimingLogger Warning] Failed to record PNG timing: {e}")
    else:
        # 多进程模式：收集时间信息
        all_timing_info = []
    with Pool(processes=n_procs) as pool:
            for result in tqdm(pool.imap_unordered(_make_gif_for_var_with_prefix, args_list),
                                  total=len(variables), desc="Generating GIFs"):
                if isinstance(result, dict):
                    all_timing_info.append(result)
    print("All GIFs generated in parallel.")
    
    # 在多进程模式下，将时间信息记录到 timing_logger（使用 record_duration 方法）
    if timing_logger is not None and all_timing_info:
        try:
            timing_logger.start("PNG generation by variable")
            for info in all_timing_info:
                var_name = info.get("var", "unknown")
                total_time = info.get("total_time", 0.0)
                frame_times = info.get("frame_times", [])
                
                # 为每帧创建子记录
                frame_subsections = []
                for idx, frame_time in enumerate(frame_times):
                    frame_subsections.append({
                        "name": f"PNG: {var_name} frame {idx}",
                        "duration": frame_time,
                        "pure_time": frame_time,
                        "subsections": [],
                        "start_time": 0,  # 这些值不会被使用，只是占位
                        "end_time": 0
                    })
                
                # 使用 record_duration 手动记录总时间
                timing_logger.record_duration(
                    f"PNG: {var_name} (total)",
                    total_time,
                    subsections=frame_subsections
                )
            
            timing_logger.end("PNG generation by variable")
        except Exception as e:
            print(f"[TimingLogger Warning] Failed to record PNG timing: {e}")

