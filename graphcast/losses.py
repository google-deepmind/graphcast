# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions (and terms for use in loss functions) used for weather."""

from typing import Mapping

from graphcast import xarray_tree
import numpy as np
from typing_extensions import Protocol
import xarray

import jax
import jax.numpy as jnp
import optax
from typing import Tuple
import torch

LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]


class LossFunction(Protocol):
  """A loss function.

  This is a protocol so it's fine to use a plain function which 'quacks like'
  this. This is just to document the interface.
  """

  def __call__(self,
               predictions: xarray.Dataset,
               targets: xarray.Dataset,
               **optional_kwargs) -> LossAndDiagnostics:
    """Computes a loss function.

    Args:
      predictions: Dataset of predictions.
      targets: Dataset of targets.
      **optional_kwargs: Implementations may support extra optional kwargs.

    Returns:
      loss: A DataArray with dimensions ('batch',) containing losses for each
        element of the batch. These will be averaged to give the final
        loss, locally and across replicas.
      diagnostics: Mapping of additional quantities to log by name alongside the
        loss. These will will typically correspond to terms in the loss. They
        should also have dimensions ('batch',) and will be averaged over the
        batch before logging.
    """


def weighted_mse_per_level(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:
  """Latitude- and pressure-level-weighted MSE loss."""
  def loss(prediction, target):
    loss = (prediction - target)**2
    loss *= normalized_latitude_weights(target).astype(loss.dtype)
    if 'level' in target.dims:
      loss *= normalized_level_weights(target).astype(loss.dtype)
    return _mean_preserving_batch(loss)

  losses = xarray_tree.map_structure(loss, predictions, targets)
  return sum_per_variable_losses(losses, per_variable_weights)




def two_channel_crossentropy_optax(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,              # ← 和旧接口保持一致，虽然这里用不到
    one_hot,                          # ← 你在 Dataset 里生成的真 one-hot
    pos_weight: float = 20.0,
    use_dynamic_weight: bool = True,  # 是否使用动态权重
    dynamic_weight_scale: float = 0.1,  # 动态权重的缩放因子
    min_pos_weight: float = 1.0,  # 最小正例权重
    max_pos_weight: float = 1000.0,  # 最大正例权重
) -> Tuple[xarray.DataArray, xarray.DataArray, dict]:
    """Soft-max CE on (u,v) logits, using pre-computed storm-mask one-hot.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets (not used but kept for interface consistency)
        one_hot: One-hot encoded labels (B, H, W, 2)
        pos_weight: Base positive weight (used if use_dynamic_weight=False)
        use_dynamic_weight: If True, compute weight dynamically based on pos/neg ratio
        dynamic_weight_scale: Scaling factor for dynamic weight calculation
        min_pos_weight: Minimum allowed positive weight
        max_pos_weight: Maximum allowed positive weight
    """
    # ── 1. 取 (u,v) 并在需要时压 level、time ─────────────────────────────
    u = predictions["u_component_of_wind"]
    v = predictions["v_component_of_wind"]
    if "level" in u.dims:
        u, v = u.mean("level"), v.mean("level")
    if "time" in u.dims and u.sizes["time"] == 1:
        u, v = u.isel(time=0), v.isel(time=0)

    # ── 2. 组织 logits → (B,H,W,2) ─────────────────────────────────────
    logits_xr = xarray.concat([u, v], dim="channel").transpose("batch", "lat", "lon", "channel")
    logits = logits_xr.data.jax_array           # JAX DeviceArray,  fp32

    # ── 3. 把 PyTorch / NumPy 的 one_hot 转成 JAX DeviceArray ───────────
    labels = jnp.asarray(one_hot, dtype=logits.dtype)   # 不再走 np.asarray!


    # ── 4. 计算 per-pixel softmax-CE ────────────────────────────────────
    ce_px = optax.softmax_cross_entropy(logits, labels)      # (B,H,W)

    # ── 5. 计算权重（动态或固定） ────────────────────────────────────────
    if use_dynamic_weight:
        # 动态权重：根据每个样本的实际正负比例计算
        # 计算每个样本的正例和负例数量
        n_pos_raw = jnp.sum(labels[..., 1], axis=(1, 2))  # (B,) 每个样本的正例数
        n_neg_raw = labels.shape[1] * labels.shape[2] - n_pos_raw  # (B,) 每个样本的负例数
        
        # 避免除零
        n_pos = jnp.maximum(n_pos_raw, 1.0)
        n_neg = jnp.maximum(n_neg_raw, 1.0)
        
        # 计算 ratio
        ratio = n_neg / n_pos
        
        # 动态权重：负例数 / 正例数 * 缩放因子（平衡类别不平衡）
        dynamic_pos_weight_before = (n_neg / n_pos) * dynamic_weight_scale
        
        # 打印调试信息：显示正负样本数量和计算出的权重（在限制之前）
        jax.debug.print(
            "[DEBUG Dynamic Weight] n_pos={n_pos}, n_neg={n_neg}, "
            "ratio={ratio}, weight_before_clip={weight_before_clip}",
            n_pos=n_pos,
            n_neg=n_neg,
            ratio=ratio,
            weight_before_clip=dynamic_pos_weight_before,
            ordered=True  # 确保打印顺序
        )
        
        # 限制权重范围，避免极端值
        dynamic_pos_weight_after = jnp.clip(dynamic_pos_weight_before, min_pos_weight, max_pos_weight)
        
        # 打印限制后的权重
        jax.debug.print(
            "[DEBUG Dynamic Weight] weight_after_clip={weight_after_clip}",
            weight_after_clip=dynamic_pos_weight_after
        )
        
        # 扩展到空间维度 (B, H, W)
        pos_weight_per_pixel = dynamic_pos_weight_after[:, None, None]
        
        # 收集指标（取batch平均，保持为 JAX 数组，在外部转换为 Python 类型）
        # 使用 stop_gradient 确保这些指标不影响梯度计算
        weight_metrics = {
            'n_pos': jax.lax.stop_gradient(jnp.mean(n_pos_raw)),  # batch平均，使用原始值
            'ratio': jax.lax.stop_gradient(jnp.mean(ratio)),  # batch平均
            'weight_before_clip': jax.lax.stop_gradient(jnp.mean(dynamic_pos_weight_before)),  # batch平均
            'weight_after_clip': jax.lax.stop_gradient(jnp.mean(dynamic_pos_weight_after)),  # batch平均
        }
    else:
        # 使用固定权重
        pos_weight_per_pixel = pos_weight
        weight_metrics = {
            'n_pos': None,
            'ratio': None,
            'weight_before_clip': pos_weight,
            'weight_after_clip': pos_weight,
        }

    # ── 6. 给正例(通道1==1)加权 ────────────────────────────────────────
    w = jnp.where(labels[..., 1] == 1, pos_weight_per_pixel, 1.0)
    weighted_ce = ce_px * w

    # ── 7. spatial mean → loss per sample ─────────────────────────────
    loss_per_sample = weighted_ce.mean(axis=(1, 2))

    # ── 8. wrap 回 xarray，保持旧函数输出格式 ───────────────────────────
    loss_da = xarray.DataArray(
        loss_per_sample,
        dims=("batch",),
        coords={"batch": logits_xr.coords["batch"]},
    )
    return loss_da, labels, weight_metrics  # 返回 loss, labels, 和指标字典





def debug_two_channel_crossentropy_optax(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    pos_weight: float = 20,
) -> xarray.DataArray:
    """
    Softmax-CE on (u,v) logits with heavy weight on storm pixels,
    with debug printouts at each major step.
    """
    # ==== Step 1: collapse 'level' ====
    u = predictions["u_component_of_wind"]
    v = predictions["v_component_of_wind"]
    if "level" in u.dims:
        u = u.mean(dim="level")
        v = v.mean(dim="level")
    jax.debug.print("==== After collapse level: u shape = {}, v shape = {}", u.shape, v.shape)

    # ==== Step 2: drop singleton 'time' ====
    if "time" in u.dims and u.sizes["time"] == 1:
        u = u.isel(time=0)
        v = v.isel(time=0)
    jax.debug.print("==== After drop time: u shape = {}, v shape = {}", u.shape, v.shape)

    # ==== Step 3: stack -> logits (B,H,W,2) ====
    logits_xr = (
        xarray.concat([u, v], dim="channel")
        .transpose("batch", "lat", "lon", "channel")
    )
    logits = logits_xr.data.jax_array  # JAX DeviceArray
    jax.debug.print("==== Logits tensor shape = {}", logits.shape)

    B, H, W, _ = logits.shape

    # # ==== Step 4: dummy labels: centre pixel = 1 ====
    # r = 40  # “radius” of region (so region is 5×5)
    # yc, xc = H // 2, W // 2
    # labels = jnp.zeros((B, H, W), dtype=jnp.int32)
    # labels = labels.at[:, yc - r : yc + r + 1, xc - r : xc + r + 1].set(1)
    # positives = labels.sum(axis=(1, 2))
    # jax.debug.print("==== Dummy labels sum (positives per sample) = {}", positives)

    # ==== Step 4: dummy labels: centre star-shaped region = 1 ====
    r = 40  # 控制星的“半径”
    yc, xc = H // 2, W // 2

    # 构造网格坐标
    yy = jnp.arange(H)[:, None]    # shape (H,1)
    xx = jnp.arange(W)[None, :]    # shape (1,W)
    dy = yy - yc                   # shape (H,W)
    dx = xx - xc                   # shape (H,W)

    # 极坐标表示
    angle = jnp.arctan2(dy, dx)    # [-π, π]
    radius = jnp.sqrt(dx**2 + dy**2)

    # 五角星的径向函数：r(θ) = R * (0.5 + 0.5*cos(5θ))
    radial_factor = 0.5 + 0.5 * jnp.cos(5 * angle)

    # 生成 2D 星形 mask
    mask2d = (radius <= r * radial_factor).astype(jnp.int32)  # shape (H,W)

    # 扩展到 batch 维
    labels = jnp.broadcast_to(mask2d[None, ...], (B, H, W))

    positives = labels.sum(axis=(1, 2))
    jax.debug.print("==== Dummy labels sum (positives per sample) = {}", positives)

    # ==== Step 5: one-hot & CE ====
    one_hot = jax.nn.one_hot(labels, num_classes=2)


    ce_px = optax.softmax_cross_entropy(logits, one_hot)
    jax.debug.print("==== CE at centre pixel = {}", ce_px[:, H//2, W//2])
    bg_mean = jnp.sum(ce_px * (1 - labels), axis=(1,2)) / (H*W - 1)
    jax.debug.print("==== Mean CE on background = {}", bg_mean)

    # ==== Step 6: positive-class weighting ====
    w_pos = jnp.asarray(pos_weight, dtype=logits.dtype)
    jax.debug.print("==== Positive weight scalar = {}", w_pos)
    weights = jnp.where(labels == 1, w_pos, 1.0)
    weighted_ce = ce_px * weights
    jax.debug.print("==== Weighted CE at centre = {}", weighted_ce[:, H//2, W//2])
    bg_wmean = jnp.sum(weighted_ce * (1 - labels), axis=(1,2)) / (H*W - 1)
    jax.debug.print("==== Mean weighted CE on background = {}", bg_wmean)

    # ==== Step 7: mean over space -> loss per sample ====
    loss_per_sample = jnp.mean(weighted_ce, axis=(1, 2))
    jax.debug.print("==== Loss per sample = {}", loss_per_sample)
    jax.debug.print("==== End of loss computation ====")

    # ==== Step 8: wrap into xarray ====
    loss_da = xarray.DataArray(
        loss_per_sample,
        dims=("batch",),
        coords={"batch": logits_xr.coords["batch"]},
    )
    return loss_da, one_hot


def debug_center_mse_optax(predictions: xarray.Dataset,
                           targets: xarray.Dataset) -> xarray.DataArray:
    """
    Debug MSE loss: force network to output 1 in a small center patch (radius r),
    and 0 elsewhere, regardless of targets. Returns per‐batch MSE.
    """
    import jax.numpy as jnp
    import xarray as xr

    # 1. take the first (and only) DataArray in predictions
    pred_da = next(iter(predictions.data_vars.values()))

    # 2. remove any singleton 'time' or 'level' dims
    for dim in ("time", "level"):
        if dim in pred_da.dims and pred_da.sizes[dim] == 1:
            pred_da = pred_da.isel({dim: 0})

    # 3. reorder dims to (batch, lat, lon)
    pred_da = pred_da.transpose("batch", "lat", "lon")

    # 4. extract the JAX array
    pred = pred_da.data.jax_array       # shape [B, H, W]

    B, H, W = pred.shape

    # 5. build the binary center mask
    r = 50  # radius of the square patch; feel free to adjust
    yc, xc = H // 2, W // 2
    labels = jnp.zeros((B, H, W), dtype=pred.dtype)
    labels = labels.at[:, yc - r:yc + r + 1, xc - r:xc + r + 1].set(0.0)

    # 6. compute per-sample mean squared error
    loss_per_sample = jnp.mean((pred - labels) ** 2, axis=(1, 2))

    # 7. wrap back into an xarray.DataArray
    loss_da = xr.DataArray(
        loss_per_sample,
        dims=("batch",),
        coords={"batch": pred_da.coords["batch"]},
    )

    # Emphaszie loss by x100
    loss_da *= 1000.0
    return loss_da




def _mean_preserving_batch(x: xarray.DataArray) -> xarray.DataArray:
  return x.mean([d for d in x.dims if d != 'batch'], skipna=False)


def sum_per_variable_losses(
    per_variable_losses: Mapping[str, xarray.DataArray],
    weights: Mapping[str, float],
) -> LossAndDiagnostics:
  """Weighted sum of per-variable losses."""
  if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
    raise ValueError(
        'Passing a weight that does not correspond to any variable '
        f'{set(weights.keys())-set(per_variable_losses.keys())}')

  weighted_per_variable_losses = {
      name: loss * weights.get(name, 1)
      for name, loss in per_variable_losses.items()
  }
  total = xarray.concat(
      weighted_per_variable_losses.values(), dim='variable', join='exact').sum(
          'variable', skipna=False)
  return total, per_variable_losses  # pytype: disable=bad-return-type


def normalized_level_weights(data: xarray.DataArray) -> xarray.DataArray:
  """Weights proportional to pressure at each level."""
  level = data.coords['level']
  return level / level.mean(skipna=False)


def normalized_latitude_weights(data: xarray.DataArray) -> xarray.DataArray:
  """Weights based on latitude, roughly proportional to grid cell area.

  This method supports two use cases only (both for equispaced values):
  * Latitude values such that the closest value to the pole is at latitude
    (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
    For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
    In this case each point with `lat` value represents a sphere slice between
    `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
    proportional to:
    `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
    we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
    that cancels during normalization.
  * Latitude values that fall exactly at the poles.
    For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
    In this case each point with `lat` value also represents
    a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
    except for the points at the poles, that represent a slice between
    `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
    The areas of the first type of point are still proportional to:
    * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
    but for the points at the poles now is:
    * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
    and we will be using these weights, depending on whether we are looking at
    pole cells, or non-pole cells (omitting the common factor of 2 which will be
    absorbed by the normalization).

    It can be shown via a limit, or simple geometry, that in the small angles
    regime, the proportion of area per pole-point is equal to 1/8th
    the proportion of area covered by each of the nearest non-pole point, and we
    test for this in the test.

  Args:
    data: `DataArray` with latitude coordinates.
  Returns:
    Unit mean latitude weights.
  """
  latitude = data.coords['lat']

  if np.any(np.isclose(np.abs(latitude), 90.)):
    weights = _weight_for_latitude_vector_with_poles(latitude)
  else:
    weights = _weight_for_latitude_vector_without_poles(latitude)

  return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
  """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
  if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
      not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at '
        '+- (90 - delta_latitude/2) degrees.')
  return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
  """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
  if (not np.isclose(np.max(latitude), 90.) or
      not np.isclose(np.min(latitude), -90.)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
  weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
  # The two checks above enough to guarantee that latitudes are sorted, so
  # the extremes are the poles
  weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
  return weights


def _check_uniform_spacing_and_get_delta(vector):
  diff = np.diff(vector)
  if not np.all(np.isclose(diff[0], diff)):
    raise ValueError(f'Vector {diff} is not uniformly spaced.')
  return diff[0]
