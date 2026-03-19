# Copyright 2024 DeepMind Technologies Limited.
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
"""DPM-Solver++ 2S sampler from https://arxiv.org/abs/2211.01095."""

from typing import Optional, Tuple, List, Dict

from graphcast import casting
from graphcast import denoisers_base
from graphcast import samplers_base as base
from graphcast import samplers_utils as utils
from graphcast import xarray_jax
import haiku as hk
import jax.numpy as jnp
import xarray
import optax
import jax



import jax, jax.numpy as jnp
from jax import tree_util
import xarray as xr
from graphcast import xarray_jax

def _unwrap(da):
    # 支持 xarray.DataArray / jnp.ndarray
    return xarray_jax.unwrap_data(da, require_jax=True) if isinstance(da, xr.DataArray) else da

def ds_global_l2(ds: xr.Dataset) -> jnp.ndarray:
    """xarray.Dataset 的整体 L2 范数（所有 data_vars 拼一起算）。"""
    vals = []
    for v in ds.data_vars.values():
        a = _unwrap(v)
        vals.append(jnp.sum(a * a))
    return jnp.sqrt(jnp.sum(jnp.stack(vals))) if vals else jnp.asarray(0., dtype=jnp.float32)

def ds_delta_l2(a: xr.Dataset, b: xr.Dataset) -> jnp.ndarray:
    """||a - b||_2（结构相同假设）。"""
    vals = []
    for k in a.data_vars.keys():
        aa = _unwrap(a[k]); bb = _unwrap(b[k])
        vals.append(jnp.sum((aa - bb) ** 2))
    return jnp.sqrt(jnp.sum(jnp.stack(vals))) if vals else jnp.asarray(0., dtype=jnp.float32)

def tree_global_l2(pytree) -> jnp.ndarray:
    """通用 PyTree 的 L2 范数（给 grads/updates 用）。"""
    leaves = tree_util.tree_leaves(pytree)
    parts = []
    for l in leaves:
        l = _unwrap(l)
        parts.append(jnp.sum(l * l))
    return jnp.sqrt(jnp.sum(jnp.stack(parts))) if parts else jnp.asarray(0., dtype=jnp.float32)

import numpy as np
def save_npy_cb(x, path: str):
    # 这个函数在 host 上执行（非 JIT 区域）
    np.save(path, np.asarray(x))



class Sampler_ReadOut(base.Sampler):
  """Sampling using DPM-Solver++ 2S from [1].

  This is combined with optional stochastic churn as described in [2].

  The '2S' terminology from [1] means that this is a second-order (2),
  single-step (S) solver. Here 'single-step' here distinguishes it from
  'multi-step' methods where the results of function evaluations from previous
  steps are reused in computing updates for subsequent steps. The solver still
  uses multiple steps though.

  [1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
  Models, https://arxiv.org/abs/2211.01095
  [2] Elucidating the Design Space of Diffusion-Based Generative Models,
  https://arxiv.org/abs/2206.00364
  """

  def __init__(self,
               denoiser: denoisers_base.Denoiser,
               max_noise_level: float,
               min_noise_level: float,
               num_noise_levels: int,
               rho: float,
               stochastic_churn_rate: float,
               churn_min_noise_level: float,
               churn_max_noise_level: float,
               noise_level_inflation_factor: float,
               selected_denoising_step: list[int]
               ):
    """Initializes the sampler.

    Args:
      denoiser: A Denoiser which predicts noise-free targets.
      max_noise_level: The highest noise level used at the start of the
        sequence of reverse diffusion steps.
      min_noise_level: The lowest noise level used at the end of the sequence of
        reverse diffusion steps.
      num_noise_levels: Determines the number of noise levels used and hence the
        number of reverse diffusion steps performed.
      rho: Parameter affecting the spacing of noise steps. Higher values will
        concentrate noise steps more around zero.
      stochastic_churn_rate: S_churn from the paper. This controls the rate
        at which noise is re-injected/'churned' during the sampling algorithm.
        If this is set to zero then we are performing deterministic sampling
        as described in Algorithm 1.
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
      selected_denoising_step: List of denoising steps at which to collect readout
        predictions. Default is [10, 15].
    """
    super().__init__(denoiser)
    self._noise_levels = utils.noise_schedule(
        max_noise_level, min_noise_level, num_noise_levels, rho)
    self._stochastic_churn = stochastic_churn_rate > 0
    self._per_step_churn_rates = utils.stochastic_churn_rate_schedule(
        self._noise_levels, stochastic_churn_rate, churn_min_noise_level,
        churn_max_noise_level)
    self._noise_level_inflation_factor = noise_level_inflation_factor
    self._selected_denoising_step = selected_denoising_step


  def guided(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      *,
      # 新增：把内层优化的配置传进来
      inner_opt_step_idxs: List[int],
      inner_opt_steps_map: Dict[int, int],  # 新增：{step: num_opt_steps}
      max_opt_steps: int,                    # 新增：最大优化次数（用于 JAX 编译）
      inner_opt_lr: float,
      loss_type: str = "readout_l2",
      target_readout_tpl: Optional[xarray.Dataset] = None,
      guidance_mask: Optional[jnp.ndarray] = None,  # 新增：选择性 mask (B, H, W)
      warp_configs: Optional[dict] = None,          # 新增：warp 配置
      **kwargs
  ):
      dtype = casting.infer_floating_dtype(targets_template)
      noise_levels = jnp.array(self._noise_levels).astype(dtype)
      per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype)

      # —— 预处理 inner_opt_step_idxs：去重、过滤越界，转 jnp.array —— 
      max_step = int(len(noise_levels) - 1)  # 最后一步不能内优（没有 next sigma）
      _steps_py = list(dict.fromkeys(int(s) for s in inner_opt_step_idxs))  # 去重并转 int
      _valid_steps_py = [s for s in _steps_py if 0 <= s < max_step]         # 过滤越界
      steps_arr = jnp.array(_valid_steps_py, dtype=jnp.int32)               # 供 JAX 比较用

      # 可选：打印一次最终有效索引
      jax.debug.print("valid inner_opt_step_idxs: {s}", s=steps_arr)
      
      # ========= Local Affine Warp 辅助函数 =========
      def create_circular_mask_jax(lat_grid, lon_grid, center_lat, center_lon, radius):
          """
          创建 circular mask（hard mask）
          Args:
              lat_grid: (H, W) 纬度网格
              lon_grid: (H, W) 经度网格
              center_lat, center_lon: 圆心坐标
              radius: 半径（度）
          Returns:
              mask: (H, W) 0-1 mask
          """
          # 简化的距离计算（平面近似）
          dist = jnp.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
          mask = (dist <= radius).astype(dtype)
          return mask
      
      def apply_inverse_affine_transform(lat_dst, lon_dst, center_lat, center_lon, 
                                         translation, rotation, scale):
          """
          应用逆仿射变换：从目标坐标计算源坐标
          Args:
              lat_dst, lon_dst: 目标坐标 (H, W)
              center_lat, center_lon: 变换中心
              translation: [delta_lat, delta_lon]
              rotation: theta (弧度)
              scale: [scale_x, scale_y]
          Returns:
              lat_src, lon_src: 源坐标 (H, W)
          """
          # 1. 相对于中心
          lat_rel = lat_dst - center_lat
          lon_rel = lon_dst - center_lon
          
          # 2. 减去平移
          lat_rel = lat_rel - translation[0]
          lon_rel = lon_rel - translation[1]
          
          # 3. 逆缩放
          lat_rel = lat_rel / scale[0]
          lon_rel = lon_rel / scale[1]
          
          # 4. 逆旋转
          cos_t = jnp.cos(rotation)
          sin_t = jnp.sin(rotation)
          lat_rot = cos_t * lat_rel + sin_t * lon_rel
          lon_rot = -sin_t * lat_rel + cos_t * lon_rel
          
          # 5. 加回中心
          lat_src = lat_rot + center_lat
          lon_src = lon_rot + center_lon
          
          return lat_src, lon_src
      
      def apply_local_affine_warp_jax(x_dataset, warp_params, warp_cfg):
          """
          对 xarray.Dataset 应用局部仿射变换（带 smooth falloff）
          Args:
              x_dataset: xarray.Dataset with (batch, time, lat, lon, [level])
              warp_params: {"translation": [2,], "rotation": scalar, "scale": [2,]}
              warp_cfg: warp 配置字典
          Returns:
              x_warped: 变换后的 xarray.Dataset
          """
          if warp_cfg is None or not warp_cfg.get("enabled", False):
              return x_dataset
          
          # 提取配置
          center_lat = warp_cfg.get("center_lat", 0.0)
          center_lon = warp_cfg.get("center_lon", 0.0)
          radius = warp_cfg.get("radius", 10.0)
          falloff_width = warp_cfg.get("falloff_width", 2.0)
          variables = warp_cfg.get("variables", None)
          
          # 获取 lat, lon 坐标
          lat_1d = x_dataset.coords["lat"].values
          lon_1d = x_dataset.coords["lon"].values
          lat_grid, lon_grid = jnp.meshgrid(lat_1d, lon_1d, indexing='ij')
          
          # 创建 circular mask
          mask = create_circular_mask_jax(lat_grid, lon_grid, center_lat, center_lon, radius)
          
          # 计算逆变换后的源坐标
          lat_src, lon_src = apply_inverse_affine_transform(
              lat_grid, lon_grid, center_lat, center_lon,
              warp_params["translation"], warp_params["rotation"], warp_params["scale"]
          )
          
          # Smooth falloff: 基于目标坐标（lat_grid/lon_grid）计算，与 warp_params 无关。
          # 注意：不能用 lat_src/lon_src（依赖 translation），否则 jnp.sqrt 在 dist=0 处
          # 梯度为 +inf，导致 grad_warp["translation"] = NaN，全链路崩溃。
          dist_dst_sq = (lat_grid - center_lat)**2 + (lon_grid - center_lon)**2
          dist_dst = jnp.sqrt(jnp.maximum(dist_dst_sq, 1e-10))  # eps 防止 sqrt(0) 梯度 NaN
          falloff_mask = jnp.where(
              dist_dst > radius,
              jnp.exp(-((dist_dst - radius) / falloff_width)**2),
              1.0
          )
          
          # 将源坐标转换为 pixel 索引（用于 map_coordinates）
          lat_min, lat_max = lat_1d.min(), lat_1d.max()
          lon_min, lon_max = lon_1d.min(), lon_1d.max()
          H, W = len(lat_1d), len(lon_1d)
          
          # 归一化到 [0, H-1] 和 [0, W-1]
          lat_idx = (lat_src - lat_min) / (lat_max - lat_min) * (H - 1)
          lon_idx = (lon_src - lon_min) / (lon_max - lon_min) * (W - 1)
          
          # 构建 coordinates array: shape (2, H, W)
          coords = jnp.stack([lat_idx, lon_idx], axis=0)
          
          # 对每个变量应用 warp
          x_warped = x_dataset.copy()
          for var_name in x_dataset.data_vars:
              # 检查是否在选择的变量列表中
              if variables is not None and var_name not in variables:
                  continue
              
              var_data = x_dataset[var_name]
              arr = xarray_jax.unwrap_data(var_data, require_jax=True)
              
              # 处理不同的维度结构
              if "level" in var_data.dims:
                  # (batch, time, level, lat, lon)
                  B, T, L, H_, W_ = arr.shape
                  arr_warped = jnp.zeros_like(arr)
                  for b in range(B):
                      for t in range(T):
                          for lev in range(L):
                              # 应用 warp
                              slice_2d = arr[b, t, lev, :, :]
                              warped_2d = jax.scipy.ndimage.map_coordinates(
                                  slice_2d, coords, order=1, mode='nearest'
                              )
                              # 应用 mask 和 falloff
                              blended = mask * falloff_mask * warped_2d + (1 - mask * falloff_mask) * slice_2d
                              arr_warped = arr_warped.at[b, t, lev, :, :].set(blended)
              else:
                  # (batch, time, lat, lon)
                  B, T, H_, W_ = arr.shape
                  arr_warped = jnp.zeros_like(arr)
                  for b in range(B):
                      for t in range(T):
                          slice_2d = arr[b, t, :, :]
                          warped_2d = jax.scipy.ndimage.map_coordinates(
                              slice_2d, coords, order=1, mode='nearest'
                          )
                          # 应用 mask 和 falloff
                          blended = mask * falloff_mask * warped_2d + (1 - mask * falloff_mask) * slice_2d
                          arr_warped = arr_warped.at[b, t, :, :].set(blended)
              
              # 更新 dataset
              x_warped = x_warped.assign({var_name: (var_data.dims, arr_warped)})
          
          return x_warped
      
      def warp_regularization_loss(warp_params, init_params):
          """
          Warp 参数的正则化 loss (L2)
          """
          trans_diff = warp_params["translation"] - init_params["translation"]
          rot_diff = warp_params["rotation"] - init_params["rotation"]
          scale_diff = warp_params["scale"] - init_params["scale"]
          
          reg_loss = (jnp.sum(trans_diff**2) + rot_diff**2 + jnp.sum(scale_diff**2))
          return reg_loss
      # ========= Local Affine Warp 辅助函数结束 =========

      # denoiser 保持与你现有的一致：返回 (x_denoised, readout)
      def denoiser(noise_level: jnp.ndarray, x: xarray.Dataset):
          bcast_noise_level = xarray_jax.DataArray(
              jnp.tile(noise_level, x.sizes['batch']), dims=('batch',))
          return self._denoiser(
              inputs=inputs,
              noisy_targets=x,
              noise_levels=bcast_noise_level,
              forcings=forcings)

      # ========= 内层 latent optimization =========
      # 只对 x 求导，不对 denoiser 参数求导；loss 可以用 readout vs target_readout 的 L2
      def _readout_l2_loss(x_var: xarray.Dataset, sigma: jnp.ndarray) -> jnp.ndarray:
          # 注意：denoiser 返回 (x_denoised, readout)，我们只用 readout 来计算损失
          _, readout = denoiser(sigma, x_var)
          # 把 readout 与 target_readout_tpl 做 L2（逐变量平均）
          losses = []
          cnt = 0
          for k, da_pred in readout.data_vars.items():
              da_tgt = target_readout_tpl[k]
              arr_pred = xarray_jax.unwrap_data(da_pred, require_jax=True)
              arr_tgt  = xarray_jax.unwrap_data(da_tgt,  require_jax=True)
              losses.append(jnp.mean((arr_pred - arr_tgt) ** 2))
              cnt += 1
          # 防御：如果没有变量，就 0
          return (sum(losses) / cnt) if cnt > 0 else jnp.asarray(0., dtype=dtype)

      def _xt_l2_loss(x_var: xarray.Dataset, sigma: jnp.ndarray) -> jnp.ndarray:
          # 如果你想沿用之前 _loss_from_xt 的“对 x_t 本身做 L2”
          losses, cnt = [], 0
          for da in x_var.data_vars.values():
              arr = xarray_jax.unwrap_data(da, require_jax=True)
              losses.append(jnp.mean(arr * arr))
              cnt += 1
          return (sum(losses) / cnt) if cnt > 0 else jnp.asarray(0., dtype=dtype)
 
      def two_channel_crossentropy_optax(
          predictions: xarray.Dataset,
          pos_weight: float = 20.0,
      ) -> xarray.DataArray:
          one_hot = target_readout_tpl
          pos_weight = 1000
          """Soft-max CE on (u,v) logits, using pre-computed storm-mask one-hot."""
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

          # save for debug
          # jax.debug.callback(
          #     save_npy_cb,
          #     logits, '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/debug_result/09-08_two_step/debug_logits.npy'
          # )
          # jax.debug.callback(
          #     save_npy_cb,
          #     labels, '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/debug_result/09-08_two_step/debug_labels.npy'
          # )

          # ── 5. 给正例(通道1==1)加权 ────────────────────────────────────────
          w = jnp.where(labels[..., 1] == 1, pos_weight, 1.0)
          weighted_ce = ce_px * w

          # ── 6. spatial mean → loss per sample ─────────────────────────────
          loss_per_sample = weighted_ce.mean(axis=(1, 2))

          # ── 7. wrap 回 xarray，保持旧函数输出格式 ───────────────────────────
          # loss_da = xarray.DataArray(
          #     loss_per_sample,
          #     dims=("batch",),
          #     coords={"batch": logits_xr.coords["batch"]},
          # )
          return loss_per_sample.mean()  # jnp scalar

      def two_channel_crossentropy_optax_with_guidance_mask(
          predictions: xarray.Dataset,
          guidance_mask: Optional[jnp.ndarray] = None,  # (B, H, W) binary mask
          pos_weight: float = 20.0,
      ) -> xarray.DataArray:
          """带 guidance_mask 的 loss 函数，只在 mask 区域计算 loss"""
          one_hot = target_readout_tpl
          pos_weight = 1000
          """Soft-max CE on (u,v) logits, using pre-computed storm-mask one-hot."""
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

          # ── 5. 给正例(通道1==1)加权 ────────────────────────────────────────
          w = jnp.where(labels[..., 1] == 1, pos_weight, 1.0)
          weighted_ce = ce_px * w

          # ── 6. 应用 guidance_mask（如果提供） ────────────────────────────────
          if guidance_mask is not None:
              # guidance_mask 形状应该是 (B, H, W)
              # 只在 mask == 1 的区域计算 loss，其他区域设为 0
              weighted_ce = weighted_ce * guidance_mask
              
              # 按 mask 区域归一化（而不是整个空间）
              mask_sum = guidance_mask.sum(axis=(1, 2), keepdims=True)  # (B, 1, 1)
              # 避免除零
              mask_sum = jnp.maximum(mask_sum, 1.0)
              loss_per_sample = weighted_ce.sum(axis=(1, 2)) / mask_sum.squeeze()
          else:
              # 原有行为：在整个空间计算
              loss_per_sample = weighted_ce.mean(axis=(1, 2))

          return loss_per_sample.mean()  # jnp scalar

      def _loss(x_var: xarray.Dataset, sigma: jnp.ndarray, warp_params=None) -> jnp.ndarray:
          # 如果提供了 warp_params，先应用 warp
          if warp_params is not None and warp_configs is not None and warp_configs.get("enabled", False):
              x_var = apply_local_affine_warp_jax(x_var, warp_params, warp_configs)
          
          _, readout = denoiser(sigma, x_var)
          # return _readout_l2_loss(x_var, sigma) if loss_type == "readout_l2" else _xt_l2_loss(x_var, sigma)
          readout_loss = 0.0
          if loss_type == "readout_l2":
              if guidance_mask is not None:
                  readout_loss = two_channel_crossentropy_optax_with_guidance_mask(readout, guidance_mask=guidance_mask, pos_weight=1000)
              else:
                  readout_loss = two_channel_crossentropy_optax(readout)
          else:
              readout_loss = _xt_l2_loss(x_var, sigma)
          
          # 添加 warp regularization
          if warp_params is not None and warp_configs is not None and warp_configs.get("enabled", False):
              init_params = {
                  "translation": jnp.array(warp_configs.get("init_translation", [0.0, 0.0])),
                  "rotation": jnp.array(warp_configs.get("init_rotation", 0.0)),
                  "scale": jnp.array(warp_configs.get("init_scale", [1.0, 1.0])),
              }
              reg_weight = warp_configs.get("regularization_weight", 1e-3)
              reg_loss = warp_regularization_loss(warp_params, init_params)
              return readout_loss + reg_weight * reg_loss
          
          return readout_loss

      # "把 x 当作参数" 来跑 optax：x 是一个 pytree（xarray.Dataset），optax 完全支持
      opt = optax.chain(
          optax.clip_by_global_norm(1.0),       # 可选：稳定训练
          optax.adam(inner_opt_lr)              # 或 optax.sgd(inner_opt_lr)
      )
      
      # 如果启用 warp，创建独立的 warp optimizer
      use_warp = warp_configs is not None and warp_configs.get("enabled", False)
      if use_warp:
          warp_lr = warp_configs.get("learning_rate", 1e-2)
          opt_warp = optax.adam(warp_lr)

      # 预先构建 step → opt_steps 的 lookup array（长度为 20，覆盖所有 denoising steps）
      # 对于不在 map 中的 step，默认值为 max_opt_steps
      _steps_lookup = jnp.array([
          inner_opt_steps_map.get(i, max_opt_steps) for i in range(20)
      ], dtype=jnp.int32)

      def inner_optimize(x_init: xarray.Dataset, sigma: jnp.ndarray, current_step: jnp.ndarray):
          """
          Per-step optimization: 根据 current_step 查找该 step 的 optimization 次数，
          用 max_opt_steps 作为 scan length（静态），但条件跳过超出的迭代。
          支持联合优化 x_t 和 warp_params（如果启用）。
          """
          # 注意：不要让优化器状态携带跨 step 的历史；每次触发都重新 init
          opt_state_x = opt.init(x_init)
          
          # 初始化 warp_params（如果启用）
          if use_warp:
              warp_params_init = {
                  "translation": jnp.array(warp_configs.get("init_translation", [0.0, 0.0]), dtype=dtype),
                  "rotation": jnp.array(warp_configs.get("init_rotation", 0.0), dtype=dtype),
                  "scale": jnp.array(warp_configs.get("init_scale", [1.0, 1.0]), dtype=dtype),
              }
              opt_state_warp = opt_warp.init(warp_params_init)
          else:
              warp_params_init = None
              opt_state_warp = None
          
          # 查找当前 step 的 optimization 次数
          this_step_limit = _steps_lookup[current_step]
          jax.debug.print("step {s}: inner_opt limit = {l}, use_warp = {w}", 
                         s=current_step, l=this_step_limit, w=use_warp)

          def one_step(carry, iteration_idx):
              x_curr, state_x, warp_curr, state_warp = carry
              
              # 只有 iteration_idx < this_step_limit 时才真正更新
              should_update = iteration_idx < this_step_limit
              
              if use_warp:
                  # 联合优化：对 x 和 warp_params 分别计算梯度
                  loss_val, (grad_x, grad_warp) = jax.value_and_grad(_loss, argnums=(0, 2))(
                      x_curr, sigma, warp_curr
                  )
                  
                  # 更新 x
                  updates_x, new_state_x = opt.update(grad_x, state_x, params=x_curr)
                  x_next = jax.tree_util.tree_map(lambda p, u: p + u, x_curr, updates_x)
                  
                  # 更新 warp_params（只优化启用的参数）
                  # 根据配置选择性优化
                  optimize_trans = warp_configs.get("optimize_translation", True)
                  optimize_rot = warp_configs.get("optimize_rotation", False)
                  optimize_scale = warp_configs.get("optimize_scale", False)
                  
                  # 屏蔽不需要优化的参数的梯度
                  grad_warp_masked = {
                      "translation": grad_warp["translation"] if optimize_trans else jnp.zeros_like(grad_warp["translation"]),
                      "rotation": grad_warp["rotation"] if optimize_rot else jnp.zeros((), dtype=dtype),
                      "scale": grad_warp["scale"] if optimize_scale else jnp.zeros_like(grad_warp["scale"]),
                  }
                  
                  updates_warp, new_state_warp = opt_warp.update(grad_warp_masked, state_warp, params=warp_curr)
                  warp_next = jax.tree_util.tree_map(lambda p, u: p + u, warp_curr, updates_warp)
                  
                  # Clip warp params to prevent extreme values
                  warp_next = {
                      "translation": jnp.clip(warp_next["translation"], -20.0, 20.0),
                      "rotation": jnp.clip(warp_next["rotation"], -jnp.pi, jnp.pi),
                      "scale": jnp.clip(warp_next["scale"], 0.5, 2.0),
                  }
                  
              else:
                  # 只优化 x（原有行为）
                  loss_val, grad_x = jax.value_and_grad(_loss)(x_curr, sigma, None)
                  updates_x, new_state_x = opt.update(grad_x, state_x, params=x_curr)
                  x_next = jax.tree_util.tree_map(lambda p, u: p + u, x_curr, updates_x)
                  warp_next = warp_curr
                  new_state_warp = state_warp

              # 条件更新：如果超出 limit 则保持原样
              x_out = jax.lax.cond(should_update, lambda: x_next, lambda: x_curr)
              state_x_out = jax.lax.cond(should_update, lambda: new_state_x, lambda: state_x)
              
              if use_warp:
                  warp_out = jax.lax.cond(should_update, lambda: warp_next, lambda: warp_curr)
                  state_warp_out = jax.lax.cond(should_update, lambda: new_state_warp, lambda: state_warp)
              else:
                  warp_out = warp_curr
                  state_warp_out = state_warp
              
              # 只在真正更新时打印
              if use_warp:
                  jax.lax.cond(
                      should_update,
                      lambda: jax.debug.print(
                          "[inner {i}]: loss={lv:.4e}, ||grad_x||={gn:.4e}, trans={t}, rot={r:.4f}",
                          i=iteration_idx, lv=loss_val, gn=tree_global_l2(grad_x), 
                          t=warp_out["translation"], r=warp_out["rotation"]
                      ),
                      lambda: None
                  )
              else:
                  jax.lax.cond(
                      should_update,
                      lambda: jax.debug.print(
                          "[inner {i}]: loss={lv:.4e}, ||grad||={gn:.4e}, ||update||={un:.4e}",
                          i=iteration_idx, lv=loss_val, gn=tree_global_l2(grad_x), un=tree_global_l2(updates_x)
                      ),
                      lambda: None
                  )
              
              return (x_out, state_x_out, warp_out, state_warp_out), loss_val

          (x_final, _, warp_final, _), loss_array = jax.lax.scan(
              one_step, 
              (x_init, opt_state_x, warp_params_init, opt_state_warp), 
              xs=jnp.arange(max_opt_steps)
          )
          # loss_array: shape (max_opt_steps,)
          # 注意：只有前 this_step_limit 个是有效值，超出部分的 loss 可能重复但不会被使用
          
          # 如果使用 warp，返回最终应用 warp 后的 x
          if use_warp:
              x_final = apply_local_affine_warp_jax(x_final, warp_final, warp_configs)
          
          return x_final, loss_array
      # ========= 内层 latent optimization 结束 =========


      # —— Warm-up / 预热：在 fori_loop 之前跑一次 denoiser —— 
      # 用最小的 noise level（或你要 inner-opt 的那个步的 sigma 都行）
      sigma_dummy = noise_levels[0]

      # 用与 targets_template 同结构的“零样本”当输入；确保有 batch 维
      x_dummy = xarray.zeros_like(targets_template)

      # 断开梯度，防止把这次热身算进图里
      x_dummy = jax.tree_map(jax.lax.stop_gradient, x_dummy)

      # 跑一次，丢弃输出（不要把任何中间结果存到全局变量）
      _, _ = denoiser(sigma_dummy, x_dummy)
      # —— Warm-up 结束 ——


      # 预分配 readouts
      readouts_init = {step: xarray.zeros_like(targets_template) for step in self._selected_denoising_step}
      x0 = xarray.zeros_like(targets_template)
      
      # 预分配 loss_history: shape (20, max_opt_steps)，填充 -1.0 表示未使用
      loss_history_init = jnp.full((20, max_opt_steps), -1.0, dtype=dtype)

      def body_fn(i: jnp.ndarray, val):
          x, readouts, loss_history = val

          # step0：初始化噪声
          def init_noise(template):
              return noise_levels[0] * utils.spherical_white_noise_like(template)
          maybe_init_noise = jnp.asarray(i == 0, dtype=noise_levels[0].dtype)
          x = x + init_noise(x) * maybe_init_noise

          sigma = noise_levels[i]

          # 可选 churn
          if self._stochastic_churn:
              x, sigma = utils.apply_stochastic_churn(
                  x, sigma,
                  stochastic_churn_rate=per_step_churn_rates[i],
                  noise_level_inflation_factor=self._noise_level_inflation_factor)

          # ===== 在指定步触发"内层 latent optimization" =====
          def do_inner_opt(x_in):
              # （可选）阻断历史梯度路径，确保不回溯跨 step 的图
              x_in = jax.tree_map(jax.lax.stop_gradient, x_in)
              return inner_optimize(x_in, sigma, i)  # 返回 (x_final, loss_array)
          
          def no_inner_opt(x_in):
              # 不做优化时，返回原样 + 空的 loss 数组
              return x_in, jnp.full((max_opt_steps,), -1.0, dtype=dtype)

          # 触发条件：当前 i 是否属于 steps_arr
          trigger = jnp.any(steps_arr == i)

          # 条件执行内层优化
          x, step_losses = jax.lax.cond(
              trigger,
              do_inner_opt,
              no_inner_opt,
              operand=x
          )
          
          # 更新 loss_history[i, :] 为当前 step 的 loss（只在 trigger 时有效）
          # step_losses 已经是 (max_opt_steps,) 的固定长度数组
          loss_history = loss_history.at[i].set(step_losses)

          # pred = (i == inner_opt_step_idx)
          # jax.debug.print("step {i}: inner_opt_triggered? {p}", i=i, p=pred)
          jax.debug.print("step {i}: inner_opt_triggered? {t}", i=i, t=trigger)


          # ====== 继续 DPM-Solver++ 2S 的两次 denoise + 两次线性更新 ======
          x_denoised, ro1 = denoiser(sigma, x)

          for step in self._selected_denoising_step:
              is_selected = jnp.equal(i, jnp.array(step))
              readouts[step] = utils.tree_where(is_selected, ro1, readouts[step])

          sigma_next = noise_levels[i + 1]
          sigma_mid = jnp.sqrt(sigma * sigma_next)
          x_mid = (sigma_mid / sigma) * x + (1.0 - (sigma_mid / sigma)) * x_denoised

          x_mid_denoised, _ = denoiser(sigma_mid, x_mid)

          x_next = (sigma_next / sigma) * x + (1.0 - (sigma_next / sigma)) * x_mid_denoised
          x_final = utils.tree_where(sigma_next == 0, x_denoised, x_next)

          return (x_final, readouts, loss_history)

      # check inner_opt_step_idx
      # jax.debug.print("======== inner_opt_step_idx = {v}", v=inner_opt_step_idx)
      # exit()

      final_state, selected_readouts, loss_history_final = hk.fori_loop(
          0, len(noise_levels) - 1,
          body_fun=body_fn,
          init_val=(x0, readouts_init, loss_history_init),
      )
      return final_state, selected_readouts, loss_history_final



  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> Tuple[xarray.Dataset, Dict[int, xarray.Dataset]]:

    dtype = casting.infer_floating_dtype(targets_template)
    noise_levels = jnp.array(self._noise_levels).astype(dtype)
    per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype)

    def denoiser(noise_level: jnp.ndarray, x: xarray.Dataset) -> Tuple[xarray.Dataset, xarray.Dataset]:
      """Computes D(x, sigma, y) and readout predictions."""
      bcast_noise_level = xarray_jax.DataArray(
          jnp.tile(noise_level, x.sizes['batch']), dims=('batch',))
      return self._denoiser(
          inputs=inputs,
          noisy_targets=x,
          noise_levels=bcast_noise_level,
          forcings=forcings)

    def body_fn(i: jnp.ndarray, val: Tuple[xarray.Dataset, Dict[int, xarray.Dataset]]) -> Tuple[xarray.Dataset, Dict[int, xarray.Dataset]]:
      x, readouts = val
      
      def init_noise(template):
        return noise_levels[0] * utils.spherical_white_noise_like(template)

      # maybe_init_noise = (i == 0).astype(noise_levels[0].dtype) # readout训练时一直使用的版本，08-26暂时comment掉了
      maybe_init_noise = jnp.asarray(i == 0, dtype=noise_levels.dtype) # 08-26 为了debug inference而进行的改动
      x = x + init_noise(x) * maybe_init_noise

      noise_level = noise_levels[i]

      if self._stochastic_churn:
        x, noise_level = utils.apply_stochastic_churn(
            x, noise_level,
            stochastic_churn_rate=per_step_churn_rates[i],
            noise_level_inflation_factor=self._noise_level_inflation_factor)

      next_noise_level = noise_levels[i + 1]
      mid_noise_level = jnp.sqrt(noise_level * next_noise_level)

      mid_over_current = mid_noise_level / noise_level

      # x_denoised = denoiser(noise_level, x, condition_1)

      x_denoised, readout1 = denoiser(noise_level, x)

      # Update readouts using tree_where to handle xarray structures
      for step in self._selected_denoising_step:
        is_selected = jnp.equal(i, jnp.array(step))
        readouts[step] = utils.tree_where(
            is_selected,
            readout1,
            readouts[step]
        )
      
      x_mid = mid_over_current * x + (1 - mid_over_current) * x_denoised

      next_over_current = next_noise_level / noise_level

      x_mid_denoised, readout2 = denoiser(mid_noise_level, x_mid)
      x_next = next_over_current * x + (1 - next_over_current) * x_mid_denoised

      final_x = utils.tree_where(next_noise_level == 0, x_denoised, x_next)
      jax.debug.print("********************* Here -111111   2m_temperature max: {x}", x=final_x["2m_temperature"].max())
      return (final_x, readouts)

    # Initialize state and pre-allocate readouts dictionary with empty templates
    noise_init = xarray.zeros_like(targets_template)
    readouts_init = {
      step: xarray.zeros_like(targets_template)
      for step in self._selected_denoising_step
    }


    # Run the loop and return both final state and selected readouts
    final_state, selected_readouts = hk.fori_loop(
        0, len(noise_levels) - 1, 
        body_fun=body_fn, 
        init_val=(noise_init, readouts_init))

    jax.debug.print("********************* Here 10   2m_temperature max: {x}", x=final_state["2m_temperature"].max())

    return final_state, selected_readouts





class Sampler(base.Sampler):
  """Sampling using DPM-Solver++ 2S from [1].

  This is combined with optional stochastic churn as described in [2].

  The '2S' terminology from [1] means that this is a second-order (2),
  single-step (S) solver. Here 'single-step' here distinguishes it from
  'multi-step' methods where the results of function evaluations from previous
  steps are reused in computing updates for subsequent steps. The solver still
  uses multiple steps though.

  [1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
  Models, https://arxiv.org/abs/2211.01095
  [2] Elucidating the Design Space of Diffusion-Based Generative Models,
  https://arxiv.org/abs/2206.00364
  """

  def __init__(self,
               denoiser: denoisers_base.Denoiser,
               max_noise_level: float,
               min_noise_level: float,
               num_noise_levels: int,
               rho: float,
               stochastic_churn_rate: float,
               churn_min_noise_level: float,
               churn_max_noise_level: float,
               noise_level_inflation_factor: float
               ):
    """Initializes the sampler.

    Args:
      denoiser: A Denoiser which predicts noise-free targets.
      max_noise_level: The highest noise level used at the start of the
        sequence of reverse diffusion steps.
      min_noise_level: The lowest noise level used at the end of the sequence of
        reverse diffusion steps.
      num_noise_levels: Determines the number of noise levels used and hence the
        number of reverse diffusion steps performed.
      rho: Parameter affecting the spacing of noise steps. Higher values will
        concentrate noise steps more around zero.
      stochastic_churn_rate: S_churn from the paper. This controls the rate
        at which noise is re-injected/'churned' during the sampling algorithm.
        If this is set to zero then we are performing deterministic sampling
        as described in Algorithm 1.
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
    """
    super().__init__(denoiser)
    self._noise_levels = utils.noise_schedule(
        max_noise_level, min_noise_level, num_noise_levels, rho)
    self._stochastic_churn = stochastic_churn_rate > 0
    self._per_step_churn_rates = utils.stochastic_churn_rate_schedule(
        self._noise_levels, stochastic_churn_rate, churn_min_noise_level,
        churn_max_noise_level)
    self._noise_level_inflation_factor = noise_level_inflation_factor

  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:

    dtype = casting.infer_floating_dtype(targets_template)  # pytype: disable=wrong-arg-types
    noise_levels = jnp.array(self._noise_levels).astype(dtype)
    per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype)

    def denoiser(noise_level: jnp.ndarray, x: xarray.Dataset) -> xarray.Dataset:
      """Computes D(x, sigma, y)."""
      bcast_noise_level = xarray_jax.DataArray(
          jnp.tile(noise_level, x.sizes['batch']), dims=('batch',))
      # Estimate the expectation of the fully-denoised target x0, conditional on
      # inputs/forcings, noisy targets and their noise level:
      return self._denoiser(
          inputs=inputs,
          noisy_targets=x,
          noise_levels=bcast_noise_level,
          forcings=forcings)

    def body_fn(i: jnp.ndarray, x: xarray.Dataset) -> xarray.Dataset:
      """One iteration of the sampling algorithm.

      Args:
        i: Sampling iteration.
        x: Noisy targets at iteration i, these will have noise level
          self._noise_levels[i].

      Returns:
        Noisy targets at the next lowest noise level self._noise_levels[i+1].
      """
      def init_noise(template):
        return noise_levels[0] * utils.spherical_white_noise_like(template)

      # Initialise the inputs if i == 0.
      # This is done here to ensure both noise sampler calls can use the same
      # spherical harmonic basis functions. While there may be a small compute
      # cost the memory savings can be significant.
      # TODO(dominicmasters): Figure out if we can merge the two noise sampler
      # calls into one to avoid this hack.
      maybe_init_noise = (i == 0).astype(noise_levels[0].dtype)
      x = x + init_noise(x) * maybe_init_noise

      noise_level = noise_levels[i]

      if self._stochastic_churn:
        # We increase the noise level of x a bit before taking it down again:
        x, noise_level = utils.apply_stochastic_churn(
            x, noise_level,
            stochastic_churn_rate=per_step_churn_rates[i],
            noise_level_inflation_factor=self._noise_level_inflation_factor)

      # Apply one step of the ODE solver to take x down to the next lowest
      # noise level.

      # Note that the Elucidating paper's choice of sigma(t)=t and s(t)=1
      # (corresponding to alpha(t)=1 in the DPM paper) as well as the standard
      # choice of r=1/2 (corresponding to a geometric mean for the s_i
      # midpoints) greatly simplifies the update from the DPM-Solver++ paper.
      # You need to do a bit of algebraic fiddling to arrive at the below after
      # substituting these choices into DPMSolver++'s Algorithm 1. The simpler
      # update we arrive at helps with intuition too.

      next_noise_level = noise_levels[i + 1]
      # This is s_{i+1} from the paper. They don't explain how the s_i are
      # chosen, but the default choice seems to be a geometric mean, which is
      # equivalent to setting all the r_i = 1/2.
      mid_noise_level = jnp.sqrt(noise_level * next_noise_level)

      mid_over_current = mid_noise_level / noise_level

      x_denoised = denoiser(noise_level, x)
      
      # jax.debug.print("##################### x_denoised 2m_temperature max: {max}", max=x_denoised['2m_temperature'].max())
      # This turns out to be a convex combination of current and denoised x,
      # which isn't entirely apparent from the paper formulae:
      x_mid = mid_over_current * x + (1 - mid_over_current) * x_denoised

      next_over_current = next_noise_level / noise_level
      x_mid_denoised = denoiser(mid_noise_level, x_mid)  # pytype: disable=wrong-arg-types
      x_next = next_over_current * x + (1 - next_over_current) * x_mid_denoised

      # For the final step to noise level 0, we do an Euler update which
      # corresponds to just returning the denoiser's prediction directly.
      #
      # In fact the behaviour above when next_noise_level == 0 is almost
      # equivalent, except that it runs the denoiser a second time to denoise
      # from noise level 0. The denoiser should just be the identity function in
      # this case, but it hasn't necessarily been trained at noise level 0 so
      # we avoid relying on this.

      
      return utils.tree_where(next_noise_level == 0, x_denoised, x_next)


    # Init with zeros but apply additional noise at step 0 to initialise the
    # state.
    noise_init = xarray.zeros_like(targets_template)
    res = hk.fori_loop(
        0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)

    return res





















# class Sampler_ReadOut_Training(base.Sampler):
#   """Sampling using DPM-Solver++ 2S from [1].

#   This is combined with optional stochastic churn as described in [2].

#   The '2S' terminology from [1] means that this is a second-order (2),
#   single-step (S) solver. Here 'single-step' here distinguishes it from
#   'multi-step' methods where the results of function evaluations from previous
#   steps are reused in computing updates for subsequent steps. The solver still
#   uses multiple steps though.

#   [1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
#   Models, https://arxiv.org/abs/2211.01095
#   [2] Elucidating the Design Space of Diffusion-Based Generative Models,
#   https://arxiv.org/abs/2206.00364
#   """

#   def __init__(self,
#                denoiser: denoisers_base.Denoiser,
#                max_noise_level: float,
#                min_noise_level: float,
#                num_noise_levels: int,
#                rho: float,
#                stochastic_churn_rate: float,
#                churn_min_noise_level: float,
#                churn_max_noise_level: float,
#                noise_level_inflation_factor: float,
#                Internal_feat_extract_denoise_step: int = 10
#                ):
#     """Initializes the sampler.

#     Args:
#       denoiser: A Denoiser which predicts noise-free targets.
#       max_noise_level: The highest noise level used at the start of the
#         sequence of reverse diffusion steps.
#       min_noise_level: The lowest noise level used at the end of the sequence of
#         reverse diffusion steps.
#       num_noise_levels: Determines the number of noise levels used and hence the
#         number of reverse diffusion steps performed.
#       rho: Parameter affecting the spacing of noise steps. Higher values will
#         concentrate noise steps more around zero.
#       stochastic_churn_rate: S_churn from the paper. This controls the rate
#         at which noise is re-injected/'churned' during the sampling algorithm.
#         If this is set to zero then we are performing deterministic sampling
#         as described in Algorithm 1.
#       churn_min_noise_level: Minimum noise level at which stochastic churn
#         occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
#       churn_max_noise_level: Maximum noise level at which stochastic churn
#         occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
#       noise_level_inflation_factor: This can be used to set the actual amount of
#         noise injected higher than what the denoiser is told has been added.
#         The motivation is to compensate for a tendency of L2-trained denoisers
#         to remove slightly too much noise / blur too much. S_noise from the
#         paper. Only used if stochastic_churn_rate > 0.
#       Internal_feat_extract_denoise_step: The noise level index (0-19) to use.
#     """
#     super().__init__(denoiser)
#     self._noise_levels = utils.noise_schedule(
#         max_noise_level, min_noise_level, num_noise_levels, rho)
#     self._stochastic_churn = stochastic_churn_rate > 0
#     self._per_step_churn_rates = utils.stochastic_churn_rate_schedule(
#         self._noise_levels, stochastic_churn_rate, churn_min_noise_level,
#         churn_max_noise_level)
#     self._noise_level_inflation_factor = noise_level_inflation_factor
#     self._extract_step = Internal_feat_extract_denoise_step

#   def __call__(
#       self,
#       inputs: xarray.Dataset,
#       targets_template: xarray.Dataset,
#       forcings: Optional[xarray.Dataset] = None,
#       **kwargs) -> xarray.Dataset:

#     dtype = casting.infer_floating_dtype(targets_template)
#     batch_size = inputs.sizes['batch']
    
#     # Sample noise levels using rho_inverse_cdf like in training
#     key = hk.next_rng_key()
#     random_values = jax.random.uniform(key, shape=(batch_size,), dtype=dtype)
#     # noise_levels = xarray_jax.DataArray(
#     #     data=utils.rho_inverse_cdf(
#     #         min_value=30,     # training_min_noise_level
#     #         max_value=36,    # training_max_noise_level
#     #         rho=7.0,         # training_noise_level_rho
#     #         cdf=random_values),
#     #     dims=('batch',))
    
#     noise_levels = xarray.DataArray(data=[20] * batch_size, dims=('batch',))
#     '''
#     noise_levels [8.00000000e+01 6.20812683e+01 4.77189827e+01 3.63043213e+01
#     2.73148670e+01 2.03050671e+01 1.48974152e+01 1.07743444e+01
#     7.67078066e+00 5.36734915e+00 3.68418932e+00 2.47535777e+00
#     1.62378585e+00 1.03676331e+00 6.41920626e-01 3.83680224e-01
#     2.20146134e-01 1.20404646e-01 6.22062944e-02 2.99999993e-02
#     0.00000000e+00]
#     '''


#     # force targets_template all be 0.1
#     # targets_template = xarray.full_like(targets_template, 0.1)
#     # Sample noise and apply it to targets like in training
#     noise = utils.spherical_white_noise_like(targets_template) * noise_levels
#     noisy_targets = targets_template + noise


#     # Perform denoising and get raw predictions
#     raw_predictions = self._denoiser(
#         inputs=inputs,
#         noisy_targets=noisy_targets,
#         noise_levels=noise_levels,
#         forcings=forcings)

#     # Return the raw predictions (which will be the graph and conditioning in ReadOut mode)
#     return raw_predictions


#       # def init_noise(template):
#       #   return noise_levels[0] * utils.spherical_white_noise_like(template)

#       # maybe_init_noise = (i == 0).astype(noise_levels[0].dtype)
#       # x = x + init_noise(x) * maybe_init_noise
#       # noise_level = noise_levels[i]
#       # if self._stochastic_churn:
#       #   # We increase the noise level of x a bit before taking it down again:
#       #   x, noise_level = utils.apply_stochastic_churn(
#       #       x, noise_level,
#       #       stochastic_churn_rate=per_step_churn_rates[i],
#       #       noise_level_inflation_factor=self._noise_level_inflation_factor)
#       # next_noise_level = noise_levels[i + 1]
#       # mid_noise_level = jnp.sqrt(noise_level * next_noise_level)

#       # mid_over_current = mid_noise_level / noise_level
#       # x_denoised = denoiser(noise_level, x)


