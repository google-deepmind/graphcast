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
"""Denoising diffusion models based on the framework of [1].

Throughout we will refer to notation and equations from [1].

  [1] Elucidating the Design Space of Diffusion-Based Generative Models
  Karras, Aittala, Aila and Laine, 2022
  https://arxiv.org/abs/2206.00364
"""

import dataclasses
from typing import Any, List, Optional, Tuple, Union

import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import xarray

from graphcast import (
    casting,
    denoiser,
    dpm_solver_plus_plus_2s,
    graphcast,
    losses,
    predictor_base,
    samplers_utils,
    typed_graph,
    xarray_jax,
)

TARGET_SURFACE_VARS = (
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',  # GenCast predicts in 12hr timesteps.
    'total_precipitation_12hr',
    'sea_surface_temperature',
)

TARGET_SURFACE_NO_PRECIP_VARS = (
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    'sea_surface_temperature',
)


TASK = graphcast.TaskConfig(
    input_variables=(
        # GenCast doesn't take precipitation as input.
        TARGET_SURFACE_NO_PRECIP_VARS
        + graphcast.TARGET_ATMOSPHERIC_VARS
        + graphcast.GENERATED_FORCING_VARS
        + graphcast.STATIC_VARS
    ),
    target_variables=TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS,
    # GenCast doesn't take incident solar radiation as a forcing.
    forcing_variables=graphcast.GENERATED_FORCING_VARS,
    pressure_levels=graphcast.PRESSURE_LEVELS_WEATHERBENCH_13,
    # GenCast takes the current frame and the frame 12 hours prior.
    input_duration='24h',
)

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ReadOutGuidanceConfig:
    # 旧字段
    steps: Sequence[int] = field(default_factory=list)
    strength: float = 0.0
    normalize_grad: bool = False
    eps: float = 1e-8

    # —— latent optimization 超参 ——
    inner_opt_step_idxs: Sequence[int] = field(default_factory=list)  # 在哪些 step 做优化
    inner_opt_steps_map: dict = field(default_factory=dict)  # 新增：{step: num_opt_steps}
    max_opt_steps: int = 1                                   # 新增：最大优化次数（用于 JAX 编译）
    inner_opt_lr: float = 1e-2                               # 学习率
    loss_type: str = "xt_l2"     # "readout_l2" | "xt_l2"

    # 目标 readout（必须提供，若用 readout_l2）
    target_readout: Optional[xarray.Dataset] = None
    
    # —— Local Affine Warp 配置 ——
    warp_configs: Optional[dict] = None  # 新增：warp 配置字典


@dataclass
class ReadOutGuidanceConfigWithMask(ReadOutGuidanceConfig):
    """扩展 ReadOutGuidanceConfig，添加 guidance_mask 支持"""
    guidance_mask: Optional[jnp.ndarray] = None  # (B, H, W) binary mask




@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
  """Configures the sampler used to draw samples from GenCast.

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
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
  """
  max_noise_level: float = 80.
  min_noise_level: float = 0.03
  num_noise_levels: int = 20
  rho: float = 7.
  # Stochastic sampler settings.
  stochastic_churn_rate: float = 2.5
  churn_min_noise_level: float = 0.75
  churn_max_noise_level: float = float('inf')
  noise_level_inflation_factor: float = 1.05

@chex.dataclass(frozen=True, eq=True)
class ReadOutSamplerConfig(SamplerConfig):
    selected_denoising_step: List[int] = dataclasses.field(
        default_factory=lambda: [10, 15]  # Default steps to collect readout predictions
    )

@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
  training_noise_level_rho: float = 7.0
  training_max_noise_level: float = 88.0
  training_min_noise_level: float = 0.02
@chex.dataclass(frozen=True, eq=True)
class ReadOutNoiseConfig(NoiseConfig):
    ReadOut_flag: bool = True  # Example new hyperparameter


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  description: str
  license: str
  params: dict[str, Any]
  task_config: graphcast.TaskConfig
  denoiser_architecture_config: denoiser.DenoiserArchitectureConfig
  sampler_config: SamplerConfig
  noise_config: NoiseConfig
  noise_encoder_config: denoiser.NoiseEncoderConfig


class GenCast(predictor_base.Predictor):
  """Predictor for a denoising diffusion model following the framework of [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364

  Unlike the paper, we have a conditional model and our denoising function
  conditions on previous timesteps.

  As the paper demonstrates, the sampling algorithm can be varied independently
  of the denoising model and its training procedure, and it is separately
  configurable here.
  """

  def __init__(
      self,
      task_config: graphcast.TaskConfig,
      denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
      sampler_config: Optional[SamplerConfig] = None,
      noise_config: Optional[NoiseConfig] = None,
      noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None,
  ):
    """Constructs GenCast."""
    self.ReadOut_flag = denoiser_architecture_config["ReadOut_flag"]

    # Output size depends on number of variables being predicted.
    num_surface_vars = len(
        set(task_config.target_variables)
        - set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_atmospheric_vars = len(
        set(task_config.target_variables)
        & set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_outputs = (
        num_surface_vars
        + len(task_config.pressure_levels) * num_atmospheric_vars
    )
    denoiser_architecture_config.node_output_size = num_outputs

    self._denoiser = denoiser.Denoiser(
        noise_encoder_config,
        denoiser_architecture_config,
    )
    self._sampler_config = sampler_config
    # Singleton to avoid re-initializing the sampler for each inference call.
    self._sampler = None
    self._noise_config = noise_config

  def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the noisy targets input to the underlying network."""
    return (noise_scale**2 + 1)**-0.5

  def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the underlying network's raw outputs."""
    return noise_scale * (noise_scale**2 + 1)**-0.5

  def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the skip connection."""
    return 1 / (noise_scale**2 + 1)

  def _loss_weighting(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    r"""The loss weighting \lambda(\sigma) from the paper."""
    return self._c_out(noise_scale) ** -2

  def _preconditioned_denoiser(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> Union[xarray.Dataset, Tuple[typed_graph.TypedGraph, chex.Array]]:
    """The preconditioned denoising function D from the paper (Eqn 7)."""
    raw_predictions, readout_predictions = self._denoiser(
        inputs=inputs,
        noisy_targets=noisy_targets * self._c_in(noise_levels),
        noise_levels=noise_levels,
        forcings=forcings,
        **kwargs)
    if self.ReadOut_flag:
        # In ReadOut mode, this returns the readout predictions
        raw_predictions_res = raw_predictions * self._c_out(noise_levels) + noisy_targets * self._c_skip(noise_levels)
        return raw_predictions_res, readout_predictions  
    else:
        # Normal mode: apply preconditioning
        return (raw_predictions * self._c_out(noise_levels) +
              noisy_targets * self._c_skip(noise_levels))

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

  def readout_train(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    
    one_hot = kwargs.pop("one_hot", None)   # 没传就得到 None
    # 提取 loss 相关的配置参数
    pos_weight = kwargs.pop("pos_weight", 20.0)
    use_dynamic_weight = kwargs.pop("use_dynamic_weight", True)
    dynamic_weight_scale = kwargs.pop("dynamic_weight_scale", 0.1)
    min_pos_weight = kwargs.pop("min_pos_weight", 1.0)
    max_pos_weight = kwargs.pop("max_pos_weight", 1000.0)

    if self._noise_config is None:
      raise ValueError('Noise config must be specified to train GenCast.')

    # Sample noise levels:
    dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    key = hk.next_rng_key()  # Get key from Haiku
    key2 = jax.random.fold_in(key, 0)  # Create a new key for noise sampling
    batch_size = inputs.sizes['batch']
    noise_levels = xarray_jax.DataArray(
        data=samplers_utils.rho_inverse_cdf(
            min_value=self._noise_config.training_min_noise_level,
            # min_value=70,
            max_value=self._noise_config.training_max_noise_level,
            rho=self._noise_config.training_noise_level_rho,
            cdf=jax.random.uniform(key2, shape=(batch_size,), dtype=dtype)),
        dims=('batch',))

    # Sample noise and apply it to targets:
    key3 = jax.random.fold_in(key, 1)  # Create a new key for noise generation
    noise = (
        samplers_utils.spherical_white_noise_like(targets) * noise_levels
    )
    noisy_targets = targets + noise

    # Readout-Gencast - Read Internal Feature + Readout Estimation
    ori_pred, readout_pred = self._preconditioned_denoiser(
        inputs, noisy_targets, noise_levels, forcings)
    

    # 0. Debug Readout itself
    # Readout-Gencast - Compute Loss
    # loss, diagnostics = losses.weighted_mse_per_level(
    #     readout_pred,
    #     targets,
    #     # Weights are same as we used for GraphCast.
    #     per_variable_weights={
    #         # Any variables not specified here are weighted as 1.0.
    #         # A single-level variable, but an important headline variable
    #         # and also one which we have struggled to get good performance
    #         # on at short lead times, so leaving it weighted at 1.0, equal
    #         # to the multi-level variables:
    #         '2m_temperature': 1.0,
    #         # New single-level variables, which we don't weight too highly
    #         # to avoid hurting performance on other variables.
    #         '10m_u_component_of_wind': 0.1,
    #         '10m_v_component_of_wind': 0.1,
    #         'mean_sea_level_pressure': 0.1,
    #         'sea_surface_temperature': 0.1,
    #         'total_precipitation_12hr': 0.1,
    #         'total_precipitation_12hr': 100000,
    #     },
    # )
    # loss *= self._loss_weighting(noise_levels)
    # return (loss, diagnostics), readout_pred, noise_levels

    # 1. Implement binary mask and loss for debugging
    # loss, one_hot = losses.debug_two_channel_crossentropy_optax(
    #     readout_pred,targets)
    # return (loss, one_hot), readout_pred, noise_levels

    loss, one_hot, weight_metrics = losses.two_channel_crossentropy_optax(
        readout_pred, targets, one_hot,
        pos_weight=pos_weight,
        use_dynamic_weight=use_dynamic_weight,
        dynamic_weight_scale=dynamic_weight_scale,
        min_pos_weight=min_pos_weight,
        max_pos_weight=max_pos_weight,
    )
    # 将指标添加到 diagnostics 中
    diagnostics = xarray.Dataset({
        'weight_metrics': xarray.DataArray(0)  # 占位符，实际指标在 weight_metrics 字典中
    })
    # 将 weight_metrics 作为额外信息返回（通过修改返回值结构）
    return (loss, one_hot, weight_metrics), readout_pred, noise_levels
    
    # loss = losses.debug_center_mse_optax(
    #     readout_pred, targets)
    
    # return (loss, None), readout_pred, noise_levels


  def readout_inference_vis(self, inputs, targets_template, forcings, **kwargs):
    # Conventional GenCast
    self._sampler = dpm_solver_plus_plus_2s.Sampler_ReadOut(
      self._preconditioned_denoiser, **self._sampler_config
    )
    predictions, all_readouts = self._sampler(inputs, targets_template, forcings, **kwargs)
    return predictions, all_readouts


  ########### Readout Guidance Optimization
  def readout_guided_inference_vis(self, inputs, targets_template, forcings, *, guidance_cfg):

      # 支持两种配置类
      if isinstance(guidance_cfg, ReadOutGuidanceConfigWithMask):
          cfg = guidance_cfg
      elif isinstance(guidance_cfg, ReadOutGuidanceConfig):
          cfg = guidance_cfg
      elif isinstance(guidance_cfg, dict):
          # 根据是否有 guidance_mask 字段选择配置类
          if "guidance_mask" in guidance_cfg and guidance_cfg["guidance_mask"] is not None:
              cfg = ReadOutGuidanceConfigWithMask(**guidance_cfg)
          else:
              # 如果字典中有 guidance_mask 但值为 None，需要先移除它
              cfg_dict = {k: v for k, v in guidance_cfg.items() if k != "guidance_mask"}
              cfg = ReadOutGuidanceConfig(**cfg_dict)
      else:
          cfg = ReadOutGuidanceConfig(**guidance_cfg)

      # 构建 sampler
      self._sampler = dpm_solver_plus_plus_2s.Sampler_ReadOut(
          self._preconditioned_denoiser, **self._sampler_config
      )

      # —— 新路径：暂停第 cfg.inner_opt_step_idx 步做内层优化 ——
      if cfg.loss_type == "readout_l2" and cfg.target_readout is None:
          raise ValueError("latent_opt(readout_l2) 需要提供 cfg.target_readout (xarray.Dataset)")

      # 提取 guidance_mask（如果存在）
      guidance_mask = None
      if isinstance(cfg, ReadOutGuidanceConfigWithMask):
          guidance_mask = cfg.guidance_mask

      predictions, readouts, loss_history = self._sampler.guided(
          inputs=inputs,
          targets_template=targets_template,
          forcings=forcings,
          # 下面是新 guided(...) 的参数
          inner_opt_step_idxs=cfg.inner_opt_step_idxs,      # which denoising steps to pause
          inner_opt_steps_map=cfg.inner_opt_steps_map,      # 新增：per-step optimization 次数
          max_opt_steps=cfg.max_opt_steps,                  # 新增：最大次数（用于 JAX 编译）
          inner_opt_lr=cfg.inner_opt_lr,
          loss_type=cfg.loss_type,
          target_readout_tpl=cfg.target_readout,            # None 时 guided 会走 xt_l2
          guidance_mask=guidance_mask,                      # 新增：选择性 mask
          warp_configs=cfg.warp_configs,                    # 新增：warp 配置
      )
      return predictions, readouts, loss_history


  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           ) -> predictor_base.LossAndDiagnostics:

    if self._noise_config is None:
      raise ValueError('Noise config must be specified to train GenCast.')

    # Sample noise levels:
    dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    key = hk.next_rng_key()
    batch_size = inputs.sizes['batch']
    # noise_levels = xarray_jax.DataArray(
    #     data=samplers_utils.rho_inverse_cdf(
    #         min_value=self._noise_config.training_min_noise_level,
    #         max_value=self._noise_config.training_max_noise_level,
    #         rho=self._noise_config.training_noise_level_rho,
    #         cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype)),
    #     dims=('batch',))
    # jax.debug.print("noise_levels: {noise_levels}", noise_levels=noise_levels)
    # jax.debug.print("============^^^^^===")
    # jax.debug.print("self._noise_config: {noise_config}", noise_config=self._noise_config)
    # jax.debug.print("============^^^^^===")
    # jax.debug.breakpoint()
    
    # for noise_levels be 20.09941
    # noise_levels = xarray.DataArray(data=[20.09941] * batch_size, dims=('batch',))
    noise_levels = xarray.DataArray(data=[20] * batch_size, dims=('batch',))

    # Sample noise and apply it to targets:
    noise = (
        samplers_utils.spherical_white_noise_like(targets) * noise_levels
    )
    noisy_targets = targets + noise

    denoised_predictions = self._preconditioned_denoiser(
        inputs, noisy_targets, noise_levels, forcings)

    loss, diagnostics = losses.weighted_mse_per_level(
        denoised_predictions,
        targets,
        # Weights are same as we used for GraphCast.
        per_variable_weights={
            # Any variables not specified here are weighted as 1.0.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            '2m_temperature': 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            '10m_u_component_of_wind': 0.1,
            '10m_v_component_of_wind': 0.1,
            'mean_sea_level_pressure': 0.1,
            'sea_surface_temperature': 0.1,
            'total_precipitation_12hr': 0.1
        },
    )
    loss *= self._loss_weighting(noise_levels)
    return loss, diagnostics


  def __call__(self, inputs, targets_template, forcings, **kwargs):
    # Conventional GenCast
    self._sampler = dpm_solver_plus_plus_2s.Sampler(
      self._preconditioned_denoiser, **self._sampler_config
    )
    res = self._sampler(inputs, targets_template, forcings, **kwargs)

    return res

  # def __call__(
  #     self,
  #     inputs: xarray.Dataset,
  #     targets_template: xarray.Dataset,
  #     forcings: Optional[xarray.Dataset] = None,
  #     **kwargs
  # ) -> xarray.Dataset:
    
  #   # Normalization Checked - Readout-Gencast and Original-Gencast are the same

  #   """Runs the predictor on a batch of inputs."""
  #   if self._sampler is None:
  #       if self.ReadOut_flag:
  #           self._sampler = dpm_solver_plus_plus_2s.Sampler_ReadOut_Training(
  #               denoiser=self._preconditioned_denoiser,
  #               Internal_feat_extract_denoise_step=self.Internal_feat_extract_denoise_step,
  #               **dataclasses.asdict(self._sampler_config)
  #           )
  #       else:
  #           self._sampler = dpm_solver_plus_plus_2s.Sampler(
  #               denoiser=self._preconditioned_denoiser,
  #               **dataclasses.asdict(self._sampler_config)
  #           )
  #   res = self._sampler(inputs, targets_template, forcings, **kwargs)
  #   return res



'''
Original-Gencast
xarray_jax.JaxArrayWrapper(Array(-3.5215538, dtype=float32)), max=<xarray.DataArray '2m_temperature' ()> Size: 4B
xarray_jax.JaxArrayWrapper(Array(1.7475021, dtype=float32)), mean=<xarray.DataArray '2m_temperature' ()> Size: 4B
xarray_jax.JaxArrayWrapper(Array(-0.02801385, dtype=float32))

ReadOut-Gencast
xarray_jax.JaxArrayWrapper(Array(-3.5215538, dtype=float32)), max=<xarray.DataArray '2m_temperature' ()> Size: 4B
xarray_jax.JaxArrayWrapper(Array(1.7475021, dtype=float32)), mean=<xarray.DataArray '2m_temperature' ()> Size: 4B
xarray_jax.JaxArrayWrapper(Array(-0.02801385, dtype=float32))
'''