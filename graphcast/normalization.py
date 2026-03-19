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
"""Wrappers for Predictors which allow them to work with normalized data.

The Predictor which is wrapped sees normalized inputs and targets, and makes
normalized predictions. The wrapper handles translating the predictions back
to the original domain.
"""

import logging
from typing import Optional, Tuple, Union

from graphcast import predictor_base
from graphcast import xarray_tree
import xarray
from graphcast import typed_graph
import jax

def normalize(values: xarray.Dataset,
              scales: xarray.Dataset,
              locations: Optional[xarray.Dataset],
              ) -> xarray.Dataset:
  """Normalize variables using the given scales and (optionally) locations."""
  def normalize_array(array):
    if array.name is None:
      raise ValueError(
          "Can't look up normalization constants because array has no name.")
    if locations is not None:
      if array.name in locations:
        array = array - locations[array.name].astype(array.dtype)
      else:
        logging.warning('No normalization location found for %s', array.name)
    if array.name in scales:
      array = array / scales[array.name].astype(array.dtype)
    else:
      logging.warning('No normalization scale found for %s', array.name)
    return array
  return xarray_tree.map_structure(normalize_array, values)


def unnormalize(values: xarray.Dataset,
                scales: xarray.Dataset,
                locations: Optional[xarray.Dataset],
                ) -> xarray.Dataset:
  """Unnormalize variables using the given scales and (optionally) locations."""
  def unnormalize_array(array):
    if array.name is None:
      raise ValueError(
          "Can't look up normalization constants because array has no name.")
    if array.name in scales:
      array = array * scales[array.name].astype(array.dtype)
    else:
      logging.warning('No normalization scale found for %s', array.name)
    if locations is not None:
      if array.name in locations:
        array = array + locations[array.name].astype(array.dtype)
      else:
        logging.warning('No normalization location found for %s', array.name)
    return array
  return xarray_tree.map_structure(unnormalize_array, values)


class InputsAndResiduals(predictor_base.Predictor):
  """Wraps with a residual connection, normalizing inputs and target residuals.

  The inner predictor is given inputs that are normalized using `locations`
  and `scales` to roughly zero-mean unit variance.

  For target variables that are present in the inputs, the inner predictor is
  trained to predict residuals (target - last_frame_of_input) that have been
  normalized using `residual_scales` (and optionally `residual_locations`) to
  roughly unit variance / zero mean.

  This replaces `residual.Predictor` in the case where you want normalization
  that's based on the scales of the residuals.

  Since we return the underlying predictor's loss on the normalized residuals,
  if the underlying predictor is a sum of per-variable losses, the normalization
  will affect the relative weighting of the per-variable loss terms (hopefully
  in a good way).

  For target variables *not* present in the inputs, the inner predictor is
  trained to predict targets directly, that have been normalized in the same
  way as the inputs.

  The transforms applied to the targets (the residual connection and the
  normalization) are applied in reverse to the predictions before returning
  them.
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      stddev_by_level: xarray.Dataset,
      mean_by_level: xarray.Dataset,
      diffs_stddev_by_level: xarray.Dataset,
      ReadOut_flag: bool = False):
    self._predictor = predictor
    self._scales = stddev_by_level
    self._locations = mean_by_level
    self._residual_scales = diffs_stddev_by_level
    self._residual_locations = None
    self._ReadOut_flag = ReadOut_flag

    

  def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
    if norm_prediction.sizes.get('time') != 1:
      raise ValueError(
          'normalization.InputsAndResiduals only supports predicting a '
          'single timestep.')
    if norm_prediction.name in inputs:
      # Residuals are assumed to be predicted as normalized (unit variance),
      # but the scale and location they need mapping to is that of the residuals
      # not of the values themselves.
      prediction = unnormalize(
          norm_prediction, self._residual_scales, self._residual_locations)
      # A prediction for which we have a corresponding input -- we are
      # predicting the residual:
      last_input = inputs[norm_prediction.name].isel(time=-1)
      prediction = prediction + last_input
      return prediction
    else:
      # A predicted variable which is not an input variable. We are predicting
      # it directly, so unnormalize it directly to the target scale/location:
      return unnormalize(norm_prediction, self._scales, self._locations)

  def _subtract_input_and_normalize_target(self, inputs, target):
    if target.sizes.get('time') != 1:
      raise ValueError(
          'normalization.InputsAndResiduals only supports wrapping predictors'
          'that predict a single timestep.')
    if target.name in inputs:
      target_residual = target
      last_input = inputs[target.name].isel(time=-1)
      target_residual = target_residual - last_input
      return normalize(
          target_residual, self._residual_scales, self._residual_locations)
    else:
      return normalize(target, self._scales, self._locations)

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> Union[xarray.Dataset, typed_graph.TypedGraph]:


    # Always normalize inputs and forcings
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    
    # Get predictions from inner predictor
    output = self._predictor(
        norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
    
    unnormalized_output = xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        output)

    # Regular mode: denormalize predictions
    return unnormalized_output

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    """Returns the loss computed on normalized inputs and targets."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: self._subtract_input_and_normalize_target(inputs, t),
        targets)
    return self._predictor.loss(
        norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)

  def readout_train(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    """The loss computed on normalized data, with unnormalized predictions."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: self._subtract_input_and_normalize_target(inputs, t),
        targets)
    # 修改：现在返回 (loss, one_hot, weight_metrics), readout_pred, noise_levels
    (loss, one_hot, weight_metrics), norm_predictions, noise_levels = self._predictor.readout_train(
        norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
    # predictions = xarray_tree.map_structure(
    #     lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
    #     norm_predictions)
    predictions = norm_predictions
    # 修改：返回新的格式，保持 one_hot 和 weight_metrics
    return (loss, one_hot, weight_metrics), predictions, noise_levels


  def readout_inference_vis(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> Union[xarray.Dataset, typed_graph.TypedGraph]:

    # Always normalize inputs and forcings
    jax.debug.print("********************* -1   2m_temperature max: {x}", x=inputs["2m_temperature"].max())
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    
    jax.debug.print("********************* 00   2m_temperature max: {x}", x=norm_inputs["2m_temperature"].max())
    # Get predictions from inner predictor
    output, all_readouts = self._predictor.readout_inference_vis(
        norm_inputs, targets_template, forcings=norm_forcings, **kwargs)
    
    jax.debug.print("********************* 01   2m_temperature max: {x}", x=output["2m_temperature"].max())
    # Unnormalize the main output
    output_unnormalized = xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        output)
    
    # jax debug output_unnormalized max min of 2m_temperature
    jax.debug.print("********************* 02   2m_temperature max: {x}", x=output_unnormalized["2m_temperature"].max())
    
    # Unnormalize each readout prediction
    # all_readouts_unnorm = {}
    # for step in all_readouts.keys():
    #     all_readouts_unnorm[step] = xarray_tree.map_structure(
    #         lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
    #         all_readouts[step])
    all_readouts_unnorm = all_readouts
    
    # Return both unnormalized predictions
    return output_unnormalized, all_readouts_unnorm

  def readout_guided_inference_vis(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: xarray.Dataset,
      *,
      guidance_cfg,
      **kwargs,
  ):
    # 1) 归一化输入/强迫
    norm_inputs   = normalize(inputs,  self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)

    # 2) 走内部 guided readout 推理
    output_norm, all_readouts_norm, loss_history = self._predictor.readout_guided_inference_vis(
        norm_inputs, targets_template, forcings=norm_forcings,
        guidance_cfg=guidance_cfg, **kwargs
    )

    # 3) 主输出反归一化（与你现有 readout_inference_vis 保持一致）
    output_unnorm = xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        output_norm
    )

    # 备注：如果你希望 readout 也反归一化，可照主输出同样处理。
    all_readouts_unnorm = all_readouts_norm
    return output_unnorm, all_readouts_unnorm, loss_history


  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    """The loss computed on normalized data, with unnormalized predictions."""
    norm_inputs = normalize(inputs, self._scales, self._locations)
    norm_forcings = normalize(forcings, self._scales, self._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: self._subtract_input_and_normalize_target(inputs, t),
        targets)
    (loss, scalars), norm_predictions = self._predictor.loss_and_predictions(
        norm_inputs, norm_target_residuals, forcings=norm_forcings, **kwargs)
    predictions = xarray_tree.map_structure(
        lambda pred: self._unnormalize_prediction_and_add_input(inputs, pred),
        norm_predictions)
    return (loss, scalars), predictions
