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
"""Wrappers for Predictors which allow them to work with data cleaned of NaNs.

The Predictor which is wrapped sees inputs and targets without NaNs, and makes
NaNless predictions.
"""

from typing import Optional, Tuple

from graphcast import predictor_base as base
import numpy as np
import xarray


class NaNCleaner(base.Predictor):
  """A predictor wrapper than removes NaNs from ingested data.

  The Predictor which is wrapped sees inputs and targets without NaNs.
  """

  def __init__(
      self,
      predictor: base.Predictor,
      var_to_clean: str,
      fill_value: xarray.Dataset,
      reintroduce_nans: bool = False,
  ):
    """Initializes the NaNCleaner."""
    self._predictor = predictor
    self._fill_value = fill_value[var_to_clean]
    self._var_to_clean = var_to_clean
    self._reintroduce_nans = reintroduce_nans

  def _clean(self, dataset: xarray.Dataset) -> xarray.Dataset:
    """Cleans the dataset of NaNs."""
    data_array = dataset[self._var_to_clean]
    dataset = dataset.assign(
        {self._var_to_clean: data_array.fillna(self._fill_value)}
    )
    return dataset

  def _maybe_reintroduce_nans(
      self, stale_inputs: xarray.Dataset, predictions: xarray.Dataset
  ) -> xarray.Dataset:
    # NaN positions don't change between input frames, if they do then
    # we should be more careful about re-introducing them.
    if self._var_to_clean in predictions.keys():
      nan_mask = np.isnan(stale_inputs[self._var_to_clean]).any(dim='time')
      with_nan_values = predictions[self._var_to_clean].where(~nan_mask, np.nan)
      predictions = predictions.assign({self._var_to_clean: with_nan_values})
    return predictions

  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs,
  ) -> xarray.Dataset:
    if self._reintroduce_nans:
      # Copy inputs before cleaning so that we can reintroduce NaNs later.
      original_inputs = inputs.copy()
    if self._var_to_clean in inputs.keys():
      inputs = self._clean(inputs)
    if forcings and self._var_to_clean in forcings.keys():
      forcings = self._clean(forcings)
    predictions = self._predictor(
        inputs, targets_template, forcings, **kwargs
    )
    if self._reintroduce_nans:
      predictions = self._maybe_reintroduce_nans(original_inputs, predictions)
    return predictions

  def loss(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs,
  ) -> base.LossAndDiagnostics:
    if self._var_to_clean in inputs.keys():
      inputs = self._clean(inputs)
    if self._var_to_clean in targets.keys():
      targets = self._clean(targets)
    if forcings and self._var_to_clean in forcings.keys():
      forcings = self._clean(forcings)
    return self._predictor.loss(
        inputs, targets, forcings, **kwargs
    )

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs,
  ) -> Tuple[base.LossAndDiagnostics, xarray.Dataset]:
    if self._reintroduce_nans:
      # Copy inputs before cleaning so that we can reintroduce NaNs later.
      original_inputs = inputs.copy()
    if self._var_to_clean in inputs.keys():
      inputs = self._clean(inputs)
    if self._var_to_clean in targets.keys():
      targets = self._clean(targets)
    if forcings and self._var_to_clean in forcings.keys():
      forcings = self._clean(forcings)

    loss, predictions = self._predictor.loss_and_predictions(
        inputs, targets, forcings, **kwargs
    )
    if self._reintroduce_nans:
      predictions = self._maybe_reintroduce_nans(original_inputs, predictions)
    return loss, predictions
