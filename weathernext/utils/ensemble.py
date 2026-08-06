# Copyright 2026 Google LLC.
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
"""Ensemble prediction wrappers for FGN."""

from typing import Optional

from weathernext.utils import predictor_base as base
import xarray


class WithSampleDim(base.Predictor):
  """Adds a leading sample dimension broadcasting the data."""

  def __init__(self, underlying: base.Predictor, num_samples: int):
    self._underlying = underlying
    self._num_samples = num_samples

  def _add_sample_axis(self, data_array: xarray.DataArray) -> xarray.DataArray:
    return data_array.expand_dims(
        sample=self._num_samples,
        axis=0)

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: Optional[xarray.Dataset] = None,
               **kwargs) -> xarray.Dataset:
    inputs = inputs.map(self._add_sample_axis)
    targets_template = targets_template.map(self._add_sample_axis)
    if forcings is not None:
      forcings = forcings.map(self._add_sample_axis)
    return self._underlying(inputs, targets_template, forcings=forcings)  # pyrefly: ignore[bad-argument-type]

  def loss(self,  # pytype: disable=signature-mismatch  # jax-ndarray
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           **kwargs,
           ) -> base.LossAndDiagnostics:
    inputs = inputs.map(self._add_sample_axis)
    targets = targets.map(self._add_sample_axis)
    if forcings is not None:
      forcings = forcings.map(self._add_sample_axis)
    return self._underlying.loss(inputs, targets, forcings=forcings, **kwargs)  # pyrefly: ignore[bad-argument-type]
