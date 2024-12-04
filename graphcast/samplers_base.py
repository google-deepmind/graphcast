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
"""Base class for diffusion samplers."""

import abc
from typing import Optional

from graphcast import denoisers_base
import xarray


class Sampler(abc.ABC):
  """A sampling algorithm for a denoising diffusion model.

  This is constructed with a denoising function, and uses it to draw samples.
  """

  _denoiser: denoisers_base.Denoiser

  def __init__(self, denoiser: denoisers_base.Denoiser):
    """Constructs Sampler.

    Args:
      denoiser: A Denoiser which has been trained with an MSE loss to predict
        the noise-free targets.
    """
    self._denoiser = denoiser

  @abc.abstractmethod
  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    """Draws a sample using self._denoiser. Contract like Predictor.__call__."""
