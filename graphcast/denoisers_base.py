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
"""Base class for Denoisers used in diffusion Predictors.

Denoisers are a bit like deterministic Predictors, except:
* Their __call__ method also conditions on noisy_targets and the noise_levels
  of those noisy targets
* They don't have an overrideable loss function (the loss is assumed to be some
  form of MSE and is implemented outside the Denoiser itself)
"""

from typing import Optional, Protocol

import xarray


class Denoiser(Protocol):
  """A denoising model that conditions on inputs as well as noise level."""

  def __call__(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    """Computes denoised targets from noisy targets.

    Args:
      inputs: Inputs to condition on, as for Predictor.__call__.
      noisy_targets: Targets which have had i.i.d. zero-mean Gaussian noise
        added to them (where the noise level used may vary along the 'batch'
        dimension).
      noise_levels: A DataArray with dimensions ('batch',) specifying the noise
        levels that were used for each example in the batch.
      forcings: Optional additional per-target-timestep forcings to condition
        on, as for Predictor.__call__.
      **kwargs: Any additional custom kwargs.

    Returns:
      Denoised predictions with the same shape as noisy_targets.
    """
