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

from typing import Optional

from graphcast import casting
from graphcast import denoisers_base
from graphcast import samplers_base as base
from graphcast import samplers_utils as utils
from graphcast import xarray_jax
import haiku as hk
import jax.numpy as jnp
import xarray


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
    return hk.fori_loop(
        0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)
