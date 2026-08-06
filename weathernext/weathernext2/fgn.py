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

"""Predictor class for a Functional Generative Network (FGN) model."""

import dataclasses
import functools
from typing import Any, Callable, Mapping, Optional, Tuple

from weathernext.utils import casting as reader_utils
from weathernext.utils import losses
from weathernext.utils import predictor_base as base
from weathernext.utils import task as task_base
import haiku as hk
import jax
import jax.numpy as jnp
import xarray
import xarray_jax


@dataclasses.dataclass
class PredictorConfig:
  task: task_base.TaskConfig
  predictor_constructor: Callable[..., Any]
  predictor_kwargs: dict[str, Any]
  predictor_wrappers: list[dict[str, Any]]


def construct_predictor(config: PredictorConfig) -> Any:
  """Constructs a predictor from a PredictorConfig."""
  predictor = config.predictor_constructor(**config.predictor_kwargs)
  for w in config.predictor_wrappers:
    predictor = w["constructor"](predictor, **w.get("kwargs", {}))
  return predictor


@dataclasses.dataclass(frozen=True, eq=True)
class CheckPoint:
  params: dict[str, Any]
  description: str
  license: str


def gaussian_noise_generator(inputs, forcings, default_dtype, noise_var_dict):
  """Generates white gaussian noise for each variable.

  Args:
    inputs: xarray.Dataset with inputs to the model
    forcings: xarray.Dataset with forcings to the model
    default_dtype: type of the noise
    noise_var_dict: dict[str, dict] - key: name of the variable - value. a dict:
      * source -> input/forcings * non_batch_shape -> shape apart from batch
      dimension, given by source

  Returns:
    New inputs and forcings
  """
  inputs, forcings = inputs.copy(), forcings.copy()
  for var, config in noise_var_dict.items():
    assert config["source"] in ["input", "forcings"]
    source = inputs if config["source"] == "input" else forcings
    source[var] = xarray_jax.Variable(
        ("batch", *[s[0] for s in config["non_batch_shape"]]),
        jax.random.normal(
            hk.next_rng_key(),
            (source.sizes["batch"], *[s[1] for s in config["non_batch_shape"]]),
            dtype=config.get("dtype", default_dtype),
        ),
    )
  return inputs, forcings


class Predictor(base.Predictor):
  """Predictor for a Functional Generative Model."""

  def __init__(
      self,
      noisy_function_constructor: Callable[..., base.Predictor],
      noisy_function_kwargs: Mapping[str, Any],
      noise_generator_constructor: Optional[
          Callable[..., Tuple[xarray.Dataset, xarray.Dataset]]
      ] = None,
      noise_generator_kwargs: Optional[Mapping[str, Any]] = None,
      fgn_loss_function_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    """Constructs a Functional Generative Model predictor.

    Args:
      noisy_function_constructor:
      noisy_function_kwargs: Constructor and kwargs for a _deterministic_
        predictor implementing the underlying neural network used for fgn. Right
        now noise generation is generated here, the noisy_function takes noise
        within the forcings.
      noise_generator_constructor:
      noise_generator_kwargs: The noise generator adds one or more fields to
        inputs/forcings with pure noise samples. Most of the time these are
        input independent. They typically are just pure noise, but they can be
        more structured.
      fgn_loss_function_kwargs: Optional extra kwargs for fgn_loss_function.
    """
    self._noisy_function = noisy_function_constructor(**noisy_function_kwargs)
    self._fgn_loss_function_kwargs = dict(fgn_loss_function_kwargs or {})
    self._noise_generator = functools.partial(
        noise_generator_constructor, **noise_generator_kwargs  # pyrefly: ignore[bad-argument-type, bad-unpacking]
    )
    self._mae_function = losses.weighted_mae_per_level

  def run_model_with_sample_as_batch(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset],
      is_training: bool,
      add_noise: bool = True,
  ) -> xarray.Dataset:
    # Stack the batch and sample axis into a combined batch axis.
    stack_batch_and_sample = (
        lambda ds: ds.rename({"batch": "_batch"})
        .stack(batch=("_batch", "sample"))
        .transpose("batch", ...)
    )
    inputs = stack_batch_and_sample(inputs)
    if forcings is not None:
      forcings = stack_batch_and_sample(forcings)
    targets = stack_batch_and_sample(targets)
    if add_noise:
      inputs, forcings = self._noise_generator(
          inputs,
          forcings,
          default_dtype=reader_utils.infer_floating_dtype(inputs),  # pytype: disable=wrong-arg-types
      )
    preds = self._noisy_function(
        inputs=inputs,
        targets_template=targets,
        forcings=forcings,  # pyrefly: ignore[bad-argument-type]
        is_training=is_training,
    )
    # Unstack the combined batch axis into separate batch and sample axes.
    preds = (
        preds.unstack("batch")
        .rename({"_batch": "batch"})
        .transpose("sample", "batch", ...)
    )
    # Match the coordinate structure with the inputs.
    preds = preds.drop_indexes(["sample", "batch"]).reset_coords(
        ["sample", "batch"], drop=True
    )
    return preds

  def crps_loss(
      self,
      all_predictions: xarray.Dataset,
      targets: xarray.Dataset,
      unbiased: bool = True,
      **per_sample_loss_kwargs,
  ) -> base.LossAndDiagnostics:

    ensemble_size = all_predictions.sizes["sample"]

    # This shift is to ensure that we are computing the correct CRPS loss when
    # the predictions and targets are residuals (as well as when they are full
    # states). Specifically, our CRPS loss is defined for *fixed targets*, not
    # when the targets are samples from a target distribution. However,
    # *residual* targets will still vary from by sample during AR rollout or
    # when initialised from ensemble ICs, and so we need to account for this.
    # The MAE terms of CRPS are unaffected by the shift, as are the MAD terms
    # when the model predicts the full state. But when all_predictions are
    # residuals, the MAD terms after the shift recover the MAD between
    # predicted weather states, rather than MAD of residuals from different
    # starting points.
    # The shift currently implemented is to subtract
    # (targets - targets.mean('sample')), thus fixing the shifted targets to
    # be the mean targets. One could also shift the predictions and targets by
    # subtracting targets, which would fix the shifted targets to be 0, though
    # in this case it is important that the loss does not depend on the scale
    # of the targets.

    shift = targets - targets.mean("sample")
    shifted_targets = targets - shift
    shifted_predictions = all_predictions - shift

    # We ensure predictions are nan whenever targets are nan. This does not
    # affect the sample-to-target absolute difference, but does affect the
    # sample-to-sample absolute difference. It ensures that any grid points that
    # are nan in the target do not contribute to the ensemble spread term of the
    # loss. Otherwise, the loss can be hacked by increasing the sample variance
    # arbitrarily for the grid points which contain nans.
    shifted_predictions = shifted_predictions.where(
        shifted_targets.notnull(), jnp.nan
    )

    sample_loss = functools.partial(
        self._mae_function, **per_sample_loss_kwargs
    )
    # Diagnostics are per-variable losses. So we follow the computations twice.
    sum_abs_err, sum_abs_err_diagnostics = 0.0, 0.0
    sum_abs_diff, sum_abs_diff_diagnostics = 0.0, 0.0
    for i in range(ensemble_size):
      sample_abs_err, sample_diagnostics_err = sample_loss(
          predictions=shifted_predictions.isel(sample=i),
          targets=shifted_targets,
      )
      sum_abs_err = sum_abs_err + sample_abs_err
      sum_abs_err_diagnostics += sample_diagnostics_err
      for j in range(i):
        this_sad, this_diagnostics_sad = sample_loss(
            predictions=shifted_predictions.isel(sample=j),
            targets=shifted_predictions.isel(sample=i),
        )
        sum_abs_diff = sum_abs_diff + this_sad
        sum_abs_diff_diagnostics += this_diagnostics_sad
    # The 0.5 factor already done by only counting j < i and dividing by n^2.
    # The unbiased version uses sum_diffs/(n(n-1)) for estimating the
    # mean absolute differences instead of sum_diffs/n^2.
    # pytype: disable=unsupported-operands
    if unbiased:
      half_mad = sum_abs_diff / (ensemble_size * (ensemble_size - 1.0))
      half_mad_diagnostics = sum_abs_diff_diagnostics / (
          ensemble_size * (ensemble_size - 1.0)
      )
    else:
      half_mad = sum_abs_diff / (ensemble_size**2)
      half_mad_diagnostics = sum_abs_diff_diagnostics / (ensemble_size**2)
    mae = sum_abs_err / ensemble_size
    mae_diagnostics = sum_abs_err_diagnostics / ensemble_size
    # CRPS = MeanAbsError - 0.5 * MeanAbsDiff
    crps_loss = mae - half_mad
    crps_loss_diagnostics = mae_diagnostics - half_mad_diagnostics
    return (crps_loss, crps_loss_diagnostics)  # pytype: disable=bad-return-type

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> tuple[base.LossAndDiagnostics, xarray.Dataset]:
    predictions = self(
        inputs,
        targets_template=targets,
        forcings=forcings,
        is_training=True,
    )
    loss, diagnostics = self.crps_loss(
        predictions, targets, **self._fgn_loss_function_kwargs)
    return (loss, diagnostics), predictions

  def loss(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> base.LossAndDiagnostics:
    (loss, diagnostics), _ = self.loss_and_predictions(
        inputs, targets, forcings
    )
    return loss, diagnostics

  def __call__(
      self,
      inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
      is_training: bool = False,  # used mostly for eval.
      **kwargs,
  ) -> xarray.Dataset:
    if "sample" in inputs.dims:
      preds = self.run_model_with_sample_as_batch(
          inputs,
          targets_template,
          forcings,
          add_noise=True,
          is_training=is_training,
      )
    else:
      inputs, forcings = self._noise_generator(
          inputs,
          forcings,
          default_dtype=reader_utils.infer_floating_dtype(inputs),  # pytype: disable=wrong-arg-types
      )
      preds = self._noisy_function(
          inputs=inputs,
          targets_template=targets_template,
          forcings=forcings,
          is_training=is_training,
          **kwargs,
      )
    return preds

