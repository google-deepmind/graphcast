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
"""Wrappers that take care of casting."""

import contextlib
from typing import Any, Mapping, Tuple

import chex
from graphcast import predictor_base
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray


PyTree = Any


class Bfloat16Cast(predictor_base.Predictor):
  """Wrapper that casts all inputs to bfloat16 and outputs to targets dtype."""

  def __init__(self, predictor: predictor_base.Predictor, enabled: bool = True):
    """Inits the wrapper.

    Args:
      predictor: predictor being wrapped.
      enabled: disables the wrapper if False, for simpler hyperparameter scans.

    """
    self._enabled = enabled
    self._predictor = predictor

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> xarray.Dataset:
    if not self._enabled:
      return self._predictor(inputs, targets_template, forcings, **kwargs)

    with bfloat16_variable_view():
      predictions = self._predictor(
          *_all_inputs_to_bfloat16(inputs, targets_template, forcings),
          **kwargs,)

    predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types
    if predictions_dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets_template)  # pytype: disable=wrong-arg-types
    return tree_map_cast(
        predictions, input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    if not self._enabled:
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    with bfloat16_variable_view():
      loss, scalars = self._predictor.loss(
          *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs)

    if loss.dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types

    # Note that casting back the loss to e.g. float32 should not affect data
    # types of the backwards pass, because the first thing the backwards pass
    # should do is to go backwards the casting op and cast back to bfloat16
    # (and xprofs seem to confirm this).
    return tree_map_cast((loss, scalars),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    if not self._enabled:
      return self._predictor.loss_and_predictions(inputs, targets, forcings,  # pytype: disable=bad-return-type  # jax-ndarray
                                                  **kwargs)

    with bfloat16_variable_view():
      (loss, scalars), predictions = self._predictor.loss_and_predictions(
          *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs)

    if loss.dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types
    if predictions_dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    return tree_map_cast(((loss, scalars), predictions),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)


def infer_floating_dtype(data_vars: Mapping[str, chex.Array]) -> np.dtype:
  """Infers a floating dtype from an input mapping of data."""
  dtypes = {
      v.dtype
      for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
  if len(dtypes) != 1:
    dtypes_and_shapes = {
        k: (v.dtype, v.shape)
        for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
    raise ValueError(
        f'Did not found exactly one floating dtype {dtypes} in input variables:'
        f'{dtypes_and_shapes}')
  return list(dtypes)[0]


def _all_inputs_to_bfloat16(
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    forcings: xarray.Dataset,
    ) -> Tuple[xarray.Dataset,
               xarray.Dataset,
               xarray.Dataset]:
  return (inputs.astype(jnp.bfloat16),
          jax.tree.map(lambda x: x.astype(jnp.bfloat16), targets),
          forcings.astype(jnp.bfloat16))


def tree_map_cast(inputs: PyTree, input_dtype: np.dtype, output_dtype: np.dtype,
                  ) -> PyTree:
  def cast_fn(x):
    if x.dtype == input_dtype:
      return x.astype(output_dtype)
  return jax.tree.map(cast_fn, inputs)


@contextlib.contextmanager
def bfloat16_variable_view(enabled: bool = True):
  """Context for Haiku modules with float32 params, but bfloat16 activations.

  It works as follows:
  * Every time a variable is requested to be created/set as np.bfloat16,
    it will create an underlying float32 variable, instead.
  * Every time a variable a variable is requested as bfloat16, it will check the
    variable is of float32 type, and cast the variable to bfloat16.

  Note the gradients are still computed and accumulated as float32, because
  the params returned by init are float32, so the gradient function with
  respect to the params will already include an implicit casting to float32.

  Args:
    enabled: Only enables bfloat16 behavior if True.

  Yields:
    None
  """

  if enabled:
    with hk.custom_creator(
        _bfloat16_creator, state=True), hk.custom_getter(
            _bfloat16_getter, state=True), hk.custom_setter(
                _bfloat16_setter):
      yield
  else:
    yield


def _bfloat16_creator(next_creator, shape, dtype, init, context):
  """Creates float32 variables when bfloat16 is requested."""
  if context.original_dtype == jnp.bfloat16:
    dtype = jnp.float32
  return next_creator(shape, dtype, init)


def _bfloat16_getter(next_getter, value, context):
  """Casts float32 to bfloat16 when bfloat16 was originally requested."""
  if context.original_dtype == jnp.bfloat16:
    assert value.dtype == jnp.float32
    value = value.astype(jnp.bfloat16)
  return next_getter(value)


def _bfloat16_setter(next_setter, value, context):
  """Casts bfloat16 to float32 when bfloat16 was originally set."""
  if context.original_dtype == jnp.bfloat16:
    value = value.astype(jnp.float32)
  return next_setter(value)
