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
"""Utils for training models in low precision."""

import functools
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp


# Wrappers for jax.lax.reduce_precision which is non-differentiable.
@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def reduce_precision(x, exponent_bits, mantissa_bits):
  return jax.tree_util.tree_map(
      lambda y: jax.lax.reduce_precision(y, exponent_bits, mantissa_bits), x)


def reduce_precision_fwd(x, exponent_bits, mantissa_bits):
  return reduce_precision(x, exponent_bits, mantissa_bits), None


def reduce_precision_bwd(exponent_bits, mantissa_bits, res, dout):
  del res  # Unused.
  return reduce_precision(dout, exponent_bits, mantissa_bits),


reduce_precision.defvjp(reduce_precision_fwd, reduce_precision_bwd)


def wrap_fn_for_upcast_downcast(inputs: Union[jnp.ndarray,
                                              Tuple[jnp.ndarray, ...]],
                                fn: Callable[[Union[jnp.ndarray,
                                                    Tuple[jnp.ndarray, ...]]],
                                             Union[jnp.ndarray,
                                                   Tuple[jnp.ndarray, ...]]],
                                f32_upcast: bool = True,
                                guard_against_excess_precision: bool = True
                                ) ->  Union[jnp.ndarray,
                                            Tuple[jnp.ndarray, ...]]:
  """Wraps `fn` to  upcast to float32 and then downcast, for use with BF16."""
  # Do not upcast if the inputs are already in float32.
  # This removes a no-op `jax.lax.reduce_precision` which is unsupported
  # in jax2tf at the moment.
  if isinstance(inputs, Tuple):
    f32_upcast = f32_upcast and inputs[0].dtype != jnp.float32
    orig_dtype = inputs[0].dtype
  else:
    f32_upcast = f32_upcast and inputs.dtype != jnp.float32
    orig_dtype = inputs.dtype

  if f32_upcast:
    inputs = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), inputs)

    if guard_against_excess_precision:
      # This is evil magic to guard against differences in precision in the QK
      # calculation between the forward pass and backwards pass. This is like
      # --xla_allow_excess_precision=false but scoped here.
      finfo = jnp.finfo(orig_dtype)  # jnp important!
      inputs = reduce_precision(inputs, finfo.nexp, finfo.nmant)

  output = fn(inputs)
  if f32_upcast:
    output = jax.tree_util.tree_map(lambda x: x.astype(orig_dtype), output)
  return output
