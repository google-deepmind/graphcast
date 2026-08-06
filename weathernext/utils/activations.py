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

"""Activation functions."""

from typing import Callable

import chex
import jax


def shifted_activation(
    x,
    *,
    activation_fn: Callable[[chex.Array], chex.Array],
    output_scale=None,
    input_offset=None,
    ):
  """Wraps an activation function with scaling options."""

  if input_offset is not None:
    x = x + input_offset
  x = activation_fn(x)

  if output_scale is not None:
    x = x * output_scale

  return x


# Wrapper to make it compatible with fiddle serialization.
def sigmoid(*args, **kwargs):
  return jax.nn.sigmoid(*args, **kwargs)
