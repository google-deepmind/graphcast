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

"""Utils for managing sharding."""

from collections.abc import Hashable, Mapping
from typing import Any

from absl import logging
import chex
from weathernext.utils import data_modalities
from weathernext.utils import sharding
from weathernext.utils import typed_graph
from weathernext.utils import update_blocks
import jax
import xarray as xr
import xarray_jax


Data = data_modalities.Data
CombinedArrays = data_modalities.CombinedArrays
SingleArray = data_modalities.SingleArray

# This `jax.debug.inspect_array_sharding` is known to be temperamental so
# we disable it by default with the expectation that it can be enabled for
# debugging purposes
DISABLE_INSPECT_SHARDING = True


def set_sharding(x, partition_spec: jax.sharding.PartitionSpec):
  if sharding.is_global_mesh_defined():
    return jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(
            x, partition_spec
        ),
        x,
    )
  else:
    return x


def inspect_sharding_if_available(
    x: Any,
    label: str = '',
) -> None:
  if sharding.is_global_mesh_defined() and not DISABLE_INSPECT_SHARDING:
    jax.debug.inspect_array_sharding(
        x, callback=lambda x: logging.info('%s: %s', label, x)
    )


def inspect_array_sharding(
    data: chex.Array, name: str = ''
) -> None:
  """Prints the sharding of the given array."""
  logging.info('%s: %s (%s)', name, data.shape, data.dtype)
  # Note that this has been disabled by default in global_prediction.utils. Set
  # utils.DISABLE_INSPECT_SHARDING to False to enable it.
  # Note we also log the shape and dtype with `inspect_sharding_if_available`
  # Since the callback is not called at the same time as the logging above. So
  # logging the shape and dtype here ensures the information about the shape and
  # dtype comes up in the same logging line as the sharding spec.
  inspect_sharding_if_available(
      data, label=f'{name} {data.shape}({data.dtype})')


def inspect_xarray_sharding(
    data: xr.Dataset | Mapping[Hashable, xr.DataArray], name: str = ''
) -> None:
  """Prints the sharding of each data array in the dataset."""
  for item_name, data_array in data.items():
    item_name = f'{name}.{item_name}{data_array.dims}'
    data = xarray_jax.unwrap(data_array.data)
    inspect_array_sharding(data, item_name)


def inspect_data_sharding(
    data: update_blocks.DataSingleOrCombinedArraysVar, name: str = ''
) -> None:
  """Prints the sharding of each data array in the SingleOrCombinedArraysVar."""
  (main, norm_conditioning, other_conditioning
   ) = update_blocks.separate_combined_arrays(data)
  if main is not None:
    item_name = f'{name}.main'
    inspect_array_sharding(main, item_name)  # pyrefly: ignore[bad-argument-type]
  if norm_conditioning is not None:
    item_name = f'{name}.norm_conditioning'
    inspect_array_sharding(norm_conditioning, item_name)  # pyrefly: ignore[bad-argument-type]
  if other_conditioning is not None:
    item_name = f'{name}.other_conditioning'
    inspect_array_sharding(other_conditioning, item_name)  # pyrefly: ignore[bad-argument-type]


def inspect_typed_graph_sharding(
    graph: typed_graph.TypedGraph, name: str = ''
) -> None:
  """Inspect the sharding of a TypedGraph."""
  # Note that the labels printed in inspect_array_sharding sometimes seem to
  # incorrectly adopt the previous value in the loop, the sharding spec printed
  # is still however correct.
  for node_key, node_set in graph.nodes.items():
    label = f'{name}.{node_key}.nodes'
    inspect_array_sharding(node_set.features, label)

  for edge_key, edge_set in graph.edges.items():
    label = f'{name}.{edge_key.name}'
    inspect_array_sharding(edge_set.features, f'{label}.edges')
    inspect_array_sharding(edge_set.indices.senders, f'{label}.senders')  # pytype: disable=attribute-error
    inspect_array_sharding(edge_set.indices.receivers, f'{label}.receivers')  # pytype: disable=attribute-error
