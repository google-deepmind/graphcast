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

"""Utilities for padding."""


import chex
import jax
import numpy as np


def get_num_padded_edges(
    num_edges: int,
    pad_edges_to_multiple_of: int = 1,
) -> int:
  """Returns the number of padded edges."""
  return round_up_to_multiple_of_int(num_edges, pad_edges_to_multiple_of)


def pad_edges(
    edge_features: chex.ArrayNumpyTree,
    senders: np.ndarray,
    receivers: np.ndarray,
    padded_size: int,
    padding_mode: str = "linearly_distributed",
    # Note that we cannot use np.nan as this may be propagated to non padding
    # nodes.
    feature_padding_value: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Pads edges to a multiple of pad_edges_to_multiple_of."""
  assert np.all(senders >= 0), "Senders should not be non-negative."
  assert np.all(receivers >= 0), "Receivers should not be non-negative."

  num_input_edges = jax.tree.map(
      lambda x: x.shape[0], jax.tree.flatten(edge_features)[0])
  assert np.all(np.array(num_input_edges) == num_input_edges[0])
  num_input_edges = num_input_edges[0]

  assert num_input_edges == len(senders) == len(receivers)
  assert num_input_edges <= padded_size
  if num_input_edges == padded_size:
    return edge_features, senders, receivers  # pyrefly: ignore[bad-return]

  # Get the locations of the indices _after_ padding.
  new_locations = get_indices_padding_locations(
      num_input_edges, padded_size, mode=padding_mode
  )
  # Initialise new arrays with all padding values.
  padded_receivers = np.full([padded_size], -1, dtype=np.int32)
  padded_senders = np.full([padded_size], -1, dtype=np.int32)
  padded_features = jax.tree.map(
      lambda x: np.ones(
          [padded_size, *x.shape[1:]], dtype=x.dtype) * feature_padding_value,
      edge_features)

  # Set the new locations to the original values.
  padded_receivers[new_locations] = receivers
  padded_senders[new_locations] = senders
  # np.put modifies in place.
  jax.tree.map(
      lambda x, y: np.put(x, new_locations, y), padded_features, edge_features
    )

  # Check non padding sender/receiver values are unchanged.
  assert np.all(
      padded_senders[padded_senders >= 0] == senders
  )
  assert np.all(
      padded_receivers[padded_receivers >= 0] == receivers
  )
  assert np.all(jax.tree.leaves(
      jax.tree.map(
          lambda x, y: np.all(x[padded_receivers >= 0] == y), padded_features, edge_features
      )
  ))
  return padded_features, padded_senders, padded_receivers


def round_up_to_multiple_of_int(x: int | float, round_value: int = 1) -> int:
  return int(np.ceil(float(x) / float(round_value)) * round_value)


def get_indices_padding_locations(
    len_before_padding: int,
    len_after_padding: int,
    mode: str = "ends",
) -> list[int]:
  """Pads indices to a given length.

  Args:
    len_before_padding: The length of current indices
    len_after_padding: The length to pad to.
    mode: The mode to pad. Two options are currently supported: 'ends' which
      pads equally at the beginning and end, and 'linearly_distributed' which
      equally space the padding throughout the array

  Returns:
    The locations of the original indices in the padded version.
  """
  indices_padding_amount = len_after_padding - len_before_padding
  assert indices_padding_amount >= 0
  if mode == "ends":
    # Pad the input at both the start and end of indices.
    top_indices_padding_amount = indices_padding_amount // 2
    bottom_indices_padding_amount = (
        indices_padding_amount - top_indices_padding_amount
    )
    new_locations = list(
        range(
            top_indices_padding_amount,
            len_after_padding - bottom_indices_padding_amount,
        )
    )
  elif mode == "linearly_distributed":
    # Linearly distribute the padding locations.
    padding_locations = list(
        np.linspace(0, len_after_padding - 1, indices_padding_amount).astype(
            np.int32
        )
    )
    assert len(padding_locations) == len(np.unique(padding_locations))
    new_locations = sorted(
        list(set(range(len_after_padding)) - set(padding_locations))
    )
  else:
    raise ValueError("Unknown mode: %s" % mode)

  assert len(set(new_locations)) == len_before_padding
  return new_locations

