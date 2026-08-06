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
"""Data-structure for storing graphs with typed edges and nodes."""

from typing import NamedTuple, Any, Union, Tuple, Mapping, TypeVar

ArrayLike = Union[Any]  # np.ndarray, jnp.ndarray, tf.tensor
ArrayLikeTree = Union[Any, ArrayLike]  # Nest of ArrayLike

_T = TypeVar('_T')


# All tensors have a "flat_batch_axis", which is similar to the leading
# axes of graph_tuples:
# * In the case of nodes this is simply a shared node and flat batch axis, with
# size corresponding to the total number of nodes in the flattened batch.
# * In the case of edges this is simply a shared edge and flat batch axis, with
# size corresponding to the total number of edges in the flattened batch.
# * In the case of globals this is simply the number of graphs in the flattened
# batch.

# All shapes may also have any additional leading shape "batch_shape".
# Options for building batches are:
# * Use a provided "flatten" method that takes a leading `batch_shape` and
#   it into the flat_batch_axis (this will be useful when using `tf.Dataset`
#   which supports batching into RaggedTensors, with leading batch shape even
#   if graphs have different numbers of nodes and edges), so the RaggedBatches
#   can then be converted into something without ragged dimensions that jax can
#   use.
# * Directly build a "flat batch" using a provided function for batching a list
#   of graphs (how it is done in `jraph`).


class NodeSet(NamedTuple):
  """Represents a set of nodes."""
  n_node: ArrayLike  # [num_flat_graphs]
  features: ArrayLikeTree  # Prev. `nodes`: [num_flat_nodes] + feature_shape


class EdgesIndices(NamedTuple):
  """Represents indices to nodes adjacent to the edges."""
  senders: ArrayLike  # [num_flat_edges]
  receivers: ArrayLike  # [num_flat_edges]


class EdgeSet(NamedTuple):
  """Represents a set of edges."""
  n_edge: ArrayLike  # [num_flat_graphs]
  indices: EdgesIndices
  features: ArrayLikeTree  # Prev. `edges`: [num_flat_edges] + feature_shape


class Context(NamedTuple):
  # `n_graph` always contains ones but it is useful to query the leading shape
  # in case of graphs without any nodes or edges sets.
  n_graph: ArrayLike  # [num_flat_graphs]
  features: ArrayLikeTree  # Prev. `globals`: [num_flat_graphs] + feature_shape


class EdgeSetKey(NamedTuple):
  name: str   # Name of the EdgeSet.

  # Sender node set name and receiver node set name connected by the edge set.
  node_sets: Tuple[str, str]


class TypedGraph(NamedTuple):
  """A graph with typed nodes and edges.

  A typed graph is made of a context, multiple sets of nodes and multiple
  sets of edges connecting those nodes (as indicated by the EdgeSetKey).
  """

  context: Context
  nodes: Mapping[str, NodeSet]
  edges: Mapping[EdgeSetKey, EdgeSet]

  def edge_key_by_name(self, name: str) -> EdgeSetKey:
    found_key = [k for k in self.edges.keys() if k.name == name]
    if len(found_key) != 1:
      raise KeyError("invalid edge key '{}'. Available edges: [{}]".format(
          name, ', '.join(x.name for x in self.edges.keys())))
    return found_key[0]

  def edge_by_name(self, name: str) -> EdgeSet:
    return self.edges[self.edge_key_by_name(name)]
