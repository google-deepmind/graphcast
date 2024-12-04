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
"""A Transformer model for weather predictions.

This model wraps the a transformer model and swaps the leading two axes of the
nodes in the input graph prior to evaluating the model to make it compatible
with a [nodes, batch, ...] ordering of the inputs.
"""

from typing import Any, Mapping, Optional

from graphcast import typed_graph
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse


Kwargs = Mapping[str, Any]


def _get_adj_matrix_for_edge_set(
    graph: typed_graph.TypedGraph,
    edge_set_name: str,
    add_self_edges: bool,
):
  """Returns the adjacency matrix for the given graph and edge set."""
  # Get nodes and edges of the graph.
  edge_set_key = graph.edge_key_by_name(edge_set_name)
  sender_node_set, receiver_node_set = edge_set_key.node_sets

  # Compute number of sender and receiver nodes.
  sender_n_node = graph.nodes[sender_node_set].n_node[0]
  receiver_n_node = graph.nodes[receiver_node_set].n_node[0]

  # Build adjacency matrix.
  adj_mat = sparse.csr_matrix((sender_n_node, receiver_n_node), dtype=np.bool_)
  edge_set = graph.edges[edge_set_key]
  s, r = edge_set.indices
  adj_mat[s, r] = True
  if add_self_edges:
    # Should only do this if we are certain the adjacency matrix is square.
    assert sender_node_set == receiver_node_set
    adj_mat[np.arange(sender_n_node), np.arange(receiver_n_node)] = True
  return adj_mat


class MeshTransformer(hk.Module):
  """A Transformer for inputs with ordering [nodes, batch, ...]."""

  def __init__(self,
               transformer_ctor,
               transformer_kwargs: Kwargs,
               name: Optional[str] = None):
    """Initialises the Transformer model.

    Args:
      transformer_ctor: Constructor for transformer.
      transformer_kwargs: Kwargs to pass to the transformer module.
      name: Optional name for haiku module.
    """
    super().__init__(name=name)
    # We defer the transformer initialisation to the first call to __call__,
    # where we can build the mask senders and receivers of the TypedGraph
    self._batch_first_transformer = None
    self._transformer_ctor = transformer_ctor
    self._transformer_kwargs = transformer_kwargs

  @hk.name_like('__init__')
  def _maybe_init_batch_first_transformer(self, x: typed_graph.TypedGraph):
    if self._batch_first_transformer is not None:
      return
    self._batch_first_transformer = self._transformer_ctor(
        adj_mat=_get_adj_matrix_for_edge_set(
            graph=x,
            edge_set_name='mesh',
            add_self_edges=True,
        ),
        **self._transformer_kwargs,
    )

  def __call__(
      self, x: typed_graph.TypedGraph,
      global_norm_conditioning: jax.Array
  ) -> typed_graph.TypedGraph:
    """Applies the model to the input graph and returns graph of same shape."""

    if set(x.nodes.keys()) != {'mesh_nodes'}:
      raise ValueError(
          f'Expected x.nodes to have key `mesh_nodes`, got {x.nodes.keys()}.'
      )
    features = x.nodes['mesh_nodes'].features
    if features.ndim != 3:  # pytype: disable=attribute-error  # jax-ndarray
      raise ValueError(
          'Expected `x.nodes["mesh_nodes"].features` to be 3, got'
          f' {features.ndim}.'
      )  # pytype: disable=attribute-error  # jax-ndarray

    # Initialise transformer and mask.
    self._maybe_init_batch_first_transformer(x)

    y = jnp.transpose(features, axes=[1, 0, 2])
    y = self._batch_first_transformer(y, global_norm_conditioning)
    y = jnp.transpose(y, axes=[1, 0, 2])
    x = x._replace(
        nodes={
            'mesh_nodes': x.nodes['mesh_nodes']._replace(
                features=y.astype(features.dtype)
            )
        }
    )
    return x
