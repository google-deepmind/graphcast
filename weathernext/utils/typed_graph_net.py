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
"""A library of typed Graph Neural Networks."""

import dataclasses
from typing import Any, Callable, Mapping, NamedTuple, Optional, Union

import chex
from weathernext.utils import typed_graph
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as np


# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = (
    jraph.ArrayTree)

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, Mapping[str, SenderFeatures], Mapping[str, ReceiverFeatures],
     Globals],
    NodeFeatures]

GNUpdateGlobalFn = Callable[
    [Mapping[str, NodeFeatures], Mapping[str, EdgeFeatures], Globals],
    Globals]


AggregationConstructor = Callable[..., jraph.AggregateEdgesToNodesFn]


class AggregateEdgesForNodesTuple(NamedTuple):
  senders: Optional[jraph.AggregateEdgesToNodesFn] = None
  receivers: Optional[jraph.AggregateEdgesToNodesFn] = None


@dataclasses.dataclass(frozen=True)
class AggregateEdgesForNodesConstructor:
  """Constructors for the sender and receiver gather functions.

  These can then be constructed in the model intself allowing additional
  information to be input.
  """

  senders: Optional[AggregationConstructor] = None
  receivers: Optional[AggregationConstructor] = None
  decorator: Callable[[Callable[..., Any]], Callable[..., Any]] = lambda x: x

  def construct_senders(self, *args, **kwargs):
    if self.senders is None:
      return None
    return self.decorator(self.senders(*args, **kwargs))

  def construct_receivers(self, *args, **kwargs):
    if self.receivers is None:
      return None
    return self.decorator(self.receivers(*args, **kwargs))


GatherFn = Callable[[jraph.NodeFeatures, jnp.ndarray], jraph.ArrayTree]

GatherConstructor = Callable[..., GatherFn]


class GatherNodesForEdgesTuple(NamedTuple):
  """Gather functions for both the senders and receivers."""
  senders: Optional[GatherFn] = None
  receivers: Optional[GatherFn] = None


@dataclasses.dataclass(frozen=True)
class GatherNodesForEdgesConstructor:
  """Constructors for the sender and receiver gather functions.

  These can then be constructed in the model intself allowing additional
  information to be input.
  """

  senders: Optional[GatherConstructor] = None
  receivers: Optional[GatherConstructor] = None
  decorator: Callable[[Callable[..., Any]], Callable[..., Any]] = lambda x: x

  def construct_senders(self, *args, **kwargs):
    if self.senders is None:
      return None
    return self.decorator(self.senders(*args, **kwargs))

  def construct_receivers(self, *args, **kwargs):
    if self.receivers is None:
      return None
    return self.decorator(self.receivers(*args, **kwargs))


GatherNodesForEdgesObj = Union[
    GatherFn,
    GatherNodesForEdgesTuple,
    GatherNodesForEdgesConstructor,
]

AggregateEdgesForNodesObj = Union[
    jraph.AggregateEdgesToNodesFn,
    AggregateEdgesForNodesTuple,
    AggregateEdgesForNodesConstructor,
]


def jax_gather(x: jnp.ndarray, indices: chex.Array) -> jnp.ndarray:
  return x[indices]


def _aggregate_edges_for_nodes_obj_to_tuple(
    fn_or_tuple: AggregateEdgesForNodesObj,
    sender_constructor_kwargs: Mapping[str, Any],
    receiver_constructor_kwargs: Mapping[str, Any],
    construct_sender_aggregator: bool = True,
    construct_receiver_aggregator: bool = True,
) -> AggregateEdgesForNodesTuple:
  """Transform a AggregateEdgesForNodesObj to a AggregateEdgesForNodesTuple."""
  if isinstance(fn_or_tuple, AggregateEdgesForNodesTuple):
    return fn_or_tuple
  elif isinstance(fn_or_tuple, AggregateEdgesForNodesConstructor):
    fn_or_tuple: AggregateEdgesForNodesConstructor

    if fn_or_tuple.senders is not None and construct_sender_aggregator:
      senders_aggregator = fn_or_tuple.construct_senders(
          **sender_constructor_kwargs
      )
    else:
      senders_aggregator = None

    if fn_or_tuple.receivers is not None and construct_receiver_aggregator:
      receivers_aggregator = fn_or_tuple.construct_receivers(
          **receiver_constructor_kwargs
      )
    else:
      receivers_aggregator = None

    return AggregateEdgesForNodesTuple(
        senders=senders_aggregator, receivers=receivers_aggregator
    )
  else:
    # must be of type models.AggregateEdgesToNodesFn
    return AggregateEdgesForNodesTuple(
        senders=fn_or_tuple, receivers=fn_or_tuple
    )


def _gather_nodes_for_edges_obj_to_tuple(
    fn_or_tuple: GatherNodesForEdgesObj,
    sender_constructor_kwargs: Mapping[str, Any],
    receiver_constructor_kwargs: Mapping[str, Any],
    construct_sender_gather: bool = True,
    construct_receiver_gather: bool = True,
) -> GatherNodesForEdgesTuple:
  """Transform a GatherNodesForEdgesObj to a GatherNodesForEdgesTuple."""
  if isinstance(fn_or_tuple, GatherNodesForEdgesTuple):
    return fn_or_tuple
  elif isinstance(fn_or_tuple, GatherNodesForEdgesConstructor):
    fn_or_tuple: GatherNodesForEdgesConstructor
    if fn_or_tuple.senders is not None and construct_sender_gather:
      sender_gather = fn_or_tuple.construct_senders(**sender_constructor_kwargs)
    else:
      sender_gather = None

    fn_or_tuple: GatherNodesForEdgesConstructor
    if fn_or_tuple.senders is not None and construct_receiver_gather:
      receiver_gather = fn_or_tuple.construct_receivers(
          **receiver_constructor_kwargs
      )
    else:
      receiver_gather = None
    return GatherNodesForEdgesTuple(
        senders=sender_gather, receivers=receiver_gather
    )
  else:
    # must be GatherFn type
    return GatherNodesForEdgesTuple(senders=fn_or_tuple, receivers=fn_or_tuple)


def GraphNetwork(  # pylint: disable=invalid-name
    update_edge_fn: Mapping[str, jraph.GNUpdateEdgeFn],
    update_node_fn: Mapping[str, GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    gather_nodes_for_edges_fn: GatherNodesForEdgesObj = jax_gather,  # pyrefly: ignore[bad-function-definition]
    aggregate_edges_for_nodes_fn: AggregateEdgesForNodesObj = jraph.segment_sum,  # pyrefly: ignore[bad-function-definition]
    aggregate_nodes_for_globals_fn: jraph.AggregateNodesToGlobalsFn = jraph.segment_sum,  # pyrefly: ignore[bad-function-definition]
    aggregate_edges_for_globals_fn: jraph.AggregateEdgesToGlobalsFn = jraph.segment_sum,  # pyrefly: ignore[bad-function-definition]
    pre_gather_senders_for_edges_fn: Optional[Mapping[str, Any]] = None,
    pre_gather_receivers_for_edges_fn: Optional[Mapping[str, Any]] = None,
    pre_gather_globals_for_edges_fn: Optional[Mapping[str, Any]] = None,
    pre_gather_edges_for_edges_fn: Optional[Mapping[str, Any]] = None,
    edge_update_remat_block_size: Optional[int] = None,
):
  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

  There is one difference. For the nodes update the class aggregates over the
  sender edges and receiver edges separately. This is a bit more general
  than the algorithm described in the paper. The original behaviour can be
  recovered by using only the receiver edge aggregations for the update.

  In addition this implementation supports softmax attention over incoming
  edge features.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: mapping of functions used to update a subset of the edge
      types, indexed by edge type name.
    update_node_fn: mapping of functions used to update a subset of the node
      types, indexed by node type name.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    gather_nodes_for_edges_fn: function used to gather nodes to edges. A simple
      function can be used which will be applied to both senders or receivers
      or, alternatively a GatherNodesForEdgesTuple or
      GatherNodesForEdgesConstructor. Thes opjects allow for the sender and
      receiver gathers to be specified indenpendantly and the construction of
      the function to be deferred till calltime.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node. A simple function can be used which will be applied to both senders
      or receivers or, alternatively a AggregateEdgesForNodesTuple or
      AggregateEdgesForNodesConstructor. Thes opjects allow for the sender and
      receiver aggregation to be specified indenpendantly and the construction
      of the function to be deferred till calltime.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    pre_gather_senders_for_edges_fn: mapping of functions applied to the senders
      before the gather, indexed by edge type name.
    pre_gather_receivers_for_edges_fn: mapping of functions applied to the
      receivers before the gather, indexed by edge type name.
    pre_gather_globals_for_edges_fn: mapping of functions applied to the globals
      before the gather, indexed by edge type name.
    pre_gather_edges_for_edges_fn: mapping of functions applied to the edges
      before the edge MLP, indexed by edge type name. "pre_gather" prefix is
      used here for consistency with the other "pre_gather" argument as they
      will typically be specified together.
    edge_update_remat_block_size: if not None, the edge update function will be
      computed in blocks of edges of this size using gradient rematerialization
      for each block.

  Returns:
    A method that applies the configured GraphNetwork.
  """

  def _apply_graph_net(graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `TypedGraph` containing the graph.

    Returns:
      Updated `TypedGraph`.
    """

    updated_graph = graph

    # Edge update.
    updated_edges = dict(updated_graph.edges)
    for edge_set_name, edge_fn in update_edge_fn.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      if pre_gather_senders_for_edges_fn:
        pre_gather_senders_fn = pre_gather_senders_for_edges_fn[edge_set_name]
      else:
        pre_gather_senders_fn = lambda x: x

      if pre_gather_receivers_for_edges_fn:
        pre_gather_receivers_fn = pre_gather_receivers_for_edges_fn[
            edge_set_name
        ]
      else:
        pre_gather_receivers_fn = lambda x: x

      if pre_gather_globals_for_edges_fn:
        pre_gather_globals_fn = pre_gather_globals_for_edges_fn[edge_set_name]
      else:
        pre_gather_globals_fn = lambda x: x

      if pre_gather_edges_for_edges_fn:
        pre_gather_edges_fn = pre_gather_edges_for_edges_fn[edge_set_name]
      else:
        pre_gather_edges_fn = lambda x: x
      updated_edges[edge_set_key] = _edge_update(
          updated_graph,
          edge_fn,
          edge_set_key,
          edge_update_remat_block_size,
          gather_nodes_for_edges_fn,
          pre_gather_senders_fn,
          pre_gather_receivers_fn,
          pre_gather_globals_fn,
          pre_gather_edges_fn,
      )
    updated_graph = updated_graph._replace(edges=updated_edges)

    # Node update.
    updated_nodes = dict(updated_graph.nodes)
    for node_set_key, node_fn in update_node_fn.items():
      updated_nodes[node_set_key] = _node_update(
          updated_graph, node_fn, node_set_key, aggregate_edges_for_nodes_fn)
    updated_graph = updated_graph._replace(nodes=updated_nodes)

    # Global update.
    if update_global_fn:
      updated_context = _global_update(
          updated_graph, update_global_fn,
          aggregate_edges_for_globals_fn,
          aggregate_nodes_for_globals_fn)
      updated_graph = updated_graph._replace(context=updated_context)

    return updated_graph

  return _apply_graph_net


def _tree_map_multi_output_into_list_of_trees(f, data_tree, num_outputs):
  # Assuming `f` is a function that returns multiple arguments
  # (e.g. `f(x) -> (y1, y2, y3)`).
  # And an input tree Tree[x] made of elements x.
  # Returns each of the outputs in a separate tree Tree[y1], Tree[y2], Tree[y3].
  tree_flat, tree_def = jax.tree.flatten(data_tree)
  if not tree_flat:
    return tuple([data_tree] * num_outputs)
  tree_flat_output = [f(leaf) for leaf in tree_flat]
  return tuple([
      jax.tree.unflatten(tree_def, tree_flat_output_i)
      for tree_flat_output_i in zip(*tree_flat_output)])


def _edge_update(
    graph,
    edge_fn,
    edge_set_key,
    edge_update_remat_block_size,
    gather_nodes_fn,
    pre_gather_senders_fn,
    pre_gather_receivers_fn,
    pre_gather_globals_fn,
    pre_gather_edges_fn,
):  # pylint: disable=invalid-name
  """Updates an edge set of a given key."""

  sender_nodes = graph.nodes[edge_set_key.node_sets[0]]
  receiver_nodes = graph.nodes[edge_set_key.node_sets[1]]
  edge_set = graph.edges[edge_set_key]
  senders = edge_set.indices.senders  # pytype: disable=attribute-error
  receivers = edge_set.indices.receivers  # pytype: disable=attribute-error

  n_edge = edge_set.n_edge
  sum_n_edge = senders.shape[0]

  # For maximum efficiency, in the case of `edge_update_remat_block_size`
  # technically we could broadcast the global features for each chunk, rather
  # broadcast everything, and then splitting, but for now, we keep it simple.
  global_features = tree.tree_map(
      lambda g: jnp.repeat(
          pre_gather_globals_fn(g),
          n_edge,
          axis=0,
          total_repeat_length=sum_n_edge,
      ),
      graph.context.features,
  )

  def update_edges(edges, senders, receivers, global_attributes):
    sender_feat_size = tree.tree_leaves(sender_nodes.features)[0].shape[0]
    receiver_feat_size = tree.tree_leaves(receiver_nodes.features)[0].shape[0]

    # Constructing the gather functions here has pros and cons, the
    # alternative would be to construct these before the GraphNetwork is called,
    # for example in the predictor, and construct a per node/edge type mapping
    # to the correct gather. Delaying the construction to this point makes
    # some aspects simpler, as the user just specifies a sender and receiver
    # aggregator and doesn't have to manage the node/edge types themselves.
    # However is does reduce the flexibility and mean multiple construction
    # calls need to be done for potentially identical functions - this could
    # have a small memory implication. If additinal flexibility is needed or the
    # memory costs are deemed critical this alternaitve approach should be
    # considered.
    gather_nodes_for_edges_tuple = _gather_nodes_for_edges_obj_to_tuple(
        gather_nodes_fn,
        sender_constructor_kwargs={
            'indices': senders,
            'num_input_rows': sender_feat_size,
        },
        receiver_constructor_kwargs={
            'indices': receivers,
            'num_input_rows': receiver_feat_size,
        },
    )

    if gather_nodes_for_edges_tuple.senders is not None:
      gather_senders = lambda data: gather_nodes_for_edges_tuple.senders(
          pre_gather_senders_fn(data), senders
      )
      sent_attributes = tree.tree_map(gather_senders, sender_nodes.features)
    else:
      sent_attributes = None

    if gather_nodes_for_edges_tuple.receivers is not None:
      gather_receivers = lambda data: gather_nodes_for_edges_tuple.receivers(
          pre_gather_receivers_fn(data), receivers
      )
      received_attributes = tree.tree_map(
          gather_receivers, receiver_nodes.features
      )
    else:
      received_attributes = None
    return edge_fn(
        pre_gather_edges_fn(edges),
        sent_attributes,
        received_attributes,
        global_attributes,
    )

  if edge_update_remat_block_size:
    update_edges_shard = hk.remat(update_edges)

    edge_update_remat_block_size = min(edge_update_remat_block_size, sum_n_edge)
    splits = np.arange(
        edge_update_remat_block_size, sum_n_edge, edge_update_remat_block_size)
    num_shards = len(splits) + 1
    edge_features = _tree_map_multi_output_into_list_of_trees(
        lambda e: jnp.split(e, splits, axis=0), edge_set.features, num_shards)
    global_features = _tree_map_multi_output_into_list_of_trees(
        lambda g: jnp.split(g, splits, axis=0), global_features, num_shards)
    senders = jnp.split(senders, splits, axis=0)
    receivers = jnp.split(receivers, splits, axis=0)

    n_chunks = len(senders)

    new_features = []
    for i in range(n_chunks):
      updated_edges = update_edges_shard(
          edge_features[i], senders[i], receivers[i], global_features[i]
      )
      new_features.append(updated_edges)
    new_features = jax.tree.map(
        lambda *x: jnp.concatenate(x, axis=0), *new_features
    )
  else:
    new_features = update_edges(
        edge_set.features, senders, receivers, global_features
    )
  return edge_set._replace(features=new_features)


def _node_update(graph, node_fn, node_set_key, aggregation_fn):  # pylint: disable=invalid-name
  """Updates an edge set of a given key."""
  node_set = graph.nodes[node_set_key]
  sum_n_node = tree.tree_leaves(node_set.features)[0].shape[0]

  sent_features = {}
  received_features = {}
  for edge_set_key, edge_set in graph.edges.items():
    senders = edge_set.indices.senders  # pytype: disable=attribute-error
    receivers = edge_set.indices.receivers  # pytype: disable=attribute-error
    sender_node_set_key = edge_set_key.node_sets[0]
    receiver_node_set_key = edge_set_key.node_sets[1]
    # Constructing the aggregation functions here has pros and cons, the
    # alternative would be to construct these before the GraphNetwork is called,
    # for example in the predictor, and construct a per node/edge type mapping
    # to the correct aggregator. Delaying the construction to this point makes
    # some aspects simpler, as the user just specifies a sender and receiver
    # aggregator and doesn't have to manage the node/edge types themselves.
    # However is does reduce the flexibility and mean multiple construction
    # calls need to be done for potentially identical functions - this could
    # have a small memory implication. If additinal flexibility is needed or the
    # memory costs are deemed critical this alternaitve approach should be
    # considered.
    aggregation_fn_tuple = _aggregate_edges_for_nodes_obj_to_tuple(
        aggregation_fn,
        sender_constructor_kwargs={
            'indices': senders,
            'num_output_rows': sum_n_node,
        },
        receiver_constructor_kwargs={
            'indices': receivers,
            'num_output_rows': sum_n_node,
        },
        construct_sender_aggregator=sender_node_set_key == node_set_key,
        construct_receiver_aggregator=receiver_node_set_key == node_set_key,
    )

    if sender_node_set_key == node_set_key:
      assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
      if aggregation_fn_tuple.senders is not None:
        sent_features[edge_set_key.name] = tree.tree_map(
            lambda e: aggregation_fn_tuple.senders(e, senders, sum_n_node),  # pylint: disable=cell-var-from-loop
            edge_set.features,
        )

    if receiver_node_set_key == node_set_key:
      assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
      if aggregation_fn_tuple.receivers is not None:
        received_features[edge_set_key.name] = tree.tree_map(
            lambda e: aggregation_fn_tuple.receivers(e, receivers, sum_n_node),  # pylint: disable=cell-var-from-loop
            edge_set.features,
        )

  n_node = node_set.n_node
  global_features = tree.tree_map(
      lambda g: jnp.repeat(g, n_node, axis=0, total_repeat_length=sum_n_node),
      graph.context.features)
  new_features = node_fn(
      node_set.features, sent_features, received_features, global_features)
  return node_set._replace(features=new_features)


def _global_update(graph, global_fn, edge_aggregation_fn, node_aggregation_fn):  # pylint: disable=invalid-name
  """Updates an edge set of a given key."""
  n_graph = graph.context.n_graph.shape[0]
  graph_idx = jnp.arange(n_graph)

  edge_features = {}
  for edge_set_key, edge_set in graph.edges.items():
    assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
    sum_n_edge = edge_set.indices.senders.shape[0]
    edge_gr_idx = jnp.repeat(
        graph_idx, edge_set.n_edge, axis=0, total_repeat_length=sum_n_edge)
    edge_features[edge_set_key.name] = tree.tree_map(
        lambda e: edge_aggregation_fn(e, edge_gr_idx, n_graph),   # pylint: disable=cell-var-from-loop
        edge_set.features)

  node_features = {}
  for node_set_key, node_set in graph.nodes.items():
    sum_n_node = tree.tree_leaves(node_set.features)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, node_set.n_node, axis=0, total_repeat_length=sum_n_node)
    node_features[node_set_key] = tree.tree_map(
        lambda n: node_aggregation_fn(n, node_gr_idx, n_graph),   # pylint: disable=cell-var-from-loop
        node_set.features)

  new_features = global_fn(node_features, edge_features, graph.context.features)
  return graph.context._replace(features=new_features)


InteractionUpdateNodeFn = Callable[
    [jraph.NodeFeatures,
     Mapping[str, SenderFeatures],
     Mapping[str, ReceiverFeatures]],
    jraph.NodeFeatures]


InteractionUpdateNodeFnNoSentEdges = Callable[
    [jraph.NodeFeatures,
     Mapping[str, ReceiverFeatures]],
    jraph.NodeFeatures]


def InteractionNetwork(  # pylint: disable=invalid-name
    update_edge_fn: Mapping[str, jraph.InteractionUpdateEdgeFn],
    update_node_fn: Mapping[
        str, Union[InteractionUpdateNodeFn, InteractionUpdateNodeFnNoSentEdges]
    ],
    gather_nodes_for_edges_fn: GatherNodesForEdgesObj = jax_gather,  # pyrefly: ignore[bad-function-definition]
    aggregate_edges_for_nodes_fn: AggregateEdgesForNodesObj = jraph.segment_sum,  # pyrefly: ignore[bad-function-definition]
    include_sent_messages_in_node_update: bool = False,
    **graph_network_kwargs,
):
  """Returns a method that applies a configured InteractionNetwork.

  An interaction network computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incoming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This implementation adds an option not in https://arxiv.org/abs/1612.00222,
  which is to include edge features for which a node is a sender in the
  arguments to the node update function.

  Args:
    update_edge_fn: mapping of functions used to update a subset of the edge
      types, indexed by edge type name.
    update_node_fn: mapping of functions used to update a subset of the node
      types, indexed by node type name.
    gather_nodes_for_edges_fn: function used to gather nodes to edges. A simple
      function can be used which will be applied to both senders or receivers
      or, alternatively a GatherNodesForEdgesTuple or
      GatherNodesForEdgesConstructor. Thes opjects allow for the sender and
      receiver gathers to be specified independently and the construction of
      the function to be deferred till calltime.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node. A simple function can be used which will be applied to both senders
      or receivers or, alternatively a AggregateEdgesForNodesTuple or
      AggregateEdgesForNodesConstructor. Thes opjects allow for the sender and
      receiver aggregation to be specified independently and the construction
      of the function to be deferred till calltime.
    include_sent_messages_in_node_update: pass edge features for which a node is
      a sender to the node update function.
    **graph_network_kwargs: kwargs for the GraphNetwork.
  """
  # An InteractionNetwork is a GraphNetwork without globals features,
  # so we implement the InteractionNetwork as a configured GraphNetwork.

  # An InteractionNetwork edge function does not have global feature inputs,
  # so we filter the passed global argument in the GraphNetwork.
  wrapped_update_edge_fn = tree.tree_map(
      lambda fn: lambda e, s, r, g: fn(e, s, r), update_edge_fn)

  # Similarly, we wrap the update_node_fn to ensure only the expected
  # arguments are passed to the Interaction net.
  if include_sent_messages_in_node_update:
    wrapped_update_node_fn = tree.tree_map(
        lambda fn: lambda n, s, r, g: fn(n, s, r), update_node_fn)
  else:
    wrapped_update_node_fn = tree.tree_map(
        lambda fn: lambda n, s, r, g: fn(n, r), update_node_fn)
  return GraphNetwork(
      update_edge_fn=wrapped_update_edge_fn,
      update_node_fn=wrapped_update_node_fn,
      gather_nodes_for_edges_fn=gather_nodes_for_edges_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      **graph_network_kwargs,
  )


def GraphMapFeatures(  # pylint: disable=invalid-name
    embed_edge_fn: Optional[Mapping[str, jraph.EmbedEdgeFn]] = None,
    embed_node_fn: Optional[Mapping[str, jraph.EmbedNodeFn]] = None,
    embed_global_fn: Optional[jraph.EmbedGlobalFn] = None):
  """Returns function which embeds the components of a graph independently.

  Args:
    embed_edge_fn: mapping of functions used to embed each edge type,
      indexed by edge type name.
    embed_node_fn: mapping of functions used to embed each node type,
      indexed by node type name.
    embed_global_fn: function used to embed the globals.
  """

  def _embed(graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:

    updated_edges = dict(graph.edges)
    if embed_edge_fn:
      for edge_set_name, embed_fn in embed_edge_fn.items():
        edge_set_key = graph.edge_key_by_name(edge_set_name)
        edge_set = graph.edges[edge_set_key]
        updated_edges[edge_set_key] = edge_set._replace(
            features=embed_fn(edge_set.features))

    updated_nodes = dict(graph.nodes)
    if embed_node_fn:
      for node_set_key, embed_fn in embed_node_fn.items():
        node_set = graph.nodes[node_set_key]
        updated_nodes[node_set_key] = node_set._replace(
            features=embed_fn(node_set.features))

    updated_context = graph.context
    if embed_global_fn:
      updated_context = updated_context._replace(
          features=embed_global_fn(updated_context.features))

    return graph._replace(edges=updated_edges, nodes=updated_nodes,
                          context=updated_context)

  return _embed
