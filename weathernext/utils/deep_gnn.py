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

"""Implementation of a Deep GNN for Typed Graphs."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
from typing import Any, Optional, TypedDict

import chex
from weathernext.utils import dense
from weathernext.utils import gather_scatter_ops
from weathernext.utils import sharding_utils
from weathernext.utils import typed_graph as typed_jraph
from weathernext.utils import typed_graph_net
import haiku as hk
import jax.numpy as jnp
import jraph
import numpy as np


GraphToGraphNetwork = Callable[[typed_jraph.TypedGraph], typed_jraph.TypedGraph]

Kwargs = Mapping[str, Any]


class GatherScatterKwargs(TypedDict, total=False):
  """Possible kwargs for gather and scatter operations."""
  explicit_sharding: bool | None = None  # pyrefly: ignore[bad-class-definition]
  remove_collectives_if_local: bool | None = None  # pyrefly: ignore[bad-class-definition]


class DeepGNN(hk.Module):
  """Deep Graph Neural Network.

  It works with TypedGraphs with typed nodes and edges. It runs message
  passing on all of the node sets and all of the edge sets in the graph.

  This class may be used for shared or unshared message passing:
  * num_message_passing_steps = N, num_processor_repetitions = 1, gives
    N layers of message passing with fully unshared weights:
    [W_1, W_2, ... , W_M] (default)
  * num_message_passing_steps = 1, num_processor_repetitions = M, gives
    N layers of message passing with fully shared weights:
    [W_1] * M
  * num_message_passing_steps = N, num_processor_repetitions = M, gives
    M*N layers of message passing with both shared and unshared message passing
    such that the weights used at each iteration are:
    [W_1, W_2, ... , W_N] * M

  """

  def __init__(
      self,
      *,
      dense_kwargs: dense.DenseLayerKwargs,
      num_message_passing_steps: int,
      num_processor_repetitions: int = 1,
      edge_dense_kwargs: dense.DenseLayerKwargs | None = None,
      use_edge_residuals: bool = True,
      gather_from_receivers: bool = True,
      edge_update_remat_block_size: Optional[int] = None,
      remat_block_size: Optional[int] = None,
      f32_aggregation: bool = False,
      gather_scatter_kwargs: GatherScatterKwargs | None = None,
      aggregate_normalization: Optional[float] = None,
      pre_gather_matmul: bool = False,
      name: str = "DeepGNN",
  ):
    """Inits the model.

    Args:
      dense_kwargs: Kwargs for dense layers.
      num_message_passing_steps: Number of unshared message passing steps in the
        processor steps.
      num_processor_repetitions: Number of times that the same processor is
        applied sequencially.
      edge_dense_kwargs: Kwargs for dense layers for the edge updates. If None,
        `dense_kwargs` will be used.
      use_edge_residuals: Whether to use residual conections between the edge
        updates. This requires input and output edge size to match.
      gather_from_receivers: Whether to include the gather from receivers.
      edge_update_remat_block_size: number of edges updated in a batch, None if
        disabling the rematerialization.
      remat_block_size: Determines number of message passing step contained in
        rematerialization block. Setting this to None or 0 switches remat off.
      f32_aggregation: Use float32 in the edge aggregation.
      gather_scatter_kwargs: Kwargs for gather and scatter operations.
      aggregate_normalization: An optional constant that normalizes the output
        of aggregate_edges_for_nodes_fn. For context, this can be used to reduce
        the shock the model undergoes when switching resolution, which increase
        the number of edges connected to a node. In particular, this is useful
        when using segment_sum, but should not be combined with segment_mean.
      pre_gather_matmul: Apply an optimisation that splits the first matmul of
        the edge MLP and applies the sender and receiver components _before_
        their respective gathers.
      name: Name of the model.
    """  # pylint: disable=g-doc-bad-indent

    super().__init__(name=name)
    self._dense_kwargs = dense_kwargs

    if edge_dense_kwargs is None:
      edge_dense_kwargs = dense_kwargs

    self._edge_dense_kwargs = edge_dense_kwargs

    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._use_edge_residuals = use_edge_residuals

    self._initialized = False
    self._edge_update_remat_block_size = edge_update_remat_block_size
    self._remat_block_size = remat_block_size
    self._f32_aggregation = f32_aggregation
    self._aggregate_normalization = aggregate_normalization
    self._pre_gather_matmul = pre_gather_matmul

    if gather_scatter_kwargs is None:
      gather_scatter_kwargs = {}
    scatter_kwargs = dict(gather_scatter_kwargs)
    self._gather_nodes_for_edges_constructor = (
        _build_gather_nodes_for_edges_constructor(
            gather_from_receivers=gather_from_receivers,
            gather_fn_kwargs=gather_scatter_kwargs,
        )
    )

    self._aggregate_edges_for_nodes_constructor = (
        _build_aggregate_edges_for_node_constructor(
            scatter_fn_kwargs=scatter_kwargs
        )
    )

  def __call__(self,
               input_graph: typed_jraph.TypedGraph,
               global_norm_conditioning: Optional[chex.Array] = None,
               # is_training is currently unused but included for consistency
               # with other models.
               is_training: Optional[bool] = None,
               ) -> typed_jraph.TypedGraph:
    """Forward pass of the learnable dynamics model.

    Args:
      input_graph: Input TypedGraph with a single array of features for
          each node type and each edge type. The shape of the features will
          typically be:
            [num_nodes/num_edges, (optional_batch_dims, ...), feature_size]
      global_norm_conditioning: Global norm conditioning array to apply
          norm conditioning after each activation normalization. The shape
          of this array should be:
            [(optional_batch_dims, ...), norm_conditioning_size]
          This norm conditining will be used globally (broadcasted to each node
          and each edge).
      is_training: Whether the model is in training mode.

    Returns:
        Output TypedGraph with the same set of node types and edge types.

    """
    sharding_utils.inspect_typed_graph_sharding(
        input_graph, name=f"{self.name}.input"
    )
    processor_networks = self._networks_builder(
        input_graph, global_norm_conditioning)

    output_graph = self._process(input_graph, processor_networks)
    sharding_utils.inspect_typed_graph_sharding(
        output_graph, name=f"{self.name}.output"
    )
    return output_graph

  @hk.transparent
  def _networks_builder(
      self,
      graph_template: typed_jraph.TypedGraph,
      global_norm_conditioning: Optional[chex.Array],
  ) -> list[GraphToGraphNetwork]:

    def build_dense_layer(
        name: str,
        dense_kwargs: dense.DenseLayerKwargs,
        sum_inputs_and_drop_first_matmul: bool = False,
        ) -> Callable[..., Any]:

      dense_layer = dense.DenseLayer(
          name=name, **dense_kwargs,
          drop_first_matmul=sum_inputs_and_drop_first_matmul,
      )
      dense_layer = functools.partial(
          dense_layer, norm_conditioning=global_norm_conditioning)
      if sum_inputs_and_drop_first_matmul:
        return dense.summed_args(dense_layer)  # pytype: disable=wrong-arg-types
      else:
        return jraph.concatenated_args(dense_layer)  # pytype: disable=wrong-arg-types

    def build_pre_gather_matmul(name: str) -> Callable[..., Any]:
      # Here we want to mimic the initialization for the first weight matrix
      # in the MLP which uses truncated normal with a stddev of 1/fan_in. We
      # don't have access to the exact fan_in as it is dependent on the
      # runtime data so we approximate it as
      # 2*node_output_size+edge_output_size. This will be correct so long as
      # we have node and edge residuals (and will raise an error otherwise), but
      # may not guaranteed to be right if we don't.
      approx_fan_in = (
          self._dense_kwargs["output_size"] * 2 +
          self._edge_dense_kwargs["output_size"]
          )
      stddev = 1.0 / np.sqrt(approx_fan_in)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)

      # The output of this layer may have to be of size "output_size", or
      # "hidden_size", depending on whether this is the last layer of the
      # mlp (num_hidden_layers==0) or not (num_hidden_layers>0).
      num_actual_hidden_layers = self._edge_dense_kwargs["num_hidden_layers"]
      if (self._edge_dense_kwargs["activate_final"] and
          self._edge_dense_kwargs["one_less_layer_when_activate_final"]):
        num_actual_hidden_layers -= 1

      if num_actual_hidden_layers == 0:
        output_size = self._edge_dense_kwargs["output_size"]
      else:
        output_size = self._edge_dense_kwargs["hidden_size"]

      return hk.Linear(
          output_size=output_size,
          w_init=w_init, with_bias=False, name=name
      )

    def build_interaction_network_layer(
        graph_template, index
    ) -> Callable[..., Any]:
      if self._pre_gather_matmul:
        pre_gather_senders_for_edges_fn = _build_update_fns_for_edge_types(
            build_pre_gather_matmul,
            graph_template,
            f"processor_edges_{index}_sender_",
        )
        pre_gather_receivers_for_edges_fn = _build_update_fns_for_edge_types(
            build_pre_gather_matmul,
            graph_template,
            f"processor_edges_{index}_receiver_",
        )
        pre_gather_edges_for_edges_fn = _build_update_fns_for_edge_types(
            build_pre_gather_matmul,
            graph_template,
            f"processor_edges_{index}_edge_",
        )
      else:
        pre_gather_senders_for_edges_fn = dict()
        pre_gather_receivers_for_edges_fn = dict()
        pre_gather_edges_for_edges_fn = dict()

      return typed_graph_net.InteractionNetwork(
          update_edge_fn=_build_update_fns_for_edge_types(
              functools.partial(
                  build_dense_layer,
                  dense_kwargs=self._edge_dense_kwargs,
                  sum_inputs_and_drop_first_matmul=self._pre_gather_matmul),
              graph_template,
              f"processor_edges_{index}_",
          ),
          update_node_fn=_build_update_fns_for_node_types(
              functools.partial(
                  build_dense_layer,
                  dense_kwargs=self._dense_kwargs),
              graph_template,
              f"processor_nodes_{index}_",
          ),
          gather_nodes_for_edges_fn=self._gather_nodes_for_edges_constructor,
          aggregate_edges_for_nodes_fn=self._aggregate_edges_for_nodes_constructor,
          include_sent_messages_in_node_update=False,
          edge_update_remat_block_size=self._edge_update_remat_block_size,
          pre_gather_senders_for_edges_fn=pre_gather_senders_for_edges_fn,
          pre_gather_receivers_for_edges_fn=pre_gather_receivers_for_edges_fn,
          pre_gather_edges_for_edges_fn=pre_gather_edges_for_edges_fn,
      )

    # The encoder graph network independently encodes edge and node features.
    def maybe_upcast_and_normalize_wrapper(
        fn: Optional[Callable[..., jnp.ndarray]],
    ):
      if fn is None:
        return None

      def wrapped_fn(data: jnp.ndarray, *args, **kwargs):
        dtype = data.dtype
        if self._f32_aggregation:
          data = data.astype(jnp.float32)
        output = fn(data, *args, **kwargs)
        if self._aggregate_normalization:
          output = output / self._aggregate_normalization
        if self._f32_aggregation:
          output = output.astype(dtype)
        return output

      return wrapped_fn

    self._aggregate_edges_for_nodes_constructor = dataclasses.replace(
        self._aggregate_edges_for_nodes_constructor,  # pytype: disable=attribute-error
        decorator=maybe_upcast_and_normalize_wrapper,
    )

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    processor_networks = []
    for step_i in range(self._num_message_passing_steps):
      processor_networks.append(
          build_interaction_network_layer(graph_template, index=step_i)
      )

    return processor_networks

  def _process(
      self, latent_graph_0: typed_jraph.TypedGraph,
      processor_networks: Sequence[GraphToGraphNetwork],
      ) -> typed_jraph.TypedGraph:
    """Processes the latent graph with several steps of message passing."""

    # Do `num_message_passing_steps` with each of the `processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    latent_graph = latent_graph_0
    for unused_repetition_i in range(self._num_processor_repetitions):

      num_processor_network = len(processor_networks)

      def apply_processors(processors, remat):

        def inner_apply(latent_graph_):
          for processor in processors:
            latent_graph_ = self._process_step(processor, latent_graph_)
          return latent_graph_

        if remat:
          inner_apply = hk.remat(inner_apply)

        return inner_apply

      block_size = self._remat_block_size

      if block_size is None or block_size == 0:
        # Apply all the processors.
        latent_graph = apply_processors(
            processor_networks, remat=False)(
                latent_graph)
      else:
        # Apply processors in blocks
        for start in range(0, num_processor_network, block_size):
          processor_block = processor_networks[start:start + block_size]
          latent_graph = apply_processors(
              processor_block, remat=True)(
                  latent_graph)

    return latent_graph

  def _process_step(
      self, processor_network_k: GraphToGraphNetwork,
      latent_graph_prev_k: typed_jraph.TypedGraph,) -> typed_jraph.TypedGraph:
    """Single step of message passing with node/edge residual connections."""

    # One step of message passing.
    latent_graph_k = processor_network_k(latent_graph_prev_k)

    # Add residuals.
    nodes_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.nodes.items():
      nodes_with_residuals[k] = prev_set._replace(
          features=prev_set.features + latent_graph_k.nodes[k].features)

    edges_maybe_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.edges.items():
      # If the input edge latent sizes do not match the processed latent sizes
      # then use_edge_residuals must be set to False to avoid a shape error.
      # Note that this removes all edge residuals but alternatively just the
      # first could be removed. This could be implemented in the future.
      if self._use_edge_residuals:
        new_edge_features = prev_set.features + latent_graph_k.edges[k].features
      else:
        new_edge_features = latent_graph_k.edges[k].features
      edges_maybe_with_residuals[k] = prev_set._replace(
          features=new_edge_features)

    latent_graph_k = latent_graph_k._replace(
        nodes=nodes_with_residuals, edges=edges_maybe_with_residuals)
    return latent_graph_k


def _build_update_fns_for_node_types(
    builder_fn: Callable[..., Any],
    graph_template: typed_jraph.TypedGraph,
    prefix: str,
) -> dict[str, Callable[..., Any]]:
  """Builds an update function for all node types or a subset of them."""
  output_fns = {}
  for node_set_name in graph_template.nodes.keys():
    output_fns[node_set_name] = builder_fn(f"{prefix}{node_set_name}")
  return output_fns


def _build_update_fns_for_edge_types(
    builder_fn: Callable[..., Any],
    graph_template: typed_jraph.TypedGraph,
    prefix: str,
    remat: bool = False,
) -> dict[str, Callable[..., Any]]:
  """Builds an edge function for all node types or a subset of them."""

  output_fns = {}
  for edge_set_key in graph_template.edges.keys():
    edge_set_name = edge_set_key.name
    fn = builder_fn(f"{prefix}{edge_set_name}")
    if remat:
      fn = hk.remat(fn)
    output_fns[edge_set_name] = fn
  return output_fns


def _get_gather_scatter_constructor(fn_name, kwargs=None):
  """Return gather or scatter constructor corresponding to function_name."""
  fn = getattr(gather_scatter_ops, fn_name)
  if kwargs is not None:
    fn = functools.partial(fn, **kwargs)
  return fn


_VALID_KWARGS_KEYS = ("explicit_sharding", "remove_collectives_if_local")


def _build_aggregate_edges_for_node_constructor(
    scatter_fn_kwargs: dict[str, Any] | None = None,
) -> typed_graph_net.AggregateEdgesForNodesConstructor:
  """Builds an aggregate edges for nodes constructor using sorted scatter."""
  if scatter_fn_kwargs is None:
    scatter_fn_kwargs = {}
  if not set(scatter_fn_kwargs.keys()).issubset(set(_VALID_KWARGS_KEYS)):
    raise ValueError(
        "Unknown keys in scatter_fn_kwargs:"
        f" {set(scatter_fn_kwargs.keys()) - set(_VALID_KWARGS_KEYS)}"
    )
  fn_kwargs = {
      k: v for k, v in scatter_fn_kwargs.items() if k in _VALID_KWARGS_KEYS
  }
  scatter_fn = _get_gather_scatter_constructor(
      "create_static_sorted_segment_sum", fn_kwargs
  )
  return typed_graph_net.AggregateEdgesForNodesConstructor(
      senders=None, receivers=scatter_fn
  )


def _build_gather_nodes_for_edges_constructor(
    gather_from_receivers: bool = True,
    gather_fn_kwargs: GatherScatterKwargs | None = None,
) -> typed_graph_net.GatherNodesForEdgesConstructor:
  """Builds a gather nodes for edges constructor using sorted gather."""
  if gather_fn_kwargs is None:
    gather_fn_kwargs = {}
  if not set(gather_fn_kwargs.keys()).issubset(set(_VALID_KWARGS_KEYS)):
    raise ValueError(
        "Unknown keys in gather_fn_kwargs:"
        f" {set(gather_fn_kwargs.keys()) - set(_VALID_KWARGS_KEYS)}"
    )
  sender_fn_kwargs = {
      k: v for k, v in gather_fn_kwargs.items() if k in _VALID_KWARGS_KEYS
  }
  receiver_fn_kwargs = {
      k: v for k, v in gather_fn_kwargs.items() if k in _VALID_KWARGS_KEYS
  }
  sender_fn = _get_gather_scatter_constructor(
      "create_static_gather", sender_fn_kwargs
  )
  if gather_from_receivers:
    receiver_fn = _get_gather_scatter_constructor(
        "create_static_sorted_gather", receiver_fn_kwargs
    )
  else:
    receiver_fn = None
  return typed_graph_net.GatherNodesForEdgesConstructor(
      senders=sender_fn, receivers=receiver_fn
  )
