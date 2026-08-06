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

"""A GNN to update TriangularMeshData and LatLonPointsData."""

from collections.abc import Mapping
import enum
from typing import Any

from absl import logging
import chex

from weathernext.utils import data_modalities
from weathernext.utils import deep_gnn
from weathernext.utils import dense
from weathernext.utils import icosahedral_mesh
from weathernext.utils import model_utils as feature_utils
from weathernext.utils import padding_utils
from weathernext.utils import sharding
from weathernext.utils import sharding_utils
from weathernext.utils import typed_graph
from weathernext.utils import update_blocks
from weathernext.utils import xarray_dense
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from scipy import spatial
import trimesh
import xarray as xr


TriangularMeshData = data_modalities.TriangularMeshData
LatLonPointsData = data_modalities.LatLonPointsData
SingleArray = update_blocks.SingleArray
SingleOrCombinedArraysVar = update_blocks.SingleOrCombinedArraysVar
AnotherSingleOrCombinedArraysVar = (
    update_blocks.AnotherSingleOrCombinedArraysVar)


Kwargs = Mapping[str, Any]


MESH_NODES_NAME = "mesh_nodes"
POINT_NODES_NAME = "point_nodes"


class ConnectivityType(enum.Enum):
  # Each point is connected to the closest point in the mesh.
  CLOSEST = 1
  # Each point is connected to the three vertices of the mesh triangular face
  # that contains the point.
  IN_TRIANGLE = 2
  # Each mesh point is connected to all points within a ball of a given radius
  # around it.
  BALL_QUERY = 3


class PointsMeshTypedGraphGNN(update_blocks.PairedBlockUpdate):
  """A GNN to simultaneously update `TriangularMeshData` and `LatLonPointsData`.

  It is implemented using a TypedGraph.

  Runs on the bipartite graph defined by the mesh and the points, according to
  the connectivity type. Note that the edge ordering is chosen such that the
  receivers are sorted. This allows for some additional performance
  optimisations.
  """

  @hk.name_like("__call__")
  def __init__(
      self,
      *,
      name: str,
      points_name: str = "points",
      mesh_name: str = "mesh",
      is_points_to_mesh: bool,
      is_mesh_to_points: bool,
      connectivity_type: ConnectivityType,
      dense_kwargs: dense.DenseLayerKwargs,
      # TODO(dominicmasters): Create a new class for the deep_gnn_kwargs to
      # allow for typing and checking the kwargs.
      deep_gnn_kwargs: Kwargs,
      spatial_edge_features_kwargs: Kwargs,
      stacked_points_inputs: bool = False,
      edge_encoder_dense_kwargs: dense.DenseLayerKwargs | None = None,
      ball_query_radius_fraction: float | None = None,
      pad_edges_to_multiple_of: int | None = None,
  ):
    """Initializes the update block.

    Args:
      name: The name of the update block, or a format string with
        `{points_name}` and `{mesh_name}` placeholders for the points and mesh
        names.
      points_name: The name of the points modality.
      mesh_name: The name of the mesh modality.
      is_points_to_mesh: Whether the edges go from the points to the mesh.
      is_mesh_to_points: Whether the edges go from the mesh to the points.
      connectivity_type: The type of connectivity to use to build edges.
      dense_kwargs: kwargs to `DenseLayer`.
      deep_gnn_kwargs: kwargs to `DeepGNN` (excluding `dense_kwargs`).
      spatial_edge_features_kwargs: kwargs to `get_edge_features`.
      stacked_points_inputs: Whether the points inputs are stacked. This means
        that the points inputs now expect to have shape [points, batch,
        *stack_size*, features]. If True, the mesh and points outputs will
        also be stacked.
      edge_encoder_dense_kwargs: Kwargs just for the edge encoder dense layer.
        If `None`, it will default to `dense_kwargs`.
      ball_query_radius_fraction: Radius of connectivity to use for when
        connectivity_type is BALL_QUERY. It is specified as a value scaled to
        the longest edge in the finest mesh. Must be left as None for other
        connectivity types.
      pad_edges_to_multiple_of: Set padding multiple for edges. Only for use in
        testing, the default should otherwise do the right thing.
    """
    if not is_points_to_mesh ^ is_mesh_to_points:
      raise NotImplementedError(
          "Exactly one of is_grid_to_mesh and is_mesh_to_grid must be True."
      )
    super().__init__(name=name.format(points_name=points_name,
                                      mesh_name=mesh_name))

    self._stacked_points_inputs = stacked_points_inputs
    self._stack_size = None
    if (
        ball_query_radius_fraction is None
        and connectivity_type == ConnectivityType.BALL_QUERY
    ):
      raise ValueError(
          "`ball_query_radius_fraction` must be set when connectivity_type is "
          "BALL_QUERY."
      )
    if (
        ball_query_radius_fraction is not None
        and connectivity_type != ConnectivityType.BALL_QUERY
    ):
      logging.warning(
          "ball_query_radius_fraction is set but connectivity_type is not "
          "BALL_QUERY so ball_query_radius_fraction will be ignored."
      )

    self._is_points_to_mesh = is_points_to_mesh
    self._is_mesh_to_points = is_mesh_to_points
    self._connectivity_type = connectivity_type
    self._deep_gnn_kwargs = deep_gnn_kwargs
    self._ball_query_radius_fraction = ball_query_radius_fraction
    self._spatial_edge_features_kwargs = spatial_edge_features_kwargs
    if pad_edges_to_multiple_of is None:
      pad_edges_to_multiple_of = sharding.get_num_shards(
          sharding.SPATIAL_AXES
      )
    self._pad_edges_to_multiple_of = pad_edges_to_multiple_of
    if edge_encoder_dense_kwargs is None:
      edge_encoder_dense_kwargs = dict(dense_kwargs)  # pyrefly: ignore[bad-assignment]
    # Important to not modify the original dense_kwargs, as it is used in other
    # places.
    maybe_stacked_dense_kwargs = dense_kwargs.copy()  # pytype: disable=attribute-error
    maybe_stacked_dense_kwargs["stack"] = stacked_points_inputs
    self._typed_graph_gnn = deep_gnn.DeepGNN(
        name="deep_gnn",
        dense_kwargs=maybe_stacked_dense_kwargs,
        **deep_gnn_kwargs,
    )
    self._edge_encoder_dense = xarray_dense.DataArrayDictDenseEncoder(
        name="edge_encoder",
        preserved_dims=("points", "batch",),
        dims_to_split=(),
        partition_spec=jax.sharding.PartitionSpec(
            sharding.SPATIAL_AXES, sharding.BATCH_LIKE_AXES
        ),
        **edge_encoder_dense_kwargs,  # pyrefly: ignore[bad-unpacking]
    )
    # This assertion should really be called before
    # maybe_stacked_dense_kwargs["stack"] is set, but this was causing weird
    # pytype errors when _typed_graph_gnn was called so doing it here instead.
    assert (
        "stack" not in dense_kwargs
    ), "stack is set by the update block."

  def __call__(
      self,
      triangular_mesh_data: TriangularMeshData[
          SingleOrCombinedArraysVar],
      lat_lon_points_data: LatLonPointsData[
          AnotherSingleOrCombinedArraysVar],
      global_data: update_blocks.GlobalDataSingleOrCombinedArrays | None = None,
      is_training: bool = False,
  ) -> tuple[TriangularMeshData[SingleOrCombinedArraysVar],
             LatLonPointsData[AnotherSingleOrCombinedArraysVar]]:
    """Runs the update block."""
    del is_training  # Unused.

    # TODO(dominicmasters): Consider using the haiku module name to label these.
    sharding_utils.inspect_data_sharding(
        triangular_mesh_data,
        name=f"{self.name}.triangular_mesh_data")
    sharding_utils.inspect_data_sharding(
        lat_lon_points_data,
        name=f"{self.name}.lat_lon_points_data")
    if global_data is not None:
      sharding_utils.inspect_data_sharding(  # pyrefly: ignore[bad-specialization]
          global_data, name=f"{self.name}.global_data")

    (mesh_inputs_main,
     mesh_inputs_norm_conditioning,
     mesh_inputs_other_conditioning) = (
         update_blocks.separate_combined_arrays(triangular_mesh_data))
    (points_inputs_main,
     points_inputs_norm_conditioning,
     points_inputs_other_conditioning) = (
         update_blocks.separate_combined_arrays(lat_lon_points_data))

    if (mesh_inputs_norm_conditioning is not None or
        points_inputs_norm_conditioning is not None):
      raise NotImplementedError("Spatial norm conditioning is not supported.")

    if (mesh_inputs_other_conditioning is not None or
        points_inputs_other_conditioning is not None):
      logging.warning("Unused global other conditioning inputs.")

    # Get the global norm/other conditioning.
    global_inputs_norm_conditioning = None
    if global_data is not None:
      (global_inputs_main,
       global_inputs_norm_conditioning,
       global_inputs_other_conditioning) = (
           update_blocks.separate_combined_arrays(global_data))  # pyrefly: ignore[bad-specialization]
      if global_inputs_main is not None:
        raise NotImplementedError("Global main inputs are not supported.")

      if global_inputs_other_conditioning is not None:
        logging.warning("Unused global other conditioning inputs.")

    if self._stacked_points_inputs:
      assert points_inputs_main.ndim == 4  # pytype: disable=attribute-error
      stack_size = points_inputs_main.shape[-2]  # pytype: disable=attribute-error
      if self._stack_size is None:
        self._stack_size = stack_size
      else:
        assert (
            self._stack_size == stack_size
        ), "Stack size cannot change between calls to points_mesh_gnn."

    mesh_inputs_main = self._maybe_add_stack_dim(mesh_inputs_main)  # pyrefly: ignore[bad-argument-type]

    # Build the input graph with the regular "linear" features.
    graph = self._build_typed_graph(
        triangular_mesh_data.replace_data(mesh_inputs_main),
        lat_lon_points_data.replace_data(points_inputs_main),  # pyrefly: ignore[bad-argument-type]
        global_inputs_norm_conditioning)

    sharding_utils.inspect_typed_graph_sharding(
        graph, name=f"{self.name}.graph")

    if global_inputs_norm_conditioning is not None:
      assert global_inputs_norm_conditioning.ndim == 2  # pytype: disable=attribute-error
      # [batch, global_features] -> [num_points=1, batch, global_features]
      global_inputs_norm_conditioning = jnp.expand_dims(
          global_inputs_norm_conditioning, axis=0  # pyrefly: ignore[bad-argument-type]
      )
      global_inputs_norm_conditioning = self._maybe_add_stack_dim(
          global_inputs_norm_conditioning
      )

    # Update the graph.
    updated_graph = self._typed_graph_gnn(
        graph, global_inputs_norm_conditioning
    )

    mesh_features = updated_graph.nodes[MESH_NODES_NAME].features
    points_features = updated_graph.nodes[POINT_NODES_NAME].features

    return (  # pyrefly: ignore[bad-return]
        update_blocks.update_main_data(
            triangular_mesh_data,
            mesh_features
        ),
        update_blocks.update_main_data(
            lat_lon_points_data,
            points_features
        ),
    )

  def _build_typed_graph(
      self,
      triangular_mesh_data: TriangularMeshData[SingleArray],
      lat_lon_points_data: LatLonPointsData[SingleArray],
      global_inputs_conditioning: SingleArray | None,
  ) -> typed_graph.TypedGraph:
    """Builds a bipartite Typed Graph."""

    (grid_lat_no_padding, grid_lon_no_padding
     ) = _get_static_lat_lon_removing_trailing_padding(
         lat_lon_points_data)
    (mesh_lat_no_padding, mesh_lon_no_padding
     ) = _get_static_lat_lon_removing_trailing_padding(
         triangular_mesh_data)

    # Build edge indices (for now it is required that the lat/lon are known
    # statically and constant across the batch, e.g. batch axis is squeezable).
    points_indices, mesh_indices = self._get_edge_indices(
        grid_lat_no_padding,
        grid_lon_no_padding,
        mesh_lat_no_padding,
        mesh_lon_no_padding,
        triangular_mesh_data.finest_faces,
    )

    # Define senders or receives according to the direction of the edges.
    if self._is_points_to_mesh:
      assert not self._is_mesh_to_points
      edges_name = "points_to_mesh_nodes"
      edge_direction = (POINT_NODES_NAME, MESH_NODES_NAME)
      senders, receivers = points_indices, mesh_indices
      sender_modality = lat_lon_points_data
      receiver_modality = triangular_mesh_data
    else:
      assert self._is_mesh_to_points
      edges_name = "mesh_to_points_nodes"
      edge_direction = (MESH_NODES_NAME, POINT_NODES_NAME)
      senders, receivers = mesh_indices, points_indices
      sender_modality = triangular_mesh_data
      receiver_modality = lat_lon_points_data

    # Sort by receivers.
    sorting_indices = np.argsort(receivers)
    senders = senders[sorting_indices]
    receivers = receivers[sorting_indices]

    # Build features for the edges and encode them.
    edge_features = feature_utils.get_edge_features(
        # Squeezing batch dimension.
        sender_lat=sender_modality.lat.squeeze(1),  # pyrefly: ignore[bad-argument-type]
        sender_lon=sender_modality.lon.squeeze(1),  # pyrefly: ignore[bad-argument-type]
        receiver_lat=receiver_modality.lat.squeeze(1),  # pyrefly: ignore[bad-argument-type]
        receiver_lon=receiver_modality.lon.squeeze(1),  # pyrefly: ignore[bad-argument-type]
        sender_indices=senders,
        receiver_indices=receivers,
        **self._spatial_edge_features_kwargs,
    )

    # Pad edges.
    num_padded_edges = padding_utils.get_num_padded_edges(
        num_edges=len(senders),
        pad_edges_to_multiple_of=self._pad_edges_to_multiple_of,
    )
    edge_features, senders, receivers = padding_utils.pad_edges(
        edge_features, senders, receivers, padded_size=num_padded_edges
    )
    # Add back a batch dimension and convert to xarray.DataArray and encode.
    edge_features = xr.Dataset(
        {k: xr.DataArray(v[:, None], dims=("points", "batch"))
         for k, v in edge_features.items()}  # pyrefly: ignore[missing-attribute]
    ).astype(triangular_mesh_data.data.dtype)  # pyrefly: ignore[missing-attribute]
    edge_features = self._edge_encoder_dense(
        dict(spatial=edge_features.to_dataarray(dim="feature")),
        global_inputs_conditioning)

    # Broadcast mesh, points, and edges features to batch size.
    batch_size = max(  # point_dims_shape is [num_points, batch_size]
        triangular_mesh_data.point_dims_shape[1],
        lat_lon_points_data.point_dims_shape[1],
    )
    mesh_node_features = _broadcast_batch_dim(
        triangular_mesh_data.data, batch_size, np_=jnp  # pyrefly: ignore[bad-argument-type]
    )
    points_node_features = _broadcast_batch_dim(
        lat_lon_points_data.data, batch_size, np_=jnp  # pyrefly: ignore[bad-argument-type]
    )
    edge_features = _broadcast_batch_dim(edge_features, batch_size, np_=jnp)
    edge_features = self._maybe_add_stack_dim(edge_features)  # pyrefly: ignore[bad-argument-type]

    # Build the graph.
    num_mesh_nodes = mesh_node_features.shape[0]
    num_point_nodes = points_node_features.shape[0]
    num_edges = edge_features.shape[0]

    return typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes={
            MESH_NODES_NAME: typed_graph.NodeSet(
                n_node=np.array([num_mesh_nodes]),
                features=mesh_node_features,
            ),
            POINT_NODES_NAME: typed_graph.NodeSet(
                n_node=np.array([num_point_nodes]),
                features=points_node_features,
            ),
        },
        edges={
            typed_graph.EdgeSetKey(
                edges_name, edge_direction
            ): typed_graph.EdgeSet(
                n_edge=np.array([num_edges]),
                indices=typed_graph.EdgesIndices(
                    senders=senders, receivers=receivers
                ),
                features=edge_features,
            ),
        },
    )

  def _get_edge_indices(
      self,
      points_latitude: np.ndarray,
      points_longitude: np.ndarray,
      mesh_latitude: np.ndarray,
      mesh_longitude: np.ndarray,
      mesh_faces: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns indices of the edges of the bipartite graph."""

    # For this method we work in cartesian coordinates of the unit sphere.
    points_cartesian = np.stack(
        feature_utils.lat_lon_to_cartesian(
            points_latitude, points_longitude
        ),
        axis=-1,
    )
    mesh_cartesian = np.stack(
        feature_utils.lat_lon_to_cartesian(
            mesh_latitude, mesh_longitude
        ),
        axis=-1,
    )
    num_point_nodes = points_cartesian.shape[0]

    if self._connectivity_type == ConnectivityType.CLOSEST:
      tree = spatial.cKDTree(mesh_cartesian)
      _, mesh_indices = tree.query(points_cartesian, k=1)  # [num_point_nodes]
      points_indices = np.arange(num_point_nodes)  # [num_point_nodes]

    elif self._connectivity_type == ConnectivityType.IN_TRIANGLE:
      mesh_trimesh = trimesh.Trimesh(
          vertices=mesh_cartesian, faces=mesh_faces, process=False
      )
      _, _, query_face_indices = mesh_trimesh.nearest.on_surface(
          points_cartesian
      )

      # [num_point_nodes, 3] with mesh node indices for each point node.
      mesh_indices = mesh_faces[query_face_indices]
      mesh_indices = mesh_indices.reshape([-1])  # [num_point_nodes * 3]

      # [num_point_nodes, 3] with grid node indices, where every row simply
      # contains the row (grid_point) index.
      points_indices = np.arange(num_point_nodes)[:, None]
      points_indices = np.tile(points_indices, [1, 3])
      points_indices = points_indices.reshape([-1])  # [num_point_nodes * 3]

    elif self._connectivity_type == ConnectivityType.BALL_QUERY:
      radius = self._ball_query_radius_fraction * _max_edge_distance(
          mesh_cartesian, mesh_faces
      )
      kd_tree = spatial.cKDTree(mesh_cartesian)

      # [num_point_nodes, variable_num_mesh_nodes_per_point_node]
      query_indices = kd_tree.query_ball_point(x=points_cartesian, r=radius)

      points_indices = []
      mesh_indices = []
      for point_index, mesh_neighbors in enumerate(query_indices):
        points_indices.append(np.repeat(point_index, len(mesh_neighbors)))
        mesh_indices.append(mesh_neighbors)

      # [num_point_nodes * average_num_mesh_nodes_per_point_node]
      points_indices = np.concatenate(points_indices, axis=0).astype(int)
      mesh_indices = np.concatenate(mesh_indices, axis=0).astype(int)
    else:
      raise ValueError(
          f"Unsupported connectivity type: {self._connectivity_type}"
      )
    return points_indices, mesh_indices

  def _maybe_add_stack_dim(self, input_array: jax.Array) -> jax.Array:
    """Adds a stack dimension to the input array if needed.

    Args:
      input_array: Input array, to potentially be stacked, with expected shape
        [num_points, batch_size, features]

    Returns:
      The stacked input array if stacking is enabled, otherwise the input array.
        The stacked array has expected shape [num_points, batch_size, features].
    """
    if self._stacked_points_inputs:
      assert self._stack_size is not None, (
          "self._stack_size must be defined when stacked_points_inputs is True."
      )
      assert input_array.ndim == 3, (
          "Input array must have 3 dimensions, found"
          f" {input_array.shape}."
      )
      input_array = jnp.expand_dims(input_array, axis=-2)
      stacked_array_shape = list(input_array.shape)
      stacked_array_shape[-2] = self._stack_size
      stacked_array = jnp.broadcast_to(input_array, stacked_array_shape)
      assert stacked_array.ndim == 4
      return stacked_array
    else:
      return input_array


def _broadcast_batch_dim(
    array: chex.Array, batch_size: int, np_=np
) -> chex.Array:
  array_shape = list(array.shape)
  array_shape[1] = batch_size
  return np_.broadcast_to(array, array_shape)


def _max_edge_distance(mesh_nodes_cartesian, mesh_faces):
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh_faces)
  edge_distances = np.linalg.norm(
      mesh_nodes_cartesian[senders] - mesh_nodes_cartesian[receivers], axis=-1
  )
  return edge_distances.max()


def _get_static_lat_lon_removing_trailing_padding(
    data_modality: data_modalities.SpatialData,
    ) -> tuple[np.ndarray, np.ndarray]:
  """Returns the lat/lon arrays with the trailing padding removed."""

  lat = data_modality.lat.squeeze(1)
  lon = data_modality.lon.squeeze(1)
  mask = data_modality.mask.squeeze(1)  # pyrefly: ignore[missing-attribute]

  # Verify they are known statically.
  assert isinstance(lat, np.ndarray)
  assert isinstance(lon, np.ndarray)
  assert isinstance(mask, np.ndarray)

  # Verify all padding is at the end.
  mask_diff = np.diff(mask.astype(np.int8))
  if np.any(mask_diff > 0):
    raise ValueError(
        f"Padding is expected at the end of the array, found: {mask}"
    )

  mask = np.broadcast_to(mask, data_modality.point_dims_shape[0])
  return lat[mask], lon[mask]
