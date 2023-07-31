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
"""A predictor that runs multiple graph neural networks on mesh data.

It learns to interpolate between the grid and the mesh nodes, with the loss
and the rollouts ultimately computed at the grid level.

It uses ideas similar to those in Keisler (2022):

Reference:
  https://arxiv.org/pdf/2202.07575.pdf

It assumes data across time and level is stacked, and operates only operates in
a 2D mesh over latitudes and longitudes.
"""

from typing import Any, Callable, Mapping, Optional

import chex
from graphcast import deep_typed_graph_net
from graphcast import grid_mesh_connectivity
from graphcast import icosahedral_mesh
from graphcast import losses
from graphcast import model_utils
from graphcast import predictor_base
from graphcast import typed_graph
from graphcast import xarray_jax
import jax.numpy as jnp
import jraph
import numpy as np
import xarray

Kwargs = Mapping[str, Any]

GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]


# https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)

# https://www.ecmwf.int/en/forecasts/datasets/set-i
PRESSURE_LEVELS_HRES_25 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600,
    700, 800, 850, 900, 925, 950, 1000)

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

PRESSURE_LEVELS = {
    13: PRESSURE_LEVELS_WEATHERBENCH_13,
    25: PRESSURE_LEVELS_HRES_25,
    37: PRESSURE_LEVELS_ERA5_37,
}

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)
EXTERNAL_FORCING_VARS = (
    "toa_incident_solar_radiation",
)
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS
STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)


@chex.dataclass(frozen=True, eq=True)
class TaskConfig:
  """Defines inputs and targets on which a model is trained and/or evaluated."""
  input_variables: tuple[str, ...]
  # Target variables which the model is expected to predict.
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  pressure_levels: tuple[int, ...]
  input_duration: str

TASK = TaskConfig(
    input_variables=(
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_ERA5_37,
    input_duration="12h",
)
TASK_13 = TaskConfig(
    input_variables=(
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration="12h",
)
TASK_13_PRECIP_OUT = TaskConfig(
    input_variables=(
        TARGET_SURFACE_NO_PRECIP_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration="12h",
)


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
  """Defines the architecture of the GraphCast neural network architecture.

  Properties:
    resolution: The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    mesh_size: How many refinements to do on the multi-mesh.
    gnn_msg_steps: How many Graph Network message passing steps to do.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.
    mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.
  """
  resolution: float
  mesh_size: int
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: Optional[float] = None


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  params: dict[str, Any]
  model_config: ModelConfig
  task_config: TaskConfig
  description: str
  license: str


class GraphCast(predictor_base.Predictor):
  """GraphCast Predictor.

  The model works on graphs that take into account:
  * Mesh nodes: nodes for the vertices of the mesh.
  * Grid nodes: nodes for the points of the grid.
  * Nodes: When referring to just "nodes", this means the joint set of
    both mesh nodes, concatenated with grid nodes.

  The model works with 3 graphs:
  * Grid2Mesh graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from grid nodes to mesh nodes using a
    fixed radius query. The grid2mesh_gnn will operate in this graph. The output
    of this stage will be a latent representation for the mesh nodes, and a
    latent representation for the grid nodes.
  * Mesh graph: Graph that contains mesh nodes only. The mesh_gnn will
    operate in this graph. It will update the latent state of the mesh nodes
    only.
  * Mesh2Grid graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from mesh nodes to grid nodes such that each grid
    nodes is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.

  """

  def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
    """Initializes the predictor."""
    self._spatial_features_kwargs = dict(
        add_node_positions=False,
        add_node_latitude=True,
        add_node_longitude=True,
        add_relative_positions=True,
        relative_longitude_local_coordinates=True,
        relative_latitude_local_coordinates=True,
    )

    # Specification of the multimesh.
    self._meshes = (
        icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config.mesh_size))

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
        embed_edges=True,  # Embed raw features of the grid2mesh edges.
        edge_latent_size=dict(grid2mesh=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        name="grid2mesh_gnn",
    )

    # Processor, which performs message passing on the multi-mesh.
    self._mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the multi-mesh edges.
        node_latent_size=dict(mesh_nodes=model_config.latent_size),
        edge_latent_size=dict(mesh=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=model_config.gnn_msg_steps,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        name="mesh_gnn",
    )

    num_surface_vars = len(
        set(task_config.target_variables) - set(ALL_ATMOSPHERIC_VARS))
    num_atmospheric_vars = len(
        set(task_config.target_variables) & set(ALL_ATMOSPHERIC_VARS))
    num_outputs = (num_surface_vars +
                   len(task_config.pressure_levels) * num_atmospheric_vars)

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        # Require a specific node dimensionaly for the grid node outputs.
        node_output_size=dict(grid_nodes=num_outputs),
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the mesh2grid edges.
        edge_latent_size=dict(mesh2grid=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        name="mesh2grid_gnn",
    )

    # Obtain the query radius in absolute units for the unit-sphere for the
    # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
    self._query_radius = (_get_max_edge_distance(self._finest_mesh)
                          * model_config.radius_query_fraction_edge_length)
    self._mesh2grid_edge_normalization_factor = (
        model_config.mesh2grid_edge_normalization_factor
    )

    # Other initialization is delayed until the first call (`_maybe_init`)
    # when we get some sample data so we know the lat/lon values.
    self._initialized = False

    # A "_init_mesh_properties":
    # This one could be initialized at init but we delay it for consistency too.
    self._num_mesh_nodes = None  # num_mesh_nodes
    self._mesh_nodes_lat = None  # [num_mesh_nodes]
    self._mesh_nodes_lon = None  # [num_mesh_nodes]

    # A "_init_grid_properties":
    self._grid_lat = None  # [num_lat_points]
    self._grid_lon = None  # [num_lon_points]
    self._num_grid_nodes = None  # num_lat_points * num_lon_points
    self._grid_nodes_lat = None  # [num_grid_nodes]
    self._grid_nodes_lon = None  # [num_grid_nodes]

    # A "_init_{grid2mesh,processor,mesh2grid}_graph"
    self._grid2mesh_graph_structure = None
    self._mesh_graph_structure = None
    self._mesh2grid_graph_structure = None

  @property
  def _finest_mesh(self):
    return self._meshes[-1]

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               is_training: bool = False,
               ) -> xarray.Dataset:
    self._maybe_init(inputs)

    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

    # Transfer data for the grid to the mesh,
    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    (latent_mesh_nodes, latent_grid_nodes
     ) = self._run_grid2mesh_gnn(grid_node_features)

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

    # Transfer data frome the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    output_grid_nodes = self._run_mesh2grid_gnn(
        updated_latent_mesh_nodes, latent_grid_nodes)

    # Conver output flat vectors for the grid nodes to the format of the output.
    # [num_grid_nodes, batch, output_size] ->
    # xarray (batch, one time step, lat, lon, level, multiple vars)
    return self._grid_node_outputs_to_prediction(
        output_grid_nodes, targets_template)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    # Forward pass.
    predictions = self(
        inputs, targets_template=targets, forcings=forcings, is_training=True)
    # Compute loss.
    loss = losses.weighted_mse_per_level(
        predictions, targets,
        per_variable_weights={
            # Any variables not specified here are weighted as 1.0.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            "2m_temperature": 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        })
    return loss, predictions  # pytype: disable=bad-return-type  # jax-ndarray

  def loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> predictor_base.LossAndDiagnostics:
    loss, _ = self.loss_and_predictions(inputs, targets, forcings)
    return loss  # pytype: disable=bad-return-type  # jax-ndarray

  def _maybe_init(self, sample_inputs: xarray.Dataset):
    """Inits everything that has a dependency on the input coordinates."""
    if not self._initialized:
      self._init_mesh_properties()
      self._init_grid_properties(
          grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)
      self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
      self._mesh_graph_structure = self._init_mesh_graph()
      self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

      self._initialized = True

  def _init_mesh_properties(self):
    """Inits static properties that have to do with mesh nodes."""
    self._num_mesh_nodes = self._finest_mesh.vertices.shape[0]
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        self._finest_mesh.vertices[:, 0],
        self._finest_mesh.vertices[:, 1],
        self._finest_mesh.vertices[:, 2])
    (
        mesh_nodes_lat,
        mesh_nodes_lon,
    ) = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    # Convert to f32 to ensure the lat/lon features aren't in f64.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Inits static properties that have to do with grid nodes."""
    self._grid_lat = grid_lat.astype(np.float32)
    self._grid_lon = grid_lon.astype(np.float32)
    # Initialized the counters.
    self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

    # Initialize lat and lon for the grid.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

  def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""

    # Create some edges according to distance between mesh and grid nodes.
    assert self._grid_lat is not None and self._grid_lon is not None
    (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=self._grid_lat,
        grid_longitude=self._grid_lon,
        mesh=self._finest_mesh,
        radius=self._query_radius)

    # Edges sending info from grid to mesh.
    senders = grid_indices
    receivers = mesh_indices

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._grid_nodes_lat,
         senders_node_lon=self._grid_nodes_lon,
         receivers_node_lat=self._mesh_nodes_lat,
         receivers_node_lon=self._mesh_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([mesh_indices.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=senders_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=receivers_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")):
            edge_set
    }
    grid2mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return grid2mesh_graph

  def _init_mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh graph."""
    merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)

    # Work simply on the mesh edges.
    senders, receivers = icosahedral_mesh.faces_to_edges(merged_mesh.faces)

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features(
        node_lat=self._mesh_nodes_lat,
        node_lon=self._mesh_nodes_lon,
        senders=senders,
        receivers=receivers,
        **self._spatial_features_kwargs,
    )

    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    assert n_mesh_node == len(node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set
    }
    mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    (grid_indices,
     mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
         grid_latitude=self._grid_lat,
         grid_longitude=self._grid_lon,
         mesh=self._finest_mesh)

    # Edges sending info from mesh to grid.
    senders = mesh_indices
    receivers = grid_indices

    # Precompute structural node and edge features according to config options.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._mesh_nodes_lat,
         senders_node_lon=self._mesh_nodes_lon,
         receivers_node_lat=self._grid_nodes_lat,
         receivers_node_lon=self._grid_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=receivers_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=senders_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
            edge_set
    }
    mesh2grid_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return mesh2grid_graph

  def _run_grid2mesh_gnn(self, grid_node_features: chex.Array,
                         ) -> tuple[chex.Array, chex.Array]:
    """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""

    # Concatenate node structural features with input features.
    batch_size = grid_node_features.shape[1]

    grid2mesh_graph = self._grid2mesh_graph_structure
    assert grid2mesh_graph is not None
    grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
    mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]
    new_grid_nodes = grid_nodes._replace(
        features=jnp.concatenate([
            grid_node_features,
            _add_batch_second_axis(
                grid_nodes.features.astype(grid_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # To make sure capacity of the embedded is identical for the grid nodes and
    # the mesh nodes, we also append some dummy zero input features for the
    # mesh nodes.
    dummy_mesh_node_features = jnp.zeros(
        (self._num_mesh_nodes,) + grid_node_features.shape[1:],
        dtype=grid_node_features.dtype)
    new_mesh_nodes = mesh_nodes._replace(
        features=jnp.concatenate([
            dummy_mesh_node_features,
            _add_batch_second_axis(
                mesh_nodes.features.astype(dummy_mesh_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # Broadcast edge structural features to the required batch size.
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(dummy_mesh_node_features.dtype), batch_size))

    input_graph = self._grid2mesh_graph_structure._replace(
        edges={grid2mesh_edges_key: new_edges},
        nodes={
            "grid_nodes": new_grid_nodes,
            "mesh_nodes": new_mesh_nodes
        })

    # Run the GNN.
    grid2mesh_out = self._grid2mesh_gnn(input_graph)
    latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features
    latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features
    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array) -> chex.Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = latent_mesh_nodes.shape[1]

    mesh_graph = self._mesh_graph_structure
    assert mesh_graph is not None
    mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
    edges = mesh_graph.edges[mesh_edges_key]

    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named "mesh" for the edges and that it uses a single set of nodes named
    # "mesh_nodes"
    msg = ("The setup currently requires to only have one kind of edge in the"
           " mesh GNN.")
    assert len(mesh_graph.edges) == 1, msg

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_mesh_nodes.dtype), batch_size))

    nodes = mesh_graph.nodes["mesh_nodes"]
    nodes = nodes._replace(features=latent_mesh_nodes)

    input_graph = mesh_graph._replace(
        edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes})

    # Run the GNN.
    return self._mesh_gnn(input_graph).nodes["mesh_nodes"].features

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
                         ) -> chex.Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = updated_latent_mesh_nodes.shape[1]

    mesh2grid_graph = self._mesh2grid_graph_structure
    assert mesh2grid_graph is not None
    mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
    grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
    mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
    edges = mesh2grid_graph.edges[mesh2grid_key]

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_grid_nodes.dtype), batch_size))

    input_graph = mesh2grid_graph._replace(
        edges={mesh2grid_key: new_edges},
        nodes={
            "mesh_nodes": new_mesh_nodes,
            "grid_nodes": new_grid_nodes
        })

    # Run the GNN.
    output_graph = self._mesh2grid_gnn(input_graph)
    output_grid_nodes = output_graph.nodes["grid_nodes"].features

    return output_grid_nodes

  def _inputs_to_grid_node_features(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> chex.Array:
    """xarrays -> [num_grid_nodes, batch, num_channels]."""

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat(
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
        stacked_inputs)
    return xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
        (-1,) + grid_xarray_lat_lon_leading.data.shape[2:])

  def _grid_node_outputs_to_prediction(
      self,
      grid_node_outputs: chex.Array,
      targets_template: xarray.Dataset,
      ) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""

    # numpy array with shape [lat_lon_node, batch, channels]
    # to xarray `DataArray` (batch, lat, lon, channels)
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
    grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
        grid_shape + grid_node_outputs.shape[1:])
    dims = ("lat", "lon", "batch", "channels")
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)


def _add_batch_second_axis(data, batch_size):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([batch_size, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]


def _get_max_edge_distance(mesh):
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max()
