
from typing import Optional
import xarray
import chex
import jax.numpy as jnp
import numpy as np

from graphcast.losses import stacked_mse
from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndDiagnostics
from graphcast.graphcast import GraphCast, ModelConfig, TaskConfig
from graphcast import xarray_jax

class StackedGraphCast(GraphCast, StackedPredictor):

    def __init__(
        self,
        model_config: ModelConfig,
        task_config: TaskConfig
        ):
        super().__init__(model_config=model_config, task_config=task_config)

        # since we don't use xarray DataArrays as inputs, we have to
        # establish the grid somehow. Seems easiest to pass it via task_config
        # just like the pressure levels
        self._init_grid_properties(
            grid_lat=np.array(task_config.latitude),
            grid_lon=np.array(task_config.longitude),
        )


    def __call__(
        self,
        inputs: chex.Array,
        ) -> chex.Array:

        self._maybe_init()

        # Convert all input data into flat vectors for each of the grid nodes.
        # xarray (batch, time, lat, lon, level, multiple vars, forcings)
        # -> [num_grid_nodes, batch, num_channels]
        grid_node_features = self._inputs_to_grid_node_features(inputs)

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
        return self._grid_node_outputs_to_prediction(output_grid_nodes)

    def loss_and_predictions(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: Optional[chex.Array | None] = None,
        ) -> tuple[StackedLossAndDiagnostics, chex.Array]:
        # Forward pass
        predictions = self(inputs)

        # Compute loss
        loss, diagnostics = stacked_mse(
            predictions=predictions,
            targets=targets,
            weights=weights,
        )
        return (loss, diagnostics), predictions

    def loss(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: Optional[chex.Array | None] = None,
        ) -> StackedLossAndDiagnostics:

        (loss, diagnostics), _ = self.loss_and_predictions(inputs, targets, weights)
        return loss, diagnostics


    def _maybe_init(self):
        if not self._initialized:
            self._init_mesh_properties()
            # grid properties initialized at __init__
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

            self._initialized = True


    def _inputs_to_grid_node_features(
        self,
        inputs: chex.Array,
        ) -> chex.Array:
        """inputs expected to be as [batch, lat, lon, channels] or [lat, lon, channels]

        Returns:
            array with shape [latxlon, batch, channels]
        """

        # NOTE: to remove this, we would have to overwrite a lot of GraphCast "batch_second_axis" code
        inputs = inputs[None] if inputs.ndim == 3 else inputs
        result = np.moveaxis(inputs, 0, 2)

        shape = (-1,) + result.shape[2:]
        result = result.reshape(shape)
        return result

    def _grid_node_outputs_to_prediction(
        self,
        grid_node_outputs: chex.Array,
        ) -> chex.Array:
        """returned as [batch, lat, lon, channels] or [lat, lon, channels]"""

        assert self._grid_lat is not None and self._grid_lon is not None
        grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])

        # produces [lat, lon, batch, channels]
        result = grid_node_outputs.reshape(
            grid_shape + grid_node_outputs.shape[1:],
        )
        # get batch dimension first again
        result = np.moveaxis(result, 2, 0)
        result = result.squeeze()
        return result
