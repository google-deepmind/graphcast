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

"""A Transformer to update features on `TriangularMeshData`."""

from collections.abc import Mapping
from typing import Any, Optional

from absl import logging
from weathernext.utils import data_modalities
from weathernext.utils import icosahedral_mesh
from weathernext.utils import sparse_transformer
from weathernext.utils import update_blocks
import jax.numpy as jnp
import numpy as np
from scipy import sparse


TriangularMeshData = data_modalities.TriangularMeshData
SingleArray = update_blocks.SingleArray
SingleOrCombinedArraysVar = update_blocks.SingleOrCombinedArraysVar


Kwargs = Mapping[str, Any]


def _get_adjacency_matrix(
    triangular_mesh_data: TriangularMeshData,
    add_self_edges: bool,
):
  """Returns the 2d adjacency matrix."""

  num_nodes = triangular_mesh_data.point_dims_shape[0]  # [n_node, batch, ...]
  s, r = icosahedral_mesh.faces_to_edges(
      triangular_mesh_data.finest_faces)

  # Build adjacency matrix.
  adj_mat = sparse.csr_matrix((num_nodes, num_nodes), dtype=np.bool_)
  adj_mat[s, r] = True
  if add_self_edges:
    adj_mat[np.arange(num_nodes), np.arange(num_nodes)] = True
  return adj_mat


class MeshSparseTransformer(update_blocks.SingleBlockUpdate):
  """A Transformer for inputs with ordering [nodes, batch, ...]."""

  def __init__(self,
               *,
               name: str,
               mesh_name: str = "mesh",
               transformer_kwargs: Kwargs):
    """Initialises the Transformer model.

    Args:
      name: The name of the update block, or a format string with a single
        `{mesh_name}` placeholder for the mesh name.
      mesh_name: The name of the mesh modality.
      transformer_kwargs: Kwargs to `sparse_transformer.Transformer`.
    """
    super().__init__(name=name.format(mesh_name=mesh_name))
    self._transformer_kwargs = transformer_kwargs

  def __call__(
      self,
      triangular_mesh_data: TriangularMeshData[SingleOrCombinedArraysVar],
      global_data: update_blocks.GlobalDataSingleOrCombinedArrays | None = None,
      is_training: Optional[bool] = None,
  ) -> TriangularMeshData[SingleOrCombinedArraysVar]:
    """Applies the model to the input graph and returns graph of same shape."""

    # Build the input graph with the regular "linear" features.
    inputs_main, inputs_norm_conditioning, inputs_other_conditioning = (
        update_blocks.separate_combined_arrays(triangular_mesh_data))

    if inputs_norm_conditioning is not None:
      raise NotImplementedError("Spatial norm conditioning is not supported.")

    if inputs_other_conditioning is not None:
      logging.warning("Unused other conditioning inputs.")

    # Get the global norm/other conditioning.
    global_inputs_norm_conditioning = None
    if global_data is not None:
      global_inputs_main, global_inputs_norm_conditioning, _ = (
          update_blocks.separate_combined_arrays(global_data)  # pyrefly: ignore[bad-specialization]
      )
      if global_inputs_main is not None:
        raise NotImplementedError("Global main inputs are not supported.")

    features = inputs_main
    if features.ndim != 3:  # pytype: disable=attribute-error  # jax-ndarray
      raise ValueError(
          "Expected `triangular_mesh_data.data` to be rank 3, got "
          f" {features.ndim}."
      )  # pytype: disable=attribute-error  # jax-ndarray

    adj_mat = _get_adjacency_matrix(
        triangular_mesh_data=triangular_mesh_data, add_self_edges=True)

    batch_first_transformer = sparse_transformer.Transformer(
        adj_mat=adj_mat,
        **self._transformer_kwargs,
    )

    # [node, batch, features] - > [batch, node, features] and back.
    inputs_batch_first = jnp.transpose(features, axes=[1, 0, 2])  # pyrefly: ignore[bad-argument-type]
    outputs_batch_first = batch_first_transformer(
        inputs_batch_first,
        global_norm_conditioning=global_inputs_norm_conditioning,
    )
    outputs = jnp.transpose(outputs_batch_first, axes=[1, 0, 2])
    return update_blocks.update_main_data(  # pyrefly: ignore[bad-return]
        triangular_mesh_data, outputs)
