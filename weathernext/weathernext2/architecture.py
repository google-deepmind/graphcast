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

"""Main model architecture."""

from collections.abc import Callable, Hashable, Mapping, Sequence
import functools
from typing import Any

from weathernext.utils import data_modalities
from weathernext.utils import dense
from weathernext.utils import model_utils as feature_utils
from weathernext.utils import sharding
from weathernext.utils import sharding_utils
from weathernext.utils import update_blocks
from weathernext.utils import update_blocks_utils as utils
from weathernext.utils import xarray_dense
from weathernext.weathernext2 import architecture_utils
import haiku as hk
import jax
import numpy as np
import xarray as xr
import xarray_jax


Kwargs = Mapping[str, Any]

CombinedArrays = data_modalities.CombinedArrays
SingleOrCombinedArraysVar = update_blocks.SingleOrCombinedArraysVar
TriangularMeshData = data_modalities.TriangularMeshData
LatLonPointsData = data_modalities.LatLonPointsData
LatLonGridData = data_modalities.LatLonGridData
GlobalData = data_modalities.GlobalData
PointsMeshUpdateConstructor = update_blocks.PointsMeshUpdateConstructor
MeshUpdateConstructor = update_blocks.MeshUpdateConstructor


class ForwardPass(hk.Module):
  """Forward pass for the underlying neural network architecture."""

  @hk.name_like("__call__")
  def __init__(
      self,
      *,
      latent_dense_kwargs: dense.DenseLayerKwargs,
      output_dense_kwargs: dense.DenseLayerKwargsExceptOutputSize,
      spatial_features_kwargs: Kwargs,
      mesh_num_splits: int,
      points_to_mesh_model_ctor: PointsMeshUpdateConstructor,
      mesh_model_ctor: MeshUpdateConstructor,
      mesh_to_grid_model_ctor: PointsMeshUpdateConstructor,
      points_lat_lon_major_axis: str | None = None,
      norm_conditioning_features: Sequence[str] = (),
      norm_conditioning_latent_dense_kwargs: (
          dense.DenseLayerKwargs | None) = None,
      per_var_activation_fns: Mapping[
          str, tuple[Callable[..., Any], Kwargs]] | None = None,
      extra_dims_to_split: tuple[str, ...] = (),
      mesh_padding_kwargs: Kwargs | None = None,
      remat_grid_to_mesh_gnn: bool = False,
      remat_mesh_gnn: bool = False,
      remat_mesh_to_grid_gnn: bool = False,
      name: str = "multimodality_forward",
      ):
    """Initializes the module.

    Args:
      latent_dense_kwargs: kwargs for the dense layers used throughout the model
        to produce various latent representations.
      output_dense_kwargs: kwargs for the dense layers used in any output mlps.
      spatial_features_kwargs: kwargs to `feature_utils.get_spatial_features` to
        create node features.
      mesh_num_splits: number of times the icosahedral mesh will be split.
      points_to_mesh_model_ctor: Mapping from data modalities names to
        constructors for the model that transforms points features to mesh
        features. A special key `None` can be used to specify a default
        constructor for all modalities not specified in the mapping. A single
        constructor can also be passed to specify the default constructor.
        If None, defaults to `grid_to_mesh_model_ctor` for backwards
        compatibility.
      mesh_model_ctor: Constructor for the model that transforms mesh features
        to new mesh features.
      mesh_to_grid_model_ctor: Constructor for the model that transforms mesh
        features to grid features.
      points_lat_lon_major_axis: The major axis to use when flattening lat-lon
        grids into points. If 'lat', the points will be ordered such that
        longitude varies faster than latitude. If 'lon', latitude varies faster
        than longitude. Defaults to 'lat'.
      norm_conditioning_features: names of features to use for norm
        conditioning.
      norm_conditioning_latent_dense_kwargs: kwargs for the dense layers used to
        encode the norm conditioning features. Must be provided if
        `norm_conditioning_features` is not empty.
      per_var_activation_fns: Mapping for variable to activation function and
        kwargs, to apply to the prediction right before returning.
      extra_dims_to_split: Additional dimensions to split weights over in
        xarray_dense. See xarray_dense.py for more details.
      mesh_padding_kwargs: Overdride default kwargs for the padding of the mesh.
        Intended for use in testing only.
      remat_grid_to_mesh_gnn: If True, the grid to mesh GNN will be wrapped in a
        remat.
      remat_mesh_gnn: If True, the mesh GNN will be wrapped in a remat.
      remat_mesh_to_grid_gnn: If True, the mesh to grid GNN will be wrapped in a
        remat.
      name: name of the module.
    """
    super().__init__(name=name)
    if mesh_padding_kwargs is None:
      mesh_padding_kwargs = dict(
          multiple_of=sharding.get_num_shards(sharding.SPATIAL_AXES),
          data_padding_value=0.0,
      )

    self._latent_dense_kwargs = latent_dense_kwargs
    self._output_dense_kwargs = output_dense_kwargs
    self._spatial_features_kwargs = spatial_features_kwargs
    self._points_to_mesh_model_ctor = points_to_mesh_model_ctor
    self._mesh_model_ctor = mesh_model_ctor
    self._mesh_to_grid_model_ctor = mesh_to_grid_model_ctor
    self._mesh_num_splits = mesh_num_splits

    # Keys in this dict must map to fields in `data_modalities.CombinedArrays`.
    self._norm_conditioning_features = norm_conditioning_features
    self._latent_dense_kwargs_norm_conditioning = (
        norm_conditioning_latent_dense_kwargs
    )

    if per_var_activation_fns is None:
      per_var_activation_fns = {}

    self._per_var_activation_fns = {}
    for var_name, (fn, fn_kwargs) in per_var_activation_fns.items():
      self._per_var_activation_fns[var_name] = functools.partial(
          fn, **fn_kwargs)

    # Order of the dims_to_split here will impact their order in variable names.
    # The string "time={value}" is not guaranteed to be unique in the name, so
    # we force it to always be the last dimension to help future variable name
    # modifications.
    self._dims_to_split = extra_dims_to_split + ("time",)
    self._mesh_padding_kwargs = mesh_padding_kwargs
    self._remat_grid_to_mesh_gnn = remat_grid_to_mesh_gnn
    self._remat_mesh_gnn = remat_mesh_gnn
    self._remat_mesh_to_grid_gnn = remat_mesh_to_grid_gnn
    if points_lat_lon_major_axis is None:
      points_lat_lon_major_axis = "lat"
    self._points_lat_lon_major_axis = points_lat_lon_major_axis

  def __call__(
      self,
      *,
      inputs: xr.Dataset,
      targets_template: xr.Dataset,
      forcings: xr.Dataset,
      is_training: bool = False,
  ) -> xr.Dataset:

    sharding_utils.inspect_xarray_sharding(
        inputs, name=f"{self.name}.inputs")
    sharding_utils.inspect_xarray_sharding(
        forcings, name=f"{self.name}.forcings")
    sharding_utils.inspect_xarray_sharding(
        targets_template, name=f"{self.name}.targets_template")

    # Get the dtype that the model will use.
    dtype = architecture_utils.get_floating_dtype((inputs, forcings))

    # Classify the input data according to the modality it lives on, and whether
    # it is meant to be used for main inputs or for norm conditioning.
    (
        global_data_norm_conditioning,
        global_main_data,
        grid_main_data,
    ) = architecture_utils.classify_input_data(
        inputs,
        forcings,
        norm_conditioning_features=self._norm_conditioning_features,
    )

    # Deal with the global data.
    # For the "main" part of the global data, we merge it with the data for each
    # grid, rather than passing it to the global data encoder.
    grid_main_data = architecture_utils.merge_global_data(
        grid_main_data, global_main_data
    )

    # Encode the global conditioning data, which will be passed to all modules.
    # GlobalData(main=None,
    #            norm_conditioning=array | None)
    global_data = encode_global_norm_conditioning_data(
        name_format="global_{}_encoder",
        global_data_norm_conditioning=global_data_norm_conditioning,  # pyrefly: ignore[bad-argument-type]
        dense_kwargs=self._latent_dense_kwargs_norm_conditioning,  # pyrefly: ignore[bad-argument-type]
        dims_to_split=self._dims_to_split,
    )

    # Setup mesh, encoding in it any mesh spatial features, as well
    # as any global features.
    input_mesh_data = TriangularMeshData.with_icosahedral_mesh(
        data=None, splits_list=[2] * self._mesh_num_splits)
    input_mesh_data = input_mesh_data.pad_data_to_multiple_of(
        padding_axis=0,
        **self._mesh_padding_kwargs,
    )
    # Note that input_mesh_data is permuted so that the finest mesh has a banded
    # adjacency.
    # MeshData(main=encoded, norm_conditioning=encoded | None)
    latent_mesh_data = encode_mesh_spatial_features(
        name="mesh_encoder",
        triangular_mesh_data=input_mesh_data,
        dense_kwargs=self._latent_dense_kwargs,
        spatial_features_kwargs=self._spatial_features_kwargs,
        dims_to_split=self._dims_to_split,
        dtype=dtype,  # pytype: disable=attribute-error
        extra_features=global_main_data,
        global_norm_conditioning=global_data.data.norm_conditioning,  # pyrefly: ignore[bad-argument-type]
    )

    # When encoding the lat lon grid data we will also keep some in grid
    # form for residual purposes.
    # LatLonPointsData(main=updated, norm_conditioning=encoded or None)
    # (for grid points)
    # LatLonGridData(main=updated, norm_conditioning=encoded or None)
    # (same data as previous but in lat lon grid format).
    latent_grid_points_data, latent_grid_data = (
        self._encode_lat_lon_grid_data_to_points_modality(
            lat=inputs["lat"],
            lon=inputs["lon"],
            grid_main_data=grid_main_data,
            global_data=global_data,
        ))

    # Grid to Mesh layer per modality.
    # MeshData(main=updated, norm_conditioning=same)
    # LatLonPointsData(main=updated, norm_conditioning=encoded or None)
    (
        updated_latent_mesh_data,
        updated_latent_grid_points_data,
    ) = self._run_points_to_mesh_model(
        latent_points_data=latent_grid_points_data,
        latent_mesh_data=latent_mesh_data,
        global_data=global_data,
        is_training=is_training,
    )

    latent_mesh_data = updated_latent_mesh_data
    lat_lon_points_latent = updated_latent_grid_points_data

    # Keep contents of "latent_grid_data", in sync with "lat_lon_points_latent".
    latent_grid_data = LatLonGridData.with_lat_lon_points(
        lat_lon_points_latent, template=latent_grid_data)

    # Mesh GNN to update the latent mesh data.
    mesh_model = self._mesh_model_ctor()
    if self._remat_mesh_gnn:
      mesh_model = utils.modality_data_remat(mesh_model)  # pyrefly: ignore[bad-specialization]
    # MeshData(main=updated, norm_conditioning=same)
    latent_mesh_data = mesh_model(
        latent_mesh_data, global_data, is_training=is_training)  # pyrefly: ignore[bad-argument-type]

    # MeshData(main=updated, norm_conditioning=same),
    # LatLonPointsData(main=updated, norm_conditioning=encoded or None)
    # -> xarray.Dataset, LatLonGridData(main=updated, norm_conditioning=same).
    predictions = self._decode_lat_lon_grid_data_from_mesh(
        targets_template=xr.Dataset(targets_template),
        latent_mesh_data=latent_mesh_data,
        encoded_grid_points_residual=lat_lon_points_latent,
        grid_data_template=latent_grid_data,
        global_data=global_data,
        is_training=is_training,
    )

    predictions = xr.Dataset(predictions)
    for var, activation_fn in self._per_var_activation_fns.items():
      predictions[var] = xarray_jax.apply_ufunc(activation_fn, predictions[var])

    return predictions

  @hk.transparent
  def _encode_lat_lon_grid_data_to_points_modality(
      self,
      lat: xr.DataArray,
      lon: xr.DataArray,
      grid_main_data: Mapping[str, xr.DataArray],
      global_data: GlobalData[CombinedArrays],
  ) -> tuple[
      LatLonPointsData[CombinedArrays],
      LatLonGridData[CombinedArrays],
  ]:

    # Encode the main data.
    # Note this function will preserve the spatial norm conditioning data
    # in `latent_grid_data` as part of the output`
    # LatLonGridData(main=encoded, norm_conditioning=array | None)
    latent_grid_data = encode_xarrays_to_lat_lon_grid_modality(
        name="grid_encoder",
        lat=lat,
        lon=lon,
        data_array_mapping=grid_main_data,  # pyrefly: ignore[bad-argument-type]
        dense_kwargs=self._latent_dense_kwargs,
        spatial_features_kwargs=self._spatial_features_kwargs,
        dims_to_split=self._dims_to_split,
        global_norm_conditioning=global_data.data.norm_conditioning,  # pyrefly: ignore[bad-argument-type]
        lat_name="lat",
        lon_name="lon",
    )

    # We change the format of the grid data to flatten the nodes. We could
    # have also done this before the encoder, but this way we can have
    # encoders that operate on the uniform grid format (e.g. CNNs).
    # LatLonPointsData(main=same, norm_conditioning=same)
    latent_grid_points_data = (
        data_modalities.LatLonPointsData.with_lat_lon_grid(
            latent_grid_data,
            major_axis=self._points_lat_lon_major_axis,
        )
    )

    # Sharding is not propagated through the `with_lat_lon_grid` call correctly
    # so we specify it manually.
    if sharding.is_global_mesh_defined():
      latent_grid_points_data = update_blocks.update_main_data(
          latent_grid_points_data,
          sharding_utils.set_sharding(
              latent_grid_points_data.data.main,
              partition_spec=jax.sharding.PartitionSpec(
                  sharding.SPATIAL_AXES,
                  sharding.BATCH_LIKE_AXES,
              ),
          ),
      )

    return latent_grid_points_data, latent_grid_data  # pyrefly: ignore[bad-return]

  @hk.transparent
  def _run_points_to_mesh_model(
      self,
      latent_points_data: LatLonPointsData[CombinedArrays],
      latent_mesh_data: TriangularMeshData[CombinedArrays],
      global_data: GlobalData[CombinedArrays],
      is_training: bool,
  ) -> tuple[
      TriangularMeshData[CombinedArrays],
      LatLonPointsData[CombinedArrays],
  ]:
    grid_to_mesh_model = self._points_to_mesh_model_ctor(points_name="grid")
    if self._remat_grid_to_mesh_gnn:
      grid_to_mesh_model = utils.modality_data_remat(grid_to_mesh_model)  # pyrefly: ignore[bad-specialization]

    # MeshData(main=updated, norm_conditioning=same)
    # LatLonPointsData(main=updated, norm_conditioning=same)
    (
        updated_latent_mesh_data,
        updated_latent_points_data,
    ) = grid_to_mesh_model(
        latent_mesh_data, latent_points_data, global_data, is_training  # pyrefly: ignore[bad-argument-type]
    )
    return updated_latent_mesh_data, updated_latent_points_data

  @hk.transparent
  def _decode_lat_lon_grid_data_from_mesh(
      self,
      targets_template: xr.Dataset,
      latent_mesh_data: TriangularMeshData[CombinedArrays],
      encoded_grid_points_residual: LatLonPointsData[CombinedArrays],
      grid_data_template: LatLonGridData[CombinedArrays],
      global_data: GlobalData[CombinedArrays],
      is_training: bool,
  ) -> xr.Dataset:

    # Mesh2Grid Model.
    mesh_to_grid_model = self._mesh_to_grid_model_ctor(points_name="grid")
    if self._remat_mesh_to_grid_gnn:
      mesh_to_grid_model = utils.modality_data_remat(mesh_to_grid_model)  # pyrefly: ignore[bad-specialization]

    # MeshData(main=updated, norm_conditioning=same)
    # LatLonPointsData(main=updated, norm_conditioning=same)
    unused_updated_latent_mesh_data, flat_latent_grid_data = mesh_to_grid_model(
        latent_mesh_data, encoded_grid_points_residual, global_data,  # pyrefly: ignore[bad-argument-type]
        is_training)

    # We change the format of the grid data to unflatten the nodes. We could
    # have also done this before the decoder, but this way we can have
    # decoders that operate on the uniform grid format (e.g. CNNs).
    # LatLonGridData(main=same, norm_conditioning=same)
    latent_grid_data = LatLonGridData.with_lat_lon_points(
        flat_latent_grid_data, template=grid_data_template)

    # Output decoder.
    # `xr.Dataset`
    return decode_xarrays_from_lat_lon_grid_modality(
        name="grid_decoder",
        lat_lon_grid_data=latent_grid_data,
        output_template=targets_template,
        dense_kwargs=self._output_dense_kwargs,
        dims_to_split=self._dims_to_split,
        lat_name="lat",
        lon_name="lon"
    )


def encode_xarrays_to_lat_lon_grid_modality(
    name: str,
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat_name: str,
    lon_name: str,
    data_array_mapping: Mapping[Hashable, xr.DataArray],
    dense_kwargs: dense.DenseLayerKwargs,
    spatial_features_kwargs: Mapping[str, Any],
    dims_to_split: Sequence[str],
    global_norm_conditioning: np.ndarray | None = None,
) -> LatLonGridData[CombinedArrays]:
  """Encodes a mapping of data arrays to a lat/lon grid modality.

  Args:
    name: The name of the encoder.
    lat: The latitude of the grid.
    lon: The longitude of the grid.
    lat_name: The name of the latitude dim.
    lon_name: The name of the longitude dim.
    data_array_mapping: A mapping of data arrays to encode. Their latitude
      and longitude coordinates must be consistent with `lat` and `lon`.
    dense_kwargs: kwargs to `DataArrayDictDenseEncoder`.
    spatial_features_kwargs: kwargs to `get_spatial_features`.
    dims_to_split: See `DataArrayDictDenseEncoder`.
    global_norm_conditioning: The global norm conditioning to use, with shape
      [batch, features].

  Returns:
    A `LatLonGridData` modality containing the encoded data.

  """

  dtype = list(data_array_mapping.values())[0].dtype
  grid_spatial_features = xr.Dataset(feature_utils.get_spatial_features(
      lat, lon, **spatial_features_kwargs
  )).astype(dtype)

  grid_spatial_features = grid_spatial_features.to_dataarray(dim="feature")
  input_xarray_dict = dict(
      spatial=grid_spatial_features,
      **data_array_mapping,
      )

  if global_norm_conditioning is not None:
    if global_norm_conditioning.ndim != 2:
      raise ValueError(
          "Global norm conditioning must be [batch, features].")
    # Add lat lon axes broadcastable.
    norm_conditioning = global_norm_conditioning[:, None, None]
  else:
    norm_conditioning = None

  latent_grid_array = xarray_dense.DataArrayDictDenseEncoder(
      name=name,
      preserved_dims=("batch", lat_name, lon_name),
      dims_to_split=dims_to_split,
      partition_spec=jax.sharding.PartitionSpec(
          sharding.BATCH_LIKE_AXES, None, sharding.SPATIAL_AXES
      ),
      **dense_kwargs,
  )(input_xarray_dict, norm_conditioning)

  output = LatLonGridData(
      data=CombinedArrays(main=latent_grid_array),
      lat=lat.data[None, :, None],
      lon=lon.data[None, None, :],
  )
  sharding_utils.inspect_data_sharding(output, name=name)
  return output


def decode_xarrays_from_lat_lon_grid_modality(
    name: str,
    lat_lon_grid_data: LatLonGridData[
        update_blocks.SingleOrCombinedArraysVar],
    output_template: xr.Dataset,
    dense_kwargs: dense.DenseLayerKwargsExceptOutputSize,
    dims_to_split: Sequence[str],
    lat_name: str,
    lon_name: str,
) -> xr.Dataset:
  """Decodes a Dataset from a lat/lon grid modality."""
  latent_data = update_blocks.separate_combined_arrays(lat_lon_grid_data)[0]  # pyrefly: ignore[bad-specialization]
  if not np.all(lat_lon_grid_data.lat ==
                output_template.coords[lat_name].data[None, :, None]):
    raise ValueError(
        "Latitudes of latent grid data and output template do not match.")

  if not np.all(lat_lon_grid_data.lon ==
                output_template.coords[lon_name].data[None, None, :]):
    raise ValueError(
        "Longitudes of latent grid data and output template do not match.")
  return xarray_dense.DataArrayDictDenseDecoder(
      name=name,
      preserved_dims=("batch", lat_name, lon_name),
      dims_to_split=dims_to_split,
      **dense_kwargs,
  )(latent_data, output_template)


def encode_mesh_spatial_features(
    name: str,
    triangular_mesh_data: data_modalities.TriangularMeshData[None],
    dense_kwargs: dense.DenseLayerKwargs,
    spatial_features_kwargs: Mapping[str, Any],
    extra_features: Mapping[str, xr.DataArray],
    dims_to_split: Sequence[str],
    dtype: np.dtype,
    global_norm_conditioning: np.ndarray | None = None,
) -> data_modalities.TriangularMeshData[CombinedArrays]:
  """Encodes the spatial features of a triangular mesh.

  For now it requires the mesh does not come with any other features of its own.

  Args:
    name: The name of the encoder.
    triangular_mesh_data: The triangular mesh for which to encode the spatial
      features.
    dense_kwargs: kwargs to `DataArrayDictDenseEncoder`.
    spatial_features_kwargs: kwargs to `get_spatial_features`.
    extra_features: Extra features that will be encoded with the mesh
      features.
    dims_to_split: See `DataArrayDictDenseEncoder`.
    dtype: The dtype of the output array.
    global_norm_conditioning: The global norm conditioning to use, with shape
      [batch, features].
  Returns:
    A `TriangularMeshData` modality containing the encoded spatial features.

  """

  if triangular_mesh_data.data is not None:
    raise NotImplementedError(
        "Triangular mesh data must not have any data features to encode.")

  # Squeeze batch dimension.
  lat = triangular_mesh_data.lat.squeeze(1)  # pyrefly: ignore[bad-argument-type]
  lon = triangular_mesh_data.lon.squeeze(1)  # pyrefly: ignore[bad-argument-type]
  mask = triangular_mesh_data.mask.squeeze(1)  # pyrefly: ignore[missing-attribute]

  lat = xr.DataArray(lat, dims=("points",))
  lon = xr.DataArray(lon, dims=("points",))

  mesh_spatial_features = feature_utils.get_spatial_features(
      lat, lon, **spatial_features_kwargs
  )
  # Nans are used as padding values for the lat, lon inputs and we don't want
  # them to propagate to the main features. Specifically they cause issues
  # with the splash attention layers where, presumably, the mask is applied as a
  # multiplication with a boolean array and thus they can contaminate the non
  # padding node features. We therefore set the masked values to 0.0 explicitly.
  # Note that mask_invalid_points is fine to use with the mesh_spatial_features,
  # which is a DataArray, because it is treated as an arbitrary tree inside the
  # function, but also because get_spatial_features(...) returns arrays which
  # match the leading dimensions of the triangular mesh data.
  mesh_spatial_features = data_modalities.mask_invalid_points(
      mesh_spatial_features, mask, 0.0)

  mesh_spatial_features = xr.Dataset(mesh_spatial_features).astype(dtype)
  mesh_spatial_features = mesh_spatial_features.to_dataarray(dim="feature")

  input_xarray_dict = dict(
      spatial=mesh_spatial_features,
      **extra_features)

  if global_norm_conditioning is not None:
    if global_norm_conditioning.ndim != 2:
      raise ValueError(
          "Global norm conditioning must be [batch, features].")
    # We could expand dim for a points axis at this point, but in practice this
    # is not needed since the points axis comes first, which means it will
    # already be broadcasted automatically.

  encoded_features = xarray_dense.DataArrayDictDenseEncoder(
      name=name,
      preserved_dims=("points", "batch"),
      dims_to_split=dims_to_split,
      partition_spec=jax.sharding.PartitionSpec(
          sharding.SPATIAL_AXES, sharding.BATCH_LIKE_AXES
      ),
      **dense_kwargs,
  )(input_xarray_dict, global_norm_conditioning)

  triangular_mesh_data = triangular_mesh_data.replace_data(  # pyrefly: ignore[bad-assignment]
      data=CombinedArrays(
          main=encoded_features,
          # Do not currently have spatial norm conditioning for the mesh.
          norm_conditioning=None,
      ))
  sharding_utils.inspect_data_sharding(triangular_mesh_data, name=name)  # pyrefly: ignore[bad-specialization]
  return triangular_mesh_data  # pyrefly: ignore[bad-return]


def encode_global_norm_conditioning_data(
    name_format: str,
    global_data_norm_conditioning: Mapping[Hashable, xr.DataArray],
    dense_kwargs: dense.DenseLayerKwargs,
    dims_to_split: Sequence[str],
) -> GlobalData[CombinedArrays]:
  """Encodes the global norm conditioning features."""

  encoded_data = xarray_dense.DataArrayDictDenseEncoder(
      name=name_format.format("norm_conditioning"),
      preserved_dims=("batch",),
      dims_to_split=dims_to_split,
      partition_spec=jax.sharding.PartitionSpec(sharding.BATCH_LIKE_AXES),
      **dense_kwargs,
  )(global_data_norm_conditioning)

  global_data = GlobalData(data=CombinedArrays(
      main=None, norm_conditioning=encoded_data))
  sharding_utils.inspect_data_sharding(global_data, name=name_format)
  return global_data

