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

"""Data modalities for the integrated model."""

import abc
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, NamedTuple, Self, TypeVar

import chex
from weathernext.utils import icosahedral_mesh
from weathernext.utils import model_utils as mesh_graph_net_feature_utils
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing_extensions import override
import xarray as xr
import xarray_jax

SingleArray = npt.ArrayLike


# A container for arrays that are used as inputs to the model, packing together
# inputs into linear layers, and inputs to norm conditioning layers, etc.
class CombinedArrays(NamedTuple):
  # Typically the main input/output to dense layers, etc.
  main: SingleArray | None

  norm_conditioning: SingleArray | None = None
  other_conditioning: SingleArray | None = None

# Classes and functions in these modules are defined as `Generic` to work as
# data containers of any tree of data, such that the same type of tree is used
# self-consistently in all of their methods. We can get pytype to enforce and
# propagate this self-consistency when doing type checking by using TypeVar's.
VALID_ARRAY_TREE_TYPES = (
    # We specify by hand the most common types of trees we use. And pytype will
    # be able to propagate these types from inputs to outputs.
    None,
    np.ndarray,
    Mapping[str, npt.ArrayLike],
    Iterable[npt.ArrayLike],
    # Then let `chex.ArrayTree` capture any other types of valid trees,
    # however since `chex.ArrayTree` is defined as a `Union`, is will not
    # propagate the specific type of tree from input to output.
    SingleArray,
    CombinedArrays,
    chex.ArrayTree,
)
ArrayTree = TypeVar("ArrayTree", *VALID_ARRAY_TREE_TYPES)  # pyrefly: ignore[invalid-annotation, invalid-type-var, not-a-type]

# To be used whenever pairs of ArrayTrees are involved.
OtherArrayTree = TypeVar("OtherArrayTree", *VALID_ARRAY_TREE_TYPES)  # pyrefly: ignore[invalid-annotation, invalid-type-var, not-a-type]

NumpyInterface = Any


DataOtherArrayTree = TypeVar("DataOtherArrayTree", bound="Data[OtherArrayTree]")  # pyrefly: ignore[invalid-annotation]

PerAxisMaskType = tuple[
    np.ndarray[tuple[int], np.dtype[np.bool_]] | tuple[bool, ...], ...
]

GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR = "_lat_lon_points_grid_major_axis"


class Data(Generic[ArrayTree], metaclass=abc.ABCMeta):
  """Base class for containing data.

  All arrays, have shape (*point_dims, *per_point_feature_dims).

  `point_dims` are any dims that represent points that _may_ have different
  associated coordinates or any form. Some examples of such dimensions may be:
  * Spatial axes such as lat lon.
  * A dim representing sparse lat lon points.
  * A batch axis that represents different instances of similar sets of points.
    (although note that the latitude and longitude values can be set to be
     broacastable for all elements in the batch).
  * A time axis that represents similar sets of points at different times (even
    if using the same coordinates if using broadcasting.)
  * Any combinations of the previous that have been flattened into a single dim.

  `per_point_feature_dims` are any dims that represent data associated with a
  specific instance of a point, always with the same coordinates.

  As a result all arrays must are broadcastable against each other on the
  leading `point_dims`.

  A mask can also be specified for the leading `point_dims` which satisfies the
  property of being broadcastable against the data arrays on the leading
  `point_dims`.
  """

  @property
  @abc.abstractmethod
  def point_dims_shape(self) -> tuple[int, ...]:
    """Returns the leading shape of the point dims."""

  @property
  @abc.abstractmethod
  def data(self) -> ArrayTree:
    """Returns the data for this modality."""

  @property
  def masked_data(self) -> ArrayTree:
    """Returns the data with invalid values set to zero."""
    return mask_invalid_points(self.data, self.mask, 0.0)  # pyrefly: ignore[bad-argument-type]

  @property
  @abc.abstractmethod
  def mask(self) -> SingleArray:
    """Returns the mask for the leading point_dims."""

  @property
  @abc.abstractmethod
  def metadata(self) -> dict[str, Any]:
    """Returns metadata dict associated with this data."""

  @abc.abstractmethod
  def replace_data(
      self: Self,
      data: OtherArrayTree,
      **other_init_kwargs,
  ) -> Self:
    """Returns a copy of the object with the data replaced."""

  @abc.abstractmethod
  def with_global_data(
      self, global_data: "GlobalData[OtherArrayTree]"
  ) -> DataOtherArrayTree:
    """Adds its own point spatial dims to the global data and returns itself."""

  def pad_data(
      self: Self,
      *,
      padding_axis: int,
      padding_amount: int,
      data_padding_value: Any,
  ) -> Self:
    """Returns a copy of the object with the data padded.

    Args:
      padding_axis: The axis to pad.
      padding_amount: The amount to pad.
      data_padding_value: The value to use for padding the data.

    Returns:
      A copy of the object with the data padded.
    """
    return self._pad_data(
        padding_axis=padding_axis,
        padding_amount=padding_amount,
        data_padding_value=data_padding_value,
    )

  def pad_data_to_multiple_of(
      self: Self,
      padding_axis: int,
      multiple_of: int,
      data_padding_value: Any
  ) -> Self:
    """Returns a copy of the object padded to a multiple of `multiple_of`."""
    if multiple_of <= 0:
      raise ValueError(
          f"multiple_of must be positive. Got {multiple_of}."
      )
    padding_amount = multiple_of - (
        (self.point_dims_shape[padding_axis] - 1) % multiple_of + 1
    )
    return self.pad_data(
        padding_axis=padding_axis,
        padding_amount=padding_amount,
        data_padding_value=data_padding_value,
    )

  def _pad_data(
      self: Self,
      *,
      padding_axis: int,
      padding_amount: int,
      data_padding_value: Any,
      additional_data: Mapping[str, ArrayTree] | None = None,
      additional_padding_values: Mapping[str, Any] | None = None,
  ) -> Self:
    """Private version of `pad_data` with optional additional data."""
    data = dict(data=self.data, **(additional_data or {}))
    data_padding_values = dict(
        data=data_padding_value, **(additional_padding_values or {})
    )
    # The output of _pad_data_impl is a dictionary with keys that match data
    # plus an additional key "mask" that is the new mask. This needs to
    # match the arguments of the class __init__.
    return self.replace_data(
        **_pad_data_impl(
            data=data,
            mask=self.mask,  # pyrefly: ignore[bad-argument-type]
            shared_leading_shape=self.point_dims_shape,
            padding_axis=padding_axis,
            padding_amount=padding_amount,
            data_padding_values=data_padding_values,
        ),
    )


class SpatialData(Data[ArrayTree]):
  """Data with associated "lat"/"lon" coordinates for the point_dims.

  The following constraints apply:
  * The lat lon arrays are broadcastable against each other on the `point_dims`.
  * The rank of the latitude and longitude arrays has to be equal to the
    number of `point_dims`.
  """

  @property
  @abc.abstractmethod
  def lat(self) -> np.ndarray:
    """Returns the latitude array."""

  @property
  @abc.abstractmethod
  def lon(self) -> np.ndarray:
    """Returns the longitude array."""

  @property
  def masked_lat(self) -> np.ndarray:
    """Returns the latitude with invalid values set to zero."""
    return mask_invalid_points(self.lat, self.mask, 0.0)  # pyrefly: ignore[bad-argument-type]

  @property
  def masked_lon(self) -> np.ndarray:
    """Returns the longitude with invalid values set to zero."""
    return mask_invalid_points(self.lon, self.mask, 0.0)  # pyrefly: ignore[bad-argument-type]

  @abc.abstractmethod
  def replace_data(
      self: Self,
      data: OtherArrayTree,
      *,
      lat: chex.Array | None = None,
      lon: chex.Array | None = None,
      **other_init_kwargs,
  ) -> Self:
    """Returns a copy of the object with the data replaced."""

  @override
  def _pad_data(
      self: Self, *args, **kwargs
  ) -> Self:
    return super()._pad_data(
        *args, **kwargs,
        additional_data=dict(lat=self.lat, lon=self.lon),  # pyrefly: ignore[bad-argument-type]
        additional_padding_values=dict(lat=np.nan, lon=np.nan),
    )


class LatLonPointsData(SpatialData[ArrayTree]):
  """Class for containing flat lists of latitude/longitude points data.

  All arrays involved (including latitude and longitude arrays) are required to
  have 2 `point_dims` corresponding to `(num_points, batch_size)`
  """

  @classmethod
  def with_lat_lon_grid(
      cls,
      lat_lon_grid_data: "LatLonGridData[OtherArrayTree]",
      major_axis: str = "lat",
  ) -> "LatLonPointsData[OtherArrayTree]":
    """Creates a `LatLonPointsData` from a `LatLonGridData`.

    Args:
      lat_lon_grid_data: The `LatLonGridData` to convert.
      major_axis: Which axis varies slowest in the flattened point dimension.
        'lat' (default) gives lat*lon ordering, 'lon' gives lon*lat ordering.

    Returns:
      `LatLonPointsData` object with the data and coordinates from the
      `LatLonGridData`, after flattening the lat/lon points in the 2d grid into
      a single list.
    """
    _, lat_size, lon_size = lat_lon_grid_data.point_dims_shape

    lat = _batch_lat_lon_to_latlon_batch(
        lat_lon_grid_data.lat,
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    lon = _batch_lat_lon_to_latlon_batch(
        lat_lon_grid_data.lon,
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    data = jax.tree_util.tree_map(
        lambda x: _batch_lat_lon_to_latlon_batch(
            x,
            lat_size,
            lon_size,
            np_=jnp,
            major_axis=major_axis,
        ),
        lat_lon_grid_data.data,
    )
    mask = _batch_lat_lon_to_latlon_batch(
        lat_lon_grid_data.mask,  # pyrefly: ignore[bad-argument-type]
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    return cls(
        data=data,  # (num_points, batch, ...)
        lat=lat,  # (num_points, batch)
        lon=lon,  # (num_points, batch)
        mask=mask,  # (num_points, batch)
        metadata={
            GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR: major_axis,
            **lat_lon_grid_data.metadata,
        },
    )

  def __init__(
      self,
      data: ArrayTree,
      lat: chex.Array,
      lon: chex.Array,
      per_axis_mask: PerAxisMaskType | None = None,
      mask: SingleArray | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    """Initializes the LatLonPointsData.

    Args:
      data: Any arbitrary tree of data with arrays, such as all arrays in the
        tree have leading axes (num_points, batch), followed by any per-array
        trailing shape.
      lat: The latitude array. Must have rank 2, with leading dimensons
        consistent with `data`.
      lon: The longitude array. Must have rank 2, with leading dimensons
        consistent with `data`.
      per_axis_mask: List of 1D boolean arrays (or tuples) defining the mask for
        each of the leading dimensions. Only one of `per_axis_mask` and `mask`
        can be set.
      mask: Dense mask for the leading dimensions of the data.
      metadata: Optional dictionary of metadata to associate with this data.
    """

    num_leading_shared_axes = 2  # (num_points, batch)

    if (lat.ndim != num_leading_shared_axes or
        lon.ndim != num_leading_shared_axes):
      raise ValueError(
          f"Latitudes and longitudes must have rank 2. Got shapes {lat.shape} "
          f"and {lon.shape}."
      )

    mask = _initialize_mask(mask, per_axis_mask, num_leading_shared_axes)

    # We leverage `ShareLeadingAxesArrayTree` to guarantee that the arrays
    # are broadcastable against each other on the leading dimensions. As a
    # implementation detail of this class, we store this as a dictionary,
    # although this dictionary never leaves the scope of this class.
    all_shared_leading_axis_arrays = {
        "lat": lat,
        "lon": lon,
        "data": data,
        "mask": mask,
    }

    self._data_container = ShareLeadingAxesArrayTree(
        all_shared_leading_axis_arrays,
        num_leading_shared_axes=num_leading_shared_axes,
    )
    self._metadata = dict(metadata) if metadata is not None else {}

    _verify_lat_lon(self)

  @property
  def point_dims_shape(self) -> tuple[int, int]:
    return self._data_container.shared_leading_shape  # pyrefly: ignore[bad-return]

  @property
  def data(self) -> ArrayTree:
    return self._data_container.array_tree["data"]

  @property
  def lat(self) -> chex.Array:  # pyrefly: ignore[bad-override]
    return self._data_container.array_tree["lat"]

  @property
  def lon(self) -> chex.Array:  # pyrefly: ignore[bad-override]
    return self._data_container.array_tree["lon"]

  @property
  def mask(self) -> SingleArray:
    return self._data_container.array_tree["mask"]

  @property
  def metadata(self) -> dict[str, Any]:
    return dict(self._metadata)

  @property
  def major_axis(self) -> str | None:
    """Returns the major axis used to create this data, or None."""
    return self._metadata.get(GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR)

  def replace_data(  # pyrefly: ignore[bad-override]
      self,
      data: OtherArrayTree,
      *,
      lat: chex.Array | None = None,
      lon: chex.Array | None = None,
      **other_init_kwargs,
  ) -> "LatLonPointsData[OtherArrayTree]":
    if lat is None:
      lat = self.lat
    if lon is None:
      lon = self.lon
    if "mask" not in other_init_kwargs:
      other_init_kwargs["mask"] = self.mask
    if "metadata" not in other_init_kwargs:
      other_init_kwargs["metadata"] = self._metadata
    return LatLonPointsData(
        data=data,
        lat=lat,
        lon=lon,
        **other_init_kwargs,
    )

  def with_global_data(  # pyrefly: ignore[bad-override]
      self, global_data: "GlobalData[OtherArrayTree]",
  ) -> "LatLonPointsData[OtherArrayTree]":
    # Add broadcastable "points" axis of size 1, in the leading position.
    data = jax.tree_util.tree_map(
        lambda x: _expand_dims(x, (0,)), global_data.data)
    return self.replace_data(data)


class GlobalData(Data[ArrayTree]):
  """Class for containing global data.

  All arrays involved are required to have 1 `point_dims` corresponding to
  `batch_size`, without any latitude or longitude arrays associated to them.

  """

  @classmethod
  def with_xarray_data_array(
      cls, data_array: xr.DataArray
  ) -> "GlobalData[npt.ArrayLike]":
    """Creates a GlobalData with an `xarray.DataArray`."""
    data_array = data_array.transpose("batch", ...)
    return cls(
        data=xarray_jax.unwrap_data(data_array),
    )

  def __init__(
      self,
      data: ArrayTree,
      per_axis_mask: PerAxisMaskType | None = None,
      mask: SingleArray | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    """Initializes the GlobalData.

    Args:
      data: Any arbitrary tree of data with arrays, such that all arrays in the
        tree have leading axes (batch,), followed by any per-array trailing
        shape.
      per_axis_mask: List of 1D boolean arrays (or tuples) defining the mask for
        each of the leading dimensions.
      mask: Dense mask for the leading dimensions of the data.
      metadata: Optional dictionary of metadata to associate with this data.
    """

    num_leading_shared_axes = 1  # (batch,)
    mask = _initialize_mask(mask, per_axis_mask, num_leading_shared_axes)

    # We leverage `ShareLeadingAxesArrayTree` to guarantee that the arrays
    # are broadcastable against each other on the leading dimensions.
    all_shared_leading_axis_arrays = {"data": data, "mask": mask}
    self._data_container = ShareLeadingAxesArrayTree(
        all_shared_leading_axis_arrays,
        num_leading_shared_axes=num_leading_shared_axes,
    )
    self._metadata = dict(metadata) if metadata is not None else {}

  @property
  def point_dims_shape(self) -> tuple[int, int]:
    return self._data_container.shared_leading_shape  # pyrefly: ignore[bad-return]

  @property
  def data(self) -> ArrayTree:
    return self._data_container.array_tree["data"]

  @property
  def mask(self) -> SingleArray:
    return self._data_container.array_tree["mask"]

  @property
  def metadata(self) -> dict[str, Any]:
    return self._metadata

  def replace_data(  # pyrefly: ignore[bad-override]
      self,
      data: OtherArrayTree,
      **other_init_kwargs,
  ) -> "GlobalData[OtherArrayTree]":
    if "mask" not in other_init_kwargs:
      other_init_kwargs["mask"] = self.mask
    if "metadata" not in other_init_kwargs:
      other_init_kwargs["metadata"] = self._metadata
    return GlobalData(data=data, **other_init_kwargs)

  def with_global_data(  # pyrefly: ignore[bad-override]
      self, global_data: "GlobalData[OtherArrayTree]",
  ) -> "GlobalData[OtherArrayTree]":
    return self.replace_data(global_data.data)


class LatLonGridData(SpatialData[ArrayTree]):
  """Class for containing uniform grid data.

  All arrays involved (including latitude and longitude arrays) are required to
  have 3 `point_dims` corresponding to `(batch_size, num_lat_points,
  num_lon_points)`.

  While not required, when latitude and longitude arrays are fixed across
  examples, they will typically have shapes `(1, num_lat_points, 1)` and
  `(1, 1, num_lon_points)`, respectively.
  """

  @classmethod
  def with_xarray_data_array(
      cls, data_array: xr.DataArray
  ) -> "LatLonGridData[npt.ArrayLike]":
    """Creates a LatLonGridData with an `xarray.DataArray`.

    Args:
      data_array: The data array to convert. Must contain "batch"/"lat"/"lon"
        dimensions and one-dimensional "lat"/"lon" coordinates.

    Returns:
      A LatLonGridData with the data from the `DataArray` in the data field.
    """
    lat, lon = _get_broadcastable_lat_lon_coords_from_grid_xarray(data_array)
    data_array = data_array.transpose("batch", "lat", "lon", ...)
    return cls(
        data=xarray_jax.unwrap_data(data_array),
        lat=lat,
        lon=lon,
    )

  @classmethod
  def with_xarray_dataset(
      cls, dataset: xr.Dataset
  ) -> "LatLonGridData[dict[str, npt.ArrayLike]]":
    """Creates a LatLonGridData with an array from an `xarray.Dataset`.

    Args:
      dataset: The dataset to convert. Must contain "batch"/"lat"/"lon"
        dimensions and one-dimensional "lat"/"lon" coordinates.

    Returns:
      A LatLonGridData with a dictionary containing the data from the `Dataset`
      in the data field.
    """

    lat, lon = _get_broadcastable_lat_lon_coords_from_grid_xarray(dataset)
    data_dict = {}
    for name, data_array in dataset.items():
      data_dict[str(name)] = xarray_jax.unwrap_data(
          data_array.transpose("batch", "lat", "lon", ...)
      )
    return cls(
        data=data_dict,
        lat=lat,
        lon=lon,
    )

  @classmethod
  def with_lat_lon_points(
      cls,
      lat_lon_points_data: LatLonPointsData[OtherArrayTree],
      template: "LatLonGridData",
      major_axis: str | None = None,
  ) -> "LatLonGridData[OtherArrayTree]":
    """Creates a `LatLonGridData` from a `LatLonPointsData` and a template.

    Args:
      lat_lon_points_data: The `LatLonPointsData` to convert.
      template: A `LatLonGridData` to use as a template for the output.
      major_axis: Which axis varies slowest in the flattened point dimension. If
        None, uses the major_axis stored on the `LatLonPointsData` (set
        automatically by `with_lat_lon_grid`). Raises ValueError if both are
        None.

    Returns:
      `LatLonGridData` object with the data from LatLonPointsData and
      coordinates from `LatLonGridData`, after unflattening the lat/lon points
      into two separate dimensions.
    """
    metadata = dict(lat_lon_points_data.metadata)
    # Remove the major_axis metadata that we don't want to pass to
    # LatLonGridData.
    metadata_major_axis = metadata.pop(GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR, None)
    if major_axis is None:
      major_axis = metadata_major_axis
    elif metadata_major_axis is not None and metadata_major_axis != major_axis:
      raise ValueError(
          "major_axis must be specified consistently. Explicitly specified "
          f"value is {major_axis}, but stored value is "
          f"{metadata_major_axis}."
      )
    if major_axis is None:
      raise ValueError(
          "major_axis must be specified either explicitly or by creating"
          " the LatLonPointsData via with_lat_lon_grid."
      )

    _, lat_size, lon_size = template.point_dims_shape

    actual_lat = _latlon_batch_to_batch_lat_lon(
        lat_lon_points_data.lat,
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    actual_lon = _latlon_batch_to_batch_lat_lon(
        lat_lon_points_data.lon,
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    if not np.all(actual_lat == template.lat):
      raise ValueError(
          f"Latitude values in template {template.lat} do not match latitude "
          f"values in LatLonPointsData {lat_lon_points_data.lat}."
      )
    if not np.all(actual_lon == template.lon):
      raise ValueError(
          f"Longitude values in template {template.lon} do not match longitude "
          f"values in LatLonPointsData {lat_lon_points_data.lon}."
      )

    mask = _latlon_batch_to_batch_lat_lon(
        lat_lon_points_data.mask,  # pyrefly: ignore[bad-argument-type]
        lat_size,
        lon_size,
        np_=np,
        major_axis=major_axis,
    )
    data = jax.tree_util.tree_map(
        lambda x: _latlon_batch_to_batch_lat_lon(
            x,
            lat_size,
            lon_size,
            np_=jnp,
            major_axis=major_axis,
        ),
        lat_lon_points_data.data,
    )
    return cls(
        data=data,
        lat=template.lat,
        lon=template.lon,
        mask=_simplify_mask(mask),  # pyrefly: ignore[bad-argument-type]
        metadata=metadata,
    )

  def __init__(
      self,
      data: ArrayTree,
      lat: np.ndarray,
      lon: np.ndarray,
      per_axis_mask: PerAxisMaskType | None = None,
      mask: SingleArray | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    """Initializes the LatLonGridData.

    Args:
      data: Any arbitrary tree of data with arrays, such as all arrays in the
        tree have leading axes (batch, lat, lon), followed by any per-array
        trailing shape.
      lat: The latitude array. Must have rank 3, with leading dimensons
        consistent with `data`.
      lon: The longitude array. Must have rank 3, with leading dimensons
        consistent with `data`.
      per_axis_mask: List of 1D boolean arrays (or tuples) defining the mask for
        each of the leading dimensions.
      mask: Dense mask for the leading dimensions of the data.
      metadata: Optional dictionary of metadata to associate with this data.
    """
    num_leading_shared_axes = 3  # (batch, lat, lon)

    if (lat.ndim != num_leading_shared_axes or
        lon.ndim != num_leading_shared_axes):
      raise ValueError(
          f"Latitudes and longitudes must have rank 3. Got shapes {lat.shape} "
          f"and {lon.shape}."
      )

    mask = _initialize_mask(mask, per_axis_mask, num_leading_shared_axes)

    # We leverage `ShareLeadingAxesArrayTree` to guarantee that the arrays
    # are broadcastable against each other on the leading dimensions. As a
    # implementation detail of this class, we store this as a dictionary,
    # although this dictionary never leaves the scope of this class.
    all_shared_leading_axis_arrays = {
        "lat": lat, "lon": lon, "data": data, "mask": mask}
    self._data_container = ShareLeadingAxesArrayTree(
        all_shared_leading_axis_arrays,
        num_leading_shared_axes=num_leading_shared_axes,
    )
    self._metadata = dict(metadata) if metadata is not None else {}
    if GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR in self._metadata:
      raise ValueError(
          f"{GRID_LAT_LON_POINT_MAJOR_AXIS_ATTR} cannot be set for"
          " LatLonGridData."
      )
    _verify_lat_lon(self)

  @property
  def point_dims_shape(self) -> tuple[int, int, int]:
    return self._data_container.shared_leading_shape  # pyrefly: ignore[bad-return]

  @property
  def data(self) -> ArrayTree:
    return self._data_container.array_tree["data"]

  @property
  def lat(self) -> np.ndarray:
    return self._data_container.array_tree["lat"]

  @property
  def lon(self) -> np.ndarray:
    return self._data_container.array_tree["lon"]

  @property
  def mask(self) -> SingleArray:
    return self._data_container.array_tree["mask"]

  @property
  def metadata(self) -> dict[str, Any]:
    return self._metadata

  def replace_data(  # pyrefly: ignore[bad-override]
      self,
      data: OtherArrayTree,
      *,
      lat: chex.Array | None = None,
      lon: chex.Array | None = None,
      **other_init_kwargs,
  ) -> "LatLonGridData[OtherArrayTree]":
    if lat is None:
      lat = self.lat
    if lon is None:
      lon = self.lon
    if "mask" not in other_init_kwargs:
      other_init_kwargs["mask"] = self.mask
    if "metadata" not in other_init_kwargs:
      other_init_kwargs["metadata"] = self._metadata
    return LatLonGridData(
        data=data,
        lat=lat,  # pyrefly: ignore[bad-argument-type]
        lon=lon,  # pyrefly: ignore[bad-argument-type]
        **other_init_kwargs,
    )

  def with_global_data(  # pyrefly: ignore[bad-override]
      self, global_data: "GlobalData[OtherArrayTree]",
  ) -> "LatLonGridData[OtherArrayTree]":
    # Add broadcastable "lat", "lon" axis of size 1, in the second and third
    # dimensions.
    data = jax.tree_util.tree_map(
        lambda x: _expand_dims(x, (1, 2,)), global_data.data)
    return self.replace_data(data)


class TriangularMeshData(LatLonPointsData[ArrayTree]):
  """Class for containing data on triangular meshes.

  All arrays involved (including latitude and longitude arrays) are required to
  have 2 shared leading dimensions corresponding to `(num_points, batch_size)`.

  Each of the points represent a vertex of a triangular mesh, and are connected
  to other mesh points by triangular faces. Typically there will be a "finest"
  mesh which connects all points, but also additional coarser meshes that only
  connect a subset of the points may be provided.

  In practice this is similar to `LatLonPointsData`, but such that the points
  correspond to the vertices of a triangular mesh, which are connected by one
  or more sets of triangular faces.
  """

  @classmethod
  def with_icosahedral_mesh(
      cls,
      *,
      splits_list: Sequence[int],
      data: OtherArrayTree = None,
      per_axis_mask: PerAxisMaskType | None = None,
      mask: SingleArray | None = None,
  ) -> "TriangularMeshData[OtherArrayTree]":
    """Creates a `TriangularMeshData` with an icosahedral mesh.

    Nodes of the mesh will be ordered such that the edge structure for the
    finest faces has a banded structure.

    Args:
      splits_list: The sequence of splits to use to build the icosahedral mesh
        by calling `get_hierarchy_of_triangular_meshes_for_sphere`.
      data: Any arbitrary tree of data with arrays, such that all arrays in the
        tree have leading axes (num_points, batch), followed by any per-array
        trailing shape. `num_points` must match the number of vertices in the
        mesh from `get_hierarchy_of_labeled_triangular_meshes_for_sphere`.
      per_axis_mask: List of 1D boolean arrays (or tuples) defining the mask for
        each of the leading dimensions.
      mask: Dense mask for the leading dimensions of the data.

    Returns :
      A `TriangularMeshData` with the data defined over the mesh, and face sets
      corresponding to all possible refinement levels of the mesh.
    """

    # TODO(alvarosg): Consider allowing to configure the constructor.
    labeled_meshes = icosahedral_mesh.get_hierarchy_of_labeled_triangular_meshes_for_sphere(  # pylint: disable=line-too-long
        splits_list=splits_list,
        finest_mesh_first=False,
        starting_refinement_level=0,
    )

    # Compute an optimal permutation on the nodes so the finest mesh has a
    # banded edge structure.
    finest_mesh = labeled_meshes[-1].mesh
    permutation, permute_func = icosahedral_mesh.get_permutation_to_banded(
        finest_mesh
    )
    # Apply the permutation to the nodes, and obtain the lat/lon coordinates.
    nodes_cartesian = finest_mesh.vertices[permutation]
    node_latitudes, node_longitudes = (
        mesh_graph_net_feature_utils.cartesian_to_lat_lon(
            *_unstack(nodes_cartesian, axis=-1)
        )
    )

    # Apply the permutation to the faces indices.
    face_sets = [
        (labeled_mesh.label, permute_func(labeled_mesh.mesh.faces))
        for labeled_mesh in labeled_meshes
    ]

    return cls(
        data=data,
        lat=node_latitudes[:, None],  # (num_nodes, batch)
        lon=node_longitudes[:, None],  # (num_nodes, batch)
        face_sets=face_sets,
        per_axis_mask=per_axis_mask,
        mask=mask,
    )

  def __init__(
      self,
      data: ArrayTree,
      lat: np.ndarray,
      lon: np.ndarray,
      face_sets: Sequence[tuple[str, np.ndarray]],
      per_axis_mask: PerAxisMaskType | None = None,
      mask: SingleArray | None = None,
      metadata: dict[str, Any] | None = None,
  ):
    """Initializes the `TriangularMeshData`.

    Args:
      data: Any arbitrary tree of data with arrays, such as all arrays in the
        tree have leading axes (num_points, batch), followed by any per-array
        trailing shape.
      lat: The latitude array. Must have rank 2, with leading dimensons
        consistent with `data`.
      lon: The longitude array. Must have rank 2, with leading dimensons
        consistent with `data`.
      face_sets: A sequence of sets of faces in increasing order of refinement.
        Each element will consist of a tuple of the form (label, faces), where
        the label is a string identifying the specific set of faces, and the
        faces is a numpy array with shape (num_faces, 3), with indices in the
        interval [0, num_points), referring to the points. The last one of the
        face sets will be the "finest" mesh, which will connect all points.
      per_axis_mask: List of 1D boolean arrays (or tuples) defining the mask for
        each of the leading dimensions. Only one of `per_axis_mask` and `mask`
        can be set.
      mask: Dense mask for the leading dimensions of the data.
      metadata: Optional dictionary of metadata to associate with this data.
    """
    super().__init__(data, lat, lon, per_axis_mask, mask, metadata=metadata)

    num_nodes = self.point_dims_shape[0]
    num_previous_faces = 0
    for _, faces in face_sets:
      if faces.ndim != 2:
        raise ValueError(f"Faces must have rank 2. Got shape {faces.shape}")
      if faces.shape[1] != 3:
        raise ValueError(
            f"Faces must have 3 vertices per face. Got shape {faces.shape}"
        )
      if faces.min() < 0 or faces.max() >= num_nodes:
        raise ValueError(
            "Faces must point to vertices in range [0, num_nodes), got "
            f"[{faces.min()}, or {faces.max()}]"
        )
      if faces.shape[0] <= num_previous_faces:
        raise ValueError(
            "Faces must be provided in increasing order of refinement"
        )
      num_previous_faces = faces.shape[0]

    self._face_sets = face_sets

  def replace_data(
      self,
      data: OtherArrayTree,
      *,
      lat: chex.Array | None = None,
      lon: chex.Array | None = None,
      **other_init_kwargs,
  ) -> "TriangularMeshData[OtherArrayTree]":
    if lat is None:
      lat = self.lat
    if lon is None:
      lon = self.lon
    if "mask" not in other_init_kwargs:
      other_init_kwargs["mask"] = self.mask
    if "metadata" not in other_init_kwargs:
      other_init_kwargs["metadata"] = self._metadata
    return TriangularMeshData(
        data=data,
        lat=lat,  # pyrefly: ignore[bad-argument-type]
        lon=lon,  # pyrefly: ignore[bad-argument-type]
        **other_init_kwargs,
        face_sets=self.face_sets,
    )

  @property
  def finest_faces(self) -> np.ndarray:
    return self._face_sets[-1][1]

  @property
  def face_sets(self) -> Sequence[tuple[str, np.ndarray]]:
    return self._face_sets


class ShareLeadingAxesArrayTree(Generic[ArrayTree]):
  """Basic data container class for spatial features, latents and outputs.

  A useful building block for data that will represent multiple modalities of
  spatial data. The only assumption of this class is that all arrays in the
  tree will have the same shape for the first `num_leading_shared_axes`
  dimensions, with the exception of size=1 which indicates broadcasting. The
  leading axes will typically be used to represent batch axes, or spatial axes.
  """

  def __init__(self, array_tree: ArrayTree, *, num_leading_shared_axes: int):
    """Inits the ShareLeadingAxesArrayTree.

    Args:
      array_tree: A tree of arrays with compatible leading shapes for the first
        `num_leading_shared_axes` dimensions. Leading shapes are compatible if
        the size of each axis is equal across all arrays in the tree, with the
        exception of size=1 which indicates broadcasting across the sizes of
        other arrays in the tree for that dimension.
      num_leading_shared_axes: The number of shared leading dimensions that
        should be compatible
    """
    self._array_tree = array_tree
    self._num_leading_shared_axes = num_leading_shared_axes
    self._shared_leading_shape = self._initialize_shared_leading_shape()

  @property
  def shared_leading_shape(self) -> tuple[int, ...]:
    """Returns leading shape with length `num_leading_shared_axes`."""
    return self._shared_leading_shape

  @property
  def array_tree(self) -> ArrayTree:
    return self._array_tree

  def _initialize_shared_leading_shape(self) -> tuple[int, ...]:
    """Infers the leading shape from the contents of the array tree."""

    # Start with ones for the output size, to iteratively broadcast the
    # ones to the sizes of the arrays we find in the flat list.
    output_shape = np.ones([self._num_leading_shared_axes], dtype=np.int64)

    flat_arrays = jax.tree_util.tree_flatten(self._array_tree)[0]
    for array in flat_arrays:
      if len(array.shape) < self._num_leading_shared_axes:
        raise ValueError(
            f"Data shape {array.shape} should have at least "
            f"`num_leading_shared_axes={self._num_leading_shared_axes}` "
            "dimensions."
        )
      this_shape = np.array(array.shape[: self._num_leading_shared_axes])

      # For anything for which the output shape is still one, the value of the
      # shape of this array takes precedence.
      output_shape = np.where(output_shape == 1, this_shape, output_shape)

      # At this point for each axis, either the output size must match the
      # size of this array, or the size of this array must be one (meaning that
      # it will be broacastable to the leading shape). Otherwise, the shape
      # is inconsistent.
      if np.any((output_shape != this_shape) & (this_shape != 1)):
        raise ValueError(
            "Inconsistent leading shapes for the first "
            f"{self._num_leading_shared_axes} axes: "
            f"{[array.shape for array in flat_arrays]}."
        )

    return tuple(map(int, tuple(output_shape)))


def _verify_lat_lon(spatial_data: SpatialData):
  """Verifies lat lon values are consistent with the spatial data."""
  lat = spatial_data.lat
  lon = spatial_data.lon

  num_point_dims = len(spatial_data.point_dims_shape)
  if lat.ndim != num_point_dims:
    raise ValueError(
        f"Latitudes must have rank <= {num_point_dims}. Got shape {lat.shape}."
    )
  if lon.ndim != num_point_dims:
    raise ValueError(
        f"Longitudes must have rank <= {num_point_dims}. Got shape {lon.shape}."
    )

  # Cannot check values on tracers.
  if (isinstance(lat, jax.core.Tracer) and
      isinstance(lon, jax.core.Tracer)):
    return

  min_lat = lat.min()
  max_lat = lat.max()
  min_lon = lon.min()
  max_lon = lon.max()
  if min_lat < -90.0 or max_lat > 90.0:
    raise ValueError(
        f"Latitudes must be in [-90, 90]. Got min_lat={min_lat},"
        f" max_lat={max_lat}."
    )

  if min_lon < 0.0 or max_lon >= 360.0:
    raise ValueError(
        f"Longitudes must be in [0, 360). Got min_lon={min_lon},"
        f" max_lon={max_lon}."
    )


def _get_broadcastable_lat_lon_coords_from_grid_xarray(
    data_array_dataset: xr.Dataset | xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns the lat/lon arrays with leading shape (batch, lat, lon)."""
  lat_coord = data_array_dataset.coords["lat"]
  lon_coord = data_array_dataset.coords["lon"]

  def _broadcast_coord(
      coord: xr.DataArray, name: str
  ) -> np.ndarray:
    """Validates dims and broadcasts a single coordinate to (batch, lat, lon)."""
    if coord.ndim == 1:
      if coord.dims != (name,):
        raise ValueError(
            f"Expected 1D coordinate '{name}' to have dims ('{name}',), "
            f"got {coord.dims}."
        )
      data = coord.data
      if name == "lat":
        return data[None, :, None]  # Broadcast batch and lon
      else:
        return data[None, None, :]  # Broadcast batch and lat
    elif coord.ndim == 2:
      if coord.dims != ("lat", "lon"):
        raise ValueError(
            f"Expected 2D coordinate '{name}' to have dims ('lat', 'lon'), "
            f"got {coord.dims}."
        )
      return coord.data[None, :, :]  # Broadcast batch
    else:
      raise ValueError(
          f"Unsupported coordinate '{name}' with {coord.ndim} dimensions "
          f"(shape {coord.shape}). Expected 1D or 2D."
      )

  lat = _broadcast_coord(lat_coord, "lat")
  lon = _broadcast_coord(lon_coord, "lon")
  return lat, lon


def _unstack(array: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
  return tuple(np.moveaxis(array, axis, 0))


def _batch_lat_lon_to_latlon_batch(
    array: chex.Array,
    lat_size: int,
    lon_size: int,
    major_axis: str,
    np_: NumpyInterface = np,
) -> chex.Array:
  """Changes shape from [batch, lat, lon, ...] to [lat*lon, batch, ...].

  Input array admits broadcasting on the lat/lon dimensions (e.g. if the
  size of "lat" is 1, it will be broadcasted to `lat_size`).

  Args:
    array: The array to reshape, with shape [batch, lat, lon, ...].
    lat_size: The expected size of the latitude dimension.
    lon_size: The expected size of the longitude dimension.
    major_axis: Which axis varies slowest in the flattened output. 'lat'
      (default) gives lat*lon ordering, 'lon' gives lon*lat ordering.
    np_: to be set to numpy or to jax.numpy.

  Returns:
    The reshaped array with shape [lat*lon, batch, ...].
  """
  assert major_axis in (
      "lat",
      "lon",
  ), f"major_axis must be lat or lon, got {major_axis}"

  # [batch, lat, lon, ...] -> [lat, lon, batch, ...]
  array = np_.moveaxis(array, 0, 2)

  if array.shape[0] == 1 and array.shape[1] == 1:
    lat_size = 1
    lon_size = 1

  lat_lon_broadcasted_shape = (lat_size, lon_size) + array.shape[2:]
  array = np_.broadcast_to(array, lat_lon_broadcasted_shape)

  if major_axis == "lon":
    # [lat, lon, batch, ...] -> [lon, lat, batch, ...]
    array = np_.moveaxis(array, 0, 1)
  else:
    assert major_axis == "lat"

  # [major, minor, batch, ...] -> [major * minor, batch, ...]
  return np_.reshape(array, [lat_size * lon_size] + list(array.shape[2:]))


def _latlon_batch_to_batch_lat_lon(
    array: chex.Array,
    lat_size: int,
    lon_size: int,
    major_axis: str,
    np_: NumpyInterface = np,
) -> chex.Array:
  """Changes shape from [lat*lon, batch, ...] to [batch, lat, lon, ...].

  Inverse of `_batch_lat_lon_to_latlon_batch`.

  Args:
    array: The array to reshape, with shape [lat*lon, batch, ...].
    lat_size: The size of the latitude dimension.
    lon_size: The size of the longitude dimension.
    major_axis: Which axis varies slowest in the flattened input. Must match the
      value used when creating the flattened representation.
    np_: to be set to numpy or to jax.numpy.

  Returns:
    The reshaped array with shape [batch, lat, lon, ...].
  """
  assert major_axis in (
      "lat",
      "lon",
  ), f"major_axis must be lat or lon, got {major_axis}"

  if array.shape[0] == 1:
    lat_size = 1
    lon_size = 1

  if major_axis == "lon":
    # [lon * lat, batch, ...] -> [lon, lat, batch, ...]
    array = np_.reshape(array, [lon_size, lat_size] + list(array.shape[1:]))
    # [lon, lat, batch, ...] -> [lat, lon, batch, ...]
    array = np_.moveaxis(array, 0, 1)
  else:
    assert major_axis == "lat"
    # [lat * lon, batch, ...] -> [lat, lon, batch, ...]
    array = np_.reshape(array, [lat_size, lon_size] + list(array.shape[1:]))

  # [lat, lon, batch, ...] -> [batch, lat, lon, ...]
  return np_.moveaxis(array, 2, 0)


def _expand_dims(
    array: chex.Array,
    axes: Sequence[int],
) -> chex.Array:
  """Compatible with numpy and jax arrays."""

  dims = list((slice(None),) * array.ndim)
  # Important to do it in order, so that the axes correspond to the locations
  # of the output array.
  for ax_i in sorted(axes):
    if ax_i < 0:
      raise ValueError(f"Negative axis {ax_i} not supported.")
    dims.insert(ax_i, None)  # pyrefly: ignore[bad-argument-type]
  return array[tuple(dims)]  # pyrefly: ignore[bad-index]


def _pad_data_impl(
    data: Mapping[str, ArrayTree],
    mask: np.ndarray,
    shared_leading_shape: Sequence[int],
    padding_axis: int,
    padding_amount: int,
    data_padding_values: Mapping[str, Any],
) -> Mapping[str, Any]:
  """Pads the data along the given axis.

  Args:
    data: A dictionary of data.
    mask: The mask.
    shared_leading_shape: The leading shape of the arrays.
    padding_axis: The axis to pad.
    padding_amount: The amount to pad.
    data_padding_values: A dictionary of padding values for each field in data.

  Returns:
    A dictionary of padded data in the same structure as data with additional
    field per_axis_mask.
  """
  if padding_amount == 0:
    return dict(mask=mask, **data)

  if padding_axis < 0:
    raise ValueError(f"Padding axis {padding_axis} must be non-negative.")

  if padding_axis < 0 or padding_axis >= len(shared_leading_shape):
    raise ValueError(
        f"Padding axis {padding_axis} must be in range [0, "
        f"{len(shared_leading_shape)})."
    )

  if set(data.keys()) != set(data_padding_values.keys()):
    raise ValueError(
        "Keys of `data` and `data_padding_values` must be the same. But got "
        f"{data.keys()=} and {data_padding_values.keys()=}."
    )

  def pad_array(
      array: np.ndarray | None,
      padding_value: Any,
      array_padding_axis: int,
  ) -> np.ndarray | None:
    if array is None:
      return None
    shared_axis_size = shared_leading_shape[padding_axis]
    if array.shape[array_padding_axis] == 1 and shared_axis_size != 1:
      # If axis is broadcastable, explicitly broadcast it.
      broadcast_shape = list(array.shape)
      broadcast_shape[array_padding_axis] = shared_axis_size
      array = np.broadcast_to(array, broadcast_shape)
    pad_width = [
        (0, 0),
    ] * array.ndim
    pad_width[array_padding_axis] = (0, padding_amount)
    return np.pad(array, pad_width, constant_values=padding_value)

  output = {
      key: jax.tree_util.tree_map(
          lambda x, key=key: pad_array(
              x,
              data_padding_values[key],
              array_padding_axis=padding_axis,
          ),
          value,
      )
      for key, value in data.items()
  }
  output["mask"] = pad_array(
      mask,
      padding_value=False,
      array_padding_axis=padding_axis,
  )
  return output


def _initialize_mask(
    mask: SingleArray | None,
    per_axis_mask: PerAxisMaskType | None,
    num_leading_dims: int) -> np.ndarray:
  """Returns the mask."""
  if mask is not None and per_axis_mask is not None:
    raise ValueError("At most one of `mask` and `per_axis_mask` can be passed.")

  if per_axis_mask is not None:
    mask = _mask_to_full_array(_simplify_per_axis_mask(per_axis_mask))
  elif mask is None:
    mask = np.ones([1] * num_leading_dims, dtype=bool)
  if mask.ndim != num_leading_dims:  # pyrefly: ignore[missing-attribute]
    raise ValueError(
        f"Mask must have rank {num_leading_dims}, got {mask.ndim}.")
  # TODO(alvarosg): Consider calling `_simplify_mask` here for numpy arrays.
  return mask  # pyrefly: ignore[bad-return]


def _simplify_mask(mask: np.ndarray) -> np.ndarray:
  """Collapses any dimensions with equal values into size 1."""
  for i in range(mask.ndim):
    any_reduce = np.any(mask, axis=i, keepdims=True)
    all_reduce = np.all(mask, axis=i, keepdims=True)
    # If the result of reducing with all or with any, is the same,
    # then we can squeeze the axis.
    if np.all(any_reduce == all_reduce):
      mask = any_reduce
  return mask


def _simplify_per_axis_mask(per_axis_mask: PerAxisMaskType) -> PerAxisMaskType:
  """Simplifies the mask by removing redundant True values."""
  # This removes unnecessary broadcasting to the mask.
  simplified_mask = []
  for axis_mask in per_axis_mask:
    if np.all(axis_mask):
      simplified_mask.append((True,))
    elif not np.any(axis_mask):
      simplified_mask.append((False,))
    else:
      simplified_mask.append(axis_mask)
  return tuple(simplified_mask)


def _mask_to_full_array(
    per_axis_mask: PerAxisMaskType,
) -> np.ndarray:
  """Broadcasts the per_axis_mask along the given axis."""
  num_axes = len(per_axis_mask)
  mask_array = np.ones([1] * num_axes, dtype=bool)
  for i, mask_axis in enumerate(per_axis_mask):
    new_shape = np.ones(num_axes, np.int32)
    new_shape[i] = len(mask_axis)
    if not np.all(mask_axis):
      # Don't broadcast axes that are all true.
      mask_array = mask_array * np.reshape(mask_axis, new_shape)
  return mask_array


def mask_invalid_points(
    array_tree: ArrayTree,
    mask_array: np.ndarray,
    mask_value: float = 0.0,
) -> ArrayTree:
  """Masks the invalid points from the array, according to the mask."""
  if isinstance(mask_array, np.ndarray) and np.all(mask_array):
    # No masking, just return the array, only possible if it is a static array.
    return array_tree

  def _mask_array_fn(array_: chex.Array) -> chex.Array:
    """Masks the invalid points from the array, according to the mask."""
    if mask_array.ndim > array_.ndim:
      raise ValueError(
          "Mask array has more dimensions than the array being masked. "
          f"{mask_array.shape=}, {array_.shape=}."
      )
    elif mask_array.ndim < array_.ndim:
      np_ = np if isinstance(mask_array, np.ndarray) else jnp
      mask_array_ = np_.expand_dims(
          mask_array, axis=tuple(range(mask_array.ndim, array_.ndim))
      )
    else:
      mask_array_ = mask_array

    # Only use numpy if both data and mask are numpy arrays.
    if (isinstance(mask_array_, np.ndarray) and
        isinstance(array_, np.ndarray)):
      np_ = np
    else:
      np_ = jnp

    return np_.where(mask_array_, array_, mask_value)

  return jax.tree.map(_mask_array_fn, array_tree)
