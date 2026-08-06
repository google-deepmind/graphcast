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
"""Utilities for gridding cyclone data."""

import collections
import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr


TOTAL_DEGREES_LAT = 180
HALF_DEGREES_LAT = TOTAL_DEGREES_LAT // 2
TOTAL_DEGREES_LON = 360
EARTH_CIRCUMFERENCE_KM = 40_075
KM_PER_NMILE = 1.852

# Define xarray shape for resulting data
GriddedDataArrayShape = collections.namedtuple(
    "GriddedDataArrayShape",
    [
        "single_level_temporal_variable",
        "time",
        "latitude",
        "longitude",
    ],
)


GridDataset = xr.Dataset
GriddingInfo = Dict[str, Any]


@dataclasses.dataclass
class GriddedData:

  zarr_key: Any
  grid_data: GridDataset
  datetime: Optional[np.datetime64] = None
  info: Optional[GriddingInfo] = None


def divides_with_tolerance(
    dividend: float,
    divisor: float,
    atol: float = 1e-6,
) -> bool:
  """Checks that dividend is divisible by divisor within tolerance.

  Args:
    dividend: dividend to check.
    divisor: divisor to check.
    atol: absolute tolerance to use for comparison.

  Returns:
    True if dividend is divisible by divisor within tolerance, False otherwise.
  """
  remainder = dividend % divisor
  close_to_zero = np.isclose(remainder, 0, atol=atol)
  close_to_divisor = np.isclose(remainder, divisor, atol=atol)
  return close_to_zero or close_to_divisor  # pyrefly: ignore[bad-return]


def num_lat_rectilinear_grid(resolution: float) -> int:
  """Number of latitude points for a rectilinear grid of given resolution."""
  if not divides_with_tolerance(HALF_DEGREES_LAT, resolution):
    raise ValueError(
        f"Total degrees {HALF_DEGREES_LAT} must be divisible by resolution"
        f" {resolution} within tolerance 1e-6."
    )
  return 2 * np.round(HALF_DEGREES_LAT / resolution).astype(int) + 1


def num_lon_rectilinear_grid(resolution: float) -> int:
  """Number of longitude points for a rectilinear grid of given resolution."""
  if not divides_with_tolerance(TOTAL_DEGREES_LON, resolution):
    raise ValueError(
        f"Total degrees {TOTAL_DEGREES_LON} must be divisible by resolution"
        f" {resolution} within tolerance 1e-6."
    )
  return np.round(TOTAL_DEGREES_LON / resolution).astype(int)


def create_grid_lat_and_lon_dimensions(
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
  """Creates lat and lon dimensions with given resolution.

  Note that the longitude dimensions returned are in the [0., 360.) range.

  Args:
    resolution: grid resolution to use

  Returns:
    grid of latlon points with specified resolution, shape (num_lat, num_lon, 2)
  """
  num_points_lat = num_lat_rectilinear_grid(resolution)
  num_points_lon = num_lon_rectilinear_grid(resolution)

  # Check that the number of latlon points is odd, which should be the case
  # whenever we are using a global grid with poles included.
  if num_points_lat % 2 != 1:
    raise AssertionError(
        f"Number of latitude points {num_points_lat} must be odd, but were "
        f"{num_points_lat}, which should never happen."
    )

  lats = np.linspace(-90.0, 90, num_points_lat)
  lons = np.linspace(0, 360, num_points_lon, endpoint=False)

  return lats, lons


def create_grid_latlons(resolution: float) -> np.ndarray:
  """Creates grid of latlon points with given resolution.

  Args:
    resolution: grid resolution to use

  Returns:
    grid of latlon points with specified resolution, shape (num_lat, num_lon, 2)
  """
  lats, lons = create_grid_lat_and_lon_dimensions(resolution=resolution)
  lons, lats = np.meshgrid(lons, lats)

  return np.stack([lats, lons], axis=-1)


def convert_grid_data_to_packed_xarray_dataset(
    datetime: np.datetime64,
    grid_data: Dict[str, np.ndarray],
    resolution_degrees: float,
    variable_prefix: Optional[str] = None,
) -> xr.Dataset:
  """Converts a dict of gridded data to a packed xarray dataset.

  Args:
    datetime: datetime of the data.
    grid_data: dict of gridded data, where keys are variable names and values
      are numpy arrays.
    resolution_degrees: resolution of the gridded data in degrees.
    variable_prefix: prefix to add to the variable names.

  Returns:
    xarray dataset of the gridded data.
  """

  # Sort variable names alphabetically for standardization.
  variable_names = sorted(grid_data.keys())

  # Stack data according to variable name and expand 1-long time dimension.
  sorted_stacked_data = np.stack(
      [grid_data[variable_name] for variable_name in variable_names],
      axis=0,
  )
  sorted_stacked_data = sorted_stacked_data[:, None, :, :]

  # Add cyclone prefix to variable names to make the variables easier to
  # distinguish from other atmospheric variables if, in the future, we merge
  # the cyclone variables with the atmospheric variables.
  prefix = "" if not variable_prefix else f"{variable_prefix}_"
  variable_names_with_prefix = list(
      f"{prefix}{variable_name}" for variable_name in variable_names
  )

  # Get grid latitudes and longitudes for given resolution
  grid_lat, grid_lon = create_grid_lat_and_lon_dimensions(
      resolution=resolution_degrees,
  )

  # Create data xarray.
  grid_data_xarray = xr.DataArray(
      sorted_stacked_data,
      coords={
          "single_level_temporal_variable": variable_names_with_prefix,
          "time": datetime[None],  # pyrefly: ignore[bad-index]
          "latitude": grid_lat,
          "longitude": grid_lon,
      },
      dims=[
          "single_level_temporal_variable",
          "time",
          "latitude",
          "longitude",
      ],
  )

  grid_data_xarray["single_level_temporal_variable"] = grid_data_xarray[
      "single_level_temporal_variable"
  ].astype(object)

  return xr.Dataset({"single_level_temporal": grid_data_xarray})
