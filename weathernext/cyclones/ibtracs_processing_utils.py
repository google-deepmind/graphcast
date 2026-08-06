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
"""Utilities for processing IBTrACS data for cyclone modelling."""

from collections.abc import Mapping, Sequence
import copy
import functools
import itertools
from typing import Callable, Dict, List, Optional, Tuple

from weathernext.cyclones import cyclone_utils as cyclones_utils
from weathernext.cyclones import data_pipeline_utils as cyclones_data_utils
import numpy as np
from sklearn import linear_model
import xarray as xr


def _get_ibtracs_quadrant_variable_tuples() -> (
    Tuple[Tuple[str, str, float], ...]
):
  agencies = ("usa", "bom", "reunion")
  threshold_speeds = (34.0, 50.0, 64.0)
  return tuple(
      (f"{source}_r{int(threshold)}", f"{source}_wind", threshold)
      for source, threshold in itertools.product(agencies, threshold_speeds)
  )


IBTrACSArray = np.ndarray
IBTrACSDataSlice = Dict[str, IBTrACSArray]

LinearRegressionModel = Callable[[np.ndarray], np.ndarray]
WindCalibrationModel = Callable[[IBTrACSDataSlice], np.ndarray]


# Names of wind variables under IBTrACS dataset
IBTRACS_WIND_VARIABLES = (
    "usa_wind",
    "cma_wind",
    "newdelhi_wind",
    "reunion_wind",
    "bom_wind",
)

ALL_WIND_VARIABLE = "all_wind"
IBTRACS_WIND_VARIABLES_WITH_ALL_WIND = IBTRACS_WIND_VARIABLES + (
    ALL_WIND_VARIABLE,
)


# Sources of max wind speed radii
IBTRACS_MAX_WIND_RADIUS_VARIABLES = (
    "usa_rmw",
    "reunion_rmw",
    "bom_rmw",
)

# Sources of minimum sea level pressure
IBTRACS_MIN_SEA_LEVEL_PRESSURE_VARIABLES = (
    "wmo_pres",
    "usa_pres",
    "cma_pres",
    "hko_pres",
    "tokyo_pres",
    "newdelhi_pres",
    "reunion_pres",
    "bom_pres",
    "nadi_pres",
    "wellington_pres",
)

# Tuples of quadrant radius variable source, corresponding winds speed variable
# source and corresponding quadrant threshold wind speed in knots.
IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLE_TUPLES = (
    _get_ibtracs_quadrant_variable_tuples()
)

# Names of quadrant max. wind radius variables under IBTrACS dataset
IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLES = tuple(
    name for name, _, _ in IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLE_TUPLES
)

# Aggregated latitude and longitude IBTrACS variables
IBTRACS_AGGREGATED_LAT_LON_VARIABLES = (
    "lat",
    "lon",
)

# Agencies from which we want to retrieve longitudes and latitudes in addition
# to the aggregated variables.
IBTRACS_LAT_LON_AGENCIES = (
    "usa",
    "cma",
    "hko",
    "tokyo",
    "newdelhi",
    "reunion",
    "bom",
    "nadi",
    "wellington",
)

IBTRACS_PER_AGENCY_LAT_VARIABLES = tuple(
    f"{agency}_lat" for agency in IBTRACS_LAT_LON_AGENCIES
)

IBTRACS_PER_AGENCY_LON_VARIABLES = tuple(
    f"{agency}_lon" for agency in IBTRACS_LAT_LON_AGENCIES
)

# Interleave variables so lat and lon appear next to each other for each source
IBTRACS_PER_AGENCY_LAT_AND_LON_VARIABLES = tuple(
    sorted(IBTRACS_PER_AGENCY_LAT_VARIABLES + IBTRACS_PER_AGENCY_LON_VARIABLES)
)

# Category IBTrACS variables
IBTRACS_CATEGORY_VARIABLES = ("usa_sshs",)

# Full list of variables in the IBTrACS dataset
IBTRACS_VARIABLES = (
    IBTRACS_AGGREGATED_LAT_LON_VARIABLES
    + IBTRACS_PER_AGENCY_LAT_AND_LON_VARIABLES
    + IBTRACS_WIND_VARIABLES
    + IBTRACS_MIN_SEA_LEVEL_PRESSURE_VARIABLES
    + IBTRACS_MAX_WIND_RADIUS_VARIABLES
    + IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLES
    + IBTRACS_CATEGORY_VARIABLES
)


# Base wind variable to calibrate against
IBTRACS_BASE_WIND_VARIABLE = "usa_wind"

IBTRACS_RESOLUTION_MINUTES = 180
IBTRACS_RESOLUTION_HOURS = IBTRACS_RESOLUTION_MINUTES // 60
VALID_GAUSSIAN_DISC_NORMALIZATION_METHODS = (
    "probability",
    "unit_mode",
)

# Flags to use for marking different imputation results for quadrant radii
QUADRANT_PRESENT_AND_NOT_IMPUTED_FLAG = "present_and_not_imputed"
QUADRANT_MISSING_IMPUTED_WITH_MEAN_FLAG = "missing_imputed_with_mean"
QUADRANT_MISSING_IMPUTED_WITH_ZERO_FLAG = "missing_imputed_with_zero"
QUADRANT_MISSING_NOT_IMPUTED_FLAG = "missing_not_imputed"

QUADRANT_PRESENT_OR_IMPUTED_FLAGS = (
    QUADRANT_PRESENT_AND_NOT_IMPUTED_FLAG,
    QUADRANT_MISSING_IMPUTED_WITH_MEAN_FLAG,
    QUADRANT_MISSING_IMPUTED_WITH_ZERO_FLAG,
)

QUADRANT_NAMES = (
    "ne",  # north-east
    "se",  # south-east
    "sw",  # south-west
    "nw",  # north-west
)

QUADRANT_RADIUS_VARIABLES = tuple()  # pylint: disable=invalid-name
for _quadrant_radius_source in IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLES:
  for _quadrant in QUADRANT_NAMES:
    QUADRANT_RADIUS_VARIABLES += (
        f"{_quadrant_radius_source}_{_quadrant}_radius",
    )

CORE_CYCLONE_VARIABLES = [
    "cyclone_exists_gaussian_unit_mode",
    "cyclone_usa_pres_disc",
    "cyclone_usa_r34_ne_radius_disc",
    "cyclone_usa_r34_nw_radius_disc",
    "cyclone_usa_r34_se_radius_disc",
    "cyclone_usa_r34_sw_radius_disc",
    "cyclone_usa_r50_ne_radius_disc",
    "cyclone_usa_r50_nw_radius_disc",
    "cyclone_usa_r50_se_radius_disc",
    "cyclone_usa_r50_sw_radius_disc",
    "cyclone_usa_r64_ne_radius_disc",
    "cyclone_usa_r64_nw_radius_disc",
    "cyclone_usa_r64_se_radius_disc",
    "cyclone_usa_r64_sw_radius_disc",
    "cyclone_usa_rmw_disc",
    "cyclone_usa_wind_disc",
    "cyclone_all_wind_disc",
]


def filter_variables_for_reduced_dataset(
    variables: Sequence[str],
) -> List[str]:
  filtered: List[str] = []
  # These variables are strings as made available in IBTrACS, e.g. usa_wind.
  for variable in variables:
    for full_variable_name in CORE_CYCLONE_VARIABLES:
      if variable in full_variable_name:
        filtered.append(variable)
        break
  return filtered


def get_cyclone_variable_names() -> List[str]:
  """Returns list of all cyclone variables generated by the gridification stage.

  Returns:
    variable names as a list of strings, in sorted alphabetical order, each
      with the prefix "cyclone_" appended.
  """

  # Cyclone existence variables created using the lat and lon variables.
  variable_names = [
      "exists_sparse",
      "exists_disc",
      "exists_gaussian_unit_mode",
      "exists_gaussian_probability",
  ]

  # Cyclone existence variables by agency.
  for agency in IBTRACS_LAT_LON_AGENCIES:
    for variable_suffix in (
        "exists_sparse",
        "exists_disc",
        "exists_gaussian_unit_mode",
        "exists_gaussian_probability",
    ):
      variable_names.append(f"{agency}_{variable_suffix}")

  # Scalar variables
  scalar_variables = (
      IBTRACS_WIND_VARIABLES
      + IBTRACS_MIN_SEA_LEVEL_PRESSURE_VARIABLES
      + IBTRACS_MAX_WIND_RADIUS_VARIABLES
      + QUADRANT_RADIUS_VARIABLES
  )

  for variable in scalar_variables:
    variable_names.append(f"{variable}_sparse")
    variable_names.append(f"{variable}_disc")

  # Category variables
  category_variables = IBTRACS_CATEGORY_VARIABLES
  for variable in category_variables:
    variable_names.append(f"{variable}_sparse")
    variable_names.append(f"{variable}_disc")

  # Quadrant shape variables
  for variable in IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLES:
    variable_names.append(f"{variable}_shape")

  return sorted(f"cyclone_{variable_name}" for variable_name in variable_names)


def is_east_of(latlons0: np.ndarray, latlons1: np.ndarray) -> np.ndarray:
  """Compares latlon points longitude-wise.

  Args:
    latlons0: first set of points to compare, shape (N, 2)
    latlons1: second set of points to compare, shape (N, 2)

  Returns:
    numpy array of bool, True if latlon0 is east of latlon1, False otherwise.
  """
  lon_diff = latlons0[..., 1] - latlons1[..., 1]
  return ((lon_diff >= 0.0) & (lon_diff < 180.0)) | (lon_diff < -180.0)


def is_north_of(latlons0: np.ndarray, latlons1: np.ndarray) -> np.ndarray:
  """Compares latlon points latitude-wise.

  Args:
    latlons0: first set of points to compare, shape (N, 2)
    latlons1: second set of points to compare, shape (N, 2)

  Returns:
    numpy array of bool, True if latlon0 is north of latlon1, False otherwise.
  """
  return (latlons0[..., 0] - latlons1[..., 0]) >= 0.0


def create_grid_with_fill(resolution: float, fill_value: float) -> np.ndarray:
  """Creates a grid with given resolution and fill value.

  Args:
    resolution: grid resolution to use.
    fill_value: fill value to use for the grid.

  Returns:
    grid filled with fill_value with specified resolution
  """
  num_points_lat = cyclones_data_utils.num_lat_rectilinear_grid(resolution)
  num_points_lon = cyclones_data_utils.num_lon_rectilinear_grid(resolution)

  return fill_value * np.ones(shape=(num_points_lat, num_points_lon))


def create_global_lat_lon_grid_coordinates(resolution: float) -> np.ndarray:
  """Creates array of latlon coords for the entire Earth, with given resolution.

  Args:
    resolution: grid resolution to use

  Returns:
    array of latlon coordinates with given resolution
  """
  num_points_lat = cyclones_data_utils.num_lat_rectilinear_grid(resolution)
  num_points_lon = cyclones_data_utils.num_lon_rectilinear_grid(resolution)

  lons, lats = np.meshgrid(
      np.arange(num_points_lon) * resolution,
      np.arange(num_points_lat) * resolution - 90.0,
  )

  return np.stack([lats, lons], axis=-1)


def create_sparse_nearest_neighbour_mask_from_latlon(
    latlons: np.ndarray,
    resolution: float,
    values: Optional[np.ndarray] = None,
    fill_value: float = 0.0,
) -> np.ndarray:
  """Create a sparse array of one-hot grid points.

  Args:
    latlons: latitude and longitude coordinates of shape (N, 2) from which to
      create sparse nearest neighbour mask.
    resolution: grid resolution to use.
    values: optional values with which to set sparse entries. If None, then it
      defaults to 1 for all latlon pairs. If passed, it must have shape (N,).
    fill_value: value with which to fill entries other than the sparsely
      populated entries.

  Returns:
    sparse mask created from latlons and values.
  """

  values = np.ones(latlons.shape[0]) if values is None else values
  assert latlons.shape[0] == values.shape[0]

  result = create_grid_with_fill(fill_value=fill_value, resolution=resolution)

  grid_latlons = create_global_lat_lon_grid_coordinates(resolution=resolution)
  grid_shape = grid_latlons.shape[:2]
  grid_latlons = grid_latlons.reshape(-1, 2)

  distances = cyclones_utils.geodesic_distance(
      latlon0=latlons[:, None, :],
      latlon1=grid_latlons[None, :, :],
  )  # (num_latlons, num_grid)

  result = result.reshape(-1)
  result[np.argmin(distances, axis=1)] = values
  result = result.reshape(grid_shape)

  return result


def create_gaussian_disc_mask_from_latlon(
    latlons: np.ndarray,
    resolution: float,
    disc_radius_km: float,
    values: Optional[np.ndarray] = None,
    normalization: str = "unit_mode",
) -> np.ndarray:
  """Create Gaussian disc mask from latlons.

  Args:
    latlons: latlon coordinates for center of Gaussian, shape (N, 2)
    resolution: resolution of the resulting grid
    disc_radius_km: radius of the Gaussian (i.e. Gaussian scale)
    values: (optional) values by which to multiply Gaussians, shape (N,)
    normalization: normalization mode which is either "unit_mode" (in which case
      each Gaussian bump has unit mode, prior to being multiplied by values) or
      "probability" (in which case each Gaussian bump is normalized to be a
      valid probability distribution over the grid).

  Returns:
    mask grid containing Gaussian discs.
  """

  values = np.ones(latlons.shape[0]) if values is None else values
  assert latlons.shape[0] == values.shape[0]
  assert normalization in VALID_GAUSSIAN_DISC_NORMALIZATION_METHODS

  grid_latlon = create_global_lat_lon_grid_coordinates(resolution=resolution)
  grid_shape = grid_latlon.shape[:2]
  grid_latlon = grid_latlon.reshape(-1, 2)

  distances = cyclones_utils.geodesic_distance(
      latlon0=latlons[:, None, :],
      latlon1=grid_latlon[None, :, :],
  )  # (num_latlons, num_grid)

  gaussian = np.exp(
      -0.5 * (distances / disc_radius_km) ** 2.0
  )  # (num_latlon, num_grid)

  if normalization == "probability":
    gaussian = gaussian / np.sum(gaussian, axis=1, keepdims=True)

  elif normalization != "unit_mode":
    raise ValueError(
        "normalization must be either 'probability' or 'unit_mode', found "
        f"was {normalization=}"
    )

  gaussian = gaussian * values[:, None]

  return gaussian.sum(axis=0).reshape(grid_shape)


def _is_in_quadrants(
    latlons: np.ndarray,
    center_latlon: np.ndarray,
    quadrant_radii_in_km: np.ndarray,
    margin_km: float = 200.0,
) -> np.ndarray:
  """Checks if latlons are in quadrants defined by center_latlon and radii.

  Args:
    latlons: latlon coordinates of points to check, shape (num_latlon, 2)
    center_latlon: latlon coordinates for center point, shape (2,)
    quadrant_radii_in_km: radii of the four quadrants in km in the order NE, SE,
      SW, NW, shape (4,).
    margin_km: additional margin to add for numerical purposes.

  Returns:
    boolean np.ndarray of shape (num_latlon,) indicating whether each of the
    points in latlons is in one of the four quadrants defined by center_latlon
    and quadrant_radii.
  """
  assert center_latlon.shape == (2,)
  assert len(latlons.shape) == 2 and latlons.shape[1] == 2
  assert quadrant_radii_in_km.shape == (4,)

  # Compute distances between input points and grid points.
  distance = cyclones_utils.geodesic_distance(
      latlon0=center_latlon[None, :],
      latlon1=latlons,
  )  # (1, num_latlons)

  # Compute binary masks specifying whether each grid point is to the east or
  # north of the center latlon point.
  is_east = is_east_of(latlons, center_latlon)
  is_north = is_north_of(latlons, center_latlon)

  # Strict inequality: points exactly at the boundary are excluded.
  is_in_quadrant_and_radius_plus_margin = (
      distance[None, :] < quadrant_radii_in_km[:, None] + margin_km
  )  # shape (4, num_latlons)

  # The order of the quadrants in IBTrACS is NE first and then clockwise, that
  # is NE, SE, SW and NW. We treat each one separately and then aggregate.
  is_in_shape = (
      (is_in_quadrant_and_radius_plus_margin[0] & is_east & is_north)
      | (is_in_quadrant_and_radius_plus_margin[1] & is_east & ~is_north)
      | (is_in_quadrant_and_radius_plus_margin[2] & ~is_east & ~is_north)
      | (is_in_quadrant_and_radius_plus_margin[3] & ~is_east & is_north)
  )  # shape (num_latlons,)

  return is_in_shape


def create_quadrant_integral_mask_from_latlons(
    latlons: np.ndarray,
    quadrant_radii: np.ndarray,
    resolution: float,
    num_subgrid: int = 4,
    margin_deg: Optional[float] = None,
    values: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Creates quadrant mask from latlons and quadrant radii.

  Args:
    latlons: latlon centers of the quadrant origins, shape (N, 2).
    quadrant_radii: radius for each quadrant in km, shape (N, 4). The order of
      the quadrants is north-east and then clockwise, i.e. NE, SE, SW, NW.
    resolution: grid resolution to use.
    num_subgrid: number of points in each side of the subgrid.
    margin_deg: margin applied to ensure all relevant grid points are selected.
      If None, it is set to twice the resolution.
    values: values to multiply quadrant mask by, shape (N,). If None, it
      defaults to 1.

  Returns:
    quadrant mask as numpy ndarray.
  """
  # Set default values if necessary and check shapes.
  values = np.ones(latlons.shape[0]) if values is None else values
  assert latlons.shape[0] == quadrant_radii.shape[0]
  assert latlons.shape[0] == values.shape[0]

  # Set up grid latlons and save grid shape for reshaping later.
  grid_latlon = create_global_lat_lon_grid_coordinates(resolution=resolution)
  grid_latlon = grid_latlon.reshape(-1, 2)

  # Default width of margin around quadrants is the twice resolution in degrees.
  margin_deg = 2.0 * resolution if margin_deg is None else margin_deg

  # Compute quadrant interpolation margin in km.
  margin_km = (
      margin_deg
      / cyclones_data_utils.TOTAL_DEGREES_LON
      * cyclones_data_utils.EARTH_CIRCUMFERENCE_KM
  )

  # Create interpolated array to store results.
  result = create_grid_with_fill(resolution=resolution, fill_value=0.0)
  result_shape = result.shape

  # Loop over distances corresponding to latlon entry and compute integrals.
  for center_latlon, radii, value in zip(
      latlons,
      quadrant_radii,
      values,
  ):

    # Get indices of points that are within quadrants (plus_margin).
    idx_in_quadrants = np.where(
        _is_in_quadrants(
            latlons=grid_latlon,
            center_latlon=center_latlon,
            quadrant_radii_in_km=radii,
            margin_km=margin_km,
        )
    )
    # Get latlon coords of grid points within quadrants (plus margin).
    latlon_in_quadrant = grid_latlon[idx_in_quadrants]  # shape (num_in_quad, 2)

    # Half-width of a sub-grid of side length `resolution`, made up of
    # `num_subgrid` equispaced points on each side.
    w = 0.5 * resolution * (num_subgrid - 1) / num_subgrid

    # Latlon coordinate deltas for subgrid points.
    subgrid_latlon_deltas = np.stack(
        np.meshgrid(
            np.linspace(-w, w, num_subgrid),
            np.linspace(-w, w, num_subgrid),
        ),
        axis=-1,
    )  # shape (num_subgrid, num_subgrid, 2)

    # Latlon coordinates of subgrids are the latlon_in_quadrant points (i.e.
    # the center points on each latlon box) plus subgrid displacement deltas.
    latlon_subgrid = (
        subgrid_latlon_deltas[None, ...] + latlon_in_quadrant[:, None, None, :]
    )  # shape (num_in_quad, num_subgrid, num_subgrid, 2)

    # Check how many of the subgrid points are in the quadrants.
    is_subgrid_in_quadrant = _is_in_quadrants(
        latlons=latlon_subgrid.reshape(-1, 2),
        center_latlon=center_latlon,
        quadrant_radii_in_km=radii,
        margin_km=0.0,
    ).reshape(latlon_subgrid.shape[:-1])

    # Value of the integral is equal to the mean number of subgrid points that
    # are in the quadrants also. Increasing the number of subgrid points
    # improves the estimate of the integral, but this accuracy is not critical.
    integral_values = np.mean(is_subgrid_in_quadrant, axis=(1, 2))

    # Update result array with contribution from the current set of quadrants.
    result = result.reshape(-1)
    result[idx_in_quadrants] += integral_values * value
    result = result.reshape(result_shape)

  return result


def create_disc_integral_mask_from_latlon(
    latlons: np.ndarray,
    resolution: float,
    disc_radius_km: float,
    num_subgrid: int = 4,
    margin_deg: Optional[float] = None,
    values: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Create interpolation disc mask from latlons.

  Args:
    latlons: latlon centers of the quadrant origins, shape (N, 2).
    resolution: grid resolution to use.
    disc_radius_km: radius of the disc to place at each latlon point.
    num_subgrid: number of points in each side of the subgrid.
    margin_deg: margin over which the mask is interpolated. If None, it defaults
      to 2. * resolution to ensure there is some decay present.
    values: values to multiply quadrant mask by, shape (N,). If None, it
      defaults to 1.

  Returns:
    disc mask as numpy.ndarray.
  """
  quadrant_radii = np.ones((latlons.shape[0], 4)) * disc_radius_km
  return create_quadrant_integral_mask_from_latlons(
      latlons=latlons,
      quadrant_radii=quadrant_radii,
      resolution=resolution,
      num_subgrid=num_subgrid,
      margin_deg=margin_deg,
      values=values,
  )


def get_common_entries_without_nans(
    data_array1: np.ndarray,
    data_array2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns entries that are non-nan in both data arrays.

  Args:
    data_array1: The first data array.
    data_array2: The second data array.

  Returns:
    A tuple of data arrays that are non-nan in both data arrays.
  """
  idx_common = np.where(~np.isnan(data_array1) & ~np.isnan(data_array2))
  return data_array1[idx_common], data_array2[idx_common]


def impute_quadrant_radii_nans(
    max_wind_speeds: np.ndarray,
    quadrant_radii: np.ndarray,
    threshold_speed_knots: float,
) -> Tuple[np.ndarray, List[str]]:
  """Imputes quadrant radii if applicable.

  Args:
    max_wind_speeds: maximum sustained wind speeds, shape (N,)
    quadrant_radii: quadrant radii, shape (N, 4)
    threshold_speed_knots: speed threshold for the given quadrants

  Returns:
    imputed quadrant radii and boolean mask that marks missingness.
  """

  assert len(max_wind_speeds.shape) == 1
  assert len(quadrant_radii.shape) == 2
  assert quadrant_radii.shape[1] == 4
  assert max_wind_speeds.shape[0] == quadrant_radii.shape[0]
  assert np.all(~np.isnan(max_wind_speeds))

  quadrant_radii = copy.deepcopy(quadrant_radii)
  imputation_flags = []

  # Loop over quadrants, i.e. loop over shape (4,) numpy.ndarrays of quadrant
  # radii. We first handle cases (1) and (2). NB: np.nanmean returns nan if all
  # entries are nan.
  for i, radii in enumerate(quadrant_radii):

    if np.all(~np.isnan(quadrant_radii)):
      imputation_flag = QUADRANT_PRESENT_AND_NOT_IMPUTED_FLAG

    else:
      quadrant_radii[i, np.isnan(radii)] = np.nanmean(radii)
      imputation_flag = QUADRANT_MISSING_IMPUTED_WITH_MEAN_FLAG

    quadrants_have_nans = np.any(np.isnan(quadrant_radii[i]))

    # Handle case (3).
    if quadrants_have_nans and (max_wind_speeds[i] >= threshold_speed_knots):
      quadrant_radii[i] = 0.0
      imputation_flag = QUADRANT_MISSING_IMPUTED_WITH_ZERO_FLAG

    elif quadrants_have_nans and (max_wind_speeds[i] < threshold_speed_knots):
      quadrant_radii[i] = 0.0
      imputation_flag = QUADRANT_MISSING_NOT_IMPUTED_FLAG

    imputation_flags.append(imputation_flag)

  return quadrant_radii, imputation_flags


def create_discs_centered_on_latlons(
    latlons: np.ndarray,
    resolution: float,
    disc_radius_km: float,
    ignore_latlons: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Creates a mask grid that is zero everywhere except close to latlons.

  Args:
    latlons: latlons around which to place ones.
    resolution: resolution of resulting masked grid.
    disc_radius_km: radius of each disc of ones.
    ignore_latlons: latlons to ignore.
    values: values to multiply quadrant mask by, shape (N,). If None, it
      defaults to 1.

  Returns:
    resulting mask of zeros and ones as numpy nd.array.
  """

  values = np.ones(latlons.shape[0]) if values is None else values
  assert latlons.shape[0] == values.shape[0]

  ignore_latlons = (
      np.zeros(latlons.shape[0]).astype(bool)
      if ignore_latlons is None
      else ignore_latlons
  )
  assert len(latlons.shape) == 2
  assert latlons.shape[1] == 2
  assert latlons.shape[0] == ignore_latlons.shape[0]

  grid_latlons = create_global_lat_lon_grid_coordinates(resolution=resolution)
  grid_shape = grid_latlons.shape[:2]
  grid_latlons = grid_latlons.reshape(-1, 2)

  if ignore_latlons.shape[0] > 0:
    latlons = latlons[~ignore_latlons]
    values = values[~ignore_latlons]

  distances = cyclones_utils.geodesic_distance(
      latlon0=latlons[:, None, :],
      latlon1=grid_latlons[None, :, :],
  ).reshape(
      -1, *grid_shape
  )  # shape (num_latlons, num_grid_lat, num_grid_lon)

  # Create array which has zeros outside discs and values within discs
  values = values[:, None, None]
  per_latlon_grids = np.where(
      distances < disc_radius_km,
      values,
      np.zeros_like(values),
  )

  # Append array of zeros to per_latlon_grids, so max defaults to zero, and we
  # gracefully handle the case where there's no latlons to process.
  zeros = np.zeros(shape=(1,) + grid_shape)
  per_latlon_grids = np.concatenate([per_latlon_grids, zeros])

  return np.max(per_latlon_grids, axis=0)


def _create_wind_speed_linear_calibration_models(
    ibtracs_data: xr.Dataset,
    wind_calibration_start_date: str,
    wind_calibration_end_date: str,
    wind_calibration_skip_if_no_overlap: bool,
    ibtracs_base_wind_variable: str = IBTRACS_BASE_WIND_VARIABLE,
) -> Mapping[str, LinearRegressionModel]:
  """Create linear calibration models for converting windspeeds across agencies.

  The calibration models consist of a linear model followed by a clipping
  operation which ensures that the calibrated wind speeds are >= 0.

  Args:
    ibtracs_data: IBTrACS data, as loaded by the helper
      cyclone_analysis_utils.load_cyclone_tracks.
    wind_calibration_start_date: start date of the wind calibration model.
    wind_calibration_end_date: end date of the wind calibration model.
    wind_calibration_skip_if_no_overlap: whether to skip wind calibration for
      sources that have no overlapping datapoints with base source. If there
      is no overlap and this option is set to False, then an error is thrown.
    ibtracs_base_wind_variable: wind variable to calibrate against

  Returns:
    A dictionary of linear calibration models.
  """

  # Filter storms by start and end dates
  def _filter_storms(
      data_array: xr.DataArray,
  ) -> np.ndarray:
    """Filters storms by start and end dates, returning values in the range.

    Args:
      data_array: The data array to filter.

    Returns:
      Filtered numpy ndarray.
    """
    start_time = data_array.start_time.astype("datetime64[s]")
    end_time = data_array.end_time.astype("datetime64[s]")
    return (
        data_array.where(
            (start_time >= np.datetime64(wind_calibration_start_date))
            & (end_time <= np.datetime64(wind_calibration_end_date))
        )
        .transpose("storm", "date_time")
        .values
    )

  def _linear_regression_with_clipping(
      x: np.ndarray,
      coefficient: np.ndarray,
      intercept: np.ndarray,
  ) -> np.ndarray:
    return np.maximum(coefficient * x + intercept, 0.0)

  base_wind_data = _filter_storms(ibtracs_data[ibtracs_base_wind_variable])
  linear_regression_models = dict()

  for wind_variable in IBTRACS_WIND_VARIABLES:
    other_wind_data = _filter_storms(ibtracs_data[wind_variable])

    # Take common storm entries for base and other data. If the two wind sources
    # have no common elements, the common_base_wind and common_other_wind arrays
    # will have no data, i.e. their leading dimensions will be zero. We catch
    # this error and give a more informative message to the user.
    common_base_wind, common_other_wind = get_common_entries_without_nans(
        base_wind_data,
        other_wind_data,
    )

    if common_base_wind.shape[0] > 0:
      # Create linear regression model and fit to data. It's important to use
      # float64 here to ensure the accuracy of the underlying linear solves.
      linear_regression = linear_model.LinearRegression().fit(
          common_base_wind[:, None].astype(np.float64),
          common_other_wind.astype(np.float64),
      )
      coefficient = float(linear_regression.coef_.squeeze())
      intercept = float(linear_regression.intercept_.squeeze())

    elif wind_calibration_skip_if_no_overlap:
      coefficient = 1.0
      intercept = 0.0

    else:
      raise ValueError(
          f"Could not build wind calibration model because {wind_variable} "
          f"(variable to be calibrated) and {ibtracs_base_wind_variable} (base "
          f"variable) have no common data from {wind_calibration_start_date=} "
          f"to {wind_calibration_end_date=}."
      )

    # Wrap linear regression with a max to ensure wind-speeds are non-negative.
    linear_regression_models[wind_variable] = functools.partial(
        _linear_regression_with_clipping,
        coefficient=coefficient,
        intercept=intercept,
    )

  return linear_regression_models


def create_wind_speed_calibration_and_aggregation_model(
    ibtracs_data: xr.Dataset,
    wind_calibration_start_date: str,
    wind_calibration_end_date: str,
    wind_calibration_skip_if_no_overlap: bool,
) -> WindCalibrationModel:
  """Create model aggregating agencies' wind speeds (after calibrating them).

  This model is used to aggregate wind speeds across agencies into a single
  product. To achieve this, it first calibrates the wind speeds against the wind
  speed from a base source. Then, whenever wind speeds are available for a given
  cyclone, the model aggregates them by taking a simple mean of the ensemble.

  Args:
    ibtracs_data: IBTrACS data, as loaded by the helper
      cyclone_analysis_utils.load_cyclone_tracks.
    wind_calibration_start_date: Start date of the wind calibration model.
    wind_calibration_end_date: End date of the wind calibration model.
    wind_calibration_skip_if_no_overlap: whether to skip wind calibration for
      sources that have no overlapping datapoints with base source. If there
      is no overlap and this option is set to False, then an error is thrown.

  Returns:
    A function that computes the aggregated wind speeds.
  """

  def compute_aggregated_wind_speeds(
      loaded_ibtracs_data: IBTrACSDataSlice,
      models: Dict[str, LinearRegressionModel],
  ) -> np.ndarray:
    """Applies calibration models and aggregates with a mean operation.

    Args:
      loaded_ibtracs_data: IBTrACS data, as loaded by the helper
        cyclone_analysis_utils.load_cyclone_tracks.
      models: A dictionary of calibration models.

    Returns:
      Calibrated and aggregated wind speeds.
    """
    calibrated_wind_speeds = np.stack(
        [model(loaded_ibtracs_data[name]) for name, model in models.items()],
        axis=-1,
    )
    return np.nanmean(calibrated_wind_speeds, axis=-1)

  # Create calibration models
  models = _create_wind_speed_linear_calibration_models(
      ibtracs_data=ibtracs_data,
      wind_calibration_start_date=wind_calibration_start_date,
      wind_calibration_end_date=wind_calibration_end_date,
      wind_calibration_skip_if_no_overlap=wind_calibration_skip_if_no_overlap,
  )

  # Partial performing calibration followed by aggregation
  return functools.partial(
      compute_aggregated_wind_speeds,
      models=models,
  )
