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
"""Module-level functions from IBTrACS beam stages, without Beam deps.

This file contains the core gridification logic extracted from the beam pipeline
stages, allowing it to be used without beam dependencies for testing and
open-sourcing.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Tuple

from weathernext.cyclones import data_pipeline_utils as utils
from weathernext.cyclones import ibtracs_processing_utils
import numpy as np
import xarray as xr


IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLE_TUPLES = (
    ibtracs_processing_utils.IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLE_TUPLES
)

IBTRACS_CATEGORY_VARIABLES = ibtracs_processing_utils.IBTRACS_CATEGORY_VARIABLES

_lon_180_to_360 = np.vectorize(
    lambda x: (x if x >= 0 else x + 360) if ~np.isnan(x) else np.nan,
    otypes=[np.float32],
)

NumThreeHourIntervals = int
IBTrACSArray = ibtracs_processing_utils.IBTrACSArray
IBTrACSDataSlice = ibtracs_processing_utils.IBTrACSDataSlice


@dataclasses.dataclass
class IBTrACSData:
  datetime: np.datetime64
  time_index: int
  data_slice: IBTrACSDataSlice


def _scale_variables_from_nmiles_to_km(
    data_slice: IBTrACSDataSlice,
    variables: Sequence[str],
) -> IBTrACSDataSlice:
  data_slice = dict(**data_slice)
  for variable in variables:
    data_slice[variable] = data_slice[variable] * utils.KM_PER_NMILE
  return data_slice


def load_time_slice_from_ibtracs(
    ibtracs_data: xr.Dataset,
    datetime: np.datetime64,
    time_index: int,
) -> IBTrACSData:
  """Loads IBTrACS data for the specified datetime.

  Args:
    ibtracs_data: IBTrACS dataset.
    datetime: datetime to retrieve data for.
    time_index: time index of the time slice being loaded.

  Returns:
    IBTrACSData containing the loaded data slice, which itself is a dict of
    data retrieved by field name, i.e. dict[str, np.ndarray] like

        {
            "latlon"   : np.ndarray,  # shape (N, 2)
            ...
            "usa_r34"  : np.ndarray,  # shape (N, 4)
            "usa_wind" : np.ndarray,  # shape (N,)
        }
  """
  # Get indices of storms which could contain data at datetime, for efficiency
  indices_of_storms_in_range = np.where(
      (datetime >= ibtracs_data.start_time.astype("datetime64[s]"))
      & (datetime <= ibtracs_data.end_time.astype("datetime64[s]"))
  )

  # First select data for storms in range, to discard out-of-range storms
  variables_of_storms_in_range = {
      name: ibtracs_data[name][indices_of_storms_in_range]
      for name in ibtracs_processing_utils.IBTRACS_VARIABLES
  }

  # Then select timestamps corresponding to exact datetime
  data_slice = {
      name: variable.values[
          np.where(variable.time.astype("datetime64[s]") == datetime)
      ]
      for name, variable in variables_of_storms_in_range.items()
  }

  # Postprocess longitudes to be in 0-360 range
  data_slice["lon"] = _lon_180_to_360(data_slice["lon"])
  data_slice["latlon"] = np.stack(
      [data_slice["lat"], data_slice["lon"]], axis=-1
  )

  for agency in ibtracs_processing_utils.IBTRACS_LAT_LON_AGENCIES:
    data_slice[f"{agency}_lon"] = _lon_180_to_360(data_slice[f"{agency}_lon"])
    data_slice[f"{agency}_latlon"] = np.stack(
        [data_slice[f"{agency}_lat"], data_slice[f"{agency}_lon"]], axis=-1
    )

  # Postprocess quadrant radius variables, converting nmiles to km
  data_slice = _scale_variables_from_nmiles_to_km(
      data_slice=data_slice,
      variables=ibtracs_processing_utils.IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLES,
  )

  # Postprocess max wind speed radius variables, converting nmiles to km
  data_slice = _scale_variables_from_nmiles_to_km(
      data_slice=data_slice,
      variables=ibtracs_processing_utils.IBTRACS_MAX_WIND_RADIUS_VARIABLES,
  )

  return IBTrACSData(
      datetime=datetime,
      time_index=time_index,
      data_slice=data_slice,
  )


def cyclone_existence_grids(
    data: IBTrACSDataSlice,
    resolution: float,
    existence_gaussian_and_disc_radius_km: float,
    core_cyclone_variables_only: bool,
) -> IBTrACSDataSlice:
  """Creates gridded cyclone existence fields.

  Args:
    data: IBTrACS data slice.
    resolution: resolution of the resulting grids.
    existence_gaussian_and_disc_radius_km: radius for gaussians and discs.
    core_cyclone_variables_only: if True, only include core variables.

  Returns:
    dict of gridded existence fields.
  """

  def _make_existence_dict(prefix: str) -> IBTrACSDataSlice:
    # Get latlons from source specified by `prefix`, and remove any nans.
    latlons = data[f"{prefix}latlon"]
    is_nan = np.any(np.isnan(latlons), axis=-1)
    latlons = latlons[np.where(~is_nan)]

    gridded_data = {
        f"{prefix}exists_gaussian_unit_mode": (
            ibtracs_processing_utils.create_gaussian_disc_mask_from_latlon(
                latlons=latlons,
                resolution=resolution,
                disc_radius_km=existence_gaussian_and_disc_radius_km,
                normalization="unit_mode",
            )
        ),
    }

    if core_cyclone_variables_only:
      return gridded_data

    return {
        **gridded_data,
        f"{prefix}exists_sparse": (
            ibtracs_processing_utils.create_sparse_nearest_neighbour_mask_from_latlon(
                latlons=latlons,
                resolution=resolution,
            )
        ),
        f"{prefix}exists_disc": (
            ibtracs_processing_utils.create_disc_integral_mask_from_latlon(
                latlons=latlons,
                resolution=resolution,
                disc_radius_km=existence_gaussian_and_disc_radius_km,
            )
        ),
        f"{prefix}exists_gaussian_probability": (
            ibtracs_processing_utils.create_gaussian_disc_mask_from_latlon(
                latlons=latlons,
                resolution=resolution,
                disc_radius_km=existence_gaussian_and_disc_radius_km,
                normalization="probability",
            )
        ),
    }

  # First, create the dictionary of gridded existence data for the
  # aggregated data source.
  grid_data = _make_existence_dict(prefix="")

  # Then, for each individual agency, add the corresponding agency-specific
  # gridded existence field.
  if not core_cyclone_variables_only:
    for agency in ibtracs_processing_utils.IBTRACS_LAT_LON_AGENCIES:
      grid_data = dict(**grid_data, **_make_existence_dict(prefix=f"{agency}_"))

  return grid_data


def cyclone_max_wind_speeds_max_wind_radii_and_min_sea_level_pressures(
    data: IBTrACSDataSlice,
    resolution: float,
    scalar_variable_disc_radius_km: float,
    core_cyclone_variables_only: bool,
) -> Tuple[IBTrACSDataSlice, utils.GriddingInfo]:
  """Creates gridded cyclone wind speeds, max wind radii, and MSLP fields.

  Args:
    data: IBTrACS data slice.
    resolution: resolution of the resulting grids.
    scalar_variable_disc_radius_km: radius of discs for scalar variables.
    core_cyclone_variables_only: if True, only include core variables.

  Returns:
    tuple of (grid_data dict, info dict).
  """
  grid_data = dict()
  info = dict()

  variables = (
      ibtracs_processing_utils.IBTRACS_WIND_VARIABLES_WITH_ALL_WIND
      + ibtracs_processing_utils.IBTRACS_MAX_WIND_RADIUS_VARIABLES
      + ibtracs_processing_utils.IBTRACS_MIN_SEA_LEVEL_PRESSURE_VARIABLES
  )
  if core_cyclone_variables_only:
    variables = ibtracs_processing_utils.filter_variables_for_reduced_dataset(
        variables
    )

  # Add nan masks and wind fields
  for variable in variables:
    # Add info to gridding info for facilitating debugging.
    info[f"{variable}_has_nan"] = bool(np.any(np.isnan(data[variable])))

    if not core_cyclone_variables_only:
      # Create sparse variable field and add to data.
      sparse_variable = (
          ibtracs_processing_utils.create_sparse_nearest_neighbour_mask_from_latlon(
              latlons=data["latlon"],
              resolution=resolution,
              values=data[variable],
              fill_value=float("nan"),
          )
      )
      grid_data[f"{variable}_sparse"] = sparse_variable

    # Create disc wind field.
    disc_variable = ibtracs_processing_utils.create_discs_centered_on_latlons(
        latlons=data["latlon"],
        resolution=resolution,
        disc_radius_km=scalar_variable_disc_radius_km,
        values=data[variable],
    )

    # Create additional mask to set entries outside wind discs to nan.
    is_in_variable_disc = (
        ibtracs_processing_utils.create_discs_centered_on_latlons(
            latlons=data["latlon"],
            resolution=resolution,
            disc_radius_km=scalar_variable_disc_radius_km,
        )
    ).astype(bool)

    # Ensure all entries in discs centered on missing data are set to nan.
    outside_variable_disc = ~is_in_variable_disc
    disc_variable[np.where(outside_variable_disc)] = np.nan

    grid_data[f"{variable}_disc"] = disc_variable

  return grid_data, info


def cyclone_category_grids(
    data: IBTrACSDataSlice,
    resolution: float,
    scalar_variable_disc_radius_km: float,
    core_cyclone_variables_only: bool,
) -> IBTrACSDataSlice:
  """Creates gridded cyclone category fields.

  Args:
    data: IBTrACS data slice.
    resolution: resolution of the resulting grids.
    scalar_variable_disc_radius_km: radius of discs for scalar variables.
    core_cyclone_variables_only: if True, only include core variables.

  Returns:
    dict of gridded category fields.
  """
  grid_data = dict()

  variables = ibtracs_processing_utils.IBTRACS_CATEGORY_VARIABLES
  if core_cyclone_variables_only:
    variables = ibtracs_processing_utils.filter_variables_for_reduced_dataset(
        variables
    )

  for category_variable in variables:
    if category_variable == "usa_sshs":
      category_sparse = (
          ibtracs_processing_utils.create_sparse_nearest_neighbour_mask_from_latlon(
              latlons=data["latlon"],
              resolution=resolution,
              values=data[category_variable] + 6,
              fill_value=float("nan"),
          )
      )
      grid_data[f"{category_variable}_sparse"] = category_sparse

      category_disc = ibtracs_processing_utils.create_discs_centered_on_latlons(
          latlons=data["latlon"],
          resolution=resolution,
          disc_radius_km=scalar_variable_disc_radius_km,
          values=data[category_variable] + 6,
      )
      category_disc[np.where(np.isnan(category_disc))] = np.nan
      grid_data[f"{category_variable}_disc"] = category_disc
    else:
      raise ValueError(
          f"Only usa_sshs categories supported, found {category_variable=}"
      )
  return grid_data


def cyclone_quadrant_grids(
    data: IBTrACSDataSlice,
    resolution: float,
    scalar_variable_disc_radius_km: float,
    quadrant_nan_mask_disc_radius_km: float,
    quadrant_num_subgrid: int,
    core_cyclone_variables_only: bool,
) -> Tuple[IBTrACSDataSlice, utils.GriddingInfo]:
  """Creates cyclone quadrant shape gridded data and radii regression targets.

  Args:
    data: IBTrACS data slice.
    resolution: resolution of the resulting grids.
    scalar_variable_disc_radius_km: radius for scalar variable discs.
    quadrant_nan_mask_disc_radius_km: radius for nan mask discs.
    quadrant_num_subgrid: number of subgrid points for quadrant integrals.
    core_cyclone_variables_only: if True, only include core variables.

  Returns:
    tuple of (grid_data dict, info dict).
  """
  grid_data = dict()
  info = dict()

  for (
      quadrant_source,
      wind_speed_source,
      wind_threshold,
  ) in IBTRACS_QUADRANT_MAX_WIND_RADIUS_VARIABLE_TUPLES:

    if not ibtracs_processing_utils.filter_variables_for_reduced_dataset(
        [quadrant_source]
    ):
      continue

    # Convert wind speed nans to zeros.
    max_wind_speeds = np.nan_to_num(data[wind_speed_source])

    # Impute missing quadrant values if possible / required.
    quadrant_radii, imputation_flags = (
        ibtracs_processing_utils.impute_quadrant_radii_nans(
            max_wind_speeds=max_wind_speeds,
            quadrant_radii=data[quadrant_source],
            threshold_speed_knots=wind_threshold,
        )
    )
    info[f"{quadrant_source}_imputation_flags"] = imputation_flags

    present_or_imputed = np.array([
        flag in ibtracs_processing_utils.QUADRANT_PRESENT_OR_IMPUTED_FLAGS
        for flag in imputation_flags
    ])

    if not core_cyclone_variables_only:
      # Create mask for quadrant data for any quadrants where data are missing.
      quadrants_missing_mask = (
          ibtracs_processing_utils.create_discs_centered_on_latlons(
              latlons=data["latlon"],
              resolution=resolution,
              disc_radius_km=quadrant_nan_mask_disc_radius_km,
              ignore_latlons=present_or_imputed,
          )
      ).astype(bool)

      # Create the actual wind gridded quadrant shape.
      quadrant_shape = (
          ibtracs_processing_utils.create_quadrant_integral_mask_from_latlons(
              latlons=data["latlon"],
              quadrant_radii=quadrant_radii,
              resolution=resolution,
              num_subgrid=quadrant_num_subgrid,
          )
      )
      quadrant_shape[np.where(quadrants_missing_mask)] = np.nan
      grid_data[f"{quadrant_source}_shape"] = quadrant_shape

    for i, quadrant_name in enumerate(ibtracs_processing_utils.QUADRANT_NAMES):
      # Get radii across all storms for one of the four quadrants.
      single_quadrant_radii = quadrant_radii[:, i]
      if not core_cyclone_variables_only:
        sparse_radii = (
            ibtracs_processing_utils.create_sparse_nearest_neighbour_mask_from_latlon(
                latlons=data["latlon"],
                resolution=resolution,
                values=single_quadrant_radii,
                fill_value=float("nan"),
            )
        )
        grid_data[f"{quadrant_source}_{quadrant_name}_radius_sparse"] = (
            sparse_radii
        )

      # Create quadrant radius disc field.
      disc_radii = ibtracs_processing_utils.create_discs_centered_on_latlons(
          latlons=data["latlon"],
          resolution=resolution,
          disc_radius_km=scalar_variable_disc_radius_km,
          values=single_quadrant_radii,
      )

      # Create additional mask to set entries outside wind discs to nan.
      is_in_disc = (
          ibtracs_processing_utils.create_discs_centered_on_latlons(
              latlons=data["latlon"],
              resolution=resolution,
              disc_radius_km=scalar_variable_disc_radius_km,
          )
      ).astype(bool)

      # Ensure all entries in discs centered on missing data are set to nan.
      disc_radii[np.where(~is_in_disc)] = np.nan
      grid_data[f"{quadrant_source}_{quadrant_name}_radius_disc"] = disc_radii

  return grid_data, info


def create_all_grid_data(
    data: IBTrACSDataSlice,
    resolution: float,
    existence_gaussian_and_disc_radius_km: float,
    scalar_variable_disc_radius_km: float,
    quadrant_nan_mask_disc_radius_km: float,
    quadrant_num_subgrid: int,
    core_cyclone_variables_only: bool,
) -> Tuple[IBTrACSDataSlice, utils.GriddingInfo]:
  """Creates all gridded cyclone data from an IBTrACS data slice.

  Args:
    data: IBTrACS data slice as loaded by load_time_slice_from_ibtracs().
    resolution: resolution of the resulting grids.
    existence_gaussian_and_disc_radius_km: radius for existence fields.
    scalar_variable_disc_radius_km: radius for scalar variable discs.
    quadrant_nan_mask_disc_radius_km: radius for quadrant nan mask discs.
    quadrant_num_subgrid: number of subgrid points for quadrant integrals.
    core_cyclone_variables_only: if True, only include core variables.

  Returns:
    tuple of (grid_data dict, info dict). The gridded data dict contains a grid
    for each variable that we have asked the pipeline to process. For example,
    when using `core_cyclone_variables_only=True`, the dict is expected to
    contain the keys:

      "exists_gaussian_unit_mode"
      "usa_wind"
      "all_wind"
      "usa_pres"
      "usa_rmw"
      "usa_r34_ne_radius_disc"
      "usa_r34_se_radius_disc"
      "usa_r34_sw_radius_disc"
      "usa_r34_nw_radius_disc"
      "usa_r50_ne_radius_disc"
      "usa_r50_se_radius_disc"
      "usa_r50_sw_radius_disc"
      "usa_r50_nw_radius_disc"
      "usa_r64_ne_radius_disc"
      "usa_r64_se_radius_disc"
      "usa_r64_sw_radius_disc"
      "usa_r64_nw_radius_disc"

    The info dict is a dictionary containing information about the data, e.g.
    whether there were any nans for a given variable, that can be useful for
    debugging.
  """
  info = dict()

  # Dictionary to store all results in
  grid_data = dict()

  # Process cyclone existence grids and category grids.
  grid_data.update(
      cyclone_existence_grids(
          data=data,
          resolution=resolution,
          existence_gaussian_and_disc_radius_km=existence_gaussian_and_disc_radius_km,
          core_cyclone_variables_only=core_cyclone_variables_only,
      )
  )
  grid_data.update(
      cyclone_category_grids(
          data=data,
          resolution=resolution,
          scalar_variable_disc_radius_km=scalar_variable_disc_radius_km,
          core_cyclone_variables_only=core_cyclone_variables_only,
      )
  )

  # Process cyclone wind speeds, and max wind speed radii.
  grid_winds_rmws_mslps, info_winds_rmws_mslps = (
      cyclone_max_wind_speeds_max_wind_radii_and_min_sea_level_pressures(
          data=data,
          resolution=resolution,
          scalar_variable_disc_radius_km=scalar_variable_disc_radius_km,
          core_cyclone_variables_only=core_cyclone_variables_only,
      )
  )
  grid_data.update(grid_winds_rmws_mslps)
  info.update(info_winds_rmws_mslps)

  # Process wind speed quadrant shapes and quadrant radii.
  grid_quad, info_quad = cyclone_quadrant_grids(
      data=data,
      resolution=resolution,
      scalar_variable_disc_radius_km=scalar_variable_disc_radius_km,
      quadrant_nan_mask_disc_radius_km=quadrant_nan_mask_disc_radius_km,
      quadrant_num_subgrid=quadrant_num_subgrid,
      core_cyclone_variables_only=core_cyclone_variables_only,
  )
  grid_data = dict(**grid_data, **grid_quad)
  info = dict(**info, **info_quad)

  return grid_data, info


def convert_ibtracs_to_gridded_data(
    ibtracs_data: IBTrACSData,
    resolution: float,
    existence_gaussian_and_disc_radius_km: float,
    scalar_variable_disc_radius_km: float,
    quadrant_nan_mask_disc_radius_km: float,
    quadrant_num_subgrid: int,
    core_cyclone_variables_only: bool,
    wind_calibration_model: Callable[[IBTrACSDataSlice], np.ndarray],
) -> utils.GriddedData:
  """Converts an IBTrACS time slice to gridded data.

  Args:
    ibtracs_data: IBTrACS data as loaded by load_time_slice_from_ibtracs().
    resolution: resolution of the resulting grids.
    existence_gaussian_and_disc_radius_km: radius for existence fields.
    scalar_variable_disc_radius_km: radius for scalar variable discs.
    quadrant_nan_mask_disc_radius_km: radius for quadrant nan mask discs.
    quadrant_num_subgrid: number of subgrid points for quadrant integrals.
    core_cyclone_variables_only: if True, only include core variables.
    wind_calibration_model: function to create calibrated "all_wind" variable.

  Returns:
    GriddedData containing the gridded xarray dataset plus metadata.
  """
  datetime = ibtracs_data.datetime
  data_slice = ibtracs_data.data_slice

  # Even though latlon data should never be missing, we check this anyway.
  assert ~np.any(
      np.isnan(data_slice["latlon"])
  ), f"Found nan in data['latlon'] for {datetime=}"

  # Add "all_wind" variable from calibration model into data slice.
  data_slice["all_wind"] = wind_calibration_model(data_slice)

  # Create all grid data and convert to xarray.
  grid_data, info = create_all_grid_data(
      data=data_slice,
      resolution=resolution,
      existence_gaussian_and_disc_radius_km=existence_gaussian_and_disc_radius_km,
      scalar_variable_disc_radius_km=scalar_variable_disc_radius_km,
      quadrant_nan_mask_disc_radius_km=quadrant_nan_mask_disc_radius_km,
      quadrant_num_subgrid=quadrant_num_subgrid,
      core_cyclone_variables_only=core_cyclone_variables_only,
  )
  grid_data = utils.convert_grid_data_to_packed_xarray_dataset(
      datetime=datetime,
      grid_data=grid_data,
      resolution_degrees=resolution,
      variable_prefix="cyclone",
  )

  return utils.GriddedData(
      datetime=datetime,
      zarr_key=None,
      grid_data=grid_data,
      info=info,
  )
