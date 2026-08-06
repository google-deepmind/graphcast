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
"""Utilities for trackers."""

from collections.abc import Mapping
from typing import Any, Literal, NewType

from absl import logging
from weathernext.cyclones import constants
import numpy as np
import pandas as pd
from scipy import interpolate
import xarray as xr


VariableName = NewType("VariableName", str)
InterpolatedValue = NewType("InterpolatedValue", float)

LON_DEG_MIN = 0
LON_DEG_MAX = 360
LAT_DEG_MIN = -90
LAT_DEG_MAX = 90
LAT_DEG_RANGE = LAT_DEG_MAX - LAT_DEG_MIN
LON_DEG_RANGE = LON_DEG_MAX - LON_DEG_MIN

NON_NEGATIVE_VARIABLES_TO_CLIP = [
    constants.MAX_SUSTAINED_WIND_SPEED_KNOTS,
    constants.RADIUS_OF_MAXIMUM_WINDS,
    constants.MIN_SEA_LEVEL_PRESSURE_HPA,
    *constants.QUADRANT_RADII_FLATTENED,
]


def slice_data_array_latlon_box_with_lon_wraparound(
    array: xr.DataArray,
    box_center_lat: float,
    box_center_lon: float,
    box_side_in_degrees_lat: float,
    box_side_in_degrees_lon: float,
    lat_dim_name: str = constants.LAT,
    lon_dim_name: str = constants.LON,
) -> xr.DataArray:
  """Slices an axis aligned box, wrapping around the longitude boundary.

  Args:
    array: data array to slice
    box_center_lat: latitude of box center
    box_center_lon: longitude of box center
    box_side_in_degrees_lat: size of box side in degrees in latitude direction
    box_side_in_degrees_lon: size of box side in degrees in longitude direction
    lat_dim_name: name for the latitude dimension, e.g. 'lat' or 'latitude'
    lon_dim_name: name for the longitude dimension, e.g. 'lon' or 'longitude'

  Returns:
    sliced box wrapping around the longitude boundary.
  """

  if box_side_in_degrees_lat > LAT_DEG_RANGE:
    raise ValueError(
        "Box side in degrees must be smaller than the latitude range in"
        f" degrees. Got {box_side_in_degrees_lat=} and {LAT_DEG_RANGE=}."
    )
  if box_side_in_degrees_lon > LON_DEG_RANGE:
    raise ValueError(
        "Box side in degrees must be smaller than the longitude range in"
        f" degrees. Got {box_side_in_degrees_lon=} and {LON_DEG_RANGE=}."
    )
  if box_center_lat < LAT_DEG_MIN or box_center_lat > LAT_DEG_MAX:
    raise ValueError(
        "Box center latitude must be within the latitude range in degrees."
        f" Got {box_center_lat=} and {LAT_DEG_MIN=} and {LAT_DEG_MAX=}."
    )
  if box_center_lon < LON_DEG_MIN or box_center_lon > LON_DEG_MAX:
    raise ValueError(
        "Box center longitude must be within the longitude range in degrees."
        f" Got {box_center_lon=} and {LON_DEG_MIN=} and {LON_DEG_MAX=}."
    )

  min_lat = box_center_lat - box_side_in_degrees_lat / 2
  max_lat = box_center_lat + box_side_in_degrees_lat / 2
  min_lon = box_center_lon - box_side_in_degrees_lon / 2
  max_lon = box_center_lon + box_side_in_degrees_lon / 2

  if not pd.Index(array.indexes[lat_dim_name]).is_monotonic_increasing:
    raise ValueError(f"Latitude dimension '{lat_dim_name}' must be sorted.")
  if not pd.Index(array.indexes[lon_dim_name]).is_monotonic_increasing:
    raise ValueError(f"Longitude dimension '{lon_dim_name}' must be sorted.")

  box_parts = [
      array.sel(**{  # pyrefly: ignore[bad-argument-type]
          lat_dim_name: slice(min_lat, max_lat),
          lon_dim_name: slice(min_lon, max_lon),
      })
  ]

  if max_lon > LON_DEG_MAX:
    box_parts.insert(
        -1,
        array.sel(**{  # pyrefly: ignore[bad-argument-type]
            lat_dim_name: slice(min_lat, max_lat),
            lon_dim_name: slice(LON_DEG_MIN, max_lon % LON_DEG_RANGE),
        }),
    )

  if min_lon < LON_DEG_MIN:
    box_parts.insert(
        0,
        array.sel(**{  # pyrefly: ignore[bad-argument-type]
            lat_dim_name: slice(min_lat, max_lat),
            lon_dim_name: slice(min_lon % LON_DEG_RANGE, LON_DEG_MAX),
        }),
    )

  if len(box_parts) == 1:
    return box_parts[0]
  elif len(box_parts) == 2:
    return xr.concat(box_parts, dim=constants.LON)
  else:
    raise ValueError(
        "Expected len(box_parts) to be 1 or 2, got"
        f" {len(box_parts)=} and {box_parts=}"
    )


def add_lon_wraparound(ds: xr.Dataset) -> xr.Dataset:
  """Adds a 360-degree longitude slice to handle wraparound for interpolation.

  Args:
    ds: xarray Dataset with a 'lon' coordinate starting at 0.

  Returns:
    Dataset with an additional 360-degree longitude slice appended.
  """
  lon_0_slice = ds.sel({constants.LON: 0})
  return xr.concat(
      [ds, lon_0_slice.assign_coords({constants.LON: 360})],
      dim=constants.LON,
  )


def scipy_2d_interp_xarray_dataset(
    ds: xr.Dataset,
    latlon: tuple[float, float],
    interpolator_kwargs: Mapping[str, Any] | None = None,
) -> dict[VariableName, InterpolatedValue]:
  """Performs a 2D interpolation at specified latlon coordinates using scipy."""
  # Validate that data variables are 2D with lat and lon dimensions.
  if set(ds.dims) != {constants.LAT, constants.LON}:
    raise ValueError(
        f"Dataset dims should be exactly {constants.LAT} and {constants.LON}. "
        f"Instead, found {ds.dims=}"
    )
  # Convert dataset of 2D DataArrays to a single 3D DataArray with 'variable'
  # as a new dimension.
  variable_dim_name = "variable"
  da = ds.to_dataarray(dim=variable_dim_name)

  grid_lat = da[constants.LAT].data
  grid_lon = da[constants.LON].data

  da_transposed = da.transpose(constants.LAT, constants.LON, variable_dim_name)

  interpolator = interpolate.RegularGridInterpolator(
      (grid_lat, grid_lon),
      da_transposed.data,
      **(interpolator_kwargs if interpolator_kwargs is not None else {}),
  )

  interpolated = interpolator(list(latlon)).ravel()
  if interpolated.shape != (len(ds.data_vars),):
    raise AssertionError(
        "Interpolated array should have shape (num_vars,). "
        f"Instead, found {interpolated.shape=}."
    )

  return {
      VariableName(str(k)): InterpolatedValue(float(v))
      for k, v in zip(da_transposed[variable_dim_name].data, interpolated)
  }


def bilinear_interpolation_with_lon_wraparound(
    gridded_predictions: xr.Dataset,
    latlon: tuple[float, float],
    tracker_column_to_gridded_variable_name: Mapping[str, str],
    interpolator_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, float]:
  """Performs bilinear interpolation on gridded predictions at a given latlon.

  Args:
    gridded_predictions: gridded predictions with longitudes in [0, 360).
    latlon: (latitude, longitude) to interpolate at.
    tracker_column_to_gridded_variable_name: mapping from tracker column names
      to the corresponding gridded variable names in gridded_predictions.
    interpolator_kwargs: optional kwargs passed to
      scipy.interpolate.RegularGridInterpolator.

  Returns:
    dict mapping tracker column names to interpolated float values.

  Raises:
    ValueError: if gridded prediction longitudes or latlon are out of range.
  """
  if gridded_predictions.lon.min() != 0:
    raise ValueError(
        "Gridded predictions should have longitudes in range [0, 360). "
        f"Instead, found {gridded_predictions.lon.min().data=}"
    )
  if gridded_predictions.lon.max() >= 360:
    raise ValueError(
        "Gridded predictions should have longitudes in range [0, 360). "
        f"Instead, found {gridded_predictions.lon.max().data=}"
    )

  if latlon[0] < -90.0 or latlon[0] > 90.0:
    raise ValueError(
        "Cyclone center latitude should be in range [-90, 90]. "
        f"Instead, found lat={latlon[0]}."
    )

  if latlon[1] < 0.0 or latlon[1] > 360:
    raise ValueError(
        "Cyclone center longitude should be in range [0, 360]. "
        f"Instead, found lon={latlon[1]}."
    )

  logging.info(
      "Performing bilinear interpolation on gridded predictions at latlon %s",
      latlon,
  )

  if latlon[1] > gridded_predictions[constants.LON].max():
    gridded_predictions = add_lon_wraparound(gridded_predictions)

  interpolated = scipy_2d_interp_xarray_dataset(
      gridded_predictions,
      latlon,
      interpolator_kwargs=interpolator_kwargs or dict(method="linear"),
  )

  logging.info(
      "After bilinear interpolation on gridded predictions at latlon\n%s",
      interpolated,
  )

  interpolated_variables = {}
  for (
      tracker_col,
      gridded_name,
  ) in tracker_column_to_gridded_variable_name.items():
    interpolated_variables[tracker_col] = float(interpolated[gridded_name])  # pyrefly: ignore[bad-index]

  return interpolated_variables


def _enforce_non_negativity_constraint(df: pd.DataFrame) -> pd.DataFrame:
  """Enforces a non-negativity constraint on cyclone variables in the dataframe.

  Given a dataframe that contains

  * w: max wind speed
  * r: radius of max wind
  * p: minimum pressure
  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  Applies the non-negativity constraint

    w, r, p, q_34, q_50, q_64 >= 0

  by clipping the variables from below by 0.

  NOTE: nan entries are preserved as nan.

  Args:
    df: dataframe to apply the constraint to

  Returns:
    df: deep copy of the dataframe with the constraint applied.
  """
  df = df.copy(deep=True)
  vars_to_clip = [
      col for col in df.columns if col in NON_NEGATIVE_VARIABLES_TO_CLIP
  ]
  df[vars_to_clip] = df[vars_to_clip].clip(lower=0)
  return df


def _enforce_quadrant_radius_zero_if_wind_speed_below_threshold(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Enforces constraint that quadrant radii are zero if wind speed too small.

  Given a dataframe that contains

  * w: max wind speed
  * r: radius of max wind
  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  Applies the non-negativity constraint

    w, r, p, q_34, q_50, q_64 >= 0

  by clipping the variables from below by 0.

  Args:
    df: dataframe to apply the constraint to

  Returns:
    df: deep copy of the dataframe with the constraint applied.
  """
  df = df.copy(deep=True)
  for threshold, quadrant_coord_to_name in constants.QUADRANT_RADII.items():
    for _, quadrant_varname in quadrant_coord_to_name.items():
      if quadrant_varname not in df.columns:
        logging.info(
            "Skipping thresholding %s as this was not in the "
            "columns of the dataframe. Found %s",
            quadrant_varname,
            df.columns,
        )
        continue
      # Determine which radii to zero out. If max wind speed is above the
      # threshold, then we already have consistency and keep the original value.
      # If max wind is nan, then we can't ascertain if we need to set the
      # quadrant to zero, so we keep the original value again.
      wind_over_threshold_or_nan = (
          df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS] >= threshold
      ) | df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS].isnull()
      thresholded_values = df[quadrant_varname].where(
          wind_over_threshold_or_nan,
          0.0,
      )
      # Only update values which were not nan in the original dataframe.
      df[quadrant_varname] = thresholded_values.where(
          df[quadrant_varname].notnull(),
          np.nan,
      )
  return df


def _enforce_quad_non_decreasing(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Enforces a monotonicity constraint on cyclone variables in the dataframe.

  Given a dataframe that contains

  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  Applies the monotonicity constraint

    q_34 >= q_50 >= q_64,

  by setting q_50 = max(q_64, q_50) and then q_34 = max(q_64, q_50, q_34).

  NOTE: values that are nan have no effect on clipping (i.e. do not clip other
  variables) and also remain nan in the result.

  Args:
    df: dataframe to apply the constraint to

  Returns:
    df: deep copy of the dataframe with the constraint applied.
  """

  df = df.copy(deep=True)
  # Important to loop over quadrant radii in descending threshold order, i.e.
  # 64kt, 50kt, 34kt to apply thresholding in the correct order.
  thresh_decreasing = sorted(constants.QUADRANT_RADII.keys(), reverse=True)
  for i, thresh_quad_to_clip in enumerate(thresh_decreasing):
    for quadrant_name in constants.QUADRANT_COORDS:
      var_to_clip = constants.QUADRANT_RADII[thresh_quad_to_clip][quadrant_name]
      if var_to_clip not in df.columns:
        logging.info(
            "Skipping thresholding %s as this was not in the "
            "columns of the dataframe. Found %s",
            var_to_clip,
            df.columns,
        )
        continue

      # Clip the quadrant radius at each wind speed with quadrant radii at each
      # of the lower wind speeds.
      for threshold_of_quad_to_clip_with in thresh_decreasing[:i]:
        var_to_clip_with = constants.QUADRANT_RADII[
            threshold_of_quad_to_clip_with
        ][quadrant_name]

        # Only try clipping if both variables are present.
        if var_to_clip_with in df.columns:
          # NOTE: when doing x.clip(y), if x is nan, the result is nan, and if
          # y is nan, then x is unchanged. This gracefully handles nans as
          # expected in the docstring.
          df[var_to_clip] = df[var_to_clip].clip(lower=df[var_to_clip_with])
        else:
          raise ValueError(
              f"Attempted to threshold {var_to_clip} with {var_to_clip_with},"
              f" but {var_to_clip_with} not found in columns {df.columns}"
          )
  return df


def _enforce_rmw_smaller_than_quad_radii_by_clipping_rmw(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Enforces a radius constraint on cyclone variables in the dataframe.

  Given a dataframe that contains

  * w: max wind speed
  * r: radius of max wind
  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  Applies the constraint

    if w >= c for some c in {34, 50, 64}, then r <= q_c.

  by decreasing the RMW to be equal to the largest quadrant radius, by clipping
  it from above.

  NOTE: if r is nan, then the result is nan, and if any q_c is nan, then it has
  no effect during the clipping stage.

  Args:
    df: dataframe to apply the constraint to

  Returns:
    df: deep copy of the dataframe with the constraint applied.
  """

  df = df.copy(deep=True)
  # Outer loop is over quadrant speed threshold, e.g. 30kt, 50kt, 64kt. At each
  # pass of the loop, we optionally clip the RMW from above by the largest
  # quadrant radius at the given speed threshold. Note that this does not depend
  # on the order in which the dictionary is iterated over because clipping is
  # a function that commutes with itself.
  for threshold, quad_coord_to_varname in constants.QUADRANT_RADII.items():
    # Inner loop is over quadrant coordinates, e.g. ne, se, sw, nw.
    all_upper_clips: list[np.ndarray] = []
    for _, quadrant_varname in quad_coord_to_varname.items():
      if quadrant_varname not in df:
        logging.info(
            "Skipping thresholding radius of max winds by %s because this was"
            " not found in the columns of the dataframe. Found %s",
            quadrant_varname,
            df.columns,
        )
        continue
      wind_over_threshold_and_not_nan = (
          df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS] >= threshold
      ) & (df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS].notnull())
      all_upper_clips.append(
          df[quadrant_varname]
          .where(
              wind_over_threshold_and_not_nan,
              np.nan,
          )
          .values
      )

    # The upper clip for RMW is the maximum clip across all quadrant radii.
    upper_clip = np.nanmax(np.stack(all_upper_clips, axis=0), axis=0)
    df[constants.RADIUS_OF_MAXIMUM_WINDS] = df[
        # NOTE: when doing x.clip(y), if x is nan, the result is nan, and if
        # y is nan, then x is unchanged. This gracefully handles nans as
        # expected in the docstring.
        constants.RADIUS_OF_MAXIMUM_WINDS
    ].clip(upper=upper_clip)
  return df


def _enforce_rmw_smaller_than_quad_radii_by_clipping_quad_radii(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Enforces a radius constraint on cyclone variables in the dataframe.

  Given a dataframe that contains

  * w: max wind speed
  * r: radius of max wind
  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  Applies the constraint

    if w >= c for some c in {34, 50, 64}, then r <= q_c.

  by increasing the largest quadrant radius to be at least as large as the RMW,
  by clipping it from below.

  Args:
    df: dataframe to apply the constraint to

  Returns:
    df: deep copy of the dataframe with the constraint applied.
  """

  # First, check that the quadrant radii are consistent between themselves, i.e.
  # that q_34 >= q_50 >= q_64 across all quadrants.
  _check_quadrant_radii_consistent(df)

  df = df.copy(deep=True)

  for threshold, quad_coord_to_varname in constants.QUADRANT_RADII.items():
    # For each threshold, we find the column name out of quad_coord_to_varname
    # that corresponds to the largest quadrant radius.
    quad_varnames_at_threshold = [
        v for v in quad_coord_to_varname.values() if v in df.columns
    ]
    if not quad_varnames_at_threshold:
      logging.info(
          "Skipping thresholding radius of max winds by quadrant radii of "
          "speed %skt because no quadrant radii were found in the columns of "
          "the dataframe. Got\n%s",
          quad_varnames_at_threshold,
          df.to_string(),
      )
      continue
    quadrant_max_radius_varname = df[quad_varnames_at_threshold]
    # Contains the name of the quadrant with the largest radius at the given
    # threshold. If all quadrants were nan, then the name will be nan, so we
    # handle this explicitly below.
    quadrant_max_radius_varname = quadrant_max_radius_varname.idxmax(axis=1)

    quadrant_max_radius_varname = quadrant_max_radius_varname.where(
        quadrant_max_radius_varname.notnull(),
        # Here we explicitly handle nan quadrant names (which can occur if all
        # quadrants were nan at the given threshold), by filling them with a
        # single dummy value. If the quadrant radii are already nan, clipping
        # will have no effect on them, they will all remain nan. So adding this
        # dummy index does not change the result (but prevents errors from
        # trying to index with a nan column name).
        quad_varnames_at_threshold[0],
    )
    rmw_clipping_values = df[constants.RADIUS_OF_MAXIMUM_WINDS].where(
        df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS] >= threshold,
        np.nan,
    )

    for i, varname in enumerate(quadrant_max_radius_varname):
      df.loc[i, [varname]] = df.loc[i, [varname]].clip(
          lower=rmw_clipping_values.iloc[i]
      )

  # Finally we re-apply the quadrant monotonicity constraint. This is because
  # in the loop above we clip the *largest* of the four quadrants from below
  # making it larger, which can break monotonicity, e.g. if the input contained
  #
  # vmax   : 55
  # rmw    : 35
  # q_50_ne: 20
  # q_50_ne: 10
  # q_34_ne: 40
  # q_34_ne: 50
  #
  # after clipping below with RMW, at this point, the dataframe would contain
  #
  # vmax   : 55
  # rmw    : 35
  # q_50_ne: 35   <<< Larger 50kt radius clipped below from 20km to 35km.
  # q_50_ne: 10
  # q_34_ne: 40   <<< But now q_50_ne > q_34_ne, which is a violation so we need
  # q_34_ne: 50       to re-apply the monotonicity constraint.
  #
  df = _enforce_quad_non_decreasing(df)

  return df


def _wind_speeds_non_increasing_with_increasing_radii(
    df: pd.DataFrame,
    quadrant_coord_to_check: str,
    include_radius_of_max_winds: bool,
) -> np.ndarray:
  """Checks that wind speeds are non-increasing as wind radii increase.

  This function raises a ValueError if the dataframe contains wind speeds that
  increase as a function of radii, including quadrant radii and the max wind
  radius. In pseudocode, assuming that we have a collection of four tuples of
  radii and speeds *corresponding to a single row in df* arranged in a list:

    radii_and_speeds = [
      (rmw, max_wind),  # included only if include_radius_of_max_winds == True
      (q34, 34),
      (q50, 50),
      (q64, 64),
    ]

  we first sort the `radii_and_speeds` list by the first entry of each tuple
  (the radius entry) and then checking that the corresponding wind speed entries
  are monotonically decreasing (as radii increase).

  NOTE: whenever entries are nan for either quadrant radii, or for the max wind
  entry, the corresponding monotonicity check is effectively skipped via nan
  checking statements (see comments in function body).

  Args:
    df: dataframe to check
    quadrant_coord_to_check: the quadrant coordinate to check, e.g. 'ne'
    include_radius_of_max_winds: whether to include the radius of max winds in
      the check. If False, only the quadrant radii are checked. If True, the
      radius of max winds and the max wind speed are included in the check.

  Returns:
    np.ndarray: boolean array indicating whether each row satisfies the
      constraint.
  """

  radii = np.zeros((0, len(df)), dtype=np.float32)
  speeds = np.zeros((0, len(df)), dtype=np.float32)

  # Include max wind radius only if specified in the kwargs.
  if (
      include_radius_of_max_winds
      and constants.RADIUS_OF_MAXIMUM_WINDS in df.columns
  ):
    radii = np.append(
        radii, df[constants.RADIUS_OF_MAXIMUM_WINDS].values[None, :], axis=0
    )
    speeds = np.append(
        speeds,
        df[constants.MAX_SUSTAINED_WIND_SPEED_KNOTS].values[None, :],
        axis=0,
    )

  for quad_threshold in constants.QUADRANT_RADII.keys():
    quad_varname = constants.QUADRANT_RADII[quad_threshold][
        quadrant_coord_to_check
    ]
    if quad_varname in df.columns:
      quad_radii = df[quad_varname].values[None, :]
      speed = np.ones_like(quad_radii) * quad_threshold

      radii = np.append(radii, quad_radii, axis=0)
      speeds = np.append(speeds, speed, axis=0)

  # Ensure nans are shared so we account for them properly in the check.
  radius_or_speed_is_nan = np.logical_or(np.isnan(radii), np.isnan(speeds))
  radii = np.where(radius_or_speed_is_nan, np.nan, radii)
  speeds = np.where(radius_or_speed_is_nan, np.nan, speeds)

  # Sorting puts nans at the end of the list.
  idx_sort = np.argsort(radii, axis=0)
  speeds_sorted_by_increasing_radii = np.take_along_axis(
      speeds,
      idx_sort,
      axis=0,
  )
  increasing_radii = np.take_along_axis(
      radii,
      idx_sort,
      axis=0,
  )

  # If there's a nan in a diff pair, the diff is also a nan. All the diffs
  # should be non-positive (wind speed decreases as radius increases), or nan.
  speed_diffs = np.diff(speeds_sorted_by_increasing_radii, axis=0)
  radii_diffs = np.diff(increasing_radii, axis=0)

  # NOTE(1): We want to check that diffs are all non-positive, but since there
  # are nans that would mess up the comparison. Instead we use a trick. We
  # check that no entries are positive: in case of a nan the comparison
  # returns False and np.any is not triggered.
  # NOTE(2): In cases where the radius of max winds is clipped by a quadrant
  # radius, this can result in two different radii in `radii` that correspond
  # to different wind speeds. The argsort over radii is therefore not
  # guaranteed to output them in the correct order. To get around this we only
  # check those diffs where the radii diff is non-zero.
  constraints_violated = np.logical_and(speed_diffs > 0.0, radii_diffs != 0.0)

  # NOTE: after np.any, the result is a boolean array of shape (N,), containing
  # one entry per row in the dataframe, indicating whether the particular row
  # has violated the monotonicity constraint.
  constraints_violated = np.any(constraints_violated, axis=0)
  constraints_violated = np.reshape(constraints_violated, (-1,))
  assert len(constraints_violated) == len(df)
  return np.logical_not(constraints_violated)


def enforce_physical_consistency_on_quadrants_and_winds(
    df: pd.DataFrame,
    rmw_and_quad_clipping_mode: Literal[
        "clip_rmw_above_using_quads",
        "clip_quads_below_using_rmw",
    ] = "clip_quads_below_using_rmw",
) -> pd.DataFrame:
  """Enforces physical wind speed and radii constraints on a dataframe.

  Our models and tracker are not constrained to produce physical wind speeds and
  wind radii, so we enforce these constraints here. Specifically given:

  * w: max wind speed
  * r: radius of max wind
  * p: minimum pressure
  * q_34, q_50, q_64: quadrant radii at 34, 50 and 64 knots (for given quadrant)

  We enforce the following constraints:

  1. all the variables are non-negative, i.e.
       w, r, p, q_34, q_50, q_64 >= 0,
     using _enforce_non_negativity_constraint. This step may change the values
     of any of the variables (if negative).

  2. if wind speed is below speed threshold of a quadrant, it must have 0-radius
      if  w < c for some c in {34, 50, 64}, then q_c = 0,
     using _enforce_quadrant_radius_zero_if_wind_speed_below_threshold. This
     step can change the value of the quadrant radii, but all other variable
     values remain the same.

  3. quadrants must have increasing radii as speed threshold decreases
       q_64 <= q_50 <= q_34,
      using _enforce_quad_non_decreasing. This step can change the value of q_50
      and q_34 but not the value of q_64: it applies the constraint by clipping
      q_50 from below using q_64, and then clipping q_34 from below by q_50. No
      other variables are changed.

  4. if max wind speed is above wind speed threshold of a given quadrant, then
     the radius of max wind must be smaller than or equal to the quadrant radius
      if w >= c for some c in {34, 50, 64}, then r <= q_c,
     using `_enforce_rmw_smaller_than_quad_radii_by_clipping_rmw` or
     `_enforce_rmw_smaller_than_quad_radii_by_clipping_quad_radii`. This step
     can change only the value of the max wind radius: it clips the radius of
     max wind to the smallest quadrant radius (whose threshold is below the max
     wind speed).

  Each of the helpers is set up so that its application does not violate the
  constraints applied by the previous helper. See the helper descriptions for
  specifics on how they apply their constraints.

  Args:
    df: forecast dataframe to apply thresholds to
    rmw_and_quad_clipping_mode: whether to clip the radius of max winds from
      above by the largest quadrant radius or to clip the largest quadrant
      radius from below by the radius of max winds. Defaults to
      "clip_quads_below_using_rmw", clips the largest quadrant radius from below
      (i.e. increases it).

  Returns:
    df: dataframe with thresholded values
  """
  logging.info("Before applying physical constraints:\n%s", df.T)

  df = df.copy(deep=True)
  df = _enforce_non_negativity_constraint(df)
  df = _enforce_quadrant_radius_zero_if_wind_speed_below_threshold(df)
  df = _enforce_quad_non_decreasing(df)
  if rmw_and_quad_clipping_mode == "clip_rmw_above_using_quads":
    df = _enforce_rmw_smaller_than_quad_radii_by_clipping_rmw(df)
  elif rmw_and_quad_clipping_mode == "clip_quads_below_using_rmw":
    df = _enforce_rmw_smaller_than_quad_radii_by_clipping_quad_radii(df)
  else:
    raise ValueError(f"Unsupported {rmw_and_quad_clipping_mode=}")

  _check_quadrant_radii_and_radius_max_winds_are_consistent(df)
  _check_quadrant_radii_consistent(df)

  logging.info("After applying physical constraints:\n%s", df.T)

  return df


def _check_quadrant_radii_and_radius_max_winds_are_consistent(df: pd.DataFrame):
  """Verifies that the wind radii and wind speeds are consistent."""
  # Check wind speed is non-increasing with increasing radii.
  constraints_satisfied = np.stack(
      [
          _wind_speeds_non_increasing_with_increasing_radii(
              df=df,
              quadrant_coord_to_check=quadrant_coord,
              # We include RMW in this check. The check is applied per-quadrant,
              # so it can return False if this monotonicity constraint is
              # violated for one of the quadrant coordinates (e.g. ne, se, sw,
              # nw), but at least one of the quadrant coordinates should return
              # True.
              include_radius_of_max_winds=True,
          )
          for quadrant_coord in constants.QUADRANT_COORDS
      ],
      axis=0,
  )
  if not np.all(np.any(constraints_satisfied, axis=0)):
    raise AssertionError(
        "Wind speeds are not non-increasing with increasing quadrant radii. "
        f"Instead found\n{df.to_string()}."
    )


def _check_quadrant_radii_consistent(df: pd.DataFrame):
  """Verifies that the quadrant radii are consistent."""
  constraints_satisfied = np.stack(
      [
          _wind_speeds_non_increasing_with_increasing_radii(
              df=df,
              quadrant_coord_to_check=quadrant_coord,
              # This time we don't include RMW in the check. This is effectively
              # checking that the quadrant radii are monotonic (ignoring RMW).
              # This constraint should be satisfied per-quadrant for all the
              # quadrant coordinates.
              include_radius_of_max_winds=False,
          )
          for quadrant_coord in constants.QUADRANT_COORDS
      ],
      axis=0,
  )
  if not np.all(constraints_satisfied):
    raise AssertionError(
        f"Quadrant radii must be monotonic. Instead found\n{df.to_string()}."
    )
