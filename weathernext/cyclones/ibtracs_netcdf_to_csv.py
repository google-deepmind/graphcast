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
"""Converts raw IBTrACS NetCDF track data to the standardised CSV format.

This module provides a function to convert a raw IBTrACS xarray.Dataset (with
dimensions (storm, date_time)) into a pandas DataFrame. This allows test data to
be derived from a single source (the NetCDF), ensuring version consistency
between gridded and tabular data.
"""

from typing import Sequence, Tuple

from absl import logging
from weathernext.cyclones import constants
from weathernext.cyclones import cyclone_utils
import numpy as np
import pandas as pd
import xarray as xr


# Column order matching the existing ibtracs_tracks.csv.
_CSV_COLUMNS = [
    constants.TRACK_ID,
    constants.LAT,
    constants.LON,
    constants.VALID_TIME,
    constants.MIN_SEA_LEVEL_PRESSURE_HPA,
    constants.MAX_SUSTAINED_WIND_SPEED_KNOTS,
    constants.RADIUS_OF_MAXIMUM_WINDS,
    constants.RADIUS_34_KNOT_WINDS_NE_KM,
    constants.RADIUS_34_KNOT_WINDS_SE_KM,
    constants.RADIUS_34_KNOT_WINDS_SW_KM,
    constants.RADIUS_34_KNOT_WINDS_NW_KM,
    constants.RADIUS_50_KNOT_WINDS_NE_KM,
    constants.RADIUS_50_KNOT_WINDS_SE_KM,
    constants.RADIUS_50_KNOT_WINDS_SW_KM,
    constants.RADIUS_50_KNOT_WINDS_NW_KM,
    constants.RADIUS_64_KNOT_WINDS_NE_KM,
    constants.RADIUS_64_KNOT_WINDS_SE_KM,
    constants.RADIUS_64_KNOT_WINDS_SW_KM,
    constants.RADIUS_64_KNOT_WINDS_NW_KM,
]

# Quadrant names in IBTrACS NetCDF order (NE, SE, SW, NW -- matching the 4
# elements along the last axis of usa_r34, usa_r50, usa_r64).
_QUADRANT_NAMES = ("ne", "se", "sw", "nw")

# Nautical miles per km, used for converting wind radii stored in nautical
# miles in the raw IBTrACS NetCDF to km.
_NMILES_PER_KM = 0.539957


def _decode_bytes_or_pass(val):
  """Decodes bytes to str if needed, returns val unchanged otherwise."""
  if isinstance(val, bytes):
    return val.decode("utf-8")
  return val


def convert_ibtracs_netcdf_to_csv_df(
    ds_tracks: xr.Dataset,
    years: int | Sequence[int],
    hours_grid: tuple[int, ...] = (0, 6, 12, 18),
) -> pd.DataFrame:
  """Converts raw IBTrACS NetCDF to CSV-format DataFrame.

  Args:
    ds_tracks: raw IBTrACS xarray.Dataset with dims (storm, date_time).
    years: year(s) to filter storms to (keeps rows with valid_time in these).
    hours_grid: hours to keep after rounding times to the nearest hour.

  Returns:
    A pandas DataFrame with CSV_COLUMNS columns.
  """
  if isinstance(years, int):
    years = [years]

  # Filter out storms that don't have any data in the specified year so we don't
  # have to process them in the loop.
  has_data_in_year = (ds_tracks["time"].dt.year.isin(years)).any("date_time")
  ds_tracks = ds_tracks.sel(storm=has_data_in_year)

  rows = []
  n_storms = ds_tracks.sizes["storm"]
  logging.info("Processing %d storms from NetCDF...", n_storms)

  for storm_idx in range(n_storms):
    storm = ds_tracks.isel(storm=storm_idx)

    # Extract the storm ID (constant across time steps).
    sid = storm["sid"].data.astype(str)

    # Time coordinate gives valid times for this storm.
    times = storm["time"].values  # shape (date_time,)

    for t_idx in range(len(times)):
      time_val = times[t_idx]

      # Skip NaT entries (padded time slots with no data).
      if pd.isna(time_val):
        continue

      lat_val = float(storm["lat"].values[t_idx])
      lon_val = float(storm["lon"].values[t_idx])

      # Skip entries where lat/lon are NaN (no observation at this time).
      if np.isnan(lat_val) or np.isnan(lon_val):
        continue

      # USA fields for scalar variables.
      usa_pres_val = _safe_float(storm["usa_pres"].values[t_idx])
      usa_wind_val = _safe_float(storm["usa_wind"].values[t_idx])
      usa_rmw_val = _safe_float(storm["usa_rmw"].values[t_idx])

      # Convert RMW from nautical miles to km.
      if not np.isnan(usa_rmw_val):
        usa_rmw_val = usa_rmw_val / _NMILES_PER_KM

      # Quadrant wind radii (shape (date_time, 4) in NetCDF).
      r34 = _extract_quadrant_radii(storm, "usa_r34", t_idx)
      r50 = _extract_quadrant_radii(storm, "usa_r50", t_idx)
      r64 = _extract_quadrant_radii(storm, "usa_r64", t_idx)

      row = {
          constants.TRACK_ID: sid,
          constants.LAT: lat_val,
          constants.LON: lon_val,
          constants.VALID_TIME: pd.Timestamp(time_val),
          constants.MIN_SEA_LEVEL_PRESSURE_HPA: usa_pres_val,
          constants.MAX_SUSTAINED_WIND_SPEED_KNOTS: usa_wind_val,
          constants.RADIUS_OF_MAXIMUM_WINDS: usa_rmw_val,
      }

      # Add quadrant radii columns.
      for wind_thresh, radii in ((34, r34), (50, r50), (64, r64)):
        for q_idx, q_name in enumerate(_QUADRANT_NAMES):
          col = f"radius_{wind_thresh}_knot_winds_{q_name}_km"
          row[col] = radii[q_idx]

      rows.append(row)

  df = pd.DataFrame(rows)

  if df.empty:
    logging.warning("No rows extracted from NetCDF.")
    return pd.DataFrame(columns=_CSV_COLUMNS)

  # Clean up empty string / whitespace values.
  with pd.option_context("future.no_silent_downcasting", True):
    df = df.replace(" ", np.nan).infer_objects(copy=False)
    df = df.replace("", np.nan).infer_objects(copy=False)

  # Filter to the specified hours grid.
  df = df[df[constants.VALID_TIME].dt.hour.isin(hours_grid)]

  # Convert numeric columns.
  datetime_cols = [constants.VALID_TIME]
  for col in df.columns:
    if col in datetime_cols:
      continue
    df[col] = pd.to_numeric(df[col], errors="ignore")

  # Filter to the specified year.
  df = df[df[constants.VALID_TIME].dt.year.isin(years)]

  # Select and reorder columns to match the existing CSV exactly.
  df = df[_CSV_COLUMNS].reset_index(drop=True)

  return df


def prepare_ibtracs_storms_dfs(
    ibtracs_ds: xr.Dataset,
    init_time: np.datetime64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Preprocesses the IBTrACS dataset for use in tracking."""
  year = pd.Timestamp(init_time).year
  all_storms_df = convert_ibtracs_netcdf_to_csv_df(
      ds_tracks=ibtracs_ds,
      # Keep the current and previous year in case we are on the boundary and
      # need to keep the previous year's data for the lookback window, and the
      # next year in case we start a forecast close to the end of the year and
      # want to load
      years=[year - 1, year, year + 1],
  )
  all_storms_df = all_storms_df[
      all_storms_df[constants.VALID_TIME].dt.hour.isin([0, 6, 12, 18])
      & (all_storms_df[constants.VALID_TIME].dt.minute == 0)
  ]
  all_storms_df[constants.TRACK_ID] = all_storms_df[constants.TRACK_ID].astype(
      str
  )
  if constants.TIME_SINCE_START in all_storms_df.columns:
    all_storms_df = all_storms_df.drop(columns=[constants.TIME_SINCE_START])
  # Tracker expects longitudes in [0, 360).
  all_storms_df[constants.LON] = all_storms_df[constants.LON] % 360

  initial_storms_df = (
      cyclone_utils.filter_observed_df_for_cyclones_t0_and_before(
          all_storms_df,
          pd.Timestamp(init_time),
          max_lookback=pd.Timedelta(days=1),
      )
  )

  return all_storms_df, initial_storms_df


def _safe_float(val) -> float:
  """Converts a value to float, returning NaN for masked/invalid values."""
  try:
    result = float(val)
  except (ValueError, TypeError):
    return np.nan
  return result


def _extract_quadrant_radii(
    storm: xr.Dataset,
    var_name: str,
    t_idx: int,
) -> list[float]:
  """Extracts the 4 quadrant radii and converts nautical miles to km."""
  if var_name not in storm:
    return [np.nan] * 4

  raw = storm[var_name].values[t_idx]  # shape (4,) or scalar
  radii = []
  if hasattr(raw, "__len__") and len(raw) == 4:
    for val in raw:
      fval = _safe_float(val)
      if not np.isnan(fval):
        fval = fval / _NMILES_PER_KM
      radii.append(fval)
  else:
    # Scalar -- should not happen for quadrant radii but handle gracefully.
    radii = [np.nan] * 4

  return radii
