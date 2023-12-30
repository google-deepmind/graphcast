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
"""Dataset utilities."""

from typing import Any, Mapping, Sequence, Tuple, Union

from graphcast import solar_radiation
import numpy as np
import pandas as pd
import xarray

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

TargetLeadTimes = Union[
    TimedeltaLike,
    Sequence[TimedeltaLike],
    slice  # with TimedeltaLike as its start and stop.
]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"
_DERIVED_VARS = {
    DAY_PROGRESS,
    f"{DAY_PROGRESS}_sin",
    f"{DAY_PROGRESS}_cos",
    YEAR_PROGRESS,
    f"{YEAR_PROGRESS}_sin",
    f"{YEAR_PROGRESS}_cos",
}
TISR = "toa_incident_solar_radiation"


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
  """Computes year progress for times in seconds.

  Args:
    seconds_since_epoch: Times in seconds since the "epoch" (the point at which
      UNIX time starts).

  Returns:
    Year progress normalized to be in the [0, 1) interval for each time point.
  """

  # Start with the pure integer division, and then float at the very end.
  # We will try to keep as much precision as possible.
  years_since_epoch = (
      seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
  )
  # Note depending on how these ops are down, we may end up with a "weak_type"
  # which can cause issues in subtle ways, and hard to track here.
  # In any case, casting to float32 should get rid of the weak type.
  # [0, 1.) Interval.
  return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
  """Computes day progress for times in seconds at each longitude.

  Args:
    seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
      point at which UNIX time starts).
    longitude: 1D array of longitudes at which day progress is computed.

  Returns:
    2D array of day progress values normalized to be in the [0, 1) inverval
      for each time point at each longitude.
  """

  # [0.0, 1.0) Interval.
  day_progress_greenwich = (
      np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
  )

  # Offset the day progress to the longitude of each point on Earth.
  longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
  day_progress = np.mod(
      day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
  )
  return day_progress.astype(np.float32)


def featurize_progress(
    name: str, dims: Sequence[str], progress: np.ndarray
) -> Mapping[str, xarray.Variable]:
  """Derives features used by ML models from the `progress` variable.

  Args:
    name: Base variable name from which features are derived.
    dims: List of the output feature dimensions, e.g. ("day", "lon").
    progress: Progress variable values.

  Returns:
    Dictionary of xarray variables derived from the `progress` values. It
    includes the original `progress` variable along with its sin and cos
    transformations.

  Raises:
    ValueError if the number of feature dimensions is not equal to the number
      of data dimensions.
  """
  if len(dims) != progress.ndim:
    raise ValueError(
        f"Number of feature dimensions ({len(dims)}) must be equal to the"
        f" number of data dimensions: {progress.ndim}."
    )
  progress_phase = progress * (2 * np.pi)
  return {
      name: xarray.Variable(dims, progress),
      name + "_sin": xarray.Variable(dims, np.sin(progress_phase)),
      name + "_cos": xarray.Variable(dims, np.cos(progress_phase)),
  }


def add_derived_vars(data: xarray.Dataset) -> None:
  """Adds year and day progress features to `data` in place if missing.

  Args:
    data: Xarray dataset to which derived features will be added.

  Raises:
    ValueError if `datetime` or `lon` are not in `data` coordinates.
  """

  for coord in ("datetime", "lon"):
    if coord not in data.coords:
      raise ValueError(f"'{coord}' must be in `data` coordinates.")

  # Compute seconds since epoch.
  # Note `data.coords["datetime"].astype("datetime64[s]").astype(np.int64)`
  # does not work as xarrays always cast dates into nanoseconds!
  seconds_since_epoch = (
      data.coords["datetime"].data.astype("datetime64[s]").astype(np.int64)
  )
  batch_dim = ("batch",) if "batch" in data.dims else ()

  # Add year progress features if missing.
  if YEAR_PROGRESS not in data.data_vars:
    year_progress = get_year_progress(seconds_since_epoch)
    data.update(
        featurize_progress(
            name=YEAR_PROGRESS,
            dims=batch_dim + ("time",),
            progress=year_progress,
        )
    )

  # Add day progress features if missing.
  if DAY_PROGRESS not in data.data_vars:
    longitude_coord = data.coords["lon"]
    day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
    data.update(
        featurize_progress(
            name=DAY_PROGRESS,
            dims=batch_dim + ("time",) + longitude_coord.dims,
            progress=day_progress,
        )
    )


def add_tisr_var(data: xarray.Dataset) -> None:
  """Adds TISR feature to `data` in place if missing.

  Args:
    data: Xarray dataset to which TISR feature will be added.

  Raises:
    ValueError if `datetime`, 'lat', or `lon` are not in `data` coordinates.
  """

  if TISR in data.data_vars:
    return

  for coord in ("datetime", "lat", "lon"):
    if coord not in data.coords:
      raise ValueError(f"'{coord}' must be in `data` coordinates.")

  # Remove `batch` dimension of size one if present. An error will be raised if
  # the `batch` dimension exists and has size greater than one.
  data_no_batch = data.squeeze("batch") if "batch" in data.dims else data

  tisr = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
      data_no_batch, use_jit=True
  )

  if "batch" in data.dims:
    tisr = tisr.expand_dims("batch", axis=0)

  data.update({TISR: tisr})


def extract_input_target_times(
    dataset: xarray.Dataset,
    input_duration: TimedeltaLike,
    target_lead_times: TargetLeadTimes,
    ) -> Tuple[xarray.Dataset, xarray.Dataset]:
  """Extracts inputs and targets for prediction, from a Dataset with a time dim.

  The input period is assumed to be contiguous (specified by a duration), but
  the targets can be a list of arbitrary lead times.

  Examples:

    # Use 18 hours of data as inputs, and two specific lead times as targets:
    # 3 days and 5 days after the final input.
    extract_inputs_targets(
        dataset,
        input_duration='18h',
        target_lead_times=('3d', '5d')
    )

    # Use 1 day of data as input, and all lead times between 6 hours and
    # 24 hours inclusive as targets. Demonstrates a friendlier supported string
    # syntax.
    extract_inputs_targets(
        dataset,
        input_duration='1 day',
        target_lead_times=slice('6 hours', '24 hours')
    )

    # Just use a single target lead time of 3 days:
    extract_inputs_targets(
        dataset,
        input_duration='24h',
        target_lead_times='3d'
    )

  Args:
    dataset: An xarray.Dataset with a 'time' dimension whose coordinates are
      timedeltas. It's assumed that the time coordinates have a fixed offset /
      time resolution, and that the input_duration and target_lead_times are
      multiples of this.
    input_duration: pandas.Timedelta or something convertible to it (e.g. a
      shorthand string like '6h' or '5d12h').
    target_lead_times: Either a single lead time, a slice with start and stop
      (inclusive) lead times, or a sequence of lead times. Lead times should be
      Timedeltas (or something convertible to). They are given relative to the
      final input timestep, and should be positive.

  Returns:
    inputs:
    targets:
      Two datasets with the same shape as the input dataset except that a
      selection has been made from the time axis, and the origin of the
      time coordinate will be shifted to refer to lead times relative to the
      final input timestep. So for inputs the times will end at lead time 0,
      for targets the time coordinates will refer to the lead times requested.
  """

  (target_lead_times, target_duration
   ) = _process_target_lead_times_and_get_duration(target_lead_times)

  # Shift the coordinates for the time axis so that a timedelta of zero
  # corresponds to the forecast reference time. That is, the final timestep
  # that's available as input to the forecast, with all following timesteps
  # forming the target period which needs to be predicted.
  # This means the time coordinates are now forecast lead times.
  time = dataset.coords["time"]
  dataset = dataset.assign_coords(time=time + target_duration - time[-1])

  # Slice out targets:
  targets = dataset.sel({"time": target_lead_times})

  input_duration = pd.Timedelta(input_duration)
  # Both endpoints are inclusive with label-based slicing, so we offset by a
  # small epsilon to make one of the endpoints non-inclusive:
  zero = pd.Timedelta(0)
  epsilon = pd.Timedelta(1, "ns")
  inputs = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
  return inputs, targets


def _process_target_lead_times_and_get_duration(
    target_lead_times: TargetLeadTimes) -> TimedeltaLike:
  """Returns the minimum duration for the target lead times."""
  if isinstance(target_lead_times, slice):
    # A slice of lead times. xarray already accepts timedelta-like values for
    # the begin/end/step of the slice.
    if target_lead_times.start is None:
      # If the start isn't specified, we assume it starts at the next timestep
      # after lead time 0 (lead time 0 is the final input timestep):
      target_lead_times = slice(
          pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step
      )
    target_duration = pd.Timedelta(target_lead_times.stop)
  else:
    if not isinstance(target_lead_times, (list, tuple, set)):
      # A single lead time, which we wrap as a length-1 array to ensure there
      # still remains a time dimension (here of length 1) for consistency.
      target_lead_times = [target_lead_times]

    # A list of multiple (not necessarily contiguous) lead times:
    target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
    target_lead_times.sort()
    target_duration = target_lead_times[-1]
  return target_lead_times, target_duration


def extract_inputs_targets_forcings(
    dataset: xarray.Dataset,
    *,
    input_variables: Tuple[str, ...],
    target_variables: Tuple[str, ...],
    forcing_variables: Tuple[str, ...],
    pressure_levels: Tuple[int, ...],
    input_duration: TimedeltaLike,
    target_lead_times: TargetLeadTimes,
    ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
  """Extracts inputs, targets and forcings according to requirements."""
  dataset = dataset.sel(level=list(pressure_levels))

  # "Forcings" include derived variables that do not exist in the original ERA5
  # or HRES datasets, as well as other variables (e.g. tisr) that need to be
  # computed manually for the target lead times. Compute the requested ones.
  if set(forcing_variables) & _DERIVED_VARS:
    add_derived_vars(dataset)
  if set(forcing_variables) & {TISR}:
    add_tisr_var(dataset)

  # `datetime` is needed by add_derived_vars but breaks autoregressive rollouts.
  dataset = dataset.drop_vars("datetime")

  inputs, targets = extract_input_target_times(
      dataset,
      input_duration=input_duration,
      target_lead_times=target_lead_times)

  if set(forcing_variables) & set(target_variables):
    raise ValueError(
        f"Forcing variables {forcing_variables} should not "
        f"overlap with target variables {target_variables}."
    )

  inputs = inputs[list(input_variables)]
  # The forcing uses the same time coordinates as the target.
  forcings = targets[list(forcing_variables)]
  targets = targets[list(target_variables)]

  return inputs, targets, forcings
