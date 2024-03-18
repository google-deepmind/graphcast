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
"""Computes TOA incident solar radiation compatible with ERA5.

The Top-Of-the-Atmosphere (TOA) incident solar radiation is available in the
ERA5 dataset as the parameter `toa_incident_solar_radiation` (or `tisr`). This
represents the TOA solar radiation flux integrated over a period of one hour
ending at the timestamp given by the `datetime` coordinate. See
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
https://codes.ecmwf.int/grib/param-db/?id=212.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xa


# Default value of the `integration_period` argument to be compatible with ERA5.
_DEFAULT_INTEGRATION_PERIOD = pd.Timedelta(hours=1)

# Default value for the `num_integration_bins` argument. This provides a good
# approximation of the solar radiation in ERA5.
_DEFAULT_NUM_INTEGRATION_BINS = 360

# The length of a Julian year in days.
# https://en.wikipedia.org/wiki/Julian_year_(astronomy)
_JULIAN_YEAR_LENGTH_IN_DAYS = 365.25

# Julian Date for the J2000 epoch, a standard reference used in astronomy.
# https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000
_J2000_EPOCH = 2451545.0

# Number of seconds in a day.
_SECONDS_PER_DAY = 60 * 60 * 24


_TimestampLike = str | pd.Timestamp | np.datetime64
_TimedeltaLike = str | pd.Timedelta | np.timedelta64


# Interface for loading Total Solar Irradiance (TSI) data.
# Returns a xa.DataArray containing yearly average TSI values with a `time`
# coordinate in units of years since 0000-1-1. E.g. 2023.5 corresponds to
# the middle of the year 2023.
TsiDataLoader = Callable[[], xa.DataArray]


# Total Solar Irradiance (TSI): Energy input to the top of the Earth's
# atmosphere in W⋅m⁻². TSI varies with time. This is the reference TSI value
# that can be used when more accurate data is not available.
# https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance
# https://github.com/ecmwf-ifs/ecrad/blob/6db82f929fb75028cc20606a04da87c0abe9b642/radiation/radiation_ecckd.F90#L296
_REFERENCE_TSI = 1361.0


def reference_tsi_data() -> xa.DataArray:
  """A TsiDataProvider that returns a single reference TSI value."""
  return xa.DataArray(
      np.array([_REFERENCE_TSI]),
      dims=["time"],
      coords={"time": np.array([0.0])},
  )


def era5_tsi_data() -> xa.DataArray:
  """A TsiDataProvider that returns ERA5 compatible TSI data."""
  # ECMWF provided the data used for ERA5, which was hardcoded in the IFS (cycle
  # 41r2). The values were scaled down to agree better with more recent
  # observations of the sun.
  time = np.arange(1951.5, 2035.5, 1.0)
  tsi = 0.9965 * np.array([
      # fmt: off
      # 1951-1995 (non-repeating sequence)
      1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773,
      1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
      1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341,
      1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
      1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345,
      1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
      1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416,
      1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
      1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996,
      # 1996-2008 (13 year cycle, repeated below)
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # 2009-2021
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # 2022-2034
      1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
      1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
      1365.8107, 1365.7240, 1365.6918,
      # fmt: on
  ])
  return xa.DataArray(tsi, dims=["time"], coords={"time": time})


# HRES compatible TSI data is from IFS cycle 47r1. The dataset can be obtained
# from the ECRAD package: https://confluence.ecmwf.int/display/ECRAD.
# The example code below can load this dataset from a local file.

# def hres_tsi_data() -> xa.DataArray:
#   with open("total_solar_irradiance_CMIP6_47r1.nc", "rb") as f:
#     with xa.load_dataset(f, decode_times=False) as ds:
#       return ds["tsi"]


_DEFAULT_TSI_DATA_LOADER: TsiDataLoader = era5_tsi_data


def get_tsi(
    timestamps: Sequence[_TimestampLike], tsi_data: xa.DataArray
) -> chex.Array:
  """Returns TSI values for the given timestamps.

  TSI values are interpolated from the provided yearly TSI data.

  Args:
    timestamps: Timestamps for which to compute TSI values.
    tsi_data: A DataArray with a single dimension `time` that has coordinates in
      units of years since 0000-1-1. E.g. 2023.5 corresponds to the middle of
      the year 2023.

  Returns:
    An Array containing interpolated TSI data.
  """
  timestamps = pd.DatetimeIndex(timestamps)
  timestamps_date = pd.DatetimeIndex(timestamps.date)
  day_fraction = (timestamps - timestamps_date) / pd.Timedelta(days=1)
  year_length = 365 + timestamps.is_leap_year
  year_fraction = (timestamps.dayofyear - 1 + day_fraction) / year_length
  fractional_year = timestamps.year + year_fraction
  return np.interp(fractional_year, tsi_data.coords["time"].data, tsi_data.data)


@dataclasses.dataclass(frozen=True)
class _OrbitalParameters:
  """Parameters characterising Earth's position relative to the Sun.

  The parameters characterize the position of the Earth in its orbit around the
  Sun for specific points in time. Each attribute is an N-dimensional array
  to represent orbital parameters for multiple points in time.

  Attributes:
    theta: The number of Julian years since the Julian epoch J2000.0.
    rotational_phase: The phase of the Earth's rotation along its axis as a
      ratio with 0 representing the phase at Julian epoch J2000.0 at exactly
      12:00 Terrestrial Time (TT). Multiplying this value by `2*pi` yields the
        phase in radians.
    sin_declination: Sine of the declination of the Sun as seen from the Earth.
    cos_declination: Cosine of the declination of the Sun as seen from the
      Earth.
    eq_of_time_seconds: The value of the equation of time, in seconds.
    solar_distance_au: Earth-Sun distance in astronomical units.
  """

  theta: chex.Array
  rotational_phase: chex.Array
  sin_declination: chex.Array
  cos_declination: chex.Array
  eq_of_time_seconds: chex.Array
  solar_distance_au: chex.Array


def _get_j2000_days(timestamp: pd.Timestamp) -> float:
  """Returns the number of days since the J2000 epoch.

  Args:
    timestamp: A timestamp for which to compute the J2000 days.

  Returns:
    The J2000 days corresponding to the input timestamp.
  """
  return timestamp.to_julian_date() - _J2000_EPOCH


def _get_orbital_parameters(j2000_days: chex.Array) -> _OrbitalParameters:
  """Computes the orbital parameters for the given J2000 days.

  Args:
    j2000_days: Timestamps represented as the number of days since the J2000
      epoch.

  Returns:
    Orbital parameters for the given timestamps. Each attribute of the return
    value is an array containing the same dimensions as the input.
  """
  # Orbital parameters are computed based on the formulas in this code, which
  # were determined empirically to produce radiation values similar to ERA5:
  # https://github.com/ECCC-ASTD-MRD/gem/blob/1d711f7b89971cd7b1e10afc7508d1135b51397d/src/rpnphy/src/base/sucst.F90
  # https://github.com/ECCC-ASTD-MRD/gem/blob/1d711f7b89971cd7b1e10afc7508d1135b51397d/src/rpnphy/src/base/fctast.cdk
  # https://github.com/ECCC-ASTD-MRD/gem/blob/1d711f7b89971cd7b1e10afc7508d1135b51397d/src/rpnphy/src/base/fcttim.cdk
  # There are many variations to these formulas, but since the goal is to match
  # the values in ERA5, the formulas were implemented as is. Comments reference
  # the notation used in those sources. Here are some additional references
  # related to the quantities being computed here:
  # https://aa.usno.navy.mil/faq/sun_approx
  # https://en.wikipedia.org/wiki/Position_of_the_Sun
  # https://en.wikipedia.org/wiki/Equation_of_time

  # Number of Julian years since the J2000 epoch (including fractional years).
  theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
  # The phase of the Earth's rotation along its axis as a ratio. 0 represents
  # Julian epoch J2000.0 at exactly 12:00 Terrestrial Time (TT).
  rotational_phase = j2000_days % 1.0

  # REL(PTETA).
  rel = 1.7535 + 6.283076 * theta
  # REM(PTETA).
  rem = 6.240041 + 6.283020 * theta
  # RLLS(PTETA).
  rlls = 4.8951 + 6.283076 * theta

  # Variables used in the three polynomials below.
  one = jnp.ones_like(theta)
  sin_rel = jnp.sin(rel)
  cos_rel = jnp.cos(rel)
  sin_two_rel = jnp.sin(2.0 * rel)
  cos_two_rel = jnp.cos(2.0 * rel)
  sin_two_rlls = jnp.sin(2.0 * rlls)
  cos_two_rlls = jnp.cos(2.0 * rlls)
  sin_four_rlls = jnp.sin(4.0 * rlls)
  sin_rem = jnp.sin(rem)
  sin_two_rem = jnp.sin(2.0 * rem)

  # Ecliptic longitude of the Sun - RLLLS(PTETA).
  rllls = jnp.dot(
      jnp.stack(
          [one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel], axis=-1
      ),
      jnp.array([4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002]),
  )

  # Angle in radians between the Earth's rotational axis and its orbital axis.
  # Equivalent to 23.4393°.
  repsm = 0.409093

  # Declination of the Sun - RDS(teta).
  sin_declination = jnp.sin(repsm) * jnp.sin(rllls)
  cos_declination = jnp.sqrt(1.0 - sin_declination**2)

  # Equation of time in seconds - RET(PTETA).
  eq_of_time_seconds = jnp.dot(
      jnp.stack(
          [
              sin_two_rlls,
              sin_rem,
              sin_rem * cos_two_rlls,
              sin_four_rlls,
              sin_two_rem,
          ],
          axis=-1,
      ),
      jnp.array([591.8, -459.4, 39.5, -12.7, -4.8]),
  )

  # Earth-Sun distance in astronomical units - RRS(PTETA).
  solar_distance_au = jnp.dot(
      jnp.stack([one, sin_rel, cos_rel], axis=-1),
      jnp.array([1.0001, -0.0163, 0.0037]),
  )

  return _OrbitalParameters(
      theta=theta,
      rotational_phase=rotational_phase,
      sin_declination=sin_declination,
      cos_declination=cos_declination,
      eq_of_time_seconds=eq_of_time_seconds,
      solar_distance_au=solar_distance_au,
  )


def _get_solar_sin_altitude(
    op: _OrbitalParameters,
    sin_latitude: chex.Array,
    cos_latitude: chex.Array,
    longitude: chex.Array,
) -> chex.Array:
  """Returns the sine of the solar altitude angle.

  All computations are vectorized. Dimensions of all the inputs should be
  broadcastable using standard NumPy rules. For example, if `op` has shape
  `(T, 1, 1)`, `latitude` has shape `(1, H, 1)`, and `longitude` has shape
  `(1, H, W)`, the return value will have shape `(T, H, W)`.

  Args:
    op: Orbital parameters characterising Earth's position relative to the Sun.
    sin_latitude: Sine of latitude coordinates.
    cos_latitude: Cosine of latitude coordinates.
    longitude: Longitude coordinates in radians.

  Returns:
    Sine of the solar altitude angle for each set of orbital parameters and each
    geographical coordinates. The returned array has the shape resulting from
    broadcasting all the inputs together.
  """
  solar_time = op.rotational_phase + op.eq_of_time_seconds / _SECONDS_PER_DAY
  # https://en.wikipedia.org/wiki/Hour_angle#Solar_hour_angle
  hour_angle = 2.0 * jnp.pi * solar_time + longitude
  # https://en.wikipedia.org/wiki/Solar_zenith_angle
  sin_altitude = (
      cos_latitude * op.cos_declination * jnp.cos(hour_angle)
      + sin_latitude * op.sin_declination
  )
  return sin_altitude


def _get_radiation_flux(
    j2000_days: chex.Array,
    sin_latitude: chex.Array,
    cos_latitude: chex.Array,
    longitude: chex.Array,
    tsi: chex.Array,
) -> chex.Array:
  """Computes the instantaneous TOA incident solar radiation flux.

  Computes the instantanous Top-Of-the-Atmosphere (TOA) incident radiation flux
  in W⋅m⁻² for the given timestamps and locations on the surface of the Earth.
  See https://en.wikipedia.org/wiki/Solar_irradiance.

  All inputs are assumed to be broadcastable together using standard NumPy
  rules.

  Args:
    j2000_days: Timestamps represented as the number of days since the J2000
      epoch.
    sin_latitude: Sine of latitude coordinates.
    cos_latitude: Cosine of latitude coordinates.
    longitude: Longitude coordinates in radians.
    tsi: Total Solar Irradiance (TSI) in W⋅m⁻². This can be a scalar (default)
      to use the same TSI value for all the inputs, or an array to allow TSI to
      depend on the timestamps.

  Returns:
    The instataneous TOA incident solar radiation flux in W⋅m⁻² for the given
    timestamps and geographical coordinates. The returned array has the shape
    resulting from broadcasting all the inputs together.
  """
  op = _get_orbital_parameters(j2000_days)
  # Attenuation of the solar radiation based on the solar distance.
  solar_factor = (1.0 / op.solar_distance_au) ** 2
  sin_altitude = _get_solar_sin_altitude(
      op, sin_latitude, cos_latitude, longitude
  )
  return tsi * solar_factor * jnp.maximum(sin_altitude, 0.0)


def _get_integrated_radiation(
    j2000_days: chex.Array,
    sin_latitude: chex.Array,
    cos_latitude: chex.Array,
    longitude: chex.Array,
    tsi: chex.Array,
    integration_period: pd.Timedelta,
    num_integration_bins: int,
) -> chex.Array:
  """Returns the TOA solar radiation flux integrated over a time period.

  Integrates the instantaneous TOA solar radiation flux over a time period.
  The input timestamps represent the end times of each integration period.
  When the integration period is one hour this approximates the
  `toa_incident_solar_radiation` (or `tisr`) parameter from the ERA5 dataset.
  See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
  https://codes.ecmwf.int/grib/param-db/?id=212.

  All inputs are assumed to be broadcastable together using standard NumPy
  rules. To approximate the integral, the instantaneous radiation is computed
  at `num_integration_bins+1` time steps using `_get_radiation_flux` and
  integrated using the trapezoidal rule. A dimension is appended at the end
  of all inputs to compute the instantaneous radiation, which is then integrated
  over to compute the final result.

  Args:
    j2000_days: Timestamps represented as the number of days since the J2000
      epoch. These correspond to the end times of each integration period.
    sin_latitude: Sine of latitude coordinates.
    cos_latitude: Cosine of latitude coordinates.
    longitude: Longitude in radians.
    tsi: Total Solar Irradiance (TSI) in W⋅m⁻².
    integration_period: Integration period.
    num_integration_bins: Number of bins to divide the `integration_period` to
      approximate the integral using the trapezoidal rule.

  Returns:
    The TOA solar radiation flux integrated over the requested time period for
    the given timestamps and geographical coordinates. Unit is J⋅m⁻² .
  """
  # Offsets for the integration time steps.
  offsets = (
      pd.timedelta_range(
          start=-integration_period,
          end=pd.Timedelta(0),
          periods=num_integration_bins + 1,
      )
      / pd.Timedelta(days=1)
  ).to_numpy()

  # Integration happens over the time dimension. Compute the instantaneous
  # radiation flux for all the integration time steps by appending a dimension
  # to all the inputs and adding `offsets` to `j2000_days` (will be broadcast
  # over all the other dimensions).
  fluxes = _get_radiation_flux(
      j2000_days=jnp.expand_dims(j2000_days, axis=-1) + offsets,
      sin_latitude=jnp.expand_dims(sin_latitude, axis=-1),
      cos_latitude=jnp.expand_dims(cos_latitude, axis=-1),
      longitude=jnp.expand_dims(longitude, axis=-1),
      tsi=jnp.expand_dims(tsi, axis=-1),
  )

  # Size of each bin in seconds. The instantaneous solar radiation flux is
  # returned in units of W⋅m⁻². Integrating over time expressed in seconds
  # yields a result in units of J⋅m⁻².
  dx = (integration_period / num_integration_bins) / pd.Timedelta(seconds=1)
  return jax.scipy.integrate.trapezoid(fluxes, dx=dx)


_get_integrated_radiation_jitted = jax.jit(
    _get_integrated_radiation,
    static_argnames=["integration_period", "num_integration_bins"],
)


def get_toa_incident_solar_radiation(
    timestamps: Sequence[_TimestampLike],
    latitude: chex.Array,
    longitude: chex.Array,
    tsi_data: xa.DataArray | None = None,
    integration_period: _TimedeltaLike = _DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS,
    use_jit: bool = False,
) -> chex.Array:
  """Computes the solar radiation incident at the top of the atmosphere.

  The solar radiation is computed for each element in `timestamps` for all the
  locations on the grid determined by the `latitude` and `longitude` parameters.

  To approximate the `toa_incident_solar_radiation` (or `tisr`) parameter from
  the ERA5 dataset, set `integration_period` to one hour (default). See
  https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation and
  https://codes.ecmwf.int/grib/param-db/?id=212.

  Args:
    timestamps: Timestamps for which to compute the solar radiation.
    latitude: The latitude coordinates in degrees of the grid for which to
      compute the solar radiation.
    longitude: The longitude coordinates in degrees of the grid for which to
      compute the solar radiation.
    tsi_data: A DataArray containing yearly TSI data as returned by a
      `TsiDataLoader`. The default is to use ERA5 compatible TSI data.
    integration_period: Timedelta to use to integrate the radiation, e.g. if
      producing radiation for 1989-11-08 21:00:00, and `integration_period` is
      "1h", radiation will be integrated from 1989-11-08 20:00:00 to 1989-11-08
      21:00:00. The default value ("1h") matches ERA5.
    num_integration_bins: Number of equally spaced bins to divide the
      `integration_period` in when approximating the integral using the
      trapezoidal rule. Performance and peak memory usage are affected by this
      value. The default (360) provides a good approximation, but lower values
      may work to improve performance and reduce memory usage.
    use_jit: Set to True to use the jitted implementation, or False (default) to
      use the non-jitted one.

  Returns:
    An 3D array with dimensions (time, lat, lon) containing the total
    top of atmosphere solar radiation integrated for the `integration_period`
    up to each timestamp.
  """
  # Add a trailing dimension to latitude to get dimensions (lat, lon).
  lat = jnp.radians(latitude).reshape((-1, 1))
  lon = jnp.radians(longitude)
  sin_lat = jnp.sin(lat)
  cos_lat = jnp.cos(lat)
  integration_period = pd.Timedelta(integration_period)
  if tsi_data is None:
    tsi_data = _DEFAULT_TSI_DATA_LOADER()
  tsi = get_tsi(timestamps, tsi_data)
  fn = (
      _get_integrated_radiation_jitted if use_jit else _get_integrated_radiation
  )

  # Compute integral for each timestamp individually. Although this could be
  # done in one step, peak memory usage would be proportional to
  # `len(timestamps) * num_integration_bins`. Computing each timestamp
  # individually reduces this to `max(len(timestamps), num_integration_bins)`.
  # E.g. memory usage for a single timestamp, with a full 0.25° grid and 360
  # integration bins is about 1.5 GB (1440 * 721 * 361 * 4 bytes); computing
  # forcings for 40 prediction steps would require 60 GB.
  results = []
  for idx, timestamp in enumerate(timestamps):
    results.append(
        fn(
            j2000_days=jnp.array(_get_j2000_days(pd.Timestamp(timestamp))),
            sin_latitude=sin_lat,
            cos_latitude=cos_lat,
            longitude=lon,
            tsi=tsi[idx],
            integration_period=integration_period,
            num_integration_bins=num_integration_bins,
        )
    )
  return jnp.stack(results, axis=0)


def get_toa_incident_solar_radiation_for_xarray(
    data_array_like: xa.DataArray | xa.Dataset,
    tsi_data: xa.DataArray | None = None,
    integration_period: _TimedeltaLike = _DEFAULT_INTEGRATION_PERIOD,
    num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS,
    use_jit: bool = False,
) -> xa.DataArray:
  """Computes the solar radiation incident at the top of the atmosphere.

  This method is a wrapper for `get_toa_incident_solar_radiation` using
  coordinates from an Xarray and returning an Xarray.

  Args:
    data_array_like: A xa.Dataset or xa.DataArray from which to take the time
      and spatial coordinates for which to compute the solar radiation. It must
      contain `lat` and `lon` spatial dimensions with corresponding coordinates.
      If a `time` dimension is present, the `datetime` coordinate should be a
      vector associated with that dimension containing timestamps for which to
      compute the solar radiation. Otherwise, the `datetime` coordinate should
      be a scalar representing the timestamp for which to compute the solar
      radiation.
    tsi_data: A DataArray containing yearly TSI data as returned by a
      `TsiDataLoader`. The default is to use ERA5 compatible TSI data.
    integration_period: Timedelta to use to integrate the radiation, e.g. if
      producing radiation for 1989-11-08 21:00:00, and `integration_period` is
      "1h", radiation will be integrated from 1989-11-08 20:00:00 to 1989-11-08
      21:00:00. The default value ("1h") matches ERA5.
    num_integration_bins: Number of equally spaced bins to divide the
      `integration_period` in when approximating the integral using the
      trapezoidal rule. Performance and peak memory usage are affected by this
      value. The default (360) provides a good approximation, but lower values
      may work to improve performance and reduce memory usage.
    use_jit: Set to True to use the jitted implementation, or False to use the
      non-jitted one.

  Returns:
    xa.DataArray with dimensions `(time, lat, lon)` if `data_array_like` had
    a `time` dimension; or dimensions `(lat, lon)` otherwise. The `datetime`
    coordinates and those for the dimensions are copied to the returned array.
    The array contains the total top of atmosphere solar radiation integrated
    for `integration_period` up to the corresponding `datetime`.

  Raises:
    ValueError: If there are missing coordinates or dimensions.
  """
  missing_dims = set(["lat", "lon"]) - set(data_array_like.dims)
  if missing_dims:
    raise ValueError(
        f"'{missing_dims}' dimensions are missing in `data_array_like`."
    )

  missing_coords = set(["datetime", "lat", "lon"]) - set(data_array_like.coords)
  if missing_coords:
    raise ValueError(
        f"'{missing_coords}' coordinates are missing in `data_array_like`."
    )

  if "time" in data_array_like.dims:
    timestamps = data_array_like.coords["datetime"].data
  else:
    timestamps = [data_array_like.coords["datetime"].data.item()]

  radiation = get_toa_incident_solar_radiation(
      timestamps=timestamps,
      latitude=data_array_like.coords["lat"].data,
      longitude=data_array_like.coords["lon"].data,
      tsi_data=tsi_data,
      integration_period=integration_period,
      num_integration_bins=num_integration_bins,
      use_jit=use_jit,
  )

  if "time" in data_array_like.dims:
    output = xa.DataArray(radiation, dims=("time", "lat", "lon"))
  else:
    output = xa.DataArray(radiation[0], dims=("lat", "lon"))

  # Preserve as many of the original coordinates as possible, so long as the
  # dimension or the coordinate still exist in the output array.
  for k, coord in data_array_like.coords.items():
    if set(coord.dims).issubset(set(output.dims)):
      output.coords[k] = coord
  return output
