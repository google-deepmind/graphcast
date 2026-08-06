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
"""Cyclone utility functions."""

from typing import Sequence, Tuple, Union

from weathernext.cyclones import constants
import numpy as np
import pandas as pd
import pyproj
from scipy import spatial


def geodesic_distance(
    latlon0: Union[np.ndarray, Sequence[float]],
    latlon1: Union[np.ndarray, Sequence[float]],
    ellipsoid: str = 'WGS84',
) -> np.ndarray:
  """Computes geodesic distance in km, between latitude/longitude degrees.

  Each latlon0/1 is an [M, N, ..., 2] array (which can have different M, N, ...,
  but must be broadcastable), with (lat, lon) on their inner axes.

  Args:
    latlon0: An [M0, N0, ..., 2] array with (lat, lon) on the inner axis. Must
      be broadcastable to M1, N1, ... below.
    latlon1: An [M1, N1, ..., 2] array with (lat, lon) on the inner axis. Must
      be broadcastable to M0, N0, ... above.
    ellipsoid: A string indicator for computing the geodesic on Earth.

  Returns:
    The distances array, with shape [M, N, ...] which is broadcasted from
      [M0, N0, ...] and [M1, N1, ...].
  """
  valid_ellipsoids = (
      'GRS80',
      'airy',
      'bessel',
      'clrk66',
      'intl',
      'WGS60',
      'WGS66',
      'WGS72',
      'WGS84',
      'sphere',
  )
  if ellipsoid not in valid_ellipsoids:
    raise ValueError(
        f"Invalid ellipsoid: '{ellipsoid}'. Valid ellipsoids are: , ".join(
            valid_ellipsoids
        )
    )
  geod = pyproj.Geod(ellps=ellipsoid)
  latlon0, latlon1 = np.broadcast_arrays(latlon0, latlon1)
  lat0, lon0 = np.rollaxis(latlon0, axis=-1)
  lat1, lon1 = np.rollaxis(latlon1, axis=-1)
  _, _, distances = geod.inv(lon0, lat0, lon1, lat1)
  distances /= 1000.0  # Convert meters to km.
  return distances


def filter_observed_df_for_cyclones_t0_and_before(
    observed_df: pd.DataFrame,
    init_time: pd.Timestamp,
    max_lookback: pd.Timedelta = pd.Timedelta(days=0),
) -> pd.DataFrame:
  """Filters a dataframe to only include cyclones present at t=0 and before.

  Args:
    observed_df: Observed cyclone tracks.
    init_time: initialisation time of the forecast, from which we select
      cyclones present at t=0, and pass all rows for these cyclones at from
      `init_time - max_lookback <= t <= init_time`.
    max_lookback: pd.Timedelta of the maximum lookback window. Defaults to 0
      days (no lookback window).

  Returns:
    A filtered dataframe containing only cyclones present at t=0 and before.
  """
  # Select rows for cyclones present at t=0, and pass all rows for these
  # cyclones at t<=0 to the tracker.
  observed_t0_df = observed_df[observed_df[constants.VALID_TIME] == init_time]
  observed_t0_cyclones = observed_t0_df[constants.TRACK_ID].unique()
  observed_df = observed_df[
      observed_df[constants.TRACK_ID].isin(observed_t0_cyclones)
  ]
  observed_df = observed_df[observed_df[constants.VALID_TIME] <= init_time]
  observed_df = observed_df[
      observed_df[constants.VALID_TIME] >= init_time - max_lookback
  ]
  return observed_df


def latlon_to_cartesian(
    lat: np.ndarray,
    lon: np.ndarray,
    args_radians: bool = False,
    radius: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Convert point from latitude-longitude to Cartesian coordinates in 3D."""

  # The sister method cartesian_to_latlon returns longitudes in [0, 360],
  # so we check that the input longitudes are in [0, 360] or NaN to avoid
  # the convention changing.
  assert np.all(
      np.logical_or(np.isnan(lon), np.logical_and(lon >= 0.0, lon <= 360.0))
  )
  assert np.all(
      np.logical_or(np.isnan(lat), np.logical_and(lat >= -90.0, lat <= 90.0))
  )

  if not args_radians:
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

  if radius is None:
    radius = 1.0

  # Convert latitudes and longitudes in three-dimensional coordinates
  x = radius * np.cos(lon) * np.cos(lat)
  y = radius * np.sin(lon) * np.cos(lat)
  z = radius * np.sin(lat)

  return x, y, z


def cartesian_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Convert point from Cartesian to latitude-longitude coordinates."""

  radius = np.sqrt(x**2 + y**2 + z**2)
  x /= radius
  y /= radius
  z /= radius

  # arcsin returns latitude in [-pi/2, pi/2]
  lat = np.arcsin(z)

  cos_lon = x / np.cos(lat)
  sin_lon = y / np.cos(lat)

  # arctan2 returns longitude in [-pi, pi], which we convert to [0, 2*pi]
  lon = np.arctan2(sin_lon, cos_lon)
  # Convert longitude in [-pi, pi] to [0, 2*pi]
  lon[lon < 0] = 2 * np.pi + lon[lon < 0]

  return np.rad2deg(lat), np.rad2deg(lon)


def cartesian_spherical_geodesic_momentum_update(
    pos_curr: np.ndarray,
    pos_prev: np.ndarray,
    momentum: float,
) -> np.ndarray:
  """Calculates momentum update from pos_curr using pos_prev, on a sphere.

  Assumes pos_curr and pos_prev are on the unit sphere, and that the dot product
  between them is positive. Assumes momentum constant is in [0, 1].

  Given a current point pos_curr and a previous point pos_curr on the unit
  sphere, the method `geodesic_momentum_update` computes another point pos_next
  in the unit sphere such that:

    - pos_next is in the great circle that passes through pos_curr and pos_prev,
      specifically in the major (rather than the minor) arc of the great circle.
    - pos_next is closer to pos_curr than to pos_prev.
    - the spherical geodesic distance between pos_curr and pos_next is equal to
      a fraction `momentum` of the spherical geodesic distance between pos_prev
      and pos_curr.

  Args:
    pos_curr: current point to which momentum update was applied.
    pos_prev: previous point used to generate the momentum update.
    momentum: amount of momentum applied, must be positive.

  Returns:
    Updated point with momentum.

  Raises:
    ValueError: if
      - momentum is not in [0, 1]
      - if the dot product between pos_curr and pos_prev is negative
      - if the shape of pos_curr or the shape of pos_prev is different from (3,)
  """

  if (pos_curr.shape != (3,)) or (pos_prev.shape != (3,)):
    raise ValueError(
        f'Received {pos_curr.shape=} {pos_prev.shape=}, expected (3,) for both.'
    )

  if momentum < 0.0 or momentum > 1.0:
    raise ValueError('Momentum must be in [0, 1].')

  if np.allclose(pos_curr, pos_prev):
    return pos_curr

  if np.isnan(pos_curr).any() or np.isnan(pos_prev).any():
    return np.nan * np.ones_like(pos_curr)

  # Compute dot product and check it's positive.
  dot = np.vecdot(pos_curr, pos_prev)
  if dot < 0:
    raise ValueError(
        f'Received {pos_curr=} {pos_prev=}, which have vector dot product '
        f'{dot=}, but `geodesic_momentum_update` assumes dot all products > 0.'
    )
  # Angle between the two vectors from center of sphere to current
  theta = np.arccos(dot)
  t = momentum * theta / (np.pi - theta)

  return spatial.geometric_slerp(pos_curr, -pos_prev, t)


def latlon_spherical_geodesic_momentum_update(
    latlon_curr: tuple[float, float],
    latlon_prev: tuple[float, float],
    momentum: float,
) -> tuple[float, float]:
  """Calculates momentum update from latlon_curr using latlon_prev, on a sphere.

  Args:
    latlon_curr: current point to which momentum update was applied.
    latlon_prev: previous point used to generate the momentum update.
    momentum: amount of momentum applied, must be positive.

  Returns:
    Updated point with momentum.
  """
  pos_curr = latlon_to_cartesian(
      lat=np.array(latlon_curr[0]),
      lon=np.array(latlon_curr[1]),
  )
  pos_curr = np.array(pos_curr)

  pos_prev = latlon_to_cartesian(
      lat=np.array(latlon_prev[0]),
      lon=np.array(latlon_prev[1]),
  )
  pos_prev = np.array(pos_prev)

  pos_next = cartesian_spherical_geodesic_momentum_update(
      pos_curr=pos_curr,
      pos_prev=pos_prev,
      momentum=momentum,
  )

  lat, lon = cartesian_to_latlon(*pos_next[:, None])

  return (lat.item(), lon.item())
