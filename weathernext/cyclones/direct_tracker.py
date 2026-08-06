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
"""Direct tracker using gridded IBTrACS data.

The direct tracker is a conceptually simple piece of code that runs on top of
generated forecasts. It operates greedily, meaning that the way it decides where
to assign next storm centers only depends on previous storm centers and not on
future centers, unlike, for example, Tempest Extremes. The tracker operates as
follows:

  1. Initialise cyclone centers, either using IBTrACS known centers or using
     direct cyclogenesis. In direct cyclogenesis, the tracker searches for
     areas of high-probability in the cyclone existence fields produced by the
     direct cyclone model (at lead time t=0), and assigns cyclone centers one by
     one, starting with the areas of highest probability and proceeding until a
     specified probability threshold is reached.
  2. Once cyclone centers are initialised the tracker proceeds to evolve them in
     time. The tracker can be forced to produce forecasts for all lead
     times provided in the gridded forecasts, or tracks can be dissipated
     the first time the cyclone existence probability scalar decreases below a
     user-specified threshold.
     In each step, the tracker:

        2.1. Produces a first guess for where the next cyclone center will be,
             based on the current and the previous cyclone center (by applying)
             a momentum update. If a previous cyclone center is not available,
             the tracker uses zero momentum.
        2.2. After producing a first guess, the tracker then refines its guess,
             by one of the following three tracking modes: `mean`, `mode` or
             `mode-then-mean`. In `mean`, the tracker simply updates its
             estimate by taking the mean position in a neighbourhood around the
             guess from 2.1, and in `mode` it uses the mode i.e. highest
              probability position instead. In `mode-then-mean` it performs an
             initial estimate using the mode of the predicted probability field
             and then it applies another refinement by taking a mean with a
             smaller refinement radius. This last option combines the best of
             both worlds, by using the mode to get a first estimate of where the
             probability blob is (without getting confused when two blobs are
             close to one another), and then refines the estimate with the mean.
        2.3. After determining the next cyclone center, the tracker estimates
             all accompanying scalar variables, e.g. max sustained wind speeds,
             that are present in the gridded forecast (and match the standard
             cyclone variable field names).
  3. This process is repeated for all initialised storms, and the results are
     collected in a single pandas dataframe.
"""

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

from absl import logging
from weathernext.cyclones import constants
from weathernext.cyclones import cyclone_utils
from weathernext.cyclones import ibtracs_processing_utils as ibtracs_utils
from weathernext.cyclones import tracker_base as base
from weathernext.cyclones import tracker_utils as utils
import numpy as np
import pandas as pd
import xarray as xr


# Griddified cyclone variables are named using the form
# '{prefix}_{variable}_{suffix}' where prefix is, for example, "cyclone",
# variable is, for example, usa_wind, cma_wind, and so on, and suffix is,
# for example, "disc".
CYCLONES_VARIABLES_PREFIX = "cyclone"
CYCLONES_SCALAR_VARIABLES_SUFFIX = "disc"

DIRECT_TRACKER_CYCLONE_VARIABLES = tuple(
    ibtracs_utils.get_cyclone_variable_names()
)

KM_PER_NMILE = 1.852

SIZE_OF_1_DEG_AT_EQUATOR_KM = 111.3
# The meridian length is the distance from the North to the South pole along
# the great arc.
EARTH_MERIDIAN_LENGTH_KM = 20_004

PROBABILITY_CYCLONE_EXISTS_DATAFRAME_NAME = "prob_cyclone_exists"

# When the DirectTracker is called with a DataFrame of initial storm data, it
# will fill in missing cyclone existence probabilities with this value.
# It just needs to be an arbitrary value larger than the dissipation threshold
# set in the tracker config (if any).
PROBABILITY_CYCLONE_EXISTS_FILLVALUE_FOR_T0 = 1.0

# Collection of "scalar" variables that the tracker will track. These are
# scalar properties of the cyclone (e.g. wind speed) that the tracker estimates
# by averaging a variable field by the field of cyclone existence probabilities.
ALL_SCALAR_VARIABLES_TO_TRACK = (
    ibtracs_utils.IBTRACS_WIND_VARIABLES
    + ibtracs_utils.IBTRACS_MIN_SEA_LEVEL_PRESSURE_VARIABLES
    + ibtracs_utils.QUADRANT_RADIUS_VARIABLES
    + ibtracs_utils.IBTRACS_MAX_WIND_RADIUS_VARIABLES
)


ALL_USA_SCALAR_VARIABLES_TO_TRACK = tuple(
    [v for v in ALL_SCALAR_VARIABLES_TO_TRACK if v.startswith("usa")]
)


# Map from prediction field names to standard names of the tracker columns.
IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME = {
    "usa_wind": constants.MAX_SUSTAINED_WIND_SPEED_KNOTS,
    "usa_rmw": constants.RADIUS_OF_MAXIMUM_WINDS,
    "usa_pres": constants.MIN_SEA_LEVEL_PRESSURE_HPA,
    "usa_r34_se_radius": constants.RADIUS_34_KNOT_WINDS_SE_KM,
    "usa_r34_nw_radius": constants.RADIUS_34_KNOT_WINDS_NW_KM,
    "usa_r34_sw_radius": constants.RADIUS_34_KNOT_WINDS_SW_KM,
    "usa_r34_ne_radius": constants.RADIUS_34_KNOT_WINDS_NE_KM,
    "usa_r50_se_radius": constants.RADIUS_50_KNOT_WINDS_SE_KM,
    "usa_r50_nw_radius": constants.RADIUS_50_KNOT_WINDS_NW_KM,
    "usa_r50_sw_radius": constants.RADIUS_50_KNOT_WINDS_SW_KM,
    "usa_r50_ne_radius": constants.RADIUS_50_KNOT_WINDS_NE_KM,
    "usa_r64_se_radius": constants.RADIUS_64_KNOT_WINDS_SE_KM,
    "usa_r64_nw_radius": constants.RADIUS_64_KNOT_WINDS_NW_KM,
    "usa_r64_sw_radius": constants.RADIUS_64_KNOT_WINDS_SW_KM,
    "usa_r64_ne_radius": constants.RADIUS_64_KNOT_WINDS_NE_KM,
}

IBTRACS_VARIABLE_NAME_TO_GRIDDED_VARIABLE_NAME = {
    v: f"{CYCLONES_VARIABLES_PREFIX}_{v}_{CYCLONES_SCALAR_VARIABLES_SUFFIX}"
    for v in IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME.keys()
}

PREDICTION_FIELD_NAME_TO_TRACKER_COLUMN_NAME = {
    f"{CYCLONES_VARIABLES_PREFIX}_{k}": v
    for k, v in IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME.items()
}

NON_NEGATIVE_VARIABLES_TO_CLIP = [
    constants.MAX_SUSTAINED_WIND_SPEED_KNOTS,
    constants.RADIUS_OF_MAXIMUM_WINDS,
    constants.MIN_SEA_LEVEL_PRESSURE_HPA,
    *constants.QUADRANT_RADII_FLATTENED,
]

TRACKER_DATAFRAME_NUMERIC_COLUMNS = [
    constants.LAT,
    constants.LON,
    PROBABILITY_CYCLONE_EXISTS_DATAFRAME_NAME,
    constants.TRACK_MERGE,
] + list(IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME.values())

# Pre-computed mapping from tracker column names to gridded variable names,
# used by the bilinear interpolation method to avoid rebuilding this mapping
# on every call.
TRACKER_COLUMN_TO_GRIDDED_VARIABLE_NAME = {
    IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME[
        v
    ]: IBTRACS_VARIABLE_NAME_TO_GRIDDED_VARIABLE_NAME[v]
    for v in ALL_USA_SCALAR_VARIABLES_TO_TRACK
}

DEFAULT_VARIABLES_TO_CHECK_FOR_NANS = [constants.LAT, constants.LON]

COLS_TO_DROP_FROM_INITIAL_STORM_DATAFRAME_IF_PRESENT = [
    # Only relevant internally to ground truth data.
    constants.TIME_SINCE_START,
]

LatLon = Tuple[float, float]
ProbabilityCycloneExists = float
ScalarVariables = Dict[str, float | None]

# Cyclones never cross this latitude:
# https://ncics.org/ibtracs/index.php?name=browse-location
MAX_ABSOLUTE_LATITUDE = 80


def _single_row_dataframe_from_dict(row_dict: Dict[str, Any]) -> pd.DataFrame:
  return pd.DataFrame.from_dict({k: [v] for k, v in row_dict.items()})


def _get_bounding_box_sides_in_degrees(
    latlon: LatLon,
    disc_radius_km: float,
) -> Tuple[float, float]:
  """Computes sides of smallest box containing a disc."""
  # First, select a smaller region of the grid around the latlon.
  disc_diameter_km = 2 * disc_radius_km
  box_side_in_degrees_lat = disc_diameter_km / SIZE_OF_1_DEG_AT_EQUATOR_KM
  # The longitude-side of the smallest box that contains the disc depends on
  # the latitude of the disc center and the radius of the disc itself. Two
  # cases that illustrate the longitude-side should be a function of both
  # the latitude of the disc center and the radius of the disc:
  #   1. Consider the case where we fix the radius of the disc and start
  #      increasing the latitude from 0 to 90. At some point, the disc will
  #      contain the pole, so the disc will contain the entire [0, 360]
  #      range of longitudes close to the pole, so any bounding box that
  #      contains the disc must have a longitude side of 360 degrees.
  #   2. Consider a disc at a fixed location, close to a pole. As we
  #      increase the radius of the disc, it will eventually contain the
  #      pole, so the again the box must contain the entire [0, 360] range.
  # We ensure this by using the northernmost and southernmost latitudes
  # contained in the disc. Pretending the disc is centered at each of these
  # latitudes scales the size of the box conservatively to ensure it
  # contains the entire disc.
  disc_half_angle_degrees = (disc_radius_km / EARTH_MERIDIAN_LENGTH_KM) * 180.0
  max_lat = min(90.0, latlon[0] + disc_half_angle_degrees)
  min_lat = max(-90.0, latlon[0] - disc_half_angle_degrees)

  # The factor by which we must divide the side of the box to ensure it
  # contains the disc.
  box_side_div_factor = np.minimum(
      np.cos(np.deg2rad(max_lat)),
      np.cos(np.deg2rad(min_lat)),
  )

  logging.info("Box side in degrees, original: %s", box_side_in_degrees_lat)
  box_side_in_degrees_lon = float(box_side_in_degrees_lat / box_side_div_factor)
  logging.info("Box side in degrees, scaled: %s", box_side_in_degrees_lat)
  # If the box wraps around the longitude boundary, which can happen for
  # large disc radii or cyclones near the poles (unlikely but can happen if
  # tracks are not being dissipated), then we need to make sure that the box
  # side in degrees longitude is at most the longitude range in degrees.
  # To avoid edge case with extreme or NaN box longitude sides when the
  # central latitude approaches the poles, we take a nanmin with 360.
  box_side_in_degrees_lon = np.nanmin(
      [box_side_in_degrees_lon, utils.LON_DEG_RANGE]
  )
  return box_side_in_degrees_lat, box_side_in_degrees_lon


def _average_in_three_dimensions_and_project_on_sphere(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> LatLon:
  """Computes average of points on sphere and re-projects back onto the sphere.

  The function first converts the latlon coordinates into three-dimensional
  Cartesian coordinates. Then, it computes the average of these points to
  obtain an average again in three dimensions which, however, will not in
  general lie on the unit sphere. We then re-project the point back onto the
  sphere and convert back to latlon spherical coordinates.

  Note that the expectation operation is not defined on a sphere, so this
  projection operation is a halfway solution to computing a weighted average
  which is somewhat like an expectation.

  Args:
    lats: latitudes of the points (same shape as longitudes)
    lons: longitudes of the points (same shape as latitudes)
    probs: probabilities by which to weigh each point (same shape as lats,
      lons). If None, then the points are averaged without weighting.

  Returns:
    lat, lon: result of averaging in three dimensions and projecting on sphere
  """
  assert len(lats.shape) == 1
  assert len(lons.shape) == 1

  if probs is None:
    probs = np.ones_like(lats)

  # Convert latitude-longitude coordinates to unit sphere cartesian coordinates
  x, y, z = cyclone_utils.latlon_to_cartesian(lats, lons)
  r = np.stack([x, y, z], axis=-1)

  # Expand extra dimension and compute spatial expectation
  r = np.mean(r * probs[:, None], axis=0)

  # Project back onto the sphere by normalising, and convert back to latlon
  r = r[:, None]
  x, y, z = r / np.sum(r**2) ** 0.5

  lat, lon = cyclone_utils.cartesian_to_latlon(x=x, y=y, z=z)

  assert lat.size == lon.size == 1
  assert len(lat.shape) == len(lon.shape) == 1
  lat = lat[0]
  lon = lon[0]

  return float(lat), float(lon)


class DirectTracker(base.CycloneTracker):
  """Tracks cyclones from griddified IBTrACS probability and attributes.

  disc_radius_scalar_variables_km: radius of geodesic disc, in km, used to
    estimate scalar variables. If None, scalar variables will be estimated using
    bilinear interpolation to read off the scalar variables in the
    neighbourhood of the cyclone center.
  disc_radius_mean_latlon_km: radius of geodesic disc, in km, used to compute
    the mean cyclone center latlons produced in the direct tracker.
  disc_radius_mode_latlon_km: radius of geodesic disc, in km, used to compute
    mode-jump updates in the direct tracker (if using mode-then-mean). This is
    the maximum radius of mode-jumps allowed in the direct tracker.
  disc_radius_mean_probability_of_existence_km: radius of geodesic disc,
    in km, used to compute the mean probability of cyclone existence.
  tracking_mode: tracking mode to use. currently supports "mean" (predicting
    mean position in gridded window), "mode" (predicting mode, i.e. position
    with maximum probability mass in gridded window) and "mode_then_mean"
    (i.e. first making an initial guess by predicting the mode and then
    refining that guess using the mean).
  momentum_constant: momentum constant to use (must be between 0. and 1.)
  temporal_resolution_hours: expected temporal resolution of the gridded
    forecasts, in hours. This is used to check the gridded forecasts received
    by the tracker are at the expected temporal resolution. Note that the
    tracker also checks that this temporal resolution is uniform (i.e. varying
    lead times are not allowed).
  prune_nearby_cyclones: whether to prune cyclone centers at a given time step
    if they are closer than distance_for_pruning_nearby_cyclones_km.
  distance_for_pruning_nearby_cyclones_km: the tracker will prune cyclone
    centers at a given time step if they are closer than this threshold.
  dissipation_min_mean_probability_threshold: probability threshold below which
    the tracker will stop tracking a cyclone. If None, the tracker will not
    dissipate cyclones and will continue tracking them until the end of the
    forecast horizon.
  cyclogenesis_min_max_probability_threshold: threshold below which any
    neighbourhoods with max probability are excluded from the result. This is
    used in cyclogenesis initialisation only.
  cyclone_prob_exists_scale_factor: scale factor to apply to the cyclone
    existence probability variable.
  enforce_physically_consistent_quadrants_and_winds: whether to enforce
    physically consistent quadrants, winds and radii by thresholding at the end
    of tracking. Defaults to False, which does not apply this operation.
  """

  def __init__(
      self,
      disc_radius_scalar_variables_km: float | None,
      disc_radius_mean_latlon_km: float,
      disc_radius_mode_latlon_km: float,
      disc_radius_mean_probability_of_existence_km: float,
      disc_radius_cyclogenesis_refinement_km: float,
      tracking_mode: str,
      momentum_constant: float,
      min_disc_radius_between_cyclogenesis_candidates_km: float,
      temporal_resolution_hours: int,
      prune_nearby_cyclones: bool = False,
      distance_for_pruning_nearby_cyclones_km: Optional[float] = None,
      probability_cyclone_exists_variable_name: Optional[str] = None,
      cyclogenesis_box_side_degrees: Optional[float] = None,
      cyclogenesis_min_max_probability_threshold: Optional[float] = None,
      cyclogenesis_min_mean_probability_threshold: Optional[float] = None,
      cyclogenesis_minimum_duration: Optional[pd.Timedelta] = None,
      dissipation_min_mean_probability_threshold: Optional[float] = None,
      cyclone_prob_exists_scale_factor: float = 1.0,
      enforce_physically_consistent_quadrants_and_winds: bool = False,
      rmw_and_quad_clipping_mode: Literal[
          "clip_rmw_above_using_quads",
          "clip_quads_below_using_rmw",
      ] = "clip_quads_below_using_rmw",
      drop_track_merge_indicator: bool = True,
  ):
    assert momentum_constant >= 0.0 and momentum_constant <= 1.0

    self.temporal_resolution_hours = temporal_resolution_hours

    self.disc_radius_scalar_variables_km = disc_radius_scalar_variables_km
    self.disc_radius_mean_latlon_km = disc_radius_mean_latlon_km
    self.disc_radius_mode_latlon_km = disc_radius_mode_latlon_km
    self.disc_radius_mean_probability_of_existence_km = (
        disc_radius_mean_probability_of_existence_km
    )
    self.disc_radius_cyclogenesis_refinement_km = (
        disc_radius_cyclogenesis_refinement_km
    )

    self.min_disc_radius_between_cyclogenesis_candidates_km = (
        min_disc_radius_between_cyclogenesis_candidates_km
    )

    self.distance_for_pruning_nearby_cyclones_km = (
        distance_for_pruning_nearby_cyclones_km
    )
    self.tracking_mode = tracking_mode
    self.momentum_constant = momentum_constant
    self.cyclogenesis_box_side_degrees = cyclogenesis_box_side_degrees
    self.cyclogenesis_min_max_probability_threshold = (
        cyclogenesis_min_max_probability_threshold
    )
    self.dissipation_min_mean_probability_threshold = (
        dissipation_min_mean_probability_threshold
    )
    if cyclogenesis_min_mean_probability_threshold is None:
      self.cyclogenesis_min_mean_probability_threshold = (
          self.dissipation_min_mean_probability_threshold
      )
    else:
      self.cyclogenesis_min_mean_probability_threshold = (
          cyclogenesis_min_mean_probability_threshold
      )
    self.cyclogenesis_minimum_duration = cyclogenesis_minimum_duration
    self.probability_cyclone_exists_variable_name = (
        probability_cyclone_exists_variable_name
    )
    self.cyclone_prob_exists_scale_factor = cyclone_prob_exists_scale_factor
    self.prune_nearby_cyclones = prune_nearby_cyclones
    self.enforce_physically_consistent_quadrants_and_winds = (
        enforce_physically_consistent_quadrants_and_winds
    )
    self.rmw_and_quad_clipping_mode = rmw_and_quad_clipping_mode
    self.drop_track_merge_indicator = drop_track_merge_indicator

  def preprocess_gridded_ds(
      self,
      gridded_ds: xr.Dataset,
      extras_ds: Optional[xr.Dataset] = None,
  ) -> xr.Dataset:
    """Selects the variables required by the direct tracker."""
    variables = [
        variable
        for variable in gridded_ds.variables
        if variable in DIRECT_TRACKER_CYCLONE_VARIABLES
    ]
    # Materialising the full dataset with the selected variables
    # should generally be a good trade-off. If RAM allows it, it means
    # the tracker doesn't make many materialisations from the dataset
    # during tracking, which adds an overhead.
    # If this leads to OOMs, we can remove this .compute().
    return gridded_ds[variables].compute()

  def _get_disc_windowed_gridded_probs_and_latlons(
      self,
      gridded_predictions: xr.Dataset,
      latlon: LatLon,
      disc_radius_km: float,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets windowed gridded probs and latlons for window centered at latlon.

    Args:
      gridded_predictions: gridded predictions including the field specified by
        self.probability_cyclone_exists_variable_name.
      latlon: latlon to center probability window at.
      disc_radius_km: radius, in km, of geodesic disc used to select predicted
        probabilities.

    Returns:
      gridded probabilities, lats and lons of window centered at latlon.
    """

    # Select gridded probabilities
    gridded_probs = gridded_predictions[
        self.probability_cyclone_exists_variable_name
    ]

    all_grid_latlons, is_in_radius = self._get_is_in_radius_grid_around_latlon(
        gridded_predictions=gridded_predictions,
        latlon=latlon,
        disc_radius_km=disc_radius_km,
    )

    # Slice out probabilities and latlons that are in the disc.
    disc_gridded_probs = gridded_probs.values[is_in_radius].flatten()
    disc_latlons = all_grid_latlons[is_in_radius]

    # Sum probabilities and normalise
    disc_gridded_probs = disc_gridded_probs / np.sum(disc_gridded_probs)

    # Separate all_grid_latlons into lats and lons
    disc_lats, disc_lons = disc_latlons.T

    assert (
        disc_lats.shape == disc_lons.shape
    ), f"{disc_lats.shape=} {disc_lons.shape=}"
    assert (
        disc_lats.shape == disc_gridded_probs.shape
    ), f"{disc_lats.shape=} {disc_gridded_probs.shape=}"

    return disc_gridded_probs, disc_lats, disc_lons

  def _get_mean_latlon_of_grid_points_within_disc(
      self,
      gridded_predictions: xr.Dataset,
      latlon: LatLon,
      disc_radius_km: float | None = None,
  ) -> LatLon:
    """Compute mean latlon position using a window centered at latlon.

    Args:
      gridded_predictions: gridded predictions including the field specified by
        self.probability_cyclone_exists_variable_name.
      latlon: latlon to center probability window at.
      disc_radius_km: radius, in km, of geodesic disc used to select predicted
        probabilities. If None, the default disc_radius_mean_latlon_km is used.

    Returns:
      mean latlon as tuple of floats.
    """

    # Get windowed gridded probabilities, latitudes and longitudes
    gridded_probs, lats, lons = (
        self._get_disc_windowed_gridded_probs_and_latlons(
            gridded_predictions=gridded_predictions,
            latlon=latlon,
            disc_radius_km=disc_radius_km or self.disc_radius_mean_latlon_km,
        )
    )

    # Compute expectation of windowed grid
    return _average_in_three_dimensions_and_project_on_sphere(
        lats=lats,
        lons=lons,
        probs=gridded_probs,
    )

  def _get_mode_latlon_of_grid_points_within_disc(
      self,
      gridded_predictions: xr.Dataset,
      latlon: LatLon,
  ) -> LatLon:
    """Compute mode latlon position using a window centered at latlon.

    Args:
      gridded_predictions: gridded predictions including the field specified by
        self.probability_cyclone_exists_variable_name.
      latlon: latlon to center probability window at.

    Returns:
      mode latlon as tuple of floats.
    """

    # Get windowed gridded probabilities, latitudes and longitudes
    gridded_probs, lats, lons = (
        self._get_disc_windowed_gridded_probs_and_latlons(
            gridded_predictions=gridded_predictions,
            latlon=latlon,
            disc_radius_km=self.disc_radius_mode_latlon_km,
        )
    )

    # Determine index of entry where gridded probs is maximum
    idx_argmax = np.unravel_index(
        np.argmax(gridded_probs),
        shape=gridded_probs.shape,
    )

    latlon = np.stack([lats, lons], axis=-1)[idx_argmax]

    return float(latlon[0]), float(latlon[1])

  def _get_is_in_radius_grid_around_latlon(
      self,
      latlon: LatLon,
      gridded_predictions: Union[xr.Dataset, xr.DataArray],
      disc_radius_km: float,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Creates gridded mask denoting whether each point is in radius of latlon.

    This function first selects a smaller region of the grid around the provided
    latlon, and then computes a boolean mask denoting which points in this
    smaller region are within the specified radius of the latlon.

    Args:
      latlon: latlon where geodesic disc is centered.
      gridded_predictions: gridded predictions from which to extract grid
        latlons.
      disc_radius_km: radius of geodesic disc, in km, to use.

    Returns:
      gridded latlons, and gridded mask denoting whether each point is in radius
      of latlon.
    """

    box_side_in_degrees_lat, box_side_in_degrees_lon = (
        _get_bounding_box_sides_in_degrees(latlon, disc_radius_km)
    )
    logging.info("Box side in degrees, final: %s", box_side_in_degrees_lon)
    sub_grid = utils.slice_data_array_latlon_box_with_lon_wraparound(
        array=gridded_predictions,  # pyrefly: ignore[bad-argument-type]
        box_center_lat=latlon[0],
        box_center_lon=latlon[1] % 360.0,
        box_side_in_degrees_lat=box_side_in_degrees_lat,
        box_side_in_degrees_lon=box_side_in_degrees_lon,
    )

    all_grid_latlons = np.stack(
        np.meshgrid(
            gridded_predictions.lat.values,
            gridded_predictions.lon.values,
            indexing="ij",
        ),
        axis=-1,
    )

    sub_grid_latlons = np.stack(
        np.meshgrid(
            sub_grid.lat.values,
            sub_grid.lon.values,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(
        -1, 2
    )  # (num_lon * num_lat, 2)

    is_in_radius_subgrid = (
        cyclone_utils.geodesic_distance(
            latlon0=np.array(latlon)[None, :],
            latlon1=sub_grid_latlons,
        )
        < disc_radius_km
    ).astype(
        bool
    )  # shape (num_lon * num_lat,)

    is_in_radius_full_grid = np.zeros(
        gridded_predictions.lat.shape + gridded_predictions.lon.shape,
        dtype=bool,
    )

    # Get the indices of the sub-grid in the original grid.
    sub_grid_lats = sub_grid.lat.values
    sub_grid_lons = sub_grid.lon.values
    lat_indices = np.searchsorted(
        gridded_predictions.lat.values, sub_grid_lats, side="left"
    )
    lon_indices = np.searchsorted(
        gridded_predictions.lon.values, sub_grid_lons, side="left"
    )

    # Set the boolean mask in the original grid to True for points in the
    # sub-grid that are within the radius.
    is_in_radius_full_grid[lat_indices[:, None], lon_indices[None, :]] = (
        is_in_radius_subgrid.reshape(sub_grid_lats.shape + sub_grid_lons.shape)
    )

    return all_grid_latlons, is_in_radius_full_grid

  def _get_mean_scalar_variables_within_mask(
      self,
      gridded_predictions: xr.Dataset,
      boolean_mask_to_include: np.ndarray,
  ) -> ScalarVariables:
    """Estimates averaged scalar variables handled by the tracker.

    Args:
      gridded_predictions: gridded predictions containing scalar variables.
      boolean_mask_to_include: mask of the same shape as gridded_predictions,
        denoting which entries of the gridded predictions should be included in
        the max. wind speed estimation calculations.

    Returns:
      estimated scalar variables for each source as floats.
    """

    # Get field of probabilities of existence and normalize, to obtain a field
    # of conditional probabilities, i.e. probability that the cyclone center is
    # in a particular grid cell, given that it exists.
    prob_exists_field = gridded_predictions[
        self.probability_cyclone_exists_variable_name
    ].values
    prob_exists_vector = prob_exists_field[boolean_mask_to_include]
    # Shape (N_grid_points_in_mask,):
    prob_exists_vector = prob_exists_vector / np.sum(prob_exists_vector)

    averaged_variables = dict()

    # Stack all scalar variables into a single array
    scalar_vars = []
    for variable in ALL_USA_SCALAR_VARIABLES_TO_TRACK:
      field_name = IBTRACS_VARIABLE_NAME_TO_GRIDDED_VARIABLE_NAME[variable]
      if field_name in gridded_predictions:
        scalar_vars.append(field_name)
      else:
        # Don't add variables that are not in the gridded_predictions
        pass

    # If there are no scalar variables, return empty dict.
    if not scalar_vars:
      return dict()

    scalar_vars_stacked = gridded_predictions[scalar_vars].to_dataarray()
    # Shape (N_variables, N_grid_points_in_mask):
    scalar_vars_vectors = scalar_vars_stacked.values[:, boolean_mask_to_include]

    # Compute average of scalar variables, weighted by the conditional
    # probabilities of cyclone existence at each grid point.
    averaged_scalar_vars = np.sum(
        scalar_vars_vectors * prob_exists_vector[None, :],
        axis=-1,
    )

    for i, variable in enumerate(ALL_USA_SCALAR_VARIABLES_TO_TRACK):
      averaged_variables[
          IBTRACS_VARIABLE_NAME_TO_TRACKER_COLUMN_NAME[variable]
      ] = averaged_scalar_vars[i]

    return averaged_variables

  def _get_scalar_variables_via_bilinear_interpolation(
      self,
      gridded_predictions: xr.Dataset,
      latlon: LatLon,
  ) -> ScalarVariables:
    """Estimates averaged scalar variables handled by the tracker.

    This function estimates the scalar variables produced by the tracker by
    performing linear interpolation on the gridded predictions, at the latitude
    and longitude of the cyclone center.

    Args:
      gridded_predictions: gridded predictions containing scalar variables.
      latlon: cyclone center latlon as tuple of floats.

    Returns:
      estimated scalar variables for each source as floats.
    """
    return utils.bilinear_interpolation_with_lon_wraparound(  # pyrefly: ignore[bad-return]
        gridded_predictions,
        latlon,
        TRACKER_COLUMN_TO_GRIDDED_VARIABLE_NAME,
    )

  def _try_to_infer_probability_cyclone_exists_variable_name(
      self,
      gridded_predictions: xr.Dataset,
  ) -> str:
    """Tries to infer the variable name for the prob. of cyclone existence."""

    # If no existence variable specified, try to infer it from
    # gridded_predictions
    existence_variables = set(
        variable
        for variable in gridded_predictions.variables
        if "exists" in str(variable)
    )

    if len(existence_variables) != 1:
      raise ValueError(
          "Tried to infer which cyclone probability existence variable to use, "
          f"but found variable names {existence_variables}."
      )

    return str(existence_variables.pop())

  def _get_mean_cyclone_probability_within_mask(
      self,
      gridded_probs: xr.DataArray,
      boolean_mask_to_include: np.ndarray,
  ) -> ProbabilityCycloneExists:
    """Computes an estimate for the probability of existence of a cyclone.

    Computes an estimate for the probability of existence of a cyclone by using
    a probability-like field gridded_probabilities, selecting the grid points
    within boolean mask, summing these probabilities and scaling the result by
    the cyclone_prob_exists_scale_factor.

    Args:
      gridded_probs: gridded probabilities to use.
      boolean_mask_to_include: boolean array of the same shape as gridded_probs,
        with ones for marginal probabilities to include in the estimation.

    Returns:
      estimate for the probability of cyclone existence.
    """

    assert boolean_mask_to_include.shape == gridded_probs.values.shape
    assert boolean_mask_to_include.dtype == bool

    prob = np.mean(gridded_probs.values * boolean_mask_to_include.astype(float))

    return min(prob * self.cyclone_prob_exists_scale_factor, 1.0)

  def _get_cyclogenesis_candidates(
      self,
      gridded_predictions: xr.DataArray,
  ) -> xr.DataArray:
    """Gets candidate lat/lons for cyclogenesis.

    Efficiently computes an xr.DataArray containing candidate lat/lons for
    cyclogenesis using vectorisation:
      1) Reshaping the (N_lat, N_lon) gridded predictions into an (N_boxes,
      N_grid_cells_per_box) array.
      2) Computes the max probability and argmax (lat/lon location) along
      the grid cell dimension.
      3) Filters out boxes with max probability below a threshold.
      4) Sorts the remaining boxes by max probability (highest to lowest).

    These candidates are refined or discarded in
    _prune_and_refine_cyclogenesis_candidates.

    Args:
      gridded_predictions: gridded predictions containing existence
        probabilities.

    Returns:
      A DataArray containing the lat/lon candidates for cyclogenesis.
    """
    gridded_cyclone_probability_da = gridded_predictions[
        self.probability_cyclone_exists_variable_name
    ]
    lat_size = gridded_cyclone_probability_da.lat.size
    lon_size = gridded_cyclone_probability_da.lon.size
    lat_range = (
        gridded_cyclone_probability_da.lat.max()
        - gridded_cyclone_probability_da.lat.min()
    ).data.item()
    lon_range = (
        gridded_cyclone_probability_da.lon.max()
        - gridded_cyclone_probability_da.lon.min()
    ).data.item()
    window_size_lat = max(
        1, round(self.cyclogenesis_box_side_degrees / lat_range * lat_size)
    )
    window_size_lon = max(
        1, round(self.cyclogenesis_box_side_degrees / lon_range * lon_size)
    )
    coarsened_predictions = gridded_cyclone_probability_da.coarsen(
        lat=window_size_lat,
        lon=window_size_lon,
        boundary="pad",
    )

    # Shape: (window_idx_lat, window_lat_idx, window_idx_lon, window_lon_idx)
    constructed_da = coarsened_predictions.construct(
        window_dim={
            constants.LAT: ("window_idx_lat", "window_lat_idx"),
            constants.LON: ("window_idx_lon", "window_lon_idx"),
        },
        keep_attrs=True,
    )
    # Shape: (window_idx, grid_cell_idx)
    flat_window_da = constructed_da.stack(
        window_idx=["window_idx_lat", "window_idx_lon"]
    ).stack(grid_cell_idx=["window_lat_idx", "window_lon_idx"])
    argmax = flat_window_da.argmax(dim="grid_cell_idx")
    maxprobs = flat_window_da.isel(grid_cell_idx=argmax)

    # Filter out boxes with max probability below threshold.
    max_probability_exceeding_threshold = maxprobs.where(
        maxprobs > self.cyclogenesis_min_max_probability_threshold,
        drop=True,
    )

    return max_probability_exceeding_threshold.to_dataset().sortby(
        self.probability_cyclone_exists_variable_name, ascending=False
    )

  def _prune_and_refine_cyclogenesis_candidates(
      self,
      gridded_predictions: xr.Dataset,
      cyclogenesis_candidates: xr.DataArray,
      active_cyclones: Optional[pd.DataFrame] = None,
  ) -> np.ndarray:
    """Prune and refine cyclogenesis centers from initial candidate lat/lons.

    For each neighbourhood:

      1. Select the mode of the probability in the neighbourhood as an initial
        guess for the cyclone center.
      2. If this initial estimate is close to another cyclone center that has
        previously been determined by this procedure, skip the following steps
        and do not add a new center for this neighbourhood.
      3. Take a disc centered at the mode estimate,
        and use this to refine the estimate by taking the mean position (as
        weighted by the probabilities in the disc).
      4. If the refined estimate is close to another cyclone center, do not add
        a new cyclone center (similar to step 2), otherwise add this new point
        as a new cyclone center.
      5. Repeat from step 1 using the next neighbourhood.

    The purpose of this mean-then-mode refinement is to handle edge cases where
    two or more cyclone blobs are in the same box window, by first selecting the
    mode to pick one of the blobs and then refining the estimate by taking a
    mean.

    Args:
      gridded_predictions: gridded predictions containing existence
        probabilities.
      cyclogenesis_candidates: lat/lon candidates for cyclogenesis.
      active_cyclones: active cyclones to avoid placing new centers near.

    Returns:
      lat/lons of final cyclogenesis centers as an (N, 2) numpy array.
    """

    if active_cyclones is None:
      centers_placed = np.zeros(shape=(0, 2))
    else:
      centers_placed = active_cyclones[[constants.LAT, constants.LON]].values

    cyclogenesis_centers = []
    cyclogenesis_candidates = cyclogenesis_candidates.to_dataframe()[
        [constants.LAT, constants.LON]
    ].values
    for candidate_latlon in cyclogenesis_candidates:
      if abs(candidate_latlon[0]) > MAX_ABSOLUTE_LATITUDE:
        logging.info(
            "Skipping candidate_latlon because it exceeded max latitude"
            " bounds: %s",
            candidate_latlon,
        )
        continue
      candidate_latlon = candidate_latlon[None, :]  # For broadcasting.
      # If mode placed close to an existing point, skip placing. We perform
      # this check *before* mean refinement, because mean refinement requires
      # is relatively computationally expensive.
      distance_guess_to_placed = cyclone_utils.geodesic_distance(
          centers_placed,
          candidate_latlon,
      )
      if (
          centers_placed.shape[0] > 0
          and np.min(distance_guess_to_placed)
          < self.min_disc_radius_between_cyclogenesis_candidates_km
      ):
        continue

      # Refine latlon guess by applying a mean estimate.
      assert candidate_latlon.shape == (1, 2)
      initial_latlon = candidate_latlon[0]
      assert initial_latlon.shape == (2,)
      refined_latlon = self._get_mean_latlon_of_grid_points_within_disc(
          gridded_predictions=gridded_predictions,
          latlon=initial_latlon,
          disc_radius_km=self.disc_radius_cyclogenesis_refinement_km,
      )

      # Check if the refined latlon has a probability of existence above the
      # cyclogenesis min mean probability threshold.
      probability_cyclone_exists = (
          self._get_cyclone_probability_of_existence_within_disc(
              gridded_predictions=gridded_predictions,
              latlon=refined_latlon,
          )
      )
      logging.info(
          "Checking whether to skip cyclogenesis candidate at %s with "
          "probability cyclone exists %s and cyclogenesis threshold %s.",
          refined_latlon,
          probability_cyclone_exists,
          self.cyclogenesis_min_mean_probability_threshold,
      )
      if (
          self.cyclogenesis_min_mean_probability_threshold is not None
          and probability_cyclone_exists
          < self.cyclogenesis_min_mean_probability_threshold
      ):
        logging.info(
            "Skipping cyclogenesis candidate at %s because probability (%f) is "
            "below cyclogenesis min mean probability threshold (%f)",
            refined_latlon,
            probability_cyclone_exists,
            self.cyclogenesis_min_mean_probability_threshold,
        )
        continue

      # The refined lat / lon may have pushed the center closer to an existing
      # storm, so we perform another minimum-distance check.
      refined_latlon = np.array(refined_latlon)[None, :]
      distances_to_placed_points = cyclone_utils.geodesic_distance(
          centers_placed,
          refined_latlon,
      )

      if (
          distances_to_placed_points.shape[0] == 0
          or np.min(distances_to_placed_points)
          > self.min_disc_radius_between_cyclogenesis_candidates_km
      ):
        centers_placed = np.concatenate((
            centers_placed,
            refined_latlon,
        ))
        cyclogenesis_centers.append(refined_latlon)

    if cyclogenesis_centers:
      return np.concatenate(cyclogenesis_centers)
    else:
      return np.zeros(shape=(0, 2))

  def _do_direct_cyclogenesis(
      self,
      gridded_predictions: xr.Dataset,
      active_cyclones: pd.DataFrame,
      lead_time: np.timedelta64,
  ) -> pd.DataFrame | None:
    """Performs direct cyclogenesis on predictions to get new cyclone centers.

    Performs direct cyclogenesis by first determining box-shaped probability
    neighbourhoods that have high enough mean probability (higher than the
    specified threshold), and placing cyclone centers in these neighbourhoods
    via an iterative greedy approach.

    Args:
      gridded_predictions: gridded predictions containing existence
        probabilities.
      active_cyclones: active cyclones to avoid placing new centers near.
      lead_time: the lead time to infer cyclogenesis for in the gridded
        predictions.

    Returns:
      Cyclone centers determined by direct cyclogenesis. If no new cyclones are
      created, returns None.
    """
    gridded_predictions = gridded_predictions.sel(time=lead_time).compute()
    cyclogenesis_candidates = self._get_cyclogenesis_candidates(
        gridded_predictions=gridded_predictions,  # pyrefly: ignore[bad-argument-type]
    )

    # Shape (N, 2):
    center_latlons = self._prune_and_refine_cyclogenesis_candidates(
        gridded_predictions=gridded_predictions,
        cyclogenesis_candidates=cyclogenesis_candidates,
        active_cyclones=active_cyclones,
    )

    all_cyclogenesis_dfs = []
    for latlon in center_latlons:
      single_cyclogenesis_dict = {
          constants.LAT: latlon[0],
          constants.LON: latlon[1],
          constants.LEAD_TIME: lead_time,
          constants.VALID_TIME: (
              pd.Timestamp(gridded_predictions.init_time.values) + lead_time
          ),
          # Temporarily add the 'cyclogenesis_' prefix to track IDs to
          # distinguish them from initialisation tracks passed to the tracker.
          constants.TRACK_ID: (
              f"cyclogenesis_{self._cyclogenesis_track_counter}"
          ),
          constants.TRACK_MERGE: 0.0,
      }
      probability_cyclone_exists = (
          self._get_cyclone_probability_of_existence_within_disc(
              gridded_predictions=gridded_predictions,
              latlon=latlon,
          )
      )
      logging.info(
          "Lead time %s, adding new cyclogenesis point with probability "
          "cyclone exists: %s",
          lead_time,
          probability_cyclone_exists,
      )
      if self.disc_radius_scalar_variables_km is not None:
        averaged_scalar_variables = self._get_all_scalar_variables_within_disc(
            gridded_predictions=gridded_predictions,
            latlon=latlon,
        )
      else:
        averaged_scalar_variables = (
            self._get_scalar_variables_via_bilinear_interpolation(
                gridded_predictions=gridded_predictions,
                latlon=latlon,
            )
        )

      single_cyclogenesis_dict.update(averaged_scalar_variables)
      single_cyclogenesis_dict[PROBABILITY_CYCLONE_EXISTS_DATAFRAME_NAME] = (
          probability_cyclone_exists
      )
      all_cyclogenesis_dfs.append(
          _single_row_dataframe_from_dict(single_cyclogenesis_dict)
      )

      # Avoid clashing track IDs.
      self._cyclogenesis_track_counter += 1

    logging.info(
        "Number of new cyclogenesis tracks: %s", len(all_cyclogenesis_dfs)
    )
    if all_cyclogenesis_dfs:
      cyclogenesis_df = pd.concat(all_cyclogenesis_dfs, ignore_index=True)
      logging.info("New cyclogenesis tracks: %s", cyclogenesis_df.to_string())
      return cyclogenesis_df  # pyrefly: ignore[bad-return]
    else:
      return None

  def _get_latlon_guess_with_momentum_update(
      self,
      latlon_at_lead_time_t: LatLon,
      latlon_at_lead_time_t_minus_one: LatLon | None,
  ) -> LatLon:
    """Estimates the next cyclone center with momentum.

    Args:
      latlon_at_lead_time_t: current latlon position of the cyclone center.
      latlon_at_lead_time_t_minus_one: latlon position of the cyclone center
        before latlon_at_lead_time_t (this can be a tuple of None, in which case
        momentum is not applied).

    Returns:
      predicted latlon as tuple of floats.
    """
    # If previous latlon is not None, apply momentum update
    if latlon_at_lead_time_t_minus_one is not None:
      return cyclone_utils.latlon_spherical_geodesic_momentum_update(
          latlon_curr=latlon_at_lead_time_t,
          latlon_prev=latlon_at_lead_time_t_minus_one,
          momentum=self.momentum_constant,
      )

    else:
      logging.warning(
          "Received prev_latlon=%s. This is probably unintended. Using the most"
          " recent cyclone position as an initial guess for the cyclone"
          " center.",
          latlon_at_lead_time_t_minus_one,
      )
      return latlon_at_lead_time_t

  def _advance_single_cyclone_active_at_lead_time_t(
      self,
      track: pd.DataFrame,
      lead_time_t_minus_one: np.timedelta64,
      lead_time_t: np.timedelta64,
      lead_time_t_plus_one: np.timedelta64,
      gridded_predictions: xr.Dataset,
      tracking_mode: str,
      track_id: str,
  ) -> pd.DataFrame:
    """Predicts cyclone trajectories for a given gridded prediction state.

    In order to predict the next cyclone position, the tracker first uses the
    last cyclone position, together with a momentum update if applicable, to
    form an initial guess for the cyclone position (note that if the previous
    latlon is None, no momentum is applied). The momentum update uses the last
    cyclone position, and the previous cyclone position, takes their difference
    and adds a fraction of this difference to the last cyclone position to form
    an initial guess:

      latlon = curr_latlon + momentum * (curr_latlon - prev_latlon)

    Then the tracker takes a latlon window around the initial guess and computes
    either the expected position of the cyclone in this window, or uses the mode
    of the distribution to produce the final estimate.

    Args:
      track: dataframe containing the current track of the cyclone.
      lead_time_t_minus_one: lead time of the previous forecast step, t-1.
      lead_time_t: lead time of the current forecast step, t.
      lead_time_t_plus_one: lead time of the next forecast step, t+1.
      gridded_predictions: gridded predictions to run the tracker on
      tracking_mode: tracking mode to use. currently supports "mean" (predicting
        mean position in gridded window), "mode" (predicting mode, i.e. position
        with maximum probability mass in gridded window) and "mode_then_mean"
        (i.e. first making an initial guess by predicting the mode and then
        refining that guess using the mean).
      track_id: track ID to use for the new track row.

    Returns:
      A row for the next step in the track. If the cyclone is dissipating, an
      empty dataframe is returned.
    """
    if lead_time_t not in track[constants.LEAD_TIME].values:
      raise ValueError(
          f"Lead time {lead_time_t} not found in track"
          f" {track[constants.LEAD_TIME]}."
      )

    track_at_lead_time = track.loc[track[constants.LEAD_TIME] == lead_time_t]
    latlon_at_lead_time_t = (
        track_at_lead_time[constants.LAT].iloc[0],
        track_at_lead_time[constants.LON].iloc[0],
    )

    if lead_time_t_minus_one in track[constants.LEAD_TIME].values:
      latlon_at_lead_time_t_minus_one = (
          track.loc[
              track[constants.LEAD_TIME] == lead_time_t_minus_one,
              constants.LAT,
          ].iloc[0],
          track.loc[
              track[constants.LEAD_TIME] == lead_time_t_minus_one,
              constants.LON,
          ].iloc[0],
      )
      logging.info(
          "Found lead_time_t_minus_one=%s in track:\n%s\nSetting "
          "latlon_at_lead_time_t_minus_one=%s (not None) which means that "
          "geodesic momentum update will be applied.",
          lead_time_t_minus_one,
          track.to_string(),
          latlon_at_lead_time_t_minus_one,
      )
    else:
      latlon_at_lead_time_t_minus_one = None
      logging.info(
          "Did not find lead_time_t_minus_one=%s in track:\n%s\nSetting "
          "latlon_at_lead_time_t_minus_one=%s which means that "
          "geodesic momentum update will *not* be applied.",
          lead_time_t_minus_one,
          track.to_string(),
          latlon_at_lead_time_t_minus_one,
      )

    logging.info(
        "latlon_at_lead_time_t: %s",
        latlon_at_lead_time_t,
    )
    logging.info(
        "latlon_at_lead_time_t_minus_one: %s",
        latlon_at_lead_time_t_minus_one,
    )
    lat_lon_t_plus_one = self._get_latlon_guess_with_momentum_update(
        latlon_at_lead_time_t=latlon_at_lead_time_t,
        latlon_at_lead_time_t_minus_one=latlon_at_lead_time_t_minus_one,
    )
    logging.info("lat_lon_t_plus_one: %s", lat_lon_t_plus_one)

    # Keep only the zero-time predictions
    gridded_predictions_t_plus_one = gridded_predictions.sel(
        time=lead_time_t_plus_one
    )
    if "forecast_datetime" in gridded_predictions_t_plus_one.dims:
      gridded_predictions_t_plus_one = gridded_predictions_t_plus_one.squeeze(
          "forecast_datetime"
      )

    # Handle mode and mean tracking cases separately
    if tracking_mode == "mode":
      lat_lon_t_plus_one = self._get_mode_latlon_of_grid_points_within_disc(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )
    elif tracking_mode == "mean":
      lat_lon_t_plus_one = self._get_mean_latlon_of_grid_points_within_disc(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )
    elif tracking_mode == "mode_then_mean":
      lat_lon_t_plus_one = self._get_mode_latlon_of_grid_points_within_disc(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )
      lat_lon_t_plus_one = self._get_mean_latlon_of_grid_points_within_disc(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )
    else:
      raise ValueError(
          "Expected tracking_mode in ('mean', 'mode', 'mode_then_mean'),"
          f" found {tracking_mode=}."
      )

    probability_cyclone_exists = (
        self._get_cyclone_probability_of_existence_within_disc(
            gridded_predictions=gridded_predictions_t_plus_one,
            latlon=lat_lon_t_plus_one,
        )
    )
    if (
        self.dissipation_min_mean_probability_threshold is not None
        and probability_cyclone_exists
        < self.dissipation_min_mean_probability_threshold
    ):
      logging.info(
          "Dissipating cyclones %s (prob=%f below threshold %f)",
          track_id,
          probability_cyclone_exists,
          self.dissipation_min_mean_probability_threshold,
      )
      # Return an empty dataframe to indicate cyclone dissipation.
      return pd.DataFrame()

    if self.disc_radius_scalar_variables_km is not None:
      scalar_variables = self._get_all_scalar_variables_within_disc(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )
    else:
      scalar_variables = self._get_scalar_variables_via_bilinear_interpolation(
          gridded_predictions=gridded_predictions_t_plus_one,
          latlon=lat_lon_t_plus_one,
      )

    track_row_t_plus_one = _single_row_dataframe_from_dict({
        constants.LAT: lat_lon_t_plus_one[0],
        constants.LON: lat_lon_t_plus_one[1],
        constants.LEAD_TIME: pd.Timedelta(lead_time_t_plus_one),
        constants.TRACK_ID: track_id,
        constants.VALID_TIME: (
            pd.Timestamp(gridded_predictions.init_time.values)
            + lead_time_t_plus_one
        ),
        constants.TRACK_MERGE: 0.0,
        PROBABILITY_CYCLONE_EXISTS_DATAFRAME_NAME: probability_cyclone_exists,
        **scalar_variables,
    })
    return track_row_t_plus_one

  @staticmethod
  def _check_for_nans_in_predicted_tracks(
      predicted_tracks: pd.DataFrame,
      ignore_lead_time_zero: bool = False,
      cols_to_check_for_nans: Optional[Sequence[str]] = None,
  ) -> None:
    """Checks for NaNs in the output track dataframe.

    Args:
      predicted_tracks: The output track dataframe.
      ignore_lead_time_zero: If True, ignore NaNs in the lead_time=0 column,
        which may have come from observed cyclone data.
      cols_to_check_for_nans: If not None, check for NaNs only in these columns.
        Otherwise, check for NaNs in all columns.
    """
    if ignore_lead_time_zero and "lead_time" in predicted_tracks.columns:
      predicted_tracks = predicted_tracks[
          predicted_tracks["lead_time"] > pd.Timedelta("0h")
      ]
    nan_cols = []
    for col in predicted_tracks.columns:
      if predicted_tracks[col].isnull().values.any():
        nan_cols.append(col)

    if cols_to_check_for_nans is None:
      cols_to_check_for_nans = predicted_tracks.columns

    unexpected_nan_cols = [
        col for col in nan_cols if col in cols_to_check_for_nans
    ]
    if unexpected_nan_cols:
      # Useful context for debugging:
      other_cols_to_log = [
          constants.INIT_TIME,
          constants.TRACK_ID,
          constants.LEAD_TIME,
      ]
      other_cols_to_log = [
          col for col in other_cols_to_log if col in predicted_tracks.columns
      ]
      cols_to_log = list(
          set(unexpected_nan_cols + other_cols_to_log)
      )  # Remove dupes.
      raise ValueError(
          f"Found NaN values in {unexpected_nan_cols} column of predicted"
          f" tracks: {predicted_tracks[cols_to_log]}"
      )

  def _get_cyclone_probability_of_existence_within_disc(
      self,
      gridded_predictions: xr.Dataset,
      latlon: tuple[float, float],
  ) -> float:
    """Returns the probability of cyclone existence within a disc."""
    return self._get_mean_cyclone_probability_within_mask(
        gridded_probs=gridded_predictions[
            self.probability_cyclone_exists_variable_name
        ],
        boolean_mask_to_include=self._get_is_in_radius_grid_around_latlon(
            latlon=latlon,
            gridded_predictions=gridded_predictions,
            disc_radius_km=self.disc_radius_mean_probability_of_existence_km,
        )[1],
    )

  def _get_all_scalar_variables_within_disc(
      self,
      gridded_predictions: xr.Dataset,
      latlon: tuple[float, float],
  ) -> ScalarVariables:
    """Returns all scalar variables within a disc around a given latlon."""
    return self._get_mean_scalar_variables_within_mask(
        gridded_predictions=gridded_predictions,
        boolean_mask_to_include=self._get_is_in_radius_grid_around_latlon(
            latlon=latlon,
            gridded_predictions=gridded_predictions,
            disc_radius_km=self.disc_radius_scalar_variables_km,  # pyrefly: ignore[bad-argument-type]
        )[1],
    )

  def _remove_nearby_cyclone_centers_based_on_which_moved_most(
      self,
      predictions_df: pd.DataFrame,
      lead_time_to_prune: pd.Timedelta,
      previous_lead_time: pd.Timedelta,
  ) -> pd.DataFrame:
    """Removes nearby cyclones based on which moved most."""

    # Make a copy of the dataframe to avoid modifying the original.
    predictions_df = predictions_df.copy(deep=True)

    # This is just used for logging, to avoid flooding the logs with the entire
    # dataframe at each iteration.
    lead_time_t_and_t_minus_one_df = predictions_df[
        (predictions_df[constants.LEAD_TIME] == lead_time_to_prune)
        | (predictions_df[constants.LEAD_TIME] == previous_lead_time)
    ]
    logging.info(
        "Pruning with lead_time_to_prune=%s, "
        "previous_lead_time=%s and "
        "lead_time_t_and_t_minus_one_df:\n%s",
        lead_time_to_prune,
        previous_lead_time,
        lead_time_t_and_t_minus_one_df.T.to_string(),
    )

    # In each iteration of this loop, we identify at most one pair of cyclones
    # that are too close to each other and remove one of them. We repeat this
    # process until no such pairs are found. The maximum number of iterations
    # is bounded by the number of cyclones, as we remove one cyclone in each
    # step.
    num_cyclones_at_lead_time = len(
        predictions_df[
            predictions_df[constants.LEAD_TIME] == lead_time_to_prune
        ]
    )
    logging.info(
        "Pruning with lead_time_to_prune=%s, num_cyclones_at_lead_time=%d",
        str(lead_time_to_prune),
        num_cyclones_at_lead_time,
    )
    for step in range(num_cyclones_at_lead_time):
      # Get cyclone centers at the most recent lead time.
      lead_time_t_df = predictions_df[
          predictions_df[constants.LEAD_TIME] == lead_time_to_prune
      ]
      logging.info(
          "Pruning with lead_time_to_prune=%s, iteration %d,"
          " lead_time_t_df:\n%s",
          str(lead_time_to_prune),
          step,
          lead_time_t_df.T.to_string(),
      )

      if len(lead_time_t_df) < 2:
        logging.info(
            "Fewer than two cyclones active (found %d) skipping cyclone "
            "pruning.",
            len(lead_time_t_df),
        )
        break

      # Get cyclone centers at the second most recent lead time, that also
      # appear at the latest lead time, i.e. if a cyclone has dissipated in the
      # previous lead time, we won't attempt to prune it.
      lead_time_t_minus_one_df = predictions_df[
          predictions_df[constants.LEAD_TIME] == previous_lead_time
      ]
      lead_time_t_minus_one_df = lead_time_t_minus_one_df[
          lead_time_t_minus_one_df[constants.TRACK_ID].isin(
              lead_time_t_df[constants.TRACK_ID]
          )
      ]
      logging.info(
          "Pruning with lead_time_to_prune=%s, iteration %d, "
          "lead_time_t_minus_one_df:\n%s",
          str(lead_time_to_prune),
          step,
          lead_time_t_minus_one_df.T.to_string(),
      )

      # Get the latlons of the cyclones at the most recent lead time, compute
      # pairwise distances and set self-distances to inf to avoid triggering
      # threshold on self distances.
      lead_time_t_latlons = lead_time_t_df[
          [constants.LAT, constants.LON]
      ].values
      distances = cyclone_utils.geodesic_distance(
          latlon0=lead_time_t_latlons[:, None, :],
          latlon1=lead_time_t_latlons[None, :, :],
      )
      np.fill_diagonal(distances, float("inf"))

      # Determine which distances are below the pruning threshold.
      pairs_below_distance_threshold = (
          distances < self.distance_for_pruning_nearby_cyclones_km
      )

      # If no pairs are under the threshold we're done. Otherwise, there's at
      # least one pair of cyclones under the threshold, one of which must be
      # removed.
      if not np.any(pairs_below_distance_threshold):
        logging.info(
            "No pairs of cyclones in the distance matrix distances=\n%s\n"
            "were below the pruning threshold of %s km. Stopping pruning loop.",
            distances,
            self.distance_for_pruning_nearby_cyclones_km,
        )
        break

      logging.info(
          "Found cyclones below pruning threshold of %s km "
          "in distance matrix \n%s\n",
          self.distance_for_pruning_nearby_cyclones_km,
          distances,
      )

      # Otherwise, pick the first pair of cyclones above the threshold, and
      # remove the one that moved the most.
      idx_below_thresh = np.where(pairs_below_distance_threshold)
      track_id_1 = lead_time_t_df.iloc[idx_below_thresh[0][0]][
          constants.TRACK_ID
      ]
      track_id_2 = lead_time_t_df.iloc[idx_below_thresh[1][0]][
          constants.TRACK_ID
      ]

      lead_time_t_cyclone_1_df = lead_time_t_df[
          lead_time_t_df[constants.TRACK_ID] == track_id_1
      ]
      lead_time_t_cyclone_2_df = lead_time_t_df[
          lead_time_t_df[constants.TRACK_ID] == track_id_2
      ]
      lead_time_t_latlons_1 = lead_time_t_cyclone_1_df[
          [constants.LAT, constants.LON]
      ].values
      lead_time_t_latlons_2 = lead_time_t_cyclone_2_df[
          [constants.LAT, constants.LON]
      ].values

      logging.info(
          "About to prune cyclones with "
          "\ntrack_id_1  %s\n"
          "with corresponding df\n%s\n"
          "\ntrack_id_2 %s\n"
          "with corresponding df\n%s\n",
          track_id_1,
          lead_time_t_cyclone_1_df.T.to_string(),
          track_id_2,
          lead_time_t_cyclone_2_df.T.to_string(),
      )

      lead_time_t_minus_one_cyclone_1_df = lead_time_t_minus_one_df[
          lead_time_t_minus_one_df[constants.TRACK_ID] == track_id_1
      ]
      lead_time_t_minus_one_cyclone_2_df = lead_time_t_minus_one_df[
          lead_time_t_minus_one_df[constants.TRACK_ID] == track_id_2
      ]
      lead_time_t_minus_one_latlons_1 = lead_time_t_minus_one_cyclone_1_df[
          [constants.LAT, constants.LON]
      ].values
      lead_time_t_minus_one_latlons_2 = lead_time_t_minus_one_cyclone_2_df[
          [constants.LAT, constants.LON]
      ].values

      distance_1 = cyclone_utils.geodesic_distance(
          latlon0=lead_time_t_latlons_1,
          latlon1=lead_time_t_minus_one_latlons_1,
      )
      distance_2 = cyclone_utils.geodesic_distance(
          latlon0=lead_time_t_latlons_2,
          latlon1=lead_time_t_minus_one_latlons_2,
      )
      logging.info(
          "track_id_1 %s moved from latlon %s to latlon %s, distance %s",
          track_id_1,
          lead_time_t_minus_one_latlons_1,
          lead_time_t_latlons_1,
          distance_1,
      )

      logging.info(
          "track_id_2 %s moved from latlon %s to latlon %s, distance %s",
          track_id_2,
          lead_time_t_minus_one_latlons_2,
          lead_time_t_latlons_2,
          distance_2,
      )

      # If distance_1 < distance_2 remove cyclone 2, based on its track id and
      # the most recent lead time.
      if distance_1 < distance_2:
        track_id_to_remove = track_id_2
      else:
        track_id_to_remove = track_id_1
      logging.info("Pruning track_id %s.", track_id_to_remove)

      # Set track_merge on the last entry of the track that is about to be
      # pruned.
      predictions_df.loc[
          (predictions_df[constants.TRACK_ID] == track_id_to_remove)
          & (predictions_df[constants.LEAD_TIME] == previous_lead_time),
          constants.TRACK_MERGE,
      ] = 1.0

      num_rows = len(predictions_df)
      predictions_df = predictions_df[
          # Keep all dataframe rows that are not the most recent lead time...
          (predictions_df[constants.LEAD_TIME] != lead_time_to_prune)
          |
          # ... as well as all rows that are not the track_id_to_remove.
          (predictions_df[constants.TRACK_ID] != track_id_to_remove)
      ]

      assert num_rows - len(predictions_df) == 1, (
          "Expected the number of rows in the dataframe to decrease by exactly "
          "one row after removing one of two nearby cyclones, but found the "
          f"number of rows to be {len(predictions_df)} instead of {num_rows}. "
          "This should not have happened."
      )

      logging.info("Pruned cyclone with track_id %s.", track_id_to_remove)
    else:
      logging.warning(
          "Cyclone pruning loop terminated due to reaching the maximum number"
          " of iterations. This may indicate an issue."
      )
    return predictions_df

  def __call__(
      self,
      gridded_ds: xr.Dataset,
      initial_storms_df: Optional[pd.DataFrame] = None,
      do_cyclogenesis: bool = True,
  ) -> pd.DataFrame:
    """Predicts cyclone trajectories for a given gridded prediction state.

    Any integer columns in the final dataframe are converted to floats to
    ensure compliance with conversion to CycloneTracks format.

    Args:
      gridded_ds: gridded predictions containing cyclone existence
        probabilities.
      initial_storms_df: pandas dataframe containing observed cyclones at
        initialisation, from which tracks will be initialized. If not provided,
        the tracker will run purely in cyclogenesis mode.
      do_cyclogenesis: If True, run cyclogenesis at each lead time.

    Returns:
      pandas dataframe containing cyclone tracks.
    """
    logging.info("Running DirectTracker on gridded_ds: %s", gridded_ds)
    if gridded_ds.sizes["time"] == 0:
      raise ValueError(
          "gridded_ds must have at least one timestep, but found zero:"
          f" {gridded_ds}"
      )

    # If probability_cyclone_exists_variable_name is None, try to infer it from
    # field names, throwing an error if there's ambiguity.
    if not self.probability_cyclone_exists_variable_name:
      self.probability_cyclone_exists_variable_name = (
          self._try_to_infer_probability_cyclone_exists_variable_name(
              gridded_predictions=gridded_ds,
          )
      )

    cols_to_check_for_nans = DEFAULT_VARIABLES_TO_CHECK_FOR_NANS
    cols_to_check_for_nans.append(self.probability_cyclone_exists_variable_name)
    for gridded_var in gridded_ds.data_vars:
      if gridded_var in PREDICTION_FIELD_NAME_TO_TRACKER_COLUMN_NAME:
        tracker_col_var = PREDICTION_FIELD_NAME_TO_TRACKER_COLUMN_NAME[
            gridded_var
        ]
        cols_to_check_for_nans.append(tracker_col_var)

    lead_times = gridded_ds.time.values
    logging.info(
        "Direct tracker received gridded data lead_times: %s", lead_times
    )
    # Check the lead times match the expected temporal resolution. We always
    # assume that the lead times are uniformly spaced, so we check that the
    # deltas between them are all equal, and also equal to the expected temporal
    # resolution.
    lead_timedeltas = np.diff(lead_times)
    unique_lead_timedelta_hours = np.unique(
        lead_timedeltas.astype("timedelta64[s]").astype(int) // 3600
    )
    if unique_lead_timedelta_hours.size == 0:
      logging.warning(
          "No lead time deltas found in gridded_ds, likely because a single "
          "lead time is present in the gridded data, found lead_times=%s.",
          lead_times,
      )
    elif len(unique_lead_timedelta_hours) > 1:
      raise ValueError(
          "Direct tracker does not support gridded data with different lead"
          " times, found lead times: %s" % unique_lead_timedelta_hours
      )
    elif unique_lead_timedelta_hours[0] != self.temporal_resolution_hours:
      raise ValueError(
          "Direct tracker does not support gridded data with lead times that"
          " are not equal to the expected temporal resolution, found lead"
          " times: %s" % unique_lead_timedelta_hours
      )
    # Discard negative lead time frames. These are only needed if the tracker
    # imposes a requirement for the cyclone to exist a certain number of
    # timesteps, which the DirectTracker does not do currently.
    lead_times = lead_times[lead_times.astype("int") >= 0]
    # Ensure that t=0h is included in the lead_times to be looped over, so
    # that t=0h cyclones passed via the initial_storms_df are advanced
    # to the next step.
    if np.timedelta64(0, "ns") not in lead_times:
      lead_times = [np.timedelta64(0, "ns"), *lead_times]
    lead_times = sorted(lead_times)
    timedelta_between_steps = lead_times[1] - lead_times[0]
    assert all(np.diff(lead_times) == timedelta_between_steps)

    # Get initial latlons for each cyclone
    if constants.INIT_TIME in gridded_ds.coords:
      init_time = gridded_ds.init_time.values
    else:
      init_time = gridded_ds.forecast_datetime.values
      gridded_ds = gridded_ds.rename({"forecast_datetime": constants.INIT_TIME})
    if isinstance(init_time, np.ndarray):
      # If init_time is an array, ensure it has only one element and retrieve
      # it.
      assert len(init_time) == 1, (
          "Expected init_time to be an np.ndarray with one element, but found"
          f" {len(init_time)} elements: {init_time}."
      )
      init_time = init_time[0]
    else:
      assert isinstance(init_time, np.datetime64), (
          "Expected init_time to be an np.ndarray or np.datetime64, but found"
          f" {type(init_time)}."
      )
    # Set init_time as new coordinate under gridded_ds.init_time.
    if constants.INIT_TIME in gridded_ds.dims:
      gridded_ds = gridded_ds.squeeze(constants.INIT_TIME)

    init_time = pd.Timestamp(init_time)

    if initial_storms_df is None:
      logging.info("Initialising tracks with no observed storms.")
      initial_storms_df = pd.DataFrame()
    else:
      cols_to_drop_from_initial_storms_df = [
          col
          for col in initial_storms_df.columns
          if col in COLS_TO_DROP_FROM_INITIAL_STORM_DATAFRAME_IF_PRESENT
      ]
      initial_storms_df = initial_storms_df.drop(
          columns=cols_to_drop_from_initial_storms_df
      )

      # Check for negative longitudes.
      if np.any(initial_storms_df[constants.LON].values < 0):
        raise ValueError(
            "initial_storms_df must not contain negative longitudes, found"
            f" {initial_storms_df['lon'].values}"
        )
      # Add arbitrary probability_cyclone_exists values to the
      # initial_storms_df. This avoids having a NaN for this column at the
      # initial timestep.
      initial_storms_df[PROBABILITY_CYCLONE_EXISTS_DATAFRAME_NAME] = (
          PROBABILITY_CYCLONE_EXISTS_FILLVALUE_FOR_T0
      )
      initial_storms_df[constants.TRACK_MERGE] = 0.0
      if constants.LEAD_TIME not in initial_storms_df.columns:
        initial_storms_df[constants.LEAD_TIME] = pd.to_timedelta(
            initial_storms_df[constants.VALID_TIME] - init_time
        )
      logging.info(
          "Initialising tracks with %d observed storms:",
          len(initial_storms_df[constants.TRACK_ID].unique()),
      )
      logging.info(initial_storms_df)

    predictions_df = initial_storms_df

    if do_cyclogenesis:
      self._cyclogenesis_track_counter = 0

    lead_times_to_advance_to_t_plus_one = lead_times[:-1]

    for lead_time_t in lead_times_to_advance_to_t_plus_one:
      lead_time_t_plus_one = lead_time_t + timedelta_between_steps
      lead_time_t_minus_one = lead_time_t - timedelta_between_steps
      logging.info(
          "Advancing from lead time %s hours to %s hours",
          pd.Timedelta(lead_time_t) / pd.Timedelta(hours=1),
          pd.Timedelta(lead_time_t_plus_one) / pd.Timedelta(hours=1),
      )

      predicted_active_track_ids_at_lead_time_t = predictions_df[
          predictions_df[constants.LEAD_TIME] == lead_time_t
      ][constants.TRACK_ID].unique()

      # First, advance all cyclones active at lead time t to lead time t+1.
      # Then, handle edge case where cyclone centers may be close to one
      # another, i.e. two cyclone centers have merged in the tracking stage.
      # Finally, perform cyclogenesis at lead time t+1, avoiding any cyclones
      # that are now active at lead time t+1.
      for track_id, track in predictions_df.groupby(constants.TRACK_ID):
        if track_id not in predicted_active_track_ids_at_lead_time_t:
          continue

        logging.info("Advancing track_id=%s", track_id)
        track_id = str(track_id)

        track = self._advance_single_cyclone_active_at_lead_time_t(
            gridded_predictions=gridded_ds,
            track=track,
            lead_time_t_minus_one=lead_time_t_minus_one,
            lead_time_t=lead_time_t,
            lead_time_t_plus_one=lead_time_t_plus_one,
            tracking_mode=self.tracking_mode,
            track_id=track_id,
        )

        predictions_df = pd.concat([predictions_df, track], ignore_index=True)

      # There are some relatively infrequent edge cases where the tracker is
      # tracking two nearby probability blobs in which the tracker merges the
      # cyclone trajectories, e.g. due to the mode-then-mean scheme honing onto
      # a single of the two blobs. To prevent these merges / tracker jumps, we
      # check if two time stamps are close to one another, and if so, we remove
      # the one which moved the furthest, i.e. jumped the most, cutting off the
      # corresponding track.
      if self.prune_nearby_cyclones:
        predictions_df = (
            self._remove_nearby_cyclone_centers_based_on_which_moved_most(
                predictions_df=predictions_df,
                lead_time_to_prune=lead_time_t_plus_one,
                previous_lead_time=lead_time_t,
            )
        )

      predictions_active_at_lead_time_t_plus_one = predictions_df[
          predictions_df[constants.LEAD_TIME] == lead_time_t_plus_one
      ]
      # Do cyclogenesis for the next timestep (now that we have advanced all
      # cyclones active at lead time t to lead time t+1, and can ignore
      # cyclogenesis candidates that are close to these cyclones).
      if do_cyclogenesis:
        new_cyclogenesis_tracks = self._do_direct_cyclogenesis(
            gridded_predictions=gridded_ds,
            active_cyclones=predictions_active_at_lead_time_t_plus_one,
            lead_time=lead_time_t_plus_one,
        )
        if new_cyclogenesis_tracks is None:
          continue
        predictions_df = pd.concat([predictions_df, new_cyclogenesis_tracks])

    # Remove negative lead times which may have been introduced by the
    # initial_storms_df for the momentum update.
    predictions_df = predictions_df[
        predictions_df[constants.LEAD_TIME].values >= np.timedelta64(0, "ns")
    ]
    predictions_df.loc[:, (constants.VALID_TIME,)] = [
        init_time + lt for lt in predictions_df[constants.LEAD_TIME]
    ]
    predictions_df.loc[:, ("init_time",)] = len(predictions_df) * [init_time]

    if "storm" in predictions_df.columns:
      predictions_df = predictions_df.drop(columns="storm")

    if not predictions_df.empty:
      predictions_df[constants.LEAD_TIME] = pd.to_timedelta(
          predictions_df[constants.LEAD_TIME]
      )
      for column in TRACKER_DATAFRAME_NUMERIC_COLUMNS:
        predictions_df[column] = pd.to_numeric(predictions_df[column])
    else:
      logging.info(
          "Direct tracker produced no tracks. Returning empty dataframe."
      )

    if do_cyclogenesis and self.cyclogenesis_minimum_duration is not None:
      # Filter out track_ids that are active for less than the minimum
      # duration.
      cyclogenesis_track_ids_to_remove = []
      cyclogenesis_track_ids_to_keep = []
      for _, (track_id, track) in enumerate(
          predictions_df.groupby(constants.TRACK_ID)
      ):
        if (
            track_id.startswith("cyclogenesis")
            and track[constants.LEAD_TIME].max()
            - track[constants.LEAD_TIME].min()
            <= self.cyclogenesis_minimum_duration
        ):
          cyclogenesis_track_ids_to_remove.append(track_id)
        elif track_id.startswith("cyclogenesis"):
          cyclogenesis_track_ids_to_keep.append(track_id)
      logging.info(
          "Removing %s cyclogenesis tracks with duration <= %s days: %s",
          len(cyclogenesis_track_ids_to_remove),
          self.cyclogenesis_minimum_duration / pd.Timedelta(days=1),
          cyclogenesis_track_ids_to_remove,
      )
      logging.info(
          "Keeping %s cyclogenesis tracks with duration > %s days: %s",
          len(cyclogenesis_track_ids_to_keep),
          self.cyclogenesis_minimum_duration / pd.Timedelta(days=1),
          cyclogenesis_track_ids_to_keep,
      )
      predictions_to_keep = ~predictions_df[constants.TRACK_ID].isin(
          cyclogenesis_track_ids_to_remove
      )
      predictions_df = predictions_df[predictions_to_keep]
      # Now that we've removed cyclogenesis tracks that are too short, replace
      # their track_ids with integers (to match TempestExtremes)
      predictions_df[constants.TRACK_ID] = predictions_df[
          constants.TRACK_ID
      ].str.removeprefix("cyclogenesis_")

    self._check_for_nans_in_predicted_tracks(
        predictions_df,
        ignore_lead_time_zero=True,
        cols_to_check_for_nans=cols_to_check_for_nans,
    )
    predictions_df = predictions_df.sort_values(
        by=[constants.VALID_TIME, constants.TRACK_ID]
    )
    predictions_df = predictions_df.reset_index(drop=True)

    # Sometimes the model may predict values for the scalar variables close to
    # zero but negative for the scalar variables. All scalar variables that the
    # tracker outputs are non-negative, so we we enforce this here.
    non_negative_variables_in_df = [
        col
        for col in predictions_df.columns
        if col in NON_NEGATIVE_VARIABLES_TO_CLIP
    ]
    predictions_df[non_negative_variables_in_df] = predictions_df[
        non_negative_variables_in_df
    ].clip(lower=0.0)

    if self.enforce_physically_consistent_quadrants_and_winds:
      predictions_df = (
          utils.enforce_physical_consistency_on_quadrants_and_winds(
              predictions_df,
              rmw_and_quad_clipping_mode=self.rmw_and_quad_clipping_mode,
          )
      )

    # Convert any integer columns to floats.
    for col in predictions_df.columns:
      if pd.api.types.is_integer_dtype(predictions_df[col]):
        predictions_df[col] = predictions_df[col].astype(float)

    if self.drop_track_merge_indicator:
      # Remove the track merge indicator column if it exists.
      predictions_df = predictions_df.drop(columns=constants.TRACK_MERGE)

    return predictions_df
