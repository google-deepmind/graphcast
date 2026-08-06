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
"""Direct tracker v1 config."""

import dataclasses
from typing import Any

from weathernext.cyclones import direct_tracker
import pandas as pd


@dataclasses.dataclass
class DirectTracker6hV1Config:
  """Configuration for the 6-hourly direct tracker v1."""
  tracker_constructor: Any = direct_tracker.DirectTracker
  tracker_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: {
      "temporal_resolution_hours": 6,
      # Disc radius for estimating scalar variables via interpolation or
      # weighted average.
      "disc_radius_scalar_variables_km": None,
      "disc_radius_mode_latlon_km": 250.0,
      "disc_radius_mean_latlon_km": 250.0,
      "tracking_mode": "mode_then_mean",
      "disc_radius_mean_probability_of_existence_km": 500.0,
      "min_disc_radius_between_cyclogenesis_candidates_km": 500.0,
      # Size of box side to use in cyclogenesis initialisation.
      "cyclogenesis_box_side_degrees": 5,
      # Minimum probability within a cyclogenesis lat-lon box to be considered
      # in the cyclogenesis initialisation.
      "cyclogenesis_min_max_probability_threshold": 0.3,
      # Cyclogenesis threshold equal to the dissipation threshold.
      "cyclogenesis_min_mean_probability_threshold": 1e-5,
      # This radius is used to further refine the cyclogenesis initial guesses
      # by taking the mean position weighted by the existence variable within a
      # disc of this radius around the initial guess.
      "disc_radius_cyclogenesis_refinement_km": 250.0,
      # Cyclogenesis tracks shorter than this duration will be removed.
      # 2.5 days was chosen to match TempestExtremes.
      "cyclogenesis_minimum_duration": pd.Timedelta("2.5d"),
      "dissipation_min_mean_probability_threshold": 1e-5,
      # Momentum constant for updating the cyclone position first guess.
      "momentum_constant": 0.9,
      "prune_nearby_cyclones": True,
      "distance_for_pruning_nearby_cyclones_km": 500.0,
      "enforce_physically_consistent_quadrants_and_winds": True,
  })


def get_config() -> DirectTracker6hV1Config:
  """Returns the config."""
  return DirectTracker6hV1Config()
