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
"""Defines constant names to refer to variables consistently."""

TRACK_ID = "track_id"
LEAD_TIME = "lead_time"
TIME_SINCE_START = "time_since_start"

INIT_TIME = "init_time"
VALID_TIME = "valid_time"

LAT = "lat"
LON = "lon"
MAX_SUSTAINED_WIND_SPEED_KNOTS = "maximum_sustained_wind_speed_knots"
MIN_SEA_LEVEL_PRESSURE_HPA = "minimum_sea_level_pressure_hpa"

RADIUS_OF_MAXIMUM_WINDS = "radius_of_maximum_winds_km"
RADII_OF_34_KNOT_WINDS = "radii_of_34_knot_winds_km"
RADII_OF_50_KNOT_WINDS = "radii_of_50_knot_winds_km"
RADII_OF_64_KNOT_WINDS = "radii_of_64_knot_winds_km"
WIND_SPEED_TO_RADIUS = {
    34: RADII_OF_34_KNOT_WINDS,
    50: RADII_OF_50_KNOT_WINDS,
    64: RADII_OF_64_KNOT_WINDS,
}

WIND_SPEED_TO_STRIKE_PROBABILITY_NAME = {
    34: "34_knot_strike_probability",
    50: "50_knot_strike_probability",
    64: "64_knot_strike_probability",
}
RADIUS_34_KNOT_WINDS_NE_KM = "radius_34_knot_winds_ne_km"
RADIUS_34_KNOT_WINDS_SE_KM = "radius_34_knot_winds_se_km"
RADIUS_34_KNOT_WINDS_SW_KM = "radius_34_knot_winds_sw_km"
RADIUS_34_KNOT_WINDS_NW_KM = "radius_34_knot_winds_nw_km"
RADIUS_50_KNOT_WINDS_NE_KM = "radius_50_knot_winds_ne_km"
RADIUS_50_KNOT_WINDS_SE_KM = "radius_50_knot_winds_se_km"
RADIUS_50_KNOT_WINDS_SW_KM = "radius_50_knot_winds_sw_km"
RADIUS_50_KNOT_WINDS_NW_KM = "radius_50_knot_winds_nw_km"
RADIUS_64_KNOT_WINDS_NE_KM = "radius_64_knot_winds_ne_km"
RADIUS_64_KNOT_WINDS_SE_KM = "radius_64_knot_winds_se_km"
RADIUS_64_KNOT_WINDS_SW_KM = "radius_64_knot_winds_sw_km"
RADIUS_64_KNOT_WINDS_NW_KM = "radius_64_knot_winds_nw_km"
QUADRANT_RADII = {
    34: {
        "ne": RADIUS_34_KNOT_WINDS_NE_KM,
        "se": RADIUS_34_KNOT_WINDS_SE_KM,
        "sw": RADIUS_34_KNOT_WINDS_SW_KM,
        "nw": RADIUS_34_KNOT_WINDS_NW_KM,
    },
    50: {
        "ne": RADIUS_50_KNOT_WINDS_NE_KM,
        "se": RADIUS_50_KNOT_WINDS_SE_KM,
        "sw": RADIUS_50_KNOT_WINDS_SW_KM,
        "nw": RADIUS_50_KNOT_WINDS_NW_KM,
    },
    64: {
        "ne": RADIUS_64_KNOT_WINDS_NE_KM,
        "se": RADIUS_64_KNOT_WINDS_SE_KM,
        "sw": RADIUS_64_KNOT_WINDS_SW_KM,
        "nw": RADIUS_64_KNOT_WINDS_NW_KM,
    },
}

QUADRANT_RADII_FLATTENED = []
for (
    wind_thresh,
    quad_acronym_to_variable_name,
) in QUADRANT_RADII.items():
  for (
      quadrant_acronym,
      variable_name,
  ) in quad_acronym_to_variable_name.items():
    QUADRANT_RADII_FLATTENED.append(variable_name)

TRACK_MERGE = "track_merge"

QUADRANT_COORDS = ("ne", "se", "sw", "nw")
