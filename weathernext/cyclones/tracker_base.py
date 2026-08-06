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
"""Base class for cyclone tracks and cyclone trackers."""

import abc
from typing import Optional

import pandas as pd
import xarray as xr


class CycloneTracker(abc.ABC):
  """Base class for cyclone trackers.

  Cyclone trackers ingest gridded predictions and return a dataframe of
  cyclone tracks.
  """

  @abc.abstractmethod
  def __init__(self, **tracker_kwargs):
    """Initializes the tracker."""
    raise NotImplementedError()

  @abc.abstractmethod
  def preprocess_gridded_ds(
      self,
      gridded_ds: xr.Dataset,
      extras_ds: Optional[xr.Dataset] = None,
  ) -> xr.Dataset:
    """Preprocesses the gridded dataset before running the tracker."""
    raise NotImplementedError()

  @abc.abstractmethod
  def __call__(
      self,
      gridded_ds: xr.Dataset,
      initial_storms_df: Optional[pd.DataFrame] = None,
      context_time_window_to_remove: Optional[pd.Timedelta] = None,
  ) -> pd.DataFrame:
    """Runs the cyclone tracker on the gridded dataset.

    Args:
      gridded_ds: xarray dataset containing gridded fields.
      initial_storms_df: Optional dataframe containing initial storms to track.
      context_time_window_to_remove: Initial time window to remove from the
        output tracks. The intended use is for removing a context window that is
        prepended to the gridded_ds to account for cyclone duration criteria.

    Returns:
      A pandas DataFrame containing cyclone tracks.
    """
    raise NotImplementedError()
