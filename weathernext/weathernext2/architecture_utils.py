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

"""Utils for building input features to the model."""

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

HashableOrStr = TypeVar("HashableOrStr", str, Hashable)


def classify_input_data(
    inputs: xr.Dataset,
    forcings: xr.Dataset,
    norm_conditioning_features: Sequence[str],
) -> tuple[
    dict[str, xr.DataArray],
    dict[str, xr.DataArray],
    dict[str, xr.DataArray],
]:
  """Classifies inputs and forcings."""

  all_expected_vars = set(norm_conditioning_features)
  all_present_vars = set(inputs) | set(forcings)
  not_found_vars = all_expected_vars - all_present_vars
  if not_found_vars:
    raise ValueError(f"Features {not_found_vars} not found.")

  inputs_norm_conditioning, inputs = split_norm_conditioning_features(
      inputs, norm_conditioning_features)
  forcings_norm_conditioning, forcings = split_norm_conditioning_features(
      forcings, norm_conditioning_features)

  inputs = _add_prefix(inputs, "input_")  # pyrefly: ignore[bad-assignment]
  forcings = _add_prefix(forcings, "forcing_")  # pyrefly: ignore[bad-assignment]
  inputs_norm_conditioning = _add_prefix(
      inputs_norm_conditioning, "input_")
  forcings_norm_conditioning = _add_prefix(
      forcings_norm_conditioning, "forcing_")

  inputs_and_forcings = {**inputs, **forcings}
  inputs_and_forcings_norm_conditioning = {
      **inputs_norm_conditioning,
      **forcings_norm_conditioning,
  }

  grid_data, global_data = sort_by_grid_and_global(inputs_and_forcings)
  global_data_norm_conditioning = inputs_and_forcings_norm_conditioning

  return (  # pyrefly: ignore[bad-return]
      global_data_norm_conditioning,
      global_data,
      grid_data,
  )


def merge_global_data(
    data: Mapping[str, xr.DataArray],
    global_data: Mapping[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
  """Merges `global_data` with data for each modality in `data_by_modality`."""
  if set(global_data.keys()) & set(data.keys()):
    raise ValueError(
        "Global data and modality data have overlapping keys: "
        f"{set(global_data.keys()) & set(data.keys())}"
    )
  return {**data, **global_data}


def sort_by_grid_and_global(
    data: Mapping[HashableOrStr, xr.DataArray],
) -> tuple[
    dict[HashableOrStr, xr.DataArray],
    dict[HashableOrStr, xr.DataArray],
]:
  """Classifies data by grid and global."""

  global_data = {}
  grid_data = {}
  for name, data_array in data.items():
    if set(data_array.dims).intersection({"lat", "lon"}):
      grid_data[name] = data_array
    else:
      global_data[name] = data_array
  return grid_data, global_data


def _add_prefix(
    dataset: Mapping[Hashable, xr.DataArray], prefix: str
) -> dict[str, xr.DataArray]:
  return {prefix + str(k): v for k, v in dataset.items()}


def split_norm_conditioning_features(
    ds: xr.Dataset,
    norm_conditioning_features: Sequence[str],
) -> tuple[xr.Dataset, xr.Dataset]:
  """Splits norm conditioning features from the data."""

  data = {}
  norm_conditioning_data = {}
  for name, data_array in ds.items():
    if name in norm_conditioning_features:
      norm_conditioning_data[name] = data_array
    else:
      data[name] = data_array
  return xr.Dataset(norm_conditioning_data), xr.Dataset(data)


def get_floating_dtype(data: chex.ArrayTree) -> np.dtype:
  """Returns the floating dtype of the data."""

  arrays = jax.tree_util.tree_leaves(data)
  dtypes = [array.dtype for array in arrays]

  # Filter arrays with all non floating dtypes.
  # TODO(alvarosg): Consider using an approach that does not require to provide
  # an exhaustive list of all dtypes.
  floating_dtypes = (np.float32, np.float64, jnp.bfloat16, np.float16)
  dtypes = [dtype for dtype in dtypes if dtype in floating_dtypes]
  output_dtype = dtypes[0]
  for dtype in dtypes:
    if dtype != output_dtype:
      raise ValueError(
          f"All arrays must have the same dtype, but found {output_dtype} and "
          f"{dtype}."
      )
  return output_dtype

