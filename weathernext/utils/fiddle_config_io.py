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

"""Utilities for loading a fiddle config."""

import pathlib
from typing import Any

import fiddle as fdl
from fiddle import daglish
from fiddle.experimental import serialization as fdl_serialization
import numpy as np
import xarray


# E.g. serialises:
#   <xarray.Dataset> Size: 1kB
#   Dimensions:                                       (level: 13)
#   Coordinates:
#     * level                                         (level) int64 104B 50 ....
#   Data variables: (12/64)
#       surface_solar_radiation_downwards             float64 8B 1.461e+06
#       ...
#   Attributes:
#       num_examples:    8192
#       ...
# Storing dims with coords:
# ...
# [
#           "Index(index=0)",
#           {
#             "type": "leaf",
#             "value": "level",
#             "paths": [
#               "<root>.data['coords']['level']['dims'][0]"
#             ]
#           }
#         ]
# [
#           "Index(index=0)",
#           {
#             "type": "leaf",
#             "value": 50,
#             "paths": [
#               "<root>.data['coords']['level']['data'][0]"
#             ]
#           }
#         ],
# variable values as numbers (floats or ints), per index:
#
# [
#           "Index(index=0)",
#           {
#             "type": "leaf",
#             "value": 307.2701314069515,
#             "paths": [
#               "<root>.data['data_vars']['geopotential']['data'][0]"
#             ]
#           }
#         ]]
# and attributes:
# [
#           "Key(key='num_examples')",
#           {
#             "type": "leaf",
#             "value": 8192,
#             "paths": [
#               "<root>.data['attrs']['num_examples']"
#             ]
#           }
#         ]
def _flatten_dataset(ds):
  return ([ds.to_dict()], None)


def _unflatten_dataset(values, _):
  return xarray.Dataset.from_dict(values[0])


def register_xarray_traverser():
  """Registers the ``xarray.Dataset`` traverser with fiddle serialization."""
  try:
    fdl_serialization.register_node_traverser(
        xarray.Dataset,
        flatten_fn=_flatten_dataset,
        unflatten_fn=_unflatten_dataset,
        path_elements_fn=lambda ds: (daglish.Attr("data"),),
    )
  except ValueError:
    pass  # Already registered.


def _fix_deserialized_dtypes(fdl_config: fdl.Buildable) -> None:
  """Fixes dtype promotions introduced by JSON round-tripping."""
  for wrapper in fdl_config.predictor_wrappers:
    for ds in wrapper["kwargs"].values():
      if not isinstance(ds, xarray.Dataset):
        continue
      # Fix coordinate dtypes (e.g. level: int64 -> int32).
      for coord_name in list(ds.coords):
        coord = ds.coords[coord_name]
        if coord.dtype == np.int64:
          assert coord_name == "level"
          ds.coords[coord_name] = coord.astype(np.int32)
      # Otherwise make sure float dataarrays are float64. This is because we
      # expect json to have promoted them all.
      for var_name in list(ds.data_vars):
        var = ds[var_name]
        if np.issubdtype(var.dtype, np.floating) and var.dtype != np.float64:
          raise ValueError(
              f"Float variable {var_name} has dtype {var.dtype} which is not "
              "float64."
          )


def get_fiddle_config(json_str: str) -> Any:
  """Loads a serialized external FGN config from a JSON file."""
  register_xarray_traverser()
  config = fdl_serialization.load_json(json_str)
  _fix_deserialized_dtypes(config)
  return fdl.build(config)


def get_fiddle_config_path_by_name(name: str) -> str:
  """Returns the path to the config json for a given model name."""
  return str(pathlib.Path(__file__).parent.parent / (name + ".json"))


def get_fiddle_config_by_name(name: str) -> Any:
  """Returns the config json for a given model name."""
  with open(get_fiddle_config_path_by_name(name), "r") as f:
    return get_fiddle_config(f.read())
